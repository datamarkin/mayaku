"""Region Proposal Network (RPN).

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §2.3
(``modeling/proposal_generator/rpn.py``):

* :class:`StandardRPNHead` — shared 3x3 conv + per-level objectness 1x1
  + per-level box-delta 1x1.
* :class:`RPN` — orchestrates anchor generation, IoU matching,
  fg/bg subsampling, smooth-L1 regression loss, BCE objectness loss,
  and proposal selection (per-image / per-level top-k + per-FPN-level
  NMS via :func:`find_top_rpn_proposals`).

Training-mode call returns ``(proposals, losses)``; inference-mode
returns ``(proposals, {})``. Each proposal entry is an
:class:`Instances` carrying ``proposal_boxes`` (a :class:`Boxes`) and
``objectness_logits`` (a 1-D tensor).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mayaku.backends.ops.nms import batched_nms
from mayaku.config.schemas import AnchorGeneratorConfig, RPNConfig
from mayaku.models.backbones._base import ShapeSpec
from mayaku.models.proposals.anchor_generator import (
    DefaultAnchorGenerator,
    build_anchor_generator,
)
from mayaku.models.proposals.box_regression import Box2BoxTransform
from mayaku.models.proposals.matcher import Matcher
from mayaku.models.proposals.sampling import subsample_labels
from mayaku.structures.boxes import Boxes, pairwise_iou
from mayaku.structures.instances import Instances

__all__ = [
    "RPN",
    "StandardRPNHead",
    "build_rpn",
    "find_top_rpn_proposals",
]

BoxRegLossType = Literal["smooth_l1", "giou"]


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------


class StandardRPNHead(nn.Module):
    """Shared 3x3 conv + per-anchor objectness/regression 1x1 heads.

    Args:
        in_channels: Channels of the FPN feature maps (256 for the
            in-scope configs).
        num_anchors: Number of anchors per spatial location (the
            ``A`` from the anchor generator).
        box_dim: Box parameterisation length (4 for axis-aligned).
    """

    def __init__(self, in_channels: int, num_anchors: int, box_dim: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.objectness_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.anchor_deltas = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1)
        for layer in (self.conv, self.objectness_logits, self.anchor_deltas):
            nn.init.normal_(layer.weight, std=0.01)
            assert layer.bias is not None
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, features: Sequence[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Apply the head to every FPN level.

        Returns:
            ``(pred_objectness_logits, pred_anchor_deltas)`` where each
            list has one ``(N, A, H_i, W_i)`` / ``(N, A*4, H_i, W_i)``
            tensor per level.
        """
        cls: list[Tensor] = []
        reg: list[Tensor] = []
        for x in features:
            t = F.relu(self.conv(x))
            cls.append(self.objectness_logits(t))
            reg.append(self.anchor_deltas(t))
        return cls, reg


# ---------------------------------------------------------------------------
# RPN
# ---------------------------------------------------------------------------


class RPN(nn.Module):
    """Region Proposal Network.

    Composition: anchor generator + RPN head + matcher + box transform.
    The losses module is inlined because the RPN-specific normalisation
    (``loss / (batch_size_per_image * num_images)`` for both cls and
    loc) doesn't reuse anywhere else.
    """

    def __init__(
        self,
        in_features: Sequence[str],
        in_shapes: dict[str, ShapeSpec],
        anchor_generator: DefaultAnchorGenerator,
        head: StandardRPNHead,
        box_transform: Box2BoxTransform,
        anchor_matcher: Matcher,
        *,
        batch_size_per_image: int = 256,
        positive_fraction: float = 0.5,
        pre_nms_topk_train: int = 2000,
        pre_nms_topk_test: int = 1000,
        post_nms_topk_train: int = 2000,
        post_nms_topk_test: int = 1000,
        nms_thresh: float = 0.7,
        min_box_size: float = 0.0,
        smooth_l1_beta: float = 0.0,
        loss_weight: float = 1.0,
        box_reg_loss_type: BoxRegLossType = "smooth_l1",
    ) -> None:
        super().__init__()
        for f in in_features:
            if f not in in_shapes:
                raise ValueError(f"in_feature {f!r} not in in_shapes ({tuple(in_shapes)})")
        self.in_features = tuple(in_features)
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_transform = box_transform
        self.anchor_matcher = anchor_matcher

        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pre_nms_topk_train = pre_nms_topk_train
        self.pre_nms_topk_test = pre_nms_topk_test
        self.post_nms_topk_train = post_nms_topk_train
        self.post_nms_topk_test = post_nms_topk_test
        self.nms_thresh = nms_thresh
        self.min_box_size = min_box_size
        self.smooth_l1_beta = smooth_l1_beta
        self.loss_weight = loss_weight
        self.box_reg_loss_type: BoxRegLossType = box_reg_loss_type
        if box_reg_loss_type != "smooth_l1":
            raise NotImplementedError(
                f"box_reg_loss_type={box_reg_loss_type!r} not supported yet — "
                "wire GIoU into _box_reg_loss when needed."
            )

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        image_sizes: Sequence[tuple[int, int]],
        features: dict[str, Tensor],
        gt_instances: Sequence[Instances] | None = None,
    ) -> tuple[list[Instances], dict[str, Tensor]]:
        feats = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(feats)  # List[Boxes] per level
        pred_cls_logits, pred_anchor_deltas = self.head(feats)

        # Reshape per-level (N, A, Hi, Wi) → (N, Hi*Wi*A) for cls,
        # and (N, A*4, Hi, Wi) → (N, Hi*Wi*A, 4) for deltas.
        pred_cls_logits = [_flatten_cls(t) for t in pred_cls_logits]
        pred_anchor_deltas = [_flatten_reg(t) for t in pred_anchor_deltas]

        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances during training"
            gt_labels, gt_boxes = self._label_and_sample_anchors(anchors, gt_instances)
            losses = self._losses(anchors, pred_cls_logits, pred_anchor_deltas, gt_labels, gt_boxes)
        else:
            losses = {}
        # Proposals are treated as fixed inputs to the ROI heads. Without
        # this no_grad guard, the ROI box-regression loss backprops through
        # box_transform.get_deltas(proposal_boxes, gt_boxes) in
        # FastRCNNOutputLayers.losses and pollutes the RPN regression
        # gradient with arbitrary signal from the ROI head. Mirrors
        # detectron2/modeling/proposal_generator/rpn.py:predict_proposals.
        with torch.no_grad():
            proposals = find_top_rpn_proposals(
                pred_cls_logits,
                pred_anchor_deltas,
                anchors,
                image_sizes,
                box_transform=self.box_transform,
                pre_nms_topk=self.pre_nms_topk_train if self.training else self.pre_nms_topk_test,
                post_nms_topk=self.post_nms_topk_train
                if self.training
                else self.post_nms_topk_test,
                nms_thresh=self.nms_thresh,
                min_box_size=self.min_box_size,
                training=self.training,
            )
        return proposals, losses

    # ------------------------------------------------------------------
    # Training: label, sample, loss
    # ------------------------------------------------------------------

    def _label_and_sample_anchors(
        self,
        anchors: list[Boxes],
        gt_instances: Sequence[Instances],
    ) -> tuple[Tensor, Tensor]:
        """`spec §2.3`: label every anchor as fg/bg/ignore via IoU
        matching, then subsample to ``batch_size_per_image``.

        Returns:
            ``gt_labels`` ``(N, R)`` int8 in ``{-1, 0, 1}``;
            ``gt_anchor_boxes`` ``(N, R, 4)`` matched GT boxes (only
            meaningful where ``gt_labels == 1``).
        """
        anchors_cat = Boxes.cat(anchors).tensor  # (R, 4)
        all_labels: list[Tensor] = []
        all_boxes: list[Tensor] = []
        for inst in gt_instances:
            gt_boxes = _gt_boxes_tensor(inst)
            if gt_boxes.numel() == 0:
                # No GT in this image → everything is background.
                labels = torch.zeros(
                    anchors_cat.shape[0], dtype=torch.int8, device=anchors_cat.device
                )
                matched = torch.zeros_like(anchors_cat)
            else:
                iou = pairwise_iou(Boxes(gt_boxes.to(anchors_cat.device)), Boxes(anchors_cat))
                matched_idx, labels = self.anchor_matcher(iou)
                matched = gt_boxes[matched_idx.cpu()].to(anchors_cat.device)
                # Subsample fg/bg to the batch size, mark the rest as -1.
                pos_idx, neg_idx = subsample_labels(
                    labels.long(),
                    num_samples=self.batch_size_per_image,
                    positive_fraction=self.positive_fraction,
                    bg_label=0,
                )
                kept = torch.full_like(labels, -1)
                kept[pos_idx] = 1
                kept[neg_idx] = 0
                labels = kept
            all_labels.append(labels)
            all_boxes.append(matched)
        return torch.stack(all_labels), torch.stack(all_boxes)

    def _losses(
        self,
        anchors: list[Boxes],
        pred_cls_logits: list[Tensor],
        pred_anchor_deltas: list[Tensor],
        gt_labels: Tensor,
        gt_boxes: Tensor,
    ) -> dict[str, Tensor]:
        anchors_cat = Boxes.cat(anchors).tensor  # (R, 4)
        # Concatenate per-level predictions along the anchor axis.
        pred_cls = torch.cat(pred_cls_logits, dim=1)  # (N, R)
        pred_reg = torch.cat(pred_anchor_deltas, dim=1)  # (N, R, 4)

        # Encode ground-truth deltas per image for the foreground anchors.
        # Doing this per image keeps memory bounded (R can be ~200k for
        # 1333x800 inputs).
        n = gt_labels.shape[0]
        anchors_per_img = anchors_cat.unsqueeze(0).expand(n, -1, -1)  # (N, R, 4)
        gt_deltas = torch.zeros_like(pred_reg)
        fg_mask_global = gt_labels == 1
        if fg_mask_global.any():
            fg_anchor = anchors_per_img[fg_mask_global]
            fg_target = gt_boxes[fg_mask_global]
            gt_deltas[fg_mask_global] = self.box_transform.get_deltas(fg_anchor, fg_target).to(
                gt_deltas.dtype
            )

        valid_mask = gt_labels >= 0
        normaliser = max(self.batch_size_per_image * n, 1)

        loss_cls = (
            F.binary_cross_entropy_with_logits(
                pred_cls[valid_mask], gt_labels[valid_mask].float(), reduction="sum"
            )
            / normaliser
        )

        loss_loc = (
            F.smooth_l1_loss(
                pred_reg[fg_mask_global],
                gt_deltas[fg_mask_global],
                beta=self.smooth_l1_beta,
                reduction="sum",
            )
            / normaliser
            if fg_mask_global.any()
            else pred_reg.sum() * 0.0
        )

        return {
            "loss_rpn_cls": loss_cls * self.loss_weight,
            "loss_rpn_loc": loss_loc * self.loss_weight,
        }


# ---------------------------------------------------------------------------
# Proposal selection
# ---------------------------------------------------------------------------


def find_top_rpn_proposals(
    pred_cls_logits: list[Tensor],
    pred_anchor_deltas: list[Tensor],
    anchors: list[Boxes],
    image_sizes: Sequence[tuple[int, int]],
    *,
    box_transform: Box2BoxTransform,
    pre_nms_topk: int,
    post_nms_topk: int,
    nms_thresh: float,
    min_box_size: float = 0.0,
    training: bool = False,
) -> list[Instances]:
    """Per-FPN-level top-k → cross-level NMS → post-NMS top-k.

    Implements `spec §2.3` (``proposal_utils.py:22-135``). NMS uses the
    FPN level id as the per-class offset so duplicates are removed
    *within* a level but proposals overlapping across levels survive.
    """
    n = len(image_sizes)
    device = pred_cls_logits[0].device

    # Per level, decode anchors to image-pixel boxes and select top-k by score.
    level_proposals: list[list[Tensor]] = []  # per image, per level: (k, 4) decoded boxes
    level_scores: list[list[Tensor]] = []
    level_ids: list[list[Tensor]] = []

    for _ in range(n):
        level_proposals.append([])
        level_scores.append([])
        level_ids.append([])

    for level, (cls_l, reg_l, anchors_l) in enumerate(
        zip(pred_cls_logits, pred_anchor_deltas, anchors, strict=True)
    ):
        # cls_l: (N, R_l), reg_l: (N, R_l, 4)
        anchors_t = anchors_l.tensor
        for img_idx in range(n):
            scores_img = cls_l[img_idx]  # (R_l,)
            deltas_img = reg_l[img_idx]  # (R_l, 4)
            num_proposals = min(pre_nms_topk, scores_img.shape[0])
            top_scores, top_idx = scores_img.topk(num_proposals, sorted=True)
            top_anchors = anchors_t[top_idx]
            top_deltas = deltas_img[top_idx]
            decoded = box_transform.apply_deltas(top_deltas, top_anchors)  # (k, 4)
            level_proposals[img_idx].append(decoded)
            level_scores[img_idx].append(top_scores)
            level_ids[img_idx].append(
                torch.full((decoded.shape[0],), level, dtype=torch.int64, device=device)
            )

    # Per-image: concat all levels, clip to image, drop tiny boxes,
    # batched NMS, post-NMS top-k.
    out: list[Instances] = []
    for img_idx, (h, w) in enumerate(image_sizes):
        boxes = torch.cat(level_proposals[img_idx], dim=0)
        scores = torch.cat(level_scores[img_idx], dim=0)
        ids = torch.cat(level_ids[img_idx], dim=0)

        finite = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
        if training and not bool(finite.all()):
            # Mirror detectron2.modeling.proposal_generator.proposal_utils.
            # find_top_rpn_proposals: surface divergence loudly during
            # training rather than silently filtering, which can mask
            # gradient corruption for many iters before NaN finally
            # propagates everywhere.
            raise FloatingPointError(
                "Predicted boxes or scores contain Inf/NaN. Training has diverged."
            )
        boxes = boxes[finite]
        scores = scores[finite]
        ids = ids[finite]

        # Clip to image and drop tiny boxes.
        b = Boxes(boxes.clone())
        b.clip((h, w))
        keep_mask = b.nonempty(threshold=min_box_size)
        boxes = b.tensor[keep_mask]
        scores = scores[keep_mask]
        ids = ids[keep_mask]

        keep = batched_nms(boxes, scores, ids.to(boxes.dtype), iou_threshold=nms_thresh)
        keep = keep[:post_nms_topk]
        inst = Instances(image_size=(h, w))
        inst.proposal_boxes = Boxes(boxes[keep])
        inst.objectness_logits = scores[keep]
        out.append(inst)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_cls(t: Tensor) -> Tensor:
    """``(N, A, H, W)`` → ``(N, H*W*A)``."""
    n, a, h, w = t.shape
    return t.permute(0, 2, 3, 1).reshape(n, h * w * a)


def _flatten_reg(t: Tensor) -> Tensor:
    """``(N, A*4, H, W)`` → ``(N, H*W*A, 4)``."""
    n, ax4, h, w = t.shape
    a = ax4 // 4
    return t.view(n, a, 4, h, w).permute(0, 3, 4, 1, 2).reshape(n, h * w * a, 4)


def _gt_boxes_tensor(inst: Instances) -> Tensor:
    if not inst.has("gt_boxes"):
        return torch.zeros(0, 4)
    boxes = inst.gt_boxes
    if isinstance(boxes, Boxes):
        return boxes.tensor
    assert isinstance(boxes, Tensor)
    return boxes


# ---------------------------------------------------------------------------
# Build factory
# ---------------------------------------------------------------------------


def build_rpn(
    rpn_cfg: RPNConfig,
    anchor_cfg: AnchorGeneratorConfig,
    in_shapes: dict[str, ShapeSpec],
) -> RPN:
    """Construct a fully-wired :class:`RPN` from typed configs.

    ``in_shapes`` is the FPN's ``output_shape()`` (or any backbone
    that produces named feature maps for ``rpn_cfg.in_features``).
    """
    in_features = rpn_cfg.in_features
    strides = [in_shapes[f].stride for f in in_features]
    channels = [in_shapes[f].channels for f in in_features]
    if len(set(channels)) != 1:
        raise ValueError(
            "All RPN input features must share channel count; "
            f"got {dict(zip(in_features, channels, strict=False))}"
        )
    in_channels = channels[0]
    anchor_generator = build_anchor_generator(anchor_cfg, strides=strides)
    nums = anchor_generator.num_anchors_per_cell
    if len(set(nums)) != 1:
        raise ValueError(
            f"StandardRPNHead requires the same A across levels; got per-level counts {nums}"
        )
    head = StandardRPNHead(in_channels=in_channels, num_anchors=nums[0])
    box_transform = Box2BoxTransform(weights=rpn_cfg.bbox_reg_weights)
    matcher = Matcher(
        thresholds=list(rpn_cfg.iou_thresholds),
        labels=list(rpn_cfg.iou_labels),
        allow_low_quality_matches=True,
    )
    return RPN(
        in_features=in_features,
        in_shapes=in_shapes,
        anchor_generator=anchor_generator,
        head=head,
        box_transform=box_transform,
        anchor_matcher=matcher,
        batch_size_per_image=rpn_cfg.batch_size_per_image,
        positive_fraction=rpn_cfg.positive_fraction,
        pre_nms_topk_train=rpn_cfg.pre_nms_topk_train,
        pre_nms_topk_test=rpn_cfg.pre_nms_topk_test,
        post_nms_topk_train=rpn_cfg.post_nms_topk_train,
        post_nms_topk_test=rpn_cfg.post_nms_topk_test,
        nms_thresh=rpn_cfg.nms_thresh,
        min_box_size=rpn_cfg.min_box_size,
        smooth_l1_beta=rpn_cfg.smooth_l1_beta,
        loss_weight=rpn_cfg.loss_weight,
        box_reg_loss_type=rpn_cfg.box_reg_loss_type,
    )
