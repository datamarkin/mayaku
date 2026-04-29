"""Mask R-CNN head + losses + inference helper.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §2.5 (`mask_head.py:215-290`)
and §2.4's `_forward_mask`:

* :class:`MaskRCNNConvUpsampleHead` — ``num_conv`` 3x3 conv-ReLU stages
  (default 4) at the pooler resolution (14), then a
  ``ConvTranspose2d(conv_dim → conv_dim, k=2, s=2)`` to double the
  spatial size, and a final ``1x1`` projection to ``num_classes``
  channels (or 1 if ``cls_agnostic_mask=True``). Output:
  ``(R, K, 28, 28)`` logits.
* :func:`mask_rcnn_loss` — per-class BCE on the gt-class channel
  against ``crop_and_resize(boxes, M)`` of the ground-truth masks.
* :func:`mask_rcnn_inference` — sigmoid + per-class slice; the actual
  paste-back to image coords lives in
  :class:`mayaku.structures.masks.ROIMasks` (Step 4) and is invoked by
  the postprocess stage in Step 16.

The training-time `select_foreground_proposals` filter happens inside
:meth:`StandardROIHeads._forward_mask` (Step 10/11 wiring) — this
module assumes its inputs are already foreground-only.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mayaku.config.schemas import ROIMaskHeadConfig
from mayaku.models.backbones._base import ShapeSpec
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances
from mayaku.structures.masks import BitMasks, PolygonMasks

__all__ = [
    "MaskRCNNConvUpsampleHead",
    "build_mask_head",
    "mask_rcnn_inference",
    "mask_rcnn_loss",
]


class MaskRCNNConvUpsampleHead(nn.Module):
    """Conv stack + 2x ConvTranspose + per-class 1x1 projection.

    Args:
        input_shape: Channels of the pooler output.
        num_classes: Foreground class count ``K``. Output has ``K``
            channels (or ``1`` if ``cls_agnostic_mask=True``).
        num_conv: Number of 3x3 conv-ReLU stages before the upsample
            (default ``4`` per spec §2.5).
        conv_dim: Channels in the conv stack and the deconv (default
            ``256``).
        cls_agnostic_mask: If true, predict a single mask channel
            shared across all classes (matches ``CLS_AGNOSTIC_MASK``).
    """

    def __init__(
        self,
        input_shape: ShapeSpec,
        num_classes: int,
        num_conv: int = 4,
        conv_dim: int = 256,
        cls_agnostic_mask: bool = False,
    ) -> None:
        super().__init__()
        if num_conv < 1:
            raise ValueError(f"num_conv must be >= 1; got {num_conv}")
        self.num_classes = num_classes
        self.cls_agnostic_mask = cls_agnostic_mask

        self.convs = nn.ModuleList()
        ch = input_shape.channels
        for _ in range(num_conv):
            self.convs.append(nn.Conv2d(ch, conv_dim, kernel_size=3, padding=1))
            ch = conv_dim
        self.deconv = nn.ConvTranspose2d(ch, conv_dim, kernel_size=2, stride=2)
        out_ch = 1 if cls_agnostic_mask else num_classes
        self.predictor = nn.Conv2d(conv_dim, out_ch, kernel_size=1)

        # Init: convs MSRA fan-out / ReLU; predictor smaller scale to
        # match the binary-classification expectation. Detectron2 uses
        # `c2_msra_fill` for the convs (`mask_head.py:267`) and a
        # `weight_init.c2_xavier_fill` on the predictor (line 282) which
        # is equivalent to xavier_uniform.
        for m in (*self.convs, self.deconv):
            assert isinstance(m, nn.Conv2d | nn.ConvTranspose2d)
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        nn.init.xavier_uniform_(self.predictor.weight)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = F.relu(conv(x))
        x = F.relu(self.deconv(x))
        out: Tensor = self.predictor(x)
        return out


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def mask_rcnn_loss(
    pred_mask_logits: Tensor,
    fg_proposals: Sequence[Instances],
    *,
    cls_agnostic: bool = False,
) -> Tensor:
    """Per-class BCE-with-logits on the gt-class mask channel.

    Args:
        pred_mask_logits: ``(R, K, M, M)`` logits from
            :class:`MaskRCNNConvUpsampleHead` (or ``(R, 1, M, M)`` if
            ``cls_agnostic=True``). ``R`` is the *foreground* proposal
            count across all images in the batch.
        fg_proposals: Foreground-only :class:`Instances`, one per image.
            Each must carry ``proposal_boxes``, ``gt_classes``, and
            ``gt_masks`` (either :class:`PolygonMasks` or
            :class:`BitMasks`).
        cls_agnostic: If true, use the single channel directly without
            per-class indexing.

    Returns:
        Scalar BCE loss. Returns a real `0.0` (with grad) when there are
        no foreground proposals, so the backward pass stays valid.
    """
    if pred_mask_logits.shape[0] == 0:
        return pred_mask_logits.sum() * 0.0

    mask_side = pred_mask_logits.shape[-1]

    # Build the per-foreground-proposal target by rasterising each
    # image's gt_masks at the head's spatial resolution. This is the
    # crop_and_resize path documented in spec §2.4.
    targets: list[Tensor] = []
    classes: list[Tensor] = []
    for inst in fg_proposals:
        if len(inst) == 0:
            continue
        boxes = inst.proposal_boxes.tensor
        gt_masks = inst.gt_masks
        assert isinstance(gt_masks, PolygonMasks | BitMasks), (
            f"gt_masks must be PolygonMasks or BitMasks; got {type(gt_masks).__name__}"
        )
        targets.append(gt_masks.crop_and_resize(boxes, mask_side).to(boxes.device))
        classes.append(inst.gt_classes)
    if not targets:
        return pred_mask_logits.sum() * 0.0

    target_t = torch.cat(targets, dim=0).to(dtype=pred_mask_logits.dtype)
    if cls_agnostic:
        # Squeeze the K=1 channel dim so shapes line up.
        pred = pred_mask_logits[:, 0]
    else:
        gt_classes = torch.cat(classes, dim=0).long()
        pred = pred_mask_logits[torch.arange(pred_mask_logits.shape[0]), gt_classes]
    return F.binary_cross_entropy_with_logits(pred, target_t, reduction="mean")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def mask_rcnn_inference(
    pred_mask_logits: Tensor, pred_instances: Sequence[Instances], *, cls_agnostic: bool = False
) -> None:
    """Attach per-instance ``pred_masks`` (sigmoid'd, K=1 channel).

    Mirrors `spec §2.5`: select the predicted-class channel, sigmoid,
    add a singleton channel dim so the result is ``(R, 1, M, M)``, and
    split back per image into ``inst.pred_masks``. The actual paste-to-
    image-resolution + threshold step lives in
    :meth:`mayaku.structures.masks.ROIMasks.to_bitmasks` and runs from
    the postprocess stage (Step 16).

    Mutates ``pred_instances`` in place; returns ``None``.
    """
    if cls_agnostic:
        mask_probs = pred_mask_logits.sigmoid()
    else:
        class_pred = torch.cat([i.pred_classes for i in pred_instances], dim=0).long()
        if pred_mask_logits.shape[0] == 0:
            mask_probs = pred_mask_logits.new_zeros(
                (0, 1, pred_mask_logits.shape[-2], pred_mask_logits.shape[-1])
            )
        else:
            sliced = pred_mask_logits[torch.arange(pred_mask_logits.shape[0]), class_pred]
            mask_probs = sliced.sigmoid()[:, None]

    sizes = [len(i) for i in pred_instances]
    for inst, chunk in zip(pred_instances, torch.split(mask_probs, sizes, dim=0), strict=True):
        inst.pred_masks = chunk


# ---------------------------------------------------------------------------
# Helper used by StandardROIHeads
# ---------------------------------------------------------------------------


def select_foreground_proposals(
    proposals: Sequence[Instances], num_classes: int
) -> tuple[list[Instances], list[Tensor]]:
    """Return per-image foreground subsets of ``proposals``.

    Foreground = ``gt_classes`` in ``[0, num_classes)`` (Detectron2
    encodes background as ``num_classes`` and "ignore" as ``-1``).
    Returns ``(fg_proposals, fg_masks)`` where ``fg_masks[i]`` is a
    bool tensor over the original proposals in image ``i``.
    """
    fg_proposals: list[Instances] = []
    fg_masks: list[Tensor] = []
    for inst in proposals:
        gt_classes = inst.gt_classes
        assert isinstance(gt_classes, Tensor)
        mask = (gt_classes >= 0) & (gt_classes < num_classes)
        fg_proposals.append(inst[mask])
        fg_masks.append(mask)
    return fg_proposals, fg_masks


# Re-export so StandardROIHeads can `from .mask_head import …` cleanly.
_ = Boxes  # silence "imported but unused" if Boxes ends up only in type comments


def build_mask_head(
    cfg: ROIMaskHeadConfig, input_shape: ShapeSpec, num_classes: int
) -> MaskRCNNConvUpsampleHead:
    """Construct a :class:`MaskRCNNConvUpsampleHead` from a typed config."""
    return MaskRCNNConvUpsampleHead(
        input_shape=input_shape,
        num_classes=num_classes,
        num_conv=cfg.num_conv,
        conv_dim=cfg.conv_dim,
        cls_agnostic_mask=cfg.cls_agnostic_mask,
    )
