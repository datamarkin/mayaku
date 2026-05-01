"""StandardROIHeads — the box-head dispatcher for Faster R-CNN.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §2.4 (`roi_heads.py:529-877`)
focused on the box path. Mask + keypoint heads plug in as additional
`_forward_*` methods in Steps 11 / 12.

The dispatcher's contract:

* :meth:`forward` takes the FPN feature dict, the RPN proposals, and
  (training only) the ground-truth ``Instances``. It returns
  ``(predictions, losses)`` where ``predictions`` is the proposal list
  augmented with ``gt_*`` fields during training, and the post-NMS
  detections during inference.
* :meth:`label_and_sample_proposals` does the matcher / sampler dance
  documented in spec §2.4: optional GT-append, IoU matching at 0.5,
  subsample to ``BATCH_SIZE_PER_IMAGE = 512`` at ``POSITIVE_FRACTION
  = 0.25``. Background is encoded as ``num_classes``; ignore as ``-1``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from mayaku.config.schemas import (
    ROIBoxHeadConfig,
    ROIHeadsConfig,
    ROIKeypointHeadConfig,
    ROIMaskHeadConfig,
)
from mayaku.models.backbones._base import ShapeSpec
from mayaku.models.heads.box_head import build_box_head
from mayaku.models.heads.fast_rcnn import FastRCNNOutputLayers
from mayaku.models.heads.keypoint_head import (
    build_keypoint_head,
    keypoint_rcnn_inference,
    keypoint_rcnn_loss,
    select_proposals_with_visible_keypoints,
)
from mayaku.models.heads.mask_head import (
    build_mask_head,
    mask_rcnn_inference,
    mask_rcnn_loss,
    select_foreground_proposals,
)
from mayaku.models.poolers import ROIPooler
from mayaku.models.proposals.box_regression import Box2BoxTransform
from mayaku.models.proposals.matcher import Matcher
from mayaku.models.proposals.sampling import subsample_labels
from mayaku.structures.boxes import Boxes, pairwise_iou
from mayaku.structures.instances import Instances
from mayaku.structures.keypoints import Keypoints
from mayaku.structures.masks import BitMasks, PolygonMasks

__all__ = ["StandardROIHeads", "build_standard_roi_heads"]


class StandardROIHeads(nn.Module):
    """Faster R-CNN box-path orchestrator."""

    def __init__(
        self,
        in_features: Sequence[str],
        in_shapes: dict[str, ShapeSpec],
        num_classes: int,
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: FastRCNNOutputLayers,
        proposal_matcher: Matcher,
        *,
        mask_pooler: ROIPooler | None = None,
        mask_head: nn.Module | None = None,
        mask_loss_weight: float = 1.0,
        cls_agnostic_mask: bool = False,
        keypoint_pooler: ROIPooler | None = None,
        keypoint_head: nn.Module | None = None,
        keypoint_loss_weight: float = 1.0,
        keypoint_loss_normalizer: str = "visible",
        keypoint_loss_static_constant: float = 0.0,
        batch_size_per_image: int = 512,
        positive_fraction: float = 0.25,
        proposal_append_gt: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = tuple(in_features)
        self.in_shapes = in_shapes
        self.num_classes = num_classes
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.proposal_matcher = proposal_matcher
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.proposal_append_gt = proposal_append_gt
        self.mask_pooler = mask_pooler
        self.mask_head = mask_head
        self.mask_loss_weight = mask_loss_weight
        self.cls_agnostic_mask = cls_agnostic_mask
        if (mask_pooler is None) != (mask_head is None):
            raise ValueError("mask_pooler and mask_head must both be set or both be None")
        self.keypoint_pooler = keypoint_pooler
        self.keypoint_head = keypoint_head
        self.keypoint_loss_weight = keypoint_loss_weight
        self.keypoint_loss_normalizer = keypoint_loss_normalizer
        self.keypoint_loss_static_constant = keypoint_loss_static_constant
        if (keypoint_pooler is None) != (keypoint_head is None):
            raise ValueError("keypoint_pooler and keypoint_head must both be set or both be None")

    @property
    def mask_on(self) -> bool:
        return self.mask_head is not None

    @property
    def keypoint_on(self) -> bool:
        return self.keypoint_head is not None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        features: dict[str, Tensor],
        proposals: list[Instances],
        targets: Sequence[Instances] | None = None,
    ) -> tuple[list[Instances], dict[str, Tensor]]:
        if self.training:
            assert targets is not None, "ROIHeads requires targets during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        feats = [features[f] for f in self.in_features]
        box_feats = self.box_pooler(feats, [p.proposal_boxes for p in proposals])
        box_feats = self.box_head(box_feats)
        predictions = self.box_predictor(box_feats)
        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                losses |= self._forward_mask_train(feats, proposals)
            if self.keypoint_on:
                losses |= self._forward_keypoint_train(feats, proposals)
            return proposals, losses
        results = self.box_predictor.inference(predictions, proposals)
        if self.mask_on:
            self._forward_mask_infer(feats, results)
        if self.keypoint_on:
            self._forward_keypoint_infer(feats, results)
        return results, {}

    # ------------------------------------------------------------------
    # Mask path
    # ------------------------------------------------------------------

    def _forward_mask_train(
        self, feats: list[Tensor], proposals: Sequence[Instances]
    ) -> dict[str, Tensor]:
        assert self.mask_pooler is not None and self.mask_head is not None
        fg_proposals, _ = select_foreground_proposals(proposals, self.num_classes)
        # Drop instances without gt_masks (e.g. an empty-GT image short-circuit).
        fg_proposals = [p for p in fg_proposals if len(p) > 0 and p.has("gt_masks")]
        if not fg_proposals:
            # Real zero-grad sentinel so the backward pass stays valid
            # (mirrors mask_rcnn_loss's empty-input handling).
            return {"loss_mask": next(self.mask_head.parameters()).sum() * 0.0}
        mask_feats = self.mask_pooler(feats, [p.proposal_boxes for p in fg_proposals])
        logits = self.mask_head(mask_feats)
        loss = mask_rcnn_loss(logits, fg_proposals, cls_agnostic=self.cls_agnostic_mask)
        return {"loss_mask": loss * self.mask_loss_weight}

    def _forward_mask_infer(self, feats: list[Tensor], results: list[Instances]) -> None:
        assert self.mask_pooler is not None and self.mask_head is not None
        if all(len(r) == 0 for r in results):
            for r in results:
                # Attach an empty (0, 1, M, M) tensor so postprocess can
                # uniformly assume the field exists.
                r.pred_masks = next(self.mask_head.parameters()).new_zeros((0, 1, 1, 1))
            return
        mask_feats = self.mask_pooler(feats, [r.pred_boxes for r in results])
        logits = self.mask_head(mask_feats)
        mask_rcnn_inference(logits, results, cls_agnostic=self.cls_agnostic_mask)

    # ------------------------------------------------------------------
    # Keypoint path
    # ------------------------------------------------------------------

    def _forward_keypoint_train(
        self, feats: list[Tensor], proposals: Sequence[Instances]
    ) -> dict[str, Tensor]:
        assert self.keypoint_pooler is not None and self.keypoint_head is not None
        fg_proposals, _ = select_foreground_proposals(proposals, self.num_classes)
        # Drop any fg proposal whose gt_keypoints have zero visible joints.
        kp_proposals, _ = select_proposals_with_visible_keypoints(fg_proposals)
        kp_proposals = [p for p in kp_proposals if len(p) > 0 and p.has("gt_keypoints")]
        if not kp_proposals:
            return {"loss_keypoint": next(self.keypoint_head.parameters()).sum() * 0.0}
        kp_feats = self.keypoint_pooler(feats, [p.proposal_boxes for p in kp_proposals])
        logits = self.keypoint_head(kp_feats)
        loss = keypoint_rcnn_loss(
            logits,
            kp_proposals,
            normalizer="visible" if self.keypoint_loss_normalizer == "visible" else "static",
            static_normalizer_constant=self.keypoint_loss_static_constant,
            loss_weight=self.keypoint_loss_weight,
        )
        return {"loss_keypoint": loss}

    def _forward_keypoint_infer(self, feats: list[Tensor], results: list[Instances]) -> None:
        assert self.keypoint_pooler is not None and self.keypoint_head is not None
        if all(len(r) == 0 for r in results):
            param = next(self.keypoint_head.parameters())
            for r in results:
                r.pred_keypoints = param.new_zeros((0, 1, 3))
                r.pred_keypoint_heatmaps = param.new_zeros((0, 1, 1, 1))
            return
        kp_feats = self.keypoint_pooler(feats, [r.pred_boxes for r in results])
        logits = self.keypoint_head(kp_feats)
        keypoint_rcnn_inference(logits, results)

    # ------------------------------------------------------------------
    # Proposal labelling
    # ------------------------------------------------------------------

    def label_and_sample_proposals(
        self, proposals: Sequence[Instances], targets: Sequence[Instances]
    ) -> list[Instances]:
        out: list[Instances] = []
        for proposals_per_image, targets_per_image in zip(proposals, targets, strict=True):
            if self.proposal_append_gt:
                proposals_per_image = _append_gt_to_proposals(
                    proposals_per_image, targets_per_image
                )
            gt_boxes = _gt_boxes_tensor(targets_per_image)
            proposal_boxes = proposals_per_image.proposal_boxes.tensor
            n_proposals = proposal_boxes.shape[0]

            if gt_boxes.numel() == 0:
                # No GT in this image — every proposal is background.
                # Sample bg up to batch_size_per_image so the classifier
                # still trains on these as negatives. Box-reg sees no
                # foreground and contributes zero (handled by
                # FastRCNNOutputLayers.losses' fg_mask check).
                gt_classes = torch.full(
                    (n_proposals,),
                    self.num_classes,
                    dtype=torch.int64,
                    device=proposal_boxes.device,
                )
                pos_idx, neg_idx = subsample_labels(
                    gt_classes,
                    num_samples=self.batch_size_per_image,
                    positive_fraction=self.positive_fraction,
                    bg_label=self.num_classes,
                )
                sampled_idx = torch.cat([pos_idx, neg_idx], dim=0)
                proposals_sampled = proposals_per_image[sampled_idx]
                proposals_sampled.gt_classes = gt_classes[sampled_idx]
                proposals_sampled.gt_boxes = torch.zeros(
                    (sampled_idx.numel(), 4), device=proposal_boxes.device
                )
                out.append(proposals_sampled)
                continue

            iou = pairwise_iou(Boxes(gt_boxes.to(proposal_boxes.device)), Boxes(proposal_boxes))
            matched_idx, matched_labels = self.proposal_matcher(iou)
            gt_classes_per_image = _gt_classes_tensor(targets_per_image).to(proposal_boxes.device)
            gt_classes = gt_classes_per_image[matched_idx]
            gt_classes[matched_labels == 0] = self.num_classes  # background
            gt_classes[matched_labels == -1] = -1  # ignore

            pos_idx, neg_idx = subsample_labels(
                gt_classes,
                num_samples=self.batch_size_per_image,
                positive_fraction=self.positive_fraction,
                bg_label=self.num_classes,
            )
            sampled_idx = torch.cat([pos_idx, neg_idx], dim=0)

            proposals_sampled = proposals_per_image[sampled_idx]
            proposals_sampled.gt_classes = gt_classes[sampled_idx]
            sampled_matches = matched_idx[sampled_idx].cpu()
            matched_gt_boxes = gt_boxes[sampled_matches].to(proposal_boxes.device)
            proposals_sampled.gt_boxes = matched_gt_boxes
            # Propagate per-task GT (masks today, keypoints in Step 12) so
            # the per-task heads can read aligned annotations off the
            # sampled proposal list.
            if targets_per_image.has("gt_masks"):
                gt_masks = targets_per_image.gt_masks
                assert isinstance(gt_masks, PolygonMasks | BitMasks)
                proposals_sampled.gt_masks = gt_masks[sampled_matches]
            if targets_per_image.has("gt_keypoints"):
                gt_keypoints = targets_per_image.gt_keypoints
                assert isinstance(gt_keypoints, Keypoints)
                proposals_sampled.gt_keypoints = gt_keypoints[
                    sampled_matches.to(gt_keypoints.device)
                ]
            out.append(proposals_sampled)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _append_gt_to_proposals(proposals: Instances, targets: Instances) -> Instances:
    """Concatenate GT boxes with a synthetic high objectness logit
    onto the proposal list (`spec §2.4` step 1)."""
    gt_boxes = _gt_boxes_tensor(targets)
    if gt_boxes.numel() == 0:
        return proposals
    device = proposals.proposal_boxes.tensor.device
    gt_boxes_dev = gt_boxes.to(device)
    new_proposals = Instances(image_size=proposals.image_size)
    new_proposals.proposal_boxes = Boxes(
        torch.cat([proposals.proposal_boxes.tensor, gt_boxes_dev], dim=0)
    )
    if proposals.has("objectness_logits"):
        existing = proposals.objectness_logits
        # Synthetic logit ≈ +∞-ish; Detectron2 uses log(1) = 0 then +1e6
        # in practice (ensures GT survives any score-threshold filtering
        # while still being a real number under fp32).
        synth = existing.new_full((gt_boxes_dev.shape[0],), 1e6)
        new_proposals.objectness_logits = torch.cat([existing, synth], dim=0)
    return new_proposals


def _gt_boxes_tensor(inst: Instances) -> Tensor:
    if not inst.has("gt_boxes"):
        return torch.zeros(0, 4)
    boxes = inst.gt_boxes
    if isinstance(boxes, Boxes):
        return boxes.tensor
    assert isinstance(boxes, Tensor)
    return boxes


def _gt_classes_tensor(inst: Instances) -> Tensor:
    if not inst.has("gt_classes"):
        return torch.zeros(0, dtype=torch.int64)
    classes = inst.gt_classes
    assert isinstance(classes, Tensor)
    return classes


# ---------------------------------------------------------------------------
# Build factory
# ---------------------------------------------------------------------------


def build_standard_roi_heads(
    cfg: ROIHeadsConfig,
    box_head_cfg: ROIBoxHeadConfig,
    in_shapes: dict[str, ShapeSpec],
    *,
    mask_head_cfg: ROIMaskHeadConfig | None = None,
    keypoint_head_cfg: ROIKeypointHeadConfig | None = None,
    test_detections_per_image: int = 100,
) -> StandardROIHeads:
    """Build a fully-wired :class:`StandardROIHeads` from typed configs.

    ``mask_head_cfg`` and ``keypoint_head_cfg`` are independently
    optional: pass either, both, or neither. The same ROI-head
    dispatcher serves Faster / Mask / Keypoint R-CNN.
    """
    in_features = cfg.in_features
    strides = [in_shapes[f].stride for f in in_features]
    channels = {in_shapes[f].channels for f in in_features}
    if len(channels) != 1:
        raise ValueError(
            "All ROI Heads input features must share channel count; "
            f"got {dict({f: in_shapes[f].channels for f in in_features})}"
        )
    in_channels = next(iter(channels))

    box_pooler = ROIPooler(
        output_size=box_head_cfg.pooler_resolution,
        scales=tuple(1.0 / s for s in strides),
        sampling_ratio=box_head_cfg.pooler_sampling_ratio,
    )
    box_head = build_box_head(box_head_cfg, ShapeSpec(channels=in_channels, stride=1))
    output_dim = box_head.output_dim if hasattr(box_head, "output_dim") else in_channels
    box_predictor = FastRCNNOutputLayers(
        input_dim=output_dim,
        num_classes=cfg.num_classes,
        box_transform=Box2BoxTransform(weights=box_head_cfg.bbox_reg_weights),
        smooth_l1_beta=box_head_cfg.smooth_l1_beta,
        cls_agnostic_bbox_reg=box_head_cfg.cls_agnostic_bbox_reg,
        score_thresh_test=cfg.score_thresh_test,
        nms_thresh_test=cfg.nms_thresh_test,
        topk_per_image=test_detections_per_image,
    )
    matcher = Matcher(
        thresholds=list(cfg.iou_thresholds),
        labels=list(cfg.iou_labels),
        allow_low_quality_matches=False,
    )

    mask_pooler: ROIPooler | None = None
    mask_head: nn.Module | None = None
    cls_agnostic_mask = False
    mask_loss_weight = 1.0
    if mask_head_cfg is not None:
        mask_pooler = ROIPooler(
            output_size=mask_head_cfg.pooler_resolution,
            scales=tuple(1.0 / s for s in strides),
            sampling_ratio=mask_head_cfg.pooler_sampling_ratio,
        )
        mask_head = build_mask_head(
            mask_head_cfg,
            ShapeSpec(channels=in_channels, stride=1),
            cfg.num_classes,
        )
        cls_agnostic_mask = mask_head_cfg.cls_agnostic_mask
        mask_loss_weight = mask_head_cfg.loss_weight

    keypoint_pooler: ROIPooler | None = None
    keypoint_head: nn.Module | None = None
    keypoint_loss_weight = 1.0
    keypoint_loss_normalizer = "visible"
    keypoint_loss_static_constant = 0.0
    if keypoint_head_cfg is not None:
        keypoint_pooler = ROIPooler(
            output_size=keypoint_head_cfg.pooler_resolution,
            scales=tuple(1.0 / s for s in strides),
            sampling_ratio=keypoint_head_cfg.pooler_sampling_ratio,
        )
        keypoint_head = build_keypoint_head(
            keypoint_head_cfg, ShapeSpec(channels=in_channels, stride=1)
        )
        keypoint_loss_weight = keypoint_head_cfg.loss_weight
        keypoint_loss_normalizer = (
            "visible" if keypoint_head_cfg.normalize_loss_by_visible_keypoints else "static"
        )
        # Upstream's static-normaliser default: K * batch * pos_fraction.
        keypoint_loss_static_constant = (
            keypoint_head_cfg.num_keypoints * cfg.batch_size_per_image * cfg.positive_fraction
        )

    return StandardROIHeads(
        in_features=in_features,
        in_shapes=in_shapes,
        num_classes=cfg.num_classes,
        box_pooler=box_pooler,
        box_head=box_head,
        box_predictor=box_predictor,
        proposal_matcher=matcher,
        mask_pooler=mask_pooler,
        mask_head=mask_head,
        mask_loss_weight=mask_loss_weight,
        cls_agnostic_mask=cls_agnostic_mask,
        keypoint_pooler=keypoint_pooler,
        keypoint_head=keypoint_head,
        keypoint_loss_weight=keypoint_loss_weight,
        keypoint_loss_normalizer=keypoint_loss_normalizer,
        keypoint_loss_static_constant=keypoint_loss_static_constant,
        batch_size_per_image=cfg.batch_size_per_image,
        positive_fraction=cfg.positive_fraction,
        proposal_append_gt=cfg.proposal_append_gt,
    )
