"""QueryRCNN detector — backbone + FPN + QueryHead (no RPN, no NMS).

Follows the original Sparse R-CNN (PeizeSun/SparseR-CNN) architecture:
absolute xyxy boxes throughout, Faster-RCNN-style delta encoding,
weight_dict applied externally to loss dict, DDP-aware normalization.

Phase 2 adds a dynamic mask head (QueryInst pattern) driven by
obj_features from the last iterative stage.
Phase 3 adds a standard heatmap keypoint head on ROI-pooled features.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from mayaku.config.schemas import MayakuConfig
from mayaku.models.backbones import build_bottom_up
from mayaku.models.heads.mask_head import mask_rcnn_inference, mask_rcnn_loss
from mayaku.models.heads.keypoint_head import (
    KRCNNConvDeconvUpsampleHead,
    keypoint_rcnn_inference,
    keypoint_rcnn_loss,
    select_proposals_with_visible_keypoints,
)
from mayaku.models.heads.query_head import QueryHead
from mayaku.models.heads.query_mask_head import QueryDynamicMaskHead
from mayaku.models.losses.set_criterion import SetCriterion
from mayaku.models.necks import FPN
from mayaku.models.poolers import ROIPooler
from mayaku.structures.boxes import Boxes
from mayaku.structures.image_list import ImageList
from mayaku.structures.instances import Instances

__all__ = ["QueryRCNN", "build_query_rcnn"]


class QueryRCNN(nn.Module):
    """Set-prediction detector with learned query proposals."""

    def __init__(
        self,
        backbone: nn.Module,
        head: QueryHead,
        criterion: SetCriterion,
        *,
        pixel_mean: Sequence[float],
        pixel_std: Sequence[float],
        num_classes: int,
        weight_dict: dict[str, float],
        feature_keys: Sequence[str],
        score_thresh: float = 0.05,
        detections_per_image: int = 100,
        inference_num_stages: int | None = None,
        inference_num_proposals: int | None = None,
        mask_head: QueryDynamicMaskHead | None = None,
        mask_pooler: ROIPooler | None = None,
        keypoint_head: nn.Module | None = None,
        keypoint_pooler: ROIPooler | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.criterion = criterion
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self._feature_keys = tuple(feature_keys)
        self.score_thresh = score_thresh
        self.detections_per_image = detections_per_image
        self.inference_num_stages = inference_num_stages
        self.inference_num_proposals = inference_num_proposals
        self.mask_head = mask_head
        self.mask_pooler = mask_pooler
        self.keypoint_head = keypoint_head
        self.keypoint_pooler = keypoint_pooler

        mean_t = torch.tensor(pixel_mean, dtype=torch.float32).view(-1, 1, 1)
        std_t = torch.tensor(pixel_std, dtype=torch.float32).view(-1, 1, 1)
        self.register_buffer("pixel_mean", mean_t, persistent=False)
        self.register_buffer("pixel_std", std_t, persistent=False)

    @property
    def mask_on(self) -> bool:
        return self.mask_head is not None

    @property
    def keypoint_on(self) -> bool:
        return self.keypoint_head is not None

    def forward(
        self, batched_inputs: Sequence[dict[str, Any]]
    ) -> dict[str, Tensor] | list[dict[str, Any]]:
        images = self._preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        feature_list = [features[k] for k in self._feature_keys]

        if self.training:
            gt_instances = [x["instances"].to(self.pixel_mean.device) for x in batched_inputs]
            targets = self._prepare_targets(gt_instances, images.image_sizes)

            outputs_list = self.head(feature_list, images.image_sizes)
            loss_dict = self.criterion(outputs_list, targets)

            if self.mask_on or self.keypoint_on:
                last = outputs_list[-1]
                mask_kp_losses = self._forward_mask_keypoint_train(
                    feature_list, last, targets, images.image_sizes,
                )
                loss_dict.update(mask_kp_losses)

            # Apply weight_dict externally (matches original Sparse R-CNN)
            weighted = {}
            for k, v in loss_dict.items():
                base_key = _strip_stage_suffix(k)
                if base_key in self.weight_dict:
                    weighted[k] = v * self.weight_dict[base_key]
                else:
                    weighted[k] = v
            return weighted

        outputs_list = self.head(
            feature_list, images.image_sizes,
            num_stages_override=self.inference_num_stages,
            num_proposals_override=self.inference_num_proposals,
        )
        return self._inference(outputs_list[-1], feature_list, images.image_sizes)

    # ------------------------------------------------------------------
    # Training: mask + keypoint branches
    # ------------------------------------------------------------------

    def _forward_mask_keypoint_train(
        self,
        feature_list: list[Tensor],
        last_outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        image_sizes: list[tuple[int, int]],
    ) -> dict[str, Tensor]:
        num_stages = len(self.head.head_series)
        indices = self.criterion.match(last_outputs, targets, stage_idx=num_stages - 1)

        fg_instances = self._build_fg_instances(indices, targets, image_sizes)

        losses: dict[str, Tensor] = {}
        if self.mask_on:
            losses.update(self._forward_mask_train(
                feature_list, last_outputs, fg_instances, indices,
            ))
        if self.keypoint_on:
            losses.update(self._forward_keypoint_train(feature_list, fg_instances))
        return losses

    def _forward_mask_train(
        self,
        feature_list: list[Tensor],
        last_outputs: dict[str, Tensor],
        fg_instances: list[Instances],
        indices: list[tuple[Tensor, Tensor]],
    ) -> dict[str, Tensor]:
        valid_mask = [len(inst) > 0 and inst.has("gt_masks") for inst in fg_instances]
        valid = [inst for inst, m in zip(fg_instances, valid_mask) if m]
        if not valid:
            return {"loss_mask": self.mask_head.kernel_fc.weight.sum() * 0.0}

        assert self.mask_pooler is not None
        mask_roi_feats = self.mask_pooler(feature_list, [inst.proposal_boxes for inst in valid])

        obj_features = last_outputs["obj_features"]  # (1, B*N, d)
        valid_indices = [idx for idx, m in zip(indices, valid_mask) if m]
        fg_obj = self._gather_fg_obj_features(
            obj_features, valid_indices, last_outputs["pred_logits"].shape[:2],
        )

        mask_logits = self.mask_head(mask_roi_feats, fg_obj)
        loss_mask = mask_rcnn_loss(mask_logits, valid, cls_agnostic=True)
        return {"loss_mask": loss_mask}

    def _forward_keypoint_train(
        self,
        feature_list: list[Tensor],
        fg_instances: list[Instances],
    ) -> dict[str, Tensor]:
        kp_instances = [
            inst for inst in fg_instances
            if len(inst) > 0 and inst.has("gt_keypoints")
        ]
        kp_instances, _ = select_proposals_with_visible_keypoints(kp_instances)
        valid = [inst for inst in kp_instances if len(inst) > 0]
        if not valid:
            return {"loss_keypoint": next(self.keypoint_head.parameters()).sum() * 0.0}

        assert self.keypoint_pooler is not None
        kp_roi_feats = self.keypoint_pooler(feature_list, [inst.proposal_boxes for inst in valid])
        kp_logits = self.keypoint_head(kp_roi_feats)
        loss_kp = keypoint_rcnn_loss(kp_logits, valid)
        return {"loss_keypoint": loss_kp}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _inference(
        self,
        outputs: dict[str, Tensor],
        feature_list: list[Tensor],
        image_sizes: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        pred_logits = outputs["pred_logits"]  # (B, N, K)
        pred_boxes = outputs["pred_boxes"]  # (B, N, 4) absolute xyxy

        batch_size = pred_logits.shape[0]
        num_proposals = pred_logits.shape[1]
        results = []

        scores = pred_logits.sigmoid()  # (B, N, K)
        labels = torch.arange(self.num_classes, device=pred_logits.device)
        labels = labels.unsqueeze(0).repeat(num_proposals, 1).flatten(0, 1)  # (N*K,)

        per_image_proposal_indices: list[Tensor] = []

        for b in range(batch_size):
            h, w = image_sizes[b]

            scores_per_image = scores[b]  # (N, K)
            box_pred = pred_boxes[b]  # (N, 4)

            scores_flat, topk_indices = scores_per_image.flatten(0, 1).topk(
                min(self.detections_per_image, scores_per_image.numel()), sorted=False
            )
            labels_per_image = labels[topk_indices]
            proposal_indices = topk_indices // self.num_classes

            box_pred_expanded = box_pred.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
            boxes_per_image = box_pred_expanded[topk_indices]

            boxes_per_image[:, 0::2].clamp_(min=0, max=w)
            boxes_per_image[:, 1::2].clamp_(min=0, max=h)

            keep = scores_flat > self.score_thresh
            inst = Instances(
                image_size=(h, w),
                pred_boxes=Boxes(boxes_per_image[keep]),
                scores=scores_flat[keep],
                pred_classes=labels_per_image[keep],
            )
            per_image_proposal_indices.append(proposal_indices[keep])
            results.append({"instances": inst})

        if self.mask_on:
            self._forward_mask_infer(feature_list, outputs, results, per_image_proposal_indices)
        if self.keypoint_on:
            self._forward_keypoint_infer(feature_list, results)

        return results

    def _forward_mask_infer(
        self,
        feature_list: list[Tensor],
        outputs: dict[str, Tensor],
        results: list[dict[str, Any]],
        per_image_proposal_indices: list[Tensor],
    ) -> None:
        instances = [r["instances"] for r in results]
        if all(len(inst) == 0 for inst in instances):
            return

        assert self.mask_pooler is not None
        mask_roi_feats = self.mask_pooler(feature_list, [inst.pred_boxes for inst in instances])

        obj_features = outputs["obj_features"].squeeze(0)  # (B*N, d)
        N = outputs["pred_logits"].shape[1]
        parts = []
        for b, pidx in enumerate(per_image_proposal_indices):
            if pidx.numel() > 0:
                parts.append(obj_features[b * N + pidx])
            else:
                parts.append(obj_features.new_zeros(0, obj_features.shape[-1]))
        fg_obj = torch.cat(parts, dim=0)

        mask_logits = self.mask_head(mask_roi_feats, fg_obj)
        mask_rcnn_inference(mask_logits, instances, cls_agnostic=True)

    def _forward_keypoint_infer(
        self,
        feature_list: list[Tensor],
        results: list[dict[str, Any]],
    ) -> None:
        instances = [r["instances"] for r in results]
        if all(len(inst) == 0 for inst in instances):
            return

        assert self.keypoint_pooler is not None
        kp_roi_feats = self.keypoint_pooler(feature_list, [inst.pred_boxes for inst in instances])
        kp_logits = self.keypoint_head(kp_roi_feats)
        keypoint_rcnn_inference(kp_logits, instances)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_targets(
        self,
        gt_instances: list[Instances],
        image_sizes: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        device = self.pixel_mean.device
        targets = []
        for instances, (h, w) in zip(gt_instances, image_sizes):
            raw_boxes = instances.gt_boxes
            gt_boxes_xyxy = raw_boxes.tensor if isinstance(raw_boxes, Boxes) else raw_boxes
            gt_classes = instances.gt_classes

            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float32, device=device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).expand(len(gt_classes), -1)

            t: dict[str, Any] = {
                "labels": gt_classes.to(device),
                "boxes_xyxy": gt_boxes_xyxy.to(device),
                "image_size_xyxy": image_size_xyxy,
                "image_size_xyxy_tgt": image_size_xyxy_tgt,
            }
            if self.mask_on and instances.has("gt_masks"):
                t["gt_masks"] = instances.gt_masks
            if self.keypoint_on and instances.has("gt_keypoints"):
                t["gt_keypoints"] = instances.gt_keypoints
            targets.append(t)
        return targets

    def _build_fg_instances(
        self,
        indices: list[tuple[Tensor, Tensor]],
        targets: list[dict[str, Any]],
        image_sizes: list[tuple[int, int]],
    ) -> list[Instances]:
        fg_list = []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            h, w = image_sizes[b]
            inst = Instances(image_size=(h, w))
            if src_idx.numel() == 0:
                device = targets[b]["labels"].device
                inst.proposal_boxes = Boxes(torch.zeros(0, 4, device=device))
                inst.gt_classes = torch.zeros(0, dtype=torch.long, device=device)
                fg_list.append(inst)
                continue
            inst.proposal_boxes = Boxes(targets[b]["boxes_xyxy"][tgt_idx])
            inst.gt_classes = targets[b]["labels"][tgt_idx]
            if "gt_masks" in targets[b]:
                inst.gt_masks = targets[b]["gt_masks"][tgt_idx]
            if "gt_keypoints" in targets[b]:
                inst.gt_keypoints = targets[b]["gt_keypoints"][tgt_idx]
            fg_list.append(inst)
        return fg_list

    def _gather_fg_obj_features(
        self,
        obj_features: Tensor,
        indices: list[tuple[Tensor, Tensor]],
        batch_proposals_shape: tuple[int, int],
    ) -> Tensor:
        B, N = batch_proposals_shape
        obj_flat = obj_features.squeeze(0)  # (B*N, d)
        parts = []
        for b, (src_idx, _) in enumerate(indices):
            if src_idx.numel() > 0:
                parts.append(obj_flat[b * N + src_idx])
        if not parts:
            return obj_flat.new_zeros(0, obj_flat.shape[-1])
        return torch.cat(parts, dim=0)

    def _preprocess_image(self, batched_inputs: Sequence[dict[str, Any]]) -> ImageList:
        device = self.pixel_mean.device
        images = [
            (x["image"].to(device).to(torch.float32) - self.pixel_mean) / self.pixel_std
            for x in batched_inputs
        ]
        size_divisibility = self._size_divisibility()
        return ImageList.from_tensors(images, size_divisibility=size_divisibility, pad_value=0.0)

    def _size_divisibility(self) -> int:
        sd = getattr(self.backbone, "size_divisibility", 1)
        if isinstance(sd, int):
            return sd
        if isinstance(sd, Tensor):
            return int(sd.item())
        return 1


def _strip_stage_suffix(key: str) -> str:
    """'loss_ce_3' → 'loss_ce'"""
    parts = key.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return key


# ---------------------------------------------------------------------------
# Build factory
# ---------------------------------------------------------------------------


def build_query_rcnn(cfg: MayakuConfig, *, backbone_weights: str | None = None) -> QueryRCNN:
    if cfg.model.meta_architecture != "query_rcnn":
        raise ValueError(
            f"build_query_rcnn requires meta_architecture='query_rcnn'; got "
            f"{cfg.model.meta_architecture!r}"
        )

    qr_cfg = cfg.model.query_rcnn_head
    if qr_cfg is None:
        raise ValueError("query_rcnn requires model.query_rcnn_head config section")

    bottom_up = build_bottom_up(cfg.model.backbone, weights=backbone_weights)

    fpn = FPN(
        bottom_up=bottom_up,
        in_features=cfg.model.fpn.in_features,
        out_channels=cfg.model.fpn.out_channels,
        norm=cfg.model.fpn.norm,
        fuse_type=cfg.model.fpn.fuse_type,
        top_block=None,
    )

    in_shapes = fpn.output_shape()
    feature_keys = tuple(sorted(in_shapes.keys()))
    pooler_scales = tuple(1.0 / in_shapes[k].stride for k in feature_keys)

    num_classes = cfg.model.roi_heads.num_classes

    head = QueryHead(
        num_proposals=qr_cfg.num_proposals,
        hidden_dim=qr_cfg.hidden_dim,
        num_heads=qr_cfg.num_heads,
        num_stages=qr_cfg.num_stages,
        dim_feedforward=qr_cfg.dim_feedforward,
        dim_dynamic=qr_cfg.dim_dynamic,
        dropout=qr_cfg.dropout,
        num_classes=num_classes,
        pooler_resolution=qr_cfg.pooler_resolution,
        pooler_scales=pooler_scales,
        pooler_sampling_ratio=0,
    )

    criterion = SetCriterion(
        num_classes=num_classes,
        cost_class=qr_cfg.cost_class,
        cost_bbox=qr_cfg.cost_bbox,
        cost_giou=qr_cfg.cost_giou,
        cascade_iou_thresholds=qr_cfg.cascade_iou_thresholds,
    )

    weight_dict: dict[str, float] = {
        "loss_ce": qr_cfg.cost_class,
        "loss_bbox": qr_cfg.cost_bbox,
        "loss_giou": qr_cfg.cost_giou,
    }

    # Phase 2: Mask head
    mask_head: QueryDynamicMaskHead | None = None
    mask_pooler: ROIPooler | None = None
    if cfg.model.query_rcnn_mask is not None:
        mask_cfg = cfg.model.query_rcnn_mask
        mask_pooler = ROIPooler(
            output_size=mask_cfg.pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=0,
        )
        mask_head = QueryDynamicMaskHead(
            hidden_dim=qr_cfg.hidden_dim,
            conv_dim=mask_cfg.conv_dim,
            num_conv=mask_cfg.num_conv,
            mask_resolution=mask_cfg.mask_resolution,
            pooler_resolution=mask_cfg.pooler_resolution,
        )
        weight_dict["loss_mask"] = mask_cfg.loss_weight

    # Phase 3: Keypoint head
    keypoint_head: KRCNNConvDeconvUpsampleHead | None = None
    keypoint_pooler: ROIPooler | None = None
    if cfg.model.query_rcnn_keypoint is not None:
        kp_cfg = cfg.model.query_rcnn_keypoint
        from mayaku.models.backbones._base import ShapeSpec
        keypoint_pooler = ROIPooler(
            output_size=kp_cfg.pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=0,
        )
        keypoint_head = KRCNNConvDeconvUpsampleHead(
            input_shape=ShapeSpec(channels=cfg.model.fpn.out_channels, stride=1),
            num_keypoints=kp_cfg.num_keypoints,
        )
        weight_dict["loss_keypoint"] = kp_cfg.loss_weight

    return QueryRCNN(
        backbone=fpn,
        head=head,
        criterion=criterion,
        pixel_mean=cfg.model.pixel_mean,
        pixel_std=cfg.model.pixel_std,
        num_classes=num_classes,
        weight_dict=weight_dict,
        feature_keys=feature_keys,
        score_thresh=cfg.model.roi_heads.score_thresh_test,
        detections_per_image=cfg.test.detections_per_image,
        inference_num_stages=qr_cfg.inference_num_stages,
        inference_num_proposals=qr_cfg.inference_num_proposals,
        mask_head=mask_head,
        mask_pooler=mask_pooler,
        keypoint_head=keypoint_head,
        keypoint_pooler=keypoint_pooler,
    )
