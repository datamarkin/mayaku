"""QueryRCNN detector — backbone + FPN + QueryHead (no RPN, no NMS).

Follows the original Sparse R-CNN (PeizeSun/SparseR-CNN) architecture:
absolute xyxy boxes throughout, Faster-RCNN-style delta encoding,
weight_dict applied externally to loss dict, DDP-aware normalization.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from mayaku.config.schemas import MayakuConfig
from mayaku.models.backbones import build_bottom_up
from mayaku.models.heads.query_head import QueryHead
from mayaku.models.losses.set_criterion import SetCriterion
from mayaku.models.necks import FPN
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

        mean_t = torch.tensor(pixel_mean, dtype=torch.float32).view(-1, 1, 1)
        std_t = torch.tensor(pixel_std, dtype=torch.float32).view(-1, 1, 1)
        self.register_buffer("pixel_mean", mean_t, persistent=False)
        self.register_buffer("pixel_std", std_t, persistent=False)

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
        return self._inference(outputs_list[-1], images.image_sizes)

    def _prepare_targets(
        self,
        gt_instances: list[Instances],
        image_sizes: list[tuple[int, int]],
    ) -> list[dict[str, Tensor]]:
        """Convert GT instances to the target format matching original Sparse R-CNN."""
        device = self.pixel_mean.device
        targets = []
        for instances, (h, w) in zip(gt_instances, image_sizes):
            raw_boxes = instances.gt_boxes
            gt_boxes_xyxy = raw_boxes.tensor if isinstance(raw_boxes, Boxes) else raw_boxes
            gt_classes = instances.gt_classes

            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float32, device=device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).expand(len(gt_classes), -1)

            targets.append({
                "labels": gt_classes.to(device),
                "boxes_xyxy": gt_boxes_xyxy.to(device),
                "image_size_xyxy": image_size_xyxy,
                "image_size_xyxy_tgt": image_size_xyxy_tgt,
            })
        return targets

    def _inference(
        self,
        outputs: dict[str, Tensor],
        image_sizes: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        """Post-process last stage predictions into per-image Instances.

        Matches original: sigmoid scores, flatten N*K, take topk=N.
        """
        pred_logits = outputs["pred_logits"]  # (B, N, K)
        pred_boxes = outputs["pred_boxes"]  # (B, N, 4) absolute xyxy

        batch_size = pred_logits.shape[0]
        num_proposals = pred_logits.shape[1]
        results = []

        scores = pred_logits.sigmoid()  # (B, N, K)
        labels = torch.arange(self.num_classes, device=pred_logits.device)
        labels = labels.unsqueeze(0).repeat(num_proposals, 1).flatten(0, 1)  # (N*K,)

        for b in range(batch_size):
            h, w = image_sizes[b]

            # Flatten across proposals × classes, take top-N
            scores_per_image = scores[b]  # (N, K)
            box_pred = pred_boxes[b]  # (N, 4)

            scores_flat, topk_indices = scores_per_image.flatten(0, 1).topk(
                min(self.detections_per_image, scores_per_image.numel()), sorted=False
            )
            labels_per_image = labels[topk_indices]

            # Each proposal's box is repeated for all classes, then indexed
            box_pred_expanded = box_pred.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
            boxes_per_image = box_pred_expanded[topk_indices]

            # Clip to image bounds
            boxes_per_image[:, 0::2].clamp_(min=0, max=w)
            boxes_per_image[:, 1::2].clamp_(min=0, max=h)

            # Filter by score threshold
            keep = scores_flat > self.score_thresh
            inst = Instances(
                image_size=(h, w),
                pred_boxes=Boxes(boxes_per_image[keep]),
                scores=scores_flat[keep],
                pred_classes=labels_per_image[keep],
            )
            results.append({"instances": inst})

        return results

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

    # Weight dict: applied to raw losses in forward()
    weight_dict = {
        "loss_ce": qr_cfg.cost_class,    # 2.0
        "loss_bbox": qr_cfg.cost_bbox,   # 5.0
        "loss_giou": qr_cfg.cost_giou,   # 2.0
    }

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
    )
