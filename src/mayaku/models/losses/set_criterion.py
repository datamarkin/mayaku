"""Hungarian-matching set prediction loss for QueryRCNN.

Faithfully follows the original Sparse R-CNN (PeizeSun/SparseR-CNN)
loss computation: sigmoid focal loss with sum reduction, L1 on
image-size-normalized boxes, GIoU on absolute xyxy, external weight
application, and DDP-aware num_boxes normalization.
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["SetCriterion"]


class SetCriterion(nn.Module):
    """Set prediction loss with Hungarian matching and deep supervision.

    Loss weights (class_weight, l1_weight, giou_weight) are applied by
    the detector's forward(), NOT inside this module — matching the
    original Sparse R-CNN's weight_dict pattern.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cascade_iou_thresholds: tuple[float, ...] = (),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.cascade_iou_thresholds = cascade_iou_thresholds

    def forward(
        self,
        outputs_list: list[dict[str, Tensor]],
        targets: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        """Compute raw (unweighted) losses with deep supervision.

        Returns loss dict with keys: loss_ce_{i}, loss_bbox_{i}, loss_giou_{i}
        for each stage i. The caller applies weight_dict to these.
        """
        num_boxes_int = sum(len(t["labels"]) for t in targets)
        num_boxes_t = torch.as_tensor(
            [num_boxes_int], dtype=torch.float32, device=outputs_list[0]["pred_logits"].device
        )
        # DDP: all-reduce num_boxes across ranks for consistent normalization
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes_t)
            num_boxes_t = num_boxes_t / torch.distributed.get_world_size()
        num_boxes = float(torch.clamp(num_boxes_t, min=1).item())

        losses: dict[str, Tensor] = {}
        for stage_idx, outputs in enumerate(outputs_list):
            stage_losses = self._single_stage_loss(outputs, targets, num_boxes, stage_idx)
            for k, v in stage_losses.items():
                losses[f"{k}_{stage_idx}"] = v
        return losses

    def _single_stage_loss(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        num_boxes: float,
        stage_idx: int = 0,
    ) -> dict[str, Tensor]:
        pred_logits = outputs["pred_logits"]  # (B, N, K)
        pred_boxes = outputs["pred_boxes"]  # (B, N, 4) absolute xyxy

        indices = self._hungarian_match(pred_logits, pred_boxes, targets, stage_idx)

        loss_ce = self._loss_labels(pred_logits, targets, indices, num_boxes)
        loss_bbox, loss_giou = self._loss_boxes(pred_boxes, targets, indices, num_boxes)

        return {"loss_ce": loss_ce, "loss_bbox": loss_bbox, "loss_giou": loss_giou}

    @torch.no_grad()
    def match(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        stage_idx: int = 0,
    ) -> list[tuple[Tensor, Tensor]]:
        """Run Hungarian matching on a single stage's outputs."""
        return self._hungarian_match(
            outputs["pred_logits"], outputs["pred_boxes"], targets, stage_idx,
        )

    def denoising_loss(
        self,
        dn: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        """Box-only DN loss (L1 + GIoU) with deep supervision.

        Each DN query's target is the clean GT it was noised from — no
        matching needed. L1 is on image-size-normalized boxes (matching the
        main box loss); both are normalized by the number of DN queries.
        Returns ``loss_dn_bbox_{i}`` / ``loss_dn_giou_{i}`` per stage.
        """
        tgt = dn["tgt_boxes"].float()  # (B, M, 4)
        valid = dn["valid"]            # (B, M) bool
        img = torch.stack(
            [t["image_size_xyxy"] for t in targets]
        ).float().unsqueeze(1)  # (B, 1, 4)
        num_dn = valid.sum().clamp(min=1)     # 0-dim tensor — no host sync
        vmask = valid.unsqueeze(-1).float()   # (B, M, 1)
        tgt_valid = tgt[valid]                # (V, 4) — loop-invariant

        losses: dict[str, Tensor] = {}
        for i, pred in enumerate(dn["pred_boxes"]):
            pred = pred.float()
            l1 = F.l1_loss(pred / img, tgt / img, reduction="none") * vmask
            # Empty valid -> empty gather -> giou sum is a clean 0; no guard needed.
            giou = paired_generalized_box_iou(pred[valid], tgt_valid)
            losses[f"loss_dn_bbox_{i}"] = l1.sum() / num_dn
            losses[f"loss_dn_giou_{i}"] = (1 - giou).sum() / num_dn
        return losses

    @torch.no_grad()
    def _hungarian_match(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        targets: list[dict[str, Tensor]],
        stage_idx: int = 0,
    ) -> list[tuple[Tensor, Tensor]]:
        # Force fp32 — fp16 overflows on absolute-xyxy GIoU and focal log
        pred_logits = pred_logits.float()
        pred_boxes = pred_boxes.float()

        batch_size, _ = pred_logits.shape[:2]
        indices = []

        iou_floor = 0.0
        if self.cascade_iou_thresholds:
            idx = min(stage_idx, len(self.cascade_iou_thresholds) - 1)
            iou_floor = self.cascade_iou_thresholds[idx]

        for b in range(batch_size):
            tgt_labels = targets[b]["labels"]
            tgt_boxes_xyxy = targets[b]["boxes_xyxy"]  # absolute xyxy

            if tgt_labels.shape[0] == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long, device=pred_logits.device),
                    torch.tensor([], dtype=torch.long, device=pred_logits.device),
                ))
                continue

            out_prob = pred_logits[b].sigmoid()
            alpha, gamma = self.focal_alpha, self.focal_gamma
            neg_cost = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost[:, tgt_labels] - neg_cost[:, tgt_labels]

            # L1 cost on image-size-normalized boxes
            image_size_xyxy = targets[b]["image_size_xyxy"]  # (4,)
            out_bbox_norm = pred_boxes[b] / image_size_xyxy.unsqueeze(0)
            tgt_bbox_norm = tgt_boxes_xyxy / targets[b]["image_size_xyxy_tgt"]
            cost_bbox = torch.cdist(out_bbox_norm, tgt_bbox_norm, p=1)

            # GIoU cost on absolute xyxy
            cost_giou = -generalized_box_iou(pred_boxes[b], tgt_boxes_xyxy)

            cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )

            if iou_floor > 0.0:
                iou = _pairwise_iou(pred_boxes[b], tgt_boxes_xyxy)
                cost = cost + (iou < iou_floor).float() * 1e6

            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long, device=pred_logits.device),
                torch.as_tensor(col_ind, dtype=torch.long, device=pred_logits.device),
            ))
        return indices

    def _loss_labels(
        self,
        pred_logits: Tensor,
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: float,
    ) -> Tensor:
        """Sigmoid focal loss — sum reduction, divided by num_boxes."""
        pred_logits = pred_logits.float()
        batch_size, num_queries, num_classes = pred_logits.shape
        target_classes = torch.full(
            (batch_size, num_queries), num_classes,
            dtype=torch.long, device=pred_logits.device,
        )
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.shape[0] > 0:
                target_classes[b, src_idx] = targets[b]["labels"][tgt_idx]

        # Flatten to (B*N, K) — matching original's flatten(0, 1)
        src_logits = pred_logits.flatten(0, 1)
        target_classes_flat = target_classes.flatten(0, 1)

        # One-hot targets
        labels = torch.zeros_like(src_logits)
        pos_inds = (target_classes_flat != num_classes).nonzero(as_tuple=True)[0]
        labels[pos_inds, target_classes_flat[pos_inds]] = 1.0

        # Focal loss: sum reduction, then / num_boxes
        loss = sigmoid_focal_loss(
            src_logits, labels,
            alpha=self.focal_alpha, gamma=self.focal_gamma,
        ) / num_boxes
        return loss

    def _loss_boxes(
        self,
        pred_boxes: Tensor,
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: float,
    ) -> tuple[Tensor, Tensor]:
        """L1 (on normalized boxes) + GIoU (on absolute xyxy)."""
        pred_boxes = pred_boxes.float()
        src_list, tgt_list, tgt_norm_list, src_norm_list = [], [], [], []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.shape[0] == 0:
                continue
            src_list.append(pred_boxes[b, src_idx])
            tgt_list.append(targets[b]["boxes_xyxy"][tgt_idx])
            # Normalize by image size for L1 (matches original)
            tgt_norm_list.append(
                targets[b]["boxes_xyxy"][tgt_idx] / targets[b]["image_size_xyxy_tgt"][tgt_idx]
            )
            image_size = targets[b]["image_size_xyxy"]
            src_norm_list.append(pred_boxes[b, src_idx] / image_size.unsqueeze(0))

        if not src_list:
            zero = pred_boxes.sum() * 0.0
            return zero, zero

        src_boxes = torch.cat(src_list, dim=0)
        tgt_boxes = torch.cat(tgt_list, dim=0)
        src_norm = torch.cat(src_norm_list, dim=0)
        tgt_norm = torch.cat(tgt_norm_list, dim=0)

        # L1 on normalized boxes
        loss_bbox = F.l1_loss(src_norm, tgt_norm, reduction="none").sum() / num_boxes

        # GIoU on absolute xyxy (paired, not full NxN matrix)
        loss_giou = (1 - paired_generalized_box_iou(src_boxes, tgt_boxes)).sum() / num_boxes

        return loss_bbox, loss_giou


# ---------------------------------------------------------------------------
# Focal loss (matches fvcore's sigmoid_focal_loss with reduction="sum")
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(
    inputs: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2.0,
) -> Tensor:
    """Sigmoid focal loss — sum over all elements."""
    p = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce
    return loss.sum()


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------

def paired_generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Element-wise GIoU between paired (N, 4) boxes. Returns (N,)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    enclose_lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    enclose_rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
    return iou - (enclose_area - union) / enclose_area.clamp(min=1e-6)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """GIoU between (N, 4) and (M, 4) boxes in xyxy absolute format. Returns (N, M)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    enclose_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enclose_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

    return iou - (enclose_area - union) / enclose_area.clamp(min=1e-6)


def _pairwise_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Standard IoU (N, 4) vs (M, 4) in xyxy. Returns (N, M)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)
