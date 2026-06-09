"""Query Generation Network — image-conditioned query init for QueryRCNN.

Follows Featurized Query R-CNN (arXiv 2206.06258): a light anchor-free
dense head on FPN levels predicts, per location,

    objectness (class-agnostic, 1ch) | box ltrb (4ch) | query feature (Dch)

Top-K locations by objectness become the initial query boxes + features
for the iterative head, replacing the blind learned embeddings. K is
fixed, so the exported graph keeps static shapes (TopK + Gather only).

Training uses one-to-one quality matching inside GT boxes
(Q = obj^(1-alpha) * IoU^alpha, Hungarian over candidates restricted to
points inside the GT), with focal loss on objectness and GIoU on the
matched boxes — per the paper, Eq. 2/3.

Following the paper, the stride-4 level (p2) is skipped: it costs the
most compute and adds no recall.
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional as F

from mayaku.models.losses.set_criterion import (
    _pairwise_iou,
    paired_generalized_box_iou,
)

__all__ = ["QueryGenerator", "qgn_loss"]


class QueryGenerator(nn.Module):
    """Dense scorer over FPN levels emitting top-K featurized queries."""

    def __init__(
        self,
        *,
        in_channels: int = 256,
        hidden_dim: int = 256,
        num_proposals: int = 300,
        strides: tuple[int, ...] = (8, 16, 32),
        quality_alpha: float = 0.8,
    ) -> None:
        super().__init__()
        self.num_proposals = num_proposals
        self.strides = strides
        self.quality_alpha = quality_alpha

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
        )
        self.objectness = nn.Conv2d(in_channels, 1, 1)
        self.ltrb = nn.Conv2d(in_channels, 4, 1)
        self.query_feat = nn.Conv2d(in_channels, hidden_dim, 1)

        for m in (self.shared[0], self.objectness, self.ltrb, self.query_feat):
            assert isinstance(m, nn.Conv2d)
            nn.init.normal_(m.weight, std=0.01)
            assert m.bias is not None
            nn.init.constant_(m.bias, 0.0)
        # focal init: start everything as background
        assert self.objectness.bias is not None
        nn.init.constant_(self.objectness.bias, -4.59)  # p ~ 0.01

    def forward(
        self,
        features: list[Tensor],
        image_sizes: list[tuple[int, int]],
        *,
        num_proposals_override: int | None = None,
    ) -> dict[str, Tensor]:
        """features: FPN levels matching self.strides (p3, p4, p5 by default).

        Returns dict with:
            boxes      (B, K, 4) absolute xyxy, clamped to image
            feats      (B, K, D) query features for the selected locations
            obj_logits (B, L)    flattened objectness over all locations
            pred_boxes (B, L, 4) decoded boxes at every location (for loss)
        """
        k = num_proposals_override or self.num_proposals
        device = features[0].device

        obj_all, box_all, feat_all, ctr_all = [], [], [], []
        for feat, stride in zip(features, self.strides, strict=True):
            t = self.shared(feat)
            obj = self.objectness(t)  # (B,1,H,W)
            ltrb = F.softplus(self.ltrb(t)) * stride  # (B,4,H,W) >= 0, px
            qf = self.query_feat(t)  # (B,D,H,W)

            h, w = feat.shape[-2:]
            ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
            xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
            cy, cx = torch.meshgrid(ys, xs, indexing="ij")
            centers = torch.stack([cx, cy], dim=-1).view(-1, 2)  # (H*W, 2)

            left, t_, r, b = ltrb.flatten(2).unbind(1)  # each (B, H*W)
            boxes = torch.stack(
                [
                    centers[None, :, 0] - left,
                    centers[None, :, 1] - t_,
                    centers[None, :, 0] + r,
                    centers[None, :, 1] + b,
                ],
                dim=-1,
            )  # (B, H*W, 4)

            obj_all.append(obj.flatten(1))  # (B, H*W)
            box_all.append(boxes)
            feat_all.append(qf.flatten(2).transpose(1, 2))  # (B, H*W, D)
            ctr_all.append(centers)

        obj_logits = torch.cat(obj_all, dim=1)  # (B, L)
        pred_boxes = torch.cat(box_all, dim=1)  # (B, L, 4)
        pred_feats = torch.cat(feat_all, dim=1)  # (B, L, D)
        all_centers = torch.cat(ctr_all, dim=0)  # (L, 2) — for the loss

        k = min(k, obj_logits.shape[1])
        _, topk = obj_logits.topk(k, dim=1)  # (B, K) — fixed K, static shape
        sel_boxes = pred_boxes.gather(1, topk[..., None].expand(-1, -1, 4))
        sel_feats = pred_feats.gather(1, topk[..., None].expand(-1, -1, pred_feats.shape[-1]))

        # clamp to per-image bounds
        whwh = torch.tensor(
            [[w, h, w, h] for h, w in image_sizes],
            dtype=torch.float32,
            device=device,
        )[:, None, :]
        sel_boxes = sel_boxes.clamp(min=0)
        sel_boxes = torch.min(sel_boxes, whwh)

        return {
            "boxes": sel_boxes,
            "feats": sel_feats,
            "obj_logits": obj_logits,
            "pred_boxes": pred_boxes,
            "centers": all_centers,
        }


@torch.no_grad()
def _quality_match(
    obj_logits: Tensor,  # (L,)
    pred_boxes: Tensor,  # (L, 4)
    gt_boxes: Tensor,  # (M, 4)
    centers: Tensor,  # (L, 2)
    alpha: float,
) -> tuple[Tensor, Tensor]:
    """One-to-one quality matching: Q = obj^(1-a) * IoU^a, candidates
    restricted to locations whose center lies inside the GT box."""
    inside = (
        (centers[:, 0][None] > gt_boxes[:, 0][:, None])
        & (centers[:, 0][None] < gt_boxes[:, 2][:, None])
        & (centers[:, 1][None] > gt_boxes[:, 1][:, None])
        & (centers[:, 1][None] < gt_boxes[:, 3][:, None])
    )  # (M, L)
    iou = _pairwise_iou(gt_boxes, pred_boxes)  # (M, L)
    obj = obj_logits.sigmoid().clamp(1e-6)[None]  # (1, L)
    quality = obj.pow(1 - alpha) * iou.clamp(1e-6).pow(alpha)
    quality = quality * inside  # zero outside GT
    gt_idx, loc_idx = linear_sum_assignment(-quality.cpu().numpy())
    gt_idx = torch.as_tensor(gt_idx, device=pred_boxes.device)
    loc_idx = torch.as_tensor(loc_idx, device=pred_boxes.device)
    # drop degenerate matches (GT with no inside candidate)
    valid = quality[gt_idx, loc_idx] > 0
    return gt_idx[valid], loc_idx[valid]


def qgn_loss(
    qgn_out: dict[str, Tensor],
    targets: list[dict[str, Tensor]],
    centers: Tensor,
    *,
    quality_alpha: float = 0.8,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> dict[str, Tensor]:
    """Focal objectness + GIoU box loss with quality matching.

    `centers` is the (L, 2) absolute center grid concatenated across the
    QGN's levels (constant per input resolution; built by the caller).
    """
    obj_logits = qgn_out["obj_logits"].float()  # (B, L)
    pred_boxes = qgn_out["pred_boxes"].float()  # (B, L, 4)

    total_obj = obj_logits.new_zeros(())
    total_giou = obj_logits.new_zeros(())
    num_gt = max(sum(len(t["labels"]) for t in targets), 1)

    for b, t in enumerate(targets):
        gt = t["boxes_xyxy"].float()
        obj_target = torch.zeros_like(obj_logits[b])
        if gt.numel():
            gt_idx, loc_idx = _quality_match(
                obj_logits[b], pred_boxes[b], gt, centers, quality_alpha
            )
            obj_target[loc_idx] = 1.0
            if loc_idx.numel():
                giou = paired_generalized_box_iou(pred_boxes[b][loc_idx], gt[gt_idx])
                total_giou = total_giou + (1.0 - giou).sum()

        p = obj_logits[b].sigmoid()
        ce = F.binary_cross_entropy_with_logits(obj_logits[b], obj_target, reduction="none")
        p_t = p * obj_target + (1 - p) * (1 - obj_target)
        a_t = focal_alpha * obj_target + (1 - focal_alpha) * (1 - obj_target)
        total_obj = total_obj + (a_t * (1 - p_t).pow(focal_gamma) * ce).sum()

    return {
        "loss_qgn_obj": total_obj / num_gt,
        "loss_qgn_giou": total_giou / num_gt,
    }
