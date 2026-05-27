"""QueryHead — iterative dynamic refinement head for QueryRCNN.

Matches the original Sparse R-CNN DynamicHead: learned proposal boxes
in cxcywh (converted to absolute xyxy before each stage), learned
proposal features, and N stages of RCNNHead refinement.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from mayaku.models.heads.query_stage import QueryStage
from mayaku.models.poolers import ROIPooler
from mayaku.structures.boxes import Boxes

__all__ = ["QueryHead"]


class QueryHead(nn.Module):
    """Iterative refinement head with learned proposals.

    Proposal boxes are stored as cxcywh (cx,cy=0.5, w,h=1.0 at init)
    and converted to absolute xyxy for each stage's ROI pooling and
    delta decoding. This matches the original Sparse R-CNN exactly.
    """

    def __init__(
        self,
        *,
        num_proposals: int = 300,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_stages: int = 6,
        dim_feedforward: int = 2048,
        dim_dynamic: int = 64,
        dropout: float = 0.0,
        num_classes: int = 80,
        pooler_resolution: int = 7,
        pooler_scales: Sequence[float] = (1 / 4, 1 / 8, 1 / 16, 1 / 32),
        pooler_sampling_ratio: int = 0,
    ) -> None:
        super().__init__()
        self.num_proposals = num_proposals
        self.hidden_dim = hidden_dim
        self.num_stages = num_stages

        # Learned proposals (original init: cx,cy=0.5; w,h=1.0)
        self.init_proposal_features = nn.Embedding(num_proposals, hidden_dim)
        self.init_proposal_boxes = nn.Embedding(num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

        # ROI pooler
        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=pooler_sampling_ratio,
        )

        # Refinement stages (independent params per stage, like original's _get_clones)
        self.head_series = nn.ModuleList([
            QueryStage(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dim_dynamic=dim_dynamic,
                dropout=dropout,
                num_classes=num_classes,
                pooler_resolution=pooler_resolution,
            )
            for _ in range(num_stages)
        ])

        # Focal loss bias init
        prior_prob = 0.01
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.head_series.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for stage in self.head_series:
            nn.init.constant_(stage.class_logits.bias, self.bias_value)

    def forward(
        self,
        features: list[Tensor],
        image_sizes: list[tuple[int, int]],
        *,
        num_stages_override: int | None = None,
        num_proposals_override: int | None = None,
    ) -> list[dict[str, Tensor]]:
        """Run iterative refinement.

        Returns list of per-stage dicts with pred_logits (B,N,K) and
        pred_boxes (B,N,4) in absolute xyxy.
        """
        batch_size = features[0].shape[0]
        device = features[0].device

        num_proposals = num_proposals_override or self.num_proposals
        num_stages = num_stages_override or self.num_stages

        # Convert learned cxcywh proposals to absolute xyxy
        proposal_boxes_xyxy = _cxcywh_to_xyxy(self.init_proposal_boxes.weight[:num_proposals])

        # Scale to absolute pixel coords per image: (B, N, 4)
        images_whwh = torch.tensor(
            [[w, h, w, h] for h, w in image_sizes],
            dtype=torch.float32, device=device,
        )  # (B, 4)
        bboxes = proposal_boxes_xyxy[None] * images_whwh[:, None, :]  # (B, N, 4) absolute xyxy

        # Expand proposal features: original uses [None].repeat(1, bs, 1)
        proposal_features = self.init_proposal_features.weight[:num_proposals]
        proposal_features = proposal_features[None].repeat(1, batch_size, 1)  # (1, B*N, d)

        outputs_class_list = []
        outputs_coord_list = []

        for stage in self.head_series[:num_stages]:
            class_logits, pred_bboxes, proposal_features = stage(
                features, bboxes, proposal_features, self.box_pooler
            )
            outputs_class_list.append(class_logits)
            outputs_coord_list.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        results = [
            {"pred_logits": cls, "pred_boxes": box}
            for cls, box in zip(outputs_class_list, outputs_coord_list)
        ]
        results[-1]["obj_features"] = proposal_features
        return results


def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
