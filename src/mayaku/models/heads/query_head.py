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

from mayaku.models.heads.query_denoising import build_dn_groups, dn_attention_mask
from mayaku.models.heads.query_generator import QueryGenerator
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
        query_generator: QueryGenerator | None = None,
        qgn_feature_indices: Sequence[int] = (),
        denoising: bool = False,
        dn_groups: int = 5,
        dn_box_noise_scale: float = 0.4,
    ) -> None:
        super().__init__()
        self.num_proposals = num_proposals
        self.hidden_dim = hidden_dim
        self.num_stages = num_stages

        # Optional QGN (Featurized Query R-CNN): image-conditioned queries
        # replace the blind learned embeddings below.
        self.query_generator = query_generator
        self.qgn_feature_indices = tuple(qgn_feature_indices)

        # Optional DN-DETR-style denoising (box-only). A single shared content
        # embedding is the "query feature" for every noised-GT box; the box
        # location carries the signal, so no per-class label embedding.
        self.denoising = denoising
        self.dn_groups = dn_groups
        self.dn_box_noise_scale = dn_box_noise_scale
        if denoising:
            self.dn_query_feat = nn.Embedding(1, hidden_dim)

        if query_generator is None:
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
        targets: list[dict[str, Tensor]] | None = None,
    ) -> list[dict[str, Tensor]]:
        """Run iterative refinement.

        Returns list of per-stage dicts with pred_logits (B,N,K) and
        pred_boxes (B,N,4) in absolute xyxy. When ``targets`` is given and
        denoising is enabled (training only), noised-GT denoising queries
        ride through the stages alongside the matching queries; their
        per-stage boxes are attached to ``results[-1]["dn"]`` for the loss
        and stripped from the matching outputs.
        """
        batch_size = features[0].shape[0]
        device = features[0].device

        num_proposals = num_proposals_override or self.num_proposals
        num_stages = num_stages_override or self.num_stages

        images_whwh = torch.tensor(
            [[w, h, w, h] for h, w in image_sizes],
            dtype=torch.float32, device=device,
        )  # (B, 4)

        qgn_out: dict[str, Tensor] | None = None
        if self.query_generator is not None:
            qgn_feats = [features[i] for i in self.qgn_feature_indices]
            qgn_out = self.query_generator(
                qgn_feats, image_sizes,
                num_proposals_override=num_proposals_override,
            )
            # Boxes are detached into stage 1 (consistent with the
            # inter-stage detach); query features keep their gradient so
            # head losses train the QGN feature branch.
            bboxes = qgn_out["boxes"].detach()  # (B, K, 4) absolute xyxy
            k = bboxes.shape[1]
            proposal_features = qgn_out["feats"].reshape(1, batch_size * k, -1)
        else:
            # Convert learned cxcywh proposals to absolute xyxy
            proposal_boxes_xyxy = _cxcywh_to_xyxy(
                self.init_proposal_boxes.weight[:num_proposals])
            bboxes = proposal_boxes_xyxy[None] * images_whwh[:, None, :]  # (B, N, 4)

            # Expand proposal features: original uses [None].repeat(1, bs, 1)
            proposal_features = self.init_proposal_features.weight[:num_proposals]
            proposal_features = proposal_features[None].repeat(1, batch_size, 1)  # (1, B*N, d)

        # Append DN queries (training only). They flow through the same stages,
        # isolated from the matching queries by an attention mask.
        num_match = bboxes.shape[1]
        dn: dict[str, Tensor] | None = None
        attn_mask: Tensor | None = None
        if self.training and self.denoising and targets is not None:
            dn = build_dn_groups(
                targets, dn_groups=self.dn_groups,
                box_noise_scale=self.dn_box_noise_scale, device=device,
            )
            if dn is not None:
                num_dn = dn["boxes"].shape[1]
                bboxes = torch.cat([bboxes, dn["boxes"]], dim=1)  # (B, N+M, 4)
                pf = proposal_features.view(batch_size, num_match, -1)
                dn_feat = self.dn_query_feat.weight.view(1, 1, -1).expand(
                    batch_size, num_dn, -1)
                proposal_features = torch.cat([pf, dn_feat], dim=1).reshape(
                    1, batch_size * (num_match + num_dn), -1)
                attn_mask = dn_attention_mask(num_match, num_dn, device)

        outputs_class_list = []
        outputs_coord_list = []
        dn_coord_list: list[Tensor] = []

        # Per-image bounds for clamping boxes between stages: (B, 1, 4)
        clamp_bounds = images_whwh[:, None, :]  # [W, H, W, H] per image

        for stage in self.head_series[:num_stages]:
            class_logits, pred_bboxes, proposal_features = stage(
                features, bboxes, proposal_features, self.box_pooler, attn_mask
            )
            # Split matching queries from DN queries (DN occupies the tail).
            outputs_class_list.append(class_logits[:, :num_match])
            outputs_coord_list.append(pred_bboxes[:, :num_match])
            if dn is not None:
                dn_coord_list.append(pred_bboxes[:, num_match:])
            bboxes = pred_bboxes.detach().clamp(min=0)
            bboxes = torch.min(bboxes, clamp_bounds)

        # Strip DN from the final obj_features so downstream (mask/keypoint)
        # heads and the criterion only ever see the matching queries.
        if dn is not None:
            total = num_match + num_dn
            proposal_features = proposal_features.view(
                batch_size, total, -1)[:, :num_match].reshape(
                1, batch_size * num_match, -1)

        results = [
            {"pred_logits": cls, "pred_boxes": box}
            for cls, box in zip(outputs_class_list, outputs_coord_list)
        ]
        results[-1]["obj_features"] = proposal_features
        if qgn_out is not None:
            results[-1]["qgn"] = qgn_out
        if dn is not None:
            results[-1]["dn"] = {
                "pred_boxes": dn_coord_list,      # list of (B, M, 4) per stage
                "tgt_boxes": dn["tgt_boxes"],     # (B, M, 4)
                "valid": dn["valid"],             # (B, M)
            }
        return results


def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
