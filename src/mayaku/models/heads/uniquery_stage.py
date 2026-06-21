"""Single iterative refinement stage for UniQuery.

Faithfully follows the original Sparse R-CNN (PeizeSun/SparseR-CNN)
RCNNHead and DynamicConv architecture: self-attention, dynamic instance
interaction, FFN with dropout, and separate cls/reg MLPs with LayerNorm.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["DynamicConv", "UniQueryStage"]

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class DynamicConv(nn.Module):
    """Instance-interactive head via dynamic convolution.

    Matches the original Sparse R-CNN DynamicConv exactly:
    proposal_feat → Linear → (param1, param2)
    roi_features × param1 → LayerNorm → ReLU → × param2 → LayerNorm → ReLU
    flatten(spatial × hidden) → Linear → LayerNorm → ReLU → output
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dim_dynamic: int = 64,
        num_dynamic: int = 2,
        pooler_resolution: int = 7,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim_dynamic = dim_dynamic
        self.num_params = hidden_dim * dim_dynamic

        self.dynamic_layer = nn.Linear(hidden_dim, num_dynamic * self.num_params)
        self.norm1 = nn.LayerNorm(dim_dynamic)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU(inplace=True)

        num_output = hidden_dim * pooler_resolution**2
        self.out_layer = nn.Linear(num_output, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, pro_features: Tensor, roi_features: Tensor) -> Tensor:
        """
        Args:
            pro_features: (1, N*B, hidden_dim)
            roi_features: (P*P, N*B, hidden_dim)
        Returns:
            (N*B, hidden_dim)
        """
        features = roi_features.permute(1, 0, 2)  # (N*B, P*P, hidden)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)  # (N*B, 1, params)

        param1 = parameters[:, :, : self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params :].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        out: Tensor = self.activation(features)

        return out


class UniQueryStage(nn.Module):
    """One stage of iterative refinement — matches original RCNNHead.

    Pipeline: self-attn → DynamicConv → FFN → cls MLP → reg MLP → predictions.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dim_dynamic: int = 64,
        dropout: float = 0.0,
        num_classes: int = 80,
        pooler_resolution: int = 7,
        num_cls_layers: int = 1,
        num_reg_layers: int = 3,
        bbox_weights: tuple[float, float, float, float] = (2.0, 2.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale_clamp = _DEFAULT_SCALE_CLAMP
        self.bbox_weights = bbox_weights

        # Self-attention
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Dynamic instance interaction
        self.inst_interact = DynamicConv(
            hidden_dim, dim_dynamic, pooler_resolution=pooler_resolution
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Classification MLP (original: NUM_CLS=1 layers of Linear+LN+ReLU)
        cls_layers: list[nn.Module] = []
        for _ in range(num_cls_layers):
            cls_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
        self.cls_module = nn.ModuleList(cls_layers)
        self.class_logits = nn.Linear(hidden_dim, num_classes)

        # Regression MLP (original: NUM_REG=3 layers of Linear+LN+ReLU)
        reg_layers: list[nn.Module] = []
        for _ in range(num_reg_layers):
            reg_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
        self.reg_module = nn.ModuleList(reg_layers)
        self.bboxes_delta = nn.Linear(hidden_dim, 4)

    def forward(
        self,
        features: list[Tensor],
        bboxes: Tensor,
        pro_features: Tensor,
        pooler: nn.Module,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            features: FPN feature maps.
            bboxes: (B, N, 4) absolute xyxy proposal boxes.
            pro_features: (1, B*N, hidden_dim) proposal features.
            pooler: ROIPooler module.
            attn_mask: optional (N, N) bool self-attention mask (True =
                blocked); used to isolate denoising queries when DN is on.
        Returns:
            class_logits: (B, N, num_classes)
            pred_bboxes: (B, N, 4) refined absolute xyxy
            obj_features: (1, B*N, hidden_dim)
        """
        from mayaku.structures.boxes import Boxes

        B, N = bboxes.shape[:2]

        # ROI pooling
        proposal_boxes = [Boxes(bboxes[b]) for b in range(B)]
        roi_features = pooler(features, proposal_boxes)  # (B*N, C, P, P)
        roi_features = roi_features.view(B * N, self.hidden_dim, -1).permute(
            2, 0, 1
        )  # (P*P, B*N, d)

        # Self-attention
        pro_features = pro_features.view(B, N, self.hidden_dim).permute(1, 0, 2)  # (N, B, d)
        pro_features2 = self.self_attn(
            pro_features, pro_features, value=pro_features, attn_mask=attn_mask
        )[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # Instance interaction (DynamicConv)
        pro_features = (
            pro_features.view(N, B, self.hidden_dim)
            .permute(1, 0, 2)
            .reshape(1, B * N, self.hidden_dim)
        )
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # FFN
        obj_features2 = self.linear2(self.dropout(F.relu(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        # Predictions
        fc_feature = obj_features.transpose(0, 1).reshape(B * N, -1)
        cls_feature = fc_feature
        reg_feature = fc_feature

        for layer in self.cls_module:
            cls_feature = layer(cls_feature)
        for layer in self.reg_module:
            reg_feature = layer(reg_feature)

        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self._apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        return (
            class_logits.view(B, N, -1),
            pred_bboxes.view(B, N, -1),
            obj_features,
        )

    def _apply_deltas(self, deltas: Tensor, boxes: Tensor) -> Tensor:
        """Standard Faster R-CNN style delta decoding on absolute xyxy boxes."""
        # Force float32: exp(dw) * width overflows fp16 on COCO-scale images
        # (e.g. exp(5)*1333 > 65504), producing Inf → NaN in the next stage's
        # ctr_x = -Inf + 0.5*Inf, which hangs the ROI-align CUDA kernel.
        deltas = deltas.float()
        boxes = boxes.float()

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # Build via stack+reshape, not strided slice-assignment: the latter
        # (``pred_boxes[:, 0::4] = ...``) traces to ONNX ``ScatterND`` which
        # ORT/TensorRT mis-evaluate (bug.md Bug 1). reshape_as is identical and
        # export-clean on every backend.
        pred_boxes = torch.stack(
            [
                pred_ctr_x - 0.5 * pred_w,
                pred_ctr_y - 0.5 * pred_h,
                pred_ctr_x + 0.5 * pred_w,
                pred_ctr_y + 0.5 * pred_h,
            ],
            dim=2,
        ).reshape_as(deltas)
        return pred_boxes
