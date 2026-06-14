"""Dynamic instance mask head for UniQuery (QueryInst pattern).

Generates per-instance convolution kernels from obj_features and applies
them to ROI-pooled FPN features. Produces class-agnostic masks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["UniQueryDynamicMaskHead"]


def _group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with the largest group count (<= max_groups) that divides
    ``channels`` — so the head works for any ``conv_dim``."""
    groups = max_groups
    while channels % groups:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class UniQueryDynamicMaskHead(nn.Module):
    """Dynamic mask head: spatial conv stack + per-instance kernel from obj_features.

    Architecture:
        1. ROI features (R, C, P, P) → num_conv × Conv3x3-GroupNorm-ReLU → (R, conv_dim, P, P)
        2. obj_features (R, hidden_dim) → Linear → kernel weights (R, conv_dim) + bias (R, 1)
        3. Per-instance 1×1 dot product, scaled by 1/sqrt(conv_dim) → (R, 1, P, P)
        4. ConvTranspose2d(k=2, s=2) → (R, 1, 2P, 2P) mask logits

    GroupNorm (step 1) + the 1/sqrt(conv_dim) dot-product scale (step 3) keep
    the mask logits at unit scale at init. Without them the conv_dim-channel dot
    product blows up (BCE ~90 instead of ~0.69): the box head's DynamicConv
    already LayerNorm's its pooled features, and this head was the lone dynamic
    head missing equivalent normalisation + fan-in scaling. The runaway loss
    otherwise dominates the shared-query gradient and collapses both the mask
    and detection branches. Cf. CondInst (normalised mask branch) and QueryInst
    (xavier mask predictor) for the same init discipline.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        conv_dim: int = 256,
        num_conv: int = 4,
        mask_resolution: int = 28,
        pooler_resolution: int = 14,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_dim = conv_dim
        self.mask_resolution = mask_resolution
        self.pooler_resolution = pooler_resolution

        # Dot-product scale: the per-instance kernel is dotted against the
        # conv_dim-channel feature at each location. Without 1/sqrt(conv_dim)
        # the logits scale as sqrt(conv_dim) and BCE explodes at init.
        self._kernel_scale = conv_dim**-0.5

        self.spatial_convs = nn.ModuleList()
        self.spatial_norms = nn.ModuleList()
        ch = hidden_dim
        for _ in range(num_conv):
            self.spatial_convs.append(nn.Conv2d(ch, conv_dim, 3, padding=1, bias=False))
            # GroupNorm keeps the conv features unit-scale regardless of the
            # (unnormalised) FPN feature magnitude feeding the pooler — so the
            # mask logits don't depend on backbone/FPN activation scale.
            self.spatial_norms.append(_group_norm(conv_dim))
            ch = conv_dim

        self.kernel_fc = nn.Linear(hidden_dim, conv_dim + 1)

        self.upsample = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)

        self._init_weights()

    def _init_weights(self) -> None:
        for conv in self.spatial_convs:
            assert isinstance(conv, nn.Conv2d)
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.xavier_uniform_(self.kernel_fc.weight)
        assert self.kernel_fc.bias is not None
        nn.init.constant_(self.kernel_fc.bias, 0)
        nn.init.kaiming_normal_(self.upsample.weight, mode="fan_out", nonlinearity="relu")
        assert self.upsample.bias is not None
        nn.init.constant_(self.upsample.bias, 0)

    def forward(self, roi_features: Tensor, obj_features: Tensor) -> Tensor:
        """
        Args:
            roi_features: (R, C, P, P) from ROIPooler at pooler_resolution.
            obj_features: (R, hidden_dim) per-instance embeddings from UniQueryStage.
        Returns:
            mask_logits: (R, 1, mask_resolution, mask_resolution) class-agnostic.
        """
        x = roi_features
        for conv, norm in zip(self.spatial_convs, self.spatial_norms, strict=True):
            x = F.relu(norm(conv(x)))

        kernel_params = self.kernel_fc(obj_features.float())
        # Fold the fan-in scale into the kernel (R, conv_dim) rather than the
        # larger einsum output — same result, fewer elements multiplied.
        weights = kernel_params[:, : self.conv_dim] * self._kernel_scale
        biases = kernel_params[:, self.conv_dim :]

        mask = torch.einsum("rchw,rc->rhw", x.float(), weights) + biases[..., None]
        mask = mask.unsqueeze(1)

        out: Tensor = self.upsample(mask)
        return out
