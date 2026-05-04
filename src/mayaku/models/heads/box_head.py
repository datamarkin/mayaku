"""Box-head feature extractor (`FastRCNNConvFCHead`).

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §2.5 (`box_head.py:25-110`):

* ``num_conv`` 3x3 conv-ReLU stages (default ``0`` for the in-scope
  configs) operating on the pooled ``(R, C, P, P)`` features.
* Flatten to ``(R, C * P * P)``.
* ``num_fc`` linear-ReLU stages of size ``fc_dim`` (default
  ``num_fc=2``, ``fc_dim=1024``).

Output is ``(R, fc_dim)`` if ``num_fc > 0`` else ``(R, C * P * P)``.

The head is the cheap part of the ROI pipeline; the heavy lifting is in
the ROI Align that produces the input. We therefore keep the conv path
optional but supported (some configs do use it).
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from mayaku.config.schemas import ROIBoxHeadConfig
from mayaku.models.backbones._base import ShapeSpec

__all__ = ["FastRCNNConvFCHead", "build_box_head"]


class FastRCNNConvFCHead(nn.Module):
    """Conv stack + FC stack on top of pooled RoI features."""

    def __init__(
        self,
        input_shape: ShapeSpec,
        pooler_resolution: int,
        num_conv: int = 0,
        conv_dim: int = 256,
        num_fc: int = 2,
        fc_dim: int = 1024,
    ) -> None:
        super().__init__()
        if num_conv == 0 and num_fc == 0:
            raise ValueError("FastRCNNConvFCHead requires num_conv > 0 or num_fc > 0")

        self.convs = nn.ModuleList()
        ch = input_shape.channels
        for _ in range(num_conv):
            self.convs.append(nn.Conv2d(ch, conv_dim, kernel_size=3, padding=1))
            ch = conv_dim
        self.fcs = nn.ModuleList()
        flat_dim = ch * pooler_resolution * pooler_resolution
        for _ in range(num_fc):
            self.fcs.append(nn.Linear(flat_dim, fc_dim))
            flat_dim = fc_dim
        self._output_dim = flat_dim

        # Init: convs are MSRA / Kaiming, fcs are Xavier (Glorot) to match
        # Detectron2 (`box_head.py:75-100`: `c2_msra_fill` for convs,
        # `c2_xavier_fill` for fcs). Earlier versions of this code used
        # `kaiming_uniform_(a=1)` on the fcs, which is ~1.4x wider than
        # Xavier on a 1024->1024 layer (std ≈ 0.061 vs ≈ 0.043) and made
        # ROI head outputs ~1.4x larger initially — measurably inflating
        # early-iter gradient norms and triggering the trainer's gradient
        # clip more aggressively than D2's setup. See compare.md.
        for m in self.convs:
            assert isinstance(m, nn.Conv2d)
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            assert m.bias is not None
            nn.init.constant_(m.bias, 0.0)
        for m in self.fcs:
            assert isinstance(m, nn.Linear)
            nn.init.xavier_uniform_(m.weight)  # matches D2's c2_xavier_fill
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = F.relu(conv(x))
        x = x.flatten(start_dim=1)
        for fc in self.fcs:
            x = F.relu(fc(x))
        return x

    @property
    def output_dim(self) -> int:
        return self._output_dim


def build_box_head(cfg: ROIBoxHeadConfig, input_shape: ShapeSpec) -> FastRCNNConvFCHead:
    """Construct a :class:`FastRCNNConvFCHead` from a typed config."""
    return FastRCNNConvFCHead(
        input_shape=input_shape,
        pooler_resolution=cfg.pooler_resolution,
        num_conv=cfg.num_conv,
        conv_dim=cfg.conv_dim,
        num_fc=cfg.num_fc,
        fc_dim=cfg.fc_dim,
    )
