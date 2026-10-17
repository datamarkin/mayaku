"""Neck: a Feature Pyramid Grid (Chen et al. 2020, arXiv 2004.03580) fused by add only.

Deep and narrow beats shallow and wide: FPG reaches 39.0 AP with 9 pathways
at 128 channels where FPN needs 1 pathway at 256 for 37.0, at the same FLOPs.
Every fusion here is an element-wise add, so there is no concatenation
anywhere in the graph.
"""

import torch.nn as nn
import torch.nn.functional as F

from mayaku.model.blocks import STRIDES, Bottleneck, ConvBN, ConvBNReLU, DenseOut, per_level, up2x
from mayaku.model.quant import ResidualQuant


class FPGStage(nn.Module):
    """One pathway. Levels are ordered fine to coarse: [s8, s16, s32].

        same   1x1 conv, level i -> level i
        up     coarse to fine: 1x1 conv, then 2x nearest. A 1x1 conv commutes
               exactly with nearest upsampling, so running it at the coarse
               resolution is the same function at a quarter of the cost.
        down   fine to coarse: 1x1 stride-2 conv. Carried only by the last
               pathway(s); it is the neck's sole bottom-up flow.
        skip   1x1 conv from two pathways back, same level.

    After fusion every level's sum is processed at its own resolution by a
    bottleneck, so each fused tensor gets same-scale spatial mixing before it
    moves on.

    Nothing about add-only fusion needs the levels to share a width: the
    cross-level convolutions already change resolution, so they change
    channel count in the same op and the add still lands on equal shapes.
    """

    def __init__(self, nl, ch, skip, down):
        super().__init__()
        self.nl = nl
        c = per_level(ch, nl)
        self.same = nn.ModuleList(ConvBN(c[i], c[i], 1) for i in range(nl))
        self.up = nn.ModuleList(ConvBN(c[i + 1], c[i], 1) for i in range(nl - 1))
        self.down = nn.ModuleList(
            ConvBN(c[i], c[i + 1], 1, 2) for i in range(nl - 1)) if down else None
        self.skip = nn.ModuleList(ConvBN(c[i], c[i], 1) for i in range(nl)) if skip else None
        self.proc = nn.ModuleList(Bottleneck(c[i], 2) for i in range(nl))
        # Each level's sum is built as conv + quantized running sum, one conv
        # at a time, so an int8 runtime fuses every add into a convolution
        # (see `ResidualQuant`); the upsampled term is the first running sum.
        self.rq_up = nn.ModuleList(ResidualQuant() for _ in range(nl - 1))
        self.rq_down = nn.ModuleList(ResidualQuant() for _ in range(nl - 1)) if down else None
        self.rq_skip = nn.ModuleList(ResidualQuant() for _ in range(nl)) if skip else None

    def forward(self, xs, prev=None):
        out = []
        for i in range(self.nl):
            y = self.same[i](xs[i])
            if i < self.nl - 1:
                y = y + self.rq_up[i](up2x(self.up[i](xs[i + 1])))
            if self.down is not None and i > 0:
                y = self.down[i - 1](xs[i - 1]) + self.rq_down[i - 1](y)
            if self.skip is not None and prev is not None:
                y = self.skip[i](prev[i]) + self.rq_skip[i](y)
            out.append(F.relu(y))
        return [p(o) for p, o in zip(self.proc, out, strict=True)]


class FPG(nn.Module):
    """Lateral 1x1s, `n_path` pathways, and a dense output block per level.

    Only the last pathway carries the fine-to-coarse connection: FPG prices
    it at nothing once same-level processing exists, and one copy keeps a
    bottom-up flow, which EfficientDet finds worth keeping.
    """

    def __init__(self, cin, ch, n_path):
        super().__init__()
        nl = len(STRIDES)
        c = per_level(ch, nl)
        self.lateral = nn.ModuleList(ConvBNReLU(a, b, 1) for a, b in zip(cin, c, strict=True))
        self.paths = nn.ModuleList(
            FPGStage(nl, c, skip=(i >= 2), down=(i == n_path - 1))
            for i in range(n_path))
        self.out = nn.ModuleList(DenseOut(c[i]) for i in range(nl))

    def forward(self, xs):
        xs = [lat(x) for lat, x in zip(self.lateral, xs, strict=True)]
        history = [xs]
        for path in self.paths:
            prev = history[-2] if len(history) >= 2 else None
            xs = path(xs, prev)
            history.append(xs)
        return [o(x) for o, x in zip(self.out, xs, strict=True)]
