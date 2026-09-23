"""Backbone: a narrow stride-4 stem, then three bottleneck stages at strides 8 / 16 / 32."""

import torch.nn as nn

from mayaku.model.blocks import Bottleneck, RepConv3x3, SPPFAdd, Stem


class Backbone(nn.Module):
    """Returns the three stage outputs, fine to coarse; the coarsest passes
    through the concat-free spatial pyramid pooling."""

    def __init__(self, cfg):
        super().__init__()
        self.stem = Stem(cfg.stem)
        chs = (cfg.stem, *tuple(cfg.width))
        self.down, self.stages = nn.ModuleList(), nn.ModuleList()
        for i, n in enumerate(cfg.depth):
            self.down.append(RepConv3x3(chs[i], chs[i + 1], 2))
            self.stages.append(nn.Sequential(
                *[Bottleneck(chs[i + 1]) for _ in range(n)]))
        self.sppf = SPPFAdd(cfg.width[-1])

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for down, stage in zip(self.down, self.stages, strict=True):
            x = stage(down(x))
            outs.append(x)
        outs[-1] = self.sppf(outs[-1])
        return outs
