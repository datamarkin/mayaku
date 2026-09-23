"""Detection head: three scales, a decoupled class / box tower per scale, raw logits out."""

import math

import torch.nn as nn

from mayaku.model.blocks import STRIDES, Fusible, fuse_bn_chain, per_level


class Head(Fusible):
    """Three detection outputs, not one.

    YOLOF's ablation (Chen et al. 2021, arXiv 2103.09460) is unambiguous:
    multi-in multi-out 35.9, multi-in single-out 23.9. Multiple outputs are
    worth about 12 AP and cost nothing the contract objects to.

    Each scale has its own class tower and its own box tower (YOLOX, Ge et
    al. 2021, arXiv 2107.08430: a coupled head converges slower and finishes
    lower). Tower weights are not shared across scales; since BatchNorm is
    per scale, a shared convolution would expand into three distinct ones at
    fusion anyway, so unsharing costs nothing at deploy.

    `width` is the head's own channel count, independent of the neck's; 0
    keeps the neck's width.

    Outputs are raw logits. No sigmoid, no DFL expectation, no top-k: those
    run on the host after the graph.
    """

    def __init__(self, ch, nc, canvas, reg_max=16, n_conv=2, width=0):
        super().__init__()
        nl = len(STRIDES)
        chs = per_level(ch, nl)
        ws = tuple(w or c for w, c in zip(per_level(width, nl), chs))
        self.widths = ws   # tower output width per level; the aux branch reads it

        def tower():
            """`n_conv` 3x3 convolutions with BatchNorm and ReLU, per scale."""
            return nn.ModuleList(
                nn.Sequential(*[m for i in range(n_conv)
                                for m in (nn.Conv2d(chs[lv] if i == 0 else ws[lv], ws[lv],
                                                    3, 1, 1, bias=False),
                                          nn.BatchNorm2d(ws[lv]), nn.ReLU())])
                for lv in range(nl))

        # `tower` is the class tower; the name is kept for checkpoint keys
        self.tower = tower()
        self.box_tower = tower()
        self.cls = nn.ModuleList(nn.Conv2d(ws[lv], nc, 1) for lv in range(nl))
        self.box = nn.ModuleList(nn.Conv2d(ws[lv], 4 * reg_max, 1) for lv in range(nl))
        self.bias_init(nc, canvas)

    def bias_init(self, nc, canvas, objects=5):
        """Start the classifier at the prior probability of an object.

        Without this the first steps are spent pushing several hundred
        thousand negative logits down from 0.5, which makes the
        classification term orders of magnitude larger than the box term. A
        level with (H/stride)(W/stride) cells on the (H, W) canvas expects
        `objects / nc` of them to be positive, so that ratio is the bias.
        RetinaNet's fixed 0.01 prior is the same idea without the per-level
        density term.
        """
        h, w = canvas
        for cls, box, s in zip(self.cls, self.box, STRIDES):
            nn.init.constant_(cls.bias, math.log(objects / nc / ((h / s) * (w / s))))
            nn.init.constant_(box.bias, 1.0)

    def towers(self, xs):
        """Per level, the (class, box) tower outputs. Split from `readout` so
        the auxiliary branch can hang its own 1x1s on the box tower."""
        return [(self.tower[i](x), self.box_tower[i](x)) for i, x in enumerate(xs)]

    def readout(self, cb):
        out = []
        for i, (c, b) in enumerate(cb):
            out.append(self.cls[i](c))
            out.append(self.box[i](b))
        return out

    def forward(self, xs):
        return self.readout(self.towers(xs))

    def fuse(self):
        self.tower = nn.ModuleList(fuse_bn_chain(t) for t in self.tower)
        self.box_tower = nn.ModuleList(fuse_bn_chain(t) for t in self.box_tower)
        return self
