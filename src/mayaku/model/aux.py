"""Auxiliary dense heads: instance masks and keypoints on one shared stride-8 trunk.

Masks follow the dynamic-kernel design of CondInst (Tian et al. 2020, arXiv
2003.05664) and RTMDet-Ins (Lyu et al. 2022, arXiv 2212.07784): the graph emits
a small stride-8 mask feature and, per anchor, the weights of a tiny per-instance
network that the host runs over that feature. Keypoints are regressed per anchor
and refined on the host against a stride-8 heatmap (CenterNet, Zhou et al. 2019,
arXiv 1904.07850). Everything in the graph stays inside the five-op contract.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mayaku.model.blocks import STRIDES, ConvBN, ConvBNReLU, up2x

MASK_CH = 8          # channels of the shared stride-8 mask feature
COORD_CH = 2         # relative-coordinate channels the host prepends
MASK_STRIDE = STRIDES[0]
# Per-instance mask head applied on the host: three 1x1 layers
# (MASK_CH + COORD_CH) -> 8 -> 8 -> 1 with biases, ReLU between. The kernel
# controller emits exactly this many numbers per anchor.
KERNEL_LAYOUT = ((MASK_CH + COORD_CH, 8), (8, 8), (8, 1))
KERNEL_PARAMS = sum(i * o + o for i, o in KERNEL_LAYOUT)   # 169


class AuxBranch(nn.Module):
    """Instance masks and keypoints on one shared stride-8 trunk.

    Trunk: P3 through a 3x3, P4 and P5 through 1x1s then 2x nearest
    upsampling, all added, one ReLU.

      mask   (B, 8, H/8, W/8)      raw mask feature; the host applies each
                                   detection's 169 kernel numbers to it
      ker_l  (B, 169, h_l, w_l)    dynamic-kernel controller per level
      heat   (B, K+2, H/8, W/8)    K keypoint heatmaps + 2 offset channels
      kpt_l  (B, 3K, h_l, w_l)     per-anchor (dx, dy, visibility) x K

    `arm` says where the per-anchor outputs (ker, kpt) read from: the
    detection head's box tower ("box_tower"), or a separate one-conv tower per
    level ("tower").

    A plain container, so `fuse_tree` folds its ConvBNs; the output 1x1s are
    biased convolutions with no norm. The branch is detachable: a checkpoint
    with it loads into a model without it and back, see `load_weights`.
    """

    def __init__(self, cin, head_in, seg=True, kpt=0, arm="box_tower", width=64):
        super().__init__()
        self.seg, self.kpt, self.arm = seg, kpt, arm
        c3, c4, c5 = cin
        self.p3 = ConvBN(c3, width, 3)
        self.p4 = ConvBN(c4, width, 1)
        self.p5 = ConvBN(c5, width, 1)
        # two 1x1s on the same trunk output (a runtime may merge them into
        # one conv, by the same block-separability SPPFAdd relies on)
        self.mask = nn.Conv2d(width, MASK_CH, 1) if seg else None
        self.heat = nn.Conv2d(width, kpt + 2, 1) if kpt else None
        self.tower = None
        if arm == "tower":
            self.tower = nn.ModuleList(ConvBNReLU(c, width, 3) for c in cin)
        pin = [width] * len(cin) if arm == "tower" else list(head_in)
        self.ker = nn.ModuleList(nn.Conv2d(c, KERNEL_PARAMS, 1) for c in pin) if seg else None
        self.kp = nn.ModuleList(nn.Conv2d(c, 3 * kpt, 1) for c in pin) if kpt else None
        for ml in (self.ker, self.kp):
            for m in ml or []:
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        if kpt:
            # heatmap channels start at a low prior, like the classifier
            with torch.no_grad():
                self.heat.bias[:kpt].fill_(math.log(0.01 / 0.99))

    def forward(self, feats, boxes):
        """Outputs in `mayaku.model.detector.output_names` order.

        feats: neck outputs fine to coarse; boxes: per-level box-tower
        outputs (unused when `arm` is "tower")."""
        f = F.relu(self.p3(feats[0]) + up2x(self.p4(feats[1]))
                   + up2x(up2x(self.p5(feats[2]))))
        per = boxes if self.tower is None else [t(x) for t, x in zip(self.tower, feats, strict=True)]
        out = []
        if self.seg:
            out += [self.mask(f)] + [k(x) for k, x in zip(self.ker, per, strict=True)]
        if self.kpt:
            out += [self.heat(f)] + [k(x) for k, x in zip(self.kp, per, strict=True)]
        return out

    def readouts(self):
        return ([m for m in (self.mask, self.heat) if m is not None]
                + list(self.ker or []) + list(self.kp or []))
