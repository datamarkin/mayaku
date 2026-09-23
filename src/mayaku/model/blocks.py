"""Building blocks, and the one fusion mechanism that turns them into the deploy graph.

Two graphs, one set of weights:

  train mode   QARepVGG multi-branch 3x3, BatchNorm everywhere.
  deploy mode  one biased 3x3 per branch group, no norm, five op types.

A module that owns weights and can collapse them derives from `Fusible` and
returns its replacement from `fuse()`; every other module is a container and
is simply walked through. Nothing duplicates a container's `forward()`, so an
edit to a block reaches the deploy graph by construction.

Deploy op set, and nothing else ever appears:

    Conv    1x1 and 3x3, stride 1 or 2, bias always
    Relu
    Add     same shape
    Resize  2x nearest
    MaxPool 3x3 stride 1
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_conv_bn_weights

# The stem takes stride 4 and each of the three stages takes another factor
# of two, so these are structural, not configurable. Everything that needs to
# know a stride imports this rather than repeating it.
STRIDES = (8, 16, 32)

# Every input canvas side is a multiple of the coarsest stride: the stride-32
# level needs it, and so does the aux branch's exact 4x upsample from stride
# 32 onto stride 8.
CANVAS_ALIGN = STRIDES[-1]


def as_canvas(canvas):
    """A side or an (H, W) pair -> (H, W), both multiples of CANVAS_ALIGN."""
    hw = (canvas, canvas) if isinstance(canvas, int) else tuple(canvas)
    assert len(hw) == 2 and all(d > 0 and d % CANVAS_ALIGN == 0 for d in hw), \
        "canvas %s: both sides must be multiples of %d" % (hw, CANVAS_ALIGN)
    return hw


def per_level(x, nl):
    """One width broadcast to every level, or one per level already, as a
    tuple of length `nl`."""
    t = (x,) * nl if isinstance(x, int) else tuple(x)
    assert len(t) == nl, t
    return t


def up2x(x):
    """The contract's one Resize: 2x nearest-neighbour upsample."""
    return F.interpolate(x, scale_factor=2, mode="nearest")


# --------------------------------------------------------------------------
# fusion primitives
# --------------------------------------------------------------------------


class Fusible(nn.Module):
    """Owns weights that collapse at export. `fuse()` returns the module that
    replaces it. Anything not deriving from this is a container and the walk
    in `fuse_tree` recurses through it untouched."""

    def fuse(self):
        raise NotImplementedError


def fuse_tree(module):
    """Replace every Fusible leaf beneath `module`, in place.

    Idempotent: a fused subtree contains no Fusible, so a second call is a
    no-op. That is why there is no deployed flag anywhere.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, Fusible):
            setattr(module, name, child.fuse())
        else:
            fuse_tree(child)
    return module


def pad_1x1_to_3x3(w):
    return F.pad(w, [1, 1, 1, 1])


def identity_3x3(channels, device, dtype):
    """The 3x3 kernel that reproduces the identity: a 1 on the centre tap."""
    eye = torch.eye(channels, device=device, dtype=dtype)
    return pad_1x1_to_3x3(eye.view(channels, channels, 1, 1))


def conv_from(weight, bias, ref):
    """`ref` with these weights and a bias. A copy, as torch's own conv-BN
    fusion makes, so whatever `ref` carries survives fusion -- a
    quantization-aware conv keeps its observer: the fused kernel reads the
    same input, so it keeps the same int8 range."""
    c = copy.deepcopy(ref)
    c.weight = nn.Parameter(weight.detach().clone())
    c.bias = nn.Parameter(bias.detach().clone())
    return c


def fuse_bn_chain(seq):
    """Collapse every conv-BN pair in a Sequential, keeping everything else."""
    out, i = [], 0
    mods = list(seq)
    while i < len(mods):
        if (i + 1 < len(mods) and isinstance(mods[i], nn.Conv2d)
                and isinstance(mods[i + 1], nn.BatchNorm2d)):
            out.append(fuse_conv_bn_eval(mods[i], mods[i + 1]))
            i += 2
        else:
            out.append(mods[i])
            i += 1
    return nn.Sequential(*out)


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------


class ConvBN(Fusible):
    """conv -> BN, folding to conv+bias at deploy. No activation."""

    def __init__(self, cin, cout, k=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))

    def fuse(self):
        return fuse_conv_bn_eval(self.conv, self.bn)


class ConvBNReLU(ConvBN):
    def forward(self, x):
        return F.relu(super().forward(x))

    def fuse(self):
        return nn.Sequential(super().fuse(), nn.ReLU())


class RepConv3x3(Fusible):
    """QARepVGG block (Chu et al. 2022, arXiv 2212.01593). Trains as three
    branches, deploys as one biased 3x3.

    The branch layout is QARepVGG's, not plain RepVGG's: BN stays on the 3x3
    branch, the 1x1 branch is a bare convolution, the identity branch is bare,
    and a single BatchNorm sits after the addition. Plain RepVGG fusion
    collapses under per-tensor int8 because all three branches pile onto the
    3x3 centre tap; this layout keeps the fused kernel quantization-friendly.
    """

    def __init__(self, cin, cout, s=1, act=True):
        super().__init__()
        self.act = act
        self.dense = ConvBN(cin, cout, 3, s)
        self.one = nn.Conv2d(cin, cout, 1, s, 0, bias=False)
        self.identity = (cin == cout and s == 1)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        y = self.dense(x) + self.one(x)
        if self.identity:
            y = y + x
        y = self.bn(y)
        return F.relu(y) if self.act else y

    def fuse(self):
        conv, bn = self.dense.conv, self.dense.bn
        w, b = fuse_conv_bn_weights(conv.weight, conv.bias, bn.running_mean,
                                    bn.running_var, bn.eps, bn.weight, bn.bias)
        w = w + pad_1x1_to_3x3(self.one.weight)
        if self.identity:
            w = w + identity_3x3(conv.out_channels, w.device, w.dtype)
        # the post-addition BN folds into the summed kernel
        w, b = fuse_conv_bn_weights(w, b, self.bn.running_mean,
                                    self.bn.running_var, self.bn.eps,
                                    self.bn.weight, self.bn.bias)
        fused = conv_from(w, b, conv)
        return nn.Sequential(fused, nn.ReLU()) if self.act else fused


class Bottleneck(nn.Module):
    """1x1 reduce -> 3x3 -> 1x1 expand -> add.

    1.0625*C^2 MACs per position against a two-3x3 basic block's 18*C^2, and
    within a few percent of a depthwise-separable block, with every op in the
    contract.

    The expand BN starts at zero (Goyal et al. 2017; SkipInit; ReZero), so
    the block starts as the identity, relu(x + 0) = x, and a deep residual
    stack is well-conditioned from step one. The gain sits in the no-decay
    group and grows freely; BN folds at export, so this is free at deploy.
    """

    def __init__(self, c, expansion=4):
        super().__init__()
        mid = max(8, c // expansion)
        self.reduce = ConvBNReLU(c, mid, 1)
        self.spatial = RepConv3x3(mid, mid, 1, act=True)
        self.expand = ConvBN(mid, c, 1)
        nn.init.zeros_(self.expand.bn.weight)

    def forward(self, x):
        return F.relu(x + self.expand(self.spatial(self.reduce(x))))


class Stem(nn.Module):
    """Three 3x3 convolutions to stride 4, kept narrow.

    A three-3x3 stem (ResNet-C, He et al. 2019) beats a single large-kernel
    one, and stride-4 width is the most expensive width in the network, so:
    three convs, few channels.
    """

    def __init__(self, cout):
        super().__init__()
        mid = max(8, cout // 2)
        self.c1 = RepConv3x3(3, mid, 2)
        self.c2 = RepConv3x3(mid, mid, 1)
        self.c3 = RepConv3x3(mid, cout, 2)

    def forward(self, x):
        return self.c3(self.c2(self.c1(x)))


class SPPFAdd(nn.Module):
    """Spatial pyramid pooling with the concatenation removed. Exact, not an
    approximation.

    Two facts compose. Max is idempotent and associative, so N chained
    MaxPool(3, s1, p1) is exactly one MaxPool(2N+1, s1); p2 == k5, p4 == k9,
    p6 == k13. And a 1x1 convolution over a channel concat is
    block-separable: W[x;y1;y2;y3] + b == Wa x + Wb y1 + Wc y2 + Wd y3 + b,
    identical MACs, identical parameters, identical output.

    Six 3x3 maxpools cost 54 comparisons against three 5x5 pools at 75, so
    this is also cheaper than the concat form.
    """

    def __init__(self, c):
        super().__init__()
        mid = c // 2
        self.pre = ConvBNReLU(c, mid, 1)
        self.pool = nn.MaxPool2d(3, 1, 1)
        self.proj = nn.ModuleList(ConvBN(mid, c, 1) for _ in range(4))

    def forward(self, x):
        t = self.pre(x)
        p2 = self.pool(self.pool(t))
        p4 = self.pool(self.pool(p2))
        p6 = self.pool(self.pool(p4))
        y = self.proj[0](t)
        for proj, scale in zip(self.proj[1:], (p2, p4, p6), strict=True):
            y = y + proj(scale)
        return F.relu(y)


class DenseOut(nn.Module):
    """Dense aggregation of a level's fused tensor, in five ops.

    The CSP / ELAN idea (CSPNet, arXiv 1911.11929; GELAN, arXiv 2402.13616):
    a half-width chain of two 3x3s whose intermediates are all kept and mixed
    by one 1x1. Concat followed by 1x1 is exactly the sum of per-input 1x1s,
    so the block is written with adds: y = ReLU(x + P1(m1) + P2(m2)) with
    m1 = 3x3(reduce(x)), m2 = 3x3(m1). Same MACs as the concat form. The two
    projections start at zero, so the block starts as the identity.
    """

    def __init__(self, c):
        super().__init__()
        h = max(8, c // 2)
        self.reduce = ConvBNReLU(c, h, 1)
        self.s1 = RepConv3x3(h, h, 1, act=True)
        self.s2 = RepConv3x3(h, h, 1, act=True)
        self.p1 = ConvBN(h, c, 1)
        self.p2 = ConvBN(h, c, 1)
        nn.init.zeros_(self.p1.bn.weight)
        nn.init.zeros_(self.p2.bn.weight)

    def forward(self, x):
        m1 = self.s1(self.reduce(x))
        m2 = self.s2(m1)
        return F.relu(x + self.p1(m1) + self.p2(m2))
