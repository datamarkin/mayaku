"""The detector: backbone, neck, head, and the optional mask / keypoint branch."""

import torch.nn as nn

from mayaku.model.aux import AuxBranch
from mayaku.model.backbone import Backbone
from mayaku.model.blocks import STRIDES, fuse_tree, per_level
from mayaku.model.head import Head
from mayaku.model.neck import FPG
from mayaku.model.tiers import TIERS


class Detector(nn.Module):
    """`forward` returns a flat list of raw maps in `output_names` order: the
    head's six tensors first, then the auxiliary branch's. Flat because
    exporters name outputs positionally; `split` gives the named view."""

    def __init__(self, cfg, nc=80):
        super().__init__()
        self.cfg, self.nc = cfg, nc
        self.backbone = Backbone(cfg)
        self.neck = FPG(cfg.width, cfg.neck, cfg.paths)
        self.head = Head(cfg.neck, nc, reg_max=cfg.reg_max,
                         n_conv=cfg.head_conv, width=cfg.head_width)
        self.aux = None
        if cfg.seg or cfg.kpt:
            self.aux = AuxBranch(per_level(cfg.neck, len(STRIDES)), self.head.widths,
                                 seg=cfg.seg, kpt=cfg.kpt,
                                 arm=cfg.aux_arm, width=cfg.aux_width)

    def split(self, preds):
        """`forward`'s flat list -> named groups."""
        return split_outputs(preds, seg=self.cfg.seg, kpt=self.cfg.kpt)

    @property
    def out_names(self):
        return output_names(seg=self.cfg.seg, kpt=self.cfg.kpt)

    def readouts(self):
        """Output projections, as opposed to hidden layers (the optimizer
        treats them differently)."""
        r = list(self.head.cls) + list(self.head.box)
        return r + (self.aux.readouts() if self.aux else [])

    def forward(self, x):
        feats = self.neck(self.backbone(x))
        cb = self.head.towers(feats)
        out = self.head.readout(cb)
        if self.aux is not None:
            out = out + self.aux(feats, [b for _, b in cb])
        return out

    def fuse(self):
        """Rewrite the train graph into the deploy graph, in place."""
        return fuse_tree(self.eval())


def build(tier="n", nc=80):
    return Detector(TIERS[tier], nc)


def output_names(seg=False, kpt=0):
    """The model's outputs, in order: cls/box interleaved per level, then, if
    present, mask + one kernel map per level, then heat + one keypoint map per
    level. The single owner of that layout; `split_outputs` groups by it."""
    nl = range(len(STRIDES))
    names = ["%s%d" % (kind, i) for i in nl for kind in ("cls", "box")]
    if seg:
        names += ["mask"] + ["ker%d" % i for i in nl]
    if kpt:
        names += ["heat"] + ["kpt%d" % i for i in nl]
    return names


def split_outputs(preds, seg=False, kpt=0):
    """The flat output list -> named groups: `head` is the detection tensors,
    `mask` / `heat` single maps, `ker` / `kpt` lists of one map per level."""
    named = dict(zip(output_names(seg, kpt), preds, strict=True))
    g = {"head": [named[n] for n in named if n[:3] in ("cls", "box")]}
    if seg:
        g["mask"], g["ker"] = named["mask"], [named[n] for n in named if n.startswith("ker")]
    if kpt:
        g["heat"], g["kpt"] = named["heat"], [named[n] for n in named if n.startswith("kpt")]
    return g


def load_weights(model, state, strict_aux=False):
    """Load a checkpoint across the auxiliary-branch boundary.

    A detector trained with masks / keypoints loads into a plain detector and
    a plain detector's weights load into one with the branch; only `aux.*`
    keys may be missing or unexpected, anything else is a real mismatch and
    raises. `strict_aux` forbids even that."""
    r = model.load_state_dict(state, strict=False)
    stray = [k for k in list(r.missing_keys) + list(r.unexpected_keys)
             if strict_aux or not k.startswith("aux.")]
    assert not stray, "state dict mismatch outside aux.*: %s" % stray[:8]
    return r
