"""The detector: backbone, neck, head, and the optional mask / keypoint branch."""

import copy

import torch.nn as nn

from mayaku.model.aux import AuxBranch
from mayaku.model.backbone import Backbone
from mayaku.model.blocks import STRIDES, as_canvas, fuse_tree, per_level
from mayaku.model.head import Head
from mayaku.model.neck import FPG
from mayaku.model.tiers import TIERS


class Detector(nn.Module):
    """`forward` returns a flat list of raw maps in `output_names` order: the
    head's six tensors first, then the auxiliary branch's. Flat because
    exporters name outputs positionally; `split` gives the named view.

    The graph runs on any canvas whose sides are multiples of 32; `canvas`
    only sets the classifier's initial object prior (`Head.bias_init`)."""

    def __init__(self, cfg, nc=80, canvas=640):
        super().__init__()
        self.cfg, self.nc = cfg, nc
        self.backbone = Backbone(cfg)
        self.neck = FPG(cfg.width, cfg.neck, cfg.paths)
        self.head = Head(cfg.neck, nc, as_canvas(canvas), reg_max=cfg.reg_max,
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

    def for_deploy(self):
        """A fused copy that runs the deployed fp32 graph: the one this model
        exports as, which a runtime quantizes from the observed ranges when
        it was trained quantization-aware. The original is left as it is."""
        from mayaku.model.quant import strip_fake_quant

        return strip_fake_quant(copy.deepcopy(self).fuse())

    def deploy_spec(self):
        """What a runtime needs to know about this network's outputs, as
        plain JSON: their names in order, the strides and DFL bins the box
        decode uses, the mask and keypoint decode constants when those heads
        exist, and whether the weights were trained quantization-aware."""
        from mayaku.model.aux import KERNEL_LAYOUT, MASK_CH, MASK_STRIDE
        from mayaku.model.kpt import sigma_values
        from mayaku.model.mask import COORD_SCALE
        from mayaku.model.quant import is_qat

        return {
            "outputs": self.out_names,
            "strides": list(STRIDES),
            "reg_max": self.cfg.reg_max,
            "qat": is_qat(self),
            "mask": {"stride": MASK_STRIDE, "channels": MASK_CH, "coord_scale": COORD_SCALE,
                     "kernel_layout": [list(layer) for layer in KERNEL_LAYOUT]} if self.cfg.seg else None,
            "keypoints": {"num": self.cfg.kpt, "sigmas": list(sigma_values(self.cfg.kpt))}
                         if self.cfg.kpt else None,
        }


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


# Which checkpoint keys may be absent on one side. The auxiliary branch is
# detachable, so its keys (`aux.`) may exist on either side alone; a model
# trained quantization-aware carries activation observers (`.act_fq.`) a plain
# one lacks; and the classifier (`head.cls.`) depends on the class count.
AUX_KEY, QAT_KEY, CLASS_KEY = "aux.", ".act_fq.", "head.cls."


def _load(model, state, optional, reinit_classes):
    """Load `state` non-strictly, then refuse any mismatch `optional(key)`
    does not allow; with `reinit_classes`, classifier weights of another shape
    are left at their initialisation instead of failing."""
    own = model.state_dict()
    shaped = [k for k, v in state.items() if k in own and v.shape != own[k].shape]
    reinit = [k for k in shaped if reinit_classes and k.startswith(CLASS_KEY)]
    wrong = [k for k in shaped if k not in reinit]
    assert not wrong, "state dict does not fit this model: %s" % wrong[:8]
    r = model.load_state_dict({k: v for k, v in state.items() if k not in reinit}, strict=False)
    stray = [k for k in list(r.missing_keys) + list(r.unexpected_keys)
             if k not in reinit and not optional(k)]
    assert not stray, "state dict does not fit this model: %s" % stray[:8]
    return {"reinitialised": reinit, "missing": list(r.missing_keys),
            "unexpected": list(r.unexpected_keys)}


def load_pretrained(model, state):
    """Warm-start `model` from pretrained weights of the same tier: the
    classifier is re-initialised when the class count changes, and the
    auxiliary branch and QAT observers may be present on either side alone.
    Anything else that fails to match (another tier) raises. Returns
    {"reinitialised", "missing", "unexpected"} key lists."""
    return _load(model, state, lambda k: k.startswith(AUX_KEY) or QAT_KEY in k,
                 reinit_classes=True)


def load_weights(model, state, strict_aux=False):
    """Load a checkpoint of this model across the auxiliary-branch boundary:
    a detector trained with masks / keypoints loads into a plain detector and
    back; only `aux.*` keys may be missing or unexpected (none with
    `strict_aux`), anything else raises."""
    return _load(model, state, lambda k: not strict_aux and k.startswith(AUX_KEY),
                 reinit_classes=False)
