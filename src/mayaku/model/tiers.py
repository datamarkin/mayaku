"""The architecture family: one frozen `Tier` per size, n / s / m / l."""

import dataclasses

from mayaku.model.blocks import STRIDES


@dataclasses.dataclass(frozen=True)
class Tier:
    """One member of the family. Frozen and hashable; vary one with
    `dataclasses.replace(TIERS["n"], seg=True)`."""

    stem: int
    width: tuple                    # backbone stage widths, fine to coarse
    depth: tuple                    # bottlenecks per stage, fine to coarse
    neck: tuple                     # neck width per level, fine to coarse
    paths: int                      # neck pathways
    head_width: int = 0             # head tower width; 0 = the neck's width per level
    head_conv: int = 2              # 3x3 convolutions per head tower
    reg_max: int = 16               # DFL bins per box side
    # Auxiliary dense heads, off by default. Turning them on adds `aux.*`
    # weights only; see `load_weights`.
    seg: bool = False               # instance masks
    kpt: int = 0                    # keypoints per instance (0 = none)
    aux_arm: str = "box_tower"      # per-anchor aux outputs read from: "box_tower" | "tower"
    aux_width: int = 64             # trunk width of the aux branch (and of its own towers)

    def __post_init__(self):
        n = len(STRIDES)
        assert len(self.width) == len(self.depth) == len(self.neck) == n, \
            "one width, depth and neck width per stride in %s" % (STRIDES,)
        assert min(self.width) > 0 and min(self.depth) > 0 and min(self.neck) > 0
        assert self.paths >= 1
        assert self.head_width >= 0 and self.head_conv >= 1
        assert self.aux_arm in ("box_tower", "tower"), self.aux_arm
        assert self.kpt >= 0 and self.aux_width >= 8


# Widths are multiples of 32. Depth and width are pushed toward the coarse
# stages, where parameters are cheap to run, and the bottleneck block makes
# backbone depth cheap enough to spend.
TIERS = {
    "n": Tier(stem=24, width=(64, 160, 320), depth=(3, 5, 7),
              neck=(64, 128, 256), paths=3, head_width=80),
    "s": Tier(stem=32, width=(144, 288, 576), depth=(3, 5, 7),
              neck=(96, 192, 384), paths=3, head_width=96),
    "m": Tier(stem=64, width=(192, 384, 768), depth=(4, 8, 10),
              neck=(192, 384, 768), paths=4, head_width=192),
    "l": Tier(stem=64, width=(256, 512, 1024), depth=(4, 8, 10),
              neck=(256, 512, 1024), paths=5, head_width=256),
}

# Not a product tier: a few tens of thousands of parameters, so a test can
# train a real network end to end in seconds.
TINY = Tier(stem=8, width=(16, 32, 64), depth=(1, 1, 1), neck=(16, 16, 16),
            paths=1, head_conv=1, reg_max=8)

# Tiers trained quantization-aware by default: the small ones, whose targets
# are int8-only accelerators. Larger tiers deploy in fp16.
QAT_TIERS = ("n", "s")
