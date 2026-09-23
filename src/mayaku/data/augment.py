"""Augmentation: the strength presets, random-parameter sampling, and the
deterministic image operations those parameters drive.

Sampling and applying are separate so an operation can be reproduced exactly
from its parameters. Augmentation strength scales with capacity, and mosaic
run to the last epoch is worse than never using it: the gain comes from
switching it off for a clean final stage (`CLEAN_AUG`), as RTMDet's second
stage does.
"""

import dataclasses

import cv2
import numpy as np

from mayaku.data.geometry import PAD


@dataclasses.dataclass(frozen=True)
class Augment:
    """One augmentation strength. Frozen; vary one with
    `dataclasses.replace(DEFAULT_AUG, mixup=0.15)`."""

    mosaic: float = 1.0          # probability of a four-image mosaic
    mixup: float = 0.0           # probability of blending in a second image
    scale: tuple = (0.5, 1.5)    # random scale range
    translate: float = 0.1       # random shift, as a fraction of the canvas
    hflip: float = 0.5           # probability of a horizontal flip
    hsv: tuple = (0.015, 0.7, 0.4)  # hue, saturation, value gains

    def __post_init__(self):
        assert 0.0 <= self.mosaic <= 1.0 and 0.0 <= self.mixup <= 1.0
        assert self.scale[0] <= self.scale[1] and self.scale[0] > 0


DEFAULT_AUG = Augment()
# The clean final stage keeps photometric jitter and the flip and drops every
# geometric transform.
CLEAN_AUG = Augment(mosaic=0.0, scale=(1.0, 1.0), translate=0.0)


def sample_affine(rng, aug, canvas, src_hw):
    """A random scale-and-translate matrix that maps a `src_hw` image onto
    the (H, W) canvas, centred up to the translate jitter. Draws scale, then
    x shift, then y shift."""
    ch, cw = canvas
    s = rng.uniform(*aug.scale)
    t = aug.translate
    m = np.eye(3, dtype=np.float32)
    m[0, 0] = m[1, 1] = s
    m[0, 2] = (0.5 + rng.uniform(-t, t)) * cw - s * src_hw[1] / 2
    m[1, 2] = (0.5 + rng.uniform(-t, t)) * ch - s * src_hw[0] / 2
    return m


def warp_image(img, m, canvas):
    """`img` under the affine `m`, cropped to the (H, W) canvas; the identity
    matrix onto an equal-sized canvas returns `img` untouched."""
    ch, cw = canvas
    if img.shape[:2] == (ch, cw) and (m[:2] == np.float32([[1, 0, 0], [0, 1, 0]])).all():
        return img
    return cv2.warpAffine(img, m[:2], (cw, ch), borderValue=(PAD, PAD, PAD))


def sample_mosaic_centre(rng, canvas):
    """The seam point of a four-image mosaic on a (2H, 2W) canvas, jittered
    so the seams do not always fall on the same pixels. Draws x, then y."""
    ch, cw = canvas
    return int(rng.uniform(0.5 * cw, 1.5 * cw)), int(rng.uniform(0.5 * ch, 1.5 * ch))


def mosaic_tile(k, centre, size, canvas):
    """Where quadrant k (top-left, top-right, bottom-left, bottom-right) of a
    mosaic lands on the (2H, 2W) canvas and which part of a `size` (w, h)
    image fills it: ((xa, xb, ya, yb), (sx, sy)). The x placement depends only
    on the column, the y only on the row; the source offset crops whatever
    falls outside the canvas."""
    ch, cw = canvas
    cx, cy = centre
    nw, nh = size
    left, top = k in (0, 2), k in (0, 1)
    xa, xb = (max(cx - nw, 0), cx) if left else (cx, min(cx + nw, 2 * cw))
    ya, yb = (max(cy - nh, 0), cy) if top else (cy, min(cy + nh, 2 * ch))
    sx = nw - (xb - xa) if left else 0
    sy = nh - (yb - ya) if top else 0
    return (xa, xb, ya, yb), (sx, sy)


def sample_hsv(rng, gains):
    """Hue, saturation and value offsets within `gains`, or None when the
    gains are all zero (no draw is made then)."""
    if not any(gains):
        return None
    return tuple(rng.uniform(-1, 1) * g for g in gains)


def apply_hsv(img, delta):
    """Photometric jitter on a BGR image: rotate hue, scale saturation and
    value by `delta` (dh, ds, dv).

    Hue is a fraction of the circle (OpenCV's 8-bit hue is half-degrees,
    [0, 180)) and rotates modulo 180 rather than scaling, which would barely
    move low hues and swing high ones.
    """
    if delta is None:
        return img
    dh, ds, dv = delta
    byte = np.arange(256, dtype=np.float64)[:, None]
    table = np.concatenate((np.mod(byte + 180 * dh, 180),
                            np.clip(byte * (1 + ds), 0, 255),
                            np.clip(byte * (1 + dv), 0, 255)), 1)
    jittered = cv2.LUT(cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
                       table.astype(np.uint8).reshape(256, 1, 3))
    return cv2.cvtColor(jittered, cv2.COLOR_HSV2BGR)


def mixup(img, other, w):
    """mixup (Zhang et al. 2017, arXiv 1710.09412): `w * img + (1 - w) * other`."""
    return (img * w + other * (1 - w)).astype(np.uint8)
