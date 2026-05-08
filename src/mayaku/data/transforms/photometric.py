"""Photometric augmentations — pure colour-space ops, no geometric effect.

These transforms only modify pixel values; they leave bboxes / masks /
keypoints unchanged. The :meth:`apply_coords` implementation is the
identity, which means :meth:`apply_box`, :meth:`apply_polygons`, and
:meth:`apply_keypoints` (all derived from ``apply_coords`` in the base
class) are no-ops too. That's the whole point of separating photometric
from geometric augmentations.

Targeted ranges for COCO-class detection:

* **Brightness ±0.4** — multiplicative factor in ``[0.6, 1.4]``
* **Contrast ±0.4** — multiplicative deviation from the per-channel mean
* **Saturation ±0.4** — interpolation between greyscale and original
* **Hue shift ±0.015** — small; too much breaks colour-dependent classes
  (e.g. traffic-light colour, sports-jersey colour)

Default colour jitter knobs and the
RandAugment-style "lighter" detection recipe. The values are conservative
relative to what classification training uses — detection is more
sensitive to colour than classification because some classes are
colour-defined.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from mayaku.data.transforms.augmentation import Augmentation, _NoOpTransform
from mayaku.data.transforms.base import Transform

__all__ = [
    "BrightnessTransform",
    "ContrastTransform",
    "HueShiftTransform",
    "RandomColorJitter",
    "SaturationTransform",
]

_F32 = npt.NDArray[np.float32]


class _PhotometricTransform(Transform):
    """Base class for pure-photometric transforms (no geometric effect)."""

    def apply_coords(self, coords: _F32) -> _F32:
        # Identity — pixel-value changes don't move spatial coordinates.
        return coords


class BrightnessTransform(_PhotometricTransform):
    """Multiply every pixel by ``factor`` (clipped to ``[0, 255]``).

    ``factor=1.0`` is a no-op. Typical training range: ``[0.6, 1.4]``.
    """

    def __init__(self, factor: float) -> None:
        self.factor = float(factor)

    def apply_image(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if image.ndim != 3 or image.shape[2] != 3:
            return image
        out = image.astype(np.float32, copy=True) * self.factor
        result: npt.NDArray[Any] = np.clip(out, 0.0, 255.0).astype(image.dtype)
        return result


class ContrastTransform(_PhotometricTransform):
    """Interpolate between the per-channel mean and the original by ``factor``.

    ``factor=0.0`` collapses to a uniform grey at the per-channel mean;
    ``factor=1.0`` is a no-op. Typical training range: ``[0.6, 1.4]``.
    """

    def __init__(self, factor: float) -> None:
        self.factor = float(factor)

    def apply_image(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if image.ndim != 3 or image.shape[2] != 3:
            return image
        img32 = image.astype(np.float32, copy=False)
        # Per-channel mean (one scalar per channel).
        mean = img32.reshape(-1, 3).mean(axis=0)
        out: npt.NDArray[Any] = (img32 - mean) * self.factor + mean
        result: npt.NDArray[Any] = np.clip(out, 0.0, 255.0).astype(image.dtype)
        return result


class SaturationTransform(_PhotometricTransform):
    """Interpolate between greyscale and the original by ``factor``.

    Uses Rec. 601 luma weights (0.299 R + 0.587 G + 0.114 B) to compute
    the greyscale image — this is the ITU-recommended weighting for
    perceptual-luminance preservation and matches torchvision's
    ``adjust_saturation``. ``factor=0.0`` is fully greyscale;
    ``factor=1.0`` is a no-op.
    """

    _LUMA_WEIGHTS: tuple[float, float, float] = (0.299, 0.587, 0.114)

    def __init__(self, factor: float) -> None:
        self.factor = float(factor)

    def apply_image(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if image.ndim != 3 or image.shape[2] != 3:
            return image
        img32 = image.astype(np.float32, copy=False)
        r, g, b = self._LUMA_WEIGHTS
        grey = img32[..., 0] * r + img32[..., 1] * g + img32[..., 2] * b
        grey3 = np.broadcast_to(grey[..., None], img32.shape)
        out: npt.NDArray[Any] = grey3 + (img32 - grey3) * self.factor
        result: npt.NDArray[Any] = np.clip(out, 0.0, 255.0).astype(image.dtype)
        return result


class HueShiftTransform(_PhotometricTransform):
    """Rotate hue by ``shift`` (in turns; ``shift=0.5`` swaps R↔C colours).

    Typical training range: ``[-0.015, 0.015]``. Anything past ~0.05 starts
    breaking colour-dependent classes; keep this knob small.

    Implementation: convert RGB → HSV, add ``shift * 180`` to H (OpenCV
    uses 0-180 range for H to fit in uint8), wrap modulo 180, convert back.
    """

    def __init__(self, shift: float) -> None:
        self.shift = float(shift)

    def apply_image(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if image.ndim != 3 or image.shape[2] != 3:
            return image
        if self.shift == 0.0:
            return image
        # Manual RGB↔HSV without cv2 to avoid the dep — performance is
        # fine for our augmentation throughput. Algorithm matches
        # colorsys.rgb_to_hsv vectorised.
        img32 = image.astype(np.float32, copy=False) / 255.0
        h, s, v = _rgb_to_hsv(img32[..., 0], img32[..., 1], img32[..., 2])
        h = (h + self.shift) % 1.0
        r, g, b = _hsv_to_rgb(h, s, v)
        out = np.stack([r, g, b], axis=-1) * 255.0
        result: npt.NDArray[Any] = np.clip(out, 0.0, 255.0).astype(image.dtype)
        return result


# ---------------------------------------------------------------------------
# Augmentation wrappers (random sampling)
# ---------------------------------------------------------------------------


class RandomColorJitter(Augmentation):
    """Compose brightness / contrast / saturation / hue jitter.

    Each component independently fires with probability ``prob`` and
    samples its own random factor inside the configured range. Defaults
    `hsv_h=0.015 hsv_s=0.7 hsv_v=0.4``
    expressed in this module's parameterisation:

    * brightness ∈ ``[1 - 0.4, 1 + 0.4]``
    * contrast   ∈ ``[1 - 0.4, 1 + 0.4]``
    * saturation ∈ ``[1 - 0.7, 1 + 0.7]``
    * hue shift  ∈ ``[-0.015, +0.015]`` turns

    Set any of the deltas to ``0.0`` to disable that component.
    """

    def __init__(
        self,
        *,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.7,
        hue: float = 0.015,
        prob: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        for name, value in (
            ("brightness", brightness),
            ("contrast", contrast),
            ("saturation", saturation),
            ("hue", hue),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0; got {value}")
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1]; got {prob}")
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self.prob = float(prob)
        self.rng = rng if rng is not None else np.random.default_rng()

    def get_transform(self, image: npt.NDArray[Any]) -> Transform:
        # Compose the active jitter ops into a single `_PhotometricList`.
        ops: list[_PhotometricTransform] = []
        if self.brightness > 0.0 and float(self.rng.random()) < self.prob:
            factor = float(self.rng.uniform(1.0 - self.brightness, 1.0 + self.brightness))
            ops.append(BrightnessTransform(factor))
        if self.contrast > 0.0 and float(self.rng.random()) < self.prob:
            factor = float(self.rng.uniform(1.0 - self.contrast, 1.0 + self.contrast))
            ops.append(ContrastTransform(factor))
        if self.saturation > 0.0 and float(self.rng.random()) < self.prob:
            factor = float(self.rng.uniform(1.0 - self.saturation, 1.0 + self.saturation))
            ops.append(SaturationTransform(factor))
        if self.hue > 0.0 and float(self.rng.random()) < self.prob:
            shift = float(self.rng.uniform(-self.hue, self.hue))
            ops.append(HueShiftTransform(shift))
        if not ops:
            return _NoOpTransform()
        if len(ops) == 1:
            return ops[0]
        return _PhotometricList(ops)


class _PhotometricList(_PhotometricTransform):
    """Compose multiple photometric transforms into one (apply left-to-right)."""

    def __init__(self, transforms: list[_PhotometricTransform]) -> None:
        self.transforms = transforms

    def apply_image(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        out = image
        for t in self.transforms:
            out = t.apply_image(out)
        return out


# ---------------------------------------------------------------------------
# RGB ↔ HSV helpers (vectorised, no cv2 dependency)
# ---------------------------------------------------------------------------


def _rgb_to_hsv(
    r: npt.NDArray[np.float32], g: npt.NDArray[np.float32], b: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Vectorised RGB→HSV. All inputs are ``float32`` in ``[0, 1]``."""
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc
    # Saturation is 0 when maxc is 0 (pure black) — guard division.
    s = np.where(maxc > 0, delta / np.maximum(maxc, 1e-12), 0.0)
    # Hue computation: which channel is dominant determines the sextant.
    rc = (maxc - r) / np.maximum(delta, 1e-12)
    gc = (maxc - g) / np.maximum(delta, 1e-12)
    bc = (maxc - b) / np.maximum(delta, 1e-12)
    h = np.where(
        maxc == r,
        bc - gc,
        np.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc),
    )
    h = (h / 6.0) % 1.0
    # Where delta is 0 (greyscale), hue is undefined; conventionally 0.
    h = np.where(delta > 0, h, 0.0)
    return h.astype(np.float32), s.astype(np.float32), v.astype(np.float32)


def _hsv_to_rgb(
    h: npt.NDArray[np.float32], s: npt.NDArray[np.float32], v: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Vectorised HSV→RGB. All inputs are ``float32`` in ``[0, 1]``."""
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    i = i % 6
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    # Build RGB by selecting per-pixel based on the sextant `i`.
    r = np.where(
        i == 0,
        v,
        np.where(i == 1, q, np.where(i == 2, p, np.where(i == 3, p, np.where(i == 4, t, v)))),
    )
    g = np.where(
        i == 0,
        t,
        np.where(i == 1, v, np.where(i == 2, v, np.where(i == 3, q, np.where(i == 4, p, p)))),
    )
    b = np.where(
        i == 0,
        p,
        np.where(i == 1, p, np.where(i == 2, t, np.where(i == 3, v, np.where(i == 4, v, q)))),
    )
    return r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)
