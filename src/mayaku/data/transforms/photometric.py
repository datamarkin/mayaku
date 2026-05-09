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
    "AutoContrastTransform",
    "BrightnessTransform",
    "ContrastTransform",
    "EqualizeTransform",
    "HueShiftTransform",
    "PosterizeTransform",
    "RandAugment",
    "RandomColorJitter",
    "SaturationTransform",
    "SolarizeTransform",
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


class SolarizeTransform(_PhotometricTransform):
    """Invert pixels at or above ``threshold`` (0-256).

    ``threshold=256`` is a no-op (no pixel reaches it). ``threshold=0``
    inverts everything.
    """

    def __init__(self, threshold: int) -> None:
        if not 0 <= threshold <= 256:
            raise ValueError(f"threshold must be in [0, 256]; got {threshold}")
        self.threshold = int(threshold)

    def apply_image(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if image.ndim != 3 or image.shape[2] != 3:
            return image
        if self.threshold >= 256:
            return image
        out = image.copy()
        mask = out >= self.threshold
        # 255 - x; preserve dtype.
        out[mask] = (255 - out[mask].astype(np.int32)).astype(image.dtype)
        return out


class PosterizeTransform(_PhotometricTransform):
    """Reduce per-channel bit depth to ``bits`` (1-8).

    ``bits=8`` is a no-op. Lower values quantise the channel to fewer
    distinct levels (``bits=4`` keeps 16 levels per channel).
    """

    def __init__(self, bits: int) -> None:
        if not 1 <= bits <= 8:
            raise ValueError(f"bits must be in [1, 8]; got {bits}")
        self.bits = int(bits)

    def apply_image(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if image.ndim != 3 or image.shape[2] != 3:
            return image
        if self.bits == 8 or image.dtype != np.uint8:
            return image
        shift = 8 - self.bits
        # Clear the low ``shift`` bits — quantises to ``2**bits`` levels.
        result: npt.NDArray[Any] = (image >> shift) << shift
        return result


class AutoContrastTransform(_PhotometricTransform):
    """Per-channel min-max stretch to ``[0, 255]``.

    Constant-value channels (where ``min == max``) pass through
    unchanged. Operates per-channel independently — the channels' joint
    distribution shifts but no channel inversion occurs.
    """

    def apply_image(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if image.ndim != 3 or image.shape[2] != 3:
            return image
        out = image.astype(np.float32, copy=True)
        for c in range(3):
            ch = out[..., c]
            lo = float(ch.min())
            hi = float(ch.max())
            if hi <= lo:
                continue
            out[..., c] = (ch - lo) * (255.0 / (hi - lo))
        result: npt.NDArray[Any] = np.clip(out, 0.0, 255.0).astype(image.dtype)
        return result


class EqualizeTransform(_PhotometricTransform):
    """Per-channel histogram equalisation (uint8 only; pass-through otherwise).

    Standard CDF normalisation: build a 256-bin histogram per channel,
    map each input value through the channel's normalised cumulative
    distribution. Constant-value channels pass through unchanged.
    """

    def apply_image(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            return image
        out = np.empty_like(image)
        n_pixels = image.shape[0] * image.shape[1]
        for c in range(3):
            ch = image[..., c]
            hist = np.bincount(ch.ravel(), minlength=256)
            cdf = hist.cumsum()
            # cdf_min: smallest non-zero CDF anchor (skip leading zeros).
            nz = cdf[cdf > 0]
            if nz.size == 0:
                out[..., c] = ch
                continue
            cdf_min = int(nz[0])
            denom = n_pixels - cdf_min
            if denom <= 0:
                # Channel is constant — equalisation is identity.
                out[..., c] = ch
                continue
            lut = np.clip(
                np.round((cdf - cdf_min) * 255.0 / denom), 0, 255
            ).astype(np.uint8)
            out[..., c] = lut[ch]
        return out


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


class RandAugment(Augmentation):
    """RandAugment (Cubuk et al. 2019), photometric-only pool for detection.

    Per call samples ``num_ops`` ops uniformly without replacement from
    a fixed pool and applies each at intensity
    ``magnitude / max_magnitude`` of its op-specific maximum strength.
    The pool is restricted to ops that don't move pixels (no rotate /
    shear / translate), so bboxes / masks / keypoints pass through
    unchanged — no annotation re-warping required.

    Pool: ``identity, auto_contrast, equalize, solarize, posterize,
    brightness, contrast, saturation, hue``. Hue is included with a
    deliberately-small max shift (~18°) because larger hue rotations
    break colour-defined classes (traffic-light colour, jersey colour).

    Two-knob API matches the paper:

    * ``num_ops`` (paper N): ops per image. 1-3 typical; default 2.
    * ``magnitude`` (paper M): intensity in ``[0, max_magnitude]``.
      Higher = stronger augmentation. Default 9 with default
      ``max_magnitude=30`` matches the COCO recipe in the paper.

    For very-small-data fine-tunes (~200-500 images) the defaults are
    a good starting point. Below ~100 images, drop ``magnitude`` to
    5-7 to avoid masking the limited signal with augmentation noise.
    """

    _OPS: tuple[str, ...] = (
        "identity",
        "auto_contrast",
        "equalize",
        "solarize",
        "posterize",
        "brightness",
        "contrast",
        "saturation",
        "hue",
    )

    def __init__(
        self,
        *,
        num_ops: int = 2,
        magnitude: float = 9.0,
        max_magnitude: float = 30.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if num_ops < 0:
            raise ValueError(f"num_ops must be >= 0; got {num_ops}")
        if max_magnitude <= 0.0:
            raise ValueError(f"max_magnitude must be > 0; got {max_magnitude}")
        if not 0.0 <= magnitude <= max_magnitude:
            raise ValueError(
                f"magnitude must be in [0, {max_magnitude}]; got {magnitude}"
            )
        if num_ops > len(self._OPS):
            raise ValueError(
                f"num_ops={num_ops} exceeds pool size {len(self._OPS)} "
                "(without-replacement sampling)"
            )
        self.num_ops = int(num_ops)
        self.magnitude = float(magnitude)
        self.max_magnitude = float(max_magnitude)
        self.rng = rng if rng is not None else np.random.default_rng()

    def _sample_op(self, name: str, m_norm: float) -> Transform:
        # ``sign`` randomises whether the op pushes the image lighter /
        # darker / more / less saturated etc. Matches the original
        # RandAugment implementation; halves the effective parameter
        # range vs. always-positive magnitudes.
        sign = 1.0 if float(self.rng.random()) < 0.5 else -1.0
        if name == "identity":
            return _NoOpTransform()
        if name == "brightness":
            return BrightnessTransform(1.0 + sign * 0.9 * m_norm)
        if name == "contrast":
            return ContrastTransform(1.0 + sign * 0.9 * m_norm)
        if name == "saturation":
            return SaturationTransform(1.0 + sign * 0.9 * m_norm)
        if name == "hue":
            # 0.05 turns ≈ 18° at full magnitude. Detection-friendly
            # ceiling — papers using larger hue ranges (e.g. 0.5)
            # quietly hurt colour-defined classes.
            return HueShiftTransform(sign * 0.05 * m_norm)
        if name == "solarize":
            # m_norm=0 → threshold=256 (no inversion); m_norm=1 → 0.
            return SolarizeTransform(threshold=int(round(256 - 256 * m_norm)))
        if name == "posterize":
            # m_norm=0 → bits=8 (no quantise); m_norm=1 → bits=4
            # (16 levels). Don't go below 4 — fewer levels destroy
            # most object structure.
            return PosterizeTransform(bits=max(4, int(round(8 - 4 * m_norm))))
        if name == "auto_contrast":
            return AutoContrastTransform()
        if name == "equalize":
            return EqualizeTransform()
        raise AssertionError(f"unknown op {name!r}")

    def get_transform(self, image: npt.NDArray[Any]) -> Transform:
        if self.num_ops == 0:
            return _NoOpTransform()
        m_norm = self.magnitude / self.max_magnitude
        # Sample without replacement so a single image gets distinct ops
        # — increases per-image diversity vs. uniform-with-replacement.
        chosen = self.rng.choice(
            np.array(self._OPS), size=self.num_ops, replace=False
        )
        ops: list[_PhotometricTransform] = []
        for name in chosen:
            t = self._sample_op(str(name), m_norm)
            if isinstance(t, _NoOpTransform):
                continue
            assert isinstance(t, _PhotometricTransform)  # pool is photometric-only
            ops.append(t)
        if not ops:
            return _NoOpTransform()
        if len(ops) == 1:
            return ops[0]
        return _PhotometricList(ops)


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
