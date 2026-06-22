"""Concrete geometric transforms used by the in-scope detectors.

`DETECTRON2_TECHNICAL_SPEC.md` §5.3 documents that the default training
augmentation for every in-scope FPN config is just two ops:
``ResizeShortestEdge`` + ``RandomFlip(horizontal=True)``. This module
implements the deterministic ``Transform`` half (i.e. ``Resize`` and
``HFlip``); :mod:`.augmentation` wraps them with the random sampling.

Everything is RGB-native per ADR 002. Resize uses Pillow ``BILINEAR``
for the RGB image and ``NEAREST`` for bitmask segmentations to match
Detectron2's behaviour.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from mayaku.data.transforms.base import Transform

__all__ = ["HFlipTransform", "LetterboxTransform", "ResizeTransform", "letterbox"]

_F32 = npt.NDArray[np.float32]


class ResizeTransform(Transform):
    """Resize an image from ``(h, w)`` to ``(new_h, new_w)``.

    Coordinate scaling is anchored at the pixel-center convention used
    throughout (`DETECTRON2_TECHNICAL_SPEC.md` §2.6 and §5.4): a pixel
    at index ``i`` represents the half-open interval ``[i, i+1)``. Under
    a multiplicative resize, coordinate ``c`` maps to ``c * scale``;
    boxes follow the same rule.
    """

    def __init__(
        self,
        h: int,
        w: int,
        new_h: int,
        new_w: int,
        interp: Image.Resampling = Image.Resampling.BILINEAR,
    ) -> None:
        self.h = h
        self.w = w
        self.new_h = new_h
        self.new_w = new_w
        self.interp = interp

    def apply_image(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if image.shape[0] != self.h or image.shape[1] != self.w:
            raise ValueError(
                f"ResizeTransform configured for ({self.h}, {self.w}) but "
                f"received image of shape {image.shape[:2]}"
            )
        if image.dtype == np.uint8:
            mode = "L" if image.ndim == 2 else None
            pil_img = Image.fromarray(image, mode=mode)
            resized = pil_img.resize((self.new_w, self.new_h), self.interp)
            return np.asarray(resized, dtype=np.uint8)
        # Fall back to a numpy path for non-uint8 (rare in this pipeline).
        # Use float32 + Pillow round-trip via mode "F" to keep behaviour close.
        if image.ndim == 2:
            pil_img = Image.fromarray(image.astype(np.float32), mode="F")
            resized = pil_img.resize((self.new_w, self.new_h), self.interp)
            return np.asarray(resized, dtype=image.dtype)
        # Multi-channel float — resize per channel.
        channels = [
            np.asarray(
                Image.fromarray(image[..., c].astype(np.float32), mode="F").resize(
                    (self.new_w, self.new_h), self.interp
                ),
                dtype=image.dtype,
            )
            for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=-1)

    def apply_coords(self, coords: _F32) -> _F32:
        scale_x = self.new_w / self.w
        scale_y = self.new_h / self.h
        out = coords.astype(np.float32, copy=True)
        out[:, 0] *= scale_x
        out[:, 1] *= scale_y
        return out

    def apply_segmentation(self, mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        # Always nearest-neighbour for label / bitmasks; do not anti-alias.
        original_interp = self.interp
        self.interp = Image.Resampling.NEAREST
        try:
            return self.apply_image(mask)
        finally:
            self.interp = original_interp


class HFlipTransform(Transform):
    """Horizontal flip about the image's vertical centerline.

    ``apply_coords((x, y)) -> (W - x, y)``. The keypoint flip-pair swap
    is *not* applied here — see :class:`TransformList.apply_keypoints`
    for why (it must run once per odd-flip parity, not per individual
    flip).
    """

    is_horizontal_flip = True

    def __init__(self, width: int) -> None:
        self.width = width

    def apply_image(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.ascontiguousarray(image[:, ::-1, ...])

    def apply_coords(self, coords: _F32) -> _F32:
        out = coords.astype(np.float32, copy=True)
        out[:, 0] = self.width - out[:, 0]
        return out


class LetterboxTransform(Transform):
    """Aspect-preserving resize into a fixed ``size × size`` square, **centred**.

    The fixed-shape inference geometry (the deploy keystone). The image is
    scaled by ``scale = size / max(h, w)`` so the long side becomes ``size``;
    the short side is padded equally on both sides (the "letterbox" bars) to a
    ``size × size`` canvas. A single static shape is what every fast path
    (torch.compile / ONNX / TensorRT / CoreML) specialises on, and a **single
    uniform scale** (not per-axis) makes the inverse exact.

    Centred padding (not top-left) matches the YOLO-family convention and keeps
    real content off the canvas edge on all four sides (mild win for border /
    small objects). The pad offset is recorded so predictions map back exactly:
    forward ``c → c·scale + pad``; inverse ``c → (c − pad)/scale``.

    ``pad_value`` fills the bars. Pass the dataset pixel mean so the padded
    region is ≈0 after the model's normalize (neutral); defaults to 0.
    """

    def __init__(
        self,
        h: int,
        w: int,
        size: int,
        *,
        pad_value: float = 0.0,
        interp: Image.Resampling = Image.Resampling.BILINEAR,
    ) -> None:
        if size <= 0:
            raise ValueError(f"LetterboxTransform size must be > 0; got {size}")
        self.h = h
        self.w = w
        self.size = size
        self.scale = size / max(h, w)
        # Long side → size; clamp against float rounding so the resized image
        # never exceeds the canvas.
        self.new_h = min(round(h * self.scale), size)
        self.new_w = min(round(w * self.scale), size)
        self.pad_top = (size - self.new_h) // 2
        self.pad_left = (size - self.new_w) // 2
        self.pad_value = pad_value
        self.interp = interp

    def _resized(self, image: npt.NDArray[Any], interp: Image.Resampling) -> npt.NDArray[Any]:
        return ResizeTransform(self.h, self.w, self.new_h, self.new_w, interp).apply_image(image)

    def _pad(self, resized: npt.NDArray[Any], fill: float) -> npt.NDArray[Any]:
        # Pad rather than fill-then-overwrite: ``np.pad`` writes only the bars and
        # copies the content once (the old ``np.full`` filled the whole canvas,
        # then ~half of it was overwritten). dtype is preserved.
        pad_width = [
            (self.pad_top, self.size - self.new_h - self.pad_top),
            (self.pad_left, self.size - self.new_w - self.pad_left),
        ]
        if resized.ndim == 3:
            pad_width.append((0, 0))  # don't pad the channel axis
        return np.pad(resized, pad_width, mode="constant", constant_values=fill)

    def apply_image(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return self._pad(self._resized(image, self.interp), self.pad_value)

    def apply_segmentation(self, mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        # Nearest-neighbour for labels; pad with 0 (background), not pad_value.
        return self._pad(self._resized(mask, Image.Resampling.NEAREST), 0)

    def apply_coords(self, coords: _F32) -> _F32:
        out = coords.astype(np.float32, copy=True)
        out[:, 0] = out[:, 0] * self.scale + self.pad_left
        out[:, 1] = out[:, 1] * self.scale + self.pad_top
        return out

    def inverse_coords(self, coords: _F32) -> _F32:
        """Map ``size × size`` letterbox coords back to the original image."""
        out = coords.astype(np.float32, copy=True)
        out[:, 0] = (out[:, 0] - self.pad_left) / self.scale
        out[:, 1] = (out[:, 1] - self.pad_top) / self.scale
        return out

    def inverse_box(self, boxes: _F32) -> _F32:
        """Map ``(N, 4)`` xyxy boxes back to the original image (un-letterbox)."""
        if boxes.size == 0:
            return boxes.reshape(0, 4).astype(np.float32, copy=False)
        out = boxes.astype(np.float32, copy=True)
        out[:, 0::2] = (out[:, 0::2] - self.pad_left) / self.scale
        out[:, 1::2] = (out[:, 1::2] - self.pad_top) / self.scale
        return out


def letterbox(
    image: npt.NDArray[Any], size: int, *, pad_value: float = 0.0
) -> tuple[npt.NDArray[Any], LetterboxTransform]:
    """Centred-letterbox ``image`` to ``size × size``; return ``(padded, transform)``.

    The geometry is built **once** as a :class:`LetterboxTransform` and reused:
    the returned transform carries ``scale``/``pad`` so the caller maps
    predictions back with ``transform.inverse_box(...)``. The eager
    ``Predictor``, the evaluator, and (later) training all go through this.
    """
    h, w = image.shape[:2]
    transform = LetterboxTransform(h, w, size, pad_value=pad_value)
    return transform.apply_image(image), transform
