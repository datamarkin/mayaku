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

__all__ = ["HFlipTransform", "ResizeTransform"]

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
