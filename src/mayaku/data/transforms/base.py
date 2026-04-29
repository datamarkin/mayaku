"""Transform protocol shared by every geometric / photometric op.

This is the same shape-of-API as Detectron2's ``fvcore.transforms``
``Transform`` (`DETECTRON2_TECHNICAL_SPEC.md` §5.7), reduced to the
subset that the in-scope detectors actually use:

* ``apply_image(img: ndarray) -> ndarray`` — the primary op.
* ``apply_coords((N, 2) ndarray) -> ndarray`` — the only pure-geometric
  primitive a subclass *must* implement.

Everything else (boxes, polygons, keypoints, segmentation) is derived
from those two by the base class so concrete transforms only override
what they need to. A horizontal flip overrides ``apply_image`` and
``apply_coords``; the box / polygon / keypoint transforms then come
for free.

The keypoint flip-pair swap is **not** baked into individual transforms.
Per `DETECTRON2_TECHNICAL_SPEC.md` §5.7, that swap belongs at the
:class:`TransformList` boundary because it depends on whether the
*total* number of horizontal flips applied is odd or even (two flips
cancel). :class:`TransformList.apply_keypoints` consults
``num_horizontal_flips`` and applies the dataset-supplied
``flip_indices`` permutation once at the end.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["Transform", "TransformList"]

_F32 = npt.NDArray[np.float32]
_U8Image = npt.NDArray[np.uint8]


class Transform:
    """Base class. Subclasses must override ``apply_image`` and ``apply_coords``."""

    # Override to True in horizontal-flip transforms; the keypoint pair
    # swap at the TransformList boundary depends on the parity of this
    # count across the composed list.
    is_horizontal_flip: bool = False

    def apply_image(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        raise NotImplementedError

    def apply_coords(self, coords: _F32) -> _F32:
        raise NotImplementedError

    # ----- derived ops ---------------------------------------------------

    def apply_box(self, boxes: _F32) -> _F32:
        """Transform ``(N, 4)`` xyxy boxes by sending the four corners
        through ``apply_coords`` and rebuilding an axis-aligned bbox.

        The min/max rebuild is necessary because affine transforms (e.g.
        rotation, shear) produce non-rectangular quads; the resulting
        AABB is the tightest box containing the warped quad. For
        translation/flip/scale this is exactly the original transformed
        rectangle.
        """
        if boxes.size == 0:
            return boxes.reshape(0, 4)
        # Build (N, 4, 2) corners
        x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        corners = (
            np.stack(
                [
                    np.stack([x0, y0], axis=1),
                    np.stack([x1, y0], axis=1),
                    np.stack([x1, y1], axis=1),
                    np.stack([x0, y1], axis=1),
                ],
                axis=1,
            )
            .reshape(-1, 2)
            .astype(np.float32, copy=False)
        )
        warped = self.apply_coords(corners).reshape(-1, 4, 2)
        out = np.empty_like(boxes, dtype=np.float32)
        out[:, 0] = warped[:, :, 0].min(axis=1)
        out[:, 1] = warped[:, :, 1].min(axis=1)
        out[:, 2] = warped[:, :, 0].max(axis=1)
        out[:, 3] = warped[:, :, 1].max(axis=1)
        return out

    def apply_polygons(self, polygons: list[_F32]) -> list[_F32]:
        """Each polygon is a flat ``[x0, y0, x1, y1, ...]`` ``float32``."""
        return [
            self.apply_coords(p.reshape(-1, 2).astype(np.float32, copy=False))
            .reshape(-1)
            .astype(np.float32, copy=False)
            for p in polygons
        ]

    def apply_keypoints(self, keypoints: _F32) -> _F32:
        """``(N, K, 3)`` of ``(x, y, v)``. Coords transform; visibility passes through."""
        if keypoints.size == 0:
            return keypoints
        n, k, _ = keypoints.shape
        xy = keypoints[..., :2].reshape(-1, 2).astype(np.float32, copy=False)
        warped = self.apply_coords(xy).reshape(n, k, 2)
        out = keypoints.astype(np.float32, copy=True)
        out[..., :2] = warped
        return out

    def apply_segmentation(self, mask: _U8Image) -> _U8Image:
        """Bitmask is just an image with one channel; nearest-neighbour by default.

        Subclasses that resize use ``Image.NEAREST`` for masks even when
        bilinear is correct for RGB. Default falls through to
        ``apply_image`` — concrete subclasses override when interp differs.
        """
        return self.apply_image(mask)


class TransformList:
    """Composed list of transforms applied left-to-right.

    Use :class:`mayaku.data.transforms.augmentation.AugmentationList` to
    *generate* one (the augmentations may sample randomness); ``TransformList``
    is the deterministic record of what was applied so the same operations
    can be replayed on annotations.
    """

    def __init__(
        self,
        transforms: Sequence[Transform],
        flip_indices: Sequence[int] | None = None,
    ) -> None:
        """``flip_indices`` is the dataset-specific keypoint flip-pair
        permutation (Step 4 / Step 5). Required when keypoints will be
        transformed *and* any ``is_horizontal_flip`` transform is in the
        list — otherwise the swap silently doesn't happen and left/right
        keypoints get scrambled."""
        self.transforms: list[Transform] = list(transforms)
        self.flip_indices: tuple[int, ...] | None = (
            tuple(flip_indices) if flip_indices is not None else None
        )

    @property
    def num_horizontal_flips(self) -> int:
        return sum(1 for t in self.transforms if t.is_horizontal_flip)

    def apply_image(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        out = image
        for t in self.transforms:
            out = t.apply_image(out)
        return out

    def apply_box(self, boxes: _F32) -> _F32:
        out = boxes.astype(np.float32, copy=True)
        for t in self.transforms:
            out = t.apply_box(out)
        return out

    def apply_polygons(self, polygons: list[_F32]) -> list[_F32]:
        out = polygons
        for t in self.transforms:
            out = t.apply_polygons(out)
        return out

    def apply_segmentation(self, mask: _U8Image) -> _U8Image:
        out = mask
        for t in self.transforms:
            out = t.apply_segmentation(out)
        return out

    def apply_keypoints(self, keypoints: _F32) -> _F32:
        """Transform coords through every step, then apply the
        flip-pair permutation once if the total number of horizontal
        flips is odd."""
        if keypoints.size == 0:
            return keypoints.astype(np.float32, copy=True)
        out = keypoints.astype(np.float32, copy=True)
        for t in self.transforms:
            out = t.apply_keypoints(out)
        if self.num_horizontal_flips % 2 == 1:
            if self.flip_indices is None:
                raise ValueError(
                    "An odd number of horizontal flips was applied but the "
                    "TransformList has no flip_indices. Pass the dataset's "
                    "keypoint flip-pair permutation when constructing the "
                    "TransformList (or AugmentationList)."
                )
            out = out[:, list(self.flip_indices), :]
        return out
