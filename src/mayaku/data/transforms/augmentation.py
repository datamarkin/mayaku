"""Random-sampling augmentations on top of deterministic transforms.

An :class:`Augmentation` produces a :class:`Transform` for the current
image — that's the seam where any randomness lives. The :class:`AugInput`
container holds the mutable image so an ``AugmentationList`` can apply
each transform to it as we build the composed list, allowing later
augmentations to react to the image's *post-previous-transform* shape
(e.g. ``RandomFlip`` needs the resized width, not the original).

This mirrors `DETECTRON2_TECHNICAL_SPEC.md` §5.7 minus the bookkeeping
features Detectron2 carries for legacy semantic-segmentation paths
(`AugInput.sem_seg`, ``StandardAugInput``); we have neither in scope.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from mayaku.data.transforms.base import Transform, TransformList
from mayaku.data.transforms.geometry import HFlipTransform, ResizeTransform

__all__ = [
    "AugInput",
    "Augmentation",
    "AugmentationList",
    "RandomFlip",
    "ResizeShortestEdge",
    "compute_resized_hw",
]


def compute_resized_hw(h: int, w: int, short_edge: int, max_size: int) -> tuple[int, int]:
    """Shortest-edge resize math, factored out so non-PIL paths (e.g. the
    GPU preprocessing branch in :class:`mayaku.inference.Predictor`) can
    reuse the exact same target dimensions as :class:`ResizeShortestEdge`.
    """
    scale = short_edge / min(h, w)
    if max(h, w) * scale > max_size:
        scale = max_size / max(h, w)
    return round(h * scale), round(w * scale)


@dataclass
class AugInput:
    """Mutable image carrier passed through ``AugmentationList``."""

    image: npt.NDArray[np.uint8]


class Augmentation:
    """Base class. Subclasses implement ``get_transform(image) -> Transform``."""

    def get_transform(self, image: npt.NDArray[Any]) -> Transform:
        raise NotImplementedError


class ResizeShortestEdge(Augmentation):
    """Resize so the short edge equals one of ``short_edge_lengths`` and
    the long edge is clamped at ``max_size``.

    Spec §5.3 default: ``short_edge_lengths=(640, 672, 704, 736, 768, 800)``,
    ``max_size=1333``, ``sample_style="choice"``. ``"range"`` samples
    uniformly between ``min(short_edge_lengths)`` and
    ``max(short_edge_lengths)``.
    """

    def __init__(
        self,
        short_edge_lengths: Sequence[int],
        max_size: int,
        sample_style: str = "choice",
        interp: Image.Resampling = Image.Resampling.BILINEAR,
        rng: np.random.Generator | None = None,
    ) -> None:
        if sample_style not in ("choice", "range"):
            raise ValueError(f"sample_style must be 'choice' or 'range'; got {sample_style!r}")
        if sample_style == "range" and len(short_edge_lengths) != 2:
            raise ValueError(
                "sample_style='range' requires exactly two short_edge_lengths "
                "(min, max); got length "
                f"{len(short_edge_lengths)}"
            )
        self.short_edge_lengths = tuple(short_edge_lengths)
        self.max_size = max_size
        self.sample_style = sample_style
        self.interp = interp
        self.rng = rng if rng is not None else np.random.default_rng()

    def get_transform(self, image: npt.NDArray[Any]) -> ResizeTransform:
        h, w = image.shape[:2]
        if self.sample_style == "choice":
            target = int(self.rng.choice(self.short_edge_lengths))
        else:
            lo, hi = self.short_edge_lengths
            target = int(self.rng.integers(lo, hi + 1))
        new_h, new_w = compute_resized_hw(h, w, target, self.max_size)
        return ResizeTransform(h, w, new_h, new_w, self.interp)


class RandomFlip(Augmentation):
    """Horizontal flip with probability ``p``."""

    def __init__(self, prob: float = 0.5, rng: np.random.Generator | None = None) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"RandomFlip prob must be in [0, 1]; got {prob}")
        self.prob = prob
        self.rng = rng if rng is not None else np.random.default_rng()

    def get_transform(self, image: npt.NDArray[Any]) -> Transform:
        if float(self.rng.random()) < self.prob:
            return HFlipTransform(width=image.shape[1])
        return _NoOpTransform()


class _NoOpTransform(Transform):
    """Identity transform; used when a random augmentation didn't fire."""

    def apply_image(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return image

    def apply_coords(self, coords: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return coords


class AugmentationList:
    """Composes a list of :class:`Augmentation` into a :class:`TransformList`.

    Calling the list with an :class:`AugInput` mutates ``aug_input.image``
    in place (each augmentation samples against the *current* image state)
    and returns the deterministic :class:`TransformList` so annotations
    can be replayed.

    ``flip_indices`` is forwarded to the returned ``TransformList`` and is
    required when the dataset has keypoints — see
    :class:`TransformList.apply_keypoints` for why.
    """

    def __init__(
        self,
        augmentations: Sequence[Augmentation],
        flip_indices: Sequence[int] | None = None,
    ) -> None:
        self.augmentations = list(augmentations)
        self.flip_indices = flip_indices

    def __call__(self, aug_input: AugInput) -> TransformList:
        transforms: list[Transform] = []
        for aug in self.augmentations:
            t = aug.get_transform(aug_input.image)
            aug_input.image = t.apply_image(aug_input.image)
            transforms.append(t)
        return TransformList(transforms, flip_indices=self.flip_indices)
