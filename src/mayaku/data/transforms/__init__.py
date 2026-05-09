"""Data-pipeline transforms (deterministic) and augmentations (random).

Splits the spec §5.7 surface in two:

* :mod:`.base` and :mod:`.geometry` — the deterministic ``Transform``
  primitives (``ResizeTransform``, ``HFlipTransform``) and the
  ``TransformList`` that records what was applied.
* :mod:`.augmentation` — the ``Augmentation`` wrappers that sample
  randomness and produce a ``Transform`` (``ResizeShortestEdge``,
  ``RandomFlip``), plus ``AugmentationList`` and ``AugInput``.

The keypoint flip-pair swap is centralised in
``TransformList.apply_keypoints``; concrete transforms never touch the
keypoint index permutation.
"""

from __future__ import annotations

from mayaku.data.transforms.augmentation import (
    AugInput,
    Augmentation,
    AugmentationList,
    RandomFlip,
    ResizeShortestEdge,
)
from mayaku.data.transforms.base import Transform, TransformList
from mayaku.data.transforms.geometry import HFlipTransform, ResizeTransform
from mayaku.data.transforms.photometric import (
    AutoContrastTransform,
    BrightnessTransform,
    ContrastTransform,
    EqualizeTransform,
    HueShiftTransform,
    PosterizeTransform,
    RandAugment,
    RandomColorJitter,
    SaturationTransform,
    SolarizeTransform,
)

__all__ = [
    "AugInput",
    "Augmentation",
    "AugmentationList",
    "AutoContrastTransform",
    "BrightnessTransform",
    "ContrastTransform",
    "EqualizeTransform",
    "HFlipTransform",
    "HueShiftTransform",
    "PosterizeTransform",
    "RandAugment",
    "RandomColorJitter",
    "RandomFlip",
    "ResizeShortestEdge",
    "ResizeTransform",
    "SaturationTransform",
    "SolarizeTransform",
    "Transform",
    "TransformList",
]
