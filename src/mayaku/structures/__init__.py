"""Per-image data structures shared by every detector head.

Channel order is **RGB** end-to-end (ADR 002,
``docs/decisions/002-rgb-native-image-ingestion.md``); :class:`ImageList`
and any caller that touches pixels assumes ``(C, H, W)`` with channels
in ``[R, G, B]`` order.
"""

from __future__ import annotations

from mayaku.structures.boxes import (
    Boxes,
    BoxMode,
    pairwise_intersection,
    pairwise_ioa,
    pairwise_iou,
)
from mayaku.structures.image_list import ImageList
from mayaku.structures.instances import Instances
from mayaku.structures.keypoints import (
    Keypoints,
    heatmaps_to_keypoints,
    keypoints_to_heatmap,
)
from mayaku.structures.masks import (
    BitMasks,
    PolygonMasks,
    ROIMasks,
    paste_masks_in_image,
)

__all__ = [
    "BitMasks",
    "BoxMode",
    "Boxes",
    "ImageList",
    "Instances",
    "Keypoints",
    "PolygonMasks",
    "ROIMasks",
    "heatmaps_to_keypoints",
    "keypoints_to_heatmap",
    "pairwise_intersection",
    "pairwise_ioa",
    "pairwise_iou",
    "paste_masks_in_image",
]
