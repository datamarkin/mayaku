"""Detector meta-architectures."""

from __future__ import annotations

from mayaku.models.detectors.faster_rcnn import FasterRCNN, build_faster_rcnn
from mayaku.models.detectors.keypoint_rcnn import build_keypoint_rcnn
from mayaku.models.detectors.mask_rcnn import build_mask_rcnn

__all__ = [
    "FasterRCNN",
    "build_faster_rcnn",
    "build_keypoint_rcnn",
    "build_mask_rcnn",
]
