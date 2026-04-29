"""Dataset loaders. Currently: COCO (detection / instance seg / keypoints)."""

from __future__ import annotations

from mayaku.data.datasets.coco import build_coco_metadata, load_coco_json

__all__ = ["build_coco_metadata", "load_coco_json"]
