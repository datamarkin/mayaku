"""Data layer: catalog, COCO loader, mapper, transforms, samplers, collate."""

from __future__ import annotations

from mayaku.data.catalog import DatasetCatalog, Metadata, default_catalog
from mayaku.data.collate import trivial_batch_collator
from mayaku.data.datasets import build_coco_metadata, load_coco_json
from mayaku.data.mapper import DatasetMapper
from mayaku.data.samplers import (
    AspectRatioGroupedDataset,
    InferenceSampler,
    TrainingSampler,
)
from mayaku.data.transforms import (
    AugInput,
    Augmentation,
    AugmentationList,
    HFlipTransform,
    RandomFlip,
    ResizeShortestEdge,
    ResizeTransform,
    Transform,
    TransformList,
)

__all__ = [
    "AspectRatioGroupedDataset",
    "AugInput",
    "Augmentation",
    "AugmentationList",
    "DatasetCatalog",
    "DatasetMapper",
    "HFlipTransform",
    "InferenceSampler",
    "Metadata",
    "RandomFlip",
    "ResizeShortestEdge",
    "ResizeTransform",
    "TrainingSampler",
    "Transform",
    "TransformList",
    "build_coco_metadata",
    "default_catalog",
    "load_coco_json",
    "trivial_batch_collator",
]
