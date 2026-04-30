"""Data layer: catalog, COCO loader, mapper, transforms, samplers, collate."""

from __future__ import annotations

from mayaku.data.catalog import DatasetCatalog, Metadata, default_catalog
from mayaku.data.collate import trivial_batch_collator
from mayaku.data.datasets import build_coco_metadata, load_coco_json
from mayaku.data.mapper import DatasetMapper
from mayaku.data.multi_sample import (
    CopyPaste,
    MixUp,
    Mosaic,
    MultiSampleAugmentation,
    MultiSampleMappedDataset,
)
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
    RandomColorJitter,
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
    "CopyPaste",
    "DatasetCatalog",
    "DatasetMapper",
    "HFlipTransform",
    "InferenceSampler",
    "Metadata",
    "MixUp",
    "Mosaic",
    "MultiSampleAugmentation",
    "MultiSampleMappedDataset",
    "RandomColorJitter",
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
