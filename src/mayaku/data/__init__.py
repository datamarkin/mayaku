"""Data layer: COCO loading, augmentation, batching, and per-node shared storage."""

from __future__ import annotations

from mayaku.data.augment import CLEAN_AUG, DEFAULT_AUG, Augment
from mayaku.data.batch import batch_to, collate
from mayaku.data.catalog import DatasetCatalog, Metadata, default_catalog
from mayaku.data.dataset import CocoDetection
from mayaku.data.serialize import SerializedList
from mayaku.data.shared import load_shared_dataset

__all__ = [
    "CLEAN_AUG",
    "DEFAULT_AUG",
    "Augment",
    "CocoDetection",
    "DatasetCatalog",
    "Metadata",
    "SerializedList",
    "batch_to",
    "collate",
    "default_catalog",
    "load_shared_dataset",
]
