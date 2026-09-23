"""Dataset-aware tuning: auto-config and dataset statistics."""

from __future__ import annotations

from mayaku.tuning.dataset_stats import DatasetStats, analyze_dataset
from mayaku.tuning.recipe import (
    AUTO_PATHS,
    MIN_IMAGES_FOR_AUTO_CONFIG,
    apply_auto_config,
    collect_set_paths,
)

__all__ = [
    "AUTO_PATHS",
    "MIN_IMAGES_FOR_AUTO_CONFIG",
    "DatasetStats",
    "analyze_dataset",
    "apply_auto_config",
    "collect_set_paths",
]
