"""Dataset-aware auto-tuning for ``mayaku train``.

This package powers the :class:`mayaku.config.AutoConfig` flow: a
single read-only pass over the COCO dataset emits a set of fine-tune
recipe overrides (anchor sizes/ARs, base_lr, max_iter, augmentation
toggles, sampler choice). Each module is a pure function, callable
independently and unit-tested in isolation.

Public surface:

* :func:`analyze_dataset` — compute :class:`DatasetStats` from
  ``load_coco_json`` output
* :func:`cluster_sizes` / :func:`cluster_aspect_ratios` — deterministic
  1-D k-means for anchor generation
* :func:`derive_overrides` — turn stats + base config into a nested
  override dict ready for :func:`mayaku.config.merge_overrides`
* :func:`collect_set_paths` / :func:`filter_unset` — track which YAML
  paths the user set explicitly and drop overrides that would clobber
  them
"""

from __future__ import annotations

from mayaku.tuning.anchor_kmeans import cluster_aspect_ratios, cluster_sizes
from mayaku.tuning.dataset_stats import DatasetStats, analyze_dataset
from mayaku.tuning.recipe import (
    MIN_IMAGES_FOR_AUTO_CONFIG,
    SizeBucket,
    collect_set_paths,
    derive_overrides,
    filter_unset,
    size_bucket,
    walk_leaves,
)

__all__ = [
    "MIN_IMAGES_FOR_AUTO_CONFIG",
    "DatasetStats",
    "SizeBucket",
    "analyze_dataset",
    "cluster_aspect_ratios",
    "cluster_sizes",
    "collect_set_paths",
    "derive_overrides",
    "filter_unset",
    "size_bucket",
    "walk_leaves",
]
