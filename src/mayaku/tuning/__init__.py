"""Dataset-aware auto-tuning for ``mayaku train``.

This package powers the :class:`mayaku.config.AutoConfig` flow: a
single read-only pass over the COCO dataset emits a set of fine-tune
recipe overrides (epoch budget, augmentation strength, sampler choice — never architecture-tuned hyperparameters;
see :data:`mayaku.tuning.recipe.ARCHITECTURE_TUNED_PATHS`). Each module
is a pure function, callable independently and unit-tested in isolation.

Public surface:

* :func:`analyze_dataset` — compute :class:`DatasetStats` from
  ``load_coco_json`` output
* :func:`derive_overrides` — turn stats + base config into a nested
  override dict ready for :func:`mayaku.config.merge_overrides`
* :func:`collect_set_paths` / :func:`filter_unset` — track which YAML
  paths the user set explicitly and drop overrides that would clobber
  them
"""

from __future__ import annotations

from mayaku.tuning.dataset_stats import DatasetStats, analyze_dataset
from mayaku.tuning.recipe import (
    FINETUNE_GRAD_ACCUM_STEPS,
    FINETUNE_IMS_PER_BATCH,
    MIN_IMAGES_FOR_AUTO_CONFIG,
    SizeBucket,
    collect_set_paths,
    derive_overrides,
    filter_unset,
    size_bucket,
    walk_leaves,
)

__all__ = [
    "FINETUNE_GRAD_ACCUM_STEPS",
    "FINETUNE_IMS_PER_BATCH",
    "MIN_IMAGES_FOR_AUTO_CONFIG",
    "DatasetStats",
    "SizeBucket",
    "analyze_dataset",
    "collect_set_paths",
    "derive_overrides",
    "filter_unset",
    "size_bucket",
    "walk_leaves",
]
