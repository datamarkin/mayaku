"""Tests for :mod:`mayaku.tuning.recipe`."""

from __future__ import annotations

import itertools

import pytest

from mayaku.config import MayakuConfig
from mayaku.tuning.dataset_stats import DatasetStats
from mayaku.tuning.recipe import (
    MIN_BOXES_FOR_ANCHOR_TUNE,
    MIN_IMAGES_FOR_AUTO_CONFIG,
    collect_set_paths,
    derive_overrides,
    filter_unset,
    size_bucket,
)

# ---------------------------------------------------------------------------
# Bucket lookup
# ---------------------------------------------------------------------------


def test_size_bucket_boundaries() -> None:
    # The thresholds in the recipe table — boundary cases pin the
    # bucket choice so a refactor can't silently widen them.
    assert size_bucket(0).name == "xs"
    assert size_bucket(499).name == "xs"
    assert size_bucket(500).name == "s"
    assert size_bucket(1_999).name == "s"
    assert size_bucket(2_000).name == "m"
    assert size_bucket(4_999).name == "m"
    assert size_bucket(5_000).name == "l"
    assert size_bucket(49_999).name == "l"
    assert size_bucket(50_000).name == "xl"
    assert size_bucket(1_000_000).name == "xl"


def test_size_bucket_lr_fits_the_documented_table() -> None:
    # The recipe table is the source of truth for the auto-config LR
    # math — pin the values so the docstring derivation stays correct.
    assert size_bucket(100).base_lr == pytest.approx(5e-4)
    assert size_bucket(1_000).base_lr == pytest.approx(1e-3)
    assert size_bucket(3_000).base_lr == pytest.approx(1e-3)
    assert size_bucket(10_000).base_lr == pytest.approx(2e-3)


# ---------------------------------------------------------------------------
# Override derivation
# ---------------------------------------------------------------------------


def _stats(
    num_images: int = 1_500,
    num_classes: int = 5,
    num_boxes: int = 1_000,
    imbalance_high: bool = False,
) -> DatasetStats:
    # Build a synthetic but realistic distribution.
    sqrt_areas = tuple(10.0 + (i % 500) for i in range(num_boxes))
    aspect_ratios = tuple(0.5 + (i % 10) / 5 for i in range(num_boxes))
    if imbalance_high:
        class_counts = {0: num_images - 5, 1: 5}
    else:
        share = max(1, num_images // num_classes)
        class_counts = {i: share for i in range(num_classes)}
    return DatasetStats(
        num_images=num_images,
        num_classes=num_classes,
        class_counts=class_counts,
        sqrt_areas=sqrt_areas,
        aspect_ratios=aspect_ratios,
        median_image_short_edge=800,
        median_image_long_edge=1333,
    )


def test_derive_overrides_empty_for_tiny_datasets() -> None:
    stats = _stats(num_images=MIN_IMAGES_FOR_AUTO_CONFIG - 1, num_boxes=5)
    assert derive_overrides(stats, MayakuConfig()) == {}


def test_derive_overrides_sets_num_classes_from_dataset() -> None:
    stats = _stats(num_classes=12)
    overrides = derive_overrides(stats, MayakuConfig())
    assert overrides["model"]["roi_heads"]["num_classes"] == 12


def test_derive_overrides_sets_anchors_when_enough_boxes() -> None:
    stats = _stats(num_boxes=MIN_BOXES_FOR_ANCHOR_TUNE)
    overrides = derive_overrides(stats, MayakuConfig())
    anchors = overrides["model"]["anchor_generator"]
    assert len(anchors["sizes"]) == 5
    # One scale per FPN level.
    assert all(len(t) == 1 for t in anchors["sizes"])
    # Three shared aspect ratios in a single tuple.
    assert len(anchors["aspect_ratios"]) == 1
    assert len(anchors["aspect_ratios"][0]) == 3


def test_derive_overrides_skips_anchors_below_min_boxes() -> None:
    stats = _stats(num_boxes=MIN_BOXES_FOR_ANCHOR_TUNE - 1)
    overrides = derive_overrides(stats, MayakuConfig())
    assert "anchor_generator" not in overrides["model"]


def test_derive_overrides_scales_lr_with_ims_per_batch() -> None:
    # Base config has ims_per_batch=16 (D2 default). Bucket base_lr is
    # anchored at bs=8, so at bs=16 it should double.
    cfg = MayakuConfig()  # ims_per_batch=16
    stats = _stats(num_images=1_500)  # bucket "s" → 1e-3 at bs=8
    overrides = derive_overrides(stats, cfg)
    assert overrides["solver"]["base_lr"] == pytest.approx(1e-3 * 2.0)


def test_derive_overrides_emits_valid_schedule() -> None:
    stats = _stats(num_images=2_000)
    overrides = derive_overrides(stats, MayakuConfig())
    sched = overrides["solver"]
    max_iter = sched["max_iter"]
    steps = sched["steps"]
    warmup = sched["warmup_iters"]
    # SolverConfig._check_schedule invariants.
    assert warmup < max_iter
    assert all(0 < s < max_iter for s in steps)
    assert all(b > a for a, b in itertools.pairwise(steps))


def test_derive_overrides_enables_repeat_factor_sampler_when_imbalanced() -> None:
    stats = _stats(num_images=1_500, imbalance_high=True)
    overrides = derive_overrides(stats, MayakuConfig())
    assert overrides["dataloader"]["sampler_train"] == "RepeatFactorTrainingSampler"


def test_derive_overrides_keeps_default_sampler_when_balanced() -> None:
    stats = _stats(num_images=1_500, imbalance_high=False)
    overrides = derive_overrides(stats, MayakuConfig())
    assert "dataloader" not in overrides


def test_derive_overrides_skips_copy_paste_when_meta_arch_is_detection() -> None:
    # Faster R-CNN can't ingest CopyPaste output even when the bucket
    # would otherwise enable it.
    stats = _stats(num_images=100_000, num_boxes=1_000)
    overrides = derive_overrides(stats, MayakuConfig())
    assert "copy_paste_prob" not in overrides["input"]


def test_derive_overrides_emits_pydantic_valid_payload() -> None:
    # Final overrides must produce a valid MayakuConfig when merged.
    # This is the real test that all the schedule math, anchor shape,
    # and AR rounding land inside Pydantic's validators.
    from mayaku.config import merge_overrides

    stats = _stats(num_images=2_000, num_boxes=500, num_classes=12)
    overrides = derive_overrides(stats, MayakuConfig())
    merged = merge_overrides(MayakuConfig(), overrides)
    assert merged.model.roi_heads.num_classes == 12
    assert merged.solver.lr_scheduler_name == "WarmupCosineLR"
    assert merged.solver.ema_enabled is True


# ---------------------------------------------------------------------------
# User-set path tracking
# ---------------------------------------------------------------------------


def test_collect_set_paths_flat() -> None:
    assert collect_set_paths({"a": 1, "b": 2}) == {"a", "b"}


def test_collect_set_paths_nested() -> None:
    paths = collect_set_paths({"solver": {"base_lr": 0.01, "max_iter": 1000}})
    assert paths == {"solver.base_lr", "solver.max_iter"}


def test_collect_set_paths_treats_lists_as_leaves() -> None:
    # tuple-of-tuples values (like anchor sizes) are leaves, not paths
    # to recurse into. The user wrote the WHOLE tuple in their YAML.
    paths = collect_set_paths(
        {"model": {"anchor_generator": {"sizes": [[32], [64]], "aspect_ratios": [[0.5, 1.0]]}}}
    )
    assert paths == {"model.anchor_generator.sizes", "model.anchor_generator.aspect_ratios"}


def test_collect_set_paths_empty() -> None:
    assert collect_set_paths({}) == set()
    assert collect_set_paths(None) == set()


# ---------------------------------------------------------------------------
# filter_unset
# ---------------------------------------------------------------------------


def test_filter_unset_drops_user_set_leaves() -> None:
    overrides = {"solver": {"base_lr": 0.01, "max_iter": 1000}}
    out = filter_unset(overrides, {"solver.base_lr"})
    assert out == {"solver": {"max_iter": 1000}}


def test_filter_unset_prunes_empty_subdicts() -> None:
    overrides = {"solver": {"base_lr": 0.01}}
    out = filter_unset(overrides, {"solver.base_lr"})
    # Whole `solver` section disappears since its only entry was
    # user-set — the result is the smallest payload merge_overrides
    # would receive.
    assert out == {}


def test_filter_unset_passes_through_when_no_overlap() -> None:
    overrides = {"solver": {"max_iter": 1000}}
    out = filter_unset(overrides, {"input.color_jitter_prob"})
    assert out == overrides


def test_filter_unset_handles_deep_nesting() -> None:
    overrides = {
        "model": {
            "anchor_generator": {"sizes": [[32]], "aspect_ratios": [[0.5]]},
            "roi_heads": {"num_classes": 5},
        }
    }
    out = filter_unset(overrides, {"model.anchor_generator.sizes"})
    assert out == {
        "model": {
            "anchor_generator": {"aspect_ratios": [[0.5]]},
            "roi_heads": {"num_classes": 5},
        }
    }
