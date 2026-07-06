"""Tests for :mod:`mayaku.tuning.recipe`."""

from __future__ import annotations

import itertools
import math

from mayaku.config import MayakuConfig
from mayaku.tuning.dataset_stats import DatasetStats
from mayaku.tuning.recipe import (
    ARCHITECTURE_TUNED_PATHS,
    MAX_FINETUNE_EPOCHS,
    MIN_BOXES_FOR_ANCHOR_TUNE,
    MIN_FINETUNE_EPOCHS,
    MIN_IMAGES_FOR_AUTO_CONFIG,
    collect_set_paths,
    derive_overrides,
    filter_unset,
    size_bucket,
    walk_leaves,
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


def _uniquery_cfg() -> MayakuConfig:
    return MayakuConfig.model_validate(
        {"model": {"meta_architecture": "uniquery", "uniquery_head": {}}}
    )


# ---------------------------------------------------------------------------
# The auto-config contract: never re-tune the model
# ---------------------------------------------------------------------------


def test_derive_overrides_never_emits_architecture_tuned_fields() -> None:
    # THE invariant: the recipe adapts the run to the dataset; the config
    # that travels with the weights owns base_lr / freeze_at / EMA. A
    # table edit that re-adds any of these must fail here.
    for cfg in (MayakuConfig(), _uniquery_cfg()):
        for n in (100, 1_500, 10_000, 100_000):
            overrides = derive_overrides(_stats(num_images=n), cfg)
            emitted = {p for p, _ in walk_leaves(overrides)}
            assert not emitted & ARCHITECTURE_TUNED_PATHS, (
                f"recipe emitted architecture-tuned field(s) "
                f"{sorted(emitted & ARCHITECTURE_TUNED_PATHS)} at n={n}"
            )


def test_anchor_overrides_gated_to_anchor_consuming_archs() -> None:
    # UniQuery has no anchor generator — emitting anchors would merge
    # validly and pollute the checkpoint sidecar.
    stats = _stats(num_boxes=MIN_BOXES_FOR_ANCHOR_TUNE)
    assert "anchor_generator" not in derive_overrides(stats, _uniquery_cfg()).get("model", {})
    assert "anchor_generator" in derive_overrides(stats, MayakuConfig())["model"]


def test_derive_overrides_empty_for_tiny_datasets() -> None:
    stats = _stats(num_images=MIN_IMAGES_FOR_AUTO_CONFIG - 1, num_boxes=5)
    assert derive_overrides(stats, MayakuConfig()) == {}


def test_derive_overrides_does_not_set_num_classes() -> None:
    # num_classes is structural (derived from the dataset unconditionally by
    # run_train), not a tuning heuristic — auto-config must NOT emit it.
    stats = _stats(num_classes=12)
    overrides = derive_overrides(stats, MayakuConfig())
    assert "roi_heads" not in overrides.get("model", {})


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
    assert "anchor_generator" not in overrides.get("model", {})


def test_derive_overrides_emits_epoch_budget() -> None:
    stats = _stats(num_images=2_000)
    overrides = derive_overrides(stats, MayakuConfig())
    # Schedule length is a dataset-adaptive epoch budget (resolved to iters
    # at train time), not raw iteration counts.
    assert overrides["solver"]["num_epochs"] > 0
    assert "max_iter" not in overrides["solver"]
    assert "steps" not in overrides["solver"]


def test_epoch_budget_total_steps_monotone_in_dataset_size() -> None:
    # The old per-bucket epoch table trained a 515-image set for FEWER
    # total steps than a 478-image one (a 34% cliff at the 500 boundary).
    # Total work must be non-decreasing in dataset size, up to the
    # ±one-epoch wobble that ceil() rounding makes unavoidable.
    cfg = MayakuConfig()  # effective batch 16
    batch = cfg.solver.effective_batch()

    def total_steps(n: int) -> int:
        epochs = derive_overrides(_stats(num_images=n), cfg)["solver"]["num_epochs"]
        return epochs * math.ceil(n / batch)

    sizes = (100, 478, 500, 515, 2_000, 5_000, 50_000)
    steps = [total_steps(n) for n in sizes]
    for (n_prev, s_prev), (n_next, s_next) in itertools.pairwise(zip(sizes, steps)):
        one_epoch = math.ceil(n_next / batch)
        assert s_next >= s_prev - one_epoch, (
            f"total steps dropped across {n_prev}->{n_next} images: {s_prev}->{s_next}"
        )


def test_epoch_budget_clamped_at_both_ends() -> None:
    cfg = MayakuConfig()
    # Tiny dataset: target steps would demand hundreds of epochs → MAX.
    tiny = derive_overrides(_stats(num_images=50), cfg)["solver"]["num_epochs"]
    assert tiny == MAX_FINETUNE_EPOCHS
    # Huge dataset: one epoch already exceeds the target → MIN.
    huge = derive_overrides(_stats(num_images=200_000), cfg)["solver"]["num_epochs"]
    assert huge == MIN_FINETUNE_EPOCHS


def test_multi_sample_aug_stays_conservative_on_small_datasets() -> None:
    # Deliberate: early A/Bs showed mosaic hurting small-set fine-tunes,
    # so it stays LOW below 2k images until the benchmark sweep re-tests
    # the direction. If you're here to raise it, bring sweep evidence.
    assert derive_overrides(_stats(num_images=100), MayakuConfig())["input"]["mosaic_prob"] == 0.1
    assert derive_overrides(_stats(num_images=1_500), MayakuConfig())["input"]["mosaic_prob"] == 0.2
    assert (
        derive_overrides(_stats(num_images=10_000), MayakuConfig())["input"]["mosaic_prob"] == 0.5
    )


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
    # num_classes is set by run_train (structural), not by derive_overrides.
    assert merged.model.anchor_generator.sizes != ((32,), (64,), (128,), (256,), (512,))
    assert merged.solver.num_epochs > 0
    assert merged.solver.ema_enabled is True


# ---------------------------------------------------------------------------
# User-set path tracking
# ---------------------------------------------------------------------------


def test_collect_set_paths_flat() -> None:
    assert collect_set_paths({"a": 1, "b": 2}) == {"a", "b"}


def test_collect_set_paths_nested() -> None:
    paths = collect_set_paths({"solver": {"base_lr": 0.01, "num_epochs": 30}})
    assert paths == {"solver.base_lr", "solver.num_epochs"}


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
    overrides = {"solver": {"base_lr": 0.01, "num_epochs": 30}}
    out = filter_unset(overrides, {"solver.base_lr"})
    assert out == {"solver": {"num_epochs": 30}}


def test_filter_unset_prunes_empty_subdicts() -> None:
    overrides = {"solver": {"base_lr": 0.01}}
    out = filter_unset(overrides, {"solver.base_lr"})
    # Whole `solver` section disappears since its only entry was
    # user-set — the result is the smallest payload merge_overrides
    # would receive.
    assert out == {}


def test_filter_unset_passes_through_when_no_overlap() -> None:
    overrides = {"solver": {"num_epochs": 30}}
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
