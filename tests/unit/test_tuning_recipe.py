"""Tests for :mod:`mayaku.tuning.recipe`."""

from __future__ import annotations

import itertools
import math

import pytest

from mayaku.config import MayakuConfig
from mayaku.tuning.dataset_stats import DatasetStats
from mayaku.tuning.recipe import (
    ARCHITECTURE_TUNED_PATHS,
    FINETUNE_BACKBONE_LR_RATIO,
    FINETUNE_LR_MAX,
    FINETUNE_LR_MIN,
    MAX_FINETUNE_EPOCHS,
    MIN_BOXES_FOR_ANCHOR_TUNE,
    MIN_FINETUNE_EPOCHS,
    MIN_IMAGES_FOR_AUTO_CONFIG,
    REFERENCE_BATCH,
    collect_set_paths,
    derive_overrides,
    filter_unset,
    finetune_base_lr,
    finetune_llrd_decay,
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


def _uniquery_cfg(hidden_dim: int = 256, optimizer: str = "SGD") -> MayakuConfig:
    return MayakuConfig.model_validate(
        {
            "model": {
                "meta_architecture": "uniquery",
                "fpn": {"out_channels": hidden_dim},
                "uniquery_head": {"hidden_dim": hidden_dim},
            },
            "solver": {"optimizer_name": optimizer},
        }
    )


# ---------------------------------------------------------------------------
# The auto-config contract: never re-tune the model
# ---------------------------------------------------------------------------


def test_derive_overrides_never_emits_architecture_tuned_fields() -> None:
    # THE invariant: the recipe adapts the run to the dataset; the config
    # that travels with the weights owns freeze_at / EMA. A table edit that
    # re-adds any of these must fail here. (base_lr is deliberately NOT in
    # the frozen set — it's regime-dependent and IS emitted; see below.)
    for cfg in (MayakuConfig(), _uniquery_cfg()):
        for n in (100, 1_500, 10_000, 100_000):
            overrides = derive_overrides(_stats(num_images=n), cfg)
            emitted = {p for p, _ in walk_leaves(overrides)}
            assert not emitted & ARCHITECTURE_TUNED_PATHS, (
                f"recipe emitted architecture-tuned field(s) "
                f"{sorted(emitted & ARCHITECTURE_TUNED_PATHS)} at n={n}"
            )


def test_derive_overrides_emits_finetune_base_lr() -> None:
    # base_lr is recipe-emitted (not inherited from the checkpoint) and comes
    # from the finetune_base_lr law; always inside the safe clamp band.
    cfg = MayakuConfig()
    overrides = derive_overrides(_stats(num_images=2_000), cfg)
    lr = overrides["solver"]["base_lr"]
    assert FINETUNE_LR_MIN <= lr <= FINETUNE_LR_MAX


def test_base_lr_slides_down_with_dataset_size_and_tier() -> None:
    # Monotonicity contract: bigger dataset -> lower LR (size glide); wider head
    # (bigger tier) -> lower LR (anchor). Both verified through derive_overrides.
    def lr_for(num_images: int, hidden: int) -> float:
        cfg = _uniquery_cfg(hidden_dim=hidden, optimizer="AdamW")
        return derive_overrides(_stats(num_images=num_images), cfg)["solver"]["base_lr"]

    # size glide: 300-img (small) >= 5000-img (large) at the same tier
    assert lr_for(300, 128) >= lr_for(5_000, 128)
    # tier anchor: nano (128) >= wide (256) at the same dataset size
    assert lr_for(1_000, 128) >= lr_for(1_000, 256)


def test_finetune_base_lr_batch_scaling_ratio() -> None:
    # SGD scales linearly with effective batch, AdamW sqrt — so at 2x batch the
    # SGD/AdamW ratio is sqrt(2). Use a mid-band tier (256) so neither hits the
    # clamp (which would distort the ratio).
    sgd = finetune_base_lr(1_500, 256, eff_batch=2 * REFERENCE_BATCH, adamw=False)
    adamw = finetune_base_lr(1_500, 256, eff_batch=2 * REFERENCE_BATCH, adamw=True)
    assert sgd / adamw == pytest.approx(math.sqrt(2), rel=1e-6)


@pytest.mark.parametrize(
    "kw",
    [
        {"num_images": -5, "hidden_dim": 128},  # negative images + int() guard
        {"num_images": 10**12, "hidden_dim": 128},  # huge dataset
        {"num_images": 0, "hidden_dim": 128},  # zero images -> max(1, .) in glide
        {"num_images": 500, "hidden_dim": -5},  # negative width -> narrow anchor
        {"num_images": 500, "hidden_dim": 99999},  # unseen wide width -> wide anchor
        {"num_images": float("nan"), "hidden_dim": 128},  # non-finite -> safe default
        {"num_images": 500, "hidden_dim": 128, "eff_batch": 100000},  # upper batch clamp
        {"num_images": 500, "hidden_dim": 128, "eff_batch": 0},  # zero batch -> max(1, .)
    ],
)
def test_finetune_base_lr_is_always_bounded_and_never_raises(kw: dict) -> None:
    # A wrong LR breaks training: no input, however degenerate, may escape the
    # clamp or raise. This is the safety contract.
    lr = finetune_base_lr(**kw)
    assert math.isfinite(lr)
    assert FINETUNE_LR_MIN <= lr <= FINETUNE_LR_MAX


# ---------------------------------------------------------------------------
# Hot-head / cold-backbone LLRD decay
# ---------------------------------------------------------------------------


def _llrd_cfg(backbone: str = "convnext_femto", llrd_enabled: bool = True) -> MayakuConfig:
    # UniQuery + a ConvNeXt backbone, the shape the checkpoint sidecar carries on
    # the mayaku fine-tune path (llrd_enabled baked true). fpn.out_channels must
    # equal uniquery_head.hidden_dim.
    return MayakuConfig.model_validate(
        {
            "model": {
                "meta_architecture": "uniquery",
                "backbone": {"name": backbone},
                "fpn": {"out_channels": 128},
                "uniquery_head": {"hidden_dim": 128},
            },
            "solver": {"llrd_enabled": llrd_enabled, "optimizer_name": "AdamW"},
        }
    )


def test_finetune_llrd_decay_lands_stem_at_ratio() -> None:
    # The whole point: decay ** (num_layers + 1) == the head->stem ratio, for any
    # depth. This is the invariant every emitted decay must satisfy.
    for num_layers in (4, 6, 12):
        decay = finetune_llrd_decay(num_layers)
        assert decay ** (num_layers + 1) == pytest.approx(FINETUNE_BACKBONE_LR_RATIO)


def test_derive_overrides_emits_depth_conditioned_llrd_decay() -> None:
    # Shallow backbones (num_layers 6) get a steeper decay than deep ones
    # (num_layers 12), both anchored to the same 1/10 head->stem ratio.
    shallow = derive_overrides(_stats(), _llrd_cfg("convnext_femto"))["solver"]["llrd_decay"]
    deep = derive_overrides(_stats(), _llrd_cfg("convnext_large"))["solver"]["llrd_decay"]
    assert shallow == pytest.approx(0.1 ** (1 / 7))  # ~0.720
    assert deep == pytest.approx(0.1 ** (1 / 13))  # ~0.838
    assert shallow < deep  # shallower net -> steeper per-layer decay


def test_derive_overrides_no_llrd_decay_when_llrd_disabled() -> None:
    # The recipe delivers the split THROUGH LLRD; if the config doesn't run
    # LLRD, the recipe must not silently turn it on via a decay override.
    overrides = derive_overrides(_stats(), _llrd_cfg("convnext_femto", llrd_enabled=False))
    assert "llrd_decay" not in overrides["solver"]
    # The default (resnet, SGD, llrd off) config likewise emits none.
    assert "llrd_decay" not in derive_overrides(_stats(), MayakuConfig())["solver"]


def test_llrd_decay_is_overridable_by_user() -> None:
    # A pinned llrd_decay must survive auto-config, like any explicit value.
    overrides = derive_overrides(_stats(), _llrd_cfg("convnext_femto"))
    filtered = filter_unset(overrides, {"solver.llrd_decay"})
    assert "llrd_decay" not in filtered["solver"]


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
