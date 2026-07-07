"""Recipe lookup + override derivation for auto-config.

Maps :class:`DatasetStats` plus the user's base :class:`MayakuConfig`
into a nested override dict that
:func:`mayaku.config.merge_overrides` can apply.

The contract (the auto-config invariant): **auto-config must deliver
results >= the tuned config that travels with the weights.** The recipe
therefore adapts the run to the dataset — it never re-tunes the model.
It only ever emits three kinds of fields:

* **dataset-structural facts** — anchor sizes/ARs (for anchor-consuming
  architectures, measured in the pipeline's actual input frame) and the
  RepeatFactor sampler switch;
* **run-scoped budgets** — ``num_epochs`` (a target-total-steps budget)
  and multi-sample augmentation strength;
* **the fine-tune learning rate** — ``base_lr`` is *regime-dependent*,
  not architecture-tuned. The checkpoint bakes the *pretraining* LR
  (1e-4 AdamW, right for COCO-scale data); fine-tuning from those
  weights to a new dataset wants ~10x more to move off the pretrained
  basin in a short run (benchmark-confirmed: inheriting the baked 1e-4
  regressed every far-from-COCO set). The recipe emits a flat fine-tune
  default (:data:`FINETUNE_BASE_LR`), batch-scaled off its validated
  anchor; a pinned ``base_lr`` (YAML/overrides) still wins via
  :func:`filter_unset`. When the config runs LLRD, ``base_lr`` is the
  *head* LR and the recipe also emits a depth-adjusted ``llrd_decay``
  (:func:`finetune_llrd_decay`) that ramps the backbone down to ~1/10 of
  it — the hot-head/cold-backbone split (see
  :data:`FINETUNE_BACKBONE_LR_RATIO`).

Genuinely architecture-tuned hyperparameters — ``freeze_at``, the EMA
constants, optimizer settings, loss weights — are NEVER emitted: the
checkpoint's embedded config owns them ("weights carry everything").
That contract is pinned by :data:`ARCHITECTURE_TUNED_PATHS` and its
test; add to the set before adding any solver/model knob to the recipe,
and it will fail the contract test — by design.

Two ancillary helpers travel with the recipe to keep the wiring in
``run_train`` short:

* :func:`collect_set_paths` walks a raw parsed YAML dict and yields
  dotted paths of every leaf the user wrote — used to tag which fields
  must NOT be auto-overridden.
* :func:`filter_unset` drops paths from a proposed override dict when
  the user already set them in YAML.

Together they preserve the load-bearing rule: **explicit user values
always win**.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final

from mayaku.config.schemas import MayakuConfig
from mayaku.tuning.anchor_kmeans import cluster_aspect_ratios, cluster_sizes
from mayaku.tuning.dataset_stats import DatasetStats

__all__ = [
    "ARCHITECTURE_TUNED_PATHS",
    "FINETUNE_BACKBONE_LR_RATIO",
    "FINETUNE_BASE_LR",
    "MAX_FINETUNE_EPOCHS",
    "MIN_BOXES_FOR_ANCHOR_TUNE",
    "MIN_FINETUNE_EPOCHS",
    "MIN_IMAGES_FOR_AUTO_CONFIG",
    "REFERENCE_BATCH",
    "SizeBucket",
    "collect_set_paths",
    "derive_overrides",
    "filter_unset",
    "finetune_llrd_decay",
    "size_bucket",
    "walk_leaves",
]

# Dotted paths the recipe is FORBIDDEN from emitting. These are tuned
# per-architecture and shipped inside the checkpoint sidecar; a
# dataset-size table overriding them is a bug class (e.g. a frozen RGB
# stem forced onto a modality that needs to adapt). Enforced by the
# contract test in tests/unit/test_tuning_recipe.py.
# NB: base_lr is deliberately NOT here — it's regime-dependent, so the
# recipe emits a fine-tune default instead. See FINETUNE_BASE_LR.
ARCHITECTURE_TUNED_PATHS: Final = frozenset(
    {
        "solver.ema_enabled",
        "solver.ema_decay",
        "solver.ema_tau",
        "model.backbone.freeze_at",
    }
)

# Meta-architectures whose model actually reads anchor_generator.
# UniQuery is query-based (no anchors) — emitting anchor overrides for
# it would merge validly and pollute the checkpoint sidecar.
ANCHOR_CONSUMING_ARCHS: Final = frozenset({"faster_rcnn", "mask_rcnn", "keypoint_rcnn"})

# Below this image count there's not enough signal to derive a sensible
# recipe — we'd either fit anchors to a few outliers or pick a
# fine-tune schedule from too small a sample. Caller skips auto-config
# entirely below the threshold.
MIN_IMAGES_FOR_AUTO_CONFIG: Final = 10

# Anchor k-means needs enough boxes to populate 5 (sizes) and 3 (ARs)
# clusters meaningfully. Below this we leave the base config's anchors
# alone rather than fit to noise.
MIN_BOXES_FOR_ANCHOR_TUNE: Final = 50

# Class-imbalance ratio above this triggers RepeatFactorTrainingSampler.
# Threshold of 10× is the empirical "the head class drowns out the
# tail" point — Gupta et al. LVIS uses similar reasoning at 100×, but
# detection datasets are usually less skewed than instance-seg LVIS.
_IMBALANCE_TRIGGER: Final = 10.0

# Fine-tune schedule: aim for a fixed total-optimizer-steps budget so
# total training work is monotone in dataset size (the old per-bucket
# epoch table trained a 515-image set for FEWER steps than a 478-image
# one). epochs = clamp(ceil(target / iters_per_epoch), min, max) —
# at effective batch 16 that's clamp(48_000 / n_images, 16, 30):
#   * <= ~1,600 images: MAX binds (30 epochs — mayaku saturates around
#     30 epochs empirically; more is wasted wall-clock even though small
#     sets make extra epochs cheap in steps),
#   * ~1,600-3,000 images: the target's taper (30 -> 16 epochs, flat
#     ~3k steps),
#   * >= ~3,000 images: MIN binds (16 epochs — enough passes to learn,
#     steps grow with data again).
# The [16, 30] band is deliberately narrow — near fixed-epochs like the
# competitors, but wall-time bounded. Knee to be re-swept on the benchmark.
TARGET_TOTAL_STEPS: Final = 3_000
MIN_FINETUNE_EPOCHS: Final = 16
MAX_FINETUNE_EPOCHS: Final = 30

# Fine-tune learning rate. base_lr is regime-dependent: the checkpoint
# bakes the PRETRAINING LR (1e-4 AdamW, right for COCO-scale data), but
# fine-tuning to a new dataset from those weights wants ~10x more to move
# the head+backbone off the pretrained basin within a short run. 1e-3 is
# benchmark-validated (recovers the far-from-COCO wins that 1e-4
# regressed; near-ceiling sets prefer a touch less — a domain-adaptive
# refinement tracked separately). LLRD still scales this down per backbone
# depth. Emitted through filter_unset, so a pinned base_lr always wins.
FINETUNE_BASE_LR: Final = 1.0e-3
# Effective batch FINETUNE_BASE_LR was validated at; the anchor for the
# LR<->batch scaling below. Matches the family configs' 4x4 = 16.
REFERENCE_BATCH: Final = 16

# Hot-head / cold-backbone fine-tune split, delivered through LLRD's existing
# per-layer ramp (no new solver knob). The re-initialised head/neck sit at
# base_lr (LLRD scale 1.0); the recipe picks llrd_decay so the input-most
# backbone layer (LLRD layer_id 0 — the stem) lands at FINETUNE_BACKBONE_LR_RATIO
# of the head LR. That is the ~10x discriminative-fine-tune separation (ULMFiT):
# the random head learns fast while the pretrained backbone barely moves. The
# 10x head->stem ratio is benchmark-validated on the smallest model; the
# deeper-scale decays it implies are principled starting points to sweep.
#
# The stem's LLRD scale is decay ** (num_layers + 1): from
# engine.optim._build_llrd_groups, scale = decay ** ((num_layers + 2) - layer_id
# - 1), and layer_id 0 gives the exponent num_layers + 1. Solving
# decay ** (num_layers + 1) = ratio for the decay gives finetune_llrd_decay()
# below. Because num_layers is set by backbone DEPTH, one fixed head->stem ratio
# yields a steeper decay for shallow backbones and a shallower one for deep ones
# — the depth-invariant way to port the ratio across model scales (at ratio=0.1:
# num_layers 6 -> 0.720, num_layers 12 -> 0.838). NB: this is a *ramp*, so the
# upper backbone stages train well above the ratio (only the stem hits it) — a
# hotter-backbone regime than a flat 1/10 split; confirm on the smallest model.
FINETUNE_BACKBONE_LR_RATIO: Final = 0.1

# ConvNeXt "res4" (== MMDet stage-2) block count per variant — the sole input to
# LLRD's num_layers bucketing (<= 9 blocks -> 6, else 12). Torch-free mirror of
# the depths in models/backbones/convnext.py; the resulting num_layers MUST
# match engine.optim._resolve_llrd_num_layers (which derives it from the built
# model). Kept honest by test_llrd_finetune_num_layers_matches_engine.
_CONVNEXT_STAGE2_BLOCKS: Final = {
    "convnext_atto": 6,
    "convnext_femto": 6,
    "convnext_pico": 6,
    "convnext_nano": 8,
    "convnext_tiny": 9,
    "convnext_small": 27,
    "convnext_base": 27,
    "convnext_large": 27,
}
# ResNet/ResNeXt LLRD depth: stem(0) + res{2,3,4,5}(1..4). Fixed at 4, matching
# engine.optim._resolve_llrd_num_layers's resnet branch.
_RESNET_LLRD_NUM_LAYERS: Final = 4


def finetune_llrd_decay(
    num_layers: int, ratio: float = FINETUNE_BACKBONE_LR_RATIO
) -> float:
    """LLRD decay putting the stem (layer_id 0) at ``ratio`` x the head LR.

    Inverts the stem's LLRD scale ``decay ** (num_layers + 1)`` (see
    :data:`FINETUNE_BACKBONE_LR_RATIO`). ``num_layers`` is the backbone's LLRD
    depth as resolved by :func:`_llrd_num_layers`.
    """
    return float(ratio ** (1.0 / (num_layers + 1)))


def _llrd_num_layers(backbone_name: str) -> int | None:
    """LLRD ``num_layers`` for a backbone variant, or ``None`` if unsupported.

    Mirrors :func:`mayaku.engine.optim._resolve_llrd_num_layers` from the name
    alone — the tuning package stays torch-free and can't build the model to
    count blocks. The two are cross-checked by
    ``test_llrd_finetune_num_layers_matches_engine``.
    """
    if backbone_name.startswith("convnext_"):
        # The name is a closed BackboneName Literal and the table is proven
        # exhaustive over convnext variants (test_recipe_llrd_stage2_table_
        # covers_all_convnext_variants), so a missing key is a genuine bug —
        # let it KeyError loudly rather than silently drop the decay.
        return 6 if _CONVNEXT_STAGE2_BLOCKS[backbone_name] <= 9 else 12
    if backbone_name.startswith(("resnet", "resnext")):
        return _RESNET_LLRD_NUM_LAYERS
    return None


@dataclass(frozen=True)
class SizeBucket:
    """A row of the size-keyed recipe table: multi-sample augmentation
    strength only. Run length comes from the target-steps budget and
    everything architecture-tuned comes from the config itself."""

    name: str
    mosaic_prob: float
    mixup_prob: float
    copy_paste_prob: float


_BUCKETS: Final[tuple[tuple[float, SizeBucket], ...]] = (
    # (upper_exclusive_image_count, bucket). Last entry uses sentinel
    # math.inf so any larger dataset still falls into a bucket.
    # Mosaic kept LOW below 2k images: early A/Bs showed multi-sample aug
    # hurting small-set fine-tunes. Those runs predate the head/recipe
    # fixes, so the direction is UNSETTLED — re-A/B on the benchmark
    # sweep before raising these (report-1 Tier 1); stay conservative
    # until then.
    (500, SizeBucket("xs", mosaic_prob=0.1, mixup_prob=0.0, copy_paste_prob=0.0)),
    (2_000, SizeBucket("s", mosaic_prob=0.2, mixup_prob=0.0, copy_paste_prob=0.0)),
    (5_000, SizeBucket("m", mosaic_prob=0.3, mixup_prob=0.0, copy_paste_prob=0.0)),
    (50_000, SizeBucket("l", mosaic_prob=0.5, mixup_prob=0.1, copy_paste_prob=0.0)),
    (math.inf, SizeBucket("xl", mosaic_prob=0.5, mixup_prob=0.1, copy_paste_prob=0.1)),
)


def size_bucket(num_images: int) -> SizeBucket:
    """Pick the recipe row for a given dataset size."""
    for upper, bucket in _BUCKETS:
        if num_images < upper:
            return bucket
    # Unreachable because the last bucket's upper is math.inf, but
    # mypy doesn't know that.
    return _BUCKETS[-1][1]


# ---------------------------------------------------------------------------
# Override derivation
# ---------------------------------------------------------------------------


def derive_overrides(
    stats: DatasetStats,
    cfg: MayakuConfig,
) -> dict[str, Any]:
    """Produce a nested override dict from dataset stats + base config.

    The returned dict mirrors :class:`MayakuConfig`'s shape (top-level
    keys are ``model``, ``solver``, ``input``, ``dataloader``) and can
    be passed straight to :func:`mayaku.config.merge_overrides`. The caller
    passes the result through :func:`filter_unset` so any field the user pinned
    is dropped — explicit user values always win.

    Never emits a path in :data:`ARCHITECTURE_TUNED_PATHS` — see the
    module docstring for the contract.

    Returns an empty dict when ``stats.num_images <
    MIN_IMAGES_FOR_AUTO_CONFIG`` — the dataset is too small to drive a
    sensible recipe.
    """
    if stats.num_images < MIN_IMAGES_FOR_AUTO_CONFIG:
        return {}

    bucket = size_bucket(stats.num_images)

    # ----- model -----
    # NB: num_classes is NOT set here — it's a structural fact of the dataset,
    # not a tuning heuristic, so run_train derives it from the COCO categories
    # whenever auto-config is enabled, at any dataset size (bypassing the
    # MIN_IMAGES_FOR_AUTO_CONFIG floor; with auto-config off the config is used
    # verbatim).
    model_overrides: dict[str, Any] = {}
    if (
        cfg.model.meta_architecture in ANCHOR_CONSUMING_ARCHS
        and stats.num_boxes >= MIN_BOXES_FOR_ANCHOR_TUNE
    ):
        sizes = cluster_sizes(stats.sqrt_areas, k=5)
        ars = cluster_aspect_ratios(stats.aspect_ratios, k=3)
        model_overrides["anchor_generator"] = {
            "sizes": tuple((s,) for s in sizes),
            "aspect_ratios": (ars,),
        }

    # ----- solver -----
    # Two solver fields, both anchored to the single-rank effective batch
    # (world_size excluded by design — recipe runs pre-DDP):
    #   * base_lr — the regime-dependent fine-tune LR (see FINETUNE_BASE_LR),
    #     scaled off its validated batch-16 anchor to this run's effective
    #     batch: linear for SGD, sqrt for AdamW (gradient-noise scaling).
    #   * num_epochs — a target-total-steps budget resolved to epochs (the
    #     engine resolves epochs back to iterations at train time).
    # Same epoch definition as resolve_schedule (engine/optim.py) — keep the
    # formulas in sync; not imported so the tuning package stays torch-free.
    eff_batch = cfg.solver.effective_batch()
    iters_per_epoch = max(1, math.ceil(stats.num_images / max(1, eff_batch)))
    num_epochs = min(
        MAX_FINETUNE_EPOCHS,
        max(MIN_FINETUNE_EPOCHS, math.ceil(TARGET_TOTAL_STEPS / iters_per_epoch)),
    )
    lr_ratio = eff_batch / REFERENCE_BATCH
    lr_factor = lr_ratio if cfg.solver.optimizer_name == "SGD" else math.sqrt(lr_ratio)
    solver_overrides: dict[str, Any] = {
        "base_lr": FINETUNE_BASE_LR * lr_factor,
        "num_epochs": num_epochs,
    }
    # Hot-head/cold-backbone split via LLRD (see FINETUNE_BACKBONE_LR_RATIO).
    # Only when the checkpoint config actually runs LLRD (the mayaku family
    # does): pick a depth-adjusted decay so the stem lands at ~1/10 the head LR.
    # base_lr (above) is the head LR; this decay ramps the backbone below it.
    # Like base_lr, the recipe owns the fine-tune value — the config's baked
    # llrd_decay is overridden, and filter_unset still lets a pinned one win.
    if cfg.solver.llrd_enabled:
        num_layers = _llrd_num_layers(cfg.model.backbone.name)
        if num_layers is not None:
            solver_overrides["llrd_decay"] = finetune_llrd_decay(num_layers)

    # ----- input augs -----
    input_overrides: dict[str, Any] = {
        "mosaic_prob": bucket.mosaic_prob,
        "mixup_prob": bucket.mixup_prob,
    }
    # CopyPaste needs mask_format='bitmask' AND a mask-bearing meta-
    # architecture. Only emit when the user is actually training masks.
    if cfg.model.meta_architecture == "mask_rcnn" and cfg.input.mask_format == "bitmask":
        input_overrides["copy_paste_prob"] = bucket.copy_paste_prob

    overrides: dict[str, Any] = {
        "solver": solver_overrides,
        "input": input_overrides,
    }
    if model_overrides:
        overrides["model"] = model_overrides

    # ----- sampler -----
    if stats.class_imbalance > _IMBALANCE_TRIGGER:
        overrides["dataloader"] = {
            "sampler_train": "RepeatFactorTrainingSampler",
            "repeat_threshold": 0.01,
        }

    return overrides


# ---------------------------------------------------------------------------
# User-set path tracking
# ---------------------------------------------------------------------------


def walk_leaves(payload: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ``(dotted_path, value)`` for every leaf in a nested mapping.

    Leaves are non-mapping values (scalars, lists, tuples). Used by
    :func:`collect_set_paths` and by the auto-config report printer.
    """
    if isinstance(payload, Mapping):
        for k, v in payload.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, Mapping):
                yield from walk_leaves(v, path)
            else:
                yield path, v


def collect_set_paths(raw: Any) -> set[str]:
    """Set of dotted paths for every leaf in a parsed YAML dict.

    Example::

        >>> collect_set_paths({"solver": {"base_lr": 0.01}})
        {'solver.base_lr'}
    """
    return {p for p, _ in walk_leaves(raw)}


def filter_unset(
    overrides: Mapping[str, Any],
    user_set_paths: Collection[str],
    prefix: str = "",
) -> dict[str, Any]:
    """Drop entries from ``overrides`` whose dotted path is user-set.

    Recurses into nested mappings. Returns a new dict; ``overrides`` is
    unchanged. Empty sub-dicts are pruned so the result is the smallest
    possible payload to feed :func:`merge_overrides`.
    """
    result: dict[str, Any] = {}
    for k, v in overrides.items():
        path = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            sub = filter_unset(v, user_set_paths, path)
            if sub:
                result[k] = sub
        elif path not in user_set_paths:
            result[k] = v
    return result
