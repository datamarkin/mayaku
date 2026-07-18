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
  default (:data:`FINETUNE_BASE_LR`, the regret-robust plateau centre;
  see :func:`finetune_base_lr`), batch-scaled off its validated batch-16
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
    "FINETUNE_GRAD_ACCUM_STEPS",
    "FINETUNE_IMS_PER_BATCH",
    "FINETUNE_LR_MAX",
    "FINETUNE_LR_MIN",
    "MAX_FINETUNE_EPOCHS",
    "MIN_BOXES_FOR_ANCHOR_TUNE",
    "MIN_FINETUNE_EPOCHS",
    "MIN_IMAGES_FOR_AUTO_CONFIG",
    "REFERENCE_BATCH",
    "SizeBucket",
    "collect_set_paths",
    "derive_overrides",
    "filter_unset",
    "finetune_base_lr",
    "finetune_llrd_decay",
    "finetune_num_epochs",
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

# Fine-tune schedule: a smooth log taper on epochs — MAX passes on small
# datasets down to MIN on large ones, log-linear across the taper's image band
# (see finetune_num_epochs). Small sets get many
# passes (cheap in steps, and they need them); large sets get few (expensive,
# and they saturate). Total training work stays monotone in dataset size — image
# count grows faster than the epoch count shrinks — which the old per-bucket
# table violated (it trained a 515-image set for FEWER steps than a 478 one):
#   * <= 1,000 images: MAX binds (30 epochs),
#   * 1,000-20,000 images: smooth 30 -> 16 taper (no hard cliff),
#   * >= 20,000 images: MIN binds (16 epochs, steps grow with data).
# The [16, 30] band is deliberately narrow — near fixed-epochs like the
# competitors, but wall-time bounded. Knee to be re-swept on the benchmark.
MIN_FINETUNE_EPOCHS: Final = 16
MAX_FINETUNE_EPOCHS: Final = 30
_EPOCH_TAPER_IMG_LO: Final = 1_000
_EPOCH_TAPER_IMG_HI: Final = 20_000

# Fine-tune learning rate. base_lr is regime-dependent: the checkpoint
# bakes the PRETRAINING LR (1e-4 AdamW, right for COCO-scale data), but
# fine-tuning to a new dataset from those weights wants more to move the
# head+backbone off the pretrained basin within a short run. 3e-4 is the
# regret-robust plateau centre — flat across dataset size and head width, so
# base_lr is a single fine-tune default (no size glide, no per-width anchor).
# LLRD still scales it down per backbone depth. Emitted via filter_unset, so a
# pin always wins.
FINETUNE_BASE_LR: Final = 3.0e-4
# Effective batch FINETUNE_BASE_LR was validated at; the anchor for the
# LR<->batch scaling below. Matches the family configs' 4x4 = 16.
REFERENCE_BATCH: Final = 16

# Fine-tune micro-batch layout: split REFERENCE_BATCH into a smaller per-step
# batch so the default memory footprint is safe on modest GPUs, while the
# effective batch (and thus the LR/epoch derivation) is unchanged. grad_accum is
# derived so ims_per_batch * grad_accum == REFERENCE_BATCH by construction. A
# memory-profiled per-host value is a future enhancement.
FINETUNE_IMS_PER_BATCH: Final = 4
FINETUNE_GRAD_ACCUM_STEPS: Final = REFERENCE_BATCH // FINETUNE_IMS_PER_BATCH

# ---------------------------------------------------------------------------
# Fine-tune base-LR law: a flat plateau-centre constant (FINETUNE_BASE_LR),
# batch-scaled and HARD-CLAMPED to a band where no swept dataset ever cratered.
#
# Design contract: the fine-tune optimum is a broad flat plateau — every LR in
# [3e-4, 1e-3] lands within ~1 AP of the per-dataset best, and where in that
# plateau a dataset's peak sits is not predictable from size, head width, or
# object scale (the residual is domain). So the law targets the regret-ROBUST
# plateau centre (3e-4, on the safe side — the plateau is skewed, under-shooting
# costs ~3x an equal over-shoot), never per-dataset peaks. No size glide, no
# per-width anchor.
# It is fail-safe: every path returns a finite value inside [LR_MIN, LR_MAX],
# bad input collapses to a safe band-centre, and nothing raises. A wrong LR
# breaks training, so the clamp is the last line of defence and is never bypassed.
# ---------------------------------------------------------------------------

# Hard output clamp — the safe zone. Below LR_MIN undertrains, above LR_MAX
# blows up; no dataset's optimum fell outside this across both tiers.
FINETUNE_LR_MIN: Final = 5.0e-5
FINETUNE_LR_MAX: Final = 1.5e-3
# Returned on any degenerate/unexpected input (centre of the flat plateau).
FINETUNE_LR_SAFE_DEFAULT: Final = 5.0e-4


def _log_frac(x: float, lo: float, hi: float) -> float:
    """Clamped log-linear position of ``x`` within ``[lo, hi]`` — 0 at/below
    ``lo``, 1 at/above ``hi``. ``max(1, x)`` guards the log domain against a
    zero/negative ``x`` (which clamps to 0, i.e. the ``lo`` end)."""
    t = (math.log(max(1, x)) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return min(1.0, max(0.0, t))


def finetune_base_lr(
    num_images: int,
    hidden_dim: int,
    *,
    eff_batch: int = REFERENCE_BATCH,
    adamw: bool = True,
) -> float:
    """Bounded, fail-safe fine-tune base LR (the head/neck LR).

    A flat plateau-centre constant (``FINETUNE_BASE_LR``), batch-scaled (sqrt for
    AdamW, linear for SGD) off the batch-16 anchor and HARD-CLAMPED to
    ``[FINETUNE_LR_MIN, FINETUNE_LR_MAX]``. The value is deliberately independent
    of ``num_images`` and ``hidden_dim`` — neither predicts where in the flat
    plateau a dataset's optimum sits (see the law header). The args are kept
    because callers pass what they know and the value may key on them again if a
    real predictor turns up. ALWAYS returns a finite value in the band and never
    raises; any degenerate input yields ``FINETUNE_LR_SAFE_DEFAULT``. base_lr is
    the head LR; the backbone LR is the LLRD ramp's job.
    """
    try:
        ratio = max(1, int(eff_batch)) / REFERENCE_BATCH
        lr = FINETUNE_BASE_LR * (math.sqrt(ratio) if adamw else ratio)
    except Exception:
        # Any bad-type/degenerate arg -> the safe band centre. isfinite below
        # also catches a nan/inf that slipped through; the clamp does the rest.
        lr = FINETUNE_LR_SAFE_DEFAULT
    if not math.isfinite(lr):
        lr = FINETUNE_LR_SAFE_DEFAULT
    return float(min(FINETUNE_LR_MAX, max(FINETUNE_LR_MIN, lr)))


def finetune_num_epochs(num_images: int) -> int:
    """Fine-tune epoch budget: a smooth log taper from MAX_FINETUNE_EPOCHS on
    small datasets to MIN_FINETUNE_EPOCHS on large ones, log-linear in image
    count from 1,000 up to 20,000 images and clamped to that band. Small sets
    get more passes (cheap in steps, and they need them); large sets fewer
    (expensive, and they saturate). Replaces the old fixed-step-budget clamp,
    whose hard MIN floor undertrained mid-size sets at a sharp cliff.
    """
    t = _log_frac(num_images, _EPOCH_TAPER_IMG_LO, _EPOCH_TAPER_IMG_HI)
    epochs = MAX_FINETUNE_EPOCHS - t * (MAX_FINETUNE_EPOCHS - MIN_FINETUNE_EPOCHS)
    return round(epochs)


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


def finetune_llrd_decay(num_layers: int, ratio: float = FINETUNE_BACKBONE_LR_RATIO) -> float:
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
    #   * num_epochs — a smooth log taper on dataset size (finetune_num_epochs);
    #     the engine resolves epochs back to iterations at train time.
    eff_batch = cfg.solver.effective_batch()
    num_epochs = finetune_num_epochs(stats.num_images)
    # base_lr is a flat plateau-centre default, batch-scaled and hard-clamped to
    # the safe band (no size/width lever — neither predicts the plateau optimum).
    # The backbone LR is the LLRD decay's job (below).
    solver_overrides: dict[str, Any] = {
        "base_lr": finetune_base_lr(
            stats.num_images,
            cfg.model.fpn.out_channels,
            eff_batch=eff_batch,
            adamw=cfg.solver.optimizer_name != "SGD",
        ),
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
