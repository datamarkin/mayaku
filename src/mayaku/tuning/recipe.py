"""Recipe lookup + override derivation for auto-config.

Maps :class:`DatasetStats` plus the user's base :class:`MayakuConfig`
into a nested override dict that
:func:`mayaku.config.merge_overrides` can apply. The recipe table is
anchored to the D2 scratch recipe (bs=16, lr=2e-2 on COCO) and applies
both linear batch-scaling and the standard 10× fine-tune drop.

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
    "MIN_BOXES_FOR_ANCHOR_TUNE",
    "MIN_IMAGES_FOR_AUTO_CONFIG",
    "SizeBucket",
    "collect_set_paths",
    "derive_overrides",
    "filter_unset",
    "size_bucket",
    "walk_leaves",
]

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


@dataclass(frozen=True)
class SizeBucket:
    """A row of the size-keyed recipe table.

    ``base_lr`` is the SGD learning rate at ``ims_per_batch=8`` after
    applying the 10× fine-tune drop on top of the D2 scratch anchor
    (bs=16, lr=2e-2 → bs=8 fine-tune, lr=1e-3). Callers re-apply
    batch-scaling for actual ``ims_per_batch`` values.
    """

    name: str
    base_lr: float
    epochs: int
    freeze_at: int
    mosaic_prob: float
    mixup_prob: float
    copy_paste_prob: float


_BUCKETS: Final[tuple[tuple[float, SizeBucket], ...]] = (
    # (upper_exclusive_image_count, bucket). Last entry uses sentinel
    # math.inf so any larger dataset still falls into a bucket.
    (
        500,
        SizeBucket(
            "xs",
            base_lr=5e-4,
            epochs=25,
            freeze_at=3,
            mosaic_prob=0.0,
            mixup_prob=0.0,
            copy_paste_prob=0.0,
        ),
    ),
    (
        2_000,
        SizeBucket(
            "s",
            base_lr=1e-3,
            epochs=15,
            freeze_at=2,
            mosaic_prob=0.0,
            mixup_prob=0.0,
            copy_paste_prob=0.0,
        ),
    ),
    (
        5_000,
        SizeBucket(
            "m",
            base_lr=1e-3,
            epochs=12,
            freeze_at=2,
            mosaic_prob=0.3,
            mixup_prob=0.0,
            copy_paste_prob=0.0,
        ),
    ),
    (
        50_000,
        SizeBucket(
            "l",
            base_lr=2e-3,
            epochs=10,
            freeze_at=2,
            mosaic_prob=0.5,
            mixup_prob=0.1,
            copy_paste_prob=0.0,
        ),
    ),
    (
        math.inf,
        SizeBucket(
            "xl",
            base_lr=2e-3,
            epochs=8,
            freeze_at=2,
            mosaic_prob=0.5,
            mixup_prob=0.1,
            copy_paste_prob=0.1,
        ),
    ),
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
# Schedule helpers
# ---------------------------------------------------------------------------


def _build_schedule(max_iter: int) -> dict[str, Any]:
    """Build ``max_iter`` / ``steps`` / ``warmup_iters`` honouring
    :meth:`SolverConfig._check_schedule`.

    Invariants enforced: ``warmup_iters < max_iter``, every step in
    ``(0, max_iter)``, and ``steps`` strictly ascending.
    """
    warmup_iters = min(500, max(0, max_iter // 10))
    if warmup_iters >= max_iter:
        warmup_iters = max(0, max_iter - 1)

    if max_iter >= 5:
        s1 = max(1, round(0.67 * max_iter))
        s2 = max(s1 + 1, round(0.89 * max_iter))
        # Squeeze both inside (0, max_iter) and keep strict ordering.
        s2 = min(s2, max_iter - 1)
        s1 = min(s1, s2 - 1)
        steps: tuple[int, ...] = (max(1, s1), s2)
    else:
        # Cosine scheduler ignores steps, but the schema validator
        # still demands a valid tuple. One mid-point step always passes.
        steps = (max(1, max_iter // 2),)

    return {"max_iter": max_iter, "steps": steps, "warmup_iters": warmup_iters}


def _epoch_schedule(
    num_images: int, ims_per_batch: int, grad_accum_steps: int, epochs: int
) -> dict[str, Any]:
    """Like :func:`_build_schedule` but derives ``max_iter`` from an
    epoch budget over the effective batch size."""
    effective_batch = max(1, ims_per_batch * grad_accum_steps)
    iters_per_epoch = max(1, math.ceil(num_images / effective_batch))
    return _build_schedule(max(2, epochs * iters_per_epoch))


# ---------------------------------------------------------------------------
# Override derivation
# ---------------------------------------------------------------------------


def derive_overrides(
    stats: DatasetStats,
    cfg: MayakuConfig,
    user_set_paths: Collection[str] = frozenset(),
) -> dict[str, Any]:
    """Produce a nested override dict from dataset stats + base config.

    The returned dict mirrors :class:`MayakuConfig`'s shape (top-level
    keys are ``model``, ``solver``, ``input``, ``dataloader``) and can
    be passed straight to :func:`mayaku.config.merge_overrides`.

    ``user_set_paths`` is used to keep the schedule self-consistent:
    when ``solver.max_iter`` is user-pinned, ``warmup_iters`` and
    ``steps`` are derived against the user's ``max_iter`` rather than
    the bucket-derived one so the Pydantic schedule validator (warmup <
    max_iter, every step in ``(0, max_iter)``) is satisfied after merge.
    The caller still passes the result through :func:`filter_unset` —
    this parameter only changes *values*, not which paths are emitted.

    Returns an empty dict when ``stats.num_images <
    MIN_IMAGES_FOR_AUTO_CONFIG`` — the dataset is too small to drive a
    sensible recipe.
    """
    if stats.num_images < MIN_IMAGES_FOR_AUTO_CONFIG:
        return {}

    bucket = size_bucket(stats.num_images)

    # ----- model -----
    model_overrides: dict[str, Any] = {
        "roi_heads": {"num_classes": stats.num_classes},
        "backbone": {"freeze_at": bucket.freeze_at},
    }
    if stats.num_boxes >= MIN_BOXES_FOR_ANCHOR_TUNE:
        sizes = cluster_sizes(stats.sqrt_areas, k=5)
        ars = cluster_aspect_ratios(stats.aspect_ratios, k=3)
        model_overrides["anchor_generator"] = {
            "sizes": tuple((s,) for s in sizes),
            "aspect_ratios": (ars,),
        }

    # ----- solver -----
    # When the user pinned max_iter (in YAML or via --max-iter), the
    # bucket-derived schedule must clamp around that value so the merge
    # doesn't violate the SolverConfig validators.
    if "solver.max_iter" in user_set_paths:
        schedule = _build_schedule(cfg.solver.max_iter)
    else:
        schedule = _epoch_schedule(
            stats.num_images,
            cfg.solver.ims_per_batch,
            cfg.solver.grad_accum_steps,
            bucket.epochs,
        )
    # Re-apply linear batch scaling so the bucket's bs=8-anchored LR
    # tracks the actual effective batch. The fine-tune drop is already
    # baked into bucket.base_lr.
    scaled_lr = bucket.base_lr * (cfg.solver.ims_per_batch / 8.0)
    solver_overrides: dict[str, Any] = {
        "base_lr": scaled_lr,
        "lr_scheduler_name": "WarmupCosineLR",
        "ema_enabled": True,
        "ema_decay": 0.9995,
        "ema_tau": 500.0,
        **schedule,
    }

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
        "model": model_overrides,
        "solver": solver_overrides,
        "input": input_overrides,
    }

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
