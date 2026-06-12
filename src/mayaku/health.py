"""Dataset health check — one scan per split, returned as a plain dict.

``health_check`` reuses the same primitives the trainer's auto-config
runs on (:func:`mayaku.data.resolve_dataset` +
:func:`mayaku.tuning.analyze_dataset`), so the numbers a user inspects
here are exactly the ones that drive the recipe. Output is JSON-friendly
(dicts / lists / scalars) so callers can print it, log it, or feed it
straight into auto-config.

    >>> import mayaku
    >>> report = mayaku.health_check("/data/myproject")   # doctest: +SKIP
    >>> report["train"]["object_size"]                    # doctest: +SKIP
    {'small': 0.62, 'medium': 0.31, 'large': 0.07}

Object sizes are bucketed in *resized* space (after the canonical
short-edge resize), using the COCO small/medium/large thresholds — that
is the scale the model actually sees, which is what resolution decisions
should key off.

``warnings`` lists factual data-hygiene problems only (degenerate boxes,
unlabelled images). Interpretation that needs a tuning threshold
(imbalance, small-object-heavy, rare classes) is deliberately left to the
auto-config recipe, where those numbers are set with justification rather
than guessed here.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from mayaku.data import build_coco_metadata, load_coco_json, resolve_dataset
from mayaku.data.catalog import Metadata
from mayaku.tuning.dataset_stats import DatasetStats, analyze_dataset

__all__ = ["health_check"]

# COCO area thresholds, expressed as sqrt(area): small < 32px, large >= 96px.
_SMALL_EDGE = 32.0
_LARGE_EDGE = 96.0


def health_check(data: str | Path) -> dict[str, Any]:
    """Scan a dataset and return per-split health statistics.

    ``data`` is a dataset directory or a ``.yaml`` descriptor, resolved
    by :func:`mayaku.data.resolve_dataset`. The result maps each present
    split (``train`` and, when defined, ``val`` / ``test``) to a dict of
    counts, distributions, and factual ``warnings``.
    """
    return {
        split: _summarize_split(split, source.images, source.annotations)
        for split, source in resolve_dataset(data).items()
    }


def _summarize_split(split: str, images: Path, annotations: Path) -> dict[str, Any]:
    metadata = build_coco_metadata(name=f"health_{split}", json_path=annotations)
    dataset_dicts = load_coco_json(
        annotations, images, metadata, keep_segmentation=False, keep_keypoints=False
    )
    stats = analyze_dataset(dataset_dicts, num_classes=len(metadata.thing_classes))
    return {
        "images": stats.num_images,
        "boxes": stats.num_boxes,
        "classes": stats.num_classes,
        "boxes_per_image": round(stats.num_boxes / stats.num_images, 2)
        if stats.num_images
        else 0.0,
        "object_size": _size_fractions(stats.sqrt_areas),
        "aspect_ratio": _percentiles(stats.aspect_ratios),
        "class_imbalance": round(stats.class_imbalance, 1),
        "class_counts": _named_counts(stats, metadata),
        "warnings": _warnings(stats),
    }


def _size_fractions(sqrt_areas: tuple[float, ...]) -> dict[str, float]:
    """Fraction of boxes in each COCO size bucket (resized space)."""
    if not sqrt_areas:
        return {"small": 0.0, "medium": 0.0, "large": 0.0}
    total = len(sqrt_areas)
    small = sum(1 for a in sqrt_areas if a < _SMALL_EDGE)
    large = sum(1 for a in sqrt_areas if a >= _LARGE_EDGE)
    return {
        "small": round(small / total, 2),
        "medium": round((total - small - large) / total, 2),
        "large": round(large / total, 2),
    }


def _percentiles(values: tuple[float, ...]) -> dict[str, float] | None:
    """p10 / median / p90 of ``values``; ``None`` when there's too little."""
    if len(values) < 2:
        return None
    deciles = statistics.quantiles(values, n=10)  # 9 cut points: [0]=p10 .. [8]=p90
    return {
        "p10": round(deciles[0], 2),
        "median": round(statistics.median(values), 2),
        "p90": round(deciles[8], 2),
    }


def _named_counts(stats: DatasetStats, metadata: Metadata) -> dict[str, int]:
    """Image-frequency per class, keyed by class name.

    ``load_coco_json`` already remaps annotation category ids to the
    contiguous range, so ``class_counts`` keys index ``thing_classes``
    directly.
    """
    names = metadata.thing_classes
    return {names[cid]: count for cid, count in stats.class_counts.items()}


def _warnings(stats: DatasetStats) -> list[str]:
    """Factual data-hygiene flags only — no tuning thresholds."""
    warnings: list[str] = []
    if stats.num_degenerate_boxes:
        warnings.append(f"{stats.num_degenerate_boxes} degenerate (zero/negative-area) boxes")
    if stats.num_images_without_annotations:
        warnings.append(f"{stats.num_images_without_annotations} images with no annotations")
    return warnings
