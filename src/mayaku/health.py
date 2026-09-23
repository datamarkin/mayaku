"""Dataset health check: one scan of one split, returned as a plain dict.

It reads the split the way training does (`mayaku.data.coco.load_coco`) and
reports the statistics auto-config works from (`mayaku.tuning.analyze_dataset`).
Box sizes are measured on the canvas the data would train at under the default
size budget (`mayaku.tuning.canvas_for_data`), bucketed with the COCO
small / medium / large thresholds -- the scale the model sees. Output is
JSON-friendly.

    >>> import mayaku
    >>> report = mayaku.health_check(                      # doctest: +SKIP
    ...     "train/_annotations.coco.json", "train/"
    ... )
    >>> report["object_size"]                             # doctest: +SKIP
    {'small': 0.62, 'medium': 0.31, 'large': 0.07}

``warnings`` lists factual data-hygiene problems only (degenerate boxes,
unlabelled images); anything that needs a tuning threshold is auto-config's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mayaku.config import MayakuConfig
from mayaku.data.canvas import canvas_for_data
from mayaku.data.coco import load_coco
from mayaku.tuning.dataset_stats import DatasetStats, analyze_dataset

__all__ = ["health_check"]

# COCO area thresholds, expressed as sqrt(area): small < 32px, large >= 96px.
_SMALL_EDGE = 32.0
_LARGE_EDGE = 96.0


def health_check(annotations: str | Path, images: str | Path,
                 cfg: MayakuConfig | None = None) -> dict[str, Any]:
    """Scan one dataset split and return its health statistics: counts, the
    canvas it would train at under `cfg` (its pinned ``input.canvas_hw``, else
    the one auto-config would pick), distributions, and factual ``warnings``."""
    cfg = cfg or MayakuConfig()
    coco = load_coco(str(images), str(annotations))
    canvas = cfg.input.canvas_hw or canvas_for_data(coco.shapes, cfg.input.size_budget)
    stats = analyze_dataset(coco, canvas)
    return {
        "images": stats.num_images,
        "boxes": stats.num_boxes,
        "classes": stats.num_classes,
        "canvas": list(canvas),
        "boxes_per_image": round(stats.num_boxes / stats.num_images, 2) if stats.num_images else 0.0,
        "object_size": _size_fractions(stats.sqrt_areas),
        "aspect_ratio": _percentiles(stats.aspect_ratios),
        "class_imbalance": round(stats.class_imbalance, 1),
        "class_counts": {coco.class_names[c]: n for c, n in stats.class_counts.items()},
        "warnings": _warnings(stats),
    }


def _size_fractions(sqrt_areas: np.ndarray) -> dict[str, float]:
    """Fraction of boxes in each COCO size bucket."""
    if not len(sqrt_areas):
        return {"small": 0.0, "medium": 0.0, "large": 0.0}
    small = float(np.mean(sqrt_areas < _SMALL_EDGE))
    large = float(np.mean(sqrt_areas >= _LARGE_EDGE))
    return {"small": round(small, 2), "medium": round(1 - small - large, 2), "large": round(large, 2)}


def _percentiles(values: np.ndarray) -> dict[str, float] | None:
    """p10 / median / p90 of ``values``; ``None`` when there's too little."""
    if len(values) < 2:
        return None
    p10, p50, p90 = np.percentile(values, [10, 50, 90], method="weibull")
    return {"p10": round(float(p10), 2), "median": round(float(p50), 2), "p90": round(float(p90), 2)}


def _warnings(stats: DatasetStats) -> list[str]:
    """Factual data-hygiene flags only — no tuning thresholds."""
    warnings: list[str] = []
    if stats.num_degenerate_boxes:
        warnings.append(f"{stats.num_degenerate_boxes} degenerate boxes (a side of a pixel or less)")
    if stats.num_images_without_annotations:
        warnings.append(f"{stats.num_images_without_annotations} images with no annotations")
    return warnings
