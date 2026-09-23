"""Aspect-aware input sizing under a compute budget.

The canvas is the one fixed (H, W) every image is letterboxed onto, for
training, evaluation, export and deployment alike. ``size_budget`` is the
budget dial -- the square-equivalent side, so the compute budget is
``size_budget ** 2`` pixels -- and :func:`snap_max_content` resolves the canvas
that holds the most real image content for the data's aspect while staying
under that budget.

Why max-content-under-budget: it is a strict Pareto win over a square
letterbox -- equal-or-more real resolution at equal-or-less compute on every
aspect -- and the never-exceed ceiling gives a hard compute / memory bound.

Every canvas is aligned to :data:`CANVAS_ALIGN` (32), the detector's coarsest
stride: both sides must divide by it for the stride-32 level, and for the
mask branch's exact 4x upsample from stride 32 onto stride 8. Nothing coarser
is needed, and a coarser grid wastes budget: at 800² a 128 grid leaves a
16:9 canvas holding 73% real content where the 32 grid holds 92%.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from mayaku.model.blocks import CANVAS_ALIGN

__all__ = [
    "ASPECT_UNIFORMITY_THRESHOLD",
    "CANVAS_ALIGN",
    "aspect_spread",
    "canvas_for_data",
    "data_aspect",
    "multi_scale_canvases",
    "snap_max_content",
]

#: Robust aspect spread (p90 / p10) at or below this: the dataset is "one
#: aspect", and a canvas at that aspect beats a square letterbox.
ASPECT_UNIFORMITY_THRESHOLD = 1.10


def aspect_spread(aspects: Sequence[float]) -> float:
    """Robust aspect spread ``p90 / p10`` (1.0 for fewer than 10 samples)."""
    n = len(aspects)
    if n < 10:
        return 1.0
    s = sorted(aspects)
    return s[(n * 9) // 10] / max(s[n // 10], 1e-9)


def data_aspect(shapes: Sequence[tuple[int, int]]) -> tuple[float, bool]:
    """Median image aspect ``W / H`` and whether the data is one aspect, from
    (height, width) image shapes. Uniform means the robust spread is within
    :data:`ASPECT_UNIFORMITY_THRESHOLD`, so a few outliers never flip it."""
    aspects = [w / h for h, w in shapes]
    if not aspects:
        return 1.0, False
    return statistics.median(aspects), aspect_spread(aspects) <= ASPECT_UNIFORMITY_THRESHOLD


def snap_max_content(budget: int, aspect: float) -> tuple[int, int]:
    """The aligned ``(H, W)`` canvas that maximises letterbox content under a
    budget.

    Args:
        budget: Max canvas area in pixels (``size_budget ** 2``). The result
            never exceeds it (``H * W <= budget``).
        aspect: Data aspect ``W / H`` (>1 landscape, <1 portrait, 1 square).

    Returns:
        ``(H, W)`` maximising real content ``min(W**2 / aspect, aspect * H**2)``
        -- the binding dimension's content after an aspect-preserving
        letterbox. For a square aspect this is the largest aligned square.
    """
    if budget <= 0:
        raise ValueError(f"budget must be > 0; got {budget}")
    if aspect <= 0:
        raise ValueError(f"aspect must be > 0; got {aspect}")
    # A side never needs to exceed the long edge of the most extreme fit,
    # sqrt(budget * max(a, 1/a)); round up to the grid for the search bound.
    reach = math.isqrt(int(budget * max(aspect, 1.0 / aspect)))
    sides = range(CANVAS_ALIGN, ((reach // CANVAS_ALIGN) + 1) * CANVAS_ALIGN + 1, CANVAS_ALIGN)

    best_content = -1.0
    best_hw = (CANVAS_ALIGN, CANVAS_ALIGN)
    for w in sides:
        for h in sides:
            if w * h > budget:
                continue
            content = min(w * w / aspect, aspect * h * h)
            if content > best_content:
                best_content = content
                best_hw = (h, w)
    return best_hw


def canvas_for_data(shapes: Sequence[tuple[int, int]], size_budget: int) -> tuple[int, int]:
    """The canvas for a dataset of (height, width) image shapes under
    ``size_budget ** 2`` pixels: a rectangle at the data's aspect when it has
    one (no padding waste), else a square, which is robust to any shape."""
    aspect, uniform = data_aspect(shapes)
    return snap_max_content(size_budget * size_budget, aspect if uniform else 1.0)


def multi_scale_canvases(deploy_canvas: tuple[int, int], min_long: int) -> list[tuple[int, int]]:
    """Training canvases for multi-scale, anchored on the deploy canvas.

    The long edge steps by ``CANVAS_ALIGN`` from ``min_long`` (floored to the grid)
    up to the deploy canvas's; each rung is the max-content canvas at the
    deploy aspect with the area that long edge implies, and the top rung is
    exactly ``deploy_canvas``, so train geometry equals deploy geometry at
    full scale. On a square canvas the rungs are the squares
    ``min_long, min_long + 32, ..., side``. Ascending, de-duplicated.
    """
    h, w = deploy_canvas
    long_edge = max(h, w)
    area, aspect = h * w, w / h
    lo = max(CANVAS_ALIGN, (min_long // CANVAS_ALIGN) * CANVAS_ALIGN)
    canvases = {(h, w)}
    for side in range(lo, long_edge, CANVAS_ALIGN):
        frac = side / long_edge
        canvases.add(snap_max_content(max(CANVAS_ALIGN ** 2, int(area * frac * frac)), aspect))
    return sorted(canvases, key=lambda c: c[0] * c[1])
