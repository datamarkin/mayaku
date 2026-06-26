"""Aspect-aware input sizing under a compute budget.

The keystone of native-size training/inference. ``size_budget`` is the *budget
dial* — the square-equivalent side, so the compute budget is ``size_budget**2``
pixels. :func:`snap_max_content` resolves the actual ``(H, W)`` canvas that
maximizes real letterbox content for the data's native aspect while staying
*under* that budget, on a stride-aligned grid.

Why max-content-under-budget (not closest-aspect, not long-edge): it's a strict
Pareto win over square letterbox — equal-or-more real resolution at equal-or-less
compute on every aspect — and the never-exceed ceiling gives a hard compute /
memory bound (no OOM past the square baseline).

Two alignment grids, on purpose:
    * **Deploy / eval / export** use ``align=128`` (the default): torch.compile-
      safe and ANE/TensorRT-friendly, so the single shipped size specialises
      best on every backend.
    * **Training** uses ``align=32`` (:func:`multi_scale_canvases`): the FPN
      stride floor and what the detector already pads to internally
      (``size_divisibility``). The finer grid gives a dense multi-scale ladder
      (e.g. 480/512/.../640 instead of the coarse 128-grid 384/512/640) — the
      coarse grid leaves ~2 AP on the table. The top rung is still pinned to the
      128-aligned deploy canvas, so train geometry == deploy at full scale.
"""

from __future__ import annotations

import math

__all__ = [
    "multi_scale_canvases",
    "resolve_canvas",
    "resolve_deploy_canvas",
    "snap_max_content",
]


def multi_scale_canvases(
    deploy_canvas: tuple[int, int],
    *,
    scale_min: float = 0.5,
    align: int = 32,
) -> list[tuple[int, int]]:
    """Multi-scale letterbox canvases for training, anchored on the deploy canvas.

    The top rung is exactly ``deploy_canvas`` (the 128-aligned deploy / export
    geometry), so train geometry == deploy at full scale. Smaller rungs step the
    long edge DOWN by ``align`` — 32, the FPN stride floor the detector already
    pads to — to roughly ``scale_min`` of the deploy *area*, each snapped to a
    max-content canvas at the deploy aspect. The fine 32 grid yields a dense
    ladder (e.g. 480/512/.../640) where the coarse 128 grid would give only
    384/512/640; that coarseness costs ~2 AP. Returns a de-duplicated list
    ascending by area; the deploy canvas is always the last entry.
    """
    if not 0.0 < scale_min <= 1.0:
        raise ValueError(f"scale_min must be in (0, 1]; got {scale_min}")
    h, w = deploy_canvas
    area = h * w
    aspect = w / h
    long_edge = max(h, w)
    # scale_min is an *area* fraction, so the long-edge floor is its sqrt;
    # round that floor UP to the grid so the smallest rung never dips below
    # scale_min of the budget.
    min_long = max(align, math.ceil(scale_min**0.5 * long_edge / align) * align)
    canvases: set[tuple[int, int]] = {deploy_canvas}
    for side in range(min_long, long_edge, align):
        frac = side / long_edge
        canvases.add(
            snap_max_content(max(align * align, int(area * frac * frac)), aspect, align=align)
        )
    return sorted(canvases, key=lambda hw: hw[0] * hw[1])


def resolve_canvas(
    size_budget: int,
    aspect: float,
    uniform: bool,
    *,
    align: int = 128,
) -> tuple[tuple[int, int], float]:
    """Resolve the train/deploy canvas from the budget dial + data aspect.

    ``size_budget`` is the budget dial (``budget = size_budget ** 2``). Uniform-aspect
    data fits a rectangle at ``aspect`` (no pad waste); diverse data falls back to
    a square (``aspect = 1.0``) — robust to any shape.

    Returns ``((H, W), budget_use)`` where ``budget_use = H * W / size_budget ** 2``
    in ``(0, 1]`` — a value well under 1 means the 128-grid left headroom and a
    larger ``size_budget`` would buy more resolution.
    """
    budget = size_budget * size_budget
    canvas = snap_max_content(budget, aspect if uniform else 1.0, align=align)
    return canvas, (canvas[0] * canvas[1]) / budget


def resolve_deploy_canvas(
    canvas_hw: tuple[int, int] | None,
    size_budget: int,
    *,
    align: int = 128,
) -> tuple[int, int]:
    """The fixed deploy/eval canvas: the pinned ``canvas_hw`` (resolved at train
    time or set manually), else the largest aligned square in the ``size_budget**2``
    budget. The single source for this fallback (resize builder + Predictor)."""
    return (
        canvas_hw
        if canvas_hw is not None
        else snap_max_content(size_budget * size_budget, 1.0, align=align)
    )


def snap_max_content(
    budget: int,
    aspect: float,
    *,
    align: int = 128,
    min_side: int = 128,
) -> tuple[int, int]:
    """Resolve the ``(H, W)`` canvas that maximizes letterbox content under budget.

    Args:
        budget: Max canvas area in pixels (``size_budget ** 2``). The result never
            exceeds it (``H * W <= budget``).
        aspect: Data aspect ``W / H`` (>1 landscape, <1 portrait, 1 square).
        align: Both sides are multiples of this. 128 is torch.compile-safe and
            also satisfies the FPN stride-32 minimum, so the size is valid on
            every backend.
        min_side: Smallest allowed side (multiple of ``align``).

    Returns:
        ``(H, W)``, both multiples of ``align``, with ``H * W <= budget``, chosen
        to maximize real content ``min(W**2 / aspect, aspect * H**2)`` — the
        binding-dimension content after an aspect-preserving letterbox. For a
        square aspect this is the largest aligned ``(s, s)`` under budget.
    """
    if budget <= 0:
        raise ValueError(f"budget must be > 0; got {budget}")
    if aspect <= 0:
        raise ValueError(f"aspect must be > 0; got {aspect}")
    # A side never needs to exceed the long edge of the most extreme fit,
    # sqrt(budget * max(a, 1/a)); round up to the grid for the search bound.
    reach = math.isqrt(int(budget * max(aspect, 1.0 / aspect)))
    max_side = ((reach // align) + 1) * align
    start = max(align, ((min_side + align - 1) // align) * align)
    sides = range(start, max_side + 1, align)

    best_content = -1.0
    best_hw = (start, start)
    for w in sides:
        for h in sides:
            if w * h > budget:
                continue
            # Real content after a uniform-scale letterbox of a `aspect`-shaped
            # image into (h, w): the binding dimension caps it.
            content = min(w * w / aspect, aspect * h * h)
            if content > best_content:
                best_content = content
                best_hw = (h, w)
    return best_hw
