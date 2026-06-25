"""Aspect-aware input sizing under a compute budget.

The keystone of native-size training/inference. ``size_budget`` is the *budget
dial* — the square-equivalent side, so the compute budget is ``size_budget**2``
pixels. :func:`snap_max_content` resolves the actual ``(H, W)`` canvas that
maximizes real letterbox content for the data's native aspect while staying
*under* that budget, on a stride-aligned grid.

Why max-content-under-budget (not closest-aspect, not long-edge): it's a strict
Pareto win over square letterbox — equal-or-more real resolution at equal-or-less
compute on every aspect — and the never-exceed ceiling gives a hard compute /
memory bound (no OOM past the square baseline). Default ``align=128`` is
torch.compile-safe (and a superset of the FPN stride-32 minimum), so one resolved
size is valid on every backend.
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
    budget: int,
    aspect: float,
    *,
    scale_min: float = 0.5,
    steps: int = 4,
    align: int = 128,
) -> list[tuple[int, int]]:
    """Multi-scale letterbox canvases for training, from a budget + aspect.

    The top (full ``budget``) is the deploy canvas — train geometry == deploy;
    smaller ones (down to ``scale_min`` of the budget) are scale-down
    augmentation. Each maximizes content under its own budget. Returns a
    de-duplicated list ascending by area.
    """
    if not 0.0 < scale_min <= 1.0:
        raise ValueError(f"scale_min must be in (0, 1]; got {scale_min}")
    fracs = (
        [scale_min + (1.0 - scale_min) * i / (steps - 1) for i in range(steps)]
        if steps > 1
        else [1.0]
    )
    out: list[tuple[int, int]] = []
    for f in fracs:
        canvas = snap_max_content(max(align * align, int(budget * f)), aspect, align=align)
        if canvas not in out:
            out.append(canvas)
    return out


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
