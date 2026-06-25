"""Tests for aspect-aware input sizing (:mod:`mayaku.tuning.sizing`)."""

from __future__ import annotations

import pytest

from mayaku.tuning.sizing import resolve_canvas, snap_max_content

BUDGET = 640 * 640  # 409,600 — the size_budget=640 square-equivalent budget


@pytest.mark.parametrize(
    ("aspect", "expected_hw"),
    [
        (1.0, (640, 640)),  # square → the largest aligned square under budget
        (4 / 3, (512, 768)),  # 4:3 landscape
        (16 / 9, (512, 768)),  # 16:9 landscape
        (3.0, (384, 1024)),  # 3:1 ultrawide
        (3.75, (384, 1024)),  # 3.75:1 — your 1500×400 case
        (9 / 16, (768, 512)),  # portrait is the transpose of 16:9
        (1 / 3.0, (1024, 384)),  # 1:3 tall — transpose of 3:1
    ],
)
def test_snap_max_content_oracle(aspect: float, expected_hw: tuple[int, int]) -> None:
    assert snap_max_content(BUDGET, aspect) == expected_hw


def test_never_exceeds_budget() -> None:
    for aspect in (1.0, 1.33, 1.78, 2.0, 3.0, 3.75, 5.0, 0.3):
        h, w = snap_max_content(BUDGET, aspect)
        assert h * w <= BUDGET, (aspect, h, w, h * w)
        assert h % 128 == 0 and w % 128 == 0


def test_maximizes_content_vs_square() -> None:
    # For any non-square aspect the chosen canvas yields >= the real content a
    # square canvas would (square reclaims nothing from padding).
    def content(h: int, w: int, a: float) -> float:
        return min(w * w / a, a * h * h)

    for aspect in (4 / 3, 16 / 9, 3.0, 3.75):
        h, w = snap_max_content(BUDGET, aspect)
        square = 640
        assert content(h, w, aspect) >= content(square, square, aspect)


def test_portrait_is_transpose_of_landscape() -> None:
    # snap(a) and snap(1/a) should be transposes (H<->W).
    h_l, w_l = snap_max_content(BUDGET, 16 / 9)
    h_p, w_p = snap_max_content(BUDGET, 9 / 16)
    assert (h_p, w_p) == (w_l, h_l)


def test_budget_scales_the_canvas() -> None:
    # A smaller budget (lower size_budget) → a smaller canvas, same aspect family.
    big = snap_max_content(640 * 640, 16 / 9)
    small = snap_max_content(512 * 512, 16 / 9)
    assert small[0] * small[1] < big[0] * big[1]


def test_resolve_canvas_diverse_is_square() -> None:
    # Diverse data ignores the aspect → largest aligned square under budget.
    (h, w), use = resolve_canvas(640, aspect=3.0, uniform=False)
    assert (h, w) == (640, 640)
    assert use == 1.0


def test_resolve_canvas_uniform_fits_aspect() -> None:
    (h, w), use = resolve_canvas(640, aspect=16 / 9, uniform=True)
    assert (h, w) == (512, 768)  # matches the oracle
    assert 0.9 < use <= 1.0  # uses ~96% of the budget


def test_resolve_canvas_reports_grid_headroom() -> None:
    # An awkward aspect that the 128 grid can't fill → budget_use well under 1
    # (this is what triggers the "raise size_budget" info line).
    (h, w), use = resolve_canvas(640, aspect=5.0, uniform=True)
    assert h * w <= 640 * 640
    assert use < 1.0


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="budget"):
        snap_max_content(0, 1.0)
    with pytest.raises(ValueError, match="aspect"):
        snap_max_content(BUDGET, 0.0)
