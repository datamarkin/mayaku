"""Aspect-aware canvas sizing (:mod:`mayaku.data.canvas`) on the 32 grid."""

from __future__ import annotations

import pytest

from mayaku.data.canvas import (
    canvas_for_data,
    data_aspect,
    multi_scale_canvases,
    snap_max_content,
)

BUDGET = 800 * 800


@pytest.mark.parametrize(
    ("aspect", "expected_hw"),
    [
        (1.0, (800, 800)),       # square: the budget dial is the side
        (4 / 3, (672, 896)),
        (16 / 9, (576, 1024)),
        (9 / 16, (1024, 576)),   # portrait is the transpose of 16:9
    ],
)
def test_snap_max_content_oracle(aspect: float, expected_hw: tuple[int, int]) -> None:
    assert snap_max_content(BUDGET, aspect) == expected_hw


def test_never_exceeds_budget_and_stays_aligned() -> None:
    for aspect in (1.0, 1.33, 1.78, 2.0, 3.0, 3.75, 5.0, 0.3):
        h, w = snap_max_content(BUDGET, aspect)
        assert h * w <= BUDGET
        assert h % 32 == 0 and w % 32 == 0


def test_maximizes_content_vs_square() -> None:
    def content(h: int, w: int, a: float) -> float:
        return min(w * w / a, a * h * h)

    for aspect in (1.33, 1.78, 3.0, 0.5):
        h, w = snap_max_content(BUDGET, aspect)
        assert content(h, w, aspect) >= content(800, 800, aspect)


def test_data_aspect_and_canvas_for_data() -> None:
    video = [(1080, 1920)] * 20
    assert data_aspect(video) == (pytest.approx(16 / 9), True)
    assert canvas_for_data(video, 800) == (576, 1024)
    mixed = [(480, 640), (1080, 1920), (800, 600)] * 10
    assert data_aspect(mixed)[1] is False
    assert canvas_for_data(mixed, 800) == (800, 800)
    # a few outliers do not flip a uniform set
    assert data_aspect([(1080, 1920)] * 95 + [(1000, 1000)] * 5)[1] is True
    assert data_aspect([]) == (1.0, False)


def test_square_ladder_steps_the_side_by_32() -> None:
    assert multi_scale_canvases((800, 800), 512) == [(s, s) for s in range(512, 801, 32)]
    assert multi_scale_canvases((800, 800), 500) == [(s, s) for s in range(480, 801, 32)]
    assert multi_scale_canvases((640, 640), 640) == [(640, 640)]


def test_rectangular_ladder_keeps_the_aspect_and_ends_on_the_canvas() -> None:
    deploy = (576, 1024)
    ladder = multi_scale_canvases(deploy, 512)
    assert ladder[-1] == deploy and ladder[0] == (288, 512)
    areas = [h * w for h, w in ladder]
    assert areas == sorted(areas) and len(set(ladder)) == len(ladder)
    assert all(h % 32 == 0 and w % 32 == 0 and h * w <= 576 * 1024 for h, w in ladder)
    assert all(abs((w / h) - 16 / 9) < 0.25 for h, w in ladder)


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="budget"):
        snap_max_content(0, 1.0)
    with pytest.raises(ValueError, match="aspect"):
        snap_max_content(BUDGET, 0.0)
