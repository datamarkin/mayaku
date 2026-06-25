"""Tests for the letterbox primitive — the fixed-size inference geometry.

Centred aspect-preserving resize to a ``size × size`` square, with an exact
uniform-scale inverse for mapping predictions back to the original image.
"""

from __future__ import annotations

import numpy as np
import pytest

from mayaku.data.transforms import LetterboxResize, LetterboxTransform, letterbox


def _img(h: int, w: int, c: int = 3) -> np.ndarray:
    return (np.random.default_rng(0).random((h, w, c)) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Geometry: shape, scale, centred padding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("h", "w"),
    [(600, 600), (300, 600), (600, 300), (480, 640), (123, 457)],
)
def test_output_is_square_size(h: int, w: int) -> None:
    out, t = letterbox(_img(h, w), size=640)
    assert out.shape == (640, 640, 3)
    # Long side maps to exactly `size`; short side fits inside.
    assert max(t.new_h, t.new_w) == 640
    assert t.new_h <= 640 and t.new_w <= 640


def test_scale_and_padding_centred() -> None:
    t = LetterboxTransform(300, 600, 640)  # wide → vertical bars
    assert t.scale == pytest.approx(640 / 600)
    assert t.new_w == 640
    assert t.new_h == round(300 * (640 / 600))  # 320
    # Equal top/bottom bars (centred), and no horizontal pad.
    assert t.pad_left == 0
    assert t.pad_top == (640 - t.new_h) // 2  # 160
    pad_bottom = 640 - t.new_h - t.pad_top
    assert abs(t.pad_top - pad_bottom) <= 1  # symmetric to within a pixel


def test_square_image_is_pure_resize_no_pad() -> None:
    t = LetterboxTransform(320, 320, 640)
    assert t.pad_top == 0 and t.pad_left == 0
    assert t.new_h == 640 and t.new_w == 640
    assert t.scale == pytest.approx(2.0)


def test_pad_value_fills_the_bars() -> None:
    out, t = letterbox(_img(300, 600), size=640, pad_value=114)
    # Top and bottom bands are the pad value; the content band is not all-pad.
    assert np.all(out[: t.pad_top] == 114)
    assert np.all(out[t.pad_top + t.new_h :] == 114)
    content = out[t.pad_top : t.pad_top + t.new_h]
    assert not np.all(content == 114)


# ---------------------------------------------------------------------------
# Exact inverse (the whole point: predictions map back cleanly)
# ---------------------------------------------------------------------------


def test_coords_roundtrip_is_identity() -> None:
    t = LetterboxTransform(457, 123, 640)
    coords = np.array([[0.0, 0.0], [60.0, 200.0], [122.0, 456.0]], dtype=np.float32)
    back = t.inverse_coords(t.apply_coords(coords))
    np.testing.assert_allclose(back, coords, atol=1e-3)


def test_box_roundtrip_is_identity() -> None:
    t = LetterboxTransform(300, 600, 640)
    boxes = np.array([[10.0, 20.0, 500.0, 250.0], [0.0, 0.0, 600.0, 300.0]], dtype=np.float32)
    back = t.inverse_box(t.apply_box(boxes))
    np.testing.assert_allclose(back, boxes, atol=1e-3)


def test_inverse_box_empty() -> None:
    t = LetterboxTransform(300, 600, 640)
    assert t.inverse_box(np.zeros((0, 4), dtype=np.float32)).shape == (0, 4)


def test_apply_coords_maps_far_corner_to_content_edge() -> None:
    # The image's bottom-right corner (600, 300) lands at the right canvas edge
    # (x=640) and the bottom of the content band (y = 320 + pad_top 160 = 480).
    t = LetterboxTransform(300, 600, 640)
    corner = t.apply_coords(np.array([[600.0, 300.0]], dtype=np.float32))[0]
    np.testing.assert_allclose(corner, [640.0, 480.0], atol=1e-2)


# ---------------------------------------------------------------------------
# Segmentation: nearest-neighbour + background pad
# ---------------------------------------------------------------------------


def test_segmentation_nearest_and_zero_pad() -> None:
    mask = np.ones((300, 600), dtype=np.uint8)  # all foreground label 1
    t = LetterboxTransform(300, 600, 640)
    out = t.apply_segmentation(mask)
    assert out.shape == (640, 640)
    assert set(np.unique(out)).issubset({0, 1})  # nearest → no interpolated labels
    assert np.all(out[: t.pad_top] == 0)  # bars are background, not pad_value


# ---------------------------------------------------------------------------
# LetterboxResize augmentation
# ---------------------------------------------------------------------------


def test_resize_fixed_size() -> None:
    aug = LetterboxResize(640)
    t = aug.get_transform(_img(300, 600))
    assert isinstance(t, LetterboxTransform)
    assert t.out_h == t.out_w == 640  # scalar size → square canvas


def test_resize_multiscale_choice_stays_in_set() -> None:
    sizes = (640, 672, 704, 736, 768, 800)
    aug = LetterboxResize(sizes, rng=np.random.default_rng(0))
    drawn = {aug.get_transform(_img(400, 500)).out_h for _ in range(40)}
    assert drawn.issubset(set(sizes))
    assert len(drawn) > 1  # actually varies


def test_resize_range_style() -> None:
    aug = LetterboxResize((640, 800), sample_style="range", rng=np.random.default_rng(0))
    for _ in range(20):
        assert 640 <= aug.get_transform(_img(400, 500)).out_h <= 800


def test_letterbox_rectangle_uniform_scale_and_inverse() -> None:
    # Portrait image (h=640, w=400) into a portrait (H=640, W=384) canvas.
    img = _img(640, 400)
    t = LetterboxTransform(640, 400, (640, 384))
    assert (t.out_h, t.out_w) == (640, 384)
    # Fit-inside uniform scale: min(640/640, 384/400) = 0.96 (width binds).
    assert t.scale == pytest.approx(0.96)
    assert t.new_w == 384 and t.new_h == round(640 * 0.96)  # 614, with vertical pad
    assert t.apply_image(img).shape[:2] == (640, 384)
    # The inverse is exact — a single uniform scale, same as the square case.
    boxes = np.array([[10.0, 20.0, 300.0, 500.0]], dtype=np.float32)
    np.testing.assert_allclose(t.inverse_box(t.apply_box(boxes)), boxes, atol=1e-2)


def test_letterbox_rectangle_reduces_to_square() -> None:
    # A scalar size and the equal-sided pair must produce identical geometry.
    sq = LetterboxTransform(300, 600, 640)
    rect = LetterboxTransform(300, 600, (640, 640))
    assert (sq.scale, sq.new_h, sq.new_w) == (rect.scale, rect.new_h, rect.new_w)


def test_resize_validates_range_arity() -> None:
    with pytest.raises(ValueError, match="range"):
        LetterboxResize((640, 700, 800), sample_style="range")
