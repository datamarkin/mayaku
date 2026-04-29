"""Tests for :mod:`mayaku.utils.image`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mayaku.utils.image import bgr_to_rgb, read_image


def test_read_image_returns_uint8_rgb(tmp_path: Path) -> None:
    rgb = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [128, 128, 128]]],
        dtype=np.uint8,
    )
    path = tmp_path / "tiny.png"
    Image.fromarray(rgb).save(path)
    out = read_image(path)
    assert out.shape == (2, 2, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, rgb)


def test_read_image_converts_grayscale_to_rgb(tmp_path: Path) -> None:
    gray = np.full((4, 4), 200, dtype=np.uint8)
    path = tmp_path / "gray.png"
    Image.fromarray(gray, mode="L").save(path)
    out = read_image(path)
    assert out.shape == (4, 4, 3)
    # All channels equal because the source was grayscale.
    assert np.all(out[..., 0] == out[..., 1])
    assert np.all(out[..., 0] == out[..., 2])


def test_read_image_converts_rgba_to_rgb(tmp_path: Path) -> None:
    rgba = np.zeros((3, 3, 4), dtype=np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = 255
    path = tmp_path / "rgba.png"
    Image.fromarray(rgba, mode="RGBA").save(path)
    out = read_image(path)
    assert out.shape == (3, 3, 3)


def test_bgr_to_rgb_swaps_channels() -> None:
    bgr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    rgb = bgr_to_rgb(bgr)
    assert rgb.shape == bgr.shape
    np.testing.assert_array_equal(rgb[0, 0], [3, 2, 1])
    np.testing.assert_array_equal(rgb[0, 1], [6, 5, 4])
    # Output is contiguous for downstream torch.from_numpy zero-copy.
    assert rgb.flags["C_CONTIGUOUS"]


def test_bgr_to_rgb_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        bgr_to_rgb(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        bgr_to_rgb(np.zeros((4, 4, 4), dtype=np.uint8))
