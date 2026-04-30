"""Tests for :mod:`mayaku.data.transforms.photometric`."""

from __future__ import annotations

import numpy as np
import pytest

from mayaku.data.transforms.augmentation import _NoOpTransform
from mayaku.data.transforms.photometric import (
    BrightnessTransform,
    ContrastTransform,
    HueShiftTransform,
    RandomColorJitter,
    SaturationTransform,
    _hsv_to_rgb,
    _rgb_to_hsv,
)


def _sample_image() -> np.ndarray:
    """A small RGB image with a couple of saturated colours and a grey patch."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    # Stamp known patches so colour-space math is checkable.
    img[0:8, 0:8] = [255, 0, 0]  # pure red
    img[0:8, 8:16] = [0, 255, 0]  # pure green
    img[0:8, 16:24] = [128, 128, 128]  # mid-grey
    return img


# ---------------------------------------------------------------------------
# Identity invariants
# ---------------------------------------------------------------------------


def test_brightness_identity_at_factor_1() -> None:
    img = _sample_image()
    out = BrightnessTransform(1.0).apply_image(img)
    np.testing.assert_array_equal(img, out)


def test_contrast_identity_at_factor_1() -> None:
    img = _sample_image()
    out = ContrastTransform(1.0).apply_image(img)
    # Float-roundtrip can drift by 1 LSB on individual pixels.
    np.testing.assert_array_equal(img, out)


def test_saturation_identity_at_factor_1() -> None:
    img = _sample_image()
    out = SaturationTransform(1.0).apply_image(img)
    np.testing.assert_array_equal(img, out)


def test_hue_identity_at_zero_shift() -> None:
    img = _sample_image()
    out = HueShiftTransform(0.0).apply_image(img)
    np.testing.assert_array_equal(img, out)


# ---------------------------------------------------------------------------
# Behavioural checks
# ---------------------------------------------------------------------------


def test_brightness_2x_doubles_dim_pixels_clips_bright() -> None:
    img = np.array([[[10, 20, 30], [200, 200, 200]]], dtype=np.uint8)
    out = BrightnessTransform(2.0).apply_image(img)
    assert out[0, 0].tolist() == [20, 40, 60]
    # 200 * 2 = 400 → clipped to 255
    assert out[0, 1].tolist() == [255, 255, 255]


def test_brightness_zero_factor_blacks_image() -> None:
    img = _sample_image()
    out = BrightnessTransform(0.0).apply_image(img)
    assert out.sum() == 0


def test_contrast_zero_factor_collapses_to_per_channel_mean() -> None:
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    expected = np.round(img.reshape(-1, 3).mean(axis=0)).astype(np.uint8)
    out = ContrastTransform(0.0).apply_image(img)
    # Every pixel of every channel should equal the per-channel mean
    # (allowing 1-LSB rounding).
    np.testing.assert_allclose(out, np.broadcast_to(expected, img.shape), atol=1)


def test_saturation_zero_factor_produces_greyscale() -> None:
    img = _sample_image()
    out = SaturationTransform(0.0).apply_image(img)
    # All three channels equal at every pixel ⇔ greyscale.
    np.testing.assert_allclose(out[..., 0], out[..., 1], atol=1)
    np.testing.assert_allclose(out[..., 1], out[..., 2], atol=1)


def test_dtype_preserved() -> None:
    img = _sample_image()
    for op in (
        BrightnessTransform(1.2),
        ContrastTransform(0.8),
        SaturationTransform(1.3),
        HueShiftTransform(0.01),
    ):
        out = op.apply_image(img)
        assert out.dtype == img.dtype
        assert out.shape == img.shape


def test_apply_coords_is_identity_for_all_photometric() -> None:
    coords = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    for op in (
        BrightnessTransform(1.5),
        ContrastTransform(0.5),
        SaturationTransform(0.0),
        HueShiftTransform(0.1),
    ):
        np.testing.assert_array_equal(op.apply_coords(coords), coords)


def test_apply_box_unchanged_under_photometric() -> None:
    """Photometric ops must not move bboxes — derived from apply_coords."""
    boxes = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    for op in (
        BrightnessTransform(2.0),
        ContrastTransform(0.5),
        SaturationTransform(0.0),
        HueShiftTransform(0.1),
    ):
        np.testing.assert_array_equal(op.apply_box(boxes), boxes)


# ---------------------------------------------------------------------------
# RGB ↔ HSV round-trip
# ---------------------------------------------------------------------------


def test_rgb_hsv_roundtrip() -> None:
    rng = np.random.default_rng(2)
    img = rng.random((64, 64, 3), dtype=np.float32)
    h, s, v = _rgb_to_hsv(img[..., 0], img[..., 1], img[..., 2])
    r, g, b = _hsv_to_rgb(h, s, v)
    out = np.stack([r, g, b], axis=-1)
    # Float HSV math has small rounding error; bound it.
    np.testing.assert_allclose(out, img, atol=1e-4)


# ---------------------------------------------------------------------------
# RandomColorJitter
# ---------------------------------------------------------------------------


def test_color_jitter_with_zero_probs_is_noop() -> None:
    rng = np.random.default_rng(0)
    aug = RandomColorJitter(prob=0.0, rng=rng)
    img = _sample_image()
    transform = aug.get_transform(img)
    # No-op transform doesn't change the image regardless of nesting.
    np.testing.assert_array_equal(transform.apply_image(img), img)


def test_color_jitter_disabling_components_via_zero_delta() -> None:
    """With all delta knobs at 0, no jitter ops fire even if prob=1."""
    rng = np.random.default_rng(0)
    aug = RandomColorJitter(
        brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, prob=1.0, rng=rng
    )
    img = _sample_image()
    out = aug.get_transform(img).apply_image(img)
    np.testing.assert_array_equal(out, img)


def test_color_jitter_returns_noop_when_nothing_fires() -> None:
    """If by chance no component fires, the returned transform is _NoOpTransform."""
    # prob=0 guarantees nothing fires.
    rng = np.random.default_rng(0)
    aug = RandomColorJitter(prob=0.0, rng=rng)
    transform = aug.get_transform(_sample_image())
    assert isinstance(transform, _NoOpTransform)


def test_color_jitter_actually_modifies_image_at_full_intensity() -> None:
    """At prob=1 + max-deltas, the output should differ from the input."""
    rng = np.random.default_rng(42)
    aug = RandomColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015, prob=1.0, rng=rng
    )
    img = _sample_image()
    out = aug.get_transform(img).apply_image(img)
    # The patch with mid-grey [128, 128, 128] is the most fragile because
    # saturation=0 maps it to itself; brightness/contrast still affect it.
    assert not np.array_equal(out, img)


def test_color_jitter_invalid_prob_raises() -> None:
    with pytest.raises(ValueError, match="prob"):
        RandomColorJitter(prob=1.5)
    with pytest.raises(ValueError, match="prob"):
        RandomColorJitter(prob=-0.1)


def test_color_jitter_negative_delta_raises() -> None:
    with pytest.raises(ValueError, match="brightness"):
        RandomColorJitter(brightness=-0.1)
    with pytest.raises(ValueError, match="contrast"):
        RandomColorJitter(contrast=-0.1)
