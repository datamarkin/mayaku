"""Tests for :mod:`mayaku.data.transforms` (base + geometry + augmentation)."""

from __future__ import annotations

import numpy as np
import pytest

from mayaku.data.transforms import (
    AugInput,
    AugmentationList,
    HFlipTransform,
    RandomFlip,
    ResizeShortestEdge,
    ResizeTransform,
    Transform,
    TransformList,
)

# ---------------------------------------------------------------------------
# ResizeTransform
# ---------------------------------------------------------------------------


def test_resize_transform_image_and_coords_consistent() -> None:
    img = np.tile(np.linspace(0, 255, 10, dtype=np.uint8)[None, :, None], (8, 1, 3))
    t = ResizeTransform(h=8, w=10, new_h=16, new_w=20)
    out = t.apply_image(img)
    assert out.shape == (16, 20, 3)
    assert out.dtype == np.uint8

    # A (5, 4) coord doubles to (10, 8).
    coords = np.array([[5.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(t.apply_coords(coords), [[10.0, 8.0]])


def test_resize_transform_box_scales_corners() -> None:
    t = ResizeTransform(h=10, w=20, new_h=20, new_w=40)
    box = np.array([[1.0, 2.0, 5.0, 6.0]], dtype=np.float32)
    out = t.apply_box(box)
    np.testing.assert_allclose(out, [[2.0, 4.0, 10.0, 12.0]])


def test_resize_transform_segmentation_uses_nearest() -> None:
    # A 4x4 mask with two distinct labels — nearest interpolation must
    # not introduce intermediate values when upsampled.
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[2:, 2:] = 1
    t = ResizeTransform(h=4, w=4, new_h=8, new_w=8)
    out = t.apply_segmentation(mask)
    assert out.shape == (8, 8)
    assert set(np.unique(out)) <= {0, 1}


def test_resize_transform_rejects_size_mismatch() -> None:
    t = ResizeTransform(h=4, w=4, new_h=8, new_w=8)
    with pytest.raises(ValueError, match="configured for"):
        t.apply_image(np.zeros((5, 4, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# HFlipTransform
# ---------------------------------------------------------------------------


def test_hflip_image_and_coords() -> None:
    img = np.zeros((3, 4, 3), dtype=np.uint8)
    img[:, 0, 0] = 255
    flip = HFlipTransform(width=4)
    out = flip.apply_image(img)
    np.testing.assert_array_equal(out[:, -1, 0], [255, 255, 255])
    coords = np.array([[1.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(flip.apply_coords(coords), [[3.0, 2.0]])


def test_hflip_is_marked_horizontal() -> None:
    assert HFlipTransform(width=4).is_horizontal_flip is True
    assert ResizeTransform(h=1, w=1, new_h=1, new_w=1).is_horizontal_flip is False


# ---------------------------------------------------------------------------
# Transform.apply_polygons / apply_keypoints
# ---------------------------------------------------------------------------


def test_apply_polygons_through_resize() -> None:
    t = ResizeTransform(h=10, w=20, new_h=20, new_w=40)
    poly = np.array([0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 4.0], dtype=np.float32)
    out = t.apply_polygons([poly])
    np.testing.assert_allclose(out[0], [0, 0, 8, 0, 8, 8, 0, 8])


def test_apply_keypoints_preserves_visibility() -> None:
    t = ResizeTransform(h=10, w=10, new_h=20, new_w=20)
    kp = np.array([[[1.0, 2.0, 2.0], [3.0, 4.0, 0.0]]], dtype=np.float32)
    out = t.apply_keypoints(kp)
    np.testing.assert_allclose(out[0, 0], [2.0, 4.0, 2.0])
    np.testing.assert_allclose(out[0, 1], [6.0, 8.0, 0.0])


# ---------------------------------------------------------------------------
# TransformList — flip-pair swap policy
# ---------------------------------------------------------------------------


def test_transform_list_applies_flip_pair_swap_on_odd_flip_count() -> None:
    # 4-keypoint dataset; flip pair: indices 1<->2.
    flip_indices = [0, 2, 1, 3]
    flip = HFlipTransform(width=10)
    tl = TransformList([flip], flip_indices=flip_indices)
    kp = np.array(
        [
            [
                [1.0, 0.0, 2.0],  # index 0
                [2.0, 0.0, 2.0],  # index 1 (will move to slot 2)
                [3.0, 0.0, 2.0],  # index 2 (will move to slot 1)
                [4.0, 0.0, 2.0],  # index 3
            ]
        ],
        dtype=np.float32,
    )
    out = tl.apply_keypoints(kp)
    # X flipped (W=10), then index 1 and 2 swapped.
    np.testing.assert_allclose(out[0, 0, 0], 9.0)  # 10 - 1
    np.testing.assert_allclose(out[0, 1, 0], 7.0)  # what was at index 2: 10-3=7
    np.testing.assert_allclose(out[0, 2, 0], 8.0)  # what was at index 1: 10-2=8
    np.testing.assert_allclose(out[0, 3, 0], 6.0)


def test_transform_list_double_flip_is_identity_on_keypoint_indices() -> None:
    # Two HFlipTransforms cancel: x is back to original AND no pair swap.
    flip_indices = [0, 2, 1, 3]
    tl = TransformList([HFlipTransform(width=10), HFlipTransform(width=10)], flip_indices)
    kp = np.array([[[1.0, 0.0, 2.0], [2.0, 0.0, 2.0]]], dtype=np.float32)
    # Only the first two keypoint slots in this fixture — give it a
    # consistent K=4 to exercise the flip_indices path.
    kp = np.array(
        [
            [
                [1.0, 0.0, 2.0],
                [2.0, 0.0, 2.0],
                [3.0, 0.0, 2.0],
                [4.0, 0.0, 2.0],
            ]
        ],
        dtype=np.float32,
    )
    out = tl.apply_keypoints(kp)
    np.testing.assert_allclose(out, kp)


def test_transform_list_demands_flip_indices_when_flipping_keypoints() -> None:
    tl = TransformList([HFlipTransform(width=10)], flip_indices=None)
    kp = np.zeros((1, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="flip_indices"):
        tl.apply_keypoints(kp)


def test_transform_list_resize_does_not_require_flip_indices() -> None:
    # No HFlip in the list → flip_indices is unused even when None.
    tl = TransformList([ResizeTransform(10, 10, 20, 20)], flip_indices=None)
    kp = np.array([[[1.0, 2.0, 2.0]]], dtype=np.float32)
    out = tl.apply_keypoints(kp)
    np.testing.assert_allclose(out[0, 0, :2], [2.0, 4.0])


def test_transform_list_apply_box_chains_through_multiple_transforms() -> None:
    tl = TransformList(
        [ResizeTransform(10, 20, 20, 40), HFlipTransform(width=40)],
        flip_indices=None,
    )
    box = np.array([[2.0, 2.0, 6.0, 4.0]], dtype=np.float32)
    out = tl.apply_box(box)
    # After resize: (4, 4, 12, 8). After flip on width=40: x' = 40 - x.
    # x0, x1 swap to keep min/max ordering: (28, 4, 36, 8) -> (28, 4, 36, 8).
    np.testing.assert_allclose(out, [[28.0, 4.0, 36.0, 8.0]])


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------


def test_resize_shortest_edge_choice_with_max_clamp() -> None:
    rng = np.random.default_rng(0)
    aug = ResizeShortestEdge((100,), max_size=120, sample_style="choice", rng=rng)
    # 50x200 image: short edge 50 → scale 100/50=2 → would yield (100, 400),
    # but 400 > max_size=120, so scale down to 120/200=0.6 → (30, 120).
    img = np.zeros((50, 200, 3), dtype=np.uint8)
    t = aug.get_transform(img)
    assert isinstance(t, ResizeTransform)
    assert (t.new_h, t.new_w) == (30, 120)


def test_resize_shortest_edge_range_samples_uniformly() -> None:
    rng = np.random.default_rng(42)
    aug = ResizeShortestEdge((400, 800), max_size=10_000, sample_style="range", rng=rng)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    targets: set[int] = set()
    for _ in range(50):
        t = aug.get_transform(img)
        targets.add(t.new_h)  # short edge = h here
    assert min(targets) >= 400 and max(targets) <= 800


def test_resize_shortest_edge_range_requires_two_lengths() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        ResizeShortestEdge((400, 600, 800), max_size=10_000, sample_style="range")


def test_random_flip_zero_prob_never_flips() -> None:
    aug = RandomFlip(prob=0.0)
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    t = aug.get_transform(img)
    # NoOp transform: not horizontal-flip and identity on coords.
    assert not getattr(t, "is_horizontal_flip", False)


def test_random_flip_full_prob_always_flips() -> None:
    aug = RandomFlip(prob=1.0)
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    t = aug.get_transform(img)
    assert isinstance(t, HFlipTransform)


def test_random_flip_prob_must_be_in_unit_interval() -> None:
    with pytest.raises(ValueError, match="prob"):
        RandomFlip(prob=-0.1)
    with pytest.raises(ValueError, match="prob"):
        RandomFlip(prob=1.1)


def test_augmentation_list_mutates_aug_input_image_and_returns_transform_list() -> None:
    rng = np.random.default_rng(0)
    augs = AugmentationList(
        [ResizeShortestEdge((40,), max_size=100, sample_style="choice", rng=rng)]
    )
    image = np.zeros((20, 60, 3), dtype=np.uint8)
    aug_input = AugInput(image=image)
    tl = augs(aug_input)
    # Image got resized in place to short=40 → scale=2 → (40, 120) clamped to
    # max=100 → final scale = min(2, 100/60) = 100/60. new_h ≈ 33, new_w = 100.
    assert isinstance(tl, TransformList)
    assert aug_input.image.shape == (33, 100, 3)


# ---------------------------------------------------------------------------
# Identity / sanity for the protocol
# ---------------------------------------------------------------------------


def test_transform_subclass_must_override_required_methods() -> None:
    class _Bad(Transform):
        pass

    bad = _Bad()
    with pytest.raises(NotImplementedError):
        bad.apply_image(np.zeros((1, 1, 3), dtype=np.uint8))
    with pytest.raises(NotImplementedError):
        bad.apply_coords(np.zeros((0, 2), dtype=np.float32))
