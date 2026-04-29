"""Tests for :mod:`mayaku.structures.keypoints`."""

from __future__ import annotations

import pytest
import torch

from mayaku.structures.keypoints import (
    Keypoints,
    heatmaps_to_keypoints,
    keypoints_to_heatmap,
)

# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


def test_construction_and_shape(device: torch.device) -> None:
    t = torch.zeros(2, 5, 3, device=device)
    kp = Keypoints(t)
    assert len(kp) == 2
    assert kp.num_keypoints == 5
    assert kp.flip_indices is None


def test_flip_indices_must_be_permutation() -> None:
    t = torch.zeros(1, 4, 3)
    # Wrong length
    with pytest.raises(ValueError, match="length K"):
        Keypoints(t, flip_indices=torch.tensor([0, 1, 2]))
    # Not a permutation
    with pytest.raises(ValueError, match="permutation"):
        Keypoints(t, flip_indices=torch.tensor([0, 0, 1, 2]))
    # Valid
    Keypoints(t, flip_indices=torch.tensor([1, 0, 3, 2]))


def test_indexing_preserves_flip_metadata() -> None:
    t = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    kp = Keypoints(t, flip_indices=torch.tensor([1, 0, 3, 2]))
    sub = kp[0]
    assert len(sub) == 1
    assert sub.flip_indices is not None
    assert torch.equal(sub.flip_indices, torch.tensor([1, 0, 3, 2]))


def test_to_moves_both_tensors(device: torch.device) -> None:
    kp = Keypoints(torch.zeros(1, 4, 3), flip_indices=torch.tensor([1, 0, 3, 2]))
    moved = kp.to(device)
    assert moved.tensor.device.type == device.type
    assert moved.flip_indices is not None
    assert moved.flip_indices.device.type == device.type


def test_cat_requires_consistent_flip_metadata() -> None:
    a = Keypoints(torch.zeros(1, 4, 3), flip_indices=torch.tensor([1, 0, 3, 2]))
    b = Keypoints(torch.zeros(1, 4, 3))
    with pytest.raises(ValueError, match="mixed flip"):
        Keypoints.cat([a, b])
    c = Keypoints(torch.zeros(1, 4, 3), flip_indices=torch.tensor([0, 1, 2, 3]))
    with pytest.raises(ValueError, match="differing flip_indices"):
        Keypoints.cat([a, c])
    out = Keypoints.cat([a, Keypoints(torch.zeros(2, 4, 3), flip_indices=a.flip_indices)])
    assert len(out) == 3


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def test_keypoints_to_heatmap_basic(device: torch.device) -> None:
    # Single instance, RoI = (0,0,8,8), heatmap_size=4 → scale = 0.5.
    # Keypoint at (2, 2, vis=2) → (floor(1), floor(1)) = (1, 1). lin = 5.
    keypoints = torch.tensor([[[2.0, 2.0, 2.0]]], device=device)
    rois = torch.tensor([[0.0, 0.0, 8.0, 8.0]], device=device)
    targets, valid = keypoints_to_heatmap(keypoints, rois, heatmap_size=4)
    assert targets.shape == (1, 1)
    assert valid[0, 0].item() is True
    assert int(targets[0, 0].item()) == 1 * 4 + 1


def test_keypoints_to_heatmap_edge_fixup(device: torch.device) -> None:
    # Keypoint exactly at the right/bottom edge → snapped to S-1.
    keypoints = torch.tensor([[[8.0, 8.0, 2.0]]], device=device)
    rois = torch.tensor([[0.0, 0.0, 8.0, 8.0]], device=device)
    targets, valid = keypoints_to_heatmap(keypoints, rois, heatmap_size=4)
    assert valid[0, 0].item() is True
    # x=3, y=3 → lin = 15
    assert int(targets[0, 0].item()) == 15


def test_keypoints_to_heatmap_invisible_marked_invalid(device: torch.device) -> None:
    # v=0 → invalid even if location is in-bounds.
    keypoints = torch.tensor([[[2.0, 2.0, 0.0]]], device=device)
    rois = torch.tensor([[0.0, 0.0, 8.0, 8.0]], device=device)
    targets, valid = keypoints_to_heatmap(keypoints, rois, heatmap_size=4)
    assert valid[0, 0].item() is False
    assert int(targets[0, 0].item()) == 0  # zeroed


def test_keypoints_to_heatmap_out_of_roi_invalid(device: torch.device) -> None:
    keypoints = torch.tensor([[[20.0, 20.0, 2.0]]], device=device)
    rois = torch.tensor([[0.0, 0.0, 8.0, 8.0]], device=device)
    _, valid = keypoints_to_heatmap(keypoints, rois, heatmap_size=4)
    assert valid[0, 0].item() is False


def test_keypoints_to_heatmap_batch_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="batch"):
        keypoints_to_heatmap(torch.zeros(2, 1, 3), torch.zeros(3, 4), heatmap_size=4)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def test_heatmaps_to_keypoints_recovers_argmax(device: torch.device) -> None:
    # Build a heatmap with two clear hot spots and verify the decoded
    # (x, y) land in the right region of image pixel space. Source cell
    # (cy, cx) under a 16x16 ROI starting at (rx, ry) decodes to roughly
    # ((cx + 0.5) * 16/8 + rx, (cy + 0.5) * 16/8 + ry); allow ±2 px slack
    # to absorb the bicubic upsampling argmax shift.
    s = 8
    maps = torch.full((1, 2, s, s), -10.0, device=device)
    maps[0, 0, 3, 5] = 10.0  # (cy=3, cx=5) → image (rx + 11, ry + 7) = (21, 27)
    maps[0, 1, 1, 1] = 10.0  # (cy=1, cx=1) → image (rx + 3,  ry + 3) = (13, 23)
    rois = torch.tensor([[10.0, 20.0, 26.0, 36.0]], device=device)
    out = heatmaps_to_keypoints(maps, rois)
    assert out.shape == (1, 2, 4)
    assert abs(out[0, 0, 0].item() - 21.0) < 2.0
    assert abs(out[0, 0, 1].item() - 27.0) < 2.0
    assert abs(out[0, 1, 0].item() - 13.0) < 2.0
    assert abs(out[0, 1, 1].item() - 23.0) < 2.0
    # Logit at the argmax recovers the source peak (column 2 = raw heatmap
    # value at the argmax). The score column (3) is normalised by the
    # low-resolution partition sum and is small for very peaky single-cell
    # heatmaps; we only assert that it's positive — the magnitude is a
    # property of the per-image pool sum, not of the decoder.
    assert out[0, 0, 2].item() > 0.0
    assert out[0, 1, 2].item() > 0.0
    assert out[0, 0, 3].item() > 0.0
    assert out[0, 1, 3].item() > 0.0


def test_heatmaps_to_keypoints_empty_input(device: torch.device) -> None:
    maps = torch.zeros(0, 2, 8, 8, device=device)
    rois = torch.zeros(0, 4, device=device)
    out = heatmaps_to_keypoints(maps, rois)
    assert out.shape == (0, 2, 4)


def test_heatmaps_to_keypoints_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="rois shape"):
        heatmaps_to_keypoints(torch.zeros(2, 1, 4, 4), torch.zeros(3, 4))


def test_heatmaps_to_keypoints_logit_column_is_raw(device: torch.device) -> None:
    # The third column should be the raw logit (before exp/normalization).
    s = 4
    maps = torch.full((1, 1, s, s), -2.0, device=device)
    maps[0, 0, 2, 1] = 7.5
    rois = torch.tensor([[0.0, 0.0, 4.0, 4.0]], device=device)
    out = heatmaps_to_keypoints(maps, rois)
    # Bicubic on a 4x4 → 4x4 is identity, so the argmax cell value (7.5)
    # is what we should read back as logit.
    assert abs(out[0, 0, 2].item() - 7.5) < 1e-3
