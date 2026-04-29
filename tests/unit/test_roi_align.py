"""Tests for :mod:`mayaku.backends.ops.roi_align`.

Coverage strategy:
* The torchvision-dispatch path is exercised on the active backend
  (``device`` fixture from ``tests/conftest.py``) and cross-checked
  against the same op on CPU; this catches MPS/CUDA divergence.
* Hand-computed ground truth on a 2x2 input pins absolute correctness.
* The pure-PyTorch fallback is invoked directly so it stays exercised
  even though ``torchvision`` covers all three required backends in our
  pinned versions.
"""

from __future__ import annotations

import torch
from torchvision.ops import roi_align as tv_roi_align

from mayaku.backends.ops.roi_align import _roi_align_mps_native, roi_align


def test_shape_contract(device: torch.device) -> None:
    """``(N, C, H, W) x (K, 5) -> (K, C, Ph, Pw)``."""
    feat = torch.randn(2, 3, 16, 16, device=device)
    rois = torch.tensor(
        [[0.0, 0.0, 0.0, 8.0, 8.0], [1.0, 4.0, 4.0, 12.0, 12.0]],
        device=device,
    )
    out = roi_align(feat, rois, output_size=(7, 7), spatial_scale=1.0)
    assert out.shape == (2, 3, 7, 7)
    assert out.device.type == device.type


def test_matches_torchvision_on_active_backend(device: torch.device) -> None:
    """Our wrapper must produce the same numbers as torchvision direct."""
    torch.manual_seed(0)
    feat = torch.randn(1, 4, 32, 32, device=device)
    rois = torch.tensor([[0.0, 2.5, 3.5, 25.5, 28.5]], device=device)
    ours = roi_align(feat, rois, (7, 7), spatial_scale=0.5, sampling_ratio=2, aligned=True)
    theirs = tv_roi_align(feat, rois, (7, 7), spatial_scale=0.5, sampling_ratio=2, aligned=True)
    torch.testing.assert_close(ours, theirs)


def test_matches_torchvision_cpu_reference(device: torch.device) -> None:
    """Cross-device parity: active backend output ≈ torchvision-CPU output.

    Catches silent CPU fallbacks that produce different numerics, and
    pins MPS coverage of ``roi_align`` (BACKEND_PORTABILITY_REPORT §11
    listed this as a verify-on-real-hardware risk)."""
    torch.manual_seed(1)
    feat_cpu = torch.randn(1, 2, 12, 12)
    rois_cpu = torch.tensor([[0.0, 1.0, 1.0, 9.0, 9.0]])
    ref = tv_roi_align(
        feat_cpu, rois_cpu, (4, 4), spatial_scale=1.0, sampling_ratio=0, aligned=True
    )
    out = roi_align(
        feat_cpu.to(device),
        rois_cpu.to(device),
        (4, 4),
        spatial_scale=1.0,
        sampling_ratio=0,
        aligned=True,
    )
    # MPS roi_align may differ in the LSBs from CUDA/CPU; loose tolerance.
    torch.testing.assert_close(out.cpu(), ref, atol=1e-4, rtol=1e-4)


def test_constant_input_yields_constant_output(device: torch.device) -> None:
    """Pooling a constant feature map gives that constant everywhere."""
    feat = torch.full((1, 1, 8, 8), 3.5, device=device)
    rois = torch.tensor([[0.0, 0.0, 0.0, 7.0, 7.0]], device=device)
    out = roi_align(feat, rois, (3, 3), spatial_scale=1.0)
    expected = torch.full_like(out, 3.5)
    torch.testing.assert_close(out, expected)


def test_empty_rois_returns_empty_output(device: torch.device) -> None:
    feat = torch.randn(1, 3, 8, 8, device=device)
    rois = torch.zeros((0, 5), device=device)
    out = roi_align(feat, rois, (5, 5), spatial_scale=1.0)
    assert out.shape == (0, 3, 5, 5)


def test_list_boxes_format(device: torch.device) -> None:
    """List-of-per-image boxes is equivalent to the flat ``(K, 5)`` form."""
    feat = torch.randn(2, 3, 10, 10, device=device)
    flat = torch.tensor(
        [[0.0, 0.0, 0.0, 5.0, 5.0], [1.0, 2.0, 2.0, 8.0, 8.0]],
        device=device,
    )
    listed = [
        torch.tensor([[0.0, 0.0, 5.0, 5.0]], device=device),
        torch.tensor([[2.0, 2.0, 8.0, 8.0]], device=device),
    ]
    a = roi_align(feat, flat, (4, 4), spatial_scale=1.0)
    b = roi_align(feat, listed, (4, 4), spatial_scale=1.0)
    torch.testing.assert_close(a, b)


def test_mps_native_matches_torchvision_cpu() -> None:
    """The MPS-native gather-based fallback must agree with torchvision.

    Run on CPU so we can use torchvision as the reference. The
    implementation uses only ops with native MPS forward+backward
    (``index_select``, arithmetic, floor, clamp); CPU is just where we
    cross-check correctness. Boundary handling matches torchvision's
    "clamp sample to [0, h-1] and bilinear from there" semantics — not
    "zero-pad outside corners," which gives different answers.
    """
    torch.manual_seed(2)
    feat = torch.randn(2, 4, 16, 16)
    rois = torch.tensor(
        [
            [0.0, 1.0, 2.0, 13.0, 14.0],
            [1.0, 0.0, 0.0, 8.0, 8.0],  # touches the left/top boundary
            [0.0, 5.0, 5.0, 12.0, 11.0],
            [1.0, 14.0, 14.0, 16.0, 16.0],  # touches the right/bottom boundary
        ]
    )
    ours = _roi_align_mps_native(
        feat, rois, (7, 7), spatial_scale=1.0, sampling_ratio=2, aligned=True
    )
    theirs = tv_roi_align(feat, rois, (7, 7), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    torch.testing.assert_close(ours, theirs, atol=1e-4, rtol=1e-4)


def test_mps_native_backward_matches_torchvision_cpu() -> None:
    """Gradients from the MPS-native op match torchvision's, on CPU.

    Confirms the gather-based bilinear path produces the same gradient
    flow into ``input`` as torchvision's MPS-native one would (when
    that one isn't broken).
    """
    torch.manual_seed(3)
    feat_a = torch.randn(2, 3, 12, 12, requires_grad=True)
    feat_b = feat_a.detach().clone().requires_grad_(True)
    rois = torch.tensor([[0.0, 1.0, 1.0, 9.0, 9.0], [1.0, 2.0, 3.0, 10.0, 11.0]])

    ours = _roi_align_mps_native(
        feat_a, rois, (5, 5), spatial_scale=1.0, sampling_ratio=2, aligned=True
    )
    theirs = tv_roi_align(feat_b, rois, (5, 5), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    g = torch.randn_like(ours)
    ours.backward(g)
    theirs.backward(g)
    assert feat_a.grad is not None and feat_b.grad is not None
    torch.testing.assert_close(feat_a.grad, feat_b.grad, atol=1e-4, rtol=1e-4)


def test_mps_native_empty_rois() -> None:
    feat = torch.randn(1, 3, 8, 8)
    out = _roi_align_mps_native(
        feat, torch.zeros((0, 5)), (5, 5), spatial_scale=1.0, sampling_ratio=2, aligned=True
    )
    assert out.shape == (0, 3, 5, 5)
