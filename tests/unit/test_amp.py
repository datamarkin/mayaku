"""Tests for :mod:`mayaku.backends.amp`."""

from __future__ import annotations

from typing import cast

import torch

from mayaku.backends.amp import NullGradScaler, autocast, make_grad_scaler
from mayaku.backends.device import Device, DeviceKind


def _from_torch(td: torch.device) -> Device:
    return Device(kind=cast(DeviceKind, td.type), index=td.index or 0)


def test_autocast_is_real_on_accelerator_noop_on_cpu(device: torch.device) -> None:
    """Inside ``autocast``, fp32 matmul should downcast on CUDA only.

    On MPS the default autocast dtype is fp32 (per ADR / device.py
    module docstring), so the matmul stays fp32 — the autocast block
    is still entered for plumbing consistency, but produces no dtype
    change. CPU is a no-op context manager.
    """
    dev = _from_torch(device)
    a = torch.randn(4, 4, device=device, dtype=torch.float32)
    b = torch.randn(4, 4, device=device, dtype=torch.float32)

    outside = a @ b
    assert outside.dtype == torch.float32

    with autocast(dev):
        inside = a @ b

    if dev.kind == "cuda":
        assert inside.dtype == torch.float16, (
            f"expected autocast to downcast on cuda, got {inside.dtype}"
        )
    else:
        # MPS default is fp32; CPU has no autocast.
        assert inside.dtype == torch.float32


def test_autocast_fp16_override_on_mps_or_cuda(device: torch.device) -> None:
    """Explicit ``dtype=torch.float16`` forces fp16 even on MPS (opt-in)."""
    dev = _from_torch(device)
    if not dev.supports_amp:
        return  # CPU autocast is a no-op; nothing to assert.
    a = torch.randn(4, 4, device=device, dtype=torch.float32)
    b = torch.randn(4, 4, device=device, dtype=torch.float32)
    with autocast(dev, dtype=torch.float16):
        inside = a @ b
    assert inside.dtype == torch.float16


def test_autocast_yields_none(device: torch.device) -> None:
    dev = _from_torch(device)
    with autocast(dev) as ctx:
        assert ctx is None


def test_make_grad_scaler_type_per_backend(device: torch.device) -> None:
    dev = _from_torch(device)
    scaler = make_grad_scaler(dev)
    if dev.kind == "cuda":
        assert isinstance(scaler, torch.amp.GradScaler)
    else:
        assert isinstance(scaler, NullGradScaler)


def test_null_grad_scaler_methods_do_not_error() -> None:
    """The stub mirrors the public surface and must be safe to call."""
    scaler = NullGradScaler()
    loss = torch.tensor(1.5, requires_grad=True)
    scaled = scaler.scale(loss)
    assert scaled is loss

    param = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.SGD([param], lr=0.1)
    # Build a real gradient so step() has something to act on.
    (param.sum() * 2).backward()

    scaler.unscale_(optimizer)
    assert scaler.step(optimizer) is None
    assert scaler.update() is None
    assert scaler.update(2.0) is None
