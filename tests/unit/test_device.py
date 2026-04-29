"""Tests for :mod:`mayaku.backends.device`.

The ``device`` fixture (see ``tests/conftest.py``) drives the
backend-sensitive checks so the same file runs meaningfully on CPU,
MPS, and CUDA hosts.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch

from mayaku.backends.device import Device, DeviceKind


def _from_torch(td: torch.device) -> Device:
    """Build a :class:`Device` matching the active torch.device."""
    return Device(kind=cast(DeviceKind, td.type), index=td.index or 0)


def test_construct_cpu() -> None:
    d = Device("cpu")
    assert d.kind == "cpu"
    assert d.index == 0
    assert d.torch == torch.device("cpu")


def test_construct_mps() -> None:
    d = Device("mps")
    assert d.torch == torch.device("mps")


def test_construct_cuda_with_index() -> None:
    d = Device("cuda", 1)
    assert d.torch == torch.device("cuda:1")


def test_frozen() -> None:
    d = Device("cpu")
    with pytest.raises(Exception):  # FrozenInstanceError, but it's stdlib-private
        d.kind = "cuda"  # type: ignore[misc]


def test_torch_round_trip(device: torch.device) -> None:
    """Constructing a Device from the active torch.device round-trips."""
    d = _from_torch(device)
    assert d.torch.type == device.type
    if device.type == "cuda":
        assert d.torch.index == (device.index or 0)


def test_supports_amp_matrix() -> None:
    assert Device("cuda").supports_amp is True
    assert Device("mps").supports_amp is True
    assert Device("cpu").supports_amp is False


def test_amp_dtype_matrix() -> None:
    assert Device("cuda").amp_dtype == torch.float16
    # MPS defaults to fp32 — the autocast block is entered for consistency,
    # but the dtype is a no-op until the user opts into fp16 explicitly.
    # Rationale lives in src/mayaku/backends/device.py module docstring.
    assert Device("mps").amp_dtype == torch.float32
    assert Device("cpu").amp_dtype is None


def test_dist_backend_matrix() -> None:
    assert Device("cuda").dist_backend == "nccl"
    assert Device("mps").dist_backend == "gloo"
    assert Device("cpu").dist_backend == "gloo"


def test_supports_pin_memory_matrix() -> None:
    assert Device("cuda").supports_pin_memory is True
    assert Device("mps").supports_pin_memory is False
    assert Device("cpu").supports_pin_memory is False


def test_synchronize_does_not_error(device: torch.device) -> None:
    """``Device.synchronize`` dispatches per backend and never raises."""
    d = _from_torch(device)
    # Schedule some work first so the call has something to wait on.
    _ = torch.zeros(8, 8, device=device).sum()
    d.synchronize()


def test_auto_picks_active_backend(device: torch.device) -> None:
    """``Device.auto`` should select the backend the fixture resolved to.

    ``MAYAKU_DEVICE`` and ``Device.auto()`` use the same precedence
    (CUDA → MPS → CPU), so on a host where the env var asks for the
    top-priority available accelerator they agree.
    """
    auto = Device.auto()
    if device.type == "cuda":
        assert auto.kind == "cuda"
    elif device.type == "mps":
        # auto() prefers cuda; if cuda is unavailable here, mps wins.
        assert auto.kind in ("cuda", "mps")
        assert torch.backends.mps.is_available()
    else:
        # cpu fixture: auto may still pick an accelerator if one is on
        # the host, since auto() ignores MAYAKU_DEVICE. Just verify it
        # picks *something* sensible.
        assert auto.kind in ("cuda", "mps", "cpu")
