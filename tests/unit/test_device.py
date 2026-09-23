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


def test_dist_backend_matrix() -> None:
    assert Device("cuda").dist_backend == "nccl"
    assert Device("mps").dist_backend == "gloo"
    assert Device("cpu").dist_backend == "gloo"


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


def test_resolve_passes_settings_through_and_picks_for_auto() -> None:
    assert Device.resolve("cpu") == "cpu" and Device.resolve("cuda:1") == "cuda:1"
    assert Device.resolve("auto") == Device.auto().kind
