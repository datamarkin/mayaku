"""Smoke tests for the conftest wiring.

Verifies that the ``device`` fixture and the ``cuda``/``mps`` markers do
the right thing on whichever machine the suite is running on. Expected
results per ``MAYAKU_DEVICE``:

    cpu  -> 1 pass, 2 skips (cuda + mps)
    mps  -> 2 passes, 1 skip (cuda)
    cuda -> 2 passes, 1 skip (mps)
"""

from __future__ import annotations

import pytest
import torch


def test_device_fixture_works(device: torch.device) -> None:
    """The session fixture returns a usable torch.device for tensor allocation."""
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "mps", "cuda")
    t = torch.zeros(3, 3, device=device)
    assert t.device.type == device.type
    assert t.sum().item() == 0.0


@pytest.mark.cuda
def test_cuda_marker(device: torch.device) -> None:
    """Should run only when MAYAKU_DEVICE=cuda; auto-skipped otherwise."""
    assert device.type == "cuda"


@pytest.mark.mps
def test_mps_marker(device: torch.device) -> None:
    """Should run only when MAYAKU_DEVICE=mps; auto-skipped otherwise."""
    assert device.type == "mps"
