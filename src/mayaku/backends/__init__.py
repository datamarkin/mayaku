"""Backend facade: the `Device` descriptor for CUDA, MPS and CPU."""

from __future__ import annotations

from mayaku.backends.device import Device, DeviceKind

__all__ = ["Device", "DeviceKind"]
