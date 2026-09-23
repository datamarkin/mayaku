"""Backend-portable kernel wrappers.

Each op exposes the same Python signature regardless of device. Internally
they prefer the torchvision kernel and fall back to a pure-PyTorch
implementation when the runtime kernel is missing.
"""

from __future__ import annotations

from .nms import batched_nms, nms

__all__ = ["batched_nms", "nms"]
