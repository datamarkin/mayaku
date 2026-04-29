"""Backend-portable kernel wrappers.

Each op exposes the same Python signature regardless of device. Internally
they prefer the torchvision kernel (CPU + CUDA + MPS in our pinned
versions; see ``BACKEND_PORTABILITY_REPORT.md`` Appendix A) and fall back
to a pure-PyTorch implementation when the runtime kernel is missing.

Deformable convolution is intentionally absent — see
``docs/decisions/001-drop-deformable-convolution.md``.
"""

from __future__ import annotations

from .nms import batched_nms, nms
from .roi_align import roi_align

__all__ = ["batched_nms", "nms", "roi_align"]
