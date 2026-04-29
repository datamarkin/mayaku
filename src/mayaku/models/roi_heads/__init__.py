"""ROI head dispatchers."""

from __future__ import annotations

from mayaku.models.roi_heads.standard import (
    StandardROIHeads,
    build_standard_roi_heads,
)

__all__ = ["StandardROIHeads", "build_standard_roi_heads"]
