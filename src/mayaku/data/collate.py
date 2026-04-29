"""Trivial collate function.

Detectron2 (`spec §5.2`) uses ``trivial_batch_collator`` because the
real batching happens later: ``ImageList.from_tensors`` pads each
image up to ``max(H, W)`` inside the model's ``preprocess_image`` step.
Using PyTorch's default ``default_collate`` here would try to stack
mismatched-size image tensors and crash.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["trivial_batch_collator"]


def trivial_batch_collator(batch: Sequence[Any]) -> list[Any]:
    """Return ``batch`` as a plain list — the model handles padding."""
    return list(batch)
