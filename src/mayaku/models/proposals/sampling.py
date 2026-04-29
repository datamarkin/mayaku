"""Foreground/background subsampling for the RPN and ROI heads.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §2.3 (``modeling/sampling.py``).
Same routine is reused by ``StandardROIHeads`` (`spec §2.4`) with a
different ``num_samples``/``positive_fraction``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["subsample_labels"]


def subsample_labels(
    labels: Tensor,
    num_samples: int,
    positive_fraction: float,
    bg_label: int,
) -> tuple[Tensor, Tensor]:
    """Sample foreground / background indices from a label tensor.

    Args:
        labels: ``(N,)`` int tensor with values in ``{-1, bg_label, fg}``
            where ``fg`` is any value other than ``-1`` or ``bg_label``.
            ``-1`` is the "ignore" sentinel from :class:`Matcher`.
        num_samples: Total samples to draw (foreground + background).
        positive_fraction: Target foreground fraction. Actual fg count is
            ``min(available_fg, int(num_samples * positive_fraction))``;
            background count fills the remainder, capped by available bg.
        bg_label: The numeric value indicating background.

    Returns:
        ``(pos_idx, neg_idx)``: long tensors of indices into ``labels``.
    """
    positive = ((labels != -1) & (labels != bg_label)).nonzero(as_tuple=False).flatten()
    negative = (labels == bg_label).nonzero(as_tuple=False).flatten()

    num_pos = int(num_samples * positive_fraction)
    num_pos = min(num_pos, positive.numel())
    num_neg = num_samples - num_pos
    num_neg = min(num_neg, negative.numel())

    # Permute on the labels' device so MPS / CUDA call stays on-device.
    perm_pos = torch.randperm(positive.numel(), device=labels.device)[:num_pos]
    perm_neg = torch.randperm(negative.numel(), device=labels.device)[:num_neg]
    return positive[perm_pos], negative[perm_neg]
