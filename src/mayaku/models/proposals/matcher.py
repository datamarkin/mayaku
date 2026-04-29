"""IoU-threshold-based matcher.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §2.3 (`Matcher`,
``modeling/matcher.py:9-127``). Used by the RPN
(``[0.3, 0.7] / [0, -1, 1]`` with ``allow_low_quality_matches=True``)
and by the ROI heads (``[0.5] / [0, 1]`` with
``allow_low_quality_matches=False``).

The matcher is a pure function — no learned state — but lives in a
small class so the thresholds and labels travel together with the
matching call site for readability.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = ["Matcher"]


class Matcher:
    """Assign each prediction to a ground-truth row by IoU thresholds.

    Args:
        thresholds: Strictly ascending list of IoU thresholds. The
            number of bands is ``len(thresholds) + 1``.
        labels: One label per band, in low→high IoU order. ``-1``
            means "ignore" by convention. Must have length
            ``len(thresholds) + 1``.
        allow_low_quality_matches: If ``True``, after the threshold
            assignment, every GT with no anchor above its highest IoU
            forces all anchors at that max-IoU value to label 1
            (foreground). RPN sets this; ROI heads do not
            (`spec §2.3` and `§2.4`).
    """

    def __init__(
        self,
        thresholds: Sequence[float],
        labels: Sequence[int],
        allow_low_quality_matches: bool = False,
    ) -> None:
        if len(labels) != len(thresholds) + 1:
            raise ValueError(
                f"len(labels)={len(labels)} must equal len(thresholds)+1={len(thresholds) + 1}"
            )
        if any(b <= a for a, b in itertools.pairwise(thresholds)):
            raise ValueError(f"thresholds must be strictly ascending; got {thresholds}")
        # Detectron2 internally extends thresholds with -inf and +inf so
        # bucketize can map any IoU into a band (0 .. len(thresholds)).
        self.thresholds: list[float] = [-float("inf"), *thresholds, float("inf")]
        self.labels: list[int] = list(labels)
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix: Tensor) -> tuple[Tensor, Tensor]:
        """Match anchors/proposals to GT.

        Args:
            match_quality_matrix: ``(M, N)`` IoU between ``M`` GT rows
                and ``N`` predictions.

        Returns:
            ``matches``: ``(N,)`` long tensor; for each prediction, the
            index of the best-matching GT row (0 if there are no GTs —
            the corresponding label entry will mark it as background or
            ignore).
            ``match_labels``: ``(N,)`` int8 tensor with values in
            ``self.labels`` (typically ``{-1, 0, 1}``).
        """
        if match_quality_matrix.numel() == 0:
            # No GTs in the image: everything is background.
            n = match_quality_matrix.shape[1]
            device = match_quality_matrix.device
            matches = torch.zeros(n, dtype=torch.long, device=device)
            match_labels = torch.full((n,), self.labels[0], dtype=torch.int8, device=device)
            return matches, match_labels

        matched_vals, matches = match_quality_matrix.max(dim=0)  # (N,)
        # Bucketize each anchor's max IoU into one of the bands.
        thresholds = torch.tensor(
            self.thresholds[1:-1], dtype=matched_vals.dtype, device=matched_vals.device
        )
        # ``right=True`` so an IoU value exactly at a threshold falls into
        # the *higher* band (spec §2.3: "IoU ≥ 0.7 → 1", not "> 0.7").
        band = torch.bucketize(matched_vals, thresholds, right=True)  # (N,)
        labels_t = torch.tensor(self.labels, dtype=torch.int8, device=matched_vals.device)
        match_labels = labels_t[band]

        if self.allow_low_quality_matches:
            self._set_low_quality_matches(match_labels, match_quality_matrix)
        return matches, match_labels

    def _set_low_quality_matches(self, match_labels: Tensor, match_quality_matrix: Tensor) -> None:
        """Force GT-to-anchor matches even when no anchor exceeds the
        high threshold (`spec §2.3`, `matcher.py:106-127`).

        For each GT, find every prediction whose IoU equals that GT's
        maximum IoU and set its label to foreground (1).
        """
        gt_max_iou, _ = match_quality_matrix.max(dim=1, keepdim=True)  # (M, 1)
        # Anchors with the per-GT max IoU. nonzero() returns
        # (gt_idx, pred_idx) pairs.
        forced = (match_quality_matrix == gt_max_iou).nonzero(as_tuple=False)
        if forced.numel() == 0:
            return
        match_labels[forced[:, 1]] = 1
