"""Tests for :mod:`mayaku.models.proposals.matcher` and ``sampling``."""

from __future__ import annotations

import pytest
import torch

from mayaku.models.proposals.matcher import Matcher
from mayaku.models.proposals.sampling import subsample_labels

# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


def test_matcher_assigns_labels_per_band() -> None:
    m = Matcher([0.3, 0.7], [0, -1, 1])
    # 1 GT, 4 anchors with IoUs 0.1, 0.4, 0.7, 0.95.
    iou = torch.tensor([[0.1, 0.4, 0.7, 0.95]])
    matches, labels = m(iou)
    assert matches.tolist() == [0, 0, 0, 0]
    assert labels.tolist() == [0, -1, 1, 1]


def test_matcher_picks_best_gt() -> None:
    m = Matcher([0.5], [0, 1])
    # 2 GTs, 3 anchors. anchor 1 best matches GT 1; anchor 2 best matches GT 0.
    iou = torch.tensor(
        [
            [0.1, 0.2, 0.9],
            [0.1, 0.8, 0.4],
        ]
    )
    matches, labels = m(iou)
    assert matches.tolist() == [0, 1, 0]
    assert labels.tolist() == [0, 1, 1]


def test_matcher_no_gt_marks_everything_background() -> None:
    m = Matcher([0.5], [0, 1])
    iou = torch.zeros(0, 5)
    matches, labels = m(iou)
    assert matches.tolist() == [0, 0, 0, 0, 0]
    assert labels.tolist() == [0, 0, 0, 0, 0]


def test_matcher_low_quality_match_forces_foreground() -> None:
    # Single GT with max IoU 0.4 — below the high threshold (0.7) so
    # without low-quality matching it would land in the ignore band.
    m_off = Matcher([0.3, 0.7], [0, -1, 1], allow_low_quality_matches=False)
    m_on = Matcher([0.3, 0.7], [0, -1, 1], allow_low_quality_matches=True)
    iou = torch.tensor([[0.1, 0.4, 0.2]])
    _, labels_off = m_off(iou)
    _, labels_on = m_on(iou)
    assert labels_off.tolist() == [0, -1, 0]
    # The anchor at the GT's max IoU (column 1) is forced to fg.
    assert labels_on.tolist() == [0, 1, 0]


def test_matcher_threshold_label_arity_validation() -> None:
    with pytest.raises(ValueError, match="len"):
        Matcher([0.5], [0])
    with pytest.raises(ValueError, match="ascending"):
        Matcher([0.5, 0.5], [0, 1, 1])


# ---------------------------------------------------------------------------
# subsample_labels
# ---------------------------------------------------------------------------


def test_subsample_labels_respects_positive_fraction() -> None:
    torch.manual_seed(0)
    labels = torch.cat([torch.full((20,), 1), torch.full((100,), 0), torch.full((50,), -1)])
    pos, neg = subsample_labels(labels, num_samples=40, positive_fraction=0.25, bg_label=0)
    # 40 * 0.25 = 10 fg, balance = 30 bg.
    assert pos.numel() == 10
    assert neg.numel() == 30
    # All sampled indices must point at the right label.
    assert (labels[pos] == 1).all()
    assert (labels[neg] == 0).all()


def test_subsample_labels_respects_availability() -> None:
    # Not enough fg → take all of them; bg fills the rest.
    labels = torch.cat([torch.full((3,), 1), torch.full((100,), 0)])
    pos, neg = subsample_labels(labels, num_samples=20, positive_fraction=0.5, bg_label=0)
    assert pos.numel() == 3  # capped at the 3 available
    assert neg.numel() == 17  # 20 - 3


def test_subsample_labels_drops_ignore_index() -> None:
    labels = torch.tensor([-1, 1, -1, 0, 0, -1])
    pos, neg = subsample_labels(labels, num_samples=10, positive_fraction=0.5, bg_label=0)
    assert (labels[pos] == 1).all()
    assert (labels[neg] == 0).all()
    # Ignore-labelled rows never appear in either set.
    for idx in pos.tolist() + neg.tolist():
        assert labels[idx] != -1


def test_subsample_labels_empty_input() -> None:
    labels = torch.tensor([], dtype=torch.long)
    pos, neg = subsample_labels(labels, num_samples=10, positive_fraction=0.5, bg_label=0)
    assert pos.numel() == 0
    assert neg.numel() == 0
