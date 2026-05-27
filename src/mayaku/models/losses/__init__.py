"""Loss modules."""

from __future__ import annotations

from mayaku.models.losses.set_criterion import SetCriterion, generalized_box_iou

__all__ = ["SetCriterion", "generalized_box_iou"]
