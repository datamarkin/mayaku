"""Proposal generation: anchors, matcher, sampling, RPN."""

from __future__ import annotations

from mayaku.models.proposals.anchor_generator import (
    DefaultAnchorGenerator,
    build_anchor_generator,
)
from mayaku.models.proposals.box_regression import Box2BoxTransform
from mayaku.models.proposals.matcher import Matcher
from mayaku.models.proposals.rpn import (
    RPN,
    StandardRPNHead,
    build_rpn,
    find_top_rpn_proposals,
)
from mayaku.models.proposals.sampling import subsample_labels

__all__ = [
    "RPN",
    "Box2BoxTransform",
    "DefaultAnchorGenerator",
    "Matcher",
    "StandardRPNHead",
    "build_anchor_generator",
    "build_rpn",
    "find_top_rpn_proposals",
    "subsample_labels",
]
