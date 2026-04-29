"""Detector factory dispatch keyed on ``MayakuConfig.meta_architecture``.

Lifts the per-architecture build factories
(:func:`build_faster_rcnn`, :func:`build_mask_rcnn`,
:func:`build_keypoint_rcnn`) behind a single entry point so the CLI
subcommands and any user script can write
``model = build_detector(cfg)`` without an ``if cfg.model.meta_architecture
== "..."`` ladder per call site.
"""

from __future__ import annotations

from torch import nn

from mayaku.config.schemas import MayakuConfig
from mayaku.models.detectors import (
    build_faster_rcnn,
    build_keypoint_rcnn,
    build_mask_rcnn,
)

__all__ = ["build_detector"]


def build_detector(cfg: MayakuConfig, *, backbone_weights: str | None = None) -> nn.Module:
    """Build the detector that matches ``cfg.model.meta_architecture``.

    ``backbone_weights="DEFAULT"`` is forwarded to the per-architecture
    factory and triggers torchvision's pretrained ImageNet weights for
    the backbone. ``None`` initialises the backbone fresh — what the
    test suite uses.
    """
    arch = cfg.model.meta_architecture
    if arch == "faster_rcnn":
        return build_faster_rcnn(cfg, backbone_weights=backbone_weights)
    if arch == "mask_rcnn":
        return build_mask_rcnn(cfg, backbone_weights=backbone_weights)
    if arch == "keypoint_rcnn":
        return build_keypoint_rcnn(cfg, backbone_weights=backbone_weights)
    raise ValueError(f"unknown meta_architecture {arch!r}")
