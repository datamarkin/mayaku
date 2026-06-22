"""Detector factory dispatch keyed on ``MayakuConfig.meta_architecture``.

Lifts the per-architecture build factories
(:func:`build_faster_rcnn`, :func:`build_mask_rcnn`,
:func:`build_keypoint_rcnn`) behind a single entry point so the CLI
subcommands and any user script can write
``model = build_detector(cfg)`` without an ``if cfg.model.meta_architecture
== "..."`` ladder per call site.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mayaku.cli._weights import config_from_weights
from mayaku.config.schemas import MayakuConfig
from mayaku.models.detectors import (
    build_faster_rcnn,
    build_keypoint_rcnn,
    build_mask_rcnn,
    build_uniquery,
)

__all__ = ["build_detector", "load_detector"]


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
    if arch == "uniquery":
        return build_uniquery(cfg, backbone_weights=backbone_weights)
    raise ValueError(f"unknown meta_architecture {arch!r}")


def load_detector(weights: Path | str) -> tuple[MayakuConfig, nn.Module]:
    """Build a detector from a self-describing checkpoint and load its weights.

    Reads the architecture from ``weights``' embedded sidecar (or a bundled
    model name), builds the matching detector, and loads the state. The
    ``num_batches_tracked`` buffers an EMA shadow accumulates are dropped — the
    deploy model uses FrozenBatchNorm2d, which has no such entry. Returns
    ``(cfg, model)``; the caller sets eval mode / device as it needs.
    """
    cfg, weights_path, _ = config_from_weights(weights)
    model = build_detector(cfg)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {k: v for k, v in state.items() if not k.endswith(".num_batches_tracked")}
    model.load_state_dict(state)
    return cfg, model
