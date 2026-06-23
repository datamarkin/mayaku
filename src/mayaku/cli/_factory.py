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

from torch import nn

from mayaku.cli._weights import resolve_weights
from mayaku.config.schemas import MayakuConfig
from mayaku.data.transforms import Augmentation, LetterboxResize, ResizeShortestEdge
from mayaku.models.detectors import (
    build_faster_rcnn,
    build_keypoint_rcnn,
    build_mask_rcnn,
    build_uniquery,
)
from mayaku.utils.checkpoint import config_from_checkpoint

__all__ = ["build_detector", "build_resize_augmentation", "load_detector"]


def build_resize_augmentation(cfg: MayakuConfig, *, for_train: bool) -> Augmentation:
    """Pick the resize augmentation from ``cfg.input.resize_mode``.

    ``letterbox`` → aspect-preserving resize+pad to a fixed square (train:
    multi-scale ``train_sizes``; inference/eval: the single ``infer_size``).
    ``shortest_edge`` → the legacy variable resize. One place so train, eval, and
    the in-train periodic eval can't drift in how they read the config.
    """
    inp = cfg.input
    if inp.resize_mode == "letterbox":
        return LetterboxResize(inp.train_sizes if for_train else (inp.infer_size,))
    if for_train:
        return ResizeShortestEdge(
            inp.min_size_train,
            max_size=inp.max_size_train,
            sample_style=inp.min_size_train_sampling,
        )
    return ResizeShortestEdge((inp.min_size_test,), max_size=inp.max_size_test)


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

    ``weights`` is a trained ``.pth`` or a hub model name; either resolves to a
    checkpoint whose embedded sidecar defines the architecture. The ``.pth`` is
    read once. The ``num_batches_tracked`` buffers an EMA shadow accumulates are
    dropped — the deploy model uses FrozenBatchNorm2d, which has no such entry.
    Returns ``(cfg, model)``; the caller sets eval mode / device as it needs.
    """
    weights_path = resolve_weights(weights)
    assert weights_path is not None  # resolve_weights only returns None for None input
    cfg, state = config_from_checkpoint(weights_path)
    model = build_detector(cfg)
    state = {k: v for k, v in state.items() if not k.endswith(".num_batches_tracked")}
    model.load_state_dict(state)
    return cfg, model
