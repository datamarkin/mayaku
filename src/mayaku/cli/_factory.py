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
from mayaku.tuning.sizing import multi_scale_canvases, resolve_deploy_canvas
from mayaku.utils.checkpoint import config_from_checkpoint

__all__ = ["build_detector", "build_resize_augmentation", "load_detector"]


def build_resize_augmentation(cfg: MayakuConfig, *, for_train: bool) -> Augmentation:
    """Pick the resize augmentation from ``cfg.input.resize_mode``.

    ``letterbox`` → aspect-preserving resize+pad to the resolved canvas under the
    ``size_budget**2`` budget (train: multi-scale down to ``train_scale_min``;
    inference/eval: the single deploy canvas). The canvas is ``canvas_hw`` when
    resolved (data-health / manual), else the largest aligned square in budget.
    ``shortest_edge`` → the legacy variable resize. One place so train, eval, and
    the in-train periodic eval can't drift in how they read the config.
    """
    inp = cfg.input
    if inp.resize_mode == "letterbox":
        budget = inp.size_budget * inp.size_budget
        aspect = inp.canvas_hw[1] / inp.canvas_hw[0] if inp.canvas_hw is not None else 1.0
        if for_train:
            return LetterboxResize(
                multi_scale_canvases(budget, aspect, scale_min=inp.train_scale_min)
            )
        return LetterboxResize([resolve_deploy_canvas(inp.canvas_hw, inp.size_budget)])
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
    if weights_path is None:  # weights is non-None here; this only narrows the type
        raise ValueError(f"could not resolve a checkpoint from {weights!r}")
    cfg, state = config_from_checkpoint(weights_path)
    # Deploy builds the architecture only — the checkpoint's state_dict carries
    # every trained weight, so don't re-load the pretrained backbone. Its file
    # lives on the training host and need not exist where we deploy; loading it
    # would be wasted I/O at best and a FileNotFoundError at worst.
    #
    # The config is frozen (immutable), so clearing a nested field means rebuilding
    # the chain bottom-up — backbone → model → cfg — one model_copy per level.
    backbone = cfg.model.backbone
    if backbone.weights_path is not None:
        model_cfg = cfg.model.model_copy(
            update={"backbone": backbone.model_copy(update={"weights_path": None})}
        )
        cfg = cfg.model_copy(update={"model": model_cfg})
    model = build_detector(cfg)
    state = {k: v for k, v in state.items() if not k.endswith(".num_batches_tracked")}
    model.load_state_dict(state)
    return cfg, model
