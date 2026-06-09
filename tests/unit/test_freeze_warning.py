"""Tests for the freeze_at-without-pretrain warning in :mod:`mayaku.cli.train`.

The warning's job is to catch the silent footgun of freezing early
backbone stages on top of random-init features. Three init sources
count as "backbone is pretrained" and should suppress it:

  1. ``pretrained_backbone=True`` (torchvision IMAGENET1K_V2)
  2. ``weights`` (a full mayaku checkpoint)
  3. ``cfg.model.backbone.weights_path`` (e.g. DINOv3 for ConvNeXt)

These tests exercise the third source — the first two are already
exercised in :mod:`tests.unit.test_cli`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import cast

import pytest

from mayaku.cli.train import run_train
from mayaku.config import MayakuConfig

_FREEZE_WARN_MATCH = "Random-init frozen features"


def _with_backbone_weights_path(cfg: MayakuConfig, value: str | None) -> MayakuConfig:
    backbone = cfg.model.backbone.model_copy(update={"weights_path": value})
    model = cfg.model.model_copy(update={"backbone": backbone})
    return cfg.model_copy(update={"model": model})


def test_freeze_warning_fires_when_no_init_source(
    toy_workspace: dict[str, Path | object], tmp_path: Path
) -> None:
    """Sanity check: with no init source and freeze_at>=1, the warning fires."""
    cfg = cast(MayakuConfig, toy_workspace["cfg_obj"])
    assert cfg.model.backbone.freeze_at >= 1, "fixture must keep freeze_at>=1 for this test"

    with pytest.warns(UserWarning, match=_FREEZE_WARN_MATCH):
        run_train(
            cfg,
            coco_gt_json=cast(Path, toy_workspace["json"]),
            image_root=cast(Path, toy_workspace["images"]),
            output_dir=tmp_path / "out_no_init",
            device="cpu",
            max_iter=1,
        )


def test_freeze_warning_suppressed_by_weights_path(
    toy_workspace: dict[str, Path | object], tmp_path: Path
) -> None:
    """``cfg.model.backbone.weights_path`` set → warning does NOT fire.

    Uses the toy ResNet config; the ResNet builder currently ignores
    ``weights_path`` (only ConvNeXt consumes it), so we can supply any
    non-None placeholder without the model trying to load a file. The
    predicate under test is purely the boolean check in run_train.
    """
    cfg = _with_backbone_weights_path(
        cast(MayakuConfig, toy_workspace["cfg_obj"]),
        "/placeholder/not-loaded-by-resnet.pth",
    )
    assert cfg.model.backbone.freeze_at >= 1

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        run_train(
            cfg,
            coco_gt_json=cast(Path, toy_workspace["json"]),
            image_root=cast(Path, toy_workspace["images"]),
            output_dir=tmp_path / "out_with_weights_path",
            device="cpu",
            max_iter=1,
        )
    matching = [w for w in captured if _FREEZE_WARN_MATCH in str(w.message)]
    assert matching == [], f"freeze warning should be suppressed; got {matching}"
