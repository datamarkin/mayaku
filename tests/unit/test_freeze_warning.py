"""Tests for the freeze_at-without-pretrain warning in :mod:`mayaku.cli.train`.

The warning's job is to catch the silent footgun of freezing early
backbone stages on top of random-init features. Warm-starting a full
mayaku checkpoint via ``--weights`` is the only init source and suppresses
the warning; anything else with ``freeze_at>=1`` trips it.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from mayaku.cli.train import run_train
from mayaku.config import MayakuConfig

_FREEZE_WARN_MATCH = "Random-init frozen features"


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
            num_epochs=1,
        )
