"""Tests for ``mayaku train --resume`` (continue from a checkpoint).

Covers the risky mechanics in isolation — resolving the resume iteration,
fast-forwarding the LR schedule, and the checkpoint carrying its iteration —
plus an end-to-end resume through :func:`mayaku.api.train` on the toy COCO
fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from mayaku.cli.train import (
    _checkpoint_iteration,
    _fast_forward_scheduler,
    run_train,
)

# ---------------------------------------------------------------------------
# _checkpoint_iteration
# ---------------------------------------------------------------------------


def test_checkpoint_iteration_prefers_stored_field() -> None:
    # The stored field wins even if the filename would parse to something else.
    ck = {"iteration": 500}
    assert _checkpoint_iteration(ck, Path("model_iter_0060000.pth")) == 500


def test_checkpoint_iteration_falls_back_to_filename() -> None:
    # Older checkpoints (saved before the field existed) parse from the name.
    assert _checkpoint_iteration({}, Path("run/train/model_iter_0060000.pth")) == 60000


def test_checkpoint_iteration_raises_when_unknown() -> None:
    with pytest.raises(ValueError, match="iteration"):
        _checkpoint_iteration({}, Path("model_final.pth"))


# ---------------------------------------------------------------------------
# _fast_forward_scheduler
# ---------------------------------------------------------------------------


def _make_scheduler() -> tuple[torch.optim.Optimizer, Any]:
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    # Any non-trivial schedule; resume must reproduce its value at start_iter.
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 1.0 / (1 + e))
    return opt, sch


def test_fast_forward_matches_stepping() -> None:
    start_iter = 50
    # Reference: a fresh scheduler stepped start_iter times (what the LRScheduler
    # hook does once per iteration), so the LR during iter `start_iter`.
    ref_opt, ref_sch = _make_scheduler()
    for _ in range(start_iter):
        ref_sch.step()
    ref_lr = ref_opt.param_groups[0]["lr"]

    ff_opt, ff_sch = _make_scheduler()
    _fast_forward_scheduler(ff_sch, start_iter)
    assert ff_sch.last_epoch == start_iter
    assert ff_opt.param_groups[0]["lr"] == pytest.approx(ref_lr)


def test_fast_forward_is_noop_at_zero() -> None:
    opt, sch = _make_scheduler()
    lr_before = opt.param_groups[0]["lr"]
    _fast_forward_scheduler(sch, 0)
    assert opt.param_groups[0]["lr"] == pytest.approx(lr_before)


# ---------------------------------------------------------------------------
# Mutual exclusivity
# ---------------------------------------------------------------------------


def test_resume_rejects_combination_with_weights(
    toy_workspace: dict[str, Any], tmp_path: Path
) -> None:
    # The guard fires before any data/model work, so the paths needn't resolve.
    with pytest.raises(ValueError, match="resume"):
        run_train(
            toy_workspace["cfg"],
            coco_gt_json=toy_workspace["json"],
            image_root=toy_workspace["images"],
            output_dir=tmp_path / "x",
            weights=toy_workspace["weights"],
            resume=tmp_path / "model_iter_0000001.pth",
        )


# ---------------------------------------------------------------------------
# End-to-end resume through mayaku.api.train
# ---------------------------------------------------------------------------


def test_train_then_resume_end_to_end(toy_workspace: dict[str, Any], tmp_path: Path) -> None:
    from mayaku.api import train

    out = tmp_path / "run"
    # Train 2 iters, checkpointing every iter so a mid-run model_iter_*.pth
    # exists to resume from.
    train(
        config=toy_workspace["cfg"],
        train_annotations=toy_workspace["json"],
        train_images=toy_workspace["images"],
        output_dir=out,
        device="cpu",
        overrides={"solver": {"num_epochs": 2, "checkpoint_period": 1}},
    )
    ckpt = out / "train" / "model_iter_0000001.pth"
    assert ckpt.is_file()
    # The checkpoint records its iteration so resume doesn't depend on the name.
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert saved.get("iteration") == 1

    # Resume from iter 1 and run to iter 3 — completes without re-initialising
    # from scratch, and the final checkpoint advances past the resume point.
    out2 = tmp_path / "run_resumed"
    result = train(
        config=toy_workspace["cfg"],
        train_annotations=toy_workspace["json"],
        train_images=toy_workspace["images"],
        output_dir=out2,
        device="cpu",
        overrides={"solver": {"num_epochs": 3, "checkpoint_period": 1}},
        resume=ckpt,
    )
    assert result["output_dir"] == out2
    final = out2 / "train" / "model_final.pth"
    assert final.is_file()
    assert torch.load(final, map_location="cpu", weights_only=False).get("iteration") == 3
