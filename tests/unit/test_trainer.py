"""The training loop: parameter groups, schedule, EMA, and a real tiny run."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from mayaku.data import DEFAULT_AUG
from mayaku.engine.trainer import BASE, EMA, build_optimizer, cosine, param_groups, set_lr, train
from mayaku.model import TINY, Detector

from ._coco_fixture import fixture


def test_param_groups_cover_every_parameter_once() -> None:
    model = Detector(TINY, 4)
    groups = param_groups(model, 5e-4)
    sizes = [sum(p.numel() for p in g["params"]) for g in groups]
    assert sum(sizes) == sum(p.numel() for p in model.parameters())
    assert groups[1]["weight_decay"] == groups[2]["weight_decay"] == 0.0
    assert [g["bias"] for g in groups] == [False, False, True]


def test_cosine() -> None:
    assert abs(cosine(BASE, 0) - 1.0) < 1e-9
    assert abs(cosine(BASE, BASE.epochs) - BASE.lr_final_frac) < 1e-9
    assert cosine(BASE, 62) < cosine(BASE, 0)


def test_warmup_hands_off_to_the_cosine() -> None:
    opt = build_optimizer(Detector(TINY, 4), BASE, accumulate=4)
    set_lr(opt, BASE, 0, 0, 100)
    # only the bias group starts above zero; a BatchNorm gain must not
    assert [g["lr"] for g in opt.param_groups] == [0.0, 0.0, BASE.warmup_bias_lr_start]
    set_lr(opt, BASE, 3, 100, 100)
    assert all(abs(g["lr"] - BASE.lr * cosine(BASE, 3)) < 1e-12 for g in opt.param_groups)


def test_ema_lags_the_model() -> None:
    model = Detector(TINY, 4)
    ema = EMA(model, decay=0.9, tau=1.0)
    ref = next(model.parameters()).clone()
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    ema.update()
    moved = (next(ema.model.parameters()) - ref).abs().mean().item()
    assert 0.0 < moved < 1.0


@pytest.mark.slow
@pytest.mark.parametrize("qat", [False, True], ids=["fp32", "qat"])
def test_tiny_run_learns(tmp_path, qat) -> None:
    """32 synthetic images through a tiny network: AP has to leave zero. The
    gate is loose on purpose -- CPU convolution is not bit-deterministic --
    so it asserts that training happened, not that it reached a number.

    `lr_ref_batch` is 4, not 64: at batch 4 the default would accumulate 16
    batches per update and 36 epochs on 32 images would be 18 steps."""
    r = dataclasses.replace(
        BASE, epochs=36, batch=4, lr_ref_batch=4, canvas=192, lr=0.02,
        final_epochs=8, warmup_epochs=1.0, warmup_iters_min=20,
        assigner_warmup=2, amp=False, qat=qat, recalibrate_images=32)
    tr = fixture(tmp_path / "train", n=32, canvas=r.canvas, aug=DEFAULT_AUG, seed=3)
    va = fixture(tmp_path / "val", n=32, canvas=r.canvas, seed=3)
    best, _, records = train(Detector(TINY, tr.nc, r.canvas), tr, va, r, eval_every=12,
                             log_every=0, out=str(tmp_path / "run"), log=lambda *_: None)
    assert best["AP50"] > 0.08
    assert records[-1]["cls"] < records[0]["cls"] / 1.5
    assert records[-1]["box"] < records[0]["box"] / 1.5
    assert (tmp_path / "run" / "best.pt").exists()
