"""QAT activation ranges: recalibration, freezing, and continuation from a checkpoint."""

from __future__ import annotations

import copy

import torch

from mayaku.model import TINY, Detector, enable_qat
from mayaku.model.quant import ActFakeQuant, ranges_frozen, recalibrate_ranges


def _qat_model() -> Detector:
    torch.manual_seed(0)
    return enable_qat(Detector(TINY, 4))


def _ranges(model: torch.nn.Module) -> list[torch.Tensor]:
    return [t.clone() for m in model.modules() if isinstance(m, ActFakeQuant)
            for t in (m.mn, m.mx)]


def _bn_stats(model: torch.nn.Module) -> list[torch.Tensor]:
    return [t.clone() for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)
            for t in (m.running_mean, m.running_var)]


def _batches(n: int = 3, seed: int = 1) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(2, 3, 64, 64, generator=g) * (i + 1) for i in range(n)]


def test_recalibrated_range_is_the_mean_of_batch_ranges() -> None:
    m = _qat_model()
    batches = _batches()
    recalibrate_ranges(m, batches)
    # the first convolution's observer sees the image itself
    obs = m.backbone.stem.c1.dense.conv.act_fq
    want_mn = torch.stack([b.amin() for b in batches]).mean()
    want_mx = torch.stack([b.amax() for b in batches]).mean()
    assert torch.allclose(obs.mn, want_mn) and torch.allclose(obs.mx, want_mx)
    assert obs.momentum == 0.99            # the training momentum is restored


def test_recalibration_is_independent_of_prior_ranges() -> None:
    m = _qat_model()
    fresh = copy.deepcopy(m)
    m.train()
    with torch.no_grad():
        for b in _batches(seed=7):         # training moves the ranges
            m(b)
    fresh.load_state_dict(m.state_dict())  # same weights and BN, other history
    batches = _batches()
    recalibrate_ranges(m, batches)
    recalibrate_ranges(fresh, batches)
    for a, b in zip(_ranges(m), _ranges(fresh), strict=True):
        assert torch.equal(a, b)


def test_range_pass_leaves_batchnorm_and_mode_alone() -> None:
    m = _qat_model().train()
    bn = _bn_stats(m)
    recalibrate_ranges(m, _batches())
    for a, b in zip(bn, _bn_stats(m), strict=True):
        assert torch.equal(a, b)
    assert m.training and all(mod.training for mod in m.modules())


def test_ranges_frozen_keeps_ranges_but_not_batchnorm() -> None:
    m = _qat_model()
    recalibrate_ranges(m, _batches())
    ranges, bn = _ranges(m), _bn_stats(m)
    m.train()
    with torch.no_grad(), ranges_frozen(m):
        m(_batches(1, seed=3)[0])
    assert all(torch.equal(a, b) for a, b in zip(ranges, _ranges(m), strict=True))
    assert not all(torch.equal(a, b) for a, b in zip(bn, _bn_stats(m), strict=True))
    assert all(o.observe for o in m.modules() if isinstance(o, ActFakeQuant))


def test_loaded_ranges_are_continued_not_overwritten() -> None:
    trained = _qat_model()
    recalibrate_ranges(trained, _batches())
    loaded = enable_qat(Detector(TINY, 4))
    loaded.load_state_dict(trained.state_dict())
    obs = loaded.backbone.stem.c1.dense.conv.act_fq
    before = obs.mn.clone()
    x = _batches(1, seed=5)[0]
    loaded.train()
    with torch.no_grad():
        loaded(x)
    assert torch.allclose(obs.mn, 0.99 * before + 0.01 * x.amin())


def test_no_qat_means_no_range_pass() -> None:
    def never():
        raise AssertionError("batches were read")
        yield

    recalibrate_ranges(Detector(TINY, 4), never())
