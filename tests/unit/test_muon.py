"""MuonSGD: orthogonalisation, exact SGD at mix 0, SGD step norm at mix 1."""

from __future__ import annotations

import torch

from mayaku.engine.muon import MuonSGD, orthogonalise


def test_orthogonalise_singular_values() -> None:
    torch.manual_seed(0)
    s = torch.linalg.svdvals(orthogonalise(torch.randn(64, 144)))
    assert 0.5 < s.min() and s.max() < 1.3


def test_mix_zero_is_nesterov_sgd() -> None:
    torch.manual_seed(0)
    w1 = torch.randn(8, 4, 3, 3, requires_grad=True)
    w2 = w1.detach().clone().requires_grad_()
    ref = torch.optim.SGD([w1], lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-3)
    ours = MuonSGD([{"params": [w2], "muon": True, "mix": 0.0, "weight_decay": 1e-3}],
                   lr=0.1, momentum=0.9)
    for _ in range(5):
        for w, opt in ((w1, ref), (w2, ours)):
            opt.zero_grad()
            (w.sin() * torch.arange(w.numel()).view_as(w)).sum().backward()
            opt.step()
    assert (w1 - w2).abs().max().item() < 1e-6


def test_mix_one_keeps_step_norm() -> None:
    torch.manual_seed(0)
    w = torch.randn(8, 4, 3, 3, requires_grad=True)
    opt = MuonSGD([{"params": [w], "muon": True, "mix": 1.0}], lr=1.0, momentum=0.0)
    before = w.detach().clone()
    (w * torch.randn_like(w)).sum().backward()
    gnorm = w.grad.norm().item()
    opt.step()
    moved = (w.detach() - before).norm().item()
    assert abs(moved - gnorm) < 1e-3 * gnorm
