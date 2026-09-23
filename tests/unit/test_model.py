"""The v3 detector: fusion is exact, the deploy graph stays inside the five-op
contract, the output layout is stable, and the aux branch is detachable."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from mayaku.model import STRIDES, TIERS, TINY, Detector, check_parity, enable_qat, load_weights
from mayaku.model.contract import DEPLOY_OPS, count, export_onnx

pytest.importorskip("onnx")

SEG_KPT = dataclasses.replace(TINY, seg=True, kpt=3)


@pytest.mark.parametrize(
    "cfg",
    [TINY, SEG_KPT, dataclasses.replace(SEG_KPT, aux_arm="tower")],
    ids=["tiny", "tiny-seg-kpt", "tiny-seg-kpt-tower"],
)
def test_fuse_parity_and_contract(cfg) -> None:
    err, scale, inv = check_parity(Detector(cfg, 4), 128)
    assert err < 1e-4 * max(scale, 1.0)
    assert set(inv) <= DEPLOY_OPS


@pytest.mark.slow
@pytest.mark.parametrize("tier", sorted(TIERS))
def test_tier_fuse_parity_and_contract(tier: str) -> None:
    err, scale, inv = check_parity(Detector(TIERS[tier], 80), 256)
    assert err < 1e-4 * max(scale, 1.0)
    assert set(inv) <= DEPLOY_OPS


def test_deploy_parameter_counts() -> None:
    """The locked range: deploy (fused) parameters per tier."""
    want = {"n": 4.75, "s": 11.93, "m": 41.35, "l": 81.27}
    for tier, millions in want.items():
        m = Detector(TIERS[tier], 80).fuse()
        params = sum(p.numel() for p in m.parameters())
        assert abs(params / 1e6 - millions) < 0.01, (tier, params)


def test_output_layout() -> None:
    m = Detector(SEG_KPT, 4).eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 3, 128, 96))
    assert len(out) == len(m.out_names) == 6 + 4 + 4
    g = m.split(out)
    assert len(g["head"]) == 6 and len(g["ker"]) == 3 and len(g["kpt"]) == 3
    assert g["mask"].shape == (1, 8, 16, 12)
    assert g["heat"].shape == (1, 3 + 2, 16, 12)
    for i, s in enumerate(STRIDES):
        assert g["head"][2 * i].shape == (1, 4, 128 // s, 96 // s)


def test_fused_export_in_contract(tmp_path) -> None:
    m = Detector(TINY, 4).fuse()
    params, flops = count(m, 128)
    assert params > 0 and flops > 0
    inv = export_onnx(m, str(tmp_path / "tiny.onnx"), 128)
    assert set(inv) <= DEPLOY_OPS


def test_qat_export_stays_in_contract(tmp_path) -> None:
    m = enable_qat(Detector(TINY, 4))
    m.train()
    with torch.no_grad():
        m(torch.randn(2, 3, 64, 64))   # observers see a range
    m.fuse()
    inv = export_onnx(m, str(tmp_path / "qat.onnx"), 64)
    assert set(inv) <= DEPLOY_OPS


def test_load_weights_across_the_aux_boundary() -> None:
    plain, with_aux = Detector(TINY, 4), Detector(SEG_KPT, 4)
    r = load_weights(plain, with_aux.state_dict())
    assert r.unexpected_keys and all(k.startswith("aux.") for k in r.unexpected_keys)
    r = load_weights(with_aux, plain.state_dict())
    assert r.missing_keys and all(k.startswith("aux.") for k in r.missing_keys)
    with pytest.raises(AssertionError):
        load_weights(plain, with_aux.state_dict(), strict_aux=True)
    other = Detector(dataclasses.replace(TINY, neck=(16, 16, 32)), 4)
    with pytest.raises(RuntimeError):
        load_weights(plain, other.state_dict())
