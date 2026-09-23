"""The v3 detector: fusion is exact, the deploy graph stays inside the five-op
contract, the output layout is stable, and the aux branch is detachable."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from mayaku.model import (
    STRIDES,
    TIERS,
    TINY,
    Detector,
    check_parity,
    enable_qat,
    load_pretrained,
    load_weights,
)
from mayaku.model.blocks import as_canvas
from mayaku.model.contract import DEPLOY_OPS, count, export_onnx

onnx = pytest.importorskip("onnx")

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
    # with masks and keypoints: that graph contains the detection graph
    cfg = dataclasses.replace(TIERS[tier], seg=True, kpt=17)
    err, scale, inv = check_parity(Detector(cfg, 80), 256)
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


@pytest.mark.parametrize("canvas", [128, (96, 160), (160, 96)])
def test_fused_export_in_contract(tmp_path, canvas) -> None:
    m = Detector(TINY, 4).fuse()
    params, flops = count(m, canvas)
    assert params > 0 and flops > 0
    inv = export_onnx(m, str(tmp_path / "tiny.onnx"), canvas)
    assert set(inv) <= DEPLOY_OPS
    dims = onnx.load(str(tmp_path / "tiny.onnx")).graph.input[0].type.tensor_type.shape.dim
    assert [d.dim_value for d in dims] == [1, 3, *as_canvas(canvas)]


def test_fuse_parity_on_a_rectangular_canvas() -> None:
    err, scale, inv = check_parity(Detector(SEG_KPT, 4), (96, 160))
    assert err < 1e-4 * max(scale, 1.0) and set(inv) <= DEPLOY_OPS


def test_canvas_must_divide_by_the_coarsest_stride() -> None:
    for bad in (100, (96, 100), (0, 32)):
        with pytest.raises(AssertionError, match="multiples of 32"):
            as_canvas(bad)
    assert as_canvas(64) == (64, 64) and as_canvas((96, 160)) == (96, 160)


def test_object_prior_follows_the_canvas() -> None:
    """A canvas with the same cell count gives the same prior, whatever its shape."""
    square = Detector(TINY, 4, 640).head.cls[0].bias
    rect = Detector(TINY, 4, (320, 1280)).head.cls[0].bias
    assert torch.equal(square, rect)
    assert (Detector(TINY, 4, 320).head.cls[0].bias > square).all()


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
    assert r["unexpected"] and all(k.startswith("aux.") for k in r["unexpected"])
    r = load_weights(with_aux, plain.state_dict())
    assert r["missing"] and all(k.startswith("aux.") for k in r["missing"])
    with pytest.raises(AssertionError):
        load_weights(plain, with_aux.state_dict(), strict_aux=True)
    other = Detector(dataclasses.replace(TINY, neck=(16, 16, 32)), 4)
    with pytest.raises(AssertionError, match="does not fit"):
        load_weights(plain, other.state_dict())


def test_load_pretrained_reinitialises_only_the_classifier() -> None:
    base = enable_qat(Detector(SEG_KPT, 80))                 # e.g. a QAT base with aux heads
    target = Detector(TINY, 3)                               # fp32, 3 classes, no aux
    report = load_pretrained(target, base.state_dict())
    assert report["reinitialised"] and all(k.startswith("head.cls.") for k in report["reinitialised"])
    assert all(k.startswith("aux.") or ".act_fq." in k for k in report["unexpected"])
    assert torch.equal(target.head.box[0].weight, base.head.box[0].weight)
    with pytest.raises(AssertionError, match="does not fit"):
        load_pretrained(Detector(dataclasses.replace(TINY, neck=(16, 16, 32)), 3), base.state_dict())
