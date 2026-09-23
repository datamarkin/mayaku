"""Deployment: Predictor from a checkpoint, ONNX export with the sidecar
embedded, and the exported artifact run by onnxruntime -- all through the one
host decode the evaluator uses."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from mayaku.config import InputConfig, KeypointConfig, MayakuConfig, ModelConfig
from mayaku.data.batch import batch_to
from mayaku.inference import ArtifactPredictor, Predictor, from_pretrained
from mayaku.inference.decode import decode_sidecar
from mayaku.inference.export import export
from mayaku.inference.export.metadata import read_sidecar
from mayaku.utils.checkpoint import build_sidecar, check_sidecar, save_checkpoint

from ._coco_fixture import fixture

ort = pytest.importorskip("onnxruntime")


def _trained_like(model):
    """Random but non-trivial weights with enough confidence to detect."""
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        for c in model.head.cls:
            c.bias.add_(8.0)
    return model


@pytest.fixture(scope="module", params=[(128, 192), (160, 160)], ids=["rect", "square"])
def run(request, tmp_path_factory):
    """A tiny tier-n-shaped checkpoint with masks and keypoints, and its data."""
    root = tmp_path_factory.mktemp("run")
    canvas = request.param
    ds = fixture(root / "data", n=6, canvas=canvas, masks=True, kpt=3)
    cfg = MayakuConfig(
        model=ModelConfig(tier="n", num_classes=ds.nc, seg=True,
                          keypoints=KeypointConfig(num=3, names=("tl", "ctr", "br"))),
        input=InputConfig(canvas_hw=canvas),
        train=dataclasses.replace(MayakuConfig().train,
                                  decode=dataclasses.replace(MayakuConfig().train.decode, conf=0.3)))
    model = _trained_like(cfg.model.build(canvas))
    ckpt = root / "best.pt"
    save_checkpoint(model, ckpt, build_sidecar(cfg, ds.coco.class_names, model))
    files = [str(f) for f in list(ds.coco.files)[:3]]
    return ds, ckpt, files


@pytest.fixture(scope="module")
def predictor(run):
    return Predictor.from_checkpoint(run[1], "cpu")


@pytest.fixture(scope="module")
def onnx_path(predictor, tmp_path_factory):
    return predictor.export("onnx", tmp_path_factory.mktemp("onnx") / "model.onnx")


def test_predictor_matches_the_evaluation_path(run, predictor) -> None:
    """Image files through the Predictor give exactly the detections the
    dataset's eval path + decode give: same preprocessing, same decode."""
    ds, _, files = run
    got = predictor.batch(files)
    samples = [ds[i] for i in range(len(files))]
    x = batch_to(torch.stack([s["img"] for s in samples]), "cpu")
    want = decode_sidecar(predictor.model(x), [s["meta"] for s in samples], predictor.sidecar)
    assert sum(len(d) for d in got) > 0
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        assert torch.equal(g.boxes, w.boxes) and torch.equal(g.scores, w.scores)
        assert torch.equal(g.masks, w.masks) and torch.equal(g.keypoints, w.keypoints)
        assert tuple(g.masks.shape[1:]) == tuple(ds.shapes[i])


def test_predictor_accepts_rgb_arrays(run, predictor) -> None:
    import cv2

    files = run[2]
    a = predictor(files[0])
    b = predictor(cv2.imread(files[0])[:, :, ::-1])
    assert torch.equal(a.boxes, b.boxes) and torch.equal(a.labels, b.labels)


def test_predictor_leaves_the_given_model_alone(run) -> None:
    from mayaku.model.quant import is_qat
    from mayaku.utils.checkpoint import read_deploy_checkpoint

    sidecar, cfg, _ = read_deploy_checkpoint(run[1])
    model = cfg.model.build(cfg.input.canvas_hw, len(sidecar["class_names"]))
    Predictor(model, sidecar, "cpu")
    assert model.training is True and is_qat(model)
    assert any(type(m).__name__ == "RepConv3x3" for m in model.modules())   # not fused


def test_onnx_export_embeds_the_sidecar_and_runs_the_same(run, predictor, onnx_path) -> None:
    ds, _, files = run
    assert check_sidecar(read_sidecar(onnx_path, "onnx"), str(onnx_path)) == predictor.sidecar
    a = ArtifactPredictor(onnx_path, "cpu")
    assert a.canvas == predictor.canvas and a.class_names == ds.coco.class_names
    for g, w in zip(a.batch(files), predictor.batch(files), strict=True):
        assert len(g) == len(w) and torch.equal(g.labels, w.labels)
        assert torch.allclose(g.boxes, w.boxes, atol=1e-2)
        assert torch.allclose(g.scores, w.scores, atol=1e-4)
        assert torch.allclose(g.keypoints[..., :2], w.keypoints[..., :2], atol=1e-1)
        # a mask may differ only on its boundary pixels
        assert (g.masks != w.masks).float().mean() < 0.01


def test_from_pretrained_picks_the_backend(run, onnx_path) -> None:
    assert isinstance(from_pretrained(run[1], "cpu"), Predictor)
    assert isinstance(from_pretrained(onnx_path, "cpu"), ArtifactPredictor)


def test_other_targets_are_not_available_yet(predictor, tmp_path) -> None:
    with pytest.raises(NotImplementedError, match="coreml"):
        predictor.export("coreml", tmp_path / "m.mlpackage")
    (tmp_path / "m.xml").write_text("<net/>")
    with pytest.raises(NotImplementedError, match="openvino"):
        ArtifactPredictor(tmp_path / "m.xml")


def test_an_onnx_file_without_a_sidecar_is_refused(onnx_path, tmp_path) -> None:
    import onnx

    m = onnx.load(str(onnx_path))
    del m.metadata_props[:]
    onnx.save(m, str(tmp_path / "bare.onnx"))
    with pytest.raises(ValueError, match="no mayaku sidecar"):
        ArtifactPredictor(tmp_path / "bare.onnx")


def test_export_rejects_a_graph_that_drifts(predictor, tmp_path, monkeypatch) -> None:
    """The parity check is what stands between a broken exporter and a user."""
    import mayaku.inference.export as ex

    monkeypatch.setattr(ex, "onnx_parity", lambda m, path: (1.0, 1.0))
    with pytest.raises(RuntimeError, match="differ"):
        export(predictor.model, predictor.sidecar, "onnx", tmp_path / "m.onnx")
