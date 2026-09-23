"""Deployment: Predictor from a checkpoint, ONNX export with the sidecar
embedded, and the exported artifact run by onnxruntime -- all through the one
host decode the evaluator uses."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torchvision.ops import box_iou

from mayaku.config import InputConfig, KeypointConfig, MayakuConfig, ModelConfig
from mayaku.data.batch import batch_to
from mayaku.inference import ArtifactPredictor, Predictor, from_pretrained
from mayaku.inference.decode import decode_sidecar
from mayaku.inference.export import TARGETS, export
from mayaku.inference.export.metadata import embed_sidecar, read_sidecar, strip_tensorrt_header
from mayaku.inference.preprocess import letterbox_batch
from mayaku.utils.checkpoint import build_sidecar, check_sidecar, save_checkpoint

from ._coco_fixture import fixture

ort = pytest.importorskip("onnxruntime")


def _trained_like(model):
    """Random but non-trivial weights with enough confidence to detect."""
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
    torch.manual_seed(0)   # the init too, so the weights do not depend on test order
    model = _trained_like(cfg.model.build(canvas))
    model.train()          # tier n is quantization-aware: give it int8 ranges
    with torch.no_grad():
        model(torch.rand(2, 3, *canvas))
    model.eval()
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
    embedded = check_sidecar(read_sidecar(onnx_path, "onnx"), str(onnx_path))
    assert embedded == {**predictor.sidecar, "export": {"target": "onnx", "precision": "fp32"}}
    a = ArtifactPredictor(onnx_path, "cpu")
    assert a.canvas == predictor.canvas and a.class_names == ds.coco.class_names
    x, _ = letterbox_batch(files, predictor.canvas)
    for gm, wm in zip(a._forward(x), predictor._forward(x), strict=True):
        assert torch.allclose(gm, wm.cpu(), atol=1e-3)
    # decoded: every confident detection has its twin. Near the threshold, or
    # tied in NMS, a 1e-4 difference in the maps may legitimately flip one.
    for g, w in zip(a.batch(files), predictor.batch(files), strict=True):
        for k in (g.scores > 0.5).nonzero().flatten().tolist():
            # its twin: same class, same box, closest score (the untrained
            # model emits many identical clipped boxes)
            twin = (box_iou(g.boxes[k:k + 1], w.boxes)[0] > 0.99) & (w.labels == g.labels[k])
            gap = torch.where(twin, (w.scores - g.scores[k]).abs(), torch.inf)
            j = int(gap.argmin())
            assert gap[j] < 1e-3
            assert torch.allclose(g.keypoints[k, :, :2], w.keypoints[j, :, :2], atol=1e-1)
            # a mask may differ only on its boundary pixels
            assert (g.masks[k] != w.masks[j]).float().mean() < 0.01


def test_from_pretrained_picks_the_backend(run, onnx_path) -> None:
    assert isinstance(from_pretrained(run[1], "cpu"), Predictor)
    assert isinstance(from_pretrained(onnx_path, "cpu"), ArtifactPredictor)


def _runtime(target: str) -> None:
    """Skip unless this host can run `target` artifacts."""
    if not TARGETS[target].module.runnable():
        pytest.skip(f"{target} does not run on this host")
    pytest.importorskip({"onnx": "onnxruntime", "coreml": "coremltools",
                         "openvino": "openvino", "tensorrt": "tensorrt"}[target])


@pytest.mark.slow
@pytest.mark.parametrize(("target", "precision"), [
    ("onnx", "int8"), ("coreml", "fp16"), ("coreml", "int8"), ("openvino", "fp32"),
    ("openvino", "fp16"), ("openvino", "int8"), ("tensorrt", "fp16"), ("tensorrt", "fp32"),
])
def test_every_target_exports_and_runs(run, predictor, tmp_path, target, precision) -> None:
    """Export checks the artifact's maps against the model itself; here the
    artifact also runs end to end through `ArtifactPredictor`."""
    _runtime(target)
    path = predictor.export(target, tmp_path / ("m" + TARGETS[target].suffix), precision)
    a = ArtifactPredictor(path, "cpu" if target != "tensorrt" else "cuda")
    assert a.sidecar["export"] == {"target": target, "precision": precision}
    got, want = a.batch(run[2]), predictor.batch(run[2])
    assert all(abs(len(g) - len(w)) <= max(3, len(w) // 10) for g, w in zip(got, want, strict=True))


def test_export_refuses_what_a_target_or_model_cannot_do(predictor, tmp_path) -> None:
    with pytest.raises(ValueError, match="fp16"):
        predictor.export("onnx", tmp_path / "m.onnx", "fp16")
    with pytest.raises(ValueError, match="unknown export target"):
        predictor.export("tflite", tmp_path / "m.tflite")
    plain = {**predictor.sidecar, "quant": {**predictor.sidecar["quant"], "qat": False}}
    with pytest.raises(ValueError, match="quantization-aware"):
        export(predictor.model, plain, "onnx", tmp_path / "m.onnx", "int8")


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

    monkeypatch.setattr(ex, "parity", lambda m, path, sidecar: (1.0, 1.0))
    with pytest.raises(RuntimeError, match="differ"):
        export(predictor.model, predictor.sidecar, "onnx", tmp_path / "m.onnx")


def test_tensorrt_sidecar_header_round_trip(tmp_path) -> None:
    """An engine has no metadata slot: the sidecar goes in a length-prefixed
    header, which reading recovers and stripping removes byte-exactly."""
    engine = tmp_path / "m.engine"
    engine.write_bytes(b"\x00\x01ENGINE-BYTES")
    embed_sidecar(engine, "tensorrt", {"schema_version": 2, "x": "é"})
    assert read_sidecar(engine, "tensorrt")["x"] == "é"
    assert strip_tensorrt_header(engine) == b"\x00\x01ENGINE-BYTES"
    bare = tmp_path / "bare.engine"
    bare.write_bytes(b"\x00\x01ENGINE-BYTES")
    assert read_sidecar(bare, "tensorrt") is None and strip_tensorrt_header(bare) == bare.read_bytes()
