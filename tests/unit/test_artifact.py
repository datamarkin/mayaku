"""Tests for artifact loading — ``from_pretrained("model.onnx")`` end-to-end.

The ONNX path runs fully here (onnx + onnxruntime installed); it's the one
exporter that produces a full-detector graph. TensorRT metadata is pure bytes so
it's tested without CUDA. Backbone-only and no-metadata artifacts must be
rejected with clear errors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mayaku.config.schemas import MayakuConfig
from mayaku.inference import ArtifactPredictor, from_pretrained
from mayaku.inference.export.dispatch import build_sample, export_detector
from mayaku.inference.export.metadata import (
    embed_sidecar,
    read_sidecar,
    strip_tensorrt_header,
)
from mayaku.models.detectors import build_faster_rcnn
from mayaku.models.detectors.uniquery import build_uniquery
from mayaku.structures.instances import Instances
from mayaku.utils.checkpoint import build_sidecar

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

_CANVAS = 128  # divisible by 32 for the resnet50 backbone; keeps the toy fast


def _uniquery_cfg(num_classes: int = 4) -> MayakuConfig:
    return MayakuConfig(
        model={
            "meta_architecture": "uniquery",
            "backbone": {"name": "resnet50"},
            "uniquery_head": {"num_proposals": 10, "num_stages": 2},
            "roi_heads": {"num_classes": num_classes},
        }
    )


def _faster_rcnn_cfg() -> MayakuConfig:
    return MayakuConfig(
        model={
            "meta_architecture": "faster_rcnn",
            "backbone": {"name": "resnet50", "freeze_at": 2, "norm": "FrozenBN"},
            "roi_heads": {"num_classes": 3},
        }
    )


def _export_uniquery_onnx(tmp_path: Path, *, num_classes: int = 4, sidecar: bool = True) -> Path:
    torch.manual_seed(0)
    cfg = _uniquery_cfg(num_classes)
    model = build_uniquery(cfg).eval()
    out = tmp_path / "uq.onnx"
    class_names = [f"c{i}" for i in range(num_classes)]
    export_detector(
        model,
        "onnx",
        out,
        sample=build_sample(_CANVAS, _CANVAS),
        sidecar=build_sidecar(cfg, class_names) if sidecar else None,
    )
    return out


# ---------------------------------------------------------------------------
# Metadata round-trip
# ---------------------------------------------------------------------------


def test_onnx_sidecar_round_trips(tmp_path: Path) -> None:
    out = _export_uniquery_onnx(tmp_path, num_classes=4)
    sidecar = read_sidecar(out, "onnx")
    assert sidecar is not None
    assert sidecar["class_names"] == ["c0", "c1", "c2", "c3"]
    # The reconstructed config validates and matches the architecture.
    cfg = MayakuConfig.model_validate(sidecar["config"])
    assert cfg.model.meta_architecture == "uniquery"
    assert cfg.model.roi_heads.num_classes == 4


def test_tensorrt_sidecar_round_trips_and_strips(tmp_path: Path) -> None:
    """TensorRT has no metadata slot — the sidecar is length-prefixed onto the
    engine bytes. Round-trip the JSON and confirm the raw engine is recoverable."""
    engine = tmp_path / "m.engine"
    raw = b"\x00\x01\x02THIS-IS-THE-ENGINE\xff"
    engine.write_bytes(raw)
    embed_sidecar(engine, "tensorrt", {"class_names": ["a", "b"], "config": {"x": 1}})

    sidecar = read_sidecar(engine, "tensorrt")
    assert sidecar is not None and sidecar["class_names"] == ["a", "b"]
    assert strip_tensorrt_header(engine) == raw


# ---------------------------------------------------------------------------
# ONNX end-to-end
# ---------------------------------------------------------------------------


def test_from_pretrained_onnx_runs_end_to_end(tmp_path: Path) -> None:
    out = _export_uniquery_onnx(tmp_path, num_classes=4)
    predictor = from_pretrained(str(out))
    assert isinstance(predictor, ArtifactPredictor)
    assert predictor.class_names == ["c0", "c1", "c2", "c3"]

    image = np.random.default_rng(0).integers(0, 256, size=(90, 140, 3), dtype=np.uint8)
    inst = predictor(image)

    assert isinstance(inst, Instances)
    assert inst.image_size == (90, 140)  # returned in ORIGINAL image coords
    assert inst.has("pred_boxes") and inst.has("scores") and inst.has("pred_classes")
    n = len(inst)
    assert inst.pred_boxes.tensor.shape == (n, 4)
    if n:
        boxes = inst.pred_boxes.tensor
        assert (boxes[:, 0::2] >= -1).all() and (boxes[:, 0::2] <= 141).all()
        assert (boxes[:, 1::2] >= -1).all() and (boxes[:, 1::2] <= 91).all()
        assert int(inst.pred_classes.max()) < 4
        assert float(inst.scores.min()) >= _uniquery_cfg().model.roi_heads.score_thresh_test


def test_artifact_predictor_accepts_image_path(tmp_path: Path) -> None:
    from PIL import Image

    out = _export_uniquery_onnx(tmp_path)
    img_path = tmp_path / "img.png"
    Image.fromarray(
        np.random.default_rng(1).integers(0, 256, size=(64, 80, 3), dtype=np.uint8)
    ).save(img_path)

    predictor = ArtifactPredictor.from_file(out)
    inst = predictor(img_path)
    assert inst.image_size == (64, 80)


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_backbone_only_artifact_rejected(tmp_path: Path) -> None:
    """A Faster R-CNN exports the backbone+FPN body (p2..p6), not a full
    detector — loading it as a runnable artifact must fail clearly."""
    torch.manual_seed(0)
    cfg = _faster_rcnn_cfg()
    model = build_faster_rcnn(cfg).eval()
    out = tmp_path / "rcnn.onnx"
    export_detector(
        model, "onnx", out, sample=build_sample(_CANVAS, _CANVAS),
        sidecar=build_sidecar(cfg, ["a", "b", "c"]),
    )
    with pytest.raises(ValueError, match="backbone-only"):
        ArtifactPredictor.from_file(out)


def test_artifact_without_metadata_rejected(tmp_path: Path) -> None:
    out = _export_uniquery_onnx(tmp_path, sidecar=False)
    with pytest.raises(ValueError, match="no embedded mayaku metadata"):
        ArtifactPredictor.from_file(out)


def test_missing_artifact_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ArtifactPredictor.from_file(tmp_path / "nope.onnx")


# ---------------------------------------------------------------------------
# CoreML / OpenVINO end-to-end (full-detector export → run)
# ---------------------------------------------------------------------------


def _export_uniquery(tmp_path: Path, target: str, filename: str, *, num_classes: int = 4) -> Path:
    torch.manual_seed(0)
    cfg = _uniquery_cfg(num_classes)
    model = build_uniquery(cfg).eval()
    out = tmp_path / filename
    export_detector(
        model,
        target,
        out,
        sample=build_sample(_CANVAS, _CANVAS),
        sidecar=build_sidecar(cfg, [f"c{i}" for i in range(num_classes)]),
    )
    return out


def _assert_runs(predictor: ArtifactPredictor) -> None:
    image = np.random.default_rng(0).integers(0, 256, size=(96, 120, 3), dtype=np.uint8)
    inst = predictor(image)
    assert isinstance(inst, Instances)
    assert inst.image_size == (96, 120)
    assert inst.has("pred_boxes") and inst.has("scores") and inst.has("pred_classes")
    if len(inst):
        assert int(inst.pred_classes.max()) < 4


@pytest.mark.skipif(__import__("platform").system() != "Darwin", reason="CoreML is macOS-only")
def test_from_pretrained_coreml_runs_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("coremltools")
    out = _export_uniquery(tmp_path, "coreml", "uq.mlpackage")
    predictor = from_pretrained(str(out))
    assert isinstance(predictor, ArtifactPredictor)
    assert predictor.class_names == ["c0", "c1", "c2", "c3"]
    _assert_runs(predictor)


def test_from_pretrained_openvino_runs_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("openvino")
    out = _export_uniquery(tmp_path, "openvino", "uq.xml")
    predictor = from_pretrained(str(out))
    assert isinstance(predictor, ArtifactPredictor)
    assert predictor.class_names == ["c0", "c1", "c2", "c3"]
    _assert_runs(predictor)
