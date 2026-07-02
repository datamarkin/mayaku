"""Tests for :mod:`mayaku.inference.predictor`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mayaku.config.schemas import (
    BackboneConfig,
    MayakuConfig,
    ModelConfig,
    ROIBoxHeadConfig,
    ROIHeadsConfig,
    RPNConfig,
)
from mayaku.inference import Predictor, from_pretrained
from mayaku.models.detectors import build_faster_rcnn
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances


def _tiny_cfg() -> MayakuConfig:
    return MayakuConfig(
        model=ModelConfig(
            meta_architecture="faster_rcnn",
            backbone=BackboneConfig(name="resnet50", freeze_at=2, norm="FrozenBN"),
            rpn=RPNConfig(
                pre_nms_topk_train=200,
                pre_nms_topk_test=100,
                post_nms_topk_train=50,
                post_nms_topk_test=20,
                batch_size_per_image=32,
            ),
            roi_heads=ROIHeadsConfig(num_classes=2, batch_size_per_image=16),
            roi_box_head=ROIBoxHeadConfig(num_fc=1, fc_dim=64),
        ),
    )


def _tiny_predictor(device: torch.device) -> Predictor:
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = build_faster_rcnn(cfg).to(device).eval()
    # Smaller test sizes than the spec defaults — keeps the toy model fast.
    return Predictor(model, min_size_test=64, max_size_test=128, device=device)


# ---------------------------------------------------------------------------
# Single-image numpy input
# ---------------------------------------------------------------------------


def test_predictor_returns_instances_on_numpy_input(device: torch.device) -> None:
    predictor = _tiny_predictor(device)
    image = np.random.default_rng(0).integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    out = predictor(image)
    assert isinstance(out, Instances)
    assert out.image_size == (96, 96)
    # Required prediction fields present (may be empty Instances).
    assert out.has("pred_boxes")
    assert out.has("scores")
    assert out.has("pred_classes")
    assert isinstance(out.pred_boxes, Boxes)


def test_predictor_rescales_predictions_to_original_image_size(
    device: torch.device,
) -> None:
    predictor = _tiny_predictor(device)
    # Non-square so we can check both axes.
    image = np.random.default_rng(0).integers(0, 256, size=(80, 120, 3), dtype=np.uint8)
    out = predictor(image)
    assert out.image_size == (80, 120)
    if len(out) > 0:
        b = out.pred_boxes.tensor
        assert (b[:, 0] >= 0).all() and (b[:, 1] >= 0).all()
        assert (b[:, 2] <= 120).all() and (b[:, 3] <= 80).all()


def test_predictor_reads_from_path_input(device: torch.device, tmp_path: Path) -> None:
    predictor = _tiny_predictor(device)
    rgb = np.random.default_rng(1).integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
    path = tmp_path / "tiny.png"
    Image.fromarray(rgb).save(path)
    out = predictor(path)
    assert out.image_size == (64, 96)
    # str path also works (mirrors the ndarray path).
    out2 = predictor(str(path))
    assert out2.image_size == (64, 96)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_predictor_rejects_wrong_shape(device: torch.device) -> None:
    predictor = _tiny_predictor(device)
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        predictor(np.zeros((64, 64), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        predictor(np.zeros((64, 64, 4), dtype=np.uint8))


def test_predictor_rejects_zero_test_sizes(device: torch.device) -> None:
    cfg = _tiny_cfg()
    model = build_faster_rcnn(cfg).to(device).eval()
    with pytest.raises(ValueError, match="min_size_test"):
        Predictor(model, min_size_test=0, max_size_test=128)


def test_predictor_rejects_pinned_memory_without_gpu_preprocess(
    device: torch.device,
) -> None:
    cfg = _tiny_cfg()
    model = build_faster_rcnn(cfg).to(device).eval()
    with pytest.raises(ValueError, match="pinned_memory=True requires gpu_preprocess"):
        Predictor(model, min_size_test=64, max_size_test=128, pinned_memory=True)


# ---------------------------------------------------------------------------
# GPU preprocessing path
# ---------------------------------------------------------------------------


@pytest.mark.cuda
def test_predictor_gpu_preprocess_returns_instances(device: torch.device) -> None:
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = build_faster_rcnn(cfg).to(device).eval()
    predictor = Predictor(
        model, min_size_test=64, max_size_test=128, device=device, gpu_preprocess=True
    )
    image = np.random.default_rng(0).integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    out = predictor(image)
    assert isinstance(out, Instances)
    assert out.image_size == (96, 96)


@pytest.mark.cuda
def test_predictor_gpu_preprocess_with_pinned_memory(device: torch.device) -> None:
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = build_faster_rcnn(cfg).to(device).eval()
    predictor = Predictor(
        model,
        min_size_test=64,
        max_size_test=128,
        device=device,
        gpu_preprocess=True,
        pinned_memory=True,
    )
    rng = np.random.default_rng(1)
    # Two different sizes to exercise the buffer-grow path.
    out_small = predictor(rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8))
    out_large = predictor(rng.integers(0, 256, size=(120, 96, 3), dtype=np.uint8))
    assert out_small.image_size == (80, 80)
    assert out_large.image_size == (120, 96)
    assert predictor._pinned_buf is not None
    assert predictor._pinned_buf.shape[0] >= 120
    assert predictor._pinned_buf.shape[1] >= 96
    assert predictor._pinned_buf.is_pinned()


def test_predictor_coerces_float_input_to_uint8(device: torch.device) -> None:
    predictor = _tiny_predictor(device)
    img_float = np.random.default_rng(2).random((96, 96, 3), dtype=np.float32) * 255.0
    out = predictor(img_float)
    assert out.image_size == (96, 96)


# ---------------------------------------------------------------------------
# Batch path
# ---------------------------------------------------------------------------


def test_predictor_batch_returns_one_instances_per_image(device: torch.device) -> None:
    predictor = _tiny_predictor(device)
    rng = np.random.default_rng(3)
    images = [
        rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8),
        rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8),
        rng.integers(0, 256, size=(96, 64, 3), dtype=np.uint8),
    ]
    outs = predictor.batch(images)
    assert len(outs) == 3
    expected_sizes = [(64, 96), (80, 80), (96, 64)]
    for inst, size in zip(outs, expected_sizes, strict=True):
        assert inst.image_size == size


# ---------------------------------------------------------------------------
# Eval-mode invariant
# ---------------------------------------------------------------------------


def test_predictor_init_puts_model_in_eval_mode(device: torch.device) -> None:
    cfg = _tiny_cfg()
    model = build_faster_rcnn(cfg).to(device).train()  # arm to training
    Predictor(model, min_size_test=64, max_size_test=128, device=device)
    # Constructor must flip to eval (so subsequent forwards return
    # detections, not loss dicts).
    assert model.training is False


# ---------------------------------------------------------------------------
# from_pretrained — high-level constructor
# ---------------------------------------------------------------------------


def test_from_pretrained_builds_from_sidecar(monkeypatch: pytest.MonkeyPatch) -> None:
    """``from_pretrained`` resolves a name to its self-describing checkpoint and
    returns a usable Predictor — the architecture comes from the sidecar, with
    no bundled-YAML config and no hand-wiring of build_detector / torch.load.
    """
    from mayaku.cli import _factory

    cfg = _tiny_cfg()
    fake_model = build_faster_rcnn(cfg)
    calls: dict[str, object] = {}

    def fake_load_detector(
        weights: str | Path,
    ) -> tuple[MayakuConfig, torch.nn.Module, list[str] | None]:
        calls["weights"] = weights
        return cfg, fake_model, ["a", "b"]

    monkeypatch.setattr(_factory, "load_detector", fake_load_detector)

    p = from_pretrained("faster_rcnn_R_50_FPN_3x", device="cpu")

    assert isinstance(p, Predictor)
    # The source is the checkpoint; its sidecar defines the arch.
    assert calls["weights"] == "faster_rcnn_R_50_FPN_3x"
    assert p.min_size_test == cfg.input.min_size_test
    assert p.max_size_test == cfg.input.max_size_test
    assert p.model.training is False  # eval-mode invariant


def test_from_pretrained_routes_artifact_suffix_to_artifact_loader(tmp_path: Path) -> None:
    """A pre-exported artifact suffix routes to ArtifactPredictor (the 'file is
    the backend' dispatch), not the checkpoint path — a missing file surfaces the
    artifact loader's FileNotFoundError, not a checkpoint/download error."""
    with pytest.raises(FileNotFoundError, match="artifact not found"):
        from_pretrained(str(tmp_path / "missing.onnx"))


# ---------------------------------------------------------------------------
# Predictor.export
# ---------------------------------------------------------------------------


def test_export_onnx_writes_file_and_returns_path(tmp_path: Path) -> None:
    """Predictor.export("onnx") writes the artifact and returns its path —
    the in-memory mirror of `mayaku export onnx`."""
    predictor = _tiny_predictor(torch.device("cpu"))
    out = tmp_path / "m.onnx"
    result = predictor.export("onnx", output=out, sample_height=64, sample_width=64)
    assert result == out
    assert out.is_file() and out.stat().st_size > 0


def test_export_default_filename_uses_source_stem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no `output`, the filename is `<source_stem>.<ext>` in the cwd —
    so `from_pretrained("mayaku-s").export("onnx")` writes `mayaku-s.onnx`."""
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).eval()
    predictor = Predictor(model, min_size_test=64, max_size_test=128, source_stem="mayaku-s")
    monkeypatch.chdir(tmp_path)
    result = predictor.export("onnx", sample_height=64, sample_width=64)
    assert result == Path("mayaku-s.onnx")
    assert (tmp_path / "mayaku-s.onnx").is_file()


def test_export_unknown_format_raises() -> None:
    predictor = _tiny_predictor(torch.device("cpu"))
    with pytest.raises(ValueError, match="unknown export format"):
        predictor.export("tflite")
