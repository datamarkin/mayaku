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
from mayaku.inference import Predictor
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
# from_config
# ---------------------------------------------------------------------------


def test_predictor_from_config_picks_up_input_sizes(device: torch.device) -> None:
    cfg = _tiny_cfg()
    model = build_faster_rcnn(cfg).to(device).eval()
    predictor = Predictor.from_config(cfg, model)
    # Default config: min_size_test=800, max_size_test=1333.
    assert predictor.min_size_test == cfg.input.min_size_test
    assert predictor.max_size_test == cfg.input.max_size_test


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
