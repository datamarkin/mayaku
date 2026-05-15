"""Tests for :mod:`mayaku.inference.export.onnx` and the CLI export path.

Build a tiny FasterRCNN, export the backbone+FPN body to ONNX, and run
the parity check against eager via the bundled onnxruntime CPU EP.
The whole suite is gated on the ``onnxruntime`` import so users
without the ``[onnx]`` extra installed see a clear skip rather than a
collection error.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from mayaku.config.schemas import (
    BackboneConfig,
    MayakuConfig,
    ModelConfig,
    ROIBoxHeadConfig,
    ROIHeadsConfig,
    RPNConfig,
)
from mayaku.inference.export import ONNXExporter
from mayaku.inference.export.base import ExportResult, ParityResult
from mayaku.models.detectors import build_faster_rcnn

_HAVE_ONNX = (
    importlib.util.find_spec("onnxruntime") is not None
    and importlib.util.find_spec("onnx") is not None
)
pytestmark = pytest.mark.skipif(
    not _HAVE_ONNX,
    reason="install with `pip install -e '.[onnx]'` to run ONNX export tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_cfg(backbone_name: str = "resnet50") -> MayakuConfig:
    """Build a minimal Faster R-CNN config wired around ``backbone_name``.

    The schema validator added with the ConvNeXt integration rejects
    ResNet-specific fields (``norm``, ``stride_in_1x1``) on ConvNeXt
    variants — branch on the family so both naming groups produce a
    valid config.
    """
    from mayaku.models.backbones import is_convnext_variant

    if is_convnext_variant(backbone_name):
        backbone = BackboneConfig(name=backbone_name, freeze_at=0)  # type: ignore[arg-type]
    else:
        backbone = BackboneConfig(name=backbone_name, freeze_at=2, norm="FrozenBN")  # type: ignore[arg-type]
    return MayakuConfig(
        model=ModelConfig(
            meta_architecture="faster_rcnn",
            backbone=backbone,
            rpn=RPNConfig(
                pre_nms_topk_train=100,
                pre_nms_topk_test=50,
                post_nms_topk_train=20,
                post_nms_topk_test=10,
                batch_size_per_image=16,
            ),
            roi_heads=ROIHeadsConfig(num_classes=2, batch_size_per_image=8),
            roi_box_head=ROIBoxHeadConfig(num_fc=1, fc_dim=32),
        ),
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_onnx_export_writes_a_file_and_records_metadata(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).eval()
    out = tmp_path / "model.onnx"
    sample = torch.randn(1, 3, 96, 96)
    result = ONNXExporter().export(model, sample, out)

    assert isinstance(result, ExportResult)
    assert out.exists()
    assert out.stat().st_size > 0
    assert result.target == "onnx"
    assert result.opset == 17
    assert result.input_names == ("image",)
    assert result.output_names == ("p2", "p3", "p4", "p5", "p6")


def test_onnx_export_supports_dynamic_batch_and_spatial_axes(tmp_path: Path) -> None:
    """Confirm the exported graph accepts inputs of a different size
    than the tracing sample. The dynamic-axes wiring is the whole
    reason we use ``torch.onnx.export`` with ``dynamic_axes=...``."""
    import onnxruntime as ort

    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).eval()
    out = tmp_path / "model.onnx"
    ONNXExporter().export(model, torch.randn(1, 3, 64, 64), out)

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    larger = torch.randn(2, 3, 128, 128).numpy()
    res = sess.run(["p2", "p3", "p4", "p5", "p6"], {"image": larger})
    # res[0] is p2 (stride 4) → (2, 256, 32, 32)
    assert res[0].shape == (2, 256, 32, 32)
    assert res[-1].shape == (2, 256, 2, 2)  # p6 stride 64


def test_onnx_export_dynamic_input_shape_false_pins_input_dims(tmp_path: Path) -> None:
    """With ``dynamic_input_shape=False`` the exported graph pins the
    sample's literal shape; running it at any other shape is rejected.
    This is the path used when targeting TensorRT (see ADR 005)."""
    import onnxruntime as ort

    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).eval()
    out = tmp_path / "model.onnx"
    sample = torch.randn(1, 3, 64, 64)
    ONNXExporter(dynamic_input_shape=False).export(model, sample, out)

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    # Same shape works.
    same_shape = sample.numpy()
    sess.run(["p2", "p3", "p4", "p5", "p6"], {"image": same_shape})
    # Different shape is rejected at session.run() time.
    with pytest.raises(Exception, match=r"(?i)(shape|invalid|got)"):
        wrong = torch.randn(2, 3, 128, 128).numpy()
        sess.run(["p2", "p3", "p4", "p5", "p6"], {"image": wrong})


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def test_onnx_parity_check_within_tolerance(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).eval()
    out = tmp_path / "model.onnx"
    sample = torch.randn(1, 3, 96, 96)
    exporter = ONNXExporter()
    exporter.export(model, sample, out)

    parity = exporter.parity_check(model, out, sample, atol=1e-3, rtol=1e-3)
    assert isinstance(parity, ParityResult)
    assert parity.target == "onnx"
    assert parity.passed, (
        f"ONNX parity failed: max_abs={parity.max_abs_error}, "
        f"max_rel={parity.max_rel_error}, per_output={parity.per_output}"
    )
    assert set(parity.per_output) == {"p2", "p3", "p4", "p5", "p6"}


def test_onnx_parity_records_per_output_errors(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).eval()
    out = tmp_path / "model.onnx"
    sample = torch.randn(1, 3, 96, 96)
    exporter = ONNXExporter()
    exporter.export(model, sample, out)

    parity = exporter.parity_check(model, out, sample)
    # Every per-output entry is a (abs, rel) tuple of finite floats.
    for name, (abs_e, rel_e) in parity.per_output.items():
        assert name in {"p2", "p3", "p4", "p5", "p6"}
        assert abs_e >= 0 and rel_e >= 0


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_run_export_onnx(tmp_path: Path) -> None:
    from mayaku.cli.export import run_export
    from mayaku.config import dump_yaml

    cfg = _tiny_cfg()
    cfg_path = tmp_path / "cfg.yaml"
    dump_yaml(cfg, cfg_path)

    model = build_faster_rcnn(cfg).eval()
    weights = tmp_path / "model.pth"
    torch.save(model.state_dict(), weights)

    out = tmp_path / "exported.onnx"
    result = run_export(
        "onnx",
        cfg_path,
        weights=weights,
        output=out,
        sample_height=96,
        sample_width=96,
    )
    assert isinstance(result, ExportResult)
    assert result.target == "onnx"
    assert out.exists()


# ---------------------------------------------------------------------------
# ONNXBackbone runtime adapter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backbone_name", ["resnet50", "dinov3_convnext_tiny"])
def test_onnx_backbone_shape_contract(tmp_path: Path, backbone_name: str) -> None:
    """Probing at a shape smaller than the export shape exercises the
    pad-then-crop path. Output dict must mirror eager backbone keys
    and shapes; values can drift under boundary effects with random
    init weights, same situation as CoreMLBackbone's analog.

    Parametrised over backbone families so the ConvNeXt op set
    (depthwise Conv2d, LayerNorm at opset 17, GELU, Linear, Permute)
    is exercised end-to-end through ONNX export + onnxruntime CPU
    inference, not just ResNet's BN-family ops.
    """
    from mayaku.inference.export import ONNXBackbone

    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg(backbone_name)).eval()
    out = tmp_path / "model.onnx"
    sample = torch.randn(1, 3, 96, 96)
    ONNXExporter().export(model, sample, out)

    ob = ONNXBackbone(out, input_height=96, input_width=96)
    assert ob.size_divisibility == 32

    probe = torch.randn(1, 3, 64, 64)
    onnx_out = ob(probe)
    eager_out = model.backbone(probe)

    assert set(onnx_out.keys()) == set(eager_out.keys()) == {"p2", "p3", "p4", "p5", "p6"}
    for name, eager_t in eager_out.items():
        assert onnx_out[name].shape == eager_t.shape


@pytest.mark.parametrize("backbone_name", ["resnet50", "dinov3_convnext_tiny"])
def test_onnx_backbone_values_match_eager_at_export_shape(
    tmp_path: Path, backbone_name: str
) -> None:
    """At the export shape (no pad/crop) with the CPU provider,
    ONNXBackbone forward must match eager within tight tolerance.
    The CoreMLExecutionProvider quantises and won't hit this bound;
    pin CPU for the test."""
    from mayaku.inference.export import ONNXBackbone

    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg(backbone_name)).eval()
    out = tmp_path / "model.onnx"
    sample = torch.randn(1, 3, 96, 96)
    ONNXExporter().export(model, sample, out)

    ob = ONNXBackbone(out, input_height=96, input_width=96, providers=("CPUExecutionProvider",))
    onnx_out = ob(sample)
    with torch.no_grad():
        eager_out = model.backbone(sample)

    for name, eager_t in eager_out.items():
        max_abs = (onnx_out[name] - eager_t).abs().max().item()
        assert max_abs < 1e-3, f"{name}: max abs diff {max_abs}"
