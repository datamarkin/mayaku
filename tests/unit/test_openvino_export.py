"""Tests for :mod:`mayaku.inference.export.openvino` and the CLI export path.

OpenVINO conversion + CPU-EP inference both run anywhere
``openvino`` is installed (Linux, macOS, Windows). Tests gate on the
existing ``openvino`` pytest marker so hosts without the
``[openvino]`` extra installed get a clear skip rather than a
collection error.
"""

from __future__ import annotations

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
from mayaku.inference.export import OpenVINOExporter
from mayaku.inference.export.base import ExportResult, ParityResult
from mayaku.models.detectors import build_faster_rcnn

pytestmark = pytest.mark.openvino  # auto-skips when openvino isn't installed


def _tiny_cfg() -> MayakuConfig:
    return MayakuConfig(
        model=ModelConfig(
            meta_architecture="faster_rcnn",
            backbone=BackboneConfig(name="resnet50", freeze_at=2, norm="FrozenBN"),
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


def test_openvino_export_writes_xml_and_bin(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).eval()
    out = tmp_path / "model.xml"
    sample = torch.randn(1, 3, 96, 96)
    result = OpenVINOExporter().export(model, sample, out)

    assert isinstance(result, ExportResult)
    assert result.target == "openvino"
    assert out.exists() and out.stat().st_size > 0
    bin_path = out.with_suffix(".bin")
    assert bin_path.exists() and bin_path.stat().st_size > 0
    assert result.input_names == ("image",)
    assert result.output_names == ("p2", "p3", "p4", "p5", "p6")
    assert result.extras["compress_to_fp16"] == "False"


def test_openvino_export_supports_fp16_compression(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).eval()
    sample = torch.randn(1, 3, 96, 96)

    out_fp32 = tmp_path / "fp32.xml"
    out_fp16 = tmp_path / "fp16.xml"
    OpenVINOExporter(compress_to_fp16=False).export(model, sample, out_fp32)
    OpenVINOExporter(compress_to_fp16=True).export(model, sample, out_fp16)

    bin_fp32 = out_fp32.with_suffix(".bin").stat().st_size
    bin_fp16 = out_fp16.with_suffix(".bin").stat().st_size
    # fp16 IR should be roughly half the size of fp32 (allow some
    # slack for header/alignment overhead).
    assert bin_fp16 < bin_fp32 * 0.7


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def test_openvino_parity_within_tolerance(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).eval()
    out = tmp_path / "model.xml"
    sample = torch.randn(1, 3, 96, 96)
    exporter = OpenVINOExporter()
    exporter.export(model, sample, out)

    parity = exporter.parity_check(model, out, sample, atol=1e-3, rtol=1e-3)
    assert isinstance(parity, ParityResult)
    assert parity.target == "openvino"
    assert parity.passed, (
        f"OpenVINO parity failed: max_abs={parity.max_abs_error}, "
        f"max_rel={parity.max_rel_error}, per_output={parity.per_output}"
    )
    assert set(parity.per_output) == {"p2", "p3", "p4", "p5", "p6"}


def test_openvino_parity_records_per_output_errors(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).eval()
    out = tmp_path / "model.xml"
    sample = torch.randn(1, 3, 96, 96)
    exporter = OpenVINOExporter()
    exporter.export(model, sample, out)

    parity = exporter.parity_check(model, out, sample)
    for name, (abs_e, rel_e) in parity.per_output.items():
        assert name in {"p2", "p3", "p4", "p5", "p6"}
        assert abs_e >= 0 and rel_e >= 0


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def test_cli_run_export_openvino(tmp_path: Path) -> None:
    from mayaku.cli.export import run_export
    from mayaku.config import dump_yaml

    cfg = _tiny_cfg()
    cfg_path = tmp_path / "cfg.yaml"
    dump_yaml(cfg, cfg_path)

    model = build_faster_rcnn(cfg).eval()
    weights = tmp_path / "model.pth"
    torch.save(model.state_dict(), weights)

    out = tmp_path / "exported.xml"
    result = run_export(
        "openvino",
        cfg_path,
        weights=weights,
        output=out,
        sample_height=96,
        sample_width=96,
    )
    assert isinstance(result, ExportResult)
    assert result.target == "openvino"
    assert out.exists()
    assert out.with_suffix(".bin").exists()
