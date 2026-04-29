"""Tests for :mod:`mayaku.inference.export.coreml` and the CLI export path.

CoreML conversion runs anywhere ``coremltools`` is installed; the
``predict`` API only works on macOS, so the parity check tolerates a
non-macOS skip. Tests are gated on ``coreml`` (the existing pytest
marker registered in ``conftest.py``) so hosts without the
``[coreml]`` extra installed get a clear skip rather than a
collection error.
"""

from __future__ import annotations

import platform
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
from mayaku.inference.export import CoreMLExporter
from mayaku.inference.export.base import ExportResult, ParityResult
from mayaku.models.detectors import build_faster_rcnn

pytestmark = pytest.mark.coreml  # auto-skips when coremltools isn't installed


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


def _exists(path: Path) -> bool:
    """``mlpackage`` is a directory bundle; treat either as 'exists'."""
    return path.exists() and (path.is_file() or path.is_dir())


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_coreml_export_writes_mlpackage(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).eval()
    out = tmp_path / "model.mlpackage"
    sample = torch.randn(1, 3, 96, 96)
    result = CoreMLExporter().export(model, sample, out)

    assert isinstance(result, ExportResult)
    assert _exists(out)
    assert result.target == "coreml"
    assert result.input_names == ("image",)
    assert result.output_names == ("p2", "p3", "p4", "p5", "p6")
    # extras carries the input shape + compute units the artefact was
    # built with — useful for downstream provenance.
    assert result.extras["compute_units"] == "CPU_ONLY"
    assert result.extras["input_shape"] == "1x3x96x96"


def test_coreml_export_rejects_multi_batch_sample(tmp_path: Path) -> None:
    model = build_faster_rcnn(_tiny_cfg()).eval()
    sample = torch.randn(2, 3, 96, 96)
    with pytest.raises(ValueError, match="single-batch"):
        CoreMLExporter().export(model, sample, tmp_path / "model.mlpackage")


def test_coreml_compute_units_unknown_rejected() -> None:
    with pytest.raises(ValueError, match="unknown CoreML compute_units"):
        CoreMLExporter(compute_units="QUANTUM_TPU")


def test_coreml_compute_precision_unknown_rejected() -> None:
    with pytest.raises(ValueError, match="unknown CoreML compute_precision"):
        CoreMLExporter(compute_precision="bf16")  # type: ignore[arg-type]


def test_coreml_export_fp16_records_precision_in_extras(tmp_path: Path) -> None:
    """fp16 export must produce a loadable artefact and record the
    precision in ``ExportResult.extras`` so deployment scripts can
    verify what was built. fp16 is what's needed for Apple Silicon
    Neural Engine execution at runtime."""
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).eval()
    out = tmp_path / "model_fp16.mlpackage"
    sample = torch.randn(1, 3, 96, 96)
    result = CoreMLExporter(compute_precision="fp16").export(model, sample, out)
    assert result.extras["compute_precision"] == "fp16"
    assert _exists(out)


def test_coreml_export_fp32_extras_default(tmp_path: Path) -> None:
    """The default fp32 path must still record itself in extras for
    symmetry with the fp16 case."""
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).eval()
    out = tmp_path / "model_fp32.mlpackage"
    sample = torch.randn(1, 3, 96, 96)
    result = CoreMLExporter().export(model, sample, out)
    assert result.extras["compute_precision"] == "fp32"


# ---------------------------------------------------------------------------
# Parity (loose tolerance because CoreML's compute path quantises differently)
# ---------------------------------------------------------------------------


def test_coreml_parity_skipped_on_non_macos_returns_passing_result(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    model = build_faster_rcnn(_tiny_cfg()).eval()
    out = tmp_path / "model.mlpackage"
    exporter = CoreMLExporter()
    exporter.export(model, torch.randn(1, 3, 96, 96), out)

    parity = exporter.parity_check(model, out, torch.randn(1, 3, 96, 96))
    assert isinstance(parity, ParityResult)
    assert parity.target == "coreml"
    if platform.system() == "Darwin":
        # On macOS the actual numerical comparison runs and we need
        # the result to be inside the configured tolerance.
        assert parity.passed, (
            f"CoreML parity failed on macOS: max_abs={parity.max_abs_error}, "
            f"per_output={parity.per_output}"
        )
    else:
        # Non-macOS: the runtime is unavailable, parity_check returns a
        # passing result with zero errors and a clear extras note.
        assert parity.passed
        assert parity.max_abs_error == 0.0


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CoreMLBackbone runtime adapter
# ---------------------------------------------------------------------------


def test_coreml_backbone_rejects_non_macos() -> None:
    """The runtime adapter requires Core ML.framework which only exists
    on macOS. Construction must fail loudly elsewhere instead of
    deferring the error until forward()."""
    if platform.system() == "Darwin":
        pytest.skip("non-macOS gate; covered by parity test on macOS")
    from mayaku.inference.export import CoreMLBackbone

    with pytest.raises(RuntimeError, match="macOS"):
        CoreMLBackbone(Path("/nonexistent.mlpackage"))


@pytest.mark.skipif(platform.system() != "Darwin", reason="Core ML.framework is macOS-only")
def test_coreml_backbone_shape_contract(tmp_path: Path) -> None:
    """Probing at a shape smaller than the export shape exercises
    CoreMLBackbone's pad-then-crop path. The output dict must carry the
    same keys and shapes the eager backbone would produce on the
    same probe — values need not match exactly because conv chains
    amplify the boundary-zero-pad difference under random-init weights.
    """
    from mayaku.inference.export import CoreMLBackbone

    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = build_faster_rcnn(cfg).eval()
    out = tmp_path / "model.mlpackage"
    sample = torch.randn(1, 3, 96, 96)
    CoreMLExporter().export(model, sample, out)

    cmb = CoreMLBackbone(out, input_height=96, input_width=96)
    assert cmb.size_divisibility == 32

    probe = torch.randn(1, 3, 64, 64)
    coreml_out = cmb(probe)
    eager_out = model.backbone(probe)

    assert set(coreml_out.keys()) == set(eager_out.keys()) == {"p2", "p3", "p4", "p5", "p6"}
    for name, eager_t in eager_out.items():
        assert coreml_out[name].shape == eager_t.shape, (
            f"{name}: CoreML shape {tuple(coreml_out[name].shape)} != eager {tuple(eager_t.shape)}"
        )


@pytest.mark.skipif(platform.system() != "Darwin", reason="Core ML.framework is macOS-only")
def test_coreml_backbone_values_match_eager_at_export_shape(tmp_path: Path) -> None:
    """At the export shape (no pad/crop boundary effect), CoreMLBackbone
    forward must match the eager backbone within the same tolerance band
    as :meth:`CoreMLExporter.parity_check`. This validates the runtime
    adapter doesn't introduce error beyond what the export already has.
    """
    from mayaku.inference.export import CoreMLBackbone

    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = build_faster_rcnn(cfg).eval()
    out = tmp_path / "model.mlpackage"
    sample = torch.randn(1, 3, 96, 96)
    CoreMLExporter().export(model, sample, out)

    cmb = CoreMLBackbone(out, input_height=96, input_width=96)
    coreml_out = cmb(sample)
    with torch.no_grad():
        eager_out = model.backbone(sample)

    for name, eager_t in eager_out.items():
        max_abs = (coreml_out[name] - eager_t).abs().max().item()
        assert max_abs < 1e-1, f"{name}: max abs diff {max_abs}"


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def test_cli_run_export_coreml(tmp_path: Path) -> None:
    from mayaku.cli.export import run_export
    from mayaku.config import dump_yaml

    cfg = _tiny_cfg()
    cfg_path = tmp_path / "cfg.yaml"
    dump_yaml(cfg, cfg_path)

    model = build_faster_rcnn(cfg).eval()
    weights = tmp_path / "model.pth"
    torch.save(model.state_dict(), weights)

    out = tmp_path / "exported.mlpackage"
    result = run_export(
        "coreml",
        cfg_path,
        weights=weights,
        output=out,
        sample_height=96,
        sample_width=96,
    )
    assert isinstance(result, ExportResult)
    assert result.target == "coreml"
    assert _exists(out)
