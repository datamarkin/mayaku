"""The exported graph's input size must be the model's deploy canvas.

A full-detector graph bakes ``(H, W)`` into its box decode (``export_forward``
reads ``image.shape`` as Python ints), so the tracing sample size is the size the
artifact runs at *forever* — ``ArtifactPredictor`` can't override it from the
sidecar. Export therefore defaults the sample to the model's deploy canvas, and
both ends refuse a disagreement rather than degrade quietly.

The regression these lock down: a model trained at, say, a 384x896 canvas used to
export at a hardcoded 640x640 and then deploy at 640x640 — wrong resolution *and*
wrong aspect — while its own embedded sidecar said 384x896. Non-square canvases
are used throughout on purpose; a square-only suite would not have caught it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mayaku.config.schemas import MayakuConfig
from mayaku.inference import from_pretrained
from mayaku.inference.export.dispatch import (
    build_sample,
    export_detector,
    resolve_export_sample_hw,
)
from mayaku.inference.export.metadata import read_sidecar
from mayaku.models.detectors import build_faster_rcnn
from mayaku.models.detectors.uniquery import build_uniquery
from mayaku.tuning.sizing import resolve_deploy_canvas
from mayaku.utils.checkpoint import build_sidecar

from ._checkpoint import save_self_describing

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

# Small, 128-aligned, and deliberately NOT square: (H=128, W=256). Both dims are
# multiples of 32 for the ResNet-50 FPN and small enough to keep the toy fast.
RECT_CANVAS: tuple[int, int] = (128, 256)


def _uniquery_cfg(canvas_hw: tuple[int, int] | None = RECT_CANVAS) -> MayakuConfig:
    """A traceable full-detector config pinned to ``canvas_hw`` (letterbox deploy)."""
    return MayakuConfig(
        model={
            "meta_architecture": "uniquery",
            "backbone": {"name": "resnet50"},
            "uniquery_head": {"num_proposals": 10, "num_stages": 2},
            "roi_heads": {"num_classes": 3},
        },
        input={"resize_mode": "letterbox", "size_budget": 256, "canvas_hw": canvas_hw},
    )


def _rcnn_cfg() -> MayakuConfig:
    """A backbone-only (non-full-detector) config on the legacy shortest_edge path."""
    return MayakuConfig(
        model={
            "meta_architecture": "faster_rcnn",
            "backbone": {"name": "resnet50", "freeze_at": 2, "norm": "FrozenBN"},
            "roi_heads": {"num_classes": 3},
        }
    )


def _onnx_input_hw(path: Path) -> tuple[int, int]:
    """Read the ``(H, W)`` the ONNX graph declares on its ``image`` input.

    Deliberately independent of ``artifact._static_hw`` — the production reader is
    one of the things under test, so the test must not agree with it by sharing it.
    """
    model = onnx.load(str(path))
    dims = model.graph.input[0].type.tensor_type.shape.dim
    return int(dims[2].dim_value), int(dims[3].dim_value)


# Module-scoped: the ResNet-50 toy costs ~0.5s to build and ~1.8s to export, and
# nothing below mutates either artifact — each test reads or re-exports from them.


@pytest.fixture(scope="module")
def uq_model() -> nn.Module:
    torch.manual_seed(0)
    return build_uniquery(_uniquery_cfg()).eval()


@pytest.fixture(scope="module")
def rcnn_model() -> nn.Module:
    torch.manual_seed(0)
    return build_faster_rcnn(_rcnn_cfg()).eval()


@pytest.fixture(scope="module")
def uq_weights(uq_model: nn.Module, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One self-describing UniQuery checkpoint pinned to ``RECT_CANVAS``."""
    path = tmp_path_factory.mktemp("uq") / "model.pth"
    return save_self_describing(path, uq_model, _uniquery_cfg())


@pytest.fixture(scope="module")
def uq_onnx(uq_weights: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The artifact ``mayaku export`` produces from ``uq_weights`` with no size flags."""
    from mayaku.cli.export import run_export

    out = tmp_path_factory.mktemp("uq_onnx") / "uq.onnx"
    run_export("onnx", uq_weights, output=out)
    return out


# ---------------------------------------------------------------------------
# resolve_export_sample_hw — the shared defaulting rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sample_hw", "expected"),
    [
        # No explicit size -> the deploy canvas, in (H, W) order, not a square.
        ((None, None), RECT_CANVAS),
        # An explicit size that matches is fine.
        (RECT_CANVAS, RECT_CANVAS),
        # Any other size is refused: it would be baked into the box decode.
        ((640, 640), None),
        # (W, H) instead of (H, W) is the easy mistake to make by hand, and on a
        # non-square canvas it is exactly as wrong as any other size.
        ((RECT_CANVAS[1], RECT_CANVAS[0]), None),
        # One flag given, one omitted: the omitted dim comes from the canvas, and
        # the resulting mismatch is still caught rather than half-applied.
        ((512, None), None),
    ],
    ids=["default", "matching", "mismatched", "transposed", "partial"],
)
def test_full_detector_sample_size(
    uq_model: nn.Module,
    sample_hw: tuple[int | None, int | None],
    expected: tuple[int, int] | None,
) -> None:
    if expected is None:
        with pytest.raises(ValueError, match="disagrees with this model's deploy canvas"):
            resolve_export_sample_hw(RECT_CANVAS, uq_model, *sample_hw)
    else:
        assert resolve_export_sample_hw(RECT_CANVAS, uq_model, *sample_hw) == expected


def test_backbone_only_graph_allows_any_sample_size(rcnn_model: nn.Module) -> None:
    """Backbone graphs export with dynamic spatial axes, so their sample size is
    a tracing detail — the documented 800x1333 examples must keep working."""
    assert resolve_export_sample_hw((640, 640), rcnn_model, 800, 1333) == (800, 1333)


def test_default_canvas_is_640_square_for_a_default_config() -> None:
    """The strict-generalisation guarantee: an unpinned default config resolves to
    the literal 640x640 the old hardcoded default used, so legacy shortest_edge
    exports are unaffected by this change."""
    cfg = _rcnn_cfg()
    assert cfg.input.canvas_hw is None
    assert resolve_deploy_canvas(cfg.input.canvas_hw, cfg.input.size_budget) == (640, 640)


# ---------------------------------------------------------------------------
# run_export — end to end through the CLI entry point
# ---------------------------------------------------------------------------


def test_run_export_traces_at_the_checkpoint_canvas(uq_onnx: Path) -> None:
    """The headline fix: export with no size flags produces a graph whose input
    is the trained rectangular canvas, matching the sidecar it embeds."""
    assert _onnx_input_hw(uq_onnx) == RECT_CANVAS
    embedded = read_sidecar(uq_onnx, "onnx")
    assert embedded is not None
    assert tuple(embedded["config"]["input"]["canvas_hw"]) == RECT_CANVAS


def test_run_export_rejects_mismatched_flags(uq_weights: Path, tmp_path: Path) -> None:
    from mayaku.cli.export import run_export

    with pytest.raises(ValueError, match="disagrees with this model's deploy canvas"):
        run_export(
            "onnx",
            uq_weights,
            output=tmp_path / "uq.onnx",
            sample_height=640,
            sample_width=640,
        )


def test_run_export_backbone_only_unchanged(rcnn_model: nn.Module, tmp_path: Path) -> None:
    """A shortest_edge R-CNN checkpoint with no explicit size still exports at
    640x640 — the no-regression guard for every pre-existing model."""
    from mayaku.cli.export import run_export

    weights = save_self_describing(tmp_path / "rcnn.pth", rcnn_model, _rcnn_cfg())
    out = tmp_path / "rcnn.onnx"
    run_export("onnx", weights, output=out, onnx_dynamic_input_shape=False)
    assert _onnx_input_hw(out) == (640, 640)


# ---------------------------------------------------------------------------
# Predictor.export — the in-memory mirror
# ---------------------------------------------------------------------------


def test_predictor_export_traces_at_the_deploy_canvas(uq_weights: Path, tmp_path: Path) -> None:
    predictor = from_pretrained(uq_weights, device="cpu")
    out = predictor.export("onnx", output=tmp_path / "from_predictor.onnx")
    assert _onnx_input_hw(out) == RECT_CANVAS


def test_predictor_export_rejects_mismatched_size(uq_weights: Path, tmp_path: Path) -> None:
    predictor = from_pretrained(uq_weights, device="cpu")
    with pytest.raises(ValueError, match="disagrees"):
        predictor.export("onnx", output=tmp_path / "bad.onnx", sample_height=640, sample_width=640)


def test_predictor_export_uses_a_directly_set_canvas(uq_model: nn.Module, tmp_path: Path) -> None:
    """A ``Predictor`` built by hand (no config) exports at the canvas it was
    given, not a square — ``canvas_hw`` is the one source for both paths."""
    from mayaku.inference.predictor import Predictor

    predictor = Predictor(uq_model, resize_mode="letterbox", canvas=RECT_CANVAS)
    out = predictor.export("onnx", output=tmp_path / "handbuilt.onnx")
    assert _onnx_input_hw(out) == RECT_CANVAS


# ---------------------------------------------------------------------------
# ArtifactPredictor — the load-time backstop for artifacts already exported wrong
# ---------------------------------------------------------------------------


def _export_at(tmp_path: Path, canvas_hw: tuple[int, int] | None, sample: tuple[int, int]) -> Path:
    """Write an artifact tracing ``sample`` while its sidecar declares ``canvas_hw``.

    Goes through ``export_detector``, the low-level seam that deliberately does no
    canvas checking — the only way to still produce what the old exporter produced,
    and the reason the load-time guard is worth having.
    """
    torch.manual_seed(0)
    cfg = _uniquery_cfg(canvas_hw)
    out = tmp_path / "stale.onnx"
    export_detector(
        build_uniquery(cfg).eval(),
        "onnx",
        out,
        sample=build_sample(*sample),
        sidecar=build_sidecar(cfg, ["a", "b", "c"]),
    )
    return out


def test_artifact_rejects_graph_that_contradicts_its_sidecar(tmp_path: Path) -> None:
    out = _export_at(tmp_path, RECT_CANVAS, (128, 128))
    with pytest.raises(ValueError, match=r"exported at 128x128 .* deploys at 128x256"):
        from_pretrained(out, device="cpu")


def test_artifact_with_unpinned_canvas_runs_at_its_graph_size(tmp_path: Path) -> None:
    """``canvas_hw`` is pinned only by a letterbox training run. With none, the
    model asserts no geometry — the ``size_budget`` fallback is not a claim data
    ever validated — so the graph's own size stands and the guard stays quiet.
    Rejecting here would break exporting at a deliberate size via
    ``export_detector``, which is how the small-canvas artifact tests work."""
    out = _export_at(tmp_path, None, (128, 128))
    assert from_pretrained(out, device="cpu")._canvas == (128, 128)


def test_artifact_feeds_its_graph_the_same_geometry_as_the_eager_model(
    uq_weights: Path, uq_onnx: Path
) -> None:
    """The property the whole fix exists for: a ``.pth`` and the artifact exported
    from it letterbox to the *same* canvas, so they see the same object scale and
    the same padding. Asserted on the tensor actually handed to the runtime rather
    than on detections — a random-init toy produces no stable boxes to compare, and
    a threshold that admits some would compare noise."""
    eager = from_pretrained(uq_weights, device="cpu")
    artifact = from_pretrained(uq_onnx, device="cpu")
    assert eager.canvas_hw == RECT_CANVAS
    assert artifact._canvas == RECT_CANVAS

    seen: list[tuple[int, ...]] = []
    inner = artifact._session.run

    def _spy(x: np.ndarray) -> dict[str, np.ndarray]:
        seen.append(x.shape)
        return inner(x)

    artifact._session.run = _spy  # type: ignore[method-assign]

    # A 90x300 image shares neither the canvas aspect nor its size, so the
    # letterbox has real work to do in both dimensions.
    image = np.random.default_rng(0).integers(0, 255, size=(90, 300, 3), dtype=np.uint8)
    inst = artifact(image)

    assert seen == [(1, 3, *RECT_CANVAS)]
    # And the detections come back in ORIGINAL image coordinates, not canvas ones.
    assert inst.image_size == (90, 300)
    if len(inst):
        boxes = inst.pred_boxes.tensor
        assert (boxes[:, 0::2] >= -1).all() and (boxes[:, 0::2] <= 301).all()
        assert (boxes[:, 1::2] >= -1).all() and (boxes[:, 1::2] <= 91).all()
