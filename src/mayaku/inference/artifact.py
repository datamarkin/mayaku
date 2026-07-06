"""Run a pre-exported artifact end-to-end — "the file is the backend".

``from_pretrained("model.onnx")`` returns an :class:`ArtifactPredictor` that
behaves like :class:`mayaku.inference.predictor.Predictor`: call it with an image,
get :class:`~mayaku.structures.instances.Instances` in original-image coordinates.

The exported graph is only the math core — a *normalised* ``(1, 3, H, W)`` image
in, ``(boxes, scores, labels)`` in the padded/letterbox frame out (no
normalisation, no resize, no score threshold, no un-letterbox). This module is
the host wrapper that reproduces the pre/post the eager model does internally:

    read image → letterbox to the graph's canvas → normalise (pixel mean/std)
    → run the runtime session → score-threshold → assemble Instances
    → un-letterbox back to original coordinates

Everything the wrapper needs (config + class names) is read from the sidecar the
exporter embedded in the artifact (see :mod:`mayaku.inference.export.metadata`),
so the file loads standalone — no ``.pth``, no config.

Only the ONNX exporter produces a full-detector graph today; CoreML / OpenVINO /
TensorRT export the backbone+FPN body only, so their artifacts are rejected here
with a clear message until full-detector export lands for them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import torch

from mayaku.config.schemas import MayakuConfig
from mayaku.data.transforms import LetterboxTransform
from mayaku.inference.export.full_detector import FULL_DETECTOR_OUTPUTS
from mayaku.inference.export.metadata import read_sidecar, target_from_suffix
from mayaku.inference.postprocess import unletterbox_instances
from mayaku.inference.predictor import ImageInput, _to_uint8_rgb
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances

__all__ = ["ArtifactPredictor"]


class _RuntimeSession(Protocol):
    """Minimal contract a per-format runtime wrapper must satisfy."""

    input_hw: tuple[int, int] | None
    output_names: tuple[str, ...]

    def run(self, x: npt.NDArray[np.float32]) -> dict[str, npt.NDArray[np.float32]]: ...


class ArtifactPredictor:
    """Run a self-describing exported artifact on images, returning ``Instances``.

    Construct via :meth:`from_file` (or :func:`mayaku.from_pretrained` with an
    artifact suffix). The session runs the graph; this class owns the pre/post.
    """

    def __init__(
        self,
        session: _RuntimeSession,
        cfg: MayakuConfig,
        class_names: list[str],
    ) -> None:
        self._session = session
        self.cfg = cfg
        self.class_names = class_names
        # Canvas the graph was exported at: the session's static input shape is
        # authoritative (the graph only accepts that size, and the export sample
        # may differ from the config's deploy canvas); fall back to the config's
        # resolved canvas when the graph is dynamic.
        if session.input_hw is not None:
            self._canvas: tuple[int, int] = session.input_hw
        else:
            from mayaku.tuning.sizing import resolve_deploy_canvas

            self._canvas = resolve_deploy_canvas(cfg.input.canvas_hw, cfg.input.size_budget)
        self._mean = np.asarray(cfg.model.pixel_mean, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.asarray(cfg.model.pixel_std, dtype=np.float32).reshape(3, 1, 1)
        self._score_thresh = float(cfg.model.roi_heads.score_thresh_test)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, source: str | Path, *, device: str = "auto") -> ArtifactPredictor:
        """Build a predictor from a pre-exported artifact file.

        The artifact must carry the mayaku sidecar (embedded at export time) and
        be a full-detector graph. Backbone-only artifacts raise.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(
                f"artifact not found: {path}. Pass a path to an exported "
                ".onnx/.mlpackage/.xml/.engine file."
            )
        target = target_from_suffix(path)
        sidecar = read_sidecar(path, target)
        if sidecar is None or "config" not in sidecar:
            raise ValueError(
                f"{path.name} has no embedded mayaku metadata — re-export it with "
                "this version so the artifact is self-describing."
            )
        cfg = MayakuConfig.model_validate(sidecar["config"])
        class_names = list(sidecar.get("class_names") or [])

        session = _build_session(path, target, device=device)
        missing = set(FULL_DETECTOR_OUTPUTS) - set(session.output_names)
        if missing:
            raise ValueError(
                f"{path.name} is a backbone-only graph (outputs {session.output_names}), "
                "not a full detector, so it can't run end-to-end. Only UniQuery models "
                "export as runnable artifacts today (ONNX)."
            )
        return cls(session, cfg, class_names)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def __call__(self, image: ImageInput) -> Instances:
        """Run inference on a single image; return :class:`Instances` in original coords."""
        arr = _to_uint8_rgb(image)
        h, w = int(arr.shape[0]), int(arr.shape[1])

        # Fixed deploy geometry: letterbox to the graph's canvas, run, un-letterbox.
        transform = LetterboxTransform(h, w, self._canvas)
        x = self._normalize(transform.apply_image(arr))
        out = self._session.run(x)

        boxes = np.asarray(out["boxes"], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(out["scores"], dtype=np.float32).reshape(-1)
        labels = np.asarray(out["labels"]).reshape(-1).astype(np.int64)

        # The graph returns a fixed top-K with no score threshold applied (it's a
        # non-traceable, variable-length op); apply it host-side, matching eager.
        # Boolean-mask indexing already yields fresh C-contiguous arrays.
        keep = scores >= self._score_thresh

        instances = Instances(image_size=self._canvas)
        instances.pred_boxes = Boxes(torch.from_numpy(boxes[keep]))
        instances.scores = torch.from_numpy(scores[keep])
        instances.pred_classes = torch.from_numpy(labels[keep])
        return unletterbox_instances(instances, transform, h, w)

    def _normalize(self, hwc: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """``(H, W, 3)`` uint8 RGB → normalised ``(1, 3, H, W)`` float32 (mean/std)."""
        # One pass to a contiguous float32 (C, H, W): astype-on-a-transposed-view
        # would copy anyway, so fold the contiguity + dtype conversion together.
        chw = np.ascontiguousarray(hwc.transpose(2, 0, 1), dtype=np.float32)
        chw = (chw - self._mean) / self._std
        return chw[None]


# ---------------------------------------------------------------------------
# Per-format runtime sessions
# ---------------------------------------------------------------------------


def _build_session(path: Path, target: str, *, device: str) -> _RuntimeSession:
    if target == "onnx":
        return _ONNXSession(path, device=device)
    if target == "coreml":
        return _CoreMLSession(path)
    if target == "openvino":
        return _OpenVINOSession(path)
    if target == "tensorrt":
        return _TensorRTSession(path)
    raise ValueError(f"unknown artifact target {target!r}")


def _static_hw(shape: object) -> tuple[int, int] | None:
    """Extract a static ``(H, W)`` from a ``[N, C, H, W]`` shape, or ``None`` if dynamic."""
    if not isinstance(shape, (list, tuple)) or len(shape) < 4:
        return None
    h, w = shape[-2], shape[-1]
    if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
        return (h, w)
    return None


class _ONNXSession:
    """onnxruntime wrapper — cross-platform, the portable default."""

    def __init__(self, path: Path, *, device: str = "auto") -> None:
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if device in ("cuda", "auto") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._sess = ort.InferenceSession(str(path), providers=providers)
        inp = self._sess.get_inputs()[0]
        self._input_name = inp.name
        self.input_hw = _static_hw(inp.shape)
        self.output_names = tuple(o.name for o in self._sess.get_outputs())

    def run(self, x: npt.NDArray[np.float32]) -> dict[str, npt.NDArray[np.float32]]:
        outs = self._sess.run(None, {self._input_name: x})
        return dict(zip(self.output_names, outs, strict=True))


class _CoreMLSession:
    """coremltools wrapper — macOS only."""

    def __init__(self, path: Path) -> None:
        import coremltools as ct

        self._model = ct.models.MLModel(str(path))
        spec = self._model.get_spec()
        inp = spec.description.input[0]
        self._input_name = inp.name
        shape = list(inp.type.multiArrayType.shape)
        self.input_hw = _static_hw([1, *shape]) if len(shape) == 3 else _static_hw(shape)
        self.output_names = tuple(o.name for o in spec.description.output)

    def run(self, x: npt.NDArray[np.float32]) -> dict[str, npt.NDArray[np.float32]]:
        out = self._model.predict({self._input_name: x})
        return {k: np.asarray(v) for k, v in out.items()}


class _OpenVINOSession:
    """OpenVINO runtime wrapper — CPU device (portable, reproducible)."""

    def __init__(self, path: Path) -> None:
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(path))
        self._compiled = core.compile_model(model, "CPU")
        self._input = self._compiled.inputs[0]
        ps = self._input.get_partial_shape()
        if len(ps) >= 4 and ps[2].is_static and ps[3].is_static:
            self.input_hw: tuple[int, int] | None = (
                int(ps[2].get_length()),
                int(ps[3].get_length()),
            )
        else:
            self.input_hw = None
        self._outputs = list(self._compiled.outputs)
        self.output_names = tuple(o.get_any_name() for o in self._outputs)

    def run(self, x: npt.NDArray[np.float32]) -> dict[str, npt.NDArray[np.float32]]:
        res = self._compiled({self._input: x})
        return {o.get_any_name(): np.asarray(res[o]) for o in self._outputs}


class _TensorRTSession:
    """TensorRT engine wrapper — CUDA host only.

    Not wired for execution yet: full-detector engine export and a CUDA host are
    both required, neither available where this was authored. Loading raises with
    a precise message rather than shipping unverified GPU code.
    """

    input_hw: tuple[int, int] | None = None
    output_names: tuple[str, ...] = ()

    def __init__(self, path: Path) -> None:
        raise NotImplementedError(
            "Running a .engine artifact end-to-end is not wired yet: it needs a "
            "full-detector TensorRT engine (the exporter currently builds the "
            "backbone+FPN body only) and a CUDA host. Use the .onnx artifact, or "
            "run the engine with TensorRT's native runtime."
        )

    def run(
        self, x: npt.NDArray[np.float32]
    ) -> dict[str, npt.NDArray[np.float32]]:  # pragma: no cover
        raise NotImplementedError
