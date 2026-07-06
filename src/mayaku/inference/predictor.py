"""Single-image and batched inference convenience wrapper.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §4.1 (`DefaultPredictor`) for the
in-scope detectors. Differences from the upstream reference, all
intentional:

* **RGB-native input only** (ADR 002). Input is either a numpy
  ``(H, W, 3)`` ``uint8`` RGB array or a path to an image file —
  ``PIL.Image.open(...).convert("RGB")`` is used for the path case via
  :func:`mayaku.utils.image.read_image`. There is no
  ``input_format`` flag.
* **`detector_postprocess` runs unconditionally** (`spec §2.9`). The
  detector returns predictions in network-input coords; the predictor
  rescales them to original image coords before returning. Callers who
  want the raw network-coord predictions can call the model directly.
* **Batch path** is a thin loop over the single-image path, not a
  separate code path. Multi-image batches add no value here because
  every image is independently resized; the right batched path is
  :func:`mayaku.engine.inference_on_dataset` (Step 15).

Inputs accepted:

* ``np.ndarray`` of shape ``(H, W, 3)``, ``dtype=uint8``, RGB.
* ``str`` or ``pathlib.Path`` pointing at an image readable by Pillow.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from mayaku.config.schemas import MayakuConfig
from mayaku.data.transforms import LetterboxTransform, ResizeShortestEdge
from mayaku.data.transforms.augmentation import compute_resized_hw
from mayaku.inference.postprocess import detector_postprocess, unletterbox_instances
from mayaku.structures.instances import Instances
from mayaku.utils.image import read_image

if TYPE_CHECKING:
    from mayaku.inference.artifact import ArtifactPredictor

__all__ = ["Predictor"]

ImageInput = npt.NDArray[np.uint8] | str | Path


class Predictor:
    """Single-image / batch inference wrapper around a built detector.

    Args:
        model: A built and loaded detector (e.g.
            :class:`mayaku.models.detectors.FasterRCNN`). Will be put
            into eval mode.
        resize_mode: ``"shortest_edge"`` (variable resize) or
            ``"letterbox"`` (fixed-canvas deploy geometry). Defaults to
            ``"shortest_edge"``.
        canvas: The resolved letterbox canvas — a scalar ``S`` (→ ``S×S``)
            or an ``(H, W)`` rectangle. Used only when
            ``resize_mode="letterbox"``.
        min_size_test: Short edge of the resized image (pixels).
            Defaults to ``800`` (`spec §6.1`).
        max_size_test: Maximum long-edge length (pixels). Defaults to
            ``1333``.
        device: Override the device the input tensors are placed on.
            Defaults to the model's first parameter's device, which is
            what you want 99% of the time.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        resize_mode: str = "shortest_edge",
        canvas: int | tuple[int, int] = 640,
        min_size_test: int = 800,
        max_size_test: int = 1333,
        device: torch.device | None = None,
        gpu_preprocess: bool = False,
        pinned_memory: bool = False,
        source_stem: str = "model",
        cfg: MayakuConfig | None = None,
        class_names: Sequence[str] | None = None,
    ) -> None:
        if resize_mode not in ("shortest_edge", "letterbox"):
            raise ValueError(
                f"resize_mode must be 'shortest_edge' or 'letterbox'; got {resize_mode!r}"
            )
        # ``canvas`` is the resolved letterbox canvas: a scalar S (→ S×S) or
        # an (H, W) rectangle. It's passed straight to LetterboxTransform.
        if resize_mode == "letterbox":
            dims = (canvas, canvas) if isinstance(canvas, int) else canvas
            if dims[0] <= 0 or dims[1] <= 0:
                raise ValueError(f"canvas must be > 0; got {canvas}")
        if resize_mode == "shortest_edge" and (min_size_test <= 0 or max_size_test <= 0):
            raise ValueError(
                f"min_size_test / max_size_test must be > 0; got ({min_size_test}, {max_size_test})"
            )
        if gpu_preprocess and resize_mode == "letterbox":
            raise ValueError("gpu_preprocess is only supported with resize_mode='shortest_edge'")
        if pinned_memory and not gpu_preprocess:
            raise ValueError(
                "pinned_memory=True requires gpu_preprocess=True; the staging "
                "buffer is shaped for the uint8 RGB upload path."
            )
        self.model = model.eval()
        self.resize_mode = resize_mode
        self.canvas = canvas
        self.min_size_test = min_size_test
        self.max_size_test = max_size_test
        self.device = device or _resolve_device(model)
        self.gpu_preprocess = gpu_preprocess
        self.pinned_memory = pinned_memory
        # Used to derive a default export filename (``<stem>.<ext>``); set by
        # from_pretrained from the source name.
        self._source_stem = source_stem
        # Carried so ``export`` can embed the self-describing sidecar (config +
        # class names) into the artifact. Absent for directly-constructed
        # predictors (e.g. tests) — export then just skips the metadata.
        self._cfg = cfg
        self._class_names = list(class_names) if class_names is not None else None
        self._resize = (
            ResizeShortestEdge(
                short_edge_lengths=(min_size_test,),
                max_size=max_size_test,
                sample_style="choice",
            )
            if resize_mode == "shortest_edge"
            else None
        )
        self._pinned_buf: torch.Tensor | None = None

    @classmethod
    def _from_cfg(
        cls,
        cfg: MayakuConfig,
        model: nn.Module,
        *,
        gpu_preprocess: bool = False,
        pinned_memory: bool = False,
        source_stem: str = "model",
        class_names: Sequence[str] | None = None,
    ) -> Predictor:
        """Internal seam: map ``cfg.input`` deploy geometry onto the constructor.

        The single place that translates a config to a ``Predictor`` — shared by
        :func:`from_pretrained` (the public entry) and the unit-test helpers that
        wrap an in-code/fake model. **Not** a public deploy API; use
        :func:`from_pretrained`.
        """
        from mayaku.tuning.sizing import resolve_deploy_canvas

        inp = cfg.input
        # Resolved letterbox canvas (shared with build_resize_augmentation).
        canvas: int | tuple[int, int] = resolve_deploy_canvas(inp.canvas_hw, inp.size_budget)
        return cls(
            model,
            resize_mode=inp.resize_mode,
            canvas=canvas,
            min_size_test=inp.min_size_test,
            max_size_test=inp.max_size_test,
            gpu_preprocess=gpu_preprocess,
            pinned_memory=pinned_memory,
            source_stem=source_stem,
            cfg=cfg,
            class_names=class_names,
        )

    @property
    def class_names(self) -> list[str] | None:
        """Ordered class names from the checkpoint sidecar (``None`` if unrecorded).

        Mirrors :attr:`ArtifactPredictor.class_names` so any ``from_pretrained``
        result exposes the same attribute — used by evaluation to align
        predictions to the ground-truth categories by name.
        """
        return self._class_names

    # ------------------------------------------------------------------
    # Public call
    # ------------------------------------------------------------------

    def __call__(self, image: ImageInput) -> Instances:
        """Run inference on a single image; return :class:`Instances`.

        The returned instances live in *original image* coordinate
        space — boxes, masks, and keypoints are rescaled by
        :func:`detector_postprocess` so callers don't have to think
        about the resize.
        """
        arr = _to_uint8_rgb(image)
        h, w = int(arr.shape[0]), int(arr.shape[1])

        if self.resize_mode == "letterbox":
            # Fixed-size deploy geometry: letterbox to the resolved canvas (square
            # S or an (H, W) rectangle), run on it, un-letterbox host-side.
            # Passing height/width = the canvas dims makes the model emit boxes in
            # canvas space (identity rescale) so the transform's inverse is exact.
            transform = LetterboxTransform(h, w, self.canvas)
            img_tensor = self._to_tensor(transform.apply_image(arr))
            instances = self._forward(
                [{"image": img_tensor, "height": transform.out_h, "width": transform.out_w}]
            )
            return unletterbox_instances(instances, transform, h, w)

        if self.gpu_preprocess:
            img_tensor = self._gpu_preprocess(arr, h, w)
        else:
            assert self._resize is not None
            img_tensor = self._to_tensor(self._resize.get_transform(arr).apply_image(arr))
        instances = self._forward([{"image": img_tensor, "height": h, "width": w}])
        # Masks/keypoints are box-relative and must be pasted to image res even
        # when boxes need no rescale (UniQuery emits boxes in image coords, so
        # the size check alone would skip the paste). Mirrors COCOEvaluator.
        needs_paste = instances.has("pred_masks") or instances.has("pred_keypoints")
        if instances.image_size != (h, w) or needs_paste:
            instances = detector_postprocess(instances, h, w)
        return instances

    def _to_tensor(self, hwc: npt.NDArray[np.uint8]) -> torch.Tensor:
        """``(H, W, 3)`` uint8 RGB → ``(3, H, W)`` float32 on the model device."""
        chw = np.ascontiguousarray(hwc.transpose(2, 0, 1))
        return torch.from_numpy(chw).to(dtype=torch.float32, device=self.device)

    def _forward(self, inputs: list[dict[str, object]]) -> Instances:
        with torch.no_grad():
            outputs = self.model(inputs)
        if not isinstance(outputs, list) or not outputs:
            raise RuntimeError(
                "Detector did not return list[dict] outputs — model is "
                "probably still in training mode (forward returns the loss "
                "dict). Wrap the call in `model.eval()` before constructing "
                "the Predictor."
            )
        instances: Instances = outputs[0]["instances"]
        return instances

    def _gpu_preprocess(self, arr: npt.NDArray[np.uint8], h: int, w: int) -> torch.Tensor:
        """GPU-side equivalent of the CPU resize path.

        Uploads the uint8 RGB array (optionally via a pinned-memory
        staging buffer for an async H2D copy), then resizes on-device
        with bilinear ``F.interpolate``. Returns a ``(3, new_h, new_w)``
        float32 tensor on ``self.device``.

        Trade-off: bilinear interpolation on CUDA does not byte-match
        Pillow's bilinear, so detection boxes will drift sub-pixel vs.
        the CPU path. This is opt-in via ``gpu_preprocess=True``.
        """
        new_h, new_w = compute_resized_hw(h, w, self.min_size_test, self.max_size_test)
        src = torch.from_numpy(arr)
        if self.pinned_memory:
            buf = self._pinned_buf
            if buf is None or buf.shape[0] < h or buf.shape[1] < w:
                buf_h = max(buf.shape[0] if buf is not None else 0, h)
                buf_w = max(buf.shape[1] if buf is not None else 0, w)
                buf = torch.empty((buf_h, buf_w, 3), dtype=torch.uint8, pin_memory=True)
                self._pinned_buf = buf
            buf[:h, :w, :].copy_(src)
            gpu_uint8 = buf[:h, :w, :].to(self.device, non_blocking=True)
        else:
            gpu_uint8 = src.to(self.device)
        chw = gpu_uint8.permute(2, 0, 1).unsqueeze(0).float()
        if (new_h, new_w) != (h, w):
            chw = torch.nn.functional.interpolate(
                chw, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
        return chw[0]

    def batch(self, images: Sequence[ImageInput]) -> list[Instances]:
        """Run :meth:`__call__` over a sequence of images.

        Returns one :class:`Instances` per input image, each in the
        corresponding original image's coordinate space. This is a
        plain Python loop — it does **not** stack images into a single
        forward pass because every image needs its own resize. For
        true batched inference (one forward over a batch of equal-sized
        images), use :func:`mayaku.engine.inference_on_dataset` with a
        proper data loader.
        """
        return [self(im) for im in images]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(
        self,
        format: str,
        *,
        output: str | Path | None = None,
        sample_height: int = 640,
        sample_width: int = 640,
        coreml_precision: str = "fp32",
        onnx_dynamic_input_shape: bool = True,
    ) -> Path:
        """Serialise this detector to a deployment target; return the artifact path.

        The in-memory mirror of ``mayaku export <format>`` — writes the same
        graph (backbone+FPN body for R-CNN, full detector for UniQuery) using the
        same exporters.

        Args:
            format: One of ``"onnx" | "coreml" | "openvino" | "tensorrt"``.
            output: Artifact path. Defaults to ``<model-name>.<ext>`` in the cwd
                (``.onnx`` / ``.mlpackage`` / ``.xml`` / ``.engine``).
            sample_height / sample_width: Tracing input size.
            coreml_precision: ``"fp32"`` (default) or ``"fp16"`` — CoreML only.
            onnx_dynamic_input_shape: ONNX only; ``False`` bakes the sample shape
                (use when targeting TensorRT — see ``docs/export/onnx.md``).

        Example:
            >>> from mayaku import from_pretrained
            >>> model = from_pretrained("mayaku-s")
            >>> path = model.export("onnx")            # -> mayaku-s.onnx
        """
        from mayaku.inference.export.dispatch import (
            AVAILABLE_TARGETS,
            TARGET_SUFFIX,
            build_sample,
            export_detector,
        )
        from mayaku.utils.checkpoint import build_sidecar

        if format not in AVAILABLE_TARGETS:
            raise ValueError(
                f"unknown export format {format!r}; expected one of {AVAILABLE_TARGETS}"
            )
        out = (
            Path(output)
            if output is not None
            else Path(f"{self._source_stem}{TARGET_SUFFIX[format]}")
        )
        # Embed the self-describing sidecar when we know the config (the
        # from_pretrained path); a directly-constructed predictor has none.
        sidecar = (
            build_sidecar(self._cfg, self._class_names or []) if self._cfg is not None else None
        )
        sample = build_sample(sample_height, sample_width)
        result = export_detector(
            self.model,
            format,
            out,
            sample=sample,
            coreml_precision=coreml_precision,
            onnx_dynamic_input_shape=onnx_dynamic_input_shape,
            sidecar=sidecar,
        )
        return result.path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(model: nn.Module) -> torch.device:
    try:
        return next(p.device for p in model.parameters())
    except StopIteration:
        # Param-less model: fall back to a CPU buffer's device, then CPU.
        for buf in model.buffers():
            return buf.device
        return torch.device("cpu")


# Pre-exported artifact suffixes — the "the file is the backend" dispatch.
# Loading one of these runs the artifact end-to-end via its native runtime
# (see :class:`mayaku.inference.artifact.ArtifactPredictor`). A ``.pth`` / bare
# name is an eager checkpoint.
_ARTIFACT_SUFFIXES = frozenset({".onnx", ".engine", ".mlpackage", ".xml"})


def from_pretrained(
    source: str | Path,
    *,
    device: str = "auto",
    gpu_preprocess: bool = False,
    pinned_memory: bool = False,
) -> Predictor | ArtifactPredictor:
    """Load a deployable detector. The single public deploy entry point.

    Returns a :class:`Predictor` for a ``.pth``/bundled name, or an
    :class:`~mayaku.inference.artifact.ArtifactPredictor` for a pre-exported
    artifact — both are callable with an image and return :class:`Instances`.

    The ``source`` *suffix selects the backend*:

    - ``.pth`` (or a bundled model name) → eager Torch checkpoint. Architecture
      + weights come from the checkpoint's embedded sidecar (no config file).
    - ``.onnx`` / ``.engine`` / ``.mlpackage`` / ``.xml`` → a pre-exported
      artifact, run directly (not wired yet — the standalone full-graph runtime).

    Args:
        source: Path to a ``.pth`` / exported artifact, or a bundled model name.
        device: ``"cpu" | "cuda" | "mps" | "auto"`` (default ``"auto"``).
        gpu_preprocess: Do resize/normalize on-device (shortest-edge only).
        pinned_memory: Use a pinned staging buffer (requires ``gpu_preprocess``).

    Example:
        >>> from mayaku import from_pretrained
        >>> p = from_pretrained("model.pth")
        >>> dets = p("image.jpg")
    """
    from mayaku.backends.device import Device
    from mayaku.cli._factory import load_detector

    suffix = Path(source).suffix.lower()
    # The file *is* the backend: a pre-exported artifact runs end-to-end via its
    # native runtime, no checkpoint involved.
    if suffix in _ARTIFACT_SUFFIXES:
        from mayaku.inference.artifact import ArtifactPredictor

        return ArtifactPredictor.from_file(source, device=device)

    if device == "auto":
        device = Device.auto().kind
    # Architecture + weights + class names all come from the checkpoint's sidecar.
    cfg, model, class_names = load_detector(source)
    model = model.to(torch.device(device))  # Predictor.__init__ flips to eval()
    return Predictor._from_cfg(
        cfg,
        model,
        gpu_preprocess=gpu_preprocess,
        pinned_memory=pinned_memory,
        source_stem=Path(source).stem,
        class_names=class_names,
    )


def _to_uint8_rgb(image: ImageInput) -> npt.NDArray[np.uint8]:
    if isinstance(image, str | Path):
        return read_image(image)
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(
            "Predictor input must be an (H, W, 3) RGB array or a file "
            f"path; got shape {arr.shape}. Use mayaku.utils.bgr_to_rgb "
            "if your decoder produced BGR."
        )
    if arr.dtype != np.uint8:
        # Permit float arrays for convenience but coerce to the canonical
        # form so downstream PIL resize behaves predictably.
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr
