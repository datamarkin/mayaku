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

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from mayaku.config.schemas import MayakuConfig
from mayaku.data.transforms import ResizeShortestEdge
from mayaku.inference.postprocess import detector_postprocess
from mayaku.structures.instances import Instances
from mayaku.utils.image import read_image

__all__ = ["Predictor"]

ImageInput = npt.NDArray[np.uint8] | str | Path


class Predictor:
    """Single-image / batch inference wrapper around a built detector.

    Args:
        model: A built and loaded detector (e.g.
            :class:`mayaku.models.detectors.FasterRCNN`). Will be put
            into eval mode.
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
        min_size_test: int = 800,
        max_size_test: int = 1333,
        device: torch.device | None = None,
    ) -> None:
        if min_size_test <= 0 or max_size_test <= 0:
            raise ValueError(
                f"min_size_test / max_size_test must be > 0; got ({min_size_test}, {max_size_test})"
            )
        self.model = model.eval()
        self.min_size_test = min_size_test
        self.max_size_test = max_size_test
        self.device = device or _resolve_device(model)
        self._resize = ResizeShortestEdge(
            short_edge_lengths=(min_size_test,),
            max_size=max_size_test,
            sample_style="choice",
        )

    # ------------------------------------------------------------------
    # Construction from a config
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: MayakuConfig, model: nn.Module) -> Predictor:
        """Build a predictor with sizes pulled from a :class:`MayakuConfig`.

        ``model`` is the already-built detector (the predictor doesn't
        construct one itself — that decision belongs to the CLI / user
        script so they can choose between Faster / Mask / Keypoint
        R-CNN explicitly and load weights when appropriate).
        """
        return cls(
            model,
            min_size_test=cfg.input.min_size_test,
            max_size_test=cfg.input.max_size_test,
        )

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        *,
        weights: str | Path | None = None,
        config: str | Path | None = None,
        device: str = "auto",
    ) -> Predictor:
        """Build a fully-loaded :class:`Predictor` from a model name in one call.

        This is the high-level zero-config constructor. ``name`` resolves
        both the bundled YAML config (via :mod:`mayaku.configs`) and the
        cached weights (via the model manifest). Override either
        independently with ``weights=`` / ``config=``.

        Args:
            name: Bundled model name, e.g. ``"faster_rcnn_R_50_FPN_3x"``.
            weights: Override the weights — accepts a model name (resolved
                via the manifest) or a filesystem path to a ``.pth``.
                Defaults to ``name``.
            config: Override the config — accepts a bundled config name
                (short or qualified ``task/name``) or a filesystem path
                to a ``.yaml``. Defaults to ``name``.
            device: ``"cpu" | "cuda" | "mps" | "auto"``. Defaults to
                ``"auto"`` which picks the best available accelerator.

        Example:
            >>> from mayaku.inference import Predictor
            >>> predictor = Predictor.from_pretrained("faster_rcnn_R_50_FPN_3x")
            >>> instances = predictor("photo.jpg")
        """
        from mayaku import configs
        from mayaku.backends.device import Device
        from mayaku.cli._factory import build_detector
        from mayaku.cli._weights import resolve_weights
        from mayaku.config import load_yaml

        if device == "auto":
            device = Device.auto().kind

        config_arg: str | Path = config if config is not None else name
        if isinstance(config_arg, Path) or Path(str(config_arg)).is_file():
            config_path = Path(str(config_arg))
        else:
            config_path = configs.path(str(config_arg))
        cfg = load_yaml(config_path)

        weights_path = resolve_weights(weights if weights is not None else name)
        if weights_path is None:
            raise ValueError(f"Could not resolve weights for {weights or name!r}")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model = build_detector(cfg).eval()
        model.load_state_dict(state)
        model = model.to(torch.device(device))

        return cls.from_config(cfg, model)

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
        resized = self._resize.get_transform(arr).apply_image(arr)
        # CHW float32 on the model's device.
        chw = np.ascontiguousarray(resized.transpose(2, 0, 1))
        img_tensor = torch.from_numpy(chw).to(dtype=torch.float32, device=self.device)
        inputs = [{"image": img_tensor, "height": h, "width": w}]
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
        if instances.image_size != (h, w):
            instances = detector_postprocess(instances, h, w)
        return instances

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
