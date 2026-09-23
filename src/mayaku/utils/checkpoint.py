"""Self-describing checkpoints: the "mayaku" sidecar, and reading it back.

Every checkpoint and every exported artifact carries the same JSON sidecar
next to the weights, so predict / eval / export rebuild the model and decode
its outputs from the file alone. `build_sidecar` is its only writer and
`check_sidecar` / `read_deploy_checkpoint` its readers; `save_checkpoint` /
`load_checkpoint` own the checkpoint container around it.

Two views, one source. `config` is the full `MayakuConfig`, for rebuilding
the model in Python. Everything else is the flat runtime contract -- canvas,
outputs, decode, preprocessing, mask and keypoint constants, quantization --
which a non-Python runtime reads without understanding the config. The flat
view is derived from the config and the trained model in `build_sidecar`, so
the two cannot disagree.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from mayaku.config import MayakuConfig
    from mayaku.model import Detector

__all__ = [
    "SIDECAR_KEY",
    "SIDECAR_SCHEMA_VERSION",
    "build_sidecar",
    "check_sidecar",
    "git_hash",
    "load_checkpoint",
    "read_deploy_checkpoint",
    "save_checkpoint",
    "select_final_weights",
]

#: Where the sidecar lives: the checkpoint dict key and every artifact's
#: metadata key.
SIDECAR_KEY = "mayaku"

#: Version of the sidecar layout. 1 is mayaku 2.x (R-CNN / UniQuery models).
SIDECAR_SCHEMA_VERSION = 2


def build_sidecar(
    cfg: MayakuConfig,
    class_names: Sequence[str],
    model: Detector,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the sidecar for a trained `model` built from `cfg`.

    The model's own `deploy_spec` supplies what it computes (outputs,
    strides, DFL bins, mask and keypoint constants, whether it is QAT); the
    config supplies the canvas, the decode thresholds and the keypoint names;
    the data layer supplies the preprocessing contract. `provenance` adds
    keys to the version and git hash recorded here.
    """
    import mayaku
    from mayaku.data.geometry import PREPROCESS

    canvas = cfg.input.canvas
    if not cfg.model.num_classes == model.nc == len(class_names):
        raise ValueError(f"model.num_classes {cfg.model.num_classes}, the head's {model.nc} "
                         f"and {len(class_names)} class names disagree")
    spec, d, kp = model.deploy_spec(), cfg.train.decode, cfg.model.keypoints
    keypoints = spec["keypoints"]
    if keypoints and kp:
        keypoints = {**keypoints, "names": list(kp.names), "flip_pairs": [list(p) for p in kp.flip_pairs]}
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "config": cfg.model_dump(mode="json"),
        "class_names": list(class_names),
        "canvas_hw": list(canvas),
        "outputs": spec["outputs"],
        "decode": {"strides": spec["strides"], "reg_max": spec["reg_max"], "conf": d.conf,
                   "iou": d.iou, "max_det": d.max_det, "topk": d.topk,
                   "multi_label": d.multi_label},
        "preprocess": dict(PREPROCESS),
        "mask": spec["mask"],
        "keypoints": keypoints,
        "quant": {"qat": spec["qat"],
                  "weights": "int8 per-channel symmetric",
                  "activations": "int8 per-tensor affine"},
        "provenance": {"mayaku_version": getattr(mayaku, "__version__", None), "git": git_hash(),
                       **(provenance or {})},
    }


def check_sidecar(sidecar: Mapping[str, Any] | None, source: str) -> Mapping[str, Any]:
    """Validate that `sidecar` is one this version reads, and return it.

    A missing sidecar means an externally produced file; schema version 1
    means a mayaku 2.x model (R-CNN / UniQuery), which only mayaku<3 runs.
    """
    if not sidecar or not isinstance(sidecar.get("config"), dict):
        raise ValueError(f"{source} carries no mayaku sidecar: it was not written by mayaku, "
                         "or its metadata was stripped")
    version = sidecar.get("schema_version")
    if version != SIDECAR_SCHEMA_VERSION:
        if version == 1:
            raise ValueError(f"{source} was trained with mayaku 2.x (sidecar schema 1); "
                             "run it with `pip install 'mayaku<3'`")
        raise ValueError(f"{source} has sidecar schema {version!r}; this mayaku reads "
                         f"{SIDECAR_SCHEMA_VERSION}")
    return sidecar


def save_checkpoint(model: torch.nn.Module, path: str | Path,
                    sidecar: Mapping[str, Any] | None = None) -> None:
    """The weights, with the sidecar next to them when there is one
    (``{"model": state, SIDECAR_KEY: sidecar}``), else the bare state dict."""
    state = model.state_dict()
    torch.save({"model": state, SIDECAR_KEY: dict(sidecar)} if sidecar else state, path)


def load_checkpoint(checkpoint_path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Deserialize a checkpoint once: ``(sidecar, model_state)``.

    ``sidecar`` is ``None`` for a bare state dict. ``weights_only=True`` is
    safe: the sidecar holds only JSON primitives, which the restricted
    unpickler allows alongside tensors.
    """
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or "model" not in obj:
        return None, obj
    sidecar = obj.get(SIDECAR_KEY)
    return (sidecar if isinstance(sidecar, dict) else None), obj["model"]


def read_deploy_checkpoint(
    checkpoint_path: Path,
) -> tuple[dict[str, Any], MayakuConfig, dict[str, Any]]:
    """``(sidecar, config, model_state)`` from a self-describing checkpoint, in
    one deserialize. Raises ``ValueError`` for a checkpoint without a v3
    sidecar (see `check_sidecar`)."""
    from mayaku.config import MayakuConfig

    sidecar, state = load_checkpoint(checkpoint_path)
    sidecar = dict(check_sidecar(sidecar, str(checkpoint_path)))
    return sidecar, MayakuConfig.model_validate(sidecar["config"]), state


def select_final_weights(train_dir: Path) -> Path:
    """The run's deliverable weights: ``best.pt`` (the EMA model at its best
    validation AP) when the run evaluated, else ``last.pt``. Raises
    ``RuntimeError`` when the run produced neither."""
    for name in ("best.pt", "last.pt"):
        if (train_dir / name).exists():
            return train_dir / name
    raise RuntimeError(f"no checkpoint under {train_dir}: training likely failed "
                       "before its first epoch finished")


def git_hash() -> str | None:
    """Best-effort short git hash of the working directory, for provenance.
    ``None`` when it cannot be determined; never raises."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None
