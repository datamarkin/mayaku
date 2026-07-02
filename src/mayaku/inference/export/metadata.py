"""Embed / read the mayaku sidecar inside exported artifacts.

The ``.pth`` checkpoint is self-describing: ``build_sidecar`` writes
``{config, class_names, ...}`` under a ``"mayaku"`` key and
``config_from_checkpoint`` reads it back. This module gives every export format
the same property — each has a metadata slot we write the same JSON into, so
``from_pretrained("model.onnx")`` reconstructs the architecture + class names
from the file alone (no sidecar file, no config).

Per-format slot:

* ONNX      — ``model.metadata_props`` (key/value strings)
* CoreML    — ``MLModel.user_defined_metadata``
* OpenVINO  — model ``rt_info``
* TensorRT  — the ``.engine`` is opaque binary with no metadata slot, so the
  JSON is length-prefixed in front of the engine bytes (``<4-byte LE len><json>
  <engine>``); :class:`mayaku.inference.artifact` strips it before deserialising.

The JSON is written compact (no spaces) so it survives OpenVINO ``rt_info``
(which historically splits string values on whitespace).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["SIDECAR_KEY", "embed_sidecar", "read_sidecar", "target_from_suffix"]

SIDECAR_KEY = "mayaku"

_SUFFIX_TO_TARGET: dict[str, str] = {
    ".onnx": "onnx",
    ".mlpackage": "coreml",
    ".xml": "openvino",
    ".engine": "tensorrt",
}


def target_from_suffix(path: str | Path) -> str:
    """Map an artifact path's suffix to its export target name."""
    suffix = Path(path).suffix.lower()
    target = _SUFFIX_TO_TARGET.get(suffix)
    if target is None:
        raise ValueError(
            f"unrecognised artifact suffix {suffix!r}; expected one of "
            f"{sorted(_SUFFIX_TO_TARGET)}"
        )
    return target


def sidecar_blob(sidecar: dict[str, Any]) -> str:
    """Serialise the sidecar to the compact JSON stored in every artifact slot.

    No spaces so it survives OpenVINO ``rt_info`` (which historically splits
    string values on whitespace). Shared by the post-hoc embedders here and the
    CoreML/OpenVINO exporters that embed inline at write time.
    """
    return json.dumps(sidecar, separators=(",", ":"))


def embed_sidecar(path: Path, target: str, sidecar: dict[str, Any]) -> None:
    """Write ``sidecar`` into ``path``'s metadata slot, post-hoc.

    Only ``onnx`` and ``tensorrt`` embed post-hoc: ONNX load-modify-save is cheap
    and safe, and the ``.engine`` is opaque so the JSON is length-prefixed onto
    it. CoreML/OpenVINO embed inline at export time — re-saving over a just-
    written ``.mlpackage``/IR in place is unsafe (copy-over-self / mmap SIGBUS) —
    so they are handled in their exporters, not here.
    """
    blob = sidecar_blob(sidecar)
    if target == "onnx":
        _embed_onnx(path, blob)
    elif target == "tensorrt":
        _embed_tensorrt(path, blob)
    else:
        raise ValueError(
            f"{target!r} embeds its sidecar inline at export time, not via embed_sidecar()"
        )


def read_sidecar(path: Path, target: str) -> dict[str, Any] | None:
    """Read the sidecar dict from ``path``, or ``None`` if it carries none."""
    if target == "onnx":
        blob = _read_onnx(path)
    elif target == "coreml":
        blob = _read_coreml(path)
    elif target == "openvino":
        blob = _read_openvino(path)
    elif target == "tensorrt":
        blob = _read_tensorrt(path)
    else:
        raise ValueError(f"unknown export target {target!r}")
    if not blob:
        return None
    parsed: dict[str, Any] = json.loads(blob)
    return parsed


# --- ONNX ------------------------------------------------------------------


def _embed_onnx(path: Path, blob: str) -> None:
    import onnx

    model = onnx.load(str(path))
    # Drop any pre-existing key so a re-embed doesn't leave duplicates.
    keep = [p for p in model.metadata_props if p.key != SIDECAR_KEY]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)
    entry = model.metadata_props.add()
    entry.key = SIDECAR_KEY
    entry.value = blob
    onnx.save(model, str(path))


def _read_onnx(path: Path) -> str | None:
    import onnx

    model = onnx.load(str(path))
    for prop in model.metadata_props:
        if prop.key == SIDECAR_KEY:
            return str(prop.value)
    return None


# --- CoreML ----------------------------------------------------------------


def _read_coreml(path: Path) -> str | None:
    import coremltools as ct

    model = ct.models.MLModel(str(path))
    value = model.user_defined_metadata.get(SIDECAR_KEY)
    return str(value) if value is not None else None


# --- OpenVINO --------------------------------------------------------------


def _read_openvino(path: Path) -> str | None:
    import openvino as ov

    core = ov.Core()
    model = core.read_model(str(path))
    try:
        value = model.get_rt_info([SIDECAR_KEY]).astype(str)
    except Exception:
        return None
    return str(value)


# --- TensorRT --------------------------------------------------------------

_TRT_LEN_BYTES = 4


def _embed_tensorrt(path: Path, blob: str) -> None:
    data = path.read_bytes()
    payload = blob.encode("utf-8")
    header = len(payload).to_bytes(_TRT_LEN_BYTES, "little")
    path.write_bytes(header + payload + data)


def _read_tensorrt(path: Path) -> str | None:
    data = path.read_bytes()
    if len(data) < _TRT_LEN_BYTES:
        return None
    n = int.from_bytes(data[:_TRT_LEN_BYTES], "little")
    start = _TRT_LEN_BYTES
    end = start + n
    if end > len(data):
        return None
    return data[start:end].decode("utf-8")


def strip_tensorrt_header(path: Path) -> bytes:
    """Return the raw engine bytes from a ``.engine`` that may carry a sidecar header.

    :func:`_embed_tensorrt` prepends ``<len><json>`` in front of the engine. The
    TensorRT session calls this to recover the deserialisable engine bytes. Files
    without a header (no ``read_sidecar`` value) are returned unchanged.
    """
    data = path.read_bytes()
    if len(data) < _TRT_LEN_BYTES:
        return data
    n = int.from_bytes(data[:_TRT_LEN_BYTES], "little")
    end = _TRT_LEN_BYTES + n
    if end > len(data):
        return data  # not our header
    return data[end:]
