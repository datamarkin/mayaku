"""Embed / read the mayaku sidecar inside exported artifacts.

A checkpoint is self-describing: `mayaku.utils.checkpoint.build_sidecar`
writes the sidecar under a ``"mayaku"`` key. This module gives every export
format the same property -- each has a metadata slot the same JSON goes into,
so ``from_pretrained("model.onnx")`` runs the artifact from the file alone.

Per-format slot:

* ONNX      — ``model.metadata_props`` (key/value strings)
* CoreML    — ``MLModel.user_defined_metadata``
* OpenVINO  — model ``rt_info``
* TensorRT  — the ``.engine`` is opaque binary with no metadata slot, so the
  JSON is length-prefixed in front of the engine bytes (``<4-byte LE len><json>
  <engine>``); `strip_tensorrt_header` removes it before deserialising.

The JSON is written compact (no spaces) so it survives OpenVINO ``rt_info``
(which historically splits string values on whitespace).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["SIDECAR_KEY", "SUFFIX_TO_TARGET", "embed_sidecar", "read_sidecar", "target_from_suffix"]

from mayaku.utils.checkpoint import SIDECAR_KEY

SUFFIX_TO_TARGET: dict[str, str] = {
    ".onnx": "onnx",
    ".mlpackage": "coreml",
    ".xml": "openvino",
    ".engine": "tensorrt",
}


def target_from_suffix(path: str | Path) -> str:
    """Map an artifact path's suffix to its export target name."""
    suffix = Path(path).suffix.lower()
    target = SUFFIX_TO_TARGET.get(suffix)
    if target is None:
        raise ValueError(
            f"unrecognised artifact suffix {suffix!r}; expected one of {sorted(SUFFIX_TO_TARGET)}"
        )
    return target


def sidecar_blob(sidecar: dict[str, Any]) -> str:
    """The sidecar as the compact JSON every artifact slot stores: no spaces,
    so it survives OpenVINO ``rt_info`` (which historically splits string
    values on whitespace)."""
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
    """Read the sidecar dict from ``path``, or ``None`` if it carries none.
    One written before exports recorded their precision gets fp32."""
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
    parsed.setdefault("export", {"target": target, "precision": "fp32"})
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
    with open(path, "rb") as f:                 # the header only, not the engine
        head = f.read(_TRT_LEN_BYTES)
        if len(head) < _TRT_LEN_BYTES:
            return None
        blob = f.read(int.from_bytes(head, "little"))
    try:
        return blob.decode("utf-8") if blob.startswith(b"{") else None
    except UnicodeDecodeError:
        return None


def strip_tensorrt_header(path: Path) -> bytes:
    """The raw engine bytes of a ``.engine`` that may carry a sidecar header
    (``<len><json>`` in front, from `_embed_tensorrt`); a file without one is
    returned whole."""
    data = path.read_bytes()
    blob = _read_tensorrt(path)
    return data[_TRT_LEN_BYTES + len(blob.encode("utf-8")):] if blob else data
