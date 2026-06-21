"""Walk models/ and emit manifest.json describing every hosted artifact.

The manifest is the source of truth that the client downloader reads:
each (model, variant) pair maps to a relative path under
``https://dtmfiles.com/mayaku/v1/models/``, with size + sha256 so the
client can verify integrity and skip cache hits.

Run from the repo root:

    python tools/generate_manifest.py
    python tools/generate_manifest.py --output models/manifest.json   # default
    python tools/generate_manifest.py --base-url https://example.com/...

The output file is what the user uploads to
``dtmfiles.com/mayaku/v1/models/manifest.json``.

Variants enumerated (per model, only those present on disk):

    pth          <name>.pth
    onnx         <name>.onnx
    onnx-fixed   <name>.800x1344.fixed.onnx
    coreml-fp16  <name>.fp16.mlpackage.zip       (run zip_mlpackages.py first)
    openvino     <name>.openvino.xml + .bin      (one entry, two URLs)

On-disk layout under each task dir:

    models/<task>/<name>/<revision>/<name>.pth

The ``latest`` pointer is the revision named in a
``models/<task>/<name>/LATEST`` file if present, else the
lexicographically greatest id (correct for ISO dates like 2026-07-15).

Emitted schema (per model)::

    "<name>": {
      "task": "detection",
      "latest": "2026-07-15",
      "revisions": {
        "2026-07-15": {"variants": {"pth": {path,size,sha256}, ...}},
        ...
      }
    }

TensorRT .engine files are intentionally not enumerated — they're
GPU-arch-specific. Users with CUDA hosts run `mayaku export tensorrt`
locally from the downloaded .onnx.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
DEFAULT_BASE_URL = "https://dtmfiles.com/mayaku/v1/models"
DEFAULT_OUTPUT = MODELS / "manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MB chunks
            h.update(chunk)
    return h.hexdigest()


def _entry(path: Path) -> dict:
    return {
        "path": str(path.relative_to(MODELS)),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _build_variants(weights: Path) -> dict[str, dict]:
    """Look up every variant that exists alongside `weights`. Skip missing."""
    name = weights.stem  # 'faster_rcnn_R_50_FPN_3x'
    parent = weights.parent
    candidates: dict[str, Path | tuple[Path, Path]] = {
        "pth":         parent / f"{name}.pth",
        "onnx":        parent / f"{name}.onnx",
        "onnx-fixed":  parent / f"{name}.800x1344.fixed.onnx",
        "coreml-fp16": parent / f"{name}.fp16.mlpackage.zip",
    }
    xml = parent / f"{name}.openvino.xml"
    bin_ = parent / f"{name}.openvino.bin"
    if xml.exists() and bin_.exists():
        candidates["openvino"] = (xml, bin_)

    out: dict[str, dict] = {}
    for key, val in candidates.items():
        if isinstance(val, tuple):
            xml_p, bin_p = val
            out[key] = {
                "files": [_entry(xml_p), _entry(bin_p)],
                "primary": str(xml_p.relative_to(MODELS)),  # the one to load
            }
        elif val.exists():
            out[key] = _entry(val)
        # else: silently skip — the client surfaces "variant not available".
    return out


def _discover(task_dir: Path):
    """Yield ``(name, revision, pth)`` for ``<name>/<rev>/<name>.pth``."""
    for model_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
        name = model_dir.name
        for rev_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            pth = rev_dir / f"{name}.pth"
            if pth.is_file():
                yield name, rev_dir.name, pth
            else:
                print(f"[skip] {rev_dir.relative_to(MODELS)} has no {name}.pth", flush=True)


def _resolve_latest(model_dir: Path, revisions: dict) -> str:
    """Pick the ``latest`` pointer for one model.

    Honors a ``LATEST`` file (single line: the revision id) when present;
    otherwise the lexicographically greatest id — correct for ISO dates
    (``2026-07-15``) and zero-padded counters (``r001``). Bare ``r1..r10``
    mis-sort; use a LATEST file or zero-pad if you go that route.
    """
    latest_file = model_dir / "LATEST"
    if latest_file.is_file():
        rev = latest_file.read_text().strip()
        if rev in revisions:
            return rev
        print(
            f"[warn] {latest_file.relative_to(MODELS)} points at {rev!r} "
            "which has no artifacts; falling back to newest",
            flush=True,
        )
    return sorted(revisions)[-1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--version", default="v1")
    args = p.parse_args()

    started = time.perf_counter()
    models: dict[str, dict[str, Any]] = {}
    for task in ("detection", "segmentation", "keypoints"):
        d = MODELS / task
        if not d.is_dir():
            continue
        for name, rev, pth in _discover(d):
            print(f"[hash] {task}/{name}@{rev}", flush=True)
            entry = models.setdefault(name, {"task": task, "revisions": {}})
            entry["revisions"][rev] = {"variants": _build_variants(pth)}

    # Resolve each model's `latest` pointer once all revisions are known.
    for name, entry in models.items():
        model_dir = MODELS / str(entry["task"]) / name
        entry["latest"] = _resolve_latest(model_dir, entry["revisions"])

    payload = {
        "version": args.version,
        "base_url": args.base_url,
        "models": models,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    elapsed = time.perf_counter() - started
    n_rev = sum(len(m["revisions"]) for m in models.values())
    n_variants = sum(
        len(r["variants"]) for m in models.values() for r in m["revisions"].values()
    )
    print(f"\n[summary] {len(models)} models, {n_rev} revisions, {n_variants} variants total, "
          f"{elapsed:.1f}s -> {args.output.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
