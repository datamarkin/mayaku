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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--version", default="v1")
    args = p.parse_args()

    started = time.perf_counter()
    models: dict[str, dict] = {}
    for task in ("detection", "segmentation", "keypoints"):
        d = MODELS / task
        if not d.is_dir():
            continue
        for pth in sorted(d.glob("*.pth")):
            name = pth.stem
            print(f"[hash] {task}/{name}", flush=True)
            models[name] = {
                "task": task,
                "variants": _build_variants(pth),
            }

    payload = {
        "version": args.version,
        "base_url": args.base_url,
        "models": models,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    elapsed = time.perf_counter() - started
    n_variants = sum(len(m["variants"]) for m in models.values())
    print(f"\n[summary] {len(models)} models, {n_variants} variants total, "
          f"{elapsed:.1f}s -> {args.output.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
