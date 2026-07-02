"""Walk models/ and emit manifest.json describing every hosted checkpoint.

The manifest is the source of truth that the client downloader reads: each
(model, revision) maps to a ``.pth`` at a relative path under
``https://dtmfiles.com/mayaku/v1/models/``, with size + sha256 so the client can
verify integrity and skip cache hits.

Run from the repo root:

    python tools/generate_manifest.py
    python tools/generate_manifest.py --output models/manifest.json   # default
    python tools/generate_manifest.py --base-url https://example.com/...

The output file is what the user uploads to
``dtmfiles.com/mayaku/v1/models/manifest.json``.

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
        "2026-07-15": {"path": ..., "size": ..., "sha256": ...},
        ...
      }
    }

Only the ``.pth`` checkpoint is hosted; deployment artifacts (onnx/coreml/
openvino/tensorrt) are produced locally via ``model.export(...)`` / ``mayaku
export`` from the .pth.
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
            "which has no checkpoint; falling back to newest",
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
            entry["revisions"][rev] = _entry(pth)

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
    print(f"\n[summary] {len(models)} models, {n_rev} revisions, "
          f"{elapsed:.1f}s -> {args.output.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
