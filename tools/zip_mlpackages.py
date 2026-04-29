"""Zip every .mlpackage directory under models/ as a sibling .mlpackage.zip.

CoreML's `.mlpackage` is a directory tree (Manifest.json + Data/), so it
can't be downloaded as a single file. This script produces stored-mode
(no-compression) zip archives next to each `.mlpackage` so they can be
hosted as one URL each. Stored mode because the fp16 weights inside are
already quantised — deflate saves <1% and slows extraction.

Run from the repo root:

    python tools/zip_mlpackages.py
    python tools/zip_mlpackages.py --force        # overwrite existing zips
    python tools/zip_mlpackages.py --task detection
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"


def _zip_mlpackage(src: Path, force: bool) -> tuple[Path, str]:
    """Zip a single .mlpackage directory in-place; return (path, status)."""
    out = src.with_suffix(src.suffix + ".zip")  # <name>.fp16.mlpackage.zip
    if out.exists() and not force:
        return out, "skip"
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        # Walk in deterministic order — keeps hashes stable across machines.
        for path in sorted(src.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(src.parent)  # include the .mlpackage/ dir
                zf.write(path, arcname=arcname)
    return out, "ok"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("detection", "segmentation", "keypoints"),
                   help="restrict to a single task")
    p.add_argument("--force", action="store_true", help="overwrite existing .zip")
    args = p.parse_args()

    tasks = ("detection", "segmentation", "keypoints") if not args.task else (args.task,)
    pkgs: list[Path] = []
    for task in tasks:
        d = MODELS / task
        if d.is_dir():
            pkgs.extend(sorted(d.glob("*.mlpackage")))

    if not pkgs:
        print("[fatal] no .mlpackage directories under models/", file=sys.stderr)
        return 1

    print(f"[plan] {len(pkgs)} mlpackages to zip", flush=True)
    started = time.perf_counter()
    ok = skip = 0
    for i, pkg in enumerate(pkgs, 1):
        t0 = time.perf_counter()
        out, status = _zip_mlpackage(pkg, force=args.force)
        elapsed = time.perf_counter() - t0
        rel = out.relative_to(REPO)
        if status == "ok":
            size_mb = out.stat().st_size / (1024 * 1024)
            print(f"[{i:>2}/{len(pkgs)}] {rel}  OK  {size_mb:.1f} MB  ({elapsed:.1f}s)", flush=True)
            ok += 1
        else:
            print(f"[{i:>2}/{len(pkgs)}] {rel}  SKIP (exists)", flush=True)
            skip += 1

    print(f"\n[summary] ok={ok}  skip={skip}  total={time.perf_counter() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
