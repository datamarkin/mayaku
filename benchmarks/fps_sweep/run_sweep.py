#!/usr/bin/env python3
"""Drive the full FPS sweep: 20 configs x 4 backends, one subprocess per cell.

Resumable: a cell whose result JSON already exists with ok=True is skipped, so
you can Ctrl-C and re-run. Each cell runs in its own process for VRAM/compile-
state isolation, with a hard timeout so a hung build can't stall the sweep.

  python run_sweep.py            # run the whole matrix
  python assemble.py             # then build fps.md + fps.csv

Results land in ./runs/results/*.json (next to this script).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RESULTS = RUNS / "results"
WORK = RUNS / "work"
BENCH = HERE / "bench_one.py"
LOG = RUNS / "sweep.log"

VARIANT_DIMS = {
    "convnext_atto":  [96, 128, 192, 256],
    "convnext_femto": [96, 128, 192, 256],
    "convnext_pico":  [128, 192, 256],
    "convnext_nano":  [128, 192, 256],
    "convnext_tiny":  [128, 192, 256],
    "convnext_base":  [128, 192, 256],
}
BACKENDS = ["eager", "compile", "trt_fp32", "trt_fp16"]
CELL_TIMEOUT = 900  # seconds per cell


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def already_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text()).get("ok"))
    except Exception:
        return False


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    cells = [
        (v, d, b)
        for v, dims in VARIANT_DIMS.items()
        for d in dims
        for b in BACKENDS
    ]
    total = len(cells)
    log(f"Sweep start: {total} cells")
    t_start = time.time()
    for i, (variant, dim, backend) in enumerate(cells, 1):
        out = RESULTS / f"{variant}_{dim}_{backend}.json"
        tag = f"{variant} hd{dim} {backend} ({i}/{total})"
        if already_ok(out):
            log(f"SKIP {tag} (cached)")
            continue
        log(f"RUN  {tag}")
        t0 = time.time()
        try:
            subprocess.run(
                [sys.executable, str(BENCH),
                 "--variant", variant, "--hidden-dim", str(dim),
                 "--backend", backend, "--out", str(out), "--workdir", str(WORK)],
                timeout=CELL_TIMEOUT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            out.write_text(json.dumps({
                "variant": variant, "hidden_dim": dim, "backend": backend,
                "ok": False, "error": f"timeout after {CELL_TIMEOUT}s",
            }, indent=2))
            log(f"  TIMEOUT {tag}")
            continue
        dt = time.time() - t0
        try:
            r = json.loads(out.read_text())
            status = f"ok fps={r.get('fps')}" if r.get("ok") else f"FAIL {r.get('error')}"
        except Exception as exc:
            status = f"no-json ({exc})"
        log(f"  done {tag} in {dt:.0f}s -> {status}")
    log(f"Sweep complete in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
