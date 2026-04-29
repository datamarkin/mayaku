"""Export every .pth in models/<task>/ to every viable deployment target.

Per model, produces five sibling artifacts in the same directory:

    <name>.onnx                       ONNX, dynamic shape (sample 640x640 trace)
    <name>.800x1344.fixed.onnx        ONNX, fixed shape 800x1344 (TRT-friendly)
    <name>.fp32.mlpackage             CoreML fp32, fixed 800x1344
    <name>.fp16.mlpackage             CoreML fp16, fixed 800x1344
    <name>.openvino.xml + .bin        OpenVINO IR, fixed 800x1344

TensorRT is skipped on macOS (no NVIDIA wheels). Re-run with --tensorrt
on a CUDA host to add `<name>.engine`.

Each variant runs in its own subprocess via `mayaku export` so a single
failure doesn't take down the rest. Existing artifacts are skipped
unless --force is passed.

Run:
    python tools/export_all.py
    python tools/export_all.py --force            # re-export even if present
    python tools/export_all.py --task detection   # one task only
    python tools/export_all.py --tensorrt         # add TRT (CUDA host)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
CONFIGS = REPO / "configs"

EVAL_H, EVAL_W = 800, 1344  # Mayaku's min_size_test=800, padded to mult of 32


def _build_jobs(target_filter: list[str], task_filter: str | None) -> list[dict]:
    jobs = []
    tasks = ("detection", "segmentation", "keypoints") if not task_filter else (task_filter,)
    for task in tasks:
        weights_dir = MODELS / task
        cfg_dir = CONFIGS / task
        if not weights_dir.is_dir() or not cfg_dir.is_dir():
            continue
        for pth in sorted(weights_dir.glob("*.pth")):
            name = pth.stem
            cfg = cfg_dir / f"{name}.yaml"
            if not cfg.is_file():
                print(f"[skip] no config for {pth.name} (looked for {cfg.name})")
                continue
            base = {"name": name, "task": task, "cfg": cfg, "weights": pth, "out_dir": weights_dir}
            for target in target_filter:
                jobs.append({**base, "target": target})
    return jobs


def _artifact_path(job: dict) -> Path:
    name = job["name"]
    out_dir = job["out_dir"]
    target = job["target"]
    return {
        "onnx-dynamic":   out_dir / f"{name}.onnx",
        "onnx-fixed":     out_dir / f"{name}.{EVAL_H}x{EVAL_W}.fixed.onnx",
        "coreml-fp32":    out_dir / f"{name}.fp32.mlpackage",
        "coreml-fp16":    out_dir / f"{name}.fp16.mlpackage",
        "openvino":       out_dir / f"{name}.openvino.xml",
        "tensorrt":       out_dir / f"{name}.engine",
    }[target]


def _command(job: dict, output: Path) -> list[str]:
    cfg, weights, target = str(job["cfg"]), str(job["weights"]), job["target"]
    base = ["mayaku", "export"]
    if target == "onnx-dynamic":
        return [*base, "onnx", cfg, "--weights", weights, "--output", str(output),
                "--sample-height", "640", "--sample-width", "640"]
    if target == "onnx-fixed":
        return [*base, "onnx", cfg, "--weights", weights, "--output", str(output),
                "--sample-height", str(EVAL_H), "--sample-width", str(EVAL_W),
                "--no-onnx-dynamic-shapes"]
    if target == "coreml-fp32":
        return [*base, "coreml", cfg, "--weights", weights, "--output", str(output),
                "--sample-height", str(EVAL_H), "--sample-width", str(EVAL_W),
                "--coreml-precision", "fp32"]
    if target == "coreml-fp16":
        return [*base, "coreml", cfg, "--weights", weights, "--output", str(output),
                "--sample-height", str(EVAL_H), "--sample-width", str(EVAL_W),
                "--coreml-precision", "fp16"]
    if target == "openvino":
        return [*base, "openvino", cfg, "--weights", weights, "--output", str(output),
                "--sample-height", str(EVAL_H), "--sample-width", str(EVAL_W)]
    if target == "tensorrt":
        return [*base, "tensorrt", cfg, "--weights", weights, "--output", str(output),
                "--sample-height", str(EVAL_H), "--sample-width", str(EVAL_W)]
    raise ValueError(f"unknown target {target!r}")


def _exists(out: Path) -> bool:
    if out.suffix == ".mlpackage" or out.is_dir():
        return out.is_dir() and any(out.iterdir())
    return out.exists()


def _remove(out: Path) -> None:
    if out.is_dir():
        shutil.rmtree(out)
    elif out.exists():
        out.unlink()
    # OpenVINO writes a sibling .bin; clean that too on --force.
    bin_path = out.with_suffix(".bin")
    if bin_path.exists():
        bin_path.unlink()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("detection", "segmentation", "keypoints"),
                   help="restrict to a single task")
    p.add_argument("--force", action="store_true", help="re-export even if artifact exists")
    p.add_argument("--tensorrt", action="store_true",
                   help="also export TensorRT (.engine) — CUDA host required")
    p.add_argument("--targets", default="onnx-dynamic,onnx-fixed,coreml-fp32,coreml-fp16,openvino",
                   help="comma-separated subset of variants to export")
    args = p.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if args.tensorrt and "tensorrt" not in targets:
        targets.append("tensorrt")

    jobs = _build_jobs(targets, args.task)
    if not jobs:
        print("[fatal] no jobs to run", file=sys.stderr)
        return 1

    print(f"[plan] {len(jobs)} export jobs across "
          f"{len({j['name'] for j in jobs})} models, "
          f"{len(targets)} variants per model", flush=True)
    for j in jobs:
        print(f"  - {j['task']}/{j['name']} -> {j['target']}", flush=True)

    started = time.perf_counter()
    results = []
    for i, job in enumerate(jobs, 1):
        out = _artifact_path(job)
        label = f"[{i:>2}/{len(jobs)}] {job['task']}/{job['name']} -> {job['target']}"

        if _exists(out) and not args.force:
            print(f"{label}  SKIP (exists at {out.name})", flush=True)
            results.append({"job": job, "status": "skip", "path": out, "elapsed": 0.0})
            continue
        if args.force and _exists(out):
            _remove(out)

        cmd = _command(job, out)
        print(f"{label}  RUNNING  ->  {out.name}", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
        elapsed = time.perf_counter() - t0

        if proc.returncode == 0 and _exists(out):
            print(f"{label}  OK ({elapsed:.1f}s)", flush=True)
            results.append({"job": job, "status": "ok", "path": out, "elapsed": elapsed})
        else:
            print(f"{label}  FAIL ({elapsed:.1f}s, rc={proc.returncode})", flush=True)
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-10:]
            for line in tail:
                print(f"    | {line}", flush=True)
            results.append({"job": job, "status": "fail", "path": out, "elapsed": elapsed,
                            "stderr_tail": "\n".join(tail)})

    total_elapsed = time.perf_counter() - started
    ok = sum(1 for r in results if r["status"] == "ok")
    skip = sum(1 for r in results if r["status"] == "skip")
    fail = sum(1 for r in results if r["status"] == "fail")
    print(f"\n[summary] ok={ok}  skip={skip}  fail={fail}  "
          f"total={total_elapsed:.1f}s", flush=True)

    if fail:
        print("\n[failures]", flush=True)
        for r in results:
            if r["status"] == "fail":
                j = r["job"]
                print(f"  - {j['task']}/{j['name']} -> {j['target']}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
