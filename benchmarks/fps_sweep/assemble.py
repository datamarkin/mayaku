#!/usr/bin/env python3
"""Aggregate per-cell JSON results into fps.md and fps.csv (repo root)."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "runs" / "results"
OUT_MD = REPO / "fps.md"
OUT_CSV = REPO / "fps.csv"

VARIANT_DIMS = {
    "convnext_atto":  [96, 128, 192, 256],
    "convnext_femto": [96, 128, 192, 256],
    "convnext_pico":  [128, 192, 256],
    "convnext_nano":  [128, 192, 256],
    "convnext_tiny":  [128, 192, 256],
    "convnext_base":  [128, 192, 256],
}
BACKENDS = ["eager", "compile", "trt_fp32", "trt_fp16"]
PRETTY = {
    "convnext_atto": "Atto", "convnext_femto": "Femto", "convnext_pico": "Pico",
    "convnext_nano": "Nano", "convnext_tiny": "Tiny", "convnext_base": "Base",
}


def env_line() -> str:
    parts = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader", "-i", "0"],
            capture_output=True, text=True,
        ).stdout.strip().splitlines()[0]
        name, mem, drv = [x.strip() for x in out.split(",")]
        parts.append(f"{name} ({mem})")
        parts.append(f"driver {drv}")
    except Exception:
        parts.append("GPU: unknown")
    try:
        import torch
        parts.append(f"torch {torch.__version__}")
    except Exception:
        pass
    try:
        import tensorrt as trt
        parts.append(f"TensorRT {trt.__version__}")
    except Exception:
        pass
    parts.append("ONNX opset 17")
    return " · ".join(parts)


def load(variant: str, dim: int, backend: str) -> dict:
    p = RESULTS / f"{variant}_{dim}_{backend}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def m(n) -> str:
    return f"{n / 1e6:.2f}" if isinstance(n, (int, float)) else "—"


def f1(x) -> str:
    return f"{x:.1f}" if isinstance(x, (int, float)) else "—"


def collect_row(variant: str, dim: int) -> dict:
    cells = {b: load(variant, dim, b) for b in BACKENDS}
    params = {}
    for b in BACKENDS:
        c = cells[b]
        if c.get("total_params"):
            params = c
            break
    return {
        "variant": PRETTY[variant],
        "timm_name": variant,
        "hidden_dim": dim,
        "backbone_params_m": params.get("backbone_params"),
        "head_params_m": params.get("head_params"),
        "total_params_m": params.get("total_params"),
        "eager_fps": cells["eager"].get("fps"),
        "compile_fp16_fps": cells["compile"].get("fps"),
        "compile_time_s": cells["compile"].get("compile_time_s"),
        "trt_fp32_fps": cells["trt_fp32"].get("fps"),
        "trt_fp16_fps": cells["trt_fp16"].get("fps"),
        "trt_fp32_build_s": cells["trt_fp32"].get("engine_build_s"),
        "trt_fp16_build_s": cells["trt_fp16"].get("engine_build_s"),
        "vram_eager": cells["eager"].get("vram_mb_peak"),
        "vram_compile": cells["compile"].get("vram_mb_peak"),
        "vram_trt_fp32": cells["trt_fp32"].get("vram_mb_peak"),
        "vram_trt_fp16": cells["trt_fp16"].get("vram_mb_peak"),
        "util_eager": cells["eager"].get("gpu_util_mean"),
        "util_compile": cells["compile"].get("gpu_util_mean"),
        "util_trt_fp32": cells["trt_fp32"].get("gpu_util_mean"),
        "util_trt_fp16": cells["trt_fp16"].get("gpu_util_mean"),
        "util_peak_eager": cells["eager"].get("gpu_util_peak"),
        "util_peak_trt_fp16": cells["trt_fp16"].get("gpu_util_peak"),
        "_cells": cells,
    }


def main() -> None:
    rows = [collect_row(v, d) for v, dims in VARIANT_DIMS.items() for d in dims]

    # ---- CSV (wide, one row per config) ----
    csv_cols = [
        "variant", "timm_name", "hidden_dim",
        "backbone_params_m", "head_params_m", "total_params_m",
        "eager_fps", "compile_fp16_fps", "compile_time_s",
        "trt_fp32_fps", "trt_fp16_fps", "trt_fp32_build_s", "trt_fp16_build_s",
        "vram_eager", "vram_compile", "vram_trt_fp32", "vram_trt_fp16",
        "util_eager", "util_compile", "util_trt_fp32", "util_trt_fp16",
        "util_peak_eager", "util_peak_trt_fp16",
    ]
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(csv_cols)
        for r in rows:
            out = []
            for c in csv_cols:
                v = r.get(c)
                if c.endswith("_params_m") and isinstance(v, (int, float)):
                    v = round(v / 1e6, 3)
                out.append("" if v is None else v)
            w.writerow(out)

    # ---- Markdown ----
    L = []
    L.append("# Mayaku UniQuery — Inference FPS / Resource Benchmark\n")
    L.append("ConvNeXt (V1-block) backbone → FPN → UniQuery head (QGN, **2 stages**). "
             "Detector `export_forward` graph (image → boxes/scores/labels), **batch 1, "
             "640×640 letterbox canvas**. Random-initialised weights (FPS/VRAM/params are "
             "weight-independent).\n")
    L.append(f"**Hardware:** {env_line()}.\n")
    L.append("**Protocol:** 100 warm-up + 1000 timed iterations per cell (CUDA-synchronised). "
             "GPU utilization & VRAM sampled with `nvidia-smi` over a separate **~8 s sustained "
             "inference window** (util = device utilization mean; VRAM = this process's peak GPU "
             "memory incl. CUDA context).\n")

    L.append("## Headline\n")
    head = ["Variant", "hidden&#8203;_dim", "Total params (M)",
            "Eager FPS", "Compile fp16 FPS", "TRT fp32 FPS", "TRT fp16 FPS",
            "Compile time (s)", "VRAM (MB)¹", "GPU util %¹"]
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "|".join(["---"] * len(head)) + "|")
    for r in rows:
        vram = r["vram_trt_fp16"] if r["vram_trt_fp16"] is not None else r["vram_eager"]
        util = r["util_trt_fp16"] if r["util_trt_fp16"] is not None else r["util_eager"]
        L.append("| " + " | ".join([
            f"**{r['variant']}**", str(r["hidden_dim"]), m(r["total_params_m"]),
            f1(r["eager_fps"]), f1(r["compile_fp16_fps"]),
            f1(r["trt_fp32_fps"]), f1(r["trt_fp16_fps"]),
            f1(r["compile_time_s"]), f1(vram), f1(util),
        ]) + " |")
    L.append("\n¹ VRAM and GPU util in the headline are from the **TensorRT fp16** "
             "(deployment) run; per-backend values are in the Detail table.\n")

    L.append("## Detail — parameters, throughput, build/compile times\n")
    head2 = ["Variant", "hidden_dim",
             "Backbone params (M)", "Head params (M)", "Total params (M)",
             "Eager FPS", "Compile fp16 FPS", "Compile time (s)",
             "TRT fp32 FPS", "TRT fp16 FPS", "TRT fp32 build (s)", "TRT fp16 build (s)"]
    L.append("| " + " | ".join(head2) + " |")
    L.append("|" + "|".join(["---"] * len(head2)) + "|")
    for r in rows:
        L.append("| " + " | ".join([
            r["variant"], str(r["hidden_dim"]),
            m(r["backbone_params_m"]), m(r["head_params_m"]), m(r["total_params_m"]),
            f1(r["eager_fps"]), f1(r["compile_fp16_fps"]), f1(r["compile_time_s"]),
            f1(r["trt_fp32_fps"]), f1(r["trt_fp16_fps"]),
            f1(r["trt_fp32_build_s"]), f1(r["trt_fp16_build_s"]),
        ]) + " |")

    L.append("\n## Detail — VRAM (MB) & GPU util (%) per backend\n")
    L.append("VRAM = peak process GPU memory; util = mean device utilization over the "
             "8 s sustained window.\n")
    head3 = ["Variant", "hidden_dim",
             "VRAM eager", "VRAM compile", "VRAM trt fp32", "VRAM trt fp16",
             "Util eager", "Util compile", "Util trt fp32", "Util trt fp16"]
    L.append("| " + " | ".join(head3) + " |")
    L.append("|" + "|".join(["---"] * len(head3)) + "|")
    for r in rows:
        L.append("| " + " | ".join([
            r["variant"], str(r["hidden_dim"]),
            f1(r["vram_eager"]), f1(r["vram_compile"]),
            f1(r["vram_trt_fp32"]), f1(r["vram_trt_fp16"]),
            f1(r["util_eager"]), f1(r["util_compile"]),
            f1(r["util_trt_fp32"]), f1(r["util_trt_fp16"]),
        ]) + " |")

    fails = []
    for r in rows:
        for b, c in r["_cells"].items():
            if c and not c.get("ok"):
                fails.append(f"- `{r['timm_name']}` hd{r['hidden_dim']} **{b}**: "
                             f"{c.get('error', 'unknown')}")
    if fails:
        L.append("\n## Failed cells\n")
        L.extend(fails)

    L.append("\n## Notes on precision\n")
    L.append("- **Eager** — fp32, TF32 left at PyTorch default.")
    L.append("- **Compile fp16** — `model.half()` + `torch.compile` (Inductor). "
             "Compile time = wall-clock of the first (compiling) call; FPS measured after.")
    L.append("- **TRT fp32** — ONNX(fp32) → TensorRT engine with the **TF32 flag cleared** "
             "(true fp32, not TF32).")
    L.append("- **TRT fp16 (pure)** — `model.half()` ONNX → **STRONGLY_TYPED** engine. "
             "Strong typing forces TensorRT to obey the graph dtypes exactly: no fp32 weight "
             "copies and **no per-layer precision autotuning** (the hybrid `FP16` builder-flag "
             "behaviour is *not* used). Every conv / matmul / attention runs in fp16. The "
             "`boxes` output stays fp32 by model design — fp16 would quantise 0–640 px box "
             "coordinates to ~0.25 px; `scores`/`image` are fp16 (verified via engine IO dtypes).")
    L.append("\n_Generated from per-cell JSON in `benchmarks/fps_sweep/runs/results/`._")

    OUT_MD.write_text("\n".join(L) + "\n")
    done = sum(1 for r in rows for b in BACKENDS if r["_cells"].get(b, {}).get("ok"))
    print(f"Wrote {OUT_MD} and {OUT_CSV} — {done}/{len(rows) * 4} cells ok.")


if __name__ == "__main__":
    main()
