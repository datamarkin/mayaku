#!/usr/bin/env python3
"""Benchmark ONE (variant, hidden_dim, backend) cell of the FPS sweep.

Backends:
  eager      fp32 PyTorch eager, export_forward graph
  compile    fp16 torch.compile (model.half + inductor)
  trt_fp32   TensorRT fp32 engine (TF32 cleared -> true fp32), from fp32 ONNX
  trt_fp16   TensorRT PURE fp16 engine: all-fp16 ONNX + STRONGLY_TYPED network
             (precision fixed by ONNX dtypes -> no fp32 fallback / no hybrid)

All four time the SAME graph: UniQuery.export_forward (image -> boxes/scores/labels),
batch 1, 640x640, 100 warmup + 1000 measured iterations. VRAM is the per-PID GPU
memory (nvidia-smi compute-apps) peak during an ~8s sustained run; GPU util is the
device utilization sampled over that same sustained window.

Writes a single JSON result file. Designed to be launched as an isolated subprocess
per cell so a failure (e.g. an fp16-unsupported layer) never poisons other cells.

Run via run_sweep.py for the full matrix, or standalone:
  python bench_one.py --variant convnext_atto --hidden-dim 128 \
      --backend trt_fp16 --out cell.json --workdir /tmp/work
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # benchmarks/fps_sweep -> repo root
BASE_YAML = REPO / "configs/detection/mayaku-n.yaml"
H = W = 640
N_WARM = 100
N_ITER = 1000
SUSTAIN_SECONDS = 8.0
WORKSPACE = 1 << 30  # 1 GiB TRT workspace
GPU_INDEX = int(os.environ.get("FPS_SWEEP_GPU", "0"))  # nvidia-smi index for util


# --------------------------------------------------------------------------- #
# GPU sampling
# --------------------------------------------------------------------------- #
def _smi(args: list[str]) -> str:
    return subprocess.run(
        ["nvidia-smi", *args], capture_output=True, text=True
    ).stdout


def pid_gpu_mem_mb(pid: int) -> float:
    out = _smi(
        ["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    )
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(pid):
            try:
                return float(parts[1])
            except ValueError:
                pass
    return 0.0


class GpuSampler(threading.Thread):
    """Polls device utilization and our PID's GPU memory until stopped."""

    def __init__(self, pid: int, interval: float = 0.15) -> None:
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self._stop_flag = False
        self.util: list[float] = []
        self.mem: list[float] = []

    def run(self) -> None:
        while not self._stop_flag:
            u = _smi(
                ["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                 "-i", str(GPU_INDEX)]
            ).strip().splitlines()
            if u:
                try:
                    self.util.append(float(u[0].strip()))
                except ValueError:
                    pass
            m = pid_gpu_mem_mb(self.pid)
            if m:
                self.mem.append(m)
            time.sleep(self.interval)

    def stop(self) -> dict:
        self._stop_flag = True
        self.join(timeout=2.0)
        return {
            "gpu_util_mean": round(sum(self.util) / len(self.util), 1) if self.util else None,
            "gpu_util_peak": round(max(self.util), 1) if self.util else None,
            "vram_mb_peak": round(max(self.mem), 1) if self.mem else None,
            "util_samples": len(self.util),
        }


# --------------------------------------------------------------------------- #
# Model build
# --------------------------------------------------------------------------- #
def build_model(variant: str, hidden_dim: int):
    import yaml
    from mayaku.config import load_yaml
    from mayaku.cli._factory import build_detector

    cfg_dict = yaml.safe_load(BASE_YAML.read_text())
    cfg_dict["model"]["backbone"]["name"] = variant
    cfg_dict["model"]["backbone"]["weights_path"] = None
    cfg_dict["model"]["fpn"]["out_channels"] = hidden_dim
    cfg_dict["model"]["uniquery_head"]["hidden_dim"] = hidden_dim
    cfg_dict["model"]["uniquery_head"]["num_stages"] = 2
    cfg_dict["input"]["size_budget"] = H

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg_dict, fh)
        tmp = fh.name
    try:
        cfg = load_yaml(tmp)
    finally:
        os.unlink(tmp)
    model = build_detector(cfg).eval()
    return model


def param_breakdown(model) -> dict:
    from mayaku.models.backbones.convnext import ConvNeXtBackbone

    backbones = [m for m in model.modules() if isinstance(m, ConvNeXtBackbone)]
    bb = sum(p.numel() for p in backbones[0].parameters()) if backbones else 0
    total = sum(p.numel() for p in model.parameters())
    return {
        "backbone_params": bb,
        "head_params": total - bb,  # FPN neck + UniQuery head
        "total_params": total,
    }


# --------------------------------------------------------------------------- #
# Torch backends (eager / compile)
# --------------------------------------------------------------------------- #
def time_torch(fwd, sample, n_warm: int, n_iter: int) -> float:
    import torch

    for _ in range(n_warm):
        fwd(sample)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fwd(sample)
    torch.cuda.synchronize()
    return n_iter / (time.perf_counter() - t0)


def sustained_torch(fwd, sample, seconds: float) -> None:
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    i = 0
    while time.perf_counter() - t0 < seconds:
        fwd(sample)
        i += 1
        if i % 64 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()


def run_torch(variant: str, hidden_dim: int, backend: str, result: dict) -> None:
    import torch
    from mayaku.inference.export.full_detector import FullDetectorAdapter

    torch.backends.cudnn.benchmark = True
    model = build_model(variant, hidden_dim)
    result.update(param_breakdown(model))

    half = backend == "compile"
    if half:
        model = model.half()
    model = model.cuda().eval()
    adapter = FullDetectorAdapter(model).eval()

    dtype = torch.float16 if half else torch.float32
    sample = torch.randn(1, 3, H, W, device="cuda", dtype=dtype)

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        if backend == "compile":
            compiled = torch.compile(adapter)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            compiled(sample)  # first call triggers compilation
            torch.cuda.synchronize()
            result["compile_time_s"] = round(time.perf_counter() - t0, 2)
            fwd = compiled
        else:
            fwd = adapter

        fps = time_torch(fwd, sample, N_WARM, N_ITER)
        result["fps"] = round(fps, 1)

        sampler = GpuSampler(os.getpid())
        sampler.start()
        sustained_torch(fwd, sample, SUSTAIN_SECONDS)
        result.update(sampler.stop())

    result["torch_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)


# --------------------------------------------------------------------------- #
# TensorRT backends
# --------------------------------------------------------------------------- #
def export_onnx(model, sample, onnx_path: Path) -> None:
    from mayaku.inference.export.onnx import ONNXExporter

    ONNXExporter().export(model, sample, onnx_path)


def build_engine(onnx_path: Path, engine_path: Path, strongly_typed: bool) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if strongly_typed:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    data = onnx_path.read_bytes()
    if not parser.parse(data):
        msgs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{msgs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE)
    if not strongly_typed:
        # fp32 engine: TF32 is on by default in TRT10 -> clear for true fp32.
        config.clear_flag(trt.BuilderFlag.TF32)
    # Static single-shape optimization profile (input is fixed 1x3xHxW).
    inp = network.get_input(0)
    shape = tuple(inp.shape)
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, min=shape, opt=shape, max=shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None (see TRT log)")
    engine_path.write_bytes(bytes(serialized))


def engine_layer_precisions(engine_path: Path) -> dict:
    """Inspect the built engine and tally per-layer precisions (best-effort)."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    try:
        inspector = engine.create_engine_inspector()
        info = json.loads(
            inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        )
        tally: dict[str, int] = {}
        for layer in info.get("Layers", []):
            prec = layer.get("Precision", "UNKNOWN") if isinstance(layer, dict) else "UNKNOWN"
            tally[prec] = tally.get(prec, 0) + 1
        return tally
    except Exception as exc:  # inspector is best-effort
        return {"inspect_error": str(exc)}


class TRTRunner:
    def __init__(self, engine_path: Path) -> None:
        import tensorrt as trt
        import torch

        self.trt = trt
        self.torch = torch
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        t2t = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.bool: torch.bool,
        }
        self.buffers = {}
        self.io_dtypes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dt = t2t[self.engine.get_tensor_dtype(name)]
            shape = tuple(self.context.get_tensor_shape(name))
            buf = torch.zeros(shape, dtype=dt, device="cuda")
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                buf.normal_()
            self.buffers[name] = buf
            self.io_dtypes[name] = str(dt)
            self.context.set_tensor_address(name, buf.data_ptr())

    def _exec(self) -> None:
        self.context.execute_async_v3(self.stream.cuda_stream)

    def bench(self, n_warm: int, n_iter: int) -> float:
        for _ in range(n_warm):
            self._exec()
        self.stream.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            self._exec()
        self.stream.synchronize()
        return n_iter / (time.perf_counter() - t0)

    def sustained(self, seconds: float) -> None:
        t0 = time.perf_counter()
        i = 0
        while time.perf_counter() - t0 < seconds:
            self._exec()
            i += 1
            if i % 64 == 0:
                self.stream.synchronize()
        self.stream.synchronize()


def run_trt(variant: str, hidden_dim: int, backend: str, result: dict, workdir: Path) -> None:
    import torch

    fp16 = backend == "trt_fp16"
    model = build_model(variant, hidden_dim)
    result.update(param_breakdown(model))

    # Pure-fp16 path: half() the whole model so the traced ONNX carries fp16
    # weights+activations, then build a STRONGLY_TYPED engine. Strong typing
    # means TRT obeys the graph dtypes exactly -- no fp32 weight fallback and
    # no per-layer precision autotuning (the hybrid FP16-flag behaviour we are
    # explicitly avoiding). The model's box-coordinate decode tail stays fp32
    # by design (fp16 would quantise 0-640 px boxes to ~0.25 px), so the
    # `boxes` output is fp32; all conv/matmul/attention compute is fp16.
    if fp16:
        model = model.half()
    model = model.cuda().eval()
    in_dtype = torch.float16 if fp16 else torch.float32
    sample = torch.randn(1, 3, H, W, device="cuda", dtype=in_dtype)

    onnx_path = workdir / f"{variant}_{hidden_dim}_{backend}.onnx"
    engine_path = workdir / f"{variant}_{hidden_dim}_{backend}.engine"
    with torch.inference_mode():
        export_onnx(model, sample, onnx_path)

    # Free the torch model before the TRT build to leave VRAM headroom.
    del model, sample
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    build_engine(onnx_path, engine_path, strongly_typed=fp16)
    result["engine_build_s"] = round(time.perf_counter() - t0, 1)
    result["engine_precisions"] = engine_layer_precisions(engine_path)

    runner = TRTRunner(engine_path)
    result["io_dtypes"] = runner.io_dtypes
    result["fps"] = round(runner.bench(N_WARM, N_ITER), 1)

    sampler = GpuSampler(os.getpid())
    sampler.start()
    runner.sustained(SUSTAIN_SECONDS)
    result.update(sampler.stop())

    onnx_path.unlink(missing_ok=True)
    engine_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--hidden-dim", type=int, required=True)
    ap.add_argument("--backend", required=True,
                    choices=["eager", "compile", "trt_fp32", "trt_fp16"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    result = {
        "variant": args.variant,
        "hidden_dim": args.hidden_dim,
        "backend": args.backend,
        "resolution": f"{H}x{W}",
        "n_warm": N_WARM,
        "n_iter": N_ITER,
        "ok": False,
    }
    try:
        if args.backend in ("eager", "compile"):
            run_torch(args.variant, args.hidden_dim, args.backend, result)
        else:
            run_trt(args.variant, args.hidden_dim, args.backend, result,
                    Path(args.workdir))
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 - record, never crash the sweep
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[{args.backend}] {args.variant} hd{args.hidden_dim}: "
          f"{'OK fps=' + str(result.get('fps')) if result['ok'] else 'FAIL ' + result.get('error', '')}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
