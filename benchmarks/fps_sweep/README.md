# FPS / resource sweep

Benchmarks the UniQuery detector (ConvNeXt V1-block backbone → FPN → 2-stage
UniQuery head) across backbone variants × `hidden_dim` × inference backend, and
writes `fps.md` + `fps.csv` at the repo root.

## Matrix
- **Variants** (torchvision ConvNeXt, V1 blocks): atto, femto, pico, nano, tiny, base
- **hidden_dim** (= `fpn.out_channels` = head dim): 128/192/256 for all; **96** also for atto & femto
- **Backends**: `eager` (fp32) · `compile` (fp16 `torch.compile`) · `trt_fp32` · `trt_fp16` (pure)
- Fixed: 2 stages, batch 1, 640×640 letterbox, random init (FPS/VRAM/params are weight-independent)

## Run
```bash
# from anywhere; uses the active python env (needs the mayaku package + tensorrt + onnx + onnxruntime)
python benchmarks/fps_sweep/run_sweep.py     # ~1–1.5 h; resumable (skips cached ok cells)
python benchmarks/fps_sweep/assemble.py      # writes fps.md + fps.csv
```
Per-cell JSON lands in `benchmarks/fps_sweep/runs/results/`. Re-running `run_sweep.py`
skips cells already marked `ok`, so Ctrl-C is safe. To force a fresh run, delete
`runs/results/`.

Single cell (debug):
```bash
python benchmarks/fps_sweep/bench_one.py \
  --variant convnext_base --hidden-dim 256 --backend trt_fp16 \
  --out /tmp/cell.json --workdir /tmp/work
```

## Protocol
- 100 warm-up + 1000 CUDA-synchronised timed iterations per cell.
- GPU util + VRAM sampled with `nvidia-smi` over a separate ~8 s sustained window
  (util = device-utilization mean; VRAM = this process's peak GPU memory).
- Each cell runs in its own subprocess for VRAM / compile-state isolation; a 900 s
  per-cell timeout prevents a hung build from stalling the sweep.
- `trt_fp16` is **pure** fp16: `model.half()` ONNX → `STRONGLY_TYPED` engine (TRT
  obeys graph dtypes; no fp32 weight copies, no per-layer precision autotuning).

## Notes for a different GPU (e.g. RTX 3090)
- Nothing is hardcoded to the 3060: paths derive from the script location and the
  hardware line in `fps.md` is auto-detected from `nvidia-smi`/torch/tensorrt.
- The 3090 has 24 GB, so larger engines build with more headroom; absolute FPS will
  rise (relative ordering should hold). Expect the full sweep to finish faster.
- Multi-GPU host: set `CUDA_VISIBLE_DEVICES=N` to pick the device. The util sampler
  reads `nvidia-smi` index 0 by default — override with `FPS_SWEEP_GPU=N` so the
  utilization reading matches the GPU torch is actually using.
