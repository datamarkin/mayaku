# Portability

Mayaku trains and runs on three backends — CUDA, MPS, and CPU — from a
single codebase. This document describes the guarantees, the abstractions
that make them hold, and the per-backend caveats.

## Backend matrix

| | CUDA | MPS | CPU |
|---|---|---|---|
| Forward / backward | ✅ | ✅ | ✅ |
| AMP (`fp16` autocast) | ✅ | ⚠️ pass-through | ❌ no autocast |
| Distributed (DDP) | ✅ `nccl` | ✅ `gloo` | ✅ `gloo` |
| `torch.distributed.barrier` | ✅ | ✅ via gloo | ✅ via gloo |
| `roi_align` | torchvision CUDA | torchvision MPS, fallback OK | torchvision CPU |
| `nms` / `batched_nms` | torchvision CUDA | torchvision MPS, fallback OK | torchvision CPU |
| ONNX export | ✅ | ✅ (CPU trace) | ✅ |
| CoreML export | n/a host | ✅ macOS only | ✅ macOS only |
| OpenVINO export | ✅ | ✅ | ✅ |
| TensorRT export | ✅ | ❌ (no NVIDIA wheels for macOS) | ❌ (CUDA-only runtime) |

✅ = required to be green; ⚠️ = supported but with caveats below.

## The `Device` abstraction

`mayaku.backends.device.Device` is the single thing every backend-aware
code path goes through. Use it instead of touching `torch.cuda` or
`torch.backends.mps` directly:

```python
from mayaku.backends import Device

dev = Device.auto()              # CUDA → MPS → CPU
dev = Device(kind="cuda", index=0)   # explicit pin
model = model.to(dev.torch)
```

Public surface:

| Attribute | What it gives you |
|---|---|
| `kind` | `"cuda"` \| `"mps"` \| `"cpu"` |
| `torch` | A `torch.device` for `.to(...)` |
| `supports_amp` | Whether `mayaku.backends.amp.autocast` is meaningful |
| `amp_dtype` | `torch.float16` on CUDA/MPS, `None` on CPU |
| `dist_backend` | `"nccl"` on CUDA, `"gloo"` elsewhere |
| `supports_pin_memory` | True only on CUDA |
| `synchronize()` | Dispatches to `torch.{cuda,mps}.synchronize`; no-op on CPU |

`Device.auto()` is what you want in user scripts. The CLI passes
`--device cpu/mps/cuda` to pin a specific backend; if omitted it falls back
to `Device.auto()`.

## Op fallbacks

`mayaku.backends.ops` re-exports three operators every detector hits on
the hot path:

- `roi_align(features, boxes, output_size, spatial_scale, sampling_ratio,
  aligned)`
- `nms(boxes, scores, iou_threshold)`
- `batched_nms(boxes, scores, idxs, iou_threshold)`

Each tries the torchvision kernel first and falls back to a pure-PyTorch
implementation only on `NotImplementedError`. On the version pins listed
in `BACKEND_PORTABILITY_REPORT.md` Appendix A (`torch>=2.4`,
`torchvision>=0.19`; the dev hosts are torch 2.11 / torchvision 0.26) the
fallback never fires in production, but it's directly exercised by
`test_*_fallback*` to keep it honest. If torchvision ever drops a kernel
for a backend we still need, the fallback keeps the build green; flag the
gap by opening an entry in [`MPS_ISSUES_DISCOVERED.md`](../MPS_ISSUES_DISCOVERED.md).

Deformable convolution is **not** in `backends.ops` — see
[ADR 001](decisions/001-drop-deformable-convolution.md).

## AMP

`mayaku.backends.amp.autocast(device, enabled, dtype)` is a context
manager that does the right thing for each backend:

- **CUDA**: `torch.autocast("cuda", dtype=fp16/bf16)` plus
  `GradScaler` in the training loop.
- **CPU**: `torch.autocast("cpu", dtype=bf16)` is supported (and is what
  PyTorch recommends), but Mayaku's `Device.amp_dtype` returns `None` for
  CPU because the speed-up isn't worth the code path complexity for the
  in-scope detectors. CPU AMP is *not* in the public API.
- **MPS**: fp32 by default (the autocast block is entered with
  `dtype=torch.float32`, so it's effectively a no-op). Opt into fp16
  by setting `solver.amp_dtype: float16` if you've validated numerics
  on your dataset — R-CNN's box-reg + mask losses are fp16-sensitive
  on every backend, and MPS fp16 is not as battle-tested as CUDA. The
  CLI also auto-sets `PYTORCH_ENABLE_MPS_FALLBACK=1` on `--device mps`
  so unsupported ATen ops fall back to CPU rather than raising; the
  fallback ops are deduplicated and summarised at end-of-run via the
  `mayaku.backends.mps` tracker. Set `MAYAKU_VERBOSE_MPS=1` to bypass
  the dedup and see PyTorch's native per-call warnings.

`AMPTrainer` uses this implicitly when `cfg.solver.amp_enabled` and
`device.supports_amp` are both true. User scripts rarely need to touch
`autocast` directly — call `AMPTrainer` and let it set the context.

## Distributed

`mayaku.engine.distributed.launch(main_fn, num_gpus, dist_backend=...)`
spawns one process per GPU via `torch.multiprocessing.spawn` and pins
each process to its `LOCAL_RANK` (`torch.cuda.set_device(LOCAL_RANK)`
when CUDA is the backend). The default `dist_backend` is
`device.dist_backend` — `nccl` on CUDA, `gloo` everywhere else.

Use `create_ddp_model(model, device)` to wrap the model for DDP — it
sets `device_ids=[device.index]` only on CUDA (the gloo backend does
not accept `device_ids`).

Helpers exposed at the package level:
- `is_main_process()`, `get_rank()`, `get_world_size()`, `synchronize()`
- `all_reduce_dict(d)` — element-wise all-reduce of a dict of tensors
- `all_gather_object(obj)` — picklable-object gather

The `multi_gpu` pytest marker auto-skips when fewer than 2 CUDA devices
are visible. The single-process gloo path is exercised by
`tests/unit/test_distributed.py` on every backend.

## Pytest backend selection

`MAYAKU_DEVICE={cpu,mps,cuda}` selects the active backend. Marker
auto-skips wired up in `tests/conftest.py`:

| Marker | Skips when |
|---|---|
| `cuda` | `MAYAKU_DEVICE != cuda` |
| `mps` | `MAYAKU_DEVICE != mps` |
| `multi_gpu` | `torch.cuda.device_count() < 2` |
| `onnx` | `import onnx` / `import onnxruntime` fails |
| `coreml` | `import coremltools` fails (i.e. not on macOS) |
| `openvino` | `import openvino` fails |
| `tensorrt` | not CUDA, or `import tensorrt` fails |
| `slow` | `-m 'not slow'` is passed |

`conftest.py` will `pytest.exit` with a clear message if you set
`MAYAKU_DEVICE=cuda` on a host without CUDA. Silent CPU fall-through has
caused false-green runs in past projects; the explicit refusal earns its
keep.

A separate guard verifies that `import mayaku` resolves to the editable
install in this checkout, not a stale install from a different clone
path. If you ever see "ModuleNotFoundError: mayaku.X" on a file that
exists on disk, run `pip install -e '.[dev]'` from the repo root.

## Per-backend caveats

### MPS

Living issues are tracked in
[`MPS_ISSUES_DISCOVERED.md`](../MPS_ISSUES_DISCOVERED.md). At present:

- AMP is a pass-through (see above).
- The `multi_gpu` story doesn't apply (MPS exposes one device).
- Numerical drift vs CPU/CUDA is mostly within the test tolerances we
  set; specific kernels that drifted are documented in the per-test
  comments.

### CUDA

- TF32 is enabled by default on Ampere+ for both PyTorch (cuDNN convs +
  matmuls) and TensorRT. The TRT exporter clears `BuilderFlag.TF32` on
  the default fp32 path and the parity check toggles
  `torch.backends.{cuda.matmul,cudnn}.allow_tf32 = False` around the
  eager forward — see [`docs/export/tensorrt.md`](export/tensorrt.md).
- `nccl` is required for multi-GPU training.

### CPU

- Single-process only by design (no `gloo`-on-CPU multi-process recipe
  in v1).
- `pin_memory` is unconditionally false (`Device.supports_pin_memory`).

## AMD GPUs (ROCm)

Mayaku trains and infers on AMD GPUs (Linux + ROCm) **with no source
changes** — the project ships zero custom CUDA/HIP kernels, every
backend-aware call site goes through the `Device` facade, and PyTorch's
ROCm runtime exposes itself under the `torch.cuda.*` namespace. Use
`MAYAKU_DEVICE=cuda` on an AMD host; the underlying HIP dispatch is
transparent.

### GPU support matrix

| GPU family | Examples | Status |
|---|---|---|
| **CDNA** (data-center) | MI200, MI250, MI300 | Full official ROCm support. First-class for training. |
| **RDNA3** (consumer) | RX 7900 XTX / 7900 XT / 7900 GRE (gfx1100) | Official ROCm support since 5.7. 24 GB VRAM on 7900 XTX is enough for ConvNeXt-Base 1x training. |
| **RDNA2** (consumer) | RX 6800 / 6900 / 6700 (gfx1030–1032) | Best-effort. Small/medium training works; ConvNeXt-Large likely OOMs. |
| **Older** (Polaris, Vega, RDNA1) | RX 5xx / 5xxx | Not supported by current ROCm. |

## Cross-machine test protocol

Per-step verification needs the same suite green on each backend:

```bash
ruff check src tests
mypy
MAYAKU_DEVICE=cpu  pytest -q
MAYAKU_DEVICE=mps  pytest -q   # Apple Silicon Mac
MAYAKU_DEVICE=cuda pytest -q   # CUDA Linux box
MAYAKU_DEVICE=cuda pytest -k multi_gpu -v   # if ≥ 2 CUDA devices
```

There is no CI; coverage is enforced manually via the per-step checklist
in [`PROJECT_STATUS.md`](../PROJECT_STATUS.md).
