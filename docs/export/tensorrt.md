# TensorRT export

Bonus deployment target on CUDA hosts. Same Graph A body as the other
targets: backbone + FPN producing `{p2..p6}`; per-image NMS / mask paste
/ keypoint decode stay Python (per `BACKEND_PORTABILITY_REPORT.md` §6:
"keep NMS out of TRT entirely").

## Install

```bash
pip install -e ".[dev,onnx,tensorrt]"
```

The `[tensorrt]` extra carries `tensorrt>=10.0` with a PEP 508 marker
that disables the dependency on macOS (`tensorrt>=10.0; sys_platform !=
'darwin'`) — NVIDIA ships no wheels for Darwin. The extra also pulls
`onnx` because the exporter routes through the ONNX front-end (see
below).

You also need:
- A CUDA-capable GPU + matching CUDA / cuDNN runtime that the installed
  TRT wheel was built against (see NVIDIA's compatibility matrix).
- TensorRT Python bindings on `PYTHONPATH` (the wheel handles this).

## Path: ONNX → TensorRT

The exporter does:

1. Build an intermediate `.onnx` via the existing
   [`ONNXExporter`](onnx.md) into a tempdir (already validated against
   eager — see `tests/unit/test_onnx_export.py`).
2. Parse it with `tensorrt.OnnxParser`.
3. Build a serialised engine via TRT's `Builder` + `IBuilderConfig` and
   write it to `out_path`.

The Python builder API is preferred over shelling out to `trtexec`
because it's deterministic across `PATH` variations, gives structured
error messages on parse failures, and lets us pin a reproducible builder
workspace size.

## Builder configuration (defaults)

| Knob | Default | Why |
|---|---|---|
| Network flags | `EXPLICIT_BATCH` | TRT 10.x requires this. |
| Workspace | 1 GiB (`workspace_bytes=1<<30`) | Enough for ResNet-50 + FPN; doesn't inflate to multi-GB. |
| FP16 | `False` | Keeps parity tight. Set `TensorRTExporter(fp16=True)` for ~2× throughput at small accuracy cost. |
| TF32 | **cleared** on the fp32 path | TRT 10.x has `BuilderFlag.TF32` set by default on Ampere+; clearing it is what makes `parity_check` agree with eager fp32. |
| Optimization profile | `min=opt=max=sample.shape` | The intermediate ONNX has dynamic batch + spatial axes; TRT requires a profile to build dynamic-shape networks. We pin a single point at the export shape. |

## Export

```bash
mayaku export tensorrt configs/faster_rcnn_R50_FPN_3x.yaml \
    --weights model.pth --output model.engine \
    --sample-height 800 --sample-width 1333
```

The output is a serialised TRT engine (a `.engine` file). Engines are
**hardware-specific** — built engines are tied to the GPU
arch / SM / driver / TRT version they were built against. Re-export per
deployment target.

Python:

```python
from mayaku.inference.export import TensorRTExporter

# Trace on the model's device — the exporter moves `sample` to model.device
# automatically.
model = build_detector(cfg).cuda().eval()
sample = torch.zeros(1, 3, 800, 1333, dtype=torch.float32)
result = TensorRTExporter(fp16=False, workspace_bytes=1<<30).export(
    model, sample, Path("model.engine")
)
```

## Parity

```python
parity = TensorRTExporter().parity_check(model, "model.engine", sample,
                                         atol=1e-2, rtol=1e-2)
assert parity.passed, parity.per_output
```

Two things to know about TRT parity:

- **The eager half is forced to strict fp32** for the duration of the
  parity check. PyTorch enables TF32 by default for cuDNN convs and
  matmuls on Ampere+; without disabling those the eager half uses TF32
  and disagrees with TRT-fp32 by ~0.25 abs. The parity check toggles
  `torch.backends.{cuda.matmul,cudnn}.allow_tf32 = False` around the
  forward and restores them on the way out.
- **Default tolerance is `atol=1e-2`** — looser than ONNX/CoreML/OpenVINO
  because TRT does kernel-selection optimisation even at fp32 and that
  introduces small drift on top of cuBLAS / cuDNN. Tighter tolerances
  are reasonable on production builds where you've validated a specific
  build's drift.

The parity check refuses CPU models (`RuntimeError: TensorRT parity
check requires a CUDA model`).

## Runtime usage

Python (TRT 10.x async API):

```python
import torch
import tensorrt as trt

logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(logger)
with open("model.engine", "rb") as fh:
    engine = runtime.deserialize_cuda_engine(fh.read())

context = engine.create_execution_context()
context.set_input_shape("image", tuple(sample.shape))

# Allocate device buffers via torch.
sample_cuda = sample.cuda().contiguous()
outputs = {n: torch.empty(tuple(context.get_tensor_shape(n)),
                          dtype=torch.float32, device="cuda")
           for n in ("p2","p3","p4","p5","p6")}

context.set_tensor_address("image", int(sample_cuda.data_ptr()))
for name, buf in outputs.items():
    context.set_tensor_address(name, int(buf.data_ptr()))

stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    context.execute_async_v3(stream_handle=stream.cuda_stream)
stream.synchronize()
# `outputs` now holds {p2..p6}; re-attach Python postprocess.
```

C++ runtime: same shape via `nvinfer1::IExecutionContext::executeV3` —
see NVIDIA's TRT samples.

## Tests

`tests/unit/test_tensorrt_export.py` is gated by `pytest.mark.tensorrt`
+ `pytest.mark.slow` — engine builds take a few seconds each, so opt
out via `-m 'not slow'` on busy CI. The marker auto-skips when either
`MAYAKU_DEVICE != cuda` or `import tensorrt` fails, so macOS / CPU-only
Linux see the 5 tests cleanly skipped.
