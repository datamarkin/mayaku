# OpenVINO export

Best-effort export to Intel's OpenVINO IR (`.xml` + `.bin`) for CPU and
Intel iGPU / dGPU deployment. Same Graph A body as the other targets:
backbone + FPN producing `{p2..p6}`; per-image NMS / mask paste /
keypoint decode stay Python.

## Install

```bash
pip install -e ".[dev,openvino]"
```

The `[openvino]` extra pulls `openvino>=2024.3`. Conversion **and** CPU
inference both work on Linux, macOS, and Windows — no platform split.

## What's in the exported graph

| Tensor | Shape | Notes |
|---|---|---|
| `image` (input) | `(N, 3, H, W)` float32 | RGB, mean/std-normalised. |
| `p2..p6` | per-FPN-level | Output ports tagged with the FPN names so the runtime can index by name. |

Pinned details:
- **Direct PyTorch conversion** via `openvino.convert_model(model,
  example_input=...)`. No intermediate ONNX hop; the converter walks
  the `nn.Module` directly.
- **`compress_to_fp16=False` by default.** OpenVINO's default writes
  the IR `.bin` at fp16 (half the disk size); we keep it fp32 for
  parity-check tightness. Re-export with
  `OpenVINOExporter(compress_to_fp16=True)` to get the smaller artefact
  for production deployments that already accept the fp16 drift.
- **`INFERENCE_PRECISION_HINT="f32"` on the CPU plugin.** The OpenVINO
  CPU plugin defaults to **fp16 inference precision** even when the IR
  is fp32, which produces ~3 abs error vs eager on the random-init
  untrained backbones the test suite uses. Pinning fp32 inference is
  what makes `parity_check` tight (`atol=1e-3`); production
  deployments that want the speed/size benefits inherently accept that
  drift.
- **Output ports indexed by name.** `compiled([sample])` returns a
  dict keyed by `ConstOutput` port objects whose iteration order isn't
  guaranteed; the parity check uses `port.get_names()` to look up
  outputs by the friendly name we tagged at export time. A re-ordering
  inside the converter surfaces as
  `RuntimeError("missing expected outputs: ...")` instead of silently
  comparing the wrong tensors.

## Export

CLI:

```bash
mayaku export openvino model_final.pth --output model.xml
```

> **Input size is read from the checkpoint — you do not pass one.** The `.pth`'s
> embedded sidecar carries the deploy canvas training resolved (`input.canvas_hw`),
> and the export traces the graph at exactly that geometry, rectangle included.
>
> `--sample-height` / `--sample-width` exist only for backbone-only R-CNN graphs,
> which export with dynamic spatial axes — there the traced size is an arbitrary
> tracing detail. A full-detector graph (the `mayaku-*` family) bakes its input
> size into the box decode, so for those any size other than the deploy canvas is
> refused rather than silently shipped.

OpenVINO writes a sibling `.bin` next to the `.xml`; both files are part
of the artefact.

Python:

```python
from mayaku.inference.export import OpenVINOExporter

result = OpenVINOExporter(compress_to_fp16=False).export(
    model, sample, Path("model.xml")
)
print(result.extras)   # includes the bin path and compress_to_fp16 flag
```

## Parity

```python
parity = OpenVINOExporter().parity_check(model, "model.xml", sample,
                                         atol=1e-3, rtol=1e-3)
assert parity.passed, parity.per_output
```

The check always runs on the **CPU device** with
`INFERENCE_PRECISION_HINT=f32` so it's reproducible everywhere. The
eager forward happens on whatever device `model` lives on; the comparison
is in CPU-space.

## Runtime usage

Python:

```python
import numpy as np
import openvino as ov

core = ov.Core()
ov_model = core.read_model("model.xml")
compiled = core.compile_model(ov_model, device_name="CPU")  # or "GPU"
img = ((rgb_uint8.astype(np.float32) - mean) / std).transpose(2, 0, 1)[None]
out = compiled([img])
features = {name: out[port][0]
            for port, _ in zip(out, ("p2","p3","p4","p5","p6"), strict=True)
            for name in port.get_names() if name in {"p2","p3","p4","p5","p6"}}
```

C++ / Java / .NET runtimes: see the [OpenVINO API
docs](https://docs.openvino.ai/) for the per-language `Core` /
`CompiledModel` / `InferRequest` shape — the IR file is the same.

## Notes

- **CPU is the lowest-common-denominator target** the parity check
  uses; you can compile against `GPU` (Intel iGPU/dGPU) or `NPU`
  for deployment without re-exporting.
- **No model-distorting changes** were needed to make OpenVINO work on
  the in-scope detectors — the exporter is a thin wrapper around
  `convert_model`.
