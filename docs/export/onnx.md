# ONNX export

ONNX is the **required** export target. Every supported backbone +
detector configuration must round-trip through ONNX with parity within
`atol=1e-3` against the eager forward; this is what `tests/unit/test_onnx_export.py`
enforces and what `BACKEND_PORTABILITY_REPORT.md` §5 mandates.

## Install

```bash
pip install -e ".[dev,onnx]"
```

The `[onnx]` extra pulls `onnx>=1.16` and `onnxruntime>=1.18`. ONNX
Runtime is used for the parity check and works on every host (CPU EP).

## What's in the exported graph

Just the **Graph A body**: backbone + FPN producing the five FPN feature
maps. Inputs and outputs:

| Tensor | Shape | Notes |
|---|---|---|
| `image` (input) | `(N, 3, H, W)` float32 | Already mean/std-normalised, RGB channel order (ADR 002). |
| `p2` | `(N, C, H/4,  W/4)`  | `C = cfg.model.fpn.out_channels` (256 by default). |
| `p3` | `(N, C, H/8,  W/8)`  | |
| `p4` | `(N, C, H/16, W/16)` | |
| `p5` | `(N, C, H/32, W/32)` | |
| `p6` | `(N, C, H/64, W/64)` | |

Dynamic axes are declared for batch + spatial (`N`, `H`, `W`), so a
single artefact serves any input size that's a multiple of
`backbone.size_divisibility` (32 for ResNet+FPN).

Pinned details:
- **Opset 17**. Gives `RoiAlign(coordinate_transformation_mode="half_pixel")`
  and modern `Resize` semantics, which match the eager kernels.
- **Legacy TorchScript exporter** (`dynamo=False`). The new dynamo path
  drags `onnxscript` into required deps; we defer that until it's worth
  the dependency cost.
- Pixel mean/std subtraction is **outside** the graph. Apply it in the
  caller (the model parameters are still on `model.pixel_mean` /
  `pixel_std` for reference).

## What's *not* in the exported graph

Per `BPR §5/§6`, the per-image Python glue stays out of ONNX by design:

- RPN top-k + per-class NMS
- Mask paste (28×28 → image-space)
- Keypoint sub-pixel decode

This split is what makes the ONNX export competitive with Ultralytics'
fused-NMS approach: no opset-version dance to find an `NonMaxSuppression`
that matches the per-class semantics, no model-distorting NMS surrogate.
You re-attach the postprocess in your runtime — see
`mayaku/inference/postprocess.py` for the canonical Python implementation.

## Export

CLI:

```bash
mayaku export onnx configs/faster_rcnn_R50_FPN_3x.yaml \
    --weights model.pth --output model.onnx \
    --sample-height 800 --sample-width 1333
```

Python:

```python
import torch
from mayaku.cli._factory import build_detector
from mayaku.config import load_yaml
from mayaku.inference.export import ONNXExporter

cfg = load_yaml("cfg.yaml")
model = build_detector(cfg).eval()
model.load_state_dict(
    torch.load("model.pth", map_location="cpu", weights_only=True)
)
sample = torch.zeros(1, 3, 800, 1333, dtype=torch.float32)
result = ONNXExporter(opset=17).export(model, sample, Path("model.onnx"))
print(result.input_names, result.output_names)
```

## Parity

```python
parity = ONNXExporter().parity_check(model, "model.onnx", sample,
                                     atol=1e-3, rtol=1e-3)
assert parity.passed, parity.per_output
```

The parity check runs the eager forward on whatever device `model` lives
on, runs the ONNX Runtime CPU EP on the same `sample`, and compares
per-output tensors. Default tolerance is `atol=1e-3` — tight enough to
catch real divergence, loose enough to absorb the small fp32 reduction
order differences between cuDNN/MPS and ORT's CPU kernels.

## Runtime usage (Python)

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
std  = np.array([58.395, 57.120, 57.375], dtype=np.float32)
img = (rgb_uint8.astype(np.float32) - mean) / std       # (H, W, 3)
img = img.transpose(2, 0, 1)[None]                      # (1, 3, H, W)
features = dict(zip(["p2","p3","p4","p5","p6"],
                    sess.run(None, {"image": img})))
# Re-attach NMS / mask paste / keypoint decode using
# `mayaku.inference.postprocess` (or your own equivalent).
```

For GPU runtimes use `CUDAExecutionProvider` (NVIDIA) or the
`CoreMLExecutionProvider` if you want ONNX-via-CoreML on Apple Silicon
without writing a separate `.mlpackage`.

## Running COCO eval against the exported artefact

`mayaku eval` accepts `--backbone-onnx` to swap the eager backbone+FPN
for an `.onnx` model loaded via `ONNXBackbone`, while keeping
RPN/ROI/postprocess in PyTorch. Mirrors the
[CoreML hybrid eval](coreml.md). `--onnx-providers` is a
comma-separated, ordered list of ORT execution providers.

```bash
# Export at a square shape that fits both image orientations
# after ResizeShortestEdge.
mayaku export onnx configs/faster_rcnn_R50_FPN_3x.yaml \
    --weights model.pth --output model.onnx \
    --sample-height 1344 --sample-width 1344

# Eval (Apple Silicon — CoreMLExecutionProvider is the fastest ORT
# path, falls back to CPUExecutionProvider if CoreML can't run an op).
mayaku eval configs/faster_rcnn_R50_FPN_3x.yaml \
    --weights model.pth \
    --backbone-onnx model.onnx \
    --onnx-providers CoreMLExecutionProvider,CPUExecutionProvider \
    --device mps \
    --json /path/to/instances_val2017.json \
    --images /path/to/val2017
```

For Faster R-CNN R50-FPN against the D2-converted weights the ORT
path matches eager AP (40.23 vs 40.22) at 3.0 it/s vs eager 5.7
it/s — see ADR 003 §1d for the full benchmark and a comparison
against the native CoreML path. On Apple Silicon, **native CoreML
is faster than going through ORT's CoreML EP** (5.2 vs 3.0 it/s),
so prefer `--backbone-mlpackage` for macOS deployment and reserve
the ONNX path for cross-platform / CUDA hosts.
