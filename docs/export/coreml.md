# CoreML export

Best-effort export to Apple's `mlprogram` format for on-device inference
on Apple Silicon. Same Graph A body as ONNX (backbone + FPN producing
`{p2..p6}`); per-image NMS / mask paste / keypoint decode stay Python
(or, in production, Swift via vImage / Accelerate).

## Install

```bash
pip install -e ".[dev,coreml]"
```

The `[coreml]` extra pulls `coremltools>=7.2`. **Conversion** runs on
any platform `coremltools` installs on, but **parity check** requires
macOS — `coremltools.models.MLModel.predict` loads
`Core ML.framework`, which only exists on Apple platforms. On non-mac
hosts the parity check returns a "skipped" `ParityResult` rather than
failing.

## What's in the exported graph

Identical to [ONNX](onnx.md):

| Tensor | Shape | Notes |
|---|---|---|
| `image` (input) | `(1, 3, H, W)` float32 | RGB, mean/std-normalised. ADR 002 — Apple Vision expects RGB so no swap. |
| `p2..p6` | per-FPN-level | Same strides as ONNX. |

Pinned details:
- **`mlprogram` format**, not `neuralnetwork`. The `mlprogram` runtime
  is the modern path with proper fp32 / fp16 control.
- **`--coreml-precision`** controls `compute_precision`:

  | flag value | enum | parity-check tolerance | NE-eligible |
  |---|---|---|---|
  | `fp32` (default) | `ct.precision.FLOAT32` | tight (`atol=1e-3`) on random init | **no** |
  | `fp16` | `ct.precision.FLOAT16` | loose (`atol=1e-2`+) on random init | yes |

  The default keeps the test-suite `parity_check` tight against the
  random-init untrained backbones the unit tests use. **Pass `fp16`
  for any real deployment** — on the UniQuery detector it is faster
  on `CPU_AND_GPU` too (18 ms vs 20 ms at 640²), independently of the
  Neural Engine, which we do not target (see below).
- **`compute_units` is an export-time no-op.** coremltools writes
  nothing about it into the `.mlpackage`, and `MLModel(path)` defaults
  to `ALL` regardless. The exporter argument only reaches
  `ExportResult.extras`, so treat it as a note-to-self, not a setting.
  **Where the graph runs is decided when it is loaded.**

  `ArtifactPredictor` therefore loads with `CPU_AND_GPU`
  (`CPU_ONLY` for `device="cpu"`), and `parity_check` pins `CPU_ONLY`
  for reproducibility. If you load an `.mlpackage` yourself, pass
  `compute_units` — the default you get otherwise is the slow one.

  **Don't pick `ALL`.** When a model has ops the NE can't run natively,
  CoreML thrashes trying to route there and `ALL` ends up *slower* than
  `CPU_AND_GPU` — sometimes by a lot. Two graphs, standalone per-image
  cost:

  | load compute_units | R50-FPN body, 1344², fp16 | UniQuery det, 640², fp32 | UniQuery det, 640², fp16 |
  |---|---|---|---|
  | `CPU_ONLY` | 229 ms | 77 ms | 47 ms |
  | **`CPU_AND_GPU`** | **85 ms** | 19 ms | **18 ms** |
  | `ALL` | 463 ms | 19 ms | 34 ms |

  `CPU_AND_GPU` is fastest or tied everywhere; `ALL` is far slower on
  two of the three. (The fp32 column is a tie — fp32 is not NE-eligible,
  so `ALL` has nowhere else to route.)

  Always benchmark against your specific graph before shipping. See
  ADR 003 §1c.1 for the val2017 throughput measurement.
- **Static input shape.** Dynamic shapes via `coremltools.RangeDim`
  interact poorly with the constant folding the converter does for
  FPN's stride-2 ops; we trace at the deployment size and leave dynamic
  shapes to ONNX/OpenVINO.

## Export

CLI:

```bash
mayaku export coreml configs/faster_rcnn_R50_FPN_3x.yaml \
    --weights model.pth --output model.mlpackage \
    --sample-height 800 --sample-width 1333
```

The output is a directory (`.mlpackage` is Apple's bundle format).

Python:

```python
from mayaku.inference.export import CoreMLExporter

result = CoreMLExporter(compute_precision="fp16").export(
    model, sample, Path("model.mlpackage")
)
```

## Parity

On macOS:

```python
parity = CoreMLExporter().parity_check(model, "model.mlpackage", sample,
                                       atol=1e-3, rtol=1e-3)
assert parity.passed, parity.per_output
```

On non-macOS hosts the parity check returns a `ParityResult` with
`passed=True, max_abs_error=0.0` and an `extras` note explaining that
the runtime isn't available. CI on Linux therefore exercises *export*
but not *parity*; the macOS leg of the test matrix is what proves
parity.

## Runtime usage (Swift / Python)

Python (macOS only):

```python
import coremltools as ct
import numpy as np

mlmodel = ct.models.MLModel("model.mlpackage",
                            compute_units=ct.ComputeUnit.CPU_AND_GPU)
img = ((rgb_uint8.astype(np.float32) - mean) / std).transpose(2, 0, 1)[None]
out = mlmodel.predict({"image": img})
features = {name: out[name] for name in ("p2","p3","p4","p5","p6")}
```

Swift: open the `.mlpackage` in Xcode to generate the typed Swift
wrapper, then call `model.prediction(image: cvPixelBuffer)`. Pair the
graph output with a Swift port of `mayaku.inference.postprocess` (NMS
via Accelerate / Metal Performance Shaders, mask paste via vImage).

## Known limitations

- **Conversion still requires `torch.jit.trace`.** PyTorch warns that
  JIT will be removed eventually; we filter the warning in
  `pyproject.toml`. Revisit when `coremltools` supports
  `torch.export`-based traces.
- **No dynamic input shapes** in v1 (see above).
- **Random-init parity is what's tested.** If you see drift > `atol`
  on a *trained* model the backbone weights are likely fp16-quantised
  somewhere in your re-export pipeline; verify with
  `compute_precision=FLOAT32`.
