# 005 — ONNX/TensorRT/OpenVINO: GPU-eager wins on GPU; OpenVINO genuinely beats PyTorch on Intel CPU

Status: accepted
Date: 2026-04-27

## Context

ADR 004 found that on Apple Silicon, CoreML's runtime delivers
classifier-grade speedups (6–22× over MPS for plain ResNet/MobileNet)
but lands at parity (1.05–1.46×) with MPS for the full R-CNN
backbone+FPN graph. That left the question open: does the symmetric
deployment story — ONNX/TensorRT on CUDA — show a different pattern?
On Linux+CUDA the ORT TRT EP is the headline export claim; if TRT
fp16 delivers 2–3× over PyTorch eager on R-CNN, the export pitch is
real. If it doesn't, ONNX falls into the same bucket as CoreML:
*deployment artifact, not throughput win*.

This ADR records the answer from a real Linux+CUDA box.

## Empirical results (Linux x86_64, CUDA 12.4, PyTorch 2.6, ORT 1.23.2, TensorRT 10.x)

### `benchmarks/onnx_throughput.py` — standalone graphs

**ResNet-50 ImageNet classifier @ 224×224** (control row — proves TRT
works):

| Row | ms/iter | Speedup vs PyTorch CUDA fp32 |
|---|---|---|
| PyTorch CUDA fp32 | 4.18 | 1.0× |
| PyTorch CUDA fp16 | 4.80 | 0.87× (Tensor Cores not engaging at this shape) |
| ORT CUDA EP fp32 | 1.61 | **2.60× faster** |
| ORT TRT fp32 | 1.30 | **3.21× faster** |
| **ORT TRT fp16** | **0.62** | **6.75× faster** ← classifier headline |

Confirms TRT is healthy on this hardware: 6.75× over PyTorch eager
for plain ResNet-50 is well within the published TRT marketing range.

**Mayaku R-50+FPN backbone @ 736×1344** (the realistic R-CNN graph):

| Row | ms/iter | Speedup vs PyTorch CUDA fp32 |
|---|---|---|
| PyTorch CUDA fp32 | 20.73 | 1.0× |
| PyTorch CUDA fp16 | 11.78 | **1.76× faster** (Tensor Cores engaging) |
| ORT CUDA EP fp32 | 45.25 | 0.46× *slower* |
| ORT TRT fp32 | 37.10 | 0.56× *slower* |
| ORT TRT fp16 | 31.69 | 0.65× *slower* |

Same direction every row: ORT (with or without TRT, with or without
fp16) is *slower* than PyTorch eager for our R-CNN graph. TRT fp16
is **2.69× slower than PyTorch fp16 eager** (31.69 ms vs 11.78 ms) —
the fp16 fusion that yields 6.75× on plain ResNet-50 yields nothing
on our graph.

### `benchmarks/onnx_eval_pipeline.py` — full eval pipeline

Includes RPN + ROI heads + postprocess on top of the backbone:

| Row | ms/iter | Speedup vs PyTorch CUDA fp32 |
|---|---|---|
| PyTorch CUDA fp32 (baseline) | 58.2 | 1.0× |
| ORT TRT fp32 | 76.9 | 0.76× *slower* |
| ORT TRT fp16 | 74.8 | 0.78× *slower* |
| ORT CUDA EP fp32 | 82.7 | 0.70× *slower* |
| ORT CPU EP fp32 | 336.5 | 0.17× *slower* |

The diluted backbone gap survives: TRT fp16 is 1.29× slower than
PyTorch eager end-to-end. Same conclusion as the standalone test.

### Intel CPU (Linux x86_64, ORT 1.23.0 with OpenVINO EP) — 2026-04-27

Same two scripts run on the user's Intel CPU box. PyTorch is on CPU
(no GPU used). Results from `benchmarks/onnx_throughput.py`:

**Plain ResNet-50 classifier @ 224×224:**

| Row | ms/iter | Speedup vs PyTorch CPU fp32 |
|---|---|---|
| PyTorch CPU fp32 | 21.86 | 1.0× |
| ORT OpenVINO EP fp32 | 6.74 | **3.24× faster** |
| ORT CPU EP fp32 | 7.85 | 2.78× faster |

**Mayaku R-50+FPN @ 736×1344:**

| Row | ms/iter | Speedup vs PyTorch CPU fp32 |
|---|---|---|
| PyTorch CPU fp32 | 537.79 | 1.0× |
| **ORT OpenVINO EP fp32** | **203.31** | **2.65× faster** ← R-CNN deployment win |
| ORT CPU EP fp32 | 234.69 | 2.29× faster |

**This is the first deployment-target row that genuinely beats the
framework's eager mode for our R-CNN graph.** Mac CoreML lost.
CUDA TRT lost. Intel OpenVINO wins by 2.65× on the realistic
detection workload. ORT's plain CPU EP also beats PyTorch CPU by
~2.3×, almost matching OpenVINO; the marginal benefit of OpenVINO
specifically over ORT-CPU on Intel hardware is ~15%.

**Caveat (measured on the same Intel-CPU box once it had CUDA torch
running too):** PyTorch CUDA on the same machine ran the R-CNN
graph at 21.5 ms. So OpenVINO's 203 ms (2.65× over PyTorch CPU) is
still **9.5× *slower* than CUDA eager** on identical hardware.
OpenVINO's win is conditional on the deployment target *not having
a CUDA-capable GPU*. For CPU-only deployments — embedded servers,
ARM via cross-compile, edge boxes, virtualised cloud CPU instances
— OpenVINO is the right deployment runtime. For any deployment
with a GPU available, the GPU's eager runtime wins.

## Diagnosis

We tested two distinct hypotheses for why ORT/TRT is slower than
PyTorch eager on R-CNN:

1. **Dynamic-shapes overhead (round 2)** — the original ONNX export
   used `dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}}`
   so the graph supports any input size. TRT's engine builder cannot
   pre-compile optimal kernels under dynamic shapes; it falls back
   to generic kernels. We added a `dynamic_input_shape: bool` flag
   to `ONNXExporter` (default `True`) and re-tested with fixed shape
   on the realistic Mayaku graph row.

   - On Mac CoreML EP, the fix worked: 0.59× → 1.29× (CoreML now
     beats MPS by 29% with fixed-shape).
   - **On Linux CUDA TRT EP, the fix did not change the verdict**:
     TRT fp16 went 31.50 ms (dynamic) → 31.69 ms (fixed). Within
     measurement noise.

   So dynamic shapes were a genuine confound on Mac but not the
   dominant cost on CUDA.

2. **Graph-structure incompatibility with TRT fusion** — the FPN's
   top-down `lateral_conv + upsample + add` pattern with 5 separate
   outputs (p2..p6) likely defeats TRT's preferred fusion templates,
   which work best for single-output forward chains with conv+bn+relu
   blocks. PyTorch+cuDNN handles small-op-heavy multi-output graphs
   well via per-layer algorithm selection (cuDNN heuristics) and
   doesn't need fusion to be efficient. The empirical result is
   consistent with this: TRT crushes plain ResNet (single output,
   classic forward chain) but loses on FPN.

This is structural, not fixable at the export-flag level. It would
require either:
- Restructuring the FPN to be more TRT-friendly (compromises the
  D2-faithful structure that ADR 003's parity validation rests on).
- Using a TRT plugin for the multi-output FPN add (real engineering;
  not in scope).
- Profiling with `trtexec --verbose` to identify the worst-offending
  ops and replacing them (defer).

3. **PyTorch CPU is a genuinely weak baseline** — explains why
   OpenVINO can clear the bar on Intel CPU but not on CUDA.
   PyTorch CPU has oneDNN integration, but it's not as deep as
   OpenVINO's hand-tuned Intel kernels (AVX-512 / AMX), and CPU
   has no equivalent of cuDNN's per-layer algorithm autotuning.
   So PyTorch CPU at 538 ms for the FPN graph leaves 2.65×
   headroom that OpenVINO claims; PyTorch CUDA at 58 ms leaves
   nothing because cuDNN already runs the optimal kernel. The
   asymmetric finding (OpenVINO wins on CPU, TRT loses on GPU)
   is what you'd predict from this baseline strength gap.

## Decision

The deployment-target story now splits by *whether the deployment
target has a GPU*:

- **GPU-available deployments (CUDA, Apple Silicon MPS):** PyTorch
  eager is the recommended runtime. The `.onnx` / `.trt` / CoreML
  exports are for **artifact format compatibility**, not for
  throughput. TRT and CoreML lose to PyTorch eager on R-CNN at
  these targets; the artifacts have other use cases (INT8 serving
  via TRT, iOS/macOS app deployment via CoreML).
- **CPU-only deployments (Intel servers without GPUs, edge boxes,
  virtualised cloud CPU instances):** **OpenVINO via ORT delivers
  a real 2.65× speedup over PyTorch CPU on the R-CNN graph.** This
  is a genuine throughput claim, not just artifact coverage. Intel
  CPU is the one platform measured where the deployment runtime
  beats the framework runtime for our workload.
- **The `.onnx` file is universally shipped** because it's the
  portable interchange format that bridges all of the above and
  more (Windows DirectML, Android NNAPI, ROCm, future
  accelerators). Even when ORT's direct EPs don't beat the
  framework runtime, the `.onnx` is the entry point.
- **Where TRT actually delivers**: classifier-shaped feature
  extraction with a single output. If a user wants to run just the
  backbone for embeddings/retrieval, TRT fp16 gives ~7× over
  PyTorch eager. Mayaku doesn't currently advertise this path.

Two source-level changes accept the empirical reality:

1. **`ONNXExporter` now exposes `dynamic_input_shape: bool = True`**
   (`src/mayaku/inference/export/onnx.py:65-95`). When `True`
   (default, preserves the existing behaviour), the graph supports
   any input size. When `False`, it pins the literal sample shape
   and TRT can pre-compile kernels. The flag is plumbed through
   `cli/export.py` and `cli/__main__.py` as
   `--onnx-dynamic-shapes / --no-onnx-dynamic-shapes`.

2. **Documentation framing change**: the README's
   "parity-tested export to every major deployment target" line is
   true at the *artifact* level (export+load+run produces the same
   numerics within `atol=1e-3`), but should not be read as "ONNX
   export is faster than PyTorch on CUDA." `docs/vs_detectron2.md`
   is updated with this caveat. The capability matrix retains the
   "ONNX/CoreML/OpenVINO/TensorRT export — first-class CLI" claim
   because that's about deployment surface, not throughput.

## Consequences

- **The `mayaku export onnx` CLI now has a `--no-onnx-dynamic-shapes`
  flag** that's the recommended setting when the user explicitly
  targets TensorRT. The default stays `--onnx-dynamic-shapes` so
  existing pipelines keep working.
- **Benchmarks scripts (`benchmarks/onnx_throughput.py`,
  `benchmarks/onnx_eval_pipeline.py`)** use the fixed-shape export
  for the R-CNN row; the standalone classifier row is unchanged.
- **No changes to `ONNXBackbone` runtime, `cli/eval.py` hybrid
  wiring, or any production code path.** The runtime path was
  already correct.
- **Tests:** `tests/unit/test_onnx_export.py` adds
  `test_onnx_export_dynamic_input_shape_false_pins_input_dims`
  confirming the new flag actually pins the graph's input shape.
  466/466 pass.

## Alternatives considered

- **Restructuring the FPN to a TRT-friendly fusion pattern.**
  Rejected — would diverge from D2-faithful FPN, breaking ADR 003's
  parity guarantee for marginal speed gain we can't predict in
  advance.
- **Per-layer `trtexec --verbose` profile to identify the slowest
  op.** Defer. Could be next-round work if a specific user comes
  back asking "make TRT faster for R-CNN." For shipping, the
  conclusion is already clear.
- **Replacing ORT's TRT EP with a direct TensorRT engine** (no ORT
  middle layer). Could save 5–15% from ORT framework overhead but
  unlikely to flip the verdict. Adds a dependency on TRT's Python
  bindings beyond what ORT provides. Not worth the complexity.
- **Drop TensorRT support entirely.** Rejected — same logic as
  CoreML: the artifact has real users (INT8 deployment, TRT-native
  serving stacks) even if it isn't the throughput story for a
  developer machine.

## What this means for the paper

The deployment-target story has *one platform-specific exception*
to the otherwise-symmetric "artifact-not-throughput" framing:

- **Mayaku ships parity-validated exports to ONNX, CoreML, OpenVINO,
  and TensorRT.** Artifacts are correct (atol=1e-3) and load on
  every supported runtime.
- **GPU-available targets (CUDA, Apple Silicon MPS):** PyTorch
  eager is the recommended runtime. The deployment exports are for
  *artifact format*, not throughput. CoreML at parity with MPS;
  TRT slightly slower than PyTorch CUDA on R-CNN. Use the artifact
  for non-PyTorch deployment, not for speed-on-the-developer-box.
- **CPU-only Intel deployment targets: OpenVINO is genuinely 2.65×
  faster than PyTorch CPU on R-CNN.** This is the one row in the
  deployment matrix where the runtime claim is a real throughput
  win, not just artifact-coverage. Useful for embedded servers,
  ARM cross-compile targets, virtualised CPU instances, edge
  boxes. Note: PyTorch CUDA on the same Intel box is still 9.5×
  *faster than* OpenVINO; the OpenVINO win is conditional on no
  GPU being available.
- **For deployment to non-PyTorch targets generally** (iOS apps,
  Windows ML stacks, ROCm, Android NNAPI, future accelerators),
  the export is the enabling layer. That's the real Mayaku value
  vs Detectron2, whose export tooling is bit-rotted (per
  `paper_plan.md`).

The asymmetric finding (Mac/CUDA at parity-or-worse, Intel CPU at
2.65× faster) makes the deployment story *more* defensible at
paper level: it's not "Mayaku is faster than D2 because of TRT"
(which we now know is false on R-CNN), it's "Mayaku reaches a
deployment-target diversity D2 doesn't, with a real throughput win
on CPU servers and parity-with-PyTorch elsewhere." Same artifact
coverage claim, sharper hardware-conditioned throughput claim.

## References

- `benchmarks/onnx_throughput.py` — standalone classifier and
  Mayaku-graph throughput sweep across all available ORT providers.
- `benchmarks/onnx_eval_pipeline.py` — full eval-pipeline gating
  test (mirrors `examples/coreml_speed_check.py` round 3).
- `benchmarks/export_smoke_all_metaarchs.py` — all-12-checkpoint
  sweep across detection / segmentation / keypoints. Zero failures;
  confirms the .onnx export path works for every meta-arch in the
  model zoo, with structural assertions on the per-meta-arch
  output fields (boxes, +masks for mask R-CNN, +keypoints for
  keypoint R-CNN).
- `tests/unit/test_onnx_export.py` —
  `test_onnx_export_dynamic_input_shape_false_pins_input_dims`
  regression for the new flag.
- ADR 004 — symmetric finding for CoreML on Apple Silicon.
- ADR 003 — D2-faithful structure constraint that rules out FPN
  graph surgery as an alternative.
