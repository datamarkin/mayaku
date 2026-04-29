# Mayaku vs Detectron2

Mayaku is a backend-portable detection / segmentation / keypoint
library that runs the same R-CNN code on CUDA, Apple Silicon (MPS),
and CPU — and exports parity-tested ONNX, CoreML, OpenVINO, and
TensorRT. It loads Detectron2's COCO checkpoints (12/12 within
±0.1 AP), so you can adopt Mayaku without retraining anything.
Pure-Python install: no custom CUDA kernels, no compiled extensions,
no wheel chase.

## Capability matrix

| Capability | Mayaku | Detectron2 |
|---|---|---|
| Apple Silicon (MPS) train + eval | first-class | not supported |
| Custom CUDA kernels at install | none — pure PyTorch + torchvision ops | required (`detectron2/_C`) |
| `pip install -e .` on macOS | works out of the box | painful |
| Configuration system | Pydantic v2 (frozen, validated) | yacs (loose, late errors) |
| Channel order | RGB-native ([ADR 002](decisions/002-rgb-native-image-ingestion.md)) | BGR (caffe2 lineage) |
| Deformable convolution | dropped ([ADR 001](decisions/001-drop-deformable-convolution.md)) | supported |
| ONNX export | parity-tested (atol=1e-3, opset 17, dynamic axes) | community / partial |
| CoreML export | first-class (`mlprogram`, fp16/fp32) for Apple-ecosystem deployment | not first-class |
| OpenVINO export | shipped, best-effort | not first-class |
| TensorRT export | shipped, CUDA host required | not first-class |
| Distributed training | DDP via `torchrun` (`nccl`/`gloo`) | similar (Linux/CUDA-centric) |
| License | Apache 2.0 | Apache 2.0 |

## Parity vs Detectron2

All 12 D2 model-zoo checkpoints reproduce
Detectron2's published COCO val2017 numbers within ±0.1 AP after
conversion through `tools/convert_d2_checkpoint.py`. Maximum observed
gap across the 12: **+0.08 AP** (`keypoint_rcnn_X_101_32x8d_FPN_3x`,
keypoint AP). Example:

| Config | D2 expected | Mayaku actual | Δ |
|---|---|---|---|
| `faster_rcnn_R_50_FPN_3x` | 40.2 | 40.23 | +0.03 |
| `mask_rcnn_R_101_FPN_3x` (box / mask) | 42.9 / 38.6 | 42.93 / 38.63 | +0.03 / +0.03 |
| `keypoint_rcnn_R_50_FPN_3x` (box / kpt) | 55.4 / 65.5 | 55.45 / 65.49 | +0.05 / -0.01 |

Full per-checkpoint table: [`docs/d2_parity_report.md`](d2_parity_report.md).
Reproduction recipe: `bash tools/convert_all_d2.sh` (downloads each
`.pkl`, converts, evaluates, regenerates the parity report).

You can take an existing Detectron2 deployment and switch the runtime
to Mayaku — same architecture families (Faster / Mask / Keypoint
R-CNN), same backbones (R-50 / R-101 / X-101_32x8d FPN), same
checkpoints, no retraining.

## What you give up (deliberate non-goals)

- **Deformable convolution** ([ADR 001](decisions/001-drop-deformable-convolution.md)) — the portability cost wasn't worth the marginal AP. Plain conv is what runs unchanged on CUDA, MPS, and CPU.
- **BGR channel-order configurability** ([ADR 002](decisions/002-rgb-native-image-ingestion.md)) — RGB-native via PIL throughout. Means a known ~1–2 AP recipe-level diff vs D2 on JPEG inputs (PIL vs cv2 chroma upsampling); this is exactly what the parity validation measures and accepts.
- **Bit-for-bit weight compatibility on every layer** — the converter handles the seam between D2's caffe2-flavoured ResNet (`conv.norm.*`, `shortcut.*`, `fpn_lateral{N}`) and Mayaku's torchvision-flavoured layout (`bn.*`, `downsample.*`, `lateral_convs.{i}`). The ±0.1 AP delta per checkpoint is the measured architectural cost.
- **Out-of-scope architectures**: rotated boxes, panoptic FPN, Cascade R-CNN, RetinaNet, DenseNet — not shipped in v1.
- **cv2-based image pipeline** — PIL only. The recipe gap above is the reason an opt-in cv2 path is on the roadmap.

## What you can do today

- **Inference.** Load any of the 12 converted COCO checkpoints; `mayaku predict` or the Python `Predictor` returns `Instances` with boxes / masks / keypoints in original-image pixel coordinates.
- **Training from scratch.** `mayaku train ...` with a YAML config + COCO-format dataset. Supports DDP via `torchrun`, AMP on CUDA (fp16 by default), fp32 on MPS by default with opt-in fp16 via `solver.amp_dtype: float16`, gradient clipping, configurable solver schedule. MPS training works for small-dataset fine-tunes — Mayaku ships a custom gather-based `roi_align` for the MPS training path because torchvision's MPS roi_align backward triggers macOS's GPU watchdog (see `MPS_ISSUES_DISCOVERED.md`). MPS auto-enables `PYTORCH_ENABLE_MPS_FALLBACK=1` and prints an op-fallback summary at end of run so users can see which (if any) other ATen ops are round-tripping to CPU; set `MAYAKU_VERBOSE_MPS=1` to see per-call warnings.
- **Fine-tuning.** Load any converted checkpoint with `--weights` and continue training on a custom COCO-format dataset. `--pretrained-backbone` loads ImageNet ResNet for the from-scratch path; the two flags are mutually exclusive.
- **Export.** `mayaku export {onnx,coreml,openvino,tensorrt}` produces a deployable artifact with documented per-target parity tolerances (see [`docs/export/`](export/)).
- **Evaluation.** `mayaku eval` against a COCO ground-truth JSON; periodic progress lines plus the standard `AP / AP50 / AP75 / APs / APm / APl` dict per task.

Per-backend guarantees, AMP rules, op fallbacks, and MPS quirks live
in [`docs/portability.md`](portability.md).

## A note on export throughput

Mayaku ships **CoreML, ONNX, OpenVINO, and TensorRT exports as
deployment artifacts** for non-PyTorch targets — iOS apps, macOS
apps, Linux CPU servers, Windows ML stacks, edge devices, INT8
quantization workflows. **The artifacts are the value, not raw
throughput on a developer machine.**

The empirical pattern, measured across three platforms in `benchmarks/`:

| Graph | Mac (CoreML/MPS) | CUDA (TRT/PyTorch) | **Intel CPU (OpenVINO)** |
|---|---|---|---|
| Plain ResNet-50 classifier @ 224×224 | 6.7× faster via CoreML+ANE | 6.7× faster via TRT fp16 | 3.2× faster via OpenVINO |
| Mayaku R-CNN R-50+FPN @ 736×1344 | **1.13–1.18× via direct CoreML** | 0.65× (slower) | **2.65× faster** |
| Mayaku R-CNN X-101+FPN @ 736×1344 | **1.41–1.46× via direct CoreML** | (not measured) | (not measured) |

Two stories in this table:

1. **GPU-available targets (Mac MPS, CUDA): the deployment runtime
   is at parity-or-worse with the framework's eager mode for
   R-CNN.** PyTorch+cuDNN/MPS eager is hard to beat — it uses the
   same underlying kernels the deployment runtimes call, with less
   framework overhead. CoreML on Mac lands at parity (1.05–1.46×).
   TensorRT on CUDA actually loses (0.65× — slower than PyTorch).
   The export artifacts are still useful (iOS deployment, INT8
   serving stacks, etc.) but not for speed on a developer box.
2. **CPU-only Intel targets: OpenVINO genuinely beats PyTorch CPU
   by 2.65× on R-CNN.** The one platform where the deployment
   runtime delivers a real throughput win on the realistic
   detection workload. Useful for embedded servers, edge boxes,
   ARM cross-compile targets, virtualised CPU instances. Caveat:
   PyTorch CUDA on the same hardware (if available) is still 9.5×
   faster than OpenVINO; the win is conditional on no GPU being
   available.

See [ADR 004](decisions/004-coreml-export-positioning.md) for the
Mac analysis, [ADR 005](decisions/005-onnx-tensorrt-positioning.md)
for the CUDA/TRT and Intel/OpenVINO analyses.

**Practical implication, by deployment target:**

- **GPU-available (CUDA, MPS):** PyTorch eager is the recommended
  runtime. Use the exports when you need the artifact format for a
  non-PyTorch target (iOS app, Windows ML stack, etc.).
- **CPU-only (Intel server without GPU):** OpenVINO is the
  recommended runtime, 2.65× faster than PyTorch CPU. Use
  `mayaku export openvino` to produce the artifact, then deploy
  via ONNX Runtime's `OpenVINOExecutionProvider`.
- **Backbone-only feature extraction (any platform):** the 6.7×
  speedup over framework eager is genuine for plain
  classifier-shaped graphs. Mayaku doesn't currently advertise
  this path; trivially achievable by exporting the FPN-less
  ResNet alone.

## Roadmap

- **DINOv2 backbones** (`DINOV2_IMPLEMENTATION.md`): replace the
  ResNet/ResNeXt family with the published DINOv2 ViT ladder
  (S / B / L / g) for stronger pretrained init. Phase A spec is
  written; execution is the next milestone.
- **Opt-in cv2 image pipeline**: closes the ~1–2 AP recipe gap for
  users who want bit-tighter parity with D2 numbers.
- **Expanded `examples/`**: minimal scripts for inference,
  fine-tuning, and export.

## Why this exists

Detectron2 is excellent and remains the reference for R-CNN-family
research. It's also painful to install (custom CUDA kernels, ABI
mismatches, no MPS path), and its extension surface assumes
Linux + CUDA. Mayaku trades a small slice of D2's feature surface
(deformable conv, exotic architectures) for clean install on every
backend, parity-tested deployment exports, and Pydantic-typed configs
that fail at load time instead of mid-training. The 12-checkpoint
parity bar means switching is a one-line change in the runtime, not
a retrain.
