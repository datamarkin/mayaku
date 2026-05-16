# Mayaku

[![CI](https://github.com/datamarkin/mayaku/actions/workflows/ci.yml/badge.svg)](https://github.com/datamarkin/mayaku/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

**Detectron2's R-CNN family, ported to where it actually deploys.** A clean reimplementation of Faster R-CNN, Mask R-CNN, and Keypoint R-CNN that runs on **Apple Silicon**, exports cleanly to **ONNX** / **CoreML** / **OpenVINO** / **TensorRT**, and reproduces all 12 of Detectron2's model-zoo checkpoints to within ±0.1 AP. Pure Python — **no custom CUDA kernels**, **no compiled extensions**, **no wheel chase**. Apache 2.0.

> The 12 D2 model-zoo configs ship pre-converted — `mayaku predict --weights faster_rcnn_R_50_FPN_3x image.jpg` runs on Apple Silicon the first time you call it. Same architectures, same numbers (±0.1 AP), with the deployment paths Detectron2 never had: `.mlpackage`, `.onnx`, `.xml`, `.engine`.

## Why migrate from Detectron2?

| | Detectron2 | Mayaku |
|---|---|---|
| Apple Silicon (MPS) train + eval | not supported | first-class |
| `pip install -e .` on macOS | painful (custom CUDA kernels, ABI mismatches) | works out of the box |
| ONNX export | community-maintained, brittle, broken for keypoint | parity-tested (atol=1e-3, opset 17) |
| CoreML export | not first-class | shipped CLI verb, fp16/fp32, `mlprogram` |
| OpenVINO export | not first-class | shipped CLI verb |
| TensorRT export | not first-class | shipped CLI verb |
| Configuration | yacs / LazyConfig | Pydantic v2 (frozen, fail-at-load) |
| Last release | v0.6 (Nov 2021) | active |

You don't have to retrain or convert anything. All 12 model-zoo checkpoints reproduce D2's published COCO val2017 numbers within ±0.1 AP and ship [pre-converted and hosted](#pre-converted-models) — `mayaku predict --weights <name>` fetches on first use. Switch the runtime, keep the weights.

### Small things that matter

- **Channel order is RGB, not BGR.** Detectron2 inherits Caffe2's BGR convention; Mayaku is RGB-native end-to-end ([ADR 002](docs/decisions/002-rgb-native-image-ingestion.md)). Feed a `cv2.imread`-shaped BGR array and Mayaku will run without complaint and produce **wrong detections, no error raised**. Either swap channels at the boundary (`img = img[:, :, ::-1]` or `mayaku.utils.bgr_to_rgb`) or — better — load with `mayaku.utils.image.read_image` (Pillow under the hood, RGB by default).

- **Pixel mean / std are RGB-ordered.** The defaults are `[123.675, 116.280, 103.530]` / `[58.395, 57.120, 57.375]` (matching torchvision's ImageNet stats in RGB). If you copy a D2 yacs config that pins `PIXEL_MEAN = [103.53, 116.28, 123.675]` (BGR), the model normalises with the channels swapped and you again silently get bad features. Don't override the defaults unless your dataset truly needs different stats — and if you do, write them in RGB order.

- **Fine-tuned `.pkl` files need a one-time conversion.** The 12 zoo checkpoints are [already converted and hosted](#pre-converted-models) — you don't need to do anything for the standard configs. If you have your own fine-tuned `.pkl` from a D2 training run, D2's caffe2-flavoured key names (`conv.norm.*`, `shortcut.*`, `fpn_lateral{N}`) won't `load_state_dict` into Mayaku's torchvision-flavoured layout. Run `python tools/convert_d2_checkpoint.py your_model.pkl -o your_model.pth` once. See [Bringing your own Detectron2 checkpoint](#bringing-your-own-detectron2-checkpoint) below.

## Detectron2 parity

All 12 checkpoints reproduce Detectron2's published numbers within ±0.1 AP. Maximum observed gap across the sweep: **+0.08 AP**.

| Config | D2 published | Mayaku | Δ |
|---|---|---|---|
| `faster_rcnn_R_50_FPN_3x` | 40.2 | 40.23 | +0.03 |
| `faster_rcnn_R_101_FPN_3x` | 42.0 | 42.00 | +0.00 |
| `faster_rcnn_X_101_32x8d_FPN_3x` | 43.0 | 43.07 | +0.07 |
| `mask_rcnn_R_50_FPN_3x` (box / mask) | 41.0 / 37.2 | 40.98 / 37.17 | -0.02 / -0.03 |
| `mask_rcnn_R_101_FPN_3x` (box / mask) | 42.9 / 38.6 | 42.93 / 38.63 | +0.03 / +0.03 |
| `mask_rcnn_X_101_32x8d_FPN_3x` (box / mask) | 44.3 / 39.5 | 44.28 / 39.52 | -0.02 / +0.02 |
| `keypoint_rcnn_R_50_FPN_3x` (box / kpt) | 55.4 / 65.5 | 55.45 / 65.49 | +0.05 / -0.01 |
| `keypoint_rcnn_R_101_FPN_3x` (box / kpt) | 56.4 / 66.1 | 56.43 / 66.04 | +0.03 / -0.06 |
| `keypoint_rcnn_X_101_32x8d_FPN_3x` (box / kpt) | 57.3 / 66.0 | 57.26 / 66.08 | -0.04 / +0.08 |

Full per-checkpoint table including the 1x configs: [`docs/d2_parity_report.md`](docs/d2_parity_report.md). Reproduce the entire sweep with `bash tools/convert_all_d2.sh` (downloads each `.pkl`, converts, evaluates, regenerates the report).

These numbers come from loading and **evaluating** D2's converged weights in Mayaku — not from training from scratch. Training-from-scratch parity (270k iters on COCO, ending at D2's published numbers) is on the roadmap, not measured.

## Pre-converted models

Pass a model name instead of a path; the CLI fetches it on first use:

```bash
mayaku predict faster_rcnn_R_50_FPN_3x image.jpg --weights faster_rcnn_R_50_FPN_3x
```

The first positional arg is either a bundled config name (the 12 zoo configs ship inside the wheel) or a path to your own `.yaml`. List bundled names with `python -c "from mayaku import configs; print(*configs.list_all(), sep='\n')"`.

For library use, `Predictor.from_pretrained` takes you from a model name to detections in one call:

```python
from mayaku.inference import Predictor

predictor = Predictor.from_pretrained("faster_rcnn_R_50_FPN_3x")  # auto: config + weights + device
instances = predictor("photo.jpg")
print(instances.pred_boxes.tensor, instances.scores, instances.pred_classes)
```

Override `weights=` / `config=` / `device=` independently if you need to. For lower-level access, `mayaku.configs` exposes the bundled YAMLs directly:

```python
from mayaku import configs
cfg_path = configs.path("faster_rcnn_R_50_FPN_3x")  # → Path
cfg_path = configs.faster_rcnn_R_50_FPN_3x          # attribute form, same Path
cfg      = configs.load("faster_rcnn_R_50_FPN_3x")  # → MayakuConfig
```

Or pre-stage with `mayaku download <name>` (all variants) or `mayaku download <name> --target <variant>`. Cached under `~/.cache/mayaku/v1/models/`, SHA256-verified.

**Variants:** `pth` (PyTorch) · `onnx` (dynamic) · `onnx-fixed` (TRT-friendly) · `coreml-fp16` (Apple Silicon) · `openvino` (Intel CPU/iGPU/Arc/NPU).

### Detection

| Model | pth | onnx | onnx-fixed | coreml-fp16 | openvino |
|---|:---:|:---:|:---:|:---:|:---:|
| faster_rcnn_R_50_FPN_1x | [⬇](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_50_FPN_1x.pth) | [⬇](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_50_FPN_1x.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_50_FPN_1x.800x1344.fixed.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_50_FPN_1x.fp16.mlpackage.zip) | [xml](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_50_FPN_1x.openvino.xml) · [bin](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_50_FPN_1x.openvino.bin) |
| faster_rcnn_R_50_FPN_3x | [⬇](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_50_FPN_3x.pth) | [⬇](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_50_FPN_3x.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_50_FPN_3x.800x1344.fixed.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_50_FPN_3x.fp16.mlpackage.zip) | [xml](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_50_FPN_3x.openvino.xml) · [bin](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_50_FPN_3x.openvino.bin) |
| faster_rcnn_R_101_FPN_3x | [⬇](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_101_FPN_3x.pth) | [⬇](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_101_FPN_3x.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_101_FPN_3x.800x1344.fixed.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_101_FPN_3x.fp16.mlpackage.zip) | [xml](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_101_FPN_3x.openvino.xml) · [bin](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_R_101_FPN_3x.openvino.bin) |
| faster_rcnn_X_101_32x8d_FPN_3x | [⬇](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_X_101_32x8d_FPN_3x.pth) | [⬇](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_X_101_32x8d_FPN_3x.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_X_101_32x8d_FPN_3x.800x1344.fixed.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_X_101_32x8d_FPN_3x.fp16.mlpackage.zip) | [xml](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_X_101_32x8d_FPN_3x.openvino.xml) · [bin](https://dtmfiles.com/mayaku/v1/models/detection/faster_rcnn_X_101_32x8d_FPN_3x.openvino.bin) |

### Segmentation

| Model | pth | onnx | onnx-fixed | coreml-fp16 | openvino |
|---|:---:|:---:|:---:|:---:|:---:|
| mask_rcnn_R_50_FPN_1x | [⬇](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_50_FPN_1x.pth) | [⬇](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_50_FPN_1x.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_50_FPN_1x.800x1344.fixed.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_50_FPN_1x.fp16.mlpackage.zip) | [xml](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_50_FPN_1x.openvino.xml) · [bin](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_50_FPN_1x.openvino.bin) |
| mask_rcnn_R_50_FPN_3x | [⬇](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_50_FPN_3x.pth) | [⬇](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_50_FPN_3x.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_50_FPN_3x.800x1344.fixed.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_50_FPN_3x.fp16.mlpackage.zip) | [xml](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_50_FPN_3x.openvino.xml) · [bin](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_50_FPN_3x.openvino.bin) |
| mask_rcnn_R_101_FPN_3x | [⬇](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_101_FPN_3x.pth) | [⬇](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_101_FPN_3x.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_101_FPN_3x.800x1344.fixed.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_101_FPN_3x.fp16.mlpackage.zip) | [xml](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_101_FPN_3x.openvino.xml) · [bin](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_R_101_FPN_3x.openvino.bin) |
| mask_rcnn_X_101_32x8d_FPN_3x | [⬇](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_X_101_32x8d_FPN_3x.pth) | [⬇](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_X_101_32x8d_FPN_3x.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_X_101_32x8d_FPN_3x.800x1344.fixed.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_X_101_32x8d_FPN_3x.fp16.mlpackage.zip) | [xml](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_X_101_32x8d_FPN_3x.openvino.xml) · [bin](https://dtmfiles.com/mayaku/v1/models/segmentation/mask_rcnn_X_101_32x8d_FPN_3x.openvino.bin) |

### Keypoints

| Model | pth | onnx | onnx-fixed | coreml-fp16 | openvino |
|---|:---:|:---:|:---:|:---:|:---:|
| keypoint_rcnn_R_50_FPN_1x | [⬇](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_50_FPN_1x.pth) | [⬇](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_50_FPN_1x.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_50_FPN_1x.800x1344.fixed.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_50_FPN_1x.fp16.mlpackage.zip) | [xml](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_50_FPN_1x.openvino.xml) · [bin](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_50_FPN_1x.openvino.bin) |
| keypoint_rcnn_R_50_FPN_3x | [⬇](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_50_FPN_3x.pth) | [⬇](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_50_FPN_3x.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_50_FPN_3x.800x1344.fixed.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_50_FPN_3x.fp16.mlpackage.zip) | [xml](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_50_FPN_3x.openvino.xml) · [bin](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_50_FPN_3x.openvino.bin) |
| keypoint_rcnn_R_101_FPN_3x | [⬇](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_101_FPN_3x.pth) | [⬇](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_101_FPN_3x.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_101_FPN_3x.800x1344.fixed.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_101_FPN_3x.fp16.mlpackage.zip) | [xml](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_101_FPN_3x.openvino.xml) · [bin](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_R_101_FPN_3x.openvino.bin) |
| keypoint_rcnn_X_101_32x8d_FPN_3x | [⬇](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.pth) | [⬇](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.800x1344.fixed.onnx) | [⬇](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.fp16.mlpackage.zip) | [xml](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.openvino.xml) · [bin](https://dtmfiles.com/mayaku/v1/models/keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.openvino.bin) |

`coreml-fp16` is a zipped `.mlpackage` directory — `mayaku download` unpacks it; if you fetch directly, `unzip` it yourself.

**TensorRT engines aren't hosted** — they're tied to a specific GPU architecture and TRT version, so a hosted file would fail to load on mismatched hardware. Build locally on your CUDA host from the `onnx-fixed` artifact:

```bash
mayaku download faster_rcnn_R_50_FPN_3x --target onnx-fixed
mayaku export tensorrt configs/detection/faster_rcnn_R_50_FPN_3x.yaml \
    --weights faster_rcnn_R_50_FPN_3x \
    --output ~/faster_rcnn_R_50_FPN_3x.engine \
    --sample-height 800 --sample-width 1344
```

Machine-readable index: [`manifest.json`](https://dtmfiles.com/mayaku/v1/models/manifest.json) (size + SHA256 per file; dtmfiles also publishes a `.sha256` sidecar next to every artifact).

## What you get

| | |
|---|---|
| Detectors | Faster R-CNN · Mask R-CNN · Keypoint R-CNN (56×56 heatmaps) |
| Backbones | ResNet-50, ResNet-101, ResNeXt-101 (32×8d) — all with FPN |
| Backends | CUDA · MPS · CPU (single codebase, single wheel) |
| Export targets | `mayaku export {onnx, coreml, openvino, tensorrt}` — all four parity-tested. TensorRT needs a CUDA host. |
| Configuration | Pydantic v2, frozen + extra-forbidden, validated at load |
| Distributed | DDP via `mayaku train --num-gpus N` (or `mayaku.api.train(..., num_gpus=N)`, or `torchrun`) — `nccl`/RCCL on CUDA/ROCm, `gloo` elsewhere |
| License | Apache 2.0 |

See [`docs/architecture.md`](docs/architecture.md) for module layout and [`docs/portability.md`](docs/portability.md) for the per-backend matrix.

## Install

```bash
pip install mayaku
```

That's enough for inference, training, evaluation, and `mayaku download`. Export targets are optional extras — install only what you need:

```bash
pip install mayaku[onnx]        # ONNX export (adds the onnx validation package)
pip install mayaku[coreml]      # CoreML export (macOS only)
pip install mayaku[openvino]    # OpenVINO export (Intel targets)
pip install mayaku[tensorrt]    # TensorRT export (CUDA Linux only)
```

**onnxruntime is not bundled in `[onnx]`** because the CPU and CUDA wheels are separate packages — installing the wrong one is the most common source of `libcudart.so` / `.dll` errors. Install the variant that matches your host:

```bash
pip install onnxruntime          # CPU · Apple Silicon · any host without NVIDIA GPU
pip install onnxruntime-gpu      # CUDA Linux / Windows — must match your CUDA version
```

If either is missing when needed, the CLI reports the correct install command for your hardware. You can combine extras freely: `pip install mayaku[onnx,coreml]`. The `tensorrt` extra carries a PEP 508 marker that makes it a no-op on macOS, so `pip install mayaku[onnx,coreml,openvino,tensorrt]` is safe to run on any host.

**Contributing:**

```bash
git clone https://github.com/datamarkin/mayaku
pip install -e ".[dev,onnx]"
```

## Bringing your own Detectron2 checkpoint

The 12 zoo checkpoints are [already converted and hosted](#pre-converted-models) — you don't need to do anything for the standard configs. This section is only for the case where you have a fine-tuned D2 `.pkl` from your own training run and want to keep using those weights.

```bash
# Convert your fine-tuned D2 .pkl → Mayaku .pth (one-shot, no network)
python tools/convert_d2_checkpoint.py your_model_final.pkl -o your_model.pth

# Use the converted .pth like any other Mayaku checkpoint
mayaku predict faster_rcnn_R_50_FPN_3x image.jpg --weights your_model.pth --device mps
```

Covers Faster / Mask / Keypoint R-CNN with R-50 / R-101 / X-101_32x8d FPN — the same architectures Mayaku ships. Head-specific rename rules are inert when the source `.pkl` doesn't contain them, so a Faster R-CNN checkpoint converts cleanly without flags. See [`tools/README.md`](tools/README.md) for the full rename table and edge cases.

If your D2 setup uses anything Mayaku doesn't ship (DCN, Cascade, Panoptic, DETR, ViTDet…), the converter can't help — see [Small things that matter](#small-things-that-matter) for the missing pieces.

## Quickstart — CLI

The `mayaku` console script bundles four subcommands. Every one takes a YAML config plus per-task arguments.

### Train

```bash
mayaku train configs/detection/faster_rcnn_R_50_FPN_3x.yaml \
    --json /data/coco/annotations/instances_train2017.json \
    --images /data/coco/train2017 \
    --output runs/frcnn_r50 \
    --pretrained-backbone \   # ImageNet weights for the ResNet (recommended)
    --device cuda             # cpu / mps / cuda; default = auto
```

Pass `--pretrained-backbone` for fine-tuning. The schema's `backbone.freeze_at=2` default freezes the ResNet stem + res2, which is only meaningful when those stages already carry useful features. For genuine from-scratch training, set `model.backbone.freeze_at: 0` in the YAML and omit the flag.

`--max-iter N` overrides `solver.max_iter` for smoke runs; `--log-period N` controls per-iteration log frequency (default 20).

### Evaluate

```bash
mayaku eval configs/detection/faster_rcnn_R_50_FPN_3x.yaml \
    --weights runs/frcnn_r50/model_final.pth \
    --json /data/coco/annotations/instances_val2017.json \
    --images /data/coco/val2017
```

Prints the COCO `AP / AP50 / AP75 / APs / APm / APl` dict for boxes, plus masks/keypoints when the meta-architecture asks for them.

### Predict (single image)

```bash
mayaku predict configs/detection/faster_rcnn_R_50_FPN_3x.yaml path/to/image.jpg \
    --weights runs/frcnn_r50/model_final.pth \
    --output detections.json
```

### Download

```bash
# List all available models and variants
mayaku download --list

# Download all variants of a model (pth + onnx + onnx-fixed + coreml-fp16 + openvino)
mayaku download faster_rcnn_R_50_FPN_3x

# Download a single variant
mayaku download faster_rcnn_R_50_FPN_3x --target coreml-fp16

# Pre-stage everything for an offline deployment machine
mayaku download --all

# Skip SHA256 verification (air-gapped or offline-cached environments)
mayaku download faster_rcnn_R_50_FPN_3x --no-verify
```

Artifacts are cached under `~/.cache/mayaku/v1/models/` (or `$XDG_CACHE_HOME/mayaku/…`). Passing a bare model name to `--weights` in `eval` and `predict` triggers the same fetch automatically on first use.

### Export

```bash
# ONNX (required target, opset 17, dynamic batch + spatial axes by default)
mayaku export onnx configs/detection/faster_rcnn_R_50_FPN_3x.yaml \
    --weights runs/frcnn_r50/model_final.pth --output model.onnx

# Same surface for the other targets
mayaku export coreml ...    --output model.mlpackage
mayaku export openvino ...  --output model.xml      # .bin written alongside
mayaku export tensorrt ...  --output model.engine   # CUDA host required
```

Per-target details (what's *in* the exported graph, what stays Python-side, parity tolerances, runtime examples) live in [`docs/export/`](docs/export/).

## Quickstart — Python API

```python
from mayaku.inference import Predictor

predictor = Predictor.from_pretrained("faster_rcnn_R_50_FPN_3x")
instances = predictor("path/to/image.jpg")
# -> mayaku.structures.Instances with .pred_boxes, .scores, .pred_classes
#    (and .pred_masks / .pred_keypoints when the architecture asks for them),
#    all in original-image pixel coordinates.
```

Use your own checkpoint or override the device:

```python
predictor = Predictor.from_pretrained(
    "faster_rcnn_R_50_FPN_3x",
    weights="runs/frcnn_r50/model_final.pth",
    device="cpu",
)
```

For full manual control (custom config object, hand-built model, swapping the backbone for an exported runtime artifact), `Predictor.from_config(cfg, model)` is still available — that's the lower-level constructor `from_pretrained` wraps.

## Examples

Runnable end-to-end scripts live in [`examples/`](examples/) — minimal copy-paste templates for inference (single image, batch, video), fine-tuning, evaluation, and export to every deployment target. Most scripts take no arguments: `python examples/predict.py` downloads the model, fetches a sample image, prints detections. To adapt, edit the model-name literal in the file.

For a complete fine-tuning script (custom dataset discovery, COCO eval hook, Mac-friendly defaults), see [`examples/finetune.py`](examples/finetune.py). For training your own loop, see [`docs/architecture.md`](docs/architecture.md) §"Engine" — `mayaku.engine.SimpleTrainer` / `AMPTrainer` are usable directly.

## Configuration

Configs are pydantic v2 models (see `mayaku.config.schemas`); YAML is the on-disk encoding. Defaults match the Detectron2 3x schedule modulo the overrides recorded in the [decision log](docs/decisions/) (RGB channel order, no rotated boxes, no deformable conv, `device="auto"`).

Minimal example:

```yaml
model:
  meta_architecture: faster_rcnn
  backbone:
    name: resnet50
    norm: FrozenBN
  roi_heads:
    num_classes: 80
solver:
  max_iter: 90000
  steps: [60000, 80000]
  base_lr: 0.02
input:
  min_size_train: [640, 672, 704, 736, 768, 800]
  max_size_train: 1333
```

Configs are frozen and reject unknown fields — you'll get a validation error at load time, not a silent default mid-training.

## Deployment & throughput

Mayaku ships **CoreML, ONNX, OpenVINO, and TensorRT exports as deployment artifacts** for non-PyTorch targets — iOS apps, macOS apps, Linux CPU servers, Windows ML stacks, edge devices, INT8 quantization workflows. **The artifacts are the value, not raw throughput on a developer machine.**

Empirical pattern, measured across three platforms in `benchmarks/`:

- **GPU-available targets (CUDA, Apple Silicon MPS)**: PyTorch eager is essentially optimal for the R-CNN graph shape — multi-output FPN with stride-2 lateral connections defeats the deployment runtimes' fusion templates. Use the exports when you need the artifact format for a non-PyTorch target, not for a speed gain on a developer box.
- **CPU-only Intel targets**: OpenVINO genuinely beats PyTorch CPU by **2.65×** on R-CNN R-50 FPN. The one row where the deployment-runtime claim is real, not just rhetorical. Useful for embedded servers, edge boxes, virtualised CPU instances.
- **Backbone-only feature extraction (any platform)**: ~6× over framework eager on Mac/CUDA, ~3× on Intel CPU. Classifier-shaped graphs are exactly what these runtimes are designed for.

Full per-platform numbers and analysis in [`docs/vs_detectron2.md`](docs/vs_detectron2.md) §"A note on export throughput" and ADRs [004](docs/decisions/004-coreml-export-positioning.md) (CoreML), [005](docs/decisions/005-onnx-tensorrt-positioning.md) (ONNX/TRT/OpenVINO).

## What's deliberately not shipped

- **Deformable convolution** ([ADR 001](docs/decisions/001-drop-deformable-convolution.md)) — the portability cost wasn't worth the marginal AP. D2's strongest DCN-cascade configs sit ~2–4 AP above what Mayaku can reach; we don't try to match them.
- **BGR channel-order configurability** ([ADR 002](docs/decisions/002-rgb-native-image-ingestion.md)) — RGB-native via PIL throughout.
- **Out-of-scope architectures**: rotated boxes, panoptic FPN, Cascade R-CNN, RetinaNet, DETR, ViTDet, PointRend, DensePose, and the rest of D2's `projects/`. Mayaku ships exactly three meta-architectures. For a researcher comparing detector families, D2 is still the right tool.
- **cv2-based image pipeline** — PIL only. Closes a known ~1–2 AP recipe gap on JPEG inputs against D2; a cv2 path is on the roadmap.

For the full honest comparison (including where D2 is still better), see [`docs/vs_detectron2.md`](docs/vs_detectron2.md).

## Roadmap

- **DINOv2 backbones** — ViT ladder (S / B / L / g) for stronger pretrained init, replacing the current ResNet/ResNeXt-only backbone surface. Not yet shipped.
- **Training-from-scratch parity validation** — current parity numbers come from loading + evaluating D2's converged weights, not training to them.
- **Opt-in cv2 image pipeline** — closes the recipe-level AP gap for users who want bit-tighter D2 parity.
- **CI on accelerators** — Linux/CPU matrix runs on every push today (see the [CI badge](https://github.com/datamarkin/mayaku/actions/workflows/ci.yml) above); MPS and CUDA self-hosted runners are still manual.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — module layout end-to-end.
- [`docs/portability.md`](docs/portability.md) — backend matrix, op fallbacks, AMP rules, MPS quirks, distributed launchers.
- [`docs/extending.md`](docs/extending.md) — adding a backbone / head / dataset / augmentation / exporter.
- [`docs/vs_detectron2.md`](docs/vs_detectron2.md) — capability matrix, parity, non-goals, deployment-throughput analysis.
- [`docs/export/`](docs/export/) — per-target export recipes (ONNX, CoreML, OpenVINO, TensorRT).
- [`docs/decisions/`](docs/decisions/) — architecture decision records.

## Development

```bash
ruff check src tests
ruff format --check src tests     # CI also runs this; missing it locally is a common cause of CI-only failures
mypy
MAYAKU_DEVICE=cpu pytest          # also: mps, cuda
```

`MAYAKU_DEVICE` selects which backend the test suite runs on. Tests marked `cuda` / `mps` / `multi_gpu` / `tensorrt` auto-skip when the active backend or the optional dependency isn't available. CI runs the CPU subset on every push (`.github/workflows/ci.yml`); MPS and CUDA stay manual until self-hosted runners are wired up.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
