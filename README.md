<!--
  Placeholder numbers are marked  XXX / XX.X  — measured, to be filled in shortly.
  In-repo reference (delete before publishing): convnext_tiny+uniquery floor = 46.68 AP @ 1x, no pretraining.
-->

# Mayaku

[![CI](https://github.com/datamarkin/mayaku/actions/workflows/ci.yml/badge.svg)](https://github.com/datamarkin/mayaku/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

**The computer vision library that learns your data fastest.**

Mayaku trains detection, segmentation, and keypoint models on your own data, built around
**UniQuery** — a query-based head — and a purpose-built **ConvNeXt** backbone. It's pure
Python with zero custom CUDA kernels: one `pip install` trains on CUDA, Apple Silicon,
ROCm, or CPU, and exports to ONNX, CoreML, OpenVINO, and TensorRT. Apache 2.0.

> Built for developers with a few hundred images of *their own thing* — retail shelves,
> defects on a line, insect wings — not another COCO leaderboard entry.

---

## Highlights

- **Fine-tunes fast.** Strong accuracy on small custom datasets in few epochs. `auto_config`
  adapts the recipe to your dataset, and changing the class count just works — the head
  reinitialises automatically.
- **UniQuery head.** Anchor-free, NMS-free, with image-conditioned query generation (QGN),
  DN-DETR-style denoising, and train-with-N / deploy-with-fewer refinement stages — a
  built-in speed/accuracy dial. Detection, instance segmentation, and keypoints from one
  head family.
- **ConvNeXt backbone, 8 sizes** — `femto · pico · nano · tiny · small · base`,
  edge-class to high-accuracy, all FPN-ready.
- **Runs everywhere.** Pure Python: no wheel chase, no ABI mismatches. CUDA / Apple Silicon
  (MPS) / ROCm / CPU from a single install.
- **Deploys everywhere.** Parity-tested exports to `onnx`, `coreml`, `openvino`, `tensorrt`.

## Install

```bash
pip install mayaku
```

That covers training, inference, evaluation, and model download. Export targets are
optional extras — install only what you need:

```bash
pip install mayaku[onnx]       # ONNX export
pip install mayaku[coreml]     # CoreML export (macOS)
pip install mayaku[openvino]   # OpenVINO export (Intel CPU/iGPU/NPU)
pip install mayaku[tensorrt]   # TensorRT export (CUDA Linux)
```

## Quickstart

Train on your own dataset in three lines:

```python
from mayaku import train

result = train(weights="uniquery_convnext_tiny", data="dataset.yaml")
print(result["final_box_ap"], result["final_weights"])
```

`data=` is a single pointer — a dataset directory (Roboflow/YOLO-style splits) or a
`.yaml` descriptor. Class names come from your COCO json; `auto_config` derives the
schedule from your dataset size. Same thing from the CLI:

```bash
mayaku train --weights uniquery_convnext_tiny --data dataset.yaml
```

Predict:

```python
from mayaku.inference import Predictor

predictor = Predictor.from_pretrained("uniquery_convnext_tiny")  # config + weights + device, auto
instances = predictor("photo.jpg")
print(instances.pred_boxes.tensor, instances.scores, instances.pred_classes)
```

## Benchmarks

### Fine-tuning on custom datasets — RF100-VL

COCO measures one big dataset; real projects are hundreds of small ones. RF100-VL
benchmarks mean mAP across 100 datasets under a fixed fine-tuning budget — the closest
public proxy for how well a model learns *your* data.

<!-- FILL: methodology — N images/dataset, fixed epochs/budget, mean ± std across the suite -->

| Model | Params (M) | RF100-VL mean mAP | Epochs to converge | vs YOLO | vs RF-DETR |
|---|---:|---:|---:|---:|---:|
| uniquery_convnext_pico | XXX | XXX | XXX | XXX | XXX |
| uniquery_convnext_tiny | XXX | XXX | XXX | XXX | XXX |
| uniquery_convnext_base | XXX | XXX | XXX | XXX | XXX |

### COCO val2017

Competitive on the standard benchmark, trained with this library. The ConvNeXt-tiny number
is a **measured floor** — a short schedule with **no backbone pretraining**; with pretraining
we expect **50+ AP**.

| Model | Params (M) |  COCO box AP | FPS <!--device--> |
|---|---:|-------------:|---:|
| uniquery_convnext_pico  | XXX |          XXX | XXX |
| uniquery_convnext_tiny  | XXX |          XXX | XXX |
| uniquery_convnext_base  | XXX |          XXX | XXX |
| uniquery_convnext_large | XXX |          XXX | XXX |

<!-- FILL: device/precision for the FPS column, e.g. "RTX 4090, fp16, 800x1344" -->

## Model zoo

All COCO-trained checkpoints are hosted and fetched on first use — pass a name instead
of a path. Each ships in five variants: `pth · onnx · onnx-fixed · coreml-fp16 · openvino`.

**ConvNeXt + UniQuery** (detection · segmentation · keypoints):

| Model | Backbone | Profile |
|---|---|---|
| uniquery_convnext_pico  | ConvNeXt-pico  | edge / real-time |
| uniquery_convnext_tiny  | ConvNeXt-tiny  | flagship balance |
| uniquery_convnext_base  | ConvNeXt-base  | high accuracy |
| uniquery_convnext_large | ConvNeXt-large | max accuracy |

<!-- FILL: finalize this ship-set after the uniquery runs settle -->

**R-CNN heads** on ResNet/ResNeXt backbones, as familiar baselines:

| Task | Head | Backbones |
|---|---|---|
| Detection             | Faster R-CNN   | R-50 / R-101 / X-101 |
| Instance segmentation | Mask R-CNN     | R-50 / R-101 / X-101 |
| Keypoints / pose      | Keypoint R-CNN | R-50 / R-101 / X-101 |

List bundled names: `python -c "from mayaku import configs; print(*configs.list_all(), sep='\n')"`.
Machine-readable index: [`manifest.json`](https://dtmfiles.com/mayaku/v1/models/manifest.json)
(size + SHA256 per file). Cached under `<project>/cache/mayaku/` (override with `MAYAKU_CACHE_DIR`).

## Why Mayaku

**Query-based, without DETR's slow convergence.** UniQuery keeps the clean query-based
design — no anchors, no NMS — but reaches its accuracy in a fraction of the training time
DETR-style detectors need. An image-conditioned **query generator (QGN)** seeds queries
from FPN features instead of learning them from scratch, and **DN-DETR-style denoising**
stabilises box regression early. Train with N refinement stages, deploy with fewer — a
speed/accuracy dial with no retraining.

**ConvNeXt, eight sizes, no timm dependency.** A clean ConvNeXt implementation (the four
small `atto…nano` sizes are custom; `tiny…large` wrap torchvision) — pick your point on the
size/accuracy curve. A key-rename shim loads `timm` and original-release weights without
making `timm` a dependency.

**Portability by design.** No deformable conv, no custom ops, RGB-native end-to-end — the
same code path runs on every backend and survives the trip through ONNX → CoreML /
OpenVINO / TensorRT.

## Deploy anywhere

```bash
mayaku export onnx     --weights <model> --output model.onnx
mayaku export coreml   --weights <model> --output model.mlpackage
mayaku export openvino --weights <model> --output model.xml      # .bin alongside
mayaku export tensorrt --weights <model> --output model.engine   # CUDA host
```

All four are parity-tested. Per-target details (graph contents, tolerances, runtime
examples) in [`docs/export/`](docs/export/). On Intel CPU targets OpenVINO measurably
beats PyTorch eager; on GPU targets the export *artifact* is the value — see
[`docs/portability.md`](docs/portability.md).

## Coming from Detectron2?

Have Detectron2 R-CNN checkpoints — your own fine-tunes or the model zoo? Convert them to
Mayaku format once and pick up Apple Silicon support, clean exports, and no CUDA kernels.
The converted weights load into the matching Mayaku architecture for inference, eval, and
fine-tuning.

```bash
python tools/convert_d2_checkpoint.py your_model_final.pkl -o your_model.pth
```

Full conversion guide, parity table, and supported architectures:
**[`docs/detectron2.md`](docs/detectron2.md)**.

## Roadmap

- <!-- FILL: e.g. larger backbones, quantization, additional export targets -->

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — module layout end-to-end.
- [`docs/portability.md`](docs/portability.md) — backend matrix, op fallbacks, MPS quirks.
- [`docs/extending.md`](docs/extending.md) — adding a backbone / head / dataset / exporter.
- [`docs/export/`](docs/export/) — per-target export recipes.
- [`docs/detectron2.md`](docs/detectron2.md) — converting Detectron2 checkpoints.
- [`docs/decisions/`](docs/decisions/) — architecture decision records.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
