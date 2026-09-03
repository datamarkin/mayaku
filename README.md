# Mayaku

[![CI](https://github.com/datamarkin/mayaku/actions/workflows/ci.yml/badge.svg)](https://github.com/datamarkin/mayaku/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/datamarkin/mayaku/blob/main/LICENSE)

**The computer vision library that learns *your* data fastest.**

Mayaku trains detection, instance segmentation, and keypoint models on your own data. It's
built around **UniQuery** — a query-based head — on a **ConvNeXt** backbone, and its base
models are **purpose-pretrained on Objects365** (365 classes) for fast fine-tuning, so they
transfer to *your* classes quickly. Pure Python, zero custom CUDA kernels: one `pip install mayaku` trains on CUDA,
Apple Silicon, ROCm, or CPU, and exports to ONNX, CoreML, OpenVINO, and TensorRT.
Apache 2.0.

![RF100-VL nano: mean COCO AP against mean training wall-clock, averaged over 100 datasets](https://raw.githubusercontent.com/datamarkin/mayaku/main/curve_nano.png)

On [RF100-VL](https://rf100-vl.org/) — 100 real, custom datasets, not another COCO split —
Mayaku's nano model reaches the highest mean AP of the three, and gets to the others'
accuracy in a fraction of their training time.

Every AP below is COCO **AP@[.50:.95]**, scored by pycocotools and averaged across datasets.

| Library | Params (M) | AP final | AP best | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|
| **mayaku-n** | **12.9** | **0.572** | **0.575** | **0.339** | **0.511** | 0.635 | 24.4 min |
| rfdetr-n | 30.5 | 0.562 | 0.574 | 0.296 | 0.495 | **0.656** | 114.6 min |
| yolo26n | 2.4 | 0.537 | 0.548 | 0.283 | 0.477 | 0.608 | **19.1 min** |

**On speed, read the chart at equal accuracy.** YOLO26n is quickest to *finish* — and
finishes lowest. Mayaku passes yolo26n's final AP after **9.0 minutes** of its own training,
where yolo26n takes 19.1; it passes rfdetr-n's after **14.3 minutes**, where rfdetr-n takes
almost 2 hours.

![RF100-VL nano: mean COCO AP split by object size class](https://raw.githubusercontent.com/datamarkin/mayaku/main/apsize_nano_common.png)

> **Benchmark scope:** nano tier, all **100** RF100-VL datasets, single RTX 3060. Each
> library runs its own default recipe to completion. No tuned hyperparameters and every
> checkpoint is scored by the same pycocotools evaluator on the same
> val split, so the AP numbers are identical in definition across libraries.
> **The s/m/l tiers are still training** and will be published as they finish, whatever they
> show.

> Built for developers with a few hundred images of *their own thing* — retail shelves,
> defects on a line, insect wings — not another COCO leaderboard entry.

---

## Highlights

- **Fine-tunes fast.** Highest mean AP on RF100-VL's 100 custom datasets, reaching the
  other libraries' accuracy in a fraction of their training time — see above. `auto_config`
  adapts the recipe (schedule, LR, augmentation) to your dataset size automatically.
- **Objects365-pretrained.** Base models are pretrained on a large, detection-native dataset
  (365 classes), a broad starting point for transfer.
- **UniQuery head.** Anchor-free, NMS-free, with image-conditioned query generation (QGN)
  and train-with-N / deploy-with-fewer refinement stages — a built-in speed/accuracy dial
  with no retraining. Detection and instance segmentation share one query representation;
  keypoints ride the same detector.
- **Six sizes, three tasks.** `mayaku-n` through `mayaku-xxl`, each in detection,
  instance-segmentation, and keypoint variants.
- **Runs everywhere.** Pure Python: no wheel chase, no ABI mismatches, no custom ops.
  CUDA / Apple Silicon (MPS) / ROCm / CPU from a single install.
- **Deploys everywhere.** Parity-tested exports to `onnx`, `coreml`, `openvino`, `tensorrt`.

## Install

```bash
pip install mayaku
```

That covers training, inference, evaluation, and model download. Export targets are optional
extras — install only what you need:

```bash
pip install mayaku[onnx]       # ONNX export
pip install mayaku[coreml]     # CoreML export (macOS)
pip install mayaku[openvino]   # OpenVINO export (Intel CPU/iGPU/NPU)
pip install mayaku[tensorrt]   # TensorRT export (CUDA Linux)
```

## Quickstart

Fine-tune on your own dataset. Point at COCO-format splits — a train annotation JSON and its
image directory; add a val split to get final eval. Pass a bundled model name as `weights`
and it fetches the pretrained checkpoint on first use; the class-specific head re-initialises
automatically when your class count differs.

```python
from pathlib import Path

from mayaku import train

result = train(
    weights="mayaku-n-det",
    train_annotations=Path("data/train/_annotations.coco.json"),
    train_images=Path("data/train"),
    val_annotations=Path("data/valid/_annotations.coco.json"),
    val_images=Path("data/valid"),
)
print(result["final_box_ap"], result["final_weights"])
```

Same thing from the CLI:

```bash
mayaku train --weights mayaku-n-det \
  --annotations data/train/_annotations.coco.json --images data/train \
  --val-annotations data/valid/_annotations.coco.json --val-images data/valid
```

Predict:

```python
from mayaku import from_pretrained

predictor = from_pretrained("mayaku-n-det")   # config + weights + device, auto
instances = predictor("photo.jpg")
print(instances.pred_boxes.tensor, instances.scores, instances.pred_classes)
```

```bash
mayaku predict mayaku-n-det photo.jpg --output result.json
```

## Throughput

Measured on a single **NVIDIA RTX 3060** — the most commonly owned GPU among Hugging Face
users ([HF hardware](https://huggingface.co/hardware)) — at 640px, TensorRT FP16, back-to-back
with no cooldown pauses between timed inferences. CUDA numbers first; MPS and CPU figures
will follow.

Two throughput numbers are reported: **engine** is the model forward pass alone, **end-to-end**
is the full path you actually run (decode → preprocess → inference → post-process). The
end-to-end number is the one that predicts your application's frame rate.

| Model | Params (M) | Engine FPS | **End-to-end FPS** | VRAM (MB) |
|---|---:|---:|---:|---:|
| `mayaku-n-det` | 12.9 | 247.7 | **198.8** | 314 |
| `mayaku-s-det` | 23.1 | 175.7 | **150.0** | 340 |
| `mayaku-m-det` | 36.0 | 137.6 | **121.0** | 370 |
| `mayaku-l-det` | 57.8 | 98.9 | **93.5** | 464 |
| `mayaku-xl-det` | 117.7 | 61.5 | **60.8** | 602 |
| `mayaku-xxl-det` | 170.9 | 51.2 | **48.3** | 664 |

## Model zoo

Base models are hosted and fetched on first use — pass a name instead of a path. Every size
ships in three task variants, `mayaku-<size>-{det,seg,key}` (e.g. `mayaku-m-seg`).

| Name | Backbone | Params (M) | Tasks | Profile |
|---|---|---:|---|---|
| `mayaku-n` | ConvNeXt-femto | 12.9 | det · seg · key | edge / real-time |
| `mayaku-s` | ConvNeXt-nano | 23.1 | det · seg · key | balanced |
| `mayaku-m` | ConvNeXt-tiny | 36.0 | det · seg · key | high accuracy |
| `mayaku-l` | ConvNeXt-tiny (wide) | 57.8 | det · seg · key | best accuracy/speed |
| `mayaku-xl` | ConvNeXt-base | 117.7 | det · seg · key | max accuracy |
| `mayaku-xxl` | ConvNeXt-base (6-stage decoder) | 170.9 | det · seg · key | accuracy ceiling |

List available names:

```bash
mayaku download --list
```

Machine-readable index: [`manifest.json`](https://dtmfiles.com/mayaku/v1/models/manifest.json)
(size + SHA256 per file). Cached under `<project>/cache/mayaku/` (override with `MAYAKU_CACHE_DIR`).

## Why Mayaku

**Query-based, without DETR's slow convergence.** UniQuery keeps the clean query-based design
— no anchors, no NMS — but reaches its accuracy in a fraction of the training time DETR-style
detectors need. An image-conditioned **query generator (QGN)** seeds queries from FPN
features instead of learning them from scratch, so the model starts from meaningful proposals
on epoch one. Train with N refinement stages, deploy with fewer — a speed/accuracy dial with
no retraining.

**Detection, segmentation, and keypoints from one family.** Detection and instance
segmentation are driven by a single shared query representation (the mask head is conditioned
on the same per-object query features); keypoints run on the same detector. One architecture,
one training path, three tasks.

**Portability by design.** No deformable conv, no custom CUDA/C++ ops, RGB-native end-to-end
— the same code path runs on every backend and survives the trip through ONNX → CoreML /
OpenVINO / TensorRT.

## Deploy anywhere

```bash
mayaku export onnx     mayaku-n-det --output model.onnx
mayaku export coreml   mayaku-n-det --output model.mlpackage
mayaku export openvino mayaku-n-det --output model.xml      # .bin alongside
mayaku export tensorrt mayaku-n-det --output model.engine   # CUDA host
```

All four are parity-tested. On Intel CPU targets OpenVINO measurably beats PyTorch eager; on
GPU targets the export *artifact* is the value.

## Built on

UniQuery stands on a line of query-based detection work: **Sparse R-CNN** (iterative dynamic
refinement), **QueryInst** (query-conditioned dynamic mask heads), **Featurized Query R-CNN**
(image-conditioned query generation), and **DN-DETR** (denoising for stable early box
regression). The backbone is **ConvNeXt**, with several sizes built on torchvision's
implementation. What Mayaku adds is the unified head family, the Objects365 pretraining
recipe, and the fine-tuning defaults that make them converge quickly on small datasets.

## Roadmap

- **Full RF100-VL results** — the nano tier is complete above; s/m/l are training.
- **Curated custom-class models** — ready-to-use weights on a hand-picked set of common
  real-world classes (people, vehicles, and more), for projects that don't want to start from
  the Objects365 base.
- **Documentation** — a proper docs site, coming soon.

## License

Apache 2.0 — see [`LICENSE`](https://github.com/datamarkin/mayaku/blob/main/LICENSE). Applies to the full library and every published
model weight.
