# Examples

End-to-end scripts for every common Mayaku workflow.
All scripts run from the repo root and auto-download weights on first use.

## Quickstart

```bash
pip install mayaku
python examples/predict.py          # detection — zero args, downloads everything
python examples/predict_segmentation.py
python examples/predict_keypoint.py
```

---

## Predict

Single-image inference. No arguments required — downloads the default model and
a sample COCO image on first run.

| Script | Architecture | Default model |
|---|---|---|
| `predict.py` | Faster R-CNN (detection) | `faster_rcnn_R_50_FPN_3x` |
| `predict_segmentation.py` | Mask R-CNN (segmentation) | `mask_rcnn_R_50_FPN_3x` |
| `predict_keypoint.py` | Keypoint R-CNN (pose) | `keypoint_rcnn_R_50_FPN_3x` |

```bash
# Use your own image
python examples/predict.py --image photo.jpg

# Switch model (auto-downloaded)
python examples/predict.py --image photo.jpg --model faster_rcnn_R_101_FPN_3x

# Save detections to JSON
python examples/predict.py --image photo.jpg --output detections.json
```

Common flags: `--model`, `--config`, `--threshold`, `--device`, `--output`.

---

## Batch Predict

Process a directory of images and write one JSON per image plus a combined summary.

```bash
python examples/batch_predict.py --images /path/to/images/

# Segmentation batch
python examples/batch_predict.py \
    --images /path/to/images/ \
    --model  mask_rcnn_R_50_FPN_3x \
    --config configs/segmentation/mask_rcnn_R_50_FPN_3x.yaml
```

---

## Predict on Video

Per-frame detection on a video file or webcam, with boxes drawn over each frame.
Requires OpenCV (`pip install opencv-python`).

```bash
python examples/predict_video.py                          # downloads sample highway clip
python examples/predict_video.py --video myclip.mp4
python examples/predict_video.py --video 0                # webcam
python examples/predict_video.py --max-frames 60          # quick smoke test
```

The annotated output is written to `examples/outputs/highway_detected.mp4` by default.

---

## Fine-tune

Fine-tune any architecture on a COCO-format dataset. Supports detection,
segmentation, and keypoint configs — just pass the right `--config`.

Dataset layout (Roboflow export or any COCO split):

```
train/
    image1.jpg
    _annotations.coco.json
valid/
    image2.jpg
    _annotations.coco.json
```

```bash
# Detection
python examples/finetune.py \
    --config  configs/detection/faster_rcnn_R_50_FPN_3x.yaml \
    --train   /data/myproject/train \
    --val     /data/myproject/valid \
    --weights faster_rcnn_R_50_FPN_3x

# Segmentation
python examples/finetune.py \
    --config  configs/segmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --train   /data/myproject/train \
    --val     /data/myproject/valid \
    --weights mask_rcnn_R_50_FPN_3x

# Keypoints
python examples/finetune.py \
    --config  configs/keypoints/keypoint_rcnn_R_50_FPN_3x.yaml \
    --train   /data/myproject/train \
    --val     /data/myproject/valid \
    --weights keypoint_rcnn_R_50_FPN_3x
```

Key flags: `--iters` (default 3000), `--batch` (default 2), `--lr` (default 1e-3),
`--eval-period` (default 500), `--device`, `--output`.

---

## Evaluate

Run COCO mAP on a validation set. Same dataset layout as fine-tune.

```bash
# Eager checkpoint
python examples/evaluate.py \
    --weights runs/finetune/model_final.pth \
    --config  configs/detection/faster_rcnn_R_50_FPN_3x.yaml \
    --val     /data/myproject/valid

# Hybrid eval — measure the deployed artifact's accuracy, not just the eager model
python examples/evaluate.py --weights ... --val ... \
    --backbone-onnx examples/outputs/model.onnx

python examples/evaluate.py --weights ... --val ... \
    --backbone-mlpackage examples/outputs/model.mlpackage   # macOS only
```

---

## Export

Export the backbone + FPN to a deployment format and verify numerical parity.

```bash
# ONNX (requires: pip install mayaku[onnx] + onnxruntime)
python examples/export_onnx.py
python examples/export_onnx.py --model faster_rcnn_R_101_FPN_3x

# CoreML — macOS only (requires: pip install mayaku[coreml])
python examples/export_coreml.py

# OpenVINO — best on Intel CPUs (requires: pip install mayaku[openvino])
python examples/export_openvino.py
python examples/export_openvino.py --compress-fp16        # half-size .bin

# TensorRT — CUDA Linux/Windows only (requires: pip install mayaku[tensorrt])
python examples/export_tensorrt.py
python examples/export_tensorrt.py --fp16                 # ~2x throughput
```

## Run an Exported Artifact

Use the exported backbone for inference, with eager PyTorch handling RPN/NMS/heads.
This is the canonical deployment pattern.

```bash
# After export_onnx.py
python examples/run_onnx.py
python examples/run_onnx.py --providers CUDAExecutionProvider,CPUExecutionProvider

# After export_coreml.py — macOS only
python examples/run_coreml.py
python examples/run_coreml.py --compute-units ALL         # ANE + GPU
```

---

## Outputs

All scripts write to `examples/outputs/` by default (gitignored).
Pass `--output` to redirect anywhere.
