# Examples

Minimal scripts — read in 30 seconds, adapt by editing the file.
No argparse: to switch model or input, edit the literal in the script.
Every script auto-downloads weights and the sample input on first run.

## Quickstart

```bash
pip install mayaku
python examples/predict.py
```

That's the whole flow — model downloaded, sample image fetched, detections printed.
Same shape for the segmentation and keypoint variants:

| Script | Architecture | Default model |
|---|---|---|
| `predict.py` | Faster R-CNN | `faster_rcnn_R_50_FPN_3x` |
| `predict_segmentation.py` | Mask R-CNN | `mask_rcnn_R_50_FPN_3x` |
| `predict_keypoint.py` | Keypoint R-CNN | `keypoint_rcnn_R_50_FPN_3x` |

To swap to another zoo model, edit the `Predictor.from_pretrained("...")` line.
Every name from `python -c "from mayaku import configs; print(*configs.list_all(), sep='\n')"` works.

## Inference

| Script | What it does |
|---|---|
| `predict.py` | Detect on one image |
| `predict_segmentation.py` | Detect + instance masks on one image |
| `predict_keypoint.py` | Detect humans + 17 COCO keypoints |
| `batch_predict.py` | Detect on every image in a directory; one JSON per input. Run as `python examples/batch_predict.py /path/to/images` |
| `predict_video.py` | Per-frame detection on a video; writes an annotated mp4. Optional `python examples/predict_video.py myclip.mp4` — defaults to downloading a sample highway clip. Requires `pip install opencv-python`. |

## Export to a deployment format

Each script exports backbone+FPN and verifies numerical parity vs eager.
Edit the model-name literal to switch architectures.

| Script | Target | Extra to install |
|---|---|---|
| `export_onnx.py` | `.onnx` | `pip install mayaku[onnx]` |
| `export_coreml.py` | `.mlpackage` (macOS only) | `pip install mayaku[coreml]` |
| `export_openvino.py` | `.xml` + `.bin` | `pip install mayaku[openvino]` |
| `export_tensorrt.py` | `.engine` (CUDA host only) | `pip install mayaku[tensorrt]` |

RPN, NMS, mask paste, and keypoint decode stay in Python — same split for all four targets. See [`docs/export/`](../docs/export/) for the rationale.

## Run an exported artifact

After exporting, swap the eager backbone for the runtime artifact while keeping PyTorch on RPN/NMS/heads. This is the canonical deployment pattern.

| Script | Pairs with |
|---|---|
| `run_onnx.py` | `export_onnx.py` (cross-platform via ONNX Runtime) |
| `run_coreml.py` | `export_coreml.py` (macOS only, runs on ANE/GPU) |

## Fine-tune + evaluate

These two keep argparse — they're real workflows, not minimal demos, and the flags map directly to your dataset / training-loop choices.

```bash
# Fine-tune on a Roboflow / COCO-format dataset
python examples/finetune.py \
    --config  configs/detection/faster_rcnn_R_50_FPN_3x.yaml \
    --train   /data/myproject/train \
    --val     /data/myproject/valid \
    --weights faster_rcnn_R_50_FPN_3x
```

Supports detection, segmentation, and keypoint configs — pick the matching `--config` + `--weights` from the [pre-converted models table](../README.md#pre-converted-models).

Dataset layout (Roboflow export or any COCO split):

```
train/
    image1.jpg
    _annotations.coco.json
valid/
    image2.jpg
    _annotations.coco.json
```

Key flags: `--iters` (default 3000), `--batch` (default 2), `--lr` (default 1e-3),
`--eval-period` (default 500), `--device`, `--output`.

```bash
# Run COCO mAP on a checkpoint
python examples/evaluate.py \
    --weights runs/finetune/model_final.pth \
    --config  configs/detection/faster_rcnn_R_50_FPN_3x.yaml \
    --val     /data/myproject/valid
```

Hybrid eval — measure the deployment artifact's accuracy, not the eager model:

```bash
python examples/evaluate.py --weights ... --val ... \
    --backbone-onnx examples/outputs/model.onnx

python examples/evaluate.py --weights ... --val ... \
    --backbone-mlpackage examples/outputs/model.mlpackage   # macOS only
```

## Outputs

Inference + export scripts write to `examples/outputs/` (gitignored).
The fine-tune script defaults to `runs/finetune/`.

## See also

- [Main README](../README.md) — install, full model zoo, library API.
- `mayaku predict <model-name> <image>` — flag-based CLI for users who don't want to write Python.
