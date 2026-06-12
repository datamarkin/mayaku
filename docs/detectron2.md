# Coming from Detectron2

Mayaku is its own detector (UniQuery + ConvNeXt), not a Detectron2 fork. But the
R-CNN family Mayaku ships (Faster / Mask / Keypoint R-CNN on R-50 / R-101 / X-101 FPN)
is architecture-compatible with Detectron2's, so you can **convert D2 checkpoints once**
and keep using your weights — now with Apple Silicon training, clean exports, and no
custom CUDA kernels.

## Convert a checkpoint

```bash
# D2 .pkl  ->  Mayaku .pth   (one-shot, no network)
python tools/convert_d2_checkpoint.py your_model_final.pkl -o your_model.pth

# Use it like any Mayaku checkpoint
mayaku predict --weights your_model.pth image.jpg --device mps
```

Covers Faster / Mask / Keypoint R-CNN with R-50 / R-101 / X-101_32x8d FPN. Head-specific
rename rules are inert when the source `.pkl` doesn't contain them, so a Faster R-CNN
checkpoint converts cleanly without flags. Full rename table and edge cases:
[`tools/README.md`](../tools/README.md).

If your D2 setup uses anything Mayaku doesn't ship (DCN, Cascade, Panoptic, DETR,
ViTDet, PointRend, DensePose), the converter can't help — those architectures aren't
implemented.

## Parity

Loading and **evaluating** D2's converged model-zoo weights in Mayaku reproduces D2's
published COCO val2017 numbers within ±0.1 AP. Maximum observed gap: **+0.08 AP**.

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

Full per-checkpoint table (incl. 1x configs): [`docs/d2_parity_report.md`](d2_parity_report.md).
These numbers come from evaluating D2's converged weights — not training from scratch.

## Two things that will silently bite you

Both produce **wrong detections with no error raised**, so they're worth stating plainly:

1. **Channel order is RGB, not BGR.** D2 inherits Caffe2's BGR convention; Mayaku is
   RGB-native ([ADR 002](decisions/002-rgb-native-image-ingestion.md)). Feed a `cv2.imread`
   BGR array and it runs but detects wrong. Load with `mayaku.utils.image.read_image`
   (PIL, RGB) or swap channels at the boundary.

2. **Pixel mean/std are RGB-ordered** — `[123.675, 116.280, 103.530]` / `[58.395, 57.120, 57.375]`.
   If you copy a D2 yacs config that pins `PIXEL_MEAN` in BGR, the model normalises with
   channels swapped. Don't override the defaults unless your dataset needs it — and write
   them RGB.
