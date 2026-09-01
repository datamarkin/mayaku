# Coming from Detectron2

Mayaku is its own detector (UniQuery + ConvNeXt), not a Detectron2 fork. But the
R-CNN family Mayaku ships (Faster / Mask / Keypoint R-CNN on R-50 / R-101 / X-101 FPN)
is architecture-compatible with Detectron2's, so you can **convert D2 checkpoints once**
and keep using your weights — now with Apple Silicon training, clean exports, and no
custom CUDA kernels.

## Convert a checkpoint

You need two files: the weights, and the `cfg.yaml` they were trained with.

```bash
mayaku convert-d2 your_model_final.pth \
    --d2-config cfg.yaml \
    --output your_model.pth \
    --class-names cat,dog          # optional; defaults to class_0, class_1, …
```

Same thing without the CLI, e.g. from your own pipeline:

```python
from mayaku.d2 import convert_d2

convert_d2("your_model_final.pth", d2_config="cfg.yaml",
           output="your_model.pth", class_names=["cat", "dog"])
```

The output is a **deploy-ready** checkpoint carrying the embedded sidecar, so it needs
no config file afterwards:

```python
from mayaku import from_pretrained

predictor = from_pretrained("your_model.pth", device="auto")
instances = predictor("image.jpg")
```

Covers Faster / Mask / Keypoint R-CNN with R-50 / R-101 / X-101_32x8d FPN, from a `.pkl`
(model zoo) or a `.pth` (your own training run). No `detectron2` install needed.

**`--d2-config` is required, and that is deliberate.** Two settings change predictions
without changing any tensor shape, so a strict weight load cannot catch them and they
have to be read from the config rather than typed by hand:

| setting | why it bites |
|---|---|
| `MODEL.PIXEL_STD` | caffe2-pretrained checkpoints use `[1, 1, 1]`; Mayaku defaults to `[58.395, 57.12, 57.375]`. Wrong by ~58x, no error. |
| `MODEL.RESNETS.STRIDE_IN_1X1` | caffe2 puts the stride on the 1x1 conv. Both layouts have identical parameter shapes. |

`INPUT.FORMAT` is read too, so the BGR→RGB channel swap happens automatically — there
is no flag to get backwards.

Conversion prints every value it carried over that differs from Mayaku's defaults, so
what the config contributed is visible rather than implicit.

### What it refuses

Anything Mayaku cannot reproduce **exactly** raises rather than converting to a
near-equivalent that predicts differently: non-`GeneralizedRCNN` architectures,
deformable conv, Cascade ROI heads, `ROIAlign` (v1) pooling, GN-normalised box/mask
heads, sigmoid/federated classification loss, ResNet depths other than 50/101, and
`MASK_ON` + `KEYPOINT_ON` together.

### Class names

Detectron2 keeps class and keypoint *names* in `MetadataCatalog`, registered at training
time — they are in neither the checkpoint nor `cfg.yaml`. Pass `--class-names` to record
them; without it the model still runs correctly, just with placeholder names. Keypoint
`flip_indices` are likewise unavailable; inference never uses them, a later fine-tune
with horizontal flip does.

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

**Near-parity, not bit-parity.** `ROIPooler` resolves `POOLER_SAMPLING_RATIO=0` to a fixed
2 samples per bin rather than D2's per-box adaptive `ceil(roi_size / output_size)`, because
the export path samples a fixed grid and the two must agree. Per-box coordinates therefore
differ slightly from D2's — on a 112-landmark keypoint model, ~1 px median on a ~800 px
object. It is a difference, not an error: measured against hand-placed ground truth the
converted model scored marginally *better* than the Detectron2 original.

## One thing that will silently bite you

Feeding **BGR pixels at inference**. D2 inherits Caffe2's BGR convention; Mayaku is
RGB-native ([ADR 002](decisions/002-rgb-native-image-ingestion.md)). Pass a `cv2.imread`
array and it runs and detects wrong, with no error. Load with
`mayaku.utils.image.read_image` (PIL, RGB) or swap channels at the boundary.

The *checkpoint's* channel order is handled for you — `convert-d2` reads `INPUT.FORMAT`
and folds the swap into the stem conv's weights, so there is no runtime flag to set. Same
for `PIXEL_MEAN` / `PIXEL_STD`: they are read from `cfg.yaml` and reordered to RGB. Don't
hand-write them.
