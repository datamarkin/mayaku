# `tools/` — throwaway one-off scripts

Scripts in this directory are *not* part of the installed `mayaku`
package. They exist to support one-time engineering tasks (validation,
data prep, etc.) and are deleted once their job is done. None of them
are documented in user-facing docs.

## `convert_d2_checkpoint.py`

**Purpose.** Convert a Detectron2 model-zoo checkpoint (e.g.
`faster_rcnn_R_50_FPN_3x` from `MODEL_ZOO`) into Mayaku's state_dict
layout, so that the existing `mayaku eval --weights` / `mayaku train
--weights` flag can consume it.

**Why this exists.** As a one-time correctness validation of Mayaku's
ResNet detector. If we can load Detectron2's converged R50-FPN weights
and reproduce its published 40.2 box AP within ±1.0 on COCO val2017,
the architecture, ROI heads, RPN, FPN, and inference pipeline are
end-to-end correct. After that result is recorded in
`docs/decisions/003-resnet-engine-validated-against-d2.md`, the script
has done its job.

**Architecture support.** Faster / Mask / Keypoint R-CNN with R-50,
R-101, and X-101_32x8d FPN backbones. The rename table covers mask head
(`mask_fcn{N}`/`deconv`/`predictor`) and keypoint head
(`conv_fcn{N}`/`score_lowres`) keys; head rules are inert when the source
`.pkl` doesn't contain them, so a Faster R-CNN checkpoint converts
cleanly with the same script. ResNeXt-101 shares the bottleneck layout
with ResNet-101; group convolutions affect tensor shape but not key
names, so no extra rule is needed.

**Deletion plan.** Drop this script and `tools/` itself in the same PR
that lands phase A of `DINOV2_IMPLEMENTATION.md` (which removes
`src/mayaku/models/backbones/resnet.py`). At that point the only
architecture this script supports has been excised from the codebase
and the script can no longer be used regardless.

### Usage

```bash
# 1. Download a Detectron2 model-zoo .pkl (no network during conversion).
#    https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
curl -LO https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl

# 2. Convert.
python tools/convert_d2_checkpoint.py model_final_b275ba.pkl \
    -o model_final.mayaku.pth

# 3. Use the converted .pth like any other Mayaku checkpoint.
mayaku eval configs/coco/detection/faster_rcnn_R_50_FPN_3x.yaml \
    --weights model_final.mayaku.pth \
    --json /data/coco/annotations/instances_val2017.json \
    --images /data/coco/val2017
```

For batch conversion of the full 12-checkpoint set, see
`tools/convert_all_d2.sh` and `tools/d2_model_zoo.tsv`.

### Channel order

The default `--channel-order bgr` reverses the input channel dimension
of `conv1` to match Mayaku's RGB-native ingestion (ADR 002). This is
correct for every `.pkl` published in Detectron2's MODEL_ZOO. Pass
`--channel-order rgb` only if you have a checkpoint someone trained
with `INPUT.FORMAT="RGB"`.

### Pickle safety

`.pkl` files run arbitrary code on `pickle.load`, so the script
**only** loads files whose top-level object is a dict containing a
`"model"` key whose value is itself a dict mapping `str → numpy.ndarray`
— anything else is rejected before any further work. This matches the
exact shape Detectron2 emits. It is *not* a substitute for trusting
the source: only run this on `.pkl` files you obtained from
Detectron2's official MODEL_ZOO URLs.
