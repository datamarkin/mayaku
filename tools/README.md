# `tools/` — throwaway one-off scripts

Scripts in this directory are *not* part of the installed `mayaku`
package. They exist to support one-time engineering tasks (validation,
data prep, etc.) and are deleted once their job is done. None of them
are documented in user-facing docs.

## `convert_convnext_backbone.py`

**Purpose.** Convert an external ConvNeXt backbone checkpoint (timm,
facebookresearch / Liu et al., or torchvision) into a Mayaku-format
backbone `state_dict`, so a maintainer can warm-start a from-scratch
train from a public ImageNet-pretrained backbone.

**Why this exists.** The `mayaku` library loads *mayaku* weights, period
— it has no knowledge of timm / facebookresearch / torchvision as
*weight sources* (no auto-download, no runtime format remapping). This
script is the one place that knows those external key layouts. It lifts
the (now-deleted) in-library remap out to `tools/`, where it belongs.

**Format support.** Auto-detected from key naming: timm
(`stages.N.blocks.Y.…`, MLP as 1×1 Conv2d → squeezed to Linear),
facebookresearch (`downsample_layers.k`, `stages.k.j.dwconv`,
`gamma (C,) → layer_scale (C,1,1)`), and torchvision (`features.k.…`,
already-`(C,1,1)` `layer_scale`). Classification-head keys (`head.*` /
`norm.*` / `classifier.*`) are dropped automatically.

**Verification.** The tool builds `ConvNeXtBackbone(variant)` and does a
strict load before writing — a wrong `--variant` or unsupported layout
fails loudly. Round-trip parity is exact (max|Δ|=0.0) for timm
(`convnext_femto`) and torchvision (`convnext_tiny`) forward outputs.

### Usage

```bash
python tools/convert_convnext_backbone.py convnext_femto.d1_in1k.pth \
    --variant convnext_femto \
    -o convnext_femto.mayaku.pth
```

The output is a bare backbone `state_dict` (keys match
`ConvNeXtBackbone(variant).state_dict()`) — the format the maintainer-only
from-scratch backbone-init path consumes. Input may be
`.pth` / `.pt` / `.bin` or `.safetensors`.

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

## `harvest_commons.py`

**Purpose.** Build a raw image pool from Wikimedia Commons that can be
labelled into an Objects365-style detection set: everyday scenes with
countable objects, ≥1500px on the shortest side, under licenses whose
obligations an Apache-2.0 model release can satisfy.

**Why this exists.** Commons has ~120M files but almost all of the bulk
is maps, scans, heraldry and herbarium sheets, and its most common
license is CC BY-SA. Taking the corpus as it comes yields neither the
content nor the license terms this project needs, so the script filters
on all three axes — license, resolution, subject — and steers the crawl
with an editable category plan rather than crawling breadth-first.

**License policy.** Accepts CC0, public domain, CC BY and CC BY-SA.
Rejects any NC/ND, GFDL-only, and anything carrying a `Restrictions`
flag (trademark, personality rights). Share-alike is included on
purpose — it is the most common license on Commons and excluding it
costs well over half the corpus; reverting that is a two-token edit at
`_SLUG_DENY`/`_TEXT_DENY`. Attribution is still owed on BY and BY-SA:
`manifest.jsonl` records author, credit, license and source URL per
image and must travel with the dataset.

**Politeness.** Hard-coded to the [robot policy](https://wikitech.wikimedia.org/wiki/Robot_policy):
2 concurrent connections to `upload.wikimedia.org`, 20 Mbps ceiling,
serial metadata requests, `Retry-After` honoured, 15-minute pause on
5xx. Refuses to start without a contact-bearing User-Agent. Expect
~8-12 img/s, i.e. ~1M images/day, from one external IP.

**Resumable.** State is a sqlite DB under the output directory; a
re-run skips every file already downloaded or already judged, and
dedups on the Commons-supplied sha1.

### Usage

```bash
UA="mayaku-harvest/1.0 (https://github.com/datamarkin/mayaku; you@example.com)"

# 1. dump the built-in plan and edit it
python tools/harvest_commons.py --dump-plan plan.json

# 2. sample each entry's yield before committing to a long crawl
python tools/harvest_commons.py --plan plan.json --estimate --user-agent "$UA"

# 3. crawl — full-resolution originals, 1500px shortest side as a floor
python tools/harvest_commons.py --plan plan.json --out /data/commons \
    --min-side 1500 --fetch original --user-agent "$UA"
```

**Budget.** Originals of qualifying files run ~4.1MB median / ~5.0MB
mean at a median 12MP. For a 2M-image pool that is ~10TB and, at the
20 Mbps policy ceiling, ~46 days from one external IP — bandwidth, not
the API, is what binds. `--fetch thumb` cuts bytes ~4x but caps
resolution at `--min-side`; `--resize-to PX` downscales on arrival,
trading resolution for disk while still paying full download cost.
Neither is on by default. To go genuinely faster, run inside
Toolforge/WMCS, which is exempt from the rate limits.
