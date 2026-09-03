# RF100-VL — AP vs wall-clock

Train each library with its **defaults** on the [RF100-VL](https://rf100-vl.org/)
datasets and measure **COCO AP as a function of training wall-clock** — the
learning curve. Three libraries — **YOLO (Ultralytics)**, **RF-DETR**, and
**Mayaku** — each in four size variants: **nano, small, medium, large**.

The one claim: *run the defaults, here's the AP you reach for a given time budget.*

## One script per library

Each `run_<lib>.py` is a single self-contained script — read it top to bottom and
you can see the whole leg is fair. For every variant and every dataset it does the
same three steps in the same order:

1. **Train** with the library's own one-call training API and its stock recipe —
   its own default epoch count included; we don't even pass `epochs`. The only added
   argument is checkpoint cadence (per-epoch), because a curve is impossible without
   it.
2. **Score** every checkpoint with the *same* pycocotools evaluator on the same COCO
   val split — never a library's internal mAP — so the AP numbers are identical in
   definition across libraries.
3. **Purge** that (variant, dataset)'s checkpoints once its `curve.csv` is written,
   so 100 datasets never need 100 datasets of checkpoints on disk at once.

There is no separate train/eval step and no library-specific branch: the only
things that differ between the three scripts are the model names and each library's
own `predict` call.

```bash
python prefetch_weights.py all                              # fetch every pretrained weight first
python run_all.py --datasets <coco_root> --legs mayaku,yolo,rfdetr

python plot.py --intersection --band                        # aggregate → figures + tables
```

`run_all.py` is the driver. It walks every (leg, variant, dataset) **one unit at a
time on a single machine**, each unit an isolated subprocess running the stock
`run_<leg>.py` unpatched. Sequential is the point: no two legs ever share the GPU, so
the wall-clock axis stays honest. A crashed unit sinks only itself, and any unit with
a `curve.csv` is skipped on re-run, so Ctrl-C and resume freely. `--legs` (default
`yolo,rfdetr`) and `--variants` (default all four) run subsets; per-unit output goes
to `results/logs/<leg>/<variant>/<dataset>.log`. Pin the GPU with
`CUDA_VISIBLE_DEVICES` in the environment — every child inherits it.

To leave it running for days, `train_large.sh` is the unattended wrapper: prefetch in
the foreground, then detach the driver under `nohup`.

Each leg also runs on its own if you prefer to drive it directly:

```bash
python run_yolo.py   --datasets <coco_root>
python run_mayaku.py --datasets <coco_root>
python run_rfdetr.py --datasets <coco_root> --device cuda
```

## Variants

| size | YOLO | RF-DETR | Mayaku |
|---|---|---|---|
| nano | `yolo26n.pt` | `RFDETRNano` | `mayaku-n-det` |
| small | `yolo26s.pt` | `RFDETRSmall` | `mayaku-s-det` |
| medium | `yolo26m.pt` | `RFDETRMedium` | `mayaku-m-det` |
| large | `yolo26l.pt` | `RFDETRLarge` | `mayaku-l-det` |

Each is the library's stock pretrained model for that size; auto-batch (YOLO),
per-variant defaults (RF-DETR), and auto-config (Mayaku) handle everything else.

## Requirements

```bash
pip install ultralytics rfdetr        # YOLO + RF-DETR legs
pip install -e .                      # Mayaku leg — from the repo root
pip install pycocotools matplotlib    # scoring + plotting (all legs)
```

Each `run_<lib>.py` imports only its own library and fails fast with an install hint
if it is missing, so you can run one leg without the others' dependencies.

## Principles

- **Defaults only.** Each script calls the library's own training API with its stock
  recipe. The single added argument is per-epoch checkpoints. We never patch a
  library or pass a tuned hyperparameter.
- **Wall-clock is wall-clock.** Whatever a library's default loop costs — its own
  per-epoch validation included — is its honest time. We don't disable it. Wall-clock
  per checkpoint is the checkpoint file's mtime minus the `t0` stamped in `meta.json`
  at train start — library-agnostic, no timing callbacks.
- **One metric.** Every checkpoint is scored by the *same* pycocotools evaluator on
  the same COCO val split, at score threshold `0.001` on all legs so COCOeval sees
  the full PR curve.
- **Default deploy weights.** All three legs are scored at each library's own default
  deploy model: YOLO's EMA (`YOLO(ckpt)` loads `ckpt["ema"]`), RF-DETR's EMA
  (`use_ema=True`; extracted from each checkpoint's EMA-callback state), and Mayaku's
  EMA shadow. No leg is handicapped against the others.
- **Eval is offline.** Scoring runs after `train()` returns, never inside the training
  loop, so it never touches the measured wall-clock.

## Dataset layout

One **COCO root**, one subdir per dataset (Roboflow COCO export):

```
<coco_root>/
  <dataset-a>/
    train/_annotations.coco.json + images
    valid/_annotations.coco.json + images   # val/ or test/ also accepted
  <dataset-b>/
    ...
```

This one root is the single source of truth: RF-DETR reads it directly, YOLO gets a
converted `_yolo/` cache written next to each dataset, and both are scored against
the COCO `valid` ground truth. You pass `--datasets <coco_root>` and each script
loops over every dataset subdir.

## Outputs

```
results/
  yolo/<variant>/<dataset>/{weights/epoch*.pt (until purged), meta.json, curve.csv}
  rfdetr/<variant>/<dataset>/{checkpoint_*.ckpt (until purged), meta.json, curve.csv}
  mayaku/<variant>/<dataset>/{train/ (until purged), meta.json, curve.csv}
  curve_<variant>.png   # mean AP vs mean wall-clock, one line per library
  curve_points.csv      # every progress index: mean wall-clock, mean AP, n_datasets
  summary.csv           # per (variant, library): dataset/checkpoint counts, mean and
                        # median total time, mean and median final AP, AP at each budget
```

`curve.csv` is one row per checkpoint: `checkpoint, wall_clock_s`, then all **12
COCO stats** (`ap, ap50, ap75, ap_small, ap_medium, ap_large, ar1, ar10, ar100,
ar_small, ar_medium, ar_large` — pycocotools order), then `n_dets`. `plot.py`
aligns datasets by **training progress** rather than by wall-clock: each dataset's
checkpoint list is prepended with the origin (wall-clock 0, AP 0) and sampled at
evenly spaced fractions of its checkpoint index, so index 0 is the origin and the
last index is that dataset's final checkpoint. Point *j* of a library's curve is
then the mean across its datasets of the wall-clock at index *j* and of the AP at
index *j*. Every point therefore averages the same datasets, and the last point is
(mean total training time, mean final-checkpoint AP). A library writing exactly
`--points` checkpoints per run is used without interpolation; one writing fewer
(Mayaku's cadence is dataset-dependent, 20–30) is linearly interpolated onto the
same index axis. One figure per size class, since nano and large live on very
different time scales.

Libraries finish different numbers of datasets while a benchmark is still running.
`--intersection` restricts every library to the datasets all of them have, so the
means cover identical data; without it each library is averaged over its own set.

## Resumability

Every leg is resumable: a `(variant, dataset)` with a `curve.csv` is skipped; one
trained but not yet scored is scored without retraining. So you can run a subset,
Ctrl-C, or add datasets freely. Because checkpoints are purged per dataset once its
`curve.csv` exists, only one dataset's checkpoints ever sit on disk at a time —
`curve.csv` is the durable artifact.

## How the pieces map

| file | role |
|---|---|
| `run_all.py` | **the driver** — every (leg, variant, dataset) sequentially, one isolated subprocess per unit, resumable |
| `run_{yolo,rfdetr,mayaku}.py` | **the whole leg** — variants × datasets, default train → shared scorer → purge |
| `common.py` | dataset discovery, COCO→YOLO prep, val lookup, pycocotools scorer, curve/meta I/O |
| `prefetch_weights.py` | pre-download every pretrained weight so no training unit stalls on a fetch |
| `train_large.sh` | unattended wrapper: prefetch, then detach the driver for the large variant |
| `plot.py` | align by training progress → mean curve per library → `curve_*.png`, `curve_points.csv`, `summary.csv` |

## Notes / to validate on the first real run

- **RF-DETR EMA extraction:** `run_rfdetr.py:_to_rfdetr_weights` pulls the EMA weights
  out of each per-epoch PyTorch-Lightning `.ckpt` (from the EMA callback's
  `average_model_state_dict`, `module.model.` prefix). It fails loud if RF-DETR's
  callback/module layout changes — smoke-test it on the first checkpoint you produce.
- **Mayaku eval cadence:** confirm Mayaku's default loop validates every epoch like
  YOLO and RF-DETR do. If it doesn't, its per-epoch checkpoint mtimes exclude a
  validation cost the other legs' mtimes include — a wall-clock asymmetry to be aware
  of (the scripts don't force an eval period; each library's default stands).
- **Class-id alignment:** all three legs map model class index `i` → the i-th COCO
  category id (ascending). This holds because every trainer sees categories in that
  order.
- **RTX 3060 (12 GB):** the large variants at default batch may be tight. That is the
  library's own auto-batch / default-batch behaviour — part of the honest benchmark,
  not something the scripts override.

## Known-bad units

These `<lib>/<variant>/<dataset>` combinations failed to produce a `curve.csv` and
were skipped on the resume pass. `plot.py --intersection` drops them automatically,
since it keeps only datasets every library has:

```
yolo/large/aerial-sheep
yolo/large/gwhd2021
yolo/large/orgharvest
rfdetr/large/exploratorium-daphnia
rfdetr/large/gwhd2021
rfdetr/large/penguin-finder-seg
```

