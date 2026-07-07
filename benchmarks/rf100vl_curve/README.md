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
python run_yolo.py   --datasets <coco_root>                  # server A
python run_mayaku.py --datasets <coco_root>                  # server A
python run_rfdetr.py --datasets <coco_root> --device cuda    # server B (RF-DETR is the slow leg)

python plot.py --results results --budgets 60,300,900,1800   # aggregate → figures + table
```

RF-DETR is far slower than YOLO or Mayaku, so run it on its own machine while
YOLO+Mayaku share the other — no two legs ever share a GPU, so wall-clock stays
honest. `--variants nano,small` runs a subset (default: all four), handy for
splitting variants across more machines.

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
  curve_<variant>.png   # aggregate mean AP vs wall-clock, one line per library
  summary.csv           # AP at each time budget, per (variant, library)
```

`curve.csv` is one row per checkpoint: `checkpoint, wall_clock_s`, then all **12
COCO stats** (`ap, ap50, ap75, ap_small, ap_medium, ap_large, ar1, ar10, ar100,
ar_small, ar_medium, ar_large` — pycocotools order), then `n_dets`. `plot.py`
interpolates each dataset's curve onto a shared per-variant time grid and averages
across datasets, producing **one figure per size class** (nano and large live on
very different time scales, so they don't share a grid). Curves are aligned across
libraries *after the fact*, so checkpoints never need to land at matching times.

## Resumability

Every leg is resumable: a `(variant, dataset)` with a `curve.csv` is skipped; one
trained but not yet scored is scored without retraining. So you can run a subset,
Ctrl-C, or add datasets freely. Because checkpoints are purged per dataset once its
`curve.csv` exists, only one dataset's checkpoints ever sit on disk at a time —
`curve.csv` is the durable artifact.

## How the pieces map

| file | role |
|---|---|
| `run_{yolo,rfdetr,mayaku}.py` | **the whole leg** — variants × datasets, default train → shared scorer → purge |
| `common.py` | dataset discovery, COCO→YOLO prep, val lookup, pycocotools scorer, curve/meta I/O |
| `plot.py` | interpolate → aggregate across datasets → per-variant `curve_*.png` + `summary.csv` |
| `plot_one.py`, `plot_series.py` | ad-hoc single-curve / hand-picked-series plots (take explicit `curve.csv` paths) |

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
