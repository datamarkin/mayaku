# RF100-VL — AP vs wall-clock

Train each library with its **defaults** on the [RF100-VL](https://rf100-vl.org/)
datasets and measure **COCO AP as a function of training wall-clock** — the
learning curve. Currently: **YOLO (Ultralytics)**, **RF-DETR**, and **Mayaku**.

The one claim: *run the defaults, here's the AP you reach for a given time budget.*

## Requirements

Each leg needs its own library — install only the one you're running:

```bash
pip install ultralytics       # YOLO leg (train_yolo.py / eval_yolo.py)
pip install rfdetr            # RF-DETR leg (train_rfdetr.py / eval_rfdetr.py)
pip install -e .              # Mayaku leg — from the repo root (train_mayaku.py / eval_mayaku.py)
pip install pycocotools matplotlib   # scoring + plotting (all legs)
```

The scripts fail fast with an install hint if their library is missing, so you
can run one leg without the other's dependency.

## Principles

- **Defaults only.** Each `train_*.py` is just the library's own one-call training
  API with its stock recipe — its own default epoch count included; we don't even
  pass `epochs`. The single added argument is checkpoint cadence (per-epoch),
  because a curve is impossible without it. Everything else — data conversion,
  scoring, plotting — is a helper that lives *outside* the training file and runs
  around or after it. We never patch a library.
- **Wall-clock is wall-clock.** Whatever a library's default loop costs — its own
  per-epoch validation included — is its honest time. We don't disable it.
- **One metric.** Every checkpoint is scored by the *same* pycocotools evaluator on
  the same COCO val split — never a library's internal mAP — so the AP numbers are
  identical in definition across libraries.
- **Eval is offline.** Our scoring runs after `train()` returns, never inside the
  training loop, so it never touches the measured wall-clock.

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
**loops over every dataset subdir** — that is the intended interface.

## Workflow — library by library

Run one library at a time (train, then score, then move on). Both phases are
**resumable**: a dataset with a `meta.json` is skipped by `train`, and one with a
`curve.csv` is skipped by `eval`. So you can run a subset, Ctrl-C, or add datasets
freely — no need to do everything at once.

```bash
# YOLO
python train_yolo.py   --datasets <coco_root>
python eval_yolo.py    --datasets <coco_root>

# RF-DETR
python train_rfdetr.py --datasets <coco_root>
python eval_rfdetr.py  --datasets <coco_root>

# Mayaku
python train_mayaku.py --datasets <coco_root>
python eval_mayaku.py  --datasets <coco_root>

# aggregate all legs into the figure + table
python plot.py --results results --budgets 60,300,900,1800
```

No `--epochs`: each library uses its own default schedule. `--device` is optional
(defaults to each library's auto-detect); pass e.g. `--device 0` for YOLO or
`--device cuda` for RF-DETR to pin a GPU.

## Outputs

```
results/
  yolo/<dataset>/{weights/epoch*.pt, meta.json, curve.csv}
  rfdetr/<dataset>/{checkpoint_*.ckpt, meta.json, curve.csv}
  mayaku/<dataset>/{train/model_iter_*.pth, meta.json, curve.csv}
  summary.csv   # AP at each time budget, per library
  curve.png     # aggregate mean AP vs wall-clock, one line per library
```

`curve.csv` is one `(checkpoint, wall_clock_s, ap, ap50, n_dets)` row per
checkpoint. **Wall-clock is the checkpoint file's mtime minus the `t0` stamped in
`meta.json` at train start** — library-agnostic, no timing callbacks. Curves are
aligned across libraries *after the fact* by interpolation in `plot.py`, so the
checkpoints never need to land at matching times.

## How the pieces map

| file | role |
|---|---|
| `train_{yolo,rfdetr,mayaku}.py` | **just the default training loop** — nothing tuned |
| `eval_{yolo,rfdetr,mayaku}.py` | offline pycocotools scoring of the checkpoints (helpers) |
| `common.py` | dataset discovery, COCO→YOLO prep, val lookup, scorer, curve/meta I/O |
| `plot.py` | interpolate → aggregate across datasets → `summary.csv` + `curve.png` |

## Storage

Checkpoints are kept per run (single seed × nano models is modest). If you ever
need to reclaim space, delete a library's checkpoints after its `eval` has written
the `curve.csv` — the curve is the durable artifact.

## Notes / to validate on the first real run

- **RF-DETR checkpoints** are PyTorch-Lightning `.ckpt` (no `model` key);
  `rfdetr.py:_to_rfdetr_weights` converts them and RF-DETR auto-resizes its head.
  It fails loud if RF-DETR's module layout changes — smoke-test it on the first
  checkpoint you produce.
- **RF-DETR EMA:** eval uses the base weights in each checkpoint. RF-DETR ships the
  *EMA* weights as its headline model, so this slightly understates it. The EMA
  weights live in the checkpoint's callback state — switch to them if that gap
  matters.
- **Mayaku model source:** fine-tunes from the pretrained **weights** (`WEIGHTS` in
  `train_mayaku.py` — a bundled name or a local `.pth`), *not* a config YAML. The
  checkpoint is self-describing (architecture comes from it) and auto-config derives
  the recipe — same as `yolo11n.pt` / `RFDETRNano()`. The one added argument is
  checkpoint cadence, `overrides={"solver": {"checkpoint_period": 1}}` (per-epoch),
  matching `save_period` / `checkpoint_interval` on the other legs.
- **Class-id alignment:** all three legs map model class index `i` → the i-th COCO
  category id (ascending). This holds because every trainer sees categories in that
  order.
- **Score threshold at eval** is `0.001` on all legs so pycocotools sees the full PR
  curve (Mayaku bakes `0.05`, so its eval lowers `model.score_thresh`). Eval-side
  only — training is untouched.
