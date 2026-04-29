# Architecture

How the source tree is organised and how a forward / backward pass flows
through it. Reading this end-to-end takes ~10 minutes and gives you the
mental model needed to extend any layer of the system.

## Layered view

```
mayaku
├── backends/         # Device abstraction, AMP, op fallbacks (the bottom)
├── structures/       # Boxes, ImageList, Instances, Keypoints, BitMasks
├── config/           # Pydantic schemas + YAML I/O
├── data/             # Catalog, COCO loader, mapper, samplers, transforms
├── models/           # Backbones → necks → proposals → heads → detectors
├── engine/           # Trainer, hooks, optim, evaluator, distributed
├── inference/        # Predictor, postprocess, export/{onnx,coreml,…}
├── cli/              # Typer entry: train / eval / predict / export
└── utils/            # Image I/O and other tiny helpers
```

Dependencies flow downward: `cli` calls `engine` and `inference`, which
call `models` and `data`, which call `structures`, which call `backends`.
Nothing in `backends` imports from anywhere else in `mayaku`. This keeps
the device abstraction (`Device`, `autocast`, the op fallbacks) usable in
isolation when you want to test backend behaviour without dragging the
whole detector graph along.

## The bottom: `backends/`

`mayaku.backends.device.Device` is the single source of truth for "where
does this tensor live and what AMP rules apply to it". It carries:

- `kind: "cpu" | "mps" | "cuda"` — which backend.
- `torch: torch.device` — what to pass to `.to(...)`.
- `supports_amp` and `amp_dtype` — what the AMP wrapper should do.

Use `Device.auto()` to pick the best available device, or
`Device(kind="cuda")` to pin one. Most of the codebase calls `Device.auto()`
once near the top of a script and threads the result through.

`mayaku.backends.amp.autocast(device, enabled, dtype)` is a context
manager that does the right thing for each backend: torch's `autocast`
on CUDA, `torch.cpu.amp.autocast` on CPU, and a pass-through on MPS
(MPS doesn't have a working autocast yet). The trainer uses this
implicitly via `AMPTrainer`; user code rarely touches it directly.

`mayaku.backends.ops` re-exports `roi_align`, `nms`, and `batched_nms`.
Each of them tries the torchvision kernel first and falls back to a
pure-PyTorch implementation only on `NotImplementedError`. On the pinned
versions the fallback never fires in production but is exercised
directly by the test suite. Deformable convolution was dropped (see
[ADR 001](decisions/001-drop-deformable-convolution.md)).

## Geometry: `structures/`

Five small dataclass-style containers shared by every detector:

- `Boxes` — `XYXY_ABS` tensor with the standard helpers (area, clip,
  pairwise IoU). Box format is fixed; there is no rotated-box class.
- `ImageList` — batched image tensor with per-image original sizes and a
  configurable size divisibility. Replaces Detectron2's `ImageList`.
- `Instances` — heterogeneous per-image field bag (boxes, scores, classes,
  masks, keypoints, …). Acts as the normalised return type of every
  detector head and the input to `detector_postprocess`.
- `Keypoints` — `(N, K, 3)` `(x, y, visibility)` tensor with flip-pair
  handling.
- `BitMasks` / `PolygonMasks` — bitmap and polygon mask formats with
  rasterisation helpers.

These are the only shapes that cross module boundaries. Heads return
`Instances`; the trainer consumes `Instances`; the COCO evaluator turns
`Instances` into JSON.

## Configuration: `config/`

`mayaku.config.schemas` defines a tree of pydantic v2 models — frozen,
extra-forbidden, defaults validated. The top-level `MayakuConfig` carries:

- `model: ModelConfig` — meta-architecture, backbone, FPN, anchors, RPN,
  ROI heads, mask/keypoint heads, pixel mean/std, device.
- `input: InputConfig` — train/test resize, mask format, flip.
- `solver: SolverConfig` — optimizer + LR schedule (defaults to the 3x
  schedule).
- `test: TestConfig` — detections per image, eval period.
- `dataloader: DataLoaderConfig` — workers, sampler, aspect grouping.

`mayaku.config.io.load_yaml(path)` parses a YAML file into a
`MayakuConfig`; `dump_yaml` writes one back. The on-disk format is plain
YAML — no `_target_` strings, no Detectron2-style `CfgNode` magic. See
the [README configuration section](../README.md#configuration) for a
minimal example.

## Data pipeline: `data/`

The contract is the Detectron2 dataset-dict idiom — each sample is a
`dict[str, Any]` with `image`, `instances`, `image_id`, etc. Components:

- `data.catalog` — registries that map a dataset name to a builder
  function and a `Metadata` object (class names, keypoint connectivity,
  evaluator type).
- `data.datasets.coco.load_coco_json` — turns COCO annotations + an
  image root into a `list[dict]` ready for the mapper.
- `data.transforms.{base,geometry,augmentation}` — per-sample
  transforms (`ResizeShortestEdge`, `RandomFlip`, …).
- `data.mapper.DatasetMapper` — applies the transform list, attaches
  `Instances`, optionally clips keypoints / converts mask formats.
- `data.samplers` — `TrainingSampler` (infinite repeating shuffle),
  `RepeatFactorTrainingSampler` (class-balancing).
- `data.collate.AspectRatioGroupedDataset` — groups same-aspect-ratio
  images into batches so padding overhead stays bounded.

The CLI's `mayaku train` wires these together; see `mayaku/cli/train.py`
for the concrete pattern.

## Models: `models/`

Five subpackages, in forward-pass order:

1. **`backbones/`** — `resnet.py` builds ResNet-50 / 101 / ResNeXt-101
   (32×8d). `_frozen_bn.py` provides `FrozenBatchNorm2d` (the C2-style
   inference-time BN that backbone weights expect). `_base.py` defines
   the shared `Backbone` interface (`forward(image) -> dict[str, Tensor]`
   keyed by stage name `res2..res5`, plus `output_shape()`).

2. **`necks/fpn.py`** — Feature Pyramid Network. Wraps a backbone and
   exposes `{p2, p3, p4, p5, p6}` at strides 4 / 8 / 16 / 32 / 64. The
   FPN-wrapped backbone is what every detector in v1 actually consumes.

3. **`proposals/`** — RPN family: `anchor_generator.py` (per-level
   anchors), `matcher.py` (anchor↔GT matching), `box_regression.py` (the
   delta encoding), `sampling.py` (positive/negative subsampling),
   `rpn.py` (the RPN head + proposal generator).

4. **`heads/` + `roi_heads/`** — The post-RPN second stage:
   - `heads/box_head.py` — `FastRCNNConvFCHead` (the box-feature CNN+FC).
   - `heads/fast_rcnn.py` — `FastRCNNOutputLayers` (box-class scores +
     deltas + losses + per-class NMS inference).
   - `heads/mask_head.py` — `MaskRCNNConvUpsampleHead` (28×28 mask
     output with paste-to-image at inference time).
   - `heads/keypoint_head.py` — `KRCNNConvDeconvUpsampleHead` (56×56
     keypoint heatmaps with sub-pixel decode).
   - `roi_heads/standard.py` — `StandardROIHeads`, the dispatcher that
     pools features for the RPN proposals and routes them through the
     three head types depending on what the meta-architecture asked for.

5. **`detectors/`** — The three top-level meta-architectures:
   `faster_rcnn.py`, `mask_rcnn.py`, `keypoint_rcnn.py`. Each is a
   `nn.Module` whose `forward()` takes a `list[dict]` of dataset dicts
   and returns either a loss dict (training) or a `list[{"instances":
   Instances}]` (inference). `mayaku.cli._factory.build_detector(cfg)`
   dispatches to the right one based on `cfg.model.meta_architecture`.

`models.poolers.ROIPooler` is the shared FPN-aware ROI pooler used by
both the box head and the mask/keypoint heads.

`models.losses` packages the per-head loss functions (Smooth L1 / GIoU
for boxes, BCE for masks, cross-entropy with visibility weighting for
keypoints) so the same loss surface is reusable from custom heads.

## Engine: `engine/`

The training stack assembles like this:

```python
optimizer = build_optimizer(model, cfg.solver)
scheduler = build_lr_scheduler(optimizer, cfg.solver)
trainer = AMPTrainer(model, loader, optimizer, device,
                     amp_dtype=cfg.solver.amp_dtype) \
          if cfg.solver.amp_enabled and device.supports_amp \
          else SimpleTrainer(model, loader, optimizer)
trainer.register_hooks([
    IterationTimer(),
    LRScheduler(scheduler),
    PeriodicCheckpointer(model, output_dir, cfg.solver.checkpoint_period,
                         optimizer=optimizer),
])
trainer.train(start_iter=0, max_iter=cfg.solver.max_iter)
```

- `TrainerBase` defines the hook contract (`before_train`, `before_step`,
  `after_step`, `after_train`).
- `SimpleTrainer` runs one forward+backward+step per iteration.
- `AMPTrainer` wraps the same loop with `mayaku.backends.amp.autocast`
  and a `GradScaler` (only when `device.supports_amp`).
- `engine.optim.build_optimizer` splits parameters into norm vs non-norm
  groups so `weight_decay_norm` can be set independently.
- `engine.optim.build_lr_scheduler` picks `WarmupMultiStepLR` or
  `WarmupCosineLR` and wires the warmup factor through a `LambdaLR`.
- `engine.distributed.launch(main_fn, num_gpus, dist_backend=..., ...)`
  spawns one process per GPU via `torch.multiprocessing.spawn`, picks
  `nccl` on CUDA / `gloo` elsewhere, and pins each process to its
  `LOCAL_RANK`. `create_ddp_model(model, device)` is the
  device-aware DDP wrapper.
- `engine.evaluator.COCOEvaluator` + `inference_on_dataset` produce the
  COCO `AP / AP50 / AP75 / APs / APm / APl` dict for boxes, masks, and
  keypoints via `pycocotools`.
- `engine.callbacks` ships `IterationTimer`, `LRScheduler`,
  `MetricsPrinter`, `PeriodicCheckpointer`, and `EvalHook`. New hooks
  subclass `HookBase`.
- `EvalHook(period, evaluator, model, data_loader)` runs a held-out
  COCO evaluation every `period` iterations (driven by `cfg.test.eval_period`)
  plus a final pass on `after_train`, printing the metrics dict so users see
  mid-training AP without stopping the run. `mayaku train` wires it
  automatically when `--val-json`/`--val-images` are supplied; if
  `cfg.test.eval_period > 0` and the val paths are missing, training
  refuses to start with a clear error.

## Inference: `inference/`

Two halves:

- **`predictor.py`** — `Predictor` wraps a built detector for
  single-image / batch inference. It accepts an RGB `np.ndarray` or a
  file path, applies `ResizeShortestEdge`, runs the detector, and
  rescales predictions to original-image coordinates via
  `detector_postprocess`. `Predictor.from_config(cfg, model)` is the
  config-driven shortcut.
- **`postprocess.py`** — `detector_postprocess(instances, height, width)`
  is the rescale + mask-paste + keypoint-shift step pulled out as a
  standalone function so callers that bypass `Predictor` can reuse it.

The four exporters (`inference/export/{onnx,coreml,openvino,tensorrt}.py`)
all satisfy the `Exporter` protocol (`base.py`) and serialise the same
**Graph A body**: backbone + FPN producing `{p2..p6}`. Per-image Python
glue (RPN top-k + per-class NMS, mask paste, keypoint sub-pixel decode)
deliberately stays out of the exported graph — `BACKEND_PORTABILITY_REPORT.md`
§5/§6 makes the case for that split. Per-target details live under
[`docs/export/`](export/).

## CLI: `cli/`

Typer entry point at `mayaku.cli.__main__:app`. Each subcommand is a thin
wrapper around a `run_*` function in its own module so you can call it
in-process from a notebook or a test:

```python
from mayaku.cli.train import run_train
run_train(Path("cfg.yaml"), coco_gt_json=..., image_root=..., output_dir=...)
```

`cli/_factory.build_detector(cfg)` is the shared "build the right
detector for `cfg.model.meta_architecture`" dispatcher; both the CLI and
user scripts should call it instead of hand-importing the per-architecture
factories.

## Tests

`tests/` mirrors the source tree (`tests/unit/` plus a top-level
`test_smoke.py`). Tests gated by hardware or optional dependencies use
the pytest markers wired up in `tests/conftest.py`: `cuda`, `mps`,
`multi_gpu`, `onnx`, `coreml`, `openvino`, `tensorrt`, `slow`. The
active backend is selected by `MAYAKU_DEVICE={cpu,mps,cuda}` — see
[`portability.md`](portability.md) for the per-backend test protocol.
