"""``mayaku train`` — train a detector on a registered COCO dataset.

A thin glue: load config, build detector + optimizer + scheduler, build
a DataLoader off `load_coco_json` + `DatasetMapper` + `TrainingSampler`,
hand off to :class:`SimpleTrainer` (or :class:`AMPTrainer` when
``cfg.solver.amp_enabled``) with the standard hooks (`IterationTimer`,
`LRScheduler`, `PeriodicCheckpointer`).
"""

from __future__ import annotations

import gc
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from mayaku.backends.device import Device
from mayaku.backends.mps import apply_mps_environment
from mayaku.cli._factory import build_detector
from mayaku.config import MayakuConfig, dump_yaml, load_yaml
from mayaku.data import (
    AspectRatioGroupedDataset,
    Augmentation,
    CopyPaste,
    DatasetMapper,
    InferenceSampler,
    MixUp,
    Mosaic,
    MultiSampleAugmentation,
    MultiSampleMappedDataset,
    RandomColorJitter,
    RandomFlip,
    ResizeShortestEdge,
    SerializedList,
    TrainingSampler,
    build_coco_metadata,
    load_coco_json,
    trivial_batch_collator,
)
from mayaku.engine import (
    AMPTrainer,
    COCOEvaluator,
    EMAHook,
    EvalHook,
    IterationTimer,
    LRScheduler,
    MetricsPrinter,
    ModelEMA,
    PeriodicCheckpointer,
    SimpleTrainer,
    build_lr_scheduler,
    build_optimizer,
)

__all__ = ["run_train"]


def run_train(
    config: Path | MayakuConfig,
    *,
    coco_gt_json: Path,
    image_root: Path,
    output_dir: Path,
    weights: Path | None = None,
    pretrained_backbone: bool = False,
    device: str | None = None,
    max_iter: int | None = None,
    log_period: int = 20,
    val_json: Path | None = None,
    val_image_root: Path | None = None,
) -> None:
    """Train a detector.

    ``config`` accepts either a YAML path or a constructed
    :class:`MayakuConfig`. Passing the object directly skips the
    YAML round-trip — useful for Python-side fine-tune scripts that
    patch a base config in code. The resolved config is always
    serialised to ``output_dir/config.yaml`` for reproducibility.
    """
    if pretrained_backbone and weights is not None:
        raise ValueError(
            "--pretrained-backbone and --weights are mutually exclusive: the "
            "first asks for ImageNet-pretrained backbone init, the second "
            "loads a full mayaku checkpoint that already includes whatever "
            "backbone weights it was trained from. Pick one."
        )

    cfg = config if isinstance(config, MayakuConfig) else load_yaml(config)
    if max_iter is not None:
        cfg = cfg.model_copy(
            update={"solver": cfg.solver.model_copy(update={"max_iter": max_iter})}
        )

    # Periodic eval requires both the dataset paths *and* an enabled
    # period; rejecting the half-configured cases up front means users
    # see the error at config-load, not at the first periodic firing
    # which could be hours into training.
    if cfg.test.eval_period > 0 and (val_json is None or val_image_root is None):
        raise ValueError(
            f"test.eval_period={cfg.test.eval_period} > 0 requires --val-json "
            "and --val-images to be passed. Either set test.eval_period=0 to "
            "disable periodic eval, or supply the val dataset paths."
        )
    if cfg.test.eval_period == 0 and (val_json is not None or val_image_root is not None):
        warnings.warn(
            "--val-json/--val-images supplied but test.eval_period=0; the val "
            "dataset will be ignored. Set test.eval_period to a positive "
            "iteration count to enable mid-training eval.",
            stacklevel=2,
        )

    # Surface the most common silent footgun: freezing early backbone stages
    # at random init (the schema's freeze_at=2 default assumes a pretrained
    # backbone). Warn here rather than after model construction so the
    # message lands before the slow torch.load / weight download.
    if not pretrained_backbone and weights is None and cfg.model.backbone.freeze_at >= 1:
        warnings.warn(
            f"Backbone is random-init but freeze_at={cfg.model.backbone.freeze_at} "
            "is freezing the early stages. Random-init frozen features cannot "
            "be recovered downstream and training will not converge — your "
            "model will detect nothing. Pass --pretrained-backbone, or set "
            "model.backbone.freeze_at=0 in the YAML for true from-scratch "
            "training.",
            stacklevel=2,
        )

    # Reject the silent caffe2-vs-torchvision mismatch that drives box AP
    # to zero. ``pretrained_backbone=True`` loads torchvision IMAGENET1K_V2
    # weights, which were trained with stride-2 in the 3x3 conv of
    # res3/res4/res5 and with std-normalised inputs. The D2 model-zoo
    # configs (``faster_rcnn_R_50_FPN_1x.yaml`` and friends) ship
    # ``stride_in_1x1=true`` and ``pixel_std=[1, 1, 1]`` so they match
    # caffe2's MSRA-pretrained R-50.pkl — using either of those settings
    # with torchvision weights miscalibrates every downstream activation
    # (~58× input-scale error or wrong feature stride), training collapses
    # to "predict background everywhere," and eval returns AP=0. The
    # ``*_modern.yaml`` configs already match torchvision; if you want to
    # train from scratch with a D2-mirror config, either load D2's
    # converted .pth via ``--weights`` or override these two fields.
    if pretrained_backbone:
        if cfg.model.backbone.stride_in_1x1:
            raise ValueError(
                "pretrained_backbone=True loads torchvision IMAGENET1K_V2 weights, "
                "but the config has model.backbone.stride_in_1x1=true (caffe2/D2 "
                "layout). The two are not weight-compatible and training will "
                "silently collapse to AP=0. Either set stride_in_1x1=false in the "
                "config (matches torchvision; what the *_modern.yaml configs do), "
                "or drop --pretrained-backbone and load D2's R-50.pkl-converted "
                "checkpoint via --weights instead."
            )
        if tuple(cfg.model.pixel_std) == (1.0, 1.0, 1.0):
            raise ValueError(
                "pretrained_backbone=True loads torchvision IMAGENET1K_V2 weights, "
                "but the config has model.pixel_std=[1, 1, 1] (caffe2-style mean-"
                "only normalisation). Torchvision weights expect std-normalised "
                "inputs; pairing them with std=1 inflates input magnitudes ~58× "
                "and triggers the same silent collapse as a stride-layout "
                "mismatch. Set pixel_std=[58.395, 57.120, 57.375] (the *_modern."
                "yaml convention) when training with --pretrained-backbone."
            )

    dev = Device(kind=device) if device else Device.auto()  # type: ignore[arg-type]
    if dev.kind == "mps":
        apply_mps_environment()
    model = build_detector(
        cfg,
        backbone_weights="DEFAULT" if pretrained_backbone else None,
    ).to(dev.torch)
    if weights is not None:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        _load_for_finetune(model, state)

    optimizer = build_optimizer(model, cfg.solver)
    scheduler = build_lr_scheduler(optimizer, cfg.solver)

    metadata = build_coco_metadata(name="cli_train", json_path=coco_gt_json)
    # Drop annotation fields the meta-architecture won't read. For
    # COCO 2017 train this saves ~3-4 GB of polygon Python objects on
    # the detection / keypoint paths and lets pycocotools' parsed JSON
    # GC fully (the parser's internal tables are otherwise pinned by
    # the polygon-list references kept in dataset_dicts).
    keep_seg = cfg.model.meta_architecture == "mask_rcnn"
    keep_kp = cfg.model.meta_architecture == "keypoint_rcnn"
    dataset_dicts_raw = load_coco_json(
        coco_gt_json,
        image_root,
        metadata,
        keep_segmentation=keep_seg,
        keep_keypoints=keep_kp,
    )
    # Pickle the dicts into one bytes buffer. Drops resident-memory
    # footprint ~10× (one allocation vs millions of small Python
    # objects) and removes the per-iter malloc-fragmentation source
    # that otherwise drifts RAM into the tens of GB on long runs.
    dataset_dicts = SerializedList(dataset_dicts_raw)
    del dataset_dicts_raw
    gc.collect()
    augmentations: list[Augmentation] = [
        ResizeShortestEdge(
            cfg.input.min_size_train,
            max_size=cfg.input.max_size_train,
            sample_style=cfg.input.min_size_train_sampling,
        ),
        RandomFlip(prob=0.5 if cfg.input.random_flip == "horizontal" else 0.0),
    ]
    if cfg.input.color_jitter_enabled:
        # Photometric jitter must run AFTER geometric ops (resize / flip):
        # the colour-space ops are channel-wise so they don't care about
        # spatial layout, but applying them on the smaller resized image
        # is ~3-4× cheaper per step than applying on the original.
        augmentations.append(
            RandomColorJitter(
                brightness=cfg.input.color_jitter_brightness,
                contrast=cfg.input.color_jitter_contrast,
                saturation=cfg.input.color_jitter_saturation,
                hue=cfg.input.color_jitter_hue,
                prob=cfg.input.color_jitter_prob,
            )
        )
    mapper = DatasetMapper(
        augmentations,
        is_train=True,
        mask_format=cfg.input.mask_format,
        keypoint_on=cfg.model.meta_architecture == "keypoint_rcnn",
        metadata=metadata if cfg.model.meta_architecture == "keypoint_rcnn" else None,
        # SerializedList returns a fresh dict on every __getitem__, so
        # the mapper's defensive deepcopy is redundant — and skipping it
        # removes the dominant per-iter source of small-Python-object
        # churn that drives glibc malloc fragmentation.
        deepcopy_input=False,
    )

    # Build the active multi-sample augmentation list. When any prob is
    # > 0 the wrapper interposes; otherwise we use the plain `_MappedList`
    # so the single-sample path stays bit-identical to before. Same
    # interface (``__len__`` + ``__getitem__``) so downstream sampler /
    # AspectRatioGroupedDataset don't care which one they get.
    multi_sample_augs: list[MultiSampleAugmentation] = []
    if cfg.input.mosaic_prob > 0.0:
        multi_sample_augs.append(
            Mosaic(prob=cfg.input.mosaic_prob, canvas_size=cfg.input.mosaic_canvas_size)
        )
    if cfg.input.mixup_prob > 0.0:
        multi_sample_augs.append(MixUp(prob=cfg.input.mixup_prob, alpha=cfg.input.mixup_alpha))
    if cfg.input.copy_paste_prob > 0.0:
        # mask_format='bitmask' is enforced upstream by InputConfig's
        # validator — no need to re-check here.
        multi_sample_augs.append(CopyPaste(prob=cfg.input.copy_paste_prob))

    mapped: Any
    if multi_sample_augs:
        mapped = MultiSampleMappedDataset(dataset_dicts, mapper, multi_sample_augs)
    else:
        mapped = _MappedList(dataset_dicts, mapper)
    sampler = TrainingSampler(size=len(mapped), shuffle=True, seed=0)
    sampled_iter: Iterator[int] = iter(sampler)
    indexed = _SamplerView(mapped, sampled_iter)
    # AspectRatioGroupedDataset is itself an iterable yielding batches
    # (`list[dict]`), which is the contract SimpleTrainer expects from
    # its `data_loader`. Skipping a DataLoader here keeps the CLI
    # single-process; the multi-worker / DDP loader builder is a
    # natural Step 14+ follow-up once we have a real distributed
    # training story to wire it into.
    loader: Any = AspectRatioGroupedDataset(indexed, batch_size=cfg.solver.ims_per_batch)

    grad_clip_norm: float | None = (
        cfg.solver.clip_gradients_value if cfg.solver.clip_gradients_enabled else None
    )
    grad_clip_type = cfg.solver.clip_gradients_type
    trainer: SimpleTrainer
    if cfg.solver.amp_enabled and dev.supports_amp:
        trainer = AMPTrainer(
            model,
            loader,
            optimizer,
            dev,
            amp_dtype=cfg.solver.amp_dtype,
            grad_clip_norm=grad_clip_norm,
            grad_clip_type=grad_clip_type,
            grad_accum_steps=cfg.solver.grad_accum_steps,
            grad_norm_log_enabled=cfg.solver.grad_norm_log_enabled,
        )
    else:
        trainer = SimpleTrainer(
            model,
            loader,
            optimizer,
            grad_clip_norm=grad_clip_norm,
            grad_clip_type=grad_clip_type,
            grad_accum_steps=cfg.solver.grad_accum_steps,
            grad_norm_log_enabled=cfg.solver.grad_norm_log_enabled,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    # Dump the resolved config alongside the run so a checkpoint always
    # has its provenance recorded next to it. Mirrors Detectron2's
    # `cfg.dump()` convention. Overwriting on each run is intentional —
    # the *resolved* config is a function of the call args, not history.
    dump_yaml(cfg, output_dir / "config.yaml")
    timer = IterationTimer()
    hooks: list[Any] = [
        timer,
        LRScheduler(scheduler),
        MetricsPrinter(optimizer=optimizer, period=log_period, timer=timer),
        PeriodicCheckpointer(model, output_dir, cfg.solver.checkpoint_period, optimizer=optimizer),
    ]

    # EMA — register AFTER the live-model checkpointer so the EMA update
    # step doesn't fight the live save. The EMA shadow is checkpointed to
    # `output_dir/ema/` so the user can pick whichever variant they want
    # to ship; the EMA weights typically score 0.3-0.5 AP higher.
    if cfg.solver.ema_enabled:
        ema = ModelEMA(model, decay=cfg.solver.ema_decay, tau=cfg.solver.ema_tau)
        hooks.append(EMAHook(ema, model))
        hooks.append(
            PeriodicCheckpointer(
                ema.shadow,
                output_dir / "ema",
                cfg.solver.checkpoint_period,
            )
        )
    if cfg.test.eval_period > 0:
        assert val_json is not None and val_image_root is not None  # validated above
        val_loader = _build_val_loader(cfg, val_json, val_image_root)
        evaluator = COCOEvaluator(val_json, output_dir=output_dir / "eval")
        hooks.append(EvalHook(cfg.test.eval_period, evaluator, model, val_loader))
    trainer.register_hooks(hooks)
    trainer.train(start_iter=0, max_iter=cfg.solver.max_iter)


def _load_for_finetune(model: torch.nn.Module, state: dict[str, Any]) -> None:
    """Load ``state`` into ``model`` with class-count-aware shape filtering.

    Fine-tuning a converted COCO checkpoint on a custom-class dataset
    always hits a shape mismatch on the class-specific tail layers
    (``box_predictor.cls_score/bbox_pred``, ``mask_head.predictor``,
    ``keypoint_head.predictor`` for the keypoint variant). Detectron2
    drops these silently in its checkpointer; we drop them with an
    explicit log line so the user can confirm the reinitialised layers
    are exactly the ones they expected.

    Mismatched-shape keys are removed from ``state`` and the load runs
    with ``strict=False``. Any *missing* keys after that round of
    filtering are also surfaced — they're the layers that will train
    from random init.
    """
    own_state = model.state_dict()
    dropped: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    filtered = dict(state)
    for k, v in list(filtered.items()):
        if k in own_state and tuple(v.shape) != tuple(own_state[k].shape):
            dropped.append((k, tuple(v.shape), tuple(own_state[k].shape)))
            del filtered[k]

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    if dropped:
        print(
            f"[train] dropped {len(dropped)} shape-mismatched key(s) — these "
            "layers will train from random init:",
            flush=True,
        )
        for k, ckpt_shape, model_shape in dropped:
            print(f"  {k}: ckpt {ckpt_shape} vs model {model_shape}", flush=True)
    if missing:
        print(f"[train] missing keys not in checkpoint: {sorted(missing)}", flush=True)
    if unexpected:
        print(
            f"[train] unexpected keys in checkpoint (ignored): {sorted(unexpected)}",
            flush=True,
        )


def _build_val_loader(cfg: Any, val_json: Path, val_image_root: Path) -> DataLoader[Any]:
    """Mirror :func:`mayaku.cli.eval.run_eval`'s val loader construction.

    Kept as a helper rather than imported from `cli.eval` so the train
    path doesn't pull in the eval CLI surface (the two are deliberately
    separate console-script entry points).
    """
    metadata = build_coco_metadata(name="cli_train_eval", json_path=val_json)
    # Eval mapper drops annotations entirely (is_train=False), so
    # neither polygons nor keypoints are needed in the loaded dicts.
    dataset_dicts = load_coco_json(
        val_json,
        val_image_root,
        metadata,
        keep_segmentation=False,
        keep_keypoints=False,
    )
    mapper = DatasetMapper(
        [ResizeShortestEdge((cfg.input.min_size_test,), max_size=cfg.input.max_size_test)],
        is_train=False,
        keypoint_on=cfg.model.meta_architecture == "keypoint_rcnn",
        metadata=metadata if cfg.model.meta_architecture == "keypoint_rcnn" else None,
        deepcopy_input=False,
    )
    # Lazy mapping — running ``mapper`` over every val dict eagerly
    # would hold ~50 GB of float32 image tensors in RAM for the whole
    # training run (5k images × ~10 MB each on COCO val2017). The
    # DataLoader fetches one at a time on demand instead.
    mapped: Any = _MappedList(SerializedList(dataset_dicts), mapper)
    sampler = InferenceSampler(len(mapped))
    return DataLoader(
        mapped,
        batch_size=1,
        sampler=sampler,
        num_workers=0,
        collate_fn=trivial_batch_collator,
    )


# ---------------------------------------------------------------------------
# Tiny lazy-mapping helpers — keep DataLoader plumbing out of the data layer
# ---------------------------------------------------------------------------


class _MappedList:
    """Wrap a list of dataset dicts + a mapper into a list-style accessor.

    Avoids materialising every mapped sample upfront — the mapper runs
    on demand, which is what we want for large datasets even with
    ``num_workers=0`` (memory).
    """

    def __init__(
        self, dataset_dicts: Sequence[dict[str, Any]], mapper: DatasetMapper
    ) -> None:
        # Accepts either a plain ``list[dict]`` or a
        # :class:`mayaku.data.SerializedList` — both quack the same way
        # for ``__len__`` / ``__getitem__``.
        self._dataset_dicts = dataset_dicts
        self._mapper = mapper

    def __len__(self) -> int:
        return len(self._dataset_dicts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._mapper(self._dataset_dicts[idx])


class _SamplerView:
    """Iterable yielding ``mapped[i]`` for each ``i`` from a sampler.

    Glue between :class:`TrainingSampler` (an infinite index stream) and
    :class:`AspectRatioGroupedDataset` (which expects an iterable of
    sample dicts). ``mapped`` is either :class:`_MappedList` (single-sample
    path) or :class:`MultiSampleMappedDataset` (Phase 1b multi-sample
    path). Both quack the same way: ``mapped[i]`` returns a mapped dict.
    """

    def __init__(
        self,
        mapped: _MappedList | MultiSampleMappedDataset,
        sampler_iter: Iterator[int],
    ) -> None:
        self._mapped = mapped
        self._iter = sampler_iter

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for idx in self._iter:
            yield self._mapped[idx]
