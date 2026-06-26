"""``mayaku train`` — train a detector on a registered COCO dataset.

A thin glue: load config, build detector + optimizer + scheduler, build
a DataLoader off `load_coco_json` + `DatasetMapper` + `TrainingSampler`,
hand off to :class:`SimpleTrainer` (or :class:`AMPTrainer` when
``cfg.solver.amp_enabled``) with the standard hooks (`IterationTimer`,
`LRScheduler`, `PeriodicCheckpointer`).
"""

from __future__ import annotations

import functools
import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from mayaku import __version__
from mayaku.backends.device import Device
from mayaku.backends.mps import apply_mps_environment
from mayaku.cli._factory import build_detector, build_resize_augmentation
from mayaku.config import MayakuConfig, dump_yaml, merge_overrides
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
    RandAugment,
    RandomColorJitter,
    RandomFlip,
    RepeatFactorTrainingSampler,
    TrainingSampler,
    load_coco_dataset,
    load_shared_dataset,
    trivial_batch_collator,
)
from mayaku.engine import (
    AMPTrainer,
    COCOEvaluator,
    EMAHook,
    EvalHook,
    IterationTimer,
    LRScheduler,
    LRSnapshotHook,
    MetricsPrinter,
    ModelEMA,
    PeriodicCheckpointer,
    SimpleTrainer,
    build_lr_scheduler,
    build_optimizer,
    create_ddp_model,
    get_rank,
    get_world_size,
    init_from_env_if_needed,
    is_main_process,
    resolve_schedule,
)
from mayaku.tuning import (
    analyze_dataset,
    collect_set_paths,
    derive_overrides,
    filter_unset,
    walk_leaves,
)
from mayaku.tuning.dataset_stats import dataset_aspect
from mayaku.tuning.sizing import resolve_canvas
from mayaku.utils import build_sidecar, git_hash

__all__ = ["run_train", "run_train_worker"]


def run_train(
    config: Path | MayakuConfig,
    *,
    coco_gt_json: Path,
    image_root: Path,
    output_dir: Path,
    weights: Path | None = None,
    pretrained_backbone: bool = False,
    device: str | None = None,
    num_epochs: int | None = None,
    log_period: int = 20,
    val_json: Path | None = None,
    val_image_root: Path | None = None,
    resume: Path | None = None,
    user_set_paths: set[str] | None = None,
) -> None:
    """Train a detector.

    ``config`` accepts either a YAML path or a constructed
    :class:`MayakuConfig`. Passing the object directly skips the
    YAML round-trip — useful for Python-side fine-tune scripts that
    patch a base config in code. The resolved config is always
    serialised to ``output_dir/config.yaml`` for reproducibility.

    ``resume`` points at a ``model_iter_*.pth`` checkpoint to continue
    training from: model weights, optimizer state, LR-schedule position,
    and (if present) the EMA shadow are restored and training resumes at
    the checkpoint's iteration. Use the *same* config the checkpoint was
    trained with. Mutually exclusive with ``weights`` / ``pretrained_backbone``.
    """
    if pretrained_backbone and weights is not None:
        raise ValueError(
            "--pretrained-backbone and --weights are mutually exclusive: the "
            "first asks for ImageNet-pretrained backbone init, the second "
            "loads a full mayaku checkpoint that already includes whatever "
            "backbone weights it was trained from. Pick one."
        )
    if resume is not None and (weights is not None or pretrained_backbone):
        raise ValueError(
            "--resume is mutually exclusive with --weights / --pretrained-backbone: "
            "resume restores the full training state (weights + optimizer + LR "
            "schedule) from a checkpoint, so there is nothing left to initialise."
        )

    # Track which config paths the user explicitly set. Auto-config later
    # uses this to skip any field the user already pinned, so explicit
    # values always win over the dataset-derived recipe. Two sources feed
    # it: ``user_set_paths`` forwarded by the caller (``mayaku.api`` passes
    # the user's ``overrides`` + ``size_budget`` here so they survive
    # auto-config), and — when ``config`` is a YAML path — the leaves of
    # that YAML. A constructed ``MayakuConfig`` carries no such record, so
    # only the caller-forwarded set protects it.
    # Fresh copy so the later ``.add("solver.num_epochs")`` never mutates the
    # caller's set.
    user_set_paths = set(user_set_paths) if user_set_paths is not None else set()
    if isinstance(config, MayakuConfig):
        cfg = config
    else:
        text = Path(config).read_text(encoding="utf-8")
        raw_yaml = yaml.safe_load(text) or {}
        if not isinstance(raw_yaml, Mapping):
            raise ValueError(
                f"YAML at {config} must be a mapping at the top level; "
                f"got {type(raw_yaml).__name__}"
            )
        raw_dict = dict(raw_yaml)
        user_set_paths |= collect_set_paths(raw_dict)
        cfg = MayakuConfig.model_validate(raw_dict)

    if num_epochs is not None:
        cfg = cfg.model_copy(
            update={"solver": cfg.solver.model_copy(update={"num_epochs": num_epochs})}
        )
        # An explicit --epochs is a user choice; preserve it through auto-config
        # even though the value didn't come from the YAML.
        user_set_paths.add("solver.num_epochs")

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
                "inputs; pairing them with std=1 inflates input magnitudes ~58x "
                "and triggers the same silent collapse as a stride-layout "
                "mismatch. Set pixel_std=[58.395, 57.120, 57.375] (the *_modern."
                "yaml convention) when training with --pretrained-backbone."
            )

    # Bring up the distributed context BEFORE loading the dataset: the
    # per-node shared loader (below) broadcasts the parsed dataset over the
    # process group, so the group must already exist. ``dev`` / ``world_size``
    # / ``rank`` are needed by the model build and sampler that follow too.
    dev = Device.auto() if not device or device == "auto" else Device(kind=device)  # type: ignore[arg-type]
    if dev.kind == "mps":
        apply_mps_environment()

    # If the user launched this process via ``torchrun`` (env vars
    # WORLD_SIZE/RANK/LOCAL_RANK set, but no init_process_group called
    # yet), bring up the process group from those env vars so the rest
    # of this function sees the right world_size/rank. No-op when
    # called from inside :func:`mayaku.engine.launch` (already inited)
    # or when WORLD_SIZE is 1.
    init_from_env_if_needed(dev)

    # Distributed-training context. ``world_size`` and ``rank`` come from
    # ``mayaku.engine.distributed`` — when this CLI runs outside of any
    # process group they return 1/0, so the single-GPU code path below
    # stays bit-identical to before.
    world_size = get_world_size()
    rank = get_rank()

    # Under DDP on CUDA, pin this Device to the rank's assigned GPU.
    # Both launch paths (``_worker_entry`` and ``init_from_env_if_needed``)
    # have already called ``torch.cuda.set_device(rank)``; mirror that
    # into the Device descriptor so ``model.to(dev.torch)`` and
    # ``create_ddp_model(model, dev)`` use ``cuda:rank`` instead of
    # ``cuda:0``. Without this every rank would silently train on cuda:0.
    if dev.kind == "cuda" and world_size > 1:
        dev = Device(kind="cuda", index=rank)

    # Drop annotation fields the meta-architecture won't read. For
    # COCO 2017 train this saves ~3-4 GB of polygon Python objects on
    # the detection / keypoint paths and lets pycocotools' parsed JSON
    # GC fully (the parser's internal tables are otherwise pinned by
    # the polygon-list references kept in dataset_dicts).
    keep_seg = cfg.model.meta_architecture == "mask_rcnn" or cfg.model.uniquery_mask is not None
    keep_kp = (
        cfg.model.meta_architecture == "keypoint_rcnn" or cfg.model.uniquery_keypoint is not None
    )

    # Everything derived from the *raw* dataset dicts — auto-config overrides
    # and RFS repeat factors — is computed inside ``_derive`` so it runs
    # exactly once per node (on the node's local rank 0) under the shared
    # loader, then broadcast to the node's other ranks. Both feed the model
    # build / sampler below, so they must be identical across ranks.
    def _derive(meta: Any, dicts: list[dict[str, Any]]) -> dict[str, Any]:
        # Dataset-aware auto-config. The recipe layer fills in fine-tune-
        # relevant heuristics (anchor sizes/ARs, base_lr, schedule, mosaic /
        # mixup probs, sampler choice) that the user did NOT set explicitly.
        # (num_classes is handled separately in the body — it's structural, not
        # a heuristic.) Tiny datasets (< MIN_IMAGES_FOR_AUTO_CONFIG) and
        # ``auto_config.enabled = False`` both short-circuit with no overrides.
        overrides: Mapping[str, Any] | None = None
        stats: Any = None
        if cfg.auto_config.enabled:
            stats = analyze_dataset(
                dicts,
                num_classes=len(meta.thing_classes),
                resize_short_edge=cfg.input.min_size_test,
                resize_max_edge=cfg.input.max_size_test,
            )
            proposed = derive_overrides(stats, cfg)
            overrides = filter_unset(proposed, user_set_paths) or None
        # RFS repeat factors, decided against the *post*-auto-config dataloader
        # (auto-config may select RepeatFactorTrainingSampler). Computed on
        # the raw dicts — a SerializedList would unpickle every dict twice —
        # and ~4 bytes per image, so cheap to broadcast.
        dataloader = merge_overrides(cfg, overrides).dataloader if overrides else cfg.dataloader
        rfs: torch.Tensor | None = None
        if dataloader.sampler_train == "RepeatFactorTrainingSampler":
            rfs = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dicts, repeat_thresh=dataloader.repeat_threshold
            )
        # Aspect-aware letterbox canvas: ALWAYS resolved from *this* run's data
        # (uniform → rectangle, diverse → square) under the size_budget² budget.
        # Re-resolving every train is what lets fine-tuning adapt: a 1:1 base model
        # fine-tuned on 16:9 data gets a 16:9 canvas, not the base's inherited
        # square. ``canvas_hw`` is a deploy artifact of training, not a user input.
        canvas: tuple[int, int] | None = None
        canvas_use = 1.0
        if cfg.input.resize_mode == "letterbox":
            # Reuse the auto-config stats' aspect profile when it already ran;
            # else a light dims-only pass (avoids re-scanning the dicts).
            if stats is not None:
                median, uniform = stats.aspect_median, stats.is_uniform_aspect
            else:
                median, uniform = dataset_aspect(dicts)
            canvas, canvas_use = resolve_canvas(cfg.input.size_budget, median, uniform)
        return {
            "overrides": overrides,
            "stats": stats,
            "repeat_factors": rfs,
            "canvas": canvas,
            "canvas_use": canvas_use,
        }

    # Load the dataset BEFORE model construction so auto-config can rewrite
    # anchor sizes, ROI head class count, schedule, etc. before any weights
    # are allocated. Under DDP this parses the annotation JSON **once per
    # node** (not once per GPU) and shares it across the node's ranks — a
    # large dataset (e.g. Objects365's 14 GB JSON) would otherwise hold N
    # copies of the dataset dicts per node and OOM host RAM. The returned
    # ``dataset_dicts`` is a SerializedList (one contiguous bytes buffer):
    # ~10× smaller resident-set and no malloc-fragmentation drift over a
    # long run. Single-GPU and N-servers×1-GPU fall through unchanged.
    metadata, dataset_dicts, derived = load_shared_dataset(
        parse_fn=lambda: load_coco_dataset(
            name="cli_train",
            json_path=coco_gt_json,
            image_root=image_root,
            keep_segmentation=keep_seg,
            keep_keypoints=keep_kp,
        ),
        derive_fn=_derive,
    )
    assert derived is not None
    if derived["overrides"]:
        cfg = merge_overrides(cfg, derived["overrides"])
        if is_main_process():
            _log_auto_config_report(derived["stats"], derived["overrides"], user_set_paths)

    # When auto-config is on (the fine-tune / API default), the class count is a
    # STRUCTURAL fact of the dataset — derive it from the COCO categories
    # regardless of dataset size (unlike the LR/anchor/schedule heuristics, which
    # need MIN_IMAGES_FOR_AUTO_CONFIG worth of data). This is what makes a
    # weights-only fine-tune on a new class count reinit the head: the rebuilt
    # head is sized to the new dataset, so the old class-specific layers
    # shape-mismatch and `_load_for_finetune` drops + reinitialises them. With
    # auto-config off (manual / replication) the config is used verbatim, and a
    # user-pinned num_classes always wins.
    if cfg.auto_config.enabled and "model.roi_heads.num_classes" not in user_set_paths:
        n_classes = len(metadata.thing_classes)
        if cfg.model.roi_heads.num_classes != n_classes:
            cfg = merge_overrides(cfg, {"model": {"roi_heads": {"num_classes": n_classes}}})
            if is_main_process():
                print(f"[train] num_classes set from dataset categories: {n_classes}", flush=True)

    # Resolve the epoch budget to iteration counts now that the dataset size is
    # known. One epoch = ceil(num_images / global_batch) optimizer steps, where
    # global_batch spans grad-accum and all DDP ranks. The scheduler, LLRD
    # snapshot milestones, and the trainer loop all run on these resolved ints
    # (max_iter / warmup_iters are derived, not config fields).
    max_iter, warmup_iters = resolve_schedule(
        cfg.solver.num_epochs,
        len(dataset_dicts),
        cfg.solver.effective_batch(world_size),
        cfg.solver.warmup_fraction,
    )
    # Bake the resolved letterbox canvas into cfg so the resize aug (train + eval)
    # and the checkpoint sidecar all carry the exact (H, W) the model deploys at.
    if derived["canvas"] is not None:
        canvas: tuple[int, int] = derived["canvas"]
        cfg = cfg.model_copy(update={"input": cfg.input.model_copy(update={"canvas_hw": canvas})})
        if is_main_process():
            _log_canvas(canvas, derived["canvas_use"])
    repeat_factors: torch.Tensor | None = derived["repeat_factors"]

    # Surface the most common silent footgun: freezing early backbone stages
    # at random init (the schema's freeze_at=2 default assumes a pretrained
    # backbone). Warn here rather than after model construction so the
    # message lands before the slow torch.load / weight download. Runs
    # after auto-config so the freeze_at being checked is the one that
    # will actually train.
    #
    # ``cfg.model.backbone.weights_path`` is loaded inside the backbone
    # __init__ before ``_apply_freeze``, so it counts as a real init source
    # here even though it isn't fed through the top-level ``weights`` path.
    backbone_initialized = (
        pretrained_backbone or weights is not None or cfg.model.backbone.weights_path is not None
    )
    if not backbone_initialized and cfg.model.backbone.freeze_at >= 1:
        warnings.warn(
            f"Backbone is random-init but freeze_at={cfg.model.backbone.freeze_at} "
            "is freezing the early stages. Random-init frozen features cannot "
            "be recovered downstream and training will not converge — your "
            "model will detect nothing. Pass --pretrained-backbone, or set "
            "model.backbone.freeze_at=0 in the YAML for true from-scratch "
            "training.",
            stacklevel=2,
        )

    model = build_detector(
        cfg,
        backbone_weights="DEFAULT" if pretrained_backbone else None,
    ).to(dev.torch)
    # ``resume`` restores the exact prior training state; ``weights`` is a
    # warm-start that re-inits the schedule. They're mutually exclusive
    # (validated above). ``start_iter`` drives the trainer + LR fast-forward.
    start_iter = 0
    resume_ckpt: dict[str, Any] | None = None
    if resume is not None:
        resume_ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        # Same architecture as the checkpoint -> strict load (a key mismatch
        # means the wrong config, which we want to fail on, not silently drop).
        model.load_state_dict(resume_ckpt["model"])
        start_iter = _checkpoint_iteration(resume_ckpt, resume)
    elif weights is not None:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        _load_for_finetune(model, state)

    # build_optimizer needs the unwrapped model so its parameter-group
    # iteration finds the actual modules; DDP wrap happens *after*. The
    # scheduler captures base LRs from the freshly-built optimizer, so it
    # must be constructed *before* the resumed optimizer state is loaded.
    optimizer = build_optimizer(model, cfg.solver)
    scheduler = build_lr_scheduler(
        optimizer, cfg.solver, max_iter=max_iter, warmup_iters=warmup_iters
    )
    if resume_ckpt is not None:
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        # Replay the schedule to start_iter. It's a pure function of the
        # iteration (no persisted state), so this reproduces the exact LR
        # training had there — and overwrites the LRs the optimizer state
        # just restored, which is what we want.
        _fast_forward_scheduler(scheduler, start_iter)
    if world_size > 1:
        model = create_ddp_model(model, dev)
    # Letterbox training draws one multi-scale canvas per image (down to the
    # train_scale_min budget fraction, top = the deploy canvas); the resize
    # builder resolves it from size_budget + canvas_hw. geometry == eval == deploy.
    augmentations: list[Augmentation] = [
        build_resize_augmentation(cfg, for_train=True),
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
    if cfg.input.randaugment_enabled:
        # Same placement rationale as color_jitter — run on the resized
        # image so the per-step cost is dominated by the geometric ops
        # rather than the photometric LUT/CDF work.
        augmentations.append(
            RandAugment(
                num_ops=cfg.input.randaugment_num_ops,
                magnitude=cfg.input.randaugment_magnitude,
            )
        )
    mapper = DatasetMapper(
        augmentations,
        is_train=True,
        mask_format=cfg.input.mask_format,
        keypoint_on=cfg.model.meta_architecture == "keypoint_rcnn"
        or cfg.model.uniquery_keypoint is not None,
        metadata=metadata
        if cfg.model.meta_architecture in ("keypoint_rcnn",)
        or cfg.model.uniquery_keypoint is not None
        else None,
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
    # Single training seed: both the samplers and the per-worker augmentation
    # RNGs derive from it, so a run is reproducible from this one value.
    seed = 0
    # Samplers are rank-aware: each rank reads a strided slice of the
    # shuffled index stream so no two ranks see the same image in the
    # same effective batch. ``num_replicas`` / ``rank`` default to 1/0
    # outside DDP, keeping the single-GPU path bit-identical.
    sampler: TrainingSampler | RepeatFactorTrainingSampler
    if repeat_factors is not None:
        sampler = RepeatFactorTrainingSampler(
            repeat_factors, seed=seed, num_replicas=world_size, rank=rank
        )
    else:
        sampler = TrainingSampler(
            size=len(mapped),
            shuffle=True,
            seed=seed,
            num_replicas=world_size,
            rank=rank,
        )
    # Seed the augmentation RNGs. With workers, ``worker_init_fn`` reseeds each
    # to an independent per-(rank, worker) stream (the augmentation objects were
    # built once in the main process and would otherwise replay identical
    # streams). ``num_workers=0`` has no workers, so this main-process reseed is
    # the live one; with workers it is the deterministic base they overwrite.
    mapped.reseed(np.random.default_rng(np.random.SeedSequence([seed, rank])))
    # ``prefetch_factor`` is only a valid DataLoader arg when there are
    # worker processes (num_workers > 0); passing it with num_workers=0
    # raises. Buffering num_workers x prefetch_factor samples lets the
    # workers stay several batches ahead so a fast GPU never waits on the
    # aspect-ratio grouper to refill (see DataLoaderConfig.prefetch_factor).
    loader_kwargs: dict[str, Any] = {}
    if cfg.dataloader.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.dataloader.prefetch_factor
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["worker_init_fn"] = functools.partial(
            _seed_worker_augmentations, base_seed=seed, rank=rank
        )
    dl = DataLoader(
        mapped,
        sampler=sampler,
        batch_size=1,
        num_workers=cfg.dataloader.num_workers,
        collate_fn=_unwrap_single,
        pin_memory=dev.supports_pin_memory,
        **loader_kwargs,
    )
    loader: Any = AspectRatioGroupedDataset(dl, batch_size=cfg.solver.ims_per_batch)

    grad_clip_norm: float | None = (
        cfg.solver.clip_gradients_value if cfg.solver.clip_gradients_enabled else None
    )
    grad_clip_type = cfg.solver.clip_gradients_type
    # amp_dtype in the config is an intent; the device clamps it to what the
    # hardware can actually deliver (bf16 only on native-bf16 GPUs, never on
    # MPS) and returns None when AMP can't run here at all.
    requested_amp = cfg.solver.amp_dtype
    amp_dtype: str | None = None
    if cfg.solver.amp_enabled:
        amp_dtype = dev.resolve_amp_dtype(requested_amp)
        if amp_dtype != requested_amp and is_main_process():
            if amp_dtype is None:
                print(
                    f"[train] AMP ({requested_amp}) unavailable on {dev.kind}; training in fp32.",
                    flush=True,
                )
            else:
                print(
                    f"[train] amp_dtype {requested_amp!r} unsupported on this {dev.kind} device; "
                    f"using {amp_dtype!r} instead.",
                    flush=True,
                )

    trainer: SimpleTrainer
    if amp_dtype is not None:
        trainer = AMPTrainer(
            model,
            loader,
            optimizer,
            dev,
            amp_dtype=amp_dtype,
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
    # Under DDP ``model`` is a DistributedDataParallel wrapper whose
    # ``state_dict`` carries ``module.`` prefixes that don't match the
    # unwrapped layout used everywhere else (eval, predict, export). We
    # save the underlying module so checkpoints stay interchangeable
    # between single-GPU and DDP runs. Only unwrap when DDP actually
    # wrapped the model (``world_size > 1`` above) — otherwise a raw
    # detector that happens to expose a submodule named ``module``
    # would be silently grabbed instead.
    unwrapped_model: torch.nn.Module
    if world_size > 1:
        inner = model.module
        assert isinstance(inner, torch.nn.Module)
        unwrapped_model = inner
    else:
        unwrapped_model = model

    # Dump the resolved config alongside the run so a checkpoint always
    # has its provenance recorded next to it. Mirrors Detectron2's
    # `cfg.dump()` convention. Rank-0 only under DDP to avoid N writers
    # racing on the same path.
    if is_main_process():
        dump_yaml(cfg, output_dir / "config.yaml")

    # Self-describing sidecar embedded in every checkpoint so a trained
    # .pth reconstructs its architecture + labels without a separate
    # config or names file. The resolved config carries the architecture
    # and normalization (pixel mean/std) plus the backbone-coupled solver
    # settings; class names come from the COCO categories. Kept under a
    # "mayaku" key so "model" stays a pure state_dict.
    checkpoint_metadata = build_sidecar(
        cfg,
        metadata.thing_classes,
        provenance={"mayaku_version": __version__, "git_hash": git_hash()},
    )

    timer = IterationTimer()
    hooks: list[Any] = [
        timer,
        LRScheduler(scheduler),
    ]
    # Side-effect hooks (stdout / disk writes) run on rank 0 only.
    # Logic-only hooks (timer, LR scheduler, EMA update) fire on every
    # rank — they need to to keep state in sync.
    if is_main_process():
        hooks.append(MetricsPrinter(optimizer=optimizer, period=log_period, timer=timer))
        hooks.append(
            PeriodicCheckpointer(
                unwrapped_model,
                output_dir,
                cfg.solver.checkpoint_period,
                optimizer=optimizer,
                metadata=checkpoint_metadata,
            )
        )
        # LLRD survival check: snapshot per-group LR at named milestones
        # so the run artifact carries proof that decay ratios are preserved
        # through warmup + cosine. Only meaningful when LLRD actually
        # splits the optimizer into per-layer groups.
        if cfg.solver.llrd_enabled:
            milestones = (warmup_iters, max_iter // 2, max_iter)
            hooks.append(LRSnapshotHook(optimizer, milestones))

    # EMA — register AFTER the live-model checkpointer so the EMA update
    # step doesn't fight the live save. The EMA shadow is checkpointed to
    # `output_dir/ema/` so the user can pick whichever variant they want
    # to ship; the EMA weights typically score 0.3-0.5 AP higher.
    ema: ModelEMA | None = None
    if cfg.solver.ema_enabled:
        # ModelEMA deep-copies the underlying model state. Deep-copying a
        # DDP wrapper would mirror the wrapper's process-group references,
        # which isn't what we want — pass the unwrapped module so the
        # shadow is a clean replica of the actual parameters.
        # ``updates=start_iter`` continues the EMA decay-warmup curve from the
        # right point on resume (0 for a fresh run). The shadow is then loaded
        # from the sibling ``ema/`` checkpoint so the averaging history isn't
        # lost; if it's missing (e.g. EMA was off in the prior run) the shadow
        # falls back to the restored live weights, which is still correct.
        ema = ModelEMA(
            unwrapped_model,
            decay=cfg.solver.ema_decay,
            tau=cfg.solver.ema_tau,
            updates=start_iter,
        )
        if resume_ckpt is not None and resume is not None:
            ema_ckpt_path = resume.parent / "ema" / resume.name
            if ema_ckpt_path.is_file():
                ema_state = torch.load(ema_ckpt_path, map_location="cpu", weights_only=False)
                ema.shadow.load_state_dict(ema_state.get("model", ema_state))
            else:
                warnings.warn(
                    f"--resume: no EMA shadow at {ema_ckpt_path}; the EMA will "
                    "restart from the resumed live weights. The final EMA AP is "
                    "unaffected once tau worth of steps have re-accumulated.",
                    stacklevel=2,
                )
        hooks.append(EMAHook(ema, unwrapped_model))
        if is_main_process():
            hooks.append(
                PeriodicCheckpointer(
                    ema.shadow,
                    output_dir / "ema",
                    cfg.solver.checkpoint_period,
                    metadata=checkpoint_metadata,
                )
            )
    if cfg.test.eval_period > 0:
        # Mid-training eval runs on every rank in v1: gating to rank 0
        # only would let rank 0 sit in eval past NCCL's 30-min watchdog
        # while the other ranks block in their next iter's all_reduce,
        # then crash. Redundant eval across N ranks is wasteful but
        # keeps every rank reaching the same collectives at roughly the
        # same wall-clock — see docs/portability.md for the offline
        # ``mayaku eval`` workaround on large val sets.
        assert val_json is not None and val_image_root is not None  # validated above
        val_loader = _build_val_loader(cfg, val_json, val_image_root)
        # Decode predictions against the val GT by the *train* class identity
        # (matched by name), so a val split that numbers/orders its categories
        # differently can't silently misalign AP (C7).
        evaluator = COCOEvaluator(
            val_json,
            output_dir=output_dir / "eval",
            class_names=metadata.thing_classes,
        )
        eval_model = ema.shadow if ema is not None else unwrapped_model
        hooks.append(EvalHook(cfg.test.eval_period, evaluator, eval_model, val_loader))
    trainer.register_hooks(hooks)
    trainer.train(start_iter=start_iter, max_iter=max_iter)


def run_train_worker(
    config: Path | MayakuConfig,
    coco_gt_json: Path,
    image_root: Path,
    output_dir: Path,
    weights: Path | None,
    pretrained_backbone: bool,
    device: str | None,
    num_epochs: int | None,
    log_period: int,
    val_json: Path | None,
    val_image_root: Path | None,
    resume: Path | None = None,
    user_set_paths: set[str] | None = None,
) -> None:
    """Positional-arg adapter for :func:`mayaku.engine.launch`.

    ``launch`` calls ``main_func(*args)``; ``run_train`` is keyword-only
    after the first positional. Module-level so :mod:`multiprocessing`
    can pickle the reference when ``launch`` spawns workers.
    """
    run_train(
        config,
        coco_gt_json=coco_gt_json,
        image_root=image_root,
        output_dir=output_dir,
        weights=weights,
        pretrained_backbone=pretrained_backbone,
        device=device,
        num_epochs=num_epochs,
        log_period=log_period,
        val_json=val_json,
        val_image_root=val_image_root,
        resume=resume,
        user_set_paths=user_set_paths,
    )


def _checkpoint_iteration(ckpt: Mapping[str, Any], path: Path) -> int:
    """Resolve the iteration to resume at from a checkpoint.

    Prefers the ``iteration`` field written by :class:`PeriodicCheckpointer`;
    falls back to parsing ``model_iter_<N>.pth`` for checkpoints saved before
    that field existed.
    """
    it = ckpt.get("iteration")
    if isinstance(it, int):
        return it
    m = re.search(r"model_iter_(\d+)", path.name)
    if m is not None:
        return int(m.group(1))
    raise ValueError(
        f"--resume checkpoint {path} carries no 'iteration' field and its name "
        "doesn't match 'model_iter_<N>.pth', so the resume iteration is unknown. "
        "Pass a periodic checkpoint (e.g. .../train/model_iter_0060000.pth)."
    )


def _fast_forward_scheduler(scheduler: Any, start_iter: int) -> None:
    """Advance an LR scheduler to ``start_iter`` and apply that step's LR.

    The schedule is a pure function of the iteration, so setting ``last_epoch``
    and re-applying reproduces the exact LR training had at ``start_iter`` in
    O(1). The benign "step before optimizer step" warning is suppressed — this
    is a deliberate schedule replay on resume, not a real training step.
    """
    if start_iter <= 0:
        return
    scheduler.last_epoch = start_iter - 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler.step()


def _log_canvas(canvas: tuple[int, int], use: float) -> None:
    """Report the resolved letterbox canvas + flag a budget-underusing grid gap."""
    print(
        f"[letterbox] canvas resolved to {canvas[1]}w x {canvas[0]}h "
        f"({use:.0%} of the size_budget^2 budget)",
        flush=True,
    )
    if use < 0.80:
        print(
            f"[letterbox] canvas uses only {use:.0%} of the budget (128-grid gap) — "
            "raise size_budget for more resolution.",
            flush=True,
        )


def _log_auto_config_report(
    stats: Any, applied: Mapping[str, Any], user_set_paths: set[str]
) -> None:
    """Print a scannable summary of what auto-config decided.

    Goes through stdout so it sits next to the existing ``[train]``
    lines from :func:`_load_for_finetune`.
    """
    print(
        f"[auto-config] N_train={stats.num_images} num_classes={stats.num_classes} "
        f"num_boxes={stats.num_boxes} imbalance={stats.class_imbalance:.1f}x",
        flush=True,
    )
    for path, value in walk_leaves(applied):
        print(f"[auto-config] {path} -> {value!r}", flush=True)
    if user_set_paths:
        kept = sorted(
            p
            for p in user_set_paths
            if p.split(".", 1)[0] in {"model", "solver", "input", "dataloader"}
        )
        if kept:
            print(f"[auto-config] user-set (preserved): {kept}", flush=True)


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
    # Eval mapper drops annotations entirely (is_train=False), so neither
    # polygons nor keypoints are needed in the loaded dicts. One COCO parse
    # for both metadata and dicts (load_coco_dataset) instead of two — and
    # routed through the same per-node shared loader as the train set, so it
    # parses once per node, not once per GPU. Matters for large eval sets
    # (LVIS / O365 val); a no-op extra broadcast for COCO's 50 MB.
    metadata, dataset_dicts, _ = load_shared_dataset(
        parse_fn=lambda: load_coco_dataset(
            name="cli_train_eval",
            json_path=val_json,
            image_root=val_image_root,
            keep_segmentation=False,
            keep_keypoints=False,
        ),
    )
    # In-train periodic eval matches deploy: letterbox to size_budget when the
    # config uses it (the evaluator un-letterboxes via the recorded transform).
    mapper = DatasetMapper(
        [build_resize_augmentation(cfg, for_train=False)],
        is_train=False,
        keypoint_on=cfg.model.meta_architecture == "keypoint_rcnn",
        metadata=metadata if cfg.model.meta_architecture == "keypoint_rcnn" else None,
        deepcopy_input=False,
    )
    # Lazy mapping — running ``mapper`` over every val dict eagerly
    # would hold ~50 GB of float32 image tensors in RAM for the whole
    # training run (5k images × ~10 MB each on COCO val2017). The
    # DataLoader fetches one at a time on demand instead.
    mapped: Any = _MappedList(dataset_dicts, mapper)
    # Shard the val indices across ranks (disjoint, no padding) so each image is
    # evaluated exactly once. The evaluator all-gathers + flat-merges per-rank
    # predictions, so a non-sharded sampler (num_replicas=1) would feed COCOeval
    # world_size duplicate detections and deflate AP. No-op at world_size=1.
    sampler = InferenceSampler(len(mapped), num_replicas=get_world_size(), rank=get_rank())
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

    def __init__(self, dataset_dicts: Sequence[dict[str, Any]], mapper: DatasetMapper) -> None:
        # Accepts either a plain ``list[dict]`` or a
        # :class:`mayaku.data.SerializedList` — both quack the same way
        # for ``__len__`` / ``__getitem__``.
        self._dataset_dicts = dataset_dicts
        self._mapper = mapper

    def reseed(self, rng: np.random.Generator) -> None:
        """Forward to the mapper so its augmentations get a fresh per-worker
        stream. See :meth:`mayaku.data.mapper.DatasetMapper.reseed`.
        """
        self._mapper.reseed(rng)

    def __len__(self) -> int:
        return len(self._dataset_dicts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._mapper(self._dataset_dicts[idx])


def _seed_worker_augmentations(_worker_id: int, *, base_seed: int, rank: int) -> None:
    """DataLoader ``worker_init_fn``: give each worker an independent
    augmentation RNG.

    The augmentation objects are built once in the main process, so without
    this every worker inherits the *same* ``Generator`` state and replays
    identical flip/scale/jitter/mosaic decisions. ``SeedSequence`` derives a
    provably-independent stream per ``(rank, worker)``; ``reseed`` points the
    worker's whole augmentation pipeline at it. Seeded once per worker
    (``persistent_workers=True``), so the stream advances across epochs.
    """
    info = torch.utils.data.get_worker_info()
    assert info is not None  # only ever called inside a DataLoader worker
    ss = np.random.SeedSequence([base_seed, rank, info.id])
    dataset: Any = info.dataset  # _MappedList / MultiSampleMappedDataset
    dataset.reseed(np.random.default_rng(ss))


def _unwrap_single(batch: list[Any]) -> Any:
    """Collate for batch_size=1: return the single item, not a list."""
    return batch[0]
