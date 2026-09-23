"""The Python entry points: `mayaku.train` and `mayaku.evaluate`.

`train` does what a training script would: resolve the config, read the
data, let auto-config fill what the user left unset, build the model,
warm-start it, train, and score the result the way it will be deployed.

    >>> import mayaku
    >>> result = mayaku.train(                              # doctest: +SKIP
    ...     weights="mayaku-n",
    ...     train_annotations="data/train/_annotations.coco.json",
    ...     train_images="data/train",
    ...     val_annotations="data/valid/_annotations.coco.json",
    ...     val_images="data/valid",
    ... )
    >>> result["final_weights"]                            # doctest: +SKIP
    PosixPath('runs/mayaku-n/train/best.pt')

A run directory holds, under ``train/``, the resolved ``config.yaml``, the
engine's ``recipe.json`` / ``tier.json`` / ``log.jsonl``, the self-describing
``best.pt`` and ``last.pt``, the resumable ``state.pt`` and ``metadata.json``;
the final evaluation goes to ``eval/metrics.json``.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from mayaku.backends.device import DeviceKind
from mayaku.config import MayakuConfig, dump_yaml, merge_overrides, read_yaml
from mayaku.data import CocoDetection
from mayaku.data.coco import CocoLabels, load_coco
from mayaku.engine.distributed import (
    broadcast_from_main,
    get_world_size,
    init_from_env_if_needed,
    is_main_process,
    launch,
    local_device,
    resolve_ddp_device,
)
from mayaku.engine.evaluation import evaluate_runner
from mayaku.engine.trainer import load_state
from mayaku.engine.trainer import train as run
from mayaku.inference import from_pretrained
from mayaku.model import load_pretrained
from mayaku.tuning import apply_auto_config, collect_set_paths
from mayaku.utils.checkpoint import (
    SIDECAR_KEY,
    build_sidecar,
    git_hash,
    read_deploy_checkpoint,
    select_final_weights,
)
from mayaku.utils.download import resolve_weights

__all__ = ["evaluate", "train"]


def train(
    config: str | Path | MayakuConfig | None = None,
    *,
    weights: str | Path | None = None,
    train_annotations: str | Path | None = None,
    train_images: str | Path | None = None,
    val_annotations: str | Path | None = None,
    val_images: str | Path | None = None,
    output_dir: str | Path | None = None,
    size_budget: int | None = None,
    num_epochs: int | None = None,
    overrides: Mapping[str, Any] | None = None,
    device: str = "auto",
    num_gpus: int = 1,
    resume: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train a detector on a COCO split; returns a result dict.

    The model comes from ``config``, ``weights``, or both:

    * ``config``: a YAML path or a `MayakuConfig`. Without ``weights`` the
      model trains from scratch.
    * ``weights``: a checkpoint path or a hosted model name. It warm-starts
      training (the classifier re-initialises when the class count differs)
      and, without ``config``, also defines the architecture: its tier,
      heads and QAT setting, with everything else derived afresh.

    Auto-config (`mayaku.tuning.apply_auto_config`) fills the fields the user
    did not set -- the class count, the canvas, and when warm-starting the
    fine-tune schedule -- from the training annotations. What counts as set:
    the keys written in a YAML config, the fields set on a `MayakuConfig`,
    everything in ``overrides`` (a nested mapping, e.g.
    ``{"train": {"lr": 0.005}}``), ``size_budget`` (``input.size_budget``)
    and ``num_epochs`` (``train.epochs``).

    With ``val_annotations`` and ``val_images`` the run evaluates every
    epoch, keeps the best EMA model as ``best.pt``, and scores that
    checkpoint through the deploy path (`evaluate`) at the end. Without
    them it trains blind and ``last.pt`` is the result.

    ``num_gpus`` > 1 trains on that many GPUs of this machine, one process
    each (``device="cpu"`` runs the ranks on the CPU, for testing). Under
    ``torchrun`` (multi-node) every process calls this, ``num_gpus`` stays 1,
    and every process returns rank 0's result. The
    recipe's ``batch`` is the global batch, so the GPU count never changes
    what is trained: each GPU takes ``batch / GPUs`` images a step.

    ``resume`` continues an interrupted run from its ``state.pt`` (or the
    run's ``train/`` directory). The config comes from the state, so
    ``config``, ``weights``, ``size_budget``, ``num_epochs`` and
    ``overrides`` must be left unset; pass the same dataset paths.

    Result keys: ``final_weights``, ``output_dir``, ``metrics`` (the final
    evaluation, None without a val split), ``best`` (the trainer's best
    epoch metrics), ``train_seconds``, ``eval_seconds`` and ``metadata``.
    """
    if train_annotations is None or train_images is None:
        raise ValueError("train_annotations (a COCO JSON) and train_images (its image "
                         "directory) are required")
    train_annotations, train_images = Path(train_annotations), Path(train_images)
    _check_split(train_annotations, train_images)
    val: tuple[Path, Path] | None = None
    if val_annotations is not None and val_images is not None:
        val = Path(val_annotations), Path(val_images)
        _check_split(*val)
    elif (val_annotations, val_images) != (None, None):
        raise ValueError("val_annotations and val_images go together")
    # Under torchrun every process runs this call and joins the group here;
    # spawned ranks (num_gpus) start below, once the config and data are resolved.
    dev = resolve_ddp_device(device, num_gpus)
    init_from_env_if_needed(dev)
    if get_world_size() > 1 and num_gpus != 1:
        raise ValueError("under torchrun the process count is torchrun's; leave num_gpus=1")
    world = max(num_gpus, get_world_size())
    if not is_main_process():
        log = _quiet

    state = None
    if resume is not None:
        if overrides or (config, weights, size_budget, num_epochs) != (None,) * 4:
            raise ValueError("resume restores the run's own config; leave config, weights, "
                             "size_budget, num_epochs and overrides unset")
        state = load_state(resume)
        resumed = state[SIDECAR_KEY]
        cfg, stem = MayakuConfig.model_validate(resumed["config"]), "resume"
        train_dir = Path(resume) if Path(resume).is_dir() else Path(resume).parent
        run_dir = train_dir.parent if output_dir is None else Path(output_dir)
        pretrained = None
    else:
        cfg, stem, pinned, pretrained = _resolve_model(config, weights)
        extra: dict[str, Any] = {}
        if size_budget is not None:
            extra["input"] = {"size_budget": size_budget}
        if num_epochs is not None:
            extra["train"] = {"epochs": num_epochs}
        for ov in (overrides, extra):
            if ov:
                cfg = merge_overrides(cfg, ov)
                pinned |= collect_set_paths(ov)
        run_dir = Path(output_dir) if output_dir is not None else Path("runs") / stem

    kp = cfg.model.keypoints
    labels = dict(masks=cfg.model.seg, kpt=kp.num if kp else 0)
    coco = load_coco(str(train_images), str(train_annotations), **labels)
    if state is None:
        cfg, changes = apply_auto_config(cfg, coco, pinned, finetune=pretrained is not None)
        for path, old, new in changes:
            log(f"[mayaku.train] auto-config {path}: {old} -> {new}")
    elif resumed["class_names"] != coco.class_names:
        raise ValueError(f"{train_annotations} is not the dataset this run was training on")

    if cfg.train.batch % world:          # the trainer checks too; this fails before spawning
        raise ValueError(f"train.batch {cfg.train.batch} does not split over {world} GPUs; "
                         f"use a multiple of {world}")
    canvas = cfg.input.canvas
    train_dir = run_dir / "train"
    job = _Job(cfg, coco, train_images, train_annotations, val, pretrained, state,
               train_dir, dev.kind)
    device = local_device(dev.kind)
    if is_main_process():
        train_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(cfg, train_dir / "config.yaml")
    per_gpu = f", {world} GPUs x {cfg.train.batch // world}" if world > 1 else ""
    log(f"[mayaku.train] tier {cfg.model.tier}, {len(coco.cat_ids)} classes, canvas "
        f"{canvas[0]}x{canvas[1]}, {cfg.train.epochs} epochs, batch {cfg.train.batch}"
        f"{per_gpu} on {dev.kind} -> {train_dir}")

    t0 = time.time()
    if num_gpus > 1:
        launch(_run_job, num_gpus, device=dev, args=(job,))
    else:
        _run_job(job, log)
    train_seconds = time.time() - t0
    if not is_main_process():
        return broadcast_from_main({})            # rank 0's result, once it has one
    best_path = train_dir / "best.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    final_weights = select_final_weights(train_dir)
    log(f"[mayaku.train] done in {train_seconds / 3600:.2f}h; final weights {final_weights}")

    metrics, eval_seconds = None, None
    if val:
        t0 = time.time()
        metrics = evaluate(final_weights, annotations=val[0], images=val[1],
                           output_dir=run_dir / "eval", device=device)
        eval_seconds = time.time() - t0
        log(f"[mayaku.train] box AP {metrics['AP']:.4f}")

    cuda = device.startswith("cuda")
    metadata = {
        "config_stem": stem,
        "tier": cfg.model.tier,
        "num_classes": cfg.model.num_classes,
        "canvas_hw": list(canvas),
        "qat": cfg.model.qat_enabled,
        "epochs": cfg.train.epochs,
        "batch": cfg.train.batch,
        "world_size": world,
        "lr": cfg.train.lr,
        "final_weights": str(final_weights),
        "best": best,
        "metrics": metrics,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "git_hash": git_hash(),
        "torch_version": torch.__version__,
        "device": device,
        "device_name": torch.cuda.get_device_name(device) if cuda else None,
    }
    (train_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return broadcast_from_main({
        "final_weights": final_weights,
        "output_dir": run_dir,
        "metrics": metrics,
        "best": best,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "metadata": metadata,
    })


def evaluate(
    weights: str | Path,
    *,
    annotations: str | Path,
    images: str | Path,
    output_dir: str | Path | None = None,
    device: str = "auto",
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """COCO metrics of a trained model on a split, measured the way it
    deploys: ``weights`` is anything `mayaku.from_pretrained` loads -- a
    checkpoint, a hosted model name, or an exported artifact, which then runs
    in its own runtime. Returns the flat metrics dict (``AP``, ``AP50``, ...,
    with ``segm_*`` and ``kpt_*`` for models with those heads), also written
    to ``<output_dir>/metrics.json`` when ``output_dir`` is set."""
    annotations, images = Path(annotations), Path(images)
    _check_split(annotations, images)
    runner = from_pretrained(weights, device=device)
    metrics: dict[str, Any] = evaluate_runner(runner, images, annotations, log=log)
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _resolve_model(
    config: str | Path | MayakuConfig | None, weights: str | Path | None,
) -> tuple[MayakuConfig, str, set[str], dict[str, Any] | None]:
    """``(cfg, run name, user-set paths, pretrained state or None)`` from the
    model source(s); see `train`."""
    pretrained, ckpt_cfg, stem = None, None, "mayaku_run"
    if weights is not None:
        path = resolve_weights(weights)
        _, ckpt_cfg, pretrained = read_deploy_checkpoint(path)
        stem = path.stem
    if isinstance(config, MayakuConfig):
        # the fields set on it are the user's; a recipe given whole is all set
        return config, stem, collect_set_paths(config.model_dump(exclude_unset=True)), pretrained
    if config is not None:
        raw, cfg = read_yaml(config)
        return cfg, Path(config).stem, collect_set_paths(raw), pretrained
    if ckpt_cfg is None:
        return MayakuConfig(), stem, set(), None
    # The checkpoint defines the network; the data and this run define the rest.
    return MayakuConfig(model=ckpt_cfg.model.architecture()), stem, set(), pretrained


@dataclasses.dataclass
class _Job:
    """What each training rank needs: the resolved config and the parsed
    training labels (sent to spawned ranks, not re-parsed), the splits, the
    warm-start or resume state, and where to write."""

    cfg: MayakuConfig
    coco: CocoLabels
    train_images: Path
    train_annotations: Path
    val: tuple[Path, Path] | None
    pretrained: dict[str, Any] | None
    state: dict[str, Any] | None
    train_dir: Path
    device_kind: DeviceKind


def _run_job(job: _Job, log: Callable[[str], None] = print) -> None:
    """Build the datasets and the model and train: the part every rank runs.
    Only rank 0 evaluates and writes, so only it reads the validation split
    and describes the checkpoints."""
    cfg, main = job.cfg, is_main_process()
    if not main:
        log = _quiet
    kp = cfg.model.keypoints
    labels: dict[str, Any] = dict(masks=cfg.model.seg, kpt=kp.num if kp else 0)
    canvas = cfg.input.canvas
    train_ds = CocoDetection(str(job.train_images), str(job.train_annotations), canvas,
                             aug=cfg.train.aug, seed=cfg.train.seed, coco=job.coco, **labels)
    val_ds = None
    if job.val and main:
        val_ds = CocoDetection(str(job.val[1]), str(job.val[0]), canvas, **labels)
    model = cfg.model.build(canvas)
    if job.pretrained is not None:
        info = load_pretrained(model, job.pretrained)
        if info["reinitialised"]:
            log(f"[mayaku.train] classifier re-initialised for {model.nc} classes")
    run(model, train_ds, val_ds, cfg.train, device=local_device(job.device_kind),
        out=str(job.train_dir), workers=cfg.dataloader.num_workers, log=log,
        sidecar=build_sidecar(cfg, job.coco.class_names, model) if main else None,
        resume=job.state)


def _quiet(*_: Any) -> None:
    """The log of a rank other than 0."""


def _check_split(annotations: Path, images: Path) -> None:
    if not annotations.is_file():
        raise FileNotFoundError(f"annotation file not found: {annotations}")
    if not images.is_dir():
        raise NotADirectoryError(f"image directory not found: {images}")

