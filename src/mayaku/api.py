"""High-level train-and-eval orchestrator.

``mayaku.api.train`` is the single entry point for "load config, train,
pick best checkpoint, eval, return result". It exists so user scripts
don't have to re-implement the orchestration each time — see
``tools/train_mayaku.py`` and ``benchmarks/training_validation/tier3.py``
for the two callers (each ~20-45 lines).

The minimum call:

>>> from pathlib import Path
>>> from mayaku.api import train
>>> result = train(
...     config="configs/detection/faster_rcnn_R_50_FPN_1x.yaml",
...     train_json=Path("/data/coco/annotations/instances_train2017.json"),
...     train_images=Path("/data/coco/train2017"),
... )                                                # doctest: +SKIP

Behaviour:

* ``config`` accepts a YAML path, a string path, or a ready
  :class:`MayakuConfig`. Paths are loaded once; objects pass through.
* ``output_dir`` defaults to ``./runs/<config_stem>/`` when the config
  came from a path, else ``./runs/mayaku_run/``.
* ``val_json`` / ``val_images`` are optional and must be provided
  together. Final eval runs iff both are set. To enable mid-training
  eval, also set ``overrides={"test": {"eval_period": N}}``; without
  the val paths, any non-zero ``eval_period`` from the YAML is
  silently zeroed with a warning so a forgotten val path doesn't make
  training look "fine" while emitting no metrics.
* ``overrides`` is passed straight to :func:`merge_overrides`, so the
  shape is the schema's natural shape (e.g. ``{"solver":
  {"base_lr": 1e-3}}`` or ``{"test": {"eval_period": 5000}}``).
  Invalid keys raise pydantic's standard "Extra inputs are not
  permitted" error.
* ``num_gpus`` (default ``1``) spawns ``num_gpus`` DDP workers via
  :func:`mayaku.engine.launch` when ``> 1``. NCCL on CUDA/ROCm, gloo
  elsewhere. Apply the linear LR scaling rule (multiply
  ``solver.base_lr`` by ``num_gpus``) when scaling up. MPS is
  single-device only and rejects ``num_gpus > 1``.
* ``pretrained_backbone`` (torchvision ImageNet init) is **not** a
  parameter — it's derived from ``cfg.model.backbone.weights_path``:
  if ``weights_path`` is set, the local file wins and torchvision
  weights are not requested; if it's unset, torchvision ImageNet init
  is used. To switch, edit the YAML's ``weights_path`` field.
* The final checkpoint comes from
  :func:`mayaku.utils.select_final_weights` (EMA shadow > live final >
  latest periodic) — same logic both bundled scripts used.

Return value (dict):

* ``final_box_ap`` / ``final_box_ap50`` / ``final_box_ap75``: floats,
  or ``None`` when eval was skipped
* ``final_weights``: ``Path`` to the chosen checkpoint
* ``output_dir``: resolved ``Path`` (useful when the caller passed
  ``None`` and let the API derive it)
* ``train_seconds`` / ``eval_seconds``: wall-clock floats;
  ``eval_seconds`` is ``None`` when eval was skipped
* ``metadata``: full dict written to ``<output_dir>/train/metadata.json``
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from mayaku.cli._weights import resolve_weights
from mayaku.cli.eval import run_eval
from mayaku.cli.train import run_train, run_train_worker
from mayaku.config import MayakuConfig, load_yaml, merge_overrides
from mayaku.config.schemas import DeviceSetting
from mayaku.data import resolve_dataset
from mayaku.engine import launch, resolve_ddp_device
from mayaku.tuning import collect_set_paths
from mayaku.utils import config_from_checkpoint, git_hash, select_final_weights

__all__ = ["train"]


def train(
    config: str | Path | MayakuConfig | None = None,
    *,
    weights: str | Path | None = None,
    data: str | Path | None = None,
    train_json: Path | None = None,
    train_images: Path | None = None,
    val_json: Path | None = None,
    val_images: Path | None = None,
    output_dir: Path | None = None,
    size_budget: int | None = None,
    num_epochs: int | None = None,
    overrides: Mapping[str, Any] | None = None,
    device: DeviceSetting = "auto",
    num_gpus: int = 1,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    """Train, pick the best checkpoint, optionally run final eval.

    Define the model with either ``config`` or ``weights`` (at least one):

    * ``config`` — a YAML path, a bundled config name (e.g.
      ``"faster_rcnn_R_50_FPN_3x"``), or a ready :class:`MayakuConfig`.
      The escape hatch for hand-built or overridden architectures.
    * ``weights`` — a bundled model name (its architecture config is
      looked up and its pretrained ``.pth`` fetched) or a trained ``.pth``
      (its embedded config, written by Task 3, defines the architecture).
      The weights also seed training; the class-specific head re-inits
      when the dataset's class count differs. ``config`` wins when both
      are given. Deriving a config from a ``.pth`` needs a checkpoint
      produced by this version or later — older ones raise, asking for
      ``config``. Only full bundled names resolve (no short aliases).

    Point the dataset at either ``data`` or the explicit paths, not both:

    * ``data`` — a dataset directory or a ``.yaml`` descriptor, resolved
      by :func:`mayaku.data.resolve_dataset`. ``train`` is required; a
      ``val`` split, when present, is used for final eval. Class names
      come from the COCO annotations, not the descriptor.
    * ``train_json`` + ``train_images`` (+ optional ``val_json`` +
      ``val_images``) — the explicit form.

    ``size_budget`` is the first-class form of the compute-budget dial: the
    letterbox canvas is the largest 128-aligned ``(H, W)`` under
    ``size_budget²`` at the data's native aspect (raise it for more resolution,
    lower it for speed). It's equivalent to
    ``overrides={"input": {"size_budget": ...}}`` and wins over both the config
    and ``overrides``.

    ``num_epochs`` is the training-length dial — the number of passes over the
    dataset (the engine resolves it to an iteration count from the dataset size
    and batch). Equivalent to ``overrides={"solver": {"num_epochs": ...}}`` and
    wins over the config and auto-config. Leave it unset to let auto-config pick
    a dataset-adaptive value (or fall back to the schema default of 16).

    **Auto-config vs. manual recipe.** When you pass a ``config`` (YAML path,
    bundled name, or :class:`MayakuConfig`), it is used *verbatim* — auto-config
    is off, so the recipe you wrote is never silently re-tuned. With no
    ``config`` (the ``weights`` + ``data`` path), the recipe is derived from
    your dataset (schedule, LR, anchors, num_classes, augmentation). In both
    cases anything you pass via ``overrides`` or ``size_budget`` is applied last
    and always wins — auto-config never overwrites a field you set explicitly.

    See the module docstring for full parameter semantics and the
    auto-detection rules (pretrained-backbone derivation, no-val
    short-circuit, output-dir defaulting). Returns a result dict.

    Final eval runs iff both ``val_json`` and ``val_images`` are set.
    For mid-training eval, pass
    ``overrides={"test": {"eval_period": N}}``.
    """
    # --- Resolve the dataset source ---------------------------------------
    if data is not None:
        if any(p is not None for p in (train_json, train_images, val_json, val_images)):
            raise ValueError(
                "Pass either data= or the explicit train_json/train_images"
                "(/val_json/val_images) paths, not both."
            )
        splits = resolve_dataset(data)
        train_images, train_json = splits["train"]
        if "val" in splits:
            val_images, val_json = splits["val"]
    elif train_json is None or train_images is None:
        raise ValueError(
            "Provide data= (a dataset directory or .yaml descriptor) or both "
            "train_json and train_images."
        )

    # --- Validate inputs early --------------------------------------------
    # Guaranteed set by the resolution block above (data= populates them,
    # the explicit path raises if either is missing).
    assert train_json is not None and train_images is not None
    if not train_json.exists():
        raise FileNotFoundError(f"train_json not found: {train_json}")
    if not train_images.is_dir():
        raise NotADirectoryError(f"train_images is not a directory: {train_images}")
    if (val_json is None) != (val_images is None):
        raise ValueError(
            "val_json and val_images must both be provided, or both omitted; "
            f"got val_json={val_json!r}, val_images={val_images!r}"
        )
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1; got {num_gpus}")

    # --- Resolve the model source (config and/or weights) -----------------
    # ``config`` wins; otherwise the architecture comes from ``weights``
    # (a bundled name or a trained .pth's embedded config). ``detector_weights``
    # is the checkpoint to load when ``weights`` was given, else None.
    cfg, config_stem, detector_weights = _resolve_model_source(config, weights)

    # --- Apply user overrides + record which fields the user pinned --------
    # Everything the user passes here is an explicit choice that must survive
    # auto-config. The values are merged into ``cfg`` now; their dotted paths
    # go into ``pinned_paths`` so the dataset-derived recipe (which runs inside
    # run_train on the no-config path) skips them and the user's value wins.
    pinned_paths: set[str] = set()
    if overrides:
        cfg = merge_overrides(cfg, overrides)
        pinned_paths |= collect_set_paths(overrides)
    # size_budget is the first-class form of the most common knob; applied last
    # so the explicit arg wins over the config and overrides. Schema validation
    # (positive, stride-32 multiple) runs inside merge_overrides.
    if size_budget is not None:
        cfg = merge_overrides(cfg, {"input": {"size_budget": size_budget}})
        pinned_paths.add("input.size_budget")
    # num_epochs is the first-class training-length knob (passes over the
    # dataset). Like size_budget, an explicit value wins over the config and
    # auto-config.
    if num_epochs is not None:
        cfg = merge_overrides(cfg, {"solver": {"num_epochs": num_epochs}})
        pinned_paths.add("solver.num_epochs")

    # A config (YAML path, bundled name, or MayakuConfig) means "train exactly
    # this recipe": auto-config is turned off so the config is used verbatim and
    # never silently re-tuned. Without a config (the weights= fine-tune path),
    # the recipe is derived from THIS dataset, so auto-config is forced on — the
    # architecture still comes from the checkpoint sidecar, but its baked-in
    # auto_config flag (set at the model's original training time) must not
    # disable re-tuning for the new dataset. Either way ``pinned_paths`` is
    # never overwritten.
    cfg = merge_overrides(cfg, {"auto_config": {"enabled": config is None}})

    # --- No-val short-circuit (after overrides, so eval_period is final) --
    eval_after = val_json is not None
    if not eval_after and cfg.test.eval_period > 0:
        warnings.warn(
            "test.eval_period > 0 but no val_json/val_images provided — "
            "mid-training eval disabled. Pass val_json + val_images to enable.",
            stacklevel=2,
        )
        cfg = merge_overrides(cfg, {"test": {"eval_period": 0}})

    # --- Resolve output_dir -----------------------------------------------
    resolved_output_dir = output_dir if output_dir is not None else Path("./runs") / config_stem
    train_dir = resolved_output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # ``weights=`` (resolved above) wins; else fall back to the config's
    # own ``model.weights`` for the YAML-driven fine-tune path.
    if detector_weights is None and cfg.model.weights is not None:
        detector_weights = Path(cfg.model.weights)
    # Torchvision ImageNet init only when nothing else seeds the weights.
    pretrained_backbone = cfg.model.backbone.weights_path is None and detector_weights is None

    # Resume restores the full training state from a checkpoint, so it
    # supersedes any weight init — drop the warm-start sources to satisfy
    # run_train's mutual-exclusivity check.
    resume_path = Path(resume) if resume is not None else None
    if resume_path is not None:
        detector_weights = None
        pretrained_backbone = False

    print(f"[mayaku.train] {config_stem} -> {resolved_output_dir}")

    # --- Train -----------------------------------------------------------
    # run_train persists the resolved config to train_dir/config.yaml
    # internally, so re-running eval / export against the run's
    # artefacts uses the exact same config that produced them.
    # Only forward val paths to run_train when mid-training eval is
    # enabled; otherwise run_train warns about "val supplied but
    # eval_period=0". Final eval below uses the val paths regardless.
    forward_val = cfg.test.eval_period > 0
    train_start = time.time()
    if num_gpus == 1:
        run_train(
            cfg,
            coco_gt_json=train_json,
            image_root=train_images,
            output_dir=train_dir,
            weights=detector_weights,
            pretrained_backbone=pretrained_backbone,
            device=device,
            val_json=val_json if forward_val else None,
            val_image_root=val_images if forward_val else None,
            resume=resume_path,
            user_set_paths=pinned_paths,
        )
    else:
        # Multi-GPU DDP: spawn ``num_gpus`` workers via :func:`launch`.
        # Each worker calls ``run_train`` and brings up its own slice of
        # the process group; we don't run any GPU work in the parent.
        # Post-train (select_final_weights, final eval, metadata) stays
        # on the parent so the return value comes back from a single
        # well-defined caller, not N racy workers.
        dev = resolve_ddp_device(device, num_gpus)
        launch(
            run_train_worker,
            num_gpus,
            device=dev,
            args=(
                cfg,
                train_json,
                train_images,
                train_dir,
                detector_weights,  # weights
                pretrained_backbone,
                device,
                None,  # num_epochs (cfg already carries it)
                20,  # log_period default
                val_json if forward_val else None,
                val_images if forward_val else None,
                resume_path,
                pinned_paths,
            ),
        )
    train_seconds = time.time() - train_start
    print(f"[mayaku.train] training done in {train_seconds:.0f}s ({train_seconds / 3600:.2f}h)")

    # --- Pick the canonical "final" checkpoint ----------------------------
    final_weights = select_final_weights(train_dir)
    print(f"[mayaku.train] final weights: {final_weights}")

    # --- Optional final eval ----------------------------------------------
    # The architecture comes from the checkpoint's embedded sidecar (the
    # resolved config run_train wrote into it — auto-config adjustments and
    # all), so eval needs only the weights.
    bbox: dict[str, Any] = {}
    eval_seconds: float | None = None
    if eval_after:
        assert val_json is not None and val_images is not None
        eval_start = time.time()
        # run_eval feeds the device string to torch.device(), which does
        # not understand the "auto" sentinel — resolve it here.
        eval_device = device
        if eval_device == "auto":
            eval_device = "cuda" if torch.cuda.is_available() else "cpu"
        metrics = run_eval(
            final_weights,
            coco_gt_json=val_json,
            image_root=val_images,
            output_dir=resolved_output_dir / "eval",
            device=eval_device,
        )
        eval_seconds = time.time() - eval_start
        raw_bbox = metrics.get("bbox") if isinstance(metrics, dict) else None
        if isinstance(raw_bbox, dict):
            bbox = raw_bbox

    final_box_ap = float(bbox["AP"]) if "AP" in bbox else None
    final_box_ap50 = float(bbox["AP50"]) if "AP50" in bbox else None
    final_box_ap75 = float(bbox["AP75"]) if "AP75" in bbox else None
    if final_box_ap is not None:
        print(f"[mayaku.train] box AP: {final_box_ap:.4f} ({final_box_ap * 100:.2f})")

    # --- Metadata ---------------------------------------------------------
    # Record CUDA info only when CUDA was actually usable for this run.
    # `torch.version.cuda` is non-None whenever the torch build supports
    # CUDA — even on CPU-only hosts — so reporting it unconditionally is
    # misleading. Gate on the runtime decision so the file accurately
    # describes where training ran.
    used_cuda = device in ("cuda", "auto") and torch.cuda.is_available()
    metadata: dict[str, Any] = {
        "config_stem": config_stem,
        "backbone": cfg.model.backbone.name,
        "weights_path": cfg.model.backbone.weights_path,
        "num_classes": cfg.model.roi_heads.num_classes,
        "num_gpus": num_gpus,
        "num_epochs": cfg.solver.num_epochs,
        "ims_per_batch": cfg.solver.ims_per_batch,
        "grad_accum_steps": cfg.solver.grad_accum_steps,
        # Single-rank effective batch (ims_per_batch × grad_accum_steps); the
        # cross-rank total is this × num_gpus. Apply the linear LR scaling rule
        # against the cross-rank value when scaling up.
        "effective_batch_size": cfg.solver.effective_batch(),
        "base_lr": cfg.solver.base_lr,
        "ema_enabled": cfg.solver.ema_enabled,
        "final_weights": str(final_weights),
        "final_box_ap": final_box_ap,
        "final_box_ap50": final_box_ap50,
        "final_box_ap75": final_box_ap75,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "git_hash": git_hash(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if used_cuda else None,
        "device_name": torch.cuda.get_device_name(0) if used_cuda else None,
    }
    (train_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return {
        "final_box_ap": final_box_ap,
        "final_box_ap50": final_box_ap50,
        "final_box_ap75": final_box_ap75,
        "final_weights": final_weights,
        "output_dir": resolved_output_dir,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Model-source resolution (config and/or weights)
# ---------------------------------------------------------------------------


def _resolve_model_source(
    config: str | Path | MayakuConfig | None,
    weights: str | Path | None,
) -> tuple[MayakuConfig, str, Path | None]:
    """Resolve ``(cfg, config_stem, detector_weights)`` from the inputs.

    ``config`` wins when present; otherwise the architecture is derived
    from ``weights``. ``detector_weights`` is the checkpoint to load when
    ``weights`` was given (a bundled name's fetched ``.pth`` or the passed
    ``.pth``), else ``None`` — the caller may still read
    ``cfg.model.weights``.
    """
    if config is not None:
        cfg, stem = _load_config(config)
        return cfg, stem, resolve_weights(weights) if weights is not None else None
    if weights is not None:
        weights_path = resolve_weights(weights)
        assert weights_path is not None  # weights is not None on this path
        cfg, _ = config_from_checkpoint(weights_path)
        return cfg, weights_path.stem, weights_path
    raise ValueError(
        "Provide config= (a YAML path or MayakuConfig) or "
        "weights= (a bundled model name or a trained .pth) so the model "
        "architecture is defined."
    )


def _load_config(config: str | Path | MayakuConfig) -> tuple[MayakuConfig, str]:
    """Load an explicit ``config``: a ``MayakuConfig`` object or a YAML file path.

    ``config`` is a maintainer escape hatch for defining/training a new
    architecture; end-user flows use ``weights=`` and never pass it. Bundled
    config *names* are no longer resolved — configs are maintainer references
    under ``configs/`` and are passed by path.
    """
    if isinstance(config, MayakuConfig):
        return config, "mayaku_run"
    path = Path(config)
    if path.exists():
        return load_yaml(path), path.stem
    raise FileNotFoundError(
        f"config file not found: {config}. Pass a .yaml path or a MayakuConfig — "
        "bundled config names are no longer resolved (configs are maintainer "
        "references under configs/)."
    )
