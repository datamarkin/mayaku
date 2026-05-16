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

from mayaku.cli.eval import run_eval
from mayaku.cli.train import run_train, run_train_worker
from mayaku.config import MayakuConfig, load_yaml, merge_overrides
from mayaku.config.schemas import DeviceSetting
from mayaku.engine import launch, resolve_ddp_device
from mayaku.utils import git_hash, select_final_weights

__all__ = ["train"]


def train(
    config: str | Path | MayakuConfig,
    *,
    train_json: Path,
    train_images: Path,
    val_json: Path | None = None,
    val_images: Path | None = None,
    output_dir: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    device: DeviceSetting = "auto",
    num_gpus: int = 1,
) -> dict[str, Any]:
    """Train, pick the best checkpoint, optionally run final eval.

    See the module docstring for full parameter semantics and the
    auto-detection rules (pretrained-backbone derivation, no-val
    short-circuit, output-dir defaulting). Returns a result dict.

    Final eval runs iff both ``val_json`` and ``val_images`` are set.
    For mid-training eval, pass
    ``overrides={"test": {"eval_period": N}}``.
    """
    # --- Validate inputs early --------------------------------------------
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

    # --- Normalize config to MayakuConfig + remember its source -----------
    if isinstance(config, MayakuConfig):
        cfg = config
        config_stem = "mayaku_run"
    else:
        config_path = Path(config)
        cfg = load_yaml(config_path)
        config_stem = config_path.stem

    # --- Apply overrides --------------------------------------------------
    if overrides:
        cfg = merge_overrides(cfg, overrides)

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

    pretrained_backbone = cfg.model.backbone.weights_path is None

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
            pretrained_backbone=pretrained_backbone,
            device=device,
            val_json=val_json if forward_val else None,
            val_image_root=val_images if forward_val else None,
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
                None,  # weights
                pretrained_backbone,
                device,
                None,  # max_iter (cfg already carries it)
                20,  # log_period default
                val_json if forward_val else None,
                val_images if forward_val else None,
            ),
        )
    train_seconds = time.time() - train_start
    print(f"[mayaku.train] training done in {train_seconds:.0f}s ({train_seconds / 3600:.2f}h)")

    # --- Pick the canonical "final" checkpoint ----------------------------
    final_weights = select_final_weights(train_dir)
    print(f"[mayaku.train] final weights: {final_weights}")

    # --- Optional final eval ----------------------------------------------
    bbox: dict[str, Any] = {}
    eval_seconds: float | None = None
    if eval_after:
        assert val_json is not None and val_images is not None
        eval_start = time.time()
        metrics = run_eval(
            cfg,
            weights=final_weights,
            coco_gt_json=val_json,
            image_root=val_images,
            output_dir=resolved_output_dir / "eval",
            device=device,
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
        "max_iter": cfg.solver.max_iter,
        "ims_per_batch": cfg.solver.ims_per_batch,
        "grad_accum_steps": cfg.solver.grad_accum_steps,
        # ``ims_per_batch`` is per-rank; cross-rank effective batch is
        # this value × num_gpus × grad_accum_steps. Apply the linear LR
        # scaling rule against the cross-rank value when scaling up.
        "effective_batch_size": cfg.solver.ims_per_batch * cfg.solver.grad_accum_steps,
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
