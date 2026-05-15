"""Checkpoint-handling helpers shared between train scripts and the
:func:`mayaku.api.train` orchestrator.

The three functions here were previously duplicated across
``tools/train_mayaku.py`` and ``benchmarks/training_validation/tier3.py``
in slightly different shapes; centralising them removes ~30 lines of
copy-paste and gives both scripts the same behaviour for the
EMA-checkpoint quirk (the EMA shadow stores ``num_batches_tracked``
buffers that won't ``strict=True``-load unless stripped first).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch

__all__ = ["git_hash", "select_final_weights", "strip_num_batches_tracked"]


def select_final_weights(train_dir: Path) -> Path:
    """Pick the canonical "final" checkpoint from ``train_dir``.

    Preference order, highest first:

    1. ``train_dir / "ema" / "model_final.pth"`` — EMA shadow, typically
       +0.3-0.5 box AP over the live weights. If present, this function
       also strips ``num_batches_tracked`` from it in-place so the file
       loads with ``strict=True`` (the EMA shadow accumulates buffers
       the live model doesn't have an entry for).
    2. ``train_dir / "model_final.pth"`` — live final.
    3. The latest ``model_iter_*.pth`` checkpoint — training crashed
       before writing ``model_final.pth``; fall back to the most-recent
       periodic checkpoint.

    Raises ``RuntimeError`` if none of the three exist — training
    likely failed before the first checkpoint period.
    """
    ema_final = train_dir / "ema" / "model_final.pth"
    if ema_final.exists():
        strip_num_batches_tracked(ema_final)
        return ema_final

    live_final = train_dir / "model_final.pth"
    if live_final.exists():
        return live_final

    candidates = sorted(train_dir.glob("model_iter_*.pth"))
    if candidates:
        return candidates[-1]

    raise RuntimeError(
        f"no checkpoint produced under {train_dir} — training likely "
        "failed before the first checkpoint period."
    )


def strip_num_batches_tracked(checkpoint_path: Path) -> None:
    """Remove ``num_batches_tracked`` entries from a saved state-dict.

    The EMA shadow tracks BN module buffers including
    ``num_batches_tracked`` (an int counter), but the live model's
    state-dict doesn't expose it as a trainable / loadable key — so
    loading the EMA checkpoint with ``strict=True`` fails with
    "unexpected keys". Strip in place so the file is drop-in compatible
    with eval / predict / export paths.

    Idempotent: if the checkpoint has no ``num_batches_tracked`` keys
    (already stripped, or BN-free model), the file is not rewritten.
    Cheap re-callability matters for ConvNeXt-Large where the EMA
    checkpoint is ~800 MB on disk.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not any(k.endswith("num_batches_tracked") for k in state["model"]):
        return
    state["model"] = {
        k: v for k, v in state["model"].items() if not k.endswith("num_batches_tracked")
    }
    torch.save(state, checkpoint_path)


def git_hash() -> str | None:
    """Best-effort short git hash for metadata.json.

    Returns ``None`` on non-git checkouts, when ``git`` isn't on PATH,
    or when any other error makes the command fail. Never raises.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
