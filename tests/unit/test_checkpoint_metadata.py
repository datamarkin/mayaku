"""Tests for the self-describing checkpoint sidecar.

The writer (:class:`mayaku.engine.PeriodicCheckpointer` with ``metadata=``)
and reader (:func:`mayaku.utils.load_checkpoint`) are exercised directly with
a tiny module — no training run needed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from mayaku.engine import PeriodicCheckpointer
from mayaku.utils import load_checkpoint
from mayaku.utils.checkpoint import class_names_from_checkpoint


def _save_once(model: nn.Module, out: Path, **kwargs: Any) -> Path:
    """Drive the checkpointer through one ``model_final.pth`` write."""
    ckpt = PeriodicCheckpointer(model, out, period=1, **kwargs)
    # after_train reads trainer.iter for the checkpoint's iteration field;
    # register_hooks binds this in real runs.
    ckpt.trainer = SimpleNamespace(iter=0)  # type: ignore[assignment]
    ckpt.before_train()
    ckpt.after_train()
    return out / "model_final.pth"


def test_metadata_written_beside_pure_state_dict(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    sidecar = {"schema_version": 1, "class_names": ["cat", "dog"]}
    path = _save_once(model, tmp_path / "run", metadata=sidecar)

    state = torch.load(path, map_location="cpu", weights_only=False)
    # Sidecar present, and "model" is still a pure state_dict (its keys are
    # the module's tensors only — no "mayaku" leaking into the weights).
    assert state["mayaku"] == sidecar
    assert set(state["model"]) == set(model.state_dict())


def test_load_checkpoint_sidecar_roundtrip(tmp_path: Path) -> None:
    sidecar = {"schema_version": 1, "config": {"model": {"meta_architecture": "faster_rcnn"}}}
    path = _save_once(nn.Linear(2, 2), tmp_path / "run", metadata=sidecar)
    assert load_checkpoint(path)[0] == sidecar


def test_class_names_from_checkpoint_roundtrip(tmp_path: Path) -> None:
    # The evaluator reads the model's training class identity from here (C7).
    sidecar = {"schema_version": 1, "class_names": ["cat", "dog"]}
    path = _save_once(nn.Linear(2, 2), tmp_path / "run", metadata=sidecar)
    assert class_names_from_checkpoint(path) == ["cat", "dog"]


def test_class_names_from_checkpoint_absent_is_none(tmp_path: Path) -> None:
    # No sidecar (or no names) → None, so the evaluator falls back to positional.
    path = _save_once(nn.Linear(2, 2), tmp_path / "run")
    assert class_names_from_checkpoint(path) is None


def test_load_checkpoint_sidecar_absent_is_none(tmp_path: Path) -> None:
    # No metadata= → no sidecar → reader returns None (old/external checkpoint).
    path = _save_once(nn.Linear(2, 2), tmp_path / "run")
    assert load_checkpoint(path)[0] is None
