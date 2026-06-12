"""Tests for the self-describing checkpoint sidecar.

The writer (:class:`mayaku.engine.PeriodicCheckpointer` with ``metadata=``)
and reader (:func:`mayaku.utils.load_checkpoint_metadata`) are exercised
directly with a tiny module — no training run needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from mayaku.engine import PeriodicCheckpointer
from mayaku.utils import load_checkpoint_metadata


def _save_once(model: nn.Module, out: Path, **kwargs: Any) -> Path:
    """Drive the checkpointer through one ``model_final.pth`` write."""
    ckpt = PeriodicCheckpointer(model, out, period=1, **kwargs)
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


def test_load_checkpoint_metadata_roundtrip(tmp_path: Path) -> None:
    sidecar = {"schema_version": 1, "config": {"model": {"meta_architecture": "faster_rcnn"}}}
    path = _save_once(nn.Linear(2, 2), tmp_path / "run", metadata=sidecar)
    assert load_checkpoint_metadata(path) == sidecar


def test_load_checkpoint_metadata_absent_returns_none(tmp_path: Path) -> None:
    # No metadata= → no sidecar → reader returns None (old/external checkpoint).
    path = _save_once(nn.Linear(2, 2), tmp_path / "run")
    assert load_checkpoint_metadata(path) is None
