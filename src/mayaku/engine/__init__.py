"""Training engine: assignment, loss, the training loop, evaluation, DDP launch."""

from __future__ import annotations

from mayaku.engine.distributed import get_world_size, is_main_process, launch, local_device

__all__ = ["get_world_size", "is_main_process", "launch", "local_device"]
