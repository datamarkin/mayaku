"""Training engine: assignment, loss, the training loop, evaluation, DDP launch."""

from __future__ import annotations

from mayaku.engine.distributed import (
    all_gather_object,
    all_reduce_dict,
    create_ddp_model,
    get_rank,
    get_world_size,
    init_from_env_if_needed,
    is_main_process,
    launch,
    resolve_ddp_device,
    synchronize,
)

__all__ = [
    "all_gather_object",
    "all_reduce_dict",
    "create_ddp_model",
    "get_rank",
    "get_world_size",
    "init_from_env_if_needed",
    "is_main_process",
    "launch",
    "resolve_ddp_device",
    "synchronize",
]
