"""Training engine: trainers, hooks, optim helpers, DDP launch."""

from __future__ import annotations

from mayaku.engine.callbacks import (
    EvalHook,
    HookBase,
    IterationTimer,
    LRScheduler,
    MemoryTrim,
    MetricsPrinter,
    PeriodicCheckpointer,
)
from mayaku.engine.distributed import (
    all_gather_object,
    all_reduce_dict,
    create_ddp_model,
    get_rank,
    get_world_size,
    is_main_process,
    launch,
    synchronize,
)
from mayaku.engine.ema import EMAHook, ModelEMA
from mayaku.engine.evaluator import (
    COCOEvaluator,
    DatasetEvaluator,
    inference_on_dataset,
    instances_to_coco_json,
)
from mayaku.engine.optim import build_lr_scheduler, build_optimizer
from mayaku.engine.trainer import AMPTrainer, SimpleTrainer, TrainerBase

__all__ = [
    "AMPTrainer",
    "COCOEvaluator",
    "DatasetEvaluator",
    "EMAHook",
    "EvalHook",
    "HookBase",
    "IterationTimer",
    "LRScheduler",
    "MemoryTrim",
    "MetricsPrinter",
    "ModelEMA",
    "PeriodicCheckpointer",
    "SimpleTrainer",
    "TrainerBase",
    "all_gather_object",
    "all_reduce_dict",
    "build_lr_scheduler",
    "build_optimizer",
    "create_ddp_model",
    "get_rank",
    "get_world_size",
    "inference_on_dataset",
    "instances_to_coco_json",
    "is_main_process",
    "launch",
    "synchronize",
]
