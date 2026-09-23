"""Typed configuration for mayaku v3.

One `MayakuConfig` describes a model and how it is trained: which tier with
which heads (`model`), the input canvas (`input`), the training recipe
(`train`), data loading (`dataloader`) and dataset-aware auto-tuning
(`auto_config`). It is what a run writes next to its weights and what every
checkpoint and exported artifact embeds, so a model can always be rebuilt
from its own files.

`train` is the engine's own `mayaku.engine.trainer.Recipe` dataclass,
validated in place rather than restated as a second schema, so the recipe's
defaults live in exactly one place.

All models are frozen and reject unknown fields; derive a variant with
`model_copy(update={...})` or `mayaku.config.merge_overrides`.
"""

from __future__ import annotations

import dataclasses
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mayaku.engine.trainer import Recipe
from mayaku.model import Detector, enable_qat
from mayaku.model.blocks import as_canvas
from mayaku.model.tiers import QAT_TIERS, TIERS, Tier

__all__ = [
    "AutoConfig",
    "DataLoaderConfig",
    "InputConfig",
    "KeypointConfig",
    "MayakuConfig",
    "ModelConfig",
    "TierName",
]

TierName = Literal["n", "s", "m", "l"]


class _BaseModel(BaseModel):
    """Shared pydantic config: immutable, strict, validate defaults."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class KeypointConfig(_BaseModel):
    """Keypoints per instance, and their names and left/right flip pairs as
    the training annotations define them (the dataset derives the pairs from
    the names). The OKS falloff constants are the model's own
    (`mayaku.model.kpt.sigmas`), not configuration."""

    num: Annotated[int, Field(gt=0)]
    names: tuple[str, ...] = ()
    flip_pairs: tuple[tuple[int, int], ...] = ()

    @model_validator(mode="after")
    def _lengths_match(self) -> KeypointConfig:
        if self.names and len(self.names) != self.num:
            raise ValueError(f"keypoints.names has {len(self.names)} entries for {self.num} keypoints")
        if any(not (0 <= a < self.num and 0 <= b < self.num) for a, b in self.flip_pairs):
            raise ValueError(f"keypoints.flip_pairs index out of range for {self.num} keypoints")
        return self


class ModelConfig(_BaseModel):
    """Which network: a tier of the family, its heads, and its classes.

    `num_classes` None means "take it from the training annotations".
    `qat` None means the tier's default: quantization-aware for the small
    tiers (`mayaku.model.tiers.QAT_TIERS`, whose targets are int8-only
    accelerators), fp32 training for fp16 deployment otherwise.
    """

    tier: TierName = "n"
    num_classes: Annotated[int, Field(gt=0)] | None = None
    seg: bool = False
    keypoints: KeypointConfig | None = None
    aux_arm: Literal["box_tower", "tower"] = "box_tower"
    qat: bool | None = None

    @property
    def qat_enabled(self) -> bool:
        return self.qat if self.qat is not None else self.tier in QAT_TIERS

    def architecture(self) -> ModelConfig:
        """This network without what its training data decided -- the class
        count and keypoint names -- which a warm start derives afresh."""
        kp = self.keypoints
        return self.model_copy(update={"num_classes": None,
                                       "keypoints": KeypointConfig(num=kp.num) if kp else None})

    def to_tier(self) -> Tier:
        """The `mayaku.model.tiers.Tier` this config builds."""
        return dataclasses.replace(TIERS[self.tier], seg=self.seg,
                                   kpt=self.keypoints.num if self.keypoints else 0,
                                   aux_arm=self.aux_arm)

    def build(self, canvas: int | tuple[int, int], num_classes: int | None = None) -> Detector:
        """The detector this config describes, quantization-aware when
        `qat_enabled`: QAT changes the graph (and the checkpoint's keys), so
        a model is built with it before training or loading weights."""
        nc = num_classes if num_classes is not None else self.num_classes
        if nc is None:
            raise ValueError("model.num_classes is unset; pass num_classes")
        model = Detector(self.to_tier(), nc, canvas)
        return enable_qat(model) if self.qat_enabled else model


class InputConfig(_BaseModel):
    """The canvas every image is letterboxed onto, for training, evaluation,
    export and deployment alike.

    `size_budget` is the compute dial: a square-equivalent side, so the
    budget is `size_budget ** 2` pixels. `canvas_hw` is the resolved (H, W):
    set from the training data's aspect at train start
    (`mayaku.data.canvas.canvas_for_data`), or pinned by hand. Both are
    multiples of 32, the detector's coarsest stride.
    """

    size_budget: Annotated[int, Field(gt=0)] = 800
    canvas_hw: tuple[int, int] | None = None

    @field_validator("size_budget", "canvas_hw")
    @classmethod
    def _on_the_grid(cls, v: int | tuple[int, int] | None) -> int | tuple[int, int] | None:
        if v is not None:
            as_canvas(v)
        return v


class DataLoaderConfig(_BaseModel):
    num_workers: Annotated[int, Field(ge=0)] = 8


class AutoConfig(_BaseModel):
    """Dataset-aware tuning at train start: fields the user did not set
    explicitly (the class count, the canvas, the fine-tune schedule) are
    derived from the training annotations. Explicit values always win."""

    enabled: bool = True


class MayakuConfig(_BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    train: Recipe = Field(default_factory=Recipe)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    auto_config: AutoConfig = Field(default_factory=AutoConfig)
