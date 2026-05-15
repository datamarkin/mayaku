"""Typed configuration layer (pydantic v2) for Mayaku.

Replaces Detectron2's ``CfgNode`` / ``LazyConfig`` machinery with
explicit dataclass-style schemas. See :mod:`mayaku.config.schemas` for
the model definitions, :mod:`mayaku.config.io` for YAML round-trip and
override merging, and :mod:`mayaku.config.schedules` for the
``schedule_{1x,2x,3x}`` factories.
"""

from __future__ import annotations

from mayaku.config.io import dump_yaml, load_yaml, merge_overrides, to_yaml_string
from mayaku.config.schedules import schedule_1x, schedule_2x, schedule_3x
from mayaku.config.schemas import (
    AnchorGeneratorConfig,
    AutoConfig,
    BackboneConfig,
    BackboneName,
    DataLoaderConfig,
    DeviceSetting,
    FPNConfig,
    InputConfig,
    MayakuConfig,
    MetaArchitecture,
    ModelConfig,
    ROIBoxHeadConfig,
    ROIHeadsConfig,
    ROIKeypointHeadConfig,
    ROIMaskHeadConfig,
    RPNConfig,
    SolverConfig,
    TestConfig,
)

__all__ = [
    "AnchorGeneratorConfig",
    "AutoConfig",
    "BackboneConfig",
    "BackboneName",
    "DataLoaderConfig",
    "DeviceSetting",
    "FPNConfig",
    "InputConfig",
    "MayakuConfig",
    "MetaArchitecture",
    "ModelConfig",
    "ROIBoxHeadConfig",
    "ROIHeadsConfig",
    "ROIKeypointHeadConfig",
    "ROIMaskHeadConfig",
    "RPNConfig",
    "SolverConfig",
    "TestConfig",
    "dump_yaml",
    "load_yaml",
    "merge_overrides",
    "schedule_1x",
    "schedule_2x",
    "schedule_3x",
    "to_yaml_string",
]
