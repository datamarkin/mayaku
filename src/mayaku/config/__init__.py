"""Typed configuration (pydantic v2): `MayakuConfig` and YAML round-trip.

See :mod:`mayaku.config.schemas` for the models and :mod:`mayaku.config.io`
for YAML load / dump and override merging.
"""

from __future__ import annotations

from mayaku.config.io import (
    dump_yaml,
    load_yaml,
    merge_overrides,
    parse_assignments,
    read_yaml,
    to_yaml_string,
)
from mayaku.config.schemas import (
    AutoConfig,
    DataLoaderConfig,
    InputConfig,
    KeypointConfig,
    MayakuConfig,
    ModelConfig,
    TierName,
)

__all__ = [
    "AutoConfig",
    "DataLoaderConfig",
    "InputConfig",
    "KeypointConfig",
    "MayakuConfig",
    "ModelConfig",
    "TierName",
    "dump_yaml",
    "load_yaml",
    "merge_overrides",
    "parse_assignments",
    "read_yaml",
    "to_yaml_string",
]
