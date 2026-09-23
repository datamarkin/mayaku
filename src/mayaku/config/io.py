"""YAML load / dump + override helpers for :class:`MayakuConfig`.

This module is intentionally thin: pydantic v2 already does almost all
the work via ``model_validate`` and ``model_dump``. The wrappers here
just bridge YAML on disk and an in-memory pydantic model, plus a
recursive deep-merge for partial overrides (CLI flags, fragment
configs).

There is deliberately no YAML inheritance: an inheritance chain causes more
confusion than it removes. Compose fragments by constructing a
:class:`MayakuConfig` in Python and calling :func:`merge_overrides` once.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from mayaku.config.schemas import MayakuConfig

__all__ = [
    "dump_yaml",
    "load_yaml",
    "merge_overrides",
    "to_yaml_string",
]


def load_yaml(path: str | Path) -> MayakuConfig:
    """Parse a YAML file into a validated :class:`MayakuConfig`.

    Empty / missing top-level sections are filled with each section's
    own defaults (e.g. an `input` section that's omitted yields
    :class:`InputConfig` defaults). Unknown keys at any level raise
    ``ValidationError`` so typos fail loudly.
    """
    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"YAML at {path} must be a mapping at the top level; got {type(raw).__name__}"
        )
    return MayakuConfig.model_validate(dict(raw))


def dump_yaml(config: MayakuConfig, path: str | Path) -> None:
    """Write ``config`` to ``path`` as a deterministic YAML document.

    Round-trips: ``load_yaml(dump_yaml(c, p))`` equals ``c``.
    """
    Path(path).write_text(to_yaml_string(config), encoding="utf-8")


def to_yaml_string(config: MayakuConfig) -> str:
    """Return ``config`` as a YAML document string.

    ``sort_keys=False`` preserves the schema's field declaration order so
    diffs over time stay meaningful. ``default_flow_style=False`` keeps
    nested maps in block form so the output is human-readable.
    """
    payload = config.model_dump(mode="python")
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)


def merge_overrides(config: MayakuConfig, overrides: Mapping[str, Any]) -> MayakuConfig:
    """Deep-merge ``overrides`` into ``config`` and re-validate.

    ``overrides`` is a (possibly nested) mapping keyed by the schema's
    field names. Nested mappings recurse into nested sub-configs; a
    leaf value replaces the corresponding field. Returns a *new*
    :class:`MayakuConfig`; ``config`` is unchanged.

    Nested mappings reach into the training recipe too:
    ``{"train": {"aug": {"mixup": 0.1}}}`` changes that one field.
    """
    merged = _deep_merge(config.model_dump(mode="python"), overrides)
    return MayakuConfig.model_validate(merged)


def _deep_merge(base: Any, override: Any) -> Any:
    """Recursive dict merge; non-mapping values in ``override`` win."""
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        merged = dict(base)
        for k, v in override.items():
            merged[k] = _deep_merge(base.get(k), v) if k in merged else v
        return merged
    return override
