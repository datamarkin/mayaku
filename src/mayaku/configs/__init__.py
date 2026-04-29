"""Bundled model configurations — `pip install mayaku` ships them with the wheel.

The 12 zoo configs are kept inside the package so installed users don't
need to clone the repo just to load a model. Every config name is unique
across the three tasks (``faster_*`` is detection, ``mask_*`` is
segmentation, ``keypoint_*`` is keypoints), so the short name alone is
enough — the resolver infers the task directory.

Three equivalent ways to get a config path:

    >>> from mayaku import configs
    >>> configs.path("faster_rcnn_R_50_FPN_3x")
    PosixPath('.../mayaku/configs/detection/faster_rcnn_R_50_FPN_3x.yaml')
    >>> configs.faster_rcnn_R_50_FPN_3x          # attribute form
    PosixPath('.../mayaku/configs/detection/faster_rcnn_R_50_FPN_3x.yaml')
    >>> configs.path("detection/faster_rcnn_R_50_FPN_3x")  # qualified form
    PosixPath('.../mayaku/configs/detection/faster_rcnn_R_50_FPN_3x.yaml')

Or skip the path step and load directly:

    >>> cfg = configs.load("faster_rcnn_R_50_FPN_3x")  # → MayakuConfig

List everything that ships:

    >>> configs.list_all()
    ['detection/faster_rcnn_R_101_FPN_3x', 'detection/faster_rcnn_R_50_FPN_1x', ...]
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mayaku.config import MayakuConfig

__all__ = ["list_all", "load", "path"]

_TASK_DIRS = ("detection", "segmentation", "keypoints")


def _root() -> Path:
    """Return the on-disk directory holding the bundled YAMLs.

    Two cases:
    * Wheel install — the YAMLs live next to this ``__init__.py`` (the
      ``[tool.hatch.build.targets.wheel.force-include]`` directive in
      ``pyproject.toml`` puts the top-level ``configs/`` tree there).
    * Source / editable install — the YAMLs are at the repo's top-level
      ``configs/`` directory and are *not* mirrored into ``src/mayaku/``,
      so we walk up to the repo root and use that path instead.
    """
    here = Path(str(files(__package__)))
    if any((here / task).is_dir() and any((here / task).glob("*.yaml")) for task in _TASK_DIRS):
        return here
    # Editable / source-tree fallback: <repo>/src/mayaku/configs → <repo>/configs
    repo_configs = here.parent.parent.parent / "configs"
    if (repo_configs / _TASK_DIRS[0]).is_dir():
        return repo_configs
    return here  # final fallback — `path()` will raise with a clear error


def path(name: str) -> Path:
    """Resolve a bundled config name to its YAML file path.

    Accepts any of:
        - ``"faster_rcnn_R_50_FPN_3x"``           — short name (auto-detects task)
        - ``"detection/faster_rcnn_R_50_FPN_3x"`` — qualified
        - either form with or without a ``.yaml`` suffix

    Raises ``FileNotFoundError`` if the name doesn't match any bundled
    config, or ``ValueError`` if a short name is ambiguous (which
    should never happen with the current zoo, but the check guards
    against future name collisions).
    """
    name = name.removesuffix(".yaml")
    root = _root()

    if "/" in name:
        candidate = root / f"{name}.yaml"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"Bundled config not found: {name}.yaml. "
            "See `mayaku.configs.list_all()` for available names."
        )

    matches = [
        root / task / f"{name}.yaml"
        for task in _TASK_DIRS
        if (root / task / f"{name}.yaml").is_file()
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No bundled config named '{name}'. Available: {list_all()}")
    raise ValueError(
        f"Ambiguous config name '{name}'; matches: "
        f"{[str(m.relative_to(root)) for m in matches]}. "
        "Use the qualified form 'task/name'."
    )


def list_all() -> list[str]:
    """Return every bundled config as a ``'task/name'`` string, sorted."""
    root = _root()
    out: list[str] = []
    for task in _TASK_DIRS:
        task_dir = root / task
        if task_dir.is_dir():
            for f in sorted(task_dir.iterdir()):
                if f.suffix == ".yaml":
                    out.append(f"{task}/{f.stem}")
    return out


def load(name: str) -> MayakuConfig:
    """Resolve ``name`` and load it as a :class:`MayakuConfig`."""
    from mayaku.config import load_yaml

    return load_yaml(path(name))


def __getattr__(name: str) -> Path:
    """Attribute-style access: ``configs.faster_rcnn_R_50_FPN_3x`` → Path."""
    try:
        return path(name)
    except FileNotFoundError as e:
        raise AttributeError(str(e)) from e


def __dir__() -> list[str]:
    """Expose bundled config short-names for IDE tab-completion."""
    return [*__all__, *(qualified.split("/", 1)[1] for qualified in list_all())]
