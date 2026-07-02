"""``mayaku download`` — fetch hosted model checkpoints on demand.

Thin wrapper around :func:`mayaku.utils.download.download_model`.
``run_download`` is the in-process entry point so tests / notebooks can
call it directly without going through Typer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mayaku.utils.download import (
    DEFAULT_MANIFEST_URL,
    download_model,
    list_models,
)

__all__ = ["run_download"]


def run_download(
    name: str | None = None,
    *,
    cache_dir: Path | None = None,
    manifest_url: str = DEFAULT_MANIFEST_URL,
    do_list: bool = False,
    do_all: bool = False,
    verify_sha256: bool = True,
) -> dict[str, Any]:
    """Drive the download CLI. Returns a JSON-friendly summary.

    Modes:

    * ``do_list=True`` — print the model index and return it.
    * ``do_all=True`` — fetch every model's checkpoint. Useful for setting up a
      fresh deployment box.
    * ``name`` set — fetch that model's checkpoint.
    """
    if do_list:
        index = list_models(manifest_url=manifest_url)
        return {"models": index}

    if do_all:
        index = list_models(manifest_url=manifest_url)
        results: dict[str, str] = {}
        for _task, names in index.items():
            for n in names:
                try:
                    path = download_model(
                        n,
                        cache_dir=cache_dir,
                        manifest_url=manifest_url,
                        verify_sha256=verify_sha256,
                    )
                    results[n] = str(path)
                except Exception as e:
                    results[n] = f"ERROR: {e}"
        return {"downloaded": results}

    if name is None:
        raise ValueError("pass --list, --all, or a model name")

    path = download_model(
        name,
        cache_dir=cache_dir,
        manifest_url=manifest_url,
        verify_sha256=verify_sha256,
    )
    return {"name": name, "path": str(path)}


def render_index(index: dict[str, list[str]]) -> str:
    lines = []
    for task in sorted(index):
        lines.append(f"\n{task}:")
        for name in index[task]:
            lines.append(f"  {name}")
    return "\n".join(lines).lstrip()


def main_print(payload: dict[str, Any]) -> None:  # for the __main__ wrapper
    if "models" in payload:
        print(render_index(payload["models"]))
    else:
        print(json.dumps(payload, indent=2, default=str))
