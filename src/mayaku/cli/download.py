"""``mayaku download`` — fetch hosted model artifacts on demand.

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
    VARIANTS,
    download_model,
    list_models,
)

__all__ = ["run_download"]


def run_download(
    name: str | None = None,
    *,
    target: str | None = None,
    cache_dir: Path | None = None,
    manifest_url: str = DEFAULT_MANIFEST_URL,
    do_list: bool = False,
    do_all: bool = False,
    verify_sha256: bool = True,
) -> dict[str, Any]:
    """Drive the download CLI. Returns a JSON-friendly summary.

    Modes:

    * ``do_list=True`` — print the model index and return it.
    * ``do_all=True`` — fetch every variant of every model. Mostly
      useful for setting up a fresh deployment box.
    * ``name`` set, ``target`` set — fetch that single variant.
    * ``name`` set, ``target`` None — fetch all variants for that name.
    """
    if do_list:
        index = list_models(manifest_url=manifest_url)
        return {"models": index}

    if do_all:
        index = list_models(manifest_url=manifest_url)
        results: dict[str, dict[str, str]] = {}
        for _task, names in index.items():
            for n in names:
                results.setdefault(n, {})
                for v in VARIANTS:
                    try:
                        path = download_model(
                            n,
                            target=v,
                            cache_dir=cache_dir,
                            manifest_url=manifest_url,
                            verify_sha256=verify_sha256,
                        )
                        results[n][v] = str(path)
                    except Exception as e:
                        results[n][v] = f"ERROR: {e}"
        return {"downloaded": results}

    if name is None:
        raise ValueError("pass --list, --all, or a model name")

    if target is None:
        # All variants for this model.
        results_single: dict[str, str] = {}
        for v in VARIANTS:
            try:
                path = download_model(
                    name,
                    target=v,
                    cache_dir=cache_dir,
                    manifest_url=manifest_url,
                    verify_sha256=verify_sha256,
                )
                results_single[v] = str(path)
            except Exception as e:
                results_single[v] = f"ERROR: {e}"
        return {"name": name, "downloaded": results_single}

    # Single variant.
    path = download_model(
        name,
        target=target,
        cache_dir=cache_dir,
        manifest_url=manifest_url,
        verify_sha256=verify_sha256,
    )
    return {"name": name, "target": target, "path": str(path)}


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
