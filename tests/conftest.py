"""Pytest configuration and shared fixtures for the Mayaku test suite.

Backend selection is driven by the ``MAYAKU_DEVICE`` environment variable
(``cpu``, ``mps``, or ``cuda``; default ``cpu``). The chosen backend must
actually be available on the host — silent fall-through to CPU has caused
false-green test runs in past projects, so unavailable accelerators raise
``pytest.exit`` with a clear message rather than skipping.

Tests build their data with `tests.unit._coco_fixture` (a synthetic COCO
split written in well under a second) and their models on the TINY tier
(`mayaku.model.TINY`), so the default run needs no downloads and no GPU.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterable
from pathlib import Path

import pytest
import torch

_VALID_DEVICES = ("cpu", "mps", "cuda")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EXPECTED_MAYAKU_INIT = _PROJECT_ROOT / "src" / "mayaku" / "__init__.py"


def _selected_device_kind() -> str:
    raw = os.environ.get("MAYAKU_DEVICE", "cpu").strip().lower()
    if raw not in _VALID_DEVICES:
        pytest.exit(
            f"MAYAKU_DEVICE={raw!r} is not one of {_VALID_DEVICES}. "
            "Set it to 'cpu', 'mps', or 'cuda' before invoking pytest.",
            returncode=2,
        )
    return raw


def _resolve_device(kind: str) -> torch.device:
    if kind == "cuda":
        if not torch.cuda.is_available():
            pytest.exit(
                "MAYAKU_DEVICE=cuda but torch.cuda.is_available() is False. "
                "Run on a CUDA host or set MAYAKU_DEVICE=cpu/mps. Refusing "
                "to silently fall back to CPU.",
                returncode=2,
            )
        return torch.device("cuda:0")
    if kind == "mps":
        if not torch.backends.mps.is_available():
            pytest.exit(
                "MAYAKU_DEVICE=mps but torch.backends.mps.is_available() is False. "
                "Run on an Apple-Silicon host with a recent PyTorch build, or "
                "set MAYAKU_DEVICE=cpu. Refusing to silently fall back to CPU.",
                returncode=2,
            )
        return torch.device("mps")
    return torch.device("cpu")


def _verify_editable_install() -> None:
    """Fail loudly if ``import mayaku`` resolves to a stale editable install.

    Hatchling's editable install writes a ``.pth`` that adds a fixed
    ``src/`` to ``sys.path``. If the project was installed from a
    different clone path (or the directory was renamed after install),
    pytest's import machinery loads a phantom package and every test
    module collection fails with ``ModuleNotFoundError`` on a
    submodule that exists on disk but not on the pinned path.

    Recovery is one command (``pip install -e '.[dev]'``); this guard
    surfaces it as a single banner-line error instead of nine
    collection tracebacks. We've hit this twice — once on macOS, once
    on Linux — so the diagnostic earns its keep.
    """
    try:
        mayaku = importlib.import_module("mayaku")
    except ImportError as exc:
        pytest.exit(
            f"`import mayaku` failed: {exc}. Run `pip install -e '.[dev]'` from {_PROJECT_ROOT}.",
            returncode=2,
        )
    actual_file = getattr(mayaku, "__file__", None)
    if actual_file is None:
        pytest.exit(
            "mayaku has no __file__ — likely resolved as a namespace package "
            "to an empty directory. Run "
            f"`pip install -e '.[dev]'` from {_PROJECT_ROOT}.",
            returncode=2,
        )
    actual = Path(actual_file).resolve()
    if actual != _EXPECTED_MAYAKU_INIT.resolve():
        pytest.exit(
            f"mayaku resolves to {actual} but tests live under "
            f"{_EXPECTED_MAYAKU_INIT}. Run `pip install -e '.[dev]'` "
            f"from {_PROJECT_ROOT} to re-pin the editable install.",
            returncode=2,
        )


def _cuda_device_count() -> int:
    """Best-effort CUDA device count; 0 when CUDA isn't available."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _markers_that_will_skip(active_kind: str) -> list[str]:
    """Return the set of registered backend markers that will be skipped."""
    skipping: list[str] = []
    if active_kind != "cuda":
        skipping.append("cuda")
    if active_kind != "mps":
        skipping.append("mps")
    if _cuda_device_count() < 2:
        skipping.append("multi_gpu")
    return skipping


def pytest_report_header(config: pytest.Config) -> str:
    """One-line session banner: which backend is active and what will skip.

    Also runs the editable-install sanity check (see
    :func:`_verify_editable_install`) — this is the first hook that
    runs before collection, so a misconfigured install fails here
    instead of as a wall of ``ModuleNotFoundError``s.
    """
    _verify_editable_install()
    kind = _selected_device_kind()
    skipping = _markers_that_will_skip(kind)
    skip_summary = ", ".join(skipping) if skipping else "none"
    return f"mayaku: MAYAKU_DEVICE={kind} | markers that will skip: {skip_summary}"


def pytest_collection_modifyitems(config: pytest.Config, items: Iterable[pytest.Item]) -> None:
    """Auto-skip tests whose backend marker doesn't match the active backend."""
    kind = _selected_device_kind()
    cuda_count = _cuda_device_count()

    for item in items:
        if "cuda" in item.keywords and kind != "cuda":
            item.add_marker(
                pytest.mark.skip(reason=f"requires MAYAKU_DEVICE=cuda (active: {kind})")
            )
        if "mps" in item.keywords and kind != "mps":
            item.add_marker(pytest.mark.skip(reason=f"requires MAYAKU_DEVICE=mps (active: {kind})"))
        if "multi_gpu" in item.keywords and cuda_count < 2:
            item.add_marker(
                pytest.mark.skip(reason=f"requires >= 2 CUDA devices (have {cuda_count})")
            )


@pytest.fixture(scope="session")
def device() -> torch.device:
    """The active torch.device for this session, per MAYAKU_DEVICE."""
    return _resolve_device(_selected_device_kind())
