"""Pytest configuration and shared fixtures for the Mayaku test suite.

Backend selection is driven by the ``MAYAKU_DEVICE`` environment variable
(``cpu``, ``mps``, or ``cuda``; default ``cpu``). The chosen backend must
actually be available on the host — silent fall-through to CPU has caused
false-green test runs in past projects, so unavailable accelerators raise
``pytest.exit`` with a clear message rather than skipping.

Run the suite once on each physical machine (Linux CPU box, Apple Silicon
Mac, CUDA Linux box) with the matching ``MAYAKU_DEVICE`` to mark a step
done — see ``PROJECT_STATUS.md`` for the per-step checklist.
"""

from __future__ import annotations

import importlib
import importlib.util
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


def _is_importable(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


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
    if not _is_importable("coremltools"):
        skipping.append("coreml")
    if not _is_importable("openvino"):
        skipping.append("openvino")
    # TensorRT requires a CUDA host *and* the tensorrt python runtime.
    # Skip on either gap so macOS / CPU-only Linux hosts get a clean
    # session message instead of an import-time crash.
    if active_kind != "cuda" or not _is_importable("tensorrt"):
        skipping.append("tensorrt")
    return skipping


def pytest_configure(config: pytest.Config) -> None:
    """Register backend markers (also declared in pyproject for redundancy)."""
    for name, doc in (
        ("cuda", "requires a CUDA device"),
        ("mps", "requires an Apple-Silicon MPS device"),
        ("multi_gpu", "requires >= 2 CUDA devices"),
        ("coreml", "requires coremltools (macOS)"),
        ("openvino", "requires openvino runtime"),
        ("tensorrt", "requires a CUDA device + tensorrt runtime"),
    ):
        config.addinivalue_line("markers", f"{name}: {doc}")


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
    coreml_available = _is_importable("coremltools")
    openvino_available = _is_importable("openvino")
    cuda_count = _cuda_device_count()
    tensorrt_available = _is_importable("tensorrt")

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
        if "coreml" in item.keywords and not coreml_available:
            item.add_marker(pytest.mark.skip(reason="coremltools not installed"))
        if "tensorrt" in item.keywords and (kind != "cuda" or not tensorrt_available):
            reason = (
                f"requires CUDA + tensorrt (active: {kind}, "
                f"tensorrt_available={tensorrt_available})"
            )
            item.add_marker(pytest.mark.skip(reason=reason))
        if "openvino" in item.keywords and not openvino_available:
            item.add_marker(pytest.mark.skip(reason="openvino not installed"))


@pytest.fixture(scope="session")
def device() -> torch.device:
    """The active torch.device for this session, per MAYAKU_DEVICE."""
    return _resolve_device(_selected_device_kind())
