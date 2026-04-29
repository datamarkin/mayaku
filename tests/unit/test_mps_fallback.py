"""Tests for :mod:`mayaku.backends.mps`.

These tests do not require an actual MPS device — they exercise the
warning interceptor and environment helper directly. The module is
mostly Python plumbing around ``warnings.showwarning``; treating it
as device-independent keeps coverage on every CI host.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

from mayaku.backends import mps as mps_module
from mayaku.backends.mps import OpFallbackTracker, apply_mps_environment


@pytest.fixture(autouse=True)
def _isolate_module_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset module-level state and env vars between tests.

    ``apply_mps_environment`` prints a one-shot orientation note guarded
    by a module flag; tests need to start from a clean slate so
    ``capsys`` captures the print on each invocation.
    """
    monkeypatch.setattr(mps_module, "_APPLIED_NOTE_PRINTED", False)
    monkeypatch.delenv("MAYAKU_VERBOSE_MPS", raising=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    yield


def _emit_fallback_warning(op: str) -> None:
    """Emit a warning matching PyTorch's MPS fallback message shape."""
    warnings.warn(
        f"The operator '{op}' is not currently supported on the MPS backend "
        "and will fall back to run on the CPU. This may have performance "
        "implications.",
        UserWarning,
        stacklevel=1,
    )


def test_package_init_file_sets_fallback_env_var() -> None:
    """The mayaku package __init__ must set PYTORCH_ENABLE_MPS_FALLBACK.

    Setting the env var lazily inside ``run_train`` is too late: by then
    ``import torch`` has already snapshotted the env. Setting it in
    ``mayaku/__init__.py`` ensures it's visible at the time PyTorch's
    MPS backend initialises — but only if mayaku is imported before
    torch in any entry-point script. We test the file content rather
    than the live env so this stays robust against other tests
    monkeypatching the var.
    """
    init_path = Path(__file__).resolve().parents[2] / "src" / "mayaku" / "__init__.py"
    text = init_path.read_text()
    assert "PYTORCH_ENABLE_MPS_FALLBACK" in text
    assert 'setdefault("PYTORCH_ENABLE_MPS_FALLBACK"' in text


def test_apply_sets_fallback_env_var(capsys: pytest.CaptureFixture[str]) -> None:
    apply_mps_environment()
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    out = capsys.readouterr().out
    assert "[mayaku/mps]" in out
    assert "CPU fallback enabled" in out


def test_apply_respects_user_set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-existing PYTORCH_ENABLE_MPS_FALLBACK value must not be overridden."""
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    apply_mps_environment()
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"


def test_apply_idempotent(capsys: pytest.CaptureFixture[str]) -> None:
    """Note prints once per process even with multiple apply calls."""
    apply_mps_environment()
    apply_mps_environment()
    out = capsys.readouterr().out
    # Exactly one "[mayaku/mps]" announcement line.
    assert out.count("[mayaku/mps] CPU fallback enabled") == 1


def test_tracker_dedups_same_op(capsys: pytest.CaptureFixture[str]) -> None:
    # Python's default warning filter deduplicates identical messages
    # before they reach showwarning; force "always" so each emit counts.
    # In real MPS runs PyTorch emits these from C++ with a fresh
    # location each time, so this filter override is just a test affordance.
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with OpFallbackTracker(label="train"):
            for _ in range(50):
                _emit_fallback_warning("aten::nonzero")
            for _ in range(3):
                _emit_fallback_warning("aten::sort")
    out = capsys.readouterr().out
    # Summary lines for each unique op, with their counts.
    assert "aten::nonzero" in out
    assert "50 calls" in out
    assert "aten::sort" in out
    assert "3 calls" in out
    # Top-line summary mentions both unique ops + the total.
    assert "53 CPU round-trips" in out
    assert "2 unique ops" in out


def test_tracker_passes_through_non_mps_warnings(
    capsys: pytest.CaptureFixture[str], recwarn: pytest.WarningsRecorder
) -> None:
    """Warnings unrelated to MPS fallback must reach the original handler."""
    with OpFallbackTracker():
        warnings.warn("unrelated deprecation", DeprecationWarning, stacklevel=1)
        _emit_fallback_warning("aten::nonzero")
    # The unrelated warning was forwarded; the MPS one was eaten.
    captured = [str(w.message) for w in recwarn]
    assert any("unrelated deprecation" in m for m in captured)
    assert not any("MPS backend" in m for m in captured)


def test_tracker_empty_run_emits_no_fallback_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with OpFallbackTracker(label="eval"):
        pass
    out = capsys.readouterr().out
    assert "no MPS->CPU fallbacks recorded" in out
    assert "(eval)" in out


def test_verbose_mode_disables_dedup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With MAYAKU_VERBOSE_MPS=1, the tracker is a passthrough.

    The native warning system continues to fire per-call; the tracker
    contributes no summary at end of context.
    """
    monkeypatch.setenv("MAYAKU_VERBOSE_MPS", "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with OpFallbackTracker():
            _emit_fallback_warning("aten::nonzero")
            _emit_fallback_warning("aten::nonzero")
    # Both warnings reached the recorder (no dedup).
    assert sum("MPS backend" in str(w.message) for w in caught) == 2
    out = capsys.readouterr().out
    assert "op-fallback summary" not in out
