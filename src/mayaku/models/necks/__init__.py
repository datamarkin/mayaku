"""Necks (top-down feature aggregators). Currently: FPN."""

from __future__ import annotations

from mayaku.models.necks.fpn import FPN, LastLevelMaxPool, build_fpn

__all__ = ["FPN", "LastLevelMaxPool", "build_fpn"]
