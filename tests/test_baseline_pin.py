"""Tests for frozen baseline v1.1 integrity and tamper blocking (B01/B02)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ipcr.baseline import project_root, verify_frozen_baseline


def test_frozen_baseline_matches_pinned_sha256() -> None:
    res = verify_frozen_baseline()
    assert res.passed is True
    assert res.version == "v1.1"
    assert res.actual_sha256 == "b3e0f04ad3f3c5c88d828c98dbc42e8fd936170df9e9fcb7a175dd7551c4c020"


def test_tampered_baseline_raises_without_rewriting_pin(tmp_path: Path) -> None:
    root = project_root()
    (tmp_path / "configs").mkdir()
    (tmp_path / "docs").mkdir()
    shutil.copy2(root / "configs" / "frozen_baseline.json", tmp_path / "configs" / "frozen_baseline.json")
    doc = tmp_path / "docs" / "最终底层基线要求_v1.1.md"
    doc.write_text("tampered content", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_frozen_baseline(tmp_path)
