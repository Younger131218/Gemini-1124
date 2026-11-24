"""Frozen baseline v1.1 hash verification (B01/B02)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BaselineVerification:
    version: str
    path: Path
    expected_sha256: str
    actual_sha256: str
    passed: bool


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def verify_frozen_baseline(root: Path | None = None) -> BaselineVerification:
    base = Path(root) if root is not None else project_root()
    pin_path = base / "configs" / "frozen_baseline.json"
    if not pin_path.exists():
        raise FileNotFoundError(f"Missing frozen baseline pin: {pin_path}")
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    doc_path = base / pin["path"]
    if not doc_path.exists():
        raise FileNotFoundError(f"Missing frozen baseline document: {doc_path}")
    expected = str(pin["sha256"]).strip().lower()
    actual = sha256_file(doc_path)
    passed = actual == expected
    result = BaselineVerification(
        version=str(pin["version"]),
        path=doc_path,
        expected_sha256=expected,
        actual_sha256=actual,
        passed=passed,
    )
    if not passed:
        raise ValueError(
            f"Frozen baseline SHA-256 mismatch for {doc_path}: "
            f"expected {expected}, got {actual}"
        )
    return result
