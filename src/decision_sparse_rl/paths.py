"""Repository path helpers with no machine-specific assumptions."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root derived from this installed source file."""

    return Path(__file__).resolve().parents[2]
