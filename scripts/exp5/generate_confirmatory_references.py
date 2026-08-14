#!/usr/bin/env python
"""Generate EXP5 demos 10-19 using the audited EXP4 reference implementation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.exp4.generate_heldout_references import main  # noqa: E402


if __name__ == "__main__":
    if "--episodes" not in sys.argv:
        sys.argv.extend(["--episodes", *map(str, range(10, 20))])
    raise SystemExit(main())
