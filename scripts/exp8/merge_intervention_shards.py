#!/usr/bin/env python
"""Reuse the exact-coverage EXP7 raw merge and hash lock for EXP8."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.exp7.merge_intervention_shards import main


if __name__ == "__main__":
    raise SystemExit(main())
