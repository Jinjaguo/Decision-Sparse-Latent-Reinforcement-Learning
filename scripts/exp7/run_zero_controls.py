#!/usr/bin/env python
"""Documented EXP7 matched-zero entry point; arguments match run_interventions.py."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.exp7.run_interventions import main

if __name__ == "__main__":
    raise SystemExit(main())
