#!/usr/bin/env python
"""Generate EXP8 references using the audited corrected-D implementation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.exp4.generate_heldout_references import main


if __name__ == "__main__":
    raise SystemExit(main())

