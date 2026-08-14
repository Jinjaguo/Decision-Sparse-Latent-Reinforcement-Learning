#!/usr/bin/env python
"""Generate requested EXP7 references with the audited corrected-D EXP4 implementation."""

import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.exp4.generate_heldout_references import main

if __name__=="__main__": raise SystemExit(main())
