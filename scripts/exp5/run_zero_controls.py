#!/usr/bin/env python
"""Run EXP5 matched-zero controls through the shared validated continuation engine."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.exp4.run_criticality import main  # noqa: E402

if __name__ == "__main__":
    defaults = ["--mode", "zero", "--manifest-dir", str(ROOT / "experiments/exp5_state_conditioned_anisotropic/manifests"), "--config", str(ROOT / "experiments/exp5_state_conditioned_anisotropic/configs/exp5.json")]
    sys.argv.extend(defaults)
    raise SystemExit(main())
