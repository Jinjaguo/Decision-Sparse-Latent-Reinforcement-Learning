#!/usr/bin/env python
"""Run EXP5 dry or full multi-radius interventions through the shared engine."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.exp4.run_criticality import main  # noqa: E402

if __name__ == "__main__":
    if "--mode" not in sys.argv: sys.argv.extend(["--mode", "full"])
    sys.argv.extend(["--manifest-dir", str(ROOT / "experiments/exp5_state_conditioned_anisotropic/manifests"), "--config", str(ROOT / "experiments/exp5_state_conditioned_anisotropic/configs/exp5.json")])
    raise SystemExit(main())
