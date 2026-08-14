#!/usr/bin/env python
"""Run EXP8 through the audited EXP7 geometry-aware corrected-D executor."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.exp3 import run_criticality as exp3
from scripts.exp4 import run_criticality as exp4
from scripts.exp7 import run_interventions as exp7


def main() -> int:
    identity = json.loads((ROOT / "experiments/exp8_continuous_contact_frame/manifests/contact_identity_schema.json").read_text(encoding="utf-8"))
    exp7.SCHEMA = identity["runtime_schema"]
    exp3.observe = exp7.observe
    exp3.rollout.__globals__["observe"] = exp7.observe
    exp3.zero_step_record = exp7.zero_step_record
    exp3.reference_from_row = exp7.reference_from_row
    exp3.effect_row = exp7.effect_row
    return exp4.main()


if __name__ == "__main__":
    raise SystemExit(main())

