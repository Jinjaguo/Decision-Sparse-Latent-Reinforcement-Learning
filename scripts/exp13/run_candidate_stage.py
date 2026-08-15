"""Execute a frozen EXP13 candidate plan through the validated EXP11 engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.exp11.run_replacement_stage as engine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("calibration", "formal"), required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--plan-run", required=True, type=Path)
    args = parser.parse_args()
    plan_root = ROOT / args.plan_run
    branches = json.loads((plan_root / "artifacts/branch_manifest.json").read_text())
    plan = json.loads((plan_root / "artifacts/candidate_plan.json").read_text())

    def frozen_build_plan(reference_run, support, stage, training_run, phase_path, contact_schema):
        if stage != args.stage:
            raise RuntimeError("stage differs from frozen plan")
        return branches, plan

    engine.build_plan = frozen_build_plan
    sys.argv = [
        "run_candidate_stage.py", "--run-id", args.run_id, "--stage", args.stage,
        "--reference-run", args.reference_run, "--authorized-families", "I-A_analytic",
    ]
    return engine.main()


if __name__ == "__main__":
    raise SystemExit(main())
