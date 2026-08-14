#!/usr/bin/env python
"""Run EXP7 through the audited EXP4 simulator while adding exact geometry state."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from scripts.exp3 import run_criticality as exp3  # noqa: E402
from scripts.exp4 import run_criticality as exp4  # noqa: E402
from scripts.exp7.contact_geometry import load_schema, measure  # noqa: E402
from decision_sparse_rl.envs.mujoco_snapshot import native_model_data  # noqa: E402


SCHEMA = None
ORIGINAL_OBSERVE = exp3.observe
ORIGINAL_ZERO = exp3.zero_step_record
ORIGINAL_REFERENCE = exp3.reference_from_row
ORIGINAL_EFFECT = exp3.effect_row


def observe(env, body_ids):
    row = ORIGINAL_OBSERVE(env, body_ids)
    model, _ = native_model_data(env.sim)
    body_names = tuple(str(model.body(i).name) for i in body_ids)
    matches = [task for task, spec in SCHEMA["tasks"].items() if tuple(spec["task_object_body_names"]) == body_names]
    if len(matches) != 1:
        raise RuntimeError(f"task-object body signature is not unique: {body_names} -> {matches}")
    row.update(measure(env, SCHEMA, matches[0]))
    return row


def zero_step_record(base, offset, observation):
    row = ORIGINAL_ZERO(base, offset, observation)
    for key in ("contact_mode_json", "signed_gap_m", "signed_gap_group", "normal_relative_velocity_mps", "group_signed_gaps_json", "group_normal_velocities_json", "pair_geometry_json"):
        row[key] = observation[key]
    return row


def reference_from_row(row):
    result = ORIGINAL_REFERENCE(row)
    result.update({
        "contact_mode": tuple(json.loads(row["contact_mode_json"])),
        "contact_mode_json": row["contact_mode_json"],
        "signed_gap_m": float(row["signed_gap_m"]),
        "signed_gap_group": row["signed_gap_group"],
        "normal_relative_velocity_mps": float(row["normal_relative_velocity_mps"]),
        "group_signed_gaps_json": row["group_signed_gaps_json"],
        "group_normal_velocities_json": row["group_normal_velocities_json"],
        "pair_geometry_json": row["pair_geometry_json"],
    })
    return result


def effect_row(base, offset, zero, perturbed, normalization):
    row = ORIGINAL_EFFECT(base, offset, zero, perturbed, normalization)
    row.update({
        "zero_contact_mode_json": zero["contact_mode_json"],
        "perturbed_contact_mode_json": perturbed["contact_mode_json"],
        "zero_signed_gap_m": zero["signed_gap_m"],
        "perturbed_signed_gap_m": perturbed["signed_gap_m"],
        "zero_normal_relative_velocity_mps": zero["normal_relative_velocity_mps"],
        "perturbed_normal_relative_velocity_mps": perturbed["normal_relative_velocity_mps"],
        "zero_group_signed_gaps_json": zero["group_signed_gaps_json"],
        "perturbed_group_signed_gaps_json": perturbed["group_signed_gaps_json"],
    })
    return row


def main() -> int:
    global SCHEMA
    manifest = ROOT / "experiments/exp7_contact_mode_response/manifests/contact_mode_schema.json"
    SCHEMA = load_schema(manifest)
    exp3.observe = observe
    exp3.rollout.__globals__["observe"] = observe
    exp3.zero_step_record = zero_step_record
    exp3.reference_from_row = reference_from_row
    exp3.effect_row = effect_row
    return exp4.main()


if __name__ == "__main__":
    raise SystemExit(main())
