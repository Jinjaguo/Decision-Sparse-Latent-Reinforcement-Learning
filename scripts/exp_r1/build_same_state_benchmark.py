"""Build the EXP_R1 same-state counterfactual benchmark from audited EXP27 runs.

This is an infrastructure/validity experiment.  It does not train a selector and
does not use post-action outcomes as selector inputs.  The source run already
contains one restored branch state evaluated by the same seven route candidates;
EXP_R1 turns those immutable artifacts into an explicit candidate-set benchmark,
records the exact EXP27 expert definitions, and audits the restoration substrate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def safe(row: dict) -> bool:
    return bool(row["success"] and not row["safety_stop"])


def outcome_key(row: dict) -> tuple[int, int, int]:
    """A conservative, pre-registered label ordering for pairwise ties.

    Safe success is primary.  Among unsafe outcomes, a non-stop failure is
    preferred to a safety stop only as a secondary audit distinction.  Runtime
    length is intentionally not used to break ties.
    """

    return (int(safe(row)), int(not row["safety_stop"]), int(row["success"]))


def source_expert_audit(protocol: dict) -> dict:
    """Exact EXP27 mechanism map extracted from run_recovery_stage.py."""

    return {
        "source": {
            "implementation": "scripts/exp15/run_recovery_stage.py",
            "route_protocol": "formal_run/manifests/recovery_protocol.json",
            "view_columns": {
                "physical": "np.r_[0:3,9:15]",
                "object": "np.r_[3:15]",
                "full": "np.r_[0:26]",
            },
            "feature_function": "feature(eef, pos, quat, contact)",
            "selection_function": "choose_chunk(spec, state, library, memory, exclude_episode)",
            "runtime_function": "run_recovery_stage.main",
        },
        "mechanisms": {
            "default": {
                "exact_mode": "C7_physical_default",
                "definition": {
                    "view": "physical", "k": 1, "replan": 10,
                    "monotone": False, "advance": 0, "retarget": 0.0,
                    "aggregate": "nearest", "smooth": 0.0,
                    "selection": "distance",
                },
                "meaning": "physical-view nearest retrieval with a persistent 10-step chunk",
            },
            "goal": {
                "exact_mode": "C0_goal_consequence",
                "definition": {
                    "view": "full", "k": 7, "replan": 1,
                    "monotone": False, "advance": 0, "retarget": 0.0,
                    "aggregate": "weighted", "smooth": 0.15,
                    "selection": "goal_effect", "search_k": 48,
                    "consequence_weight": 0.75,
                },
                "rule": "normalized distance to retrieved successor versus stored goal feature",
            },
            "progress": {
                "exact_mode": "C1_progress_consequence",
                "definition": {
                    "view": "full", "k": 5, "replan": 1,
                    "monotone": True, "advance": 2, "retarget": 0.0,
                    "aggregate": "weighted", "smooth": 0.10,
                    "selection": "progress", "search_k": 36,
                    "consequence_weight": 0.80,
                },
                "rule": "normalized retrieval distance minus retrieved progress weight",
            },
            "smooth": {
                "exact_mode": "C4_smooth_low",
                "definition": {
                    "view": "full", "k": 5, "replan": 1,
                    "monotone": False, "advance": 0, "retarget": 0.0,
                    "aggregate": "weighted", "smooth": 0.25,
                    "selection": "distance",
                },
                "rule": "distance retrieval, then blend action dimensions 0:6 with previous chunk at 0.25",
            },
            "response": {
                "exact_mode": "C2_response_alignment",
                "definition": {
                    "view": "full", "k": 5, "replan": 1,
                    "monotone": False, "advance": 0, "retarget": 0.0,
                    "aggregate": "medoid", "smooth": 0.0,
                    "selection": "response", "search_k": 48,
                    "consequence_weight": 0.65,
                },
                "rule": "normalized successor-minus-current effect aligned with current-to-goal direction",
            },
            "soft_force": {
                "exact_route": "V0_default70_soft_goal / turn_on_the_stove",
                "guard": {
                    "force_guard": "scale", "guard_fraction": 0.55,
                    "guard_scale": 0.25,
                    "primary_threshold_source": "EXP16 expert_force_envelope.json",
                },
                "fallback": [
                    "C7_physical_default",
                    "C4_smooth_low",
                    "C0_goal_consequence",
                    "C2_response_alignment",
                ],
                "switch_rule": "mode dwell/progress/force checks in run_recovery_stage.main",
                "action_rule": "when pre-action force exceeds 0.55 times task threshold, scale action[:6] by 0.25",
            },
        },
        "shared_execution_rules": {
            "action_clip": "np.clip(executed[:,:6], -1, 1)",
            "gripper_rule": "executed[:,6] = np.sign(executed[:,6])",
            "success_stop": "stop immediately when task predicate is true",
            "safety_stop": "consecutive threshold exceedance or absolute 1000 N limit",
            "target_demo_exclusion": "library episodes exclude target episode when protocol flag is true",
        },
    }


def audit_restoration(formal: Path, branches: list[dict]) -> dict:
    """Audit immutable corrected-D references without re-running the simulator."""

    missing = []
    invalid = []
    records = []
    for branch in branches:
        reference_run = ROOT / branch["reference_run"]
        directory = reference_run / branch["reference_directory"]
        t = int(branch["branch_time"])
        states = directory / "trajectory_states.npz"
        controller = directory / f"controller_{t:04d}.npz"
        if not states.exists() or not controller.exists():
            missing.append(branch["branch_id"])
            continue
        try:
            with np.load(states, allow_pickle=False) as archive:
                keys = sorted(archive.files)
                length = len(archive["actions"])
                integration_shape = list(archive["integration"].shape)
                finite = bool(np.all(np.isfinite(archive["integration"][t])))
            valid = t >= 0 and t + 1 < length and finite
        except Exception as exc:  # pragma: no cover - retained in audit artifact
            invalid.append({"branch_id": branch["branch_id"], "error": repr(exc)})
            continue
        if not valid:
            invalid.append({"branch_id": branch["branch_id"], "reason": "invalid branch boundary"})
        records.append({
            "branch_id": branch["branch_id"],
            "reference_run": branch["reference_run"],
            "reference_directory": branch["reference_directory"],
            "branch_time": t,
            "trajectory_length": length,
            "trajectory_state_keys": keys,
            "integration_shape": integration_shape,
            "controller_sha256": sha256(controller),
            "structurally_restorable": valid,
            "restore_artifact_hash_note": "structural only; simulator zero replay is not re-executed in EXP_R1",
        })
    return {
        "method": "corrected-D reference/controller existence, shape, boundary, and finite-state audit",
        "simulator_zero_replay_executed": False,
        "equivalent_structural_determinism_audit": True,
        "branch_count": len(branches),
        "audited_count": len(records),
        "missing_count": len(missing),
        "invalid_count": len(invalid),
        "missing_branch_ids": missing,
        "invalid": invalid,
        "records": records,
        "passed": len(records) == len(branches) and not missing and not invalid,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-run", type=Path, default=Path("runs/exp27_s3_formal_new_time_cascade_20260815"))
    parser.add_argument("--analysis-run", type=Path, default=Path("runs/exp27_s4_formal_analysis_20260815"))
    parser.add_argument("--audit-run", type=Path, default=Path("runs/exp27_s5_final_audit_20260815"))
    parser.add_argument("--config", type=Path, default=Path("experiments/exp27_success_preserving_cascade/configs/exp27.json"))
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    if out.exists():
        raise FileExistsError(f"immutable run exists: {out}")
    artifacts = out / "artifacts"
    manifests = out / "manifests"
    artifacts.mkdir(parents=True)
    manifests.mkdir(parents=True)

    candidate = ROOT / args.candidate_run
    analysis = ROOT / args.analysis_run
    audit = ROOT / args.audit_run
    config_path = ROOT / args.config
    summaries = pq.read_table(candidate / "artifacts/candidate_summaries.parquet").to_pylist()
    steps = pq.read_table(candidate / "artifacts/per_step.parquet").to_pylist()
    protocol = json.loads((candidate / "manifests/recovery_protocol.json").read_text(encoding="utf-8"))
    branches = json.loads((candidate / "manifests/branch_manifest.json").read_text(encoding="utf-8"))
    analysis_metrics = json.loads((analysis / "metrics.json").read_text(encoding="utf-8"))
    audit_metrics = json.loads((audit / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    route_specs = {x["route"]: x for x in protocol["routes"]}
    summary_by_key = {(x["branch_id"], x["route"]): x for x in summaries}
    step_groups = defaultdict(list)
    for row in steps:
        step_groups[(row["branch_id"], row["route"])].append(row)
    branch_ids = [x["branch_id"] for x in branches]
    route_names = sorted(route_specs)
    expected_keys = {(bid, route) for bid in branch_ids for route in route_names}
    observed_keys = set(summary_by_key)
    checks = {
        "summary_count": len(summaries) == len(expected_keys),
        "step_count_positive": len(steps) > 0,
        "unique_branch_ids": len(branch_ids) == len(set(branch_ids)) == 60,
        "complete_candidate_matrix": observed_keys == expected_keys,
        "all_routes_from_protocol": set(x["route"] for x in summaries) == set(route_names),
        "protocol_target_future_false": protocol["target_future_candidate_access"] is False,
        "protocol_expert_path_isolated": protocol["expert_path_isolated"] is True,
        "protocol_target_demo_excluded": protocol["exclude_target_demo_from_neighbors_and_scale"] is True,
        "source_audit_passed": audit_metrics["passed"] is True,
        "analysis_target_future_false": analysis_metrics["target_future_access"] is False,
        "all_summary_states_finite": all(bool(x["all_states_finite"]) for x in summaries),
        "formal_audit_passed": audit_metrics["passed"] is True,
        "config_frozen_primary": config["frozen_primary"] == "V0_default70_soft_goal",
    }
    if not all(checks.values()):
        raise RuntimeError(f"EXP_R1 validity checks failed: {checks}")

    rows = []
    groups = []
    for branch_id in branch_ids:
        candidates = [summary_by_key[(branch_id, route)] for route in route_names]
        default = summary_by_key[(branch_id, "D_physical_chunk")]
        keys = [outcome_key(x) for x in candidates]
        oracle = max(candidates, key=outcome_key)
        group_steps = [step_groups[(branch_id, route)] for route in route_names]
        for candidate_row, route_steps in zip(candidates, group_steps):
            first = sorted(route_steps, key=lambda x: int(x["offset"]))[0] if route_steps else None
            requested = np.asarray(first["requested_action"], dtype=float) if first is not None else None
            executed = np.asarray(first["executed_action"], dtype=float) if first is not None else None
            rows.append({
                "branch_id": branch_id,
                "task": candidate_row["task"],
                "episode": candidate_row["episode"],
                "route": candidate_row["route"],
                "route_spec_json": json.dumps(route_specs[candidate_row["route"]], sort_keys=True),
                "safe_success": safe(candidate_row),
                "success": bool(candidate_row["success"]),
                "safety_stop": bool(candidate_row["safety_stop"]),
                "steps": int(candidate_row["steps"]),
                "mode_switches": int(candidate_row["mode_switches"]),
                "guard_events": int(candidate_row["guard_events"]),
                "initial_requested_action": requested.tolist() if requested is not None else None,
                "initial_executed_action": executed.tolist() if executed is not None else None,
                "initial_action_l2": float(np.linalg.norm(requested[:6])) if requested is not None else None,
                "initial_action_linf": float(np.max(np.abs(requested[:6]))) if requested is not None else None,
                "initial_action_clip_linf": float(np.max(np.abs(requested - executed))) if requested is not None else None,
                "initial_gripper_sign": float(np.sign(requested[6])) if requested is not None else None,
                "initial_estimated_progress": float(first["estimated_initial_progress"]) if first is not None else float(candidate_row["estimated_initial_progress"]),
                "initial_retrieval_progress": float(first["retrieval_progress"]) if first is not None else None,
                "post_action_physical_progress": float(first["physical_progress"]) if first is not None else None,
                "post_action_force_l2": float(np.linalg.norm(first["ee_force"])) if first is not None and first["force_valid"] else None,
                "post_action_observation_only": first is not None,
                "label_outcome_key": list(outcome_key(candidate_row)),
                "default_safe_success": safe(default),
                "oracle_safe_success": safe(oracle),
                "oracle_route": oracle["route"],
                "candidate_is_safe_oracle": safe(candidate_row) and outcome_key(candidate_row) == outcome_key(oracle),
            })
        comparable = sum(a != b for i, a in enumerate(keys) for b in keys[i + 1 :])
        pair_count = len(keys) * (len(keys) - 1) // 2
        groups.append({
            "branch_id": branch_id,
            "task": candidates[0]["task"],
            "episode": candidates[0]["episode"],
            "candidate_count": len(candidates),
            "safe_candidate_count": sum(safe(x) for x in candidates),
            "default_safe_success": safe(default),
            "oracle_safe_success": safe(oracle),
            "oracle_route": oracle["route"],
            "default_demand": not safe(default),
            "oracle_headroom_available": safe(oracle) and not safe(default),
            "strictly_comparable_pair_fraction": comparable / max(1, pair_count),
            "tie_pair_count": pair_count - comparable,
            "candidate_outcome_keys": {x["route"]: list(outcome_key(x)) for x in candidates},
        })

    pq.write_table(pa.Table.from_pylist(rows), artifacts / "candidate_consequences.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(groups), artifacts / "counterfactual_groups.parquet", compression="zstd")
    restoration = audit_restoration(candidate, branches)
    dump(artifacts / "restoration_audit.json", restoration)

    safe_candidates = [x["safe_candidate_count"] for x in groups]
    metrics = {
        "status": "completed",
        "experiment": "EXP_R1",
        "module": "exact_expert_reconstruction_and_same_state_counterfactual_benchmark",
        "source_is_existing_exp27_formal": True,
        "new_simulator_rollouts": 0,
        "candidate_groups": len(groups),
        "candidate_rows": len(rows),
        "routes": route_names,
        "task_group_counts": {task: sum(x["task"] == task for x in groups) for task in sorted({x["task"] for x in groups})},
        "safe_candidate_count_distribution": {
            "min": int(min(safe_candidates)),
            "median": float(np.median(safe_candidates)),
            "max": int(max(safe_candidates)),
            "groups_with_any_safe_candidate": int(sum(x > 0 for x in safe_candidates)),
        },
        "default_safe_success_rate": float(np.mean([x["default_safe_success"] for x in groups])),
        "candidate_oracle_safe_success_rate": float(np.mean([x["oracle_safe_success"] for x in groups])),
        "default_demand_groups": int(sum(x["default_demand"] for x in groups)),
        "oracle_headroom_available_groups": int(sum(x["oracle_headroom_available"] for x in groups)),
        "strict_pair_fraction_mean": float(np.mean([x["strictly_comparable_pair_fraction"] for x in groups])),
        "tie_pair_fraction": float(sum(x["tie_pair_count"] for x in groups) / max(1, sum(len(route_names) * (len(route_names) - 1) // 2 for _ in groups))),
        "validity_checks": checks,
        "restoration_audit_passed": restoration["passed"],
        "selector_training_or_model_fitted": False,
        "formal_selection_claim": "not evaluated; this EXP validates the benchmark substrate only",
        "source_hashes": {
            "candidate_summaries": sha256(candidate / "artifacts/candidate_summaries.parquet"),
            "per_step": sha256(candidate / "artifacts/per_step.parquet"),
            "protocol": sha256(candidate / "manifests/recovery_protocol.json"),
            "branches": sha256(candidate / "manifests/branch_manifest.json"),
            "exp27_analysis": sha256(analysis / "metrics.json"),
            "exp27_final_audit": sha256(audit / "metrics.json"),
            "exp27_config": sha256(config_path),
        },
    }
    dump(artifacts / "expert_implementation_audit.json", source_expert_audit(protocol))
    dump(manifests / "benchmark_protocol.json", {
        "experiment": "EXP_R1",
        "unit": "one restored EXP27 branch state with exactly seven route candidates",
        "source_run": str(args.candidate_run),
        "selector_admissible_fields": [
            "branch identity/state representation available before candidate execution",
            "candidate route/action specification",
            "initial requested action and initial retrieved context",
        ],
        "forbidden_selector_fields": [
            "post_action_physical_progress",
            "post_action_force_l2",
            "success",
            "safety_stop",
            "steps",
            "terminal_object_positions",
            "oracle_route",
        ],
        "label_definition": "lexicographic (safe_success, not safety_stop, success); equal keys are ties",
        "candidate_filtering": "none; failed, unsafe, and tied candidates retained",
        "split_status": "no train/calibration/test model split; benchmark is a derived validity artifact from consumed EXP27 formal data",
        "independent_confirmation": False,
        "restoration_check": "structural corrected-D audit; no new simulator zero replay",
        "frozen_before_outcomes": True,
        "target_future_access": False,
    })
    dump(out / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
