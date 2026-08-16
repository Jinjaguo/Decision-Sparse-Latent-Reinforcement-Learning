"""Audit whether EXP_R1 contains admissible pre-action features for EXP_R2 baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def unique_count(values: list[object]) -> int:
    normalized = []
    for value in values:
        if value is None:
            normalized.append("<MISSING>")
        elif isinstance(value, list):
            normalized.append(json.dumps(value, sort_keys=True))
        else:
            normalized.append(str(value))
    return len(set(normalized))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--benchmark-run", type=Path, default=Path("runs/exp_r1_s1_same_state_benchmark_20260816_r2"))
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    if out.exists():
        raise FileExistsError(f"immutable run exists: {out}")
    artifacts = out / "artifacts"
    manifests = out / "manifests"
    artifacts.mkdir(parents=True)
    manifests.mkdir(parents=True)

    source = ROOT / args.benchmark_run
    rows = pq.read_table(source / "artifacts/candidate_consequences.parquet").to_pylist()
    groups = defaultdict(list)
    for row in rows:
        groups[row["branch_id"]].append(row)
    route_order = sorted({row["route"] for row in rows})

    # These are the only current-state/candidate fields present in the R1 table.
    # Post-action fields are listed separately and are never used as inputs here.
    current_state_fields = ["initial_estimated_progress", "initial_retrieval_progress"]
    candidate_action_fields = [
        "initial_requested_action", "initial_executed_action", "initial_action_l2",
        "initial_action_linf", "initial_action_clip_linf", "initial_gripper_sign",
    ]
    post_action_fields = [
        "post_action_physical_progress", "post_action_force_l2", "safe_success",
        "success", "safety_stop", "steps", "oracle_route",
    ]

    group_rows = []
    for branch_id, candidates in sorted(groups.items()):
        task = candidates[0]["task"]
        state_unique = {field: unique_count([x[field] for x in candidates]) for field in current_state_fields}
        action_unique = {field: unique_count([x[field] for x in candidates]) for field in candidate_action_fields}
        group_rows.append({
            "branch_id": branch_id,
            "task": task,
            "episode": candidates[0]["episode"],
            "candidate_count": len(candidates),
            "state_field_unique_counts_json": json.dumps(state_unique, sort_keys=True),
            "candidate_action_unique_counts_json": json.dumps(action_unique, sort_keys=True),
            "state_only_distinguishes_candidates": any(x > 1 for x in state_unique.values()),
            "any_candidate_action_available": any(x["initial_action_l2"] is not None for x in candidates),
            "all_candidate_actions_available": all(x["initial_action_l2"] is not None for x in candidates),
        })

    state_only_groups = sum(x["state_only_distinguishes_candidates"] for x in group_rows)
    action_rows = sum(row["initial_action_l2"] is not None for row in rows)
    missing_action_rows = len(rows) - action_rows
    field_rows = []
    for field in current_state_fields + candidate_action_fields + post_action_fields:
        values = [row[field] for row in rows]
        missing = sum(value is None for value in values)
        field_rows.append({
            "field": field,
            "role": "current_state" if field in current_state_fields else ("candidate_action" if field in candidate_action_fields else "post_action_label"),
            "row_count": len(values),
            "missing_count": missing,
            "missing_fraction": missing / max(1, len(values)),
            "available_for_online_selector": field in current_state_fields + candidate_action_fields,
            "used_in_exp_r2_selection": False,
        })
    pq.write_table(pa.Table.from_pylist(group_rows), artifacts / "group_feature_audit.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(field_rows), artifacts / "field_audit.parquet", compression="zstd")

    feasibility = {
        "retrieval_only": {
            "status": "blocked_as_faithful_comparison",
            "reason": "The R1 route-level artifact does not contain the historical action-library row identity needed to reproduce state-to-action retrieval. Current-state fields are common across candidates in the relevant zero-step Bowl groups, so a state-only retriever cannot identify a route there.",
            "state_only_distinguishable_group_fraction": state_only_groups / max(1, len(group_rows)),
        },
        "scalar_verifier": {
            "status": "not_authorized_yet",
            "reason": "A scalar model could fit route priors, but without a complete pre-action candidate context it would mostly measure route identity and post-hoc outcome censoring rather than scalar action verification.",
            "route_identity_available": True,
            "candidate_action_rows_available": action_rows,
        },
        "factorized_consequence": {
            "status": "not_authorized_yet",
            "reason": "The measurable physical-progress and force fields in R1 are post-action observations. They may be future training targets after instrumentation, but cannot be online inputs; current pre-action fields are insufficient to test factorized prediction fairly.",
            "post_action_rows": len(rows),
            "pre_action_state_fields": current_state_fields,
        },
    }
    dump(artifacts / "baseline_feasibility.json", feasibility)
    dump(manifests / "feature_protocol.json", {
        "experiment": "EXP_R2",
        "question": "Does the current same-state artifact contain sufficient admissible pre-action context for matched retrieval-only, scalar, and factorized selectors?",
        "current_state_fields": current_state_fields,
        "candidate_action_fields": candidate_action_fields,
        "post_action_fields_forbidden_as_inputs": post_action_fields,
        "route_identity_is_not_current_state": True,
        "no_models_fitted": True,
        "no_post_action_selection": True,
        "split": "not opened because prerequisite feature audit failed for a fair model comparison",
    })

    metrics = {
        "status": "completed",
        "experiment": "EXP_R2",
        "module": "admissible_pre_action_feature_and_baseline_feasibility_audit",
        "source_run": str(args.benchmark_run),
        "candidate_rows": len(rows),
        "candidate_groups": len(group_rows),
        "routes": route_order,
        "current_state_fields": current_state_fields,
        "candidate_action_fields": candidate_action_fields,
        "post_action_fields": post_action_fields,
        "state_only_distinguishable_groups": state_only_groups,
        "state_only_distinguishable_group_fraction": state_only_groups / max(1, len(group_rows)),
        "candidate_action_rows_available": action_rows,
        "candidate_action_rows_missing": missing_action_rows,
        "candidate_action_missing_fraction": missing_action_rows / max(1, len(rows)),
        "zero_step_candidate_rows": missing_action_rows,
        "models_fitted": False,
        "baseline_feasibility": feasibility,
        "source_hash": sha256(source / "artifacts/candidate_consequences.parquet"),
        "interpretation": "Instrumentation prerequisite identified; do not claim a negative ranking result from this audit.",
    }
    dump(out / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
