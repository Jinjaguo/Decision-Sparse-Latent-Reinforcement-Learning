#!/usr/bin/env python
"""Build the EXP6 outcome-blind eight-branch-per-demo reference subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402
from scripts.exp4.freeze_protocol import articulation_progress, bowl_progress  # noqa: E402

TASKS = ["open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_plate", "turn_on_the_stove"]
CONTACT_TOKEN = {TASKS[0]: "wooden_cabinet_1", TASKS[1]: "akita_black_bowl_1", TASKS[2]: "flat_stove_1"}
TARGETS = np.asarray([0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_pairs(boundary: dict) -> list[str]:
    pairs = {"|".join(sorted((str(x["geom1_name"]), str(x["geom2_name"])))) for x in boundary["contact_pairs"]}
    return sorted(pairs)


def task_robot_contact(boundary: dict, task: str) -> bool:
    token = CONTACT_TOKEN[task]
    for pair in exact_pairs(boundary):
        left, right = pair.split("|", 1)
        if (token in left and "gripper0_" in right) or (token in right and "gripper0_" in left):
            return True
    return False


def gripper_state(command: float) -> str:
    if command < -0.5:
        return "negative"
    if command > 0.5:
        return "positive"
    return "neutral"


def select(boundaries: list[dict], task: str) -> list[dict]:
    progress = bowl_progress(boundaries)["clipped"] if task == TASKS[1] else articulation_progress(boundaries)["clipped"]
    candidates = []
    for index, boundary in enumerate(boundaries):
        norm = index / max(len(boundaries) - 1, 1)
        contact = task_robot_contact(boundary, task)
        grip = gripper_state(float(boundary["progress_channels"]["gripper_command"]))
        predicate = bool(boundary["progress_channels"]["exact_task_predicate"])
        candidates.append({"action_index": index, "normalized_time": norm, "physical_progress_raw": float(progress[index]), "physical_progress_clipped": float(progress[index]), "reference_contact_state": contact, "reference_gripper_state": grip, "reference_predicate_state": predicate, "reference_contact_pairs": exact_pairs(boundary), "reference_stratum": f"contact={int(contact)}|gripper={grip}|predicate={int(predicate)}"})
    selected, used = [], set()
    counts = {name: Counter() for name in ("reference_contact_state", "reference_gripper_state", "reference_predicate_state", "reference_stratum")}
    for slot, target in enumerate(TARGETS):
        scored = []
        for row in candidates:
            if row["action_index"] in used:
                continue
            diversity = sum(counts[name][row[name]] for name in counts)
            distance = 0.55 * abs(row["normalized_time"] - target) + 0.25 * abs(row["physical_progress_clipped"] - target) + 0.04 * diversity
            scored.append((distance, abs(row["normalized_time"] - target), row["action_index"], row))
        distance, _, _, winner = min(scored, key=lambda value: value[:3])
        winner = dict(winner)
        winner.update({"branch_time": winner["action_index"], "kind": "exp6_reference_stratified", "branch_index": slot, "target_octile": float(target), "selection_distance": float(distance), "deterministic_replacement_reason": "minimum frozen reference-only time/progress/diversity cost among unused boundaries"})
        selected.append(winner); used.add(winner["action_index"])
        for name in counts:
            counts[name][winner[name]] += 1
    return sorted(selected, key=lambda row: row["action_index"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    run = create_run_directory(args.run_root, args.run_id); reference = args.reference_run.resolve()
    source_path = reference / "artifacts/reference_snapshots_manifest.json"; source = json.loads(source_path.read_text())
    if not source["gate"]["passed"] or len(source["episodes"]) != 30:
        raise RuntimeError("EXP6 requires the passing 30-demo EXP5 reference cohort")
    trajectories = []
    for record in source["episodes"]:
        boundaries = json.loads((reference / record["relative_directory"] / "boundaries.json").read_text())
        branches = select(boundaries, record["task"])
        trajectories.append({"suite": record["suite"], "task": record["task"], "task_id": record["task_id"], "episode": record["episode"], "trajectory_length": len(boundaries), "branches": branches})
    trajectories.sort(key=lambda row: (TASKS.index(row["task"]), int(row["episode"].split("_")[-1])))
    calibration = []
    for task in TASKS:
        task_rows = [(trajectory, branch) for trajectory in trajectories if trajectory["task"] == task for branch in trajectory["branches"]]
        contact = [pair for pair in task_rows if pair[1]["reference_contact_state"]]
        no_contact = [pair for pair in task_rows if not pair[1]["reference_contact_state"]]
        chosen = []
        if contact and no_contact:
            chosen = [min(no_contact, key=lambda pair: pair[1]["normalized_time"]), max(contact, key=lambda pair: pair[1]["normalized_time"])]
        else:
            chosen = [min(task_rows, key=lambda pair: pair[1]["normalized_time"]), max(task_rows, key=lambda pair: pair[1]["normalized_time"])]
        for trajectory, branch in chosen:
            calibration.append({"task": task, "episode": trajectory["episode"], "branch_time": branch["action_index"], "reference_contact_state": branch["reference_contact_state"], "normalized_time": branch["normalized_time"], "selection_rule": "earliest no-contact plus latest contact when both exist; otherwise earliest plus latest"})
    manifest = {"schema_version": 1, "source_reference_run_id": source["run_id"], "source_reference_manifest_sha256": sha(source_path), "selection_is_q_outcome_blind": True, "selection_rule": "eight fixed octile targets with frozen reference-only time/progress/stratum diversity cost", "contact_definition": "exact target-object/gripper0_ named contact pair", "trajectory_count": len(trajectories), "branches_per_demo": 8, "branch_count": sum(len(x["branches"]) for x in trajectories), "trajectories": trajectories}
    contacts = sum(int(branch["reference_contact_state"]) for trajectory in trajectories for branch in trajectory["branches"])
    metrics = {"run_id": args.run_id, "status": "completed", "gate": {"passed": len(trajectories) == 30 and manifest["branch_count"] == 240 and len(calibration) == 6}, "trajectories": len(trajectories), "branches": manifest["branch_count"], "contact_branches": contacts, "noncontact_branches": manifest["branch_count"] - contacts, "calibration_branches": len(calibration)}
    write_json(run / "artifacts/exp6_branch_candidates.json", manifest); write_json(run / "artifacts/calibration_branch_candidates.json", {"schema_version": 1, "branches": calibration}); write_json(run / "artifacts/failure_examples.json", [])
    write_run_record(run, config={"stage": "EXP6 outcome-blind branch subset", "reference_run": reference.name}, command=shlex.join([sys.executable, *sys.argv]), environment={"python": sys.version, "numpy": np.__version__}, git_state={"project": git_record(ROOT)}, metrics=metrics, stdout=json.dumps(metrics), stderr="")
    print(json.dumps(metrics, indent=2)); return 0 if metrics["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
