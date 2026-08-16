"""Run EXP_R3 with an explicit pre-action candidate interface.

EXP27 is not modified.  This runner reuses its frozen branch manifest and
route set, writes the candidate record before any route outcome is available,
and performs a matched zero replay for every restored branch.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
import pickle
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_sparse_rl.envs.libero_runtime import bootstrap_runtime, environment_kwargs, load_episode, load_selection, task_source_record
from scripts.exp3.run_criticality import restore_d
from scripts.exp7.contact_geometry import load_schema
import scripts.exp11.run_replacement_stage as engine
import scripts.exp15.run_recovery_stage as recovery


TASKS = engine.TASKS


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def atomic_dump(path: Path, value: object) -> None:
    """Write a recoverable JSON artifact without exposing a partial file."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_pickle(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def parquet(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty artifact: {path}")
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def append_progress(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def select_branches(branches: list[dict], branches_per_task: int | None, max_branches: int | None) -> list[dict]:
    """Select a deterministic, task-balanced calibration cohort when requested."""

    selected = list(branches)
    if branches_per_task is not None:
        grouped = defaultdict(list)
        for branch in branches:
            grouped[branch["task"]].append(branch)
        selected = []
        for task in sorted(grouped):
            cohort = sorted(grouped[task], key=lambda row: (int(row["episode"].split("_")[-1]), int(row["branch_time"]), row["branch_id"]))
            selected.extend(cohort[:branches_per_task])
        selected.sort(key=lambda row: (row["task"], int(row["episode"].split("_")[-1]), int(row["branch_time"]), row["branch_id"]))
    if max_branches is not None:
        selected = selected[:max_branches]
    return selected


def checkpoint_path(checkpoint_dir: Path, branch_id: str) -> Path:
    safe_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in branch_id)
    return checkpoint_dir / f"{safe_id}.json"


def safe(row: dict) -> bool:
    return bool(row["success"] and not row["safety_stop"])


def redirect_robosuite_log(path: Path) -> None:
    """Keep robosuite's legacy fixed log inside the immutable run artifact."""

    original = logging.FileHandler

    def factory(filename, *args, **kwargs):
        if Path(str(filename)).name.lower() == "robosuite.log":
            filename = str(path)
        return original(filename, *args, **kwargs)

    logging.FileHandler = factory


def state_hash(obs: dict) -> str:
    payload = obs["integration"].tobytes() + np.asarray(obs["eef_position"], dtype=np.float64).tobytes()
    return sha_bytes(payload)


def pre_action_record(branch: dict, spec: dict, obs: dict, requested, executed, retrieved, retrieval_progress, restore_hash: str, predicate_already_true: bool) -> dict:
    record = {
        "branch_id": branch["branch_id"],
        "task": branch["task"],
        "episode": branch["episode"],
        "route": spec["route"],
        "candidate_spec_json": json.dumps(spec, sort_keys=True),
        "restore_hash": restore_hash,
        "pre_action_state_hash": state_hash(obs),
        "pre_action_eef_position": np.asarray(obs["eef_position"], dtype=float).tolist(),
        "pre_action_object_positions": np.asarray(obs["object_positions"], dtype=float).tolist(),
        "pre_action_object_quaternions": np.asarray(obs["object_quaternions"], dtype=float).tolist(),
        "pre_action_contact_mode_json": obs["contact_mode_json"],
        "pre_action_ee_force": np.asarray(obs["ee_force"], dtype=float).tolist(),
        "pre_action_force_valid": bool(obs["force_valid"]),
        "pre_action_predicate": bool(obs["predicate"]),
        "predicate_already_true": bool(predicate_already_true),
        "executed": bool(executed is not None),
        "requested_action": requested.tolist() if requested is not None else None,
        "executed_action": executed.tolist() if executed is not None else None,
        "retrieved_indices": [int(x) for x in retrieved],
        "retrieval_progress": float(retrieval_progress) if retrieval_progress is not None else None,
        "pre_outcome_hash": None,
    }
    canonical = json.dumps({k: v for k, v in record.items() if k != "pre_outcome_hash"}, sort_keys=True, separators=(",", ":")).encode()
    record["pre_outcome_hash"] = sha_bytes(canonical)
    return record


def run_route(env, branch: dict, spec: dict, target_actions: np.ndarray, integrations: np.ndarray, controller: Path, body_ids, contact_schema: dict, task: str, library: dict, safety_envelope: dict | None) -> tuple[dict, list[dict], dict]:
    t = int(branch["branch_time"])
    restore_d(env, integrations[t], controller)
    obs = engine.observation(env, body_ids, contact_schema, task)
    restore_hash = sha_bytes(integrations[t].tobytes() + bytes.fromhex(sha_file(controller)))
    control = {**spec, **spec.get("task_controls", {}).get(task, {})}
    memory = {}
    pending = None
    requested = None
    retrieved = []
    clip_count = 0
    action_count = 0
    safety = False
    success = bool(obs["predicate"])
    route_steps = []
    pre_record = None
    exceedance_count = 0
    absolute_200 = False
    active_stage = -1
    mode_index = 0
    mode_switches = 0
    mode_started = 0
    progress_history = []
    physical_history = []
    retrieval_progress = 0.0
    guard_events = 0
    previous_action = None
    estimated_progress = recovery.estimate_progress(recovery.runtime_state(obs), library[task], branch["episode"])
    threshold = float(safety_envelope["tasks"][task]["primary_threshold_n"]) if safety_envelope else 200.0
    required = int(safety_envelope["tasks"][task]["consecutive_exceedances_to_stop"]) if safety_envelope else 1
    route_limit = int(control.get("max_steps", 320))

    if success:
        pre_record = pre_action_record(branch, spec, obs, None, None, [], None, restore_hash, True)

    for offset in range(route_limit):
        if success:
            break
        stage = sum(offset >= x for x in control.get("switch_steps", []))
        stage_start = 0 if stage == 0 else control["switch_steps"][stage - 1]
        if "stages" in control:
            active = recovery.EXP17_BASE[control["stages"][stage]]
        elif "modes" in control:
            mode_names = control.get("task_modes", {}).get(task, control["modes"])
            for band in control.get("progress_bands", {}).get(task, []):
                if estimated_progress >= float(band["minimum"]):
                    mode_names = band["modes"]
                    break
            active = recovery.EXP22_MODES[mode_names[min(mode_index, len(mode_names) - 1)]]
            stage = mode_index
            stage_start = mode_started
        else:
            active = control
        if stage != active_stage:
            memory = {}
            pending = None
            active_stage = stage
        if pending is None or (offset - stage_start) % active["replan"] == 0:
            requested, pending, retrieved, retrieval_progress = recovery.choose_chunk(active, recovery.runtime_state(obs), library[task], memory, branch["episode"])
            progress_history.append((offset, retrieval_progress))
        local = (offset - stage_start) % active["replan"]
        action = pending[min(local, len(pending) - 1)].copy()
        req = requested[min(local, len(requested) - 1)]
        if pre_record is None:
            # The first action is known to be executable from the pre-action
            # predicate; record that fact before env.step, rather than
            # mutating the selector row after the outcome exists.
            pre_record = pre_action_record(branch, spec, obs, requested, action, retrieved, retrieval_progress, restore_hash, False)
        clip = bool(np.any(np.abs(req[:6] - action[:6]) > 1e-12))
        guarded = False
        pre_force = float(np.linalg.norm(obs["ee_force"])) if obs["force_valid"] else float("nan")
        if control.get("force_guard") and np.isfinite(pre_force) and pre_force > float(control.get("guard_fraction", .7)) * threshold:
            guarded = True
            guard_events += 1
            if control["force_guard"] == "retract" and previous_action is not None:
                action[:6] = -float(control.get("guard_gain", 1.0)) * previous_action[:6]
            elif control["force_guard"] == "scale":
                action[:6] *= float(control.get("guard_scale", .25))
            else:
                action[:6] = 0.0
                pending = None
        action[:6] = np.clip(action[:6], -1, 1)
        action[6] = np.sign(action[6])
        clip_count += int(clip)
        action_count += 1
        previous_action = action.copy()
        env.step(action)
        obs = engine.observation(env, body_ids, contact_schema, task)
        force = float(np.linalg.norm(obs["ee_force"])) if obs["force_valid"] else float("nan")
        absolute_200 = absolute_200 or bool(np.isfinite(force) and force > 200)
        exceedance_count = exceedance_count + 1 if np.isfinite(force) and force > threshold else 0
        safety = bool(exceedance_count >= required or (np.isfinite(force) and force > 1000))
        success = bool(obs["predicate"])
        physical = recovery.task_physical_progress(env, task, obs)
        physical_history.append((offset, physical))
        route_steps.append({
            "branch_id": branch["branch_id"], "task": task, "episode": branch["episode"], "route": spec["route"],
            "offset": offset, "coordination_mode": active.get("selection", active.get("aggregate", "distance")),
            "mode_index": mode_index, "estimated_initial_progress": estimated_progress,
            "retrieval_progress": retrieval_progress, "physical_progress": physical, "force_guarded": guarded,
            "requested_action": req.tolist(), "executed_action": action.tolist(), "clipped": clip,
            "retrieved_indices": [int(x) for x in retrieved], "eef_position": obs["eef_position"].tolist(),
            "object_positions": obs["object_positions"].tolist(), "predicate": success,
            "contact_mode_json": obs["contact_mode_json"], "ee_force": obs["ee_force"].tolist(),
            "force_valid": obs["force_valid"], "safety_stop": safety,
        })
        if safety:
            break
        if "modes" in control and not success:
            mode_names = control.get("task_modes", {}).get(task, control["modes"])
            for band in control.get("progress_bands", {}).get(task, []):
                if estimated_progress >= float(band["minimum"]):
                    mode_names = band["modes"]
                    break
            dwell = offset - mode_started + 1
            switch = guarded
            if dwell >= int(control.get("maximum_dwell", 10**9)):
                switch = True
            window = int(control.get("stall_window", 0))
            recent = [v for step, v in progress_history if step >= offset - window + 1]
            if window and dwell >= int(control.get("minimum_dwell", window)) and len(recent) >= max(4, window // 2):
                half = len(recent) // 2
                switch = switch or max(recent[half:]) - max(recent[:half]) < float(control.get("minimum_progress_gain", 0.0))
            physical_window = int(control.get("physical_stall_window", 0))
            physical_recent = [v for step, v in physical_history if step >= offset - physical_window + 1]
            if physical_window and dwell >= int(control.get("minimum_dwell", physical_window)) and len(physical_recent) >= max(4, physical_window // 2):
                half = len(physical_recent) // 2
                switch = switch or max(physical_recent[half:]) - max(physical_recent[:half]) < float(control.get("minimum_physical_gain", 0.0))
            switch = switch or bool(np.isfinite(force) and force > .8 * threshold and dwell >= int(control.get("minimum_dwell", 1)))
            if switch and mode_index + 1 < len(mode_names):
                mode_index += 1
                mode_switches += 1
                mode_started = offset + 1
                active_stage = -1
                progress_history = []

    if pre_record is None:
        raise RuntimeError(f"candidate {branch['branch_id']} {spec['route']} never produced a pre-action record")
    # Keep the selector-facing row strictly pre-outcome.  The route outcome
    # belongs in candidate_summaries/per_step; writing it back here would
    # leak a post-action label into the admissible input table.
    summary = {
        "branch_id": branch["branch_id"], "task": task, "episode": branch["episode"], "route": spec["route"],
        "success": success, "safety_stop": safety, "absolute_200_exceeded": absolute_200,
        "steps": len(route_steps), "mode_switches": mode_switches, "guard_events": guard_events,
        "estimated_initial_progress": estimated_progress, "final_mode_index": mode_index,
        "clipped_action_fraction": clip_count / max(1, action_count),
        "all_states_finite": all(np.all(np.isfinite(x["eef_position"])) for x in route_steps),
        "terminal_contact_mode_json": obs["contact_mode_json"],
        "terminal_object_positions": obs["object_positions"].tolist(),
        "predicate_already_true": bool(pre_record["predicate_already_true"]),
        "pre_action_state_hash": pre_record["pre_action_state_hash"],
        "restore_hash": restore_hash,
    }
    return summary, route_steps, pre_record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("calibration", "formal"), required=True)
    parser.add_argument("--resume", action="store_true", help="resume an interrupted run from branch checkpoints")
    parser.add_argument("--branches-per-task", type=int, help="deterministic balanced calibration cohort size")
    parser.add_argument("--max-branches", type=int, help="optional deterministic cap after cohort selection")
    parser.add_argument("--branch-manifest", type=Path, default=Path("runs/exp27_s3_formal_new_time_cascade_20260815/manifests/branch_manifest.json"))
    parser.add_argument("--training-run", type=Path, default=Path("runs/exp8_s2_independent_refs_20260814"))
    parser.add_argument("--reference-run", type=Path, default=Path("runs/exp7_s2_independent_refs_20260814"), help="fallback reference run for legacy manifests without reference_run")
    parser.add_argument("--safety-envelope", type=Path, default=Path("runs/exp16_s0_expert_safety_audit_20260815/artifacts/expert_force_envelope.json"))
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    if out.exists() and not args.resume:
        raise FileExistsError(f"immutable run exists: {out}")
    artifacts, manifests = out / "artifacts", out / "manifests"
    checkpoints = artifacts / "checkpoints"
    artifacts.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    checkpoint_index_path = artifacts / "checkpoint_index.json"
    started = datetime.now(timezone.utc).isoformat()
    stdout, stderr = io.StringIO(), io.StringIO()
    env = None
    source_branches = json.loads((ROOT / args.branch_manifest).read_text(encoding="utf-8"))
    if args.branches_per_task is not None and args.branches_per_task <= 0:
        raise ValueError("--branches-per-task must be positive")
    if args.max_branches is not None and args.max_branches <= 0:
        raise ValueError("--max-branches must be positive")
    branches = select_branches(source_branches, args.branches_per_task, args.max_branches)
    if not branches:
        raise ValueError("branch selection produced an empty cohort")
    routes = recovery.EXP27_ROUTES
    protocol = {
        "experiment": "EXP_R3", "stage": args.stage, "routes": routes,
        "default_route": "D_physical_chunk", "training_run": str(args.training_run),
        "branch_manifest": str(args.branch_manifest), "target_future_candidate_access": False,
        "target_future_used_for_zero_audit_only": True, "expert_path_isolated": True,
        "exclude_target_demo_from_neighbors_and_scale": True, "pre_outcome_plan_frozen": True,
        "matched_zero_replay": "full reference continuation twice from each corrected-D restore",
        "maximum_rollout_steps": 320,
        "pre_outcome_excludes_post_action_labels": True,
        "source_branch_count": len(source_branches), "selected_branch_count": len(branches),
        "branches_per_task": args.branches_per_task, "max_branches": args.max_branches,
        "checkpoint_policy": "one atomic JSON checkpoint after every completed branch",
    }
    if not args.resume:
        dump(manifests / "candidate_protocol.json", protocol)
        dump(manifests / "source_branch_manifest.json", source_branches)
        dump(manifests / "branch_manifest.json", branches)
        dump(manifests / "preoutcome_hashes.json", {
            "protocol": sha_file(manifests / "candidate_protocol.json"),
            "source_branches": sha_file(manifests / "source_branch_manifest.json"),
            "branches": sha_file(manifests / "branch_manifest.json"),
        })
    else:
        existing = json.loads((manifests / "candidate_protocol.json").read_text(encoding="utf-8"))
        if existing.get("selected_branch_count") != len(branches) or existing.get("routes") != routes:
            raise RuntimeError("resume protocol does not match the requested branch or route plan")
    completed = {}
    try:
        training = (ROOT / args.training_run).resolve()
        training_manifest_hash = sha_file(training / "artifacts/reference_snapshots_manifest.json")
        append_progress(progress_path, {"event": "library_build_started", "training_manifest_hash": training_manifest_hash, "utc": datetime.now(timezone.utc).isoformat()})
        library_cache = artifacts / "training_library.pkl"
        library_cache_meta = artifacts / "training_library_meta.json"
        if library_cache.exists() and library_cache_meta.exists():
            cache_meta = json.loads(library_cache_meta.read_text(encoding="utf-8"))
            if cache_meta.get("training_manifest_hash") != training_manifest_hash:
                raise RuntimeError("training library cache hash does not match the requested training run")
            with library_cache.open("rb") as stream:
                library = pickle.load(stream)
            append_progress(progress_path, {"event": "library_cache_loaded", "cache": str(library_cache), "utc": datetime.now(timezone.utc).isoformat()})
        else:
            library = recovery.build_library(training)
            atomic_pickle(library_cache, library)
            atomic_dump(library_cache_meta, {"training_manifest_hash": training_manifest_hash, "library_tasks": sorted(library), "utc": datetime.now(timezone.utc).isoformat()})
            append_progress(progress_path, {"event": "library_build_completed", "cache": str(library_cache), "utc": datetime.now(timezone.utc).isoformat()})
        safety_envelope = json.loads((ROOT / args.safety_envelope).read_text(encoding="utf-8"))
        append_progress(progress_path, {"event": "safety_envelope_loaded", "utc": datetime.now(timezone.utc).isoformat()})
        selection, task_manifest = load_selection(ROOT / "experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json", ROOT / "experiments/exp1_decision_sparsity/manifests/tasks.json")
        selected = {x["name"]: x for x in selection["tasks"]}
        append_progress(progress_path, {"event": "task_manifests_loaded", "task_count": len(selected), "utc": datetime.now(timezone.utc).isoformat()})
        redirect_robosuite_log(artifacts / "robosuite.log")
        append_progress(progress_path, {"event": "robosuite_log_redirected", "utc": datetime.now(timezone.utc).isoformat()})
        numba_cache = artifacts / "numba_cache"
        numba_cache.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(numba_cache.resolve())
        append_progress(progress_path, {"event": "numba_cache_configured", "numba_cache_dir": str(numba_cache.resolve()), "utc": datetime.now(timezone.utc).isoformat()})
        # LIBERO's source bootstrap intentionally uses a write-once config
        # directory.  A resumed run must not reuse that directory, because
        # write_libero_config(..., exist_ok=False) would fail before import;
        # keep the original config as evidence and allocate an immutable
        # per-process config for this bootstrap attempt.
        config_directory = artifacts / "libero_config"
        if config_directory.exists():
            config_directory = artifacts / f"libero_config_resume_{os.getpid()}"
            append_progress(progress_path, {
                "event": "libero_config_reallocated",
                "config_directory": str(config_directory.resolve()),
                "utc": datetime.now(timezone.utc).isoformat(),
            })
        wrapper, robosuite_root, assets_root = bootstrap_runtime(ROOT / "third_party/LIBERO", ROOT / "data", config_directory)
        append_progress(progress_path, {"event": "libero_runtime_bootstrapped", "utc": datetime.now(timezone.utc).isoformat()})
        channel_schema = json.loads((ROOT / "experiments/exp3_time_indexed_q_criticality/manifests/effect_channel_schema.json").read_text(encoding="utf-8"))
        contact_schema = load_schema(ROOT / "experiments/exp7_contact_mode_conditioned/manifests/contact_mode_schema.json")
        append_progress(progress_path, {"event": "schemas_loaded", "utc": datetime.now(timezone.utc).isoformat()})
        summaries, steps, pre_rows, zero_rows = [], [], [], []
        completed = {}
        if args.resume:
            for checkpoint in sorted(checkpoints.glob("*.json")):
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                if payload.get("experiment") != "EXP_R3" or "branch_id" not in payload:
                    continue
                completed[payload["branch_id"]] = payload
            for branch in branches:
                payload = completed.get(branch["branch_id"])
                if payload is None:
                    continue
                zero_rows.extend(payload["zero_rows"])
                summaries.extend(payload["summaries"])
                pre_rows.extend(payload["pre_rows"])
                steps.extend(payload["steps"])
        atomic_dump(checkpoint_index_path, {
            "experiment": "EXP_R3", "status": "running", "selected_branch_count": len(branches),
            "completed_branch_ids": [branch["branch_id"] for branch in branches if branch["branch_id"] in completed],
        })
        append_progress(progress_path, {
            "event": "run_started", "stage": args.stage, "selected_branch_count": len(branches),
            "resumed_branch_count": len(completed), "route_count": len(routes), "utc": datetime.now(timezone.utc).isoformat(),
        })
        grouped = defaultdict(list)
        for branch in branches:
            grouped[(branch["task"], branch["episode"])].append(branch)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            for (task, episode), demo_branches in grouped.items():
                append_progress(progress_path, {
                    "event": "group_started", "task": task, "episode": episode,
                    "branch_count": len(demo_branches), "utc": datetime.now(timezone.utc).isoformat(),
                })
                task_def = selected[task]
                source = task_source_record(task_manifest, task_def["suite"], task_def["task_id"])
                env = wrapper.ControlEnv(**environment_kwargs(Path(source["bddl_file_path"])))
                load_episode(env, dataset_path=ROOT / "data" / task_def["demonstration_relative_path"], episode_index=int(episode.split("_")[-1]), robosuite_package_root=robosuite_root, libero_assets_root=assets_root)
                ref_root = ROOT / demo_branches[0].get("reference_run", args.reference_run)
                ref_dir = ref_root / demo_branches[0]["reference_directory"]
                with np.load(ref_dir / "trajectory_states.npz", allow_pickle=False) as archive:
                    target_actions = np.asarray(archive["actions"], dtype=float)
                    integrations = np.asarray(archive["integration"], dtype=float)
                body_ids = [int(env.sim.model.body_name2id(name)) for name in channel_schema["task_object_audit"][task]["bodies"]]
                for branch in demo_branches:
                    if branch["branch_id"] in completed:
                        append_progress(progress_path, {
                            "event": "branch_skipped_from_checkpoint", "branch_id": branch["branch_id"],
                            "task": task, "episode": episode, "utc": datetime.now(timezone.utc).isoformat(),
                        })
                        continue
                    append_progress(progress_path, {
                        "event": "branch_started", "branch_id": branch["branch_id"], "task": task,
                        "episode": episode, "branch_time": int(branch["branch_time"]),
                        "completed_branch_count": len(completed), "utc": datetime.now(timezone.utc).isoformat(),
                    })
                    t = int(branch["branch_time"])
                    controller = ref_dir / f"controller_{t:04d}.npz"
                    restore_hash = sha_bytes(integrations[t].tobytes() + bytes.fromhex(sha_file(controller)))
                    restore_d(env, integrations[t], controller)
                    zero_a = engine.rollout(env, target_actions, t, None, body_ids, contact_schema, task)
                    restore_d(env, integrations[t], controller)
                    zero_b = engine.rollout(env, target_actions, t, None, body_ids, contact_schema, task)
                    if len(zero_a) != len(zero_b):
                        raise RuntimeError(f"zero replay length mismatch: {branch['branch_id']}")
                    diffs = [engine.compare(a, b) for a, b in zip(zero_a, zero_b)]
                    zero_row = {
                        "branch_id": branch["branch_id"], "task": task, "episode": episode, "branch_time": t,
                        "restore_hash": restore_hash, "continuation_length": len(zero_a),
                        "maximum_twin_integration_l2": max((x["integration_l2"] for x in diffs), default=0.0),
                        "maximum_twin_eef_position_l2": max((x["eef_position_l2"] for x in diffs), default=0.0),
                        "any_predicate_divergence": any(x["predicate_divergence"] for x in diffs),
                        "all_finite": all(np.all(np.isfinite(x["integration"])) for x in zero_a + zero_b),
                    }
                    zero_rows.append(zero_row)
                    branch_summaries, branch_steps, branch_pre_rows = [], [], []
                    for spec in routes:
                        summary, route_steps, pre_record = run_route(env, branch, spec, target_actions, integrations, controller, body_ids, contact_schema, task, library, safety_envelope)
                        summaries.append(summary)
                        steps.extend(route_steps)
                        pre_rows.append(pre_record)
                        branch_summaries.append(summary)
                        branch_steps.extend(route_steps)
                        branch_pre_rows.append(pre_record)
                    checkpoint = {
                        "experiment": "EXP_R3", "stage": args.stage, "branch_id": branch["branch_id"],
                        "task": task, "episode": episode, "branch_time": t,
                        "zero_rows": [zero_row], "summaries": branch_summaries,
                        "pre_rows": branch_pre_rows, "steps": branch_steps,
                        "completed_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    atomic_dump(checkpoint_path(checkpoints, branch["branch_id"]), checkpoint)
                    completed[branch["branch_id"]] = checkpoint
                    atomic_dump(checkpoint_index_path, {
                        "experiment": "EXP_R3", "status": "running", "selected_branch_count": len(branches),
                        "completed_branch_ids": [item["branch_id"] for item in branches if item["branch_id"] in completed],
                    })
                    append_progress(progress_path, {
                        "event": "branch_completed", "branch_id": branch["branch_id"], "task": task,
                        "episode": episode, "completed_branch_count": len(completed),
                        "candidate_count": len(summaries), "per_step_count": len(steps),
                        "utc": datetime.now(timezone.utc).isoformat(),
                    })
                env.close()
                env = None
                append_progress(progress_path, {
                    "event": "group_completed", "task": task, "episode": episode,
                    "completed_branch_count": len(completed), "utc": datetime.now(timezone.utc).isoformat(),
                })
        parquet(artifacts / "pre_outcome_candidates.parquet", pre_rows)
        parquet(artifacts / "candidate_summaries.parquet", summaries)
        parquet(artifacts / "per_step.parquet", steps)
        parquet(artifacts / "matched_zero_controls.parquet", zero_rows)
        zero_pass = all(x["maximum_twin_integration_l2"] <= 1e-10 and x["maximum_twin_eef_position_l2"] <= 1e-10 and not x["any_predicate_divergence"] and x["all_finite"] for x in zero_rows)
        metrics = {
            "status": "completed", "experiment": "EXP_R3", "stage": args.stage,
            "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
            "source_branch_count": len(source_branches), "branch_count": len(branches), "route_count": len(routes),
            "candidate_count": len(summaries), "pre_outcome_candidate_count": len(pre_rows),
            "per_step_count": len(steps), "zero_control_count": len(zero_rows),
            "zero_gate_passed": zero_pass, "target_future_candidate_access": False,
            "target_future_used_for_zero_audit_only": True,
            "predicate_already_true_count": sum(x["predicate_already_true"] for x in pre_rows),
            "pre_action_rows_with_action": sum(x["requested_action"] is not None for x in pre_rows),
            "pre_action_rows_without_action": sum(x["requested_action"] is None for x in pre_rows),
            "complete_candidate_matrix": len(summaries) == len(branches) * len(routes),
            "checkpoint_count": len(completed), "branches_per_task": args.branches_per_task,
            "max_branches": args.max_branches,
            "source_hashes": {"branch_manifest": sha_file(ROOT / args.branch_manifest), "training_manifest": sha_file(training / "artifacts/reference_snapshots_manifest.json"), "safety_envelope": sha_file(ROOT / args.safety_envelope)},
        }
        dump(out / "metrics.json", metrics)
        atomic_dump(checkpoint_index_path, {"experiment": "EXP_R3", "status": "completed", "selected_branch_count": len(branches), "completed_branch_ids": [branch["branch_id"] for branch in branches]})
        append_progress(progress_path, {"event": "run_completed", "status": metrics["status"], "candidate_count": len(summaries), "utc": datetime.now(timezone.utc).isoformat()})
        (out / "stdout.log").write_text(stdout.getvalue(), encoding="utf-8")
        (out / "stderr.log").write_text(stderr.getvalue(), encoding="utf-8")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return 0 if zero_pass and metrics["complete_candidate_matrix"] else 2
    except KeyboardInterrupt:
        if env is not None:
            env.close()
        partial = {
            "status": "interrupted", "experiment": "EXP_R3", "stage": args.stage,
            "started_utc": started, "interrupted_utc": datetime.now(timezone.utc).isoformat(),
            "source_branch_count": len(source_branches), "branch_count": len(branches),
            "route_count": len(routes), "candidate_count": len(summaries),
            "pre_outcome_candidate_count": len(pre_rows), "per_step_count": len(steps),
            "zero_control_count": len(zero_rows), "checkpoint_count": len(completed),
            "complete_candidate_matrix": len(summaries) == len(branches) * len(routes),
        }
        dump(out / "metrics.json", partial)
        atomic_dump(checkpoint_index_path, {
            "experiment": "EXP_R3", "status": "interrupted", "selected_branch_count": len(branches),
            "completed_branch_ids": [branch["branch_id"] for branch in branches if branch["branch_id"] in completed],
        })
        append_progress(progress_path, {"event": "run_interrupted", "checkpoint_count": len(completed), "utc": datetime.now(timezone.utc).isoformat()})
        (out / "stdout.log").write_text(stdout.getvalue(), encoding="utf-8")
        (out / "stderr.log").write_text(stderr.getvalue(), encoding="utf-8")
        raise
    except Exception as exc:
        if env is not None:
            env.close()
        stderr.write(traceback.format_exc())
        dump(out / "metrics.json", {"status": "failed", "error": repr(exc), "checkpoint_count": len(locals().get("completed", {}))})
        (out / "stdout.log").write_text(stdout.getvalue(), encoding="utf-8")
        (out / "stderr.log").write_text(stderr.getvalue(), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
