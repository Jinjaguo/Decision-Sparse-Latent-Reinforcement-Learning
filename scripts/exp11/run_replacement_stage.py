"""Run EXP11 matched-zero and structured action-chunk replacements.

The intervention plan is materialized and hashed before the first perturbed
rollout.  Calibration and formal pilots use the same audited corrected-D
snapshot/controller restore path.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import shlex
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from decision_sparse_rl.envs.libero_runtime import bootstrap_runtime, environment_kwargs, load_episode, load_selection, task_source_record
from decision_sparse_rl.metrics.criticality import quaternion_geodesic, rotation_geodesic
from decision_sparse_rl.metrics.exp10 import frozen_phase_sequence
from decision_sparse_rl.metrics.exp11 import dct_modes, smooth_pulse_modes, spline_modes
from scripts.exp3.run_criticality import restore_d
from scripts.exp7.contact_geometry import load_schema, measure


TASKS = ("open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_plate", "turn_on_the_stove")
FAMILIES = ("I-A_analytic", "I-B_residual", "I-C_phase_edit", "I-E_gripper_timing")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def parquet(path: Path, rows: list[dict]) -> None:
    if not rows: raise RuntimeError(f"refusing to write empty required artifact: {path.name}")
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def pair_names(row: dict) -> list[str]:
    return ["|".join(sorted((f"{p['geom1_id']}:{p['geom1_name']}", f"{p['geom2_id']}:{p['geom2_name']}"))) for p in row["contact_pairs"]]


def phases_from_boundaries(rows: list[dict], schema: dict, task: str) -> np.ndarray:
    relevant = set(schema["tasks"][task]["pair_groups"].get("target_gripper", [] )[i]["pair"] for i in range(len(schema["tasks"][task]["pair_groups"].get("target_gripper", []))))
    contacts = [bool(set(pair_names(r)) & relevant) for r in rows]
    predicates = [bool(r["task_success"]) for r in rows]
    times = np.arange(len(rows), dtype=np.float64) / max(1, len(rows) - 1)
    return frozen_phase_sequence(times, contacts, predicates, min_dwell=2)


def choose_branches(phases: np.ndarray, actions: np.ndarray, task: str) -> list[dict]:
    candidates = []
    for phase in (1, 2, 3, 4):
        idx = np.flatnonzero(phases == phase)
        idx = idx[idx + 10 <= len(actions)]
        if len(idx): candidates.append({"branch_time": int(idx[0]), "phase": f"P{phase}", "kind": "first_crossing"})
    # Keep exactly four reference-only landmarks, filling unavailable phases by
    # distinct normalized-time landmarks.  No intervention outcome is read.
    used = {x["branch_time"] for x in candidates}
    for q in (.18, .38, .58, .78):
        if len(candidates) >= 4: break
        i = min(len(actions) - 11, max(0, int(round(q * (len(actions) - 1)))))
        while i in used and i + 10 < len(actions): i += 1
        used.add(i); candidates.append({"branch_time": i, "phase": f"P{int(phases[i])}", "kind": "time_landmark"})
    # If the task supports gripper edits, prefer one branch immediately before
    # a sign transition while retaining four landmarks.
    signs = np.sign(actions[:, 6]); transitions = np.flatnonzero((signs[1:] != signs[:-1]) & (signs[1:] != 0) & (signs[:-1] != 0)) + 1
    transitions = [int(i) for i in transitions if 1 <= i <= len(actions) - 10]
    if task != TASKS[0] and transitions:
        t = transitions[0] - 1
        candidates[-1] = {"branch_time": t, "phase": f"P{int(phases[t])}", "kind": "gripper_transition"}
    return sorted(candidates[:4], key=lambda x: x["branch_time"])


def training_residual_modes(task: str, phase: int, length: int, training_run: Path, phase_path: Path) -> np.ndarray:
    labels = pq.read_table(phase_path, filters=[("task", "=", task)]).to_pylist()
    lookup = {(r["episode"], int(r["action_index"])): int(r["phase_index"]) for r in labels}
    manifest = json.loads((training_run / "artifacts/reference_snapshots_manifest.json").read_text())
    chunks = []
    for rec in manifest["episodes"]:
        if rec["task"] != task: continue
        with np.load(training_run / rec["relative_directory"] / "trajectory_states.npz", allow_pickle=False) as z: a = np.asarray(z["actions"], dtype=np.float64)
        p = np.asarray([lookup[(rec["episode"], i)] for i in range(len(a))])
        idx = [i for i in range(len(a) - length + 1) if p[i] == phase and np.all(p[i:i+length] == phase)]
        for i in idx[::max(1, len(idx)//20)]: chunks.append(a[i:i+length, :6])
    if len(chunks) < 4: return np.empty((0, length, 6))
    x = np.asarray(chunks); centered = (x - x.mean(0)).reshape(len(x), -1)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    modes = vt[:2].reshape(-1, length, 6)
    return np.asarray([m / max(np.max(np.abs(m)), 1e-12) for m in modes])


def build_plan(reference_run: Path, support: dict, stage: str, training_run: Path, phase_path: Path, contact_schema: dict) -> tuple[list[dict], list[dict]]:
    records = json.loads((reference_run / "artifacts/reference_snapshots_manifest.json").read_text())["episodes"]
    branches, replacements = [], []
    for rec in records:
        ref_dir = reference_run / rec["relative_directory"]
        boundaries = json.loads((ref_dir / "boundaries.json").read_text())
        with np.load(ref_dir / "trajectory_states.npz", allow_pickle=False) as z: actions = np.asarray(z["actions"], dtype=np.float64)
        phases = phases_from_boundaries(boundaries, contact_schema, rec["task"])
        selected = choose_branches(phases, actions, rec["task"])
        for branch in selected:
            branch_id = f"{rec['task']}|{rec['episode']}|{branch['branch_time']}"
            branches.append({**branch, "branch_id": branch_id, "task": rec["task"], "episode": rec["episode"], "trajectory_length": len(actions), "reference_directory": rec["relative_directory"]})
            k = 10; t = branch["branch_time"]; chunk = actions[t:t+k]
            dominant = int(np.argmax(np.std(chunk[:, :6], axis=0)))
            if support[rec["task"]]["I-A_analytic"]:
                for family_name, maker in (("dct", dct_modes), ("spline", spline_modes), ("pulse", smooth_pulse_modes)):
                    for mode_index, mode in enumerate(maker(k, 3)):
                        if stage == "formal" and (family_name == "pulse" and mode_index > 0 or family_name != "pulse" and mode_index > 1): continue
                        temporal = mode / max(np.max(np.abs(mode)), 1e-12)
                        basis = np.zeros((k, 7)); basis[:, dominant] = temporal
                        for amplitude in ((.05, .10) if stage == "calibration" else (.10,)):
                            for sign in (-1, 1):
                                replacements.append({"branch_id": branch_id, "family": "I-A_analytic", "basis_family": family_name, "mode_index": mode_index, "channel": dominant, "chunk_length": k, "amplitude": amplitude, "sign": sign, "basis": basis.tolist()})
            phase = int(branch["phase"][1:])
            if support[rec["task"]]["I-B_residual"]:
                for mode_index, basis6 in enumerate(training_residual_modes(rec["task"], phase, k, training_run, phase_path)):
                    basis = np.zeros((k, 7)); basis[:, :6] = basis6
                    for amplitude in ((.05, .10) if stage == "calibration" else (.10,)):
                        for sign in (-1, 1): replacements.append({"branch_id": branch_id, "family": "I-B_residual", "basis_family": "residual_svd", "mode_index": mode_index, "channel": -1, "chunk_length": k, "amplitude": amplitude, "sign": sign, "basis": basis.tolist()})
            if support[rec["task"]]["I-C_phase_edit"]:
                for shift in (-1, 1): replacements.append({"branch_id": branch_id, "family": "I-C_phase_edit", "basis_family": "reference_index_shift", "mode_index": shift, "channel": -1, "chunk_length": k, "amplitude": 1.0, "sign": shift, "basis": []})
            signs = np.sign(actions[:, 6])
            transitions = np.flatnonzero((signs[1:] != signs[:-1]) & (signs[1:] != 0) & (signs[:-1] != 0)) + 1
            if support[rec["task"]]["I-E_gripper_timing"] and any(t <= x < t+k for x in transitions):
                for shift in (-1, 1): replacements.append({"branch_id": branch_id, "family": "I-E_gripper_timing", "basis_family": "sign_transition_shift", "mode_index": shift, "channel": 6, "chunk_length": k, "amplitude": 1.0, "sign": shift, "basis": []})
    for i, row in enumerate(replacements): row["intervention_id"] = f"{stage}|r{i:05d}"
    return branches, replacements


def modified_chunk(actions: np.ndarray, branch: dict, spec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, k = int(branch["branch_time"]), int(spec["chunk_length"]); ref = actions[t:t+k].copy(); requested = ref.copy()
    if spec["family"] in ("I-A_analytic", "I-B_residual"):
        requested += float(spec["sign"] * spec["amplitude"]) * np.asarray(spec["basis"])
    elif spec["family"] == "I-C_phase_edit":
        shift = int(spec["mode_index"]); indexes = np.clip(np.arange(t, t+k) + shift, 0, len(actions)-1); requested = actions[indexes].copy()
    elif spec["family"] == "I-E_gripper_timing":
        shift = int(spec["mode_index"]); requested[:, 6] = actions[np.clip(np.arange(t, t+k)+shift, 0, len(actions)-1), 6]
    executed = requested.copy(); executed[:, :6] = np.clip(executed[:, :6], -1, 1); executed[:, 6] = np.sign(executed[:, 6])
    return ref, requested, executed


def observation(env, body_ids: list[int], schema: dict, task: str) -> dict:
    robot = env.robots[0]; arm_q = [int(x) for x in robot._ref_joint_pos_indexes]; arm_v = [int(x) for x in robot._ref_joint_vel_indexes]; eef = int(robot.eef_site_id)
    geom = measure(env, schema, task)
    force_valid = torque_valid = True
    try: force = np.asarray(robot.ee_force, dtype=np.float64).reshape(-1)
    except Exception: force_valid = False; force = np.full(3, np.nan)
    try: torque = np.asarray(robot.ee_torque, dtype=np.float64).reshape(-1)
    except Exception: torque_valid = False; torque = np.full(3, np.nan)
    return {"integration": np.asarray(env.sim.get_state().flatten(), dtype=np.float64), "arm_q": np.asarray(env.sim.data.qpos[arm_q]).copy(), "arm_qvel": np.asarray(env.sim.data.qvel[arm_v]).copy(), "eef_position": np.asarray(env.sim.data.site_xpos[eef]).copy(), "eef_orientation": np.asarray(env.sim.data.site_xmat[eef]).copy(), "eef_linear_velocity": np.asarray(robot._hand_vel, dtype=np.float64).copy(), "object_positions": np.asarray(env.sim.data.body_xpos[body_ids]).copy(), "object_quaternions": np.asarray(env.sim.data.body_xquat[body_ids]).copy(), "predicate": bool(env.check_success()), "ee_force": force, "ee_torque": torque, "force_valid": force_valid, "torque_valid": torque_valid, **geom}


def rollout(env, actions: np.ndarray, start: int, replacement: np.ndarray | None, body_ids, schema, task) -> list[dict]:
    rows = []
    for i in range(start, len(actions)):
        action = replacement[i-start] if replacement is not None and i-start < len(replacement) else actions[i]
        env.step(action); rows.append(observation(env, body_ids, schema, task))
    return rows


def compare(a: dict, b: dict) -> dict:
    return {"integration_l2": float(np.linalg.norm(b["integration"]-a["integration"])), "arm_q_l2": float(np.linalg.norm(b["arm_q"]-a["arm_q"])), "arm_qvel_l2": float(np.linalg.norm(b["arm_qvel"]-a["arm_qvel"])), "eef_position_l2": float(np.linalg.norm(b["eef_position"]-a["eef_position"])), "eef_orientation_geodesic": rotation_geodesic(a["eef_orientation"], b["eef_orientation"]), "task_object_position_l2": float(np.linalg.norm(b["object_positions"]-a["object_positions"])), "task_object_orientation_geodesic_mean": float(np.mean([quaternion_geodesic(x,y) for x,y in zip(a["object_quaternions"], b["object_quaternions"])])), "regime_changed": a["contact_mode_json"] != b["contact_mode_json"], "signed_gap_delta_m": float(b["signed_gap_m"]-a["signed_gap_m"]), "normal_velocity_delta_mps": float(b["normal_relative_velocity_mps"]-a["normal_relative_velocity_mps"]), "force_l2_delta": float(np.linalg.norm(b["ee_force"]-a["ee_force"])) if a["force_valid"] and b["force_valid"] else float("nan"), "predicate_divergence": bool(a["predicate"] != b["predicate"])}


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--run-id", required=True); p.add_argument("--stage", choices=("calibration", "formal"), required=True); p.add_argument("--reference-run", required=True, type=Path); p.add_argument("--support-run", default=Path("runs/exp11_s0_s6_support_audit_20260814"), type=Path); p.add_argument("--training-run", default=Path("runs/exp8_s2_independent_refs_20260814"), type=Path); p.add_argument("--phase-run", default=Path("runs/exp10_a0_phase_macro_dataset_r2_20260814"), type=Path); p.add_argument("--authorized-families", nargs="*"); p.add_argument("--taskwise-authorization", type=Path); p.add_argument("--max-replacements", type=int)
    args = p.parse_args(); out = ROOT / "runs" / args.run_id
    if out.exists(): raise FileExistsError(f"immutable run exists: {out}")
    artifacts, manifests = out / "artifacts", out / "manifests"; artifacts.mkdir(parents=True); manifests.mkdir()
    stdout, stderr = io.StringIO(), io.StringIO(); env = None; started = datetime.now(timezone.utc).isoformat()
    try:
        contact_schema = load_schema(ROOT / "experiments/exp7_contact_mode_conditioned/manifests/contact_mode_schema.json")
        stage0 = json.loads((ROOT / args.support_run / "artifacts/stage0_family_support.json").read_text()); support = stage0["family_task_support"]
        if args.authorized_families:
            allowed = set(args.authorized_families)
            for task in support:
                for family in FAMILIES: support[task][family] = support[task][family] and family in allowed
        if args.taskwise_authorization:
            taskwise = json.loads((ROOT / args.taskwise_authorization).read_text())["task_family_authorization"]
            for task in support:
                for family in FAMILIES: support[task][family] = support[task][family] and bool(taskwise.get(task, {}).get(family, False))
        reference_run = (ROOT / args.reference_run).resolve(); training_run = (ROOT / args.training_run).resolve(); phase_path = ROOT / args.phase_run / "artifacts/reference_phase_labels.parquet"
        ref_manifest = json.loads((reference_run / "artifacts/reference_snapshots_manifest.json").read_text())
        if not ref_manifest["gate"]["passed"]: raise RuntimeError("reference gate failed")
        branches, specs = build_plan(reference_run, support, args.stage, training_run, phase_path, contact_schema)
        if args.max_replacements: specs = specs[:args.max_replacements]
        cohort = {"stage": args.stage, "reference_run": reference_run.name, "episodes": [{"task": r["task"], "episode": r["episode"], "success": r["success"]} for r in ref_manifest["episodes"]], "complete_demonstration_isolation": True, "qualification": "all requested demos succeeded and all snapshots finite"}
        raw_schema = {"per_step_required": ["requested_action", "executed_action", "reference_action", "clip_flags", "restore_hash", "eef_position", "eef_orientation", "eef_linear_velocity", "object_positions", "object_quaternions", "predicate", "contact_mode", "signed_gap", "normal_relative_velocity", "ee_force", "ee_torque"], "gripper": "exact sign only", "wrench_missing_policy": "NaN plus validity flag; never imputed"}
        allocation = {"families": sorted(set(x["family"] for x in specs)), "replacement_count": len(specs), "counts": {f: sum(x["family"] == f for x in specs) for f in FAMILIES}, "selection_before_outcomes": True}
        if args.stage == "formal" and len(allocation["families"]) > 3: raise RuntimeError("formal protocol permits at most three intervention families")
        paired = {"zero": "exact reference continuation from same corrected-D integration/controller snapshot", "perturbed": "replace first K actions then resume recorded reference actions", "paired_unit": "branch", "terminal_rollout": True}
        amplitude = {"continuous_normalized_action": [.05, .10], "signs": [-1, 1], "action_bounds": [-1, 1], "effect_threshold": {"macro_effect": .05, "terminal_object_position_m": .001, "regime_change_fraction": .05}, "frozen_before_outcomes": True}
        zero_gate = {"maximum_twin_integration_l2": 1e-10, "maximum_twin_eef_position_l2": 1e-10, "predicate_divergence_allowed": False}
        action_gate = {"requested_executed_linf_p95": 1e-12, "maximum_clipped_chunk_fraction": .10, "all_states_finite": True}
        wrench_schema = {"force_source": "robosuite SingleArm.ee_force from wrist force-torque sensor", "torque_source": "robosuite SingleArm.ee_torque", "runtime_audit_required": True, "no_imputation": True}
        manifest_payloads = {"calibration_cohort_manifest.json" if args.stage == "calibration" else "formal_cohort_manifest.json": cohort, "calibration_branch_manifest.json" if args.stage == "calibration" else "formal_branch_manifest.json": branches, "calibration_family_allocation.json" if args.stage == "calibration" else "formal_replacement_manifest.json": allocation, "amplitude_spec.json": amplitude, "paired_replacement_spec.json": paired, "raw_schema.json": raw_schema, "zero_gate.json": zero_gate, "action_execution_gate.json": action_gate, "wrench_force_schema.json": wrench_schema, "replacement_plan.json": specs}
        if args.stage == "formal":
            manifest_payloads.update({
                "selected_intervention_families.json": {"families": allocation["families"], "maximum": 3, "source": "Stage-1 authorization and effect ranking"},
                "model_route_specs.json": {"P-A": "linear functional ridge", "P-B": "object-centric ridge", "P-C": "phase-switching experts", "P-D": "GRU temporal predictor", "P-E": "object-graph MLP", "P-F": "residual mixture only if multimodality audit passes", "P-G": "terminal consequence classifier/regressor", "baselines": ["global mean", "action norm linear"]},
                "crossfit_manifest.json": {"unit": "complete (task, episode)", "folds": 5, "basis_fit": "training demos only", "conformal_subset": "deterministic held-in demo split", "no_row_split": True},
                "training_seed_manifest.json": {"numpy": 111011, "torch": 111012, "bootstrap": 111013, "mixture": 111014},
                "conformal_spec.json": {"method": "demo-level split conformal", "path_score": "max_t |y-mu|/scale", "target_coverage": .90, "accepted_interval": [.85,.95]},
                "statistical_analysis_plan.json": {"resampling_unit": "complete demonstration", "bootstrap_replicates": 1000, "trajectory_gate_relative_improvement": .15, "terminal_p90_improvement": .15, "task_replication": "2 of 3 tasks"},
                "macro_sensitivity_spec.json": {"effects": ["phase", "temporal mode", "amplitude", "paired sign asymmetry"], "amplitude_monotonicity": "Spearman and paired mean"},
                "decision_sparsity_spec.json": {"rank_reliability_min": .60, "top20_consequence_mass_min": .50, "heldout_mode_rank_rho_min": .60, "task_replication": "2 of 3"},
                "scientific_decision_rule.json": {"axes": ["causal_macro_effect", "temporal_basis_transfer", "object_centric_prediction", "switching_dynamics", "trajectory_distribution", "terminal_consequence", "macro_decision_sparsity"], "labels": ["replicated", "promising_pilot", "inconclusive", "negative"]},
                "gpu_analysis_spec.json": {"permitted": True, "simulator": "MuJoCo CPU validated path", "neural_routes": "CUDA when torch.cuda.is_available", "fallback": "CPU with same frozen seeds"},
            })
        for name, value in manifest_payloads.items(): dump(manifests / name, value)
        manifest_hashes = {name: sha_file(manifests / name) for name in manifest_payloads}; dump(manifests / "preoutcome_hashes.json", manifest_hashes)

        selection, task_manifest = load_selection(ROOT / "experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json", ROOT / "experiments/exp1_decision_sparsity/manifests/tasks.json"); selected = {x["name"]: x for x in selection["tasks"]}
        wrapper, robosuite_root, assets_root = bootstrap_runtime(ROOT / "third_party/LIBERO", ROOT / "data", artifacts / "libero_config")
        channel_schema = json.loads((ROOT / "experiments/exp3_time_indexed_q_criticality/manifests/effect_channel_schema.json").read_text())
        zero_summaries, replacements, per_steps, fidelity = [], [], [], []
        zero_lookup = {}; specs_by_branch = defaultdict(list)
        for spec in specs: specs_by_branch[spec["branch_id"]].append(spec)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            for demo_key, demo_branches in _group(branches, lambda x: (x["task"], x["episode"])).items():
                task, episode = demo_key; task_def = selected[task]; source = task_source_record(task_manifest, task_def["suite"], task_def["task_id"])
                env = wrapper.ControlEnv(**environment_kwargs(Path(source["bddl_file_path"])))
                load_episode(env, dataset_path=ROOT / "data" / task_def["demonstration_relative_path"], episode_index=int(episode.split("_")[-1]), robosuite_package_root=robosuite_root, libero_assets_root=assets_root)
                ref_dir = reference_run / demo_branches[0]["reference_directory"]
                with np.load(ref_dir / "trajectory_states.npz", allow_pickle=False) as z: actions = np.asarray(z["actions"], dtype=np.float64); integrations = np.asarray(z["integration"], dtype=np.float64)
                body_ids = [int(env.sim.model.body_name2id(name)) for name in channel_schema["task_object_audit"][task]["bodies"]]
                for branch in demo_branches:
                    t = int(branch["branch_time"]); controller = ref_dir / f"controller_{t:04d}.npz"; restore_hash = sha_bytes(integrations[t].tobytes() + bytes.fromhex(sha_file(controller)))
                    restore_d(env, integrations[t], controller); za = rollout(env, actions, t, None, body_ids, contact_schema, task)
                    restore_d(env, integrations[t], controller); zb = rollout(env, actions, t, None, body_ids, contact_schema, task)
                    diffs = [compare(a,b) for a,b in zip(za,zb)]
                    zsum = {**branch, "restore_hash": restore_hash, "continuation_length": len(za), "maximum_twin_integration_l2": max(x["integration_l2"] for x in diffs), "maximum_twin_eef_position_l2": max(x["eef_position_l2"] for x in diffs), "any_predicate_divergence": any(x["predicate_divergence"] for x in diffs), "terminal_success_a": za[-1]["predicate"], "terminal_success_b": zb[-1]["predicate"], "all_finite": all(np.all(np.isfinite(x["integration"])) for x in za+zb)}
                    zero_summaries.append(zsum); zero_lookup[branch["branch_id"]] = za
                    for spec in specs_by_branch[branch["branch_id"]]:
                        ref_chunk, requested, executed = modified_chunk(actions, branch, spec); clip = np.abs(requested-executed) > 1e-12
                        restore_d(env, integrations[t], controller); perturbed = rollout(env, actions, t, executed, body_ids, contact_schema, task)
                        effects = []
                        for offset, (zero, obs) in enumerate(zip(za, perturbed)):
                            effect = compare(zero, obs); norm = np.mean([effect["eef_position_l2"]/.01, effect["eef_orientation_geodesic"]/.1, effect["task_object_position_l2"]/.01])
                            req = requested[offset].tolist() if offset < len(requested) else actions[t+offset].tolist(); exe = executed[offset].tolist() if offset < len(executed) else actions[t+offset].tolist(); refa = actions[t+offset].tolist()
                            per_steps.append({"intervention_id": spec["intervention_id"], "task": task, "episode": episode, "branch_time": t, "phase": branch["phase"], "family": spec["family"], "basis_family": spec["basis_family"], "mode_index": spec["mode_index"], "amplitude": spec["amplitude"], "sign": spec["sign"], "continuation_offset": offset, "absolute_action_index": t+offset, "requested_action": req, "executed_action": exe, "reference_action": refa, "clip_flags": clip[offset].tolist() if offset < len(clip) else [False]*7, "restore_hash": restore_hash, "eef_position": obs["eef_position"].tolist(), "eef_orientation": obs["eef_orientation"].tolist(), "eef_linear_velocity": obs["eef_linear_velocity"].tolist(), "task_object_positions": obs["object_positions"].tolist(), "task_object_quaternions": obs["object_quaternions"].tolist(), "task_predicate": obs["predicate"], "contact_mode_json": obs["contact_mode_json"], "signed_gap_m": obs["signed_gap_m"], "normal_relative_velocity_mps": obs["normal_relative_velocity_mps"], "ee_force": obs["ee_force"].tolist(), "ee_torque": obs["ee_torque"].tolist(), "force_valid": obs["force_valid"], "torque_valid": obs["torque_valid"], **effect, "normalized_macro_step_effect": float(norm)})
                            effects.append((effect, norm))
                        h10 = effects[:min(10,len(effects))]; h20 = effects[:min(20,len(effects))]; terminal = effects[-1][0]
                        summary = {**{k:v for k,v in spec.items() if k != "basis"}, "task": task, "episode": episode, "branch_time": t, "phase": branch["phase"], "restore_hash": restore_hash, "requested_executed_linf": float(np.max(np.abs(requested-executed))), "clipped_action_fraction": float(np.mean(clip)), "clipped_chunk": bool(np.any(clip)), "macro_effect_h10": float(np.mean([x[1] for x in h10])), "macro_effect_h20": float(np.mean([x[1] for x in h20])), "trajectory_energy_h10": float(np.mean([x[1]**2 for x in h10])), "trajectory_energy_h20": float(np.mean([x[1]**2 for x in h20])), "terminal_object_position_l2": terminal["task_object_position_l2"], "terminal_orientation_geodesic": terminal["task_object_orientation_geodesic_mean"], "regime_change_fraction_h20": float(np.mean([x[0]["regime_changed"] for x in h20])), "terminal_success_flip": terminal["predicate_divergence"], "terminal_zero_success": za[-1]["predicate"], "terminal_perturbed_success": perturbed[-1]["predicate"], "all_states_finite": all(np.all(np.isfinite(x["integration"])) for x in perturbed)}
                        replacements.append(summary); fidelity.append({"intervention_id": spec["intervention_id"], "task": task, "family": spec["family"], "requested_executed_linf": summary["requested_executed_linf"], "clipped_chunk": summary["clipped_chunk"], "clipped_action_fraction": summary["clipped_action_fraction"], "all_states_finite": summary["all_states_finite"]})
                    print(json.dumps({"stage": args.stage, "branch": branch["branch_id"], "zero": len(zero_summaries), "replacements": len(replacements)}, sort_keys=True))
                env.close(); env = None
        zero_pass = max(x["maximum_twin_integration_l2"] for x in zero_summaries) <= zero_gate["maximum_twin_integration_l2"] and not any(x["any_predicate_divergence"] for x in zero_summaries) and all(x["all_finite"] for x in zero_summaries)
        parquet(artifacts / ("calibration_zero_controls.parquet" if args.stage == "calibration" else "zero_controls.parquet"), zero_summaries)
        parquet(artifacts / ("calibration_replacements.parquet" if args.stage == "calibration" else "replacements.parquet"), replacements)
        parquet(artifacts / ("calibration_per_step.parquet" if args.stage == "calibration" else "per_step_response.parquet"), per_steps)
        parquet(artifacts / "execution_fidelity.parquet", fidelity)
        effect_summary = []
        for (task, family), rows in _group(replacements, lambda x: (x["task"], x["family"])).items():
            values = np.asarray([r["macro_effect_h10"] for r in rows]); clips = np.asarray([r["clipped_chunk"] for r in rows]); effect_summary.append({"task":task,"family":family,"count":len(rows),"median_macro_effect_h10":float(np.median(values)),"p75_macro_effect_h10":float(np.quantile(values,.75)),"mean_macro_effect_h10":float(np.mean(values)),"nontrivial_fraction":float(np.mean(values>=amplitude["effect_threshold"]["macro_effect"])),"clipped_chunk_fraction":float(np.mean(clips)),"requested_executed_linf_p95":float(np.quantile([r["requested_executed_linf"] for r in rows],.95)),"terminal_success_flip_fraction":float(np.mean([r["terminal_success_flip"] for r in rows]))})
        parquet(artifacts / "effect_size_summary.parquet", effect_summary)
        force_valid = float(np.mean([r["force_valid"] and r["torque_valid"] for r in per_steps])); wrench_audit = {"force_torque_valid_fraction":force_valid,"schema_valid":force_valid==1.0,"source":"robosuite wrist sensor","no_imputation":True}; dump(artifacts / "wrench_schema_audit.json", wrench_audit)
        auth = {}
        for family in sorted(set(r["family"] for r in replacements)):
            rows = [r for r in effect_summary if r["family"] == family]; valid = all(r["clipped_chunk_fraction"]<=.10 and r["requested_executed_linf_p95"]<=1e-12 for r in rows); nontrivial = any(r["p75_macro_effect_h10"]>=.05 and r["nontrivial_fraction"]>=.10 for r in rows); auth[family]={"support":True,"execution_valid":valid,"nontrivial_effect":nontrivial,"authorized":bool(zero_pass and valid and nontrivial)}
        authorization = {"stage":args.stage,"zero_gate_passed":zero_pass,"families":auth,"authorized_families":[f for f,v in auth.items() if v["authorized"]]}; dump(artifacts / "family_authorization.json", authorization)
        report = "# EXP11 calibration report\n\n" + f"- Zero gate: **{'PASS' if zero_pass else 'FAIL'}**\n- Replacements: **{len(replacements)}**\n- Wrench validity: **{force_valid:.3f}**\n- Authorized: **{', '.join(authorization['authorized_families']) or 'none'}**\n\n" + "|task|family|n|median H10|p75 H10|nontrivial|clip|\n|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(f"|{r['task']}|{r['family']}|{r['count']}|{r['median_macro_effect_h10']:.4f}|{r['p75_macro_effect_h10']:.4f}|{r['nontrivial_fraction']:.3f}|{r['clipped_chunk_fraction']:.3f}|" for r in effect_summary) + "\n"
        (artifacts / "calibration_report.md").write_text(report, encoding="utf-8")
        metrics={"status":"completed","run_id":args.run_id,"stage":args.stage,"started_utc":started,"completed_utc":datetime.now(timezone.utc).isoformat(),"reference_count":len(ref_manifest["episodes"]),"branch_count":len(branches),"replacement_count":len(replacements),"per_step_count":len(per_steps),"zero_gate_passed":zero_pass,"wrench_valid_fraction":force_valid,"authorization":authorization,"manifest_hashes":manifest_hashes}
        dump(out / "metrics.json", metrics); dump(out / "config.json", {"command":shlex.join([sys.executable,*sys.argv])}); (out/"stdout.log").write_text(stdout.getvalue(),encoding="utf-8"); (out/"stderr.log").write_text(stderr.getvalue(),encoding="utf-8"); print(json.dumps(metrics,indent=2)); return 0 if zero_pass else 2
    except Exception as exc:
        if env is not None: env.close()
        stderr.write(traceback.format_exc()); dump(out/"metrics.json",{"status":"failed","error":repr(exc)}); (out/"stdout.log").write_text(stdout.getvalue(),encoding="utf-8"); (out/"stderr.log").write_text(stderr.getvalue(),encoding="utf-8"); raise


def _group(rows, key):
    result=defaultdict(list)
    for row in rows: result[key(row)].append(row)
    return result


if __name__ == "__main__": raise SystemExit(main())
