"""EXP11 Stage-0 reference-only support, basis, and availability audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from decision_sparse_rl.metrics.exp11 import dct_modes, residual_basis, smooth_pulse_modes, spline_modes, subspace_similarity


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_plate", "turn_on_the_stove")
LENGTHS = (5, 10, 20)


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def sampled_indexes(indexes, maximum=20):
    if len(indexes) <= maximum: return indexes
    return [indexes[i] for i in np.linspace(0, len(indexes) - 1, maximum).astype(int)]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True); parser.add_argument("--reference-run", default="runs/exp8_s2_independent_refs_20260814"); parser.add_argument("--phase-run", default="runs/exp10_a0_phase_macro_dataset_r2_20260814")
    args = parser.parse_args(); out = ROOT / "runs" / args.run_id
    if out.exists(): raise FileExistsError(f"immutable run exists: {out}")
    artifacts, manifests, plots = out / "artifacts", out / "manifests", out / "plots"; artifacts.mkdir(parents=True); manifests.mkdir(); plots.mkdir()
    reference = ROOT / args.reference_run; phase_path = ROOT / args.phase_run / "artifacts/reference_phase_labels.parquet"
    reference_manifest = reference / "artifacts/reference_snapshots_manifest.json"
    sources = {
        "reference_manifest": reference_manifest, "phase_labels": phase_path,
        "single_arm_source": Path(r"C:\Users\Guoji\anaconda3\envs\libero-exp1\Lib\site-packages\robosuite\robots\single_arm.py"),
        "osc_source": Path(r"C:\Users\Guoji\anaconda3\envs\libero-exp1\Lib\site-packages\robosuite\controllers\osc.py"),
        "osc_config": Path(r"C:\Users\Guoji\anaconda3\envs\libero-exp1\Lib\site-packages\robosuite\controllers\config\osc_pose.json"),
        "panda_gripper_source": Path(r"C:\Users\Guoji\anaconda3\envs\libero-exp1\Lib\site-packages\robosuite\models\grippers\panda_gripper.py"),
    }
    now = datetime.now(timezone.utc).isoformat(); source_hashes = {k: {"path": str(v), "sha256": sha(v)} for k, v in sources.items()}
    semantics = {
        "action_dim": 7, "controller": "OSC_POSE", "action_limits": [-1.0, 1.0],
        "channels": {"0:3": "controller-scaled EEF position delta", "3:6": "controller-scaled axis-angle orientation delta", "6": "Panda gripper sign command"},
        "controller_output_scale": {"position_m_at_normalized_one": .05, "orientation_axis_angle_rad_at_normalized_one": .5},
        "controller_input_clipping": "BaseController.scale_action clips each arm input to [-1,1] before scaling",
        "gripper_semantics": "PandaGripper uses sign(action); each policy step integrates current_action by [-.01,+.01] for its two fingers, clipped to [-1,1]",
        "gripper_continuous_interpolation_allowed": False,
        "source_hashes": {k: v["sha256"] for k, v in source_hashes.items() if k.endswith("source") or k == "osc_config"},
        "audited_utc": now,
    }
    phase_schema = {"source": str(phase_path), "source_sha256": sha(phase_path), "landmarks": [f"first crossing P{i}" for i in range(1, 7)], "selection": "reference-only, monotone EXP10 P0-P6", "outcome_blind": True}
    chunk_spec = {"lengths": list(LENGTHS), "continuous_channels": list(range(6)), "gripper_channel": 6, "right_padding": "not allowed for Stage-0 rank/support rows", "calibration_candidate_lengths": [5, 10]}
    analytic_spec = {"families": ["dct", "cubic_bernstein_spline", "smooth_gaussian_pulse"], "modes_per_length": 3, "low_frequency_only": True, "temporal_mode_linf_normalized": 1.0, "channel_application": "one audited continuous arm channel at a time"}
    residual_spec = {"construction": "task x phase x K, training/reference demos only", "leave_demo_out": True, "rank": 3, "sampling_cap_per_demo_phase": 20, "nontrivial_singular_ratio": .02, "stable_subspace_similarity": .5}
    phase_edit_spec = {"edits": ["one-step advance", "one-step delay", "phase-duration extend/shorten"], "implementation": "deterministic index shift inside the same frozen reference phase", "minimum_phase_support_demos": 8, "outcome_blind": True}
    taskspace_spec = {"available": False, "reason": "OSC channel semantics are audited, but no exact task-progress-to-OSC/IK mapping is validated for all three tasks; no targeted task-space edit is improvised"}
    gripper_spec = {"edits": ["sign-transition advance by one step", "sign-transition delay by one step"], "minimum_transition_demos_per_task": 8, "continuous_interpolation": False}
    unused = {task: {"unused_candidates": list(range(41, 50)), "count": 9, "calibration_ascending": [41, 42], "remaining_after_calibration": list(range(43, 50)), "stage2_status_if_all_qualify": "availability-limited pilot (7/task < 8 full-confirmation floor)"} for task in TASKS}
    for name, value in {
        "exp11_source_hash_manifest.json": source_hashes, "action_semantics_audit.json": semantics,
        "phase_landmark_schema.json": phase_schema, "chunk_length_spec.json": chunk_spec,
        "analytic_basis_spec.json": analytic_spec, "residual_basis_spec.json": residual_spec,
        "phase_edit_spec.json": phase_edit_spec, "taskspace_edit_spec.json": taskspace_spec,
        "gripper_timing_spec.json": gripper_spec, "unused_demo_availability.json": unused,
    }.items(): (manifests / name).write_text(json.dumps(value, indent=2), encoding="utf-8")

    phase_rows = pq.read_table(phase_path).to_pylist(); phase_lookup = {(r["task"], r["episode"], int(r["action_index"])): int(r["phase_index"]) for r in phase_rows}
    refs = json.loads(reference_manifest.read_text(encoding="utf-8"))["episodes"]
    actions, phases = {}, {}
    for record in refs:
        key = (record["task"], record["episode"]); path = reference / record["relative_directory"] / "trajectory_states.npz"
        with np.load(path, allow_pickle=False) as z: actions[key] = np.asarray(z["actions"], dtype=np.float64)
        phases[key] = np.asarray([phase_lookup[(key[0], key[1], i)] for i in range(len(actions[key]))])

    support_rows, spectrum_rows, stability_rows, bound_rows = [], [], [], []
    for task in TASKS:
        demos = sorted(k for k in actions if k[0] == task)
        for phase in range(7):
            gripper_event_demos = 0
            for key in demos:
                idx = np.flatnonzero(phases[key] == phase); signs = np.sign(actions[key][:, 6])
                if any(i > 0 and signs[i] != 0 and signs[i - 1] != 0 and signs[i] != signs[i - 1] for i in idx): gripper_event_demos += 1
            for length in LENGTHS:
                chunks, labels = [], []
                demo_counts = 0
                for key in demos:
                    idx = [int(i) for i in np.flatnonzero(phases[key] == phase) if i + length <= len(actions[key]) and np.all(phases[key][i : i + length] == phase)]
                    idx = sampled_indexes(idx)
                    if idx: demo_counts += 1
                    for i in idx: chunks.append(actions[key][i : i + length, :6]); labels.append(key[1])
                if chunks:
                    x = np.asarray(chunks); channel_var = np.var(x.reshape(-1, 6), axis=0); mean_action = x.mean((0, 1))
                    _, basis, singular = residual_basis(x, rank=3) if len(x) > 3 else (None, np.empty((0, length * 6)), np.asarray([]))
                    ratios = singular / max(singular[0] if len(singular) else 1, 1e-15); rank = int(np.sum(ratios >= .02))
                    similarities = []
                    if len(basis):
                        for held in sorted(set(labels)):
                            train = x[np.asarray(labels) != held]
                            if len(train) > 3:
                                _, loo, _ = residual_basis(train, rank=3); similarities.append(subspace_similarity(basis, loo))
                    stability = float(np.mean(similarities)) if similarities else 0.0
                else:
                    x = np.empty((0, length, 6)); channel_var = np.zeros(6); mean_action = np.zeros(6); singular = np.asarray([]); ratios = np.asarray([]); rank = 0; stability = 0.0
                support_rows.append({"task": task, "phase": f"P{phase}", "chunk_length": length, "demo_count": demo_counts, "chunk_count": len(chunks), "mean_action": mean_action.tolist(), "channel_variance": channel_var.tolist(), "nontrivial_channel_count": int(np.sum(channel_var > 1e-5)), "residual_rank": rank, "gripper_transition_demo_count": gripper_event_demos})
                for i, (value, ratio) in enumerate(zip(singular[:12], ratios[:12])): spectrum_rows.append({"task": task, "phase": f"P{phase}", "chunk_length": length, "singular_index": i, "singular_value": float(value), "relative_singular_value": float(ratio)})
                stability_rows.append({"task": task, "phase": f"P{phase}", "chunk_length": length, "leave_demo_out_subspace_similarity": stability, "heldout_demo_count": len(similarities), "stable_top3": bool(rank >= 3 and stability >= .5)})
                if chunks:
                    for family, maker in (("dct", dct_modes), ("spline", spline_modes), ("pulse", smooth_pulse_modes)):
                        modes = maker(length, 3)
                        for mode_index, mode in enumerate(modes):
                            temporal = mode / np.max(np.abs(mode))
                            for amplitude in (.05, .10):
                                total = clipped = 0
                                for chunk in x:
                                    for channel in range(6):
                                        requested = chunk.copy(); requested[:, channel] += amplitude * temporal
                                        clipped += int(np.any((requested < -1) | (requested > 1))); total += 1
                                bound_rows.append({"task": task, "phase": f"P{phase}", "chunk_length": length, "basis_family": family, "mode_index": mode_index, "amplitude": amplitude, "evaluated_chunks_channels": total, "clipped_chunks_channels": clipped, "clipping_fraction": clipped / max(1, total), "orthogonality_max_error": float(np.max(np.abs(modes @ modes.T - np.eye(3)))), "roughness": float(np.mean(np.diff(mode, n=2) ** 2))})
    for rows, name in ((support_rows, "phase_action_support.parquet"), (spectrum_rows, "action_residual_spectrum.parquet"), (stability_rows, "basis_stability.parquet"), (bound_rows, "action_bound_audit.parquet")):
        pq.write_table(pa.Table.from_pylist(rows), artifacts / name, compression="zstd")

    family_support = {}
    for task in TASKS:
        sr = [r for r in support_rows if r["task"] == task and r["chunk_length"] in (5, 10)]
        br = [r for r in bound_rows if r["task"] == task and r["chunk_length"] in (5, 10) and r["amplitude"] == .10]
        st = [r for r in stability_rows if r["task"] == task and r["chunk_length"] in (5, 10)]
        analytic = any(r["demo_count"] >= 8 and r["nontrivial_channel_count"] >= 3 for r in sr) and np.mean([r["clipping_fraction"] for r in br]) <= .10
        residual = any(r["demo_count"] >= 8 and r["residual_rank"] >= 3 and next(x["leave_demo_out_subspace_similarity"] for x in st if x["phase"] == r["phase"] and x["chunk_length"] == r["chunk_length"]) >= .5 for r in sr)
        phase_edit = any(r["demo_count"] >= 8 and r["chunk_count"] >= 30 for r in sr)
        gripper_count = max(r["gripper_transition_demo_count"] for r in sr); gripper = gripper_count >= 8
        family_support[task] = {"I-A_analytic": bool(analytic), "I-B_residual": bool(residual), "I-C_phase_edit": bool(phase_edit), "I-D_taskspace": False, "I-E_gripper_timing": bool(gripper), "gripper_transition_demo_count": gripper_count}
    stage0 = {"support_rule": {"minimum_demos": 8, "minimum_nontrivial_modes": 3, "residual_subspace_similarity": .5, "maximum_reference_clipping_at_alpha_0.10": .10}, "family_task_support": family_support, "supported_families": [family for family in ("I-A_analytic", "I-B_residual", "I-C_phase_edit", "I-E_gripper_timing") if any(family_support[t][family] for t in TASKS)], "stage1_authorized": any(any(v for k, v in row.items() if k.startswith("I-")) for row in family_support.values()), "taskspace_unavailable_not_improvised": True, "stage2_full_confirmation_availability": False, "stage2_if_reached": "availability-limited pilot"}
    (artifacts / "stage0_family_support.json").write_text(json.dumps(stage0, indent=2), encoding="utf-8"); (manifests / "stage0_support_audit.json").write_text(json.dumps(stage0, indent=2), encoding="utf-8")

    # Stage-0 plots.
    fig, ax = plt.subplots(figsize=(9, 5))
    for task in TASKS:
        rows = [r for r in spectrum_rows if r["task"] == task and r["phase"] == "P3" and r["chunk_length"] == 10]
        ax.plot([r["singular_index"] for r in rows], [r["relative_singular_value"] for r in rows], marker="o", label=task)
    ax.set(yscale="log", title="P3 K10 residual spectrum", xlabel="index", ylabel="relative singular value"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(plots / "action_residual_spectrum.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for ax, (name, maker) in zip(axes, (("DCT", dct_modes), ("Spline", spline_modes), ("Pulse", smooth_pulse_modes))):
        for mode in maker(20, 3): ax.plot(mode); ax.set_ylabel(name)
    axes[-1].set_xlabel("chunk step"); fig.tight_layout(); fig.savefig(plots / "temporal_basis_examples.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5)); labels = [f"{r['task'][:6]}-{r['phase']}-K{r['chunk_length']}" for r in stability_rows]; ax.scatter(range(len(labels)), [r["leave_demo_out_subspace_similarity"] for r in stability_rows]); ax.axhline(.5, linestyle="--"); ax.set(title="Leave-demo-out top-3 basis stability", ylabel="subspace similarity", xlabel="task-phase-length cell"); fig.tight_layout(); fig.savefig(plots / "basis_stability_by_phase.png", dpi=150); plt.close(fig)
    metrics = {"status": "completed", "run_id": args.run_id, "stage": "EXP11 Stage 0", **stage0, "support_rows": len(support_rows), "spectrum_rows": len(spectrum_rows), "bound_rows": len(bound_rows)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8"); (out / "config.json").write_text(json.dumps({"command": " ".join(sys.argv), "created_utc": now}, indent=2), encoding="utf-8"); (out / "stdout.log").write_text(json.dumps(metrics, indent=2), encoding="utf-8"); (out / "stderr.log").write_text("", encoding="utf-8"); print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
