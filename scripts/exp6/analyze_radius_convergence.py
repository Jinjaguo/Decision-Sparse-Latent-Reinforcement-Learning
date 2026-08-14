#!/usr/bin/env python
"""Run the frozen CPU source-of-truth EXP6 convergence, trust, and contact analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.metrics.exp6 import antithetic_asymmetry, projector, projector_similarity, relative_discrepancy, signal_to_floor, trust_region_passes  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402

TASKS = ["open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_plate", "turn_on_the_stove"]
SHORT = {TASKS[0]: "Drawer", TASKS[1]: "Bowl", TASKS[2]: "Stove"}
CONTACT_TOKEN = {TASKS[0]: "wooden_cabinet_1", TASKS[1]: "akita_black_bowl_1", TASKS[2]: "flat_stove_1"}
ORDER = ["r0003125", "r000625", "r00125", "r0025", "r005"]
RADIUS = {"r0003125": 0.0003125, "r000625": 0.000625, "r00125": 0.00125, "r0025": 0.0025, "r005": 0.005}


def write_parquet(rows: list[dict], path: Path) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rho(left, right) -> float:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    value = spearmanr(left, right).statistic
    return 0.0 if not np.isfinite(value) else float(value)


def bh(values: list[float]) -> list[float]:
    order = np.argsort(values); output = np.empty(len(values)); running = 1.0
    for reverse_index in range(len(values) - 1, -1, -1):
        index = order[reverse_index]; running = min(running, values[index] * len(values) / (reverse_index + 1)); output[index] = running
    return output.tolist()


def task_contact_set(raw: str, task: str) -> set[str]:
    token = CONTACT_TOKEN[task]; result = set()
    for pair in json.loads(raw):
        if token in pair and "gripper0_" in pair:
            result.add(pair)
    return result


def boxplot(frame: pd.DataFrame, column: str, group: str, output: Path, title: str) -> None:
    labels = list(dict.fromkeys(frame[group].tolist())); values = [frame.loc[frame[group] == label, column].dropna().values for label in labels]
    fig, axis = plt.subplots(figsize=(8, 5)); axis.boxplot(values, labels=labels, showfliers=False); axis.set_title(title); axis.set_ylabel(column); fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-id", required=True); parser.add_argument("--raw-run", type=Path, required=True); parser.add_argument("--calibration-run", type=Path, required=True); parser.add_argument("--gpu-run", type=Path, required=True); parser.add_argument("--manifest-dir", type=Path, required=True); parser.add_argument("--run-root", type=Path, default=ROOT / "runs"); args = parser.parse_args()
    run = create_run_directory(args.run_root, args.run_id); artifacts = run / "artifacts"; plots = run / "plots"; plots.mkdir(); started = time.perf_counter(); raw = args.raw_run.resolve(); manifests = args.manifest_dir.resolve()
    raw_metrics = json.loads((raw / "metrics.json").read_text()); calibration = json.loads((args.calibration_run / "artifacts/resolution_gate.json").read_text()); gpu = json.loads((args.gpu_run / "artifacts/gpu_audit.json").read_text())
    if not raw_metrics["gate"]["passed"] or not calibration["primary_smallest_resolvable"]:
        raise RuntimeError("formal raw or numerical calibration gate failed")
    if gpu["passed"]:
        raise RuntimeError("this source-of-truth run is preregistered CPU because the frozen GPU equivalence audit failed")
    rows = pq.read_table(raw / "artifacts/interventions.parquet").to_pylist(); zero_floor = max(calibration["zero_floor"][key] for key in ("integration_state", "signed_physical_output", "scalar_effect", "operator_spectral")); grouped = {}
    for row in rows:
        grouped.setdefault((row["task"], row["episode"], int(row["branch_time"]), row["radius_label"]), []).append(row)
    operator_rows, heldout_rows, matrices = [], [], []
    for (task, episode, branch_time, label), group in sorted(grouped.items()):
        lookup = {(int(row["direction_index"]), int(row["sign"])): row for row in group}; basis = np.stack([np.asarray(lookup[(direction, 1)]["unit_direction_scaled_coordinates"]) for direction in range(7)], axis=1); radius = RADIUS[label]; columns, asymmetries, responses = [], [], []
        for direction in range(7):
            plus = np.asarray(lookup[(direction, 1)]["signed_output_remaining_horizon_mean"]); minus = np.asarray(lookup[(direction, -1)]["signed_output_remaining_horizon_mean"]); columns.append((plus - minus) / (2.0 * radius)); asymmetries.append(antithetic_asymmetry(plus, minus)); responses.append(float(np.linalg.norm((plus - minus) / 2.0)))
        operator = np.stack(columns, axis=1); gram = operator.T @ operator; eigenvalues, right = np.linalg.eigh(gram); order = np.argsort(eigenvalues)[::-1]; eigenvalues = np.maximum(eigenvalues[order], 0.0); right = right[:, order]; singular = np.sqrt(eigenvalues); physical = basis @ right; total = float(eigenvalues.sum()); probability = eigenvalues / total if total else np.zeros_like(eigenvalues); effective_rank = float(np.exp(-np.sum(probability[probability > 0] * np.log(probability[probability > 0])))) if total else 0.0
        held_plus = np.asarray(lookup[(7, 1)]["signed_output_remaining_horizon_mean"]); held_minus = np.asarray(lookup[(7, -1)]["signed_output_remaining_horizon_mean"]); actual = (held_plus - held_minus) / (2.0 * radius); unit = np.asarray(lookup[(7, 1)]["unit_direction_scaled_coordinates"]); predicted = operator @ (basis.T @ unit); error = float(np.linalg.norm(predicted - actual) / (np.linalg.norm(actual) + 1e-12)); raw_response = min(responses + [float(np.linalg.norm((held_plus - held_minus) / 2.0))]); base = {"task": task, "episode": episode, "branch_time": branch_time, "radius_label": label, "radius_fraction": radius, "reference_contact_state": bool(group[0]["reference_contact_state"]), "reference_gripper_state": group[0]["reference_gripper_state"], "reference_predicate_state": bool(group[0]["reference_predicate_state"]), "reference_stratum": group[0]["reference_stratum"]}
        operator_rows.append({**base, "spectral_norm": float(singular[0]), "frobenius_norm": float(np.linalg.norm(operator)), "leading_eigenvalue_share": float(probability[0]), "effective_rank": effective_rank, "singular_values": singular.tolist(), "top1_physical_direction": physical[:, 0].tolist(), "top2_physical_directions": physical[:, :2].reshape(-1).tolist(), "mean_sign_asymmetry": float(np.mean(asymmetries)), "maximum_sign_asymmetry": float(np.max(asymmetries)), "minimum_antithetic_response_norm": raw_response, "signal_to_floor": signal_to_floor(raw_response, zero_floor)})
        actual_scalar = float((lookup[(7, 1)]["primary_remaining_horizon_mean"] + lookup[(7, -1)]["primary_remaining_horizon_mean"]) / 2.0); heldout_rows.append({**base, "actual_response_norm": float(np.linalg.norm(actual)), "predicted_response_norm": float(np.linalg.norm(predicted)), "vector_relative_error": error, "actual_scalar_effect": actual_scalar, "predicted_scalar_effect": float(np.linalg.norm(predicted)), "sign_asymmetry": antithetic_asymmetry(held_plus, held_minus)})
        matrices.append((task, episode, branch_time, label, operator, gram))
    op = pd.DataFrame(operator_rows); held = pd.DataFrame(heldout_rows); write_parquet(operator_rows, artifacts / "radius_operator_summary.parquet"); write_parquet(heldout_rows, artifacts / "heldout_direction_prediction.parquet"); write_parquet([{key: row[key] for key in ("task", "episode", "branch_time", "radius_label", "radius_fraction", "mean_sign_asymmetry", "maximum_sign_asymmetry")} for row in operator_rows], artifacts / "sign_asymmetry_by_radius.parquet"); write_parquet([{key: row[key] for key in ("task", "episode", "branch_time", "radius_label", "radius_fraction", "minimum_antithetic_response_norm", "signal_to_floor")} for row in operator_rows], artifacts / "signal_to_floor.parquet")
    max_dimension = max(value[4].shape[0] for value in matrices); array = np.full((len(matrices), max_dimension, 7), np.nan); grams = np.empty((len(matrices), 7, 7)); keys = []
    for index, value in enumerate(matrices):
        array[index, :value[4].shape[0]] = value[4]; grams[index] = value[5]; keys.append("|".join(map(str, value[:4])))
    np.savez_compressed(artifacts / "operator_matrices_by_radius.npz", operators=array, grams=grams, keys=np.asarray(keys))
    opmap = {(row.task, row.episode, row.branch_time, row.radius_label): row for row in op.itertuples()}; heldmap = {(row.task, row.episode, row.branch_time, row.radius_label): row for row in held.itertuples()}; adjacent = []
    for task, episode, branch_time in sorted({key[:3] for key in opmap}):
        for lower, upper in zip(ORDER[:-1], ORDER[1:]):
            left, right = opmap[(task, episode, branch_time, lower)], opmap[(task, episode, branch_time, upper)]; lp1 = projector(np.asarray(left.top1_physical_direction)[:, None], 1); rp1 = projector(np.asarray(right.top1_physical_direction)[:, None], 1); lp2 = projector(np.asarray(left.top2_physical_directions).reshape(7, 2), 2); rp2 = projector(np.asarray(right.top2_physical_directions).reshape(7, 2), 2); top1 = projector_similarity(lp1, rp1, 1); top2 = projector_similarity(lp2, rp2, 2); spectral = relative_discrepancy(left.spectral_norm, right.spectral_norm); frobenius = relative_discrepancy(left.frobenius_norm, right.frobenius_norm); asymmetry = max(left.maximum_sign_asymmetry, right.maximum_sign_asymmetry); held_error = max(heldmap[(task, episode, branch_time, lower)].vector_relative_error, heldmap[(task, episode, branch_time, upper)].vector_relative_error); floor_ratio = min(left.signal_to_floor, right.signal_to_floor); passed, failed = trust_region_passes(top1, top2, spectral, asymmetry, held_error, floor_ratio)
            adjacent.append({"task": task, "episode": episode, "branch_time": branch_time, "lower_radius_label": lower, "upper_radius_label": upper, "lower_radius": RADIUS[lower], "upper_radius": RADIUS[upper], "comparison": f"{lower}_vs_{upper}", "top1_similarity": top1, "top2_similarity": top2, "relative_spectral_discrepancy": spectral, "relative_frobenius_discrepancy": frobenius, "maximum_sign_asymmetry": asymmetry, "maximum_heldout_vector_error": held_error, "minimum_signal_to_floor": floor_ratio, "convergence_passed": passed, "failed_criteria": failed})
    adj = pd.DataFrame(adjacent); write_parquet(adjacent, artifacts / "adjacent_radius_comparisons.parquet")
    trust = []
    for (task, episode, branch_time), group in adj.groupby(["task", "episode", "branch_time"]):
        passed = group[group.convergence_passed].sort_values("upper_radius"); winner = None if passed.empty else passed.iloc[-1]
        branch_meta = opmap[(task, episode, branch_time, ORDER[0])]; trust.append({"task": task, "episode": episode, "branch_time": int(branch_time), "reference_contact_state": bool(branch_meta.reference_contact_state), "reference_gripper_state": branch_meta.reference_gripper_state, "reference_predicate_state": bool(branch_meta.reference_predicate_state), "smallest_resolvable_radius": RADIUS[ORDER[0]], "largest_convergent_radius": None if winner is None else float(winner.upper_radius), "largest_convergent_interval": None if winner is None else f"{winner.lower_radius_label}_to_{winner.upper_radius_label}", "trust_region_status": "unresolved" if winner is None else "resolved", "failed_criteria": sorted(set(item for values in group.failed_criteria for item in values)), "minimum_signal_to_floor": float(group.minimum_signal_to_floor.min())})
    trust_frame = pd.DataFrame(trust); write_parquet(trust, artifacts / "trust_region_summary.parquet"); aggregate = []
    for group_name, group_column in (("task", "task"), ("contact", "reference_contact_state"), ("gripper", "reference_gripper_state"), ("predicate", "reference_predicate_state")):
        for value, frame in trust_frame.groupby(group_column):
            aggregate.append({"group_type": group_name, "group_value": str(value), "branches": len(frame), "resolved_fraction": float((frame.trust_region_status == "resolved").mean()), "median_largest_radius": None if frame.largest_convergent_radius.dropna().empty else float(frame.largest_convergent_radius.dropna().median())})
    write_parquet(aggregate, artifacts / "trust_region_aggregates.parquet")
    # Exact task-relevant contact modes over the first ten continuation steps.
    columns = ["task", "episode", "branch_time", "radius_label", "direction_index", "sign", "continuation_offset", "perturbed_contact_pairs_json"]; effect = pq.read_table(raw / "artifacts/per_step_effects.parquet", columns=columns).to_pandas(); effect = effect[effect.continuation_offset < 10]; contact_rows = []
    for (task, episode, branch_time, label, direction), frame in effect.groupby(["task", "episode", "branch_time", "radius_label", "direction_index"]):
        plus = {int(row.continuation_offset): task_contact_set(row.perturbed_contact_pairs_json, task) for row in frame[frame.sign == 1].itertuples()}; minus = {int(row.continuation_offset): task_contact_set(row.perturbed_contact_pairs_json, task) for row in frame[frame.sign == -1].itertuples()}; offsets = sorted(set(plus) & set(minus)); divergent = [offset + 1 for offset in offsets if plus[offset] != minus[offset]]
        def through(step: int) -> bool:
            return all(plus[offset] == minus[offset] for offset in offsets if offset < step)
        changes = [{"step": offset + 1, "plus": sorted(plus[offset]), "minus": sorted(minus[offset])} for offset in offsets if plus[offset] != minus[offset]]
        contact_rows.append({"task": task, "episode": episode, "branch_time": int(branch_time), "radius_label": label, "direction_index": int(direction), "same_contact_mode_step_1": through(1), "same_contact_mode_through_3": through(3), "same_contact_mode_through_5": through(5), "same_contact_mode_through_10": through(10), "first_contact_mode_divergence_step": None if not divergent else min(divergent), "contact_mode_diverged": bool(divergent), "exact_pair_set_changes_json": json.dumps(changes, sort_keys=True)})
    write_parquet(contact_rows, artifacts / "contact_mode_transitions.parquet"); contact = pd.DataFrame(contact_rows); branch_contact = contact.groupby(["task", "episode", "branch_time", "radius_label"]).contact_mode_diverged.any().reset_index(); target = adj[(adj.lower_radius_label == "r000625") & (adj.upper_radius_label == "r00125")].copy(); diverged = branch_contact[branch_contact.radius_label.isin(["r000625", "r00125"])].groupby(["task", "episode", "branch_time"]).contact_mode_diverged.any().reset_index(); conditioned = target.merge(diverged, on=["task", "episode", "branch_time"]); conditioned["convergence_failed"] = ~conditioned.convergence_passed
    write_parquet(conditioned.to_dict("records"), artifacts / "contact_conditioned_convergence.parquet"); demo_differences = []
    for (task, episode), frame in conditioned.groupby(["task", "episode"]):
        yes = frame[frame.contact_mode_diverged].convergence_failed; no = frame[~frame.contact_mode_diverged].convergence_failed
        if len(yes) and len(no): demo_differences.append({"task": task, "episode": episode, "failure_rate_diverged": float(yes.mean()), "failure_rate_preserved": float(no.mean()), "difference": float(yes.mean() - no.mean())})
    write_parquet(demo_differences, artifacts / "contact_demo_cluster_effects.parquet")
    # Frozen hypothesis inference.
    small = target.copy(); demo_top1 = small.groupby(["task", "episode"]).top1_similarity.median().reset_index(); h1_fraction = float((demo_top1.top1_similarity >= 0.80).mean()); rng = np.random.Generator(np.random.PCG64(960031)); boot = []
    for _ in range(4000):
        task_samples = rng.choice(TASKS, len(TASKS), replace=True); values = []
        for task in task_samples:
            pool = demo_top1[demo_top1.task == task].top1_similarity.values; values.extend(rng.choice(pool, len(pool), replace=True))
        boot.append(float(np.median(values)))
    h1_ci = np.quantile(boot, [0.025, 0.975]).tolist(); h2_fraction = float((small.relative_spectral_discrepancy <= 0.20).mean()); smallest = held[held.radius_label == ORDER[0]]; demo_held = []
    for (task, episode), frame in smallest.groupby(["task", "episode"]): demo_held.append({"task": task, "episode": episode, "rho": rho(frame.actual_response_norm.values, frame.predicted_response_norm.values), "median_vector_error": float(frame.vector_relative_error.median())})
    held_demo = pd.DataFrame(demo_held); write_parquet(demo_held, artifacts / "heldout_demo_summary.parquet"); h3_rho = float(held_demo.rho.median()); h3_error = float(smallest.vector_relative_error.median())
    differences = np.asarray([row["difference"] for row in demo_differences]); observed = float(np.mean(differences)) if len(differences) else 0.0; contact_boot = []
    for _ in range(4000):
        contact_boot.append(float(np.mean(rng.choice(differences, len(differences), replace=True))) if len(differences) else 0.0)
    contact_ci = np.quantile(contact_boot, [0.025, 0.975]).tolist(); permutation_rng = np.random.Generator(np.random.PCG64(960032)); permutations = [float(np.mean(differences * permutation_rng.choice([-1, 1], len(differences)))) if len(differences) else 0.0 for _ in range(4000)]; p_value = float((1 + sum(value >= observed for value in permutations)) / 4001); q_value = bh([p_value])[0]
    hypotheses = {"H1": {"passed": h1_fraction >= 0.70 and h1_ci[0] > 0.65, "demo_fraction_median_top1_ge_0_80": h1_fraction, "hierarchical_median_95_ci": h1_ci}, "H2": {"passed": h2_fraction >= 0.70, "branch_fraction_spectral_discrepancy_le_0_20": h2_fraction}, "H3": {"passed": h3_rho >= 0.65 and h3_error <= 0.35, "demo_median_rank_rho": h3_rho, "median_vector_relative_error": h3_error}, "H4": {"passed": observed > 0 and contact_ci[0] > 0 and q_value < 0.05, "demo_cluster_failure_rate_difference": observed, "bootstrap_95_ci": contact_ci, "permutation_p": p_value, "bh_q": q_value, "eligible_demo_clusters": len(differences)}}
    if hypotheses["H1"]["passed"] and hypotheses["H2"]["passed"] and hypotheses["H3"]["passed"]: classification = "small_radius_local_operator_converges"
    elif hypotheses["H4"]["passed"]: classification = "contact_mode_conditioned_convergence"
    elif not calibration["primary_smallest_resolvable"]: classification = "numerical_resolution_prevents_local_limit_test"
    elif not hypotheses["H1"]["passed"] or not hypotheses["H2"]["passed"]: classification = "nonsmooth_response_persists_below_exp5_radius"
    else: classification = "no_support"
    decision = {"schema_version": 1, "classification": classification, "classification_priority_applied": True, "hypotheses": hypotheses, "trust_region": {"resolved_branches": int((trust_frame.trust_region_status == "resolved").sum()), "total_branches": len(trust_frame), "resolved_fraction": float((trust_frame.trust_region_status == "resolved").mean())}, "numerical_resolution": calibration, "gpu_formal_used": False, "gpu_reason": "frozen absolute GPU/CPU equivalence gate failed; CPU retained as source of truth", "scheduler_eligible": False, "latent_rl_eligible": False}
    write_json(artifacts / "hypothesis_tests.json", hypotheses); write_json(artifacts / "scientific_decision.json", decision); write_json(artifacts / "zero_floor.json", calibration["zero_floor"])
    # Required raw aliases and immutable evidence copies.
    shutil.copy2(raw / "artifacts/interventions.parquet", artifacts / "radius_interventions.parquet"); shutil.copy2(raw / "artifacts/per_step_effects.parquet", artifacts / "per_step_effects.parquet"); shutil.copy2(raw / "artifacts/zero_controls.parquet", artifacts / "zero_controls.parquet"); shutil.copy2(raw / "artifacts/raw_hash_manifest.json", artifacts / "raw_hash_manifest.json"); shutil.copy2(raw / "artifacts/failure_examples.json", artifacts / "failure_examples.json"); shutil.copy2(args.calibration_run / "artifacts/zero_repeatability.parquet", artifacts / "zero_repeatability.parquet"); shutil.copy2(args.gpu_run / "artifacts/gpu_audit.json", artifacts / "gpu_audit.json"); shutil.copy2(args.gpu_run / "artifacts/gpu_cpu_equivalence.json", artifacts / "gpu_cpu_equivalence.json"); write_parquet([{key: row[key] for key in ("task", "episode", "branch_time", "radius_label", "direction_index", "sign", "signed_output_remaining_horizon_mean")} for row in rows], artifacts / "signed_output_vectors.parquet")
    # Sixteen preregistered non-placeholder plots.
    radius_labels = ORDER; radius_positions = np.arange(len(radius_labels));
    boxplot(op, "signal_to_floor", "radius_label", plots / "signal_to_zero_floor_by_radius.png", "Signal above matched-zero floor by radius")
    repeat = pq.read_table(args.calibration_run / "artifacts/zero_repeatability.parquet").to_pandas(); repeat["repeat_error"] = repeat[["scalar_max_abs", "vector_max_abs"]].max(axis=1); boxplot(repeat, "repeat_error", "radius_label", plots / "repeatability_by_radius.png", "Calibration repeatability by radius")
    boxplot(op, "spectral_norm", "radius_label", plots / "spectral_norm_vs_radius.png", "Finite-radius spectral norm")
    boxplot(adj, "top1_similarity", "comparison", plots / "top1_similarity_adjacent_radii.png", "Adjacent-radius top-1 similarity"); boxplot(adj, "top2_similarity", "comparison", plots / "top2_similarity_adjacent_radii.png", "Adjacent-radius top-2 similarity"); boxplot(adj, "relative_spectral_discrepancy", "comparison", plots / "spectral_discrepancy_adjacent_radii.png", "Adjacent-radius spectral discrepancy")
    boxplot(op, "maximum_sign_asymmetry", "radius_label", plots / "sign_asymmetry_vs_radius.png", "Antithetic asymmetry by radius"); boxplot(held, "vector_relative_error", "radius_label", plots / "heldout_prediction_error_vs_radius.png", "Held-out vector error by radius")
    fig, axis = plt.subplots(figsize=(8, 5)); resolved = trust_frame.largest_convergent_radius.dropna(); axis.hist(resolved, bins=np.asarray([0.0003, 0.000625, 0.00125, 0.0025, 0.0055])); axis.set(title="Largest resolved trust-region radius", xlabel="radius", ylabel="branches"); fig.tight_layout(); fig.savefig(plots / "trust_region_radius_distribution.png", dpi=160); plt.close(fig)
    task_plot = pd.DataFrame(aggregate); boxplot(trust_frame.assign(task_short=trust_frame.task.map(SHORT), resolved=(trust_frame.trust_region_status == "resolved").astype(float)), "resolved", "task_short", plots / "trust_region_by_task.png", "Resolved trust regions by task"); boxplot(trust_frame.assign(contact=trust_frame.reference_contact_state.astype(str), resolved=(trust_frame.trust_region_status == "resolved").astype(float)), "resolved", "contact", plots / "trust_region_by_contact_state.png", "Resolved trust regions by reference contact")
    contact_plot = conditioned.groupby("contact_mode_diverged").convergence_failed.mean().reset_index(); fig, axis = plt.subplots(figsize=(6, 5)); axis.bar(contact_plot.contact_mode_diverged.astype(str), contact_plot.convergence_failed); axis.set(title="Contact divergence vs convergence failure", ylabel="failure rate"); fig.tight_layout(); fig.savefig(plots / "contact_mode_divergence_vs_convergence.png", dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 5)); timing = contact.first_contact_mode_divergence_step.dropna(); axis.hist(timing, bins=np.arange(0.5, 11.5, 1)); axis.set(title="First contact-mode divergence", xlabel="future policy step"); fig.tight_layout(); fig.savefig(plots / "contact_mode_divergence_timing.png", dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5)); [axis.plot(np.arange(1, 8), row.singular_values, alpha=0.06, color=f"C{ORDER.index(row.radius_label)}") for row in op.itertuples()]; axis.set_yscale("log"); axis.set(title="Operator spectrum across radii", xlabel="singular component"); fig.tight_layout(); fig.savefig(plots / "operator_spectrum_vs_radius.png", dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 5)); axis.bar(np.arange(len(demo_top1)), demo_top1.top1_similarity); axis.axhline(0.80, color="red", linestyle="--"); axis.set(title="Demo-level 0.000625–0.00125 top-1 convergence", ylabel="median similarity"); fig.tight_layout(); fig.savefig(plots / "demo_level_small_radius_convergence.png", dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4)); values = [max(row["operator_max_abs"] for row in gpu["records"]), max(row["gram_max_abs"] for row in gpu["records"]), max(row["eigenvalue_max_abs"] for row in gpu["records"]), max(max(row["top1_projector_max_abs"], row["top2_projector_max_abs"]) for row in gpu["records"])]; axis.bar(["operator", "Gram", "eigen", "projector"], np.maximum(values, 1e-18)); axis.set_yscale("log"); axis.set_title("Failed frozen GPU/CPU absolute-equivalence gate"); fig.tight_layout(); fig.savefig(plots / "gpu_cpu_equivalence.png", dpi=160); plt.close(fig)
    analysis_hashes = {path.name: sha(path) for path in artifacts.iterdir() if path.is_file()}; write_json(artifacts / "analysis_hashes.json", analysis_hashes)
    metrics = {"run_id": args.run_id, "status": "completed", "gate": {"passed": True}, "analysis_device": "CPU source of truth", "operator_rows": len(operator_rows), "adjacent_comparisons": len(adjacent), "contact_pairs": len(contact_rows), "trust_regions_resolved": decision["trust_region"]["resolved_branches"], "classification": classification, "hypotheses": hypotheses, "plot_count": len(list(plots.glob("*.png"))), "wall_time_seconds": time.perf_counter() - started}
    write_run_record(run, config={"stage": "EXP6 formal convergence/trust/contact analysis", "raw_run": raw.name, "gpu_run": args.gpu_run.name}, command=shlex.join([sys.executable, *sys.argv]), environment={"python": sys.version, "numpy": np.__version__, "analysis_device": "CPU"}, git_state={"project": git_record(ROOT)}, metrics=metrics, stdout=json.dumps(metrics), stderr=""); print(json.dumps(metrics, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
