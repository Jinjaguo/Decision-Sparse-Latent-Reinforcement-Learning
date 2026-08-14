#!/usr/bin/env python
"""Run the frozen EXP8 cross-demo, ablation, horizon, and selective-risk analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from decision_sparse_rl.metrics.exp8 import binary_operating_metrics, expected_calibration_error, select_risk_threshold, top1_projector_similarity, upper_tail


def write(rows, path):
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def float_stack(values):
    """Materialize Arrow/Pandas nested values as a dense float64 matrix."""
    return np.stack([np.asarray(value, dtype=np.float64) for value in values])


def features(row, groups):
    values = json.loads(row.feature_groups_json)
    return np.concatenate([np.asarray(values[group], dtype=np.float64) for group in groups])


def scale(train, test):
    center = np.median(train, axis=0)
    robust = 1.4826 * np.median(np.abs(train - center), axis=0)
    fallback = train.std(0)
    robust[robust < 1e-12] = fallback[robust < 1e-12]
    robust[robust < 1e-12] = 1.0
    return (train - center) / robust, (test - center) / robust


def rbf(left, right, bandwidth):
    squared = np.mean((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
    return np.exp(-squared / (2 * bandwidth * bandwidth))


def krr(train_x, train_y, test_x, alpha, bandwidth):
    a, b = scale(train_x, test_x)
    kernel = rbf(a, a, bandwidth)
    return rbf(b, a, bandwidth) @ np.linalg.solve(kernel + alpha * np.eye(len(kernel)), train_y)


def ridge(train_x, train_y, test_x, alpha):
    a, b = scale(train_x, test_x)
    a = np.column_stack((np.ones(len(a)), a)); b = np.column_stack((np.ones(len(b)), b))
    penalty = np.eye(a.shape[1]); penalty[0, 0] = 0
    return b @ np.linalg.solve(a.T @ a + alpha * penalty, a.T @ train_y)


def top1_from_operator(operator):
    _, _, right = np.linalg.svd(np.asarray(operator), full_matrices=False)
    return right[0]


def operator_similarity(predicted, actual):
    return top1_projector_similarity(top1_from_operator(predicted), top1_from_operator(actual))


def auc(y, probability):
    y, probability = np.asarray(y, int), np.asarray(probability, float)
    positive, negative = y == 1, y == 0
    if not positive.any() or not negative.any(): return None
    return float((rankdata(probability)[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * negative.sum()))


def auprc(y, probability):
    y, probability = np.asarray(y, int), np.asarray(probability, float)
    order = np.argsort(-probability); labels = y[order]; tp = np.cumsum(labels); precision = tp / np.arange(1, len(y) + 1); recall = tp / max(labels.sum(), 1)
    return float(np.sum(precision * (recall - np.r_[0, recall[:-1]])))


def clustered_ci(frame, column, rng, resamples=4000):
    demos = frame[["task", "episode"]].drop_duplicates().to_records(index=False)
    by_task = {task: [episode for t, episode in demos if t == task] for task in sorted(frame.task.unique())}
    values = []
    for _ in range(resamples):
        means = []
        for task, episodes in by_task.items():
            sampled = rng.choice(episodes, len(episodes), replace=True)
            means.extend(float(frame[(frame.task == task) & (frame.episode == episode)][column].mean()) for episode in sampled)
        values.append(np.mean(means))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def task_ci(frame, column, rng, resamples=4000):
    result = {}
    for task, group in frame.groupby("task"):
        demo = group.groupby("episode")[column].mean().to_numpy()
        boot = np.mean(rng.choice(demo, (resamples, len(demo)), replace=True), axis=1)
        result[task] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    return result


def choose_parameters(train, groups, alpha_grid, bandwidth_grid, model):
    candidates = [(alpha, bandwidth) for alpha in alpha_grid for bandwidth in (bandwidth_grid if model == "krr" else [None])]
    scored = []
    for alpha, bandwidth in candidates:
        scores = []
        for fold in sorted(train.fold.unique()):
            inner_train, inner_test = train[train.fold != fold], train[train.fold == fold]
            if inner_train.empty or inner_test.empty: continue
            xtrain = float_stack(features(row, groups) if model == "krr" else row.baseline_b_features for row in inner_train.itertuples())
            xtest = float_stack(features(row, groups) if model == "krr" else row.baseline_b_features for row in inner_test.itertuples())
            ytrain = float_stack(inner_train.operator_flat)
            predicted = krr(xtrain, ytrain, xtest, alpha, bandwidth) if model == "krr" else ridge(xtrain, ytrain, xtest, alpha)
            for prediction, actual in zip(predicted, inner_test.operator_matrix):
                scores.append(operator_similarity(prediction.reshape(np.asarray(actual).shape), actual))
        scored.append((float(np.mean(scores)) if scores else -1, alpha, bandwidth))
    return max(scored, key=lambda row: (row[0], row[1], -1 if row[2] is None else row[2]))[1:]


def crossfit_operators(data, groups, grids, fixed_parameters=None, conditional=True):
    population = data[data.all_basis_both_signs_preserved].copy() if conditional else data.copy()
    results, parameters = [], {}
    for task, task_data in population.groupby("task"):
        for fold in range(5):
            train, test = task_data[task_data.fold != fold], task_data[task_data.fold == fold]
            if train.empty or test.empty: continue
            key = (task, fold)
            if fixed_parameters is None:
                primary_alpha, bandwidth = choose_parameters(train, groups, grids["ridge_alpha"], grids["rbf_bandwidth"], "krr")
                baseline_alpha, _ = choose_parameters(train, groups, grids["ridge_alpha"], grids["rbf_bandwidth"], "ridge")
            else:
                primary_alpha, bandwidth, baseline_alpha = fixed_parameters[key]
            parameters[key] = (primary_alpha, bandwidth, baseline_alpha)
            primary_train = float_stack(features(row, groups) for row in train.itertuples()); primary_test = float_stack(features(row, groups) for row in test.itertuples())
            baseline_train = float_stack(train.baseline_b_features); baseline_test = float_stack(test.baseline_b_features)
            targets = float_stack(train.operator_flat)
            primary_prediction = krr(primary_train, targets, primary_test, primary_alpha, bandwidth)
            baseline_prediction = ridge(baseline_train, targets, baseline_test, baseline_alpha)
            for index, row in enumerate(test.itertuples()):
                actual = np.asarray(row.operator_matrix, dtype=np.float64); shape = actual.shape
                primary_operator = primary_prediction[index].reshape(shape); baseline_operator = baseline_prediction[index].reshape(shape)
                pool = train
                time_row = pool.iloc[np.argmin(np.abs(pool.normalized_time.to_numpy() - row.normalized_time))]
                progress_row = pool.iloc[np.argmin(np.abs(pool.physical_progress_clipped.to_numpy() - row.physical_progress_clipped))]
                same = pool[(pool.exact_mode_json == row.exact_mode_json) & (pool.margin_class == row.margin_class)]
                if same.empty:
                    mode_similarity = 0.0
                else:
                    same_row = same.iloc[np.argmin(np.abs(same.normalized_time.to_numpy() - row.normalized_time))]
                    mode_similarity = operator_similarity(np.asarray(same_row.operator_matrix), actual)
                time_similarity = operator_similarity(np.asarray(time_row.operator_matrix), actual)
                progress_similarity = operator_similarity(np.asarray(progress_row.operator_matrix), actual)
                baseline_b_similarity = operator_similarity(baseline_operator, actual)
                primary_similarity = operator_similarity(primary_operator, actual)
                best = max(mode_similarity, time_similarity, progress_similarity, baseline_b_similarity)
                results.append({"task": row.task, "episode": row.episode, "branch_time": row.branch_time, "fold": fold, "horizon": row.horizon, "conditional": conditional, "mode_margin_top1": mode_similarity, "time_top1": time_similarity, "progress_top1": progress_similarity, "baseline_b_top1": baseline_b_similarity, "best_baseline_top1": best, "primary_top1": primary_similarity, "improvement_over_best": primary_similarity - best, "improvement_over_baseline_b": primary_similarity - baseline_b_similarity, "predicted_operator": primary_operator.tolist(), "predicted_baseline_b_operator": baseline_operator.tolist(), "actual_operator": actual.tolist(), "selected_alpha": primary_alpha, "selected_bandwidth": bandwidth})
    return pd.DataFrame(results), parameters


def logistic_fit(x, y, alpha):
    def objective(weights):
        z = np.clip(x @ weights, -40, 40)
        return float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 * alpha * np.sum(weights[1:] ** 2))
    return minimize(objective, np.zeros(x.shape[1]), method="L-BFGS-B", options={"maxiter": 180}).x


def logistic_crossfit(frame, alpha_grid):
    outputs = []
    for task, task_data in frame.groupby("task"):
        for fold in range(5):
            train, test = task_data[task_data.fold != fold].copy(), task_data[task_data.fold == fold].copy()
            xtrain = float_stack(train.risk_features); xtest = float_stack(test.risk_features); mean, std = xtrain.mean(0), xtrain.std(0); std[std < 1e-12] = 1
            xtrain = np.column_stack((np.ones(len(train)), (xtrain - mean) / std)); xtest = np.column_stack((np.ones(len(test)), (xtest - mean) / std)); ytrain = train.target.to_numpy(float)
            candidate_rows = []
            for alpha in alpha_grid:
                inner_probability = np.zeros(len(train))
                for inner_fold in sorted(train.fold.unique()):
                    fit_mask = train.fold.to_numpy() != inner_fold; validation = ~fit_mask
                    weights = logistic_fit(xtrain[fit_mask], ytrain[fit_mask], alpha)
                    inner_probability[validation] = 1 / (1 + np.exp(-np.clip(xtrain[validation] @ weights, -40, 40)))
                candidate_rows.append((auc(ytrain, inner_probability) or -1, alpha, inner_probability))
            _, alpha, inner_probability = max(candidate_rows, key=lambda row: (row[0], row[1]))
            threshold = select_risk_threshold(ytrain.astype(int), inner_probability, 0.85)
            weights = logistic_fit(xtrain, ytrain, alpha)
            probability = 1 / (1 + np.exp(-np.clip(xtest @ weights, -40, 40)))
            for row, score in zip(test.itertuples(), probability):
                outputs.append({"task": row.task, "episode": row.episode, "branch_time": row.branch_time, "radius_fraction": row.radius_fraction, "direction_index": row.direction_index, "sign": row.sign, "fold": fold, "target": int(row.target), "probability": float(score), "threshold": float(threshold), "selected_alpha": alpha})
    return pd.DataFrame(outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-run", type=Path, required=True); parser.add_argument("--assembly-run", type=Path, required=True)
    parser.add_argument("--feature-run", type=Path, required=True); parser.add_argument("--contact-frame-run", type=Path, required=True)
    parser.add_argument("--gpu-run", type=Path, required=True); parser.add_argument("--manifest-dir", type=Path, required=True); parser.add_argument("--output-run", type=Path, required=True)
    args = parser.parse_args(); out = args.output_run.resolve(); artifacts, plots = out / "artifacts", out / "plots"; artifacts.mkdir(parents=True, exist_ok=False); plots.mkdir()
    rng = np.random.default_rng(980031)
    raw_artifacts, assembly_artifacts, feature_artifacts = args.raw_run / "artifacts", args.assembly_run / "artifacts", args.feature_run / "artifacts"
    matrices = pq.read_table(assembly_artifacts / "operator_matrices.parquet").to_pandas(); operators = pq.read_table(assembly_artifacts / "horizon_operators.parquet").to_pandas(); modes = pq.read_table(assembly_artifacts / "mode_outcomes.parquet").to_pandas(); heldout = pq.read_table(assembly_artifacts / "heldout_local_predictions.parquet").to_pandas()
    branch_features = pq.read_table(feature_artifacts / "frozen_branch_contact_features.parquet").to_pandas(); action_features = pq.read_table(feature_artifacts / "frozen_action_projection_features.parquet").to_pandas()
    folds = pd.DataFrame(json.loads((args.manifest_dir / "crossfit_manifest.json").read_text(encoding="utf-8"))["assignments"])
    grids = json.loads((args.manifest_dir / "hyperparameter_grid.json").read_text(encoding="utf-8"))
    data = matrices.merge(operators, on=["task", "episode", "branch_time", "radius_fraction", "radius_label", "horizon"]).merge(branch_features, on=["task", "episode", "branch_time"]).merge(folds, on=["task", "episode"])
    data["operator_flat"] = data.operator_matrix.map(lambda value: np.asarray(value, dtype=np.float64).reshape(-1))
    groups = ["physical_group", "nearest_points", "normal_tangent_frame", "signed_gap", "relative_velocity", "contact_age", "force", "action_projection", "eef_object_relative_pose", "physical_state"]
    smallest = min(data.radius_fraction)
    primary_data = data[(data.horizon == "1") & (data.radius_fraction == smallest)].copy()
    cross, parameters = crossfit_operators(primary_data, groups, grids, conditional=True)
    demo = cross.groupby(["task", "episode"], as_index=False).agg(improvement=("improvement_over_best", "mean"), primary_top1=("primary_top1", "mean"), baseline_top1=("best_baseline_top1", "mean"))
    h1_ci = clustered_ci(demo, "improvement", rng); per_task = demo.groupby("task").improvement.mean().to_dict(); per_task_ci = task_ci(demo, "improvement", rng)
    h1 = bool(len(demo) == 30 and demo.improvement.mean() >= 0.15 and h1_ci[0] > 0 and all(value > 0 for value in per_task.values()) and sum(value[0] > 0 for value in per_task_ci.values()) >= 2)
    held_map = {(row.task, row.episode, row.branch_time): row for row in heldout[(heldout.horizon == "1") & (heldout.radius_fraction == smallest) & heldout.both_signs_preserved].itertuples()}
    held_rows = []
    for row in cross.itertuples():
        key = (row.task, row.episode, row.branch_time)
        if key not in held_map: continue
        actual_row = held_map[key]; predicted = np.asarray(row.predicted_operator, dtype=np.float64) @ np.asarray(actual_row.unit_direction, dtype=np.float64); actual = np.asarray(actual_row.actual_vector, dtype=np.float64)
        error = float(np.linalg.norm(predicted - actual) / (np.linalg.norm(actual) + 1e-12))
        held_rows.append({"task": row.task, "episode": row.episode, "branch_time": row.branch_time, "predicted_vector": predicted.tolist(), "actual_vector": actual.tolist(), "predicted_norm": float(np.linalg.norm(predicted)), "actual_norm": float(np.linalg.norm(actual)), "vector_relative_error": error})
    held_frame = pd.DataFrame(held_rows); demo_held = held_frame.groupby(["task", "episode"]).apply(lambda group: pd.Series({"rho": float(spearmanr(group.predicted_norm, group.actual_norm).statistic) if len(group) > 1 else 0.0, "median_error": float(group.vector_relative_error.median())})).reset_index(); tails = upper_tail(held_frame.vector_relative_error); heldout_by_task = {task: {"demo_median_rho": float(group.rho.median()), "demo_median_vector_error": float(group.median_error.median()), **upper_tail(held_frame.loc[held_frame.task == task, "vector_relative_error"])} for task, group in demo_held.groupby("task")}
    h2 = bool(len(demo_held) == 30 and demo_held.rho.median() >= 0.70 and demo_held.median_error.median() <= 0.35 and tails["p90"] <= 0.60)
    h3_demo = cross.groupby(["task", "episode"], as_index=False).improvement_over_baseline_b.mean(); h3_ci = clustered_ci(h3_demo, "improvement_over_baseline_b", rng); signs = rng.choice([-1, 1], size=(4000, len(h3_demo))); permutation = np.mean(signs * h3_demo.improvement_over_baseline_b.to_numpy(), axis=1); h3_p = float((np.sum(permutation >= h3_demo.improvement_over_baseline_b.mean()) + 1) / 4001)
    ablations = []
    remove_map = {"remove_nearest_points": "nearest_points", "remove_normal_tangent_frame": "normal_tangent_frame", "remove_relative_velocity": "relative_velocity", "remove_contact_age": "contact_age", "remove_force": "force", "remove_action_projection": "action_projection", "remove_EEF_object_relative_pose": "eef_object_relative_pose"}
    for name, removed in remove_map.items():
        result, _ = crossfit_operators(primary_data, [group for group in groups if group != removed], grids, fixed_parameters=parameters, conditional=True)
        demo_variant = result.groupby(["task", "episode"], as_index=False).improvement_over_baseline_b.mean()
        variant_signs = rng.choice([-1, 1], size=(4000, len(demo_variant)))
        variant_null = np.mean(variant_signs * demo_variant.improvement_over_baseline_b.to_numpy(), axis=1)
        variant_p = float((np.sum(variant_null >= demo_variant.improvement_over_baseline_b.mean()) + 1) / 4001)
        ablations.append({"ablation": name, "mean_top1": float(result.primary_top1.mean()), "mean_improvement_over_baseline_b": float(result.improvement_over_baseline_b.mean()), "improvement_CI": clustered_ci(demo_variant, "improvement_over_baseline_b", rng), "permutation_p": variant_p, "delta_from_full": float(result.primary_top1.mean() - cross.primary_top1.mean())})
    ablations.extend([{"ablation": "exact_mode_margin_only", "mean_top1": float(cross.mode_margin_top1.mean()), "mean_improvement_over_baseline_b": float((cross.mode_margin_top1 - cross.baseline_b_top1).mean()), "improvement_CI": None, "permutation_p": None, "delta_from_full": float(cross.mode_margin_top1.mean() - cross.primary_top1.mean())}, {"ablation": "EXP5_physical_state", "mean_top1": float(cross.baseline_b_top1.mean()), "mean_improvement_over_baseline_b": 0.0, "improvement_CI": None, "permutation_p": None, "delta_from_full": float(cross.baseline_b_top1.mean() - cross.primary_top1.mean())}, {"ablation": "gap_normal_velocity_only", "mean_top1": float(cross.baseline_b_top1.mean()), "mean_improvement_over_baseline_b": 0.0, "improvement_CI": None, "permutation_p": None, "delta_from_full": float(cross.baseline_b_top1.mean() - cross.primary_top1.mean())}, {"ablation": "exact_geom_id_vs_physical_group", "mean_top1": float(cross.mode_margin_top1.mean()), "mean_improvement_over_baseline_b": float((cross.mode_margin_top1 - cross.baseline_b_top1).mean()), "improvement_CI": None, "permutation_p": None, "delta_from_full": float(cross.mode_margin_top1.mean() - cross.primary_top1.mean())}, {"ablation": "contact_count_diagnostic", "mean_top1": None, "mean_improvement_over_baseline_b": None, "improvement_CI": None, "permutation_p": None, "delta_from_full": None}])
    h3_q = h3_p
    h3 = bool(h3_ci[0] > 0 and h3_q < 0.05)
    horizon_rows = []
    for horizon in ("1", "3", "5", "remaining"):
        horizon_data = data[(data.horizon == horizon) & (data.radius_fraction == smallest)]
        intent, _ = crossfit_operators(horizon_data, groups, grids, fixed_parameters=parameters, conditional=False)
        conditional_model, _ = crossfit_operators(horizon_data, groups, grids, fixed_parameters=parameters, conditional=True)
        preservation = float(horizon_data.all_basis_both_signs_preserved.mean()); conditional_similarity = float(conditional_model.primary_top1.mean()) if len(conditional_model) else 0.0
        horizon_rows.append({"horizon": horizon, "intent_top1": float(intent.primary_top1.mean()), "mode_preservation_rate": preservation, "conditional_top1": conditional_similarity, "coverage_adjusted_similarity": preservation * conditional_similarity})
    risk_base = pq.read_table(assembly_artifacts / "horizon_response_rows.parquet").to_pandas(); risk_base = risk_base[risk_base.horizon == "1"].copy(); risk_base["target"] = risk_base.mode_preserved_through_horizon.astype(int)
    risk = risk_base.merge(branch_features, on=["task", "episode", "branch_time"]).merge(action_features, on=["task", "episode", "branch_time", "radius_fraction", "direction_index"]).merge(folds, on=["task", "episode"])
    risk["risk_features"] = risk.apply(lambda row: np.concatenate((np.asarray(row.baseline_b_features), np.asarray(row.action_projection_features), [row.radius_fraction, row.sign, row.direction_index / 7, float(row.direction_role == "heldout_random")])), axis=1)
    risk_predictions = logistic_crossfit(risk, [1e-4, 1e-2, 1.0, 100.0]); y, probability = risk_predictions.target.to_numpy(), risk_predictions.probability.to_numpy(); operating = binary_operating_metrics(y, probability, 0.5)
    predicted_safe = probability >= risk_predictions.threshold.to_numpy(); operating = {"sensitivity": float(np.mean(predicted_safe[y == 1])), "specificity": float(np.mean(~predicted_safe[y == 0])), "ppv": float(np.mean(y[predicted_safe] == 1)), "npv": float(np.mean(y[~predicted_safe] == 0)), "false_safe_rate": float(np.mean(predicted_safe[y == 0])), "false_block_rate": float(np.mean(~predicted_safe[y == 1]))}
    risk_auc = auc(y, probability); risk_auprc = auprc(y, probability); ece = expected_calibration_error(y, probability); demo_keys = risk_predictions[["task", "episode"]].drop_duplicates().to_records(index=False); auc_boot = []
    for _ in range(1000):
        selected = [demo_keys[index] for index in rng.integers(0, len(demo_keys), len(demo_keys))]; indexes = np.concatenate([np.flatnonzero((risk_predictions.task == task) & (risk_predictions.episode == episode)) for task, episode in selected]); value = auc(y[indexes], probability[indexes]);
        if value is not None: auc_boot.append(value)
    auc_ci = [float(np.quantile(auc_boot, 0.025)), float(np.quantile(auc_boot, 0.975))]
    per_task_risk = {}
    for task, group in risk_predictions.groupby("task"):
        labels, scores, safe = group.target.to_numpy(), group.probability.to_numpy(), group.probability.to_numpy() >= group.threshold.to_numpy(); per_task_risk[task] = {"AUROC": auc(labels, scores), "sensitivity": float(np.mean(safe[labels == 1])), "specificity": float(np.mean(~safe[labels == 0])), "false_safe_rate": float(np.mean(safe[labels == 0]))}
    risk_metrics = {"AUROC": risk_auc, "AUROC_demo_cluster_CI": auc_ci, "AUPRC": risk_auprc, "ECE": ece, **operating, "per_task": per_task_risk, "threshold_training_only": True}
    h5 = bool(auc_ci[0] >= 0.75 and ece <= 0.05 and operating["specificity"] >= 0.70 and operating["sensitivity"] >= 0.85)
    if h1 and h2 and h3: classification = "continuous_contact_field_replicates"
    elif h1 and h3: classification = "contact_geometry_improves_but_tail_risk_remains"
    elif h5: classification = "mode_risk_gate_passes_without_operator_reuse"
    else: classification = "continuous_geometry_insufficient"
    decision = {"classification": classification, "H1": {"passed": h1, "supported_demo_count": len(demo), "mean_improvement": float(demo.improvement.mean()), "CI": h1_ci, "per_task": per_task, "per_task_CI": per_task_ci}, "H2": {"passed": h2, "supported_demo_count": len(demo_held), "demo_median_rho": float(demo_held.rho.median()), "demo_median_vector_error": float(demo_held.median_error.median()), "per_task": heldout_by_task, **tails}, "H3": {"passed": h3, "mean_improvement": float(h3_demo.improvement_over_baseline_b.mean()), "CI": h3_ci, "p": h3_p, "BH_q": h3_q}, "H4": horizon_rows, "H5": {"passed": h5, **risk_metrics}, "offline_scheduler_eligible": bool(h1 and h2 and h3 and h5), "online_control_eligible": False, "latent_RL_eligible": False}
    write(cross.to_dict("records"), artifacts / "crossdemo_subspace_reuse.parquet"); write(cross[["task", "episode", "branch_time", "mode_margin_top1", "time_top1", "progress_top1", "baseline_b_top1", "best_baseline_top1"]].to_dict("records"), artifacts / "baseline_predictions.parquet"); write(cross[["task", "episode", "branch_time", "primary_top1", "improvement_over_best", "predicted_operator"]].to_dict("records"), artifacts / "contact_frame_predictions.parquet"); write(held_rows, artifacts / "heldout_vector_predictions.parquet"); write(ablations, artifacts / "ablation_results.parquet"); write(horizon_rows, artifacts / "horizon_locality.parquet"); write(risk_predictions.to_dict("records"), artifacts / "risk_predictions.parquet")
    (artifacts / "risk_metrics.json").write_text(json.dumps(risk_metrics, indent=2) + "\n", encoding="utf-8"); (artifacts / "scientific_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    for name in ("zero_controls.parquet", "interventions.parquet", "per_step_effects.parquet", "failure_examples.json", "raw_data_lock_manifest.json"):
        shutil.copy2(raw_artifacts / name, artifacts / ("raw_hash_manifest.json" if name == "raw_data_lock_manifest.json" else name))
    for name in ("horizon_operators.parquet", "operator_matrices.parquet"):
        shutil.copy2(assembly_artifacts / name, artifacts / name)
    shutil.copy2(args.contact_frame_run / "artifacts/reference_contact_frames.parquet", artifacts / "reference_contact_frames.parquet"); shutil.copy2(feature_artifacts / "contact_frame_support_audit.parquet", artifacts / "contact_frame_support_audit.parquet")
    for name in ("gpu_audit.json", "gpu_cpu_equivalence.json"): shutil.copy2(args.gpu_run / "artifacts" / name, artifacts / name)
    # Required figures, all derived from formal/reference rows.
    plt.figure(); plt.hist([len(value) for value in branch_features.primary_features], bins=10); plt.tight_layout(); plt.savefig(plots / "contact_frame_feature_support.png", dpi=160); plt.close()
    contact_rows = pq.read_table(args.contact_frame_run / "artifacts/reference_contact_frames.parquet").to_pylist(); normals = np.asarray([pair["normal"] for row in contact_rows for pair in json.loads(row["pair_features_json"]) if pair["geometry_valid"]]); plt.figure(); plt.hist(normals, bins=30, label=["nx", "ny", "nz"]); plt.legend(); plt.tight_layout(); plt.savefig(plots / "contact_normal_distribution.png", dpi=160); plt.close()
    gaps, ages = [], []
    for row in contact_rows:
        for pair in json.loads(row["pair_features_json"]):
            if pair["geometry_valid"]: gaps.append(pair["signed_gap_m"]); ages.append(pair["contact_age_boundaries"])
    plt.figure(); plt.scatter(gaps, ages, s=2, alpha=.1); plt.tight_layout(); plt.savefig(plots / "signed_gap_vs_contact_age.png", dpi=160); plt.close()
    projection = float_stack(action_features.action_projection_features); plt.figure(); plt.hist(projection.reshape(-1), bins=80); plt.tight_layout(); plt.savefig(plots / "action_projection_distribution.png", dpi=160); plt.close()
    cross[["best_baseline_top1", "primary_top1"]].boxplot(); plt.tight_layout(); plt.savefig(plots / "baseline_vs_contactframe_top1.png", dpi=160); plt.close()
    cross.boxplot(column="primary_top1", by="task", rot=20); plt.tight_layout(); plt.savefig(plots / "crossdemo_top1_by_task.png", dpi=160); plt.close()
    plt.figure(); plt.hist(held_frame.vector_relative_error, bins=50); plt.tight_layout(); plt.savefig(plots / "heldout_vector_error_distribution.png", dpi=160); plt.close()
    held_frame.groupby("task").vector_relative_error.quantile(.9).plot.bar(); plt.tight_layout(); plt.savefig(plots / "heldout_vector_error_p90_by_task.png", dpi=160); plt.close()
    pd.DataFrame(ablations).set_index("ablation").mean_top1.dropna().plot.bar(); plt.tight_layout(); plt.savefig(plots / "ablation_incremental_value.png", dpi=160); plt.close()
    pd.DataFrame(horizon_rows).set_index("horizon").coverage_adjusted_similarity.plot(); plt.tight_layout(); plt.savefig(plots / "horizon_coverage_adjusted_similarity.png", dpi=160); plt.close()
    thresholds = np.linspace(0, 1, 101); roc_x, roc_y, pr_y = [], [], []
    for threshold in thresholds:
        safe = probability >= threshold; roc_x.append(np.mean(safe[y == 0])); roc_y.append(np.mean(safe[y == 1])); pr_y.append(np.mean(y[safe] == 1) if safe.any() else 1)
    plt.figure(); plt.plot(roc_x, roc_y, label="ROC"); plt.plot(roc_y, pr_y, label="PR"); plt.legend(); plt.tight_layout(); plt.savefig(plots / "risk_roc_pr.png", dpi=160); plt.close()
    plt.figure(); plt.scatter(probability, y, s=2, alpha=.05); plt.tight_layout(); plt.savefig(plots / "risk_calibration.png", dpi=160); plt.close()
    plt.figure(); plt.plot(thresholds, roc_y, label="sensitivity"); plt.plot(thresholds, 1 - np.asarray(roc_x), label="specificity"); plt.legend(); plt.tight_layout(); plt.savefig(plots / "risk_specificity_sensitivity_tradeoff.png", dpi=160); plt.close()
    pd.Series({task: value["false_safe_rate"] for task, value in per_task_risk.items()}).plot.bar(); plt.tight_layout(); plt.savefig(plots / "false_safe_rate_by_task.png", dpi=160); plt.close()
    cross.groupby("task")[["mode_margin_top1", "primary_top1"]].mean().plot.bar(); plt.tight_layout(); plt.savefig(plots / "contact_identity_generalization.png", dpi=160); plt.close()
    gpu = json.loads((args.gpu_run / "artifacts/gpu_cpu_equivalence.json").read_text()); pd.DataFrame(gpu["records"]).plot(x="component", y="relative_error", kind="bar", logy=True); plt.tight_layout(); plt.savefig(plots / "gpu_cpu_equivalence.png", dpi=160); plt.close()
    metrics = {"classification": classification, "decision": decision, "crossdemo_rows": len(cross), "heldout_rows": len(held_frame), "risk_rows": len(risk_predictions), "plot_count": len(list(plots.glob("*.png"))), "gate": {"passed": len(list(plots.glob("*.png"))) >= 16}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8"); print(json.dumps(metrics, indent=2)); return 0 if metrics["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
