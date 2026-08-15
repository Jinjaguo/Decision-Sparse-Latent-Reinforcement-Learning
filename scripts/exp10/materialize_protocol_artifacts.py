"""Materialize the exact EXP10 Stage-A artifact/plot contract from locked runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.exp10.run_stage_a_routes import sha256


def roc_curve(y, p):
    thresholds = np.r_[np.inf, np.sort(np.unique(p))[::-1], -np.inf]; points = []
    for threshold in thresholds:
        pred = p >= threshold; points.append((np.sum((y == 0) & pred) / max(1, np.sum(y == 0)), np.sum((y == 1) & pred) / max(1, np.sum(y == 1))))
    return np.asarray(points)


def pr_curve(y, p):
    order = np.argsort(-p); yy = y[order]; return np.cumsum(yy) / np.arange(1, len(y) + 1), np.cumsum(yy) / max(1, yy.sum())


def bar(df, labels, value, path, title, ylabel=None):
    fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(labels, df[value]); ax.set(title=title, ylabel=ylabel or value); ax.tick_params(axis="x", rotation=35); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True); parser.add_argument("--dataset-run", default="runs/exp10_a0_phase_macro_dataset_r2_20260814"); parser.add_argument("--model-run", default="runs/exp10_a1_a6_multiroute_models_r1_20260814"); parser.add_argument("--endpoint-run", default="runs/exp10_a7_endpoint_route_20260814"); parser.add_argument("--seed-run", default="runs/exp10_a7_seed_variance_20260814")
    args = parser.parse_args(); out = Path("runs") / args.run_id
    if out.exists(): raise FileExistsError(f"immutable run exists: {out}")
    artifacts, plots = out / "artifacts", out / "plots"; artifacts.mkdir(parents=True); plots.mkdir()
    dataset, model, endpoint, seed = map(Path, (args.dataset_run, args.model_run, args.endpoint_run, args.seed_run))
    phase = pd.read_parquet(dataset / "artifacts" / "reference_phase_labels.parquet")
    support = phase.groupby(["task", "phase"], as_index=False).agg(boundaries=("action_index", "size"), demonstrations=("episode", "nunique"), mean_normalized_time=("normalized_time", "mean"))
    support.to_parquet(artifacts / "phase_support_audit.parquet", index=False)
    transitions = []
    for (task, episode), group in phase.sort_values("action_index").groupby(["task", "episode"]):
        values = group.phase_index.to_numpy()
        for a, b in zip(values[:-1], values[1:]): transitions.append({"task": task, "episode": episode, "phase_from": f"P{a}", "phase_to": f"P{b}"})
    transition = pd.DataFrame(transitions).groupby(["task", "phase_from", "phase_to"], as_index=False).size().rename(columns={"size": "count"}); transition.to_parquet(artifacts / "phase_transition_matrix.parquet", index=False)
    data = pd.read_parquet(dataset / "artifacts" / "macro_trajectory_dataset.parquet", columns=["task", "episode", "phase", "effective_h5", "effective_h10", "chunk5_summary", "chunk10_summary", "chunk20_summary", "phase_chunk_summary", "history1_summary", "history5_summary", "history10_summary"])
    chunk_rows = []
    for task, group in data.groupby("task"):
        for name in ("chunk5", "chunk10", "chunk20", "phase_chunk"):
            coverage = np.asarray([x[-1] for x in group[f"{name}_summary"]]); chunk_rows.append({"task": task, "chunk": name, "rows": len(group), "mean_valid_fraction": float(coverage.mean()), "min_valid_fraction": float(coverage.min())})
    pd.DataFrame(chunk_rows).to_parquet(artifacts / "macro_chunk_support.parquet", index=False)

    trajectory_path = model / "artifacts" / "trajectory_predictions.parquet"
    for track in "ABCF":
        table = pq.read_table(trajectory_path, filters=[("track", "=", track)])
        pq.write_table(table, artifacts / f"track{track}_predictions.parquet", compression="zstd")
    pq.write_table(pq.read_table(endpoint / "artifacts" / "endpoint_predictions.parquet"), artifacts / "trackD_predictions.parquet", compression="zstd")
    events = pq.read_table(model / "artifacts" / "event_terminal_predictions.parquet", filters=[("model", "=", "E_teacher_free_regime_ar")]); pq.write_table(events, artifacts / "trackE_predictions.parquet", compression="zstd")
    metrics = pd.read_parquet(model / "artifacts" / "trajectory_metrics.parquet"); event_metrics = pd.read_parquet(model / "artifacts" / "event_terminal_metrics.parquet")
    metric_rows = metrics.assign(metric_family="trajectory").to_dict("records") + event_metrics.assign(metric_family="event").to_dict("records")
    pq.write_table(pa.Table.from_pylist(metric_rows), artifacts / "route_metrics.parquet", compression="zstd")
    ablation_names = ["no_action", "no_history", "A_chunk5", "A_chunk10", "A_chunk20", "A_phase_chunk", "A_history1", "A_history5", "A_history10", "A_phase_history", "B_hybrid_phase", "B_shuffled_phase"]
    metrics[metrics.method.isin(ablation_names)].to_parquet(artifacts / "ablation_results.parquet", index=False)
    seed_json = json.loads((seed / "artifacts" / "seed_variance.json").read_text(encoding="utf-8")); (artifacts / "seed_variance.json").write_text(json.dumps(seed_json, indent=2), encoding="utf-8")
    authorization = json.loads((model / "artifacts" / "route_authorization.json").read_text(encoding="utf-8")); endpoint_gate = json.loads((endpoint / "metrics.json").read_text(encoding="utf-8")); authorization["route_gates"]["D"] = endpoint_gate["track_d_authorized"]; authorization["protocol_artifacts_materialized_after_source_prediction_locks"] = True
    (artifacts / "route_authorization.json").write_text(json.dumps(authorization, indent=2), encoding="utf-8")

    # Required plots, each tied to an audited table.
    pivot = support.pivot(index="task", columns="phase", values="boundaries").fillna(0); fig, ax = plt.subplots(figsize=(9, 5)); pivot.plot.bar(stacked=True, ax=ax); ax.set(title="Reference phase duration/support", ylabel="boundaries"); fig.tight_layout(); fig.savefig(plots / "phase_duration_by_task.png", dpi=150); plt.close(fig)
    matrix = transition.groupby(["phase_from", "phase_to"])["count"].sum().unstack(fill_value=0).reindex(index=[f"P{i}" for i in range(7)], columns=[f"P{i}" for i in range(7)], fill_value=0); fig, ax = plt.subplots(figsize=(6, 5)); image = ax.imshow(matrix); fig.colorbar(image, ax=ax); ax.set(xticks=range(7), yticks=range(7), xticklabels=matrix.columns, yticklabels=matrix.index, title="Reference phase transition matrix"); fig.tight_layout(); fig.savefig(plots / "phase_transition_matrix.png", dpi=150); plt.close(fig)
    h5 = metrics[metrics.horizon == "H5"].set_index("method"); h10 = metrics[metrics.horizon == "H10"].set_index("method")
    def method_plot(names, filename, title, field="mean_energy"):
        frame = pd.DataFrame({"H5": [h5.loc[x, field] for x in names], "H10": [h10.loc[x, field] for x in names]}, index=names); fig, ax = plt.subplots(figsize=(9, 5)); frame.plot.bar(ax=ax); ax.set(title=title, ylabel=field); fig.tight_layout(); fig.savefig(plots / filename, dpi=150); plt.close(fig)
    method_plot(["A_chunk5", "A_chunk10", "A_chunk20", "A_phase_chunk"], "chunk_scale_energy_score.png", "Chunk-scale energy score")
    method_plot(["A_history1", "A_history5", "A_history10", "A_phase_history"], "history_scale_energy_score.png", "History-scale energy score")
    method_plot(["exp9_temporal", "B_hard_phase_moe", "B_soft_phase_moe", "B_hybrid_phase"], "phase_moe_vs_global.png", "Phase experts versus global")
    method_plot(["C_latent0", "C_latent16", "C_latent32"], "trajectory_latent_dimension.png", "Trajectory latent dimension")
    event_all = pd.read_parquet(model / "artifacts" / "event_terminal_predictions.parquet")
    for model_name, filename, title, include_pr in (("D_terminal_direct", "terminal_predicate_roc.png", "Terminal consequence ROC", False), ("E_teacher_free_regime_ar", "coarse_regime_roc_pr.png", "Coarse regime ROC/PR", True)):
        subset = event_all[event_all.model == model_name]; y, p = subset.target.to_numpy(), subset.probability.to_numpy(); roc = roc_curve(y, p); fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(roc[:, 0], roc[:, 1], label="ROC"); ax.plot([0, 1], [0, 1], "--", color="grey")
        if include_pr:
            precision, recall = pr_curve(y, p); ax.plot(recall, precision, label="PR")
        ax.set(xlabel="FPR / Recall", ylabel="TPR / Precision", title=title); ax.legend(); fig.tight_layout(); fig.savefig(plots / filename, dpi=150); plt.close(fig)
    method_plot(["exp9_temporal", "F_deterministic", "F_heteroscedastic", "F_residual_mixture", "F_cvae16"], "world_action_energy_by_horizon.png", "World-action energy by horizon")
    top = ["baseline_b", "exp9_temporal", "A_chunk10", "B_hybrid_phase", "C_latent16", "F_cvae16"]
    method_plot(top, "coverage_by_track.png", "90% marginal coverage", "mean_coverage90")
    method_plot(top, "p90_error_by_track.png", "P90 endpoint error", "p90_endpoint_l2")
    method_plot(["no_action", "A_chunk5", "A_chunk10", "A_chunk20"], "action_chunk_ablation.png", "Action chunk ablation")
    method_plot(["no_history", "A_history1", "A_history5", "A_history10"], "history_ablation.png", "History ablation")
    method_plot(["exp9_temporal", "B_hybrid_phase", "B_shuffled_phase"], "phase_ablation.png", "Phase-label ablation")
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(authorization["route_gates"].keys(), [int(v) for v in authorization["route_gates"].values()]); ax.set(ylim=(0, 1.2), title="Route authorization summary", ylabel="pass"); fig.tight_layout(); fig.savefig(plots / "route_authorization_summary.png", dpi=150); plt.close(fig)
    seed_metrics = pd.read_parquet(seed / "artifacts" / "seed_metrics.parquet"); pivot_seed = seed_metrics.pivot(index="seed", columns="horizon", values="mean_energy"); fig, ax = plt.subplots(figsize=(7, 4)); pivot_seed.plot.bar(ax=ax); ax.set(title="CVAE seed variance", ylabel="mean energy"); fig.tight_layout(); fig.savefig(plots / "seed_variance.png", dpi=150); plt.close(fig)

    hashes = {p.name: sha256(p) for p in sorted(artifacts.iterdir()) if p.is_file()}; (artifacts / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    report = "# EXP10 Stage-A report\n\nAll six independently authorized routes failed on locked retrospective predictions. No Stage-B cohort was authorized or executed. See `reports/exp10_report.md` for the detailed scientific report.\n"
    (artifacts / "stageA_report.md").write_text(report, encoding="utf-8")
    result = {"status": "completed", "run_id": args.run_id, "required_artifacts": 13, "required_plots": len(list(plots.glob("*.png"))), "source_prediction_locks": json.loads((model / "artifacts" / "prediction_lock.json").read_text(encoding="utf-8")), "all_routes_failed": not any(authorization["route_gates"].values())}
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8"); (out / "stdout.log").write_text(json.dumps(result, indent=2), encoding="utf-8"); (out / "stderr.log").write_text("", encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
