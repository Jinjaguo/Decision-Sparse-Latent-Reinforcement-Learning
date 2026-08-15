"""Complete-demo cross-fitted EXP12 consequence-ranking comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_sparse_rl.metrics.exp12 import (
    abstaining_choice,
    catastrophic_selection,
    deterministic_choice,
    nominal_improvement_opportunity,
    pairwise_accuracy,
    pairwise_preferences,
    regret,
    sha256_file,
    top1_accuracy,
)


ENCODINGS = {"E1_raw": "x_e1", "E2_object": "x_e2", "E3_contact": "x_e3", "E4_history": "x_e4"}
SPECIALISTS = ("contact_only", "motion_only", "outcome_only")
COMBINATIONS = (
    "contact_only", "motion_only", "outcome_only", "contact_motion", "contact_outcome",
    "motion_outcome", "all_consequences", "hard_filter_motion", "learned_small",
    "pairwise_comparator", "listwise", "tail_aware", "lexicographic",
    "all_plus_uncertainty", "abstaining", "richer_future", "nominal", "random",
)


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_once(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"prediction lock already exists: {path}")
    if not rows:
        raise RuntimeError(f"refusing empty prediction lock: {path}")
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def ridge_fit(x, y, l2=3.0, weights=None, intercept=True):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    mu = x.mean(0)
    sd = x.std(0)
    sd[sd < 1e-8] = 1.0
    z = (x - mu) / sd
    if intercept:
        z = np.c_[np.ones(len(z)), z]
    root = np.ones(len(z)) if weights is None else np.sqrt(np.asarray(weights, float))
    zw, yw = z * root[:, None], y * root[:, None]
    if z.shape[1] <= len(z):
        penalty = np.eye(z.shape[1]) * l2
        if intercept:
            penalty[0, 0] = 0
        coef = np.linalg.solve(zw.T @ zw + penalty, zw.T @ yw)
    else:
        coef = zw.T @ np.linalg.solve(zw @ zw.T + np.eye(len(zw)) * l2, yw)
    return mu, sd, coef, intercept


def ridge_predict(model, x):
    mu, sd, coef, intercept = model
    z = (np.asarray(x, float) - mu) / sd
    if intercept:
        z = np.c_[np.ones(len(z)), z]
    return z @ coef


class SharedMLP(nn.Module):
    def __init__(self, inputs: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x):
        return self.net(x)


def shared_predict(xfit, yfit, xall, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = xfit.mean(0), xfit.std(0)
    sd[sd < 1e-8] = 1
    xf = torch.tensor((xfit - mu) / sd, dtype=torch.float32, device=device)
    xa = torch.tensor((xall - mu) / sd, dtype=torch.float32, device=device)
    yf = torch.tensor(yfit, dtype=torch.float32, device=device)
    model = SharedMLP(xfit.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    weights = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32, device=device)
    for _ in range(80):
        optimizer.zero_grad()
        prediction = model(xf)
        loss = (((prediction - yf) ** 2) * weights).mean()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(xa).cpu().numpy(), sum(p.numel() for p in model.parameters())


def build_pairwise(x, quality, groups, indexes, tolerance=.05):
    index_set = set(np.asarray(indexes, int).tolist())
    features, labels, weights = [], [], []
    for members in groups.values():
        members = [i for i in members if i in index_set]
        for a, b, label, gap in pairwise_preferences(quality[members], tolerance):
            if label == 0:
                continue
            features.append(x[members[a]] - x[members[b]])
            labels.append(label)
            weights.append(1.0 + min(gap, 2.0))
    return np.asarray(features), np.asarray(labels, float), np.asarray(weights, float)


def within_group_rank_target(quality, groups, indexes):
    target = np.zeros(len(indexes), dtype=float)
    lookup = {value: pos for pos, value in enumerate(indexes)}
    index_set = set(indexes)
    for members in groups.values():
        selected = [i for i in members if i in index_set]
        if not selected:
            continue
        order = np.argsort(np.argsort(quality[selected], kind="mergesort"), kind="mergesort")
        denom = max(1, len(selected) - 1)
        for i, rank in zip(selected, order):
            target[lookup[i]] = rank / denom
    return target


def ece(labels, probability, bins=10):
    labels, probability = np.asarray(labels, float), np.clip(np.asarray(probability, float), 0, 1)
    total = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        mask = (probability >= lo) & (probability < lo + 1 / bins + 1e-12)
        if np.any(mask):
            total += mask.mean() * abs(labels[mask].mean() - probability[mask].mean())
    return float(total)


def bootstrap_difference(left, right, demos, seed=121202):
    keys = sorted(set(demos))
    lookup_left = {d: np.mean([v for v, k in zip(left, demos) if k == d]) for d in keys}
    lookup_right = {d: np.mean([v for v, k in zip(right, demos) if k == d]) for d in keys}
    delta = np.asarray([lookup_left[d] - lookup_right[d] for d in keys])
    rng = np.random.default_rng(seed)
    draws = [float(np.mean(rng.choice(delta, len(delta), replace=True))) for _ in range(2000)]
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def bars(path, labels, values, title, ylabel):
    fig, ax = plt.subplots(figsize=(max(7, len(values) * .55), 4))
    ax.bar(np.arange(len(values)), values)
    ax.set_xticks(np.arange(len(values)), labels, rotation=30, ha="right")
    ax.set(title=title, ylabel=ylabel)
    save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-run", type=Path, default=Path("runs/exp12_a0_candidate_dataset_r1_20260815"))
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    if out.exists():
        raise FileExistsError(f"immutable run exists: {out}")
    artifacts, plots = out / "artifacts", out / "plots"
    artifacts.mkdir(parents=True)
    plots.mkdir()
    started = time.perf_counter()
    dataset_path = ROOT / args.dataset_run / "artifacts/candidate_dataset.parquet"
    rows = [x for x in pq.read_table(dataset_path).to_pylist() if x["primary_valid"]]
    ids = np.asarray([x["candidate_id"] for x in rows])
    branch = np.asarray([x["branch_id"] for x in rows])
    task = np.asarray([x["task"] for x in rows])
    demo = np.asarray([x["demo_key"] for x in rows])
    nominal = np.asarray([x["is_nominal"] for x in rows], bool)
    catastrophic = np.asarray([x["catastrophic"] for x in rows], bool)
    y = np.asarray([[x["contact_quality"], x["motion_quality"], x["outcome_quality"]] for x in rows], float)
    quality = np.asarray([x["composite_quality"] for x in rows], float)
    future = np.asarray([x["future_rich20"] for x in rows], float)
    features = {name: np.asarray([x[column] for x in rows], float) for name, column in ENCODINGS.items()}
    groups = defaultdict(list)
    for index, key in enumerate(branch):
        groups[key].append(index)
    split = json.loads((ROOT / args.dataset_run / "manifests/crossfit_manifest.json").read_text())["assignments"]
    fold_map = {x["demo_key"]: int(x["fold"]) for x in split}
    folds = np.asarray([fold_map[x] for x in demo])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = (121201, 121202, 121203)

    prediction = {name: np.zeros((len(rows), 3), float) for name in (*ENCODINGS, "E5_separate", "E6_shared", "RICH20")}
    uncertainty = np.zeros(len(rows), float)
    direct_scores = {name: np.zeros(len(rows), float) for name in ("pairwise_comparator", "listwise", "tail_aware", "richer_future", "learned_small")}
    richer_prediction = np.zeros_like(future)
    abstain_threshold = {}
    parameter_counts = defaultdict(int)

    for fold in range(5):
        test = folds == fold
        outer = ~test
        outer_demos = sorted(set(demo[outer]))
        cal_demos = {d for i, d in enumerate(outer_demos) if i % 4 == 0}
        cal = outer & np.isin(demo, list(cal_demos))
        fit = outer & ~cal
        fit_idx, cal_idx, test_idx = map(np.flatnonzero, (fit, cal, test))
        all_idx = np.r_[cal_idx, test_idx]
        for name, x in features.items():
            model = ridge_fit(x[fit], y[fit], l2=3.0)
            prediction[name][test] = ridge_predict(model, x[test])
        x = features["E3_contact"]
        for target, l2 in enumerate((10.0, 3.0, 10.0)):
            model = ridge_fit(x[fit], y[fit, target], l2=l2)
            prediction["E5_separate"][test, target] = ridge_predict(model, x[test]).ravel()
        ensemble = []
        for seed in seeds:
            pred, count = shared_predict(x[fit], y[fit], x[all_idx], seed, device)
            ensemble.append(pred)
            parameter_counts["E6_shared"] = count
        ensemble = np.asarray(ensemble)
        mean = ensemble.mean(0)
        prediction["E6_shared"][test] = mean[len(cal_idx):]
        uncertainty[test] = ensemble[:, len(cal_idx):].std(0).mean(1)
        cal_uncertainty = ensemble[:, :len(cal_idx)].std(0).mean(1)
        abstain_threshold[fold] = float(np.quantile(cal_uncertainty, .9)) if len(cal_uncertainty) else float("inf")

        pair_x, pair_y, pair_w = build_pairwise(x, quality, groups, fit_idx)
        pair_model = ridge_fit(pair_x, pair_y, l2=10, weights=pair_w, intercept=False)
        direct_scores["pairwise_comparator"][test] = ridge_predict(pair_model, x[test]).ravel()
        rank_target = within_group_rank_target(quality, groups, fit_idx)
        direct_scores["listwise"][test] = ridge_predict(ridge_fit(x[fit], rank_target, 10), x[test]).ravel()
        nominal_by_branch = {key: quality[next(i for i in members if nominal[i])] for key, members in groups.items()}
        tail_weights = np.asarray([1 + 4 * catastrophic[i] + min(3, abs(quality[i] - nominal_by_branch[branch[i]])) for i in fit_idx])
        direct_scores["tail_aware"][test] = ridge_predict(ridge_fit(x[fit], quality[fit], 10, tail_weights), x[test]).ravel()

        rich_model = ridge_fit(x[fit], future[fit], l2=20)
        rich_all = ridge_predict(rich_model, x[all_idx])
        richer_prediction[test] = rich_all[len(cal_idx):]
        evaluator = ridge_fit(future[fit], y[fit], l2=20)
        prediction["RICH20"][test] = ridge_predict(evaluator, rich_all[len(cal_idx):])
        direct_scores["richer_future"][test] = np.c_[prediction["RICH20"][test], np.ones(test.sum())] @ np.asarray([1.0, 1.0, 4.0, 0.0])

        cal_pred = np.zeros((len(cal_idx), 3))
        for target, l2 in enumerate((10.0, 3.0, 10.0)):
            model = ridge_fit(x[fit], y[fit, target], l2=l2)
            cal_pred[:, target] = ridge_predict(model, x[cal]).ravel()
        coordinator = ridge_fit(cal_pred, quality[cal], l2=3)
        direct_scores["learned_small"][test] = ridge_predict(coordinator, prediction["E5_separate"][test]).ravel()

    for key in prediction:
        prediction[key][:, 0] = np.clip(prediction[key][:, 0], -1, 1.5)
        prediction[key][:, 2] = np.clip(prediction[key][:, 2], 0, 1)
    card = prediction["E5_separate"]
    scores = {
        "contact_only": card[:, 0], "motion_only": card[:, 1], "outcome_only": card[:, 2],
        "contact_motion": card[:, 0] + card[:, 1], "contact_outcome": card[:, 0] + 4 * card[:, 2],
        "motion_outcome": card[:, 1] + 4 * card[:, 2], "all_consequences": card[:, 0] + card[:, 1] + 4 * card[:, 2],
        "hard_filter_motion": np.where((card[:, 0] >= .5) & (card[:, 2] >= .5), card[:, 1], -10 + card[:, 0] + card[:, 2]),
        "learned_small": direct_scores["learned_small"], "pairwise_comparator": direct_scores["pairwise_comparator"],
        "listwise": direct_scores["listwise"], "tail_aware": direct_scores["tail_aware"],
        "lexicographic": 100 * card[:, 2] + 10 * card[:, 0] + card[:, 1],
        "all_plus_uncertainty": card[:, 0] + card[:, 1] + 4 * card[:, 2] - uncertainty,
        "abstaining": card[:, 0] + card[:, 1] + 4 * card[:, 2], "richer_future": direct_scores["richer_future"],
        "nominal": nominal.astype(float), "random": np.zeros(len(rows)),
    }

    consequence_rows = []
    for i, row in enumerate(rows):
        record = {"candidate_id": ids[i], "branch_id": branch[i], "task": task[i], "demo_key": demo[i], "fold": int(folds[i]), "actual_contact": y[i, 0], "actual_motion": y[i, 1], "actual_outcome": y[i, 2], "actual_composite": quality[i], "uncertainty": uncertainty[i]}
        for name, value in prediction.items():
            record[f"{name}_card"] = value[i].tolist()
        consequence_rows.append(record)
    write_once(artifacts / "consequence_predictions.parquet", consequence_rows)
    write_once(artifacts / "richer_future_predictions.parquet", [{"candidate_id": ids[i], "actual_future": future[i].tolist(), "predicted_future": richer_prediction[i].tolist()} for i in range(len(rows))])

    setwise_rows, route_group_metrics = [], defaultdict(list)
    for group_key, members in sorted(groups.items()):
        members = np.asarray(members, int)
        nominal_local = int(np.flatnonzero(nominal[members])[0])
        q = quality[members]
        for route in COMBINATIONS:
            fallback = False
            if route == "random":
                selected_local = deterministic_choice(ids[members].tolist(), salt=group_key)
            elif route == "abstaining":
                selected_local, fallback = abstaining_choice(scores[route][members], uncertainty[members], nominal_local, abstain_threshold[int(folds[members[0]])])
            else:
                selected_local = int(np.argmax(scores[route][members]))
            selected = int(members[selected_local])
            opportunity, gap = nominal_improvement_opportunity(q, nominal_local, .05)
            nominal_quality = q[nominal_local]
            record = {
                "route": route, "branch_id": group_key, "task": task[selected], "demo_key": demo[selected],
                "selected_candidate_id": ids[selected], "oracle_candidate_id": ids[members[int(np.argmax(q))]],
                "nominal_candidate_id": ids[members[nominal_local]], "selected_quality": quality[selected],
                "oracle_quality": float(np.max(q)), "nominal_quality": float(nominal_quality),
                "top1_correct": top1_accuracy(q, selected_local, .05), "regret": regret(q, selected_local),
                "catastrophic_selection": catastrophic_selection(catastrophic[members], selected_local),
                "opportunity": opportunity, "opportunity_gap": gap,
                "opportunity_captured": bool(opportunity and quality[selected] > nominal_quality + .05),
                "fallback": fallback,
            }
            setwise_rows.append(record)
            route_group_metrics[route].append(record)
    write_once(artifacts / "setwise_ranking_predictions.parquet", setwise_rows)

    route_metrics = []
    for route in COMBINATIONS:
        records = route_group_metrics[route]
        pa = [pairwise_accuracy(quality[members], scores[route][members], .05) for members in groups.values()]
        opportunities = [x for x in records if x["opportunity"]]
        route_metrics.append({
            "route": route, "pairwise_accuracy": float(np.nanmean(pa)),
            "top1_accuracy": float(np.mean([x["top1_correct"] for x in records])),
            "median_regret": float(np.median([x["regret"] for x in records])),
            "p90_regret": float(np.quantile([x["regret"] for x in records], .9)),
            "p95_regret": float(np.quantile([x["regret"] for x in records], .95)),
            "catastrophic_selection_rate": float(np.mean([x["catastrophic_selection"] for x in records])),
            "opportunity_capture": float(np.mean([x["opportunity_captured"] for x in opportunities])) if opportunities else float("nan"),
            "fallback_rate": float(np.mean([x["fallback"] for x in records])),
        })
    write_once(artifacts / "ablation_results.parquet", route_metrics)
    best = min([x for x in route_metrics if x["route"] not in ("nominal", "random")], key=lambda x: (x["p90_regret"], x["median_regret"], -x["top1_accuracy"]))
    best_route = best["route"]

    pair_rows = []
    for group_key, members in groups.items():
        members = np.asarray(members, int)
        for a, b, label, gap in pairwise_preferences(quality[members], .05):
            if label:
                pair_rows.append({"branch_id": group_key, "candidate_i": ids[members[a]], "candidate_j": ids[members[b]], "true_preference": label, "quality_gap": gap, "score_i": float(scores[best_route][members[a]]), "score_j": float(scores[best_route][members[b]]), "correct": bool((scores[best_route][members[a]] - scores[best_route][members[b]]) * label > 0)})
    write_once(artifacts / "pairwise_ranking_predictions.parquet", pair_rows)

    encoding_metrics = []
    for name, pred in prediction.items():
        error = np.abs(pred - y)
        encoding_metrics.append({"encoding": name, "contact_mae": float(error[:, 0].mean()), "motion_mae": float(error[:, 1].mean()), "outcome_brier": float(np.mean((pred[:, 2] - y[:, 2]) ** 2)), "outcome_ece": ece(y[:, 2], pred[:, 2])})
    write_once(artifacts / "encoding_results.parquet", encoding_metrics)
    rich_future_mae = float(np.mean(np.abs(richer_prediction - future)))
    nominal_records = route_group_metrics["nominal"]
    best_records = route_group_metrics[best_route]
    single_best = min((x for x in route_metrics if x["route"] in SPECIALISTS), key=lambda x: x["p90_regret"])
    random_metric = next(x for x in route_metrics if x["route"] == "random")
    nominal_metric = next(x for x in route_metrics if x["route"] == "nominal")
    richer_metric = next(x for x in route_metrics if x["route"] == "richer_future")
    best_regret = [-x["regret"] for x in best_records]
    nominal_regret = [-x["regret"] for x in nominal_records]
    regret_ci = bootstrap_difference(best_regret, nominal_regret, [x["demo_key"] for x in best_records])
    axes = {
        "contact_prediction": "supported" if min(x["contact_mae"] for x in encoding_metrics) < .2 else "unsupported",
        "motion_prediction": "supported" if min(x["motion_mae"] for x in encoding_metrics) < .2 else "unsupported",
        "outcome_prediction": "supported" if min(x["outcome_brier"] for x in encoding_metrics) < .1 else "unsupported",
        "pairwise_ranking": "supported" if best["pairwise_accuracy"] >= .65 else "unsupported",
        "setwise_ranking": "supported" if best["top1_accuracy"] > max(nominal_metric["top1_accuracy"], random_metric["top1_accuracy"]) else "unsupported",
        "tail_safety": "supported" if best["catastrophic_selection_rate"] <= single_best["catastrophic_selection_rate"] and best["p95_regret"] <= single_best["p95_regret"] else "unsupported",
        "uncertainty_calibration": "supported" if next(x for x in route_metrics if x["route"] == "abstaining")["catastrophic_selection_rate"] <= next(x for x in route_metrics if x["route"] == "all_consequences")["catastrophic_selection_rate"] else "inconclusive",
        "consequence_sufficiency_vs_richer_future": "supported" if best["p90_regret"] <= richer_metric["p90_regret"] and best["top1_accuracy"] >= richer_metric["top1_accuracy"] else "unsupported",
    }
    offline_promising = axes["pairwise_ranking"] == "supported" and axes["setwise_ranking"] == "supported" and best["p90_regret"] < single_best["p90_regret"]
    decision = {
        "availability": "retrospective availability-limited EXP11 pilot",
        "offline_candidate_ranking": "promising" if offline_promising else "not_yet_supported",
        "best_encoding": min(encoding_metrics, key=lambda x: x["contact_mae"] + x["motion_mae"] + x["outcome_brier"])["encoding"],
        "best_loss": best_route if best_route in ("pairwise_comparator", "listwise", "tail_aware") else "L1_independent",
        "best_combination": best_route,
        "best_consequence_subset": best_route if best_route in COMBINATIONS[:7] else "contact_motion_outcome",
        "uncertainty_useful": axes["uncertainty_calibration"] == "supported",
        "tail_problem_remaining": best["catastrophic_selection_rate"] > 0 or best["p95_regret"] > .1,
        "richer_future_baseline_status": {"route_metrics": richer_metric, "future_mae": rich_future_mae, "predicted_outputs": int(future.shape[1])},
        "oracle_opportunity": {"groups": int(sum(x["opportunity"] for x in best_records)), "total_groups": len(best_records), "tasks_with_opportunity": sorted(set(x["task"] for x in best_records if x["opportunity"]))},
        "regret_improvement_vs_nominal_demo_ci": regret_ci,
        "axes": axes,
        "recommended_next_module": "candidate_generation_diversity" if sum(x["opportunity"] for x in best_records) < .3 * len(best_records) else "ranking_tail_or_closed_loop",
    }
    dump(artifacts / "scientific_decision.json", decision)
    failures = sorted([x for x in best_records if x["regret"] > .05 or x["catastrophic_selection"]], key=lambda x: (x["catastrophic_selection"], x["regret"]), reverse=True)[:30]
    dump(artifacts / "failure_examples.json", failures)
    ranking_metrics = {"best": best, "single_best": single_best, "nominal": nominal_metric, "random": random_metric, "richer_future": richer_metric, "routes": route_metrics, "encoding_metrics": encoding_metrics}
    dump(artifacts / "ranking_metrics.json", ranking_metrics)
    dump(artifacts / "tail_metrics.json", {"best_route": best_route, "median_regret": best["median_regret"], "p90_regret": best["p90_regret"], "p95_regret": best["p95_regret"], "catastrophic_selection_rate": best["catastrophic_selection_rate"], "failure_count": len(failures)})

    task_names = sorted(set(task))
    best_task = []
    for task_name in task_names:
        subset = [x for x in best_records if x["task"] == task_name]
        best_task.append({"task": task_name, "top1": float(np.mean([x["top1_correct"] for x in subset])), "regret": float(np.mean([x["regret"] for x in subset])), "catastrophic": float(np.mean([x["catastrophic_selection"] for x in subset]))})
    sizes = [len(x) for x in groups.values()]
    bars(plots / "candidate_set_size_distribution.png", [str(x) for x in sorted(set(sizes))], [sizes.count(x) for x in sorted(set(sizes))], "Candidate-set size", "groups")
    opportunity = pq.read_table(ROOT / args.dataset_run / "artifacts/candidate_opportunity.parquet").to_pylist()
    bars(plots / "oracle_improvement_opportunity.png", task_names, [sum(x["opportunity"] for x in opportunity if x["task"] == t) for t in task_names], "Oracle improvement opportunity", "groups")
    bars(plots / "pairwise_accuracy_by_task.png", task_names, [float(np.nanmean([pairwise_accuracy(quality[np.asarray(m)], scores[best_route][np.asarray(m)]) for k, m in groups.items() if task[m[0]] == t])) for t in task_names], "Pairwise accuracy", "accuracy")
    bars(plots / "top1_accuracy_by_task.png", [x["task"] for x in best_task], [x["top1"] for x in best_task], "Top-1 by task", "accuracy")
    fig, ax = plt.subplots(); ax.hist([x["regret"] for x in best_records], bins=20); ax.set(title="Selected-candidate regret", xlabel="regret", ylabel="groups"); save(fig, plots / "regret_distribution.png")
    bars(plots / "catastrophic_selection_rate.png", [x["route"] for x in route_metrics], [x["catastrophic_selection_rate"] for x in route_metrics], "Catastrophic selection", "rate")
    bars(plots / "specialist_ablation.png", list(COMBINATIONS[:7]), [next(x["top1_accuracy"] for x in route_metrics if x["route"] == r) for r in COMBINATIONS[:7]], "Specialist subsets", "top-1")
    bars(plots / "encoding_ablation.png", [x["encoding"] for x in encoding_metrics], [x["contact_mae"] + x["motion_mae"] + x["outcome_brier"] for x in encoding_metrics], "Encoding aggregate error", "error")
    bars(plots / "loss_ablation.png", ["independent", "pairwise", "listwise", "tail"], [next(x["p90_regret"] for x in route_metrics if x["route"] == r) for r in ("all_consequences", "pairwise_comparator", "listwise", "tail_aware")], "Loss routes", "P90 regret")
    combination_names = ("hard_filter_motion", "all_consequences", "learned_small", "lexicographic", "abstaining")
    bars(plots / "combination_ablation.png", list(combination_names), [next(x["p90_regret"] for x in route_metrics if x["route"] == r) for r in combination_names], "Combination routes", "P90 regret")
    thresholds = np.quantile(uncertainty, np.linspace(.1, 1, 10)); coverage, accuracy = [], []
    all_score = scores["all_consequences"]
    for threshold in thresholds:
        accepted = uncertainty <= threshold; coverage.append(float(accepted.mean())); accuracy.append(float(pairwise_accuracy(quality[accepted], all_score[accepted])) if accepted.sum() > 2 else float("nan"))
    fig, ax = plt.subplots(); ax.plot(coverage, accuracy, marker="o"); ax.set(title="Uncertainty abstention", xlabel="candidate coverage", ylabel="pairwise accuracy"); save(fig, plots / "uncertainty_abstention_curve.png")
    bars(plots / "consequence_vs_richer_future.png", [best_route, "richer_future"], [best["p90_regret"], richer_metric["p90_regret"]], "Prediction scope comparison", "P90 regret")
    bars(plots / "failure_examples_summary.png", task_names, [sum(x["task"] == t for x in failures) for t in task_names], "Failure examples", "count")

    lock_files = [artifacts / x for x in ("consequence_predictions.parquet", "pairwise_ranking_predictions.parquet", "setwise_ranking_predictions.parquet", "richer_future_predictions.parquet", "ablation_results.parquet", "encoding_results.parquet")]
    dump(artifacts / "prediction_hash_manifest.json", {str(x.name): sha256_file(x) for x in lock_files})
    metrics = {
        "status": "completed", "run_id": args.run_id, "device": device, "sample_count": len(rows),
        "group_count": len(groups), "demo_count": len(set(demo)), "best_route": best_route,
        "decision": decision, "parameter_counts": dict(parameter_counts),
        "training_seconds": time.perf_counter() - started,
        "source_dataset_sha256": sha256_file(dataset_path),
        "prediction_hashes_verified": all(sha256_file(x) == json.loads((artifacts / "prediction_hash_manifest.json").read_text())[x.name] for x in lock_files),
    }
    dump(out / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    raise SystemExit(main())
