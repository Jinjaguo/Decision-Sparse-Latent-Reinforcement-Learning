#!/usr/bin/env python
"""Apply the frozen EXP3 statistical plan and create all required plots."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from decision_sparse_rl.logging.run_directory import write_json  # noqa: E402
from decision_sparse_rl.metrics.criticality import aggregate_interventions, concentration_metrics  # noqa: E402


def rho(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or np.all(np.asarray(x) == x[0]) or np.all(np.asarray(y) == y[0]): return 0.0
    value = float(spearmanr(x, y).correlation)
    return 0.0 if not np.isfinite(value) else value


def group(rows: List[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    result: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows: result.setdefault(tuple(row[key] for key in keys), []).append(row)
    return result


def save_table(rows: List[Dict[str, Any]], path: Path) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def bh_adjust(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values); adjusted = np.empty_like(values); running = 1.0
    for rank_index in range(len(values) - 1, -1, -1):
        original = order[rank_index]; rank = rank_index + 1
        running = min(running, float(values[original] * len(values) / rank)); adjusted[original] = running
    return adjusted.tolist()


def adjusted_event_test(payload: List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]], rng: np.random.Generator, resamples: int = 4000) -> Tuple[float, float]:
    """Event coefficient controlling horizon, phase, and demo (therefore task)."""
    demo_names = [f"{task}|{episode}" for task, episode, _, _, _ in payload]
    unique_demos = sorted(set(demo_names))
    y_parts = []; base_parts = []; event_parts = []
    for (task, episode, times, values, mask), demo in zip(payload, demo_names):
        phase_middle = ((times >= 1/3) & (times < 2/3)).astype(float)
        phase_late = (times >= 2/3).astype(float)
        demo_fixed = np.asarray([[float(demo == name) for name in unique_demos[1:]] for _ in times])
        base_parts.append(np.column_stack([np.ones(len(times)), 1.0 - times, phase_middle, phase_late, demo_fixed]))
        y_parts.append(np.log(np.maximum(values, 1e-15))); event_parts.append(mask.astype(float))
    y = np.concatenate(y_parts); base = np.vstack(base_parts); observed_event = np.concatenate(event_parts)
    def coefficient(event: np.ndarray) -> float:
        return float(np.linalg.lstsq(np.column_stack([base, event]), y, rcond=None)[0][-1])
    observed = coefficient(observed_event); null = []
    for _ in range(resamples):
        permuted = np.concatenate([rng.permutation(mask.astype(float)) for _, _, _, _, mask in payload])
        null.append(coefficient(permuted))
    p = float((1 + sum(value >= observed for value in null)) / (1 + len(null)))
    return observed, p


def demo_label(task: str, episode: str) -> str:
    short = {"open_the_middle_drawer_of_the_cabinet": "drawer", "turn_on_the_stove": "stove", "put_the_bowl_on_the_plate": "bowl"}[task]
    return f"{short}/{episode}"


def bootstrap_top20(interventions: List[Dict[str, Any]], resamples: int) -> List[float]:
    rng = np.random.default_rng(830031)
    tasks = sorted({r["task"] for r in interventions})
    by_demo = group(interventions, ("task", "episode"))
    estimates = []
    for _ in range(resamples):
        demo_stats = []
        for sampled_task in rng.choice(tasks, len(tasks), replace=True):
            demos = sorted(k[1] for k in by_demo if k[0] == sampled_task)
            for sampled_demo in rng.choice(demos, len(demos), replace=True):
                rows = by_demo[(sampled_task, sampled_demo)]
                branch_values = []
                for _, branch_rows in group(rows, ("branch_time",)).items():
                    by_direction = group(branch_rows, ("direction_index",))
                    sampled = []
                    direction_ids = sorted(k[0] for k in by_direction)
                    for direction in rng.choice(direction_ids, 4, replace=True):
                        signed = by_direction[(int(direction),)]
                        indexes = rng.integers(0, len(signed), 2)
                        sampled.extend(float(signed[i]["primary_remaining_horizon_mean"]) for i in indexes)
                    branch_values.append(float(np.median(sampled)))
                demo_stats.append(concentration_metrics(branch_values)["top20_mass"])
        estimates.append(float(np.median(demo_stats)))
    return estimates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(); run_dir = args.run_dir.resolve(); artifacts = run_dir / "artifacts"
    manifest_dir = artifacts / "frozen_manifests"
    sap = json.loads((manifest_dir / "statistical_analysis_plan.json").read_text())
    decision_rule = json.loads((manifest_dir / "scientific_decision_rule.json").read_text())
    event_manifest = json.loads((manifest_dir / "event_manifest.json").read_text())
    primary_spec = json.loads((manifest_dir / "primary_metric_spec.json").read_text())
    interventions = pq.read_table(artifacts / "interventions.parquet").to_pylist()
    steps = pq.read_table(artifacts / "per_step_effects.parquet").to_pylist()
    if len(interventions) != 864: raise RuntimeError("analysis requires all 864 interventions")

    # Per-intervention component means, then frozen eight-way branch aggregation.
    component_names = ["normalized_arm_q", "normalized_arm_qvel", "normalized_eef_position", "normalized_eef_orientation", "normalized_task_object_position", "normalized_task_object_orientation"]
    step_groups = group(steps, ("intervention_id",))
    component_by_intervention = {key[0]: {name: float(np.mean([x[name] for x in rows])) for name in component_names} for key, rows in step_groups.items()}
    branch_rows = []
    for key, rows in group(interventions, ("task", "episode", "branch_time")).items():
        if len(rows) != 8: raise RuntimeError(f"branch {key} has {len(rows)} rather than 8 interventions")
        stats = aggregate_interventions([x["primary_remaining_horizon_mean"] for x in rows])
        first = rows[0]
        record = {"task": key[0], "episode": key[1], "branch_time": key[2], "branch_normalized_time": first["branch_normalized_time"], "branch_kind": first["branch_kind"], "remaining_horizon": first["remaining_horizon"],
                  **{f"primary_{name}": value for name, value in stats.items()},
                  "fraction_interventions_above_threshold": float(np.mean([x["primary_remaining_horizon_mean"] > primary_spec["meaningful_effect_threshold"] for x in rows])),
                  "terminal_object_effect_median": float(np.median([x["terminal_object_position_l2"] + x["terminal_object_orientation_geodesic_mean"] / np.pi for x in rows])),
                  "predicate_divergence_fraction_median": float(np.median([x["predicate_divergence_fraction"] for x in rows])),
                  "success_flip_count": int(sum(x["success_flip"] for x in rows))}
        for name in component_names: record[f"{name}_median"] = float(np.median([component_by_intervention[x["intervention_id"]][name] for x in rows]))
        branch_rows.append(record)
    branch_rows.sort(key=lambda x: (x["task"], x["episode"], x["branch_normalized_time"]))
    save_table(branch_rows, artifacts / "branch_summary.parquet")

    # Demonstration concentration and direction/sign robustness.
    demo_concentration = []; robustness = []
    intervention_demo = group(interventions, ("task", "episode"))
    branch_demo = group(branch_rows, ("task", "episode"))
    for key, rows in branch_demo.items():
        ordered = sorted(rows, key=lambda x: x["branch_normalized_time"])
        metrics = concentration_metrics([x["primary_median"] for x in ordered])
        demo_concentration.append({"task": key[0], "episode": key[1], **metrics, "total_effect_mass": float(sum(x["primary_median"] for x in ordered))})
        raw = intervention_demo[key]
        curves = {}
        for direction in range(4):
            for sign in (-1, 1):
                selected = sorted([x for x in raw if x["direction_index"] == direction and x["sign"] == sign], key=lambda x: x["branch_normalized_time"])
                curves[(direction, sign)] = [x["primary_remaining_horizon_mean"] for x in selected]
        pairwise = [rho(curves[a], curves[b]) for a, b in itertools.combinations(curves, 2)]
        sign_rhos = [rho(curves[(d, -1)], curves[(d, 1)]) for d in range(4)]
        direction_curves = [[(a + b) / 2 for a, b in zip(curves[(d, -1)], curves[(d, 1)])] for d in range(4)]
        direction_rhos = [rho(direction_curves[a], direction_curves[b]) for a, b in itertools.combinations(range(4), 2)]
        robustness.append({"task": key[0], "episode": key[1], "median_pairwise_spearman": float(np.median(pairwise)), "p25_pairwise_spearman": float(np.percentile(pairwise, 25)), "p75_pairwise_spearman": float(np.percentile(pairwise, 75)),
                           "median_sign_spearman": float(np.median(sign_rhos)), "median_direction_spearman": float(np.median(direction_rhos)), "stable_at_0_5": float(np.median(pairwise)) >= 0.5})
    save_table(demo_concentration, artifacts / "demo_concentration.parquet"); save_table(robustness, artifacts / "direction_robustness.parquet")

    # Cross-demo agreement uses only the ten shared outcome-blind quantile branches.
    cross = []
    for task in sorted({x["task"] for x in branch_rows}):
        demos = sorted({x["episode"] for x in branch_rows if x["task"] == task})
        curves = {demo: [x["primary_median"] for x in sorted([r for r in branch_rows if r["task"] == task and r["episode"] == demo and r["branch_kind"] == "temporal_quantile"], key=lambda x: x["branch_normalized_time"])] for demo in demos}
        for a, b in itertools.combinations(demos, 2): cross.append({"task": task, "demo_a": a, "demo_b": b, "spearman": rho(curves[a], curves[b])})
    save_table(cross, artifacts / "cross_demo_rank_agreement.parquet")

    # Frozen event windows, with demo-stratified enrichment and within-demo permutations.
    event_results = []; rng = np.random.default_rng(830032); radius = event_manifest["event_window_normalized_radius"]
    for event_type in sorted({x["event_type"] for x in event_manifest["events"]}):
        ratios = []; demo_payload = []; adjusted_payload = []
        for event in [x for x in event_manifest["events"] if x["event_type"] == event_type and x["present"]]:
            rows = branch_demo[(event["task"], event["episode"])]
            values = np.asarray([x["primary_median"] for x in rows]); mask = np.asarray([abs(x["branch_normalized_time"] - event["normalized_time"]) <= radius for x in rows])
            if mask.any() and (~mask).any():
                ratio_value = float(np.mean(values[mask]) / max(np.mean(values[~mask]), 1e-15)); ratios.append(ratio_value); demo_payload.append((values, mask))
                adjusted_payload.append((event["task"], event["episode"], np.asarray([x["branch_normalized_time"] for x in rows]), values, mask))
        observed = float(np.median(ratios)) if ratios else float("nan")
        null = []
        for _ in range(int(sap["permutation"].split(";")[-1].strip().split()[2]) if False else 4000):
            perm_ratios = []
            for values, mask in demo_payload:
                shuffled = rng.permutation(values); perm_ratios.append(float(np.mean(shuffled[mask]) / max(np.mean(shuffled[~mask]), 1e-15)))
            if perm_ratios: null.append(float(np.median(perm_ratios)))
        p = float((1 + sum(x >= observed for x in null)) / (1 + len(null))) if null else float("nan")
        adjusted_beta, adjusted_p = adjusted_event_test(adjusted_payload, rng) if adjusted_payload else (float("nan"), float("nan"))
        event_results.append({"event_type": event_type, "present_demo_count": len(ratios), "median_enrichment_ratio": observed, "permutation_p_one_sided": p,
                              "adjusted_log_effect_coefficient": adjusted_beta, "adjusted_permutation_p_one_sided": adjusted_p,
                              "adjustment": "remaining-horizon linear term + early/middle/late phase + demo fixed effects (therefore task controlled); event labels permuted within demo"})
    raw_q = bh_adjust([x["permutation_p_one_sided"] for x in event_results]); adjusted_q = bh_adjust([x["adjusted_permutation_p_one_sided"] for x in event_results])
    for row, q_raw, q_adjusted in zip(event_results, raw_q, adjusted_q):
        row["permutation_q_bh"] = q_raw; row["adjusted_permutation_q_bh"] = q_adjusted
    save_table(event_results, artifacts / "event_enrichment.parquet")

    # Cluster-aware primary CI, outcome relevance, LODO, and frozen decision rule.
    bootstrap = bootstrap_top20(interventions, 4000)
    top20_values = [x["top20_mass"] for x in demo_concentration]
    top20_median = float(np.median(top20_values)); top20_ci = [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))]
    task_agreement = {task: float(np.median([x["spearman"] for x in cross if x["task"] == task])) for task in sorted({x["task"] for x in cross})}
    stable_count = sum(x["stable_at_0_5"] for x in robustness)
    lodo = []
    for omitted in [(x["task"], x["episode"]) for x in demo_concentration]:
        values = [x["top20_mass"] for x in demo_concentration if (x["task"], x["episode"]) != omitted]
        lodo.append({"omitted_task": omitted[0], "omitted_episode": omitted[1], "median_top20_mass": float(np.median(values))})
    save_table(lodo, artifacts / "leave_one_demo_out.parquet")
    terminal_rho = rho([x["primary_median"] for x in branch_rows], [x["terminal_object_effect_median"] for x in branch_rows])
    predicate_rho = rho([x["primary_median"] for x in branch_rows], [x["predicate_divergence_fraction_median"] for x in branch_rows])
    branch_broad_fraction = float(np.mean([x["fraction_interventions_above_threshold"] >= 0.5 for x in branch_rows]))
    intervention_saturation = float(np.mean([x["primary_remaining_horizon_mean"] > primary_spec["meaningful_effect_threshold"] for x in interventions]))
    strong = top20_median >= 0.50 and top20_ci[0] > 0.25 and sum(v >= 0.5 for v in task_agreement.values()) >= 2 and stable_count >= 6 and min(x["median_top20_mass"] for x in lodo) >= 0.45
    partial = (not strong) and ((top20_median >= 0.40 and top20_ci[0] > 0.25) or sum(v >= 0.5 for v in task_agreement.values()) == 1)
    saturation = (not strong and not partial) and intervention_saturation >= 0.95 and top20_median < 0.40
    broad = (not strong and not partial and not saturation) and top20_median < 0.40 and branch_broad_fraction >= 0.80
    classification = "strong_support" if strong else ("partial_support" if partial else ("saturation_flag" if saturation else ("broad_sensitivity" if broad else "no_support")))
    statistics = {"primary": {"demo_median_top20_mass": top20_median, "hierarchical_bootstrap_95ci": top20_ci, "uniform_null": 0.25},
                  "task_cross_demo_median_spearman": task_agreement, "stable_direction_sign_demo_count": stable_count,
                  "leave_one_demo_out_min_top20": min(x["median_top20_mass"] for x in lodo), "branch_broad_fraction": branch_broad_fraction,
                  "intervention_threshold_exceedance_fraction": intervention_saturation, "terminal_object_spearman": terminal_rho,
                  "predicate_divergence_spearman": predicate_rho, "success_flip_count": int(sum(x["success_flip"] for x in interventions)),
                  "event_results": event_results, "counts": {"interventions": len(interventions), "per_step_effects": len(steps), "branches": len(branch_rows), "demos": len(demo_concentration)}}
    scientific = {"classification": classification, "boolean_checks": {"strong_support": strong, "partial_support": partial, "saturation_flag": saturation, "broad_sensitivity": broad, "no_support": classification == "no_support"},
                  "observed_statistics": statistics, "frozen_rule": decision_rule}
    write_json(artifacts / "statistical_results.json", statistics); write_json(artifacts / "scientific_decision.json", scientific)
    for filename in ("direction_manifest.json", "event_manifest.json", "effect_normalization.json"):
        (artifacts / filename).write_bytes((manifest_dir / filename).read_bytes())

    # Required plots.
    plot_dir = artifacts / "plots"; plot_dir.mkdir(exist_ok=True)
    demos = sorted(branch_demo); labels = [demo_label(*x) for x in demos]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for ax, key in zip(axes.flat, demos):
        rows = sorted(branch_demo[key], key=lambda x: x["branch_normalized_time"]); ax.plot([x["branch_normalized_time"] for x in rows], [x["primary_median"] for x in rows], "o-"); ax.set_title(demo_label(*key)); ax.set_yscale("symlog", linthresh=1e-7)
    fig.supxlabel("Normalized branch time"); fig.supylabel("Primary criticality (median of 8)"); fig.tight_layout(); fig.savefig(plot_dir / "criticality_vs_normalized_time_per_demo.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for ax, key in zip(axes.flat, demos):
        rows = sorted(branch_demo[key], key=lambda x: x["branch_normalized_time"])
        for name in component_names: ax.plot([x["branch_normalized_time"] for x in rows], [x[f"{name}_median"] for x in rows], label=name.replace("normalized_", ""), alpha=.8)
        ax.set_title(demo_label(*key)); ax.set_yscale("symlog", linthresh=1e-7)
    axes.flat[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(plot_dir / "criticality_components_per_demo.png", dpi=170); plt.close(fig)
    x = np.arange(len(labels)); fig, ax = plt.subplots(figsize=(12, 5)); ax.errorbar(x, [r["median_pairwise_spearman"] for r in robustness], yerr=[[r["median_pairwise_spearman"]-r["p25_pairwise_spearman"] for r in robustness], [r["p75_pairwise_spearman"]-r["median_pairwise_spearman"] for r in robustness]], fmt="o"); ax.axhline(.5, ls="--", c="k"); ax.set_xticks(x, labels, rotation=45, ha="right"); ax.set_ylabel("Pairwise temporal-rank Spearman"); fig.tight_layout(); fig.savefig(plot_dir / "direction_sign_variability.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    for key in demos:
        values = np.sort([r["primary_median"] for r in branch_demo[key]])[::-1]; mass = np.cumsum(values) / max(values.sum(), 1e-15); ax.plot(np.arange(1, 13)/12, mass, label=demo_label(*key))
    ax.plot([0,1],[0,1],"k--",label="uniform"); ax.set_xlabel("Fraction of branch points"); ax.set_ylabel("Cumulative effect mass"); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(plot_dir / "effect_mass_concentration_curves.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5)); width=.25
    for i, name in enumerate(("top10_mass","top20_mass","top30_mass")): ax.bar(x+(i-1)*width,[r[name] for r in demo_concentration],width,label=name)
    ax.set_xticks(x,labels,rotation=45,ha="right"); ax.set_ylabel("Effect mass"); ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / "topk_effect_mass_by_demo.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5)); ax.bar(x-.18,[r["gini"] for r in demo_concentration],.36,label="Gini"); ax.bar(x+.18,[1-r["normalized_entropy"] for r in demo_concentration],.36,label="1 - normalized entropy"); ax.set_xticks(x,labels,rotation=45,ha="right"); ax.legend(); fig.tight_layout(); fig.savefig(plot_dir / "concentration_index_by_demo.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(np.arange(len(cross)),[r["spearman"] for r in cross]); ax.axhline(.5,ls="--",c="k"); ax.set_xticks(np.arange(len(cross)),[f"{demo_label(r['task'],r['demo_a'])}–{r['demo_b']}" for r in cross],rotation=55,ha="right",fontsize=7); ax.set_ylabel("Quantile-curve Spearman"); fig.tight_layout(); fig.savefig(plot_dir / "cross_demo_rank_agreement.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5)); ax.bar([r["event_type"] for r in event_results],[r["median_enrichment_ratio"] for r in event_results]); ax.axhline(1,ls="--",c="k"); ax.set_ylabel("Inside/outside median enrichment"); ax.tick_params(axis="x",rotation=20); fig.tight_layout(); fig.savefig(plot_dir / "event_window_enrichment.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,5)); ax.scatter([r["remaining_horizon"] for r in branch_rows],[r["primary_median"] for r in branch_rows],alpha=.7); ax.set_xlabel("Remaining horizon"); ax.set_ylabel("Primary criticality"); ax.set_yscale("symlog",linthresh=1e-7); fig.tight_layout(); fig.savefig(plot_dir / "criticality_vs_remaining_horizon.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,5)); ax.scatter([r["terminal_object_effect_median"] for r in branch_rows],[r["primary_median"] for r in branch_rows],alpha=.7); ax.set_xlabel("Terminal object effect"); ax.set_ylabel("Primary criticality"); ax.set_xscale("symlog",linthresh=1e-9); ax.set_yscale("symlog",linthresh=1e-7); fig.tight_layout(); fig.savefig(plot_dir / "criticality_vs_terminal_object_effect.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,5)); ax.scatter([r["predicate_divergence_fraction_median"] for r in branch_rows],[r["primary_median"] for r in branch_rows],alpha=.7); ax.set_xlabel("Predicate-divergence fraction"); ax.set_ylabel("Primary criticality"); ax.set_yscale("symlog",linthresh=1e-7); fig.tight_layout(); fig.savefig(plot_dir / "criticality_vs_predicate_divergence.png", dpi=170); plt.close(fig)
    flips = [r for r in interventions if r["success_flip"]]; fig, ax = plt.subplots(figsize=(9,5));
    for task in sorted({r["task"] for r in interventions}):
        rows=[r for r in flips if r["task"]==task]; ax.scatter([r["branch_normalized_time"] for r in rows],[demo_label(r["task"],r["episode"]) for r in rows],label=task,alpha=.7)
    ax.set_xlabel("Normalized branch time"); ax.set_title(f"Terminal success flips (n={len(flips)})"); fig.tight_layout(); fig.savefig(plot_dir / "success_flip_locations.png", dpi=170); plt.close(fig)

    metrics_path = run_dir / "metrics.json"; metrics = json.loads(metrics_path.read_text()); metrics["analysis"] = {"status": "completed", "scientific_classification": classification, **statistics}; write_json(metrics_path, metrics)
    (run_dir / "analysis_command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(scientific, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
