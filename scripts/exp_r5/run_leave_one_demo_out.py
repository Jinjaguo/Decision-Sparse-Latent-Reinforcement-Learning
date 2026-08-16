"""EXP_R5 leave-one-demo-out robustness audit for EXP_R4 baselines."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.exp_r4.run_baselines import (
    FORBIDDEN_INPUT_FIELDS,
    TASKS,
    bootstrap_ci,
    feature_vector,
    fit_model,
    grouped_metrics,
    predict,
    retrieval_score,
    split_for_episode,
)


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def build_rows(source: Path) -> tuple[list[dict], set[str]]:
    pre_table = pq.read_table(source / "pre_outcome_candidates.parquet")
    summary_table = pq.read_table(source / "candidate_summaries.parquet")
    pre_fields = set(pre_table.schema.names)
    forbidden = sorted(pre_fields & FORBIDDEN_INPUT_FIELDS)
    if forbidden:
        raise RuntimeError(f"post-outcome fields leaked into pre-outcome table: {forbidden}")
    pre = pre_table.to_pylist()
    summaries = summary_table.to_pylist()
    if len(pre) != len(summaries) or len({row["pre_outcome_hash"] for row in pre}) != len(pre):
        raise RuntimeError("pre-outcome table is not hash-unique and one-to-one")
    outcomes = {(row["branch_id"], row["route"]): row for row in summaries}
    if len(outcomes) != len(summaries):
        raise RuntimeError("duplicate outcome key")
    rows = []
    for row in pre:
        outcome = outcomes[(row["branch_id"], row["route"])]
        item = dict(row)
        item["success"] = int(bool(outcome["success"]))
        item["unsafe"] = int(bool(outcome["safety_stop"]))
        item["utility"] = int(bool(outcome["success"]) and not bool(outcome["safety_stop"]))
        item["retrieval_score"] = retrieval_score(row)
        item["demo"] = row["episode"]
        rows.append(item)
    return rows, pre_fields


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-run", type=Path, default=Path("runs/exp_r3_s1_instrumented_candidate_matrix_20260816_r4"))
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    if out.exists():
        raise FileExistsError(f"immutable run exists: {out}")
    artifacts = out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    rows, pre_fields = build_rows(ROOT / args.input_run / "artifacts")
    demos = sorted({row["demo"] for row in rows}, key=lambda value: int(value.rsplit("_", 1)[-1]))
    if demos != [f"demo_{index}" for index in range(10)]:
        raise RuntimeError(f"unexpected demo cohort: {demos}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, __import__("os").cpu_count() or 1))
    seed = 20260816
    random.seed(seed)
    np.random.seed(seed)
    all_predictions = []
    fold_metrics = {}
    for fold_index, held_demo in enumerate(demos):
        train = [row for row in rows if row["demo"] != held_demo]
        test = [row for row in rows if row["demo"] == held_demo]
        x_train = np.asarray([feature_vector(row, include_action=True) for row in train], dtype=np.float32)
        x_test = np.asarray([feature_vector(row, include_action=True) for row in test], dtype=np.float32)
        mean = x_train.mean(axis=0)
        scale = x_train.std(axis=0)
        scale[scale < 1e-6] = 1.0
        normalized_train = (x_train - mean) / scale
        scalar, _ = fit_model(normalized_train, np.asarray([[row["utility"]] for row in train], dtype=np.float32), 1, device, seed + fold_index)
        factorized, _ = fit_model(normalized_train, np.asarray([[row["success"], row["unsafe"]] for row in train], dtype=np.float32), 2, device, seed + 100 + fold_index)
        scalar_probability = predict(scalar, x_test, mean, scale, device).reshape(-1)
        factorized_probability = predict(factorized, x_test, mean, scale, device)
        for index, row in enumerate(test):
            item = {key: row[key] for key in ("branch_id", "task", "episode", "route", "success", "unsafe", "utility")}
            item.update({
                "held_out_demo": held_demo,
                "retrieval_score": float(row["retrieval_score"]),
                "scalar_score": float(scalar_probability[index]),
                "factorized_success_probability": float(factorized_probability[index, 0]),
                "factorized_unsafe_probability": float(factorized_probability[index, 1]),
                "factorized_score": float(factorized_probability[index, 0] * (1.0 - factorized_probability[index, 1])),
            })
            all_predictions.append(item)
        fold_metrics[held_demo] = {}
        for model_name, score_key in (("retrieval_only", "retrieval_score"), ("scalar", "scalar_score"), ("factorized", "factorized_score")):
            values = grouped_metrics([item for item in all_predictions if item["held_out_demo"] == held_demo], score_key)
            values["bootstrap_ci_top1"] = bootstrap_ci([item for item in all_predictions if item["held_out_demo"] == held_demo], score_key, "top1", seed + fold_index)
            values["bootstrap_ci_map"] = bootstrap_ci([item for item in all_predictions if item["held_out_demo"] == held_demo], score_key, "map", seed + 100 + fold_index)
            fold_metrics[held_demo][model_name] = values
    pq.write_table(pa.Table.from_pylist(all_predictions), artifacts / "predictions.parquet", compression="zstd")
    aggregate = {}
    for model_name, score_key in (("retrieval_only", "retrieval_score"), ("scalar", "scalar_score"), ("factorized", "factorized_score")):
        aggregate[model_name] = grouped_metrics(all_predictions, score_key)
        aggregate[model_name]["bootstrap_ci_top1"] = bootstrap_ci(all_predictions, score_key, "top1", seed)
        aggregate[model_name]["bootstrap_ci_map"] = bootstrap_ci(all_predictions, score_key, "map", seed + 1)
    protocol = {
        "experiment": "EXP_R5",
        "input_run": str(args.input_run),
        "features_are_pre_outcome_only": True,
        "forbidden_fields_detected": sorted(pre_fields & FORBIDDEN_INPUT_FIELDS),
        "folds": "one held-out demo across all three tasks; six branches per fold",
        "target_demo_excluded_from_fit_and_normalization": True,
        "device": str(device),
        "seed": seed,
        "bootstrap_samples": 2000,
    }
    dump(artifacts / "protocol.json", protocol)
    dump(out / "metrics.json", {"status": "completed", "experiment": "EXP_R5", "device": str(device), "rows": len(all_predictions), "fold_count": len(demos), "aggregate": aggregate, "folds": fold_metrics})
    print(json.dumps({"status": "completed", "device": str(device), "aggregate": aggregate}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
