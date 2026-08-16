"""EXP_R6 factorized-head and input ablations under leave-one-demo-out folds."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.exp_r4.run_baselines import (  # noqa: E402
    FORBIDDEN_INPUT_FIELDS,
    bootstrap_ci,
    feature_vector,
    fit_model,
    grouped_metrics,
    predict,
    retrieval_score,
)
from scripts.exp_r5.run_leave_one_demo_out import build_rows  # noqa: E402


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def variant_features(row: dict, variant: str) -> list[float]:
    values = np.asarray(feature_vector(row, include_action=variant != "factorized_no_action"), dtype=np.float32)
    if variant == "factorized_no_route":
        # feature_vector layout starts with 3 task one-hot entries and 9 route
        # one-hot entries; remove route identity without changing dimensions.
        values[3:12] = 0.0
    return values.tolist()


def score_from_probability(probability: np.ndarray, variant: str) -> float:
    if variant == "scalar_full":
        return float(probability[0])
    if variant == "success_only":
        return float(probability[0])
    if variant == "safety_only":
        return float(1.0 - probability[0])
    return float(probability[0] * (1.0 - probability[1]))


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
    forbidden = sorted(pre_fields & FORBIDDEN_INPUT_FIELDS)
    if forbidden:
        raise RuntimeError(f"forbidden pre-outcome fields: {forbidden}")
    demos = [f"demo_{index}" for index in range(10)]
    variants = ("scalar_full", "factorized_full", "factorized_no_route", "factorized_no_action", "success_only", "safety_only")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    seed = 20260816
    random.seed(seed)
    np.random.seed(seed)
    predictions = []
    for variant_index, variant in enumerate(variants):
        variant_rows = []
        for fold_index, held_demo in enumerate(demos):
            train = [row for row in rows if row["episode"] != held_demo]
            test = [row for row in rows if row["episode"] == held_demo]
            x_train = np.asarray([variant_features(row, variant) for row in train], dtype=np.float32)
            x_test = np.asarray([variant_features(row, variant) for row in test], dtype=np.float32)
            mean = x_train.mean(axis=0)
            scale = x_train.std(axis=0)
            scale[scale < 1e-6] = 1.0
            normalized_train = (x_train - mean) / scale
            if variant == "scalar_full":
                targets = np.asarray([[row["utility"]] for row in train], dtype=np.float32)
                output_dim = 1
            elif variant == "success_only":
                targets = np.asarray([[row["success"]] for row in train], dtype=np.float32)
                output_dim = 1
            elif variant == "safety_only":
                targets = np.asarray([[row["unsafe"]] for row in train], dtype=np.float32)
                output_dim = 1
            else:
                targets = np.asarray([[row["success"], row["unsafe"]] for row in train], dtype=np.float32)
                output_dim = 2
            model, _ = fit_model(normalized_train, targets, output_dim, device, seed + variant_index * 100 + fold_index)
            probabilities = predict(model, x_test, mean, scale, device)
            for index, row in enumerate(test):
                item = {
                    "variant": variant,
                    "held_out_demo": held_demo,
                    "branch_id": row["branch_id"],
                    "task": row["task"],
                    "episode": row["episode"],
                    "route": row["route"],
                    "success": row["success"],
                    "unsafe": row["unsafe"],
                    "utility": row["utility"],
                    "score": score_from_probability(probabilities[index], variant),
                }
                variant_rows.append(item)
        predictions.extend(variant_rows)
    pq.write_table(pa.Table.from_pylist(predictions), artifacts / "predictions.parquet", compression="zstd")
    metrics = {"status": "completed", "experiment": "EXP_R6", "device": str(device), "input_rows": len(rows), "variants": {}}
    for variant in variants:
        variant_rows = [row for row in predictions if row["variant"] == variant]
        metrics["variants"][variant] = grouped_metrics(variant_rows, "score")
        metrics["variants"][variant]["bootstrap_ci_top1"] = bootstrap_ci(variant_rows, "score", "top1", seed + variants.index(variant))
        metrics["variants"][variant]["bootstrap_ci_map"] = bootstrap_ci(variant_rows, "score", "map", seed + 100 + variants.index(variant))
        metrics["variants"][variant]["by_task"] = {}
        for task in sorted({row["task"] for row in variant_rows}):
            subset = [row for row in variant_rows if row["task"] == task]
            metrics["variants"][variant]["by_task"][task] = grouped_metrics(subset, "score")
    dump(artifacts / "protocol.json", {
        "experiment": "EXP_R6",
        "input_run": str(args.input_run),
        "folds": "demo_0..demo_9 leave-one-demo-out",
        "variants": list(variants),
        "features_are_pre_outcome_only": True,
        "forbidden_fields_detected": forbidden,
        "target_demo_excluded_from_fit_and_normalization": True,
        "device": str(device),
        "seed": seed,
        "bootstrap_samples": 2000,
    })
    dump(out / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
