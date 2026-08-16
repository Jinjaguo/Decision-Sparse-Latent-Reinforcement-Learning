"""EXP_R4 offline baselines for the corrected EXP_R3 candidate matrix.

The simulator is intentionally not imported here.  All model inputs are
checked to be pre-outcome fields, while success and safety are joined only as
training/evaluation labels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
TASKS = (
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_plate",
    "turn_on_the_stove",
)
ROUTES = (
    "D_physical_chunk",
    "V0_default70_soft_goal",
    "V1_default110_soft_goal",
    "V2_soft_diverse_control",
    "V3_goal_guarded_stove",
    "V4_response_soft_goal",
    "V5_phase_risk_stove",
    "V6_soft_default_goal",
    "V7_medoid_goal_soft",
)
FORBIDDEN_INPUT_FIELDS = {
    "success",
    "safety_stop",
    "absolute_200_exceeded",
    "predicate_after_execution",
    "physical_progress",
    "terminal_contact_mode_json",
    "terminal_object_positions",
}


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def one_hot(value: str, vocabulary: tuple[str, ...]) -> list[float]:
    return [float(value == item) for item in vocabulary]


def episode_number(value: str) -> int:
    return int(value.rsplit("_", 1)[-1])


def split_for_episode(value: str) -> str:
    # Fixed demo-level split, identical for every task and independent of
    # outcomes: demos 0--5 train, 6--7 validation, 8--9 test.
    number = episode_number(value)
    if number <= 5:
        return "train"
    if number <= 7:
        return "validation"
    return "test"


def parse_spec(row: dict) -> dict:
    return json.loads(row["candidate_spec_json"])


def flatten_action(value, width: int = 10, dim: int = 7) -> list[float]:
    if value is None:
        return [0.0] * (width * dim) + [0.0]
    array = np.asarray(value, dtype=np.float64).reshape(width, dim)
    return array.reshape(-1).tolist() + [1.0]


def pad_numeric(value, width: int) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size > width:
        raise ValueError(f"object feature has {array.size} values, exceeds fixed width {width}")
    return np.pad(array, (0, width - array.size), mode="constant").tolist()


def retrieval_features(row: dict) -> list[float]:
    indices = np.asarray(row.get("retrieved_indices", []), dtype=np.float64)
    if indices.size == 0:
        index_stats = [0.0, 0.0, 0.0, 0.0]
    else:
        index_stats = [
            float(indices.mean()),
            float(indices.min()),
            float(indices.max()),
            float(indices.size),
        ]
    return [
        float(row.get("retrieval_progress") or 0.0),
        *index_stats,
    ]


def feature_vector(row: dict, *, include_action: bool) -> list[float]:
    spec = parse_spec(row)
    object_positions = np.asarray(row["pre_action_object_positions"], dtype=np.float64).reshape(-1)
    object_quaternions = np.asarray(row["pre_action_object_quaternions"], dtype=np.float64).reshape(-1)
    eef = np.asarray(row["pre_action_eef_position"], dtype=np.float64).reshape(-1)
    force = np.asarray(row["pre_action_ee_force"], dtype=np.float64).reshape(-1)
    numeric_spec = [
        float(spec.get("advance", 0.0)),
        float(spec.get("k", 0.0)),
        float(spec.get("max_steps", 0.0)),
        float(spec.get("replan", 0.0)),
        float(spec.get("retarget", 0.0)),
        float(spec.get("smooth", 0.0)),
        float(bool(spec.get("monotone", False))),
    ]
    categorical_spec = one_hot(str(spec.get("aggregate", "")), ("nearest", "medoid", "risk_weighted", ""))
    categorical_spec += one_hot(str(spec.get("view", "")), ("physical", "response", "goal", ""))
    vector = [
        *one_hot(row["task"], TASKS),
        *one_hot(row["route"], ROUTES),
        *eef.tolist(),
        *pad_numeric(object_positions, 6),
        *pad_numeric(object_quaternions, 8),
        *force.tolist(),
        float(bool(row["pre_action_force_valid"])),
        float(bool(row["pre_action_predicate"])),
        float(bool(row["predicate_already_true"])),
        float(bool(row["executed"])),
        *retrieval_features(row),
        *numeric_spec,
        *categorical_spec,
    ]
    if include_action:
        vector += flatten_action(row.get("requested_action"))
        action = row.get("executed_action")
        vector += (np.asarray(action, dtype=np.float64).reshape(-1).tolist() if action is not None else [0.0] * 7)
    result = np.asarray(vector, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"non-finite admissible feature for {row['branch_id']} {row['route']}")
    return result.tolist()


def retrieval_score(row: dict) -> float:
    """A label-free retrieval-only score, with deterministic route tie-breaks."""

    # The score is available before the candidate outcome.  Route order is a
    # fixed protocol tie-break, never an outcome-derived route prior.
    return float(row.get("retrieval_progress") or 0.0) + 1e-9 * (len(ROUTES) - ROUTES.index(row["route"]))


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def fit_model(x_train: np.ndarray, y_train: np.ndarray, output_dim: int, device: torch.device, seed: int) -> tuple[MLP, dict]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = MLP(x_train.shape[1], output_dim).to(device)
    x = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    positive = y.sum(axis=0)
    negative = y.shape[0] - positive
    pos_weight = torch.where(positive > 0, negative / torch.clamp(positive, min=1.0), torch.ones_like(positive))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    model.train()
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    return model, {"epochs": 300, "learning_rate": 0.01, "positive_weight": pos_weight.detach().cpu().tolist()}


def predict(model: MLP, x: np.ndarray, mean: np.ndarray, scale: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        values = model(torch.as_tensor((x - mean) / scale, dtype=torch.float32, device=device))
        return torch.sigmoid(values).detach().cpu().numpy()


def tie_aware_top_k(rows: list[dict], score_key: str, k: int) -> float:
    ordered = sorted(rows, key=lambda row: (-float(row[score_key]), row["route"]))
    total = 0.0
    remaining = k
    index = 0
    while remaining > 0 and index < len(ordered):
        score = float(ordered[index][score_key])
        end = index + 1
        while end < len(ordered) and math.isclose(float(ordered[end][score_key]), score, rel_tol=0.0, abs_tol=1e-10):
            end += 1
        group = ordered[index:end]
        take = min(remaining, len(group))
        total += take * float(sum(row["utility"] for row in group) / len(group))
        remaining -= take
        index = end
    return total / float(k)


def average_precision(rows: list[dict], score_key: str) -> float | None:
    ordered = sorted(rows, key=lambda row: (-float(row[score_key]), row["route"]))
    positives = sum(row["utility"] for row in ordered)
    if positives == 0:
        return None
    seen = 0.0
    total = 0.0
    for index, row in enumerate(ordered, start=1):
        if row["utility"]:
            seen += 1.0
            total += seen / index
    return total / positives


def grouped_metrics(rows: list[dict], score_key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["branch_id"]].append(row)
    top1, top3, aps = [], [], []
    for group in groups.values():
        top1.append(tie_aware_top_k(group, score_key, 1))
        top3.append(tie_aware_top_k(group, score_key, 3))
        value = average_precision(group, score_key)
        if value is not None:
            aps.append(value)
    return {
        "branch_count": len(groups),
        "candidate_count": len(rows),
        "tie_aware_top1_utility": float(np.mean(top1)) if top1 else None,
        "tie_aware_top3_utility": float(np.mean(top3)) if top3 else None,
        "mean_average_precision": float(np.mean(aps)) if aps else None,
        "positive_branch_support": len(aps),
    }


def bootstrap_ci(rows: list[dict], score_key: str, metric_key: str, seed: int, samples: int = 2000) -> list[float] | None:
    groups = defaultdict(list)
    for row in rows:
        groups[row["branch_id"]].append(row)
    group_values = []
    for group in groups.values():
        if metric_key == "top1":
            group_values.append(tie_aware_top_k(group, score_key, 1))
        elif metric_key == "map":
            value = average_precision(group, score_key)
            if value is not None:
                group_values.append(value)
    if not group_values:
        return None
    rng = np.random.default_rng(seed)
    values = np.asarray(group_values, dtype=np.float64)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[draws].mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-run", type=Path, default=Path("runs/exp_r3_s1_instrumented_candidate_matrix_20260816_r3"))
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    if out.exists():
        raise FileExistsError(f"immutable run exists: {out}")
    artifacts = out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    source = ROOT / args.input_run / "artifacts"
    pre_table = pq.read_table(source / "pre_outcome_candidates.parquet")
    summary_table = pq.read_table(source / "candidate_summaries.parquet")
    pre_fields = set(pre_table.schema.names)
    forbidden = sorted(pre_fields & FORBIDDEN_INPUT_FIELDS)
    if forbidden:
        raise RuntimeError(f"post-outcome fields leaked into pre-outcome table: {forbidden}")
    pre = pre_table.to_pylist()
    summaries = summary_table.to_pylist()
    if len(pre) != len(summaries) or len({row["pre_outcome_hash"] for row in pre}) != len(pre):
        raise RuntimeError("pre-outcome rows are not one-to-one and hash-unique")
    summary_by_key = {(row["branch_id"], row["route"]): row for row in summaries}
    if len(summary_by_key) != len(summaries):
        raise RuntimeError("duplicate outcome key")
    rows = []
    for row in pre:
        key = (row["branch_id"], row["route"])
        if key not in summary_by_key:
            raise RuntimeError(f"missing outcome for {key}")
        outcome = summary_by_key[key]
        enriched = dict(row)
        enriched["split"] = split_for_episode(row["episode"])
        enriched["success"] = int(bool(outcome["success"]))
        enriched["unsafe"] = int(bool(outcome["safety_stop"]))
        enriched["utility"] = int(bool(outcome["success"]) and not bool(outcome["safety_stop"]))
        enriched["retrieval_score"] = retrieval_score(row)
        rows.append(enriched)
    if {row["split"] for row in rows} != {"train", "validation", "test"}:
        raise RuntimeError("demo-level split is incomplete")
    feature_all = np.asarray([feature_vector(row, include_action=True) for row in rows], dtype=np.float32)
    train_mask = np.asarray([row["split"] == "train" for row in rows])
    mean = feature_all[train_mask].mean(axis=0)
    scale = feature_all[train_mask].std(axis=0)
    scale[scale < 1e-6] = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    seed = 20260816
    random.seed(seed)
    np.random.seed(seed)
    normalized_train = (feature_all[train_mask] - mean) / scale
    scalar, scalar_meta = fit_model(normalized_train, np.asarray([[row["utility"]] for row in rows if row["split"] == "train"], dtype=np.float32), 1, device, seed)
    factorized, factorized_meta = fit_model(normalized_train, np.asarray([[row["success"], row["unsafe"]] for row in rows if row["split"] == "train"], dtype=np.float32), 2, device, seed + 1)
    scalar_probability = predict(scalar, feature_all, mean, scale, device).reshape(-1)
    factorized_probability = predict(factorized, feature_all, mean, scale, device)
    for index, row in enumerate(rows):
        row["scalar_score"] = float(scalar_probability[index])
        row["factorized_success_probability"] = float(factorized_probability[index, 0])
        row["factorized_unsafe_probability"] = float(factorized_probability[index, 1])
        row["factorized_score"] = row["factorized_success_probability"] * (1.0 - row["factorized_unsafe_probability"])
    prediction_rows = [
        {key: row[key] for key in ("branch_id", "task", "episode", "route", "split", "success", "unsafe", "utility", "retrieval_score", "scalar_score", "factorized_score", "factorized_success_probability", "factorized_unsafe_probability")}
        for row in rows
    ]
    pq.write_table(pa.Table.from_pylist(prediction_rows), artifacts / "predictions.parquet", compression="zstd")
    protocols = {
        "experiment": "EXP_R4",
        "input_run": str(args.input_run),
        "features_are_pre_outcome_only": True,
        "forbidden_fields_detected": forbidden,
        "split": {"train": "demo_0..demo_5", "validation": "demo_6..demo_7", "test": "demo_8..demo_9"},
        "target_demo_excluded_from_retrieval": True,
        "retrieval_only_uses_outcome_labels": False,
        "scalar_target": "success_and_not_safety_stop",
        "factorized_targets": ["success", "safety_stop"],
        "bootstrap_samples": 2000,
        "seed": seed,
        "device": str(device),
    }
    dump(artifacts / "protocol.json", protocols)
    dump(artifacts / "feature_schema.json", {"dimension": int(feature_all.shape[1]), "pre_outcome_fields": sorted(pre_fields), "forbidden_fields": sorted(FORBIDDEN_INPUT_FIELDS)})
    dump(artifacts / "model_meta.json", {"scalar": scalar_meta, "factorized": factorized_meta})
    metrics = {"status": "completed", "experiment": "EXP_R4", "device": str(device), "input_rows": len(rows), "split_counts": {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")}, "models": {}}
    for model_name, score_key in (("retrieval_only", "retrieval_score"), ("scalar", "scalar_score"), ("factorized", "factorized_score")):
        metrics["models"][model_name] = {}
        for split in ("train", "validation", "test"):
            subset = [row for row in rows if row["split"] == split]
            values = grouped_metrics(subset, score_key)
            values["bootstrap_ci_top1"] = bootstrap_ci(subset, score_key, "top1", seed + len(model_name))
            values["bootstrap_ci_map"] = bootstrap_ci(subset, score_key, "map", seed + len(model_name) + 1)
            metrics["models"][model_name][split] = values
    dump(out / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
