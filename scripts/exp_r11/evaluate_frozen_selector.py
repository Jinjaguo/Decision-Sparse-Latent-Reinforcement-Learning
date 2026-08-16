"""Evaluate the frozen development selector on the EXP26 confirmation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.exp_r4.run_baselines import MLP, feature_vector

def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def selector_stats(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["branch_id"]].append(row)
    selected = [sorted(group, key=lambda row: (-float(row["factorized_score"]), row["route"]))[0] for group in groups.values()]
    return {
        "branch_count": len(selected),
        "success_rate": float(np.mean([row["success"] for row in selected])),
        "safe_success_rate": float(np.mean([row["utility"] for row in selected])),
        "safety_stop_rate": float(np.mean([row["unsafe"] for row in selected])),
        "route_frequency": {route: sum(row["route"] == route for row in selected) for route in sorted({row["route"] for row in selected})},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--confirmation-run", type=Path, default=Path("runs/exp_r11_s2_confirmation_matrix_20260816_r2"))
    parser.add_argument("--frozen-run", type=Path, default=Path("runs/exp_r11_s0_frozen_factorized_selector_20260816_r2"))
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    if out.exists():
        raise FileExistsError(f"immutable run exists: {out}")
    artifacts = out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    confirmation = ROOT / args.confirmation_run / "artifacts"
    frozen = ROOT / args.frozen_run / "artifacts"
    pre_table = pq.read_table(confirmation / "pre_outcome_candidates.parquet")
    summary_rows = pq.read_table(confirmation / "candidate_summaries.parquet").to_pylist()
    pre = pre_table.to_pylist()
    if len(pre) != 216 or len(summary_rows) != 216:
        raise RuntimeError("confirmation matrix is incomplete")
    forbidden = {"success", "safety_stop", "predicate_after_execution", "physical_progress", "terminal_object_positions"} & set(pre_table.schema.names)
    if forbidden:
        raise RuntimeError(f"confirmation pre-outcome leakage: {sorted(forbidden)}")
    outcomes = {(row["branch_id"], row["route"]): row for row in summary_rows}
    stats = np.load(frozen / "feature_stats.npz")
    mean, scale = np.asarray(stats["mean"], dtype=np.float32), np.asarray(stats["scale"], dtype=np.float32)
    checkpoint = torch.load(frozen / "factorized_selector_state.pt", map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(int(checkpoint["input_dim"]), int(checkpoint["output_dim"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    x = np.asarray([feature_vector(row, include_action=True) for row in pre], dtype=np.float32)
    if x.shape[1] != len(mean):
        raise RuntimeError("confirmation feature dimension differs from frozen selector")
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.as_tensor((x - mean) / scale, dtype=torch.float32, device=device))).cpu().numpy()
    rows = []
    for index, row in enumerate(pre):
        outcome = outcomes[(row["branch_id"], row["route"])]
        rows.append({
            "branch_id": row["branch_id"], "task": row["task"], "episode": row["episode"], "route": row["route"],
            "success": int(bool(outcome["success"])), "unsafe": int(bool(outcome["safety_stop"])),
            "utility": int(bool(outcome["success"]) and not bool(outcome["safety_stop"])),
            "factorized_success_probability": float(probabilities[index, 0]),
            "factorized_unsafe_probability": float(probabilities[index, 1]),
            "factorized_score": float(probabilities[index, 0] * (1.0 - probabilities[index, 1])),
        })
    pq.write_table(pa.Table.from_pylist(rows), artifacts / "predictions.parquet", compression="zstd")
    by_branch = defaultdict(list)
    for row in rows:
        by_branch[row["branch_id"]].append(row)
    selected = [sorted(group, key=lambda row: (-row["factorized_score"], row["route"]))[0] for group in by_branch.values()]
    defaults = [sorted(group, key=lambda row: (row["route"] != "D_physical_chunk"))[0] for group in by_branch.values()]
    oracles = [max(group, key=lambda row: (row["utility"], -row["unsafe"], -row["factorized_score"])) for group in by_branch.values()]
    result = {
        "status": "completed", "experiment": "EXP_R11", "confirmation_branch_count": len(by_branch), "candidate_count": len(rows),
        "zero_gate_passed": json.loads((ROOT / args.confirmation_run / "metrics.json").read_text())["zero_gate_passed"],
        "frozen_model_state_sha256": sha_file(frozen / "factorized_selector_state.pt"),
        "selected": selector_stats(rows),
        "default_D": selector_stats(defaults),
        "oracle": selector_stats(oracles),
        "target_future_access": False,
        "threshold_tuned_on_confirmation": False,
    }
    dump(artifacts / "protocol.json", {"experiment": "EXP_R11", "confirmation_run": str(args.confirmation_run), "frozen_run": str(args.frozen_run), "forbidden_pre_outcome_fields": sorted(forbidden), "threshold": "none", "target_future_access": False})
    dump(out / "metrics.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
