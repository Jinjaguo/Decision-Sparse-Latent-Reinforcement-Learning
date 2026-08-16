"""Freeze the EXP_R11 factorized selector using development R3 only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.exp_r4.run_baselines import MLP, feature_vector, fit_model

def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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
    source = ROOT / args.input_run / "artifacts"
    pre = pq.read_table(source / "pre_outcome_candidates.parquet").to_pylist()
    summaries = pq.read_table(source / "candidate_summaries.parquet").to_pylist()
    outcomes = {(row["branch_id"], row["route"]): row for row in summaries}
    if len(pre) != 540 or len(outcomes) != 540:
        raise RuntimeError("development R3 matrix is incomplete")
    x = np.asarray([feature_vector(row, include_action=True) for row in pre], dtype=np.float32)
    y = np.asarray([[int(bool(outcomes[(row["branch_id"], row["route"])] ["success"])), int(bool(outcomes[(row["branch_id"], row["route"])] ["safety_stop"]))] for row in pre], dtype=np.float32)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    seed = 20260816
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, __import__("os").cpu_count() or 1))
    model, training = fit_model((x - mean) / scale, y, 2, device, seed)
    model_path = artifacts / "factorized_selector_state.pt"
    torch.save({"state_dict": model.state_dict(), "input_dim": int(x.shape[1]), "output_dim": 2}, model_path)
    np.savez(artifacts / "feature_stats.npz", mean=mean, scale=scale)
    protocol = {
        "experiment": "EXP_R11",
        "stage": "frozen_selector",
        "development_input_run": str(args.input_run),
        "development_candidate_count": len(pre),
        "target_confirmation_accessed": False,
        "target_confirmation_outcomes_accessed": False,
        "feature_dimension": int(x.shape[1]),
        "seed": seed,
        "device": str(device),
        "training": training,
        "model_state_sha256": sha_file(model_path),
    }
    dump(artifacts / "protocol.json", protocol)
    dump(out / "metrics.json", {"status": "completed", "experiment": "EXP_R11", "stage": "frozen_selector", "development_rows": len(pre), "feature_dimension": int(x.shape[1]), "model_state_sha256": protocol["model_state_sha256"]})
    print(json.dumps(protocol, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
