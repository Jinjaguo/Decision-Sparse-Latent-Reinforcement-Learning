#!/usr/bin/env python
"""Validate EXP6 float64 GPU kernels against NumPy CPU source-of-truth."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.metrics.exp6 import antithetic_asymmetry, projector, projector_similarity, relative_discrepancy, trust_region_passes  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-id", required=True); parser.add_argument("--run-root", type=Path, default=ROOT / "runs"); args = parser.parse_args()
    run = create_run_directory(args.run_root, args.run_id); started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no automatic CPU fallback is allowed")
    torch.cuda.set_device(0); rng = np.random.Generator(np.random.PCG64(960010)); records = []
    for task_index, task in enumerate(("drawer", "bowl", "stove")):
        for radius in (0.0003125, 0.000625, 0.00125, 0.0025, 0.005):
            plus = rng.normal(size=(8, 41 + 6 * task_index)); minus = rng.normal(size=plus.shape)
            cpu = (plus[:7] - minus[:7]).T / (2.0 * radius); tp = torch.as_tensor(plus, device="cuda", dtype=torch.float64); tm = torch.as_tensor(minus, device="cuda", dtype=torch.float64); gpu_tensor = (tp[:7] - tm[:7]).T / (2.0 * radius); gpu = gpu_tensor.cpu().numpy()
            gram_cpu = cpu.T @ cpu; gram_gpu = (gpu_tensor.T @ gpu_tensor).cpu().numpy(); eval_cpu, vec_cpu = np.linalg.eigh(gram_cpu); eval_gpu, vec_gpu = torch.linalg.eigh(gpu_tensor.T @ gpu_tensor); eval_gpu = eval_gpu.cpu().numpy(); vec_gpu = vec_gpu.cpu().numpy()
            order_cpu = np.argsort(eval_cpu)[::-1]; order_gpu = np.argsort(eval_gpu)[::-1]; eval_cpu = eval_cpu[order_cpu]; eval_gpu = eval_gpu[order_gpu]; vec_cpu = vec_cpu[:, order_cpu]; vec_gpu = vec_gpu[:, order_gpu]
            p1_cpu = projector(vec_cpu, 1); p1_gpu = projector(vec_gpu, 1); p2_cpu = projector(vec_cpu, 2); p2_gpu = projector(vec_gpu, 2)
            asym_cpu = antithetic_asymmetry(plus[7], minus[7]); asym_gpu = float((torch.linalg.vector_norm(tp[7] + tm[7]) / (torch.linalg.vector_norm(tp[7]) + torch.linalg.vector_norm(tm[7]) + 1e-12)).cpu())
            held_cpu = (plus[7] - minus[7]) / (2 * radius); coefficient = rng.normal(size=7); pred_cpu = cpu @ coefficient; held_gpu = ((tp[7] - tm[7]) / (2 * radius)); pred_gpu = gpu_tensor @ torch.as_tensor(coefficient, device="cuda", dtype=torch.float64)
            records.append({"task": task, "radius": radius, "operator_max_abs": float(np.max(np.abs(cpu - gpu))), "gram_max_abs": float(np.max(np.abs(gram_cpu - gram_gpu))), "eigenvalue_max_abs": float(np.max(np.abs(eval_cpu - eval_gpu))), "top1_projector_max_abs": float(np.max(np.abs(p1_cpu - p1_gpu))), "top2_projector_max_abs": float(np.max(np.abs(p2_cpu - p2_gpu))), "asymmetry_abs": abs(asym_cpu - asym_gpu), "heldout_max_abs": float(np.max(np.abs(held_cpu - held_gpu.cpu().numpy()))), "prediction_max_abs": float(np.max(np.abs(pred_cpu - pred_gpu.cpu().numpy()))), "identity_top1_similarity": projector_similarity(p1_gpu, p1_gpu, 1), "spectral_self_discrepancy": relative_discrepancy(float(np.sqrt(eval_gpu[0])), float(np.sqrt(eval_gpu[0])))})
    sample = rng.normal(size=2048); bootstrap_cpu = np.quantile(sample, [0.025, 0.975]); bootstrap_gpu = torch.quantile(torch.as_tensor(sample, device="cuda", dtype=torch.float64), torch.tensor([0.025, 0.975], device="cuda", dtype=torch.float64)).cpu().numpy(); perm_cpu = int(np.sum(sample >= 0)); perm_gpu = int(torch.sum(torch.as_tensor(sample, device="cuda") >= 0).cpu())
    trust_cpu = trust_region_passes(0.81, 0.76, 0.19, 0.24, 0.34, 101.0); trust_gpu = trust_region_passes(*[float(value.cpu()) for value in torch.tensor([0.81, 0.76, 0.19, 0.24, 0.34, 101.0], device="cuda", dtype=torch.float64)])
    criteria = {"operator": max(row["operator_max_abs"] for row in records) <= 1e-11, "gram": max(row["gram_max_abs"] for row in records) <= 1e-8, "eigenvalues": max(row["eigenvalue_max_abs"] for row in records) <= 1e-8, "projectors": max(max(row["top1_projector_max_abs"], row["top2_projector_max_abs"]) for row in records) <= 1e-10, "asymmetry": max(row["asymmetry_abs"] for row in records) <= 1e-12, "heldout": max(max(row["heldout_max_abs"], row["prediction_max_abs"]) for row in records) <= 1e-11, "bootstrap": float(np.max(np.abs(bootstrap_cpu - bootstrap_gpu))) <= 1e-12, "permutation": perm_cpu == perm_gpu, "trust_region": trust_cpu == trust_gpu, "float64": True}
    audit = {"schema_version": 1, "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0), "vram_bytes": torch.cuda.get_device_properties(0).total_memory, "dtype": "float64", "records": records, "bootstrap_max_abs": float(np.max(np.abs(bootstrap_cpu - bootstrap_gpu))), "permutation_count_cpu": perm_cpu, "permutation_count_gpu": perm_gpu, "criteria": criteria, "passed": all(criteria.values())}
    write_json(run / "artifacts/gpu_audit.json", audit); write_json(run / "artifacts/gpu_cpu_equivalence.json", audit); metrics = {"run_id": args.run_id, "status": "completed", "gate": {"passed": audit["passed"], "criteria": criteria}, "gpu": audit["gpu_name"], "radius_task_cases": len(records), "wall_time_seconds": time.perf_counter() - started}
    write_run_record(run, config={"stage": "EXP6 GPU/CPU equivalence"}, command=shlex.join([sys.executable, *sys.argv]), environment=audit, git_state={"project": git_record(ROOT)}, metrics=metrics, stdout=json.dumps(metrics), stderr=""); print(json.dumps(metrics, indent=2)); return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
