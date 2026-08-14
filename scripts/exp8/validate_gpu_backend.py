#!/usr/bin/env python
"""Fresh float64 CPU/GPU equivalence audit for all EXP8 analysis primitives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def compare(name, cpu, gpu, atol=1e-10, rtol=1e-8):
    left, right = np.asarray(cpu), np.asarray(gpu)
    absolute = float(np.max(np.abs(left - right)))
    scale = float(max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-300))
    return {"component": name, "absolute_error": absolute, "relative_error": absolute / scale, "passed": absolute <= atol + rtol * scale}


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def torch_median(value):
    ordered, _ = torch.sort(value, dim=0)
    count = ordered.shape[0]
    return ordered[count // 2] if count % 2 else (ordered[count // 2 - 1] + ordered[count // 2]) / 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    artifacts = args.run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; EXP8 forbids automatic fallback in the equivalence audit")
    torch.cuda.set_device(0)
    rng = np.random.default_rng(980010)
    records = []
    x = rng.normal(size=(80, 24))
    center = np.median(x, axis=0); scale = 1.4826 * np.median(np.abs(x - center), axis=0); fallback = x.std(0); scale[scale < 1e-12] = fallback[scale < 1e-12]; scale[scale < 1e-12] = 1.0
    cpu_normalized = (x - center) / scale
    tx = torch.tensor(x, dtype=torch.float64, device="cuda")
    tcenter = torch_median(tx); tscale = 1.4826 * torch_median(torch.abs(tx - tcenter)); tfallback = torch.std(tx, dim=0, unbiased=False); tscale = torch.where(tscale < 1e-12, tfallback, tscale); tscale = torch.where(tscale < 1e-12, torch.ones_like(tscale), tscale)
    gpu_normalized = ((tx - tcenter) / tscale).cpu().numpy()
    records.append(compare("contact_frame_coordinates_and_normalization", cpu_normalized, gpu_normalized))
    squared = np.sum((cpu_normalized[:, None, :] - cpu_normalized[None, :, :]) ** 2, axis=2)
    kernel = np.exp(-squared / 2.0)
    tk = torch.exp(-torch.sum((torch.tensor(cpu_normalized, dtype=torch.float64, device="cuda")[:, None, :] - torch.tensor(cpu_normalized, dtype=torch.float64, device="cuda")[None, :, :]) ** 2, dim=2) / 2.0)
    records.append(compare("kernel_matrix", kernel, tk.cpu().numpy()))
    y = rng.normal(size=(80, 12))
    alpha = 1e-2
    cpu_prediction = kernel @ np.linalg.solve(kernel + alpha * np.eye(80), y)
    ty = torch.tensor(y, dtype=torch.float64, device="cuda")
    gpu_prediction = (tk @ torch.linalg.solve(tk + alpha * torch.eye(80, dtype=torch.float64, device="cuda"), ty)).cpu().numpy()
    records.append(compare("kernel_ridge_prediction", cpu_prediction, gpu_prediction))
    operator = rng.normal(size=(42, 7))
    gram = operator.T @ operator
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    top = eigenvectors[:, -1]
    toperator = torch.tensor(operator, dtype=torch.float64, device="cuda")
    tvalues, tvectors = torch.linalg.eigh(toperator.T @ toperator)
    records.append(compare("gram_matrix", gram, (toperator.T @ toperator).cpu().numpy()))
    records.append(compare("eigenspectrum", eigenvalues, tvalues.cpu().numpy()))
    records.append(compare("top1_projector", np.outer(top, top), torch.outer(tvectors[:, -1], tvectors[:, -1]).cpu().numpy()))
    direction = rng.normal(size=7)
    records.append(compare("heldout_vector", operator @ direction, (toperator @ torch.tensor(direction, dtype=torch.float64, device="cuda")).cpu().numpy()))
    demo_values = rng.normal(size=30)
    indexes = rng.integers(0, 30, size=(500, 30))
    cpu_bootstrap = demo_values[indexes].mean(1)
    gpu_bootstrap = torch.tensor(demo_values, dtype=torch.float64, device="cuda")[torch.tensor(indexes, dtype=torch.long, device="cuda")].mean(1).cpu().numpy()
    records.append(compare("cluster_bootstrap", cpu_bootstrap, gpu_bootstrap))
    signs = rng.choice([-1.0, 1.0], size=(500, 30))
    cpu_permutation = (signs * demo_values).mean(1)
    gpu_permutation = (torch.tensor(signs, dtype=torch.float64, device="cuda") * torch.tensor(demo_values, dtype=torch.float64, device="cuda")).mean(1).cpu().numpy()
    records.append(compare("cluster_permutation", cpu_permutation, gpu_permutation))
    weights = rng.normal(size=24)
    probability = sigmoid(cpu_normalized @ weights)
    gpu_probability = torch.sigmoid(torch.tensor(cpu_normalized, dtype=torch.float64, device="cuda") @ torch.tensor(weights, dtype=torch.float64, device="cuda")).cpu().numpy()
    records.append(compare("risk_probability", probability, gpu_probability))
    labels = rng.integers(0, 2, size=80)
    ids = np.minimum((probability * 10).astype(int), 9)
    cpu_ece = sum(np.mean(ids == index) * abs(np.mean(labels[ids == index]) - np.mean(probability[ids == index])) for index in np.unique(ids))
    tprob = torch.tensor(probability, dtype=torch.float64, device="cuda")
    tlabels = torch.tensor(labels, dtype=torch.float64, device="cuda")
    tids = torch.clamp((tprob * 10).long(), max=9)
    gpu_ece = torch.zeros((), dtype=torch.float64, device="cuda")
    for index in range(10):
        mask = tids == index
        if bool(torch.any(mask)):
            gpu_ece += mask.double().mean() * torch.abs(tlabels[mask].mean() - tprob[mask].mean())
    records.append(compare("calibration_metrics", cpu_ece, gpu_ece.cpu().numpy()))
    audit = {"schema_version": 1, "gpu_name": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda, "dtype": "float64", "tolerances": {"atol": 1e-10, "rtol": 1e-8}, "records": records, "passed": all(record["passed"] for record in records), "no_fallback": True}
    (artifacts / "gpu_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (artifacts / "gpu_cpu_equivalence.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.run_dir / "metrics.json").write_text(json.dumps({"gate": {"passed": audit["passed"]}, "gpu": audit["gpu_name"], "cases": len(records)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
