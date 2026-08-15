"""Cross-fitted EXP10 Stage-A multi-route comparison on locked retrospective data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from decision_sparse_rl.metrics.exp10 import interval_coverage, one_hot
from decision_sparse_rl.metrics.exp9 import binary_metrics, energy_score, rank_auc, select_specificity_threshold


SEEDS = [101001, 101002, 101003]
HORIZONS = ["h5", "h10"]
METHODS = [
    "baseline_b", "exp9_temporal", "no_action", "no_history",
    "A_chunk5", "A_chunk10", "A_chunk20", "A_phase_chunk",
    "A_history1", "A_history5", "A_history10", "A_phase_history",
    "B_hard_phase_moe", "B_soft_phase_moe", "B_hybrid_phase", "B_shuffled_phase",
    "C_latent0", "C_latent16", "C_latent32",
    "F_deterministic", "F_heteroscedastic", "F_residual_mixture", "F_cvae16",
]
TRACK = {m: m.split("_", 1)[0] if m[0] in "ABCDEF" and m[1:2] == "_" else "common" for m in METHODS}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def project(x: np.ndarray, width: int, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[1] <= width:
        return x
    rng = np.random.default_rng(seed + x.shape[1] * 17 + width)
    matrix = rng.choice([-1.0, 1.0], size=(x.shape[1], width)) / math.sqrt(width)
    return x @ matrix


def standardize(train: np.ndarray, test: np.ndarray):
    mean, std = train.mean(0), train.std(0)
    std[std < 1e-8] = 1.0
    return (train - mean) / std, (test - mean) / std


def ridge(train_x, train_y, test_x, alpha=0.05):
    tx, vx = standardize(train_x, test_x)
    tx = np.column_stack([tx, np.ones(len(tx))])
    vx = np.column_stack([vx, np.ones(len(vx))])
    gram = tx.T @ tx + alpha * np.diag(np.r_[np.ones(tx.shape[1] - 1), 0.0])
    coef = np.linalg.solve(gram, tx.T @ train_y)
    return tx @ coef, vx @ coef


def gaussian_samples(mean, residuals, seed, draws=20, phase_train=None, phase_test=None):
    rng = np.random.default_rng(seed)
    global_std = np.maximum(np.std(residuals, axis=0, ddof=1), 1e-7)
    std = np.repeat(global_std[None, :], len(mean), axis=0)
    if phase_train is not None:
        for phase in range(7):
            source = residuals[np.asarray(phase_train) == phase]
            if len(source) >= 30:
                std[np.asarray(phase_test) == phase] = np.maximum(np.std(source, axis=0, ddof=1), 1e-7)
    samples = mean[:, None, :] + std[:, None, :] * rng.standard_normal((len(mean), draws, mean.shape[1]))
    return samples, std


def residual_mixture_samples(mean, residuals, seed, draws=20):
    rng = np.random.default_rng(seed)
    _, _, vh = np.linalg.svd(residuals - residuals.mean(0), full_matrices=False)
    score = residuals @ vh[0]
    groups = score >= np.median(score)
    samples = np.empty((len(mean), draws, mean.shape[1]))
    std_acc = np.zeros_like(mean)
    for i in range(len(mean)):
        picked = rng.integers(0, 2, draws)
        for j, group in enumerate(picked):
            pool = residuals[groups == bool(group)]
            samples[i, j] = mean[i] + pool[rng.integers(0, len(pool))]
        std_acc[i] = samples[i].std(0)
    return samples, np.maximum(std_acc, 1e-7)


def lowrank_predict(train_x, train_y, test_x, latent_dim):
    center = train_y.mean(0)
    if latent_dim == 0:
        return np.repeat(center[None], len(train_y), 0), np.repeat(center[None], len(test_x), 0)
    _, _, vh = np.linalg.svd(train_y - center, full_matrices=False)
    basis = vh[: min(latent_dim, len(vh))]
    z = (train_y - center) @ basis.T
    ztr, zte = ridge(train_x, z, test_x)
    return center + ztr @ basis, center + zte @ basis


def phase_expert(train_x, train_y, test_x, train_phase, test_phase, soft=False):
    global_train, global_test = ridge(train_x, train_y, test_x)
    pred_train, pred_test = global_train.copy(), global_test.copy()
    for phase in range(7):
        tr = np.flatnonzero(train_phase == phase)
        te = np.flatnonzero(test_phase == phase)
        if len(tr) < max(40, train_x.shape[1] // 2) or not len(te):
            continue
        local_train, local_test = ridge(train_x[tr], train_y[tr], test_x[te], alpha=0.1)
        pred_train[tr] = .7 * local_train + .3 * global_train[tr] if soft else local_train
        pred_test[te] = .7 * local_test + .3 * global_test[te] if soft else local_test
    return pred_train, pred_test


class CVAE(nn.Module):
    def __init__(self, xdim, ydim, latent=16):
        super().__init__()
        self.latent = latent
        self.enc = nn.Sequential(nn.Linear(xdim + ydim, 96), nn.SiLU(), nn.Linear(96, 2 * latent))
        self.dec = nn.Sequential(nn.Linear(xdim + latent, 96), nn.SiLU(), nn.Linear(96, ydim))
        self.log_std = nn.Parameter(torch.full((ydim,), -2.0))

    def encode(self, x, y):
        q = self.enc(torch.cat([x, y], 1)); return q[:, : self.latent], torch.clamp(q[:, self.latent :], -6, 3)

    def decode(self, x, z):
        return self.dec(torch.cat([x, z], 1))


def cvae_fit_predict(train_x, train_y, test_x, seed, device, draws=20):
    tx, vx = standardize(train_x, test_x)
    ym, ys = train_y.mean(0), np.maximum(train_y.std(0), 1e-7)
    ty = (train_y - ym) / ys
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model = CVAE(tx.shape[1], ty.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    x = torch.as_tensor(tx, dtype=torch.float32, device=device)
    y = torch.as_tensor(ty, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(10):
        for start in range(0, len(x), 1024):
            idx = rng.permutation(len(x))[start : start + 1024]
            mu, lv = model.encode(x[idx], y[idx]); z = mu + torch.exp(.5 * lv) * torch.randn_like(mu)
            recon = model.decode(x[idx], z)
            loss = ((recon - y[idx]) ** 2).mean() + .01 * (-.5 * (1 + lv - mu * mu - torch.exp(lv))).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    def sample(arr):
        xt = torch.as_tensor(arr, dtype=torch.float32, device=device)
        values = []
        with torch.no_grad():
            for _ in range(draws):
                values.append(model.decode(xt, torch.randn((len(xt), 16), device=device)).cpu().numpy() * ys + ym)
        return np.stack(values, 1)
    tr_samples, te_samples = sample(tx), sample(vx)
    return tr_samples.mean(1), te_samples.mean(1), te_samples, int(sum(p.numel() for p in model.parameters()))


def ap_score(y, score):
    y = np.asarray(y, dtype=int); order = np.argsort(-np.asarray(score), kind="mergesort"); yy = y[order]
    return float(np.sum((np.cumsum(yy) / np.arange(1, len(yy) + 1)) * yy) / max(1, yy.sum()))


def demo_bootstrap(values, baseline, task, episode, seed=10, reps=2000):
    grouped = defaultdict(list)
    for v, b, t, e in zip(values, baseline, task, episode): grouped[(t, e)].append(float(b - v))
    delta = np.asarray([np.mean(v) for v in grouped.values()]); rng = np.random.default_rng(seed)
    boot = np.asarray([np.mean(delta[rng.integers(0, len(delta), len(delta))]) for _ in range(reps)])
    return float(delta.mean()), [float(x) for x in np.quantile(boot, [.025, .975])]


def auc_bootstrap(y, p, task, episode, seed=11, reps=1000):
    demos = sorted(set(zip(task, episode))); rng = np.random.default_rng(seed); vals = []
    for _ in range(reps):
        chosen = [demos[i] for i in rng.integers(0, len(demos), len(demos))]
        idx = np.concatenate([np.flatnonzero((task == t) & (episode == e)) for t, e in chosen])
        vals.append(rank_auc(y[idx], p[idx]))
    vals = np.asarray([x for x in vals if np.isfinite(x)])
    return [float(x) for x in np.quantile(vals, [.025, .975])]


def make_features(rows):
    arrays = {}
    raw = {
        "base": np.asarray([r["baseline_features"] for r in rows]),
        "phase": np.asarray([r["phase_history_features"] for r in rows]),
    }
    for width in (1, 5, 10): raw[f"h{width}"] = np.asarray([r[f"history{width}_summary"] for r in rows])
    for name in ("chunk5", "chunk10", "chunk20", "phase_chunk"):
        raw[name] = np.asarray([r[f"{name}_summary"] for r in rows])
    base = project(raw["base"], 48, 1010)
    hist = {k: project(v, 24, 2010 + int(k[1:])) for k, v in raw.items() if k.startswith("h")}
    actions = {k: project(v, 18, 3010 + len(k)) for k, v in raw.items() if k.startswith("chunk") or k == "phase_chunk"}
    phase = raw["phase"]
    arrays["baseline_b"] = base
    arrays["exp9_temporal"] = np.c_[base, hist["h5"], actions["chunk5"]]
    arrays["no_action"] = np.c_[base, hist["h10"], phase]
    arrays["no_history"] = np.c_[base, actions["chunk10"], phase]
    for c in (5, 10, 20): arrays[f"A_chunk{c}"] = np.c_[base, hist["h10"], actions[f"chunk{c}"], phase]
    arrays["A_phase_chunk"] = np.c_[base, hist["h10"], actions["phase_chunk"], phase]
    for h in (1, 5, 10): arrays[f"A_history{h}"] = np.c_[base, hist[f"h{h}"], actions["chunk10"], phase]
    arrays["A_phase_history"] = np.c_[base, hist["h10"], actions["chunk10"], phase]
    core = np.c_[base, hist["h10"], actions["chunk10"]]
    arrays["B_hard_phase_moe"] = core; arrays["B_soft_phase_moe"] = core
    arrays["B_hybrid_phase"] = np.c_[core, phase]
    rng = np.random.default_rng(101010); shuffled = phase.copy()
    for task in sorted({r["task"] for r in rows}):
        idx = np.flatnonzero(np.asarray([r["task"] == task for r in rows])); shuffled[idx] = shuffled[rng.permutation(idx)]
    arrays["B_shuffled_phase"] = np.c_[core, shuffled]
    for name in ("C_latent0", "C_latent16", "C_latent32", "F_deterministic", "F_heteroscedastic", "F_residual_mixture", "F_cvae16"):
        arrays[name] = np.c_[core, phase]
    return arrays


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True); parser.add_argument("--dataset-run", default="runs/exp10_a0_phase_macro_dataset_r1_20260814")
    args = parser.parse_args(); out = Path("runs") / args.run_id
    if out.exists(): raise FileExistsError(f"immutable run exists: {out}")
    artifacts, plots = out / "artifacts", out / "plots"; artifacts.mkdir(parents=True); plots.mkdir()
    start = time.time(); rows = pq.read_table(Path(args.dataset_run) / "artifacts" / "macro_trajectory_dataset.parquet").to_pylist()
    xmap = make_features(rows); n = len(rows)
    task = np.asarray([r["task"] for r in rows]); episode = np.asarray([r["episode"] for r in rows]); fold = np.asarray([r["fold"] for r in rows]); role = np.asarray([r["direction_role"] for r in rows]); phase = np.asarray([r["phase_index"] for r in rows])
    event = np.asarray([r["macro_adverse_event"] for r in rows], dtype=int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction_rows, fit_records = [], []
    score_store = {}
    cvae_params = []
    for horizon in HORIZONS:
        y_all = [np.asarray(r[f"response_{horizon}"], dtype=np.float64) for r in rows]
        for method in METHODS:
            scores = np.full(n, np.nan); cover = np.full(n, np.nan); errors = np.full(n, np.nan)
            for current_task in sorted(set(task)):
                ti = np.flatnonzero(task == current_task); y = np.vstack([y_all[i] for i in ti]); x = xmap[method][ti]; p = phase[ti]; f = fold[ti]; rr = role[ti]
                for current_fold in range(5):
                    train = np.flatnonzero((f != current_fold) & (rr == "basis")); test = np.flatnonzero(f == current_fold)
                    if method == "B_hard_phase_moe": tr_mean, te_mean = phase_expert(x[train], y[train], x[test], p[train], p[test], soft=False)
                    elif method == "B_soft_phase_moe": tr_mean, te_mean = phase_expert(x[train], y[train], x[test], p[train], p[test], soft=True)
                    elif method.startswith("C_latent"):
                        latent = int(method.replace("C_latent", "")); tr_mean, te_mean = lowrank_predict(x[train], y[train], x[test], latent)
                    elif method == "F_cvae16":
                        means, samples_list, pars = [], [], []
                        for seed in SEEDS:
                            tr0, te0, sam0, par = cvae_fit_predict(x[train], y[train], x[test], seed + current_fold, device)
                            means.append(te0); samples_list.append(sam0); pars.append(par)
                        tr_mean = ridge(x[train], y[train], x[test])[0]; te_mean = np.mean(means, 0); samples = np.concatenate(samples_list, 1); std = np.maximum(samples.std(1), 1e-7); cvae_params.extend(pars)
                    else: tr_mean, te_mean = ridge(x[train], y[train], x[test])
                    residual = y[train] - tr_mean
                    if method == "F_deterministic" or method == "C_latent0":
                        samples = np.repeat(te_mean[:, None, :], 20, axis=1); std = np.zeros_like(te_mean)
                    elif method == "F_residual_mixture": samples, std = residual_mixture_samples(te_mean, residual, 1000 + current_fold)
                    elif method == "F_heteroscedastic": samples, std = gaussian_samples(te_mean, residual, 1000 + current_fold, phase_train=p[train], phase_test=p[test])
                    elif method != "F_cvae16": samples, std = gaussian_samples(te_mean, residual, 1000 + current_fold)
                    es = energy_score(samples, y[test]); cov = np.mean(np.abs(y[test] - te_mean) <= 1.6448536269514722 * std, axis=1); err = np.linalg.norm(te_mean - y[test], axis=1)
                    global_idx = ti[test]; scores[global_idx], cover[global_idx], errors[global_idx] = es, cov, err
                    full_distribution = method in {"exp9_temporal", "A_chunk10", "B_hybrid_phase", "C_latent16", "F_heteroscedastic", "F_cvae16"}
                    for local, gi in enumerate(global_idx):
                        prediction_rows.append({"method": method, "track": TRACK[method], "horizon": horizon.upper(), "intervention_id": rows[gi]["intervention_id"], "task": task[gi], "episode": episode[gi], "fold": int(fold[gi]), "phase": f"P{phase[gi]}", "energy_score": float(es[local]), "coverage_90": float(cov[local]), "endpoint_l2": float(err[local]), "predictive_mean": te_mean[local].tolist() if full_distribution else [], "predictive_std": std[local].tolist() if full_distribution else []})
            score_store[(method, horizon)] = scores
            fit_records.append({"method": method, "track": TRACK[method], "horizon": horizon.upper(), "mean_energy": float(np.mean(scores)), "mean_coverage90": float(np.mean(cover)), "mean_endpoint_l2": float(np.mean(errors)), "p90_endpoint_l2": float(np.quantile(errors, .9)), "all_rows_scored": bool(np.isfinite(scores).all())})
            print(method, horizon, fit_records[-1]["mean_energy"], flush=True)

    # Phase/event classifiers for the strongest B model and direct terminal route.
    event_predictions = []
    endpoint_target = [np.asarray(r["endpoint_response"], dtype=np.float64) for r in rows]
    terminal_y = np.asarray([r["terminal_success_flip"] for r in rows], dtype=int)
    terminal_position = np.asarray([r["terminal_position_error"] for r in rows])
    for model, outcome in (("B_hybrid_phase", event), ("E_teacher_free_regime_ar", event), ("D_terminal_direct", terminal_y)):
        x = xmap["B_hybrid_phase"]
        probability = np.full(n, np.nan); threshold = np.full(n, np.nan); position_pred = np.full(n, np.nan); endpoint_pred = [None] * n
        for current_task in sorted(set(task)):
            ti = np.flatnonzero(task == current_task); xt = x[ti]; yy = outcome[ti]; ff = fold[ti]; rr = role[ti]
            for current_fold in range(5):
                tr = np.flatnonzero((ff != current_fold) & (rr == "basis")); te = np.flatnonzero(ff == current_fold)
                ptr, pte = ridge(xt[tr], yy[tr, None], xt[te], alpha=.2); ptr = 1 / (1 + np.exp(-np.clip(4 * (ptr[:, 0] - .5), -20, 20))); pte = 1 / (1 + np.exp(-np.clip(4 * (pte[:, 0] - .5), -20, 20)))
                if model == "E_teacher_free_regime_ar":
                    zero_train = np.asarray([rows[ti[j]]["zero_regime_sequence"] for j in tr], dtype=int)
                    pert_train = np.asarray([rows[ti[j]]["perturbed_regime_sequence"] for j in tr], dtype=int)
                    zero_test = np.asarray([rows[ti[j]]["zero_regime_sequence"] for j in te], dtype=int)
                    mask_train = np.asarray([rows[ti[j]]["regime_sequence_mask"] for j in tr], dtype=float)
                    mask_test = np.asarray([rows[ti[j]]["regime_sequence_mask"] for j in te], dtype=float)
                    prev_train = np.zeros((len(tr), 7)); prev_test = np.zeros((len(te), 7)); safe_train = np.ones(len(tr)); safe_test = np.ones(len(te))
                    for step in range(10):
                        target_onehot = one_hot(pert_train[:, step])
                        ar_train_x = np.c_[xt[tr], prev_train]; ar_test_x = np.c_[xt[te], prev_test]
                        logits_train, logits_test = ridge(ar_train_x, target_onehot, ar_test_x, alpha=.2)
                        logits_train -= logits_train.max(1, keepdims=True); logits_test -= logits_test.max(1, keepdims=True)
                        prob_train = np.exp(logits_train); prob_train /= prob_train.sum(1, keepdims=True)
                        prob_test = np.exp(logits_test); prob_test /= prob_test.sum(1, keepdims=True)
                        safe_train *= np.where(mask_train[:, step] > .5, prob_train[np.arange(len(tr)), zero_train[:, step]], 1.0)
                        safe_test *= np.where(mask_test[:, step] > .5, prob_test[np.arange(len(te)), zero_test[:, step]], 1.0)
                        prev_train = one_hot(np.argmax(prob_train, axis=1)); prev_test = one_hot(np.argmax(prob_test, axis=1))
                    ptr, pte = 1 - safe_train, 1 - safe_test
                th = select_specificity_threshold(yy[tr], ptr, .85)
                probability[ti[te]] = pte; threshold[ti[te]] = th
                _, pp = ridge(xt[tr], terminal_position[ti][tr, None], xt[te]); position_pred[ti[te]] = pp[:, 0]
                endpoint_task = np.vstack([endpoint_target[i] for i in ti])
                _, ep = ridge(xt[tr], endpoint_task[tr], xt[te])
                for local, value in zip(ti[te], ep): endpoint_pred[local] = value.tolist()
        for i in range(n): event_predictions.append({"model": model, "intervention_id": rows[i]["intervention_id"], "task": task[i], "episode": episode[i], "fold": int(fold[i]), "target": int(outcome[i]), "probability": float(probability[i]), "threshold": float(threshold[i]), "terminal_position_prediction": float(position_pred[i]), "endpoint_prediction": endpoint_pred[i]})

    pred_path = artifacts / "trajectory_predictions.parquet"; pq.write_table(pa.Table.from_pylist(prediction_rows), pred_path, compression="zstd")
    event_path = artifacts / "event_terminal_predictions.parquet"; pq.write_table(pa.Table.from_pylist(event_predictions), event_path, compression="zstd")
    lock = {"trajectory_predictions_sha256": sha256(pred_path), "event_terminal_predictions_sha256": sha256(event_path), "rows": len(prediction_rows), "locked_before_authorization": True}
    (artifacts / "prediction_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")

    # Aggregate only after the two prediction files are locked.
    base = {h: score_store[("exp9_temporal", h)] for h in HORIZONS}; aggregates = []
    for record in fit_records:
        method, horizon = record["method"], record["horizon"].lower()
        improvement = (np.mean(base[horizon]) - record["mean_energy"]) / np.mean(base[horizon])
        diff, ci = demo_bootstrap(score_store[(method, horizon)], base[horizon], task, episode, seed=1010 + len(aggregates))
        task_positive = {t: bool(np.mean(score_store[(method, horizon)][task == t]) < np.mean(base[horizon][task == t])) for t in sorted(set(task))}
        aggregates.append({**record, "relative_energy_improvement_vs_exp9_temporal": float(improvement), "absolute_demo_improvement": diff, "demo_bootstrap_ci": ci, "positive_tasks": int(sum(task_positive.values())), "task_positive": task_positive})
    pq.write_table(pa.Table.from_pylist(aggregates), artifacts / "trajectory_metrics.parquet", compression="zstd")

    event_metrics = []
    for model in ("B_hybrid_phase", "E_teacher_free_regime_ar", "D_terminal_direct"):
        rr = [r for r in event_predictions if r["model"] == model]; y = np.asarray([r["target"] for r in rr]); p = np.asarray([r["probability"] for r in rr]); th = np.asarray([r["threshold"] for r in rr]); pred = p >= th
        sensitivity = float(np.sum((y == 1) & pred) / max(1, np.sum(y == 1))); specificity = float(np.sum((y == 0) & ~pred) / max(1, np.sum(y == 0)))
        event_metrics.append({"model": model, "auroc": rank_auc(y, p), "auroc_demo_bootstrap_ci": auc_bootstrap(y, p, task, episode), "auprc": ap_score(y, p), "sensitivity": sensitivity, "specificity": specificity, "false_safe_rate": 1 - specificity})
    pq.write_table(pa.Table.from_pylist(event_metrics), artifacts / "event_terminal_metrics.parquet", compression="zstd")

    lookup = {(r["method"], r["horizon"]): r for r in aggregates}
    em = {r["model"]: r for r in event_metrics}
    route_gates = {
        "A": any(lookup[(m, "H5")]["relative_energy_improvement_vs_exp9_temporal"] >= .15 and lookup[(m, "H10")]["relative_energy_improvement_vs_exp9_temporal"] >= .15 and lookup[(m, "H5")]["demo_bootstrap_ci"][0] > 0 and lookup[(m, "H10")]["demo_bootstrap_ci"][0] > 0 and min(lookup[(m, "H5")]["positive_tasks"], lookup[(m, "H10")]["positive_tasks"]) >= 2 for m in METHODS if m.startswith("A_")),
        "B": em["B_hybrid_phase"]["auroc_demo_bootstrap_ci"][0] >= .75 and em["B_hybrid_phase"]["specificity"] >= .55 and em["B_hybrid_phase"]["sensitivity"] >= .85 and lookup[("B_hybrid_phase", "H5")]["relative_energy_improvement_vs_exp9_temporal"] >= .10,
        "C": any(lookup[(m, "H5")]["relative_energy_improvement_vs_exp9_temporal"] >= .15 and lookup[(m, "H10")]["relative_energy_improvement_vs_exp9_temporal"] >= .15 for m in ("C_latent16", "C_latent32")),
        "D": em["D_terminal_direct"]["auroc"] >= .80 and em["D_terminal_direct"]["auroc_demo_bootstrap_ci"][0] >= .70,
        "E": em["E_teacher_free_regime_ar"]["auroc_demo_bootstrap_ci"][0] >= .75 and em["E_teacher_free_regime_ar"]["auprc"] >= .70 and em["E_teacher_free_regime_ar"]["specificity"] >= .55 and em["E_teacher_free_regime_ar"]["sensitivity"] >= .85 and em["E_teacher_free_regime_ar"]["false_safe_rate"] <= .45,
        "F": any(lookup[(m, "H5")]["relative_energy_improvement_vs_exp9_temporal"] >= .20 and lookup[(m, "H10")]["relative_energy_improvement_vs_exp9_temporal"] >= .20 and .85 <= lookup[(m, "H5")]["mean_coverage90"] <= .95 and .85 <= lookup[(m, "H10")]["mean_coverage90"] <= .95 for m in ("F_heteroscedastic", "F_residual_mixture", "F_cvae16")),
    }
    authorized = [k for k, v in route_gates.items() if v][:3]
    authorization = {"route_gates": route_gates, "qualified_routes": authorized, "new_cohort_authorized": bool(authorized), "maximum_stage_b_routes": 3, "decision_after_prediction_lock": True}
    (artifacts / "route_authorization.json").write_text(json.dumps(authorization, indent=2), encoding="utf-8")

    # Sixteen diagnostic plots, generated from the locked predictions/metrics.
    plot_specs = [
        ("01_energy_h5", "mean_energy", "H5"), ("02_energy_h10", "mean_energy", "H10"),
        ("03_improvement_h5", "relative_energy_improvement_vs_exp9_temporal", "H5"), ("04_improvement_h10", "relative_energy_improvement_vs_exp9_temporal", "H10"),
        ("05_coverage_h5", "mean_coverage90", "H5"), ("06_coverage_h10", "mean_coverage90", "H10"),
        ("07_endpoint_h5", "mean_endpoint_l2", "H5"), ("08_endpoint_h10", "mean_endpoint_l2", "H10"),
        ("09_p90_endpoint_h5", "p90_endpoint_l2", "H5"), ("10_p90_endpoint_h10", "p90_endpoint_l2", "H10"),
    ]
    for name, metric, horizon in plot_specs:
        vals = [r for r in aggregates if r["horizon"] == horizon]; vals = sorted(vals, key=lambda z: z[metric])
        fig, ax = plt.subplots(figsize=(10, 6)); ax.barh([v["method"] for v in vals], [v[metric] for v in vals]); ax.set(title=f"{metric} {horizon}", xlabel=metric); fig.tight_layout(); fig.savefig(plots / f"{name}.png", dpi=140); plt.close(fig)
    for idx, metric in enumerate(("auroc", "auprc", "sensitivity", "specificity"), start=11):
        fig, ax = plt.subplots(figsize=(7, 4)); ax.bar([r["model"] for r in event_metrics], [r[metric] for r in event_metrics]); ax.set_ylim(0, 1); ax.set(title=metric, ylabel=metric); ax.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(plots / f"{idx:02d}_event_{metric}.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); phases, counts = np.unique(phase, return_counts=True); ax.bar([f"P{x}" for x in phases], counts); ax.set(title="Phase support", ylabel="interventions"); fig.tight_layout(); fig.savefig(plots / "15_phase_support.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(route_gates.keys(), [int(v) for v in route_gates.values()]); ax.set_ylim(0, 1.2); ax.set(title="Independent route authorization", ylabel="pass"); fig.tight_layout(); fig.savefig(plots / "16_route_authorization.png", dpi=140); plt.close(fig)

    metrics = {"status": "completed", "run_id": args.run_id, "device": str(device), "methods": len(METHODS), "prediction_rows": len(prediction_rows), "event_prediction_rows": len(event_predictions), "prediction_lock": lock, "authorization": authorization, "cvae_parameter_count_max": max(cvae_params) if cvae_params else 0, "wall_seconds": time.time() - start}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8"); (out / "config.json").write_text(json.dumps({"command": " ".join(sys.argv), "seeds": SEEDS, "stage": "EXP10 Stage A routes"}, indent=2), encoding="utf-8"); (out / "stdout.log").write_text(json.dumps(metrics, indent=2), encoding="utf-8"); (out / "stderr.log").write_text("", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
