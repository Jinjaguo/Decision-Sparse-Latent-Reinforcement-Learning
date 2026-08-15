"""EXP9 Stage-A model comparison on locked EXP8 data.

This program is development-only.  It cross-fits whole demonstrations and never
uses held-out-direction rows for fitting.  Test predictions are written once before
the feasibility gate is evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from decision_sparse_rl.metrics.exp9 import (
    binary_metrics,
    ece,
    energy_score,
    lifted_features,
    rank_auc,
    select_specificity_threshold,
)


HORIZONS = ["1", "3", "5"]
SEEDS = [190901, 190902, 190903]
MODELS = ["baseline_b", "factorized", "lifted_hybrid", "graph_mixture", "temporal_sequence_mixture"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def standardize(train, test):
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (train - mean) / scale, (test - mean) / scale, mean, scale


def ridge_fit_predict(train_x, train_y, test_x, alpha=1e-2):
    x = np.concatenate([train_x, np.ones((len(train_x), 1))], axis=1)
    xt = np.concatenate([test_x, np.ones((len(test_x), 1))], axis=1)
    reg = np.sqrt(alpha) * np.eye(x.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.lstsq(np.vstack([x, reg]), np.vstack([train_y, np.zeros((x.shape[1], train_y.shape[1]))]), rcond=None)[0]
    return x @ coef, xt @ coef


def logistic_fit_predict(train_x, train_y, test_x, device):
    torch.manual_seed(9019)
    x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    xt = torch.as_tensor(test_x, dtype=torch.float32, device=device)
    layer = nn.Linear(train_x.shape[1], train_y.shape[1]).to(device)
    positives = np.maximum(train_y.sum(axis=0), 1.0)
    negatives = np.maximum(len(train_y) - positives, 1.0)
    pos_weight = torch.as_tensor(negatives / positives, dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.025, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(layer(x), y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return torch.sigmoid(layer(x)).cpu().numpy(), torch.sigmoid(layer(xt)).cpu().numpy()


class GraphMDN(nn.Module):
    def __init__(self, input_dim, output_dim=26, horizons=3, components=3):
        super().__init__()
        self.h, self.k, self.d = horizons, components, output_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.SiLU(), nn.Linear(128, 96), nn.SiLU())
        self.event = nn.Linear(96, horizons)
        self.mixture = nn.Linear(96, horizons * components * (2 * output_dim + 1))

    def forward(self, x):
        z = self.encoder(x)
        raw = self.mixture(z).reshape(-1, self.h, self.k, 2 * self.d + 1)
        logits = raw[..., 0]
        mean = raw[..., 1 : 1 + self.d]
        log_std = torch.clamp(raw[..., 1 + self.d :], -5.0, 2.5)
        return self.event(z), logits, mean, log_std


class SequenceMDN(nn.Module):
    def __init__(self, history_dim=61, action_dim=7, intervention_dim=25, output_dim=26, horizons=3, components=3):
        super().__init__()
        self.h, self.k, self.d = horizons, components, output_dim
        self.history = nn.GRU(history_dim + 1, 48, batch_first=True)
        self.action = nn.GRU(action_dim + 1, 24, batch_first=True)
        self.encoder = nn.Sequential(nn.Linear(48 + 24 + intervention_dim, 96), nn.SiLU(), nn.Linear(96, 80), nn.SiLU())
        self.event = nn.Linear(80, horizons)
        self.mixture = nn.Linear(80, horizons * components * (2 * output_dim + 1))

    def forward(self, inputs):
        history, history_mask, action, action_mask, intervention = inputs
        h, _ = self.history(torch.cat([history, history_mask[..., None]], dim=2))
        a, _ = self.action(torch.cat([action, action_mask[..., None]], dim=2))
        hden = history_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        aden = action_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        hp = (h * history_mask[..., None]).sum(dim=1) / hden
        ap = (a * action_mask[..., None]).sum(dim=1) / aden
        z = self.encoder(torch.cat([hp, ap, intervention], dim=1))
        raw = self.mixture(z).reshape(-1, self.h, self.k, 2 * self.d + 1)
        return self.event(z), raw[..., 0], raw[..., 1 : 1 + self.d], torch.clamp(raw[..., 1 + self.d :], -5.0, 2.5)


def model_parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def mdn_loss(outputs, event_y, response_y):
    event_logits, mix_logits, mean, log_std = outputs
    target = response_y[:, :, None, :]
    component_logprob = -0.5 * (((target - mean) / torch.exp(log_std)) ** 2 + 2 * log_std + math.log(2 * math.pi)).sum(dim=-1)
    response_nll = -torch.logsumexp(torch.log_softmax(mix_logits, dim=-1) + component_logprob, dim=-1).mean()
    positives = event_y.sum(dim=0).clamp_min(1.0)
    pos_weight = (len(event_y) - positives).clamp_min(1.0) / positives
    event_loss = nn.functional.binary_cross_entropy_with_logits(event_logits, event_y, pos_weight=pos_weight)
    return response_nll + 0.35 * event_loss


def batch_inputs(kind, arrays, indices, device):
    if kind == "graph":
        return torch.as_tensor(arrays["graph"][indices], dtype=torch.float32, device=device)
    return tuple(torch.as_tensor(arrays[name][indices], dtype=torch.float32, device=device) for name in ("history", "history_mask", "action", "action_mask", "intervention"))


def train_neural(kind, arrays, train_idx, test_idx, y_event, y_response, output_dim, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if kind == "graph":
        model = GraphMDN(arrays["graph"].shape[1], output_dim=output_dim).to(device)
    else:
        model = SequenceMDN(intervention_dim=arrays["intervention"].shape[1], output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(32):
        order = rng.permutation(train_idx)
        for start in range(0, len(order), 512):
            idx = order[start : start + 512]
            inp = batch_inputs(kind, arrays, idx, device)
            ey = torch.as_tensor(y_event[idx], dtype=torch.float32, device=device)
            ry = torch.as_tensor(y_response[idx], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = mdn_loss(model(inp), ey, ry)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()

    def predict(indices):
        result = []
        with torch.no_grad():
            for start in range(0, len(indices), 1024):
                idx = indices[start : start + 1024]
                result.append(tuple(x.cpu().numpy() for x in model(batch_inputs(kind, arrays, idx, device))))
        return tuple(np.concatenate([x[j] for x in result], axis=0) for j in range(4))

    return predict(train_idx), predict(test_idx), model_parameter_count(model)


def mdn_samples(outputs, seed, draws=24):
    event_logits, mix_logits, mean, log_std = outputs
    rng = np.random.default_rng(seed)
    weights = np.exp(mix_logits - np.max(mix_logits, axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    n, h, k, d = mean.shape
    samples = np.empty((n, h, draws, d), dtype=np.float64)
    for i in range(n):
        for j in range(h):
            components = rng.choice(k, size=draws, p=weights[i, j])
            samples[i, j] = mean[i, j, components] + np.exp(log_std[i, j, components]) * rng.standard_normal((draws, d))
    event_prob = 1.0 / (1.0 + np.exp(-event_logits))
    expected = np.sum(weights[..., None] * mean, axis=2)
    variance = np.sum(weights[..., None] * (np.exp(2 * log_std) + mean**2), axis=2) - expected**2
    return event_prob, expected, np.sqrt(np.maximum(variance, 1e-12)), samples, -np.sum(weights * np.log(np.maximum(weights, 1e-12)), axis=-1)


def average_precision(y, score):
    y = np.asarray(y, dtype=np.int64)
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="mergesort")
    ys = y[order]
    if ys.sum() == 0:
        return float("nan")
    precision = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float(np.sum(precision * ys) / ys.sum())


def bootstrap_demo_difference(records, model, horizon, baseline="baseline_b", seed=1909, reps=2000):
    by_model = defaultdict(dict)
    for r in records:
        if r["horizon"] == horizon and r["model"] in (model, baseline):
            by_model[r["model"]][(r["task"], r["episode"])] = r["energy_score"]
    keys = sorted(set(by_model[model]) & set(by_model[baseline]))
    diff = np.asarray([by_model[baseline][k] - by_model[model][k] for k in keys])
    rng = np.random.default_rng(seed)
    boot = np.asarray([np.mean(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(reps)])
    return float(np.mean(diff)), [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def make_plots(plot_dir, prevalence, risk_rows, dist_rows, comparison, seed_rows):
    plot_dir.mkdir(parents=True, exist_ok=True)
    tasks = sorted(set(r["task"] for r in prevalence))
    x = np.arange(len(HORIZONS))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for task in tasks:
        vals = [next(r["preserved_rate"] for r in prevalence if r["task"] == task and r["horizon"] == h) for h in HORIZONS]
        ax.plot(x, vals, marker="o", label=task.replace("_", " "))
    ax.set_xticks(x, ["H1", "H3", "H5"]); ax.set_ylabel("Exact-mode preservation rate"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(plot_dir / "event_prevalence_by_task_horizon.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for model in MODELS:
        rows = [r for r in risk_rows if r["model"] == model and r["horizon"] in ("1", "3")]
        ax.scatter(np.mean([r["specificity"] for r in rows]), np.mean([r["sensitivity"] for r in rows]), label=model)
    ax.axhline(0.85, color="k", ls="--"); ax.axvline(0.50, color="k", ls=":"); ax.set_xlabel("Specificity"); ax.set_ylabel("Sensitivity"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(plot_dir / "specificity_sensitivity_tradeoff.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5)); width = 0.16
    for i, model in enumerate(MODELS):
        vals = [next(r["mean_energy_score"] for r in dist_rows if r["model"] == model and r["horizon"] == h) for h in HORIZONS]
        ax.bar(x + (i - 2) * width, vals, width, label=model)
    ax.set_xticks(x, ["H1", "H3", "H5"]); ax.set_ylabel("Demo-averaged energy score (lower better)"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(plot_dir / "energy_score_by_model_horizon.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in MODELS:
        vals = [next(r["coverage_90"] for r in dist_rows if r["model"] == model and r["horizon"] == h) for h in HORIZONS]
        ax.plot(x, vals, marker="o", label=model)
    ax.axhspan(0.85, 0.95, color="green", alpha=0.12); ax.set_xticks(x, ["H1", "H3", "H5"]); ax.set_ylabel("Marginal 90% interval coverage"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(plot_dir / "predictive_interval_coverage.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [r["model"] for r in comparison if r["model"] != "baseline_b"]
    vals = [min(r["relative_improvement_h1"], r["relative_improvement_h3"]) for r in comparison if r["model"] != "baseline_b"]
    colors = ["tab:green" if r["stage_a_pass"] else "tab:red" for r in comparison if r["model"] != "baseline_b"]
    ax.bar(names, vals, color=colors); ax.axhline(0, color="k"); ax.set_ylabel("Worst H1/H3 relative energy improvement"); ax.tick_params(axis="x", rotation=20); fig.tight_layout()
    fig.savefig(plot_dir / "stageA_model_selection.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    grouped = defaultdict(list)
    for r in seed_rows: grouped[r["model"]].append(r["validation_nll"])
    ax.boxplot([grouped[k] for k in sorted(grouped)], labels=sorted(grouped)); ax.set_ylabel("Final training NLL proxy"); ax.tick_params(axis="x", rotation=20); fig.tight_layout()
    fig.savefig(plot_dir / "seed_variance.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-run", required=True)
    parser.add_argument("--output-run", required=True)
    args = parser.parse_args()
    target_run = Path(args.target_run)
    out = Path(args.output_run)
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    artifacts, plots = out / "artifacts", out / "plots"
    artifacts.mkdir(parents=True); plots.mkdir()
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_records = pq.read_table(target_run / "artifacts" / "hybrid_inputs.parquet").to_pylist()
    target_records = pq.read_table(target_run / "artifacts" / "hybrid_targets.parquet").to_pylist()
    prevalence = pq.read_table(target_run / "artifacts" / "retrospective_event_prevalence.parquet").to_pylist()
    ids = [r["intervention_id"] for r in input_records]
    id_index = {value: i for i, value in enumerate(ids)}
    n = len(ids)
    arrays = {
        "baseline": np.vstack([r["baseline_input"] for r in input_records]),
        "graph": np.vstack([r["graph_input"] for r in input_records]),
        "history": np.asarray([r["history_sequence"] for r in input_records], dtype=np.float64),
        "history_mask": np.asarray([r["history_mask"] for r in input_records], dtype=np.float64),
        "action": np.asarray([r["action_chunk"] for r in input_records], dtype=np.float64),
        "action_mask": np.asarray([r["action_chunk_mask"] for r in input_records], dtype=np.float64),
    }
    arrays["intervention"] = arrays["baseline"][:, 57:]
    tasks = np.asarray([r["task"] for r in input_records])
    episodes = np.asarray([r["episode"] for r in input_records])
    folds = np.asarray([r["fold"] for r in input_records], dtype=int)
    basis = np.asarray([r["direction_role"] == "basis" for r in input_records])
    response_dims = {}
    for r in target_records:
        response_dims.setdefault(r["task"], len(r["signed_response_vector"]))
        if response_dims[r["task"]] != len(r["signed_response_vector"]):
            raise RuntimeError("response dimension changed within task")
    max_response_dim = max(response_dims.values())
    y_event = np.zeros((n, 3), dtype=np.float64)
    y_response = np.zeros((n, 3, max_response_dim), dtype=np.float64)
    for r in target_records:
        if r["horizon"] in HORIZONS:
            i, h = id_index[r["intervention_id"]], HORIZONS.index(r["horizon"])
            y_event[i, h] = float(r["mode_preserved"])
            value = np.asarray(r["signed_response_vector"], dtype=np.float64)
            y_response[i, h, : len(value)] = value

    prediction_rows, seed_rows, parameter_counts = [], [], {}
    training_seconds = defaultdict(float)
    for task in sorted(set(tasks)):
        response_dim = response_dims[task]
        for fold in range(5):
            test_idx = np.flatnonzero((tasks == task) & (folds == fold))
            train_idx = np.flatnonzero((tasks == task) & (folds != fold) & basis)
            y_train_flat = y_response[train_idx, :, :response_dim].reshape(len(train_idx), -1)
            y_mean = y_train_flat.mean(axis=0)
            y_scale = y_train_flat.std(axis=0); y_scale[y_scale < 1e-5] = 1e-5
            y_train_z = ((y_train_flat - y_mean) / y_scale).reshape(len(train_idx), 3, response_dim)
            y_response_z = np.zeros((n, 3, response_dim), dtype=np.float64)
            y_response_z[test_idx] = ((y_response[test_idx, :, :response_dim].reshape(len(test_idx), -1) - y_mean) / y_scale).reshape(len(test_idx), 3, response_dim)
            y_response_z[train_idx] = y_train_z

            bx_train, bx_test, _, _ = standardize(arrays["baseline"][train_idx], arrays["baseline"][test_idx])
            t0 = time.time()
            train_prob, test_prob = logistic_fit_predict(bx_train, y_event[train_idx], bx_test, device)
            train_mu, test_mu = ridge_fit_predict(bx_train, y_train_z.reshape(len(train_idx), -1), bx_test)
            train_mu = train_mu.reshape(-1, 3, response_dim); test_mu = test_mu.reshape(-1, 3, response_dim)
            residual_std = np.std(y_train_z - train_mu, axis=0); residual_std[residual_std < 0.03] = 0.03
            training_seconds["baseline_b"] += time.time() - t0

            linear_predictions = {"baseline_b": (train_prob, test_prob, train_mu, test_mu, residual_std)}
            fx_train = np.concatenate([bx_train, train_prob], axis=1); fx_test = np.concatenate([bx_test, test_prob], axis=1)
            t0 = time.time(); ftrain_mu, ftest_mu = ridge_fit_predict(fx_train, y_train_z.reshape(len(train_idx), -1), fx_test)
            ftrain_mu, ftest_mu = ftrain_mu.reshape(-1, 3, response_dim), ftest_mu.reshape(-1, 3, response_dim)
            fstd = np.std(y_train_z - ftrain_mu, axis=0); fstd[fstd < 0.03] = 0.03
            training_seconds["factorized"] += time.time() - t0
            linear_predictions["factorized"] = (train_prob, test_prob, ftrain_mu, ftest_mu, fstd)

            lifted_train = lifted_features(bx_train, 57); lifted_test = lifted_features(bx_test, 57)
            lifted_train, lifted_test, _, _ = standardize(lifted_train, lifted_test)
            t0 = time.time(); ltrain_prob, ltest_prob = logistic_fit_predict(lifted_train, y_event[train_idx], lifted_test, device)
            ltrain_mu, ltest_mu = ridge_fit_predict(np.concatenate([lifted_train, ltrain_prob], axis=1), y_train_z.reshape(len(train_idx), -1), np.concatenate([lifted_test, ltest_prob], axis=1), alpha=0.05)
            ltrain_mu, ltest_mu = ltrain_mu.reshape(-1, 3, response_dim), ltest_mu.reshape(-1, 3, response_dim)
            lstd = np.std(y_train_z - ltrain_mu, axis=0); lstd[lstd < 0.03] = 0.03
            training_seconds["lifted_hybrid"] += time.time() - t0
            linear_predictions["lifted_hybrid"] = (ltrain_prob, ltest_prob, ltrain_mu, ltest_mu, lstd)

            for model_name, (ptr, pte, mutr, mute, std) in linear_predictions.items():
                thresholds = [select_specificity_threshold(y_event[train_idx, h], ptr[:, h], 0.85) for h in range(3)]
                rng = np.random.default_rng(9000 + fold)
                samples = mute[:, :, None, :] + std[None, :, None, :] * rng.standard_normal((len(test_idx), 3, 32, response_dim))
                for local, global_idx in enumerate(test_idx):
                    for h, horizon in enumerate(HORIZONS):
                        truth_z = (y_response[global_idx, h, :response_dim] - y_mean.reshape(3, response_dim)[h]) / y_scale.reshape(3, response_dim)[h]
                        score = energy_score(samples[local : local + 1, h], truth_z[None])[0]
                        prediction_rows.append({"model": model_name, "task": task, "episode": episodes[global_idx], "fold": fold, "intervention_id": ids[global_idx], "horizon": horizon, "response_dimension": response_dim, "event_probability": float(pte[local, h]), "event_label": int(y_event[global_idx, h]), "threshold": float(thresholds[h]), "predicted_mean": mute[local, h].tolist(), "predicted_std": std[h].tolist(), "energy_score": float(score), "coverage_90": float(np.mean(np.abs(truth_z - mute[local, h]) <= 1.644854 * std[h])), "mixture_entropy": 0.0})

            # Fold-local normalization for neural inputs, fit on training demos only.
            neural_arrays = {}
            for name in ("graph", "history", "action", "intervention"):
                shape = arrays[name].shape
                flat = arrays[name].reshape(n, -1)
                tr, te, mean, scale = standardize(flat[train_idx], flat[test_idx])
                normalized = np.empty_like(flat); normalized[train_idx] = tr; normalized[test_idx] = te
                neural_arrays[name] = normalized.reshape(shape)
            neural_arrays["history_mask"] = arrays["history_mask"]
            neural_arrays["action_mask"] = arrays["action_mask"]

            for model_name, kind in (("graph_mixture", "graph"), ("temporal_sequence_mixture", "sequence")):
                ensemble_train, ensemble_test, counts = [], [], []
                t0 = time.time()
                for seed in SEEDS:
                    tr_out, te_out, count = train_neural(kind, neural_arrays, train_idx, test_idx, y_event, y_response_z, response_dim, seed, device)
                    ensemble_train.append(tr_out); ensemble_test.append(te_out); counts.append(count)
                training_seconds[model_name] += time.time() - t0
                parameter_counts[model_name] = counts[0]
                if len(set(counts)) != 1 or counts[0] >= 500000:
                    raise RuntimeError("neural parameter budget violated")
                train_event = np.mean([1.0 / (1.0 + np.exp(-o[0])) for o in ensemble_train], axis=0)
                thresholds = [select_specificity_threshold(y_event[train_idx, h], train_event[:, h], 0.85) for h in range(3)]
                seed_sample_sets, seed_event, seed_mean, seed_std, seed_entropy = [], [], [], [], []
                for seed, output in zip(SEEDS, ensemble_test):
                    ep, mu, std, samples, entropy = mdn_samples(output, seed + fold, draws=16)
                    seed_event.append(ep); seed_mean.append(mu); seed_std.append(std); seed_sample_sets.append(samples); seed_entropy.append(entropy)
                    seed_rows.append({"model": model_name, "task": task, "fold": fold, "seed": seed, "validation_nll": float(np.mean(-np.log(np.maximum(ep * y_event[test_idx] + (1 - ep) * (1 - y_event[test_idx]), 1e-8))))})
                ep = np.mean(seed_event, axis=0); mu = np.mean(seed_mean, axis=0); std = np.sqrt(np.mean([s**2 + m**2 for s, m in zip(seed_std, seed_mean)], axis=0) - mu**2); samples = np.concatenate(seed_sample_sets, axis=2); entropy = np.mean(seed_entropy, axis=0)
                for local, global_idx in enumerate(test_idx):
                    for h, horizon in enumerate(HORIZONS):
                        truth_z = (y_response[global_idx, h, :response_dim] - y_mean.reshape(3, response_dim)[h]) / y_scale.reshape(3, response_dim)[h]
                        score = energy_score(samples[local : local + 1, h], truth_z[None])[0]
                        prediction_rows.append({"model": model_name, "task": task, "episode": episodes[global_idx], "fold": fold, "intervention_id": ids[global_idx], "horizon": horizon, "response_dimension": response_dim, "event_probability": float(ep[local, h]), "event_label": int(y_event[global_idx, h]), "threshold": float(thresholds[h]), "predicted_mean": mu[local, h].tolist(), "predicted_std": std[local, h].tolist(), "energy_score": float(score), "coverage_90": float(np.mean(np.abs(truth_z - mu[local, h]) <= 1.644854 * std[local, h])), "mixture_entropy": float(entropy[local, h])})

    # Write and lock cross-fitted predictions before aggregate gate evaluation.
    prediction_path = artifacts / "retrospective_predictions.parquet"
    pq.write_table(pa.Table.from_pylist(prediction_rows), prediction_path, compression="zstd")
    prediction_hash = sha256(prediction_path)
    (artifacts / "prediction_lock.json").write_text(json.dumps({"sha256": prediction_hash, "rows": len(prediction_rows), "written_before_gate": True}, indent=2), encoding="utf-8")

    demo_rows = []
    for key in sorted(set((r["model"], r["task"], r["episode"], r["horizon"]) for r in prediction_rows)):
        subset = [r for r in prediction_rows if (r["model"], r["task"], r["episode"], r["horizon"]) == key]
        demo_rows.append({"model": key[0], "task": key[1], "episode": key[2], "horizon": key[3], "energy_score": float(np.mean([r["energy_score"] for r in subset])), "coverage_90": float(np.mean([r["coverage_90"] for r in subset]))})

    risk_rows, dist_rows = [], []
    for model in MODELS:
        for horizon in HORIZONS:
            subset = [r for r in prediction_rows if r["model"] == model and r["horizon"] == horizon]
            y = np.asarray([r["event_label"] for r in subset]); p = np.asarray([r["event_probability"] for r in subset]); pred = np.asarray([p[i] >= subset[i]["threshold"] for i in range(len(subset))])
            tp, tn = np.sum((y == 1) & pred), np.sum((y == 0) & ~pred); fp, fn = np.sum((y == 0) & pred), np.sum((y == 1) & ~pred)
            risk_rows.append({"model": model, "horizon": horizon, "n": len(y), "auroc": rank_auc(y, p), "auprc": average_precision(y, p), "ece": ece(y, p), "sensitivity": float(tp / max(1, tp + fn)), "specificity": float(tn / max(1, tn + fp)), "false_safe_rate": float(fp / max(1, tn + fp))})
            ds = [r for r in demo_rows if r["model"] == model and r["horizon"] == horizon]
            ps = [r for r in subset]
            dist_rows.append({"model": model, "horizon": horizon, "mean_energy_score": float(np.mean([r["energy_score"] for r in ds])), "coverage_90": float(np.mean([r["coverage_90"] for r in ps])), "median_predicted_std": float(np.median(np.concatenate([r["predicted_std"] for r in ps]))), "p95_predicted_std": float(np.quantile(np.concatenate([r["predicted_std"] for r in ps]), 0.95)), "mean_mixture_entropy": float(np.mean([r["mixture_entropy"] for r in ps]))})

    baseline_scores = {r["horizon"]: r["mean_energy_score"] for r in dist_rows if r["model"] == "baseline_b"}
    comparison = []
    for model in MODELS:
        scores = {r["horizon"]: r["mean_energy_score"] for r in dist_rows if r["model"] == model}
        risks = {r["horizon"]: r for r in risk_rows if r["model"] == model}
        dists = {r["horizon"]: r for r in dist_rows if r["model"] == model}
        improvements = {h: (baseline_scores[h] - scores[h]) / baseline_scores[h] for h in HORIZONS}
        risk_pass = min(risks[h]["sensitivity"] for h in ("1", "3")) >= 0.85 and min(risks[h]["specificity"] for h in ("1", "3")) > 0.50
        response_pass = improvements["1"] > 0 and improvements["3"] > 0
        uncertainty_pass = all(dists[h]["median_predicted_std"] > 1e-4 and dists[h]["p95_predicted_std"] > dists[h]["median_predicted_std"] * 1.01 and 0.05 < dists[h]["coverage_90"] < 0.99 for h in ("1", "3"))
        stage_pass = model != "baseline_b" and risk_pass and response_pass and uncertainty_pass
        delta1, ci1 = bootstrap_demo_difference(demo_rows, model, "1") if model != "baseline_b" else (0.0, [0.0, 0.0])
        delta3, ci3 = bootstrap_demo_difference(demo_rows, model, "3") if model != "baseline_b" else (0.0, [0.0, 0.0])
        comparison.append({"model": model, "energy_h1": scores["1"], "energy_h3": scores["3"], "energy_h5": scores["5"], "relative_improvement_h1": improvements["1"], "relative_improvement_h3": improvements["3"], "relative_improvement_h5": improvements["5"], "risk_sensitivity_min_h1_h3": min(risks[h]["sensitivity"] for h in ("1", "3")), "risk_specificity_min_h1_h3": min(risks[h]["specificity"] for h in ("1", "3")), "uncertainty_nondegenerate": uncertainty_pass, "risk_pass": risk_pass, "response_pass": response_pass, "stage_a_pass": stage_pass, "demo_energy_delta_h1": delta1, "demo_energy_delta_h1_ci": ci1, "demo_energy_delta_h3": delta3, "demo_energy_delta_h3_ci": ci3, "parameter_count": parameter_counts.get(model, 0)})

    go = any(r["stage_a_pass"] for r in comparison)
    selected = sorted([r for r in comparison if r["stage_a_pass"]], key=lambda r: -(min(r["relative_improvement_h1"], r["relative_improvement_h3"])))[:3]
    architecture = {"stage": "A_retrospective_development_only", "new_cohort_authorized": go, "selection": [r["model"] for r in selected], "gate_definition": {"risk": "specificity > 0.50 and sensitivity >= 0.85 at both H1 and H3", "response": "energy score strictly better than Baseline B at H1 and H3", "uncertainty": "positive variable scale and non-extreme coverage at H1 and H3"}, "classification_if_stop": None if go else "action_conditioning_insufficient", "prediction_sha256": prediction_hash}
    pq.write_table(pa.Table.from_pylist(comparison), artifacts / "retrospective_model_comparison.parquet")
    pq.write_table(pa.Table.from_pylist(risk_rows), artifacts / "retrospective_risk_metrics.parquet")
    pq.write_table(pa.Table.from_pylist(dist_rows), artifacts / "retrospective_distribution_metrics.parquet")
    pq.write_table(pa.Table.from_pylist(seed_rows), artifacts / "retrospective_seed_metrics.parquet")
    (artifacts / "architecture_selection.json").write_text(json.dumps(architecture, indent=2), encoding="utf-8")
    cost = {"device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "dtype": "float32 neural / float64 scores", "training_seeds": SEEDS, "seconds_by_model": dict(training_seconds), "wall_seconds": time.time() - start_time, "parameter_counts": parameter_counts}
    (artifacts / "retrospective_training_cost.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")
    effects = [r for r in comparison if r["model"] != "baseline_b"]
    power = {"development_only": True, "demo_count": 30, "best_h1_relative_improvement": max(r["relative_improvement_h1"] for r in effects), "best_h3_relative_improvement": max(r["relative_improvement_h3"] for r in effects), "confirmatory_collection_recommended": go, "note": "No confirmatory power projection is actionable when the minimum feasibility gate fails." if not go else "Use locked demo-cluster effect distributions for prospective simulation."}
    (artifacts / "retrospective_power_estimate.json").write_text(json.dumps(power, indent=2), encoding="utf-8")
    make_plots(plots, prevalence, risk_rows, dist_rows, comparison, seed_rows)
    metrics = {"status": "completed", "stage": "A", "development_only": True, "gate": {"passed": go}, "classification": None if go else "action_conditioning_insufficient", "prediction_sha256": prediction_hash, "models": len(MODELS), "predictions": len(prediction_rows), "device": str(device)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "command.txt").write_text(" ".join(__import__("sys").argv), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
