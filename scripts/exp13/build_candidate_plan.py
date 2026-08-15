"""Freeze multi-source EXP13 candidate chunks before simulator outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_sparse_rl.metrics.exp11 import dct_modes, smooth_pulse_modes
from decision_sparse_rl.metrics.exp13 import fit_progress_direction, nearest_library, shift_chunk, temporal_warp
from scripts.exp11.run_replacement_stage import training_residual_modes
from scripts.exp12.prepare_ranking import TASKS, TASK_BODIES, object_arrays


ALL_FAMILIES = ("G1_multichannel", "G2_temporal", "G3_residual", "G4_library", "G5_progress", "G6_guided", "G7_gripper", "G8_composed")


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def progress_series(boundaries: list[dict], task: str) -> np.ndarray:
    if task in (TASKS[0], TASKS[2]):
        q = np.asarray([x["progress_channels"]["joint_qpos"] for x in boundaries], float)
        return (q - q[0]) / (q[-1] - q[0] + 1e-12)
    distance = np.asarray([x["progress_channels"]["bowl_to_plate_planar_distance_m"] for x in boundaries], float)
    return 1.0 - distance / max(distance[0], 1e-6)


def context(boundary: dict, task: str, phase: int) -> np.ndarray:
    pos, _ = object_arrays(boundary, task)
    padded = np.zeros((2, 3)); padded[:len(pos)] = pos
    return np.r_[boundary["eef_position"], padded.reshape(-1), np.eye(7)[phase]]


def load_reference_cache(reference_run: Path):
    manifest = json.loads((reference_run / "artifacts/reference_snapshots_manifest.json").read_text())
    cache = {}
    for record in manifest["episodes"]:
        directory = reference_run / record["relative_directory"]
        boundaries = json.loads((directory / "boundaries.json").read_text())
        with np.load(directory / "trajectory_states.npz", allow_pickle=False) as z:
            actions = np.asarray(z["actions"], float)
        phase = np.asarray([min(6, int(round(i / max(1, len(actions) - 1) * 6))) for i in range(len(actions))])
        cache[(record["task"], record["episode"])] = {"boundaries": boundaries, "actions": actions, "phase": phase, "progress": progress_series(boundaries, record["task"]), "directory": record["relative_directory"]}
    return cache


def training_examples(cache, task: str, excluded_episode: str):
    summaries, delta, contexts, chunks, episodes = [], [], [], [], []
    for (candidate_task, episode), item in cache.items():
        if candidate_task != task or episode == excluded_episode:
            continue
        actions, progress = item["actions"], item["progress"]
        for index in range(0, len(actions) - 10, 4):
            summaries.append(actions[index:index+10, :6].mean(0))
            delta.append(progress[index+10] - progress[index])
            contexts.append(context(item["boundaries"][index], task, int(item["phase"][index])))
            chunks.append(actions[index:index+10])
            episodes.append(episode)
    return map(np.asarray, (summaries, delta, contexts, chunks, episodes))


def simple_guidance_fit(exp12_rows: list[dict], task: str, excluded_demo: str):
    selected = [x for x in exp12_rows if x["task"] == task and not x["is_nominal"] and x["demo_key"] != excluded_demo]
    if len(selected) < 20:
        return None
    x = np.asarray([np.r_[np.mean(np.asarray(r["x_e1"])[-70:].reshape(10, 7)[:, :6], 0), np.std(np.asarray(r["x_e1"])[-70:].reshape(10, 7)[:, :6], 0)] for r in selected])
    y = np.asarray([r["composite_quality"] for r in selected])
    mu, sd = x.mean(0), x.std(0); sd[sd < 1e-8] = 1; z = (x-mu)/sd
    w = np.linalg.solve(z.T@z + np.eye(z.shape[1])*10, z.T@y)
    return mu, sd, w


def guidance_score(model, candidate):
    if model is None:
        return 0.0
    arm = np.asarray(candidate)[:, :6]
    feat = np.r_[arm.mean(0), arm.std(0)]
    mu, sd, w = model
    return float(((feat-mu)/sd)@w)


def candidate_pool(task, episode, t, phase, item, cache, training_run, phase_path, exp12_rows):
    actions, boundaries = item["actions"], item["boundaries"]
    ref = actions[t:t+10].copy(); phase = int(phase)
    summaries, progress_delta, library_context, library_chunks, library_episodes = training_examples(cache, task, episode)
    direction = fit_progress_direction(summaries, progress_delta, l2=3.0)
    top_channels = np.argsort(-np.abs(direction))[:3]
    candidates = []
    pulse = smooth_pulse_modes(10, 3)[0]; pulse /= max(np.max(np.abs(pulse)), 1e-12)
    for channel in top_channels:
        for sign in (-1, 1):
            desired = ref.copy(); desired[:, channel] += .08 * sign * pulse
            candidates.append(("G1_multichannel", f"pulse_c{channel}_{sign:+d}", desired))
    for shift in (-2, -1, 1, 2):
        candidates.append(("G2_temporal", f"shift_{shift:+d}", shift_chunk(actions, t, 10, shift)))
    for scale in (.8, 1.2):
        candidates.append(("G2_temporal", f"warp_{scale:.1f}", temporal_warp(ref, scale)))
    modes = training_residual_modes(task, phase, 10, training_run, phase_path)
    for mode_index, mode in enumerate(modes[:2]):
        for sign in (-1, 1):
            desired = ref.copy(); desired[:, :6] += .08 * sign * mode
            candidates.append(("G3_residual", f"residual_{mode_index}_{sign:+d}", desired))
    query = context(boundaries[t], task, phase)
    if len(library_context) >= 2:
        chosen = nearest_library(query, library_context, np.zeros(len(library_context), bool), min(2, len(library_context)))
        for rank, index in enumerate(chosen):
            candidates.append(("G4_library", f"nearest_{rank}_{library_episodes[index]}", library_chunks[index]))
    for amplitude in (.05, .10):
        desired = ref.copy(); desired[:, :6] += amplitude * direction[None, :] * pulse[:, None]
        candidates.append(("G5_progress", f"progress_{amplitude:.2f}", desired))
    signs = np.sign(actions[:, 6]); transitions = np.flatnonzero((signs[1:] != signs[:-1]) & (signs[1:] != 0) & (signs[:-1] != 0)) + 1
    for shift in (-2, -1, 1, 2):
        if any(t <= x < t + 10 for x in transitions):
            desired = ref.copy(); desired[:, 6] = actions[np.clip(np.arange(t, t+10)+shift, 0, len(actions)-1), 6]
            candidates.append(("G7_gripper", f"gripper_{shift:+d}", desired))
    for scale in (.8, 1.2):
        desired = temporal_warp(ref, scale); desired[:, :6] += .05 * direction[None, :] * pulse[:, None]
        candidates.append(("G8_composed", f"progress_warp_{scale:.1f}", desired))
    model = simple_guidance_fit(exp12_rows, task, f"{task}|{episode}")
    ranked = sorted(candidates, key=lambda x: guidance_score(model, x[2]), reverse=True)
    for rank, (_, source, desired) in enumerate(ranked[:2]):
        mixed = .5 * ref + .5 * desired; mixed[:, 6] = desired[:, 6]
        candidates.append(("G6_guided", f"guided_{rank}_{source}", mixed))
    return ref, candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("calibration", "formal"), required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--branch-manifest", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--training-run", type=Path, default=Path("runs/exp8_s2_independent_refs_20260814"))
    parser.add_argument("--phase-run", type=Path, default=Path("runs/exp10_a0_phase_macro_dataset_r2_20260814"))
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    if out.exists(): raise FileExistsError(f"immutable run exists: {out}")
    artifacts, manifests = out/"artifacts", out/"manifests"; artifacts.mkdir(parents=True); manifests.mkdir()
    reference = ROOT / args.reference_run
    branches = json.loads((ROOT / args.branch_manifest).read_text())
    cache = load_reference_cache(reference)
    exp12_rows = pq.read_table(ROOT/"runs/exp12_a0_candidate_dataset_r1_20260815/artifacts/candidate_dataset.parquet").to_pylist()
    authorization = {task: list(ALL_FAMILIES) for task in TASKS}
    if args.authorization:
        authorization = json.loads((ROOT / args.authorization).read_text())["authorized_by_task"]
    plan = []
    support = defaultdict(lambda: defaultdict(int))
    training_run = ROOT / args.training_run
    phase_path = ROOT / args.phase_run / "artifacts/reference_phase_labels.parquet"
    for branch in branches:
        key=(branch["task"], branch["episode"]); item=cache[key]; t=int(branch["branch_time"])
        ref, candidates = candidate_pool(branch["task"], branch["episode"], t, int(branch["phase"][1:]), item, cache, training_run, phase_path, exp12_rows)
        permitted = set(authorization[branch["task"]]); chosen=[]; family_count=defaultdict(int); seen=set()
        for family, source, desired in candidates:
            if family not in permitted: continue
            digest=hashlib.sha256(np.round(desired,10).tobytes()).hexdigest()
            if digest in seen: continue
            if args.stage=="formal" and (len(chosen)>=12 or family_count[family]>=3): continue
            seen.add(digest); family_count[family]+=1; chosen.append((family,source,desired)); support[branch["task"]][family]+=1
        for family, source, desired in chosen:
            basis=desired-ref
            plan.append({"branch_id":branch["branch_id"],"family":"I-A_analytic","generator_family":family,"basis_family":family,"candidate_source":source,"mode_index":family_count[family],"channel":-1,"chunk_length":10,"amplitude":1.0,"sign":1,"basis":basis.tolist()})
    for index,row in enumerate(plan): row["intervention_id"]=f"exp13_{args.stage}|r{index:05d}"
    dump(artifacts/"branch_manifest.json",branches);dump(artifacts/"candidate_plan.json",plan)
    summary={"stage":args.stage,"branch_count":len(branches),"candidate_count":len(plan),"support":{t:dict(v) for t,v in support.items()},"authorization":authorization,"frozen_before_outcomes":True}
    dump(artifacts/"candidate_support.json",summary)
    sources=[ROOT/args.branch_manifest,reference/"artifacts/reference_snapshots_manifest.json",ROOT/"experiments/exp13_candidate_generation/configs/exp13.json"]
    dump(manifests/"source_hashes.json",{str(x.relative_to(ROOT)):sha(x) for x in sources})
    dump(out/"metrics.json",summary);print(json.dumps(summary,indent=2));return 0


if __name__=="__main__": raise SystemExit(main())
