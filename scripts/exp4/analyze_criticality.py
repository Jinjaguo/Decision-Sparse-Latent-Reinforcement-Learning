#!/usr/bin/env python
"""Run the frozen EXP4 GPU-audited confirmatory and secondary analyses."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import kendalltau, rankdata, spearmanr
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from decision_sparse_rl.logging.run_directory import write_json  # noqa: E402
from decision_sparse_rl.metrics.criticality import concentration_metrics  # noqa: E402
from decision_sparse_rl.metrics.exp4 import basis_rms, direction_pair_metrics, first_crossing_interpolate, gram_spectrum  # noqa: E402


SHORT = {"open_the_middle_drawer_of_the_cabinet": "drawer", "turn_on_the_stove": "stove", "put_the_bowl_on_the_plate": "bowl"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def verify_raw_hashes(artifacts: Path) -> Dict[str, str]:
    lock=json.loads((artifacts/"raw_artifact_hashes.json").read_text()); verified={name:sha256(artifacts/name) for name in lock["sha256"]}
    if verified!=lock["sha256"]: raise RuntimeError("raw artifact hash mismatch; analysis stopped")
    return verified


def equivalence_passes(comparisons: Dict[str, Any], tolerances: Dict[str, float]) -> bool:
    return comparisons["aggregation_s_rms_abs"]<=tolerances["scalar_atol"] and comparisons["concentration_max_abs"]<=tolerances["scalar_atol"] and comparisons["spearman_rank_inputs_exact"] and comparisons["bootstrap_max_abs"]<=tolerances["scalar_atol"] and comparisons["gram_max_abs"]<=tolerances["matrix_atol"] and comparisons["singular_values_max_abs"]<=tolerances["spectrum_atol"] and comparisons["interpolation_max_abs"]<=tolerances["scalar_atol"] and comparisons["interpolation_sources_exact"]


def group(rows: Iterable[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    result: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows: result.setdefault(tuple(row[key] for key in keys), []).append(row)
    return result


def save_table(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows: raise RuntimeError(f"refusing empty table: {path}")
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def rho(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64); b = np.asarray(right, dtype=np.float64)
    if a.size != b.size or a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]): return 0.0
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else 0.0


def tau(left: Sequence[float], right: Sequence[float]) -> float:
    value = float(kendalltau(left, right, variant="b").statistic)
    return value if np.isfinite(value) else 0.0


def gpu_rank(values: torch.Tensor) -> torch.Tensor:
    # torch 1.11 has no ``stable`` keyword. EXP4 checks rank inputs for ties before
    # formal use and the fixed equivalence vector is tie-free.
    if torch.unique(values).numel() != values.numel():
        raise RuntimeError("GPU rank path requires tie-free inputs under torch 1.11")
    order = torch.argsort(values); ranks = torch.empty_like(values, dtype=torch.float64); ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float64)
    return ranks


def gpu_rho(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    a = gpu_rank(left); b = gpu_rank(right); a = a - a.mean(); b = b - b.mean(); denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return torch.tensor(0.0, device=left.device, dtype=torch.float64) if denominator == 0 else (a @ b) / denominator


def gpu_concentration(values: torch.Tensor) -> Dict[str, torch.Tensor]:
    total = values.sum()
    if float(total) == 0.0: return {"top10_mass": total, "top20_mass": total, "top30_mass": total, "gini": total, "normalized_entropy": torch.ones_like(total)}
    ordered = torch.sort(values, descending=True).values; ascending = torch.flip(ordered, dims=[0]); n = values.numel(); index = torch.arange(1, n + 1, device=values.device, dtype=torch.float64); probabilities = values / total; positive = probabilities[probabilities > 0]
    return {"top10_mass": ordered[: math.ceil(.1*n)].sum()/total, "top20_mass": ordered[: math.ceil(.2*n)].sum()/total, "top30_mass": ordered[: math.ceil(.3*n)].sum()/total, "gini": 2*(index*ascending).sum()/(n*total)-(n+1)/n, "normalized_entropy": -(positive*torch.log(positive)).sum()/math.log(n)}


def gpu_first_crossing(progress: torch.Tensor, values: torch.Tensor, grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = []; sources = []
    for target in grid:
        chosen = None
        for i in range(progress.numel() - 1):
            low = torch.minimum(progress[i], progress[i+1]); high = torch.maximum(progress[i], progress[i+1])
            if bool((low <= target) & (target <= high)):
                if bool(progress[i] == progress[i+1]): value = values[i]
                else: value = values[i] + (target-progress[i])/(progress[i+1]-progress[i])*(values[i+1]-values[i])
                chosen = (value, i); break
        if chosen is None:
            distance = torch.abs(progress-target); index = int(torch.argmin(distance)); chosen = (values[index], index)
        outputs.append(chosen[0]); sources.append(chosen[1])
    return torch.stack(outputs), torch.tensor(sources, device=progress.device, dtype=torch.int64)


def icc_2_1(matrix: np.ndarray) -> float:
    x = np.asarray(matrix, dtype=np.float64); n, k = x.shape
    if n < 2 or k < 2: return 0.0
    grand = x.mean(); row_mean = x.mean(axis=1); col_mean = x.mean(axis=0); msr = k*np.sum((row_mean-grand)**2)/(n-1); msc = n*np.sum((col_mean-grand)**2)/(k-1); residual = x-row_mean[:,None]-col_mean[None,:]+grand; mse = np.sum(residual**2)/((n-1)*(k-1)); denominator = msr+(k-1)*mse+k*(msc-mse)/n
    return 0.0 if denominator == 0 else float((msr-mse)/denominator)


def jaccard_matrix(curves: List[np.ndarray], fraction: float = .2) -> np.ndarray:
    sets = []
    for curve in curves:
        count = int(np.ceil(len(curve)*fraction)); sets.append(set(np.argsort(curve)[-count:].tolist()))
    matrix = np.empty((len(sets), len(sets)))
    for i, a in enumerate(sets):
        for j, b in enumerate(sets): matrix[i,j] = len(a & b)/len(a | b)
    return matrix


def bh_adjust(values: Sequence[float]) -> List[float]:
    p = np.asarray(values, dtype=np.float64); order = np.argsort(p); ranked = p[order]; adjusted = np.minimum.accumulate((ranked*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]; out = np.empty_like(adjusted); out[order] = np.clip(adjusted,0,1); return out.tolist()


def adjusted_event_test(payload: List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]], rng: np.random.Generator, resamples: int) -> Tuple[float, float]:
    demo_names = [f"{task}|{episode}" for task, episode, _, _, _ in payload]; unique = sorted(set(demo_names)); rows=[]; targets=[]; masks=[]
    for (task, episode, times, values, mask), demo in zip(payload, demo_names):
        phase=np.minimum((times*3).astype(int),2); design=np.column_stack([np.ones(len(times)),mask.astype(float),times,(phase==1).astype(float),(phase==2).astype(float),*[np.full(len(times),demo==name,dtype=float) for name in unique[1:]]]); rows.append(design); targets.append(np.log(np.maximum(values,1e-15))); masks.append(mask)
    x=np.vstack(rows); y=np.concatenate(targets); observed=float(np.linalg.lstsq(x,y,rcond=None)[0][1]); null=[]
    for _ in range(resamples):
        offset=0; perm=[]
        for mask in masks: perm.append(rng.permutation(mask)); offset+=len(mask)
        xp=x.copy(); xp[:,1]=np.concatenate(perm).astype(float); null.append(float(np.linalg.lstsq(xp,y,rcond=None)[0][1]))
    p=float((1+np.sum(np.asarray(null)>=observed))/(1+resamples)); return observed,p


def gpu_audit_and_equivalence(interventions: List[Dict[str, Any]], device: torch.device, spec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    started=time.perf_counter(); first=next(iter(group(interventions,("task","episode","branch_time")).values())); basis=sorted([x for x in first if x["direction_role"]=="basis"],key=lambda x:(x["direction_index"],x["sign"])); effects=np.asarray([x["primary_remaining_horizon_mean"] for x in basis]).reshape(7,2); cpu_s=effects.mean(axis=1); cpu_rms=basis_rms(cpu_s); tensor=torch.tensor(effects,device=device,dtype=torch.float64); gpu_s=tensor.mean(dim=1); gpu_rms=torch.sqrt(torch.mean(gpu_s**2))
    vector=np.asarray([.2,.1,.4,.3,.7,.6,.5,.8,.9,.15,.25,.35]); cpu_c=concentration_metrics(vector); gpu_c=gpu_concentration(torch.tensor(vector,device=device,dtype=torch.float64))
    cpu_ranks=rankdata(vector,method="ordinal")-1; gpu_ranks=gpu_rank(torch.tensor(vector,device=device,dtype=torch.float64)).cpu().numpy()
    rng=np.random.default_rng(940099); indexes=rng.integers(0,len(vector),size=(128,len(vector))); cpu_boot=np.median(vector[indexes],axis=1); idx=torch.tensor(indexes,device=device); sampled=torch.sort(torch.tensor(vector,device=device)[idx],dim=1).values; midpoint=sampled.shape[1]//2; gpu_boot=((sampled[:,midpoint-1]+sampled[:,midpoint])/2).cpu().numpy()
    signed=np.stack([np.asarray(x["signed_output_remaining_horizon_mean"]) for x in basis]); columns=(signed[1::2]-signed[0::2]).T; cpu_operator=gram_spectrum(columns,.005); c=torch.tensor(columns,device=device,dtype=torch.float64)/(2*.005); gpu_gram=c.T@c; gpu_singular=torch.sqrt(torch.clamp(torch.linalg.eigvalsh(gpu_gram),min=0)).flip(0)
    p=np.asarray([0,.8,.4,1.]); y=np.asarray([0.,8.,4.,10.]); grid=np.linspace(0,1,11); cpu_interp,cpu_src=first_crossing_interpolate(p,y,grid); gpu_interp,gpu_src=gpu_first_crossing(torch.tensor(p,device=device),torch.tensor(y,device=device),torch.tensor(grid,device=device))
    comparisons={"aggregation_s_rms_abs":abs(cpu_rms-float(gpu_rms.cpu())),"concentration_max_abs":max(abs(cpu_c[name]-float(gpu_c[name].cpu())) for name in gpu_c),"spearman_rank_inputs_exact":bool(np.array_equal(cpu_ranks,gpu_ranks)),"bootstrap_max_abs":float(np.max(np.abs(cpu_boot-gpu_boot))),"gram_max_abs":float(np.max(np.abs(cpu_operator["gram"]-gpu_gram.cpu().numpy()))),"singular_values_max_abs":float(np.max(np.abs(cpu_operator["singular_values"]-gpu_singular.cpu().numpy()))),"interpolation_max_abs":float(np.max(np.abs(cpu_interp-gpu_interp.cpu().numpy()))),"interpolation_sources_exact":bool(np.array_equal(cpu_src,gpu_src.cpu().numpy()))}
    tol=spec["tolerances"]; passed=equivalence_passes(comparisons,tol)
    props=torch.cuda.get_device_properties(device); smi=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total,memory.free","--format=csv,noheader"],text=True).strip(); audit={"torch_version":torch.__version__,"torch_cuda_version":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),"device":str(device),"device_index":torch.cuda.current_device(),"device_name":torch.cuda.get_device_name(device),"device_total_memory_bytes":props.total_memory,"nvidia_smi":smi,"dtype":"float64","calibration_wall_time_seconds":time.perf_counter()-started,"fallback_used":False}
    return audit,{"passed":passed,"tolerances":tol,"comparisons":comparisons,"calibration_branch":{key:first[0][key] for key in ("task","episode","branch_time")},"cpu_truth_retained":True}


def hierarchical_bootstrap_gpu(effect_tensor: torch.Tensor, resamples: int, seed: int) -> np.ndarray:
    device=effect_tensor.device; gen=torch.Generator(device=device); gen.manual_seed(seed); chunks=[]; batch=128
    for start in range(0,resamples,batch):
        b=min(batch,resamples-start); task_idx=torch.randint(0,3,(b,3),generator=gen,device=device); demo_idx=torch.randint(0,7,(b,3,7),generator=gen,device=device); samples=[]
        for bi in range(b):
            demos=[]
            for ti in range(3):
                task=int(task_idx[bi,ti]); rows=effect_tensor[task,demo_idx[bi,ti]]  # 7x12x7x2
                dir_idx=torch.randint(0,7,(7,12,7),generator=gen,device=device); sign_idx=torch.randint(0,2,(7,12,7,2),generator=gen,device=device); dgrid=torch.arange(7,device=device)[:,None,None].expand(7,12,7); bgrid=torch.arange(12,device=device)[None,:,None].expand(7,12,7); selected=rows[dgrid,bgrid,dir_idx]; selected=torch.gather(selected,3,sign_idx).mean(dim=3); branch=torch.sqrt(torch.mean(selected**2,dim=2)); ordered=torch.sort(branch,dim=1,descending=True).values; top=ordered[:,:3].sum(dim=1)/torch.clamp(branch.sum(dim=1),min=1e-15); demos.append(top)
            samples.append(torch.median(torch.cat(demos)))
        chunks.append(torch.stack(samples).cpu())
    return torch.cat(chunks).numpy()


def variance_components(effect_tensor_all: np.ndarray) -> Dict[str, Any]:
    y=np.asarray(effect_tensor_all,dtype=np.float64); grand=y.mean(); total=float(np.sum((y-grand)**2)); shape=y.shape
    task=y.mean(axis=(1,2,3,4)); demo=y.mean(axis=(2,3,4)); progress=y.mean(axis=(0,1,3,4)); direction=y.mean(axis=(0,1,2,4)); sign=y.mean(axis=(0,1,2,3))
    ss_task=shape[1]*shape[2]*shape[3]*shape[4]*np.sum((task-grand)**2); ss_demo=shape[2]*shape[3]*shape[4]*np.sum((demo-task[:,None])**2); ss_progress=shape[0]*shape[1]*shape[3]*shape[4]*np.sum((progress-grand)**2); ss_direction=shape[0]*shape[1]*shape[2]*shape[4]*np.sum((direction-grand)**2); ss_sign=shape[0]*shape[1]*shape[2]*shape[3]*np.sum((sign-grand)**2); explained=ss_task+ss_demo+ss_progress+ss_direction+ss_sign; residual=max(total-explained,0.0); values={"task":ss_task,"demo_within_task":ss_demo,"progress_position":ss_progress,"direction":ss_direction,"sign":ss_sign,"residual_and_interactions":residual}
    return {"method":"balanced orthogonal ANOVA sum-of-squares decomposition; demo nested within task; residual contains interactions and within-cell variation","total_sum_squares":total,"sum_squares":values,"variance_share":{k:(float(v/total) if total else 0.0) for k,v in values.items()}}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-dir",type=Path,required=True); args=parser.parse_args(); run=args.run_dir.resolve(); artifacts=run/"artifacts"; manifests=artifacts/"frozen_manifests"
    verified=verify_raw_hashes(artifacts)
    metrics=json.loads((run/"metrics.json").read_text());
    if not metrics.get("gate",{}).get("passed"): raise RuntimeError("formal execution gate failed")
    interventions=pq.read_table(artifacts/"interventions.parquet").to_pylist(); steps=pq.read_table(artifacts/"per_step_effects.parquet").to_pylist(); gpu_spec=json.loads((manifests/"gpu_analysis_spec.json").read_text()); progress_spec=json.loads((manifests/"progress_alignment_spec.json").read_text()); event_manifest=json.loads((manifests/"event_manifest.json").read_text()); sap=json.loads((manifests/"statistical_analysis_plan.json").read_text())
    if not torch.cuda.is_available(): raise RuntimeError("EXP4 formal GPU path requires CUDA")
    device=torch.device("cuda:0"); torch.cuda.set_device(device); audit,equivalence=gpu_audit_and_equivalence(interventions,device,gpu_spec); write_json(artifacts/"gpu_audit.json",audit); write_json(artifacts/"gpu_cpu_equivalence.json",equivalence)
    if not equivalence["passed"]: raise RuntimeError("CPU/GPU equivalence gate failed")
    analysis_started=time.perf_counter(); by_branch=group(interventions,("task","episode","branch_time")); branch_keys=sorted(by_branch); direction_rows=[]; branch_rows=[]; operator_rows=[]; branch_primary_gpu=[]; effect_tensor=np.empty((3,7,12,7,2)); effect_tensor_all=np.empty((3,7,12,8,2)); task_names=sorted({x["task"] for x in interventions}); task_index={x:i for i,x in enumerate(task_names)}
    for key in branch_keys:
        rows=by_branch[key]; first=rows[0]; paired={}
        for di in range(8):
            drows=sorted([x for x in rows if x["direction_index"]==di],key=lambda x:x["sign"]); pair=direction_pair_metrics(drows[1]["primary_remaining_horizon_mean"],drows[0]["primary_remaining_horizon_mean"],1e-15); paired[di]=(pair,drows)
            direction_rows.append({"task":key[0],"episode":key[1],"branch_time":key[2],"branch_normalized_time":first["branch_normalized_time"],"physical_progress_raw":first["physical_progress_raw"],"physical_progress_clipped":first["physical_progress_clipped"],"direction_index":di,"direction_role":drows[0]["direction_role"],"symmetric_effect":pair["symmetric"],"asymmetry":pair["asymmetry"],"negative_effect":drows[0]["primary_remaining_horizon_mean"],"positive_effect":drows[1]["primary_remaining_horizon_mean"]})
        basis_values=np.asarray([paired[i][0]["symmetric"] for i in range(7)]); gpu_value=float(torch.sqrt(torch.mean(torch.tensor(basis_values,device=device,dtype=torch.float64)**2)).cpu()); branch_primary_gpu.append(gpu_value); historical=float(np.median([x["primary_remaining_horizon_mean"] for x in rows])); terminal=np.median([x["terminal_object_position_l2"]+x["terminal_object_orientation_geodesic_mean"]/np.pi for x in rows]); pred=np.median([x["predicate_divergence_fraction"] for x in rows])
        branch_rows.append({"task":key[0],"episode":key[1],"branch_time":key[2],"branch_normalized_time":first["branch_normalized_time"],"physical_progress_raw":first["physical_progress_raw"],"physical_progress_clipped":first["physical_progress_clipped"],"branch_kind":first["branch_kind"],"remaining_horizon":first["remaining_horizon"],"primary_s_rms":gpu_value,"historical_exp3_style_median":historical,"basis_s_median":float(np.median(basis_values)),"basis_s_maximum":float(np.max(basis_values)),"basis_s_cv":float(np.std(basis_values)/max(np.mean(basis_values),1e-15)),"largest_direction_share":float(np.max(basis_values)/max(np.sum(basis_values),1e-15)),"heldout_random_s":paired[7][0]["symmetric"],"basis_asymmetry_median":float(np.median([paired[i][0]["asymmetry"] for i in range(7)])),"terminal_object_effect_median":float(terminal),"predicate_divergence_fraction_median":float(pred),"success_flip_count":int(sum(x["success_flip"] for x in rows))})
        columns=[]
        for di in range(7):
            drows=paired[di][1]; negative=np.asarray(drows[0]["signed_output_remaining_horizon_mean"]); positive=np.asarray(drows[1]["signed_output_remaining_horizon_mean"]); columns.append(positive-negative)
        c=torch.tensor(np.stack(columns,axis=1),device=device,dtype=torch.float64)/(2*.005); gram=c.T@c; eig=torch.clamp(torch.linalg.eigvalsh(gram),min=0).flip(0); singular=torch.sqrt(eig); eig_np=eig.cpu().numpy(); sing_np=singular.cpu().numpy(); positive_eig=eig_np[eig_np>1e-15]
        operator_rows.append({"task":key[0],"episode":key[1],"branch_time":key[2],"branch_normalized_time":first["branch_normalized_time"],"physical_progress_clipped":first["physical_progress_clipped"],"valid":True,"output_dimension":int(c.shape[0]),"gram_matrix":gram.cpu().numpy().tolist(),"eigenvalues":eig_np.tolist(),"singular_values":sing_np.tolist(),"spectral_norm":float(sing_np[0]),"frobenius_norm":float(torch.linalg.vector_norm(c).cpu()),"effective_rank":float(np.exp(-np.sum((eig_np/eig_np.sum())*np.log(np.maximum(eig_np/eig_np.sum(),1e-15))))) if eig_np.sum()>0 else 0.0,"condition_number_nonzero":float(np.sqrt(positive_eig[0]/positive_eig[-1])) if len(positive_eig)>1 else 0.0,"top_eigenvalue_share":float(eig_np[0]/eig_np.sum()) if eig_np.sum()>0 else 0.0})
    branch_rows.sort(key=lambda x:(x["task"],x["episode"],x["branch_time"])); save_table(branch_rows,artifacts/"branch_summary.parquet"); save_table(direction_rows,artifacts/"direction_resolved_summary.parquet"); save_table(operator_rows,artifacts/"operator_summary.parquet")
    branch_demo=group(branch_rows,("task","episode")); demo_keys=sorted(branch_demo); demo_concentration=[]; robustness=[]; progress_rows=[]; grid=np.asarray(progress_spec["physical_progress_grid"]); time_curves={}; progress_curves={}; source_curves={}
    for task,episode in demo_keys:
        rows=sorted(branch_demo[(task,episode)],key=lambda x:x["branch_normalized_time"]); values=[x["primary_s_rms"] for x in rows]; cm=concentration_metrics(values); demo_concentration.append({"task":task,"episode":episode,**cm,"total_effect_mass":float(sum(values))})
        dcurves=[]
        for di in range(7): dcurves.append([next(x["symmetric_effect"] for x in direction_rows if x["task"]==task and x["episode"]==episode and x["branch_time"]==r["branch_time"] and x["direction_index"]==di) for r in rows])
        direction_rhos=[rho(dcurves[a],dcurves[b]) for a,b in itertools.combinations(range(7),2)]; random_curve=[next(x["symmetric_effect"] for x in direction_rows if x["task"]==task and x["episode"]==episode and x["branch_time"]==r["branch_time"] and x["direction_index"]==7) for r in rows]; random_rho=rho(values,random_curve); robustness.append({"task":task,"episode":episode,"median_basis_direction_spearman":float(np.median(direction_rhos)),"p25":float(np.percentile(direction_rhos,25)),"p75":float(np.percentile(direction_rhos,75)),"stable_at_0_5":float(np.median(direction_rhos))>=.5,"heldout_random_vs_basis_spearman":random_rho})
        quantile=sorted([x for x in rows if x["branch_kind"]=="temporal_quantile"],key=lambda x:x["branch_normalized_time"]); time_curves[(task,episode)]=np.asarray([x["primary_s_rms"] for x in quantile]); p=torch.tensor([x["physical_progress_clipped"] for x in quantile],device=device,dtype=torch.float64); y=torch.tensor([x["primary_s_rms"] for x in quantile],device=device,dtype=torch.float64); aligned,source=gpu_first_crossing(p,y,torch.tensor(grid,device=device,dtype=torch.float64)); progress_curves[(task,episode)]=aligned.cpu().numpy(); source_curves[(task,episode)]=source.cpu().numpy()
        for gi,(value,source_index) in enumerate(zip(progress_curves[(task,episode)],source_curves[(task,episode)])): progress_rows.append({"task":task,"episode":episode,"grid_index":gi,"physical_progress":float(grid[gi]),"aligned_primary_s_rms":float(value),"source_temporal_segment_index":int(source_index)})
    save_table(progress_rows,artifacts/"progress_aligned_curves.parquet"); save_table(demo_concentration,artifacts/"demo_concentration.parquet"); save_table(robustness,artifacts/"direction_robustness.parquet")
    replication=[]
    for task in task_names:
        demos=sorted([key for key in demo_keys if key[0]==task])
        for a,b in itertools.combinations(demos,2):
            tr=rho(time_curves[a],time_curves[b]); pr=rho(progress_curves[a],progress_curves[b]); replication.append({"task":task,"demo_a":a[1],"demo_b":b[1],"time_spearman":tr,"progress_spearman":pr,"delta_spearman":pr-tr,"time_kendall_tau_b":tau(time_curves[a],time_curves[b]),"progress_kendall_tau_b":tau(progress_curves[a],progress_curves[b]),"time_icc_2_1":None,"progress_icc_2_1":None,"time_top20_jaccard_mean":None,"progress_top20_jaccard_mean":None})
    for task in task_names:
        demos=sorted([key for key in demo_keys if key[0]==task]); tmat=np.stack([time_curves[x] for x in demos]); pmat=np.stack([progress_curves[x] for x in demos]); replication.append({"task":task,"demo_a":"TASK_ICC","demo_b":"TASK_ICC","time_spearman":float("nan"),"progress_spearman":float("nan"),"delta_spearman":float("nan"),"time_kendall_tau_b":float("nan"),"progress_kendall_tau_b":float("nan"),"time_icc_2_1":icc_2_1(tmat),"progress_icc_2_1":icc_2_1(pmat),"time_top20_jaccard_mean":float(np.mean(jaccard_matrix(list(tmat))[np.triu_indices(7,1)])),"progress_top20_jaccard_mean":float(np.mean(jaccard_matrix(list(pmat))[np.triu_indices(7,1)]))})
    save_table(replication,artifacts/"replication_summary.parquet")
    # Balanced tensors and formal GPU bootstrap.
    branch_ordinal={(task,episode,row["branch_time"]):i for task,episode in demo_keys for i,row in enumerate(sorted(branch_demo[(task,episode)],key=lambda x:x["branch_normalized_time"]))}; demo_ordinal={(task,episode):int(episode.split("_")[-1])-3 for task,episode in demo_keys}
    for row in interventions:
        ti=task_index[row["task"]]; di=demo_ordinal[(row["task"],row["episode"])]; bi=branch_ordinal[(row["task"],row["episode"],row["branch_time"])]; si=0 if row["sign"]==-1 else 1; effect_tensor_all[ti,di,bi,row["direction_index"],si]=row["primary_remaining_horizon_mean"]
        if row["direction_index"]<7: effect_tensor[ti,di,bi,row["direction_index"],si]=row["primary_remaining_horizon_mean"]
    bootstrap=hierarchical_bootstrap_gpu(torch.tensor(effect_tensor,device=device,dtype=torch.float64),sap["bootstrap"]["resamples"],sap["bootstrap"]["seed"]); top20=np.asarray([x["top20_mass"] for x in demo_concentration]); top20_median=float(np.median(top20)); ci=[float(np.percentile(bootstrap,2.5)),float(np.percentile(bootstrap,97.5))]
    pair_rows=[x for x in replication if x["demo_a"]!="TASK_ICC"]; deltas=np.asarray([x["delta_spearman"] for x in pair_rows]); observed_delta=float(np.median(deltas)); task_progress={task:float(np.median([x["progress_spearman"] for x in pair_rows if x["task"]==task])) for task in task_names}; task_time={task:float(np.median([x["time_spearman"] for x in pair_rows if x["task"]==task])) for task in task_names}
    rng=np.random.default_rng(sap["permutation"]["seed"]); task_delta=np.asarray([np.median([x["delta_spearman"] for x in pair_rows if x["task"]==task]) for task in task_names]); signs=rng.choice([-1,1],size=(sap["permutation"]["resamples"],3)); null=np.median(signs*task_delta,axis=1); alignment_p=float((1+np.sum(null>=observed_delta))/(1+len(null))); cluster_boot=[]
    for _ in range(sap["bootstrap"]["resamples"]): sampled_tasks=rng.choice(task_names,3,replace=True); vals=[]; [vals.extend(rng.choice([x["delta_spearman"] for x in pair_rows if x["task"]==task],21,replace=True)) for task in sampled_tasks]; cluster_boot.append(float(np.median(vals)))
    alignment_ci=[float(np.percentile(cluster_boot,2.5)),float(np.percentile(cluster_boot,97.5))]
    lodo=[]
    for omitted in demo_keys: lodo.append({"omitted_task":omitted[0],"omitted_episode":omitted[1],"median_top20_mass":float(np.median([x["top20_mass"] for x in demo_concentration if (x["task"],x["episode"])!=omitted]))})
    save_table(lodo,artifacts/"leave_one_demo_out.parquet"); loto=[]
    for task in task_names: loto.append({"omitted_task":task,"median_top20_mass":float(np.median([x["top20_mass"] for x in demo_concentration if x["task"]!=task]))})
    save_table(loto,artifacts/"leave_one_task_out.parquet")
    variance=variance_components(effect_tensor_all); write_json(artifacts/"variance_components.json",variance)
    # Secondary event analysis using frozen controls and within-demo permutation.
    rng_event=np.random.default_rng(940032); event_results=[]
    for event_type in sorted({x["event_type"] for x in event_manifest["events"]}):
        payload=[]; ratios=[]
        for event in [x for x in event_manifest["events"] if x["event_type"]==event_type and x["present"]]:
            rows=branch_demo[(event["task"],event["episode"])]; times=np.asarray([x["branch_normalized_time"] for x in rows]); values=np.asarray([x["primary_s_rms"] for x in rows]); mask=np.abs(times-event["normalized_time"])<=event_manifest["time_window_radius"]
            if mask.any() and (~mask).any(): ratios.append(float(values[mask].mean()/max(values[~mask].mean(),1e-15))); payload.append((event["task"],event["episode"],times,values,mask))
        beta,p=adjusted_event_test(payload,rng_event,sap["permutation"]["resamples"]) if payload else (float("nan"),float("nan")); event_results.append({"event_type":event_type,"present_demo_count":len(payload),"median_enrichment_ratio":float(np.median(ratios)) if ratios else float("nan"),"adjusted_log_effect_coefficient":beta,"adjusted_permutation_p_one_sided":p})
    q=bh_adjust([x["adjusted_permutation_p_one_sided"] for x in event_results]); [row.update({"adjusted_permutation_q_bh":value}) for row,value in zip(event_results,q)]; save_table(event_results,artifacts/"event_enrichment.parquet")
    stable_fraction=float(np.mean([x["stable_at_0_5"] for x in robustness])); random_agreement=float(np.median([x["heldout_random_vs_basis_spearman"] for x in robustness])); lodo_min=float(min(x["median_top20_mass"] for x in lodo)); task_pass=sum(value>=.5 for value in task_progress.values()); strong=top20_median>=.5 and ci[0]>.25 and observed_delta>=.15 and task_pass>=2 and stable_fraction>=.70 and lodo_min>=.45 and random_agreement>=.5
    if strong: classification="strong_replicated_progress_aligned_sparsity"
    elif ci[0]>.25: classification="replicated_nonuniformity_without_aligned_sparse_times"
    elif task_pass==1 and observed_delta>=.15: classification="task_specific_progress_alignment"
    elif stable_fraction<.70: classification="direction_instability"
    elif ci[0]<=.25: classification="uniform_or_broad_sensitivity"
    else: classification="no_support"
    terminal_rho=rho([x["primary_s_rms"] for x in branch_rows],[x["terminal_object_effect_median"] for x in branch_rows]); predicate_rho=rho([x["primary_s_rms"] for x in branch_rows],[x["predicate_divergence_fraction_median"] for x in branch_rows]); statistics={"primary":{"heldout_demo_median_top20_mass":top20_median,"hierarchical_bootstrap_95ci":ci,"uniform_null":.25},"alignment":{"median_delta_spearman":observed_delta,"cluster_bootstrap_95ci":alignment_ci,"cluster_permutation_p_one_sided":alignment_p,"task_time_median_spearman":task_time,"task_progress_median_spearman":task_progress},"direction":{"stable_demo_fraction":stable_fraction,"stable_demo_count":int(sum(x["stable_at_0_5"] for x in robustness)),"heldout_random_median_spearman":random_agreement},"lodo_min_top20":lodo_min,"terminal_object_spearman":terminal_rho,"predicate_divergence_spearman":predicate_rho,"success_flip_count":int(sum(x["success_flip"] for x in interventions)),"event_results":event_results,"counts":{"demos":21,"branches":252,"interventions":4032,"per_step_effects":len(steps)},"raw_hashes_verified":True}
    checks={"heldout_median_top20_at_least_0_50":top20_median>=.5,"bootstrap_lower_above_uniform":ci[0]>.25,"progress_delta_at_least_0_15":observed_delta>=.15,"two_tasks_progress_rho_at_least_0_50":task_pass>=2,"direction_robustness_at_least_70_percent":stable_fraction>=.70,"lodo_min_at_least_0_45":lodo_min>=.45,"heldout_random_agreement_at_least_0_50":random_agreement>=.5}; scientific={"classification":classification,"strong_rule_passed":strong,"boolean_checks":checks,"observed_statistics":statistics,"frozen_rule":json.loads((manifests/"scientific_decision_rule.json").read_text())}; write_json(artifacts/"statistical_results.json",statistics); write_json(artifacts/"scientific_decision.json",scientific)
    # Required plots.
    plot=artifacts/"plots"; plot.mkdir(exist_ok=True); labels=[f"{SHORT[t]}/{e}" for t,e in demo_keys]; colors={task:plt.cm.tab10(i) for i,task in enumerate(task_names)}
    fig,axes=plt.subplots(7,3,figsize=(15,20));
    for ax,key in zip(axes.flat,demo_keys): rows=sorted(branch_demo[key],key=lambda x:x["branch_normalized_time"]); ax.plot([x["branch_normalized_time"] for x in rows],[x["primary_s_rms"] for x in rows],"o-"); ax.set_title(f"{SHORT[key[0]]}/{key[1]}"); ax.set_yscale("symlog",linthresh=1e-8)
    fig.supxlabel("Normalized time"); fig.supylabel("S_RMS"); fig.tight_layout(); fig.savefig(plot/"heldout_criticality_vs_normalized_time_per_demo.png",dpi=150); plt.close(fig)
    fig,axes=plt.subplots(7,3,figsize=(15,20));
    for ax,key in zip(axes.flat,demo_keys): ax.plot(grid,progress_curves[key],"o-"); ax.set_title(f"{SHORT[key[0]]}/{key[1]}"); ax.set_yscale("symlog",linthresh=1e-8)
    fig.supxlabel("Physical progress"); fig.supylabel("Aligned S_RMS"); fig.tight_layout(); fig.savefig(plot/"heldout_criticality_vs_physical_progress_per_demo.png",dpi=150); plt.close(fig)
    pairs=[x for x in replication if x["demo_a"]!="TASK_ICC"]; fig,ax=plt.subplots(figsize=(7,6)); ax.scatter([x["time_spearman"] for x in pairs],[x["progress_spearman"] for x in pairs],c=[colors[x["task"]] for x in pairs]); ax.plot([-1,1],[-1,1],"k--"); ax.set(xlabel="Normalized-time rho",ylabel="Physical-progress rho"); fig.tight_layout(); fig.savefig(plot/"time_vs_progress_crossdemo_rho.png",dpi=170); plt.close(fig)
    heat=np.zeros((7,21));
    for di in range(7):
        curves=[]
        for key in demo_keys:
            rows=sorted([x for x in direction_rows if x["task"]==key[0] and x["episode"]==key[1] and x["direction_index"]==di],key=lambda x:x["branch_normalized_time"]); qrows=[x for x in rows if next(b["branch_kind"] for b in branch_demo[key] if b["branch_time"]==x["branch_time"])=="temporal_quantile"]; aligned,_=first_crossing_interpolate([x["physical_progress_clipped"] for x in qrows],[x["symmetric_effect"] for x in qrows],grid); curves.append(aligned)
        heat[di]=np.mean(curves,axis=0)
    fig,ax=plt.subplots(figsize=(11,4)); im=ax.imshow(heat,aspect="auto",origin="lower"); fig.colorbar(im,ax=ax,label="Mean symmetric effect"); ax.set(xlabel="Progress grid index",ylabel="Basis direction"); fig.tight_layout(); fig.savefig(plot/"direction_basis_heatmap_by_progress.png",dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); basis_dir=[x for x in direction_rows if x["direction_role"]=="basis"]; ax.scatter([x["symmetric_effect"] for x in basis_dir],[x["asymmetry"] for x in basis_dir],alpha=.25,s=10); ax.set(xlabel="Sign-symmetric S",ylabel="Asymmetry A",xscale="symlog"); fig.tight_layout(); fig.savefig(plot/"sign_symmetric_vs_asymmetry.png",dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5)); ax.bar(np.arange(21),[x["heldout_random_vs_basis_spearman"] for x in robustness]); ax.axhline(.5,ls="--",c="k"); ax.set_xticks(np.arange(21),labels,rotation=55,ha="right",fontsize=7); ax.set_ylabel("Held-out random vs basis rho"); fig.tight_layout(); fig.savefig(plot/"heldout_random_direction_agreement.png",dpi=170); plt.close(fig)
    exp3=pq.read_table(REPOSITORY_ROOT/"runs/exp3_t6_full_20260814T022200/artifacts/branch_summary.parquet").to_pylist(); exp3_group=group(exp3,("task","episode")); fig,ax=plt.subplots(figsize=(7,6));
    for cohort,curves,style in (("EXP3",[[x["primary_median"] for x in rows] for rows in exp3_group.values()],"--"),("EXP4",[[x["primary_s_rms"] for x in rows] for rows in branch_demo.values()],"-")):
        ys=[]
        for values in curves: ordered=np.sort(values)[::-1]; ys.append(np.cumsum(ordered)/ordered.sum())
        ax.plot(np.arange(1,13)/12,np.mean(ys,axis=0),style,label=cohort)
    ax.plot([0,1],[0,1],"k:"); ax.legend(); ax.set(xlabel="Fraction of branch points",ylabel="Cumulative effect mass"); fig.tight_layout(); fig.savefig(plot/"exp3_vs_exp4_concentration_curves.png",dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5)); ax.bar(np.arange(21),[x["top20_mass"] for x in demo_concentration]); ax.axhline(.25,ls="--",c="k"); ax.set_xticks(np.arange(21),labels,rotation=55,ha="right",fontsize=7); ax.set_ylabel("Top-20 mass"); fig.tight_layout(); fig.savefig(plot/"top20_mass_heldout_per_demo.png",dpi=170); plt.close(fig)
    anis=[np.median([r["basis_s_cv"] for r in branch_demo[key]]) for key in demo_keys]; fig,ax=plt.subplots(figsize=(12,5)); ax.bar(np.arange(21),anis); ax.set_xticks(np.arange(21),labels,rotation=55,ha="right",fontsize=7); ax.set_ylabel("Median direction CV"); fig.tight_layout(); fig.savefig(plot/"direction_anisotropy_per_demo.png",dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); ax.scatter([x["physical_progress_clipped"] for x in operator_rows],[x["spectral_norm"] for x in operator_rows],alpha=.5); ax.set(xlabel="Physical progress",ylabel="Operator spectral norm",yscale="symlog"); fig.tight_layout(); fig.savefig(plot/"operator_spectrum_by_progress.png",dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); names=list(variance["variance_share"]); ax.bar(names,[variance["variance_share"][x] for x in names]); ax.tick_params(axis="x",rotation=35); ax.set_ylabel("ANOVA variance share"); fig.tight_layout(); fig.savefig(plot/"variance_components.png",dpi=170); plt.close(fig)
    time_j=jaccard_matrix([time_curves[k] for k in demo_keys]); prog_j=jaccard_matrix([progress_curves[k] for k in demo_keys]);
    for matrix,name in ((time_j,"topk_overlap_matrix_time.png"),(prog_j,"topk_overlap_matrix_progress.png")):
        fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(matrix,vmin=0,vmax=1); fig.colorbar(im,ax=ax); ax.set_title("Top-20 Jaccard"); fig.tight_layout(); fig.savefig(plot/name,dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5)); ax.bar(np.arange(21),[x["median_top20_mass"] for x in lodo]); ax.axhline(.45,ls="--",c="k"); ax.set_xticks(np.arange(21),[f"{SHORT[x['omitted_task']]}/{x['omitted_episode']}" for x in lodo],rotation=55,ha="right",fontsize=7); fig.tight_layout(); fig.savefig(plot/"lodo_summary.png",dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); ax.bar([SHORT[x["omitted_task"]] for x in loto],[x["median_top20_mass"] for x in loto]); ax.axhline(.45,ls="--",c="k"); fig.tight_layout(); fig.savefig(plot/"leave_one_task_out_summary.png",dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); ax.bar([x["event_type"] for x in event_results],[x["adjusted_log_effect_coefficient"] for x in event_results]); ax.axhline(0,c="k",ls="--"); ax.tick_params(axis="x",rotation=20); ax.set_ylabel("Adjusted log coefficient"); fig.tight_layout(); fig.savefig(plot/"event_enrichment_adjusted.png",dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); ax.scatter([x["terminal_object_effect_median"] for x in branch_rows],[x["primary_s_rms"] for x in branch_rows],alpha=.45); ax.set(xlabel="Terminal object effect",ylabel="S_RMS",xscale="symlog",yscale="symlog"); fig.tight_layout(); fig.savefig(plot/"terminal_outcome_relevance.png",dpi=170); plt.close(fig)
    labels_eq=list(equivalence["comparisons"]); vals=[0 if isinstance(equivalence["comparisons"][x],bool) and equivalence["comparisons"][x] else (1 if isinstance(equivalence["comparisons"][x],bool) else equivalence["comparisons"][x]) for x in labels_eq]; fig,ax=plt.subplots(figsize=(10,5)); ax.bar(np.arange(len(vals)),vals); ax.set_yscale("symlog",linthresh=1e-16); ax.set_xticks(np.arange(len(vals)),labels_eq,rotation=55,ha="right",fontsize=7); ax.set_ylabel("CPU/GPU discrepancy (boolean pass -> 0)"); fig.tight_layout(); fig.savefig(plot/"gpu_cpu_equivalence.png",dpi=170); plt.close(fig)
    audit["formal_analysis"]={"device":str(device),"device_name":torch.cuda.get_device_name(device),"dtype":"float64","batch_size_bootstrap":128,"bootstrap_resamples":sap["bootstrap"]["resamples"],"permutation_resamples":sap["permutation"]["resamples"],"memory_allocated_bytes":torch.cuda.memory_allocated(device),"max_memory_allocated_bytes":torch.cuda.max_memory_allocated(device),"memory_reserved_bytes":torch.cuda.memory_reserved(device),"wall_time_seconds":time.perf_counter()-analysis_started,"fallback_used":False,"operations_executed_on_gpu":["basis aggregation","Gram/eigenspectrum","progress interpolation","hierarchical bootstrap","rank calibration"]}; write_json(artifacts/"gpu_audit.json",audit)
    metrics["analysis"]={"status":"completed","scientific_classification":classification,**statistics}; write_json(run/"metrics.json",metrics); (run/"analysis_command.txt").write_text(" ".join(sys.argv)+"\n",encoding="utf-8"); print(json.dumps(scientific,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
