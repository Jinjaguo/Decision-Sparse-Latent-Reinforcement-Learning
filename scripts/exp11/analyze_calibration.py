"""Outcome analysis and plots for the frozen EXP11 calibration run."""

import argparse, json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def save(fig, path): fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-id",required=True); p.add_argument("--calibration-run",required=True,type=Path); a=p.parse_args()
    out=Path("runs")/a.run_id
    if out.exists(): raise FileExistsError(out)
    artifacts,plots=out/"artifacts",out/"plots"; artifacts.mkdir(parents=True);plots.mkdir()
    source=a.calibration_run; rows=pq.read_table(source/"artifacts/calibration_replacements.parquet").to_pylist()
    effects=np.asarray([r["macro_effect_h10"] for r in rows]); amplitudes=np.asarray([r["amplitude"] for r in rows]); families=sorted(set(r["family"] for r in rows)); phases=sorted(set(r["phase"] for r in rows)); tasks=sorted(set(r["task"] for r in rows))
    stats=[]
    for family in families:
        rr=[r for r in rows if r["family"]==family]; continuous=[r for r in rr if r["amplitude"] in (.05,.10)]
        amp_corr=float(np.corrcoef([r["amplitude"] for r in continuous],[r["macro_effect_h10"] for r in continuous])[0,1]) if len(set(r["amplitude"] for r in continuous))>1 else float("nan")
        by_pair=defaultdict(dict)
        for r in continuous: by_pair[(r["branch_id"],r["basis_family"],r["mode_index"],r["amplitude"])][r["sign"]]=r["macro_effect_h10"]
        asym=[abs(v[1]-v[-1])/(v[1]+v[-1]+1e-12) for v in by_pair.values() if -1 in v and 1 in v]
        stats.append({"family":family,"count":len(rr),"mean_effect":float(np.mean([r["macro_effect_h10"] for r in rr])),"median_effect":float(np.median([r["macro_effect_h10"] for r in rr])),"amplitude_effect_correlation":amp_corr,"paired_sign_asymmetry_mean":float(np.mean(asym)) if asym else float("nan"),"clipped_chunk_fraction":float(np.mean([r["clipped_chunk"] for r in rr]))})
    pq.write_table(pa.Table.from_pylist(stats),artifacts/"calibration_route_statistics.parquet",compression="zstd")

    fig,ax=plt.subplots(figsize=(8,4)); ax.hist(effects,bins=40); ax.axvline(.05,color="r",ls="--");ax.set(xlabel="H10 macro effect",ylabel="count",title="Calibration effect distribution");save(fig,plots/"calibration_effect_distribution.png")
    fig,ax=plt.subplots(figsize=(7,4))
    for family in families:
        rr=[r for r in rows if r["family"]==family and r["amplitude"] in (.05,.10)];
        if rr: ax.scatter([r["amplitude"] for r in rr],[r["macro_effect_h10"] for r in rr],s=8,alpha=.3,label=family)
    ax.set(xlabel="normalized action amplitude",ylabel="H10 macro effect");ax.legend(fontsize=7);save(fig,plots/"effect_vs_amplitude.png")
    fig,ax=plt.subplots(figsize=(8,4)); ax.bar(families,[np.mean([r["clipped_chunk"] for r in rows if r["family"]==f]) for f in families]);ax.set(ylabel="clipped chunk fraction",title="Execution clipping by family");ax.tick_params(axis="x",rotation=20);save(fig,plots/"clipping_rate_by_family.png")
    matrix=np.zeros((len(tasks),len(phases)))
    for i,t in enumerate(tasks):
        for j,ph in enumerate(phases):
            v=[r["macro_effect_h10"] for r in rows if r["task"]==t and r["phase"]==ph]; matrix[i,j]=np.mean(v) if v else np.nan
    fig,ax=plt.subplots(figsize=(8,4));im=ax.imshow(matrix,aspect="auto");ax.set_xticks(range(len(phases)),phases);ax.set_yticks(range(len(tasks)),[x[:12] for x in tasks]);fig.colorbar(im,ax=ax,label="mean H10 effect");save(fig,plots/"phase_effect_heatmap_calibration.png")
    metrics={"status":"completed","source_run":source.name,"replacement_count":len(rows),"statistics":stats,"strongest_phase_by_task":{t:max(phases,key=lambda ph:np.mean([r["macro_effect_h10"] for r in rows if r["task"]==t and r["phase"]==ph]) if any(r["task"]==t and r["phase"]==ph for r in rows) else -1) for t in tasks}}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8");print(json.dumps(metrics,indent=2))


if __name__=="__main__":main()
