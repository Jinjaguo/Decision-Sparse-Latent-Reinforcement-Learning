#!/usr/bin/env python
"""Create a compact, reproducible descriptive summary of locked EXP7 results."""

from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def describe(values):
    x=np.asarray(values,float); return {"count":int(len(x)),"median":float(np.median(x)),"mean":float(np.mean(x)),"p05":float(np.quantile(x,.05)),"p95":float(np.quantile(x,.95))}
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--analysis-run",type=Path,required=True); p.add_argument("--raw-run",type=Path,required=True); args=p.parse_args(); art=args.analysis_run/"artifacts"; c=pq.read_table(art/"within_mode_convergence.parquet").to_pandas(); modes=pq.read_table(art/"mode_outcomes.parquet").to_pandas(); held=pq.read_table(art/"heldout_direction_prediction.parquet").to_pandas(); cross=pq.read_table(art/"mode_conditioned_crossdemo.parquet").to_pandas(); geometry=pq.read_table(art/"reference_contact_geometry.parquet").to_pandas(); raw=pq.read_table(args.raw_run/"artifacts/interventions.parquet",columns=["task","success_flip"]).to_pandas(); decision=json.loads((art/"scientific_decision.json").read_text()); summary={"decision":decision,"raw":{"interventions":int(len(raw)),"per_step_rows":pq.read_metadata(args.raw_run/"artifacts/per_step_effects.parquet").num_rows,"success_flips":int(raw.success_flip.sum()),"success_flips_by_task":raw.groupby("task").success_flip.sum().astype(int).to_dict()},"reference_geometry":{"boundaries":int(len(geometry)),"margin_class_counts":geometry.boundary_margin_class.value_counts().to_dict(),"signed_gap_by_task":{task:describe(g.signed_gap_m) for task,g in geometry.groupby("task")}},"mode_preservation_rate_by_horizon":modes.groupby("horizon").both_signs_preserved.mean().to_dict(),"transition_counts_by_horizon":modes.groupby(["horizon","transition_category"]).size().unstack(fill_value=0).to_dict("index"),"convergence":{"intent_top1_by_horizon":{h:describe(g.top1_similarity) for h,g in c.groupby("horizon")},"conditional_top1_by_horizon":{h:describe(g.top1_similarity) for h,g in c[c.conditional_preserved].groupby("horizon")},"conditional_h1_by_task":{task:{"top1":describe(g.top1_similarity),"top2":describe(g.top2_similarity),"spectral":describe(g.relative_spectral_discrepancy),"asymmetry":describe(g.sign_asymmetry),"pass_rate":float(g.passes.mean())} for task,g in c[(c.horizon=="1")&c.conditional_preserved].groupby("task")}},"heldout_conditional_h1_smallest":{"vector_error":describe(held[(held.horizon=="1")&held.both_signs_preserved&(held.radius_fraction==held.radius_fraction.min())].vector_relative_error)},"crossdemo":{"improvement":describe(cross.improvement_over_better),"by_task":cross.groupby("task").improvement_over_better.mean().to_dict()},"formal_artifact_gate":json.loads((args.analysis_run/"artifacts/output_audit.json").read_text())["gate"]}; (art/"descriptive_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n"); print(json.dumps(summary,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
