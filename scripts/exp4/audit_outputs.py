#!/usr/bin/env python
"""Independently audit EXP4 artifacts and produce report-ready detail tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import pyarrow.parquet as pq


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from scripts.exp4.analyze_criticality import group, sha256, verify_raw_hashes  # noqa: E402
from decision_sparse_rl.logging.run_directory import write_json  # noqa: E402


REQUIRED_PLOTS = [
    "heldout_criticality_vs_normalized_time_per_demo.png", "heldout_criticality_vs_physical_progress_per_demo.png",
    "time_vs_progress_crossdemo_rho.png", "direction_basis_heatmap_by_progress.png", "sign_symmetric_vs_asymmetry.png",
    "heldout_random_direction_agreement.png", "exp3_vs_exp4_concentration_curves.png", "top20_mass_heldout_per_demo.png",
    "direction_anisotropy_per_demo.png", "operator_spectrum_by_progress.png", "variance_components.png",
    "topk_overlap_matrix_time.png", "topk_overlap_matrix_progress.png", "lodo_summary.png", "leave_one_task_out_summary.png",
    "event_enrichment_adjusted.png", "terminal_outcome_relevance.png", "gpu_cpu_equivalence.png",
]


def describe(values: List[float]) -> Dict[str, float]:
    x=np.asarray(values,dtype=np.float64)
    return {"minimum":float(np.min(x)),"p25":float(np.percentile(x,25)),"median":float(np.median(x)),"p75":float(np.percentile(x,75)),"p95":float(np.percentile(x,95)),"maximum":float(np.max(x)),"mean":float(np.mean(x))}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-dir",type=Path,required=True); args=parser.parse_args(); run=args.run_dir.resolve(); artifacts=run/"artifacts"; manifest_dir=artifacts/"frozen_manifests"
    raw=verify_raw_hashes(artifacts); interventions=pq.read_table(artifacts/"interventions.parquet").to_pylist(); branches=pq.read_table(artifacts/"branch_summary.parquet").to_pylist(); directions=pq.read_table(artifacts/"direction_resolved_summary.parquet").to_pylist(); concentration=pq.read_table(artifacts/"demo_concentration.parquet").to_pylist(); robustness=pq.read_table(artifacts/"direction_robustness.parquet").to_pylist(); operators=pq.read_table(artifacts/"operator_summary.parquet").to_pylist(); replication=pq.read_table(artifacts/"replication_summary.parquet").to_pylist(); scientific=json.loads((artifacts/"scientific_decision.json").read_text()); gpu=json.loads((artifacts/"gpu_audit.json").read_text()); equivalence=json.loads((artifacts/"gpu_cpu_equivalence.json").read_text())
    keys={(x["task"],x["episode"],x["branch_time"]) for x in interventions}; coverage=group(interventions,("task","episode","branch_time")); manifests=[x for x in manifest_dir.glob("*.json") if x.name!="manifest_hashes.json"]
    criteria={"raw_hashes_match":len(raw)==3,"execution_gate_passed":json.loads((run/"metrics.json").read_text())["gate"]["passed"],"exactly_4032_interventions":len(interventions)==4032,"exactly_252_branches":len(keys)==252,"sixteen_per_branch":all(len(x)==16 for x in coverage.values()),"exactly_2016_direction_pairs":len(directions)==2016,"exactly_252_operator_rows":len(operators)==252,"all_operator_rows_valid":all(x["valid"] for x in operators),"zero_failure_examples":json.loads((artifacts/"failure_examples.json").read_text())==[],"fourteen_frozen_manifests":len(manifests)==14,"eighteen_required_plots":all((artifacts/"plots"/name).exists() and (artifacts/"plots"/name).stat().st_size>0 for name in REQUIRED_PLOTS),"gpu_equivalence_passed":equivalence["passed"],"gpu_formal_no_fallback":not gpu["formal_analysis"]["fallback_used"],"scientific_decision_present":bool(scientific["classification"])}
    audit={"schema_version":1,"passed":all(criteria.values()),"criteria":criteria,"raw_sha256":raw,"counts":{"interventions":len(interventions),"branches":len(keys),"direction_pairs":len(directions),"operators":len(operators),"manifests":len(manifests),"plots":len(REQUIRED_PLOTS)},"required_plots":REQUIRED_PLOTS}
    write_json(artifacts/"artifact_audit.json",audit)
    task_summary=[]
    for task,rows in group(concentration,("task",)).items():
        task_name=task[0]; robust=[x for x in robustness if x["task"]==task_name]; pairs=[x for x in replication if x["task"]==task_name and x["demo_a"]!="TASK_ICC"]; op=[x for x in operators if x["task"]==task_name]; ints=[x for x in interventions if x["task"]==task_name]
        task_summary.append({"task":task_name,"demo_count":len(rows),"median_top20_mass":float(np.median([x["top20_mass"] for x in rows])),"median_time_spearman":float(np.median([x["time_spearman"] for x in pairs])),"median_progress_spearman":float(np.median([x["progress_spearman"] for x in pairs])),"median_delta_spearman":float(np.median([x["delta_spearman"] for x in pairs])),"stable_demo_count":int(sum(x["stable_at_0_5"] for x in robust)),"median_direction_robustness":float(np.median([x["median_basis_direction_spearman"] for x in robust])),"median_random_agreement":float(np.median([x["heldout_random_vs_basis_spearman"] for x in robust])),"median_operator_spectral_norm":float(np.median([x["spectral_norm"] for x in op])),"median_operator_top_eigenvalue_share":float(np.median([x["top_eigenvalue_share"] for x in op])),"success_flips":int(sum(x["success_flip"] for x in ints))})
    # Component means per intervention, preserving row order and avoiding pandas.
    component_columns=["normalized_arm_q","normalized_arm_qvel","normalized_eef_position","normalized_eef_orientation","normalized_task_object_position","normalized_task_object_orientation"]
    table=pq.read_table(artifacts/"per_step_effects.parquet",columns=["intervention_id",*component_columns]); ids=np.asarray(table["intervention_id"].to_pylist(),dtype=object); starts=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1]; counts=np.diff(np.r_[starts,len(ids)]); component_summary={}
    for name in component_columns:
        values=np.asarray(table[name].combine_chunks().to_numpy(),dtype=np.float64); means=np.add.reduceat(values,starts)/counts; component_summary[name]=describe(means.tolist())
    detail={"schema_version":1,"classification":scientific["classification"],"primary_branch_s_rms":describe([x["primary_s_rms"] for x in branches]),"historical_exp3_style_branch_median":describe([x["historical_exp3_style_median"] for x in branches]),"basis_asymmetry":describe([x["asymmetry"] for x in directions if x["direction_role"]=="basis"]),"basis_direction_cv":describe([x["basis_s_cv"] for x in branches]),"operator_spectral_norm":describe([x["spectral_norm"] for x in operators]),"operator_top_eigenvalue_share":describe([x["top_eigenvalue_share"] for x in operators]),"component_intervention_means":component_summary,"task_summary":task_summary,"per_demo_concentration":concentration,"per_demo_robustness":robustness,"scientific_statistics":scientific["observed_statistics"]}
    write_json(artifacts/"analysis_detail.json",detail); print(json.dumps({"audit":audit,"task_summary":task_summary,"primary":detail["primary_branch_s_rms"],"asymmetry":detail["basis_asymmetry"],"operator":detail["operator_spectral_norm"],"components":component_summary},indent=2,sort_keys=True)); return 0 if audit["passed"] else 2


if __name__=="__main__": raise SystemExit(main())
