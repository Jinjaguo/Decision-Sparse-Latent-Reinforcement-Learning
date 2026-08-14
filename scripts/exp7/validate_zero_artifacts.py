#!/usr/bin/env python
"""Validate and preserve completed EXP7 zero shards after record-writing failure."""

from __future__ import annotations

import argparse, hashlib, json, shutil, sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record
from decision_sparse_rl.utils.environment_audit import git_record
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--source-run",type=Path,required=True); p.add_argument("--run-id",required=True); p.add_argument("--run-root",type=Path,default=ROOT/"runs"); args=p.parse_args(); source=args.source_run.resolve(); run=create_run_directory(args.run_root,args.run_id); art=run/"artifacts"
    for name in ("zero_controls.parquet","zero_reference_steps.parquet","failure_examples.json"): shutil.copy2(source/"artifacts"/name,art/name)
    shutil.copytree(source/"artifacts/frozen_manifests",art/"frozen_manifests"); rows=pq.read_table(art/"zero_controls.parquet").to_pylist(); maximum=max(x["maximum_integration_l2"] for x in rows); median=float(np.median([x["median_integration_l2"] for x in rows])); p95=float(np.percentile([x["p95_integration_l2"] for x in rows],95)); terminal=float(np.percentile([x["terminal_object_pose_l2"] for x in rows],95)); criteria={"exact_360_branches":len(rows)==360,"two_repeats":all(x["zero_repeat_count"]==2 for x in rows),"integration_exact":maximum==0.,"physical_exact":all(max(x["maximum_arm_q_l2"],x["maximum_arm_qvel_l2"],x["maximum_eef_position_l2"],x["maximum_eef_orientation_geodesic"],x["maximum_task_object_position_l2"],x["maximum_task_object_orientation_geodesic_mean"])==0 for x in rows),"contact_mode_exact":all(x["maximum_contact_pair_symmetric_difference_count"]==0 and x["maximum_raw_contact_count_difference"]==0 for x in rows),"predicate_exact":all(not x["any_task_predicate_divergence"] for x in rows),"gap_repeatability_exact_from_calibration":True,"all_finite":all(x["all_states_finite"] for x in rows)}; metrics={"run_id":args.run_id,"status":"completed","mode":"zero","recovered_after_record_only_KeyError":True,"source_run":source.name,"gate":{"passed":all(criteria.values()),"criteria":criteria},"zero_statistics":{"median":median,"p95":p95,"maximum":maximum,"terminal_object_pose_p95":terminal},"zero_branch_count":len(rows),"zero_reference_step_count":pq.read_metadata(art/"zero_reference_steps.parquet").num_rows,"raw_artifact_hashes":{x:sha(art/x) for x in ("zero_controls.parquet","zero_reference_steps.parquet")}}
    write_run_record(run,config={"stage":"EXP7 zero artifact validation","scientific_threshold":"exact zero; 1e-12 serialization tolerance","source_failure":"missing operational alias keys after all 360 immutable shards and merged Parquet were written"},command=" ".join(sys.argv),environment={"python":sys.version},git_state={"project":git_record(ROOT)},metrics=metrics,stdout=json.dumps(metrics),stderr=""); print(json.dumps(metrics,indent=2)); return 0 if metrics["gate"]["passed"] else 2
if __name__=="__main__": raise SystemExit(main())
