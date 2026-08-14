#!/usr/bin/env python
"""Merge three immutable EXP7 formal shards and lock exact frozen coverage."""

from __future__ import annotations

import argparse, hashlib, json, shlex, shutil, sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record
from decision_sparse_rl.utils.environment_audit import git_record

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()
def merge(paths,output):
    writer=None
    try:
        for path in paths:
            table=pq.read_table(path); writer=writer or pq.ParquetWriter(output,table.schema,compression="zstd"); writer.write_table(table)
    finally:
        if writer: writer.close()
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--run-id",required=True); p.add_argument("--source-runs",nargs="+",required=True); p.add_argument("--zero-run",type=Path,required=True); p.add_argument("--manifest-dir",type=Path,required=True); p.add_argument("--run-root",type=Path,default=ROOT/"runs"); args=p.parse_args(); run=create_run_directory(args.run_root,args.run_id); art=run/"artifacts"; ips=[]; eps=[]; sources={}
    for name in args.source_runs:
        source=args.run_root/name; metrics=json.loads((source/"metrics.json").read_text());
        if not metrics["gate"]["passed"]: raise RuntimeError(f"failed source shard {name}")
        ips.append(source/"artifacts/interventions.parquet"); eps.append(source/"artifacts/per_step_effects.parquet"); sources[name]=sha(source/"metrics.json")
    merge(ips,art/"interventions.parquet"); merge(eps,art/"per_step_effects.parquet"); zero=args.zero_run.resolve()
    for name in ("zero_controls.parquet","zero_reference_steps.parquet"): shutil.copy2(zero/"artifacts"/name,art/name)
    shutil.copytree(args.manifest_dir.resolve(),art/"frozen_manifests"); rows=pq.read_table(art/"interventions.parquet").to_pylist(); directions=json.loads((args.manifest_dir/"direction_basis_manifest.json").read_text())["directions"]; expected={(x["task"],x["episode"],x["branch_time"],x["radius_label"],x["direction_index"],s,0) for x in directions for s in (-1,1)}; actual={(x["task"],x["episode"],x["branch_time"],x["radius_label"],x["direction_index"],x["sign"],x["repeat_index"]) for x in rows}; failures=[x for x in rows if not x["joint_limit_valid"] or not x["all_states_finite"] or x["non_arm_max_linf"]>1e-12 or x["q_injection_max_abs_error"]>1e-15]; schema=set(pq.read_schema(art/"per_step_effects.parquet").names)
    criteria={"exact_17280":len(rows)==17280,"unique_rows":len(actual)==len(rows),"exact_frozen_coverage":actual==expected,"three_shards":len(args.source_runs)==3,"no_failures":not failures,"geometry_fields":{"zero_contact_mode_json","perturbed_contact_mode_json","zero_signed_gap_m","perturbed_signed_gap_m"}.issubset(schema)}; hashes={name:sha(art/name) for name in ("zero_controls.parquet","zero_reference_steps.parquet","interventions.parquet","per_step_effects.parquet")}; write_json(art/"raw_data_lock_manifest.json",{"locked_before_analysis":True,"source_metrics_sha256":sources,"sha256":hashes}); write_json(art/"failure_examples.json",failures[:100]); metrics={"run_id":args.run_id,"status":"completed","gate":{"passed":all(criteria.values()),"criteria":criteria},"intervention_count":len(rows),"per_step_effect_count":pq.read_metadata(art/"per_step_effects.parquet").num_rows,"success_flip_count":sum(x["success_flip"] for x in rows),"maximum_non_arm_linf":max(x["non_arm_max_linf"] for x in rows),"maximum_q_injection_error":max(x["q_injection_max_abs_error"] for x in rows),"raw_hashes":hashes}; write_run_record(run,config={"stage":"EXP7 raw merge and lock","sources":args.source_runs},command=shlex.join([sys.executable,*sys.argv]),environment={"python":sys.version,"pyarrow":pa.__version__},git_state={"project":git_record(ROOT)},metrics=metrics,stdout=json.dumps(metrics),stderr=""); print(json.dumps(metrics,indent=2)); return 0 if metrics["gate"]["passed"] else 2
if __name__=="__main__": raise SystemExit(main())
