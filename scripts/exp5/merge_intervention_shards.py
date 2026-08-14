#!/usr/bin/env python
"""Merge immutable EXP5 q shards and enforce exact frozen coverage."""
from __future__ import annotations
import argparse, hashlib, json, shlex, shutil, sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record
from decision_sparse_rl.utils.environment_audit import git_record
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def merge(paths:list[Path],out:Path)->None:
 writer=None
 try:
  for p in paths:
   t=pq.read_table(p)
   if writer is None: writer=pq.ParquetWriter(out,t.schema,compression='zstd')
   writer.write_table(t)
 finally:
  if writer: writer.close()
def main()->int:
 p=argparse.ArgumentParser(); p.add_argument('--run-id',required=True); p.add_argument('--source-runs',nargs='+',required=True); p.add_argument('--zero-run',type=Path,required=True); p.add_argument('--run-root',type=Path,default=ROOT/'runs'); p.add_argument('--manifest-dir',type=Path,default=ROOT/'experiments/exp5_state_conditioned_anisotropic/manifests'); a=p.parse_args(); run=create_run_directory(a.run_root,a.run_id); ips=[]; eps=[]; source={}
 for n in a.source_runs:
  d=(a.run_root/n).resolve(); m=json.loads((d/'metrics.json').read_text())
  if not m['gate']['passed']: raise RuntimeError(f'failed source shard {n}')
  ips.append(d/'artifacts/interventions.parquet'); eps.append(d/'artifacts/per_step_effects.parquet'); source[n]=sha(d/'metrics.json')
 art=run/'artifacts'; merge(ips,art/'interventions.parquet'); merge(eps,art/'per_step_effects.parquet'); zero=a.zero_run.resolve(); shutil.copy2(zero/'artifacts/zero_controls.parquet',art/'zero_controls.parquet'); shutil.copy2(zero/'artifacts/zero_reference_steps.parquet',art/'zero_reference_steps.parquet'); shutil.copytree(a.manifest_dir.resolve(),art/'frozen_manifests')
 rows=pq.read_table(art/'interventions.parquet').to_pylist(); manifest=json.loads((a.manifest_dir/'direction_basis_manifest.json').read_text()); expected={(x['task'],x['episode'],x['branch_time'],x['radius_label'],x['direction_index'],s) for x in manifest['directions'] for s in [-1,1]}; actual={(x['task'],x['episode'],x['branch_time'],x['radius_label'],x['direction_index'],x['sign']) for x in rows}; failures=[x for x in rows if not x['joint_limit_valid'] or not x['all_states_finite'] or x['non_arm_max_linf']>1e-12]
 criteria={'exact_16896':len(rows)==16896,'unique_rows':len(actual)==len(rows),'exact_frozen_coverage':actual==expected,'no_execution_failures':not failures,'all_source_shards':len(a.source_runs)==15}; hashes={n:sha(art/n) for n in ['zero_controls.parquet','zero_reference_steps.parquet','interventions.parquet','per_step_effects.parquet']}; write_json(art/'raw_hash_manifest.json',{'schema_version':1,'locked_before_analysis':True,'source_metrics_sha256':source,'sha256':hashes}); write_json(art/'failure_examples.json',failures[:100]); metrics={'run_id':a.run_id,'status':'completed','gate':{'passed':all(criteria.values()),'criteria':criteria},'intervention_count':len(rows),'per_step_effect_count':pq.read_metadata(art/'per_step_effects.parquet').num_rows,'success_flip_count':sum(x['success_flip'] for x in rows),'maximum_non_arm_linf':max(x['non_arm_max_linf'] for x in rows),'by_radius':{r:sum(x['radius_label']==r for x in rows) for r in ['small','main','large']},'raw_hashes':hashes}
 write_run_record(run,config={'stage':'EXP5-12/13 raw merge and lock','source_runs':a.source_runs},command=shlex.join([sys.executable,*sys.argv]),environment={'python':sys.version,'pyarrow':pa.__version__},git_state={'project':git_record(ROOT)},metrics=metrics,stdout=json.dumps(metrics),stderr=''); print(json.dumps(metrics,indent=2)); return 0 if metrics['gate']['passed'] else 2
if __name__=='__main__': raise SystemExit(main())
