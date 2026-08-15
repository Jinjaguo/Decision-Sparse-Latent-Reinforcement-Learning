"""Freeze taskwise force envelopes from successful expert calibration suffixes."""

from __future__ import annotations

import argparse,contextlib,hashlib,io,json,sys,traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from decision_sparse_rl.envs.libero_runtime import bootstrap_runtime,environment_kwargs,load_episode,load_selection,task_source_record
from scripts.exp3.run_criticality import restore_d
from scripts.exp7.contact_geometry import load_schema
import scripts.exp11.run_replacement_stage as engine


def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--reference-run",type=Path,required=True);p.add_argument("--branch-manifest",type=Path,required=True);args=p.parse_args();out=ROOT/"runs"/args.run_id
    if out.exists():raise FileExistsError(f"immutable run exists: {out}")
    artifacts,manifests=out/"artifacts",out/"manifests";artifacts.mkdir(parents=True);manifests.mkdir();branches=json.loads((ROOT/args.branch_manifest).read_text());dump(manifests/"branch_manifest.json",branches);env=None;stdout=io.StringIO();stderr=io.StringIO()
    try:
        selection,task_manifest=load_selection(ROOT/"experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json",ROOT/"experiments/exp1_decision_sparsity/manifests/tasks.json");selected={x["name"]:x for x in selection["tasks"]};wrapper,robosuite_root,assets_root=bootstrap_runtime(ROOT/"third_party/LIBERO",ROOT/"data",artifacts/"libero_config");channel_schema=json.loads((ROOT/"experiments/exp3_time_indexed_q_criticality/manifests/effect_channel_schema.json").read_text());contact_schema=load_schema(ROOT/"experiments/exp7_contact_mode_conditioned/manifests/contact_mode_schema.json")
        grouped=defaultdict(list);rows=[];summaries=[]
        for b in branches:grouped[(b["task"],b["episode"])].append(b)
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            for (task,episode),demo_branches in grouped.items():
                task_def=selected[task];source=task_source_record(task_manifest,task_def["suite"],task_def["task_id"]);env=wrapper.ControlEnv(**environment_kwargs(Path(source["bddl_file_path"])))
                load_episode(env,dataset_path=ROOT/"data"/task_def["demonstration_relative_path"],episode_index=int(episode.split("_")[-1]),robosuite_package_root=robosuite_root,libero_assets_root=assets_root);ref_dir=(ROOT/args.reference_run)/demo_branches[0]["reference_directory"]
                with np.load(ref_dir/"trajectory_states.npz",allow_pickle=False) as z:actions=np.asarray(z["actions"],float);integrations=np.asarray(z["integration"],float)
                body_ids=[int(env.sim.model.body_name2id(name)) for name in channel_schema["task_object_audit"][task]["bodies"]]
                for branch in demo_branches:
                    t=int(branch["branch_time"]);restore_d(env,integrations[t],ref_dir/f"controller_{t:04d}.npz");roll=engine.rollout(env,actions,t,None,body_ids,contact_schema,task);forces=[]
                    for offset,obs in enumerate(roll):
                        force=float(np.linalg.norm(obs["ee_force"])) if obs["force_valid"] else float("nan");forces.append(force);rows.append({"branch_id":branch["branch_id"],"task":task,"episode":episode,"offset":offset,"force_norm":force,"torque_norm":float(np.linalg.norm(obs["ee_torque"])) if obs["torque_valid"] else float("nan"),"predicate":bool(obs["predicate"]),"contact_mode_json":obs["contact_mode_json"],"signed_gap_m":obs["signed_gap_m"]})
                    summaries.append({"branch_id":branch["branch_id"],"task":task,"success":bool(roll[-1]["predicate"]),"maximum_force_norm":float(np.nanmax(forces)),"p995_force_norm":float(np.nanquantile(forces,.995))})
                env.close();env=None
        if not all(x["success"] for x in summaries):raise RuntimeError("expert safety audit includes failed upper bound")
        thresholds={}
        for task in engine.TASKS:
            values=np.asarray([x["force_norm"] for x in rows if x["task"]==task and np.isfinite(x["force_norm"])])
            thresholds[task]={"p99":float(np.quantile(values,.99)),"p995":float(np.quantile(values,.995)),"maximum":float(np.max(values)),"primary_threshold_n":float(max(200.,np.quantile(values,.995)*1.10)),"consecutive_exceedances_to_stop":3}
        envelope={"source":"successful expert calibration suffixes","target_future_candidate_access":False,"primary":"max(200 N, 1.10 * taskwise expert P99.5)","absolute_emergency_ceiling_n":1000.0,"tasks":thresholds,"frozen_before_candidate_outcomes":True};dump(artifacts/"expert_force_envelope.json",envelope);pq.write_table(pa.Table.from_pylist(rows),artifacts/"expert_per_step.parquet",compression="zstd");pq.write_table(pa.Table.from_pylist(summaries),artifacts/"expert_summaries.parquet",compression="zstd");dump(out/"metrics.json",{"status":"completed","branch_count":len(branches),"per_step_count":len(rows),"thresholds":thresholds});(out/"stdout.log").write_text(stdout.getvalue(),encoding="utf-8");print(json.dumps(envelope,indent=2));return 0
    except Exception as exc:
        if env is not None:env.close()
        stderr.write(traceback.format_exc());dump(out/"metrics.json",{"status":"failed","error":repr(exc)});(out/"stderr.log").write_text(stderr.getvalue(),encoding="utf-8");raise


if __name__=="__main__":raise SystemExit(main())
