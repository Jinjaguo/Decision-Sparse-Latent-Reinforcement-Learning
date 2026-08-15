"""Run reference-free feedback candidates from corrected-D branch states."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))

from decision_sparse_rl.envs.libero_runtime import bootstrap_runtime,environment_kwargs,load_episode,load_selection,task_source_record
from decision_sparse_rl.metrics.exp15 import monotone_window,standardized_distance,weighted_chunk
from scripts.exp3.run_criticality import restore_d
from scripts.exp7.contact_geometry import load_schema
import scripts.exp11.run_replacement_stage as engine

TASKS=engine.TASKS
ROUTES=[
    {"route":"D_physical_chunk","view":"physical","k":1,"replan":10,"monotone":False,"retarget":0.0,"aggregate":"nearest"},
    {"route":"R1_object_feedback","view":"object","k":1,"replan":1,"monotone":False,"retarget":0.0,"aggregate":"nearest"},
    {"route":"R1_contact_feedback","view":"full","k":3,"replan":1,"monotone":False,"retarget":0.0,"aggregate":"weighted"},
    {"route":"R2_monotone_feedback","view":"full","k":1,"replan":1,"monotone":True,"retarget":0.0,"aggregate":"nearest"},
    {"route":"R2_monotone_chunk","view":"full","k":1,"replan":10,"monotone":True,"retarget":0.0,"aggregate":"nearest"},
    {"route":"R3_weighted_feedback","view":"full","k":5,"replan":1,"monotone":False,"retarget":0.0,"aggregate":"weighted"},
    {"route":"R4_retarget_low","view":"object","k":1,"replan":1,"monotone":True,"retarget":0.25,"aggregate":"nearest"},
    {"route":"R4_retarget_high","view":"object","k":1,"replan":1,"monotone":True,"retarget":0.50,"aggregate":"nearest"},
    {"route":"R7_conservative","view":"full","k":7,"replan":1,"monotone":False,"retarget":0.0,"aggregate":"median"},
]
EXP16_ROUTES=[
    {"route":"D_physical_chunk","view":"physical","k":1,"replan":10,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"nearest","smooth":0.0},
    {"route":"S2_task_weighted","view":"full","k":5,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"weighted","smooth":0.0},
    {"route":"S3_progress_1","view":"full","k":3,"replan":1,"monotone":True,"advance":1,"retarget":0.0,"aggregate":"weighted","smooth":0.0},
    {"route":"S3_progress_3","view":"full","k":3,"replan":1,"monotone":True,"advance":3,"retarget":0.0,"aggregate":"weighted","smooth":0.0},
    {"route":"S4_persistent_chunk","view":"full","k":1,"replan":5,"monotone":True,"advance":5,"retarget":0.0,"aggregate":"nearest","smooth":0.0},
    {"route":"S5_conservative_median","view":"full","k":7,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"median","smooth":0.0},
    {"route":"S5_medoid","view":"full","k":7,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"medoid","smooth":0.0},
    {"route":"S7_smooth_weighted","view":"full","k":5,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"weighted","smooth":0.5},
    {"route":"S8_contact_smooth","view":"full","k":3,"replan":2,"monotone":True,"advance":2,"retarget":0.0,"aggregate":"weighted","smooth":0.35},
]
EXP17_ROUTES=[
    {"route":"D_physical_chunk","view":"physical","k":1,"replan":10,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"nearest","smooth":0.0},
    {"route":"H1_weighted_k3","view":"full","k":3,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"weighted","smooth":0.0},
    {"route":"H1_weighted_k9","view":"full","k":9,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"weighted","smooth":0.0},
    {"route":"H2_median_k9","view":"full","k":9,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"median","smooth":0.0},
    {"route":"H2_medoid_k9","view":"full","k":9,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"medoid","smooth":0.0},
    {"route":"H3_smooth_low","view":"full","k":5,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"weighted","smooth":0.25},
    {"route":"H3_smooth_high","view":"full","k":5,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"weighted","smooth":0.75},
    {"route":"H4_progress_persistent","view":"full","k":3,"replan":2,"monotone":True,"advance":2,"retarget":0.0,"aggregate":"weighted","smooth":0.25},
    {"route":"H5_short_chunk","view":"full","k":1,"replan":2,"monotone":True,"advance":2,"retarget":0.0,"aggregate":"nearest","smooth":0.0},
]
VIEW={"physical":np.r_[0:3,9:15],"object":np.r_[3:15],"full":np.r_[0:26]}


def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")
def parquet(path,rows):pq.write_table(pa.Table.from_pylist(rows),path,compression="zstd")
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def padded(pos,quat):
    p=np.zeros((2,3));q=np.zeros((2,4));p[:len(pos)]=pos;q[:len(quat)]=quat
    return p,q


def contact_flags(value) -> np.ndarray:
    text=str(value).lower()
    return np.asarray([float("gripper" in text or "finger" in text),float("plate" in text or "cabinet" in text or "stove" in text),float(text not in ("","[]","()"))])


def feature(eef,pos,quat,contact):
    p,q=padded(np.asarray(pos,float),np.asarray(quat,float));relative=p-eef
    return np.r_[eef,p.reshape(-1),relative.reshape(-1),q.reshape(-1),contact_flags(contact)]


def boundary_objects(boundary,task):
    from scripts.exp12.prepare_ranking import TASK_BODIES
    names=boundary["body_names"];return np.asarray([boundary["body_positions"][names.index(x)] for x in TASK_BODIES[task]],float),np.asarray([boundary["body_quaternions"][names.index(x)] for x in TASK_BODIES[task]],float)


def build_library(training_run:Path,k=10):
    records=json.loads((training_run/"artifacts/reference_snapshots_manifest.json").read_text())["episodes"];libraries={}
    for task in TASKS:
        rows=[]
        for record in records:
            if record["task"]!=task:continue
            directory=training_run/record["relative_directory"];boundaries=json.loads((directory/"boundaries.json").read_text())
            with np.load(directory/"trajectory_states.npz",allow_pickle=False) as z:actions=np.asarray(z["actions"],float)
            for i in range(len(actions)):
                pos,quat=boundary_objects(boundaries[i],task);chunk=actions[np.clip(np.arange(i,i+k),0,len(actions)-1)]
                contact=" ".join(f"{x['geom1_name']}|{x['geom2_name']}" for x in boundaries[i]["contact_pairs"])
                rows.append({"episode":record["episode"],"index":i,"feature":feature(np.asarray(boundaries[i]["eef_position"]),pos,quat,contact),"chunk":chunk,"eef":np.asarray(boundaries[i]["eef_position"],float),"anchor":pos[0]})
        matrix=np.asarray([x["feature"] for x in rows]);libraries[task]={"rows":rows,"matrix":matrix,"scale":matrix.std(0),"episodes":np.asarray([x["episode"] for x in rows]),"indexes":np.asarray([x["index"] for x in rows])}
    return libraries


def choose_chunk(spec,state,library,memory):
    cols=VIEW[spec["view"]];matrix=library["matrix"][:,cols];query=state["feature"][cols];distance=standardized_distance(query,matrix,library["scale"][cols]);pool=np.arange(len(matrix))
    if spec["monotone"] and memory.get("episode") is not None:
        constrained=monotone_window(library["episodes"],library["indexes"],memory["episode"],memory["index"],30)
        if len(constrained):pool=constrained
    ordered=pool[np.argsort(distance[pool],kind="stable")];selected=ordered[:spec["k"]]
    first=library["rows"][int(selected[0])];memory["episode"]=first["episode"];memory["index"]=int(first["index"])+int(spec.get("advance",0))
    chunks=np.asarray([library["rows"][int(i)]["chunk"] for i in selected])
    if spec["aggregate"]=="weighted":chunk=weighted_chunk(chunks,distance[selected])
    elif spec["aggregate"]=="median":
        chunk=np.median(chunks,axis=0);chunk[:,6]=np.where(np.median(np.sign(chunks[:,:,6]),axis=0)>=0,1.,-1.)
    elif spec["aggregate"]=="medoid":
        center=np.median(chunks,axis=0);chunk=chunks[int(np.argmin(np.linalg.norm((chunks-center).reshape(len(chunks),-1),axis=1)))].copy()
    else:chunk=chunks[0].copy()
    if spec["retarget"]:
        source_relative=first["eef"]-first["anchor"];current_relative=state["eef"]-state["pos"][0];correction=(source_relative-current_relative)/.05
        chunk[:,:3]+=spec["retarget"]*correction[None,:]
    smooth=float(spec.get("smooth",0.0))
    if smooth and memory.get("previous_chunk") is not None:
        chunk[:,:6]=(1-smooth)*chunk[:,:6]+smooth*memory["previous_chunk"][:,:6]
    memory["previous_chunk"]=chunk.copy()
    requested=chunk.copy();executed=chunk.copy();executed[:,:6]=np.clip(executed[:,:6],-1,1);executed[:,6]=np.sign(executed[:,6]);return requested,executed,selected.tolist()


def runtime_state(obs):
    return {"eef":np.asarray(obs["eef_position"]),"pos":np.asarray(obs["object_positions"]),"quat":np.asarray(obs["object_quaternions"]),"feature":feature(np.asarray(obs["eef_position"]),obs["object_positions"],obs["object_quaternions"],obs["contact_mode_json"])}


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--stage",choices=("calibration","formal"),required=True);p.add_argument("--reference-run",type=Path,required=True);p.add_argument("--branch-manifest",type=Path,required=True);p.add_argument("--training-run",type=Path,default=Path("runs/exp8_s2_independent_refs_20260814"));p.add_argument("--authorization",type=Path);p.add_argument("--route-set",choices=("exp15","exp16","exp17"),default="exp15");p.add_argument("--safety-envelope",type=Path);p.add_argument("--maximum-steps",type=int,default=80)
    args=p.parse_args();out=ROOT/"runs"/args.run_id
    if out.exists():raise FileExistsError(f"immutable run exists: {out}")
    artifacts,manifests=out/"artifacts",out/"manifests";artifacts.mkdir(parents=True);manifests.mkdir();started=datetime.now(timezone.utc).isoformat();stdout=io.StringIO();stderr=io.StringIO();env=None
    try:
        training=(ROOT/args.training_run).resolve();library=build_library(training);branches=json.loads((ROOT/args.branch_manifest).read_text());routes={"exp15":ROUTES,"exp16":EXP16_ROUTES,"exp17":EXP17_ROUTES}[args.route_set];safety_envelope=json.loads((ROOT/args.safety_envelope).read_text()) if args.safety_envelope else None
        if args.authorization:
            allowed=set(json.loads((ROOT/args.authorization).read_text())["authorized_routes"]);routes=[x for x in routes if x["route"] in allowed or x["route"]=="D_physical_chunk"]
        protocol={"stage":args.stage,"route_set":args.route_set,"routes":routes,"default_route":"D_physical_chunk","training_run":training.name,"training_hash":sha(training/"artifacts/reference_snapshots_manifest.json"),"target_future_candidate_access":False,"expert_path_isolated":True,"maximum_rollout_steps":args.maximum_steps,"safety_envelope":str(args.safety_envelope) if args.safety_envelope else None,"frozen_before_outcomes":True}
        dump(manifests/"recovery_protocol.json",protocol);dump(manifests/"branch_manifest.json",branches);dump(manifests/"preoutcome_hashes.json",{"protocol":sha(manifests/"recovery_protocol.json"),"branches":sha(manifests/"branch_manifest.json")})
        selection,task_manifest=load_selection(ROOT/"experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json",ROOT/"experiments/exp1_decision_sparsity/manifests/tasks.json");selected={x["name"]:x for x in selection["tasks"]};wrapper,robosuite_root,assets_root=bootstrap_runtime(ROOT/"third_party/LIBERO",ROOT/"data",artifacts/"libero_config")
        channel_schema=json.loads((ROOT/"experiments/exp3_time_indexed_q_criticality/manifests/effect_channel_schema.json").read_text());contact_schema=load_schema(ROOT/"experiments/exp7_contact_mode_conditioned/manifests/contact_mode_schema.json")
        summaries=[];steps=[];experts=[]
        grouped=defaultdict(list)
        for b in branches:grouped[(b["task"],b["episode"])].append(b)
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            for (task,episode),demo_branches in grouped.items():
                task_def=selected[task];source=task_source_record(task_manifest,task_def["suite"],task_def["task_id"]);env=wrapper.ControlEnv(**environment_kwargs(Path(source["bddl_file_path"])))
                load_episode(env,dataset_path=ROOT/"data"/task_def["demonstration_relative_path"],episode_index=int(episode.split("_")[-1]),robosuite_package_root=robosuite_root,libero_assets_root=assets_root)
                ref_dir=(ROOT/args.reference_run)/demo_branches[0]["reference_directory"]
                with np.load(ref_dir/"trajectory_states.npz",allow_pickle=False) as z:target_actions=np.asarray(z["actions"],float);integrations=np.asarray(z["integration"],float)
                body_ids=[int(env.sim.model.body_name2id(name)) for name in channel_schema["task_object_audit"][task]["bodies"]]
                for branch in demo_branches:
                    t=int(branch["branch_time"]);controller=ref_dir/f"controller_{t:04d}.npz"
                    # Isolated evaluation-only expert path. It is never passed to choose_chunk.
                    restore_d(env,integrations[t],controller);expert_rows=engine.rollout(env,target_actions,t,None,body_ids,contact_schema,task);experts.append({"branch_id":branch["branch_id"],"task":task,"success":bool(expert_rows[-1]["predicate"]),"steps":len(expert_rows)})
                    for spec in routes:
                        restore_d(env,integrations[t],controller);obs=engine.observation(env,body_ids,contact_schema,task);memory={};pending=None;requested=None;retrieved=[];clip_count=0;action_count=0;safety=False;success=bool(obs["predicate"]);route_steps=[];exceedance_count=0;absolute_200=False
                        for offset in range(args.maximum_steps):
                            if success:break
                            if pending is None or offset%spec["replan"]==0:
                                requested,pending,retrieved=choose_chunk(spec,runtime_state(obs),library[task],memory)
                            local=offset%spec["replan"];action=pending[min(local,len(pending)-1)];req=requested[min(local,len(requested)-1)];clip=bool(np.any(np.abs(req[:6]-action[:6])>1e-12));clip_count+=clip;action_count+=1
                            env.step(action);obs=engine.observation(env,body_ids,contact_schema,task);force=float(np.linalg.norm(obs["ee_force"])) if obs["force_valid"] else float("nan");absolute_200=absolute_200 or bool(np.isfinite(force) and force>200);threshold=float(safety_envelope["tasks"][task]["primary_threshold_n"]) if safety_envelope else 200.;required=int(safety_envelope["tasks"][task]["consecutive_exceedances_to_stop"]) if safety_envelope else 1;exceedance_count=exceedance_count+1 if np.isfinite(force) and force>threshold else 0;safety=bool(exceedance_count>=required or (np.isfinite(force) and force>1000));success=bool(obs["predicate"])
                            route_steps.append({"branch_id":branch["branch_id"],"task":task,"episode":episode,"route":spec["route"],"offset":offset,"requested_action":req.tolist(),"executed_action":action.tolist(),"clipped":clip,"retrieved_indices":retrieved,"eef_position":obs["eef_position"].tolist(),"object_positions":obs["object_positions"].tolist(),"predicate":success,"contact_mode_json":obs["contact_mode_json"],"ee_force":obs["ee_force"].tolist(),"force_valid":obs["force_valid"],"safety_stop":safety})
                            if safety:break
                        steps.extend(route_steps);summaries.append({"branch_id":branch["branch_id"],"task":task,"episode":episode,"route":spec["route"],"success":success,"safety_stop":safety,"absolute_200_exceeded":absolute_200,"steps":len(route_steps),"clipped_action_fraction":clip_count/max(1,action_count),"all_states_finite":all(np.all(np.isfinite(x["eef_position"])) for x in route_steps),"terminal_contact_mode_json":obs["contact_mode_json"],"terminal_object_positions":obs["object_positions"].tolist()})
                    print(json.dumps({"branch":branch["branch_id"],"policies":len(routes),"summaries":len(summaries)},sort_keys=True))
                env.close();env=None
        parquet(artifacts/"candidate_summaries.parquet",summaries);parquet(artifacts/"per_step.parquet",steps);parquet(artifacts/"expert_upper_bound.parquet",experts)
        metrics={"status":"completed","run_id":args.run_id,"stage":args.stage,"started_utc":started,"completed_utc":datetime.now(timezone.utc).isoformat(),"branch_count":len(branches),"route_count":len(routes),"candidate_rollout_count":len(summaries),"per_step_count":len(steps),"target_future_candidate_access":False,"expert_success_rate":float(np.mean([x["success"] for x in experts]))}
        dump(out/"metrics.json",metrics);(out/"stdout.log").write_text(stdout.getvalue(),encoding="utf-8");(out/"stderr.log").write_text(stderr.getvalue(),encoding="utf-8");print(json.dumps(metrics,indent=2));return 0
    except Exception as exc:
        if env is not None:env.close()
        stderr.write(traceback.format_exc());dump(out/"metrics.json",{"status":"failed","error":repr(exc)});(out/"stderr.log").write_text(stderr.getvalue(),encoding="utf-8");raise


if __name__=="__main__":raise SystemExit(main())
