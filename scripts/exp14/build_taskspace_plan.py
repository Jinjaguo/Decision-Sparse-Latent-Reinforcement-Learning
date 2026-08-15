"""Freeze EXP14 object-centric candidate chunks before outcomes are observed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from decision_sparse_rl.metrics.exp14 import diverse_topk, object_frame_target, ridge_inverse, unit_vector, waypoint_chunk
from scripts.exp12.prepare_ranking import TASKS, TASK_BODIES

ROUTES = ("T1_lookahead", "T2_waypoint", "T3_object_frame", "T4_retarget", "T5_inverse", "T6_skill", "T7_guided", "T8_composed")


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def body_pos(boundary: dict, name: str) -> np.ndarray:
    return np.asarray(boundary["body_positions"][boundary["body_names"].index(name)], float)


def load_cache(reference_run: Path):
    manifest=json.loads((reference_run/"artifacts/reference_snapshots_manifest.json").read_text())
    cache={}
    for record in manifest["episodes"]:
        directory=reference_run/record["relative_directory"]
        boundaries=json.loads((directory/"boundaries.json").read_text())
        with np.load(directory/"trajectory_states.npz",allow_pickle=False) as z: actions=np.asarray(z["actions"],float)
        cache[(record["task"],record["episode"])]={"boundaries":boundaries,"actions":actions,"directory":record["relative_directory"]}
    return cache


def anchor_names(task: str) -> list[str]:
    if task == TASKS[1]: return [TASK_BODIES[task][0], TASK_BODIES[task][1]]
    return [TASK_BODIES[task][0]]


def training_segments(cache, task, excluded, fraction, k=10):
    rows=[]
    for (candidate_task,episode),item in cache.items():
        if candidate_task != task or episode == excluded: continue
        actions,boundaries=item["actions"],item["boundaries"]
        i=min(len(actions)-k-1,max(0,int(round(fraction*(len(actions)-k-1)))))
        e0=np.asarray(boundaries[i]["eef_position"],float);e1=np.asarray(boundaries[i+k]["eef_position"],float)
        rows.append({"episode":episode,"index":i,"chunk":actions[i:i+k].copy(),"eef_delta":e1-e0,"eef":e0,"boundaries":boundaries})
    return rows


def learned_task_direction(cache, task, excluded) -> np.ndarray:
    deltas=[]
    for (candidate_task,episode),item in cache.items():
        if candidate_task != task or episode == excluded: continue
        b=item["boundaries"]; deltas.append(np.asarray(b[-1]["eef_position"])-np.asarray(b[max(0,len(b)-21)]["eef_position"]))
    return unit_vector(np.median(deltas,axis=0)) if deltas else np.zeros(3)


def candidate_pool(task,episode,t,item,cache,k=10):
    actions,boundaries=item["actions"],item["boundaries"];ref=actions[t:t+k].copy();boundary=boundaries[t]
    fraction=t/max(1,len(actions)-1);eef=np.asarray(boundary["eef_position"],float); candidates=[]
    # T1: structurally larger temporal moves than EXP13.
    for offset in (5,10,20):
        idx=np.clip(np.arange(t,t+k)+offset,0,len(actions)-1)
        candidates.append(("T1_lookahead",f"lookahead_{offset}",actions[idx].copy()))
    segments=training_segments(cache,task,episode,fraction,k)
    # T2: canonical object-relative EEF waypoints from other demos.
    for anchor_name in anchor_names(task):
        relative=[]
        for segment in segments:
            b=segment["boundaries"][min(len(segment["boundaries"])-1,segment["index"]+k)]
            relative.append(np.asarray(b["eef_position"],float)-body_pos(b,anchor_name))
        if relative:
            target=object_frame_target(body_pos(boundary,anchor_name),relative)
            for taper in (False,True):
                desired=waypoint_chunk(ref,target-eef,taper=taper);desired[:,3:6]=ref[:,3:6]
                candidates.append(("T2_waypoint",f"{anchor_name}_{'taper' if taper else 'constant'}",desired))
    # T3: object-frame directions and learned task completion axis.
    directions=[]
    for anchor_name in anchor_names(task): directions.append((anchor_name,unit_vector(body_pos(boundary,anchor_name)-eef)))
    directions.extend([("task_axis",learned_task_direction(cache,task,episode)),("vertical",np.asarray([0.,0.,1.]))])
    for name,direction in directions:
        for sign in (1.,-1.):
            desired=ref.copy();desired[:,:3]=np.clip(ref[:,:3]+sign*.25*direction,-1,1)
            candidates.append(("T3_object_frame",f"{name}_{int(sign):+d}",desired))
    # T4: retarget the EEF displacement of other demos, preserving their orientation/gripper semantics.
    for rank,segment in enumerate(segments[:3]):
        desired=waypoint_chunk(ref,segment["eef_delta"],taper=bool(rank%2));desired[:,3:]=segment["chunk"][:,3:]
        candidates.append(("T4_retarget",f"demo_{segment['episode']}_{segment['index']}",desired))
    # T5: regularized local inverse response from training segments.
    if len(segments)>=3:
        x=np.asarray([s["chunk"][:,:6].mean(0) for s in segments]);y=np.asarray([s["eef_delta"] for s in segments])
        desired_delta=.04*learned_task_direction(cache,task,episode)
        for l2 in (.1,1.,10.):
            command=np.clip(ridge_inverse(x,y,desired_delta,l2=l2),-1,1);desired=ref.copy();desired[:,:6]=command
            candidates.append(("T5_inverse",f"ridge_{l2:g}",desired))
    # T6: state-applicable semantic skills, offered as alternatives rather than outcome-selected actions.
    if task==TASKS[0]:
        pull=learned_task_direction(cache,task,episode)
        for gain in (.35,.60):
            desired=ref.copy();desired[:,:3]=gain*pull;desired[:,6]=1
            candidates.append(("T6_skill",f"drawer_hold_pull_{gain:.2f}",desired))
            staged=ref.copy();staged[:4,:3]=.35*unit_vector(body_pos(boundary,TASK_BODIES[task][0])-eef);staged[4:,:3]=gain*pull;staged[:,6]=1
            candidates.append(("T6_skill",f"drawer_approach_pull_{gain:.2f}",staged))
    elif task==TASKS[1]:
        bowl=body_pos(boundary,TASK_BODIES[task][0]);plate=body_pos(boundary,TASK_BODIES[task][1]);toward_bowl=unit_vector(bowl-eef);toward_plate=unit_vector(plate-bowl)
        for lift in (.20,.35):
            desired=ref.copy();desired[:5,:3]=.35*toward_bowl;desired[5:,:3]=.35*toward_plate+lift*np.asarray([0,0,1.]);desired[:,6]=1
            candidates.append(("T6_skill",f"bowl_grasp_lift_transport_{lift:.2f}",desired))
            settle=ref.copy();settle[:6,:3]=.40*toward_plate;settle[6:,:3]=np.asarray([0,0,-.25]);settle[:7,6]=1;settle[7:,6]=-1
            candidates.append(("T6_skill",f"bowl_transport_settle_{lift:.2f}",settle))
    else:
        toward=unit_vector(body_pos(boundary,TASK_BODIES[task][0])-eef)
        for gain in (.30,.50):
            desired=ref.copy();desired[:,:3]=gain*toward;desired[:,6]=1
            candidates.append(("T6_skill",f"stove_press_{gain:.2f}",desired))
    # T7: frozen geometry/effort score with diversity, no target outcome.
    if candidates:
        features=np.asarray([np.r_[c[2][:,:6].mean(0),c[2][:,:6].std(0)] for c in candidates])
        score=-np.linalg.norm(features[:,:6],axis=1)+.25*np.linalg.norm(features[:,6:],axis=1)
        for rank,index in enumerate(diverse_topk(features,score,min(3,len(candidates)),.10)):
            _,source,base=candidates[index];desired=.35*ref+.65*base;desired[:,6]=base[:,6]
            candidates.append(("T7_guided",f"geometry_rank_{rank}_{source}",desired))
    # T8: restricted two-source compositions.
    t1=next((c for c in candidates if c[0]=="T1_lookahead"),None);t6=next((c for c in candidates if c[0]=="T6_skill"),None)
    t2=next((c for c in candidates if c[0]=="T2_waypoint"),None)
    for left,right,name in ((t1,t6,"lookahead_skill"),(t2,t6,"waypoint_skill")):
        if left and right:
            desired=.5*left[2]+.5*right[2];desired[:,6]=right[2][:,6]
            candidates.append(("T8_composed",name,desired))
    return ref,candidates


def main() -> int:
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--stage",choices=("calibration","formal"),required=True);p.add_argument("--reference-run",type=Path,required=True);p.add_argument("--training-reference-run",type=Path,default=Path("runs/exp8_s2_independent_refs_20260814"));p.add_argument("--branch-manifest",type=Path,required=True);p.add_argument("--authorization",type=Path)
    args=p.parse_args();out=ROOT/"runs"/args.run_id
    if out.exists():raise FileExistsError(f"immutable run exists: {out}")
    artifacts,manifests=out/"artifacts",out/"manifests";artifacts.mkdir(parents=True);manifests.mkdir()
    reference=ROOT/args.reference_run;cache=load_cache(reference);training_cache=load_cache(ROOT/args.training_reference_run);branches=json.loads((ROOT/args.branch_manifest).read_text())
    authorization={task:list(ROUTES) for task in TASKS}
    if args.authorization:authorization=json.loads((ROOT/args.authorization).read_text())["authorized_by_task"]
    plan=[];support=defaultdict(lambda:defaultdict(int))
    for branch in branches:
        task,episode,t=branch["task"],branch["episode"],int(branch["branch_time"]);ref,candidates=candidate_pool(task,episode,t,cache[(task,episode)],training_cache)
        chosen=[];counts=defaultdict(int);seen=set();permitted=set(authorization[task]);by_route=defaultdict(list)
        for route,source,desired in candidates:
            if route not in permitted:continue
            digest=hashlib.sha256(np.round(desired,10).tobytes()).hexdigest()
            if digest in seen:continue
            seen.add(digest);by_route[route].append((route,source,desired))
        if args.stage=="formal":
            ordered=[route for route in authorization[task] if route in by_route]
            for rank in range(4):
                for route in ordered:
                    if rank < len(by_route[route]) and len(chosen) < 16:chosen.append(by_route[route][rank])
        else:
            for route in ROUTES:chosen.extend(by_route[route])
        for route,_,_ in chosen:counts[route]+=1;support[task][route]+=1
        for route,source,desired in chosen:
            plan.append({"branch_id":branch["branch_id"],"family":"I-A_analytic","generator_family":route,"basis_family":route,"candidate_source":source,"mode_index":counts[route],"channel":-1,"chunk_length":len(desired),"amplitude":1.0,"sign":1,"basis":(desired-ref).tolist()})
    for i,row in enumerate(plan):row["intervention_id"]=f"exp14_{args.stage}|r{i:05d}"
    dump(artifacts/"branch_manifest.json",branches);dump(artifacts/"candidate_plan.json",plan)
    summary={"stage":args.stage,"branch_count":len(branches),"candidate_count":len(plan),"support":{t:dict(v) for t,v in support.items()},"authorization":authorization,"proposal_training_reference_run":args.training_reference_run.name,"target_demo_outcomes_used":False,"frozen_before_outcomes":True}
    dump(artifacts/"candidate_support.json",summary);dump(out/"metrics.json",summary);print(json.dumps(summary,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
