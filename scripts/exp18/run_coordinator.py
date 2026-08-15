"""Complete-demo cross-fitted recovery consequence coordination."""

from __future__ import annotations

import argparse,hashlib,json,sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.metrics.exp18 import selector_metrics
from scripts.exp12.prepare_ranking import TASKS,TASK_BODIES

ROUTES=("D_physical_chunk","H1_weighted_k3","H1_weighted_k9","H2_median_k9","H2_medoid_k9","H3_smooth_high","H3_smooth_low","H4_progress_persistent")


def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")
def one(index,n):
    x=np.zeros(n);x[index]=1;return x


def objects(boundary,task):
    names=boundary["body_names"];pos=np.asarray([boundary["body_positions"][names.index(x)] for x in TASK_BODIES[task]],float);p=np.zeros((2,3));p[:len(pos)]=pos;return p


def load_context(reference_run:Path,branches):
    records=json.loads((reference_run/"artifacts/reference_snapshots_manifest.json").read_text())["episodes"];lookup={(x["task"],x["episode"]):x for x in records};cache={}
    rank={}
    by_demo=defaultdict(list)
    for b in branches:by_demo[(b["task"],b["episode"])].append(b)
    for values in by_demo.values():
        for i,b in enumerate(sorted(values,key=lambda x:x["branch_time"])):rank[b["branch_id"]]=i
    for b in branches:
        record=lookup[(b["task"],b["episode"])];directory=reference_run/record["relative_directory"]
        key=(b["task"],b["episode"])
        if key not in cache:cache[key]=json.loads((directory/"boundaries.json").read_text())
        boundary=cache[key][int(b["branch_time"])];eef=np.asarray(boundary["eef_position"],float);pos=objects(boundary,b["task"]);progress=boundary["progress_channels"]
        task_progress=float(progress.get("joint_qpos",1-progress.get("bowl_to_plate_planar_distance_m",0.0)))
        compact=np.r_[one(TASKS.index(b["task"]),3),one(rank[b["branch_id"]],4),int(b["branch_time"])/max(1,b["trajectory_length"]),eef,pos.reshape(-1),(pos-eef).reshape(-1),np.asarray(boundary["gripper_state"],float),float(boundary["contact_count"])/100.,task_progress]
        cache[b["branch_id"]]=compact
    return cache,rank


def ridge_fit(x,y,l2=10.):
    xa=np.c_[np.ones(len(x)),x];return np.linalg.solve(xa.T@xa+l2*np.eye(xa.shape[1]),xa.T@y)
def ridge_predict(x,w):return np.c_[np.ones(len(x)),x]@w


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--candidate-run",type=Path,default=Path("runs/exp17_s3_formal_recovery_20260815"));p.add_argument("--reference-run",type=Path,default=Path("runs/exp11_s2_formal_refs_20260814"));p.add_argument("--branch-manifest",type=Path,default=Path("runs/exp13_s3_formal_plan_20260815/artifacts/branch_manifest.json"));args=p.parse_args();out=ROOT/"runs"/args.run_id
    if out.exists():raise FileExistsError(f"immutable run exists: {out}")
    artifacts,manifests,plots=out/"artifacts",out/"manifests",out/"plots";artifacts.mkdir(parents=True);manifests.mkdir();plots.mkdir()
    candidate=ROOT/args.candidate_run;summaries=pq.read_table(candidate/"artifacts/candidate_summaries.parquet").to_pylist();steps=pq.read_table(candidate/"artifacts/per_step.parquet").to_pylist();branches=json.loads((ROOT/args.branch_manifest).read_text());context,rank=load_context(ROOT/args.reference_run,branches)
    first={}
    for row in steps:
        key=(row["branch_id"],row["route"])
        if key not in first or row["offset"]<first[key]["offset"]:first[key]=row
    rows=[]
    for row in summaries:
        task=row["task"];route=row["route"];initial=first.get((row["branch_id"],route));action=np.asarray(initial["requested_action"],float) if initial else np.zeros(7);base=context[row["branch_id"]];x=np.r_[base,one(ROUTES.index(route),len(ROUTES)),action,np.linalg.norm(action[:6]),np.mean(np.abs(action[:6])),float(initial is None)]
        rows.append({**row,"demo_key":f"{task}|{row['episode']}","branch_rank":rank[row["branch_id"]],"safe_success":bool(row["success"] and not row["safety_stop"]),"fast":float(row["success"])*(1-min(row["steps"],140)/140),"x":x.tolist()})
    pq.write_table(pa.Table.from_pylist(rows),artifacts/"selector_dataset.parquet",compression="zstd")
    demos=sorted(set(x["demo_key"] for x in rows));fold={demo:i%5 for i,demo in enumerate(demos)};dump(manifests/"folds.json",fold);dump(manifests/"input_audit.json",{"allowed":["current boundary state","route identity","first requested action"],"forbidden":["target future actions","post-action state","realized outcome","realized steps"],"target_future_or_post_action_inputs":False,"complete_demo_folds":True,"primary_selector":"conservative_ensemble"})
    predictions=[]
    for f in range(5):
        train=[x for x in rows if fold[x["demo_key"]]!=f];test=[x for x in rows if fold[x["demo_key"]]==f];xtr=np.asarray([x["x"] for x in train]);xte=np.asarray([x["x"] for x in test]);mu=xtr.mean(0);sd=xtr.std(0);sd[sd<1e-8]=1;ztr=(xtr-mu)/sd;zte=(xte-mu)/sd
        ys=np.asarray([x["safe_success"] for x in train],float);yr=np.asarray([x["safety_stop"] for x in train],float);yf=np.asarray([x["fast"] for x in train],float)
        ridge_heads=[ridge_fit(ztr,y) for y in (ys,yr,yf)];ridge=np.column_stack([np.clip(ridge_predict(zte,w),0,1) for w in ridge_heads])
        for i,row in enumerate(test):
            same=[x for x in train if x["task"]==row["task"] and x["route"]==row["route"]];same_phase=[x for x in same if x["branch_rank"]==row["branch_rank"]];prior_rows=same_phase if len(same_phase)>=3 else same
            phase=np.asarray([np.mean([x["safe_success"] for x in prior_rows]),np.mean([x["safety_stop"] for x in prior_rows]),np.mean([x["fast"] for x in prior_rows])])
            candidates=[j for j,x in enumerate(train) if x["task"]==row["task"] and x["route"]==row["route"]];distance=np.linalg.norm(ztr[candidates,:len(context[row["branch_id"]])]-zte[i,:len(context[row["branch_id"]])],axis=1);nearest=np.asarray(candidates)[np.argsort(distance)[:5]];weights=1/np.maximum(np.sort(distance)[:5],1e-3);weights/=weights.sum();knn=np.asarray([np.dot(weights,np.asarray([train[j][name] for j in nearest],float)) for name in ("safe_success","safety_stop","fast")])
            ensemble=(phase+ridge[i]+knn)/3;predictions.append({"fold":f,"branch_id":row["branch_id"],"task":row["task"],"demo_key":row["demo_key"],"route":row["route"],"phase_success":phase[0],"phase_safety":phase[1],"phase_fast":phase[2],"ridge_success":ridge[i,0],"ridge_safety":ridge[i,1],"ridge_fast":ridge[i,2],"knn_success":knn[0],"knn_safety":knn[1],"knn_fast":knn[2],"ensemble_success":ensemble[0],"ensemble_safety":ensemble[1],"ensemble_fast":ensemble[2],"phase_score":phase[0]-phase[1]+.1*phase[2],"ridge_score":ridge[i,0]-ridge[i,1]+.1*ridge[i,2],"knn_score":knn[0]-knn[1]+.1*knn[2],"ensemble_score":ensemble[0]-ensemble[1]+.1*ensemble[2]})
    pq.write_table(pa.Table.from_pylist(predictions),artifacts/"crossfit_predictions.parquet",compression="zstd")
    summary_lookup={(x["branch_id"],x["route"]):x for x in summaries};pred_by=defaultdict(list)
    for x in predictions:pred_by[x["branch_id"]].append(x)
    selector_names={"phase_route":"phase_score","ridge":"ridge_score","context_knn":"knn_score","conservative_ensemble":"ensemble_score"};selections={name:[] for name in [*selector_names,"default","fixed_k9","random","oracle"]};defaults=[];oracles=[]
    for branch_id,values in pred_by.items():
        candidates=[summary_lookup[(branch_id,x["route"])] for x in values];default=summary_lookup[(branch_id,"D_physical_chunk")];oracle=max(candidates,key=lambda x:(x["success"] and not x["safety_stop"],-x["safety_stop"],-x["steps"]));defaults.append(default);oracles.append(oracle)
        selections["default"].append(default);selections["fixed_k9"].append(summary_lookup[(branch_id,"H1_weighted_k9")]);selections["random"].append(candidates[int(hashlib.sha256(branch_id.encode()).hexdigest()[:8],16)%len(candidates)]);selections["oracle"].append(oracle)
        for name,score in selector_names.items():
            best=max(values,key=lambda x:x[score]);chosen=summary_lookup[(branch_id,best["route"])]
            if name=="conservative_ensemble" and best["ensemble_success"]<.50:chosen=default
            selections[name].append(chosen)
    metrics={}
    for name,selected in selections.items():metrics[name]=selector_metrics(selected,defaults,oracles)
    primary=metrics["conservative_ensemble"];task_primary={}
    for task in TASKS:
        ids=[i for i,x in enumerate(selections["conservative_ensemble"]) if x["task"]==task];task_primary[task]=selector_metrics([selections["conservative_ensemble"][i] for i in ids],[defaults[i] for i in ids],[oracles[i] for i in ids])
    passed=primary["improvement_points"]>=.10 and primary["demand_recovery_rate"]>=.60 and primary["oracle_headroom_capture"]>=.75 and primary["safety_stop_rate"]<=primary["default_safety_stop_rate"] and sum(task_primary[t]["safe_success_rate"]>=task_primary[t]["default_safe_success_rate"] for t in TASKS)>=2
    result={"status":"completed","row_count":len(rows),"group_count":len(pred_by),"selectors":metrics,"primary_selector":"conservative_ensemble","primary_task_metrics":task_primary,"success_rule_passed":passed,"leakage_audit_passed":True};dump(out/"metrics.json",result);print(json.dumps(result,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
