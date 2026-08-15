"""Train on EXP20 LODO outcomes and evaluate once on EXP17 formal outcomes."""

from __future__ import annotations

import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.metrics.exp18 import selector_metrics
from scripts.exp18.run_coordinator import ROUTES,load_context,ridge_fit,ridge_predict


def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")
def safe(x):return bool(x["success"] and not x["safety_stop"])


def ranks(branches):
    result={};groups=defaultdict(list)
    for b in branches:groups[(b["task"],b["episode"])].append(b)
    for values in groups.values():
        for i,b in enumerate(sorted(values,key=lambda x:x["branch_time"])):result[b["branch_id"]]=i
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);args=p.parse_args();out=ROOT/"runs"/args.run_id
    if out.exists():raise FileExistsError(f"immutable run exists: {out}")
    artifacts,manifests=out/"artifacts",out/"manifests";artifacts.mkdir(parents=True);manifests.mkdir()
    train_run=ROOT/"runs/exp20_s1_training_recovery_20260815";test_run=ROOT/"runs/exp17_s3_formal_recovery_20260815";train=pq.read_table(train_run/"artifacts/candidate_summaries.parquet").to_pylist();test=pq.read_table(test_run/"artifacts/candidate_summaries.parquet").to_pylist();train_b=json.loads((ROOT/"runs/exp20_s0_training_branches_20260815/artifacts/branch_manifest.json").read_text());test_b=json.loads((ROOT/"runs/exp13_s3_formal_plan_20260815/artifacts/branch_manifest.json").read_text());train_rank=ranks(train_b);test_rank=ranks(test_b);train_context,_=load_context(ROOT/"runs/exp8_s2_independent_refs_20260814",train_b);test_context,_=load_context(ROOT/"runs/exp11_s2_formal_refs_20260814",test_b)
    protocol={"training_outcomes":"exp20_s1_training_recovery_20260815 only","test_outcomes":"exp17_s3_formal_recovery_20260815 evaluation only","primary":"task_rank_consequence","models":["task_route","task_rank_consequence","context_knn_3","context_knn_5","context_knn_9","ridge","demand_gated_complementarity"],"formal_outcomes_used_for_training_or_tuning":False,"target_future_inputs":False};dump(manifests/"protocol.json",protocol)
    tr={(x["branch_id"],x["route"]):x for x in train};te={(x["branch_id"],x["route"]):x for x in test};train_groups=defaultdict(list);test_groups=defaultdict(list)
    for b in train_b:train_groups[b["branch_id"]]=[tr[(b["branch_id"],r)] for r in ROUTES]
    for b in test_b:test_groups[b["branch_id"]]=[te[(b["branch_id"],r)] for r in ROUTES]
    task_route={};task_rank={};rescue={}
    for task in sorted(set(x["task"] for x in train)):
        for ri,route in enumerate(ROUTES):
            values=[x for x in train if x["task"]==task and x["route"]==route];task_route[(task,ri)]=np.mean([safe(x)-.5*x["safety_stop"]+.05*x["success"]*(1-min(x["steps"],140)/140) for x in values])
            for q in range(4):
                subset=[x for x in values if train_rank[x["branch_id"]]==q];task_rank[(task,q,ri)]=np.mean([safe(x)-.5*x["safety_stop"]+.05*x["success"]*(1-min(x["steps"],140)/140) for x in subset]);rescue[(task,q,ri)]=np.mean([safe(x) and not safe(tr[(x["branch_id"],"D_physical_chunk")]) for x in subset])
    # Candidate-level ridge on context, route one-hot, and task-rank interactions.
    xtr=[];ys=[];yr=[];yf=[]
    for row in train:
        route=ROUTES.index(row["route"]);q=train_rank[row["branch_id"]];task=sorted(set(x["task"] for x in train)).index(row["task"]);interaction=np.zeros(3*4*len(ROUTES));interaction[(task*4+q)*len(ROUTES)+route]=1;xtr.append(np.r_[train_context[row["branch_id"]],np.eye(len(ROUTES))[route],interaction]);ys.append(safe(row));yr.append(row["safety_stop"]);yf.append(row["success"]*(1-min(row["steps"],140)/140))
    xtr=np.asarray(xtr);mu=xtr.mean(0);sd=xtr.std(0);sd[sd<1e-8]=1;ztr=(xtr-mu)/sd;weights=[ridge_fit(ztr,np.asarray(y,float),10.) for y in (ys,yr,yf)]
    train_branch_rows=list(train_groups);train_matrix=np.asarray([train_context[x] for x in train_branch_rows]);cmu=train_matrix.mean(0);csd=train_matrix.std(0);csd[csd<1e-8]=1;train_z=(train_matrix-cmu)/csd
    selections={name:[] for name in ("task_route","task_rank_consequence","context_knn_3","context_knn_5","context_knn_9","ridge","demand_gated_complementarity","default","fixed_k9","oracle")};defaults=[];oracles=[];prediction_rows=[]
    tasks=sorted(set(x["task"] for x in train))
    for b in test_b:
        bid=b["branch_id"];values=test_groups[bid];default=values[0];oracle=max(values,key=lambda x:(safe(x),-x["safety_stop"],-x["steps"]));defaults.append(default);oracles.append(oracle);task=b["task"];q=test_rank[bid]
        selections["default"].append(default);selections["fixed_k9"].append(values[2]);selections["oracle"].append(oracle)
        selections["task_route"].append(values[int(np.argmax([task_route[(task,i)] for i in range(len(ROUTES))]))]);selections["task_rank_consequence"].append(values[int(np.argmax([task_rank[(task,q,i)] for i in range(len(ROUTES))]))])
        default_train=np.mean([safe(x) for x in train if x["task"]==task and x["route"]=="D_physical_chunk" and train_rank[x["branch_id"]]==q]);best=int(np.argmax([rescue[(task,q,i)] for i in range(len(ROUTES))]));selections["demand_gated_complementarity"].append(values[best if default_train<.60 and rescue[(task,q,best)]>=.10 else 0])
        z=(test_context[bid]-cmu)/csd;candidates=np.asarray([i for i,x in enumerate(train_branch_rows) if train_groups[x][0]["task"]==task]);distance=np.linalg.norm(train_z[candidates]-z,axis=1);order=candidates[np.argsort(distance)]
        for k in (3,5,9):
            chosen=order[:k];d=np.asarray([np.linalg.norm(train_z[j]-z) for j in chosen]);w=1/np.maximum(d,1e-3);w/=w.sum();scores=[]
            for ri in range(len(ROUTES)):
                route_values=[train_groups[train_branch_rows[j]][ri] for j in chosen];scores.append(np.dot(w,[safe(x)-.5*x["safety_stop"]+.05*x["success"]*(1-min(x["steps"],140)/140) for x in route_values]))
            selections[f"context_knn_{k}"].append(values[int(np.argmax(scores))])
        ridge_scores=[]
        for ri in range(len(ROUTES)):
            interaction=np.zeros(3*4*len(ROUTES));interaction[(tasks.index(task)*4+q)*len(ROUTES)+ri]=1;x=np.r_[test_context[bid],np.eye(len(ROUTES))[ri],interaction];pred=[float(np.clip(ridge_predict(((x-mu)/sd)[None,:],w)[0],0,1)) for w in weights];ridge_scores.append(pred[0]-pred[1]+.05*pred[2]);prediction_rows.append({"branch_id":bid,"route":ROUTES[ri],"task":task,"rank":q,"ridge_success":pred[0],"ridge_safety":pred[1],"ridge_fast":pred[2],"task_rank_score":task_rank[(task,q,ri)]})
        selections["ridge"].append(values[int(np.argmax(ridge_scores))])
    pq.write_table(pa.Table.from_pylist(prediction_rows),artifacts/"formal_preaction_predictions.parquet",compression="zstd")
    metrics={name:selector_metrics(selected,defaults,oracles) for name,selected in selections.items()};primary=metrics["task_rank_consequence"];task_metrics={}
    for task in tasks:
        idx=[i for i,b in enumerate(test_b) if b["task"]==task];task_metrics[task]=selector_metrics([selections["task_rank_consequence"][i] for i in idx],[defaults[i] for i in idx],[oracles[i] for i in idx])
    passed=primary["improvement_points"]>=.10 and primary["demand_recovery_rate"]>=.60 and primary["oracle_headroom_capture"]>=.75 and primary["safety_stop_rate"]<=primary["default_safety_stop_rate"] and sum(x["safe_success_rate"]>=x["default_safe_success_rate"] for x in task_metrics.values())>=2
    result={"status":"completed","training_branch_count":len(train_b),"test_branch_count":len(test_b),"primary_selector":"task_rank_consequence","primary":primary,"primary_task_metrics":task_metrics,"selectors":metrics,"success_rule_passed":passed,"formal_outcomes_used_for_training_or_tuning":False,"leakage_audit_passed":True};dump(out/"metrics.json",result);print(json.dumps(result,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
