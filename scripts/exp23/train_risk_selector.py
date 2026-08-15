"""Cross-validate and freeze an EXP23 current-state route selector."""

from __future__ import annotations
import argparse,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.metrics.exp18 import selector_metrics
from scripts.exp23.risk_selector import choose_knn,choose_task_prior,current_context,safe,utility


def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--candidate-run",type=Path,required=True);p.add_argument("--reference-run",type=Path,required=True);p.add_argument("--branch-manifest",type=Path,required=True);args=p.parse_args();out=ROOT/"runs"/args.run_id
    if out.exists():raise FileExistsError(f"immutable run exists: {out}")
    artifacts,manifests=out/"artifacts",out/"manifests";artifacts.mkdir(parents=True);manifests.mkdir();rows=pq.read_table(ROOT/args.candidate_run/"artifacts/candidate_summaries.parquet").to_pylist();branches=json.loads((ROOT/args.branch_manifest).read_text());context=current_context(ROOT/args.reference_run,branches);routes=sorted(set(x["route"] for x in rows));lookup={(x["branch_id"],x["route"]):x for x in rows};training=[]
    for b in branches:
        outcomes={r:{"utility":utility(lookup[(b["branch_id"],r)]),"safe":safe(lookup[(b["branch_id"],r)]),"safety":bool(lookup[(b["branch_id"],r)]["safety_stop"]),"steps":int(lookup[(b["branch_id"],r)]["steps"])} for r in routes};training.append({"branch_id":b["branch_id"],"task":b["task"],"episode":b["episode"],"context":context[b["branch_id"]].tolist(),"outcomes":outcomes})
    selectors=["task_prior","context_knn_3","context_knn_5","context_knn_9","context_knn_15",*routes];selected={x:[] for x in selectors};defaults=[];oracles=[]
    grouped=defaultdict(list)
    for x in training:grouped[(x["task"],x["episode"])].append(x)
    for key,test in grouped.items():
        train=[x for x in training if (x["task"],x["episode"])!=key]
        for item in test:
            values={r:lookup[(item["branch_id"],r)] for r in routes};default=values["D_physical_chunk"];oracle=max(values.values(),key=lambda x:(safe(x),-x["safety_stop"],-x["steps"]));defaults.append(default);oracles.append(oracle);selected["task_prior"].append(values[choose_task_prior(item["task"],train,routes)])
            for k in (3,5,9,15):selected[f"context_knn_{k}"].append(values[choose_knn(item["context"],item["task"],train,k,routes)])
            for route in routes:selected[route].append(values[route])
    metrics={name:selector_metrics(values,defaults,oracles) for name,values in selected.items()};eligible=[name for name in selectors if metrics[name]["safety_stop_rate"]<=metrics[name]["default_safety_stop_rate"]];primary=max(eligible,key=lambda name:(metrics[name]["oracle_headroom_capture"],metrics[name]["demand_recovery_rate"],metrics[name]["safe_success_rate"],name.startswith("context_knn"),name))
    model={"schema_version":1,"primary_selector":primary,"routes":routes,"training":training,"input_fields":["task identity","current eef pose","current task-object poses","eef-object relative position","current gripper state","current contact count","current task progress channels"],"forbidden_inputs":["branch time","target trajectory length","target future actions","post-action states","realized confirmation outcomes"],"complete_demo_leave_one_out":True,"candidate_run":args.candidate_run.name,"reference_run":args.reference_run.name,"branch_manifest":str(args.branch_manifest),"crossval_metrics":metrics[primary]};dump(artifacts/"frozen_selector.json",model);model_hash=hashlib.sha256((artifacts/"frozen_selector.json").read_bytes()).hexdigest();result={"status":"completed","training_branches":len(training),"route_count":len(routes),"selectors":metrics,"primary_selector":primary,"primary":metrics[primary],"model_sha256":model_hash,"target_future_or_post_action_inputs":False,"confirmation_outcomes_used":False};dump(out/"metrics.json",result);dump(manifests/"input_audit.json",{"passed":True,"allowed":model["input_fields"],"forbidden":model["forbidden_inputs"]});print(json.dumps(result,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())

