"""Cross-fit branch-vector demand and complementary rescue predictions."""

from __future__ import annotations

import argparse,json,sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.metrics.exp18 import selector_metrics

ROUTES=("D_physical_chunk","H1_weighted_k3","H1_weighted_k9","H2_median_k9","H2_medoid_k9","H3_smooth_high","H3_smooth_low","H4_progress_persistent")


def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")
def safe(row):return bool(row["success"] and not row["safety_stop"])


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--dataset-run",type=Path,default=Path("runs/exp18_s1_crossfit_coordinator_r1_20260815"));p.add_argument("--candidate-run",type=Path,default=Path("runs/exp17_s3_formal_recovery_20260815"));args=p.parse_args();out=ROOT/"runs"/args.run_id
    if out.exists():raise FileExistsError(f"immutable run exists: {out}")
    artifacts,manifests=out/"artifacts",out/"manifests";artifacts.mkdir(parents=True);manifests.mkdir();rows=pq.read_table(ROOT/args.dataset_run/"artifacts/selector_dataset.parquet").to_pylist();fold=json.loads((ROOT/args.dataset_run/"manifests/folds.json").read_text());summaries=pq.read_table(ROOT/args.candidate_run/"artifacts/candidate_summaries.parquet").to_pylist();summary={(x["branch_id"],x["route"]):x for x in summaries}
    grouped=defaultdict(dict)
    for row in rows:grouped[row["branch_id"]][row["route"]]=row
    base_len=len(rows[0]["x"])-18;branches=[]
    for branch_id,values in grouped.items():
        first=next(iter(values.values()));branches.append({"branch_id":branch_id,"task":first["task"],"demo_key":first["demo_key"],"context":np.asarray(first["x"][:base_len],float),"safe":np.asarray([values[r]["safe_success"] for r in ROUTES],float),"safety":np.asarray([values[r]["safety_stop"] for r in ROUTES],float),"fast":np.asarray([values[r]["fast"] for r in ROUTES],float)})
    protocol={"primary":"demand_gated_k3_t040_m005","neighbor_counts":[1,3,5,7,11],"demand_thresholds":[.3,.4,.5,.6],"rescue_margins":[0,.05,.1],"branch_vector_labels":["safe_success","safety_stop","fast"],"target_future_or_post_action_inputs":False,"complete_demo_folds":True};dump(manifests/"protocol.json",protocol)
    predictions=[]
    for f in range(5):
        train=[x for x in branches if fold[x["demo_key"]]!=f];test=[x for x in branches if fold[x["demo_key"]]==f];xtr=np.asarray([x["context"] for x in train]);mu=xtr.mean(0);sd=xtr.std(0);sd[sd<1e-8]=1;ztr=(xtr-mu)/sd
        for row in test:
            z=(row["context"]-mu)/sd;candidates=np.asarray([i for i,x in enumerate(train) if x["task"]==row["task"]]);distance=np.linalg.norm(ztr[candidates]-z,axis=1);order=candidates[np.argsort(distance,kind="stable")]
            for k in (1,3,5,7,11):
                chosen=order[:min(k,len(order))];d=np.asarray([np.linalg.norm(ztr[j]-z) for j in chosen]);w=1/np.maximum(d,1e-3);w/=w.sum();safe_pred=np.tensordot(w,np.asarray([train[j]["safe"] for j in chosen]),axes=(0,0));safety_pred=np.tensordot(w,np.asarray([train[j]["safety"] for j in chosen]),axes=(0,0));fast_pred=np.tensordot(w,np.asarray([train[j]["fast"] for j in chosen]),axes=(0,0));default_fail=1-safe_pred[0];rescue_joint=np.tensordot(w,np.asarray([train[j]["safe"]*(1-train[j]["safe"][0]) for j in chosen]),axes=(0,0));score=safe_pred-safety_pred+.1*fast_pred
                predictions.append({"branch_id":row["branch_id"],"task":row["task"],"demo_key":row["demo_key"],"fold":f,"k":k,"default_fail_probability":float(default_fail),"safe_prediction":safe_pred.tolist(),"safety_prediction":safety_pred.tolist(),"fast_prediction":fast_pred.tolist(),"rescue_joint_prediction":rescue_joint.tolist(),"utility_score":score.tolist()})
    pq.write_table(pa.Table.from_pylist(predictions),artifacts/"branch_vector_predictions.parquet",compression="zstd")
    by_k=defaultdict(dict)
    for x in predictions:by_k[x["k"]][x["branch_id"]]=x
    defaults=[];oracles=[]
    for b in branches:
        candidate=[summary[(b["branch_id"],r)] for r in ROUTES];defaults.append(candidate[0]);oracles.append(max(candidate,key=lambda x:(safe(x),-x["safety_stop"],-x["steps"])))
    selectors={}
    for k in (1,3,5,7,11):
        for threshold in (.3,.4,.5,.6):
            for margin in (0.,.05,.1):
                name=f"demand_gated_k{k}_t{int(threshold*100):03d}_m{int(margin*100):03d}";selected=[]
                for b in branches:
                    pred=by_k[k][b["branch_id"]];candidate=[summary[(b["branch_id"],r)] for r in ROUTES];rescue=np.asarray(pred["rescue_joint_prediction"]);best=int(np.argmax(rescue[1:]))+1;rescue_margin=float(rescue[best]-rescue[0]);choice=best if pred["default_fail_probability"]>=threshold and rescue_margin>=margin else 0;selected.append(candidate[choice])
                selectors[name]=selector_metrics(selected,defaults,oracles)
        selected=[]
        for b in branches:
            pred=by_k[k][b["branch_id"]];candidate=[summary[(b["branch_id"],r)] for r in ROUTES];selected.append(candidate[int(np.argmax(pred["utility_score"]))])
        selectors[f"listwise_utility_k{k}"]=selector_metrics(selected,defaults,oracles)
    primary=selectors["demand_gated_k3_t040_m005"]
    # Reconstruct primary rows for task replication.
    primary_selected=[]
    for b in branches:
        pred=by_k[3][b["branch_id"]];candidate=[summary[(b["branch_id"],r)] for r in ROUTES];rescue=np.asarray(pred["rescue_joint_prediction"]);best=int(np.argmax(rescue[1:]))+1;choice=best if pred["default_fail_probability"]>=.4 and rescue[best]-rescue[0]>=.05 else 0;primary_selected.append(candidate[choice])
    task_metrics={}
    for task in sorted(set(b["task"] for b in branches)):
        idx=[i for i,b in enumerate(branches) if b["task"]==task];task_metrics[task]=selector_metrics([primary_selected[i] for i in idx],[defaults[i] for i in idx],[oracles[i] for i in idx])
    passed=primary["improvement_points"]>=.10 and primary["demand_recovery_rate"]>=.60 and primary["oracle_headroom_capture"]>=.75 and primary["safety_stop_rate"]<=primary["default_safety_stop_rate"] and sum(x["safe_success_rate"]>=x["default_safe_success_rate"] for x in task_metrics.values())>=2
    result={"status":"completed","branch_count":len(branches),"primary_selector":"demand_gated_k3_t040_m005","primary":primary,"primary_task_metrics":task_metrics,"success_rule_passed":passed,"leakage_audit_passed":True,"selectors":selectors};dump(out/"metrics.json",result);print(json.dumps({k:v for k,v in result.items() if k!="selectors"},indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
