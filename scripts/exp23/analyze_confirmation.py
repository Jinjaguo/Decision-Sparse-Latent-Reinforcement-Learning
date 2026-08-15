"""Apply the frozen EXP23 selector once to the untouched confirmation outcomes."""

from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.metrics.exp18 import selector_metrics
from scripts.exp23.risk_selector import choose_knn,choose_task_prior,current_context,safe


def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--candidate-run",type=Path,required=True);p.add_argument("--reference-run",type=Path,required=True);p.add_argument("--branch-manifest",type=Path,required=True);p.add_argument("--selector",type=Path,required=True);args=p.parse_args();out=ROOT/"runs"/args.run_id
    if out.exists():raise FileExistsError(f"immutable run exists: {out}")
    artifacts=out/"artifacts";artifacts.mkdir(parents=True);rows=pq.read_table(ROOT/args.candidate_run/"artifacts/candidate_summaries.parquet").to_pylist();branches=json.loads((ROOT/args.branch_manifest).read_text());context=current_context(ROOT/args.reference_run,branches);selector_path=ROOT/args.selector;model=json.loads(selector_path.read_text());routes=model["routes"];lookup={(x["branch_id"],x["route"]):x for x in rows};selected=[];defaults=[];oracles=[];choices=[]
    for b in branches:
        values={r:lookup[(b["branch_id"],r)] for r in routes};default=values["D_physical_chunk"];oracle=max(values.values(),key=lambda x:(safe(x),-x["safety_stop"],-x["steps"]));name=model["primary_selector"]
        if name=="task_prior":route=choose_task_prior(b["task"],model["training"],routes)
        elif name.startswith("context_knn_"):route=choose_knn(context[b["branch_id"]],b["task"],model["training"],int(name.rsplit("_",1)[1]),routes)
        else:route=name
        selected.append(values[route]);defaults.append(default);oracles.append(oracle);choices.append({"branch_id":b["branch_id"],"task":b["task"],"route":route})
    primary=selector_metrics(selected,defaults,oracles);task_metrics={}
    for task in sorted(set(x["task"] for x in branches)):
        idx=[i for i,b in enumerate(branches) if b["task"]==task];task_metrics[task]=selector_metrics([selected[i] for i in idx],[defaults[i] for i in idx],[oracles[i] for i in idx])
    passed=primary["improvement_points"]>=.10 and primary["demand_recovery_rate"]>=.60 and primary["oracle_headroom_capture"]>=.75 and primary["safety_stop_rate"]<=primary["default_safety_stop_rate"] and sum(x["safe_success_rate"]>=x["default_safe_success_rate"] for x in task_metrics.values())>=2
    dump(artifacts/"preaction_choices.json",choices);result={"status":"completed","group_count":len(branches),"primary_selector":model["primary_selector"],"primary":primary,"primary_task_metrics":task_metrics,"success_rule_passed":passed,"selector_sha256":hashlib.sha256(selector_path.read_bytes()).hexdigest(),"all_states_finite":all(x["all_states_finite"] for x in rows),"target_future_access":False,"confirmation_outcomes_used_for_training_or_tuning":False};dump(out/"metrics.json",result);print(json.dumps(result,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())

