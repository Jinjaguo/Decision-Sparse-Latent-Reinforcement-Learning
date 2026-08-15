"""Evaluate the frozen EXP25 physical-progress coordinator."""

from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.metrics.exp18 import selector_metrics

PRIMARY="U0_physical_progress_primary"


def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")
def safe(x):return bool(x["success"] and not x["safety_stop"])


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--candidate-run",type=Path,required=True);args=p.parse_args();out=ROOT/"runs"/args.run_id
    if out.exists():raise FileExistsError(f"immutable run exists: {out}")
    artifacts=out/"artifacts";artifacts.mkdir(parents=True);rows=pq.read_table(ROOT/args.candidate_run/"artifacts/candidate_summaries.parquet").to_pylist();routes=sorted(set(x["route"] for x in rows));branches=sorted(set(x["branch_id"] for x in rows));lookup={(x["branch_id"],x["route"]):x for x in rows}
    expected=len(rows)//len(routes)
    if len(branches)!=expected:raise RuntimeError(f"incomplete unique denominator: {len(branches)} != {expected}")
    selected=[];defaults=[];oracles=[]
    for bid in branches:
        values=[lookup[(bid,r)] for r in routes];selected.append(lookup[(bid,PRIMARY)]);defaults.append(lookup[(bid,"D_physical_chunk")]);oracles.append(max(values,key=lambda x:(safe(x),-x["safety_stop"],-x["steps"])))
    primary=selector_metrics(selected,defaults,oracles);tasks=sorted(set(x["task"] for x in rows));task_metrics={}
    for task in tasks:
        idx=[i for i,x in enumerate(defaults) if x["task"]==task];task_metrics[task]=selector_metrics([selected[i] for i in idx],[defaults[i] for i in idx],[oracles[i] for i in idx])
    passed=primary["improvement_points"]>=.10 and primary["demand_recovery_rate"]>=.60 and primary["oracle_headroom_capture"]>=.75 and primary["safety_stop_rate"]<=primary["default_safety_stop_rate"] and sum(x["safe_success_rate"]>=x["default_safe_success_rate"] for x in task_metrics.values())>=2
    result={"status":"completed","group_count":len(branches),"complete_unique_denominator":True,"primary":PRIMARY,"primary_metrics":primary,"primary_task_metrics":task_metrics,"route_metrics":{r:selector_metrics([lookup[(bid,r)] for bid in branches],defaults,oracles) for r in routes},"success_rule_passed":passed,"all_states_finite":all(x["all_states_finite"] for x in rows),"target_future_access":False};dump(out/"metrics.json",result);print(json.dumps(result,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())

