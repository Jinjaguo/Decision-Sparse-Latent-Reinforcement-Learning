"""Evaluate the calibration-frozen EXP22 consequence coordinator."""

from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.metrics.exp18 import selector_metrics

PRIMARY="F3_diverse_periodic_cycle"


def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")
def safe(x):return bool(x["success"] and not x["safety_stop"])


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--coordination-run",type=Path,required=True);p.add_argument("--oracle-run",type=Path,required=True);args=p.parse_args();out=ROOT/"runs"/args.run_id
    if out.exists():raise FileExistsError(f"immutable run exists: {out}")
    artifacts=out/"artifacts";artifacts.mkdir(parents=True);coord=pq.read_table(ROOT/args.coordination_run/"artifacts/candidate_summaries.parquet").to_pylist();base=pq.read_table(ROOT/args.oracle_run/"artifacts/candidate_summaries.parquet").to_pylist();cg=defaultdict(list);bg=defaultdict(list)
    for x in coord:cg[x["branch_id"]].append(x)
    for x in base:bg[x["branch_id"]].append(x)
    branches=sorted(cg);defaults=[];oracles=[];routes=sorted(set(x["route"] for x in coord));selected={r:[] for r in routes}
    if PRIMARY not in routes:raise ValueError(f"frozen primary missing: {PRIMARY}")
    for bid in branches:
        default=next(x for x in cg[bid] if x["route"]=="D_physical_chunk");oracle=max(bg[bid],key=lambda x:(safe(x),-x["safety_stop"],-x["steps"]));defaults.append(default);oracles.append(oracle)
        for route in routes:selected[route].append(next(x for x in cg[bid] if x["route"]==route))
    metrics={route:selector_metrics(values,defaults,oracles) for route,values in selected.items()};primary=metrics[PRIMARY];tasks=sorted(set(x["task"] for x in coord));task_metrics={}
    for task in tasks:
        idx=[i for i,x in enumerate(defaults) if x["task"]==task];task_metrics[task]=selector_metrics([selected[PRIMARY][i] for i in idx],[defaults[i] for i in idx],[oracles[i] for i in idx])
    noninferior=sum(x["safe_success_rate"]>=x["default_safe_success_rate"] for x in task_metrics.values());passed=primary["improvement_points"]>=.10 and primary["demand_recovery_rate"]>=.60 and primary["oracle_headroom_capture"]>=.75 and primary["safety_stop_rate"]<=primary["default_safety_stop_rate"] and noninferior>=2
    switches=[x.get("mode_switches",0) for x in selected[PRIMARY]]
    result={"status":"completed","group_count":len(branches),"primary":PRIMARY,"primary_metrics":primary,"primary_task_metrics":task_metrics,"route_ablations":metrics,"mean_mode_switches":sum(switches)/len(switches),"all_states_finite":all(x["all_states_finite"] for x in coord),"success_rule_passed":passed,"target_future_access":False,"oracle_source":"EXP17 evaluation headroom only"};dump(out/"metrics.json",result);print(json.dumps(result,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())

