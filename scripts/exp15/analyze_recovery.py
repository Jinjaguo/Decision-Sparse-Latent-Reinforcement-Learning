"""Analyze safe availability, demand, and calibration authorization for EXP15."""

from __future__ import annotations

import argparse,json,sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.metrics.exp15 import recovery_metrics


def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--candidate-run",type=Path,required=True);p.add_argument("--stage",choices=("calibration","formal"),required=True);args=p.parse_args();out=ROOT/"runs"/args.run_id
    if out.exists():raise FileExistsError(f"immutable run exists: {out}")
    artifacts,plots=out/"artifacts",out/"plots";artifacts.mkdir(parents=True);plots.mkdir();rows=pq.read_table(ROOT/args.candidate_run/"artifacts/candidate_summaries.parquet").to_pylist();overall=recovery_metrics(rows,"D_physical_chunk")
    tasks={task:recovery_metrics([x for x in rows if x["task"]==task],"D_physical_chunk") for task in sorted(set(x["task"] for x in rows))}
    route_metrics=[];authorized=[]
    for route in sorted(set(x["route"] for x in rows)):
        values=[x for x in rows if x["route"]==route];success=float(np.mean([x["success"] and not x["safety_stop"] for x in values]));clip=float(np.mean([x["clipped_action_fraction"] for x in values]));safety=float(np.mean([x["safety_stop"] for x in values]));finite=all(x["all_states_finite"] for x in values)
        route_metrics.append({"route":route,"count":len(values),"safe_success_rate":success,"mean_clipped_action_fraction":clip,"safety_stop_rate":safety,"all_states_finite":finite})
    eligible=[x for x in route_metrics if x["route"]!="D_physical_chunk" and x["safe_success_rate"]>=.25 and x["mean_clipped_action_fraction"]<=.10 and x["safety_stop_rate"]<=.10 and x["all_states_finite"]]
    authorized=["D_physical_chunk"]+[x["route"] for x in sorted(eligible,key=lambda x:x["safe_success_rate"],reverse=True)[:7]]
    authorization={"stage":args.stage,"authorized_routes":authorized,"default_always_retained":True,"calibration_only":args.stage=="calibration"};dump(artifacts/"route_authorization.json",authorization);pq.write_table(pa.Table.from_pylist(route_metrics),artifacts/"route_metrics.parquet")
    valid=float(np.mean([x["all_states_finite"] and x["clipped_action_fraction"]<=.10 for x in rows]));success_rule=overall["safe_candidate_availability"]>=.70 and sum(x["safe_candidate_availability"]>=.60 for x in tasks.values())>=2 and overall["decision_demand_rate"]>=.30 and overall["demand_recovery_rate"]>=.60 and valid>=.90
    metrics={"status":"completed","stage":args.stage,**overall,"task_metrics":tasks,"valid_candidate_fraction":valid,"route_metrics":route_metrics,"authorized_routes":authorized,"success_rule_passed":success_rule};dump(out/"metrics.json",metrics);print(json.dumps(metrics,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
