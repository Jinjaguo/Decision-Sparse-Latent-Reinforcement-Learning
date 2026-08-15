"""Final reproducibility, leakage, denominator, pooled, and bootstrap audit for EXP27."""

from __future__ import annotations
import argparse,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.metrics.exp18 import selector_metrics

PRIMARY="V0_default70_soft_goal"


def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def safe(x):return bool(x["success"] and not x["safety_stop"])


def selections(rows):
    routes=sorted(set(x["route"] for x in rows));groups=defaultdict(list)
    for x in rows:groups[x["branch_id"]].append(x)
    selected=[];defaults=[];oracles=[]
    for bid in sorted(groups):
        values=groups[bid];selected.append(next(x for x in values if x["route"]==PRIMARY));defaults.append(next(x for x in values if x["route"]=="D_physical_chunk"));oracles.append(max(values,key=lambda x:(safe(x),-x["safety_stop"],-x["steps"])))
    return selected,defaults,oracles,routes


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--calibration-run",type=Path,required=True);p.add_argument("--formal-run",type=Path,required=True);p.add_argument("--analysis-run",type=Path,required=True);p.add_argument("--branch-run",type=Path,required=True);p.add_argument("--config",type=Path,required=True);args=p.parse_args();out=ROOT/"runs"/args.run_id
    if out.exists():raise FileExistsError(f"immutable run exists: {out}")
    artifacts=out/"artifacts";artifacts.mkdir(parents=True);cal=ROOT/args.calibration_run;formal=ROOT/args.formal_run;analysis=ROOT/args.analysis_run;branch_run=ROOT/args.branch_run;config=ROOT/args.config
    cal_rows=pq.read_table(cal/"artifacts/candidate_summaries.parquet").to_pylist();formal_rows=pq.read_table(formal/"artifacts/candidate_summaries.parquet").to_pylist();formal_steps=pq.read_table(formal/"artifacts/per_step.parquet").to_pylist();selected,defaults,oracles,routes=selections(formal_rows);pooled=selections(cal_rows+formal_rows);formal_metric=selector_metrics(selected,defaults,oracles);pooled_metric=selector_metrics(*pooled[:3]);branches=json.loads((branch_run/"artifacts/branch_manifest.json").read_text());protocol=json.loads((formal/"manifests/recovery_protocol.json").read_text());pre=json.loads((formal/"manifests/preoutcome_hashes.json").read_text());analysis_metrics=json.loads((analysis/"metrics.json").read_text());cfg=json.loads(config.read_text())
    by_demo=defaultdict(list)
    for i,x in enumerate(defaults):by_demo[(x["task"],x["episode"])].append(i)
    demo_keys=sorted(by_demo);rng=np.random.default_rng(270027);boot=[]
    for _ in range(5000):
        sampled=rng.choice(len(demo_keys),len(demo_keys),replace=True);idx=[i for j in sampled for i in by_demo[demo_keys[int(j)]]];boot.append(selector_metrics([selected[i] for i in idx],[defaults[i] for i in idx],[oracles[i] for i in idx]))
    ci={name:[float(np.quantile([x[name] for x in boot],.025)),float(np.quantile([x[name] for x in boot],.975))] for name in ("improvement_points","demand_recovery_rate","oracle_headroom_capture","safety_stop_rate")}
    checks={
        "formal_success_rule":analysis_metrics["success_rule_passed"] is True,
        "formal_60_unique_groups":len(branches)==60 and len({x["branch_id"] for x in branches})==60,
        "formal_complete_route_matrix":len(formal_rows)==60*len(routes) and len(routes)==7,
        "formal_steps_present_and_finite":len(formal_steps)>0 and all(np.all(np.isfinite(x["eef_position"])) for x in formal_steps),
        "target_future_access_false":protocol["target_future_candidate_access"] is False and analysis_metrics["target_future_access"] is False,
        "expert_path_isolated":protocol["expert_path_isolated"] is True,
        "target_demo_excluded":protocol["exclude_target_demo_from_neighbors_and_scale"] is True,
        "protocol_hash_matches":pre["protocol"]==sha(formal/"manifests/recovery_protocol.json"),
        "branch_hash_matches":pre["branches"]==sha(formal/"manifests/branch_manifest.json"),
        "frozen_primary_matches":cfg["frozen_primary"]==PRIMARY and analysis_metrics["primary"]==PRIMARY,
        "formal_metric_recomputed":all(abs(formal_metric[k]-analysis_metrics["primary_metrics"][k])<1e-12 for k in formal_metric),
        "pooled_gate":pooled_metric["improvement_points"]>=.10 and pooled_metric["demand_recovery_rate"]>=.60 and pooled_metric["oracle_headroom_capture"]>=.75 and pooled_metric["safety_stop_rate"]<=pooled_metric["default_safety_stop_rate"],
        "bootstrap_improvement_lower_positive":ci["improvement_points"][0]>0,
    }
    result={"status":"completed","passed":all(checks.values()),"checks":checks,"formal_metrics":formal_metric,"pooled_calibration_formal_metrics":pooled_metric,"formal_demo_cluster_bootstrap_95ci":ci,"bootstrap_replicates":5000,"formal_demo_count":len(demo_keys),"source_hashes":{"config":sha(config),"formal_protocol":sha(formal/"manifests/recovery_protocol.json"),"formal_branches":sha(formal/"manifests/branch_manifest.json"),"formal_summaries":sha(formal/"artifacts/candidate_summaries.parquet"),"formal_steps":sha(formal/"artifacts/per_step.parquet"),"analysis":sha(analysis/"metrics.json")}};dump(out/"metrics.json",result);print(json.dumps(result,indent=2));return 0 if result["passed"] else 2


if __name__=="__main__":raise SystemExit(main())

