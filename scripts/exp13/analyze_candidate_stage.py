"""Analyze EXP13 proposal fidelity, diversity, and oracle opportunity by source."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from decision_sparse_rl.metrics.exp13 import authorize_family
from scripts.exp12.prepare_ranking import TASKS, TASK_BODIES, motion_quality, object_arrays


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_pq(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def load_cache(reference_run: Path):
    records = json.loads((reference_run / "artifacts/reference_snapshots_manifest.json").read_text())["episodes"]
    cache = {}
    for record in records:
        directory = reference_run / record["relative_directory"]
        boundaries = json.loads((directory / "boundaries.json").read_text())
        with np.load(directory / "trajectory_states.npz", allow_pickle=False) as z:
            terminal_pos = np.asarray(z["terminal_body_positions"], float); terminal_quat = np.asarray(z["terminal_body_quaternions"], float)
        names = list(boundaries[0]["body_names"]); task = record["task"]
        reference_pos = np.asarray([terminal_pos[names.index(x)] for x in TASK_BODIES[task]])
        reference_quat = np.asarray([terminal_quat[names.index(x)] for x in TASK_BODIES[task]])
        cache[(task, record["episode"])] = (boundaries, reference_pos, reference_quat)
    return cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--plan-run", type=Path)
    parser.add_argument("--stage", choices=("calibration", "formal"), required=True)
    args = parser.parse_args()
    out = ROOT / "runs" / args.run_id
    if out.exists(): raise FileExistsError(f"immutable run exists: {out}")
    artifacts, plots = out/"artifacts", out/"plots"; artifacts.mkdir(parents=True); plots.mkdir()
    source = ROOT / args.candidate_run / "artifacts"
    prefix = "calibration_" if args.stage == "calibration" else ""
    summaries = pq.read_table(source / f"{prefix}replacements.parquet").to_pylist()
    step_path = source / ("calibration_per_step.parquet" if args.stage == "calibration" else "per_step_response.parquet")
    steps = pq.read_table(step_path).to_pylist()
    by_step=defaultdict(list)
    for row in steps: by_step[row["intervention_id"]].append(row)
    cache=load_cache(ROOT/args.reference_run); rows=[]; nominal_quality={}
    for summary in summaries:
        task,episode=summary["task"],summary["episode"];boundaries,reference_pos,reference_quat=cache[(task,episode)]
        initial_pos,initial_quat=object_arrays(boundaries[0],task)
        if (task,episode) not in nominal_quality:
            nm=motion_quality(task,initial_pos,initial_quat,reference_pos,reference_quat,reference_pos,reference_quat)
            nominal_quality[(task,episode)]=4+1+float(np.clip(nm,-1,1.5))
        terminal=max(by_step[summary["intervention_id"]],key=lambda x:x["continuation_offset"])
        pos=np.asarray(terminal["task_object_positions"],float);quat=np.asarray(terminal["task_object_quaternions"],float)
        motion=motion_quality(task,initial_pos,initial_quat,pos,quat,reference_pos,reference_quat)
        contact=1-float(summary["regime_change_fraction_h20"]);outcome=float(summary["terminal_perturbed_success"]);composite=4*outcome+contact+float(np.clip(motion,-1,1.5))
        fidelity_valid = bool(summary["all_states_finite"]) and not bool(summary["clipped_chunk"])
        rows.append({"candidate_id":summary["intervention_id"],"branch_id":summary["branch_id"],"task":task,"episode":episode,"generator_family":summary["generator_family"],"candidate_source":summary["candidate_source"],"clipped_chunk":bool(summary["clipped_chunk"]),"all_states_finite":bool(summary["all_states_finite"]),"fidelity_valid":fidelity_valid,"outcome_quality":outcome,"contact_quality":contact,"motion_quality":motion,"composite_quality":composite,"nominal_quality":nominal_quality[(task,episode)],"improves_nominal":bool(fidelity_valid and composite>nominal_quality[(task,episode)]+.05),"macro_effect_h10":float(summary["macro_effect_h10"])})
    write_pq(artifacts/"candidate_quality.parquet",rows)
    grouped=defaultdict(list)
    for row in rows: grouped[(row["task"],row["generator_family"])].append(row)
    family_metrics=[];authorized_by_task={task:[] for task in TASKS}
    for (task,family),values in sorted(grouped.items()):
        by_branch=defaultdict(list)
        for row in values:by_branch[row["branch_id"]].append(row)
        ranges=[max(x["composite_quality"] for x in group)-min(x["composite_quality"] for x in group) for group in by_branch.values()]
        opportunity=[any(x["improves_nominal"] for x in group) for group in by_branch.values()]
        clipped=float(np.mean([x["clipped_chunk"] for x in values]));success=float(np.mean([x["outcome_quality"] for x in values]));opp=float(np.mean(opportunity));diversity=float(np.mean(ranges))
        family_metrics.append({"task":task,"generator_family":family,"candidate_count":len(values),"group_count":len(by_branch),"clipped_chunk_fraction":clipped,"success_rate":success,"opportunity_rate":opp,"diversity":diversity,"mean_macro_effect":float(np.mean([x["macro_effect_h10"] for x in values])),"authorized":authorize_family(clipped,success,opp,diversity)})
    for task in TASKS:
        eligible=[x for x in family_metrics if x["task"]==task and x["authorized"]]
        eligible=sorted(eligible,key=lambda x:(x["opportunity_rate"],x["success_rate"],x["diversity"]),reverse=True)[:4]
        authorized_by_task[task]=[x["generator_family"] for x in eligible]
    write_pq(artifacts/"family_metrics.parquet",family_metrics)
    authorization={"stage":args.stage,"authorized_by_task":authorized_by_task,"maximum_per_task":4,"rules":{"clipped_chunk_fraction_max":.10,"success_rate_min":.80,"opportunity_rate":"strictly positive","diversity_min":.02},"calibration_only":args.stage=="calibration"}
    dump(artifacts/"family_authorization.json",authorization)
    if args.plan_run:
        branch_path = ROOT/args.plan_run/"artifacts/branch_manifest.json"
        branch_records = json.loads(branch_path.read_text())
    else:
        branch_records = list({x["branch_id"]: {"branch_id":x["branch_id"],"task":x["task"]} for x in summaries}.values())
    branch_info = {x["branch_id"]: x for x in branch_records}
    all_groups={branch_id: [] for branch_id in branch_info}
    for row in rows:all_groups[row["branch_id"]].append(row)
    opportunity=[]
    for key, values in all_groups.items():
        task = branch_info[key]["task"]
        valid = [x for x in values if x["fidelity_valid"]]
        opportunity.append({"branch_id":key,"task":task,"opportunity":any(x["improves_nominal"] for x in valid),"gap":max([0.0]+[x["composite_quality"]-x["nominal_quality"] for x in valid]),"best_family":max(valid,key=lambda x:x["composite_quality"])["generator_family"] if valid else "nominal_no_authorized_candidate"})
    write_pq(artifacts/"opportunity_by_group.parquet",opportunity)
    labels=[f"{x['task'][:6]}:{x['generator_family'][1:3]}" for x in family_metrics];values=[x["opportunity_rate"] for x in family_metrics]
    fig,ax=plt.subplots(figsize=(14,5));ax.bar(range(len(values)),values);ax.set_xticks(range(len(values)),labels,rotation=70);ax.set(title="Candidate opportunity by task and family",ylabel="group opportunity rate");fig.tight_layout();fig.savefig(plots/"family_opportunity.png",dpi=160);plt.close(fig)
    task_opp={task:float(np.mean([x["opportunity"] for x in opportunity if x["task"]==task])) for task in TASKS}
    gaps=np.asarray([x["gap"] for x in opportunity],float)
    valid_fraction=float(np.mean([x["fidelity_valid"] for x in rows])) if rows else 0.0
    terminal_failure=float(np.mean([not x["outcome_quality"] for x in rows])) if rows else 0.0
    catastrophic=float(np.mean([(not x["outcome_quality"]) and x["contact_quality"] < .5 for x in rows])) if rows else 0.0
    metrics={"status":"completed","stage":args.stage,"candidate_count":len(rows),"group_count":len(all_groups),"valid_candidate_fraction":valid_fraction,"terminal_failure_rate":terminal_failure,"catastrophic_contact_rate":catastrophic,"overall_opportunity_rate":float(np.mean([x["opportunity"] for x in opportunity])),"median_oracle_improvement_gap":float(np.median(gaps)),"p90_oracle_improvement_gap":float(np.quantile(gaps,.9)),"task_opportunity_rate":task_opp,"authorized_by_task":authorized_by_task,"family_metrics":family_metrics}
    dump(out/"metrics.json",metrics);print(json.dumps(metrics,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
