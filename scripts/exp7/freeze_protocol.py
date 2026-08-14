#!/usr/bin/env python
"""Freeze all EXP7 hypotheses, branches, directions, horizons, and GPU rules."""

from __future__ import annotations

import argparse, hashlib, json, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))
from scripts.exp4.freeze_protocol import canonical_basis

TASKS=["open_the_middle_drawer_of_the_cabinet","put_the_bowl_on_the_plate","turn_on_the_stove"]
RADII=[0.0003125,0.000625,0.00125]


def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def common(kind,now,project,sources): return {"schema_version":1,"manifest_type":kind,"frozen_at_utc":now,"project_sha":project,"sources":sources,"outcome_blind":True,"formal_q_outcomes_observed":False}


def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--reference-run",type=Path,required=True); p.add_argument("--branch-input",type=Path,required=True); p.add_argument("--geometry",type=Path,required=True); p.add_argument("--margin-spec",type=Path,required=True); p.add_argument("--output",type=Path,required=True); args=p.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    unexpected=[x for x in out.iterdir() if x.name != "contact_mode_schema.json"]
    if unexpected: raise FileExistsError(f"refusing nonempty EXP7 manifest directory: {unexpected}")
    now=datetime.now(timezone.utc).isoformat(); project=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(); refs=json.loads((args.reference_run/"artifacts/reference_snapshots_manifest.json").read_text()); branches=json.loads(args.branch_input.read_text()); geometry=pq.read_table(args.geometry).to_pylist(); margin=json.loads(args.margin_spec.read_text()); limits_list=json.loads((ROOT/"runs/exp2_r5_q_smoke_20260814T012633/artifacts/joint_limit_manifest.json").read_text()); limits={x["task"]:x for x in limits_list}
    sources={"reference_run":args.reference_run.resolve().name,"hashes":{"reference_manifest":sha(args.reference_run/"artifacts/reference_snapshots_manifest.json"),"branch_input":sha(args.branch_input),"reference_geometry":sha(args.geometry),"margin_spec":sha(args.margin_spec)}}; seed=int.from_bytes(hashlib.sha256(f"EXP7_DIRECTIONS_V1|{project}|{sources['hashes']['branch_input']}".encode()).digest()[:8],"little")
    ref_lookup={(x["task"],x["episode"]):x for x in refs["episodes"]}; geo_lookup={};
    for row in geometry: geo_lookup.setdefault((row["task"],row["episode"]),[]).append(row)
    direction_rows=[]; heldout=[]
    for ti,trajectory in enumerate(branches["trajectories"]):
        directory=args.reference_run/ref_lookup[(trajectory["task"],trajectory["episode"])]["relative_directory"]
        boundaries=json.loads((directory/"boundaries.json").read_text())
        limit=limits[trajectory["task"]]; lower=np.asarray(limit["lower"]); upper=np.asarray(limit["upper"]); span=upper-lower; used={int(x["action_index"]) for x in trajectory["branches"]}
        for bi,branch in enumerate(trajectory["branches"]):
            basis,_,rng=canonical_basis(np.random.SeedSequence(seed,spawn_key=(ti,bi))); random=rng.standard_normal(7); random/=np.linalg.norm(random); vectors=[basis[:,i] for i in range(7)]+[random]
            def admissible(index):
                q=np.asarray(boundaries[index]["panda_arm_q"])
                return all(np.all(q+s*r*span*v>=lower) and np.all(q+s*r*span*v<=upper) for r in RADII for v in vectors for s in (-1,1))
            if not admissible(int(branch["action_index"])):
                original=int(branch["action_index"]); candidates=sorted((x for x in geo_lookup[(trajectory["task"],trajectory["episode"])] if int(x["action_index"]) not in used),key=lambda x:(abs(int(x["action_index"])-original),int(x["action_index"])))
                replacement=next((x for x in candidates if admissible(int(x["action_index"]))),None)
                if replacement is None: raise RuntimeError(f"no joint-limit-valid replacement for {trajectory['task']}/{trajectory['episode']}/{original}")
                used.remove(original); used.add(int(replacement["action_index"])); branch.update({"action_index":int(replacement["action_index"]),"branch_time":int(replacement["action_index"]),"normalized_time":replacement["normalized_time"],"physical_progress_clipped":replacement["physical_progress_clipped"],"reference_contact_mode_json":replacement["contact_mode_json"],"reference_signed_gap_m":replacement["signed_gap_m"],"boundary_margin_class":replacement["boundary_margin_class"],"reference_gripper_state":replacement["gripper_state"],"reference_predicate_state":replacement["predicate"],"replacement_reason":f"nearest unused joint-limit-valid reference boundary replacing {original}"})
            q=np.asarray(boundaries[int(branch["action_index"])]["panda_arm_q"])
            for ri,radius in enumerate(RADII):
                for di,vector in enumerate(vectors):
                    row={"task":trajectory["task"],"episode":trajectory["episode"],"branch_time":int(branch["action_index"]),"radius_fraction":radius,"radius_label":f"r{radius:.7f}".rstrip("0"),"direction_index":di,"direction_role":"basis" if di<7 else "heldout_random","execution_position":ri*8+di,"unit_direction_scaled_coordinates":vector.tolist(),"unsigned_delta_q":(radius*span*vector).tolist(),"both_signs_within_joint_limits":True}; direction_rows.append(row)
                    if di==7: heldout.append(row)
    base=lambda kind:common(kind,now,project,sources); manifests={}
    manifests["exp7_cohort_manifest.json"]={**base("exp7_cohort_manifest"),"independent":True,"trajectory_count":30,"cohort":[{"task":x["task"],"episode":x["episode"]} for x in refs["episodes"]],"selection_rule":refs["selection_rule"]}
    contact=json.loads((ROOT/"experiments/exp7_contact_mode_response/manifests/contact_mode_schema.json").read_text()); contact.update(base("contact_mode_schema")); manifests["contact_mode_schema.json"]=contact
    manifests["contact_pair_group_manifest.json"]={**base("contact_pair_group_manifest"),"groups":["target_gripper","target_environment","gripper_environment","task_object_environment"],"exact_runtime_pairs_by_task":{t:contact["tasks"][t]["pair_groups"] for t in TASKS}}
    manifests["signed_gap_spec.json"]={**base("signed_gap_spec"),"api":"mujoco.mj_geomDistance","distance_max_m":contact["distance_max_m"],"aggregation":"minimum signed surface distance over each frozen group; global value is minimum group gap","normal_velocity":"nearest-point normal projected exact geom Jacobian relative velocity","no_body_center_substitution":True}
    manifests["boundary_margin_spec.json"]={**base("boundary_margin_spec"),**margin}
    branch_manifest={**base("branch_manifest"),**branches,"branch_count":360,"branches_per_demo":12}; manifests["branch_manifest.json"]=branch_manifest
    manifests["radius_manifest.json"]={**base("radius_manifest"),"radii":RADII,"expected_signed_interventions":17280}
    manifests["direction_basis_manifest.json"]={**base("direction_basis_manifest"),"master_seed_uint64":seed,"directions":direction_rows,"direction_rows":len(direction_rows)}
    manifests["heldout_direction_manifest.json"]={**base("heldout_direction_manifest"),"role":"eighth random direction excluded from seven-column fit","directions":heldout}
    manifests["horizon_manifest.json"]={**base("horizon_manifest"),"horizons":[1,3,5,"remaining"],"primary":1,"aggregation":"duration-normalized mean of first min(H, continuation length) signed output vectors","mode_preserved_through_H":"perturbed exact grouped mode equals matched-zero exact grouped mode at every offset 1..H"}
    manifests["signed_output_vector_spec.json"]={**base("signed_output_vector_spec"),"inherited_exactly_from":"EXP6","order":["arm_q[7]","arm_qvel[7]","eef_position[3]","eef_orientation_rotvec[3]","task_object_position[3*n]","task_object_orientation_rotvec[3*n]"]}
    manifests["operator_metric_spec.json"]={**base("operator_metric_spec"),"operator":"J[:,j]=(y_plus-y_minus)/(2r)","metrics":{"top1_min":.8,"top2_min":.75,"spectral_discrepancy_max":.2,"sign_asymmetry_max":.25,"heldout_vector_error_max":.35}}
    manifests["preserved_mode_inclusion_spec.json"]={**base("preserved_mode_inclusion_spec"),"intent_to_perturb":"all planned interventions retained","conditional":"both signs preserve matched-zero mode through horizon","transition_categories":["A_both_preserve","B_same_new_mode","C_signs_different_modes","D_one_sign_changes"],"no_deletion":True}
    features=["signed_gap_m","normal_relative_velocity_mps","contact_mode","eef_relative_pose","arm_q","arm_qvel","gripper_state","current_action","physical_progress","candidate_direction","radius"]
    manifests["mode_preservation_predictor_feature_spec.json"]={**base("mode_preservation_predictor_feature_spec"),"features":features,"forbidden":["response","future_success","terminal_state","perturbed_outcome"],"reference_only":True}
    folds={f"fold_{i}":[x["episode"] for x in refs["episodes"] if int(x["episode"].split("_")[-1])%5==i] for i in range(5)}; manifests["mode_preservation_predictor_crossfit_manifest.json"]={**base("mode_preservation_predictor_crossfit_manifest"),"folds":folds,"unit":"demonstration","model":"logistic regression with one-hot task/mode"}
    manifests["mode_preservation_predictor_decision_rule.json"]={**base("mode_preservation_predictor_decision_rule"),"threshold_selection":"training-fold Youden J, frozen before held-out labels","metrics":["AUROC","AUPRC","Brier","ECE","sensitivity","specificity"],"scheduler_readiness":"AUROC lower CI >=0.70 and ECE <=0.10"}
    manifests["statistical_analysis_plan.json"]={**base("statistical_analysis_plan"),"independent_unit":"demonstration","bootstrap":{"resamples":4000,"seed":970031,"hierarchy":"task then demonstration"},"multiple_comparisons":"BH FDR .05","H1":"at least 70% demos meet all four H1 preserved-mode convergence cutoffs and hierarchical top1 CI lower >.65","H2":"interior minus near+ambiguous convergence positive with demo-cluster CI excluding zero and BH q<.05","H3":"smallest-radius H1 preserved demo-median rank rho>=.65 and vector error<=.35","H4":"same mode+margin cross-demo top1 improves >=.15 over better time/progress baseline with CI lower>0","H5":"reference-only mode preservation predictor"}
    manifests["gpu_analysis_spec.json"]={**base("gpu_analysis_spec"),"device":"cuda:0 RTX 4090","dtype":"float64","simulator":"CPU","absolute_and_relative_tolerances":{"scalar":{"atol":1e-12,"rtol":1e-10},"matrix":{"atol":1e-11,"rtol":1e-9},"spectrum":{"atol":1e-10,"rtol":1e-8}},"no_threshold_relaxation":True,"no_automatic_cpu_fallback":True}
    manifests["scientific_decision_rule.json"]={**base("scientific_decision_rule"),"classification_priority":["within_mode_short_horizon_operator_converges","boundary_margin_explains_hybrid_nonsmoothness","contact_modes_explanatory_but_not_predictable","within_mode_nonsmoothness_persists","contact_schema_not_identifiable","no_support"],"H5_reported_separately":True}
    for name,value in manifests.items(): (out/name).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    for name in ("effect_channel_schema.json","effect_normalization.json","primary_metric_spec.json"):
        shutil.copy2(ROOT/"experiments/exp4_replicated_progress_criticality/manifests"/name,out/name)
    config={"experiment":"EXP7","signs":[-1,1],"directions_per_branch":24,"expected_branches":360,"non_arm_integration_linf_max":1e-12,"q_injection_atol":1e-15,"zero_integration_l2_max":1e-12,"meaningful_effect_threshold":0.01,"radii":RADII}; (out.parent/"configs").mkdir(exist_ok=True); (out.parent/"configs/exp7.json").write_text(json.dumps(config,indent=2)+"\n")
    hashes={name:sha(out/name) for name in manifests}; (out/"manifest_hashes.json").write_text(json.dumps({**base("manifest_hashes"),"manifests":hashes},indent=2,sort_keys=True)+"\n"); metrics={"manifest_count":len(manifests),"branches":sum(len(x["branches"]) for x in branches["trajectories"]),"direction_rows":len(direction_rows),"planned_signed_interventions":2*len(direction_rows),"gate":{"passed":len(manifests)==19 and len(direction_rows)==360*24}}; print(json.dumps(metrics,indent=2)); return 0 if metrics["gate"]["passed"] else 2


if __name__=="__main__": raise SystemExit(main())
