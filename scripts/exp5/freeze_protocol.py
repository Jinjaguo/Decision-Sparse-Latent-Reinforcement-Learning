#!/usr/bin/env python
"""Freeze outcome-blind EXP5 state matching, branches, directions, and analysis plan."""

from __future__ import annotations

import argparse, hashlib, json, subprocess, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from decision_sparse_rl.metrics.exp5 import robust_scales, shrinkage_covariance, mahalanobis_cost, monotone_match, select_prototype_branches  # noqa: E402
from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402
from scripts.exp4.freeze_protocol import articulation_progress, bowl_progress, canonical_basis  # noqa: E402

TASKS = ["open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_plate", "turn_on_the_stove"]
TARGET_BODY = {TASKS[0]: "wooden_cabinet_1_cabinet_middle", TASKS[1]: "akita_black_bowl_1_main", TASKS[2]: "flat_stove_1_button"}
CONTACT_TOKEN = {TASKS[0]: "wooden_cabinet_1", TASKS[1]: "akita_black_bowl_1", TASKS[2]: "flat_stove_1_button"}
FLOORS = {"q": .01, "qvel": .02, "position": .002, "orientation": .01, "velocity": .005, "gripper": .005, "object": .002, "binary": 1.0}

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def current_sha() -> str:
    return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()

def common(kind:str, ts:str, project_sha:str, sources:dict)->dict:
    return {"schema_version":1,"manifest_type":kind,"frozen_at_utc":ts,"project_sha":project_sha,"sources":sources,"outcome_blind":True,"confirmatory_q_outcomes_observed":False}

def body(boundary:dict, name:str)->tuple[np.ndarray,np.ndarray]:
    i=boundary["body_names"].index(name)
    return np.asarray(boundary["body_positions"][i],float),np.asarray(boundary["body_quaternions"][i],float)

def contact(boundary:dict, token:str)->float:
    return float(any(token in str(x.get("geom1_name","")) or token in str(x.get("geom2_name","")) for x in boundary["contact_pairs"]))

def descriptor(directory:Path, task:str, limits:dict)->tuple[np.ndarray,list[str],list[dict]]:
    boundaries=json.loads((directory/"boundaries.json").read_text()); low=np.asarray(limits["lower"]); span=np.asarray(limits["upper"])-low
    rows=[]; meta=[]; names=[]
    for t,b in enumerate(boundaries):
        with np.load(directory/f"controller_{t:04d}.npz",allow_pickle=False) as z:
            q=(np.asarray(b["panda_arm_q"])-low)/span; qv=np.asarray(z["controller__joint_vel"]); rot=np.asarray(b["eef_orientation_matrix"]); eef=np.asarray(b["eef_position"])
            values=list(q)+list(qv)+list(eef)+list(rot[:,:2].reshape(-1))+list(z["controller__ee_pos_vel"])+list(z["controller__ee_ori_vel"])+[float(np.sum(np.abs(b["gripper_state"]))),float(z["gripper__current_action"].reshape(-1)[0]),contact(b,CONTACT_TOKEN[task]),float(b["progress_channels"]["exact_task_predicate"])]
            local_names=[f"q_norm_{i}" for i in range(7)]+[f"qvel_{i}" for i in range(7)]+[f"eef_pos_{i}" for i in range(3)]+[f"eef_rot6d_{i}" for i in range(6)]+[f"eef_linvel_{i}" for i in range(3)]+[f"eef_angvel_{i}" for i in range(3)]+["gripper_opening_l1","gripper_current_action","task_contact","exact_predicate"]
            p,quat=body(b,TARGET_BODY[task])
            if task==TASKS[1]:
                pp,pq_=body(b,"plate_1_main"); extra=list(p)+list(quat)+list(pp)+list(pq_)+list(p-pp)+list(eef-p)
                extra_names=[f"bowl_pos_{i}" for i in range(3)]+[f"bowl_quat_{i}" for i in range(4)]+[f"plate_pos_{i}" for i in range(3)]+[f"plate_quat_{i}" for i in range(4)]+[f"bowl_minus_plate_{i}" for i in range(3)]+[f"eef_minus_bowl_{i}" for i in range(3)]
            else:
                extra=list(p)+list(quat)+list(eef-p)+[float(b["progress_channels"]["joint_qpos"])]
                prefix="drawer" if task==TASKS[0] else "stove_button"; extra_names=[f"{prefix}_pos_{i}" for i in range(3)]+[f"{prefix}_quat_{i}" for i in range(4)]+[f"eef_minus_{prefix}_{i}" for i in range(3)]+[f"{prefix}_joint_qpos"]
            values+=extra; local_names+=extra_names
            rows.append(values); meta.append({"action_index":t,"normalized_time":t/max(len(boundaries)-1,1),"predicate":bool(b["task_success"])})
            if not names: names=local_names
    x=np.asarray(rows,float)
    if not np.all(np.isfinite(x)): raise RuntimeError(f"nonfinite descriptor {task}/{directory.name}")
    return x,names,meta

def farthest(x:np.ndarray,k:int)->np.ndarray:
    selected=[int(np.argmin(np.linalg.norm(x-np.median(x,axis=0),axis=1)))]; d=np.linalg.norm(x-x[selected[0]],axis=1)
    while len(selected)<k:
        nxt=int(np.argmax(d)); selected.append(nxt); d=np.minimum(d,np.linalg.norm(x-x[nxt],axis=1))
    return np.asarray(selected)

def progress(boundaries:list[dict],task:str)->np.ndarray:
    p=bowl_progress(boundaries) if task==TASKS[1] else articulation_progress(boundaries)
    return np.asarray(p["clipped"],float)

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--run-id",required=True); ap.add_argument("--development-reference-run",type=Path,required=True); ap.add_argument("--confirmatory-reference-run",type=Path,required=True); ap.add_argument("--output",type=Path,default=ROOT/"experiments/exp5_state_conditioned_anisotropic/manifests"); ap.add_argument("--run-root",type=Path,default=ROOT/"runs"); args=ap.parse_args()
    run=create_run_directory(args.run_root,args.run_id); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False); ts=datetime.now(timezone.utc).isoformat(); project=current_sha()
    dev=args.development_reference_run.resolve(); conf=args.confirmatory_reference_run.resolve(); config=json.loads((ROOT/"experiments/exp5_state_conditioned_anisotropic/configs/exp5.json").read_text()); limits_list=json.loads((ROOT/"runs/exp2_r5_q_smoke_20260814T012633/artifacts/joint_limit_manifest.json").read_text()); limits={x["task"]:x for x in limits_list}
    dm=json.loads((dev/"artifacts/reference_snapshots_manifest.json").read_text()); cm=json.loads((conf/"artifacts/reference_snapshots_manifest.json").read_text());
    if not dm["gate"]["passed"] or not cm["gate"]["passed"]: raise RuntimeError("reference gate failed")
    sources={"runs":{"development_references":dev.name,"confirmatory_references":conf.name},"hashes":{"development_references":sha(dev/"artifacts/reference_snapshots_manifest.json"),"confirmatory_references":sha(conf/"artifacts/reference_snapshots_manifest.json"),"config":sha(ROOT/"experiments/exp5_state_conditioned_anisotropic/configs/exp5.json")}}
    data={}; descriptor_rows=[]; schemas={}; scales={}; folds=[]
    for cohort,root,manifest in [("development",dev,dm),("confirmatory",conf,cm)]:
        for rec in manifest["episodes"]:
            task,ep=rec["task"],rec["episode"]; x,names,meta=descriptor(root/rec["relative_directory"],task,limits[task]); data[(cohort,task,ep)]={"x":x,"meta":meta,"record":rec,"dir":root/rec["relative_directory"]}; schemas[task]=names
            for i,row in enumerate(x): descriptor_rows.append({"cohort":cohort,"task":task,"episode":ep,"action_index":i,"descriptor":row.tolist()})
    for task in TASKS:
        dx=np.concatenate([v["x"] for (c,t,_),v in data.items() if c=="development" and t==task]); names=schemas[task]; floor=np.asarray([FLOORS["q"] if n.startswith("q_norm") else FLOORS["qvel"] if n.startswith("qvel") else FLOORS["orientation"] if "rot6d" in n or "quat" in n else FLOORS["velocity"] if "vel" in n else FLOORS["gripper"] if "gripper" in n else FLOORS["binary"] if n in ("task_contact","exact_predicate") else FLOORS["object"] for n in names]); center,scale=robust_scales(dx,floor); weights=np.asarray([.25 if n in ("task_contact","exact_predicate") else 1. for n in names]); scales[task]={"center":center,"scale":scale,"weights":weights,"floor":floor}
        eps=sorted([int(k[2].split("_")[-1]) for k in data if k[0]=="confirmatory" and k[1]==task])
        for pos,e in enumerate(eps): folds.append({"task":task,"episode":f"demo_{e}","fold":pos%5,"position_in_task":pos})
    fold_lookup={(x["task"],x["episode"]):x["fold"] for x in folds}; prototype_records=[]; branch_trajs=[]; match_rows=[]; baseline_rows=[]
    for task in TASKS:
        s=scales[task]
        for fold in range(5):
            fit=[]; origin=[]
            for (cohort,t,ep),v in data.items():
                if t!=task or (cohort=="confirmatory" and fold_lookup[(task,ep)]==fold): continue
                z=(v["x"]-s["center"])/s["scale"]*s["weights"]; fit.append(z); origin += [(cohort,ep,i) for i in range(len(z))]
            fitx=np.concatenate(fit); inds=farthest(fitx,16); prot=fitx[inds]; cov=shrinkage_covariance(fitx,config["shrinkage"]); precision=np.linalg.pinv(cov)
            train_dist=[]
            for z in fit: train_dist.extend(np.min(mahalanobis_cost(z,prot,precision),axis=1).tolist())
            threshold=float(np.quantile(train_dist,config["match_rejection_quantile"])); prototype_records.append({"task":task,"fold":fold,"fit_reference_count":len(fit),"prototype_vectors":prot.tolist(),"source_states":[{"cohort":origin[i][0],"episode":origin[i][1],"action_index":origin[i][2]} for i in inds],"precision":precision.tolist(),"rejection_threshold":threshold})
            for (cohort,t,ep),v in data.items():
                if cohort!="confirmatory" or t!=task or fold_lookup[(task,ep)]!=fold: continue
                z=(v["x"]-s["center"])/s["scale"]*s["weights"]; selected=select_prototype_branches(z,prot); boundaries=json.loads((v["dir"]/"boundaries.json").read_text()); prog=progress(boundaries,task); branches=[]
                for pid,idx in enumerate(selected):
                    dist=float(mahalanobis_cost(z[idx:idx+1],prot[pid:pid+1],precision)[0,0]); branches.append({"action_index":int(idx),"branch_time":int(idx),"normalized_time":float(idx/max(len(z)-1,1)),"physical_progress_raw":float(prog[idx]),"physical_progress_clipped":float(prog[idx]),"kind":"state_prototype","state_prototype_id":pid,"crossfit_fold":fold,"state_match_distance":dist,"state_match_accepted":dist<=threshold})
                branch_trajs.append({"suite":v["record"]["suite"],"task":task,"task_id":v["record"]["task_id"],"episode":ep,"trajectory_length":len(z),"branches":branches})
    branch_trajs.sort(key=lambda x:(TASKS.index(x["task"]),int(x["episode"].split("_")[-1])))
    # Outcome-blind alignment diagnostics on reference descriptors.
    for task in TASKS:
        demos=[x for x in branch_trajs if x["task"]==task]
        for ai,a in enumerate(demos):
            za=(data[("confirmatory",task,a["episode"])]["x"]-scales[task]["center"])/scales[task]["scale"]*scales[task]["weights"]
            for b in demos[ai+1:]:
                zb=(data[("confirmatory",task,b["episode"])]["x"]-scales[task]["center"])/scales[task]["scale"]*scales[task]["weights"]; precision=np.eye(za.shape[1]); costs=mahalanobis_cost(za,zb,precision); mm=monotone_match(costs,config["sakoe_chiba_fraction"])
                match_rows.append({"task":task,"episode_a":a["episode"],"episode_b":b["episode"],"method":"state","mean_cost":float(mm["mean_cost"]),"path_length":len(mm["path"]),"path":mm["path"].tolist()})
                baseline_rows.append({"task":task,"episode_a":a["episode"],"episode_b":b["episode"],"time_mean_cost":float(np.mean([costs[int(round(i*(len(za)-1)/15)),int(round(i*(len(zb)-1)/15))] for i in range(16)])),"state_mean_cost":float(mm["mean_cost"])})
    pq.write_table(pa.Table.from_pylist(descriptor_rows),run/"artifacts/development_state_descriptor.parquet",compression="zstd"); pq.write_table(pa.Table.from_pylist(match_rows),run/"artifacts/development_pairwise_match_tables.parquet",compression="zstd"); pq.write_table(pa.Table.from_pylist(baseline_rows),run/"artifacts/development_baseline_comparisons.parquet",compression="zstd")
    (run/"artifacts/development_state_matching_report.md").write_text(f"# EXP5 reference-only state matching development\n\nDescriptors: {len(descriptor_rows):,} states. Confirmatory pairwise state paths: {len(match_rows)}. No q-intervention outcome was read.\n",encoding="utf-8")
    base=common("crossfit_fold_manifest",ts,project,sources); manifests={}
    manifests["crossfit_fold_manifest.json"]={**base,"folds":folds,"rule":"sorted qualified episode index modulo five per task; each fold has two held-out demos"}
    manifests["state_descriptor_schema.json"]={**common("state_descriptor_schema",ts,project,sources),"tasks":{t:[{"name":n,"index":i,"source":"audited reference boundary/controller snapshot"} for i,n in enumerate(schemas[t])] for t in TASKS},"runtime_identifiers":{"target_body":TARGET_BODY,"contact_prefix":CONTACT_TOKEN}}
    manifests["state_descriptor_scaling.json"]={**common("state_descriptor_scaling",ts,project,sources),"development_only":True,"tasks":{t:{"center":scales[t]["center"].tolist(),"scale":scales[t]["scale"].tolist(),"physical_floor":scales[t]["floor"].tolist(),"weight":scales[t]["weights"].tolist()} for t in TASKS}}
    manifests["state_distance_spec.json"]={**common("state_distance_spec",ts,project,sources),"metric":"task/fold-specific shrinkage Mahalanobis","shrinkage":config["shrinkage"],"binary_weight":.25,"missing_values":"forbidden"}
    manifests["prototype_manifest.json"]={**common("prototype_manifest",ts,project,sources),"prototypes_per_task_fold":16,"selection":"deterministic farthest-point medoids on development plus crossfit training references","records":prototype_records}
    manifests["matching_spec.json"]={**common("matching_spec",ts,project,sources),"algorithm":"monotone dynamic time warping","sakoe_chiba_fraction":config["sakoe_chiba_fraction"],"baselines":["normalized_time","EXP4_scalar_progress","robot_only","object_only","unconstrained_nearest_neighbor"]}
    branch_manifest={**common("confirmatory_branch_manifest",ts,project,sources),"branch_count":sum(len(x["branches"]) for x in branch_trajs),"branches_per_demo":16,"selection":"nearest unique state to fold-trained prototypes","trajectories":branch_trajs}
    manifests["confirmatory_branch_manifest.json"]=branch_manifest; manifests["branch_manifest.json"]={**branch_manifest,"manifest_type":"branch_manifest"}
    manifests["match_rejection_manifest.json"]={**common("match_rejection_manifest",ts,project,sources),"quantile":config["match_rejection_quantile"],"threshold_source":"fold training references only","rejections_preserved":True}
    seed_material=f"EXP5_DIRECTIONS_V1|{project}|{hashlib.sha256(json.dumps(branch_trajs,sort_keys=True).encode()).hexdigest()}"; seed0=int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8],"little"); direction_rows=[]; held=[]; large=[]
    for ti,tr in enumerate(branch_trajs):
        lim=limits[tr["task"]]; span=np.asarray(lim["upper"])-np.asarray(lim["lower"]); boundaries=json.loads((conf/cm["episodes"][[ (x["task"],x["episode"]) for x in cm["episodes"]].index((tr["task"],tr["episode"]))]["relative_directory"]/"boundaries.json").read_text())
        for bi,b in enumerate(tr["branches"]):
            basis,audit,rng=canonical_basis(np.random.SeedSequence(seed0,spawn_key=(ti,bi))); rnd=rng.standard_normal(7); rnd/=np.linalg.norm(rnd); q=np.asarray(boundaries[b["action_index"]]["panda_arm_q"]); is_large=((ti*16+bi)%5==0)
            if is_large: large.append({"task":tr["task"],"episode":tr["episode"],"branch_time":b["action_index"],"state_prototype_id":b["state_prototype_id"]})
            for radius,label in [(.0025,"small"),(.005,"main")]+([(.01,"large")] if is_large else []):
                for di,vec,role in [(i,basis[:,i],"basis") for i in range(7)]+[(7,rnd,"heldout_random")]:
                    delta=radius*span*vec
                    if not all(np.all(q+s*delta>=lim["lower"]) and np.all(q+s*delta<=lim["upper"]) for s in [-1,1]): raise RuntimeError("joint limit violation")
                    row={"task":tr["task"],"episode":tr["episode"],"branch_time":b["action_index"],"state_prototype_id":b["state_prototype_id"],"crossfit_fold":b["crossfit_fold"],"radius_fraction":radius,"radius_label":label,"direction_index":di,"direction_role":role,"execution_position":({"small":0,"main":8,"large":16}[label]+di),"unit_direction_scaled_coordinates":vec.tolist(),"unsigned_delta_q":delta.tolist(),"both_signs_within_joint_limits":True}; direction_rows.append(row)
                    if role=="heldout_random": held.append(row)
    manifests["direction_basis_manifest.json"]={**common("direction_basis_manifest",ts,project,sources),"seed_material":seed_material,"master_seed_uint64":seed0,"bit_generator":"PCG64","directions":direction_rows,"direction_rows":len(direction_rows),"signed_intervention_count":2*len(direction_rows)}
    manifests["heldout_direction_manifest.json"]={**common("heldout_direction_manifest",ts,project,sources),"role":"eighth direction, excluded from seven-column fitted operator","directions":held}
    manifests["radius_manifest.json"]={**common("radius_manifest",ts,project,sources),"radii":[{"label":"small","fraction":.0025,"branches":480},{"label":"main","fraction":.005,"branches":480},{"label":"large","fraction":.01,"branches":96}],"expected_signed_interventions":16896}
    manifests["large_radius_calibration_subset.json"]={**common("large_radius_calibration_subset",ts,project,sources),"selection":"every fifth branch in frozen task/demo/prototype order","count":len(large),"branches":large}
    exp4=ROOT/"experiments/exp4_replicated_progress_criticality/manifests"
    for new,old in [("effect_channel_schema.json","effect_channel_schema.json"),("effect_normalization.json","effect_normalization.json")]:
        value=json.loads((exp4/old).read_text()); value.update(common(new[:-5],ts,project,sources)); manifests[new]=value
    manifests["signed_output_vector_spec.json"]={**common("signed_output_vector_spec",ts,project,sources),"order":["arm_q[7]","arm_qvel[7]","eef_position[3]","eef_orientation_rotvec[3]","task_object_position[3*n]","task_object_orientation_rotvec[3*n]"],"aggregation":"duration-normalized remaining-horizon mean","finite_required":True}
    manifests["scalar_metric_spec.json"]={**common("scalar_metric_spec",ts,project,sources),"branch_metric":"S_RMS=sqrt(mean_j(((E+j+E-j)/2)^2)) over seven basis directions","radii_separate":True,"meaningful_effect_threshold":config["meaningful_effect_threshold"]}
    manifests["primary_metric_spec.json"]={**manifests["scalar_metric_spec.json"],"manifest_type":"primary_metric_spec"}
    manifests["operator_metric_spec.json"]={**common("operator_metric_spec",ts,project,sources),"operator":"J[:,j]=(y_plus-y_minus)/(2r), seven basis directions","gram":"J.T@J","primary_subspace_k":1,"secondary_subspace_k":2,"projector_similarity":"1-||Pa-Pb||F/sqrt(2k)","naming_rule":"finite-radius response operator unless local-linearity gate passes"}
    manifests["linearity_gate.json"]={**common("linearity_gate",ts,project,sources),"all_required":{"small_main_top1_similarity_min":.70,"relative_spectral_norm_discrepancy_max":.35,"sign_asymmetry_max":.40,"heldout_relative_prediction_error_max":.50}}
    manifests["statistical_analysis_plan.json"]={**common("statistical_analysis_plan",ts,project,sources),"independent_unit":"demonstration","bootstrap":{"resamples":config["bootstrap_resamples"],"seed":950031,"hierarchy":"task then demo"},"permutation":{"resamples":config["permutation_resamples"],"seed":950032,"unit":"demo within task"},"multiple_comparisons":"BH FDR 0.05","crossfit":5,"lodo":True,"loto":True}
    manifests["gpu_analysis_spec.json"]={**common("gpu_analysis_spec",ts,project,sources),"device":"cuda:0 RTX 4090","dtype":"float64","simulator":"CPU","gpu_workloads":["Mahalanobis","operator assembly","SVD/eigendecomposition","bootstrap","permutation"],"tolerances":{"scalar_atol":1e-12,"scalar_rtol":1e-10,"matrix_atol":1e-11,"spectrum_atol":1e-10}}
    manifests["scientific_decision_rule.json"]={**common("scientific_decision_rule",ts,project,sources),"strong_all_required":["state scalar delta over better baseline median >=0.15","cluster bootstrap CI lower >0","at least 2/3 task state scalar rho >=0.60","at least 2/3 task median matched top1 >=0.70","at least 70% demos cross-radius top1 >=0.70","heldout prediction median rho >=0.60","LODO scalar improvement positive","LOTO qualitatively stable","confirmatory families BH-FDR <0.05"],"classification_priority":["state_conditioned_replicated_anisotropic_criticality","state_alignment_only_without_subspace_replication","subspace_replication_without_scalar_sparsity","finite_radius_nonlinearity_dominates","trajectory_specific_criticality","no_confirmatory_support"]}
    for name,value in manifests.items(): write_json(out/name,value)
    hashes={name:sha(out/name) for name in manifests}; write_json(out/"manifest_hashes.json",{**common("manifest_hashes",ts,project,sources),"manifests":hashes})
    metrics={"run_id":args.run_id,"status":"completed","gate":{"passed":len(branch_trajs)==30 and sum(len(x["branches"]) for x in branch_trajs)==480 and len(direction_rows)==8448 and len(large)==96},"descriptor_states":len(descriptor_rows),"confirmatory_demos":len(branch_trajs),"branches":480,"direction_rows":len(direction_rows),"planned_interventions":2*len(direction_rows),"large_branches":len(large),"manifest_count":len(manifests)}
    write_run_record(run,config={"stage":"EXP5-1 through EXP5-9 protocol freeze"},command=" ".join(sys.argv),environment={"python":sys.version,"numpy":np.__version__},git_state={"project":git_record(ROOT)},metrics=metrics,stdout=json.dumps(metrics,sort_keys=True),stderr=""); print(json.dumps(metrics,indent=2)); return 0 if metrics["gate"]["passed"] else 2

if __name__=="__main__": raise SystemExit(main())
