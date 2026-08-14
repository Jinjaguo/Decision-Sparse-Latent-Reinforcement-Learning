#!/usr/bin/env python
"""Audit runtime geom ownership and freeze exact EXP7 pair groups."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import mujoco

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.envs.libero_runtime import bootstrap_runtime, environment_kwargs, load_episode, load_selection, task_source_record
from decision_sparse_rl.envs.mujoco_snapshot import native_model_data
from scripts.exp7.contact_geometry import geom_name, measure

TARGET={"open_the_middle_drawer_of_the_cabinet":"wooden_cabinet_1","put_the_bowl_on_the_plate":"akita_black_bowl_1","turn_on_the_stove":"flat_stove_1"}


def descendants(model, root):
    body={int(root)}; changed=True
    while changed:
        old=len(body); body.update(i for i in range(int(model.nbody)) if int(model.body_parentid[i]) in body); changed=len(body)!=old
    return body


def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--reference-run",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--audit-output",type=Path,required=True); args=p.parse_args()
    ref=args.reference_run.resolve(); manifest=json.loads((ref/"artifacts/reference_snapshots_manifest.json").read_text()); selection,tasks=load_selection(ROOT/"experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json",ROOT/"experiments/exp1_decision_sparsity/manifests/tasks.json"); selected={x["name"]:x for x in selection["tasks"]}
    wrapper,robosuite_root,assets_root=bootstrap_runtime(ROOT/"third_party/LIBERO",ROOT/"data",args.audit_output.parent/"libero_config")
    schema={"schema_version":1,"mujoco_version":mujoco.__version__,"distance_api":"mujoco.mj_geomDistance smallest signed surface distance; negative means penetration","distance_max_m":0.05,"tasks":{}}; audits=[]
    for task in sorted(selected):
        meta=selected[task]; source=task_source_record(tasks,meta["suite"],meta["task_id"]); env=wrapper.ControlEnv(**environment_kwargs(Path(source["bddl_file_path"])))
        first=next(x for x in manifest["episodes"] if x["task"]==task); load_episode(env,dataset_path=ROOT/"data"/meta["demonstration_relative_path"],episode_index=int(first["episode"].split("_")[-1]),robosuite_package_root=robosuite_root,libero_assets_root=assets_root)
        model,_=native_model_data(env.sim); target_root=int(env.env.obj_body_id[TARGET[task]]); target_bodies=descendants(model,target_root)
        all_object_bodies=set().union(*(descendants(model,int(v)) for v in env.env.obj_body_id.values())); other_object_bodies=all_object_bodies-target_bodies
        gripper_names=set(env.robots[0].gripper.contact_geoms); gripper_ids={i for i in range(int(model.ngeom)) if geom_name(model,i) in gripper_names}; target_ids={i for i in range(int(model.ngeom)) if int(model.geom_bodyid[i]) in target_bodies}; other_object_ids={i for i in range(int(model.ngeom)) if int(model.geom_bodyid[i]) in other_object_bodies}
        observed={}
        for episode in [x for x in manifest["episodes"] if x["task"]==task]:
            for boundary in json.loads((ref/episode["relative_directory"]/"boundaries.json").read_text()):
                for contact in boundary["contact_pairs"]:
                    a,b=int(contact["geom1_id"]),int(contact["geom2_id"]); key=tuple(sorted((a,b))); observed[key]=(contact["geom1_name"],contact["geom2_name"])
        groups={"target_gripper":[],"target_environment":[],"gripper_environment":[],"task_object_environment":[]}
        for (a,b),_ in sorted(observed.items()):
            aset={a,b}
            if aset & target_ids and aset & gripper_ids: group="target_gripper"
            elif aset & target_ids and aset & other_object_ids: group="task_object_environment"
            elif aset & target_ids: group="target_environment"
            elif aset & gripper_ids and not aset.issubset(gripper_ids): group="gripper_environment"
            else: continue
            pair="|".join(sorted((f"{a}:{geom_name(model,a)}",f"{b}:{geom_name(model,b)}")))
            groups[group].append({"geom1_id":a,"geom1_name":geom_name(model,a),"geom2_id":b,"geom2_name":geom_name(model,b),"pair":pair})
        groups={k:v for k,v in groups.items() if v}; relevant=sorted({p["pair"] for values in groups.values() for p in values}); required=[geom_name(model,next(iter(target_ids)))]
        channel=json.loads((ROOT/"experiments/exp4_replicated_progress_criticality/manifests/effect_channel_schema.json").read_text())
        schema["tasks"][task]={"target_body":TARGET[task],"required_geom_names":required,"task_object_body_names":channel["task_object_audit"][task]["bodies"],"pair_groups":groups,"relevant_pairs":relevant,"ownership_source":{"target":"env.env.obj_body_id plus model body ancestry","gripper":"env.robots[0].gripper.contact_geoms","contacts":"reference mjData.contact geom IDs"}}
        audit={"task":task,"ngeom":int(model.ngeom),"target_geom_count":len(target_ids),"gripper_geom_count":len(gripper_ids),"observed_relevant_pair_count":len(relevant),"groups":{k:len(v) for k,v in groups.items()}}
        audits.append(audit); env.close()
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(schema,indent=2,sort_keys=True)+"\n"); gate={"passed":len(schema["tasks"])==3 and all(len(x["relevant_pairs"])>0 for x in schema["tasks"].values()),"criteria":{"three_tasks":len(schema["tasks"])==3,"every_task_has_surface_pairs":all(len(x["relevant_pairs"])>0 for x in schema["tasks"].values())}}
    result={"schema_version":1,"gate":gate,"runtime_audits":audits,"official_docs":{"function":"https://mujoco.readthedocs.io/en/3.2.3/APIreference/APIfunctions.html#mj-geomdistance","contact_type":"https://mujoco.readthedocs.io/en/3.2.3/APIreference/APItypes.html#mjcontact"}}
    args.audit_output.parent.mkdir(parents=True,exist_ok=True); args.audit_output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2)); return 0 if gate["passed"] else 2


if __name__=="__main__": raise SystemExit(main())
