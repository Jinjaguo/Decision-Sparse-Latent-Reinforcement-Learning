"""Current-state-only risk selector utilities for EXP23."""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from scripts.exp12.prepare_ranking import TASKS,TASK_BODIES


def one(i,n):
    x=np.zeros(n);x[i]=1;return x


def current_context(reference_run:Path,branches:list[dict]):
    records=json.loads((reference_run/"artifacts/reference_snapshots_manifest.json").read_text())["episodes"];lookup={(x["task"],x["episode"]):x for x in records};cache={};result={}
    for b in branches:
        record=lookup[(b["task"],b["episode"])];directory=reference_run/record["relative_directory"];key=(b["task"],b["episode"])
        if key not in cache:cache[key]=json.loads((directory/"boundaries.json").read_text())
        row=cache[key][int(b["branch_time"])];names=row["body_names"];pos=np.zeros((2,3));quat=np.zeros((2,4))
        selected=TASK_BODIES[b["task"]]
        for i,name in enumerate(selected):pos[i]=row["body_positions"][names.index(name)];quat[i]=row["body_quaternions"][names.index(name)]
        eef=np.asarray(row["eef_position"],float);progress=row["progress_channels"]
        progress_values=[float(progress.get(name,0.0)) for name in ("joint_qpos","gripper_to_bowl_distance_m","bowl_to_plate_planar_distance_m","bowl_bottom_minus_plate_top_m")]
        result[b["branch_id"]]=np.r_[one(TASKS.index(b["task"]),len(TASKS)),eef,np.asarray(row["eef_orientation_matrix"],float).reshape(-1),pos.reshape(-1),quat.reshape(-1),(pos-eef).reshape(-1),np.asarray(row["gripper_state"],float),float(row["contact_count"])/100.,progress_values]
    return result


def safe(row):return bool(row["success"] and not row["safety_stop"])
def utility(row):return float(safe(row))-.75*float(row["safety_stop"])+.03*float(row["success"])*(1-min(int(row["steps"]),300)/300)


def choose_knn(context,task,training,k,routes):
    candidates=[x for x in training if x["task"]==task];matrix=np.asarray([x["context"] for x in candidates]);query=np.asarray(context);scale=matrix.std(0);scale[scale<1e-8]=1.;distance=np.linalg.norm((matrix-query)/scale,axis=1);order=np.argsort(distance)[:min(k,len(distance))];weight=1/np.maximum(distance[order],1e-3);weight/=weight.sum();scores=[]
    for route in routes:scores.append(float(np.dot(weight,[x["outcomes"][route]["utility"] for x in np.asarray(candidates,dtype=object)[order]])))
    return routes[int(np.argmax(scores))]


def choose_task_prior(task,training,routes):
    values=[x for x in training if x["task"]==task];return max(routes,key=lambda r:np.mean([x["outcomes"][r]["utility"] for x in values]))

