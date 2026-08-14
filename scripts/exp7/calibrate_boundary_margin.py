#!/usr/bin/env python
"""Restore every independent reference boundary and calibrate signed-gap precision."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.envs.libero_runtime import bootstrap_runtime, environment_kwargs, load_episode, load_selection, task_source_record
from decision_sparse_rl.envs import controller_snapshot, mujoco_snapshot
from scripts.exp7.contact_geometry import load_schema, measure


def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--reference-run",type=Path,required=True); p.add_argument("--schema",type=Path,required=True); p.add_argument("--run-dir",type=Path,required=True); args=p.parse_args()
    run=args.run_dir.resolve(); (run/"artifacts").mkdir(parents=True,exist_ok=False); ref=args.reference_run.resolve(); ref_manifest=json.loads((ref/"artifacts/reference_snapshots_manifest.json").read_text()); schema=load_schema(args.schema)
    selection,tasks=load_selection(ROOT/"experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json",ROOT/"experiments/exp1_decision_sparsity/manifests/tasks.json"); selected={x["name"]:x for x in selection["tasks"]}; wrapper,robosuite_root,assets_root=bootstrap_runtime(ROOT/"third_party/LIBERO",ROOT/"data",run/"artifacts/libero_config")
    rows=[]; repeats=[]
    for trajectory_index,record in enumerate(ref_manifest["episodes"]):
        task=selected[record["task"]]; source=task_source_record(tasks,task["suite"],task["task_id"]); env=wrapper.ControlEnv(**environment_kwargs(Path(source["bddl_file_path"]))); load_episode(env,dataset_path=ROOT/"data"/task["demonstration_relative_path"],episode_index=int(record["episode"].split("_")[-1]),robosuite_package_root=robosuite_root,libero_assets_root=assets_root)
        directory=ref/record["relative_directory"]; boundaries=json.loads((directory/"boundaries.json").read_text()); archive=np.load(directory/"trajectory_states.npz",allow_pickle=False); integrations=np.asarray(archive["integration"])
        for index,boundary in enumerate(boundaries):
            snapshot=mujoco_snapshot.MujocoSnapshot("integration",mujoco_snapshot.state_spec("integration"),integrations[index].copy()); mujoco_snapshot.restore(env.sim,snapshot,forward=True); controller_snapshot.restore(env,controller_snapshot.deserialize(directory/f"controller_{index:04d}.npz")); geometry=measure(env,schema,record["task"])
            command=float(boundary["progress_channels"]["gripper_command"]); grip="negative" if command < -.5 else "positive" if command > .5 else "neutral"
            rows.append({"task":record["task"],"episode":record["episode"],"action_index":index,"trajectory_length":len(boundaries),"normalized_time":index/max(len(boundaries)-1,1),"physical_progress_clipped":float(boundary["progress_channels"].get("joint_qpos",boundary["progress_channels"].get("bowl_to_plate_planar_distance_m",index/max(len(boundaries)-1,1)))),"gripper_state":grip,"gripper_command":command,"predicate":bool(boundary["progress_channels"]["exact_task_predicate"]),**{k:v for k,v in geometry.items() if k != "contact_mode"}})
            if index in set(np.linspace(0,len(boundaries)-1,4,dtype=int)):
                values=[]
                for repeat in range(4):
                    mujoco_snapshot.restore(env.sim,snapshot,forward=True); controller_snapshot.restore(env,controller_snapshot.deserialize(directory/f"controller_{index:04d}.npz")); values.append(measure(env,schema,record["task"]))
                repeats.append({"task":record["task"],"episode":record["episode"],"action_index":index,"signed_gap_range_m":float(np.ptp([x["signed_gap_m"] for x in values])),"mode_repeat_exact":len({x["contact_mode_json"] for x in values})==1,"normal_velocity_range_mps":float(np.ptp([x["normal_relative_velocity_mps"] for x in values]))})
        archive.close(); env.close(); print(json.dumps({"trajectory":f"{record['task']}/{record['episode']}","boundaries":len(boundaries)}))
    precision=max([x["signed_gap_range_m"] for x in repeats]+[0.0]); velocity_precision=max([x["normal_velocity_range_mps"] for x in repeats]+[0.0]); near=max(1e-12,10*precision); far=1e-3
    for row in rows:
        value=abs(row["signed_gap_m"]); row["boundary_margin_class"]="ambiguous" if value<near else "near_boundary" if value<far else "interior"
    pq.write_table(pa.Table.from_pylist(rows),run/"artifacts/reference_contact_geometry.parquet",compression="zstd"); pq.write_table(pa.Table.from_pylist(repeats),run/"artifacts/boundary_margin_calibration.parquet",compression="zstd")
    spec={"schema_version":1,"m_near_m":near,"m_far_m":far,"near_rule":"max(1e-12 m, 10 * maximum four-restore signed-gap range)","far_rule":"fixed physical 1 mm reference-only threshold","signed_gap_repeatability_max_m":precision,"normal_velocity_repeatability_max_mps":velocity_precision,"mode_all_exact":all(x["mode_repeat_exact"] for x in repeats),"calibration_boundary_count":len(repeats),"reference_boundary_count":len(rows),"gate":{"passed":precision<=1e-10 and all(x["mode_repeat_exact"] for x in repeats)}}
    (run/"artifacts/boundary_margin_spec.json").write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n"); (run/"metrics.json").write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n"); print(json.dumps(spec,indent=2)); return 0 if spec["gate"]["passed"] else 2


if __name__=="__main__": raise SystemExit(main())
