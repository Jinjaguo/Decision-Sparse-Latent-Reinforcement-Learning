#!/usr/bin/env python
"""Audit exact EXP5 development/confirmatory HDF5 availability before references."""

from __future__ import annotations

import argparse, contextlib, importlib.metadata, io, json, shlex, sys, traceback
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from decision_sparse_rl.envs.libero_runtime import load_selection  # noqa: E402
from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record  # noqa: E402
from decision_sparse_rl.utils.environment_audit import git_record  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-id", required=True); p.add_argument("--run-root", type=Path, default=ROOT / "runs"); p.add_argument("--dataset-root", type=Path, default=ROOT / "data"); args = p.parse_args()
    run = create_run_directory(args.run_root, args.run_id); out, err = io.StringIO(), io.StringIO(); rows=[]
    config={"stage":"EXP5-0","development":list(range(3,10)),"confirmatory":list(range(10,20)),"no_substitution":True}; gate={"passed":False}
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            selection,_=load_selection(ROOT/"experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json",ROOT/"experiments/exp1_decision_sparsity/manifests/tasks.json")
            for task in selection["tasks"]:
                path=args.dataset_root/task["demonstration_relative_path"]
                with h5py.File(path,"r") as h:
                    for episode in range(3,20):
                        key=f"demo_{episode}"; group=h["data"][key]; states=group["states"]; actions=group["actions"]
                        row={"task":task["name"],"episode":key,"cohort":"development" if episode<10 else "confirmatory","file":str(path.resolve()),"file_exists":path.exists(),"states":len(states),"actions":len(actions),"same_length":len(states)==len(actions),"model_xml_present":bool(group.attrs.get("model_file")),"finite_length":len(actions)>1}
                        rows.append(row); print(json.dumps(row,sort_keys=True))
            criteria={"exactly_21_development":sum(r["cohort"]=="development" for r in rows)==21,"exactly_30_confirmatory":sum(r["cohort"]=="confirmatory" for r in rows)==30,"all_files_exist":all(r["file_exists"] for r in rows),"all_lengths_valid":all(r["same_length"] and r["finite_length"] for r in rows),"all_model_xml_present":all(r["model_xml_present"] for r in rows)}; gate={"passed":all(criteria.values()),"criteria":criteria}
            write_json(run/"artifacts/availability.json",{"schema_version":1,"rows":rows,"gate":gate}); write_json(run/"artifacts/failure_examples.json",[r for r in rows if not all([r["file_exists"],r["same_length"],r["finite_length"],r["model_xml_present"]])])
        metrics={"run_id":args.run_id,"status":"completed","gate":gate,"development_count":21,"confirmatory_count":30}; write_run_record(run,config=config,command=shlex.join([sys.executable,*sys.argv]),environment={"python":sys.version,"h5py":h5py.__version__,"mujoco":importlib.metadata.version("mujoco"),"robosuite":importlib.metadata.version("robosuite")},git_state={"project":git_record(ROOT)},stdout=out.getvalue(),stderr=err.getvalue(),metrics=metrics); return 0 if gate["passed"] else 2
    except Exception as exc:
        err.write(traceback.format_exc()); write_run_record(run,config=config,command=shlex.join([sys.executable,*sys.argv]),environment={"python":sys.version},git_state={"project":git_record(ROOT)},stdout=out.getvalue(),stderr=err.getvalue(),metrics={"run_id":args.run_id,"status":"failed","gate":{"passed":False},"error":repr(exc)}); return 1

if __name__=="__main__": raise SystemExit(main())
