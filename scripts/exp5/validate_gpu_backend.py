#!/usr/bin/env python
"""Audit EXP5 float64 GPU numerical kernels against CPU truth."""

from __future__ import annotations
import argparse, json, shlex, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record
from decision_sparse_rl.metrics.exp5 import central_operator, operator_geometry, projector, projector_similarity
from decision_sparse_rl.utils.environment_audit import git_record

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--run-id",required=True); p.add_argument("--run-root",type=Path,default=ROOT/"runs"); p.add_argument("--manifest-dir",type=Path,default=ROOT/"experiments/exp5_state_conditioned_anisotropic/manifests"); a=p.parse_args(); run=create_run_directory(a.run_root,a.run_id); started=time.perf_counter()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.cuda.set_device(0); rng=np.random.Generator(np.random.PCG64(950010)); records=[]
    for task_i,task in enumerate(["open_the_middle_drawer_of_the_cabinet","put_the_bowl_on_the_plate","turn_on_the_stove"]):
        for radius in [.0025,.005]:
            plus=rng.normal(size=(7,39+task_i*6)); minus=rng.normal(size=plus.shape); cpu=central_operator(plus,minus,radius); tg=(torch.as_tensor(plus,device="cuda",dtype=torch.float64)-torch.as_tensor(minus,device="cuda",dtype=torch.float64)).T/(2*radius); gpu=tg.cpu().numpy(); cg=operator_geometry(cpu); gram_gpu=(tg.T@tg).cpu().numpy(); sv_gpu=torch.linalg.svdvals(tg).cpu().numpy(); p1=projector(cg["right_vectors"],1); sim=projector_similarity(p1,p1,1)
            records.append({"task":task,"radius":radius,"operator_max_abs":float(np.max(np.abs(cpu-gpu))),"gram_max_abs":float(np.max(np.abs(cg["gram"]-gram_gpu))),"singular_max_abs":float(np.max(np.abs(cg["singular_values"]-sv_gpu))),"identity_projector_similarity":sim})
    criteria={"operator":max(x["operator_max_abs"] for x in records)<=1e-11,"gram":max(x["gram_max_abs"] for x in records)<=1e-8,"singular":max(x["singular_max_abs"] for x in records)<=1e-8,"projector":all(x["identity_projector_similarity"]>=1-1e-12 for x in records),"float64":torch.tensor([1.],device="cuda",dtype=torch.float64).dtype==torch.float64}
    audit={"torch_version":torch.__version__,"torch_cuda_version":torch.version.cuda,"cuda_available":True,"device_index":0,"gpu_name":torch.cuda.get_device_name(0),"vram_bytes":torch.cuda.get_device_properties(0).total_memory,"dtype":"float64","formal_device":"cuda:0","records":records,"criteria":criteria,"passed":all(criteria.values())}; write_json(run/"artifacts/gpu_audit.json",audit); write_json(run/"artifacts/gpu_cpu_equivalence.json",audit)
    metrics={"run_id":a.run_id,"status":"completed","gate":{"passed":audit["passed"],"criteria":criteria},"gpu":audit["gpu_name"],"wall_time_seconds":time.perf_counter()-started}; write_run_record(run,config={"stage":"EXP5-10 GPU equivalence","manifest_dir":str(a.manifest_dir)},command=shlex.join([sys.executable,*sys.argv]),environment=audit,git_state={"project":git_record(ROOT)},metrics=metrics,stdout=json.dumps(metrics),stderr=""); print(json.dumps(metrics,indent=2)); return 0 if audit["passed"] else 2
if __name__=="__main__": raise SystemExit(main())
