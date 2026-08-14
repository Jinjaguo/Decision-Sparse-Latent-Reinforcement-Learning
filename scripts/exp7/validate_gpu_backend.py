#!/usr/bin/env python
"""Scale-aware float64 CPU/GPU equivalence gate frozen for EXP7."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]


def within(cpu,gpu,atol,rtol):
    error=float(np.max(np.abs(cpu-gpu))); scale=float(max(np.max(np.abs(cpu)),np.max(np.abs(gpu)),1e-300)); return error,error/scale,error <= atol+rtol*scale


def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",type=Path,required=True); args=p.parse_args(); out=args.run_dir/"artifacts"; out.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; EXP7 forbids automatic fallback")
    rng=np.random.default_rng(970010); records=[]; torch.cuda.set_device(0)
    for task_dim in (35,41,47):
        for radius in (.0003125,.000625,.00125):
            plus=rng.normal(size=(8,task_dim)); minus=rng.normal(size=(8,task_dim)); cpu=(plus[:7]-minus[:7]).T/(2*radius); gpu=((torch.tensor(plus[:7],device="cuda",dtype=torch.float64)-torch.tensor(minus[:7],device="cuda",dtype=torch.float64)).T/(2*radius))
            gram_cpu=cpu.T@cpu; gram_gpu=(gpu.T@gpu).cpu().numpy(); eval_cpu=np.linalg.eigvalsh(gram_cpu); eval_gpu=torch.linalg.eigvalsh(gpu.T@gpu).cpu().numpy()
            op=within(cpu,gpu.cpu().numpy(),1e-11,1e-9); gram=within(gram_cpu,gram_gpu,1e-11,1e-9); spectrum=within(eval_cpu,eval_gpu,1e-10,1e-8)
            records.append({"task_output_dim":task_dim,"radius":radius,"operator_abs":op[0],"operator_rel":op[1],"operator_pass":op[2],"gram_abs":gram[0],"gram_rel":gram[1],"gram_pass":gram[2],"spectrum_abs":spectrum[0],"spectrum_rel":spectrum[1],"spectrum_pass":spectrum[2]})
    audit={"schema_version":1,"gpu_name":torch.cuda.get_device_name(0),"torch":torch.__version__,"cuda":torch.version.cuda,"dtype":"float64","scale_aware_rule":"abs_error <= atol + rtol * max_abs_scale","records":records,"passed":all(x[k] for x in records for k in ("operator_pass","gram_pass","spectrum_pass")),"thresholds":{"operator":{"atol":1e-11,"rtol":1e-9},"gram":{"atol":1e-11,"rtol":1e-9},"spectrum":{"atol":1e-10,"rtol":1e-8}}}
    (out/"gpu_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); (out/"gpu_cpu_equivalence.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); (args.run_dir/"metrics.json").write_text(json.dumps({"gate":{"passed":audit["passed"]},"gpu":audit["gpu_name"],"cases":len(records)},indent=2)+"\n"); print(json.dumps(audit,indent=2)); return 0 if audit["passed"] else 2


if __name__=="__main__": raise SystemExit(main())
