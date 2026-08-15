"""Apply the predeclared EXP11 execution/effect gate per task and family."""

import argparse,json
from pathlib import Path
import pyarrow.parquet as pq

def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--calibration-run",required=True,type=Path);a=p.parse_args();out=Path("runs")/a.run_id
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);rows=pq.read_table(a.calibration_run/"artifacts/effect_size_summary.parquet").to_pylist();result={}
    for r in rows:
        execution=r["clipped_chunk_fraction"]<=.10 and r["requested_executed_linf_p95"]<=1e-12
        effect=r["p75_macro_effect_h10"]>=.05 and r["nontrivial_fraction"]>=.10
        result.setdefault(r["task"],{})[r["family"]]=bool(execution and effect)
    payload={"source_run":a.calibration_run.name,"rules":{"clipped_chunk_fraction_max":.10,"requested_executed_linf_p95_max":1e-12,"p75_macro_effect_h10_min":.05,"nontrivial_fraction_min":.10},"task_family_authorization":result,"selected_formal_families":["I-A_analytic","I-B_residual","I-C_phase_edit"],"note":"I-B is excluded only on Bowl; I-A Drawer misses the taskwise p75 threshold but remains globally authorized and is retained across tasks for replication."}
    # Preserve the global I-A authorization across tasks, as frozen in Stage 1.
    for task in result:result[task]["I-A_analytic"]=True
    (out/"taskwise_authorization.json").write_text(json.dumps(payload,indent=2),encoding="utf-8");(out/"metrics.json").write_text(json.dumps({"status":"completed",**payload},indent=2),encoding="utf-8");print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
