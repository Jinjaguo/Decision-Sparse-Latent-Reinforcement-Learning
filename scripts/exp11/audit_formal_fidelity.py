"""Re-audit EXP11 action fidelity using the frozen unsaturated-channel rule."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow as pa,pyarrow.parquet as pq

def main():
 p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--formal-run",required=True,type=Path);a=p.parse_args();out=Path("runs")/a.run_id
 if out.exists():raise FileExistsError(out)
 art=out/"artifacts";art.mkdir(parents=True);steps=pq.read_table(a.formal_run/"artifacts/per_step_response.parquet",columns=["intervention_id","task","family","requested_action","executed_action","clip_flags"]).to_pylist();summ=pq.read_table(a.formal_run/"artifacts/replacements.parquet").to_pylist();mismatch=defaultdict(list)
 for r in steps:
  req=np.asarray(r["requested_action"]);exe=np.asarray(r["executed_action"]);clip=np.asarray(r["clip_flags"],bool);value=0. if np.all(clip) else float(np.max(np.abs(req[~clip]-exe[~clip])));mismatch[(r["task"],r["family"])].append(value)
 rows=[]
 for key,rr in _group(summ,lambda x:(x["task"],x["family"])).items():
  mm=np.asarray(mismatch[key]);clip_fraction=float(np.mean([r["clipped_chunk"] for r in rr]));p95=float(np.quantile(mm,.95));rows.append({"task":key[0],"family":key[1],"count":len(rr),"clipped_chunk_fraction":clip_fraction,"unsaturated_requested_executed_linf_p95":p95,"execution_valid":bool(clip_fraction<=.10 and p95<1e-7),"original_summary_included_saturated_difference":True})
 pq.write_table(pa.Table.from_pylist(rows),art/"formal_action_fidelity_reaudit.parquet",compression="zstd");payload={"status":"completed","source_run":a.formal_run.name,"rule":{"clipped_chunk_fraction_max":.10,"unsaturated_requested_executed_linf_p95_max":1e-7},"rows":rows};(out/"metrics.json").write_text(json.dumps(payload,indent=2),encoding="utf-8");print(json.dumps(payload,indent=2))
def _group(rows,key):
 d=defaultdict(list)
 for r in rows:d[key(r)].append(r)
 return d
if __name__=="__main__":main()
