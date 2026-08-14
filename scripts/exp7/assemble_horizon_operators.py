#!/usr/bin/env python
"""Assemble all EXP7 horizon operators, tests, predictor artifacts, and figures."""

from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from decision_sparse_rl.metrics.exp6 import projector_similarity, relative_discrepancy
from decision_sparse_rl.metrics.exp7 import antithetic_asymmetry, relative_error, transition_category

HORIZONS=(1,3,5,"remaining")

def dump(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True); pq.write_table(pa.Table.from_pylist(rows),path,compression="zstd")
def ci(values,rng,n=4000):
    a=np.asarray(values,float)
    if not len(a): return [None,None]
    means=np.mean(rng.choice(a,(n,len(a)),replace=True),axis=1); return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]
def mode(raw): return tuple(json.loads(raw))
def pca_sim(a,b,k):
    _,_,va=np.linalg.svd(a,full_matrices=False); _,_,vb=np.linalg.svd(b,full_matrices=False); return projector_similarity(va.T,vb.T,min(k,a.shape[1],b.shape[1]))
def bh(ps):
    p=np.asarray(ps); order=np.argsort(p); q=np.empty_like(p); running=1.
    for rank,idx in reversed(list(enumerate(order,1))): running=min(running,p[idx]*len(p)/rank); q[idx]=running
    return q.tolist()
def plot_box(frame,col,group,path,title):
    labels=sorted(frame[group].dropna().unique()); values=[frame.loc[frame[group]==x,col].dropna() for x in labels]; plt.figure(figsize=(8,4)); plt.boxplot(values,labels=labels); plt.title(title); plt.xticks(rotation=20,ha="right"); plt.tight_layout(); plt.savefig(path,dpi=160); plt.close()

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--raw-run",type=Path,required=True); p.add_argument("--zero-run",type=Path,required=True); p.add_argument("--reference-run",type=Path,required=True); p.add_argument("--geometry",type=Path,required=True); p.add_argument("--gpu-run",type=Path,required=True); p.add_argument("--output-run",type=Path,required=True); args=p.parse_args(); raw=args.raw_run.resolve(); out=args.output_run.resolve(); art=out/"artifacts"; plots=out/"plots"; art.mkdir(parents=True,exist_ok=False); plots.mkdir()
    interventions=pq.read_table(raw/"artifacts/interventions.parquet").to_pandas(); steps=pq.read_table(raw/"artifacts/per_step_effects.parquet").to_pandas(); geometry=pq.read_table(args.geometry).to_pandas(); branch=json.loads((ROOT/"experiments/exp7_contact_mode_response/manifests/branch_manifest.json").read_text()); directions=json.loads((ROOT/"experiments/exp7_contact_mode_response/manifests/direction_basis_manifest.json").read_text())["directions"]
    direction={(x["task"],x["episode"],x["branch_time"],x["radius_fraction"],x["direction_index"]):x for x in directions}; vectors={key:group.sort_values("continuation_offset") for key,group in steps.groupby("intervention_id")}; horizon_rows=[]; outcome_rows=[]
    for row in interventions.to_dict("records"):
        group=vectors[row["intervention_id"]]
        for horizon in HORIZONS:
            take=group if horizon=="remaining" else group.head(int(horizon)); vec=np.mean(np.vstack(take["signed_normalized_physical_output_vector"]),axis=0); preserved=bool(np.all(take["zero_contact_mode_json"]==take["perturbed_contact_mode_json"])); terminal=take.iloc[-1]
            horizon_rows.append({**{k:row[k] for k in ("task","episode","branch_time","radius_fraction","radius_label","direction_index","direction_role","sign","intervention_id")},"horizon":str(horizon),"signed_output_vector":vec.tolist(),"response_norm":float(np.linalg.norm(vec)),"mode_preserved_through_horizon":preserved,"reference_mode_json":terminal["zero_contact_mode_json"],"perturbed_mode_json":terminal["perturbed_contact_mode_json"],"reference_signed_gap_m":float(take.iloc[0]["zero_signed_gap_m"]),"reference_normal_velocity_mps":float(take.iloc[0]["zero_normal_relative_velocity_mps"])})
    hdf=pd.DataFrame(horizon_rows); pair_keys=["task","episode","branch_time","radius_fraction","radius_label","direction_index","direction_role","horizon"]; operators=[]; matrices=[]; outcomes=[]
    for key,group in hdf.groupby(pair_keys):
        plus=group[group.sign==1].iloc[0]; minus=group[group.sign==-1].iloc[0]; ref=mode(plus.reference_mode_json); pm=mode(plus.perturbed_mode_json); mm=mode(minus.perturbed_mode_json); category=transition_category(ref,pm,mm); both=bool(plus.mode_preserved_through_horizon and minus.mode_preserved_through_horizon)
        outcomes.append({**dict(zip(pair_keys,key)),"transition_category":category,"plus_preserved":bool(plus.mode_preserved_through_horizon),"minus_preserved":bool(minus.mode_preserved_through_horizon),"both_signs_preserved":both,"reference_mode_json":plus.reference_mode_json,"plus_mode_json":plus.perturbed_mode_json,"minus_mode_json":minus.perturbed_mode_json})
    odf=pd.DataFrame(outcomes); hdf=hdf.merge(odf[pair_keys+["transition_category","both_signs_preserved"]],on=pair_keys)
    basis=hdf[hdf.direction_role=="basis"]
    op_keys=["task","episode","branch_time","radius_fraction","radius_label","horizon"]
    for key,group in basis.groupby(op_keys):
        cols=[]; asym=[]; preserved=[]
        for di in range(7):
            dg=group[group.direction_index==di]; plus=dg[dg.sign==1].iloc[0]; minus=dg[dg.sign==-1].iloc[0]; pv=np.asarray(plus.signed_output_vector); mv=np.asarray(minus.signed_output_vector); cols.append((pv-mv)/(2*key[3])); asym.append(antithetic_asymmetry(pv,mv)); preserved.append(bool(plus.mode_preserved_through_horizon and minus.mode_preserved_through_horizon))
        J=np.column_stack(cols); singular=np.linalg.svd(J,compute_uv=False); rec={**dict(zip(op_keys,key)),"intent_to_perturb":True,"all_basis_both_signs_preserved":all(preserved),"preserved_direction_fraction":float(np.mean(preserved)),"spectral_norm":float(singular[0]),"frobenius_norm":float(np.linalg.norm(J)),"leading_share":float(singular[0]**2/np.sum(singular**2)),"effective_rank":float(np.exp(-np.sum((singular**2/np.sum(singular**2))*np.log(singular**2/np.sum(singular**2)+1e-300)))),"median_sign_asymmetry":float(np.median(asym))}; operators.append(rec); matrices.append({**dict(zip(op_keys,key)),"operator_matrix":J.tolist()})
    opdf=pd.DataFrame(operators); matrix_lookup={(x["task"],x["episode"],x["branch_time"],x["radius_fraction"],x["horizon"]):np.asarray(x["operator_matrix"]) for x in matrices}; convergence=[]
    for (task,episode,branch_time,horizon),group in opdf.groupby(["task","episode","branch_time","horizon"]):
        group=group.sort_values("radius_fraction"); rows=group.to_dict("records")
        for left,right in zip(rows[:-1],rows[1:]):
            jl=matrix_lookup[(task,episode,branch_time,left["radius_fraction"],horizon)]; jr=matrix_lookup[(task,episode,branch_time,right["radius_fraction"],horizon)]; top1=pca_sim(jl,jr,1); top2=pca_sim(jl,jr,2); spec=relative_discrepancy(left["spectral_norm"],right["spectral_norm"]); asym=max(left["median_sign_asymmetry"],right["median_sign_asymmetry"]); conditional=left["all_basis_both_signs_preserved"] and right["all_basis_both_signs_preserved"]
            convergence.append({"task":task,"episode":episode,"branch_time":branch_time,"horizon":horizon,"radius_small":left["radius_fraction"],"radius_large":right["radius_fraction"],"top1_similarity":top1,"top2_similarity":top2,"relative_spectral_discrepancy":spec,"sign_asymmetry":asym,"conditional_preserved":conditional,"passes":top1>=.8 and top2>=.75 and spec<=.2 and asym<=.25})
    cdf=pd.DataFrame(convergence); meta=geometry[["task","episode","action_index","boundary_margin_class","normalized_time","physical_progress_clipped","gripper_state","predicate","contact_mode_json","signed_gap_m","normal_relative_velocity_mps"]].rename(columns={"action_index":"branch_time"}); cdf=cdf.merge(meta,on=["task","episode","branch_time"],how="left")
    held=[]
    for _,row in hdf[(hdf.direction_role=="heldout_random") & (hdf.sign==1)].iterrows():
        minus=hdf[(hdf.task==row.task)&(hdf.episode==row.episode)&(hdf.branch_time==row.branch_time)&(hdf.radius_fraction==row.radius_fraction)&(hdf.horizon==row.horizon)&(hdf.direction_index==7)&(hdf.sign==-1)].iloc[0]; actual=(np.asarray(row.signed_output_vector)-np.asarray(minus.signed_output_vector))/(2*row.radius_fraction); J=matrix_lookup[(row.task,row.episode,row.branch_time,row.radius_fraction,row.horizon)]; d=direction[(row.task,row.episode,row.branch_time,row.radius_fraction,7)]; basis_vectors=np.column_stack([direction[(row.task,row.episode,row.branch_time,row.radius_fraction,i)]["unit_direction_scaled_coordinates"] for i in range(7)]); coeff=basis_vectors.T@np.asarray(d["unit_direction_scaled_coordinates"]); pred=J@coeff
        held.append({"task":row.task,"episode":row.episode,"branch_time":row.branch_time,"radius_fraction":row.radius_fraction,"horizon":row.horizon,"both_signs_preserved":bool(row.mode_preserved_through_horizon and minus.mode_preserved_through_horizon),"predicted_norm":float(np.linalg.norm(pred)),"actual_norm":float(np.linalg.norm(actual)),"vector_relative_error":relative_error(pred,actual)})
    held_df=pd.DataFrame(held)
    smallest=cdf[(cdf.horizon=="1")&(cdf.radius_small==min(cdf.radius_small))]; demo=smallest.groupby(["task","episode"]).agg(top1=("top1_similarity","median"),top2=("top2_similarity","median"),spec=("relative_spectral_discrepancy","median"),asym=("sign_asymmetry","median")).reset_index(); demo["passes_all"]=((demo.top1>=.8)&(demo.top2>=.75)&(demo.spec<=.2)&(demo.asym<=.25)); rng=np.random.default_rng(970031); h1_fraction=float(demo.passes_all.mean()); h1_ci=ci(demo.top1,rng)
    interior=smallest.groupby(["task","episode","boundary_margin_class"]).top1_similarity.median().reset_index(); piv=interior.pivot_table(index=["task","episode"],columns="boundary_margin_class",values="top1_similarity"); diffs=[]
    for _,r in piv.iterrows():
        near=np.nanmax([r.get("near_boundary",np.nan),r.get("ambiguous",np.nan)])
        if np.isfinite(r.get("interior",np.nan)) and np.isfinite(near): diffs.append(r["interior"]-near)
    h2_ci=ci(diffs,rng); h2_p=float((np.sum(np.asarray(diffs)<=0)+1)/(len(diffs)+1)) if diffs else 1.; h2_q=bh([h2_p])[0]
    h3_rows=[]
    hsmall=held_df[(held_df.horizon=="1")&(held_df.radius_fraction==held_df.radius_fraction.min())&(held_df.both_signs_preserved)]
    for key,g in hsmall.groupby(["task","episode"]): h3_rows.append({"task":key[0],"episode":key[1],"rho":float(spearmanr(g.predicted_norm,g.actual_norm).statistic) if len(g)>1 else 0.,"median_vector_error":float(g.vector_relative_error.median())})
    h3=pd.DataFrame(h3_rows); h3_rho=float(h3.rho.median()) if len(h3) else 0.; h3_err=float(h3.median_vector_error.median()) if len(h3) else float("inf")
    # Cross-demo matching: compare same task+mode+margin nearest neighbor against independently nearest time/progress baselines.
    cross=[]; base_ops=opdf[(opdf.horizon=="1")&(opdf.radius_fraction==min(opdf.radius_fraction))].merge(meta,on=["task","episode","branch_time"])
    for _,r in base_ops.iterrows():
        pool=base_ops[(base_ops.task==r.task)&(base_ops.episode!=r.episode)]
        same=pool[(pool.contact_mode_json==r.contact_mode_json)&(pool.boundary_margin_class==r.boundary_margin_class)]
        if same.empty or pool.empty: continue
        a=matrix_lookup[(r.task,r.episode,r.branch_time,r.radius_fraction,"1")]; sm=same.iloc[(same.normalized_time-r.normalized_time).abs().argmin()]; tm=pool.iloc[(pool.normalized_time-r.normalized_time).abs().argmin()]; pm=pool.iloc[(pool.physical_progress_clipped-r.physical_progress_clipped).abs().argmin()]
        sim=lambda x:pca_sim(a,matrix_lookup[(x.task,x.episode,x.branch_time,x.radius_fraction,"1")],1); ss,ts,ps=sim(sm),sim(tm),sim(pm); cross.append({"task":r.task,"episode":r.episode,"branch_time":r.branch_time,"same_mode_margin_top1":ss,"time_top1":ts,"progress_top1":ps,"improvement_over_better":ss-max(ts,ps)})
    crossdf=pd.DataFrame(cross); h4=float(crossdf.improvement_over_better.mean()) if len(crossdf) else -1.; h4_ci=ci(crossdf.improvement_over_better if len(crossdf) else [],rng)
    # Reference-only cross-fitted preservation predictor.
    pred=hdf[hdf.horizon=="1"].copy(); pred["target"]=pred.mode_preserved_through_horizon.astype(int); pred=pred.merge(meta,on=["task","episode","branch_time"],suffixes=("","_meta")); pred["mode_size"]=pred.reference_mode_json.map(lambda x:len(json.loads(x))); pred["gripper_code"]=pred.gripper_state.map({"negative":-1,"neutral":0,"positive":1}); pred["fold"]=pred.episode.map(lambda x:int(x.split("_")[-1])%5)
    features=["radius_fraction","sign","direction_index","reference_signed_gap_m","reference_normal_velocity_mps","normalized_time","physical_progress_clipped","mode_size","gripper_code"]; preds=np.zeros(len(pred))
    for fold in range(5):
        train=pred.fold!=fold; test=~train
        if pred.loc[train,"target"].nunique()<2: preds[test]=pred.loc[train,"target"].mean(); continue
        model_lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced",random_state=970040)); model_lr.fit(pred.loc[train,features],pred.loc[train,"target"]); preds[test]=model_lr.predict_proba(pred.loc[test,features])[:,1]
    pred["probability"]=preds; y=pred.target.to_numpy(); auroc=float(roc_auc_score(y,preds)) if len(np.unique(y))==2 else None; auprc=float(average_precision_score(y,preds)); brier=float(brier_score_loss(y,preds)); bins=np.minimum((preds*10).astype(int),9); ece=float(sum(np.mean(bins==b)*abs(np.mean(y[bins==b])-np.mean(preds[bins==b])) for b in np.unique(bins))); threshold=.5; sensitivity=float(np.mean(preds[y==1]>=threshold)) if np.any(y==1) else None; specificity=float(np.mean(preds[y==0]<threshold)) if np.any(y==0) else None
    predictor_metrics={"AUROC":auroc,"AUPRC":auprc,"Brier":brier,"ECE":ece,"threshold":threshold,"sensitivity":sensitivity,"specificity":specificity,"positive_rate":float(y.mean()),"crossfit_unit":"demonstration","features_used":features}
    h1=bool(h1_fraction>=.7 and h1_ci[0] is not None and h1_ci[0]>.65); h2=bool(diffs and h2_ci[0]>0 and h2_q<.05); h3pass=bool(h3_rho>=.65 and h3_err<=.35); h4pass=bool(h4>=.15 and h4_ci[0] is not None and h4_ci[0]>0); predictor_ready=bool(auroc is not None and auroc>=.7 and ece<=.1)
    if h1 and h3pass and h4pass: classification="within_mode_short_horizon_operator_converges"
    elif h2: classification="boundary_margin_explains_hybrid_nonsmoothness"
    elif not predictor_ready: classification="contact_modes_explanatory_but_not_predictable"
    else: classification="within_mode_nonsmoothness_persists"
    decision={"classification":classification,"H1":{"passed":h1,"demo_fraction":h1_fraction,"top1_hierarchical_ci":h1_ci},"H2":{"passed":h2,"demo_differences":len(diffs),"mean_difference":float(np.mean(diffs)) if diffs else None,"ci":h2_ci,"p":h2_p,"bh_q":h2_q},"H3":{"passed":h3pass,"demo_median_rho":h3_rho,"demo_median_vector_error":h3_err},"H4":{"passed":h4pass,"mean_improvement":h4,"ci":h4_ci},"H5":{"scheduler_ready":predictor_ready,**predictor_metrics},"analysis_unit":"demonstration","intent_to_perturb_primary":True,"conditional_preserved_reported":True}
    dump(horizon_rows,art/"horizon_operator_summary.parquet"); dump(outcomes,art/"mode_outcomes.parquet"); dump(operators,art/"operator_summary.parquet"); dump(matrices,art/"operator_matrices.parquet"); dump(cdf.to_dict("records"),art/"within_mode_convergence.parquet"); dump(interior.to_dict("records"),art/"boundary_margin_analysis.parquet"); dump(cdf.groupby(["horizon","radius_small","radius_large"],as_index=False).agg(top1_similarity=("top1_similarity","median"),top2_similarity=("top2_similarity","median"),spectral_discrepancy=("relative_spectral_discrepancy","median"),sign_asymmetry=("sign_asymmetry","median")).to_dict("records"),art/"horizon_comparison.parquet"); dump(held,art/"heldout_vector_errors.parquet"); dump(cross,art/"mode_conditioned_crossdemo.parquet"); dump(pred[["task","episode","branch_time","radius_fraction","direction_index","sign","target","fold","probability"]].to_dict("records"),art/"mode_preservation_predictor_predictions.parquet"); (art/"mode_preservation_predictor_metrics.json").write_text(json.dumps(predictor_metrics,indent=2)+"\n"); (art/"scientific_decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    raw_hashes={name:hashlib.sha256((raw/"artifacts"/name).read_bytes()).hexdigest() for name in ("interventions.parquet","per_step_effects.parquet","zero_controls.parquet")}; (art/"raw_data_lock_manifest.json").write_text(json.dumps({"raw_run":raw.name,"hashes":raw_hashes},indent=2)+"\n"); (art/"failure_examples.json").write_text((raw/"artifacts/failure_examples.json").read_text());
    for name in ("gpu_audit.json","gpu_cpu_equivalence.json"): (art/name).write_bytes((args.gpu_run/"artifacts"/name).read_bytes())
    # Required compact figures.
    repeat=pq.read_table(args.geometry.parent/"boundary_margin_calibration.parquet").to_pandas(); plot_box(repeat,"signed_gap_range_m","task",plots/"signed_gap_repeatability.png","Signed-gap repeatability")
    freq=odf.transition_category.value_counts(); plt.figure(); freq.plot.bar(); plt.tight_layout(); plt.savefig(plots/"contact_mode_frequency.png",dpi=160); plt.close()
    plot_box(meta,"signed_gap_m","task",plots/"boundary_margin_distribution.png","Boundary signed gaps")
    plot_box(cdf,"top1_similarity","radius_small",plots/"within_mode_top1_by_radius.png","Top-1 convergence")
    plot_box(cdf,"top2_similarity","radius_small",plots/"within_mode_top2_by_radius.png","Top-2 convergence")
    plot_box(cdf,"relative_spectral_discrepancy","radius_small",plots/"within_mode_spectral_discrepancy.png","Spectral discrepancy")
    plot_box(cdf,"sign_asymmetry","radius_small",plots/"within_mode_sign_asymmetry.png","Sign asymmetry")
    plot_box(cdf,"top1_similarity","boundary_margin_class",plots/"convergence_by_boundary_margin.png","Convergence by margin")
    plot_box(cdf,"top1_similarity","horizon",plots/"convergence_by_horizon.png","Convergence by horizon")
    plt.figure(); cdf.groupby("conditional_preserved").top1_similarity.mean().plot.bar(); plt.tight_layout(); plt.savefig(plots/"intent_to_perturb_vs_conditional.png",dpi=160); plt.close()
    plt.figure(); freq.plot.bar(); plt.tight_layout(); plt.savefig(plots/"mode_transition_categories.png",dpi=160); plt.close()
    plot_box(held_df,"vector_relative_error","radius_fraction",plots/"heldout_vector_error_within_mode.png","Heldout vector error")
    if len(crossdf): crossdf[["same_mode_margin_top1","time_top1","progress_top1"]].boxplot(); plt.tight_layout(); plt.savefig(plots/"crossdemo_subspace_time_vs_progress_vs_mode.png",dpi=160); plt.close()
    fpr,tpr,_=roc_curve(y,preds); pr,rc,_=precision_recall_curve(y,preds); plt.figure(); plt.plot(fpr,tpr,label="ROC"); plt.plot(rc,pr,label="PR"); plt.legend(); plt.tight_layout(); plt.savefig(plots/"mode_predictor_roc_pr.png",dpi=160); plt.close()
    plt.figure(); plt.scatter(preds,y,alpha=.05); plt.xlabel("probability"); plt.ylabel("preserved"); plt.tight_layout(); plt.savefig(plots/"mode_predictor_calibration.png",dpi=160); plt.close()
    plot_box(cdf,"top1_similarity","task",plots/"task_specific_hybrid_summary.png","Task summary")
    gpu=json.loads((art/"gpu_cpu_equivalence.json").read_text()); plt.figure(); plt.semilogy([max(x["operator_rel"],1e-20) for x in gpu["records"]],label="operator rel"); plt.semilogy([max(x["gram_rel"],1e-20) for x in gpu["records"]],label="gram rel"); plt.legend(); plt.tight_layout(); plt.savefig(plots/"gpu_cpu_equivalence.png",dpi=160); plt.close()
    metrics={"classification":classification,"raw_interventions":len(interventions),"per_step_rows":len(steps),"operator_rows":len(opdf),"mode_outcome_rows":len(odf),"plot_count":len(list(plots.glob("*.png"))),"decision":decision,"gate":{"passed":len(list(plots.glob("*.png")))==17}}; (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n"); print(json.dumps(metrics,indent=2)); return 0 if metrics["gate"]["passed"] else 2

if __name__=="__main__": raise SystemExit(main())
