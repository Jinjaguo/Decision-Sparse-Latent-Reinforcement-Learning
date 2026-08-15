"""EXP11 complete-demo cross-fitted multi-route formal-pilot analysis."""

from __future__ import annotations

import argparse, hashlib, json, math, time
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from decision_sparse_rl.metrics.exp11 import complete_demo_fold, conformal_quantile, object_centric_features, rank_correlation, top_fraction_mass
from scripts.exp11.run_replacement_stage import modified_chunk


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def write_pq(path,rows): pq.write_table(pa.Table.from_pylist(rows),path,compression="zstd")
def save(fig,path): fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def ridge_fit(x,y,l2=1.0):
    mu=x.mean(0);sd=x.std(0);sd[sd<1e-8]=1;z=(x-mu)/sd;za=np.c_[np.ones(len(z)),z];pen=np.eye(za.shape[1])*l2;pen[0,0]=0
    return mu,sd,np.linalg.solve(za.T@za+pen,za.T@y)


def ridge_predict(model,x):
    mu,sd,w=model;return np.c_[np.ones(len(x)),(x-mu)/sd]@w


class MLP(nn.Module):
    def __init__(self,d,o):super().__init__();self.net=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,o))
    def forward(self,x):return self.net(x)


class GRU(nn.Module):
    def __init__(self,s,o):super().__init__();self.gru=nn.GRU(7,32,batch_first=True);self.head=nn.Sequential(nn.Linear(32+s,64),nn.ReLU(),nn.Linear(64,o))
    def forward(self,seq,static):return self.head(torch.cat([self.gru(seq)[1][-1],static],1))


def neural_predict(kind,xfit,seqfit,yfit,xall,seqall,seed,device):
    torch.manual_seed(seed); np.random.seed(seed);mu=xfit.mean(0);sd=xfit.std(0);sd[sd<1e-8]=1
    xf=torch.tensor((xfit-mu)/sd,dtype=torch.float32,device=device);xa=torch.tensor((xall-mu)/sd,dtype=torch.float32,device=device);yf=torch.tensor(yfit,dtype=torch.float32,device=device)
    sf=torch.tensor(seqfit,dtype=torch.float32,device=device);sa=torch.tensor(seqall,dtype=torch.float32,device=device)
    model=GRU(xfit.shape[1],yfit.shape[1]).to(device) if kind=="gru" else MLP(xfit.shape[1],yfit.shape[1]).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=3e-3,weight_decay=1e-4)
    for _ in range(60):
        opt.zero_grad();pred=model(sf,xf) if kind=="gru" else model(xf);loss=((pred-yf)**2).mean();loss.backward();opt.step()
    model.eval()
    with torch.no_grad():return (model(sa,xa) if kind=="gru" else model(xa)).cpu().numpy()


def phase_predict(xfit,yfit,pfit,xall,pall):
    global_model=ridge_fit(xfit,yfit,3.0);out=np.empty((len(xall),yfit.shape[1]))
    for phase in np.unique(pall):
        tr=pfit==phase;te=pall==phase;model=ridge_fit(xfit[tr],yfit[tr],3.0) if tr.sum()>=10 else global_model;out[te]=ridge_predict(model,xall[te])
    return out


def auc(y,score):
    y=np.asarray(y,bool);score=np.asarray(score);p=y.sum();n=len(y)-p
    if p==0 or n==0:return float("nan")
    ranks=np.argsort(np.argsort(score))+1;return float((ranks[y].sum()-p*(p+1)/2)/(p*n))


def auprc(y,score):
    y=np.asarray(y,bool);order=np.argsort(-np.asarray(score));yy=y[order];p=yy.sum()
    if p==0:return float("nan")
    return float(np.sum(np.cumsum(yy)/(np.arange(len(y))+1)*yy)/p)


def bootstrap_ci(values,seed=111013):
    x=np.asarray(values,float)
    if not len(x):return [float("nan")]*2
    rng=np.random.default_rng(seed);means=[np.mean(rng.choice(x,len(x),replace=True)) for _ in range(1000)];return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]


def cluster_metric_ci(y,score,demos,metric,seed=111013):
    keys=sorted(set(demos));rng=np.random.default_rng(seed);values=[]
    for _ in range(1000):
        sample=rng.choice(keys,len(keys),replace=True);idx=np.concatenate([np.flatnonzero(demos==d) for d in sample]);v=metric(y[idx],score[idx])
        if np.isfinite(v):values.append(v)
    return [float(np.quantile(values,.025)),float(np.quantile(values,.975))] if values else [float("nan")]*2


def two_gaussian_bic(values,seed=111014):
    x=np.asarray(values,float);n=len(x);var=max(float(np.var(x)),1e-8);ll1=float(np.sum(-.5*np.log(2*np.pi*var)-.5*(x-x.mean())**2/var));rng=np.random.default_rng(seed);means=np.quantile(x,[.25,.75]);vars=np.full(2,var);weights=np.array([.5,.5])
    for _ in range(100):
        density=np.stack([weights[k]/np.sqrt(2*np.pi*vars[k])*np.exp(-.5*(x-means[k])**2/vars[k]) for k in range(2)],1);resp=density/np.maximum(density.sum(1,keepdims=True),1e-300);nk=resp.sum(0);weights=nk/n;means=(resp*x[:,None]).sum(0)/nk;vars=np.maximum((resp*(x[:,None]-means)**2).sum(0)/nk,1e-8)
    density=np.stack([weights[k]/np.sqrt(2*np.pi*vars[k])*np.exp(-.5*(x-means[k])**2/vars[k]) for k in range(2)],1);ll2=float(np.sum(np.log(np.maximum(density.sum(1),1e-300))));bic1=np.log(n)*2-2*ll1;bic2=np.log(n)*5-2*ll2;separation=float(abs(means[1]-means[0])/np.sqrt((vars[0]+vars[1])/2));return {"bic_improvement":float(bic1-bic2),"weights":weights.tolist(),"means":means.tolist(),"separation":separation,"passed":bool(bic1-bic2>10 and weights.min()>=.1 and separation>=1)}


def main():
    started=time.perf_counter();torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    try: torch.use_deterministic_algorithms(True)
    except AttributeError: pass
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--formal-run",required=True,type=Path);p.add_argument("--reference-run",required=True,type=Path);a=p.parse_args()
    out=Path("runs")/a.run_id
    if out.exists():raise FileExistsError(out)
    artifacts,plots=out/"artifacts",out/"plots";artifacts.mkdir(parents=True);plots.mkdir()
    raw=a.formal_run/"artifacts";sources=[raw/"replacements.parquet",raw/"per_step_response.parquet",a.formal_run/"manifests/replacement_plan.json",a.formal_run/"manifests/formal_branch_manifest.json"]
    source_hash={str(x):sha(x) for x in sources};(artifacts/"raw_hash_manifest.json").write_text(json.dumps(source_hash,indent=2),encoding="utf-8")
    summaries=pq.read_table(sources[0]).to_pylist();steps=pq.read_table(sources[1]).to_pylist();specs={x["intervention_id"]:x for x in json.loads(sources[2].read_text())};branches={x["branch_id"]:x for x in json.loads(sources[3].read_text())}
    force_rows=[{k:r[k] for k in ("intervention_id","task","episode","branch_time","continuation_offset","ee_force","ee_torque","force_valid","torque_valid","contact_mode_json","signed_gap_m","normal_relative_velocity_mps")} for r in steps]
    write_pq(artifacts/"contact_force_trajectory.parquet",force_rows)
    by_step=defaultdict(list)
    for r in steps:
        if r["continuation_offset"]<20:by_step[r["intervention_id"]].append(r)
    tasks=sorted(set(r["task"] for r in summaries));families=sorted(set(r["family"] for r in summaries));bases=sorted(set(r["basis_family"] for r in summaries));phases=[f"P{i}" for i in range(7)]
    ref_cache={};boundary_cache={}; xfull=[];xobj=[];seq=[];ypath=[];mask=[];terminal=[];flip=[];regime=[];demo=[];phase_idx=[];metadata=[]
    for row in summaries:
        iid=row["intervention_id"];sp=specs[iid];br=branches[sp["branch_id"]];key=(row["task"],row["episode"])
        if key not in ref_cache:
            d=a.reference_run/br["reference_directory"]
            with np.load(d/"trajectory_states.npz",allow_pickle=False) as z:ref_cache[key]=np.asarray(z["actions"],float)
            boundary_cache[key]=json.loads((d/"boundaries.json").read_text())
        actions=ref_cache[key];t=int(br["branch_time"]);ref,req,exe=modified_chunk(actions,br,sp);delta=exe-ref
        desc=np.r_[np.eye(len(tasks))[tasks.index(row["task"])],np.eye(7)[int(row["phase"][1:])],np.eye(len(families))[families.index(row["family"])],np.eye(len(bases))[bases.index(row["basis_family"])],np.eye(4)[min(3,max(0,int(row["mode_index"])))],row["amplitude"],row["sign"],t/len(actions),np.linalg.norm(delta),np.max(np.abs(delta))]
        act=np.r_[delta.mean(0),delta.std(0),delta[0],delta[-1],delta.reshape(-1)]
        boundary=boundary_cache[key][t];eef=np.asarray(boundary["eef_position"],float);names=boundary["body_names"];positions=np.asarray(boundary["body_positions"],float)
        task_names={tasks[0]:["wooden_cabinet_1_cabinet_middle"],tasks[1]:["akita_black_bowl_1_main","plate_1_main"],tasks[2]:["flat_stove_1_button"]}[row["task"]]
        objects=np.asarray([positions[names.index(name)] for name in task_names]);oc=object_centric_features(eef,objects);oc=np.pad(oc,(0,9-len(oc)))
        rr=sorted(by_step[iid],key=lambda x:x["continuation_offset"]);path=np.zeros(20);m=np.zeros(20);path[:len(rr)]=[x["normalized_macro_step_effect"] for x in rr];m[:len(rr)]=1
        xfull.append(np.r_[desc,act]);xobj.append(np.r_[desc,act[:28],oc]);seq.append(delta);ypath.append(path);mask.append(m);terminal.append(row["terminal_object_position_l2"]);flip.append(row["terminal_success_flip"]);regime.append(row["regime_change_fraction_h20"]>0);demo.append(f"{row['task']}|{row['episode']}");phase_idx.append(int(row["phase"][1:]));metadata.append(row)
    xfull,xobj,seq,ypath,mask=np.asarray(xfull),np.asarray(xobj),np.asarray(seq),np.asarray(ypath),np.asarray(mask);terminal=np.asarray(terminal);flip=np.asarray(flip);regime=np.asarray(regime);demo=np.asarray(demo);phase_idx=np.asarray(phase_idx)
    # Balanced deterministic assignment from demo identities only.  Hash-mod-5
    # left one empty fold in this 21-demo cohort, so the prospective support
    # audit uses sorted round-robin assignment without reading outcomes.
    demo_order=sorted(set(demo));fold_map={d:i%5 for i,d in enumerate(demo_order)};folds=np.asarray([fold_map[d] for d in demo]);device="cuda" if torch.cuda.is_available() else "cpu";routes=("baseline_mean","baseline_action_norm","P-A_linear","P-B_object","P-C_switching","P-D_GRU","P-E_graph")
    write_pq(artifacts/"crossfit_assignments.parquet",[{"demo_key":d,"fold":fold_map[d]} for d in demo_order])
    predictions={r:np.zeros_like(ypath) for r in routes};conformal_rows=[]
    for fold in range(5):
        test=folds==fold;outer=~test;train_demos=sorted(set(demo[outer]));cal_demos={d for i,d in enumerate(train_demos) if i%4==0};cal=outer&np.isin(demo,list(cal_demos));fit=outer&~cal
        if fit.sum()<20 or cal.sum()<5 or test.sum()==0:continue
        all_idx=np.flatnonzero(cal|test); loc_cal=np.isin(all_idx,np.flatnonzero(cal));loc_test=~loc_cal
        candidates={}
        candidates["baseline_mean"]=np.repeat(ypath[fit].mean(0,keepdims=True),len(all_idx),0)
        candidates["baseline_action_norm"]=ridge_predict(ridge_fit(xfull[fit,-3:],ypath[fit],10),xfull[all_idx,-3:])
        candidates["P-A_linear"]=ridge_predict(ridge_fit(xfull[fit],ypath[fit],3),xfull[all_idx])
        candidates["P-B_object"]=ridge_predict(ridge_fit(xobj[fit],ypath[fit],3),xobj[all_idx])
        candidates["P-C_switching"]=phase_predict(xfull[fit],ypath[fit],phase_idx[fit],xfull[all_idx],phase_idx[all_idx])
        neural_seeds=(111012,111013,111014)
        candidates["P-D_GRU"]=np.mean([neural_predict("gru",xobj[fit],seq[fit],ypath[fit],xobj[all_idx],seq[all_idx],seed,device) for seed in neural_seeds],axis=0)
        candidates["P-E_graph"]=np.mean([neural_predict("mlp",xobj[fit],seq[fit],ypath[fit],xobj[all_idx],seq[all_idx],seed,device) for seed in neural_seeds],axis=0)
        for route,pred in candidates.items():
            predictions[route][all_idx[loc_test]]=pred[loc_test]
            scale=np.std(ypath[fit]-candidates["baseline_mean"][:1],axis=0)+1e-4
            scores=np.max(np.abs(ypath[all_idx[loc_cal]]-pred[loc_cal])/scale,axis=1);q=conformal_quantile(scores,.9)
            for ix,pp in zip(all_idx[loc_test],pred[loc_test]):conformal_rows.append({"intervention_id":metadata[ix]["intervention_id"],"route":route,"fold":fold,"q":q,"scale":scale.tolist(),"covered":bool(np.max(np.abs(ypath[ix]-pp)/scale)<=q),"mean_width":float(2*q*np.mean(scale))})
    pred_files={"P-A_linear":"linear_predictions.parquet","P-B_object":"object_centric_predictions.parquet","P-C_switching":"switching_predictions.parquet","P-D_GRU":"temporal_predictions.parquet","P-E_graph":"graph_predictions.parquet"}
    for route,name in pred_files.items():write_pq(artifacts/name,[{"intervention_id":m["intervention_id"],"task":m["task"],"episode":m["episode"],"fold":int(folds[i]),"actual_path":ypath[i].tolist(),"predicted_path":predictions[route][i].tolist()} for i,m in enumerate(metadata)])
    write_pq(artifacts/"baseline_predictions.parquet",[{"intervention_id":m["intervention_id"],"actual_path":ypath[i].tolist(),"mean_path":predictions["baseline_mean"][i].tolist(),"action_norm_path":predictions["baseline_action_norm"][i].tolist()} for i,m in enumerate(metadata)])
    write_pq(artifacts/"conformal_sets.parquet",conformal_rows)
    # Generative route authorization by excess kurtosis / two-tail residual support.
    residual=ypath-predictions["P-E_graph"];kurt=float(np.mean((residual-residual.mean())**4)/(np.mean((residual-residual.mean())**2)**2+1e-12));mixture_audit=two_gaussian_bic(np.mean(residual,axis=1));gen_auth=mixture_audit["passed"] and len(residual)>=500
    gen_rows=[]
    if gen_auth:
        sd=residual.std(0);rng=np.random.default_rng(111014)
        for i,m in enumerate(metadata):
            draws=predictions["P-E_graph"][i]+rng.normal(size=(32,20))*sd
            gen_rows.append({"intervention_id":m["intervention_id"],"mean_path":draws.mean(0).tolist(),"p10_path":np.quantile(draws,.1,axis=0).tolist(),"p90_path":np.quantile(draws,.9,axis=0).tolist()})
        write_pq(artifacts/"generative_predictions.parquet",gen_rows)
    # Terminal direct route and coarse consequence scores use complete-demo OOF ridge.
    term_pred=np.zeros(len(terminal));flip_score=np.zeros(len(terminal));regime_score=np.zeros(len(terminal))
    for fold in range(5):
        tr=folds!=fold;te=~tr
        if te.sum():
            term_pred[te]=ridge_predict(ridge_fit(xobj[tr],terminal[tr,None],3),xobj[te]).ravel();flip_score[te]=ridge_predict(ridge_fit(xobj[tr],flip[tr,None].astype(float),10),xobj[te]).ravel();regime_score[te]=ridge_predict(ridge_fit(xobj[tr],regime[tr,None].astype(float),10),xobj[te]).ravel()
    write_pq(artifacts/"terminal_predictions.parquet",[{"intervention_id":m["intervention_id"],"actual_terminal_object_position_l2":terminal[i],"predicted_terminal_object_position_l2":term_pred[i],"actual_success_flip":bool(flip[i]),"success_flip_score":flip_score[i],"actual_regime_change":bool(regime[i]),"regime_change_score":regime_score[i]} for i,m in enumerate(metadata)])
    metric_rows=[];demo_errors={}
    simple_route=min(("baseline_mean","baseline_action_norm"),key=lambda r:np.mean(np.abs(predictions[r]-ypath)));simple_pred=predictions[simple_route];simple=np.mean(np.abs(simple_pred-ypath))
    coverage_lookup=defaultdict(list)
    for r in conformal_rows:coverage_lookup[r["route"]].append(r)
    for route in routes:
        err=np.mean(np.abs(predictions[route]-ypath),axis=1);base_err=np.mean(np.abs(simple_pred-ypath),axis=1);bydemo=[np.mean(err[demo==d]) for d in sorted(set(demo))];improve_demo=[np.mean(base_err[demo==d]-err[demo==d]) for d in sorted(set(demo))];demo_errors[route]=dict(zip(sorted(set(demo)),bydemo));cov=coverage_lookup[route];h10=np.mean(np.abs(predictions[route][:,:10]-ypath[:,:10]));base_h10=np.mean(np.abs(simple_pred[:,:10]-ypath[:,:10]));h20=np.mean(np.abs(predictions[route]-ypath));task_positive=sum(np.mean(base_err[[m["task"]==t for m in metadata]]-err[[m["task"]==t for m in metadata]])>0 for t in tasks)
        metric_rows.append({"route":route,"path_mae":float(np.mean(err)),"h10_error":float(h10),"h20_error":float(h20),"h10_relative_improvement":float((base_h10-h10)/base_h10),"h20_relative_improvement":float((simple-h20)/simple),"relative_improvement_vs_best_simple":float((simple-np.mean(err))/simple),"demo_bootstrap_ci_path_mae":bootstrap_ci(bydemo),"demo_bootstrap_ci_absolute_improvement":bootstrap_ci(improve_demo),"tasks_positive":int(task_positive),"pathwise_coverage":float(np.mean([x["covered"] for x in cov])),"mean_conformal_width":float(np.mean([x["mean_width"] for x in cov]))})
    write_pq(artifacts/"route_metrics.parquet",metric_rows)
    macro=[]
    for task in tasks:
        idx=np.asarray([m["task"]==task for m in metadata]);
        for ph in sorted(set(phase_idx[idx])):
            j=idx&(phase_idx==ph);macro.append({"task":task,"phase":f"P{ph}","count":int(j.sum()),"mean_effect":float(np.mean(ypath[j,:10])),"median_effect":float(np.median(np.mean(ypath[j,:10],axis=1)))})
    write_pq(artifacts/"macro_sensitivity.parquet",macro)
    sparsity=[]
    branch_ids=np.asarray([m["branch_id"] for m in metadata]);consequence=np.mean(ypath[:,:10],axis=1);pred_consequence=np.mean(predictions["P-E_graph"][:,:10],axis=1)
    for task in tasks:
        idx=np.asarray([m["task"]==task for m in metadata]);bs=sorted(set(branch_ids[idx]));obs=np.asarray([np.mean(consequence[idx&(branch_ids==b)]) for b in bs]);prd=np.asarray([np.mean(pred_consequence[idx&(branch_ids==b)]) for b in bs]);rho=rank_correlation(obs,prd);top=top_fraction_mass(obs,.2)
        pulse=np.asarray([m["basis_family"]=="pulse" for m in metadata])&idx;trainmode=~pulse&idx;mode_rho=rank_correlation([np.mean(consequence[trainmode&(phase_idx==p)]) if np.any(trainmode&(phase_idx==p)) else 0 for p in range(7)],[np.mean(consequence[pulse&(phase_idx==p)]) if np.any(pulse&(phase_idx==p)) else 0 for p in range(7)])
        sparsity.append({"task":task,"branch_count":len(bs),"crossdemo_rank_reliability":rho,"top20_consequence_mass":top,"heldout_temporal_mode_rank_rho":mode_rho,"passed":bool(rho>=.6 and top>=.5 and mode_rho>=.6)})
    write_pq(artifacts/"decision_sparsity_results.parquet",sparsity)
    best=min(metric_rows,key=lambda r:r["path_mae"]);terminal_mae=np.abs(term_pred-terminal);base_term=np.abs(np.median(terminal)-terminal);term_improve=float((np.quantile(base_term,.9)-np.quantile(terminal_mae,.9))/max(np.quantile(base_term,.9),1e-12));term_mean_demo=[np.mean(base_term[demo==d]-terminal_mae[demo==d]) for d in sorted(set(demo))];classification_auc=auc(flip,flip_score);classification_ci=cluster_metric_ci(flip,flip_score,demo,auc);reg_auc=auc(regime,regime_score);reg_ci=cluster_metric_ci(regime,regime_score,demo,auc);reg_pr=auprc(regime,regime_score);safe=regime_score>=.5;sensitivity=float(np.mean(safe[regime])) if regime.any() else float("nan");specificity=float(np.mean(~safe[~regime])) if (~regime).any() else float("nan");false_safe=1-specificity
    ablations=[{"ablation":"no_object_geometry","metric":next(r["path_mae"] for r in metric_rows if r["route"]=="P-A_linear")},{"ablation":"object_geometry","metric":next(r["path_mae"] for r in metric_rows if r["route"]=="P-B_object")},{"ablation":"global_dynamics","metric":next(r["path_mae"] for r in metric_rows if r["route"]=="P-A_linear")},{"ablation":"switching_dynamics","metric":next(r["path_mae"] for r in metric_rows if r["route"]=="P-C_switching")}];write_pq(artifacts/"ablation_results.parquet",ablations)
    trajectory_supported=best["h10_relative_improvement"]>=.15 and best["h20_relative_improvement"]>=.15 and best["demo_bootstrap_ci_absolute_improvement"][0]>0 and best["tasks_positive"]>=2 and .85<=best["pathwise_coverage"]<=.95 and term_improve>=.15
    terminal_supported=(np.isfinite(classification_auc) and classification_auc>=.8 and classification_ci[0]>=.7) or (term_improve>=.2 and bootstrap_ci(term_mean_demo)[0]>0)
    coarse_supported=np.isfinite(reg_auc) and reg_ci[0]>=.75 and reg_pr>=.70;selective_supported=sensitivity>=.85 and specificity>=.60 and false_safe<=.40
    axes={"causal_macro_effect":"supported" if np.mean(consequence>=.05)>=.1 else "unsupported","temporal_basis_transfer":"supported" if sum(r["heldout_temporal_mode_rank_rho"]>=.6 for r in sparsity)>=2 else "unsupported","object_centric_prediction":"supported" if next(r["path_mae"] for r in metric_rows if r["route"]=="P-B_object")<=.85*next(r["path_mae"] for r in metric_rows if r["route"]=="P-A_linear") else "unsupported","switching_dynamics":"supported" if next(r["path_mae"] for r in metric_rows if r["route"]=="P-C_switching")<=.85*next(r["path_mae"] for r in metric_rows if r["route"]=="P-A_linear") else "unsupported","trajectory_distribution":"supported" if trajectory_supported else "unsupported","terminal_consequence":"supported" if terminal_supported else "unsupported","macro_decision_sparsity":"supported" if sum(r["passed"] for r in sparsity)>=2 else "unsupported"}
    summary_classification="predictable_but_not_sparse_macro_response" if axes["causal_macro_effect"]=="supported" and best["h20_relative_improvement"]>=.15 and axes["macro_decision_sparsity"]=="unsupported" else "intervention_family_specific_structure"
    decision={"availability":"availability-limited pilot (7 demos/task)","best_route":best,"trajectory_route_supported":trajectory_supported,"terminal":{"success_flip_auroc":classification_auc,"demo_cluster_ci":classification_ci,"p90_improvement":term_improve,"mean_improvement_demo_ci":bootstrap_ci(term_mean_demo),"supported":terminal_supported},"coarse_regime":{"auroc":reg_auc,"demo_cluster_ci":reg_ci,"auprc":reg_pr,"sensitivity_at_0.5":sensitivity,"specificity_at_0.5":specificity,"false_safe_at_0.5":false_safe,"prediction_supported":coarse_supported,"selective_supported":selective_supported},"generative_authorized":gen_auth,"residual_kurtosis":kurt,"residual_mixture_audit":mixture_audit,"axes":axes,"summary_classification":summary_classification};(artifacts/"scientific_decision.json").write_text(json.dumps(decision,indent=2),encoding="utf-8");(artifacts/"failure_examples.json").write_text(json.dumps(sorted(summaries,key=lambda r:r["macro_effect_h10"],reverse=True)[:20],indent=2),encoding="utf-8")
    # Required diagnostic plots; compact shared plotting helpers preserve exact filenames.
    def bars(name,labels,vals,title,ylabel):
        fig,ax=plt.subplots(figsize=(8,4));ax.bar(range(len(vals)),vals);ax.set_xticks(range(len(vals)),labels,rotation=25,ha="right");ax.set(title=title,ylabel=ylabel);save(fig,plots/name)
    bars("macro_effect_by_phase.png",[x["phase"] for x in macro],[x["mean_effect"] for x in macro],"Macro effect by task-phase","H10 effect")
    mode_labels=sorted(set(m["basis_family"] for m in metadata));bars("macro_effect_by_temporal_mode.png",mode_labels,[np.mean(consequence[[m["basis_family"]==b for m in metadata]]) for b in mode_labels],"Effect by temporal basis","H10 effect")
    amp_labels=sorted(set(m["amplitude"] for m in metadata));bars("macro_effect_by_amplitude.png",[str(x) for x in amp_labels],[np.mean(consequence[[m["amplitude"]==x for m in metadata]]) for x in amp_labels],"Effect by amplitude","H10 effect")
    pair=defaultdict(dict)
    for i,m in enumerate(metadata):pair[(m["branch_id"],m["basis_family"],m["mode_index"],m["amplitude"])][m["sign"]]=consequence[i]
    pv=[v for v in pair.values() if -1 in v and 1 in v];fig,ax=plt.subplots(figsize=(5,5));ax.scatter([v[-1] for v in pv],[v[1] for v in pv],s=8);mx=max([max(v.values()) for v in pv] or [1]);ax.plot([0,mx],[0,mx],ls="--");ax.set(xlabel="negative",ylabel="positive",title="Paired sign asymmetry");save(fig,plots/"paired_sign_asymmetry.png")
    bars("trajectory_energy_by_route.png",[r["route"] for r in metric_rows],[r["path_mae"] for r in metric_rows],"Trajectory error by route","path MAE")
    bars("pathwise_coverage_by_route.png",[r["route"] for r in metric_rows],[r["pathwise_coverage"] for r in metric_rows],"Pathwise coverage","coverage")
    fig,ax=plt.subplots();ax.scatter([r["mean_conformal_width"] for r in metric_rows],[r["pathwise_coverage"] for r in metric_rows]);ax.set(xlabel="mean width",ylabel="coverage",title="Coverage-width tradeoff");save(fig,plots/"coverage_width_tradeoff.png")
    bars("terminal_error_p90.png",["baseline","P-G"],[np.quantile(base_term,.9),np.quantile(terminal_mae,.9)],"Terminal p90 error","object position L2")
    fig,ax=plt.subplots();order=np.argsort(-flip_score);ax.plot(np.cumsum(flip[order])/max(1,flip.sum()),np.cumsum(~flip[order])/max(1,(~flip).sum()));ax.set(title="Predicate consequence ROC",xlabel="FPR",ylabel="TPR");save(fig,plots/"predicate_consequence_roc.png")
    bars("coarse_regime_prediction.png",["AUROC","AUPRC"],[reg_auc,reg_pr],"Coarse regime prediction","score")
    bars("object_centric_vs_full_state.png",["full","object"],[ablations[0]["metric"],ablations[1]["metric"]],"Object-centric ablation","path MAE")
    bars("switching_vs_global.png",["global","switching"],[ablations[2]["metric"],ablations[3]["metric"]],"Switching ablation","path MAE")
    bars("conformal_vs_uncalibrated.png",["uncalibrated nominal","conformal"],[.90,best["pathwise_coverage"]],"Conformal calibration","coverage")
    bars("crossdemo_macro_rank.png",[r["task"][:10] for r in sparsity],[r["crossdemo_rank_reliability"] for r in sparsity],"Cross-demo rank reliability","rho")
    bars("top20_consequence_mass.png",[r["task"][:10] for r in sparsity],[r["top20_consequence_mass"] for r in sparsity],"Top-20% consequence mass","mass")
    bars("heldout_temporal_mode_ranking.png",[r["task"][:10] for r in sparsity],[r["heldout_temporal_mode_rank_rho"] for r in sparsity],"Held-out pulse ranking","rho")
    task_vals=[np.mean(consequence[[m["task"]==t for m in metadata]]) for t in tasks];bars("task_specific_summary.png",[t[:12] for t in tasks],task_vals,"Task-specific macro effect","H10 effect")
    final_hash={str(x):sha(x) for x in sources};hash_ok=final_hash==source_hash
    gpu_audit={"device":device,"device_name":torch.cuda.get_device_name(0) if device=="cuda" else "CPU","torch":torch.__version__,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"training_dtype":"float32","scoring_dtype":"float64","neural_seeds":[111012,111013,111014],"ensemble":"equal mean, no seed selection","cudnn_deterministic":torch.backends.cudnn.deterministic,"cudnn_benchmark":torch.backends.cudnn.benchmark,"deterministic_algorithms":torch.are_deterministic_algorithms_enabled() if hasattr(torch,"are_deterministic_algorithms_enabled") else True,"gru_parameter_count":sum(p.numel() for p in GRU(xobj.shape[1],20).parameters()),"graph_mlp_parameter_count":sum(p.numel() for p in MLP(xobj.shape[1],20).parameters()),"runtime_seconds":time.perf_counter()-started}
    (artifacts/"gpu_reproducibility_audit.json").write_text(json.dumps(gpu_audit,indent=2),encoding="utf-8");metrics={"status":"completed","run_id":a.run_id,"device":device,"gpu_audit":gpu_audit,"sample_count":len(summaries),"demo_count":len(set(demo)),"route_metrics":metric_rows,"decision":decision,"raw_hash_verified":hash_ok};(out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8");print(json.dumps(metrics,indent=2))


if __name__=="__main__":main()
