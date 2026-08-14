#!/usr/bin/env python
"""Run locked EXP5 operator, linearity, state-alignment, and decision analyses."""
from __future__ import annotations
import argparse, hashlib, json, math, shlex, shutil, sys, time
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from decision_sparse_rl.logging.run_directory import create_run_directory, write_json, write_run_record
from decision_sparse_rl.metrics.exp5 import central_operator, operator_geometry, projector, projector_similarity, mahalanobis_cost, monotone_match, bh_fdr
from decision_sparse_rl.utils.environment_audit import git_record
TASKS=['open_the_middle_drawer_of_the_cabinet','put_the_bowl_on_the_plate','turn_on_the_stove']; SHORT={TASKS[0]:'Drawer',TASKS[1]:'Bowl',TASKS[2]:'Stove'}
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def rho(a,b)->float:
 if len(a)<3 or np.std(a)==0 or np.std(b)==0:return 0.0
 v=float(spearmanr(a,b).statistic); return v if np.isfinite(v) else 0.0
def pqwrite(rows,path):
 if not rows: raise RuntimeError(f'empty {path}')
 pq.write_table(pa.Table.from_pylist(rows),path,compression='zstd')
def pair(rows,role='basis'):
 d={}
 for x in rows:
  if x['direction_role']==role:d[(x['direction_index'],x['sign'])]=x
 return d
def plotbox(df,col,group,path,title,ylabel=None):
 fig,ax=plt.subplots(figsize=(8,5)); keys=list(dict.fromkeys(df[group])); vals=[df.loc[df[group]==k,col].dropna().values for k in keys]; ax.boxplot(vals,labels=[SHORT.get(k,k) for k in keys],showfliers=False); ax.set_title(title); ax.set_ylabel(ylabel or col); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)
def main()->int:
 p=argparse.ArgumentParser(); p.add_argument('--run-id',required=True); p.add_argument('--raw-run',type=Path,required=True); p.add_argument('--development-run',type=Path,required=True); p.add_argument('--gpu-run',type=Path,required=True); p.add_argument('--run-root',type=Path,default=ROOT/'runs'); p.add_argument('--manifest-dir',type=Path,default=ROOT/'experiments/exp5_state_conditioned_anisotropic/manifests'); a=p.parse_args(); run=create_run_directory(a.run_root,a.run_id); art=run/'artifacts'; plots=run/'plots'; plots.mkdir(); started=time.perf_counter(); raw=a.raw_run.resolve(); lock=json.loads((raw/'artifacts/raw_hash_manifest.json').read_text())
 for n,h in lock['sha256'].items():
  if sha(raw/'artifacts'/n)!=h: raise RuntimeError(f'raw lock mismatch {n}')
 rows=pq.read_table(raw/'artifacts/interventions.parquet').to_pylist(); groups=defaultdict(list)
 for x in rows:groups[(x['task'],x['episode'],x['branch_time'],x['radius_label'])].append(x)
 scalar=[]; ops=[]; held=[]; matrices=[]
 for key,g in groups.items():
  task,ep,bt,rad=key; d=pair(g); basis=[]; sj=[]; asym=[]
  for j in range(7):
   plus,minus=d[(j,1)],d[(j,-1)]; basis.append(np.asarray(plus['unit_direction_scaled_coordinates'])); sj.append((plus['primary_remaining_horizon_mean']+minus['primary_remaining_horizon_mean'])/2); asym.append(abs(plus['primary_remaining_horizon_mean']-minus['primary_remaining_horizon_mean'])/(plus['primary_remaining_horizon_mean']+minus['primary_remaining_horizon_mean']+1e-15))
  plus=np.asarray([d[(j,1)]['signed_output_remaining_horizon_mean'] for j in range(7)]); minus=np.asarray([d[(j,-1)]['signed_output_remaining_horizon_mean'] for j in range(7)]); radius=float(g[0]['radius_fraction']); jmat=central_operator(plus,minus,radius)
  tg=torch.as_tensor(jmat,device='cuda',dtype=torch.float64); gram=(tg.T@tg); evals,evecs=torch.linalg.eigh(gram); order=torch.argsort(evals,descending=True); evals=evals[order].cpu().numpy(); rv=evecs[:,order].cpu().numpy(); sv=np.sqrt(np.maximum(evals,0)); total=float(np.sum(evals)); probs=evals/total if total else np.zeros_like(evals); er=float(np.exp(-np.sum(probs[probs>0]*np.log(probs[probs>0])))) if total else 0.; bmat=np.asarray(basis).T; phys=bmat@rv; p1=projector(phys,1); p2=projector(phys,2)
  random=pair(g,'heldout_random'); hp,hm=random[(7,1)],random[(7,-1)]; actual=(np.asarray(hp['signed_output_remaining_horizon_mean'])-np.asarray(hm['signed_output_remaining_horizon_mean']))/(2*radius); u=np.asarray(hp['unit_direction_scaled_coordinates']); coef=bmat.T@u; pred=jmat@coef; rel=float(np.linalg.norm(pred-actual)/(np.linalg.norm(actual)+1e-15))
  base={'task':task,'episode':ep,'branch_time':bt,'radius_label':rad,'radius_fraction':radius,'state_prototype_id':g[0]['state_prototype_id'],'crossfit_fold':g[0]['crossfit_fold']}; terminal=max(x['terminal_object_position_l2']+x['terminal_object_orientation_geodesic_mean'] for x in g)
  scalar.append({**base,'S_RMS':float(np.sqrt(np.mean(np.square(sj)))),'basis_scalar_mean':float(np.mean(sj)),'sign_asymmetry':float(np.mean(asym)),'terminal_consequence':terminal,'any_success_flip':any(x['success_flip'] for x in g)})
  ops.append({**base,'output_dimension':jmat.shape[0],'spectral_norm':float(sv[0]),'frobenius_norm':float(np.linalg.norm(jmat)),'leading_eigenvalue_share':float(probs[0]),'effective_rank':er,'condition_number':float(sv[0]/sv[-1]) if sv[-1]>1e-12 else None,'singular_values':sv.tolist(),'top1_physical_direction':phys[:,0].tolist(),'top2_physical_directions':phys[:,:2].reshape(-1).tolist(),'terminal_consequence':terminal,'any_success_flip':any(x['success_flip'] for x in g)})
  held.append({**base,'actual_operator_response_norm':float(np.linalg.norm(actual)),'predicted_operator_response_norm':float(np.linalg.norm(pred)),'relative_prediction_error':rel,'actual_scalar_effect':float((hp['primary_remaining_horizon_mean']+hm['primary_remaining_horizon_mean'])/2),'predicted_scalar_effect':float(np.linalg.norm(pred))})
  matrices.append((key,jmat,gram.cpu().numpy(),phys))
 sdf=pd.DataFrame(scalar); odf=pd.DataFrame(ops); hdf=pd.DataFrame(held); pqwrite(scalar,art/'scalar_branch_summary.parquet'); pqwrite(ops,art/'operator_summary.parquet'); pqwrite(held,art/'heldout_direction_prediction.parquet')
 maxdim=max(x[1].shape[0] for x in matrices); arr=np.full((len(matrices),maxdim,7),np.nan); grams=np.empty((len(matrices),7,7));
 for i,(_,m,g,_) in enumerate(matrices):arr[i,:m.shape[0]]=m; grams[i]=g
 np.savez_compressed(art/'operator_matrices.npz',operators=arr,grams=grams,keys=np.asarray(['|'.join(map(str,x[0])) for x in matrices]))
 opmap={(x['task'],x['episode'],x['branch_time'],x['radius_label']):x for x in ops}; scmap={(x['task'],x['episode'],x['branch_time'],x['radius_label']):x for x in scalar}; lin=[]
 for (task,ep,bt,rad),small in opmap.items():
  if rad!='small':continue
  main=opmap[(task,ep,bt,'main')]; v1=np.asarray(small['top1_physical_direction'])[:,None]; w1=np.asarray(main['top1_physical_direction'])[:,None]; v2=np.asarray(small['top2_physical_directions']).reshape(7,2); w2=np.asarray(main['top2_physical_directions']).reshape(7,2); sim1=projector_similarity(projector(v1,1),projector(w1,1),1); sim2=projector_similarity(projector(v2,2),projector(w2,2),2); disc=abs(small['spectral_norm']-main['spectral_norm'])/(.5*(small['spectral_norm']+main['spectral_norm'])+1e-15); asym=max(scmap[(task,ep,bt,'small')]['sign_asymmetry'],scmap[(task,ep,bt,'main')]['sign_asymmetry']); pred=max(next(x['relative_prediction_error'] for x in held if (x['task'],x['episode'],x['branch_time'],x['radius_label'])==(task,ep,bt,'small')),next(x['relative_prediction_error'] for x in held if (x['task'],x['episode'],x['branch_time'],x['radius_label'])==(task,ep,bt,'main'))); passed=sim1>=.7 and disc<=.35 and asym<=.4 and pred<=.5; row={'task':task,'episode':ep,'branch_time':bt,'comparison':'small_vs_main','top1_similarity':sim1,'top2_similarity':sim2,'relative_spectral_discrepancy':disc,'maximum_sign_asymmetry':asym,'maximum_heldout_relative_error':pred,'linearity_passed':passed}; lin.append(row)
  large=opmap.get((task,ep,bt,'large'))
  if large:
   lv1=np.asarray(large['top1_physical_direction'])[:,None]; lv2=np.asarray(large['top2_physical_directions']).reshape(7,2); lin.append({'task':task,'episode':ep,'branch_time':bt,'comparison':'main_vs_large','top1_similarity':projector_similarity(projector(w1,1),projector(lv1,1),1),'top2_similarity':projector_similarity(projector(w2,2),projector(lv2,2),2),'relative_spectral_discrepancy':abs(main['spectral_norm']-large['spectral_norm'])/(.5*(main['spectral_norm']+large['spectral_norm'])+1e-15),'maximum_sign_asymmetry':max(scmap[(task,ep,bt,'main')]['sign_asymmetry'],scmap[(task,ep,bt,'large')]['sign_asymmetry']),'maximum_heldout_relative_error':next(x['relative_prediction_error'] for x in held if (x['task'],x['episode'],x['branch_time'],x['radius_label'])==(task,ep,bt,'large')),'linearity_passed':None})
 pqwrite(lin,art/'cross_radius_linearity.parquet'); ldf=pd.DataFrame(lin)
 # Rebuild branch descriptors from the reference-only freeze artifact.
 desc=pq.read_table(a.development_run.resolve()/'artifacts/development_state_descriptor.parquet').to_pylist(); dl={(x['task'],x['episode'],x['action_index']):np.asarray(x['descriptor']) for x in desc if x['cohort']=='confirmatory'}; schema=json.loads((a.manifest_dir/'state_descriptor_schema.json').read_text()); branch=json.loads((a.manifest_dir/'branch_manifest.json').read_text())['trajectories']; main=sdf[sdf.radius_label=='main']; mainop=odf[odf.radius_label=='main']; matches=[]
 for task in TASKS:
  demos=[x for x in branch if x['task']==task]; names=[x['name'] for x in schema['tasks'][task]]; robot_idx=np.asarray([i for i,n in enumerate(names) if n.startswith(('q_','qvel_','eef_','gripper_'))]); object_idx=np.asarray([i for i in range(len(names)) if i not in set(robot_idx)])
  for ai,ta in enumerate(demos):
   for tb in demos[ai+1:]:
    ba=sorted(ta['branches'],key=lambda x:x['action_index']); bb=sorted(tb['branches'],key=lambda x:x['action_index']); xa=np.asarray([dl[(task,ta['episode'],x['action_index'])] for x in ba]); xb=np.asarray([dl[(task,tb['episode'],x['action_index'])] for x in bb]); va=np.asarray([scmap[(task,ta['episode'],x['action_index'],'main')]['S_RMS'] for x in ba]); vb=np.asarray([scmap[(task,tb['episode'],x['action_index'],'main')]['S_RMS'] for x in bb]); pa=np.asarray([x['physical_progress_clipped'] for x in ba]); pb=np.asarray([x['physical_progress_clipped'] for x in bb]); mappings={}
    mappings['time']=[(i,i) for i in range(16)]; ri,ci=linear_sum_assignment(np.abs(pa[:,None]-pb[None,:])); mappings['progress']=list(zip(ri.tolist(),ci.tolist()))
    for method,ia in [('state',np.arange(xa.shape[1])),('robot_only',robot_idx),('object_only',object_idx)]:
     z1=xa[:,ia]; z2=xb[:,ia]; scale=np.std(np.concatenate([z1,z2]),axis=0); scale=np.maximum(scale,1e-6); cost=np.linalg.norm((z1[:,None,:]-z2[None,:,:])/scale,axis=2); mappings[method]=[tuple(map(int,x)) for x in monotone_match(cost,.25)['path']]
    mappings['unconstrained_nearest_neighbor']=[(i,int(np.argmin(np.linalg.norm(xb-xa[i],axis=1)))) for i in range(16)]
    for method,pairs in mappings.items():
     aa=np.asarray([va[i] for i,j in pairs]); vv=np.asarray([vb[j] for i,j in pairs]); sims1=[]; sims2=[]
     for i,j in pairs:
      oa=opmap[(task,ta['episode'],ba[i]['action_index'],'main')]; ob=opmap[(task,tb['episode'],bb[j]['action_index'],'main')]; a1=np.asarray(oa['top1_physical_direction'])[:,None]; b1=np.asarray(ob['top1_physical_direction'])[:,None]; a2=np.asarray(oa['top2_physical_directions']).reshape(7,2); b2=np.asarray(ob['top2_physical_directions']).reshape(7,2); sims1.append(projector_similarity(projector(a1,1),projector(b1,1),1)); sims2.append(projector_similarity(projector(a2,2),projector(b2,2),2))
     matches.append({'task':task,'episode_a':ta['episode'],'episode_b':tb['episode'],'method':method,'pair_count':len(pairs),'pairs':[list(x) for x in pairs],'scalar_spearman':rho(aa,vv),'median_top1_similarity':float(np.median(sims1)),'median_top2_similarity':float(np.median(sims2))})
 pqwrite(matches,art/'state_match_tables.parquet'); mdf=pd.DataFrame(matches); pqwrite(matches,art/'subspace_similarity.parquet')
 pairkeys=mdf[['task','episode_a','episode_b']].drop_duplicates(); cross=[]
 for _,k in pairkeys.iterrows():
  z=mdf[(mdf.task==k.task)&(mdf.episode_a==k.episode_a)&(mdf.episode_b==k.episode_b)]; vals={x.method:x.scalar_spearman for x in z.itertuples()}; better=max(vals['time'],vals['progress']); cross.append({'task':k.task,'episode_a':k.episode_a,'episode_b':k.episode_b,'time_rho':vals['time'],'progress_rho':vals['progress'],'state_rho':vals['state'],'robot_only_rho':vals['robot_only'],'object_only_rho':vals['object_only'],'better_exp4_baseline_rho':better,'state_improvement':vals['state']-better,'state_top1_similarity':float(z[z.method=='state'].median_top1_similarity.iloc[0]),'state_top2_similarity':float(z[z.method=='state'].median_top2_similarity.iloc[0])})
 pqwrite(cross,art/'crossfit_results.parquet'); cdf=pd.DataFrame(cross); demo=[]
 for task in TASKS:
  for ep in sorted(set(cdf[cdf.task==task].episode_a)|set(cdf[cdf.task==task].episode_b)):
   z=cdf[(cdf.task==task)&((cdf.episode_a==ep)|(cdf.episode_b==ep))]; demo.append({'task':task,'episode':ep,'mean_state_improvement':float(z.state_improvement.mean()),'state_rho':float(z.state_rho.mean())})
 ddf=pd.DataFrame(demo); rng=np.random.Generator(np.random.PCG64(950031)); boot=[]
 for _ in range(4000):
  ts=rng.choice(TASKS,3,replace=True); vals=[]
  for t in ts:
   z=ddf[ddf.task==t].mean_state_improvement.values; vals.extend(rng.choice(z,len(z),replace=True))
  boot.append(np.median(vals))
 bt=torch.as_tensor(boot,device='cuda',dtype=torch.float64); ci=torch.quantile(bt,torch.tensor([.025,.975],device='cuda',dtype=torch.float64)).cpu().numpy(); obs=float(np.median(ddf.mean_state_improvement)); prng=np.random.Generator(np.random.PCG64(950032)); perms=np.asarray([np.median(ddf.mean_state_improvement.values*prng.choice([-1,1],len(ddf))) for _ in range(4000)]); pval=float((1+np.sum(perms>=obs))/(len(perms)+1)); qval=float(bh_fdr([pval])[0])
 h_demo=[]
 for (task,ep),z in hdf[hdf.radius_label=='main'].groupby(['task','episode']):h_demo.append({'task':task,'episode':ep,'prediction_rho':rho(z.actual_operator_response_norm,z.predicted_operator_response_norm),'median_relative_error':float(z.relative_prediction_error.median())})
 pqwrite(h_demo,art/'heldout_direction_demo_summary.parquet'); hd=pd.DataFrame(h_demo); linmain=ldf[ldf.comparison=='small_vs_main']; cross_demo=linmain.groupby(['task','episode']).top1_similarity.median().reset_index(); linear_fraction=float(linmain.linearity_passed.mean()); task_state={t:float(cdf[cdf.task==t].state_rho.median()) for t in TASKS}; task_top1={t:float(cdf[cdf.task==t].state_top1_similarity.median()) for t in TASKS}; med_delta=float(cdf.state_improvement.median()); held_rho=float(hd.prediction_rho.median()); lodo=all(float(ddf.drop(i).mean_state_improvement.median())>0 for i in ddf.index); loto=all(float(ddf[ddf.task!=t].mean_state_improvement.median())>0 for t in TASKS)
 criteria={'scalar_delta_at_least_0_15':med_delta>=.15,'bootstrap_lower_positive':ci[0]>0,'two_tasks_state_rho_at_least_0_60':sum(v>=.6 for v in task_state.values())>=2,'two_tasks_top1_at_least_0_70':sum(v>=.7 for v in task_top1.values())>=2,'seventy_percent_demos_cross_radius_top1':float(np.mean(cross_demo.top1_similarity>=.7))>=.7,'heldout_prediction_rho_at_least_0_60':held_rho>=.6,'lodo_positive':lodo,'loto_stable':loto,'fdr_0_05':qval<.05}; criteria={k:bool(v) for k,v in criteria.items()}; strong=all(criteria.values())
 if strong: classification='state_conditioned_replicated_anisotropic_criticality'
 elif criteria['scalar_delta_at_least_0_15'] and criteria['bootstrap_lower_positive']: classification='state_alignment_only_without_subspace_replication'
 elif criteria['two_tasks_top1_at_least_0_70']: classification='subspace_replication_without_scalar_sparsity'
 elif linear_fraction<.7: classification='finite_radius_nonlinearity_dominates'
 elif med_delta<=0 and np.median(list(task_top1.values()))<.7: classification='trajectory_specific_criticality'
 else: classification='no_confirmatory_support'
 term=[]
 for task in TASKS:
  s=sdf[(sdf.task==task)&(sdf.radius_label=='main')]; o=odf[(odf.task==task)&(odf.radius_label=='main')]; term.append({'task':task,'scalar_terminal_rho':rho(s.S_RMS,s.terminal_consequence),'spectral_terminal_rho':rho(o.spectral_norm,o.terminal_consequence),'success_flip_branches':int(s.any_success_flip.sum())})
 pqwrite(term,art/'terminal_relevance.parquet'); decision={'schema_version':1,'classification':classification,'strong_rule_passed':strong,'criteria':criteria,'metrics':{'median_state_improvement':med_delta,'bootstrap_95_ci':ci.tolist(),'permutation_p':pval,'bh_q':qval,'task_state_rho':task_state,'task_top1_similarity':task_top1,'cross_radius_demo_fraction_top1_ge_0_70':float(np.mean(cross_demo.top1_similarity>=.7)),'local_linearity_branch_fraction':linear_fraction,'heldout_prediction_median_rho':held_rho,'main_large_top1_median':float(ldf[ldf.comparison=='main_vs_large'].top1_similarity.median()),'main_large_spectral_discrepancy_median':float(ldf[ldf.comparison=='main_vs_large'].relative_spectral_discrepancy.median())},'oracle_scheduler_eligible':classification=='state_conditioned_replicated_anisotropic_criticality','latent_rl_eligible':False}; write_json(art/'scientific_decision.json',decision)
 # Required non-placeholder diagnostic plots.
 plotbox(pd.DataFrame([{'task':x['task'],'distance':next(b['state_match_distance'] for t in branch if t['task']==x['task'] and t['episode']==x['episode'] for b in t['branches'] if b['action_index']==x['branch_time'])} for x in scalar if x['radius_label']=='main']),'distance','task',plots/'state_match_distance_distribution.png','Frozen state-match distance')
 fig,ax=plt.subplots(figsize=(8,5)); [ax.boxplot(cdf[c],positions=[i],widths=.6) for i,c in enumerate(['time_rho','progress_rho','state_rho'],1)]; ax.set_xticks([1,2,3],['time','progress','state']); ax.set_ylabel('pairwise Spearman'); ax.set_title('Alignment comparison'); fig.tight_layout(); fig.savefig(plots/'time_vs_progress_vs_state_alignment.png',dpi=160); plt.close(fig)
 plotbox(cdf,'state_rho','task',plots/'scalar_replication_by_alignment.png','State-aligned scalar replication','Spearman'); plotbox(cdf,'state_top1_similarity','task',plots/'top1_subspace_similarity_by_task.png','Top-1 state-matched subspace'); plotbox(cdf,'state_top2_similarity','task',plots/'top2_subspace_similarity_by_task.png','Top-2 state-matched subspace'); plotbox(odf,'leading_eigenvalue_share','task',plots/'leading_eigenvalue_share_by_state.png','Leading eigenvalue share')
 fig,ax=plt.subplots(figsize=(8,5)); [ax.plot(x.singular_values,alpha=.12,color='C0') for x in odf[odf.radius_label=='main'].itertuples()]; ax.set_yscale('log'); ax.set_title('Operator spectra across state prototypes'); ax.set_xlabel('component'); fig.tight_layout(); fig.savefig(plots/'operator_spectrum_by_state_prototype.png',dpi=160); plt.close(fig)
 plotbox(ldf[ldf.comparison=='small_vs_main'],'relative_spectral_discrepancy','task',plots/'cross_radius_spectral_consistency.png','Small-main spectral discrepancy'); plotbox(ldf[ldf.comparison=='small_vs_main'],'top1_similarity','task',plots/'cross_radius_subspace_similarity.png','Small-main top-1 similarity'); plotbox(sdf,'sign_asymmetry','radius_label',plots/'sign_asymmetry_by_radius.png','Sign asymmetry by radius')
 fig,ax=plt.subplots(figsize=(6,6)); ax.scatter(hdf.actual_operator_response_norm,hdf.predicted_operator_response_norm,s=8,alpha=.35); lo=min(ax.get_xlim()[0],ax.get_ylim()[0]); hi=max(ax.get_xlim()[1],ax.get_ylim()[1]); ax.plot([lo,hi],[lo,hi],'k--'); ax.set(xlabel='actual norm',ylabel='predicted norm',title='Held-out direction prediction'); fig.tight_layout(); fig.savefig(plots/'heldout_direction_prediction.png',dpi=160); plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(sdf[sdf.radius_label=='main'].S_RMS,sdf[sdf.radius_label=='main'].terminal_consequence,s=9,alpha=.4); ax.set(xlabel='S_RMS',ylabel='terminal consequence',title='Scalar sensitivity vs terminal outcome'); fig.tight_layout(); fig.savefig(plots/'state_conditioned_scalar_vs_terminal_outcome.png',dpi=160); plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,5)); z=odf[odf.radius_label=='main']; ax.scatter(z.spectral_norm,z.terminal_consequence,s=9,alpha=.4); ax.set(xlabel='spectral norm',ylabel='terminal consequence',title='Spectral sensitivity vs terminal outcome'); fig.tight_layout(); fig.savefig(plots/'state_conditioned_spectral_vs_terminal_outcome.png',dpi=160); plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,5)); x=np.arange(3); ax.bar(x-.18,[task_state[t] for t in TASKS],.36,label='scalar rho'); ax.bar(x+.18,[task_top1[t] for t in TASKS],.36,label='top1 sim'); ax.set_xticks(x,[SHORT[t] for t in TASKS]); ax.legend(); ax.set_title('Task specificity'); fig.tight_layout(); fig.savefig(plots/'task_specificity_summary.png',dpi=160); plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,5)); ax.bar(np.arange(len(ddf)),ddf.mean_state_improvement); ax.axhline(0,color='k'); ax.set_title('LODO inputs: per-demo state improvement'); fig.tight_layout(); fig.savefig(plots/'lodo_loto_summary.png',dpi=160); plt.close(fig)
 gpu=json.loads((a.gpu_run.resolve()/'artifacts/gpu_cpu_equivalence.json').read_text()); shutil.copy2(a.gpu_run.resolve()/'artifacts/gpu_audit.json',art/'gpu_audit.json'); shutil.copy2(a.gpu_run.resolve()/'artifacts/gpu_cpu_equivalence.json',art/'gpu_cpu_equivalence.json'); fig,ax=plt.subplots(figsize=(8,4)); vals=[max(x['operator_max_abs'] for x in gpu['records']),max(x['gram_max_abs'] for x in gpu['records']),max(x['singular_max_abs'] for x in gpu['records'])]; ax.bar(['operator','Gram','singular'],np.maximum(vals,1e-18)); ax.set_yscale('log'); ax.set_title('GPU/CPU maximum absolute discrepancies'); fig.tight_layout(); fig.savefig(plots/'gpu_cpu_equivalence.png',dpi=160); plt.close(fig)
 shutil.copy2(raw/'artifacts/failure_examples.json',art/'failure_examples.json'); write_json(art/'analysis_hashes.json',{p.name:sha(p) for p in art.iterdir() if p.is_file()}); metrics={'run_id':a.run_id,'status':'completed','gate':{'passed':True},'gpu_used':True,'operator_rows':len(ops),'scalar_rows':len(scalar),'state_match_rows':len(matches),'linearity_rows':len(lin),'classification':classification,'decision_metrics':decision['metrics'],'terminal_relevance':term,'plot_count':len(list(plots.glob('*.png'))),'wall_time_seconds':time.perf_counter()-started}; write_run_record(run,config={'stage':'EXP5-14 through EXP5-20','raw_run':raw.name},command=shlex.join([sys.executable,*sys.argv]),environment={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},git_state={'project':git_record(ROOT)},metrics=metrics,stdout=json.dumps(metrics),stderr=''); print(json.dumps(metrics,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
