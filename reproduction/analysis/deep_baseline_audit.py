"""Local diagnostic audit, excluded from experience-learning inputs."""
import os,sys,csv,json,hashlib,tarfile
from pathlib import Path
from workspace_paths import workspace_root
from collections import Counter
B=workspace_root();sys.path.insert(0,str(B/'analysis'))
O=B/'experiments/analysis/MAIR_vs_AgenticIR_noexp_test160_20261016'
os.environ['MPLCONFIGDIR']=str(O/'sequence_cache/matplotlib')
import numpy as np,cv2
from skimage.metrics import structural_similarity,peak_signal_noise_ratio
import matplotlib.pyplot as plt
from compare_baselines import stats,save_csv
R=list(csv.DictReader((O/'tables/paired_samples.csv').open(encoding='utf-8-sig')))
for r in R:
 for k in list(r):
  if any(t in k for t in ('psnr','ssim','elapsed_s','tool_calls','tool_errors')):r[k]=float(r[k])
P={r['sample_id']:r for r in R}
seq=list(csv.DictReader((O/'tables/tool_sequences_per_sample.csv').open(encoding='utf-8-sig')))
S={(r['sample_id'],r['method']):r for r in seq}
cache=O/'tables/input_reference_metrics.csv'
if cache.exists():inputs={r['sample_id']:r for r in csv.DictReader(cache.open(encoding='utf-8-sig'))}
else:
 inputs={}
 for r in R:
  sid=r['sample_id'];x=cv2.imread(str(O/'inputs/test/lq'/(sid+'.png')));y=cv2.imread(str(O/'inputs/test/hq'/(r['hq_id']+'.png')))
  assert x is not None and y is not None
  resized=x.shape!=y.shape
  if resized:x=cv2.resize(x,(y.shape[1],y.shape[0]),interpolation=cv2.INTER_CUBIC)
  inputs[sid]=dict(sample_id=sid,input_psnr=float(peak_signal_noise_ratio(y,x,data_range=255)),input_ssim=float(structural_similarity(y,x,channel_axis=2,data_range=255)),input_resized=resized)
 save_csv(cache,list(inputs.values()))
for r in R:
 r.update({k:float(inputs[r['sample_id']][k]) for k in ('input_psnr','input_ssim')})
 for m in ('mair','agenticir'):
  for metric in ('psnr','ssim'):r[m+'_gain_'+metric]=r[m+'_'+metric]-r['input_'+metric]
summary={};groups=[]
for g in dict.fromkeys(r['degradation'] for r in R):
 rr=[r for r in R if r['degradation']==g]
 st=stats(rr);st['degradation']=g
 st['psnr_contribution_to_overall']=st['delta_psnr']/8
 st['ssim_contribution_to_overall']=st['delta_ssim']/8
 st['input_psnr']=float(np.mean([r['input_psnr'] for r in rr]));st['input_ssim']=float(np.mean([r['input_ssim'] for r in rr]))
 for m in ('mair','agenticir'):
  for metric in ('psnr','ssim'):
   v=np.array([r[m+'_gain_'+metric] for r in rr]);st[m+'_gain_'+metric]=float(v.mean());st[m+'_regressed_'+metric]=int((v < -1e-9).sum())
 groups.append(st)
save_csv(O/'tables/deep_degradation_diagnostics.csv',groups)
hq=[]
for h in sorted({r['hq_id'] for r in R}):hq.append(dict(hq_id=h,**stats([r for r in R if r['hq_id']==h])))
save_csv(O/'tables/deep_hq_consistency.csv',hq)
summary['positive_hq_mean_psnr']=sum(r['delta_psnr']>0 for r in hq)
summary['positive_hq_mean_ssim']=sum(r['delta_ssim']>0 for r in hq)
summary['without_rainhaze_hazeSR']=stats([r for r in R if r['degradation'] not in ('rain+haze','haze+low-resolution')])
summary['trimmed_outliers']=stats(sorted(R,key=lambda r:r['delta_psnr'])[8:-8])
summary['two_haze_group_share_psnr']=sum(g['psnr_contribution_to_overall'] for g in groups if g['degradation'] in ('rain+haze','haze+low-resolution'))/np.mean([r['delta_psnr'] for r in R])
# Label/task overlap is diagnostic: synthetic labels do not imply each task remains necessary.
mapping={'noise':'denoising','blur':'motion deblurring','rain':'deraining','haze':'dehazing','low-light':'brightening','low-resolution':'super-resolution','jpeg':'jpeg compression artifact removal'}
coverage=[]
for r in R:
 expected={mapping[t] for t in r['degradation'].split('+')}
 for method in ('MAIR','AgenticIR'):
  s=S[r['sample_id'],method]
  plan=set(filter(None,s['initial_plan'].split(' > ')));final=set(filter(None,s['final_tasks'].split(' > ')))
  coverage.append(dict(sample_id=r['sample_id'],degradation=r['degradation'],method=method,expected_tasks=' > '.join(sorted(expected)),initial_missing=' > '.join(sorted(expected-plan)),initial_extra=' > '.join(sorted(plan-expected)),final_missing=' > '.join(sorted(expected-final)),final_extra=' > '.join(sorted(final-expected)),initial_exact=expected==plan,final_exact=expected==final))
save_csv(O/'tables/deep_task_coverage.csv',coverage)
summary['coverage']={m:{key:sum(bool(r[key]) for r in coverage if r['method']==m) for key in ('initial_missing','initial_extra','final_missing','final_extra','initial_exact','final_exact')} for m in ('MAIR','AgenticIR')}
# Compare paired outcomes conditional on recorded planning behavior, never counterfactual effects.
behavior=[]
for name,predicate in [('agent_no_replans',lambda s:int(s['replans'])==0),('agent_replans',lambda s:int(s['replans'])>0),('agent_inversion_corrected',lambda s:int(s['initial_inversions'])>0 and int(s['final_inversions'])==0),('agent_inversion_persisted',lambda s:int(s['initial_inversions'])>0 and int(s['final_inversions'])>0)]:
 rr=[r for r in R if predicate(S[r['sample_id'],'AgenticIR'])]
 st=dict(group=name,**stats(rr))
 for m in ('mair','agenticir'):
  st[m+'_calls_mean']=float(np.mean([r[m+'_tool_calls'] for r in rr]));st[m+'_elapsed_mean']=float(np.mean([r[m+'_elapsed_s'] for r in rr]))
  st[m+'_gain_psnr']=float(np.mean([r[m+'_gain_psnr'] for r in rr]));st[m+'_gain_ssim']=float(np.mean([r[m+'_gain_ssim'] for r in rr]))
 behavior.append(st)
save_csv(O/'tables/deep_search_behavior.csv',behavior)
summary['method_vs_input']={m:{'mean_gain_psnr':float(np.mean([r[m+'_gain_psnr'] for r in R])),'mean_gain_ssim':float(np.mean([r[m+'_gain_ssim'] for r in R])),'psnr_regressions':sum(r[m+'_gain_psnr'] < -1e-9 for r in R),'ssim_regressions':sum(r[m+'_gain_ssim'] < -1e-9 for r in R)} for m in ('mair','agenticir')}
save_csv(O/'tables/deep_per_sample_input_gains.csv',R)
fig,ax=plt.subplots(1,2,figsize=(13,5))
y=np.arange(len(groups));colors=['#26799D' if g['delta_psnr']>=0 else '#D4783C' for g in groups]
ax[0].barh(y,[g['psnr_contribution_to_overall'] for g in groups],color=colors);ax[0].set_yticks(y,[g['degradation'] for g in groups]);ax[0].invert_yaxis();ax[0].axvline(0,color='gray',linewidth=.8);ax[0].set_xlabel('Contribution to overall PSNR difference (dB)');ax[0].set_title('Which conditions drive the mean advantage?')
for j,m in enumerate(('mair','agenticir')):
 ax[1].scatter([r['input_psnr'] for r in R],[r[m+'_gain_psnr'] for r in R],s=17,alpha=.6,label=m.upper() if m=='mair' else 'AgenticIR',color=['#26799D','#D4783C'][j])
ax[1].axhline(0,color='gray',linewidth=.8);ax[1].set_xlabel('Input PSNR (dB)');ax[1].set_ylabel('Output PSNR - input PSNR (dB)');ax[1].legend();ax[1].set_title('Improvement versus leaving the input unchanged')
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(O/'figures'/('08_deep_diagnostics.'+ext),dpi=220,bbox_inches='tight')
plt.close(fig)
(O/'deep_audit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2))
