"""Observational prior-compliance outcomes and archived search trees."""
import os,csv,json,sys,tarfile
from pathlib import Path
from workspace_paths import workspace_root
from collections import Counter
BASE=workspace_root();sys.path.insert(0,str(BASE/'analysis'))
O=BASE/'experiments/analysis/MAIR_vs_AgenticIR_noexp_test160_20261016'
os.environ['MPLCONFIGDIR']=str(O/'sequence_cache/matplotlib')
from compare_baselines import stats,save_csv
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
rows=list(csv.DictReader((O/'tables/tool_sequences_per_sample.csv').open(encoding='utf-8-sig')))
pairs=list(csv.DictReader((O/'tables/paired_samples.csv').open(encoding='utf-8-sig')))
for p in pairs:
    for k in list(p):
        if any(t in k for t in ('psnr','ssim','elapsed_s','tool_calls','tool_errors')):p[k]=float(p[k])
P={p['sample_id']:p for p in pairs};A={r['sample_id']:r for r in rows if r['method']=='AgenticIR'}
def multi(s):return len(set(s.split(' > ')))>1
results=[];counts=[];bygroup=[]
for phase in ('initial','final'):
    for scope in ('all','multistage'):
        for compliant in (True,False):
            rr=[r for r in A.values() if (scope=='all' or multi(r['initial_stages' if phase=='initial' else 'final_stages'])) and (int(r[phase+'_inversions'])==0)==compliant]
            st=stats([P[r['sample_id']] for r in rr]);results.append(dict(phase=phase,scope=scope,agent_compliant=compliant,**st))
    for g in dict.fromkeys(r['degradation'] for r in A.values()):
        for compliant in (True,False):
            rr=[r for r in A.values() if r['degradation']==g and (int(r[phase+'_inversions'])==0)==compliant]
            if rr:bygroup.append(dict(phase=phase,degradation=g,agent_compliant=compliant,**stats([P[r['sample_id']] for r in rr])))
save_csv(O/'tables/prior_compliance_outcomes.csv',results);save_csv(O/'tables/prior_compliance_by_degradation.csv',bygroup)
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
for i,phase in enumerate(('initial','final')):
    rr=[r for r in results if r['phase']==phase and r['scope']=='multistage']
    for j,r in enumerate(rr):
        ax[i].bar(j-.17,r['mair_psnr'],width=.32,color='#26799D',label='MAIR' if j==0 else None)
        ax[i].bar(j+.17,r['agenticir_psnr'],width=.32,color='#D4783C',label='AgenticIR' if j==0 else None)
        ax[i].text(j,max(r['mair_psnr'],r['agenticir_psnr'])+.3,f"n={r['n']} | delta={r['delta_psnr']:+.2f}",ha='center',fontsize=9)
    ax[i].set_xticks([0,1],['AgenticIR compliant','AgenticIR inverted']);ax[i].set_title(phase.title()+' plan/chain: multi-stage only');ax[i].set_ylim(0,32);ax[i].set_ylabel('Mean PSNR (dB)');ax[i].legend()
fig.suptitle('Same images within each subgroup; subgroup membership is observational',fontsize=11);fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(O/'figures'/('07_prior_compliance_quality.'+ext),dpi=220,bbox_inches='tight')
plt.close(fig)
# Three deterministic cases: simple reversal, multi-step correction, most adjustments.
selected=['019__rain+blur','070__rain+blur','010__rain+blur']
D=O/'search_trees';D.mkdir(exist_ok=True)
short={'motion deblurring':'Deblur','denoising':'Denoise','super-resolution':'SR','deraining':'Derain','dehazing':'Dehaze','brightening':'Brighten','jpeg compression artifact removal':'DeJPEG','defocus deblurring':'Defocus'}
stage={'motion deblurring':2,'denoising':2,'super-resolution':2,'deraining':3,'dehazing':3,'brightening':3,'jpeg compression artifact removal':1,'defocus deblurring':2}
lognames={}
for sid in selected:
    s=json.loads((O/'sequence_cache'/(sid+'.json')).read_text(encoding='utf8'))
    root=Path(s['tree']['img_path'].replace('/','/'))
    name=str(root.parent.parent.parent/'logs/workflow.log').replace('\\','/')
    lognames['runs/'+name.split('/runs/')[1]]=sid
    final=list(zip(s['execution_path']['subtasks'],s['execution_path']['tools']))
    nodes=[dict(label='Input',parent=None,path=[],severity='')]
    def walk(tree,parent,path):
        for task,branch in tree.get('children',{}).items():
            for tool,child in branch['tools'].items():
                cur=path+[(task,tool)];idx=len(nodes)
                nodes.append(dict(label=f"{short.get(task,task)} [S{stage[task]}]\n{tool}\n{child.get('severity','?')}",parent=parent,path=cur,severity=child.get('severity','')))
                walk(child,idx,cur)
    walk(s['tree'],0,[])
    # Each leaf gets one vertical slot; internal nodes centered over children.
    kids={i:[j for j,n in enumerate(nodes) if n['parent']==i] for i in range(len(nodes))};cursor=[0]
    def place(i):
        if not kids[i]:y=cursor[0];cursor[0]+=1
        else:y=float(np.mean([place(j) for j in kids[i]]))
        nodes[i]['y']=y;return y
    place(0);leaves=cursor[0];depth=max(len(n['path']) for n in nodes)
    fig,ax=plt.subplots(figsize=(max(11,3.2*(depth+1)),max(5,.83*leaves+2.7)))
    for i,n in enumerate(nodes):
        x=len(n['path'])*3.4;y=-n['y'];active=n['path']==final[:len(n['path'])]
        if n['parent'] is not None:
            par=nodes[n['parent']];ax.plot([len(par['path'])*3.4,x],[-par['y'],y],color='#23854C' if active else '#B4BBC2',linewidth=2.6 if active else 1,zorder=1)
        ax.text(x,y,n['label'],ha='center',va='center',fontsize=8.5,bbox=dict(boxstyle='round,pad=.45',facecolor='#E1F2E4' if active else '#F0F1F2',edgecolor='#23854C' if active else '#A9AFB5'),zorder=2)
    logpath=D/(sid+'_workflow.log')
    if logpath.exists():
        import re
        names={'input':0}
        for i,n in enumerate(nodes):
            if n['path']:names['-'.join(task+'@'+tool for task,tool in n['path'])]=i
        latest=None;rb=0
        for line in logpath.read_text(encoding='utf8').splitlines():
            hit=re.match(r'.* result: (.+) with .+ severity\.',line)
            if hit:latest=hit.group(1)
            hit=re.match(r'Roll back for (\d+) step\(s\) to (.+) with agenda ',line)
            if hit:
                rb+=1;target=hit.group(2)
                if latest in names and target in names:
                    a0=nodes[names[latest]];b0=nodes[names[target]]
                    ax.annotate('',xy=(len(b0['path'])*3.4,-b0['y']+.15),xytext=(len(a0['path'])*3.4,-a0['y']+.2),arrowprops=dict(arrowstyle='->',linestyle='--',color='#BE4040',alpha=.75,connectionstyle='arc3,rad=.16'),zorder=1)
        (D/(sid+'_rollback_events.txt')).write_text('\n'.join(line for line in logpath.read_text(encoding='utf8').splitlines() if line.startswith(('Roll back','Adjusted plan:'))),encoding='utf8')
    changes=s['plan'].get('adjusted',[])
    title=f"{sid}: {s['n_invocations']} attempts; {len(changes)} plan adjustments\nInitial: {' > '.join(short.get(t,t) for t in s['plan']['initial'])} | Final: {' > '.join(short.get(t,t) for t,_ in final)}"
    ax.set_title(title,fontsize=12,pad=28);ax.set_xlim(-1.6,depth*3.4+1.6);ax.set_ylim(-leaves+.15,1);ax.axis('off')
    fig.text(.02,.012,'Green: final path. Gray: explored alternatives. Red dashed: logged rollback from the last selected result to its ancestor. Layout is a search tree, not invocation chronology.\nSeverity labels evaluate the targeted degradation; they are not PSNR. Failed tool calls may lack tree nodes.',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,1))
    for ext in ('png','pdf'):fig.savefig(D/(sid+'.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    (D/(sid+'_adjustments.json')).write_text(json.dumps(dict(selection_reason={'019__rain+blur':'Simple scene-to-imaging inversion corrected','070__rain+blur':'Five adjustments with correction to stage order','010__rain+blur':'Maximum recorded adjustments (6)'}[sid],plan=s['plan'],final=s['execution_path'],tool_errors=s.get('tool_errors',[])),indent=2),encoding='utf8')
# Extract authentic workflow logs for chronological rollback evidence.
archive=BASE/'experiments/results/AgenticIR_noexp_test160_deepseek-v4-flash-vision-exp_20261015/agenticir_completed_verified.tar.gz'
missing={n:sid for n,sid in lognames.items() if not (D/(sid+'_workflow.log')).exists()}
if missing:
    with tarfile.open(archive,'r|gz') as tf:
        for member in tf:
            n=member.name.removeprefix('./')
            if n in missing:
                sid=missing.pop(n);(D/(sid+'_workflow.log')).write_bytes(tf.extractfile(member).read())
                if not missing:break
    if missing:raise RuntimeError(missing)
(O/'prior_outcomes.json').write_text(json.dumps(results,indent=2),encoding='utf8')
print(json.dumps(results,indent=2))
