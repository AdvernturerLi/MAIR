"""Audit committed trajectories; distinguish plans, attempts and accepted chains."""
import csv,json,tarfile,os
from pathlib import Path
from workspace_paths import workspace_root
from collections import Counter
BASE=workspace_root()
OUT=BASE/'experiments/analysis/MAIR_vs_AgenticIR_noexp_test160_20261016'
R=BASE/'experiments/results'
M=R/'MAIR_noexp_test160_deepseek-v4-flash-vision-exp_20261015'
A=R/'AgenticIR_noexp_test160_deepseek-v4-flash-vision-exp_20261015'
CACHE=OUT/'sequence_cache'; CACHE.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(CACHE/'matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
STAGE={'jpeg compression artifact removal':1,'jpeg artifact removal':1,'jpeg compression artifacts removal':1,'denoising':2,'motion deblurring':2,'defocus deblurring':2,'super-resolution':2,'deraining':3,'dehazing':3,'brightening':3}
def stage(task):
    if task not in STAGE:raise ValueError('Unknown subtask: '+task)
    return STAGE[task]
def violations(seq):
    s=[stage(x) for x in seq]
    return sum(s[i]>s[j] for i in range(len(s)) for j in range(i+1,len(s)))
def save(name,rows):
    with (OUT/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
records=json.loads((A/'completed_results.json').read_text(encoding='utf8'))
wanted={r['output'].split('/runs/')[1].rsplit('/',1)[0]+'/agent_summary.json':r['sample_id'] for r in records}
missing={n:i for n,i in wanted.items() if not (CACHE/(i+'.json')).exists()}
if missing:
    with tarfile.open(A/'agenticir_completed_verified.tar.gz','r|gz') as tf:
        for member in tf:
            n=member.name.removeprefix('./').removeprefix('runs/')
            if n in missing:
                (CACHE/(missing[n]+'.json')).write_bytes(tf.extractfile(member).read())
                del missing[n]
                if not missing:break
    if missing:raise RuntimeError(missing)
paired={r['sample_id']:r for r in csv.DictReader((OUT/'tables/paired_samples.csv').open(encoding='utf-8-sig'))}
rows=[];patterns=[]
for r in records:
    sid=r['sample_id'];a=json.loads((CACHE/(sid+'.json')).read_text(encoding='utf8'))
    trajectories=list((M/'runs'/sid).glob('attempt_*/trajectory.json'))
    if len(trajectories)!=1:raise RuntimeError('Ambiguous MAIR attempt '+sid)
    t=json.loads(trajectories[0].read_text(encoding='utf8'))
    plans=[x['plan'] for x in t if x['step']=='plan']
    if len(plans)!=1:raise RuntimeError('Ambiguous plan '+sid)
    accepted=[x for x in t if x['step']=='stop' or (x['step']=='expert_output' and not x.get('retained_input',False))]
    # Early threshold success returns before expert_output; include its stop event.
    if accepted:
        last=accepted[-1]
        output=last.get('output') or next(x['output'] for x in reversed(t[:t.index(last)]) if x['step']=='tool_reflection' and x['tool']==last['tool'])
        import hashlib
        local=M/'runs'/output.split('/runs/')[1]
        assert hashlib.sha256(local.read_bytes()).digest()==hashlib.sha256((trajectories[0].parent/'result.png').read_bytes()).digest(),sid
    chain=[x['subtask'] for x in accepted]
    attempts=[x for x in t if x['step']=='tool_execution']
    for method,plan,seq,tools,calls,errors,replans in [
        ('MAIR',plans[0],chain,[x.get('tool') for x in accepted],len(attempts),sum(x['step']=='tool_error' for x in t),0),
        ('AgenticIR',a['plan']['initial'],a['execution_path']['subtasks'],a['execution_path']['tools'],a['n_invocations'],len(a.get('tool_errors',[])),len(a['plan'].get('adjusted',[])))]:
        p=paired[sid]; expected=int(float(p[method.lower()+'_tool_calls']))
        if calls!=expected:raise RuntimeError((sid,method,calls,expected))
        rows.append(dict(sample_id=sid,degradation=p['degradation'],method=method,initial_plan=' > '.join(plan),initial_stages=' > '.join(str(stage(x)) for x in plan),initial_inversions=violations(plan),final_tasks=' > '.join(seq),final_tools=' > '.join(str(x) for x in tools),final_stages=' > '.join(str(stage(x)) for x in seq),final_inversions=violations(seq),final_chain_length=len(seq),attempted_calls=calls,calls_outside_final_chain=calls-len(seq),tool_errors=errors,replans=replans,psnr=p[method.lower()+'_psnr'],ssim=p[method.lower()+'_ssim']))
summary=[]
for method in ('MAIR','AgenticIR'):
    rr=[r for r in rows if r['method']==method]
    summary.append(dict(method=method,n=len(rr),initial_violating_samples=sum(r['initial_inversions']>0 for r in rr),final_violating_samples=sum(r['final_inversions']>0 for r in rr),attempted_calls=sum(r['attempted_calls'] for r in rr),final_chain_steps=sum(r['final_chain_length'] for r in rr),calls_outside_final_chain=sum(r['calls_outside_final_chain'] for r in rr),tool_errors=sum(r['tool_errors'] for r in rr),replanning_samples=sum(r['replans']>0 for r in rr),replans=sum(r['replans'] for r in rr)))
    for (group,seq),n in Counter((r['degradation'],r['final_tasks']) for r in rr).most_common():patterns.append(dict(method=method,degradation=group,final_tasks=seq,count=n))
save('tool_sequences_per_sample.csv',rows);save('tool_sequence_summary.csv',summary);save('tool_sequence_patterns.csv',patterns)
fig,ax=plt.subplots(1,2,figsize=(10,4))
for i,key in enumerate(('initial_violating_samples','final_violating_samples')):
    ax[0].bar([i*3+j for j in range(2)],[s[key] for s in summary],color=['#26799D','#D4783C'],label=key)
ax[0].text(0,2,'MAIR: 0',ha='center',fontsize=9);ax[0].text(3,2,'MAIR: 0',ha='center',fontsize=9)
ax[0].text(1,67,'AgenticIR: 65',ha='center',fontsize=9);ax[0].text(4,59,'AgenticIR: 57',ha='center',fontsize=9)
ax[0].set_ylim(0,75)
ax[0].set_xticks([.5,3.5],['Initial plan','Final chain']);ax[0].set_ylabel('Samples with stage-order inversions / 160')
for j,s in enumerate(summary):
    ax[1].bar(j,s['final_chain_steps'],color='#59956B',label='Final chain steps' if j==0 else None)
    ax[1].bar(j,s['calls_outside_final_chain'],bottom=s['final_chain_steps'],color='#D7A45B',label='Other attempted calls' if j==0 else None)
ax[1].set_xticks([0,1],['MAIR','AgenticIR']);ax[1].set_ylabel('Tool calls');ax[1].legend()
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(OUT/'figures'/('06_tool_sequences.'+ext),dpi=240,bbox_inches='tight')
(CACHE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2))

grouped=[]
for group in dict.fromkeys(r['degradation'] for r in rows):
    rr=[r for r in rows if r['method']=='AgenticIR' and r['degradation']==group]
    grouped.append(dict(degradation=group,n=len(rr),agent_initial_inversions=sum(r['initial_inversions']>0 for r in rr),agent_final_inversions=sum(r['final_inversions']>0 for r in rr),mair_final_inversions=0,agent_replanning_samples=sum(r['replans']>0 for r in rr)))
save('tool_sequence_by_degradation.csv',grouped)
