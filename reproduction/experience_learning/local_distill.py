"""Offline evidence-to-text pilot summaries; explicitly not LLM distillation."""
import json,time,hashlib,os
from pathlib import Path
from collections import defaultdict
ROOT=Path('/root/autodl-tmp/experience_validation_pilot_20261016')
def write(p,data):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
state={'pid':os.getpid(),'mode':'local_statistical_summary','external_calls':False,'methods':{}}
write(ROOT/'local_distillation_progress.json',state)
for_wait=set(('agenticir','mair'))
while for_wait:
    for method in list(for_wait):
        source=ROOT/method/('fail_rate.json' if method=='agenticir' else 'order_scores.json')
        if not source.exists():continue
        raw=source.read_text();data=json.loads(raw);texts=[]
        for combo,orders in data.items():
            scores={order:(entry['fail rate']['total'] if method=='agenticir' else entry['mean_score']) for order,entry in orders.items()}
            best=min(scores,key=scores.get) if method=='agenticir' else max(scores,key=scores.get)
            texts.append(f"Validation-pilot task set {combo}: order statistics {json.dumps(scores,sort_keys=True)}. The observed {'lowest mean residual failure rate' if method=='agenticir' else 'highest mean seven-IQA score'} is for {best}. This is only two HQ sources; no general ordering claim is established.")
        if method=='mair':texts.append('These observations concern within-stage ordering only; preserve compression -> imaging -> scene across stages.')
        evidence={'method':method,'kind':'local_statistical_pilot_summary_not_llm_distillation','pilot_only':True,'test_data_used':False,'external_calls':False,'source':str(source),'source_sha256':hashlib.sha256(raw.encode()).hexdigest(),'raw_statistics':data,'distilled':'\n'.join(texts),'not_for_confirmatory_test':True,'limitation':'2 HQ pilot only; matched tool/HQ stability analysis and full exploration remain required before freezing experience.'}
        write(ROOT/method/'candidate_experience.local_statistics.json',evidence)
        state['methods'][method]='pilot_summary_ready';for_wait.remove(method)
    if not for_wait:state['state']='completed';write(ROOT/'local_distillation_progress.json',state);break
    progress=ROOT/'progress.json'
    if progress.exists() and json.loads(progress.read_text()).get('state') in ('needs_attention','interrupted'):
        state['state']='waiting_for_exploration_recovery';write(ROOT/'local_distillation_progress.json',state)
    else:
        state['state']='waiting_for_local_statistics';write(ROOT/'local_distillation_progress.json',state)
    time.sleep(30)
