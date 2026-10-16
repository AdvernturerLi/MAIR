import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path('/root/autodl-tmp/experience_validation_pairwise7_10hq_20261016_ridcp_excluded')
PROJECT=Path('/root/autodl-tmp/AgenticIR_mairtools_20261015')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(PROJECT))
from pairwise_evidence import build
state={'pid':os.getpid(),'state':'waiting_for_local_evaluations_and_review','methods':{},'export':'aggregate text statistics only; no images'}
def update(**kw):
 state.update(kw);tmp=ROOT/'distillation_progress.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(ROOT/'distillation_progress.json')
update()
while not all((ROOT/p).exists() for p in ('agenticir/fail_rate.json','mair/scored_sequences.json','mair/order_scores.json','review_approved.json')):time.sleep(30)
approval=json.loads((ROOT/'review_approved.json').read_text())
assert approval['validation_passed'] is True
for relative,expected in approval['protected_sha256'].items():
 assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,'Reviewed inputs changed: '+relative
try:
 import yaml
 from llm import GPT4
 cfg=yaml.safe_load((PROJECT/'config.alternative.runtime.yml').read_text())
 assert cfg.get('OPENAI_BASE_URL','').rstrip('/') in ('https://api.deepseek.com','https://api.deepseek.com/v1')
 del cfg
 for method in ('agenticir','mair'):
  folder=ROOT/method
  if method=='agenticir':records=[json.loads(line) for line in (folder/'observations.jsonl').read_text().splitlines()]
  else:records=json.loads((folder/'scored_sequences.json').read_text())
  assert len(records)==2540
  evidence=build(records,method)
  evidence['aggregate']=json.loads((folder/('fail_rate.json' if method=='agenticir' else 'order_scores.json')).read_text())
  if method=='mair':evidence['aggregate']={k:v for k,v in evidence['aggregate'].items() if k in evidence['task_order_comparisons']}
  (folder/'matched_order_evidence.json').write_text(json.dumps(evidence,indent=2))
  raw=json.dumps(evidence,indent=2)
  prompt=('Distill scheduling experience from controlled synthetic validation exploration: 10 independent HQ sources, seven double-degradation combinations, no test data. '
    'Each tool combination is a repeated measure; do not call 2540 leaves independent images. Follow the utility and order delta definitions exactly. '
    'Use per-HQ direction stability, matched tool assignments, confidence intervals, and leave-one-HQ-out results. '
    'Do not claim causality or certainty from aggregate ranks. If sources or tool assignments conflict, describe conditionality or abstain. '
    'No evidence covers three simultaneous imaging degradations; do not invent noise+blur+SR conditional rules. '
    'Provide concise plain-text scheduling guidance with explicit coverage, evidence strength and counterexamples. ')
  prompt+=('AgenticIR: learn global task ordering using residual degradation failure rates. Lower failure rate is better; utility is negative failure rate. ' if method=='agenticir' else 'MAIR: ONLY learn within-stage ordering. The evidence is restricted to same-stage task pairs. Preserve compression -> imaging -> scene across stages. Higher signed seven-IQA score is better. ')
  prompt+='\nObserved statistics:\n'+raw
  (folder/'distillation_input.json').write_text(json.dumps({'prompt':prompt,'external_export':'aggregate text statistics only'},indent=2))
  update(state='running',method=method)
  gpt=GPT4(config_path=PROJECT/'config.alternative.runtime.yml',silent=True)
  distilled=gpt(prompt=prompt);assert isinstance(distilled,str) and distilled.strip()
  value={'raw':raw,'distilled':distilled,'source':str(ROOT/'protocol.json'),'kind':'independently_distilled_synthetic_validation_pairwise','method':method,'model':gpt.model,'test_data_used':False,'pilot_only':False,'hq_count':10,'observations_sha256':hashlib.sha256(raw.encode()).hexdigest(),'coverage':list(evidence['task_order_comparisons']),'review_required_before_test':True}
  (folder/'schedule_experience.json').write_text(json.dumps(value,indent=2));state['methods'][method]='completed';update()
 update(state='completed')
except BaseException as error:
 update(state='needs_attention',error=f'{type(error).__name__}: {error}');raise
