import hashlib,json,os,shutil,socket,subprocess,time
from pathlib import Path
BASE=Path('/root/autodl-tmp');OLD=BASE/'experience_validation_pilot_20261016_ridcp_excluded';ROOT=BASE/'experience_validation_pairwise7_10hq_20261016_ridcp_excluded';PROJECT=BASE/'AgenticIR_mairtools_20261015'
assert json.loads((OLD/'progress.json').read_text())['state']=='completed'
assert json.loads((OLD/'deepseek_distillation_progress.json').read_text())['state']=='completed'
assert json.loads((OLD/'offline_iqa_verification.json').read_text())['state']=='all_offline_loads_passed'
assert not (ROOT/'launch.json').exists(),'Refusing duplicate expansion launch'
with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',5001))==0,'DepictQA required'
ROOT.mkdir(exist_ok=True)
for source,target in [('pilot_pairwise.py','pilot.py'),('pairwise_evidence.py','pairwise_evidence.py'),('distill_pairwise_queue.py','distill_pairwise_queue.py')]:shutil.copy2(BASE/source,ROOT/target)
reused=0
for record in (OLD/'nodes').glob('*/*/node.json'):
 data=json.loads(record.read_text())
 if data.get('status')!='success':continue
 assert all(step[1]!='ridcp' for step in data['path'])
 shutil.copytree(record.parent,ROOT/'nodes'/record.parent.relative_to(OLD/'nodes'),dirs_exist_ok=True);reused+=1
assert reused==178
python='/root/miniconda3/envs/diffbir/bin/python';env=dict(os.environ)
for key in ('OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL','PYTHONPATH'):env.pop(key,None)
env.update(TZ='Asia/Shanghai',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
command=[python,'-u',str(ROOT/'pilot.py'),'run','--project',str(PROJECT),'--data',str(BASE/'MAIR/experiments/datasets/mair_benchmark_v1'),'--output',str(ROOT),'--config',str(PROJECT/'config.alternative.runtime.yml'),'--scope','pairwise-full','--exclude-tool','ridcp']
with (ROOT/'runner.log').open('ab') as log:proc=subprocess.Popen(command,cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
with (ROOT/'distillation.log').open('ab') as log:queue=subprocess.Popen([python,'-u',str(ROOT/'distill_pairwise_queue.py')],cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
meta={'pid':proc.pid,'distillation_pid':queue.pid,'command':command,'reused_cached_nodes':reused,'expected_nodes':3140,'expected_leaves':2540,'sample_count':70,'hq_count':10,'excluded_tools':['ridcp'],'source_pilot':str(OLD),'external_export':'DeepSeek aggregate statistics text only after local evaluation; no validation images','test_data_used':False,'scope':'Seven double-degradation combinations; conditional triple experiment not yet launched'}
(ROOT/'launch.json').write_text(json.dumps(meta,indent=2));time.sleep(5)
assert proc.poll() is None,(ROOT/'runner.log').read_text()[-2500:]
print(json.dumps(meta,indent=2));print((ROOT/'progress.json').read_text());print((ROOT/'distillation_progress.json').read_text())
