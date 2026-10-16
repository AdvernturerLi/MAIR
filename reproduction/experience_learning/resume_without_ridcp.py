import json,os,shutil,subprocess,time
from pathlib import Path
BASE=Path('/root/autodl-tmp');old=BASE/'experience_validation_pilot_20261016';new=BASE/'experience_validation_pilot_20261016_ridcp_excluded'
assert json.loads((old/'progress.json').read_text())['state']=='needs_attention'
assert not (new/'launch.json').exists(),'Already launched'
new.mkdir(exist_ok=True)
shutil.copy2(BASE/'pilot_experience_v2.py',new/'pilot.py')
reused=0
for record in (old/'nodes').glob('*/*/node.json'):
    data=json.loads(record.read_text())
    if data.get('status')!='success' or any(step[1]=='ridcp' for step in data['path']):continue
    target=new/'nodes'/record.parent.relative_to(old/'nodes')
    shutil.copytree(record.parent,target,dirs_exist_ok=True);reused+=1
# Stop only the two old waiting queues after confirming their exact command lines.
for pid,script in ((8423,'local_distill.py'),(9722,'distill_deepseek_queue.py')):
    p=Path('/proc')/str(pid)/'cmdline'
    if p.exists() and script in p.read_bytes().decode():os.kill(pid,15)
python='/root/miniconda3/envs/diffbir/bin/python';project=BASE/'AgenticIR_mairtools_20261015'
env=dict(os.environ)
for key in ('OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL','PYTHONPATH'):env.pop(key,None)
env['TZ']='Asia/Shanghai';env['EXPERIENCE_RUN']=str(new)
command=[python,'-u',str(new/'pilot.py'),'run','--project',str(project),'--data',str(BASE/'MAIR/experiments/datasets/mair_benchmark_v1'),'--output',str(new),'--config',str(project/'config.alternative.runtime.yml'),'--exclude-tool','ridcp']
with (new/'runner.log').open('ab') as log:
    proc=subprocess.Popen(command,cwd=project,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
with (new/'deepseek_distillation.log').open('ab') as log:
    queue=subprocess.Popen([python,'-u',str(BASE/'distill_deepseek_queue.py')],cwd=project,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
meta={'pid':proc.pid,'deepseek_queue_pid':queue.pid,'command':command,'reused_success_nodes':reused,'excluded_tool':'ridcp','reason':'Runtime unavailable: missing deform_conv_ext; weight presence was insufficient availability check','previous_run':str(old),'baseline_results_modified':False,'remaining_work':'Full shared-profile experience/no-experience comparisons must use the same tool exclusions.'}
(new/'launch.json').write_text(json.dumps(meta,indent=2));(old/'successor_run.json').write_text(json.dumps({'successor':str(new),'reason':meta['reason']},indent=2))
time.sleep(4)
assert proc.poll() is None,(new/'runner.log').read_text()[-2000:]
print(json.dumps(meta,indent=2));print((new/'progress.json').read_text())
