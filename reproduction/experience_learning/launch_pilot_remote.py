import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

base=Path('/root/autodl-tmp');out=base/'experience_validation_pilot_20261016'
out.mkdir(exist_ok=True)
project=base/'AgenticIR_mairtools_20261015'
python='/root/miniconda3/envs/diffbir/bin/python'
with socket.socket() as sock:
    assert sock.connect_ex(('127.0.0.1',5001))==0,'DepictQA eval service required'
assert not (out/'launch.json').exists(),'Refusing duplicate launch'
environment=dict(os.environ)
for key in ('OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL','PYTHONPATH'):
    environment.pop(key,None)
environment['TZ']='Asia/Shanghai'
source=base/'pilot_experience.py'
frozen=out/'pilot.py';shutil.copy2(source,frozen)
command=[python,'-u',str(frozen),'run','--project',str(project),
    '--data',str(base/'MAIR/experiments/datasets/mair_benchmark_v1'),
    '--output',str(out),'--config',str(project/'config.alternative.runtime.yml')]
with (out/'runner.log').open('ab') as log:
    process=subprocess.Popen(command,cwd=project,env=environment,stdin=subprocess.DEVNULL,
                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
with (out/'iqa_preparation.log').open('ab') as log:
    warmer=subprocess.Popen([python,'-u',str(base/'warm_iqa.py')],cwd=project,env=environment,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'launch.json').write_text(json.dumps({'pid':process.pid,'command':command,'iqa_preparation_pid':warmer.pid},indent=2))
time.sleep(4)
assert process.poll() is None,(out/'runner.log').read_text()[-3000:]
print((out/'progress.json').read_text())
print('IQA_PREPARATION_PID',warmer.pid)
