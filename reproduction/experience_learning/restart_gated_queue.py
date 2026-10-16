import json,os,shutil,subprocess
from pathlib import Path
BASE=Path('/root/autodl-tmp');ROOT=BASE/'experience_validation_pairwise7_10hq_20261016_ridcp_excluded'
p=Path('/proc/17915/cmdline')
if p.exists():
 assert 'distill_pairwise_queue.py' in p.read_bytes().decode();os.kill(17915,15)
shutil.copy2(BASE/'distill_pairwise_queue.py',ROOT/'distill_pairwise_queue.py')
shutil.copy2(BASE/'validate_pairwise_run.py',ROOT/'validate_pairwise_run.py')
with (ROOT/'distillation.log').open('ab') as log:
 proc=subprocess.Popen(['/root/miniconda3/envs/diffbir/bin/python','-u',str(ROOT/'distill_pairwise_queue.py')],cwd=BASE/'AgenticIR_mairtools_20261015',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(ROOT/'review_gate_revision.json').write_text(json.dumps({'distillation_pid':proc.pid,'change':'Require passing complete-image/metric validation and protected input hashes before any external text distillation','exploration_untouched':True},indent=2));print(proc.pid)
