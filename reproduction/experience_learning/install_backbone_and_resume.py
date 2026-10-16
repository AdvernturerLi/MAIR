import gc,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
BASE=Path('/root/autodl-tmp');RUN=BASE/'experience_validation_pilot_20261016_ridcp_excluded';SRC=BASE/'maniqa_backbone'
meta=json.loads((SRC/'manifest.json').read_text())
assert meta['repo_id']=='timm/vit_base_patch8_224.augreg2_in21k_ft_in1k'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while chunk:=f.read(1024*1024):h.update(chunk)
 return h.hexdigest()
source=SRC/meta['filename'];assert sha(source)==meta['sha256']
cache=Path('/root/.cache/huggingface/hub')/('models--'+meta['repo_id'].replace('/','--'))
blob=cache/'blobs'/meta['sha256'];blob.parent.mkdir(parents=True,exist_ok=True)
if not blob.exists():shutil.copy2(source,blob)
assert sha(blob)==meta['sha256']
snapshot=cache/'snapshots'/meta['commit']/meta['filename'];snapshot.parent.mkdir(parents=True,exist_ok=True)
if not snapshot.exists():snapshot.symlink_to(Path('../../blobs')/meta['sha256'])
assert sha(snapshot)==meta['sha256']
refs=cache/'refs/main';refs.parent.mkdir(parents=True,exist_ok=True);refs.write_text(meta['commit'])
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import pyiqa
report={'state':'checking','backbone':meta,'models':{}}
for name in ('lpips','dists','maniqa','clipiqa','musiq'):
 try:
  model=pyiqa.create_metric(name,device='cpu');report['models'][name]='offline_load_passed';del model;gc.collect()
 except Exception as error:
  report['models'][name]=f'{type(error).__name__}: {error}';report['state']='needs_attention';(RUN/'offline_iqa_verification.json').write_text(json.dumps(report,indent=2));raise
 (RUN/'offline_iqa_verification.json').write_text(json.dumps(report,indent=2))
report['state']='all_offline_loads_passed';(RUN/'offline_iqa_verification.json').write_text(json.dumps(report,indent=2))
old=json.loads((RUN/'progress.json').read_text());assert old['state']=='needs_attention'
pid=old['pid'];assert not Path('/proc',str(pid)).exists(),'Old exploration process still alive'
(RUN/'interruption_maniqa_missing_backbone.json').write_text(json.dumps(old,indent=2))
launch=json.loads((RUN/'launch.json').read_text());env=dict(os.environ);env['TZ']='Asia/Shanghai'
for key in ('OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL','PYTHONPATH'):env.pop(key,None)
with (RUN/'runner.log').open('ab') as log:
 proc=subprocess.Popen(launch['command'],cwd=BASE/'AgenticIR_mairtools_20261015',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
recovery={'pid':proc.pid,'reused_exploration_nodes':178,'reused_agenticir_observations':len((RUN/'agenticir/observations.jsonl').read_text().splitlines()),'change':'Added exact official timm pretrained backbone to offline cache; no tool/scoring/prompt changes','backbone_sha256':meta['sha256']}
(RUN/'recovery_maniqa_backbone.json').write_text(json.dumps(recovery,indent=2));time.sleep(4)
assert proc.poll() is None,(RUN/'runner.log').read_text()[-2000:]
print(json.dumps(recovery,indent=2));print((RUN/'progress.json').read_text())
