"""Wait for local aggregate statistics, then distill text via authorized DeepSeek API."""
import hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(os.environ.get('EXPERIENCE_RUN','/root/autodl-tmp/experience_validation_pilot_20261016'))
PROJECT=Path('/root/autodl-tmp/AgenticIR_mairtools_20261015')
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(ROOT))
from pilot import Pilot
import yaml
config=PROJECT/'config.alternative.runtime.yml'
cfg=yaml.safe_load(config.read_text())
assert cfg.get('OPENAI_BASE_URL','').rstrip('/') in ('https://api.deepseek.com','https://api.deepseek.com/v1'),'Expected authorized DeepSeek endpoint'
assert cfg.get('OPENAI_API_KEY'),'Missing configured API key'
del cfg
state={'pid':os.getpid(),'state':'waiting_for_all_local_statistics','methods':{},'export':'aggregate statistics text only; no images','authorization':'User explicitly requested DeepSeek API for distillation after clarifying exploration requires evaluation models only.'}
def progress(**kw):
    state.update(kw);tmp=ROOT/'deepseek_distillation_progress.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(ROOT/'deepseek_distillation_progress.json')
progress()
files={'agenticir':ROOT/'agenticir/fail_rate.json','mair':ROOT/'mair/order_scores.json'}
# Freeze both local evaluation outputs before either distillation begins.
while not all(p.exists() for p in files.values()):time.sleep(30)
pilot=object.__new__(Pilot);pilot.root=ROOT
pilot.args=SimpleNamespace(config=PROJECT/'config.alternative.runtime.yml')
pilot.progress=progress
try:
    for method,source in files.items():
        if (ROOT/method/'schedule_experience.json').exists():
            state['methods'][method]='already_present';continue
        before=hashlib.sha256(source.read_bytes()).hexdigest()
        pilot.distill(method,json.loads(source.read_text()))
        assert hashlib.sha256(source.read_bytes()).hexdigest()==before,'Statistics changed during distillation'
        state['methods'][method]='completed';progress()
    progress(state='completed',phase='completed')
except BaseException as error:
    progress(state='needs_attention',error=f'{type(error).__name__}: {error}');raise
