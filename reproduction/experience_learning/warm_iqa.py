"""Prepare mandatory IQA model weights without competing for restoration GPU."""
import gc
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='60'
import pyiqa

out=Path('/root/autodl-tmp/experience_validation_pilot_20261016')
out.mkdir(exist_ok=True)
report={'pyiqa_version':pyiqa.__version__,'models':{},'state':'running'}
for name in ('lpips','dists','maniqa','clipiqa','musiq'):
    start=time.monotonic()
    print('PREPARING',name,flush=True)
    try:
        model=pyiqa.create_metric(name,device='cpu')
        report['models'][name]={'status':'ready','lower_better':bool(model.lower_better),'elapsed_s':round(time.monotonic()-start,2)}
        del model;gc.collect()
    except Exception as error:
        report['models'][name]={'status':'error','error':f'{type(error).__name__}: {error}'}
        print('MODEL_ERROR',name,type(error).__name__,str(error),flush=True)
    (out/'iqa_preparation.json').write_text(json.dumps(report,indent=2))
report['state']='ready' if all(x['status']=='ready' for x in report['models'].values()) else 'needs_attention'
(out/'iqa_preparation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
