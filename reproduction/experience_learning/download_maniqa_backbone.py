"""Fetch the exact timm backbone requested by installed MANIQA."""
import urllib.request,json,hashlib
from pathlib import Path
from workspace_paths import workspace_root
repo='timm/vit_base_patch8_224.augreg2_in21k_ft_in1k';name='pytorch_model.bin'
out=workspace_root()/'experiments/model_cache/maniqa_backbone';out.mkdir(parents=True,exist_ok=True)
def read(url):
 with urllib.request.urlopen(url,timeout=60) as r:return json.load(r)
meta=read('https://huggingface.co/api/models/'+repo)
commit=meta['sha'];tree=read('https://huggingface.co/api/models/'+repo+'/tree/'+commit)
entry=next(x for x in tree if x['path']==name)
expected=entry['lfs']['oid'];url='https://huggingface.co/'+repo+'/resolve/'+commit+'/'+name
path=out/name
if not path.exists():
 with urllib.request.urlopen(url,timeout=60) as r,path.with_suffix('.partial').open('wb') as f:
  while b:=r.read(1024*1024):f.write(b)
 path.with_suffix('.partial').replace(path)
h=hashlib.sha256()
with path.open('rb') as f:
 while b:=f.read(1024*1024):h.update(b)
assert h.hexdigest()==expected,(h.hexdigest(),expected)
record=dict(repo_id=repo,commit=commit,filename=name,sha256=expected,bytes=path.stat().st_size,source=url)
(out/'manifest.json').write_text(json.dumps(record,indent=2),encoding='utf8');print(json.dumps(record),flush=True)
