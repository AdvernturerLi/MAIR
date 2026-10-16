"""Download official public weights locally; no experiment data is transmitted."""
import urllib.request,hashlib,json,time
from pathlib import Path
from workspace_paths import workspace_root
from concurrent.futures import ThreadPoolExecutor
OUT=workspace_root()/'experiments/model_cache/pyiqa';OUT.mkdir(parents=True,exist_ok=True)
names=['LPIPS_v0.1_alex-df73285e.pth','LPIPS_v0.1_vgg-a78928a0.pth','DISTS_weights-f5e65c96.pth','ckpt_koniq10k.pt','musiq_koniq_ckpt-e95806b9.pth']
urls=['https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/'+n for n in names]
urls+=['https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt']
def get(url):
    path=OUT/url.rsplit('/',1)[1]
    try:
        if not path.exists():
            tmp=path.with_suffix(path.suffix+'.partial')
            with urllib.request.urlopen(url,timeout=50) as r,tmp.open('wb') as f:
                while chunk:=r.read(1024*1024):f.write(chunk)
            tmp.replace(path)
        h=hashlib.sha256()
        with path.open('rb') as f:
            while b:=f.read(1024*1024):h.update(b)
        result=dict(file=path.name,bytes=path.stat().st_size,sha256=h.hexdigest(),source=url,status='downloaded')
    except Exception as e:result=dict(source=url,status='error',error=str(e))
    print(json.dumps(result),flush=True);return result
with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(get,urls))
(OUT/'download_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf8')
