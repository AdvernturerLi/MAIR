"""Export immutable successful nodes and recoverable metadata; never credentials/weights."""
import argparse,csv,hashlib,io,json,tarfile
from pathlib import Path
from datetime import datetime,timezone

def digest(data):return hashlib.sha256(data).hexdigest()
def export(root,index_path):
 root=Path(root).resolve();assert root.name=='experience_validation_pairwise7_10hq_20261016_ridcp_excluded'
 old=json.loads(Path(index_path).read_text()) if index_path else {}
 files={};inventory={};success=0;new_nodes=0
 def add(name,data):
  assert not name.startswith('/') and '..' not in Path(name).parts
  inventory[name]=digest(data)
  if old.get(name)!=inventory[name]:files[name]=data
 for rec in sorted((root/'nodes').glob('*/*/node.json')):
  n=json.loads(rec.read_text())
  if n.get('status')!='success':continue
  image=rec.parent/'image/output.png';data=image.read_bytes();assert digest(data)==n['output_sha256'],str(image)
  success+=1;name='run/'+image.relative_to(root).as_posix()
  if old.get(name)!=digest(data):new_nodes+=1
  add(name,data);add('run/'+rec.relative_to(root).as_posix(),rec.read_bytes())
  for log in rec.parent.glob('attempt_*.log'):add('run/'+log.relative_to(root).as_posix(),log.read_bytes())
 # Root source and small metadata; mutable JSON is included only when complete.
 for folder in [root,root/'agenticir',root/'mair']:
  if not folder.exists():continue
  for f in sorted(folder.iterdir()):
   if not f.is_file() or f.suffix not in ('.json','.jsonl','.py','.log'):continue
   if 'runtime_config' in f.name or '.runtime.' in f.name:continue
   data=f.read_bytes()
   if f.suffix=='.json':
    try:json.loads(data)
    except (ValueError,UnicodeDecodeError):continue
   if f.suffix=='.jsonl':
    data=data[:data.rfind(b'\n')+1]
    for line in data.splitlines():json.loads(line)
   add('run/'+f.relative_to(root).as_posix(),data)
 # Include exact source validation images required to resume on a replacement server.
 launch=json.loads((root/'launch.json').read_text());command=launch['command'];dataset=Path(command[command.index('--data')+1]).resolve()
 protocol=json.loads((root/'protocol.json').read_text());wanted=set(protocol['samples'])
 with (dataset/'validation_manifest.csv').open() as f:rows=[r for r in csv.DictReader(f) if r['sample_id'] in wanted]
 assert len(rows)==70
 for r in rows:
  for kind in ('hq','lq'):
   path=(dataset/r[kind+'_path']).resolve();assert dataset in path.parents
   name='dataset/'+path.relative_to(dataset).as_posix()
   if name in inventory:continue
   data=path.read_bytes();assert digest(data)==r[kind+'_sha256'];add(name,data)
 for name in ('validation_manifest.csv','test_manifest.csv'):add('dataset/'+name,(dataset/name).read_bytes())
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=root.parent/'experience_backups';out.mkdir(exist_ok=True)
 archive=out/('pairwise7_'+stamp+'.tar.gz')
 manifest={'created_at_utc':stamp,'run':str(root),'success_nodes_in_snapshot':success,'new_nodes_in_archive':new_nodes,'files':{name:digest(data) for name,data in files.items()},'all_snapshot_files':inventory,'excluded':'credentials, runtime configs, model weights, incomplete nodes, test images'}
 payload=json.dumps(manifest,indent=2).encode();files={'snapshot_manifest.json':payload,**files}
 with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
  for name,data in files.items():
   member=tarfile.TarInfo(name);member.size=len(data);member.mode=0o600;tar.addfile(member,io.BytesIO(data))
 h=hashlib.sha256()
 with archive.open('rb') as f:
  while block:=f.read(1048576):h.update(block)
 result={'archive':str(archive),'archive_sha256':h.hexdigest(),'archive_bytes':archive.stat().st_size,'success_nodes_in_snapshot':success,'new_nodes_in_archive':new_nodes,'created_at_utc':stamp}
 archive.with_suffix('.export.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--existing-index');a=p.parse_args();print(json.dumps(export(a.root,a.existing_index)))
