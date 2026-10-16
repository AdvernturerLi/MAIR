"""Download and verify a non-disruptive incremental server snapshot."""
import hashlib,json,shlex,subprocess,tarfile,io
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
from PIL import Image
from workspace_paths import workspace_root
ROOT='/root/autodl-tmp/experience_validation_pairwise7_10hq_20261016_ridcp_excluded'
BASE=workspace_root();LOCAL=BASE/'experiments/experience/validation_pairwise7_10hq_20261016_ridcp_excluded_partial';LOCAL.mkdir(parents=True,exist_ok=True)
HISTORY=LOCAL/'history';HISTORY.mkdir(exist_ok=True)
SSH='C:/Windows/System32/OpenSSH/ssh.exe';SCP='C:/Windows/System32/OpenSSH/scp.exe'
OPTIONS=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(BASE/'MAIR/experiments/paper_audit_20261015/known_hosts').replace('\\','/'),'-i','C:/Users/pc/.ssh/id_ed25519']
index=LOCAL/'verified_index.json';temp_remote='/root/autodl-tmp/pairwise_backup_existing_index.json'
def run(args):return subprocess.run(args,check=True,capture_output=True,text=True,encoding='utf8',timeout=900).stdout
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while block:=f.read(1048576):h.update(block)
 return h.hexdigest()
# A verified index is written only after a complete, hash-checked extraction.
if index.exists():run([SCP,*OPTIONS,'-P','29119',str(index),'root@10.251.171.6:'+temp_remote])
command=['/root/miniconda3/envs/diffbir/bin/python','/root/autodl-tmp/export_completed_snapshot.py','--root',ROOT]
if index.exists():command+=['--existing-index',temp_remote]
meta=json.loads(run([SSH,*OPTIONS,'-p','29119','root@10.251.171.6',shlex.join(command)]))
archive=HISTORY/PurePosixPath(meta['archive']).name
run([SCP,*OPTIONS,'-P','29119','root@10.251.171.6:'+meta['archive'],str(archive)])
assert sha(archive)==meta['archive_sha256'],'Downloaded archive SHA mismatch'
verified={};images=0
with tarfile.open(archive,'r|gz') as tar:
 first=tar.next();assert first.name=='snapshot_manifest.json'
 manifest=json.load(tar.extractfile(first))
 for member in tar:
  if member.name=='snapshot_manifest.json':continue
  path=PurePosixPath(member.name);assert member.isfile() and not path.is_absolute() and '..' not in path.parts and not any(':' in p or '\\' in p for p in path.parts)
  data=tar.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==manifest['files'][member.name]
  if path.suffix.lower() in ('.png','.jpg','.jpeg'):
   with Image.open(io.BytesIO(data)) as image:image.verify()
   images+=1
  target=LOCAL.joinpath(*path.parts);target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists() and sha(target)==manifest['files'][member.name]:continue
  tmp=target.with_suffix(target.suffix+'.backup_tmp');tmp.write_bytes(data);tmp.replace(target)
# Confirm the full restorable snapshot, including files inherited from earlier increments.
for name,expected in manifest['all_snapshot_files'].items():
 path=LOCAL.joinpath(*PurePosixPath(name).parts);assert path.is_file() and sha(path)==expected,name
previous=json.loads(index.read_text()) if index.exists() else {};previous.update(manifest['all_snapshot_files'])
tmp=index.with_suffix('.tmp');tmp.write_text(json.dumps(previous,indent=2));tmp.replace(index)
report={**meta,'local_archive':str(archive),'downloaded_archive_sha256_passed':True,'file_sha256_passed':True,'all_snapshot_files_restorable':True,'new_images_readability_checked':images,'verified_at_utc':datetime.now(timezone.utc).isoformat()}
(HISTORY/(archive.stem+'.verification.json')).write_text(json.dumps(report,indent=2));(LOCAL/'LATEST_BACKUP.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
