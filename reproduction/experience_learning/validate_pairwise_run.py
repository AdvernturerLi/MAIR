"""Validate complete local exploration/evaluations before permitting distillation."""
import argparse,csv,hashlib,json,math,os
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import numpy as np
from pairwise_evidence import build

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def validate(root,approve=False):
 root=Path(root);protocol=json.loads((root/'protocol.json').read_text());progress=json.loads((root/'progress.json').read_text())
 assert progress['state']=='completed' and progress['completed_nodes']==3140 and progress['completed_leaves']==2540
 assert protocol['validation_hq_disjoint_from_test'] is True and protocol['test_data_used'] is False
 assert protocol['excluded_tools']==['ridcp'] and len(protocol['samples'])==70
 assert len({s.split('__')[0] for s in protocol['samples']})==10
 leaves=json.loads((root/'leaves.json').read_text());assert len(leaves)==2540
 expected={x['output'] for x in leaves};assert len(expected)==2540
 assert {r['sample_id'] for r in leaves}==set(protocol['samples'])
 agent=[json.loads(line) for line in (root/'agenticir/observations.jsonl').read_text().splitlines()]
 mair=json.loads((root/'mair/scored_sequences.json').read_text())
 for records in (agent,mair):
  assert len(records)==2540 and {r['output'] for r in records}==expected
  assert len({(r['sample_id'],tuple(r['plan']),tuple(r['tools'])) for r in records})==2540
  for r in records:
   assert len(r['plan'])==2 and 'ridcp' not in r['tools']
   assert all(tool in protocol['tools'][task] for task,tool in zip(r['plan'],r['tools']))
 for r in agent:
  assert set(r['failures'])==set(r['degradations'])==set(r['severity'])
  assert all(r['failures'][d]==(r['severity'][d] in ('medium','high','very high')) for d in r['degradations'])
 for r in mair:
  assert set(r['metrics'])=={'psnr','ssim','lpips','dists','maniqa','clipiqa','musiq'}
  assert all(math.isfinite(v) for v in r['metrics'].values()) and math.isfinite(r['score'])
 norm=json.loads((root/'mair/normalization.json').read_text())
 for r in mair:
  expected_score=sum((-1 if n in ('lpips','dists') else 1)*(v-norm[r['sample_id']][n]['mean'])/norm[r['sample_id']][n]['std'] if norm[r['sample_id']][n]['std']>1e-12 else 0 for n,v in r['metrics'].items())
  assert abs(expected_score-r['score'])<1e-8
 evidence={method:build(records,method) for method,records in [('agenticir',agent),('mair',mair)]}
 assert len(evidence['agenticir']['task_order_comparisons'])==7 and len(evidence['mair']['task_order_comparisons'])==3
 assert all(x['hq_count']==10 for e in evidence.values() for x in e['task_order_comparisons'].values())
 # Read and hash every committed node image; inspect dimensions without decoding GPU data.
 from PIL import Image
 nodes=list((root/'nodes').glob('*/*/node.json'));assert len(nodes)==3140
 for path in nodes:
  n=json.loads(path.read_text());assert n['status']=='success'
  image=path.parent/'image/output.png';assert sha(image)==n['output_sha256']
  with Image.open(image) as im:im.verify()
 protected=['protocol.json','leaves.json','agenticir/observations.jsonl','agenticir/fail_rate.json','mair/observations.jsonl','mair/scored_sequences.json','mair/order_scores.json','mair/normalization.json']
 report={'validation_passed':True,'checked_at_utc':datetime.now(timezone.utc).isoformat(),'nodes_checked':3140,'leaves_checked':2540,'validation_hq_count':10,'tool_images_sha256_and_readability_passed':True,'both_methods_matched_order_evidence_passed':True,'protected_sha256':{f:sha(root/f) for f in protected},'experience_review_still_required_after_distillation':True}
 (root/'validation_report.json').write_text(json.dumps(report,indent=2))
 if approve:(root/'review_approved.json').write_text(json.dumps(report,indent=2))
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--approve',action='store_true');a=p.parse_args()
 print(json.dumps(validate(a.root,a.approve),indent=2))
