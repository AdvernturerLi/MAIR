"""Build portable metric tables and validate final paired dataset coverage."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
from collections import defaultdict

root=Path(__file__).resolve().parents[1]/'experiments/results/AgenticIR_noexp_test160_deepseek-v4-flash-vision-exp_20261015'
verification=json.loads((root/'LOCAL_VERIFICATION.json').read_text())
assert verification['status']=='verified' and verification['completed_images']==160
with (root/'metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
    rows=list(csv.DictReader(stream))
with (root/'manifest.csv').open(newline='') as stream:
    manifest=list(csv.DictReader(stream))
assert len(rows)==len(manifest)==160
assert {r['sample_id'] for r in rows}=={r['sample_id'] for r in manifest}
assert all(r['status']=='success' and (root/r['output']).is_file() for r in rows)
summary=json.loads((root/'RESULT_SUMMARY.json').read_text())
assert abs(statistics.mean(float(r['psnr']) for r in rows)-summary['mean_rgb_psnr'])<1e-10
assert abs(statistics.mean(float(r['ssim']) for r in rows)-summary['mean_rgb_ssim'])<1e-10
groups=defaultdict(list)
for row in rows:groups[row['sample_id'].split('__',1)[1]].append(row)
assert len(groups)==8 and all(len(values)==20 for values in groups.values())
with (root/'metrics_by_degradation.csv').open('w',encoding='utf-8-sig',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=['degradation','count','mean_rgb_psnr','mean_rgb_ssim','metric_resized_count'])
    writer.writeheader()
    for name, values in sorted(groups.items()):
        writer.writerow({'degradation':name,'count':len(values),
            'mean_rgb_psnr':statistics.mean(float(r['psnr']) for r in values),
            'mean_rgb_ssim':statistics.mean(float(r['ssim']) for r in values),
            'metric_resized_count':sum(r['metric_resized']=='True' for r in values)})
mair=root.parent/'MAIR_noexp_test160_deepseek-v4-flash-vision-exp_20261015'
assert hashlib.sha256((root/'manifest.csv').read_bytes()).hexdigest()==hashlib.sha256((mair/'manifest.csv').read_bytes()).hexdigest()
verification.update(final_experiment_complete=True,manifest_images=160,metric_rows=160,
                    degradation_groups=8,paired_mair_manifest_identical=True)
(root/'LOCAL_VERIFICATION.json').write_text(json.dumps(verification,indent=2))
print(json.dumps({'status':'finalized','images':160,'metric_rows':160,
    'mean_psnr':summary['mean_rgb_psnr'],'mean_ssim':summary['mean_rgb_ssim'],
    'verified_archive_files':verification['verified_files_in_archive']},indent=2))
