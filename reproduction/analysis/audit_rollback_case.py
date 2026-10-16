import json,tarfile,hashlib,csv
from pathlib import Path
from workspace_paths import workspace_root
import cv2
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
from PIL import Image,ImageDraw,ImageFont
B=workspace_root();O=B/'experiments/analysis/MAIR_vs_AgenticIR_noexp_test160_20261016';A=B/'experiments/results/AgenticIR_noexp_test160_deepseek-v4-flash-vision-exp_20261015'
sid='019__rain+blur';s=json.loads((O/'sequence_cache'/(sid+'.json')).read_text(encoding='utf8'))
branch=s['tree']['children']['deraining'];firsttool=branch['best_tool'];n=branch['tools'][firsttool]
branch2=n['children']['motion deblurring'];secondtool=branch2['best_tool'];n2=branch2['tools'][secondtool]
source='runs/'+n2['img_path'].split('/runs/')[1];dest=O/'search_trees'/('019_failed_initial_plan.png')
if not dest.exists():
 with tarfile.open(A/'agenticir_completed_verified.tar.gz','r|gz') as tf:
  for member in tf:
   if member.name.removeprefix('./')==source:dest.write_bytes(tf.extractfile(member).read());break
assert dest.exists(),source
y=cv2.imread(str(O/'inputs/test/hq/019.png'))
rows=[];paths=[O/'inputs/test/lq'/(sid+'.png'),dest,A/'restored_images'/(sid+'.png'),O/'inputs/test/hq/019.png']
for label,path in zip(['Input','Abandoned initial plan','Final after rollback','Reference'],paths):
 x=cv2.imread(str(path));assert x is not None
 if x.shape!=y.shape:x=cv2.resize(x,(y.shape[1],y.shape[0]),interpolation=cv2.INTER_CUBIC)
 rows.append(dict(label=label,psnr=float(peak_signal_noise_ratio(y,x,data_range=255)) if label!='Reference' else None,ssim=float(structural_similarity(y,x,channel_axis=2,data_range=255)) if label!='Reference' else None))
record=dict(sample_id=sid,abandoned_tools=[firsttool,secondtool],final=s['execution_path'],abandoned_source_archive_member=source,abandoned_image_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),metrics=rows,warning='Observed branch-to-final change switches order and tools simultaneously; not an isolated order effect.')
(O/'search_trees/019_rollback_quality.json').write_text(json.dumps(record,indent=2),encoding='utf8')
canvas=Image.new('RGB',(1600,560),'white');d=ImageDraw.Draw(canvas)
try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
except OSError:font=ImageFont.load_default()
for i,(path,row) in enumerate(zip(paths,rows)):
 im=Image.open(path).convert('RGB');im.thumbnail((390,420));canvas.paste(im,(i*400+(400-im.width)//2,85+(420-im.height)//2))
 d.text((i*400+8,8),row['label'],font=font,fill='black')
 if row['psnr'] is not None:d.text((i*400+8,36),f"PSNR {row['psnr']:.3f} / SSIM {row['ssim']:.4f}",font=font,fill='black')
d.text((10,520),'019 rain+blur: abandoned Derain -> Deblur vs final Deblur -> Derain; tools also changed.',font=font,fill='black')
canvas.save(O/'collages/rollback_019_rain_blur.png')
print(json.dumps(record,indent=2))
