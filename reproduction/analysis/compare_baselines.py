"""Reproducible paired baseline analysis; raw results are never overwritten."""
import csv
import hashlib
import json
from pathlib import Path
from workspace_paths import workspace_root
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE=workspace_root()
RESULTS=BASE/'experiments/results'
OUT=BASE/'experiments/analysis/MAIR_vs_AgenticIR_noexp_test160_20261016'
MAIR=RESULTS/'MAIR_noexp_test160_deepseek-v4-flash-vision-exp_20261015'
AGENT=RESULTS/'AgenticIR_noexp_test160_deepseek-v4-flash-vision-exp_20261015'
COLORS={'MAIR':'#26799D','AgenticIR':'#D4783C'}
SEED=20261016

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def save_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def figure(name,fig):
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=240,bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)

def stats(rows):
    out={'n':len(rows),'hq_count':len({r['hq_id'] for r in rows})}
    for metric in ('psnr','ssim'):
        for method in ('mair','agenticir'):
            values=np.array([r[f'{method}_{metric}'] for r in rows])
            out[f'{method}_{metric}']=float(values.mean())
        delta=np.array([r[f'delta_{metric}'] for r in rows])
        out[f'delta_{metric}']=float(delta.mean())
        out[f'mair_wins_{metric}']=int((delta>1e-9).sum())
        out[f'agenticir_wins_{metric}']=int((delta< -1e-9).sum())
        out[f'ties_{metric}']=int((abs(delta)<=1e-9).sum())
        # Cluster bootstrap over source HQ images, retaining their degradations.
        clusters=sorted({r['hq_id'] for r in rows})
        totals=np.array([sum(r[f'delta_{metric}'] for r in rows if r['hq_id']==h) for h in clusters])
        counts=np.array([sum(r['hq_id']==h for r in rows) for h in clusters])
        rng=np.random.default_rng(SEED)
        index=rng.integers(0,len(clusters),size=(10000,len(clusters)))
        samples=totals[index].sum(1)/counts[index].sum(1)
        lo,hi=np.quantile(samples,[.025,.975])
        out[f'delta_{metric}_ci_low']=float(lo);out[f'delta_{metric}_ci_high']=float(hi)
    return out

def crop_roi(reference):
    h,w=reference.shape[:2]
    side=min(128,h,w)
    gray=cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY)
    candidates=[]
    for fy in (.25,.5,.75):
        for fx in (.25,.5,.75):
            x=max(0,min(w-side,int(w*fx-side/2)));y=max(0,min(h-side,int(h*fy-side/2)))
            score=float(cv2.Laplacian(gray[y:y+side,x:x+side],cv2.CV_32F).var())
            candidates.append((score,x,y,side))
    _,x,y,side=max(candidates)
    return x,y,side,side

def montage(row,reason,manifest):
    key=row['sample_id'];data=manifest[key]
    paths=[OUT/'inputs'/data['lq_path'],MAIR/'restored_images'/f'{key}.png',
           AGENT/'restored_images'/f'{key}.png',OUT/'inputs'/data['hq_path']]
    arrays=[cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB) for p in paths]
    gt=arrays[-1];h,w=gt.shape[:2];roi=crop_roi(gt);x,y,rw,rh=roi
    arrays=[cv2.resize(a,(w,h),interpolation=cv2.INTER_CUBIC) if a.shape[:2]!=(h,w) else a for a in arrays]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    bold=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',23)
    canvas=Image.new('RGB',(1480,620),'white');draw=ImageDraw.Draw(canvas)
    draw.text((20,12),f'{key} | {reason}',font=bold,fill='#172B3A')
    labels=['Input (LQ)','MAIR','AgenticIR','Reference (HQ)']
    for i,(label,arr) in enumerate(zip(labels,arrays)):
        left=20+i*365
        draw.text((left,50),label,font=bold,fill=COLORS.get(label,'#172B3A'))
        if i in (1,2):
            method=('mair','agenticir')[i-1]
            note=f"PSNR {row[method+'_psnr']:.2f} dB / SSIM {row[method+'_ssim']:.3f}"
            if row[method+'_resized']:note+=' [bicubic]'
            draw.text((left,80),note,font=small,fill='#334155')
        im=Image.fromarray(arr);im.thumbnail((345,250),Image.Resampling.LANCZOS)
        ox=left+(345-im.width)//2;oy=110+(250-im.height)//2
        canvas.paste(im,(ox,oy))
        sx=im.width/w;sy=im.height/h
        draw.rectangle((ox+x*sx,oy+y*sy,ox+(x+rw)*sx,oy+(y+rh)*sy),outline='#D62839',width=2)
        patch=Image.fromarray(arr[y:y+rh,x:x+rw]).resize((210,210),Image.Resampling.NEAREST)
        canvas.paste(patch,(left+67,382))
        draw.rectangle((left+66,381,left+278,593),outline='#D62839',width=2)
    draw.text((20,600),'ROI chosen using HQ edge energy only. Display interpolation does not modify saved outputs.',font=small,fill='#475569')
    name=f'{len(SELECTION):02d}_{key}'
    canvas.save(OUT/'collages'/f'{name}.png')
    canvas.save(OUT/'collages'/f'{name}.pdf',resolution=150)
    SELECTION.append({'sample_id':key,'selection_reason':reason,'roi_xywh':list(roi),
                      'delta_psnr':row['delta_psnr'],'delta_ssim':row['delta_ssim'],
                      'file':f'collages/{name}.png'})

SELECTION=[]

def main():
    for folder in ('figures','collages','tables'):(OUT/folder).mkdir(parents=True,exist_ok=True)
    manifest={r['sample_id']:r for r in read_csv(MAIR/'manifest.csv')}
    assert hashlib.sha256((MAIR/'manifest.csv').read_bytes()).digest()==hashlib.sha256((AGENT/'manifest.csv').read_bytes()).digest()
    a={r['sample_id']:r for r in read_csv(MAIR/'metrics.csv')}
    b={r['sample_id']:r for r in read_csv(AGENT/'metrics.csv')}
    assert set(a)==set(b)==set(manifest) and len(a)==160
    rows=[]
    for key in manifest:
        assert a[key]['status']==b[key]['status']=='success'
        r={'sample_id':key,'hq_id':manifest[key]['hq_id'],'degradation':key.split('__',1)[1]}
        for method,source in (('mair',a[key]),('agenticir',b[key])):
            for field in ('psnr','ssim','elapsed_s','tool_calls','tool_errors'):
                r[f'{method}_{field}']=float(source[field])
            r[f'{method}_resized']=source['metric_resized'].lower()=='true'
        for metric in ('psnr','ssim'):r[f'delta_{metric}']=r[f'mair_{metric}']-r[f'agenticir_{metric}']
        rows.append(r)
    assert len({r['hq_id'] for r in rows})==20
    save_csv(OUT/'tables/paired_samples.csv',rows)
    groups={name:[r for r in rows if r['degradation']==name] for name in dict.fromkeys(r['degradation'] for r in rows)}
    by_group=[{'degradation':name,**stats(rr)} for name,rr in groups.items()]
    save_csv(OUT/'tables/by_degradation.csv',by_group)
    subsets={'all':rows,'both_native_size':[r for r in rows if not r['mair_resized'] and not r['agenticir_resized']],
             'any_bicubic':[r for r in rows if r['mair_resized'] or r['agenticir_resized']]}
    sensitivity=[{'subset':name,**stats(rr)} for name,rr in subsets.items() if rr]
    save_csv(OUT/'tables/size_sensitivity.csv',sensitivity)
    overall=stats(rows)
    costs=[]
    for method in ('mair','agenticir'):
        rr=np.array([r[f'{method}_elapsed_s'] for r in rows])
        calls=np.array([r[f'{method}_tool_calls'] for r in rows])
        costs.append({'method':method,'successful_records_total_s':float(rr.sum()),
                      'mean_s':float(rr.mean()),'median_s':float(np.median(rr)),
                      'p90_s':float(np.quantile(rr,.9)),'mean_tool_calls':float(calls.mean()),
                      'recorded_tool_errors':int(sum(r[f'{method}_tool_errors'] for r in rows))})
    save_csv(OUT/'tables/runtime_and_tools.csv',costs)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'figure.facecolor':'white'})
    names=list(groups);labels=[n.replace('+',' +\n') for n in names];x=np.arange(len(names));width=.36
    fig,axes=plt.subplots(1,2,figsize=(14,4.7),layout='constrained')
    for ax,metric in zip(axes,('psnr','ssim')):
        for offset,method,label in ((-.5,'mair','MAIR'),(.5,'agenticir','AgenticIR')):
            ax.bar(x+offset*width,[g[f'{method}_{metric}'] for g in by_group],width,label=label,color=COLORS[label])
        ax.set_xticks(x,labels,rotation=35,ha='right',fontsize=8)
        ax.set_ylabel('PSNR (dB)' if metric=='psnr' else 'SSIM');ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True);ax.legend(frameon=False)
    fig.suptitle('No-experience baselines: 160 paired images (20 source images)')
    figure('01_by_degradation',fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    palette=plt.get_cmap('tab10')
    for ax,metric in zip(axes,('psnr','ssim')):
        for i,(name,rr) in enumerate(groups.items()):
            ax.scatter([r[f'agenticir_{metric}'] for r in rr],[r[f'mair_{metric}'] for r in rr],s=22,alpha=.7,color=palette(i),label=name)
        values=[r[f'{m}_{metric}'] for r in rows for m in ('mair','agenticir')]
        lo,hi=min(values),max(values);pad=(hi-lo)*.05
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'--',color='#64748B',lw=1)
        ax.set_xlim(lo-pad,hi+pad);ax.set_ylim(lo-pad,hi+pad)
        ax.set_xlabel(f'AgenticIR {metric.upper()}');ax.set_ylabel(f'MAIR {metric.upper()}');ax.grid(alpha=.15)
    axes[1].legend(loc='lower right',fontsize=6,frameon=True)
    fig.suptitle('Each point is one paired test sample; dashed line indicates equality')
    figure('02_paired_scatter',fig)
    fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
    for ax,metric in zip(axes,('psnr','ssim')):
        means=np.array([g[f'delta_{metric}'] for g in by_group])
        lower=np.array([g[f'delta_{metric}_ci_low'] for g in by_group]);upper=np.array([g[f'delta_{metric}_ci_high'] for g in by_group])
        ax.errorbar(x,means,yerr=np.maximum(np.vstack((means-lower,upper-means)),0),fmt='o',color=COLORS['MAIR'],capsize=4)
        ax.axhline(0,color='#64748B',ls='--',lw=1)
        ax.set_xticks(x,[n.replace('+',' + ') for n in names],rotation=15,ha='right',fontsize=8)
        ax.set_ylabel('MAIR - AgenticIR\n'+('PSNR (dB)' if metric=='psnr' else 'SSIM'));ax.grid(axis='y',alpha=.15)
    fig.suptitle('Paired mean differences with 95% source-image cluster bootstrap intervals')
    figure('03_paired_delta_ci',fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
    for ax,field,label in zip(axes,('elapsed_s','tool_calls'),('Successful-record time (s, log scale)','Tool invocations')):
        values=[[r[f'{method}_{field}'] for r in rows] for method in ('mair','agenticir')]
        box=ax.boxplot(values,tick_labels=['MAIR','AgenticIR'],patch_artist=True,showfliers=True)
        for patch,color in zip(box['boxes'],COLORS.values()):patch.set_facecolor(color);patch.set_alpha(.65)
        if field=='elapsed_s':ax.set_yscale('log')
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Deployment cost; includes service loading and excludes earlier interrupted attempts')
    figure('04_runtime_and_tools',fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,metric in zip(axes,('psnr','ssim')):
        for offset,method,label in ((-.5,'mair','MAIR'),(.5,'agenticir','AgenticIR')):
            ax.bar(np.arange(len(sensitivity))+offset*width,[s[f'{method}_{metric}'] for s in sensitivity],width,color=COLORS[label],label=label)
        ax.set_xticks(np.arange(len(sensitivity)),[f"{s['subset']}\nn={s['n']}" for s in sensitivity],fontsize=8)
        ax.set_ylabel(metric.upper());ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Sensitivity to output resizing; native subset is diagnostic, not a replacement benchmark')
    figure('05_size_sensitivity',fig)
    for name,rr in groups.items():
        median=float(np.median([r['delta_psnr'] for r in rr]))
        chosen=min(rr,key=lambda r:(abs(r['delta_psnr']-median),r['sample_id']))
        montage(chosen,'nearest group median PSNR difference',manifest)
    selected={r['sample_id'] for r in SELECTION}
    for reverse,label in ((True,'largest MAIR PSNR advantage'),(False,'largest AgenticIR PSNR advantage')):
        for row in sorted(rows,key=lambda r:r['delta_psnr'],reverse=reverse):
            if row['sample_id'] not in selected:
                montage(row,label,manifest);selected.add(row['sample_id']);break
    (OUT/'collage_selection.json').write_text(json.dumps(SELECTION,indent=2))
    report={'overall':overall,'by_degradation':by_group,'size_sensitivity':sensitivity,'runtime':costs,
            'bootstrap':{'unit':'source HQ image','clusters':20,'replicates':10000,'seed':SEED},
            'source_files':{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (MAIR/'metrics.csv',AGENT/'metrics.csv',MAIR/'manifest.csv')},
            'collages':SELECTION}
    (OUT/'analysis_summary.json').write_text(json.dumps(report,indent=2))
    blocks=['<html><meta charset="utf-8"><title>MAIR vs AgenticIR</title><style>body{font:16px Arial;max-width:1500px;margin:30px auto;background:#f4f6f8}img{width:100%;background:white;margin-bottom:25px}h2{margin-top:35px}</style><h1>MAIR vs AgenticIR: no-experience test160</h1><p>Positive deltas favor MAIR. Selection rules and ROI coordinates are recorded in collage_selection.json.</p>']
    for path in sorted((OUT/'figures').glob('*.png')):blocks.append(f'<h2>{path.stem}</h2><img src="figures/{path.name}">')
    for row in SELECTION:blocks.append(f'<h2>{row["sample_id"]}</h2><p>{row["selection_reason"]}</p><img src="{row["file"]}">')
    blocks.append('</html>');(OUT/'gallery.html').write_text('\n'.join(blocks))
    print(json.dumps({'overall':overall,'size_sensitivity':sensitivity,'costs':costs,'figures':5,'collages':len(SELECTION)},indent=2))

if __name__=='__main__':main()
