"""Controlled validation-only exploration, separately evaluated for both methods.

This pilot is not the authors' five-dataset / 13000-sequence experiment.
"""
import argparse
import atexit
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

SEED=20261016
MAPPING={'low resolution':'super-resolution','noise':'denoising','motion blur':'motion deblurring',
 'defocus blur':'defocus deblurring','rain':'deraining','haze':'dehazing','dark':'brightening',
 'jpeg compression artifact':'jpeg compression artifact removal'}
STAGE={'super-resolution':2,'denoising':2,'motion deblurring':2,'defocus deblurring':2,
 'deraining':3,'dehazing':3,'brightening':3,'jpeg compression artifact removal':1}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def load_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def count_nodes(tasks,tools):
    return sum(len(tools[s])*(1+count_nodes([t for t in tasks if t!=s],tools)) for s in tasks) if tasks else 0

def count_leaves(tasks,tools):
    return math.factorial(len(tasks))*math.prod(len(tools[s]) for s in tasks)

def worker(args):
    os.chdir(args.project);sys.path.insert(0,str(args.project))
    from executor import executor
    tool=next(t for t in executor.toolbox_router[args.subtask] if t.tool_name==args.tool)
    assert tool.available(),f'Missing {args.subtask}/{args.tool}'
    tool(args.input,args.output,silent=False)

class Pilot:
    def __init__(self,args):
        self.args=args;self.root=args.output;self.root.mkdir(parents=True,exist_ok=True)
        os.chdir(args.project);sys.path.insert(0,str(args.project))
        self.tools=json.loads((args.project/'tool_alignment.json').read_text())['tools']
        self.excluded_tools=sorted(set(getattr(args,'exclude_tool',[])))
        for task,tools in self.tools.items():
            self.tools[task]=[tool for tool in tools if tool not in self.excluded_tools]
            assert self.tools[task],f'No usable tools for {task}'
        validation=load_csv(args.data/'validation_manifest.csv');test=load_csv(args.data/'test_manifest.csv')
        assert not {r['hq_id'] for r in validation}&{r['hq_id'] for r in test}
        assert not {r['hq_sha256'] for r in validation}&{r['hq_sha256'] for r in test}
        self.scope=getattr(args,'scope','pilot')
        if self.scope=='pilot':
            self.rows=[r for r in validation if r['hq_id'] in ('084','054') and r['sample_id'].split('__',1)[1] in ('noise+blur','rain+haze')]
            assert len(self.rows)==4
        else:
            self.rows=[r for r in validation if len(json.loads(r['degradation_sequence']))==2]
            assert len(self.rows)==70 and len({r['hq_id'] for r in self.rows})==10
        assert all(r['split']=='validation' for r in self.rows)
        for r in self.rows:
            for kind in ('hq','lq'):assert sha(args.data/r[kind+'_path'])==r[kind+'_sha256']
        self.total_nodes=sum(count_nodes([MAPPING[d] for d in json.loads(r['degradation_sequence'])],self.tools) for r in self.rows)
        self.total_leaves=sum(count_leaves([MAPPING[d] for d in json.loads(r['degradation_sequence'])],self.tools) for r in self.rows)
        if self.scope=='pilot' and not self.excluded_tools:assert self.total_nodes==196 and self.total_leaves==160
        if self.scope=='pairwise-full' and self.excluded_tools==['ridcp']:assert self.total_nodes==3140 and self.total_leaves==2540
        plan={'kind':'controlled_validation_'+self.scope,'test_data_used':False,'samples':[r['sample_id'] for r in self.rows],
          'validation_hq_disjoint_from_test':True,'expected_tool_invocations':self.total_nodes,'expected_leaves':self.total_leaves,
          'full_validation_expected_tool_invocations':sum(count_nodes([MAPPING[d] for d in json.loads(r['degradation_sequence'])],self.tools) for r in validation),
          'tools':self.tools,'excluded_tools':self.excluded_tools,'tool_profile_sha256':sha(args.project/'tool_alignment.json'),
          'script_sha256':sha(Path(__file__)),'data_resize':'none during exploration; INTER_CUBIC to HQ only for IQA if needed',
          'label_source':'Known synthetic validation degradation labels, not online coarse perception',
          'agenticir_evaluation':'DepictQA residual severity >= medium counted as failure',
          'mair_evaluation':'Seven metrics, signed within-image population z-scores summed; constant metrics contribute zero',
          'normalization_scope_note':'Within-image candidate pool is an explicit implementation choice; authors exact normalization pool unavailable',
          'scope':('Only noise+motion-blur and rain+haze on two validation HQ images; preliminary experience' if self.scope=='pilot' else 'All seven existing synthetic validation pairs on ten HQ sources; not the authors real-world five-dataset reproduction'),
          'seed':SEED,'scope_id':self.scope,'hq_count':len({r['hq_id'] for r in self.rows}),'sample_count':len(self.rows),'coverage':[list(x) for x in sorted({tuple(json.loads(r['degradation_sequence'])) for r in self.rows})]}
        plan['external_distillation_enabled']=bool(args.distill)
        if (self.root/'protocol.json').exists():
            assert json.loads((self.root/'protocol.json').read_text())==plan,'Resume source or protocol changed'
        else:write(self.root/'protocol.json',plan)
        self.nodes_done=0;self.leaves=[];self.child=None;self.started=time.monotonic()
        self.state={'state':'running','pid':os.getpid(),'phase':'exploration','expected_nodes':self.total_nodes,'expected_leaves':self.total_leaves}
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)

    def cleanup(self):
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid,signal.SIGTERM)
            try:self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(self.child.pid,signal.SIGKILL);self.child.wait()

    def stop(self,*args):
        self.cleanup();raise KeyboardInterrupt('Pilot interrupted; successful cached nodes retained')

    def progress(self,**kw):
        self.state.update(kw,completed_nodes=self.nodes_done,elapsed_s=round(time.monotonic()-self.started,1),
                          updated_at_utc=datetime.now(timezone.utc).isoformat())
        write(self.root/'progress.json',self.state)

    def tree(self,row,tasks,prefix,input_dir):
        if not tasks:
            output=input_dir/'output.png'
            self.leaves.append({'sample_id':row['sample_id'],'degradations':json.loads(row['degradation_sequence']),
              'plan':[step[0] for step in prefix],'tools':[step[1] for step in prefix],
              'output':str(output),'hq':str(self.args.data/row['hq_path'])})
            return
        for task in tasks:
            for tool in self.tools[task]:
                path=prefix+[(task,tool)]
                key=hashlib.sha256(json.dumps(path).encode()).hexdigest()[:20]
                node=self.root/'nodes'/row['sample_id']/key
                node.mkdir(parents=True,exist_ok=True)
                output=node/'image';output.mkdir(exist_ok=True)
                record=node/'node.json'
                cached=json.loads(record.read_text()) if record.exists() else {}
                valid=cached.get('status')=='success' and (output/'output.png').is_file() and cached['output_sha256']==sha(output/'output.png')
                if not valid:
                    # Only this node's previous partial image directory is reset.
                    for p in output.iterdir():
                        if p.is_file():p.unlink()
                        elif p.is_dir():shutil.rmtree(p)
                    attempts=[]
                    for attempt in (1,2):
                        self.progress(phase='exploration',sample_id=row['sample_id'],current_task=task,current_tool=tool,node=key,attempt=attempt)
                        command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--project',str(self.args.project),
                          '--subtask',task,'--tool',tool,'--input',str(input_dir),'--output',str(output)]
                        begin=time.monotonic()
                        with (node/f'attempt_{attempt}.log').open('ab') as log:
                            self.child=subprocess.Popen(command,cwd=self.args.project,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                            while self.child.poll() is None:
                                self.progress(current_node_elapsed_s=round(time.monotonic()-begin,1))
                                if time.monotonic()-begin>600:self.cleanup();break
                                time.sleep(5)
                        code=self.child.returncode
                        attempts.append({'attempt':attempt,'returncode':code,'elapsed_s':round(time.monotonic()-begin,2)})
                        if code==0 and (output/'output.png').is_file():break
                        for p in output.iterdir():
                            if p.is_file():p.unlink()
                            elif p.is_dir():shutil.rmtree(p)
                    if code!=0 or not (output/'output.png').exists():
                        write(record,{'status':'failed','path':path,'attempts':attempts})
                        raise RuntimeError(f'Exploration node failed: {row["sample_id"]}/{task}/{tool}; inspect {node}')
                    write(record,{'status':'success','path':path,'attempts':attempts,'output_sha256':sha(output/'output.png')})
                self.nodes_done+=1
                self.progress()
                self.tree(row,[t for t in tasks if t!=task],path,output)

    def explore(self):
        for row in self.rows:
            source=self.root/'inputs'/row['sample_id'];source.mkdir(parents=True,exist_ok=True)
            dest=source/'input.png'
            if not dest.exists():shutil.copy2(self.args.data/row['lq_path'],dest)
            assert sha(dest)==row['lq_sha256']
            tasks=[MAPPING[d] for d in json.loads(row['degradation_sequence'])]
            self.tree(row,tasks,[],source)
        assert self.nodes_done==self.total_nodes and len(self.leaves)==self.total_leaves
        write(self.root/'leaves.json',self.leaves)

    def agenticir_score(self):
        from llm import DepictQA
        from runtime_safety import parse_response
        dqa=DepictQA(silent=True)
        path=self.root/'agenticir/observations.jsonl';path.parent.mkdir(exist_ok=True)
        old=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        done={r['output'] for r in old}
        with path.open('a') as out:
            for index,leaf in enumerate(self.leaves,1):
                if leaf['output'] in done:continue
                self.progress(phase='agenticir_scoring',scored_leaves=index-1,current_output=leaf['output'])
                severity={d:parse_response(dqa(img_path=Path(leaf['output']),task='eval_degradation',degradation=d))[0][1] for d in leaf['degradations']}
                record={**leaf,'severity':severity,'failures':{d:s in ('medium','high','very high') for d,s in severity.items()}}
                out.write(json.dumps(record)+'\n');out.flush();os.fsync(out.fileno())
        records=[json.loads(line) for line in path.read_text().splitlines()]
        assert len(records)==self.total_leaves
        hub=defaultdict(dict)
        for record in records:
            combo='+'.join(record['degradations']);order='+'.join(record['plan'])
            data=hub[combo].setdefault(order,{'total':0,**{d:0 for d in record['degradations']}})
            data['total']+=1
            for d,failed in record['failures'].items():data[d]+=failed
        for combo,plans in hub.items():
            for plan,data in plans.items():
                rates={d:count/data['total'] for d,count in data.items() if d!='total'}
                rates['total']=sum(rates.values())/len(rates);data['fail rate']=rates
            hub[combo]=dict(sorted(plans.items(),key=lambda pair:pair[1]['fail rate']['total']))
        write(self.root/'agenticir/fail_rate.json',hub)
        return hub

    def mair_score(self):
        # Every metric is mandatory; unavailable metrics stop this phase rather
        # than silently distilling a different objective.
        import cv2,numpy as np,torch,pyiqa
        from skimage.metrics import peak_signal_noise_ratio,structural_similarity
        folder=self.root/'mair';folder.mkdir(exist_ok=True)
        metrics={}
        for name in ('lpips','dists','maniqa','clipiqa','musiq'):
            self.progress(phase='mair_metric_loading',metric=name)
            metrics[name]=pyiqa.create_metric(name,device='cuda')
        path=folder/'observations.jsonl'
        old=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        done={r['output'] for r in old}
        with path.open('a') as out:
            for index,leaf in enumerate(self.leaves,1):
                if leaf['output'] in done:continue
                self.progress(phase='mair_scoring',scored_leaves=index-1,current_output=leaf['output'])
                image=cv2.cvtColor(cv2.imread(leaf['output']),cv2.COLOR_BGR2RGB)
                hq=cv2.cvtColor(cv2.imread(leaf['hq']),cv2.COLOR_BGR2RGB)
                resized=image.shape!=hq.shape
                if resized:image=cv2.resize(image,(hq.shape[1],hq.shape[0]),interpolation=cv2.INTER_CUBIC)
                values={'psnr':float(peak_signal_noise_ratio(hq,image,data_range=255)),
                        'ssim':float(structural_similarity(hq,image,channel_axis=2,data_range=255))}
                ti=torch.from_numpy(image.transpose(2,0,1).copy()).float().unsqueeze(0).cuda()/255
                tr=torch.from_numpy(hq.transpose(2,0,1).copy()).float().unsqueeze(0).cuda()/255
                with torch.no_grad():
                    for name,metric in metrics.items():
                        torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
                        values[name]=float(metric(ti,tr).item() if name in ('lpips','dists') else metric(ti).item())
                assert all(math.isfinite(v) for v in values.values()),values
                record={**leaf,'metrics':values,'metric_resized':resized}
                out.write(json.dumps(record)+'\n');out.flush();os.fsync(out.fileno())
        records=[json.loads(line) for line in path.read_text().splitlines()]
        assert len(records)==self.total_leaves
        grouped=defaultdict(list)
        for row in records:grouped[row['sample_id']].append(row)
        normalization={}
        for sample,items in grouped.items():
            normalization[sample]={}
            for name in ('psnr','ssim','lpips','dists','maniqa','clipiqa','musiq'):
                values=np.array([r['metrics'][name] for r in items]);mean=float(values.mean());std=float(values.std(ddof=0))
                normalization[sample][name]={'mean':mean,'std':std,'lower_better':name in ('lpips','dists')}
            for row in items:
                row['score']=sum(((-1 if name in ('lpips','dists') else 1)*(v-normalization[sample][name]['mean'])/normalization[sample][name]['std'])
                    if normalization[sample][name]['std']>1e-12 else 0 for name,v in row['metrics'].items())
        write(folder/'scored_sequences.json',records);write(folder/'normalization.json',normalization)
        hub=defaultdict(dict)
        for row in records:
            combo='+'.join(row['degradations']);plan='+'.join(row['plan'])
            entry=hub[combo].setdefault(plan,{'scores':[],'stage':STAGE[row['plan'][0]]})
            entry['scores'].append(row['score'])
        for combo,plans in hub.items():
            for plan,entry in plans.items():
                values=entry.pop('scores');entry.update(count=len(values),mean_score=float(np.mean(values)),max_score=float(max(values)))
        write(folder/'order_scores.json',hub)
        return hub

    def distill(self,name,observations):
        from llm import GPT4
        self.progress(phase='distillation',method=name)
        raw=json.dumps(observations,indent=2)
        common=('These are controlled validation-pilot records from 2 source HQ images, not test images. '
            'Only noise+motion-blur and rain+haze are supported. Do not invent evidence for other combinations. '
            'Describe uncertainty, conditional ordering guidance, and conflicts between source cases. '
            'Output concise plain text, no code. These are preliminary empirical observations, not the original authors experience.\n')
        if name=='agenticir':
            prompt=common+'Distill scheduling knowledge from residual degradation failure rates. Lower is better. '
        else:
            prompt=common+('Distill ONLY within-stage scheduling guidance for MAIR: stage 2 noise/motion-blur; stage 3 rain/haze. '
                'Never change the fixed cross-stage order compression -> imaging -> scene. '
                'Scores are signed sums of seven within-image standardized IQA metrics; higher is better. ')
        prompt+='\nObserved statistics:\n'+raw
        folder=self.root/name
        write(folder/'distillation_input.json',{'prompt':prompt,'model_config_source':str(self.args.config)})
        gpt=GPT4(config_path=self.args.config,silent=True)
        distilled=gpt(prompt=prompt)
        assert isinstance(distilled,str) and distilled.strip()
        value={'raw':raw,'distilled':distilled,'source':str(self.root/'protocol.json'),
            'kind':'independently_distilled_controlled_validation_pilot','method':name,
            'model':gpt.model,'test_data_used':False,'pilot_only':True,
            'observations_sha256':hashlib.sha256(raw.encode()).hexdigest(),'coverage':['noise+motion blur','rain+haze']}
        write(folder/'schedule_experience.json',value)

    def run(self):
        try:
            self.explore()
            agent=self.agenticir_score()
            if self.args.distill:self.distill('agenticir',agent)
            mair=self.mair_score()
            if self.args.distill:self.distill('mair',mair)
            self.progress(state='completed',phase='completed',completed_leaves=len(self.leaves),
                          external_distillation_enabled=bool(self.args.distill))
        except BaseException as error:
            self.progress(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'needs_attention',error=f'{type(error).__name__}: {error}')
            raise

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('worker');p.add_argument('--project',type=Path,required=True);p.add_argument('--subtask',required=True);p.add_argument('--tool',required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--project',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--scope',choices=['pilot','pairwise-full'],default='pilot')
    p.add_argument('--exclude-tool',action='append',default=[],help='Explicitly excluded runtime-unusable tools; recorded in protocol')
    p.add_argument('--distill',action='store_true',help='External statistics export; requires explicit user authorization')
    args=parser.parse_args()
    if args.command=='worker':worker(args)
    else:Pilot(args).run()

if __name__=='__main__':main()
