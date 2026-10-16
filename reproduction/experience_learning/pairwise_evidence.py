"""Matched order evidence; each source HQ is one statistical unit."""
from collections import defaultdict
import numpy as np
STAGE={'denoising':2,'motion deblurring':2,'super-resolution':2,'deraining':3,'dehazing':3,'brightening':3,'jpeg compression artifact removal':1}
def build(records,method):
    combos=defaultdict(list)
    for row in records:
        assert len(row['plan'])==2
        if method=='mair' and len({STAGE[t] for t in row['plan']})>1:continue
        combos['+'.join(row['degradations'])].append(row)
    result={}
    for combo,rr in combos.items():
        orders=sorted({tuple(r['plan']) for r in rr});assert len(orders)==2
        paired=defaultdict(dict)
        for r in rr:
            assignment=tuple(sorted(zip(r['plan'],r['tools'])))
            key=(r['sample_id'],assignment)
            utility=(float(r['score']) if method=='mair' else -sum(r['failures'].values())/len(r['failures']))
            order=tuple(r['plan']);assert order not in paired[key]
            paired[key][order]=utility
        per_hq=defaultdict(list);per_tool=defaultdict(list)
        for (sample,assignment),values in paired.items():
            assert set(values)==set(orders),'Missing matched order counterpart'
            delta=values[orders[1]]-values[orders[0]]
            per_hq[sample.split('__')[0]].append(delta);per_tool[str(assignment)].append(delta)
        means=np.array([np.mean(v) for h,v in sorted(per_hq.items())]);rng=np.random.default_rng(20261016)
        bootstrap=means[rng.integers(0,len(means),size=(10000,len(means)))].mean(1)
        loo=[float(np.delete(means,i).mean()) for i in range(len(means))] if len(means)>1 else []
        result[combo]={'orders':orders,'delta_definition':'utility(order[1]) - utility(order[0]); positive favors order[1]',
            'utility':'signed seven-IQA within-image z-score sum' if method=='mair' else 'negative unweighted mean residual failure rate',
            'hq_count':len(means),'matched_tool_pairs':len(paired),'mean_paired_delta':float(means.mean()),'hq_bootstrap_95ci':[float(x) for x in np.quantile(bootstrap,[.025,.975])],
            'per_hq_mean_delta':{h:float(np.mean(v)) for h,v in sorted(per_hq.items())},'leave_one_hq_out_mean_deltas':loo,
            'positive_pair_fraction':float(np.mean([d>1e-12 for values in per_hq.values() for d in values])),
            'per_tool_assignment':{tool:{'n':len(v),'mean_delta':float(np.mean(v)),'positive_fraction':float(np.mean(np.array(v)>1e-12))} for tool,v in sorted(per_tool.items())}}
    return {'method':method,'test_data_used':False,'independent_unit':'source HQ; tool combinations are repeated measurements, not independent images','normalization':'MAIR uses a fixed complete within-image candidate pool; subsets reuse its normalization','task_order_comparisons':result}
