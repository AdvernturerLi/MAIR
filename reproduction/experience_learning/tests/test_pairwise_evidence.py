import pytest
from experience_learning.pairwise_evidence import build

def row(sample,plan,tools,score):
 return dict(sample_id=sample,degradations=['motion blur','noise'],plan=plan,tools=tools,score=score)

def test_matching_follows_task_tool_assignment_when_order_reverses():
 records=[]
 for h in ('001','002'):
  records.extend([row(h+'__noise+blur',['denoising','motion deblurring'],['dn','db'],3),row(h+'__noise+blur',['motion deblurring','denoising'],['db','dn'],1)])
 x=build(records,'mair')['task_order_comparisons']['motion blur+noise']
 assert x['matched_tool_pairs']==2 and x['hq_count']==2
 assert x['mean_paired_delta']==-2 and x['hq_bootstrap_95ci']==[-2,-2]

def test_missing_order_is_not_silently_dropped():
 with pytest.raises(AssertionError):build([row('001__noise+blur',['denoising','motion deblurring'],['dn','db'],3)],'mair')

def test_mair_does_not_distill_cross_stage_pairs():
 records=[dict(sample_id='001__rain+blur',degradations=['rain','motion blur'],plan=['deraining','motion deblurring'],tools=['a','b'],score=1)]
 assert build(records,'mair')['task_order_comparisons']=={}
