"""Frozen schemas, independent review, no reference-conditioned inputs, BPE protocols."""
import copy
import json
import shutil

import pytest
import torch

from minillm.chinese_foundation_eval import (load_foundation,review_decisions,aggregate_foundation,
                                            accuracy_interval,require_comparable)
from minillm.diagnostic_common import FOUNDATION
from minillm.evaluation_candidates import candidate_prediction,load_candidates
from minillm.model import MiniLLM
from minillm.training_manifest import file_sha256


def test_frozen_counts_and_pending_review():
    candidates,prompts=load_foundation()
    assert len(candidates)==120 and len(prompts)==30
    assert {c:sum(r['category']==c for r in candidates) for c in ('chinese','world','commonsense','math_logic','english_code')}=={'chinese':30,'world':40,'commonsense':20,'math_logic':20,'english_code':10}
    assert all(r['review_status']=='REVIEW_PENDING' for r in candidates+prompts)
    assert {k:sum(r['reference']==k for r in candidates) for k in 'ABCD'}==dict.fromkeys('ABCD',30)


def test_freeze_rejects_modified_assets(tmp_path):
    shutil.copytree(FOUNDATION,tmp_path/'assets')
    with (tmp_path/'assets/candidates.jsonl').open('a') as f:f.write('{}\n')
    with pytest.raises(ValueError):load_foundation(tmp_path/'assets')


@pytest.mark.parametrize('mutation',['duplicate','illegal_reference','duplicate_choices'])
def test_invalid_candidates_rejected(tmp_path,mutation):
    task=load_foundation()[0][0];task=copy.deepcopy(task)
    if mutation=='illegal_reference':task['reference']='Z'
    if mutation=='duplicate_choices':task['candidates']['B']=task['candidates']['A']
    rows=[task,task] if mutation=='duplicate' else [task]
    path=tmp_path/'custom.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    with pytest.raises(ValueError):load_candidates(path)


def test_only_independent_accept_can_enter_formal_counts(tmp_path):
    candidates,prompts=load_foundation();task=candidates[0]
    row={'id':task['id'],'question_version':1,'asset_sha256':file_sha256(FOUNDATION/'candidates.jsonl'),
         'action':'ACCEPT','reviewer':'human reviewer','reason':'Verified uniqueness and stable fact.','human_review':True}
    path=tmp_path/'review.jsonl';path.write_text(json.dumps(row))
    assert review_decisions(path,candidates,prompts)[task['id']]['action']=='ACCEPT'
    def prediction(action):
        return {'review_action':action,'category':'world','prediction':{'protocols':{p:{'status':'SCORED','score':{'correct':True,'status':'scored'}} for p in ('separate','joint_shared_prefix')}}}
    rows=[prediction(a) for a in ('REVIEW_PENDING','ACCEPT','REJECT','REVISE')]
    assert aggregate_foundation(rows,reviewed_only=True)['separate']['scored']==1
    row['human_review']=False;path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='human review'):review_decisions(path,candidates,prompts)
    row['human_review']=True;row['asset_sha256']='0'*64;path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='frozen'):review_decisions(path,candidates,prompts)


def test_review_duplicate_rejected(tmp_path):
    candidates,prompts=load_foundation()
    template=(FOUNDATION/'review-template.jsonl').read_text().splitlines()[0]
    path=tmp_path/'reviews.jsonl';path.write_text(template+'\n'+template+'\n')
    with pytest.raises(ValueError,match='duplicate'):review_decisions(path,candidates,prompts)


def test_reference_does_not_change_model_inputs(tiny_tokenizer,nano_cfg):
    model=MiniLLM(nano_cfg).eval();task=copy.deepcopy(load_foundation()[0][0]);inputs=[]
    hook=model.register_forward_pre_hook(lambda m,args:inputs.append(args[0].clone()))
    a=candidate_prediction(model,tiny_tokenizer,task)
    first=[t.clone() for t in inputs];inputs.clear()
    task['reference']=next(k for k in task['candidates'] if k!=task['reference'])
    b=candidate_prediction(model,tiny_tokenizer,task);hook.remove()
    assert len(first)==len(inputs) and all(torch.equal(x,y) for x,y in zip(first,inputs))
    assert a['prompt_token_ids']==tiny_tokenizer.encode(task['prompt'])
    assert a['selected']==b['selected'] and a['protocols'].keys()=={'separate','joint_shared_prefix'}
    for choice in a['candidates'].values():assert choice['tokens']==len(choice['token_ids']) and choice['negative_log_likelihood']>0


def test_bpe_boundary_uses_production_audit(tiny_tokenizer,nano_cfg):
    task={'id':'boundary','category':'chinese','prompt':'人','candidates':{'A':'工智能','B':'民','C':'类','D':'生'},'reference':'A','text_category':'zh'}
    r=candidate_prediction(MiniLLM(nano_cfg),tiny_tokenizer,task)
    assert r['boundary_audit']['inconsistent_candidates']
    assert r['protocols']['joint_shared_prefix']['status'].startswith(('SCORED','NOT_SCORABLE'))


def test_uncertainty_and_protocol_identity():
    assert accuracy_interval(0,0)['wilson_95'] is None
    assert accuracy_interval(0,4)['wilson_95'][1]>0
    with pytest.raises(ValueError,match='protocols'):require_comparable({'boundary_policy':'separate'},{'boundary_policy':'joint_shared_prefix'})


@pytest.mark.parametrize('field',['benchmark_sha256','tokenizer_fingerprint','text_policy','boundary_policy','normalization','decoding','review_sha256','model_config','implementation_sha256'])
def test_changed_protocol_cannot_produce_ability_delta(field):
    left=dict.fromkeys(('benchmark_sha256','tokenizer_fingerprint','text_policy','boundary_policy','normalization','decoding','review_sha256','model_config','implementation_sha256'),'fixed')
    require_comparable(left,left.copy())
    right=left.copy();right[field]='changed'
    with pytest.raises(ValueError,match='protocols'):require_comparable(left,right)


def test_unscorable_items_keep_tokens_and_exclude_denominator(tiny_tokenizer,nano_cfg,monkeypatch):
    import minillm.chinese_foundation_eval as foundation
    candidates,prompts=load_foundation();task=copy.deepcopy(candidates[0]);task['prompt']='中文'*200
    monkeypatch.setattr(foundation,'load_foundation',lambda directory:([task],[]))
    result=foundation.evaluate_foundation(MiniLLM(nano_cfg),tiny_tokenizer)
    assert result['exploratory']['separate']['scored']==0
    assert result['exploratory']['separate']['unscorable']==1
    assert result['rows'][0]['prediction']['candidate_token_ids']
