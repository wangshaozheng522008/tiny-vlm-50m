"""Real BPE merge across an answer boundary; no external data/model required."""
import json
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

from minillm.evaluation import EvaluationConfig, evaluate_model
from minillm.evaluation_candidates import candidate_boundary_audit, candidate_prediction, aggregate_candidates
from minillm.evaluation_compare import compare_evaluations
from minillm.model import MiniLLM
from minillm.text_policy import WIDTH_FOLD
from minillm.tokenizer import MiniLLMTokenizer, SPECIAL_TOKENS
from minillm.train import save_model_checkpoint


@pytest.fixture
def boundary_tokenizer():
    # One actual BPE merge: encode('a')+encode('b') != encode('ab').
    vocab={s:i for i,s in enumerate(SPECIAL_TOKENS)}
    for symbol in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocab[symbol]=len(vocab)
    vocab['ab']=len(vocab)
    raw=Tokenizer(models.BPE(vocab=vocab,merges=[('a','b')],unk_token='<unk>'))
    raw.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder=decoders.ByteLevel()
    raw.add_special_tokens(SPECIAL_TOKENS)
    tokenizer=MiniLLMTokenizer(raw);tokenizer.validate_layout()
    return tokenizer


@pytest.fixture
def boundary_model(boundary_tokenizer,nano_cfg):
    torch.set_num_threads(1);torch.manual_seed(71)
    return MiniLLM(replace(nano_cfg,vocab_size=boundary_tokenizer.vocab_size)).eval()


def task(prompt='x a'):
    return {'id':'cross-merge','category':'language','prompt':prompt,'candidates':{'A':'b','B':'c'},'reference':'A'}


def test_joint_merge_audit_and_independent_nll(boundary_tokenizer,boundary_model):
    tok,model=boundary_tokenizer,boundary_model
    row=candidate_prediction(model,tok,task())
    audit=row['boundary_audit']
    assert audit['inconsistent_candidates']==['A']
    assert audit['joint_shared_prefix_tokens']<len(row['prompt_token_ids'])
    assert row['candidates']==row['protocols']['separate']['candidates']
    assert row['boundary_policy_name']=='separate'
    # The legacy primary remains continuation-only, exactly as before.
    with torch.inference_mode():
        for key,text in task()['candidates'].items():
            prefix=tok.encode(task()['prompt']);tail=tok.encode(text)
            logits=model(torch.tensor([prefix+tail])).logits
            expected=F.cross_entropy(logits[0,len(prefix)-1:-1].float(),torch.tensor(tail),reduction='sum')
            assert row['candidates'][key]['negative_log_likelihood']==pytest.approx(float(expected),abs=1e-6)
            ids=audit['candidates'][key]['joint_token_ids'];start=audit['joint_shared_prefix_tokens']
            logits=model(torch.tensor([ids])).logits
            expected=F.cross_entropy(logits[0,start-1:-1].float(),torch.tensor(ids[start:]),reduction='sum')
            joint=row['protocols']['joint_shared_prefix']['candidates'][key]
            assert joint['negative_log_likelihood']==pytest.approx(float(expected),abs=1e-6)
            assert joint['tokens']==len(ids)-start
            assert joint['contains_retokenized_prompt'] and joint['prompt_characters_in_scored_tail']==1
    report=aggregate_candidates([row])
    assert report['boundary_audit']['inconsistent_tasks']==1
    assert report['boundary_audit']['inconsistent_candidates']==1
    assert report['boundary_audit']['total_candidates']==2
    assert set(report['protocols'])=={'separate','joint_shared_prefix'}
    primary=candidate_prediction(model,tok,task(),boundary_policy='joint_shared_prefix')
    assert primary['candidates']==row['protocols']['joint_shared_prefix']['candidates']
    flipped=candidate_prediction(model,tok,{**task(),'reference':'B'})
    assert flipped['boundary_audit']==audit and flipped['candidates']==row['candidates']
    assert flipped['protocols']['joint_shared_prefix']['candidates']==row['protocols']['joint_shared_prefix']['candidates']


def test_equal_boundary_protocols_and_mean(boundary_tokenizer,boundary_model):
    row=candidate_prediction(boundary_model,boundary_tokenizer,task('x '),normalization='mean')
    assert row['boundary_audit']['inconsistent_candidates']==[]
    for key in task()['candidates']:
        a=row['protocols']['separate']['candidates'][key]
        b=row['protocols']['joint_shared_prefix']['candidates'][key]
        assert a['negative_log_likelihood']==b['negative_log_likelihood']
        assert a['score']==b['score']==-a['negative_log_likelihood']/a['tokens']
        assert not b['contains_retokenized_prompt']


def test_no_shared_context_is_unscorable_not_fabricated(boundary_tokenizer,boundary_model):
    row=candidate_prediction(boundary_model,boundary_tokenizer,task('a'),boundary_policy='joint_shared_prefix')
    assert row['score']=={'correct':None,'status':'NOT_SCORABLE_NO_SHARED_CONTEXT'}
    assert row['protocols']['separate']['score']['correct'] is not None
    report=aggregate_candidates([row])
    assert report['status']=='INCOMPLETE'
    assert report['categories']['language']['accuracy'] is None
    assert report['categories']['language']['unscorable']==1


def test_normalization_context_change_and_declared_code(boundary_tokenizer,boundary_model):
    tok=boundary_tokenizer;tok.text_policy=WIDTH_FOLD
    question={**task('x'),'candidates':{'A':'=１','B':'=２'}}
    row=candidate_prediction(boundary_model,tok,question)
    assert not row['boundary_audit']['normalization_context_consistent']
    assert row['protocols']['joint_shared_prefix']['status']=='NOT_SCORABLE_NORMALIZATION_CONTEXT'
    protected=candidate_prediction(boundary_model,tok,{**question,'text_category':'code'})
    assert protected['boundary_audit']['normalization_context_consistent']
    assert protected['text_category']=='code'
    for key,text in question['candidates'].items():
        assert tok.decode(protected['boundary_audit']['candidates'][key]['continuation_token_ids'])==text


def test_report_identity_separates_protocols(tmp_path,boundary_tokenizer,boundary_model):
    ckpt=tmp_path/'model.pt';tok=tmp_path/'tokenizer.json';boundary_tokenizer.save(tok)
    save_model_checkpoint(ckpt,boundary_model,boundary_tokenizer,step=0,tokens=0,val_loss=None)
    probes=tmp_path/'probes.jsonl';probes.write_text(json.dumps({'schema_version':1,**task()})+'\n')
    cfg=EvaluationConfig(seq_len=16)
    kwargs=dict(benchmark=None,prompts=None,candidate_benchmark=probes)
    r=evaluate_model(ckpt,tok,tmp_path/'separate',config=cfg,**kwargs)
    assert r['candidate_likelihood']['boundary_audit']['affected_task_ids']==['cross-merge']
    evaluate_model(ckpt,tok,tmp_path/'joint',config=replace(cfg,candidate_boundary_policy='joint_shared_prefix'),**kwargs)
    with pytest.raises(ValueError,match='incompatible'):
        compare_evaluations([tmp_path/'separate',tmp_path/'joint'],tmp_path/'comparison')
