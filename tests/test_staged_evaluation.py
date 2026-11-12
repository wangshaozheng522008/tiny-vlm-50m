import json
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from minillm.assessment_io import records
from minillm.evaluation import EvaluationConfig, PRETRAIN_CANDIDATES, PRETRAIN_PROMPTS, evaluate_model, load_tasks
from minillm.evaluation_candidates import candidate_prediction, load_candidates
from minillm.evaluation_compare import compare_evaluations
from minillm.model import MiniLLM
from minillm.train import save_model_checkpoint
from minillm.utils import set_seed


def probe():
    return {'schema_version':1,'id':'natural','category':'math','prompt':'1+1=',
            'candidates':{'A':'2','B':'3'},'reference':'A'}


def test_candidate_shift_and_reference_is_not_model_input(tiny_tokenizer,nano_cfg):
    torch.set_num_threads(1);set_seed(43)
    model=MiniLLM(nano_cfg).eval()
    task=probe()
    a=candidate_prediction(model,tiny_tokenizer,task)
    b=candidate_prediction(model,tiny_tokenizer,{**task,'reference':'B'})
    assert a['candidates']==b['candidates'] and a['selected']==b['selected']
    prefix=tiny_tokenizer.encode(task['prompt']);tail=tiny_tokenizer.encode('2')
    with torch.inference_mode():
        logits=model(torch.tensor([prefix+tail])).logits
        expected=F.cross_entropy(logits[0,len(prefix)-1:-1].float(),torch.tensor(tail),reduction='sum')
    assert a['candidates']['A']['negative_log_likelihood']==pytest.approx(float(expected),abs=1e-6)
    assert a['candidates']['A']['tokens']==len(tail)
    mean=candidate_prediction(model,tiny_tokenizer,task,normalization='mean')
    assert mean['candidates']['A']['score']==pytest.approx(-float(expected)/len(tail))
    with pytest.raises(ValueError,match='context'):
        candidate_prediction(model,tiny_tokenizer,{**task,'prompt':'x'*10000})


def test_candidate_tie_is_not_correct_and_specials_refused(tiny_tokenizer,nano_cfg):
    model=MiniLLM(nano_cfg).eval()
    for parameter in model.parameters():
        parameter.data.zero_()
    row=candidate_prediction(model,tiny_tokenizer,probe())
    assert row['selected'] is None and row['score']=={'correct':False,'status':'tie'}
    with pytest.raises(ValueError,match='special token'):
        candidate_prediction(model,tiny_tokenizer,{**probe(),'candidates':{'A':'<eos>','B':'3'}})


def test_frozen_pretrain_assets():
    rows=load_candidates(PRETRAIN_CANDIDATES)
    assert len(rows)==24 and len(load_tasks(PRETRAIN_PROMPTS,scored=False))==12
    assert len({r['reference'] for r in rows})==4


def test_stage_outputs_and_comparison(tmp_path,tiny_tokenizer,nano_cfg):
    torch.set_num_threads(1);set_seed(42)
    model=MiniLLM(nano_cfg)
    ckpt=tmp_path/'model.pt';tok=tmp_path/'tokenizer.json';tiny_tokenizer.save(tok)
    save_model_checkpoint(ckpt,model,tiny_tokenizer,step=0,tokens=0,val_loss=None,extra={'mode':'pretrain'})
    candidates=tmp_path/'candidates.jsonl';candidates.write_text(json.dumps(probe())+'\n')
    tasks=tmp_path/'tasks.jsonl';tasks.write_text(''.join(json.dumps(dict(id=mode,category='math',prompt='1+1=',
        reference='2',scoring='exact_match',generation_mode=mode))+'\n' for mode in ('raw','chat')))
    fixed=tmp_path/'prompts.jsonl';fixed.write_text(json.dumps(dict(id='p',prompt='hello',generation_mode='raw'))+'\n')
    config=EvaluationConfig(stage='pretrain',seq_len=16,max_new_tokens=2)
    kwargs=dict(benchmark=tasks,prompts=fixed,candidate_benchmark=candidates,config=config)
    a=evaluate_model(ckpt,tok,tmp_path/'pretrain',**kwargs)
    b=evaluate_model(ckpt,tok,tmp_path/'repeat',**kwargs)
    assert a['stage']['chat']=='NOT_RUN' and a['scores']['chat']['tasks']==0
    assert a['scores']['raw']['tasks']==1 and a['candidate_likelihood']['tasks']==1
    assert a['diagnostics']['generations']==2 # candidates excluded from degeneration metrics
    assert a['candidate_likelihood']==b['candidate_likelihood']
    ra=list(records(tmp_path/'pretrain/predictions.jsonl'));rb=list(records(tmp_path/'repeat/predictions.jsonl'))
    assert [r.get('generated_token_ids') for r in ra]==[r.get('generated_token_ids') for r in rb]
    comp=compare_evaluations([tmp_path/'pretrain',tmp_path/'repeat'],tmp_path/'comparison')
    assert comp['experiments'][1]['candidate_likelihood']['math']['delta']==0
    sft=evaluate_model(ckpt,tok,tmp_path/'sft',**{**kwargs,'config':replace(config,stage='sft')})
    assert sft['scores']['raw']['tasks']==0 and sft['scores']['chat']['tasks']==1
    assert sft['stage']['checkpoint_mode']=='pretrain' # do not claim it has had SFT
    with pytest.raises(ValueError,match='incompatible'):
        compare_evaluations([tmp_path/'pretrain',tmp_path/'sft'],tmp_path/'incompatible')


@pytest.mark.parametrize('change',[{'schema_version':2},{'reference':'Z'},{'candidates':{'A':'2','B':'2'}}])
def test_bad_candidate_schema(tmp_path,change):
    path=tmp_path/'probe.jsonl';path.write_text(json.dumps({**probe(),**change})+'\n')
    with pytest.raises(ValueError,match='invalid candidate'):
        load_candidates(path)
