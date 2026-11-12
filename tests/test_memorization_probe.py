"""Isolation, explicit resource admission, and deterministic learning trajectories."""
import json
from dataclasses import replace

import pytest
import torch

from minillm.diagnostic_common import Budget, MEMORIZATION
from minillm.memorization_probe import load_memorization,prepare_model,run_memorization
from minillm.recovery_state import compare_tree
from minillm.train import load_training_state


def test_split_and_category_mix():
    train,heldout=load_memorization()
    assert len(train)==160 and len(heldout)==32
    assert not {r['text'] for r in train}&{r['text'] for r in heldout}
    assert {c:sum(r['category']==c for r in train) for c in ('zh','facts','math','mixed','code')}=={'zh':80,'facts':32,'math':16,'mixed':16,'code':16}


@pytest.mark.parametrize('kwargs',[
    {'max_steps':0},{'max_steps':1001},{'max_effective_tokens':0},
    {'max_effective_tokens':2_000_001},{'max_wall_seconds':0},
    {'max_wall_seconds':601},{'max_wall_seconds':float('nan')}])
def test_invalid_budgets_rejected(kwargs):
    with pytest.raises(ValueError):replace(Budget(),**kwargs).validate()


def test_87m_requires_explicit_admission():
    train,_=load_memorization()
    with pytest.raises(ValueError,match='enable-87m'):
        prepare_model('87m','cpu',train,seed=42)
    with pytest.raises(ValueError,match='CUDA'):
        prepare_model('87m','cpu',train,seed=42,enable_87m=True,tokenizer_path='unused.json')


def test_gpu_budget_rejected_before_model_allocation(monkeypatch):
    train,_=load_memorization()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(512*1024**2,1024*1024**2))
    with pytest.raises(ValueError,match='GPU budget'):
        prepare_model('nano','cuda',train,seed=42,gpu_memory_budget_mib=512)


def test_seed_and_budget_reproduce_trajectory(tmp_path):
    paths=[];results=[]
    for i in range(2):
        out=tmp_path/str(i);out.mkdir();paths.append(out)
        results.append(run_memorization(out,budget=Budget(max_steps=4,max_effective_tokens=1000,max_wall_seconds=60),seed=19))
    for name in ('single','corpus'):
        a=load_training_state(paths[0]/name/'state.pt');b=load_training_state(paths[1]/name/'state.pt')
        compare_tree(a['model'],b['model'],exact=True)
        compare_tree(a['optimizer'],b['optimizer'],exact=True)
        compare_tree(a['rng'],b['rng'],exact=True)
        assert results[0]['experiments'][name]['curve']==results[1]['experiments'][name]['curve']
    assert all(r['training_steps']<=4 and r['effective_tokens']<=1000 for r in results)
    assert results[0]['learning']=='LEARNING_INCONCLUSIVE'
    assert results[0]['experiments']['corpus']['independent_samples']['prefix_continuation']['samples']==8


def test_heldout_training_text_overlap_rejected(tmp_path):
    import shutil
    from minillm.training_manifest import file_sha256
    folder=tmp_path/'assets';shutil.copytree(MEMORIZATION,folder)
    train,heldout=load_memorization()
    heldout[0].update({k:train[0][k] for k in ('text','prefix','continuation')})
    path=folder/'heldout.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in heldout))
    version=json.loads((folder/'benchmark-version.json').read_text());version['files']['heldout.jsonl']['sha256']=file_sha256(path)
    (folder/'benchmark-version.json').write_text(json.dumps(version))
    with pytest.raises(ValueError,match='overlap'):load_memorization(folder)


def test_tiny_token_budget_cannot_overrun(tmp_path):
    out=tmp_path/'limited';out.mkdir()
    result=run_memorization(out,budget=Budget(max_steps=100,max_effective_tokens=126,max_wall_seconds=60))
    assert result['effective_tokens']<=126 and result['training_steps']<=100
    assert result['experiments']['corpus']['status']=='NOT_RUN_TOKEN_BUDGET'
    assert result['engineering']=='ENGINEERING_FAIL' and result['learning']=='LEARNING_INCONCLUSIVE'


def test_reserved_prose_tokens_rejected(tiny_tokenizer,tmp_path):
    from minillm.memorization_probe import pack_rows
    with pytest.raises(ValueError,match='reserved'):
        pack_rows(tmp_path/'bad.bin',tiny_tokenizer,[{'text':'正文<pad>','text_category':'zh'}])
    assert not (tmp_path/'bad.bin').exists()
