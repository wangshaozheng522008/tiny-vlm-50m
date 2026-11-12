"""Optimizer, RNG and continued-trajectory checks with CPU Nano and dropout."""
import copy
import importlib.util
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from minillm.assessment_io import read_json
from minillm.config import MiniLLMConfig
from minillm.model import MiniLLM
from minillm.recovery_state import compare_tree
from minillm.train import save_model_checkpoint
from test_evaluation import eval_assets
from test_workload import config

spec=importlib.util.spec_from_file_location('pilot_recovery_script',Path(__file__).resolve().parents[1]/'scripts/check_pilot_recovery.py')
recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)


def stochastic_assets(assets):
    cfg=MiniLLMConfig.load(assets['cfg']);cfg.hidden_dropout=.2;cfg.attention_dropout=.2
    cfg.save(assets['cfg'])
    from minillm.tokenizer import MiniLLMTokenizer
    save_model_checkpoint(assets['ckpt'],MiniLLM(cfg),MiniLLMTokenizer.load(assets['tokenizer']),step=0,tokens=0,val_loss=None)


def test_twenty_steps_two_resumes_optimizer_rng_and_continuation(eval_assets,tmp_path):
    stochastic_assets(eval_assets)
    cfg={**config(eval_assets,tmp_path),'max_steps':20,'warmup_ratio':.2}
    original=recovery.training._train_step
    def stochastic_step(*args,**kwargs):
        result=original(*args,**kwargs)
        random.random();np.random.random();torch.rand(3)
        return result
    with patch.object(recovery.training,'_train_step',side_effect=stochastic_step):
        result=recovery.check(eval_assets['dataset'],eval_assets['tokenizer'],str(eval_assets['cfg']),cfg,tmp_path/'recovery',steps=20)
    assert result['interruptions']==[5,12] and result['steps']==20
    for key in ('optimizer_state_equal','rng_state_equal','continuation_equal','sample_order_equal','lr_equal'):assert result[key] is True
    assert result['optimizer_max_difference']==0 and result['maximum_parameter_difference']==0
    assert result['continuation']['updates']==8 and result['continuation']['from_step']==12
    assert read_json(tmp_path/'recovery/recovery.json')==result


@pytest.mark.parametrize('corruption',['optimizer','lr','rng'])
def test_recovery_report_never_passes_mutated_state(eval_assets,tmp_path,corruption):
    original=recovery.load_training_state
    def load(path,**kwargs):
        result=original(path,**kwargs)
        if Path(path).parent.name=='resumed' and result['global_step']==3:
            result=copy.deepcopy(result)
            if corruption=='optimizer':next(iter(result['optimizer']['state'].values()))['exp_avg']+=1
            if corruption=='lr':result['optimizer']['param_groups'][0]['lr']+=.001
            if corruption=='rng':result['rng']['torch'][0]^=1
        return result
    with patch.object(recovery,'load_training_state',side_effect=load),pytest.raises(AssertionError,match='optimizer|rng'):
        recovery.check(eval_assets['dataset'],eval_assets['tokenizer'],str(eval_assets['cfg']),config(eval_assets,tmp_path),tmp_path/'recovery',steps=3)
    assert not (tmp_path/'recovery').exists()


@pytest.mark.parametrize('domain',['python','numpy','torch','cuda'])
def test_rng_comparison_is_exact_for_all_domains(domain):
    state={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
           'cuda':[torch.tensor([1,2,3],dtype=torch.uint8)]}
    changed=copy.deepcopy(state)
    if domain=='python':changed[domain]=random.Random(987).getstate()
    elif domain=='numpy':changed[domain][1][0]^=1
    elif domain=='torch':changed[domain][0]^=1
    else:changed[domain][0][0]^=1
    with pytest.raises(AssertionError,match=domain):compare_tree(state,changed,exact=True,path='rng')


def test_optimizer_structure_hyperparameters_and_tolerances():
    value={'state':{0:{'step':torch.tensor(20.),'exp_avg':torch.tensor([.2]),'exp_avg_sq':torch.tensor([.3])}},
           'param_groups':[{'params':[0],'lr':.0001,'betas':(.9,.95),'eps':1e-8,'weight_decay':.1}]}
    changed=copy.deepcopy(value);changed['state'][0]['exp_avg']+=1e-7
    assert 0<compare_tree(value,changed)<1e-6
    for mutate in ('step','group','shape','missing'):
        changed=copy.deepcopy(value)
        if mutate=='step':changed['state'][0]['step']+=.00001
        elif mutate=='group':changed['param_groups'][0]['lr']=.0002
        elif mutate=='shape':changed['state'][0]['exp_avg']=torch.ones(2)
        else:del changed['state'][0]['exp_avg_sq']
        with pytest.raises(AssertionError):compare_tree(value,changed,path='optimizer')
