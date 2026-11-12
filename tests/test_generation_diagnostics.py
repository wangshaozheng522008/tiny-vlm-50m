"""Token metrics at empty/EOS/repetition boundaries and separate deterministic decoders."""
import pytest
from minillm.generation_diagnostics import token_metrics,diagnose_generation
from minillm.model import MiniLLM


def test_empty_eos_and_illegal_tokens(tiny_tokenizer):
    empty=token_metrics([],tiny_tokenizer,max_new_tokens=8)
    assert empty['empty'] and empty['distinct_2'] is None and empty['token_4gram_repeat_ratio']==0
    eos=token_metrics([2],tiny_tokenizer,max_new_tokens=8)
    assert eos['early_eos'] and eos['empty'] and not eos['special_token_anomalies']
    with pytest.raises(ValueError,match='illegal'):token_metrics([tiny_tokenizer.vocab_size],tiny_tokenizer,max_new_tokens=8)
    with pytest.raises(ValueError,match='after EOS'):token_metrics([2,10],tiny_tokenizer,max_new_tokens=8)


def test_repetition_loop_and_special_anomalies(tiny_tokenizer):
    r=token_metrics([11]*8,tiny_tokenizer,max_new_tokens=8)
    assert r['token_4gram_repeat_ratio']==.8 and r['longest_identical_run']==8 and r['tail_loop_period']==1
    r=token_metrics([10,11]*4,tiny_tokenizer,max_new_tokens=8)
    assert r['tail_loop_period']==2 and r['hit_length_limit']
    assert token_metrics([0,1,4],tiny_tokenizer,max_new_tokens=8)['special_token_anomalies']==[0,1,4]


def test_decoder_modes_reproduce_separately(tiny_tokenizer,nano_cfg):
    model=MiniLLM(nano_cfg).eval();tasks=[{'id':'p','prompt':'今天天气','text_category':'zh'}]
    a=diagnose_generation(model,tiny_tokenizer,tasks,seed=17,max_new_tokens=8)
    b=diagnose_generation(model,tiny_tokenizer,tasks,seed=17,max_new_tokens=8)
    assert a==b and set(a['modes'])=={'greedy','temperature','top_k'}
    assert all(r['summary']['generated']==1 for r in a['modes'].values())


def test_context_overflow_marked_unscorable(tiny_tokenizer,nano_cfg):
    a=diagnose_generation(MiniLLM(nano_cfg),tiny_tokenizer,[{'id':'long','prompt':'语言'*200}],max_new_tokens=8)
    assert all(r['summary']['unscorable']==1 and r['summary']['empty_output']['denominator']==0 for r in a['modes'].values())
