"""Policy identity, code safety and training/inference consistency."""
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

from minillm.corpus import Pipeline, inventory, make_content, read_records
from minillm.data import PackedPretrainDataset, SFTDataset
from minillm.model import MiniLLM
from minillm.quantize import save_quantized
from minillm.text_policy import IDENTITY, WIDTH_FOLD, apply_text_policy, checkpoint_text_policy
from minillm.tokenizer import MiniLLMTokenizer
from minillm.train import TrainingOptions, fit, save_model_checkpoint
from minillm.training_manifest import dataset_identity
from minillm.checkpoint import CheckpointError
from minillm.evaluation import EvaluationConfig, evaluate_model
from minillm.workload import resolve_plan


def test_conservative_width_fold():
    text = '中文，数字１２３：x² ≥ ０；`print("Ａ")` 与 $x²＋１$。'
    expected = '中文,数字123:x² ≥ 0;`print("Ａ")` 与 $x²＋１$。'
    assert apply_text_policy(text) == text
    assert apply_text_policy(text, WIDTH_FOLD) == expected
    assert apply_text_policy(expected, WIDTH_FOLD) == expected
    for code in ['def foo():\n    return "Ａ，１２"\n', 'x = "Ａ，Ｂ"', '```python\nprint("Ａ")\n```']:
        assert apply_text_policy(code, WIDTH_FOLD) == code
    assert apply_text_policy('Ａ，Ｂ', WIDTH_FOLD, category='code') == 'Ａ，Ｂ'
    with pytest.raises(ValueError, match='unknown text policy'):
        apply_text_policy('text', 'nfkc-unversioned')


def test_tokenizer_cli_no_hidden_nfkc(tmp_path):
    script = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/train_tokenizer.py'))
    text = '中文，数字１２３，数学x²与代码保持一致。' * 3
    path = tmp_path/'raw.jsonl'; path.write_text(json.dumps({'text':text},ensure_ascii=False)+'\n')
    documents = list(script['weighted_documents']({'zh':[path]}, {'zh':1}, 1000, 42))
    assert documents == [text] == script['build_sample']([path])
    tokenizer = MiniLLMTokenizer.train(documents, vocab_size=264)
    assert tokenizer._tok.normalizer is None
    assert tokenizer.decode(tokenizer.encode(text)) == text


def policy_dataset(tmp_path, tiny_tokenizer):
    raw = tmp_path/'raw.jsonl'
    raw.write_text(''.join(json.dumps({'text':f'第{i}个文档：人工智能，数字１２３与x²。','source':'synthetic'},ensure_ascii=False)+'\n' for i in range(20)))
    tok = tmp_path/'tokenizer.json';tiny_tokenizer.save(tok)
    content = make_content({'text_policy':WIDTH_FOLD,'split':[.6,.2,.2],'quality':{'min_chars':1},'deduplication':{'method':'exact'}})
    pipe = Pipeline(tmp_path/'dataset',content,inventory([str(raw)]),log=lambda *_:None)
    manifest = pipe.tokenize(tok)
    return pipe, manifest, tok


def test_pipeline_fit_encode_and_model_policy(tmp_path,tiny_tokenizer,nano_cfg):
    pipe, manifest, tok = policy_dataset(tmp_path,tiny_tokenizer)
    encoded = []
    for row in read_records(pipe.out/'train.text.jsonl'):
        assert '１２３' not in row['text'] and 'x²' in row['text']
        encoded.extend(tiny_tokenizer.encode_document(row['text']))
    assert np.fromfile(pipe.out/'train.bin',dtype='<u2').tolist() == encoded
    ds = PackedPretrainDataset(pipe.out/'train.bin',16)
    info,_ = dataset_identity(ds,-100)
    assert info['text_policy'] == WIDTH_FOLD
    torch.set_num_threads(1)
    model = MiniLLM(nano_cfg)
    fit(TrainingOptions(out=str(tmp_path/'run'),max_steps=1,batch_size=1,grad_accum=1,
        device='cpu',precision='fp32',threads=1),model,tiny_tokenizer,ds,None)
    payload=torch.load(tmp_path/'run/model.pt',weights_only=False)
    state=torch.load(tmp_path/'run/state.pt',weights_only=False)
    assert checkpoint_text_policy(payload) == checkpoint_text_policy(state) == WIDTH_FOLD
    cfg_path=tmp_path/'config.json';nano_cfg.save(cfg_path)
    opts=dict(out=str(tmp_path/'planned'),init_from=str(tmp_path/'run/model.pt'),seq_len=16,
              max_steps=1,batch_size=1,grad_accum=1,threads=1,device='cpu',precision='fp32')
    # A published profile is also visible to the workload planner.
    assert resolve_plan(pipe.out,tok,str(cfg_path),opts)[-1]['manifest']['dataset']['train']['text_policy']==WIDTH_FOLD
    quant=tmp_path/'quant.pt';save_quantized(quant,model,4,tokenizer_fingerprint=tiny_tokenizer.fingerprint,metadata=payload)
    assert checkpoint_text_policy(torch.load(quant,weights_only=False)) == WIDTH_FOLD
    # Existing weights cannot silently switch to a different input contract.
    legacy=tmp_path/'legacy.pt';save_model_checkpoint(legacy,model,tiny_tokenizer,step=0,tokens=0,val_loss=None)
    with pytest.raises(CheckpointError,match='text policy differs'):
        fit(TrainingOptions(out=str(tmp_path/'bad'),init_from=str(legacy),max_steps=1,device='cpu',precision='fp32'),model,tiny_tokenizer,ds,None)
    assert not (tmp_path/'bad').exists()
    with pytest.raises(ValueError,match='text policy differs'):
        resolve_plan(pipe.out,tok,str(cfg_path),{**opts,'init_from':str(legacy)})
    with pytest.raises(ValueError,match='text policy mismatch'):
        evaluate_model(legacy,tok,tmp_path/'bad-eval',dataset=pipe.out,split='val',benchmark=None,prompts=None,
                       config=EvaluationConfig(seq_len=16))


def test_runtime_chat_and_raw_share_policy_without_changing_vocab(tmp_path,tiny_tokenizer):
    tokenizer=MiniLLMTokenizer.load(tiny_tokenizer_path(tmp_path,tiny_tokenizer))
    before=tokenizer.fingerprint
    raw='你好，１２３。'
    legacy=tokenizer.encode(raw)
    tokenizer.text_policy=WIDTH_FOLD
    assert tokenizer.encode(raw)==tokenizer.encode('你好,123。')
    assert tokenizer.encode(raw)!=legacy
    a=tokenizer.chat_encoding([{'role':'user','content':raw},{'role':'assistant','content':raw}])
    b=tokenizer.chat_encoding([{'role':'user','content':'你好,123。'},{'role':'assistant','content':'你好,123。'}])
    assert a.ids==b.ids and a.labels==b.labels
    assert tokenizer.fingerprint==before
    ds=SFTDataset(tmp_path/'sft.jsonl',tokenizer,128,records=[{'messages':[{'role':'user','content':raw},{'role':'assistant','content':raw}]}])
    (tmp_path/'sft.jsonl').write_text('{}\n')
    assert dataset_identity(ds,-100)[0]['text_policy']==WIDTH_FOLD
    assert checkpoint_text_policy({})==IDENTITY
    with pytest.raises(ValueError,match='disagrees'):
        checkpoint_text_policy({'text_policy':IDENTITY,'manifest':{'dataset':{'train':{'text_policy':WIDTH_FOLD}}}})


def tiny_tokenizer_path(tmp_path,tokenizer):
    path=tmp_path/'tokenizer.json';tokenizer.save(path);return path


def test_runtime_policy_cannot_mislabel_legacy_training(tmp_path,tiny_tokenizer,nano_cfg,packed_data):
    tokenizer=MiniLLMTokenizer.load(tiny_tokenizer_path(tmp_path,tiny_tokenizer))
    tokenizer.text_policy=WIDTH_FOLD
    with pytest.raises(CheckpointError,match='runtime tokenizer text policy differs'):
        fit(TrainingOptions(out=str(tmp_path/'bad'),max_steps=1,device='cpu',precision='fp32'),
            MiniLLM(nano_cfg),tokenizer,PackedPretrainDataset(packed_data[0],16),None)
    assert not (tmp_path/'bad').exists()
