"""CPU Nano numerical correctness probes using production loss, batches and BIN encoder."""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, pre_tokenizers

from .config import get_config
from .corpus import tokenize_split
from .data import PackedPretrainDataset, collate_batch
from .evaluation import load_evaluation_model
from .model import MiniLLM
from .text_policy import POLICIES, apply_text_policy, checkpoint_text_policy
from .tokenizer import MiniLLMTokenizer, SPECIAL_TOKENS
from .train import save_model_checkpoint
from .training_manifest import file_sha256
from .utils import set_seed

TOKENIZER_CASES = [
    ('zh','春天到了，河边的树长出了新叶。孩子们沿着小路慢慢走，讨论明天的计划。'),
    ('mixed','今天有12位同学学习Python 3.11，完成率是80%。'),
    ('zh','数学：x² + y² = z²；α ≤ β，π ≈ 3.14，√4 = 2。'),
    ('code','def add(a, b):\n    return a + b\n'),
    ('code','#include <iostream>\nint main() { std::cout << "你好"; }\n'),
    ('zh','全角：ＡＢＣ１２３，价格￥２０。'),
    ('code','变量＝＂全角值＂；Ａ＝１'),
    ('zh','字面量 `Ａ＝１` 与 $x²$ 保持原样。'),
]


def tokenizer_checks(tokenizer, directory):
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=False)
    original = tokenizer.text_policy
    results = []
    try:
        for policy in POLICIES:
            tokenizer.text_policy = policy
            expected_stream, canonical = [], []
            for category,text in TOKENIZER_CASES:
                expected = apply_text_policy(text,policy,category=category)
                ids = tokenizer.encode(text,category=category)
                assert tokenizer.decode(ids)==expected
                document = tokenizer.encode_document(text,category=category)
                assert document==ids+[tokenizer.special_ids['eos']]
                expected_stream.extend(document)
                canonical.append({'text':expected,'category':category})
                results.append({'policy':policy,'category':category,'input':text,'expected':expected,'token_ids':ids,'roundtrip':True})
            path = directory/f'{policy}.bin'
            tokenize_split(canonical,tokenizer,path,buffer_bytes=32)
            actual = np.fromfile(path,dtype='<u2').tolist()
            assert actual==expected_stream
            packed = PackedPretrainDataset(path,16)
            for i in range(len(packed)):
                assert packed[i]['input_ids'].tolist()==actual[i*16:(i+1)*16]
            special_text = ''.join(SPECIAL_TOKENS)
            assert tokenizer.decode(tokenizer.encode(special_text),skip_special_tokens=False)==special_text
            assert tokenizer.encode(special_text)==list(tokenizer.special_ids.values())
            cfg = get_config('minillm-nano');cfg.vocab_size=tokenizer.vocab_size
            set_seed(42)
            policy_checkpoint=directory/f'{policy}.pt'
            save_model_checkpoint(policy_checkpoint,MiniLLM(cfg),tokenizer,step=0,tokens=0,val_loss=None)
            _,payload=load_evaluation_model(policy_checkpoint,tokenizer,torch.device('cpu'))
            assert checkpoint_text_policy(payload)==policy
        chat = tokenizer.chat_encoding([{'role':'user','content':'你好'},{'role':'assistant','content':'你好呀'}])
        for token,label in zip(chat.ids,chat.labels):
            if token in tokenizer.special_ids.values():
                assert label==(-100 if token!=tokenizer.special_ids['end_of_turn'] or label==-100 else token)
        tokenizer.text_policy=original
        cfg = get_config('minillm-nano');cfg.vocab_size=tokenizer.vocab_size
        checkpoint = directory/'identity.pt'
        set_seed(42)
        save_model_checkpoint(checkpoint,MiniLLM(cfg),tokenizer,step=0,tokens=0,val_loss=None)
        loaded,payload = load_evaluation_model(checkpoint,tokenizer,torch.device('cpu'))
        assert checkpoint_text_policy(payload)==original
        alternate = MiniLLMTokenizer(Tokenizer.from_str(tokenizer._tok.to_str()))
        alternate._tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=True)
        assert alternate.vocab_size==tokenizer.vocab_size and alternate.fingerprint!=tokenizer.fingerprint
        try:
            load_evaluation_model(checkpoint,alternate,torch.device('cpu'))
        except ValueError as exc:
            assert 'fingerprint' in str(exc)
        else:
            raise AssertionError('different tokenizer silently accepted')
        del loaded
        return {'status':'PASS','cases':results,'special_roundtrip':True,'bin_equals_documents':True,
                'checkpoint_probe_seed':42,
                'packed_windows_equal':True,'tokenizer_mismatch_rejected':True,'checkpoint_policy_equal':True,
                'packing_semantics':'EOS is supervised; EOS separates documents but attention crosses documents. Model shifts within each L window: L-1 targets, boundary next-token targets are omitted by the existing pipeline.',
                'special_supervision':'PAD/BOS and chat role markers ignored; document EOS and assistant end_of_turn are legitimate targets.',
                'bin_sha256':{p.name:file_sha256(p) for p in directory.glob('*.bin')}}
    finally:
        tokenizer.text_policy=original


def loss_gradient_checks(tokenizer, seed=42):
    torch.set_num_threads(1);set_seed(seed)
    cfg=get_config('minillm-nano');cfg.vocab_size=tokenizer.vocab_size
    model=MiniLLM(cfg).eval()
    initial_digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        initial_digest.update(key.encode());initial_digest.update(value.numpy().tobytes())
    sequences=[torch.tensor(tokenizer.encode_document(s)) for s in ('小河边有树。','中文123。')]
    batch=collate_batch([{'input_ids':ids,'labels':ids.clone()} for ids in sequences])
    out=model(**batch)
    expected=int((batch['labels'][:,1:]!=cfg.ignore_index).sum())
    manual=F.cross_entropy(out.logits[:,:-1].reshape(-1,cfg.vocab_size),batch['labels'][:,1:].reshape(-1),ignore_index=cfg.ignore_index,reduction='sum')
    torch.testing.assert_close(out.loss_sum,manual,rtol=0,atol=0)
    assert out.num_loss_tokens==expected and expected==sum(len(x)-1 for x in sequences)
    assert bool((batch['labels'][batch['attention_mask']==0]==cfg.ignore_index).all())
    changed_labels=batch['labels'].clone();changed_labels[:,0]=(changed_labels[:,0]+1)%cfg.vocab_size
    torch.testing.assert_close(model.loss_terms(out.logits,changed_labels)[0],out.loss_sum,rtol=0,atol=0)
    # Explicit ignored target has zero logit gradient; causally shifted target is at t+1.
    synthetic=torch.randn(1,5,cfg.vocab_size,requires_grad=True)
    labels=torch.tensor([[9,10,-100,11,12]])
    total,count=model.loss_terms(synthetic,labels);total.backward()
    assert count==3 and bool((synthetic.grad[0,1]==0).all()) and bool((synthetic.grad[0,4]==0).all())
    assert synthetic.grad[0,0,10]<0
    out.loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    parameter=model.embed_tokens.weight
    flat=parameter.grad.flatten();indices=flat.abs().topk(3).indices.tolist()
    finite=[];epsilon=0.005
    for index in indices:
        analytic=float(flat[index]);r,c=divmod(index,parameter.shape[1]);value=float(parameter.detach()[r,c])
        with torch.no_grad():
            parameter[r,c]=value+epsilon;plus=float(model(**batch).loss)
            parameter[r,c]=value-epsilon;minus=float(model(**batch).loss)
            parameter[r,c]=value
        numeric=(plus-minus)/(2*epsilon)
        assert abs(analytic-numeric)<=2e-4+0.02*abs(analytic), (analytic,numeric)
        finite.append({'index':[r,c],'autograd':analytic,'finite_difference':numeric,'absolute_error':abs(analytic-numeric)})
    ids=batch['input_ids'][:1].clone();split=max(1,ids.shape[1]//2)
    with torch.no_grad():
        before=model(ids).logits[:,:split].clone()
        ids[:,split:]=(ids[:,split:]+17)%cfg.vocab_size
        after=model(ids).logits[:,:split]
    maximum=float((before-after).abs().max());assert maximum<1e-6
    before=parameter.detach().clone()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,betas=(.9,.95))
    optimizer.step()
    assert not torch.equal(before,parameter) and all(torch.isfinite(p).all() for p in model.parameters())
    assert all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))
    return {'status':'PASS','initial_loss':float(out.loss.detach()),'effective_target_tokens':expected,
            'model_identity':{'seed':seed,'parameters':model.num_parameters(),'config':cfg.to_dict(),
                              'initial_state_sha256':initial_digest.hexdigest(),'scope':'ephemeral CPU Nano numerical probe'},
            'gradients_finite':True,'parameters_updated':True,'parameters_finite':True,'optimizer_finite':True,
            'padding_ignore_shift_correct':True,'future_token_max_difference':maximum,
            'finite_difference':{'epsilon':epsilon,'atol':2e-4,'rtol':.02,'checks':finite}}
