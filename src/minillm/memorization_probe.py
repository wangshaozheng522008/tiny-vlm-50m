"""Bounded memorization probes using fit and production packed pretraining data."""
from __future__ import annotations

import math
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import get_config
from .corpus import tokenize_split
from .diagnostic_common import Budget, MEMORIZATION, frozen_rows, model_identity
from .data import PackedPretrainDataset, collate_batch
from .generation_diagnostics import token_metrics, summarize_generations
from .model import MiniLLM
from .tokenizer import MiniLLMTokenizer
from .train import TrainingOptions, evaluate, fit, load_training_state, save_model_checkpoint
from .training_manifest import file_sha256
from .utils import set_seed


def load_memorization(directory=MEMORIZATION):
    train=frozen_rows(Path(directory)/'train.jsonl');heldout=frozen_rows(Path(directory)/'heldout.jsonl')
    if not 128<=len(train)<=256 or not 16<=len(heldout)<=64:
        raise ValueError('memorization requires 128..256 train and 16..64 withheld rows')
    seen_ids,seen_text=set(),set()
    for split,rows in [('train',train),('heldout',heldout)]:
        for row in rows:
            if (row.get('split')!=split or row.get('id') in seen_ids or row.get('text') in seen_text
                    or not all(isinstance(row.get(k),str) and row[k] for k in ('id','text','prefix','continuation','text_category'))
                    or row['text']!=row['prefix']+row['continuation'] or len(row['text'])>256):
                raise ValueError('invalid/duplicate memorization row or train/heldout overlap')
            seen_ids.add(row['id']);seen_text.add(row['text'])
    return train,heldout


def prepare_model(kind, device, train, *, seed, enable_87m=False, tokenizer_path=None, gpu_memory_budget_mib=None):
    if kind not in ('nano','87m') or device not in ('cpu','cuda'):
        raise ValueError('unknown diagnostic model/device')
    if kind=='87m' and (not enable_87m or not tokenizer_path):
        raise ValueError('87m requires --enable-87m and an existing --tokenizer; formal tokenizer training is forbidden')
    if device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if kind=='87m' and device!='cuda':
        raise ValueError('bounded 87m probes require CUDA and an explicit GPU memory budget')
    torch.set_num_threads(1)
    if tokenizer_path:
        tokenizer=MiniLLMTokenizer.load(tokenizer_path)
    else:
        # Diagnostic Nano tokenizer only; fit exclusively to training rows.
        tokenizer=MiniLLMTokenizer.train([r['text'] for r in train],vocab_size=512)
    cfg=get_config('minillm-'+kind)
    if kind=='nano':
        if tokenizer.vocab_size>4096:
            raise ValueError('CPU Nano diagnostic tokenizer must have <=4096 tokens (bounded memory)')
        cfg.vocab_size=tokenizer.vocab_size
    tokenizer.assert_vocab_size(cfg.vocab_size)
    memory={'status':'NOT_APPLICABLE_CPU_NANO'}
    if device=='cuda':
        estimate_mib=cfg.num_parameters*20/1024**2+768
        free,total=torch.cuda.mem_get_info()
        if gpu_memory_budget_mib is None or not math.isfinite(gpu_memory_budget_mib) or not estimate_mib<=gpu_memory_budget_mib<=free/1024**2:
            raise ValueError(f'GPU budget must cover conservative estimate {estimate_mib:.0f} MiB and not exceed free VRAM')
        memory={'estimated_mib':estimate_mib,'budget_mib':gpu_memory_budget_mib,'free_mib':free/1024**2,'total_mib':total/1024**2,
                'limitation':'Conservative admission estimate; allocator limit is also enforced.'}
        torch.cuda.set_per_process_memory_fraction(gpu_memory_budget_mib/(total/1024**2))
    set_seed(seed)
    return MiniLLM(cfg),tokenizer,memory


def pack_rows(path, tokenizer, rows):
    for row in rows:
        ids=tokenizer.encode(row['text'],category=row['text_category'])
        if any(t in tokenizer.special_ids.values() for t in ids):
            raise ValueError('memorization prose contains reserved special tokens; refusing accidental supervision')
    tokenize_split([{'text':tokenizer.prepare_text(r['text'],category=r['text_category']),
                     'category':r['text_category']} for r in rows],tokenizer,path)


@torch.inference_mode()
def sample_metrics(model, tokenizer, rows, *, generation_limit=16):
    """Teacher forcing covers every row; generation covers a fixed, declared subset."""
    model.eval();device=next(model.parameters()).device
    count=min(generation_limit,len(rows))
    selected={round(i*(len(rows)-1)/(count-1)) for i in range(count)} if count>1 else ({0} if count else set())
    correct=tokens=0;predictions=[];prefix_correct=prefix_tokens=exact=0
    for i,row in enumerate(rows):
        full=tokenizer.encode_document(row['text'],category=row['text_category'])
        if len(full)>model.cfg.max_position_embeddings:
            raise ValueError('memorization sample exceeds context; no silent truncation')
        ids=torch.tensor([full],device=device)
        out=model(ids)
        correct+=int((out.logits[0,:-1].argmax(-1)==ids[0,1:]).sum());tokens+=len(full)-1
        if i not in selected:continue
        prefix=tokenizer.encode(row['prefix'],category=row['text_category'])
        shared=0
        for a,b in zip(prefix,full):
            if a!=b:break
            shared+=1
        if not shared or shared==len(full):
            raise ValueError('memorization prefix has no shared conditioning token or target')
        expected=full[shared:];prompt=full[:shared]
        budget=min(64,model.cfg.max_position_embeddings-len(prompt))
        if len(expected)>budget:
            raise ValueError('continuation exceeds generation budget')
        generated=model.generate(torch.tensor([prompt],device=device),max_new_tokens=budget,
                                 temperature=0,top_k=0,eos_token_id=tokenizer.special_ids['eos'])[0,len(prompt):].tolist()
        matched=sum(a==b for a,b in zip(expected,generated));denominator=max(len(expected),len(generated))
        prefix_correct+=matched;prefix_tokens+=denominator;exact+=expected==generated
        predictions.append({'id':row['id'],'status':'GENERATED','prompt_token_ids':prompt,'expected_token_ids':expected,
                            'token_ids':generated,'exact':expected==generated,
                            'shared_prefix_tokens':shared,'independent_prefix_tokens':len(prefix),
                            **token_metrics(generated,tokenizer,max_new_tokens=budget)})
    return {'teacher_forced':{'correct':correct,'tokens':tokens,'accuracy':correct/tokens},
            'prefix_continuation':{'correct_tokens':prefix_correct,'tokens':prefix_tokens,
                'accuracy':prefix_correct/prefix_tokens if prefix_tokens else None,
                'exact':exact,'samples':len(predictions),'exact_rate':exact/len(predictions) if predictions else None,
                'selection':'Evenly spaced indices across the frozen row order, including endpoints; no output-based selection',
                'boundary_policy':'joint_shared_prefix of actual document encoding and fixed prefix; target includes EOS; extra/missing output tokens count against accuracy'},
            'generation':summarize_generations(predictions),'predictions':predictions}


def run_memorization(out, *, budget=Budget(), seed=42, kind='nano', device='cpu', enable_87m=False,
                     tokenizer_path=None, gpu_memory_budget_mib=None):
    budget.validate();out=Path(out);started=time.monotonic()
    train,heldout=load_memorization()
    model,tokenizer,memory=prepare_model(kind,device,train,seed=seed,enable_87m=enable_87m,
                                        tokenizer_path=tokenizer_path,gpu_memory_budget_mib=gpu_memory_budget_mib)
    tokenizer.save(out/'tokenizer.json')
    data=out/'data';data.mkdir()
    pack_rows(data/'train.bin',tokenizer,train);pack_rows(data/'heldout.bin',tokenizer,heldout)
    # One fixed packed sample, repeated unchanged in every epoch.
    fixed=tokenizer.encode_document(train[0]['text'],category=train[0]['text_category'])
    fixed=(fixed*math.ceil(65/len(fixed)))[:65]
    np.asarray(fixed,dtype='<u2').tofile(data/'single.bin')
    seq_len=64
    datasets={'single':PackedPretrainDataset(data/'single.bin',seq_len),
              'corpus':PackedPretrainDataset(data/'train.bin',seq_len)}
    # Exactly the same fresh initialization for both experiments; second model is allocated only after first is freed.
    initial=out/'initial.pt';save_model_checkpoint(initial,model,tokenizer,step=0,tokens=0,val_loss=None,
                                                  extra={'mode':'diagnostic_scratch','diagnostic_model':kind})
    initial_state={k:v.clone() for k,v in model.state_dict().items()}
    results={};used_steps=used_tokens=0
    for name,target in [('single',max(1,budget.max_steps//2)),('corpus',budget.max_steps-budget.max_steps//2)]:
        set_seed(seed)
        model.load_state_dict(initial_state)
        dataset=datasets[name];batch_size=1 if name=='single' else 4
        steps=min(target,(budget.max_effective_tokens-used_tokens)//(batch_size*(seq_len-1)))
        if steps<1:
            results[name]={'status':'NOT_RUN_TOKEN_BUDGET'};continue
        model.to(device)
        before=evaluate(model,dataset,torch.device(device),batch_size=batch_size)
        before_metrics=sample_metrics(model,tokenizer,train if name=='corpus' else train[:1],generation_limit=0)
        independent_before=sample_metrics(model,tokenizer,heldout,generation_limit=0) if name=='corpus' else None
        remaining=budget.max_wall_seconds-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('memorization wall budget exhausted')
        opts=TrainingOptions(out=str(out/name),max_steps=steps,batch_size=batch_size,grad_accum=1,
            lr=0.003 if kind=='nano' else 0.0003,min_lr=0.0003 if kind=='nano' else 0.00003,
            warmup_ratio=.02,device=device,precision='fp32',threads=1,seed=seed,
            max_minutes=remaining/60,log_every=1,save_every=0,val_every=0,
            extra_meta={'scope':'memorization only; not v4 pretraining'})
        summary=fit(opts,model,tokenizer,dataset,None)
        used_steps+=summary['steps'];used_tokens+=summary['tokens_seen']
        after=evaluate(model,dataset,torch.device(device),batch_size=batch_size)
        curves=[json.loads(line) for line in (out/name/'training-log.jsonl').read_text().splitlines()]
        curves=[r for r in curves if r['event']=='train']
        state=load_training_state(out/name/'state.pt',mmap=True)
        finite=all(torch.isfinite(v).all() for v in state['model'].values())
        optimizer_finite=all(torch.isfinite(v).all() for s in state['optimizer']['state'].values() for v in s.values() if torch.is_tensor(v))
        updated=any(not torch.equal(v,initial_state[k]) for k,v in state['model'].items())
        gradients=all(math.isfinite(r['gradient_norm']) for r in curves)
        reduction=1-after['loss']/before['loss']
        results[name]={'status':'COMPLETED','initial_loss':before['loss'],'final_loss':after['loss'],
            'loss_reduction_fraction':reduction,'gradients_finite':gradients,'parameters_finite':bool(finite),
            'parameters_updated':updated,'optimizer_finite':bool(optimizer_finite),'initial_teacher_forced':before_metrics['teacher_forced'],
            'initial_independent_teacher_forced':independent_before['teacher_forced'] if independent_before else None,
            'curve':[{'step':r['global_step'],'loss':r['loss'],'lr':r['learning_rate'],'effective_tokens':r['tokens_seen']} for r in curves],
            'summary':summary,'options':asdict(opts),'model':model_identity(model,out/name/'model.pt'),
            'training_samples':sample_metrics(model,tokenizer,train if name=='corpus' else train[:1]),
            'independent_samples':sample_metrics(model,tokenizer,heldout,generation_limit=8) if name=='corpus' else None}
        del state
    single=results['single'];corpus=results['corpus']
    engineering=all(r.get('status')=='COMPLETED' and all(r.get(k) for k in ('gradients_finite','parameters_finite','parameters_updated','optimizer_finite')) for r in results.values())
    learned=engineering and single['loss_reduction_fraction']>=.5
    return {'schema_version':1,'model_kind':kind,'seed':seed,'budget':asdict(budget),'memory_admission':memory,
            'initial_model':model_identity(model,initial),'tokenizer_fingerprint':tokenizer.fingerprint,'text_policy':tokenizer.text_policy,
            'data_sha256':{p.name:file_sha256(p) for p in data.glob('*.bin')},
            'training_rows':len(train),'heldout_rows':len(heldout),'training_steps':used_steps,'effective_tokens':used_tokens,
            'experiments':results,'engineering':'ENGINEERING_PASS' if engineering else 'ENGINEERING_FAIL',
            'learning':'LEARNING_DEMONSTRATED' if learned else 'LEARNING_INCONCLUSIVE',
            'failure_localization':None if learned else {
                'single_status':single.get('status'),'single_loss_reduction':single.get('loss_reduction_fraction'),
                'threshold':.5,'steps_completed':used_steps,'effective_tokens':used_tokens,
                'reason':'Nonfinite or unchanged training state' if not engineering else 'Finite updates observed, but declared budget did not demonstrate >=50% fixed-batch loss reduction; insufficient learning evidence, not proof of a broken optimizer.'},
            'interpretation':'Training-set memorization and independent heldout performance are separate. Heldout uses one withheld prose template, not a representative generalization benchmark. Scratch 87m memorization never establishes real pretrained 87m capability.'}
