#!/usr/bin/env python3
"""Actual-model continuous vs two-resume check in isolated output directories."""
import argparse
import json
import signal
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import minillm.train as training
from minillm.assessment_io import report_directory
from minillm.corpus_utils import file_info
from minillm.data import EpochBatchSampler,collate_batch
from minillm.train import fit,load_model_checkpoint,load_training_state
from minillm.training_manifest import write_json
from minillm.utils import set_seed
from minillm.workload import resolve_plan


def check(dataset,tokenizer,model_config,config,out,steps=20):
    opts,extra,cfg,tok,train,val,_,device,_,_=resolve_plan(dataset,tokenizer,model_config,config)
    if steps<3 or not opts.init_from or opts.resume:
        raise ValueError('requires weights-only initialization and at least three steps')
    source=file_info(Path(opts.init_from))
    with report_directory(out) as directory:
        original=training._train_step
        traces={}
        def segment(current,stop,order,lrs):
            set_seed(current.seed)
            model,_,payload=load_model_checkpoint(current.init_from or opts.init_from,torch.device('cpu'))
            del payload
            sampler=EpochBatchSampler(len(train),current.batch_size,current.seed)
            def batches(epoch,cursor):
                for indices in sampler.batches(epoch,cursor):
                    order.extend((epoch,i) for i in indices)
                    yield collate_batch([train[i] for i in indices])
            batches.sampler_seed=current.seed
            count=load_training_state(current.resume,mmap=True)['global_step'] if current.resume else 0
            def step(*args,**kwargs):
                nonlocal count
                result=original(*args,**kwargs);lrs.append(args[-1]);count+=1
                if count==stop:
                    signal.raise_signal(signal.SIGTERM if stop==5 else signal.SIGINT)
                return result
            with patch.object(training,'_train_step',side_effect=step):
                summary=fit(current,model,tok,train,val,batch_factory=batches)
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
            return summary
        base=replace(opts,out=str(directory/'continuous'),max_steps=steps,val_every=0,save_every=0,log_every=1)
        full_order,full_lrs=[],[]
        full=segment(base,None,full_order,full_lrs)
        current=replace(base,out=str(directory/'resumed'))
        order,lrs=[],[]
        # 5/12 for the planned 20-update check; deterministic earlier boundaries for Nano tests.
        first=min(5,steps-2);second=min(12,steps-1)
        for stop in (first,second,None):
            resumed=segment(current,stop,order,lrs)
            current=replace(current,init_from=None,resume=str(Path(current.out)/'state.pt'))
        continuous=torch.load(directory/'continuous/model.pt',map_location='cpu',weights_only=False,mmap=True)
        restarted=torch.load(directory/'resumed/model.pt',map_location='cpu',weights_only=False,mmap=True)
        maximum=0.
        for name,v in continuous['model'].items():
            w=restarted['model'][name]
            torch.testing.assert_close(v,w,atol=1e-6,rtol=1e-5)
            maximum=max(maximum,float((v-w).abs().max()))
        keys=('global_step','epoch','micro_batches_consumed_in_epoch','samples_seen','tokens_seen')
        assert full_order==order and full_lrs==lrs
        assert all(full[k]==resumed[k] for k in keys)
        assert file_info(Path(opts.init_from))==source
        report={'status':'PASS','steps':steps,'interruptions':[first,second],
                'maximum_parameter_difference':maximum,'atol':1e-6,'rtol':1e-5,
                'sample_order_equal':True,'lr_equal':True,'progress':{k:full[k] for k in keys},
                'initial_checkpoint':source,'scope':'isolated recovery experiment; not part of pilot token count'}
        write_json(directory/'recovery.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','tokenizer','config','training-config','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--steps',type=int,default=20);args=p.parse_args()
    try:r=check(args.dataset,args.tokenizer,args.config,args.training_config,args.out,args.steps)
    except Exception as exc:print('FAIL recovery:',exc,file=sys.stderr);return 1
    print(json.dumps(r,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
