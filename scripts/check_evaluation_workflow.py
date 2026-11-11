#!/usr/bin/env python3
"""Reproduce PR4 examples with synthetic data and eight Nano updates (not a pilot)."""
import argparse
import contextlib
import io
import json
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import torch
from minillm.assessment_io import read_json
from minillm.config import get_config
from minillm.corpus import Pipeline, inventory, make_content
from minillm.data import PackedPretrainDataset
from minillm.evaluation import EvaluationConfig, evaluate_model
from minillm.evaluation_compare import compare_evaluations
from minillm.model import MiniLLM
from minillm.tokenizer import MiniLLMTokenizer
from minillm.train import TrainingOptions, fit, save_model_checkpoint
from minillm.training_analysis import analyze_training
from minillm.training_pilot import evaluate_pilot
from minillm.training_readiness import check_readiness
from minillm.training_manifest import write_json
from minillm.utils import set_seed


def check(out):
    out=Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('choose a new --out; demo never overwrites existing artifacts')
    out.mkdir(parents=True,exist_ok=True)
    rng=random.Random(81)
    raw=out/'synthetic.jsonl'
    raw.write_text(''.join(json.dumps({'text':''.join(rng.choices('人工智能模型数据学习科学数学未来世界方法知识天气城市',k=160))+str(i),'source':'synthetic-only','license':'synthetic-only'},ensure_ascii=False)+'\n' for i in range(120)))
    data=out/'dataset'
    pipe=Pipeline(data,make_content({'split':[.6,.2,.2],'deduplication':{'method':'exact'}}),inventory([str(raw)]),log=lambda *_:None)
    pipe.tokenize(vocab_size=264)
    tokenizer_path=data/'tokenizer.json';tokenizer=MiniLLMTokenizer.load(tokenizer_path)
    # Explicit Nano variant: same weights architecture, context512 stored in checkpoint.
    # This is never silently applied when evaluating an existing context128 checkpoint.
    cfg=get_config('minillm-nano');cfg.vocab_size=tokenizer.vocab_size;cfg.max_position_embeddings=512
    cfg_path=out/'nano-context512.json';cfg.save(cfg_path)
    torch.set_num_threads(1);set_seed(42);model=MiniLLM(cfg)
    initial=out/'initial.pt'
    save_model_checkpoint(initial,model,tokenizer,step=0,tokens=0,val_loss=None,extra={'mode':'untrained'})
    evaluation_config=EvaluationConfig(seq_len=32,token_budget=256,max_new_tokens=8)
    write_json(out/'evaluation-config.json',asdict(evaluation_config))
    baseline=evaluate_model(initial,tokenizer_path,out/'initial-evaluation',dataset=data,config=evaluation_config)
    opts=TrainingOptions(out=str(out/'run'),max_steps=8,batch_size=2,grad_accum=2,device='cpu',precision='fp32',threads=1,seed=42,log_every=1,val_every=1,save_every=2)
    summary=fit(opts,model,tokenizer,PackedPretrainDataset(data/'train.bin',32),PackedPretrainDataset(data/'val.bin',32))
    final=evaluate_model(out/'run/model.pt',tokenizer_path,out/'final-evaluation',dataset=data,config=evaluation_config)
    comparison=compare_evaluations([out/'initial-evaluation',out/'final-evaluation'],out/'comparison')
    analysis=analyze_training(out/'run',out/'training-analysis')
    fresh={**asdict(opts),'out':str(out/'prospective-run'),'seq_len':32}
    write_json(out/'training-config.json',fresh)
    ready=check_readiness(data,tokenizer_path,str(cfg_path),fresh,out/'ready')
    unknown=check_readiness(data,tokenizer_path,str(cfg_path),{**fresh,'gpu_memory_limit_mib':1024},out/'incomplete')
    bad=out/'fault-injection-dataset';shutil.copytree(data,bad)
    (bad/'val.bin').write_bytes(b'fault injection')
    blocked=check_readiness(bad,tokenizer_path,str(cfg_path),fresh,out/'blocked')
    pilot=evaluate_pilot(out/'run',out/'initial-evaluation',out/'final-evaluation',out/'pilot-acceptance')
    assert ready['status']=='READY' and unknown['status']=='INCOMPLETE' and blocked['status']=='BLOCKED'
    assert analysis['status']=='OK' and pilot['engineering']=='ENGINEERING_PASS'
    result={'scope':'Synthetic Nano context512, 8 CPU updates; NOT a 10M-token pilot or 87M capability result',
            'initial_lm':baseline['language_model'],'final_lm':final['language_model'],
            'initial_scores':baseline['scores'],'final_scores':final['scores'],
            'actual_training_tokens':summary['tokens_seen'],'global_step':summary['global_step'],
            'readiness':[ready['status'],blocked['status'],unknown['status']],
            'engineering':pilot['engineering'],'convergence':pilot['convergence'],'capability':pilot['capability'],
            'curve_rows':analysis['curve_rows'],'cuda_probe':ready['details'].get('cuda_operator_probe',{'status':'NOT_RUN'}),
            'files':{'initial':'initial-evaluation/evaluation.json','final':'final-evaluation/evaluation.json','comparison':'comparison/comparison.md','curves':'training-analysis/curves/training.csv'}}
    write_json(out/'example-summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);args=p.parse_args()
    try:
        result=check(args.out)
    except Exception as exc:
        print(f'FAIL evaluation workflow: {exc}',file=sys.stderr);return 1
    print(json.dumps(result,ensure_ascii=False,indent=2));print('PASS evaluation workflow');return 0

if __name__=='__main__':
    raise SystemExit(main())
