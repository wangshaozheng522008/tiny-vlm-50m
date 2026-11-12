"""CLI orchestration and reviewable reports; all model computation is delegated to existing modules."""
from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .chinese_foundation_eval import evaluate_foundation, load_foundation
from .diagnostic_common import (Budget, FOUNDATION, MEMORIZATION, ROOT, base_manifest, file_identity,
                                model_identity, new_output, wall_deadline, write_failure, cpu_memory_limit, code_identity)
from .evaluation import load_evaluation_model
from .generation_diagnostics import DECODERS, diagnose_generation
from .memorization_probe import run_memorization
from .text_policy import checkpoint_text_policy
from .tokenizer import MiniLLMTokenizer
from .training_diagnostics import loss_gradient_checks, tokenizer_checks
from .training_manifest import file_sha256, write_json


def evaluation_protocol(model, tokenizer, reviews=None, *, seed=42):
    return {'benchmark_sha256':{p.name:file_sha256(p) for p in FOUNDATION.iterdir() if p.is_file()},
            'tokenizer_fingerprint':tokenizer.fingerprint,'text_policy':tokenizer.text_policy,
            'boundary_policy':'separate','normalization':'sum','model_config':model.cfg.to_dict(),
            'decoding':{'modes':DECODERS,'max_new_tokens':24,'early_eos_tokens':4,'use_cache':True,'stop':'EOS','seed':seed,'per_prompt_seed':'seed+frozen_index'},
            'implementation_sha256':code_identity()['implementation_sha256'],
            'review_sha256':file_sha256(Path(reviews)) if reviews else None}


def evaluate_checkpoint(checkpoint, tokenizer_path, out, *, seed=42, device='cpu', reviews=None):
    tokenizer=MiniLLMTokenizer.load(tokenizer_path)
    model,payload=load_evaluation_model(checkpoint,tokenizer,torch.device(device))
    tokenizer.text_policy=checkpoint_text_policy(payload)
    _,prompts=load_foundation()
    candidates=evaluate_foundation(model,tokenizer,reviews=reviews)
    generation=diagnose_generation(model,tokenizer,prompts,seed=seed)
    write_json(out/'candidate-results.json',candidates)
    write_json(out/'generation-results.json',generation)
    # A separate blank review sheet captures observable text without auto-judging semantics.
    with (out/'generation-review.jsonl').open('x',encoding='utf-8') as handle:
        for name,result in generation['modes'].items():
            for row in result['rows']:
                handle.write(json.dumps({'id':row['id'],'mode':name,'text':row.get('text'),
                    'action':'REVIEW_PENDING','readability':None,'reviewer':'','reason':''},ensure_ascii=False)+'\n')
    return candidates,generation,{'model':model_identity(model,checkpoint),
        'tokenizer':file_identity(tokenizer_path),'protocol':evaluation_protocol(model,tokenizer,reviews,seed=seed),
        'checkpoint_mode':payload.get('mode'),'scope':'Read-only checkpoint inference. Diagnostic scratch models cannot establish pretrained 87m capability.'}


def report(out, manifest, *, training=None, memorization=None, candidates=None, generation=None):
    engineering=training.get('engineering','ENGINEERING_FAIL') if training else (memorization or {}).get('engineering','NOT_RUN')
    learning=(memorization or {}).get('learning','NOT_RUN')
    evaluation=(candidates or {}).get('status','NOT_RUN')
    criteria={'training_chain_correct':engineering=='ENGINEERING_PASS',
              'nano_learning_demonstrated':learning=='LEARNING_DEMONSTRATED' and (memorization or {}).get('model_kind')=='nano',
              'candidate_review_complete':evaluation=='EVALUATION_REVIEWED',
              'raw_readability_review_complete':False,
              'v4_data_tokenizer_plan_reviewed':False}
    gate={'engineering':engineering,'learning':learning,'evaluation':evaluation,
          'criteria':criteria,'ready_for_pr8':all(criteria.values()),
          'required_next':['Human review of frozen questions and raw continuations',
                           'Versioned v4 tokenizer/corpus plan and contamination audit',
                           'Explicit bounded 87m hardware probe before any real training'],
          'reason':'This baseline is not authorization or evidence for a formal v4 training run.'}
    write_json(out/'pr8-readiness.json',gate)
    lines=['# MiniLLM training and Chinese foundation diagnostics','',
           f"Engineering: **{engineering}**; learning: **{learning}**; evaluation: **{evaluation}**.",'',
           f"Seed: {manifest['seed']}; Git: `{manifest['code']['git_commit']}`; dirty: {manifest['code']['git_dirty']}.",
           f"Environment: `{json.dumps(manifest['environment'],ensure_ascii=False)}`.",'',
           'All identities, configurations, token IDs, NLLs, loss curves and per-item results are in the adjacent JSON files.']
    if training:
        lines+=['',f"Correctness: tokenizer {training['tokenizer']['status']}; gradients/loss {training['loss_gradient']['status']}; resume {training['recovery']['status']}.",
                f"Resume parameter maximum difference: {training['recovery']['max_parameter_difference']}; AdamW maximum difference: {training['recovery']['optimizer_max_difference']}."]
    if memorization:
        lines+=['',f"Model: {memorization['model_kind']}; parameters: {memorization['initial_model']['parameters']}; initial checkpoint SHA256: `{memorization['initial_model']['checkpoint']['sha256']}`.",
                f"Tokenizer fingerprint: `{memorization['tokenizer_fingerprint']}`; policy: `{memorization['text_policy']}`.",
                f"Training: {memorization['training_steps']} updates, {memorization['effective_tokens']} effective targets.",
                '', '| Experiment | Initial loss | Final loss | Reduction | Teacher forced accuracy | Prefix exact |',
                '|---|---:|---:|---:|---:|---:|']
        for name,r in memorization['experiments'].items():
            if r.get('status')!='COMPLETED':continue
            a=r['training_samples'];p=a['prefix_continuation']
            lines.append(f"| {name} | {r['initial_loss']:.6f} | {r['final_loss']:.6f} | {r['loss_reduction_fraction']:.2%} | {a['teacher_forced']['accuracy']:.2%} | {p['exact']}/{p['samples']} |")
            with (out/f'{name}-loss.csv').open('x') as handle:
                handle.write('step,loss,learning_rate,effective_tokens\n')
                for point in r['curve']:handle.write(f"{point['step']},{point['loss']},{point['lr']},{point['effective_tokens']}\n")
        independent=memorization['experiments']['corpus'].get('independent_samples')
        if independent:
            a=independent['teacher_forced'];p=independent['prefix_continuation']
            lines+=['',f"Independent heldout: teacher forced {a['correct']}/{a['tokens']}; prefix exact {p['exact']}/{p['samples']}. Memorization is not world knowledge or representative generalization."]
    if candidates:
        lines+=['','Chinese candidates are **'+candidates['label']+'**. Only human ACCEPT items enter formal counts.',
                '', '| Protocol | Category | Correct/scored | Unscorable | Wilson 95% |', '|---|---|---:|---:|---|']
        for policy,data in candidates['exploratory'].items():
            for category,r in data['categories'].items():
                lines.append(f"| {policy} | {category} | {r['correct']}/{r['scored']} | {r['unscorable']} | {r['wilson_95']} |")
        formal=candidates['formal']['separate'];lines+=['',f"Formal accepted denominator: {formal['scored']}; formal correct: {formal['correct']}."]
    if generation:
        lines+=['','| Decoder | Generated | Empty | Early EOS | Mean length | Mean 4-gram repeat |','|---|---:|---:|---:|---:|---:|']
        for mode,r in generation['modes'].items():
            s=r['summary'];lines.append(f"| {mode} | {s['generated']} | {s['empty_output']['numerator']} | {s['early_eos']['numerator']} | {s['generated_length']['mean']} | {s['repeat_4gram']['mean']} |")
    lines+=['','Known limits: heldout templates are narrow; candidate wording, length and token frequency biases remain. Wilson intervals do not capture benchmark design bias. Raw continuation needs human readability review. Decoder statistics are separate trend indicators. Different protocol identities prohibit direct capability deltas. CPU equality does not guarantee equality on other hardware.',
            '', 'PR #8: engineering/learning evidence may inform planning, but human review and a versioned v4 data/tokenizer plan remain required. This report does not authorize formal 87m/v4 training.']
    (out/'diagnostic-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest['conclusions']=gate
    manifest['artifacts']={str(p.relative_to(out)):{'sha256':file_sha256(p),'bytes':p.stat().st_size}
                           for p in out.rglob('*') if p.is_file() and p.name!='diagnostic-manifest.json'}
    write_json(out/'diagnostic-manifest.json',manifest)


def run_workflow(args, out, budget):
    started=time.monotonic()
    manifest=base_manifest(args.seed,budget,args.device)
    manifest['assets']={str(p.relative_to(ROOT)):file_identity(p) for d in (MEMORIZATION,FOUNDATION)
                        for p in d.iterdir() if p.is_file()}
    manifest['requested_options']=vars(args).copy()
    manifest['memory_limit']=args.memory_limit
    write_json(out/'diagnostic-manifest.json',manifest)
    memorization=training=candidates=generation=None
    if args.command in ('memorization','all'):
        memdir=out/'memorization';memdir.mkdir()
        memorization=run_memorization(memdir,budget=budget,seed=args.seed,kind=args.model,device=args.device,
            enable_87m=args.enable_87m,tokenizer_path=args.tokenizer,gpu_memory_budget_mib=args.gpu_memory_budget_mib)
        write_json(out/'memorization-results.json',memorization)
        manifest['memorization']={'model_kind':args.model,'tokenizer_fingerprint':memorization['tokenizer_fingerprint'],
                                  'initial_model':memorization['initial_model'],'data_sha256':memorization['data_sha256']}
    if args.command=='all':
        tokenizer=MiniLLMTokenizer.load(out/'memorization/tokenizer.json')
        checks=tokenizer_checks(tokenizer,out/'tokenizer-checks')
        gradients=loss_gradient_checks(tokenizer,args.seed)
        remaining=budget.max_wall_seconds-(time.monotonic()-started)
        with (out/'recovery-driver.log').open('x') as recovery_log:
            subprocess.run([sys.executable,str(ROOT/'scripts/check_training_reliability.py'),
                            '--out',str(out/'recovery'),'--seed',str(args.seed)],check=True,timeout=max(.01,remaining),
                           stdout=recovery_log,stderr=subprocess.STDOUT)
        recovery=json.loads((out/'recovery/reliability-report.json').read_text())
        training={'engineering':'ENGINEERING_PASS','tokenizer':checks,'loss_gradient':gradients,
                  'recovery':{'status':'PASS',**recovery},
                  'fixed_overhead':'Separate CPU Nano correctness probes; recovery uses fixed 7-update plan, two resumes and backup. Memorization max_steps/tokens apply only to the two learning experiments; all phases share the wall watchdog.'}
        write_json(out/'training-diagnostics.json',training)
    if args.command in ('all','evaluate'):
        ckpt=args.ckpt if args.command=='evaluate' else out/'memorization/corpus/model.pt'
        tok=args.tokenizer if args.command=='evaluate' else out/'memorization/tokenizer.json'
        if args.command=='all' and not Path(ckpt).exists():
            for filename in ('candidate-results.json','generation-results.json'):
                write_json(out/filename,{'status':'NOT_RUN','reason':'No corpus checkpoint within memorization budget'})
            manifest['evaluation']={'status':'NOT_RUN_TOKEN_BUDGET'}
        else:
            candidates,generation,evaluation_identity=evaluate_checkpoint(ckpt,tok,out,seed=args.seed,device=args.device,reviews=args.reviews)
            manifest['evaluation']=evaluation_identity
    report(out,manifest,training=training,memorization=memorization,candidates=candidates,generation=generation)
    return (training is None or training['engineering']=='ENGINEERING_PASS') and (memorization is None or memorization['engineering']=='ENGINEERING_PASS' and (args.model!='nano' or memorization['learning']=='LEARNING_DEMONSTRATED'))


def main(command='all', argv=None):
    parser=argparse.ArgumentParser(description='Bounded MiniLLM diagnostics; no formal v4 training')
    parser.add_argument('--out',default=f'reports/diagnostics/{command}-'+time.strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--max-steps',type=int,default=200)
    parser.add_argument('--max-effective-tokens',type=int,default=50000)
    parser.add_argument('--max-wall-seconds',type=float,default=120)
    parser.add_argument('--max-memory-mib',type=int,default=2048,help='CPU virtual address-space headroom beyond imported PyTorch; hard RLIMIT_AS, not RSS')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--model',choices=['nano','87m'],default='nano')
    parser.add_argument('--enable-87m',action='store_true')
    parser.add_argument('--gpu-memory-budget-mib',type=float)
    parser.add_argument('--tokenizer');parser.add_argument('--ckpt');parser.add_argument('--reviews')
    args=parser.parse_args(argv);args.command=command
    out=None
    try:
        budget=Budget(args.max_steps,args.max_effective_tokens,args.max_wall_seconds,args.max_memory_mib);budget.validate()
        if command=='all' and (args.model!='nano' or args.device!='cpu'):
            raise ValueError('full correctness workflow is CPU Nano; use run_memorization_probe.py for explicit 87m probes')
        if command=='evaluate' and (not args.ckpt or not args.tokenizer):
            raise ValueError('evaluation requires --ckpt and --tokenizer')
        if args.model=='87m' and (not args.enable_87m or not args.tokenizer):
            raise ValueError('87m requires --enable-87m and existing --tokenizer')
        torch.set_num_threads(1)
        out=new_output(args.out)
        memory=cpu_memory_limit(budget.max_memory_mib) if args.device=='cpu' else contextlib.nullcontext({'kind':'CUDA allocator admission'})
        with memory as memory_limit, wall_deadline(budget.max_wall_seconds):
            args.memory_limit=memory_limit
            passed=run_workflow(args,out,budget)
        print(f"{'PASS' if passed else 'FAIL'} diagnostics: {out}")
        return 0 if passed else 1
    except Exception as exc:
        if out is not None:write_failure(out,exc)
        print(f'FAIL diagnostics: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 1
