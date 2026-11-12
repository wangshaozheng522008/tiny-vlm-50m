"""Raw model.generate diagnostics, separately aggregated for each decoder."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch

from .evaluation import repetition

DECODERS = {
    'greedy': {'temperature': 0.0, 'top_k': 0},
    'temperature': {'temperature': 0.8, 'top_k': 0},
    'top_k': {'temperature': 0.8, 'top_k': 20},
}


def token_metrics(ids, tokenizer, *, max_new_tokens, early_eos_tokens=4):
    if max_new_tokens <= 0 or early_eos_tokens < 1 or len(ids) > max_new_tokens:
        raise ValueError('invalid generation metric budget')
    if any(type(t) is not int or not 0 <= t < tokenizer.vocab_size for t in ids):
        raise ValueError('illegal generated token ID')
    eos = tokenizer.special_ids['eos']
    stop = ids.index(eos) if eos in ids else None
    if stop is not None and stop != len(ids)-1:
        raise ValueError('tokens after EOS in single-row generation')
    content = ids[:stop] if stop is not None else ids
    text = tokenizer.decode(content)
    pairs = list(zip(content, content[1:]))
    longest = current = 0
    previous = None
    for t in content:
        current = current + 1 if t == previous else 1
        longest = max(longest, current); previous = t
    # Observable periodicity only; semantics/readability require human review.
    period = next((p for p in range(1, min(8, len(content)//3)+1)
                   if len(content) >= p*3 and content[-p:] == content[-2*p:-p] == content[-3*p:-2*p]), None)
    return {'text': text, 'generated_tokens': len(ids), 'content_tokens': len(content),
            'empty': not text.strip(), 'eos_emitted': stop is not None,
            'early_eos': stop is not None and stop < early_eos_tokens,
            'hit_length_limit': len(ids) == max_new_tokens and stop is None,
            'special_token_anomalies': [t for t in content if t in tokenizer.special_ids.values()],
            'distinct_2': len(set(pairs))/len(pairs) if pairs else None,
            'longest_identical_run': longest, 'tail_loop_period': period,
            **repetition(text, content), 'readability_review': 'REVIEW_PENDING'}


def distribution(values):
    return {'count': len(values), 'min': min(values) if values else None,
            'max': max(values) if values else None,
            'mean': float(np.mean(values)) if values else None,
            'p50': float(np.percentile(values,50)) if values else None,
            'p90': float(np.percentile(values,90)) if values else None,
            'histogram': dict(sorted(Counter(values).items()))}


def summarize_generations(rows):
    scored = [r for r in rows if r['status'] == 'GENERATED']
    n = len(scored)
    def rate(key):
        return {'numerator': sum(bool(r[key]) for r in scored), 'denominator': n,
                'rate': sum(bool(r[key]) for r in scored)/n if n else None}
    return {'tasks':len(rows), 'generated':n, 'unscorable':len(rows)-n,
            'empty_output':rate('empty'), 'early_eos':rate('early_eos'),
            'eos_emitted':rate('eos_emitted'), 'length_limit':rate('hit_length_limit'),
            'special_token_anomaly':rate('special_token_anomalies'),
            'observable_tail_loop':rate('tail_loop_period'),
            'generated_length':distribution([r['generated_tokens'] for r in scored]),
            'repeat_4gram':distribution([r['token_4gram_repeat_ratio'] for r in scored]),
            'distinct_2':distribution([r['distinct_2'] for r in scored if r['distinct_2'] is not None]),
            'longest_identical_run':distribution([r['longest_identical_run'] for r in scored]),
            'readability_review':'REVIEW_PENDING'}


@torch.inference_mode()
def diagnose_generation(model, tokenizer, prompts, *, seed=42, max_new_tokens=24, modes=None):
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 64 or len(prompts)>150:
        raise ValueError('generation budgets: <=150 prompts, 1..64 new tokens')
    modes = DECODERS if modes is None else modes
    if not modes:
        raise ValueError('at least one decoder required')
    device = next(model.parameters()).device
    result = {}
    for name, params in modes.items():
        if set(params) != {'temperature','top_k'} or not math.isfinite(params['temperature']) or params['temperature']<0 or type(params['top_k']) is not int or params['top_k']<0:
            raise ValueError('invalid decoder parameters')
        rows = []
        for i, task in enumerate(prompts):
            ids = tokenizer.encode(task['prompt'], category=task.get('text_category','zh'))
            row = {'id':task['id'], 'prompt':task['prompt'], 'prompt_token_ids':ids,
                   'mode':name, 'seed':seed+i, 'parameters':{**params,'max_new_tokens':max_new_tokens,'use_cache':True,'stop_token_ids':[tokenizer.special_ids['eos']]}}
            if not ids or any(t in tokenizer.special_ids.values() for t in ids):
                row['status'] = 'NOT_SCORABLE_EMPTY_OR_SPECIAL_PROMPT'
            elif len(ids)+max_new_tokens > model.cfg.max_position_embeddings:
                row['status'] = 'NOT_SCORABLE_CONTEXT'
            else:
                generator = torch.Generator(device=device).manual_seed(seed+i)
                inputs = torch.tensor([ids],device=device)
                generated = model.generate(inputs,max_new_tokens=max_new_tokens, generator=generator,
                    eos_token_id=tokenizer.special_ids['eos'], **params)[0,len(ids):].tolist()
                row.update(status='GENERATED',token_ids=generated,
                           **token_metrics(generated,tokenizer,max_new_tokens=max_new_tokens))
            rows.append(row)
        result[name] = {'parameters':{**params,'max_new_tokens':max_new_tokens,'early_eos_tokens':4},
                        'rows':rows,'summary':summarize_generations(rows)}
    return {'schema_version':1,'status':'EVALUATION_PROVISIONAL','label':'PROVISIONAL','modes':result,
            'interpretation':'Raw continuation; decoder modes are separate. Periodicity is a trend flag, not semantic nonsense detection or evidence of knowledge. No automatic semantic correctness score.'}
