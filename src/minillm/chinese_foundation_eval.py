"""Frozen Chinese foundation probes with independent human-review decisions."""
from __future__ import annotations

import math
from pathlib import Path

from .assessment_io import records
from .diagnostic_common import FOUNDATION, frozen_rows
from .evaluation_candidates import BOUNDARY_POLICIES, candidate_prediction, load_candidates
from .training_manifest import file_sha256


def load_foundation(directory=FOUNDATION):
    directory = Path(directory)
    candidates = load_candidates(directory/'candidates.jsonl')
    prompts = frozen_rows(directory/'prompts.jsonl')
    frozen_rows(directory/'review-template.jsonl')
    seen, texts = set(), set()
    for row in candidates+prompts:
        if (row.get('review_status') != 'REVIEW_PENDING' or row.get('question_version') != 1
                or not isinstance(row.get('source'),str) or not row['source'].strip()
                or row['id'] in seen or row['prompt'] in texts
                or row.get('text_category') not in ('zh','en','mixed','code')
                or 'candidates' in row and len(row['candidates'])!=4
                or any(len(v)>128 for v in row.get('candidates',{}).values())
                or len(row['prompt'])>256
                or 'candidates' not in row and row.get('generation_mode')!='raw'):
            raise ValueError('invalid/duplicate foundation task or embedded review acceptance')
        seen.add(row['id']); texts.add(row['prompt'])
    if len(candidates)>150 or len(prompts)>50:
        raise ValueError('foundation task budget exceeded')
    return candidates, prompts


def review_decisions(path, candidates, prompts, directory=FOUNDATION):
    """An ACCEPT is bound to the frozen file/version; REVISE requires a new asset version."""
    decisions = {}
    known = {r['id']:r for r in candidates+prompts}
    if path is None:
        return decisions
    for row in records(path):
        key = row.get('id')
        if key not in known or key in decisions or row.get('action') not in ('REVIEW_PENDING','ACCEPT','REJECT','REVISE'):
            raise ValueError('unknown/duplicate review ID or invalid review action')
        task = known[key]
        asset = Path(directory)/('candidates.jsonl' if 'candidates' in task else 'prompts.jsonl')
        if row.get('question_version') != task['question_version'] or row.get('asset_sha256') != file_sha256(asset):
            raise ValueError('review does not match frozen asset/version')
        if row['action'] != 'REVIEW_PENDING' and (row.get('human_review') is not True
                or any(not isinstance(row.get(k),str) or not row[k].strip() for k in ('reviewer','reason'))):
            raise ValueError('human review requires reviewer, reason and explicit human_review=true attestation')
        decisions[key] = row
    return decisions


def accuracy_interval(correct, total):
    if not 0 <= correct <= total:
        raise ValueError('invalid accuracy counts')
    if not total:
        return {'correct':correct,'scored':0,'accuracy':None,'wilson_95':None}
    z = 1.959963984540054
    p = correct/total; denom = 1+z*z/total
    center = (p+z*z/(2*total))/denom
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return {'correct':correct,'scored':total,'accuracy':p,'wilson_95':[max(0,center-half), min(1,center+half)]}


def aggregate_foundation(rows, *, reviewed_only):
    selected = [r for r in rows if not reviewed_only or r['review_action']=='ACCEPT']
    result = {}
    for policy in BOUNDARY_POLICIES:
        def group(items):
            scored = [r['prediction']['protocols'][policy] for r in items
                      if r['prediction']['protocols'][policy]['status']=='SCORED']
            return {**accuracy_interval(sum(bool(r['score']['correct']) for r in scored),len(scored)),
                    'tasks':len(items),'unscorable':len(items)-len(scored),
                    'ties':sum(r['score']['status']=='tie' for r in scored)}
        result[policy] = {**group(selected), 'categories':{
            category:group([r for r in selected if r['category']==category])
            for category in sorted({r['category'] for r in rows})}}
    return result


def evaluate_foundation(model, tokenizer, *, directory=FOUNDATION, reviews=None):
    candidates, prompts = load_foundation(directory)
    decisions = review_decisions(reviews,candidates,prompts,directory)
    rows = []
    for task in candidates:
        try:
            prediction = candidate_prediction(model,tokenizer,task,normalization='sum',boundary_policy='separate')
        except ValueError as exc:
            # Only declared item-level input failures may be excluded. Model/numeric errors propagate.
            if not any(s in str(exc) for s in ('context/empty/special token violation','empty candidate prefix','collapse to identical token IDs')):
                raise
            prediction = {'id':task['id'],'prompt_token_ids':tokenizer.encode(task['prompt'],category=task['text_category']),
                'candidate_token_ids':{key:tokenizer.encode(text,category=task['text_category']) for key,text in task['candidates'].items()},
                'protocols':{p:{'status':'NOT_SCORABLE_INPUT','reason':str(exc),'candidates':{},'selected':None,
                              'score':{'correct':None,'status':'NOT_SCORABLE_INPUT'}} for p in BOUNDARY_POLICIES}}
        rows.append({'id':task['id'],'category':task['category'],
                     'review_action':decisions.get(task['id'],{}).get('action','REVIEW_PENDING'),
                     'prediction':prediction})
    actions = [r['review_action'] for r in rows]
    finalized = all(a in ('ACCEPT','REJECT') for a in actions) and 'ACCEPT' in actions
    lengths=[{'id':r['id'],'reference_characters':len(r['candidates'][r['reference']]),
              'choice_characters':{k:len(v) for k,v in r['candidates'].items()},
              'reference_position':r['reference']} for r in candidates]
    return {'schema_version':1, 'version':'chinese-foundation-v1',
            'status':'EVALUATION_REVIEWED' if finalized else 'EVALUATION_PROVISIONAL',
            'label':'REVIEWED' if finalized else 'PROVISIONAL',
            'primary_boundary_policy':'separate','normalization':'sum','rows':rows,
            'formal':aggregate_foundation(rows,reviewed_only=True),
            'exploratory':aggregate_foundation([r for r in rows if r['review_action'] not in ('REJECT','REVISE')],reviewed_only=False),
            'review_counts':{a:actions.count(a) for a in ('REVIEW_PENDING','ACCEPT','REJECT','REVISE')},
            'review_file_sha256':file_sha256(Path(reviews)) if reviews else None,
            'answer_bias_audit':{'reference_positions':{k:sum(r['reference']==k for r in candidates) for k in 'ABCD'},
                                 'character_lengths':lengths,'note':'Token lengths/NLL per choice are retained in predictions. Balanced labels and similar character lengths do not eliminate frequency or wording bias.'},
            'interpretation':'Only ACCEPT contributes to formal counts. Pending items are exploratory. Wilson intervals describe item uncertainty, not benchmark representativeness. Separate and joint_shared_prefix must never be mixed or compared as a capability delta.'}


def require_comparable(left, right):
    """Protocol identity must match before any downstream capability comparison."""
    keys = ('benchmark_sha256','tokenizer_fingerprint','text_policy','boundary_policy','normalization','decoding','review_sha256','model_config','implementation_sha256')
    if any(k not in left or k not in right or left[k]!=right[k] for k in keys):
        raise ValueError('incompatible diagnostic evaluation protocols; capability deltas forbidden')
