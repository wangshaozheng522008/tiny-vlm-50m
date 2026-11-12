import json
from pathlib import Path

import pytest

from minillm.corpus_semantics import audit_semantics, curate_review, semantic_flags
from minillm.assessment_io import records


def write_rows(path,rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));return path


def test_flags_are_explicit_review_not_truth():
    assert semantic_flags('12 + 8 = 21')[0]['reason']=='literal_arithmetic_mismatch'
    assert semantic_flags('１２＋８＝２１')[0]['reason']=='literal_arithmetic_mismatch'
    assert semantic_flags('18 ÷ 3 = 6')==[]
    assert semantic_flags('0.1 + 0.2 = 0.3')==[]
    assert semantic_flags('1*2 + 2*3 + 3*4 + 4*5 = 40。')==[]
    assert semantic_flags('5x - 3 + 3 = 17 + 3。')==[]
    assert semantic_flags('10 ÷ 3 = 3 余1。')==[]
    assert semantic_flags('12 * 300 = 3,600')==[]
    assert semantic_flags('31,680/60 = 528 英尺/分钟。')==[]
    assert semantic_flags('200 / 30 = 6.6667 天。')==[]
    assert semantic_flags('10 ÷ 3 = 3 又 1/3。')==[]
    assert semantic_flags('2 / 0 = 0')
    assert semantic_flags('示例反例：“2+2=5”') # requires context review, never auto-drop
    assert semantic_flags('print("2+2=5")',category='code')==[]
    assert semantic_flags('任务：计算。输入：1+1。输出：2。')[0]['reason']=='flattened_instruction_record'


def test_review_queue_order_determinism_and_fail_closed_curation(tmp_path):
    rows=[{'text':f'这是第{i}条学习记录，1+1={2 if i%2 else 3}。','source':'synthetic'} for i in range(30)]
    source=write_rows(tmp_path/'a.jsonl',rows)
    a=audit_semantics([source],tmp_path/'review-a',sample=10)
    source_b=write_rows(tmp_path/'b.jsonl',list(reversed(rows)))
    b=audit_semantics([source_b],tmp_path/'review-b',sample=10)
    assert a['status']=='REVIEW_PENDING' and a['coverage']['semantic_error_rate'] is None
    assert a['flagged_documents']==b['flagged_documents']==15
    assert (tmp_path/'review-a/review.jsonl').read_bytes()==(tmp_path/'review-b/review.jsonl').read_bytes()
    queue=list(records(tmp_path/'review-a/review.jsonl'))
    decisions=write_rows(tmp_path/'decisions.jsonl',[{'id':r['id'],'action':'exclude' if r['flags'] else 'accept',
        'reason':'synthetic fixture known answer','reviewer':'test fixture'} for r in queue])
    result=curate_review(tmp_path/'review-a',decisions,tmp_path/'curated')
    assert result['status']=='REVIEWED_SUBSET' and not result['counts'].get('unreviewed_included')
    assert all(not semantic_flags(r['text']) for r in records(tmp_path/'curated/accepted.jsonl'))
    with pytest.raises(ValueError,match='non-empty'):
        curate_review(tmp_path/'review-a',decisions,tmp_path/'curated')
    decisions.write_text('{}\n')
    with pytest.raises(ValueError,match='invalid'):
        curate_review(tmp_path/'review-a',decisions,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_tampered_or_incomplete_review_refused(tmp_path):
    source=write_rows(tmp_path/'a.jsonl',[{'text':'算式 1+1=3，这是错误的例子。'}])
    audit_semantics([source],tmp_path/'review',sample=2)
    decisions=write_rows(tmp_path/'decisions.jsonl',[])
    with pytest.raises(ValueError,match='requires a decision'):
        curate_review(tmp_path/'review',decisions,tmp_path/'bad')
    (tmp_path/'review/review.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='artifact changed'):
        curate_review(tmp_path/'review',decisions,tmp_path/'bad')


def test_source_statistics_have_a_hard_bound(tmp_path):
    source=write_rows(tmp_path/'sources.jsonl',[{'text':f'文档{i}','source':f'source-{i}'} for i in range(129)])
    with pytest.raises(ValueError,match='128 source labels'):
        audit_semantics([source],tmp_path/'report',sample=2)
    assert not (tmp_path/'report').exists()
