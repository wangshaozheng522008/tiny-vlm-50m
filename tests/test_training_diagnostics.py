"""Independent numerical checks of the production loss, tokenizer and resume paths."""
import json
from pathlib import Path

import pytest
import torch

from minillm.diagnostic_common import new_output
from minillm.text_policy import checkpoint_text_policy
from minillm.training_diagnostics import loss_gradient_checks, tokenizer_checks
from minillm.diagnostic_workflow import main


def test_policy_roundtrip_and_production_bin(tiny_tokenizer,tmp_path):
    report=tokenizer_checks(tiny_tokenizer,tmp_path/'checks')
    assert report['bin_equals_documents'] and report['tokenizer_mismatch_rejected']
    width=[r for r in report['cases'] if r['policy']=='width-fold-prose-v1']
    assert any(r['expected']!=r['input'] for r in width)
    assert all(r['expected']==r['input'] for r in width if r['category']=='code')
    assert tiny_tokenizer.text_policy=='identity-v1'


def test_loss_gradient_causality_and_finite_difference(tiny_tokenizer):
    report=loss_gradient_checks(tiny_tokenizer,seed=7)
    assert report['parameters_updated'] and report['optimizer_finite']
    assert report['effective_target_tokens']>0
    assert report['future_token_max_difference']<1e-6
    assert len(report['finite_difference']['checks'])==3
    assert all(abs(r['autograd'])>0 for r in report['finite_difference']['checks'])


def test_checkpoint_policy_conflict_rejected():
    with pytest.raises(ValueError,match='disagrees'):
        checkpoint_text_policy({'text_policy':'identity-v1','manifest':{'dataset':{'train':{'text_policy':'width-fold-prose-v1'}}}})


def test_historical_output_never_overwritten(tmp_path):
    out=tmp_path/'history';out.mkdir();(out/'evidence.json').write_text('immutable')
    with pytest.raises(FileExistsError):new_output(out)
    assert main('all',['--out',str(out)])==1
    assert list(out.iterdir())==[out/'evidence.json']
    assert (out/'evidence.json').read_text()=='immutable'


def test_wall_budget_emits_failure(tmp_path,monkeypatch):
    import time
    import minillm.diagnostic_workflow as workflow
    monkeypatch.setattr(workflow,'run_workflow',lambda *args:time.sleep(3))
    out=tmp_path/'timed'
    assert main('all',['--out',str(out),'--max-wall-seconds','1'])==1
    assert json.loads((out/'failure.json').read_text())['exception']=='TimeoutError'


def test_complete_nano_workflow_manifest_and_acceptance(tmp_path):
    from minillm.training_manifest import file_sha256
    out=tmp_path/'workflow'
    assert main('all',['--out',str(out),'--max-wall-seconds','60'])==0
    manifest=json.loads((out/'diagnostic-manifest.json').read_text())
    assert {'seed','budget','code','environment','assets','evaluation','conclusions','memory_limit','artifacts'}<=manifest.keys()
    assert manifest['memory_limit']['kind']=='RLIMIT_AS'
    assert manifest['conclusions']==json.loads((out/'pr8-readiness.json').read_text())
    assert manifest['conclusions']['learning']=='LEARNING_DEMONSTRATED'
    assert manifest['conclusions']['evaluation']=='EVALUATION_PROVISIONAL'
    gate=manifest['conclusions']
    assert gate['ready_for_pr8'] is False  # Known pending candidate review, not an unassessed failure.
    assert gate['status']=='NOT_READY' and gate['scope']=='PR8_INFORMATIONAL_ONLY'
    assert gate['failed_criteria']==['candidate_review_complete']
    assert gate['criteria']['raw_readability_review_complete']=='NOT_ASSESSED'
    assert gate['criteria']['v4_data_tokenizer_plan_reviewed']=='NOT_ASSESSED'
    for path,expected in manifest['artifacts'].items():
        assert file_sha256(out/path)==expected['sha256']
    assert {'training-diagnostics.json','memorization-results.json','candidate-results.json','generation-results.json','diagnostic-report.md'}<=manifest['artifacts'].keys()
    training=json.loads((out/'training-diagnostics.json').read_text())
    assert training['recovery']['optimizer_state_equal'] and training['recovery']['rng_state_equal']
    assert training['recovery']['max_parameter_difference']==0
    candidates=json.loads((out/'candidate-results.json').read_text())
    assert candidates['formal']['separate']['scored']==0
    assert candidates['label']=='PROVISIONAL'


def test_cpu_memory_limit_is_restored():
    import resource
    from minillm.diagnostic_common import cpu_memory_limit
    previous=resource.getrlimit(resource.RLIMIT_AS)
    with cpu_memory_limit(128) as limit:
        assert resource.getrlimit(resource.RLIMIT_AS)[0]==limit['absolute_virtual_limit_bytes']
    assert resource.getrlimit(resource.RLIMIT_AS)==previous


def test_insufficient_budget_keeps_structured_report(tmp_path):
    out=tmp_path/'insufficient'
    assert main('all',['--out',str(out),'--max-effective-tokens','126','--max-steps','2','--max-wall-seconds','60'])==1
    result=json.loads((out/'memorization-results.json').read_text())
    assert result['effective_tokens']<=126 and result['learning']=='LEARNING_INCONCLUSIVE'
    assert (out/'diagnostic-report.md').is_file()
    assert json.loads((out/'candidate-results.json').read_text())['status']=='NOT_RUN'
    assert json.loads((out/'diagnostic-manifest.json').read_text())['conclusions']['engineering']=='ENGINEERING_PASS'
    # Correctness passes independently; learning/incomplete memorization does not acquire a blanket PASS.


@pytest.mark.parametrize('criteria,status,ready,failed,unassessed',[
    ({'training_chain_correct':'PASS','nano_learning_demonstrated':'PASS','candidate_review_complete':'PASS',
      'raw_readability_review_complete':'NOT_ASSESSED','v4_data_tokenizer_plan_reviewed':'NOT_ASSESSED'},
     'NOT_ASSESSED',None,[],['raw_readability_review_complete','v4_data_tokenizer_plan_reviewed']),
    ({'engineering':'FAIL','human_review':'NOT_ASSESSED'},'NOT_READY',False,['engineering'],['human_review']),
    ({'engineering':'PASS','human_review':'PASS'},'READY',True,[],[]),
    ({'engineering':'NOT_ASSESSED','human_review':'NOT_ASSESSED'},'NOT_ASSESSED',None,[],['engineering','human_review']),
])
def test_pr8_readiness_uses_explicit_statuses(criteria,status,ready,failed,unassessed):
    from minillm.diagnostic_workflow import summarize_pr8_readiness
    result=summarize_pr8_readiness(criteria)
    assert result=={'status':status,'ready_for_pr8':ready,'failed_criteria':failed,'unassessed_criteria':unassessed}


@pytest.mark.parametrize('criteria',[{}, {'review':True}, {'review':None}, {'review':'UNKNOWN'}])
def test_pr8_readiness_rejects_missing_or_ambiguous_assessments(criteria):
    from minillm.diagnostic_workflow import summarize_pr8_readiness
    with pytest.raises(ValueError,match='explicit'):
        summarize_pr8_readiness(criteria)


def test_reviewed_candidates_leave_external_pr8_evidence_unassessed(tmp_path):
    from minillm.diagnostic_workflow import report
    manifest={'seed':42,'code':{'git_commit':'test','git_dirty':False},'environment':{}}
    candidates={'status':'EVALUATION_REVIEWED','label':'REVIEWED',
                'exploratory':{'separate':{'categories':{}}},'formal':{'separate':{'scored':1,'correct':1}}}
    report(tmp_path,manifest,candidates=candidates)
    gate=json.loads((tmp_path/'pr8-readiness.json').read_text())
    assert gate['schema_version']==2 and gate['scope']=='PR8_INFORMATIONAL_ONLY'
    assert gate['criteria']['candidate_review_complete']=='PASS'
    assert gate['criteria']['training_chain_correct']=='NOT_ASSESSED'
    assert gate['criteria']['nano_learning_demonstrated']=='NOT_ASSESSED'
    assert gate['criteria']['raw_readability_review_complete']=='NOT_ASSESSED'
    assert gate['criteria']['v4_data_tokenizer_plan_reviewed']=='NOT_ASSESSED'
    assert gate['ready_for_pr8'] is None and gate['status']=='NOT_ASSESSED'
    assert gate['failed_criteria']==[]
    assert gate==json.loads((tmp_path/'diagnostic-manifest.json').read_text())['conclusions']
    text=(tmp_path/'diagnostic-report.md').read_text()
    assert 'ready_for_pr8: `null`' in text and 'does not block PR #7 completion' in text
