"""Crash recovery and provenance invariants, exclusively temporary CPU Nano runs."""
import copy
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from minillm.assessment_io import read_json
from minillm.checkpoint import load_training_state
from minillm.corpus_utils import file_info, identity
from minillm.pilot_launcher import launch_pilot
from minillm.pilot_lineage import record_resume, run_lock, sealed
from minillm.training_manifest import write_json
from minillm.workload import resolve_plan
from test_evaluation import eval_assets
from test_workload import config


def launch(assets,cfg,out):
    return launch_pilot(assets['dataset'],assets['tokenizer'],str(assets['cfg']),cfg,out,execute=True)


def resume_config(cfg):
    return {**cfg,'init_from':None,'resume':str(Path(cfg['out'])/'state.pt')}


def start(assets,tmp_path):
    cfg=config(assets,tmp_path)
    result=launch(assets,cfg,tmp_path/'launch')
    assert result['executed']
    return cfg,read_json(Path(cfg['out'])/'pilot-lineage.json')


def test_identity_durable_before_first_update(eval_assets,tmp_path):
    import minillm.train as training
    original=training._train_step
    cfg=config(eval_assets,tmp_path);observed=[]
    def step(*args,**kwargs):
        lineage=read_json(Path(cfg['out'])/'pilot-lineage.json')
        binding=read_json(tmp_path/'launch/run-identity.json')
        assert lineage==binding
        assert lineage['training_manifest']==read_json(Path(cfg['out'])/'training-manifest.json')
        assert lineage['initial_launch']['request_id']==read_json(tmp_path/'launch/launch-manifest.json')['request_id']
        observed.append(True)
        return original(*args,**kwargs)
    with patch.object(training,'_train_step',side_effect=step):launch(eval_assets,cfg,tmp_path/'launch')
    assert len(observed)==4


@pytest.mark.skipif(os.name!='posix',reason='SIGKILL and flock require POSIX')
def test_sigkill_after_valid_checkpoint_resumes(eval_assets,tmp_path):
    cfg={**config(eval_assets,tmp_path),'val_every':0}
    config_path=tmp_path/'training.json';write_json(config_path,cfg)
    child="""
import os,signal,sys
from unittest.mock import patch
import minillm.train as training
from minillm.pilot_launcher import main
original=training._append_log
def append(out,record):
 original(out,record)
 if record.get('event')=='checkpoint':os.kill(os.getpid(),signal.SIGKILL)
with patch.object(training,'_append_log',side_effect=append):main(sys.argv[1:])
"""
    args=[sys.executable,'-c',child,'--dataset',str(eval_assets['dataset']),'--tokenizer',str(eval_assets['tokenizer']),
          '--config',str(eval_assets['cfg']),'--training-config',str(config_path),'--out',str(tmp_path/'killed-launch'),'--execute']
    env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src'),'CUDA_VISIBLE_DEVICES':''}
    p=subprocess.run(args,env=env,capture_output=True,text=True,timeout=45)
    assert p.returncode==-signal.SIGKILL,p.stdout+p.stderr
    state=load_training_state(Path(cfg['out'])/'state.pt');assert state['global_step']==1
    lineage=read_json(Path(cfg['out'])/'pilot-lineage.json');assert lineage['training_manifest']==state['manifest']
    result=launch(eval_assets,resume_config(cfg),tmp_path/'resumed')
    assert result['summary']['global_step']==4
    assert read_json(Path(cfg['out'])/'pilot-lineage.json')['resume_history'][0]['checkpoint_step']==1


def test_missing_lineage_recovers_from_durable_run_binding(eval_assets,tmp_path):
    cfg,first=start(eval_assets,tmp_path);path=Path(cfg['out'])/'pilot-lineage.json';path.unlink()
    for name in ('model.pt','best.pt'):(Path(cfg['out'])/name).unlink()
    opts=resume_config(cfg)
    with pytest.raises(ValueError,match='cannot be guessed'):
        resolve_plan(eval_assets['dataset'],eval_assets['tokenizer'],str(eval_assets['cfg']),opts)
    opts['lineage_report']=str(tmp_path/'launch')
    # Inspection is read-only, including when migration/recovery is possible.
    resolve_plan(eval_assets['dataset'],eval_assets['tokenizer'],str(eval_assets['cfg']),opts)
    assert not path.exists()
    launch(eval_assets,opts,tmp_path/'recovered')
    recovered=read_json(path)
    assert recovered['initial_launch']==first['initial_launch']
    assert recovered['recovery']['reason']=='missing_lineage'
    assert len(recovered['resume_history'])==1


def legacy(assets,tmp_path,*,keep_lineage):
    cfg,current=start(assets,tmp_path)
    record=read_json(tmp_path/'launch/launch-manifest.json')
    for k in ('request_id','report_path','training_manifest','tokenizer_fingerprint','source_path'):record.pop(k,None)
    record['schema_version']=1;record['request_id']=identity(record)
    report=tmp_path/'legacy-launch';report.mkdir();write_json(report/'launch-manifest.json',record)
    for name in ('model.pt','best.pt'):
        path=Path(cfg['out'])/name
        if path.exists():
            model=torch.load(path,map_location='cpu',weights_only=False)
            model['pilot_request_id']=record['request_id'];torch.save(model,path)
    path=Path(cfg['out'])/'pilot-lineage.json'
    if keep_lineage:
        write_json(path,{'initialization':current['initialization'],'training_manifest':current['training_manifest'],'launch_report':str(report)})
    else:path.unlink()
    return cfg,record,report


def test_v1_lineage_read_only_then_atomic_migration(eval_assets,tmp_path):
    cfg,record,report=legacy(eval_assets,tmp_path,keep_lineage=True)
    path=Path(cfg['out'])/'pilot-lineage.json';before=path.read_bytes()
    resolve_plan(eval_assets['dataset'],eval_assets['tokenizer'],str(eval_assets['cfg']),resume_config(cfg))
    assert path.read_bytes()==before
    launch(eval_assets,resume_config(cfg),tmp_path/'migration')
    value=read_json(path)
    assert value['schema_version']==2 and value['migration']['from_schema_version']==1
    assert value['initial_launch']=={'request_id':record['request_id'],'report_path':str(report.resolve())}
    assert len(value['resume_history'])==1


def test_missing_v1_lineage_requires_same_step_model_evidence(eval_assets,tmp_path):
    cfg,_,report=legacy(eval_assets,tmp_path,keep_lineage=False)
    opts={**resume_config(cfg),'lineage_report':str(report)}
    launch(eval_assets,opts,tmp_path/'recovered')
    assert read_json(Path(cfg['out'])/'pilot-lineage.json')['recovery']['reason']=='missing_lineage'


def test_legacy_missing_source_evidence_fails_closed(eval_assets,tmp_path):
    cfg,_,report=legacy(eval_assets,tmp_path,keep_lineage=False)
    for name in ('model.pt','best.pt'):(Path(cfg['out'])/name).unlink()
    with pytest.raises(ValueError,match='no matching checkpoint model'):
        launch(eval_assets,{**resume_config(cfg),'lineage_report':str(report)},tmp_path/'rejected')
    assert not (Path(cfg['out'])/'pilot-lineage.json').exists()


@pytest.mark.parametrize('field',['initialization','run_id','tokenizer','history','checksum'])
def test_lineage_corruption_or_inconsistency_rejected(eval_assets,tmp_path,field):
    cfg,lineage=start(eval_assets,tmp_path);path=Path(cfg['out'])/'pilot-lineage.json'
    if field=='initialization':lineage['initialization']['sha256']='0'*64
    if field=='run_id':lineage['training_manifest']['run_id']='foreign'
    if field=='tokenizer':lineage['tokenizer_fingerprint']='foreign'
    if field=='history':lineage['resume_history']=[{'request_id':'unknown'}]
    if field!='checksum':lineage=sealed(lineage)
    else:lineage['content_id']='0'*64
    write_json(path,lineage)
    with pytest.raises((ValueError,KeyError)):
        launch(eval_assets,resume_config(cfg),tmp_path/'rejected')


def test_checkpoint_corruption_refused(eval_assets,tmp_path):
    cfg,_=start(eval_assets,tmp_path);(Path(cfg['out'])/'state.pt').write_bytes(b'broken')
    with pytest.raises(RuntimeError,match='cannot load training checkpoint'):
        launch(eval_assets,resume_config(cfg),tmp_path/'rejected')


def test_external_state_or_relabelled_directory_cannot_claim_identity(eval_assets,tmp_path):
    cfg,_=start(eval_assets,tmp_path)
    other={**cfg,'out':str(tmp_path/'other')};launch(eval_assets,other,tmp_path/'other-launch')
    with pytest.raises(ValueError,match='ownership mismatch'):
        launch(eval_assets,{**resume_config(cfg),'resume':str(tmp_path/'other/state.pt')},tmp_path/'foreign')
    # Even copying a complete foreign checkpoint and its manifest does not match this lineage.
    import shutil
    for name in ('state.pt','training-manifest.json'):shutil.copyfile(tmp_path/'other'/name,Path(cfg['out'])/name)
    with pytest.raises(ValueError,match='lineage manifest'):
        launch(eval_assets,resume_config(cfg),tmp_path/'relabeled')


def test_two_resumes_keep_initial_and_idempotent_history(eval_assets,tmp_path):
    cfg,first=start(eval_assets,tmp_path)
    for name in ('resume1','resume2'):launch(eval_assets,resume_config(cfg),tmp_path/name)
    path=Path(cfg['out'])/'pilot-lineage.json';value=read_json(path);before=path.read_bytes()
    assert value['initial_launch']==first['initial_launch']
    assert len(value['resume_history'])==2
    assert len({e['request_id'] for e in value['resume_history']})==2
    entry=value['resume_history'][0]
    record_resume(cfg['out'],value,entry['report_path'],entry['request_id'],entry['checkpoint_step'])
    assert path.read_bytes()==before
    with pytest.raises(ValueError,match='reused'):
        record_resume(cfg['out'],value,tmp_path/'different',entry['request_id'],entry['checkpoint_step'])


def test_duplicate_fresh_start_cannot_overwrite(eval_assets,tmp_path):
    cfg,_=start(eval_assets,tmp_path);before=file_info(Path(cfg['out'])/'state.pt')
    result=launch(eval_assets,cfg,tmp_path/'duplicate')
    assert result['status']=='BLOCKED' and not result.get('executed')
    assert file_info(Path(cfg['out'])/'state.pt')==before


def test_initial_lineage_write_failure_never_creates_checkpoint(eval_assets,tmp_path):
    import minillm.checkpoint as checkpoint
    original=checkpoint.os.replace;cfg=config(eval_assets,tmp_path)
    def replace(source,target):
        if Path(target).name=='pilot-lineage.json':raise OSError('injected atomic publish failure')
        return original(source,target)
    with patch.object(checkpoint.os,'replace',side_effect=replace),pytest.raises(OSError,match='injected'):
        launch(eval_assets,cfg,tmp_path/'launch')
    out=Path(cfg['out'])
    assert not (out/'state.pt').exists() and not (out/'training-manifest.json').exists()
    assert not (out/'pilot-lineage.json').exists() and not list(out.glob('.*.tmp'))


def test_interrupted_resume_lineage_write_preserves_previous_record(eval_assets,tmp_path):
    import minillm.checkpoint as checkpoint
    cfg,_=start(eval_assets,tmp_path);path=Path(cfg['out'])/'pilot-lineage.json';before=path.read_bytes()
    state_before=file_info(Path(cfg['out'])/'state.pt');original=checkpoint.os.replace
    def replace(source,target):
        if Path(target)==path:raise OSError('interrupted lineage replace')
        return original(source,target)
    with patch.object(checkpoint.os,'replace',side_effect=replace),pytest.raises(OSError,match='interrupted'):
        launch(eval_assets,resume_config(cfg),tmp_path/'resume')
    assert path.read_bytes()==before and file_info(Path(cfg['out'])/'state.pt')==state_before
    assert not list(path.parent.glob('.*.tmp'))


def test_active_launcher_lock_rejects_second_execution(eval_assets,tmp_path):
    cfg=config(eval_assets,tmp_path)
    with run_lock(cfg['out']),pytest.raises(ValueError,match='another pilot launcher'):
        launch(eval_assets,cfg,tmp_path/'locked')


def test_legacy_invalid_initialization_not_migrated(eval_assets,tmp_path):
    cfg,_,_=legacy(eval_assets,tmp_path,keep_lineage=True);path=Path(cfg['out'])/'pilot-lineage.json'
    value=read_json(path);value['initialization']['sha256']='0'*64;write_json(path,value);before=path.read_bytes()
    with pytest.raises(ValueError,match='source/output mismatch'):
        launch(eval_assets,resume_config(cfg),tmp_path/'rejected')
    assert path.read_bytes()==before


def test_fresh_initialization_identity(eval_assets,tmp_path):
    cfg={**config(eval_assets,tmp_path),'init_from':None}
    launch(eval_assets,cfg,tmp_path/'fresh')
    lineage=read_json(Path(cfg['out'])/'pilot-lineage.json')
    assert lineage['initialization']=={'kind':'fresh','seed':cfg['seed']}
    launch(eval_assets,resume_config(cfg),tmp_path/'resumed')
    assert read_json(Path(cfg['out'])/'pilot-lineage.json')['initial_launch']==lineage['initial_launch']


def test_initialization_report_cannot_override_existing_lineage(eval_assets,tmp_path):
    cfg,_=start(eval_assets,tmp_path)
    with pytest.raises(ValueError,match='disagrees with lineage'):
        launch(eval_assets,{**resume_config(cfg),'lineage_report':str(tmp_path/'different')},tmp_path/'rejected')


def test_duplicate_history_is_rejected_even_with_updated_checksum(eval_assets,tmp_path):
    cfg,_=start(eval_assets,tmp_path);launch(eval_assets,resume_config(cfg),tmp_path/'resume')
    path=Path(cfg['out'])/'pilot-lineage.json';value=read_json(path)
    value['resume_history'].append(copy.deepcopy(value['resume_history'][0]));write_json(path,sealed(value))
    with pytest.raises(ValueError,match='duplicate resume history'):
        launch(eval_assets,resume_config(cfg),tmp_path/'rejected')


def test_two_interrupted_resumes_record_actual_checkpoint_steps(eval_assets,tmp_path):
    import minillm.train as training
    original=training._train_step;cfg={**config(eval_assets,tmp_path),'val_every':0}
    def phase(options,stop,name):
        count=load_training_state(options['resume'])['global_step'] if options.get('resume') else 0
        def step(*args,**kwargs):
            nonlocal count
            result=original(*args,**kwargs);count+=1
            if count==stop:signal.raise_signal(signal.SIGTERM)
            return result
        with patch.object(training,'_train_step',side_effect=step):return launch(eval_assets,options,tmp_path/name)
    phase(cfg,2,'initial')
    first=read_json(Path(cfg['out'])/'pilot-lineage.json')['initial_launch']
    phase(resume_config(cfg),3,'resume1');last=phase(resume_config(cfg),None,'resume2')
    lineage=read_json(Path(cfg['out'])/'pilot-lineage.json')
    assert last['summary']['global_step']==4 and lineage['initial_launch']==first
    assert [e['checkpoint_step'] for e in lineage['resume_history']]==[2,3]


def test_v1_overwritten_reference_requires_original_and_retains_legacy_resume(eval_assets,tmp_path):
    cfg,original_record,original_report=legacy(eval_assets,tmp_path,keep_lineage=True)
    launch(eval_assets,resume_config(cfg),tmp_path/'old-resume')
    record=read_json(tmp_path/'old-resume/launch-manifest.json')
    for key in ('request_id','report_path','training_manifest','tokenizer_fingerprint','source_path','checkpoint'):record.pop(key,None)
    record['schema_version']=1;record['request_id']=identity(record)
    write_json(tmp_path/'old-resume/launch-manifest.json',record)
    path=Path(cfg['out'])/'pilot-lineage.json';current=read_json(path)
    write_json(path,{'initialization':current['initialization'],'training_manifest':current['training_manifest'],
                    'launch_report':str(tmp_path/'old-resume')})
    with pytest.raises(ValueError,match='lost its first launch'):
        launch(eval_assets,resume_config(cfg),tmp_path/'insufficient')
    launch(eval_assets,{**resume_config(cfg),'lineage_report':str(original_report)},tmp_path/'migrated')
    migrated=read_json(path)
    assert migrated['initial_launch']['request_id']==original_record['request_id']
    assert migrated['migration']['legacy_request_id']==record['request_id']
    assert migrated['migration']['legacy_launch_report']==str(tmp_path/'old-resume')
    assert 'unavailable' in migrated['migration']['history_scope']
    assert len(migrated['resume_history'])==1


def test_identity_callback_cannot_mutate_training_plan(eval_assets,tmp_path):
    from minillm.data import PackedPretrainDataset
    from minillm.tokenizer import MiniLLMTokenizer
    from minillm.train import fit
    from minillm.workload import parse_training_config
    cfg=config(eval_assets,tmp_path);opts,_=parse_training_config(cfg)
    def callback(manifest):manifest['training']['target_optimizer_updates']=1
    summary=fit(opts,eval_assets['model'],MiniLLMTokenizer.load(eval_assets['tokenizer']),
                PackedPretrainDataset(eval_assets['dataset']/'train.bin',16),
                PackedPretrainDataset(eval_assets['dataset']/'val.bin',16),on_run_prepared=callback)
    assert summary['global_step']==4
    assert read_json(Path(cfg['out'])/'training-manifest.json')['training']['target_optimizer_updates']==4
