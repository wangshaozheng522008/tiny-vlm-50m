"""Representative-probe and launch gates; CPU Nano, no external data."""
import copy
import json
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from minillm.assessment_io import read_json
from minillm.corpus_utils import file_info, identity
from minillm.pilot_launcher import launch_pilot
from minillm.workload import parse_training_config, probe_workload, resolve_plan, verify_workload
from test_evaluation import eval_assets


def config(assets,tmp_path):
    return {"out":str(tmp_path/"run"),"init_from":str(assets["ckpt"]),"seq_len":16,
            "batch_size":2,"grad_accum":2,"max_steps":4,"device":"cpu","precision":"fp32",
            "threads":1,"seed":7,"log_every":1,"save_every":1,"val_every":1,"val_limit":1}


def probe(assets,tmp_path,cfg=None):
    cfg=cfg or config(assets,tmp_path)
    return probe_workload(assets["dataset"],assets["tokenizer"],str(assets["cfg"]),cfg,tmp_path/"probe")


def test_full_probe_rng_weights_and_no_real_output_mutation(eval_assets,tmp_path):
    cfg=config(eval_assets,tmp_path)
    before=file_info(eval_assets["ckpt"])
    python=random.getstate();numpy=np.random.get_state();torch_rng=torch.get_rng_state()
    report=probe(eval_assets,tmp_path,cfg)
    assert random.getstate()==python
    assert np.array_equal(np.random.get_state()[1],numpy[1])
    assert torch.equal(torch.get_rng_state(),torch_rng)
    assert file_info(eval_assets["ckpt"])==before and not (tmp_path/"run").exists()
    assert report["status"]=="PASS" and report["measurements"]["optimizer_steps"]==3
    assert report["measurements"]["effective_tokens"]==3*2*2*15
    assert report["measurements"]["checkpoint_roundtrip"]
    policy=resolve_plan(eval_assets["dataset"],eval_assets["tokenizer"],str(eval_assets["cfg"]),cfg)[-1]
    assert verify_workload(tmp_path/"probe",policy,float("inf"))==report


@pytest.mark.parametrize("change",[{"batch_size":1},{"grad_accum":1},{"seq_len":8},{"lr":.0002},{"seed":10},
                                  {"grad_checkpointing":True},{"max_steps":5},{"val_limit":2}])
def test_changed_plan_rejects_proof(eval_assets,tmp_path,change):
    cfg=config(eval_assets,tmp_path);probe(eval_assets,tmp_path,cfg)
    policy=resolve_plan(eval_assets["dataset"],eval_assets["tokenizer"],str(eval_assets["cfg"]),{**cfg,**change})[-1]
    with pytest.raises(ValueError,match="incompatible"):
        verify_workload(tmp_path/"probe",policy,float("inf"))


def test_damaged_and_incomplete_proof(eval_assets,tmp_path):
    report=probe(eval_assets,tmp_path);path=tmp_path/"probe/workload.json"
    report["measurements"]["optimizer_steps"]=1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match="damaged"):
        verify_workload(path,report["policy"],float("inf"))
    report["content_id"]=identity({k:v for k,v in report.items() if k!="content_id"})
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match="incomplete"):
        verify_workload(path,report["policy"],float("inf"))


def test_dry_run_and_explicit_execution_match_then_resume(eval_assets,tmp_path):
    cfg=config(eval_assets,tmp_path);probe(eval_assets,tmp_path,cfg)
    cfg["workload_report"]=str(tmp_path/"probe")
    dry=launch_pilot(eval_assets["dataset"],eval_assets["tokenizer"],str(eval_assets["cfg"]),cfg,tmp_path/"inspect")
    assert dry["status"]=="READY" and not (tmp_path/"run").exists()
    result=launch_pilot(eval_assets["dataset"],eval_assets["tokenizer"],str(eval_assets["cfg"]),cfg,tmp_path/"execute",execute=True)
    assert result["summary"]["global_step"]==4
    lineage=read_json(tmp_path/"run/pilot-lineage.json")
    assert lineage["initialization"]["sha256"]==file_info(eval_assets["ckpt"])["sha256"]
    resume={**cfg,"init_from":None,"resume":str(tmp_path/"run/state.pt")}
    continued=launch_pilot(eval_assets["dataset"],eval_assets["tokenizer"],str(eval_assets["cfg"]),resume,tmp_path/"resumed",execute=True)
    assert continued["summary"]["global_step"]==4
    with pytest.raises(ValueError,match="non-empty"):
        launch_pilot(eval_assets["dataset"],eval_assets["tokenizer"],str(eval_assets["cfg"]),cfg,tmp_path/"execute",execute=True)


def test_failed_readiness_cannot_call_fit(eval_assets,tmp_path):
    cfg={**config(eval_assets,tmp_path),"planned_effective_tokens":1}
    with patch("minillm.pilot_launcher.fit",side_effect=AssertionError("must not train")):
        result=launch_pilot(eval_assets["dataset"],eval_assets["tokenizer"],str(eval_assets["cfg"]),cfg,tmp_path/"blocked",execute=True)
    assert result["status"]=="BLOCKED" and not (tmp_path/"run").exists()


@pytest.mark.parametrize("change",[{"batch_size":0},{"grad_accum":False},{"seq_len":1},
                                  {"threads":0},{"extra_meta":{"global_step":999}}])
def test_bad_configuration_rejected(eval_assets,tmp_path,change):
    with pytest.raises(ValueError):
        parse_training_config({**config(eval_assets,tmp_path),**change})


def test_too_few_updates_rejected(eval_assets,tmp_path):
    with pytest.raises(ValueError,match="three"):
        probe_workload(eval_assets["dataset"],eval_assets["tokenizer"],str(eval_assets["cfg"]),config(eval_assets,tmp_path),tmp_path/"short",updates=2)


def test_cpu_evidence_cannot_approve_gpu_limit(eval_assets,tmp_path):
    cfg=config(eval_assets,tmp_path);probe(eval_assets,tmp_path,cfg)
    cfg.update(workload_report=str(tmp_path/"probe"),gpu_memory_limit_mib=1024)
    report=launch_pilot(eval_assets["dataset"],eval_assets["tokenizer"],str(eval_assets["cfg"]),cfg,tmp_path/"cpu-memory")
    assert report["status"]=="BLOCKED"


def test_partial_accumulation_and_epoch_rollover(eval_assets,tmp_path):
    data=tmp_path/"short-data";data.mkdir()
    for name in ("train.bin","val.bin"):
        (np.arange(11*16+1)%200+8).astype("<u2").tofile(data/name)
    cfg={**config(eval_assets,tmp_path),"grad_accum":4}
    report=probe_workload(data,eval_assets["tokenizer"],str(eval_assets["cfg"]),cfg,tmp_path/"probe")
    assert report["measurements"]["effective_tokens"]==(8+3+8)*15
    assert [r["tokens"] for r in report["measurements"]["updates"]]==[120,45,120]


@pytest.mark.parametrize("failure",["reserved","allocated","free","hardware"])
def test_gpu_evidence_budget_and_environment_validation(eval_assets,tmp_path,failure):
    report=probe(eval_assets,tmp_path)
    policy=report["policy"]
    policy["manifest"]["execution"]["device"]="cuda"
    policy["environment"]={"device":"cuda","device_name":"mock GPU"}
    m=report["measurements"]
    m.update(peak_allocated_bytes=100*2**20,peak_reserved_bytes=150*2**20,
             free_before_bytes=1000*2**20,device_total_bytes=1000*2**20,limit_bytes=800*2**20)
    if failure in ("reserved","allocated"):m["peak_"+failure+"_bytes"]=900*2**20
    if failure=="free":m["free_before_bytes"]=10
    expected=copy.deepcopy(policy)
    if failure=="hardware":expected["environment"]["device_name"]="different GPU"
    report["content_id"]=identity({k:v for k,v in report.items() if k!="content_id"})
    path=tmp_path/"proof.json";path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_workload(path,expected,800)


def test_execution_rehashes_inputs_after_readiness(eval_assets,tmp_path):
    cfg=config(eval_assets,tmp_path)
    def changed(*args,**kwargs):
        with eval_assets["ckpt"].open("ab") as handle:handle.write(b"changed")
        return {"status":"READY"}
    with patch("minillm.pilot_launcher.check_readiness",side_effect=changed), \
         patch("minillm.pilot_launcher.fit",side_effect=AssertionError("must not train")):
        with pytest.raises(ValueError,match="changed after readiness"):
            launch_pilot(eval_assets["dataset"],eval_assets["tokenizer"],str(eval_assets["cfg"]),cfg,tmp_path/"changed",execute=True)


def test_unavailable_cuda_never_falls_back_to_cpu_training(eval_assets,tmp_path):
    cfg={**config(eval_assets,tmp_path),"device":"cuda","workload_report":"unverifiable-without-device"}
    with patch("torch.cuda.is_available",return_value=False), \
         patch("minillm.pilot_launcher.fit",side_effect=AssertionError("must not train")):
        report=launch_pilot(eval_assets["dataset"],eval_assets["tokenizer"],str(eval_assets["cfg"]),cfg,tmp_path/"no-cuda",execute=True)
    assert report["status"]=="INCOMPLETE" and not report["executed"]


@pytest.mark.parametrize("kind",["training","dataset"])
def test_reports_cannot_populate_protected_directories(eval_assets,tmp_path,kind):
    cfg=config(eval_assets,tmp_path)
    target=tmp_path/"run" if kind=="training" else eval_assets["dataset"]
    out=target/"report"
    with pytest.raises(ValueError,match="outside"):
        probe_workload(eval_assets["dataset"],eval_assets["tokenizer"],str(eval_assets["cfg"]),cfg,out)
    with pytest.raises(ValueError,match="outside"):
        launch_pilot(eval_assets["dataset"],eval_assets["tokenizer"],str(eval_assets["cfg"]),cfg,out)
    assert not out.exists()
