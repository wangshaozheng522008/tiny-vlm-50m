from dataclasses import asdict
from unittest.mock import patch

import pytest

from minillm.training_readiness import check_readiness, main, planned_tokens
from test_evaluation import eval_assets
from test_training_analysis import trained_run


def options(tmp_path):
    return {"out":str(tmp_path/"new-run"),"seq_len":16,"batch_size":1,"grad_accum":1,
            "max_steps":2,"device":"cpu","precision":"fp32","threads":1,"seed":7}


def readiness(assets,tmp_path,config=None,name="readiness"):
    return check_readiness(assets["dataset"],assets["tokenizer"],str(assets["cfg"]),
                           config or options(tmp_path),tmp_path/name)


def test_normal_ready(eval_assets,tmp_path):
    with patch("torch.cuda.is_available",return_value=False):
        report=readiness(eval_assets,tmp_path)
    assert report["status"]=="READY",report["checks"]
    assert report["details"]["token_budget"]["planned_effective_tokens"]==30
    assert any(c["check"]=="CUDA compatibility" and c["status"]=="NOT_RUN" and not c["required"] for c in report["checks"])


@pytest.mark.parametrize("failure",["data","tokenizer","val","disk","output","schedule","budget","batch"])
def test_blocking_failures(eval_assets,tmp_path,failure):
    assets=eval_assets.copy();config=options(tmp_path)
    if failure=="data":
        with (assets["dataset"]/"train.bin").open("r+b") as handle:handle.write(b"\xff\xff")
    elif failure=="tokenizer":
        assets["tokenizer"]=tmp_path/"bad-tokenizer.json";assets["tokenizer"].write_text("{}")
    elif failure=="val":
        (assets["dataset"]/"val.bin").write_bytes(b"")
    elif failure=="output":
        path=tmp_path/"new-run";path.mkdir();(path/"model.pt").write_bytes(b"existing")
    elif failure=="schedule":
        config["lr"]=-1
    elif failure=="budget":
        config["planned_effective_tokens"]=300
    elif failure=="batch":
        config["grad_accum"]=0
    if failure=="disk":
        with patch("minillm.training_readiness.shutil.disk_usage",return_value=type("Space",(),{"free":1})()):
            report=readiness(assets,tmp_path,config)
    else:
        report=readiness(assets,tmp_path,config)
    assert report["status"]=="BLOCKED"


def test_unverified_required_hardware_is_incomplete(eval_assets,tmp_path):
    config={**options(tmp_path),"device":"cuda"}
    with patch("torch.cuda.is_available",return_value=False):
        report=readiness(eval_assets,tmp_path,config)
    assert report["status"]=="INCOMPLETE",report["checks"]
    assert any(c["required"] and c["status"]=="UNKNOWN" for c in report["checks"])


def test_resume_manifest_cursor_and_output_ownership(trained_run,tmp_path):
    assets,run,_,opts=trained_run
    config={**asdict(opts),"seq_len":16,"resume":str(run/"state.pt")}
    good=readiness(assets,tmp_path,config)
    assert good["status"]=="READY",good["checks"]
    bad=readiness(assets,tmp_path,{**config,"grad_accum":1},name="incompatible")
    assert bad["status"]=="BLOCKED"


def test_exact_budget_handles_incomplete_update():
    assert planned_tokens(11,2,4,12,1)==8*11
    assert planned_tokens(11,2,4,12,2)==11*11
    assert planned_tokens(11,2,4,12,5)==30*11


@pytest.mark.parametrize("delta,baseline,exception,expected,code",[
    (64,0,None,"INCOMPLETE",2),
    (512,0,None,"BLOCKED",1),
    (64,4096,None,"INCOMPLETE",2),
    (None,0,RuntimeError("CUDA operator failed"),"BLOCKED",1),
])
def test_short_cuda_probe_never_approves_training_vram(eval_assets,tmp_path,delta,baseline,exception,expected,code):
    import json
    from minillm.assessment_io import read_json
    from minillm.training_readiness import SHORT_PROBE_SCOPE
    config={**options(tmp_path),"device":"cuda","seq_len":128,"batch_size":4,"gpu_memory_limit_mib":128}
    config_path=tmp_path/"training.json";config_path.write_text(json.dumps(config))
    def probe(cfg,opts,tokenizer,device,state,seq_len):
        if exception:
            raise exception
        assert device.type=="cuda" and opts.batch_size==4 and seq_len==128
        return {"model_config":cfg.to_dict(),"device":str(device),"sequence_length":16,"batch_size":1,
                "peak_vram_mib":baseline+delta,"baseline_vram_mib":baseline,
                "probe_peak_delta_mib":delta,"scope":SHORT_PROBE_SCOPE}
    with patch("torch.cuda.is_available",return_value=True), \
         patch("minillm.training_readiness._probe_configured_model",side_effect=probe) as measured, \
         patch("minillm.training_readiness.environment",side_effect=lambda device:{"device":str(device)}):
        result=main(["--dataset",str(eval_assets["dataset"]),"--tokenizer",str(eval_assets["tokenizer"]),
                     "--config",str(eval_assets["cfg"]),"--training-config",str(config_path),
                     "--out",str(tmp_path/"readiness")])
    report=read_json(tmp_path/"readiness/readiness.json")
    assert report["status"]==expected and result==code
    measured.assert_called_once()
    gpu=next(c for c in report["checks"] if c["check"]=="GPU memory probe limit")
    assert gpu["status"]==("FAIL" if delta is not None and delta>128 else "UNKNOWN")
    assert SHORT_PROBE_SCOPE in gpu["detail"]
    compatibility=[c for c in report["checks"] if c["check"]=="CUDA compatibility"]
    assert compatibility and all(c["status"]==("FAIL" if exception else "PASS") for c in compatibility)
    if not exception:
        assert report["details"]["probe"]["model_config"]==eval_assets["model"].cfg.to_dict()
        assert report["details"]["probe"]["peak_vram_mib"]==baseline+delta
    assert SHORT_PROBE_SCOPE in (tmp_path/"readiness/readiness.md").read_text()


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_limit_without_cuda_remains_unknown(eval_assets,tmp_path,device):
    config={**options(tmp_path),"device":device,"gpu_memory_limit_mib":128}
    with patch("torch.cuda.is_available",return_value=False):
        report=readiness(eval_assets,tmp_path,config)
    assert report["status"]=="INCOMPLETE"
    assert next(c for c in report["checks"] if c["check"]=="GPU memory probe limit")["status"]=="UNKNOWN"
    assert not any(c["check"]=="CUDA compatibility" and c["status"]=="PASS" for c in report["checks"])
