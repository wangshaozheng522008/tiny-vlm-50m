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
