import copy
import json
import signal
from unittest.mock import patch

import pytest

from minillm.assessment_io import records
from minillm.data import PackedPretrainDataset
from minillm.train import TrainingOptions, fit
from minillm.training_analysis import TrendConfig, analyze_training, trends
from minillm.training_pilot import evaluate_pilot
from minillm.tokenizer import MiniLLMTokenizer
from test_evaluation import eval_assets, run_evaluation


@pytest.fixture
def trained_run(eval_assets,tmp_path):
    tokenizer=MiniLLMTokenizer.load(eval_assets["tokenizer"])
    run=tmp_path/"run"
    options=TrainingOptions(out=str(run),device="cpu",precision="fp32",threads=1,
        seed=7,batch_size=2,grad_accum=2,max_steps=8,log_every=1,val_every=1,save_every=2)
    summary=fit(options,eval_assets["model"],tokenizer,
        PackedPretrainDataset(eval_assets["dataset"]/"train.bin",16),
        PackedPretrainDataset(eval_assets["dataset"]/"val.bin",16))
    return eval_assets,run,summary,options


def test_real_training_analysis(trained_run,tmp_path):
    _,run,summary,_=trained_run
    report=analyze_training(run,tmp_path/"analysis")
    assert report["status"]=="OK",report["errors"]
    assert report["curve_rows"][-1]["tokens_seen"]==summary["tokens_seen"]
    assert (tmp_path/"analysis/curves/training.csv").is_file()
    assert report["checkpoint"]["status"]=="PASS"


@pytest.mark.parametrize("kind,expected",[
    ("fall","VALIDATION_DECREASING"),("flat","VALIDATION_PLATEAU"),
    ("diverge","TRAIN_DOWN_VALIDATION_UP"),("gradient","SUSTAINED_HIGH_GRADIENT"),
    ("speed","THROUGHPUT_DROP"),("jump","LOSS_JUMP")])
def test_configurable_trend_rules(kind,expected):
    train=[{"loss":4-i*.1,"gradient_norm":1,"tokens_per_second":100} for i in range(8)]
    val=[{"loss":4-i*.1} for i in range(8)]
    if kind=="flat":
        val=[{"loss":3.} for _ in range(8)]
    elif kind=="diverge":
        val=[{"loss":3+i*.1} for i in range(8)]
    elif kind=="gradient":
        for row in train:row["gradient_norm"]=200
    elif kind=="speed":
        for row in train[-3:]:row["tokens_per_second"]=10
    elif kind=="jump":
        train[-1]["loss"]=20
    assert expected in trends(train,val,TrendConfig())
    assert "TRAIN_DOWN_VALIDATION_UP" not in trends(train[:1],val[:1],TrendConfig())


@pytest.mark.parametrize("failure",["missing","conflict","nan","manifest","checkpoint","duplicate"])
def test_log_and_artifact_failures(trained_run,tmp_path,failure):
    _,run,_,_=trained_run
    path=run/"training-log.jsonl";rows=list(records(path))
    train=[r for r in rows if r["event"]=="train"]
    if failure=="missing":
        rows.remove(train[2])
    elif failure=="conflict":
        rows.append({**train[-1],"loss":train[-1]["loss"]+1})
    elif failure=="nan":
        train[2]["loss"]=float("nan")
    elif failure=="manifest":
        (run/"training-manifest.json").write_text("{}")
    elif failure=="checkpoint":
        (run/"state.pt").write_bytes(b"incomplete")
    elif failure=="duplicate":
        rows.append(copy.deepcopy(train[-1]))
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    report=analyze_training(run,tmp_path/"analysis")
    if failure=="duplicate":
        assert report["status"]=="OK" and report["identical_duplicate_records"]==1
    else:
        assert report["status"]=="ISSUES" and report["errors"]


def test_post_run_pilot_engineering_acceptance(trained_run,tmp_path):
    assets,run,summary,_=trained_run
    run_evaluation(assets,tmp_path/"baseline")
    final={**assets,"ckpt":run/"model.pt"}
    run_evaluation(final,tmp_path/"final")
    report=evaluate_pilot(run,tmp_path/"baseline",tmp_path/"final",tmp_path/"pilot")
    assert report["engineering"]=="ENGINEERING_PASS",report["errors"]
    assert report["tokens_seen"]==summary["tokens_seen"]
    assert report["estimated_logged_wall_seconds"]>0
    assert report["capability"] in ("CAPABILITY_IMPROVING","CAPABILITY_INCONCLUSIVE")
    (run/"model.pt").write_bytes(b"broken")
    failed=evaluate_pilot(run,tmp_path/"baseline",tmp_path/"final",tmp_path/"pilot-broken")
    assert failed["engineering"]=="ENGINEERING_FAIL"


def train_validation_case(assets,tmp_path,*,configured,val_every=1,interrupt_at=None):
    import minillm.train as training
    tokenizer=MiniLLMTokenizer.load(assets["tokenizer"])
    run=tmp_path/"validation-run"
    opts=TrainingOptions(out=str(run),device="cpu",precision="fp32",threads=1,
        seed=7,batch_size=2,grad_accum=1,max_steps=3,log_every=1,val_every=val_every,val_limit=1)
    original=training._train_step
    updates=0
    def step(*args,**kwargs):
        nonlocal updates
        result=original(*args,**kwargs);updates+=1
        if updates==interrupt_at:
            signal.raise_signal(signal.SIGTERM)
        return result
    with patch.object(training,"_train_step",side_effect=step):
        summary=fit(opts,assets["model"],tokenizer,
                    PackedPretrainDataset(assets["dataset"]/"train.bin",16),
                    PackedPretrainDataset(assets["dataset"]/"val.bin",16) if configured else None)
    return run,summary


def test_no_validation_dataset_needs_no_metrics_and_only_train_curve(eval_assets,tmp_path):
    run,summary=train_validation_case(eval_assets,tmp_path,configured=False)
    assert not (run/"metrics.json").exists() and summary["final_val"] is None
    report=analyze_training(run,tmp_path/"analysis")
    assert report["status"]=="OK",report["errors"]
    assert report["validation"]["status"]=="NOT_RUN"
    assert report["validation"]["reason"]=="not_configured"
    assert report["validation_rows"]==[]
    assert all("validation_loss" not in r and "perplexity" not in r for r in report["curve_rows"])
    assert {p.name for p in (tmp_path/"analysis/curves").iterdir()}=={"training.csv"}


@pytest.mark.parametrize("val_every",[0,1])
def test_final_validation_even_when_periodic_disabled(eval_assets,tmp_path,val_every):
    run,_=train_validation_case(eval_assets,tmp_path,configured=True,val_every=val_every)
    report=analyze_training(run,tmp_path/"analysis")
    assert report["status"]=="OK",report["errors"]
    assert report["validation"]["status"]=="COMPLETED" and report["validation"]["final_status"]=="COMPLETED"
    assert len(report["validation_rows"])==(1 if val_every==0 else 3)
    assert (tmp_path/"analysis/curves/validation.csv").exists()


@pytest.mark.parametrize("failure",["metrics_missing","log_missing","all_records_missing","corrupt_metrics"])
def test_configured_validation_artifacts_cannot_disappear(eval_assets,tmp_path,failure):
    run,_=train_validation_case(eval_assets,tmp_path,configured=True)
    if failure in ("metrics_missing","all_records_missing"):
        (run/"metrics.json").unlink()
    elif failure=="corrupt_metrics":
        (run/"metrics.json").write_text("broken")
    if failure in ("log_missing","all_records_missing"):
        path=run/"training-log.jsonl"
        path.write_text("".join(json.dumps(r)+"\n" for r in records(path) if r["event"]!="validation"))
    report=analyze_training(run,tmp_path/"analysis")
    assert report["status"]=="ISSUES"
    assert report["validation"]["status"]=="ERROR" and report["errors"]


@pytest.mark.parametrize("interrupt_at,val_every,observations",[(1,0,0),(1,1,0),(2,1,1)])
def test_interruption_does_not_require_unexecuted_final_validation(eval_assets,tmp_path,interrupt_at,val_every,observations):
    run,summary=train_validation_case(eval_assets,tmp_path,configured=True,val_every=val_every,interrupt_at=interrupt_at)
    assert summary["stop_reason"]=="SIGTERM" and summary["final_val"] is None
    assert (run/"metrics.json").exists()==bool(observations)
    report=analyze_training(run,tmp_path/"analysis")
    assert report["status"]=="OK",report["errors"]
    assert report["validation"]["status"]==("COMPLETED" if observations else "NOT_RUN")
    assert report["validation"]["configured"] and report["validation"]["observations"]==observations
    assert report["validation"]["final_status"]=="NOT_RUN"
    if observations:
        (run/"metrics.json").unlink()
        assert analyze_training(run,tmp_path/"missing-prior")["validation"]["status"]=="ERROR"
