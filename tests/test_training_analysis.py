import copy
import json

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
