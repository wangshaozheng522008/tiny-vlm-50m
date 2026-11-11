"""Deterministic synthetic evaluation; no network or external judge."""
import copy
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from minillm.assessment_io import read_json, records, verified_evaluation
from minillm.corpus import Pipeline, inventory, make_content
from minillm.corpus_utils import file_info
from minillm.data import collate_batch
from minillm.evaluation import (DEFAULT_BENCHMARK, DEFAULT_PROMPTS, EvaluationConfig,
    evaluate_batches, evaluate_model, generate_prediction, load_evaluation_model,
    load_tasks, ppl_from_totals, prepare_eval_data, repetition)
from minillm.evaluation_compare import compare_evaluations
from minillm.evaluation_contamination import contamination_audit
from minillm.evaluation_scoring import aggregate_scores, score_answer, validate_task
from minillm.model import MiniLLM
from minillm.quantize import save_quantized
from minillm.train import TrainingOptions, fit, save_model_checkpoint
from minillm.utils import set_seed


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    return path


def task(**kwargs):
    return {"id":"test","category":"math","prompt":"1+1=","reference":"2",
            "scoring":"exact_match","generation_mode":"raw","max_new_tokens":4,**kwargs}


@pytest.fixture
def eval_model(nano_cfg):
    torch.set_num_threads(1);set_seed(7)
    return MiniLLM(nano_cfg)


@pytest.fixture
def eval_assets(tmp_path,tiny_tokenizer,nano_cfg):
    import random
    rng=random.Random(37)
    raw=jsonl(tmp_path/"raw.jsonl",[{"text":"".join(rng.choices("中文学习模型数据科学知识训练未来天气城市",k=120))+str(i)} for i in range(80)])
    tok=tmp_path/"tokenizer.json";tiny_tokenizer.save(tok)
    pipeline=Pipeline(tmp_path/"dataset",make_content({"split":[.6,.2,.2],"quality":{"min_chars":20},"deduplication":{"method":"exact"}}),inventory([str(raw)]),log=lambda *_:None)
    pipeline.tokenize(tok)
    torch.set_num_threads(1);set_seed(7)
    model=MiniLLM(nano_cfg)
    ckpt=tmp_path/"initial.pt"
    save_model_checkpoint(ckpt,model,tiny_tokenizer,step=0,tokens=0,val_loss=None,extra={"mode":"pretrain"})
    cfg_path=tmp_path/"model-config.json";nano_cfg.save(cfg_path)
    benchmark=jsonl(tmp_path/"benchmark.jsonl",[task(),task(id="chat",prompt="1+1?",generation_mode="chat")])
    prompts=jsonl(tmp_path/"prompts.jsonl",[{"id":"fixed","prompt":"hello","generation_mode":"raw","max_new_tokens":4}])
    return dict(dataset=pipeline.out,tokenizer=tok,ckpt=ckpt,model=model,cfg=cfg_path,benchmark=benchmark,prompts=prompts)


def run_evaluation(assets,out,**kwargs):
    args=dict(dataset=assets["dataset"],benchmark=assets["benchmark"],prompts=assets["prompts"],
              config=EvaluationConfig(seq_len=16,token_budget=31,max_new_tokens=4))
    args.update(kwargs)
    return evaluate_model(assets["ckpt"],assets["tokenizer"],out,**args)


def test_shift_padding_ignore_and_weighted_loss(eval_model):
    model=eval_model
    a={"input_ids":torch.tensor([8,9,10,11]),"labels":torch.tensor([8,9,-100,11])}
    b={"input_ids":torch.tensor([12,13]),"labels":torch.tensor([12,13])}
    batch=collate_batch([a,b])
    # Deliberately non-ignore padding labels: evaluation must honor the mask.
    batch["labels"][1,2:]=0
    with torch.inference_mode():
        logits=model(batch["input_ids"],attention_mask=batch["attention_mask"]).logits
        expected=(F.cross_entropy(logits[0,0:1],torch.tensor([9]),reduction="sum")+
                  F.cross_entropy(logits[0,2:3],torch.tensor([11]),reduction="sum")+
                  F.cross_entropy(logits[1,0:1],torch.tensor([13]),reduction="sum"))
    result=evaluate_batches(model,[batch],torch.device("cpu"))
    assert result["tokens"]==3 and result["loss"]==pytest.approx(float(expected)/3,abs=1e-6)
    assert result["perplexity"]==pytest.approx(math.exp(result["loss"]))
    separate=evaluate_batches(model,[collate_batch([a]),collate_batch([b])],torch.device("cpu"))
    assert separate["loss"]==pytest.approx(result["loss"],abs=1e-6)
    limited=evaluate_batches(model,[batch],torch.device("cpu"),token_budget=2)
    assert limited["tokens"]==2


def test_empty_overflow_and_nonfinite(eval_model):
    with pytest.raises(ValueError,match="empty"):
        evaluate_batches(eval_model,[],torch.device("cpu"))
    with pytest.raises(ValueError,match="nonfinite"):
        ppl_from_totals(float("nan"),1)
    result=ppl_from_totals(1000.,1)
    assert result["perplexity"] is None and result["perplexity_status"]=="overflow_unrepresentable"
    assert not result["display_clipped"]


@pytest.mark.parametrize("rule,output,expected,extra",[
    ("exact_match","2",True,{}),("exact_match"," 2",False,{}),
    ("normalized_match"," Hello. ",True,{"reference":"hello","normalization":["strip","terminal_punctuation","casefold"]}),
    ("normalized_match","a b",False,{"reference":"ab","normalization":["strip"]}),
    ("numeric_match","2.000",True,{}),("numeric_match","2元",False,{}),
    ("numeric_match","NaN",False,{}),("numeric_match","2e0",True,{}),
    ("numeric_match","2.001",True,{"atol":.002}),("numeric_match","2.1",False,{"atol":.002}),
    ("numeric_match","2.5",False,{"numeric_type":"integer"}),
    ("multiple_choice","A",True,{"prompt":"A.甲 B.乙","reference":"A","options":{"A":"甲","B":"乙"}}),
    ("multiple_choice","答案A和B",False,{"prompt":"A.甲 B.乙","reference":"A","options":{"A":"甲","B":"乙"}}),
    ("manual_review","anything",None,{"reference":None}),
])
def test_strict_scoring(rule,output,expected,extra):
    assert score_answer(output,task(scoring=rule,**extra))["correct"] is expected


@pytest.mark.parametrize("changes",[
    {"reference":""},{"reference":False},{"scoring":"unknown"},
    {"scoring":"numeric_match","reference":"NaN"},
    {"scoring":"numeric_match","reference":"2.5","numeric_type":"integer"},
    {"scoring":"numeric_match","atol":-1},
    {"scoring":"normalized_match","normalization":["remove_all_punctuation"]},
    {"scoring":"multiple_choice","reference":"C","options":{"A":"甲","B":"乙"}},
])
def test_invalid_references(changes):
    with pytest.raises(ValueError):
        validate_task(task(**changes))


def test_modes_separate_and_manual_not_correct():
    predictions=[{"kind":"benchmark","generation_mode":mode,"category":"x","score":{"correct":correct}}
                 for mode,correct in (("raw",True),("chat",False),("chat",None))]
    result=aggregate_scores(predictions)
    assert result["raw"]["categories"]["x"]["accuracy"]==1
    assert result["chat"]["categories"]["x"]=={"correct":0,"scored":1,"manual_review":1,"accuracy":0}
    assert "total" not in result


def test_generation_seed_stops_anomalies_and_context(eval_model,tiny_tokenizer):
    cfg=EvaluationConfig(seq_len=16,max_new_tokens=4,temperature=.8)
    a=generate_prediction(eval_model,tiny_tokenizer,task(),cfg)
    torch.rand(20)
    b=generate_prediction(eval_model,tiny_tokenizer,task(),cfg)
    assert a["generated_token_ids"]==b["generated_token_ids"]
    assert not eval_model.training
    def scripted(ids,**kwargs):
        assert kwargs["stop_token_ids"]==(2,7)
        return torch.cat([ids,torch.tensor([[4,7]])],dim=1)
    with patch.object(eval_model,"generate",side_effect=scripted):
        row=generate_prediction(eval_model,tiny_tokenizer,task(generation_mode="chat",prompt="hi"),cfg)
    assert row["stop_reason"]=="end_of_turn" and row["generated_token_ids"]==[4,7]
    assert row["special_token_anomalies"]==[4] and "<|" not in row["decoded_text"]
    assert row["empty_output"]
    with pytest.raises(ValueError,match="exceeds context"):
        generate_prediction(eval_model,tiny_tokenizer,task(prompt="a"*2000),cfg)


def test_repetition_ratios():
    r=repetition("你好。你好。",[1,2,3,4]*3)
    assert r["sentence_repeat_ratio"]==.5
    assert r["token_4gram_repeat_ratio"]==pytest.approx(5/9)
    assert repetition("",[])["token_4gram_repeat_ratio"]==0


@pytest.mark.parametrize("bits",[None,4,8])
def test_fp_and_quantized_paths_read_only(eval_assets,tmp_path,tiny_tokenizer,bits):
    assets=eval_assets.copy()
    if bits:
        path=tmp_path/f"int{bits}.pt"
        save_quantized(path,assets["model"],bits=bits,tokenizer_fingerprint=tiny_tokenizer.fingerprint)
        assets["ckpt"]=path
    before={p:file_info(p) for p in [assets["ckpt"],assets["tokenizer"],assets["dataset"]/"val.bin"]}
    report=run_evaluation(assets,tmp_path/"report")
    verified_evaluation(tmp_path/"report")
    assert report["language_model"]["tokens"]==31
    assert report["checkpoint"]["quantized_bits"]==bits
    assert before=={p:file_info(p) for p in before}
    assert len(list(records(tmp_path/"report/predictions.jsonl")))==3
    with pytest.raises(ValueError,match="non-empty"):
        run_evaluation(assets,tmp_path/"report")


def test_reproducible_evaluation_and_mismatch_protection(eval_assets,tmp_path,tiny_tokenizer):
    a=run_evaluation(eval_assets,tmp_path/"a")
    b=run_evaluation(eval_assets,tmp_path/"b")
    assert a["language_model"]==b["language_model"]
    assert a["evaluation_id"]==b["evaluation_id"]
    comparison=compare_evaluations([tmp_path/"a",tmp_path/"b"],tmp_path/"comparison")
    assert comparison["comparable"] and comparison["experiments"][1]["loss_delta"]==0
    run_evaluation(eval_assets,tmp_path/"different",config=EvaluationConfig(seq_len=8,max_new_tokens=4))
    with pytest.raises(ValueError,match="incompatible"):
        compare_evaluations([tmp_path/"a",tmp_path/"different"],tmp_path/"rejected")
    independent=compare_evaluations([tmp_path/"a",tmp_path/"different"],tmp_path/"independent",allow_incomparable=True)
    assert not independent["comparable"] and independent["experiments"][1]["loss_delta"] is None
    with pytest.raises(ValueError,match="context"):
        prepare_eval_data(eval_assets["dataset"],None,"val",tiny_tokenizer,1024,eval_assets["model"].cfg)
    payload=torch.load(eval_assets["ckpt"],weights_only=False)
    for key,value in (("tokenizer_fingerprint","bad"),("architecture","tinyvlm"),
                      ("config",{**payload["config"],"model_type":"tinyvlm"}),
                      ("config",{**payload["config"],"arch_version":1})):
        modified={**payload,key:value};torch.save(modified,tmp_path/"wrong.pt")
        with pytest.raises(ValueError):
            load_evaluation_model(tmp_path/"wrong.pt",tiny_tokenizer,torch.device("cpu"))


def test_window_tail_custom_bin_and_test_distinction(eval_assets,tmp_path,tiny_tokenizer):
    path=tmp_path/"custom.bin";np.arange(34,dtype="<u2").tofile(path)
    data,info=prepare_eval_data(None,path,"val",tiny_tokenizer,16,eval_assets["model"].cfg)
    assert len(data)==2 and info["effective_loss_tokens"]==30 and info["unused_tail_tokens"]==2
    assert info["split"]=="custom"
    test=run_evaluation(eval_assets,tmp_path/"test",split="test")
    assert test["dataset"]["split"]=="test"
    path.write_bytes(b"")
    with pytest.raises((ValueError,OSError)):
        prepare_eval_data(None,path,"val",tiny_tokenizer,16,eval_assets["model"].cfg)


def test_contamination_checks_sources_and_sft(tmp_path):
    question="原始评测问题：请列出一到三个数，只写英文逗号分隔的数字。"
    rows=[task(prompt=question,reference="1,2,3")]
    source=jsonl(tmp_path/"sft.jsonl",[{"messages":[{"role":"user","content":question},{"role":"assistant","content":"1,2,3"}]}])
    report=contamination_audit(rows,sft_data=[source])
    assert report["sources"][0]["exact_task_ids"]==["test"]
    assert contamination_audit(rows)["status"]=="NOT_RUN"


def test_frozen_original_benchmark_and_prompts():
    benchmark=load_tasks(DEFAULT_BENCHMARK,scored=True)
    prompts=load_tasks(DEFAULT_PROMPTS,scored=False)
    assert len(benchmark)==160 and len(prompts)==40
    assert {r["generation_mode"] for r in benchmark}=={"raw","chat"}
    version=read_json(DEFAULT_BENCHMARK.parent/"benchmark-version.json")
    for name,info in version["files"].items():
        assert file_info(DEFAULT_BENCHMARK.parent/name)["sha256"]==info["sha256"]


def test_eval_weights_unchanged_and_failure_not_published(eval_model,eval_assets,tmp_path):
    before={k:v.clone() for k,v in eval_model.state_dict().items()}
    batch=collate_batch([{"input_ids":torch.tensor([8,9,10]),"labels":torch.tensor([8,9,10])}])
    evaluate_batches(eval_model,[batch],torch.device("cpu"))
    assert all(torch.equal(v,eval_model.state_dict()[k]) for k,v in before.items())
    with pytest.raises(ValueError,match="context"):
        run_evaluation(eval_assets,tmp_path/"failed",config=EvaluationConfig(seq_len=1024,max_new_tokens=4))
    assert not (tmp_path/"failed").exists() and not list(tmp_path.glob(".failed.*"))


def test_manifest_policy_tampering_is_rejected(eval_assets,tmp_path):
    from minillm.training_manifest import write_json
    run_evaluation(eval_assets,tmp_path/"evaluation")
    path=tmp_path/"evaluation/evaluation-manifest.json"
    manifest=read_json(path);manifest["comparison_policy"]["config"]["seq_len"]=3
    write_json(path,manifest)
    with pytest.raises(ValueError,match="inconsistent"):
        verified_evaluation(tmp_path/"evaluation")


def test_manual_tasks_and_sft_checkpoint_metadata(eval_assets,tmp_path,tiny_tokenizer):
    benchmark=jsonl(tmp_path/"manual.jsonl",[task(scoring="manual_review",reference=None)])
    payload=torch.load(eval_assets["ckpt"],weights_only=False);payload["mode"]="sft"
    sft=tmp_path/"sft.pt";torch.save(payload,sft)
    report=run_evaluation({**eval_assets,"ckpt":sft},tmp_path/"manual-evaluation",benchmark=benchmark)
    assert report["checkpoint"]["mode"]=="sft"
    assert report["scores"]["raw"]["categories"]["math"]["scored"]==0
    assert report["scores"]["raw"]["categories"]["math"]["manual_review"]==1


def test_frozen_version_rejects_changed_answers(tmp_path):
    from minillm.training_manifest import write_json
    path=jsonl(tmp_path/"benchmark.jsonl",[task()])
    write_json(tmp_path/"benchmark-version.json",{"schema_version":1,"version":"test-v1","files":{"benchmark.jsonl":{"sha256":file_info(path)["sha256"],"records":1}}})
    jsonl(path,[task(reference="wrong")])
    with pytest.raises(ValueError,match="frozen"):
        load_tasks(path,scored=True)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="optional CUDA equivalence probe")
def test_cuda_fp32_loss_matches_cpu(eval_model):
    batch=collate_batch([{"input_ids":torch.tensor([8,9,10,11]),"labels":torch.tensor([8,9,10,11])}])
    cpu=evaluate_batches(eval_model,[batch],torch.device("cpu"))
    gpu=evaluate_batches(eval_model.to("cuda"),[batch],torch.device("cuda"))
    assert gpu["tokens"]==cpu["tokens"] and gpu["loss"]==pytest.approx(cpu["loss"],abs=1e-4)


@pytest.mark.parametrize("scored",[True,False])
def test_registered_assets_are_verified(tmp_path,scored):
    from minillm.training_manifest import write_json
    name="benchmark.jsonl" if scored else "prompts.jsonl"
    path=jsonl(tmp_path/name,[task()])
    write_json(tmp_path/"benchmark-version.json",{"schema_version":1,"version":"test-v1",
        "files":{name:{"sha256":file_info(path)["sha256"],"records":1}}})
    assert len(load_tasks(path,scored=scored))==1
    renamed=tmp_path/"renamed.jsonl";renamed.write_bytes(path.read_bytes())
    with pytest.raises(ValueError,match="renamed.jsonl.*not registered"):
        load_tasks(renamed,scored=scored)


@pytest.mark.parametrize("failure",["sha","count","unregistered","json","root","schema_missing",
                                  "schema_unsupported","schema_bool","version_missing","files_missing",
                                  "files_list","bad_sha","bad_count","broken_declaration"])
def test_invalid_frozen_declarations_are_rejected(tmp_path,failure):
    path=jsonl(tmp_path/"benchmark.jsonl",[task()])
    version={"schema_version":1,"version":"test-v1","files":{"benchmark.jsonl":{
        "sha256":file_info(path)["sha256"],"records":1}}}
    if failure=="sha":version["files"]["benchmark.jsonl"]["sha256"]="0"*64
    elif failure=="count":version["files"]["benchmark.jsonl"]["records"]=2
    elif failure=="unregistered":version["files"]={"different.jsonl":version["files"]["benchmark.jsonl"]}
    elif failure=="root":version=[]
    elif failure=="schema_missing":del version["schema_version"]
    elif failure=="schema_unsupported":version["schema_version"]=2
    elif failure=="schema_bool":version["schema_version"]=True
    elif failure=="version_missing":del version["version"]
    elif failure=="files_missing":del version["files"]
    elif failure=="files_list":version["files"]=[]
    elif failure=="bad_sha":version["files"]["benchmark.jsonl"]["sha256"]="invalid"
    elif failure=="bad_count":version["files"]["benchmark.jsonl"]["records"]=True
    declaration=tmp_path/"benchmark-version.json"
    if failure=="broken_declaration":
        declaration.symlink_to(tmp_path/"missing.json")
    else:
        declaration.write_text("{" if failure=="json" else json.dumps(version))
    before=path.read_bytes()
    with pytest.raises(ValueError,match="frozen"):
        load_tasks(path,scored=True)
    assert path.read_bytes()==before


def test_custom_unfrozen_identity_and_independent_prompt_version(eval_assets,tmp_path):
    from minillm.training_manifest import write_json
    # Benchmark has no declaration; Fixed Prompts has its own strict version.
    directory=tmp_path/"frozen-prompts";directory.mkdir()
    prompts=jsonl(directory/"prompts.jsonl",[{"id":"p","prompt":"hi","generation_mode":"raw","max_new_tokens":4}])
    declaration=directory/"benchmark-version.json"
    write_json(declaration,{"schema_version":1,"version":"prompts-v1","files":{
        prompts.name:{"sha256":file_info(prompts)["sha256"],"records":1}}})
    run_evaluation(eval_assets,tmp_path/"custom-report",prompts=prompts)
    manifest=read_json(tmp_path/"custom-report/evaluation-manifest.json")
    policy=manifest["comparison_policy"]
    assert policy["benchmark_provenance"]["status"]=="CUSTOM_UNFROZEN"
    assert policy["benchmark_provenance"]["sha256"]==file_info(eval_assets["benchmark"])["sha256"]
    assert policy["fixed_prompts_provenance"]["status"]=="FROZEN"
    assert str(declaration.resolve()) in manifest["inputs"]
