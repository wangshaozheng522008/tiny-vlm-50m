"""Offline V4 fixture; no research corpus, historical model, or GPU is used."""
from pathlib import Path

from minillm.config import MINILLM_NANO
from minillm.corpus_utils import file_info
from minillm.evaluation import EvaluationConfig
from minillm.tokenizer import MiniLLMTokenizer
from minillm.training_manifest import write_json
from test_corpus_v4_manifest import _formal_pipeline


def assets(tmp_path):
    pipeline = _formal_pipeline(tmp_path)
    pipeline.finalize()
    tokenizer = pipeline.out / "tokenizer.json"
    cfg = MINILLM_NANO.copy()
    cfg.vocab_size = MiniLLMTokenizer.load(tokenizer).vocab_size
    model_config = tmp_path / "nano.json"
    cfg.save(model_config)
    suite = tmp_path / "fixture-evaluation"
    suite.mkdir()
    (suite / "candidates.jsonl").write_text(
        '{"schema_version":1,"id":"fixture-math","category":"math","prompt":"1+1=",'
        '"candidates":{"A":"2","B":"3"},"reference":"A"}\n')
    (suite / "prompts.jsonl").write_text(
        '{"id":"fixture-generation","prompt":"中文","generation_mode":"raw","max_new_tokens":2}\n')
    write_json(suite / "benchmark-version.json", {"schema_version": 1, "version": "cpu-nano-fixture-v1",
        "files": {name: {"sha256": file_info(suite / name)["sha256"], "records": 1}
                  for name in ("candidates.jsonl", "prompts.jsonl")}})
    training = {"out": str(tmp_path / "run"), "seq_len": 16, "batch_size": 1,
                "grad_accum": 2, "max_steps": 4, "target_tokens": 100,
                "planned_effective_tokens": 120, "allow_data_repetition": False,
                "device": "cpu", "precision": "fp32", "threads": 1, "seed": 42,
                "log_every": 1, "val_every": 1, "save_every": 1, "val_limit": 1}
    return {"pipeline": pipeline, "dataset": pipeline.out, "tokenizer": tokenizer,
            "model_config": str(model_config), "training_config": training,
            "workspace": tmp_path / "workspace", "cpu_nano_test": True,
            "evaluation_config": EvaluationConfig(stage="pretrain", seq_len=16, token_budget=31, max_new_tokens=2),
            "suites": [suite]}


def arguments(value):
    return {k: v for k, v in value.items() if k != "pipeline"}
