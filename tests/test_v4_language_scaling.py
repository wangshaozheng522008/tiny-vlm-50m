"""Approval, recovery, and evaluation contracts for the 2M language profile."""

import json
from dataclasses import replace
from pathlib import Path
import signal

import numpy as np
import pytest
import torch

from minillm import train, v4_language_scaling as scaling
from minillm.assessment_io import read_json
from minillm.checkpoint import load_training_state
from minillm.corpus_utils import CorpusError, file_info
from minillm.corpus_v4 import load_config
from minillm.data import PackedPretrainDataset
from minillm.evaluation import EvaluationConfig
from minillm.language_scaling_evaluation import evaluation_data, verify_int4_size
from minillm.model import MiniLLM
from minillm.training_milestones import verify_milestone
from minillm.v4_language_pilot import LANGUAGE_CONFIG, is_language, validate_language_policy
from minillm.v4_training_pilot import validate_pilot_budget
from minillm.workload import parse_training_config


def test_version_budget_and_schedule_are_explicit_and_legacy_stays_compatible(monkeypatch):
    monkeypatch.chdir(scaling.ROOT)
    old, new = load_config(LANGUAGE_CONFIG), load_config(scaling.CORPUS_CONFIG)
    for config in (old, new):
        assert is_language(config)
        validate_language_policy(config)
    assert new["targets"]["category_token_ratios"] == old["targets"]["category_token_ratios"]
    new["targets"]["category_token_ratios"]["zh_general"] -= .01
    with pytest.raises(CorpusError, match="pinned"):
        validate_language_policy(new)

    raw = read_json(scaling.TRAINING_CONFIG)
    scaling.validate_training_config(raw, scaling.WORKSPACE)
    opts, extra = parse_training_config(raw)
    assert validate_pilot_budget(1960, opts, extra, corpus_version=scaling.VERSION)["planned_effective_tokens"] == 2_005_080
    with pytest.raises(ValueError, match="actual fixed windows"):
        validate_pilot_budget(1959, opts, extra, corpus_version=scaling.VERSION)
    with pytest.raises(ValueError, match="exactly 1,000,000"):
        validate_pilot_budget(1960, opts, extra, corpus_version=old["corpus_version"])
    with pytest.raises(ValueError, match="pinned independent"):
        scaling.validate_training_config({**raw, "max_steps": 123}, scaling.WORKSPACE)
    with pytest.raises(ValueError, match="pinned formal evaluation"):
        scaling.validate_training_config(raw, scaling.WORKSPACE, EvaluationConfig(seed=43))
    with pytest.raises(ValueError, match="pinned Step 0"):
        scaling.validate_launch_config({**raw, "init_from": str(scaling.HISTORICAL)})


def test_new_approval_scope_and_pinned_inventory(monkeypatch):
    raw = {"corpus_versions": ["minillm-corpus-v4-language-pilot-1m-v1"]}
    with pytest.raises(CorpusError, match="2M training scope"):
        scaling.validate_scope(raw)
    entry = {"corpus_versions": [scaling.VERSION], "training_scope": {
        "corpus_version": scaling.VERSION, "base_content_id": scaling.BASE_ID,
        "effective_train_tokens": scaling.TARGET, "reviewer": "software fixture",
        "reviewed_at": "2026-11-26", "confirmation": "software fixture only"},
        "v4_source": {"category": "zh_general", "language": "zh", "family_policy": "document",
                      "priority": 1, "transformation_history": []},
        "release": "software fixture", "local_inventory": {"files": {"/tmp/fixture": {"bytes": 18, "sha256": "a"*64}}}}
    scaling.validate_scope(entry)
    old = {"source_id": "fixture", "source_review": {"entry": entry}}
    monkeypatch.setattr(scaling, "base_manifest", lambda *names: {"license_sources": {"sources": [old]}})
    scaling.validate_source_set({"sources": [old]})
    changed = {**entry, "local_inventory": {"files": {"/tmp/fixture": {"bytes": 18, "sha256": "b"*64}}}}
    with pytest.raises(CorpusError, match="snapshot"):
        scaling.validate_source_set({"sources": [{**old, "source_review": {"entry": changed}}]})
    with pytest.raises(CorpusError, match="2M training scope"):
        scaling.validate_scope({**entry, "training_scope": {**entry["training_scope"], "effective_train_tokens": 1_006_632}})


def test_fixed_release_rejects_val_hash_and_tokenizer_drift(tmp_path, monkeypatch):
    names = scaling.SPLIT_FILES + scaling.EVAL_FILES + ("tokenizer.json",)
    for name in names:
        (tmp_path / name).write_text(name)
    manifest = {"files": {name: file_info(tmp_path / name) for name in names}}
    monkeypatch.setattr(scaling, "base_manifest", lambda *args: manifest)
    assert scaling.verify_pinned_release(tmp_path) == manifest
    for changed in ("val.bin", "tokenizer.json"):
        (tmp_path / changed).write_text("wrong")
        with pytest.raises(CorpusError, match=changed):
            scaling.verify_pinned_release(tmp_path)
        (tmp_path / changed).write_text(changed)


def test_sampler_consumes_exact_245_updates_once():
    opts, _ = parse_training_config(scaling.TRAINING_CONFIG)
    ratios = load_config(scaling.CORPUS_CONFIG)["targets"]["category_token_ratios"]
    docs = [{"category": k, "tokens": int(1960 * r) * 1024} for k, r in ratios.items() if r]
    docs[-1]["tokens"] += 1
    report = scaling.sampler_preview(docs, 1024, 1960, ratios, opts)
    assert report["consumed_windows"] == 1960
    assert report["effective_tokens"] == 245 * 8184
    assert report["categories"]["zh_general"]["consumed_effective_tokens"] == 882 * 1023
    with pytest.raises(ValueError):
        scaling.sampler_preview(docs, 1024, 1959, ratios, opts)


def test_milestones_resume_same_rng_weights_and_schedule(tmp_path, tiny_tokenizer, nano_cfg, packed_data, monkeypatch):
    data, val = [PackedPretrainDataset(path, 16) for path in packed_data]
    opts = train.TrainingOptions(out=str(tmp_path / "resume"), max_steps=4, milestone_steps=(0, 2, 4),
                                 device="cpu", threads=1, batch_size=1, grad_accum=2,
                                 log_every=1, val_every=2, save_every=2)

    def model():
        train.set_seed(42)
        return MiniLLM(nano_cfg)

    initial = model()
    uninterrupted = model()
    interrupted_update = train._train_step
    count = 0

    def stop(*args, **kwargs):
        nonlocal count
        result = interrupted_update(*args, **kwargs)
        count += 1
        if count == 2:
            signal.raise_signal(signal.SIGTERM)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(train, "_train_step", stop)
        partial = train.fit(opts, initial, tiny_tokenizer, data, val)
    assert partial["global_step"] == 2
    out = Path(opts.out)
    plan = read_json(out / "training-manifest.json")
    snapshots = {step: verify_milestone(out / "milestones" / f"step-{step}", plan, step=step, tokens=step*30)
                 for step in (0, 2)}
    with (out / "training-log.jsonl").open("a") as handle:
        handle.write('{"event":"train","global_step":3,"tokens_seen":90}\n')
    result = train.fit(replace(opts, resume=str(out / "state.pt")), model(), tiny_tokenizer, data, val)
    assert result["tokens_seen"] == 120
    assert list((out / "recovery").glob("step-2-*/training-log.jsonl"))
    for step, snapshot in snapshots.items():
        assert snapshot == verify_milestone(out / "milestones" / f"step-{step}", plan, step=step, tokens=step*30)

    other = tmp_path / "full"
    train.fit(replace(opts, out=str(other)), uninterrupted, tiny_tokenizer, data, val)
    a = load_training_state(out / "state.pt")
    b = load_training_state(other / "state.pt")
    assert a["manifest"]["training"] == b["manifest"]["training"]
    assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert all(torch.equal(tensor, b["model"][key]) for key, tensor in a["model"].items())
    log = [json.loads(line) for line in (out / "training-log.jsonl").read_text().splitlines()]
    assert [row["learning_rate"] for row in log if row["event"] == "train"] == [
        train.lr_at(i, 4, opts.lr, opts.min_lr, 0) for i in range(4)]
    verify_milestone(out / "milestones/step-4", plan, step=4, tokens=120)


def test_common_windows_and_int4_inclusive_limit(tmp_path):
    files = {}
    for split in ("train", "val", "test"):
        np.arange(17, dtype=np.uint16).tofile(tmp_path / f"{split}.bin")
        index = tmp_path / f"{split}.token-index.jsonl"
        index.write_text('{"category":"zh_general","tokens":17}\n')
        files[index.name] = file_info(index)
    budget = tmp_path / "budget-report.json"
    budget.write_text('{"category_token_ratios":{"zh_general":1.0}}')
    files[budget.name] = file_info(budget)
    (tmp_path / "dataset-manifest.json").write_text(json.dumps({"files": files}))
    config = EvaluationConfig(seq_len=4, token_budget=6)
    _, _, left = evaluation_data(config, original=tmp_path)
    _, _, right = evaluation_data(config, original=tmp_path)
    assert left == right
    assert left["train"]["supervised_tokens"] == 6
    assert left["val"]["supervised_tokens"] == left["test"]["supervised_tokens"] == 12
    export = tmp_path / "model-int4.pt"
    with export.open("wb") as handle:
        handle.truncate(50_000_000)
    assert verify_int4_size(export)["bytes"] == 50_000_000
    with export.open("r+b") as handle:
        handle.truncate(50_000_001)
    with pytest.raises(ValueError, match="exceeds"):
        verify_int4_size(export)


def test_runner_rejects_a_historical_report_directory():
    import subprocess
    import sys

    result = subprocess.run([sys.executable, str(scaling.ROOT / "scripts/run_v4_language_2m.py"),
                             "prepare", "--out", str(scaling.ROOT / "reports/v4-language-degeneration-diagnostics")],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "reports must be separate" in result.stderr
