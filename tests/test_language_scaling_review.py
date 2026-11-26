"""Regression coverage for the PR #18 recovery and evaluation gates."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from minillm import language_scaling_evaluation as scaling_eval, train
from minillm.assessment_io import read_json
from minillm.checkpoint import CheckpointError, load_training_state
from minillm.corpus_utils import CorpusError, file_info, identity
from minillm.data import EpochBatchSampler, PackedPretrainDataset
from minillm.evaluation import EvaluationConfig
from minillm.language_generation_analysis import STRATEGIES
from minillm.model import MiniLLM
from minillm.training_milestones import reconcile_best_checkpoint, require_identical_weights


def _train_four_updates(tmp_path, tokenizer, model_config, packed_data, monkeypatch, losses):
    train_data, val = (PackedPretrainDataset(path, 16) for path in packed_data)
    opts = train.TrainingOptions(out=str(tmp_path / "run"), max_steps=4, milestone_steps=(0, 2, 4),
                                 device="cpu", threads=1, batch_size=1, grad_accum=2,
                                 log_every=1, val_every=1, save_every=1)
    scores = iter(losses)
    monkeypatch.setattr(train, "evaluate", lambda *a, **kw: _validation(next(scores)))
    train.set_seed(42)
    train.fit(opts, MiniLLM(model_config), tokenizer, train_data, val)
    return opts, Path(opts.out), train_data, val


def _validation(loss):
    return {"loss": loss, "perplexity": float(np.exp(loss)), "tokens": 30, "batches": 1}


@pytest.mark.parametrize("change", [None, "wrong_loss", "damaged", "missing", "wrong_tensor",
                                    "wrong_tokens", "wrong_config", "wrong_tokenizer", "wrong_run", "wrong_sha"])
def test_reconcile_best_retains_valid_artifact_or_recovers_exact_weights(
    tmp_path, tiny_tokenizer, nano_cfg, packed_data, monkeypatch, change,
):
    opts, out, data, _ = _train_four_updates(tmp_path, tiny_tokenizer, nano_cfg, packed_data, monkeypatch,
                                              (1.0, 0.9, 0.8, 0.7))
    best, checkpoint = out / "best.pt", out / "state.pt"
    original = file_info(best)
    expected = load_training_state(checkpoint)
    if change == "missing":
        best.unlink()
    elif change == "damaged":
        best.write_bytes(b"partial checkpoint")
    elif change is not None:
        payload = torch.load(best, weights_only=False)
        if change == "wrong_tensor":
            next(iter(payload["model"].values())).view(-1)[0] += 1
        else:
            field, value = {"wrong_loss": ("val_loss", .01), "wrong_tokens": ("tokens_seen", 119),
                            "wrong_tokenizer": ("tokenizer_fingerprint", "wrong"), "wrong_config": ("config", {}),
                            "wrong_run": ("training_run_id", "foreign"), "wrong_sha": ("saved_at", "rewritten")}[change]
            payload[field] = value
        torch.save(payload, best)
    discarded = file_info(best) if best.exists() else None
    sampler = EpochBatchSampler(len(data), opts.batch_size, opts.seed)
    record = reconcile_best_checkpoint(out, expected, checkpoint, data, sampler)
    assert record["step"] == 4 and record["val_loss"] == .7
    restored = torch.load(best, weights_only=False)
    require_identical_weights(expected["model"], restored["model"], "best checkpoint fixture")
    archives = sorted((out / "recovery").glob("best-step-*/best.pt"))
    if change not in (None, "missing"):
        assert len(archives) == 1 and file_info(archives[0]) == discarded
    else:
        assert not archives
    if change is None:
        assert record["file"] == original
    assert reconcile_best_checkpoint(out, expected, checkpoint, data, sampler) == record
    assert sorted((out / "recovery").glob("best-step-*/best.pt")) == archives
    assert read_json(out / "best-checkpoint-status.json") == {
        "status": "AVAILABLE", "historical_best_loss": .7, "checkpoint": record,
    }


@pytest.mark.parametrize(("recoverable", "replay_loss"), [(True, .95), (False, .95), (False, .05)])
def test_rollback_drops_future_best_and_preserves_the_training_trajectory(
    tmp_path, tiny_tokenizer, nano_cfg, packed_data, monkeypatch, recoverable, replay_loss,
):
    losses = (.9, .8, .6, .1) if recoverable else (.2, .8, .6, .1)
    opts, out, data, val = _train_four_updates(tmp_path, tiny_tokenizer, nano_cfg, packed_data, monkeypatch, losses)
    previous = out / "state.prev.pt"
    rollback = load_training_state(previous)
    assert rollback["global_step"] == 3
    original = load_training_state(out / "state.pt")
    discarded = file_info(out / "best.pt")
    original_lr = [row["learning_rate"] for row in map(json.loads, (out / "training-log.jsonl").read_text().splitlines())
                   if row["event"] == "train"]

    monkeypatch.setattr(train, "evaluate", lambda *a, **kw: _validation(replay_loss))
    train.fit(replace(opts, resume=str(previous)), MiniLLM(nano_cfg), tiny_tokenizer, data, val)
    recovered = load_training_state(out / "state.pt")
    assert recovered["history"][:-1] == rollback["history"]
    assert recovered["history"][-1]["loss"] == replay_loss
    assert recovered["history"] == read_json(out / "metrics.json")
    assert recovered["best_loss"] == min(rollback["best_loss"], replay_loss)
    best_status = read_json(out / "best-checkpoint-status.json")
    if replay_loss < rollback["best_loss"]:
        best = torch.load(out / "best.pt", weights_only=False)
        assert (best["step"], best["val_loss"]) == (4, replay_loss)
        assert best_status["status"] == "AVAILABLE"
    elif recoverable:
        best = torch.load(out / "best.pt", weights_only=False)
        assert (best["step"], best["tokens_seen"], best["val_loss"]) == (3, 90, .6)
        require_identical_weights(rollback["model"], best["model"], "historical best checkpoint fixture")
        assert best_status["status"] == "AVAILABLE"
    else:
        assert not (out / "best.pt").exists()
        assert best_status == {"status": "UNAVAILABLE", "historical_best_loss": .2, "checkpoint": None}
        assert recovered["best_checkpoint"] == rollback["best_checkpoint"]
    archive = list((out / "recovery").glob("best-step-*/best.pt"))
    assert len(archive) == 1 and file_info(archive[0]) == discarded
    require_identical_weights(original["model"], recovered["model"], "resumed weights")
    assert original["optimizer"]["param_groups"] == recovered["optimizer"]["param_groups"]
    for key, slot in original["optimizer"]["state"].items():
        require_identical_weights(slot, recovered["optimizer"]["state"][key], "resumed AdamW")
    assert torch.equal(original["rng"]["torch"], recovered["rng"]["torch"])
    new_lr = [row["learning_rate"] for row in map(json.loads, (out / "training-log.jsonl").read_text().splitlines())
              if row["event"] == "train"]
    assert new_lr == original_lr


@pytest.fixture
def historical_report(tmp_path, monkeypatch):
    directory = tmp_path / "history"
    directory.mkdir()
    config = EvaluationConfig(**read_json(scaling_eval.FORMAL_CONFIG))
    rows = [{"id": str(i), "kind": "benchmark", "generation_mode": "raw", "text_category": "zh",
             "prompt_token_ids": [10], "generation_budget": 50, "generated_token_ids": [14, 2],
             "stop_ids": [2, 7], "stop_token_id": 2, "stop_reason": "eos"} for i in range(300)]
    policy = {"config": asdict(config), "tokenizer_fingerprint": scaling_eval.FINGERPRINT,
              "benchmark_sha256": file_info(scaling_eval.ROOT / scaling_eval.BENCHMARK)["sha256"]}
    checkpoint = {"sha256": scaling_eval.HISTORICAL_FILE["sha256"], "global_step": 123}
    sealed = {"policy": policy, "checkpoint_sha256": checkpoint["sha256"]}
    evaluation_id = identity(sealed)
    (directory / "evaluation.json").write_text(json.dumps({"evaluation_id": evaluation_id, "checkpoint": checkpoint,
           "dataset": {"dataset_content_id": scaling_eval.BASE_ID}}))
    (directory / "evaluation.md").write_text("software test")
    (directory / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    files = {path.name: file_info(path) for path in directory.iterdir()}
    (directory / "evaluation-manifest.json").write_text(json.dumps({
        "schema_version": 1, "identity": sealed, "evaluation_id": evaluation_id, "checkpoint_sha256": checkpoint["sha256"],
        "tokenizer_fingerprint": scaling_eval.FINGERPRINT, "generation": asdict(config),
        "comparison_policy": policy, "files": files,
    }))
    monkeypatch.setattr(scaling_eval, "HISTORICAL_EVALUATION_FILES", {p.name: file_info(p) for p in directory.iterdir()
                        if p.name != "evaluation.md"})
    return directory, config, rows


@pytest.mark.parametrize("change", [None, "missing", "duplicate", "config", "generation_mode",
                                    "generated_token_ids", "stop_reason", "stop_token_id"])
def test_scaling_historical_greedy_uses_the_formal_gate(historical_report, change):
    directory, config, rows = historical_report
    previous, manifest = scaling_eval.load_historical_greedy(directory)
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "config":
        config = replace(config, seed=43)
    elif change:
        rows[0][change] = None
    args = (previous, rows, previous, manifest, config, STRATEGIES["greedy"], 1.0)
    if change is None:
        proof = scaling_eval.check_scaling_historical_greedy(*args)
        assert proof["status"] == "PASS" and proof["matched_tasks"] == 300
        assert proof["historical_evaluation_id"] == manifest["evaluation_id"]
    else:
        with pytest.raises(ValueError, match="Greedy"):
            scaling_eval.check_scaling_historical_greedy(*args)


@pytest.mark.parametrize("name", ["evaluation-manifest.json", "evaluation.json", "predictions.jsonl", "evaluation.md"])
def test_historical_report_rejects_tampered_files(historical_report, name):
    directory, _, _ = historical_report
    path = directory / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises((ValueError, CorpusError), match=name):
        scaling_eval.load_historical_greedy(directory)


@pytest.fixture
def publication(tmp_path, nano_cfg):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    np.zeros(1969 * 1024, dtype=np.uint16).tofile(dataset / "train.bin")
    (dataset / "dataset-manifest.json").write_text(json.dumps({"corpus_version": scaling_eval.VERSION}))
    data = PackedPretrainDataset(dataset / "train.bin", 1024)
    sampler = EpochBatchSampler(len(data), 1, 42)
    run = tmp_path / "run"
    for step in (0, 123, 245):
        directory = run / "milestones" / f"step-{step}"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text("{}")
    manifest = {"run_id": "software-run", "mode": "pretrain", "model_architecture": train.ARCHITECTURE,
                "model_config": nano_cfg.to_dict(), "training_state_version": train.TRAINING_STATE_VERSION,
                "dataset": {"train": {"dataset_content_id": "software-data", "effective_loss_tokens": 1969 * 1023},
                            "val": None},
                "execution": {}, "tokenizer_fingerprint": scaling_eval.FINGERPRINT,
                "tokenizer_vocab_size": nano_cfg.vocab_size}
    manifest["training"] = {"scheduler": "warmup_cosine", "optimizer": "AdamW", "target_optimizer_updates": 245,
                            "scheduler_total_steps": 245, "micro_batches_per_epoch": len(sampler),
                            "optimizer_updates_per_epoch": 247, "grad_accum": 8, "batch_size": 1, "seq_len": 1024,
                            "eps": 1e-8, "betas": [.9, .95], "weight_decay": .1}
    initiation = {"kind": "weights_only", "sha256": scaling_eval.INITIAL_FILE["sha256"]}
    lineage = {"initialization": initiation, "initial_launch": {"request_id": "software-request"}, "resume_history": []}
    model = {"architecture": train.ARCHITECTURE, "arch_version": train.ARCH_VERSION, "mode": "pretrain",
             "config": manifest["model_config"], "model": {"a": torch.ones(2), "b": torch.ones(1)},
             "step": 245, "tokens_seen": 2_005_080, "tokenizer_fingerprint": scaling_eval.FINGERPRINT,
             "tokenizer_vocab_size": nano_cfg.vocab_size, "pilot_initialization": initiation,
             "pilot_request_id": "software-request", "seq_len": 1024, "training_run_id": manifest["run_id"]}
    state = {key: model[key] for key in ("architecture", "arch_version", "config", "model",
                                          "tokenizer_fingerprint", "tokenizer_vocab_size")}
    state.update(global_step=245, tokens_seen=2_005_080, samples_seen=1960, epoch=0, micro_batches_consumed_in_epoch=1960,
                 manifest=manifest, training_state_version=train.TRAINING_STATE_VERSION, rng=train.rng_state(),
                 history=[{"step": 245, "loss": 7.5}], best_loss=7.5, scaler=None,
                 optimizer={"param_groups": [{"params": [0], "betas": (.9, .95), "eps": 1e-8, "weight_decay": .1,
                                               "lr": 3e-5}],
                            "state": {0: {"step": torch.tensor(245), "exp_avg": torch.zeros(2),
                                          "exp_avg_sq": torch.zeros(2)}}})
    summary = {"stop_reason": "target_reached", "pilot_initialization": initiation, "pilot_request_id": "software-request"}
    for name, obj in (("training-manifest.json", manifest), ("pilot-lineage.json", lineage), ("train-summary.json", summary)):
        (run / name).write_text(json.dumps(obj))
    milestone = run / "milestones" / "step-245"
    torch.save({**model, "val_loss": 7.5, "saved_at": "at-final-validation"}, run / "model.pt")
    torch.save({**model, "val_loss": None, "saved_at": "before-final-validation"}, milestone / "model.pt")
    torch.save(state, run / "state.pt")
    torch.save({**state, "history": [], "best_loss": None}, milestone / "state.pt")
    return dataset, run, manifest, summary, load_training_state(milestone / "state.pt"), data, sampler, lineage


@pytest.mark.parametrize(("path", "change", "message"), [
    (None, None, None),
    ("model.pt", "tensor", "tensor a"),
    ("state.pt", "tensor", "tensor a"),
    ("model.pt", "missing", "tensor names"),
    ("model.pt", "shape", "shape, dtype or values"),
    ("model.pt", "dtype", "shape, dtype or values"),
    ("model.pt", "step", "counters"),
    ("state.pt", "tokens", "tokens_seen"),
    ("state.pt", "samples", "samples_seen"),
    ("state.pt", "manifest", "training manifest"),
    ("model.pt", "tokenizer", "tokenizer"),
    ("model.pt", "lineage", "pilot lineage"),
    ("model.pt", "initialization", "pilot lineage"),
])
def test_final_publication_matches_the_validated_milestone(publication, path, change, message):
    _, run, *arguments = publication
    if path:
        path = run / path
        obj = torch.load(path, weights_only=False)
        if change == "tensor":
            obj["model"]["a"][0] *= 2
        elif change == "missing":
            obj["model"].pop("b")
        elif change == "shape":
            obj["model"]["a"] = obj["model"]["a"].reshape(1, 2)
        elif change == "dtype":
            obj["model"]["a"] = obj["model"]["a"].double()
        elif change == "manifest":
            obj["manifest"]["run_id"] = "another-run"
        else:
            field, value = {"step": ("step", 244), "tokens": ("tokens_seen", 2_005_079), "samples": ("samples_seen", 1_959),
                            "tokenizer": ("tokenizer_fingerprint", "changed"), "lineage": ("pilot_request_id", "foreign"),
                            "initialization": ("pilot_initialization", {})}[change]
            obj[field] = value
        torch.save(obj, path)
        with pytest.raises(CheckpointError, match=message):
            scaling_eval._verify_final_publication(run, *arguments)
    else:
        proof = scaling_eval._verify_final_publication(run, *arguments)
        assert proof["tensor_comparison"] == {"status": "EXACT", "tensors": 2,
                                              "compared_fields": ["name", "shape", "dtype", "values"]}
        files = proof["files"]
        assert files["model.pt"] != files["milestones/step-245/model.pt"]
        assert files["state.pt"] != files["milestones/step-245/state.pt"]


def test_final_gate_failure_prevents_export_and_formal_publication(tmp_path, monkeypatch, publication):
    dataset, run, *arguments = publication
    payload = torch.load(run / "model.pt", weights_only=False)
    payload["model"]["a"][0] *= 2
    torch.save(payload, run / "model.pt")
    monkeypatch.setattr(scaling_eval, "validate_v4_pilot_policy", lambda *a: {"ok": True})
    monkeypatch.setattr(scaling_eval, "verify_pinned_release", lambda *a: {})
    monkeypatch.setattr(scaling_eval, "base_manifest", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(scaling_eval, "_completed_nodes", lambda *a: scaling_eval._verify_final_publication(run, *arguments))
    monkeypatch.setattr(scaling_eval, "save_quantized", lambda *a, **kw: pytest.fail("invalid model reached INT4 export"))
    out = tmp_path / "evaluation"
    with pytest.raises(CheckpointError, match="published model.pt: tensor"):
        scaling_eval.evaluate_scaling(out, dataset=dataset)
    assert not out.exists()


def test_historical_mismatch_aborts_the_formal_report(tmp_path, monkeypatch, historical_report):
    historical, _, rows = historical_report
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset-manifest.json").write_text(json.dumps({"corpus_version": scaling_eval.VERSION,
                                                             "content_id": "software-data"}))
    (historical / "training-manifest.json").write_text(json.dumps({"training": {},
                       "dataset": {"train": {"dataset_content_id": scaling_eval.BASE_ID}}}))
    monkeypatch.setattr(scaling_eval, "HISTORICAL", historical / "model.pt")
    monkeypatch.setattr(scaling_eval, "HISTORICAL_FINAL", historical)
    monkeypatch.setattr(scaling_eval, "validate_v4_pilot_policy", lambda *a: {"ok": True})
    monkeypatch.setattr(scaling_eval, "verify_pinned_release", lambda *a: {"content_id": scaling_eval.BASE_ID})
    monkeypatch.setattr(scaling_eval, "base_manifest", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(scaling_eval, "_completed_nodes", lambda *a: (
        {scaling_eval.BASELINE: historical / "model.pt"}, {}, {"training": {}, "model_config": {}}, {}))
    splits = {"train": None, "val": None}
    monkeypatch.setattr(scaling_eval, "evaluation_data", lambda *a: (splits, splits, splits))
    monkeypatch.setattr(scaling_eval.MiniLLMTokenizer, "load", lambda *a: SimpleNamespace(fingerprint=scaling_eval.FINGERPRINT))
    monkeypatch.setattr(scaling_eval, "vocabulary", lambda *a: [])
    monkeypatch.setattr(scaling_eval, "load_tasks", lambda *a, scored: rows if scored else rows[:5])
    monkeypatch.setattr(scaling_eval, "_curves", lambda *a: None)
    model = SimpleNamespace(cfg=SimpleNamespace(to_dict=lambda: {}))
    monkeypatch.setattr(scaling_eval, "load_evaluation_model", lambda *a: (model, {"step": 123, "tokens_seen": 1_006_632}))
    monkeypatch.setattr(scaling_eval, "evaluate_indexed_windows", lambda *a: {})
    monkeypatch.setattr(scaling_eval, "trace_predictions", lambda *a: ([], {}, {}))
    monkeypatch.setattr(scaling_eval, "batch_generations", lambda *a, **kw: [{**rows[0], "stop_reason": "changed"}, *rows[1:]])
    monkeypatch.setattr(scaling_eval, "save_quantized", lambda *a, **kw: pytest.fail("historical mismatch reached export"))
    out = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="historical Greedy mismatch.*stop_reason"):
        scaling_eval.evaluate_scaling(out, dataset=dataset)
    assert not out.exists()
