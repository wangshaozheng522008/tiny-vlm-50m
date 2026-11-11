"""CPU nano regression tests for complete-update state transactions."""

from __future__ import annotations

import errno
import json
import os
import random
import signal
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

import minillm.checkpoint as checkpoint
import minillm.train as training
from minillm.data import EpochBatchSampler, PackedPretrainDataset, SFTDataset, collate_batch
from minillm.sft import sft_batch_factory
from minillm.train import CheckpointError, TrainingOptions, fit, load_training_state
from minillm.utils import set_seed


@pytest.fixture
def small_data(tmp_path, tiny_tokenizer):
    path = tmp_path / "train.bin"
    (np.arange(11 * 12 + 1) % (tiny_tokenizer.vocab_size - 16) + 16).astype(np.uint16).tofile(path)
    return PackedPretrainDataset(path, 12)


def options(out, **kwargs):
    return TrainingOptions(**{
        "out": str(out), "device": "cpu", "precision": "fp32", "threads": 1,
        "seed": 17, "batch_size": 2, "grad_accum": 4, "max_steps": 7,
        "lr": 1e-3, "min_lr": 1e-4, "warmup_ratio": 0.4,
        "log_every": 1, "save_every": 1, "val_every": 2, "val_limit": 1, **kwargs,
    })


def new_model(cfg, seed=17, dropout=0.0):
    set_seed(seed)
    cfg = replace(cfg, hidden_dropout=dropout, attention_dropout=dropout)
    return training.MiniLLM(cfg)


def run_segment(opts, model, tokenizer, data, *, stop_at=None, sig=signal.SIGINT,
                order=None, draws=None, mode="pretrain", val=None):
    """Deliver real termination signals from inside an update, then finish it."""
    sampler = EpochBatchSampler(len(data), opts.batch_size, opts.seed)

    def factory(epoch, start_batch):
        for indices in sampler.batches(epoch, start_batch):
            if order is not None:
                order.extend((epoch, i) for i in indices)
            yield collate_batch([data[i] for i in indices])

    factory.sampler_seed = opts.seed
    original = training._train_step
    done = load_training_state(opts.resume)["global_step"] if opts.resume else 0

    def step(*args, **kwargs):
        nonlocal done
        if draws is not None:
            draws.append((random.random(), float(np.random.random()), float(torch.rand(()))))
        result = original(*args, **kwargs)
        done += 1
        if done == stop_at:
            os.kill(os.getpid(), sig)
        return result

    with patch.object(training, "_train_step", step):
        summary = fit(opts, model, tokenizer, data, val, mode=mode, batch_factory=factory)
    return summary


@pytest.mark.parametrize("grad_accum,cuts,target", [(1, (1, 6), 9), (4, (1, 2), 7), (4, (2, 3), 7)])
def test_multiple_resume_matches_continuous(tmp_path, tiny_tokenizer, nano_cfg, small_data,
                                           grad_accum, cuts, target):
    base = options(tmp_path / "full", grad_accum=grad_accum, max_steps=target)
    full = new_model(nano_cfg, dropout=0.15)
    full_order, full_draws = [], []
    expected = run_segment(base, full, tiny_tokenizer, small_data,
                           order=full_order, draws=full_draws, val=small_data)
    expected_rng = training.rng_state()
    full_state = load_training_state(Path(base.out) / "state.pt")
    split = replace(base, out=str(tmp_path / "split"))
    split_order, split_draws = [], []
    model = new_model(nano_cfg, dropout=0.15)
    for index, cut in enumerate((*cuts, None)):
        actual = run_segment(split, model, tiny_tokenizer, small_data, stop_at=cut,
                             sig=signal.SIGINT if index == 0 else signal.SIGTERM,
                             order=split_order, draws=split_draws, val=small_data)
        if cut is not None:
            assert actual["steps"] == cut
            if grad_accum == 4 and cut == 2:
                assert (actual["epoch"], actual["micro_batches_consumed_in_epoch"]) == (1, 0)
            split = replace(split, resume=str(Path(split.out) / "state.pt"))
            model = new_model(nano_cfg, seed=999, dropout=0.15)  # resume must restore RNG
    assert full_order == split_order
    assert full_draws == split_draws
    for key in ("steps", "epoch", "micro_batches_consumed_in_epoch", "samples_seen", "tokens_seen"):
        assert actual[key] == expected[key]
    for name, value in full.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    resumed_state = load_training_state(Path(split.out) / "state.pt")
    for key, params in full_state["optimizer"]["state"].items():
        for name, value in params.items():
            torch.testing.assert_close(value, resumed_state["optimizer"]["state"][key][name], rtol=0, atol=0)
    assert torch.equal(expected_rng["torch"], training.rng_state()["torch"])
    def lrs(directory):
        return [r["learning_rate"] for line in (Path(directory) / "training-log.jsonl").read_text().splitlines()
                if (r := json.loads(line))["event"] == "train"]
    assert lrs(base.out) == lrs(split.out)
    assert actual["steps"] > actual["optimizer_updates_per_epoch"]  # epochs defaults to one
    # Replaying a completed resume must not execute, consume data, or rotate the checkpoint.
    before = (Path(split.out) / "state.pt").read_bytes()
    with patch.object(training, "_train_step", side_effect=AssertionError("extra update")):
        fit(split, model, tiny_tokenizer, small_data, small_data)
    assert (Path(split.out) / "state.pt").read_bytes() == before


def test_sft_resume_and_unequal_token_weighting(tmp_path, tiny_tokenizer, nano_cfg):
    path = tmp_path / "sft.jsonl"
    records = [{"messages": [{"role": "user", "content": f"问题{i}"},
                              {"role": "assistant", "content": "模型" * (i + 1)}]} for i in range(11)]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records))
    data = SFTDataset(path, tiny_tokenizer, 96)
    base = options(tmp_path / "full", max_steps=5)
    full = new_model(nano_cfg, dropout=0.1)
    order, draws = [], []
    expected = run_segment(base, full, tiny_tokenizer, data, order=order, draws=draws, mode="sft", val=data)
    split = replace(base, out=str(tmp_path / "split"))
    split_order, split_draws = [], []
    for cut in (1, 2, None):
        model = new_model(nano_cfg, dropout=0.1)
        actual = run_segment(split, model, tiny_tokenizer, data, stop_at=cut, order=split_order,
                             draws=split_draws, mode="sft", val=data)
        split = replace(split, resume=str(Path(split.out) / "state.pt"))
    assert order == split_order and draws == split_draws
    assert actual["tokens_seen"] == expected["tokens_seen"]
    for k, v in full.state_dict().items():
        torch.testing.assert_close(v, model.state_dict()[k], rtol=0, atol=0)
    # The production SFT factory uses the same resumable micro-batch semantics.
    factory = sft_batch_factory(data, 2, base.seed, 0, -100)
    all_batches = list(factory(0, 0))
    assert all(torch.equal(a["labels"], b["labels"]) for a, b in zip(all_batches[4:], factory(0, 4)))
    # Training log loss is weighted over actual effective tokens, including the short group.
    logged = [json.loads(s) for s in (Path(base.out) / "training-log.jsonl").read_text().splitlines()]
    assert sum(r["loss_tokens"] for r in logged if r["event"] == "train") == expected["tokens_seen"]


@pytest.fixture
def partial_run(tmp_path, tiny_tokenizer, nano_cfg, small_data):
    opts = options(tmp_path / "partial", max_steps=4)
    model = new_model(nano_cfg)
    run_segment(opts, model, tiny_tokenizer, small_data, stop_at=2)
    return replace(opts, resume=str(Path(opts.out) / "state.pt")), model


@pytest.mark.parametrize("change,match", [
    ({"batch_size": 1}, "batch_size"), ({"grad_accum": 1}, "grad_accum"),
    ({"seed": 20}, "seed"), ({"max_steps": 5}, "max_steps"),
    ({"lr": 0.002}, "learning_rate"), ({"min_lr": 0.0002}, "min_learning_rate"),
    ({"warmup_ratio": 0.1}, "warmup"), ({"weight_decay": 0.2}, "weight_decay"),
    ({"grad_clip": 0.5}, "grad_clip"), ({"precision": "bf16"}, "CPU"),
    ({"grad_checkpointing": True}, "grad_checkpointing"),
])
def test_resume_rejects_plan_changes(partial_run, tiny_tokenizer, small_data, change, match):
    opts, model = partial_run
    before = Path(opts.resume).read_bytes()
    with pytest.raises((CheckpointError, ValueError), match=match):
        fit(replace(opts, **change), model, tiny_tokenizer, small_data, None)
    assert Path(opts.resume).read_bytes() == before


def test_resume_checks_data_tokenizer_config_mode(partial_run, tiny_tokenizer, nano_cfg, small_data):
    opts, model = partial_run
    def reject(data=small_data, tok=tiny_tokenizer, current_model=model, mode="pretrain", match="manifest"):
        with pytest.raises(CheckpointError, match=match):
            fit(opts, current_model, tok, data, None, mode=mode)
    reject(data=PackedPretrainDataset(small_data.path, 11), match="seq_len")
    # fingerprint is derived from serialized tokenizer JSON; change its backend contents.
    from minillm.tokenizer import MiniLLMTokenizer
    from conftest import synthetic_texts
    changed_tok = MiniLLMTokenizer.train(synthetic_texts(seed=91), vocab_size=tiny_tokenizer.vocab_size)
    cfg = replace(nano_cfg, vocab_size=changed_tok.vocab_size)
    reject(tok=changed_tok, current_model=new_model(cfg), match="tokenizer")
    reject(current_model=new_model(replace(nano_cfg, rope_theta=20000)), match="model_config")
    reject(mode="sft", match="mode")
    original = small_data.path.read_bytes()
    changed = bytearray(original)
    changed[0] ^= 1
    small_data.path.write_bytes(changed)
    reject(match="sha256")  # identical path and length, different bytes
    small_data.path.write_bytes(original)


def test_safe_runtime_changes_and_validation_rng(partial_run, tiny_tokenizer, small_data):
    opts, model = partial_run
    rng = training.rng_state()
    training.evaluate(model, small_data, torch.device("cpu"), batch_size=2, max_batches=1)
    assert torch.equal(rng["torch"], training.rng_state()["torch"])
    summary = fit(replace(opts, log_every=99, save_every=0, val_every=1, val_limit=2, max_minutes=60),
                  model, tiny_tokenizer, small_data, None)
    assert summary["steps"] == 4


@pytest.mark.parametrize("field,value", [
    ("micro_batches_consumed_in_epoch", 99), ("micro_batches_consumed_in_epoch", 1),
    ("epoch", 99), ("global_step", 99), ("samples_seen", 99), ("tokens_seen", 99),
])
def test_invalid_progress_rejected(partial_run, tiny_tokenizer, small_data, field, value):
    opts, model = partial_run
    state = load_training_state(opts.resume)
    state[field] = value
    torch.save(state, opts.resume)
    with pytest.raises(CheckpointError, match="cursor|global_step|samples_seen|tokens_seen"):
        fit(opts, model, tiny_tokenizer, small_data, None)


def test_epoch_only_plan(tmp_path, tiny_tokenizer, nano_cfg, small_data):
    opts = options(tmp_path / "epochs", max_steps=0, epochs=2)
    result = fit(opts, new_model(nano_cfg), tiny_tokenizer, small_data, None)
    assert (result["steps"], result["epoch"], result["micro_batches_consumed_in_epoch"]) == (4, 2, 0)
    assert result["samples_seen"] == 22 and result["tokens_seen"] == 22 * 11


def test_time_budget_saves_complete_update(tmp_path, tiny_tokenizer, nano_cfg, small_data):
    opts = options(tmp_path / "budget", max_minutes=0.001)
    original = training._train_step
    def step(*args):
        result = original(*args)
        # Advance the timer once one complete update is available (no sleeps).
        clock[0] += 120
        return result
    clock = [0.0]
    with patch.object(training, "_train_step", step), patch.object(training.time, "monotonic", lambda: clock[0]):
        result = fit(opts, new_model(nano_cfg), tiny_tokenizer, small_data, None)
    assert result["steps"] == 1 and result["stop_reason"] == "time_budget"
    assert load_training_state(Path(opts.out) / "state.pt")["micro_batches_consumed_in_epoch"] == 4


def test_seed_before_initialization(nano_cfg):
    a, b, c = new_model(nano_cfg, 1), new_model(nano_cfg, 1), new_model(nano_cfg, 2)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    assert any(not torch.equal(a.state_dict()[k], c.state_dict()[k]) for k in a.state_dict())
    set_seed(12)
    before = (random.random(), np.random.random(), torch.rand(()))
    set_seed(12)
    after = (random.random(), np.random.random(), torch.rand(()))
    assert before == after


def test_legacy_and_incomplete_state_rejected(tmp_path):
    path = tmp_path / "state.pt"
    torch.save({"architecture": "minillm", "arch_version": 2, "step": 2, "batch_in_epoch": 4}, path)
    with pytest.raises(CheckpointError, match="ambiguous.*cursor"):
        load_training_state(path)
    torch.save({"architecture": "minillm", "arch_version": 2, "training_state_version": 2}, path)
    with pytest.raises(CheckpointError, match="missing/invalid"):
        load_training_state(path)


@pytest.mark.parametrize("failure", ["serialize", "fsync", "rotate", "replace"])
def test_atomic_save_failure_preserves_files(tmp_path, failure):
    path, prev = tmp_path / "state.pt", tmp_path / "state.prev.pt"
    checkpoint.atomic_torch_save({"value": 1}, path)
    checkpoint.atomic_torch_save({"value": 2}, path, backup=prev)
    error = OSError(errno.ENOSPC, "disk full")
    original_replace = os.replace
    def replace_file(source, dest):
        if (failure == "rotate" and Path(dest) == prev) or (failure == "replace" and Path(dest) == path):
            raise error
        return original_replace(source, dest)
    def serialize(payload, handle):
        handle.write(b"partial write")
        raise error
    target, effect = {
        "serialize": ("torch.save", serialize), "fsync": ("os.fsync", error),
        "rotate": ("os.replace", replace_file), "replace": ("os.replace", replace_file),
    }[failure]
    with patch(target, side_effect=effect), pytest.raises(OSError, match="disk full"):
        checkpoint.atomic_torch_save({"value": 3}, path, backup=prev)
    assert torch.load(path, weights_only=False)["value"] == 2
    assert torch.load(prev, weights_only=False)["value"] in (1, 2)
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_backup_link_fallback(tmp_path):
    path, prev = tmp_path / "state.pt", tmp_path / "state.prev.pt"
    checkpoint.atomic_torch_save({"value": 1}, path)
    with patch("os.link", side_effect=OSError("links unavailable")):
        checkpoint.atomic_torch_save({"value": 2}, path, backup=prev)
    assert torch.load(prev, weights_only=False)["value"] == 1
    assert torch.load(path, weights_only=False)["value"] == 2


def test_corrupt_primary_explicit_backup_and_both_invalid(partial_run, tiny_tokenizer, small_data):
    opts, model = partial_run
    primary = Path(opts.resume)
    backup = primary.with_name("state.prev.pt")
    assert load_training_state(backup)["global_step"] == 1
    primary.write_bytes(b"corrupt zip")
    with pytest.raises(CheckpointError, match="explicitly resume"):
        load_training_state(primary)
    # Save while resuming a backup must never rotate a corrupt primary over it.
    previous = backup.read_bytes()
    result = run_segment(replace(opts, resume=str(backup), save_every=0), model,
                         tiny_tokenizer, small_data, stop_at=2)
    assert result["steps"] == 2
    assert backup.read_bytes() == previous
    assert load_training_state(primary)["global_step"] == 2
    for path in (primary, backup):
        path.write_bytes(b"broken")
        with pytest.raises(CheckpointError, match="cannot load"):
            load_training_state(path)
    temp = primary.with_name(".state.pt.orphan.tmp")
    torch.save({"looks": "complete"}, temp)
    with pytest.raises(CheckpointError, match="temporary"):
        load_training_state(temp)


@pytest.mark.parametrize("kind", ["loss", "gradient", "lr", "oom", "save"])
def test_failed_update_never_overwrites_valid_state(partial_run, tiny_tokenizer, small_data, kind, capsys):
    opts, model = partial_run
    opts = replace(opts, val_every=1)
    primary = Path(opts.resume)
    before = primary.read_bytes()
    best = primary.with_name("best.pt")
    best_before = best.read_bytes() if best.exists() else None
    handles = []
    if kind == "loss":
        def bad_loss(module, args, out):
            out.loss_sum = out.loss_sum * float("nan")
        handles.append(model.register_forward_hook(bad_loss))
    if kind == "gradient":
        handles.append(next(model.parameters()).register_hook(lambda grad: grad * float("inf")))
    target, effect = {
        "loss": ("lr_at", training.lr_at), "gradient": ("lr_at", training.lr_at),
        "lr": ("lr_at", lambda *args: float("nan")),
        "oom": ("_train_step", RuntimeError("CUDA out of memory")),
        "save": ("save_training_state", OSError(errno.ENOSPC, "disk full")),
    }[kind]
    try:
        with patch.object(training, target, side_effect=effect), pytest.raises((RuntimeError, OSError)):
            fit(opts, model, tiny_tokenizer, small_data, None)
    finally:
        for handle in handles:
            handle.remove()
    assert primary.read_bytes() == before
    if best_before is not None:
        assert best.read_bytes() == best_before
    output = capsys.readouterr().out
    assert "[error]" in output
    if kind == "save":
        assert "FAILED" in output


def test_amp_skipped_update_does_not_commit_counters(partial_run, tiny_tokenizer, small_data):
    opts, model = partial_run
    before = Path(opts.resume).read_bytes()
    actual_step = training._train_step
    # Real CPU GradScaler exercises unscale and overflow detection without CUDA.
    scaler = torch.amp.GradScaler("cpu", init_scale=1e38)
    def overflowing_step(model, group, optimizer, ignored_scaler, device, dtype, opts, lr):
        return actual_step(model, group, optimizer, scaler, device, dtype, opts, lr)
    handle = next(model.parameters()).register_hook(lambda grad: grad * float("inf"))
    try:
        with patch.object(training, "_train_step", overflowing_step), pytest.raises(training.AMPOverflowError, match="skipped.*global_step unchanged"):
            fit(opts, model, tiny_tokenizer, small_data, None)
    finally:
        handle.remove()
    assert Path(opts.resume).read_bytes() == before
    assert load_training_state(opts.resume)["global_step"] == 2


def test_scaler_scale_decrease_detected(nano_cfg, tiny_tokenizer, small_data):
    model = new_model(nano_cfg)
    optimizer = torch.optim.AdamW(model.parameters())
    class SkippingScaler:
        scale_value = 65536
        def is_enabled(self): return True
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def get_scale(self): return self.scale_value
        def step(self, optimizer): pass
        def update(self): self.scale_value /= 2
    group = [collate_batch([small_data[0]])]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(training.AMPOverflowError, match="GradScaler skipped"):
        training._train_step(model, group, optimizer, SkippingScaler(), torch.device("cpu"),
                             None, options("unused"), 0.001)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_log_loss_is_token_weighted_across_unequal_groups(tmp_path, tiny_tokenizer, nano_cfg, small_data):
    opts = options(tmp_path / "weighted", max_steps=2, log_every=99, save_every=0, val_every=0)
    results = []
    original = training._train_step
    def step(*args):
        result = original(*args)
        results.append(result)
        return result
    with patch.object(training, "_train_step", step):
        fit(opts, new_model(nano_cfg), tiny_tokenizer, small_data, None)
    records = [json.loads(s) for s in (Path(opts.out) / "training-log.jsonl").read_text().splitlines()]
    log = next(r for r in records if r["event"] == "train")
    assert [r.tokens for r in results] == [88, 33]
    assert log["loss"] == pytest.approx(sum(r.loss_sum for r in results) / 121)
    assert log["loss"] != pytest.approx(sum(r.loss_sum / r.tokens for r in results) / 2, abs=1e-5)


def test_nonfinite_validation_does_not_replace_best(tmp_path, tiny_tokenizer, nano_cfg, small_data):
    opts = options(tmp_path / "validation-failure", val_every=0)
    model = new_model(nano_cfg)
    run_segment(opts, model, tiny_tokenizer, small_data, stop_at=1, val=small_data)
    opts = replace(opts, resume=str(Path(opts.out) / "state.pt"), val_every=1)
    path = Path(opts.out) / "best.pt"
    path.write_bytes(b"last good best model")
    before = Path(opts.resume).read_bytes()
    def bad_validation(module, args, out):
        if not module.training:
            out.loss_sum = out.loss_sum * float("nan")
    handle = model.register_forward_hook(bad_validation)
    try:
        with pytest.raises(training.NumericalError, match="validation loss"):
            fit(opts, model, tiny_tokenizer, small_data, small_data)
    finally:
        handle.remove()
    assert path.read_bytes() == b"last good best model"
    assert Path(opts.resume).read_bytes() == before


@pytest.mark.parametrize("mode", ["pretrain", "sft"])
def test_cli_seeds_before_model_creation(tmp_path, tiny_tokenizer, small_data, mode):
    import minillm.sft as sft
    module = training if mode == "pretrain" else sft
    tok = tmp_path / "tokenizer.json"
    tiny_tokenizer.save(tok)
    if mode == "sft":
        data = tmp_path / "sft.jsonl"
        data.write_text(json.dumps({"messages": [{"role": "user", "content": "问题"},
                                                 {"role": "assistant", "content": "答案"}]}))
        length = ["--max-seq-len", "96"]
    else:
        data, length = small_data.path, ["--seq-len", "12"]
    weights = []
    def capture(opts, model, *args, **kwargs):
        weights.append({k: v.clone() for k, v in model.state_dict().items()})
        return {}
    for seed in (4, 4, 5):
        argv = ["minillm", "--config", "minillm-nano", "--data", str(data),
                "--tokenizer", str(tok), "--seed", str(seed), "--device", "cpu", *length]
        with patch("sys.argv", argv), patch.object(module, "fit", capture):
            module.main()
    assert all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])
    assert any(not torch.equal(weights[0][k], weights[2][k]) for k in weights[0])


def test_missing_adamw_state_rejected(partial_run):
    opts, _ = partial_run
    state = load_training_state(opts.resume)
    state["optimizer"]["state"] = {}
    torch.save(state, opts.resume)
    with pytest.raises(CheckpointError, match="missing AdamW"):
        load_training_state(opts.resume)


def test_backup_resume_retains_chosen_backup_on_failed_replace(partial_run, tiny_tokenizer, small_data):
    opts, model = partial_run
    primary, backup = Path(opts.resume), Path(opts.out) / "state.prev.pt"
    previous = backup.read_bytes()
    # Primary still deserializes, but its cursor is unusable. Never rotate it
    # over the explicitly selected valid backup while recovering.
    payload = load_training_state(primary)
    payload["micro_batches_consumed_in_epoch"] = 999
    torch.save(payload, primary)
    original_replace = os.replace
    def fail_primary(source, dest):
        if Path(dest) == primary:
            raise OSError(errno.ENOSPC, "disk full")
        return original_replace(source, dest)
    with patch("os.replace", side_effect=fail_primary), pytest.raises(OSError, match="disk full"):
        run_segment(replace(opts, resume=str(backup), save_every=0), model,
                    tiny_tokenizer, small_data, stop_at=2)
    assert backup.read_bytes() == previous
    assert load_training_state(backup)["global_step"] == 1
    assert not list(primary.parent.glob(".*.tmp"))


def snapshot(directory):
    return {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize("init_from", [None, "previous-model.pt"])
def test_fresh_run_cannot_overwrite_existing_training(partial_run, tiny_tokenizer, small_data, init_from):
    opts, model = partial_run
    out = Path(opts.out)
    before = snapshot(out)
    with patch.object(training, "_train_step", side_effect=AssertionError("must not train")):
        with pytest.raises(CheckpointError, match="non-empty.*output directory"):
            fit(replace(opts, resume=None, init_from=init_from), model, tiny_tokenizer, small_data, None)
    assert snapshot(out) == before


@pytest.mark.parametrize("artifact", [
    "training-manifest.json", "state.pt", "state.prev.pt", "model.pt", "best.pt",
    "training-log.jsonl", "tokenizer.json", "config.json",
])
def test_fresh_run_refuses_individual_artifacts(tmp_path, tiny_tokenizer, nano_cfg, small_data, artifact):
    out = tmp_path / "occupied"
    out.mkdir()
    (out / artifact).write_bytes(b"existing artifact; do not replace")
    before = snapshot(out)
    with pytest.raises(CheckpointError, match="non-empty.*output directory"):
        fit(options(out), new_model(nano_cfg), tiny_tokenizer, small_data, None)
    assert snapshot(out) == before


def test_resume_cannot_mix_independent_runs_with_identical_configuration(
    partial_run, tiny_tokenizer, nano_cfg, small_data, tmp_path,
):
    source, _ = partial_run
    other = replace(source, out=str(tmp_path / "other-run"), resume=None)
    run_segment(other, new_model(nano_cfg), tiny_tokenizer, small_data, stop_at=1)
    source_id = load_training_state(source.resume)["manifest"]["run_id"]
    target_id = load_training_state(Path(other.out) / "state.pt")["manifest"]["run_id"]
    assert source_id != target_id
    source_before, target_before = snapshot(Path(source.out)), snapshot(Path(other.out))
    with pytest.raises(CheckpointError, match="another training run"):
        fit(replace(source, out=other.out), new_model(nano_cfg), tiny_tokenizer, small_data, None)
    assert snapshot(Path(source.out)) == source_before
    assert snapshot(Path(other.out)) == target_before


@pytest.mark.parametrize("manifest_contents", [None, "broken JSON", "[]", "{}", '{"run_id": "another-run"}'])
def test_resume_refuses_unverifiable_output_directory(
    partial_run, tiny_tokenizer, nano_cfg, small_data, tmp_path, manifest_contents,
):
    opts, _ = partial_run
    out = tmp_path / "unverified"
    out.mkdir()
    (out / "training-log.jsonl").write_text("existing log\n")
    if manifest_contents is not None:
        (out / "training-manifest.json").write_text(manifest_contents)
    before = snapshot(out)
    with pytest.raises(CheckpointError, match="ownership|another training run"):
        fit(replace(opts, out=str(out)), new_model(nano_cfg), tiny_tokenizer, small_data, None)
    assert snapshot(out) == before


def test_resume_refuses_changed_destination_manifest(partial_run, tiny_tokenizer, small_data):
    opts, model = partial_run
    out = Path(opts.out)
    path = out / "training-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["training"]["learning_rate"] *= 2
    path.write_text(json.dumps(manifest))
    before = snapshot(out)
    with pytest.raises(CheckpointError, match="learning_rate"):
        fit(opts, model, tiny_tokenizer, small_data, None)
    assert snapshot(out) == before


def test_resume_into_empty_directory_inherits_run_identity(partial_run, tiny_tokenizer, nano_cfg, small_data, tmp_path):
    opts, _ = partial_run
    source_before = snapshot(Path(opts.out))
    resumed = replace(opts, out=str(tmp_path / "continued"))
    Path(resumed.out).mkdir()
    summary = fit(resumed, new_model(nano_cfg), tiny_tokenizer, small_data, None)
    assert summary["steps"] == 4
    manifest = json.loads((Path(resumed.out) / "training-manifest.json").read_text())
    assert manifest["run_id"] == load_training_state(opts.resume)["manifest"]["run_id"]
    assert load_training_state(Path(resumed.out) / "state.pt")["manifest"] == manifest
    assert snapshot(Path(opts.out)) == source_before


def test_early_v2_without_run_identity_requires_empty_destination(
    partial_run, tiny_tokenizer, nano_cfg, small_data, tmp_path,
):
    opts, _ = partial_run
    state = load_training_state(opts.resume)
    del state["manifest"]["run_id"]
    torch.save(state, opts.resume)
    before = snapshot(Path(opts.out))
    with pytest.raises(CheckpointError, match="no run_id.*ownership"):
        fit(opts, new_model(nano_cfg), tiny_tokenizer, small_data, None)
    assert snapshot(Path(opts.out)) == before
    migrated = replace(opts, out=str(tmp_path / "migrated"))
    summary = fit(migrated, new_model(nano_cfg), tiny_tokenizer, small_data, None)
    assert summary["steps"] == 4
    manifest = json.loads((Path(migrated.out) / "training-manifest.json").read_text())
    assert manifest["run_id"]
    assert snapshot(Path(opts.out)) == before



def test_fresh_run_accepts_existing_empty_directory(tmp_path, tiny_tokenizer, nano_cfg, small_data):
    out = tmp_path / "empty"
    out.mkdir()
    result = fit(options(out, max_steps=1), new_model(nano_cfg), tiny_tokenizer, small_data, None)
    assert result["steps"] == 1
    manifest = json.loads((out / "training-manifest.json").read_text())
    assert manifest["run_id"] == load_training_state(out / "state.pt")["manifest"]["run_id"]


@pytest.mark.parametrize("final_loss,expected_best_step", [(0.5, 3), (2.0, 2)])
def test_final_validation_tracks_best_weights_and_saved_metrics(
    tmp_path, tiny_tokenizer, nano_cfg, small_data, final_loss, expected_best_step,
):
    opts = options(tmp_path / "final-validation", max_steps=3, val_every=2)
    model = new_model(nano_cfg)
    weights = {}
    losses = iter([(2, 1.0), (3, final_loss)])
    def validation(current, *args, **kwargs):
        step, loss = next(losses)
        weights[step] = {k: v.clone() for k, v in current.state_dict().items()}
        return {"loss": loss, "perplexity": float(np.exp(loss)), "tokens": 22, "batches": 1}
    with patch.object(training, "evaluate", side_effect=validation) as evaluate:
        summary = fit(opts, model, tiny_tokenizer, small_data, small_data)
    assert evaluate.call_count == 2
    best = torch.load(Path(opts.out) / "best.pt", weights_only=False)
    assert best["step"] == expected_best_step
    assert best["val_loss"] == min(1.0, final_loss)
    for name, value in weights[expected_best_step].items():
        torch.testing.assert_close(best["model"][name], value, rtol=0, atol=0)
    assert any(not torch.equal(weights[2][k], weights[3][k]) for k in weights[2])
    state = load_training_state(Path(opts.out) / "state.pt")
    saved_summary = json.loads((Path(opts.out) / "train-summary.json").read_text())
    metrics = json.loads((Path(opts.out) / "metrics.json").read_text())
    assert state["best_loss"] == summary["best_val_loss"] == saved_summary["best_val_loss"] == min(1.0, final_loss)
    assert summary["final_val"]["loss"] == final_loss
    assert [(r["step"], r["loss"]) for r in state["history"]] == [(2, 1.0), (3, final_loss)]
    assert metrics == state["history"]
    # Refreshing final metrics must not rotate the just-saved step 3 over step 2.
    assert load_training_state(Path(opts.out) / "state.prev.pt")["global_step"] == 2


@pytest.mark.parametrize("val_every", [0, 100])
def test_final_validation_creates_first_best_checkpoint(
    tmp_path, tiny_tokenizer, nano_cfg, small_data, val_every,
):
    opts = options(tmp_path / "only-final-validation", max_steps=2, val_every=val_every)
    model = new_model(nano_cfg)
    summary = fit(opts, model, tiny_tokenizer, small_data, small_data)
    best = torch.load(Path(opts.out) / "best.pt", weights_only=False)
    assert best["step"] == 2
    assert best["val_loss"] == summary["final_val"]["loss"] == summary["best_val_loss"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(best["model"][name], value, rtol=0, atol=0)
    state = load_training_state(Path(opts.out) / "state.pt")
    assert state["best_loss"] == best["val_loss"]
    assert len(state["history"]) == 1 and state["history"][0]["step"] == 2


def test_final_validation_reuses_periodic_result(tmp_path, tiny_tokenizer, nano_cfg, small_data):
    opts = options(tmp_path / "periodic-final-validation", max_steps=2, val_every=2)
    with patch.object(training, "evaluate", wraps=training.evaluate) as evaluate:
        summary = fit(opts, new_model(nano_cfg), tiny_tokenizer, small_data, small_data)
    assert evaluate.call_count == 1
    state = load_training_state(Path(opts.out) / "state.pt")
    assert state["best_loss"] == summary["best_val_loss"] == summary["final_val"]["loss"]
    assert len(state["history"]) == 1


def test_final_validation_failure_preserves_completed_update_and_previous_best(
    tmp_path, tiny_tokenizer, nano_cfg, small_data,
):
    opts = options(tmp_path / "failed-final-validation", max_steps=3, val_every=2)
    model = new_model(nano_cfg)
    actual_evaluate = training.evaluate
    previous_best = []
    def validation(*args, **kwargs):
        if not previous_best:
            previous_best.append({k: v.clone() for k, v in model.state_dict().items()})
            return actual_evaluate(*args, **kwargs)
        raise training.NumericalError("non-finite final validation loss")
    with patch.object(training, "evaluate", side_effect=validation), pytest.raises(training.NumericalError):
        fit(opts, model, tiny_tokenizer, small_data, small_data)
    state = load_training_state(Path(opts.out) / "state.pt")
    best = torch.load(Path(opts.out) / "best.pt", weights_only=False)
    assert state["global_step"] == 3 and best["step"] == 2
    assert state["best_loss"] == best["val_loss"]
    for name, value in previous_best[0].items():
        torch.testing.assert_close(best["model"][name], value, rtol=0, atol=0)
