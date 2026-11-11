#!/usr/bin/env python3
"""CPU nano acceptance: continuous vs two resumes, backup and manifest failures.

Run with python scripts/check_training_reliability.py [--out /tmp/report-dir].
No downloaded corpus, GPU, pytest, or pre-existing tokenizer is required.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import signal
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

import minillm.train as training
from minillm.config import MINILLM_NANO
from minillm.data import EpochBatchSampler, PackedPretrainDataset, collate_batch
from minillm.tokenizer import MiniLLMTokenizer
from minillm.train import CheckpointError, TrainingOptions, fit, load_training_state
from minillm.utils import set_seed


def check(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1)
    texts = [" ".join(str(rng.randrange(10000)) for _ in range(20)) for _ in range(1000)]
    tokenizer = MiniLLMTokenizer.train(texts, vocab_size=512)
    cfg = replace(MINILLM_NANO, vocab_size=tokenizer.vocab_size,
                  hidden_dropout=0.15, attention_dropout=0.1)
    path = directory / "train.bin"
    (np.arange(11 * 12 + 1) % (tokenizer.vocab_size - 16) + 16).astype(np.uint16).tofile(path)
    data = PackedPretrainDataset(path, 12)
    base = TrainingOptions(out=str(directory / "continuous"), device="cpu", precision="fp32",
                           threads=1, seed=17, batch_size=2, grad_accum=4, max_steps=7,
                           warmup_ratio=0.4, log_every=1, save_every=1, val_every=0)
    def model():
        set_seed(base.seed)
        return training.MiniLLM(cfg)
    def segment(opts, current, stop, order, draws):
        sampler = EpochBatchSampler(len(data), opts.batch_size, opts.seed)
        def factory(epoch, cursor):
            for indices in sampler.batches(epoch, cursor):
                order.extend((epoch, i) for i in indices)
                yield collate_batch([data[i] for i in indices])
        factory.sampler_seed = opts.seed
        original = training._train_step
        count = load_training_state(opts.resume)["global_step"] if opts.resume else 0
        def step(*args, **kwargs):
            nonlocal count
            draws.append((random.random(), float(np.random.random()), float(torch.rand(()))))
            result = original(*args, **kwargs)
            count += 1
            if count == stop:
                os.kill(os.getpid(), signal.SIGTERM if stop == 2 else signal.SIGINT)
            return result
        with patch.object(training, "_train_step", step):
            return fit(opts, current, tokenizer, data, None, batch_factory=factory)

    full, full_order, full_draws = model(), [], []
    continuous = segment(base, full, None, full_order, full_draws)
    split = replace(base, out=str(directory / "resumed"))
    split_order, split_draws, checkpoints = [], [], []
    for stop in (1, 2, None):
        current = model()
        resumed = segment(split, current, stop, split_order, split_draws)
        checkpoints.append({k: resumed[k] for k in ("steps", "epoch", "micro_batches_consumed_in_epoch")})
        split = replace(split, resume=str(Path(split.out) / "state.pt"))
    max_diff = max(float((v - current.state_dict()[k]).abs().max()) for k, v in full.state_dict().items())
    assert max_diff == 0.0, f"parameter difference {max_diff}"
    assert full_order == split_order, "sample consumption differs"
    assert full_draws == split_draws, "RNG draws differ"
    keys = ("steps", "epoch", "micro_batches_consumed_in_epoch", "samples_seen", "tokens_seen")
    assert all(continuous[k] == resumed[k] for k in keys), "progress counters differ"
    def lrs(out):
        return [r["learning_rate"] for line in (Path(out) / "training-log.jsonl").read_text().splitlines()
                if (r := json.loads(line))["event"] == "train"]
    assert lrs(base.out) == lrs(split.out), "LR sequence differs"
    assert resumed["steps"] == 7 > resumed["optimizer_updates_per_epoch"] == 2
    with patch.object(training, "_train_step", side_effect=AssertionError("completed step repeated")):
        fit(split, current, tokenizer, data, None)
    try:
        fit(replace(split, grad_accum=2), current, tokenizer, data, None)
    except CheckpointError:
        pass
    else:
        raise AssertionError("changed configuration accepted")
    raw = path.read_bytes()
    modified = bytearray(raw)
    modified[0] ^= 1
    path.write_bytes(modified)
    try:
        fit(split, current, tokenizer, data, None)
    except CheckpointError:
        pass
    else:
        raise AssertionError("changed dataset accepted")
    finally:
        path.write_bytes(raw)
    primary = Path(split.resume)
    backup = primary.with_name("state.prev.pt")
    primary.write_bytes(b"interrupted/corrupt checkpoint")
    try:
        load_training_state(primary)
    except CheckpointError:
        pass
    else:
        raise AssertionError("corrupt primary accepted")
    backup_step = load_training_state(backup)["global_step"]
    recovered_model = model()
    recovered = segment(replace(split, resume=str(backup)), recovered_model, None, [], [])
    assert recovered["steps"] == 7
    for k, v in full.state_dict().items():
        torch.testing.assert_close(v, recovered_model.state_dict()[k], rtol=0, atol=0)
    for item in (primary, backup):
        item.write_bytes(b"corrupt")
        try:
            load_training_state(item)
        except CheckpointError:
            pass
        else:
            raise AssertionError("invalid checkpoint accepted")
    return {
        "continuous": {k: continuous[k] for k in keys}, "two_resumes": {k: resumed[k] for k in keys},
        "interruptions_and_final": checkpoints, "max_parameter_difference": max_diff,
        "sample_order_equal": True, "rng_equal": True, "lr_equal": True,
        "updates_per_epoch": 2, "target_updates": 7, "backup_recovery_step": backup_step,
        "checks": ["continuous_vs_two_resumes", "SIGINT_and_SIGTERM", "incomplete_group_and_epoch_boundary",
                   "max_steps_beyond_one_epoch", "completed_resume_no_extra_step", "configuration_rejected",
                   "dataset_hash_rejected", "corrupt_primary_explicit_backup", "both_invalid_rejected"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="keep test artifacts and JSON report here")
    args = parser.parse_args()
    with contextlib.ExitStack() as stack:
        directory = args.out or Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="minillm-reliability-")))
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                report = check(directory)
        except Exception as exc:
            print(captured.getvalue())
            print(f"FAIL training reliability: {exc}", file=sys.stderr)
            return 1
        for name in report["checks"]:
            print(f"PASS {name}")
        print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
        if args.out:
            (directory / "reliability-report.json").write_text(json.dumps(report, indent=2))
            (directory / "reliability.log").write_text(captured.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
