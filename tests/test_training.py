"""Training pipeline: packed data, batching, LR schedule, checkpoints, resume."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from minillm.config import ARCH_VERSION, MINILLM_87M, MINILLM_NANO
from minillm.data import (
    EpochBatchSampler,
    PackedPretrainDataset,
    collate_batch,
    index_jsonl,
)
from minillm.model import MiniLLM
from minillm.train import (
    CheckpointError,
    TrainingOptions,
    evaluate,
    fit,
    load_model_checkpoint,
    lr_at,
)
from minillm.tokenizer import EOS_ID



def test_packed_dataset_yields_next_token_pairs(packed_data, tiny_tokenizer):
    train, _ = packed_data
    dataset = PackedPretrainDataset(train, 64)
    assert len(dataset) == dataset.num_samples
    item = dataset[0]
    assert item["input_ids"].shape == (64,) and item["labels"].shape == (64,)
    assert torch.equal(item["input_ids"], item["labels"])
    raw = np.memmap(train, dtype=np.uint16, mode="r")
    assert torch.equal(item["input_ids"], torch.tensor(np.asarray(raw[:64]).astype(np.int64)))
    # the window target is the following token
    assert int(raw[64]) >= 0
    assert (np.asarray(raw) == EOS_ID).sum() == 80  # one EOS per document


def test_packed_dataset_rejects_short_file(tmp_path):
    path = tmp_path / "short.bin"
    np.asarray([1, 2, 3], dtype=np.uint16).tofile(path)
    with pytest.raises(ValueError, match="fewer than"):
        PackedPretrainDataset(path, 16)


def test_epoch_batch_sampler_is_deterministic_and_resumable():
    sampler = EpochBatchSampler(10, 3, seed=5)
    assert list(sampler.batches(0)) == list(sampler.batches(0))
    assert list(sampler.batches(0)) != list(sampler.batches(1))
    batches = list(sampler.batches(0))
    assert sorted(index for batch in batches for index in batch) == list(range(10))
    assert all(len(batch) <= 3 for batch in batches)
    assert len(sampler) == 4


def test_collate_pads_and_masks():
    batch = [
        {"input_ids": torch.tensor([4, 5, 6]), "labels": torch.tensor([4, 5, 6])},
        {"input_ids": torch.tensor([7]), "labels": torch.tensor([-100])},
    ]
    out = collate_batch(batch, pad_token_id=0, ignore_index=-100)
    assert out["input_ids"].shape == (2, 3)
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert out["labels"][1].tolist() == [-100, -100, -100]
    assert out["input_ids"][1].tolist() == [7, 0, 0]


def test_lr_schedule():
    assert lr_at(0, 100, 1e-3, 1e-5, 10) == pytest.approx(1e-4)
    assert lr_at(9, 100, 1e-3, 1e-5, 10) == pytest.approx(1e-3)
    assert lr_at(100, 100, 1e-3, 1e-5, 10) == pytest.approx(1e-5, rel=1e-6)
    values = [lr_at(step, 100, 1e-3, 1e-5, 10) for step in range(10, 100)]
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:])), "cosine must decrease"


def test_index_jsonl_counts_non_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "a"}\n\n{"text": "b"}\n{"text": "c"}\n', encoding="utf-8")
    offsets = index_jsonl(path)
    assert len(offsets) == 3
    assert offsets[0] == 0


def test_vlm_checkpoint_is_rejected_with_explanation(tmp_path):
    path = tmp_path / "old-vlm.pt"
    torch.save({"config": {"hidden_size": 512}, "model": {}}, path)
    with pytest.raises(CheckpointError, match="TinyVLM"):
        load_model_checkpoint(path, torch.device("cpu"))


def test_fit_trains_saves_and_resumes(tmp_path, tiny_tokenizer, nano_cfg, packed_data):
    train_path, val_path = packed_data
    train_set = PackedPretrainDataset(train_path, 64)
    val_set = PackedPretrainDataset(val_path, 64)
    out = tmp_path / "run"
    base = dict(
        out=str(out), epochs=1, batch_size=2, grad_accum=2, lr=1e-3, min_lr=1e-4,
        warmup_ratio=0.1, save_every=2, val_every=2, val_limit=1, log_every=1,
        seed=0, device="cpu", precision="fp32", threads=2,
    )
    model = MiniLLM(nano_cfg)
    # A completed plan can be resumed idempotently; extending it is explicit new work.
    summary = fit(TrainingOptions(**base, max_steps=2), model, tiny_tokenizer, train_set, val_set)
    assert summary["steps"] == 2
    for name in ("model.pt", "state.pt", "tokenizer.json", "config.json",
                 "train-summary.json", "metrics.json"):
        assert (out / name).exists(), name
    payload = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    assert payload["architecture"] == "minillm" and payload["arch_version"] == ARCH_VERSION
    assert payload["step"] == 2 and payload["tokenizer_fingerprint"] == tiny_tokenizer.fingerprint
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics and math.isfinite(metrics[0]["loss"])

    resumed = MiniLLM(nano_cfg)
    opts = TrainingOptions(**{**base, "epochs": 2, "max_steps": 2, "resume": str(out / "state.pt")})
    summary2 = fit(opts, resumed, tiny_tokenizer, train_set, val_set)
    assert summary2["steps"] == 2, "an already completed plan must execute no extra update"
    final = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    assert final["step"] == 2
    assert final["tokens_seen"] == payload["tokens_seen"]


def test_fit_rejects_tokenizer_mismatch(tmp_path, tiny_tokenizer, packed_data):
    train_path, _ = packed_data
    from minillm.config import MINILLM_NANO

    cfg = MINILLM_NANO.copy()
    cfg.vocab_size = tiny_tokenizer.vocab_size + 1
    with pytest.raises(ValueError, match="vocab_size"):
        fit(
            TrainingOptions(out=str(tmp_path / "x"), max_steps=1, batch_size=2, grad_accum=1,
                            log_every=1, device="cpu", precision="fp32", threads=1),
            MiniLLM(cfg), tiny_tokenizer, PackedPretrainDataset(train_path, 64), None,
        )


def test_evaluate_returns_finite_perplexity(nano_cfg, packed_data):
    _, val_path = packed_data
    model = MiniLLM(nano_cfg)
    result = evaluate(model, PackedPretrainDataset(val_path, 64), torch.device("cpu"),
                      batch_size=2, max_batches=2)
    assert result["loss"] > 0 and math.isfinite(result["perplexity"])
    assert result["tokens"] == 2 * 2 * 63
