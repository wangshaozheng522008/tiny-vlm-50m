"""SFT data: assistant-only loss, safe truncation, dedupe, and a real training step."""

from __future__ import annotations

import json

import pytest
import torch

from minillm.config import MINILLM_NANO
from minillm.data import SFTDataset, collate_batch
from minillm.model import MiniLLM
from minillm.sft import sft_batch_factory
from minillm.tokenizer import END_OF_TURN_ID, PAD_ID, USER_ID, MiniLLMTokenizer
from minillm.train import TrainingOptions, fit


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


@pytest.fixture()
def sft_records():
    return [
        {"messages": [
            {"role": "system", "content": "你是一个中文助手。"},
            {"role": "user", "content": "什么是机器学习？"},
            {"role": "assistant", "content": "机器学习是人工智能的一个分支，它让模型从数据中学习规律。"},
        ]},
        {"messages": [
            {"role": "user", "content": "写一个加法函数。"},
            {"role": "assistant", "content": "def add(a, b):\n    return a + b"},
            {"role": "user", "content": "解释一下这个函数。"},
            {"role": "assistant", "content": "它接收两个参数并返回它们的和。"},
        ]},
    ]


def test_sft_labels_only_on_assistant_content(tmp_path, tiny_tokenizer, sft_records):
    path = write_jsonl(tmp_path / "sft.jsonl", sft_records)
    dataset = SFTDataset(path, tiny_tokenizer, 256)
    assert len(dataset) == 2
    item = dataset[0]
    ids, labels = item["input_ids"].tolist(), item["labels"].tolist()
    supervised = tiny_tokenizer.decode([t for t, l in zip(ids, labels) if l != -100], skip_special_tokens=True)
    assert "机器学习是人工智能的一个分支" in supervised
    assert "什么是机器学习" not in supervised, "user turn must not be supervised"
    assert "你是一个中文助手" not in supervised, "system prompt must not be supervised"
    assert labels[-2] == END_OF_TURN_ID and ids[-2] == END_OF_TURN_ID, (
        "the closing marker is supervised with its real id, not PAD (0)"
    )
    for token, label in zip(ids, labels):
        assert label in (-100, token), "labels must be real target ids, never offsets"


def test_sft_last_turn_only(tmp_path, tiny_tokenizer, sft_records):
    path = write_jsonl(tmp_path / "sft.jsonl", sft_records)
    all_turns = SFTDataset(path, tiny_tokenizer, 256, supervise_all_assistant_turns=True)
    last_only = SFTDataset(path, tiny_tokenizer, 256, supervise_all_assistant_turns=False)
    supervised_all = sum(int((s["labels"] != -100).sum()) for s in all_turns.items)
    supervised_last = sum(int((s["labels"] != -100).sum()) for s in last_only.items)
    assert supervised_last < supervised_all


def test_sft_truncation_keeps_supervision(tmp_path, tiny_tokenizer):
    long_context = "背景信息。" * 400
    record = {"messages": [
        {"role": "system", "content": "系统提示。"},
        {"role": "user", "content": long_context},
        {"role": "assistant", "content": "这是最终答案。"},
    ]}
    path = write_jsonl(tmp_path / "long.jsonl", [record])
    dataset = SFTDataset(path, tiny_tokenizer, 64)
    assert len(dataset) == 1
    assert dataset.stats["truncated"] == 1
    item = dataset[0]
    assert item["input_ids"].shape[0] == 64
    assert item["labels"].shape[0] == item["input_ids"].shape[0]
    assert (item["labels"] != -100).any(), "truncation must not delete all supervision"
    for token, label in zip(item["input_ids"].tolist(), item["labels"].tolist()):
        assert label in (-100, token), "truncation must keep label/token alignment"
    supervised = tiny_tokenizer.decode(
        [t for t, l in zip(item["input_ids"].tolist(), item["labels"].tolist()) if l != -100],
        skip_special_tokens=True,
    )
    assert "最终答案" in supervised


def test_sft_drops_duplicates_and_malformed(tmp_path, tiny_tokenizer, sft_records):
    records = [
        sft_records[0],
        sft_records[0],  # exact duplicate
        {"messages": []},
        {"messages": [{"role": "user", "content": "只有提问，没有回答"}]},
        {"messages": "not a list"},
    ]
    path = write_jsonl(tmp_path / "mixed.jsonl", records)
    dataset = SFTDataset(path, tiny_tokenizer, 128)
    assert len(dataset) == 1
    assert dataset.stats["dropped"] == 4


def test_sft_rejects_unusable_file(tmp_path, tiny_tokenizer):
    path = write_jsonl(tmp_path / "bad.jsonl", [{"messages": []}])
    with pytest.raises(ValueError, match="no usable"):
        SFTDataset(path, tiny_tokenizer, 128)


def test_sft_loss_is_finite_and_training_runs(tmp_path, tiny_tokenizer, sft_records):
    cfg = MINILLM_NANO.copy()
    cfg.vocab_size = tiny_tokenizer.vocab_size
    train_path = write_jsonl(tmp_path / "train.jsonl", sft_records)
    val_path = write_jsonl(tmp_path / "val.jsonl", sft_records[:1])
    train_set = SFTDataset(train_path, tiny_tokenizer, 128)
    val_set = SFTDataset(val_path, tiny_tokenizer, 128)
    model = MiniLLM(cfg)

    batch = collate_batch([train_set[0], train_set[1]], 0, -100)
    out = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
    assert torch.isfinite(out.loss) and out.num_loss_tokens > 0
    assert out.num_loss_tokens < batch["labels"].numel(), "prompt tokens must be excluded"

    opts = TrainingOptions(
        out=str(tmp_path / "sft-run"), epochs=2, batch_size=2, grad_accum=1, max_steps=2,
        lr=1e-3, warmup_ratio=0.0, save_every=2, val_every=2, log_every=1,
        seed=0, device="cpu", precision="fp32", threads=2,
    )
    factory = sft_batch_factory(train_set, 2, 0, 0, -100)
    summary = fit(opts, model, tiny_tokenizer, train_set, val_set, mode="sft", batch_factory=factory)
    assert summary["mode"] == "sft" and summary["steps"] == 2
    assert (tmp_path / "sft-run" / "model.pt").exists()
    payload = torch.load(tmp_path / "sft-run" / "model.pt", map_location="cpu", weights_only=False)
    assert payload["mode"] == "sft"


def test_chat_template_used_by_sft_matches_inference(tiny_tokenizer):
    messages = [{"role": "user", "content": "你好"}]
    prompt = tiny_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    encoding = tiny_tokenizer.chat_encoding(messages, add_generation_prompt=True)
    assert prompt == encoding.ids
    assert all(label == tiny_tokenizer.ignore_index for label in encoding.labels)


# --------------------------------------------------------------------- P0 labels


def test_sft_labels_are_real_input_ids(tmp_path, tiny_tokenizer, sft_records):
    """Every supervised label must be the token the model actually has to emit."""
    dataset = SFTDataset(write_jsonl(tmp_path / "sft.jsonl", sft_records), tiny_tokenizer, 256)
    for item in dataset.items:
        ids, labels = item["input_ids"].tolist(), item["labels"].tolist()
        assert len(ids) == len(labels)
        for token, label in zip(ids, labels):
            assert label in (-100, token), "labels may only be -100 or the real token id"
        supervised = [t for t, l in zip(ids, labels) if l != -100]
        assert supervised, "assistant turns must carry loss"
        assert PAD_ID not in supervised, "PAD must never be a training target"
        assert any(t != PAD_ID for t in supervised)
        assert supervised[-1] == END_OF_TURN_ID, "the answer terminator is supervised"


def test_sft_supervises_every_assistant_turn(tmp_path, tiny_tokenizer, sft_records):
    dataset = SFTDataset(write_jsonl(tmp_path / "sft.jsonl", sft_records), tiny_tokenizer, 256)
    ids = dataset.items[1]["input_ids"].tolist()
    labels = dataset.items[1]["labels"].tolist()
    supervised = tiny_tokenizer.decode(
        [t for t, l in zip(ids, labels) if l != -100], skip_special_tokens=True
    )
    assert "def add" in supervised and "它接收两个参数" in supervised
    assert "写一个加法函数" not in supervised and "解释一下这个函数" not in supervised
    assert USER_ID not in [t for t, l in zip(ids, labels) if l != -100]


def test_sft_last_turn_only_keeps_real_labels(tmp_path, tiny_tokenizer, sft_records):
    dataset = SFTDataset(
        write_jsonl(tmp_path / "sft.jsonl", sft_records), tiny_tokenizer, 256,
        supervise_all_assistant_turns=False,
    )
    ids = dataset.items[1]["input_ids"].tolist()
    labels = dataset.items[1]["labels"].tolist()
    for token, label in zip(ids, labels):
        assert label in (-100, token)
    supervised = tiny_tokenizer.decode(
        [t for t, l in zip(ids, labels) if l != -100], skip_special_tokens=True
    )
    assert "它接收两个参数" in supervised
    assert "def add" not in supervised, "earlier assistant turns are not supervised"


def test_sft_training_learns_targets_not_pad(tmp_path, tiny_tokenizer):
    """Regression for the PAD-label bug: the model must learn the real answer."""
    torch.manual_seed(0)
    cfg = MINILLM_NANO.copy()
    cfg.vocab_size = tiny_tokenizer.vocab_size
    records = [
        {"messages": [
            {"role": "user", "content": "天空是什么颜色？"},
            {"role": "assistant", "content": "天空是蓝色的。"},
        ]},
        {"messages": [
            {"role": "user", "content": "一加一等于几？"},
            {"role": "assistant", "content": "一加一等于二。"},
        ]},
    ]
    path = write_jsonl(tmp_path / "qa.jsonl", records)
    train_set = SFTDataset(path, tiny_tokenizer, 128)
    model = MiniLLM(cfg)

    opts = TrainingOptions(
        out=str(tmp_path / "qa-run"), epochs=60, batch_size=2, grad_accum=1,
        max_steps=60, lr=5e-3, min_lr=1e-3, warmup_ratio=0.0, save_every=0,
        val_every=0, val_limit=1, log_every=1, seed=0, device="cpu",
        precision="fp32", threads=2,
    )
    factory = sft_batch_factory(train_set, 2, 0, 0, -100)
    summary = fit(opts, model, tiny_tokenizer, train_set, train_set, mode="sft", batch_factory=factory)
    assert summary["steps"] == 60

    model.eval()
    item = train_set[0]
    ids = item["input_ids"][None]
    labels = item["labels"][None]
    with torch.no_grad():
        out = model(ids, labels=labels)
        predictions = out.logits[:, :-1].argmax(-1)
    targets = labels[:, 1:]
    mask = targets != -100
    accuracy = ((predictions == targets) & mask).sum().item() / mask.sum().item()
    assert accuracy > 0.5, f"supervised accuracy only {accuracy:.2%}; are labels still PAD?"

    prompt = tiny_tokenizer.apply_chat_template(
        [{"role": "user", "content": "天空是什么颜色？"}], add_generation_prompt=True
    )
    with torch.no_grad():
        generated = model.generate(
            torch.tensor([prompt]), max_new_tokens=14, eos_token_id=cfg.eos_token_id,
            stop_token_ids=[END_OF_TURN_ID],
        )
    new_tokens = generated[0, len(prompt):].tolist()
    assert new_tokens, "generation produced no tokens"
    assert new_tokens != [PAD_ID] * len(new_tokens), "model collapsed to emitting PAD"
    assert PAD_ID not in new_tokens, "a trained answer should never be PAD"
