import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from tiny_vlm.data import IndexedTextDataset, MultimodalDataset, build_example
from tiny_vlm.tokenizer import BOS_ID, EOS_ID, TinyTokenizer
from tiny_vlm.train import EpochBatchSampler

ROOT = Path(__file__).resolve().parents[1]


def load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_text", ROOT / "scripts" / "prepare_text.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tokenizer():
    return TinyTokenizer.train([
        "人工智能是计算机科学的一个分支，它研究如何让机器表现出智能。",
        "语言模型通过预测下一个词来学习文本中的统计规律。",
        "山间的清晨，薄雾在松林之间缓慢流动，鸟鸣声从远处传来。",
        "The quick brown fox jumps over the lazy dog while the sun sets.",
        "训练一个小模型需要在数据质量、算力和时间之间做取舍。",
    ], vocab_size=400)


def test_quality_filters_and_normalise(tokenizer):
    prepare = load_prepare_module()
    assert prepare.normalise("  a\r\n\r\n\r\n b\u3000c  ") == "a\n\n b c"
    assert prepare.quality_issue("短文本", 20, 20000, deep=False) == "too_short"
    assert prepare.quality_issue("正常的中文句子。" * 4, 20, 20000, deep=False) is None
    assert prepare.quality_issue("正常文本" * 5 + "\ufffd", 20, 20000, deep=False) == "replacement_char"
    assert prepare.quality_issue("正常文本" * 5 + "\x07", 20, 20000, deep=False) == "control_char"
    assert prepare.quality_issue("😀" * 30, 20, 20000, deep=False) == "low_printable_ratio"
    assert prepare.quality_issue("啊" * 30, 20, 20000, deep=False) == "repeated_char"
    assert prepare.quality_issue("这是一个正常长度的中文段落，用来测试短语重复过滤。" * 4, 20, 20000,
                                 deep=True) == "repeated_phrase"


def test_chunk_text_is_bounded_and_character_safe(tokenizer):
    prepare = load_prepare_module()
    text = "秋天到了，树叶变成金色。" * 30
    pieces = prepare.chunk_text(tokenizer, text, limit=30, min_tokens=4)
    assert len(pieces) > 1
    for piece in pieces:
        ids = tokenizer.encode(piece, add_special_tokens=False)
        assert 4 <= len(ids) <= 30
        assert "\ufffd" not in piece
        assert tokenizer.decode(ids) == piece
    # Short text that fits stays in one piece, too-short text disappears.
    assert prepare.chunk_text(tokenizer, "你好世界", limit=30, min_tokens=4) == ["你好世界"]
    assert prepare.chunk_text(tokenizer, "好", limit=30, min_tokens=4) == []


def test_prepare_text_splits_before_chunking(tmp_path, tokenizer, monkeypatch):
    prepare = load_prepare_module()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    good = [
        "人工智能是计算机科学的一个分支，它研究如何让机器表现出智能。这是第二句话。",
        "语言模型通过预测下一个词来学习文本中的统计规律，并且需要大量高质量语料。",
        "山间的清晨，薄雾在松林之间缓慢流动，鸟鸣声从远处传来，空气清冷而干净。",
        "训练一个小模型需要在数据质量、算力和时间之间做取舍，这也是工程的一部分。",
        "模型的结构决定了参数预算，词表大小又直接影响嵌入层的体积和分词效率。",
        "数据清洗包括去重、过滤乱码和删除明显低质量的文本，顺序会影响最终样本。",
        "评估验证集损失只能说明分布内的拟合程度，实际生成质量还需要人工查看。",
        "图片描述和图文问答任务需要单独的视觉数据，纯文本训练不能替代它们。",
        "秋天的叶子从绿色变成金黄，风吹过树林时发出细碎的声响，让人想起旧时光。",
        "在四GB显存的显卡上，批量大小、梯度累积和序列长度共同决定了训练是否可行。",
    ]
    records = [{"text": text} for text in good]
    records += [
        {"text": "太短。"},
        {"text": good[0]},  # exact duplicate
        {"text": "坏文本" * 8 + "\ufffd"},
        {"text": "😀" * 40},
        {"text": good[1].replace("。", "，") * 4},  # repeated phrase
    ]
    source = tmp_path / "raw.jsonl"
    source.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records),
                      encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prepare_text", "--input", str(source),
        "--out", str(out), "--tokenizer", str(tokenizer_path), "--sample", "20",
        "--max-seq-len", "24", "--min-chars", "20", "--min-chunk-tokens", "4",
        "--val-ratio", "0.2", "--test-ratio", "0.2", "--seed", "7"])
    prepare.main()

    report = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
    assert report["sample_kept"] == len(good)  # duplicates and junk removed
    assert report["scan"]["rejected"]["too_short"] == 1
    assert report["scan"]["duplicates"]["exact"] == 1
    assert report["splits"]["train"]["texts"] + report["splits"]["val"]["texts"] + \
        report["splits"]["test"]["texts"] == len(good)
    assert report["ratios"]["train"] == pytest.approx(0.6)
    assert report["leakage"]["overlapping_chunks"] == 0

    seen = {}
    for split in ("train", "val", "test"):
        lines = (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        assert lines
        for line in lines:
            text = json.loads(line)["text"]
            assert "\ufffd" not in text and text.strip()
            ids, labels = build_example(tokenizer, "", text, 24, -100)
            assert len(ids) <= 24 and ids[0] == BOS_ID and ids[-1] == EOS_ID
            assert labels[0] == -100 and labels[-1] == EOS_ID
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            assert digest not in seen, f"chunk duplicated across {seen.get(digest)} and {split}"
            seen[digest] = split
    assert len((out / "preview.json").read_text(encoding="utf-8")) > 10


def test_indexed_dataset_matches_eager_reader(tmp_path, tokenizer):
    path = tmp_path / "chunks.jsonl"
    texts = ["这是第一段预训练语料，包含若干中文句子。",
             "第二段文本用来检查索引读取与一次性读取是否一致。",
             "第三段比较短。"]
    path.write_text("\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in texts) + "\n",
                    encoding="utf-8")
    indexed = IndexedTextDataset(path, tokenizer, max_seq_len=32)
    eager = MultimodalDataset(path, tokenizer, max_seq_len=32)
    assert len(indexed) == len(eager) == len(texts)
    assert indexed.mode_counts == {"text": len(texts)}
    assert indexed.modes == ["text"] * len(texts)
    for i in range(len(texts)):
        left, right = indexed[i], eager[i]
        assert torch.equal(left["input_ids"], right["input_ids"])
        assert torch.equal(left["labels"], right["labels"])
        assert int(left["mode_id"]) == int(right["mode_id"]) == 0
    # Many accesses reuse the cached handle and stay correct.
    assert torch.equal(indexed[len(texts) - 1]["input_ids"], indexed[2]["input_ids"])

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"prompt": "q", "answer": "a"}), encoding="utf-8")
    broken = IndexedTextDataset(bad, tokenizer, max_seq_len=32)
    with pytest.raises(ValueError, match="expected"):
        _ = broken[0]


def test_epoch_batch_sampler_resumes_without_skipping():
    full = EpochBatchSampler(10, 3, seed=5, epoch=0)
    flat = [index for batch in full for index in batch]
    assert sorted(flat) == list(range(10))
    assert len(full) == 4
    resumed = EpochBatchSampler(10, 3, seed=5, epoch=0, start_batch=2)
    assert len(resumed) == 2
    assert [index for batch in resumed for index in batch] == flat[6:]
    assert list(EpochBatchSampler(10, 3, seed=5, epoch=0))[0] == full.batches[0]
    assert EpochBatchSampler(10, 3, seed=5, epoch=1).batches != full.batches
    with pytest.raises(ValueError):
        EpochBatchSampler(10, 3, seed=5, epoch=0, start_batch=99)


def test_training_resumes_optimizer_and_progress(tmp_path, tokenizer, monkeypatch):
    from tiny_vlm import train
    from tiny_vlm.config import TextConfig, VisionConfig, VLMConfig

    cfg = VLMConfig(
        vision=VisionConfig(img_size=32, patch_size=16, d_model=32, depth=2, num_heads=4),
        text=TextConfig(vocab_size=tokenizer.vocab_size, max_seq_len=32,
                        d_model=32, depth=2, num_heads=4),
    )
    monkeypatch.setattr(train, "get_config", lambda _: copy.deepcopy(cfg))
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"text": "训练语料第%d句，用来测试断点恢复流程。" % i},
                                         ensure_ascii=False) for i in range(8)), encoding="utf-8")
    tokenizer_path = tmp_path / "tok.json"
    tokenizer.save(tokenizer_path)
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(data), "--val-data", str(data),
        "--tokenizer", str(tokenizer_path), "--out", str(out), "--batch-size", "1",
        "--grad-accum", "1", "--epochs", "2", "--max-steps", "2", "--save-every", "1",
        "--workers", "0", "--device", "cpu", "--indexed"])
    train.main()
    state = torch.load(out / "state.pt", weights_only=False)
    assert state["step"] == 2 and state["optimizer"]["state"]
    first_weights = {k: v.clone() for k, v in state["model"].items()}

    # Resume without --data/--val-data/--tokenizer: they come from the state file.
    monkeypatch.setattr(sys, "argv", ["train", "--resume", str(out / "state.pt"),
        "--out", str(out), "--max-steps", "4", "--save-every", "1",
        "--workers", "0", "--device", "cpu"])
    train.main()
    resumed = torch.load(out / "state.pt", weights_only=False)
    assert resumed["step"] == 4
    assert resumed["total_steps"] == 4
    assert any(not torch.equal(first_weights[k], resumed["model"][k]) for k in first_weights)
    history = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert history and all("validation" in entry for entry in history)
    assert (out / "best.pt").exists() and (out / "config.json").exists()
    assert TinyTokenizer.load(out / "tokenizer.json").fingerprint == tokenizer.fingerprint
