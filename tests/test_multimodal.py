import copy
import json
import sys

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from tiny_vlm.config import TextConfig, VisionConfig, VLMConfig
from tiny_vlm.data import (
    MODES, MultimodalDataset, build_example, build_prompt, collate_fn, read_records,
)
from tiny_vlm.model import TinyVLM
from tiny_vlm.tokenizer import BOS_ID, EOS_ID, TinyTokenizer
from tiny_vlm.train import evaluate, make_sampler, train_update, update_batches


@pytest.fixture
def tokenizer():
    return TinyTokenizer.train([
        "What color is the image? red blue green circle square",
        "Describe this image: a red circle", "你好，请描述图片。这是红色圆形。",
    ], vocab_size=400)


@pytest.fixture
def config(tokenizer):
    return VLMConfig(
        vision=VisionConfig(img_size=32, patch_size=16, d_model=32, depth=2, num_heads=4),
        text=TextConfig(vocab_size=tokenizer.vocab_size, max_seq_len=96,
                        d_model=32, depth=2, num_heads=4),
    )


@pytest.fixture
def records_path(tmp_path):
    Image.new("RGB", (32, 32), "red").save(tmp_path / "red.png")
    records = [
        {"prompt": "What color is a red circle?", "answer": "red"},
        {"image": "red.png", "answer": "a red circle"},
        {"image": "red.png", "prompt": "What color is the image?", "answer": "red"},
    ]
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def test_three_modes_and_mixed_collation(tokenizer, records_path):
    dataset = MultimodalDataset(records_path, tokenizer, img_size=32)
    assert dataset.modes == list(MODES)
    batch = collate_fn([dataset[i] for i in range(3)])
    assert batch["image_mask"].tolist() == [False, True, True]
    assert batch["mode_id"].tolist() == [0, 1, 2]
    assert batch["pixel_values"].shape == (3, 3, 32, 32)
    assert dataset[1]["input_ids"][0].item() == BOS_ID
    assert dataset[1]["labels"][1].item() != -100
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    assert "pixel_values" not in collate_fn([dataset[0]])


def test_prefix_matches_inference_and_preserves_answer(tokenizer):
    prompt = "你好，请描述图片。"
    ids, labels = build_example(tokenizer, prompt, "这是红色圆形。", 96)
    prefix = build_prompt(tokenizer, prompt)
    assert ids[:len(prefix)] == prefix
    assert EOS_ID not in prefix
    assert labels[:len(prefix)] == [-100] * len(prefix)
    assert ids[-1] == labels[-1] == EOS_ID
    ids, labels = build_example(tokenizer, prompt * 50, "red", 5)
    assert len(ids) <= 5 and ids[0] == BOS_ID
    assert any(t != -100 for t in labels[1:])
    assert tokenizer.decode([t for t in labels if t != -100]) == "red"


def test_raw_text_and_legacy_caption(tokenizer, tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text(json.dumps({"text": "你好"}) + "\n" +
                    json.dumps({"image": "red.png", "caption": "a red circle"}))
    records = read_records(path)
    assert records[0].mode == "text" and records[0].prompt == ""
    assert records[1].mode == "mixed" and records[1].prompt == "Describe this image:"
    assert records[1].image == tmp_path / "red.png"
    dataset = MultimodalDataset(path, tokenizer, img_size=32)
    assert dataset[0]["labels"][1].item() != -100


@pytest.mark.parametrize("record", [{"answer": "red"}, {"prompt": "q", "answer": " "},
                                      {"image": "", "answer": "red"}, []])
def test_invalid_records_have_source_location(record, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=r"bad.jsonl:1:"):
        read_records(path)


def test_multiple_sources_resolve_images_locally(tmp_path):
    paths = []
    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        path = folder / "records.jsonl"
        path.write_text(json.dumps({"image": "red.png", "answer": "red"}))
        paths.append(path)
    records = read_records(paths)
    assert records[0].image != records[1].image


@pytest.mark.parametrize("with_padding", [False, True])
def test_missing_images_match_standalone_text(config, with_padding):
    torch.manual_seed(4)
    model = TinyVLM(config).eval()
    ids = torch.randint(5, config.text.vocab_size, (2, 8))
    image_mask = torch.tensor([False, True])
    pixels = torch.randn(2, 3, 32, 32)
    mask = torch.ones_like(ids) if with_padding else None
    if mask is not None:
        mask[:, -2:] = 0
    with torch.no_grad():
        mixed = model(ids, pixel_values=pixels, image_mask=image_mask, attention_mask=mask)
        text = model(ids[:1], attention_mask=mask[:1] if mask is not None else None)
        image = model(ids[1:], pixel_values=pixels[1:], attention_mask=mask[1:] if mask is not None else None)
    torch.testing.assert_close(mixed["logits"][:1], text["logits"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(mixed["logits"][1:], image["logits"], atol=1e-6, rtol=1e-5)


def test_missing_images_are_not_encoded_and_gradients_are_finite(config, monkeypatch):
    model = TinyVLM(config)
    encoded_sizes = []
    encode = model.encode_images

    def spy(pixels):
        encoded_sizes.append(pixels.shape[0])
        return encode(pixels)

    monkeypatch.setattr(model, "encode_images", spy)
    ids = torch.randint(5, config.text.vocab_size, (3, 6))
    pixels = torch.randn(3, 3, 32, 32, requires_grad=True)
    out = model(ids, pixel_values=pixels, image_mask=torch.tensor([False, True, True]), labels=ids)
    out["loss"].backward()
    assert encoded_sizes == [2]
    assert torch.isfinite(pixels.grad).all()
    assert pixels.grad[0].abs().sum() == 0
    assert pixels.grad[1:].abs().sum() > 0
    assert model.vision.patch_embed.proj.weight.grad.abs().sum() > 0
    encoded_sizes.clear()
    model(ids, pixel_values=pixels.detach(), image_mask=torch.zeros(3, dtype=torch.bool))
    assert encoded_sizes == []


def test_generation_caches_images_and_respects_context_limit(config, monkeypatch):
    model = TinyVLM(config)
    calls = []
    encode = model.encode_images

    def spy(pixels):
        calls.append(pixels.shape[0])
        return encode(pixels)

    monkeypatch.setattr(model, "encode_images", spy)
    ids = torch.tensor([[BOS_ID]])
    result = model.generate(ids, pixel_values=torch.randn(1, 3, 32, 32),
                            max_new_tokens=4, temperature=0)
    assert result.shape == (1, 5) and calls == [1]
    near_limit = torch.full((1, config.text.max_seq_len - 1), BOS_ID)
    assert model.generate(near_limit, max_new_tokens=8, temperature=0).shape[1] == config.text.max_seq_len


def test_token_weighted_accumulation_matches_large_batch(config, tokenizer, records_path):
    torch.manual_seed(4)
    original = TinyVLM(config)
    accumulated = copy.deepcopy(original)
    dataset = MultimodalDataset(records_path, tokenizer, img_size=32, max_seq_len=96)
    items = [dataset[i] for i in range(3)]
    large = [collate_fn(items)]
    small = [collate_fn([item]) for item in items]
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    for model, batches in ((original, large), (accumulated, small)):
        train_update(model, batches, torch.optim.SGD(model.parameters(), lr=0.01), scaler,
                     torch.device("cpu"))
    for expected, actual in zip(original.parameters(), accumulated.parameters()):
        torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-5)
    assert [len(group) for group in update_batches(range(5), 2)] == [2, 2, 1]


def test_balanced_sampler_weights_and_missing_modes():
    sampler = make_sampler(["text"] * 10 + ["image", "mixed"], [1, 1, 1], 42)
    assert float(sampler.weights[:10].sum()) == pytest.approx(float(sampler.weights[10]))
    assert make_sampler(["text"], None, 42) is None
    with pytest.raises(ValueError):
        make_sampler(["text"], [0, 1, 1], 42)


def test_train_validate_warm_start_and_all_inference_modes(config, tokenizer, records_path,
                                                         tmp_path, monkeypatch, capsys):
    from tiny_vlm import generate, train

    monkeypatch.setattr(train, "get_config", lambda _: copy.deepcopy(config))
    out_dir = tmp_path / "run"
    tokenizer_path = tmp_path / "original-tokenizer.json"
    tokenizer.save(tokenizer_path)
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(records_path),
        "--val-data", str(records_path), "--tokenizer", str(tokenizer_path),
        "--out", str(out_dir), "--batch-size", "1", "--grad-accum", "2",
        "--epochs", "1", "--workers", "0", "--device", "cpu"])
    train.main()
    payload = torch.load(out_dir / "model.pt", weights_only=False)
    assert payload["step"] == 2  # remainder batch must produce an update
    assert payload["training_modes"] == {"text": 1, "image": 1, "mixed": 1}
    history = json.loads((out_dir / "metrics.json").read_text())
    assert set(history[0]["validation"]["modes"]) == set(MODES)
    assert (out_dir / "best.pt").exists()
    assert TinyTokenizer.load(out_dir / "tokenizer.json").fingerprint == tokenizer.fingerprint

    for mode in MODES:
        args = ["generate", "--ckpt", str(out_dir / "model.pt"), "--tokenizer",
                str(out_dir / "tokenizer.json"), "--max-new-tokens", "2", "--device", "cpu"]
        if mode != "image":
            args += ["--prompt", "What color is the image?"]
        if mode != "text":
            args += ["--image", str(records_path.parent / "red.png")]
        monkeypatch.setattr(sys, "argv", args)
        generate.main()

    warm_dir = tmp_path / "warm"
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(records_path),
        "--init-from", str(out_dir / "model.pt"), "--freeze-vision", "--out", str(warm_dir),
        "--epochs", "1", "--max-steps", "1", "--workers", "0", "--device", "cpu"])
    train.main()
    warm = torch.load(warm_dir / "model.pt", weights_only=False)
    assert warm["step"] == 1
    torch.testing.assert_close(warm["model"]["vision.patch_embed.proj.weight"],
                               payload["model"]["vision.patch_embed.proj.weight"])

    model, _ = generate.load_model(out_dir / "model.pt", torch.device("cpu"))
    dataset = MultimodalDataset(records_path, tokenizer, img_size=32)
    model.train()
    metrics = evaluate(model, DataLoader(dataset, batch_size=3, collate_fn=collate_fn), torch.device("cpu"))
    assert model.training and set(metrics["modes"]) == set(MODES)
    capsys.readouterr()


def test_quantized_models_preserve_input_modes_and_tokenizer_identity(config, tokenizer, tmp_path):
    from tokenizers import Tokenizer

    from tiny_vlm.generate import load_model, validate_tokenizer
    from tiny_vlm.quantize import save_quantized

    path = tmp_path / "model-int8.pt"
    model = TinyVLM(config)
    save_quantized(path, model, config, metadata={"tokenizer_fingerprint": tokenizer.fingerprint,
                                                "training_modes": {mode: 1 for mode in MODES}})
    restored, _ = load_model(path, torch.device("cpu"))
    validate_tokenizer(restored, tokenizer)
    for prompt, pixels in [("red", None), ("", torch.randn(1, 3, 32, 32)),
                           ("What color?", torch.randn(1, 3, 32, 32))]:
        ids = torch.tensor([build_prompt(tokenizer, prompt)])
        generated = restored.generate(ids, pixel_values=pixels, max_new_tokens=1, temperature=0)
        assert generated.shape[1] == ids.shape[1] + 1
    raw = json.loads(tokenizer._tok.to_str())
    tokens = [token for token, value in raw["model"]["vocab"].items() if value >= 5][:2]
    vocab = raw["model"]["vocab"]
    vocab[tokens[0]], vocab[tokens[1]] = vocab[tokens[1]], vocab[tokens[0]]
    wrong = TinyTokenizer(Tokenizer.from_str(json.dumps(raw)))
    assert wrong.vocab_size == tokenizer.vocab_size
    with pytest.raises(ValueError, match="merge rules"):
        validate_tokenizer(restored, wrong)
