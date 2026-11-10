"""Checkpoint loading and generation paths (fp32 and quantised)."""

from __future__ import annotations

import pytest
import torch

from minillm.config import MINILLM_NANO
from minillm.generate import load_any_checkpoint
from minillm.model import MiniLLM
from minillm.quantize import save_quantized
from minillm.tokenizer import EOS_ID, MiniLLMTokenizer
from minillm.train import save_model_checkpoint


@pytest.fixture()
def checkpoint(tmp_path, tiny_tokenizer, nano_cfg):
    model = MiniLLM(nano_cfg).eval()
    path = tmp_path / "model.pt"
    save_model_checkpoint(path, model, tiny_tokenizer, step=3, tokens=99, val_loss=1.5)
    return path, model


def test_load_checkpoint_restores_weights(checkpoint):
    path, model = checkpoint
    loaded, payload = load_any_checkpoint(path, torch.device("cpu"))
    assert payload["architecture"] == "minillm" and payload["step"] == 3
    for (name, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(a, b), name
    assert not loaded.training


def test_load_rejects_vision_language_checkpoint(tmp_path):
    path = tmp_path / "vlm.pt"
    torch.save({"architecture": "tiny_vlm", "config": {}, "model": {}}, path)
    with pytest.raises(SystemExit, match="TinyVLM"):
        load_any_checkpoint(path, torch.device("cpu"))


def test_greedy_generation_is_deterministic(checkpoint, tiny_tokenizer):
    path, _ = checkpoint
    model, _ = load_any_checkpoint(path, torch.device("cpu"))
    prompt = torch.tensor([tiny_tokenizer.apply_chat_template(
        [{"role": "user", "content": "你好"}], add_generation_prompt=True
    )])
    a = model.generate(prompt, max_new_tokens=12, temperature=0.0)
    b = model.generate(prompt, max_new_tokens=12, temperature=0.0)
    assert a.shape[1] == prompt.shape[1] + 12
    assert torch.equal(a, b)
    assert isinstance(tiny_tokenizer.decode(a[0, prompt.shape[1]:].tolist()), str)


def test_cache_and_no_cache_greedy_match(checkpoint, tiny_tokenizer):
    path, _ = checkpoint
    model, _ = load_any_checkpoint(path, torch.device("cpu"))
    prompt = torch.tensor([[1, 5, 42, 43, 44, 6]])
    cached = model.generate(prompt, max_new_tokens=8, use_cache=True)
    plain = model.generate(prompt, max_new_tokens=8, use_cache=False)
    assert torch.equal(cached, plain)


def test_generation_stops_at_eos(checkpoint):
    path, _ = checkpoint
    model, _ = load_any_checkpoint(path, torch.device("cpu"))
    prompt = torch.tensor([[1, 5, 42]])
    out = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    eos_positions = (out[0] == EOS_ID).nonzero().flatten().tolist()
    assert out.shape[1] <= prompt.shape[1] + 6
    if eos_positions:
        assert eos_positions[0] == out.shape[1] - 1, "no tokens are produced after EOS"


def test_quantized_checkpoint_loads_for_generation(tmp_path, checkpoint, tiny_tokenizer, nano_cfg):
    _, model = checkpoint
    qpath = save_quantized(tmp_path / "model-int4.pt", model, bits=4,
                           tokenizer_fingerprint=tiny_tokenizer.fingerprint)
    loaded, payload = load_any_checkpoint(qpath, torch.device("cpu"))
    assert payload["quantized_bits"] == 4
    assert loaded.lm_head.weight.data_ptr() == loaded.embed_tokens.weight.data_ptr()
    out = loaded.generate(torch.tensor([[1, 5, 42]]), max_new_tokens=4, temperature=0.0)
    assert out.shape[1] == 7


def test_context_overflow_is_an_explicit_error(checkpoint):
    path, _ = checkpoint
    model, _ = load_any_checkpoint(path, torch.device("cpu"))
    cfg = model.cfg
    prompt = torch.ones((1, cfg.max_position_embeddings), dtype=torch.long)
    with pytest.raises(ValueError, match="max_position_embeddings"):
        model.generate(prompt, max_new_tokens=1)
    with pytest.raises(ValueError, match="max_new_tokens"):
        model.generate(prompt[:, :2], max_new_tokens=cfg.max_position_embeddings)
