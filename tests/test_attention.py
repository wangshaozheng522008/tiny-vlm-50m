"""KV cache, masking and generation-equivalence tests."""

from __future__ import annotations

import pytest
import torch

from minillm.config import get_config
from minillm.model import MiniLLM


@pytest.fixture()
def model_and_ids():
    torch.manual_seed(0)
    cfg = get_config("minillm-nano")
    model = MiniLLM(cfg).eval()
    ids = torch.randint(4, cfg.vocab_size, (2, 10))
    return model, ids, cfg


def test_prefill_matches_full_forward(model_and_ids):
    model, ids, _ = model_and_ids
    with torch.no_grad():
        full = model(ids, use_cache=False)
        prefill = model(ids[:, :6], use_cache=True)
    assert torch.allclose(prefill.logits[:, -1], full.logits[:, 5], atol=1e-5)


def test_cached_decode_matches_full_forward(model_and_ids):
    model, ids, _ = model_and_ids
    with torch.no_grad():
        full = model(ids, use_cache=False)
        out = model(ids[:, :4], use_cache=True)
        cache = out.past_key_values
        for step in range(4, ids.shape[1]):
            out = model(ids[:, step : step + 1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            assert torch.allclose(out.logits[:, 0], full.logits[:, step], atol=1e-5)


def test_chunked_prefill_with_cache(model_and_ids):
    """A multi-token chunk decoded against a cache must respect the offset mask."""
    model, ids, _ = model_and_ids
    with torch.no_grad():
        full = model(ids[:, :8], use_cache=False)
        first = model(ids[:, :3], use_cache=True)
        second = model(ids[:, 3:8], past_key_values=first.past_key_values, use_cache=True)
    assert torch.allclose(second.logits, full.logits[:, 3:8], atol=1e-5)


def test_single_token_prefill(model_and_ids):
    model, ids, _ = model_and_ids
    with torch.no_grad():
        single = model(ids[:, :1], use_cache=True)
        full = model(ids[:, :1], use_cache=False)
    assert single.past_key_values[0][0].shape[2] == 1
    assert torch.allclose(single.logits, full.logits, atol=1e-6)


def test_greedy_generation_identical_with_and_without_cache(model_and_ids):
    model, ids, _ = model_and_ids
    prompt = ids[:, :5]
    with torch.no_grad():
        cached = model.generate(prompt, max_new_tokens=6, temperature=0.0, use_cache=True)
        plain = model.generate(prompt, max_new_tokens=6, temperature=0.0, use_cache=False)
    assert torch.equal(cached, plain)
    assert cached.shape == (prompt.shape[0], prompt.shape[1] + 6)


def test_sampling_is_reproducible_with_generator(model_and_ids):
    model, ids, _ = model_and_ids
    prompt = ids[:, :4]
    with torch.no_grad():
        a = model.generate(prompt, max_new_tokens=5, temperature=0.8, top_k=20,
                           generator=torch.Generator().manual_seed(7))
        b = model.generate(prompt, max_new_tokens=5, temperature=0.8, top_k=20,
                           generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_eos_stops_generation(model_and_ids):
    model, ids, cfg = model_and_ids
    prompt = ids[:, :4]
    with torch.no_grad():
        greedy = model.generate(prompt, max_new_tokens=1, temperature=0.0)
        eos = int(greedy[0, -1])
        out = model.generate(prompt, max_new_tokens=6, temperature=0.0, eos_token_id=eos)
    assert out.shape[1] <= prompt.shape[1] + 6
    # every token after the first EOS is EOS again
    tail = out[:, prompt.shape[1]:]
    assert (tail == eos).any()
    first = int((tail[0] != eos).sum())
    assert torch.all(tail[0, first:] == eos)


def test_context_length_boundary(model_and_ids):
    model, _, cfg = model_and_ids
    limit = cfg.max_position_embeddings
    with torch.no_grad():
        logits = model(torch.zeros(1, limit, dtype=torch.long)).logits
    assert logits.shape[1] == limit
    with pytest.raises(ValueError, match="exceeds max_position_embeddings"):
        model(torch.zeros(1, limit + 1, dtype=torch.long))
    with pytest.raises(ValueError, match="max_position_embeddings"):
        model.generate(torch.zeros(1, limit - 2, dtype=torch.long), max_new_tokens=5)
    with pytest.raises(ValueError, match="max_new_tokens"):
        model.generate(torch.zeros(1, 4, dtype=torch.long), max_new_tokens=-1)


def test_padding_mask_matches_unpadded_sequences():
    """Right-padded batches must give the same real-token logits as single rows."""
    torch.manual_seed(1)
    cfg = get_config("minillm-nano")
    model = MiniLLM(cfg).eval()
    a = torch.randint(4, cfg.vocab_size, (1, 5))
    b = torch.randint(4, cfg.vocab_size, (1, 8))
    padded = torch.full((2, 8), cfg.pad_token_id, dtype=torch.long)
    padded[0, :5] = a
    padded[1] = b[0]
    mask = torch.zeros(2, 8, dtype=torch.long)
    mask[0, :5] = 1
    mask[1] = 1
    with torch.no_grad():
        batch = model(padded, attention_mask=mask, use_cache=False)
        solo_a = model(a, use_cache=False)
        solo_b = model(b, use_cache=False)
    assert torch.allclose(batch.logits[0, :5], solo_a.logits[0], atol=1e-5)
    assert torch.allclose(batch.logits[1], solo_b.logits[0], atol=1e-5)


def test_loss_ignores_padding():
    torch.manual_seed(2)
    cfg = get_config("minillm-nano")
    model = MiniLLM(cfg)
    ids = torch.randint(4, cfg.vocab_size, (1, 6))
    labels_padded = torch.full((2, 8), cfg.ignore_index, dtype=torch.long)
    labels_padded[0, :6] = ids[0]
    padded_ids = torch.full((2, 8), cfg.pad_token_id, dtype=torch.long)
    padded_ids[0, :6] = ids[0]
    mask = torch.zeros(2, 8, dtype=torch.long)
    mask[0, :6] = 1
    with torch.no_grad():
        out = model(padded_ids, attention_mask=mask, labels=labels_padded)
    # only the 5 shifted real targets are supervised
    assert out.num_loss_tokens == 5
