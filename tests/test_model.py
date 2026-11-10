"""Structural tests for MiniLLM-87M: budget, GQA, RoPE, SwiGLU, weight tying."""

from __future__ import annotations

import pytest
import torch

from minillm.config import MINILLM_87M, MiniLLMConfig, get_config
from minillm.model import (
    MiniLLM,
    apply_rope,
    precompute_rope,
    repeat_kv,
    rotate_half,
)

BUDGET = {
    "embedding": 15_728_640,
    "attention": 15_728_640,
    "ffn": 55_050_240,
    "norm": 21_120,
    "total": 86_528_640,
}


def test_config_matches_spec():
    cfg = MINILLM_87M
    assert cfg.vocab_size == 24576
    assert cfg.hidden_size == 640
    assert cfg.num_hidden_layers == 16
    assert cfg.num_attention_heads == 10
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 64
    assert cfg.intermediate_size == 1792
    assert cfg.max_position_embeddings == 2048
    assert cfg.num_kv_groups == 5


def test_parameter_budget_is_exact():
    cfg = get_config("minillm-87m")
    assert cfg.param_breakdown() == BUDGET
    model = MiniLLM(cfg)
    assert model.num_parameters() == BUDGET["total"]
    assert model.parameter_breakdown() == BUDGET


def test_lm_head_shares_embedding_storage():
    model = MiniLLM(get_config("minillm-nano"))
    assert model.lm_head.weight is model.embed_tokens.weight
    assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()
    # A tied checkpoint stores one tensor even though state_dict exposes two names.
    state = model.state_dict()
    assert state["lm_head.weight"].data_ptr() == state["embed_tokens.weight"].data_ptr()


def test_rope_tables_are_not_checkpoint_state():
    model = MiniLLM(get_config("minillm-nano"))
    assert not any("rope" in key for key in model.state_dict())


def test_rope_is_relative():
    """rope(q, p) . rope(k, p + d) must not depend on p."""
    torch.manual_seed(0)
    head_dim = 8
    cos, sin = precompute_rope(head_dim, 16, 10000.0)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)
    scores = []
    for offset in (0, 3, 7):
        q_rot = apply_rope(q, cos[offset : offset + 1], sin[offset : offset + 1])
        k_rot = apply_rope(k, cos[offset + 5 : offset + 6], sin[offset + 5 : offset + 6])
        scores.append((q_rot * k_rot).sum().item())
    assert max(scores) - min(scores) < 1e-4


def test_rotate_half_matches_definition():
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    assert torch.equal(rotate_half(x), torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]]))


def test_gqa_head_mapping():
    cfg = get_config("minillm-nano")
    model = MiniLLM(cfg)
    attn = model.layers[0].self_attn
    assert attn.n_heads == cfg.num_attention_heads
    assert attn.n_kv_heads == cfg.num_key_value_heads
    assert attn.n_rep == cfg.num_attention_heads // cfg.num_key_value_heads
    assert attn.q_proj.out_features == cfg.num_attention_heads * cfg.head_dim
    assert attn.k_proj.out_features == cfg.num_key_value_heads * cfg.head_dim
    assert attn.v_proj.out_features == cfg.num_key_value_heads * cfg.head_dim
    assert attn.o_proj.out_features == cfg.hidden_size
    # repeat_kv maps each KV head onto n_rep consecutive query heads
    kv = torch.arange(cfg.num_key_value_heads * 2, dtype=torch.float32).reshape(
        1, cfg.num_key_value_heads, 1, 2
    )
    repeated = repeat_kv(kv, attn.n_rep)
    assert repeated.shape[1] == cfg.num_attention_heads
    for head in range(cfg.num_attention_heads):
        assert torch.equal(repeated[:, head], kv[:, head // attn.n_rep])


def test_swiglu_dimensions():
    cfg = get_config("minillm-nano")
    mlp = MiniLLM(cfg).layers[0].mlp
    assert mlp.gate_proj.out_features == cfg.intermediate_size
    assert mlp.up_proj.out_features == cfg.intermediate_size
    assert mlp.down_proj.in_features == cfg.intermediate_size
    assert mlp.gate_proj.bias is None and mlp.down_proj.bias is None


def test_attention_and_mlp_have_no_bias():
    model = MiniLLM(get_config("minillm-nano"))
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            assert module.bias is None, name


def test_forward_backward_on_nano():
    torch.manual_seed(0)
    cfg = get_config("minillm-nano")
    model = MiniLLM(cfg)
    ids = torch.randint(4, cfg.vocab_size, (2, 16))
    out = model(ids, labels=ids)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(out.loss) and out.loss.item() > 0
    assert 0 < out.num_loss_tokens <= 2 * 15
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # the embedding (tied to the LM head) is the first parameter and receives gradient
    assert model.embed_tokens.weight.grad is not None


def test_max_position_is_enforced():
    cfg = get_config("minillm-nano")
    model = MiniLLM(cfg)
    too_long = torch.zeros(1, cfg.max_position_embeddings + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="max_position_embeddings"):
        model(too_long)


def test_config_validation_rejects_bad_shapes():
    with pytest.raises(ValueError, match="hidden_size"):
        MiniLLMConfig(hidden_size=641)
    with pytest.raises(ValueError, match="multiple"):
        MiniLLMConfig(num_attention_heads=10, num_key_value_heads=3)
    with pytest.raises(ValueError, match="num_key_value_heads"):
        MiniLLMConfig(num_attention_heads=2, num_key_value_heads=4, hidden_size=128)
    with pytest.raises(ValueError, match="pad_token_id"):
        MiniLLMConfig(vocab_size=8, pad_token_id=9)


def test_config_roundtrip(tmp_path):
    cfg = get_config("minillm-nano")
    path = tmp_path / "config.json"
    cfg.save(path)
    loaded = MiniLLMConfig.load(path)
    assert loaded.to_dict() == cfg.to_dict()
    with pytest.raises(ValueError, match="unknown config fields"):
        MiniLLMConfig.from_dict({**cfg.to_dict(), "mystery": 1})
