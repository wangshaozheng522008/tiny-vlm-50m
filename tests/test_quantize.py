"""INT4/INT8 weight-only quantisation: packing, budget, reload fidelity."""

from __future__ import annotations

import math

import pytest
import torch

from minillm.config import MINILLM_87M
from minillm.model import MiniLLM
from minillm.quantize import (
    BUDGET_BYTES,
    dequantize_model,
    effective_group,
    load_quantized,
    pack_int4,
    quantize_model,
    quantize_weight,
    save_quantized,
    unpack_int4,
)


def test_pack_unpack_roundtrip():
    codes = torch.randint(-8, 8, (4, 33), dtype=torch.int8)
    packed = pack_int4(codes)
    assert packed.shape == (4, 17) and packed.dtype == torch.uint8
    restored = unpack_int4(packed)[:, :33]
    assert torch.equal(restored, codes)
    assert int(packed.min()) >= 0 and int(packed.max()) <= 255


def test_effective_group_divides_in_features():
    assert effective_group(640, 128) == 128
    assert effective_group(1792, 128) == 128
    assert effective_group(96, 128) == 96
    assert effective_group(127, 128) == 127  # group >= in_features: one group
    assert effective_group(288, 128) in (96, 144, 72, 48) or 288 % effective_group(288, 128) == 0


def test_quantize_weight_error_is_bounded():
    torch.manual_seed(0)
    weight = torch.randn(32, 256)
    codes, scales, orig_in, group = quantize_weight(weight, bits=4, group_size=128)
    assert orig_in == 256 and group == 128 and codes.shape == (32, 256)
    restored = (codes.float().reshape(32, -1, group) * scales.unsqueeze(-1)).reshape(32, 256)
    error = (restored - weight).abs()
    assert error.max() <= scales.max() / 2 + 1e-6, "rounding error must stay within half a step"
    assert error.mean() < weight.std() * 0.2


def test_nano_quantize_roundtrip(nano_cfg, tmp_path):
    model = MiniLLM(nano_cfg).eval()
    payload = quantize_model(model, bits=4, group_size=128)
    assert payload["quantized"], "linear/embedding weights must be quantised"
    assert set(payload["fp16"]), "norms stay in fp16"
    # the tied embedding/LM head is captured exactly once
    assert "embed_tokens.weight" in payload["quantized"]
    assert "lm_head.weight" not in payload["quantized"]

    path = save_quantized(tmp_path / "nano-int4.pt", model, bits=4)
    reloaded = load_quantized(path, "cpu")
    assert reloaded.lm_head.weight.data_ptr() == reloaded.embed_tokens.weight.data_ptr()
    probe = torch.randint(0, nano_cfg.vocab_size, (2, 24))
    with torch.no_grad():
        a = model(probe).logits
        b = reloaded(probe).logits
    diff = (a - b).abs()
    assert diff.mean() < 0.2, diff.mean().item()
    assert (a.argmax(-1) == b.argmax(-1)).float().mean() > 0.75


def test_int8_is_more_accurate_than_int4(nano_cfg, tmp_path):
    model = MiniLLM(nano_cfg).eval()
    probe = torch.randint(0, nano_cfg.vocab_size, (1, 32))
    with torch.no_grad():
        reference = model(probe).logits
    errors = {}
    for bits in (4, 8):
        path = save_quantized(tmp_path / f"m{bits}.pt", model, bits=bits)
        reloaded = load_quantized(path, "cpu")
        with torch.no_grad():
            errors[bits] = float((reloaded(probe).logits - reference).abs().mean())
    assert errors[8] < errors[4]


def test_real_87m_int4_file_fits_the_budget():
    """Analytic size of the shipped 87M INT4 artifact (measured exactly in benchmark)."""
    cfg = MINILLM_87M
    from minillm.quantize import effective_group

    shapes = [("embed", cfg.vocab_size, cfg.hidden_size)]
    for _ in range(cfg.num_hidden_layers):
        shapes += [
            ("q", cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim),
            ("k", cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim),
            ("v", cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim),
            ("o", cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
            ("gate", cfg.hidden_size, cfg.intermediate_size),
            ("up", cfg.hidden_size, cfg.intermediate_size),
            ("down", cfg.intermediate_size, cfg.hidden_size),
        ]
    total = 0
    for _, out_f, in_f in shapes:
        group = effective_group(in_f, 128)
        padded = math.ceil(in_f / group) * group
        total += out_f * (padded // 2) + out_f * (padded // group) * 2
    norms = (2 * cfg.num_hidden_layers + 1) * cfg.hidden_size * 2
    total += norms
    assert total < BUDGET_BYTES, f"{total:,} bytes"
    assert total < 48_000_000, f"{total:,} bytes leaves too little margin"
