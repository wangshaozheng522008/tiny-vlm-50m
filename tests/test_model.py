import pytest
import torch

from tiny_vlm.config import TextConfig, VisionConfig, VLMConfig, get_config
from tiny_vlm.model import TinyVLM, count_parameters
from tiny_vlm.quantize import (
    dequantize_model,
    pack_int4,
    quantize_model,
    quantize_weight,
    dequantize_weight,
    unpack_int4,
)
from tiny_vlm.utils import estimate_sizes


def small_config() -> VLMConfig:
    return VLMConfig(
        vision=VisionConfig(img_size=32, patch_size=16, d_model=32, depth=2, num_heads=4),
        text=TextConfig(vocab_size=64, max_seq_len=32, d_model=32, depth=2, num_heads=4),
    )


@pytest.fixture
def model() -> TinyVLM:
    torch.manual_seed(0)
    return TinyVLM(small_config())


def test_preset_fits_quantised_budget():
    cfg = get_config("tiny-50m")
    model = TinyVLM(cfg)
    stats = count_parameters(model)
    sizes = estimate_sizes(stats.total)
    assert sizes["int8"] < 50, f"int8 size {sizes['int8']:.1f} MB exceeds the 50 MB budget"
    assert sizes["int4"] < 25


def test_forward_shapes_and_loss(model):
    ids = torch.randint(0, 64, (2, 10))
    labels = ids.clone()
    pixels = torch.randn(2, 3, 32, 32)
    out = model(ids, pixel_values=pixels, labels=labels)
    assert out["logits"].shape == (2, 10, 64)
    assert out["n_image_tokens"] == 4  # (32/16)^2
    assert torch.isfinite(out["loss"])


def test_forward_without_images(model):
    ids = torch.randint(0, 64, (2, 10))
    out = model(ids, labels=ids.clone())
    assert out["n_image_tokens"] == 0
    assert out["logits"].shape == (2, 10, 64)


def test_masked_labels_are_ignored(model):
    ids = torch.randint(0, 64, (2, 10))
    masked = ids.clone()
    masked[:, :8] = -100
    full = model(ids, labels=ids.clone())["loss"]
    partial = model(ids, labels=masked)["loss"]
    assert torch.isfinite(full) and torch.isfinite(partial)
    assert full.item() != partial.item()


def test_padding_does_not_change_real_token_logits(model):
    """Right-padded sequences must produce identical logits at real positions."""
    ids = torch.randint(0, 64, (1, 5))
    mask = torch.ones(1, 5, dtype=torch.long)
    with torch.no_grad():
        bare = model(ids)["logits"]
        masked = model(ids, attention_mask=mask)["logits"]
    torch.testing.assert_close(bare, masked, atol=1e-5, rtol=1e-4)


def test_gradients_flow(model):
    ids = torch.randint(0, 64, (2, 6))
    out = model(ids, pixel_values=torch.randn(2, 3, 32, 32), labels=ids.clone())
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_generate_extends_sequence(model):
    ids = torch.randint(0, 64, (1, 4))
    out = model.generate(ids, pixel_values=torch.randn(1, 3, 32, 32), max_new_tokens=5, temperature=0.0)
    assert out.shape == (1, 9)


def test_multi_image_prefix(model):
    ids = torch.randint(0, 64, (2, 6))
    out = model(ids, pixel_values=torch.randn(2, 3, 3, 32, 32))
    assert out["n_image_tokens"] == 12


def test_quantize_weight_roundtrip_is_close():
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    codes, scales, orig_in = quantize_weight(w, bits=8, group_size=64)
    restored = dequantize_weight(codes, scales, orig_in, group_size=64)
    assert restored.shape == w.shape
    assert (restored - w).abs().mean() < 0.05


def test_int4_packing_roundtrip():
    codes = torch.randint(-8, 8, (4, 7), dtype=torch.int8)
    unpacked = unpack_int4(pack_int4(codes))
    torch.testing.assert_close(unpacked[:, :7], codes)


def test_quantized_checkpoint_reloads(model):
    cfg = model.cfg
    payload = quantize_model(model, bits=8, group_size=32)
    restored = dequantize_model(payload, cfg)

    model.eval()
    ids = torch.randint(0, 64, (1, 8))
    pixels = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        before = model(ids, pixel_values=pixels)["logits"]
        after = restored(ids, pixel_values=pixels)["logits"]
    assert before.shape == after.shape
    assert (before - after).abs().max() < 0.5


def test_tied_embeddings_are_stored_once():
    """A tied lm_head must not double the embedding's bytes in the checkpoint."""
    model = TinyVLM(get_config("tiny-50m"))
    payload = quantize_model(model, bits=8, group_size=128)
    assert "token_embedding.weight" in payload["quantized"]
    assert "lm_head.weight" not in payload["quantized"]
    assert "lm_head.weight" not in payload["fp16"]
    # 28.2M params at int8 plus scales and fp16 leftovers must stay under budget.
    size = sum(
        e["codes"].numel() * e["codes"].element_size() + e["scales"].numel() * 2
        for e in payload["quantized"].values()
    ) + sum(t.numel() * t.element_size() for t in payload["fp16"].values())
    assert size / 1024**2 < 50


def test_int4_checkpoint_is_smaller_than_int8(model):
    int8 = quantize_model(model, bits=8, group_size=32)
    int4 = quantize_model(model, bits=4, group_size=32)
    int8_bytes = sum(t.numel() * t.element_size() for t in int8["fp16"].values()) + sum(
        e["codes"].numel() * e["codes"].element_size() + e["scales"].numel() * 2
        for e in int8["quantized"].values()
    )
    int4_bytes = sum(t.numel() * t.element_size() for t in int4["fp16"].values()) + sum(
        e["codes"].numel() * e["codes"].element_size() + e["scales"].numel() * 2
        for e in int4["quantized"].values()
    )
    assert int4_bytes < int8_bytes
