import pytest
import torch

from tiny_vlm.config import TextConfig, VisionConfig, VLMConfig, get_config
from tiny_vlm.model import TinyVLM, count_parameters
from tiny_vlm.quantize import (
    dequantize_model,
    dequantize_weight,
    effective_group,
    load_quantized,
    pack_int4,
    quantize_model,
    quantize_weight,
    save_quantized,
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


def test_effective_group_picks_a_divisor():
    # 448 = 2**6 * 7; 112 is the largest divisor <= 128, and 112 > 128/2.
    assert effective_group(448, 128) == 112
    assert effective_group(1792, 128) == 128  # already divides
    assert effective_group(64, 128) == 64  # group_size above in_features
    assert 448 % effective_group(448, 128) == 0


def test_quantize_weight_does_not_pad_when_a_divisor_exists():
    """Padding 448 -> 512 wasted 14% of every such weight and blew the budget."""
    w = torch.randn(32, 448)
    codes, scales, orig_in, group = quantize_weight(w, bits=8, group_size=128)
    assert orig_in == 448
    assert group == 112
    assert codes.shape == (32, 448), f"codes were padded to {codes.shape}"
    assert scales.shape == (32, 4)


def test_quantize_weight_roundtrip_is_close():
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    codes, scales, orig_in, group = quantize_weight(w, bits=8, group_size=64)
    restored = dequantize_weight(codes, scales, orig_in, group)
    assert restored.shape == w.shape
    assert (restored - w).abs().mean() < 0.05


def test_quantize_weight_falls_back_to_padding_for_awkward_shapes():
    """A prime-ish in_features has no divisor in range, so padding is the fallback."""
    w = torch.randn(8, 131)
    codes, scales, orig_in, group = quantize_weight(w, bits=8, group_size=128)
    assert orig_in == 131
    assert codes.shape == (8, 256)  # padded to a multiple of the group
    restored = dequantize_weight(codes, scales, orig_in, group)
    assert restored.shape == (8, 131)
    assert (restored - w).abs().mean() < 0.1


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


def test_quantize_cli_verifies_on_the_models_own_device(tmp_path, monkeypatch):
    """Regression: the CLI built its probe tensors on CPU while `load_quantized` placed the
    model on CUDA, so the verification forward pass blew up on any GPU machine. Invisible on
    CPU-only CI, which is exactly why it needs a test."""
    import sys

    from tiny_vlm import quantize

    cfg = small_config()
    ckpt = tmp_path / "model.pt"
    torch.save({"model": TinyVLM(cfg).state_dict(), "config": cfg, "step": 0}, ckpt)
    out = tmp_path / "model-int8.pt"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quantize", "--ckpt", str(ckpt), "--out", str(out),
            "--bits", "8", "--group-size", "32",
        ],
    )
    quantize.main()
    assert out.exists()


def test_load_quantized_honours_an_explicit_device(model, tmp_path):
    from tiny_vlm.quantize import load_quantized

    path = tmp_path / "model-int8.pt"
    save_quantized(path, model, model.cfg, bits=8, group_size=32)
    reloaded = load_quantized(path, device="cpu")
    assert next(reloaded.parameters()).device.type == "cpu"
    # Round-trip a forward pass with inputs on the same device as the model.
    with torch.no_grad():
        out = reloaded(torch.randint(0, 64, (1, 6), device="cpu"), pixel_values=torch.randn(1, 3, 32, 32))
    assert out["logits"].shape == (1, 6, 64)


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
