"""Weight-only quantisation, used to fit the model into the 50 MB budget.

Every 2-D weight of a `nn.Linear` / `nn.Embedding` is quantised symmetrically with
per-group scales; everything else (norms, biases, position embeddings) stays in fp16.
4-bit codes are packed two-per-byte so the reported size is a real file size, not an
estimate.

The quantised checkpoint stores codes plus scales. `load_quantized` dequantises back
into a regular `TinyVLM`, so inference needs no custom kernels — the model is small
enough that fp32 activations are not the bottleneck.

    python -m tiny_vlm.quantize --ckpt runs/tiny-vlm/model.pt --bits 8
    python -m tiny_vlm.quantize --ckpt runs/tiny-vlm/model.pt --bits 4 --group-size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VLMConfig
from .model import TinyVLM
from .utils import get_device

QMAX = {4: 7, 8: 127}


def _quantizable_names(model: nn.Module) -> set[str]:
    """Parameter names whose tensors are 2-D lookup/projection weights."""
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            weight = getattr(module, "weight", None)
            if weight is not None and weight.dim() == 2:
                names.add(f"{module_name}.weight" if module_name else "weight")
    return names


def _storage_key(tensor: torch.Tensor) -> tuple:
    """Identity for weight tying.

    `state_dict()` hands back distinct tensor objects for tied parameters, so
    `id(tensor)` does not deduplicate them — `data_ptr()` does. Without this, a tied
    embedding is written twice and the checkpoint is ~20% larger than it needs to be.
    """
    return (tensor.data_ptr(), tensor.storage_offset(), tuple(tensor.shape))


def quantize_weight(weight: torch.Tensor, bits: int = 8, group_size: int = 128):
    """Symmetric per-group quantisation along the input dimension.

    Returns `(codes, scales, orig_in_features)`. `codes` may be padded so the last
    dimension divides evenly by `group_size`.
    """
    if bits not in QMAX:
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    w = weight.detach().float()
    out_f, in_f = w.shape
    group = min(group_size, in_f) if group_size > 0 else in_f
    pad = (group - in_f % group) % group
    if pad:
        w = F.pad(w, (0, pad))

    grouped = w.reshape(out_f, -1, group)
    qmax = QMAX[bits]
    scales = (grouped.abs().amax(dim=-1, keepdim=True) / qmax).clamp_(min=1e-8)
    codes = torch.round(grouped / scales).clamp_(-qmax - 1, qmax).to(torch.int8)
    return codes.reshape(out_f, -1), scales.squeeze(-1), in_f


def dequantize_weight(codes: torch.Tensor, scales: torch.Tensor, orig_in: int, group_size: int) -> torch.Tensor:
    out_f, padded_in = codes.shape
    group = min(group_size, orig_in) if group_size > 0 else orig_in
    grouped = codes.float().reshape(out_f, -1, group)
    if scales.shape[1] != grouped.shape[1]:
        raise ValueError(f"scale shape {tuple(scales.shape)} does not match codes {tuple(codes.shape)}")
    w = (grouped * scales.unsqueeze(-1)).reshape(out_f, padded_in)
    return w[:, :orig_in]


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack int4 codes in [-8, 7] two-per-byte along the last dimension."""
    flat = (codes.to(torch.int16) + 8).to(torch.uint8)
    if flat.shape[-1] % 2:
        flat = F.pad(flat, (0, 1))
    return (flat[..., 0::2] | (flat[..., 1::2] << 4)).to(torch.uint8)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    lo = (packed & 0x0F).to(torch.int16) - 8
    hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    interleaved = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return interleaved.to(torch.int8)


def quantize_model(model: TinyVLM, bits: int = 8, group_size: int = 128) -> dict:
    """Split the model state into quantised and full-precision parts."""
    names = _quantizable_names(model)
    state = model.state_dict()
    quantized: dict[str, dict] = {}
    fp16: dict[str, torch.Tensor] = {}
    captured: set[tuple] = set()

    for name, tensor in state.items():
        key = _storage_key(tensor)
        if name not in names or key in captured:
            continue
        captured.add(key)
        codes, scales, orig_in = quantize_weight(tensor, bits, group_size)
        quantized[name] = {
            "codes": pack_int4(codes) if bits == 4 else codes,
            "scales": scales.to(torch.float16),
            "orig_in": orig_in,
            "shape": tuple(tensor.shape),
            "bits": bits,
        }

    # Anything not captured above stays in fp16. A tied alias (e.g. `lm_head.weight`
    # backed by `token_embedding.weight`) shares storage with an already-captured tensor
    # and is skipped here, then restored from its alias at load time.
    for name, tensor in state.items():
        if _storage_key(tensor) not in captured:
            fp16[name] = tensor.to(torch.float16)

    return {"quantized": quantized, "fp16": fp16, "bits": bits, "group_size": group_size}


def dequantize_model(payload: dict, cfg: VLMConfig) -> TinyVLM:
    """Rebuild a full-precision `TinyVLM` from a quantised payload."""
    model = TinyVLM(cfg)
    target = model.state_dict()
    merged: dict[str, torch.Tensor] = {}
    seen: set[str] = set()

    for name, entry in payload["quantized"].items():
        codes = entry["codes"]
        if entry["bits"] == 4:
            codes = unpack_int4(codes)
        weight = dequantize_weight(codes, entry["scales"].float(), entry["orig_in"], payload["group_size"])
        merged[name] = weight.to(target[name].dtype).reshape(entry["shape"])
        seen.add(name)

    for name, tensor in payload["fp16"].items():
        if name in seen:
            continue
        merged[name] = tensor.to(target[name].dtype)
        seen.add(name)

    # Tied aliases are absent from the payload; recover them by matching storage in the
    # freshly built model, which ties its own weights the same way.
    for name in [k for k in target if k not in merged]:
        alias = next(
            (
                other
                for other in merged
                if other in target and target[other].data_ptr() == target[name].data_ptr()
            ),
            None,
        )
        if alias is None:
            raise KeyError(f"checkpoint is missing parameter {name!r}")
        merged[name] = merged[alias]

    model.load_state_dict(merged)
    model.eval()
    return model


def save_quantized(path: str | Path, model: TinyVLM, cfg: VLMConfig, bits: int = 8, group_size: int = 128) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = quantize_model(model, bits, group_size)
    payload["config"] = cfg
    torch.save(payload, path)
    return path


def load_quantized(path: str | Path, device: str | None = None) -> TinyVLM:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = dequantize_model(payload, payload["config"])
    return model.to(get_device(device))


def main() -> None:
    p = argparse.ArgumentParser(description="Quantise a TinyVLM checkpoint")
    p.add_argument("--ckpt", default="runs/tiny-vlm/model.pt", help="fp32 checkpoint from train.py")
    p.add_argument("--out", help="output path (default: alongside --ckpt)")
    p.add_argument("--bits", type=int, default=8, choices=[4, 8])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--budget-mb", type=float, default=50.0)
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: VLMConfig = ckpt["config"]
    model = TinyVLM(cfg)
    model.load_state_dict(ckpt["model"])

    out = Path(args.out) if args.out else ckpt_path.with_name(f"model-int{args.bits}.pt")
    save_quantized(out, model, cfg, args.bits, args.group_size)

    size_mb = out.stat().st_size / 1024**2
    verdict = "OK" if size_mb <= args.budget_mb else "OVER BUDGET"
    print(f"int{args.bits} (group_size={args.group_size}) -> {out}")
    print(f"  {size_mb:.2f} MB / {args.budget_mb:.0f} MB budget  [{verdict}]")

    reloaded = load_quantized(out)
    with torch.no_grad():
        probe = torch.randint(0, cfg.text.vocab_size, (1, 8))
        pixels = torch.randn(1, cfg.vision.in_channels, cfg.vision.img_size, cfg.vision.img_size)
        logits = reloaded(probe, pixel_values=pixels)["logits"]
    print(f"  reloaded and ran a forward pass: logits {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
