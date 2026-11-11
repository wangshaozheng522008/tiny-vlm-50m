"""Weight-only INT4/INT8 quantisation for MiniLLM.

Every 2-D weight of a Linear/Embedding is quantised symmetrically with per-group
scales and, for 4-bit, packed two codes per byte. Norms stay in fp16, the tied
LM head is stored once, and RoPE tables are never saved because they are derived
from the config.

`effective_group` shrinks the requested group to a divisor of `in_features`
instead of zero-padding the weight. That matters for a budget that is checked
against the real file size:

    python -m minillm.quantize --ckpt runs/minillm/model.pt
    python -m minillm.quantize --ckpt runs/minillm/model.pt --bits 4 --group-size 128

The saved file is stat()ed, reloaded, run, and compared against the fp32 logits;
a file at or above the budget is a hard failure, not a warning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ARCH_VERSION, ARCHITECTURE, MiniLLMConfig
from .model import MiniLLM
from .utils import get_device, human_bytes

QMAX = {4: 7, 8: 127}
BUDGET_BYTES = 50_000_000


def _quantizable_names(model: nn.Module) -> set[str]:
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            weight = getattr(module, "weight", None)
            if weight is not None and weight.dim() == 2:
                names.add(f"{module_name}.weight" if module_name else "weight")
    return names


def _storage_key(tensor: torch.Tensor) -> tuple:
    """Identity for weight tying: state_dict aliases share storage, not objects."""
    return (tensor.data_ptr(), tensor.storage_offset(), tuple(tensor.shape))


def effective_group(in_f: int, group_size: int) -> int:
    """Largest divisor of `in_f` within (group_size/2, group_size], else 0."""
    if group_size <= 0 or group_size >= in_f:
        return in_f
    for candidate in range(group_size, max(group_size // 2, 1) - 1, -1):
        if in_f % candidate == 0:
            return candidate
    return 0


def quantize_weight(weight: torch.Tensor, bits: int = 4, group_size: int = 128):
    if bits not in QMAX:
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    w = weight.detach().float()
    out_f, in_f = w.shape
    group = effective_group(in_f, group_size) or group_size
    pad = (group - in_f % group) % group
    if pad:
        w = F.pad(w, (0, pad))
    grouped = w.reshape(out_f, -1, group)
    qmax = QMAX[bits]
    scales = (grouped.abs().amax(dim=-1, keepdim=True) / qmax).clamp_(min=1e-8)
    codes = torch.round(grouped / scales).clamp_(-qmax - 1, qmax).to(torch.int8)
    return codes.reshape(out_f, -1), scales.squeeze(-1), in_f, group


def dequantize_weight(codes: torch.Tensor, scales: torch.Tensor, orig_in: int, group: int) -> torch.Tensor:
    out_f, padded_in = codes.shape
    grouped = codes.float().reshape(out_f, -1, group)
    if scales.shape[1] != grouped.shape[1]:
        raise ValueError(f"scale shape {tuple(scales.shape)} does not match codes {tuple(codes.shape)}")
    return (grouped * scales.unsqueeze(-1)).reshape(out_f, padded_in)[:, :orig_in]


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack int4 codes in [-8, 7] two per byte along the last dimension."""
    flat = (codes.to(torch.int16) + 8).to(torch.uint8)
    if flat.shape[-1] % 2:
        flat = F.pad(flat, (0, 1))
    return (flat[..., 0::2] | (flat[..., 1::2] << 4)).to(torch.uint8)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    lo = (packed & 0x0F).to(torch.int16) - 8
    hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    interleaved = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return interleaved.to(torch.int8)


def quantize_model(model: MiniLLM, bits: int = 4, group_size: int = 128) -> dict:
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
        codes, scales, orig_in, group = quantize_weight(tensor, bits, group_size)
        quantized[name] = {
            "codes": pack_int4(codes) if bits == 4 else codes,
            "scales": scales.to(torch.float16),
            "orig_in": orig_in,
            "group": group,
            "shape": tuple(tensor.shape),
            "bits": bits,
        }
    # Anything not captured (RMSNorm weights) stays fp16. The tied LM head shares
    # storage with the embedding, so it is skipped here and restored by alias.
    for name, tensor in state.items():
        if _storage_key(tensor) not in captured:
            fp16[name] = tensor.to(torch.float16)
    return {"quantized": quantized, "fp16": fp16, "bits": bits, "group_size": group_size}


def dequantize_model(payload: dict) -> MiniLLM:
    cfg = MiniLLMConfig.from_dict(payload["config"])
    model = MiniLLM(cfg)
    target = model.state_dict()
    merged: dict[str, torch.Tensor] = {}

    for name, entry in payload["quantized"].items():
        codes = entry["codes"]
        if entry["bits"] == 4:
            codes = unpack_int4(codes)
        weight = dequantize_weight(codes, entry["scales"].float(), entry["orig_in"], entry["group"])
        merged[name] = weight.to(target[name].dtype).reshape(entry["shape"])
    for name, tensor in payload["fp16"].items():
        if name not in merged:
            merged[name] = tensor.to(target[name].dtype)
    # Tied aliases are absent from the payload; recover them by storage identity.
    for name in [key for key in target if key not in merged]:
        alias = next(
            (other for other in merged if target[other].data_ptr() == target[name].data_ptr()), None
        )
        if alias is None:
            raise KeyError(f"checkpoint is missing parameter {name!r}")
        merged[name] = merged[alias]
    model.load_state_dict(merged)
    model.eval()
    return model


def save_quantized(path: str | Path, model: MiniLLM, bits: int = 4, group_size: int = 128,
                   *, tokenizer_fingerprint: str | None = None, metadata: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = quantize_model(model, bits, group_size)
    payload.update(
        {
            "architecture": ARCHITECTURE,
            "arch_version": ARCH_VERSION,
            "config": model.cfg.to_dict(),
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "quantized_bits": bits,
        }
    )
    if metadata:
        for key in ("step", "tokens_seen", "mode", "val_loss"):
            if key in metadata:
                payload[key] = metadata[key]
    torch.save(payload, path)
    return path


def load_quantized(path: str | Path, device: str | None = None) -> MiniLLM:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("architecture") != ARCHITECTURE:
        raise RuntimeError(
            f"{path}: not a MiniLLM quantised checkpoint (architecture={payload.get('architecture')!r})"
        )
    model = dequantize_model(payload)
    return model.to(get_device(device))


def main() -> None:
    p = argparse.ArgumentParser(description="Quantise a MiniLLM checkpoint")
    p.add_argument("--ckpt", required=True, help="fp32 MiniLLM checkpoint")
    p.add_argument("--out", help="output path (default: <ckpt>-int4.pt)")
    p.add_argument("--bits", type=int, default=4, choices=[4, 8])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--budget-bytes", type=int, default=BUDGET_BYTES)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    ckpt = torch.load(Path(args.ckpt), map_location="cpu", weights_only=False)
    if ckpt.get("architecture") != ARCHITECTURE:
        raise SystemExit(f"{args.ckpt}: not a MiniLLM checkpoint")
    cfg = MiniLLMConfig.from_dict(ckpt["config"])
    model = MiniLLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out = Path(args.out) if args.out else Path(args.ckpt).with_name(f"model-int{args.bits}.pt")
    save_quantized(out, model, args.bits, args.group_size,
                   tokenizer_fingerprint=ckpt.get("tokenizer_fingerprint"), metadata=ckpt)
    size = out.stat().st_size
    groups = sorted({e["group"] for e in torch.load(out, map_location="cpu", weights_only=False)["quantized"].values()})
    print(f"int{args.bits} (group_size<={args.group_size}, effective {groups}) -> {out}")
    print(f"  {human_bytes(size)} / budget {human_bytes(args.budget_bytes)}")
    if size >= args.budget_bytes:
        raise SystemExit(f"FAIL: {size:,} bytes is not below the {args.budget_bytes:,}-byte budget")

    device = get_device(args.device)
    reloaded = load_quantized(out, args.device)
    probe = torch.randint(4, cfg.vocab_size, (1, 16), device=device)
    with torch.no_grad():
        q_logits = reloaded(probe).logits.float()
        fp_logits = model.to(device)(probe).logits.float()
    diff = (q_logits - fp_logits).abs()
    agree = (q_logits.argmax(-1) == fp_logits.argmax(-1)).float().mean().item()
    print(f"  reloaded on {next(reloaded.parameters()).device}: logits {tuple(q_logits.shape)}")
    print(f"  max|Δlogit| {diff.max():.4f} | mean|Δlogit| {diff.mean():.4f} | argmax agreement {agree:.1%}")
    if not (reloaded.lm_head.weight.data_ptr() == reloaded.embed_tokens.weight.data_ptr()):
        raise SystemExit("FAIL: LM head and embedding no longer share storage")
    print("  weight tying preserved; LM head stored once")
    generated = reloaded.generate(probe[:, :4], max_new_tokens=8, temperature=0.0)
    print(f"  smoke generation: {generated.shape[1] - 4} tokens")


if __name__ == "__main__":
    main()
