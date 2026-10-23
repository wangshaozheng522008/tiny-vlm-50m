"""Training entry point.

    python -m tiny_vlm.train --config tiny-50m --data data/captions.jsonl --epochs 3
    python -m tiny_vlm.train --config nano-50m --synthetic 256 --epochs 1   # smoke test
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import get_config
from .data import CaptionDataset, SyntheticCaptionDataset, collate_fn
from .model import TinyVLM, count_parameters
from .tokenizer import TinyTokenizer
from .utils import estimate_sizes, get_device, set_seed


def build_tokenizer(args, cache_dir: Path) -> TinyTokenizer:
    path = Path(args.tokenizer) if args.tokenizer else cache_dir / "tokenizer.json"
    if path.exists():
        return TinyTokenizer.load(path)
    if not args.text_corpus:
        raise SystemExit(
            f"no tokenizer at {path}; pass --tokenizer or --text-corpus to train one"
        )
    corpus = [line for line in Path(args.text_corpus).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[tok] training BPE with vocab_size={args.vocab_size} on {len(corpus)} lines")
    tok = TinyTokenizer.train(corpus, vocab_size=args.vocab_size, save_path=path)
    return tok


def build_dataset(args, tokenizer: TinyTokenizer, cfg):
    if args.synthetic:
        return SyntheticCaptionDataset(
            size=args.synthetic,
            tokenizer=tokenizer,
            img_size=cfg.vision.img_size,
            max_seq_len=cfg.text.max_seq_len,
            ignore_index=cfg.ignore_index,
        )
    if not args.data:
        raise SystemExit("pass --data path/to/captions.jsonl, or --synthetic N for a smoke test")
    return CaptionDataset(
        jsonl_path=args.data,
        tokenizer=tokenizer,
        img_size=cfg.vision.img_size,
        max_seq_len=cfg.text.max_seq_len,
        ignore_index=cfg.ignore_index,
    )


def lr_at(step: int, total: int, peak: float, warmup: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return peak * (min_ratio + (1 - min_ratio) * cos)


def main() -> None:
    p = argparse.ArgumentParser(description="Train TinyVLM")
    p.add_argument("--config", default="tiny-50m", help="preset name (see tiny_vlm.config.PRESETS)")
    p.add_argument("--data", help="JSONL of {image, caption}")
    p.add_argument("--synthetic", type=int, default=0, help="use N synthetic samples instead of --data")
    p.add_argument("--tokenizer", help="path to tokenizer.json")
    p.add_argument("--text-corpus", help="plain-text file to train the BPE tokenizer on")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--out", default="runs/tiny-vlm")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=0, help="stop early (0 = full run)")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_config(args.config)
    tokenizer = build_tokenizer(args, out_dir)
    cfg.text.vocab_size = tokenizer.vocab_size

    model = TinyVLM(cfg).to(device)
    stats = count_parameters(model)
    sizes = estimate_sizes(stats.total)
    print(
        f"[model] {stats.total / 1e6:.1f}M params "
        f"(vision {stats.vision / 1e6:.1f}M / text {stats.text / 1e6:.1f}M) | "
        f"int8 {sizes['int8']:.1f} MB, int4 {sizes['int4']:.1f} MB "
        f"(budget {50} MB)"
    )
    if sizes["int8"] > 50:
        print("[warn] INT8 size exceeds the 50 MB budget; consider a smaller preset or vocab")

    dataset = build_dataset(args, tokenizer, cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=lambda b: collate_fn(b, cfg.pad_token_id),
        drop_last=len(dataset) > args.batch_size,
    )

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.dim() < 2 else decay).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    step = 0
    t0 = time.time()
    done = False
    for epoch in range(args.epochs):
        running, seen = 0.0, 0
        for i, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(
                    input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )["loss"]
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()

            running += loss.item() * args.grad_accum
            seen += 1

            if (i + 1) % args.grad_accum == 0:
                lr = lr_at(step, total_steps, args.lr, args.warmup)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0:
                    print(
                        f"epoch {epoch + 1}/{args.epochs} step {step}/{total_steps} "
                        f"loss {running / seen:.4f} lr {lr:.2e} "
                        f"{(time.time() - t0) / step:.2f}s/step"
                    )
                    running, seen = 0.0, 0

            if args.max_steps and step >= args.max_steps:
                done = True
                break
        if done:
            break

    ckpt = out_dir / "model.pt"
    torch.save(
        {"model": model.state_dict(), "config": cfg, "step": step},
        ckpt,
    )
    cfg.save(out_dir / "config.json")
    print(f"[done] saved {ckpt} ({ckpt.stat().st_size / 1024**2:.1f} MB fp32 checkpoint)")
    print(f"[note] run `python -m tiny_vlm.quantize --ckpt {ckpt}` to shrink it ~4x (int8)")


if __name__ == "__main__":
    main()