"""Train one decoder on text-only, image-only, and image+text samples."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from functools import partial
from itertools import chain, islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import get_config
from .data import (
    MODES, MultimodalDataset, SyntheticMultimodalDataset, collate_fn,
    read_records, record_texts, synthetic_texts,
)
from .generate import load_model
from .model import TinyVLM, count_parameters
from .tokenizer import TinyTokenizer
from .utils import describe_device, estimate_sizes, get_device, peak_memory_mb, set_seed


def build_tokenizer(args, cache_dir: Path, records=None) -> TinyTokenizer:
    path = Path(args.tokenizer) if args.tokenizer else cache_dir / "tokenizer.json"
    if path.exists():
        return TinyTokenizer.load(path)
    if args.tokenizer:
        raise ValueError(f"tokenizer does not exist: {path}")
    texts = record_texts(records) if records is not None else synthetic_texts()
    if args.text_corpus:
        corpus = Path(args.text_corpus).read_text(encoding="utf-8").splitlines()
        texts = chain(texts, (line for line in corpus if line.strip()))
    print(f"[tok] training BPE on training inputs and answers, vocab_size<={args.vocab_size}")
    return TinyTokenizer.train(texts, vocab_size=args.vocab_size, save_path=path)


def build_dataset(args, tokenizer, cfg, records=None):
    options = dict(tokenizer=tokenizer, img_size=cfg.vision.img_size,
                   max_seq_len=cfg.text.max_seq_len, ignore_index=cfg.ignore_index)
    if args.synthetic:
        return SyntheticMultimodalDataset(
            size=args.synthetic, modes=args.synthetic_modes, seed=args.seed, **options
        )
    return MultimodalDataset(args.data, records=records, **options)


def make_sampler(modes, weights, seed):
    """Optional task sampling: weights specify mode probabilities, not per-row weights."""
    if weights is None:
        return None
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("modality weights must be finite and nonnegative")
    counts = Counter(modes)
    by_mode = dict(zip(MODES, weights))
    row_weights = [by_mode[mode] / counts[mode] for mode in modes]
    if not any(row_weights):
        raise ValueError("at least one mode present in the data must have positive weight")
    return WeightedRandomSampler(row_weights, num_samples=len(modes), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def lr_at(step: int, total: int, peak: float, warmup: int, min_ratio: float = 0.1) -> float:
    warmup = min(warmup, max(total - 1, 0))
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup - 1, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return peak * (min_ratio + (1 - min_ratio) * cos)


def update_batches(loader, grad_accum):
    iterator = iter(loader)
    while True:
        batches = list(islice(iterator, grad_accum))
        if not batches:
            return
        yield batches


def model_inputs(batch, device):
    return {key: value.to(device, non_blocking=True)
            for key, value in batch.items() if key != "mode_id"}


def train_update(model, batches, optimizer, scaler, device):
    """Normalise by actual target tokens across the entire accumulation window."""
    tokens = sum(int((batch["labels"][:, 1:] != model.cfg.ignore_index).sum()) for batch in batches)
    if tokens == 0:
        raise ValueError("training update contains no supervised answer tokens")
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for batch in batches:
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(**model_inputs(batch, device))
            loss = out["loss_sum"] / tokens
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite training loss")
        scaler.scale(loss).backward()
        total_loss += float(out["loss_sum"].detach())
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return total_loss, tokens


@torch.no_grad()
def evaluate(model, loader, device):
    was_training = model.training
    model.eval()
    totals = {mode: {"loss_sum": 0.0, "tokens": 0, "correct": 0, "samples": 0} for mode in MODES}
    try:
        for batch in loader:
            mode_ids = batch["mode_id"].to(device)
            inputs = model_inputs(batch, device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(**inputs)["logits"]
            labels = inputs["labels"]
            for mode_id, mode in enumerate(MODES):
                rows = mode_ids == mode_id
                if not bool(rows.any()):
                    continue
                loss_sum, n_tokens = model.loss_terms(logits[rows], labels[rows])
                valid = labels[rows, 1:] != model.cfg.ignore_index
                correct = (logits[rows, :-1].argmax(-1) == labels[rows, 1:]) & valid
                totals[mode]["loss_sum"] += float(loss_sum)
                totals[mode]["tokens"] += int(n_tokens)
                totals[mode]["correct"] += int(correct.sum())
                totals[mode]["samples"] += int(rows.sum())
    finally:
        model.train(was_training)
    metrics = {
        mode: {"loss": t["loss_sum"] / t["tokens"],
               "token_accuracy": t["correct"] / t["tokens"],
               "samples": t["samples"], "tokens": t["tokens"]}
        for mode, t in totals.items() if t["tokens"]
    }
    if not metrics:
        raise ValueError("validation data contains no supervised tokens")
    # Give each evaluated mode equal influence on best-checkpoint selection.
    return {"macro_loss": sum(m["loss"] for m in metrics.values()) / len(metrics), "modes": metrics}


def save_checkpoint(path, model, tokenizer, step, dataset_modes):
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": model.cfg, "step": step,
                "tokenizer_fingerprint": tokenizer.fingerprint,
                "training_modes": dict(dataset_modes)}, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="tiny-50m")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", nargs="+", help="one or more mixed-mode JSONL files")
    source.add_argument("--synthetic", type=int, default=0, help="N grounded toy examples for smoke testing")
    p.add_argument("--synthetic-modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--val-data", nargs="+", help="held-out JSONL files; never used to train the tokenizer")
    p.add_argument("--tokenizer")
    p.add_argument("--text-corpus", help="extra plain text for BPE training, not model training")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--init-from", help="warm-start model weights; starts a new optimiser and LR schedule")
    p.add_argument("--freeze-vision", action="store_true", help="freeze the warm-started ViT, keep projector trainable")
    p.add_argument("--modality-weights", nargs=3, type=float, metavar=("TEXT", "IMAGE", "MIXED"),
                   help="relative mode sampling probabilities, e.g. 1 1 1; default: natural data ratio")
    p.add_argument("--out", default="runs/tiny-vlm")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=0, help="optimiser updates, 0 = full run")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.grad_accum, args.log_every) <= 0:
        p.error("epochs, batch-size, grad-accum, and log-every must be positive")
    if min(args.synthetic, args.max_steps, args.workers, args.warmup) < 0:
        p.error("synthetic, max-steps, workers, and warmup must be nonnegative")
    if args.synthetic == 0 and not args.data:
        p.error("synthetic must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        p.error("lr must be positive and finite")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error("weight-decay must be finite and nonnegative")
    if args.freeze_vision and not args.init_from:
        p.error("freeze-vision requires --init-from to avoid freezing random visual weights")

    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[device] {describe_device(device)}")
    records = read_records(args.data) if args.data else None
    if args.init_from:
        model, cfg = load_model(args.init_from, device)
        tok_path = Path(args.tokenizer) if args.tokenizer else Path(args.init_from).parent / "tokenizer.json"
        if not tok_path.exists():
            p.error("warm-start requires the checkpoint's original tokenizer; supply --tokenizer")
        tokenizer = TinyTokenizer.load(tok_path)
        payload = torch.load(args.init_from, map_location="cpu", weights_only=False)
        expected = payload.get("tokenizer_fingerprint")
        sibling = Path(args.init_from).parent / "tokenizer.json"
        if expected is None and sibling.exists():
            expected = TinyTokenizer.load(sibling).fingerprint
        if expected is not None and expected != tokenizer.fingerprint:
            p.error("tokenizer vocabulary or merge rules differ from the warm-start checkpoint")
        if tokenizer.vocab_size != cfg.text.vocab_size:
            p.error("tokenizer vocabulary size differs from the warm-start checkpoint")
        del payload
        print(f"[init] loaded weights from {args.init_from}; starting a new training schedule")
    else:
        cfg = get_config(args.config)
        tokenizer = build_tokenizer(args, out_dir, records)
        cfg.text.vocab_size = tokenizer.vocab_size
        model = TinyVLM(cfg).to(device)
    tokenizer.save(out_dir / "tokenizer.json")
    if args.freeze_vision:
        model.vision.requires_grad_(False)
    stats = count_parameters(model)
    sizes = estimate_sizes(stats.total)
    print(f"[model] {stats.total / 1e6:.2f}M params | estimated INT8 {sizes['int8']:.1f} MB")
    if sizes["int8"] > 50:
        print("[warn] estimated INT8 weights exceed 50 MB; reduce model or vocabulary size")

    dataset = build_dataset(args, tokenizer, cfg, records)
    print(f"[data] {dataset.mode_counts}")
    missing = set(MODES) - set(dataset.modes)
    if missing:
        print(f"[data] no training samples for: {', '.join(sorted(missing))}")
    sampler = make_sampler(dataset.modes, args.modality_weights, args.seed)
    collate = partial(collate_fn, pad_token_id=cfg.pad_token_id, ignore_index=cfg.ignore_index)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          collate_fn=collate, pin_memory=device.type == "cuda")
    loader = DataLoader(dataset, shuffle=sampler is None, sampler=sampler, **loader_options)
    val_loader = None
    if args.val_data:
        val_dataset = MultimodalDataset(args.val_data, tokenizer, cfg.vision.img_size,
                                        cfg.text.max_seq_len, cfg.ignore_index)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
        print(f"[val data] {val_dataset.mode_counts}")

    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (no_decay if param.dim() < 2 else decay).append(param)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_loss = float("inf")
    step = 0
    t0 = time.time()
    model.train()
    for epoch in range(args.epochs):
        epoch_loss, epoch_tokens = 0.0, 0
        for batches in update_batches(loader, args.grad_accum):
            lr = lr_at(step, total_steps, args.lr, args.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss_sum, tokens = train_update(model, batches, optimizer, scaler, device)
            epoch_loss += loss_sum
            epoch_tokens += tokens
            step += 1
            if step % args.log_every == 0 or step == total_steps:
                print(f"epoch {epoch + 1}/{args.epochs} step {step}/{total_steps} "
                      f"loss {epoch_loss / epoch_tokens:.4f} lr {lr:.2e} "
                      f"{(time.time() - t0) / step:.2f}s/step")
            if step >= total_steps:
                break
        entry = {"epoch": epoch + 1, "step": step, "train_loss": epoch_loss / epoch_tokens}
        if val_loader is not None:
            validation = evaluate(model, val_loader, device)
            entry["validation"] = validation
            print(f"[val] {json.dumps(validation, ensure_ascii=False)}")
            if validation["macro_loss"] < best_loss:
                best_loss = validation["macro_loss"]
                save_checkpoint(out_dir / "best.pt", model, tokenizer, step, dataset.mode_counts)
        history.append(entry)
        save_checkpoint(out_dir / "model.pt", model, tokenizer, step, dataset.mode_counts)
        (out_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if step >= total_steps:
            break
    cfg.save(out_dir / "config.json")
    print(f"[done] saved {out_dir / 'model.pt'}, tokenizer.json, config.json, metrics.json")
    peak = peak_memory_mb(device)
    if peak is not None:
        print(f"[mem] peak GPU allocation {peak:.0f} MB")


if __name__ == "__main__":
    main()
