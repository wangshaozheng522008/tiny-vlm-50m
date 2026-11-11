"""Pretraining for MiniLLM, plus the training loop shared with SFT.

The loop is deliberately boring: token-weighted cross entropy, AdamW, cosine
learning rate with warmup, gradient clipping, gradient accumulation, optional
AMP, optional activation checkpointing, periodic checkpoints and a resumable
state file (weights + optimizer + scaler + RNG + dataloader position).

Checkpoints carry an architecture tag; a TinyVLM (vision-language) checkpoint
is rejected with an explanation instead of being half-loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import ARCH_VERSION, ARCHITECTURE, MiniLLMConfig, get_config
from .data import EpochBatchSampler, PackedPretrainDataset, collate_batch
from .model import MiniLLM
from .tokenizer import MiniLLMTokenizer
from .utils import describe_device, get_device, peak_memory_mb, peak_rss_mb, set_seed


# --------------------------------------------------------------------------- state


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].to("cpu") if torch.is_tensor(state["torch"]) else state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def resolve_precision(requested: str, device: torch.device) -> str:
    if requested != "auto":
        return requested
    if device.type == "cuda":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "fp32"


def autocast_dtype(precision: str) -> torch.dtype | None:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)


def lr_at(step: int, total: int, peak: float, min_lr: float, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    if total <= warmup:
        return peak
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- io


class CheckpointError(RuntimeError):
    """Raised when a checkpoint does not belong to this architecture."""


def load_model_checkpoint(path: str | Path, device: torch.device) -> tuple[MiniLLM, MiniLLMConfig, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointError(f"{path}: not a MiniLLM checkpoint")
    architecture = payload.get("architecture")
    if architecture != ARCHITECTURE:
        raise CheckpointError(
            f"{path}: checkpoint architecture is {architecture!r}, expected {ARCHITECTURE!r}. "
            "This looks like a TinyVLM (vision-language) checkpoint; MiniLLM has no vision "
            "tower or image tokens, so these weights cannot be loaded. Train a MiniLLM "
            "checkpoint from scratch (or from another MiniLLM checkpoint)."
        )
    if int(payload.get("arch_version", 0)) != ARCH_VERSION:
        raise CheckpointError(
            f"{path}: architecture version {payload.get('arch_version')} != {ARCH_VERSION}"
        )
    if "config" not in payload or "model" not in payload:
        raise CheckpointError(f"{path}: checkpoint is missing 'config' or 'model'")
    cfg = MiniLLMConfig.from_dict(payload["config"])
    model = MiniLLM(cfg)
    model.load_state_dict(payload["model"])
    return model.to(device), cfg, payload


def save_model_checkpoint(
    path: Path,
    model: MiniLLM,
    tokenizer: MiniLLMTokenizer,
    *,
    step: int,
    tokens: int,
    val_loss: float | None,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": ARCHITECTURE,
        "arch_version": ARCH_VERSION,
        "config": model.cfg.to_dict(),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "step": step,
        "tokens_seen": tokens,
        "val_loss": val_loss,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def save_training_state(
    path: Path,
    model: MiniLLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    tokenizer: MiniLLMTokenizer,
    *,
    step: int,
    epoch: int,
    batch_in_epoch: int,
    best_loss: float | None,
    history: list[dict],
    tokens: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": ARCHITECTURE,
            "arch_version": ARCH_VERSION,
            "config": model.cfg.to_dict(),
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "step": step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "best_loss": best_loss,
            "history": history,
            "tokens_seen": tokens,
            "rng": rng_state(),
        },
        path,
    )


def load_training_state(path: str | Path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("architecture") != ARCHITECTURE:
        raise CheckpointError(
            f"{path}: not a MiniLLM training state (architecture={payload.get('architecture')!r})"
        )
    return payload


# --------------------------------------------------------------------------- data


def make_pretrain_loader(
    dataset: Dataset,
    sampler: EpochBatchSampler,
    epoch: int,
    *,
    start_batch: int = 0,
    workers: int = 0,
    pad_token_id: int = 0,
    ignore_index: int = -100,
) -> Iterator[dict[str, torch.Tensor]]:
    """Batches for one epoch, optionally starting mid-epoch after a resume."""
    batches = sampler.batches(epoch)
    if start_batch:
        for _ in range(start_batch):
            next(batches, None)
    for indices in batches:
        items = [dataset[i] for i in indices]
        yield collate_batch(items, pad_token_id, ignore_index)


def evaluate(
    model: MiniLLM,
    dataset: Dataset,
    device: torch.device,
    *,
    batch_size: int = 8,
    max_batches: int | None = None,
    pad_token_id: int = 0,
    ignore_index: int = -100,
) -> dict:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, pad_token_id, ignore_index),
    )
    total_sum = 0.0
    total_tokens = 0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            total_sum += float(out.loss_sum)
            total_tokens += out.num_loss_tokens
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
    model.train()
    if total_tokens == 0:
        raise RuntimeError("validation produced no supervised tokens")
    return {
        "loss": total_sum / total_tokens,
        "perplexity": math.exp(min(total_sum / total_tokens, 20)),
        "tokens": total_tokens,
        "batches": batches,
    }


# --------------------------------------------------------------------------- loop


@dataclass
class TrainingOptions:
    out: str = "runs/minillm"
    epochs: int = 1
    batch_size: int = 2
    grad_accum: int = 16
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_ratio: float = 0.02
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    max_steps: int = 0
    save_every: int = 200
    val_every: int = 200
    val_limit: int = 0
    max_minutes: float = 0.0
    log_every: int = 10
    seed: int = 42
    device: str | None = None
    precision: str = "auto"
    grad_checkpointing: bool = False
    threads: int = 0
    resume: str | None = None
    init_from: str | None = None
    extra_meta: dict = field(default_factory=dict)


def fit(
    opts: TrainingOptions,
    model: MiniLLM,
    tokenizer: MiniLLMTokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset | None,
    *,
    mode: str = "pretrain",
    batch_sampler_seed: int = 0,
    batch_factory=None,
) -> dict:
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(opts.device)
    if opts.threads:
        torch.set_num_threads(opts.threads)
    precision = resolve_precision(opts.precision, device)
    dtype = autocast_dtype(precision)
    set_seed(opts.seed)
    model.to(device)
    model.train()
    model.gradient_checkpointing = bool(opts.grad_checkpointing)
    if model.cfg.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"model vocab_size={model.cfg.vocab_size} != tokenizer vocab_size={tokenizer.vocab_size}"
        )

    total_samples = len(train_dataset)
    steps_per_epoch = max(1, math.ceil(total_samples / max(opts.batch_size * opts.grad_accum, 1)))
    total_steps = opts.max_steps or steps_per_epoch * opts.epochs
    warmup = max(1, int(total_steps * opts.warmup_ratio))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opts.lr, betas=(0.9, 0.95), weight_decay=opts.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16") if device.type == "cuda" else None
    start_step = 0
    start_epoch = 0
    start_batch = 0
    best_loss: float | None = None
    history: list[dict] = []
    tokens_seen = 0

    if opts.resume:
        state = load_training_state(opts.resume)
        if state["tokenizer_fingerprint"] != tokenizer.fingerprint:
            raise CheckpointError("resume tokenizer fingerprint does not match the current tokenizer")
        if state["tokenizer_vocab_size"] != tokenizer.vocab_size:
            raise CheckpointError("resume tokenizer vocabulary size differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_step = int(state["step"])
        start_epoch = int(state["epoch"])
        start_batch = int(state["batch_in_epoch"])
        best_loss = state.get("best_loss")
        history = state.get("history", [])
        tokens_seen = int(state.get("tokens_seen", 0))
        set_rng_state(state.get("rng"))
        print(f"[resume] {opts.resume} at step {start_step}, epoch {start_epoch + 1}, "
              f"batch {start_batch}")

    print(f"[device] {describe_device(device)} | precision {precision} | threads {torch.get_num_threads()}")
    print(f"[model] {model.num_parameters():,} params | vocab {model.cfg.vocab_size} | "
          f"seq {model.cfg.max_position_embeddings}")
    print(f"[data] {mode}: {total_samples:,} samples, {steps_per_epoch:,} steps/epoch, "
          f"total {total_steps:,} updates ({total_steps * opts.batch_size * opts.grad_accum:,} samples)")

    started = time.time()
    last_log = started
    running_loss = 0.0
    running_tokens = 0
    steps_since_log = 0
    interrupted = False
    last_epoch = start_epoch
    last_batch = start_batch

    def handle_signal(signum, frame):  # noqa: ANN001 - signal handler signature
        nonlocal interrupted
        interrupted = True
        print(f"\n[signal] caught {signum}; finishing the current step and saving")

    old_handler = signal.signal(signal.SIGINT, handle_signal)

    try:
        for epoch in range(start_epoch, opts.epochs):
            first_update = start_batch if epoch == start_epoch else 0
            # `batch_in_epoch` counts optimizer updates; the streams below are
            # consumed in micro-batches, so skip grad_accum batches per update.
            first_batch = first_update * opts.grad_accum
            if batch_factory is not None:
                batches = batch_factory(epoch, first_batch)
            else:
                sampler = EpochBatchSampler(
                    total_samples, opts.batch_size, seed=opts.seed + batch_sampler_seed
                )
                batches = make_pretrain_loader(
                    train_dataset, sampler, epoch, start_batch=first_batch,
                    pad_token_id=model.cfg.pad_token_id, ignore_index=model.cfg.ignore_index,
                )
            batch_in_epoch = first_batch
            while True:
                group = _take(batches, opts.grad_accum)
                if not group:
                    break
                step = start_step + 1
                lr = lr_at(step - 1, total_steps, opts.lr, opts.min_lr, warmup)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                loss, tokens = _train_step(model, group, optimizer, scaler, device, dtype, opts, optimizer.param_groups[0]["lr"])
                running_loss += loss
                running_tokens += tokens
                steps_since_log += 1
                tokens_seen += tokens
                start_step = step
                batch_in_epoch += 1
                last_epoch, last_batch = epoch, batch_in_epoch

                if start_step % opts.log_every == 0 or time.time() - last_log > 30:
                    now = time.time()
                    elapsed = now - last_log
                    rate = running_tokens / max(elapsed, 1e-6)
                    print(f"epoch {epoch + 1}/{opts.epochs} step {start_step}/{total_steps} "
                          f"loss {running_loss / max(steps_since_log, 1):.4f} lr {lr:.2e} "
                          f"{rate:,.0f} tok/s {elapsed / max(steps_since_log, 1):.2f}s/step")
                    last_log = now
                    running_loss = 0.0
                    running_tokens = 0
                    steps_since_log = 0

                if val_dataset is not None and opts.val_every and start_step % opts.val_every == 0:
                    metrics = evaluate(
                        model, val_dataset, device, batch_size=opts.batch_size,
                        max_batches=opts.val_limit or None,
                        pad_token_id=model.cfg.pad_token_id, ignore_index=model.cfg.ignore_index,
                    )
                    print(f"[val] step {start_step}: loss {metrics['loss']:.4f} "
                          f"ppl {metrics['perplexity']:.2f} tokens {metrics['tokens']:,}")
                    history.append({"step": start_step, "mode": mode, **metrics})
                    _write_metrics(out, history)
                    if best_loss is None or metrics["loss"] < best_loss:
                        best_loss = metrics["loss"]
                        save_model_checkpoint(
                            out / "best.pt", model, tokenizer, step=start_step,
                            tokens=tokens_seen, val_loss=best_loss,
                            extra={"mode": mode, **opts.extra_meta},
                        )

                if opts.save_every and start_step % opts.save_every == 0:
                    save_training_state(
                        out / "state.pt", model, optimizer, scaler, tokenizer,
                        step=start_step, epoch=epoch, batch_in_epoch=batch_in_epoch,
                        best_loss=best_loss, history=history, tokens=tokens_seen,
                    )
                    print(f"[save] step {start_step} -> {out / 'state.pt'}")

                if interrupted:
                    break
                if opts.max_minutes and (time.time() - started) / 60 >= opts.max_minutes:
                    print(f"[budget] {opts.max_minutes:.0f} minutes reached; stopping after step {start_step}")
                    interrupted = True
                    break
                if start_step >= total_steps:
                    break
            if interrupted or start_step >= total_steps:
                break
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise SystemExit(
                "CUDA out of memory. Try a smaller --batch-size (e.g. 1), a shorter "
                "--seq-len (e.g. 512), --grad-checkpointing, or CPU training."
            ) from exc
        raise
    finally:
        signal.signal(signal.SIGINT, old_handler)

    elapsed = time.time() - started
    metrics = evaluate(
        model, val_dataset, device, batch_size=opts.batch_size,
        max_batches=opts.val_limit or None,
        pad_token_id=model.cfg.pad_token_id, ignore_index=model.cfg.ignore_index,
    ) if val_dataset is not None else None
    save_model_checkpoint(
        out / "model.pt", model, tokenizer, step=start_step, tokens=tokens_seen,
        val_loss=metrics["loss"] if metrics else None,
        extra={"mode": mode, **opts.extra_meta},
    )
    save_training_state(
        out / "state.pt", model, optimizer, scaler, tokenizer,
        step=start_step, epoch=last_epoch, batch_in_epoch=last_batch,
        best_loss=best_loss, history=history, tokens=tokens_seen,
    )
    tokenizer.save(out / "tokenizer.json")
    model.cfg.save(out / "config.json")
    summary = {
        "mode": mode,
        "steps": start_step,
        "tokens_seen": tokens_seen,
        "seconds": round(elapsed, 1),
        "best_val_loss": best_loss,
        "final_val": metrics,
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "peak_gpu_mb": round(peak_memory_mb(device) or 0, 1),
        "precision": precision,
        "batch_size": opts.batch_size,
        "grad_accum": opts.grad_accum,
        "grad_checkpointing": opts.grad_checkpointing,
        "seed": opts.seed,
        "params": model.num_parameters(),
        **opts.extra_meta,
    }
    (out / "train-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] {out} | {start_step} steps | {tokens_seen:,} tokens | "
          f"{elapsed / 60:.1f} min | peak RSS {summary['peak_rss_mb']:.0f} MB")
    if metrics:
        print(f"[done] val loss {metrics['loss']:.4f} (ppl {metrics['perplexity']:.2f})")
    return summary


def _take(iterator: Iterator[dict], n: int) -> list[dict]:
    group = []
    for _ in range(n):
        try:
            group.append(next(iterator))
        except StopIteration:
            break
    return group


def _train_step(
    model: MiniLLM,
    group: Sequence[dict],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    dtype: torch.dtype | None,
    opts: TrainingOptions,
    lr: float,
) -> tuple[float, int]:
    """One optimizer update over `grad_accum` micro-batches, token-weighted."""
    group_tokens = sum(
        int((batch["labels"][:, 1:] != model.cfg.ignore_index).sum()) for batch in group
    )
    group_tokens = max(group_tokens, 1)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for batch in group:
        batch = {k: v.to(device) for k, v in batch.items()}
        if dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = out.loss_sum / group_tokens
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            out = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = out.loss_sum / group_tokens
            loss.backward()
        total_loss += float(out.loss_sum.detach())
    if scaler is not None and scaler.is_enabled():
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), opts.grad_clip)
    if scaler is not None and scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return total_loss / group_tokens, group_tokens


def _write_metrics(out: Path, history: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pretrain MiniLLM")
    p.add_argument("--config", default="minillm-87m", help="preset name or path to config.json")
    p.add_argument("--data", required=True, help="train.bin produced by scripts/prepare_text.py")
    p.add_argument("--val-data", help="val.bin; omit to skip validation")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", default="runs/minillm")
    p.add_argument("--init-from", help="MiniLLM checkpoint to warm-start from")
    p.add_argument("--resume", help="state.pt to resume from")
    p.add_argument("--seq-len", type=int, default=0, help="0 = max_position_embeddings")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--val-limit", type=int, default=0, help="max validation batches")
    p.add_argument("--max-minutes", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = get_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.resume:
        state = load_training_state(args.resume)
        cfg = MiniLLMConfig.from_dict(state["config"])
        model = MiniLLM(cfg)
        model.load_state_dict(state["model"])
        tokenizer = MiniLLMTokenizer.load(args.tokenizer)
    elif args.init_from:
        model, cfg, payload = load_model_checkpoint(args.init_from, torch.device("cpu"))
        tokenizer = MiniLLMTokenizer.load(args.tokenizer)
        if payload.get("tokenizer_fingerprint") not in (None, tokenizer.fingerprint):
            raise SystemExit("init-from tokenizer fingerprint does not match --tokenizer")
        if payload.get("tokenizer_vocab_size") not in (None, tokenizer.vocab_size):
            raise SystemExit("init-from tokenizer vocabulary size does not match --tokenizer")
    else:
        cfg = get_config(args.config) if not Path(args.config).exists() else MiniLLMConfig.load(args.config)
        tokenizer = MiniLLMTokenizer.load(args.tokenizer)
        cfg.vocab_size = tokenizer.vocab_size
        model = MiniLLM(cfg)
        print(f"[init] fresh {cfg.model_type} weights")
    tokenizer.assert_vocab_size(model.cfg.vocab_size)
    seq_len = args.seq_len or model.cfg.max_position_embeddings
    if seq_len > model.cfg.max_position_embeddings:
        raise SystemExit("--seq-len exceeds max_position_embeddings")

    train_dataset = PackedPretrainDataset(args.data, seq_len)
    val_dataset = PackedPretrainDataset(args.val_data, seq_len) if args.val_data else None
    if val_dataset is not None and val_dataset.path.resolve() == train_dataset.path.resolve():
        raise SystemExit("--val-data must not be the training file")
    opts = TrainingOptions(
        out=args.out, epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, min_lr=args.min_lr, warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip, max_steps=args.max_steps,
        save_every=args.save_every, val_every=args.val_every, val_limit=args.val_limit,
        max_minutes=args.max_minutes, log_every=args.log_every, seed=args.seed,
        device=args.device, precision=args.precision,
        grad_checkpointing=args.grad_checkpointing, threads=args.threads,
        resume=args.resume, init_from=args.init_from,
        extra_meta={"config_name": args.config, "seq_len": seq_len, "data": args.data},
    )
    print(f"[data] train {train_dataset.num_tokens:,} tokens ({len(train_dataset):,} samples of {seq_len})")
    fit(opts, model, tokenizer, train_dataset, val_dataset, mode="pretrain")


if __name__ == "__main__":
    main()
