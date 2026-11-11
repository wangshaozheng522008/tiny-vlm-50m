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
import random
import signal
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .checkpoint import (
    CheckpointError, TRAINING_STATE_VERSION, atomic_torch_save, load_training_state,
)
from .training_manifest import build_manifest, compare_manifests, write_json
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


def load_model_checkpoint(path: str | Path, device: torch.device) -> tuple[MiniLLM, MiniLLMConfig, dict]:
    path = Path(path)
    if path.name.startswith(".") or path.suffix == ".tmp":
        raise CheckpointError(f"{path}: temporary files cannot be loaded as checkpoints")
    payload = torch.load(path, map_location="cpu", weights_only=False)
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
    atomic_torch_save(payload, path)


def save_training_state(
    path: Path, model: MiniLLM, optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None, tokenizer: MiniLLMTokenizer, *,
    global_step: int, epoch: int, micro_batches_consumed_in_epoch: int,
    samples_seen: int, best_loss: float | None, history: list[dict], tokens: int,
    manifest: dict, preserve_backup: bool = False,
) -> None:
    backup = None if preserve_backup else path.with_name("state.prev.pt")
    if path.exists() and backup is not None:
        try:
            # mmap validates metadata without allocating another full AdamW state.
            load_training_state(path, mmap=True)
        except CheckpointError:
            print(f"[checkpoint] invalid primary {path}; preserving existing backup")
            backup = None
    payload = {
        "architecture": ARCHITECTURE, "arch_version": ARCH_VERSION,
        "training_state_version": TRAINING_STATE_VERSION,
        "config": model.cfg.to_dict(),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "tokenizer_fingerprint": tokenizer.fingerprint, "tokenizer_vocab_size": tokenizer.vocab_size,
        "global_step": global_step, "epoch": epoch,
        "micro_batches_consumed_in_epoch": micro_batches_consumed_in_epoch,
        "samples_seen": samples_seen, "tokens_seen": tokens,
        "best_loss": best_loss, "history": history, "rng": rng_state(), "manifest": manifest,
    }
    atomic_torch_save(payload, path, backup=backup)


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
    batches = sampler.batches(epoch, start_batch=start_batch)
    for indices in batches:
        items = [dataset[i] for i in indices]
        yield collate_batch(items, pad_token_id, ignore_index)


def evaluate(*args, **kwargs) -> dict:
    # DataLoader allocates a base seed even with shuffle=False. Validation must
    # not perturb dropout/RNG when validation frequency changes across resumes.
    state = rng_state()
    model = args[0] if args else kwargs["model"]
    was_training = model.training
    try:
        result = _evaluate(*args, **kwargs)
        if not math.isfinite(result["loss"]):
            raise NumericalError("non-finite validation loss; best.pt will not be updated")
        return result
    finally:
        model.train(was_training)
        set_rng_state(state)


def _evaluate(
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


def _validate_progress(state: dict, manifest: dict, dataset: Dataset, sampler: EpochBatchSampler) -> None:
    plan = manifest["training"]
    epoch = state["epoch"]
    cursor = state["micro_batches_consumed_in_epoch"]
    if not 0 <= cursor < len(sampler) or cursor % plan["grad_accum"]:
        raise CheckpointError("resume micro-batch cursor is outside the dataset or not at an update boundary")
    expected_step = epoch * plan["optimizer_updates_per_epoch"] + cursor // plan["grad_accum"]
    if state["global_step"] != expected_step or expected_step > plan["target_optimizer_updates"]:
        raise CheckpointError("resume global_step/epoch/micro-batch cursor exceeds or contradicts the training plan")
    prefix_samples = min(cursor * plan["batch_size"], len(dataset))
    if state["samples_seen"] != epoch * len(dataset) + prefix_samples:
        raise CheckpointError("resume samples_seen contradicts the data cursor")
    if isinstance(dataset, PackedPretrainDataset):
        prefix_tokens = prefix_samples * (dataset.seq_len - 1)
    else:
        prefix_tokens = sum(int((dataset[i]["labels"][1:] != manifest["model_config"]["ignore_index"]).sum())
                            for i in sampler.indices(epoch)[:prefix_samples])
    expected_tokens = epoch * manifest["dataset"]["train"]["effective_loss_tokens"] + prefix_tokens
    if state["tokens_seen"] != expected_tokens:
        raise CheckpointError("resume tokens_seen contradicts the data cursor")


def _prepare_run_manifest(out: Path, manifest: dict, resumed: dict | None = None) -> dict:
    """Check directory ownership before writing any artifacts, including logs.

    A configuration fingerprint is not a run identity: separate starts using
    the same seed and data must still own separate output directories.
    """
    run_id = None
    if resumed is not None:
        compare_manifests({k: v for k, v in resumed.items() if k != "run_id"}, manifest)
        run_id = resumed.get("run_id")
        if "run_id" in resumed and (not isinstance(run_id, str) or not run_id):
            raise CheckpointError("resume manifest has an invalid run_id")
    if out.exists():
        if not out.is_dir():
            raise CheckpointError(f"{out}: training output must be a directory")
        if any(out.iterdir()):
            if resumed is None:
                raise CheckpointError(
                    f"{out}: non-empty training output directory; refusing to overwrite existing files. "
                    "Use --resume for this run, or select a new empty --out (also for --init-from)."
                )
            if run_id is None:
                raise CheckpointError(
                    f"{out}: resume checkpoint has no run_id; cannot verify output directory ownership. "
                    "Resume into a new empty --out to migrate this v2 state without overwriting old results."
                )
            try:
                existing = json.loads((out / "training-manifest.json").read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise CheckpointError(
                    f"{out}: cannot verify output directory ownership: missing/unreadable training-manifest.json. "
                    "Select a new empty --out."
                ) from exc
            if not isinstance(existing, dict) or existing.get("run_id") != run_id:
                raise CheckpointError(f"{out}: output directory belongs to another training run (run_id mismatch)")
            compare_manifests(resumed, existing)
    # uuid4 uses OS entropy, without advancing Python/NumPy/Torch training RNG.
    return {**manifest, "run_id": run_id or uuid.uuid4().hex}


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
    """Train a fixed plan, saving only complete optimizer updates.

    Callers constructing fresh models must seed *before* constructing the model.
    A resume restores RNG after all setup; it never reseeds the resumed stream.
    Custom batch factories must follow EpochBatchSampler order and expose
    sampler_seed when using a seed different from opts.seed+batch_sampler_seed.
    """
    device = get_device(opts.device)
    if opts.threads:
        torch.set_num_threads(opts.threads)
    precision = resolve_precision(opts.precision, device)
    dtype = autocast_dtype(precision)
    if device.type != "cuda" and precision != "fp32":
        raise ValueError("CPU training supports fp32; fp16/bf16 require CUDA")
    if not opts.resume:
        set_seed(opts.seed)
    if opts.resume and opts.init_from:
        raise ValueError("--resume and --init-from are mutually exclusive")
    if model.cfg.vocab_size != tokenizer.vocab_size:
        raise ValueError(f"model vocab_size={model.cfg.vocab_size} != tokenizer vocab_size={tokenizer.vocab_size}")
    sampler = EpochBatchSampler(
        len(train_dataset), opts.batch_size,
        seed=getattr(batch_factory, "sampler_seed", opts.seed + batch_sampler_seed),
    )
    manifest, total_tokens = build_manifest(
        opts, model, tokenizer, train_dataset, val_dataset, mode=mode,
        sampler=sampler, precision=precision, device=device,
    )
    out = Path(opts.out)
    if not opts.resume:
        manifest = _prepare_run_manifest(out, manifest)
    plan = manifest["training"]
    total_steps = plan["target_optimizer_updates"]
    warmup = plan["warmup_steps"]
    model.to(device)
    model.train()
    model.gradient_checkpointing = bool(opts.grad_checkpointing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opts.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=opts.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16") if device.type == "cuda" else None
    global_step = epoch = cursor = samples_seen = tokens_seen = 0
    best_loss: float | None = None
    history: list[dict] = []
    if opts.resume:
        state = load_training_state(opts.resume)
        manifest = _prepare_run_manifest(out, manifest, state["manifest"])
        _validate_progress(state, manifest, train_dataset, sampler)
        try:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if scaler is not None and scaler.is_enabled():
                if not state["scaler"]:
                    raise CheckpointError("resume is missing required AMP scaler state")
                scaler.load_state_dict(state["scaler"])
            set_rng_state(state["rng"])
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            raise CheckpointError(f"invalid resume model/optimizer/scaler/RNG: {exc}") from exc
        global_step, epoch, cursor = (state[key] for key in (
            "global_step", "epoch", "micro_batches_consumed_in_epoch"))
        samples_seen, tokens_seen = state["samples_seen"], state["tokens_seen"]
        best_loss, history = state["best_loss"], state["history"]
        del state
        print(f"[resume] {opts.resume}: global_step {global_step}, epoch {epoch}, micro-batch {cursor}")

    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "training-manifest.json", manifest)
    tokenizer.save(out / "tokenizer.json")
    model.cfg.save(out / "config.json")
    print(f"[device] {describe_device(device)} | precision {precision} | threads {torch.get_num_threads()}")
    print(f"[model] {model.num_parameters():,} params | mode {mode}")
    print(f"Total training tokens: {total_tokens:,} effective loss tokens/epoch")
    print(f"Total training samples: {len(train_dataset):,}")
    print(f"Micro-batches per epoch: {len(sampler):,}")
    print(f"Optimizer updates per epoch: {plan['optimizer_updates_per_epoch']:,}")
    print(f"Target optimizer updates: {total_steps:,}")
    estimate = total_steps * total_tokens / plan["optimizer_updates_per_epoch"]
    print(f"Planned effective token budget (Estimated): {estimate:,.0f}")
    print(f"Warmup steps: {warmup:,}")
    print(f"Scheduler total steps: {total_steps:,}")

    started = last_log = time.monotonic()
    running_loss_sum = 0.0
    running_tokens = steps_since_log = 0
    interrupted = False
    stop_reason = "target_reached"
    last_saved_step = global_step if opts.resume and Path(opts.resume).resolve() == (out / "state.pt").resolve() else -1
    last_checkpoint_seconds = None
    # Explicit backup recovery treats the old primary as untrusted, even if a
    # damaged file can still deserialize. Preserve the chosen recovery point.
    preserve_backup = bool(opts.resume and Path(opts.resume).name == "state.prev.pt")

    def save_state():
        nonlocal last_saved_step, last_checkpoint_seconds, preserve_backup
        if last_saved_step == global_step:
            return
        checkpoint_started = time.monotonic()
        try:
            save_training_state(
                out / "state.pt", model, optimizer, scaler, tokenizer,
                global_step=global_step, epoch=epoch,
                micro_batches_consumed_in_epoch=cursor, samples_seen=samples_seen,
                best_loss=best_loss, history=history, tokens=tokens_seen, manifest=manifest,
                preserve_backup=preserve_backup,
            )
        except Exception as exc:
            print(f"[checkpoint] FAILED step {global_step}: {exc}; previous valid checkpoint retained", flush=True)
            raise
        last_saved_step = global_step
        preserve_backup = False
        last_checkpoint_seconds = time.monotonic() - checkpoint_started
        record = {"event": "checkpoint", "global_step": global_step, "status": "saved",
                  "seconds": last_checkpoint_seconds}
        _append_log(out, record)
        print(f"[checkpoint] saved step {global_step} -> {out / 'state.pt'} ({last_checkpoint_seconds:.2f}s)")

    def validation():
        if val_dataset is None:
            return None
        metrics = evaluate(
            model, val_dataset, device, batch_size=opts.batch_size,
            max_batches=opts.val_limit or None,
            pad_token_id=model.cfg.pad_token_id, ignore_index=model.cfg.ignore_index,
        )
        print(f"[val] step {global_step}: loss {metrics['loss']:.4f} ppl {metrics['perplexity']:.2f}")
        return metrics

    def handle_signal(signum, frame):
        nonlocal interrupted, stop_reason
        interrupted, stop_reason = True, signal.Signals(signum).name
        print(f"\n[signal] caught {stop_reason}; finishing current optimizer update then saving", flush=True)

    old_handlers = {sig: signal.signal(sig, handle_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while global_step < total_steps and not interrupted:
            if opts.max_minutes and (time.monotonic() - started) >= opts.max_minutes * 60:
                stop_reason = "time_budget"
                interrupted = True
                break
            batches = iter(batch_factory(epoch, cursor)) if batch_factory is not None else make_pretrain_loader(
                train_dataset, sampler, epoch, start_batch=cursor,
                pad_token_id=model.cfg.pad_token_id, ignore_index=model.cfg.ignore_index,
            )
            while global_step < total_steps and not interrupted:
                remaining = min(opts.grad_accum, len(sampler) - cursor)
                group = _take(batches, remaining)
                if len(group) != remaining:
                    raise RuntimeError("batch stream ended before the declared dataset cursor; refusing to skip data")
                step_started = time.monotonic()
                lr = lr_at(global_step, total_steps, opts.lr, opts.min_lr, warmup)
                _check_lr(lr, opts.lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                result = _train_step(model, group, optimizer, scaler, device, dtype, opts, lr)
                # Commit all counters only after an actual, finite optimizer update.
                global_step += 1
                cursor += len(group)
                samples_seen += sum(batch["input_ids"].shape[0] for batch in group)
                tokens_seen += result.tokens
                running_loss_sum += result.loss_sum
                running_tokens += result.tokens
                steps_since_log += 1
                completed_epoch, completed_cursor = epoch, cursor
                if cursor == len(sampler):
                    epoch, cursor = epoch + 1, 0
                step_seconds = time.monotonic() - step_started
                now = time.monotonic()
                if opts.max_minutes and now - started >= opts.max_minutes * 60:
                    interrupted, stop_reason = True, "time_budget"
                    print(f"[budget] stopping after complete step {global_step}")
                if (opts.log_every and global_step % opts.log_every == 0) or now - last_log > 30 or global_step == total_steps or interrupted:
                    elapsed = max(now - last_log, 1e-6)
                    record = {
                        "event": "train", "global_step": global_step,
                        "epoch": completed_epoch, "micro_batches_consumed_in_epoch": completed_cursor,
                        "micro_batches_per_epoch": len(sampler), "samples_seen": samples_seen,
                        "tokens_seen": tokens_seen, "loss": running_loss_sum / running_tokens,
                        "loss_tokens": running_tokens, "learning_rate": lr, "gradient_norm": result.grad_norm,
                        "tokens_per_second": running_tokens / elapsed, "step_seconds": step_seconds,
                        "mean_step_seconds": elapsed / steps_since_log, "peak_gpu_mb": peak_memory_mb(device),
                    }
                    _append_log(out, record)
                    print(f"step {global_step}/{total_steps} epoch {completed_epoch} "
                          f"micro-batch {completed_cursor}/{len(sampler)} tokens {tokens_seen:,} "
                          f"loss {record['loss']:.4f} lr {lr:.2e} grad {result.grad_norm:.3f} "
                          f"{record['tokens_per_second']:,.0f} tok/s {step_seconds:.3f}s/step")
                    last_log, running_loss_sum, running_tokens, steps_since_log = now, 0.0, 0, 0
                if interrupted:
                    break  # Persist first; do not spend the shutdown budget validating.
                if val_dataset is not None and opts.val_every and global_step % opts.val_every == 0:
                    metrics = validation()
                    history.append({"step": global_step, "mode": mode, **metrics})
                    _write_metrics(out, history)
                    _append_log(out, {"event": "validation", "global_step": global_step, **metrics})
                    if best_loss is None or metrics["loss"] < best_loss:
                        save_model_checkpoint(
                            out / "best.pt", model, tokenizer, step=global_step, tokens=tokens_seen,
                            val_loss=metrics["loss"], extra={"mode": mode, **opts.extra_meta},
                        )
                        best_loss = metrics["loss"]
                if opts.save_every and global_step % opts.save_every == 0:
                    save_state()
                if cursor == 0:
                    break
        save_state()
        metrics = None if interrupted else validation()
        save_model_checkpoint(
            out / "model.pt", model, tokenizer, step=global_step, tokens=tokens_seen,
            val_loss=metrics["loss"] if metrics else None, extra={"mode": mode, **opts.extra_meta},
        )
    except Exception as exc:
        optimizer.zero_grad(set_to_none=True)
        message = str(exc)
        if "out of memory" in message.lower():
            message = "CUDA OOM; reduce --batch-size/--seq-len or use --grad-checkpointing. " + message
        print(f"[error] training aborted at completed global_step {global_step}: {message}. "
              "No partial update will be saved; resume the last valid state.pt or explicit state.prev.pt.", flush=True)
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)

    summary = {
        "mode": mode, "steps": global_step, "global_step": global_step, "epoch": epoch,
        "micro_batches_consumed_in_epoch": cursor, "samples_seen": samples_seen,
        "tokens_seen": tokens_seen, "target_optimizer_updates": total_steps,
        "optimizer_updates_per_epoch": plan["optimizer_updates_per_epoch"],
        "seconds": round(time.monotonic() - started, 3), "stop_reason": stop_reason,
        "best_val_loss": best_loss, "final_val": metrics,
        "last_checkpoint_seconds": last_checkpoint_seconds,
        "peak_rss_mb": round(peak_rss_mb(), 1), "peak_gpu_mb": round(peak_memory_mb(device) or 0, 1),
        "precision": precision, "batch_size": opts.batch_size, "grad_accum": opts.grad_accum,
        "grad_checkpointing": opts.grad_checkpointing, "seed": opts.seed,
        "params": model.num_parameters(), **opts.extra_meta,
    }
    write_json(out / "train-summary.json", summary)
    print(f"[done] {out} | {global_step} updates | {samples_seen:,} samples | {tokens_seen:,} loss tokens | {stop_reason}")
    return summary


def _take(iterator: Iterator[dict], n: int) -> list[dict]:
    group = []
    for _ in range(n):
        try:
            group.append(next(iterator))
        except StopIteration:
            break
    return group


class NumericalError(RuntimeError):
    """An update failed; only the last persisted complete update is resumable."""


class AMPOverflowError(NumericalError):
    """GradScaler skipped an optimizer update (fatal; counters are not advanced)."""


@dataclass
class StepResult:
    loss_sum: float
    tokens: int
    grad_norm: float


def _check_lr(lr: float, peak: float) -> None:
    if not math.isfinite(lr) or lr < 0 or lr > peak * (1 + 1e-12):
        raise NumericalError(f"abnormal learning rate: {lr!r}")


def _train_step(
    model: MiniLLM, group: Sequence[dict], optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None, device: torch.device, dtype: torch.dtype | None,
    opts: TrainingOptions, lr: float,
) -> StepResult:
    """Token-weighted gradients; a failed/skipped update never returns success."""
    _check_lr(lr, opts.lr)
    group_tokens = sum(int((batch["labels"][:, 1:] != model.cfg.ignore_index).sum()) for batch in group)
    if group_tokens <= 0:
        raise NumericalError("optimizer group has no effective loss tokens")
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    amp = scaler is not None and scaler.is_enabled()
    for batch in group:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None and device.type == "cuda"):
            out = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = out.loss_sum / group_tokens
        loss_sum = float(out.loss_sum.detach())
        if not math.isfinite(loss_sum):
            raise NumericalError(f"non-finite training loss: {loss_sum}")
        if amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        total_loss += loss_sum
    if not math.isfinite(total_loss):
        raise NumericalError("non-finite accumulated training loss")
    if amp:
        scaler.unscale_(optimizer)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), opts.grad_clip))
    if not math.isfinite(norm):
        if amp:
            # Do not call optimizer.step on non-finite gradients. Abort rather
            # than consume data while silently counting a skipped update.
            raise AMPOverflowError("AMP overflow/non-finite gradient norm: optimizer update skipped; global_step unchanged")
        raise NumericalError(f"non-finite gradient norm: {norm}")
    if amp:
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            raise AMPOverflowError("AMP overflow: GradScaler skipped update; global_step unchanged")
    else:
        optimizer.step()
    return StepResult(total_loss, group_tokens, norm)


def _write_metrics(out: Path, history: list[dict]) -> None:
    write_json(out / "metrics.json", history)


def _append_log(out: Path, record: dict) -> None:
    with (out / "training-log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")


# --------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pretrain MiniLLM")
    p.add_argument("--config", default=None, help="preset/path (fresh default minillm-87m); checked when supplied with --resume")
    p.add_argument("--data", required=True, help="train.bin produced by scripts/prepare_text.py")
    p.add_argument("--val-data", help="val.bin; omit to skip validation")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", default="runs/minillm")
    p.add_argument("--init-from", help="MiniLLM checkpoint to warm-start from")
    p.add_argument("--resume", help="state.pt (or explicit state.prev.pt); restores the fixed training plan")
    p.add_argument("--seq-len", type=int, default=0, help="0 = max_position_embeddings")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=0, help="cumulative target updates; >0 overrides --epochs")
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
    set_seed(args.seed)  # before any model constructor (including resume setup)
    device = get_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.resume:
        state = load_training_state(args.resume, mmap=True)
        cfg = MiniLLMConfig.from_dict(state["config"])
        tokenizer = MiniLLMTokenizer.load(args.tokenizer)
        if args.config:
            requested_cfg = MiniLLMConfig.load(args.config) if Path(args.config).exists() else get_config(args.config)
            requested_cfg.vocab_size = tokenizer.vocab_size
            if requested_cfg.to_dict() != cfg.to_dict():
                raise CheckpointError("resume Model Config differs from --config")
        del state  # release the mapped checkpoint before allocating new weights
        model = MiniLLM(cfg)
        # fit validates and loads the full state once setup is complete.
    elif args.init_from:
        model, cfg, payload = load_model_checkpoint(args.init_from, torch.device("cpu"))
        tokenizer = MiniLLMTokenizer.load(args.tokenizer)
        if payload.get("tokenizer_fingerprint") not in (None, tokenizer.fingerprint):
            raise SystemExit("init-from tokenizer fingerprint does not match --tokenizer")
        if payload.get("tokenizer_vocab_size") not in (None, tokenizer.vocab_size):
            raise SystemExit("init-from tokenizer vocabulary size does not match --tokenizer")
        del payload
    else:
        cfg = get_config(args.config or "minillm-87m") if not args.config or not Path(args.config).exists() else MiniLLMConfig.load(args.config)
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
