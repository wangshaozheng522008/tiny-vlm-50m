"""Train one decoder on text-only, image-only, and image+text samples.

Besides the existing one-shot training this script supports:

* ``--indexed``: read a large pre-chunked text corpus from disk without keeping
  it in memory,
* ``--save-every``: write a resumable checkpoint (weights, optimizer, AMP state,
  data progress) every N optimiser updates,
* ``--resume``: continue where the last resumable checkpoint stopped,
* ``--max-minutes``: stop gracefully after a wall-clock budget and save.
"""

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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .config import get_config
from .data import (
    MODES, IndexedTextDataset, MultimodalDataset, SyntheticMultimodalDataset, collate_fn,
    read_records, record_texts, synthetic_texts,
)
from .generate import load_model
from .model import TinyVLM, count_parameters
from .tokenizer import TinyTokenizer
from .utils import (describe_device, estimate_sizes, get_device, peak_memory_mb,
                    peak_rss_mb, set_seed)


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
    if args.indexed:
        return IndexedTextDataset(args.data, tokenizer=tokenizer,
                                  max_seq_len=cfg.text.max_seq_len,
                                  ignore_index=cfg.ignore_index)
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


class EpochBatchSampler:
    """Deterministic per-epoch batch order that can start in the middle.

    Epoch N always produces the same permutation for a given seed, so a resumed
    run skips no data and does not need to save sampler state.
    """

    def __init__(self, size: int, batch_size: int, seed: int, epoch: int, start_batch: int = 0):
        if size <= 0 or batch_size <= 0:
            raise ValueError("size and batch_size must be positive")
        generator = torch.Generator().manual_seed((seed + epoch * 1_000_003) % (2**63 - 1))
        order = torch.randperm(size, generator=generator).tolist()
        self.batches = [order[i:i + batch_size] for i in range(0, size, batch_size)]
        if not 0 <= start_batch <= len(self.batches):
            raise ValueError(f"start_batch {start_batch} outside [0, {len(self.batches)}]")
        self.start = start_batch

    def __len__(self) -> int:
        return len(self.batches) - self.start

    def __iter__(self):
        return iter(self.batches[self.start:])


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


def save_training_state(path, model, tokenizer, optimizer, scaler, *, step, epoch,
                        batch_in_epoch, epoch_loss, epoch_tokens, best_loss, history,
                        total_steps, args, dataset_modes):
    """Resumable checkpoint: weights plus everything needed to continue the schedule."""
    torch.save({
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": model.cfg,
        "step": step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "epoch_loss": epoch_loss,
        "epoch_tokens": epoch_tokens,
        "best_loss": best_loss,
        "history": history,
        "total_steps": total_steps,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "training_modes": dict(dataset_modes),
        "args": vars(args),
    }, path)


# argparse defaults that resume restores when the caller did not override them.
CLI_DEFAULTS = {
    "epochs": 3, "batch_size": 8, "grad_accum": 1, "lr": 3e-4, "warmup": 100,
    "weight_decay": 0.1, "workers": 2, "seed": 42, "threads": 0,
    "save_every": 200, "val_every": 0, "val_limit": 0, "max_minutes": 0.0,
    "max_steps": 0, "log_every": 20,
}
# Changing these changes the meaning of `batch_in_epoch`, so resume refuses to.
FROZEN_ON_RESUME = ("batch_size", "grad_accum", "seed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="tiny-50m")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--data", nargs="+", help="one or more mixed-mode JSONL files")
    source.add_argument("--synthetic", type=int, default=0, help="N grounded toy examples for smoke testing")
    p.add_argument("--synthetic-modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--val-data", nargs="+", help="held-out JSONL files; never used to train the tokenizer")
    p.add_argument("--tokenizer")
    p.add_argument("--text-corpus", help="extra plain text for BPE training, not model training")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--init-from", help="warm-start model weights; starts a new optimiser and LR schedule")
    p.add_argument("--freeze-vision", action="store_true", help="freeze the warm-started ViT, keep projector trainable")
    p.add_argument("--indexed", action="store_true",
                   help="read --data/--val-data as pre-chunked {\"text\"} JSONL without loading it all")
    p.add_argument("--resume", help="continue from a resumable state saved with --save-every")
    p.add_argument("--save-every", type=int, default=200,
                   help="optimiser updates between resumable checkpoints; 0 disables periodic saves")
    p.add_argument("--val-every", type=int, default=0,
                   help="optimiser updates between validation runs; 0 = epoch end and final stop")
    p.add_argument("--val-limit", type=int, default=0,
                   help="cap validation samples (0 = all); useful for mid-epoch validation")
    p.add_argument("--max-minutes", type=float, default=0,
                   help="stop gracefully after this many minutes; 0 = no budget")
    p.add_argument("--threads", type=int, default=0,
                   help="torch CPU threads; 0 keeps the default (more can be slower on shared hosts)")
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

    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing = {"model", "optimizer", "config", "step"} - set(resume_state)
        if missing:
            p.error(f"{args.resume} is not a resumable checkpoint (missing {sorted(missing)}); "
                    "use --init-from for weights-only warm starts")
        stored = resume_state.get("args", {})
        if not args.data and not args.synthetic:
            args.data = stored.get("data")
            args.synthetic = stored.get("synthetic", 0)
            args.synthetic_modes = stored.get("synthetic_modes", list(MODES))
            args.indexed = args.indexed or stored.get("indexed", False)
        if not args.val_data:
            args.val_data = stored.get("val_data")
        args.freeze_vision = args.freeze_vision or stored.get("freeze_vision", False)
        if args.out == "runs/tiny-vlm":  # default: write beside the state file
            args.out = str(Path(args.resume).parent)
        for name, default in CLI_DEFAULTS.items():
            saved = stored.get(name)
            if saved is not None and getattr(args, name) == default and saved != default:
                setattr(args, name, saved)
                print(f"[resume] keeping {name}={saved} from the checkpoint")
        for name in FROZEN_ON_RESUME:
            saved = stored.get(name)
            if saved is not None and getattr(args, name) != saved:
                p.error(f"{name} cannot change while resuming ({saved} -> {getattr(args, name)}); "
                        "start a new run with --init-from and a new --out instead")
        if args.init_from:
            p.error("--resume already restores weights; remove --init-from")
    elif not args.data and not args.synthetic:
        p.error("one of --data, --synthetic, or --resume is required")
    if min(args.epochs, args.batch_size, args.grad_accum, args.log_every) <= 0:
        p.error("epochs, batch-size, grad-accum, and log-every must be positive")
    if min(args.synthetic, args.max_steps, args.workers, args.warmup,
           args.save_every, args.val_every, args.val_limit, args.threads) < 0:
        p.error("synthetic, max-steps, workers, warmup, save-every, val-every, val-limit and "
                "threads must be nonnegative")
    if args.synthetic == 0 and not args.data and resume_state is None:
        p.error("synthetic must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        p.error("lr must be positive and finite")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error("weight-decay must be finite and nonnegative")
    if not math.isfinite(args.max_minutes) or args.max_minutes < 0:
        p.error("max-minutes must be finite and nonnegative")
    if args.freeze_vision and not args.init_from and resume_state is None:
        p.error("freeze-vision requires --init-from to avoid freezing random visual weights")
    if args.indexed and not args.tokenizer and not (args.init_from or args.resume):
        p.error("--indexed needs an existing tokenizer (pass --tokenizer or --init-from/--resume)")
    if args.resume and args.modality_weights:
        p.error("--resume does not support --modality-weights; the weighted sampler is not resumable")
    if args.indexed and args.modality_weights:
        p.error("--indexed text data is text-only; --modality-weights is unnecessary")

    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[device] {describe_device(device)} | torch threads {torch.get_num_threads()}")
    records = None if args.indexed else (read_records(args.data) if args.data else None)
    if resume_state is not None:
        cfg = resume_state["config"]
        tok_path = Path(args.tokenizer) if args.tokenizer else Path(args.resume).parent / "tokenizer.json"
        if not tok_path.exists():
            p.error("resume requires the checkpoint's tokenizer; supply --tokenizer")
        tokenizer = TinyTokenizer.load(tok_path)
        expected = resume_state.get("tokenizer_fingerprint")
        if expected is not None and expected != tokenizer.fingerprint:
            p.error("tokenizer vocabulary or merge rules differ from the resumed checkpoint")
        if tokenizer.vocab_size != cfg.text.vocab_size:
            p.error("tokenizer vocabulary size differs from the resumed checkpoint")
        model = TinyVLM(cfg)
        model.load_state_dict(resume_state["model"])
        model.to(device)
        print(f"[resume] {args.resume} at step {resume_state.get('step')}, "
              f"epoch {resume_state.get('epoch', 0) + 1}")
    elif args.init_from:
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
    val_loader = None
    if args.val_data:
        if args.indexed:
            val_dataset = IndexedTextDataset(args.val_data, tokenizer=tokenizer,
                                             max_seq_len=cfg.text.max_seq_len,
                                             ignore_index=cfg.ignore_index)
        else:
            val_dataset = MultimodalDataset(args.val_data, tokenizer, cfg.vision.img_size,
                                            cfg.text.max_seq_len, cfg.ignore_index)
        print(f"[val data] {val_dataset.mode_counts}")
        if args.val_limit and args.val_limit < len(val_dataset):
            val_dataset = Subset(val_dataset, range(args.val_limit))
            print(f"[val data] limited to {args.val_limit} samples for mid-run checks")
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (no_decay if param.dim() < 2 else decay).append(param)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))
    batches_per_epoch = math.ceil(len(dataset) / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    computed_total = steps_per_epoch * args.epochs
    if args.max_steps:
        computed_total = min(computed_total, args.max_steps)
    step, epoch, start_batch = 0, 0, 0
    epoch_loss, epoch_tokens = 0.0, 0
    best_loss = float("inf")
    history: list[dict] = []
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if resume_state is not None:
        step = int(resume_state.get("step", 0))
        epoch = int(resume_state.get("epoch", 0))
        start_batch = int(resume_state.get("batch_in_epoch", 0))
        epoch_loss = float(resume_state.get("epoch_loss", 0.0))
        epoch_tokens = int(resume_state.get("epoch_tokens", 0))
        best_loss = float(resume_state.get("best_loss", float("inf")))
        history = list(resume_state.get("history", []))
        try:
            optimizer.load_state_dict(resume_state["optimizer"])
        except (KeyError, ValueError) as exc:
            p.error(f"could not restore optimiser state: {exc}")
        # A CPU checkpoint stores no AMP state (its GradScaler was disabled). Loading it
        # into an enabled CUDA scaler either raises or replays scale 1.0, which underflows
        # fp16 gradients for thousands of steps, so start a fresh scale in that case.
        scaler_state = resume_state.get("scaler") or {}
        if device.type == "cuda":
            if scaler_state and scaler_state.get("scale", 1.0) > 1.0:
                scaler.load_state_dict(scaler_state)
            else:
                scaler = torch.amp.GradScaler("cuda")
                print(f"[resume] fresh AMP grad scale {scaler.get_scale():.0f} for CUDA")
        print(f"[resume] restored step {step}, epoch {epoch + 1}, batch {start_batch}, "
              f"best loss {best_loss:.4f}")
    # On resume the original horizon is kept unless the caller passes a new one.
    if resume_state is not None and args.epochs == 3 and args.max_steps == 0:
        total_steps = int(resume_state.get("total_steps", computed_total))
    else:
        total_steps = computed_total
    if total_steps <= step:
        p.error(f"no updates left: step {step} >= total steps {total_steps}; "
                "increase --epochs or --max-steps")
    print(f"[schedule] {steps_per_epoch} updates/epoch, total {total_steps}, "
          f"batch {args.batch_size} x accum {args.grad_accum} = "
          f"{args.batch_size * args.grad_accum} sequences/update")

    run_config = {
        "architecture": args.config, "device": str(device),
        "train_data": args.data, "validation_data": args.val_data,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "lr": args.lr, "warmup": args.warmup, "weight_decay": args.weight_decay,
        "epochs": args.epochs, "max_steps": args.max_steps, "total_steps": total_steps,
        "max_minutes": args.max_minutes, "save_every": args.save_every,
        "val_every": args.val_every, "val_limit": args.val_limit,
        "indexed": args.indexed, "modality_weights": args.modality_weights,
        "seed": args.seed, "threads": torch.get_num_threads(),
        "init_from": args.init_from, "resume": args.resume,
        "tokenizer": str(out_dir / "tokenizer.json"),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "vocab_size": cfg.text.vocab_size,
        "train_samples": len(dataset), "train_modes": dict(Counter(dataset.modes)),
    }
    (out_dir / "run-config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg.save(out_dir / "config.json")

    def flush(history_epoch, next_epoch, next_batch, validate=False, train_loss=None,
              epoch_batches=None):
        nonlocal best_loss
        if validate and val_loader is not None:
            validation = evaluate(model, val_loader, device)
            print(f"[val] step {step}: {json.dumps(validation, ensure_ascii=False)}", flush=True)
            if validation["macro_loss"] < best_loss:
                best_loss = validation["macro_loss"]
                save_checkpoint(out_dir / "best.pt", model, tokenizer, step, dataset.mode_counts)
            history.append({"epoch": history_epoch, "step": step,
                            "epoch_batches": batch_index if epoch_batches is None else epoch_batches,
                            "train_loss": train_loss, "validation": validation})
        save_checkpoint(out_dir / "model.pt", model, tokenizer, step, dataset.mode_counts)
        save_training_state(out_dir / "state.pt", model, tokenizer, optimizer, scaler,
                            step=step, epoch=next_epoch, batch_in_epoch=next_batch,
                            epoch_loss=epoch_loss, epoch_tokens=epoch_tokens,
                            best_loss=best_loss, history=history, total_steps=total_steps,
                            args=args, dataset_modes=dataset.mode_counts)
        (out_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"[save] step {step} -> {out_dir / 'state.pt'}", flush=True)

    deadline = time.time() + args.max_minutes * 60 if args.max_minutes else None
    t0 = time.time()
    trained_at_start = step
    last_val_step = -1
    batch_index = start_batch
    model.train()
    stopped = False
    try:
        while epoch < args.epochs and step < total_steps and not stopped:
            if sampler is None:
                batch_sampler = EpochBatchSampler(len(dataset), args.batch_size, args.seed,
                                                  epoch, start_batch)
                loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate,
                                    num_workers=args.workers, pin_memory=device.type == "cuda")
            else:
                batch_sampler = None
                loader = DataLoader(dataset, shuffle=False, sampler=sampler, **loader_options)
            batch_index = start_batch
            for batches in update_batches(loader, args.grad_accum):
                lr = lr_at(step, total_steps, args.lr, args.warmup)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                loss_sum, tokens = train_update(model, batches, optimizer, scaler, device)
                epoch_loss += loss_sum
                epoch_tokens += tokens
                step += 1
                batch_index += len(batches)
                if step % args.log_every == 0 or step == total_steps:
                    elapsed = (time.time() - t0) / max(step - trained_at_start, 1)
                    print(f"epoch {epoch + 1}/{args.epochs} step {step}/{total_steps} "
                          f"loss {epoch_loss / max(epoch_tokens, 1):.4f} "
                          f"(last {loss_sum / tokens:.4f}) lr {lr:.2e} "
                          f"{elapsed:.2f}s/step", flush=True)
                if args.val_every and step % args.val_every == 0:
                    flush(epoch + 1, epoch, batch_index, validate=True,
                          train_loss=epoch_loss / max(epoch_tokens, 1))
                    last_val_step = step
                    model.train()
                elif args.save_every and step % args.save_every == 0:
                    flush(epoch + 1, epoch, batch_index)
                if step >= total_steps:
                    stopped = True
                    break
                if deadline is not None and time.time() >= deadline:
                    print(f"[stop] wall-clock budget reached at step {step}", flush=True)
                    stopped = True
                    break
            train_loss = epoch_loss / epoch_tokens if epoch_tokens else None
            if stopped:
                # Mid-epoch stop: keep the same epoch/batch position for resuming.
                flush(epoch + 1, epoch, batch_index,
                      validate=last_val_step != step, train_loss=train_loss)
                break
            # Epoch finished: the next resume continues at the start of the next epoch.
            next_epoch, next_batch = epoch + 1, 0
            epoch_loss, epoch_tokens = 0.0, 0
            flush(epoch + 1, next_epoch, next_batch,
                  validate=last_val_step != step, train_loss=train_loss,
                  epoch_batches=batches_per_epoch)
            last_val_step = step
            epoch, start_batch = next_epoch, 0
    except KeyboardInterrupt:
        print("\n[stop] interrupted; saving resumable state", flush=True)
        flush(epoch + 1, epoch, batch_index)
        raise SystemExit(130)
    cfg.save(out_dir / "config.json")
    print(f"[done] saved {out_dir / 'model.pt'}, state.pt, tokenizer.json, config.json, metrics.json")
    peak = peak_memory_mb(device)
    if peak is not None:
        print(f"[mem] peak GPU allocation {peak:.0f} MB")
    print(f"[mem] peak process RSS {peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
