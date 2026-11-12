"""Supervised fine-tuning: instruction data with assistant-only loss.

Input is JSONL with a ``messages`` list (system/user/assistant). The chat
template comes from :mod:`minillm.tokenizer` and is shared with the web UI, so
the training and inference formats cannot drift apart.

    python -m minillm.sft --init-from runs/minillm/model.pt \
        --data data/sft/train.jsonl --val-data data/sft/val.jsonl \
        --tokenizer runs/minillm/tokenizer.json --out runs/minillm-sft \
        --batch-size 2 --grad-accum 8 --lr 1e-4 --epochs 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import MiniLLMConfig, get_config
from .data import EpochBatchSampler, SFTDataset
from .model import MiniLLM
from .tokenizer import DEFAULT_SYSTEM, MiniLLMTokenizer
from .train import (
    CheckpointError,
    TrainingOptions,
    fit,
    load_model_checkpoint,
    load_training_state,
    make_pretrain_loader,
)
from .utils import get_device, set_seed
from .text_policy import IDENTITY, checkpoint_text_policy


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Supervised fine-tuning for MiniLLM")
    p.add_argument("--data", required=True, help='JSONL with {"messages": [...]}')
    p.add_argument("--val-data", help="held-out SFT JSONL")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--config", default=None, help="fresh preset (default minillm-87m), or explicit resume config check")
    p.add_argument("--init-from", help="pretraining checkpoint to start from")
    p.add_argument("--resume", help="SFT state.pt to resume from")
    p.add_argument("--out", default="runs/minillm-sft")
    p.add_argument("--system", default=DEFAULT_SYSTEM,
                   help="default system prompt when a record has none")
    p.add_argument("--last-assistant-turn-only", action="store_true",
                   help="supervise only the final assistant message")
    p.add_argument("--max-seq-len", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=0, help="cumulative target updates; >0 overrides --epochs")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--val-limit", type=int, default=0)
    p.add_argument("--max-minutes", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--threads", type=int, default=0)
    return p


def sft_batch_factory(dataset: Dataset, batch_size: int, seed: int, pad: int, ignore: int):
    """Deterministic per-epoch shuffling with a resumable batch offset."""

    sampler = EpochBatchSampler(len(dataset), batch_size, seed=seed)

    def factory(epoch: int, start_batch: int):
        return make_pretrain_loader(dataset, sampler, epoch, start_batch=start_batch,
                                    pad_token_id=pad, ignore_index=ignore)

    factory.sampler_seed = seed

    return factory


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)  # fresh weights must be seeded before model construction
    device = get_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    policy = IDENTITY

    if args.resume:
        state = load_training_state(args.resume, mmap=True)
        policy = checkpoint_text_policy(state)
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
        policy = checkpoint_text_policy(payload)
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
        print("[init] fresh weights (no --init-from); SFT from scratch is only useful for tests")
    tokenizer.assert_vocab_size(model.cfg.vocab_size)
    tokenizer.text_policy = policy

    seq_len = args.max_seq_len or model.cfg.max_position_embeddings
    train_set = SFTDataset(
        args.data, tokenizer, seq_len,
        supervise_all_assistant_turns=not args.last_assistant_turn_only,
        system=args.system,
    )
    val_set = SFTDataset(
        args.val_data, tokenizer, seq_len,
        supervise_all_assistant_turns=not args.last_assistant_turn_only,
        system=args.system,
    ) if args.val_data else None
    print(f"[data] sft train: {train_set.stats}")
    if val_set is not None:
        print(f"[data] sft val: {val_set.stats}")

    opts = TrainingOptions(
        out=args.out, epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, min_lr=args.min_lr, warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip, max_steps=args.max_steps,
        save_every=args.save_every, val_every=args.val_every, val_limit=args.val_limit,
        max_minutes=args.max_minutes, log_every=args.log_every, seed=args.seed,
        device=args.device, precision=args.precision,
        grad_checkpointing=args.grad_checkpointing, threads=args.threads,
        resume=args.resume, init_from=args.init_from,
        extra_meta={
            "sft_data": args.data,
            "sft_val": args.val_data,
            "max_seq_len": seq_len,
            "supervise_all_assistant_turns": not args.last_assistant_turn_only,
            "train_stats": train_set.stats,
        },
    )
    factory = sft_batch_factory(
        train_set, args.batch_size, args.seed, model.cfg.pad_token_id, model.cfg.ignore_index
    )
    fit(opts, model, tokenizer, train_set, val_set, mode="sft", batch_factory=factory)


if __name__ == "__main__":
    main()
