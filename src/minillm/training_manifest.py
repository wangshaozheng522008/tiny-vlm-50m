"""Content identity and a fixed training plan, checked before mutating a run."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from .checkpoint import CheckpointError, TRAINING_STATE_VERSION, atomic_write
from .config import ARCHITECTURE
from .data import PackedPretrainDataset, SFTDataset


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_identity(dataset, ignore_index: int) -> tuple[dict | None, int]:
    if dataset is None:
        return None, 0
    identity = {
        "path": str(dataset.path.resolve()), "sha256": file_sha256(dataset.path),
        "type": type(dataset).__name__, "samples": len(dataset),
    }
    if isinstance(dataset, PackedPretrainDataset):
        identity["seq_len"] = dataset.seq_len
        tokens = len(dataset) * (dataset.seq_len - 1)  # model shifts labels inside each window
    elif isinstance(dataset, SFTDataset):
        identity.update(seq_len=dataset.max_seq_len, system=dataset.system,
                        supervise_all_assistant_turns=dataset.supervise_all)
        # Also cover in-memory records overrides, filtering and tokenisation results.
        digest = hashlib.sha256()
        tokens = 0
        for item in dataset.items:
            for key in ("input_ids", "labels"):
                values = item[key].numpy().astype("<i8", copy=False)
                digest.update(len(values).to_bytes(8, "little"))
                digest.update(values.tobytes())
            tokens += int((item["labels"][1:] != ignore_index).sum())
        identity["effective_sha256"] = digest.hexdigest()
    else:
        raise ValueError("reliable training requires PackedPretrainDataset or SFTDataset")
    identity["effective_loss_tokens"] = tokens
    return identity, tokens


def build_manifest(opts, model, tokenizer, train_dataset, val_dataset, *, mode, sampler, precision, device):
    if mode not in ("pretrain", "sft"):
        raise ValueError("training mode must be pretrain or sft")
    if opts.batch_size <= 0 or opts.grad_accum <= 0 or opts.epochs <= 0 or opts.max_steps < 0:
        raise ValueError("batch_size, grad_accum and epochs must be positive; max_steps must be >= 0")
    if opts.log_every < 0 or opts.save_every < 0 or opts.val_every < 0 or opts.val_limit < 0:
        raise ValueError("logging, save and validation intervals must be >= 0 (0 disables)")
    if not (math.isfinite(opts.lr) and math.isfinite(opts.min_lr) and 0 <= opts.min_lr <= opts.lr and opts.lr > 0):
        raise ValueError("learning rates must be finite with 0 <= min_lr <= lr and lr > 0")
    if not math.isfinite(opts.warmup_ratio) or not 0 <= opts.warmup_ratio <= 1:
        raise ValueError("warmup_ratio must be finite and in [0, 1]")
    if not math.isfinite(opts.grad_clip) or opts.grad_clip <= 0:
        raise ValueError("grad_clip must be finite and positive")
    if not math.isfinite(opts.weight_decay) or opts.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not math.isfinite(opts.max_minutes) or opts.max_minutes < 0:
        raise ValueError("max_minutes must be finite and nonnegative")
    train, tokens = dataset_identity(train_dataset, model.cfg.ignore_index)
    val, _ = dataset_identity(val_dataset, model.cfg.ignore_index)
    if train["seq_len"] < 2 or train["seq_len"] > model.cfg.max_position_embeddings:
        raise ValueError("sequence length must be in [2, max_position_embeddings]")
    updates = math.ceil(len(sampler) / opts.grad_accum)
    target = opts.max_steps if opts.max_steps > 0 else updates * opts.epochs
    warmup = int(target * opts.warmup_ratio)
    return {
        "training_state_version": TRAINING_STATE_VERSION,
        "model_architecture": ARCHITECTURE, "model_config": model.cfg.to_dict(),
        "tokenizer_fingerprint": tokenizer.fingerprint, "tokenizer_vocab_size": tokenizer.vocab_size,
        "mode": mode, "dataset": {"train": train, "val": val},
        "training": {
            "seed": opts.seed, "sampler_seed": sampler.seed,
            "batch_size": opts.batch_size, "grad_accum": opts.grad_accum,
            "seq_len": train["seq_len"], "max_steps": opts.max_steps,
            "epochs": opts.epochs if opts.max_steps == 0 else None,
            "target_optimizer_updates": target, "micro_batches_per_epoch": len(sampler),
            "optimizer_updates_per_epoch": updates,
            "learning_rate": opts.lr, "min_learning_rate": opts.min_lr,
            "warmup_ratio": opts.warmup_ratio, "warmup_steps": warmup,
            "scheduler": "warmup_cosine", "scheduler_total_steps": target,
            "optimizer": "AdamW", "betas": [0.9, 0.95], "eps": 1e-8,
            "weight_decay": opts.weight_decay, "grad_clip": opts.grad_clip,
        },
        # Strict policy: changes require a new weights-only run, not an equivalent resume.
        "execution": {"precision": precision, "device": str(device),
                      "threads": torch.get_num_threads(),
                      "grad_checkpointing": opts.grad_checkpointing},
    }, tokens


def compare_manifests(saved: dict, current: dict) -> None:
    def differences(a, b, prefix=""):
        if isinstance(a, dict) and isinstance(b, dict):
            return [item for key in sorted(a.keys() | b.keys())
                    for item in differences(a.get(key), b.get(key), f"{prefix}.{key}".strip("."))]
        return [prefix] if a != b else []

    changed = differences(saved, current)
    if changed:
        raise CheckpointError(
            "resume incompatible with saved manifest: " + ", ".join(changed)
            + ". Resume restores the original plan; extend is not implemented. "
            "Use --init-from only for an intentional new training run."
        )


def write_json(path: Path, payload) -> None:
    data = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    atomic_write(path, lambda handle: handle.write(data))
