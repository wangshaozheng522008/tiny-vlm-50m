"""Content identity and a fixed training plan, checked before mutating a run."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
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


def implementation_identity(*entrypoints: Path) -> dict:
    """Hash the runtime package and the experiment's entrypoints/configuration."""
    package = Path(__file__).resolve().parent
    root = package.parents[1]
    paths = set(package.rglob("*.py")) | {Path(path).resolve() for path in entrypoints}
    files = {str(path.relative_to(root)): file_sha256(path) for path in sorted(paths)}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return {"git_commit": commit, "files": files,
            "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


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
        version_path = dataset.path.parent / "dataset-manifest.json"
        if version_path.exists():
            from .corpus_utils import identity as content_identity
            try:
                version = json.loads(version_path.read_text())
                expected = version["files"][dataset.path.name]
                if (version["schema_version"] != 1
                        or content_identity(version["identity"]) != version["content_id"]
                        or version["identity"]["files"] != version["files"]
                        or expected != {"bytes": dataset.path.stat().st_size, "sha256": identity["sha256"]}):
                    raise ValueError("dataset version/file hash mismatch")
                identity["dataset_content_id"] = version["content_id"]
                identity["dataset_tokenizer_fingerprint"] = version["tokenizer"]["fingerprint"]
                if "text_policy" in version.get("content_config", {}):
                    from .text_policy import validate_policy
                    identity["text_policy"] = validate_policy(version["content_config"]["text_policy"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise CheckpointError(f"invalid published dataset manifest {version_path}: {exc}") from exc
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
        if getattr(dataset, "text_policy", "identity-v1") != "identity-v1":
            identity["text_policy"] = dataset.text_policy
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
    from .text_policy import IDENTITY
    policy = train.get("text_policy", IDENTITY)
    if val and val.get("text_policy", IDENTITY) != policy:
        raise CheckpointError("train/validation text policies differ")
    for dataset in (train, val):
        if dataset and dataset.get("dataset_tokenizer_fingerprint", tokenizer.fingerprint) != tokenizer.fingerprint:
            raise CheckpointError("published dataset tokenizer fingerprint does not match training tokenizer")
    if train["seq_len"] < 2 or train["seq_len"] > model.cfg.max_position_embeddings:
        raise ValueError("sequence length must be in [2, max_position_embeddings]")
    updates = math.ceil(len(sampler) / opts.grad_accum)
    target = opts.max_steps if opts.max_steps > 0 else updates * opts.epochs
    warmup = int(target * opts.warmup_ratio)
    milestones = opts.milestone_steps
    if (not isinstance(milestones, (list, tuple)) or any(type(step) is not int or not 0 <= step <= target for step in milestones)
            or list(milestones) != sorted(set(milestones))):
        raise ValueError("milestone_steps must be unique, increasing completed steps within the training plan")
    return {
        "training_state_version": TRAINING_STATE_VERSION,
        "model_architecture": ARCHITECTURE, "model_config": model.cfg.to_dict(),
        "tokenizer_fingerprint": tokenizer.fingerprint, "tokenizer_vocab_size": tokenizer.vocab_size,
        "mode": mode, "dataset": {"train": train, "val": val},
        **({"provenance": opts.provenance} if opts.provenance else {}),
        "training": {
            "seed": opts.seed, "sampler_seed": sampler.seed,
            "batch_size": opts.batch_size, "grad_accum": opts.grad_accum,
            "seq_len": train["seq_len"], "max_steps": opts.max_steps,
            "epochs": opts.epochs if opts.max_steps == 0 else None,
            "target_optimizer_updates": target, "micro_batches_per_epoch": len(sampler),
            "optimizer_updates_per_epoch": updates,
            "learning_rate": opts.lr, "min_learning_rate": opts.min_lr,
            "warmup_ratio": opts.warmup_ratio, "warmup_steps": warmup,
            "scheduler": "warmup_cosine", "scheduler_total_steps": target - int(opts.cosine_final_update),
            **({"cosine_final_update": True} if opts.cosine_final_update else {}),
            **({"validate_before_milestone": True} if opts.validate_before_milestone else {}),
            "optimizer": "AdamW", "betas": [0.9, 0.95], "eps": 1e-8,
            "weight_decay": opts.weight_decay, "grad_clip": opts.grad_clip,
            **({"milestone_steps": list(milestones)} if milestones else {}),
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
