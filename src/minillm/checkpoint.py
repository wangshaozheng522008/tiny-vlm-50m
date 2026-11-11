"""Crash-safe checkpoint files. Only load checkpoints from trusted sources."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path

import torch

from .config import ARCH_VERSION, ARCHITECTURE, MiniLLMConfig

TRAINING_STATE_VERSION = 2


class CheckpointError(RuntimeError):
    """Invalid, incompatible or incomplete checkpoint."""


def _sync_directory(directory: Path) -> None:
    if os.name == "posix":
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write(path: str | Path, writer, *, backup: Path | None = None) -> None:
    """Write+fsync, optionally preserve the old inode, then replace on the same FS.

    At most one new full checkpoint is written. A hard link rotates the old
    state without copying AdamW tensors; filesystems without links use a copy.
    Uncatchable process death can leave hidden .tmp files, which are never loaded.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    previous: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if backup is not None and path.exists():
            fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{backup.name}.", suffix=".tmp")
            os.close(fd)
            previous = Path(name)
            previous.unlink()
            try:
                os.link(path, previous)
            except OSError:
                with path.open("rb") as source, previous.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
            os.replace(previous, backup)
            _sync_directory(path.parent)
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        for item in (temporary, previous):
            if item is not None:
                item.unlink(missing_ok=True)


def atomic_torch_save(payload: dict, path: str | Path, *, backup: Path | None = None) -> None:
    atomic_write(path, lambda handle: torch.save(payload, handle), backup=backup)


def validate_training_state(payload: object, path: str | Path) -> dict:
    if not isinstance(payload, dict):
        raise CheckpointError(f"{path}: training state must be a dictionary")
    if payload.get("architecture") != ARCHITECTURE or payload.get("arch_version") != ARCH_VERSION:
        raise CheckpointError(f"{path}: incompatible MiniLLM architecture/version")
    version = payload.get("training_state_version")
    if version != TRAINING_STATE_VERSION:
        raise CheckpointError(
            f"{path}: unsupported training_state_version={version!r}. Legacy state.pt has an "
            "ambiguous batch_in_epoch cursor and no verifiable training plan/data manifest; "
            "safe resume is impossible. Use --init-from for an explicit weights-only new run."
        )
    required = {
        "config": dict, "model": dict, "optimizer": dict, "manifest": dict,
        "rng": dict, "history": list, "tokenizer_fingerprint": str,
        "tokenizer_vocab_size": int, "global_step": int, "epoch": int,
        "micro_batches_consumed_in_epoch": int, "samples_seen": int, "tokens_seen": int,
    }
    for key, kind in required.items():
        if key not in payload or not isinstance(payload[key], kind):
            raise CheckpointError(f"{path}: missing/invalid training state field {key!r}")
    for key in ("global_step", "epoch", "micro_batches_consumed_in_epoch", "samples_seen", "tokens_seen"):
        if type(payload[key]) is not int or payload[key] < 0:
            raise CheckpointError(f"{path}: invalid nonnegative counter {key!r}")
    try:
        MiniLLMConfig.from_dict(payload["config"])
        manifest = payload["manifest"]
        plan = manifest["training"]
        groups = payload["optimizer"]["param_groups"]
        valid = (
            bool(payload["model"]) and all(torch.is_tensor(v) for v in payload["model"].values())
            and isinstance(payload["optimizer"]["state"], dict)
            and isinstance(groups, list) and len(groups) == 1
            and isinstance(payload["history"], list)
            and (payload["scaler"] is None or isinstance(payload["scaler"], dict))
            and (payload["best_loss"] is None or math.isfinite(payload["best_loss"]))
            and all(k in payload["rng"] for k in ("python", "numpy", "torch", "cuda"))
            and torch.is_tensor(payload["rng"]["torch"])
            and payload["rng"]["torch"].dtype == torch.uint8
            and isinstance(payload["rng"]["python"], tuple)
            and isinstance(payload["rng"]["numpy"], tuple)
            and (payload["rng"]["cuda"] is None or isinstance(payload["rng"]["cuda"], list))
            and manifest["training_state_version"] == TRAINING_STATE_VERSION
            and manifest["model_architecture"] == ARCHITECTURE
            and manifest["model_config"] == payload["config"]
            and manifest["tokenizer_fingerprint"] == payload["tokenizer_fingerprint"]
            and manifest["tokenizer_vocab_size"] == payload["tokenizer_vocab_size"]
            and isinstance(manifest["dataset"]["train"], dict)
            and "val" in manifest["dataset"] and isinstance(manifest["execution"], dict)
            and manifest["mode"] in ("pretrain", "sft")
            and plan["optimizer"] == "AdamW" and plan["scheduler"] == "warmup_cosine"
            and plan["target_optimizer_updates"] > 0
            and plan["micro_batches_per_epoch"] > 0 and plan["optimizer_updates_per_epoch"] > 0
            and tuple(groups[0]["betas"]) == tuple(plan["betas"])
            and groups[0]["eps"] == plan["eps"]
            and groups[0]["weight_decay"] == plan["weight_decay"]
            and math.isfinite(groups[0]["lr"]) and groups[0]["lr"] >= 0
        )
        if not valid:
            raise ValueError("invalid state structure or optimizer parameters")
        parameter_ids = groups[0]["params"]
        optimizer_state = payload["optimizer"]["state"]
        if payload["global_step"] > 0:
            if not parameter_ids or set(optimizer_state) != set(parameter_ids):
                raise ValueError("missing AdamW parameter state after completed updates")
            for slot in optimizer_state.values():
                if not (torch.is_tensor(slot["step"]) and slot["step"].numel() == 1
                        and slot["step"].item() == payload["global_step"]
                        and torch.is_tensor(slot["exp_avg"]) and torch.is_tensor(slot["exp_avg_sq"])
                        and slot["exp_avg"].shape == slot["exp_avg_sq"].shape):
                    raise ValueError("invalid AdamW moments or update count inconsistent with global_step")
        elif optimizer_state:
            raise ValueError("optimizer state exists before any completed update")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise CheckpointError(f"{path}: incomplete/invalid model, optimizer, RNG or manifest: {exc}") from exc
    return payload


def load_training_state(path: str | Path, *, mmap: bool = False) -> dict:
    path = Path(path)
    if path.name.startswith(".") or path.suffix == ".tmp":
        raise CheckpointError(f"{path}: temporary files cannot be loaded as checkpoints")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
        return validate_training_state(payload, path)
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(
            f"{path}: cannot load training checkpoint: {exc}. "
            "If available, explicitly resume with --resume /path/to/state.prev.pt."
        ) from exc
