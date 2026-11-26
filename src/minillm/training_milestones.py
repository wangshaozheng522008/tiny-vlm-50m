"""Immutable paired model/training-state snapshots at committed optimizer steps."""

import os
import json
from pathlib import Path

import torch

from .assessment_io import read_json, report_directory
from .checkpoint import CheckpointError, atomic_write, load_training_state
from .corpus_utils import file_info, sha256, verify_file
from .train import save_model_checkpoint
from .training_manifest import write_json


def verify_milestone(path, manifest, *, step, tokens):
    path = Path(path)
    receipt = read_json(path / "manifest.json")
    if (receipt["schema_version"], receipt["global_step"], receipt["tokens_seen"],
            receipt["training_manifest"]) != (1, step, tokens, manifest):
        raise CheckpointError(f"{path}: milestone belongs to a different training state")
    for name in ("state.pt", "model.pt"):
        verify_file(path / name, receipt["files"][name])
    return receipt


def save_milestone(out, model, tokenizer, manifest, *, step, tokens, extra):
    primary = out / "state.pt"
    path = out / "milestones" / f"step-{step}"
    if path.exists():
        receipt = verify_milestone(path, manifest, step=step, tokens=tokens)
        if file_info(primary) != receipt["files"]["state.pt"]:
            # Recovery may reserialize metadata for this step. Ensure a replay
            # has the same weights and AdamW buffers before retaining the node.
            saved, current = (load_training_state(p, mmap=True) for p in (path / "state.pt", primary))
            if (saved["optimizer"]["param_groups"] != current["optimizer"]["param_groups"]
                    or any(not torch.equal(t, current["model"][k]) for k, t in saved["model"].items())
                    or any(not torch.equal(t, current["optimizer"]["state"][p][k])
                           for p, slot in saved["optimizer"]["state"].items() for k, t in slot.items())):
                raise CheckpointError(f"{path}: replay differs from its immutable milestone")
        return
    with report_directory(path) as directory:
        # The primary is replaced atomically by the trainer, never changed in
        # place. Its inode is retained by this distinct archive name.
        os.link(primary, directory / "state.pt")
        save_model_checkpoint(directory / "model.pt", model, tokenizer, step=step, tokens=tokens,
                              val_loss=None, extra=extra)
        write_json(directory / "manifest.json", {
            "schema_version": 1, "global_step": step, "tokens_seen": tokens,
            "training_manifest": manifest,
            "files": {name: file_info(directory / name) for name in ("state.pt", "model.pt")},
        })


def reconcile_recovery_log(out, state, checkpoint):
    """Archive logged work beyond the authoritative committed checkpoint before replay."""
    path = out / "training-log.jsonl"
    if not path.is_file():
        if state["global_step"]:
            raise CheckpointError("milestone-enabled resume requires the original per-update log")
        return
    before = path.read_bytes()
    lines = before.splitlines(keepends=True)
    kept = []
    step = state["global_step"]
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            if i == len(lines) - 1 and not line.endswith(b"\n"):
                break  # an incomplete append after a crash
            raise
        if row.get("global_step", 0) > step:
            continue
        if (row.get("event") == "validation" and row.get("global_step") == step
                and not any(entry["step"] == step and entry["loss"] == row["loss"] for entry in state["history"])):
            continue
        kept.append(line if line.endswith(b"\n") else line + b"\n")
    after = b"".join(kept)
    metrics = out / "metrics.json"
    metrics_differ = metrics.is_file() and read_json(metrics) != state["history"]
    if after != before or metrics_differ:
        scope = f"step-{step}-{sha256(path)}"
        archive = out / "recovery" / scope
        if not archive.exists():
            with report_directory(archive) as directory:
                (directory / path.name).write_bytes(before)
                if metrics.is_file():
                    (directory / metrics.name).write_bytes(metrics.read_bytes())
                write_json(directory / "manifest.json", {"run_id": state["manifest"]["run_id"],
                           "recovered_step": step, "checkpoint": file_info(checkpoint),
                           "original_log": file_info(path), "removed_rows": len(lines) - len(kept)})
        if after != before:
            atomic_write(path, lambda handle: handle.write(after))
        if metrics_differ:
            write_json(metrics, state["history"])
