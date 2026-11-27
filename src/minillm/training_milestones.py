"""Immutable paired model/training-state snapshots at committed optimizer steps."""

import os
import json
from pathlib import Path

import torch

from .assessment_io import read_json, report_directory
from .checkpoint import CheckpointError, atomic_torch_save, atomic_write, load_training_state
from .corpus_utils import file_info, sha256, verify_file
from .pilot_lineage import unseal
from .train import _tokens_at_cursor, _validate_progress, save_model_checkpoint
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


def require_identical_weights(expected, actual, context):
    if not isinstance(actual, dict) or actual.keys() != expected.keys():
        raise CheckpointError(f"{context}: tensor names/count differ")
    for name, tensor in expected.items():
        other = actual[name]
        if (not isinstance(other, torch.Tensor) or tensor.shape != other.shape
                or tensor.dtype != other.dtype or not torch.equal(tensor, other)):
            raise CheckpointError(f"{context}: tensor {name} differs in shape, dtype or values")


def verify_model_metadata(model, state, *, context):
    expected = {key: state[key] for key in ("architecture", "arch_version", "config",
                                           "tokenizer_fingerprint", "tokenizer_vocab_size")}
    expected.update(step=state["global_step"], tokens_seen=state["tokens_seen"], mode=state["manifest"]["mode"])
    if not isinstance(model, dict) or any(model.get(key) != value for key, value in expected.items()):
        raise CheckpointError(f"{context}: checkpoint counters, model config or tokenizer differ")
    if model.get("training_run_id", state["manifest"]["run_id"]) != state["manifest"]["run_id"]:
        raise CheckpointError(f"{context}: checkpoint training run differs")


def write_best_status(out, loss, record):
    status = "AVAILABLE" if record else "NO_VALIDATION" if loss is None else "UNAVAILABLE"
    write_json(out / "best-checkpoint-status.json", {"status": status, "historical_best_loss": loss,
                                                     "checkpoint": record})


def _best_reference(state, dataset, sampler):
    loss = state["best_loss"]
    recorded = state.get("best_checkpoint")
    if recorded is None:
        step = next((row["step"] for row in state["history"] if row["loss"] == loss), None)
    else:
        step = recorded.get("step") if isinstance(recorded, dict) else None
    if type(step) is not int or not 0 <= step <= state["global_step"] or loss is None:
        return None
    manifest, plan = state["manifest"], state["manifest"]["training"]
    epoch, update = divmod(step, plan["optimizer_updates_per_epoch"])
    samples = min(update * plan["grad_accum"] * plan["batch_size"], len(dataset))
    tokens = _tokens_at_cursor(manifest, dataset, sampler, epoch, samples)
    reference = {"step": step, "tokens_seen": tokens, "val_loss": loss}
    if recorded is not None and any(recorded.get(key) != value for key, value in reference.items()):
        return None
    return {**reference, "file": recorded.get("file") if recorded is not None else None}


def _best_state(out, state, reference, dataset, sampler):
    step, tokens = reference["step"], reference["tokens_seen"]
    if step == state["global_step"]:
        return state
    manifest = state["manifest"]
    milestone = out / "milestones" / f"step-{step}"
    if milestone.is_dir():
        verify_milestone(milestone, manifest, step=step, tokens=tokens)
        candidates = (milestone / "state.pt",)
    else:
        candidates = (out / "state.pt", out / "state.prev.pt")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            candidate = load_training_state(path, mmap=True)
            if (candidate["manifest"] == manifest and (candidate["global_step"], candidate["tokens_seen"]) == (step, tokens)
                    and (path.parent == milestone or any(row["step"] == step and row["loss"] == reference["val_loss"]
                                                         for row in candidate["history"]))):
                _validate_progress(candidate, manifest, dataset, sampler)
                return candidate
        except CheckpointError:
            continue
    return None


def reconcile_best_checkpoint(out, state, checkpoint, dataset, sampler):
    """Reconcile the published best without altering the committed validation history."""
    manifest = state["manifest"]
    reference = _best_reference(state, dataset, sampler)
    best = out / "best.pt"
    lineage_path = out / "pilot-lineage.json"
    lineage = unseal(read_json(lineage_path)) if lineage_path.is_file() else None
    if lineage is not None and lineage["training_manifest"] != manifest:
        raise CheckpointError("best checkpoint recovery lineage differs from the resumed run")
    anchor = _best_state(out, state, reference, dataset, sampler) if reference else None
    try:
        published = read_json(out / "best-checkpoint-status.json")
    except (OSError, ValueError):
        published = None

    def validated(path):
        if reference is None or not path.is_file():
            return None
        info = file_info(path)
        if reference["file"] is not None and info != reference["file"]:
            # A prior repair can serialize the same trusted weights with a new SHA.
            if (path != best or anchor is None or published != {"status": "AVAILABLE",
                    "historical_best_loss": reference["val_loss"], "checkpoint": {**reference, "file": info}}):
                return None
        try:
            model = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        except Exception:
            # torch.load can raise multiple format errors for a partial save.
            return None
        try:
            verify_model_metadata(model, {**state, "global_step": reference["step"],
                                          "tokens_seen": reference["tokens_seen"]}, context=str(path))
            if model["val_loss"] != reference["val_loss"]:
                return None
            if lineage is not None:
                request = model.get("pilot_request_id")
                allowed = {lineage["initial_launch"]["request_id"]} | {
                    row["request_id"] for row in lineage["resume_history"] if row["checkpoint_step"] <= reference["step"]}
                if (model.get("pilot_initialization") != lineage["initialization"]
                        or (request not in allowed and not (request is None and model.get("training_run_id") == manifest["run_id"]))):
                    return None
            if anchor is not None:
                require_identical_weights(anchor["model"], model["model"], str(path))
            elif reference["file"] != info and not (reference["file"] is None and lineage is not None and path == best):
                return None
            return {**reference, "file": info}
        except (CheckpointError, KeyError):
            # An unreadable inference artifact must not block an otherwise valid state recovery.
            return None

    record = validated(best)
    if record is None and best.is_file():
        info = file_info(best)
        archive = out / "recovery" / f"best-step-{state['global_step']}-{info['sha256']}"
        if not archive.exists():
            with report_directory(archive) as directory:
                os.link(best, directory / best.name)
                write_json(directory / "manifest.json", {"run_id": manifest["run_id"],
                           "checkpoint": file_info(checkpoint), "recovered_step": state["global_step"],
                           "discarded_best": info})
        verify_file(archive / best.name, info)
        best.unlink()
    if record is None and reference:
        for path in sorted((out / "recovery").glob("best-step-*/best.pt")):
            record = validated(path)
            if record is not None:
                os.link(path, best)
                break
        if record is None and anchor is not None:
            payload = {key: anchor[key] for key in ("architecture", "arch_version", "config", "model",
                                                     "tokenizer_fingerprint", "tokenizer_vocab_size")}
            payload.update(step=reference["step"], tokens_seen=reference["tokens_seen"],
                           val_loss=reference["val_loss"], mode=manifest["mode"], training_run_id=manifest["run_id"])
            policy = manifest["dataset"]["train"].get("text_policy", "identity-v1")
            if policy != "identity-v1":
                payload["text_policy"] = policy
            if lineage is not None:
                payload["pilot_initialization"] = lineage["initialization"]
                payload["seq_len"] = manifest["training"]["seq_len"]
            atomic_torch_save(payload, best)
            record = {**reference, "file": file_info(best)}
    write_best_status(out, state["best_loss"], record)
    return record or state.get("best_checkpoint")


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


def reconcile_recovery_log(out, state, checkpoint, *, initial_step=0):
    """Archive logged work beyond the authoritative committed checkpoint before replay."""
    path = out / "training-log.jsonl"
    if not path.is_file():
        if state["global_step"] != initial_step:
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
    metrics_differ = (read_json(metrics) if metrics.is_file() else []) != state["history"]
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
