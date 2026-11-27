"""An independent FP32 continuation of the committed V5 10M AdamW state."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import math
from pathlib import Path
import shutil
import signal
import time
import uuid

import torch

from . import v5_language_training as profile
from .assessment_io import environment, read_json, records, write_json
from .checkpoint import CheckpointError, atomic_write, load_training_state
from .config import MiniLLMConfig
from .corpus_budget import read_token_index
from .corpus_utils import file_info, verify_file
from .data import EpochBatchSampler, PackedPretrainDataset, collate_batch
from .model import MiniLLM
from .tokenizer import MiniLLMTokenizer
from .train import (NumericalError, TrainingOptions, _append_log, _train_step, _validate_progress,
                    evaluate, lr_at, save_model_checkpoint, save_training_state, set_rng_state)
from .training_manifest import dataset_identity
from .training_milestones import (reconcile_recovery_log, require_identical_weights, save_milestone,
                                  verify_milestone, verify_model_metadata, write_best_status)
from .v4_admission import validate_v4_pilot_policy
from .v5_repeat_sampling import RepeatPlan, accounting

CONFIG = profile.ROOT / "configs/pilot-v5-language-20m-repeat-t1200.json"
AUTHORIZATION = profile.ROOT / "configs/v5-language-20m-repeat-authorization.json"
RUN = profile.ROOT / "runs/v5-language-pilot-20m-repeat-v1"
REPORT = profile.ROOT / "reports/v5-language-pilot-20m-repeat-v1"
PARENT_STEP = 1222
PARENT_WINDOWS = 9776
PARENT_TOKENS = 10_000_848
PARENT_STATE = {"bytes": 1_101_445_353,
                "sha256": "97529f5f1b4fe0e9a4a34c3b6ead1620f57ab6f2567d6dfdf9d503e108f325ea"}
PARENT_MODEL = {"bytes": 409_075_993,
                "sha256": "9c4912a7e36455de328dc37cb925089a14f08492f0bbc5d280bc1945ea7db702"}
MILESTONES = (1467, 1711, 1955, 2200, 2444)
WINDOWS_PER_UPDATE = 8
SUPERVISED_PER_WINDOW = 1023
TOKENS_PER_UPDATE = WINDOWS_PER_UPDATE * SUPERVISED_PER_WINDOW


def _settings():
    settings = read_json(CONFIG)
    expected = {
        "out": str(RUN.relative_to(profile.ROOT)), "report": str(REPORT.relative_to(profile.ROOT)),
        "sampler_mode": "continue_frozen_epochs", "parent_step": PARENT_STEP,
        "parent_effective_tokens": PARENT_TOKENS, "stage_steps": PARENT_STEP,
        "stage_effective_tokens": PARENT_TOKENS, "cumulative_effective_tokens": 2 * PARENT_TOKENS,
        "seq_len": 1024, "batch_size": 1, "grad_accum": WINDOWS_PER_UPDATE,
        "precision": "fp32", "device": "cuda", "threads": 1, "grad_checkpointing": True,
        "seed": 42, "lr": 0.00015, "min_lr": 0.00001, "warmup_steps": 20,
        "save_every": 50, "val_every": 100, "gpu_memory_limit_mib": 2971.75,
        "milestone_steps": list(MILESTONES),
    }
    if settings != expected:
        raise ValueError("training configuration differs from the independently scoped repeat plan")
    return settings


def _authorization(settings):
    evidence = read_json(AUTHORIZATION)
    if any(evidence.get(key) != value for key, value in {
            "experiment": "v5-language-20m-repeat-v1", "corpus_version": profile.V5_VERSION,
            "content_id": profile.CONTENT_ID, "parent_state_sha256": PARENT_STATE["sha256"],
            "purpose": "internal_research", "additional_effective_tokens": settings["stage_effective_tokens"],
            "cumulative_effective_token_cap": settings["cumulative_effective_tokens"],
    }.items()) or not evidence.get("request_excerpt") or not evidence.get("source"):
        raise ValueError("repeat-stage authorization is missing or outside the requested scope")
    return {"evidence": evidence, "file": file_info(AUTHORIZATION)}


def _load_model(path):
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def parent_checkpoint(train, val, sampler):
    run = profile.RUN
    verify_file(run / "state.pt", PARENT_STATE)
    verify_file(run / "model.pt", PARENT_MODEL)
    state = load_training_state(run / "state.pt", mmap=True)
    manifest = read_json(run / "training-manifest.json")
    summary = read_json(run / "train-summary.json")
    plan = manifest["training"]
    if (state["manifest"] != manifest or summary["stop_reason"] != "target_reached"
            or (state["global_step"], state["samples_seen"], state["tokens_seen"], state["epoch"],
                state["micro_batches_consumed_in_epoch"]) != (PARENT_STEP, PARENT_WINDOWS, PARENT_TOKENS, 0, PARENT_WINDOWS)
            or any(state[key] != summary[key] for key in ("global_step", "samples_seen", "tokens_seen", "epoch",
                                                          "micro_batches_consumed_in_epoch"))
            or (plan["seq_len"], plan["batch_size"], plan["grad_accum"], plan["sampler_seed"]) != (1024, 1, 8, 42)
            or manifest["execution"]["precision"] != "fp32"
            or any(dataset_identity(data, manifest["model_config"]["ignore_index"])[0] != manifest["dataset"][split]
                   for split, data in (("train", train), ("val", val)))):
        raise CheckpointError("historical checkpoint differs from the pinned FP32 sampler and plan")
    _validate_progress(state, manifest, train, sampler)
    path = run / "milestones" / "step-1222"
    node = verify_milestone(path, manifest, step=PARENT_STEP, tokens=PARENT_TOKENS)
    milestone_state = load_training_state(path / "state.pt", mmap=True)
    if any(milestone_state[key] != state[key] for key in (
            "manifest", "global_step", "samples_seen", "tokens_seen", "epoch", "micro_batches_consumed_in_epoch")):
        raise CheckpointError("historical milestone state disagrees with final state")
    for model_path in (run / "model.pt", path / "model.pt"):
        model = _load_model(model_path)
        verify_model_metadata(model, state, context=str(model_path))
        require_identical_weights(state["model"], model["model"], str(model_path))
    require_identical_weights(state["model"], milestone_state["model"], str(path / "state.pt"))
    if len(state["model"]) != 147 or any(tensor.dtype != torch.float32 for tensor in state["model"].values()):
        raise CheckpointError("historical checkpoint must contain 147 FP32 tensors")
    parent = {"main_commit": "343595694ba83631991887aed9722ac9b8b833a9", "run_id": manifest["run_id"],
              "state": PARENT_STATE, "model": PARENT_MODEL, "milestone": node["files"],
              "milestone_manifest": file_info(path / "manifest.json"), "matching_fp32_tensors": len(state["model"]),
              "training_manifest": file_info(run / "training-manifest.json"),
              "pilot_lineage": file_info(run / "pilot-lineage.json")}
    return state, parent


@dataclass(frozen=True)
class PreparedStage:
    settings: dict
    train: PackedPretrainDataset
    val: PackedPretrainDataset
    tokenizer: MiniLLMTokenizer
    parent: dict
    lineage: dict
    authorization: dict
    replay: RepeatPlan
    preview: dict
    documents: list[dict]
    corpus_plan: dict

    def manifest(self):
        return stage_manifest(self.parent, self.lineage, self.settings, self.authorization, self.preview)


def prepare(*, publish=True, check_hardware=True):
    settings = _settings()
    authorization = _authorization(settings)
    torch.set_num_threads(settings["threads"])
    admission = validate_v4_pilot_policy(profile.DATASET)
    if not admission["ok"] or admission["content_id"] != profile.CONTENT_ID:
        raise ValueError(f"V5 frozen release failed admission: {admission['errors']}")
    profile.preview(profile.DATASET)
    train, val = (PackedPretrainDataset(profile.DATASET / f"{split}.bin", settings["seq_len"])
                  for split in ("train", "val"))
    tokenizer = MiniLLMTokenizer.load(profile.DATASET / "tokenizer.json")
    sampler = EpochBatchSampler(len(train), settings["batch_size"], settings["seed"])
    if (len(sampler), len(val)) != (9789, 216):
        raise ValueError("frozen V5 train or validation window count changed")
    parent, lineage = parent_checkpoint(train, val, sampler)
    if tokenizer.fingerprint != parent["tokenizer_fingerprint"]:
        raise ValueError("continuation tokenizer differs from the parent")
    corpus_plan = read_json(profile.DATASET / ".v4/plan.json")
    documents = read_token_index([profile.DATASET / "train.token-index.jsonl"])["records"]
    replay = RepeatPlan(sampler, parent["samples_seen"], settings["stage_steps"] * settings["grad_accum"])
    preview = accounting(replay, documents, corpus_plan, settings["seq_len"], replay.added_windows, enforce=True)
    if (preview["stage"]["effective_tokens"], preview["cumulative"]["effective_tokens"],
            replay.cursor(0), replay.cursor(16), replay.cursor(replay.added_windows)) != (
            PARENT_TOKENS, 2 * PARENT_TOKENS, (0, PARENT_WINDOWS), (1, 3), (1, 9763)):
        raise ValueError("planned budget or epoch crossover differs from the approved run")
    device = torch.device(settings["device"])
    if check_hardware and (not torch.cuda.is_available()
            or torch.cuda.mem_get_info(device)[0] < settings["gpu_memory_limit_mib"] * 1024**2
            or shutil.disk_usage(RUN.parent).free < 20 * 1024**3):
        raise ValueError("repeat training requires CUDA, available GPU memory and 20 GiB of free disk")
    if publish:
        write_json(REPORT / "preflight.json", {"status": "PASS", "admission": admission, "authorization": authorization,
                   "parent": lineage, "sampler_replay": preview, "environment": environment(device)})
        print(f"[preflight] 147 parent tensors match; {replay.added_windows:,} windows across {preview['stage']['epochs']}",
              flush=True)
    return PreparedStage(settings, train, val, tokenizer, parent, lineage, authorization, replay, preview,
                         documents, corpus_plan)


@contextmanager
def run_lock() -> Iterator[None]:
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def stage_lr(index, settings):
    # Last actually applied update is the cosine endpoint.
    return lr_at(index, settings["stage_steps"] - 1, settings["lr"], settings["min_lr"], settings["warmup_steps"])


def stage_manifest(parent, lineage, settings, authorization, preview):
    original = parent["manifest"]
    plan = dict(original["training"])
    plan.update(max_steps=PARENT_STEP + settings["stage_steps"],
                target_optimizer_updates=PARENT_STEP + settings["stage_steps"],
                milestone_steps=settings["milestone_steps"], learning_rate=settings["lr"],
                min_learning_rate=settings["min_lr"], warmup_steps=settings["warmup_steps"],
                warmup_ratio=settings["warmup_steps"] / settings["stage_steps"],
                scheduler_total_steps=settings["stage_steps"])
    return {**original, "training": plan, "run_id": uuid.uuid4().hex, "continuation": {
        "schema_version": 1, "parent": lineage, "settings": settings, "authorization": authorization,
        "stage_sampler": preview, "stage_lr_index_origin": 0, "stage_lr_final_index": settings["stage_steps"] - 1,
    }}


def verify_progress(state, manifest, replay, settings):
    step = state["global_step"] - settings["parent_step"]
    if state["manifest"] != manifest or not 0 <= step <= settings["stage_steps"]:
        raise CheckpointError("recovery checkpoint is outside this continuation plan")
    windows = step * settings["grad_accum"]
    epoch, cursor = replay.cursor(windows)
    stage = state["continuation"]
    if ((state["epoch"], state["micro_batches_consumed_in_epoch"], state["samples_seen"], state["tokens_seen"]) !=
            (epoch, cursor, replay.parent_windows + windows,
             settings["parent_effective_tokens"] + windows * (settings["seq_len"] - 1))
            or stage["stage_step"] != step or stage["next_lr_index"] != step
            or stage["stage_effective_tokens"] != windows * (settings["seq_len"] - 1)
            or (step and state["optimizer"]["param_groups"][0]["lr"] != stage_lr(step - 1, settings))):
        raise CheckpointError("recovery counters, epoch cursor or stage scheduler disagree")
    history = state["history"]
    steps = [row["step"] for row in history]
    scheduled = [settings["parent_step"]] + [
        settings["parent_step"] + index for index in range(1, step + 1)
        if index % settings["val_every"] == 0 or settings["parent_step"] + index in settings["milestone_steps"]
        or index == settings["stage_steps"]]
    if steps != scheduled or any(not math.isfinite(row["loss"]) for row in history):
        raise CheckpointError("continuation validation history differs from the scheduled checkpoints")
    best = state["best_checkpoint"]
    best_row = min(history, key=lambda row: row["loss"])
    if ((state["best_loss"], best["step"], best["val_loss"], best["tokens_seen"]) !=
            (best_row["loss"], best_row["step"], best_row["loss"],
             settings["parent_effective_tokens"] + (best_row["step"] - settings["parent_step"])
             * settings["grad_accum"] * (settings["seq_len"] - 1))
            or stage["overfit_streak"] != overfit_streak(history)
            or any(not math.isfinite(stage[key]) or stage[key] < 0 for key in ("seconds", "peak_gpu_mib"))):
        raise CheckpointError("continuation validation tracker or timing metadata differs")
    return step


def overfit_streak(history):
    best, streak = float("inf"), 0
    for row in history:
        loss = row["loss"]
        streak = streak + 1 if loss > 1.03 * best else 0
        best = min(best, loss)
    return streak


def verify_update_log(rows, settings, replay, stage_step):
    if [row["stage_step"] for row in rows] != list(range(1, stage_step + 1)):
        raise CheckpointError("continuation log does not cover all committed updates in order")
    order = replay.stage_order()
    width, origin = settings["grad_accum"], settings["parent_step"]
    tokens_per_update = width * (settings["seq_len"] - 1)
    evidence = {"logged_updates": 0, "replayed_from_cursor_updates": 0}
    for row in rows:
        step = row["stage_step"]
        end = step * width
        if ((row["global_step"], row["tokens_seen"], row["loss_tokens"]) !=
                (origin + step, settings["parent_effective_tokens"] + step * tokens_per_update, tokens_per_update)
                or (row["epoch"], row["micro_batches_consumed_in_epoch"]) != replay.cursor(end)
                or row["learning_rate"] != stage_lr(step - 1, settings)):
            raise CheckpointError(f"logged update {step} disagrees with the continuation schedule or sampler cursor")
        if "window_indices" not in row:
            # Early versions logged the deterministic cursor. A cursor-only
            # prefix remains verifiable against the frozen seed and epoch plan.
            if evidence["logged_updates"]:
                raise CheckpointError("per-update window evidence ends prematurely")
            evidence["replayed_from_cursor_updates"] += 1
        elif row["window_indices"] != order[end - width:end]:
            raise CheckpointError(f"logged update {step} windows differ from actual sampler replay")
        else:
            evidence["logged_updates"] += 1
    return evidence


def _best_path(out, record):
    return out / "validation-best" / f"step-{record['step']}-{record['file']['sha256']}.pt"


def _publish_best(out, record):
    source = _best_path(out, record)
    verify_file(source, record["file"])
    output = out / "best.pt"
    if not output.is_file() or file_info(output) != record["file"]:
        atomic_write(output, lambda handle: _copy_model(source, handle))
    write_best_status(out, record["val_loss"], record)


def _copy_model(source, handle):
    with source.open("rb") as model_file:
        shutil.copyfileobj(model_file, handle, length=1024 * 1024)


def _model_metadata(manifest, stage_step):
    return {"mode": "pretrain", "training_run_id": manifest["run_id"], "seq_len": manifest["training"]["seq_len"],
            "stage_step": stage_step, "parent_run_id": manifest["continuation"]["parent"]["run_id"]}


def _best_record(out, source, step, tokens, loss):
    record = {"step": step, "tokens_seen": tokens, "val_loss": loss, "file": file_info(source)}
    path = _best_path(out, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        atomic_write(path, lambda handle: _copy_model(source, handle))
    return record


def _initialize_manifest(out, expected):
    path = out / "training-manifest.json"
    entries = {item.name for item in out.iterdir()}
    if entries <= {".lock"}:
        write_json(path, expected)
        return expected
    # A failure before the first state save leaves only initialization files.
    unstarted = {".lock", path.name, "validation-best", "best.pt", "best-checkpoint-status.json",
                 "tokenizer.json", "config.json"}
    if path.name in entries and entries <= unstarted:
        manifest = read_json(path)
        if {**expected, "run_id": manifest["run_id"]} == manifest:
            return manifest
    raise CheckpointError("continuation output already exists; use its own recovery state")


def _already_completed(out, resume, stage_step, settings, state):
    summary_path, model_path = out / "train-summary.json", out / "model.pt"
    if (resume != "state.pt" or stage_step != settings["stage_steps"]
            or not summary_path.is_file() or not model_path.is_file()):
        return False
    summary = read_json(summary_path)
    if summary["stop_reason"] != "target_reached":
        return False
    if any(summary[key] != state[key] for key in ("global_step", "samples_seen", "tokens_seen", "best_checkpoint")):
        raise CheckpointError("published completion differs from the saved recovery state")
    model = _load_model(model_path)
    verify_model_metadata(model, state, context=str(model_path))
    require_identical_weights(state["model"], model["model"], str(model_path))
    print("[training] verified the already completed continuation", flush=True)
    return True


def execute(resume=None):
    if resume not in (None, "state.pt", "state.prev.pt"):
        raise CheckpointError("continuation accepts only its own primary or backup recovery state")
    prepared = prepare()
    settings, replay = prepared.settings, prepared.replay
    train, val, tokenizer, parent = prepared.train, prepared.val, prepared.tokenizer, prepared.parent
    expected = prepared.manifest()
    manifest_path = RUN / "training-manifest.json"
    if resume:
        manifest = read_json(manifest_path)
        if {**expected, "run_id": manifest["run_id"]} != manifest:
            raise CheckpointError("continuation lineage or plan differs from the saved run")
        checkpoint = RUN / resume
        state = load_training_state(checkpoint, mmap=True)
        stage_step = verify_progress(state, manifest, replay, settings)
        if state["history"][0] != parent["history"][-1]:
            raise CheckpointError("continuation validation history lost the pinned 10M baseline")
        if _already_completed(RUN, resume, stage_step, settings, state):
            return 0
        reconcile_recovery_log(RUN, state, checkpoint, initial_step=PARENT_STEP)
        if stage_step:
            rows = [row for row in records(RUN / "training-log.jsonl") if row["event"] == "train"]
            verify_update_log(rows, settings, replay, stage_step)
        for step in MILESTONES:
            if step < state["global_step"]:
                verify_milestone(RUN / "milestones" / f"step-{step}", manifest, step=step,
                                 tokens=step * TOKENS_PER_UPDATE)
        history = list(state["history"])
        best = state["best_checkpoint"]
        _publish_best(RUN, best)
        previous_seconds = state["continuation"]["seconds"]
        previous_peak = state["continuation"]["peak_gpu_mib"]
    else:
        manifest = _initialize_manifest(RUN, expected)
        state, stage_step = parent, 0
        history = [dict(parent["history"][-1])]
        best = _best_record(RUN, profile.RUN / "model.pt", PARENT_STEP, PARENT_TOKENS, history[0]["loss"])
        _publish_best(RUN, best)
        previous_seconds = previous_peak = 0.0
        tokenizer.save(RUN / "tokenizer.json")
        MiniLLMConfig.from_dict(state["config"]).save(RUN / "config.json")

    opts = TrainingOptions(lr=settings["lr"], grad_accum=settings["grad_accum"],
                           grad_clip=manifest["training"]["grad_clip"])
    device = torch.device(settings["device"])
    model = MiniLLM(MiniLLMConfig.from_dict(state["config"]))
    if model.num_parameters() != 86_528_640:
        raise ValueError("continuation architecture has the wrong parameter count")
    model.load_state_dict(state["model"], strict=True)
    model.to(device).train()
    model.gradient_checkpointing = settings["grad_checkpointing"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr, betas=tuple(manifest["training"]["betas"]),
                                  eps=manifest["training"]["eps"], weight_decay=manifest["training"]["weight_decay"])
    optimizer.load_state_dict(state["optimizer"])
    set_rng_state(state["rng"])
    del parent, state, prepared
    order = replay.stage_order()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    preserve_backup = resume == "state.prev.pt"
    saved_step = stage_step if resume == "state.pt" else None
    interrupted = None

    def stop(signum, _):
        nonlocal interrupted
        interrupted = signal.Signals(signum).name

    def peak():
        return max(previous_peak, torch.cuda.max_memory_allocated(device) / 1024**2)

    def seconds():
        return previous_seconds + time.monotonic() - started

    def save():
        nonlocal preserve_backup, saved_step
        epoch, cursor = replay.cursor(stage_step * opts.grad_accum)
        step = PARENT_STEP + stage_step
        save_training_state(
            RUN / "state.pt", model, optimizer, None, tokenizer, global_step=step, epoch=epoch,
            micro_batches_consumed_in_epoch=cursor, samples_seen=step * opts.grad_accum,
            best_loss=best["val_loss"], history=history, tokens=step * TOKENS_PER_UPDATE, manifest=manifest,
            preserve_backup=preserve_backup, best_checkpoint=best,
            continuation={"stage_step": stage_step, "next_lr_index": stage_step,
                          "stage_effective_tokens": stage_step * TOKENS_PER_UPDATE,
                          "overfit_streak": overfit_streak(history), "seconds": seconds(), "peak_gpu_mib": peak()},
        )
        preserve_backup, saved_step = False, stage_step

    def capture():
        step = PARENT_STEP + stage_step
        if step in MILESTONES:
            save_milestone(RUN, model, tokenizer, manifest, step=step, tokens=step * TOKENS_PER_UPDATE,
                           extra=_model_metadata(manifest, stage_step))

    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if saved_step is None:
            save()
        capture()
        print(f"[training] stage {stage_step}/{settings['stage_steps']} on {environment(device)['device_name']}", flush=True)
        while stage_step < settings["stage_steps"] and not interrupted:
            start = time.monotonic()
            offset = stage_step * opts.grad_accum
            windows = order[offset:offset + opts.grad_accum]
            group = [collate_batch([train[window]], pad_token_id=model.cfg.pad_token_id,
                                   ignore_index=model.cfg.ignore_index) for window in windows]
            rate = stage_lr(stage_step, settings)
            optimizer.param_groups[0]["lr"] = rate
            result = _train_step(model, group, optimizer, None, device, None, opts, rate)
            if result.tokens != TOKENS_PER_UPDATE or peak() > settings["gpu_memory_limit_mib"]:
                raise CheckpointError("update supervision or GPU allocation exceeded the admitted plan")
            stage_step += 1
            step = PARENT_STEP + stage_step
            elapsed = time.monotonic() - start
            epoch, cursor = replay.cursor(stage_step * opts.grad_accum)
            record = {"event": "train", "global_step": step, "stage_step": stage_step, "tokens_seen": step * TOKENS_PER_UPDATE,
                      "epoch": epoch, "micro_batches_consumed_in_epoch": cursor,
                      "window_indices": windows, "loss": result.loss_sum / result.tokens,
                      "loss_tokens": result.tokens, "learning_rate": rate, "gradient_norm": result.grad_norm,
                      "step_seconds": elapsed, "seconds": seconds(), "peak_gpu_mb": peak()}
            _append_log(RUN, record)
            print(f"[step] stage={stage_step} cumulative={step} loss={record['loss']:.4f} lr={rate:.8g} "
                  f"gradient={result.grad_norm:.4f} seconds={elapsed:.3f}", flush=True)
            if stage_step % settings["val_every"] == 0 or step in MILESTONES:
                metrics = evaluate(model, val, device, batch_size=settings["batch_size"],
                                   pad_token_id=model.cfg.pad_token_id, ignore_index=model.cfg.ignore_index)
                if not math.isfinite(metrics["loss"]) or metrics["tokens"] != len(val) * SUPERVISED_PER_WINDOW:
                    raise NumericalError("validation loss or supervision is invalid")
                history.append({"step": step, "mode": "pretrain", **metrics})
                streak = overfit_streak(history)
                _append_log(RUN, {"event": "validation", "global_step": step, "stage_step": stage_step, **metrics,
                                  "overfit_streak": streak, "warning": streak >= 3})
                print(f"[validation] step={step} loss={metrics['loss']:.6f} streak={streak}", flush=True)
                if streak == 3:
                    print("[warning] three consecutive validation losses exceed the previous best by 3%", flush=True)
                if metrics["loss"] < best["val_loss"]:
                    candidate = RUN / ".best-candidate.pt"
                    save_model_checkpoint(candidate, model, tokenizer, step=step, tokens=step * TOKENS_PER_UPDATE,
                                          val_loss=metrics["loss"], extra=_model_metadata(manifest, stage_step))
                    best = _best_record(RUN, candidate, step, step * TOKENS_PER_UPDATE, metrics["loss"])
                    candidate.unlink()
                    _publish_best(RUN, best)
                write_json(RUN / "metrics.json", history)
            if stage_step % settings["save_every"] == 0 or step in MILESTONES:
                save()
                capture()
        if saved_step != stage_step:
            save()
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)

    step = PARENT_STEP + stage_step
    if stage_step == settings["stage_steps"]:
        save_model_checkpoint(RUN / "model.pt", model, tokenizer, step=step, tokens=step * TOKENS_PER_UPDATE,
                              val_loss=history[-1]["loss"], extra=_model_metadata(manifest, stage_step))
    rows = [row for row in records(RUN / "training-log.jsonl") if row["event"] == "train"] if stage_step else []
    duration = seconds()
    active = sum(row["step_seconds"] for row in rows)
    completed = stage_step == settings["stage_steps"]
    summary = {"stop_reason": "target_reached" if completed else interrupted, "stage_steps": stage_step,
               "global_step": step, "stage_effective_tokens": stage_step * TOKENS_PER_UPDATE,
               "tokens_seen": step * TOKENS_PER_UPDATE, "samples_seen": step * opts.grad_accum,
               "epoch": replay.cursor(stage_step * opts.grad_accum)[0],
               "micro_batches_consumed_in_epoch": replay.cursor(stage_step * opts.grad_accum)[1],
               "seconds": duration, "optimizer_update_seconds": active,
               "tokens_per_second": stage_step * TOKENS_PER_UPDATE / duration,
               "update_tokens_per_second": stage_step * TOKENS_PER_UPDATE / active if active else None,
               "peak_gpu_mib": peak(), "peak_gpu_reserved_mib_this_process": torch.cuda.max_memory_reserved(device) / 1024**2,
               "final_val": history[-1] if history[-1]["step"] == step else None, "best_checkpoint": best,
               "overfit_warnings": [row["step"] for index, row in enumerate(history, 1)
                                    if overfit_streak(history[:index]) == 3],
               "environment": environment(device)}
    write_json(RUN / "train-summary.json", summary)
    print(f"[done] {summary}", flush=True)
    return 0 if completed else 2
