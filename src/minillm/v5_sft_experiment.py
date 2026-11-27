"""PR #22 training and bounded memorization experiment."""
from __future__ import annotations

from collections import Counter
import fcntl
import os
from pathlib import Path
import random
import subprocess
import time

import torch

from .assessment_io import environment, read_json, records, report_directory
from .config import ARCHITECTURE, ARCH_VERSION
from .corpus_utils import file_info
from .sft import sft_batch_factory
from .text_policy import checkpoint_text_policy
from .tokenizer import END_OF_TURN_ID, MiniLLMTokenizer
from .train import DIAGNOSTIC_ONLY, TrainingOptions, _train_step, evaluate, fit, load_model_checkpoint
from .training_manifest import implementation_identity, write_json
from .v5_sft_data import CONFIG, QUOTAS, ROOT, StrictChatDataset, frozen_manifest, write_rows
from .v4_language_scaling import FINGERPRINT

DEVICE = torch.device("cuda:0")


def base_assets(settings):
    base = ROOT / settings["base"]
    paths = {"checkpoint": base / "model.pt", "tokenizer": base / "tokenizer.json", "config": base / "config.json"}
    evidence = {name: file_info(path) for name, path in paths.items()}
    for name, info in evidence.items():
        if info["sha256"] != settings[f"base_{name}_sha256"]:
            raise ValueError(f"PR #21 {name} SHA256 mismatch")
    path = paths["checkpoint"]
    tokenizer = MiniLLMTokenizer.load(paths["tokenizer"])
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if (tokenizer.fingerprint != FINGERPRINT or tokenizer.vocab_size != 24576
            or payload.get("tokenizer_fingerprint") != tokenizer.fingerprint
            or payload.get("tokenizer_vocab_size") != tokenizer.vocab_size
            or payload["config"] != read_json(paths["config"])
            or (payload["architecture"], payload["arch_version"], payload["mode"], payload["step"], payload["tokens_seen"])
               != (ARCHITECTURE, ARCH_VERSION, "pretrain", 2444, 20001696)
            or len(payload["model"]) != 147 or any(t.dtype != torch.float32 for t in payload["model"].values())):
        raise ValueError("unexpected 20M base architecture, tokenizer, tensor set or training metadata")
    tokenizer.text_policy = checkpoint_text_policy(payload)
    return tokenizer, path, {**evidence, "fingerprint": tokenizer.fingerprint, "text_policy": tokenizer.text_policy,
                             "fp32_tensors": len(payload["model"]), "config_values": payload["config"]}


def load_base(checkpoint):
    model, _, _ = load_model_checkpoint(checkpoint, torch.device("cpu"))
    if model.num_parameters() != 86528640:
        raise ValueError("unexpected base parameter count")
    return model


def gpu_snapshot():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the PR #22 experiment")
    free, total = torch.cuda.mem_get_info(DEVICE)
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True).splitlines()
    return {"free_bytes": free, "total_bytes": total, "nvidia_smi": smi}


def gpu_probe(settings, checkpoint, evidence, report):
    with report_directory(report / "gpu-preflight") as directory:
        before = gpu_snapshot()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        model = load_base(checkpoint).to(DEVICE).train()
        model.gradient_checkpointing = settings["grad_checkpointing"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], betas=(0.9, 0.95),
                                      weight_decay=settings["weight_decay"])
        # Workload-only tokens. The probe writes no model or corpus.
        length = settings["max_seq_len"]
        ids = torch.arange(1024, 1024 + length).unsqueeze(0)
        labels = ids.clone()
        labels[:, :length // 2] = model.cfg.ignore_index
        batch = {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}
        opts = TrainingOptions(lr=settings["lr"], grad_clip=settings["grad_clip"])
        start = time.monotonic()
        result = _train_step(model, [batch] * settings["grad_accum"], optimizer, None, DEVICE, None, opts, settings["lr"])
        torch.cuda.synchronize(DEVICE)
        write_json(directory / "probe.json", {"purpose": "synthetic-token GPU capacity probe", "base": evidence,
                   "environment": environment(DEVICE), "before": before, "after": gpu_snapshot(),
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "input_tokens": ids.numel() * settings["grad_accum"], "workload_loss_tokens": result.tokens,
                   "elapsed_seconds": time.monotonic() - start,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(DEVICE),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(DEVICE)})


def mask_report(dataset, rows, tokenizer, seed):
    picks = random.Random(seed).sample(range(len(rows)), 20)
    return [{"id": rows[i]["id"], "messages": rows[i]["messages"],
             "input_ids": dataset[i]["input_ids"].tolist(), "labels": dataset[i]["labels"].tolist(),
             "decoded": tokenizer.decode(dataset[i]["input_ids"].tolist(), skip_special_tokens=False),
             "supervised": tokenizer.decode([label for label in dataset[i]["labels"].tolist()
                                             if label != tokenizer.ignore_index], skip_special_tokens=False),
             "end_of_turn_positions": [j for j, label in enumerate(dataset[i]["labels"].tolist())
                                       if label == END_OF_TURN_ID]} for i in picks]


def _write_or_verify(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"{path}: existing artifact differs")
    else:
        write_json(path, value)


def train_experiment(settings, mode, resume, *, base):
    tokenizer, checkpoint, evidence = base
    out, report, dataset = (ROOT / settings[key] for key in ("out", "report", "dataset"))
    if (report / "test-unlock.json").exists() or (report / "unseen/protocol-lock.json").exists():
        raise ValueError("the final evaluation protocol and training artifacts are locked")
    frozen = frozen_manifest(dataset, tokenizer, settings["max_seq_len"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if not (report / "gpu-preflight/probe.json").exists():
        gpu_probe(settings, checkpoint, evidence, report)
        torch.cuda.empty_cache()
    before = gpu_snapshot()
    path = dataset / "train.jsonl"
    rows = list(records(path))
    train = StrictChatDataset(path, tokenizer, settings["max_seq_len"], rows=rows)
    if Counter(row["topic_category"] for row in rows) != QUOTAS["train"]:
        raise ValueError("wrong train quotas")
    report.mkdir(parents=True, exist_ok=True)
    _write_or_verify(report / "mask-audit.json", mask_report(train, rows, tokenizer, settings["seed"]))
    execution = implementation_identity(CONFIG, ROOT / "scripts/run_v5_language_sft_5k.py",
                                        ROOT / "scripts/generate_v5_sft_data.py")
    provenance = {"implementation": execution, "frozen_dataset": file_info(dataset / "manifest.json"),
                  "base": evidence}
    kwargs = {key: settings[key] for key in ("epochs", "batch_size", "grad_accum", "precision", "device", "threads",
                                             "lr", "min_lr", "warmup_ratio", "weight_decay", "grad_clip",
                                             "seed", "grad_checkpointing", "save_every")}
    metadata = {"scope": DIAGNOSTIC_ONLY, "dataset_content_id": frozen["content_id"]}
    diagnostic = mode == "diagnose"
    if diagnostic:
        spec = settings["diagnostic"]
        simple = [row for row in rows if len(row["messages"]) == 2 and len(row["messages"][1]["content"]) <= 35]
        picked = random.Random(settings["seed"]).sample(simple, spec["examples"])
        _write_or_verify(report / "diagnostic-selection.json", [row["id"] for row in picked])
        train = StrictChatDataset(path, tokenizer, settings["max_seq_len"], rows=picked)
        val = train
        out = out / "diagnostic"
        kwargs.update(lr=spec["lr"], min_lr=spec["min_lr"], max_steps=spec["updates"],
                      val_every=20, log_every=1, milestone_steps=tuple(range(0, spec["updates"] + 1, 20)))
        metadata["experiment"] = "overfit32"
    else:
        from .v5_sft_evaluation import _receipt
        diagnostic_report = report / "diagnostic-result"
        diagnosis = read_json(diagnostic_report / "manifest.json")
        _receipt(diagnostic_report, diagnosis["binding"])
        if diagnosis["dataset_content_id"] != frozen["content_id"] or diagnosis["updates"] != settings["diagnostic"]["updates"]:
            raise ValueError("the 32-example diagnostic must be completed for this frozen corpus")
        provenance["diagnostic"] = file_info(report / "diagnostic-result/manifest.json")
        path_val = dataset / "val.jsonl"
        val_rows = list(records(path_val))
        if Counter(row["topic_category"] for row in val_rows) != QUOTAS["val"]:
            raise ValueError("wrong validation quotas")
        val = StrictChatDataset(path_val, tokenizer, settings["max_seq_len"], rows=val_rows)
        out = out / "formal"
        updates = (len(train) + kwargs["grad_accum"] - 1) // kwargs["grad_accum"]
        kwargs.update(val_every=updates, milestone_steps=tuple(updates * i for i in range(settings["epochs"] + 1)),
                      log_every=1)
        metadata["experiment"] = "sft5k"
    report_name = "diagnostic" if diagnostic else "formal"
    provenance["training_protocol"] = {"validation_interval": kwargs["val_every"], "validation_limit": 0,
                                      "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                      "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                                      "environment": environment(DEVICE)}
    _write_or_verify(report / f"{report_name}-token-plan.json",
                     {"unique_examples": len(train), "unique_input_tokens": sum(len(x["input_ids"]) for x in train.items),
                      "unique_assistant_tokens": train.stats["supervised_tokens"]})
    write_json(report / f"{report_name}-gpu-latest.json", before)
    state = str(out / resume) if resume else None
    opts = TrainingOptions(out=str(out), resume=state, provenance=provenance, extra_meta=metadata,
                           validate_before_milestone=True, cosine_final_update=True,
                           init_from=None if resume else str(checkpoint), **kwargs)
    model = load_base(checkpoint)
    factory = sft_batch_factory(train, opts.batch_size, opts.seed, model.cfg.pad_token_id, model.cfg.ignore_index)
    if diagnostic and not resume:
        baseline = evaluate(model.to(DEVICE), train, DEVICE, batch_size=1,
                            pad_token_id=model.cfg.pad_token_id, ignore_index=model.cfg.ignore_index)
        _write_or_verify(report / "diagnostic-base-loss.json", baseline)
    result = fit(opts, model, tokenizer, train, val, mode="sft", batch_factory=factory)
    if result["stop_reason"] == "target_reached":
        if diagnostic:
            from .v5_sft_evaluation import diagnostic_result
            diagnostic_result(model, tokenizer, picked, report, out, frozen["content_id"], result)
        else:
            steps = [r for r in records(out / "training-log.jsonl") if r["event"] == "train"]
            if [r["global_step"] for r in steps] != list(range(1, result["steps"] + 1)):
                raise ValueError("missing or duplicated optimizer steps in the recovery log")
            active_seconds = sum(r["step_seconds"] for r in steps)
            input_tokens = settings["epochs"] * sum(len(x["input_ids"]) for x in train.items)
            write_json(report / "training-complete.json", {**result,
                       "cumulative_input_tokens": input_tokens,
                       "cumulative_assistant_tokens": result["tokens_seen"],
                       "optimizer_step_seconds": active_seconds,
                       "input_tokens_per_optimizer_second": input_tokens / active_seconds,
                       "assistant_tokens_per_optimizer_second": result["tokens_seen"] / active_seconds,
                       "max_logged_gpu_mb": max(r["peak_gpu_mb"] for r in steps),
                       "training_log": file_info(out / "training-log.jsonl")})


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify-base", "gpu-probe", "baseline-knowledge", "diagnose", "train", "evaluate"))
    parser.add_argument("--resume", choices=("state.pt", "state.prev.pt"))
    parser.add_argument("--generalization", type=Path)
    args = parser.parse_args()
    if args.resume and args.command not in ("diagnose", "train"):
        parser.error("--resume applies only to diagnose/train")
    settings = read_json(CONFIG)
    torch.set_num_threads(settings["threads"])
    base = base_assets(settings)
    report = ROOT / settings["report"]
    report.mkdir(parents=True, exist_ok=True)
    if args.command in ("diagnose", "train", "evaluate") and not (ROOT / settings["dataset"] / "manifest.json").is_file():
        raise SystemExit("PENDING: the independently reviewed SFT dataset has not been frozen")
    if args.command == "verify-base":
        model = load_base(base[1])
        write_json(report / "base-verification.json", {**base[2], "parameters": model.num_parameters()})
    else:
        with (report / ".experiment.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.command == "gpu-probe":
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                torch.use_deterministic_algorithms(True)
                gpu_probe(settings, base[1], base[2], report)
            elif args.command == "evaluate":
                from .v5_sft_evaluation import evaluate_experiment
                evaluate_experiment(settings, base, report, args.generalization)
            elif args.command == "baseline-knowledge":
                from .v5_sft_evaluation import baseline_knowledge
                baseline_knowledge(base, report / "baseline-knowledge", DEVICE)
            else:
                train_experiment(settings, args.command, args.resume, base=base)
