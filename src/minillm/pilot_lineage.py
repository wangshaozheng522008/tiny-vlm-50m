"""Durable pilot identity and verified launch history; never repair by guessing."""
from __future__ import annotations

import contextlib
import copy
import re
from dataclasses import asdict
from pathlib import Path

import torch

from .assessment_io import read_json
from .corpus_utils import file_info, identity
from .training_manifest import compare_manifests, write_json


def sealed(value):
    value = {k: v for k, v in value.items() if k != "content_id"}
    return {**value, "content_id": identity(value)}


def unseal(value):
    if not isinstance(value, dict) or value.get("content_id") != sealed(value)["content_id"]:
        raise ValueError("damaged pilot lineage/identity checksum")
    return value


def _initialization(value, manifest):
    if not isinstance(value, dict):
        raise ValueError("missing initialization identity")
    if value.get("kind") == "weights_only":
        if set(value) != {"kind", "sha256"} or not isinstance(value["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
            raise ValueError("invalid initialization SHA256")
    elif value.get("kind") == "fresh":
        if set(value) != {"kind", "seed"} or type(value["seed"]) is not int or value["seed"] != manifest["training"]["seed"]:
            raise ValueError("invalid fresh initialization seed")
    else:
        raise ValueError("unknown initialization kind")


def _launch(path, manifest, initialization, out, *, initial):
    # Imports are lazy because workload.resolve_plan also reads lineage.
    from .workload import parse_training_config
    from .train import resolve_precision
    path = Path(path).resolve()
    record = read_json(path / "launch-manifest.json")
    if (type(record.get("schema_version")) is not int or record["schema_version"] not in (1, 2) or record.get("execute_requested") is not True
            or record.get("status") != "READY" or record.get("initialization") != initialization
            or record.get("request_id") != identity({k: v for k, v in record.items() if k != "request_id"})
            or record.get("training_output") != str(out.resolve())):
        raise ValueError("launch record identity/source/output mismatch")
    opts, extra = parse_training_config(record["configuration"])
    if asdict(opts) != record["resolved_options"] or Path(opts.out).resolve() != out.resolve():
        raise ValueError("launch record configuration mismatch")
    if initial and opts.resume or not initial and not opts.resume:
        raise ValueError("launch record does not identify the first startup/resume")
    if not initial and Path(opts.resume).resolve().parent != out.resolve():
        raise ValueError("historical resume points outside the owned training directory")
    plan = manifest["training"]
    mapping = {"seed": opts.seed, "sampler_seed": opts.seed, "batch_size": opts.batch_size,
               "grad_accum": opts.grad_accum, "seq_len": extra["seq_len"], "max_steps": opts.max_steps,
               "epochs": opts.epochs if opts.max_steps == 0 else None,
               "learning_rate": opts.lr, "min_learning_rate": opts.min_lr,
               "warmup_ratio": opts.warmup_ratio, "weight_decay": opts.weight_decay, "grad_clip": opts.grad_clip}
    device = torch.device(opts.device or manifest["execution"]["device"])
    if (any(plan[k] != v for k, v in mapping.items()) or record["model_config"] != manifest["model_config"]
            or record["dataset"] != manifest["dataset"]
            or manifest["execution"] != {"precision": resolve_precision(opts.precision, device),
                                         "device": str(device), "threads": opts.threads,
                                         "grad_checkpointing": opts.grad_checkpointing}):
        raise ValueError("launch record training/data/tokenizer manifest mismatch")
    if record["schema_version"] == 2:
        expected = {k: v for k, v in manifest.items() if k != "run_id"} if initial else manifest
        if (record.get("training_manifest") != expected
                or record.get("tokenizer_fingerprint") != manifest["tokenizer_fingerprint"]
                or record.get("report_path") != str(path)):
            raise ValueError("launch record manifest/run_id mismatch")
    if initial:
        _initialization(initialization, manifest)
        if initialization["kind"] == "weights_only":
            source = Path(record.get("source_path") or opts.init_from or "")
            if not opts.init_from or file_info(source)["sha256"] != initialization["sha256"]:
                raise ValueError("initialization source SHA256 mismatch/unavailable")
            payload = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
            if payload.get("config") != manifest["model_config"] or payload.get("tokenizer_fingerprint") != manifest["tokenizer_fingerprint"]:
                raise ValueError("initialization source config/tokenizer mismatch")
        elif opts.init_from or opts.seed != initialization["seed"]:
            raise ValueError("fresh initialization source mismatch")
    return record


def _base(manifest, initialization, report, request_id, out):
    return {"schema_version": 2, "initialization": initialization, "training_manifest": manifest,
            "training_output": str(out.resolve()), "tokenizer_fingerprint": manifest["tokenizer_fingerprint"],
            "dataset": manifest["dataset"],
            "initial_launch": {"request_id": request_id, "report_path": str(Path(report).resolve())},
            "resume_history": []}


def persist_initial(out, manifest, initialization, report, request_id):
    """Called after fit checks an empty output, before model/optimizer setup or updates."""
    out = Path(out)
    if not manifest.get("run_id") or out.exists() and any(out.iterdir()):
        raise ValueError("refusing to assign pilot identity to an existing training directory")
    record = _launch(report, manifest, initialization, out, initial=True)
    if record["request_id"] != request_id:
        raise ValueError("initial launch request mismatch")
    lineage = sealed(_base(manifest, initialization, report, request_id, out))
    # External binding survives loss of the in-run lineage; both precede any checkpoint.
    write_json(Path(report) / "run-identity.json", lineage)
    write_json(out / "pilot-lineage.json", lineage)


def _validate_v2(lineage, manifest, out):
    unseal(lineage)
    allowed = {"schema_version", "initialization", "training_manifest", "training_output",
               "tokenizer_fingerprint", "dataset", "initial_launch", "resume_history",
               "content_id", "migration", "recovery"}
    if (type(lineage.get("schema_version")) is not int or lineage["schema_version"] != 2 or lineage.get("training_manifest") != manifest
            or set(lineage) - allowed or read_json(out / "training-manifest.json") != manifest
            or lineage.get("training_output") != str(out.resolve())
            or lineage.get("tokenizer_fingerprint") != manifest["tokenizer_fingerprint"]
            or lineage.get("dataset") != manifest["dataset"] or not isinstance(lineage.get("resume_history"), list)):
        raise ValueError("lineage manifest/run_id/data/tokenizer/output mismatch")
    initial = lineage["initial_launch"]
    if not isinstance(initial, dict) or set(initial) != {"request_id", "report_path"}:
        raise ValueError("invalid first launch identity")
    if "migration" in lineage:
        migrated = lineage["migration"]
        if not isinstance(migrated, dict) or type(migrated.get("from_schema_version")) is not int or migrated["from_schema_version"] != 1:
            raise ValueError("inconsistent legacy migration history")
        if migrated.get("legacy_launch_report") == initial["report_path"]:
            if set(migrated) != {"from_schema_version", "legacy_launch_report"}:
                raise ValueError("inconsistent legacy migration history")
        else:
            if (set(migrated) != {"from_schema_version", "legacy_launch_report", "legacy_request_id", "history_scope"}
                    or migrated["history_scope"] != "legacy resume reference retained; prior checkpoint steps unavailable"):
                raise ValueError("inconsistent legacy resume history")
            legacy = _launch(migrated["legacy_launch_report"], manifest, lineage["initialization"], out, initial=False)
            if legacy["request_id"] != migrated["legacy_request_id"]:
                raise ValueError("legacy resume request identity mismatch")
    if "recovery" in lineage:
        recovered = lineage["recovery"]
        if (not isinstance(recovered, dict) or set(recovered) != {"reason", "checkpoint_sha256", "history_scope"}
                or recovered["reason"] != "missing_lineage"
                or not re.fullmatch(r"[0-9a-f]{64}", recovered["checkpoint_sha256"])
                or recovered["history_scope"] != "only verified records retained; missing resume history is not invented"):
            raise ValueError("inconsistent lineage recovery history")
    record = _launch(initial["report_path"], manifest, lineage["initialization"], out, initial=True)
    if record["request_id"] != initial["request_id"]:
        raise ValueError("initial launch request mismatch")
    if record["schema_version"] == 2:
        binding = unseal(read_json(Path(initial["report_path"]) / "run-identity.json"))
        if binding != sealed(_base(manifest, lineage["initialization"], initial["report_path"], initial["request_id"], out)):
            raise ValueError("initial run_id binding mismatch")
    seen = set()
    for entry in lineage["resume_history"]:
        if not isinstance(entry, dict) or set(entry) != {"request_id", "report_path", "checkpoint_step"}:
            raise ValueError("invalid resume history record")
        step = entry["checkpoint_step"]
        if type(step) is not int or not 0 <= step <= manifest["training"]["target_optimizer_updates"] or entry["request_id"] in seen:
            raise ValueError("invalid/duplicate resume history request")
        seen.add(entry["request_id"])
        resumed = _launch(entry["report_path"], manifest, lineage["initialization"], out, initial=False)
        checkpoint = resumed.get("checkpoint", {})
        if (resumed["schema_version"] != 2 or resumed["request_id"] != entry["request_id"]
                or checkpoint.get("global_step") != step or checkpoint.get("run_id") != manifest["run_id"]
                or not re.fullmatch(r"[0-9a-f]{64}", checkpoint.get("sha256", ""))):
            raise ValueError("resume history checkpoint/run_id mismatch")
    return lineage


def load_for_resume(opts, state, current_manifest, train, sampler, *, report=None):
    """Read/validate only. Recovery is published only after execute passes readiness."""
    from .train import _validate_progress
    out = Path(opts.out).resolve()
    manifest = state["manifest"]
    if (Path(opts.resume).resolve().parent != out or not isinstance(manifest.get("run_id"), str)
            or not manifest["run_id"] or read_json(out / "training-manifest.json") != manifest):
        raise ValueError("resume output/manifest/run_id ownership mismatch")
    compare_manifests({k: v for k, v in manifest.items() if k != "run_id"}, current_manifest)
    _validate_progress(state, manifest, train, sampler)
    if any(not torch.isfinite(v).all() for v in state["model"].values()):
        raise ValueError("nonfinite checkpoint model")
    if any(not torch.isfinite(slot[k]).all() for slot in state["optimizer"]["state"].values() for k in ("exp_avg", "exp_avg_sq")):
        raise ValueError("nonfinite checkpoint optimizer")
    path = out / "pilot-lineage.json"
    if path.exists():
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError("invalid pilot lineage")
        if value.get("schema_version") == 2:
            if report and Path(report).resolve() != Path(value["initial_launch"]["report_path"]).resolve():
                raise ValueError("explicit initialization report disagrees with lineage")
            return _validate_v2(value, manifest, out)
        if (type(value.get("schema_version", 1)) is not int or value.get("schema_version", 1) != 1 or value.get("training_manifest") != manifest
                or set(value) - {"schema_version", "initialization", "training_manifest", "launch_report"}):
            raise ValueError("invalid legacy pilot lineage/manifest")
        legacy_path = Path(value["launch_report"]).resolve()
        declared = read_json(legacy_path / "launch-manifest.json")
        if declared.get("configuration", {}).get("resume"):
            if not report:
                raise ValueError("legacy lineage lost its first launch reference; supply the original lineage_report")
            last = _launch(legacy_path, manifest, value["initialization"], out, initial=False)
            initial_path = Path(report).resolve()
            migration = {"from_schema_version": 1, "legacy_launch_report": str(legacy_path),
                         "legacy_request_id": last["request_id"],
                         "history_scope": "legacy resume reference retained; prior checkpoint steps unavailable"}
        else:
            if report and Path(report).resolve() != legacy_path:
                raise ValueError("explicit initialization report disagrees with legacy lineage")
            initial_path = legacy_path
            migration = {"from_schema_version": 1, "legacy_launch_report": str(legacy_path)}
        original = _launch(initial_path, manifest, value["initialization"], out, initial=True)
        result = _base(manifest, value["initialization"], initial_path, original["request_id"], out)
        result["migration"] = migration
        return _validate_v2(sealed(result), manifest, out)
    if not report:
        raise ValueError("missing pilot lineage: supply lineage_report pointing to the original durable launch evidence; initialization cannot be guessed")
    record = read_json(Path(report) / "launch-manifest.json")
    init = record.get("initialization")
    record = _launch(report, manifest, init, out, initial=True)
    binding = Path(report) / "run-identity.json"
    if binding.exists():
        base = _validate_v2(read_json(binding), manifest, out)
    else:
        # v1 did not persist a run_id externally. Require a same-step saved model
        # whose weights and launch metadata independently agree with this state.
        if record["schema_version"] != 1:
            raise ValueError("missing durable run_id binding")
        matched = False
        for name in ("model.pt", "best.pt"):
            candidate = out / name
            if not candidate.exists():
                continue
            model = torch.load(candidate, map_location="cpu", weights_only=False, mmap=True)
            if (model.get("step") == state["global_step"] and model.get("tokens_seen") == state["tokens_seen"]
                    and model.get("pilot_request_id") == record["request_id"] and model.get("pilot_initialization") == init
                    and model.get("config") == state["config"] and model.get("tokenizer_fingerprint") == state["tokenizer_fingerprint"]
                    and set(model.get("model", {})) == set(state["model"])
                    and all(torch.equal(v, state["model"][k]) for k, v in model["model"].items())):
                matched = True
                break
        if not matched:
            raise ValueError("missing lineage: no matching checkpoint model/launch evidence for initialization")
        base = _base(manifest, init, report, record["request_id"], out)
    result = {**base, "recovery": {"reason": "missing_lineage", "checkpoint_sha256": file_info(Path(opts.resume))["sha256"],
                                  "history_scope": "only verified records retained; missing resume history is not invented"}}
    return _validate_v2(sealed(result), manifest, out)


def record_resume(out, lineage, report, request_id, checkpoint_step):
    """Idempotent, atomic append. Never replace the initial launch identity."""
    out = Path(out)
    lineage = copy.deepcopy(lineage)
    _validate_v2(lineage, lineage["training_manifest"], out)
    entry = {"request_id": request_id, "report_path": str(Path(report).resolve()), "checkpoint_step": checkpoint_step}
    existing = [e for e in lineage["resume_history"] if e["request_id"] == request_id]
    if existing:
        if existing != [entry]:
            raise ValueError("resume request reused with different evidence")
        return
    lineage["resume_history"].append(entry)
    lineage = sealed(lineage)
    _validate_v2(lineage, lineage["training_manifest"], out)
    write_json(out / "pilot-lineage.json", lineage)


@contextlib.contextmanager
def run_lock(out):
    """POSIX advisory lock is released even on SIGKILL; no stale PID guessing."""
    import fcntl
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / f".{out.name}.pilot.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another pilot launcher owns this training directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
