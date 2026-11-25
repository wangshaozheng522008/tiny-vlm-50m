"""V4 pilot orchestration; all optimization stays in the existing pilot launcher."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import sys
import time

import torch

from .assessment_io import environment, read_json, records, report_directory, verified_evaluation
from .config import MINILLM_87M, MINILLM_NANO, MiniLLMConfig, get_config
from .corpus_utils import file_info, identity
from .data import PackedPretrainDataset
from .evaluation import EvaluationConfig, evaluate_model, task_asset_provenance
from .model import MiniLLM
from .pilot_launcher import launch_pilot
from .pilot_lineage import sealed, unseal
from .train import save_model_checkpoint
from .training_manifest import write_json
from .training_pilot import evaluate_pilot
from .training_readiness import planned_tokens
from .utils import set_seed
from .v4_admission import audit_v4_dataset, validate_v4_pilot_policy
from .v4_language_pilot import BENCHMARK, is_language
from .workload import (hardware, parse_training_config, preserved_process, probe_workload,
                       resolve_plan, verify_workload)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITES = [ROOT / "data/evaluation/pretrain-v1", ROOT / "data/evaluation/chinese-foundation-v1"]
LANGUAGE_SUITE = ROOT / BENCHMARK
LANGUAGE_CONTINUATIONS = ROOT / "data/evaluation/chinese-language-continuations-v1/prompts.jsonl"
PILOT_TARGET_TOKENS = 1_000_000


def _evaluation_suites(manifest):
    return [LANGUAGE_SUITE] if manifest and is_language(manifest) else DEFAULT_SUITES


def _suite_assets(suite):
    suite = Path(suite)
    if suite == LANGUAGE_SUITE:
        return {"benchmark": suite, "prompts": LANGUAGE_CONTINUATIONS}
    return {"prompts": suite / "prompts.jsonl", "candidate_benchmark": suite / "candidates.jsonl"}


def validate_pilot_budget(samples, opts, extra, *, cpu_nano_test=False):
    """One cumulative fixed plan, including Resume; rounding is < one update."""
    target = extra.get("target_tokens")
    if type(target) is not int or (target < 1 if cpu_nano_test else target != PILOT_TARGET_TOKENS):
        raise ValueError("V4 Pilot requires exactly 1,000,000 integer target tokens (CPU fixtures use their independent small budget)")
    if opts.max_steps < 3 or extra.get("allow_data_repetition", False) is not False:
        raise ValueError("at least three fixed updates and no corpus repetition required")
    seq_len = extra["seq_len"]
    planned = planned_tokens(samples, opts.batch_size, opts.grad_accum, seq_len, opts.max_steps)
    independent = samples * (seq_len - 1)
    update = opts.batch_size * opts.grad_accum * (seq_len - 1)
    if not target <= planned < target + update or planned > independent:
        raise ValueError("fixed plan misses target/rounding or repeats independent windows")
    declared = extra.get("planned_effective_tokens", planned)
    if type(declared) is not int or declared != planned:
        raise ValueError("declared planned token budget differs from actual fixed windows")
    if not cpu_nano_test and independent < 1_020_000:
        raise ValueError("at least 1.02M independent effective tokens required")
    return {"planned_effective_tokens": planned, "independent_effective_tokens": independent,
            "target_tokens": target, "optimizer_updates": opts.max_steps,
            "full_update_effective_tokens": update, "train_windows": samples,
            "allow_data_repetition": False}


def evaluation_policy(config, suites):
    config.validate()
    if config.stage != "pretrain":
        raise ValueError("V4 pretraining requires the pretrain evaluation protocol")
    assets = {}
    for suite in suites:
        for path in _suite_assets(suite).values():
            count = sum(1 for _ in records(path))
            provenance = task_asset_provenance(path, count)
            if provenance["status"] != "FROZEN":
                raise ValueError("pilot evaluation assets must be frozen")
            assets[str(path.resolve())] = provenance
    if not assets:
        raise ValueError("frozen evaluation suites required")
    return {"config": asdict(config), "assets": assets}


def _request(dataset, tokenizer, cfg, raw, evaluation, cpu_nano_test):
    manifest = read_json(Path(dataset) / "dataset-manifest.json")
    return {"dataset_content_id": manifest["content_id"], "text_policy": manifest["text_policy"],
            "tokenizer": file_info(Path(tokenizer)), "model_config": cfg.to_dict(),
            "training": {k: v for k, v in raw.items()
                         if k not in ("resume", "workload_report", "lineage_report")},
            "evaluation": evaluation, "scope": "CPU_NANO_TEST" if cpu_nano_test else "87M_GPU_PILOT"}


def _workspace_plan(workspace, request):
    plan = unseal(read_json(Path(workspace) / "pilot-plan.json"))
    if plan["request"] != request or plan["initial_checkpoint"] != file_info(Path(workspace) / "initial.pt"):
        raise ValueError("pilot workspace data/tokenizer/plan/initial weights changed")
    return plan


def _resolved(raw, workspace):
    result = dict(raw)
    if not result.get("resume"):
        result["init_from"] = str((Path(workspace) / "initial.pt").resolve())
    result["workload_report"] = str(Path(raw.get("workload_report") or Path(workspace) / "workload").resolve())
    return result


def preflight(dataset, tokenizer, model_config, training_config, workspace, *,
              cpu_nano_test=False, evaluation_config=None, suites=None, require_workload=True,
              approved_config_path=None):
    """Read-only admission. Never construct an 87M model on failed data gates."""
    raw = read_json(training_config) if isinstance(training_config, (str, Path)) else dict(training_config)
    opts, extra = parse_training_config(raw)
    cfg = MiniLLMConfig.load(model_config) if Path(model_config).is_file() else get_config(model_config)
    evaluation_config = evaluation_config or EvaluationConfig(stage="pretrain")
    blockers, details = [], {}

    def block(code, reason):
        blockers.append(code)
        details.setdefault(code, []).append(str(reason))

    audit = audit_v4_dataset(dataset, tokenizer_path=tokenizer, require_source_review=not cpu_nano_test)
    manifest = None
    if not audit["ok"]:
        block("BLOCKED_DATA_NOT_READY", "; ".join(audit["errors"]))
    else:
        manifest = read_json(Path(dataset) / "dataset-manifest.json")
    if approved_config_path is not None and not cpu_nano_test:
        pilot_policy = {"ok": False, "status": "BLOCKED_POLICY_MISMATCH",
                        "errors": ["production Pilot policy is pinned to the repository-approved configuration"]}
    else:
        pilot_policy = validate_v4_pilot_policy(dataset, approved_config_path)
    if not pilot_policy["ok"]:
        block(pilot_policy["status"], "; ".join(pilot_policy["errors"]))
    try:
        from .tokenizer import MiniLLMTokenizer
        tok = MiniLLMTokenizer.load(tokenizer)
        tok.assert_vocab_size(cfg.vocab_size)
        for name in ("pad", "bos", "eos", "unk"):
            if getattr(cfg, name + "_token_id") != tok.special_ids[name]:
                raise ValueError("model/tokenizer special token IDs mismatch")
        if manifest is None:
            raise ValueError("no admitted frozen V4 train-only tokenizer evidence")
        if tok.fingerprint != manifest["tokenizer_fingerprint"]:
            raise ValueError("tokenizer fingerprint mismatch")
    except Exception as exc:
        block("BLOCKED_TOKENIZER_NOT_READY", exc)
    expected = MINILLM_NANO.copy() if cpu_nano_test else MINILLM_87M.copy()
    if cpu_nano_test:
        expected.vocab_size = cfg.vocab_size
    if cfg.to_dict() != expected.to_dict():
        block("BLOCKED_MODEL_CONFIG", "requires the unchanged Nano test / 86,528,640-parameter preset")
    if raw.get("init_from"):
        block("BLOCKED_INITIALIZATION", "V4 pilot starts from its own saved random initialization; old weights are not adopted")
    if extra.get("allow_data_repetition", False) is not False:
        block("BLOCKED_TOKEN_BUDGET", "V4 pilot does not allow corpus repetition")
    budget = None
    try:
        train = PackedPretrainDataset(Path(dataset) / "train.bin", extra["seq_len"])
        val = PackedPretrainDataset(Path(dataset) / "val.bin", extra["seq_len"])
        if not len(train) or not len(val) or extra["seq_len"] > cfg.max_position_embeddings:
            raise ValueError("nonempty complete train/val windows within model context required")
        budget = validate_pilot_budget(len(train), opts, extra, cpu_nano_test=cpu_nano_test)
        if manifest and is_language(manifest):
            frozen = read_json(Path(dataset) / "budget-report.json")
            if (train.seq_len != frozen["seq_len"]
                    or budget["independent_effective_tokens"] != manifest["effective_train_tokens"]):
                raise ValueError("training window length and independent budget must match the frozen language Pilot")
        budget["val_windows"] = len(val)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        block("BLOCKED_TOKEN_BUDGET", exc)
    device = torch.device(opts.device or "cpu")
    available = torch.cuda.is_available()
    hw = hardware(device) if device.type == "cuda" and available else environment(torch.device("cpu"))
    hw.update(requested_device=str(device), cuda_available=available)
    if device.type == "cuda" and available:
        hw["free_memory_bytes"], hw["total_memory_bytes"] = torch.cuda.mem_get_info(device)
    if cpu_nano_test:
        if device.type != "cpu":
            block("BLOCKED_DEVICE", "CPU_NANO_TEST requires an explicit CPU device")
    elif device.type != "cuda" or not available:
        block("BLOCKED_GPU_NOT_READY", "explicit CUDA device unavailable; no CPU fallback")
    elif "T1200" not in hw["device_name"]:
        block("BLOCKED_GPU_NOT_READY", "this pilot configuration targets NVIDIA T1200; probe a separate plan for other hardware")
    ancestor = Path(opts.out).resolve().parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    disk = {"free_bytes": shutil.disk_usage(ancestor).free,
            "required_bytes": int(cfg.num_parameters * 44 * extra.get("disk_margin_factor", 1.2))}
    if disk["free_bytes"] < disk["required_bytes"]:
        block("BLOCKED_DISK", disk)
    if Path(opts.out).exists() and (not Path(opts.out).is_dir() or any(Path(opts.out).iterdir())) and not opts.resume:
        block("BLOCKED_OUTPUT_OCCUPIED", opts.out)
    evaluation = request = None
    try:
        expected_suites = _evaluation_suites(manifest)
        active_suites = list(suites or expected_suites)
        if not cpu_nano_test and active_suites != expected_suites:
            raise ValueError("production Pilot uses its pinned frozen evaluation suites")
        evaluation = evaluation_policy(evaluation_config, active_suites)
        if manifest is not None:
            request = _request(dataset, tokenizer, cfg, raw, evaluation, cpu_nano_test)
            request["approved_policy_sha256"] = pilot_policy.get("approved_policy_sha256")
        if Path(workspace).exists():
            _workspace_plan(workspace, request)
        elif opts.resume:
            raise ValueError("resume requires the original pilot workspace and initial checkpoint")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        block("BLOCKED_PILOT_IDENTITY", exc)
    proof = None
    if require_workload:
        try:
            if blockers:
                raise ValueError("admission failed; workload evidence cannot approve this run")
            _workspace_plan(workspace, request)
            policy = resolve_plan(dataset, tokenizer, model_config, _resolved(raw, workspace))[-1]
            proof = verify_workload(_resolved(raw, workspace)["workload_report"], policy,
                                    extra.get("gpu_memory_limit_mib", float("inf")))
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            block("BLOCKED_WORKLOAD_NOT_READY", exc)
    return {"schema_version": 1, "status": "READY" if not blockers else blockers[0],
            "readiness": "READY" if not blockers else "BLOCKED", "blockers": sorted(set(blockers)),
            "blocker_details": details, "scope": "CPU_NANO_TEST" if cpu_nano_test else "87M_GPU_PILOT",
            "dataset_audit": audit, "pilot_policy": pilot_policy, "data": {"content_id": manifest["content_id"],
                "corpus_version": manifest["corpus_version"], "license_status": manifest["license_review_status"]} if manifest else None,
            "model": {"parameters": cfg.num_parameters, "configuration": cfg.to_dict()},
            "tokenizer_fingerprint": manifest["tokenizer_fingerprint"] if manifest else None,
            "budget": budget, "environment": hw, "disk": disk, "request": request,
            "workload": {"status": proof["status"], "content_id": proof["content_id"],
                         "measurements": proof["measurements"]} if proof else {"status": "NOT_RUN"}}


def _initialize(workspace, request, model_config, tokenizer):
    if Path(workspace).exists():
        return _workspace_plan(workspace, request)
    from .tokenizer import MiniLLMTokenizer
    cfg = MiniLLMConfig.load(model_config) if Path(model_config).is_file() else get_config(model_config)
    with report_directory(workspace) as directory, preserved_process():
        set_seed(request["training"]["seed"])
        tok = MiniLLMTokenizer.load(tokenizer)
        tok.text_policy = request["text_policy"]
        model = MiniLLM(cfg)
        save_model_checkpoint(directory / "initial.pt", model, tok, step=0, tokens=0, val_loss=None,
                              extra={"mode": "pretrain", "baseline_status": "UNTRAINED_RANDOM_INITIALIZATION"})
        plan = sealed({"schema_version": 1, "request": request,
                       "initialization": "random_weights_saved_before_training",
                       "initial_checkpoint": file_info(directory / "initial.pt")})
        write_json(directory / "pilot-plan.json", plan)
    return plan


def _evaluations(checkpoint, tokenizer, dataset, destination, config, suites, device):
    result = []
    for index, suite in enumerate(suites):
        out = Path(destination) / f"suite-{index}"
        assets = _suite_assets(suite)
        measure_train = "benchmark" in assets
        if out.exists():
            report, manifest = verified_evaluation(out)
            policy = manifest["comparison_policy"]
            candidate_provenance = policy.get("candidate_provenance") or {}
            evidence = {"benchmark": policy.get("benchmark_sha256"), "prompts": policy.get("fixed_prompts_sha256"),
                        "candidate_benchmark": candidate_provenance.get("sha256")}
            if (report["checkpoint"]["sha256"] != file_info(Path(checkpoint))["sha256"]
                    or policy["config"] != asdict(config)
                    or any(evidence[name] != file_info(assets[name])["sha256"] if name in assets
                           else evidence[name] is not None for name in evidence)
                    or policy["dataset"]["dataset_content_id"] != read_json(Path(dataset) / "dataset-manifest.json")["content_id"]
                    or (measure_train and (policy.get("train_dataset") or {}).get("sha256")
                        != file_info(Path(dataset) / "train.bin")["sha256"])):
                raise ValueError("baseline evaluation no longer matches initial weights/data/frozen evaluation policy")
        else:
            evaluate_model(checkpoint, tokenizer, out, dataset=dataset, split="val",
                           benchmark=assets.get("benchmark"), prompts=assets.get("prompts", ""),
                           candidate_benchmark=assets.get("candidate_benchmark", ""),
                           config=config, device=device, measure_train=measure_train)
        result.append(out)
    return result


def run_pilot(dataset, tokenizer, model_config, training_config, out, workspace, *,
              execute=False, probe=False, cpu_nano_test=False, evaluation_config=None, suites=None,
              approved_config_path=None):
    raw = read_json(training_config) if isinstance(training_config, (str, Path)) else dict(training_config)
    opts, _ = parse_training_config(raw)
    out, workspace = Path(out).resolve(), Path(workspace).resolve()
    protected = [Path(dataset).resolve(), Path(opts.out).resolve(), workspace]
    if any(out == p or p in out.parents or out in p.parents for p in protected) or any(
            workspace == p or p in workspace.parents or workspace in p.parents for p in protected[:2]):
        raise ValueError("report, workspace, dataset and training output must be separate directories")
    if execute and probe:
        raise ValueError("probe and execute are separate explicit stages")
    evaluation_config = evaluation_config or EvaluationConfig(stage="pretrain")
    pre = preflight(dataset, tokenizer, model_config, raw, workspace, cpu_nano_test=cpu_nano_test,
                    evaluation_config=evaluation_config, suites=suites, require_workload=not probe,
                    approved_config_path=approved_config_path)
    suites = list(suites or _evaluation_suites(pre["data"]))
    report = {"schema_version": 1, "scope": pre["scope"], "status": pre["status"],
              "training_status": "NOT_RUN", "gpu_training_status": "NOT_RUN", "preflight": pre,
              "initialization": None, "training": None, "evaluation": None, "recovery": {"status": "NOT_RUN"},
              "performance": None, "stability": {"status": "NOT_RUN", "errors": None},
              "engineering": "NOT_RUN", "convergence": "NOT_RUN", "capability": "NOT_RUN"}
    with report_directory(out) as directory:
        write_json(directory / "preflight.json", pre)
        write_json(directory / "pilot-v4-1m-report.json", report)
    if pre["readiness"] != "READY":
        return report
    try:
        if not probe:
            from .training_readiness import check_readiness
            ready = check_readiness(dataset, tokenizer, model_config, _resolved(raw, workspace), out / "readiness")
            report["readiness"] = ready
            if ready["status"] != "READY":
                report["status"] = "BLOCKED_READINESS"
                write_json(out / "pilot-v4-1m-report.json", report)
                return report
        if probe:
            plan = _initialize(workspace, pre["request"], model_config, tokenizer)
            probe_config = {**_resolved(raw, workspace), "resume": None,
                            "init_from": str(workspace / "initial.pt")}
            proof = probe_workload(dataset, tokenizer, model_config, probe_config, probe_config["workload_report"])
            report.update(status="WORKLOAD_" + proof["status"], initialization=plan,
                          workload={"status": proof["status"], "measurements": proof["measurements"]})
        elif execute:
            plan = _workspace_plan(workspace, pre["request"])
            baseline = _evaluations(workspace / "initial.pt", tokenizer, dataset, workspace / "baseline",
                                    evaluation_config, suites, opts.device)
            # Recheck all V4 evidence after baseline/probe and immediately before
            # the existing launcher performs its own readiness/rehash/ownership checks.
            again = preflight(dataset, tokenizer, model_config, raw, workspace, cpu_nano_test=cpu_nano_test,
                              evaluation_config=evaluation_config, suites=suites,
                              approved_config_path=approved_config_path)
            if again["readiness"] != "READY" or again["request"] != pre["request"]:
                raise ValueError("V4 inputs/evaluation/workload changed after preflight")
            launched_at = time.perf_counter()
            report.update(training_status="EXECUTION_ATTEMPTED",
                          gpu_training_status="NOT_RUN" if cpu_nano_test else "EXECUTION_ATTEMPTED")
            write_json(out / "pilot-v4-1m-report.json", report)
            if torch.device(opts.device or "cpu").type == "cuda":
                torch.cuda.reset_peak_memory_stats(torch.device(opts.device))
            launch = launch_pilot(dataset, tokenizer, model_config, _resolved(raw, workspace), out / "launch", execute=True)
            report["launch"] = launch
            if not launch.get("executed"):
                report.update(status="BLOCKED_READINESS", training_status="NOT_RUN", gpu_training_status="NOT_RUN")
            else:
                summary = launch["summary"]
                report.update(initialization=plan, training=summary, training_status="INCOMPLETE",
                              gpu_training_status="NOT_RUN" if cpu_nano_test else "INCOMPLETE",
                              status="INCOMPLETE", recovery={"status": "CHECKPOINT_VALIDATED_BY_LAUNCHER",
                              "resumed": bool(opts.resume), "lineage": str(Path(opts.out) / "pilot-lineage.json")})
                report["performance"] = {"launch_process_seconds": time.perf_counter() - launched_at,
                    "training_process_seconds": summary["seconds"],
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated() if opts.device.startswith("cuda") else None,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved() if opts.device.startswith("cuda") else None,
                    "memory_scope": "this launch/readiness/training/checkpoint process, before final evaluation"}
                if pre["data"] and is_language(pre["data"]):
                    from .v4_training_accounting import trained_category_consumption

                    report["train_category_consumption"] = trained_category_consumption(
                        dataset, opts.out, summary, pre["data"]["content_id"])
                rows = [r for r in records(Path(opts.out) / "training-log.jsonl") if r.get("event") == "train"]
                report["last_training_interval"] = {k: rows[-1][k] for k in
                    ("loss", "learning_rate", "gradient_norm", "tokens_per_second", "step_seconds")} if rows else None
                if summary["stop_reason"] == "target_reached":
                    if summary["global_step"] != pre["budget"]["optimizer_updates"] or summary["tokens_seen"] != pre["budget"]["planned_effective_tokens"]:
                        raise ValueError("actual optimizer updates/tokens differ from fixed pilot plan")
                    final = _evaluations(Path(opts.out) / "model.pt", tokenizer, dataset, out / "final",
                                         evaluation_config, suites, opts.device)
                    acceptance = [evaluate_pilot(opts.out, b, f, out / f"acceptance-{i}")
                                  for i, (b, f) in enumerate(zip(baseline, final))]
                    passed = all(a["engineering"] == "ENGINEERING_PASS" for a in acceptance)
                    report.update(status="COMPLETED" if passed else "FAILED_ACCEPTANCE", training_status="COMPLETED", evaluation=acceptance,
                                  engineering="ENGINEERING_PASS" if all(a["engineering"] == "ENGINEERING_PASS" for a in acceptance) else "ENGINEERING_FAIL",
                                  convergence=[a["convergence"] for a in acceptance],
                                  capability="INCONCLUSIVE_REQUIRES_MORE_EVIDENCE",
                                  gpu_training_status="NOT_RUN" if cpu_nano_test else "COMPLETED")
                    report["stability"] = {"status": "CHECKED", "errors": [e for a in acceptance for e in a["errors"]],
                        "scope": "existing finite loss/gradient/optimizer guards and full append-log analysis; failures abort, no OOM/precision retry"}
                    if pre["data"] and is_language(pre["data"]):
                        initial_eval, _ = verified_evaluation(baseline[0])
                        final_eval, _ = verified_evaluation(final[0])
                        report["language_checkpoint_comparison"] = [
                            {key: evaluation[key] for key in (
                                "checkpoint", "train_language_model", "language_model", "scores", "diagnostics")}
                            for evaluation in (initial_eval, final_eval)]
                    report["performance"].update(total_logged_seconds=acceptance[0]["estimated_logged_wall_seconds"],
                        average_logged_tokens_per_second=acceptance[0]["average_logged_tokens_per_second"])
        write_json(out / "pilot-v4-1m-report.json", report)
    except Exception as exc:
        report.update(status="FAILED", error=str(exc))
        report["stability"] = {"status": "FAILED", "errors": [type(exc).__name__ + ": " + str(exc)]}
        write_json(out / "pilot-v4-1m-report.json", report)
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="artifacts/corpus/v4-pilot-1m-v1")
    parser.add_argument("--tokenizer")
    parser.add_argument("--config", default="minillm-87m")
    parser.add_argument("--training-config", default="configs/pilot-v4-1m-t1200.json")
    parser.add_argument("--out", default="reports/pr9-v4-pilot")
    parser.add_argument("--workspace", default="artifacts/pilot-v4-1m-v1")
    parser.add_argument("--evaluation-config", default="configs/evaluation.pretrain.json")
    parser.add_argument("--test-suite", action="append", type=Path,
                        help="custom frozen fixture suites are accepted only with --cpu-nano-test")
    parser.add_argument("--test-policy-config", type=Path,
                        help="explicit versioned CPU fixture policy; cannot override production policy")
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--probe", action="store_true")
    stages.add_argument("--execute", action="store_true")
    parser.add_argument("--cpu-nano-test", action="store_true", help="CPU-only fixture scope, never GPU acceptance")
    args = parser.parse_args(argv)
    try:
        report = run_pilot(args.dataset, args.tokenizer or str(Path(args.dataset) / "tokenizer.json"),
                           args.config, args.training_config, args.out, args.workspace,
                           execute=args.execute, probe=args.probe, cpu_nano_test=args.cpu_nano_test,
                           evaluation_config=EvaluationConfig(**read_json(args.evaluation_config)), suites=args.test_suite,
                           approved_config_path=args.test_policy_config)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"BLOCKED V4 pilot: {exc}", file=sys.stderr)
        return 1
    print(report["status"], "training=" + report["training_status"], "scope=" + report["scope"])
    return 0 if report["status"] in ("READY", "COMPLETED", "WORKLOAD_PASS") else 1
