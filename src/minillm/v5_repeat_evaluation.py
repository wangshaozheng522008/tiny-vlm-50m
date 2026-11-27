"""Evaluate validated V5 continuation nodes and publish an optional final Test."""

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import csv
import json
import math
import time

import torch

from . import v5_language_evaluation as original
from . import v5_repeat_training as experiment
from .assessment_io import environment, publish_manifest, read_json, records, report_directory, write_json
from .checkpoint import CheckpointError, load_training_state
from .corpus_utils import file_info, verify_file
from .evaluation import EvaluationConfig, load_evaluation_model, load_tasks
from .evaluation_windows import evaluate_indexed_windows, indexed_train_plan
from .language_diagnostics import BENCHMARK, CONTINUATIONS, PROBES
from .language_generation_analysis import SEEDS, STRATEGIES
from .language_scaling_evaluation import DECODE, _completed_nodes, _quantized_report, evaluation_data
from .language_token_analysis import vocabulary
from .training_milestones import require_identical_weights, verify_milestone, verify_model_metadata
from .v4_language_scaling import base_manifest
from .v5_language_training_report import knowledge_score, plot_results
from .v5_repeat_diagnostics import LONG_SPAN, MIN_SPAN, generation_diagnostics, train_overlap
from .v5_repeat_sampling import accounting
from .v5_repeat_training import MILESTONES, PARENT_STEP, TOKENS_PER_UPDATE

FINAL = f"step{MILESTONES[-1]}"
INT4 = f"{FINAL}_int4"
HISTORICAL_RECEIPT = {"bytes": 3726, "sha256": "333a095606a8af8e51aea5fb05262f9e01182346a891a50c1c3cd646ff3cdbb0"}
SAMPLED = {original.FINAL, original.INT4, FINAL, INT4}


def _context(tokenizer, model_config, device):
    profile = experiment.profile
    config = EvaluationConfig(**read_json(profile.EVALUATION_CONFIG))
    files = [f"{split}.{kind}" for split in ("train", "val", "test") for kind in ("bin", "token-index.jsonl")]
    base_manifest(*files)
    data, indices, windows = evaluation_data(config)
    assets = {str(path): file_info(path) for path in (profile.ROOT / BENCHMARK, CONTINUATIONS, PROBES)}
    benchmark = [{**task, "generation_mode": "raw"} for task in load_tasks(profile.ROOT / BENCHMARK, scored=True)
                 if task["generation_mode"] in ("raw", "both")]
    probes = load_tasks(CONTINUATIONS, scored=False) + [
        {**row, "generation_mode": "raw", "text_category": "zh"} for row in read_json(PROBES)["probes"]]
    if (len(benchmark), len(probes)) != (300, 9):
        raise ValueError("the fixed generation population changed")
    validation = original._all_windows(profile.DATASET / "val.bin", config)
    implementation = {name: file_info(Path(__file__).with_name(name)) for name in (
        "v5_language_evaluation.py", "language_scaling_evaluation.py", "evaluation.py",
        "evaluation_windows.py", "language_diagnostics.py", "language_generation_analysis.py",
        "evaluation_scoring.py", "quantize.py", "model.py")}
    protocol = {"config": asdict(config), "implementation": implementation, "assets": assets,
                "windows": windows, "v5_val": validation[2], "training_content_id": profile.CONTENT_ID,
                "seeds": list(SEEDS), "strategies": {name: STRATEGIES[name] for name in DECODE},
                "batch_size": 24, "repetition_penalty": 1.0}
    runtime = {"tokenizer": tokenizer, "model_config": model_config, "config": config, "device": device,
               "vocab": vocabulary(tokenizer), "historical": {}, "benchmark": benchmark, "probes": probes,
               "v5_val": validation, "protocol": protocol,
               "common": {key: (data[key], indices[key], plan) for key, plan in windows.items()}}
    return runtime


def _history(context, expected_consumption):
    profile = experiment.profile
    data = profile.REPORT / "evaluation"
    model_paths, audits, _, proof = _completed_nodes(
        profile.RUN, profile.DATASET, steps=profile.MILESTONES, version=profile.V5_VERSION,
        training_config=profile.TRAINING_CONFIG)
    if audits[original.FINAL]["consumption"] != expected_consumption["consumption"]:
        raise CheckpointError("PR #20 sampler consumption differs from the frozen single-epoch replay")
    names = [original.BASELINE, original.HISTORICAL_LABEL, *(f"step{step}" for step in profile.MILESTONES), original.INT4]
    node_manifests = {name: file_info(data / "nodes" / name / "manifest.json") for name in names}
    original._verify_comparison(data / "comparison", node_manifests, context["protocol"], audits, proof,
                                expected_consumption["sources"], expected=HISTORICAL_RECEIPT)
    int4 = original._verify_int4_export(model_paths[original.FINAL], data / "int4", proof)
    model_paths[original.INT4] = int4
    results = {}
    for name in (original.FINAL, original.INT4):
        node = data / "nodes" / name
        if not node.is_dir():
            raise ValueError("PR #20 evaluation cache must already exist")
        results[name] = original._evaluate_node(node, name, model_paths[name],
                                               {**context, "final_test_output": data / "final-v5-test"})
    return results, {name: data / "nodes" / name for name in results}, model_paths


def _verify_stage(prepared):
    settings, replay, lineage = prepared.settings, prepared.replay, prepared.lineage
    run = experiment.RUN
    manifest = read_json(run / "training-manifest.json")
    reference = prepared.manifest()
    if {**reference, "run_id": manifest["run_id"]} != manifest:
        raise CheckpointError("evaluation training lineage or plan differs")
    summary = read_json(run / "train-summary.json")
    final = load_training_state(run / "state.pt", mmap=True)
    stage = experiment.verify_progress(final, manifest, replay, settings)
    fields = ("global_step", "samples_seen", "tokens_seen", "epoch", "micro_batches_consumed_in_epoch")
    if (stage != settings["stage_steps"] or summary["stop_reason"] != "target_reached"
            or any(summary[key] != final[key] for key in (*fields, "best_checkpoint"))
            or summary["final_val"] != final["history"][-1]
            or final["history"][0] != prepared.parent["history"][-1]
            or summary["stage_steps"] != stage
            or summary["stage_effective_tokens"] != final["continuation"]["stage_effective_tokens"]):
        raise CheckpointError("a completed, consistent continuation stage is required")
    best = final["best_checkpoint"]
    verify_file(run / "best.pt", best["file"])
    status = read_json(run / "best-checkpoint-status.json")
    if status != {"status": "AVAILABLE", "checkpoint": best, "historical_best_loss": best["val_loss"]}:
        raise CheckpointError("published validation selection differs from the committed state")

    paths, audits = {}, {}
    evidence_paths = ["state.pt", "model.pt", "train-summary.json", "training-manifest.json", "training-log.jsonl",
                      "metrics.json", "best-checkpoint-status.json", "best.pt"]
    for step in MILESTONES:
        node = run / "milestones" / f"step-{step}"
        name = f"step{step}"
        receipt = verify_milestone(node, manifest, step=step, tokens=step * TOKENS_PER_UPDATE)
        state = load_training_state(node / "state.pt", mmap=True)
        index = experiment.verify_progress(state, manifest, replay, settings)
        if state["global_step"] != step:
            raise CheckpointError(f"{node}: state does not match milestone")
        path = node / "model.pt"
        model = experiment._load_model(path)
        verify_model_metadata(model, state, context=str(path))
        if (model.get("stage_step") != index or model.get("parent_run_id") != lineage["run_id"]
                or model.get("training_run_id") != manifest["run_id"]):
            raise CheckpointError("model's continuation lineage differs")
        require_identical_weights(state["model"], model["model"], str(path))
        consumption = accounting(replay, prepared.documents, prepared.corpus_plan, settings["seq_len"],
                                 index * settings["grad_accum"], enforce=True)
        if consumption["cumulative"]["effective_tokens"] != state["tokens_seen"]:
            raise CheckpointError("milestone sampler replay disagrees with committed consumption")
        paths[name], audits[name] = path, {"checkpoint": receipt, "consumption": consumption}
        evidence_paths.extend(str((node / filename).relative_to(run)) for filename in ("manifest.json", "state.pt", "model.pt"))
    if read_json(run / "metrics.json") != final["history"] or audits[FINAL]["consumption"] != prepared.preview:
        raise CheckpointError("completed validation or sampler evidence differs from the committed state")
    require_identical_weights(state["model"], final["model"], "final rolling recovery state")
    published = experiment._load_model(run / "model.pt")
    verify_model_metadata(published, final, context="published final model")
    if any(published.get(key) != model[key] for key in ("stage_step", "parent_run_id", "training_run_id")):
        raise CheckpointError("published model continuation lineage differs")
    require_identical_weights(final["model"], published["model"], "published final model")
    proof = {"status": "PASS", "files": {name: file_info(run / name) for name in evidence_paths},
             "training_state": {key: final[key] for key in fields} | {"stage_step": stage},
             "selected_by_validation": best, "parent": lineage, "run_id": manifest["run_id"],
             "tensor_comparison": {"status": "EXACT", "tensors": len(final["model"])}}
    return paths, audits, proof, summary


def _historical_train(output, history, paths, context):
    receipt = {"protocol": context["protocol"]["windows"]["v5_train"],
               "checkpoints": {name: result["checkpoint"] for name, result in history.items()}}
    if output.exists():
        proof = original._verify_report(output, required=("losses.json",))
        if proof["identity"] != receipt or proof["status"] != "EVALUATED":
            raise ValueError("historical fixed V5 Train loss differs from this protocol")
        return read_json(output / "losses.json")
    results = {}
    for name in history:
        model, payload = load_evaluation_model(paths[name], context["tokenizer"], context["device"])
        del payload
        results[name] = evaluate_indexed_windows(model, *context["common"]["v5_train"], context["device"],
                                                 context["config"].batch_size)
        del model
    with report_directory(output) as directory:
        write_json(directory / "losses.json", results)
        publish_manifest(directory, "manifest.json", {"status": "EVALUATED", "identity": receipt})
    return results


def _learning_log(directory, settings, replay):
    parents = [row for row in records(experiment.profile.RUN / "training-log.jsonl") if row["event"] == "train"]
    current = [row for row in records(experiment.RUN / "training-log.jsonl") if row["event"] == "train"]
    if [row["global_step"] for row in parents] != list(range(1, PARENT_STEP + 1)):
        raise ValueError("complete PR #20 per-update evidence is required")
    validation = {row["step"]: row["loss"] for path in (experiment.profile.RUN, experiment.RUN)
                  for row in read_json(path / "metrics.json")}
    evidence = experiment.verify_update_log(current, settings, replay, settings["stage_steps"])
    write_json(directory / "window-evidence.json", evidence)
    with (directory / "loss-and-learning-rate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("step", "stage_step", "tokens_seen", "observed_learning_rate",
                         "train_loss", "val_loss", "gradient_norm", "update_seconds"))
        for row in parents + current:
            step = row["global_step"]
            stage_step = row.get("stage_step", step)
            if (row["tokens_seen"], row["loss_tokens"]) != (step * TOKENS_PER_UPDATE, TOKENS_PER_UPDATE):
                raise ValueError(f"update {step} does not agree with the cumulative token budget")
            writer.writerow((step, stage_step, row["tokens_seen"], row["learning_rate"], row["loss"],
                             validation.get(step), row["gradient_norm"], row["step_seconds"]))


def _diagnose(directory, rows, tokens, documents):
    matches = train_overlap(tokens, documents, rows)
    by_cohort = defaultdict(list)
    with (directory / "train-overlap.jsonl").open("w", encoding="utf-8") as handle:
        for row, match in zip(rows, matches, strict=True):
            key = {field: row[field] for field in ("checkpoint", "strategy", "seed", "kind", "id")}
            by_cohort[tuple(key[field] for field in ("checkpoint", "strategy", "seed", "kind"))].append((row, match))
            handle.write(json.dumps({**key, "match": match}, ensure_ascii=False) + "\n")
    aggregates = []
    for (name, strategy, seed, kind), cohort in by_cohort.items():
        count = len(cohort)
        lengths = [match["matched_tokens"] if match else 0 for _, match in cohort]
        aggregates.append({"checkpoint": name, "strategy": strategy, "seed": seed, "kind": kind,
                           "at_least_16_tokens": sum(length >= MIN_SPAN for length in lengths),
                           "at_least_32_tokens": sum(length >= LONG_SPAN for length in lengths),
                           "at_least_32_fraction": sum(length >= LONG_SPAN for length in lengths) / count,
                           "max_matched_train_tokens": max(lengths),
                           **generation_diagnostics([row for row, _ in cohort])})
    write_json(directory / "generation-diagnostics.json", aggregates)


def _aggregate(output, results, nodes, context, audits, proof, summary, prepared, started):
    destination = output / "comparison"
    protocol = context["protocol"]
    consumption = audits[FINAL]["consumption"]
    sources = {period: consumption[period]["sources"] for period in ("stage", "cumulative")}
    manifests = {name: file_info(node / "manifest.json") for name, node in nodes.items()}
    if destination.exists():
        original._verify_comparison(destination, manifests, protocol, audits, proof, sources)
        if results != read_json(destination / "comparison.json"):
            raise ValueError("published comparison metrics differ from verified evaluation nodes")
        return destination
    with report_directory(destination) as directory:
        predictions = [row for node in nodes.values() for row in records(node / "predictions.jsonl")]
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        _diagnose(directory, predictions, prepared.train.tokens, prepared.documents)
        _learning_log(directory, prepared.settings, prepared.replay)
        write_json(directory / "comparison.json", results)
        write_json(directory / "comparison-policy.json", protocol)
        write_json(directory / "checkpoint-and-consumption.json", audits)
        write_json(directory / "actual-source-consumption.json", sources)
        write_json(directory / "final-checkpoint-consistency.json", proof)
        write_json(directory / "training-summary.json", summary)
        write_json(directory / "next-token-traces.json",
                   {name: read_json(node / "next-token-traces.json") for name, node in nodes.items()})
        for fp32, quantized, label in ((original.FINAL, original.INT4, "10m"), (FINAL, INT4, "20m")):
            logits = {name: torch.load(nodes[name] / "next-token-logits.pt", weights_only=True)
                      for name in (fp32, quantized)}
            _quantized_report(directory, results, logits, final=fp32, quantized=quantized)
            paired = read_json(directory / "fp32-int4.json")
            paired["paired_loss_and_perplexity"] = {
                split: {kind: {key: results[name]["lm"][split][key] for key in ("loss", "perplexity")}
                        for kind, name in (("fp32", fp32), ("int4", quantized))}
                for split in ("train", "val", "v5_train", "v5_val")}
            write_json(directory / f"fp32-int4-{label}.json", paired)
        (directory / "fp32-int4.json").unlink()
        plot_results(directory, results)
        publish_manifest(directory, "manifest.json", {"status": "EVALUATED", "environment": environment(context["device"]),
                         "node_manifests": manifests, "elapsed_seconds_this_invocation": time.monotonic() - started,
                         "prediction_count": len(predictions), "human_review": "PENDING"})
    return destination


def _locked_test(output, comparison, context, model_path, proof):
    receipt = {"checkpoint": file_info(model_path), "protocol": context["protocol"], "node": FINAL,
               "locked_comparison": file_info(comparison / "manifest.json"),
               "selected_by_validation": proof["selected_by_validation"],
               "historical_10m_test_previously_evaluated": True}
    if output.exists():
        return original._final_test(output, receipt, context)
    model, payload = load_evaluation_model(model_path, context["tokenizer"], context["device"])
    del payload
    return original._final_test(output, receipt, context, model)


def evaluate(*, final_test=False):
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise ValueError("repeat evaluation requires CUDA")
    with experiment.run_lock():
        prepared = experiment.prepare(publish=False, check_hardware=False)
        train, tokenizer = prepared.train, prepared.tokenizer
        device = torch.device("cuda")
        paths, audits, proof, summary = _verify_stage(prepared)
        context = _context(tokenizer, prepared.parent["config"], device)
        historical, history_nodes, history_paths = _history(context, experiment.profile.preview(experiment.profile.DATASET))
        config = context["config"]
        documents, plan = indexed_train_plan(experiment.profile.DATASET, train, seed=config.seed,
                                             token_budget=config.token_budget)
        frozen = read_json(experiment.profile.DATASET / "dataset-manifest.json")
        plan["bin"] = frozen["files"]["train.bin"]
        context["common"].pop("test")
        context["common"]["v5_train"] = (train, documents, plan)
        windows = {key: data[2] for key, data in context["common"].items()}
        context["protocol"] = {
            **context["protocol"], "windows": windows, "repeat_stage_sampled_nodes": sorted(SAMPLED),
            "repeat_evaluation": file_info(Path(__file__)), "train_overlap": {
                "implementation": file_info(Path(__file__).with_name("v5_repeat_diagnostics.py")),
                "minimum_exact_contiguous_content_tokens": MIN_SPAN, "long_span_tokens": LONG_SPAN,
                "train_bin": frozen["files"]["train.bin"], "token_index": frozen["files"]["train.token-index.jsonl"],
            },
        }
        context["sampled"] = SAMPLED
        output = experiment.REPORT / "evaluation"
        old_losses = _historical_train(output / "historical-v5-train", historical, history_paths, context)
        for name, result in historical.items():
            result["lm"] = {split: result["lm"][split] for split in ("train", "val", "v5_val")}
            result["lm"]["v5_train"] = old_losses[name]
        results, nodes = dict(historical), dict(history_nodes)
        paths[INT4] = original._quantize(paths[FINAL], output / "int4", tokenizer, device, proof)
        for name, path in paths.items():
            print(f"[evaluation] {name}", flush=True)
            destination = output / "nodes" / name
            results[name] = original._evaluate_node(destination, name, path, context)
            nodes[name] = destination
        for result in results.values():
            result["loss_gap_v5_val_minus_fixed_v5_train"] = (
                result["lm"]["v5_val"]["loss"] - result["lm"]["v5_train"]["loss"])
            result["world_knowledge_greedy"] = dict(zip(("correct", "total"), knowledge_score(result), strict=True))
            result["human_review"] = "PENDING"
        comparison = _aggregate(output, results, nodes, context, audits, proof, summary, prepared, started)
        if final_test:
            loss, _ = _locked_test(output / "final-v5-test", comparison, context, paths[FINAL], proof)
            if not math.isfinite(loss["loss"]):
                raise ValueError("final test loss is nonfinite")
            print(f"[final test] loss={loss['loss']:.6f}", flush=True)
        print(f"[evaluation complete] {comparison}", flush=True)
