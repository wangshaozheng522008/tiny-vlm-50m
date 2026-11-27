"""Paired evaluation for a completed, admitted 2M language Pilot."""

import csv
from dataclasses import replace
from itertools import islice
import json
from pathlib import Path

import torch

from .assessment_io import environment, publish_manifest, read_json, records, report_directory, verified_evaluation
from .checkpoint import CheckpointError, load_training_state
from .corpus_budget import read_token_index
from .corpus_utils import file_info, verify_file
from .data import EpochBatchSampler, PackedPretrainDataset
from .evaluation import EvaluationConfig, load_evaluation_model, load_tasks
from .evaluation_windows import evaluate_indexed_windows, indexed_train_plan, representative_windows
from .language_diagnostics import (BENCHMARK, CONTINUATIONS, FORMAL_CONFIG, HISTORICAL_FINAL, PROBES,
                                  _record_predictions, verify_historical_greedy)
from .language_generation_analysis import (SEEDS, STRATEGIES, batch_generations, generation_metrics,
                                           quantization_comparison, trace_predictions)
from .language_token_analysis import vocabulary
from .pilot_lineage import _validate_v2
from .quantize import save_quantized
from .tokenizer import MiniLLMTokenizer
from .train import _validate_progress, lr_at
from .training_manifest import dataset_identity, write_json
from .training_milestones import require_identical_weights, verify_milestone, verify_model_metadata
from .v4_admission import validate_v4_pilot_policy
from .v4_language_scaling import (BASE, BASE_ID, FINGERPRINT, HISTORICAL, HISTORICAL_FILE, INITIAL, INITIAL_FILE,
                                  OUTPUT, ROOT, TARGET, TRAINING_CONFIG, VERSION, base_manifest, verify_pinned_release)
from .v4_training_accounting import category_consumption

BASELINE = "historical_step123"
FINAL = "step245"
DECODE = ("greedy", "standard_sampling", "high_temperature", "nucleus")
HUMAN_CRITERIA = ("fluency", "semantic_coherence", "context_relevance", "fact_correctness", "meaningless_repetition")
HISTORICAL_EVALUATION_FILES = {
    "evaluation-manifest.json": {"bytes": 11068, "sha256": "4b37c5f95a558efdd78ba4b3007b20184f6bc210754552cc7f3204ce937a8791"},
    "evaluation.json": {"bytes": 7374, "sha256": "5848b5375c2857ab747c0849a540a240a28432734126a07bc52b83b8ad083c08"},
    "predictions.jsonl": {"bytes": 498274, "sha256": "783ab9e8e3d2370d3200b54e302b6a3c2aa278ec1bb9fe7289380196546859f8"},
}


def load_historical_greedy(directory):
    for name, info in HISTORICAL_EVALUATION_FILES.items():
        verify_file(directory / name, info)
    report, manifest = verified_evaluation(directory)
    policy = manifest["comparison_policy"]
    if (report["checkpoint"]["sha256"] != HISTORICAL_FILE["sha256"]
            or report["checkpoint"]["global_step"] != 123
            or report["dataset"]["dataset_content_id"] != BASE_ID
            or manifest["tokenizer_fingerprint"] != FINGERPRINT
            or policy["benchmark_sha256"] != file_info(ROOT / BENCHMARK)["sha256"]):
        raise ValueError("historical formal evaluation does not identify PR #16 Step 123")
    predictions = [row for row in records(directory / "predictions.jsonl") if row["kind"] == "benchmark"]
    return predictions, manifest


def check_scaling_historical_greedy(benchmark, current, historical, manifest, config, strategy, penalty):
    result = verify_historical_greedy(benchmark, historical, current, manifest, config, strategy, penalty)
    return {**result, "historical_evaluation_id": manifest["evaluation_id"],
            "historical_checkpoint": HISTORICAL_FILE, "historical_files": HISTORICAL_EVALUATION_FILES}


def evaluation_data(config, *, original=BASE):
    """Bind every checkpoint to the same original Train sample and fixed evaluation BINs."""
    data, index, plan = {}, {}, {}
    for split in ("train", "val", "test"):
        data[split] = PackedPretrainDataset(original / f"{split}.bin", config.seq_len)
        if split == "train":
            index[split], plan[split] = indexed_train_plan(original, data[split], seed=config.seed,
                                                           token_budget=config.token_budget)
        else:
            index[split] = read_token_index([original / f"{split}.token-index.jsonl"])["records"]
            plan[split] = representative_windows(index[split], data[split], seed=config.seed, token_budget=0)
        plan[split]["bin"] = file_info(original / f"{split}.bin")
        plan[split]["token_index"] = file_info(original / f"{split}.token-index.jsonl")
    return data, index, plan


def _verify_step_state(state, manifest, data, sampler, step):
    if state["manifest"] != manifest:
        raise CheckpointError(f"checkpoint {step}: training manifest differs")
    _validate_progress(state, manifest, data, sampler)
    if (state["global_step"], state["samples_seen"], state["tokens_seen"]) != (step, step * 8, step * 8184):
        raise CheckpointError(f"checkpoint {step}: committed step, samples or supervision tokens differ")


def _verify_pilot_model(model, state, lineage, context):
    verify_model_metadata(model, state, context=context)
    requests = {lineage["initial_launch"]["request_id"]} | {
        row["request_id"] for row in lineage["resume_history"] if row["checkpoint_step"] <= state["global_step"]}
    if (model.get("pilot_initialization") != lineage["initialization"] or model.get("pilot_request_id") not in requests
            or model.get("seq_len") != state["manifest"]["training"]["seq_len"]):
        raise CheckpointError(f"{context}: checkpoint initialization or pilot lineage differs")


def _verify_final_publication(run, manifest, summary, milestone_state, data, sampler, lineage, *, steps=(0, 123, 245)):
    final_step = steps[-1]
    paths = {"training-manifest.json": run / "training-manifest.json", "train-summary.json": run / "train-summary.json",
             "pilot-lineage.json": run / "pilot-lineage.json", "state.pt": run / "state.pt", "model.pt": run / "model.pt"}
    for step in steps:
        name = f"milestones/step-{step}/manifest.json"
        paths[name] = run / name
    for filename in ("state.pt", "model.pt"):
        name = f"milestones/step-{final_step}/{filename}"
        paths[name] = run / name
    final = load_training_state(paths["state.pt"], mmap=True)
    _verify_step_state(final, manifest, data, sampler, final_step)
    model = torch.load(paths["model.pt"], map_location="cpu", mmap=True, weights_only=False)
    _verify_pilot_model(model, final, lineage, "published model.pt")
    if any(summary.get(key) != model[key] for key in ("pilot_initialization", "pilot_request_id")):
        raise CheckpointError("published model.pt departure from the train summary lineage")
    require_identical_weights(milestone_state["model"], final["model"], "published state.pt")
    require_identical_weights(milestone_state["model"], model["model"], "published model.pt")
    return {"status": "PASS", "files": {name: file_info(path) for name, path in paths.items()},
            "training_state": {key: final[key] for key in ("global_step", "samples_seen", "tokens_seen")}
            | {"stop_reason": summary["stop_reason"], "scheduler_total_steps": manifest["training"]["scheduler_total_steps"]},
            "run_id": manifest["run_id"], "dataset_content_id": manifest["dataset"]["train"]["dataset_content_id"],
            "tokenizer_fingerprint": manifest["tokenizer_fingerprint"], "initialization": lineage["initialization"],
            "model_config": manifest["model_config"],
            "tensor_comparison": {"status": "EXACT", "tensors": len(final["model"]),
                                  "compared_fields": ["name", "shape", "dtype", "values"]}}


def _completed_nodes(run, dataset, *, steps=(0, 123, 245), version=VERSION,
                     training_config=TRAINING_CONFIG):
    final_step = steps[-1]
    summary = read_json(run / "train-summary.json")
    manifest = read_json(run / "training-manifest.json")
    policy = manifest["training"]
    training = read_json(training_config)
    if (summary["global_step"], summary["samples_seen"], summary["tokens_seen"], summary["stop_reason"],
            policy["target_optimizer_updates"], policy["scheduler_total_steps"], policy["milestone_steps"],
            manifest["tokenizer_fingerprint"]) != (final_step, final_step * 8, final_step * 8184, "target_reached",
                                                final_step, final_step, list(steps), FINGERPRINT):
        raise ValueError(f"a completed {final_step}-step language training state is required")
    data = PackedPretrainDataset(dataset / "train.bin", 1024)
    val = PackedPretrainDataset(dataset / "val.bin", 1024)
    frozen = read_json(dataset / "dataset-manifest.json")
    if (frozen["corpus_version"] != version or frozen["content_id"] != manifest["dataset"]["train"]["dataset_content_id"]
            or frozen["tokenizer_fingerprint"] != FINGERPRINT
            or any(dataset_identity(split_data, manifest["model_config"]["ignore_index"])[0] != manifest["dataset"][split]
                   for split, split_data in (("train", data), ("val", val)))):
        raise ValueError("training dataset and recorded run differ")
    lineage = _validate_v2(read_json(run / "pilot-lineage.json"), manifest, run)
    if lineage["initialization"] != {"sha256": INITIAL_FILE["sha256"], "kind": "weights_only"}:
        raise ValueError("training lineage does not identify the pinned PR #16 initialization")
    sampler = EpochBatchSampler(len(data), training["batch_size"], training["seed"])
    index = read_token_index([dataset / "train.token-index.jsonl"])["records"]
    nodes, audits = {BASELINE: HISTORICAL}, {}
    verify_file(INITIAL, INITIAL_FILE)
    verify_file(HISTORICAL, HISTORICAL_FILE)
    initial = torch.load(INITIAL, map_location="cpu", mmap=True, weights_only=False)
    for step in steps:
        directory = run / "milestones" / f"step-{step}"
        receipt = verify_milestone(directory, manifest, step=step, tokens=step * 8184)
        state = load_training_state(directory / "state.pt", mmap=True)
        _verify_step_state(state, manifest, data, sampler, step)
        path = directory / "model.pt"
        model = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        _verify_pilot_model(model, state, lineage, f"milestone {step}")
        require_identical_weights(state["model"], model["model"], f"milestone {step}")
        if step == 0:
            require_identical_weights(initial["model"], model["model"], "pinned PR #16 initialization")
        windows = [window for batch in islice(sampler.batches(0), step * policy["grad_accum"]) for window in batch]
        consumption = category_consumption(index, 1024, windows)
        if consumption["effective_tokens"] != state["tokens_seen"]:
            raise ValueError("observed sampler does not reproduce the train state")
        audits[f"step{step}"] = {"checkpoint": receipt, "consumption": consumption}
        nodes[f"step{step}"] = path
    final_proof = _verify_final_publication(run, manifest, summary, state, data, sampler, lineage, steps=steps)
    return nodes, audits, manifest, final_proof


def _curves(run, directory, manifest):
    plan = manifest["training"]
    planned = [lr_at(i, plan["scheduler_total_steps"], plan["learning_rate"],
                     plan["min_learning_rate"], plan["warmup_steps"]) for i in range(plan["scheduler_total_steps"])]
    steps = [row for row in records(run / "training-log.jsonl") if row["event"] == "train"]
    by_step = {row["global_step"]: row for row in steps}
    if list(by_step) != list(range(1, len(planned) + 1)) or len(steps) != len(planned):
        raise ValueError("complete per-update training log required for the learning-rate curve")
    val = {row["step"]: row["loss"] for row in read_json(run / "metrics.json")}

    with (directory / "loss-and-learning-rate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("step", "tokens_seen", "planned_learning_rate", "observed_learning_rate", "train_loss",
                         "val_loss", "gradient_norm", "samples_seen"))
        for step, expected in enumerate(planned, 1):
            row = by_step[step]
            if (row["learning_rate"], row["loss_tokens"], row["tokens_seen"], row["samples_seen"]) != (
                    expected, 8184, step * 8184, step * 8):
                raise ValueError(f"Step {step} training/LR evidence disagrees with the single planned schedule")
            writer.writerow((step, row["tokens_seen"], expected, row["learning_rate"],
                             row["loss"], val.get(step), row["gradient_norm"], row["samples_seen"]))


def _metrics(rows):
    diagnostics = generation_metrics(rows)
    content = sum(row["content_tokens"] for row in rows)
    return {**diagnostics,
            "han_output_fraction": diagnostics["han_outputs"] / len(rows),
            "eos_fraction": sum(row["stop_reason"] == "eos" for row in rows) / len(rows),
            "unexpected_special_token_fraction": sum(len(row["special_token_anomalies"])
                                                     for row in rows) / max(content, 1),
            "manual_review_pending": sum(row.get("score", {}).get("status") == "manual_review" for row in rows)}


def verify_int4_size(path):
    info = file_info(path)
    if info["bytes"] > 50_000_000:
        raise ValueError("final INT4 export exceeds 50,000,000 bytes")
    return info


def _comparison_table(directory, results):
    lines = ["# 2M language scaling comparison", "",
             "| Checkpoint | Targets | Common Train Loss | Val Loss | Test Loss | Greedy Han | Greedy 4-gram repeat | Auto |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in results.items():
        lm = row["lm"]
        greedy = row["generation"][0]["benchmark"]
        scores = greedy["scores"]["raw"]["categories"].values()
        correct, scored = sum(s["correct"] for s in scores), sum(s["scored"] for s in scores)
        test = f"{lm['test']['loss']:.4f}" if "test" in lm else "—"
        han = f"{greedy['han_character_fraction']:.2%}" if greedy["han_character_fraction"] is not None else "—"
        repeat = greedy["mean_token_4gram_repeat_ratio"]
        lines.append(f"| {name} | {row['tokens_seen']:,} | {lm['train']['loss']:.4f} | {lm['val']['loss']:.4f} | "
                     f"{test} | {han} | {repeat:.2%} | {correct}/{scored} |")
    lines += ["", "Human fluency and factual review remains pending. The historical 1M run used a different "
              "data-consumption order and LR schedule. Train Loss uses one common sample throughout."]
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_scaling(out, *, dataset=OUTPUT):
    # A separate process consumes immutable completed checkpoints. Evaluation
    # never runs inside fit and cannot advance its sampler, optimizer or RNG.
    policy = validate_v4_pilot_policy(dataset)
    if not policy["ok"]:
        raise ValueError("evaluation requires the admitted frozen 2M dataset: " + "; ".join(policy["errors"]))
    current = read_json(dataset / "dataset-manifest.json")
    if current["corpus_version"] != VERSION:
        raise ValueError("expected the 2M profile")
    original = verify_pinned_release(dataset)
    base_manifest("train.bin", "train.token-index.jsonl", "budget-report.json")
    if not torch.cuda.is_available():
        raise ValueError("matched 87M comparison uses the available CUDA device")
    torch.set_num_threads(1)
    device = torch.device("cuda")
    run = ROOT / read_json(TRAINING_CONFIG)["out"]
    nodes, audits, manifest, final_proof = _completed_nodes(run, dataset)
    historical = read_json(HISTORICAL.parent / "training-manifest.json")
    if historical["dataset"]["train"]["dataset_content_id"] != BASE_ID:
        raise ValueError("historical schedule is from a different 1M run")

    previous_greedy, historical_evaluation = load_historical_greedy(ROOT / HISTORICAL_FINAL)
    config = EvaluationConfig(**read_json(FORMAL_CONFIG))
    data, index, window_plan = evaluation_data(config)
    tokenizer = MiniLLMTokenizer.load(dataset / "tokenizer.json")
    if tokenizer.fingerprint != FINGERPRINT:
        raise ValueError("comparison tokenizer differs")
    vocab = vocabulary(tokenizer)
    benchmark = [{**task, "generation_mode": "raw"} for task in load_tasks(ROOT / BENCHMARK, scored=True)
                 if task["generation_mode"] in ("raw", "both")]
    probes = load_tasks(CONTINUATIONS, scored=False) + [
        {**row, "generation_mode": "raw", "text_category": "zh"} for row in read_json(PROBES)["probes"]]
    if (len(benchmark), len(probes)) != (300, 9):
        raise ValueError("unexpected formal benchmark or diagnostic asset size")
    results, logits = {}, {}
    with report_directory(out) as directory:
        write_json(directory / "common-lm-windows.json", window_plan)
        write_json(directory / "checkpoint-and-consumption.json", audits)
        write_json(directory / "final-checkpoint-consistency.json", final_proof)
        write_json(directory / "comparison-policy.json", {
            "config": read_json(FORMAL_CONFIG), "decoding": {name: STRATEGIES[name] for name in DECODE},
            "seeds": list(SEEDS), "repetition_penalty": 1.0, "batch_size": 24,
            "assets": {str(p): file_info(p) for p in (ROOT / BENCHMARK, CONTINUATIONS, PROBES)},
            "fixed_train_content_id": original["content_id"], "training_content_id": current["content_id"],
            "training_plan": manifest["training"], "historical_plan": historical["training"],
            "historical_limits": "different training samples and LR trajectory; the within-run comparison shares one schedule",
            "test_policy": "one predeclared full Test Loss at final FP32; never used for tuning",
            "human_review_criteria": list(HUMAN_CRITERIA),
        })
        _curves(run, directory, manifest)
        fp4 = directory / "step245-int4.pt"
        loop = [BASELINE, "step0", "step123", FINAL, "step245_int4"]
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for name in loop:
                path = fp4 if name == "step245_int4" else nodes[name]
                if name == FINAL:
                    verify_file(path, audits[FINAL]["checkpoint"]["files"]["model.pt"])
                model, payload = load_evaluation_model(path, tokenizer, device)
                if model.cfg.to_dict() != manifest["model_config"]:
                    raise ValueError("comparison model configuration differs")
                step, tokens = payload["step"], payload["tokens_seen"]
                if name == FINAL:
                    verify_file(path, audits[FINAL]["checkpoint"]["files"]["model.pt"])
                    save_quantized(fp4, model.cpu(), bits=4, group_size=128, tokenizer_fingerprint=FINGERPRINT,
                                   metadata=payload)
                    model.to(device)
                    verify_int4_size(fp4)
                del payload
                splits = ("train", "val", "test") if name == FINAL else ("train", "val")
                loss = {split: evaluate_indexed_windows(model, data[split], index[split], window_plan[split],
                                                       device, config.batch_size) for split in splits}
                traces, _, logits[name] = trace_predictions(model, tokenizer, probes, vocab, config)
                first = [{"id": row["id"], **row["steps"][0],
                          "teacher_forced_suffix_loss": row["teacher_forced_suffix_loss"]} for row in traces]
                experiments = []
                for strategy in DECODE:
                    for seed in (config.seed,) if strategy == "greedy" else SEEDS:
                        settings = {"checkpoint": name, "strategy": strategy, "seed": seed,
                                    "repetition_penalty": 1.0}
                        generation = replace(config, seed=seed)
                        scored = batch_generations(model, tokenizer, benchmark, generation, STRATEGIES[strategy],
                                                   1.0, batch_size=24, scored=True)
                        if name == BASELINE and strategy == "greedy":
                            proof = check_scaling_historical_greedy(benchmark, scored, previous_greedy,
                                                                   historical_evaluation, generation, STRATEGIES[strategy], 1.0)
                            write_json(directory / "historical-greedy-check.json", proof)
                        for row in scored:
                            if row["score"]["correct"] is None:
                                row["human_review"] = {"status": "PENDING", "criteria": list(HUMAN_CRITERIA)}
                        continuations = batch_generations(model, tokenizer, probes, generation, STRATEGIES[strategy],
                                                          1.0, batch_size=24)
                        _record_predictions(handle, scored + continuations, settings)
                        experiments.append({**settings, "benchmark": _metrics(scored),
                                            "continuations": _metrics(continuations)})
                results[name] = {"checkpoint": file_info(path), "step": step, "tokens_seen": tokens,
                                 "lm": loss, "first_token": first, "generation": experiments}
                del model
        write_json(directory / "comparison.json", results)
        _comparison_table(directory, results)
        _quantized_report(directory, results, logits)
        publish_manifest(directory, "manifest.json", {"schema_version": 1,
                         "status": "EVALUATED", "environment": environment(device)})


def _quantized_report(directory, results, logits, *, final=FINAL, quantized="step245_int4"):
    rows = {}
    for row in records(directory / "predictions.jsonl"):
        if row["checkpoint"] in (final, quantized):
            rows.setdefault(row["checkpoint"], {})[(row["id"], row["strategy"], row["seed"])] = row["generated_token_ids"]
    fp, int4 = rows[final], rows[quantized]
    if fp.keys() != int4.keys():
        raise ValueError("FP32 and INT4 prediction cohort differs")
    write_json(directory / "fp32-int4.json", {
        "runtime": "INT4 weights are dequantized to FP32; no native low-bit acceleration measured",
        "checkpoint": results[quantized]["checkpoint"], "group_size": 128,
        "logits": quantization_comparison(logits[final], logits[quantized]),
        "paired_generation": [{"id": task, "strategy": strategy, "seed": seed,
                               "exact_tokens": fp[task, strategy, seed] == int4[task, strategy, seed]}
                              for task, strategy, seed in fp],
        "fp32_val_loss": results[final]["lm"]["val"]["loss"],
        "int4_val_loss": results[quantized]["lm"]["val"]["loss"],
    })
