"""Run read-only Pilot diagnostics and isolated 87M memorization."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .assessment_io import environment, publish_manifest, read_json, records, report_directory, verified_evaluation
from .corpus_utils import file_info
from .corpus_v4 import verify_frozen
from .evaluation import EvaluationConfig, evaluate_batches, fixed_window_batches, load_evaluation_model, load_tasks
from .evaluation_contamination import contamination_audit
from .evaluation_windows import evaluate_indexed_windows, indexed_train_plan
from .language_correctness import verify_attention, verify_shift, verify_training_fixture
from .language_generation_analysis import (SEEDS, STRATEGIES, batch_generations, choose_sampling_strategy,
                                           evaluate_sweep, generation_metrics, quantization_comparison,
                                           trace_predictions, verify_sampled_eos)
from .language_memorization import run_memorization
from .language_token_analysis import expansion_budget, token_distributions, vocabulary
from .quantize import quantize_model
from .text_policy import checkpoint_text_policy
from .tokenizer import MiniLLMTokenizer
from .training_manifest import write_json
from .utils import get_device
from .v4_language_pilot import BENCHMARK, PROJECT_ROOT
from .v4_training_accounting import category_consumption, replay_trained_windows

CONTINUATIONS = PROJECT_ROOT / "data/evaluation/chinese-language-continuations-v1/prompts.jsonl"
PROBES = PROJECT_ROOT / "configs/language-degeneration-probes-v1.json"
FORMAL_CONFIG = PROJECT_ROOT / "configs/evaluation.v4-language-pilot.json"
REPORT = Path("reports/v4-language-degeneration-diagnostics")
RUN = Path("runs/v4-language-pilot-first-training")
DATASET = Path("artifacts/corpus/v4-language-pilot-first-training-reviewed")
WORKSPACE = Path("artifacts/v4-language-pilot-first-training/workspace")
HISTORICAL_FINAL = Path("reports/v4-language-pilot-first-training/execute/final/suite-0")


def _equal_payload(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _equal_payload(left[key], right[key]) for key in left)
    return left == right


def verify_checkpoint_lineage(assets, identities, state, tokenizer):
    """Bind the initial and final FP32 models and regenerate the serialized INT4 payload."""
    for name, step, tokens in (("step0", 0, 0), ("step123", 123, state["tokens_seen"]),
                               ("int4", 123, state["tokens_seen"])):
        payload = torch.load(assets[name], map_location="cpu", weights_only=False, mmap=True)
        if (payload["config"] != state["config"] or payload["tokenizer_fingerprint"] != tokenizer.fingerprint
                or checkpoint_text_policy(payload) != tokenizer.text_policy
                or (payload.get("step"), payload.get("tokens_seen")) != (step, tokens)):
            raise ValueError(f"{name} does not match the Pilot checkpoint metadata")
        if name == "step0":
            changed = {"tensors": sum(not torch.equal(tensor, state["model"][key])
                                      for key, tensor in payload["model"].items()),
                       "tensors_checked": len(payload["model"]),
                       "embedding_max_absolute_delta": float((payload["model"]["embed_tokens.weight"] -
                                                             state["model"]["embed_tokens.weight"]).abs().max())}
            if changed["tensors"] == 0:
                raise ValueError("Pilot optimizer did not change any checkpoint weights")
        elif name == "step123":
            if (payload["pilot_initialization"]["sha256"] != identities["step0"]["sha256"]
                    or not _equal_payload(payload["model"], state["model"])):
                raise ValueError("final FP32 does not match both the initial lineage and committed state")
            model, _ = load_evaluation_model(assets[name], tokenizer, torch.device("cpu"))
            exact_quantization = quantize_model(model, bits=4, group_size=128)
            del model, _
        elif (identities["int4"]["bytes"] > 50_000_000 or payload["quantized_bits"] != 4
              or not _equal_payload(exact_quantization, {key: payload[key] for key in exact_quantization})):
            raise ValueError("saved INT4 does not reproduce the FP32 export or exceeds its budget")
    return changed


def _record_predictions(handle, rows, experiment):
    for row in rows:
        handle.write(json.dumps({**experiment, **row}, ensure_ascii=False, allow_nan=False) + "\n")


def _tokenizer_validation(tokenizer, documents):
    examples = [
        ("zh", "我把杯子放在桌上，等水凉一点再喝。"),
        ("zh", "這間圖書館星期六開門，歡迎大家來讀書。"),
        ("zh", "，。！？；：「」、《》\n"),
        ("mixed", "明天 9:30 在 A2 教室集合，带上 USB 线。"),
        ("zh", "春天  火车站  学习  我们一起去公园。"),
    ]
    special = set(tokenizer.special_ids.values())
    rows = []
    for category, text in examples:
        ids = tokenizer.encode(text, category=category)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        doc = tokenizer.encode_document(text, category=category)
        rows.append({"category": category, "input": text, "token_ids": ids,
                     "token_texts": [tokenizer.decode([token]) for token in ids], "decoded": decoded,
                     "matches_text_policy": decoded == tokenizer.prepare_text(text, category=category),
                     "unexpected_special_ids": sorted(special.intersection(ids)),
                     "document_appended_one_eos": doc == ids + [tokenizer.special_ids["eos"]]})
    return {"fingerprint": tokenizer.fingerprint, "vocab_size": tokenizer.vocab_size,
            "text_policy": tokenizer.text_policy, "special_ids": tokenizer.special_ids,
            "cases": rows, "frozen_train_documents": documents}


def run_pilot_diagnostics(out, dataset, run, workspace, historical, device, benchmark_batch_size):
    out, dataset, run, workspace, historical = map(Path, (out, dataset, run, workspace, historical))
    initial, fp32, int4 = workspace / "initial.pt", run / "model.pt", run / "model-int4.pt"
    tokenizer_path = run / "tokenizer.json"
    assets = {"step0": initial, "step123": fp32, "int4": int4, "tokenizer": tokenizer_path,
              "state": run / "state.pt", "frozen_dataset": dataset / "dataset-manifest.json",
              "pilot_plan": workspace / "pilot-plan.json", "historical_evaluation": historical / "evaluation-manifest.json",
              "benchmark": PROJECT_ROOT / BENCHMARK, "continuations": CONTINUATIONS,
              "diagnostic_probes": PROBES, "formal_config": FORMAL_CONFIG}
    with report_directory(out) as directory:
        inputs = {name: file_info(path) for name, path in assets.items()}
        print("Verifying frozen corpus and Pilot lineage", flush=True)
        frozen = verify_frozen(dataset)
        state, _, data, windows = replay_trained_windows(
            dataset, run, read_json(run / "train-summary.json"), frozen["content_id"])
        checkpoint_records = verified_evaluation(historical)[0]
        if (checkpoint_records["checkpoint"]["sha256"] != inputs["step123"]["sha256"]
                or checkpoint_records["dataset"]["dataset_content_id"] != frozen["content_id"]
                or read_json(assets["pilot_plan"])["initial_checkpoint"] != inputs["step0"]
                or state["global_step"] != 123):
            raise ValueError("historical evaluation, workspace and frozen corpus do not identify this Pilot")
        tokenizer = MiniLLMTokenizer.load(tokenizer_path)
        if tokenizer.fingerprint != frozen["tokenizer"]["fingerprint"] or tokenizer.vocab_size != 24576:
            raise ValueError("expected the real 24,576-token frozen Pilot tokenizer")
        tokenizer.text_policy = checkpoint_text_policy(state)
        changed = verify_checkpoint_lineage(assets, inputs, state, tokenizer)
        vocab = vocabulary(tokenizer)
        config = EvaluationConfig(**read_json(FORMAL_CONFIG))
        fixed = load_tasks(CONTINUATIONS, scored=False)
        extra = [{**task, "generation_mode": "raw", "text_category": "zh"} for task in read_json(PROBES)["probes"]]
        tasks = fixed + extra
        benchmark = [{**task, "generation_mode": "raw"} for task in load_tasks(PROJECT_ROOT / BENCHMARK, scored=True)
                     if task["generation_mode"] in ("raw", "both")]
        if len(benchmark) != 300 or len(fixed) != 5:
            raise ValueError("unexpected language evaluation asset size")
        train_index, train_windows = indexed_train_plan(dataset, data, seed=config.seed, token_budget=config.token_budget)
        write_json(directory / "train-windows.json", train_windows)
        consumed = category_consumption(train_index, data.seq_len, windows)
        if consumed["effective_tokens"] != state["tokens_seen"] or len(windows) != state["samples_seen"]:
            raise ValueError("replayed Train windows disagree with optimizer state")
        write_json(directory / "consumed-windows.json", {"indices": windows, **consumed})
        distributions, boundaries, order = token_distributions(dataset, data, train_index, windows, tokenizer, vocab)
        for category, summary in consumed["categories"].items():
            measured = distributions["consumed_supervised"]["by_category"].get(category, {}).get("tokens", 0)
            if measured != summary["consumed_effective_tokens"]:
                raise ValueError("observed Token totals do not reproduce training consumption")
        write_json(directory / "token-distributions.json", distributions)
        tokenizer_evidence = _tokenizer_validation(tokenizer, boundaries)
        write_json(directory / "tokenizer-validation.json", tokenizer_evidence)
        ratios = read_json(dataset / "budget-report.json")
        training_plan = state["manifest"]["training"]
        budget = expansion_budget(order, ratios["category_token_ratios"], data.seq_len,
                                  batch_size=training_plan["batch_size"], grad_accum=training_plan["grad_accum"],
                                  seed=training_plan["sampler_seed"],
                                  tolerance=ratios["fraction_tolerance"])
        write_json(directory / "expansion-budget.json", budget)
        contamination = contamination_audit([], dataset=dataset, fixed_prompts=tasks)
        write_json(directory / "continuation-contamination.json",
                   {**contamination, "asset_hashes": {key: inputs[key] for key in ("continuations", "diagnostic_probes")},
                    "scope": "DIAGNOSTIC_ONLY evaluation audit; frozen release policy stays intact"})

        software = verify_training_fixture()
        print("Running Step 0, Step 123 and INT4 inference", flush=True)
        inference, logits_by_model, sweep_results, full_results = {}, {}, [], []
        real_eos_prompt = next((row["prompt_token_ids"] for row in records(historical / "predictions.jsonl")
                                if row.get("stop_reason") == "eos"), None)
        for name in ("step0", "step123", "int4"):
            model, payload = load_evaluation_model(assets[name], tokenizer, device)
            step, tokens = payload["step"], payload["tokens_seen"]
            if model.num_parameters() != 86_528_640:
                raise ValueError(f"{name} does not have the expected 87M parameter count")
            del payload
            traces, arrays, logits_by_model[name] = trace_predictions(model, tokenizer, tasks, vocab, config)
            write_json(directory / f"probability-traces-{name}.json", traces)
            np.savez_compressed(directory / f"probability-arrays-{name}.npz", **arrays)
            inference[name] = {"step": step, "tokens_seen": tokens, "parameters": model.num_parameters(),
                               "config": model.cfg.to_dict(),
                               "first_token": [{"id": row["id"], "top_1": row["steps"][0]["top_20"][0],
                                                **{key: row["steps"][0][key] for key in (
                                                    "punctuation_probability", "entropy_nats", "target",
                                                    "newline_whitespace_probability")},
                                                "teacher_forced_suffix_loss": row["teacher_forced_suffix_loss"]}
                                               for row in traces]}
            if name != "int4":
                inference[name]["representative_train"] = evaluate_indexed_windows(
                    model, data, train_index, train_windows, device, config.batch_size)
            if name == "step123":
                ids = traces[0]["prompt_ids"]
                inference[name]["attention"] = verify_attention(model, ids, eos_prompt=real_eos_prompt)
                if real_eos_prompt is None:
                    inference[name]["attention"]["sampled_eos_stop"] = verify_sampled_eos(
                        model, [row["prompt_ids"] for row in traces])
                inference[name]["label_shift"] = verify_shift(model, data[train_windows["window_indices"][0]]["input_ids"].tolist())
                inference[name]["historical_prefix_train_loss"] = evaluate_batches(
                    model, fixed_window_batches(data, config.batch_size), device, config.token_budget)
            model_rows = []
            with (directory / f"decode-sweep-{name}.jsonl").open("w", encoding="utf-8") as handle:
                for experiment, predictions in evaluate_sweep(model, tokenizer, tasks, max_new_tokens=32):
                    _record_predictions(handle, predictions, experiment)
                    model_rows.append({"model": name, **experiment, **generation_metrics(predictions)})
            sweep_results.extend(model_rows)
            print(f"{name} fixed-prefix decoding complete", flush=True)
            if name == "step123":
                chosen = choose_sampling_strategy(model_rows)
                print(f"Full 300-task raw benchmark: greedy and {chosen} (seeds {SEEDS})", flush=True)
                with (directory / "full-language-predictions.jsonl").open("w", encoding="utf-8") as handle:
                    for strategy, seeds in (("greedy", (SEEDS[0],)), (chosen, SEEDS)):
                        for seed in seeds:
                            experiment = {"strategy": strategy, "seed": seed, "parameters": STRATEGIES[strategy],
                                          "repetition_penalty": 1.0}
                            predictions = batch_generations(
                                model, tokenizer, benchmark,
                                EvaluationConfig(**(read_json(FORMAL_CONFIG) | {"seed": seed})),
                                STRATEGIES[strategy], 1.0, batch_size=benchmark_batch_size, scored=True)
                            _record_predictions(handle, predictions, experiment)
                            full_results.append({**experiment, **generation_metrics(predictions)})
            del model

        compared = quantization_comparison(logits_by_model["step123"], logits_by_model["int4"])
        sweep_fp32 = list(records(directory / "decode-sweep-step123.jsonl"))
        sweep_int4 = list(records(directory / "decode-sweep-int4.jsonl"))
        quantization_generation = []
        for left, right in zip(sweep_fp32, sweep_int4, strict=True):
            if any(left[key] != right[key] for key in ("strategy", "seed", "repetition_penalty", "id")):
                raise ValueError("quantization sweep settings do not align")
            a, b = left["generated_token_ids"], right["generated_token_ids"]
            divergence = next((pos + 1 for pos, (x, y) in enumerate(zip(a, b)) if x != y), None)
            if divergence is None and len(a) != len(b):
                divergence = min(len(a), len(b)) + 1
            quantization_generation.append({"strategy": left["strategy"], "seed": left["seed"],
                                            "penalty": left["repetition_penalty"], "id": left["id"],
                                            "first_different_token": divergence})
        write_json(directory / "quantization.json", {"int4_bitwise_payload_matches_fp32_cpu_export": True,
                   "execution": "dequantized INT4 weights in FP32 runtime", "first_logits": compared,
                   "paired_generation": quantization_generation})
        write_json(directory / "decoding-metrics.json", {"fixed_max_new_tokens": 32,
                   "fixed_batch_size": 16, "benchmark_batch_size": benchmark_batch_size,
                   "seed_policy": "one generator per seed and experiment; fixed ordered batches grouped by output budget",
                   "penalty_policy": "sign-aware, applied once per distinct visible prompt or generated token",
                   "sweep": sweep_results, "benchmark": full_results, "selected_strategy": chosen,
                   "selection": "unpenalized Step 123 fixed probes: most outputs with Han, then lowest mean 4-gram repetition"})
        write_json(directory / "correctness.json", {"software": software, "real_step123": {
            "attention": inference["step123"]["attention"], "label_shift": inference["step123"]["label_shift"],
            "step0_to_step123_parameters": changed, "committed_state_matches_export": True,
            "consumed_supervised_tokens": consumed["effective_tokens"],
            "optimizer_updates": state["global_step"]}})
        write_json(directory / "models.json", {"frozen_content_id": frozen["content_id"],
                   "tokenizer_fingerprint": tokenizer.fingerprint, "inputs": inputs, "models": inference,
                   "historical_formal": {key: checkpoint_records[key] for key in
                                         ("train_language_model", "language_model", "scores", "diagnostics")}})
        for key, expected in inputs.items():
            if file_info(assets[key]) != expected:
                raise ValueError(f"diagnostic input changed during analysis: {key}")
        publish_manifest(directory, "manifest.json", {"schema_version": 1, "scope": "DIAGNOSTIC_ONLY",
                         "environment": environment(device), "inputs": inputs})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pilot = commands.add_parser("pilot", help="read-only PR #16 artifact analysis")
    pilot.add_argument("--dataset", type=Path, default=DATASET)
    pilot.add_argument("--run", type=Path, default=RUN)
    pilot.add_argument("--workspace", type=Path, default=WORKSPACE)
    pilot.add_argument("--historical-final", type=Path, default=HISTORICAL_FINAL)
    pilot.add_argument("--benchmark-batch-size", type=int, default=24)
    pilot.add_argument("--out", type=Path, default=REPORT / "pilot")
    memory = commands.add_parser("memorize", help="explicitly run bounded DIAGNOSTIC_ONLY training")
    memory.add_argument("--tokenizer", type=Path, default=RUN / "tokenizer.json")
    memory.add_argument("--run-dir", type=Path, default=Path("runs/v4-language-degeneration-memorization"))
    memory.add_argument("--out", type=Path, default=REPORT / "memorization")
    memory.add_argument("--steps", type=int, default=500)
    for command in (pilot, memory):
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads <= 0 or (args.command == "pilot" and args.benchmark_batch_size <= 0):
        parser.error("threads and batch size must be positive")
    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    if args.command == "pilot":
        run_pilot_diagnostics(args.out, args.dataset, args.run, args.workspace, args.historical_final,
                              device, args.benchmark_batch_size)
    else:
        run_memorization(args.tokenizer, args.out, args.run_dir, device=device, steps=args.steps)
    return 0
