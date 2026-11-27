"""Paired V5 milestone evaluation using the historical, fixed evaluation protocol."""

from dataclasses import asdict, replace
from pathlib import Path
import shutil

import torch

from . import v5_language_training as profile
from .assessment_io import environment, publish_manifest, read_json, records, report_directory, write_json
from .corpus_budget import read_token_index
from .corpus_utils import file_info, verify_file
from .data import PackedPretrainDataset
from .evaluation import EvaluationConfig, load_evaluation_model, load_tasks
from .evaluation_windows import evaluate_indexed_windows, representative_windows
from .language_diagnostics import (BENCHMARK, CONTINUATIONS, HISTORICAL_FINAL, PROBES,
                                   _record_predictions, verify_historical_greedy)
from .language_generation_analysis import SEEDS, STRATEGIES, batch_generations, trace_predictions
from .language_scaling_evaluation import (BASELINE, DECODE, HUMAN_CRITERIA, _completed_nodes, _curves,
                                           _metrics, _quantized_report, check_scaling_historical_greedy,
                                           evaluation_data, load_historical_greedy, verify_int4_size)
from .language_token_analysis import vocabulary
from .quantize import save_quantized
from .v4_admission import validate_v4_pilot_policy
from .v4_language_scaling import BASE_ID, FINGERPRINT, SCALING_CONTENT_ID, base_manifest
from .v5_language_corpus import sampler_budget

HISTORICAL_2M = profile.ROOT / "runs/v4-language-pilot-2m-v1/milestones/step-245/model.pt"
HISTORICAL_2M_FILE = {"bytes": 409075993, "sha256": "a390b51c6fc4a140a4040f079d7c396ac38b579fa1b4a1403a2a8f14efdc5a76"}
HISTORICAL_REPORT = profile.ROOT / "reports/v4-language-pilot-2m-v1/evaluate"
HISTORICAL_RECEIPT = {"bytes": 1718, "sha256": "65bc9ae43f66d1032cafe86613c77d8cd13f2df50f1e425d996b464351e3e0d2"}
HISTORICAL_LABEL = "historical_2m"
FINAL = "step1222"
INT4 = "step1222_int4"
SAMPLED = {HISTORICAL_LABEL, "step245", "step489", FINAL, INT4}


def _verify_report(directory, expected=None):
    if expected:
        verify_file(directory / "manifest.json", expected)
    manifest = read_json(directory / "manifest.json")
    for name, info in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("evaluation evidence requires local filenames")
        verify_file(directory / name, info)
    return manifest


def historical_2m(config, assets):
    verify_file(HISTORICAL_2M, HISTORICAL_2M_FILE)
    manifest = _verify_report(HISTORICAL_REPORT, HISTORICAL_RECEIPT)
    policy = read_json(HISTORICAL_REPORT / "comparison-policy.json")
    comparison = read_json(HISTORICAL_REPORT / "comparison.json")["step245"]
    if (manifest["status"] != "EVALUATED" or EvaluationConfig(**policy["config"]) != config
            or policy["fixed_train_content_id"] != BASE_ID or policy["training_content_id"] != SCALING_CONTENT_ID
            or policy["assets"] != assets or comparison["checkpoint"] != HISTORICAL_2M_FILE
            or (comparison["step"], comparison["tokens_seen"]) != (245, 2_005_080)):
        raise ValueError("saved historical 2M comparison differs from the fixed protocol")
    rows = [row for row in records(HISTORICAL_REPORT / "predictions.jsonl")
            if row["checkpoint"] == "step245" and row["strategy"] == "greedy" and row["kind"] == "benchmark"]
    generation = asdict(config)
    return rows, {"generation": generation, "comparison_policy": {**policy, "config": generation},
                  "files": manifest["files"]}


def _all_windows(path, config):
    data = PackedPretrainDataset(path, config.seq_len)
    index = read_token_index([path.with_name(path.stem + ".token-index.jsonl")])["records"]
    plan = representative_windows(index, data, seed=config.seed, token_budget=0)
    plan.update(bin=file_info(path), token_index=file_info(path.with_name(path.stem + ".token-index.jsonl")))
    return data, index, plan


def _generation_metrics(rows):
    result = _metrics(rows)
    chars = sum(len(row["decoded_text"]) for row in rows)
    whitespace = sum(sum(character.isspace() for character in row["decoded_text"]) for row in rows)
    abnormal = sum(row["diagnostics"]["abnormal_characters"] for row in rows)
    result.update(whitespace_character_fraction=whitespace / max(chars, 1),
                  valid_nonspace_character_fraction=(chars - whitespace - abnormal) / max(chars, 1),
                  sentence_ending_fraction=sum(row["diagnostics"]["ends_with_sentence_punctuation"]
                                               for row in rows) / len(rows))
    return result


def _final_test(output, receipt, context, model=None):
    """Publish the held-out measurement separately so a resumed node can reuse it."""
    if not output.exists():
        if model is None:
            raise ValueError("final evaluation lacks its held-out measurement")
        config = context["config"]
        data, index, windows = _all_windows(profile.DATASET / "test.bin", config)
        with report_directory(output) as directory:
            write_json(directory / "windows.json", windows)
            loss = evaluate_indexed_windows(model, data, index, windows, context["device"], config.batch_size)
            write_json(directory / "loss.json", loss)
            publish_manifest(directory, "manifest.json", {"status": "EVALUATED_FINAL_FP32", "identity": receipt})
    evidence = _verify_report(output)
    if evidence["identity"] != receipt or evidence["status"] != "EVALUATED_FINAL_FP32":
        raise ValueError("V5 Test measurement differs from the final FP32 checkpoint or protocol")
    return read_json(output / "loss.json"), file_info(output / "manifest.json")


def _evaluate_node(destination, name, path, context):
    config, tokenizer, device = (context[key] for key in ("config", "tokenizer", "device"))
    receipt = {"checkpoint": file_info(path), "protocol": context["protocol"], "node": name}
    if destination.exists():
        existing = _verify_report(destination)
        if existing["identity"] != receipt or existing["status"] != "EVALUATED":
            raise ValueError(f"existing node differs from the requested evaluation: {name}")
        result = read_json(destination / "result.json")
        if name == FINAL:
            loss, test_file = _final_test(context["final_test_output"], receipt, context)
            if loss != result["lm"]["v5_test"] or test_file != existing["final_test_manifest"]:
                raise ValueError("final comparison differs from its held-out measurement")
        return result

    model, payload = load_evaluation_model(path, tokenizer, device)
    if model.cfg.to_dict() != context["model_config"]:
        raise ValueError("evaluated model architecture differs")
    step, tokens = payload["step"], payload["tokens_seen"]
    del payload
    with report_directory(destination) as directory:
        splits = {"train": context["common"]["train"], "val": context["common"]["val"],
                  "test": context["common"]["test"], "v5_val": context["v5_val"]}
        losses = {key: evaluate_indexed_windows(model, *window_data, device, config.batch_size)
                  for key, window_data in splits.items()}
        trace, _, logits = trace_predictions(model, tokenizer, context["probes"], context["vocab"], config)
        torch.save(logits, directory / "next-token-logits.pt")
        write_json(directory / "next-token-traces.json", trace)
        checked = {}
        experiments = []
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for strategy in (DECODE if name in SAMPLED else ("greedy",)):
                for seed in ((config.seed,) if strategy == "greedy" else SEEDS):
                    generation = replace(config, seed=seed)
                    settings = {"checkpoint": name, "strategy": strategy, "seed": seed, "repetition_penalty": 1.0}
                    benchmark = context["benchmark"]
                    scored = batch_generations(model, tokenizer, benchmark, generation, STRATEGIES[strategy],
                                               1.0, batch_size=24, scored=True)
                    if strategy == "greedy" and name in context["historical"]:
                        previous, manifest = context["historical"][name]
                        check = check_scaling_historical_greedy if name == BASELINE else verify_historical_greedy
                        checked[name] = check(benchmark, scored, previous, manifest, generation, STRATEGIES[strategy], 1.0)
                    continued = batch_generations(model, tokenizer, context["probes"], generation, STRATEGIES[strategy],
                                                  1.0, batch_size=24)
                    for row in (*scored, *continued):
                        if row.get("score", {}).get("correct") is None:
                            row["human_review"] = {"status": "PENDING", "criteria": list(HUMAN_CRITERIA)}
                    _record_predictions(handle, scored + continued, settings)
                    experiments.append({**settings, "benchmark": _generation_metrics(scored),
                                        "continuations": _generation_metrics(continued)})
        evidence = {"status": "EVALUATED", "identity": receipt}
        if name == FINAL:
            losses["v5_test"], evidence["final_test_manifest"] = _final_test(
                context["final_test_output"], receipt, context, model)
        result = {"checkpoint": receipt["checkpoint"], "step": step, "tokens_seen": tokens,
                  "lm": losses, "generation": experiments, "historical_greedy": checked}
        write_json(directory / "result.json", result)
        publish_manifest(directory, "manifest.json", evidence)
    return result


def _quantize(path, output, tokenizer, device, final_proof):
    if output.exists():
        receipt = _verify_report(output)
        if receipt["source"] != final_proof:
            raise ValueError("existing INT4 export belongs to a different final checkpoint")
        return output / "model-int4.pt"
    with report_directory(output) as directory:
        model, payload = load_evaluation_model(path, tokenizer, device)
        target = directory / "model-int4.pt"
        save_quantized(target, model.cpu(), bits=4, group_size=128, tokenizer_fingerprint=FINGERPRINT, metadata=payload)
        verify_int4_size(target)
        publish_manifest(directory, "manifest.json", {"status": "EXPORTED", "source": final_proof, "group_size": 128})
    return output / "model-int4.pt"


def evaluate():
    from .tokenizer import MiniLLMTokenizer

    if not torch.cuda.is_available():
        raise ValueError("matched comparison requires CUDA")
    device = torch.device("cuda")
    torch.set_num_threads(1)
    admission = validate_v4_pilot_policy(profile.DATASET)
    if not admission["ok"] or admission["content_id"] != profile.CONTENT_ID:
        raise ValueError(f"evaluation requires the admitted frozen 10M release: {admission['errors']}")
    base_manifest(*(f"{split}.{kind}" for split in ("train", "val", "test")
                    for kind in ("bin", "token-index.jsonl")))
    nodes, audits, manifest, final_proof = _completed_nodes(
        profile.RUN, profile.DATASET, steps=profile.MILESTONES, version=profile.V5_VERSION,
        training_config=profile.TRAINING_CONFIG)
    nodes[HISTORICAL_LABEL] = HISTORICAL_2M
    config = EvaluationConfig(**read_json(profile.EVALUATION_CONFIG))
    assets = {str(path): file_info(path) for path in (profile.ROOT / BENCHMARK, CONTINUATIONS, PROBES)}
    historical = {BASELINE: load_historical_greedy(profile.ROOT / HISTORICAL_FINAL),
                  HISTORICAL_LABEL: historical_2m(config, assets)}
    benchmark = [{**task, "generation_mode": "raw"} for task in load_tasks(profile.ROOT / BENCHMARK, scored=True)
                 if task["generation_mode"] in ("raw", "both")]
    probes = load_tasks(CONTINUATIONS, scored=False) + [
        {**row, "generation_mode": "raw", "text_category": "zh"} for row in read_json(PROBES)["probes"]]
    if (len(benchmark), len(probes)) != (300, 9):
        raise ValueError("unexpected fixed evaluation population")
    data, indices, windows = evaluation_data(config)
    v5_val = _all_windows(profile.DATASET / "val.bin", config)
    tokenizer = MiniLLMTokenizer.load(profile.DATASET / "tokenizer.json")
    if tokenizer.fingerprint != FINGERPRINT or tokenizer.vocab_size != 24576:
        raise ValueError("unexpected evaluation tokenizer")
    implementation = {name: file_info(Path(__file__).with_name(name)) for name in (
        "v5_language_evaluation.py", "language_scaling_evaluation.py", "evaluation.py",
        "evaluation_windows.py", "language_diagnostics.py", "language_generation_analysis.py",
        "evaluation_scoring.py", "quantize.py", "model.py")}
    protocol = {"config": asdict(config), "implementation": implementation, "assets": assets, "windows": windows, "v5_val": v5_val[2],
                "training_content_id": profile.CONTENT_ID, "seeds": list(SEEDS),
                "strategies": {name: STRATEGIES[name] for name in DECODE}, "batch_size": 24, "repetition_penalty": 1.0}
    context = {"config": config, "tokenizer": tokenizer, "device": device, "protocol": protocol,
               "model_config": manifest["model_config"], "benchmark": benchmark, "probes": probes,
               "vocab": vocabulary(tokenizer), "historical": historical, "v5_val": v5_val,
               "common": {key: (data[key], indices[key], plan) for key, plan in windows.items()}}
    output = profile.REPORT / "evaluation"
    context["final_test_output"] = output / "final-v5-test"
    for name in (BASELINE, HISTORICAL_LABEL, *(f"step{step}" for step in profile.MILESTONES)):
        print(f"[V5 evaluation] {name}", flush=True)
        _evaluate_node(output / "nodes" / name, name, nodes[name], context)
    nodes[INT4] = _quantize(nodes[FINAL], output / "int4", tokenizer, device, final_proof)
    _evaluate_node(output / "nodes" / INT4, INT4, nodes[INT4], context)
    _aggregate(output, [BASELINE, HISTORICAL_LABEL, *(f"step{step}" for step in profile.MILESTONES), INT4],
               protocol, audits, manifest, final_proof, device)


def _aggregate(output, names, protocol, audits, manifest, final_proof, device):
    from .v5_language_training_report import plot_results

    nodes = {name: output / "nodes" / name for name in names}
    results = {name: read_json(path / "result.json") for name, path in nodes.items()}
    destination = output / "comparison"
    node_manifests = {name: file_info(node / "manifest.json") for name, node in nodes.items()}
    if destination.exists():
        if _verify_report(destination)["node_manifests"] != node_manifests:
            raise ValueError("published comparison belongs to different evaluation nodes")
        return
    actual = sampler_budget(profile.DATASET, read_json(profile.DATASET / ".v4/plan.json"))
    expected = actual["budgets"]["10M"]
    consumption = audits[FINAL]["consumption"]
    if consumption != expected["consumption"] or expected["status"] != "FEASIBLE":
        raise ValueError("committed final sampler does not match the frozen budget")
    with report_directory(destination) as directory:
        write_json(directory / "comparison.json", results)
        write_json(directory / "comparison-policy.json", protocol)
        write_json(directory / "checkpoint-and-consumption.json", audits)
        write_json(directory / "actual-source-consumption.json", expected["sources"])
        write_json(directory / "final-checkpoint-consistency.json", final_proof)
        _curves(profile.RUN, directory, manifest)
        with (directory / "predictions.jsonl").open("wb") as handle:
            for node in nodes.values():
                with (node / "predictions.jsonl").open("rb") as predictions:
                    shutil.copyfileobj(predictions, handle)
        logits = {name: torch.load(nodes[name] / "next-token-logits.pt", weights_only=True)
                  for name in (FINAL, INT4)}
        _quantized_report(directory, results, logits, final=FINAL, quantized=INT4)
        paired = read_json(directory / "fp32-int4.json")
        paired["paired_loss_and_perplexity"] = {
            split: {label: {key: results[name]["lm"][split][key] for key in ("loss", "perplexity")}
                    for label, name in (("fp32", FINAL), ("int4", INT4))}
            for split in ("train", "val", "test", "v5_val")}
        write_json(directory / "fp32-int4.json", paired)
        plot_results(directory, results)
        publish_manifest(directory, "manifest.json", {"status": "EVALUATED", "environment": environment(device),
                         "node_manifests": node_manifests})
