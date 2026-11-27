"""Paired continuation, chat, and recognition evaluations for PR #22."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math
from pathlib import Path
import re

import torch

from .assessment_io import environment, publish_manifest, read_json, records, report_directory
from .checkpoint import load_training_state
from .corpus_utils import file_info, verify_file
from .evaluation import EvaluationConfig, load_evaluation_model, repetition
from .evaluation_candidates import _rank, _score_tail
from .evaluation_scoring import score_answer
from .evaluation_windows import evaluate_indexed_windows
from .generate import stop_ids_for, truncate_at_stop
from .language_generation_analysis import STRATEGIES
from .language_generation_analysis import SEEDS, batch_generations
from .language_scaling_evaluation import DECODE
from .quantization_provenance import verify_quantized_tensors
from .quantize import BUDGET_BYTES, save_quantized
from .tokenizer import ASSISTANT_ID, END_OF_TURN_ID, EOS_ID, SYSTEM_ID, USER_ID
from .train import evaluate
from .training_manifest import implementation_identity, write_json
from .training_milestones import require_identical_weights, verify_milestone, verify_model_metadata
from .v5_language_evaluation import _all_windows, _generation_metrics
from .v5_language_training import DATASET as PRETRAIN_DATASET, EVALUATION_CONFIG
from .v5_sft_data import (BENCHMARK, CHECKS, CONFIG, ROOT, StrictChatDataset, check_overlap,
                          frozen_manifest, historical_tasks, normalized, validate_messages, write_rows)

KNOWLEDGE = ROOT / "configs/v5-sft-knowledge-probes.json"
CHAT_DECODE = (("greedy", 42), *(("standard_sampling", seed) for seed in SEEDS))
BUDGET = 96
RUBRIC = ("relevance", "fluency", "completeness", "fact_accuracy", "instruction_following", "context_understanding")


def answer_index(rows, width=12):
    index = defaultdict(set)
    for row in rows:
        for turn in row["messages"]:
            if turn["role"] != "assistant":
                continue
            text = normalized(turn["content"])
            for i in range(len(text) - width + 1):
                index[text[i:i+width]].add((row["id"], text))
    return index


def match_answer(text, index, width=12):
    query = normalized(text)
    best = (0, "")
    for i in range(len(query) - width + 1):
        prefix = query[i:i+width]
        for row_id, answer in index.get(prefix, ()):
            j = answer.find(prefix)
            while j != -1:
                n = width
                while i + n < len(query) and j + n < len(answer) and query[i+n] == answer[j+n]:
                    n += 1
                best = max(best, (n, row_id))
                j = answer.find(prefix, j + 1)
    return {"matched_chars": best[0], "train_id": best[1] or None}


def _generate(model, tokenizer, ids, strategy, generator, *, raw=False, budget=BUDGET):
    stops = stop_ids_for(raw)
    device = next(model.parameters()).device
    inputs = torch.tensor([ids], device=device)
    generated = model.generate(inputs, max_new_tokens=budget, generator=generator,
                               stop_token_ids=stops, eos_token_id=model.cfg.eos_token_id,
                               **STRATEGIES[strategy])[0, len(ids):].tolist()
    kept = truncate_at_stop(generated, stops)
    text = tokenizer.decode(kept)
    return {"text": text, "generated_ids": generated, "ended": bool(generated and generated[-1] in stops),
            "eos": bool(generated and generated[-1] == EOS_ID),
            "end_of_turn": bool(generated and generated[-1] == END_OF_TURN_ID),
            "role_confusion": bool(set(kept).intersection((SYSTEM_ID, USER_ID, ASSISTANT_ID))
                                   or re.search(r"(?m)^\s*(用户|助手|系统|user|assistant|system)\s*[:：]", text)),
            **repetition(text, kept)}


@torch.inference_mode()
def chat_predictions(model, tokenizer, rows, protocols, index=None):
    model.eval()
    for strategy, seed in protocols:
        generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
        for row in rows:
            chat = row["messages"]
            for turn in range(1, len(chat), 2):
                prompt = chat[:turn]
                ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
                result = _generate(model, tokenizer, ids, strategy, generator)
                reference = chat[turn]["content"]
                yield {"id": row["id"], "category": row["topic_category"], "turn": turn // 2 + 1,
                       "prompt": prompt, "reference": reference, "strategy": strategy, "seed": seed, **result,
                       "exact_surface_match": result["text"].strip() == reference,
                       "normalized_match": normalized(result["text"]) == normalized(reference),
                       "training_overlap": match_answer(result["text"], index) if index is not None else None,
                       "human_review": {"status": "PENDING", "criteria": RUBRIC}}


def chat_summary(predictions):
    groups = defaultdict(list)
    for row in predictions:
        key = (row["strategy"], row["seed"])
        for condition in ("all", row["category"], "context_turn" if row["turn"] > 1 else "initial_turn"):
            groups[(*key, condition)].append(row)
    metrics = ("ended", "eos", "end_of_turn", "role_confusion", "normalized_match", "exact_surface_match",
               "token_4gram_repeat_ratio", "character_3gram_repeat_ratio")
    return [{"strategy": key[0], "seed": key[1], "subset": key[2], "outputs": len(rows),
             **{name: sum(row[name] for row in rows) / len(rows) for name in metrics},
             "train_overlap_12_chars": (sum(r["training_overlap"]["matched_chars"] >= 12 for r in rows) / len(rows)
                                       if rows[0]["training_overlap"] is not None else None),
             "human_review": "PENDING"} for key, rows in groups.items()]


def _receipt(directory, binding):
    manifest = read_json(directory / "manifest.json")
    if manifest["binding"] != binding:
        raise ValueError(f"{directory}: immutable report binding differs")
    for name, info in manifest["files"].items():
        if name != Path(name).name:
            raise ValueError("invalid report file name")
        verify_file(directory / name, info)
    return manifest


def diagnostic_result(model, tokenizer, rows, report, out, content_id, summary):
    target = report / "diagnostic-result"
    binding = {"checkpoint": file_info(out / "model.pt"), "dataset_content_id": content_id, "updates": summary["steps"]}
    if target.exists():
        _receipt(target, binding)
        return
    predictions = list(chat_predictions(model, tokenizer, rows, CHAT_DECODE[:1]))
    with report_directory(target) as directory:
        write_rows(directory / "predictions.jsonl", predictions)
        write_json(directory / "result.json", {"base": read_json(report / "diagnostic-base-loss.json"),
                   "trained": summary["final_val"], "generation": chat_summary(predictions),
                   "learning_curve": file_info(out / "training-log.jsonl")})
        publish_manifest(directory, "manifest.json", {**binding, "binding": binding})


def knowledge_specs(world):
    probes = read_json(KNOWLEDGE)
    if [row["id"] for row in probes] != [row["id"] for row in world]:
        raise ValueError("knowledge probes do not cover the frozen 97 questions")
    for row, task in zip(probes, world, strict=True):
        if (row["answer"] != task["reference"] or not row["cloze"].endswith("：")
                or row["answer"] in row["cloze"]
                or len({row["answer"], *row["distractors"]}) != len(row["distractors"]) + 1):
            raise ValueError(f"{task['id']}: malformed or leaking candidate probe")
        yield row, task


@torch.inference_mode()
def knowledge_predictions(model, tokenizer, specs):
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(42)
    for probe, task in specs:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": task["prompt"]}], add_generation_prompt=True)
        cloze = tokenizer.encode(probe["cloze"], category="zh")
        options = sorted([probe["answer"], *probe["distractors"]])
        rankings = {}
        # Each option is scored in isolation, so its position cannot affect the prompt.
        for condition, prefix in (("chat", prompt), ("cloze", cloze)):
            likelihoods = {}
            for option in options:
                tail = tokenizer.encode(option, category="zh")
                likelihoods[option] = _score_tail(model, prefix + tail, len(prefix), "mean")
                likelihoods[option]["conditional_probability"] = math.exp(-likelihoods[option]["negative_log_likelihood"])
            if len({tuple(row["token_ids"]) for row in likelihoods.values()}) != len(likelihoods):
                raise ValueError(f"{task['id']}: choices collapse under the tokenizer")
            rankings[condition] = _rank(likelihoods, probe["answer"])
        blank = _generate(model, tokenizer, cloze, "greedy", generator, raw=True, budget=32)
        yield {"id": task["id"], "rankings": rankings, "cloze": probe["cloze"],
               "free_cloze": blank, "free_cloze_score": score_answer(blank["text"], task),
               "reference": probe["answer"], "chance": 1 / len(options), "independent_review": "PENDING"}


def knowledge_summary(knowledge):
    return {"total": len(knowledge), "mean_chance": sum(r["chance"] for r in knowledge) / len(knowledge),
            "free_cloze_correct": sum(r["free_cloze_score"]["correct"] for r in knowledge),
            "choice_correct": {k: sum(r["rankings"][k]["score"]["correct"] for r in knowledge)
                               for k in ("chat", "cloze")}}


def baseline_knowledge(base, target, device):
    tokenizer, checkpoint, evidence = base
    probes = list(knowledge_specs(historical_tasks()[1]))
    binding = {"base": evidence, "probes": file_info(KNOWLEDGE), "history": file_info(ROOT / BENCHMARK),
               "environment": environment(device), "implementation": file_info(Path(__file__))}
    if target.exists():
        _receipt(target, binding)
        return
    model, _ = load_evaluation_model(checkpoint, tokenizer, device)
    with report_directory(target) as directory:
        predictions = list(knowledge_predictions(model, tokenizer, probes))
        write_rows(directory / "historical-knowledge.jsonl", predictions)
        write_json(directory / "result.json", knowledge_summary(predictions))
        publish_manifest(directory, "manifest.json", {"binding": binding})


def generalization_rows(path, train, heldout, historical):
    """Validate independent model/human review and the declared relationship to training."""
    if path is None:
        return []
    by_id = {row["id"]: row for row in train}
    taken = by_id.keys() | {row["id"] for row in heldout}
    train_facts = {row["fact_family"] for row in train}
    train_expressions = {row["question_family"] for row in train}
    cases = list(records(path))
    seen = set()
    for row in cases:
        review = row["review"]
        if (row["id"] in taken or row["id"] in seen or not review["reviewer"] or not review["evidence"]
                or review["origin"] not in ("human", "independent_model")
                or any(review["checks"].get(key) != "PASS" for key in CHECKS)
                or any(not row[key] for key in ("fact_family", "question_family", "source"))):
            raise ValueError("generalization review or identity is incomplete")
        validate_messages(row["messages"])
        kind = row["topic_category"]
        anchor = by_id.get(row.get("anchor_id"))
        if kind == "fact_generalization":
            valid = anchor and row["fact_family"] == anchor["fact_family"] and row["question_family"] not in train_expressions
        elif kind == "expression_generalization":
            valid = anchor and row["question_family"] == anchor["question_family"] and row["fact_family"] not in train_facts
        else:
            valid = kind == "unseen_knowledge" and not row.get("anchor_id") and row["fact_family"] not in train_facts
        if not valid:
            raise ValueError("generalization subtype and frozen family membership disagree")
        seen.add(row["id"])
    if (set(r["topic_category"] for r in cases) != {"fact_generalization", "expression_generalization", "unseen_knowledge"}
            or check_overlap((*train, *heldout, *cases), historical)):
        raise ValueError("generalization population is incomplete or has lexical overlap")
    return cases


def quantized_best(best, tokenizer, directory):
    binding = {"source": file_info(best), "bits": 4, "group_size": 128,
               "implementation": {name: file_info(ROOT / "src/minillm" / name)
                                  for name in ("quantize.py", "quantization_provenance.py")}}
    model, metadata = load_evaluation_model(best, tokenizer, torch.device("cpu"))
    target = directory / "model-int4.pt"
    if directory.exists():
        _receipt(directory, binding)
    else:
        with report_directory(directory) as staged:
            save_quantized(staged / target.name, model, 4, 128,
                           tokenizer_fingerprint=tokenizer.fingerprint, metadata=metadata)
            if (staged / target.name).stat().st_size > BUDGET_BYTES:
                raise ValueError("INT4 exceeds 50,000,000 bytes")
            publish_manifest(staged, "manifest.json", {"binding": binding})
    if target.stat().st_size > BUDGET_BYTES:
        raise ValueError("INT4 exceeds 50,000,000 bytes")
    compressed = torch.load(target, map_location="cpu", weights_only=True, mmap=True)
    if (any(compressed.get(key) != metadata[key] for key in ("architecture", "arch_version", "config", "mode",
                                                            "step", "tokens_seen", "tokenizer_fingerprint"))
            or compressed.get("quantized_bits") != 4):
        raise ValueError("INT4 checkpoint metadata does not match best FP32 source")
    verify_quantized_tensors(compressed, model)
    return target


def evaluate_experiment(settings, base, report, generalization):
    from .v5_sft_experiment import DEVICE, _write_or_verify

    tokenizer, checkpoint, _ = base
    dataset = ROOT / settings["dataset"]
    frozen = frozen_manifest(dataset, tokenizer, settings["max_seq_len"])
    out = ROOT / settings["out"] / "formal"
    state = load_training_state(out / "state.pt", mmap=True)
    manifest = state["manifest"]
    step_size = manifest["training"]["optimizer_updates_per_epoch"]
    epoch_steps = [step_size * n for n in range(1, settings["epochs"] + 1)]
    best_status = read_json(out / "best-checkpoint-status.json")
    best = best_status["checkpoint"]
    if (manifest["provenance"]["frozen_dataset"] != file_info(dataset / "manifest.json")
            or state["global_step"] != epoch_steps[-1] or [row["step"] for row in state["history"]] != epoch_steps
            or best_status["status"] != "AVAILABLE" or best["step"] != min(state["history"], key=lambda r: r["loss"])["step"]
            or best["val_loss"] != state["best_loss"]):
        raise ValueError("formal run has no complete validation-based selection")
    verify_file(out / "best.pt", best["file"])
    nodes = {"base20m": checkpoint}
    for i, step in enumerate(epoch_steps, 1):
        node = out / "milestones" / f"step-{step}"
        verify_milestone(node, manifest, step=step, tokens=frozen["identity"]["statistics"]["train"]["assistant_tokens"] * i)
        nodes[f"epoch{i}"] = node / "model.pt"
    best_model = torch.load(out / "best.pt", map_location="cpu", weights_only=True, mmap=True)
    best_epoch = torch.load(nodes[f"epoch{epoch_steps.index(best['step']) + 1}"], map_location="cpu",
                            weights_only=True, mmap=True)
    verify_model_metadata(best_model, {**state, "global_step": best["step"], "tokens_seen": best["tokens_seen"]},
                          context="selected checkpoint")
    require_identical_weights(best_epoch["model"], best_model["model"], "validation best")
    del best_model, best_epoch, state
    nodes["best_fp32"] = out / "best.pt"
    nodes["best_int4"] = quantized_best(out / "best.pt", tokenizer, report / "int4")
    raw_config = EvaluationConfig(**read_json(EVALUATION_CONFIG))
    validation_windows = _all_windows(PRETRAIN_DATASET / "val.bin", raw_config)
    tasks, world = historical_tasks()
    raw_tasks = [{**row, "generation_mode": "raw"} for row in tasks]
    probes = list(knowledge_specs(world))
    lock = {"selection": best, "nodes": {name: file_info(path) for name, path in nodes.items()},
            "training_manifest": file_info(out / "training-manifest.json"),
            "frozen_dataset": file_info(dataset / "manifest.json"), "history": file_info(ROOT / BENCHMARK),
            "knowledge_probes": file_info(KNOWLEDGE), "pretrain_val": validation_windows[2],
            "raw_config": file_info(EVALUATION_CONFIG), "chat_protocol": [list(pair) for pair in CHAT_DECODE],
            "chat_budget": BUDGET, "environment": environment(DEVICE),
            "generalization": file_info(generalization) if generalization else None,
            "implementation": implementation_identity(CONFIG, ROOT / "scripts/run_v5_language_sft_5k.py")}
    _write_or_verify(report / "test-unlock.json", lock)
    train, val, test = (list(records(dataset / f"{split}.jsonl")) for split in ("train", "val", "test"))
    cloze_tasks = [{"id": probe["id"] + "-cloze", "prompt": probe["cloze"]} for probe, _ in probes]
    if check_overlap(train, cloze_tasks):
        raise ValueError("training questions overlap historical cloze prompts")
    general = generalization_rows(generalization, train, (*val, *test), (*tasks, *cloze_tasks))
    val_data = StrictChatDataset(dataset / "val.jsonl", tokenizer, settings["max_seq_len"], rows=val)
    test_data = StrictChatDataset(dataset / "test.jsonl", tokenizer, settings["max_seq_len"], rows=test)
    index = answer_index(train)
    results = {}
    for name, path in nodes.items():
        target = report / "evaluation" / name
        binding = {"lock": file_info(report / "test-unlock.json"), "node": name}
        if target.exists():
            _receipt(target, binding)
        else:
            model, _ = load_evaluation_model(path, tokenizer, DEVICE)
            with report_directory(target) as directory:
                losses = {split: evaluate(model, data, DEVICE, batch_size=1,
                                          pad_token_id=model.cfg.pad_token_id, ignore_index=model.cfg.ignore_index)
                          for split, data in (("val", val_data), ("test", test_data))}
                losses["original_v5_val"] = evaluate_indexed_windows(model, *validation_windows,
                                                                     DEVICE, raw_config.batch_size)
                chat_results = {}
                for split, rows in (("val", val), ("test", test), ("generalization", general)):
                    predictions = list(chat_predictions(model, tokenizer, rows, CHAT_DECODE, index))
                    write_rows(directory / f"chat-{split}.jsonl", predictions)
                    chat_results[split] = chat_summary(predictions) if predictions else "PENDING"
                raw_results = []
                with (directory / "historical-continuation.jsonl").open("x", encoding="utf-8") as handle:
                    from .language_diagnostics import _record_predictions
                    for strategy in DECODE:
                        for seed in ((raw_config.seed,) if strategy == "greedy" else SEEDS):
                            decoded = batch_generations(model, tokenizer, raw_tasks, replace(raw_config, seed=seed),
                                                        STRATEGIES[strategy], 1.0, batch_size=24, scored=True)
                            meta = {"strategy": strategy, "seed": seed}
                            _record_predictions(handle, decoded, meta)
                            raw_results.append({**meta, **_generation_metrics(decoded),
                                                "ended": sum(row["stop_reason"] == "eos" for row in decoded)/len(decoded)})
                knowledge = list(knowledge_predictions(model, tokenizer, probes))
                write_rows(directory / "historical-knowledge.jsonl", knowledge)
                write_json(directory / "result.json", {"loss": losses, "raw": raw_results, "chat": chat_results,
                           "knowledge": knowledge_summary(knowledge)})
                publish_manifest(directory, "manifest.json", {"binding": binding})
            del model
        results[name] = read_json(target / "result.json")
    def greedy_rows(name):
        return {r["id"]: r for r in records(report / "evaluation" / name / "chat-test.jsonl")
                if r["strategy"] == "greedy" and r["turn"] == 1}

    aligned = {name: greedy_rows(name) for name in nodes}
    examples = []
    for category in sorted({row["topic_category"] for row in test}):
        examples.extend([row for row in test if row["topic_category"] == category][:6])
    comparison = [{"id": row["id"], "question": row["messages"][0]["content"],
                   "reference": row["messages"][1]["content"],
                   "responses": {name: entries[row["id"]]["text"] for name, entries in aligned.items()}}
                  for row in examples]
    _write_or_verify(report / "paired-test-examples.json", comparison)
    _write_or_verify(report / "results.json", {"results": results, "human_language_scoring": "PENDING",
                                              "int4_bytes": nodes["best_int4"].stat().st_size})
