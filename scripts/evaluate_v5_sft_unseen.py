#!/usr/bin/env python3
"""Prioritized side-by-side evaluation on the frozen unseen questions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re
from statistics import mean
import time

import torch

from audit_v5_sft_local import PAIR_SYSTEM, REVIEWER
from generate_v5_sft_local import TEXT, inference, table, until_valid
from minillm.assessment_io import environment, publish_manifest, read_json, records, report_directory
from minillm.checkpoint import load_training_state
from minillm.corpus_utils import file_info, identity, verify_file
from minillm.evaluation import load_evaluation_model, repetition
from minillm.generate import stop_ids_for, truncate_at_stop
from minillm.language_generation_analysis import STRATEGIES
from minillm.tokenizer import ASSISTANT_ID, END_OF_TURN_ID, EOS_ID, SYSTEM_ID, USER_ID
from minillm.train import evaluate
from minillm.training_manifest import implementation_identity, write_json
from minillm.training_milestones import require_identical_weights, verify_milestone, verify_model_metadata
from minillm.v5_sft_data import CONFIG, QUOTAS, ROOT, StrictChatDataset, frozen_manifest, normalized, write_rows
from minillm.v5_sft_evaluation import BUDGET, _receipt, answer_index, chat_summary, match_answer, quantized_best
from minillm.v5_sft_experiment import DEVICE, _write_or_verify, base_assets

PROTOCOLS = (("greedy", 42), ("standard_sampling", 42))
BATCH_SIZE = 16


def training_nodes(settings, dataset, frozen, base):
    out = ROOT / settings["out"] / "formal"
    state = load_training_state(out / "state.pt", mmap=True)
    manifest = state["manifest"]
    updates = manifest["training"]["optimizer_updates_per_epoch"]
    epoch_steps = [updates * i for i in range(1, settings["epochs"] + 1)]
    status = read_json(out / "best-checkpoint-status.json")
    best = status["checkpoint"]
    if (manifest["provenance"]["frozen_dataset"] != file_info(dataset / "manifest.json")
            or state["global_step"] != epoch_steps[-1]
            or [r["step"] for r in state["history"]] != epoch_steps
            or status["status"] != "AVAILABLE"
            or best["step"] != min(state["history"], key=lambda r: r["loss"])["step"]
            or best["val_loss"] != state["best_loss"]):
        raise ValueError("formal training or validation-based model selection is incomplete")
    nodes = {"base20m": base}
    expected_tokens = frozen["identity"]["statistics"]["train"]["assistant_tokens"]
    for i, step in enumerate(epoch_steps, 1):
        path = out / "milestones" / f"step-{step}"
        verify_milestone(path, manifest, step=step, tokens=expected_tokens * i)
        nodes[f"epoch{i}"] = path / "model.pt"
    selected = out / "best.pt"
    verify_file(selected, best["file"])
    saved, epoch = (torch.load(path, map_location="cpu", mmap=True, weights_only=True)
                    for path in (selected, nodes[f"epoch{epoch_steps.index(best['step'])+1}"]))
    verify_model_metadata(saved, {**state, "global_step": best["step"], "tokens_seen": best["tokens_seen"]},
                          context=str(selected))
    require_identical_weights(epoch["model"], saved["model"], "validation-selected unseen evaluation")
    return nodes, best, selected


def requests(rows, tokenizer):
    pending = []
    for row in rows:
        for turn in range(1, len(row["messages"]), 2):
            prompt = row["messages"][:turn]
            pending.append({"id": row["id"], "category": row["topic_category"], "turn": turn // 2 + 1,
                            "prompt": prompt, "reference": row["messages"][turn]["content"],
                            "prompt_ids": tokenizer.apply_chat_template(prompt, add_generation_prompt=True)})
    return sorted(pending, key=lambda row: len(row["prompt_ids"]))


@torch.inference_mode()
def generate_predictions(model, tokenizer, prompts, index, protocols=PROTOCOLS):
    device = next(model.parameters()).device
    stops = stop_ids_for(False)
    for strategy, seed in protocols:
        generator = torch.Generator(device=device).manual_seed(seed)
        for start in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[start:start+BATCH_SIZE]
            width = max(len(item["prompt_ids"]) for item in batch)
            inputs = torch.tensor([[model.cfg.pad_token_id] * (width - len(item["prompt_ids"]))
                                   + item["prompt_ids"] for item in batch], device=device)
            mask = torch.tensor([[False] * (width-len(item["prompt_ids"])) + [True] * len(item["prompt_ids"])
                                 for item in batch], device=device)
            replies = model.generate(inputs, max_new_tokens=BUDGET, attention_mask=mask,
                                     generator=generator, eos_token_id=model.cfg.eos_token_id,
                                     stop_token_ids=stops, **STRATEGIES[strategy])[:, width:].tolist()
            for row, generated in zip(batch, replies, strict=True):
                kept = truncate_at_stop(generated, stops)
                stop = generated[len(kept)] if len(kept) < len(generated) else None
                text = tokenizer.decode(kept)
                yield {**row, "strategy": strategy, "seed": seed, "text": text,
                       "generated_ids": generated[:len(kept)+(stop is not None)],
                       "ended": stop is not None, "eos": stop == EOS_ID, "end_of_turn": stop == END_OF_TURN_ID,
                       "normalized_match": normalized(text) == normalized(row["reference"]),
                       "exact_surface_match": text.strip() == row["reference"],
                       "training_overlap": match_answer(text, index),
                       "role_confusion": bool(set(kept) & {SYSTEM_ID, USER_ID, ASSISTANT_ID}
                                              or re.search(r"(?m)^\s*(用户|助手|系统)\s*[:：]", text)),
                       **repetition(text, kept)}


def generation_stage(settings):
    tokenizer, base, _ = base_assets(settings)
    dataset = ROOT / settings["dataset"]
    frozen = frozen_manifest(dataset, tokenizer, settings["max_seq_len"])
    report = ROOT / settings["report"] / "unseen"
    report.mkdir(parents=True, exist_ok=True)
    nodes, best, selected = training_nodes(settings, dataset, frozen, base)
    nodes["best_fp32"] = selected
    nodes["best_int4"] = quantized_best(selected, tokenizer, report / "int4")
    binding = {"selection": best, "nodes": {name: file_info(path) for name, path in nodes.items()},
               "dataset": file_info(dataset / "manifest.json"),
               "protocols": [list(item) for item in PROTOCOLS], "budget": BUDGET, "batch_size": BATCH_SIZE,
               "environment": environment(DEVICE), "context": "reference history on subsequent turns",
               "implementation": implementation_identity(CONFIG, Path(__file__))}
    lock_path = report / "protocol-lock.json"
    _write_or_verify(lock_path, binding)
    test, train = (list(records(dataset / f"{split}.jsonl")) for split in ("test", "train"))
    if Counter(r["topic_category"] for r in test) != QUOTAS["test"]:
        raise ValueError("unseen evaluation requires the entire frozen Test split")
    test_data = StrictChatDataset(dataset / "test.jsonl", tokenizer, settings["max_seq_len"], rows=test)
    index = answer_index(train)
    prompts = requests(test, tokenizer)
    summary = {}
    for name, path in nodes.items():
        dest = report / "nodes" / name
        receipt = {"protocol_lock": file_info(lock_path), "node": name}
        if dest.exists():
            _receipt(dest, receipt)
        else:
            model, _ = load_evaluation_model(path, tokenizer, DEVICE)
            loss = evaluate(model, test_data, DEVICE, batch_size=1, pad_token_id=model.cfg.pad_token_id,
                            ignore_index=model.cfg.ignore_index)
            started = time.monotonic()
            predictions = list(generate_predictions(model, tokenizer, prompts, index))
            if len(predictions) != len(prompts) * len(PROTOCOLS):
                raise ValueError("incomplete unseen predictions")
            with report_directory(dest) as staged:
                write_rows(staged / "predictions.jsonl", predictions)
                write_json(staged / "result.json", {"loss": loss, "greedy_and_sampling": chat_summary(predictions),
                           "seconds": time.monotonic() - started, "human_review": "PENDING"})
                publish_manifest(staged, "manifest.json", {"binding": receipt})
            del model
        summary[name] = read_json(dest / "result.json")
        print(f"{name} Test assistant loss {summary[name]['loss']['loss']:.5f}; {len(prompts)} turns", flush=True)
    _write_or_verify(report / "generation-summary.json", summary)
    greedy = {name: {(r["id"], r["turn"]): r for r in records(report / "nodes" / name / "predictions.jsonl")
                     if r["strategy"] == "greedy"} for name in nodes}
    pairs = [{"id": r["id"], "turn": r["turn"], "category": r["category"], "prompt": r["prompt"],
              "reference": r["reference"],
              "answers": {name: items[r["id"], r["turn"]]["text"] for name, items in greedy.items()}}
             for r in sorted(prompts, key=lambda r: (r["id"], r["turn"]))]
    _write_or_verify(report / "all-paired-greedy.json", pairs)


def judge_group(index, rows, directory, port):
    path = directory / f"{index:04}.json"
    if path.exists():
        return read_json(path)["decisions"]
    entries, flips = [], []
    for i, row in enumerate(rows, 1):
        flip = int(identity((row["id"], row["turn"]))[-1], 16) % 2
        flips.append(flip)
        models = ("best_fp32", "base20m") if flip else ("base20m", "best_fp32")
        entries.append({"id": i, "category": row["category"], "prompt": row["prompt"], "reference": row["reference"],
                        "A": row["answers"][models[0]], "B": row["answers"][models[1]]})
    fields = {"id": {"type": "integer", "minimum": 1, "maximum": len(rows)},
              "a_pass": {"type": "boolean"}, "b_pass": {"type": "boolean"},
              "preference": {"enum": ["A", "B", "tie"]}, "a_reason": TEXT, "b_reason": TEXT}
    def parse(result):
        answers = json.loads(result["content"])["rows"]
        if {r["id"] for r in answers} != set(range(1, len(rows)+1)) or len(answers) != len(rows):
            raise ValueError("incomplete blind judgments")
        return sorted(answers, key=lambda r: r["id"])

    answers = until_valid(path, path, lambda n: inference(port, PAIR_SYSTEM, json.dumps(entries, ensure_ascii=False),
                          table(fields, len(rows)), seed=151500+index*5+n, limit=420, temperature=.10), parse)
    result = []
    if answers:
        for row, judged, flip in zip(rows, answers, flips, strict=True):
            base, best = ("b", "a") if flip else ("a", "b")
            preferred = {"tie": "tie", base.upper(): "base20m", best.upper(): "best_fp32"}[judged["preference"]]
            result.append({"id": row["id"], "turn": row["turn"], "category": row["category"],
                           "base_pass": judged[base+"_pass"], "best_pass": judged[best+"_pass"],
                           "preferred": preferred, "base_reason": judged[base+"_reason"],
                           "best_reason": judged[best+"_reason"]})
    write_json(path, {"decisions": result, "inputs": [r["id"] for r in rows]})
    return result


def review_stage(settings, port):
    root = ROOT / settings["report"] / "unseen"
    pairs_path = root / "all-paired-greedy.json"
    pairs = read_json(pairs_path)
    directory = root / "blind-review"
    directory.mkdir(exist_ok=True)
    reviewer_path = (ROOT / "artifacts/v5-language-sft-5k-v1/campaign-v1/models"
                     / "Qwen3.5-4B-Q4_K_M.gguf")
    _write_or_verify(directory / "profile.json",
                     {"reviewer": REVIEWER, "weights": file_info(reviewer_path), "human_review": "PENDING",
                      "input": file_info(pairs_path), "pair_system": PAIR_SYSTEM, "blind": "stable randomized A/B"})
    groups = [pairs[i:i+3] for i in range(0, len(pairs), 3)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        judged = [row for batch in pool.map(lambda args: judge_group(*args, directory, port), enumerate(groups))
                  for row in batch]
    populations = defaultdict(list)
    for row in judged:
        for key in ("overall", row["category"], "followup" if row["turn"] > 1 else "initial"):
            populations[key].append(row)
    summary = {key: {"reviewed": len(values), "base_pass_rate": mean(r["base_pass"] for r in values),
                     "best_pass_rate": mean(r["best_pass"] for r in values),
                     "preference": dict(Counter(r["preferred"] for r in values))}
               for key, values in populations.items()}
    _write_or_verify(root / "blind-review-summary.json",
                     {"results": summary, "reviewed": len(judged), "expected": len(pairs),
                      "status": "COMPLETE" if len(judged) == len(pairs) else "PARTIAL",
                      "interpretation": "independent-model judgment, human review pending"})
    indexed = {(r["id"], r["turn"]): r for r in judged}
    rng = random.Random(42)
    cases = []
    for category in sorted(QUOTAS["test"]):
        eligible = [r for r in pairs if r["category"] == category and r["turn"] == 1]
        for row in rng.sample(eligible, 6):
            cases.append({**row, "independent_judgment": indexed.get((row["id"], row["turn"]))})
    _write_or_verify(root / "30-paired-examples.json", cases)
    if len(judged) != len(pairs):
        raise RuntimeError("blind review produced incomplete coverage; see the saved partial report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "review"))
    parser.add_argument("--port", type=int, default=18145)
    args = parser.parse_args()
    settings = read_json(CONFIG)
    torch.set_num_threads(settings["threads"])
    if args.stage == "generate":
        generation_stage(settings)
    else:
        review_stage(settings, args.port)


if __name__ == "__main__":
    main()
