"""Probability traces, decoding sweeps and numerical checks on MiniLLM weights."""

from collections import defaultdict
import math
import time

import numpy as np
import torch

from .evaluation import EvaluationConfig, prediction_row, validate_generation
from .evaluation_scoring import aggregate_scores, score_answer
from .generate import stop_ids_for
from .language_token_analysis import is_punctuation
from .tokenizer import EOS_ID


STRATEGIES = {
    "greedy": {"temperature": 0.0, "top_k": 0, "top_p": 1.0},
    "low_temperature": {"temperature": 0.5, "top_k": 40, "top_p": 1.0},
    "standard_sampling": {"temperature": 0.8, "top_k": 50, "top_p": 1.0},
    "high_temperature": {"temperature": 1.0, "top_k": 100, "top_p": 1.0},
    "nucleus": {"temperature": 0.8, "top_k": 0, "top_p": 0.9},
}
SEEDS = (42, 43, 44)
PENALTIES = (1.0, 1.1, 1.2)


def probability_summary(logits, vocab, punctuation, target=None):
    logits = logits.float()
    logp = logits.log_softmax(-1)
    probabilities = logp.exp()
    values, ids = logits.topk(20)
    top = [{**vocab[token], "probability": math.exp(logprob), "logit": value}
           for token, logprob, value in zip(ids.tolist(), logp[ids].tolist(), values.tolist())]
    return {"top_20": top, "max_probability": top[0]["probability"],
            "top_1_logit_margin": top[0]["logit"] - top[1]["logit"],
            "punctuation_probability": float(probabilities[punctuation].sum()),
            "newline_whitespace_probability": float(probabilities[
                [row["id"] for row in vocab if row["class"] == "newline_whitespace"]].sum()),
            "entropy_nats": float(torch.special.entr(probabilities).sum()),
            "target": {**vocab[target], "probability": float(probabilities[target]),
                       "rank": int((logits > logits[target]).sum()) + 1}
            if target is not None else None}, probabilities.cpu().numpy()


@torch.inference_mode()
def trace_predictions(model, tokenizer, tasks, vocab, config):
    device = next(model.parameters()).device
    punctuation = [row["id"] for row in vocab if row["class"] == "punctuation"]
    result, arrays, initial = [], {}, {}
    for task in tasks:
        ids, _ = validate_generation(model, tokenizer, task, config)
        suffix = task.get("suffix")
        target_ids = None
        if suffix:
            combined = tokenizer.encode(task["prompt"] + suffix, category=task["text_category"])
            if combined[:len(ids)] != ids or len(combined) <= len(ids):
                raise ValueError(f"{task['id']}: no joint tokenization boundary for the reference")
            target_ids = combined[len(ids):]
        current = torch.tensor([ids], device=device)
        if target_ids:
            combined_ids = torch.tensor([ids + target_ids], device=device)
            teacher = model(combined_ids).logits[0, len(ids)-1:-1].float()
            losses = teacher.logsumexp(-1) - teacher.gather(1, combined_ids[:, len(ids):].T).squeeze(1)
            fixed_loss = {"mean_loss": float(losses.mean()), "tokens": len(target_ids)}
        else:
            fixed_loss = None
        cache = None
        steps, distribution = [], []
        for index in range(32):
            output = model(current, past_key_values=cache, use_cache=True)
            logits = output.logits[0, -1]
            if index == 0:
                initial[task["id"]] = logits.cpu()
            entry, probabilities = probability_summary(
                logits, vocab, punctuation, target_ids[0] if target_ids and index == 0 else None)
            # Match the production Greedy tie rule.
            next_id = int(logits.argmax())
            steps.append({"step": index + 1, "greedy_token_id": next_id, **entry})
            distribution.append(probabilities)
            if next_id == EOS_ID:
                break
            current, cache = torch.tensor([[next_id]], device=device), output.past_key_values
        arrays[task["id"]] = np.stack(distribution)
        result.append({"id": task["id"], "prompt": task["prompt"], "prompt_ids": ids,
                       "reference_suffix": suffix, "reference_token_ids": target_ids,
                       "reference_status": "ORIGINAL_FIXED_SUFFIX" if suffix else "OPEN_ENDED_NO_GOLD_TOKEN",
                       "teacher_forced_suffix_loss": fixed_loss, "steps": steps})
    return result, arrays, initial


@torch.inference_mode()
def batch_generations(model, tokenizer, tasks, config, strategy, penalty, *, batch_size=16, scored=False):
    """Left-pad in fixed budget groups and seed one RNG stream per full sweep."""
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(config.seed)
    groups = defaultdict(list)
    for index, task in enumerate(tasks):
        ids, budget = validate_generation(model, tokenizer, task, config)
        if task["generation_mode"] != "raw":
            raise ValueError("diagnostic batched generation requires raw prompts")
        groups[budget].append((index, task, ids))
    result = {}
    stops = stop_ids_for(raw=True)
    for budget, group in sorted(groups.items()):
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            width = max(len(ids) for _, _, ids in batch)
            padded = [[model.cfg.pad_token_id] * (width - len(ids)) + ids for _, _, ids in batch]
            inputs = torch.tensor(padded, device=device)
            mask = torch.tensor([[False] * (width - len(ids)) + [True] * len(ids) for _, _, ids in batch], device=device)
            started = time.perf_counter()
            ids_by_row = model.generate(inputs, max_new_tokens=budget, generator=generator,
                                        stop_token_ids=stops, eos_token_id=model.cfg.eos_token_id,
                                        attention_mask=mask, repetition_penalty=penalty,
                                        **strategy)[:, width:].tolist()
            seconds = time.perf_counter() - started
            for (index, task, ids), generated in zip(batch, ids_by_row, strict=True):
                for position, token in enumerate(generated):
                    if token in stops:
                        generated = generated[:position + 1]
                        break
                row = {"kind": "benchmark" if scored else "fixed_prompt",
                       **prediction_row(tokenizer, task, ids, generated, budget, stops, seconds / len(batch))}
                row["diagnostics"]["punctuation_characters"] = sum(map(is_punctuation, row["decoded_text"]))
                if scored:
                    row["reference"], row["scoring"] = task.get("reference"), task["scoring"]
                    if task.get("rubric"):
                        row["rubric"] = task["rubric"]
                    row["score"] = score_answer(row["decoded_text"], task)
                    if row["special_token_anomalies"] and row["score"]["correct"] is not None:
                        row["score"] = {"correct": False, "status": "invalid_special_tokens"}
                result[index] = row
    return [result[index] for index in range(len(tasks))]


def generation_metrics(rows):
    size = len(rows)
    diagnostics = [row["diagnostics"] for row in rows]
    chars = sum(row["nonspace_characters"] for row in diagnostics)
    def fraction(key):
        return sum(row[key] for row in diagnostics) / chars if chars else None
    return {"outputs": size, "nonspace_characters": chars,
            "han_character_fraction": fraction("han_characters"),
            "han_outputs": sum(row["han_characters"] > 0 for row in diagnostics),
            "punctuation_character_fraction": fraction("punctuation_characters"),
            "abnormal_character_fraction": fraction("abnormal_characters"),
            "replacement_characters": sum(row["replacement_characters"] for row in diagnostics),
            "mean_token_4gram_repeat_ratio": sum(row["token_4gram_repeat_ratio"] for row in diagnostics) / size,
            "mean_character_3gram_repeat_ratio": sum(row["character_3gram_repeat_ratio"] for row in diagnostics) / size,
            "empty_output_fraction": sum(row["empty_output"] for row in rows) / size,
            "special_token_anomalies": sum(len(row["special_token_anomalies"]) for row in rows),
            "scores": aggregate_scores(rows) if rows[0]["kind"] == "benchmark" else None}


def evaluate_sweep(model, tokenizer, tasks, max_new_tokens):
    for name, strategy in STRATEGIES.items():
        for penalty in PENALTIES:
            for seed in SEEDS if name != "greedy" else (SEEDS[0],):
                config = EvaluationConfig(seed=seed, max_new_tokens=max_new_tokens)
                rows = batch_generations(model, tokenizer, tasks, config, strategy, penalty)
                yield {"strategy": name, "parameters": strategy, "seed": seed, "repetition_penalty": penalty}, rows


def choose_sampling_strategy(rows):
    """Select on fixed probes using unpenalized samples, before viewing benchmark outputs."""
    return max((name for name in STRATEGIES if name != "greedy"), key=lambda name: (
        sum(row["han_outputs"] for row in rows if row["strategy"] == name and row["repetition_penalty"] == 1.0),
        -sum(row["mean_token_4gram_repeat_ratio"] for row in rows
             if row["strategy"] == name and row["repetition_penalty"] == 1.0)))


@torch.inference_mode()
def verify_attention(model, ids, *, steps=12, eos_prompt=None):
    device = next(model.parameters()).device
    inputs = torch.tensor([ids], device=device)
    changed = inputs.clone()
    split = max(1, len(ids) // 2)
    changed[:, split:] = (changed[:, split:] + 7) % model.cfg.vocab_size
    initial = model(inputs).logits
    future_delta = float((initial[:, :split] - model(changed).logits[:, :split]).abs().max())
    if future_delta > 3e-5:
        raise AssertionError(f"causal attention leaked future tokens: delta={future_delta}")

    args = {"max_new_tokens": steps, "eos_token_id": model.cfg.eos_token_id}
    cache_ids = model.generate(inputs, use_cache=True, **args)
    full_ids = model.generate(inputs, use_cache=False, **args)
    # Follow the same Greedy path when comparing per-step logits, even after a near tie.
    state = model(inputs, use_cache=True)
    deltas, disagreements = [], []
    for index, token in enumerate(cache_ids[0, len(ids):]):
        prefix = cache_ids[:, :len(ids) + index]
        uncached = model(prefix).logits[0, -1]
        cached = state.logits[0, -1]
        delta = float((cached - uncached).abs().max())
        deltas.append(delta)
        torch.testing.assert_close(cached, uncached, atol=5e-5, rtol=2e-4)
        if int(cached.argmax()) != int(uncached.argmax()):
            top = uncached.topk(2)
            margin = float(top.values[0] - top.values[1])
            if margin > delta * 2:
                raise AssertionError("cache argmax differs outside numerical near-tie tolerance")
            disagreements.append({"step": index + 1, "cached": int(cached.argmax()),
                                  "uncached": int(uncached.argmax()), "uncached_margin": margin,
                                  "max_absolute_logit_delta": delta})
        if index + 1 < cache_ids.shape[1] - len(ids):
            state = model(token.reshape(1, 1), past_key_values=state.past_key_values, use_cache=True)
    # A multi-token chunk must use the cache length for both RoPE and causal masking.
    tail = model(inputs[:, split:], past_key_values=model(inputs[:, :split], use_cache=True).past_key_values,
                 position_offset=split, use_cache=True).logits
    torch.testing.assert_close(tail, initial[:, split:], atol=5e-5, rtol=2e-4)
    if not torch.equal(cache_ids, full_ids) and not disagreements:
        raise AssertionError("cached and uncached generation diverged without a recorded numerical near tie")

    maximum = model.cfg.max_position_embeddings
    context = inputs.repeat(1, math.ceil(maximum / len(ids)))[:, :maximum]
    at_max = model.generate(context[:, :-1], max_new_tokens=1)
    assert at_max.shape[1] == maximum
    assert torch.equal(model.generate(context, max_new_tokens=0), context)
    try:
        model.generate(context, max_new_tokens=1)
    except ValueError as exc:
        if "max_position_embeddings" not in str(exc):
            raise
    else:
        raise AssertionError("generation accepted a beyond-context request")
    real_eos = {"status": "NO_GREEDY_EOS_PROMPT_AVAILABLE"}
    if eos_prompt:
        prompt = torch.tensor([eos_prompt], device=device)
        result = model.generate(prompt, max_new_tokens=96, eos_token_id=model.cfg.eos_token_id)
        ids_generated = result[0, len(eos_prompt):].tolist()
        assert ids_generated[-1] == model.cfg.eos_token_id and model.cfg.eos_token_id not in ids_generated[:-1]
        real_eos = {"status": "PASS", "prompt_ids": eos_prompt, "generated_ids": ids_generated}
    first = int(initial[0, -1].argmax())
    custom_stop = model.generate(inputs, max_new_tokens=steps, eos_token_id=model.cfg.eos_token_id,
                                 stop_token_ids=(first,))
    assert custom_stop[0, len(ids):].tolist() == [first]
    return {"status": "PASS", "future_token_max_delta": future_delta, "cache_step_max_absolute_deltas": deltas,
            "cache_chunk_max_absolute_delta": float((tail - initial[:, split:]).abs().max()),
            "greedy_cache_ids": cache_ids[0, len(ids):].tolist(), "greedy_uncached_ids": full_ids[0, len(ids):].tolist(),
            "cache_greedy_equal": torch.equal(cache_ids, full_ids), "near_ties": disagreements,
            "context_limit": maximum, "context_limit_checks": "PASS", "eos_stop": real_eos,
            "stop_on_real_top_token": {"status": "PASS", "token_id": first}}


@torch.inference_mode()
def verify_sampled_eos(model, prompts, seed_limit=4096):
    """Exercise the configured EOS with real logits and a recorded sampling seed."""
    device = next(model.parameters()).device
    eos = model.cfg.eos_token_id
    candidates = []
    for ids in prompts:
        inputs = torch.tensor([ids], device=device)
        logits = model(inputs, use_cache=True).logits[:, -1]
        candidates.append((float(logits.softmax(-1)[0, eos]), inputs, logits))
    probability, inputs, logits = max(candidates, key=lambda row: row[0])
    generator = torch.Generator(device=device)
    # Separate stop-logic verification. This seed search is outside quality experiments.
    for seed in range(seed_limit):
        if int(model._sample(logits, 1.0, 0, generator.manual_seed(seed))) == eos:
            generated = model.generate(inputs, max_new_tokens=8, temperature=1.0,
                                       eos_token_id=eos, generator=generator.manual_seed(seed))
            tail = generated[0, inputs.shape[1]:].tolist()
            assert tail == [eos]
            return {"status": "PASS", "mode": "seeded_full_vocabulary_sampling", "seed": seed,
                    "prompt_ids": inputs[0].tolist(), "eos_probability": probability, "generated_ids": tail}
    return {"status": "NO_EOS_DRAW_IN_PROBE", "seeds_checked": seed_limit, "eos_probability": probability}


def quantization_comparison(fp, quantized):
    rows = []
    for task, first in fp.items():
        second = quantized[task]
        delta = second - first
        common = set(first.topk(20).indices.tolist()) & set(second.topk(20).indices.tolist())
        rows.append({"id": task, "max_absolute_logit_delta": float(delta.abs().max()),
                     "logit_rmse": float(delta.square().mean().sqrt()), "top20_overlap": len(common),
                     "top1_same": int(first.argmax()) == int(second.argmax()),
                     "kl_fp32_to_dequantized": float((first.softmax(-1) *
                         (first.log_softmax(-1) - second.log_softmax(-1))).sum())})
    return rows
