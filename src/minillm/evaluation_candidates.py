"""Natural-prefix candidate likelihood probes, separate from instruction scores."""
from __future__ import annotations

import math

import torch

from .assessment_io import records
from .text_policy import validate_text_category

BOUNDARY_POLICIES = ("separate", "joint_shared_prefix")


def load_candidates(path):
    from .evaluation import task_asset_provenance
    rows, seen = [], set()
    for row in records(path) if path else ():
        validate_text_category(row.get("text_category", "auto"))
        choices = row.get("candidates")
        if (type(row.get("schema_version")) is not int or row["schema_version"] != 1
                or any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("id", "category", "prompt"))
                or row["id"] in seen or not isinstance(choices, dict) or not 2 <= len(choices) <= 8
                or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in choices.items())
                or len(set(choices.values())) != len(choices) or row.get("reference") not in choices):
            raise ValueError("invalid candidate probe schema/id/choices/reference")
        rows.append(row); seen.add(row["id"])
        if len(rows) > 10000:
            raise ValueError("candidate probe limit exceeded")
    if path and not rows:
        raise ValueError("empty candidate probes")
    task_asset_provenance(path, len(rows))
    return rows


def candidate_boundary_audit(tokenizer, task):
    """Compare actual token IDs; never assume BPE concatenation is compositional."""
    category = validate_text_category(task.get("text_category", "auto"))
    prompt = task["prompt"]
    prefix = tokenizer.encode(prompt, category=category)
    if not prefix:
        raise ValueError("empty candidate prefix")
    prepared_prompt = tokenizer.prepare_text(prompt, category=category)
    choices, shared = {}, list(prefix)
    for key, text in task["candidates"].items():
        continuation = tokenizer.encode(text, category=category)
        prepared_joint = tokenizer.prepare_text(prompt + text, category=category)
        joint = tokenizer._tok.encode(prepared_joint)
        count = 0
        for a,b in zip(shared, joint.ids):
            if a != b:
                break
            count += 1
        shared = shared[:count]
        choices[key] = {"separate_token_ids": prefix + continuation, "continuation_token_ids": continuation,
                        "joint_token_ids": joint.ids, "joint_offsets": joint.offsets,
                        "matches_joint": prefix + continuation == joint.ids,
                        "normalization_consistent": prepared_joint == prepared_prompt + tokenizer.prepare_text(text, category=category)}
    return {"prompt_token_ids": prefix, "text_category": category,
            "prepared_prompt_characters": len(prepared_prompt), "joint_shared_prefix_tokens": len(shared),
            "inconsistent_candidates": [k for k,v in choices.items() if not v["matches_joint"]],
            "normalization_context_consistent": all(v["normalization_consistent"] for v in choices.values()),
            "candidates": choices}


def _rank(results, reference):
    best = max(r["score"] for r in results.values())
    winners = [key for key, r in results.items() if math.isclose(r["score"], best, rel_tol=0, abs_tol=1e-7)]
    selected = winners[0] if len(winners) == 1 else None
    return {"status": "SCORED", "candidates": results, "selected": selected,
            "score": {"correct": selected == reference, "status": "tie" if selected is None else "scored"}}


def _score_tail(model, ids, start, normalization):
    if not 1 <= start < len(ids):
        raise ValueError("candidate tail requires at least one conditioning token and one target")
    inputs = torch.tensor([ids], device=next(model.parameters()).device)
    labels = inputs.clone(); labels[:, :start] = model.cfg.ignore_index
    output = model(inputs, labels=labels)
    count = len(ids)-start
    if output.num_loss_tokens != count or not torch.isfinite(output.loss_sum):
        raise ValueError("candidate effective token count/nonfinite loss")
    nll = float(output.loss_sum)
    return {"token_ids": ids[start:], "tokens": count, "negative_log_likelihood": nll,
            "score": -nll/(count if normalization == "mean" else 1)}


@torch.inference_mode()
def candidate_prediction(model, tokenizer, task, *, normalization="sum", boundary_policy="separate"):
    if normalization not in ("sum", "mean") or boundary_policy not in BOUNDARY_POLICIES:
        raise ValueError("invalid candidate likelihood normalization/boundary policy")
    model.eval()
    audit = candidate_boundary_audit(tokenizer, task)
    prefix = audit["prompt_token_ids"]
    choices = audit["candidates"]
    results, encoded = {}, set()
    for key, text in task["candidates"].items():
        row = choices[key]
        ids, continuation = row["separate_token_ids"], row["continuation_token_ids"]
        if (not continuation or len(ids)>model.cfg.max_position_embeddings
                or any(t in tokenizer.special_ids.values() for t in ids)):
            raise ValueError(f"{task['id']}: candidate context/empty/special token violation")
        if tuple(continuation) in encoded:
            raise ValueError("candidate choices collapse to identical token IDs under text policy")
        encoded.add(tuple(continuation))
        results[key] = {"text": text, **_score_tail(model, ids, len(prefix), normalization)}
    protocols = {"separate": _rank(results, task["reference"])}
    shared = audit["joint_shared_prefix_tokens"]
    reason = None
    if not audit["normalization_context_consistent"]:
        reason = "NOT_SCORABLE_NORMALIZATION_CONTEXT"
    elif not shared:
        reason = "NOT_SCORABLE_NO_SHARED_CONTEXT"
    elif len({tuple(r["joint_token_ids"]) for r in choices.values()}) != len(choices):
        reason = "NOT_SCORABLE_IDENTICAL_JOINT_CHOICES"
    elif any(not shared < len(r["joint_token_ids"]) <= model.cfg.max_position_embeddings for r in choices.values()):
        reason = "NOT_SCORABLE_CONTEXT"
    elif any(t in tokenizer.special_ids.values() for r in choices.values() for t in r["joint_token_ids"]):
        reason = "NOT_SCORABLE_SPECIAL_TOKENS"
    if reason:
        protocols["joint_shared_prefix"] = {"status": reason, "candidates": {}, "selected": None,
                                            "score": {"correct": None, "status": reason}}
    else:
        joint_results = {}
        for key,text in task["candidates"].items():
            row = choices[key]
            score = results[key] if shared == len(prefix) and row["matches_joint"] else _score_tail(model, row["joint_token_ids"], shared, normalization)
            joint_results[key] = {"text": text, **score,
                "contains_retokenized_prompt": shared < len(prefix),
                "prompt_characters_in_scored_tail": max(0,audit["prepared_prompt_characters"]-row["joint_offsets"][shared][0])}
        protocols["joint_shared_prefix"] = _rank(joint_results, task["reference"])
    primary = protocols[boundary_policy]
    return {"kind": "candidate_likelihood", "id": task["id"], "category": task["category"],
            "generation_mode": "raw", "prompt": task["prompt"], "prompt_token_ids": prefix,
            "text_category": audit["text_category"],
            "candidates": primary["candidates"], "reference": task["reference"], "selected": primary["selected"],
            "score": primary["score"], "boundary_audit": audit, "protocols": protocols,
            "normalization": normalization,
            "boundary_policy_name": boundary_policy,
            "boundary_policy": "separate: continuation-only NLL, matching generation input; joint_shared_prefix: joint-encoded suffix after a context shared by prompt and ALL choices, may include retokenized prompt. Different metrics, no mixed ranking."}


def aggregate_candidates(rows):
    rows = [r for r in rows if r["kind"] == "candidate_likelihood"]
    policies = {r["boundary_policy_name"] for r in rows}
    normalizations = {r["normalization"] for r in rows}
    if len(policies)>1 or len(normalizations)>1:
        raise ValueError("cannot aggregate mixed candidate boundary/normalization policies")
    result = _aggregate(rows)
    result["primary_boundary_policy"] = next(iter(policies), None)
    result["normalization"] = next(iter(normalizations), None)
    result["boundary_audit"] = {"inconsistent_tasks": sum(bool(r["boundary_audit"]["inconsistent_candidates"]) for r in rows),
        "inconsistent_candidates": sum(len(r["boundary_audit"]["inconsistent_candidates"]) for r in rows),
        "total_candidates": sum(len(r["boundary_audit"]["candidates"]) for r in rows),
        "affected_task_ids": [r["id"] for r in rows if r["boundary_audit"]["inconsistent_candidates"]],
        "bpe_inconsistent_tasks": sum(r["boundary_audit"]["normalization_context_consistent"] and bool(r["boundary_audit"]["inconsistent_candidates"]) for r in rows),
        "normalization_context_mismatches": sum(not r["boundary_audit"]["normalization_context_consistent"] for r in rows)}
    result["protocols"] = {name: _aggregate([{**r, **r["protocols"][name]} for r in rows]) for name in BOUNDARY_POLICIES}
    result["interpretation"] = "Original small probes; separate and joint_shared_prefix are different metrics. Joint scoring may include prompt text. Length/wording biases remain; not open-ended reasoning or chat ability."
    return result


def _aggregate(rows):
    categories = {}
    for name in sorted({r["category"] for r in rows}):
        subset = [r for r in rows if r["category"] == name]
        scored = [r for r in subset if r["score"]["correct"] is not None]
        categories[name] = {"correct": sum(r["score"]["correct"] for r in scored), "tasks": len(subset),
            "scored": len(scored), "unscorable": len(subset)-len(scored),
            "accuracy": sum(r["score"]["correct"] for r in scored)/len(scored) if scored else None,
            "ties": sum(r["score"]["status"] == "tie" for r in scored)}
    return {"status": "NOT_RUN" if not rows else "INCOMPLETE" if any(r["score"]["correct"] is None for r in rows) else "COMPLETED",
            "tasks": len(rows), "categories": categories}
