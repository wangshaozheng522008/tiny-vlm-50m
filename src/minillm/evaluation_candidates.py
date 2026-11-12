"""Natural-prefix candidate likelihood probes, separate from instruction scores."""
from __future__ import annotations

import math

import torch

from .assessment_io import records


def load_candidates(path):
    from .evaluation import task_asset_provenance
    rows, seen = [], set()
    for row in records(path) if path else ():
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


@torch.inference_mode()
def candidate_prediction(model, tokenizer, task, *, normalization="sum"):
    if normalization not in ("sum", "mean"):
        raise ValueError("candidate likelihood normalization must be sum or mean")
    model.eval()
    prefix = tokenizer.encode(task["prompt"])
    if not prefix:
        raise ValueError("empty candidate prefix")
    results, encoded = {}, set()
    device = next(model.parameters()).device
    for key, text in task["candidates"].items():
        # Same boundary as generation: tokenize prefix first, then continuation.
        # Reference never enters model input or candidate ordering.
        continuation = tokenizer.encode(text)
        if (not continuation or len(prefix)+len(continuation) > model.cfg.max_position_embeddings
                or any(t in tokenizer.special_ids.values() for t in prefix+continuation)):
            raise ValueError(f"{task['id']}: candidate context/empty/special token violation")
        if tuple(continuation) in encoded:
            raise ValueError("candidate choices collapse to identical token IDs under text policy")
        encoded.add(tuple(continuation))
        ids = torch.tensor([prefix+continuation], device=device)
        labels = ids.clone(); labels[:, :len(prefix)] = model.cfg.ignore_index
        output = model(ids, labels=labels)
        if output.num_loss_tokens != len(continuation) or not torch.isfinite(output.loss_sum):
            raise ValueError("candidate effective token count/nonfinite loss")
        nll = float(output.loss_sum)
        results[key] = {"text": text, "token_ids": continuation, "tokens": len(continuation),
                        "negative_log_likelihood": nll,
                        "score": -nll/(len(continuation) if normalization == "mean" else 1)}
    best = max(r["score"] for r in results.values())
    winners = [key for key, r in results.items() if math.isclose(r["score"], best, rel_tol=0, abs_tol=1e-7)]
    selected = winners[0] if len(winners) == 1 else None
    return {"kind": "candidate_likelihood", "id": task["id"], "category": task["category"],
            "generation_mode": "raw", "prompt": task["prompt"], "prompt_token_ids": prefix,
            "candidates": results, "reference": task["reference"], "selected": selected,
            "score": {"correct": selected == task["reference"], "status": "tie" if selected is None else "scored"},
            "normalization": normalization,
            "boundary_policy": "encode prefix and each continuation separately; internal causal shift, only continuation labels counted"}


def aggregate_candidates(rows):
    rows = [r for r in rows if r["kind"] == "candidate_likelihood"]
    categories = {}
    for name in sorted({r["category"] for r in rows}):
        subset = [r for r in rows if r["category"] == name]
        correct = sum(r["score"]["correct"] for r in subset)
        categories[name] = {"correct": correct, "tasks": len(subset), "accuracy": correct/len(subset),
                            "ties": sum(r["selected"] is None for r in subset)}
    return {"status": "COMPLETED" if rows else "NOT_RUN", "tasks": len(rows), "categories": categories,
            "interpretation": "small original natural-prefix probes; candidate likelihood has token-length and wording bias, not open-ended reasoning or chat ability"}
