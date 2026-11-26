"""Deterministic indexed Train-window sampling and category-weighted loss."""

from bisect import bisect_right
from collections import Counter
from itertools import accumulate
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from .assessment_io import read_json
from .corpus_budget import read_token_index
from .corpus_utils import identity, verify_file


def label_spans(documents, seq_len, windows):
    """Yield (category, start, stop) spans in shifted-label coordinates."""
    ends = list(accumulate(row["tokens"] for row in documents))
    for window in windows:
        start, end = window * seq_len + 1, (window + 1) * seq_len
        origin = start
        doc = bisect_right(ends, start)
        spans = []
        while start < end:
            stop = min(end, ends[doc])
            spans.append((documents[doc]["category"], start - origin, stop - origin))
            start, doc = stop, doc + 1
        yield spans


def representative_windows(documents, data, *, seed, token_budget, required=None):
    if sum(row["tokens"] for row in documents) != data.num_tokens:
        raise ValueError("Train Token Index does not cover the evaluation BIN")
    targets = data.seq_len - 1
    if targets <= 0:
        raise ValueError("representative windows must have a shifted target")
    capacity = min(len(data), token_budget // targets) if token_budget else len(data)
    compositions = []
    for spans in label_spans(documents, data.seq_len, range(len(data))):
        counts = Counter()
        for category, start, end in spans:
            counts[category] += end - start
        compositions.append(counts)
    population = sum(compositions, Counter())
    if required is None:
        required = set(population)
    missing = set(required) - set(population)
    if missing:
        raise ValueError(f"positive-weight categories lack complete-window targets: {sorted(missing)}")
    if capacity <= 0:
        raise ValueError("Train evaluation budget cannot cover a complete window")
    candidates = list(range(len(data)))
    random.Random(seed).shuffle(candidates)
    order = {window: index for index, window in enumerate(candidates)}
    selected, counts = [], Counter()
    if capacity == len(data):
        selected, counts = sorted(candidates), population
    else:
        # Sample within category strata, using actual label-position counts so
        # mixed windows can cover multiple categories and contribute proportionally.
        remaining = set(candidates)
        expected = {c: count / len(data) for c, count in population.items()}
        missing = set(required)
        while len(selected) < capacity:
            size = len(selected) + 1
            if missing:
                window = max(remaining, key=lambda w: (len(missing & compositions[w].keys()),
                                                        -order[w]))
            else:
                desired = {c: value * size for c, value in expected.items()}
                window = min(remaining, key=lambda w: (
                    sum(abs(counts[c] + compositions[w][c] - target) for c, target in desired.items()),
                    order[w]))
            remaining.remove(window)
            selected.append(window)
            counts.update(compositions[window])
            missing.difference_update(compositions[window])
        if missing:
            raise ValueError(f"Train evaluation budget did not cover required categories: {sorted(missing)}")
        selected.sort()
    return {"policy": "representative-indexed-complete-windows-v1", "seed": seed,
            "requested_token_budget": token_budget, "seq_len": data.seq_len,
            "window_indices": selected, "window_indices_sha256": identity(selected),
            "by_window": [dict(compositions[index]) for index in selected],
            "population_targets_by_category": dict(population),
            "sampled_targets_by_category": dict(counts),
            "coverage_required": sorted(required), "supervised_tokens": len(selected) * targets,
            "scope": "all_complete_windows" if len(selected) == len(data) else "sampled_complete_windows"}


def indexed_train_plan(directory, data, *, seed, token_budget):
    directory = Path(directory)
    files = read_json(directory / "dataset-manifest.json")["files"]
    index = directory / "train.token-index.jsonl"
    budget = directory / "budget-report.json"
    for path in (index, budget):
        verify_file(path, files[path.name])
    documents = read_token_index([index])["records"]
    ratios = read_json(budget)["category_token_ratios"]
    plan = representative_windows(documents, data, seed=seed, token_budget=token_budget,
                                  required={category for category, ratio in ratios.items() if ratio > 0})
    return documents, {**plan, "token_index": files[index.name], "budget_report": files[budget.name]}


@torch.inference_mode()
def evaluate_indexed_windows(model, data, documents, plan, device, batch_size):
    from .evaluation import ppl_from_totals

    model.eval()
    sums = Counter()
    windows = plan["window_indices"]
    all_spans = label_spans(documents, data.seq_len, windows)
    for offset in range(0, len(windows), batch_size):
        batch = windows[offset:offset + batch_size]
        inputs = torch.stack([data[index]["input_ids"] for index in batch]).to(device)
        logits = model(inputs).logits[:, :-1].float()
        losses = F.cross_entropy(logits.transpose(1, 2), inputs[:, 1:], reduction="none").double().cpu()
        for row in losses:
            spans = next(all_spans)
            for category, start, stop in spans:
                sums[category] += float(row[start:stop].sum())
    counts = plan["sampled_targets_by_category"]
    return {**ppl_from_totals(sum(sums.values()), plan["supervised_tokens"]),
            "batches": (len(windows) + batch_size - 1) // batch_size,
            "window_policy": plan["policy"], "scope": plan["scope"],
            "window_indices_sha256": plan["window_indices_sha256"],
            "by_category": {category: ppl_from_totals(sums[category], count)
                            for category, count in counts.items()}}
