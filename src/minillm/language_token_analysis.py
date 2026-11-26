"""Token/character accounting for the frozen language pilot and observed labels."""

import math
import unicodedata
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np

from .assessment_io import records
from .corpus_budget import supervision_by_category
from .corpus_quality import V4_CATEGORIES
from .corpus_utils import identity
from .data import EpochBatchSampler
from .v4_training_accounting import category_consumption


def is_han(char):
    return unicodedata.name(char, "").startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH"))


def is_punctuation(char):
    return unicodedata.category(char).startswith("P")


def vocabulary(tokenizer):
    special = {token: name for name, token in tokenizer.special_ids.items()}
    rows = []
    for token in range(tokenizer.vocab_size):
        text = tokenizer.decode([token], skip_special_tokens=False)
        core = text.strip()
        if token in special:
            kind = "special"
        elif not text or "\ufffd" in text:
            kind = "byte_fragment_or_replacement"
        elif not core:
            kind = "newline_whitespace" if "\n" in text or "\r" in text else "other_whitespace"
        elif all(is_punctuation(char) for char in core):
            kind = "punctuation"
        elif all(is_han(char) for char in core):
            kind = "single_han" if len(core) == 1 else "multiple_han"
        else:
            kind = "other_or_mixed"
        rows.append({"id": token, "text": text, "class": kind, "special": special.get(token)})
    return rows


def character_counts(text):
    return Counter({"characters": len(text),
                    "han": sum(map(is_han, text)),
                    "punctuation": sum(map(is_punctuation, text)),
                    "whitespace": sum(char.isspace() for char in text),
                    "newline": text.count("\n"),
                    "replacement": text.count("\ufffd")})


def token_summary(counts, vocab):
    total = int(counts.sum())
    ranking = sorted(np.flatnonzero(counts).tolist(), key=lambda token: (-counts[token], token))
    classes = Counter()
    for row, count in zip(vocab, counts.tolist(), strict=True):
        classes[row["class"]] += count
    def stat(count):
        return {"count": int(count), "fraction": float(count / total) if total else None}
    return {"tokens": total, "classes": {key: stat(value) for key, value in classes.items()},
            "top_100": [{**vocab[token], **stat(counts[token])} for token in ranking[:100]],
            "individual": [{**row, **stat(counts[row["id"]])} for row in vocab
                           if row["special"] or row["text"] in ("，", ",", "。", ".", "\n", "\n\n")]}


def token_distributions(dataset, data, index, windows, tokenizer, vocab):
    """Re-encode the full split and verify that selected documents reproduce the binary."""
    dataset = Path(dataset)
    size = tokenizer.vocab_size
    scopes = ("full_train_raw", "frozen_raw", "frozen_supervised", "consumed_supervised")
    bins = {key: {} for key in scopes}
    chars = {key: Counter() for key in scopes}
    def update(scope, category, ids, text):
        counts = bins[scope].setdefault(category, np.zeros(size, dtype=np.int64))
        counts += np.bincount(ids, minlength=size)
        chars[scope].update(character_counts(text))

    consumed = np.zeros(len(data), dtype=bool)
    consumed[windows] = True
    doc_ids = set()
    order = []
    boundaries = {"documents": 0, "matching_terminal_eos": 0, "interior_special_tokens": Counter()}
    for row in records(dataset / "train.text.jsonl"):
        if row["split"] != "train" or row["document_id"] in doc_ids:
            raise ValueError("full Train split contains a duplicate or non-Train row")
        doc_ids.add(row["document_id"])
        ids = tokenizer.encode_document(row["text"], category=row["language"])
        order.append({"category": row["category"], "tokens": len(ids), "document_id": row["document_id"]})
        update("full_train_raw", row["category"], ids, tokenizer.decode(ids))
    position = 0
    seen = set()
    special = {value: key for key, value in tokenizer.special_ids.items()}
    for row, entry in zip(records(dataset / "train.selected.jsonl"), index, strict=True):
        if any(row[key] != entry[key] for key in ("document_id", "category", "language", "source_id")):
            raise ValueError("frozen selection does not match token index")
        doc_id = row["document_id"]
        if row["split"] != "train" or doc_id not in doc_ids or doc_id in seen:
            raise ValueError("selected entry is not a unique Train document")
        seen.add(doc_id)
        ids = np.asarray(tokenizer.encode_document(row["text"], category=row["language"]), dtype=np.int64)
        end = position + entry["tokens"]
        if not np.array_equal(ids, data.tokens[position:end]):
            raise ValueError("frozen document does not reproduce packed tokens")
        boundaries["documents"] += 1
        boundaries["matching_terminal_eos"] += int(ids[-1] == tokenizer.special_ids["eos"])
        boundaries["interior_special_tokens"].update(special[token] for token in ids[:-1] if token in special)
        category = row["category"]
        update("frozen_raw", category, ids, tokenizer.decode(ids))
        for window in range(position // data.seq_len, min(len(data), (end - 1) // data.seq_len + 1)):
            start = max(position, window * data.seq_len + 1)
            stop = min(end, (window + 1) * data.seq_len)
            supervised = data.tokens[start:stop]
            decoded = tokenizer.decode(supervised)
            update("frozen_supervised", category, supervised, decoded)
            if consumed[window]:
                update("consumed_supervised", category, supervised, decoded)
        position = end
    if position != data.num_tokens:
        raise ValueError("token index does not cover frozen Train BIN")
    result = {}
    for scope, groups in bins.items():
        result[scope] = {"overall": token_summary(sum(groups.values()), vocab),
                         "by_category": {category: token_summary(groups.get(category, np.zeros(size, dtype=np.int64)), vocab)
                                         for category in V4_CATEGORIES},
                         "decoded_character_counts": dict(chars[scope]),
                         "decoded_character_fractions": {
                             key: value / chars[scope]["characters"] for key, value in chars[scope].items()
                             if key != "characters"}}
    return result, boundaries, order


def expansion_budget(order, ratios, seq_len, *, batch_size, grad_accum, seed, tolerance):
    """Use actual encoding lengths and the existing independent-window policy.

    The whole-document/seeded-prefix release must be rebuilt and audited
    separately. A supply ceiling alone cannot qualify an extension run.
    """
    raw = Counter()
    groups = {c: [] for c in V4_CATEGORIES if ratios[c] > 0}
    for row in order:
        category = row["category"]
        raw[category] += row["tokens"]
        if category in groups:
            groups[category].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: (identity([seed, row["document_id"]]), row["document_id"]))
    # Match category-block layout and document EOS accounting. The discarded
    # boundary positions depend on block placement, hence the separate raw bound.
    packed = supervision_by_category(
        ((c, raw[c]) for c in groups if raw[c]), seq_len)
    active = {c: round(ratio * 100) for c, ratio in ratios.items() if ratio > 0}
    effective_limits = {c: packed["by_category"][c] * 100 // percent for c, percent in active.items()}
    raw_limits = {c: raw[c] * 100 // percent for c, percent in active.items()}
    limiting = min(effective_limits, key=effective_limits.get)
    update_tokens = batch_size * grad_accum * (seq_len - 1)
    absolute_ceiling = min(raw_limits.values())

    def structural_preview(full_updates):
        required = full_updates * update_tokens
        planned_windows = full_updates * batch_size * grad_accum
        targets = {c: (required * percent + 99) // 100 for c, percent in active.items()}
        selected, cursor = [], 0
        for i, (category, rows) in enumerate(groups.items()):
            start = cursor - math.ceil(cursor / seq_len)
            last_category = i == len(groups) - 1
            for row in rows:
                cursor += row["tokens"]
                selected.append(row)
                end = ((cursor - 1) // seq_len) * seq_len if last_category else cursor
                if last_category and end < planned_windows * seq_len:
                    continue
                category_targets = end - math.ceil(end / seq_len) - start
                if category_targets >= targets[category]:
                    break
        packing = supervision_by_category(((r["category"], r["tokens"]) for r in selected), seq_len)
        frozen_counts = packing["by_category"]
        release_feasible = packing["windows"] >= planned_windows and all(
            frozen_counts[c] >= targets[c] and abs(frozen_counts[c] / packing["effective_tokens"] - ratios[c]) <= tolerance
            for c in active)
        sampled = None
        if release_feasible:
            sampler = EpochBatchSampler(packing["windows"], batch_size, seed)
            windows = [window for batch in islice(sampler.batches(0), full_updates * grad_accum) for window in batch]
            sampled = category_consumption(selected, seq_len, windows)
            release_feasible = len(windows) == planned_windows and all(
                abs(sampled["categories"][c]["consumed_share"] - ratios[c]) <= tolerance for c in active)
        return {"status": "STRUCTURALLY_FEASIBLE_PENDING_APPROVAL" if release_feasible else "PACKING_INFEASIBLE",
                "updates": full_updates, "planned_effective_tokens": required, "target_by_category": targets,
                "frozen_candidate": packing,
                "selected_document_order_sha256": identity([r["document_id"] for r in selected]),
                "sampled_training_mix": sampled}

    # The absolute raw bound proves that the first feasible full-update candidate
    # below it is maximal for this whole-document selection and fixed sampler.
    maximum = None
    for updates in range(absolute_ceiling // update_tokens, 0, -1):
        candidate = structural_preview(updates)
        if candidate["status"] == "STRUCTURALLY_FEASIBLE_PENDING_APPROVAL":
            maximum = candidate
            break
    requests = {}
    for budget in (2_000_000, 4_000_000):
        full_updates = math.ceil(budget / update_tokens)
        required = full_updates * update_tokens
        preview = structural_preview(full_updates)
        counts = preview["target_by_category"]
        requests[str(budget)] = {"rounded_effective_tokens": required, "full_updates": full_updates,
                                 "target_by_category": counts,
                                 "supply_gap_by_category": {c: max(counts[c] - packed["by_category"][c], 0) for c in active},
                                 "preview": preview,
                                 "approval": "PENDING_NEW_HUMAN_SCOPE_APPROVAL", "updates_run": 0}
    return {"allocation": ratios, "frozen_release_share_tolerance": tolerance,
            "raw_by_category": dict(raw), "supervised_in_full_category_block_pack": packed["by_category"],
            "limiting_category": limiting, "measured_supply_ceiling": effective_limits[limiting],
            "measured_supply_ceiling_aligned_to_full_updates": effective_limits[limiting] // update_tokens * update_tokens,
            "absolute_raw_supply_upper_bound": absolute_ceiling,
            "max_feasible_seeded_full_update_preview": maximum,
            "selection_policy": "whole-document category prefixes ordered by SHA256(seed, document ID)",
            "selection_and_sampler_seed": seed,
            "requests": requests, "status": "NOT_RUN",
            "approved_trainable_extension_tokens": 0,
            "scope": "inventory only; exact post-selection budgets need a new version, human source review and release gates"}
