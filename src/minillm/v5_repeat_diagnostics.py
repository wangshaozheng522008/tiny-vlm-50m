"""Generation length, punctuation loops and exact Train-document token overlap."""

from collections import defaultdict
from itertools import groupby
import math
import statistics

import numpy as np

from .language_token_analysis import is_punctuation

MIN_SPAN = 16
LONG_SPAN = 32
HASH_BASE = 65537


def _lengths(counts):
    ordered = sorted(counts)
    return {"min": ordered[0], "median": statistics.median(ordered),
            "p95_nearest_rank": ordered[math.ceil(.95 * len(ordered)) - 1], "max": ordered[-1],
            "mean": statistics.mean(ordered)}


def generation_diagnostics(rows):
    return {
        "outputs": len(rows),
        "content_token_length": _lengths([row["content_tokens"] for row in rows]),
        "nonspace_character_length": _lengths([row["diagnostics"]["nonspace_characters"] for row in rows]),
        "identical_punctuation_run_at_least_four": sum(
            any(is_punctuation(char) and sum(1 for _ in run) >= 4
                for char, run in groupby(row["decoded_text"])) for row in rows),
        "mean_sentence_repeat_ratio": statistics.mean(row["diagnostics"]["sentence_repeat_ratio"] for row in rows),
    }


def _hash(values):
    result = 0
    for value in values:
        result = (result * HASH_BASE + int(value)) & ((1 << 64) - 1)
    return result


def train_overlap(tokens, documents, rows, *, span=MIN_SPAN):
    """Match exact contiguous content tokens, verifying every hash hit within a document.

    Each result records the longest verified match of at least `span` tokens.
    Token positions are zero-based within the generated content and document.
    """
    if span <= 0:
        raise ValueError("minimum overlap span must be positive")
    ends = np.cumsum([row["tokens"] for row in documents], dtype=np.int64)
    if len(ends) == 0 or ends[-1] != len(tokens):
        raise ValueError("Train token index does not cover the packed corpus")
    outputs = [tuple(row["generated_token_ids"][:row["content_tokens"]]) for row in rows]
    queries = defaultdict(list)
    for seq in dict.fromkeys(outputs):
        for start in range(len(seq) - span + 1):
            seed = seq[start:start + span]
            queries[_hash(seed)].append((seq, start, seed))
    origins = {}
    keys = np.array(sorted(queries), dtype=np.uint64)
    if not len(keys):
        return [None] * len(outputs)
    for begin in range(0, len(tokens) - span + 1, 524288):
        stop = min(len(tokens) - span + 1, begin + 524288)
        hashes = np.zeros(stop - begin, dtype=np.uint64)
        for shift in range(span):
            hashes *= HASH_BASE
            hashes += tokens[begin + shift:stop + shift]
        matched = np.isin(hashes, keys, kind="sort").nonzero()[0] + begin
        if not len(matched):
            continue
        doc_ids = np.searchsorted(ends, matched, side="right")
        for pos, doc in zip(matched.tolist(), doc_ids.tolist(), strict=True):
            if pos + span > ends[doc]:
                continue
            actual = tuple(tokens[pos:pos + span].tolist())
            for seq, start, seed in queries[_hash(actual)]:
                if actual != seed or (seq in origins
                        and len(seq) - start <= origins[seq]["matched_tokens"]):
                    continue
                limit = min(len(seq) - start, ends[doc] - pos)
                length = span
                while length < limit and tokens[pos + length] == seq[start + length]:
                    length += 1
                if seq not in origins or length > origins[seq]["matched_tokens"]:
                    origins[seq] = {"matched_tokens": length, "document_id": documents[doc]["document_id"],
                                    "document_offset": pos - int(ends[doc - 1]) if doc else pos,
                                    "output_offset": start}
    return [origins.get(seq) for seq in outputs]
