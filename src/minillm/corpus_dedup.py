"""Near-duplicate retrieval with a documented LSH error model (CPU only).

The v4 pipeline never compares all document pairs.  Each document is split into
bounded chunks at paragraph boundaries; every chunk gets a *one-permutation
MinHash* (OPH) signature over language-aware shingles.  Signatures are indexed
in SQLite by LSH bands.  Candidate pairs must then pass exact Jaccard
confirmation on the full chunk shingle sets, so the reported similarity is an
exact set similarity, not a signature estimate.

Error model
-----------
With ``b`` bands of ``r`` rows, two chunks with Jaccard similarity ``J`` share a
bucket with probability ``1 - (1 - J**r)**b`` (standard MinHash-LSH analysis,
the same curve applies to densified OPH).  ``candidate_probability`` exposes
that value; the default ``b=32, r=4`` gives ~1.0 recall at J=0.85 and ~0.87 at
J=0.5.  Low bands may retrieve unrelated candidates; the exact-Jaccard step is
mandatory and configurable thresholds are applied there.  This is a bounded,
deterministic candidate-retrieval design, not an O(N**2) all-pairs comparison.

All hashing uses BLAKE2b with an explicit seed, so signatures, band buckets and
cluster representatives are reproducible across processes and input order.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass

import numpy as np

from .corpus_quality import CJK, LATIN, canonical, fragments
from .corpus_utils import identity

DEDUP_METHODS = ("exact", "minhash_lsh")
_WORD = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]")


@dataclass(frozen=True)
class NearDupPolicy:
    method: str = "minhash_lsh"
    threshold: float = 0.85
    num_bins: int = 128
    bands: int = 32
    rows: int = 4
    seed: int = 42
    zh_ngram: int = 5
    en_ngram: int = 3
    code_ngram: int = 8
    include_code: bool = False
    max_shingles_per_chunk: int = 20000
    max_candidates_per_chunk: int = 512
    chunk_chars: int = 20000

    def __post_init__(self):
        if self.method not in DEDUP_METHODS:
            raise ValueError(f"unknown deduplication method: {self.method}")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("deduplication threshold must be in (0, 1]")
        if self.num_bins < 8 or self.num_bins > 4096 or self.num_bins & (self.num_bins - 1):
            raise ValueError("num_bins must be a power of two in [8, 4096]")
        if self.bands <= 0 or self.rows <= 0 or self.bands * self.rows != self.num_bins:
            raise ValueError("bands * rows must equal num_bins")
        if not 2 <= self.zh_ngram <= 16 or not 1 <= self.en_ngram <= 6 or not 2 <= self.code_ngram <= 32:
            raise ValueError("invalid shingle sizes")
        if self.max_shingles_per_chunk < 16 or self.max_candidates_per_chunk < 1:
            raise ValueError("invalid shingle/candidate budget")
        if not 256 <= self.chunk_chars <= 2_000_000:
            raise ValueError("chunk_chars must be in [256, 2,000,000]")
        if type(self.include_code) is not bool:
            raise ValueError("include_code must be boolean")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")

    @classmethod
    def from_config(cls, config: dict | None) -> "NearDupPolicy":
        config = dict(config or {})
        known = {field for field in cls.__dataclass_fields__}
        unknown = set(config) - known
        if unknown:
            raise ValueError(f"unknown deduplication fields: {sorted(unknown)}")
        return cls(**config)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def policy_sha256(self) -> str:
        return identity({"version": "near-dedup-minhash-oph-v1", **self.to_dict()})


def candidate_probability(jaccard: float, bands: int, rows: int) -> float:
    """Probability that two chunks with this Jaccard share at least one band."""
    if not 0.0 <= jaccard <= 1.0:
        raise ValueError("jaccard must be in [0, 1]")
    return 1.0 - (1.0 - jaccard ** rows) ** bands


def _hash64(text: str, seed: int) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8,
                                          key=str(seed).encode("ascii")).digest(), "big")


def shingles(text: str, language: str, policy: NearDupPolicy) -> tuple[frozenset[int], bool]:
    """Language-aware shingles, bounded per chunk.

    Chinese/mixed text uses whitespace-free character n-grams, English uses word
    n-grams, and code uses case-sensitive character n-grams.  Code is excluded
    from near-duplicate retrieval by default; exact normalized dedup still runs.
    """
    if language == "code":
        if not policy.include_code:
            return frozenset(), False
        normalized = re.sub(r"\s+", " ", text)
        n = policy.code_ngram
    elif language == "en":
        words = _WORD.findall(canonical(text, "en").casefold())
        n = policy.en_ngram
        normalized = " ".join(words)
    else:
        normalized = re.sub(r"\s+", "", canonical(text, language)).casefold()
        n = policy.zh_ngram
    if len(normalized) < n:
        grams = [normalized] if normalized else []
    elif language == "en":
        tokens = normalized.split(" ")
        grams = [" ".join(tokens[i:i + n]) for i in range(max(1, len(tokens) - n + 1))]
    else:
        grams = [normalized[i:i + n] for i in range(len(normalized) - n + 1)]
    hashed = sorted({_hash64(gram, policy.seed) for gram in grams})
    truncated = len(hashed) > policy.max_shingles_per_chunk
    if truncated:
        # Deterministic, evenly spaced subsample; the truncation is reported so
        # a build can require review instead of silently claiming full coverage.
        step = len(hashed) / policy.max_shingles_per_chunk
        hashed = [hashed[int(index * step)] for index in range(policy.max_shingles_per_chunk)]
    return frozenset(hashed), truncated


def oph_signature(values: frozenset[int], num_bins: int) -> tuple[int, ...]:
    """One-permutation MinHash with deterministic gap densification."""
    if not values:
        return tuple(0 for _ in range(num_bins))
    shift = num_bins.bit_length() - 1
    mask = num_bins - 1
    minima: dict[int, int] = {}
    for value in values:
        bucket = value & mask
        candidate = value >> shift
        current = minima.get(bucket)
        if current is None or candidate < current:
            minima[bucket] = candidate
    filled: list[int | None] = [minima.get(bucket) for bucket in range(num_bins)]
    for bucket in range(num_bins):
        if filled[bucket] is None:
            for step in range(1, num_bins + 1):
                value = minima.get((bucket + step) % num_bins)
                if value is not None:
                    filled[bucket] = value
                    break
            else:
                filled[bucket] = 0
    return tuple(int(value) for value in filled)


def band_keys(signature: tuple[int, ...], bands: int, rows: int) -> list[tuple[int, str]]:
    if bands * rows != len(signature):
        raise ValueError("signature length must equal bands * rows")
    return [(band, ",".join(f"{value:016x}" for value in signature[band * rows:(band + 1) * rows]))
            for band in range(bands)]


def estimate_similarity(signature_a: tuple[int, ...], signature_b: tuple[int, ...]) -> float:
    if len(signature_a) != len(signature_b):
        raise ValueError("signatures must have equal length")
    return sum(a == b for a, b in zip(signature_a, signature_b)) / max(len(signature_a), 1)


def exact_jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    return intersection / max(len(a) + len(b) - intersection, 1)


def iter_chunks(text: str, limit: int):
    """Deterministic (index, chunk) pairs shared by indexing and confirmation."""
    index = 0
    for chunk in fragments(text, limit):
        if not chunk.strip():
            continue
        yield index, chunk
        index += 1


def chunk_features(text: str, language: str, policy: NearDupPolicy,
                   chunk_chars: int | None = None):
    """Yield (chunk_index, chunk_text, shingle_set, signature, truncated)."""
    limit = chunk_chars or policy.chunk_chars
    for index, chunk in iter_chunks(text, limit):
        values, truncated = shingles(chunk, language, policy)
        if not values:
            continue
        yield index, chunk, values, oph_signature(values, policy.num_bins), truncated


def chunk_text(text: str, chunk_index: int, policy: NearDupPolicy) -> str | None:
    """Select the same deterministic chunk without materialising earlier ones."""
    for index, chunk in iter_chunks(text, policy.chunk_chars):
        if index == chunk_index:
            return chunk
    return None


class MinHashLSHIndex:
    """Disk-backed signature index; memory use stays bounded by one document."""

    def __init__(self, db, policy: NearDupPolicy, prefix: str = "near_dedup"):
        if not prefix.isidentifier():
            raise ValueError("invalid index prefix")
        self.db, self.policy, self.prefix = db, policy, prefix
        self.db.executescript(f"""
            CREATE TABLE IF NOT EXISTS {prefix}_chunks(
                document_id TEXT, chunk_index INTEGER, chars INTEGER, signature BLOB,
                truncated INTEGER, PRIMARY KEY(document_id, chunk_index));
            CREATE TABLE IF NOT EXISTS {prefix}_buckets(
                band INTEGER, bucket TEXT, document_id TEXT, chunk_index INTEGER,
                PRIMARY KEY(band, bucket, document_id, chunk_index));
            CREATE INDEX IF NOT EXISTS {prefix}_bucket_lookup ON {prefix}_buckets(band, bucket);
        """)
        self.stats = {"chunks": 0, "buckets": 0, "candidate_queries": 0,
                      "candidate_comparisons": 0, "candidate_truncations": 0,
                      "max_candidates_observed": 0, "truncated_chunks": 0,
                      "signature_estimated_comparisons": 0}

    def add(self, document_id: str, text: str, language: str):
        inserted = 0
        for index, chunk, values, signature, truncated in chunk_features(text, language, self.policy):
            blob = np.asarray(signature, dtype=">u8").tobytes()
            self.db.execute(f"INSERT OR REPLACE INTO {self.prefix}_chunks VALUES(?,?,?,?,?)",
                            (document_id, index, len(chunk), blob, int(truncated)))
            rows = [(band, bucket, document_id, index)
                    for band, bucket in band_keys(signature, self.policy.bands, self.policy.rows)]
            self.db.executemany(f"INSERT OR IGNORE INTO {self.prefix}_buckets VALUES(?,?,?,?)", rows)
            self.stats["chunks"] += 1
            self.stats["buckets"] += len(rows)
            self.stats["truncated_chunks"] += int(truncated)
            inserted += 1
        return inserted

    def candidates(self, document_id: str, chunk_index: int,
                   signature: tuple[int, ...]) -> list[tuple[str, int]]:
        found: set[tuple[str, int]] = set()
        for band, bucket in band_keys(signature, self.policy.bands, self.policy.rows):
            for row in self.db.execute(
                    f"SELECT document_id, chunk_index FROM {self.prefix}_buckets "
                    "WHERE band=? AND bucket=? AND NOT (document_id=? AND chunk_index=?)",
                    (band, bucket, document_id, chunk_index)):
                found.add((row[0], row[1]))
        ordered = sorted(found)
        self.stats["candidate_queries"] += 1
        self.stats["max_candidates_observed"] = max(self.stats["max_candidates_observed"], len(ordered))
        if len(ordered) > self.policy.max_candidates_per_chunk:
            self.stats["candidate_truncations"] += 1
            ordered = ordered[:self.policy.max_candidates_per_chunk]
        return ordered

    def signature(self, document_id: str, chunk_index: int) -> tuple[int, ...]:
        row = self.db.execute(
            f"SELECT signature FROM {self.prefix}_chunks WHERE document_id=? AND chunk_index=?",
            (document_id, chunk_index)).fetchone()
        if row is None:
            raise ValueError("missing indexed chunk")
        return tuple(int(value) for value in np.frombuffer(row[0], dtype=">u8"))

    def annotated(self) -> dict:
        return {**self.stats,
                "method": "one-permutation-minhash+lsh+exact-jaccard",
                "threshold": self.policy.threshold,
                "bands": self.policy.bands, "rows": self.policy.rows,
                "candidate_probability_at_threshold": round(candidate_probability(
                    self.policy.threshold, self.policy.bands, self.policy.rows), 12),
                "candidate_probability_at_0.5": round(candidate_probability(
                    0.5, self.policy.bands, self.policy.rows), 6)}


def document_pairs(text: str, language: str, policy: NearDupPolicy):
    """Convenience generator of (chunk_index, shingles) for tests and audits."""
    for index, _, values, _, _ in chunk_features(text, language, policy):
        yield index, values


def language_counts(text: str) -> dict:
    """Small diagnostic used by reports; not a classifier replacement."""
    return {"hanzi": len(CJK.findall(text)), "latin": len(LATIN.findall(text)),
            "characters": len(text), "words": max(1, len(text.split()))}


def estimate_tokens(text: str, chars_per_token: float) -> int:
    if not math.isfinite(chars_per_token) or chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return max(1, math.ceil(len(text) / chars_per_token))
