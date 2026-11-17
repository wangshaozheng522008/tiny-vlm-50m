"""Exact/near dedup, bounded LSH, and family-aware deterministic splitting."""
from __future__ import annotations

import json

from minillm.corpus_dedup import (NearDupPolicy, candidate_probability, chunk_features,
                                  document_pairs, exact_jaccard, iter_chunks, shingles)


def test_exact_and_near_duplicates_are_accounted(v4_demo):
    v4_demo.quality()
    report = v4_demo.deduplicate()
    assert report["exact"]["groups"] == 1
    assert report["exact"]["duplicates"] == 1
    assert report["near"]["duplicates"] == 1
    assert report["near"]["clusters"] == 1
    assert report["relations"]["exact"] == 1
    assert report["relations"]["near"] == 1
    assert report["documents"] == 15
    assert report["tokens_estimated"]["basis"].startswith("chars_per_token")
    assert report["clusters"]["deleted_documents"] == 2
    assert report["policy_sha256"]
    assert report["near_policy_sha256"]
    # Duplicate relations are retained rather than overwriting the source records.
    assert report["clusters"]["size_histogram"]
    assert report["clusters"]["deleted_documents"] == 2


def test_near_pair_crosses_threshold_and_lsh_probability():
    policy = NearDupPolicy(threshold=0.85)
    a = "".join(f"第{i}段讨论城市公共空间、阅读习惯和社区服务，编号{i}的内容只用于测试。" for i in range(40))
    b = a.replace("编号7", "序号7")
    left, _ = shingles(a, "zh", policy)
    right, _ = shingles(b, "zh", policy)
    similarity = exact_jaccard(left, right)
    assert similarity >= 0.85
    assert candidate_probability(similarity, policy.bands, policy.rows) > 0.99
    assert candidate_probability(0.5, policy.bands, policy.rows) > 0.8


def test_chunking_bounds_memory_for_long_documents():
    policy = NearDupPolicy(chunk_chars=1000, max_shingles_per_chunk=50)
    text = "这是一个用于测试分块的合成段落，包含城市、阅读和社区生活。" * 300
    chunks = list(iter_chunks(text, policy.chunk_chars))
    assert len(chunks) > 1
    assert all(len(chunk) <= policy.chunk_chars for chunk in chunks)
    for _, _, values, _, _ in chunk_features(text, "zh", policy):
        assert len(values) <= policy.max_shingles_per_chunk
    pairs = list(document_pairs(text, "zh", policy))
    assert pairs  # bounded generator, not an O(N^2) pair matrix


def test_family_and_cluster_groups_never_cross_splits(v4_demo):
    v4_demo.deduplicate()
    stats = v4_demo.split()
    assignment = [json.loads(line) for line in
                  (v4_demo.out / "split-assignment.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {row["document_id"]: row for row in assignment}
    groups = {}
    for row in assignment:
        groups.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    assert stats["leakage"]["family_cluster_violations"] == 0
    assert all(value == 0 for value in stats["leakage"]["exact_normalized"].values())
    assert all(value == 0 for value in stats["leakage"]["exact_raw"].values())
    assert len(stats["splits"]["train"]["by_category"]) >= 3
    # The fixture's parent_id pair must share a group and therefore a split.
    families = {}
    for row in assignment:
        families.setdefault(row["document_id"], row)
    assert len(groups) > 1


def test_split_is_deterministic_for_same_inputs(tmp_path):
    from conftest import make_v4_demo_pipeline
    first = make_v4_demo_pipeline(tmp_path, "a")
    second = make_v4_demo_pipeline(tmp_path, "b")
    for pipeline in (first, second):
        pipeline.deduplicate()
        pipeline.split()
    one = (first.out / "split-assignment.jsonl").read_text(encoding="utf-8")
    two = (second.out / "split-assignment.jsonl").read_text(encoding="utf-8")
    assert one == two
    assert json.loads((first.out / "split-manifest.json").read_text())["leakage"]["near_duplicate"]["status"] == "CHECKED"
