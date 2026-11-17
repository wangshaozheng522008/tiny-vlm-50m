"""Scale, batching and benchmark regression tests for corpus v4.

They intentionally use only small synthetic runs; the 100k/1m stress tests are
explicit CLI invocations documented in docs/corpus-v4-performance.md and are not
part of normal pytest/CI.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from conftest import make_v4_demo_pipeline
from minillm.corpus_benchmark import generate_synthetic_corpus, run_benchmark
from minillm.corpus_utils import CorpusError, sha256
from minillm.corpus_v4 import CorpusV4Pipeline, build_registry, inventory, load_config


def _pipeline_from_generated(tmp_path, documents=300, seed=7):
    input_dir = tmp_path / "input"
    generation = generate_synthetic_corpus(input_dir, documents, seed=seed)
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "corpus_v4.yaml")
    raw = json.loads((input_dir / generation["manifest"]).read_text(encoding="utf-8"))
    registry = build_registry(raw["sources"], demo=True, license_policy=config["license_policy"])
    inputs = inventory(registry, input_dir)
    out = tmp_path / "out"
    pipeline = CorpusV4Pipeline(out, config, registry, inputs, demo=True, log=lambda *_: None)
    return pipeline, generation


# Captured from the pre-SQLite-staging implementation on the synthetic demo.
# Group ids, ordering and text rows stay byte-identical. Only the report hash
# changes for split-manifest schema v2's complete cross-split scan evidence.
REFERENCE_1000_SPLIT_HASHES = {
    "split-assignment.jsonl": "06a34c17f14df9fe4614a228018c3fa39e5aaeb395105f45a611a5d81a98c09e",
    "split-manifest.json": "1f3aa679ff086b93bf47b15b5544873ba91b1f5dbdec2ff2813ca362cccbbe76",
    "train.text.jsonl": "f8d3b9fdc90b76f408b4e573015b384afd90fdc143a1482b3b96a6e99b124eae",
    "val.text.jsonl": "291aa0015222c3ae82e42c8113509c76bc08326dcf437e16385e783dcfc584bc",
    "test.text.jsonl": "76292d0d74a5cc5106d7cb652071c614c818014532dd26bcaa2529b2595675e6",
}

REFERENCE_SPLIT_HASHES = {
    "split-assignment.jsonl": "5b359ae6f579c000908405723c308b33b55eb6930cae4bb985a230ec0c65085f",
    "split-manifest.json": "9322322f06c1fd0b809ffe082561459499c4389751c2a4e0a45ccc00df952f6d",
    "train.text.jsonl": "0368b2778ac525df6fe5135e1b45c46259cbc463c6e28d1ce52955a0822961a2",
    "val.text.jsonl": "8aee49902ccb26178c82265df2089b2891139be210a28519230bb58851a95f14",
    "test.text.jsonl": "7e06e92303d1d31b7e76e262e0f0cd91a49a1879de8f9d0a8e57bdcb00c0d73a",
}


def test_split_outputs_match_pre_refactor_reference_hashes(tmp_path):
    pipeline = make_v4_demo_pipeline(tmp_path)
    pipeline.ingest()
    pipeline.quality()
    pipeline.deduplicate()
    pipeline.split()
    for name, expected in REFERENCE_SPLIT_HASHES.items():
        assert sha256(pipeline.out / name) == expected, name
    # No temporary scratch or split files survive a successful run.
    assert not (pipeline.out / ".v4" / "work" / "split-scratch.sqlite").exists()
    assert not list(pipeline.out.glob(".*.text.jsonl.tmp"))
    assert not (pipeline.out / ".split-assignment.jsonl.tmp").exists()


def test_1000_document_split_matches_pre_refactor_reference(tmp_path):
    pipeline, _ = _pipeline_from_generated(tmp_path, documents=1000, seed=42)
    pipeline.ingest()
    pipeline.quality()
    pipeline.deduplicate()
    pipeline.split()
    for name, expected in REFERENCE_1000_SPLIT_HASHES.items():
        assert sha256(pipeline.out / name) == expected, name


def test_family_and_cluster_transitive_components_never_cross_splits(tmp_path):
    pipeline, generation = _pipeline_from_generated(tmp_path, documents=300)
    pipeline.ingest()
    pipeline.quality()
    dedup = pipeline.deduplicate()
    assert dedup["clusters"]["deleted_documents"] > 0
    assert generation["exact_duplicate_events"] > 0
    assert generation["near_duplicate_events"] > 0
    pipeline.split()
    assignment = {
        row["document_id"]: row["split"]
        for row in (json.loads(line) for line in
                    (pipeline.out / "split-assignment.jsonl").read_text(encoding="utf-8").splitlines())
    }
    assert len(assignment) == len(set(assignment)) == pipeline.stats["split"]["documents"]
    with sqlite3.connect(pipeline.out / "deduplicated.sqlite") as db:
        rows = list(db.execute(
            "SELECT document_id, document_family_id, dedup_cluster FROM docs "
            "WHERE dedup_kept=1 ORDER BY document_id"))
    parent = {}

    def find(item):
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for document_id, _, _ in rows:
        find(document_id)
    for column in (1, 2):
        first = {}
        for row in rows:
            first.setdefault(row[column], row[0])
            union(first[row[column]], row[0])
    components = {}
    for document_id, _, _ in rows:
        components.setdefault(find(document_id), set()).add(assignment[document_id])
    assert components
    assert all(len(splits) == 1 for splits in components.values())
    assert all(part["documents"] > 0 for part in pipeline.stats["split"]["splits"].values())


def test_split_repeatability_and_small_benchmark_report(tmp_path):
    first, generation = _pipeline_from_generated(tmp_path / "one", documents=120)
    first.ingest()
    first.quality()
    first.deduplicate()
    first.split()
    second, _ = _pipeline_from_generated(tmp_path / "two", documents=120)
    second.ingest()
    second.quality()
    second.deduplicate()
    second.split()
    assert sha256(first.out / "split-assignment.jsonl") == sha256(second.out / "split-assignment.jsonl")
    assert sha256(first.out / "split-manifest.json") == sha256(second.out / "split-manifest.json")
    assert sha256(first.out / "train.text.jsonl") == sha256(second.out / "train.text.jsonl")
    assert sha256(first.out / "val.text.jsonl") == sha256(second.out / "val.text.jsonl")
    assert sha256(first.out / "test.text.jsonl") == sha256(second.out / "test.text.jsonl")
    assert generation["input_sha256"] == json.loads(
        (tmp_path / "one" / "input" / "generation.json").read_text())["input_sha256"]


def test_split_failure_leaves_no_complete_outputs(tmp_path):
    pipeline = make_v4_demo_pipeline(tmp_path)
    pipeline.ingest()
    pipeline.quality()
    pipeline.deduplicate()

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected split failure after temporary files")

    pipeline._split_leakage = fail
    with pytest.raises(CorpusError, match="stage split FAILED"):
        pipeline.split()
    for name in ("train.text.jsonl", "val.text.jsonl", "test.text.jsonl",
                 "split-assignment.jsonl", "split-manifest.json"):
        assert not (pipeline.out / name).exists(), name
    assert not list(pipeline.out.glob(".*.tmp"))
    assert not (pipeline.out / ".v4" / "split.stage.json").exists()
    assert not (pipeline.out / ".v4" / "work" / "split-scratch.sqlite").exists()


def test_benchmark_writes_machine_readable_report(tmp_path):
    out = tmp_path / "bench"
    report = run_benchmark(Path(__file__).resolve().parents[1] / "configs" / "corpus_v4.yaml",
                           out, documents=120, seed=7, log=lambda *_: None)
    persisted = json.loads((out / "benchmark-report.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "COMPLETED"
    assert persisted["parameters"]["documents"] == 120
    assert persisted["corpus"]["category_counts"]["en_general"] == 10
    assert persisted["corpus"]["exact_duplicate_events"] > 0
    assert persisted["corpus"]["near_duplicate_events"] > 0
    assert persisted["metrics"]["retained_documents"] > 0
    assert persisted["metrics"]["deleted_documents"] > 0
    assert persisted["metrics"]["peak_rss_bytes"] > 0
    assert persisted["metrics"]["sqlite_bytes"] > 0
    assert set(persisted["metrics"]["stage_seconds"]) == {"ingest", "quality", "dedup", "split"}
    assert persisted["metrics"]["leakage"]["exact_normalized"] == {
        "train/test": 0, "train/val": 0, "val/test": 0}
    assert (out / "benchmark-summary.md").is_file()
    assert report["metrics"]["retained_documents"] == persisted["metrics"]["retained_documents"]
    assert not (out / "tokenizer.json").exists()
    assert not (out / "tokenizers").exists()


def test_benchmark_partial_stages_are_reported_as_not_run(tmp_path):
    out = tmp_path / "partial"
    run_benchmark(Path(__file__).resolve().parents[1] / "configs" / "corpus_v4.yaml",
                  out, documents=60, seed=11, stages=("ingest",), log=lambda *_: None)
    report = json.loads((out / "benchmark-report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PARTIAL"
    assert set(report["metrics"]["stage_seconds"]) == {"ingest"}
    assert report["metrics"]["not_run"] == ["quality", "dedup", "split"]
    assert report["metrics"]["retained_documents"] is None
    assert report["metrics"]["deleted_documents"] is None
    assert report["metrics"]["leakage"] is None
    assert "NOT_RUN" in (out / "benchmark-summary.md").read_text(encoding="utf-8")


def test_split_ignores_physical_database_row_order(tmp_path):
    pipeline, _ = _pipeline_from_generated(tmp_path, documents=120, seed=5)
    pipeline.ingest()
    pipeline.quality()
    pipeline.deduplicate()
    pipeline.split()
    baseline = {name: sha256(pipeline.out / name) for name in
                ("split-assignment.jsonl", "split-manifest.json", "train.text.jsonl")}
    with sqlite3.connect(pipeline.out / "deduplicated.sqlite") as db:
        db.execute("CREATE TABLE docs_reversed AS SELECT * FROM docs ORDER BY document_id DESC")
        db.execute("DROP TABLE docs")
        db.execute("ALTER TABLE docs_reversed RENAME TO docs")
        db.commit()
    pipeline._build_split()
    assert {name: sha256(pipeline.out / name) for name in baseline} == baseline


def test_5000_document_batch_boundary_loses_no_documents(tmp_path):
    pipeline, _ = _pipeline_from_generated(tmp_path, documents=5000, seed=42)
    pipeline.ingest()
    pipeline.quality()
    dedup = pipeline.deduplicate()
    pipeline.split()
    assignment = [json.loads(line) for line in
                  (pipeline.out / "split-assignment.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(assignment) == len({row["document_id"] for row in assignment})
    assert len(assignment) == dedup["documents"]
    assert sum(part["documents"] for part in pipeline.stats["split"]["splits"].values()) == dedup["documents"]
    assert all(part["documents"] > 0 for part in pipeline.stats["split"]["splits"].values())
    assert pipeline.stats["split"]["leakage"]["family_cluster_violations"] == 0


def test_synthetic_generator_is_deterministic_and_has_cross_split_inputs(tmp_path):
    first = generate_synthetic_corpus(tmp_path / "a", 120, seed=9)
    second = generate_synthetic_corpus(tmp_path / "b", 120, seed=9)
    assert first["input_sha256"] == second["input_sha256"]
    assert first["files"] == second["files"]
    assert first["exact_duplicate_events"] > 0
    assert first["near_duplicate_events"] > 0
    assert first["family_groups"] > 0
    assert set(first["category_counts"]) == {
        "zh_general", "zh_encyclopedic", "zh_education", "en_general", "math_logic_code"}
