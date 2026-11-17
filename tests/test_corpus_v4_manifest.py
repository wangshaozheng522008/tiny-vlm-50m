"""Dataset manifest, effective-token budget and the READY/FROZEN gate."""
from __future__ import annotations

import copy
import json
import random
import sqlite3
from pathlib import Path

import pytest

from minillm.corpus_budget import packed_effective_tokens, plan, supervision_by_category
from minillm.corpus_utils import CorpusError, sha256, write_json
from minillm.corpus_v4 import (CorpusV4Pipeline, build_registry, inventory, load_config,
                               verify_frozen)

def _root():
    return Path(__file__).resolve().parents[1]


def _ratios():
    return {"zh_general": 0.40, "zh_encyclopedic": 0.25, "zh_education": 0.15,
            "en_general": 0.10, "math_logic_code": 0.10}


def _formal_pipeline(tmp_path, *, verify_near=True, near=None, extra_sources=(), split_ratios=None, configure=None):
    root = _root()
    config = load_config(root / "configs" / "corpus_v4.yaml")
    config["targets"]["effective_train_tokens"] = 2048
    config["targets"]["seq_len"] = 128
    config["tokenizer"]["candidates"] = [{"name": "fixture-320", "vocab_size": 320},
                                        {"name": "fixture-384", "vocab_size": 384}]
    config["tokenizer"]["selected"] = "fixture-320"
    config["tokenizer"]["output_root"] = str(tmp_path / "tokenizers")
    config["split"]["verify_near_cross_split"] = verify_near
    if split_ratios:
        config["split"]["ratios"] = split_ratios
    config["deduplication"]["near"].update(near or {})
    if configure:
        configure(config)
    manifest = json.loads((root / "tests" / "fixtures" / "corpus_v4" / "demo-sources.json").read_text(
        encoding="utf-8"))
    sources = []
    for source in manifest["sources"]:
        item = copy.deepcopy(source)
        item["provenance_status"] = "VERIFIED"
        if item["source_id"] == "fixture-zh-education":
            item["license_status"] = "APPROVED"
            item["license_reviewer"] = "fixture-reviewer"
            item["license_reviewed_at"] = "2025-05-17"
        sources.append(item)
    sources.extend(extra_sources)
    registry = build_registry(sources, demo=False, license_policy=config["license_policy"])
    inputs = inventory(registry, root / "tests" / "fixtures" / "corpus_v4")
    return CorpusV4Pipeline(tmp_path / "formal", config, registry, inputs, demo=False,
                            log=lambda *_: None)


def test_effective_token_math_and_insufficient_corpus():
    assert packed_effective_tokens(0, 128) == (0, 0)
    assert packed_effective_tokens(128, 128) == (0, 0)
    assert packed_effective_tokens(129, 128) == (127, 1)
    assert packed_effective_tokens(1001, 128) == (889, 7)
    by_category = supervision_by_category([("zh_general", 1001)], 128)
    assert by_category["effective_tokens"] == 889
    assert sum(by_category["by_category"].values()) == 889
    empty = {category: {"documents": 0, "effective_tokens": 0} for category in _ratios()}
    report = plan(empty, 100_000_000, _ratios(), seq_len=1024,
                  tokenizer_fingerprint="f" * 64, allow_repetition=False)
    assert report["status"] == "INSUFFICIENT_CORPUS"
    assert report["total_gap_effective_tokens"] == 100_000_000
    assert set(report["blocking_conditions"]) >= {"TOTAL_CORPUS_SHORTFALL", "INSUFFICIENT_CORPUS",
                                                  "REPETITION_NOT_ALLOWED"}
    no_tokenizer = plan(empty, 1, _ratios(), seq_len=1024, tokenizer_fingerprint=None)
    assert no_tokenizer["status"] == "NOT_ASSESSED"


def test_demo_manifest_has_all_required_fields_and_stage_hashes(v4_demo):
    v4_demo.deduplicate()
    v4_demo.split()
    v4_demo.train_tokenizers()
    v4_demo.evaluate_tokenizers()
    v4_demo.tokenize()
    v4_demo.contamination()
    v4_demo.budget()
    audit = v4_demo.audit()
    manifest = v4_demo.finalize(audit=audit)
    required = {"corpus_version", "source_manifest_sha256", "cleaning_policy_sha256",
                "dedup_policy_sha256", "split_manifest_sha256", "tokenizer_fingerprint",
                "text_policy", "train_bin_sha256", "val_bin_sha256", "test_bin_sha256",
                "effective_train_tokens", "category_token_distribution", "license_review_status",
                "contamination_audit_status", "freeze_status"}
    assert required <= set(manifest)
    assert manifest["freeze_status"] == "NOT_READY"
    assert manifest["dataset_status"] == "DEMO_ONLY"
    assert manifest["effective_train_tokens"] > 0
    assert set(manifest["category_token_distribution"]["effective"]) == set(_ratios())
    assert sum(manifest["category_token_distribution"]["effective"].values()) == manifest["effective_train_tokens"]
    assert all(manifest[f"{split}_bin_sha256"] for split in ("train", "val", "test"))
    assert manifest["identity"]["stage_hashes"]
    markers = {path.name for path in (v4_demo.out / ".v4").glob("*.stage.json")}
    assert {"ingested.stage.json", "quality.stage.json", "dedup.stage.json", "split.stage.json",
            "tokenizers.stage.json", "tokenizer_eval.stage.json", "tokenized.stage.json"} <= markers
    assert set(manifest["identity"]["stage_hashes"]) >= {"ingested", "quality", "dedup", "split",
                                                         "tokenizers", "tokenizer_eval", "tokenized"}
    tokenization = json.loads((v4_demo.out / "tokenization-report.json").read_text(encoding="utf-8"))
    distribution = tokenization["source_distribution"]
    assert distribution["totals"]["documents"] == sum(row["documents"] for row in distribution["sources"].values())
    assert distribution["totals"]["tokens"] == sum(row["tokens"] for row in distribution["sources"].values())
    for key in ("document_ratio", "character_ratio", "token_ratio"):
        assert sum(row[key] for row in distribution["sources"].values()) == pytest.approx(1.0)


def test_formal_ready_dataset_can_freeze_and_is_tamper_evident(tmp_path):
    pipeline = _formal_pipeline(tmp_path)
    manifest = pipeline.finalize()
    assert manifest["freeze_status"] == "READY"
    assert manifest["dataset_status"] == "FORMAL"
    assert manifest["license_review_status"] == "APPROVED"
    assert manifest["contamination_audit_status"] == "CHECKED_NO_HITS"
    assert manifest["budget_status"] == "READY_FOR_TRAINING"
    assert (pipeline.out / "FROZEN").is_file()
    check = pipeline.stats["audit"]["checks"]["split_isolation"]
    assert check["status"] == "PASS"
    assert check["detail"]["leakage"]["near_duplicate"]["status"] == "CHECKED"
    verified = verify_frozen(pipeline.out)
    assert verified["content_id"] == manifest["content_id"]
    with pytest.raises(CorpusError, match="already frozen"):
        pipeline.finalize()
    (pipeline.out / "split-manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CorpusError, match="SHA256/size mismatch"):
        verify_frozen(pipeline.out)


def _assert_cannot_freeze(pipeline, audit, expected):
    assert audit["checks"]["split_isolation"]["status"] == expected
    assert audit["status"] != "READY"
    assert "split_isolation" in audit["blockers"]
    manifest = pipeline.finalize(audit=audit)
    assert manifest["freeze_status"] != "READY"
    assert not (pipeline.out / "FROZEN").exists()
    assert not (pipeline.out / "freeze.json").exists()
    # Simulate an older finalizer accepting the same hash-valid artifacts.
    # The verifier must reject their evidence even with formal READY flags.
    manifest["freeze_status"] = "READY"
    write_json(pipeline.out / "dataset-manifest.json", manifest)
    write_json(pipeline.out / "freeze.json", {
        "schema_version": 1, "content_id": manifest["content_id"],
        "freeze_status": "READY", "dataset_status": "FORMAL",
        "manifest_sha256": sha256(pipeline.out / "dataset-manifest.json"),
        "corpus_version": manifest["corpus_version"],
        "effective_train_tokens": manifest["effective_train_tokens"],
    })
    write_json(pipeline.out / "FROZEN", {"content_id": manifest["content_id"]})
    with pytest.raises(CorpusError, match="split isolation evidence"):
        verify_frozen(pipeline.out)


@pytest.mark.parametrize("field,value,expected", [
    ("status", "NOT_ASSESSED", "NOT_ASSESSED"),
    ("status", "INCOMPLETE", "FAIL"), ("status", "FAILED", "FAIL"),
    ("status", "UNKNOWN", "FAIL"), ("status", None, "NOT_ASSESSED"),
    ("hits", 1, "FAIL"), ("candidate_truncations", 1, "FAIL"),
    ("truncated_chunks", 1, "FAIL"),
    ("hits", None, "NOT_ASSESSED"), ("candidate_truncations", None, "NOT_ASSESSED"),
    ("truncated_chunks", None, "NOT_ASSESSED"),
    ("candidate_queries", None, "NOT_ASSESSED"),
    ("candidate_comparisons", None, "NOT_ASSESSED"),
    ("max_similarity", None, "NOT_ASSESSED"), ("threshold", None, "NOT_ASSESSED"),
    ("candidate_truncations", -1, "FAIL"), ("truncated_chunks", "0", "FAIL"),
    ("hits", True, "FAIL"), ("candidate_queries", 0.0, "FAIL"),
])
def test_split_evidence_blocks_audit_freeze_and_verification(tmp_path, monkeypatch,
                                                            field, value, expected):
    pipeline = _formal_pipeline(tmp_path)
    scan = pipeline._cross_split_near

    def changed_report(scratch):
        report = scan(scratch)
        if value is None:
            del report[field]
        else:
            report[field] = value
        return report

    monkeypatch.setattr(pipeline, "_cross_split_near", changed_report)
    _assert_cannot_freeze(pipeline, pipeline.audit(), expected)


def test_disabled_cross_split_scan_cannot_freeze(tmp_path):
    pipeline = _formal_pipeline(tmp_path, verify_near=False)
    audit = pipeline.audit()
    near = audit["checks"]["split_isolation"]["detail"]["leakage"]["near_duplicate"]
    assert near["status"] == "NOT_ASSESSED"
    assert near["candidate_queries"] == near["candidate_comparisons"] == 0
    _assert_cannot_freeze(pipeline, audit, "NOT_ASSESSED")


@pytest.mark.parametrize("field", ["leakage", "near_duplicate"])
def test_missing_cross_split_evidence_cannot_freeze(tmp_path, monkeypatch, field):
    pipeline = _formal_pipeline(tmp_path)
    scan = pipeline._split_leakage

    def missing_evidence(*args, **kwargs):
        report = scan(*args, **kwargs)
        if field == "leakage":
            return {}
        del report[field]
        return report

    monkeypatch.setattr(pipeline, "_split_leakage", missing_evidence)
    _assert_cannot_freeze(pipeline, pipeline.audit(), "NOT_ASSESSED")


def test_real_feature_truncations_prevent_freeze(tmp_path):
    pipeline = _formal_pipeline(tmp_path, near={"max_shingles_per_chunk": 16})
    audit = pipeline.audit()
    near = audit["checks"]["split_isolation"]["detail"]["leakage"]["near_duplicate"]
    assert near["status"] == "INCOMPLETE"
    assert near["truncated_chunks"] > 0
    _assert_cannot_freeze(pipeline, audit, "FAIL")


def test_configuration_change_cannot_upgrade_unassessed_scan(tmp_path):
    pipeline = _formal_pipeline(tmp_path, verify_near=False)
    audit = pipeline.audit()
    pipeline.config["split"]["verify_near_cross_split"] = True
    with pytest.raises(CorpusError, match="configuration changed"):
        pipeline.finalize(audit=audit)
    with pytest.raises(CorpusError, match="configuration changed"):
        pipeline.audit()
    assert not (pipeline.out / "FROZEN").exists()


def test_ready_flag_cannot_override_unassessed_evidence(tmp_path):
    pipeline = _formal_pipeline(tmp_path, verify_near=False)
    audit = pipeline.audit()
    forged = copy.deepcopy(audit)
    forged["status"] = "READY"
    with pytest.raises(CorpusError, match="published audit report"):
        pipeline.finalize(audit=forged)
    write_json(pipeline.out / "audit-report.json", forged)
    with pytest.raises(CorpusError, match="passing validation evidence"):
        pipeline.finalize(audit=forged)
    assert not (pipeline.out / "FROZEN").exists()


def test_split_evidence_changed_after_audit_cannot_freeze(tmp_path):
    pipeline = _formal_pipeline(tmp_path)
    audit = pipeline.audit()
    path = pipeline.out / "split-manifest.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["leakage"]["near_duplicate"]["status"] = "INCOMPLETE"
    write_json(path, report)
    with pytest.raises(CorpusError, match="evidence changed since audit"):
        pipeline.finalize(audit=audit)
    assert not (pipeline.out / "FROZEN").exists()


def test_dataset_identity_changed_after_audit_cannot_freeze(tmp_path):
    pipeline = _formal_pipeline(tmp_path)
    audit = pipeline.audit()
    # A file outside split evidence is still bound by the audited content_id.
    path = pipeline.out / "budget-report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["test_extra_evidence"] = True
    write_json(path, report)
    with pytest.raises(CorpusError, match="identity changed since audit"):
        pipeline.finalize(audit=audit)
    assert not (pipeline.out / "FROZEN").exists()


def test_real_lsh_candidate_truncations_reach_split_audit(tmp_path, monkeypatch):
    path = tmp_path / "collisions.jsonl"
    rng = random.Random(42)
    common = " ".join(f"sharedword{chr(97 + i)}" for i in range(26))
    texts = []
    for _ in range(40):
        suffix = " ".join("".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=10))
                          for _ in range(10))
        texts.append({"text": common + " " + suffix})
    path.write_text("".join(json.dumps(row) + "\n" for row in texts), encoding="utf-8")
    source = json.loads((_root() / "tests/fixtures/corpus_v4/demo-sources.json").read_text())["sources"][4]
    source.update(source_id="lsh-collision-fixture", paths=[str(path)], provenance_status="VERIFIED")
    pipeline = _formal_pipeline(tmp_path, near={"max_candidates_per_chunk": 1},
                                extra_sources=[source],
                                split_ratios={"train": 0.5, "val": 0.25, "test": 0.25})
    pipeline.deduplicate()
    from minillm.corpus_dedup import MinHashLSHIndex
    observed = {"queries": 0, "truncations": 0}
    candidates = MinHashLSHIndex.candidates

    def observe(self, *args):
        before = self.stats["candidate_truncations"]
        result = candidates(self, *args)
        observed["queries"] += 1
        observed["truncations"] += self.stats["candidate_truncations"] - before
        assert len(result) <= 1
        return result

    monkeypatch.setattr(MinHashLSHIndex, "candidates", observe)
    audit = pipeline.audit()
    near = audit["checks"]["split_isolation"]["detail"]["leakage"]["near_duplicate"]
    assert observed["truncations"] > 0
    assert near["candidate_truncations"] == observed["truncations"]
    assert near["candidate_queries"] == observed["queries"]
    assert near["candidate_comparisons"] > 0
    assert near["hits"] == 0
    assert near["truncated_chunks"] == 0
    assert near["status"] == "INCOMPLETE"
    _assert_cannot_freeze(pipeline, audit, "FAIL")


@pytest.mark.parametrize("long_split,expected", [("train", 1), ("val", 2), ("test", 2)])
def test_cross_split_feature_truncations_include_index_and_query(tmp_path, long_split, expected):
    pipeline = _formal_pipeline(tmp_path, near={"max_shingles_per_chunk": 16})
    # Minimal real source/split tables isolate each phase's feature truncation.
    long_text = "".join(chr(0x4E00 + i) for i in range(100))
    short_text = "这是用于验证跨集合特征统计的短文本"
    scratch = pipeline.work / "feature-test.sqlite"
    with sqlite3.connect(pipeline.out / "deduplicated.sqlite") as source, sqlite3.connect(scratch) as meta:
        source.execute("CREATE TABLE docs(document_id TEXT, normalized_text TEXT, "
                       "detected_language TEXT, dedup_kept INTEGER)")
        meta.execute("CREATE TABLE doc_meta(document_id TEXT, split TEXT, "
                     "normalized_sha256 TEXT, content_sha256 TEXT)")
        for name in ("train", "val", "test"):
            source.execute("INSERT INTO docs VALUES(?,?,?,1)",
                           (name, long_text if name == long_split else short_text, "zh"))
            meta.execute("INSERT INTO doc_meta VALUES(?,?,?,?)", (name, name, name, name))
    near = pipeline._split_leakage(scratch)["near_duplicate"]
    assert near["status"] == "INCOMPLETE"
    assert near["truncated_chunks"] == expected
    assert near["candidate_queries"] == 3
    assert near["candidate_truncations"] == 0


def test_cross_split_scan_exception_is_never_safe(tmp_path, monkeypatch):
    pipeline = _formal_pipeline(tmp_path)
    from minillm.corpus_dedup import MinHashLSHIndex
    pipeline.deduplicate()

    def fail(*_args):
        raise RuntimeError("cross-split query failed")

    monkeypatch.setattr(MinHashLSHIndex, "candidates", fail)
    with pytest.raises(CorpusError, match="stage split FAILED: cross-split query failed"):
        pipeline.finalize()
    assert not (pipeline.out / "FROZEN").exists()
    assert not (pipeline.out / "split-manifest.json").exists()
    assert not (pipeline.work / "cross-split-near.sqlite").exists()
