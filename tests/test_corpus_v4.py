"""Corpus v4 configuration, source admission and end-to-end demo lifecycle."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from minillm.corpus_utils import CorpusError
from minillm.corpus_v4 import (CorpusV4Pipeline, build_registry, inventory, load_config,
                               status_report, validate_config, validate_split_isolation,
                               verify_frozen)

ROOT = Path(__file__).resolve().parents[1]


def _source(source_id, status="APPROVED", license_value="CC0-1.0", **extra):
    source = {
        "source_id": source_id, "source_name": f"Synthetic {source_id}",
        "source_uri": f"fixture://{source_id}", "license": license_value,
        "license_status": status, "license_evidence": "synthetic fixture evidence",
        "license_reviewer": "fixture-reviewer", "license_reviewed_at": "2025-05-17",
        "category": "zh_general", "language": "zh", "provenance_status": "VERIFIED",
        "paths": ["fixture.jsonl"], "family_policy": "document",
    }
    source.update(extra)
    return source


def test_ratios_and_targets_are_config_driven():
    config = load_config(ROOT / "configs" / "corpus_v4.yaml")
    ratios = config["targets"]["category_token_ratios"]
    assert set(ratios) == {"zh_general", "zh_encyclopedic", "zh_education", "en_general", "math_logic_code"}
    assert sum(ratios.values()) == pytest.approx(1.0)
    changed = copy.deepcopy(config)
    changed["targets"]["category_token_ratios"] = {
        "zh_general": 0.5, "zh_encyclopedic": 0.2, "zh_education": 0.1,
        "en_general": 0.1, "math_logic_code": 0.1}
    assert changed["targets"]["category_token_ratios"]["zh_general"] == 0.5
    broken = copy.deepcopy(config)
    broken["targets"]["category_token_ratios"]["en_general"] = 0.5
    with pytest.raises(CorpusError, match="sum to one"):
        validate_config(broken)


def test_source_admission_is_explicit_and_never_license_string_inference(tmp_path):
    approved = _source("approved")
    review = _source("review", "REVIEW_REQUIRED", license_reviewer="", license_reviewed_at="")
    rejected = _source("rejected", "REJECTED", license_value="CC-BY-NC-ND-4.0")
    unverified = _source("unverified", "REVIEW_REQUIRED", license_value="LICENSE_UNVERIFIED",
                         license_reviewer="", license_reviewed_at="")
    registry = build_registry([approved, review, rejected, unverified], demo=False,
                              license_policy={"approved_only": True})
    assert [source["source_id"] for source in registry["sources"]] == ["approved"]
    reasons = {source["source_id"]: source["reason"] for source in registry["excluded"]}
    assert reasons == {"review": "license_review_required", "rejected": "license_rejected",
                       "unverified": "license_review_required"}
    with pytest.raises(CorpusError, match="unverified licence"):
        build_registry([_source("bad", "APPROVED", license_value="LICENSE_UNVERIFIED")],
                       demo=False, license_policy={"approved_only": True})


def test_formal_provenance_and_missing_approval_block_a_build(tmp_path):
    config = load_config(ROOT / "configs" / "corpus_v4.yaml")
    report = status_report(config, ROOT, demo=False)
    assert report["status"] == "NOT_READY"
    assert "NO_APPROVED_SOURCES" in report["blockers"]
    # An APPROVED licence with unverified provenance is still not admissible.
    source = _source("unverified-provenance", provenance_status="UNVERIFIED")
    registry = build_registry([source], demo=False, license_policy={"approved_only": True})
    assert not registry["sources"]
    assert registry["excluded"][0]["reason"] == "provenance_unverified"


def test_demo_end_to_end_is_honestly_demo_only(v4_demo):
    stats = v4_demo.ingest()
    assert stats["documents"] > 0
    v4_demo.quality()
    v4_demo.deduplicate()
    v4_demo.split()
    v4_demo.train_tokenizers()
    evaluation = v4_demo.evaluate_tokenizers()
    assert evaluation["selection"]["mode"] == "AUTO_DEMO_ONLY"
    v4_demo.tokenize()
    v4_demo.contamination()
    v4_demo.budget()
    audit = v4_demo.audit()
    manifest = v4_demo.finalize(audit=audit)
    assert manifest["dataset_status"] == "DEMO_ONLY"
    assert manifest["freeze_status"] == "NOT_READY"
    assert (v4_demo.out / "DEMO_ONLY").is_file()
    assert not (v4_demo.out / "FROZEN").exists()
    assert audit["status"] == "NOT_READY"
    assert manifest["content_id"] == audit["checks"]["manifest_completeness"]["detail"]["content_id"]
    with pytest.raises(CorpusError, match="not formally frozen"):
        verify_frozen(v4_demo.out)


def test_existing_nonempty_output_is_never_overwritten(tmp_path):
    out = tmp_path / "occupied"
    out.mkdir()
    (out / "keep.txt").write_text("old artifact", encoding="utf-8")
    config = load_config(ROOT / "configs" / "corpus_v4.yaml")
    source = _source("approved")
    registry = build_registry([source], demo=True, license_policy=config["license_policy"])
    with pytest.raises(CorpusError, match="nonempty output"):
        CorpusV4Pipeline(out, config, registry, [{"source_id": "approved", "path": "x",
                                                  "bytes": 1, "sha256": "0" * 64}], demo=True)


def test_missing_source_fails_inventory_before_ingest(tmp_path):
    source = _source("approved", paths=["missing.jsonl"])
    registry = build_registry([source], demo=True, license_policy={"approved_only": False})
    with pytest.raises(CorpusError, match="missing source file"):
        inventory(registry, tmp_path)


def _complete_leakage():
    return {
        "exact_normalized": {"train/val": 0, "train/test": 0, "val/test": 0},
        "exact_raw": {"train/val": 0, "train/test": 0, "val/test": 0},
        "family_cluster_violations": 0,
        "near_duplicate": {"status": "CHECKED", "hits": 0, "max_similarity": 0.0,
                           "candidate_truncations": 0, "truncated_chunks": 0,
                           "candidate_queries": 3, "candidate_comparisons": 0,
                           "threshold": 0.85},
    }


def test_split_isolation_requires_complete_evidence():
    leakage = _complete_leakage()
    assert validate_split_isolation(leakage) == "PASS"
    assert validate_split_isolation(None) == "NOT_ASSESSED"
    assert validate_split_isolation({}) == "NOT_ASSESSED"
    assert validate_split_isolation([]) == "FAIL"


@pytest.mark.parametrize("field", ["hits", "candidate_truncations", "truncated_chunks",
                                    "candidate_queries", "candidate_comparisons"])
@pytest.mark.parametrize("value", [-1, "0", 0.0, True, None])
def test_split_isolation_rejects_invalid_near_counts(field, value):
    leakage = _complete_leakage()
    leakage["near_duplicate"][field] = value
    assert validate_split_isolation(leakage) == "FAIL"


@pytest.mark.parametrize("field,value", [
    ("max_similarity", float("nan")), ("max_similarity", float("inf")),
    ("max_similarity", -0.1), ("max_similarity", 1.1), ("max_similarity", False),
    ("threshold", 0), ("threshold", 1.1), ("threshold", "0.85"),
])
def test_split_isolation_rejects_invalid_similarity_statistics(field, value):
    leakage = _complete_leakage()
    leakage["near_duplicate"][field] = value
    assert validate_split_isolation(leakage) == "FAIL"


@pytest.mark.parametrize("field", ["exact_normalized", "exact_raw", "family_cluster_violations"])
@pytest.mark.parametrize("value", [1, -1, True, "0"])
def test_split_isolation_rejects_exact_or_family_violations(field, value):
    leakage = _complete_leakage()
    if field == "family_cluster_violations":
        leakage[field] = value
    else:
        leakage[field]["train/val"] = value
    assert validate_split_isolation(leakage) == "FAIL"


@pytest.mark.parametrize("field", ["exact_normalized", "exact_raw", "family_cluster_violations",
                                    "near_duplicate"])
def test_split_isolation_rejects_missing_evidence(field):
    leakage = _complete_leakage()
    del leakage[field]
    assert validate_split_isolation(leakage) == "NOT_ASSESSED"
