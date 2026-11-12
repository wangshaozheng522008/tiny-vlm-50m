"""Dataset manifest, effective-token budget and the READY/FROZEN gate."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from minillm.corpus_budget import packed_effective_tokens, plan, supervision_by_category
from minillm.corpus_utils import CorpusError
from minillm.corpus_v4 import (CorpusV4Pipeline, build_registry, inventory, load_config,
                               verify_frozen)

def _root():
    return Path(__file__).resolve().parents[1]


def _ratios():
    return {"zh_general": 0.40, "zh_encyclopedic": 0.25, "zh_education": 0.15,
            "en_general": 0.10, "math_logic_code": 0.10}


def _formal_pipeline(tmp_path):
    root = _root()
    config = load_config(root / "configs" / "corpus_v4.yaml")
    config["targets"]["effective_train_tokens"] = 2048
    config["targets"]["seq_len"] = 128
    config["tokenizer"]["candidates"] = [{"name": "fixture-320", "vocab_size": 320},
                                        {"name": "fixture-384", "vocab_size": 384}]
    config["tokenizer"]["selected"] = "fixture-320"
    config["tokenizer"]["output_root"] = str(tmp_path / "tokenizers")
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
    verified = verify_frozen(pipeline.out)
    assert verified["content_id"] == manifest["content_id"]
    with pytest.raises(CorpusError, match="already frozen"):
        pipeline.finalize()
    (pipeline.out / "split-manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CorpusError, match="SHA256/size mismatch"):
        verify_frozen(pipeline.out)
