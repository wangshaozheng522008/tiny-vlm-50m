"""Tokenizer v4 candidates: train-only fitting, fixed-text metrics, parameter costs."""
from __future__ import annotations

import json

import pytest

from minillm.tokenizer_v4 import (build_selection, compare_candidates, guard_train_only,
                                  parameter_budget, validate_candidates, _selection_hint)


def test_candidate_sizes_are_validated_not_assumed():
    assert validate_candidates([{"name": "v4-24576", "vocab_size": 24576},
                                {"name": "v4-32768", "vocab_size": 32768}]) == [
        {"name": "v4-24576", "vocab_size": 24576},
        {"name": "v4-32768", "vocab_size": 32768}]
    with pytest.raises(ValueError):
        validate_candidates([{"name": "v4-24576", "vocab_size": 24576},
                             {"name": "v4-24576", "vocab_size": 32768}])
    with pytest.raises(ValueError):
        validate_candidates([{"name": "v4-1", "vocab_size": 100}])


def test_parameter_budget_reports_larger_vocab_cost_without_changing_layers():
    report = parameter_budget([24576, 32768], seq_len=1024)
    small, large = report["vocab_sizes"]["24576"], report["vocab_sizes"]["32768"]
    assert small["total_parameters"] == 86_528_640
    assert large["total_parameters"] > small["total_parameters"]
    assert large["delta_parameters_vs_baseline"] > 5_000_000
    assert large["delta_fraction_vs_baseline"] > 0.02
    assert large["transformer_parameters"] == small["transformer_parameters"]
    assert large["lm_head_tied"] is True
    assert large["exceeds_87m_reference"] is True
    assert small["exceeds_87m_reference"] is False
    assert large["adamw_state_bytes"] > small["adamw_state_bytes"]
    assert report["architecture_unchanged"] is True


def test_small_gain_does_not_justify_a_larger_vocabulary():
    config = {"min_gain_fraction": 0.03, "max_param_increase_fraction": 0.02}
    comparisons = {
        "small": {"vocab_size": 24576, "zh_characters_per_token": 1.80},
        "large": {"vocab_size": 32768, "zh_characters_per_token": 1.82},
    }
    hint = _selection_hint(comparisons, config)
    assert hint["requires_explicit_selection"] is True
    assert hint["preferred_lower_cost"] == "small"
    assert hint["zh_compression_gain"] < config["min_gain_fraction"]


def test_selection_is_deterministic_and_train_only(v4_demo):
    v4_demo.deduplicate()
    v4_demo.split()
    train = v4_demo.out / "train.text.jsonl"
    manifest = guard_train_only(train, v4_demo.out / "split-manifest.json")
    assert manifest["files"]["train.text.jsonl"]["sha256"]
    selection = build_selection(train, category_weights=v4_demo.config["tokenizer"]["training"][
        "category_weights"], max_chars=30_000_000, seed=42)
    again = build_selection(train, category_weights=v4_demo.config["tokenizer"]["training"][
        "category_weights"], max_chars=30_000_000, seed=42)
    assert selection["selected_documents"] == again["selected_documents"]
    assert selection["selection"] == again["selection"]
    assert set(selection["selected_by_category"]) == {
        "zh_general", "zh_encyclopedic", "zh_education", "en_general", "math_logic_code"}
    before = (v4_demo.out / "val.text.jsonl").read_bytes()
    (v4_demo.out / "val.text.jsonl").write_bytes(before + b"\n")
    with pytest.raises(ValueError, match="held-out split"):
        guard_train_only(train, v4_demo.out / "split-manifest.json")


def test_evaluation_metrics_cover_chinese_english_math_code_and_roundtrip(v4_demo):
    v4_demo.deduplicate()
    v4_demo.split()
    candidates = v4_demo.train_tokenizers()
    report = v4_demo.evaluate_tokenizers()
    for row in candidates["candidates"]:
        assert row["fingerprint"] and row["vocab_size"] in (320, 384)
        assert row["metadata"]["train_split_sha256"]
        assert row["metadata"]["training_corpus_sha256"]
        assert row["metadata"]["selection_deterministic"] is True
    comparison = report["comparison"]
    assert set(comparison) == {"demo-320", "demo-384"}
    for row in report["candidates"].values():
        assert row["zh"]["characters_per_token"] > 0
        assert row["zh"]["characters_per_1024_tokens"] > 0
        assert row["categories"]["en"]["tokens_per_word"] > 0
        assert row["categories"]["math"]["characters_per_token"] > 0
        assert row["categories"]["python"]["characters_per_token"] > 0
        assert row["categories"]["cpp"]["characters_per_token"] > 0
        assert row["categories"]["special"]["tokens"] > 0
        assert row["roundtrip_ok"] is True
        assert row["document_boundary_ok"] is True
    assert report["recommendation"]["status"] == "HUMAN_DECISION_REQUIRED"
    assert report["parameter_budget"]["vocab_sizes"]
    assert report["selection"]["mode"] == "AUTO_DEMO_ONLY"
