"""Readiness must remain honest when all real sources are unapproved."""
from pathlib import Path

import pytest

from minillm.corpus_utils import CorpusError
from minillm.corpus_v4 import load_config
from minillm.v4_readiness import corpus_readiness, inspect_local, tokenizer_readiness
from minillm.v4_admission import validate_v4_pilot_policy
from v4_pilot_helpers import assets

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {"pilot": ROOT / "configs/corpus_v4_pilot_1m.yaml", "formal": ROOT / "configs/corpus_v4.yaml"}


def test_independent_budgets_with_no_data(tmp_path):
    r = corpus_readiness(CONFIGS, dataset_dirs={k: tmp_path / k for k in CONFIGS})
    assert r["data_status"] == "DATA_NOT_READY"
    for label, target in (("pilot", 1_020_000), ("formal", 100_000_000)):
        v = r["versions"][label]
        assert v["status"] == "INSUFFICIENT_CORPUS"
        assert v["independent_effective_tokens"] == 0
        assert v["budget"]["total_gap_effective_tokens"] == target
        assert not v["admission"]["ok"]
        assert all(row["stage_statistics"] == "NOT_RUN" for row in v["budget"]["categories"].values())
    assert r["gpu_training"] == "NOT_RUN"
    assert load_config(CONFIGS["pilot"])["corpus_version"] != load_config(CONFIGS["formal"])["corpus_version"]
    with pytest.raises(CorpusError, match="cannot share"):
        corpus_readiness(CONFIGS, dataset_dirs={k: tmp_path / "same" for k in CONFIGS})


def test_missing_candidates_not_assessed_and_no_reference_invention(tmp_path):
    r = tokenizer_readiness({"pilot": tmp_path}, eval_asset=ROOT / "configs/tokenizer-v4-evaluation.jsonl", baseline_path=tmp_path / "missing")
    assert r["status"] == "NOT_ASSESSED"
    assert r["tokenizer_status"] == "TOKENIZER_NOT_FINALIZED"
    assert r["historical_reference_only"]["status"] == "NOT_AVAILABLE"
    assert set(r["versions"]["pilot"]["candidates"]) == {"v4-24576", "v4-32768"}


def test_topic_labels_are_estimates_and_unknown_is_not_knowledge(tmp_path):
    p = tmp_path / "source.txt"
    p.write_text("物理与化学是不同的研究领域。\n\n没有主题关键词的原创短句。\n", encoding="utf-8")
    r = inspect_local([p])
    assert r["documents"] == 2 and r["topic_unknown_documents"] == 1
    assert r["topics_estimated"]["physics"] == r["topics_estimated"]["chemistry"] == 1
    assert r["effective_tokens"] is None


def test_existing_pr9_policy_and_frozen_compatibility(tmp_path):
    # Synthetic policy fixture only. Real sources remain REVIEW_REQUIRED.
    fixture = assets(tmp_path)
    assert validate_v4_pilot_policy(fixture["dataset"], fixture["approved_config_path"])["ok"]
    assert not validate_v4_pilot_policy(fixture["dataset"])["ok"]
    report = corpus_readiness({"software_fixture_only": fixture["approved_config_path"]},
                              dataset_dirs={"software_fixture_only": fixture["dataset"]})
    version = report["versions"]["software_fixture_only"]
    assert version["status"] == "DATA_READY"
    assert version["independent_effective_tokens"] >= 2048
    assert all(row["cleaning_documents"] >= row["deduplicated_documents"]
               for row in version["budget"]["categories"].values())


def test_pr9_production_cannot_promote_legacy_inline_approval(tmp_path):
    from unittest.mock import patch
    from minillm.v4_admission import audit_v4_dataset
    from minillm.v4_training_pilot import run_pilot
    from v4_pilot_helpers import arguments
    fixture = assets(tmp_path)
    assert audit_v4_dataset(fixture["dataset"])["ok"]  # Explicit legacy CPU compatibility.
    strict = audit_v4_dataset(fixture["dataset"], require_source_review=True)
    assert not strict["ok"] and strict["errors"]
    args = arguments(fixture)
    args.update(cpu_nano_test=False, approved_config_path=None)
    with patch("minillm.v4_training_pilot.launch_pilot", side_effect=AssertionError("must not launch")):
        result = run_pilot(**args, out=tmp_path / "rejected-production", execute=True)
    assert result["training_status"] == "NOT_RUN"
    assert "BLOCKED_DATA_NOT_READY" in result["preflight"]["blockers"]
