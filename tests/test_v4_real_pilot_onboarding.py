"""Exercise the real-source interface with original software-only records."""
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from minillm.corpus_utils import CorpusError, file_info, json_bytes, write_json
from minillm.corpus_v4 import CorpusV4Pipeline, build_registry, inventory, load_config, verify_frozen
from minillm.v4_pilot_readiness import pilot_readiness
from minillm.v4_pilot_sampling import (PILOT_VERSION, SAMPLE_FILE, SAMPLE_MANIFEST,
                                      prepare_sample, verify_sample)
from minillm.v4_sources import load_source_review
from minillm.wiki_dump import extract_dump
from test_v4_source_onboarding import reviewed_fixture_entry

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/corpus_v4_pilot_1m.yaml"
FIXTURES = ROOT / "tests/fixtures/corpus_v4"
CATEGORIES = {
    "zh_general": ("zh", "zh_general"),
    "zh_encyclopedic": ("zh", "zh_encyclopedic"),
    "zh_education": ("zh", "zh_education"),
    "en_general": ("en", "en_general"),
    "math_logic_code": ("auto", "math_code"),
}


def _rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def _write_rows(path, rows):
    path.write_bytes(b"".join(json_bytes(row) for row in rows))


def _entry(path, category="zh_encyclopedic", language="zh"):
    source_id = "software-fixture-" + category
    spec = dict(source_id=source_id, source_name="Original software-only fixture",
                source_uri="https://example.org/original-software-fixture",
                license="FIXTURE_ONLY", license_status="APPROVED", license_evidence="Original test only",
                license_reviewer="fixture-only", license_reviewed_at="2025-05-17",
                provenance_status="VERIFIED", category=category, language=language, paths=[str(path)])
    return reviewed_fixture_entry(spec, PILOT_VERSION, [{"source_id": source_id, "path": str(path),
                                                        **file_info(path)}])


def _ledger(path, entries):
    write_json(path, {"schema_version": 1, "sources": entries})
    return path


def _pipeline(tmp_path, ledger, *, out=None):
    config = load_config(CONFIG, source_registry=ledger)
    registry = build_registry(config["sources"], demo=False, license_policy=config["license_policy"])
    return CorpusV4Pipeline(out or tmp_path / "pipeline", config, registry,
                            inventory(registry, ROOT, config=config), log=lambda *_: None)


def _sample_entry(source_entry, sample_path):
    draft, = _rows(sample_path / "review-entry.json")
    original = source_entry["v4_source"]
    assert draft["v4_source"]["license_status"] == "REVIEW_REQUIRED"
    assert draft["v4_source"]["license_reviewer"] == draft["v4_source"]["license_reviewed_at"] == ""
    # These decisions exist only within the synthetic test's temporary registry.
    for key in ("license_status", "license_evidence", "license_reviewer", "license_reviewed_at"):
        draft["v4_source"][key] = original[key]
    return draft


@pytest.fixture
def wiki_source(tmp_path):
    xml = tmp_path / "original.xml"
    xml.write_text('<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">'
                   '<page><title>软件夹具</title><ns>0</ns><id>11</id><revision><id>42</id>'
                   '<text>原创正文与[[目标|链接显示]]。</text></revision></page></mediawiki>', encoding="utf-8")
    output = tmp_path / "original.jsonl"
    manifest = tmp_path / "original.jsonl.manifest.json"
    extract_dump(xml, output, source_id="software-fixture-zh_encyclopedic", manifest_path=manifest)
    return output, _entry(output)


def test_wiki_extraction_sample_and_human_registry_reach_v4_ingest(tmp_path, wiki_source):
    raw, entry = wiki_source
    out = tmp_path / "sample"
    manifest = prepare_sample([raw], out, entry=entry, seed=42, basis_points=10000)
    assert _rows(out / SAMPLE_FILE) == _rows(raw)
    assert _rows(raw)[0]["revision_id"] == "42"
    assert verify_sample(out / SAMPLE_MANIFEST) == manifest
    raw_ledger = _ledger(tmp_path / "raw.json", [entry])
    audit = pilot_readiness(load_config(CONFIG, source_registry=raw_ledger), ROOT, tmp_path / "missing",
                            sample_manifests=[out / SAMPLE_MANIFEST])
    source = audit["source_status"][entry["v4_source"]["source_id"]]
    assert source["available_files"][0]["wiki_conversion"]["pages_kept"] == 1
    assert source["source_evidence_status"] == "VERIFIED"
    reviewed = _sample_entry(entry, out)
    pipeline = _pipeline(tmp_path, _ledger(tmp_path / "reviewed.json", [reviewed]))
    assert pipeline.ingest()["documents"] == 1
    assert pipeline.inputs[0]["sha256"] == manifest["output"]["sha256"]


@pytest.mark.parametrize(("change", "match"), [
    ("no-license-reference", "license_references"),
    ("missing", "missing source file"),
    ("tampered", "SHA256/size"),
    ("version", "different corpus version"),
])
def test_local_override_still_enforces_pr10_gate(tmp_path, wiki_source, change, match):
    path, entry = wiki_source
    if change == "no-license-reference":
        entry["license_references"] = []
    elif change == "missing":
        path.unlink()
    elif change == "tampered":
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        entry["corpus_versions"] = ["minillm-corpus-v4-draft"]
    with pytest.raises(CorpusError, match=match):
        _pipeline(tmp_path, _ledger(tmp_path / "reviewed.json", [entry]))


def test_sample_determinism_family_identity_and_duplicate_accounting(tmp_path):
    raw = tmp_path / "input.jsonl"
    rows = [{"id": f"{family}-{part}", "parent_id": f"family:{family}", "revision_id": "20",
             "title": "软件测试", "text": f"原创抽样正文 {family} 和 {part}"}
            for family in range(30) for part in range(2)]
    _write_rows(raw, [*rows, *rows])
    entry = _entry(raw)
    manifest = prepare_sample([raw], tmp_path / "first", entry=entry, seed=42, basis_points=5000)
    second = prepare_sample([raw], tmp_path / "second", entry=entry, seed=42, basis_points=5000)
    selected = _rows(tmp_path / "first" / SAMPLE_FILE)
    assert manifest == second
    assert manifest["complete_input_scan"] and manifest["partial_extraction"]
    assert manifest["counts"]["selected_documents"] == manifest["counts"]["duplicate_documents"] == len(selected)
    assert 0 < len(selected) < len(rows)
    assert set(Counter(row["parent_id"] for row in selected).values()) == {2}
    by_id = {row["id"]: row for row in rows}
    assert all({k: row[k] for k in by_id[row["id"]]} == by_id[row["id"]] for row in selected)
    assert manifest["inputs"][str(raw)] == file_info(raw)
    with pytest.raises(CorpusError, match="already exists"):
        prepare_sample([raw], tmp_path / "first", entry=entry, seed=42, basis_points=5000)


@pytest.mark.parametrize("problem", ["invalid-json", "empty", "source-conflict", "conflicting-id"])
def test_bad_inputs_cannot_publish_samples(tmp_path, problem):
    raw = tmp_path / "input.jsonl"
    entry_row = {"id": "1", "text": "原创软件测试"}
    if problem == "invalid-json":
        raw.write_bytes(json_bytes(entry_row) + b"bad JSON\n")
    elif problem == "empty":
        raw.write_text("", encoding="utf-8")
    elif problem == "source-conflict":
        _write_rows(raw, [{**entry_row, "source": "other"}])
    else:
        _write_rows(raw, [entry_row, {**entry_row, "text": "不同的原创正文"}])
    out = tmp_path / "sample"
    with pytest.raises(CorpusError):
        prepare_sample([raw], out, entry=_entry(raw), seed=42, basis_points=10000)
    assert not out.exists()


def test_changed_sample_input_blocks_v4_resume(tmp_path, wiki_source):
    raw, entry = wiki_source
    sample = tmp_path / "sample"
    prepare_sample([raw], sample, entry=entry, seed=42, basis_points=10000)
    reviewed = _sample_entry(entry, sample)
    ledger = _ledger(tmp_path / "reviewed.json", [reviewed])
    pipeline = _pipeline(tmp_path, ledger)
    pipeline.ingest()
    before = file_info(pipeline.out / "ingested.sqlite")
    raw.write_bytes(raw.read_bytes() + b"\n")
    with pytest.raises(CorpusError, match="SHA256/size mismatch"):
        verify_sample(sample / SAMPLE_MANIFEST)
    with pytest.raises(CorpusError, match="SHA256/size mismatch"):
        _pipeline(tmp_path, ledger)
    assert file_info(pipeline.out / "ingested.sqlite") == before


def test_sample_manifest_cannot_be_reused_for_another_release(tmp_path, wiki_source):
    raw, entry = wiki_source
    sample = tmp_path / "sample"
    prepare_sample([raw], sample, entry=entry, seed=42, basis_points=10000)
    reviewed = _sample_entry(entry, sample)
    reviewed["release"] = "another-release"
    with pytest.raises(CorpusError, match="sample differs from reviewed scope"):
        _pipeline(tmp_path, _ledger(tmp_path / "reviewed.json", [reviewed]))


def test_clean_v4_plan_split_reports_missing_category_and_never_counts_val_test(tmp_path):
    entries = []
    for category, (language, filename) in CATEGORIES.items():
        path = tmp_path / f"{category}.jsonl"
        rows = [{**row, "id": str(index), "parent_id": "shared" if index <= 1 else f"family-{index}"}
                for index, row in enumerate(_rows(FIXTURES / f"{filename}.jsonl"))]
        if category == "en_general":
            rows.extend([{**rows[0], "id": "duplicate"}, {"id": "short", "text": "short original"}])
        _write_rows(path, rows)
        entries.append(_entry(path, category, language))
    ledger = _ledger(tmp_path / "reviewed.json", entries)
    out = tmp_path / "pipeline"
    cli = subprocess.run([sys.executable, str(ROOT / "scripts/plan_corpus_v4.py"), "--config", str(CONFIG),
                          "--source-registry", str(ledger), "--out", str(out), "--stage", "split"],
                         cwd=ROOT, capture_output=True, text=True)
    assert cli.returncode == 0, cli.stderr
    config = load_config(CONFIG, source_registry=ledger)
    report = pilot_readiness(config, ROOT, out)
    assert report["approved_source_count"] == report["available_source_count"] == 5
    assert report["source_evidence_status"] == report["quality_status"] == report["dedup_status"] == "VERIFIED"
    assert report["split_status"] == "PASS"
    assert report["status"] == "DATA_NOT_READY"
    assert "ESTIMATED_TRAIN_SUPPLY_SHORTFALL" in report["blockers"]
    assert report["tokenizer_status"] == "TOKENIZER_NOT_FINALIZED"
    assert report["effective_train_tokens"] is None
    assert report["budget_gap"]["unverified_target_tokens"] == 1_020_000
    by_source = report["source_status"]["software-fixture-en_general"]
    assert by_source["quality"]["rejected"] >= 1
    assert by_source["dedup"]["dedup_deleted"] >= 1
    assert {cat: row["target_effective_train_tokens"] for cat, row in report["category_coverage"].items()} == {
        "zh_general": 408000, "zh_encyclopedic": 255000, "zh_education": 153000,
        "en_general": 102000, "math_logic_code": 102000,
    }
    observed = Counter()
    families = {}
    estimates = config["budget"]["chars_per_token"]
    for split in ("train", "val", "test"):
        rows = _rows(out / f"{split}.text.jsonl")
        assert rows
        for row in rows:
            family = row["document_family_id"]
            assert families.setdefault(family, split) == split
            if split == "train":
                observed[row["category"]] += math.ceil(len(row["text"]) / estimates[row["language"]])
    assert report["estimated_train_tokens"] == sum(observed.values())
    assert all(row["estimated_train_tokens"] == observed[cat] for cat, row in report["category_coverage"].items())
    assert not (out / "FROZEN").exists()
    with pytest.raises(FileNotFoundError, match="dataset-manifest.json"):
        verify_frozen(out)
    manifest = file_info(out / "split-manifest.json")
    assert _pipeline(tmp_path, ledger, out=out).split()["documents"] > 0
    assert file_info(out / "split-manifest.json") == manifest
    stale_config = copy.deepcopy(config)
    stale_config["seed"] += 1
    stale = pilot_readiness(stale_config, ROOT, out)
    assert "STAGE_EVIDENCE_INVALID" in stale["blockers"]

    narrowed = _ledger(tmp_path / "single.json", entries[:1])
    missing = pilot_readiness(load_config(CONFIG, source_registry=narrowed), ROOT, tmp_path / "not-built")
    assert "CATEGORY_DATA_INSUFFICIENT" in missing["blockers"]
    assert "math_logic_code" in missing["insufficient_categories"]


def test_committed_registry_remains_candidate_only_and_reports_missing(tmp_path):
    config = load_config(CONFIG)
    register = load_source_review(ROOT / "configs/corpus_v4_sources.review.json")
    assert len(register["sources"]) == 7
    assert all(entry["v4_source"]["license_status"] == "REVIEW_REQUIRED" for entry in register["sources"])
    report = pilot_readiness(config, tmp_path, tmp_path / "no-dataset")
    assert report["approved_source_count"] == report["available_source_count"] == 0
    assert all(row["file_status"] == "SOURCE_FILES_MISSING" for row in report["source_status"].values())
    assert report["quality_status"] == report["split_status"] == "NOT_RUN"
    assert report["status"] == "DATA_NOT_READY"


def test_exact_readiness_budget_recomputes_only_supervised_train_positions(tmp_path, monkeypatch):
    from minillm import v4_pilot_readiness, v4_tokenizer_decision
    from minillm.corpus_budget import supervision_by_category

    config = load_config(CONFIG)
    selection = tmp_path / "candidate.json"
    selection.write_bytes(b"original software-test tokenizer identity")
    (tmp_path / "tokenizer.json").write_bytes(selection.read_bytes())
    (tmp_path / "tokenizer-selection.json").touch()
    (tmp_path / ".v4").mkdir()
    (tmp_path / ".v4/tokenized.stage.json").touch()
    fingerprint = "f" * 64
    monkeypatch.setattr(v4_tokenizer_decision, "verify_selection",
                        lambda _: {"fingerprint": fingerprint, "path": str(selection)})
    # Stage hash enforcement is exercised by the V4 Split/resume integration test.
    monkeypatch.setattr(v4_pilot_readiness, "_stage", lambda *_: None)
    _write_rows(tmp_path / "train.token-index.jsonl", [
        {"category": "zh_general", "tokens": 1024}, {"category": "en_general", "tokens": 3}])
    for split in ("val", "test"):
        _write_rows(tmp_path / f"{split}.token-index.jsonl", [{"category": "math_logic_code", "tokens": 2_000_000}])
    (tmp_path / "train.bin").write_bytes(bytes(2 * 1027))
    packing = supervision_by_category([("zh_general", 1024), ("en_general", 3)], 1024)
    write_json(tmp_path / "tokenization-report.json", {
        "packing": packing, "seq_len": 1024, "tokenizer": {"fingerprint": fingerprint},
        "effective_train_tokens": packing["effective_tokens"]})
    report = {"budget_gap": {}}
    coverage = {cat: {} for cat in CATEGORIES}
    v4_pilot_readiness._token_budget(config, tmp_path, ({}, "split-key"), report, coverage)
    assert report["effective_train_tokens"] == 1023
    assert coverage["math_logic_code"]["effective_train_tokens"] == 0
    assert coverage["zh_general"]["effective_train_tokens"] == 1023
    assert report["budget_gap"]["measured_category_gap_total"] == 1_020_000 - 1023


def test_staged_fallback_excludes_conversion_sidecars(tmp_path):
    config = load_config(CONFIG)
    staged = tmp_path / "data/staged/candidate-zh-wikibooks-v1"
    staged.mkdir(parents=True)
    _write_rows(staged / "documents.jsonl", [{"text": "原创测试，独立软件夹具。"}])
    _write_rows(staged / "provenance.jsonl", [{"metadata": "test only"}])
    (staged / "errors.jsonl").touch()
    report = pilot_readiness(config, tmp_path, tmp_path / "no-build")
    source = report["source_status"]["candidate-zh-wikibooks"]
    assert source["file_status"] == "SOURCE_FILES_VERIFIED"
    assert source["available_documents"] == 1 and len(source["available_files"]) == 1
    assert source["license_status"] == "REVIEW_REQUIRED" and source["admitted"] is False
