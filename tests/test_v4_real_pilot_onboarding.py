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

from minillm.corpus_utils import CorpusError, file_info, identity, json_bytes, write_json
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
    assert source["preview_sample"]["scope"] == "RAW_INPUTS"
    assert source["preview_sample"]["verification"] == "INTEGRITY_CHECKED"
    assert source["available_documents"] == manifest["counts"]["documents"]
    assert source["available_characters"] == manifest["counts"]["characters"]
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
    assert verify_sample(tmp_path / "first" / SAMPLE_MANIFEST, reproduce=True) == manifest
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


@pytest.fixture
def sampled_families(tmp_path):
    raw = tmp_path / "original.jsonl"
    rows = [{"id": f"{n:02}-{part}", "parent_id": f"family:{n}",
             "text": f"原创新夹具家庭 {n} 的第 {part} 项测试正文。"}
            for n in range(40) for part in range(2)]
    _write_rows(raw, rows)
    entry = _entry(raw)
    out = tmp_path / "sample"
    manifest = prepare_sample([raw], out, entry=entry, seed=42, basis_points=5000)
    assert 1 < manifest["counts"]["selected_documents"] < len(rows)
    return raw, out, entry, manifest


def _rewrite_manifest(out, manifest):
    manifest["sha256"] = identity({k: v for k, v in manifest.items() if k != "sha256"})
    write_json(out / SAMPLE_MANIFEST, manifest)


@pytest.mark.parametrize("change", ["seed", "basis-points", "statistics", "delete", "add", "edit"])
def test_reproduction_rejects_self_consistent_but_wrong_sample(sampled_families, change):
    raw, out, _, manifest = sampled_families
    sample_file = out / SAMPLE_FILE
    if change == "seed":
        manifest["seed"] = 43
    elif change == "basis-points":
        manifest.update(basis_points=10000, partial_extraction=False)
    elif change == "statistics":
        manifest["counts"]["selected_families"] += 1
    else:
        selected = _rows(sample_file)
        if change == "delete":
            selected.pop()
        elif change == "add":
            selected_ids = {row["id"] for row in selected}
            excluded = next(row for row in _rows(raw) if row["id"] not in selected_ids)
            selected.append({**excluded, "source": manifest["source_id"]})
        else:
            selected[0]["text"] += "修改的测试内容。"
        _write_rows(sample_file, selected)
        manifest["output"] = file_info(sample_file)

    _rewrite_manifest(out, manifest)
    assert verify_sample(out / SAMPLE_MANIFEST) == manifest
    with pytest.raises(CorpusError, match="deterministic selection|missing selected record"):
        verify_sample(out / SAMPLE_MANIFEST, reproduce=True)
    assert file_info(sample_file) == manifest["output"]


@pytest.mark.parametrize(("key", "value"), [
    ("seed", True), ("basis_points", 0), ("basis_points", 10001), ("basis_points", 5000.0),
    ("max_record_bytes", False), ("max_record_bytes", 8 * 1048576 + 1), ("max_record_bytes", -1),
    ("algorithm", "future-algorithm"), ("complete_input_scan", 1), ("partial_extraction", False),
    ("family_policy", "page-without-parent"), ("corpus_version", "different-corpus"),
])
def test_sample_rejects_invalid_manifest_parameters(sampled_families, key, value):
    _, out, _, manifest = sampled_families
    manifest[key] = value
    _rewrite_manifest(out, manifest)
    with pytest.raises(CorpusError, match="invalid Pilot sample manifest"):
        verify_sample(out / SAMPLE_MANIFEST, reproduce=True)


def test_reproduction_rejects_conflicting_original_ids_even_after_rehash(sampled_families):
    raw, out, _, manifest = sampled_families
    rows = _rows(raw)
    selected, *_ = _rows(out / SAMPLE_FILE)
    _write_rows(raw, [*rows, {**selected, "text": "和已选文档身份冲突的原创内容。"}])
    manifest["inputs"][str(raw)] = file_info(raw)
    _rewrite_manifest(out, manifest)
    with pytest.raises(CorpusError, match="conflicting selected records"):
        verify_sample(out / SAMPLE_MANIFEST, reproduce=True)


def test_reproduction_rejects_changed_original_identity(sampled_families):
    raw, out, _, _ = sampled_families
    with raw.open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(CorpusError, match="SHA256/size mismatch"):
        verify_sample(out / SAMPLE_MANIFEST, reproduce=True)


def test_legacy_file_family_manifest_replays_in_reviewed_scope(tmp_path):
    raw = tmp_path / "original.jsonl"
    _write_rows(raw, [{"id": "11", "text": "第一段原创软件夹具。"}, {"id": "12", "text": "第二段原创软件夹具。"}])
    entry = _entry(raw)
    entry["v4_source"]["family_policy"] = "file"
    out = tmp_path / "sample"
    manifest = prepare_sample([raw], out, entry=entry, seed=42, basis_points=10000)
    assert manifest["family_policy"] == "file"
    assert verify_sample(out / SAMPLE_MANIFEST, reproduce=True) == manifest
    reviewed = _sample_entry(entry, out)
    manifest.pop("family_policy")
    _rewrite_manifest(out, manifest)
    reviewed["v4_source"]["transformation_history"][0].update(file_info(out / SAMPLE_MANIFEST))
    assert verify_sample(out / SAMPLE_MANIFEST, reproduce=True, family_policy="file") == manifest
    assert _pipeline(tmp_path, _ledger(tmp_path / "reviewed.json", [reviewed])).ingest()["documents"] == 2


@pytest.mark.parametrize("approved", [False, True])
def test_sample_output_preview_uses_selected_counts_and_existing_approval(tmp_path, sampled_families, approved):
    _, out, entry, manifest = sampled_families
    reviewed = _sample_entry(entry, out) if approved else _rows(out / "review-entry.json")[0]
    ledger = _ledger(tmp_path / "sample-register.json", [reviewed])
    report = pilot_readiness(load_config(CONFIG, source_registry=ledger), ROOT, tmp_path / "no-build",
                             sample_manifests=[out / SAMPLE_MANIFEST])
    source = report["source_status"][manifest["source_id"]]
    selected = manifest["counts"]["selected_documents"]
    assert source["file_status"] == "SOURCE_FILES_VERIFIED"
    assert source["available_files"][0]["sha256"] == manifest["output"]["sha256"]
    assert source["available_documents"] == selected < manifest["counts"]["documents"]
    assert source["available_characters"] == manifest["counts"]["selected_characters"]
    assert source["preview_sample"]["counts"]["documents"] == manifest["counts"]["documents"]
    assert source["preview_sample"]["scope"] == "SAMPLE_OUTPUT"
    assert source["preview_sample"]["verification"] == ("REPRODUCED" if approved else "INTEGRITY_CHECKED")
    assert source["admitted"] is approved
    assert report["category_coverage"][manifest["category"]]["admitted_documents"] == (selected if approved else 0)


@pytest.mark.parametrize("extra_kind", ["jsonl", "txt", "original"])
def test_sample_output_preview_rejects_additional_files(tmp_path, sampled_families, extra_kind):
    raw, out, entry, manifest = sampled_families
    extra = raw if extra_kind == "original" else tmp_path / f"extra.{extra_kind}"
    if extra_kind == "jsonl":
        _write_rows(extra, [{"id": "extra", "text": "原创测试夹具中的额外文档。"}])
    elif extra_kind == "txt":
        extra.write_text("原创测试夹具中的额外文档。\n", encoding="utf-8")
    reviewed = _sample_entry(entry, out)
    reviewed["v4_source"]["paths"].append(str(extra))
    reviewed["local_inventory"]["files"][str(extra)] = file_info(extra)
    ledger = _ledger(tmp_path / "sample-register.json", [reviewed])
    config = load_config(CONFIG, source_registry=ledger)

    # Both files have review evidence, so rejection must come from preview scope.
    registry = build_registry(config["sources"], demo=False, license_policy=config["license_policy"])
    assert {item["path"] for item in inventory(registry, ROOT, config=config)} == {
        str(out / SAMPLE_FILE), str(extra),
    }

    report = pilot_readiness(config, ROOT, tmp_path / "no-build", sample_manifests=[out / SAMPLE_MANIFEST])
    source = report["source_status"][manifest["source_id"]]
    assert source["file_status"] == "SOURCE_FILES_INVALID"
    assert source["input_error"] == "sample preview requires an exact output file set"
    assert source["admitted"] is False
    assert report["admitted_source_count"] == 0
    assert source["available_documents"] == source["available_characters"] == 0
    assert source["available_files"] == []


@pytest.mark.parametrize(("change", "reason"), [
    ("raw-missing", "SHA256/size mismatch"),
    ("raw-changed", "SHA256/size mismatch"),
    ("wrong-release", "configured source scope"),
    ("wrong-output-hash", "reviewed scope or file identity"),
    ("unbound", "transformation_history"),
    ("selection-rehashed", "deterministic selection"),
])
def test_formal_sample_preview_fails_closed(tmp_path, sampled_families, change, reason):
    raw, out, entry, manifest = sampled_families
    reviewed = _sample_entry(entry, out)
    source = reviewed["v4_source"]
    if change == "raw-missing":
        raw.unlink()
    elif change == "raw-changed":
        with raw.open("ab") as handle:
            handle.write(b"\n")
    elif change == "wrong-release":
        reviewed["release"] = "different-release"
    elif change == "unbound":
        source["transformation_history"].clear()
    else:
        if change == "wrong-output-hash":
            manifest["output"]["sha256"] = "0" * 64
        else:
            rows = _rows(out / SAMPLE_FILE)
            rows[0]["text"] = "伪造并重新计算身份的原创测试数据。"
            _write_rows(out / SAMPLE_FILE, rows)
            manifest["output"] = file_info(out / SAMPLE_FILE)
            reviewed["local_inventory"]["files"][str(out / SAMPLE_FILE)] = manifest["output"]
        _rewrite_manifest(out, manifest)
        source["transformation_history"][0].update(file_info(out / SAMPLE_MANIFEST))
    ledger = _ledger(tmp_path / "sample-register.json", [reviewed])
    report = pilot_readiness(load_config(CONFIG, source_registry=ledger), ROOT, tmp_path / "no-build",
                             sample_manifests=[out / SAMPLE_MANIFEST])
    verdict = report["source_status"][source["source_id"]]
    assert verdict["file_status"] == "SOURCE_FILES_INVALID"
    assert not verdict["admitted"] and verdict["source_evidence_status"] != "VERIFIED"
    assert reason in (verdict.get("input_error") or verdict["evidence_error"])
    if change == "selection-rehashed":
        with pytest.raises(CorpusError, match="deterministic selection"):
            _pipeline(tmp_path, ledger)


def test_preview_rejects_a_manifest_for_another_source(tmp_path, sampled_families):
    _, out, entry, manifest = sampled_families
    reviewed = _sample_entry(entry, out)
    manifest["source_id"] = "other-reviewed-source"
    _rewrite_manifest(out, manifest)
    ledger = _ledger(tmp_path / "sample-register.json", [reviewed])
    with pytest.raises(CorpusError, match="sample source absent from the review registry"):
        pilot_readiness(load_config(CONFIG, source_registry=ledger), ROOT, tmp_path / "no-build",
                        sample_manifests=[out / SAMPLE_MANIFEST])


@pytest.mark.parametrize(("policy_ok", "policy_status", "frozen_problem", "expected_blocker"), [
    (True, "POLICY_MATCH", None, None),
    (False, "BLOCKED_POLICY_MISMATCH", None, "FROZEN_POLICY_INVALID"),
    (False, "BLOCKED_POLICY_EVIDENCE_MISSING", None, "FROZEN_POLICY_INVALID"),
    (True, "BLOCKED_POLICY_MISMATCH", None, "FROZEN_POLICY_INVALID"),
    (1, "POLICY_MATCH", None, "FROZEN_POLICY_INVALID"),
    (True, "POLICY_MATCH", "identity", "FROZEN_VERIFICATION_FAILED"),
    (True, "POLICY_MATCH", "budget", "FROZEN_VERIFICATION_FAILED"),
])
def test_frozen_reporting_requires_integrity_budget_and_policy(
        tmp_path, monkeypatch, policy_ok, policy_status, frozen_problem, expected_blocker):
    from minillm import v4_admission, v4_pilot_readiness

    reviewed = []
    for category, (language, _) in CATEGORIES.items():
        path = tmp_path / (category + ".jsonl")
        _write_rows(path, [{"id": "1", "text": "审批路径的独立软件夹具。"}])
        reviewed.append(_entry(path, category, language))
    ledger = _ledger(tmp_path / "registry.json", reviewed)
    config = load_config(CONFIG, source_registry=ledger)
    target = config["targets"]["effective_train_tokens"]
    out = tmp_path / "frozen"
    out.mkdir()
    (out / "FROZEN").touch()
    (out / "contamination-report.json").touch()

    def stages(config, registry, directory, result, by_source, coverage):
        result.update(split_status="PASS", dedup_status="VERIFIED", quality_status="VERIFIED")
        for row in coverage.values():
            row["train_documents"] = 1
        return object()

    def token_budget(config, directory, stages, result, coverage):
        result.update(tokenizer_status="HUMAN_DECISION_VERIFIED", effective_train_tokens=target)

    def frozen(_):
        if frozen_problem == "identity":
            raise CorpusError("frozen input file SHA256/size mismatch")
        return {"effective_train_tokens": target - 1 if frozen_problem == "budget" else target}

    calls = []
    policy = {"ok": policy_ok, "status": policy_status, "errors": [] if not expected_blocker else ["fixture policy error"]}

    def validate(directory):
        calls.append(directory)
        return policy

    monkeypatch.setattr(v4_pilot_readiness, "_stages", stages)
    monkeypatch.setattr(v4_pilot_readiness, "_token_budget", token_budget)
    monkeypatch.setattr(v4_pilot_readiness, "_contamination", lambda *_: "CHECKED_NO_HITS")
    monkeypatch.setattr(v4_pilot_readiness, "verify_frozen", frozen)
    monkeypatch.setattr(v4_admission, "validate_v4_pilot_policy", validate)
    report = pilot_readiness(config, ROOT, out)

    if expected_blocker:
        assert report["status"] == "DATA_NOT_READY"
        assert {expected_blocker, "FINAL_AUDIT_REQUIRED"} <= set(report["blockers"])
    else:
        assert report["status"] == "FROZEN_VERIFIED"
        assert "FINAL_AUDIT_REQUIRED" not in report["blockers"]
    assert report["frozen_integrity_status"] == ("FAIL" if frozen_problem == "identity" else "VERIFIED")
    if frozen_problem:
        assert report["frozen_budget_match"] is (False if frozen_problem == "budget" else None)
        assert report["pilot_policy_validation"] is None and not calls
        assert report["frozen_error"]
    else:
        assert report["frozen_budget_match"] is True
        assert report["pilot_preflight_status"] == policy_status
        assert report["pilot_policy_validation"] == policy
        assert calls == [out]


def test_sample_cannot_claim_itself_as_an_original_input(sampled_families):
    _, out, _, manifest = sampled_families
    manifest["inputs"] = {str(out / SAMPLE_FILE): manifest["output"]}
    _rewrite_manifest(out, manifest)
    with pytest.raises(CorpusError, match="output cannot also be an original input"):
        verify_sample(out / SAMPLE_MANIFEST, reproduce=True)


def test_reproduction_cleans_up_scratch_after_success_and_failure(tmp_path, sampled_families, monkeypatch):
    import tempfile

    _, out, _, manifest = sampled_families
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    assert verify_sample(out / SAMPLE_MANIFEST, reproduce=True) == manifest
    assert not list(scratch.iterdir())

    manifest["counts"]["selected_families"] += 1
    _rewrite_manifest(out, manifest)
    with pytest.raises(CorpusError, match="statistics differ"):
        verify_sample(out / SAMPLE_MANIFEST, reproduce=True)
    assert not list(scratch.iterdir())
