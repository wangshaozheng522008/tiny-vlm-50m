"""Regression evidence from generated software fixtures, never training data."""
from __future__ import annotations

import json
import random
from collections import Counter

import pytest

from minillm.corpus_budget import plan, supervision_by_category
from minillm.corpus_contamination import audit
from minillm.corpus_utils import CorpusError, file_info, write_json
from minillm.corpus_v4 import CorpusV4Pipeline, build_registry, inventory, load_config, status_report, verify_frozen
from minillm.evaluation import load_tasks
from minillm.evaluation_scoring import score_answer
from minillm.v4_admission import validate_v4_pilot_policy
from minillm.v4_language_pilot import BENCHMARK, LANGUAGE_CONFIG, LANGUAGE_VERSION, SELECTED_TRAIN
from minillm.v4_pilot_readiness import pilot_readiness
from minillm.v4_pilot_supply import analyze_supply
from minillm.v4_tokenizer_decision import language_baseline, verify_baseline_record, verify_evaluation_stage
from minillm.v4_training_accounting import category_consumption
from minillm.v4_training_pilot import LANGUAGE_CONTINUATIONS, LANGUAGE_SUITE, evaluation_policy
from minillm.evaluation import EvaluationConfig, generated_text_diagnostics
from test_v4_real_pilot_onboarding import ROOT, _entry, _ledger, _rows, _write_rows


@pytest.fixture(scope="module")
def language_pipeline(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("language-pilot-software-fixtures")
    rng = random.Random(181)
    han = "".join(chr(i) for i in range(0x4E00, 0x4E00 + 2048))
    entries = []
    for category, documents in (("zh_general", 580), ("zh_encyclopedic", 400),
                                ("zh_education", 210), ("en_general", 235)):
        path = workspace / f"{category}.jsonl"
        language = "en" if category == "en_general" else "zh"
        rows = []
        for i in range(documents):
            if language == "zh":
                text = "。".join("".join(rng.choices(han, k=100)) for _ in range(11)) + "。"
            else:
                text = " ".join("".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=13)) for _ in range(220))
            rows.append({"text": text, "parent_id": f"original-test-family-{category}-{i}"})
        _write_rows(path, rows)
        entry = _entry(path, category, language)
        entry["corpus_versions"] = [LANGUAGE_VERSION]
        entries.append(entry)
    local = _ledger(workspace / "software-fixture-reviews.json", entries)
    config = load_config(LANGUAGE_CONFIG, source_registry=local)
    config["tokenizer"]["output_root"] = str(workspace / "fixture-candidates")
    registry = build_registry(config["sources"], demo=False, license_policy=config["license_policy"])
    return CorpusV4Pipeline(workspace / "fixture-dataset", config, registry,
                            inventory(registry, ROOT, config=config), log=lambda *_: None)


def test_zero_category_is_only_allowed_for_language_policy(tmp_path):
    current = load_config(LANGUAGE_CONFIG)
    ratios = current["targets"]["category_token_ratios"]
    assert ratios == dict(zh_general=.45, zh_encyclopedic=.30, zh_education=.15, en_general=.10, math_logic_code=0)
    for name in ("corpus_v4.yaml", "corpus_v4_pilot_1m.yaml"):
        older = load_config(ROOT / "configs" / name)
        assert older["targets"]["category_token_ratios"]["math_logic_code"] > 0
        older["targets"]["category_token_ratios"] = ratios
        from minillm.corpus_v4 import validate_config
        with pytest.raises(CorpusError, match="zero category weight"):
            validate_config(older)
    current["targets"]["effective_train_tokens"] = 100
    with pytest.raises(CorpusError, match="pinned"):
        status_report(current, ROOT)
    report = pilot_readiness(load_config(LANGUAGE_CONFIG), ROOT, tmp_path)
    assert report["category_coverage"]["math_logic_code"]["status"] == "NOT_REQUIRED"
    assert "math_logic_code" not in report["insufficient_categories"]
    assert report["status"] == "DATA_NOT_READY"


def test_zero_weight_supply_and_mix_fail_closed():
    config = load_config(LANGUAGE_CONFIG)
    targets = config["targets"]
    ratios = targets["category_token_ratios"]
    available = {c: {"documents": 1, "effective_tokens": round(1_020_000 * r)} for c, r in ratios.items() if r}

    def budget():
        return plan(available, 1_020_000, ratios, seq_len=1024, tokenizer_fingerprint="fixture",
                    fraction_tolerance=.01)

    assert budget()["categories"]["math_logic_code"]["coverage"] is None
    assert budget()["categories"]["math_logic_code"]["status"] == "NOT_REQUIRED"
    available["math_logic_code"] = {"documents": 1, "effective_tokens": 100_000}
    assert budget()["total_available_effective_tokens"] == 1_020_000
    assert budget()["status"] == "CATEGORY_MIX_OUT_OF_TOLERANCE"
    del available["math_logic_code"]
    available["zh_general"]["effective_tokens"] += 90_000
    assert budget()["status"] == "CATEGORY_MIX_OUT_OF_TOLERANCE"
    available["en_general"]["effective_tokens"] -= 1
    assert budget()["status"] == "INSUFFICIENT_CORPUS"
    assert supervision_by_category([], 1024)["effective_tokens"] == 0


def test_old_review_cannot_admit_language_source(tmp_path):
    path = tmp_path / "original.txt"
    path.write_text("原创软件测试文字。")
    entry = _entry(path, "zh_general")
    config = load_config(LANGUAGE_CONFIG, source_registry=_ledger(tmp_path / "old.json", [entry]))
    registry = build_registry(config["sources"], demo=False, license_policy=config["license_policy"])
    with pytest.raises(CorpusError, match="different corpus version"):
        inventory(registry, ROOT, config=config)
    entry["v4_source"]["license_status"] = "REVIEW_REQUIRED"
    entry["corpus_versions"] = [LANGUAGE_VERSION]
    config = load_config(LANGUAGE_CONFIG, source_registry=_ledger(tmp_path / "draft.json", [entry]))
    assert status_report(config, ROOT)["status"] == "NOT_READY"
    assert not build_registry(config["sources"], demo=False, license_policy=config["license_policy"])["sources"]


def test_language_asset_review_scoring_and_contamination():
    path = ROOT / BENCHMARK
    items = load_tasks(path, scored=True)
    assert Counter(row["category"] for row in items) == dict(
        fluency=60, daily=50, geohistory=50, science=50, concept=40, instructions=50)
    prompt = items[0]
    assert score_answer("完整句子。", prompt)["correct"] is None
    matched = next(task for task in items if task["scoring"] == "keyword_match")
    result = score_answer(f"答案中提到了{matched['reference']}。", matched)
    assert result["keyword_hit"] and result["correct"] is None
    fact = next(row for row in items if row["reference"] == "巴黎")
    assert score_answer("巴黎。", fact)["correct"]
    assert not score_answer("巴黎或伦敦。", fact)["correct"]
    assets = [{"path": str(path), "kind": "language"}]
    clean = audit(assets, [{"document_id": "fact", "text": "巴黎位于塞纳河畔。"}])
    copied = audit(assets, [{"document_id": "copied", "text": fact["prompt"] + fact["reference"]}])
    assert clean["status"] == "CHECKED_NO_HITS"
    assert copied["status"] == "REVIEW_REQUIRED"
    assert copied["stats"]["high_risk_hits"] > 0


def test_pinned_baseline_path_bytes_hash_and_absence(tmp_path, monkeypatch):
    from minillm import v4_language_pilot

    monkeypatch.setattr(v4_language_pilot, "PROJECT_ROOT", tmp_path)
    config = load_config(LANGUAGE_CONFIG)
    baseline, info = language_baseline(config)
    assert info is None
    assert baseline == tmp_path / config["tokenizer"]["baseline"]
    missing = {"path": str(baseline), "status": "NOT_AVAILABLE"}
    verify_baseline_record(missing, baseline, info)
    for fake in ({**missing, "path": str(tmp_path / "other.json")}, None, {**missing, "sha256": "x"}):
        with pytest.raises(CorpusError, match="Baseline"):
            verify_baseline_record(fake, baseline, info)
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b"original software fixture baseline")
    path, info = language_baseline(config)
    available = {"path": str(path), "status": "AVAILABLE", **info}
    verify_baseline_record(available, path, info)
    for fake in (missing, {**available, "bytes": info["bytes"] + 1}, {**available, "sha256": "f" * 64}):
        with pytest.raises(CorpusError, match="Baseline"):
            verify_baseline_record(fake, path, info)
    baseline.write_bytes(b"same byte count, different bytes!!!")
    with pytest.raises(CorpusError, match="Baseline"):
        verify_baseline_record(available, *language_baseline(config))
    config["tokenizer"]["baseline"] = str(tmp_path / "other.json")
    with pytest.raises(CorpusError, match="pinned"):
        language_baseline(config)


def test_consumed_category_counts_use_loss_positions_and_unique_windows():
    rows = [{"category": category, "tokens": tokens}
            for category, tokens in (("zh_general", 3), ("zh_encyclopedic", 3), ("en_general", 5))]
    measured = category_consumption(rows, 4, [1, 0])
    assert measured["effective_tokens"] == 6
    counts = {c: row["consumed_effective_tokens"] for c, row in measured["categories"].items()}
    assert counts == dict(zh_general=2, zh_encyclopedic=2, zh_education=0, en_general=2, math_logic_code=0)
    subset = category_consumption(rows, 4, [1])
    assert subset["categories"]["zh_general"]["difference_percentage_points"] == pytest.approx(-100 / 3)
    assert subset["frozen_packing"]["raw_tokens"] == 11
    for bad in ([0, 0], [2], [-1]):
        with pytest.raises(ValueError, match="complete Train windows"):
            category_consumption(rows, 4, bad)


def test_language_evaluation_pins_continuations_and_counts_raw_text():
    assert set(evaluation_policy(EvaluationConfig(stage="pretrain"), [LANGUAGE_SUITE])["assets"]) == {
        str(LANGUAGE_SUITE), str(LANGUAGE_CONTINUATIONS)}
    assert len(load_tasks(LANGUAGE_CONTINUATIONS, scored=False)) == 5
    assert generated_text_diagnostics("汉 字X\ufffd\u200b。") == {
        "nonspace_characters": 6, "han_characters": 2, "abnormal_characters": 2,
        "replacement_characters": 1, "ends_with_sentence_punctuation": True}


def test_language_formal_end_to_end(language_pipeline):
    pipeline = language_pipeline
    out = pipeline.out
    assert status_report(pipeline.config, ROOT)["status"] == "READY_FOR_TOKENIZER_EVALUATION"
    pipeline.evaluate_tokenizers()
    report = analyze_supply(pipeline.config, ROOT, out)
    assert report["math_logic_code"]["status"] == "NOT_REQUIRED"
    assert "MATH_CODE_SOURCE_REQUIRED" not in report["blockers"]
    assert report["tokenizer"]["evaluation_status"] == "COMPARED"
    assert report["totals"]["effective_train_tokens"] is None
    assert "HUMAN_TOKENIZER_DECISION_REQUIRED" in report["blockers"]
    marker = out / ".v4/tokenizer_eval.stage.json"
    original = marker.read_bytes()
    forged = json.loads(original)
    forged["key"] = "f" * 64
    write_json(marker, forged)
    try:
        with pytest.raises(CorpusError, match="stage key"):
            verify_evaluation_stage(out, pipeline.config, base=ROOT)
        invalid = analyze_supply(pipeline.config, ROOT, out)
        assert invalid["tokenizer"]["evaluation_status"] == "INVALID"
        assert not invalid["tokenizer"]["candidates"]
        assert invalid["totals"]["effective_train_tokens"] is None
    finally:
        marker.write_bytes(original)
    evaluation_path = out / "tokenizer-evaluation.json"
    evaluation_bytes = evaluation_path.read_bytes()
    evaluation = json.loads(evaluation_bytes)
    evaluation["baseline"] = {**evaluation["baseline"], "path": str(out / "alternate.json")}
    write_json(evaluation_path, evaluation)
    patched = json.loads(original)
    patched["files"][str(evaluation_path)] = file_info(evaluation_path)
    write_json(marker, patched)
    try:
        with pytest.raises(CorpusError, match="Baseline"):
            verify_evaluation_stage(out, pipeline.config, base=ROOT)
    finally:
        evaluation_path.write_bytes(evaluation_bytes)
        marker.write_bytes(original)
    with pytest.raises(CorpusError, match="HUMAN_TOKENIZER_DECISION_REQUIRED"):
        pipeline.finalize()
    assert not (out / "FROZEN").exists()
    draft = json.loads((out / "tokenizer-decision.draft.json").read_text())
    assert draft["reviewer"] is draft["reviewed_at"] is None
    fixture_decision = dict(draft, reviewer="original fixture only", reviewed_at="2025-05-17",
                            reason="Synthetic test", model_budget_conclusion="Synthetic test scope")
    decision = out.parent / "fixture-decision.json"
    write_json(decision, fixture_decision)
    pipeline.record_tokenizer_decision(decision)
    pipeline.tokenize()
    selected = _rows(out / SELECTED_TRAIN)
    supply = {row["document_id"] for row in _rows(out / "train.text.jsonl")}
    ids = {row["document_id"] for row in selected}
    assert ids < supply and len(ids) == len(selected)
    for name in ("val", "test"):
        assert ids.isdisjoint(row["document_id"] for row in _rows(out / f"{name}.text.jsonl"))
    from minillm.tokenizer import MiniLLMTokenizer
    from minillm.v4_language_pilot import select_train

    replayed, accounting = select_train(out, pipeline.config, MiniLLMTokenizer.load(out / "tokenizer.json"))
    assert replayed == selected
    assert accounting["packing"]["effective_tokens"] >= 1_020_000
    assert sum(row["raw_tokens"] for row in accounting["supply"].values()) > accounting["packing"]["raw_tokens"]
    audit_report = pipeline.audit()
    assert audit_report["status"] == "READY", audit_report
    manifest = pipeline.finalize(audit=audit_report)
    assert verify_frozen(out)["content_id"] == manifest["content_id"]
    policy = validate_v4_pilot_policy(out)
    assert policy["ok"], policy
    report = analyze_supply(pipeline.config, ROOT, out)
    assert report["status"] == "FROZEN_VERIFIED", report["blockers"]
    for key in ("train_selection", "budget"):
        assert key in manifest["identity"]["stage_hashes"]
    assert report["totals"]["effective_train_tokens"] == accounting["packing"]["effective_tokens"]
    # Published, passing audit and budget cannot be rewritten to change release evidence.
    budget_path = out / "budget-report.json"
    original = budget_path.read_bytes()
    budget = json.loads(original)
    budget["categories"]["zh_general"]["available_effective_tokens"] = 0
    write_json(budget_path, budget)
    try:
        with pytest.raises(CorpusError, match="SHA256"):
            verify_frozen(out)
    finally:
        budget_path.write_bytes(original)
