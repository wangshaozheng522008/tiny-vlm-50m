"""Synthetic-only decisions exercise validation, not actual human approval."""
import copy
import json
import random
from pathlib import Path

import pytest

from minillm.corpus_utils import CorpusError, file_info, identity, sha256, write_json
from minillm.corpus_v4 import status_report
from minillm.tokenizer import MiniLLMTokenizer, SPECIAL_TOKEN_IDS
from minillm.tokenizer_v4 import (build_selection, candidate_record, compare_candidates,
                                 guard_train_only, load_eval_texts, parameter_budget, train_candidate)
from minillm.v4_tokenizer_decision import training_data_id, validate_decision, verify_selection
from test_corpus_v4_manifest import _formal_pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_unselected_can_prepare_evaluate_and_resume_but_not_freeze(tmp_path):
    def configure(c):
        c["tokenizer"]["selected"] = None
    p = _formal_pipeline(tmp_path, configure=configure)
    p.split()
    evaluated = p.evaluate_tokenizers()
    assert evaluated["selection"] is None
    assert not (p.out / "FROZEN").exists()
    with pytest.raises(CorpusError, match="no tokenizer selected"):
        p.finalize()
    resumed = type(p)(p.out, p.config, p.registry, p.inputs, log=lambda *_: None)
    assert resumed.evaluate_tokenizers() == evaluated
    assert sha256(resumed.out / "train.text.jsonl") == sha256(p.out / "train.text.jsonl")
    # Local approved sources permit preparation even without a final choice.
    c = copy.deepcopy(p.config)
    c["sources"] = copy.deepcopy(p.registry["sources"])
    for source in c["sources"]:
        source["paths"] = [i["path"] for i in p.inputs if i["source_id"] == source["source_id"]]
        # Resolved registry includes derived keys; status_report expects source specs.
        for key in ("reason", "admitted"):
            source.pop(key, None)
    # Existing pipeline setup proves admitted inputs; report on raw fixture specs.
    raw = json.loads((ROOT / "tests/fixtures/corpus_v4/demo-sources.json").read_text())["sources"]
    c["sources"] = [s for s in raw if s["license_status"] == "APPROVED"]
    for s in c["sources"]:
        s["provenance_status"] = "VERIFIED"
        s["paths"] = [str(ROOT / "tests/fixtures/corpus_v4" / path) for path in s["paths"]]
    assert status_report(c, ROOT)["status"] == "READY_FOR_TOKENIZER_EVALUATION"


def test_interrupted_candidates_reuse_verified_complete_candidate(tmp_path, monkeypatch):
    import minillm.tokenizer_v4 as module
    p = _formal_pipeline(tmp_path)
    original = module.train_candidate
    def interrupt(*args, **kwargs):
        if kwargs["name"] == "fixture-384":
            raise RuntimeError("synthetic interrupted process")
        return original(*args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(module, "train_candidate", interrupt)
        with pytest.raises(CorpusError, match="interrupted"):
            p.train_tokenizers()
    first = tmp_path / "tokenizers/fixture-320/tokenizer.json"
    before = file_info(first)
    assert not (p.out / ".v4/tokenizers.stage.json").exists()
    p.train_tokenizers()
    assert file_info(first) == before
    assert len(p.stats["tokenizers"]["candidates"]) == 2


def test_inadequate_candidate_training_is_atomic(tmp_path):
    train = tmp_path / "train.text.jsonl"
    train.write_text(json.dumps({"text": "tiny", "category": "en_general", "language": "en"}) + "\n")
    weights = {c: float(c == "en_general") for c in ("zh_general", "zh_encyclopedic", "zh_education", "en_general", "math_logic_code")}
    selection = build_selection(train, category_weights=weights, max_chars=1000, seed=42)
    with pytest.raises(ValueError, match="TOKENIZER_TRAINING_BLOCKED"):
        train_candidate(selection, train, name="v4-24576", vocab_size=24576,
                        out_root=tmp_path / "tokens", text_policy="identity-v1",
                        split_manifest_sha256="f"*64, training_config={})
    assert not (tmp_path / "tokens/v4-24576").exists()


@pytest.fixture(scope="module")
def decision_dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("original-synthetic-24576-not-real-data")
    rng = random.Random(92)
    words = ["".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=14)) for _ in range(9000)]
    # Repeated artificial words are solely for fitting an exact-size test vocab.
    rows = [{"document_id": str(i), "category": "en_general", "normalized_text": " ".join(words[i:i+100])}
            for i in range(0, len(words), 100)] * 2
    # JsonlRecords expects the published V4 text contract (text, category, id).
    for i, row in enumerate(rows):
        rows[i] = {"document_id": str(i), "category": "en_general", "language": "en", "text": row["normalized_text"]}
    (d / "train.text.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for name in ("val", "test"):
        (d / f"{name}.text.jsonl").write_text(json.dumps({"text": "Held out original test only", "category": "en_general"}) + "\n")
    write_json(d / "split-manifest.json", {"files": {f"{name}.text.jsonl": file_info(d / f"{name}.text.jsonl") for name in ("train", "val", "test")}})
    weights = {k: float(k == "en_general") for k in ("zh_general", "zh_encyclopedic", "zh_education", "en_general", "math_logic_code")}
    selection = build_selection(d / "train.text.jsonl", category_weights=weights, max_chars=1_000_000, seed=42)
    candidate = train_candidate(selection, d / "train.text.jsonl", name="v4-24576", vocab_size=24576,
        out_root=d / "candidates", text_policy="identity-v1", split_manifest_sha256=sha256(d / "split-manifest.json"), training_config={})
    texts = load_eval_texts(ROOT / "configs/tokenizer-v4-evaluation.jsonl")
    comparison = compare_candidates([candidate], texts)
    comparison["selection"] = None
    write_json(d / "tokenizer-candidates.json", {"candidates": [candidate]})
    write_json(d / "tokenizer-evaluation.json", comparison)
    decision = {"candidate": "v4-24576", "fingerprint": candidate["fingerprint"], "vocab_size": 24576,
                "training_data_content_id": training_data_id(d), "train_split_sha256": sha256(d / "train.text.jsonl"),
                "evaluation_report_sha256": sha256(d / "tokenizer-evaluation.json"),
                "reason": "Software regression fixture only; NOT a real approval", "reviewer": "synthetic-test-only",
                "reviewed_at": "2025-05-17", "model_budget_conclusion": "86,528,640 parameters; synthetic test scope"}
    return d, candidate, decision


def test_exact_24576_roundtrip_eos_and_decision(decision_dataset):
    d, candidate, decision = decision_dataset
    tok = MiniLLMTokenizer.load(candidate["path"])
    assert tok.vocab_size == 24576 and tok.special_ids == SPECIAL_TOKEN_IDS
    metrics = compare_candidates([candidate], load_eval_texts(ROOT / "configs/tokenizer-v4-evaluation.jsonl"))["candidates"]["v4-24576"]
    assert metrics["roundtrip_ok"] and metrics["document_boundary_ok"]
    assert "p99" in metrics["overall_token_length_distribution"]
    assert set(metrics["knowledge_topics"]) == {"geography_country", "history", "basic_science", "daily_life", "math_logic", "technology"}
    assert all(row["roundtrip_ok"] and row["document_boundary_ok"] for row in metrics["knowledge_topics"].values())
    selection = validate_decision(d, decision)
    assert selection["mode"] == "HUMAN_REVIEWED"
    write_json(d / "tokenizer-selection.json", selection)
    (d / ".v4").mkdir(exist_ok=True)
    plan = {"schema_version": 4, "scope": "synthetic unit test only"}
    plan["plan_key"] = identity(plan)
    write_json(d / ".v4/plan.json", plan)
    evaluation_key = identity([plan["plan_key"], "synthetic evaluation"])
    write_json(d / ".v4/tokenizer_eval.stage.json", {"key": evaluation_key,
        "files": {str((d / "tokenizer-evaluation.json").resolve()): file_info(d / "tokenizer-evaluation.json")}})
    write_json(d / ".v4/tokenizer_selection.stage.json", {"key": identity([4, plan["plan_key"], "tokenizer_selection",
        {"evaluation": evaluation_key, "decision": selection["decision_sha256"]}]),
        "files": {str((d / "tokenizer-selection.json").resolve()): file_info(d / "tokenizer-selection.json")}})
    assert verify_selection(d) == selection
    selection["decision"]["reason"] = "changed after publication"
    write_json(d / "tokenizer-selection.json", selection)
    with pytest.raises(CorpusError, match="changed"):
        verify_selection(d)


@pytest.mark.parametrize("field,value", [("vocab_size", 32768), ("reviewer", ""), ("reviewed_at", "bad"),
    ("train_split_sha256", "f"*64), ("fingerprint", "f"*64), ("evaluation_report_sha256", "f"*64),
    ("training_data_content_id", "f"*64)])
def test_decision_fail_closed(decision_dataset, field, value):
    d, _, decision = decision_dataset
    changed = {**decision, field: value}
    with pytest.raises((ValueError, CorpusError)):
        validate_decision(d, changed)


@pytest.mark.parametrize("name", ["train", "val", "test"])
def test_any_split_mutation_invalidates_candidate(decision_dataset, name):
    d, _, decision = decision_dataset
    path = d / f"{name}.text.jsonl"
    saved = path.read_bytes()
    try:
        path.write_bytes(saved + b'{}\n')
        with pytest.raises((ValueError, CorpusError)):
            validate_decision(d, decision)
    finally:
        path.write_bytes(saved)


def test_candidate_file_tamper_and_missing_heldout(decision_dataset):
    d, candidate, _ = decision_dataset
    path = Path(candidate["path"])
    saved = path.read_bytes()
    try:
        path.write_bytes(saved + b" ")
        with pytest.raises(CorpusError, match="SHA256"):
            candidate_record(path.parent)
    finally:
        path.write_bytes(saved)
    heldout = d / "val.text.jsonl"
    saved = heldout.read_bytes()
    try:
        heldout.unlink()
        with pytest.raises(ValueError, match="held-out"):
            guard_train_only(d / "train.text.jsonl", d / "split-manifest.json")
    finally:
        heldout.write_bytes(saved)


def test_parameter_int4_budgets_are_estimates_not_gpu_measurements():
    rows = parameter_budget([24576, 32768], seq_len=2048)["vocab_sizes"]
    assert rows["24576"]["total_parameters"] == 86_528_640
    assert rows["32768"]["total_parameters"] == 91_771_520
    assert rows["24576"]["int4_under_50m_tensor_payload"]
    assert rows["24576"]["int4_serialized_bytes"] is None


def test_human_decision_stage_full_freeze_and_pr9_admission(tmp_path):
    """Exact 24k vocabulary on original synthetic test data, no real approval."""
    from minillm.corpus_v4 import verify_frozen
    from minillm.v4_admission import audit_v4_dataset, validate_v4_pilot_policy
    rng = random.Random(92)
    raw = tmp_path / "original-test-only.jsonl"
    with raw.open("w") as handle:
        for i in range(100):
            words = ["".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=14)) for _ in range(100)]
            handle.write(json.dumps({"text": " ".join(words + list(reversed(words))), "parent_id": f"synthetic-{i}"}) + "\n")
    source = dict(source_id="original-synthetic-english", source_name="Software fixture only",
                  source_uri=None, license="TEST_FIXTURE_ONLY", license_status="APPROVED",
                  license_evidence="Original synthetic regression data; no real source approval",
                  license_reviewer="software-test-only", license_reviewed_at="2025-05-17",
                  category="en_general", language="en", provenance_status="VERIFIED", paths=[str(raw)])
    extra_sources = [source]
    # Supply independent Chinese examples as well: the English stress corpus
    # must not satisfy another category's fixed supervision quota.
    passages = [
        "阅读练习要求学生先观察文章的结构，找出每段的主要内容，再比较作者提出的问题与引用的证据。老师让小组轮流说明自己的判断，其他同学可以指出遗漏的细节。完成讨论后，每个人独立写下结论，并解释哪些文字支持自己的理解。这样的练习重视思考过程，也帮助学生认识到同一句话在不同语境中可能具有不同作用。",
        "手工课程开始前，桌面需要整理干净，纸张和工具按照用途分别摆放。学生先在草稿上画出设计，再测量各部分长度，检查连接的位置是否合适。制作过程中如果出现偏差，可以记录原因并调整下一步操作。展示作品时，大家介绍自己采用的方法，说明哪些尝试没有成功，以及以后准备怎样改进。",
        "城市档案保存了道路、建筑和居民生活的许多记录。研究者把地图、照片和文字材料按年代排列，观察街区边界怎样发生变化。他们还需要辨别记录者的视角，注意材料中的空白。不同证据之间有时存在矛盾，需要核对形成背景才能解释。资料整理完成后，原件仍应妥善保存，使后来的读者能够重新检查结论。",
        "山地溪流沿着坡面汇入较大的河道，沿途携带的泥沙会随着水流速度变化而沉积。河岸植物的根系能够固定部分土壤，枯落物也为小型生物提供栖息空间。观察者可以在不同季节记录水位和岸边植被的变化，但单次观察不足以代表长期趋势。连续资料有助于区分短暂天气影响与流域环境变化。"]
    for category, texts in (("zh_education", passages[:2]), ("zh_encyclopedic", passages[2:])):
        path = tmp_path / f"original-{category}.jsonl"
        path.write_text("".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts))
        extra_sources.append({**source, "source_id": f"original-{category}", "language": "zh",
                              "category": category, "paths": [str(path)]})
    def configure(c):
        c["tokenizer"]["selected"] = None
        c["tokenizer"]["candidates"] = [{"name": "v4-24576", "vocab_size": 24576}]
    p = _formal_pipeline(tmp_path, configure=configure, extra_sources=extra_sources)
    evaluation = p.evaluate_tokenizers()
    assert evaluation["selection"] is None
    candidate = p.stats["tokenizers"]["candidates"][0]
    decision = dict(candidate="v4-24576", fingerprint=candidate["fingerprint"], vocab_size=24576,
                    training_data_content_id=training_data_id(p.out), train_split_sha256=sha256(p.out / "train.text.jsonl"),
                    evaluation_report_sha256=sha256(p.out / "tokenizer-evaluation.json"),
                    reason="Test human decision stage only", reviewer="synthetic-test-only",
                    reviewed_at="2025-05-17", model_budget_conclusion="86528640; original fixture only")
    decision_path = tmp_path / "test-decision.json"
    write_json(decision_path, decision)
    p.record_tokenizer_decision(decision_path)
    resumed = type(p)(p.out, p.config, p.registry, p.inputs, log=lambda *_: None)
    resumed.tokenize()
    manifest = resumed.finalize()
    assert manifest["freeze_status"] == "READY"
    assert verify_frozen(p.out)["content_id"] == manifest["content_id"]
    assert audit_v4_dataset(p.out)["ok"]
    profile = tmp_path / "synthetic-profile-only.json"
    write_json(profile, p.config)
    assert validate_v4_pilot_policy(p.out, profile)["ok"]
    assert not validate_v4_pilot_policy(p.out)["ok"]
    selection_path = p.out / "tokenizer-selection.json"
    changed = json.loads(selection_path.read_text())
    changed["decision"]["reviewer"] = "tampered"
    write_json(selection_path, changed)
    with pytest.raises(CorpusError):
        verify_frozen(p.out)
