"""Contracts for the V5 candidate inventory and its human approval gate."""
from __future__ import annotations

import hashlib
import json
import random
import string
from io import BytesIO
from pathlib import Path

import pytest

from minillm import v5_language_corpus as v5
from minillm.corpus_quality import content_hash, normalise
from minillm.corpus_utils import CorpusError, file_info, identity, write_json
from minillm.corpus_versions import (V5_CATEGORY_RATIOS, V5_TARGET, V5_TOKENS_PER_UPDATE,
                                     V5_UPDATES, V5_VERSION)
from minillm.v4_language_pilot import is_language


def _source_entry(path: Path, source_id: str, category: str, language: str,
                  *, approved=False, scope=True, docs=10, characters=5000, tokens=3000):
    uri = "https://example.invalid/" + source_id
    entry = {
        "publisher": "software fixture",
        "release": "fixture-2026",
        "license_references": [{"uri": "https://example.invalid/licence",
                                "verification": "software fixture only"}],
        "commercial_use": "NOT_APPROVED",
        "redistribution": "NOT_APPROVED",
        "model_training": "APPROVED_INTERNAL_RESEARCH_ONLY",
        "license_obligations": "software fixture",
        "local_inventory": {
            "raw_sha256": file_info(path)["sha256"],
            "documents": docs, "characters": characters, "estimated_tokens": tokens,
            "files": {str(path.resolve()): file_info(path)},
        },
        "v4_source": {
            "source_id": source_id,
            "source_name": source_id,
            "source_uri": uri,
            "license": "CC-BY-SA-4.0",
            "license_status": "APPROVED" if approved else "REVIEW_REQUIRED",
            "license_evidence": "software fixture evidence" if approved else "",
            "license_reviewer": "software fixture reviewer" if approved else "",
            "license_reviewed_at": "2026-11-26" if approved else "",
            "category": category,
            "language": language,
            "provenance_status": "VERIFIED",
            "acquisition_date": "2026-11-26",
            "paths": [str(path)],
            "family_policy": "document",
            "priority": 5,
        },
        "corpus_versions": [V5_VERSION] if approved else [],
        "upstream_snapshot_uri": "https://example.invalid/dump",
    }
    if approved and scope:
        entry["training_scope"] = {
            "corpus_version": V5_VERSION,
            "base_content_id": v5.BASE_ID,
            "effective_train_tokens": v5.V5_TARGET,
            "category_token_ratios": V5_CATEGORY_RATIOS,
            "decision": "APPROVED",
            "reviewer": "software fixture reviewer",
            "reviewed_at": "2026-11-26",
            "confirmation": "software fixture only; this is not a human approval",
            "public_weight_release": False,
            "commercial_use": False, "redistribution": False, "model_training": True,
        }
    return entry


def _write_jsonl(path: Path, texts, *, ids=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, text in enumerate(texts):
            handle.write(json.dumps({"id": str(index) if ids is None else ids[index],
                                     "text": text}, ensure_ascii=False) + "\n")
    return path


def _review(tmp_path, *, approved=False, categories=None):
    categories = categories or {
        "candidate-zh-wikipedia": ("zh_encyclopedic", "zh"),
        "candidate-zh-wikinews": ("zh_general", "zh"),
        "candidate-zh-wikibooks": ("zh_education", "zh"),
        "candidate-en-wikisource": ("en_general", "en"),
    }
    entries = []
    for source_id, (category, language) in categories.items():
        path = _write_jsonl(tmp_path / f"{source_id}.jsonl",
                            [f"{source_id} 文档 {index} 是一段用于测试的现代中文说明文本。" for index in range(10)])
        entries.append(_source_entry(path, source_id, category, language, approved=approved))
    return {"schema_version": 1, "approval_authority": "software fixture", "sources": entries}


def test_pending_sources_are_readable_but_cannot_enter_a_scope(tmp_path):
    review = _review(tmp_path)
    entry = review["sources"][0]
    with pytest.raises(CorpusError, match="fresh corpus v5"):
        v5.validate_scope(entry)
    entry = _review(tmp_path, approved=True)["sources"][0]
    v5.validate_scope(entry)
    entry["training_scope"]["effective_train_tokens"] = 2_005_080
    with pytest.raises(CorpusError, match="pinned budget"):
        v5.validate_scope(entry)


def test_scope_requires_this_version_and_a_real_confirmation(tmp_path):
    entry = _review(tmp_path, approved=True)["sources"][0]
    entry["corpus_versions"] = ["minillm-corpus-v4-language-pilot-2m-v1"]
    with pytest.raises(CorpusError, match="fresh corpus v5"):
        v5.validate_scope(entry)
    entry = _review(tmp_path, approved=True)["sources"][0]
    entry["training_scope"]["confirmation"] = "PENDING"
    with pytest.raises(CorpusError, match="confirmation"):
        v5.validate_scope(entry)
    entry = _review(tmp_path, approved=True)["sources"][0]
    entry["v4_source"]["license_status"] = "REVIEW_REQUIRED"
    with pytest.raises(CorpusError, match="APPROVED"):
        v5.validate_scope(entry)


def test_source_set_requires_every_positive_category_and_no_math(tmp_path):
    review = _review(tmp_path, approved=True)
    registry = {"sources": [{"source_id": e["v4_source"]["source_id"],
                             "category": e["v4_source"]["category"],
                             "source_review": {"entry": e}} for e in review["sources"]]}
    assert v5.validate_source_set(registry)
    registry["sources"] = [s for s in registry["sources"] if s["category"] != "en_general"]
    with pytest.raises(CorpusError, match="do not cover"):
        v5.validate_source_set(registry)
    registry = {"sources": [{"source_id": e["v4_source"]["source_id"],
                             "category": "math_logic_code",
                             "source_review": {"entry": e}} for e in review["sources"][:1]]}
    with pytest.raises(CorpusError, match="math/code"):
        v5.validate_source_set(registry)


def _approval_fixture(tmp_path, review):
    request = v5.approval_request(review)
    request_path = tmp_path / "request.json"
    write_json(request_path, request)
    evidence = {
        "corpus_version": V5_VERSION,
        "decision": "APPROVED",
        "reviewer": "software fixture reviewer",
        "recorded_at": "2026-11-26T12:00:00+08:00",
        "verbatim_human_confirmation": "software fixture only",
        "effective_train_tokens": v5.V5_TARGET,
        "category_token_ratios": V5_CATEGORY_RATIOS,
        "sources": [{"source_id": entry["v4_source"]["source_id"],
                     "release": entry["release"],
                     "files": entry["local_inventory"]["files"],
                     "decision": "APPROVED"} for entry in review["sources"]],
        "approval_request": file_info(request_path),
    }
    evidence_path = tmp_path / "approval.json"
    write_json(evidence_path, evidence)
    return evidence_path, request_path


def test_approval_evidence_must_bind_the_request_and_every_file(tmp_path):
    review = _review(tmp_path, approved=True)
    evidence_path, request_path = _approval_fixture(tmp_path, review)
    evidence = json.loads(evidence_path.read_text())
    result = v5.validate_approval(evidence_path, request_path, review)
    assert result["request_sha256"] == file_info(request_path)["sha256"]

    for changes, error in (
            ({"reviewer": "different reviewer"}, "reviewer or date"),
            ({"recorded_at": "2026-11-25"}, "reviewer or date")):
        write_json(evidence_path, evidence | changes)
        with pytest.raises(CorpusError, match=error):
            v5.validate_approval(evidence_path, request_path, review)
    original_request = json.loads(request_path.read_text())
    for key, changes in (
            ("requested_scope", {"commercial_use": True}),
            ("requested_budget", {"effective_train_tokens": 1}),
            ("requested_budget", {"category_token_ratios": {}})):
        write_json(request_path, original_request | {key: original_request[key] | changes})
        write_json(evidence_path, evidence | {"approval_request": file_info(request_path)})
        with pytest.raises(CorpusError, match="budget or authorized scope"):
            v5.validate_approval(evidence_path, request_path, review)
    write_json(request_path, original_request)

    evidence["sources"][0]["files"] = {str(tmp_path / "other.jsonl"): {"sha256": "0" * 64, "bytes": 1}}
    write_json(evidence_path, evidence)
    with pytest.raises(CorpusError, match="file identities"):
        v5.validate_approval(evidence_path, request_path, review)

    review["sources"][0]["v4_source"]["license_status"] = "REVIEW_REQUIRED"
    with pytest.raises(CorpusError, match="incomplete"):
        v5.validate_approval(evidence_path, request_path, review)


def test_deterministic_sampling_is_resumable_and_seed_bound(tmp_path):
    path = _write_jsonl(tmp_path / "source.jsonl",
                        [f"第 {index} 篇现代中文测试文档，用于验证确定性抽样与断点恢复。" for index in range(100)])
    config = {"seed": 42, "commit_every": 7}
    spec = {"source_id": "fixture", "path": str(path), "sample_rate": 1.0,
            "target_raw_tokens": 1000, "max_source_share": 1.0,
            "expected_language": "zh", "category": "zh_general"}
    sampled = tmp_path / "sampled"
    first = v5.sample_source(spec, config, sampled, stop_after=23)
    assert first["status"] == "PARTIAL" and first["records_kept"] == 23
    second = v5.sample_source(spec, config, sampled)
    assert second["status"] == "COMPLETE"
    rows = v5._load_sampled(sampled)
    assert [row["source_record_id"] for row in rows] == [str(index) for index in range(100)]
    assert len({row["document_id"] for row in rows}) == 100

    rerun = tmp_path / "rerun"
    v5.sample_source(spec, config, rerun)
    again = v5._load_sampled(rerun)
    assert [row["sample_rank"] for row in rows] == [row["sample_rank"] for row in again]

    config["seed"] = 43
    with pytest.raises(CorpusError, match="sampling policy changed"):
        v5.sample_source(spec, config, sampled)
    changed = tmp_path / "changed"
    v5.sample_source({**spec, "sample_rate": 0.5}, config, changed)
    changed_rows = v5._load_sampled(changed)
    assert [row["document_id"] for row in changed_rows] != [row["document_id"] for row in rows][:len(changed_rows)]


def test_sampling_refuses_changed_input(tmp_path):
    path = _write_jsonl(tmp_path / "source.jsonl", ["第一段文本，用于验证输入身份绑定。"])
    spec = {"source_id": "fixture", "path": str(path), "sample_rate": 1.0,
            "target_raw_tokens": 1, "max_source_share": 1.0,
            "expected_language": "zh", "category": "zh_general"}
    config = {"seed": 42, "commit_every": 1}
    sampled = tmp_path / "sampled"
    v5.sample_source(spec, config, sampled, stop_after=1)
    path.write_text(path.read_text(encoding="utf-8") + json.dumps({"id": "2", "text": "changed"}) + "\n",
                    encoding="utf-8")
    with pytest.raises(CorpusError, match="changed"):
        v5.sample_source(spec, config, sampled)


def test_download_resume_uses_range_and_verifies_sha1(tmp_path):
    payload = (b"0123456789abcdef" * 65536) + b"tail"
    file = tmp_path / "wiki.bz2"
    partial = file.with_name(file.name + ".part")
    partial.write_bytes(payload[:1024])

    def opener(request, timeout):
        assert request.get_header("Range") == "bytes=1024-"
        response = BytesIO(payload[1024:])
        response.status = 206
        return response

    report = v5.download_resumable("https://example.invalid/wiki.bz2", file,
                                   expected_sha1=hashlib.sha1(payload).hexdigest(), opener=opener)
    assert file.read_bytes() == payload
    assert report["resumed_from"] == 1024
    assert not partial.exists()
    partial.write_bytes(payload[:1024])
    with pytest.raises(CorpusError, match="SHA1"):
        v5.download_resumable("https://example.invalid/wiki.bz2", file,
                              expected_sha1="0" * 40, opener=opener)


def _historical_fixture(tmp_path, monkeypatch, val, test):
    from minillm import v4_language_scaling as scaling
    from minillm.corpus_versions import V4_1M_VERSION, V4_2M_VERSION

    sets = []
    for version, dirname, id_name, info_name in (
            (V4_1M_VERSION, "BASE", "BASE_ID", "BASE_MANIFEST"),
            (V4_2M_VERSION, "BASE_2M", "SCALING_CONTENT_ID", "SCALING_MANIFEST")):
        directory = tmp_path / version
        directory.mkdir()
        monkeypatch.setattr(v5, dirname, directory)
        files = {}
        for split, text in (("val", val), ("test", test)):
            file = directory / f"{split}.text.jsonl"
            v5._write_rows(file, [{"document_id": "old-" + split, "text": text}])
            files[file.name] = file_info(file)
            sets.append({"name": f"{version}-{split}", "split": split, "language": "zh",
                         "path": str(file), "manifest": str(directory / "dataset-manifest.json"),
                         "file": file.name})
        payload = {"files": files, "version": version}
        content_id = identity(payload)
        write_json(directory / "FROZEN", {"content_id": content_id})
        manifest = directory / "dataset-manifest.json"
        write_json(manifest, {"corpus_version": version, "content_id": content_id,
                              "identity": payload, "files": files,
                              "freeze_status": "READY", "dataset_status": "FORMAL"})
        monkeypatch.setattr(scaling, info_name, file_info(manifest))
        monkeypatch.setattr(scaling, id_name, content_id)
    monkeypatch.setattr(v5, "BASE_ID", scaling.BASE_ID)
    config = tmp_path / "historical.json"
    write_json(config, {"schema_version": 1, "corpus_version": V5_VERSION,
                        "base_content_id": v5.BASE_ID, "sets": sets})
    monkeypatch.setattr(v5, "HISTORICAL_CONFIG", config)
    return config


def test_cross_version_audit_removes_exact_and_near_duplicates(tmp_path, tiny_tokenizer, monkeypatch):
    exact = "历史验证集文档讲述北京是一座历史悠久的城市，拥有丰富的文化遗产和现代城市生活。城市中有许多博物馆、公园、学校、医院和商业中心，居民在日常生活中使用公共交通、自行车和步行出行。北京也是中国重要的政治、文化、教育和国际交往中心。"
    near = "历史验证集文档讲述北京是一座历史悠久的城市，拥有丰富的文化遗产和现代城市生活。城市中有许多博物馆、公园、学校、医院和商业中心，居民在日常生活中使用公共交通、自行车和步行出行。北京也是中国重要的政治、文化、教育和国际交往重心。"
    clean = "这是一段完全独立的现代中文说明文本，用于测试候选语料。"
    config = _historical_fixture(tmp_path, monkeypatch, exact, clean)
    historical = v5.load_historical(config)
    clean = "这是另外一份全新的语料，用途包括人文地理和社会风俗。"
    rows = []
    for index, text in enumerate((exact, near, clean)):
        detected = "zh"
        rows.append({"document_id": f"fixture:{index}", "source_id": "fixture",
                     "category": "zh_general", "language": detected,
                     "text": normalise(text, detected),
                     "tokens": len(tiny_tokenizer.encode_document(text, category=detected)),
                     "sample_rank": f"{index:064x}", "chinese_style": "modern"})
    kept, report = v5.cross_version_audit(rows, historical,
                                          v5.NearDupPolicy.from_config({"threshold": 0.80}),
                                          db_path=tmp_path / "historical.sqlite")
    assert len(kept) == 1 and kept[0]["document_id"] == "fixture:2"
    assert report["by_kind"] == {"EXACT": 1, "NEAR": 1}


def test_selection_respects_targets_and_source_caps():
    rows = []
    for source in ("news", "voyage"):
        for index in range(10):
            rows.append({"document_id": f"{source}:{index}", "source_id": source,
                         "category": "zh_general", "tokens": 100,
                         "sample_rank": f"{index:064x}"})
    selected, report = v5.select_candidates(
        rows, {"zh_general": 1000}, {"news": 0.6, "voyage": 0.6}, log=lambda *_: None)
    assert report["zh_general"]["selected_raw_tokens"] == 1000
    from collections import Counter
    counts = Counter(row["source_id"] for row in selected)
    assert counts["news"] <= 6 and counts["voyage"] <= 6
    assert sum(counts.values()) == 10


def test_training_preview_uses_full_optimizer_updates_and_exact_packing():
    rows = []
    quotas = {category: int(V5_TARGET * ratio) + 10 * 1024
              for category, ratio in V5_CATEGORY_RATIOS.items() if ratio}
    for category, tokens in quotas.items():
        room = tokens - 1
        while room > 0:
            take = min(1024, room)
            room -= take
            rows.append({"document_id": f"{category}:{len(rows)}", "source_id": category,
                         "category": category, "tokens": take,
                         "sample_rank": f"{len(rows):064x}"})
    preview = v5.training_preview(rows)
    ten = preview["budgets"]["10M"]
    assert ten["planned_updates"] == V5_UPDATES
    assert ten["planned_supervised_tokens"] == V5_TARGET == V5_UPDATES * V5_TOKENS_PER_UPDATE
    assert ten["required_windows"] == V5_UPDATES * 8
    assert preview["status"] == "ESTIMATED"
    assert ten["available_windows"] == preview["candidate_packing"]["windows"]
    assert ten["feasible_on_candidate_inventory"] is True
    assert preview["budgets"]["100M"]["feasible_on_candidate_inventory"] is False
    held = v5.training_preview(rows, holdout=0.04)
    assert held["holdout_fraction"] == 0.04
    assert (held["budgets"]["10M"]["category_supply"]["zh_general"]
            < preview["budgets"]["10M"]["category_supply"]["zh_general"])


def test_prepare_fails_closed_without_human_approval(tmp_path, monkeypatch):
    review = _review(tmp_path)
    registry_path = tmp_path / "review.json"
    write_json(registry_path, review)
    monkeypatch.setattr(v5, "SOURCE_REGISTRY", registry_path)
    # prepare() takes the registry explicitly and must not create a freeze marker.
    v5.prepare(tmp_path / "prepare", registry_path=registry_path)
    status = json.loads((tmp_path / "prepare/status.json").read_text(encoding="utf-8"))
    assert status["status"] == "BLOCKED_SOURCE_APPROVAL"
    assert status["approved_sources"] == 0
    assert not (tmp_path / "prepare/FROZEN").exists()
    assert (tmp_path / "prepare/source-approval-request.json").is_file()


def test_prepare_reports_ready_when_every_source_is_approved(tmp_path):
    review = _review(tmp_path, approved=True)
    registry_path = tmp_path / "review.json"
    write_json(registry_path, review)
    request = v5.prepare(tmp_path / "prepare", registry_path=registry_path)
    status = json.loads((tmp_path / "prepare/status.json").read_text(encoding="utf-8"))
    assert request["status"] == "APPROVED_READY_FOR_BUILD"
    assert status["status"] == "READY_FOR_BUILD"
    assert status["approved_sources"] == len(review["sources"])
    assert status["pending_sources"] == []
    assert status["formal_freeze"] == "NOT_RUN"
    assert (tmp_path / "prepare/source-review.approved.json").is_file()
    assert not (tmp_path / "prepare/source-review.pending.json").exists()
    assert all(source["new_approval"] == "APPROVED" for source in request["requested_sources"])


def test_legacy_language_profiles_remain_language_profiles():
    assert is_language({"corpus_version": "minillm-corpus-v4-language-pilot-1m-v1"})
    assert is_language({"corpus_version": "minillm-corpus-v4-language-pilot-2m-v1"})
    assert is_language({"corpus_version": V5_VERSION})
    assert not is_language({"corpus_version": "minillm-corpus-v4-pilot-1m-v1"})


def test_overlong_documents_split_only_at_boundaries_without_text_loss():
    text = ("第一段现代中文说明文本，用于验证长文档分段。" * 500 + "\n\n"
            + "第二段继续说明北京、上海、广州和深圳的城市生活。" * 500 + "\n\n"
            + "x" * 5000)
    parts = v5.split_long_text(text, 20000)
    assert len(parts) > 1
    assert "".join(parts) == text
    assert all(len(part) <= 20000 for part in parts)
    assert parts[0].startswith("第一段")
    assert v5.split_long_text("短文本", 20000) == ["短文本"]


@pytest.fixture
def formal_v5(tmp_path, monkeypatch, tiny_tokenizer):
    """Small independent corpus, with only the trusted asset anchors replaced."""
    import yaml

    from minillm import v4_language_pilot as language, v4_language_scaling as scaling

    target = 31 * 8 * 96
    monkeypatch.setattr(v5, "V5_TARGET", target)
    monkeypatch.setattr(v5, "V5_UPDATES", 96)
    monkeypatch.setattr(v5, "V5_SEQ_LEN", 32)
    monkeypatch.setattr(v5, "V5_CANDIDATE_TARGET", 60_000)
    monkeypatch.setattr(v5, "TRAINING_BUDGETS", (("4M", 92, 31 * 8 * 92), ("10M", 96, target),
                                                ("50M", 200, 31 * 8 * 200)))
    monkeypatch.setattr(v5, "V5_TOLERANCE", .015)
    monkeypatch.setattr(language, "LANGUAGE_TOLERANCE", .015)
    rng = random.Random(42)

    def text_for(lang):
        if lang == "zh":
            return "".join(chr(rng.randrange(0x4E00, 0x5F00)) for _ in range(132)) + "。"
        return " ".join("".join(rng.choices(string.ascii_lowercase, k=6)) for _ in range(24)) + "."

    excluded = text_for("zh")
    _historical_fixture(tmp_path, monkeypatch, excluded, text_for("zh"))
    original_config = yaml.safe_load(v5.CORPUS_CONFIG.read_text())
    config = tmp_path / "corpus.json"
    registry = tmp_path / "registry.json"
    original_config["source_registry"] = str(registry)
    original_config["targets"].update(effective_train_tokens=target, seq_len=32)
    write_json(config, original_config)
    monkeypatch.setattr(v5, "CORPUS_CONFIG", config)
    monkeypatch.setattr(language, "V5_CONFIG", config)

    review = _review(tmp_path, approved=True)
    candidate_sources = []
    for entry in review["sources"]:
        source = entry["v4_source"]
        path = Path(source["paths"][0])
        texts = [text_for(source["language"]) for _ in range(180)]
        if source["category"] == "zh_general":
            texts.append(excluded)
        _write_jsonl(path, texts)
        info = file_info(path)
        entry["local_inventory"].update(files={str(path): info}, raw_sha256=info["sha256"])
        candidate_sources.append({"source_id": source["source_id"], "path": str(path), "sample_rate": 1,
                                  "target_raw_tokens": 15_000, "max_source_share": 1,
                                  "expected_language": source["language"]})
    write_json(registry, review)
    candidates = tmp_path / "candidates.json"
    write_json(candidates, {"schema_version": 1, "corpus_version": V5_VERSION, "seed": 42,
                           "target_candidate_effective_tokens": 60_000, "commit_every": 1000, "sources": candidate_sources})
    approval, request = _approval_fixture(tmp_path, review)
    monkeypatch.setattr(v5, "SOURCE_REGISTRY", registry)
    monkeypatch.setattr(v5, "SOURCE_REGISTRY_SHA256", file_info(registry)["sha256"])
    monkeypatch.setattr(v5, "APPROVAL_SHA256", file_info(approval)["sha256"])
    monkeypatch.setattr(v5, "CANDIDATE_CONFIG", candidates)

    historical = v5.BASE
    monkeypatch.setattr(scaling, "BASE", historical)
    tok = historical / "tokenizer.json"
    tiny_tokenizer.save(tok)
    monkeypatch.setattr(v5, "TOKENIZER", tok)
    monkeypatch.setattr(v5, "TOKENIZER_FINGERPRINT", tiny_tokenizer.fingerprint)
    monkeypatch.setattr(v5, "TOKENIZER_VOCAB", tiny_tokenizer.vocab_size)
    receipt = {"name": "fixture", "path": str(tok), "fingerprint": tiny_tokenizer.fingerprint,
               "vocab_size": tiny_tokenizer.vocab_size, "mode": "HUMAN_REVIEWED",
               "reused_from_content_id": v5.BASE_ID, "v5_decision": "REUSE_FROZEN_TOKENIZER"}
    monkeypatch.setattr(v5, "tokenizer_receipt", lambda: receipt)
    write_json(historical / "tokenizer-evaluation.json", {"selection": receipt})
    train_sha, split_sha = "a" * 64, "b" * 64
    write_json(historical / "tokenizer-candidates.json", {"candidates": [{**receipt, "metadata": {
        "demo_only": False, "train_split_sha256": train_sha, "split_manifest_sha256": split_sha,
        "fingerprint": tiny_tokenizer.fingerprint, "special_token_ids": tiny_tokenizer.special_ids,
        "text_policy": "identity-v1"}}]})
    monkeypatch.setattr(scaling, "base_manifest", lambda *args: {
        "files": {"train.text.jsonl": {"sha256": train_sha}},
        "identity": {"v4": {"split_manifest_sha256": split_sha}}})
    return config, registry, approval, request, candidates, excluded


def _formal_pipeline(tmp_path, assets, **overrides):
    from minillm.corpus_v4 import CorpusV4Pipeline, build_registry, inventory, load_config

    config_path, _, approval, request, _, _ = assets
    config = load_config(config_path)
    registry = build_registry(config["sources"], demo=False, license_policy=config["license_policy"])
    return CorpusV4Pipeline(tmp_path / "formal", config, registry, inventory(registry, v5.ROOT, config=config),
                            **({"v5_approval": approval, "v5_request": request} | overrides))


def test_v5_formal_build_audit_freeze_verify_and_admit(tmp_path, formal_v5):
    from minillm.corpus_v4 import verify_frozen
    from minillm.v4_admission import validate_v4_pilot_policy

    _, registry, approval, request, _, _ = formal_v5
    dataset = tmp_path / "formal"
    result = v5.build(dataset, registry_path=registry, approval_path=approval, request_path=request,
                      out_dir=tmp_path / "report")
    assert result["policy"]["ok"]
    assert verify_frozen(dataset) == result["manifest"]
    assert validate_v4_pilot_policy(dataset)["status"] == "POLICY_MATCH"
    audit = json.loads((dataset / "audit-report.json").read_text())
    assert audit["checks"]["v5_historical_dedup"]["detail"]["candidate_removed"] == 1
    assert all(row["removed_documents"] == 0 for row in
               audit["checks"]["v5_historical_dedup"]["detail"]["splits"].values())
    report = json.loads((dataset / v5.SAMPLER_REPORT).read_text())
    for name in ("4M", "10M"):
        row = report["budgets"][name]
        assert row["status"] == "FEASIBLE"
        assert row["consumption"]["effective_tokens"] == row["updates"] * 8 * 31
        assert row["window_gap"] == 0
    assert report["budgets"]["50M"]["window_gap"] > 0
    approval.write_text("{}")
    with pytest.raises(CorpusError, match="approval"):
        verify_frozen(dataset)


def test_historical_document_reinsertion_blocks_formal_audit(tmp_path, formal_v5):
    pipeline = _formal_pipeline(tmp_path, formal_v5)
    pipeline.tokenize()
    path = pipeline.out / "val.text.jsonl"
    rows = v5._load_rows(path)
    rows[0]["text"] = formal_v5[-1]
    v5._write_rows(path, rows)
    with pytest.raises(CorpusError, match="screened candidate lineage"):
        v5.formal_audit_checks(pipeline.out, pipeline.plan)
    # Rehash the modified split and lineage as if a cached report also claimed it passed.
    row = rows[0]
    digest = content_hash(row["text"], row["language"])
    row["normalized_sha256"] = digest
    lineage_file = pipeline.out / v5.CANDIDATE_LINEAGE
    lineage = v5._load_rows(lineage_file)
    record = next(entry for entry in lineage if entry["document_id"] == row["document_id"])
    row["transformation_history"].remove(record)
    record["content_hash"] = digest
    record["normalized_text_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
    row["transformation_history"].append(record)
    v5._write_rows(path, rows)
    v5._write_rows(lineage_file, lineage)
    report_file = pipeline.out / v5.CANDIDATE_REPORT
    report = json.loads(report_file.read_text())
    report["lineage"] = file_info(lineage_file)
    write_json(report_file, report)
    _, checks = v5.formal_audit_checks(pipeline.out, pipeline.plan)
    passed, detail = checks["v5_historical_dedup"]
    assert not passed and detail["splits"]["val"]["removed_documents"] == 1
    assert not (pipeline.out / "FROZEN").exists()


def test_formal_admission_failure_does_not_publish_ready(tmp_path, formal_v5, monkeypatch):
    from minillm import v4_admission

    pipeline = _formal_pipeline(tmp_path, formal_v5)
    audit = pipeline.audit()
    assert audit["status"] == "READY", audit["blockers"]

    def reject(directory, *, prospective_manifest):
        assert not (directory / "FROZEN").exists()
        return {"ok": False, "errors": ["forced prospective admission failure"]}

    monkeypatch.setattr(v4_admission, "_validate_v4_pilot_policy", reject)
    with pytest.raises(CorpusError, match="prospective admission failure"):
        pipeline.finalize(audit=audit)
    assert not (pipeline.out / "FROZEN").exists()
    assert json.loads((pipeline.out / "dataset-manifest.json").read_text())["freeze_status"] == "NOT_READY"


def test_formal_plan_rejects_approval_and_invalidates_cache(tmp_path, formal_v5, monkeypatch):
    from minillm import corpus_v4

    with pytest.raises(CorpusError, match="recorded approval"):
        _formal_pipeline(tmp_path, formal_v5, v5_approval=None)
    pipeline = _formal_pipeline(tmp_path, formal_v5)
    pipeline.ingest()
    config = formal_v5[4]
    original = config.read_bytes()
    data = json.loads(original)
    data["sources"][0]["max_source_share"] = .95
    write_json(config, data)
    with pytest.raises(CorpusError, match="select a new --out"):
        _formal_pipeline(tmp_path, formal_v5)
    config.write_bytes(original)
    historical = v5.HISTORICAL_CONFIG
    existing = historical.read_bytes()
    historical.write_bytes(existing + b"\n")
    with pytest.raises(CorpusError, match="select a new --out"):
        _formal_pipeline(tmp_path, formal_v5)
    historical.write_bytes(existing)
    for filename in ("v5_language_corpus.py", "corpus_versions.py"):
        with monkeypatch.context() as patch:
            sha256 = corpus_v4.sha256
            patch.setattr(corpus_v4, "sha256", lambda path: "f" * 64
                          if path.name == filename else sha256(path))
            assert corpus_v4.implementation_sha256() != pipeline.plan["implementation_sha256"]
            with pytest.raises(CorpusError, match="select a new --out"):
                _formal_pipeline(tmp_path, formal_v5)
    evidence_path = formal_v5[2]
    evidence = json.loads(evidence_path.read_text())
    evidence["verbatim_human_confirmation"] += " changed"
    write_json(evidence_path, evidence)
    with pytest.raises(CorpusError, match="recorded human decision"):
        _formal_pipeline(tmp_path, formal_v5)


def test_historical_compatibility_pins_all_three_fields():
    from minillm import v4_language_scaling as scaling
    from minillm.v4_admission import implementation_compatible
    from minillm.corpus_versions import V4_1M_VERSION, V4_2M_VERSION

    for version, content_id, recorded in (
            (V4_1M_VERSION, scaling.BASE_ID, scaling.BASE_IMPLEMENTATION),
            (V4_2M_VERSION, scaling.SCALING_CONTENT_ID, scaling.SCALING_IMPLEMENTATION)):
        manifest = {"content_id": content_id, "corpus_version": version,
                    "identity": {"v4": {"implementation_sha256": recorded}}}
        assert implementation_compatible(manifest, "current")
        assert not implementation_compatible({**manifest, "content_id": "different"}, "current")
        assert not implementation_compatible({**manifest, "corpus_version": V5_VERSION}, "current")
        manifest["identity"]["v4"]["implementation_sha256"] = "wrong"
        assert not implementation_compatible(manifest, "current")


@pytest.mark.parametrize("directory", [v5.BASE, v5.BASE_2M], ids=["real-1M", "real-2M"])
def test_real_historical_frozen_admission(directory):
    from minillm.corpus_v4 import verify_frozen
    from minillm.v4_admission import validate_v4_pilot_policy

    if not (directory / "FROZEN").is_file():
        pytest.skip("requires local reviewed historical corpus and source assets")
    manifest = verify_frozen(directory)
    policy = validate_v4_pilot_policy(directory)
    assert policy["ok"], policy
    assert policy["content_id"] == manifest["content_id"]
    assert manifest["files"]["tokenizer.json"]["sha256"] == (
        "fb8ac32998348837ef7108bba5df9a8648d5295450f0ddb8fc2ff250ebc42b8d")
