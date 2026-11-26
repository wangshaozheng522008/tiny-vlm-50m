"""Contracts for the V5 candidate inventory and its human approval gate."""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from minillm import v5_language_corpus as v5
from minillm.corpus_quality import content_hash, normalise
from minillm.corpus_utils import CorpusError, file_info, write_json
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
        "commercial_use": "software fixture",
        "redistribution": "software fixture",
        "model_training": "software fixture",
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
            "effective_train_tokens": V5_TARGET,
            "category_token_ratios": V5_CATEGORY_RATIOS,
            "decision": "APPROVED",
            "reviewer": "software fixture reviewer",
            "reviewed_at": "2026-11-26",
            "confirmation": "software fixture only; this is not a human approval",
            "public_weight_release": False,
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


def test_approval_evidence_must_bind_the_request_and_every_file(tmp_path):
    review = _review(tmp_path, approved=True)
    request = v5.approval_request(review)
    request_path = tmp_path / "request.json"
    write_json(request_path, request)
    evidence = {
        "corpus_version": V5_VERSION,
        "decision": "APPROVED",
        "reviewer": "software fixture reviewer",
        "recorded_at": "2026-11-26T12:00:00+08:00",
        "verbatim_human_confirmation": "software fixture only",
        "effective_train_tokens": V5_TARGET,
        "category_token_ratios": V5_CATEGORY_RATIOS,
        "sources": [{"source_id": entry["v4_source"]["source_id"],
                     "release": entry["release"],
                     "files": entry["local_inventory"]["files"],
                     "decision": "APPROVED"} for entry in review["sources"]],
        "approval_request": file_info(request_path),
    }
    evidence_path = tmp_path / "approval.json"
    write_json(evidence_path, evidence)
    result = v5.validate_approval(evidence_path, request_path, review)
    assert result["request_sha256"] == file_info(request_path)["sha256"]

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


class _RangeHandler(BaseHTTPRequestHandler):
    payload = b""

    def do_GET(self):
        start = 0
        header = self.headers.get("Range")
        if header and header.startswith("bytes="):
            start = int(header.split("=", 1)[1].split("-", 1)[0])
        body = self.payload[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(self.payload)-1}/{len(self.payload)}")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_download_resume_uses_range_and_verifies_sha1(tmp_path):
    payload = (b"0123456789abcdef" * 65536) + b"tail"
    _RangeHandler.payload = payload
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        destination = tmp_path / "download.bin"
        part = destination.with_name(destination.name + ".part")
        part.write_bytes(payload[:500000])
        result = v5.download_resumable(f"http://127.0.0.1:{server.server_port}/file",
                                       destination, expected_sha1=hashlib.sha1(payload).hexdigest())
        assert result["resumed_from"] == 500000
        assert destination.read_bytes() == payload
        assert not part.exists()
        destination = tmp_path / "bad.bin"
        part = destination.with_name(destination.name + ".part")
        part.write_bytes(payload[:10])
        with pytest.raises(CorpusError, match="SHA1"):
            v5.download_resumable(f"http://127.0.0.1:{server.server_port}/file",
                                  destination, expected_sha1="0" * 40)
        assert not destination.exists()
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _historical_fixture(tmp_path):
    texts = [("历史验证集文档讲述北京是一座历史悠久的城市，拥有丰富的文化遗产和现代城市生活。城市中有许多博物馆、公园、学校、医院和商业中心，居民在日常生活中使用公共交通、自行车和步行出行。北京也是中国重要的政治、文化、教育和国际交往中心。", "历史验证集文档讲述北京是一座历史悠久的城市，拥有丰富的文化遗产和现代城市生活。城市中有许多博物馆、公园、学校、医院和商业中心，居民在日常生活中使用公共交通、自行车和步行出行。北京也是中国重要的政治、文化、教育和国际交往中心。"),
             ("历史测试集文档，讲述长江是中国最长的河流，流域内分布着许多重要城市和农田，沿岸居民依靠灌溉、航运和渔业生活。", "历史测试集文档，讲述长江是中国最长的河流，流域内分布着许多重要城市和农田，沿岸居民依靠灌溉、航运和渔业生活。")]
    sets = []
    for index, split in enumerate(("val", "test")):
        path = _write_jsonl(tmp_path / f"{split}.text.jsonl", [texts[index][0]])
        manifest = {"files": {f"{split}.text.jsonl": file_info(path)}}
        manifest_path = tmp_path / f"{split}.manifest.json"
        write_json(manifest_path, manifest)
        sets.append({"name": f"fixture-{split}", "split": split, "language": "zh",
                     "path": str(path), "manifest": str(manifest_path),
                     "file": f"{split}.text.jsonl"})
    config_path = tmp_path / "historical.json"
    write_json(config_path, {"schema_version": 1, "corpus_version": V5_VERSION,
                             "sets": sets})
    return v5.load_historical(config_path)


def test_cross_version_audit_removes_exact_and_near_duplicates(tmp_path, tiny_tokenizer):
    historical = _historical_fixture(tmp_path)
    exact = "历史验证集文档讲述北京是一座历史悠久的城市，拥有丰富的文化遗产和现代城市生活。城市中有许多博物馆、公园、学校、医院和商业中心，居民在日常生活中使用公共交通、自行车和步行出行。北京也是中国重要的政治、文化、教育和国际交往中心。"
    near = "历史验证集文档讲述北京是一座历史悠久的城市，拥有丰富的文化遗产和现代城市生活。城市中有许多博物馆、公园、学校、医院和商业中心，居民在日常生活中使用公共交通、自行车和步行出行。北京也是中国重要的政治、文化、教育和国际交往重心。"
    clean = "这是一段完全独立的现代中文说明文本，用于测试候选语料。"
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


def test_formal_v5_quality_splits_overlong_rows_without_losing_family(tmp_path):
    import sqlite3

    from minillm.corpus_v4 import CorpusV4Pipeline

    text = "长文段落一，用于验证正式管线的切分。" * 800 + "\n\n" + "长文段落二，继续验证切分。" * 800
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE docs(
        document_id TEXT PRIMARY KEY, source_id TEXT, source_name TEXT, source_uri TEXT,
        license TEXT, license_evidence TEXT, license_status TEXT, provenance_status TEXT,
        collection_version TEXT, declared_category TEXT, declared_language TEXT,
        document_family_id TEXT, priority INTEGER, file_sha256 TEXT, line_number INTEGER,
        content_sha256 TEXT, raw_text TEXT, raw_bytes INTEGER, metadata TEXT)""")
    db.execute("INSERT INTO docs(document_id, source_id, source_name, source_uri, license, "
               "license_evidence, license_status, provenance_status, collection_version, "
               "declared_category, declared_language, document_family_id, priority, file_sha256, "
               "line_number, content_sha256, raw_text, raw_bytes, metadata) "
               "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
               ("doc1", "source", "Source", "https://example.invalid", "CC-BY-SA-4.0", "e",
                "APPROVED", "VERIFIED", "v5", "zh_general", "zh", "family1", 5, "f" * 64,
                1, "0" * 64, text, len(text.encode()), "{}"))
    for column in ("normalized_text", "normalized_bytes", "normalized_sha256", "detected_language",
                   "category", "quality_action", "quality_flags", "quality_rules_version",
                   "quality_policy_sha256"):
        db.execute(f"ALTER TABLE docs ADD COLUMN {column}")
    dummy = type("Dummy", (), {"v5": True})()
    assert CorpusV4Pipeline._split_overlong_quality(dummy, db, 20000) == 1
    rows = list(db.execute("SELECT document_id, raw_text, document_family_id, metadata FROM docs ORDER BY document_id"))
    assert len(rows) > 1
    assert "".join(row[1] for row in rows) == text
    assert {row[2] for row in rows} == {"family1"}
    assert all(row[0].startswith("doc1::p") for row in rows)
    assert all(json.loads(row[3])["parent_document_id"] == "doc1" for row in rows)
