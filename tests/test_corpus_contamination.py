"""Eval contamination: benchmark copies are reviewed, ordinary facts are not deleted."""
from __future__ import annotations

import json
from pathlib import Path

from minillm.corpus_contamination import audit, load_benchmark_items

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "evaluation" / "minillm-zh-benchmark.jsonl"
FIXED = ROOT / "data" / "evaluation" / "fixed-prompts.jsonl"


def _documents(*texts):
    for index, text in enumerate(texts):
        yield {"document_id": f"doc-{index}", "text": text, "language": "zh",
               "category": "zh_general", "document_family_id": f"family-{index}"}


def test_benchmark_prompt_copy_requires_review():
    items, provenance, errors = load_benchmark_items([{"path": str(BENCHMARK), "kind": "scored"}])
    assert not errors
    assert provenance[0]["status"] == "FROZEN"
    assert items and items[0].benchmark_id.startswith("minillm-zh-v1/")
    prompt = items[0].prompt
    report = audit([{"path": str(BENCHMARK), "kind": "scored"}], _documents(prompt + "\n补充说明。"))
    assert report["status"] == "REVIEW_REQUIRED"
    hit = next(hit for hit in report["hits"] if hit["train_document_id"] == "doc-0")
    assert hit["overlap_type"] in ("PROMPT_VERBATIM", "QA_ITEM_REPLICATION", "QA_FORMAT_REPLICATION")
    assert hit["disposition"] == "REVIEW"
    assert hit["benchmark_id"] and hit["similarity"] > 0
    assert hit["evidence"]["snippet"]


def test_ordinary_world_fact_is_not_contamination():
    report = audit([{"path": str(BENCHMARK), "kind": "scored"}],
                   _documents("北京是中国的首都，也是一座历史悠久的城市。"))
    assert report["status"] == "CHECKED_NO_HITS"
    assert report["stats"]["high_risk_hits"] == 0
    assert all(hit["disposition"] == "KEEP" for hit in report["hits"])


def test_fixed_prompt_replication_is_high_risk():
    prompt = json.loads(FIXED.read_text(encoding="utf-8").splitlines()[0])["prompt"]
    report = audit([{"path": str(FIXED), "kind": "fixed_prompts"}],
                   _documents(prompt + " 这是训练文档中的复制内容。"))
    assert report["status"] == "REVIEW_REQUIRED"
    assert any(hit["overlap_type"] == "PROMPT_VERBATIM" for hit in report["hits"])


def test_reviewer_decisions_can_clear_or_reject(tmp_path):
    prompt = load_benchmark_items([{"path": str(BENCHMARK), "kind": "scored"}])[0][0].prompt
    assets = [{"path": str(BENCHMARK), "kind": "scored"}]
    pending = audit(assets, _documents(prompt))
    assert pending["status"] == "REVIEW_REQUIRED"
    high = [hit for hit in pending["hits"] if hit["disposition"] == "REVIEW"]
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text("".join(json.dumps({
        "train_document_id": hit["train_document_id"], "benchmark_id": hit["benchmark_id"],
        "action": "accept_risk", "reviewer": "human-reviewer",
        "reason": "documented benchmark-style drill; accepted for this experimental corpus"}) + "\n"
        for hit in high), encoding="utf-8")
    cleared = audit(assets, _documents(prompt), decisions=decisions)
    assert cleared["status"] == "REVIEWED"
    assert cleared["decisions"]["accepted_risk"] == len(high)
    assert cleared["decisions"]["unreviewed_high_risk"] == 0
    exclude = tmp_path / "exclude.jsonl"
    exclude.write_text("".join(json.dumps({
        "train_document_id": hit["train_document_id"], "benchmark_id": hit["benchmark_id"],
        "action": "exclude", "reviewer": "human-reviewer",
        "reason": "benchmark text must not train"}) + "\n" for hit in high), encoding="utf-8")
    rejected = audit(assets, _documents(prompt), decisions=exclude)
    assert rejected["status"] == "FAIL"
    assert rejected["decisions"]["excluded_documents_present"]


def test_undeclared_asset_is_not_assessed(tmp_path):
    fake = tmp_path / "fake-benchmark.jsonl"
    fake.write_text(json.dumps({"id": "x", "prompt": "这是一个假的评测问题吗？"}) + "\n", encoding="utf-8")
    report = audit([{"path": str(fake), "kind": "scored"}], _documents("这是一个假的评测问题吗？"))
    assert report["status"] in ("NOT_ASSESSED", "FAIL")
    assert report["errors"]
