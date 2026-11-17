"""Original synthetic fixtures test software behavior, never real approval."""
import copy
import json
from pathlib import Path

import pytest

from minillm.corpus_utils import CorpusError, file_info
from minillm.corpus_v4 import build_registry, load_config, status_report
from minillm.v4_sources import load_source_review, prepare_source

ROOT = Path(__file__).resolve().parents[1]


def test_real_review_registry_never_auto_approves():
    review = load_source_review(ROOT / "configs/corpus_v4_sources.review.json")
    assert {e["v4_source"]["category"] for e in review["sources"]} == {
        "zh_general", "zh_encyclopedic", "zh_education", "en_general", "math_logic_code"}
    for e in review["sources"]:
        assert e["v4_source"]["license_status"] == "REVIEW_REQUIRED"
        assert not e["v4_source"]["license_reviewer"]
        assert not e["v4_source"]["license_reviewed_at"]
    for name in ("corpus_v4.yaml", "corpus_v4_pilot_1m.yaml"):
        c = load_config(ROOT / "configs" / name)
        r = status_report(c, ROOT)
        assert r["status"] == "NOT_READY"
        assert "NO_APPROVED_SOURCES" in r["blockers"]
        assert all(s["transformation_history"][-1]["kind"] == "source_review" for s in c["sources"])


def test_local_conversion_keeps_family_isolation_when_rows_reordered(tmp_path):
    from test_corpus_v4_manifest import _formal_pipeline
    rows = [{"text": "Original software fixture passage about reading maps and comparing locations. " * 2 + tail,
             "parent_id": "same-original-family"} for tail in ("First separate chapter.", "Second separate chapter.")]
    raw = tmp_path / "input.jsonl"
    hashes = []
    for n, ordered in enumerate((rows, list(reversed(rows)))):
        raw.write_text("".join(json.dumps(r) + "\n" for r in ordered))
        out = tmp_path / f"converted-{n}"
        prepare_source([raw], out, source_id="reordered-fixture", category="en_general", language="en")
        hashes.append(file_info(out / "documents.jsonl"))
        source = dict(source_id="reordered-fixture", source_name="Original software test fixture",
                      source_uri=None, license="FIXTURE_ONLY", license_status="APPROVED",
                      license_evidence="Synthetic fixture; not real source approval", license_reviewer="test-only",
                      license_reviewed_at="2025-05-17", provenance_status="VERIFIED", category="en_general",
                      language="en", paths=[str(out / "documents.jsonl")])
        p = _formal_pipeline(tmp_path / f"pipeline-{n}", extra_sources=[source])
        p.split()
        manifest = json.loads((p.out / "split-manifest.json").read_text())
        assert manifest["leakage"]["family_cluster_violations"] == 0
        assert manifest["leakage"]["near_duplicate"]["status"] == "CHECKED"
    assert hashes[0] == hashes[1]


@pytest.mark.parametrize("missing", ["license_evidence", "license_reviewer", "license_reviewed_at"])
def test_declared_approval_without_evidence_rejected(missing):
    spec = copy.deepcopy(load_config(ROOT / "configs/corpus_v4.yaml")["sources"][0])
    spec.update(license_status="APPROVED", license_evidence="synthetic test evidence",
                license_reviewer="software fixture only", license_reviewed_at="2025-05-17",
                provenance_status="VERIFIED")
    spec[missing] = ""
    with pytest.raises(CorpusError, match="APPROVED requires evidence"):
        build_registry([spec], demo=False, license_policy={"approved_only": True})


def test_conversion_stable_parents_and_order(tmp_path):
    rows = [{"text": "原创测试甲，保留空白与标点。", "parent_id": "chapter-a"},
            {"text": "原创测试乙，同一文档家族。", "parent_id": "chapter-a"},
            {"text": "x"}, {"text": "x"}]
    source = tmp_path / "raw.jsonl"
    def convert(name):
        return prepare_source([source], tmp_path / name, source_id="synthetic-only",
                              category="zh_general", language="zh")
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    original = file_info(source)
    a = convert("a")
    assert file_info(source) == original
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reversed(rows)))
    b = convert("b")
    assert a["files"]["documents.jsonl"] == b["files"]["documents.jsonl"]
    assert a["files"]["provenance.jsonl"] != b["files"]["provenance.jsonl"]
    assert a["documents"] == b["documents"] == 4
    assert a["inputs"][str(source.resolve())] == original
    docs = [json.loads(line) for line in (tmp_path / "a/documents.jsonl").read_text().splitlines()]
    assert len({d["id"] for d in docs}) == 3
    assert len([d for d in docs if d["parent_id"] == "chapter-a"]) == 2
    assert sorted(d["text"] for d in docs) == sorted(r["text"] for r in rows)
    assert a["license_status"] == "NOT_GRANTED_BY_CONVERSION"
    evidence = [json.loads(line) for line in (tmp_path / "a/provenance.jsonl").read_text().splitlines()]
    assert len(evidence) == 4 and all(row["input_sha256"] == original["sha256"] for row in evidence)
    with pytest.raises(ValueError, match="non-empty"):
        convert("a")


def test_utf8_json_errors_and_txt_families(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_bytes(b'{"text":"valid"}\nnot json\n{"text":4}\n{"text":"\xff"}\n')
    r = prepare_source([bad], tmp_path / "bad-out", source_id="fixture-json", category="en_general", language="en")
    assert (r["documents"], r["errors"]) == (1, 3)
    assert len((tmp_path / "bad-out/errors.jsonl").read_text().splitlines()) == 3
    txt = tmp_path / "paragraphs.txt"
    txt.write_text("第一段测试文本\n\n第二段测试文本\n", encoding="utf-8")
    r = prepare_source([txt], tmp_path / "txt-out", source_id="fixture-txt", category="zh_general", language="zh")
    docs = [json.loads(line) for line in (tmp_path / "txt-out/documents.jsonl").read_text().splitlines()]
    assert r["documents"] == 2
    assert docs[0]["parent_id"] == docs[1]["parent_id"]


def test_review_obligations_and_duplicate_authorities_fail_closed(tmp_path):
    review = load_source_review(ROOT / "configs/corpus_v4_sources.review.json")
    review["sources"][0]["v4_source"]["license_status"] = "APPROVED"
    p = tmp_path / "review.json"
    p.write_text(json.dumps(review))
    with pytest.raises(CorpusError, match="pinned release"):
        load_source_review(p)
    c = {"source_registry": str(p), "sources": [{}]}
    p.write_text(json.dumps(c))
    with pytest.raises(CorpusError, match="two approval"):
        load_config(p)


def test_review_binds_actual_local_content_and_metadata(tmp_path):
    """Synthetic approval record solely for exercising file identity checks."""
    from minillm.corpus_v4 import inventory
    from minillm.v4_sources import reviewed_source_specs
    raw = tmp_path / "original.txt"
    raw.write_text("Original software fixture text, no external source approval.\n")
    entry = copy.deepcopy(load_source_review(ROOT / "configs/corpus_v4_sources.review.json")["sources"][0])
    entry.update(publisher="Original software fixture", release="test-v1", commercial_use="TEST_ONLY",
                 redistribution="TEST_ONLY")
    entry["v4_source"].update(license_status="APPROVED", license_evidence="synthetic test only",
        license_reviewer="test-only", license_reviewed_at="2025-05-17", provenance_status="VERIFIED", paths=[str(raw)])
    review = tmp_path / "review.json"
    def save():
        review.write_text(json.dumps({"schema_version": 1, "sources": [entry]}))
    save()
    with pytest.raises(CorpusError, match="local file SHA256"):
        load_source_review(review)
    entry["local_inventory"]["files"] = {str(raw.resolve()): file_info(raw)}
    save()
    specs = reviewed_source_specs(review)
    registry = build_registry(specs, demo=False, license_policy={"approved_only": True})
    assert len(inventory(registry, tmp_path)) == 1
    old_history = specs[0]["transformation_history"]
    entry["redistribution"] = "Different test obligation"
    save()
    assert old_history != reviewed_source_specs(review)[0]["transformation_history"]
    raw.write_text("Changed original after test review.\n")
    with pytest.raises(CorpusError, match="differs from reviewed"):
        inventory(registry, tmp_path)
