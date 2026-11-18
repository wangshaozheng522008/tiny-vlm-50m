"""Original synthetic fixtures test software behavior, never real approval."""
import copy
import json
from pathlib import Path

import pytest

from minillm.corpus_utils import CorpusError, file_info
from minillm.corpus_v4 import build_registry, load_config, status_report
from minillm.v4_sources import load_source_review, prepare_source

ROOT = Path(__file__).resolve().parents[1]


def reviewed_fixture_entry(source, version, inputs):
    """Original software fixture evidence; never a real source approval."""
    source = copy.deepcopy(source)
    source.pop("source_review", None)
    if any(i["source_id"] == source["source_id"] for i in inputs):
        source["paths"] = [i["path"] for i in inputs if i["source_id"] == source["source_id"]]
    source["transformation_history"] = [h for h in source.get("transformation_history", [])
                                       if not (isinstance(h, dict) and h.get("kind") == "source_review")]
    return dict(publisher="Original software fixture author", release="synthetic-v1",
                license_references=[{"uri": "https://example.org/synthetic-license", "verification":
                    "Manual software test review only; no real source approval"}],
                commercial_use="Allowed only within original synthetic software tests",
                redistribution="Allowed only for this original synthetic fixture",
                model_training="Reviewed for synthetic CPU software regression only",
                license_obligations="Preserve original synthetic fixture attribution; no external licence claim",
                corpus_versions=[version], limitations=["TEST_ONLY; not real licence evidence"],
                local_inventory={"files": {i["path"]: {k: i[k] for k in ("sha256", "bytes")}
                                           for i in inputs if i["source_id"] == source["source_id"]}},
                v4_source=source)


def reviewed_fixture_pipeline(tmp_path, *, local_license=None):
    from test_corpus_v4_manifest import _formal_pipeline
    from minillm.corpus_v4 import CorpusV4Pipeline, inventory
    from minillm.v4_sources import reviewed_source_specs
    legacy = _formal_pipeline(tmp_path / "legacy-setup")
    c = copy.deepcopy(legacy.config)
    entries = [reviewed_fixture_entry(s, c["corpus_version"], legacy.inputs) for s in legacy.registry["sources"]]
    if local_license:
        from minillm.corpus_utils import sha256
        for entry in entries:
            entry["license_references"] = [{"uri": str(local_license), "sha256": sha256(local_license),
                "verification": "Manual original synthetic software evidence review only"}]
    ledger = tmp_path / "original-synthetic-reviews.json"
    ledger.write_text(json.dumps({"schema_version": 1, "sources": entries}))
    c["sources"] = reviewed_source_specs(ledger, corpus_version=c["corpus_version"])
    registry = build_registry(c["sources"], demo=False, license_policy=c["license_policy"])
    inputs = inventory(registry, ROOT, config=c)
    return CorpusV4Pipeline(tmp_path / "reviewed", c, registry, inputs, log=lambda *_: None)


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
    entry = review["sources"][0]
    entry["v4_source"]["license_status"] = "APPROVED"
    # The registry fills in over time, so pin one placeholder explicitly instead
    # of relying on whichever field happens to be blank first.
    entry["commercial_use"] = "PENDING"
    p = tmp_path / "review.json"
    p.write_text(json.dumps(review))
    with pytest.raises(CorpusError, match="commercial_use"):
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
    entry["v4_source"].update(license_status="APPROVED", license_evidence="synthetic test only",
        license_reviewer="test-only", license_reviewed_at="2025-05-17", provenance_status="VERIFIED", paths=[str(raw)])
    entry = reviewed_fixture_entry(entry["v4_source"], "synthetic-source-review-v1", [])
    review = tmp_path / "review.json"
    def save():
        review.write_text(json.dumps({"schema_version": 1, "sources": [entry]}))
    save()
    with pytest.raises(CorpusError, match="local file SHA256"):
        load_source_review(review)
    entry["local_inventory"]["files"] = {str(raw.resolve()): file_info(raw)}
    save()
    specs = reviewed_source_specs(review, corpus_version="synthetic-source-review-v1")
    registry = build_registry(specs, demo=False, license_policy={"approved_only": True})
    assert len(inventory(registry, tmp_path)) == 1
    old_history = specs[0]["transformation_history"]
    entry["redistribution"] = "Different test obligation"
    save()
    assert old_history != reviewed_source_specs(review, corpus_version="synthetic-source-review-v1")[0]["transformation_history"]
    raw.write_text("Changed original after test review.\n")
    with pytest.raises(CorpusError, match="differs from reviewed"):
        inventory(registry, tmp_path)


@pytest.fixture
def approved_review_entry(tmp_path):
    raw = tmp_path / "original-software-fixture.txt"
    raw.write_text("Original software test text. No real source approval is asserted.\n")
    source = dict(source_id="source-safety-fixture", source_name="Original synthetic software fixture",
        source_uri=None, license="FIXTURE_ONLY", license_status="APPROVED", license_evidence="Original test fixture only",
        license_reviewer="synthetic-test-only", license_reviewed_at="2025-05-17", provenance_status="VERIFIED",
        category="en_general", language="en", paths=[str(raw)])
    return reviewed_fixture_entry(source, "minillm-corpus-v4-pilot-1m-v1",
        [{"source_id": source["source_id"], "path": str(raw), **file_info(raw)}])


def save_review(tmp_path, entry):
    path = tmp_path / "test-only-source-review.json"
    path.write_text(json.dumps({"schema_version": 1, "sources": [entry]}))
    return path


@pytest.mark.parametrize("field", ["publisher", "release", "commercial_use", "redistribution", "model_training", "license_obligations"])
@pytest.mark.parametrize("value", [" PENDING ", "NOT_PINNED ", "unknown", " UNVERIFIED", "tbd", "TODO", "n/a", " ", None, 7])
def test_approved_review_rejects_placeholders_and_types(tmp_path, approved_review_entry, field, value):
    approved_review_entry[field] = value
    with pytest.raises(CorpusError):
        load_source_review(save_review(tmp_path, approved_review_entry))


@pytest.mark.parametrize("value", [[], None, "example", [None], [{}], [{"uri": "https://example.org/license"}],
    [{"verification": "Manually reviewed fixture"}], [{"uri": "https://example.org/license", "verification": "PENDING"}],
    [{"uri": " N/A ", "verification": "Manual fixture review"}], [{"uri": 42, "verification": "Manual fixture review"}],
    [{"uri": "https://", "verification": "Manual fixture review"}], [{"uri": "https://example.org/license", "verification": False}]])
def test_approved_review_requires_real_reference_structure(tmp_path, approved_review_entry, value):
    approved_review_entry["license_references"] = value
    with pytest.raises(CorpusError):
        load_source_review(save_review(tmp_path, approved_review_entry))


@pytest.mark.parametrize("field", ["license_evidence", "license_reviewer", "license_reviewed_at"])
@pytest.mark.parametrize("value", [" PENDING ", None, 42, ""])
def test_approved_review_requires_human_evidence(tmp_path, approved_review_entry, field, value):
    approved_review_entry["v4_source"][field] = value
    with pytest.raises(CorpusError):
        load_source_review(save_review(tmp_path, approved_review_entry))


@pytest.mark.parametrize("as_uri", [False, True])
def test_local_copyright_evidence_is_verified(tmp_path, approved_review_entry, as_uri):
    from minillm.corpus_utils import sha256
    evidence = tmp_path / "COPYRIGHT.txt"
    evidence.write_text("Copyright statement for original software fixture ONLY.")
    approved_review_entry["license_references"] = [{"uri": evidence.as_uri() if as_uri else str(evidence),
        "sha256": sha256(evidence), "verification": "Manual software fixture copyright review"}]
    path = save_review(tmp_path, approved_review_entry)
    saved = path.read_bytes()
    assert load_source_review(path)["sources"][0] == approved_review_entry
    assert path.read_bytes() == saved  # Validation never normalizes the saved evidence.
    evidence.write_text("Changed after review")
    with pytest.raises(CorpusError, match="local license evidence"):
        load_source_review(path)
    evidence.unlink()
    with pytest.raises(CorpusError, match="local license evidence"):
        load_source_review(path)


def test_unreviewed_incomplete_candidate_remains_readable(tmp_path, approved_review_entry):
    source = approved_review_entry["v4_source"]
    source.update(license_status="REVIEW_REQUIRED", license_evidence="", license_reviewer="", license_reviewed_at="")
    entry = {"v4_source": source, "license_references": None, "publisher": " PENDING "}
    assert load_source_review(save_review(tmp_path, entry))["sources"] == [entry]


def production_fixture_sources(tmp_path, entry):
    from minillm.corpus_v4 import inventory
    from minillm.v4_sources import reviewed_source_specs
    c = load_config(ROOT / "configs/corpus_v4_pilot_1m.yaml")
    path = save_review(tmp_path, entry)
    c["sources"] = reviewed_source_specs(path, corpus_version=c["corpus_version"])
    registry = build_registry(c["sources"], demo=False, license_policy=c["license_policy"])
    inputs = inventory(registry, ROOT, config=c)
    return c, registry, inputs


def test_production_sources_require_full_review_before_ingest(tmp_path, approved_review_entry):
    from minillm.corpus_v4 import CorpusV4Pipeline, inventory
    c, registry, inputs = production_fixture_sources(tmp_path, approved_review_entry)
    p = CorpusV4Pipeline(tmp_path / "admitted", c, registry, inputs, log=lambda *_: None)
    assert p.ingest()["documents"] == 1
    source = copy.deepcopy(approved_review_entry["v4_source"])
    c["sources"] = [source]
    inline = build_registry([source], demo=False, license_policy=c["license_policy"])
    # Old API remains usable, but its evidence cannot enter a production Plan.
    assert inventory(inline, ROOT)
    with pytest.raises(CorpusError, match="source_review"):
        inventory(inline, ROOT, config=c)
    with pytest.raises(CorpusError, match="source_review"):
        CorpusV4Pipeline(tmp_path / "rejected", c, inline, inputs)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("change", ["marker_only", "source_id", "license", "license_evidence", "review_metadata",
    "other_version", "wrong_files", "wrong_input_source", "missing_review", "rehash_foreign_source", "rehash_wrong_file"])
def test_review_content_and_input_binding_cannot_be_forged(tmp_path, approved_review_entry, change):
    from minillm.v4_sources import validate_reviewed_source
    c, registry, inputs = production_fixture_sources(tmp_path, approved_review_entry)
    source = copy.deepcopy(registry["sources"][0])
    if change == "marker_only":
        source["source_review"] = {"schema_version": 2, "sha256": "f" * 64}
    elif change in ("source_id", "license", "license_evidence"):
        source[change] = "different-source-or-evidence"
    elif change == "review_metadata":
        source["source_review"]["entry"]["publisher"] = "Different publisher"
    elif change == "other_version":
        c["corpus_version"] = "other-reviewed-version"
    elif change == "wrong_files":
        inputs[0]["sha256"] = "f" * 64
    elif change == "wrong_input_source":
        inputs[0]["source_id"] = "other-source"
    elif change.startswith("rehash_"):
        from minillm.corpus_utils import identity
        from minillm.v4_sources import review_reference
        record = source["source_review"]
        if change == "rehash_foreign_source":
            record["entry"]["v4_source"]["source_id"] = "foreign-source"
        else:
            record["entry"]["local_inventory"]["files"] = {str(tmp_path / "not-ingested.txt"): {"bytes": 9, "sha256": "f" * 64}}
        record["entry_sha256"] = identity(record["entry"])
        record["sha256"] = identity({k: v for k, v in record.items() if k != "sha256"})
        source["transformation_history"] = [review_reference(source)]
    else:
        source.pop("source_review")
    with pytest.raises(CorpusError):
        validate_reviewed_source(source, inputs, corpus_version=c["corpus_version"])


def test_local_evidence_changed_after_freeze_is_rejected(tmp_path):
    from minillm.corpus_v4 import verify_frozen
    license_path = tmp_path / "ORIGINAL-FIXTURE-COPYRIGHT.txt"
    license_path.write_text("Original software fixture permission. Not real source approval.")
    p = reviewed_fixture_pipeline(tmp_path, local_license=license_path)
    p.finalize()
    assert verify_frozen(p.out)
    license_path.write_text("Changed after the fixture was frozen.")
    with pytest.raises(CorpusError, match="local license evidence"):
        verify_frozen(p.out)


def test_approved_raw_file_mutation_cannot_pass_build(tmp_path, approved_review_entry):
    from minillm.corpus_v4 import CorpusV4Pipeline
    c, registry, inputs = production_fixture_sources(tmp_path, approved_review_entry)
    Path(inputs[0]["path"]).write_text("Changed after Inventory")
    with pytest.raises(CorpusError, match="SHA256/size mismatch"):
        CorpusV4Pipeline(tmp_path / "rejected", c, registry, inputs)


def test_review_identity_is_bound_to_source_version_and_ingest_plan(tmp_path):
    from minillm.corpus_v4 import verify_frozen
    from minillm.v4_admission import audit_v4_dataset, validate_v4_pilot_policy
    from minillm.corpus_utils import write_json
    p = reviewed_fixture_pipeline(tmp_path)
    manifest = p.finalize()
    assert verify_frozen(p.out)["content_id"] == manifest["content_id"]
    assert audit_v4_dataset(p.out)["checks"]["source_reviews"]
    profile = tmp_path / "explicit-software-policy.json"
    write_json(profile, p.config)
    assert validate_v4_pilot_policy(p.out, profile)["ok"]
    manifest["license_sources"]["sources"][0].pop("source_review")
    write_json(p.out / "dataset-manifest.json", manifest)
    with pytest.raises(CorpusError, match="source review"):
        verify_frozen(p.out)
    assert not audit_v4_dataset(p.out)["ok"]


@pytest.mark.parametrize("stage", ["audit", "finalize"])
def test_freeze_rechecks_review_after_pipeline_was_created(tmp_path, stage):
    p = reviewed_fixture_pipeline(tmp_path)
    if stage == "finalize":
        p.audit()
    p.registry["sources"][0]["source_review"]["entry"]["redistribution"] = "Changed after construction"
    with pytest.raises(CorpusError, match="source review"):
        getattr(p, stage)()
    assert not (p.out / "FROZEN").exists()


def test_document_metadata_references_source_review_without_copying_file_list(tmp_path):
    """Measure SQLite metadata payload against the previous per-document layout."""
    import sqlite3
    from minillm.corpus_v4 import CorpusV4Pipeline, inventory
    from minillm.corpus_utils import identity, write_json
    from minillm.v4_sources import reviewed_source_specs
    directory = tmp_path / "original-synthetic-files"
    directory.mkdir()
    inputs = []
    for i in range(64):
        path = directory / f"fixture-{i:03d}.jsonl"
        path.write_text("".join(json.dumps({"text": f"Original source metadata software test file {i} paragraph {j}."}) + "\n"
                                for j in range(4)))
        inputs.append({"source_id": "metadata-fixture", "path": str(path), **file_info(path)})
    source = dict(source_id="metadata-fixture", source_name="Original synthetic software fixture",
        source_uri=None, license="FIXTURE_ONLY", license_status="APPROVED", license_evidence="Original software test fixture only",
        license_reviewer="synthetic-test-only", license_reviewed_at="2025-05-17", provenance_status="VERIFIED",
        category="en_general", language="en", paths=[str(directory)])
    c = load_config(ROOT / "configs/corpus_v4_pilot_1m.yaml")
    entry = reviewed_fixture_entry(source, c["corpus_version"], inputs)
    path = save_review(tmp_path, entry)
    c["sources"] = reviewed_source_specs(path, corpus_version=c["corpus_version"])
    registry = build_registry(c["sources"], demo=False, license_policy=c["license_policy"])
    observed = inventory(registry, ROOT, config=c)
    p = CorpusV4Pipeline(tmp_path / "metadata-run", c, registry, observed, log=lambda *_: None)
    p.ingest()
    db = sqlite3.connect(p.out / "ingested.sqlite")
    rows = db.execute("SELECT metadata, file_sha256 FROM docs").fetchall()
    assert len(rows) == 256
    review = p.plan["registry"]["sources"][0]["source_review"]
    old_evidence = {"kind": "source_review", "entry_sha256": identity(entry),
                    "review": {k: v for k, v in entry.items() if k != "v4_source"}}
    old_db = sqlite3.connect(tmp_path / "old-layout.sqlite")
    db.backup(old_db)
    before_payload = after_payload = 0
    for i, (text, input_sha) in enumerate(rows):
        meta = json.loads(text)
        assert meta["source_id"] == source["source_id"]
        assert meta["source_review_sha256"] == review["sha256"]
        assert meta["input_file_sha256"] == input_sha
        assert "files" not in text and "local_inventory" not in text and "license_references" not in text
        old = copy.deepcopy(meta)
        for key in ("source_id", "source_review_sha256", "input_file_sha256"):
            old.pop(key)
        old["transformation_history"] = [old_evidence]
        old_text = json.dumps(old, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        old_db.execute("UPDATE docs SET metadata=? WHERE rowid=?", (old_text, i + 1))
        before_payload += len(old_text.encode())
        after_payload += len(text.encode())
    old_db.commit()
    old_db.execute("VACUUM")
    db.close()
    old_db.close()
    statistics = {"scope": "synthetic metadata layout comparison; no real source approval", "documents": 256,
        "source_files": 64, "old_metadata_bytes": before_payload, "new_metadata_bytes": after_payload,
        "old_sqlite_bytes": (tmp_path / "old-layout.sqlite").stat().st_size,
        "new_sqlite_bytes": (p.out / "ingested.sqlite").stat().st_size}
    write_json(tmp_path / "metadata-storage-comparison.json", statistics)
    assert after_payload < before_payload / 10
    assert statistics["new_sqlite_bytes"] < statistics["old_sqlite_bytes"]


def test_review_references_preserve_document_family_dedup_and_split(tmp_path):
    import sqlite3
    from minillm.corpus_v4 import CorpusV4Pipeline, inventory
    from minillm.corpus_utils import sha256
    reviewed = reviewed_fixture_pipeline(tmp_path)
    c = copy.deepcopy(reviewed.config)
    c["sources"] = [s["source_review"]["entry"]["v4_source"] for s in reviewed.registry["sources"]]
    registry = build_registry(c["sources"], demo=False, license_policy=c["license_policy"])
    legacy = CorpusV4Pipeline(tmp_path / "legacy-comparison", c, registry, inventory(registry, ROOT), log=lambda *_: None)
    for p in (reviewed, legacy):
        p.split()
    assert sha256(reviewed.out / "split-assignment.jsonl") == sha256(legacy.out / "split-assignment.jsonl")
    def signature(p):
        db = sqlite3.connect(p.out / "deduplicated.sqlite")
        try:
            return db.execute("SELECT document_id, document_family_id, dedup_cluster, dedup_kept FROM docs ORDER BY document_id").fetchall()
        finally:
            db.close()
    assert signature(reviewed) == signature(legacy)
