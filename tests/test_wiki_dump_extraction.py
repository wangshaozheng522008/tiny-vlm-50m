"""Wiki dump extraction: one page becomes one traceable JSONL record."""
import bz2
import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree.ElementTree import ParseError

import pytest

from minillm import wiki_dump
from minillm.corpus_utils import CorpusError, file_info, identity, json_bytes
from minillm.wiki_dump import clean_wikitext, extract_dump, iter_pages, write_manifest

ROOT = Path(__file__).resolve().parents[1]

PAGE = """<page>
    <title>{title}</title>
    <ns>{ns}</ns>
    <id>{page_id}</id>
    {extra}
    <revision>
      <id>{rev}</id>
      <text>{text}</text>
    </revision>
  </page>"""


def _dump(tmp_path, pages, name="dump.xml.bz2"):
    """Real dumps carry the export namespace, so every fixture keeps it."""
    body = "".join(PAGE.format(**page) for page in pages)
    xml = ('<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" version="0.11">'
           f'<siteinfo><sitename>fixture</sitename></siteinfo><pages>{body}</pages></mediawiki>')
    path = tmp_path / name
    with bz2.open(path, "wb") as handle:
        handle.write(xml.encode("utf-8"))
    return path


def _page(page_id, text, title="Fixture page", ns=0, rev=1, extra=""):
    return {"page_id": page_id, "text": text, "title": title, "ns": ns, "rev": rev, "extra": extra}


@pytest.mark.parametrize("raw, expected", [
    ("plain text", "plain text"),
    ("[[目标页面|显示文字]]是链接", "显示文字是链接"),
    ("[[单一目标]]保留", "单一目标保留"),
    ("前{{模板}}后", "前后"),
    ("前{{外层|{{内层}}参数}}后", "前后"),
    ("前<ref name=\"a\">引用内容</ref>后", "前后"),
    ("前<ref name=\"a\"/>后", "前后"),
    ("前<!-- 注释 -->后", "前后"),
    ("前[[File:示例.png|缩略图|说明]]后", "前后"),
    ("前[[Category:分类名]]后", "前后"),
    ("== 标题 ==\n正文", "标题\n正文"),
    ("'''粗体'''和''斜体''", "粗体和斜体"),
    ("前<span class=\"x\">内容</span>后", "前内容后"),
    ("[https://example.org 外部链接文字]", "外部链接文字"),
    ("前[https://example.org]后", "前后"),
    ("{| class=\"wikitable\"\n|-\n|单元格\n|}", ""),
    ("* 列表项", "列表项"),
    ("前&mdash;后", "前—后"),
    ("__NOTOC__\n正文", "正文"),
    ("<gallery>\nFile:示例.png|说明\n</gallery>", ""),
    ("  \u00a0 ", ""),
])
def test_clean_wikitext_reduces_markup_without_reordering(raw, expected):
    assert clean_wikitext(raw) == expected


def test_extract_dump_keeps_main_namespace_and_reports_skips(tmp_path):
    dump = _dump(tmp_path, [
        _page(10, "第一段正文，包含[[链接|文字]]与'''强调'''。", title="正文甲"),
        _page(11, "重定向说明", title="重定向乙", extra='<redirect title="正文甲"/>'),
        _page(12, "模板内容", title="Template:甲", ns=10),
        _page(13, "", title="空文本丙"),
    ])
    out = tmp_path / "out" / "dump.jsonl"
    manifest = extract_dump(dump, out, source_id="fixture-zh-wikisource")

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["10"]
    assert rows[0]["parent_id"] == "10"
    assert rows[0]["source"] == "fixture-zh-wikisource"
    assert rows[0]["title"] == "正文甲"
    assert rows[0]["revision_id"] == "1"
    assert rows[0]["text"] == "第一段正文，包含文字与强调。"
    assert manifest["pages_seen"] == 4
    assert manifest["pages_kept"] == 1
    assert manifest["skipped"] == {"namespace": 1, "redirect": 1, "empty_after_cleaning": 1}
    assert manifest["output"]["bytes"] == out.stat().st_size
    assert len(manifest["output"]["sha256"]) == 64
    assert len(manifest["input"]["sha256"]) == 64
    assert manifest["sha256"]


def test_extract_dump_honours_namespace_selection_and_page_cap(tmp_path):
    dump = _dump(tmp_path, [
        _page(1, "正文一"),
        _page(2, "正文二"),
        _page(3, "作者页", title="Author:某人", ns=102),
    ])
    out = tmp_path / "out.jsonl"
    manifest = extract_dump(dump, out, source_id="fixture", namespaces=(0, 102), max_pages=2)
    rows = out.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert manifest["pages_kept"] == 2
    assert manifest["namespaces"] == [0, 102]


def test_extract_dump_rejects_missing_input(tmp_path):
    with pytest.raises(CorpusError, match="dump not found"):
        extract_dump(tmp_path / "nope.xml.bz2", tmp_path / "out.jsonl", source_id="fixture")


def test_extract_dump_requires_source_id(tmp_path):
    dump = _dump(tmp_path, [_page(1, "正文")])
    with pytest.raises(CorpusError, match="source_id is required"):
        extract_dump(dump, tmp_path / "out.jsonl", source_id="")


def test_extract_dump_accepts_rootless_stream_slice(tmp_path):
    """A multistream slice starts at a <page> and has no <mediawiki> wrapper."""
    fragment = "".join(PAGE.format(**_page(i, f"第{i}段正文内容")) for i in (1, 2, 3))
    path = tmp_path / "slice.xml.bz2"
    with bz2.open(path, "wb") as handle:
        handle.write(fragment.encode("utf-8"))
    out = tmp_path / "slice.jsonl"
    manifest = extract_dump(path, out, source_id="fixture")
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["1", "2", "3"]
    assert manifest["pages_kept"] == 3


@pytest.mark.parametrize("alias", ["same", "relative", "symlink-out", "symlink-in", "hardlink"])
def test_extract_rejects_input_output_aliases(tmp_path, monkeypatch, alias):
    dump = _dump(tmp_path, [_page(1, "正文")])
    original = file_info(dump)
    out = dump
    alias_path = tmp_path / "alias.xml.bz2"
    if alias == "relative":
        monkeypatch.chdir(tmp_path)
        out = Path(dump.name)
    elif alias == "symlink-out":
        alias_path.symlink_to(dump)
        out = alias_path
    elif alias == "symlink-in":
        alias_path.symlink_to(dump)
        dump = alias_path
    elif alias == "hardlink":
        alias_path.hardlink_to(dump)
        out = alias_path

    with pytest.raises(CorpusError, match="JSONL output path conflicts with input"):
        extract_dump(dump, out, source_id="fixture")

    assert file_info(dump) == file_info(out) == original
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("kind", ["existing-file", "dangling-symlink"])
def test_extract_preserves_existing_destination(tmp_path, kind):
    dump = _dump(tmp_path, [_page(1, "正文")])
    out = tmp_path / "out.jsonl"
    missing = tmp_path / "missing.jsonl"
    if kind == "existing-file":
        out.write_bytes(b"previous output")
    else:
        out.symlink_to(missing)
    with pytest.raises(CorpusError, match="JSONL output already exists"):
        extract_dump(dump, out, source_id="fixture")
    if kind == "existing-file":
        assert out.read_bytes() == b"previous output"
    else:
        assert out.readlink() == missing
        assert not missing.exists()


def test_extract_preserves_other_tasks_partial_files(tmp_path, monkeypatch):
    dump = _dump(tmp_path, [_page(1, "正文")])
    out = tmp_path / "out.jsonl"
    manifest_path = tmp_path / "manifest.json"
    legacy = tmp_path / "out.jsonl.partial"
    collision = tmp_path / ".wiki-dump-collides.partial"
    legacy.write_bytes(b"previous partial")
    collision.write_bytes(b"other worker")
    names = iter(["collides", "owned-jsonl", "owned-manifest"])
    monkeypatch.setattr(wiki_dump.tempfile, "_get_candidate_names", lambda: names)

    extract_dump(dump, out, source_id="fixture", manifest_path=manifest_path)

    assert out.is_file() and manifest_path.is_file()
    assert sorted(tmp_path.glob("*.partial")) == [collision, legacy]
    assert legacy.read_bytes() == b"previous partial"
    assert collision.read_bytes() == b"other worker"


def test_extract_does_not_stage_at_a_final_path(tmp_path, monkeypatch):
    dump = _dump(tmp_path, [])
    out = tmp_path / ".wiki-dump-output.partial"
    manifest_path = tmp_path / ".wiki-dump-manifest.partial"
    names = iter(["output"])
    monkeypatch.setattr(wiki_dump.tempfile, "_get_candidate_names", lambda: names)
    parse = wiki_dump.iterparse

    def check_paths(*args, **kwargs):
        assert not out.exists() and not manifest_path.exists()
        return parse(*args, **kwargs)

    monkeypatch.setattr(wiki_dump, "iterparse", check_paths)
    with pytest.raises(CorpusError, match="no pages kept"):
        extract_dump(dump, out, source_id="fixture", manifest_path=manifest_path)
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("conflict", ["jsonl", "manifest"])
def test_extract_preserves_destination_created_at_publication(tmp_path, monkeypatch, conflict):
    dump = _dump(tmp_path, [_page(1, "正文")])
    out = tmp_path / "out.jsonl"
    manifest_path = tmp_path / "manifest.json"
    target = out if conflict == "jsonl" else manifest_path
    link = os.link

    def competing_publish(source, destination):
        if destination == target:
            with target.open("xb") as other:
                other.write(b"other worker")
        link(source, destination)

    monkeypatch.setattr(wiki_dump.os, "link", competing_publish)
    message = "already exists" if conflict == "jsonl" else "verified JSONL published.*already exists"
    with pytest.raises(CorpusError, match=message):
        extract_dump(dump, out, source_id="fixture", manifest_path=manifest_path)
    assert target.read_bytes() == b"other worker"
    assert not list(tmp_path.glob("*.partial"))
    if conflict == "jsonl":
        assert not manifest_path.exists()
    else:
        assert json.loads(out.read_bytes())["text"] == "正文"


def test_extract_fails_safely_without_hard_link_support(tmp_path, monkeypatch):
    dump = _dump(tmp_path, [_page(1, "正文")])
    out = tmp_path / "out.jsonl"
    manifest_path = tmp_path / "manifest.json"

    def link(*args):
        raise OSError("hard links unsupported")

    monkeypatch.setattr(wiki_dump.os, "link", link)
    with pytest.raises(CorpusError, match="cannot publish.*atomic hard link.*unsupported"):
        extract_dump(dump, out, source_id="fixture", manifest_path=manifest_path)
    assert set(tmp_path.iterdir()) == {dump}


@pytest.mark.parametrize("failure", ["malformed-xml", "interrupted"])
def test_failed_extraction_cleans_staging_only(tmp_path, monkeypatch, failure):
    dump = tmp_path / "dump.xml"
    page = PAGE.format(**_page(1, "正文"))
    ending = "<broken></mediawiki>" if failure == "malformed-xml" else "</mediawiki>"
    dump.write_text(f"<mediawiki>{page}{ending}", encoding="utf-8")
    before = file_info(dump)
    out = tmp_path / "out.jsonl"
    manifest_path = tmp_path / "manifest.json"
    existing_partial = tmp_path / "out.jsonl.partial"
    existing_partial.write_bytes(b"untouched")

    if failure == "interrupted":
        extract_pages = wiki_dump._extract_pages

        def interrupted(*args, **kwargs):
            extract_pages(*args, **kwargs)
            raise KeyboardInterrupt

        monkeypatch.setattr(wiki_dump, "_extract_pages", interrupted)
        with pytest.raises(KeyboardInterrupt):
            extract_dump(dump, out, source_id="fixture", manifest_path=manifest_path)
    else:
        with pytest.raises(CorpusError, match="cannot extract.*mismatched tag"):
            extract_dump(dump, out, source_id="fixture", manifest_path=manifest_path)
    assert file_info(dump) == before
    assert existing_partial.read_bytes() == b"untouched"
    assert set(tmp_path.iterdir()) == {dump, existing_partial}


def _cli(dump, out, *options):
    return subprocess.run([sys.executable, str(ROOT / "scripts/extract_wiki_dump.py"), str(dump),
                           "--out", str(out), "--source-id", "fixture", *options],
                          capture_output=True, text=True, encoding="utf-8", check=False)


@pytest.mark.parametrize("conflict", ["input", "output", "symlink-parent", "default-exists"])
def test_cli_rejects_manifest_conflict_before_writing_jsonl(tmp_path, conflict):
    dump = _dump(tmp_path, [_page(1, "正文")])
    before = file_info(dump)
    out = tmp_path / "out.jsonl"
    if conflict == "input":
        result = _cli(dump, out, "--manifest", str(dump))
    elif conflict == "output":
        result = _cli(dump, out, "--manifest", str(out))
    elif conflict == "symlink-parent":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        result = _cli(dump, out, "--manifest", str(alias / out.name))
    else:
        manifest = tmp_path / "out.jsonl.manifest.json"
        manifest.write_bytes(b"prior manifest")
        result = _cli(dump, out)
        assert manifest.read_bytes() == b"prior manifest"
    assert result.returncode != 0 and not result.stdout
    assert "Manifest" in result.stderr
    assert "already exists" in result.stderr or "path conflicts" in result.stderr
    assert not out.exists()
    assert not list(tmp_path.glob("*.partial"))
    assert file_info(dump) == before


def test_standalone_manifest_writer_refuses_aliases_and_existing_file(tmp_path):
    dump = _dump(tmp_path, [_page(1, "正文")])
    out = tmp_path / "out.jsonl"
    manifest = extract_dump(dump, out, source_id="fixture")
    alias = tmp_path / "alias"
    alias.symlink_to(out)
    input_link = tmp_path / "input-link"
    input_link.hardlink_to(dump)
    for path in (dump, input_link, out, alias):
        with pytest.raises(CorpusError, match="Manifest path conflicts"):
            write_manifest(manifest, path)
    destination = tmp_path / "manifest.json"
    assert write_manifest(manifest, destination) == manifest
    with pytest.raises(CorpusError, match="Manifest already exists"):
        write_manifest({**manifest, "source_id": "another"}, destination)
    assert destination.read_bytes() == json_bytes(manifest)
    assert file_info(dump)["sha256"] == manifest["input"]["sha256"]
    assert file_info(out)["sha256"] == manifest["output"]["sha256"]


def test_parser_detaches_finished_pages_from_their_actual_parent(tmp_path, monkeypatch):
    dump = _dump(tmp_path, [_page(index, "正文") for index in range(60)])
    parse = wiki_dump.iterparse
    detached = []

    def observe(*args, **kwargs):
        stack = []
        for event, element in parse(*args, **kwargs):
            if event == "start":
                stack.append(element)
                yield event, element
            else:
                stack.pop()
                parent = stack[-1] if stack else None
                yield event, element
                if wiki_dump._local(element.tag) == "page":
                    assert parent is not None and element not in parent
                    detached.append(True)

    monkeypatch.setattr(wiki_dump, "iterparse", observe)
    assert len(list(iter_pages(dump))) == len(detached) == 60


def test_nested_xml_text_keeps_child_text_and_tails(tmp_path):
    dump = _dump(tmp_path, [_page(1, "前<em>中<page>的</page>间</em>后", title="原始标题", rev=41)])
    out = tmp_path / "out.jsonl"
    manifest = extract_dump(dump, out, source_id="fixture")
    assert manifest["pages_seen"] == 1
    assert out.read_bytes() == json_bytes({
        "id": "1", "namespace": 0, "parent_id": "1", "revision_id": "41",
        "source": "fixture", "text": "前中的间后", "title": "原始标题",
    })


def test_page_size_limit_counts_utf8_across_nested_text(tmp_path, monkeypatch):
    monkeypatch.setattr(wiki_dump, "MAX_PAGE_BYTES", 6)
    dump = _dump(tmp_path, [_page(1, "a<em>正文</em>"), _page(2, "正<em>文</em>"),
                            _page(3, "oversized")])
    out = tmp_path / "out.jsonl"
    manifest = extract_dump(dump, out, source_id="fixture")
    assert manifest["skipped"] == {"oversized_page": 2}
    assert json.loads(out.read_bytes())["id"] == "2"


@pytest.mark.parametrize("end", ["close-early", "invalid-bzip2", "invalid-xml"])
def test_parser_closes_compressed_and_raw_handles(tmp_path, monkeypatch, end):
    dump = _dump(tmp_path, [_page(1, "正文"), _page(2, "正文")])
    if end == "invalid-bzip2":
        dump.write_bytes(b"not a bzip2 stream")
    elif end == "invalid-xml":
        dump.write_bytes(bz2.compress(b"<mediawiki><broken></mediawiki>"))
    handles = []

    class TrackingBZ2(bz2.BZ2File):
        def __init__(self, raw):
            super().__init__(raw)
            handles.extend((raw, self))

    monkeypatch.setattr(wiki_dump.bz2, "BZ2File", TrackingBZ2)
    pages = iter_pages(dump)
    if end == "close-early":
        assert next(pages)[0]["id"] == "1"
        assert all(not handle.closed for handle in handles)
        pages.close()
    else:
        with pytest.raises(OSError if end == "invalid-bzip2" else ParseError):
            next(pages)
    assert len(handles) == 2 and all(handle.closed for handle in handles)


def test_parser_does_not_resolve_external_entities(tmp_path):
    external = tmp_path / "external.txt"
    external.write_text("external text", encoding="utf-8")
    dump = tmp_path / "dump.xml"
    page = PAGE.format(**_page(1, "&external;"))
    dump.write_text(f'<!DOCTYPE mediawiki [<!ENTITY external SYSTEM "{external.as_uri()}">]>'
                    f'<mediawiki>{page}</mediawiki>', encoding="utf-8")
    with pytest.raises(CorpusError, match="undefined entity"):
        extract_dump(dump, tmp_path / "out.jsonl", source_id="fixture")


@pytest.mark.parametrize("change", ["rewrite", "replace-identical", "truncate", "delete", "repoint-symlink"])
def test_input_identity_changes_fail_before_publication(tmp_path, monkeypatch, change):
    dump = _dump(tmp_path, [_page(1, "正文"), _page(2, "正文")])
    original = dump.read_bytes()
    input_path = dump
    replacement = tmp_path / "replacement.xml.bz2"
    if change == "repoint-symlink":
        input_path = tmp_path / "alias.xml.bz2"
        input_path.symlink_to(dump)
    if change in {"replace-identical", "repoint-symlink"}:
        replacement.write_bytes(original)
    out = tmp_path / "out.jsonl"
    manifest_path = tmp_path / "manifest.json"
    extract_pages = wiki_dump._extract_pages

    def changed_input(*args, **kwargs):
        result = extract_pages(*args, **kwargs)
        if change == "rewrite":
            dump.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        elif change == "replace-identical":
            replacement.replace(dump)
        elif change == "truncate":
            dump.write_bytes(original[:10])
        elif change == "delete":
            dump.unlink()
        else:
            input_path.unlink()
            input_path.symlink_to(replacement)
        return result

    monkeypatch.setattr(wiki_dump, "_extract_pages", changed_input)
    with pytest.raises(CorpusError, match="input identity changed"):
        extract_dump(input_path, out, source_id="fixture", manifest_path=manifest_path,
                     max_pages=1 if change == "repoint-symlink" else None)
    assert not out.exists() and not manifest_path.exists()
    assert not list(tmp_path.glob("*.partial"))
    if change == "delete":
        assert not dump.exists()
    elif change == "truncate":
        assert dump.read_bytes() == original[:10]
    elif change == "rewrite":
        assert dump.read_bytes() == original[:-1] + bytes([original[-1] ^ 1])
    else:
        assert dump.read_bytes() == input_path.read_bytes() == original


@pytest.mark.parametrize("failure", ["sha256", "bytes", "unreadable"])
def test_input_reverification_checks_complete_file_info(tmp_path, monkeypatch, failure):
    dump = _dump(tmp_path, [_page(1, "正文")])
    out = tmp_path / "out.jsonl"
    manifest_path = tmp_path / "manifest.json"
    before = file_info(dump)
    input_checks = []

    def changed_file_info(path):
        info = file_info(path)
        if path == dump:
            input_checks.append(info)
            if len(input_checks) == 2:
                if failure == "unreadable":
                    raise PermissionError("input cannot be read")
                return {**info, failure: "0" * 64 if failure == "sha256" else -1}
        return info

    monkeypatch.setattr(wiki_dump, "file_info", changed_file_info)
    with pytest.raises(CorpusError, match="input identity"):
        extract_dump(dump, out, source_id="fixture", manifest_path=manifest_path)
    assert input_checks == [before, before]
    assert file_info(dump) == before
    assert set(tmp_path.iterdir()) == {dump}


def test_input_is_hashed_before_and_after_and_manifest_binds_verification(tmp_path, monkeypatch):
    dump = _dump(tmp_path, [_page(1, "正文与[[链接|文字]]。", title="标题", rev=32)])
    out = tmp_path / "out.jsonl"
    manifest_path = tmp_path / "manifest.json"
    hashes = []

    def observe(path):
        info = file_info(path)
        assert not out.exists() and not manifest_path.exists()
        if path == dump:
            hashes.append(info)
        return info

    monkeypatch.setattr(wiki_dump, "file_info", observe)
    manifest = extract_dump(dump, out, source_id="fixture", manifest_path=manifest_path,
                            max_pages=2)
    assert hashes == [file_info(dump)] * 2
    assert manifest["input"] == {"path": str(dump), **hashes[0]}
    assert manifest["output"] == {"path": str(out), **file_info(out)}
    assert manifest["complete_input_scan"] is True and manifest["partial_extraction"] is False
    assert manifest["input_verified_after_extraction"] is True
    assert manifest["extraction_status"] == "COMPLETE" and manifest["max_pages"] == 2
    assert manifest_path.read_bytes() == json_bytes(manifest)
    assert manifest["sha256"] == identity({k: v for k, v in manifest.items() if k != "sha256"})
    assert out.read_bytes() == json_bytes({
        "id": "1", "namespace": 0, "parent_id": "1", "revision_id": "32",
        "source": "fixture", "text": "正文与文字。", "title": "标题",
    })


def test_cli_page_cap_marks_partial_scan_with_full_dump_identity(tmp_path):
    dump = _dump(tmp_path, [_page(1, "模板", ns=10), _page(2, "正文"), _page(3, "后续正文")])
    out = tmp_path / "out.jsonl"
    result = _cli(dump, out, "--max-pages", "1")
    assert result.returncode == 0, result.stderr
    manifest = json.loads(out.with_suffix(".jsonl.manifest.json").read_bytes())
    assert json.loads(result.stdout) == manifest
    assert manifest["extraction_status"] == "PARTIAL"
    assert manifest["partial_extraction"] is True and manifest["complete_input_scan"] is False
    assert manifest["max_pages"] == manifest["pages_kept"] == 1
    assert manifest["pages_seen"] == 2 and manifest["skipped"] == {"namespace": 1}
    assert manifest["input_verified_after_extraction"] is True
    assert manifest["input"] == {"path": str(dump), **file_info(dump)}


@pytest.mark.parametrize(("pages", "skipped"), [
    ([], {}),
    ([_page(1, "模板", ns=10)], {"namespace": 1}),
    ([_page(1, "重定向", extra='<redirect title="目标"/>')], {"redirect": 1}),
    ([_page(1, "{{模板}}")], {"empty_after_cleaning": 1}),
])
def test_extract_refuses_empty_jsonl_and_records_skip_diagnostics(tmp_path, pages, skipped):
    dump = _dump(tmp_path, pages)
    out = tmp_path / "out.jsonl"
    manifest_path = tmp_path / "manifest.json"
    with pytest.raises(CorpusError, match="no pages kept") as error:
        extract_dump(dump, out, source_id="fixture", manifest_path=manifest_path)
    assert f"pages_seen={len(pages)}, skipped={skipped}" in str(error.value)
    assert set(tmp_path.iterdir()) == {dump}


def test_cli_empty_dump_fails_without_success_manifest(tmp_path):
    dump = tmp_path / "empty.xml"
    dump.touch()
    result = _cli(dump, tmp_path / "out.jsonl")
    assert result.returncode != 0 and not result.stdout
    assert "no pages kept (pages_seen=0, skipped={})" in result.stderr
    assert set(tmp_path.iterdir()) == {dump}


@pytest.mark.parametrize("max_pages", [0, -1, True, False, 1.0, "1"])
def test_invalid_max_pages_is_rejected_before_io(tmp_path, max_pages):
    with pytest.raises(CorpusError, match="max_pages must be a positive integer"):
        extract_dump(tmp_path / "missing.xml", tmp_path / "new" / "out.jsonl",
                     source_id="fixture", max_pages=max_pages)
    assert not list(tmp_path.iterdir())


def _run_scale_extraction(directory, count):
    directory.mkdir()
    dump = directory / "dump.xml"
    with dump.open("w", encoding="utf-8") as handle:
        handle.write('<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">')
        for index in range(count):
            handle.write(PAGE.format(**_page(index, f"合成正文{index}，含[[链接|文字]]。")))
        handle.write("</mediawiki>")
    out = directory / "out.jsonl"
    worker = """
import json
import sys
import time
from minillm.wiki_dump import extract_dump


def peak_rss_kib():
    if sys.platform.startswith("linux"):
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    if sys.platform == "darwin":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    return None


baseline = peak_rss_kib()
start = time.monotonic()
manifest = extract_dump(sys.argv[1], sys.argv[2], source_id="fixture")
print(json.dumps({"pages": manifest["pages_kept"], "pages_seen": manifest["pages_seen"],
                  "status": manifest["extraction_status"], "seconds": time.monotonic() - start,
                  "baseline_kib": baseline, "peak_kib": peak_rss_kib()}))
"""
    result = subprocess.run([sys.executable, "-c", worker, str(dump), str(out)],
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                            capture_output=True, text=True, encoding="utf-8", timeout=120, check=True)
    metrics = json.loads(result.stdout)
    print(f"Synthetic XML, {count:,} pages: {metrics['seconds']:.3f} s, "
          f"peak RSS {metrics['peak_kib']} KiB (pre-extraction {metrics['baseline_kib']} KiB)")
    assert metrics["pages_seen"] == metrics["pages"] == count
    assert metrics["status"] == "COMPLETE"
    return out, metrics


def test_extract_10000_synthetic_pages_preserves_every_record(tmp_path):
    out, _ = _run_scale_extraction(tmp_path / "10k", 10_000)
    count = 0
    with out.open("rb") as rows:
        for index, row in enumerate(rows):
            assert row == json_bytes({
                "id": str(index), "parent_id": str(index), "namespace": 0, "revision_id": "1",
                "source": "fixture", "text": f"合成正文{index}，含文字。", "title": "Fixture page",
            })
            count += 1
    assert count == 10_000


@pytest.mark.skipif(not sys.platform.startswith("linux") and sys.platform != "darwin",
                    reason="peak RSS measurement requires Linux or macOS")
def test_extract_100000_synthetic_pages_has_bounded_rss(tmp_path):
    _, small = _run_scale_extraction(tmp_path / "10k", 10_000)
    out, large = _run_scale_extraction(tmp_path / "100k", 100_000)
    with out.open("rb") as rows:
        for count, row in enumerate(rows, 1):
            pass
    assert count == 100_000 and json.loads(row)["id"] == "99999"
    small_growth = small["peak_kib"] - small["baseline_kib"]
    large_growth = large["peak_kib"] - large["baseline_kib"]
    # Use the worker's own baseline to allow for import/allocator noise across processes.
    assert large_growth <= max(2 * small_growth, small["baseline_kib"] // 100), (small, large)
