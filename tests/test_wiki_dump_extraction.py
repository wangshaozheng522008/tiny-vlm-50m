"""Wiki dump extraction: one page becomes one traceable JSONL record."""
import bz2
import json

import pytest

from minillm.corpus_utils import CorpusError
from minillm.wiki_dump import clean_wikitext, extract_dump

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
