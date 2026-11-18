"""Wikimedia ``pages-articles`` XML extraction into canonical JSONL.

Raw dumps are not admissible V4 input: ``corpus_v4.inventory`` only keeps TXT and
JSONL files, so a ``.xml.bz2`` page dump cannot enter the pipeline by itself.
This module turns one wiki page into one JSONL record and makes no quality
decision -- length limits, language checks and every rejection rule stay in
``corpus_v4`` so that extraction never silently filters text.

Wiki markup is reduced to readable text with a deliberately small, explainable
rule set. Anything this cleaner cannot express (tables, templates, parser
functions) is removed rather than guessed at, and the record keeps the original
page id so later stages can trace a document back to its page.

One trade-off is intentional: a template whose parameter text contains braces
survives, because the same shape appears in LaTeX formulae (``{{x^{c+1}}``)
where the braces *are* the content. Deleting them would destroy maths pages, so
the residue is left for the V4 quality rules to judge instead of being guessed
away here.
"""
from __future__ import annotations

import bz2
import html
import json
import os
import re
import time
from pathlib import Path
from xml.etree.ElementTree import iterparse

from .corpus_utils import CorpusError, file_info, identity, json_bytes, write_json

DEFAULT_NAMESPACES = (0,)
MAX_PAGE_BYTES = 8 * 1024 * 1024

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF_PAIR = re.compile(r"<ref\b[^>]*>.*?</ref\s*>", re.S)
_REF_SELF = re.compile(r"<ref\b[^>]*/\s*>", re.S)
_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
_GALLERY = re.compile(r"<gallery\b[^>]*>.*?</gallery\s*>", re.S)
_MAGIC = re.compile(r"__[A-Z_]+__")
_LINK_FILE = re.compile(r"\[\[(?:[A-Za-z]+|文件|檔案|图像|圖像|分类|分類|作者|翻譯|翻译)\s*:[^\]]*\]\]")
_LINK = re.compile(r"\[\[([^\]|]*)(?:\|([^\]]*))?\]\]")
_EXTLINK_LABELLED = re.compile(r"\[(?:https?|ftp)://[^\s\]]+\s+([^\]]*)\]")
_EXTLINK_BARE = re.compile(r"\[(?:https?|ftp)://[^\]]*\]")
_HTML = re.compile(r"</?[A-Za-z][^>]*>")
_HEADING = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.M)
_LIST_MARKER = re.compile(r"^[ *#:;]+", re.M)
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}", re.S)
_QUOTES = re.compile(r"'{2,}")
_SPACES = re.compile(r"[ \t\u00a0]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_DROP_LINE = re.compile(r"^[\s\W_]*$", re.U)


def clean_wikitext(text: str) -> str:
    """Reduce wiki markup to plain text without inventing or reordering content."""
    if not text:
        return ""
    value = _COMMENT.sub("", text)
    value = _REF_PAIR.sub("", value)
    value = _REF_SELF.sub("", value)
    value = _GALLERY.sub("", value)
    value = _TABLE.sub("", value)
    value = _MAGIC.sub("", value)
    # Nested templates need repetition; the bound keeps hostile input finite.
    for _ in range(12):
        reduced = _TEMPLATE.sub("", value)
        if reduced == value:
            break
        value = reduced
    value = _LINK_FILE.sub("", value)
    value = _LINK.sub(lambda m: (m.group(2) if m.group(2) is not None else m.group(1)) or "", value)
    value = _EXTLINK_LABELLED.sub(lambda m: m.group(1), value)
    value = _EXTLINK_BARE.sub("", value)
    value = _HTML.sub("", value)
    value = html.unescape(html.unescape(value))
    value = _HEADING.sub(lambda m: m.group(1), value)
    value = _LIST_MARKER.sub("", value)
    value = _QUOTES.sub("", value)
    value = _SPACES.sub(" ", value)
    lines = [line.strip() for line in value.splitlines()]
    lines = [line for line in lines if line and not _DROP_LINE.match(line)]
    value = "\n".join(lines)
    return _BLANK_LINES.sub("\n\n", value).strip()


def _local(tag):
    """Drop the export XML namespace; ElementTree keeps ``{uri}page`` tags."""
    return tag.rsplit("}", 1)[-1]


def _child(element, name):
    for child in element:
        if _local(child.tag) == name:
            return child
    return None


def _childtext(element, name):
    child = _child(element, name)
    return child.text if child is not None else None


def _page_record(page, namespaces):
    title = (_childtext(page, "title") or "").strip()
    ns_text = (_childtext(page, "ns") or "").strip()
    try:
        namespace = int(ns_text)
    except ValueError:
        return None, "invalid_namespace"
    if namespace not in namespaces:
        return None, "namespace"
    if _child(page, "redirect") is not None:
        return None, "redirect"
    page_id = (_childtext(page, "id") or "").strip()
    if not page_id:
        return None, "missing_page_id"
    revision = _child(page, "revision")
    text_element = _child(revision, "text") if revision is not None else None
    if text_element is None or text_element.get("deleted") is not None:
        return None, "text_removed"
    text = text_element.text or ""
    if len(text.encode("utf-8", "ignore")) > MAX_PAGE_BYTES:
        return None, "oversized_page"
    cleaned = clean_wikitext(text)
    if not cleaned:
        return None, "empty_after_cleaning"
    return {
        "id": page_id,
        "parent_id": page_id,
        "title": title,
        "revision_id": (_childtext(revision, "id") or "").strip() or None,
        "namespace": namespace,
        "text": cleaned,
    }, None


class _FragmentReader:
    """Re-emit a page fragment list as one well-formed XML document.

    A ``multistream`` dump sliced on bzip2 stream boundaries is a sequence of
    ``<page>`` elements without the ``<mediawiki>`` wrapper, so a stock parser
    reports "junk after document element". Slicing is the only affordable way to
    sample a 10 GiB dump, so the wrapper is restored here instead of forcing the
    caller back to a full download.
    """

    def __init__(self, handle, prefix, suffix, head=b""):
        self._handle = handle
        self._prefix = prefix
        self._suffix = suffix
        self._head = head
        self._state = 0

    def read(self, size=-1):
        if self._state == 0:
            self._state = 1
            if self._prefix:
                return self._prefix
        if self._head:
            data, self._head = self._head, b""
            return data
        data = self._handle.read(size)
        if data:
            return data
        if self._suffix:
            suffix, self._suffix = self._suffix, b""
            return suffix
        return b""


def _open_document(path):
    """Open a dump and, for bare page fragments, wrap it in a synthetic root."""
    opener = bz2.open if path.suffix == ".bz2" else open
    handle = opener(path, "rb")
    head = handle.read(4096)
    if b"<mediawiki" in head or head.lstrip()[:5] == b"<?xml":
        return handle, _FragmentReader(handle, b"", b"", head=head), False
    return handle, _FragmentReader(handle, b"<mediawiki>", b"</mediawiki>", head=head), True


def iter_pages(dump_path, *, namespaces=DEFAULT_NAMESPACES, log=None):
    """Yield ``(record, None)`` or ``(None, skip_reason)`` for every page."""
    path = Path(dump_path)
    handle, reader, wrapped = _open_document(path)
    try:
        context = iterparse(reader, events=("end",))
        for _, element in context:
            if _local(element.tag) != "page":
                continue
            record, reason = _page_record(element, namespaces)
            element.clear()
            yield record, reason
    finally:
        handle.close()


def extract_dump(dump_path, out_path, *, source_id, namespaces=DEFAULT_NAMESPACES,
                 max_pages=None, log=None):
    """Write one JSONL record per page and return a manifest of the conversion."""
    path = Path(dump_path)
    if not path.is_file():
        raise CorpusError(f"dump not found: {path}")
    if not isinstance(source_id, str) or not source_id:
        raise CorpusError("source_id is required")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".partial")
    skipped = {}
    pages = kept = characters = 0
    started = time.monotonic()
    before = file_info(path)
    try:
        with temporary.open("wb") as sink:
            for record, reason in iter_pages(path, namespaces=namespaces, log=log):
                pages += 1
                if reason is not None:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue
                record["source"] = source_id
                sink.write(json_bytes(record))
                kept += 1
                characters += len(record["text"])
                if max_pages is not None and kept >= max_pages:
                    break
        os.replace(temporary, out)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = {
        "schema": "wiki-dump-extraction/1",
        "source_id": source_id,
        "input": {"path": str(path.resolve()), **before},
        "output": {"path": str(out.resolve()), **file_info(out)},
        "namespaces": sorted(namespaces),
        "pages_seen": pages,
        "pages_kept": kept,
        "characters": characters,
        "skipped": dict(sorted(skipped.items())),
        "seconds": round(time.monotonic() - started, 3),
        "cleaner": "clean_wikitext/1",
    }
    manifest["sha256"] = identity({k: v for k, v in manifest.items() if k != "sha256"})
    return manifest


def write_manifest(manifest, path):
    write_json(Path(path), manifest)
    return manifest
