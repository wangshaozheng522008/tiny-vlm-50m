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
import os
import re
import stat
import tempfile
import time
from contextlib import ExitStack, closing, contextmanager, nullcontext
from pathlib import Path
from xml.etree.ElementTree import ParseError, iterparse

from .corpus_utils import CorpusError, file_info, identity, json_bytes

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


def _bounded_text(element):
    parts = []
    remaining = MAX_PAGE_BYTES
    block_chars = 64 * 1024
    for text in element.itertext():
        if len(text) > remaining:
            return None
        if text.isascii():
            remaining -= len(text)
        else:
            for start in range(0, len(text), block_chars):
                remaining -= len(text[start:start + block_chars].encode("utf-8"))
                if remaining < 0:
                    return None
        parts.append(text)
    return "".join(parts)


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
    text = _bounded_text(text_element)
    if text is None:
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


def _document_reader(handle):
    head = handle.read(4096)
    if b"<mediawiki" in head or head.lstrip()[:5] == b"<?xml":
        return _FragmentReader(handle, b"", b"", head=head)
    return _FragmentReader(handle, b"<mediawiki>", b"</mediawiki>", head=head)


def _iter_pages(raw, *, compressed, namespaces):
    with (bz2.BZ2File(raw) if compressed else nullcontext(raw)) as handle:
        reader = _document_reader(handle)
        stack = []
        page = None
        for event, element in iterparse(reader, events=("start", "end")):
            if event == "start":
                stack.append(element)
                if page is None and _local(element.tag) == "page":
                    page = element
                continue

            stack.pop()
            finished_page = element is page
            if finished_page:
                record, reason = _page_record(element, namespaces)
                page = None
            if page is None:
                if stack:
                    stack[-1].remove(element)
                element.clear()
            if finished_page:
                yield record, reason


def iter_pages(dump_path, *, namespaces=DEFAULT_NAMESPACES, log=None):
    """Yield ``(record, None)`` or ``(None, skip_reason)`` for every page."""
    path = Path(dump_path)
    with path.open("rb") as raw:
        yield from _iter_pages(raw, compressed=path.suffix == ".bz2", namespaces=namespaces)


def _destination(path):
    path = Path(path)
    try:
        return path.parent.resolve() / path.name
    except RuntimeError as exc:
        raise CorpusError(f"cannot resolve output directory for {path}: {exc}") from exc


def _check_destination(path, label, **reserved):
    for name, other in reserved.items():
        try:
            same_file = path == other or path.samefile(other)
        except FileNotFoundError:
            same_file = False
        if same_file:
            raise CorpusError(f"{label} path conflicts with {name}: {path}")
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise CorpusError(f"{label} already exists: {path}")


@contextmanager
def _temporary_output(path, *other_destinations):
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = ".wiki-dump-"
    # A caller may choose a final filename that looks like a staging filename.
    while any(target.name.casefold().startswith(prefix) for target in (path, *other_destinations)):
        prefix = "." + prefix
    sink = tempfile.NamedTemporaryFile(dir=path.parent, prefix=prefix,
                                       suffix=".partial", delete=False)
    temporary = Path(sink.name)
    try:
        with sink:
            yield temporary, sink
    finally:
        temporary.unlink(missing_ok=True)


def _publish(temporary, path):
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise CorpusError(f"output already exists: {path}") from exc
    except (OSError, NotImplementedError) as exc:
        raise CorpusError(f"cannot publish {path} with an atomic hard link: {exc}") from exc


def _input_stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _check_input(path, raw, expected):
    try:
        opened = _input_stamp(os.fstat(raw.fileno()))
        current = _input_stamp(path.stat())
    except OSError as exc:
        raise CorpusError(f"input identity changed: {path}: {exc}") from exc
    if opened != expected or current != expected:
        raise CorpusError(f"input identity changed: {path} (file or metadata differs)")


def _verified_file_info(path, raw, expected):
    _check_input(path, raw, expected)
    try:
        info = file_info(path)
    except OSError as exc:
        raise CorpusError(f"cannot verify input identity (file changed or became unreadable): {path}: {exc}") from exc
    _check_input(path, raw, expected)
    return info


def _extract_pages(raw, sink, *, compressed, namespaces, source_id, max_pages):
    skipped = {}
    pages = kept = characters = 0
    complete = True
    with closing(_iter_pages(raw, compressed=compressed, namespaces=namespaces)) as records:
        for record, reason in records:
            pages += 1
            if reason is not None:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            record["source"] = source_id
            sink.write(json_bytes(record))
            kept += 1
            characters += len(record["text"])
            if kept == max_pages:
                complete = False
                break
    return pages, kept, characters, dict(sorted(skipped.items())), complete


def extract_dump(dump_path, out_path, *, source_id, namespaces=DEFAULT_NAMESPACES,
                 max_pages=None, log=None, manifest_path=None):
    """Publish verified JSONL, optionally followed by its conversion manifest."""
    if not isinstance(source_id, str) or not source_id:
        raise CorpusError("source_id is required")
    if max_pages is not None and (type(max_pages) is not int or max_pages <= 0):
        raise CorpusError("max_pages must be a positive integer")
    path = Path(dump_path)
    try:
        original = path.stat()
    except FileNotFoundError as exc:
        raise CorpusError(f"dump not found: {path}") from exc
    if not stat.S_ISREG(original.st_mode):
        raise CorpusError(f"dump is not a regular file: {path}")
    input_path = path.resolve()
    out = _destination(out_path)
    _check_destination(out, "JSONL output", input=input_path)
    manifest_out = _destination(manifest_path) if manifest_path is not None else None
    if manifest_out is not None:
        _check_destination(manifest_out, "Manifest", input=input_path, JSONL=out)
        if out in manifest_out.parents or manifest_out in out.parents:
            raise CorpusError("Manifest and JSONL paths must be separate files")

    expected = _input_stamp(original)
    namespaces = tuple(namespaces)
    started = time.monotonic()
    try:
        raw = path.open("rb")
    except OSError as exc:
        raise CorpusError(f"cannot open input for identity verification: {path}: {exc}") from exc
    with raw, ExitStack() as staged:
        before = _verified_file_info(path, raw, expected)
        other_destinations = (manifest_out,) if manifest_out is not None else ()
        temporary, sink = staged.enter_context(_temporary_output(out, *other_destinations))
        try:
            pages, kept, characters, skipped, complete = _extract_pages(
                raw, sink, compressed=path.suffix == ".bz2", namespaces=namespaces,
                source_id=source_id, max_pages=max_pages,
            )
        except (OSError, EOFError, ParseError) as exc:
            _check_input(path, raw, expected)
            raise CorpusError(f"cannot extract {path}: {exc}") from exc
        sink.close()
        output_info = file_info(temporary)
        if _verified_file_info(path, raw, expected) != before:
            raise CorpusError(f"input identity changed: {path} (SHA256 or byte size differs)")
        if not kept:
            raise CorpusError(f"no pages kept (pages_seen={pages}, skipped={skipped})")

        manifest = {
            "schema": "wiki-dump-extraction/1",
            "source_id": source_id,
            "input": {"path": str(input_path), **before},
            "output": {"path": str(out), **output_info},
            "namespaces": sorted(namespaces),
            "pages_seen": pages,
            "pages_kept": kept,
            "characters": characters,
            "skipped": skipped,
            "seconds": round(time.monotonic() - started, 3),
            "cleaner": "clean_wikitext/1",
            "extraction_status": "COMPLETE" if complete else "PARTIAL",
            "complete_input_scan": complete,
            "partial_extraction": not complete,
            "max_pages": max_pages,
            "input_verified_after_extraction": True,
        }
        manifest["sha256"] = identity(manifest)
        if manifest_out is not None:
            manifest_temp, manifest_sink = staged.enter_context(_temporary_output(manifest_out, out))
            manifest_sink.write(json_bytes(manifest))
            manifest_sink.close()

        _check_input(path, raw, expected)
        _publish(temporary, out)
        if manifest_out is not None:
            try:
                _publish(manifest_temp, manifest_out)
            except CorpusError as exc:
                raise CorpusError(f"verified JSONL published at {out}; Manifest publication failed: {exc}") from exc
    return manifest


def write_manifest(manifest, path):
    destination = _destination(path)
    input_path = _destination(manifest["input"]["path"])
    output_path = _destination(manifest["output"]["path"])
    _check_destination(destination, "Manifest", input=input_path, JSONL=output_path)
    with _temporary_output(destination, input_path, output_path) as (temporary, sink):
        sink.write(json_bytes(manifest))
        sink.close()
        _publish(temporary, destination)
    return manifest
