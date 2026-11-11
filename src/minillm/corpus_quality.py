"""Small, explainable CPU corpus rules. No language model or network access."""
from __future__ import annotations

import hashlib
import html
import re
import unicodedata

import numpy as np

CATEGORIES = ("zh", "en", "code", "mixed", "unknown")
CJK = re.compile(r"[\u3400-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")
CODE = re.compile(r"(?m)^\s*(?:def |class |import |from \w+ import |#include|function |const |let |public |SELECT |CREATE TABLE)|=>|```|</?(?:div|html|body)\b")
CONTACT = re.compile(r"https?://\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:微信|QQ|电话|tel)\s*[:：]?\s*[\w-]{5,}", re.I)
AD = re.compile(r"加微信|扫码领取|点击购买|限时优惠|免费领取|博彩|buy now|limited offer", re.I)
BLOCK_HTML = re.compile(r"<(?:script|style|nav|header|footer)\b[^>]*>.*?</(?:script|style|nav|header|footer)\s*>", re.I | re.S)
TAGS = re.compile(r"</?[A-Za-z][A-Za-z0-9:_-]*(?:\s+[^<>]{0,2048})?\s*/?>")
HTML_MARKUP = re.compile(r"</[A-Za-z][A-Za-z0-9:_-]*\s*>|<(?:html|body|div|p|br|nav|script|style|article|section|table)(?:\s|/?>)", re.I)

DEFAULT_QUALITY = {
    "min_chars": 80, "max_chars": 20000, "max_parent_chars": 200000,
    "max_record_bytes": 1048576, "long_document": "split", "strip_html": True,
    "unicode_checks": True, "repetition_checks": True, "advertising_checks": True,
    "category_min_chars": {"code": 20, "unknown": 20},
    "min_information_ratio": {"zh": 0.15, "en": 0.35, "code": 0.10, "mixed": 0.15, "unknown": 0.15},
}


def classify(text: str, hint: str | None = None) -> str:
    if hint in CATEGORIES and hint != "unknown":
        return hint
    # HTML is cleaned as prose, rather than misclassified as source code.
    if CODE.search(text) and not re.search(r"<(?:html|body|div|p|nav)\b", text, re.I):
        return "code"
    zh, en = len(CJK.findall(text)), len(LATIN.findall(text))
    if zh and en > max(12, zh * 0.4):
        return "mixed"
    if zh:
        return "zh"
    if en >= max(4, len(text) * 0.2):
        return "en"
    return "unknown"


def normalise(text: str, category: str = "unknown", *, strip_html: bool = True) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    if category != "code":
        text = text.replace("\u00a0", " ").replace("\u3000", " ")
        if strip_html:
            # Plain inequalities and <variable> notation are not HTML tags.
            if HTML_MARKUP.search(text):
                text = BLOCK_HTML.sub("\n", text)
                text = re.sub(r"</?(?:p|br|div|li|h[1-6])\b[^>]*>", "\n", text, flags=re.I)
                text = TAGS.sub("", text)
            text = html.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip("\r\n " if category != "code" else "\r\n")


def canonical(text: str, category: str) -> str:
    return text if category == "code" else re.sub(r"\s+", " ", text).strip()


def content_hash(text: str, category: str) -> str:
    return hashlib.sha256(canonical(text, category).encode("utf-8")).hexdigest()


def quality_issue(text: str, category: str, cfg: dict) -> str | None:
    if not text.strip():
        return "empty"
    if len(text) < cfg["category_min_chars"].get(category, cfg["min_chars"]):
        return "too_short"
    if len(text) > cfg["max_parent_chars"]:
        return "parent_too_long"
    if len(text) > cfg["max_chars"] and cfg["long_document"] == "reject":
        return "too_long"
    if cfg["unicode_checks"]:
        if "\ufffd" in text:
            return "replacement_character"
        if any(unicodedata.category(c) in ("Cs", "Cn") or (unicodedata.category(c) == "Cc" and c not in "\n\t") for c in text):
            return "invalid_unicode"
    informative = sum(c.isalnum() for c in text)
    # Numeric/formula texts remain usable: a Chinese-letter-only rule would reject them.
    if informative / len(text) < cfg["min_information_ratio"].get(category, 0.15):
        return "low_information"
    if cfg["advertising_checks"] and category != "code":
        if len(CONTACT.findall(text)) >= 5 or len(AD.findall(text)) >= 3:
            return "advertising_contacts"
    if cfg["repetition_checks"] and category != "code":
        if re.search(r"(.)\1{29,}", text):
            return "repeated_character"
        sentences = [s.strip() for s in re.split(r"[\n。！？!?.]", text) if len(s.strip()) >= 10]
        if len(sentences) >= 4 and (len(sentences) - len(set(sentences))) / len(sentences) >= 0.5:
            return "repeated_sentence_or_paragraph"
    return None


def fragments(text: str, limit: int):
    """Prefer paragraph boundaries; even a single huge paragraph has a hard bound."""
    pending = ""
    for paragraph in text.splitlines(keepends=True):
        if pending and len(pending) + len(paragraph) > limit:
            yield pending.rstrip("\r\n")
            pending = ""
        while len(paragraph) > limit:
            yield paragraph[:limit]
            paragraph = paragraph[limit:]
        pending += paragraph
    if pending.strip():
        yield pending.rstrip("\r\n")


def shingles(text: str, category: str, ngram: int = 4) -> set[str]:
    text = canonical(text, category).casefold()
    return {text[i:i + ngram] for i in range(max(1, len(text) - ngram + 1))}


def simhash(features: set[str]) -> int:
    """64-bit, equal-weight full-document character shingles, bounded by one document."""
    votes = np.zeros(64, dtype=np.int64)
    block = bytearray()
    for feature in features:
        block.extend(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest())
        if len(block) >= 65536:
            bits = np.unpackbits(np.frombuffer(block, dtype=np.uint8)).reshape(-1, 64)
            votes += bits.sum(axis=0).astype(np.int64) * 2 - len(bits)
            block.clear()
    if block:
        bits = np.unpackbits(np.frombuffer(block, dtype=np.uint8)).reshape(-1, 64)
        votes += bits.sum(axis=0).astype(np.int64) * 2 - len(bits)
    result = 0
    for vote in votes:
        result = (result << 1) | int(vote >= 0)
    return result


def bands(fingerprint: int, radius: int):
    offset = 0
    for index in range(radius + 1):
        width = 64 // (radius + 1) + int(index < 64 % (radius + 1))
        yield index, (fingerprint >> offset) & ((1 << width) - 1)
        offset += width


def similar(a: set[str], b: set[str], threshold: float) -> bool:
    intersection = len(a & b)
    return intersection / max(len(a) + len(b) - intersection, 1) >= threshold
