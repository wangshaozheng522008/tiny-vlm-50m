"""Small, explainable CPU corpus rules. No language model or network access.

``quality_issue`` is the frozen v1/v2 single-reason filter used by the legacy
pipeline.  ``evaluate_quality`` is the versioned v4 rule engine: every rule
returns a structured finding (rule id, severity, observed value, threshold,
action, explanation) so a reviewer can see exactly what was measured.  The v4
engine never deletes a document merely for containing digits, code, technical
terms or some English; those rules are REVIEW-level and code is exempt from the
prose-specific density rules.
"""
from __future__ import annotations

import hashlib
import html
import math
import re
import unicodedata
from dataclasses import dataclass

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


def raw_content_hash(text: str) -> str:
    """Category-neutral original identity; preserve indentation and meaningful spaces."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def comparison_hash(text: str) -> str:
    """Category-neutral split guard, not a deletion key for whitespace-sensitive code."""
    text = unicodedata.normalize("NFC", text)
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


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


# ---------------------------------------------------------------------------
# v4 rule engine
# ---------------------------------------------------------------------------

QUALITY_RULES_VERSION = "corpus-quality-v4.1"
QUALITY_ACTIONS = ("KEEP", "REVIEW", "REJECT")
_ACTION_RANK = {"KEEP": 0, "REVIEW": 1, "REJECT": 2}
_SEVERITY = {"KEEP": "info", "REVIEW": "warning", "REJECT": "error"}

V4_CATEGORIES = ("zh_general", "zh_encyclopedic", "zh_education", "en_general", "math_logic_code")
_EXPECTED_LANGUAGE = {
    "zh_general": ("zh", "mixed"),
    "zh_encyclopedic": ("zh", "mixed"),
    "zh_education": ("zh", "mixed"),
    "en_general": ("en", "mixed"),
    "math_logic_code": ("zh", "en", "mixed", "code", "unknown"),
}

_RULE_FIELDS = {
    "empty_or_short": {"min_chars"},
    "extreme_length": {"max_chars"},
    "repeated_characters": {"max_run"},
    "replacement_characters": {"max_ratio"},
    "html_residue": {"max_tags", "max_tag_ratio"},
    "boilerplate_noise": {"max_matches"},
    "punctuation_density": {"min_ratio", "max_ratio"},
    "digit_density": {"max_ratio"},
    "repeated_paragraph": {"min_repeats", "max_ratio"},
    "language_mismatch": set(),
    "unprintable_ratio": {"max_ratio"},
    "pii_contacts": {"max_contacts"},
    "low_information": {"min_ratio"},
}

DEFAULT_QUALITY_V4 = {
    "min_chars": 80,
    "max_chars": 20000,
    "max_record_bytes": 1048576,
    "strip_html": True,
    "rules": {
        "empty_or_short": {"enabled": True, "min_chars": 80, "action": "REJECT"},
        "extreme_length": {"enabled": True, "max_chars": 20000, "action": "REVIEW"},
        "repeated_characters": {"enabled": True, "max_run": 30, "action": "REJECT"},
        "replacement_characters": {"enabled": True, "max_ratio": 0.001, "action": "REJECT"},
        "html_residue": {"enabled": True, "max_tags": 3, "max_tag_ratio": 0.002, "action": "REVIEW"},
        "boilerplate_noise": {"enabled": True, "max_matches": 3, "action": "REVIEW"},
        "punctuation_density": {"enabled": True, "min_ratio": 0.0, "max_ratio": 0.30, "action": "REVIEW"},
        "digit_density": {"enabled": True, "max_ratio": 0.45, "action": "REVIEW"},
        "repeated_paragraph": {"enabled": True, "min_repeats": 4, "max_ratio": 0.5, "action": "REJECT"},
        "language_mismatch": {"enabled": True, "action": "REVIEW"},
        "unprintable_ratio": {"enabled": True, "max_ratio": 0.02, "action": "REJECT"},
        "pii_contacts": {"enabled": True, "max_contacts": 3, "action": "REVIEW"},
        "low_information": {"enabled": True,
                            "min_ratio": {"zh": 0.15, "en": 0.35, "code": 0.10,
                                          "mixed": 0.15, "unknown": 0.15},
                            "action": "REVIEW"},
    },
    # Prose density/repetition rules would mis-fire on legitimate source code.
    "code_exempt_rules": ["punctuation_density", "digit_density", "repeated_characters",
                          "repeated_paragraph", "boilerplate_noise", "html_residue"],
}

_BOILERPLATE = re.compile(
    r"首页|登录|注册|版权所有|All rights reserved|Cookie|订阅我们|分享到|点击查看更多|"
    r"相关推荐|上一篇|下一篇|加入收藏|设为首页|扫码关注|广告", re.I)
_TAG_TOTAL = re.compile(r"</?[A-Za-z][A-Za-z0-9:_-]*(?:\s[^<>]{0,2048})?\s*/?>")
_ZH_PUNCT = re.compile(r"[，。！？；：“”‘’、（）《》【】…—]")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_CONTACT_V4 = CONTACT
_SENTENCE_SPLIT = re.compile(r"[\n。！？!?.]")


@dataclass(frozen=True)
class QualityFinding:
    """One explainable rule observation; never a hidden boolean filter."""

    rule_id: str
    severity: str
    observed_value: object
    threshold: object
    action: str
    explanation: str

    def to_dict(self) -> dict:
        return {"rule_id": self.rule_id, "severity": self.severity,
                "observed_value": self.observed_value, "threshold": self.threshold,
                "action": self.action, "explanation": self.explanation}


def _finite_number(value, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"quality threshold {name} must be a finite number")
    return float(value)


def merge_quality_v4(overrides: dict | None = None) -> dict:
    """Merge and validate a v4 quality policy; every rule keeps its action."""
    import copy

    merged = copy.deepcopy(DEFAULT_QUALITY_V4)
    overrides = overrides or {}
    if not isinstance(overrides, dict):
        raise ValueError("quality policy must be a mapping")
    unknown = set(overrides) - set(DEFAULT_QUALITY_V4)
    if unknown:
        raise ValueError(f"unknown quality policy fields: {sorted(unknown)}")
    for key, value in overrides.items():
        if key != "rules":
            merged[key] = value
    rule_overrides = overrides.get("rules", {})
    if not isinstance(rule_overrides, dict):
        raise ValueError("quality.rules must be a mapping")
    for rule_id, values in rule_overrides.items():
        if rule_id not in _RULE_FIELDS:
            raise ValueError(f"unknown quality rule: {rule_id}")
        if not isinstance(values, dict):
            raise ValueError(f"quality rule {rule_id} must be a mapping")
        unknown = set(values) - _RULE_FIELDS[rule_id] - {"enabled", "action"}
        if unknown:
            raise ValueError(f"unknown fields for quality rule {rule_id}: {sorted(unknown)}")
        merged["rules"][rule_id].update(values)
    if not 0 < merged["min_chars"] <= merged["max_chars"] <= 2_000_000:
        raise ValueError("require 0 < quality.min_chars <= quality.max_chars <= 2,000,000")
    if not 1 <= merged["max_record_bytes"] <= 8 * 1048576:
        raise ValueError("quality.max_record_bytes must be in [1, 8 MiB]")
    if type(merged["strip_html"]) is not bool:
        raise ValueError("quality.strip_html must be a boolean")
    for rule_id, rule in merged["rules"].items():
        if type(rule.get("enabled")) is not bool:
            raise ValueError(f"quality rule {rule_id}.enabled must be boolean")
        if rule.get("action") not in QUALITY_ACTIONS:
            raise ValueError(f"quality rule {rule_id}.action must be one of {QUALITY_ACTIONS}")
        for field, value in rule.items():
            if field in ("enabled", "action", "min_ratio"):
                continue
            _finite_number(value, name=f"{rule_id}.{field}")
        if "min_ratio" in rule:
            if rule_id == "low_information":
                if not isinstance(rule["min_ratio"], dict):
                    raise ValueError("quality rule low_information.min_ratio must be a mapping")
                for category, value in rule["min_ratio"].items():
                    _finite_number(value, name=f"{rule_id}.min_ratio.{category}")
            else:
                _finite_number(rule["min_ratio"], name=f"{rule_id}.min_ratio")
    exempt = merged.get("code_exempt_rules")
    if not isinstance(exempt, list) or any(rule not in _RULE_FIELDS for rule in exempt):
        raise ValueError("quality.code_exempt_rules must be a list of known rules")
    merged["rules"]["empty_or_short"]["min_chars"] = merged["min_chars"]
    merged["rules"]["extreme_length"]["max_chars"] = merged["max_chars"]
    return merged


def _finding(rule_id: str, action: str, observed, threshold, explanation: str) -> QualityFinding:
    return QualityFinding(rule_id=rule_id, severity=_SEVERITY[action], observed_value=observed,
                          threshold=threshold, action=action, explanation=explanation)


def detect_language(text: str, *, category: str | None = None) -> str:
    """Chinese-first language detector using more than the presence of hanzi.

    Signals: hanzi / Latin-letter / digit distribution, Chinese punctuation,
    code markers and line structure.  The declared document category may make a
    code result more likely but never overrides clearly Chinese or English prose.
    """
    stripped = text.strip()
    if not stripped:
        return "unknown"
    cjk = len(CJK.findall(text))
    latin = len(LATIN.findall(text))
    zh_punct = len(_ZH_PUNCT.findall(text))
    non_space = max(1, sum(1 for ch in text if not ch.isspace()))
    symbols = sum(1 for ch in text if ch in "{}[]();=<>+-*/%&|!_:.,")
    code_markers = len(CODE.findall(text))
    lines = text.count("\n") + 1
    if code_markers and (symbols / non_space > 0.10 or lines >= 3) and cjk < max(20, latin * 0.5):
        return "code"
    if cjk:
        if latin > max(12, cjk * 0.4):
            return "mixed"
        if cjk >= 2 or zh_punct:
            return "zh"
    if latin >= 6:
        return "en"
    if cjk:
        return "zh"
    return "unknown"


def infer_category(language: str, text: str) -> str:
    """Best-effort category for sources that declare ``category: auto`` only."""
    if language == "code":
        return "math_logic_code"
    if language == "en":
        return "en_general"
    if language in ("zh", "mixed"):
        return "zh_general"
    # Formula-heavy unknown text is planned as math/code, not as natural prose.
    if sum(ch.isdigit() for ch in text) >= 8 or any(ch in text for ch in "=+-*/×÷"):
        return "math_logic_code"
    return "zh_general"


def evaluate_quality(text: str, *, language: str, category: str,
                     policy: dict | None = None) -> list[QualityFinding]:
    """Evaluate every enabled v4 rule and return all triggered findings."""
    policy = policy or DEFAULT_QUALITY_V4
    if language not in CATEGORIES:
        raise ValueError(f"unknown language: {language}")
    if category not in V4_CATEGORIES:
        raise ValueError(f"unknown v4 category: {category}")
    rules = policy["rules"]
    findings: list[QualityFinding] = []

    def enabled(rule_id):
        return rules[rule_id]["enabled"] and not (language == "code" and rule_id in policy["code_exempt_rules"])

    length = len(text)
    if enabled("empty_or_short") and length < rules["empty_or_short"]["min_chars"]:
        findings.append(_finding("empty_or_short", rules["empty_or_short"]["action"], length,
                                 rules["empty_or_short"]["min_chars"],
                                 "document is empty or shorter than the minimum usable length"))
    if enabled("extreme_length") and length > rules["extreme_length"]["max_chars"]:
        findings.append(_finding("extreme_length", rules["extreme_length"]["action"], length,
                                 rules["extreme_length"]["max_chars"],
                                 "document exceeds the maximum length; split or review it"))
    if not text:
        return findings

    if enabled("repeated_characters"):
        longest, run = 1, 1
        for previous, current in zip(text, text[1:]):
            if current == previous and not current.isspace():
                run += 1
                longest = max(longest, run)
            else:
                run = 1
        if longest >= rules["repeated_characters"]["max_run"]:
            findings.append(_finding("repeated_characters", rules["repeated_characters"]["action"],
                                     longest, rules["repeated_characters"]["max_run"],
                                     "a single character repeats without break (garbage/spam signal)"))

    if enabled("replacement_characters"):
        ratio = text.count("\ufffd") / length
        threshold = rules["replacement_characters"]["max_ratio"]
        if ratio > threshold:
            findings.append(_finding("replacement_characters", rules["replacement_characters"]["action"],
                                     round(ratio, 6), threshold,
                                     "U+FFFD replacement characters indicate a broken decode"))

    if enabled("html_residue"):
        tags = _TAG_TOTAL.findall(text)
        tag_ratio = sum(len(tag) for tag in tags) / length
        threshold = rules["html_residue"]
        if len(tags) > threshold["max_tags"] or tag_ratio > threshold["max_tag_ratio"]:
            findings.append(_finding("html_residue", threshold["action"],
                                     {"tags": len(tags), "tag_ratio": round(tag_ratio, 6)},
                                     {"max_tags": threshold["max_tags"], "max_tag_ratio": threshold["max_tag_ratio"]},
                                     "HTML markup remains after normalization"))

    if enabled("boilerplate_noise"):
        matches = _BOILERPLATE.findall(text)
        if len(matches) >= rules["boilerplate_noise"]["max_matches"]:
            findings.append(_finding("boilerplate_noise", rules["boilerplate_noise"]["action"],
                                     len(matches), rules["boilerplate_noise"]["max_matches"],
                                     "navigation/advertising template phrases dominate the document"))

    non_space = max(1, sum(1 for ch in text if not ch.isspace()))
    if enabled("punctuation_density"):
        ratio = len(_PUNCT.findall(text)) / non_space
        rule = rules["punctuation_density"]
        if ratio < rule["min_ratio"] or ratio > rule["max_ratio"]:
            findings.append(_finding("punctuation_density", rule["action"], round(ratio, 6),
                                     {"min": rule["min_ratio"], "max": rule["max_ratio"]},
                                     "punctuation density is outside the prose range"))

    if enabled("digit_density"):
        ratio = sum(ch.isdigit() for ch in text) / non_space
        threshold = rules["digit_density"]["max_ratio"]
        if ratio > threshold:
            findings.append(_finding("digit_density", rules["digit_density"]["action"],
                                     round(ratio, 6), threshold,
                                     "digit-heavy text is kept for review, never rejected for being numeric"))

    if enabled("repeated_paragraph"):
        rule = rules["repeated_paragraph"]
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= 20]
        segments = paragraphs if len(paragraphs) >= rule["min_repeats"] else [
            s.strip() for s in _SENTENCE_SPLIT.split(text) if len(s.strip()) >= 20]
        if len(segments) >= rule["min_repeats"]:
            unique = len(set(segments))
            ratio = (len(segments) - unique) / len(segments)
            if ratio >= rule["max_ratio"]:
                findings.append(_finding("repeated_paragraph", rule["action"],
                                         {"segments": len(segments), "duplicate_ratio": round(ratio, 6)},
                                         {"min_repeats": rule["min_repeats"], "max_ratio": rule["max_ratio"]},
                                         "the document repeats the same paragraph/sentence many times"))

    if enabled("unprintable_ratio"):
        bad = sum(1 for ch in text if unicodedata.category(ch) in ("Cs", "Cn")
                  or (unicodedata.category(ch) == "Cc" and ch not in "\n\t"))
        ratio = bad / length
        threshold = rules["unprintable_ratio"]["max_ratio"]
        if ratio > threshold:
            findings.append(_finding("unprintable_ratio", rules["unprintable_ratio"]["action"],
                                     round(ratio, 6), threshold,
                                     "too many unprintable/control code points"))

    if enabled("pii_contacts"):
        contacts = _CONTACT_V4.findall(text)
        threshold = rules["pii_contacts"]["max_contacts"]
        if len(contacts) >= threshold:
            findings.append(_finding("pii_contacts", rules["pii_contacts"]["action"], len(contacts),
                                     threshold, "many contact details/URLs may indicate personal data or spam"))

    if enabled("low_information"):
        ratio = sum(ch.isalnum() for ch in text) / max(non_space, 1)
        threshold = rules["low_information"]["min_ratio"].get(language, 0.15)
        if ratio < threshold:
            findings.append(_finding("low_information", rules["low_information"]["action"],
                                     round(ratio, 6), threshold,
                                     "informative character ratio is low; review instead of silent deletion"))

    if enabled("language_mismatch"):
        expected = _EXPECTED_LANGUAGE.get(category)
        if expected and language not in expected:
            findings.append(_finding("language_mismatch", rules["language_mismatch"]["action"],
                                     language, list(expected),
                                     "detected language disagrees with the declared category"))

    return findings


def aggregate_action(findings: list[QualityFinding]) -> str:
    action = "KEEP"
    for finding in findings:
        if _ACTION_RANK[finding.action] > _ACTION_RANK[action]:
            action = finding.action
    return action


def quality_action_rank(action: str) -> int:
    if action not in QUALITY_ACTIONS:
        raise ValueError(f"unknown quality action: {action}")
    return _ACTION_RANK[action]
