"""Benchmark-contamination audit for v4 training text.

The audit uses the repository's frozen PR #6 / PR #7 evaluation assets.  Assets
must carry a matching ``benchmark-version.json`` declaration; an undeclared file
is reported as ``CUSTOM_UNFROZEN`` and cannot produce a READY contamination
status.

Two different situations are deliberately kept apart:

* **Contamination**: a training document reproduces a benchmark question,
  option set, answer format or a long span of a benchmark item.  These are
  ``REVIEW`` by default; nothing is silently declared clean.
* **Ordinary world knowledge**: a document states a fact that a benchmark also
  asks about (for example "北京是中国的首都").  Short, declarative fact overlap
  is reported as ``ALLOWED_KNOWLEDGE`` and is not deleted.

The distinction is implemented with contiguous match confirmation, a minimum
question length, question-format detection and option/answer co-occurrence.
It is a bounded lexical audit, not a semantic guarantee; the report states the
limitations explicitly.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .assessment_io import records
from .corpus_utils import identity
from .evaluation import task_asset_provenance

CONTAMINATION_STATUSES = ("NOT_ASSESSED", "CHECKED_NO_HITS", "REVIEW_REQUIRED", "REVIEWED", "FAIL")
HIGH_RISK_TYPES = ("QA_ITEM_REPLICATION", "PROMPT_VERBATIM", "LONG_SPAN_OVERLAP", "QA_FORMAT_REPLICATION")
LOW_RISK_TYPES = ("ANSWER_ONLY", "SHORT_KNOWLEDGE_OVERLAP")
_QUESTION = re.compile(r"[：:？?]\s*$|反义词|近义词|下一个|下列|以下|哪一?个|选择|填空|只回答|答案|选项|计算|等于|多少", re.I)
_OPTION_LINE = re.compile(r"(?:^|\s)([A-H])[.、．:：]\s*", re.M)
_WS = re.compile(r"\s+")


def comparison_text(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip().casefold()


def _ngrams(text: str, n: int) -> set[str]:
    if len(text) < n:
        return set()
    return {text[index:index + n] for index in range(len(text) - n + 1)}


@dataclass(frozen=True)
class BenchmarkVariant:
    kind: str
    text: str
    question_like: bool


@dataclass
class BenchmarkItem:
    benchmark_id: str
    asset: str
    category: str
    prompt: str
    reference: str | None
    options: dict[str, str]
    variants: list[BenchmarkVariant] = field(default_factory=list)
    document_family_id: str | None = None


def _looks_like_question(prompt: str) -> bool:
    return bool(_QUESTION.search(prompt))


def load_benchmark_items(assets: list[dict]) -> tuple[list[BenchmarkItem], list[dict], list[str]]:
    """Load and verify frozen assets; return items, provenance, errors."""
    items: list[BenchmarkItem] = []
    provenance: list[dict] = []
    errors: list[str] = []
    seen: set[str] = set()
    for spec in assets:
        if not isinstance(spec, dict) or set(spec) - {"path", "kind", "enabled"}:
            raise ValueError("contamination asset entries need path/kind/enabled")
        if spec.get("enabled", True) is False:
            continue
        path = Path(spec["path"])
        if not path.is_file():
            errors.append(f"missing contamination asset: {path}")
            continue
        try:
            rows = list(records(path))
            declared = task_asset_provenance(path, len(rows))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"invalid contamination asset {path}: {exc}")
            continue
        provenance.append({"path": str(path), "kind": spec.get("kind", "benchmark"),
                           "records": len(rows), "sha256": declared["sha256"],
                           "status": declared["status"],
                           "version": declared.get("declaration", {}).get("version") if isinstance(declared.get("declaration"), dict) else None,
                           "declaration_sha256": declared.get("declaration_sha256")})
        if declared["status"] != "FROZEN":
            errors.append(f"contamination asset is not frozen: {path}")
        version = None
        if isinstance(declared.get("declaration"), dict):
            version = declared["declaration"].get("version")
        for row in rows:
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                errors.append(f"{path}: record without prompt")
                continue
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id.strip():
                errors.append(f"{path}: record without id")
                continue
            benchmark_id = f"{version or path.stem}/{row_id}"
            if benchmark_id in seen:
                errors.append(f"duplicate benchmark id across assets: {benchmark_id}")
                continue
            seen.add(benchmark_id)
            candidates = row.get("candidates") if isinstance(row.get("candidates"), dict) else {}
            options = row.get("options") if isinstance(row.get("options"), dict) else candidates
            options = {str(k): str(v) for k, v in options.items() if isinstance(v, str) and v.strip()}
            reference = row.get("reference")
            reference = reference if isinstance(reference, str) else None
            item = BenchmarkItem(benchmark_id=benchmark_id, asset=str(path),
                                 category=str(row.get("category", "unknown")),
                                 prompt=prompt, reference=reference, options=options,
                                 document_family_id=row.get("document_family_id") or row.get("family_id"))
            question_like = _looks_like_question(prompt) or bool(options)
            item.variants.append(BenchmarkVariant("PROMPT", prompt, question_like))
            if options:
                option_text = " ".join(f"{key}.{value}" for key, value in options.items())
                item.variants.append(BenchmarkVariant("OPTIONS", prompt + " " + option_text, True))
                if reference:
                    answer_text = options.get(reference, reference)
                    item.variants.append(BenchmarkVariant(
                        "QA_ITEM", f"{prompt} {option_text} 答案：{reference}", True))
                    if answer_text and answer_text != reference:
                        item.variants.append(BenchmarkVariant("ANSWER", answer_text, False))
            elif reference:
                item.variants.append(BenchmarkVariant("PROMPT_ANSWER", prompt + reference, question_like))
                item.variants.append(BenchmarkVariant("ANSWER", reference, False))
            items.append(item)
    return items, provenance, errors


def _variant_grams(items: list[BenchmarkItem]) -> tuple[dict, list[BenchmarkVariant], list[str]]:
    index: dict[tuple[int, str], set[int]] = {}
    flat: list[BenchmarkVariant] = []
    owners: list[str] = []
    for item in items:
        for variant in item.variants:
            text = comparison_text(variant.text)
            if len(text) < 8:
                continue
            n = 16 if len(text) >= 16 else 8
            grams = _ngrams(text, n)
            if not grams:
                continue
            position = len(flat)
            flat.append(variant)
            owners.append(item.benchmark_id)
            for gram in grams:
                index.setdefault((n, gram), set()).add(position)
    return index, flat, owners


def _doc_ngrams(text: str) -> dict[int, set[str]]:
    normalized = comparison_text(text)
    return {8: _ngrams(normalized, 8), 16: _ngrams(normalized, 16)}


def _first_snippet(text: str, probe: str, length: int = 160) -> str:
    position = text.find(probe[:24])
    if position < 0:
        return text[:length]
    start = max(0, position - 30)
    return text[start:start + length]


def scan_documents(documents, items: list[BenchmarkItem], *,
                   min_prompt_chars: int = 12, long_span_containment: float = 0.8,
                   max_hits: int = 5000) -> dict:
    """Scan an iterable of {"document_id", "text", "language", ...} records."""
    if not 1 <= min_prompt_chars <= 200:
        raise ValueError("min_prompt_chars must be in [1, 200]")
    if not 0.5 <= long_span_containment <= 1.0:
        raise ValueError("long_span_containment must be in [0.5, 1]")
    index, flat, owners = _variant_grams(items)
    items_by_id = {item.benchmark_id: item for item in items}
    hits: list[dict] = []
    stats = {"documents_scanned": 0, "documents_with_hits": 0, "high_risk_hits": 0,
             "allowed_knowledge_hits": 0, "truncated": False,
             "family_association": "NOT_ASSESSED"}
    for document in documents:
        if not isinstance(document, dict) or not isinstance(document.get("text"), str):
            continue
        document_id = document.get("document_id") or document.get("id")
        if not isinstance(document_id, str) or not document_id:
            continue
        stats["documents_scanned"] += 1
        text = document["text"]
        normalized = comparison_text(text)
        grams = _doc_ngrams(text)
        counts: dict[int, int] = {}
        for n, values in grams.items():
            for gram in values:
                for variant_index in index.get((n, gram), ()):
                    counts[variant_index] = counts.get(variant_index, 0) + 1
        found: dict[str, dict] = {}
        for variant_index, matched in counts.items():
            variant, benchmark_id = flat[variant_index], owners[variant_index]
            item = items_by_id[benchmark_id]
            variant_text = comparison_text(variant.text)
            n = 16 if len(variant_text) >= 16 else 8
            total = max(1, len(variant_text) - n + 1)
            containment = matched / total
            if containment < 0.6:
                continue
            verbatim = variant_text in normalized
            kind = variant.kind
            overlap_type = None
            severity = "REVIEW"
            similarity = round(min(1.0, matched / total), 6)
            if verbatim and kind == "QA_ITEM":
                overlap_type, similarity = "QA_ITEM_REPLICATION", 1.0
            elif verbatim and kind == "OPTIONS" and variant.question_like:
                overlap_type = "QA_FORMAT_REPLICATION"
            elif verbatim and kind == "PROMPT":
                if len(variant_text) >= min_prompt_chars or (variant.question_like and len(variant_text) >= 8):
                    overlap_type = "PROMPT_VERBATIM"
                else:
                    overlap_type, severity = "SHORT_KNOWLEDGE_OVERLAP", "KEEP"
            elif verbatim and kind == "PROMPT_ANSWER":
                prompt_text = comparison_text(item.prompt)
                if len(prompt_text) >= min_prompt_chars or (variant.question_like and len(prompt_text) >= 8):
                    overlap_type = "PROMPT_VERBATIM"
                else:
                    overlap_type, severity = "SHORT_KNOWLEDGE_OVERLAP", "KEEP"
            elif len(variant_text) >= 32 and containment >= long_span_containment:
                overlap_type = "LONG_SPAN_OVERLAP"
            elif verbatim and kind == "ANSWER":
                overlap_type, severity = "ANSWER_ONLY", "KEEP"
            if overlap_type is None:
                continue
            # Option-set replication upgrades a prompt hit even when whitespace
            # or punctuation keeps the full item from being one contiguous match.
            if overlap_type == "PROMPT_VERBATIM" and item.options:
                present = sum(
                    1 for key, value in item.options.items()
                    if re.search(re.escape(key) + r"\s*[.、．:：]\s*" + re.escape(value), normalized))
                if present >= 2:
                    overlap_type = ("QA_ITEM_REPLICATION" if present == len(item.options)
                                    else "QA_FORMAT_REPLICATION")
                    similarity = max(similarity, round(present / len(item.options), 6))
            # Answer text alone is ordinary knowledge unless the question or
            # option format is also reproduced in this document.
            if overlap_type == "ANSWER_ONLY" and any(
                    name in ("PROMPT_VERBATIM", "QA_ITEM_REPLICATION", "QA_FORMAT_REPLICATION", "LONG_SPAN_OVERLAP")
                    for name in found):
                continue
            hit = {"train_document_id": document_id, "benchmark_id": benchmark_id,
                   "overlap_type": overlap_type, "similarity": similarity,
                   "disposition": "REVIEW" if severity == "REVIEW" else "KEEP",
                   "severity": severity, "asset": item.asset, "category": item.category,
                   "evidence": {"variant": kind, "matched_ngrams": matched,
                                "variant_ngrams": total,
                                "contiguous": bool(verbatim),
                                "snippet": _first_snippet(text, variant_text)}}
            previous = found.get(benchmark_id)
            if previous is None or _type_rank(hit) > _type_rank(previous):
                found[benchmark_id] = hit
        document_family = document.get("document_family_id")
        if document_family:
            for item in items:
                if item.document_family_id and item.document_family_id == document_family:
                    found[item.benchmark_id] = {
                        "train_document_id": document_id, "benchmark_id": item.benchmark_id,
                        "overlap_type": "DOCUMENT_FAMILY_ASSOCIATION", "similarity": 1.0,
                        "disposition": "REVIEW", "severity": "REVIEW", "asset": item.asset,
                        "category": item.category,
                        "evidence": {"variant": "FAMILY", "matched_ngrams": 0, "variant_ngrams": 0,
                                     "contiguous": False, "snippet": f"family={document_family}"}}
        for benchmark_id, hit in sorted(found.items()):
            if hit["disposition"] == "REVIEW":
                stats["high_risk_hits"] += 1
            else:
                stats["allowed_knowledge_hits"] += 1
            if len(hits) >= max_hits:
                stats["truncated"] = True
                continue
            hits.append(hit)
        if found:
            stats["documents_with_hits"] += 1
    return {"hits": hits, "stats": stats,
            "benchmark_items": len(items), "variants_indexed": len(flat),
            "min_prompt_chars": min_prompt_chars,
            "long_span_containment": long_span_containment}


def _type_rank(hit: dict) -> int:
    order = {"QA_ITEM_REPLICATION": 6, "PROMPT_VERBATIM": 5, "QA_FORMAT_REPLICATION": 4,
             "LONG_SPAN_OVERLAP": 3, "ANSWER_ONLY": 2, "SHORT_KNOWLEDGE_OVERLAP": 1}
    return order.get(hit["overlap_type"], 0)


def apply_decisions(hits: list[dict], decisions_path: Path | None) -> dict:
    """Validate reviewer dispositions for high-risk hits.

    Decisions are JSONL rows: {"train_document_id", "benchmark_id", "action":
    "accept_risk"|"exclude", "reviewer", "reason"}.  Every high-risk hit needs
    one decision; ``exclude`` means the document must be removed before freeze.
    """
    high = [hit for hit in hits if hit["disposition"] == "REVIEW"]
    result = {"decisions": 0, "accepted_risk": 0, "exclude": 0,
              "unreviewed_high_risk": len(high), "excluded_documents_present": [],
              "status": "REVIEW_REQUIRED" if high else "CHECKED_NO_HITS"}
    if decisions_path is None:
        return result
    path = Path(decisions_path)
    if not path.is_file():
        raise ValueError(f"missing contamination decisions: {path}")
    keyed = {(hit["train_document_id"], hit["benchmark_id"]) for hit in high}
    decisions: dict[tuple[str, str], dict] = {}
    for row in records(path):
        required = {"train_document_id", "benchmark_id", "action", "reviewer", "reason"}
        if set(row) != required:
            raise ValueError("contamination decision fields must be exactly " + ", ".join(sorted(required)))
        key = (row["train_document_id"], row["benchmark_id"])
        if key in decisions or key not in keyed:
            raise ValueError("duplicate or unknown contamination decision")
        if row["action"] not in ("accept_risk", "exclude"):
            raise ValueError("contamination decision action must be accept_risk or exclude")
        if any(not isinstance(row[name], str) or not row[name].strip() or len(row[name]) > 4000
               for name in ("reviewer", "reason")):
            raise ValueError("contamination decision needs reviewer and reason")
        decisions[key] = row
    result["decisions"] = len(decisions)
    for hit in high:
        decision = decisions.get((hit["train_document_id"], hit["benchmark_id"]))
        if decision is None:
            continue
        result["unreviewed_high_risk"] -= 1
        if decision["action"] == "accept_risk":
            result["accepted_risk"] += 1
        else:
            result["exclude"] += 1
            result["excluded_documents_present"].append(hit["train_document_id"])
    if result["excluded_documents_present"]:
        result["status"] = "FAIL"
    elif result["unreviewed_high_risk"] == 0:
        result["status"] = "REVIEWED"
    return result


def audit(assets: list[dict], documents, *, decisions: Path | None = None,
          min_prompt_chars: int = 12, long_span_containment: float = 0.8,
          max_hits: int = 5000) -> dict:
    """Full audit report; ``documents`` is consumed exactly once."""
    loaded, provenance, errors = load_benchmark_items(assets)
    if errors and not loaded:
        return {"schema_version": 1, "status": "NOT_ASSESSED", "assets": provenance,
                "errors": errors, "hits": [], "stats": {"documents_scanned": 0},
                "limitations": "No usable frozen evaluation asset; contamination was not assessed."}
    scan = scan_documents(documents, loaded, min_prompt_chars=min_prompt_chars,
                          long_span_containment=long_span_containment, max_hits=max_hits)
    decisions_report = apply_decisions(scan["hits"], decisions)
    high = [hit for hit in scan["hits"] if hit["disposition"] == "REVIEW"]
    status = decisions_report["status"]
    if errors:
        status = "FAIL"
    elif not high and not scan["stats"]["truncated"]:
        status = "CHECKED_NO_HITS"
    elif scan["stats"]["truncated"]:
        status = "FAIL"
    report = {"schema_version": 1, "status": status, "assets": provenance,
              "errors": errors, "benchmark_items": scan["benchmark_items"],
              "variants_indexed": scan["variants_indexed"],
              "hits": scan["hits"], "stats": scan["stats"],
              "decisions": decisions_report,
              "rules": {"min_prompt_chars": min_prompt_chars,
                        "long_span_containment": long_span_containment,
                        "ngram": "8 for short variants, 16 for long variants; contiguous match confirmation",
                        "high_risk_types": list(HIGH_RISK_TYPES),
                        "low_risk_types": list(LOW_RISK_TYPES)},
              "limitations": ("Lexical, bounded audit over the declared frozen assets. Short declarative "
                              "world-knowledge overlap is kept, but this is not semantic proof and does not "
                              "exclude unknown or future benchmarks.")}
    report["content_id"] = identity({"assets": provenance, "hits": report["hits"],
                                     "rules": report["rules"], "status": status})
    return report
