"""Bounded reference-set contamination scan; no claim of exhaustive decontamination."""
from collections import defaultdict
from pathlib import Path

from .assessment_io import records
from .corpus_quality import bands, comparison_hash, shingles, simhash, similar
from .corpus_utils import file_info


def contamination_audit(tasks, *, dataset=None, sft_data=()):
    variants, exact, buckets = [], defaultdict(set), defaultdict(set)
    for task in tasks:
        reference=task.get("reference") or ""
        for text in {task["prompt"], task["prompt"] + "\n" + reference, task["prompt"] + reference}:
            features = shingles(text, "mixed")
            fingerprint = simhash(features)
            index = len(variants)
            variants.append((task["id"], features, fingerprint))
            exact[comparison_hash(text)].add(task["id"])
            for key in bands(fingerprint, 4):
                buckets[key].add(index)
    sources = []
    if dataset:
        sources.extend((f"pretrain/{split}", Path(dataset)/f"{split}.text.jsonl") for split in ("train", "val", "test"))
    sources.extend(("sft", Path(path)) for path in sft_data)
    reports = []
    for kind, path in sources:
        exact_ids, near_ids, scanned, unsupported, near_skipped = set(), set(), 0, 0, 0
        before = file_info(path)
        for record in records(path):
            texts = []
            if isinstance(record.get("text"), str):
                texts.append(record["text"])
            if isinstance(record.get("messages"), list):
                messages = record["messages"]
                texts.extend(m["content"] for m in messages if isinstance(m, dict) and isinstance(m.get("content"), str))
                texts.append("\n".join(m["content"] for m in messages if isinstance(m, dict) and isinstance(m.get("content"), str)))
            if not texts:
                unsupported += 1
            scanned += 1
            for text in texts:
                exact_ids.update(exact.get(comparison_hash(text), ()))
                # Bound full-document feature state. Large records still get exact checks.
                if len(text) > 20000:
                    near_skipped += 1
                    continue
                features = shingles(text, "mixed"); fingerprint = simhash(features)
                candidates = set()
                for key in bands(fingerprint, 4):
                    candidates.update(buckets.get(key, ()))
                for index in candidates:
                    task_id, other, fp = variants[index]
                    if (fingerprint ^ fp).bit_count() <= 4 and similar(features, other, .85):
                        near_ids.add(task_id)
        if file_info(path) != before:
            raise ValueError(f"contamination input changed: {path}")
        reports.append({"kind": kind, "file": str(path), **before, "records_scanned": scanned,
                        "exact_task_ids": sorted(exact_ids), "near_task_ids": sorted(near_ids - exact_ids),
                        "unsupported_records": unsupported, "near_skipped_long_texts": near_skipped})
    return {"status": "CHECKED_KNOWN_SOURCES" if sources else "NOT_RUN",
            "sources": reports, "exact_rule": "category-neutral whitespace-folded full prompt / prompt+reference / individual SFT messages and joined conversation",
            "near_rule": "full character 4-grams, SimHash64 Hamming<=4 + Jaccard>=.85, texts<=20000 chars",
            "limitations": "no substring, semantic or unknown training-corpus exclusion; short-answer coincidences are not proof of memorization; no exhaustive contamination guarantee"}
