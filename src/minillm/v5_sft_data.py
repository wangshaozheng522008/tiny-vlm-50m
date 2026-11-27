"""Admission and freezing of the independently reviewed Chinese SFT corpus."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import chain
import json
from pathlib import Path
import re
import unicodedata

from .assessment_io import read_json, records, report_directory
from .corpus_utils import file_info, identity, verify_file
from .data import SFTDataset
from .tokenizer import ASSISTANT_ID, END_OF_TURN_ID
from .training_manifest import write_json
from .v4_language_pilot import BENCHMARK

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/pilot-v5-language-sft-5k-t1200.json"
TRAIN = dict(zip(("日常生活对话", "基础世界知识", "常识及概念解释", "简单指令理解", "不确定问题回答"),
                 (1750, 1500, 1000, 500, 250)))
HELDOUT = dict(zip(TRAIN, (175, 150, 100, 50, 25)))
QUOTAS = {"train": TRAIN, "val": HELDOUT, "test": HELDOUT}
CHECKS = ("simplified_chinese", "factuality", "relevance", "naturalness", "context",
          "semantic_dedup", "historical_contamination", "answer_templates")
FAMILIES = ("fact_family", "question_family", "dialogue_family")
HAN = re.compile(r"[\u3400-\u9fff]")


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def validate_messages(messages):
    if not isinstance(messages, list) or len(messages) not in (2, 4, 6):
        raise ValueError("expected 1–3 user/assistant pairs")
    for i, message in enumerate(messages):
        expected = ("user", "assistant")[i % 2]
        if not isinstance(message, dict) or message.keys() != {"role", "content"} or message["role"] != expected:
            raise ValueError("invalid message schema/role order")
        content = message["content"]
        if (not isinstance(content, str) or not (1 if i % 2 else 2) <= len(content) <= (100 if i % 2 else 160)
                or (not HAN.search(content) and not (i % 2 and len(content) <= 12))
                or content != content.strip()
                or re.search(r"<[^>]+>|[\r\n`#]|(.)\1{4,}", content)
                or any(unicodedata.category(c).startswith("C") for c in content)
                or (i % 2 and len(re.findall(r"[。！？!?]+", content.rstrip("。！？!?"))) > 2)):
            raise ValueError("invalid Chinese content, answer length, sentence count or markup")


class StrictChatDataset(SFTDataset):
    """Use the project template, with loss beginning after the inference prefix."""

    def __init__(self, path, tokenizer, max_seq_len, *, rows=None):
        raw = list(records(path)) if rows is None else rows
        chats = [row["messages"] for row in raw]
        for chat in chats:
            validate_messages(chat)
        super().__init__(path, tokenizer, max_seq_len, records=[{"messages": chat} for chat in chats])
        if self.stats["truncated"] or self.stats["dropped"]:
            raise ValueError(f"{path}: strict SFT forbids truncation and dropped records")
        for item, chat in zip(self.items, chats):
            ids, labels = item["input_ids"], item["labels"]
            for turn in range(1, len(chat), 2):
                prefix = tokenizer.apply_chat_template(chat[:turn], add_generation_prompt=True)
                if not prefix or ids[:len(prefix)].tolist() != prefix:
                    raise ValueError(f"{path}: inference prefix differs from training tokenization")
                role = max(i for i, token in enumerate(prefix) if token == ASSISTANT_ID)
                end = ids.tolist().index(END_OF_TURN_ID, len(prefix))
                if end <= len(prefix):
                    raise ValueError(f"{path}: empty tokenized assistant answer")
                labels[role:len(prefix)] = tokenizer.ignore_index
        self.stats["supervised_tokens"] = sum(int((x["labels"][1:] != tokenizer.ignore_index).sum()) for x in self.items)


def historical_tasks():
    from .evaluation import load_tasks
    tasks = load_tasks(ROOT / BENCHMARK, scored=True)
    world = [row for row in tasks if row["category"] in ("geohistory", "science") and row["scoring"] == "normalized_match"]
    if (len(tasks), len(world)) != (300, 97):
        raise ValueError("historical benchmark population changed")
    return tasks, world


def check_overlap(rows, historical):
    """Exhaustive exact/Jaccard checks of normalized question text via an inverted index."""
    grams_by_key, postings, normalized_keys, flags = {}, defaultdict(set), {}, []
    questions = [(f"historical:{row['id']}", row["prompt"]) for row in historical]
    questions.extend((row["id"], m["content"]) for row in rows for m in row["messages"] if m["role"] == "user")
    for i, (key, text) in enumerate(questions):
        clean = normalized(text)
        grams = {clean[j:j+3] for j in range(len(clean)-2)}
        matches = set(normalized_keys.get(clean, ()))
        for j in set(chain.from_iterable(postings[g] for g in grams)):
            old = grams_by_key[j]
            if len(grams & old) >= 0.7 * len(grams | old):
                matches.add(j)
        for j in sorted(matches):
            other = questions[j][0]
            if not key.startswith("historical:"):
                flags.append([other, key])
        for gram in grams:
            postings[gram].add(i)
        grams_by_key[i] = grams
        normalized_keys.setdefault(clean, []).append(i)
    return flags


def metrics(dataset):
    inputs = [len(item["input_ids"]) for item in dataset.items]
    targets = [int((item["labels"][1:] != dataset.tokenizer.ignore_index).sum()) for item in dataset.items]
    lengths = [len(m["content"]) for row in dataset.raw for m in row["messages"] if m["role"] == "assistant"]

    def distribution(values):
        ordered = sorted(values)
        return {"min": ordered[0], "p50": ordered[len(ordered)//2],
                "p95": ordered[int(0.95 * (len(ordered)-1))], "max": ordered[-1]}

    return {"examples": len(inputs), "input_tokens": sum(inputs), "assistant_tokens": sum(targets),
            "input_lengths": distribution(inputs), "assistant_token_lengths": distribution(targets),
            "answer_characters": distribution(lengths),
            "multiturn": sum(len(row["messages"]) > 2 for row in dataset.raw)}


def freeze(source, reviews, output, tokenizer, max_seq_len, *, inspected):
    candidate_rows = list(records(source))
    by_id = {row["id"]: row for row in candidate_rows}
    if not 8000 <= len(by_id) <= 10000 or len(by_id) != len(candidate_rows):
        raise ValueError("expected 8000–10000 uniquely identified generated candidates")
    review_rows = list(records(reviews))
    if len({row["id"] for row in review_rows}) != len(review_rows):
        raise ValueError("duplicate independent review")
    selected = defaultdict(list)
    owners = {}
    for review in review_rows:
        row = by_id[review["id"]]
        if review["status"] not in ("PASS", "FAIL", "PENDING"):
            raise ValueError("invalid review status")
        if review["status"] != "PASS":
            continue
        if (review.get("message_sha256") != identity(row["messages"])
                or row.get("message_sha256") != review["message_sha256"]
                or not str(review.get("reviewer") or "").strip() or not str(review.get("evidence") or "").strip()
                or review.get("origin") not in ("human", "independent_model")
                or review["reviewer"] == row["generation"]["model"]
                or any(review["checks"].get(check) != "PASS" for check in CHECKS)):
            raise ValueError(f"{row['id']}: independent review evidence incomplete")
        validate_messages(row["messages"])
        split = row["split"]
        family = {key: normalized(review[key]) for key in FAMILIES}
        for key, value in family.items():
            if not value or owners.setdefault((key, value), split) != split:
                raise ValueError(f"empty or cross-split family: {key}={value}")
        selected[split].append({**row, **family})
    inspected["candidate_reviews"] = dict(Counter(row["status"] for row in review_rows))
    inspected["candidate_reviews"]["PENDING"] = inspected["candidate_reviews"].get("PENDING", 0) + len(by_id.keys() - {r["id"] for r in review_rows})
    if set(selected) != QUOTAS.keys():
        raise ValueError("all three independently reviewed splits are required")
    for split, quotas in QUOTAS.items():
        if Counter(row["topic_category"] for row in selected[split]) != quotas:
            raise ValueError(f"{split}: category quotas differ")
        fraction = sum(len(row["messages"]) > 2 for row in selected[split]) / len(selected[split])
        if not 0.08 <= fraction <= 0.12:
            raise ValueError(f"{split}: expected approximately 10% multi-turn")
    all_rows = [row for split in QUOTAS for row in selected[split]]
    historical, _ = historical_tasks()
    inspected["lexical_flags"] = check_overlap(all_rows, historical)
    if inspected["lexical_flags"]:
        raise ValueError("duplicate/near-duplicate questions detected; inspect audit and revise selection")
    answers = [normalized(m["content"]) for row in all_rows for m in row["messages"] if m["role"] == "assistant"]
    common = Counter(a[:8] for a in answers if len(a) >= 8).most_common(20)
    inspected["answer_prefix_concentration"] = common
    if common and common[0][1] > len(answers) * 0.02:
        raise ValueError("answer opening exceeds 2% concentration")
    with report_directory(output) as directory:
        stats = {}
        for split in QUOTAS:
            selected[split].sort(key=lambda row: row["id"])
            path = directory / f"{split}.jsonl"
            write_rows(path, selected[split])
            stats[split] = {**metrics(StrictChatDataset(path, tokenizer, max_seq_len)), "categories": QUOTAS[split]}
        write_rows(directory / "reviews.jsonl", review_rows)
        document = {"files": {p.name: file_info(p) for p in directory.iterdir()},
                    "candidates": file_info(source), "historical_benchmark": file_info(ROOT / BENCHMARK),
                    "tokenizer": tokenizer.fingerprint, "max_seq_len": max_seq_len,
                    "statistics": stats, "audit": inspected}
        write_json(directory / "manifest.json", {"schema_version": 1, "status": "FROZEN",
                                                "content_id": identity(document), "identity": document})


def frozen_manifest(path, tokenizer, max_seq_len):
    manifest = read_json(path / "manifest.json")
    data = manifest["identity"]
    if (manifest["status"] != "FROZEN" or manifest["schema_version"] != 1
            or manifest["content_id"] != identity(data)
            or (data["tokenizer"], data["max_seq_len"]) != (tokenizer.fingerprint, max_seq_len)):
        raise ValueError("incompatible frozen SFT manifest")
    verify_file(ROOT / BENCHMARK, data["historical_benchmark"])
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "reviews.jsonl"):
        verify_file(path / name, data["files"][name])
    return manifest
