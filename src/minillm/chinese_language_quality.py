"""Read-only heuristics for Chinese Train quality. This never filters records."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from .corpus import read_records
from .corpus_utils import file_info, identity, verify_file

# High-precision character cues only. Shared and ambiguous forms are omitted.
_SIMPLIFIED = set("国书语学会这们时说话对为过还听让给从后场门车体实变应当图区务样历现条")
_TRADITIONAL = set("國書語學會這們時說話對為過還聽讓給從後場門車體實變應當圖區務樣歷現條")
_CLASSICAL = re.compile(r"之乎|乎哉|曰[：:]?|矣[。！？；，]|也[。；]|吾|其[一二三]曰|者也|於是乎")
_MODERN = re.compile(r"我们|我們|他们|他們|可以|需要|现在|現在|因为|因為|这个|這個|如果|问题|問題")
_WIKI = re.compile(r"\{\{|\}\}|\[\[|\]\]|(?m:^\s*={2,}[^\n]+={2,}\s*$)|</?(?:ref|gallery|nowiki)\b", re.I)
_STYLES = {
    "dialogue": re.compile(r"[说說问問答][：:]\s*[“「]|[“「][^”」\n]{4,}[”」]"),
    "narration": re.compile(r"忽然|接著|接着|后来|後來|那天|次日|走进|走進|走到|转身|轉身"),
    "exposition": re.compile(r"例如|因此|原因|通常|步骤|步驟|首先|也就是"),
    "encyclopedic": re.compile(r"位于|位於|成立于|成立於|又称|又稱|起源于|起源於|分布于|分布於"),
}
_LIMITATIONS = [
    "字形仅用少量简繁特征字，混合标记不代表全文混写，且无法区分区域和版本。",
    "古文和现代中文从小型词表命中推断，现代文中的引用和旧刊白话可能误分；未知不计为现代。",
    "句段按标点或换行切分，标题与韵文可能计为句段。文体是非互斥线索，不能当作人工标签。",
    "重复段落按去空白后的段落精确比较，仅统计至少 16 字的段落；噪声与异常规则会遗漏其他形式。",
]


def analyze_chinese_train(path, split_manifest, *, corpus_version):
    from .tokenizer_v4 import guard_train_only

    split = guard_train_only(path, split_manifest)
    train_info = split["files"]["train.text.jsonl"]
    groups = {}
    for row in read_records(path):
        if row["category"] not in ("zh_general", "zh_encyclopedic", "zh_education"):
            continue
        text = row["text"]
        counts, lengths, seen_paragraphs = groups.setdefault(row["category"], (Counter(), Counter(), set()))
        counts.update(documents=1)
        counts["characters"] += len(text)
        simple = bool(_SIMPLIFIED.intersection(text))
        traditional = bool(_TRADITIONAL.intersection(text))
        script = "mixed" if simple and traditional else "simplified" if simple else "traditional" if traditional else "unknown"
        counts[f"script_{script}"] += 1
        classical = len(_CLASSICAL.findall(text)) >= 2
        modern = bool(_MODERN.search(text))
        register = "mixed" if classical and modern else "classical" if classical else "modern" if modern else "unknown"
        counts[f"register_{register}"] += 1
        for sentence in re.split(r"[。！？!?；;\n]+", text):
            if sentence.strip():
                lengths[len(sentence.strip())] += 1
        for character in text:
            kind = unicodedata.category(character)[0]
            counts["punctuation"] += kind == "P"
            counts["special_characters"] += kind == "S" or (kind == "C" and not character.isspace())
        counts["wiki_marker_documents"] += bool(_WIKI.search(text))
        counts["abnormal_documents"] += bool("\ufffd" in text or re.search(r"([^\s])\1{19,}", text))
        repeated = False
        for paragraph in text.splitlines():
            normalized = re.sub(r"\s+", "", paragraph)
            if len(normalized) < 16:
                continue
            digest = identity(normalized)
            if digest in seen_paragraphs:
                counts["repeated_paragraph_occurrences"] += 1
                repeated = True
            seen_paragraphs.add(digest)
        counts["documents_with_repeated_paragraph"] += repeated
        for style, pattern in _STYLES.items():
            counts[f"style_{style}"] += bool(pattern.search(text))
    verify_file(path, train_info)
    report = {"schema_version": 1, "corpus_version": corpus_version, "train": train_info,
              "basis": "all quality-kept deduplicated Train documents before balanced selection",
              "limitations": _LIMITATIONS, "categories": {}}
    for category, (counts, lengths, _) in sorted(groups.items()):
        n = lengths.total()
        median = None
        seen = 0
        for length, count in sorted(lengths.items()):
            seen += count
            if 2 * seen >= n:
                median = length
                break
        report["categories"][category] = {
            "documents": counts["documents"], "characters": counts["characters"],
            "script_cues": {k: counts[f"script_{k}"] for k in ("simplified", "traditional", "mixed", "unknown")},
            "register_cues": {k: counts[f"register_{k}"] for k in ("modern", "classical", "mixed", "unknown")},
            "sentence_segments": n, "median_segment_characters": median,
            "punctuation_fraction": counts["punctuation"] / counts["characters"],
            "special_character_fraction": counts["special_characters"] / counts["characters"],
            **{k: counts[k] for k in ("wiki_marker_documents", "abnormal_documents",
                                      "repeated_paragraph_occurrences", "documents_with_repeated_paragraph")},
            "style_cues": {k: counts[f"style_{k}"] for k in _STYLES}}
    return report


def quality_markdown(report):
    lines = ["# 中文语料质量诊断", "", f"源版本为 `{report['corpus_version']}`。统计范围是 Split 后的全量 Train。",
             "", "## 字形与语体线索", "",
             "| 类别 | 文档 | 简体 / 繁体 / 混合 / 未知 | 现代 / 古文 / 混合 / 未知 | 句段数 | 句段字数中位数 |",
             "|---|---:|---|---|---:|---:|"]
    for category, row in report["categories"].items():
        script = " / ".join(map(str, row["script_cues"].values()))
        register = " / ".join(map(str, row["register_cues"].values()))
        lines.append(f"| {category} | {row['documents']:,} | {script} | {register} | "
                     f"{row['sentence_segments']:,} | {row['median_segment_characters']} |")
    lines += ["", "## 噪声与文体线索", "",
              "| 类别 | 标点 | 特殊字符 | Wiki 标记文档 | 异常文档 | 重复段落次数 | 对话 / 叙述 / 说明 / 百科 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for category, row in report["categories"].items():
        styles = " / ".join(map(str, row["style_cues"].values()))
        lines.append(f"| {category} | {row['punctuation_fraction']:.2%} | {row['special_character_fraction']:.2%} | "
                     f"{row['wiki_marker_documents']} | {row['abnormal_documents']} | "
                     f"{row['repeated_paragraph_occurrences']} | {styles} |")
    lines += ["", "## 规则局限", "", *(f"- {item}" for item in report["limitations"])]
    return "\n".join(lines) + "\n"
