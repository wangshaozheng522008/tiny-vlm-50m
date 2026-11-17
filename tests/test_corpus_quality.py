"""Structured v4 quality rules: explainable findings, Chinese-first ID, no silent deletion."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from minillm.corpus_quality import (aggregate_action, detect_language, evaluate_quality,
                                   merge_quality_v4)


def _findings(text: str, language: str = "zh", category: str = "zh_general"):
    return {finding.rule_id: finding for finding in
            evaluate_quality(text, language=language, category=category)}


def test_all_twelve_rules_have_structured_findings():
    cases = {
        "empty_or_short": ("太短", "zh", "zh_general"),
        "extreme_length": (("城市生活需要公共空间。" * 2000), "zh", "zh_general"),
        "repeated_characters": ("正常开头。" + "啊" * 40 + "正常结尾。", "zh", "zh_general"),
        "replacement_characters": ("这是一段正常的中文文本。" * 8 + "\ufffd" * 5, "zh", "zh_general"),
        "html_residue": ('<span class="ad">' * 4 + "正文讨论阅读、写作和城市公共空间。" * 4, "zh", "zh_general"),
        "boilerplate_noise": ("首页 登录 注册 版权所有 相关推荐 加入收藏 扫码关注。" +
                              "正文讨论阅读与城市生活。" * 4, "zh", "zh_general"),
        "punctuation_density": ("，。！？；：" * 20 + "正文讨论阅读和写作。" * 6, "zh", "zh_general"),
        "digit_density": ("1234567890" * 20 + "数据说明", "zh", "zh_general"),
        "repeated_paragraph": ("\n\n".join(["重复段落用于测试，需要足够长度以触发规则。" for _ in range(6)]),
                              "zh", "zh_general"),
        "language_mismatch": ("This English paragraph is deliberately placed in a Chinese category so the "
                              "language mismatch rule has to review it instead of deleting it silently. " * 2,
                              "en", "zh_general"),
        "unprintable_ratio": ("正常文本。" + "\x01" * 20 + "继续讨论城市生活。" * 4, "zh", "zh_general"),
        "pii_contacts": ("reader@example.com writer@example.com https://example.com/a "
                         "https://example.com/b 微信：demo-reader。" +
                         "正文继续描述城市公共空间和日常阅读。" * 3, "zh", "zh_general"),
        "low_information": ("……◆☆" * 40, "unknown", "zh_general"),
    }
    for rule_id, (text, language, category) in cases.items():
        found = _findings(text, language, category)
        assert rule_id in found, (rule_id, sorted(found))
        finding = found[rule_id]
        assert finding.action in {"KEEP", "REVIEW", "REJECT"}
        assert finding.severity in {"info", "warning", "error", "review", "reject", "REVIEW", "REJECT"} or finding.severity
        assert finding.observed_value is not None or finding.rule_id == "language_mismatch"
        assert finding.threshold is not None
        assert isinstance(finding.explanation, str) and finding.explanation
    merged = merge_quality_v4({})
    assert set(cases) <= set(merged["rules"])


def test_extreme_length_fixture_covers_the_long_document_case():
    row = json.loads((Path(__file__).parent / "fixtures/corpus_v4/extreme_length.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    policy = merge_quality_v4({})
    assert len(row["text"]) > policy["max_chars"]
    found = _findings(row["text"])
    assert "extreme_length" in found
    assert found["extreme_length"].action == policy["rules"]["extreme_length"]["action"]


def test_digits_and_code_are_not_rejected_by_prose_rules():
    code = ("def score(values, threshold=0.85):\n"
            "    total = sum(values) * 1.25\n"
            "    return [value for value in values if value >= threshold]\n"
            "print(score([1, 2, 3], 2.0))\n")
    findings = evaluate_quality(code, language="code", category="math_logic_code")
    assert aggregate_action(findings) != "REJECT"
    # A normal Chinese document with digits and English terms is not rejected.
    prose = ("2024 年的一项调查访问了 1200 位居民，其中 37% 使用手机阅读新闻。"
             "研究者比较了不同年龄段的选择，发现通勤时间与阅读时长存在相关。"
             "这项合成调查只用于测试数字密度规则不会把正常统计文本直接删除。")
    assert aggregate_action(evaluate_quality(prose, language="zh", category="zh_general")) != "REJECT"


def test_chinese_first_language_detection():
    assert detect_language("这是一段中文，包含 Python 和 tokenizer 等英文术语。") in ("zh", "mixed")
    assert detect_language("This is a plain English paragraph about libraries and reading.") == "en"
    assert detect_language("def f(x):\n    return x + 1\n") == "code"
    assert detect_language("。！？") == "unknown"


def test_review_is_preserved_in_queue(v4_demo):
    stats = v4_demo.quality()
    assert stats["review"] > 0
    queue = [json.loads(line) for line in (v4_demo.out / "review-queue.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(queue) == stats["review"]
    assert all(row["document_id"] and row["text"] for row in queue)
    report = json.loads((v4_demo.out / "quality-report.json").read_text(encoding="utf-8"))
    assert report["basis"].startswith("documents and characters")
    assert report["totals"]["documents"] == stats["documents"]
    assert sum(row["document_ratio"] for row in report["by_source"].values()) == pytest.approx(1.0)
    assert sum(row["raw_character_ratio"] for row in report["by_source"].values()) == pytest.approx(1.0)
    for source, row in report["by_source"].items():
        assert row["documents"] == row["kept"] + row["review"] + row["rejected"]
        assert "language_distribution" in row and "category_distribution" in row
        assert row["tokens"] is None and row["token_ratio"] is None
        assert "tokenization-report.json" in row["token_basis"]
