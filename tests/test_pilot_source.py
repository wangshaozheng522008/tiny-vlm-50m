import json
from pathlib import Path

import pytest

from minillm.assessment_io import records
from minillm.corpus_quality import comparison_hash
from minillm.corpus_utils import file_info
from minillm.pilot_source import prepare_source


def write(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return path


def test_historical_exclusion_cross_category_near_and_reproducibility(tmp_path):
    old_text = "春天的河岸边，孩子们观察花朵和昆虫。" + "每一种生物都能在自然环境中找到适合自己的位置。" * 12
    code = "def square(x):\n    return x * x\n"
    historical = write(tmp_path / "old.jsonl", [{"text": old_text}, {"text": code, "category": "code"}])
    keep = "冬夜的实验室里，研究员仔细记录恒星光谱，比较不同元素的发射波长。"
    source = write(tmp_path / "raw.jsonl", [{"text": "skip"}, {"text": old_text, "source": "repost"},
        {"text": code.replace("\n", " "), "category": "zh"}, {"text": "新标题：" + old_text}, {"text": keep}, {"text": "outside"}])
    reports = [prepare_source(source, [historical], tmp_path / f"out{i}", skip=1, take=4, radius=8) for i in range(2)]
    assert reports[0] == reports[1]
    assert reports[0]["counts"] == {"historical_records": 2, "candidates": 4, "excluded_exact": 2, "excluded_near": 1, "emitted": 1}
    assert [r["text"] for r in records(tmp_path / "out0/source.jsonl")] == [keep]
    assert file_info(tmp_path / "out0/source.jsonl") == reports[0]["source"]
    assert not list((tmp_path / "out0").glob("*.sqlite*"))
    assert comparison_hash(keep) != comparison_hash(old_text)
    with pytest.raises(ValueError, match="non-empty output"):
        prepare_source(source, [historical], tmp_path / "out0")


def test_historical_exclusion_failure_does_not_publish(tmp_path):
    historical = write(tmp_path / "old.jsonl", [{"text": "known"}])
    source = write(tmp_path / "raw.jsonl", [{"text": "unseen"}, {"text": 3}])
    with pytest.raises(ValueError, match="string text"):
        prepare_source(source, [historical], tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out.*"))
    write(source, [{"text": "known"}])
    with pytest.raises(ValueError, match="no source records"):
        prepare_source(source, [historical], tmp_path / "out")


def test_historical_exclusion_candidate_overflow_refuses_truncated_check(tmp_path):
    text = "历史资料详细描述城市规划、道路建设和居民日常生活。" * 10
    historical = write(tmp_path / "old.jsonl", [{"text": text + "甲"}, {"text": text + "乙"}])
    source = write(tmp_path / "raw.jsonl", [{"text": text + "丙"}])
    with pytest.raises(ValueError, match="candidate budget"):
        prepare_source(source, [historical], tmp_path / "out", max_candidates=1, radius=8)
    assert not (tmp_path / "out").exists()
