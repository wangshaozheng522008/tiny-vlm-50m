#!/usr/bin/env python3
"""Verify a historical reviewed Split, then publish local reuse and quality reports."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minillm.assessment_io import report_directory
from minillm.chinese_language_quality import analyze_chinese_train, quality_markdown
from minillm.corpus_utils import CorpusError, verify_file, write_json
from minillm.corpus_v4 import _split_isolation_check
from minillm.v4_language_pilot import LANGUAGE_VERSION, LEGACY_PILOT_VERSION
from minillm.v4_pilot_sampling import SAMPLE_KIND
from minillm.v4_sources import validate_source_plan


def reuse_draft(plan):
    if plan["corpus_version"] != LEGACY_PILOT_VERSION or plan["demo"] is not False:
        raise CorpusError("reuse draft requires the original formal Pilot source plan")
    hashes = validate_source_plan(plan, plan["registry"], require=True, verify_inputs=True)
    sources = []
    for spec in plan["generation_config"]["sources"]:
        if spec["source_id"] not in hashes or spec["category"] == "math_logic_code":
            continue
        entry = copy.deepcopy(spec["source_review"]["entry"])
        source = entry["v4_source"]
        source.update(license_status="REVIEW_REQUIRED", license_reviewer="", license_reviewed_at=None,
                      license_evidence="")
        for field in ("commercial_use", "redistribution", "model_training", "license_obligations"):
            entry[field] = "PENDING"
        entry["corpus_versions"] = [LANGUAGE_VERSION]
        for item in source["transformation_history"]:
            if isinstance(item, dict) and item.get("kind") == SAMPLE_KIND:
                item["reuse_from_corpus_version"] = LEGACY_PILOT_VERSION
        sources.append(entry)
    return {"schema_version": 1, "sources": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.historical_dataset.resolve()
    plan = json.loads((dataset / ".v4/plan.json").read_text(encoding="utf-8"))
    from minillm.corpus_utils import identity

    draft = reuse_draft(plan)
    keys = {}
    for stage, extra in (
        ("ingested", lambda: {}),
        ("quality", lambda: {"ingested": keys["ingested"], "quality": plan["policies"]["quality_policy_sha256"],
                             "text_policy": plan["policies"]["text_policy"]}),
        ("dedup", lambda: {"quality": keys["quality"], "dedup": plan["policies"]["dedup_policy_sha256"]}),
        ("split", lambda: {"dedup": keys["dedup"], "split": plan["policies"]["split_policy_sha256"],
                           "seed": plan["generation_config"]["split"]["seed"]}),
    ):
        marker = json.loads((dataset / f".v4/{stage}.stage.json").read_text(encoding="utf-8"))
        keys[stage] = identity([4, plan["plan_key"], stage, extra()])
        if keys[stage] != marker["key"]:
            raise CorpusError(f"historical {stage} differs from source plan")
        for path, info in marker["files"].items():
            verify_file(Path(path), info)
        if stage == "split" and str(dataset / "split-manifest.json") not in marker["files"]:
            raise CorpusError("historical Split stage lacks manifest binding")
    if _split_isolation_check(dataset)["status"] != "PASS":
        raise CorpusError("historical dataset does not have verified split isolation")
    report = analyze_chinese_train(dataset / "train.text.jsonl", dataset / "split-manifest.json",
                                   corpus_version=plan["corpus_version"])
    report["source_plan_key"] = plan["plan_key"]
    with report_directory(args.out) as output:
        write_json(output / "source-review.draft.json", draft)
        write_json(output / "chinese-language-quality-report.json", report)
        (output / "chinese-language-quality-report.md").write_text(quality_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
