"""Deterministic, offline performance benchmark for the corpus-v4 pipeline.

The benchmark generates its own synthetic JSONL sources, runs the *real* v4
ingest/quality/dedup/split stages, and records wall time, peak RSS, SQLite size,
temporary-file high-water mark and output size.  It never downloads data, uses a
GPU, or trains a tokenizer/model.  Large runs are opt-in through the CLI.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import sqlite3
import sys
import threading
import time
from pathlib import Path

from .corpus_utils import atomic_write, identity, sha256, write_json
from .corpus_v4 import CorpusV4Pipeline, build_registry, inventory, load_config

BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_DOCUMENTS = 1000
STAGES = ("ingest", "quality", "dedup", "split")
CATEGORY_ORDER = ("zh_general", "zh_encyclopedic", "zh_education", "en_general", "math_logic_code")
CATEGORY_SHARE = {"zh_general": 40, "zh_encyclopedic": 25, "zh_education": 15,
                  "en_general": 10, "math_logic_code": 10}
SOURCE_ORDER = ("bench-zh-general", "bench-zh-encyclopedic", "bench-zh-education",
                "bench-en-general", "bench-math-code")
SOURCE_FOR_CATEGORY = {
    "zh_general": "bench-zh-general",
    "zh_encyclopedic": "bench-zh-encyclopedic",
    "zh_education": "bench-zh-education",
    "en_general": "bench-en-general",
    "math_logic_code": "bench-math-code",
}


def _category_for(index: int) -> str:
    position = index % 100
    boundary = 0
    for category in CATEGORY_ORDER:
        boundary += CATEGORY_SHARE[category]
        if position < boundary:
            return category
    return CATEGORY_ORDER[-1]


def _cjk_marker(index: int, seed: int, salt: int, length: int = 180) -> str:
    """A deterministic, document-unique CJK sequence.

    The generator uses unique content rather than one shared paragraph plus a
    short suffix.  Shared paragraphs make LSH retrieve almost every pair as a
    candidate even when the true Jaccard is below the policy threshold, which
    turns a scale benchmark into an O(n^2) exact-comparison test.
    """
    # SHAKE gives a per-document pseudorandom sequence.  A simple affine
    # sequence is not unique enough: different documents can share many
    # 5-grams, which makes LSH retrieve almost every pair as a candidate and
    # turns the benchmark into an accidental quadratic test.
    stream = hashlib.shake_256(f"{seed}:{salt}:{index}".encode("utf-8")).digest(length * 2)
    return "".join(
        chr(0x4E00 + (int.from_bytes(stream[position * 2:position * 2 + 2], "big") % 18000))
        for position in range(length))


def _zh_clauses(index: int, seed: int) -> str:
    return (f"合成社区观察样本{index}（seed={seed}）。"
            + _cjk_marker(index, seed, 1)
            + "。这些文字仅用于离线性能测试。")


def _zh_encyclopedic(index: int, seed: int) -> str:
    return (f"合成百科样本{index}（seed={seed}）。"
            + _cjk_marker(index, seed, 2)
            + "。条目内容不代表真实事实。")


def _zh_education(index: int, seed: int) -> str:
    return (f"合成课程样本{index}（seed={seed}）。"
            + _cjk_marker(index, seed, 3)
            + "。材料仅用于测试清洗与切分。")


def _letters(number: int, width: int) -> str:
    # Exclude letters that can combine into the contact keywords used by the
    # quality PII rule ("qq", "tel") while remaining a base-22 token alphabet.
    alphabet = "abcdfghijkmnoprsuvwyz"
    value = int(number)
    out = []
    for _ in range(width):
        out.append(alphabet[value % len(alphabet)])
        value //= len(alphabet)
    return "".join(reversed(out))


def _en_text(index: int, seed: int) -> str:
    # Letter-only tokens avoid triggering the digit-density review rule while
    # still making every English document a unique word-trigram sequence.
    tokens = " ".join(f"zt{_letters(index, 6)}{_letters(position, 4)}" for position in range(44))
    return (f"Synthetic row {index}, seed {seed}. Unique tokens {tokens}.")


def _math_text(index: int, seed: int) -> str:
    return (f"合成数学样本{index}（seed={seed}）。"
            + _cjk_marker(index, seed, 4)
            + f"。设变量值为 {index % 97 + 3}，仅用于离线基准。")


def _code_text(index: int, seed: int) -> str:
    tokens = " ".join(f"ct{index:06x}{position:04x}" for position in range(24))
    return (
        f"# synthetic benchmark helper {index}; seed={seed}\n"
        f"# unique tokens {tokens}\n"
        "def adjusted_score(values, threshold=0.5):\n"
        "    \"\"\"Return values above a synthetic threshold.\"\"\"\n"
        "    cutoff = threshold * max(len(values), 1)\n"
        "    return [value for value in values if value >= cutoff]\n"
    )


def _base_text(category: str, index: int, seed: int) -> str:
    if category == "zh_general":
        return _zh_clauses(index, seed)
    if category == "zh_encyclopedic":
        return _zh_encyclopedic(index, seed)
    if category == "zh_education":
        return _zh_education(index, seed)
    if category == "en_general":
        return _en_text(index, seed)
    return _code_text(index, seed) if index % 2 else _math_text(index, seed)


def _near_variant(text: str) -> str:
    """Change a short suffix while keeping most character 5-grams intact."""
    marker = "补充说明：这是同一合成样本的近重复变体。"
    if text.endswith(marker):
        return text + "（第二变体）"
    return text + marker


def _source_specs() -> list[dict]:
    return [
        {
            "source_id": "bench-zh-general", "source_name": "Synthetic benchmark Chinese general",
            "source_uri": "benchmark://synthetic/zh-general", "license": "CC0-1.0",
            "license_status": "APPROVED",
            "license_evidence": "Generated by corpus_benchmark.py; no external text.",
            "license_reviewer": "benchmark-generator", "license_reviewed_at": "2025-05-17",
            "category": "zh_general", "language": "zh", "provenance_status": "DEMO_ONLY",
            "paths": ["zh-general.jsonl"], "priority": 10, "family_policy": "document",
        },
        {
            "source_id": "bench-zh-encyclopedic", "source_name": "Synthetic benchmark Chinese encyclopedic",
            "source_uri": "benchmark://synthetic/zh-encyclopedic", "license": "CC0-1.0",
            "license_status": "APPROVED",
            "license_evidence": "Generated by corpus_benchmark.py; no external text.",
            "license_reviewer": "benchmark-generator", "license_reviewed_at": "2025-05-17",
            "category": "zh_encyclopedic", "language": "zh", "provenance_status": "DEMO_ONLY",
            "paths": ["zh-encyclopedic.jsonl"], "priority": 9, "family_policy": "document",
        },
        {
            "source_id": "bench-zh-education", "source_name": "Synthetic benchmark Chinese education",
            "source_uri": "benchmark://synthetic/zh-education", "license": "CC0-1.0",
            "license_status": "APPROVED",
            "license_evidence": "Generated by corpus_benchmark.py; no external text.",
            "license_reviewer": "benchmark-generator", "license_reviewed_at": "2025-05-17",
            "category": "zh_education", "language": "zh", "provenance_status": "DEMO_ONLY",
            "paths": ["zh-education.jsonl"], "priority": 8, "family_policy": "document",
        },
        {
            "source_id": "bench-en-general", "source_name": "Synthetic benchmark English general",
            "source_uri": "benchmark://synthetic/en-general", "license": "CC0-1.0",
            "license_status": "APPROVED",
            "license_evidence": "Generated by corpus_benchmark.py; no external text.",
            "license_reviewer": "benchmark-generator", "license_reviewed_at": "2025-05-17",
            "category": "en_general", "language": "en", "provenance_status": "DEMO_ONLY",
            "paths": ["en-general.jsonl"], "priority": 7, "family_policy": "document",
        },
        {
            "source_id": "bench-math-code", "source_name": "Synthetic benchmark math and code",
            "source_uri": "benchmark://synthetic/math-code", "license": "CC0-1.0",
            "license_status": "APPROVED",
            "license_evidence": "Generated by corpus_benchmark.py; no external text.",
            "license_reviewer": "benchmark-generator", "license_reviewed_at": "2025-05-17",
            "category": "math_logic_code", "language": "auto", "provenance_status": "DEMO_ONLY",
            "paths": ["math-code.jsonl"], "priority": 6, "family_policy": "document",
        },
    ]


def generate_synthetic_corpus(directory: str | Path, documents: int, *, seed: int = 42) -> dict:
    """Write deterministic JSONL sources and return generation metadata.

    Exact duplicates use the last document of the same category; near duplicates
    append a short suffix to a previous document.  Every group of three documents
    shares a parent id, so split isolation exercises both family and dedup-cluster
    connectivity.
    """
    if type(documents) is not int or documents < 30:
        raise ValueError("documents must be an integer >= 30")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    specs = _source_specs()
    manifest = {"schema_version": 4, "demo_only": True,
                "note": "Synthetic benchmark; generated by corpus_benchmark.py. No external text.",
                "seed": seed, "sources": specs}
    manifest_path = directory / "sources.json"
    write_json(manifest_path, manifest)
    paths = {source["source_id"]: directory / source["paths"][0] for source in specs}
    handles = {source_id: path.open("w", encoding="utf-8") for source_id, path in paths.items()}
    category_counts = {category: 0 for category in CATEGORY_ORDER}
    source_counts = {source_id: 0 for source_id in SOURCE_ORDER}
    last_by_category: dict[str, str] = {}
    exact_events = 0
    near_events = 0
    family_groups = 0
    try:
        for index in range(documents):
            category = _category_for(index)
            source_id = SOURCE_FOR_CATEGORY[category]
            text = _base_text(category, index, seed)
            # Keep duplicate events sparse at scale, but guarantee coverage in
            # small deterministic runs too.  Near variants are only injected for
            # Chinese prose so code (excluded from near-dedup by policy) and
            # English 3-gram comparisons do not dominate small benchmarks.
            exact_step = 997 if documents >= 5000 else 31
            near_step = 499 if documents >= 5000 else 37
            if index >= exact_step and index % exact_step == 0 and category in last_by_category:
                text = last_by_category[category]
                exact_events += 1
            elif (index >= near_step and index % near_step == 0 and category in last_by_category
                  and category in ("zh_general", "zh_encyclopedic", "zh_education")):
                text = _near_variant(last_by_category[category])
                near_events += 1
            parent = f"family-{index // 3}"
            if index % 3 == 0:
                family_groups += 1
            record = {"text": text, "parent_id": parent, "author": "synthetic-benchmark",
                      "original_format": "jsonl", "acquisition_date": "2025-05-17"}
            handles[source_id].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            category_counts[category] += 1
            source_counts[source_id] += 1
            last_by_category[category] = text
    finally:
        for handle in handles.values():
            handle.close()
    file_hashes = {}
    for source_id, path in paths.items():
        file_hashes[source_id] = {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
    # Aggregate input identity over the manifest and all source files.
    digest = hashlib.sha256()
    for source_id in sorted(file_hashes):
        digest.update(source_id.encode())
        digest.update(file_hashes[source_id]["sha256"].encode())
    input_sha256 = digest.hexdigest()
    metadata = {
        "schema_version": BENCHMARK_SCHEMA_VERSION, "seed": seed,
        "documents": documents, "directory": str(directory.resolve()),
        "manifest": "sources.json", "manifest_sha256": sha256(manifest_path),
        "input_sha256": input_sha256, "files": file_hashes,
        "category_counts": category_counts, "source_counts": source_counts,
        "exact_duplicate_events": exact_events, "near_duplicate_events": near_events,
        "family_groups": family_groups,
        "generator": "corpus_benchmark.generate_synthetic_corpus",
    }
    write_json(directory / "generation.json", metadata)
    return metadata


def _rss_bytes() -> int:
    # Linux ru_maxrss is KiB; macOS is bytes.  This benchmark is documented for
    # Linux and falls back to the process high-water mark when available.
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value) * (1024 if sys.platform != "darwin" else 1)


def _tree_bytes(root: Path, *, exclude_control: bool = False) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if exclude_control and any(part == ".v4" for part in path.parts):
            continue
        try:
            total += path.stat().st_size
        except OSError:
            pass
    return total


class ResourceSampler(threading.Thread):
    """Sample temp/work-dir size and process RSS without touching pipeline state."""

    def __init__(self, out: Path, interval: float = 0.25):
        super().__init__(daemon=True)
        self.out = Path(out)
        self.interval = interval
        self.stop_event = threading.Event()
        self.temp_peak = 0
        self.rss_peak = _rss_bytes()

    def _sample(self):
        self.rss_peak = max(self.rss_peak, _rss_bytes())
        work = self.out / ".v4" / "work"
        temp = _tree_bytes(work)
        # Split temporary files are hidden in the output root.
        try:
            for path in self.out.glob(".*.tmp"):
                if path.is_file():
                    temp += path.stat().st_size
        except OSError:
            pass
        self.temp_peak = max(self.temp_peak, temp)

    def run(self):
        while not self.stop_event.wait(self.interval):
            self._sample()

    def stop(self) -> dict:
        self.stop_event.set()
        self.join(timeout=2)
        self._sample()
        return {"peak_rss_bytes": self.rss_peak, "temp_peak_bytes": self.temp_peak}


def _stage_stats(pipeline: CorpusV4Pipeline) -> dict:
    ingest = pipeline.stats.get("ingested", {})
    quality = pipeline.stats.get("quality", {})
    dedup = pipeline.stats.get("dedup", {})
    split = pipeline.stats.get("split", {})
    return {"ingest": ingest, "quality": quality, "dedup": dedup, "split": split}


def _metric_value(stats: dict, *keys, default=None):
    value = stats
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _display(value) -> str:
    return "NOT_RUN" if value is None else str(value)


def _display_json(value) -> str:
    return "NOT_RUN" if value is None else f"`{json.dumps(value, ensure_ascii=False, sort_keys=True)}`"


def _summary(report: dict) -> str:
    metrics = report["metrics"]
    lines = [
        "# Corpus v4 performance benchmark",
        "",
        f"- status: `{report['status']}`",
        f"- documents requested: {report['parameters']['documents']:,}",
        f"- generated input: `{report['corpus']['directory']}`",
        f"- seed: {report['parameters']['seed']}",
        f"- input SHA256: `{report['corpus']['input_sha256']}`",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| total seconds | {metrics['total_seconds']:.3f} |",
    ]
    for name, seconds in metrics["stage_seconds"].items():
        lines.append(f"| {name} seconds | {seconds:.3f} |")
    lines += [
        f"| peak RSS bytes | {metrics['peak_rss_bytes']} |",
        f"| peak RSS MiB | {metrics['peak_rss_bytes'] / 1048576:.2f} |",
        f"| SQLite bytes | {metrics['sqlite_bytes']} |",
        f"| temporary peak bytes | {metrics['temp_peak_bytes']} |",
        f"| output bytes | {metrics['output_bytes']} |",
        f"| retained documents | {_display(metrics['retained_documents'])} |",
        f"| deleted documents | {_display(metrics['deleted_documents'])} |",
        f"| groups | {_display(metrics['group_count'])} |",
        f"| leakage | {_display_json(metrics['leakage'])} |",
        f"| not run | {', '.join(metrics['not_run']) if metrics['not_run'] else 'none'} |",
        "",
        "All values above are measured by this run; absent optional stages are `NOT_RUN`.",
        "",
    ]
    return "\n".join(lines)


def run_benchmark(config_path: str | Path, out: str | Path, *, documents: int = DEFAULT_DOCUMENTS,
                  seed: int = 42, input_dir: str | Path | None = None,
                  stages: tuple[str, ...] | list[str] = STAGES, log=print) -> dict:
    """Run a benchmark and write ``benchmark-report.json`` + summary.

    ``demo=True`` is used deliberately: generated sources are synthetic and this
    function must never imply that they are a real licensed corpus.  The four
    measured stages are the exact pipeline stages; no tokenizer is trained.
    """
    requested = tuple(stages)
    unknown = set(requested) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown benchmark stages: {sorted(unknown)}")
    if not requested:
        raise ValueError("at least one benchmark stage is required")
    out = Path(out)
    input_dir = Path(input_dir) if input_dir else out.parent / f"{out.name}-input"
    if out.exists():
        raise ValueError(f"benchmark output already exists; choose a new directory: {out}")
    if input_dir.exists():
        raise ValueError(f"benchmark input already exists; choose a new directory: {input_dir}")
    config = load_config(Path(config_path))
    corpus = generate_synthetic_corpus(input_dir, documents, seed=seed)
    raw_manifest = json.loads((input_dir / corpus["manifest"]).read_text(encoding="utf-8"))
    registry = build_registry(raw_manifest["sources"], demo=True,
                              license_policy=config["license_policy"])
    inputs = inventory(registry, input_dir)
    out.mkdir(parents=True)
    report_path = out / "benchmark-report.json"
    summary_path = out / "benchmark-summary.md"
    sampler = ResourceSampler(out)
    sampler.start()
    started = time.perf_counter()
    timings: dict[str, float] = {}
    pipeline = None
    report = None
    try:
        pipeline = CorpusV4Pipeline(out, config, registry, inputs, demo=True, log=log)

        def timed(name, function):
            begin = time.perf_counter()
            result = function()
            timings[name] = round(time.perf_counter() - begin, 6)
            return result

        # Prerequisites are explicit: a split benchmark necessarily includes its
        # upstream stages, but every executed stage gets its own wall time.
        timed("ingest", pipeline.ingest)
        if "quality" in requested or "dedup" in requested or "split" in requested:
            timed("quality", pipeline.quality)
        if "dedup" in requested or "split" in requested:
            timed("dedup", pipeline.deduplicate)
        if "split" in requested:
            timed("split", pipeline.split)
        elapsed = time.perf_counter() - started
        resources = sampler.stop()
        stats = _stage_stats(pipeline)
        split_stats = stats["split"]
        leakage = split_stats.get("leakage") if split_stats else None
        dedup_stats = stats["dedup"]
        retained = int(split_stats["documents"]) if split_stats else None
        deleted = (int(_metric_value(dedup_stats, "clusters", "deleted_documents", default=0) or 0)
                   if dedup_stats else None)
        group_count = int(split_stats["groups"]) if split_stats else None
        not_run = [stage for stage in STAGES if stage not in timings]
        sqlite_files = sorted(out.glob("*.sqlite"))
        sqlite_bytes = sum(path.stat().st_size for path in sqlite_files)
        report = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "status": "COMPLETED" if not not_run else "PARTIAL",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "parameters": {"documents": documents, "seed": seed, "requested_stages": list(requested),
                           "config": str(Path(config_path).resolve()),
                           "input_dir": str(input_dir.resolve()), "out": str(out.resolve())},
            "environment": {
                "python": sys.version.split()[0], "platform": platform.platform(),
                "sqlite": sqlite3.sqlite_version, "cpu_count": os.cpu_count(),
                "gpu": "not used", "network": "not used",
                "config_sha256": identity(config),
            },
            "corpus": corpus,
            "metrics": {
                "total_seconds": round(elapsed, 6),
                "stage_seconds": timings,
                "peak_rss_bytes": resources["peak_rss_bytes"],
                "temp_peak_bytes": resources["temp_peak_bytes"],
                "sqlite_bytes": sqlite_bytes,
                "sqlite_files": {path.name: path.stat().st_size for path in sqlite_files},
                "output_bytes": _tree_bytes(out),
                "output_payload_bytes": _tree_bytes(out, exclude_control=True),
                "retained_documents": retained,
                "deleted_documents": deleted,
                "group_count": group_count,
                "leakage": leakage,
                "not_run": not_run,
                "stage_stats": stats,
            },
            "limitations": (
                "Synthetic corpus, CPU only, no tokenizer/model training. Peak RSS includes the "
                "whole Python process and is not a per-stage allocation measurement. Temp high-water "
                "is sampled at 0.25s intervals and can miss a shorter spike."),
        }
        report["summary_path"] = str(summary_path)
        atomic_write(report_path, lambda handle: handle.write(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"))
        atomic_write(summary_path, lambda handle: handle.write(_summary(report).encode("utf-8")))
        return report
    except Exception as exc:
        resources = sampler.stop()
        elapsed = time.perf_counter() - started
        failure = {
            "schema_version": BENCHMARK_SCHEMA_VERSION, "status": "FAILED",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "parameters": {"documents": documents, "seed": seed, "requested_stages": list(requested),
                           "input_dir": str(input_dir.resolve()), "out": str(out.resolve())},
            "corpus": corpus, "error": f"{type(exc).__name__}: {exc}",
            "metrics": {"total_seconds": round(elapsed, 6), "stage_seconds": timings,
                        "peak_rss_bytes": resources["peak_rss_bytes"],
                        "temp_peak_bytes": resources["temp_peak_bytes"]},
        }
        try:
            atomic_write(report_path, lambda handle: handle.write(
                json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"))
        except OSError:
            pass
        raise
    finally:
        if sampler.is_alive():
            sampler.stop()
