"""Pinned policy and whole-document selection for the Chinese language Pilot."""
from __future__ import annotations

import json
from fractions import Fraction
from math import ceil
from pathlib import Path

from .corpus import read_records
from .corpus_budget import read_token_index, supervision_by_category
from .corpus_quality import V4_CATEGORIES
from .corpus_utils import CorpusError, file_info, identity, sha256
from .v4_language_scaling import CORPUS_CONFIG as SCALING_CONFIG, VERSION as SCALING_VERSION

LANGUAGE_VERSION = "minillm-corpus-v4-language-pilot-1m-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LANGUAGE_CONFIG = PROJECT_ROOT / "configs/corpus_v4_language_pilot_1m.yaml"
LEGACY_PILOT_VERSION = "minillm-corpus-v4-pilot-1m-v1"
LANGUAGE_TOLERANCE = 0.01
BENCHMARK = "data/evaluation/chinese-language-v1/tasks.jsonl"
BENCHMARK_SHA256 = "92e7e2899a30d9648eb05035acfbaf7c2736f85cdbec774664c3385c8c42762e"
TOKENIZER_EVAL_SHA256 = "c50e2fbac13ca8dbcc8bbec9a2b5c7e8fbaa4094ce7807276241158a22d6f0ec"
SELECTION_REPORT = "train-selection.json"
AUDIT_RECEIPT = ".v4/audit-receipt.json"
SELECTED_TRAIN = "train.selected.jsonl"


def is_language(config):
    return config["corpus_version"] in (LANGUAGE_VERSION, SCALING_VERSION)


def language_config(version):
    return {LANGUAGE_VERSION: LANGUAGE_CONFIG, SCALING_VERSION: SCALING_CONFIG}[version]


def validate_version_targets(config):
    """Only the named language profile can opt out of a category."""
    if not is_language(config) and 0 in config["targets"]["category_token_ratios"].values():
        raise CorpusError("zero category weight requires the approved language Pilot version")


def validate_language_policy(config):
    from .corpus_v4 import load_config
    from .v4_admission import _pilot_policy

    approved = load_config(language_config(config["corpus_version"]))
    if (_pilot_policy(config) != _pilot_policy(approved)
            or config["tokenizer"]["candidates"] != approved["tokenizer"]["candidates"]
            or config["tokenizer"]["eval_asset"] != approved["tokenizer"]["eval_asset"]
            or config["tokenizer"]["baseline"] != approved["tokenizer"]["baseline"]
            or config["tokenizer"]["selected"] is not None):
        raise CorpusError("language Pilot requires the pinned repository generation policy")
    from .corpus_contamination import load_benchmark_items

    root = PROJECT_ROOT
    if (sha256(root / BENCHMARK) != BENCHMARK_SHA256
            or sha256(root / approved["tokenizer"]["eval_asset"]) != TOKENIZER_EVAL_SHA256):
        raise CorpusError("language Pilot fixed evaluation asset hash changed")
    _, _, errors = load_benchmark_items([
        {"path": str(root / asset["path"]), "kind": asset["kind"]}
        for asset in approved["contamination"]["assets"]])
    if errors:
        raise CorpusError("invalid language Pilot contamination assets: " + "; ".join(errors))


def tokenized_extra(config, *, split, fingerprint, seq_len, selection_key=None):
    extra = {"split": split, "tokenizer": fingerprint, "seq_len": seq_len,
             "text_policy": config["text_policy"]}
    if is_language(config):
        extra["train_selection"] = selection_key
    return extra


def split_text_path(directory, name, *, balanced):
    if balanced and name == "train":
        return directory / SELECTED_TRAIN
    return directory / f"{name}.text.jsonl"


def _supervised_through(position, seq_len):
    """Untruncated supervision in [0, position), excluding each window start."""
    return position - (position + seq_len - 1) // seq_len


def select_train(directory, config, tokenizer):
    """Measure full Train supply and choose one seeded prefix per category.

    Each document remains whole (including EOS). Adjacent category blocks
    allow exact accounting as each block is assembled. The last block extends
    to a complete packing window. The final budget independently checks the
    resulting mixture and every category's actual supervised minimum.
    """
    from .tokenizer_v4 import guard_train_only

    path = directory / "train.text.jsonl"
    split_manifest = directory / "split-manifest.json"
    guard_train_only(path, split_manifest)
    targets = config["targets"]
    ratios = targets["category_token_ratios"]
    seq_len = targets["seq_len"]
    seed = config["seed"]
    groups = {category: [] for category in V4_CATEGORIES}
    ids = set()
    full_order = []
    for row in read_records(path):
        doc_id = row["document_id"]
        if row["split"] != "train" or doc_id in ids:
            raise CorpusError("Train selection requires unique Train-only documents")
        ids.add(doc_id)
        tokens = len(tokenizer.encode_document(row["text"], category=row["language"]))
        groups[row["category"]].append((identity([seed, doc_id]), tokens, row))
        full_order.append((row["category"], tokens))
    selected = []
    index = []
    cursor = 0
    required = [category for category in groups if ratios[category] > 0]
    quotas = {c: (ceil(targets["effective_train_tokens"] * Fraction(str(r)))
                  if config["corpus_version"] == SCALING_VERSION else round(targets["effective_train_tokens"] * r))
              for c, r in ratios.items()}
    for category in required:
        target = quotas[category]
        start = _supervised_through(cursor, seq_len)
        cutoff = None
        for _, tokens, row in sorted(groups[category], key=lambda entry: (entry[0], entry[2]["document_id"])):
            cursor += tokens
            selected.append(row)
            index.append({key: row[key] for key in ("document_id", "category", "language", "source_id")}
                         | {"tokens": tokens})
            if cutoff is None and _supervised_through(cursor, seq_len) - start >= target:
                cutoff = (cursor + seq_len - 1) // seq_len * seq_len + 1
            if cutoff is not None and (category != required[-1] or cursor >= cutoff):
                break
    packing = supervision_by_category(((r["category"], r["tokens"]) for r in index), seq_len)
    supply_packing = supervision_by_category(full_order, seq_len)
    supply = {category: {"documents": len(rows), "raw_tokens": sum(row[1] for row in rows),
                         "effective_tokens_in_full_train": supply_packing["by_category"][category]}
              for category, rows in groups.items()}
    report = {"algorithm": "seeded-category-prefix-v1", "seed": seed, "seq_len": seq_len,
              "tokenizer_fingerprint": tokenizer.fingerprint, "train_input": file_info(path),
              "split_manifest": file_info(split_manifest), "supply": supply,
              "targets": quotas,
              "selected_documents": index, "packing": packing,
              "selected_raw_tokens_by_category": {
                  c: sum(row["tokens"] for row in index if row["category"] == c) for c in ratios},
              "minimum_packed_effective_tokens": (
                  (targets["effective_train_tokens"] + seq_len - 2) // (seq_len - 1) * (seq_len - 1)),
              "fraction_tolerance": LANGUAGE_TOLERANCE}
    return selected, report


def verify_train_selection(directory, config, tokenizer, *, verify_bin=False):
    from hashlib import sha256

    from .corpus_utils import json_bytes

    rows, expected = select_train(directory, config, tokenizer)
    digest = sha256()
    size = 0
    for row in rows:
        data = json_bytes(row)
        digest.update(data)
        size += len(data)
    actual = json.loads((directory / SELECTION_REPORT).read_text(encoding="utf-8"))
    selected_file = {"sha256": digest.hexdigest(), "bytes": size}
    if (actual != {**expected, "selected_file": selected_file}
            or file_info(directory / SELECTED_TRAIN) != selected_file
            or read_token_index([directory / "train.token-index.jsonl"])["records"] != expected["selected_documents"]
            or (directory / "train.bin").stat().st_size != 2 * expected["packing"]["raw_tokens"]):
        raise CorpusError("balanced Train selection, Token Index or BIN differs from deterministic evidence")
    if verify_bin:
        import numpy as np

        with (directory / "train.bin").open("rb") as handle:
            for row in rows:
                tokens = tokenizer.encode_document(row["text"], category=row["language"])
                data = np.asarray(tokens, dtype="<u2").tobytes()
                if handle.read(len(data)) != data:
                    raise CorpusError("balanced Train BIN differs from selected document encoding")
    return actual


def audit_receipt(directory, plan, report):
    names = ("audit-report.json", "budget-report.json", "contamination-report.json", SELECTION_REPORT,
             "tokenization-report.json", "split-manifest.json", "train.bin", "val.bin", "test.bin")
    return {"plan_key": plan["plan_key"], "status": report["status"],
            "files": {name: file_info(directory / name) for name in names}}


def verify_audit_receipt(directory, plan):
    report = json.loads((directory / "audit-report.json").read_text(encoding="utf-8"))
    saved = json.loads((directory / AUDIT_RECEIPT).read_text(encoding="utf-8"))
    if saved != audit_receipt(directory, plan, report):
        raise CorpusError("language Pilot audit differs from its generation evidence")
    return report


def verify_language_release(directory, plan, manifest):
    """Independently enforce the language-specific release gates."""
    from .corpus_budget import plan as plan_budget
    from .corpus_contamination import audit as contamination_audit
    from .tokenizer import MiniLLMTokenizer
    from .v4_tokenizer_decision import verify_selection

    config = plan["generation_config"]
    if (config["corpus_version"] != manifest["corpus_version"]
            or config["dataset_name"] != manifest["dataset_name"]
            or manifest["identity"]["v4"]["corpus_version"] != config["corpus_version"]
            or not is_language(config) or plan["demo"] is not False):
        raise CorpusError("language Pilot release version or plan mismatch")
    validate_language_policy(config)
    if config["corpus_version"] == SCALING_VERSION:
        from .v4_language_scaling import verify_pinned_release
        verify_pinned_release(directory)
    selected = verify_selection(directory)
    tokenizer = MiniLLMTokenizer.load(directory / "tokenizer.json")
    if tokenizer.fingerprint != selected["fingerprint"]:
        raise CorpusError("language Pilot tokenizer differs from reviewed candidate")
    required_categories = {c for c, weight in config["targets"]["category_token_ratios"].items() if weight}
    if required_categories - {source["category"] for source in manifest["license_sources"]["sources"]}:
        raise CorpusError("language Pilot lacks approved sources for a positive category")
    evidence = verify_train_selection(directory, config, tokenizer, verify_bin=True)
    stages = manifest["identity"]["stage_hashes"]

    def key(name, extra):
        return identity([4, plan["plan_key"], name, extra])

    split = stages["split"]["key"]
    expected_selection = key("train_selection", {"split": split, "tokenizer": tokenizer.fingerprint})
    seq_len = config["targets"]["seq_len"]
    expected_tokenized = key("tokenized", tokenized_extra(
        config, split=split, fingerprint=tokenizer.fingerprint, seq_len=seq_len,
        selection_key=expected_selection))
    if (stages["train_selection"]["key"] != expected_selection
            or stages["tokenized"]["key"] != expected_tokenized
            or any(stages["train_selection"]["files"].get(name) != manifest["files"][name]
                   for name in (SELECTION_REPORT, SELECTED_TRAIN))):
        raise CorpusError("language Pilot selection and Tokenize stage binding differs")
    packing = evidence["packing"]
    ratios = config["targets"]["category_token_ratios"]
    available = {c: {"documents": packing["documents_by_category"][c],
                     "effective_tokens": packing["by_category"][c]} for c in ratios}
    budget = plan_budget(available, config["targets"]["effective_train_tokens"], ratios,
                         seq_len=seq_len, tokenizer_fingerprint=tokenizer.fingerprint,
                         fraction_tolerance=LANGUAGE_TOLERANCE)
    recorded_budget = json.loads((directory / "budget-report.json").read_text(encoding="utf-8"))
    if any((directory / f"{name}.bin").stat().st_size < 2 * (seq_len + 1) for name in ("val", "test")):
        raise CorpusError("language Pilot requires a complete packing window in Val and Test")
    if (budget["status"] != "READY_FOR_TRAINING"
            or budget["content_id"] != recorded_budget["content_id"]
            or budget["categories"] != recorded_budget["categories"]
            or manifest["budget"]["target_effective_train_tokens"] != budget["target_effective_train_tokens"]
            or manifest["effective_train_tokens"] != packing["effective_tokens"]
            or manifest["category_token_distribution"]["effective"] != packing["by_category"]):
        raise CorpusError("language Pilot effective output or category mix fails the version budget")
    root = PROJECT_ROOT
    policy = config["contamination"]
    contamination = json.loads((directory / "contamination-report.json").read_text(encoding="utf-8"))
    fresh = contamination_audit(
        [{**asset, "path": str((root / asset["path"]).resolve())} for asset in policy["assets"]],
        read_records(directory / "train.text.jsonl"),
        decisions=root / policy["decisions"] if policy["decisions"] else None,
        min_prompt_chars=policy["min_prompt_chars"], max_hits=policy["max_hits"],
        long_span_containment=policy["long_span_containment"])
    if (fresh != {k: v for k, v in contamination.items() if k != "demo_only"} or contamination["errors"]
            or contamination["stats"]["truncated"] or contamination["status"] != manifest["contamination_audit_status"]
            or contamination["status"] not in ("CHECKED_NO_HITS", "REVIEWED")
            or (contamination["status"] == "CHECKED_NO_HITS" and contamination["stats"]["high_risk_hits"] != 0)
            or (contamination["status"] == "REVIEWED"
                and contamination["decisions"]["unreviewed_high_risk"] != 0)):
        raise CorpusError("language Pilot requires complete cleared contamination evidence")
    audit = verify_audit_receipt(directory, plan)
    required = {"license_review", "cleaning_policy", "dedup_policy", "split_isolation",
                "tokenizer_selection", "balanced_train_selection", "bin_reproducibility",
                "token_budget", "contamination_audit", "manifest_completeness"}
    if (audit["status"] != "READY" or audit["demo_only"] is not False or audit["blockers"]
            or not required <= audit["checks"].keys()
            or any(check["status"] != "PASS" for check in audit["checks"].values())
            or audit["checks"]["manifest_completeness"]["detail"]["content_id"] != manifest["content_id"]):
        raise CorpusError("language Pilot requires passing Audit bound to the release identity")
