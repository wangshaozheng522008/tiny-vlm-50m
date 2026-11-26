"""Pinned artifacts and source scope for the independent 2M scaling profile."""

from pathlib import Path

from .assessment_io import read_json, records
from .corpus_utils import CorpusError, file_info, identity, verify_file

ROOT = Path(__file__).resolve().parents[2]
VERSION = "minillm-corpus-v4-language-pilot-2m-v1"
CORPUS_CONFIG = ROOT / "configs/corpus_v4_language_pilot_2m.yaml"
TRAINING_CONFIG = ROOT / "configs/pilot-v4-language-2m-t1200.json"
BASE = ROOT / "artifacts/corpus/v4-language-pilot-first-training-reviewed"
INITIAL = ROOT / "artifacts/v4-language-pilot-first-training/workspace/initial.pt"
HISTORICAL = ROOT / "runs/v4-language-pilot-first-training/model.pt"
OUTPUT = ROOT / "artifacts/corpus" / VERSION
WORKSPACE = ROOT / "artifacts/v4-language-pilot-2m-v1/workspace"
REPORT = ROOT / "reports/v4-language-pilot-2m-v1"
BASE_ID = "7222651ef09e4e09c2d613474f50712e8233d1d4ff9017bda897e597ebdb6da5"
BASE_IMPLEMENTATION = "15b8beb04ed88b91f96d70b59df72022dc995fea9bb50b0a8eda195e5359255d"
BASE_MANIFEST = {"bytes": 40768, "sha256": "6160624ea9962eff0d07a4aa976f8b82e9a7cfc36ba84773c9dbd4d55a7791c7"}
FINGERPRINT = "845a24b9f21458b1d2161c60dfc8a77d5ed40235ece7ef46891677e423727b8a"
INITIAL_FILE = {"bytes": 346160987, "sha256": "bfdd11421cb247d5b8c88b7b268f8f4353e0bc12c5fcfae49085929799cc3e99"}
HISTORICAL_FILE = {"bytes": 409075993, "sha256": "aff866a2bf1999356e83aad21b6dc1e7145ef9931c9712ea6249d9def19d5840"}
TARGET = 2_005_080
SPLIT_FILES = ("train.text.jsonl", "val.text.jsonl", "test.text.jsonl", "split-assignment.jsonl", "split-manifest.json")
EVAL_FILES = tuple(f"{split}.{suffix}" for split in ("val", "test") for suffix in ("bin", "token-index.jsonl"))


def base_manifest(*names):
    """Read the pinned historical receipt and verify requested immutable inputs."""
    verify_file(BASE / "dataset-manifest.json", BASE_MANIFEST)
    manifest = read_json(BASE / "dataset-manifest.json")
    if (manifest["content_id"] != BASE_ID or identity(manifest["identity"]) != BASE_ID
            or manifest["identity"]["files"] != manifest["files"]
            or manifest["freeze_status"] != "READY" or manifest["tokenizer_fingerprint"] != FINGERPRINT
            or read_json(BASE / "FROZEN") != {"content_id": BASE_ID}):
        raise CorpusError("historical frozen corpus identity differs")
    for name in names:
        verify_file(BASE / name, manifest["files"][name])
    return manifest


def historical_entries(manifest):
    return {source["source_id"]: source["source_review"]["entry"] for source in manifest["license_sources"]["sources"]}


def validate_scope(entry):
    """Require a new human decision for both this corpus and its supervised budget."""
    from .v4_sources import _review_text

    scope = entry.get("training_scope")
    fields = {"corpus_version": VERSION, "base_content_id": BASE_ID, "effective_train_tokens": TARGET}
    if (entry.get("corpus_versions") != [VERSION] or not isinstance(scope, dict)
            or any(scope.get(key) != value for key, value in fields.items())):
        raise CorpusError("new human 2M training scope approval required")
    _review_text(scope.get("confirmation"), "2M human confirmation")
    reviewer = _review_text(scope.get("reviewer"), "2M reviewer")
    from datetime import date, datetime
    try:
        reviewed = date.fromisoformat(_review_text(scope.get("reviewed_at"), "2M review date"))
        source = entry["v4_source"]
        licensed = datetime.fromisoformat(_review_text(source.get("license_reviewed_at"), "2M source review date")
                                           .replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise CorpusError("2M review dates must be valid ISO dates") from exc
    if (source.get("license_status") != "APPROVED" or reviewed != licensed
            or reviewer != _review_text(source.get("license_reviewer"), "2M source reviewer")):
        raise CorpusError("2M requires a matching fresh source approval decision")


def validate_source_set(registry):
    previous = historical_entries(base_manifest())
    current = {row["source_id"]: row["source_review"]["entry"] for row in registry["sources"]}
    if current.keys() != previous.keys():
        raise CorpusError("2M source set differs from the pinned historical inventory")
    for name, entry in current.items():
        validate_scope(entry)
        old = previous[name]
        if (entry["local_inventory"]["files"] != old["local_inventory"]["files"]
                or entry["release"] != old["release"]
                or entry["v4_source"].get("priority", 0) != old["v4_source"].get("priority", 0)
                or any(entry["v4_source"][field] != old["v4_source"][field]
                       for field in ("category", "language", "family_policy", "transformation_history"))):
            raise CorpusError(f"2M source snapshot or split policy differs for {name}")


def verify_split_reuse(path, generated, manifest):
    """Compare every freshly processed document before retaining historical bytes."""
    def content(row):
        # The new review lives in the new registry. Historical records keep their
        # byte-exact split and per-document provenance for paired evaluation.
        fields = {k: v for k, v in row.items() if k not in ("license_evidence", "collection_version")}
        fields["transformation_history"] = [entry for entry in row["transformation_history"]
                                             if not (isinstance(entry, dict) and entry.get("kind") == "source_review")]
        return fields

    verify_file(BASE / path, manifest["files"][path])
    for old, new in zip(records(BASE / path), records(generated), strict=True):
        if content(old) != content(new):
            raise CorpusError(f"fresh {path} differs from the historical document at {new['document_id']}")


def verify_pinned_release(directory, *, encoded=True):
    names = SPLIT_FILES + ("tokenizer.json",) + EVAL_FILES if encoded else SPLIT_FILES
    manifest = base_manifest(*names)
    for name in names:
        verify_file(directory / name, manifest["files"][name])
    return manifest


def tokenizer_receipt(directory):
    from .v4_tokenizer_decision import verify_selection
    verify_pinned_release(directory, encoded=False)
    original = base_manifest("tokenizer.json", "tokenizer-selection.json", "tokenizer-evaluation.json",
                             "tokenizer-candidates.json")
    receipt = verify_selection(BASE)
    if receipt["fingerprint"] != FINGERPRINT or receipt["vocab_size"] != 24576:
        raise CorpusError("2M requires the original human-reviewed 24,576 tokenizer")
    return {**receipt, "path": str(BASE / "tokenizer.json"), "reused_from_content_id": BASE_ID,
            "historical_selection": original["files"]["tokenizer-selection.json"]}


def verify_tokenizer_reuse(directory):
    receipt = tokenizer_receipt(directory)
    plan = read_json(directory / ".v4/plan.json")
    stage = read_json(directory / ".v4/tokenizer_selection.stage.json")
    split = read_json(directory / ".v4/split.stage.json")["key"]
    expected = identity([4, plan["plan_key"], "tokenizer_selection", {"split": split, "base_content_id": BASE_ID}])
    if (stage["key"] != expected or read_json(directory / "tokenizer-selection.json") != receipt
            or stage["files"].get(str((directory / "tokenizer-selection.json").resolve()))
            != file_info(directory / "tokenizer-selection.json")):
        raise CorpusError("2M tokenizer reuse receipt differs from the approved decision")
    for name in ("tokenizer-evaluation.json", "tokenizer-candidates.json"):
        if (file_info(directory / name) != file_info(BASE / name)
                or stage["files"].get(str((directory / name).resolve())) != file_info(BASE / name)):
            raise CorpusError("2M historical tokenizer evidence changed")
    return receipt


def validate_training_config(raw, workspace, evaluation=None):
    from dataclasses import asdict
    from .evaluation import EvaluationConfig

    expected = read_json(TRAINING_CONFIG)
    compared = {k: v for k, v in raw.items() if k not in ("resume", "workload_report", "lineage_report")}
    if compared != expected or (Path(compared["out"]).resolve() != ROOT / expected["out"]
                                or Path(workspace).resolve() != WORKSPACE):
        raise ValueError("2M requires the pinned independent training configuration and workspace")
    if raw.get("resume"):
        resume = Path(raw["resume"]).resolve()
        if resume not in {ROOT / expected["out"] / name for name in ("state.pt", "state.prev.pt")}:
            raise ValueError("resume requires this 2M run's state.pt or state.prev.pt")
    if evaluation is not None and asdict(evaluation) != asdict(EvaluationConfig(**read_json(
            ROOT / "configs/evaluation.v4-language-pilot.json"))):
        raise ValueError("2M requires the pinned formal evaluation configuration")


def validate_launch_config(raw):
    init = raw.get("init_from")
    validate_training_config({k: v for k, v in raw.items() if k != "init_from"}, WORKSPACE)
    if raw.get("resume"):
        if init:
            raise ValueError("2M resume cannot also initialize new weights")
    elif not init or Path(init).resolve() != WORKSPACE / "initial.pt":
        raise ValueError("2M launch requires the independently created workspace with the pinned Step 0")
    else:
        verify_file(Path(init), INITIAL_FILE)


def sampler_preview(index, seq_len, window_count, ratios, opts):
    from fractions import Fraction
    from itertools import islice
    from .data import EpochBatchSampler
    from .v4_training_accounting import category_consumption

    sampler = EpochBatchSampler(window_count, opts.batch_size, opts.seed)
    windows = [window for batch in islice(sampler.batches(0), opts.max_steps * opts.grad_accum) for window in batch]
    result = category_consumption(index, seq_len, windows)
    if result["effective_tokens"] != TARGET or len(windows) != 1960:
        raise ValueError("2M sampler cannot supply exactly 1,960 independent windows")
    for category, share in ratios.items():
        observed = result["categories"][category]["consumed_effective_tokens"]
        if abs(Fraction(observed, TARGET) - Fraction(str(share))) > Fraction("0.01") or (not share and observed):
            raise ValueError(f"2M sampler is outside the pinned category tolerance for {category}")
    return result


def preview_consumption(dataset, data, opts):
    from .corpus_budget import read_token_index

    index = read_token_index([Path(dataset) / "train.token-index.jsonl"])["records"]
    budget = read_json(Path(dataset) / "budget-report.json")
    return sampler_preview(index, data.seq_len, len(data), budget["category_token_ratios"], opts)


def prepare_request(out):
    """Verify the old inventory, write drafts only and stop before formal construction."""
    from copy import deepcopy
    from types import SimpleNamespace
    from .assessment_io import publish_manifest, report_directory, environment
    from .corpus_v4 import load_config
    from .language_diagnostics import BENCHMARK, CONTINUATIONS, FORMAL_CONFIG, PROBES
    from .tokenizer import MiniLLMTokenizer
    from .train import lr_at
    from .training_manifest import write_json
    from .v4_language_pilot import LANGUAGE_VERSION, select_train
    from .v4_pilot_sampling import SAMPLE_KIND, load_sample_manifest, verify_sample_binding
    from .v4_sources import validate_reviewed_source
    import torch

    names = (*SPLIT_FILES, *EVAL_FILES, "tokenizer.json", "train.bin", "train.token-index.jsonl", "budget-report.json")
    manifest = base_manifest(*names)
    verify_file(INITIAL, INITIAL_FILE)
    verify_file(HISTORICAL, HISTORICAL_FILE)
    draft, sources = [], []
    for source in manifest["license_sources"]["sources"]:
        validate_reviewed_source(source, corpus_version=LANGUAGE_VERSION)
        original = source["source_review"]["entry"]
        for path, info in original["local_inventory"]["files"].items():
            verify_file(Path(path), info)
        samples = []
        for ref in original["v4_source"]["transformation_history"]:
            if ref["kind"] == SAMPLE_KIND:
                path = Path(ref["manifest"])
                sample = load_sample_manifest(path)
                verify_sample_binding(path, sample, original, LANGUAGE_VERSION)
                samples.append({"manifest": file_info(path), "candidate_snapshot": sample["inputs"],
                                "sample_output": sample["output"], "scope_of_local_check":
                                "reviewed sample payload and manifest hashes verified; raw source replay required at build"})
        if not samples:
            raise CorpusError(f"{source['source_id']}: no pinned sample lineage")
        old = deepcopy(original)
        old["corpus_versions"] = [VERSION]
        old["training_scope"] = {"corpus_version": VERSION, "base_content_id": BASE_ID,
                                 "effective_train_tokens": TARGET, "confirmation": None,
                                 "reviewer": None, "reviewed_at": None}
        old["v4_source"].update(license_status="REVIEW_REQUIRED", license_reviewer=None, license_reviewed_at=None)
        sources.append({"id": source["source_id"], "category": source["category"], "release": original["release"],
                        "existing_approval_versions": original["corpus_versions"],
                        "historical_uses": {k: original[k] for k in ("commercial_use", "redistribution", "model_training",
                                                                     "license_obligations")},
                        "samples": samples, "new_approval": "PENDING_HUMAN"})
        draft.append(old)
    config = load_config(CORPUS_CONFIG)
    tokenizer = MiniLLMTokenizer.load(BASE / "tokenizer.json")
    if tokenizer.fingerprint != FINGERPRINT:
        raise CorpusError("historical tokenizer identity mismatch")
    _, selection = select_train(BASE, config, tokenizer)
    training = read_json(TRAINING_CONFIG)
    packing = selection["packing"]
    sampler = sampler_preview(selection["selected_documents"], 1024, packing["windows"],
                              config["targets"]["category_token_ratios"], SimpleNamespace(**training))
    warmup = int(training["max_steps"] * training["warmup_ratio"])
    lr = [{"step": step + 1, "planned_lr": lr_at(step, training["max_steps"], training["lr"], training["min_lr"], warmup)}
          for step in range(training["max_steps"])]
    proof = {"status": "BLOCKED_SOURCE_SCOPE_APPROVAL", "corpus_version": VERSION, "approved_budget": 0,
             "training": "NOT_RUN", "freeze": "NOT_RUN", "evaluation": "NOT_RUN", "int4": "NOT_RUN",
             "historical_content_id": BASE_ID, "tokenizer_fingerprint": FINGERPRINT,
             "checkpoints": {"step0": INITIAL_FILE, "historical_step123": HISTORICAL_FILE},
             "fixed_files": {name: manifest["files"][name] for name in names},
             "environment": {"cuda_available": torch.cuda.is_available(),
                             "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None},
             "next": "Obtain independent human approval for each pinned source and run the build stage."}
    with report_directory(out) as directory:
        write_json(directory / "status.json", proof)
        write_json(directory / "source-approval-request.json", {"corpus_version": VERSION, "sources": sources})
        write_json(directory / "source-review.pending.json", {"schema_version": 1, "sources": draft})
        write_json(directory / "packing-preview.json", {"status": "PREVIEW_ONLY_NOT_FROZEN",
                   "packing": packing, "quotas": selection["targets"], "raw_full_supply": selection["supply"],
                   "selected_raw_tokens_by_category": selection["selected_raw_tokens_by_category"],
                   "selected_document_order_sha256": identity(selection["selected_documents"]), "sampler": sampler})
        write_json(directory / "training-plan.json", {"status": "NOT_RUN", "configuration": training,
                   "initialized_from": INITIAL_FILE, "planned_updates": lr, "warmup_steps": warmup})
        write_json(directory / "evaluation-plan.json", {"status": "NOT_RUN",
                   "common_train_data": manifest["files"]["train.bin"], "frozen_val_test": proof["fixed_files"],
                   "files": {str(p): file_info(p) for p in (ROOT / BENCHMARK, CONTINUATIONS, PROBES, FORMAL_CONFIG)}})
        publish_manifest(directory, "manifest.json", {"schema_version": 1, "status": proof["status"],
                         "environment": environment(torch.device("cpu"))})
    return proof
