"""Attribute a language Pilot's committed Loss targets to frozen categories."""
from bisect import bisect_right
from itertools import islice
from pathlib import Path

from .assessment_io import read_json
from .checkpoint import load_training_state
from .corpus_budget import read_token_index, supervision_by_category
from .corpus_utils import file_info, identity, verify_file
from .data import EpochBatchSampler, PackedPretrainDataset
from .train import _validate_progress
from .training_manifest import dataset_identity


def category_consumption(documents, seq_len, windows):
    """Count document overlaps with the label interval of each sampled window."""
    packing = supervision_by_category(
        ((row["category"], row["tokens"]) for row in documents), seq_len)
    ends = [0]
    for row in documents:
        ends.append(ends[-1] + row["tokens"])
    windows = list(windows)
    if len(windows) != len(set(windows)) or any(
            type(window) is not int or not 0 <= window < packing["windows"] for window in windows):
        raise ValueError("sampled windows must be unique complete Train windows")
    consumed = dict.fromkeys(packing["by_category"], 0)
    for window in windows:
        start, end = window * seq_len + 1, (window + 1) * seq_len
        doc = bisect_right(ends, start) - 1
        while start < end:
            stop = min(ends[doc + 1], end)
            consumed[documents[doc]["category"]] += stop - start
            start, doc = stop, doc + 1
    total = len(windows) * (seq_len - 1)
    if sum(consumed.values()) != total:
        raise ValueError("sampler and category counts disagree")
    mix = {}
    for category, frozen in packing["by_category"].items():
        frozen_share = frozen / packing["effective_tokens"] if packing["effective_tokens"] else None
        sampled_share = consumed[category] / total if total else None
        mix[category] = {"frozen_effective_tokens": frozen, "frozen_share": frozen_share,
                         "consumed_effective_tokens": consumed[category], "consumed_share": sampled_share,
                         "difference_percentage_points": (
                             100 * (sampled_share - frozen_share) if sampled_share is not None else None)}
    return {"effective_tokens": total, "consumed_windows": len(windows),
            "window_order_sha256": identity(windows), "frozen_packing": packing, "categories": mix}


def replay_trained_windows(dataset, run, summary, frozen_content_id):
    """Recover the exact committed independent Train windows in sampler order."""
    dataset, run = Path(dataset), Path(run)
    frozen = read_json(dataset / "dataset-manifest.json")
    if frozen["content_id"] != frozen_content_id:
        raise ValueError("frozen dataset identity changed after training")
    index = dataset / "train.token-index.jsonl"
    verify_file(index, frozen["files"][index.name])
    state = load_training_state(run / "state.pt", mmap=True)
    manifest = state["manifest"]
    if manifest != read_json(run / "training-manifest.json") or any(
            state[key] != summary[key] for key in ("global_step", "tokens_seen", "samples_seen", "epoch",
                                                   "micro_batches_consumed_in_epoch")):
        raise ValueError("training summary, manifest and committed checkpoint disagree")
    plan = manifest["training"]
    data = PackedPretrainDataset(dataset / "train.bin", manifest["dataset"]["train"]["seq_len"])
    data_identity, _ = dataset_identity(data, manifest["model_config"]["ignore_index"])
    if data_identity != manifest["dataset"]["train"] or data_identity["dataset_content_id"] != frozen_content_id:
        raise ValueError("training checkpoint and frozen Train data disagree")
    sampler = EpochBatchSampler(len(data), plan["batch_size"], plan["sampler_seed"])
    _validate_progress(state, manifest, data, sampler)
    epoch, cursor = state["epoch"], state["micro_batches_consumed_in_epoch"]
    if epoch > 1 or (epoch == 1 and cursor):
        raise ValueError("language Pilot must not repeat a Train window")
    batches = sampler.batches(0)
    if epoch == 0:
        batches = islice(batches, cursor)
    windows = [i for batch in batches for i in batch]
    return state, frozen, data, windows


def trained_category_consumption(dataset, run, summary, frozen_content_id):
    dataset, run = Path(dataset), Path(run)
    state, frozen, data, windows = replay_trained_windows(dataset, run, summary, frozen_content_id)
    index = dataset / "train.token-index.jsonl"
    result = category_consumption(read_token_index([index])["records"], data.seq_len, windows)
    if (result["effective_tokens"] != state["tokens_seen"]
            or result["consumed_windows"] != state["samples_seen"]
            or result["frozen_packing"]["effective_tokens"] != frozen["effective_train_tokens"]
            or result["frozen_packing"]["by_category"] != frozen["category_token_distribution"]["effective"]):
        raise ValueError("sampler accounting does not reproduce frozen and trained Loss totals")
    return {**result, "checkpoint": file_info(run / "state.pt"), "dataset_content_id": frozen_content_id,
            "index": frozen["files"][index.name], "sampler_seed": state["manifest"]["training"]["sampler_seed"]}
