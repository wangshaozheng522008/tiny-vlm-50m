"""Budget, epoch-boundary, continuation and memorization regressions for PR #21."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from minillm import v5_language_training as profile
from minillm import v5_repeat_training as training
from minillm.assessment_io import read_json, records, write_json
from minillm.checkpoint import CheckpointError
from minillm.corpus_budget import read_token_index
from minillm.data import EpochBatchSampler
from minillm.training_milestones import reconcile_recovery_log
from minillm.v5_repeat_diagnostics import generation_diagnostics, train_overlap
from minillm.v5_repeat_sampling import RepeatPlan, accounting


def test_independent_scope_and_stage_scheduler(monkeypatch, tmp_path):
    settings = training._settings()
    evidence = training._authorization(settings)["evidence"]
    assert evidence["cumulative_effective_token_cap"] == 20_001_696
    assert evidence["purpose"] == "internal_research"
    assert training.stage_lr(0, settings) == pytest.approx(0.0000075)
    assert training.stage_lr(19, settings) == training.stage_lr(20, settings) == settings["lr"]
    assert training.stage_lr(1221, settings) == settings["min_lr"]
    assert [step * training.TOKENS_PER_UPDATE for step in training.MILESTONES] == [
        12_005_928, 14_002_824, 15_999_720, 18_004_800, 20_001_696,
    ]
    replacement = tmp_path / "approval.json"
    monkeypatch.setattr(training, "AUTHORIZATION", replacement)
    write_json(replacement, {**evidence, "cumulative_effective_token_cap": 10_000_848})
    with pytest.raises(ValueError, match="outside the requested scope"):
        training._authorization(settings)


def test_cross_epoch_update_preserves_eight_windows():
    sampler = EpochBatchSampler(9789, 1, seed=42)
    plan = RepeatPlan(sampler, 9776, 9776)
    windows = plan.stage_order()
    assert len(windows) == 9776
    assert windows[:8] == sampler.indices(0)[9776:9784]
    assert windows[8:16] == sampler.indices(0)[9784:] + sampler.indices(1)[:3]
    assert plan.cursor(8) == (0, 9784)
    assert plan.cursor(16) == (1, 3)
    assert plan.cursor(len(windows)) == (1, 9763)
    with pytest.raises(ValueError, match="bounded second epoch"):
        RepeatPlan(sampler, 9776, 9803)


def test_replay_rejects_duplicates_in_same_epoch(monkeypatch):
    sampler = EpochBatchSampler(10, 1, 42)
    original = sampler.indices
    monkeypatch.setattr(sampler, "indices", lambda epoch: original(epoch) if epoch == 0 else [0] * 10)
    plan = RepeatPlan(sampler, 8, 8)
    with pytest.raises(ValueError, match="repeated or omitted"):
        plan.stage_order()


def test_actual_frozen_sampler_replays_source_caps():
    if not (profile.DATASET / "FROZEN").exists():
        pytest.skip("requires the local frozen V5 release")
    sampler = EpochBatchSampler(9789, 1, 42)
    replay = RepeatPlan(sampler, 9776, 9776)
    plan = read_json(profile.DATASET / ".v4/plan.json")
    documents = read_token_index([profile.DATASET / "train.token-index.jsonl"])["records"]
    proof = accounting(replay, documents, plan, 1024, 9776, enforce=True)
    stage, cumulative = proof["stage"], proof["cumulative"]
    assert stage["effective_tokens"] == 10_000_848
    assert stage["new_effective_tokens"] == 13_299
    assert stage["previously_consumed_effective_tokens"] == 9_987_549
    assert cumulative["effective_tokens"] == 20_001_696
    assert cumulative["unique_effective_tokens"] == 10_014_147
    assert cumulative["repeated_effective_tokens"] == 9_987_549
    assert all(period["category_mix_ok"] and period["source_caps_ok"] for period in (stage, cumulative))


def test_restore_progress_and_epoch_scheduler_after_crossing():
    settings = training._settings()
    replay = RepeatPlan(EpochBatchSampler(9789, 1, 42), 9776, 9776)
    manifest = {"run_id": "repeat"}
    history = [{"step": 1222, "loss": 6.0}]
    state = {"manifest": manifest, "global_step": 1224, "samples_seen": 9792, "tokens_seen": 10_017_216,
             "epoch": 1, "micro_batches_consumed_in_epoch": 3, "history": history,
             "best_loss": 6.0, "best_checkpoint": {"step": 1222, "tokens_seen": 10_000_848, "val_loss": 6.0},
             "optimizer": {"param_groups": [{"lr": training.stage_lr(1, settings)}]},
             "continuation": {"stage_step": 2, "stage_effective_tokens": 16_368, "next_lr_index": 2,
                              "overfit_streak": 0, "seconds": 1., "peak_gpu_mib": 3.}}
    assert training.verify_progress(state, manifest, replay, settings) == 2
    for field, corrupted in (
        ("global_step", 1225), ("epoch", 0), ("samples_seen", 9789),
        ("micro_batches_consumed_in_epoch", 4), ("tokens_seen", 10_017_215),
    ):
        with pytest.raises(CheckpointError):
            training.verify_progress({**state, field: corrupted}, manifest, replay, settings)
    for name in ("stage_step", "next_lr_index", "stage_effective_tokens", "overfit_streak"):
        tampered = deepcopy(state)
        tampered["continuation"][name] += 1
        with pytest.raises(CheckpointError):
            training.verify_progress(tampered, manifest, replay, settings)
    validate = lambda losses: training.overfit_streak([{"loss": loss} for loss in losses])
    assert validate([5.0, 5.17, 5.21, 5.3]) == 3
    assert validate([5.0, 5.17, 5.21, 5.14, 5.3]) == 1
    assert validate([5.0, 5.34, 4.9, 5.3]) == 1


def test_backup_rollback_to_stage_zero_archives_uncommitted_updates(tmp_path):
    checkpoint = tmp_path / "state.prev.pt"
    torch.save({}, checkpoint)
    history = [{"step": 1222, "loss": 6.0}]
    state = {"global_step": 1222, "history": history, "manifest": {"run_id": "repeat"}}
    path = tmp_path / "training-log.jsonl"
    reconcile_recovery_log(tmp_path, state, checkpoint, initial_step=1222)
    path.write_text('{"event":"train","global_step":1223}\n{"event":', encoding="utf-8")
    reconcile_recovery_log(tmp_path, state, checkpoint, initial_step=1222)
    assert list(records(path)) == []
    evidence, = (tmp_path / "recovery").glob("*/manifest.json")
    assert read_json(evidence)["recovered_step"] == 1222
    assert read_json(tmp_path / "metrics.json") == history


def test_exact_overlap_respects_document_boundaries_and_stop_tokens():
    words = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint16)
    docs = [{"document_id": "A", "tokens": 5}, {"document_id": "B", "tokens": 5}]
    generated = [{"generated_token_ids": [100, 2, 3, 4, 5], "content_tokens": 4},
                 {"generated_token_ids": [4, 5, 6, 7], "content_tokens": 4},
                 {"generated_token_ids": [6, 7, 8, 9, 10, 0], "content_tokens": 5}]
    assert train_overlap(words, docs, generated, span=3) == [
        {"matched_tokens": 3, "document_id": "A", "document_offset": 1, "output_offset": 1},
        None,
        {"matched_tokens": 5, "document_id": "B", "document_offset": 0, "output_offset": 0},
    ]


def test_generation_length_and_identical_punctuation_runs():
    data = [{"content_tokens": 4, "decoded_text": "，，，，",
             "diagnostics": {"nonspace_characters": 4, "sentence_repeat_ratio": 0}},
            {"content_tokens": 2, "decoded_text": "，，  ，",
             "diagnostics": {"nonspace_characters": 3, "sentence_repeat_ratio": .2}}]
    result = generation_diagnostics(data)
    assert result["identical_punctuation_run_at_least_four"] == 1
    assert result["content_token_length"]["median"] == 3
    assert result["nonspace_character_length"]["p95_nearest_rank"] == 4


def test_log_replays_cursor_only_prefix_and_checks_explicit_windows():
    settings = training._settings()
    replay = RepeatPlan(EpochBatchSampler(9789, 1, 42), 9776, 9776)
    order = replay.stage_order()
    rows = []
    for step in (1, 2):
        epoch, cursor = replay.cursor(step * 8)
        rows.append({"stage_step": step, "global_step": 1222 + step, "epoch": epoch,
                     "micro_batches_consumed_in_epoch": cursor, "learning_rate": training.stage_lr(step - 1, settings),
                     "tokens_seen": (1222 + step) * 8184, "loss_tokens": 8184})
    rows[1]["window_indices"] = order[8:16]
    assert training.verify_update_log(rows, settings, replay, 2) == {
        "logged_updates": 1, "replayed_from_cursor_updates": 1,
    }
    rows[1]["window_indices"] = order[:8]
    with pytest.raises(CheckpointError, match="windows differ"):
        training.verify_update_log(rows, settings, replay, 2)
    with pytest.raises(CheckpointError, match="in order"):
        training.verify_update_log(rows[::-1], settings, replay, 2)
