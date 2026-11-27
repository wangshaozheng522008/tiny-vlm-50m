"""Contracts for the independent 10M training and evaluation profile."""

from dataclasses import asdict, replace

import pytest
import torch

from minillm import v5_language_training as v5
from minillm.assessment_io import read_json
from minillm.evaluation import EvaluationConfig
from minillm.training_milestones import require_identical_weights
from minillm.v4_training_pilot import validate_pilot_budget
from minillm.v5_language_evaluation import (FINAL, HISTORICAL_LABEL, INT4, SAMPLED,
                                              _verify_report, historical_2m)
from minillm.workload import parse_training_config


def test_pinned_budget_milestones_and_schedule():
    opts, extra = parse_training_config(v5.TRAINING_CONFIG)
    budget = validate_pilot_budget(9789, opts, extra, corpus_version=v5.V5_VERSION)
    assert budget["planned_effective_tokens"] == 10_000_848
    assert budget["independent_effective_tokens"] == 10_014_147
    assert int(opts.warmup_ratio * opts.max_steps) == 20
    assert [step * v5.V5_TOKENS_PER_UPDATE for step in opts.milestone_steps] == [
        0, 2_005_080, 4_001_976, 5_998_872, 8_003_952, 10_000_848]
    with pytest.raises(ValueError, match="9,789"):
        validate_pilot_budget(9776, opts, extra, corpus_version=v5.V5_VERSION)
    with pytest.raises(ValueError, match="fixed"):
        validate_pilot_budget(9789, replace(opts, milestone_steps=()), extra, corpus_version=v5.V5_VERSION)
    with pytest.raises(ValueError, match="no corpus repetition"):
        validate_pilot_budget(9789, opts, {**extra, "allow_data_repetition": True}, corpus_version=v5.V5_VERSION)


def test_profile_protects_the_frozen_dataset_and_independent_resume(monkeypatch, tmp_path):
    from minillm.assessment_io import write_json

    monkeypatch.setattr(v5, "DATASET", tmp_path)
    manifest = tmp_path / "dataset-manifest.json"
    write_json(manifest, {"corpus_version": v5.V5_VERSION, "content_id": v5.CONTENT_ID})
    seen = []
    monkeypatch.setattr(v5, "verify_file", lambda path, info: seen.append((path, info)))
    raw = read_json(v5.TRAINING_CONFIG)
    v5.validate_profile(tmp_path, raw, v5.WORKSPACE)
    assert seen == [(v5.INITIAL, v5.INITIAL_FILE)]
    for name in ("state.pt", "state.prev.pt"):
        v5.validate_launch_config(tmp_path, {**raw, "resume": str(v5.RUN / name)})
    with pytest.raises(ValueError, match="only its own"):
        v5.validate_launch_config(tmp_path, {**raw, "resume": "runs/v4-language-pilot-2m-v1/state.pt"})
    with pytest.raises(ValueError, match="pinned PR #16"):
        v5.validate_launch_config(tmp_path, {**raw, "init_from": "runs/v4-language-pilot-2m-v1/model.pt"})
    with pytest.raises(ValueError, match="pinned independent"):
        v5.validate_profile(tmp_path, {**raw, "warmup_ratio": .02}, v5.WORKSPACE)
    write_json(manifest, {"corpus_version": v5.V5_VERSION, "content_id": "changed"})
    with pytest.raises(ValueError, match="pinned formal"):
        v5.validate_profile(tmp_path, raw, v5.WORKSPACE)


def test_frozen_actual_sampler_agrees_with_release():
    if not (v5.DATASET / "FROZEN").exists():
        pytest.skip("requires the local, approved V5 release")
    budget = v5.preview(v5.DATASET)
    assert budget["consumption"]["consumed_windows"] == 9776
    assert all(row["within_cap"] for row in budget["sources"].values())


def test_historical_2m_identity_and_cohorts():
    from minillm.v5_language_evaluation import HISTORICAL_REPORT

    if not (HISTORICAL_REPORT / "manifest.json").is_file():
        pytest.skip("requires the local verified PR #18 evaluation and checkpoint")
    config = EvaluationConfig(**read_json(v5.EVALUATION_CONFIG))
    policy = read_json(HISTORICAL_REPORT / "comparison-policy.json")
    assert EvaluationConfig(**policy["config"]) == config
    rows, receipt = historical_2m(config, policy["assets"])
    assert len({row["id"] for row in rows}) == len(rows) == 300
    assert receipt["generation"] == asdict(config)
    assert SAMPLED == {HISTORICAL_LABEL, "step245", "step489", FINAL, INT4}
    with pytest.raises(ValueError, match="fixed protocol"):
        historical_2m(replace(config, max_new_tokens=95), policy["assets"])


def test_evaluation_artifact_tampering_and_weight_mismatch(tmp_path):
    from minillm.assessment_io import publish_manifest
    from minillm.checkpoint import CheckpointError
    from minillm.corpus_utils import CorpusError

    file = tmp_path / "predictions.jsonl"
    file.write_text("{}\n")
    publish_manifest(tmp_path, "manifest.json", {"status": "EVALUATED"})
    assert _verify_report(tmp_path)["status"] == "EVALUATED"
    file.write_text('{"changed":true}\n')
    with pytest.raises(CorpusError, match="predictions"):
        _verify_report(tmp_path)
    state = {"weight": torch.ones(1)}
    require_identical_weights(state, state, FINAL)
    with pytest.raises(CheckpointError, match="weight"):
        require_identical_weights(state, {"weight": torch.zeros(1)}, FINAL)


def test_launcher_blocks_v5_before_loading_a_model_on_failed_admission(tmp_path, monkeypatch):
    from minillm import v4_admission
    from minillm.assessment_io import write_json
    from minillm.pilot_launcher import launch_pilot

    write_json(tmp_path / "dataset-manifest.json", {"corpus_version": v5.V5_VERSION})
    monkeypatch.setattr(v4_admission, "validate_v4_pilot_policy",
                        lambda path: {"ok": False, "errors": ["approval evidence changed"]})
    with pytest.raises(ValueError, match="approval evidence changed"):
        launch_pilot(tmp_path, tmp_path / "tokenizer.json", "minillm-87m",
                     v5.TRAINING_CONFIG, tmp_path / "launch", execute=True)
    assert not (tmp_path / "launch").exists()


def test_final_heldout_measurement_is_bound_and_reused(tmp_path, monkeypatch):
    from minillm import v5_language_evaluation as evaluation

    monkeypatch.setattr(evaluation, "_all_windows", lambda *args: (None, None, {"windows": 1}))
    calls = []

    def measure(*args):
        calls.append(args)
        return {"loss": 3.0}

    monkeypatch.setattr(evaluation, "evaluate_indexed_windows", measure)
    context = {"config": EvaluationConfig(), "device": "cpu"}
    receipt = {"checkpoint": {"sha256": "example"}, "node": evaluation.FINAL}
    output = tmp_path / "heldout"
    first = evaluation._final_test(output, receipt, context, model=object())
    assert evaluation._final_test(output, receipt, context) == first
    assert len(calls) == 1
    with pytest.raises(ValueError, match="differs"):
        evaluation._final_test(output, {**receipt, "checkpoint": {"sha256": "other"}}, context)
