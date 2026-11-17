import signal
from pathlib import Path
from unittest.mock import patch

import pytest

import minillm.train as training
from minillm.assessment_io import read_json
from minillm.corpus_utils import file_info
from minillm.recovery_state import compare_tree
from minillm.v4_training_pilot import run_pilot
from v4_pilot_helpers import arguments, assets


def _interrupted(args, tmp_path):
    run_pilot(**args, out=tmp_path / "probe", probe=True)
    original = training._train_step
    steps = 0

    def stop(*a, **kw):
        nonlocal steps
        result = original(*a, **kw)
        steps += 1
        if steps == 2:
            signal.raise_signal(signal.SIGTERM)
        return result

    with patch.object(training, "_train_step", side_effect=stop):
        report = run_pilot(**args, out=tmp_path / "interrupted", execute=True)
    assert report["training_status"] == "INCOMPLETE"
    assert report["training"]["global_step"] == 2
    assert report["training"]["tokens_seen"] == 60
    return {**args, "training_config": {**args["training_config"],
             "resume": str(Path(args["training_config"]["out"]) / "state.pt")}}


def test_v4_continuous_and_interrupted_resume_match_optimizer_rng_and_cursor(tmp_path):
    args = arguments(assets(tmp_path))
    resumed_args = _interrupted(args, tmp_path)
    resumed = run_pilot(**resumed_args, out=tmp_path / "resumed", execute=True)
    assert resumed["status"] == "COMPLETED", resumed
    assert resumed["training"]["tokens_seen"] == 120
    run = Path(args["training_config"]["out"])
    previous = training.load_training_state(run / "state.prev.pt", mmap=True)
    assert previous["global_step"] < 4
    lineage = read_json(run / "pilot-lineage.json")
    assert len(lineage["resume_history"]) == 1
    continuous_args = {**args, "workspace": tmp_path / "continuous-workspace",
                       "training_config": {**args["training_config"], "out": str(tmp_path / "continuous-run")}}
    run_pilot(**continuous_args, out=tmp_path / "continuous-probe", probe=True)
    continuous = run_pilot(**continuous_args, out=tmp_path / "continuous", execute=True)
    assert continuous["status"] == "COMPLETED", continuous
    a = training.load_training_state(run / "state.pt", mmap=True)
    b = training.load_training_state(tmp_path / "continuous-run/state.pt", mmap=True)
    assert compare_tree(a["model"], b["model"], exact=True) == 0
    assert compare_tree(a["optimizer"], b["optimizer"], exact=True) == 0
    compare_tree(a["rng"], b["rng"], exact=True)
    for key in ("global_step", "tokens_seen", "samples_seen", "epoch", "micro_batches_consumed_in_epoch"):
        assert a[key] == b[key]
    log_a = [r["learning_rate"] for r in read_records(run / "training-log.jsonl") if r.get("event") == "train"]
    log_b = [r["learning_rate"] for r in read_records(tmp_path / "continuous-run/training-log.jsonl") if r.get("event") == "train"]
    assert log_a == log_b and len(log_a) == 4


def read_records(path):
    from minillm.assessment_io import records
    return list(records(path))


@pytest.mark.parametrize("change", ["data", "tokenizer", "corrupt_checkpoint"])
def test_v4_resume_rejects_changed_data_tokenizer_or_corrupt_state(tmp_path, change):
    args = arguments(assets(tmp_path))
    resume_args = _interrupted(args, tmp_path)
    run = Path(args["training_config"]["out"])
    state_before = file_info(run / "state.pt")
    if change == "data":
        with (args["dataset"] / "train.bin").open("r+b") as handle:
            handle.write(b"\x00\x00")
    elif change == "tokenizer":
        tokenizer = read_json(args["tokenizer"])
        tokenizer["added_tokens"][0]["content"] = "<changed-pad>"
        from minillm.training_manifest import write_json
        write_json(args["tokenizer"], tokenizer)
    else:
        (run / "state.pt").write_bytes(b"corrupted checkpoint")
    with patch("minillm.v4_training_pilot.launch_pilot", side_effect=AssertionError("must not resume")):
        result = run_pilot(**resume_args, out=tmp_path / "rejected-resume", execute=True)
    assert result["training_status"] == "NOT_RUN"
    assert result["preflight"]["readiness"] == "BLOCKED"
    if change != "corrupt_checkpoint":
        assert file_info(run / "state.pt") == state_before


def test_resume_can_refresh_workload_in_new_directory_without_touching_state_or_old_proof(tmp_path):
    args = arguments(assets(tmp_path))
    resumed = _interrupted(args, tmp_path)
    state = Path(resumed["training_config"]["resume"])
    old_proof = args["workspace"] / "workload/workload.json"
    before = file_info(state), file_info(old_proof)
    resumed["training_config"]["workload_report"] = str(tmp_path / "new-workload")
    report = run_pilot(**resumed, out=tmp_path / "probe-refresh", probe=True)
    assert report["status"] == "WORKLOAD_PASS"
    assert before == (file_info(state), file_info(old_proof))
    result = run_pilot(**resumed, out=tmp_path / "resume-new-proof", execute=True)
    assert result["status"] == "COMPLETED", result
    assert result["training"]["tokens_seen"] == 120


@pytest.mark.parametrize("target,steps,planned", [(100, 8, 240), (180, 6, 180)])
def test_resume_cannot_restart_or_extend_the_original_cumulative_budget(tmp_path, target, steps, planned):
    args = arguments(assets(tmp_path))
    resumed = _interrupted(args, tmp_path)
    state = Path(resumed["training_config"]["resume"])
    before = file_info(state)
    resumed["training_config"].update(target_tokens=target, max_steps=steps, planned_effective_tokens=planned)
    with patch("minillm.v4_training_pilot.launch_pilot", side_effect=AssertionError("must not launch")):
        result = run_pilot(**resumed, out=tmp_path / "enlarged-resume", execute=True)
    assert result["training_status"] == "NOT_RUN"
    assert {"BLOCKED_TOKEN_BUDGET", "BLOCKED_PILOT_IDENTITY"} & set(result["preflight"]["blockers"])
    assert file_info(state) == before
