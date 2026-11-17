from pathlib import Path
from unittest.mock import patch

import pytest

from minillm.assessment_io import read_json, verified_evaluation
from minillm.corpus_utils import file_info
from minillm.v4_training_pilot import preflight, run_pilot
from v4_pilot_helpers import arguments, assets


def test_cpu_nano_end_to_end_probe_readiness_train_resume_evaluation(tmp_path):
    value = assets(tmp_path)
    args = arguments(value)
    with patch("minillm.v4_training_pilot.launch_pilot", side_effect=AssertionError("inspection must not launch")):
        dry = run_pilot(**args, out=tmp_path / "inspect")
    assert dry["training_status"] == "NOT_RUN"
    assert "BLOCKED_WORKLOAD_NOT_READY" in dry["preflight"]["blockers"]
    assert not Path(args["training_config"]["out"]).exists()
    probed = run_pilot(**args, out=tmp_path / "probe", probe=True)
    assert probed["status"] == "WORKLOAD_PASS", probed
    assert probed["workload"]["measurements"]["optimizer_steps"] == 3
    assert probed["training_status"] == "NOT_RUN"
    with patch("minillm.v4_training_pilot.launch_pilot", side_effect=AssertionError("no --execute")):
        ready = run_pilot(**args, out=tmp_path / "ready")
    assert ready["status"] == "READY"
    assert ready["readiness"]["status"] == "READY"
    initial = file_info(args["workspace"] / "initial.pt")
    result = run_pilot(**args, out=tmp_path / "execute", execute=True)
    assert result["status"] == "COMPLETED", result
    assert result["training"]["global_step"] == 4
    assert result["training"]["tokens_seen"] == 120
    assert result["engineering"] == "ENGINEERING_PASS"
    assert result["gpu_training_status"] == "NOT_RUN"
    assert file_info(args["workspace"] / "initial.pt") == initial
    baseline, bm = verified_evaluation(args["workspace"] / "baseline/suite-0")
    final, fm = verified_evaluation(tmp_path / "execute/final/suite-0")
    assert baseline["checkpoint"]["global_step"] == 0
    assert baseline["checkpoint"]["sha256"] == initial["sha256"]
    assert bm["comparison_policy"] == fm["comparison_policy"]
    assert final["checkpoint"]["global_step"] == 4
    lineage = read_json(Path(args["training_config"]["out"]) / "pilot-lineage.json")
    assert lineage["initialization"]["sha256"] == initial["sha256"]
    resumed_args = {**args, "training_config": {**args["training_config"],
                   "resume": str(Path(args["training_config"]["out"]) / "state.pt")}}
    resumed = run_pilot(**resumed_args, out=tmp_path / "resume", execute=True)
    assert resumed["status"] == "COMPLETED", resumed
    assert resumed["training"]["global_step"] == 4
    assert resumed["training"]["tokens_seen"] == 120
    assert resumed["recovery"]["resumed"]


@pytest.mark.parametrize("change", [{"seq_len": 8}, {"lr": .0001}, {"seed": 43}])
def test_changed_plan_cannot_reuse_workload(tmp_path, change):
    args = arguments(assets(tmp_path))
    run_pilot(**args, out=tmp_path / "probe", probe=True)
    changed = {**args, "training_config": {**args["training_config"], **change}}
    result = run_pilot(**changed, out=tmp_path / "changed", execute=True)
    assert "BLOCKED_WORKLOAD_NOT_READY" in result["preflight"]["blockers"]
    assert result["training_status"] == "NOT_RUN"


def test_frozen_evaluation_assets_cannot_change_between_stages(tmp_path):
    args = arguments(assets(tmp_path))
    run_pilot(**args, out=tmp_path / "probe", probe=True)
    with (args["suites"][0] / "prompts.jsonl").open("a") as handle:
        handle.write("\n")
    report = run_pilot(**args, out=tmp_path / "changed-assets", execute=True)
    assert "BLOCKED_PILOT_IDENTITY" in report["preflight"]["blockers"]
    assert report["training_status"] == "NOT_RUN"


def test_missing_or_damaged_workload_proof_never_calls_training(tmp_path):
    args = arguments(assets(tmp_path))
    run_pilot(**args, out=tmp_path / "probe", probe=True)
    path = args["workspace"] / "workload/workload.json"
    proof = path.read_bytes()
    for name, content in (("damaged", b"{}"), ("missing", None)):
        if content is None:
            path.unlink()
        else:
            path.write_bytes(content)
        with patch("minillm.v4_training_pilot.launch_pilot", side_effect=AssertionError("must not train")):
            report = run_pilot(**args, out=tmp_path / name, execute=True)
        assert "BLOCKED_WORKLOAD_NOT_READY" in report["preflight"]["blockers"]
        assert report["training_status"] == "NOT_RUN"
    path.write_bytes(proof)
