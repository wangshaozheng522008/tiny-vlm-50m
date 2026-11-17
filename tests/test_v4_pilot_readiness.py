import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from minillm.assessment_io import read_json
from minillm.corpus_audit import audit_dataset
from minillm.training_manifest import write_json
from minillm.v4_admission import audit_v4_dataset
from minillm.v4_training_pilot import preflight, run_pilot
from minillm.workload import parse_training_config
from minillm.training_readiness import planned_tokens
from v4_pilot_helpers import arguments, assets


@pytest.fixture
def v4_assets(tmp_path):
    return assets(tmp_path)


def test_frozen_v4_passes_shared_admission_and_legacy_adapter(v4_assets):
    for check in (audit_v4_dataset, audit_dataset):
        report = check(v4_assets["dataset"], tokenizer_path=v4_assets["tokenizer"], retokenize=True, verify_inputs=True)
        assert report["ok"], report["errors"]
    report = preflight(**arguments(v4_assets), require_workload=False)
    assert report["readiness"] == "READY", report["blocker_details"]
    assert report["budget"]["planned_effective_tokens"] == 120
    assert report["scope"] == "CPU_NANO_TEST"


@pytest.mark.parametrize("failure", ["demo", "unapproved", "missing_freeze", "missing_evidence",
                                     "flags_only", "tokenizer", "vocabulary", "budget", "occupied"])
def test_v4_pilot_rejects_invalid_admission(v4_assets, tmp_path, failure):
    args = arguments(v4_assets)
    dataset = args["dataset"]
    path = dataset / "dataset-manifest.json"
    manifest = read_json(path)
    expected = "BLOCKED_DATA_NOT_READY"
    if failure == "demo":
        manifest["dataset_status"] = "DEMO_ONLY"
    elif failure == "unapproved":
        manifest["license_sources"]["sources"][0]["license_status"] = "REVIEW_REQUIRED"
    elif failure == "missing_freeze":
        (dataset / "FROZEN").unlink()
    elif failure == "missing_evidence":
        split = read_json(dataset / "split-manifest.json")
        del split["leakage"]["near_duplicate"]["status"]
        write_json(dataset / "split-manifest.json", split)
    elif failure == "flags_only":
        manifest["freeze_status"] = "READY"
        manifest["license_review_status"] = "APPROVED"
        manifest["license_sources"]["sources"][0]["license_evidence"] = ""
    elif failure == "tokenizer":
        from minillm.tokenizer import MiniLLMTokenizer
        alternate = tmp_path / "alternate.json"
        MiniLLMTokenizer.train([f"completely different vocabulary words item {i} and documents"
                               for i in range(1000)], vocab_size=320, save_path=alternate)
        args["tokenizer"] = alternate
        expected = "BLOCKED_TOKENIZER_NOT_READY"
    elif failure == "vocabulary":
        cfg = read_json(args["model_config"])
        cfg["vocab_size"] += 1
        write_json(Path(args["model_config"]), cfg)
        expected = "BLOCKED_TOKENIZER_NOT_READY"
    elif failure == "budget":
        args["training_config"] = {**args["training_config"], "max_steps": 100_000, "target_tokens": 3_000_000,
                                   "planned_effective_tokens": 3_000_000}
        expected = "BLOCKED_TOKEN_BUDGET"
    else:
        run = Path(args["training_config"]["out"])
        run.mkdir()
        (run / "historical.pt").write_bytes(b"preserve PR5")
        expected = "BLOCKED_OUTPUT_OCCUPIED"
    write_json(path, manifest)
    with patch("minillm.v4_training_pilot.launch_pilot", side_effect=AssertionError("must not launch")):
        report = run_pilot(**args, out=tmp_path / "blocked", execute=True)
    assert expected in report["preflight"]["blockers"]
    assert report["training_status"] == report["gpu_training_status"] == "NOT_RUN"
    assert not args["workspace"].exists()


def test_unavailable_gpu_does_not_fall_back_to_cpu(v4_assets, tmp_path):
    args = arguments(v4_assets)
    args.update(cpu_nano_test=False, model_config="minillm-87m",
                training_config={**args["training_config"], "device": "cuda"})
    with patch("torch.cuda.is_available", return_value=False), \
         patch("minillm.v4_training_pilot.launch_pilot", side_effect=AssertionError("no fallback")):
        report = run_pilot(**args, out=tmp_path / "gpu-unavailable", execute=True)
    assert "BLOCKED_GPU_NOT_READY" in report["preflight"]["blockers"]
    assert report["gpu_training_status"] == "NOT_RUN"


def test_candidate_1m_budget_is_exact_and_formal_100m_target_unchanged():
    root = Path(__file__).resolve().parents[1]
    cfg = read_json(root / "configs/pilot-v4-1m-t1200.json")
    opts, extra = parse_training_config(cfg)
    assert 8 * 1023 == 8184
    assert planned_tokens(1024, opts.batch_size, opts.grad_accum, extra["seq_len"], opts.max_steps) == 1_006_632
    from minillm.corpus_v4 import load_config
    formal = load_config(root / "configs/corpus_v4.yaml")
    pilot = load_config(root / "configs/corpus_v4_pilot_1m.yaml")
    assert formal["targets"]["effective_train_tokens"] == 100_000_000
    assert pilot["targets"]["effective_train_tokens"] == 1_020_000
    assert pilot["targets"]["category_token_ratios"] == formal["targets"]["category_token_ratios"]
    assert pilot["corpus_version"] != formal["corpus_version"]
    for name in ("quality", "deduplication", "split", "license_policy", "contamination"):
        assert pilot[name] == formal[name]


@pytest.mark.parametrize("field,value", [("license_status", "REVIEW_REQUIRED"),
    ("license_status", "REJECTED"), ("license_evidence", ""),
    ("license_reviewer", ""), ("license_reviewed_at", "not-a-date")])
def test_ready_flags_and_rebound_source_hashes_cannot_approve_invalid_sources(v4_assets, field, value):
    # Reach semantic source validation even if someone recomputes local hashes.
    from minillm.corpus_utils import identity
    path = v4_assets["dataset"] / "dataset-manifest.json"
    manifest = read_json(path)
    registry = manifest["license_sources"]
    registry["sources"][0][field] = value
    registry["sha256"] = identity({k: v for k, v in registry.items() if k != "sha256"})
    manifest["identity"]["v4"]["source_manifest_sha256"] = registry["sha256"]
    manifest["source_manifest_sha256"] = registry["sha256"]
    manifest["content_id"] = identity(manifest["identity"])
    write_json(path, manifest)
    report = audit_v4_dataset(v4_assets["dataset"])
    assert not report["ok"]
    assert "APPROVED" in report["errors"][0] or "approved" in report["errors"][0]


def test_repetition_old_initialization_and_gpu_test_mode_are_rejected(v4_assets):
    args = arguments(v4_assets)
    for change, blocker in [({"allow_data_repetition": True}, "BLOCKED_TOKEN_BUDGET"),
                            ({"init_from": "runs/pr5/model.pt"}, "BLOCKED_INITIALIZATION"),
                            ({"device": "cuda"}, "BLOCKED_DEVICE")]:
        cfg = {**args["training_config"], **change}
        report = preflight(**{**args, "training_config": cfg}, require_workload=False)
        assert blocker in report["blockers"]
