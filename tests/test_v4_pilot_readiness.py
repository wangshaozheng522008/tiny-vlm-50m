import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from minillm.assessment_io import read_json
from minillm.corpus_audit import audit_dataset
from minillm.training_manifest import write_json
from minillm.v4_admission import audit_v4_dataset, validate_v4_pilot_policy
from minillm.v4_training_pilot import preflight, run_pilot, validate_pilot_budget
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
    assert report["pilot_policy"]["ok"]
    assert report["pilot_policy"]["status"] == "POLICY_MATCH"


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
    assert report["preflight"]["pilot_policy"]["status"] == "BLOCKED_POLICY_MISMATCH"
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


def _set_policy(path, value):
    def change(config):
        item = config
        for key in path[:-1]:
            item = item[key]
        item[path[-1]] = copy.deepcopy(value)
    return change


@pytest.mark.parametrize("path,value", [
    (("quality", "rules", "empty_or_short", "min_chars"), 79),
    (("quality", "rules", "empty_or_short", "min_chars"), 81),
    (("quality", "rules", "punctuation_density", "max_ratio"), .31),
    (("deduplication", "near", "threshold"), .84),
    (("deduplication", "near", "method"), "exact"),
    (("split", "verify_near_cross_split"), False),
    (("split", "seed"), 43),
    (("split", "ratios"), {"train": .94, "val": .03, "test": .03}),
    (("contamination", "enabled"), False),
    (("contamination", "min_prompt_chars"), 13),
    (("contamination", "long_span_containment"), .9),
    (("targets", "category_token_ratios"), {"zh_general": .39, "zh_encyclopedic": .26,
        "zh_education": .15, "en_general": .1, "math_logic_code": .1}),
    (("text_policy",), "width-fold-prose-v1"),
    (("targets", "effective_train_tokens"), 2047),
    (("targets", "seq_len"), 127),
    (("audit", "retokenize"), "sample"),
    (("corpus_version",), "different-but-still-pilot"),
])
def test_generation_policy_mismatch_never_launches(tmp_path, path, value):
    from minillm.corpus_v4 import verify_frozen
    value = assets(tmp_path, policy_change=_set_policy(path, value))
    args = arguments(value)
    # These are genuinely generated reports, not broken hash simulations. Most
    # altered policies can still freeze under their own self-consistent rules.
    if (args["dataset"] / "FROZEN").exists():
        verify_frozen(args["dataset"])
    with patch("minillm.v4_training_pilot.launch_pilot", side_effect=AssertionError("must not launch")):
        report = run_pilot(**args, out=tmp_path / "policy-rejected", execute=True)
    assert report["preflight"]["pilot_policy"]["status"] == "BLOCKED_POLICY_MISMATCH"
    assert report["training_status"] == "NOT_RUN"
    assert not Path(args["training_config"]["out"]).exists()
    assert not args["workspace"].exists()


def test_self_consistent_frozen_weaker_quality_is_rejected_by_approved_policy(tmp_path):
    from minillm.corpus_v4 import verify_frozen
    from minillm.corpus_utils import identity
    value = assets(tmp_path, policy_change=_set_policy(("quality", "rules", "empty_or_short", "min_chars"), 79))
    frozen = verify_frozen(value["dataset"])
    assert audit_v4_dataset(value["dataset"])["ok"]
    assert frozen["content_id"] == identity(frozen["identity"])
    report = validate_v4_pilot_policy(value["dataset"], value["approved_config_path"])
    assert report["status"] == "BLOCKED_POLICY_MISMATCH"
    assert report["generation_policy_sha256"] != report["approved_policy_sha256"]


def test_replaced_contamination_asset_policy_is_rejected(tmp_path):
    def change(config):
        config["contamination"]["assets"] = config["contamination"]["assets"][:-1]
    value = assets(tmp_path, policy_change=change)
    from minillm.corpus_v4 import verify_frozen
    verify_frozen(value["dataset"])
    report = validate_v4_pilot_policy(value["dataset"], value["approved_config_path"])
    assert report["status"] == "BLOCKED_POLICY_MISMATCH"


@pytest.mark.parametrize("missing", ["generation_config", "quality_hash", "plan", "approved_config", "review", "review_check", "freeze_record"])
def test_missing_policy_evidence_fails_closed(v4_assets, tmp_path, missing):
    args = arguments(v4_assets)
    path = args["dataset"] / ".v4/plan.json"
    if missing == "plan":
        path.unlink()
    elif missing == "approved_config":
        args["approved_config_path"].unlink()
    elif missing == "review":
        (args["dataset"] / "audit-report.json").unlink()
    elif missing == "freeze_record":
        (args["dataset"] / "freeze.json").unlink()
    elif missing == "review_check":
        path = args["dataset"] / "audit-report.json"
        review = read_json(path)
        del review["checks"]["cleaning_policy"]
        write_json(path, review)
    else:
        plan = read_json(path)
        if missing == "generation_config":
            del plan[missing]
        else:
            del plan["policies"]["quality_policy_sha256"]
        write_json(path, plan)
    # Generic PR8 verification remains supported; stricter Pilot promotion needs
    # the additional generation/review evidence.
    from minillm.corpus_v4 import verify_frozen
    verify_frozen(args["dataset"])
    with patch("minillm.v4_training_pilot.launch_pilot", side_effect=AssertionError("must not launch")):
        report = run_pilot(**args, out=tmp_path / "missing-evidence", execute=True)
    assert report["preflight"]["pilot_policy"]["status"] == "BLOCKED_POLICY_EVIDENCE_MISSING"
    assert report["training_status"] == "NOT_RUN"
    assert not args["workspace"].exists()


def test_rebinding_only_config_and_plan_hashes_cannot_change_generation_evidence(v4_assets):
    from minillm.corpus_utils import identity
    from minillm.corpus_v4 import verify_frozen
    path = v4_assets["dataset"] / ".v4/plan.json"
    plan = read_json(path)
    plan["generation_config"]["runtime"]["commit_every"] += 1
    plan["config_sha256"] = identity(plan["generation_config"])
    plan["plan_key"] = identity({k: v for k, v in plan.items() if k != "plan_key"})
    write_json(path, plan)
    verify_frozen(v4_assets["dataset"])
    report = validate_v4_pilot_policy(v4_assets["dataset"], v4_assets["approved_config_path"])
    assert report["status"] == "BLOCKED_POLICY_MISMATCH"
    assert "stage" in report["errors"][0]


def test_approved_source_and_tokenizer_selection_and_runtime_can_vary(tmp_path):
    def change(config):
        config["sources"][0]["paths"] = ["reviewed-source-location.jsonl"]
        config["tokenizer"]["selected"] = "fixture-384"
        config["runtime"]["sqlite_cache_mib"] = 32
        config["hardware"]["device"] = "CPU fixture"
    value = assets(tmp_path, policy_change=change)
    report = validate_v4_pilot_policy(value["dataset"], value["approved_config_path"])
    assert report["ok"], report


def test_policy_quality_rules_version_is_checked_against_generation(v4_assets, tmp_path):
    from minillm.corpus_v4 import verify_frozen
    with patch("minillm.corpus_v4.QUALITY_RULES_VERSION", "corpus-quality-other-version"), \
         patch("minillm.corpus_quality.QUALITY_RULES_VERSION", "corpus-quality-other-version"):
        value = assets(tmp_path / "different-rules")
    verify_frozen(value["dataset"])
    report = validate_v4_pilot_policy(value["dataset"], value["approved_config_path"])
    assert report["status"] == "BLOCKED_POLICY_MISMATCH"
    assert "quality policy hash" in report["errors"][0]


def test_approved_policy_is_content_based_not_its_filename(v4_assets, tmp_path):
    renamed = tmp_path / "reviewed-policy-copy.yaml"
    renamed.write_bytes(v4_assets["approved_config_path"].read_bytes())
    report = validate_v4_pilot_policy(v4_assets["dataset"], renamed)
    assert report["status"] == "POLICY_MATCH", report


@pytest.fixture
def actual_pilot_windows(tmp_path):
    import numpy as np
    from minillm.data import PackedPretrainDataset
    # Real BIN windows for pure budget checks; this is not a frozen training
    # corpus and is never sent to the launcher or model.
    path = tmp_path / "budget-windows.bin"
    np.zeros(1024 * 1024 + 1, dtype="<u2").tofile(path)
    return path, PackedPretrainDataset


@pytest.mark.parametrize("seq_len,steps,planned", [(1024, 123, 1_006_632), (512, 245, 1_001_560)])
def test_fixed_million_budget_allows_actual_window_rounding(actual_pilot_windows, tmp_path, seq_len, steps, planned):
    path, dataset_type = actual_pilot_windows
    cfg = read_json(Path(__file__).resolve().parents[1] / "configs/pilot-v4-1m-t1200.json")
    opts, extra = parse_training_config({**cfg, "seq_len": seq_len, "max_steps": steps,
                                         "planned_effective_tokens": planned})
    report = validate_pilot_budget(len(dataset_type(path, seq_len)), opts, extra)
    assert report["target_tokens"] == 1_000_000
    assert report["planned_effective_tokens"] == planned
    assert planned <= report["independent_effective_tokens"]


@pytest.mark.parametrize("target", [1_000_001, 2_000_000, 10_000_000, 100_000_000, 999_999,
                                    True, False, 1_000_000.0, "1000000"])
def test_production_target_is_exact_integer_and_never_launches(v4_assets, actual_pilot_windows, tmp_path, target):
    cfg = read_json(Path(__file__).resolve().parents[1] / "configs/pilot-v4-1m-t1200.json")
    opts, extra = parse_training_config({**cfg, "target_tokens": target})
    path, dataset_type = actual_pilot_windows
    with pytest.raises(ValueError, match="exactly 1,000,000 integer"):
        validate_pilot_budget(len(dataset_type(path, 1024)), opts, extra)
    args = arguments(v4_assets)
    args.update(cpu_nano_test=False, approved_config_path=None,
                training_config={**args["training_config"], "target_tokens": target})
    with patch("minillm.v4_training_pilot.launch_pilot", side_effect=AssertionError("must not launch")):
        report = run_pilot(**args, out=tmp_path / "bad-target", execute=True)
    assert "BLOCKED_TOKEN_BUDGET" in report["preflight"]["blockers"]
    assert report["training_status"] == "NOT_RUN"


@pytest.mark.parametrize("change", [{"max_steps": 124}, {"max_steps": 10000},
    {"planned_effective_tokens": 1_000_000}, {"planned_effective_tokens": 1_006_632.0},
    {"planned_effective_tokens": True}, {"allow_data_repetition": True}])
def test_steps_declaration_and_repetition_cannot_bypass_budget(actual_pilot_windows, change):
    cfg = read_json(Path(__file__).resolve().parents[1] / "configs/pilot-v4-1m-t1200.json")
    opts, extra = parse_training_config({**cfg, **change})
    path, dataset_type = actual_pilot_windows
    with pytest.raises(ValueError):
        validate_pilot_budget(len(dataset_type(path, extra["seq_len"])), opts, extra)


@pytest.mark.parametrize("change", [{"max_steps": 8}, {"max_steps": 10000},
    {"planned_effective_tokens": 121}, {"planned_effective_tokens": 120.0},
    {"planned_effective_tokens": True}, {"allow_data_repetition": True}])
def test_invalid_runtime_budget_never_calls_launcher(v4_assets, tmp_path, change):
    args = arguments(v4_assets)
    args["training_config"] = {**args["training_config"], **change}
    with patch("minillm.v4_training_pilot.launch_pilot", side_effect=AssertionError("must not launch")):
        report = run_pilot(**args, out=tmp_path / "bad-runtime-budget", execute=True)
    assert "BLOCKED_TOKEN_BUDGET" in report["preflight"]["blockers"]
    assert report["training_status"] == "NOT_RUN"
    assert not Path(args["training_config"]["out"]).exists()
    assert not args["workspace"].exists()
