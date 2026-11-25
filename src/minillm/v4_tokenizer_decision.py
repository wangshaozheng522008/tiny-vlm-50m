"""Human decision evidence, independently versioned after candidate evaluation."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

from .corpus_utils import CorpusError, identity, sha256, verify_file
from .tokenizer_v4 import candidate_record, guard_train_only, parameter_budget


def training_data_id(directory: Path) -> str:
    return identity({"train_split_sha256": sha256(directory / "train.text.jsonl"),
                     "split_manifest_sha256": sha256(directory / "split-manifest.json")})


def validate_decision(directory: Path, decision: dict) -> dict:
    """No recommendation, filename or declared fingerprint grants approval."""
    directory = Path(directory)
    required = {"candidate", "fingerprint", "vocab_size", "training_data_content_id",
                "train_split_sha256", "evaluation_report_sha256", "reason", "reviewer",
                "reviewed_at", "model_budget_conclusion"}
    if not isinstance(decision, dict) or not required <= decision.keys():
        raise CorpusError("human tokenizer decision is incomplete")
    for key in required - {"vocab_size"}:
        if not isinstance(decision[key], str) or not decision[key].strip():
            raise CorpusError(f"human tokenizer decision missing {key}")
    try:
        datetime.date.fromisoformat(decision["reviewed_at"])
    except ValueError as exc:
        raise CorpusError("decision reviewed_at requires a real ISO date") from exc
    if type(decision["vocab_size"]) is not int or decision["vocab_size"] != 24576:
        raise CorpusError("formal 87M tokenizer decision requires vocab_size=24576; 32768 is research only")
    guard_train_only(directory / "train.text.jsonl", directory / "split-manifest.json")
    report = json.loads((directory / "tokenizer-candidates.json").read_text(encoding="utf-8"))
    candidate = next((c for c in report["candidates"] if c["name"] == decision["candidate"]), None)
    if candidate is None:
        raise CorpusError("decision candidate was not trained for this dataset")
    verified = candidate_record(Path(candidate["directory"]))
    if candidate != verified:
        raise CorpusError("candidate report differs from verified candidate")
    metadata = verified["metadata"]
    evaluation = json.loads((directory / "tokenizer-evaluation.json").read_text(encoding="utf-8"))
    metrics = evaluation["candidates"][decision["candidate"]]
    if (metadata["demo_only"] or metadata["vocab_size"] != 24576
            or decision["fingerprint"] != verified["fingerprint"]
            or metrics["fingerprint"] != verified["fingerprint"]
            or not metrics["roundtrip_ok"] or not metrics["document_boundary_ok"]
            or metadata["train_split_sha256"] != sha256(directory / "train.text.jsonl")
            or metadata["split_manifest_sha256"] != sha256(directory / "split-manifest.json")
            or decision["training_data_content_id"] != training_data_id(directory)
            or decision["train_split_sha256"] != metadata["train_split_sha256"]
            or decision["evaluation_report_sha256"] != sha256(directory / "tokenizer-evaluation.json")):
        raise CorpusError("human decision/candidate/train/evaluation identity or correctness mismatch")
    budget = parameter_budget([24576])["vocab_sizes"]["24576"]
    if budget["total_parameters"] != 86528640 or not budget["int4_under_50m_tensor_payload"]:
        raise CorpusError("tokenizer exceeds the model parameter or INT4 tensor budget")
    return {"name": verified["name"], "path": verified["path"], "fingerprint": verified["fingerprint"],
            "tokenizer": {"selected": verified["name"], "fingerprint": verified["fingerprint"]},
            "vocab_size": 24576, "mode": "HUMAN_REVIEWED", "reason": decision["reason"],
            "decision": decision, "decision_sha256": identity(decision)}


def verify_selection(directory: Path) -> dict:
    directory = Path(directory)
    selection = json.loads((directory / "tokenizer-selection.json").read_text(encoding="utf-8"))
    if selection != validate_decision(directory, selection.get("decision")):
        raise CorpusError("published tokenizer decision was changed")
    plan = json.loads((directory / ".v4/plan.json").read_text(encoding="utf-8"))
    if plan["plan_key"] != identity({k: v for k, v in plan.items() if k != "plan_key"}):
        raise CorpusError("tokenizer decision generation plan identity mismatch")
    stages = {}
    for name, artifact in (("tokenizer_eval", "tokenizer-evaluation.json"),
                           ("tokenizer_selection", "tokenizer-selection.json")):
        stage = json.loads((directory / f".v4/{name}.stage.json").read_text(encoding="utf-8"))
        if str((directory / artifact).resolve()) not in stage["files"]:
            raise CorpusError("tokenizer decision lacks stage artifact evidence")
        for filename, info in stage["files"].items():
            verify_file(Path(filename), info)
        stages[name] = stage
    from .v4_language_pilot import LANGUAGE_CONFIG, is_language

    if plan.get("generation_config") and is_language(plan["generation_config"]):
        verify_evaluation_stage(directory, plan["generation_config"], base=LANGUAGE_CONFIG.parents[1])
    expected = identity([4, plan["plan_key"], "tokenizer_selection",
                         {"evaluation": stages["tokenizer_eval"]["key"],
                          "decision": selection["decision_sha256"]}])
    if stages["tokenizer_selection"]["key"] != expected:
        raise CorpusError("tokenizer decision stage key differs from generation identity")
    return selection


def verify_evaluation_stage(directory, config, *, base):
    """Verify the evaluation and candidate stage keys before using their measurements."""
    from .corpus_utils import file_info

    directory = Path(directory).resolve()
    plan = json.loads((directory / ".v4/plan.json").read_text(encoding="utf-8"))
    if (plan["config_sha256"] != identity(config)
            or plan["plan_key"] != identity({k: v for k, v in plan.items() if k != "plan_key"})):
        raise CorpusError("tokenizer evaluation generation plan mismatch")

    def stage(name, artifact):
        result = json.loads((directory / f".v4/{name}.stage.json").read_text(encoding="utf-8"))
        if str(directory / artifact) not in result["files"]:
            raise CorpusError(f"{name} lacks its published report")
        for filename, info in result["files"].items():
            verify_file(Path(filename), info)
        return result

    split = stage("split", "split-manifest.json")
    candidate_stage = stage("tokenizers", "tokenizer-candidates.json")
    evaluation_stage = stage("tokenizer_eval", "tokenizer-evaluation.json")
    training = config["tokenizer"]["training"]
    candidate_report = json.loads((directory / "tokenizer-candidates.json").read_text(encoding="utf-8"))
    expected_candidates = config["tokenizer"]["candidates"]
    candidates = candidate_report["candidates"]
    if ([{k: c[k] for k in ("name", "vocab_size")} for c in candidates] != expected_candidates
            or candidate_report["demo_only"] is not False):
        raise CorpusError("tokenizer evaluation has unexpected candidates")
    for candidate in candidates:
        verified = candidate_record(Path(candidate["directory"]))
        metadata = verified["metadata"]
        if (candidate != verified
                or metadata["train_split_sha256"] != sha256(directory / "train.text.jsonl")
                or metadata["split_manifest_sha256"] != sha256(directory / "split-manifest.json")
                or metadata["text_policy"] != config["text_policy"]
                or metadata["demo_only"] is not False
                or any(metadata["training_selection"][k] != training[k]
                       for k in ("seed", "max_chars", "category_weights"))):
            raise CorpusError("tokenizer candidate differs from the Train split or training policy")
    expected = identity([4, plan["plan_key"], "tokenizers", {
        "split": split["key"], "candidates": expected_candidates,
        "selection": {"train_sha256": sha256(directory / "train.text.jsonl"),
                      **{k: training[k] for k in ("seed", "max_chars", "category_weights")}}}])
    evaluation = json.loads((directory / "tokenizer-evaluation.json").read_text(encoding="utf-8"))
    baseline = (evaluation.get("baseline") or {}).get("path")
    baseline_path = Path(baseline) if baseline else None
    expected_evaluation = identity([4, plan["plan_key"], "tokenizer_eval", {
        "tokenizers": expected, "eval": file_info(Path(base) / config["tokenizer"]["eval_asset"]),
        "baseline": file_info(baseline_path) if baseline_path and baseline_path.is_file() else None}])
    actual_metrics = evaluation["candidates"]
    if (candidate_stage["key"] != expected or evaluation_stage["key"] != expected_evaluation
            or evaluation["demo_only"] is not False
            or set(actual_metrics) != {c["name"] for c in candidates}
            or any(actual_metrics[c["name"]]["fingerprint"] != c["fingerprint"] for c in candidates)):
        raise CorpusError("tokenizer evaluation stage key or candidate metrics mismatch")
    return evaluation_stage
