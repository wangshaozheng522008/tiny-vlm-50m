"""Frozen V4 evidence adapter shared by readiness, evaluation and the V4 pilot."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .assessment_io import read_json
from .corpus_utils import CorpusError, file_info, identity, verify_file
from .tokenizer import MiniLLMTokenizer
from .v4_language_pilot import (language_config,
                                is_language, split_text_path, tokenized_extra)

APPROVED_PILOT_CONFIG = Path(__file__).resolve().parents[2] / "configs/corpus_v4_pilot_1m.yaml"


def _pilot_policy(config):
    """Content-affecting policy; source approvals and reviewed choices vary."""
    policy = {key: config[key] for key in (
        "schema_version", "corpus_version", "dataset_name", "seed", "text_policy",
        "targets", "quality", "deduplication", "split", "contamination",
        "license_policy", "budget", "audit")} | {
        "tokenizer_training": config["tokenizer"]["training"],
        "tokenizer_evaluation": config["tokenizer"]["evaluation"]}
    if is_language(config):
        from .v4_language_pilot import LANGUAGE_TOLERANCE

        policy["contamination"] = {k: v for k, v in policy["contamination"].items() if k != "decisions"}
        from .corpus_versions import V5_VERSION

        algorithm = ("seeded-source-quota-prefix-v1" if config["corpus_version"] == V5_VERSION
                     else "seeded-category-prefix-v1")
        policy["train_selection"] = {"fraction_tolerance": LANGUAGE_TOLERANCE, "algorithm": algorithm}
    return policy


def validate_v4_pilot_policy(dataset_dir, approved_config_path=None):
    """Bind actual generation policy to frozen stages, then to approved policy.

    The caller supplies a reviewed, versioned configuration, never an expected
    hash. Production preflight pins the repository configuration. CPU fixtures
    may select an explicit independent test profile.
    """
    return _validate_v4_pilot_policy(dataset_dir, approved_config_path)


def _validate_v4_pilot_policy(dataset_dir, approved_config_path=None, *, prospective_manifest=None):
    from .corpus_v4 import V4_SCHEMA_VERSION, implementation_sha256, load_config, _verify_frozen
    from .corpus_quality import QUALITY_RULES_VERSION
    from .corpus_dedup import NearDupPolicy
    from .corpus_contamination import load_benchmark_items

    directory = Path(dataset_dir)
    report = {"ok": False, "status": "BLOCKED_POLICY_EVIDENCE_MISSING", "errors": []}
    try:
        plan = read_json(directory / ".v4/plan.json")
        actual = plan["generation_config"]
        pinned_profile = language_config(actual["corpus_version"]) if is_language(actual) else APPROVED_PILOT_CONFIG
        approved = load_config(approved_config_path or pinned_profile)
        policies = plan["policies"]
        for key in ("quality_policy_sha256", "dedup_policy_sha256", "split_policy_sha256",
                    "tokenizer_config_sha256", "text_policy"):
            policies[key]
        # Missing fields must never be defaulted by a configuration loader.
        actual_policy, approved_policy = _pilot_policy(actual), _pilot_policy(approved)
        report["approved_policy_sha256"] = identity(approved_policy)
        report["generation_policy_sha256"] = identity(actual_policy)
        if report["generation_policy_sha256"] != report["approved_policy_sha256"]:
            report["status"] = "BLOCKED_POLICY_MISMATCH"
            report["errors"] = ["generation policy differs from the approved versioned Pilot policy"]
            return report
        frozen = _verify_frozen(directory, prospective_manifest=prospective_manifest)
        recorded = frozen["identity"]["v4"]
        report["status"] = "BLOCKED_POLICY_MISMATCH"

        def require(condition, reason):
            if not condition:
                raise CorpusError(reason)

        require(identity(actual) == plan["config_sha256"], "generation config hash mismatch")
        require(identity({k: v for k, v in plan.items() if k != "plan_key"}) == plan["plan_key"],
                "generation plan hash mismatch")
        require(plan["schema_version"] == V4_SCHEMA_VERSION and plan["demo"] is False,
                "non-formal V4 generation plan")
        require(plan["corpus_version"] == actual["corpus_version"] == frozen["corpus_version"]
                == recorded["corpus_version"] and plan["dataset_name"] == actual["dataset_name"]
                == frozen["dataset_name"], "Pilot version identity mismatch")
        require(plan["implementation_sha256"] == recorded["implementation_sha256"]
                and implementation_compatible(frozen, implementation_sha256()),
                "generation implementation identity mismatch")
        require(plan["registry"] == frozen["license_sources"] and plan["registry"]["sha256"]
                == recorded["source_manifest_sha256"], "generation source registry binding mismatch")
        from .v4_sources import requires_source_review, validate_source_plan
        # _verify_frozen already reproduced the reviewed source bytes and historical samples.
        validate_source_plan(plan, frozen["license_sources"], require=requires_source_review(approved))

        quality_hash = identity({"rules_version": QUALITY_RULES_VERSION, "quality": approved["quality"]})
        require(policies["quality_policy_sha256"] == recorded["cleaning_policy_sha256"] == quality_hash,
                "quality policy hash mismatch")
        require(policies["dedup_policy_sha256"] == recorded["dedup_policy_sha256"]
                == identity(approved["deduplication"]), "deduplication policy hash mismatch")
        require(policies["split_policy_sha256"] == identity(approved["split"]), "split policy hash mismatch")
        require(policies["tokenizer_config_sha256"] == identity(actual["tokenizer"]),
                "reviewed tokenizer configuration binding mismatch")
        require(policies["text_policy"] == recorded["text_policy"] == approved["text_policy"],
                "text policy binding mismatch")

        stages = frozen["identity"]["stage_hashes"]

        def stage(name, extra=None):
            key = identity([V4_SCHEMA_VERSION, plan["plan_key"], name, extra or {}])
            require(stages[name]["key"] == key, f"{name} stage is not bound to the generation plan")
            return key

        ingested = stage("ingested")
        quality = stage("quality", {"ingested": ingested, "quality": quality_hash,
                                    "text_policy": approved["text_policy"]})
        dedup = stage("dedup", {"quality": quality, "dedup": policies["dedup_policy_sha256"]})
        split = stage("split", {"dedup": dedup, "split": policies["split_policy_sha256"],
                                "seed": approved["split"]["seed"]})
        selection_key = None
        if is_language(approved):
            selection_key = stage("train_selection", {"split": split, "tokenizer": recorded["tokenizer_fingerprint"]})
        stage("tokenized", tokenized_extra(approved, split=split,
                                          fingerprint=recorded["tokenizer_fingerprint"],
                                          seq_len=approved["targets"]["seq_len"], selection_key=selection_key))
        quality_report = read_json(directory / "quality-report.json")
        dedup_report = read_json(directory / "dedup-report.json")
        split_report = read_json(directory / "split-manifest.json")
        require(quality_report["rules_version"] == QUALITY_RULES_VERSION
                and quality_report["policy_sha256"] == quality_hash, "quality report binding mismatch")
        require(dedup_report["policy_sha256"] == policies["dedup_policy_sha256"]
                and dedup_report["exact"]["enabled"] is approved["deduplication"]["exact"]
                and dedup_report["policy"] == approved["deduplication"]["near"]
                and dedup_report["near_policy_sha256"] == NearDupPolicy.from_config(
                    approved["deduplication"]["near"]).policy_sha256, "deduplication report binding mismatch")
        require(all(split_report[k] == approved["split"][k] for k in ("strategy", "seed", "ratios"))
                and recorded["split_fractions"] == approved["split"]["ratios"], "split report binding mismatch")

        # Resolve approved assets from the project root, independent of cwd.
        root = Path(__file__).resolve().parents[2]
        assets = [{**asset, "path": str((root / asset["path"]).resolve())}
                  for asset in approved["contamination"]["assets"]]
        _, provenance, errors = load_benchmark_items(assets)
        contamination = read_json(directory / "contamination-report.json")
        require(not errors and bool(provenance) and contamination["assets"] == provenance,
                "frozen contamination asset identities differ from approved assets")
        require(contamination["rules"]["min_prompt_chars"] == approved["contamination"]["min_prompt_chars"]
                and contamination["rules"]["long_span_containment"] == approved["contamination"]["long_span_containment"]
                and not contamination["errors"] and contamination["stats"]["truncated"] is False,
                "contamination scan policy/completeness mismatch")
        require(contamination["content_id"] == identity({k: contamination[k] for k in (
            "assets", "hits", "rules", "status")}), "contamination audit identity mismatch")
        budget = read_json(directory / "budget-report.json")
        target = approved["targets"]["effective_train_tokens"]
        require(budget["target_effective_train_tokens"] == frozen["budget"]["target_effective_train_tokens"]
                == target and budget["seq_len"] == approved["targets"]["seq_len"],
                "Pilot effective-token target binding mismatch")
        require(budget["category_token_ratios"] == approved["targets"]["category_token_ratios"]
                and all(budget["categories"][k]["target_effective_tokens"] == target * ratio
                    for k, ratio in approved["targets"]["category_token_ratios"].items()),
                "Pilot category ratio binding mismatch")
        require(budget["status"] == recorded["budget_status"] == "READY_FOR_TRAINING"
                and budget["allow_repetition"] is approved["budget"]["allow_repetition"]
                and budget["content_id"] == identity({k: budget[k] for k in (
                    "target_effective_train_tokens", "seq_len", "category_token_ratios", "categories",
                    "tokenizer_fingerprint", "status")}), "Pilot budget review identity mismatch")
        audit = _audit_v4_dataset(directory, frozen)
        require(audit["ok"], "; ".join(audit["errors"]))
        review = read_json(directory / "audit-report.json")
        required_checks = ("license_review", "cleaning_policy", "dedup_policy", "split_isolation",
                           "tokenizer_selection", "bin_reproducibility", "token_budget",
                           "contamination_audit", "manifest_completeness")
        require(review["status"] == "READY" and review["demo_only"] is False
                and not review["blockers"] and all(review["checks"][k]["status"] == "PASS" for k in required_checks)
                and all(c["status"] == "PASS" for c in review["checks"].values())
                and review["checks"]["manifest_completeness"]["detail"]["content_id"] == frozen["content_id"],
                "Pilot review is not bound to this frozen dataset identity")
        if prospective_manifest is None:
            freeze = read_json(directory / "freeze.json")
            require(freeze["content_id"] == read_json(directory / "FROZEN")["content_id"] == frozen["content_id"]
                    and freeze["manifest_sha256"] == file_info(directory / "dataset-manifest.json")["sha256"],
                    "Pilot freeze record identity mismatch")
        report.update(ok=True, status="POLICY_MATCH", content_id=frozen["content_id"],
                      plan_key=plan["plan_key"], config_sha256=plan["config_sha256"])
    except (FileNotFoundError, KeyError) as exc:
        report.update(status="BLOCKED_POLICY_EVIDENCE_MISSING", errors=[str(exc)])
    except Exception as exc:
        report["errors"] = [str(exc)]
    return report


def implementation_compatible(manifest, current):
    from .corpus_versions import V4_1M_VERSION, V4_2M_VERSION
    from .v4_language_scaling import (BASE_ID, BASE_IMPLEMENTATION, SCALING_CONTENT_ID,
                                      SCALING_IMPLEMENTATION)

    historical = {
        (V4_1M_VERSION, BASE_ID): BASE_IMPLEMENTATION,
        (V4_2M_VERSION, SCALING_CONTENT_ID): SCALING_IMPLEMENTATION,
    }
    recorded = manifest["identity"]["v4"]["implementation_sha256"]
    key = (manifest["corpus_version"], manifest["content_id"])
    return recorded == historical.get(key, current)


def audit_v4_dataset(directory, *, manifest=None, tokenizer_path=None,
                     retokenize=False, verify_inputs=False, require_source_review=False):
    """Verify the original V4 gate, plus its source and train-only evidence.

    This is an adapter, not an alternative way to freeze a corpus. It never
    accepts demo data or upgrades an unassessed/failed V4 audit.
    """
    from .corpus_v4 import verify_frozen

    try:
        frozen = verify_frozen(directory)
        if manifest is not None and manifest != frozen:
            raise CorpusError("supplied V4 manifest differs from the frozen manifest")
    except Exception as exc:
        return {"ok": False, "errors": [str(exc)], "checks": {}, "splits": {}}
    return _audit_v4_dataset(directory, frozen, tokenizer_path=tokenizer_path, retokenize=retokenize,
                             verify_inputs=verify_inputs, require_source_review=require_source_review)


def _audit_v4_dataset(directory, manifest, *, tokenizer_path=None, retokenize=False,
                      verify_inputs=False, require_source_review=False):
    from .corpus_v4 import build_registry

    directory = Path(directory)
    report = {"ok": False, "errors": [], "checks": {}, "splits": {}}
    try:
        recorded = manifest["identity"]["v4"]
        registry = manifest["license_sources"]
        registry_hash = identity({k: v for k, v in registry.items() if k != "sha256"})
        if registry_hash != registry["sha256"] or registry_hash != recorded["source_manifest_sha256"]:
            raise CorpusError("source approval evidence identity mismatch")
        if registry["license_policy"].get("approved_only") is not True:
            raise CorpusError("V4 source policy must require approved sources")
        approved = build_registry(registry["sources"], demo=False, license_policy={"approved_only": True})
        if not approved["sources"] or approved["excluded"]:
            raise CorpusError("included sources are not all approved with verified provenance")
        from .v4_sources import validate_source_plan
        source_reviews = validate_source_plan(read_json(directory / ".v4/plan.json"), registry,
                                              require=require_source_review)
        if source_reviews:
            report["checks"]["source_reviews"] = source_reviews
        report["checks"]["source_approval"] = "APPROVED"
        contamination = read_json(directory / "contamination-report.json")
        if contamination["status"] != recorded["contamination_audit_status"]:
            raise CorpusError("contamination evidence identity mismatch")
        hits = contamination["stats"]["high_risk_hits"]
        if type(hits) is not int or hits < 0:
            raise CorpusError("invalid contamination hit statistics")
        if contamination["status"] == "CHECKED_NO_HITS":
            if hits != 0:
                raise CorpusError("unresolved contamination hits")
        elif contamination["status"] != "REVIEWED" or contamination.get("decisions", {}).get(
                "unreviewed_high_risk") != 0 or type(contamination.get("decisions", {}).get(
                "unreviewed_high_risk")) is not int:
            raise CorpusError("contamination review evidence is incomplete")
        tok = MiniLLMTokenizer.load(tokenizer_path or directory / "tokenizer.json")
        if (tok.fingerprint != manifest["tokenizer"]["fingerprint"]
                or tok.fingerprint != recorded["tokenizer_fingerprint"]
                or tok.vocab_size != manifest["tokenizer"]["vocab_size"]):
            raise CorpusError("tokenizer fingerprint/vocabulary mismatch")
        selection = read_json(directory / "tokenizer-evaluation.json")["selection"]
        if (directory / "tokenizer-selection.json").is_file():
            from .v4_tokenizer_decision import verify_selection
            selection = verify_selection(directory)
        candidates = read_json(directory / "tokenizer-candidates.json")["candidates"]
        selected = next((c for c in candidates if c["fingerprint"] == tok.fingerprint
                         and c["name"] == selection["name"]), None)
        if selected is None or selection.get("mode") not in ("EXPLICIT_CONFIG", "HUMAN_REVIEWED"):
            raise CorpusError("V4 tokenizer has no explicit reviewed train-only selection")
        metadata = selected["metadata"]
        from .corpus_versions import V4_2M_VERSION, V5_VERSION

        training_origin = manifest
        if manifest["corpus_version"] in (V4_2M_VERSION, V5_VERSION):
            from .v4_language_scaling import base_manifest

            training_origin = base_manifest("train.text.jsonl", "split-manifest.json", "tokenizer.json")
        if (metadata.get("demo_only") is not False
                or metadata["train_split_sha256"] != training_origin["files"]["train.text.jsonl"]["sha256"]
                or metadata["split_manifest_sha256"] != training_origin["identity"]["v4"]["split_manifest_sha256"]
                or metadata["fingerprint"] != tok.fingerprint
                or metadata["special_token_ids"] != tok.special_ids
                or metadata["text_policy"] != recorded["text_policy"]):
            raise CorpusError("tokenizer train-only provenance/text policy mismatch")
        # The selected candidate metadata and tokenizer are also bound by the
        # original tokenizers stage; verify_frozen has verified those files.
        if file_info(Path(selected["path"])) != manifest["files"]["tokenizer.json"]:
            raise CorpusError("frozen tokenizer differs from its selected candidate")
        tok.text_policy = recorded["text_policy"]
        for name in ("train", "val", "test"):
            path = directory / f"{name}.bin"
            if path.stat().st_size % 2:
                raise CorpusError("odd-sized uint16 V4 BIN")
            tokens = np.memmap(path, dtype="<u2", mode="r")
            for start in range(0, len(tokens), 524288):
                if int(tokens[start:start + 524288].max()) >= tok.vocab_size:
                    raise CorpusError(f"{name}: token outside tokenizer vocabulary")
            report["splits"][name] = {"tokens": len(tokens)}
            del tokens
            if retokenize:
                from .corpus import read_records
                with path.open("rb") as handle:
                    for row in read_records(split_text_path(directory, name, balanced=is_language(manifest))):
                        ids = tok.encode_document(row["text"], category=row["language"])
                        expected = np.asarray(ids, dtype="<u2").tobytes()
                        if handle.read(len(expected)) != expected:
                            raise CorpusError(f"{name}: V4 retokenization differs")
                    if handle.read(1):
                        raise CorpusError(f"{name}: trailing V4 tokens")
        if verify_inputs:
            for item in read_json(directory / ".v4" / "plan.json")["inputs"]:
                verify_file(Path(item["path"]), {k: item[k] for k in ("bytes", "sha256")})
        report.update(ok=True, content_id=manifest["content_id"], corpus_version=manifest["corpus_version"])
        report["checks"].update(frozen_manifest="full", tokenizer_train_only="full",
                                contamination="cleared", retokenization="full" if retokenize else "not requested")
    except Exception as exc:
        report["errors"].append(str(exc))
    return report
