"""Frozen V4 evidence adapter shared by readiness, evaluation and the V4 pilot."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .assessment_io import read_json
from .corpus_utils import CorpusError, file_info, identity, verify_file
from .tokenizer import MiniLLMTokenizer


def audit_v4_dataset(directory, *, manifest=None, tokenizer_path=None,
                     retokenize=False, verify_inputs=False):
    """Verify the original V4 gate, plus its source and train-only evidence.

    This is an adapter, not an alternative way to freeze a corpus. It never
    accepts demo data or upgrades an unassessed/failed V4 audit.
    """
    from .corpus_v4 import build_registry, verify_frozen

    directory = Path(directory)
    report = {"ok": False, "errors": [], "checks": {}, "splits": {}}
    try:
        frozen = verify_frozen(directory)
        if manifest is not None and manifest != frozen:
            raise CorpusError("supplied V4 manifest differs from the frozen manifest")
        manifest = frozen
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
        candidates = read_json(directory / "tokenizer-candidates.json")["candidates"]
        selected = next((c for c in candidates if c["fingerprint"] == tok.fingerprint
                         and c["name"] == selection["name"]), None)
        if selected is None or selection.get("mode") != "EXPLICIT_CONFIG":
            raise CorpusError("V4 tokenizer has no explicit reviewed train-only selection")
        metadata = selected["metadata"]
        if (metadata.get("demo_only") is not False
                or metadata["train_split_sha256"] != manifest["files"]["train.text.jsonl"]["sha256"]
                or metadata["split_manifest_sha256"] != recorded["split_manifest_sha256"]
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
                    for row in read_records(directory / f"{name}.text.jsonl"):
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
