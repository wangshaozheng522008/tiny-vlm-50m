"""Read-only input identities and transactional, non-overwriting report publication."""
from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path

import torch

from .checkpoint import _sync_directory
from .corpus_utils import file_info, identity
from .training_manifest import write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def records(path):
    with Path(path).open(encoding="utf-8") as handle:
        line_number = 0
        while line := handle.readline(16 * 1024 * 1024 + 1):
            line_number += 1
            if not line.strip():
                continue
            if len(line) > 16 * 1024 * 1024:
                raise ValueError(f"{path}:{line_number}: record exceeds 16 MiB")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield row


def environment(device):
    return {"python_version": platform.python_version(), "torch_version": torch.__version__,
            "platform": platform.platform(), "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "cuda_version": torch.version.cuda, "threads": torch.get_num_threads(),
            "numerical_scope": "same software/device; cross-GPU/version bitwise equality is not guaranteed"}


@contextlib.contextmanager
def report_directory(out):
    """Publish the directory only after every artifact is complete."""
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"{out}: non-empty output; choose a new report directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        yield temporary
        for root, _, files in os.walk(temporary, topdown=False):
            for name in files:
                with (Path(root)/name).open("rb") as handle:
                    os.fsync(handle.fileno())
            _sync_directory(Path(root))
        if out.exists():
            out.rmdir()
        os.rename(temporary, out)
        _sync_directory(out.parent)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def publish_manifest(directory, name, payload):
    files = {p.name: file_info(p) for p in directory.iterdir() if p.is_file()}
    write_json(directory / name, {**payload, "files": files})


def verified_evaluation(directory):
    directory = Path(directory)
    manifest = read_json(directory / "evaluation-manifest.json")
    if manifest["schema_version"] != 1 or identity(manifest["identity"]) != manifest["evaluation_id"]:
        raise ValueError("invalid evaluation identity")
    if (manifest["comparison_policy"] != manifest["identity"]["policy"]
            or manifest["checkpoint_sha256"] != manifest["identity"]["checkpoint_sha256"]
            or manifest["tokenizer_fingerprint"] != manifest["comparison_policy"]["tokenizer_fingerprint"]):
        raise ValueError("evaluation manifest contains inconsistent policy/checkpoint identities")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or file_info(directory / name) != expected:
            raise ValueError(f"evaluation artifact damaged: {name}")
    if not {"evaluation.json", "evaluation.md", "predictions.jsonl"} <= manifest["files"].keys():
        raise ValueError("incomplete evaluation report")
    report = read_json(directory / "evaluation.json")
    if report["evaluation_id"] != manifest["evaluation_id"]:
        raise ValueError("evaluation report/manifest identity mismatch")
    if report["checkpoint"]["sha256"] != manifest["checkpoint_sha256"]:
        raise ValueError("evaluation checkpoint identity mismatch")
    json.dumps(report, allow_nan=False)
    return report, manifest
