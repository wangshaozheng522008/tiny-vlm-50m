"""Disk-backed corpus stages and deterministic identities (standard library only)."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from .checkpoint import atomic_write, _sync_directory


class CorpusError(RuntimeError):
    pass


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def identity(value) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_file(path: Path, expected: dict):
    if not path.is_file() or file_info(path) != expected:
        raise CorpusError(f"file missing or SHA256/size mismatch: {path}")


def write_json(path: Path, value):
    data = json_bytes(value)
    atomic_write(path, lambda handle: handle.write(data))


def connect(path: Path, cache_mib: int = 16) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA temp_store=FILE")
    db.execute(f"PRAGMA cache_size={-cache_mib * 1024}")
    db.execute("PRAGMA foreign_keys=ON")
    return db


@contextmanager
def database_output(path: Path, *, source: Path | None = None, cache_mib: int = 16):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    temporary = Path(name)
    db = connect(temporary, cache_mib)
    try:
        if source:
            previous = connect(source, cache_mib)
            try:
                previous.backup(db, pages=256)
            finally:
                previous.close()
        yield db
        db.commit()
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise CorpusError(f"SQLite integrity check failed: {path}")
        db.close()
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        db.close()
        temporary.unlink(missing_ok=True)
        Path(str(temporary) + "-journal").unlink(missing_ok=True)


class Stages:
    """A stage marker is published last, and reused only after full hash verification."""
    def __init__(self, out: Path, work: Path, runtime: dict, log=print):
        self.out, self.work, self.runtime, self.log = out, work, runtime, log
        self.control = out / ".pipeline"
        self.control.mkdir(parents=True, exist_ok=True)
        work.mkdir(parents=True, exist_ok=True)

    def run(self, name: str, key: str, producer):
        marker = self.control / f"{name}.stage.json"
        if marker.exists():
            try:
                record = json.loads(marker.read_text())
                if record["key"] != key:
                    raise CorpusError("content/config mismatch; select a new --out")
                for filename, info in record["files"].items():
                    verify_file(Path(filename), info)
            except (OSError, ValueError, KeyError, CorpusError) as exc:
                raise CorpusError(f"stage {name}: cannot reuse verified checkpoint: {exc}") from exc
            self.log(f"[reuse] {name}: verified {len(record['files'])} artifact(s)")
            return record
        started = time.monotonic()
        self.log(f"[stage] {name}: started")
        try:
            stats, paths = producer()
            record = {"key": key, "stats": stats, "files": {str(p.resolve()): file_info(p) for p in paths},
                      "seconds": round(time.monotonic() - started, 6)}
            if "sha256" in stats and (len(paths) != 1 or record["files"][str(paths[0].resolve())] != {k: stats[k] for k in ("bytes", "sha256")}):
                raise CorpusError("streaming output checksum/size differs from the written artifact")
            write_json(marker, record)
        except Exception as exc:
            raise CorpusError(f"stage {name} FAILED: {exc}") from exc
        count = stats.get("documents", stats.get("records_seen", 0))
        self.log(f"[stage] {name}: {count:,} documents, {count / max(record['seconds'], 1e-6):,.0f} docs/s; {stats}")
        return record


def histogram_add(histogram: dict, length: int):
    bucket = str(0 if length == 0 else 1 << (length.bit_length() - 1))
    histogram[bucket] = histogram.get(bucket, 0) + 1
