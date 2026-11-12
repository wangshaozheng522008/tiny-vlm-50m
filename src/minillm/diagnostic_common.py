"""Small, bounded diagnostic runs and content identities; no training algorithms."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import signal
import resource
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .assessment_io import environment, records
from .training_manifest import file_sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
MEMORIZATION = ROOT / 'data/diagnostics/memorization-v1'
FOUNDATION = ROOT / 'data/evaluation/chinese-foundation-v1'


@dataclass(frozen=True)
class Budget:
    max_steps: int = 200
    max_effective_tokens: int = 50000
    max_wall_seconds: float = 120
    max_memory_mib: int = 2048

    def validate(self):
        if (type(self.max_steps) is not int or not 2 <= self.max_steps <= 1000
                or type(self.max_effective_tokens) is not int or not 126 <= self.max_effective_tokens <= 2_000_000
                or not math.isfinite(self.max_wall_seconds) or not 1 <= self.max_wall_seconds <= 600):
            raise ValueError('diagnostic budgets: steps [2,1000], effective tokens [126,2000000], wall seconds [1,600]')
        if type(self.max_memory_mib) is not int or not 128 <= self.max_memory_mib <= 8192:
            raise ValueError('CPU additional virtual memory budget must be 128..8192 MiB')


@contextlib.contextmanager
def cpu_memory_limit(additional_mib):
    """Bound allocations beyond imported PyTorch's existing virtual address space.

    This is RLIMIT_AS, not an RSS promise: PyTorch/CUDA libraries already reserve
    address space at import. Child correctness processes inherit this ceiling.
    """
    current_kib = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                           if line.startswith('VmSize:')))
    previous = resource.getrlimit(resource.RLIMIT_AS)
    cap = current_kib*1024 + additional_mib*1024**2
    if previous[0] != resource.RLIM_INFINITY:
        cap = min(cap, previous[0])
    resource.setrlimit(resource.RLIMIT_AS, (cap, previous[1]))
    try:
        yield {'kind':'RLIMIT_AS','baseline_virtual_bytes':current_kib*1024,
               'additional_mib':additional_mib,'absolute_virtual_limit_bytes':cap,
               'interpretation':'Virtual address-space ceiling, including inherited child limit; not an RSS cap.'}
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)


@contextlib.contextmanager
def wall_deadline(seconds):
    """POSIX wall watchdog; interrupted fits retain only their completed checkpoints."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('positive finite wall deadline required')
    def expired(signum, frame):
        raise TimeoutError('diagnostic wall budget exhausted')
    previous = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous)


def new_output(path):
    path = Path(path).resolve()
    # Atomic ownership claim; even an existing empty directory is not reused.
    path.mkdir(parents=True, exist_ok=False)
    return path


def code_identity():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    files = sorted([*ROOT.glob('src/minillm/*.py'), *ROOT.glob('scripts/*.py')])
    hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in files}
    return {'git_commit': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
            'implementation_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            'source_files': hashes}


def base_manifest(seed, budget, device):
    return {'schema_version': 1, 'seed': seed, 'budget': asdict(budget),
            'code': code_identity(), 'environment': environment(torch.device(device)),
            'created_unix': time.time(), 'interpretation': 'Independent engineering, memorization, and provisional evaluation dimensions; no combined capability score.'}


def file_identity(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': file_sha256(path), 'bytes': path.stat().st_size}


def model_identity(model, checkpoint):
    return {'parameters': model.num_parameters(), 'config': model.cfg.to_dict(),
            'checkpoint': file_identity(checkpoint)}


def frozen_rows(path):
    from .evaluation import task_asset_provenance
    rows = list(records(path))
    provenance = task_asset_provenance(path, len(rows))
    if provenance['status'] != 'FROZEN':
        raise ValueError('diagnostics require a frozen asset declaration')
    return rows


def write_failure(out, exc):
    write_json(Path(out) / 'failure.json', {'status': 'ENGINEERING_FAIL', 'exception': type(exc).__name__,
                                          'reason': str(exc), 'capability_conclusion': 'NOT_ESTABLISHED'})
