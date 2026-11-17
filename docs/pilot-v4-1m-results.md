# V4 1M-token GPU pretraining pilot

## Actual status (2026-11-17)

**GPU training: NOT_RUN. Code acceptance and GPU acceptance are separate.**
The real preflight returns `BLOCKED_DATA_NOT_READY`, with
`BLOCKED_TOKENIZER_NOT_READY`, `BLOCKED_TOKEN_BUDGET` and
`BLOCKED_WORKLOAD_NOT_READY` also recorded. The machine has an NVIDIA T1200
Laptop GPU, CUDA is available, and the model preset has 86,528,640 parameters.
Hardware availability does not approve data or establish a workload result.

The independent source configuration has **0 APPROVED**, 3 REVIEW_REQUIRED,
and 0 admitted sources. `tokenizer.selected` is null. There is no frozen
`artifacts/corpus/v4-pilot-1m-v1` dataset, train-only 24,576-token Pilot tokenizer,
independent-token measurement, or matching 87M GPU workload evidence. The
historical MiniMind research versions and PR5 10M pilot are not used.

The compact actual audit is [pilot-v4-1m-report.json](pilot-v4-1m-report.json).
All GPU baseline/final Loss, PPL, throughput, memory peaks, resume checks and
capability deltas are **NOT_RUN/null**. No GPU Loss curve or estimated result is
presented as a measurement. This work does not produce a usable chat model.

## Data and trainer connection

`configs/corpus_v4_pilot_1m.yaml` copies the original admission, quality, dedup,
split, contamination and five-category policies into the independent version
`minillm-corpus-v4-pilot-1m-v1`, targeting **1,020,000** independent effective
training tokens. The formal `configs/corpus_v4.yaml` remains at **100,000,000**.
Pilot readiness never promotes that formal corpus. Source approval and explicit
tokenizer selection remain human decisions; no source is automatically approved.

`v4_admission.audit_v4_dataset()` always calls the original `verify_frozen()`.
It also checks source approval evidence and its identity, contamination decisions,
the selected train-only tokenizer metadata, special IDs, fingerprint, text policy,
and BIN token ranges. The small V4 dispatch in `corpus_audit.audit_dataset()` lets
the existing readiness and evaluation code consume V4 without the old PR3 schema
assumptions. Old corpus audit behavior is retained. Demo/unfrozen/unapproved data,
incomplete Split evidence and edited READY/FROZEN flags cannot approve a pilot.

`scripts/run_v4_training_pilot.py` only orchestrates existing components:
`probe_workload`, `check_readiness`, `launch_pilot`, `pilot_lineage`,
`evaluate_model`, `evaluate_pilot`, `training_analysis` and `evaluation_compare`.
There is no new optimization loop. Architecture, Attention/GQA, SwiGLU, RoPE,
Loss, AdamW, LR formula and checkpoint compatibility rules are unchanged.

The tokenizer must match the unchanged 24,576-vocabulary 87M preset. The workflow
defaults to random initialization, saves those weights before probing, and uses
that exact checkpoint SHA for the workload, baseline and training initialization.
The initial checkpoint is labeled `UNTRAINED_RANDOM_INITIALIZATION`. Old model
weights are not silently adopted. The workspace request binds data identity,
tokenizer, model, seed, fixed plan, evaluation policy and CPU-test/GPU scope.

## Candidate plan and workload

`configs/pilot-v4-1m-t1200.json` is an **unverified GPU candidate**, not an accepted
memory configuration: FP32, activation checkpointing, L=1024, B=1, accumulation=8,
123 updates, LR=0.0003, min LR=0.00003, warmup=0.02, seed=42, one CPU thread.

Each full update supervises `8 * (1024 - 1) = 8,184` shifted loss positions.
123 full updates give **1,006,632** positions. The existing `planned_tokens()`
recomputes this using actual complete windows, partial batches and accumulation
boundaries; unique positions are `len(PackedPretrainDataset) * (L - 1)`.
The plan must reach 1M within one update's rounding, fit within the independently
available tokens, and match `planned_effective_tokens`. Corpus repetition and
silent target reduction are rejected. Completed state counters must equal the
fixed optimizer-update and supervised-token plan.

The original probe uses real-length forward/backward, AdamW initialization,
at least three full updates, validation, and a full state save/load. It binds
model/data/tokenizer/initialization/plan/software/device UUID/driver identities
and checks allocated/reserved peaks and free memory against the existing limit
and 80% capacity cap. CPU evidence cannot approve GPU execution. No PR9 87M GPU
probe has run, so its peaks, throughput and pass/fail result remain NOT_RUN.

If L=1024 OOMs, stop. L=512 or different accumulation/precision requires a new
fixed plan, actual token-budget check, workspace/output and workload evidence.
Do not modify the parameters of an existing Resume run. There is no automatic
GPU/CPU fallback or OOM retry with changed settings.

## Commands after data approval

Choose new immutable output/report directories for every version and invocation.

```bash
# Human approvals and explicit tokenizer selection must be completed first.
python scripts/plan_corpus_v4.py --config configs/corpus_v4_pilot_1m.yaml \
  --out artifacts/corpus/v4-pilot-1m-v1

# Default: checks and reports only; returns BLOCKED with current inputs.
python scripts/run_v4_training_pilot.py

# Isolated measurement; does not add probe tokens to the pilot run.
python scripts/run_v4_training_pilot.py --probe --out reports/pr9-v4-probe

# Inspect with matched workload proof, then explicitly execute.
python scripts/run_v4_training_pilot.py --out reports/pr9-v4-ready
python scripts/run_v4_training_pilot.py --execute --out reports/pr9-v4-execute
```

The default report directory contains `preflight.json` and
`pilot-v4-1m-report.json`. A nonempty report directory is rejected. The separate
workspace `artifacts/pilot-v4-1m-v1` preserves the initial checkpoint, request,
workload proof and baseline. Probe-stage preflight checks prerequisites without
requiring a preexisting workload; execution-stage preflight requires that proof
and the existing readiness result before baseline/launch.

Baseline and final use the same frozen PR6 `pretrain-v1` and PR7
`chinese-foundation-v1` candidates/prompts and evaluation configuration. They
measure fixed Validation loss/PPL, natural-prefix Chinese/math/logic likelihood
and fixed Chinese continuations. Full comparison policies must match. Test is
not used for checkpoint selection or tuning. Test assets are never added to
training. The original benchmarks are provisional diagnostics, not evidence of
general ability. Loss improvement alone does not establish capability gain.

For Resume, copy the fixed training JSON, add
`"resume": "runs/pilot-v4-1m-t1200/state.pt"`, and retain every plan parameter,
the original workspace and initial checkpoint. Use `--training-config` to select
that JSON and a new `--out` report directory. The original lineage validates
run ownership, original initialization, data/tokenizer identity, optimizer/RNG,
cursor, LR schedule and counters; corrupt state is rejected without silent
backup loading. Existing state rotation remains unchanged.

If hardware/software changes invalidate a workload proof, set `workload_report`
in the Resume JSON to a **new** proof directory and explicitly run `--probe`.
This probes the original initial weights in isolation, not the in-progress
state. It preserves state.pt and the old proof. Then recheck and explicitly
resume. Changing Seq Len/LR/Seed still fails the original fixed-plan identity.

## CPU verification

The ordinary suite runs offline and does not require T1200. CPU-only fixture
scope requires `--cpu-nano-test`, the unchanged Nano preset (fixture vocab size
only), explicit CPU, and the same full formal V4 freeze/admission checks. It
cannot approve DEMO_ONLY data, CUDA, or an 87M run. Custom frozen fixture suites
are accepted only in this test scope (`--test-suite`, `--evaluation-config`).

The persisted end-to-end regression isolates CUDA with `CUDA_VISIBLE_DEVICES=''`.
It exercises preflight, a real three-update CPU workload, existing readiness,
random-weight baseline, training, interrupted and completed Resume, final
evaluation, comparison and engineering report publication. The Nano fixture
plan is 4 updates / **120 supervised tokens**, interrupted at 2 / **60**.
Continuous and resumed model/AdamW/RNG/cursor/counter/LR results match exactly.
These fixture metrics are not GPU pilot results.

New tests cover legal frozen admission; demo/unapproved/rejected/invalid approval
evidence; missing freeze/Split evidence; tokenizer/vocabulary mismatches; missing
independent budget; default no-execution; unavailable GPU; missing/damaged/stale
workload; changed Seq Len/LR/Seed; changed data/tokenizer/corrupt Resume; frozen
evaluation consistency; protected historical outputs; deterministic token
accounting; exact recovery; and proof refresh without state/proof replacement.

Actual validation used the repository `.venv` Python 3.11.17:

| Check | Result |
| --- | --- |
| `python -m pytest -q -o addopts=''` | **631 passed**, exit 0 |
| Three new V4 pilot test files | **29 passed**, exit 0 |
| Existing readiness/workload/staged-evaluation plus initial pilot regressions | **75 passed**, exit 0 |
| CUDA-isolated end-to-end training/resume/proof-refresh regressions | **3 passed**, exit 0 |
| `./scripts/smoke_test.sh` | PASS, exit 0 |
| `SYNTHETIC=1 ./scripts/smoke_test.sh` | PASS, exit 0 |
| `python scripts/check_training_reliability.py --out reports/pr9-v4-validation/reliability` | PASS, exit 0; model/optimizer differences 0, sample/RNG/LR state matches |
| Real V4 pilot preflight | `BLOCKED_DATA_NOT_READY`, exit 1 as intended; training NOT_RUN |

Logs and full CPU fixture artifacts are preserved locally under
`reports/pr9-v4-validation/` (`pytest-final.log`, `new-regressions-final.log`,
`cpu-nano-final.log`, `cpu-nano-final/`, `smoke.log`, `smoke-synthetic.log`,
`reliability.log`, `preflight-release.log`, `source-status.json`). Reproduce the
CUDA-isolated flow with:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q -o addopts='' \
  tests/test_v4_training_pilot.py::test_cpu_nano_end_to_end_probe_readiness_train_resume_evaluation \
  tests/test_v4_pilot_resume.py::test_v4_continuous_and_interrupted_resume_match_optimizer_rng_and_cursor \
  tests/test_v4_pilot_resume.py::test_resume_can_refresh_workload_in_new_directory_without_touching_state_or_old_proof
```

The default smoke checks the historical CPU Nano pipeline with locally available
inputs. Neither smoke run's data, checkpoints or metrics are V4 GPU pilot assets.
No historical PR5 acceptance metrics were substituted for this experiment.

## Blockers, risks and next step

- **P0 data:** no approved Pilot sources or frozen version. Supply documented
  human approvals and enough independent Chinese-first five-category data, then
  run the unchanged V4 pipeline through full audit/freeze.
- **P0 tokenizer:** no selected/frozen train-only V4 24,576-token tokenizer.
  Fit only on admitted Train, review/select it explicitly and freeze its evidence.
- **P1 GPU acceptance:** no exact-plan 87M workload or actual PR9 training,
  GPU recovery, baseline/final convergence or capability evidence. Only a matched
  passing T1200 probe plus READY and `--execute` can release this gate.
- **P2 interpretation:** LSH still has probabilistic recall, the frozen capability
  probes are provisional, and cross-software/hardware bitwise recovery is not
  promised. Full runtime reports are local checksum-bound evidence, not signed
  third-party attestations. Sparse logging/short pilots may leave trends or total
  timing inconclusive; nulls are preserved instead of extrapolated.

No new architecture/trainer P0/P1 issue was found in this change. Existing PR5
datasets, models, checkpoints and reports, formal 100M targets, old tokenizers,
and PR6/PR7 frozen assets were not modified. The next step is clearing data and
tokenizer admission, then measuring the bounded 1M plan. Larger training should
wait for actual GPU engineering/recovery and evaluation evidence plus human
review of compute/data budgets; this PR does not start a longer run.
