# 87M T1200 pilot workflow

This workflow uses the existing pretrain checkpoint and original 24,576-token tokenizer. It starts a new weights-only task: AdamW, LR schedule and this-run token counters restart. Legacy state.pt is never resumed. No model, optimizer formula, training cursor or data-pipeline semantics were changed.

## Reproduce the data candidate

The user confirmed noncommercial research use. The upstream MiniMind snapshot declares Apache-2.0 and CC-BY-NC-2.0; individual records have no license/source mapping. The combined label records that upstream declaration, not an independent permission audit or a commercial-use assurance.

Input: existing pretrain_t2t_mini.jsonl, revision 312afb4f76391145c6902f765bb51691c09a12f5, SHA256 6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c. No new corpus was downloaded. The source guard skips the first 40,000 raw records used by the historical experiment, scans the next 200,000 records, and excludes exact and recognized near matches to all 24,000 known old Train/Val/Test texts. The corpus pipeline then deterministically selects up to 128,000 deduplicated parents. This is a contiguous pilot sample, not a representative sample of all upstream data.

```bash
python scripts/prepare_pilot_source.py \
  --input data/minimind/pretrain_t2t_mini.jsonl \
  --historical runs/minillm-87m/corpus/train.text.jsonl \
    runs/minillm-87m/corpus/val.text.jsonl runs/minillm-87m/corpus/test.text.jsonl \
  --skip-records 40000 --take-records 200000 \
  --out data/pilot-sources/minimind-unexposed-v2
python scripts/prepare_text.py --config configs/pilot-corpus.research.json \
  --out data/processed/minillm-pilot-research-v3 \
  --tokenizer runs/minillm-87m/corpus/tokenizer.json
python scripts/audit_dataset.py --dataset data/processed/minillm-pilot-research-v3 \
  --retokenize --verify-inputs --report reports/pr5-pilot-v3/dataset-audit.json
```

The source guard uses disk-indexed category-neutral hashes before/after cleaning and full-text SimHash64 Hamming<=4 plus 4-gram Jaccard>=.85 (including code). Candidate overflow fails instead of truncating coverage. It atomically publishes source.jsonl and historical-exclusion.json with input/output hashes, counts and scope. It cannot exclude arbitrary substrings, semantic rewrites or unknown historical exposure. Independently recompute exact overlap between the final emitted splits and all historical splits; the required count is zero.

The first v1 attempt was stopped at 65 updates / 1,063,920 tokens after historical overlap was found: old Train/new Val360, old Train/new Test369, old Val/new Train378, old Test/new Train364. Its within-version audit passed, which was insufficient for a warm start. Preserve that version and its checkpoints, exclude it from acceptance, and restart the final v3 experiment from the original weights with fresh optimizer/counters. Do not resume v1 into v3.

The historical-exclusion source is named v2 and is reused byte-for-byte. The first corpus from that source (dataset v2, sample 120,000) had 14,967,513 effective Train tokens and was not promoted because it missed the 15M minimum. Dataset v3 uses 128,000 parents and is checked against that minimum before any training. No v2 weights are reused.

Published versions are immutable. Reuse a verified version or choose a new directory; do not edit it. Freeze the existing tokenizer. Train/Val/Test exact and recognized-group isolation is audited by PR3. PR4 contamination scanning of known text records is also required before training; it does not exhaustively exclude substrings, semantic rewrites or unknown historical sources.

## Measure a representative load

```bash
python scripts/probe_training_workload.py \
  --dataset data/processed/minillm-pilot-research-v3 \
  --tokenizer runs/minillm-87m/corpus/tokenizer.json --config minillm-87m \
  --training-config configs/pilot-t1200-10m.json \
  --out reports/pr5-pilot-v3/workload-1024
```

The probe uses an isolated model and AdamW, real fixed-window batches, full sequence length and gradient accumulation. It executes at least three successful updates, validation with the requested limit (0=full), and complete-state save/load in a disposable directory. It restores caller RNG states and thread count, preserves source weights, and does not create the training output directory. CPU Nano is supported for CI; CPU evidence cannot approve a GPU budget.

CUDA records both peak allocated and reserved bytes, device total and initially free memory. PASS requires both peaks below the smaller of the requested limit and 80% of visible device memory, with that budget initially free. This covers first AdamW state allocation, later updates, validation and saving; it does not guarantee future training never OOMs. For this machine, test Seq Len 1024, then 512/256 only if the longer configuration fails. Freeze the longest passing configuration before training; B=1, accumulation=16, FP32, activation checkpointing, one CPU thread, seed42.

workload.json binds the exact training manifest, data hashes/paths, model config, initial checkpoint SHA, tokenizer, precision/checkpointing, validation limit, device UUID/driver/software, and relevant implementation hashes. It is a local measurement record with a checksum, not a signed third-party attestation. Changed conditions require a new probe; relocated datasets or another GPU cannot reuse the old report.

## Inspect, verify recovery, then explicitly execute

```bash
python scripts/run_pretraining_pilot.py \
  --dataset data/processed/minillm-pilot-research-v3 \
  --tokenizer runs/minillm-87m/corpus/tokenizer.json --config minillm-87m \
  --training-config configs/pilot-t1200-10m.json --out reports/pr5-pilot-v3/inspect
python scripts/check_pilot_recovery.py \
  --dataset data/processed/minillm-pilot-research-v3 \
  --tokenizer runs/minillm-87m/corpus/tokenizer.json --config minillm-87m \
  --training-config configs/pilot-t1200-10m.json --out reports/pr5-pilot-v3/recovery-20
```

The first command only inspects. The recovery command uses separate output directories for 20 continuous updates and 20 updates with interruptions after 5 and 12 (SIGTERM/SIGINT). It compares complete sample order, LR sequence, counters and parameters with atol1e-6/rtol1e-5. Recovery-experiment tokens are not added to pilot training totals.

Evaluate the original pretrain model on the published Validation split, the frozen 160 tasks and 40 prompts before starting. Use the same evaluation config after training; do not use Test to tune decisions. The source checkpoint's historical counters are not the new pilot counters.

```bash
python scripts/run_pretraining_pilot.py \
  --dataset data/processed/minillm-pilot-research-v3 \
  --tokenizer runs/minillm-87m/corpus/tokenizer.json --config minillm-87m \
  --training-config configs/pilot-t1200-10m.json \
  --out reports/pr5-pilot-v3/launch --execute
```

The launcher uses one config for readiness and fit, and requires matching full-load evidence for CUDA execution. It preserves READY/BLOCKED/INCOMPLETE exit codes 0/1/2; unavailable required CUDA never falls back to CPU training. A launch-manifest is atomically published before training and records initialization SHA, configuration and data identities. Inputs are rehashed after readiness. It appends an execution result after fit and records pilot-lineage.json inside the owned run, linked to its exact training manifest.

Resume uses the same fixed plan, removes init_from, and sets resume to the new run's state.pt. Keep the original probe and use a new launch-report directory. Lineage is required; SIGKILL before the launcher can write lineage needs explicit provenance recovery from its durable launch record, not guessed initialization. Changing steps/schedule is still not resume; extend is not implemented.

After completion, run the existing training analysis, checkpoint comparison and pilot acceptance tools. Final Test evaluation is separate. Stop and diagnose engineering failures; capability improvement is not an absolute-loss threshold or permission to launch one billion tokens. Keep generated reports/data/checkpoints local; only compact evidence and configurations belong in Git.

## Completed pilot

See [pilot-results.md](pilot-results.md) for the actual 611-update / 10,000,848-token run, independent post-run verification, fixed-budget baseline/final/Test results and limitations. Compact machine-readable evidence is in [pilot-evidence.json](pilot-evidence.json). Engineering passed and validation improved; Raw/Chat exact scores stayed zero, so capability improvement is inconclusive. No further training was started.
