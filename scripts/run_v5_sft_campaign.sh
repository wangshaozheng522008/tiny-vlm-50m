#!/usr/bin/env bash
set -Eeuo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
python="$root/.venv/bin/python"
artifacts="$root/artifacts/v5-language-sft-5k-v1"
work="$artifacts/campaign-v1"
runtime="$artifacts/local-inference"
models="$work/models"
reports="$root/reports/v5-language-sft-5k-v1"
runs="$root/runs/v5-language-sft-5k-v1"
teacher="$runtime/Qwen3-4B-Q4_K_M.gguf"
reviewer="$models/Qwen3.5-4B-Q4_K_M.gguf"
mkdir -p "$work"
exec 9>"$work/.supervisor.lock"
flock -n 9 || exit 75
server_pid=
phase=setup

mark_phase() {
  phase="$1"
  printf '%s %s\n' "$(date -Is)" "$phase" > "$work/phase.tmp"
  mv "$work/phase.tmp" "$work/phase"
  printf '[%s] %s\n' "$(date -Is)" "$phase"
}

stop_server() {
  if [ -n "$server_pid" ]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    for attempt in $(seq 1 20); do
      if ! kill -0 "$server_pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -KILL "$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
    server_pid=
  fi
}

on_exit() {
  code=$?
  stop_server
  if [ "$code" -ne 0 ]; then
    printf '%s FAILED %s exit=%s\n' "$(date -Is)" "$phase" "$code" > "$work/phase.tmp"
    mv "$work/phase.tmp" "$work/phase"
  fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_server() {
  model="$1"
  port="$2"
  context="$3"
  slots="$4"
  logfile="$5"
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    printf 'Port %s already serves another inference process\n' "$port" >&2
    return 1
  fi
  noble="$runtime/noble/usr/lib/x86_64-linux-gnu"
  bin="$runtime/runtime/llama-b12810"
  cuda="$runtime/runtime/cudart-llama-b12810-bin-ubuntu-cuda-12.8-x64"
  (
    cd "$bin"
    exec env GGML_CUDA_DISABLE_GRAPHS=1 "$noble/ld-linux-x86-64.so.2" \
       --library-path "$noble:$bin:$cuda" "$bin/llama-server" -m "$model" \
       --host 127.0.0.1 --port "$port" -c "$context" -np "$slots" -b 256 -ub 256 -ngl 99 -fa off -t 4
  ) >> "$logfile" 2>&1 &
  server_pid=$!
  for attempt in $(seq 1 90); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      return
    fi
    kill -0 "$server_pid"
    sleep 2
  done
  printf 'Local inference did not become ready on port %s\n' "$port" >&2
  return 1
}

audit() {
  "$python" -u scripts/audit_v5_sft_local.py "$work" --previous "$artifacts/offline-batches" \
     --reviewer "$reviewer" --embedding "$models/bge-small-zh-v1.5" --stage "$1"
}

train_stage() {
  command="$1"
  directory="$2"
  if [ -f "$directory/state.pt" ]; then
    "$python" -u scripts/run_v5_language_sft_5k.py "$command" --resume state.pt
  else
    "$python" -u scripts/run_v5_language_sft_5k.py "$command"
  fi
}

if [ ! -f "$work/collected/manifest.json" ]; then
  mark_phase generation
  start_server "$teacher" 18144 8192 8 "$work/generator-server.log"
  "$python" -u scripts/generate_v5_sft_local.py "$work/generation" --weights "$teacher"
  stop_server
  mark_phase collection
  audit collect
  exec /bin/bash "$0" 9>&-
fi
if [ ! -f "$artifacts/frozen/manifest.json" ]; then
  mark_phase review_calibration
  start_server "$reviewer" 18145 12288 6 "$work/reviewer-server.log"
  audit calibrate
  mark_phase independent_review
  audit review
  stop_server
  mark_phase semantic_screen
  PYTHONPATH="$work/deps${PYTHONPATH:+:$PYTHONPATH}" audit semantic
  mark_phase semantic_pair_review
  start_server "$reviewer" 18145 12288 6 "$work/reviewer-server.log"
  audit semantic-review
  stop_server
  mark_phase freeze
  audit freeze
fi
if [ ! -f "$reports/diagnostic-result/manifest.json" ]; then
  mark_phase diagnostic
  train_stage diagnose "$runs/diagnostic"
fi
if [ ! -f "$reports/training-complete.json" ]; then
  mark_phase training
  train_stage train "$runs/formal"
fi
mark_phase unseen_generation
"$python" -u scripts/evaluate_v5_sft_unseen.py generate
mark_phase unseen_review
start_server "$reviewer" 18145 12288 6 "$work/reviewer-server.log"
"$python" -u scripts/evaluate_v5_sft_unseen.py review
stop_server
mark_phase COMPLETE
