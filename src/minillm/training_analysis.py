"""Analyze PR2 logs in append order without hiding conflicting resume records."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .assessment_io import read_json, records, report_directory
from .checkpoint import load_training_state
from .corpus_utils import file_info
from .training_manifest import write_json


@dataclass(frozen=True)
class TrendConfig:
    window: int = 3
    smoothing: int = 1
    loss_delta: float = .02
    plateau_delta: float = .01
    gradient_limit: float = 100.
    throughput_drop: float = .4
    jump_loss: float = 2.

    def validate(self):
        if self.window < 2 or self.smoothing < 1:
            raise ValueError("window >= 2, smoothing >= 1 required")
        if any(not math.isfinite(v) or v < 0 for k, v in asdict(self).items() if k not in ("window","smoothing")):
            raise ValueError("trend thresholds must be finite and nonnegative")
        if not 0 < self.throughput_drop < 1:
            raise ValueError("throughput_drop must be between 0 and 1")


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(x) for x in value.values())
    if isinstance(value, list):
        return all(finite_tree(x) for x in value)
    return not isinstance(value, float) or math.isfinite(value)


def safe_tree(value):
    if isinstance(value, dict):
        return {k:safe_tree(v) for k,v in value.items()}
    if isinstance(value, list):
        return [safe_tree(v) for v in value]
    return str(value) if isinstance(value, float) and not math.isfinite(value) else value


def trends(train, validation, cfg):
    result = []
    def smoothed(values):
        return [statistics.mean(values[max(0,i-cfg.smoothing+1):i+1]) for i in range(len(values))]
    w = cfg.window
    vals = smoothed([v["loss"] for v in validation])
    tr = smoothed([r["loss"] for r in train])
    if len(vals) >= w*2:
        old, new = statistics.mean(vals[-2*w:-w]), statistics.mean(vals[-w:])
        if new < old-cfg.loss_delta:
            result.append("VALIDATION_DECREASING")
        elif abs(new-old) <= cfg.plateau_delta:
            result.append("VALIDATION_PLATEAU")
        if len(tr) >= w*2 and new > old+cfg.loss_delta and statistics.mean(tr[-w:]) < statistics.mean(tr[-2*w:-w])-cfg.loss_delta:
            result.append("TRAIN_DOWN_VALIDATION_UP")
    if len(train) >= w and all(r.get("gradient_norm", 0) > cfg.gradient_limit for r in train[-w:]):
        result.append("SUSTAINED_HIGH_GRADIENT")
    speeds = [r["tokens_per_second"] for r in train if r.get("tokens_per_second") is not None]
    if len(speeds) >= 2*w and statistics.mean(speeds[-w:]) < statistics.mean(speeds[-2*w:-w])*(1-cfg.throughput_drop):
        result.append("THROUGHPUT_DROP")
    if any(abs(a["loss"]-b["loss"]) > cfg.jump_loss for a,b in zip(train,train[1:])):
        result.append("LOSS_JUMP")
    return result or ["INSUFFICIENT_EVIDENCE" if len(vals)<2*w else "NO_RULE_TRIGGERED"]


def analyze_training(run, out, config=None):
    cfg = config or TrendConfig(); cfg.validate()
    run = Path(run)
    errors, warnings, inputs, logs = [], [], {}, []
    artifacts = {}
    for name in ("training-manifest.json", "metrics.json", "train-summary.json"):
        try:
            artifacts[name] = read_json(run/name); inputs[name] = file_info(run/name)
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
    manifest, summary = artifacts.get("training-manifest.json",{}), artifacts.get("train-summary.json",{})
    if not isinstance(manifest,dict):
        errors.append("training manifest must be an object");manifest={}
    if not isinstance(summary,dict):
        errors.append("training summary must be an object");summary={}
    if not all(k in manifest for k in ("training","dataset","model_config","tokenizer_fingerprint","run_id")):
        errors.append("incomplete training manifest")
    try:
        inputs["training-log.jsonl"] = file_info(run/"training-log.jsonl")
        for record in records(run/"training-log.jsonl"):
            logs.append(record)
    except (OSError, ValueError) as exc:
        errors.append(f"training-log.jsonl: {exc}")
    train, validation, checkpoints, seen, duplicates = [], [], [], {}, 0
    previous_step, previous_tokens = 0, 0
    for line, record in enumerate(logs, 1):
        if not finite_tree(record):
            errors.append(f"line {line}: nonfinite number")
            continue
        event = record.get("event")
        if event == "train":
            try:
                step, tokens = record["global_step"], record["tokens_seen"]
                if type(step) is not int or type(tokens) is not int or step <= 0 or tokens <= 0:
                    raise ValueError("invalid progress")
                if step in seen:
                    if record == seen[step]:
                        duplicates += 1; continue
                    errors.append(f"line {line}: conflicting duplicate step {step}")
                    continue
                if step <= previous_step or tokens <= previous_tokens:
                    errors.append(f"line {line}: step/token rewind after resume")
                if record["loss_tokens"] != tokens-previous_tokens:
                    errors.append(f"line {line}: missing training token/log coverage")
                for key in ("loss","loss_tokens","learning_rate","gradient_norm","tokens_per_second","step_seconds"):
                    value=record[key]
                    if isinstance(value,bool) or not isinstance(value,(int,float)) or value<0:
                        raise ValueError(f"invalid {key}")
                if record["tokens_per_second"]==0 or record["loss_tokens"]<=0:
                    raise ValueError("zero token count/throughput")
                previous_step, previous_tokens = step, tokens
                seen[step] = record; train.append(record)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"line {line}: incomplete train record: {exc}")
        elif event == "validation":
            if type(record.get("global_step")) is not int or not isinstance(record.get("loss"), (int,float)):
                errors.append(f"line {line}: invalid validation record")
            else:
                step=record["global_step"]
                if validation and step<validation[-1]["step"]:
                    errors.append(f"line {line}: validation step rewind")
                if validation and step==validation[-1]["step"]:
                    warnings.append(f"line {line}: revalidation of step {step}; latest observation used for trends")
                    validation[-1]={**record,"step":step}
                else:
                    validation.append({**record,"step":step})
        elif event == "checkpoint":
            if type(record.get("global_step")) is not int or record["global_step"]<0:
                errors.append(f"line {line}: invalid checkpoint step");continue
            if checkpoints and record.get("global_step",-1)<checkpoints[-1].get("global_step",0):
                errors.append(f"line {line}: checkpoint step rewind")
            checkpoints.append(record)
            if record.get("status") != "saved":
                errors.append(f"line {line}: checkpoint failed")
        else:
            warnings.append(f"line {line}: unrecognized event {event}")
    metrics = artifacts.get("metrics.json", [])
    if not isinstance(metrics, list) or not finite_tree(metrics):
        errors.append("invalid/nonfinite metrics.json"); metrics = []
    for record in metrics:
        if not isinstance(record, dict) or type(record.get("step")) is not int or not isinstance(record.get("loss"),(int,float)):
            errors.append("incomplete validation metric")
        elif not any(v["step"]==record["step"] and v["loss"]==record["loss"] for v in validation):
            errors.append(f"validation metric step {record['step']} missing from append log")
    if summary:
        if not finite_tree(summary):
            errors.append("nonfinite summary")
        if summary.get("tokens_seen") != previous_tokens or summary.get("global_step") != previous_step:
            errors.append("summary progress disagrees with training log")
    state_status = "NOT_RUN"
    try:
        state = load_training_state(run/"state.pt", mmap=True)
        state_status = "PASS"; inputs["state.pt"] = file_info(run/"state.pt")
        if state["manifest"] != manifest or state["global_step"] != summary.get("global_step") or state["tokens_seen"] != summary.get("tokens_seen"):
            errors.append("checkpoint/manifest/summary mismatch"); state_status="FAIL"
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        state_status="FAIL"; errors.append(f"checkpoint: {exc}")
    valid_steps = {r["global_step"]:r for r in train}
    validation_rows=[]
    for v in validation:
        step=v["step"];tokens=valid_steps.get(step,{}).get("tokens_seen")
        axis_source="training log"
        if tokens is None and manifest.get("mode")=="pretrain":
            try:
                plan=manifest["training"];samples=manifest["dataset"]["train"]["samples"]
                epochs,remainder=divmod(step,plan["optimizer_updates_per_epoch"])
                tokens=(epochs*samples+min(remainder*plan["batch_size"]*plan["grad_accum"],samples))*(plan["seq_len"]-1)
                axis_source="exact fixed-window plan derivation"
            except (KeyError,TypeError,ZeroDivisionError):
                axis_source="UNKNOWN"
        elif tokens is None:
            axis_source="UNKNOWN; sparse SFT logs do not expose this exact token cursor"
        validation_rows.append({"global_step":step,"tokens_seen":tokens,"loss":v["loss"],
                                "perplexity":v.get("perplexity"),"axis_source":axis_source})
    rows = []
    for r in train:
        val = [v["loss"] for v in validation if v["step"]==r["global_step"]]
        rows.append({**{k:r.get(k) for k in ("global_step","tokens_seen","loss","learning_rate","gradient_norm","tokens_per_second","step_seconds")},
                     "validation_loss":val[-1] if val else None,
                     "validation_train_gap":val[-1]-r["loss"] if val else None})
    intervals = [b.get("global_step",0)-a.get("global_step",0) for a,b in zip(checkpoints,checkpoints[1:])]
    for name,info in inputs.items():
        if file_info(run/name)!=info:
            errors.append(f"{name} changed during analysis; use an immutable run snapshot")
    report = {"schema_version":1,"status":"ISSUES" if errors else "OK","errors":errors,"warnings":warnings,
              "input_files":inputs,"training_manifest":manifest,"summary":safe_tree(summary),
              "raw_records":safe_tree(logs),"curve_rows":rows,"validation_rows":validation_rows,"trends":trends(train, validation,cfg),
              "configuration":asdict(cfg),"identical_duplicate_records":duplicates,
              "checkpoint":{"status":state_status,"step_intervals_in_append_order":intervals},
              "coverage":"loss_tokens must cover cumulative token differences; sparse logging is allowed; no sorting to conceal rewinds",
              "axis":"tokens_seen primary, global_step auxiliary; losses are recorded token-weighted intervals",
              "timing_scope":"summary.seconds is the last process segment; sum logged loss_tokens / tokens_per_second estimates logged wall time including overhead",
              "validation_with_unlogged_step":[v["step"] for v in validation if v["step"] not in valid_steps]}
    with report_directory(out) as directory:
        (directory/"curves").mkdir()
        with (directory/"curves/training.csv").open("w",newline="") as handle:
            fields=list(rows[0]) if rows else ["global_step","tokens_seen","loss"]
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        with (directory/"curves/validation.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=["global_step","tokens_seen","loss","perplexity","axis_source"])
            writer.writeheader();writer.writerows(validation_rows)
        write_json(directory/"training-analysis.json",report)
        (directory/"training-analysis.md").write_text("# Training analysis\n\nStatus: "+report["status"]+
            "\n\nTrends: "+", ".join(report["trends"])+"\n\nErrors:\n\n"+"\n".join("- "+e for e in errors)+
            "\n\nRaw records are retained in JSON. CSV uses cumulative effective tokens and global steps. Rules are observations, not automatic training decisions.\n")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",required=True);p.add_argument("--out",required=True)
    p.add_argument("--config")
    args=p.parse_args(argv)
    try:
        report=analyze_training(args.run,args.out,TrendConfig(**(read_json(args.config) if args.config else {})))
    except (ValueError,OSError,RuntimeError,TypeError,KeyError) as exc:
        print(f"FAIL analysis: {exc}",file=sys.stderr);return 1
    print(report["status"]+": "+", ".join(report["trends"]))
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
