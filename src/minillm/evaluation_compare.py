"""Compare only hash-verified evaluations with identical metric/generation policies."""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

from .assessment_io import records, report_directory, verified_evaluation
from .evaluation import (DEFAULT_BENCHMARK, DEFAULT_PROMPTS, EvaluationConfig, evaluate_model)
from .training_manifest import write_json


def compare_evaluations(directories, out, *, allow_incomparable=False):
    verified=[verified_evaluation(path) for path in directories]
    if not verified:
        raise ValueError("at least one evaluation is required")
    baseline, base_manifest=verified[0]
    comparable=all(m["comparison_policy"]==base_manifest["comparison_policy"] for _,m in verified)
    if not comparable and not allow_incomparable:
        raise ValueError("incompatible tokenizer, dataset, model config or evaluation policy; use --allow-incomparable for separate experiments without deltas")
    comparisons=[]
    for directory,(report,manifest) in zip(directories,verified):
        lm=report["language_model"]; base_lm=baseline["language_model"]
        changes={}
        for mode,score in report["scores"].items():
            changes[mode]={}
            for category,result in score["categories"].items():
                old=baseline["scores"].get(mode,{}).get("categories",{}).get(category,{}).get("accuracy")
                value=result["accuracy"]
                changes[mode][category]={"accuracy":value,"delta":value-old if comparable and value is not None and old is not None else None}
        performance=report["performance"]
        candidate_changes = {}
        for category, value in report.get("candidate_likelihood", {}).get("categories", {}).items():
            old = baseline.get("candidate_likelihood", {}).get("categories", {}).get(category, {}).get("accuracy")
            candidate_changes[category] = {**value, "delta": value["accuracy"]-old if comparable and old is not None else None}
        comparisons.append({"evaluation_id":report["evaluation_id"],"checkpoint":report["checkpoint"],
            "language_model":lm,"loss_delta":lm["loss"]-base_lm["loss"] if comparable and lm and base_lm else None,
            "scores":changes,"candidate_likelihood":candidate_changes,"stage":report.get("stage"),"environment":report["environment"],
            "performance_reference":{**performance,"generation_tokens_per_second":
                performance["generation_tokens"]/max(performance["generation_seconds"],1e-9),
                "ranking":"not ranked; hardware/environment and warm-up differ"},
            "predictions":list(records(Path(directory)/"predictions.jsonl"))})
    result={"schema_version":1,"comparable":comparable,"experiments":comparisons,
            "scope":"Raw/Chat separate; deltas only for identical comparison policies; no throughput ranking",
            "policy":base_manifest["comparison_policy"] if comparable else None}
    with report_directory(out) as directory:
        write_json(directory/"comparison.json",result)
        lines=["# Checkpoint comparison","",f"Comparable: {comparable}","",
               "| Checkpoint | Step | Tokens seen | Loss | PPL | Loss delta |","|---|---:|---:|---:|---:|---:|"]
        for row in comparisons:
            checkpoint=row["checkpoint"];lm=row["language_model"] or {}
            lines.append(f"| {html.escape(checkpoint['path'])} | {checkpoint['global_step']} | {checkpoint['tokens_seen']} | {lm.get('loss')} | {lm.get('perplexity')} | {row['loss_delta']} |")
        for mode in ("raw","chat"):
            lines.extend(["",f"## {mode} accuracy and baseline delta","","| Category | "+" | ".join(str(i+1) for i in range(len(comparisons)))+" |",
                          "|---|"+"---:|"*len(comparisons)])
            categories=sorted({c for row in comparisons for c in row["scores"][mode]})
            for category in categories:
                lines.append("| "+category+" | "+" | ".join(str(row["scores"][mode].get(category)) for row in comparisons)+" |")
        lines.extend(["", "## Natural-prefix likelihood (separate metric)", ""])
        for row in comparisons:
            lines.append(json.dumps(row["candidate_likelihood"], ensure_ascii=False))
        lines.extend(["","## Fixed prompt generations","","Inference times and memory are references; no hardware ranking.",""])
        keyed=[{(p["kind"],p["id"]):p for p in row["predictions"]} for row in comparisons]
        for key,prediction in keyed[0].items():
            if key[0]!="fixed_prompt":
                continue
            lines.extend([f"### {prediction['id']} ({prediction['generation_mode']})","",
                          html.escape(prediction["prompt"]),"",
                          "| "+" | ".join(str(i+1) for i in range(len(comparisons)))+" |",
                          "|"+"---|"*len(comparisons),
                          "| "+" | ".join(html.escape(rows.get(key,{}).get("decoded_text","MISSING")).replace("|","&#124;").replace("\n","<br>") for rows in keyed)+" |",""])
        (directory/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoints",nargs="+");group.add_argument("--evaluations",nargs="+")
    p.add_argument("--tokenizer");p.add_argument("--dataset");p.add_argument("--bin",dest="bin_path")
    p.add_argument("--split",choices=["val","test"],default="val")
    p.add_argument("--benchmark",default=str(DEFAULT_BENCHMARK));p.add_argument("--prompts",default=str(DEFAULT_PROMPTS))
    p.add_argument("--config");p.add_argument("--out",required=True);p.add_argument("--device",default="cpu")
    p.add_argument("--allow-incomparable",action="store_true")
    args=p.parse_args(argv)
    try:
        if args.evaluations:
            report=compare_evaluations(args.evaluations,args.out,allow_incomparable=args.allow_incomparable)
        else:
            if not args.tokenizer:
                raise ValueError("--tokenizer is required with --checkpoints")
            config=EvaluationConfig(**(json.loads(Path(args.config).read_text()) if args.config else {}))
            with report_directory(args.out) as directory:
                evaluations=[]
                for i,checkpoint in enumerate(args.checkpoints):
                    child=directory/f"checkpoint-{i:03}"
                    evaluate_model(checkpoint,args.tokenizer,child,dataset=args.dataset,bin_path=args.bin_path,
                                   split=args.split,benchmark=args.benchmark or None,prompts=args.prompts or None,config=config,device=args.device)
                    evaluations.append(child)
                report=compare_evaluations(evaluations,directory/"summary",allow_incomparable=args.allow_incomparable)
                # Keep reports and their manifests alongside a top-level summary.
                for name in ("comparison.json","comparison.md"):
                    (directory/"summary"/name).rename(directory/name)
                (directory/"summary").rmdir()
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
        print(f"FAIL comparison: {exc}",file=sys.stderr);return 1
    print(f"PASS comparison: {len(report['experiments'])} checkpoints; comparable={report['comparable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
