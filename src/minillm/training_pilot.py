"""Post-run pilot acceptance. Does not start training or authorize a billion-token run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .assessment_io import read_json, report_directory, verified_evaluation
from .checkpoint import load_training_state
from .config import MiniLLMConfig
from .corpus_utils import file_info
from .data import EpochBatchSampler, PackedPretrainDataset
from .evaluation_compare import compare_evaluations
from .model import MiniLLM
from .train import _validate_progress, set_rng_state
from .training_analysis import analyze_training
from .training_manifest import dataset_identity, write_json


def evaluate_pilot(run, baseline_eval, final_eval, out):
    run=Path(run)
    baseline,bm=verified_evaluation(baseline_eval)
    final,fm=verified_evaluation(final_eval)
    errors=[]
    with report_directory(out) as directory:
        analysis=analyze_training(run,directory/"analysis")
        comparison=compare_evaluations([baseline_eval,final_eval],directory/"comparison")
        errors.extend(analysis["errors"])
        summary=read_json(run/"train-summary.json")
        try:
            state=load_training_state(run/"state.pt",mmap=True)
            manifest=read_json(run/"training-manifest.json")
            train=manifest["dataset"]["train"]
            if train["type"]!="PackedPretrainDataset":
                raise ValueError("pilot engineering acceptance currently requires a pretrain fixed-window run")
            data=PackedPretrainDataset(train["path"],train["seq_len"])
            if dataset_identity(data,state["config"]["ignore_index"])[0]!=train:
                raise ValueError("training data identity changed")
            sampler=EpochBatchSampler(len(data),manifest["training"]["batch_size"],manifest["training"]["sampler_seed"])
            _validate_progress(state,manifest,data,sampler)
            if state["global_step"]!=manifest["training"]["target_optimizer_updates"]:
                raise ValueError("pilot stopped before planned target")
            model=torch.load(run/"model.pt",map_location="cpu",weights_only=False)
            if (file_info(run/"model.pt")["sha256"]!=final["checkpoint"]["sha256"]
                    or model.get("step")!=state["global_step"] or model.get("tokens_seen")!=state["tokens_seen"]
                    or model.get("config")!=state["config"] or model.get("tokenizer_fingerprint")!=state["tokenizer_fingerprint"]):
                raise ValueError("final evaluation/model/state do not identify the same completed checkpoint")
            if set(model["model"])!=set(state["model"]) or any(not torch.equal(v,state["model"][k]) for k,v in model["model"].items()):
                raise ValueError("model.pt weights differ from recoverable training state")
            restored=MiniLLM(MiniLLMConfig.from_dict(state["config"]))
            restored.load_state_dict(state["model"],strict=True)
            optimizer=torch.optim.AdamW(restored.parameters())
            optimizer.load_state_dict(state["optimizer"])
            for parameter,slot in optimizer.state.items():
                for key in ("exp_avg","exp_avg_sq"):
                    if slot[key].shape!=parameter.shape or not torch.isfinite(slot[key]).all():
                        raise ValueError("unrecoverable optimizer moments")
            set_rng_state(state["rng"])
            del optimizer,restored
            baseline_step=baseline["checkpoint"]["global_step"]
            if baseline_step is not None and baseline_step>state["global_step"]:
                raise ValueError("baseline checkpoint is later than final checkpoint")
            val=manifest["dataset"]["val"]
            for evaluation,em in ((baseline,bm),(final,fm)):
                if (em["tokenizer_fingerprint"]!=manifest["tokenizer_fingerprint"]
                        or em["comparison_policy"]["model_config"]!=manifest["model_config"]
                        or not evaluation["dataset"] or evaluation["dataset"]["split"]!="val"
                        or evaluation["dataset"]["sha256"]!=val["sha256"]
                        or evaluation["dataset"].get("dataset_content_id")!=val.get("dataset_content_id")):
                    raise ValueError("pilot validation/model/data/tokenizer identity mismatch (Test is not a convergence decision input)")
        except Exception as exc:
            errors.append(str(exc))
        lm0,lm1=baseline["language_model"],final["language_model"]
        convergence="CONVERGENCE_IMPROVING" if not errors and lm0 and lm1 and lm1["loss"]<lm0["loss"] and "VALIDATION_DECREASING" in analysis["trends"] else "CONVERGENCE_INCONCLUSIVE"
        changes=[r["delta"] for modes in comparison["experiments"][-1]["scores"].values() for r in modes.values() if r["delta"] is not None]
        capability="CAPABILITY_IMPROVING" if not errors and changes and any(x>0 for x in changes) and not any(x<0 for x in changes) else "CAPABILITY_INCONCLUSIVE"
        logs=[r for r in analysis["raw_records"] if r.get("event")=="train"]
        duration=sum(r["loss_tokens"]/r["tokens_per_second"] for r in logs if isinstance(r.get("tokens_per_second"),(int,float)) and r["tokens_per_second"]>0)
        accounted=sum(r.get("loss_tokens",0) for r in logs)
        full_timing=duration>0 and accounted==summary["tokens_seen"] and not analysis["identical_duplicate_records"]
        report={"schema_version":1,"engineering":"ENGINEERING_PASS" if not errors else "ENGINEERING_FAIL",
                "convergence":convergence,"capability":capability,"errors":errors,
                "tokens_seen":summary["tokens_seen"],"global_step":summary["global_step"],
                "last_process_seconds":summary["seconds"],"estimated_logged_wall_seconds":duration if full_timing else None,
                "average_logged_tokens_per_second":summary["tokens_seen"]/duration if full_timing else None,
                "timing_scope":"sum of logged interval durations; includes validation/checkpoint overhead; exact total lifetime not recorded by PR2",
                "final_training_loss":logs[-1].get("loss") if logs else None,
                "training_loss_scope":"last token-weighted logged interval",
                "initial_language_model":lm0,"final_language_model":lm1,
                "capability_changes":comparison["experiments"][-1]["scores"],
                "generation_examples":comparison["experiments"][-1]["predictions"],
                "baseline_lineage":"caller-supplied comparison checkpoint; PR2 does not cryptographically record initial-weight lineage",
                "numerical_or_resume_anomalies":analysis["errors"],
                "recommendation":"Resolve engineering failures before proceeding." if errors else
                    "Review convergence, per-category changes and compute budget; consider a bounded next pilot. This report does not authorize or automatically recommend one billion tokens."}
        write_json(directory/"pilot-evaluation.json",report)
        (directory/"pilot-evaluation.md").write_text("# Pilot acceptance\n\n"+
            "\n\n".join((report["engineering"],convergence,capability,report["recommendation"]))+
            "\n\n"+chr(96)*3+"json\n"+json.dumps({k:v for k,v in report.items() if k!="generation_examples"},ensure_ascii=False,indent=2)+"\n"+chr(96)*3+"\n")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("run","baseline-eval","final-eval","out"):
        p.add_argument("--"+name,required=True)
    args=p.parse_args(argv)
    try:
        report=evaluate_pilot(args.run,args.baseline_eval,args.final_eval,args.out)
    except (ValueError,OSError,RuntimeError,KeyError,TypeError) as exc:
        print(f"ENGINEERING_FAIL: {exc}",file=sys.stderr);return 1
    print(report["engineering"],report["convergence"],report["capability"])
    return int(report["engineering"]!="ENGINEERING_PASS")


if __name__ == "__main__":
    raise SystemExit(main())
