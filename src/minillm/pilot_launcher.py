"""One configuration for readiness and explicit pilot execution."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from .assessment_io import read_json, report_directory
from .corpus_utils import identity
from .corpus_utils import file_info
from .model import MiniLLM
from .checkpoint import load_training_state
from .pilot_lineage import load_for_resume, persist_initial, record_resume, run_lock
from .training_readiness import check_readiness
from .training_manifest import write_json
from .train import fit, load_model_checkpoint
from .utils import set_seed
from .workload import parse_training_config, resolve_plan


def launch_pilot(dataset, tokenizer_path, model_config, training_config, report_out, *, execute=False):
    # A protected report directory also prevents duplicate execution of a launch request.
    requested,_=parse_training_config(training_config)
    for protected in (Path(requested.out).resolve(),Path(dataset).resolve()):
        if Path(report_out).resolve()==protected or protected in Path(report_out).resolve().parents:
            raise ValueError("launch reports must be outside training output and immutable dataset")
    with report_directory(report_out) as directory:
        config = read_json(training_config) if isinstance(training_config,(str,Path)) else dict(training_config)
        parsed,_=parse_training_config(config)
        if torch.device(parsed.device or "cpu").type=="cuda" and not torch.cuda.is_available():
            readiness=check_readiness(dataset,tokenizer_path,model_config,config,directory/"readiness")
            launch={"schema_version":1,"execute_requested":execute,"status":readiness["status"],
                    "configuration":config,"executed":False,
                    "note":"required CUDA unavailable; no CPU training fallback"}
            write_json(directory/"launch-manifest.json",launch)
            return launch
        opts, extra, cfg, tok, train, val, sampler, device, _, policy = resolve_plan(
            dataset,tokenizer_path,model_config,config)
        torch.set_num_threads(opts.threads)
        if execute and device.type=="cuda" and not extra.get("workload_report"):
            raise ValueError("CUDA execution requires a representative workload report")
        readiness = check_readiness(dataset,tokenizer_path,model_config,config,directory/"readiness")
        launch = {"schema_version":2,"execute_requested":execute,"status":readiness["status"],
                  "report_path":str(Path(report_out).resolve()),
                  "training_manifest":policy["manifest"],"tokenizer_fingerprint":tok.fingerprint,
                  "source_path":str(Path(opts.init_from).resolve()) if opts.init_from else None,
                  "configuration":config,"resolved_options":asdict(opts),"initialization":policy["initialization"],
                  "dataset":policy["manifest"]["dataset"],"model_config":cfg.to_dict(),
                  "training_output":str(Path(opts.out).resolve()),
                  "note":"weights-only initialization resets optimizer and this-run counters; no legacy resume"}
        if opts.resume:
            state=load_training_state(opts.resume,mmap=True)
            launch["training_manifest"]=state["manifest"]
            launch["checkpoint"]={"sha256":file_info(Path(opts.resume))["sha256"],
                                  "global_step":state["global_step"],"run_id":state["manifest"]["run_id"]}
            del state
        launch["request_id"] = identity(launch)
        write_json(directory/"launch-manifest.json",launch)
    # The launch record is durable before training creates any state or takes a step.
    if readiness["status"]!="READY" or not execute:
        return launch
    with run_lock(opts.out):
        if resolve_plan(dataset,tokenizer_path,model_config,config)[-1]!=policy:
            raise ValueError("launch inputs changed after readiness")
        if opts.resume:
            state=load_training_state(opts.resume,mmap=True)
            if launch["checkpoint"]!={"sha256":file_info(Path(opts.resume))["sha256"],
                                       "global_step":state["global_step"],"run_id":state["manifest"]["run_id"]}:
                raise ValueError("resume checkpoint changed after readiness")
            lineage=load_for_resume(opts,state,policy["manifest"],train,sampler,report=extra.get("lineage_report"))
            record_resume(opts.out,lineage,report_out,launch["request_id"],state["global_step"])
            del state
        else:
            set_seed(opts.seed)
        if opts.init_from:
            model,loaded_cfg,payload=load_model_checkpoint(opts.init_from,torch.device("cpu"))
            if (loaded_cfg.to_dict()!=cfg.to_dict() or payload.get("tokenizer_fingerprint")!=tok.fingerprint
                    or file_info(Path(opts.init_from))["sha256"]!=policy["initialization"]["sha256"]):
                raise ValueError("initial model changed after readiness")
            del payload
        else:
            model=MiniLLM(cfg)
        opts.extra_meta={"pilot_request_id":launch["request_id"],
                         "pilot_initialization":policy["initialization"],"seq_len":extra["seq_len"]}
        def prepared(manifest):
            if {k:v for k,v in manifest.items() if k!="run_id"}!=policy["manifest"]:
                raise ValueError("prepared training manifest changed after readiness")
            persist_initial(opts.out,manifest,policy["initialization"],report_out,launch["request_id"])
        summary=fit(opts,model,tok,train,val,on_run_prepared=prepared if not opts.resume else None)
    write_json(Path(report_out)/"execution-result.json",summary)
    return {**launch,"executed":True,"summary":summary}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("dataset","tokenizer","config","training-config","out"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--execute",action="store_true")
    args=p.parse_args(argv)
    try:
        report=launch_pilot(args.dataset,args.tokenizer,args.config,args.training_config,args.out,execute=args.execute)
    except Exception as exc:
        print(f"BLOCKED launch: {exc}",file=sys.stderr);return 1
    print(report["status"],"executed" if report.get("executed") else "inspection only")
    return {"READY":0,"BLOCKED":1,"INCOMPLETE":2}[report["status"]]


if __name__=="__main__":
    raise SystemExit(main())
