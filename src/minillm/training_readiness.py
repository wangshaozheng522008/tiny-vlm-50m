"""Read-only launch checks. A probe is not a guarantee of production memory usage."""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from .assessment_io import environment, read_json, report_directory
from .checkpoint import load_training_state
from .config import ARCHITECTURE, ARCH_VERSION, MiniLLMConfig, get_config
from .corpus_audit import audit_dataset
from .corpus_utils import file_info
from .data import EpochBatchSampler, PackedPretrainDataset
from .model import MiniLLM
from .tokenizer import MiniLLMTokenizer
from .train import (TrainingOptions, _prepare_run_manifest, _validate_progress,
                    resolve_precision, set_rng_state)
from .training_manifest import build_manifest, write_json
from .utils import get_device, set_seed


def planned_tokens(samples, batch_size, grad_accum, seq_len, steps):
    updates = math.ceil(math.ceil(samples/batch_size)/grad_accum)
    epochs, remainder = divmod(steps, updates)
    return (epochs*samples+min(remainder*batch_size*grad_accum,samples))*(seq_len-1)


def check_readiness(dataset, tokenizer_path, model_config, training_config, out, *, device=None):
    checks = []
    details = {}
    def add(name, status, message, required=True):
        checks.append({"check":name,"status":status,"required":required,"detail":str(message)})
    tokenizer = cfg = opts = train = val = manifest = state = None
    try:
        raw = read_json(training_config) if isinstance(training_config,(str,Path)) else dict(training_config)
        seq_len = raw.pop("seq_len")
        target_tokens = raw.pop("target_tokens", None)
        exact_budget = raw.pop("planned_effective_tokens",None)
        allow_repeat = raw.pop("allow_data_repetition",False)
        memory_limit = raw.pop("gpu_memory_limit_mib",None)
        disk_factor = raw.pop("disk_margin_factor",1.2)
        if (type(seq_len) is not int or seq_len<2 or type(allow_repeat) is not bool
                or not isinstance(disk_factor,(int,float)) or not math.isfinite(disk_factor) or disk_factor<1):
            raise ValueError("invalid sequence, repetition or disk configuration")
        for value in (target_tokens, exact_budget):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError("Token budgets must be positive integers")
        if memory_limit is not None and (not isinstance(memory_limit,(int,float)) or not math.isfinite(memory_limit) or memory_limit<=0):
            raise ValueError("GPU memory limit must be finite and positive")
        opts=TrainingOptions(**raw)
        for key in ("batch_size","grad_accum","epochs"):
            if type(getattr(opts,key)) is not int or getattr(opts,key)<=0:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("max_steps","save_every","val_every","log_every","val_limit","threads"):
            if type(getattr(opts,key)) is not int or getattr(opts,key)<0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if type(opts.seed) is not int or not 0<=opts.seed<2**32:
            raise ValueError("seed must fit uint32")
        if device is not None:
            opts.device=device
        if not opts.out or Path(opts.out).resolve()==Path(out).resolve() or Path(opts.out).resolve() in Path(out).resolve().parents:
            raise ValueError("readiness report must be outside the prospective training directory")
        cfg=MiniLLMConfig.load(model_config) if Path(model_config).is_file() else get_config(model_config)
        if cfg.model_type!=ARCHITECTURE or cfg.arch_version!=ARCH_VERSION:
            raise ValueError("incompatible model config identity/version")
        if model_config=="minillm-87m" and cfg.num_parameters!=86_528_640:
            raise ValueError("87M architecture budget mismatch")
        add("Configuration parsing","PASS","strict TrainingOptions; explicit sequence and budget policy")
    except Exception as exc:
        add("Configuration parsing","FAIL",exc)
    try:
        audit=audit_dataset(Path(dataset),tokenizer_path=Path(tokenizer_path))
    except Exception as exc:
        audit={"ok":False,"errors":[f"cannot audit tokenizer/dataset: {exc}"]}
    details["dataset_audit"]=audit
    add("Dataset integrity","PASS" if audit["ok"] else "FAIL","; ".join(audit["errors"]) or "full PR3 file, token, identity and known-group audit")
    try:
        tokenizer=MiniLLMTokenizer.load(tokenizer_path)
        published=read_json(Path(dataset)/"dataset-manifest.json")
        if tokenizer.fingerprint!=published["tokenizer"]["fingerprint"]:
            raise ValueError("tokenizer fingerprint mismatch")
        if cfg is None:
            raise ValueError("model configuration unavailable")
        tokenizer.assert_vocab_size(cfg.vocab_size)
        if not 2<=seq_len<=cfg.max_position_embeddings:
            raise ValueError("sequence length outside model context")
        for key in ("pad","bos","eos","unk"):
            if getattr(cfg,key+"_token_id")!=tokenizer.special_ids[key]:
                raise ValueError("model/tokenizer special ID mismatch")
        details["model"]={"configuration":cfg.to_dict(),"analytic_parameters":cfg.num_parameters,
                          "expected_87m_parameters":86_528_640 if model_config=="minillm-87m" else None}
        add("Tokenizer identity","PASS",tokenizer.fingerprint)
    except Exception as exc:
        add("Tokenizer identity","FAIL",exc)
    try:
        selected_device=get_device(opts.device if opts else device)
        if selected_device.type not in ("cpu","cuda"):
            raise ValueError("only CPU and CUDA readiness probes are supported")
    except (RuntimeError,ValueError) as exc:
        add("Device configuration","FAIL",exc)
        selected_device=torch.device("cpu")
    try:
        if opts is None or cfg is None or tokenizer is None:
            raise ValueError("configuration/tokenizer unavailable")
        torch.set_num_threads(opts.threads or 1)
        train=PackedPretrainDataset(Path(dataset)/"train.bin",seq_len)
        val=PackedPretrainDataset(Path(dataset)/"val.bin",seq_len)
        add("Train/Validation windows","PASS",f"{len(train)} / {len(val)} complete windows; L-1 targets each")
        sampler=EpochBatchSampler(len(train),opts.batch_size,opts.seed)
        precision=resolve_precision(opts.precision,selected_device) if selected_device.type!="cuda" or torch.cuda.is_available() else opts.precision
        if opts.resume:
            state=load_training_state(opts.resume,mmap=True)
            if precision=="auto":
                precision=state["manifest"]["execution"]["precision"]
        if precision not in ("fp32","fp16","bf16","auto") or selected_device.type=="cpu" and precision!="fp32":
            raise ValueError("incompatible device/precision")
        manifest,_=build_manifest(opts,SimpleNamespace(cfg=cfg),tokenizer,train,val,mode="pretrain",
                                  sampler=sampler,precision=precision,device=selected_device)
        details["training_manifest"]=manifest
        details["requested_training"]={**asdict(opts),"seq_len":seq_len,"target_tokens":target_tokens,
            "planned_effective_tokens":exact_budget,"allow_data_repetition":allow_repeat,
            "gpu_memory_limit_mib":memory_limit,"disk_margin_factor":disk_factor}
        target=manifest["training"]["target_optimizer_updates"]
        budget=planned_tokens(len(train),opts.batch_size,opts.grad_accum,seq_len,target)
        unique=len(train)*(seq_len-1)
        if exact_budget is not None and budget!=exact_budget:
            raise ValueError(f"planned_effective_tokens={exact_budget} differs from exact fixed-window budget={budget}")
        if target_tokens is not None and not target_tokens<=budget<target_tokens+opts.batch_size*opts.grad_accum*(seq_len-1):
            raise ValueError("max_steps does not reach target_tokens within one update's rounding")
        if budget>unique and not allow_repeat:
            raise ValueError("budget repeats the corpus; explicitly set allow_data_repetition=true or supply more data")
        details["token_budget"]={"planned_effective_tokens":budget,"unique_effective_tokens":unique,
                                 "target_tokens":target_tokens,"allow_data_repetition":allow_repeat,
                                 "target_optimizer_updates":target}
        add("Training configuration","PASS",details["token_budget"])
        if opts.resume:
            _prepare_run_manifest(Path(opts.out),manifest,state["manifest"])
            _validate_progress(state,manifest,train,sampler)
        else:
            _prepare_run_manifest(Path(opts.out),manifest)
        if opts.init_from:
            from .evaluation import load_evaluation_model
            initial,_=load_evaluation_model(opts.init_from,tokenizer,torch.device("cpu"))
            if initial.cfg.to_dict()!=cfg.to_dict():
                raise ValueError("init-from model config mismatch")
            del initial
        if opts.resume and opts.init_from:
            raise ValueError("resume and init_from are mutually exclusive")
        add("Checkpoint compatibility","PASS","verified saved plan, ownership and cursor" if state else "fresh task with protected empty output")
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
        add("Training configuration / Checkpoint compatibility","FAIL",exc)
    if opts and cfg:
        try:
            parent=Path(opts.out).resolve().parent
            while not parent.exists():
                parent=parent.parent
            if not parent.is_dir():
                raise ValueError("checkpoint ancestor is not a directory")
            with tempfile.NamedTemporaryFile(dir=parent,prefix=".minillm-readiness-",delete=True) as probe:
                probe.write(b"probe");probe.flush();os.fsync(probe.fileno())
            # New + primary + previous state (12 bytes/parameter), model and best (4 each).
            required=int(cfg.num_parameters*44*disk_factor)
            free=shutil.disk_usage(parent).free
            details["disk"]={"available_bytes":free,"estimated_required_bytes":required,"margin_factor":disk_factor,
                             "scope":"conservative full state/model/best/atomic rotation, not dataset growth"}
            add("Disk space","PASS" if free>=required else "FAIL",details["disk"])
        except (OSError,ValueError) as exc:
            add("Disk space / Checkpoint write permission","FAIL",exc)
    required_cuda=selected_device.type=="cuda" or (opts is not None and memory_limit is not None)
    cuda_available=torch.cuda.is_available()
    if not cuda_available:
        add("CUDA compatibility","UNKNOWN" if required_cuda else "NOT_RUN","CUDA device unavailable",required_cuda)
    elif selected_device.type!="cuda":
        try:
            # A small operator check is informative even for a CPU launch.
            probe_cfg=get_config("minillm-nano")
            gpu_probe=MiniLLM(probe_cfg).cuda()
            torch.cuda.reset_peak_memory_stats()
            probe_ids=torch.arange(8,24,device="cuda").unsqueeze(0)
            gpu_loss=gpu_probe(probe_ids,labels=probe_ids).loss
            gpu_loss.backward()
            if not torch.isfinite(gpu_loss):
                raise ValueError("nonfinite CUDA probe loss")
            details["cuda_operator_probe"]={**environment(torch.device("cuda")),
                "model":"minillm-nano","peak_vram_mib":torch.cuda.max_memory_allocated()/1024**2,
                "scope":"small operator probe; does not estimate 87M production VRAM"}
            add("CUDA compatibility","PASS",details["cuda_operator_probe"],False)
            del gpu_probe
            if memory_limit is not None:
                add("GPU memory probe limit","UNKNOWN","configured-model CUDA probe required for specified limit; select device=cuda")
        except Exception as exc:
            add("CUDA compatibility","FAIL",exc,required_cuda)
    probe_device=selected_device if selected_device.type!="cuda" or cuda_available else torch.device("cpu")
    if cfg and tokenizer and opts:
        try:
            set_seed(opts.seed)
            model=MiniLLM(cfg).to(probe_device)
            model.gradient_checkpointing=opts.grad_checkpointing
            if opts.init_from and not state:
                from .evaluation import load_evaluation_model
                model,_=load_evaluation_model(opts.init_from,tokenizer,probe_device)
                model.train()
                model.gradient_checkpointing=opts.grad_checkpointing
            if state:
                model.load_state_dict(state["model"],strict=True)
                optimizer=torch.optim.AdamW(model.parameters())
                optimizer.load_state_dict(state["optimizer"])
                for slot in optimizer.state.values():
                    if any(torch.is_tensor(value) and not torch.isfinite(value).all() for value in slot.values()):
                        raise ValueError("checkpoint contains nonfinite optimizer state")
                set_rng_state(state["rng"])
                del optimizer
            if model.num_parameters()!=cfg.num_parameters:
                raise ValueError("constructed/analytic parameter count mismatch")
            length=min(seq_len,16)
            ids=torch.arange(length,device=probe_device).unsqueeze(0)%(cfg.vocab_size-8)+8
            if probe_device.type=="cuda":
                torch.cuda.reset_peak_memory_stats(probe_device)
            precision=resolve_precision(opts.precision,probe_device)
            dtype={"fp16":torch.float16,"bf16":torch.bfloat16}.get(precision)
            with torch.autocast(device_type=probe_device.type,dtype=dtype,enabled=dtype is not None and probe_device.type=="cuda"):
                result=model(ids,labels=ids)
            if not torch.isfinite(result.loss):
                raise ValueError("nonfinite forward loss")
            result.loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("nonfinite probe gradient")
            peak=torch.cuda.max_memory_allocated(probe_device)/1024**2 if probe_device.type=="cuda" else None
            details["probe"]={"device":str(probe_device),"sequence_length":length,"batch_size":1,
                              "loss":float(result.loss.detach()),"peak_vram_mib":peak,
                              "scope":"configured model forward/backward only; no production batch/sequence or AdamW step OOM guarantee"}
            add("Model Forward/Backward","PASS",details["probe"])
            if probe_device.type=="cuda":
                add("CUDA compatibility","PASS",environment(probe_device),required_cuda)
                if memory_limit is not None:
                    add("GPU memory probe limit","PASS" if peak<=memory_limit else "FAIL",f"{peak:.2f} MiB vs limit {memory_limit} MiB")
            del model
        except (OSError,ValueError,RuntimeError,KeyError) as exc:
            add("Model Forward/Backward","FAIL",exc)
    else:
        add("Model Forward/Backward","UNKNOWN","configuration or tokenizer failed")
    status="BLOCKED" if any(c["required"] and c["status"]=="FAIL" for c in checks) else "INCOMPLETE" if any(c["required"] and c["status"] in ("UNKNOWN","NOT_RUN") for c in checks) else "READY"
    report={"schema_version":1,"status":status,"checks":checks,"details":details,
            "environment":environment(torch.device("cpu")),"scope":"preflight only; no formal training launched"}
    with report_directory(out) as directory:
        write_json(directory/"readiness.json",report)
        (directory/"readiness.md").write_text("# Training readiness\n\n"+status+"\n\n"+
            "| Check | Status | Required |\n|---|---|---|\n"+
            "\n".join(f"| {c['check']} | {c['status']} | {c['required']} |" for c in checks)+"\n\nProbe limits and details are recorded in readiness.json.\n")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("dataset","tokenizer","config","training-config","out"):
        p.add_argument("--"+arg,required=True)
    p.add_argument("--device")
    args=p.parse_args(argv)
    try:
        report=check_readiness(args.dataset,args.tokenizer,args.config,args.training_config,args.out,device=args.device)
    except (ValueError,OSError,RuntimeError,KeyError,TypeError) as exc:
        print(f"BLOCKED: {exc}",file=sys.stderr);return 1
    for check in report["checks"]:
        print(f"{check['check']:<48} {check['status']}")
    print(report["status"])
    return {"READY":0,"BLOCKED":1,"INCOMPLETE":2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
