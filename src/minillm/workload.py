"""Representative training probes, bound to an exact fixed training plan."""
from __future__ import annotations

import argparse
import contextlib
import gc
import math
import subprocess
import sys
import tempfile
import time
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import torch

from .assessment_io import environment, read_json, report_directory
from .checkpoint import load_training_state
from .config import MiniLLMConfig, get_config
from .corpus_utils import file_info, identity
from .data import EpochBatchSampler, PackedPretrainDataset, collate_batch
from .evaluation import load_evaluation_model
from .model import MiniLLM
from .train import (TrainingOptions, _train_step, autocast_dtype, evaluate, lr_at,
                    resolve_precision, rng_state, save_training_state, set_rng_state)
from .training_manifest import build_manifest, write_json
from .utils import get_device, set_seed

EXTRAS = {"seq_len", "target_tokens", "planned_effective_tokens", "allow_data_repetition",
          "gpu_memory_limit_mib", "disk_margin_factor", "workload_report", "lineage_report"}


def parse_training_config(config):
    raw = read_json(config) if isinstance(config, (str, Path)) else dict(config)
    extra = {k: raw.pop(k) for k in list(raw) if k in EXTRAS}
    seq_len = extra.get("seq_len")
    if type(seq_len) is not int or seq_len < 2:
        raise ValueError("explicit seq_len >= 2 required")
    if raw.get("extra_meta"):
        raise ValueError("extra_meta is managed by the pilot launcher")
    opts = TrainingOptions(**raw)
    for name in ("batch_size", "grad_accum", "epochs", "threads"):
        if type(getattr(opts, name)) is not int or getattr(opts, name) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("max_steps", "log_every", "save_every", "val_every", "val_limit"):
        if type(getattr(opts, name)) is not int or getattr(opts, name) < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if type(opts.seed) is not int or not 0 <= opts.seed < 2**32:
        raise ValueError("seed must fit uint32")
    if opts.resume and opts.init_from:
        raise ValueError("resume and init_from are mutually exclusive")
    return opts, extra


def hardware(device):
    result = environment(device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(total_memory_bytes=props.total_memory, capability=list(torch.cuda.get_device_capability(device)),
                      uuid=str(getattr(props, "uuid", "unknown")))
        try:
            result["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
                text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            result["driver"] = "UNKNOWN"
    return result


def implementation():
    return identity({name: file_info(Path(__file__).with_name(name)) for name in (
        "workload.py", "train.py", "model.py", "data.py", "tokenizer.py", "checkpoint.py",
        "training_manifest.py", "config.py", "utils.py", "evaluation.py")})


def resolve_plan(dataset, tokenizer_path, model_config, config):
    opts, extra = parse_training_config(config)
    torch.set_num_threads(opts.threads)
    device = get_device(opts.device)
    if device.type not in ("cpu", "cuda") or device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CPU/CUDA device unavailable; no fallback")
    cfg = MiniLLMConfig.load(model_config) if Path(model_config).is_file() else get_config(model_config)
    from .tokenizer import MiniLLMTokenizer
    tok = MiniLLMTokenizer.load(tokenizer_path)
    tok.assert_vocab_size(cfg.vocab_size)
    seq_len = extra["seq_len"]
    train = PackedPretrainDataset(Path(dataset)/"train.bin", seq_len)
    val = PackedPretrainDataset(Path(dataset)/"val.bin", seq_len)
    sampler = EpochBatchSampler(len(train), opts.batch_size, opts.seed)
    precision = resolve_precision(opts.precision, device)
    if device.type == "cpu" and precision != "fp32":
        raise ValueError("CPU probe requires fp32")
    manifest, _ = build_manifest(opts, SimpleNamespace(cfg=cfg), tok, train, val, mode="pretrain",
                                 sampler=sampler, precision=precision, device=device)
    # Resume shares the original initialization identity; checkpoint progress is separate.
    init = None
    if opts.resume:
        state = load_training_state(opts.resume, mmap=True)
        from .pilot_lineage import load_for_resume
        lineage = load_for_resume(opts, state, manifest, train, sampler, report=extra.get("lineage_report"))
        init = lineage["initialization"]
    elif opts.init_from:
        init = {"sha256": file_info(Path(opts.init_from))["sha256"], "kind": "weights_only"}
    else:
        init = {"seed": opts.seed, "kind": "fresh"}
    policy = {"manifest": manifest, "initialization": init, "validation_limit": opts.val_limit,
              "environment": hardware(device), "implementation": implementation()}
    return opts, extra, cfg, tok, train, val, sampler, device, precision, policy


@contextlib.contextmanager
def preserved_process():
    rng, threads = rng_state(), torch.get_num_threads()
    try:
        yield
    finally:
        set_rng_state(rng)
        torch.set_num_threads(threads)


def probe_workload(dataset, tokenizer_path, model_config, training_config, out, *, updates=3):
    if type(updates) is not int or updates < 3:
        raise ValueError("at least three full optimizer updates required")
    requested,_=parse_training_config(training_config)
    for protected in (Path(requested.out).resolve(),Path(dataset).resolve()):
        if Path(out).resolve()==protected or protected in Path(out).resolve().parents:
            raise ValueError("probe reports must be outside training output and immutable dataset")
    with report_directory(out) as directory, preserved_process():
        opts, extra = parse_training_config(training_config)
        torch.set_num_threads(opts.threads)
        opts, extra, cfg, tok, train, val, sampler, device, precision, policy = resolve_plan(
            dataset, tokenizer_path, model_config, training_config)
        if opts.resume:
            raise ValueError("probe the original initialization; do not probe by continuing a real state")
        if updates > policy["manifest"]["training"]["target_optimizer_updates"]:
            raise ValueError("probe updates exceed fixed plan")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            free_before, total = torch.cuda.mem_get_info(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            free_before = total = None
        set_seed(opts.seed)
        if opts.init_from:
            model, _ = load_evaluation_model(opts.init_from, tok, device)
            if model.cfg.to_dict() != cfg.to_dict():
                raise ValueError("initial checkpoint config differs from requested model")
        else:
            model = MiniLLM(cfg).to(device)
        model.train()
        model.gradient_checkpointing = opts.grad_checkpointing
        optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr, betas=(.9,.95), eps=1e-8, weight_decay=opts.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=precision=="fp16") if device.type=="cuda" else None
        step_rows, epoch, cursor, samples, tokens = [], 0, 0, 0, 0
        plan = policy["manifest"]["training"]
        for step in range(updates):
            indices = list(islice(sampler.batches(epoch, cursor),min(opts.grad_accum,len(sampler)-cursor)))
            group = [collate_batch([train[i] for i in batch]) for batch in indices]
            if device.type=="cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            lr = lr_at(step, plan["target_optimizer_updates"], opts.lr, opts.min_lr, plan["warmup_steps"])
            for p in optimizer.param_groups: p["lr"] = lr
            result = _train_step(model, group, optimizer, scaler, device, autocast_dtype(precision), opts, lr)
            if device.type=="cuda": torch.cuda.synchronize(device)
            seconds = time.perf_counter()-started
            cursor += len(group)
            samples += sum(len(batch) for batch in indices)
            tokens += result.tokens
            step_rows.append({"global_step":step+1,"loss":result.loss_sum/result.tokens,
                              "tokens":result.tokens,"seconds":seconds,"tokens_per_second":result.tokens/seconds})
            if cursor==len(sampler): epoch,cursor=epoch+1,0
        metrics = evaluate(model,val,device,batch_size=opts.batch_size,max_batches=opts.val_limit or None)
        if not math.isfinite(metrics["loss"]): raise ValueError("nonfinite validation loss")
        with tempfile.TemporaryDirectory(prefix="minillm-workload-") as temporary:
            path = Path(temporary)/"state.pt"
            started = time.perf_counter()
            save_training_state(path,model,optimizer,scaler,tok,global_step=updates,epoch=epoch,
                                micro_batches_consumed_in_epoch=cursor,samples_seen=samples,
                                best_loss=metrics["loss"],history=[],tokens=tokens,manifest=policy["manifest"])
            loaded = load_training_state(path,mmap=True)
            if loaded["global_step"]!=updates: raise ValueError("checkpoint roundtrip failed")
            checkpoint_seconds = time.perf_counter()-started
            del loaded
        if device.type=="cuda":
            torch.cuda.synchronize(device)
            allocated = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            cap = min(int(total*.8),int(extra.get("gpu_memory_limit_mib",total/2**20*.8)*2**20))
            passed = allocated<=cap and reserved<=cap and free_before>=cap
        else:
            allocated=reserved=cap=None;passed=True
        measurements = {"updates":step_rows,"optimizer_steps":updates,"validation":metrics,
                        "checkpoint_roundtrip":True,"checkpoint_seconds":checkpoint_seconds,
                        "peak_allocated_bytes":allocated,"peak_reserved_bytes":reserved,"device_total_bytes":total,
                        "free_before_bytes":free_before,"limit_bytes":cap,"effective_tokens":tokens,
                        "training_tokens_per_second":tokens/sum(r["seconds"] for r in step_rows),
                        "scope":"full configured micro-batches/sequence/accumulation, AdamW updates, validation and state roundtrip; not a guarantee against future OOM"}
        report = {"schema_version":1,"status":"PASS" if passed else "BLOCKED",
                  "policy":policy,"measurements":measurements}
        report["content_id"] = identity(report)
        write_json(directory/"workload.json",report)
        (directory/"workload.md").write_text("# Representative training probe\n\n"+report["status"]+"\n\n"+measurements["scope"]+"\n")
        # Rehash input identities after operations; never publish a stale proof.
        if resolve_plan(dataset,tokenizer_path,model_config,training_config)[-1] != policy:
            raise ValueError("probe inputs changed during measurement")
    return report


def verify_workload(path, expected_policy, memory_limit_mib):
    path = Path(path)
    report = read_json(path/"workload.json" if path.is_dir() else path)
    if (report.get("schema_version")!=1 or report.get("status")!="PASS"
            or report.get("content_id")!=identity({k:v for k,v in report.items() if k!="content_id"})
            or report.get("policy")!=expected_policy):
        raise ValueError("workload proof damaged, failed, or incompatible with this exact plan/hardware/software")
    m = report["measurements"]
    if (m["optimizer_steps"]<3 or len(m["updates"])!=m["optimizer_steps"] or not m["checkpoint_roundtrip"]
            or not math.isfinite(m["validation"]["loss"])):
        raise ValueError("incomplete representative workload proof")
    if expected_policy["environment"]["device"].startswith("cuda"):
        limit = min(m["limit_bytes"], int(memory_limit_mib*2**20)) if math.isfinite(memory_limit_mib) else m["limit_bytes"]
        if not (0<m["peak_allocated_bytes"]<=limit and 0<m["peak_reserved_bytes"]<=limit
                and m["free_before_bytes"]>=limit):
            raise ValueError("workload exceeds memory budget")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("dataset","tokenizer","config","training-config","out"): p.add_argument("--"+name,required=True)
    p.add_argument("--updates",type=int,default=3)
    args=p.parse_args(argv)
    try:
        r=probe_workload(args.dataset,args.tokenizer,args.config,args.training_config,args.out,updates=args.updates)
    except Exception as exc:
        print(f"BLOCKED workload: {exc}",file=sys.stderr);return 1
    print(r["status"],r["measurements"])
    return 0 if r["status"]=="PASS" else 1
