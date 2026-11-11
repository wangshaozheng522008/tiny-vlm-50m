"""Fixed-window loss, deterministic capability evaluation and read-only reports."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .assessment_io import environment, publish_manifest, read_json, records, report_directory
from .config import ARCHITECTURE, ARCH_VERSION, MiniLLMConfig
from .corpus_audit import audit_dataset
from .corpus_utils import file_info, identity
from .data import PackedPretrainDataset, collate_batch
from .evaluation_scoring import aggregate_scores, score_answer, validate_task
from .generate import stop_ids_for, truncate_at_stop
from .model import MiniLLM
from .quantize import dequantize_model
from .tokenizer import DEFAULT_SYSTEM, MiniLLMTokenizer, SPECIAL_TOKEN_IDS
from .training_manifest import dataset_identity, write_json
from .utils import get_device, peak_rss_mb, set_seed

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = ROOT / "data/evaluation/minillm-zh-benchmark.jsonl"
DEFAULT_PROMPTS = ROOT / "data/evaluation/fixed-prompts.jsonl"


@dataclass(frozen=True)
class EvaluationConfig:
    seq_len: int = 512
    batch_size: int = 1
    token_budget: int = 65536
    max_new_tokens: int = 64
    seed: int = 42
    temperature: float = 0.0
    top_k: int = 0
    system: str | None = DEFAULT_SYSTEM
    threads: int = 1

    def validate(self):
        for key in ("seq_len", "batch_size", "max_new_tokens", "threads"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.seq_len < 2 or type(self.token_budget) is not int or self.token_budget < 0:
            raise ValueError("seq_len >= 2 and token_budget >= 0 required (0 = all)")
        if not math.isfinite(self.temperature) or self.temperature < 0 or type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("invalid sampling configuration")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must fit uint32")


def load_evaluation_model(path, tokenizer, device):
    path = Path(path)
    if path.name.startswith(".") or path.suffix == ".tmp":
        raise ValueError("temporary checkpoints cannot be evaluated")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"cannot deserialize checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("architecture") != ARCHITECTURE or payload.get("arch_version") != ARCH_VERSION:
        raise ValueError("incompatible checkpoint architecture/version")
    try:
        cfg = MiniLLMConfig.from_dict(payload["config"])
    except Exception as exc:
        raise ValueError(f"incompatible model config: {exc}") from exc
    if cfg.model_type != ARCHITECTURE or cfg.arch_version != ARCH_VERSION:
        raise ValueError("incompatible model config identity/version")
    tokenizer.assert_vocab_size(cfg.vocab_size)
    if payload.get("tokenizer_fingerprint") != tokenizer.fingerprint:
        raise ValueError("checkpoint tokenizer fingerprint missing or incompatible")
    for name in ("pad", "bos", "eos", "unk"):
        if getattr(cfg, f"{name}_token_id") != SPECIAL_TOKEN_IDS[name]:
            raise ValueError(f"incompatible model special token {name}")
    if "quantized" in payload:
        if payload.get("quantized_bits") not in (4, 8):
            raise ValueError("unsupported quantization")
        model = dequantize_model(payload)
    else:
        model = MiniLLM(cfg)
        model.load_state_dict(payload["model"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("checkpoint contains nonfinite weights")
    return model.to(device).eval(), payload


def ppl_from_totals(loss_sum, tokens):
    if tokens <= 0:
        raise ValueError("empty evaluation: no valid loss tokens")
    loss = loss_sum / tokens
    if not math.isfinite(loss):
        raise ValueError("nonfinite evaluation loss")
    try:
        ppl = math.exp(loss)
    except OverflowError:
        ppl = None
    return {"loss": loss, "loss_sum": loss_sum, "tokens": tokens, "perplexity": ppl,
            "perplexity_status": "overflow_unrepresentable" if ppl is None else "finite",
            "display_clipped": False}


@torch.inference_mode()
def evaluate_batches(model, batches, device, token_budget=0):
    """Labels stay aligned with inputs. MiniLLM performs the only shift."""
    model.eval()
    total, count, evaluated_batches = 0.0, 0, 0
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"].clone()
        mask = batch.get("attention_mask")
        if mask is not None:
            labels[mask == 0] = model.cfg.ignore_index
            labels[:, 1:][mask[:, :-1] == 0] = model.cfg.ignore_index
        valid = labels[:, 1:] != model.cfg.ignore_index
        remaining = token_budget - count if token_budget else int(valid.sum())
        if remaining <= 0:
            if token_budget:
                break
            continue
        if int(valid.sum()) > remaining:
            selected = valid.flatten().cumsum(0).reshape(valid.shape) > remaining
            labels[:, 1:][selected] = model.cfg.ignore_index
        out = model(batch["input_ids"], labels=labels, attention_mask=mask)
        if not torch.isfinite(out.loss_sum):
            raise ValueError("nonfinite evaluation loss")
        total += float(out.loss_sum)
        count += out.num_loss_tokens
        evaluated_batches += 1
        if token_budget and count == token_budget:
            break
    return {**ppl_from_totals(total, count), "batches": evaluated_batches}


def fixed_window_batches(dataset, batch_size):
    for start in range(0, len(dataset), batch_size):
        yield collate_batch([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])


def prepare_eval_data(dataset, bin_path, split, tokenizer, seq_len, cfg):
    if seq_len > cfg.max_position_embeddings:
        raise ValueError("seq_len exceeds checkpoint context; it will not be silently changed")
    if dataset and bin_path:
        raise ValueError("choose --dataset or --bin, not both")
    if not dataset and not bin_path:
        return None, None
    if dataset:
        if split not in ("val", "test"):
            raise ValueError("only explicit val/test evaluation splits are supported")
        audit = audit_dataset(Path(dataset))
        if not audit["ok"]:
            raise ValueError("dataset audit failed: " + "; ".join(audit["errors"]))
        manifest = read_json(Path(dataset) / "dataset-manifest.json")
        if manifest["tokenizer"]["fingerprint"] != tokenizer.fingerprint:
            raise ValueError("dataset tokenizer fingerprint mismatch")
        path = Path(dataset) / f"{split}.bin"
    else:
        path, split = Path(bin_path), "custom"
    if path.stat().st_size % 2:
        raise ValueError("odd-sized uint16 evaluation file")
    data = PackedPretrainDataset(path, seq_len)
    for start in range(0, data.tokens.size, 524288):
        if int(data.tokens[start:start + 524288].max()) >= tokenizer.vocab_size:
            raise ValueError("evaluation Token outside tokenizer vocabulary")
    info, _ = dataset_identity(data, cfg.ignore_index)
    info.update(split=split, window_policy="PR2 fixed L inputs, internal shift, L-1 targets/window; boundary targets and tail omitted",
                complete_windows=len(data), input_tokens_per_window=seq_len,
                effective_targets_per_window=seq_len - 1,
                unused_tail_tokens=data.num_tokens - len(data) * seq_len,
                custom_tokenizer_provenance="not independently verifiable for custom BIN" if not dataset else "published manifest verified")
    return data, info


def repetition(text, ids):
    def repeated(values, n):
        grams = [tuple(values[i:i+n]) for i in range(max(0, len(values)-n+1))]
        return (len(grams) - len(set(grams))) / len(grams) if grams else 0.0
    sentences = [s.strip() for s in re.split(r"[。！？!?\n]+", text) if s.strip()]
    return {"token_4gram_repeat_ratio": repeated(ids, 4), "character_3gram_repeat_ratio": repeated(text, 3),
            "sentence_repeat_ratio": (len(sentences)-len(set(sentences)))/len(sentences) if sentences else 0.0}


def prompt_ids(tokenizer, task, config):
    if task["generation_mode"] == "raw":
        return tokenizer.encode(task["prompt"])
    return tokenizer.apply_chat_template([{"role": "user", "content": task["prompt"]}],
                                         system=config.system, add_generation_prompt=True)


def validate_generation(model, tokenizer, task, config):
    ids = prompt_ids(tokenizer, task, config)
    budget = min(config.max_new_tokens, task.get("max_new_tokens", config.max_new_tokens))
    if not ids or type(budget) is not int or budget <= 0 or len(ids) + budget > model.cfg.max_position_embeddings:
        raise ValueError(f"{task['id']}: prompt ({len(ids)}) + generation ({budget}) exceeds context; no implicit truncation")
    return ids, budget


@torch.inference_mode()
def generate_prediction(model, tokenizer, task, config):
    model.eval()
    ids, budget = validate_generation(model, tokenizer, task, config)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(config.seed)
    stops = stop_ids_for(task["generation_mode"] == "raw")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = model.generate(torch.tensor([ids], device=device), max_new_tokens=budget,
                            temperature=config.temperature, top_k=config.top_k, generator=generator,
                            stop_token_ids=stops, eos_token_id=model.cfg.eos_token_id)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    generated = output[0, len(ids):].tolist()
    kept = truncate_at_stop(generated, stops)
    stop_token = generated[len(kept)] if len(kept) < len(generated) else None
    text = tokenizer.decode(kept, skip_special_tokens=True)
    anomalies = [t for t in kept if t in SPECIAL_TOKEN_IDS.values()]
    return {"id": task["id"], "category": task.get("category", "observation"),
            "generation_mode": task["generation_mode"], "prompt": task["prompt"],
            "prompt_token_ids": ids, "generated_token_ids": generated, "decoded_text": text,
            "generated_tokens": len(generated), "content_tokens": len(kept), "generation_budget": budget,
            "stop_reason": {2: "eos", 7: "end_of_turn"}.get(stop_token, "max_new_tokens"),
            "stop_token_id": stop_token, "stop_ids": list(stops),
            "early_end": stop_token is not None and len(generated) < budget,
            "empty_output": not bool(text.strip()), "special_token_anomalies": anomalies,
            "seconds": seconds, "tokens_per_second": len(generated)/max(seconds, 1e-9),
            "diagnostics": repetition(text, kept)}


def load_tasks(path, *, scored):
    tasks, seen = [], set()
    for task in records(path) if path else ():
        if scored:
            validate_task(task)
        elif (not isinstance(task.get("prompt"), str) or not task["prompt"].strip()
              or task.get("generation_mode") not in ("raw", "chat")):
            raise ValueError("invalid fixed prompt")
        if not isinstance(task.get("id"), str) or task["id"] in seen:
            raise ValueError("missing/duplicate task id")
        seen.add(task["id"]); tasks.append(task)
        if len(tasks) > 10000:
            raise ValueError("evaluation task limit 10000 exceeded")
    if path and not tasks:
        raise ValueError("empty benchmark/prompts")
    version_path = Path(path).parent/"benchmark-version.json" if path else None
    if version_path and version_path.exists():
        version = read_json(version_path)
        info = version.get("files",{}).get(Path(path).name)
        if info and (file_info(Path(path))["sha256"] != info["sha256"] or len(tasks) != info["records"]):
            raise ValueError("frozen evaluation version hash/count mismatch; publish a new version")
    return tasks


def evaluate_model(ckpt, tokenizer_path, out, *, dataset=None, bin_path=None, split="val",
                   benchmark=DEFAULT_BENCHMARK, prompts=DEFAULT_PROMPTS, config=None, device="cpu",
                   sft_data=()):
    config = config or EvaluationConfig()
    config.validate()
    torch.set_num_threads(config.threads)
    set_seed(config.seed)
    device = get_device(device)
    with report_directory(out) as directory:
        paths = [Path(p) for p in (ckpt, tokenizer_path, benchmark, prompts, *sft_data) if p]
        if benchmark and (Path(benchmark).parent/"benchmark-version.json").exists():
            paths.append(Path(benchmark).parent/"benchmark-version.json")
        before = {str(p.resolve()): file_info(p) for p in paths}
        try:
            tokenizer = MiniLLMTokenizer.load(tokenizer_path)
        except Exception as exc:
            raise ValueError(f"invalid tokenizer: {exc}") from exc
        model, payload = load_evaluation_model(ckpt, tokenizer, device)
        data, data_info = prepare_eval_data(dataset, bin_path, split, tokenizer, config.seq_len, model.cfg)
        tasks, fixed = load_tasks(benchmark, scored=True), load_tasks(prompts, scored=False)
        for task in tasks + fixed:
            validate_generation(model, tokenizer, task, config)
        policy = {"config": asdict(config), "model_config": model.cfg.to_dict(),
                  "dataset": {k:v for k,v in (data_info or {}).items() if k != "path"},
                  "benchmark_sha256": file_info(Path(benchmark))["sha256"] if benchmark else None,
                  "fixed_prompts_sha256": file_info(Path(prompts))["sha256"] if prompts else None,
                  "tokenizer_fingerprint": tokenizer.fingerprint,
                  "benchmark_provenance": read_json(Path(benchmark).parent/"benchmark-version.json")
                      if benchmark and (Path(benchmark).parent/"benchmark-version.json").exists() else {"source":"custom; caller supplied","version":"content SHA256"},
                  "implementation": identity({name: file_info(Path(__file__).with_name(name)) for name in
                      ("evaluation.py", "evaluation_scoring.py", "evaluation_contamination.py", "assessment_io.py",
                       "generate.py", "model.py", "tokenizer.py", "data.py", "config.py")})}
        evaluation_identity = {"checkpoint_sha256": file_info(Path(ckpt))["sha256"], "policy": policy}
        evaluation_id = identity(evaluation_identity)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        lm = evaluate_batches(model, fixed_window_batches(data, config.batch_size), device, config.token_budget) if data is not None else None
        if device.type=="cuda":
            torch.cuda.synchronize(device)
        lm_seconds=time.perf_counter()-started
        predictions = []
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for kind, rows in (("benchmark", tasks), ("fixed_prompt", fixed)):
                for task in rows:
                    row = {"kind": kind, **generate_prediction(model, tokenizer, task, config)}
                    if kind == "benchmark":
                        row["reference"], row["scoring"] = task.get("reference"), task["scoring"]
                        row["score"] = score_answer(row["decoded_text"], task)
                        if row["special_token_anomalies"] and row["score"]["correct"] is not None:
                            row["score"]={"correct":False,"status":"invalid_special_tokens"}
                    predictions.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        from .evaluation_contamination import contamination_audit
        contamination = contamination_audit(tasks, dataset=dataset, sft_data=sft_data)
        for path, expected in before.items():
            if file_info(Path(path)) != expected:
                raise ValueError(f"input changed during evaluation: {path}")
        if data is not None and dataset_identity(data, model.cfg.ignore_index)[0]["sha256"] != data_info["sha256"]:
            raise ValueError("evaluation BIN changed during evaluation")
        seconds = time.perf_counter() - started
        report = {"schema_version": 1, "evaluation_id": evaluation_id, "language_model": lm,
                  "scores": aggregate_scores(predictions), "dataset": data_info,
                  "checkpoint": {"sha256": evaluation_identity["checkpoint_sha256"], "path": str(Path(ckpt).resolve()),
                    "global_step": payload.get("global_step", payload.get("step")),
                    "tokens_seen": payload.get("tokens_seen"), "mode": payload.get("mode", "unknown"),
                    "quantized_bits": payload.get("quantized_bits"), "parameters": model.num_parameters(),
                    "execution": "dequantized to FP32; no native low-bit kernels" if "quantized" in payload else "FP32"},
                  "diagnostics": {"scope": "degeneration indicators, not quality scores",
                      "generations": len(predictions), "empty_outputs": sum(p["empty_output"] for p in predictions),
                      "early_endings": sum(p["early_end"] for p in predictions),
                      "empty_output_rate":sum(p["empty_output"] for p in predictions)/len(predictions) if predictions else None,
                      "early_end_rate":sum(p["early_end"] for p in predictions)/len(predictions) if predictions else None,
                      "mean_generated_tokens":sum(p["generated_tokens"] for p in predictions)/len(predictions) if predictions else None,
                      "mean_token_4gram_repeat_ratio":sum(p["diagnostics"]["token_4gram_repeat_ratio"] for p in predictions)/len(predictions) if predictions else None,
                      "special_token_anomalies": sum(len(p["special_token_anomalies"]) for p in predictions)},
                  "contamination": contamination, "environment": environment(device),
                  "performance": {"total_seconds": seconds, "generation_tokens": sum(p["generated_tokens"] for p in predictions),
                      "language_model_seconds":lm_seconds if lm else None,
                      "loss_tokens_per_second":lm["tokens"]/max(lm_seconds,1e-9) if lm else None,
                      "generation_seconds": sum(p["seconds"] for p in predictions), "peak_process_rss_mib": peak_rss_mb(),
                      "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device)/1024**2 if device.type == "cuda" else None},
                  "interpretation": "original probes; untrained scores do not demonstrate ability; no test-driven training decisions"}
        write_json(directory / "evaluation.json", report)
        lines = ["# MiniLLM evaluation", "", f"Checkpoint: {ckpt}", f"Stage: {report['checkpoint']['mode']}", "",
                 f"Language model ({split if dataset else 'custom/none'}): {json.dumps(lm, ensure_ascii=False)}", "",
                 "Raw and Chat are reported separately; no combined total.", ""]
        for mode, score in report["scores"].items():
            lines.extend([f"## {mode}", "", "| Category | Correct / scored | Accuracy |", "|---|---:|---:|"])
            for category, s in score["categories"].items():
                lines.append(f"| {category} | {s['correct']} / {s['scored']} | {s['accuracy']} |")
            lines.append("")
        lines.extend(["## Diagnostics and contamination", "", "Diagnostics indicate degeneration, not quality.", "",
                      chr(96)*3+"json", json.dumps({"diagnostics":report["diagnostics"],"contamination":contamination}, ensure_ascii=False, indent=2), chr(96)*3])
        (directory / "evaluation.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        publish_manifest(directory, "evaluation-manifest.json", {"schema_version": 1,
            "evaluation_id": evaluation_id, "identity": evaluation_identity, "comparison_policy": policy,
            "checkpoint_sha256": evaluation_identity["checkpoint_sha256"], "tokenizer_fingerprint": tokenizer.fingerprint,
            "evaluation_dataset_sha256": data_info["sha256"] if data_info else None,
            "generation": asdict(config), "environment": report["environment"], "inputs": before})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True); parser.add_argument("--tokenizer", dest="tokenizer_path", required=True)
    parser.add_argument("--dataset"); parser.add_argument("--bin", dest="bin_path")
    parser.add_argument("--split", choices=["val","test"], default="val")
    parser.add_argument("--out", required=True); parser.add_argument("--config")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--sft-data", nargs="*", default=[])
    parser.add_argument("--device", default="cpu")
    for name in ("seq-len", "batch-size", "token-budget", "max-new-tokens", "seed", "top-k", "threads"):
        parser.add_argument("--"+name, type=int)
    parser.add_argument("--temperature", type=float)
    args = vars(parser.parse_args(argv))
    try:
        config_path = args.pop("config")
        supplied = read_json(config_path) if config_path else {}
        for key in EvaluationConfig.__dataclass_fields__:
            if key in args:
                value = args.pop(key)
                if value is not None:
                    supplied[key] = value
        args["benchmark"] = args["benchmark"] or None
        args["prompts"] = args["prompts"] or None
        report = evaluate_model(**args, config=EvaluationConfig(**supplied))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"FAIL evaluation: {exc}", file=sys.stderr); return 1
    print(json.dumps({"evaluation_id": report["evaluation_id"], "language_model": report["language_model"], "scores": report["scores"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
