"""Evaluation: tiny built-in benchmarks, perplexity, and real performance numbers.

Scores are reported exactly as measured. Every report records how many steps the
checkpoint was trained for; a checkpoint with ``step == 0`` (random init) is
labelled untrained and its accuracy is explicitly marked as not meaningful.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from .config import MiniLLMConfig
from .data import stream_documents
from .model import MiniLLM
from .quantize import load_quantized
from .tokenizer import DEFAULT_SYSTEM, MiniLLMTokenizer
from .train import load_model_checkpoint
from .utils import (
    describe_device,
    get_device,
    human_bytes,
    peak_memory_mb,
    peak_rss_mb,
    reset_peak_memory,
    set_seed,
)

# Short, deterministic tasks: the answer is a fixed string, so "exact match"
# means the generated continuation starts with that string after normalisation.
BENCHMARKS = {
    "language": [
        {"id": "capital", "prompt": "中国的首都是", "answers": ["北京"]},
        {"id": "boiling", "prompt": "在标准大气压下，水在摄氏", "answers": ["100", "一百"]},
        {"id": "poem", "prompt": "白日依山尽，黄河入海流。欲穷千里目，", "answers": ["更上一层楼"]},
        {"id": "idiom", "prompt": "画蛇添足的意思是", "answers": ["多余", "多此一举"]},
    ],
    "reasoning": [
        {"id": "add", "prompt": "1 + 1 =", "answers": ["2", "二"]},
        {"id": "mul", "prompt": "3 × 4 =", "answers": ["12", "十二"]},
        {"id": "weekday", "prompt": "如果今天是星期一，那么明天是", "answers": ["星期二", "周二"]},
        {"id": "compare", "prompt": "9 和 7 哪个大？答案是", "answers": ["9"]},
    ],
    "instruction": [
        {"id": "translate", "prompt": "把“你好”翻译成英文：", "answers": ["hello"]},
        {"id": "number", "prompt": "用汉字写出数字 3：", "answers": ["三"]},
        {"id": "format", "prompt": "只输出两个字：好的", "answers": ["好的"]},
        {"id": "refuse", "prompt": "请用一句话说明你不会做什么：", "answers": []},
    ],
}

NORMALISE = str.maketrans({" ": "", "\n": "", "\t": "", "，": "", "。": "", "：": "", ":": "", "？": "", "?": "", "！": "", "!": ""})


def normalise(text: str) -> str:
    return text.translate(NORMALISE).strip().lower()


def score_output(output: str, answers: list[str]) -> dict:
    if not answers:
        return {"exact_match": None, "contains": None, "output": output}
    head = normalise(output)[:40]
    hits = [normalise(a) for a in answers]
    return {
        "exact_match": any(head.startswith(answer) for answer in hits),
        "contains": any(answer in head for answer in hits),
        "output": output,
    }


@torch.no_grad()
def run_benchmarks(model: MiniLLM, tokenizer: MiniLLMTokenizer, device: torch.device,
                   max_new_tokens: int = 16) -> dict:
    results: dict[str, list] = {}
    for category, tasks in BENCHMARKS.items():
        rows = []
        for task in tasks:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": task["prompt"]}], add_generation_prompt=True
            )
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            output = model.generate(
                input_ids, max_new_tokens=max_new_tokens, temperature=0.0,
                eos_token_id=model.cfg.eos_token_id,
            )
            text = tokenizer.decode(output[0, len(ids):].tolist(), skip_special_tokens=True)
            rows.append({"id": task["id"], "prompt": task["prompt"], **score_output(text, task["answers"])})
        scored = [row for row in rows if row["exact_match"] is not None]
        results[category] = {
            "tasks": rows,
            "exact_match": (sum(bool(r["exact_match"]) for r in scored) / len(scored)) if scored else None,
            "contains": (sum(bool(r["contains"]) for r in scored) / len(scored)) if scored else None,
            "scored_tasks": len(scored),
        }
    return results


@torch.no_grad()
def perplexity(model: MiniLLM, tokenizer: MiniLLMTokenizer, texts: list[str],
               device: torch.device, seq_len: int, max_docs: int = 200) -> dict:
    docs = texts[:max_docs] if max_docs else texts
    total_sum = 0.0
    total_tokens = 0
    for text in docs:
        ids = tokenizer.encode_document(text)
        for start in range(0, max(len(ids) - 1, 1), seq_len):
            chunk = ids[start : start + seq_len + 1]
            if len(chunk) < 2:
                break
            input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
            out = model(input_ids[:, :-1], labels=input_ids[:, :-1])
            total_sum += float(out.loss_sum.detach())
            total_tokens += out.num_loss_tokens
    return {
        "loss": total_sum / max(total_tokens, 1),
        "perplexity": math.exp(min(total_sum / max(total_tokens, 1), 20)),
        "tokens": total_tokens,
        "documents": len(docs),
    }


@torch.no_grad()
def performance(model: MiniLLM, tokenizer: MiniLLMTokenizer, device: torch.device,
                prefill_tokens: int = 128, decode_tokens: int = 64, repeats: int = 3) -> dict:
    """Prefill/decode throughput measured with the KV cache, median of `repeats`."""
    reset_peak_memory(device)
    vocab = model.cfg.vocab_size
    prompt = torch.randint(4, vocab, (1, prefill_tokens), device=device)
    metrics: dict[str, float] = {}

    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(2):  # warmup
        model.generate(prompt, max_new_tokens=4, temperature=0.0)
    sync()

    first_token: list[float] = []
    prefill_rates: list[float] = []
    decode_rates: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        out = model(prompt, use_cache=True)
        sync()
        prefill = time.perf_counter() - started
        first_token.append(prefill)
        prefill_rates.append(prefill_tokens / prefill)

        cache = out.past_key_values
        logits = out.logits[:, -1]
        started = time.perf_counter()
        for _ in range(decode_tokens):
            token = logits.argmax(-1, keepdim=True)
            out = model(token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits = out.logits[:, -1]
        sync()
        decode = time.perf_counter() - started
        decode_rates.append(decode_tokens / decode)

    metrics.update(
        {
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "ttft_ms": statistics.median(first_token) * 1000,
            "prefill_tok_s": statistics.median(prefill_rates),
            "decode_tok_s": statistics.median(decode_rates),
            "device": str(device),
            "precision": "fp32",
        }
    )
    # End-to-end generation (prefill + decode + sampling overhead).
    started = time.perf_counter()
    generated = model.generate(prompt, max_new_tokens=decode_tokens, temperature=0.0,
                               use_cache=True)
    sync()
    end_to_end = time.perf_counter() - started
    metrics["e2e_tok_s"] = (generated.shape[1] - prompt.shape[1]) / max(end_to_end, 1e-6)
    metrics["peak_ram_mb"] = peak_rss_mb()
    if device.type == "cuda":
        metrics["peak_vram_mb"] = peak_memory_mb(device)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate MiniLLM")
    p.add_argument("--ckpt", required=True, help="fp32 checkpoint")
    p.add_argument("--int4", help="optional INT4 checkpoint for size/latency comparison")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--ppl-data", nargs="*", default=[], help='held-out JSONL {"text": ...}')
    p.add_argument("--max-docs", type=int, default=200)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--prefill", type=int, default=128)
    p.add_argument("--decode", type=int, default=64)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--samples", type=int, default=3, help="open-ended generations to show")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="runs/eval.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    model, cfg, payload = load_model_checkpoint(args.ckpt, torch.device("cpu"))
    tokenizer = MiniLLMTokenizer.load(args.tokenizer)
    tokenizer.assert_vocab_size(cfg.vocab_size)
    model.to(device).eval()
    steps = int(payload.get("step", 0) or 0)
    trained = steps > 0 or payload.get("val_loss") is not None
    print(f"[model] {describe_device(device)} | {model.num_parameters():,} params | "
          f"{'trained' if trained else 'UNTRAINED (random init)'} | steps={steps}")

    report: dict = {
        "architecture": payload.get("architecture"),
        "params": model.num_parameters(),
        "param_breakdown": model.parameter_breakdown(),
        "config": cfg.to_dict(),
        "checkpoint": {"path": str(args.ckpt), "step": steps, "trained": trained},
        "sizes": {
            "fp32_bytes": Path(args.ckpt).stat().st_size,
            "fp32_human": human_bytes(Path(args.ckpt).stat().st_size),
        },
        "device": describe_device(device),
        "tokenizer_fingerprint": tokenizer.fingerprint,
    }
    if args.int4:
        int4_path = Path(args.int4)
        size = int4_path.stat().st_size
        report["sizes"]["int4_bytes"] = size
        report["sizes"]["int4_human"] = human_bytes(size)
        report["sizes"]["int4_under_50mb"] = size < 50_000_000
        report["quantization"] = {
            key: payload[key] for key in ("quantized_bits",) if key in payload
        } or {"note": "INT4 file supplied separately"}
        print(f"[size] fp32 {human_bytes(report['sizes']['fp32_bytes'])} | "
              f"int4 {human_bytes(size)} | under 50 MB: {size < 50_000_000}")

    if not trained:
        print("[warn] checkpoint has no training steps: benchmark and perplexity scores below "
              "describe a randomly initialised model and are NOT evidence of language ability.")

    benchmarks = run_benchmarks(model, tokenizer, device, args.max_new_tokens)
    report["benchmarks"] = benchmarks
    for category, result in benchmarks.items():
        value = result["exact_match"]
        print(f"[bench] {category:12s} exact_match={'-' if value is None else f'{value:.0%}'} "
              f"({result['scored_tasks']} tasks)")

    if args.ppl_data:
        texts = []
        for path in args.ppl_data:
            texts.extend(t for t in stream_documents([path]) if t.strip())
        ppl = perplexity(model, tokenizer, texts, device, args.seq_len, args.max_docs)
        report["perplexity"] = ppl
        print(f"[ppl] loss {ppl['loss']:.4f} ppl {ppl['perplexity']:.2f} "
              f"({ppl['tokens']:,} tokens, {ppl['documents']} docs)")

    report["performance"] = performance(model, tokenizer, device, args.prefill, args.decode, args.repeats)
    perf = report["performance"]
    print(f"[perf] TTFT {perf['ttft_ms']:.1f} ms | prefill {perf['prefill_tok_s']:.0f} tok/s | "
          f"decode {perf['decode_tok_s']:.1f} tok/s | e2e {perf['e2e_tok_s']:.1f} tok/s | "
          f"peak RAM {perf['peak_ram_mb']:.0f} MB"
          + (f" | VRAM {perf['peak_vram_mb']:.0f} MB" if "peak_vram_mb" in perf else ""))

    samples = []
    prompts = [
        "用三句话介绍你自己。",
        "解释一下什么是语言模型。",
        "写一句关于春天的诗。",
    ][: args.samples]
    for prompt in prompts:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                            add_generation_prompt=True)
        out = model.generate(torch.tensor([ids], device=device), max_new_tokens=64,
                             temperature=0.8, top_k=40, eos_token_id=cfg.eos_token_id)
        samples.append({
            "prompt": prompt,
            "output": tokenizer.decode(out[0, len(ids):].tolist(), skip_special_tokens=True),
        })
    report["samples"] = samples
    if not trained:
        report["disclaimer"] = (
            "This checkpoint was not trained; benchmark accuracies and samples come from "
            "random weights and do not demonstrate any language capability."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
