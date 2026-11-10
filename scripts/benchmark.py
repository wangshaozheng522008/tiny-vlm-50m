#!/usr/bin/env python3
"""Report real sizes and latency for a MiniLLM checkpoint.

    python scripts/benchmark.py --ckpt runs/minillm/model.pt \
        --int4 runs/minillm/model-int4.pt --tokenizer runs/minillm/tokenizer.json

Everything printed comes from the filesystem or a measured run: parameter counts
from the instantiated model, file sizes from stat(), latency from the KV-cache
decode loop with a synchronised CUDA timer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from minillm.config import MiniLLMConfig, get_config  # noqa: E402
from minillm.evaluate import performance  # noqa: E402
from minillm.model import MiniLLM  # noqa: E402
from minillm.quantize import load_quantized  # noqa: E402
from minillm.tokenizer import MiniLLMTokenizer  # noqa: E402
from minillm.train import load_model_checkpoint  # noqa: E402
from minillm.utils import describe_device, get_device, human_bytes, peak_rss_mb  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", help="fp32 checkpoint")
    p.add_argument("--int4", help="INT4 checkpoint")
    p.add_argument("--tokenizer", help="tokenizer.json")
    p.add_argument("--config", default="minillm-87m", help="used when --ckpt is omitted")
    p.add_argument("--device", default=None)
    p.add_argument("--prefill", type=int, default=128)
    p.add_argument("--decode", type=int, default=64)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    report: dict = {"device": describe_device(device)}

    if args.ckpt:
        model, cfg, payload = load_model_checkpoint(args.ckpt, torch.device("cpu"))
        model.to(device).eval()
        report["checkpoint"] = {
            "path": args.ckpt, "step": payload.get("step"),
            "bytes": Path(args.ckpt).stat().st_size,
            "human": human_bytes(Path(args.ckpt).stat().st_size),
        }
    else:
        cfg = get_config(args.config)
        model = MiniLLM(cfg).to(device).eval()
        report["checkpoint"] = None

    report["config"] = cfg.to_dict()
    report["params"] = model.num_parameters()
    report["param_breakdown"] = model.parameter_breakdown()
    print(f"params: {report['params']:,} on {describe_device(device)}")
    for name, count in report["param_breakdown"].items():
        print(f"  {name:12s} {count:>12,}")

    if args.int4:
        size = Path(args.int4).stat().st_size
        report["int4"] = {"path": args.int4, "bytes": size, "human": human_bytes(size),
                          "under_50mb": size < 50_000_000}
        print(f"int4: {human_bytes(size)} | under 50 MB: {size < 50_000_000}")

    if args.tokenizer:
        tokenizer = MiniLLMTokenizer.load(args.tokenizer)
        report["perf_fp32"] = performance(model, tokenizer, device, args.prefill, args.decode, args.repeats)
        print(f"fp32: TTFT {report['perf_fp32']['ttft_ms']:.1f} ms, "
              f"prefill {report['perf_fp32']['prefill_tok_s']:.0f} tok/s, "
              f"decode {report['perf_fp32']['decode_tok_s']:.1f} tok/s, "
              f"peak RAM {report['perf_fp32']['peak_ram_mb']:.0f} MB")
        if args.int4:
            quantized = load_quantized(args.int4, str(device))
            report["perf_int4"] = performance(quantized, tokenizer, device, args.prefill,
                                              args.decode, args.repeats)
            print(f"int4: TTFT {report['perf_int4']['ttft_ms']:.1f} ms, "
                  f"prefill {report['perf_int4']['prefill_tok_s']:.0f} tok/s, "
                  f"decode {report['perf_int4']['decode_tok_s']:.1f} tok/s, "
                  f"peak RAM {report['perf_int4']['peak_ram_mb']:.0f} MB")

    report["peak_rss_mb"] = peak_rss_mb()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
