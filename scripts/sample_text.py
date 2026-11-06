"""Measure free-running text generation quality from a checkpoint.

The teacher-forced loss in ``evaluate.py`` says nothing about whether the model
can actually produce valid text: a byte-level BPE model can rank bytes well and
still emit an invalid UTF-8 sequence when decoded greedily. This script reports
both, on in-distribution prefixes taken from a held-out JSONL file:

* ``valid`` : fraction of generations whose decoded text has no U+FFFD,
* ``lead``  : characters matching the reference continuation from the start,
* sample continuations and a few fixed prompts, for human inspection.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from tiny_vlm.generate import load_model, validate_tokenizer
from tiny_vlm.tokenizer import BOS_ID, EOS_ID, TinyTokenizer
from tiny_vlm.utils import get_device

DEFAULT_PROMPTS = [
    "人工智能是",
    "秋天的树叶",
    "请介绍一下你自己。",
    "给我生成一首有关秋天的诗歌。",
    "如何提高孩子的阅读能力？",
]


def longest_common_prefix(left: str, right: str) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--data", help="held-out JSONL with {\"text\": ...} prefixes")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    parser.add_argument("--out", help="optional JSON report path")
    parser.add_argument("--device")
    args = parser.parse_args()
    if min(args.samples, args.prefix_tokens, args.new_tokens) < 0 or args.top_k < 0:
        parser.error("sample counts must be nonnegative")
    if args.temperature < 0:
        parser.error("temperature must be nonnegative")

    device = get_device(args.device)
    model, cfg = load_model(args.ckpt, device)
    tokenizer = TinyTokenizer.load(args.tokenizer or Path(args.ckpt).parent / "tokenizer.json")
    try:
        validate_tokenizer(model, tokenizer)
    except ValueError as exc:
        parser.error(str(exc))

    main_device = next(model.parameters()).device
    results = {"checkpoint": args.ckpt, "temperature": args.temperature,
               "top_k": args.top_k, "continuations": [], "prompts": []}
    if args.data:
        texts = []
        for line in Path(args.data).read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.append(json.loads(line)["text"])
        leads, valid = [], 0
        for text in texts[: args.samples]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= args.prefix_tokens + 1:
                continue
            # Cutting at an arbitrary token can split a multi-byte character;
            # only feed the model a prefix that decodes to valid UTF-8.
            prefix_ids = ids[: args.prefix_tokens]
            while prefix_ids and "\ufffd" in tokenizer.decode(prefix_ids):
                prefix_ids = prefix_ids[:-1]
            prefix = [BOS_ID] + prefix_ids
            # A truncated reference can end inside a character; trim only that tail.
            reference = tokenizer.decode(ids[len(prefix_ids):
                                              len(prefix_ids) + args.new_tokens]).rstrip("\ufffd")
            out = model.generate(torch.tensor([prefix], device=main_device),
                                 max_new_tokens=args.new_tokens,
                                 temperature=args.temperature, top_k=args.top_k,
                                 eos_token_id=EOS_ID)
            generated = tokenizer.decode(out[0, len(prefix):].tolist())
            # The fixed token budget usually cuts the last character in half;
            # that is an artefact of the evaluation, not an invalid generation.
            body = generated.rstrip("\ufffd")
            lead = longest_common_prefix(body, reference)
            leads.append(lead)
            valid += int("\ufffd" not in body)
            results["continuations"].append({
                "prefix": tokenizer.decode(prefix[1:]),
                "reference": reference,
                "generated": generated,
                "leading_match": lead,
                "valid": "\ufffd" not in body,
            })
        if leads:
            lengths = [len(c["generated"]) for c in results["continuations"]]
            results["summary"] = {
                "samples": len(leads),
                "valid_rate": valid / len(leads),
                "mean_chars": round(statistics.mean(lengths), 1),
                "empty_rate": sum(length == 0 for length in lengths) / len(lengths),
                "leading_match_mean": round(statistics.mean(leads), 2),
                "leading_match_median": statistics.median(leads),
                "exact_first_8_chars": sum(lead >= 8 for lead in leads) / len(leads),
            }
    for prompt in args.prompts:
        ids = torch.tensor([[BOS_ID, *tokenizer.encode(prompt, add_special_tokens=False)]],
                           device=main_device)
        if ids.shape[1] >= cfg.text.max_seq_len:
            results["prompts"].append({"prompt": prompt, "generated": None,
                                       "error": "prompt too long"})
            continue
        out = model.generate(ids, max_new_tokens=args.new_tokens,
                             temperature=args.temperature, top_k=args.top_k,
                             eos_token_id=EOS_ID)
        generated = tokenizer.decode(out[0, ids.shape[1]:].tolist())
        body = generated.rstrip("\ufffd")
        results["prompts"].append({"prompt": prompt, "generated": body,
                                   "valid": "\ufffd" not in body})

    report = json.dumps(results, indent=2, ensure_ascii=False)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
