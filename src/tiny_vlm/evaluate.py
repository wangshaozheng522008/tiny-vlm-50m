"""Report held-out losses and greedy generation exact matches for each input mode."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import MODES, MultimodalDataset, build_prompt, collate_fn
from .generate import load_model, validate_tokenizer
from .tokenizer import EOS_ID, TinyTokenizer
from .train import evaluate
from .utils import get_device


@torch.no_grad()
def evaluate_generation(model, dataset, tokenizer, device, per_mode=5, max_new_tokens=32):
    report = {mode: {"samples": 0, "exact": 0, "examples": []} for mode in MODES}
    for index, record in enumerate(dataset.records):
        stats = report[record.mode]
        if stats["samples"] >= per_mode:
            continue
        ids = torch.tensor([build_prompt(tokenizer, record.prompt)], device=device)
        if ids.shape[1] >= model.cfg.text.max_seq_len:
            raise ValueError("evaluation prompt exceeds generation context; shorten the prompt")
        item = dataset[index]
        pixels = item.get("pixel_values")
        if pixels is not None:
            pixels = pixels.unsqueeze(0).to(device)
        generated = model.generate(ids, pixel_values=pixels, max_new_tokens=max_new_tokens,
                                   temperature=0, eos_token_id=EOS_ID)
        prediction = tokenizer.decode(generated[0, ids.shape[1]:].tolist()).strip()
        stats["samples"] += 1
        stats["exact"] += int(prediction == record.answer.strip())
        stats["examples"].append({"prompt": record.prompt,
                                  "image": str(record.image) if record.image else None,
                                  "reference": record.answer, "prediction": prediction})
        if all(report[mode]["samples"] >= per_mode for mode in set(dataset.modes)):
            break
    return {mode: {**stats, "exact_match": stats["exact"] / stats["samples"]}
            for mode, stats in report.items() if stats["samples"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", help="default: tokenizer.json beside checkpoint")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--generate-per-mode", type=int, default=5,
                        help="greedy-generation samples per mode, 0 = loss metrics only")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--out", help="optional JSON report path")
    args = parser.parse_args()
    if args.batch_size <= 0 or min(args.generate_per_mode, args.max_new_tokens) < 0:
        parser.error("batch-size must be positive; generation counts must be nonnegative")
    device = get_device(args.device)
    model, cfg = load_model(args.ckpt, device)
    tok_path = Path(args.tokenizer) if args.tokenizer else Path(args.ckpt).parent / "tokenizer.json"
    tokenizer = TinyTokenizer.load(tok_path)
    try:
        validate_tokenizer(model, tokenizer)
    except ValueError as exc:
        parser.error(str(exc))
    dataset = MultimodalDataset(args.data, tokenizer, cfg.vision.img_size,
                                 cfg.text.max_seq_len, cfg.ignore_index)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=partial(collate_fn, pad_token_id=cfg.pad_token_id,
                                           ignore_index=cfg.ignore_index))
    metrics = {"checkpoint": args.ckpt, "data": args.data,
               "teacher_forced": evaluate(model, loader, device)}
    if args.generate_per_mode:
        metrics["generation"] = evaluate_generation(model, dataset, tokenizer, device,
                                                      args.generate_per_mode, args.max_new_tokens)
    report = json.dumps(metrics, indent=2, ensure_ascii=False)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
