#!/usr/bin/env bash
# End-to-end smoke test on synthetic data: tokenizer -> train -> quantise -> generate.
# Runs in under a minute on CPU and needs no dataset.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-python}"
OUT="runs/smoke"

echo "==> unit tests"
$PY -m pytest -q

echo "==> training a few steps on synthetic data"
$PY -m tiny_vlm.train \
  --config nano-50m \
  --synthetic 64 \
  --text-corpus /dev/stdin <<<"a red circle
a blue square
a green triangle" \
  --vocab-size 512 \
  --epochs 1 \
  --batch-size 8 \
  --max-steps 4 \
  --workers 0 \
  --out "$OUT"

echo "==> quantising to int8"
$PY -m tiny_vlm.quantize --ckpt "$OUT/model.pt" --bits 8 --group-size 64

echo "==> generating from a random image"
$PY - <<'EOF'
import torch
from PIL import Image
from tiny_vlm.generate import load_model
from tiny_vlm.tokenizer import TinyTokenizer
from tiny_vlm.data import build_transform
from tiny_vlm.utils import get_device

device = get_device()
model, cfg = load_model("runs/smoke/model-int8.pt", device)
tok = TinyTokenizer.load("runs/smoke/tokenizer.json")
img = Image.fromarray((torch.rand(224, 224, 3).numpy() * 255).astype("uint8"))
pixels = build_transform(cfg.vision.img_size)(img).unsqueeze(0).to(device)
ids = torch.tensor([tok.encode("Describe this image:", add_special_tokens=True)], device=device)
out = model.generate(ids, pixel_values=pixels, max_new_tokens=16, temperature=0.0)
print("generated:", repr(tok.decode(out[0, ids.shape[1]:].tolist())))
EOF

echo "==> OK"
