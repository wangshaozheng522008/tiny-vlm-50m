#!/usr/bin/env bash
# End-to-end smoke test: corpus -> split -> tokenizer -> packed bins -> pretrain
# -> SFT -> INT4 export (file-size check) -> reload -> generate.
#
# Defaults to the tiny `minillm-nano` preset so the whole run takes about a minute.
# Use a real corpus + the 87M preset for a full run:
#   CONFIG=minillm-87m SAMPLE=4000 SEQ_LEN=512 STEPS=20 SFT_STEPS=10 ./scripts/smoke_test.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
WORK="${WORK:-$(mktemp -d -t minillm-smoke-XXXXXX)}"
CONFIG="${CONFIG:-minillm-nano}"
SAMPLE="${SAMPLE:-600}"
SEQ_LEN="${SEQ_LEN:-64}"
STEPS="${STEPS:-6}"
SFT_STEPS="${SFT_STEPS:-4}"
VOCAB="${VOCAB:-1024}"
DEVICE="${DEVICE:-cpu}"
THREADS="${THREADS:-4}"
BUDGET="${BUDGET:-50000000}"
mkdir -p "$WORK"
echo "== MiniLLM smoke test =="
echo "workdir : $WORK"
echo "python  : $PY"
echo "config  : $CONFIG (vocab $VOCAB, seq $SEQ_LEN, $STEPS pretrain / $SFT_STEPS SFT steps, device $DEVICE)"

# ---------------------------------------------------------------- 1. corpus
CORPUS_SRC=""
for candidate in data/minimind/pretrain_t2t_mini.jsonl data/pretrain-minimind/train.jsonl; do
  if [ -f "$candidate" ]; then CORPUS_SRC="$candidate"; break; fi
done
if [ -n "$CORPUS_SRC" ]; then
  echo "[1/8] source corpus: $CORPUS_SRC"
  INPUT_ARGS=(--input "$CORPUS_SRC")
else
  echo "[1/8] no local corpus found; synthesising Chinese text"
  "$PY" - "$WORK/synthetic.jsonl" <<'PY'
import json, sys
words = ["人工智能", "机器学习", "模型", "训练", "数据", "语言", "中文", "代码", "数学", "推理"]
patterns = ["{a}是{b}的重要组成部分。", "研究{a}需要理解{b}和{c}。", "{a}可以帮助我们完成{b}。"]
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    for i in range(1500):
        a, b, c = (words[(i * 7 + k) % len(words)] for k in (0, 3, 6))
        handle.write(json.dumps({"text": patterns[i % 3].format(a=a, b=b, c=c) * 3},
                                ensure_ascii=False) + "\n")
PY
  INPUT_ARGS=(--input "$WORK/synthetic.jsonl")
fi

# ---------------------------------------------------------------- 2. split
echo "[2/8] split 96/2/2 raw texts (tokenizer never sees val/test)"
"$PY" scripts/prepare_text.py "${INPUT_ARGS[@]}" --out "$WORK/corpus" \
  --sample "$SAMPLE" --max-records "${MAX_RECORDS:-20000}" --min-chars 20 --max-chars 4000

# ---------------------------------------------------------------- 3. tokenizer
echo "[3/8] train tokenizer (vocab $VOCAB) on the train split only"
"$PY" scripts/train_tokenizer.py --zh "$WORK/corpus/train.text.jsonl" \
  --vocab-size "$VOCAB" --out "$WORK/corpus/tokenizer.json" \
  --report "$WORK/corpus/tokenizer-report.json" \
  --forbid "$WORK/corpus/val.text.jsonl" "$WORK/corpus/test.text.jsonl" \
  --eval-zh "$WORK/corpus/val.text.jsonl"

# ---------------------------------------------------------------- 4. pack
echo "[4/8] tokenise splits into packed uint16 streams"
"$PY" scripts/prepare_text.py "${INPUT_ARGS[@]}" --out "$WORK/corpus" \
  --tokenize-only --tokenizer "$WORK/corpus/tokenizer.json"

# ---------------------------------------------------------------- 5. pretrain
echo "[5/8] pretrain ($STEPS steps)"
"$PY" -m minillm.train --config "$CONFIG" \
  --data "$WORK/corpus/train.bin" --val-data "$WORK/corpus/val.bin" \
  --tokenizer "$WORK/corpus/tokenizer.json" --out "$WORK/run" \
  --seq-len "$SEQ_LEN" --batch-size 2 --grad-accum 2 \
  --max-steps "$STEPS" --save-every 2 --val-every 2 --log-every 1 \
  --device "$DEVICE" --threads "$THREADS" --precision auto

# ---------------------------------------------------------------- 6. sft
echo "[6/8] SFT ($SFT_STEPS steps, assistant-only loss)"
"$PY" - "$WORK/sft.jsonl" <<'PY'
import json, sys
qa = [
    ("什么是机器学习？", "机器学习是人工智能的一个分支，它让模型从数据中学习规律。"),
    ("写一个加法函数。", "def add(a, b):\n    return a + b"),
    ("用一句话解释语言模型。", "语言模型根据上文预测下一个词的概率分布。"),
    ("介绍一下你自己。", "我是 MiniLLM，一个本地运行的 87M 参数中文语言模型。"),
]
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    for index in range(40):
        question, answer = qa[index % len(qa)]
        question = f"{question}（第 {index + 1} 题）"
        handle.write(json.dumps({"messages": [
            {"role": "system", "content": "你是一个有帮助的中文助手。"},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]}, ensure_ascii=False) + "\n")
PY
"$PY" -m minillm.sft --init-from "$WORK/run/model.pt" --data "$WORK/sft.jsonl" \
  --val-data "$WORK/sft.jsonl" --tokenizer "$WORK/corpus/tokenizer.json" \
  --out "$WORK/sft" --batch-size 2 --grad-accum 1 --max-steps "$SFT_STEPS" \
  --save-every 2 --val-every 2 --log-every 1 --device "$DEVICE" --threads "$THREADS" \
  --max-seq-len 96

# ---------------------------------------------------------------- 7. int4 + size
echo "[7/8] export INT4 and check the file-size budget ($BUDGET bytes)"
"$PY" -m minillm.quantize --ckpt "$WORK/sft/model.pt" \
  --out "$WORK/sft/model-int4.pt" --bits 4 --group-size 128 \
  --budget-bytes "$BUDGET" --device "$DEVICE"

# ---------------------------------------------------------------- 8. reload + generate
echo "[8/8] reload the INT4 file and generate"
"$PY" -m minillm.generate --ckpt "$WORK/sft/model-int4.pt" \
  --tokenizer "$WORK/corpus/tokenizer.json" --prompt "你好，介绍一下你自己。" \
  --max-new-tokens 16 --temperature 0.0 --device "$DEVICE"

echo "== smoke test passed =="
echo "artifacts: $WORK"
