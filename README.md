# MiniLLM-87M

一个**中文优先的纯文本因果语言模型**：`86,528,640` 参数，INT4 权重文件小于 **50 MB**，
纯 PyTorch 手写（不依赖 HuggingFace Transformers），CPU / CUDA 均可训练和推理。

> 它不是一个能对标通用大模型的助手。它的定位是：**在「87M 参数 + 50 MB 权重」的硬约束下，
> 把中文预训练、SFT、INT4 导出、KV-cache 推理、本地 Web 对话这条链路完整跑通**，
> 适合作为端侧部署、教学和二次开发的最小可运行实现。
>
> 另外需要明确：仓库附带的演示 checkpoint 只训练了很少的步数（见 [训练记录](#训练记录)），
> 它的困惑度和生成质量远未收敛，**不能**当作可用的中文助手。所有评测数字都按实测报告，
> 没有训练过的模型会明确标注「未训练」。

---

## 1. 硬性指标

| 项目 | 目标 | 实测 |
|---|---|---|
| 参数量 | ~87M | **86,528,640** |
| 词表 | 24,576 | 24,576（ByteLevel BPE，中文优先） |
| 上下文 | 2,048 | 2,048（RoPE θ=10000） |
| INT4 权重文件 | **< 50,000,000 字节** | 见 [INT4 体积](#int4-体积) |
| 依赖 | 纯 PyTorch，无 HF Transformers | torch + tokenizers + numpy |
| 视觉能力 | 无 | 无（旧 VLM 代码/依赖已删除） |

### 参数预算

| 模块 | 每层 | 层数 | 小计 |
|---|---:|---:|---:|
| Token Embedding（与 LM Head 共享） | 15,728,640 | 1 | 15,728,640 |
| Attention（GQA 10Q/2KV，head_dim 64） | 983,040 | 16 | 15,728,640 |
| SwiGLU FFN（640 → 1792 → 640） | 3,440,640 | 16 | 55,050,240 |
| RMSNorm | 1,280 | 16 | 20,480 |
| Final RMSNorm | 640 | 1 | 640 |
| **合计** | | | **86,528,640** |

INT4 只存一份 embedding（LM Head 通过 weight tying 复用），因此 50 MB 的预算刚好够用。

---

## 2. 架构

```
input ids (B, T)
   │
   ▼
Embedding(24576, 640) ──────────────────────────┐
   │                                            │ tied
   ▼                                            │
┌───────────────────────────────┐ × 16          │
│ RMSNorm                       │               │
│ Multi-Head Attention (GQA)    │               │
│   q: 10 heads × 64 = 640      │               │
│   k/v: 2 heads × 64 = 128     │               │
│   RoPE（rotate-half）          │               │
│   scaled_dot_product_attention │               │
│ + residual                    │               │
│ RMSNorm                       │               │
│ SwiGLU: 640 → 1792 → 640      │               │
│ + residual                    │               │
└───────────────────────────────┘               │
   │                                            │
   ▼                                            │
RMSNorm → LM Head(640, 24576) ◄─────────────────┘
   │
   ▼
logits (B, T, 24576)
```

- **Pre-norm**，无 bias，dropout = 0，`tie_word_embeddings=True`。
- GQA：10 个 query head 共享 2 组 KV，`num_kv_groups=5`。优先使用
  `F.scaled_dot_product_attention(..., enable_gqa=True)`，旧版本 PyTorch 自动回退到
  `repeat_kv`，两条路径在测试里逐元素对齐。
- RoPE 表按 `(max_position, head_dim/2)` 预计算，作为 `persistent=False` 的 buffer，
  **不进入 state_dict**，量化文件里也不会重复存储。
- KV cache：prefill 一次，decode 每步单 token；cache 位置由已缓存的长度决定，
  因此 `use_cache=True/False` 的 logits 与 greedy 输出完全一致（有测试）。

---

## 3. 安装

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # torch, tokenizers, numpy, pytest
pip install -e .                       # 可选：安装 minillm-* 命令行
```

CPU 即可运行；CUDA 版 PyTorch 会自动使用 GPU。显存 4 GB 的卡建议
`--batch-size 1 --grad-checkpointing`（见 [显存建议](#显存与-4-gb-显卡)）。

---

## 4. 五条命令跑通全流程

```bash
# 0) 数据：JSONL，每行 {"text": "..."}
python scripts/prepare_text.py \
    --input data/minimind/pretrain_t2t_mini.jsonl \
    --out runs/minillm-87m/corpus --sample 24000 --max-records 40000

# 1) 只用 train split 训练 tokenizer（val/test 通过 --forbid 强制隔离）
python scripts/train_tokenizer.py \
    --zh runs/minillm-87m/corpus/train.text.jsonl --weights 1 0 0 \
    --vocab-size 24576 --out runs/minillm-87m/corpus/tokenizer.json \
    --forbid runs/minillm-87m/corpus/val.text.jsonl runs/minillm-87m/corpus/test.text.jsonl

# 2) 把三个 split 分别打包成 uint16 token 流
python scripts/prepare_text.py --input data/minimind/pretrain_t2t_mini.jsonl \
    --out runs/minillm-87m/corpus --tokenize-only \
    --tokenizer runs/minillm-87m/corpus/tokenizer.json

# 3) 预训练（可随时 Ctrl-C，用 --resume 继续）
python -m minillm.train --data runs/minillm-87m/corpus/train.bin \
    --val-data runs/minillm-87m/corpus/val.bin \
    --tokenizer runs/minillm-87m/corpus/tokenizer.json --out runs/minillm-87m/run \
    --batch-size 1 --grad-accum 16 --grad-checkpointing --device cuda

# 4) SFT（assistant-only loss）
python -m minillm.sft --init-from runs/minillm-87m/run/model.pt \
    --data data/sft/train.jsonl --val-data data/sft/val.jsonl \
    --tokenizer runs/minillm-87m/corpus/tokenizer.json --out runs/minillm-87m/sft

# 5) INT4 导出 + 文件大小检查 + 重新加载 + 生成
python -m minillm.quantize --ckpt runs/minillm-87m/sft/model.pt
python -m minillm.generate --ckpt runs/minillm-87m/sft/model-int4.pt \
    --tokenizer runs/minillm-87m/corpus/tokenizer.json --prompt "你好"
```

一条命令的端到端冒烟测试（默认用 `minillm-nano` 小模型，约 1 分钟）：

```bash
./scripts/smoke_test.sh
CONFIG=minillm-87m SAMPLE=4000 SEQ_LEN=512 STEPS=20 SFT_STEPS=10 DEVICE=cuda \
    ./scripts/smoke_test.sh      # 小规模真实 87M 全流程
```

---

## 5. 数据格式

预训练（`scripts/prepare_text.py` 的输入）：

```json
{"text": "人工智能是研究如何让机器表现出智能行为的学科。"}
```

SFT（`minillm.sft` 的输入）：

```json
{"messages": [
  {"role": "system", "content": "你是一个有帮助的中文助手。"},
  {"role": "user", "content": "什么是机器学习？"},
  {"role": "assistant", "content": "机器学习是人工智能的一个分支……"}
]}
```

- 支持多轮、多文件；system 可省略（会用默认系统提示词补齐，保证格式一致）。
- 预训练 split 在**原始文本层面**按 96/2/2 划分，之后才 tokenize，所以不会有文档跨 split；
  每个文档以 EOS 结尾，训练窗口只在本 split 内滑动。
- `prepare_text.py` 做质量过滤（长度、控制字符、可打印比例、重复字符/短语/行、字符多样性）、
  精确去重、前缀近去重和蓄水池采样；`dataset.json` 记录来源 sha256、过滤计数和 token 统计。

### Tokenizer

- ByteLevel BPE，24,576 词表，包含 256 个字节符号，**任意字符串都可编码**（不会退化成 UNK）。
- 固定特殊 token id（模型配置里硬编码，加载时校验）：

| id | token | 用途 |
|---:|---|---|
| 0 | `<pad>` | padding |
| 1 | `<bos>` | 对话开头 |
| 2 | `<eos>` | 文档/对话结束 |
| 3 | `<unk>` | 保留 |
| 4 | `<\|system\|>` | 系统提示 |
| 5 | `<\|user\|>` | 用户 |
| 6 | `<\|assistant\|>` | 助手 |
| 7 | `<\|end_of_turn\|>` | 一轮结束（受监督） |

- 语料不足时直接报错，**不会静默产出更小的词表**。
- 训练脚本默认按 `0.8 / 0.15 / 0.05` 的中/英/代码权重采样，并可用 `--forbid` 拒绝验证集。
- 实测中文效率：**0.528 token/字**（24,576 词表，真实中文语料上训练）。

### 对话模板（训练与推理共用）

```
<bos><|system|>\n{system}<|end_of_turn|>\n
<|user|>\n{user}<|end_of_turn|>\n
<|assistant|>\n{reply}<|end_of_turn|>\n...
```

SFT 只对 assistant 内容与其后的 `<|end_of_turn|>` 计算 loss；system、user、
角色标记和换行都被 `-100` 屏蔽。`--last-assistant-turn-only` 可只监督最后一轮。
超长样本保留**尾部**（最近的 assistant 回复），保证截断后仍有有效监督。

---

## 6. 训练

默认超参（可在命令行覆盖）：

| 参数 | 默认值 |
|---|---|
| `--epochs` / `--max-steps` | 1 / 0（不限） |
| `--batch-size` / `--grad-accum` | 2 / 16 |
| `--seq-len` | `max_position_embeddings`（2048） |
| `--lr` / `--min-lr` | 3e-4 / 3e-5 |
| `--warmup-ratio` | 0.02（cosine 衰减） |
| `--weight-decay` / `--grad-clip` | 0.1 / 1.0 |
| `--precision` | `auto`（CPU→fp32，CUDA→bf16/fp16） |
| `--grad-checkpointing` | 关闭 |

- **token 加权 loss**：每个 microbatch 用 `loss_sum / 组内有效 token 数` 反传，
  日志里的 loss 也是按 token 加权的交叉熵。
- **断点续训**：`state.pt` 保存权重、优化器、GradScaler、step、epoch/epoch 内位置和
  Python/NumPy/Torch/CUDA RNG 状态；`--resume` 从中恢复，CPU→CUDA 迁移时 scaler 会重新初始化。
- AMP、梯度累积、梯度裁剪、周期性验证、`metrics.json`、`best.pt` 均已实现。
- **精度实测**（T1200 Laptop，sm_75，batch 1 × seq 1024）：fp32 **1,864 tok/s**、
  bf16 1,553 tok/s、fp16 620 tok/s。fp16 autocast 在这张 Turing 卡上反而最慢，
  所以 `--precision auto` 在这台机器上选 bf16，正式训练建议显式 `--precision fp32`。
- 训练中断（Ctrl-C 或 SIGINT）会在当前 step 结束后保存再退出。

### 显存与 4 GB 显卡

87M 参数 fp32 权重约 330 MiB，AdamW 状态约 660 MiB，加上梯度约 1 GB；
1024 token、micro-batch 1 的激活在开启 `--grad-checkpointing` 后很小。
实测 T1200 Laptop（3.6 GB）用 `--batch-size 1 --grad-accum 16 --grad-checkpointing` 可以训练，
但这**不是**「任何 4 GB 卡都必然不 OOM」的承诺：驱动、PyTorch 版本、`--seq-len` 都会影响。
OOM 时按提示依次尝试 `--batch-size 1`、`--seq-len 512`、`--grad-checkpointing`。

---

## 7. INT4 体积

权重按 `group_size=128`、对称、per-group scale（fp16）做 weight-only 量化：

- Linear 与 Embedding 权重 → INT4，两个 code 打包进一个字节；
- RMSNorm 保持 FP16；
- LM Head 与 Embedding 共享存储，**只存一份**；
- 加载时整体反量化回 FP32 再前向（`load_quantized`），因此推理显存/内存与 fp32 相同，
  **INT4 只减小磁盘体积，不减小运行时内存**——这是有意的取舍，本项目不做自定义 CUDA kernel。

实测：fp32 checkpoint 409,066,855 字节（390.12 MiB），INT4 **44,730,004 字节（44.73 MB / 42.66 MiB）**，
距 50,000,000 字节预算还有 5.27 MB 余量（详见[实测结果](#13-实测结果)）。
`minillm.quantize` 会 `stat()` 真实文件大小，超过 50,000,000 字节直接退出码非 0。

---

## 8. 评测与性能

```bash
python -m minillm.evaluate --ckpt runs/minillm-87m/sft/model.pt \
    --int4 runs/minillm-87m/sft/model-int4.pt \
    --tokenizer runs/minillm-87m/corpus/tokenizer.json \
    --ppl-data runs/minillm-87m/corpus/val.text.jsonl --out runs/minillm-87m/eval.json

python scripts/benchmark.py --ckpt runs/minillm-87m/sft/model.pt \
    --int4 runs/minillm-87m/sft/model-int4.pt \
    --tokenizer runs/minillm-87m/corpus/tokenizer.json --out runs/minillm-87m/bench.json
```

`evaluate` 包含：语言/推理/指令跟随三类小基准（确定性贪心解码、精确匹配）、
验证集困惑度、TTFT / prefill / decode 吞吐、峰值内存，以及若干开放式生成样例。
**没有训练过的 checkpoint 会被标注 `trained: false`，分数仅供参考。**

---

## 9. 本地 Web 对话

```bash
python -m minillm.web --ckpt runs/minillm-87m/sft/model-int4.pt \
    --tokenizer runs/minillm-87m/corpus/tokenizer.json --port 8765
# 浏览器打开 http://127.0.0.1:8765
```

- 只监听 `127.0.0.1`，无鉴权、无联网、无图片输入。
- 多轮对话、系统提示词、温度 / Top-K / 最大新 token / 上下文长度调节、
  每条回复的 TTFT、预填充与解码速度、总耗时、清空、导出 JSON / Markdown。
- 服务端用标准库 `http.server`，不引入 Flask/FastAPI。

---

## 10. 测试

```bash
pytest                 # 80 个测试
./scripts/smoke_test.sh
```

覆盖：参数预算与权重共享、RoPE 相对性、GQA 映射、cache/no-cache 等价、
padding 不参与 loss、tokenizer 特殊 id 与往返、对话模板监督范围、数据打包与断点续训、
SFT 截断与去重、INT4 打包/误差/文件大小、Web API 与路径穿越防护。

---

## 11. 目录

```
src/minillm/
  config.py      模型配置 + 参数预算校验
  model.py       RMSNorm / GQA / SwiGLU / RoPE / KV cache / generate
  tokenizer.py   24k BPE、特殊 token、对话模板、效率统计
  data.py        打包 token 流、SFT 数据集、批采样器
  train.py       预训练 + 共用训练循环 + checkpoint/resume
  sft.py         SFT 入口（assistant-only loss）
  quantize.py    INT4/INT8 打包、体积检查、反量化加载
  generate.py    CLI 生成
  evaluate.py    小基准 / 困惑度 / 性能
  web.py         本地对话服务
  web_assets/    前端（纯文本）
scripts/
  prepare_text.py     过滤 / 去重 / 划分 / 打包
  train_tokenizer.py  只用 train split 训练 BPE
  benchmark.py        参数量 / 体积 / 延迟
  smoke_test.sh       端到端冒烟测试
tests/                pytest 测试
```

---

## 12. 与旧版 tiny-vlm-50m 的差异

旧项目是「ViT + 文本 decoder」的多模态模型（46.95M 参数，INT8 46.30 MB）。
本次按需求重构为纯文本 MiniLLM-87M：

- 删除全部视觉代码：`ViT`、图像 token、投影层、`web_assets` 中的图片 UI；
- 删除 `torchvision`、`pillow` 依赖；
- 旧的 VLM checkpoint 在 `load_model_checkpoint` / `generate` / `web` 中都会因
  `architecture` 字段不匹配而**明确报错**，不会出现「加载一半」的静默失败；
- tokenizer 词表从 16,384 扩到 24,576，特殊 token 布局重新固定为 0–7；
- 新增：真正的中文 tokenizer 训练脚本、SFT 模式、INT4 导出、本地 Web 对话、
  评测与基准脚本、断点续训状态文件。

---

## 13. 实测结果

以下数字全部来自本机真实运行，产物在 `runs/minillm-87m/`（不纳入 git）。

### 环境与数据

| 项目 | 值 |
|---|---|
| CPU | i7-11850H（16 线程），15 GB RAM |
| GPU | NVIDIA T1200 Laptop，3.6 GB，sm_75 |
| PyTorch | 2.14.1+cu126，Python 3.11 |
| 语料 | `data/minimind/pretrain_t2t_mini.jsonl` 前 40,000 条 → 过滤后 24,000 篇 |
| 划分 | train 23,040 / val 480 / test 480 篇（原始文本层面 96/2/2） |
| token | train 3,963,438 / val 79,041 / test 85,609 |
| tokenizer | 24,576 词表，24812 merges，训练语料 6.36M 字符，耗时 18 s，中文 **0.528 token/字** |

### 预训练（`minillm.train`，160 步）

| step | train loss | val loss | val ppl |
|---:|---:|---:|---:|
| 10 | — | 7.693 | 2,193 |
| 20 | — | 7.331 | 1,527 |
| 40 | — | 7.031 | 1,132 |
| 120 | 6.63 | 6.636 | 762 |
| 160 | 6.55 | **6.562** | **707** |

- 2,618,880 tokens（约 0.66 个 epoch），17.1 min，1,864 tok/s，峰值 RSS 3.8 GB。
- 训练 loss 从 ~7.1 降到 6.55，仍未收敛；这个规模的模型通常需要数十亿 token。

### 文件体积

| 文件 | 字节 | 人类可读 |
|---|---:|---|
| `model.pt`（fp32） | 409,066,855 | 409.07 MB / 390.12 MiB |
| `model-int4.pt` | **44,730,004** | **44.73 MB / 42.66 MiB** ✅ |
| `tokenizer.json` | 2,090,092 | 2.09 MB |

INT4 组成：packed 权重 ≈ 42.4 MB + per-group fp16 scale ≈ 1.34 MB + FP16 norms ≈ 42 KB。
LM Head 与 embedding 共享，只存一份；加载时反量化回 fp32（所以运行时内存与 fp32 相同）。

### 推理性能（`scripts/benchmark.py`，prefill 128 / decode 64，取中位数）

| 设备 | 精度 | TTFT | prefill | decode | 峰值内存 |
|---|---|---:|---:|---:|---:|
| CUDA T1200 | fp32 | 17.4 ms | 7,353 tok/s | **134.6 tok/s** | 1,358 MB RAM / 368 MB VRAM |
| CUDA T1200 | INT4 反量化 | 17.4 ms | 7,338 tok/s | 98.7 tok/s | 2,076 MB RAM |
| CPU i7-11850H | fp32 | 63.5 ms | 1,010 tok/s | **39.1 tok/s** | 1,277 MB |
| CPU i7-11850H | INT4 反量化 | 67.2 ms | 961 tok/s | 38.4 tok/s | 1,958 MB |

> INT4 在 CUDA 上解码更慢，因为本项目在加载时整体反量化成 fp32，之后没有 INT4 计算 kernel；
> 它的收益是**磁盘体积**，不是运行时速度或内存。这是需求里允许的「反量化加载」方案。

### 质量评测（`minillm.evaluate`，贪心解码，确定性）

| checkpoint | 语言 | 推理 | 指令跟随 | 验证集 PPL |
|---|---:|---:|---:|---:|
| 预训练 160 步 | 0% | 25% | 0% | **780.78**（loss 6.66） |
| 预训练 + SFT 60 步 | 0% | 0% | 0% | **18,872,995**（loss 16.75） |

- 12 道题里只答对 1 道（`3 × 4 =` → `12`），属于随机水平，**不构成任何能力证据**。
- 生成样例（预训练模型，`--raw`）：「人工智能是」→ `2. \n2. \n,`；SFT 模型在 Web 里对
  「什么是机器学习？」输出空字符串（24 个 token 全是换行）。这就是「远未收敛」的具体样子。
- **SFT 出现了灾难性遗忘**：在 2,850 条「抽取式摘要/续写」合成数据上以 lr 1e-4 训练 60 步后，
  SFT val loss 降到 0.0036（学到的是「抄输入」），但通用文本 PPL 从 781 崩到 1.9e7。
  真实的 SFT 需要混入通用数据、更小的学习率、以及更多样的指令数据。

### 复现命令

```bash
# 预训练（fp32 在 sm_75 上最快）
python -m minillm.train --data runs/minillm-87m/corpus/train.bin \
    --val-data runs/minillm-87m/corpus/val.bin \
    --tokenizer runs/minillm-87m/corpus/tokenizer.json --out runs/minillm-87m/run \
    --batch-size 1 --grad-accum 16 --precision fp32 --device cuda

# SFT（先用 scripts/make_sft_sample.py 造演示数据）
python scripts/make_sft_sample.py --text runs/minillm-87m/corpus/train.text.jsonl \
    --out runs/minillm-87m/sft --count 3000
python -m minillm.sft --init-from runs/minillm-87m/run/model.pt \
    --data runs/minillm-87m/sft/train.jsonl --val-data runs/minillm-87m/sft/val.jsonl \
    --tokenizer runs/minillm-87m/corpus/tokenizer.json --out runs/minillm-87m/sft-run \
    --batch-size 2 --grad-accum 8 --max-seq-len 512 --lr 1e-4 --grad-checkpointing

# INT4 + 体积检查 + 重新加载
python -m minillm.quantize --ckpt runs/minillm-87m/sft-run/model.pt --bits 4

# 评测 / 性能
python -m minillm.evaluate --ckpt runs/minillm-87m/run/model.pt \
    --tokenizer runs/minillm-87m/corpus/tokenizer.json \
    --ppl-data runs/minillm-87m/corpus/val.text.jsonl --out runs/minillm-87m/eval.json
python scripts/benchmark.py --ckpt runs/minillm-87m/sft-run/model.pt \
    --int4 runs/minillm-87m/sft-run/model-int4.pt \
    --tokenizer runs/minillm-87m/corpus/tokenizer.json --device cuda

# Web（本地）
python -m minillm.web --ckpt runs/minillm-87m/sft-run/model-int4.pt \
    --tokenizer runs/minillm-87m/corpus/tokenizer.json --port 8765
```

### 已知限制

1. **演示 checkpoint 远未收敛**，PPL 780、12 题对 1 题，生成结果没有实际可用性。
2. SFT 演示数据是脚本合成的抽取式任务，SFT 后出现灾难性遗忘（PPL 781 → 1.9e7）。
3. INT4 只减体积，不减推理内存；没有实现 INT4 计算 kernel（需求允许反量化加载）。
4. 训练脚本没有实现多卡 / FSDP / ZeRO；单卡 4 GB 以内可用，但更大规模需要另做并行。
5. Web 端是单会话、无鉴权，只适合 `127.0.0.1` 本地使用。
6. 没有做安全对齐（拒答、越狱防护等）。
