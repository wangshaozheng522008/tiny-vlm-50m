# tiny-vlm-50m

一个**量化后权重文件小于 50 MB** 的多模态小语言模型。输入图像 + 文字，输出文字。

- 参数量：**46.95M**
- INT8 权重文件：**46.30 MB** ✅（实测，group_size=128）
- INT4 权重文件：**24.89 MB** ✅（实测，group_size=64）
- 训练框架：PyTorch，纯手写，无 HuggingFace 依赖（只用到 `tokenizers` 训练 BPE）
- CPU 可训练、可推理

> 这不是一个能跟 GPT-4V 掰手腕的模型。它的定位是：**在 50 MB 的硬约束下，把「看图说话」这件事跑通**，
> 适合端侧部署、嵌入到 App / 嵌入式设备、以及作为多模态训练流程的最小可运行参考实现。

---

## 架构

```
        图像 (3, 336, 336)
              │
      ┌───────▼────────┐
      │  ViT Encoder   │  patch 16×16 → 441 tokens
      │  d=448, L=8    │  ~19.8M params
      └───────┬────────┘
              │ (B, 441, 448)
      ┌───────▼────────┐
      │ Linear 投影层   │  ~0.2M params
      └───────┬────────┘
              │ (B, 441, 448)
              ▼
   ┌──────────────────────────────────┐
   │  拼接：[图像 tokens | 文字 tokens] │   → 441 + 256 = 697
   └──────────────┬───────────────────┘
                  │
         ┌────────▼─────────┐
         │  Causal Decoder  │  d=448, L=8, heads=7
         │  d=448, L=8      │  ~19.4M params
         └────────┬─────────┘
                  │
            ┌─────▼─────┐
            │  LM Head  │  与 token embedding 共享权重
            └─────┬─────┘
                  ▼
              文字输出
```

图像 token 作为**前缀**拼在文字序列前面，decoder 在 `[图像, 文字]` 整体上做因果注意力。
于是每个文字 token 都能看到完整图像，而文字部分保持自回归。这是 LLaVA 式的前缀方案，
但省掉了单独的 cross-attention 层——在这个规模下，每一层都得省着用。

### 参数预算

| 模块 | 参数量 | 说明 |
|---|---:|---|
| ViT Encoder | 19.82M | patch embed 0.34M + 441 位置编码 0.20M + 8 × block 19.27M |
| 投影层 + 视觉位置编码 | 0.40M | Linear(448→448) 0.20M + vision_pos 0.20M |
| Token Embedding | 7.34M | 16k BPE 词表 × 448 |
| Text Decoder | 19.39M | 8 × Transformer block + 位置编码 |
| LM Head | 0（共享） | 与 token embedding 绑定 |
| **合计** | **46.95M** | |

词表是单个模块里最大的一笔：16k 词表占 7.34M 参数，是整个文本侧的 27%。
如果你的数据以中文为主，把这笔预算换到更大的词表（24k）通常比换到更多层更划算——
16k BPE 在中文上分词效率明显偏低。这也是 Roadmap 里 `d448 L7/7 + 24k 词表` 那个变体的动机。

**为什么是 336px 而不是 224px：** 分辨率几乎不吃参数预算（只多 0.19M 的 pos_embed，
196 → 441 个 token），但图像 token 数涨 2.25 倍，对小目标和小字号文字的识别提升最直接。
代价是注意力计算量按 token 数平方增长，CPU 训练明显变慢。

---

## 快速开始

```bash
git clone https://github.com/wangshaozheng522008/tiny-vlm-50m.git
cd tiny-vlm-50m

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

一条命令跑通全流程（合成数据，CPU 上不到一分钟，不需要任何数据集）：

```bash
./scripts/smoke_test.sh
```

---

## 训练

### 数据格式

JSONL，每行一个 `{image, caption}`，`image` 是相对 JSONL 文件所在目录的路径：

```json
{"image": "images/cat.jpg", "caption": "a grey cat sleeping on a blue sofa"}
{"image": "images/street.jpg", "caption": "a busy street with red buses"}
```

参考 `examples/captions.example.jsonl`。

### 先训 tokenizer

词表需要单独训练，且 `vocab_size` 必须和模型配置一致：

```bash
python - <<'EOF'
from tiny_vlm.tokenizer import TinyTokenizer
texts = [l for l in open("corpus.txt", encoding="utf-8") if l.strip()]
TinyTokenizer.train(texts, vocab_size=16384, save_path="runs/tiny-vlm/tokenizer.json")
EOF
```

### 训练

```bash
python -m tiny_vlm.train \
  --config tiny-50m \
  --data data/captions.jsonl \
  --tokenizer runs/tiny-vlm/tokenizer.json \
  --epochs 3 \
  --batch-size 16 \
  --lr 3e-4 \
  --out runs/tiny-vlm
```

（`--batch-size 16` 在 4GB 显卡上会 OOM，GPU 的 batch 配置见下面「GPU 训练」一节。）

也可以在训练时直接传 `--text-corpus` 让脚本自己训 tokenizer。

**Loss 只算在回答部分。** prompt（`Describe this image:`）对应的 label 被置为 `-100`，
模型不会被训练去复述 prompt——这个细节在小模型上影响很明显，否则模型会把大部分容量
浪费在记忆一个固定前缀上。

冒烟测试（合成数据，验证流程通不通）：

```bash
python -m tiny_vlm.train --config nano-50m --synthetic 256 --epochs 1 --max-steps 20
```

### GPU 训练

`get_device()` 会自动选 CUDA，cuda 上自动开 AMP（fp16），无需额外参数。
但 PyTorch 默认的 pip 包是 **CPU-only** 的，得先换成 CUDA 版：

```bash
pip uninstall -y torch torchvision
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
```

国内走 `download.pytorch.org` 可能很慢或断流（大 wheel 走 `download-r2.pytorch.org` CDN，
代理容易在中途 SSL EOF）。加 `--retries 20 --timeout 300` 会好一些。

#### 实测（NVIDIA T1200 Laptop, 4GB, sm_75）

336px / 697 token / `tiny-50m` / AMP fp16 / AdamW：

| batch | ms/step | 峰值显存 | samples/s |
|---:|---:|---:|---:|
| 1 | 1194 | 932 MB | 0.8 |
| 2 | 993 | 1119 MB | 2.0 |
| 4 | 1952 | 1592 MB | 2.0 |
| **8** | **3872** | **2535 MB** | **2.1** |
| 16 | OOM | — | — |

同样配置在 CPU 上是 0.53–0.74 samples/s，所以这块卡大约快 **3 倍**。

**4GB 显存下 batch 最大就是 8。** 想要更大的等效 batch 用梯度累积：

```bash
python -m tiny_vlm.train --config tiny-50m --data data/captions.jsonl \
  --batch-size 8 --grad-accum 4 --device cuda --out runs/tiny-vlm
```

训练结束会打印峰值显存占用，方便判断还能不能加 batch：

```
[mem] peak GPU allocation 1791 MB / 3715 MB (48%)
```

两点注意：

- 吞吐在 batch ≥ 2 之后就平了（~2 samples/s）。瓶颈是显存带宽不是算力，
  加 batch 换不来吞吐，只换显存。
- sm_75 是 Turing，**没有原生 BF16**，所以 AMP 用的是 fp16，不要手动切 bf16。
  （`torch.cuda.is_bf16_supported()` 在这块卡上会返回 `True`，别信它。）

---

## 量化

权重仅量化（weight-only），每个 group 独立算 symmetric scale：

```bash
python -m tiny_vlm.quantize --ckpt runs/tiny-vlm/model.pt --bits 8 --group-size 128
python -m tiny_vlm.quantize --ckpt runs/tiny-vlm/model.pt --bits 4 --group-size 64
```

输出会直接打印真实文件大小和是否超预算：

```
int8 (group_size<=128, effective [112, 128]) -> runs/tiny-vlm/model-int8.pt
  46.30 MB / 50 MB budget  [OK]
  reloaded and ran a forward pass: logits (1, 8, 16384)
```

几个设计取舍：

- **只量化 `nn.Linear` 和 `nn.Embedding` 的 2D 权重**，norm / bias / 位置编码保持 fp16。
  这些张量加起来不到 1 MB，量化它们省不了多少，反而更容易掉点。
- **4-bit 真的做了打包**（两个 code 塞一个 byte），所以报出来的体积是真实文件大小，
  不是「参数量 × 0.5」的估算。
- **加载时反量化**回 fp32 再送进普通 `TinyVLM`，不依赖任何自定义 kernel。
  代价是推理显存不省，收益是不用为不同后端各写一套算子——50M 参数以下这个交换是划算的。
- **group 大小会自动取整除值，而不是补零。** 按 input 维分组要求 input 能均分。
  `d_model=448` 配 `group_size=128` 时，天真的做法是把 448 补到 512，每个这样的权重
  白白膨胀 14%——这一步把 INT8 推到了 **50.85 MB，直接超预算**。改成在 `(group_size/2, group_size]`
  里找 448 的最大因数（112），体积降到 **46.30 MB**，精度不受影响。
  只在上半区找是关键：如果允许很小的因数，素数长度的 input 会退化成「每个元素一个 scale」，
  反而更大；找不到就回退到补零。有效 group 随张量而异，所以存在 payload 里。

  > 两个都是实测踩出来的坑：`state_dict()` 返回的绑定权重是两个**不同的 tensor 对象**
  > （`id()` 不同、`data_ptr()` 相同），用 `id()` 去重会失效导致 embedding 存两遍
  > （33.90 MB → 27.81 MB）；而补零膨胀把 46.30 MB 顶到了 50.85 MB。

用法：

```python
from tiny_vlm.quantize import load_quantized

model = load_quantized("runs/tiny-vlm/model-int8.pt")
```

---

## 推理

```bash
python -m tiny_vlm.generate \
  --ckpt runs/tiny-vlm/model-int8.pt \
  --tokenizer runs/tiny-vlm/tokenizer.json \
  --image cat.jpg \
  --prompt "Describe this image:"
```

`--temperature 0` 是贪心解码（默认），大于 0 走 top-k 采样。

Python 里直接用：

```python
import torch
from PIL import Image
from tiny_vlm import TinyVLM, TinyTokenizer, get_config
from tiny_vlm.data import build_transform
from tiny_vlm.quantize import load_quantized

cfg = get_config("tiny-50m")
model = load_quantized("runs/tiny-vlm/model-int8.pt")
tokenizer = TinyTokenizer.load("runs/tiny-vlm/tokenizer.json")

image = Image.open("cat.jpg").convert("RGB")
pixels = build_transform(cfg.vision.img_size)(image).unsqueeze(0)

prompt = "Describe this image:"
input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)])
out = model.generate(input_ids, pixel_values=pixels, max_new_tokens=32, temperature=0.0)
print(tokenizer.decode(out[0, input_ids.shape[1]:].tolist()))
```

---

## 项目结构

```
src/tiny_vlm/
├── config.py     # 模型配置 + PRESETS（tiny-50m / nano-50m）
├── model.py      # ViT encoder + 投影层 + causal decoder + generate
├── tokenizer.py  # ByteLevel BPE 封装，固定 special token id
├── data.py       # CaptionDataset / 合成数据集 / collate
├── train.py      # 训练循环（AdamW + warmup-cosine + AMP + 梯度累积）
├── quantize.py   # int8 / int4 权重量化、打包、反量化加载
├── generate.py   # 单图推理 CLI
└── utils.py      # 设备选择、显存/体积估算
tests/test_model.py       # 16 个单元测试
scripts/smoke_test.sh     # 端到端冒烟测试
```

---

## 已知取舍

这些不是 bug，是在 50 MB 预算下有意做的妥协：

1. **图像 token 之间也是因果注意力。** 真正的 ViT 里 patch 之间是双向的。
   这里为了复用同一套 decoder block，图像前缀内部也走因果掩码。
   代价是后面的 patch 看不到前面的——信息流是单向的，能 work，但不是最优。
   336px 下前缀有 441 个 token，这个限制比 224px 时更值得修。
   要修的话，得给图像段单独跑一遍双向注意力再拼进去。

2. **没有 KV cache。** `generate` 每步重算整个前缀。
   好处是代码短、不会有 cache 实现 bug；代价是长序列生成慢——697 的前缀长度下比 384 配置明显。
   但 ViT 编码只做一次（`encode_images` 的结果被缓存复用），所以每步重算的主要是 decoder。
   目标是端侧小模型的话，这里值得补。

3. **单轮对话。** 没有多轮历史拼接的逻辑，每次调用独立。

4. **加载时反量化**，推理内存不省（见上文「量化」一节）。

5. **16k 词表对中文偏小。** 以中文语料为主的话，分词效率会明显低于 24k+ 的词表。
   扩词表是 Roadmap 里优先级最高的一项。

6. **都没有在真实数据上训过。** 目前所有验证都是结构/流程层面的（参数量、体积、
   量化误差、梯度是否流动）。**没有任何质量结论**——比如 int4 反量化后的
   argmax 一致率是 0.5，但那是随机初始化权重上的数字，训练之后才有参考价值。

7. **`nano-50m` 预设没验证过效果**，只是给冒烟测试用的更小配置。

---

## Roadmap

- [ ] KV cache + 增量解码
- [ ] 图像段双向注意力（把 ViT 输出和文本段解耦）
- [ ] 词表扩到 24k，改善中文分词
- [ ] 多图交错输入（`<image>` 占位 token 方案）
- [ ] ONNX / GGUF 导出
- [ ] 在 COCO Captions 子集上训一个可用的公开 checkpoint

---

## License

MIT
