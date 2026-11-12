# 规范化、语义审查与分阶段评测

本次工作修复后续实验的输入契约，并增加诊断工具；没有训练87M模型，也没有修改既有权重、Tokenizer、BIN、固定160题/40提示或历史试训练证据。

## 1. 输入规范化是实验配置

此前 `scripts/train_tokenizer.py` 对训练文本隐式执行NFKC，而编码时没有同样转换。脚本现在直接拟合所提供的文本，效率样本采用相同口径。推荐让 `prepare_text.py` 先产出文本划分，再在同一流水线中训练Tokenizer和编码。

两种输入策略：

| 策略 | 用法 |
|---|---|
| `identity-v1` | 旧实验默认值，不额外转换编码输入。继续训练既有模型保持该策略。 |
| `width-fold-prose-v1` | **新模型实验**显式选择，将散文中的全角ASCII字符及全角空格转为半角；不使用NFKC，不转换数学上标，不改代码类别、可识别源码、反引号字面量及行内 `$…$`。 |

代码识别是保守启发式。技术文档应声明 `category=code` 或使用代码围栏；本策略不能证明任意未标注文本的语义等价。它也不承诺提升模型能力，Token减少只表示编码效率变化。旧版清洗的NFC/HTML/空白规则仍独立执行。

在**新的**语料配置中设置 `"text_policy": "width-fold-prose-v1"`，使用新的输出目录。策略参与数据计划和内容身份，写入训练数据身份和新Checkpoint元数据；SFT、评测、生成与Web入口继承Checkpoint策略。INT4/INT8仅传递可选元数据，张量格式不变。未知策略、Train/Val策略不一致、已有权重与新策略不一致均拒绝。旧Checkpoint缺少该字段时明确按 `identity-v1` 处理，旧训练Manifest不添加默认字段。

`configs/pretrain-corpus.quality-v1.example.json` 提供新实验模板。来源许可留为 `unknown`，且默认排除未知许可；应填写实际来源记录，不能为了让示例运行而编造许可。完成来源和审查后，在新的输出目录执行 `prepare_text.py --config … --train-tokenizer --out …`，仅用Train拟合词表。已有Checkpoint不得换用这个新Tokenizer。

Tokenizer JSON和词表指纹保持独立；同一个词表配不同输入策略属于不同实验，不能据此直接Resume。既有模型应先在原输入策略下继续观察；如要解决旧Tokenizer拟合分布的历史偏差，需要另外设计新实验，不能修改旧词表或声称旧权重已被修好。

只读测量已有文本的策略影响：

```bash
.venv/bin/python scripts/audit_text_policy.py \
  --input data/processed/minillm-pilot-research-v3/train.text.jsonl \
  --tokenizer runs/pilot-10m-t1200-v3/tokenizer.json \
  --sample 512 --seed 42 --out reports/normalization-review
```

所有有效记录检查是否变化；Token开销只在有界、固定Seed的样本上测量。超过8192字符的文档不进入Token样本，数量单独报告。该命令不生成新训练数据。

## 2. 语义质量：审查、决策、独立产物

原有结构过滤不等于事实或逻辑核验。新工具流式扫描本地JSONL/TXT，用SQLite保存全局审查索引，不请求外部模型或下载数据。

```bash
.venv/bin/python scripts/audit_corpus_semantics.py audit \
  --input data/processed/minillm-pilot-research-v3/train.text.jsonl \
  --sample 200 --seed 42 --out reports/semantic-review
```

产物包括 `semantic-audit.json`、`review.jsonl` 和带输入/规则/文件哈希的 `review-manifest.json`。半数队列优先从标记文档中选取，其余按全文哈希和Seed抽样；不能将队列中的错误比例当作全库错误率。扫描规则只覆盖：

- 简单独立字面二元算式的不一致。跳过长表达式、含变量表达式、千分位片段、余数和带分数；除法允许所展示精度内的舍入。
- 展平在正文中的“指令/输入/输出”格式。这是领域和格式提示，不是错误标签。

引用反例、谜题、自定义规则仍可能触发。**标记不会自动删除文档；没有标记也不代表正确。** 全库事实核验为 `NOT_CHECKED`，人工审查为 `NOT_RUN`，初始状态为 `REVIEW_PENDING`。

审查者另写JSONL，每条用队列中的完整 `id`，明确理由和审查者：

```json
{"id":"<review.jsonl中的64位id>","action":"exclude","reviewer":"姓名或审查系统标识","reason":"50-60=-10，原文写10并给出错误找零结论"}
```

`action` 为 `accept` 或 `exclude`。核验不了的条目先不发布：工具要求队列每条都有有效决定，不允许用空理由补齐。接受表示一次有范围的审查，不构成任意事实的证明。

```bash
.venv/bin/python scripts/audit_corpus_semantics.py curate \
  --review reports/semantic-review --decisions reviews/decisions.jsonl \
  --out data/reviewed-source-v1
```

默认只输出明确接受的子集到新的 `accepted.jsonl`，完整保留决定及审查来源哈希。可显式传 `--include-unreviewed`，此时报告为 `PARTIALLY_REVIEWED`，列出未审查输入量；不得当作全库语义验收通过。过期输入、损坏队列、不完整/冲突决定及已有输出目录均拒绝。输出仍需经过原有质量、去重和泄漏审计。

现有MiniMind来源仍仅用于已确认的非商业研究；此工具不会补造逐条许可。**不得混合已见Train后重新随机划分，拿新Val/Test评估既有权重**；保留历史留出集和数据暴露记录，新语料版本另做历史污染审计。

## 3. 分阶段评测

旧CLI默认 `stage=legacy`，保留固定160题和40提示的原评测路径。新增配置明确指定阶段：

```bash
# 预训练：Val LM Loss/PPL + 自然前缀候选似然 + 12个Raw续写；Chat NOT_RUN
.venv/bin/python scripts/evaluate_model.py \
  --ckpt runs/pilot-10m-t1200-v3/model.pt \
  --tokenizer runs/pilot-10m-t1200-v3/tokenizer.json \
  --dataset data/processed/minillm-pilot-research-v3 \
  --config configs/evaluation.pretrain.json --device cpu --out reports/pretrain-stage

# SFT：固定Benchmark的Chat任务为主；Raw生成仅作观察
.venv/bin/python scripts/evaluate_model.py \
  --ckpt runs/minillm-sft/model.pt --tokenizer runs/minillm-sft/tokenizer.json \
  --config configs/evaluation.sft.json --device cpu --out reports/sft-stage

.venv/bin/python scripts/compare_checkpoints.py \
  --evaluations reports/baseline-pretrain reports/final-pretrain --out reports/stage-comparison
```

新资产 `data/evaluation/pretrain-v1/` 独立于旧固定题库，包含24条原创自然前缀候选题和12个Raw提示；SHA256、题量和版本固定。尚未经过独立人工复核，仅作为初步诊断，不能据其少量题目推断通用能力。

候选似然将前缀和每个候选分别编码后拼接；只在候选Token上计算一次因果Shift的NLL。参考选项不参与构造输入。默认使用总对数似然，也可显式选择 `candidate_normalization=mean`；两者存在不同长度偏差，不能混着比较。每个候选保存Token ID、长度和NLL；同分不判正确，策略导致候选编码重合则报错。

候选结果与Raw/Chat生成评分分开，未执行显示 `NOT_RUN`，不会合成总分。报告保存请求阶段和Checkpoint实际模式；对预训练模型运行SFT评测不代表模型已接受SFT。自定义Raw指令题在预训练配置下仍只能作为指令式诊断。比较要求阶段、输入策略、资产、配置和数据身份一致。旧报告与新代码报告的实现哈希不同，不能直接作为同条件比较；应重新只读评测基线。

LM窗口仍采用既有固定L输入、内部Shift、每窗L−1有效目标，边界目标与不足一窗尾部不计；未修改训练Loss、AdamW、学习率、游标或恢复数值语义。
