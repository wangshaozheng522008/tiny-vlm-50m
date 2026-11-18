# PR #14：真实 V4 Pilot 构建与 Token 供给缺口分析

本文件记录 2026-11-18 使用本地已批准的四份 Wikimedia 样本运行真实
Ingest → Quality → Dedup → Split → Tokenizer 候选评估的结果。报告是只读分析，
**没有**选择 Tokenizer、tokenize、跑污染/预算/最终审计或 Freeze。

- 分析报告：[supply-analysis.md](../reports/v4-pilot-pr14-20261118/supply-analysis.md) /
  [supply-analysis.json](../reports/v4-pilot-pr14-20261118/supply-analysis.json)
- 报告 `content_id`：`40619548c7e06570430fcd699d39058456c8efa0d24301fe145179ff3cfc4ffb`
- 最终结论：`DATA_NOT_READY`；估算缺口 102,000 有效 Token；实测 Token 为 `null`。

## 输入与身份

| Source | 类别 | 输入文档 | 输入字节 | SHA256 |
|---|---|---:|---:|---|
| `candidate-en-wikisource` | en_general | 1,836 | 22,718,156 | `24662a67…a11111` |
| `candidate-zh-wikibooks` | zh_education | 1,891 | 9,988,174 | `4abe2474…1b7eb` |
| `candidate-zh-wikipedia` | zh_encyclopedic | 3,916 | 8,122,462 | `9f79c4c1…bea49` |
| `candidate-zh-wikisource` | zh_general | 905 | 7,915,382 | `eae0513d…77e1` |

- 全部来自本地已批准 Registry；审核记录、绝对路径和真实语料均未提交。
- `candidate-zh-wikisource` 输入是 20 × 10 MiB 的派生 multistream 切片，不是完整上游发布；
  因此本文计数描述**已批准样本**，不能外推为完整语料规模。
- `candidate-zh-education`、`candidate-math-code`、`candidate-python-docs` 仍为
  `REVIEW_REQUIRED`，被 Registry 排除，未进入 Ingest。

阶段身份保存在 `artifacts/corpus/v4-pilot-pr14-20261118-v2/.v4/*.stage.json`；报告中给出
`plan_key`、`implementation_sha256`、`config_sha256`、各阶段 Key、输出 SHA256 和输入身份。
最终 `plan_key` 为 `1274a5c6…f9afd`，`implementation_sha256` 为 `2bb88dd4…de75`。

## 处理结果

| 阶段 | 结果 |
|---|---|
| Ingest | 8,548 / 8,548；每个来源输入数完全一致 |
| Quality | KEEP 6,922；REJECT 988；REVIEW 638 |
| Dedup | KEEP 6,881；删除 41（exact 14 / near 27） |
| Split | Train 6,605 / Val 138 / Test 138；Train 13,191,864 字符 |
| 家族隔离 | family cluster violations 0；exact raw/normalized 泄漏 0；near hits 0 |
| Split Manifest SHA256 | `8b0597be482f4c04f752a9b75028d3239d283aa55e30d75315657c498497b24c` |

Quality 的主要动作是短文本 `empty_or_short`（978 条 REJECT）、超长 `extreme_length`
（337 条 REVIEW）、联系方式 `pii_contacts`（255 条 REVIEW）和语言不匹配
`language_mismatch`（87 条 REVIEW）。没有放宽任何阈值。638 条 REVIEW 仍保留在
`review-queue.jsonl`，未计入 KEEP。

Dedup 只删除 KEEP 文档：exact 14 条（6 组），near 27 条（18 簇）。Val/Test 与 Train 的
家族隔离由 Split 阶段强制，Val/Test 不参与 Tokenizer 训练。

## 分类别供给与缺口

估算使用配置中的 `chars_per_token`，**不是**实测值：

| 类别 | 目标 | 估算 Train Token | 估算缺口 | 状态 |
|---|---:|---:|---:|---|
| zh_general | 408,000 | 1,213,233 | 0 | 估算达标 |
| zh_encyclopedic | 255,000 | 1,446,983 | 0 | 估算达标 |
| zh_education | 153,000 | 1,071,842 | 0 | 估算达标 |
| en_general | 102,000 | 1,591,906 | 0 | 估算达标 |
| math_logic_code | 102,000 | 0 | 102,000 | `MATH_CODE_SOURCE_REQUIRED` |
| **合计** | **1,020,000** | **5,323,964** | **102,000** | `DATA_NOT_READY` |

Split 估算：

| Split | 文档 | 字符 | 估算 Token | 计入 Train 预算 |
|---|---:|---:|---:|---|
| Train | 6,605 | 13,191,864 | 5,323,964 | 是 |
| Val | 138 | 194,770 | 93,640 | 否 |
| Test | 138 | 213,641 | 94,471 | 否 |

正式选择 Tokenizer 后，实测有效监督 Token 才可用。当前所有 `effective_train_tokens`
均为 `null`，原因固定为 `NOT_MEASURED: tokenizer not formally selected`；没有把估算值
改名为实测值，也没有用 Val/Test 填充 Train 预算。

## Tokenizer 候选评估

24,576 和 32,768 两个候选均只在 Train 上训练/评估，评估文本使用固定 `identity-v1` 文本集：

| 候选 | 参数量 | 中文 chars/token | 英文 tokens/word | 代码 chars/token | INT4 tensor payload < 50M | 训练内存（保守估计） |
|---|---:|---:|---:|---:|---|---|
| `v4-24576` | 86,528,640 | 1.5197 | 1.3333 | 2.2712 | true | 2,535,879,168 B，fits T1200 |
| `v4-32768` | 91,771,520 | 1.5755 | 1.2222 | 2.3103 | true | 2,640,736,768 B，fits T1200 |

评估结论为 `HUMAN_DECISION_REQUIRED`：32,768 的中文压缩只提升 3.67%，但词表行增加
33.33%、总参数增加 6.06%，因此无法自动判定“更优”。**没有自动选择 Tokenizer**，也没有
生成 `tokenizer-selection.json` 或 Freeze。

## 实际阻断项

最终分析报告返回 6 个 blocker：

- `MATH_CODE_SOURCE_REQUIRED` / `SOURCE_FILES_MISSING`：数学/代码来源没有本地文件，
  102,000 Token 缺口无法用现有四类补齐。
- `CATEGORY_DATA_INSUFFICIENT` / `ESTIMATED_TRAIN_SUPPLY_SHORTFALL`：`math_logic_code`
  为 0，整体估算因此仍不达标。
- `HUMAN_TOKENIZER_DECISION_REQUIRED`：评估已完成，但必须由人工签署选择。
- `TOKENIZER_NOT_FINALIZED`：未选择 Tokenizer，故未 tokenize，实测预算不可得。

`plan_corpus_v4.py --status-only` 的关键字段是：

- `status`: `READY_FOR_TOKENIZER_EVALUATION`
- `stages.tokenizer_selection`: `HUMAN_TOKENIZER_DECISION_REQUIRED`
- `stages.freeze`: `NOT_READY`
- `stages.freeze_blockers`: `FINAL_AUDIT_REQUIRED`, `HUMAN_TOKENIZER_DECISION_REQUIRED`

`--stage all` 复用了 6 个已完成阶段，随后以退出码 2 停在
`HUMAN_TOKENIZER_DECISION_REQUIRED`，并确认 `FROZEN` 不存在。这符合“不自动选择、不自动 Freeze”
的要求。

## 数学/代码候选核查

本机不存在 `data/raw/candidate-math-code` 或 `data/raw/candidate-python-docs`，也没有对应的
转换 JSONL、样本 Manifest 或人工批准记录。两者在 Registry 中均为 `REVIEW_REQUIRED`，
`candidate-python-docs` 的 PSF-2.0/0BSD 例外仍待人工审核。

因此没有下载新数据、没有伪造数学题/代码文档，也没有自行修改许可状态。接入步骤和所需证据见
[v4-math-code-onboarding.md](v4-math-code-onboarding.md)。

## 复现命令

```bash
# 只读状态检查（本地 Registry 不提交）
.venv/bin/python scripts/plan_corpus_v4.py \
  --config configs/corpus_v4_pilot_1m.yaml \
  --source-registry "$PILOT_REGISTRY" \
  --out artifacts/corpus/v4-pilot-pr14-20261118-v2 --status-only

# 真实构建到 Split
.venv/bin/python scripts/plan_corpus_v4.py \
  --config configs/corpus_v4_pilot_1m.yaml \
  --source-registry "$PILOT_REGISTRY" \
  --out artifacts/corpus/v4-pilot-pr14-20261118-v2 --stage split

# Tokenizer 候选训练/评估（stop before selection）
.venv/bin/python scripts/plan_corpus_v4.py \
  --config configs/corpus_v4_pilot_1m.yaml \
  --source-registry "$PILOT_REGISTRY" \
  --out artifacts/corpus/v4-pilot-pr14-20261118-v2 --stage evaluate

# 只读供给分析（发布新报告目录，不覆盖）
.venv/bin/python scripts/analyze_v4_pilot_supply.py \
  --config configs/corpus_v4_pilot_1m.yaml \
  --source-registry "$PILOT_REGISTRY" \
  --dataset artifacts/corpus/v4-pilot-pr14-20261118-v2 \
  --out reports/v4-pilot-pr14-20261118 \
  --base . --run-id v4-pilot-pr14-20261118
```

`$PILOT_REGISTRY` 是运行者本地的已批准审核 Registry；它包含审核记录和本地路径，不进入 Git。

## 测试与环境

- 新增测试：`tests/test_v4_real_pilot_onboarding.py` 中的 10 个 PR14 场景，`10 passed`；完整
  onboarding 文件 `68 passed`；全量回归 `966 passed in 599.22s`（`-o addopts=''`）。
- 覆盖：四类存在/数学缺失、五类存在、Quality 大量拒绝、Dedup 大量重复、Split 家族隔离、
  估算 vs 实测、Val/Test 排除、无批准来源阻断、未批准来源不能 Ingest、原始输入变化导致阶段
  身份失败、只读报告可重复、未完成数据不能 Freeze。
- 全部夹具均为原创软件测试数据，不包含受版权保护的语料。
- CPU smoke：`SYNTHETIC=1 ./scripts/smoke_test.sh` 通过；`./scripts/smoke_test.sh`（本地语料，
  默认 CPU）也通过。
- 环境：Linux x86_64，16 vCPU，15 GiB RAM，Python 3.11.17，NVIDIA T1200 Laptop 4 GiB；
  本次构建和测试均使用 CPU，未运行 GPU 训练，也未测量训练吞吐。

## 训练就绪判断

**当前不能进入预训练数据消费阶段。** 现有四类在估算口径上均超过各自目标，但：

1. `math_logic_code` 仍为 0，整体配比不完整；
2. Tokenizer 未人工选定，实测有效 Token 未知；
3. 638 条 REVIEW 未处理，正式训练前需明确处置；
4. 污染、预算、最终审计和 Freeze 均未执行。

只有在补齐已批准的数学/代码来源、人工签署 Tokenizer、完成 tokenize → 污染 → 预算 → 审计 →
Freeze 全链路后，才可评估进入下一训练阶段。
