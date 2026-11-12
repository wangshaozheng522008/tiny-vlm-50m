# 语料 v4 结果报告（PR #8）

日期：2026-11-12。结论先行：**本 PR 没有产出可正式训练的 100M 语料，也没有可正式发布的
24,576/32,768 Tokenizer**。流水线、规则、比较方法、预算和冻结门禁已经可运行；演示全部来自
原创合成 fixture，明确标记 `DEMO_ONLY`。正式语料状态为 `NOT_READY`。

## 1. 实际做了什么

- 新增 `configs/corpus_v4.yaml`，比例目标为中文 80%（通用 40、百科 25、教育 15）、
  英文 10%、数学/逻辑/代码 10%，目标 100M **监督 Token**；
- 新增文档级 ingest、13 条结构化质量规则、中文优先语言识别；
- 新增精确去重（normalized SHA256）、MinHash+LSH 近似去重（精确 Jaccard 复核、分块、
  有界候选）和按 family/cluster 的固定 seed 划分；
- 新增针对 PR #6/#7 冻结评测资产的污染审计，并区分普通世界知识与题目复制；
- 新增 24,576/32,768 ByteLevel BPE 候选的 train-only 拟合、固定文本评估、参数/显存预算；
- 新增按真实监督位置计算的 100M 预算 planner 和 `INSUFFICIENT_CORPUS` 输出；
- 新增阶段化 SQLite 流水线、阶段哈希、manifest、`verify_frozen()` 和 `FROZEN` 门禁；
- 新增 6 个测试文件、原创合成 fixture 和一条端到端 demo 命令。

## 2. 数据来源与真实语料数量

### 正式配置

`configs/corpus_v4.yaml` 当前只有 3 个候选来源，全部是 `REVIEW_REQUIRED`：

| source_id | 许可证 | 状态 |
| --- | --- | --- |
| `candidate-zh-wikipedia` | CC-BY-SA-4.0 | REVIEW_REQUIRED，未审核 |
| `candidate-zh-education` | LICENSE_UNVERIFIED | REVIEW_REQUIRED，禁止 ingest |
| `candidate-math-code` | LICENSE_UNVERIFIED | REVIEW_REQUIRED，禁止 ingest |

`APPROVED` 数量为 **0**。因此正式构建不会启动，`status-report.json` 的 blockers 为：

```
NO_ADMISSIBLE_SOURCES
NO_APPROVED_SOURCES
TOKENIZER_NOT_SELECTED
```

这不是少填了一个标志位：没有来源证据就不能伪造许可证，也不能把 REVIEW_REQUIRED 自动升级。

### 本机真实数据（只做现状审计，未进入 v4 正式构建）

- `data/minimind/pretrain_t2t_mini.jsonl`：1.24 GB，1,270,238 条记录，
  SHA256 前缀 `6dd6716c84ab3689…`；记录本身没有逐条许可字段；
- 上游声明为 `Apache-2.0 AND CC-BY-NC-2.0; upstream declaration`，
  含 **CC-BY-NC** 非商业限制，且没有“记录 → 许可证”映射，不能由域名/公开可访问性推断；
- 历史 v3 使用其 sample 配置的 128,000 条构建，processed v3 计数：
  - train 15,982,298 tokens；
  - val 331,671 tokens；
  - test 329,343 tokens；
  - 合计 16,643,312 tokens；
- v3 train 的真实语言比例：zh 87.41%、en 0.02%、code 1.67%、mixed 10.87%、
  unknown 0.03%。这是**旧 v3 的语言口径**，不是 v4 的五个目标类目；
- 旧 v3 Tokenizer：24,576 词表，指纹 `42155349a7a25cfd…`，仅作为 baseline 读取，未修改。

结论：现成本机数据的许可证状态是 `NOT_CLEARED`，且 v3 只处理了 128k/1.27M 条记录；
没有证据表明“剩余记录 + 去重后”能满足 100M 唯一监督 Token。不能把 1.24 GB 文件大小
直接换算成 100M 可用 Token。

## 3. 演示结果（合成 fixture，非正式语料）

命令：

```bash
python scripts/plan_corpus_v4.py --config configs/corpus_v4.yaml \
    --demo --out reports/corpus-v4-demo
```

产物在 `reports/corpus-v4-demo/`（已 gitignore），关键数字：

| 阶段 | 结果 |
| --- | --- |
| ingest | 29 篇合成文档，8 个来源（含 1 个 REJECTED、2 个 REVIEW_REQUIRED） |
| quality | kept 17 / review 6 / rejected 6；13 条规则均有结构化单测，demo 触发其中 12 条（`extreme_length` 由独立超长 fixture 单测覆盖） |
| exact dedup | 1 组、1 篇重复；保留 duplicate 关系 |
| near dedup | 1 组、1 篇重复；阈值 0.85，精确 Jaccard 复核；候选未截断 |
| 去重后 | 15 篇；15,782 → 15,299 字符；14 个 singleton + 1 个 size-2 簇 |
| split | train 13 / val 1 / test 1；字符 14,950 / 242 / 107；14 个 group |
| leakage | exact raw/normalized 全 0；family 违规 0；near 命中 0（max 0.0） |
| tokenize | train 21,635 raw tokens；169 个窗口；effective 21,463；尾部 3 个 token 未监督 |
| budget（demo 目标 2,048） | available 21,463，gap 0，`READY_FOR_TRAINING`（但仍受许可阻断） |
| contamination | 13 篇扫描，0 命中，`CHECKED_NO_HITS` |
| audit | `NOT_READY`，blockers `DEMO_ONLY`、`license_review`；其余检查 PASS |
| freeze | `NOT_READY`；写入 `DEMO_ONLY` 标记，**没有** `FROZEN` |

demo 的 source token 比例（注意三种口径分母不同）：

| source | 文档占比 | 字符占比 | token 占比 |
| --- | ---: | ---: | ---: |
| fixture-zh-general | 13.33% | 82.14% | 80.55% |
| fixture-zh-family | 13.33% | 1.86% | 3.00% |
| fixture-zh-encyclopedic | 20.00% | 2.50% | 4.24% |
| fixture-zh-education | 13.33% | 1.58% | 2.64% |
| fixture-en-general | 13.33% | 5.83% | 3.97% |
| fixture-math-code | 20.00% | 5.39% | 4.51% |
| fixture-unverified | 6.67% | 0.70% | 1.09% |

demo 的 15 篇文档中有 1 篇 `fixture-unverified`，它只存在于合成 demo；正式构建会排除它。
这也说明 demo 的“READY_FOR_TRAINING”只是预算算术成立，不是可训练许可。

## 4. 旧 Tokenizer 与 v4 候选

### 实际可比较的对象

- baseline：`runs/minillm-87m/corpus/tokenizer.json`，24,576 词表，
  fingerprint `42155349a7a25cfd…`；
- demo 候选：`demo-320` / `demo-384`，仅用于验证评测代码路径；vocab 太小，
  对正式 24k/32k 没有参考价值；
- v4 24,576 / 32,768：**没有训练**。正式配置 `selected: null`，且没有足够、已批准的
  train-only 文本；不会用 v3 的 val/test 或 demo 语料拟合。

因此本报告不提供“v4 比旧 Tokenizer 好多少”的结论。`tokenizer-evaluation.json` 中 demo 的
中文压缩数字（例如 demo-320 的 0.397 字符/token）只证明指标链路可运行，不能外推到正式语料。

### 参数/显存预算

`parameter_budget()` 只改 `vocab_size`，层数、hidden、heads、FFN、位置编码均不变：

| vocab | 总参数 | 相对 baseline | FP32 权重 | AdamW 状态 |
| ---: | ---: | ---: | ---: | ---: |
| 24,576 | 86,528,640 | 0 | 330.08 MiB | 660.19 MiB |
| 32,768 | 91,771,520 | +5,242,880（+6.06%） | 350.08 MiB | 700.16 MiB |

32,768 超出“约 87M”的规模，且每序列 logits 从约 96 MiB 增至约 128 MiB（seq_len=1024，
batch=1）。除非中文/知识/代码收益经人工审核足够大，否则**不因“更大”而选择 32,768**。

## 5. 100M Token 缺口

正式目标是 100M **监督** Token（中文约 80M，英文+数学/代码约 20M）。按
`windows = floor((raw-1)/seq_len)`、每窗口监督 `seq_len-1` 计算：

- 现有 v3 train 15,982,298 raw tokens，即使全部进入完整窗口，监督 Token 也只会更少，
  约 15.96M（seq_len=1024 时的实际上界）；
- 相对 100M，至少缺约 **84M**；
- 按 v4 目标类目看，中文目标 80M，而 v3 train 的中文只有 13,971,571 tokens（且旧语言口径
  不能直接当 v4 类目）；英文目标 10M，而 v3 train 英文只有 3,309 tokens；代码/数学目标
  10M，而 v3 train code 只有 266,580 tokens；
- 这不是“再采样几倍”就能解决的问题：需要新的、许可清晰的来源，以及重新去重和污染审计；
- planner 在 `allow_repetition: false` 下会输出 `INSUFFICIENT_CORPUS`，不会用重复样本补齐。

用 demo 配置验证了预算逻辑（2,048 目标、21,463 可用、gap 0），但那不是 100M 语料结论。

## 6. 污染审计

- 已实际加载并校验 6 个冻结评测资产（minillm-zh-benchmark、fixed-prompts、
  pretrain-v1 candidates/prompts、chinese-foundation-v1 candidates/prompts）；
- demo 的 13 篇 train 文档全部扫描，0 个高风险命中、0 个 allowed-knowledge 命中；
- 单元测试另外验证：复制 benchmark prompt → `REVIEW_REQUIRED`；
  “北京是中国的首都” → `CHECKED_NO_HITS`，不被当作泄漏；
- 正式 v3/真实来源**没有**经过 v4 污染审计；现有 `CHECKED_NO_HITS` 只属于 demo。
  真实语料在冻结前必须重新运行并对每条高风险命中做人工 `accept_risk`/`exclude`。

## 7. 许可与冻结状态

| 项目 | 状态 |
| --- | --- |
| 正式来源许可 | `NOT_CLEARED`：0 个 APPROVED；本机 minimind 来源含 CC-BY-NC，且无逐条映射 |
| demo 许可 | `DEMO_ONLY`（明确不是正式审核） |
| 污染审计 | demo `CHECKED_NO_HITS`；正式真实语料 `NOT_ASSESSED` |
| Token 预算 | demo 算术通过；正式 100M `INSUFFICIENT_CORPUS` |
| Tokenizer 选择 | `selected: null`；demo 的 `AUTO_DEMO_ONLY` 不是正式选择 |
| 冻结 | `NOT_READY`；没有 `FROZEN`，`verify_frozen()` 会拒绝 demo |

## 8. PR #9 是否可继续

**可以继续做“准备性工作”，但不能启动 87M 正式长训练，也不能声明 100M 语料就绪。**

PR #9 可以直接复用的输入：

- `configs/corpus_v4.yaml` 和 `docs/corpus-v4-design.md`；
- 13 条质量规则、语言/类目判定、`review-queue.jsonl` 格式；
- 精确/近似去重策略、split manifest 和 leakage 检查；
- `tokenizer_v4.py` 的 train-only 选择、指纹、固定文本评估和参数预算；
- `corpus_budget.py` 的真实监督 Token 预算和 `INSUFFICIENT_CORPUS` 语义；
- `corpus_v4.py` 的阶段、manifest、`verify_frozen()`；
- `tests/test_corpus_*.py`、`tests/test_tokenizer_v4.py`、fixture 和 demo 命令。

PR #9 开始正式训练前必须补齐：

1. 人类审核通过的来源清单（至少覆盖中文 80M、英文/数学/代码 20M 的候选来源）；
2. 每个来源的许可证证据、逐来源/逐记录映射和 reviewer/date；
3. 真实语料的 formal ingest、去重、split、污染审计和人工复核；
4. 在足够 train-only 文本上训练并人工选择 24,576 或 32,768（或维持旧词表）；
5. 重新生成 manifest，所有检查 PASS 后写 `FROZEN`；不得用本 PR 的 demo 产物冒充。

## 9. 未完成事项

- 正式来源许可审核与逐条映射（**未完成**）；
- 真实 1.24 GB 语料的 v4 ingest/质量审核（**未完成**）；
- 正式 24,576/32,768 Tokenizer 拟合（**未完成**，无合格 train-only 语料）；
- 正式真实语料污染审计（**未完成**）；
- 100M 预算补齐（**未完成**，缺口约 84M+）；
- 真实语料的 `review-queue` 人工处理（**未完成**）；
- 100M 规模下的吞吐/内存基准（只做了流式/有界设计和小数据验证）。

## 10. 风险

**P0（阻断）**

- 现成 minimind 来源含 CC-BY-NC，且无逐条许可映射；不能作为正式商用/开放训练语料；
- 100M 监督 Token 严重不足（至少缺约 84M），类别比例也不匹配；
- 32,768 参数超过“约 87M”，会改变原定模型规模，不能默认选择；
- 正式真实语料未做污染审计，不能声明无泄漏。

**P1（需人工决策）**

- 近似去重 LSH 存在概率性漏召回；报告已给概率曲线，但阈值需要按真实语料校准；
- 质量规则是启发式，`REVIEW` 队列需要人工判断；
- 旧 v3 的语言/类目口径与 v4 五类目不同，不能直接混用；
- Tokenizer 仅在 30M 字符预算上采样；正式语料更大时需要确认采样代表性和 train-only 隔离。

**P2（工程/文档）**

- `pyyaml` 已加入运行时依赖；旧环境需要重新安装；
- 大型真实语料下 SQLite 提交批次、磁盘占用和 LSH 候选质量仍需基准；
- demo 的 `tokenizers/` 与正式 `artifacts/tokenizers/` 路径分离，避免 demo 被误当正式产物。
