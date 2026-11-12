# MiniLLM-87M 语料 v4 设计

本文描述 PR #8 的语料与 Tokenizer 基础：**中文优先、来源可审计、规则可解释、结果可复现**。
本 PR 只建设数据与 Tokenizer 候选，不训练 87M 正式模型，不修改模型结构、损失或优化器。

## 1. 范围与非目标

范围内：

- `configs/corpus_v4.yaml`：比例、质量规则、去重、划分、Tokenizer 候选、预算和审计参数；
- 文档级 ingest、质量审核、精确/近似去重、分组划分、评测污染审计；
- 24,576 / 32,768 两个 ByteLevel BPE 候选的比较方法，以及参数/显存预算；
- 按监督目标数（不是原始字符数）计算的 100M Token 预算；
- 可恢复的 SQLite 阶段流水线、内容哈希、manifest 与冻结门禁；
- 仅使用原创合成数据的端到端演示和测试。

非目标：

- 不下载网页、不自动推断许可证、不把 REVIEW_REQUIRED 来源偷偷转入正式语料；
- 不训练 87M 模型，不修改 `MINILLM_87M` 的层数、hidden、heads 或训练核心；
- 不覆盖已有 Tokenizer、checkpoint、评测资产或历史报告；
- 不把测试集/验证集用于 Tokenizer 拟合，不用重复样本填充 Token 预算；
- 不自动选择 Tokenizer，不自动合并 PR。

## 2. 文档数据契约

每篇文档在 ingest 后至少包含：

| 字段 | 含义 |
| --- | --- |
| `document_id` | 来源内稳定 ID（内容 + 行号/路径派生），跨阶段不变 |
| `source_id` / `source_name` / `source_uri` | 来源标识与出处 |
| `license` / `license_evidence` | 许可证字符串和人类可读证据；证据不是“域名推断” |
| `language` | `zh` / `en` / `mixed` / `code` / `unknown` |
| `category` | `zh_general` / `zh_encyclopedic` / `zh_education` / `en_general` / `math_logic_code` |
| `text` | 原始文本（规范化前的可追溯版本） |
| `content_sha256` | 原始文本 SHA256 |
| `normalized_sha256` | 规范化文本 SHA256，精确去重键 |
| `document_family_id` | 同族文档（同一本书/页面/父文档）的稳定分组 ID |
| `collection_version` | 采集批次/版本 |
| `quality_flags` | 规则 ID、动作和证据的结构化列表 |
| `provenance_status` | `VERIFIED` / `UPSTREAM_DECLARED` / `DEMO_ONLY` / `UNVERIFIED` |
| `author` / `publication_date` / `acquisition_date` / `original_format` / `transformation_history` | 可选元数据，JSONL 可直接提供 |

`normalized_sha256` 与 `content_sha256` 分开保存：规范化可能合并空白/HTML，但原始内容仍可复核。
`document_family_id` 缺省时用稳定内容键；显式 `parent_id` 会保留为同族。

## 3. 来源准入

`build_registry()` 只认显式 `license_status`：

- `APPROVED`：必须同时有非空 `license_evidence`、`license_reviewer` 和 ISO 日期 `license_reviewed_at`；
- `REVIEW_REQUIRED`：保留在来源清单中说明原因，但**正式构建不进入**；
- `REJECTED`：明确排除并保留原因；
- 任何 `LICENSE_UNVERIFIED` 都不得出现在 `APPROVED` 中；即使填写 APPROVED 也会报错；
- 域名、文件扩展名、公开可访问性都不构成许可证据；
- 正式模式要求 `provenance_status != UNVERIFIED` 且 `license_policy.approved_only: true`。

正式配置当前只有三个候选来源，全部为 `REVIEW_REQUIRED`，因此
`python scripts/plan_corpus_v4.py --status-only` 返回 `NOT_READY`：

```
NO_ADMISSIBLE_SOURCES
NO_APPROVED_SOURCES
TOKENIZER_NOT_SELECTED
```

这是**正确结果**，不是待修复的 bug。补充来源需要人填写证据并重新审核。

## 4. 质量规则

规则在 `src/minillm/corpus_quality.py` 中实现，版本 `corpus-quality-v4.1`。每条触发结果都是
`QualityFinding`：

```json
{
  "rule_id": "replacement_characters",
  "severity": "error",
  "observed_value": 0.0032,
  "threshold": 0.001,
  "action": "REJECT",
  "explanation": "U+FFFD replacement characters indicate a broken decode"
}
```

| 规则 | 默认动作 | 说明 |
| --- | --- | --- |
| `empty_or_short` | REJECT | 少于 80 字符 |
| `extreme_length` | REVIEW | 超过 20,000 字符，建议切分 |
| `repeated_characters` | REJECT | 单字符连续 30 次以上 |
| `replacement_characters` | REJECT | U+FFFD 比例过高 |
| `html_residue` | REVIEW | 标签残留数量/比例超限 |
| `boilerplate_noise` | REVIEW | 导航、版权、扫码等模板句 |
| `punctuation_density` | REVIEW | 标点密度超出正常散文范围 |
| `digit_density` | REVIEW | 数字很多时只复核，不直接删 |
| `repeated_paragraph` | REJECT | 段落/句子重复比例过高 |
| `language_mismatch` | REVIEW | 中文类目中出现英文等 |
| `unprintable_ratio` | REJECT | 控制字符/未分配码点过多 |
| `pii_contacts` | REVIEW | 邮箱、URL、联系方式过多 |
| `low_information` | REVIEW | 信息字符比例过低 |

代码文本会跳过 `empty_or_short` 之外的散文型规则（`code_exempt_rules`）；数字、英文术语、
数学符号本身不会触发删除。`REVIEW` 文档进入 `review-queue.jsonl`，保留完整文本和证据，不从
原始输入删除。质量报告按来源分别记录：

- 文档数：`documents`、`kept`、`review`、`rejected`；
- 字符数：原始/规范化字符数；
- Token 数：在 Tokenizer 选定后由 `tokenization-report.json` 的
  `source_distribution` 单独给出，附 `document_ratio`、`character_ratio`、`token_ratio`。
  三种口径的分母不同，不能混用。

语言识别是中文优先的启发式：纯中文/少量英文术语为 `zh` 或 `mixed`，纯英文为 `en`，代码结构
明显为 `code`，无可用信号为 `unknown`。它是审计工具，不是语言识别模型。

## 5. 精确与近似去重

### 精确去重

- 键：`normalized_sha256`；
- 合并顺序与输入遍历顺序无关：先按内容键分组，再用并查集合并；
- 代表文档按 `(-priority, source_id, document_id)` 稳定选择；
- 保留 `exact_duplicate` 关系（被删文档、代表文档、来源、字符数），不覆盖原始记录。

### 近似去重

`NearDupPolicy` 默认 `threshold=0.85`、128 bins、32 bands × 4 rows、seed 42：

- 中文用字符 5-gram，英文用字符/词 3-gram，代码用 8-gram（默认不对代码开启）；
- 文档按 `chunk_chars=20000` 切块，单块最多 20,000 shingles、512 候选；
- 每篇文档的 shingle 集合有界，不构造 O(N²) 文档对矩阵；
- LSH 只负责召回；命中后用 **精确 Jaccard** 复核，低于阈值不合并；
- 报告给出 `candidate_probability_at_threshold`、候选截断标记和限制说明；
- 被合并文档保留 `near_duplicate` 关系和相似度。

### 划分与泄漏

- 以 `document_family_id` / 去重簇为单位，用固定 seed 的稳定哈希排序；
- 一个 group 只进入 train/val/test 其中一个，因此同族文档不会跨 split；
- 报告 `exact_normalized`、`exact_raw`、`family_cluster_violations`；
- 启用 `verify_near_cross_split` 时，分别检查 train→val/test 和 val→test 的 LSH 候选，
  用精确 Jaccard 复核并报告最大相似度；命中即阻止 READY。

## 6. 评测污染审计

`src/minillm/corpus_contamination.py` 只使用冻结评测资产（PR #6/#7），并校验
`benchmark-version.json` 的哈希和记录数。结果逐条给出：

`train_document_id`、`benchmark_id`、`overlap_type`、`similarity`、`evidence`、`disposition`。

- `PROMPT_VERBATIM`、`QA_ITEM_REPLICATION`、`QA_FORMAT_REPLICATION`、`LONG_SPAN_OVERLAP`：
  默认 `REVIEW`；
- `ANSWER_ONLY`、`SHORT_KNOWLEDGE_OVERLAP`：普通世界知识，`KEEP`；
- “北京是中国的首都”这类短陈述不因与评测题同主题而被判泄漏；
- 高风险命中必须由人类在 `decisions` 中逐条填写 `accept_risk` 或 `exclude`；未审完为
  `REVIEW_REQUIRED`；`exclude` 的文档仍在数据集中时状态为 `FAIL`；
- 未声明的自定义资产记为 `CUSTOM_UNFROZEN`，不能产生 READY。

这是**有界词法审计**，不是语义保证；报告显式记录该限制。

## 7. Tokenizer v4

### 候选与训练

- 候选：`v4-24576` 和 `v4-32768`，ByteLevel BPE，固定特殊 token 布局；
- 只用 `train.text.jsonl`；`guard_train_only()` 校验 split manifest 中 train 文件的 SHA256；
- 固定 seed 和按类目预算的确定性采样；`build_selection()` 记录选中文档数、字符数、
  各类目预算和训练配置；
- 输出目录：`artifacts/tokenizers/v4-24576/`、`artifacts/tokenizers/v4-32768/`；
  已存在的目录不会被覆盖；
- `candidate.json` 保存 `train_split_sha256`、`training_corpus_sha256`、采样参数、文本策略、
  Tokenizer 指纹；词表文件本身不可由旧版本重命名覆盖。

### 评估指标

固定评测文本 `tests/fixtures/corpus_v4/tokenizer-eval.jsonl`，覆盖：

- 中文散文/知识/教育：字符/token、token/字符、每 1024 token 的中文字符数；
- 英文：token/word；
- 数学、Python、C++：字符/token、token/文档；
- 特殊字符/emoji/控制字符往返；
- 文档边界：`encode_document()` 末尾 EOS 且解码可逆；
- 每个类目的 token 长度分布（min/p25/median/p75/p95/max/mean）。

比较对象：旧 24,576 baseline（若文件存在）、v4 24,576、v4 32,768。报告不假设“越大越好”：
`_selection_hint()` 只在中文压缩收益达到阈值且参数增长可接受时给出候选，否则要求人工决定；
自动选择只允许出现在 `--demo`，并标记 `AUTO_DEMO_ONLY`。

### 参数与显存

`parameter_budget()` 只改 `vocab_size`，保持层数/hidden/heads/FFN/位置编码不变。报告包含
embedding、LM head、是否 tied、transformer 参数、总参数、FP32 权重、梯度、AdamW 状态、
每序列 logits、保守训练内存和设备预算。当前 87M 配置下：

- 24,576：86,528,640 参数；
- 32,768：91,771,520 参数（+5,242,880，约 +6.06%），超过“约 87M”；
- 因此 32,768 不能仅凭中文压缩略有提升就被选中；正式选择必须人工写入 `selected`。

## 8. 100M Token 预算

`src/minillm/corpus_budget.py` 按**真实监督位置**计算：

```
windows = floor((raw_tokens - 1) / seq_len)
supervised = windows * (seq_len - 1)
```

- 每个文档末尾 EOS 计入 raw，但最后一个 token 不作为目标；
- 打包窗口的 shift 后每个窗口只有 `seq_len - 1` 个监督目标；
- 未进入完整窗口的尾部 token 单独报告为 `unsupervised_tail_tokens`；
- 类目计数是打包流中的精确位置，不用“字符数 ÷ chars_per_token”冒充 Token 数。

planner 对每个类目报告可用文档数、可用监督 Token、目标、缺口、所需重复系数、覆盖率和
可用占比，并给出 `READY_FOR_TRAINING` / `INSUFFICIENT_CORPUS` / `NOT_ASSESSED`。
100M 目标不等于 100M 唯一 Token：缺口只有在显式允许重复时才可能被重复采样，但 planner
**不会自动填充**；报告同时给出 `independent_effective_tokens` 和 `resampled_effective_tokens`，
默认 `allow_repetition: false`，缺口会保留为 `INSUFFICIENT_CORPUS`。

## 9. 阶段、manifest 与冻结

阶段：

```
INGESTED → NORMALIZED → QUALITY_REVIEWED → DEDUPLICATED → SPLIT
→ TOKENIZER_EVALUATED → TOKENIZED → AUDITED → FROZEN
```

每个阶段只写入新文件/新版本目录；`.v4/*.stage.json` 保存阶段键（输入内容、配置和上游键的
哈希）和输出文件 SHA256。相同输入可复用，输入变化会拒绝复用并要求新 `--out`。
`plan.json` 记录配置、来源清单、输入文件、代码实现哈希；非空目录没有匹配计划时不会被覆盖。

`dataset-manifest.json` 至少包含：

```
corpus_version, source_manifest_sha256, implementation_sha256,
cleaning_policy_sha256, dedup_policy_sha256, split_manifest_sha256,
tokenizer_fingerprint, text_policy, train/val/test_bin_sha256,
effective_train_tokens, category_token_distribution,
license_review_status, contamination_audit_status, freeze_status,
identity (含 stage hashes), files, budget, contamination
```

审计检查：许可、质量策略哈希、去重策略、split 文件/泄漏、Tokenizer 选择与指纹、bin 可重新
编码、Token 预算、污染状态、manifest 完整性。只有在**全部 PASS**且非 demo 时才写 `FROZEN`；
`license_review` 不是 APPROVED、污染未清除或预算不足都会得到 `NOT_READY`。
`verify_frozen()` 重新验证 manifest 身份和每个文件哈希，任何篡改都会报错。

## 10. 演示数据

`tests/fixtures/corpus_v4/` 全部为原创合成文本，覆盖精确重复、近似重复、短文本、重复字符、
U+FFFD、HTML 残留、模板句、标点/数字密度、重复段落、中英不匹配、控制字符、联系方式、
低信息、超长文档、同族文档和代码。`demo-sources.json` 中：

- 5 个 APPROVED 来源带合成 reviewer/date；
- 2 个 REVIEW_REQUIRED（其中一个为 LICENSE_UNVERIFIED），1 个 REJECTED（CC-BY-NC-ND-4.0）；
- 全部 `provenance_status: DEMO_ONLY`。

演示命令：

```bash
python scripts/plan_corpus_v4.py --config configs/corpus_v4.yaml \
    --demo --out reports/corpus-v4-demo
```

输出固定标记 `DEMO_ONLY`，`dataset_status=DEMO_ONLY`、`freeze_status=NOT_READY`，
不写 `FROZEN`。演示数字不能当作正式语料质量或 Tokenizer 效果。

## 11. 复现与环境

- Python 3.11；依赖：torch、tokenizers、numpy、pyyaml、pytest；
- 无网络下载、无 HF datasets、无 GPU 依赖；Tokenizer 训练可在 CPU 运行；
- SQLite 缓存、提交批次、缓冲大小在 `runtime` 中独立于内容身份记录；
- 大文件流式读写，文档级 token index 写入磁盘，避免把全部 token 聚合进内存；
- 可恢复：阶段标记最后写入；中途失败不会发布半成品标记。

## 12. 人工责任与未完成项

以下不是代码能自动完成的事项：

1. 为每个正式来源建立“来源—许可证—证据”映射，并由人类审核签字；
2. 审核 `review-queue.jsonl` 中所有 REVIEW 文档；
3. 审核污染审计的每一条高风险命中，填写 `decisions`；
4. 在比较报告基础上显式选择 Tokenizer（或选择不扩容）；
5. 用真实来源完成 100M 监督 Token 的预算；不足时接受 `INSUFFICIENT_CORPUS`，
   不得用重复样本静默填充。

在这些完成前，正式语料保持 `NOT_READY`，PR #9 的 87M 正式训练不应启动。
