# V5 千万 Token 中文语料扩充与冻结

**状态：`BLOCKED_SOURCE_APPROVAL`（候选库存已就绪；正式冻结与 GPU 训练未运行）**

本 PR 交付 `minillm-corpus-v5-language-10m-v1` 的可复现候选库存、审批门禁、10M 训练预算预览和测试，但**没有**正式冻结语料，也没有启动 GPU 训练。全部 6 个来源仍是 `REVIEW_REQUIRED`；没有人类审批记录，因此 `FROZEN`、`freeze.json` 和正式 `dataset-manifest.json` 均不存在。候选语料不是可训练语料，不得被标记为 trainable。

| 项目 | 结果 |
|---|---|
| 候选 inventory 状态 | `CANDIDATE_INVENTORY_READY` |
| 候选清单 content id | `3ed83ea299a6439e0b5bb279089ee642415e3578fa299be9f1f062cf7bb1e93f` |
| 候选清单文件状态 | `CANDIDATE_ONLY_NOT_TRAINABLE` |
| 清理后唯一候选 | 30,209 篇 / 28,661,960 tokens / 49,841,686 字符 |
| 12–15M 候选选包 | 13,994,640 effective tokens（原始 14,009,038） |
| 15M 候选冲刺目标 | 13,994,640 / 15,000,000（差 1,005,360；`INSUFFICIENT_CORPUS`） |
| 10M 冻结预排 | 10,569,551 raw / 10,558,383 effective；预留 4% val/test 后约 10,136,047 effective |
| 10,000,848 预算 | 1,222 次完整优化器更新 × 8,184 supervised tokens |
| 正式冻结 | `NOT_RUN` / `BLOCKED_SOURCE_APPROVAL` |
| GPU 训练 | `NOT_RUN` |
| 历史 1M / 2M 语料 | 未修改 |
| 历史 Val / Test | 未修改 |

`CANDIDATE_INVENTORY_READY` 指 10M 目标已有足够、配比合格的候选供给；15M 候选冲刺仍为 `INSUFFICIENT_CORPUS`，不能把 13.99M 说成 15M。

复现命令：

```bash
.venv/bin/python scripts/prepare_v5_language_corpus.py inventory
.venv/bin/python scripts/prepare_v5_language_corpus.py prepare
.venv/bin/python -m pytest -q -o addopts='' tests/test_v5_language_corpus.py
```

`inventory` 只写 `reports/` 下的候选报告，不写 `FROZEN`；`prepare` 只写审批请求和 `BLOCKED_SOURCE_APPROVAL` 状态。`build` 需要 `--approval` 和 `--request`，没有人类批准文件时会在构造正式管线前失败。

## 1. 目标与不变量

- **目标**：10,000,848 个 effective supervised tokens，按完整优化器更新口径为 1,222 × 8,184。`seq_len=1024`；每个窗口贡献 1,023 个监督目标；1 次更新 = 8 个窗口。
- **类别比例**（容差 ±1 个百分点）：中文通用 45%、中文百科 30%、中文教育 15%、英文通用 10%、数学/代码 0%。不放宽质量阈值来凑 token，不使用重复采样或重复窗口。
- **模型**：参数量保持 86,528,640；不修改架构、位置编码、训练循环或 INT4 量化策略。INT4 产物仍要求 ≤50 MB。
- **Tokenizer**：复用 PR #16 冻结 tokenizer，vocab 24,576，fingerprint `845a24b9f21458b1d2161c60dfc8a77d5ed40235ece7ef46891677e423727b8a`；不重新训练、不改变词表。
- **范围**：本 PR 不包含数学、代码、逻辑推理语料，也不运行正式 GPU 训练。
- **兼容性**：1M content id `7222651ef09e4e09c2d613474f50712e8233d1d4ff9017bda897e597ebdb6da5`，2M content id `dcf709c91a5499f9a3b430c935ce5ede08d8abbad1456bf89cbeea7bcb92f8d1`；两者均保持字节不变。
- **新验证/测试集**：V5 配置使用 `family_cluster_hash`、96/2/2 分组切分，从 V5 候选组中独立生成；不复用历史 train/val/test，只复用冻结 tokenizer。正式切分在审批后的 freeze 阶段生成，本 PR 不提交任何候选训练数据。
- **历史固定评测**：`data/evaluation/chinese-language-v1/tasks.jsonl` 等 7 个 frozen asset 仍参与污染审计；历史 Val/Test 仍用于交叉去重，不替换、不扩写。

## 2. 来源清单与审批状态

> “上述来源仅为候选，不代表已经批准。”
>
> “新增来源及扩大既有来源使用范围必须取得人工审批。”
>
> “未经批准不得进入正式冻结语料，不得伪造审批记录。”

六个来源全部为 `license_status: REVIEW_REQUIRED`，`training_scope` 不存在，商用/再分发/模型训练/许可义务均为 `PENDING`。下表是**候选**登记，不是批准清单。

| 来源 | 类别 | 发布快照 / 上游 | 许可（待审） | 许可状态 | 商用 / 再分发 / 训练 |
|---|---|---|---|---|---|
| `candidate-zh-wikipedia` | `zh_encyclopedic` | `zhwiki-20261101-pages-articles` / https://dumps.wikimedia.org/zhwiki/20261101/ | CC-BY-SA-4.0 | `REVIEW_REQUIRED` | `PENDING` / `PENDING` / `PENDING` |
| `candidate-zh-wikinews` | `zh_general` | `zhwikinews-20261101-pages-articles` / https://dumps.wikimedia.org/zhwikinews/20261101/ | CC-BY-2.5 | `REVIEW_REQUIRED` | `PENDING` / `PENDING` / `PENDING` |
| `candidate-zh-wikivoyage` | `zh_general` | `zhwikivoyage-20261101-pages-articles` / https://dumps.wikimedia.org/zhwikivoyage/20261101/ | CC-BY-SA-4.0 | `REVIEW_REQUIRED` | `PENDING` / `PENDING` / `PENDING` |
| `candidate-zh-wikibooks` | `zh_education` | `zhwikibooks-20261101-pages-articles` / https://dumps.wikimedia.org/zhwikibooks/20261101/ | CC-BY-SA-4.0 | `REVIEW_REQUIRED` | `PENDING` / `PENDING` / `PENDING` |
| `candidate-zh-wikiversity` | `zh_education` | `zhwikiversity-20261101-pages-articles` / https://dumps.wikimedia.org/zhwikiversity/20261101/ | CC-BY-SA-4.0 | `REVIEW_REQUIRED` | `PENDING` / `PENDING` / `PENDING` |
| `candidate-en-wikisource` | `en_general` | `enwikisource-20261101-pages-articles` / https://dumps.wikimedia.org/enwikisource/20261101/ | Public-domain works plus CC-BY-SA-4.0 contributions | `REVIEW_REQUIRED` | `PENDING` / `PENDING` / `PENDING` |

来源登记与证据在 [`configs/corpus_v5_sources.review.json`](../configs/corpus_v5_sources.review.json)。每个来源记录 source URL、release/version、license、本地 raw JSONL 与 `.xml.bz2` 的 SHA256/字节数、文档数、字符数、估算 tokens、转换历史、`upstream_snapshot_uri` 和许可参考链接。原始 dump、JSONL 和缓存均不提交；登记中的路径是生成机的本地绝对路径。

| 来源 | 文档数 | 原始字符数 | 估算 tokens（字符/1.422 等） | 输入 JSONL SHA256 |
|---|---:|---:|---:|---|
| `candidate-zh-wikipedia` | 1,562,728 | 1,095,459,302 | 770,365,589 | `46f63cef5be5cd3c0eaa14c268169b78046c93b2e4b4413bb3432904021dae58` |
| `candidate-zh-wikinews` | 18,659 | 6,491,950 | 4,627,345 | `7c6c1a4c3c54f5a157a9be1c25bbaa76994e14f678b33d1d9d38a8d390d70671` |
| `candidate-zh-wikivoyage` | 5,598 | 6,156,468 | 4,432,476 | `ead0492ed5a293f1258280a12452c9d5b80dd68781f0a5f6271c85aa36f164ef` |
| `candidate-zh-wikibooks` | 7,668 | 17,809,379 | 11,775,973 | `492c9fff68fa854fbbbf5470e45c0985b769ef5b4f2adc8e40257b7050ac5849` |
| `candidate-zh-wikiversity` | 3,555 | 6,359,645 | 4,396,018 | `88cc63bdf03a49e13b63f7e121cb216d538f3919b633e10b91cf65af7f4d42b7` |
| `candidate-en-wikisource` | 169,732 | 1,937,748,972 | 518,017,908 | `89de7544153da2b70e931ef65cc33fc6cf1391fdf00891879504e6597ec7f623` |

`en-wikisource` 同时包含公有领域作品和 CC-BY-SA 贡献，商用/再分发/署名/同条件分享义务必须由人类逐项确认；不能因为“多数是公有领域”就推定整包可训练。任何新来源（例如 FineWeb2、Wikidata 或其他 web corpus）或扩大 zhwiki/en-wikisource 的采样范围，都需要重新走审批，不能在候选阶段自动扩权。

### 2.1 审批材料

`prepare` 生成：

- `reports/v5-language-corpus-10m-v1/prepare/source-approval-request.json`：6 个来源、文件 SHA256、请求的 10,000,848 预算、45/30/15/10/0 比例、内部研究用途、非商用、非再分发、非公开发布权重。
- `reports/v5-language-corpus-10m-v1/prepare/source-review.pending.json`：当前登记副本。
- `reports/v5-language-corpus-10m-v1/prepare/status.json`：`BLOCKED_SOURCE_APPROVAL`，`approved_sources: 0`。

人类批准文件必须包含：`corpus_version`、`decision=APPROVED`、真实 reviewer、`recorded_at`、逐字确认、精确 10,000,848、精确类别比例、每个来源的 release 与文件 SHA256、以及 `approval_request` 的文件身份。`validate_approval` 会拒绝占位 reviewer、错误版本、错误文件集合、被篡改的 request 和任何仍是 `REVIEW_REQUIRED` 的来源。当前没有这样的文件；本 PR 不生成、不推断、不伪造。

`build` 在 `validate_approval` 之前不会构造正式管线；即使 token 数学上足够，缺少 approvals 也会失败。正式 `verify_frozen` 还会重新检查 source review、预算、污染审计、tokenizer 复用、stage key 和 `FROZEN` 身份。

## 3. 候选库存：采样、清洗、去重

### 3.1 确定性采样与断点恢复

- 采样排序键：`sha256(seed \0 source_id \0 record_id)`；`seed=42`；每个来源独立配额，不按行号或文件顺序取样。
- 大文件流式读取，每 1,000 条写状态，输出 `.part`，完成后原子改名为 `.jsonl`；恢复时从已提交字节继续。
- 状态绑定输入路径、SHA256、sample rate、seed 和输出字节数；输入变化会拒绝恢复。
- `download_resumable` 使用 HTTP Range，校验 206/Content-Range 和官方 SHA1；不完整下载不会发布为目标文件。

### 3.2 清洗与长文切分

44,525 篇被采样文档（34,098,688 sampled tokens）中，有 401 篇超过 20,000 字符上限。它们只在段落/句子边界切分，切分后 45,615 个候选片段；拼接结果逐字节等于原文，未改写、未截断、未丢段。长文切分是“保留整篇/整段”的实现，不是降低质量阈值。

质量 KEEP 30,633 个片段；其中 68 个精确重复先被 exact-hash 去掉，剩 30,565 个片段（28,844,180 tokens）进入近重复。质量动作（按片段计）：

| 动作 | 数量 | 处理 |
|---|---:|---|
| `KEEP` | 30,633 | 进入候选 |
| `REJECT` | 13,676 | 垃圾/空短/重复字符/重复段落 |
| `REVIEW` | 1,306 | 排除出候选供给；不是批准 |

主要规则触发：`empty_or_short` 13,607、`language_mismatch` 441、`pii_contacts` 532、`punctuation_density` 350、`boilerplate_noise` 100、`digit_density` 77、`extreme_length` 0（切分后不再触发）、`html_residue` 5、`repeated_characters` 40、`repeated_paragraph` 29。`REVIEW` 全部不进入候选，也不被当作“人工批准”。

| 来源 | 采样率 | 采样文档 | 采样 tokens | 保留片段 | 保留 tokens | REVIEW 文档 | REJECT 文档 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `candidate-zh-wikipedia` | 0.005 | 7,678 | 3,662,518 | 6,425 | 3,508,108 | 45 | 1,215 |
| `candidate-zh-wikinews` | 1.0 | 18,659 | 4,626,387 | 9,048 | 4,507,655 | 60 | 9,552 |
| `candidate-zh-wikivoyage` | 1.0 | 5,598 | 4,432,261 | 4,090 | 3,158,419 | 173 | 1,370 |
| `candidate-zh-wikibooks` | 1.0 | 7,668 | 11,751,385 | 6,272 | 10,146,215 | 614 | 1,069 |
| `candidate-zh-wikiversity` | 1.0 | 3,555 | 4,395,216 | 3,070 | 3,542,482 | 160 | 405 |
| `candidate-en-wikisource` | 0.008 | 1,367 | 5,230,921 | 1,660 | 3,981,301 | 254 | 65 |
| **合计** |  | **44,525** | **34,098,688** | **30,565** | **28,844,180** | **1,306** | **13,676** |

注意：`candidate-zh-wikinews` 的 9,552 个 REJECT 主要是短讯/占位页；`candidate-en-wikisource` 的 REVIEW 主要是长作品中的 PII/模板信号，切分后已尽量保留可用部分，但仍不把 REVIEW 当训练数据。

### 3.3 近重复、历史交叉与污染

- MinHash-LSH：128 bins、32 bands × 4 rows、threshold 0.85、5-gram（zh）/3-gram（en）；30,565 → 30,286，删除 279 篇近重复，`candidate_truncations=0`。
- 历史 Val/Test 交叉：276 篇历史文档（1M 的 val/test；2M 文件字节相同），检查 30,286 篇候选；删除 76 篇 exact + 1 篇 near，共 77 篇。历史文件 SHA256 未变：val `25229b9c…4d`、test `b58cb2c0…c4`。
- 正式评测污染：7 个 frozen assets、686 个 benchmark items、1,295 个 variants；扫描 30,209 篇；`CHECKED_NO_HITS`、high-risk hits 0、删除 0、6 个低风险 `ANSWER_ONLY`/`SHORT_KNOWLEDGE_OVERLAP` 保留并记录。这是有界词法审计，不是语义证明；6 个低风险命中不代表可以跳过未来/未知 benchmark 的审计。
- 清洁审计后：30,209 篇、28,661,960 tokens、49,841,686 字符。

## 4. 类别供给与 10M 冻结预排

### 4.1 候选选包（12–15M 建议区间）

按 15M 目标比例选包，得到 **13,994,640 effective tokens**（14,009,038 raw），在建议的 12–15M 候选区间内；15M 冲刺目标差 1,005,360 effective（类别缺口合计 1,007,079，但 zh_education 超额 1,719）。缺口主要是 zhwiki 只能供 3,500,670（目标 4,500,000），缺口 999,330；英文差 2,187，中文通用差 5,562。

| 类别 | 候选有效 tokens | 15M 目标 | 缺口 | 10M 配额 | 10M 余量 |
|---|---:|---:|---:|---:|---:|
| `zh_general` | 6,744,438 | 6,750,000 | 5,562 | 4,500,382 | +2,244,056 |
| `zh_encyclopedic` | 3,500,670 | 4,500,000 | 999,330 | 3,000,255 | +500,415 |
| `zh_education` | 2,251,719 | 2,250,000 | 0 | 1,500,128 | +751,591 |
| `en_general` | 1,497,813 | 1,500,000 | 2,187 | 1,000,085 | +497,728 |
| `math_logic_code` | 0 | 0 | 0 | 0 | 0 |
| **合计** | **13,994,640** | **15,000,000** | **1,005,360**（净额） | **10,000,848** | **+3,993,792** |

候选选包的单来源占比：zhwiki 25.0%、wikinews 28.9%、wikivoyage 19.3%、en-wikisource 10.7%、wikibooks 10.4%、wikiversity 5.6%。没有任何单来源超过 30%。

### 4.2 10M 冻结预排（为 96/2/2 预留）

正式切分是 96% train / 2% val / 2% test；不能把 10M 候选直接当作 10M train。预排先按 `10,000,848 / 0.96` 预留，另加 1% 与每类 10 个窗口的 packing 余量，选 10,558,383 effective tokens；按 4% holdout 估算，train 约 **10,136,047** effective tokens，覆盖 10,000,848 并留 135,199 余量。正式切分按 family cluster 分组，实际 train 数量可能略有出入；冻结门禁会要求精确预算，预排不是冻结结果。

| 类别 | 预排有效 tokens | 预排占比 | 10M 配额 | 4% holdout 后供给 |
|---|---:|---:|---:|---:|
| `zh_general` | 4,747,155 | 44.961% | 4,500,382 | 4,557,268 |
| `zh_encyclopedic` | 3,163,652 | 29.963% | 3,000,255 | 3,037,105 |
| `zh_education` | 1,587,048 | 15.031% | 1,500,128 | 1,523,566 |
| `en_general` | 1,060,528 | 10.044% | 1,000,085 | 1,018,106 |
| `math_logic_code` | 0 | 0% | 0 | 0 |
| **合计** | **10,558,383** | **100%** | **10,000,848** | **10,136,047** |

- 比例偏差：-0.039pp / -0.037pp / +0.031pp / +0.044pp，全部在 ±1pp 内。
- 预排 `budget_for` 状态是 `READY_FOR_TRAINING`（指 token 数学余量），但 `blocking_conditions` 同时列出 `LICENSE_REVIEW_NOT_APPROVED` 和 `CONTAMINATION_NOT_CLEARED`。**这不是可训练状态**；正式冻结仍被审批阻断。
- 该预排不产生 split 文件、不产生 train.bin、不产生 `FROZEN`。

### 4.3 来源集中度与语言风格

10M 预排的来源占比（按 raw tokens）：

| 来源 | tokens | 占比 |
|---|---:|---:|
| `candidate-zh-wikipedia` | 3,166,744 | 29.96% |
| `candidate-zh-wikinews` | 2,847,010 | 26.94% |
| `candidate-zh-wikivoyage` | 1,904,786 | 18.02% |
| `candidate-en-wikisource` | 1,062,411 | 10.05% |
| `candidate-zh-wikibooks` | 1,032,523 | 9.77% |
| `candidate-zh-wikiversity` | 556,077 | 5.26% |

单来源最高 29.96%，低于 30% 的内部预警线；中文通用仅由 wikinews/wikivoyage 两源构成，二者合计 44.96%，仍低于 50%。这不是永久配额，正式审批可以收紧。

中文部分的 `chinese_style` 启发式统计（只用于报告，不作为过滤门槛）：

| 风格 | 文档数 | tokens | 占中文 tokens |
|---|---:|---:|---:|
| 现代汉语 | 12,808 | 8,144,652 | 85.67% |
| 混合 | 461 | 304,920 | 3.21% |
| 古汉语倾向 | 116 | 30,345 | 0.32% |
| 未判定 | 1,413 | 1,027,223 | 10.80% |

古汉语倾向占比很低，不存在“靠古汉语/文言文凑量”的问题；英文 1,062,411 tokens 单独计入 `en_general`，不计入中文风格比例。“未判定”主要是短文本/传统字混排，未从质量门禁中放行任何 REVIEW 文档。

## 5. 数据管线与门禁

### 5.1 流程

1. **INGESTED**：只接受已登记的本地 JSONL；逐文件 SHA256/字节数校验；流式读取，不把全量 raw 载入内存。V5 的 `max_record_bytes` 为 8 MiB，仅为容纳长篇作品的安全上限；它不是质量阈值，也不放行 REVIEW。
2. **NORMALIZED**：保留段落与 EOS；HTML 清理、Unicode/空白规范化；不做摘要或改写。
3. **QUALITY_REVIEWED**：V4 规则集 `corpus-quality-v4.1`；短/空、重复字符、重复段落、HTML 残留、模板/广告、标点密度、数字密度、PII、语言不匹配、异常字符；REVIEW 排除，REJECT 丢弃。
4. **长文切分**：候选 inventory 在归一化后切分；正式 V5 quality 阶段在评估前对 ingested 行调用 `_split_overlong_quality`，同样只在段落/句子边界切分并保留原 `document_family_id`。两条路径都验证拼接字节一致；切分不是质量豁免，切分后的片段仍逐条过全部质量规则。
5. **DEDUPLICATED**：exact content hash + MinHash-LSH 近重复；跨来源。
6. **SPLIT**：`family_cluster_hash`，96/2/2，`verify_near_cross_split=true`；从 V5 组独立生成，不复用历史 split。
7. **TOKENIZER_EVALUATED / TOKENIZED**：复用冻结 tokenizer；V5 不训练 tokenizer，不写新的 tokenizer 决策。
8. **AUDITED / FROZEN**：预算精确检查、污染审计、source approval、stage key、manifest、`FROZEN` 身份检查；任何一项失败都拒绝冻结。

### 5.2 V4 复用方式

V5 只在 `corpus_version == minillm-corpus-v5-language-10m-v1` 时进入新分支：

- `CorpusV4Pipeline` 的 `v5` 分支复用冻结 tokenizer 的 `tokenizer-selection.json`、evaluation/candidates 副本；`train_tokenizers`/`evaluate_tokenizers`/`record_tokenizer_decision` 明确拒绝。
- Tokenizer selection stage key 绑定 `split` 和 `BASE_ID`（1M content id），因此 1M/2M 的历史行为不变。
- `verify_frozen` 对 V5 重新计算同一 stage key，并在语言 release 末尾调用 `v5.verify_release`；1M/2M 不进入该分支。
- `v4_sources` 对 V5 调用 `v5.validate_scope` / `v5.validate_source_set`；V4 的 1M/2M 路径不改变。
- `v4_language_pilot.is_language`/`language_config` 增加 V5；1M/2M 的 quota 公式和发布校验保持原样。

### 5.3 可追溯性

每个候选片段保留：`document_id`、`parent_document_id`、`part_index`、`part_count`、`source_id`、`source_record_id`、`revision_id`、`source_url`、`input_path`、`input_sha256`、`sample_rank`、`language`、`category`、`tokens`、`characters`、`content_hash`、`chinese_style`。`sample_source` 的 `.state.json` 记录输入身份和已提交字节；`candidate-manifest.json` 记录 inventory 和所有报告的 SHA256。

## 6. 训练预算预览（不运行训练）

基准：PR #18 的 2M 实测 **245 updates / 1,531.928 s = 6.252767 s/update**。以下是按 8,184 supervised tokens/update 的完整优化器更新口径外推，不是训练承诺； `available_windows` 已扣除 4% val/test holdout。

| 预算 | 完整更新数 | supervised tokens | 需要窗口 | 可用窗口 | 窗口缺口 | 类别缺口 | 预计时间 | 磁盘（估算） | 可行性 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| 4M | 489 | 4,001,976 | 3,912 | 9,898 | 0 | 0 | 50.96 min | ≈6.4 GB | ✅ |
| 10M | 1,222 | 10,000,848 | 9,776 | 9,898 | 0 | 0 | 2 h 7 min | ≈6.4 GB | ✅ |
| 50M | 6,110 | 50,004,240 | 48,880 | 9,898 | 38,982 | zh_general 17,944,640；zh_encyclopedic 11,964,167；zh_education 5,977,070；en 3,982,318 | 10 h 37 min | ≈6.4 GB | ❌ 不足 |
| 100M | 12,219 | 100,000,296 | 97,752 | 9,898 | 87,854 | zh_general 40,442,866；zh_encyclopedic 26,962,984；zh_education 13,476,479；en 8,981,924 | 21 h 13 min | ≈6.4 GB | ❌ 不足 |

说明：

- 4M 与 10M 的类别余量均为正；10M 的 `zh_encyclopedic` 余量最小（预排 3,163,652 vs 配额 3,000,255），这是审批时最需要关注的类别。
- 50M/100M 不是“再采样一遍”能解决：唯一窗口只有 9,898，禁止重复窗口；需要新增已审批来源或扩大已批准来源的 scope。
- 磁盘是按 2M run 实测 6.4 GB（含 `model.pt`、`best.pt`、`state.pt`、`state.prev.pt` 与 milestones）估算；若大预算增加 milestone 数量，磁盘会更高。四次预览的总量约 25.6 GB，不含原始语料与中间缓存。
- 时间只外推计算/更新开销；不包含数据加载、评测、checkpoint 写入的波动。GPU 训练不在本 PR 范围内。

## 7. 冻结身份、历史兼容与验证集

- **冻结 tokenizer**：`artifacts/corpus/v4-language-pilot-first-training-reviewed/tokenizer.json`；vocab 24,576；fingerprint `845a24b9…b8a`；V5 receipt 的 `reused_from_content_id` 为 1M content id。
- **1M/2M 不变**：未重写 `artifacts/corpus/v4-language-pilot-first-training-reviewed/` 或 `artifacts/corpus/minillm-corpus-v4-language-pilot-2m-v1/`；两者仍可被 `verify_frozen` 独立验证。
- **历史 Val/Test 不变**：1M val/test SHA256 `25229b9c…364d` / `b58cb2c0…15c4`；2M manifest 记录相同文件。V5 交叉审计删除 77 篇重复，而不是修改历史文件。
- **新 Val/Test**：V5 config 的 96/2/2 `family_cluster_hash` 会在正式 freeze 时生成新的 V5 val/test；它们来自 V5 候选组，不复用历史 split，且会再次与历史 val/test 和 7 个 frozen benchmark 做污染审计。由于审批未完成，本 PR 不生成也不提交这些集合。
- **冻结门禁**：`verify_frozen` 要求 `freeze_status=READY`、`dataset_status=FORMAL`、content id/文件一致、stage keys 一致、source plan 一致、tokenizer selection 一致、split isolation PASS、`FROZEN` 标记、license/contamination/budget 通过。V5 额外要求 source approval 和 10,000,848 的精确预算。
- 当前 `freeze_status=BLOCKED_SOURCE_APPROVAL`；不存在 `FROZEN` 文件，`freeze.json` 和正式 `dataset-manifest.json` 也未生成。

## 8. 测试与复现

新增 `tests/test_v5_language_corpus.py`（14 项）覆盖：

- 来源审批/许可门禁：pending、旧 `corpus_versions`、错误预算、占位 reviewer、未批准的 scope 全部拒绝；6 类正类别必须覆盖，`math_logic_code` 必须为 0。
- 审批证据：request 文件身份、逐来源 release/文件 SHA256、reviewer/日期/逐字确认；篡改 request 或文件集合会失败。
- 大文件流式与断点恢复：Range/206、SHA1 校验、`.part` 恢复、错误 SHA1 不发布。
- 确定性采样/配额：seed 绑定、resume 后顺序稳定、输入变化拒绝、source cap 生效。
- 跨版本去重/污染：exact + near 历史重复删除、历史 doc id 稳定性、contamination 接口。
- 精确 token 预算：`training_preview` 的 1,222 × 8,184、9,776 窗口、4M/10M 可行、100M 不可行。
- 长文切分：候选路径与正式 `_split_overlong_quality` 行切分都只在边界切、拼接字节一致、每段 ≤20,000 字符、family 不变。
- 1M/2M 兼容：`is_language` 仍识别历史 profile；V4 回归测试 166 项通过。
- 冻结身份：V5 tokenizer reuse receipt 与 source approval 绑定的代码路径；正式 freeze 由审批后的 build 验证。

本机执行结果（详见 PR 描述）：

```text
.venv/bin/python -m pytest -q -o addopts='' tests/test_v5_language_corpus.py
# 14 passed

.venv/bin/python -m pytest -q -o addopts=''
# 1082 passed in 1026.26s (0:17:06)

SYNTHETIC=1 bash scripts/smoke_test.sh
# == smoke test passed ==
# 434,816 params; INT4 0.24 MB / 50 MB; reload + generate OK
```

## 9. 已知问题与扩充计划

1. **审批是唯一硬阻断（P0）**。6 个来源均未批准；没有批准就不能 build/freeze，也不能训练。下一步是由人类 reviewer 逐源确认 license/商用/再分发/模型训练/署名义务，填写 approval evidence，并重新生成 request 绑定。
2. **15M 候选冲刺差 1.0M**。主要缺 zhwiki（3.50M vs 4.50M）。若审批允许扩大 zhwiki 使用范围，将 sample_rate 从 0.005 提高到约 0.007，可补 ~1.0M；这会扩大既有来源使用范围，必须重新审批，不能在候选阶段自动做。
3. **10M 的 zh_encyclopedic 余量最小**（预排 3.16M vs 配额 3.00M）。正式 split 的组粒度可能使 train 侧略低于预排；冻结门禁会拒绝不足，届时优先补 zhwiki，而不是降低比例或重复窗口。
4. **REVIEW/REJECT 比例仍高**：wikinews 的 9,552 个 REJECT 包含大量短讯/占位/模板信号；wikibooks/wikivoyage/enwikisource 的 REVIEW 包含 PII、模板、标点/长度信号。后续可改进规则解释和来源特定预处理，但不得把 REVIEW 直接转为 KEEP。
5. **污染审计是词法的**。已清除 high-risk 命中，保留 6 个低风险 answer-only 命中；不能证明对未知/未来 benchmark 无语义污染。新增 benchmark 时必须重跑。
6. **50M/100M 不可行**。唯一窗口 9,898，远低于 48,880/97,752；需要新增已审批来源（例如 FineWeb2 需单独审批，不能用“无需审批”推定）或扩大已批准来源 scope。不要在训练时重复采样。
7. **无 GPU 训练**。本 PR 只交付候选库存、预算预览和门禁；4M/10M 的可行性是 token 数学与时间外推，不是训练结果。

## 10. 交付边界

- 新增：`src/minillm/v5_language_corpus.py`、`src/minillm/corpus_versions.py`、`configs/corpus_v5_language_10m.yaml`、`configs/corpus_v5_candidates.json`、`configs/corpus_v5_sources.review.json`、`configs/corpus_v5_historical-eval.json`、`scripts/prepare_v5_language_corpus.py`、`tests/test_v5_language_corpus.py`。
- 修改：`corpus_v4.py`、`v4_language_pilot.py`、`v4_sources.py`、`v4_tokenizer_decision.py` 的 V5 分支；1M/2M 路径保持原行为。
- 不提交：`data/raw/`、`artifacts/`、`reports/`、`runs/`、任何 `.bin`/`.pt`/大缓存。候选报告在 `reports/v5-language-corpus-10m-v1/`，已被 `.gitignore` 排除。
- 不合并：PR 保持打开，等待人类审批和后续 freeze/训练 PR。
