# V5 千万 Token 中文语料扩充与冻结

本次复审以 PR #19 的 `7b5d787f0a3001890d06bcd4a30018a9f65c7a3c` 为基准，目标版本为 `minillm-corpus-v5-language-10m-v1`。训练配比为中文通用 45%、百科 30%、教育 15%、英文 10%，各类别容差 ±1 个百分点，数学/代码为 0%。沿用 86,528,640 参数模型、24,576 词表和原有 INT4 算法。GPU 训练未启动。

## 正式构建结果

正式 Build、完整 Audit、Finalize 和 `verify_frozen()` 已通过，`validate_v4_pilot_policy()` 返回 `POLICY_MATCH`。发布状态为 `READY` / `FROZEN`。正式内容 ID 为 `2a688379c66d701d41cede75961ca71af0dbb8254c2a5c065220b19ee4ea54ce`。实现指纹为 `19348de69ff70357e6f0bf001e249b6bf29aeb1bfcce2ef575d777397fdb3cd1`。
本机工作目录为 `artifacts/corpus/minillm-corpus-v5-language-10m-v1-reviewed-final`。

确定性抽样取得 44,525 篇原始文档。质量阶段切分 401 篇长文，排除 68 个精确重复并保留 30,565 个候选部分。近似去重排除 279 个候选部分，历史交叉去重随后排除 77 个（精确 76，近似 1）。逐个核对文档 ID 后确认原候选排除的 77 篇均在重放的排除名单中。来源配额选择得到 19,997 个候选部分、14,009,038 个原始 Token。

历史交叉扫描索引包含 276 篇独立历史文档和 354 个语言索引条目。正式 Train、Val、Test 的历史精确/近似命中数分别为 0、0、0。正式污染审计状态为 `CHECKED_NO_HITS`，扫描 19,206 篇 Train 候选，高风险命中 0，允许的常识命中 2，扫描截断数为 0。新 Val/Test 来自 V5 家族与近似簇隔离切分。完整 Audit 的 15 项检查全部通过，14,916 篇 BIN 文档通过逐篇全量重编码。

| 切分 | BIN 中的文档 | 原始 Token | BIN 字节数 |
|---|---:|---:|---:|
| Train | 14,125 | 10,024,394 | 20,048,788 |
| Val | 395 | 221,595 | 443,190 |
| Test | 396 | 268,571 | 537,142 |

Train 压包有 **9,789 个独立完整窗口**，对应 **10,014,147 个有效监督目标**。

## 实际 Sampler 预算

1024 个输入 Token 产生 1023 个监督目标。优化器每次更新消费 8 个不同窗口，共 8,184 个监督目标。以下读实际 Train BIN 和 Token Index，使用种子 42 重放第一轮打乱。耗时以 PR #18 的 T1200 实测 245 步、1,531.928 秒线性估算。最低存储按 uint16 Train BIN 计算。

| 预算 | 更新步数 | 目标监督 Token | 需要窗口 | 窗口缺口 | 预计小时 | 最低 Train BIN 字节 | 可行性 |
|---|---:|---:|---:|---:|---:|---:|---|
| 4M | 489 | 4,001,976 | 3,912 | 0 | 0.849 | 8,011,778 | `FEASIBLE` |
| 10M | 1,222 | 10,000,848 | 9,776 | 0 | 2.122 | 20,021,250 | `FEASIBLE` |
| 50M | 6,110 | 50,004,240 | 48,880 | 39,091 | 10.612 | 100,106,242 | `BLOCKED` |
| 100M | 12,219 | 100,000,296 | 97,752 | 87,963 | 21.223 | 200,196,098 | `BLOCKED` |

| 类别 | 完整 Train 库存 | 4M 实际消费 | 10M 实际消费 |
|---|---:|---:|---:|
| 中文通用 | 4,505,158 (44.988%) | 1,788,204 (44.683%) | 4,495,951 (44.956%) |
| 中文百科 | 3,000,872 (29.966%) | 1,190,028 (29.736%) | 2,998,826 (29.986%) |
| 中文教育 | 1,500,741 (14.986%) | 608,406 (15.203%) | 1,500,741 (15.006%) |
| 英文通用 | 1,007,376 (10.060%) | 415,338 (10.378%) | 1,005,330 (10.052%) |
| 数学/代码 | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |

| 来源 | 类别内上限 | 完整 Train 库存 | 4M 实际消费 | 10M 实际消费 |
|---|---:|---:|---:|---:|
| `candidate-en-wikisource` | 100% | 1,007,376 (100.000%) | 415,338 (100.000%) | 1,005,330 (100.000%) |
| `candidate-zh-wikibooks` | 65% | 818,576 (54.545%) | 312,484 (51.361%) | 818,576 (54.545%) |
| `candidate-zh-wikinews` | 60% | 2,253,158 (50.013%) | 885,893 (49.541%) | 2,247,098 (49.980%) |
| `candidate-zh-wikipedia` | 100% | 3,000,872 (100.000%) | 1,190,028 (100.000%) | 2,998,826 (100.000%) |
| `candidate-zh-wikiversity` | 55% | 682,165 (45.455%) | 295,922 (48.639%) | 682,165 (45.455%) |
| `candidate-zh-wikivoyage` | 60% | 2,252,000 (49.987%) | 902,311 (50.459%) | 2,248,853 (50.020%) |

来源表的百分比以各自所属类别为分母。各类别缺口、窗口顺序 SHA256 和完整计数可在 `v5-sampler-budget.json` 核对。

## 文件校验

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `budget-report.json` | 3,026 | `30168c4653ab117428fb05e2248c30ef94bffdef186e140fb6365f9d32c00b71` |
| `contamination-report.json` | 5,110 | `5c439f43f441f36051ebe843edd27d272bc0dcc8abf131f83260790c6085d34b` |
| `dedup-report.json` | 1,635 | `57eae8d6e873c78977d1e6458500eb6c1aa838d23c0ba3c9d125cbad8b7544db` |
| `quality-report.json` | 4,490 | `2e6664df89c5e6fcff59a2f1ad1a322213f5cf130d9ade759848a7bbf44422b4` |
| `split-assignment.jsonl` | 4,094,820 | `c3c7e4bd8de423f5404dd3b1ab6c4a16f49eba1d53dca14f434e03b46a3e9131` |
| `split-manifest.json` | 1,963 | `0eef5248b8e1d6307465bf6249d32d884a1dd2a8023d1be1e35f8e0617227825` |
| `test.bin` | 537,142 | `c6e4288a3f8d4fd6642686c1320e6a7ae4dcc3c1734a64fd7f3c7ea46f9c9c78` |
| `test.text.jsonl` | 2,154,228 | `6360c36427cea328d3927085a6ed6767bee776f61db0172b2d9299715ab570e4` |
| `test.token-index.jsonl` | 66,069 | `4860eef7be06ef7707b90b370154328bd55f30393772d4e079ab1e5194d2d27e` |
| `tokenization-report.json` | 6,917 | `149b7cb712df62f7cad77a9a778360358e19e618b150699e211cfca131d765d0` |
| `tokenizer-candidates.json` | 5,469 | `394da1ed9441920300713b78679a836f7b06b44b9b08c97f254da5931698460f` |
| `tokenizer-evaluation.json` | 33,240 | `699fdf3fc6f4be162ff3cd85b4a625b6555ff6432fcba111b30798acf7e522e0` |
| `tokenizer-selection.json` | 1,510 | `1c57a8bc194f33e6b6ff59dc535f2664103d5189ca69524941054c5619c73121` |
| `tokenizer.json` | 932,773 | `fb8ac32998348837ef7108bba5df9a8648d5295450f0ddb8fc2ff250ebc42b8d` |
| `train-selection.json` | 2,239,018 | `83fcdb55f4fc4e754251522745cd21c6a732fc71e22cfebbbf84d1568ed1bb1f` |
| `train.bin` | 20,048,788 | `b08ec1b79fb2f724dacb3920fc8eab7eb9fcd83c9881c3356f8417781b6b852e` |
| `train.selected.jsonl` | 77,870,114 | `fc3338e0af1ef667727e8857fdb103f308b95377d3d0a3acaf49d45311f75014` |
| `train.text.jsonl` | 105,720,168 | `eb3ff6314938905578afa1be435dc29131a1c367c59495668fd5240dd08333c4` |
| `train.token-index.jsonl` | 2,364,454 | `33ab36ca901a297361b822faa7c06f2dbdbb0c3773e5831730036359c0ff828a` |
| `v5-candidate-lineage.jsonl` | 22,952,524 | `76e3d811148e75fde9139dfb4bdf0e1cea36f34029d3ad8b6fed94171441225e` |
| `v5-candidate-selection.json` | 35,363 | `352de6f8b9fba3d583f93401195c55c56747d170c3f7c7c889c008a102ece11e` |
| `v5-formal-screening.json` | 10,197 | `7470ef63d36740ad1f1978addf24c4d3ad9edf111d26c3fa76666418bea04cb3` |
| `v5-sampler-budget.json` | 12,058 | `6701308ab339885b47073bc94d81369be3ba34db7d4bfa2d206d772e096f8044` |
| `val.bin` | 443,190 | `93679de521ea1ee6405b402027b54a73ebcdf5ceb055e71c7be3c43aa7937f17` |
| `val.text.jsonl` | 1,976,073 | `8b7fbb1ac0ae56d1595f78618bffdeeb97bcc3b74b997ba03d618b647f42417a` |
| `val.token-index.jsonl` | 65,818 | `05270ffbf62d5f0d393585b8bf06fd1de2059cd90a6a871dd24d0f0f1b36fcac` |
| `audit-report.json` | 14,050 | `cd5cc48ae0effc439b97fd346d3d4da716075a4ac1eadeafb2bbe4fca7b6f8e9` |
| `dataset-manifest.json` | 58,587 | `c0b8af697ba34e6e0161eaa51b5c5ccecd422e40c08cb318f96e239a8c689bf0` |
| `freeze.json` | 323 | `1437fcb948aa35219e3caf74bf7622ef93cf481cb3fcacd394e827cf5d5a994e` |
| `FROZEN` | 82 | `33aebf373c99aa9574dbc9ab8a0237557a1b9fdc55c4b254590b8a01baa392af` |

## 复核命令

在仓库根目录使用本机审批登记所指的只读来源快照。重建时将 `FRESH_OUTPUT` 设为新的空目录，复核命令读取本次冻结目录。

```bash
.venv/bin/python scripts/prepare_v5_language_corpus.py build \
  --dataset "$FRESH_OUTPUT" \
  --approval configs/corpus_v5_source-approval.json \
  --request configs/corpus_v5_source-approval-request.json
.venv/bin/python scripts/plan_corpus_v4.py --verify artifacts/corpus/minillm-corpus-v5-language-10m-v1-reviewed-final
```

原始语料、二进制和本机报告均保存在 Git 忽略目录。

## 复审修复

| 问题 | 修复 |
|---|---|
| P1 正式冻结调用链 | V5 从本模块调用 `validate_source_set()`。正式 Build、Audit、Finalize、`verify_frozen()` 和准入通过同一端到端回归覆盖。 |
| P1 候选与正式流程 | 正式 Ingest 从批准快照重放确定性抽样、规范化、长文边界切分、质量过滤、精确及近似去重、历史交叉去重、污染筛查和来源配额选择。候选文档谱系记录来源、原始位置、父文档/切片、哈希和转换。Audit 将 Train、Val、Test 与谱系逐篇对账，并重新扫描历史重复。 |
| P1 审批与发布 | 保留六条原始审批记录。构建计划绑定审批人、日期、确认内容、请求范围、输入快照、文件 SHA256、候选配置及历史评测身份。通用入口执行同一核验。Finalize 在写入正式 READY 与 FROZEN 前执行完整准入，`verify_frozen()` 复验审批和来源。 |
| P1 历史兼容 | 准入只接受当前实现指纹，或历史版本、内容 ID 与原实现指纹的精确组合。Tokenizer 复用时，训练来源按 PR #16 的冻结 Train 证据验证。 |
| P2 预算 | 候选预算标记为 `ESTIMATED`。正式预算从实际 Train BIN 与 Token Index 重放种子 42 的 `EpochBatchSampler`，按类别和来源统计实际监督目标。 |
| P2 实现与缓存 | `implementation_sha256()` 纳入 `v5_language_corpus.py`、`corpus_versions.py`、训练计数及准入实现。候选配置、来源配额、历史配置与审批身份参与构建计划和阶段键。 |

审批人 `wangshaozheng` 于 2026-11-26 确认六个来源仅用于内部研究训练。商业使用、对外再分发和公开权重均未获授权。来源快照与许可义务见[来源登记](../configs/corpus_v5_sources.review.json)。请求和确认记录见[审批请求](../configs/corpus_v5_source-approval-request.json)、[审批证据](../configs/corpus_v5_source-approval.json)。

## 历史兼容与评测

| 版本 | 内容 ID | 原始实现 SHA256 | 冻结 Manifest SHA256 |
|---|---|---|---|
| PR #16 1M | `7222651ef09e4e09c2d613474f50712e8233d1d4ff9017bda897e597ebdb6da5` | `15b8beb04ed88b91f96d70b59df72022dc995fea9bb50b0a8eda195e5359255d` | `6160624ea9962eff0d07a4aa976f8b82e9a7cfc36ba84773c9dbd4d55a7791c7` |
| PR #18 2M | `dcf709c91a5499f9a3b430c935ce5ede08d8abbad1456bf89cbeea7bcb92f8d1` | `afbbcc5899127874ccab4291af6c00aa8f5e8d7d9cf6180d8858ac121a753965` | `cf98c062a2c03fddf1b5641684a714fe0725d6eab0e469b323d50836111cba4e` |

使用真实资产运行 `verify_frozen()` 和 `validate_v4_pilot_policy()`，1M 与 2M 均通过冻结复验并返回 `POLICY_MATCH`。两份冻结 Manifest 分别校验。1M 和 2M 的 Val/Test 文本、BIN、Token Index 以及 Tokenizer 各自字节相同，交叉去重索引对相同的历史文档只建一次。共用 Tokenizer 指纹为 `845a24b9f21458b1d2161c60dfc8a77d5ed40235ece7ef46891677e423727b8a`。

| 两份历史语料共用的文件 | SHA256 |
|---|---|
| `tokenizer.json` | `fb8ac32998348837ef7108bba5df9a8648d5295450f0ddb8fc2ff250ebc42b8d` |
| `val.text.jsonl` | `25229b9cd18b7cb1c7c064a8334388afaa6a782db4592c532edf6ca5dbe9364d` |
| `test.text.jsonl` | `b58cb2c04ca9024a347ddc64001afe05b4db16d56bc3511f88062179253615c4` |
| `val.bin` | `eaa0a3ca78a4371cde39f29cf0397a35183ba81fc8dc00cd8ace58e734b7ac10` |
| `test.bin` | `d4133d8142a8d8d5689afd3fb83c3c31cf4594bd7590aa150ba4a41dbc14048f` |
| `val.token-index.jsonl` | `4ae574e47c3c039ab8efd2876ea9351c3b20ed5f95e1e905ad164c4eaebf607b` |
| `test.token-index.jsonl` | `fb66ff418ee7b66f471b724ffe06d866c7b196fe54a630ca8f9c0518cefc0a43` |

## 批准来源

| 来源 | 类别 | 发布快照 / 上游 | 许可 | 审批状态 | 商用 / 再分发 / 训练 |
|---|---|---|---|---|---|
| `candidate-zh-wikipedia` | `zh_encyclopedic` | `zhwiki-20261101-pages-articles` / https://dumps.wikimedia.org/zhwiki/20261101/ | CC-BY-SA-4.0 | `APPROVED`（内部研究） | `NOT_APPROVED` / `NOT_APPROVED` / `APPROVED_INTERNAL_RESEARCH_ONLY` |
| `candidate-zh-wikinews` | `zh_general` | `zhwikinews-20261101-pages-articles` / https://dumps.wikimedia.org/zhwikinews/20261101/ | CC-BY-2.5 | `APPROVED`（内部研究） | `NOT_APPROVED` / `NOT_APPROVED` / `APPROVED_INTERNAL_RESEARCH_ONLY` |
| `candidate-zh-wikivoyage` | `zh_general` | `zhwikivoyage-20261101-pages-articles` / https://dumps.wikimedia.org/zhwikivoyage/20261101/ | CC-BY-SA-4.0 | `APPROVED`（内部研究） | `NOT_APPROVED` / `NOT_APPROVED` / `APPROVED_INTERNAL_RESEARCH_ONLY` |
| `candidate-zh-wikibooks` | `zh_education` | `zhwikibooks-20261101-pages-articles` / https://dumps.wikimedia.org/zhwikibooks/20261101/ | CC-BY-SA-4.0 | `APPROVED`（内部研究） | `NOT_APPROVED` / `NOT_APPROVED` / `APPROVED_INTERNAL_RESEARCH_ONLY` |
| `candidate-zh-wikiversity` | `zh_education` | `zhwikiversity-20261101-pages-articles` / https://dumps.wikimedia.org/zhwikiversity/20261101/ | CC-BY-SA-4.0 | `APPROVED`（内部研究） | `NOT_APPROVED` / `NOT_APPROVED` / `APPROVED_INTERNAL_RESEARCH_ONLY` |
| `candidate-en-wikisource` | `en_general` | `enwikisource-20261101-pages-articles` / https://dumps.wikimedia.org/enwikisource/20261101/ | Public-domain works plus CC-BY-SA-4.0 contributions | `APPROVED`（内部研究） | `NOT_APPROVED` / `NOT_APPROVED` / `APPROVED_INTERNAL_RESEARCH_ONLY` |

## 测试

| 测试 | 结果 |
|---|---|
| 全量 `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q -o addopts='' -rs` | 1,078 通过，1 跳过，10 个 Web 用例首次运行因沙箱禁止监听本机回环端口而在初始化时报错 |
| 获准监听后运行 `.venv/bin/python -m pytest -q -o addopts='' -rs tests/test_web.py` | 11 通过，覆盖全量运行中报错的 10 项 |
| `.venv/bin/python -m pytest -q -o addopts='' -rs tests/test_v5_language_corpus.py` | 21 通过，含真实 1M/2M 复验和准入 |
| `SYNTHETIC=1 bash scripts/smoke_test.sh` | CPU Smoke Test 通过，覆盖预训练、SFT、INT4 导出及加载生成 |

全量测试中唯一跳过的是 `tests/test_evaluation.py` 的可选 CUDA 等价性探针，本次在 CPU 上验证。合并两次测试结果后，1,088 项通过，1 项跳过。
