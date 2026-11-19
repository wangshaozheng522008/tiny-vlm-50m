# V4 Pilot 供给与 Token 缺口分析（v4-pilot-pr14-20261118）

- **结论**：`DATA_NOT_READY`
- **数据集**：`v4-pilot-pr14-20261118-v2` / `minillm-corpus-v4-pilot-1m-v1`
- **来源证据**：`VERIFIED`；已批准 4，已准入 4
- **Token 目标**：1,020,000 有效监督 Token；估算合计 5,323,964；实测合计 null
- **估算缺口**：102,000；**实测缺口**：null
- **Blockers**：`CATEGORY_DATA_INSUFFICIENT`, `ESTIMATED_TRAIN_SUPPLY_SHORTFALL`, `HUMAN_TOKENIZER_DECISION_REQUIRED`, `MATH_CODE_SOURCE_REQUIRED`, `SOURCE_FILES_MISSING`, `TOKENIZER_NOT_FINALIZED`

## 阶段身份与输入

- `plan_key`：`1274a5c6a99a811722c1f9721ebc59d0e22da41d962a8cad4c4464c0035f9afd`
- `implementation_sha256`：`2bb88dd4add9418231c7c9efb218844f36dbdb493af3f52d523ad26ec38bde75`
- `config_sha256`：`eec69679d483c6e09ee6105987e10b3361a558f6626b04eeccb0e0bf73488f58`

| Stage | Key | 输出文件数 |
|---|---|---:|
| ingested | `ddc37338ab92f43ca3afab22873e287e02c1bb7066d32bc99008d8d12dca0326` | 1 |
| quality | `f9a72d8bf168cd9cba5a52137f5b069a717d446340e3e32b16950c74ba9d0259` | 3 |
| dedup | `01062911960b75a178e8e9ca137420f79f2b3d8ba460e78a7258578484f54e27` | 2 |
| split | `6caedb110793bc97b45441c221ef2de1ceaa9f4048c1b47a3ec935ac7961510b` | 5 |
| tokenizers | `bbb254ea3d096f2a1056b7d8fb1eed943f6ecb14f6cc17786e48efeeb8f648c5` | 7 |
| tokenizer_eval | `2a4f5d9332840b09af17092272621f5f399347cef6a508dbae813b6d4d7f4deb` | 1 |

| 输入 | Source | 字节 | SHA256 |
|---|---|---:|---|
| `documents.jsonl` | `candidate-en-wikisource` | 22,718,156 | `24662a6796081356e6935866290e3a493a618c7f4c4ca2190660d690e9a11111` |
| `documents.jsonl` | `candidate-zh-wikibooks` | 9,988,174 | `4abe24742faf52cee2e0370730fe98f7200ae914f39df9114734433f6fe1b7eb` |
| `documents.jsonl` | `candidate-zh-wikipedia` | 8,122,462 | `9f79c4c102aa377b66ec766681652e14d6f2539c41ca834d7b27f3abe63bea49` |
| `documents.jsonl` | `candidate-zh-wikisource` | 7,915,382 | `eae0513d59ae1e4b90768e77c4f0e035cd3f11f31e8deaea71721f235fe677e1` |

## 来源准入与处理统计

| 来源 | 类别 | 许可 | 输入文档 | Ingest | KEEP | REJECT | REVIEW | Dedup 删除（exact/near） | Train/Val/Test |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `candidate-en-wikisource` | en_general | APPROVED | 1,836 | 1,836 | 1,400 | 82 | 354 | 3 (2/1) | 1,358/19/20 |
| `candidate-math-code` | math_logic_code | REVIEW_REQUIRED | 0 | 0 | null | null | null | null (null/null) | 0/0/0 |
| `candidate-python-docs` | math_logic_code | REVIEW_REQUIRED | 0 | 0 | null | null | null | null (null/null) | 0/0/0 |
| `candidate-zh-education` | zh_education | REVIEW_REQUIRED | 0 | 0 | null | null | null | null (null/null) | 0/0/0 |
| `candidate-zh-wikibooks` | zh_education | APPROVED | 1,891 | 1,891 | 1,481 | 258 | 152 | 9 (6/3) | 1,399/36/37 |
| `candidate-zh-wikipedia` | zh_encyclopedic | APPROVED | 3,916 | 3,916 | 3,249 | 636 | 31 | 15 (6/9) | 3,106/66/62 |
| `candidate-zh-wikisource` | zh_general | APPROVED | 905 | 905 | 792 | 12 | 101 | 14 (0/14) | 742/17/19 |

合计：输入 8,548，Ingest 8,548，KEEP 6,922，REJECT 988，REVIEW 638，Dedup 删除 41，Split 6,605/138/138。

## 分类别供给与缺口

| 类别 | 目标 | 估算 Train Token | 估算缺口 | 估算达成率 | 实测 Token | 实测缺口 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| zh_general | 408,000 | 1,213,233 | 0 | 297.4% | null | null | NOT_MEASURED_TOKENIZER_NOT_FINALIZED |
| zh_encyclopedic | 255,000 | 1,446,983 | 0 | 567.4% | null | null | NOT_MEASURED_TOKENIZER_NOT_FINALIZED |
| zh_education | 153,000 | 1,071,842 | 0 | 700.6% | null | null | NOT_MEASURED_TOKENIZER_NOT_FINALIZED |
| en_general | 102,000 | 1,591,906 | 0 | 1560.7% | null | null | NOT_MEASURED_TOKENIZER_NOT_FINALIZED |
| math_logic_code | 102,000 | 0 | 102,000 | 0.0% | null | null | NOT_MEASURED_TOKENIZER_NOT_FINALIZED |

### Split 与 Train 预算

| Split | 文档 | 字符 | 估算 Token | 计入 Train 预算 |
|---|---:|---:|---:|---|
| train | 6,605 | 13,191,864 | 5,323,964 | true |
| val | 138 | 194,770 | 93,640 | false |
| test | 138 | 213,641 | 94,471 | false |

- Val/Test 不计入 Train 预算（`val_test_in_train_budget=false`）。
- 估算 Token 不是实测值；实测合计为 `null`，依据 `NOT_MEASURED: tokenizer not formally selected`。

## Tokenizer 评估

- 评估状态：`COMPARED`；选择状态：`HUMAN_TOKENIZER_DECISION_REQUIRED`；证据状态：`UNVERIFIED`；Tokenization：`BLOCKED`。
- 建议：`HUMAN_DECISION_REQUIRED` / 低成本候选 `None`；仅为建议，仍需人工签署。

| 候选 | 词表 | 中文 chars/token | 英文 tokens/word | 代码 chars/token | 参数量 | INT4 < 50M |
|---|---:|---:|---:|---:|---:|---|
| `v4-24576` | 24576 | 1.5197 | 1.3333 | 2.2712 | 86,528,640 | true |
| `v4-32768` | 32768 | 1.5755 | 1.2222 | 2.3103 | 91,771,520 | true |

- 固定评估文本类别：`cpp, en, math, python, special, zh_education, zh_knowledge, zh_prose`。

### 流水线阶段状态

文件存在不等于阶段成功；以下状态来自 Readiness 的正式验证结果。

| 阶段 | 状态 |
|---|---|
| Tokenizer 选择 | `BLOCKED` |
| Tokenize | `BLOCKED` |
| Contamination | `NOT_RUN` |
| Budget | `BLOCKED` |
| Audit | `NOT_RUN` |
| Freeze | `NOT_RUN` |

## 数学／代码类别

- 状态：`MATH_CODE_SOURCE_REQUIRED`；目标 102,000，估算缺口 102,000，实测 null，实测缺口 null。
- 候选 `candidate-math-code`：许可 `REVIEW_REQUIRED`，文件 `SOURCE_FILES_MISSING`，已批准输入 0。
- 候选 `candidate-python-docs`：许可 `REVIEW_REQUIRED`，文件 `SOURCE_FILES_MISSING`，已批准输入 0。
- 未伪造数学题、代码文档、许可结论或人工审核记录；不从网络自动下载未确认授权的数据。

## 一致性检查

- 状态：`PASS`；失败项：无。
- `PASS` source/category approved input documents: 8548 == 8548
- `PASS` category targets equal Pilot target: 1020000 == 1020000
- `PASS` ingest total vs source rows: 8548 == 8548
- `PASS` ingest total vs category rows: 8548 == 8548
- `PASS` quality documents vs source rows: 8548 == 8548
- `PASS` quality keep vs source rows: 6922 == 6922
- `PASS` quality keep vs category rows: 6922 == 6922
- `PASS` quality reject vs source rows: 988 == 988
- `PASS` quality reject vs category rows: 988 == 988
- `PASS` quality review vs source rows: 638 == 638
- `PASS` quality review vs category rows: 638 == 638
- `PASS` dedup deleted vs source rows: 41 == 41
- `PASS` dedup deleted vs category rows: 41 == 41
- `PASS` dedup exact total: 14 == 14
- `PASS` dedup near total: 27 == 27
- `PASS` dedup kept + deleted == quality keep: 6922 == 6922
- `PASS` split train documents vs source rows: 6605 == 6605
- `PASS` split train documents vs category rows: 6605 == 6605
- `PASS` split train total vs assignment: 6605 == 6605
- `PASS` split train assignment characters: 13191864 == 13191864
- `PASS` split val documents vs source rows: 138 == 138
- `PASS` split val documents vs category rows: 138 == 138
- `PASS` split val total vs assignment: 138 == 138
- `PASS` split val assignment characters: 194770 == 194770
- `PASS` split test documents vs source rows: 138 == 138
- `PASS` split test documents vs category rows: 138 == 138
- `PASS` split test total vs assignment: 138 == 138
- `PASS` split test assignment characters: 213641 == 213641
- `PASS` Train characters vs source rows: 13191864 == 13191864
- `PASS` Train characters vs category rows: 13191864 == 13191864
- `PASS` Train estimated tokens vs readiness: 5323964 == 5323964
- `PASS` Train estimated tokens vs Train split: 5323964 == 5323964
- `PASS` Val/Test excluded from Train budget: 5323964 == 5323964
- `PASS` Train documents are a subset of dedup KEEP: true == true

## 下一步

- 获取已批准的数学/代码来源，固定发布版本，完成许可审核、转换和样本证据；剩余目标为 102,000 个有效 Token。
- 由人工审核者签署 Tokenizer 选择；不允许自动选择。
- 人工选定 Tokenizer 后，执行 tokenize、contamination、budget、audit，再评估 Freeze。
- 为供给不足类别增加已批准数据：math_logic_code。
- 在所有类别达标且最终审计通过前，保持 DATA_NOT_READY。

## 限制

- Counts describe the locally approved sample inputs, not the complete upstream release.
- Estimated tokens use the configured chars-per-token ratio and are never reported as measured.
- Val/Test splits are excluded from all Train budget calculations.
- No tokenizer was automatically selected and no dataset was frozen by this analysis.
- A source review hash proves identity; it does not replace a human licence/training decision.
