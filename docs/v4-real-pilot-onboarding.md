# 真实来源接入 V4 1M Pilot

Pilot 版本为 `minillm-corpus-v4-pilot-1m-v1`，目标为 1,020,000 个实际有效监督 Token。五类目标仍为 40%、25%、15%、10%、10%。真实输入、预览样本和审核工作副本保存在本地。

## 从本地 JSONL 生成抽样和审核草稿

```bash
python scripts/prepare_v4_pilot_sample.py \
  /readonly/wiki/zhwiki-20261101-pages.jsonl \
  --source-id candidate-zh-wikipedia --basis-points 25 \
  --out data/staged/pr12-zhwiki-sample-v1
```

每个来源使用独立的输出目录。输入可以是 PR #11 转换产物或现有 `prepare_v4_sources.py` 暂存的
`documents.jsonl`，也可以显式指定外部只读目录中的 JSONL 文件。保持上游发布版与项目许可的对应关系。
给定 seed、source ID 和家族身份，SHA256 排名决定该家族是否入样。25 个基点对应 0.25% 的哈希空间，
实际行数随家族体积变化。哈希选择扫描整个传入文件，文件是上游切片时还要保留那层抽样来源。
抽样既要给质量拒绝、去重和 Train/Val/Test 留出余量，也要给 packing 留出余量。输出统计中的字符和估算
供调整抽样量使用，有效监督 Token 从选定 Tokenizer 编码后的 Train 索引计算。

抽样保留页面字段、原文和父文档身份。重复的文档 ID 与相同内容只写一次，同 ID 的冲突记录使抽样失败。
普通同文档哈希去重由后续 V4 管理。输入 SHA256、字节数、算法、seed、上限和输出身份进入
`pilot-sample-manifest.json`。`review-entry.json` 是同一审核清单格式的草稿，含样本文件的实际绝对路径、
SHA256 和字节数。已有输出目录应换新路径，变更输入后重新抽样和核验。

## 本地人工审核

复制 `configs/corpus_v4_sources.review.json` 到本地工作目录，用草稿替换相同 source ID 的条目，保留其余来源。
直接使用完整文件时可在审核条目填写准确的 JSONL 或 TXT 路径和适用的文件身份，避免目录中混入校验清单、
错误队列或其他转换结果。一个版本的清单每个 source ID 只能出现一次。

复核发布者、具体版本、上游 URI、许可和逐项证据，填写商业使用、再分发、训练用途、义务、署名、ISO 日期
和实际审核范围。维基文库需要按作品范围审核，包括许可、法域、引用、翻译及附加条款。若只获准使用已核验
作品，请先准备仅包含这些作品的 JSONL 作为抽样输入。人工记录对抽样后的实际文件负责，按
`minillm-corpus-v4-pilot-1m-v1` 填写适用版本。其他版本需要各自的人工许可范围。

审核完整后由人将对应条目标记为 `APPROVED`。PR #10 的审核校验继续检查完整批准条目、发布版本和实际 Ingest
文件。包含抽样溯源的条目还会重验抽样 Manifest、选样结果和原始 JSONL。保留这些证据以便重跑、
冻结和 Pilot 再准入。未经审核的真实样本可以用于文件盘点和审阅，Split 仍依赖正式来源门禁。

## 运行和验证

```bash
python scripts/plan_corpus_v4.py --config configs/corpus_v4_pilot_1m.yaml \
  --source-registry data/staged/pilot-review.json --status-only
python scripts/plan_corpus_v4.py --config configs/corpus_v4_pilot_1m.yaml \
  --source-registry data/staged/pilot-review.json --stage split
python scripts/report_v4_readiness.py --pilot-only \
  --source-registry data/staged/pilot-review.json --out reports/pilot-readiness-run-1
```

对于只需做文件盘点的本地来源，可附 `--base` 指定项目数据根目录，或用 `--sample` 逐项提供已验证的样本
Manifest。正式报告区分可读来源、获批来源和适用版本已验证的来源。当五类 Train 数据和 Split 隔离
有证据时报告是否具备候选评估条件。预算判断以正式人工选定的 Tokenizer 和 Train packing 为依据。
实际不足的类别保持缺口，选定的中文百科、教科书和维基文库分别保留各自类别。

来源、配置和阶段文件仍由 V4 Plan、阶段 SHA256 及冻结机制绑定。阶段失败后用相同输入与配置重试，
损坏的阶段或变更的输入用新输出目录重建。候选比较继续用 Train-only 的 24,576 和 32,768，
评估资产绑定哈希。24,576 的正式选择由人工单独签署，完成预算和污染证据后仍须通过 V4 最终审计。
已有冻结数据需经 `verify_frozen` 复验方可报告 `FROZEN_VERIFIED`。

## PR #12 本地验证快照

已收到所选来源可用于项目训练的人工确认。逐来源的署名、ISO 日期、使用及履约结论仍待录入，
所以本次快照保留现有 7 条候选审核状态。四份 Wikimedia JSONL 已完整读取并与原始 dump 及转换
Manifest 对照。可读文档数按 V4 的 1 MiB 单记录上限统计，原始页计数和超限条数见报告。

| 来源 | V4 类别 | 可读文档 | 预览文档 |
|---|---|---:|---:|
| 中文维基百科 | zh_encyclopedic | 1,562,728 | 3,916 |
| 中文维基文库（上游切片） | zh_general | 94,715 | 905 |
| 中文维基教科书 | zh_education | 7,666 | 1,891 |
| 英文维基文库 | en_general | 169,704 | 1,836 |
| 数学／代码、Python 文档 | math_logic_code | 0 | 0 |
| 其他中文开放教育候选 | zh_education | 0 | 0 |

完整发布版、文件和样本摘要以及各来源待补材料保存在 [PR #12 就绪快照](../reports/pr12-pilot-readiness.json)。
实际许可范围需逐来源确认。[Wikimedia 使用条款](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use) ·
[中文维基文库版权说明](https://zh.wikisource.org/wiki/Wikisource:版权信息/全文) ·
[英文维基文库版权政策](https://en.wikisource.org/wiki/Wikisource:Copyright_policy) ·
[Python 文档许可](https://docs.python.org/3/license.html)

真实数据的 V4 Quality、Dedup、Split、候选评估及污染检查尚未运行。训练有效 Token 的实测值和差额
待正式 Tokenizer 定版和编码，当前完整的 1,020,000 Token 目标尚无测量证据。PR #12 的原创夹具经
常规 V4 入口验证已审核来源、拒绝与审核队列、去重、家族切分和阶段身份绑定。
