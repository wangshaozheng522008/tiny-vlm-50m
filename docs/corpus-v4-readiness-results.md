# PR10 接入、Tokenizer 与冻结就绪性记录

日期：2026-11-17。本次只完成工程实现、CPU 验证和只读就绪审计，未批准真实来源，未训练正式候选，未执行 GPU Pilot。

| 验收项 | 实际状态 |
|---|---|
| 工程 | CODE_READY |
| 来源数据 | DATA_NOT_READY |
| Tokenizer | TOKENIZER_NOT_FINALIZED |
| 独立 1M Pilot | PILOT_DATA_NOT_READY |
| 正式 100M | FORMAL_CORPUS_NOT_READY |

## 来源与预算

单一来源审核清单覆盖五类，共 7 个候选：APPROVED 0、REVIEW_REQUIRED 7、REJECTED 0。
所有真实审核人/日期为空，具体版本尚未固定；没有自动批准或下载语料。
候选许可和限制、接入步骤见 [来源审核](corpus-v4-source-review.md)。

本地历史 MiniMind 文件只读实测：1,241,043,656 字节，1,270,238 条记录，486,491,829 字符；
SHA256 `6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c`。
其混合来源未取得本次所需审核，全部排除在合格 V4 预算之外；未将字符、历史估算或文件大小换算成独立监督 Token。
7 个候选的登记本地路径均没有可准入输入。正式清洗/去重/Tokenize/Audit/Freeze 未运行，相关统计为 NOT_RUN/null。

| 类别 | 比例 | 合格独立监督 Token | Pilot 目标/缺口 | 100M 目标/缺口 |
|---|---:|---:|---:|---:|
| zh_general | 40% | 0 | 408,000 | 40,000,000 |
| zh_encyclopedic | 25% | 0 | 255,000 | 25,000,000 |
| zh_education | 15% | 0 | 153,000 | 15,000,000 |
| en_general | 10% | 0 | 102,000 | 10,000,000 |
| math_logic_code | 10% | 0 | 102,000 | 10,000,000 |
| 合计 | 100% | 0 | 1,020,000 | 100,000,000 |

两套独立版本、输出目录与冻结记录继续隔离；没有重复抽样、拼接补量或降低目标。
只读历史盘点约 57.92 秒、21,931 条/秒，进程峰值 RSS 505,068 KiB；这是包含排除材料的盘点，并非正式 V4 规模压测。

## Tokenizer 和模型

正式新 V4-24576、V4-32768 均 NOT_ASSESSED，原因是没有获批 Train Split。
没有实际人工决定或正式定版，空决定模板不能准入。
历史 24,576 仅参考：16 条原创固定文本 Round-trip/EOS 全通过；中文分组的 386 个 Unicode 字符编码为 292 Token。
六类知识主题分别输出分词效率，不能据此声称模型具备知识能力。
逐类别、逐条和主题数据见 [比较报告](../reports/pr10-tokenizer-comparison.json)。

24,576 对应 86,528,640 参数、346,114,560 字节 FP32 权重；现有 INT4 group-128 张量负载估算 44,647,680 字节。
32,768 对应 91,771,520 参数，只作研究对照，不能作为正式 87M 定版。
上下文仍为 2048；模型、Loss、AdamW、Scheduler、INT4 格式与训练恢复语义均未修改。
没有导出正式权重，INT4 序列化大小和本次 GPU 峰值显存没有实测。

## 冻结与 PR9

正式构建不再因为 `selected: null` 而拒绝 Ingest/Quality/Dedup/Split/候选评估。
人工决定通过独立阶段绑定候选、训练文本身份、评估哈希、审核理由和审核人/日期，
并写入生效的 `tokenizer.selected`，避免修改原始 Plan 引发循环依赖或覆盖历史产物。
生产规模/24k 词表在 Tokenize、Audit、Freeze、verify_frozen 及 PR9 准入均验证决定证据。

原 PR8 九项审核、完整 Split 近重复门禁和 Manifest/阶段身份继续执行。
与 main 的策略投影逐项比较：Pilot 与正式配置的质量、去重、Split、污染资产/阈值、比例、Text Policy、
Tokenizer 训练选择规则和预算策略完全一致。仅调整来源登记方式、候选选择范围、候选目录与独立分词测评文本。
PR9 固定 1M 训练预算没有修改。

实际 PR9 Preflight：BLOCKED_DATA_NOT_READY，training=NOT_RUN。
阻断还包括策略证据、Tokenizer、Token 预算和 GPU 负载证明未就绪；没有调用正式训练器或创建训练 Checkpoint。
机器有 T1200，PyTorch 2.14.1+cu126/CUDA 12.6；这仅是环境盘点，不是工作负载验收。
未执行 GPU 训练，没有本次真实 Loss/PPL、吞吐、训练显存或恢复指标。

## 验证

使用 `.venv/bin/python`（Python 3.11.17）；常规测试设 `CUDA_VISIBLE_DEVICES=''` 固定 CPU。

| 检查 | 结果 |
|---|---|
| `python -m pytest -q -o addopts=''` | 715 passed、1 skipped，334.20 秒；跳过可选 CUDA 测试 |
| PR10 三个新增测试文件 | 30 passed |
| PR9 三个 Pilot/Readiness/Resume 文件 | 84 passed |
| `./scripts/smoke_test.sh` | PASS |
| `SYNTHETIC=1 ./scripts/smoke_test.sh` | PASS |
| `python scripts/check_training_reliability.py` | PASS；参数和 Optimizer 差异 0，样本顺序/RNG/LR 相同 |
| 两个正式配置 `--status-only` | NOT_READY，退出 0，无数据构建 |
| CLI 分阶段合成 Fixture | 未选择时评估/恢复成功；all 等待人工；freeze 拒绝，无 BIN/FROZEN |
| CPU V4 Demo | PASS，DEMO_ONLY、NOT_READY，无正式 FROZEN |
| CPU Nano 完整流程和 Resume | PR9 回归通过；4 次累计更新、120 监督 Token，连续/恢复模型及 Optimizer 完全一致 |
| 正反向冻结与 PR9 策略 | 合成原始 24,576 候选经决定/恢复/冻结通过测试策略；篡改决定拒绝；不符合生产策略仍拒绝 |
| 原 PR8 Split 固定快照 | Demo/1k 的文件哈希不变，断言未修改 |
| 100k/1M 文档压测、本次 GPU 训练 | NOT_RUN |

单独重跑 1k 原创合成基准：1,000 输入/清洗 KEEP、精确重复 32、近重复 21、最终 947；
扫描 CHECKED、hits 0、候选/特征截断均 0。约 9.22 秒、峰值 RSS 538,161,152 字节、SQLite 23,285,760 字节、
临时峰值 17,506,304 字节。它不代表真实语料预算，报告中 Token 估算仍明确标注估算单位。
只读 readiness 命令自身不运行基准；基准记录独立保存。

新增报告字段、选择文件与代码身份会改变后续 Manifest/整体 Content ID；这是身份机制的预期行为，
没有更新或覆盖历史数据/冻结资产，也没有跳过旧快照校验。旧小词表 Fixture 兼容；
缺少必要选择证据的历史生产规模报告不能默认升级为可冻结状态。

## 变更文件

- `.gitignore`、`README.md`
- `configs/corpus_v4.yaml`、`configs/corpus_v4_pilot_1m.yaml`
- `configs/corpus_v4_sources.review.json`、`configs/tokenizer-v4-evaluation.jsonl`
- `scripts/plan_corpus_v4.py`、`scripts/prepare_v4_sources.py`、`scripts/report_v4_readiness.py`
- `src/minillm/corpus_v4.py`、`src/minillm/tokenizer_v4.py`、`src/minillm/v4_admission.py`
- `src/minillm/v4_sources.py`、`src/minillm/v4_tokenizer_decision.py`、`src/minillm/v4_readiness.py`
- `tests/test_v4_source_onboarding.py`、`tests/test_v4_tokenizer_finalization.py`、`tests/test_v4_corpus_readiness.py`
- `docs/corpus-v4-source-review.md`、`docs/tokenizer-v4-selection.md`、`docs/tokenizer-v4-decision.md`、本文件
- `reports/pr10-corpus-readiness.json`、`reports/pr10-tokenizer-comparison.json`
- `reports/pr10-policy-compatibility.json`、`reports/pr10-benchmark-1k.json`、`reports/pr10-validation.json`

## 剩余条件

未发现新增代码级 P0/P1/P2；结论限本次 CPU 回归、Fixture 与已检查路径。
P0 数据前置阻断仍在：真实人工许可审核、可验证的原始/转换输入、足够且按类别覆盖的独立语料、
Train-only 24,576 候选训练与正式人工决定、污染审核、两套预算及正式冻结证据。
GPU 验收没有执行，不能由 CPU 测试或内存公式推断通过。

后续先完成真实来源审批与数据处理，再比较候选并记录人工决定；各版本通过九项审核后冻结，
Pilot 调用 verify_frozen/validate_v4_pilot_policy 和 PR9 只读 Preflight。
只有再取得有效 T1200 负载证据和显式训练授权，才可开展 PR9 约 1M 监督 Token GPU 预演。
100M 长训练仍不在本 PR 范围内。
