# 2M 中文语言能力对照训练

基于 PR #17 合并提交 `4da97bc93488919f7ff38aad67513e5d6c5a216e`。两次同一训练计划内的观测用于比较新增训练量。历史 PR #16 的 Step 123 另作对照，其学习率总计划为 123 步，实际窗口集合与顺序也可能不同。

## 当前状态

**`BLOCKED_SOURCE_SCOPE_APPROVAL`**。T1200 可用。本地四个已审来源仅批准用于 `minillm-corpus-v4-language-pilot-1m-v1`。将原批准文件传给 2M build 会被拒绝。2M 已批准有效训练预算为 **0**。正式冻结、GPU Probe、训练、四节点统一评测与 INT4 导出均为 **`NOT_RUN`**。没有新的训练 Loss、生成能力或量化结论。

| 对照节点 | 计划累计监督 Token | 本次统一评测 |
|---|---:|---|
| 历史 PR #16 Step 123 | 1,006,632（历史实测） | NOT_RUN。历史 Greedy 为 0/197 |
| 新实验 Step 0 | 0 | NOT_RUN。历史原始权重文件已核验 |
| 新实验 Step 123 | 1,006,632 | NOT_RUN |
| 新实验 Step 245 | 2,005,080 | NOT_RUN |

## 待审批清单与真实库存准备

使用独立版本 `minillm-corpus-v4-language-pilot-2m-v1`。执行过以下只读准备命令。默认报告路径为 `reports/v4-language-pilot-2m-v1/prepare/`，含带 SHA256 的清单、待审批 Registry、正式计划和证据。另有计划学习率图 `planned-learning-rate.svg`，训练尚未执行。

```bash
.venv/bin/python scripts/run_v4_language_2m.py prepare
```

| 来源 | 复核对象 | 2M 决定 |
|---|---|---|
| 中文维基文库 | PR #16 固定样本、源快照和作品级使用范围 | 待人工 |
| 中文维基百科 | PR #16 固定样本、源快照、许可与归属 | 待人工 |
| 中文维基教科书 | PR #16 固定样本、源快照、许可与归属 | 待人工 |
| 英文维基文库 | PR #16 固定样本、源快照和作品级使用范围 | 待人工 |

准备阶段逐一核对了四份本地候选来源快照、样本文件和采样清单的 SHA256，保留许可陈述和历史训练范围。历史三组分割的 6,881 篇文档已逐条核对原始与规范化文本哈希，分割间完全重复数为 0。正式 build 会重新核对快照并复现采样。审批人从 `source-review.pending.json` 制作独立 Registry，逐来源复核样本及使用范围，记录真实审核人、日期、确认内容与许可证据。现有来源审核字段也须由审核人更新。新决定绑定 **2M 版本、原始冻结内容 ID 和 2,005,080 有效监督 Token 预算**。原决定仍然只对应其历史版本。候选范围限定这四份原样本，审批待处理文件中全部为 `REVIEW_REQUIRED`。

冻结基线内容 ID 为 `7222651ef09e4e09c2d613474f50712e8233d1d4ff9017bda897e597ebdb6da5`，Tokenizer 指纹为 `845a24b9f21458b1d2161c60dfc8a77d5ed40235ece7ef46891677e423727b8a`。固定 Val BIN SHA256 为 `eaa0a3ca78a4371cde39f29cf0397a35183ba81fc8dc00cd8ace58e734b7ac10`，Test BIN 为 `d4133d8142a8d8d5689afd3fb83c3c31cf4594bd7590aa150ba4a41dbc14048f`。

### 真实语料打包预览

用原始 Train 文档池和冻结词表进行确定性的整文档编码与 sampler 重放。下表是 **`PREVIEW_ONLY_NOT_FROZEN`**，实际训练消费量仍为 0。类别权重保持 45/30/15/10/0，容差为 ±1 个百分点。

| 类别 | 全量原始编码 | 预览入选原始 Token | 预览冻结有效 Token | 预览 sampler 监督 Token | 预览消费占比 |
|---|---:|---:|---:|---:|---:|
| 中文通用 | 1,176,437 | 907,221 | 906,335 | 900,197 | 44.90% |
| 中文百科 | 1,783,752 | 602,258 | 601,669 | 600,646 | 29.96% |
| 中文教育 | 1,281,909 | 301,095 | 300,801 | 298,755 | 14.90% |
| 英文通用 | 1,714,855 | 206,422 | 205,482 | 205,482 | 10.25% |
| 数学／逻辑／代码 | 0 | 0 | 0 | 0 | 0.00% |
| 合计 | 5,956,953 | 2,016,996 | 2,014,287 | 2,005,080 | 100.00% |

预览包含 1,969 个完整窗口，计划从 Seed 42 的排列中消费前 1,960 个唯一窗口。窗口顺序的 SHA256 为 `b0c7071d3c933c7ec10b21019c9e5ab9a2eed0957ce67af656b2abb60ded4da2`。预览的各类别原始数量以本地 `packing-preview.json` 的精确编码记录为准。

## 获批后的执行流程

从仓库根目录运行。新的 Registry 和污染处置文件仅保存在本地。两个已知历史高风险匹配的人工决定只能在确认绑定的 Train 与评测资产一致且适用后显式提供。全量污染扫描仍会重新执行。调整内容计划应使用新的 `--dataset` 目录与新的报告目录，后续阶段沿用同一数据路径。

```bash
PYTHON=.venv/bin/python
RUNNER=scripts/run_v4_language_2m.py

"$PYTHON" "$RUNNER" build --source-registry "$REVIEWED_2M_REGISTRY" \
  --contamination-decisions "$REVIEWED_CONTAMINATION_DECISIONS"
"$PYTHON" "$RUNNER" probe
"$PYTHON" "$RUNNER" execute
"$PYTHON" "$RUNNER" evaluate
```

Build 复跑 V4 的 Ingest、Quality、Dedup、Split。新结果须通过隔离审计，并与已冻结历史 Train/Val/Test 的文档内容及分组逐条核对。权利范围由 2M Registry 单独绑定，原分割文件保留原始记录和字节。复用此前人工审核通过的 Tokenizer，重新编码后核对 Val/Test BIN 与 Token Index 字节级一致。新 Train 按原顺序规则选择整文档、保留 EOS，独立计算监督预算，再通过污染、编码、配额和冻结门禁。构建成功后须同时获得 `READY` 和 `POLICY_MATCH`。

训练继续使用 86,528,640 参数模型、原 Step 0 的 SHA256 `bfdd11421cb247d5b8c88b7b268f8f4353e0bc12c5fcfae49085929799cc3e99`、Seed 42、FP32、1×8 梯度累积和 1,024 序列。单一 AdamW 学习率计划跨 245 步，4 步 warmup，峰值 `3e-4`，最小 `3e-5`，余弦衰减，梯度裁剪 1.0。Step 0、123、245 各归档独立的权重、训练状态和哈希清单。完整评测从归档读取，独立于训练进程。训练器内已有的 Val 计算保留随机状态。

需要恢复时，在同一 2M 数据版本与工作区中执行 `execute --resume state.pt --out <新报告路径>`。如经确认需要前一个合法状态，用 `state.prev.pt`。恢复时沿用原优化器、RNG、sampler 游标和 245 步 Scheduler。已记录但尚未提交的训练日志先归档，再从正式状态中重放。

## 已实现的统一评测

- 固定原 1M Train 的 64 个代表性窗口，共 65,472 个监督位置。四个 FP32 节点和最终 INT4 使用同一套公共 Train，并使用原 Val 的全部 101 个完整窗口。逐位置计算总体及分类 Loss、PPL。原 Test 的 101 个完整窗口仅在预定的最终 FP32 节点使用一次。参考后缀提供 teacher-forced Loss、下一 Token 概率和排名。
- 每个节点运行原始 300 题、五条固定续写和四条诊断前缀。Greedy 和三种采样策略统一使用每题原预算。采样 Seed 为 42、43、44，重复惩罚固定为 1.0。保存每条文本、完整 Token ID、停止原因和退化指标。
- 保留原评分器和知识分类。人工评估项标为 `PENDING`，并记录评估用 rubrics、通顺程度、连贯性、相关性、事实准确性与无意义重复项目。
- 最终 Step 245 按 Group Size 128 导出 INT4，校验 50,000,000 字节上限。按相同提示和配置比较 FP32/INT4 的 Loss、Top-1、Top-20 与生成。INT4 权重以反量化 FP32 运行。
- 评测报告包含 `comparison.md`、配对窗口与配置、各节点文件身份、实际监督消费、每步 Loss/LR 曲线数据、逐条预测，以及 `fp32-int4.json`。原始语料、权重和运行报告保留在 Git 忽略目录。

## 软件回归与真实资产核验

- `python -m pytest -q -o addopts=''` 全量通过 1,027 项，0 项跳过，耗时 320.08 秒。测试在实现提交 `9fddffe` 的隔离 worktree 执行，并接入真实中文 Tokenizer 效率测试语料。
- 2M 准入、训练恢复与训练谱系专项测试通过 105 项。恢复测试使用 CPU 软件夹具验证了中途保存、随机状态恢复、权重一致性及不中断的学习率计划。
- 1M 语言 Pilot 专项通过 8 项。CPU Smoke Test 通过。
- 原 1M 冻结集重新通过 `FROZEN_VERIFIED` 和 `POLICY_MATCH`。提交原 1M Registry 给 2M build 时，入口以退出码 2 拒绝扩大适用范围。
- 运行日志和真实资产核验凭证位于本地 `reports/v4-language-pilot-2m-v1/verification/`。

## 后续扩展判断

先完成针对 2M 的来源批准、重建和真实实验。比较时结合配对 Loss、逐条输出及人工评分判断中文语句和基本世界知识是否改善。Step 123 到 Step 245 共用一个学习率计划，后半段的学习率和已消费语料仍随更新变化。

PR #17 在固定名义配额和现有容差下找到的最大打包预览为 2,610,696 个监督 Token。4M 的名义配额至少还缺 625,602 个中文通用监督 Token，10M 也需要独立的来源供给和审核。当前没有 2M 结果可支持扩大训练。[PR #17 预算与范围核算](v4-language-degeneration-diagnostics.md#扩展库存与审批)
