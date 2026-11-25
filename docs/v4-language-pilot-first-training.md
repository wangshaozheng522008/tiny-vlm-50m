# 87M 中文语言 Pilot 首轮运行

正式运行使用 `minillm-corpus-v4-language-pilot-1m-v1`、`minillm-87m` 和 T1200 配置。冻结门槛是 1,020,000 个独立监督 Token。首轮从 Seed 42 的随机初始化开始，123 次更新各消费 8 个 1024 长度窗口，计划合计 1,006,632 个实际 Loss 目标。保留原有 45% / 30% / 15% / 10% / 0% 语料配比及 ±1 个百分点容差。

## 来源及 Tokenizer

从仓库根目录使用本地的已审核 Registry。本次 Registry 仅对固定的中文维基文库、中文维基百科、中文维基教科书和英文维基文库样本授权 1M 中文研究预训练。原有商用、对外再分发、公开权重、署名及归属条件继续适用。提供给工具的 Registry、批准证据、Tokenizer 决策和污染审阅记录留在本地。

```bash
PYTHON=.venv/bin/python
PILOT_CONFIG=reports/v4-language-pilot-pr16-source-verification/corpus-reviewed-contamination.yaml
PILOT_REGISTRY=reports/v4-language-pilot-pr16-source-verification/source-review.approved.json
LANGUAGE_DATASET=artifacts/corpus/v4-language-pilot-first-training-reviewed
REPORT=reports/v4-language-pilot-first-training
TRAINING_CONFIG=configs/pilot-v4-language-1m-t1200.json
EVALUATION_CONFIG=configs/evaluation.v4-language-pilot.json
TRAIN_WORKSPACE=artifacts/v4-language-pilot-first-training/workspace
test -f "$PILOT_CONFIG" && test -f "$PILOT_REGISTRY"

"$PYTHON" scripts/plan_corpus_v4.py --config "$PILOT_CONFIG" \
  --source-registry "$PILOT_REGISTRY" --out "$LANGUAGE_DATASET" --stage split
"$PYTHON" scripts/plan_corpus_v4.py --config "$PILOT_CONFIG" \
  --source-registry "$PILOT_REGISTRY" --out "$LANGUAGE_DATASET" --stage evaluate
```

从数据集目录复制 `tokenizer-decision.draft.json` 到本地审核目录。确认实际评估与指纹后，由人工填写审核人、日期、理由及预算结论。24,576 是正式 87M 候选，32,768 为研究对照。Baseline 来自固定配置，评估证据会校验其规范路径、可用性、SHA256 和字节数。改动 Baseline 会使旧评估证据失效。

```bash
TOKENIZER_DECISION=reports/v4-language-pilot-pr16-source-verification/tokenizer-decision.approved.json
test -f "$TOKENIZER_DECISION"
"$PYTHON" scripts/plan_corpus_v4.py --config "$PILOT_CONFIG" \
  --source-registry "$PILOT_REGISTRY" --out "$LANGUAGE_DATASET" \
  --stage select --decision "$TOKENIZER_DECISION"
```

## 污染处置与正式冻结

全量 Train 重新扫描固定评测。审核人按现有格式记录高风险命中的决定，每行包含 `train_document_id`、`benchmark_id`、`action`、`reviewer` 和 `reason`。只有人工确认的普通短语重合可以用 `accept_risk`。有效的 `exclude` 判定要求受影响文档从同范围、再经审核的输入退出，随后重建并重验 Split、Tokenizer 与训练身份。

本次使用的本地配置沿用仓库固定语言策略，`contamination.decisions` 指向经确认的本地处置文件。它使用独立数据集目录。若改动需重新生成数据集，复核新的 Tokenizer 决策草稿是否逐字节相同。证据发生变化时，由人工确认新的指纹与评估。

确认来源、Tokenizer 和污染处置后，使用同一配置及输出目录运行各阶段。也可用现有 `--stage all --decision "$TOKENIZER_DECISION"` 在一个进程中顺序执行各门禁。

```bash
for stage in tokenize contamination budget audit freeze; do
  "$PYTHON" scripts/plan_corpus_v4.py --config "$PILOT_CONFIG" \
    --source-registry "$PILOT_REGISTRY" --out "$LANGUAGE_DATASET" --stage "$stage" || break
done
"$PYTHON" scripts/plan_corpus_v4.py --verify "$LANGUAGE_DATASET"
```

只有同时得到 `FROZEN_VERIFIED` 和 `POLICY_MATCH` 才进入 GPU 验证。选样报告保留全量编码供给、入选文档、原始 Token、完整窗口与尾部、逐类实际监督 Token 和比例。训练再按已持久化 Checkpoint 的 sampler Seed 与游标重放窗口，从 Token Index 精确核算各类别实际消费量。

## T1200 Probe、训练与验证

```bash
"$PYTHON" scripts/run_v4_training_pilot.py --dataset "$LANGUAGE_DATASET" \
  --config minillm-87m --training-config "$TRAINING_CONFIG" \
  --evaluation-config "$EVALUATION_CONFIG" --workspace "$TRAIN_WORKSPACE" \
  --out "$REPORT/probe" --probe
```

Probe 返回 `WORKLOAD_PASS` 后，用同一个 Workspace、同一训练配置与评测配置执行。

```bash
"$PYTHON" scripts/run_v4_training_pilot.py --dataset "$LANGUAGE_DATASET" \
  --config minillm-87m --training-config "$TRAINING_CONFIG" \
  --evaluation-config "$EVALUATION_CONFIG" --workspace "$TRAIN_WORKSPACE" \
  --out "$REPORT/execute" --execute
```

`pilot-v4-1m-report.json` 汇总实际训练消费、CUDA 峰值、耗时和吞吐。Step 0 与最终 Checkpoint 在相同固定 Train / Val 窗口、Tokenizer、Seed 和生成配置上评估。训练过程记录的在线 Loss 与 Checkpoint 定点评估分别保留。300 条正式题集继续按 raw 模式评估。另有独立冻结的 5 条中文续写前缀，原始文本连同已有评分依据全部留在逐条预测及对比报告。句末标点是完整性的启发式指标。基本常识得分变化和开放式生成均需结合样本检查。

验收使用现有训练分析的累计 Token 曲线 CSV、Checkpoint 与优化器恢复验证。正式训练通过验收后可用现有量化入口导出并检查文件大小及重载。

```bash
"$PYTHON" -m minillm.quantize \
  --ckpt runs/v4-language-pilot-first-training/model.pt \
  --out runs/v4-language-pilot-first-training/model-int4.pt \
  --bits 4 --budget-bytes 50000000 --device cpu
```

保留 87M 本地权重与私人审核记录。进入 100M 阶段还需独立完成该阶段的数据和预算准入。首轮用于验证工程链路和初步语言学习趋势。

## 首轮真实运行结果

四个固定来源已取得本次人工批准。重新构建得到 8,548 条输入、6,922 条 Quality KEEP、6,881 条去重后文档，Train／Val／Test 为 6,605／138／138 条，Split 隔离通过。人工确认了真实 Train-only `v4-24576` 候选，指纹为 `845a24b9f21458b1d2161c60dfc8a77d5ed40235ece7ef46891677e423727b8a`。两个 `The morning sun` 短语命中均已人工处置，正式污染审计为 `REVIEWED`，未处置高风险命中为 0。最终 Audit 十项通过。内容 ID `7222651ef09e4e09c2d613474f50712e8233d1d4ff9017bda897e597ebdb6da5` 获得 `FROZEN_VERIFIED` 与 `POLICY_MATCH`。

| 类别 | 全量编码供给 | 冻结监督 Token | 冻结占比 | 训练消费监督 Token | 消费占比 |
|---|---:|---:|---:|---:|---:|
| 中文通用 | 1,176,437 | 459,004 | 44.5565% | 444,682 | 44.1752% |
| 中文百科 | 1,783,752 | 308,633 | 29.9597% | 303,518 | 30.1518% |
| 中文教育 | 1,281,909 | 154,931 | 15.0395% | 151,862 | 15.0861% |
| 英文通用 | 1,714,855 | 107,593 | 10.4443% | 106,570 | 10.5868% |
| 数学／代码 | 0 | 0 | 0% | 0 | 0% |
| 合计 | 5,956,953 | 1,030,161 | 100% | 1,006,632 | 100% |

冻结集含 1,007 个完整窗口和 350 个未监督的尾部 Token。T1200 `WORKLOAD_PASS` 的显存峰值为分配 1,902,630,912 字节、保留 2,239,758,336 字节，完成三次代表性更新和 Checkpoint 读回。正式训练从独立随机 Step 0 初始化，在 767.187 秒内完成 123 次更新，重放验证消费了 984 个唯一窗口。日志覆盖口径平均为 1,350.3 Token/s，训练过程的显存峰值为分配 1,901,320,192 字节、保留 2,239,758,336 字节。训练器控制台另有按每轮平均值外推的预算估计；实际值经训练日志、独立窗口和 Checkpoint 校验。

| 65,536 目标定点口径 | Step 0 | Step 123 |
|---|---:|---:|
| Train Loss | 10.2452 | 7.4426 |
| Train PPL | 28,147.38 | 1,707.16 |
| Val Loss | 10.2494 | 8.0911 |
| Val PPL | 28,265.45 | 3,265.43 |
| 自动评分正确数 | 0 / 197 | 0 / 197 |
| 汉字比例 | 52.69% | 0% |
| 平均 Token 4-gram 重复率 | 86.92% | 95.78% |
| 空输出率 | 0% | 3.28% |
| 异常字符比例 | 0.82% | 0% |

全量 Val 的最终 Loss 为 7.9653、PPL 2,879.38，最后一个在线 Train Loss 区间为 7.9523。固定题集和五条续写均采用相同的 raw greedy 配置。Step 123 五条续写中，四条只重复逗号，一条输出句号后连续换行。题集中的基础事实自动评分没有提升，开放式题目保留人工评审依据。当前测量显示 Loss 下降，生成结果仍未展示中文流畅性或基础知识能力。

工程验收为 `ENGINEERING_PASS`。正式运行连续完成，最终 Checkpoint 的权重、优化器和 RNG 已验证可恢复，实际中断恢复另由软件夹具覆盖。INT4 文件为 44,730,004 字节，通过 CPU 重载、权重绑定与生成检查。模型以及完整 Loss 曲线、逐条 raw 输出、配对评测和本地审核证据留在工作区。正式 100M 配置的独立来源检查仍返回 `NOT_READY`。下一阶段需补齐该配置的审批及数据门禁，并用小规模诊断复核当前贪心生成的标点退化。

语言 Pilot 专项测试 8 项通过，全量 996 项通过，CPU Smoke Test 通过。
