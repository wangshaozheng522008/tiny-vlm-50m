# 87M 中文语言 Pilot 首轮运行

正式运行使用 `minillm-corpus-v4-language-pilot-1m-v1`、`minillm-87m` 和 T1200 配置。冻结门槛是 1,020,000 个独立监督 Token。首轮从 Seed 42 的随机初始化开始，123 次更新各消费 8 个 1024 长度窗口，计划合计 1,006,632 个实际 Loss 目标。保留原有 45% / 30% / 15% / 10% / 0% 语料配比及 ±1 个百分点容差。

## 来源及 Tokenizer

从仓库根目录使用本地的已审核 Registry。本次 Registry 仅对固定的中文维基文库、中文维基百科、中文维基教科书和英文维基文库样本授权 1M 中文研究预训练。原有商用、对外再分发、公开权重、署名及归属条件继续适用。提供给工具的 Registry、批准证据、Tokenizer 决策和污染审阅记录留在本地。

```bash
PYTHON=.venv/bin/python
PILOT_CONFIG=configs/corpus_v4_language_pilot_1m.yaml
PILOT_REGISTRY=reports/v4-language-pilot-pr16-source-verification/source-review.approved.json
LANGUAGE_DATASET=artifacts/corpus/v4-language-pilot-first-training
REPORT=reports/v4-language-pilot-first-training
TRAINING_CONFIG=configs/pilot-v4-language-1m-t1200.json
EVALUATION_CONFIG=configs/evaluation.v4-language-pilot.json
TRAIN_WORKSPACE=artifacts/v4-language-pilot-first-training/workspace
test -f "$PILOT_REGISTRY"

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

若采用本次独立的污染处置配置，先将下面两个变量指向已生成的本地配置及其对应数据集。它仍受仓库固定语言策略校验。若重建后 Tokenizer 决策草稿的任何指纹不同，由人工确认新的证据。

```bash
PILOT_CONFIG=reports/v4-language-pilot-pr16-source-verification/corpus-reviewed-contamination.yaml
LANGUAGE_DATASET=artifacts/corpus/v4-language-pilot-first-training-reviewed
```

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
