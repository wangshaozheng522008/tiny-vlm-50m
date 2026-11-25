# 中文语言能力与世界知识训练路线

## Phase 1 · 1M Pilot

使用独立版本 `minillm-corpus-v4-language-pilot-1m-v1`。序列长度 1024，Seed 42，数据按文档家族拆为 96% Train、2% Val、2% Test。保持 MiniLLM-87M Transformer 配置及 50 MB INT4 导出目标。Pilot 验证来源许可、数据处理、Tokenizer、训练入口、实际 Loss 计数、Checkpoint 与评测流程。

| 类别 | 旧 Pilot 比例 | 新 Pilot 比例 | 新监督 Token 下限 | 历史 Train 估算供给 |
|---|---:|---:|---:|---:|
| 中文通用 | 40% | 45% | 459,000 | 1,213,233 |
| 中文百科 | 25% | 30% | 306,000 | 1,446,983 |
| 中文教育 | 15% | 15% | 153,000 | 1,071,842 |
| 英文通用 | 10% | 10% | 102,000 | 1,591,906 |
| 数学、逻辑与代码 | 10% | 0% | 0 | 0 |

历史 Train 估算来自 PR #14 的字符口径。同一历史构建从 8,548 条样本文档中得到 6,922 条 Quality KEEP、6,881 条去重后文档和 6,605 条 Train 文档。工具验证了四个 Wikimedia 来源的原始文件指纹、样本 Manifest、抽样复现、旧审批以及旧阶段证据。新版本真实数据仍处于 `DATA_NOT_READY`。新版本全量实测供给、入选文档数、有效监督 Token 及类别实测比例均待测。

### 来源审批与构建

下列命令以仓库根目录为工作目录。所有本地审核文件、原始语料以及构建产物保存在 Git 忽略的目录。

```bash
python scripts/prepare_v4_language_pilot.py \
  --historical-dataset artifacts/corpus/v4-pilot-pr14-20261118-v2 \
  --out reports/v4-language-pilot-reuse
```

审核人从 `source-review.draft.json` 创建本地审核 Registry，逐条复核中文维基文库、中文维基百科、中文维基教科书和英文维基文库。草稿绑定原有样本文件及样本 Manifest SHA256，并注明历史样本来源版本。审核人填写新版本的训练授权判断、商用及再分发条件、归属和其他义务。旧审批只覆盖旧 Pilot。未通过审核的文档与来源留在原有审核流程。

```bash
PROFILE=configs/corpus_v4_language_pilot_1m.yaml
DATASET=artifacts/corpus/v4-language-pilot-1m-v1
python scripts/plan_corpus_v4.py --config "$PROFILE" --source-registry "$REVIEWED_REGISTRY" \
  --out "$DATASET" --stage evaluate
```

上面执行全新的 Ingest、Quality、Dedup、Split 和候选评估。新 Split、Tokenizer 字符采样配比和评估资产改变了训练与评估身份。Tokenizer 只读取经过 Split 校验的 Train 文档。候选报告附带现代中文说明文、常用词、口语、百科及既有英文、代码和编码往返测试。

历史 24,576 候选为 86,528,640 参数，中文 1.5197 字/Token、英文 1.3333 Token/词。历史 32,768 候选为 91,771,520 参数，中文 1.5755 字/Token、英文 1.2222 Token/词。24,576 优先进入人工评审，两个历史指标均不可充作新版本指标。INT4 Tensor Payload 只估算张量字节数。正式导出后独立实测完整文件是否小于 50 MB。

候选评估会生成带指纹的 `tokenizer-decision.draft.json`。审核人复制草稿，填写实际审核人、ISO 日期、理由和模型预算结论。仓库中的 [决策草稿模板](../configs/tokenizer-v4-language-decision.draft.json) 展示字段。来源审批满足后，缺少有效 Tokenizer 决策的阻断状态是 `HUMAN_TOKENIZER_DECISION_REQUIRED`。

```bash
python scripts/plan_corpus_v4.py --config "$PROFILE" --source-registry "$REVIEWED_REGISTRY" \
  --out "$DATASET" --stage all --decision "$TOKENIZER_DECISION"
python scripts/analyze_v4_pilot_supply.py --config "$PROFILE" --source-registry "$REVIEWED_REGISTRY" \
  --dataset "$DATASET" --out reports/v4-language-pilot-supply
python scripts/plan_corpus_v4.py --verify "$DATASET"
```

### 选样、监督计数和冻结

用所选 Tokenizer 测量全量 Train 供给。选择从每个正权重类别的文档开始，按 `hash(seed, document_id)` 排序取前缀。保留整份文档、EOS 和来源元数据。每类样本只取一次，Val/Test 继续独立发布。有限的最后一个类别文档可继续加入以闭合 packing window。选择报告记下输入 Split 和选中文件指纹、逐文档 Token 数、供给、入选数和实际监督分布。

1024 个输入 Token 的完整窗口提供 1023 个 Loss 目标。要达到 1,020,000 个目标，至少需要 998 个完整窗口，即 1,020,954 个监督 Token。按文档截取类别会使最后一份文档跨过目标，最后一类还要考虑无 Loss 的窗口起始位置和残尾。门禁独立验证每类监督下限、总量和占比。版本策略允许每个正权重类别偏离目标占比最多 1 个百分点，以容纳整文档和完整窗口的粒度；超过容差或低于任一 Token 下限均阻止 Freeze。零权重类别标记为 `NOT_REQUIRED`。

污染检查覆盖全量 Train，包括 Tokenizer 能接触的文档。冻结需要通过来源、Quality、Dedup、Split、Tokenizer 决策、选择重放、Token Budget、污染和 Audit 验证。历史产品与新构建各自保留证据。当前真实候选因审批范围阻断，尚未运行新版本构建或模型训练。用全部固定资产对历史 6,605 条 Train 文档做了一次只读预扫，得到 2 条待处置高风险命中，均来自既有资产。新语言题集在这次预扫中没有高风险命中。正式新版本需要重新扫描与处置。

### 固定评测

[中文语言与世界知识评测集](../data/evaluation/chinese-language-v1/tasks.jsonl) 含 300 条原创评测，覆盖流畅性、日常常识、地理历史、自然科学、概念解释和简单指令。历史和科学事实附参考来源，评测内容和文件哈希由版本 Manifest 固定。完整匹配有明确的答案归一化和少量别名。关键词命中单独报告，开放式题目使用人工 rubric。重复率、乱码线索、生成空白、停止原因和句末标点辅助定位问题。已有污染规则继续阻断未解决的高风险评测复制，并区分常识重合。

用同一评测集对真实 Checkpoint 分别运行预训练 raw completion 与 SFT chat，报告内保留模式。以下参数关闭旧候选和额外固定提示，只运行这 300 条。

```bash
python scripts/evaluate_model.py --ckpt "$CKPT" --tokenizer "$TOKENIZER" \
  --stage pretrain --benchmark data/evaluation/chinese-language-v1/tasks.jsonl \
  --prompts "" --candidate-benchmark "" --out reports/language-eval-pretrain
```

SFT 将 `--stage` 改为 `sft` 并使用独立输出目录。人工评审在 `predictions.jsonl` 中逐条查看题目的 rubric 和实际回答。当前没有本 Pilot 模型成绩。

## Phase 2 · 100M 中文基础预训练

从独立 Pilot 的数据和训练验证结果出发，另立中文基础训练方案，保持当前正式 100M 配置与门禁。以现代中文说明文、日常生活常识和通俗科普为重点，补齐基础社会与地理知识。按授权采集可追溯的候选版本，例如明确允许目标用途的公共机构文本或有明示授权的百科／教育文本；逐份复核文本与第三方材料权利、署名、再分发和模型训练条款。可以与内容作者建立书面授权或委托原创文本，以补充稀缺主题。

历史 [中文质量分析](chinese-language-quality-report.md) 给出了字形、语体、噪声和重复段落线索。扩充前按类别和来源抽样复核这些启发式指标，优先收集自然现代说明、科普、地理与生活文本，再独立评估新增覆盖。外部候选完成版本化人工来源审批后方可准入。

## Phase 3 · 500M–1B Token 扩展预训练

逐批扩充高质量现代中文、百科及通用文本，保持跨批去重、文档家族隔离和评测污染检查。分别跟踪数据增量、实际受监督 Token、Loss、生成连贯性和开放式人工评价。用固定评测比较知识覆盖与表达稳定性，按真实导出文件核对体积目标。

## Phase 4 · 中文问答 SFT

建立独立的 SFT 数据版本，首批规划 5,000–20,000 条。覆盖高质量自然问答、现代中文对话、概念解释和简单指令。优先委托或自编、人工复核的问答和对话，适用时记录事实来源与授权。按对话和来源家族拆分，保持预训练、SFT 与固定评测资产的独立管理，并在训练前扫描重合。
