# V5 中文问答 SFT 诊断

## 当前结果

实验基于 PR #21 合并提交 `a95c7cbc86e247361586cb8c23d67c3f8e59a24a`。已核对 `runs/v5-language-pilot-20m-repeat-v1/model.pt` 的 SHA256 `b5df228811485fe3cefa6fe1e910e70c7d836861f03479d4c7fad921ff656859`。模型有 86,528,640 个参数、147 个 FP32 权重张量，词表为 24,576。配置和 Tokenizer 文件哈希及 Tokenizer 指纹均已核对。未改动历史资产、预训练语料、架构或 Tokenizer。

| 项目 | 观测 |
| --- | --- |
| 合成数据 | Codex 直接编写 100 条验证候选，其中 11 条多轮。本机 Qwen 生成 55 条训练候选。未发生外部生成 API 调用 |
| 数据审核、冻结、实际样本 Mask 审计 | PENDING |
| CPU Smoke Test | 通过。项目原有小模型流水线跑通 CPU 预训练、SFT、INT4 导出和重载 |
| 专项 pytest | 49 项通过，覆盖 SFT Mask、尾部累积、恢复、量化来源与 V5 历史兼容 |
| 全量 pytest | 四个分片共覆盖 1,121 项。首轮 10 项涉及旧 INT4 测试夹具及旧评测凭据格式，修订后逐项复测通过。合并逐项结果为 1,121 项通过 |
| GPU 负载探测 | T1200、FP32、梯度检查点、512 长度、8 次累积、确定性模式。一次更新消费 4,096 个输入探测 Token、2,048 个探测监督 Token，3.294 秒，峰值分配 1,901,438,976 字节，峰值保留 1,939,865,600 字节 |
| PR #20、#21 历史 INT4 | 使用对应 FP32 源重新量化，codes、scales、FP16 张量和量化元数据均与历史缓存一致 |
| 20M Base 原始 V5 Val | 216 个窗口、220,968 个监督 Token，Loss 5.569691，PPL 262.3531 |
| 20M Base 的 97 题知识诊断 | 自由填空 0/97，Chat 候选 21/97，填空候选 17/97，按选项数计算的随机期望约 25.1/97 |
| 32 条训练候选预诊断 | T1200 上完成 160 次全参数更新。Assistant Loss 6.752604 → 0.000862，Greedy 逐字复现 0/32 → 32/32，Chat 回合结束 0/32 → 32/32 |
| 冻结集的正式 32 条诊断、正式 GPU SFT | PENDING |
| 500 条 Test、最佳 SFT INT4、人工评分和 30 组 SFT 问答对比 | PENDING |

本地 `reports/v5-language-sft-5k-v1/pytest-summary.json` 汇总了四个分片与复测证据及 SHA256。GPU 探测日志和环境信息见同目录的 `gpu-preflight/probe.json`，SHA256 为 `e2da4e0fdc8d1d84e0dd6e05f4b479a2005bf6796c59c4a9fa7037182c493e69`。探测使用程序生成的 Token ID，未生成语料或保存权重。此耗时只描述单步容量探测。

97 题的逐题条件概率、预测与汇总见本地 `reports/v5-language-sft-5k-v1/baseline-knowledge/`，Manifest 的 SHA256 为 `f99874f5f0a0f960db47b8889a45ed845cceeaaf1f27eef81071e3d2c7ca074a`。Base 在自由填空和候选排名中都缺少可靠的事实信号，现有观测不足以将失败归因于自由生成。干扰项独立复核以及 SFT 后对照仍是 `PENDING`。已见日常问答的记忆能力由候选预诊断验证，未见问题泛化、语言质量、遗忘和正式 SFT 的答案背诵仍待测量。

选择下一阶段时，若模型可以记住 32 条问答，而未见事实在自由生成和候选识别中持续较弱，优先扩大预训练。若候选识别明显高于随机期望且优于自由问答，继续用 SFT 改善问答表达。

## 数据生成与审核

### 无 API 批次

可直接导入模型在对话会话中生成的 JSONL，也可以导入本地推理产物。每行包含 `messages`、`topic`、`topic_category`、`fact_family`、`question_family`、`dialogue_family`。每个 50–100 条的批次附一个来源 JSON，记录 `model`、`source`、`evidence`、`prompt`、`parameters`、`usage`、`scope`、`api_calls=0`、`historical_answers_visible`。会话未提供的采样参数或用量如实记为未知。训练批次要求训练用途授权，且生成上下文没有历史题答案。

```bash
.venv/bin/python scripts/generate_v5_sft_data.py import-offline \
  /path/to/batch.jsonl /path/to/receipt.json \
  artifacts/v5-language-sft-5k-v1/offline-batches/living-001 --split train --batch living-001
.venv/bin/python scripts/generate_v5_sft_data.py collect-offline \
  artifacts/v5-language-sft-5k-v1/candidates artifacts/v5-language-sft-5k-v1/offline-batches/*
```

两批直接生成的验证候选位于 `artifacts/v5-language-sft-5k-v1/offline-batches/direct-val-001/` 和 `direct-val-002/`。以原 Tokenizer 统计得到 7,309 个输入 Token、3,015 个 Assistant 监督 Token。该会话此前读取过历史知识题答案，两批来源都限定为 `evaluation_only`。验证批次之间及与历史集的文本近重复审查无命中，独立语义与事实审核为 `PENDING`。

独立训练候选批次 `qwen-train-001` 使用 T1200 上的 Qwen3-4B Q4_K_M、llama.cpp b12810 和本地回环通信。模型请求只接收自行规划的 55 个生活情境和格式提示。59 次本地调用共记录 16,103 个输入 Token，其中 14,068 个命中缓存，输出 2,950 个 Token，耗时 84.55 秒，外部生成 API 费用为 0。逐条审核中，Codex 从 55 条结构合格候选中筛出 35 条适合隔离记忆实验的单轮问答。审核记录是独立模型审核，正式语料的完整审核仍待完成。[Qwen 权重及 Apache-2.0 许可](https://huggingface.co/Qwen/Qwen3-4B-GGUF/tree/bc640142c66e1fdd12af0bd68f40445458f3869b)

全量合并仍要求 8,000–10,000 条候选，正式 SFT 仍使用独立审核后发布的 5,000/500/500 冻结集。

### 冻结前的训练候选预诊断

在 35 条独立模型初筛可用的单轮训练候选中，以 Seed 42 抽取 32 条日常问答。从 PR #21 的 FP32 权重和新 AdamW 开始，使用与正式诊断一致的 160 次更新预算。训练耗时 206.05 秒，消费 24,480 个有效监督 Token，峰值 GPU 分配为 1,774.5 MiB。Greedy token 四元组重复率从 58.00% 降至 0%，字符三元组重复率从 65.23% 降至 0.10%。回合正常结束使用 Chat 模板的 `<|end_of_turn|>`。

| 完成更新数 | 训练样本 Assistant Loss |
| ---: | ---: |
| 0 | 6.752604 |
| 20 | 1.252020 |
| 40 | 0.090439 |
| 60 | 0.075515 |
| 80 | 0.006163 |
| 100 | 0.002017 |
| 120 | 0.001246 |
| 140 | 0.001002 |
| 160 | 0.000862 |

证据位于 `reports/v5-language-sft-5k-v1/candidate-probe32/`，含 20 条 Mask 解码审计、32 组前后真实生成对照及最终指标，`result.json` 的 SHA256 为 `2582a4ea3c331083c7f4254919ee59a0bccc6f17fd91d9bf2f155d7b7d4d67f7`。预诊断权重存放于独立的 `candidate-probe32` 目录，带 `DIAGNOSTIC_ONLY`。这次测量覆盖训练题复现。正式选模、泛化和 Test 流程仍等待冻结数据。

### 预诊断模型的未见题实测

以固定 Seed 42 从两个验证候选批次的五个类别各取六题。这 30 题均未出现在 32 条预诊断训练集中，共含 33 个回答回合。使用相同 CPU、Greedy 和 96 Token 预算，追问使用参考历史。结果仅用于观察小样本记忆会如何影响未见问题。

| 指标 | 20M Base | 32 条预诊断模型 |
| --- | ---: | ---: |
| 正常结束 | 3/33 | 33/33 |
| 平均 Token 四元组重复率 | 54.50% | 0.14% |
| 命中至少 12 字的训练答案片段 | 0/33 | 29/33 |
| 逐条查看后判定回答切题 | 0/33 | 0/33 |

例如，询问候鸟为什么会在湖边停下来，预诊断模型回答「可以，不过得确认一下店家有没有这种选择。你先说说想点什么。」源数据作者逐条查看后记录了这一判断，独立人审待完成。完整原始输出与定性记录在 `reports/v5-language-sft-5k-v1/candidate-probe-unseen-val/`。这些候选与正式锁定的 Test 输出分别保存。

### 本地全量流水线

`scripts/run_v5_sft_campaign.sh` 串联 98 个规划批次、生成、独立模型审核、语义筛查、冻结、32 条诊断、三轮正式 SFT 和未见问题评测。每个主题先保存范围和家族清单，再独立生成最多 100 条候选。生成端只有主题与素材。独立审核用先前已核查的正反例做质量校准，未达标时暂停正式冻结。相似度检索由 BGE 完成，边界匹配交由审查模型复核，预训练历史题与训练、验证、测试隔离在该阶段检查。

正式未见题评测先绑定 Validation 选模结果、全部节点权重和数据 SHA256，再读取 500 条 Test 进行 Greedy 及固定种子采样。评测保留所有原始回复，另外盲化比较 Base 和最佳 SFT 的语义回答与指令执行效果。人工评分保留 `PENDING`。实际进度写入 `artifacts/v5-language-sft-5k-v1/campaign-v1/phase` 及 `supervisor.log`；未见问题原始输出和逐题比较将在 `reports/v5-language-sft-5k-v1/unseen/` 发布。

### API 批次

使用已安装项目的 Python 环境，从仓库根目录运行。生成服务使用 HTTPS Chat Completions 接口。环境变量 `V5_SFT_API_URL` 是完整请求 URL，`V5_SFT_API_KEY` 是密钥。授权文件放在仓库外，字段包括 `provider`、`model`、`source`、`evidence`、`scope`、`currency`、`input_per_million`、`output_per_million`。只有核实条款或书面许可覆盖本次模型训练时才填写 `scope=training_allowed`。可用 `actual_models` 列出该证据覆盖的服务端实际模型 ID。费率应来自所选服务的计费记录，密钥只放在环境变量中。

```bash
mkdir -p reports/v5-language-sft-5k-v1
.venv/bin/python scripts/generate_v5_sft_data.py plan /tmp/v5-sft-batch-plan.json

for split in train val test; do
  .venv/bin/python scripts/generate_v5_sft_data.py generate --split "$split" \
    --phase families --authorization /path/to/authorization.json
  .venv/bin/python scripts/generate_v5_sft_data.py generate --split "$split" \
    --phase candidates --authorization /path/to/authorization.json
done

.venv/bin/python scripts/generate_v5_sft_data.py collect artifacts/v5-language-sft-5k-v1/candidates
```

各集合分开规划并生成。90 批预先限定主题范围，每批先生成 30 条细分事实、问题、对话族清单，再将该清单用于 100 条候选问答。70 批用于训练候选，验证和测试各 10 批。重跑已完成的操作会复用对应批次。失败和用量以批次记录，汇总成本使用服务返回的 Token 和授权凭据中的费率，另列缺失用量的请求数。历史评测题及答案只进入本地污染审核和评测。

独立审核员使用 `candidates.jsonl` 的编号和 `message_sha256` 记录审核结果。每条 JSONL 应含 `id`、`message_sha256`、`status`、`origin`、`reviewer`、`evidence`、`checks`、`fact_family`、`question_family`、`dialogue_family`。审核来源可以是 `human` 或 `independent_model`。模型审核记录实际模型、提示词、审查范围及证据位置，人审记录可核对的来源。审核通过时统一事实同义名称，并确保所有重复簇有一致的族标签。审核应覆盖简体中文、稳定事实与来源、问答相关性、自然度、多轮上下文、跨集合语义近重复、历史题及改写污染、答案模板集中度。字段名见 `minillm.v5_sft_data.CHECKS`。证据缺失记为 `PENDING`。

```bash
.venv/bin/python scripts/generate_v5_sft_data.py freeze \
  artifacts/v5-language-sft-5k-v1/candidates /path/to/independent-reviews.jsonl \
  artifacts/v5-language-sft-5k-v1/frozen \
  --audit reports/v5-language-sft-5k-v1/data-audit.json
```

冻结步骤检查 8,000–10,000 条候选和最终 5,000/500/500 配额，其中 Train 五类为 1,750/1,500/1,000/500/250。另检查角色与文本格式、各集合约 10% 多轮、家族隔离、历史题及候选间归一化精确重复、字符三元组 Jaccard 达 0.7 的近重复，以及长答复开头的集中度。简繁和深层语义判断由审核记录支持。发布包含带审核来源的 JSONL、内容 ID、每个文件的 SHA256、类别、字符长度及实际 Tokenizer 长度与有效监督统计。

额外泛化文件由独立流程生成并审阅。每条包含 `id`、`messages`、`topic_category`、`fact_family`、`question_family`、`source`、`review`，有训练锚点时含 `anchor_id`。审核对象同样记录来源、模型或审核员、证据和各检查结果。

| 子集 | 样本设计 |
| --- | --- |
| `fact_generalization` | 已学事实，独立编写非近重复的新问法，链接同一训练事实族 |
| `expression_generalization` | 用新情境和不同表述检验已学指令或沟通意图，所需信息包含在问题中，链接同一训练问题族 |
| `unseen_knowledge` | 在训练中从未出现的事实族，无训练锚点 |

## 训练和恢复

```bash
.venv/bin/python scripts/run_v5_language_sft_5k.py verify-base
.venv/bin/python scripts/run_v5_language_sft_5k.py gpu-probe
.venv/bin/python scripts/run_v5_language_sft_5k.py baseline-knowledge
.venv/bin/python scripts/run_v5_language_sft_5k.py diagnose
.venv/bin/python scripts/run_v5_language_sft_5k.py train
.venv/bin/python scripts/run_v5_language_sft_5k.py train --resume state.pt
.venv/bin/python scripts/run_v5_language_sft_5k.py evaluate --generalization /path/to/generalization.jsonl
```

`verify-base`、`gpu-probe` 和 `baseline-knowledge` 可在数据冻结前运行。训练前会再次采集 GPU 快照并导出 20 条真实训练样本的输入 ID、标签、解码与结束标记位置。每轮答案从和推理提示完全一致的前缀之后开始监督，Chat 的 EOS 使用现有 `<|end_of_turn|>`。超长或无法与推理模板对齐的样本禁止冻结。Batch Size 为 1，按样本动态长度运行。

32 条诊断用 Seed 42 从 Train 的短答案单轮样本固定抽取，预算 160 次更新，每 20 步测量训练集 Assistant Loss 并记录学习曲线。诊断采用峰值 LR 3e-4，以有限步数观察记忆能力。最终报告包含实际生成、准确复现、结束标记、角色和重复统计。若诊断显示 Loss 或复现异常，先检查 Mask、模板、优化器、Tokenizer 和生成边界。

正式训练和诊断分别从相同的 20M FP32 权重、新 AdamW 状态启动。正式配方见 `configs/pilot-v5-language-sft-5k-t1200.json`。实际 5,000 条时每 Epoch 625 次更新，3 Epoch 共 1,875 次，Warmup 56 次，最后一次更新到达最低 LR。计算以实际微批次数向上取整，尾部组按有效 Assistant Token 归一化。训练 Manifest 绑定仓库提交、运行包全部 Python 文件与入口和配置的 SHA256、冻结数据、基础权重及环境。训练日志按更新记录监督 Token、耗时、显存与 LR。结束汇总另记录独立与累计输入和监督 Token、有效优化耗时及吞吐。

每 Epoch 执行完整 Validation。保存 Epoch 0/1/2/3、最佳权重和最终权重，恢复状态每 50 步更新。主状态损坏时使用 `--resume state.prev.pt`。恢复会重建固定 Sampler 并确认游标、监督 Token 和 LR，同时恢复 AdamW 及 RNG。历史 Manifest 使用原有训练与恢复协议。PR #20/#21 历史评测只兼容已知旧实现哈希，重用 INT4 时新增确定性张量验证。正式及诊断 SFT 权重带诊断用途标记。

## 选模与评测

仅完整 Validation Loss 决定最优权重。锁定方案、数据和所有节点 SHA256 后读取 Test。历史 300 题沿用续写模板、预算、Greedy 和固定种子采样配置。Chat 另用 96 Token 预算、Greedy 和标准采样三个固定种子，逐轮使用参考历史上下文。准确复现与长片段匹配为机械指标。相关性、自然度、语义准确性、指令遵循和上下文理解，由评分人员依据原始输出审查，尚未完成的评分为 `PENDING`。

97 道历史可自动计分的世界知识题增加独立填空提示与同类候选。备选答案逐一在相同条件下计算序列概率和每 Token 平均对数概率，排名使用后者，候选项没有在提示中排列。另报告自由填空及按选项数计算的随机基线。选项与题干独立复核状态为 `PENDING`。候选排名受提示表述、词频和干扰项难度影响，仅与自由生成结合用于初步诊断。

各节点输出原始 V5 Val Loss、问答 Loss/PPL、重复率、结束率、角色混淆和训练答案匹配。最终对照 Base、Epoch 1/2/3、最佳 FP32 与最佳 INT4，写入按五类各选六题的真实问答比较。INT4 使用既有算法、Group Size 128、预算 50,000,000 字节，缓存复用前从 FP32 逐张量复核。
