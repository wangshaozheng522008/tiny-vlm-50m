# 本轮只读诊断与实现验收

完整数值、资产哈希、环境及代码文件身份见 [quality-stage-evidence.json](quality-stage-evidence.json)。这是本次新评测，不覆盖 [pilot-evidence.json](pilot-evidence.json) 中的历史实机结果。

## 输入策略影响

只读扫描v3 Train的122,910篇文本，其中116,005篇会在 `width-fold-prose-v1` 下变化。固定Seed、按全文哈希选择的512篇样本，含EOS的Token数由63,757降为54,601，减少14.3608%；488篇发生变化。超过8192字符的1篇不进入Token样本。这不是全库精确Token节省量，也不证明质量改善；没有把新策略应用到旧模型或BIN。

## 语义审查

最终规则标记379篇待审文档：361条展平指令格式提示、18条简单字面算式提示。200条队列保持 `REVIEW_PENDING`，全库事实核验为 `NOT_CHECKED`。明确错误与引用反例的对照见 [审查笔记](corpus-review-notes.md)；没有自动删除或生成新的训练集。

## 87M同条件预训练评测

两次评测都使用CPU FP32、单线程、L=512、65,536个有效Val目标Token、Greedy、最多16个新Token。数据、Tokenizer、24候选题和12个Raw提示、比较策略一致。未运行Chat，状态为 `NOT_RUN`。

| 指标 | 原初始化Checkpoint | 完成试训练的Checkpoint |
|---|---:|---:|
| Validation Loss | 6.7583106682 | 5.1211032318 |
| 知识候选题 | 2/4 | 3/4 |
| 逻辑候选题 | 1/4 | 4/4 |
| 数学候选题 | 1/4 | 2/4 |
| 基础英文候选题 | 0/4 | 2/4 |
| 中文语言候选题 | 0/4 | 0/4 |
| 基础代码候选题 | 0/4 | 0/4 |

候选题只是24条尚待独立人工复核的原创诊断，不合成为总分；存在词频、措辞及Token长度偏差，不能推断已具备聊天或开放推理能力。Raw生成平均重复4-gram比例两者均为0.1923，最终模型12条提示中有1条空输出，不能据此宣称生成质量全面改善。

新CPU Loss与历史T1200数值有约微量浮点差异；当前对照表在同一CPU口径内比较。原初始化文件保留旧任务160步/2,618,880 Token元数据，最终文件的611步/10,000,848 Token属于新的试训练任务，不能把两者的Step简单相减作为本轮训练长度。

在已知v3三个文本Split共128,000篇上检查整条Prompt/Prompt+答案精确重复及既有SimHash/Jaccard近重复规则，没有匹配；不覆盖子串、语义改写或未知上游语料，不能证明绝无污染。小规模题目不是新的训练优化目标。

## CPU Nano与保护检查

原评测工作流及新预训练评测均完成CPU Nano端到端运行。Nano仅用于功能验收，不能据其分数或资源用量推断87M能力。旧模型、训练状态、数据版本、固定160题/40提示及历史测量共15个受保护文件逐一SHA256复核，全部未变化。

本次没有执行新的87M训练、完整GPU训练负载或扩大Token预算。下一步应先完成语料审查及来源核对，再决定是否另立规范化实验；既有模型继续训练仍须保持旧输入策略与原Tokenizer。

## 回归结果

| 命令（Python使用项目 `.venv`） | 实际结果 |
|---|---|
| `python -m pytest -q -o addopts=''` | 402 passed，137.36秒；含16项新增测试 |
| `./scripts/smoke_test.sh` | PASS，CPU Nano、本地语料、临时输出 |
| `SYNTHETIC=1 ./scripts/smoke_test.sh` | PASS，CPU Nano、合成数据 |
| `python scripts/check_training_reliability.py` | 11项PASS；参数最大差异0，样本顺序/RNG/LR一致 |
| `python scripts/check_evaluation_workflow.py --out reports/quality-stage-v1/nano-workflow-delivery` | PASS |

新增测试覆盖保守规范化、Tokenizer拟合/编码一致性、策略传递与不兼容拒绝、审查队列确定性/损坏/不完整决定/来源统计上限、候选NLL的Shift及Mask、同分、特殊Token、阶段分离和跨阶段比较拒绝。最终源代码在这些回归通过后没有继续调整。
