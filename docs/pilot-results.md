# MiniLLM-87M：1000万有效Token试训练验收

结论：`ENGINEERING_PASS`、`CONVERGENCE_IMPROVING`、`CAPABILITY_INCONCLUSIVE`。语言建模指标改善；确定性能力题仍未答对，生成仍有重复与空输出，不能视为可用聊天模型。

## 任务与进度

使用原预训练模型权重与原24,576词表，启动新的weights-only任务，优化器与本轮计数重新开始。原模型的160步/2,618,880 Token是历史计数，不能与本轮计数相减来计算训练量。未使用旧格式state.pt恢复。

| 项目 | 实际结果 |
|---|---|
| 本轮训练 | 611次更新 / 10,000,848个有效Loss Token |
| 数据进度 | Epoch 0，Micro-batch游标9776，样本9776 |
| 停止原因 | target_reached；目标1000万按完整更新取整，实际多848 Token |
| 配置 | 87M，FP32，Seq1024，Batch1，Accum16，checkpointing，Seed42，线程1 |
| 模型参数 | 86,528,640，架构和Tokenizer未改动 |
| fit报告耗时 | 6,854.633秒，约1小时54分15秒 |
| 含fit开销的平均吞吐 | 1458.99有效Token/s |
| 训练峰值 | allocated 1813.2 MiB；RSS 2145.1 MiB（原日志字段名为mb） |

fit耗时包含本次fit内训练、验证与保存，不包括数据准备、预探测、恢复对照和后续评测。日志间隔估算为6,801.316秒，少于fit总计6,854.633秒；不要将间隔估算当作整个工作流的精确时长。

## 固定条件下的评测

以下三项均为Seq512、Batch1、65,536个有效Loss Token，使用固定配置、种子和原Tokenizer。基线与最终Validation的评测策略已逐字段核对一致。Test是最终独立观察，没有用于训练优化或选取超参数。

| 指标 | 基线Validation | 最终Validation | 最终Test |
|---|---:|---:|---:|
| Loss | 6.758310618 | 5.121105591 | 5.084662708 |
| PPL | 861.186094 | 167.520476 | 161.525449 |

Test本次只运行语言建模指标，没有在Test报告中再次运行160题Benchmark；空题目列表不是0分成绩。

训练期间另用Seq1024、完整330,429个有效Validation Token。该口径的结果如下，不与上表直接混用。

| Global Step | 累计有效训练Token | 全量Validation Loss |
|---:|---:|---:|
| 100 | 1,636,800 | 6.100586585 |
| 200 | 3,273,600 | 5.714374951 |
| 300 | 4,910,400 | 5.459869808 |
| 400 | 6,547,200 | 5.276395668 |
| 500 | 8,184,000 | 5.163978257 |
| 600 | 9,820,800 | 5.105035829 |
| 611 | 10,000,848 | 5.100000655 |

最终全量Validation PPL为164.022015。训练分析`OK`，无errors/warnings，趋势`VALIDATION_DECREASING`；没有检出非有限数值、恢复异常或进度缺失。

## 能力与生成观察

| 模式 | 基线 | 最终 |
|---|---:|---:|
| Raw Completion | 0/64 | 0/64 |
| Chat Instruction | 0/96 | 0/96 |

原160题仍保持固定版本、答案和哈希，未为提高评分修改题集。其人工复核状态仍为pending；这是项目内部观察题集，不是已验证的通用能力排行榜。预训练模型的Chat低分也不能单独解释为没有相关知识。

40条固定Prompt的空输出从17降为3，提前结束从18降为9，特殊Token异常均为0。全部200次生成（160题+40固定Prompt）的平均Token四元组重复比例从0.5596降为0.4461；它只是退化诊断，不能作为质量分数。仍存在循环重复和不相关内容。

下面按固定题集顺序展示Raw前两条、Chat前两条，未挑选最佳样例。展示生成内容，完整Token ID、停止原因和计时保存在本地predictions.jsonl。

```json
{
  "id": "fixed-raw-001",
  "mode": "raw",
  "prompt": "我叫小林，今天想介绍一下自己。",
  "baseline": "",
  "final": "",
  "final_stop_reason": "eos"
}
```

```json
{
  "id": "fixed-raw-002",
  "mode": "raw",
  "prompt": "人工智能是一种",
  "baseline": "，如“2. \n2. \n2. \n2. \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
  "final": "，例如“狗”和“猫”等。因此，“狗”和“狗”和“猫”等。",
  "final_stop_reason": "eos"
}
```

```json
{
  "id": "fixed-chat-001",
  "mode": "chat",
  "prompt": "请用两句话介绍你自己。",
  "baseline": "",
  "final": "它，\n1. 夏天\n2. 夏天\n3. 夏天\n4. 夏天\n5. 夏天\n5. 夏天\n5. 秋天\n5. 秋天\n5. 秋天\n5. 秋天\n5. 秋天\n5. 秋天",
  "final_stop_reason": "max_new_tokens"
}
```

```json
{
  "id": "fixed-chat-002",
  "mode": "chat",
  "prompt": "简单解释人工智能。",
  "baseline": "",
  "final": "人工智能，人工智能是人工智能，人工智能是人工智能，人工智能，人工智能，人工智能，人工智能，人工智能，人工智能，人工智能，人工智能，人工智能，人工智能，人工智能，人工智能，",
  "final_stop_reason": "max_new_tokens"
}
```

## 工程完整性与数据身份

- 主状态611步、备份600步均通过进度校验；所有AdamW更新计数与相应Step一致，两个状态的动量均有限。
- `model.pt`、`best.pt`和最终`state.pt`权重逐Tensor完全一致；best.pt对应611步，best_loss等于最终全量Validation Loss。
- 最终模型、最终评测及Test模型SHA一致，源预训练文件SHA保持不变；初始化谱系和训练Manifest一致。
- 真实87M的20步连续/两次中断恢复，样本顺序、LR和进度一致，参数最大差0；恢复实验Token不计入本轮训练。
- 训练后再次全量重编码并验证所有数据产物、原始输入、Tokenizer及历史源哈希，审计通过。
- 数据v3有效Train Token为15,965,961；内部划分及新旧数据九组精确重叠均为0。历史排除另识别并剔除了2篇近重复。
- 固定题集在已知新预训练文本及3,000条旧SFT文本中未检出指定完整记录/近重复命中；不宣称穷尽子串、语义改写或未知来源。

最终模型SHA256：`6046ebc61b9196c554758b5982f44da00b4a7e4e6daebfa6100af6a1dea9fec9`。数据Content ID：`6e4d04c74bf35f07c43f752ebb9deee894fe3414fb38ebee2b1c7816cb8d4a00`。

v1曾因历史暴露而停止（65步/1,063,920 Token），其结果排除；v2有效Train Token不足1500万，未进入试训练。两者产物均保留，v3未继承v1权重。原始语料仅按用户确认用于非商业研究，许可为上游组合声明，缺少逐条来源许可映射。

## 负载与回归验证

实际T1200/87M/Seq1024的3次完整AdamW预探测：allocated峰值1,902,630,912 bytes、reserved峰值2,239,758,336 bytes，均低于80%预算3,116,105,728 bytes。Seq1024通过，512/256回退未执行。这是预探测峰值，不能冒充完整训练的reserved峰值，也不保证未来永不OOM。

| 命令 | 实际结果 |
|---|---|
| `.venv/bin/python -m pytest -q -o addopts=''` | 352 passed in99.08s |
| `./scripts/smoke_test.sh` | PASS |
| `SYNTHETIC=1 ./scripts/smoke_test.sh` | PASS |
| `.venv/bin/python scripts/check_training_reliability.py` | 11项PASS |
| `.venv/bin/python scripts/check_evaluation_workflow.py --out /tmp/minillm-pr5-final-evaluation-workflow` | PASS |
| 实机显存、20步恢复、完整试训练、最终评测/曲线/对比/验收/Test | 全部完成；工程PASS |

上述回归针对本分支实现运行；其后仅补充文档和证据。现有87M参数量和INT4预算测试通过，没有修改量化格式。

## 本地产物与下一阶段

模型及完整状态：`runs/pilot-10m-t1200-v3/`。详细报告：`reports/pr5-pilot-v3/`，其中baseline、final-evaluation、test-evaluation、analysis、comparison、acceptance分别保存对应结果，curves为CSV。大文件和报告目录未提交到Git，紧凑指标见[pilot-evidence.json](pilot-evidence.json)，复现步骤见[pilot-runbook.md](pilot-runbook.md)。

下一阶段先人工复核固定题与退化样例，检查语料质量并设计许可清楚的多来源配比，再制定独立、固定的后续预算及验收计划。现有候选是连续区间样本，英文Token约0.02%，不代表平衡混合语料。当前resume禁止静默改变训练步数/LR计划；延长训练必须明确设计extend或另开weights-only任务。没有自动启动更大预算，也没有自动合并PR。
