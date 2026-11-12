# PR #6：类别贯通与候选BPE边界修复

本次为PR #6审查后的代码修复。新证据见 [pr6-category-boundary-evidence.json](pr6-category-boundary-evidence.json)。原 [quality-stage-evidence.json](quality-stage-evidence.json)、历史诊断、模型权重、训练状态及数据未改；未执行87M训练或重新生成语料。

## P1：显式代码类别

问题是清洗使用了 `category=code`，而Tokenizer接口及调用链丢掉了类别，运行时默认auto会把 `Ａ＝１` 变成 `A=1`。

修复后的契约：

- `prepare_text`、`encode`、`encode_document` 接受 `category`；批量编码接受与文本对齐的 `categories`。
- 清洗后解析类别，再执行策略；保留这个类别，不因全角转换后的新正则特征重新改变标签。HTML实体先清理，避免拟合后才出现需要转换的字符。
- Pipeline将已记录的类别传给编码，绑定数据版本的文本策略，并在打包前验证文本已规范化；拟合文本与编码文本不一致时拒绝发布。
- SFT记录可声明默认类别，消息可以覆盖（包括system）；生成CLI增加 `--text-category`，Web API和历史保留标签，评测任务可声明独立的 `text_category`。
- 未标注文本仍走auto。类别不是词表属性，不能丢掉显式代码标签后声称任意代码都能被正则重新识别；未知类别会明确拒绝。

回归使用 `Ａ＝１`、`［１，２］`、`变量＝＂全角值＂`、单个全角标识符 `Ａ`，确认拟合文本、BIN Token、显式代码运行时编码一致；同时覆盖混合散文、HTML实体、SFT监督标签、system消息、Raw生成与Web历史。原词表JSON和特殊Token ID不变。

## P2：边界审计及独立对照

默认 `separate` 仍分别编码Prompt和候选，只给续接Token计分；不再假定它等于全文BPE编码。每题输出两者Token ID、联合偏移、是否一致，汇总受影响题/候选数与ID。

新增独立的 `joint_shared_prefix` 诊断：全文联合编码，以独立Prompt和全部联合候选共有的Token前缀为条件，对其后的尾段计分。该尾段可能包含重分词的Prompt字符；报告其数量。它不是严格的“答案专属NLL”，也不是对原分数的唯一正确替代。选项集合及总和/均值归一化都会影响口径，因此两种指标分开展示。

若没有共同上下文，或联合分词超出预算、选项坍缩、拼接改变了规范化结果，对照返回 `NOT_SCORABLE_*`；不补BOS、不伪造零分。选择主口径用 `--candidate-boundary-policy`，写入Manifest，跨口径直接比较被拒绝。

## 真实只读对照

在原24题和原24,576词表上，两份87M权重均发现 **1/24题、1/96候选** 分词边界不一致：`pretrain-016`，Prompt为“小明比小红高，小红比小李高，三人中最高的是”，受影响选项A为“一样”。

| 模型 | 默认separate选择 | joint_shared_prefix选择 | 逻辑维度：separate / joint |
|---|---|---|---|
| 试训练前初始化权重 | A（一样，错误） | B（小明，正确） | 1/4 / 2/4 |
| 已完成试训练的权重 | B（小明，正确） | B（小明，正确） | 4/4 / 4/4 |

独立Prompt有18个Token，共同上下文有16个；联合口径的尾段包含Prompt末尾3个字符。该例说明评分口径确实会影响选择，不能将这种变化解释为模型新学会了知识。

新评测为CPU FP32、单线程、候选题专用，只读加载原权重。默认separate的所有候选Token ID、所选答案与旧诊断逐一相同，**NLL最大差异为0**。其他维度的两种口径结果相同。新报告位于 `reports/pr6-category-boundary/{baseline-candidates,final-candidates,comparison}/`；没有重算LM Loss或重新执行语料污染审计，不冒用旧检查覆盖范围。

## 验证

使用项目 `.venv/bin/python`：

- `python -m pytest -q -o addopts=''`：**413 passed in 132.43s**，比上轮增加11项。
- `./scripts/smoke_test.sh`：PASS。
- `SYNTHETIC=1 ./scripts/smoke_test.sh`：PASS。
- `python scripts/check_training_reliability.py`：11项PASS。
- `python scripts/check_evaluation_workflow.py --out reports/pr6-category-boundary/nano-workflow`：PASS。
- 两份原87M权重的候选只读评测及同口径比较：PASS。

冻结权重、数据、原固定题库及上轮诊断共20个文件SHA256核对无变化。实际模型参数量、INT4存储预算和训练恢复数值逻辑未修改。新测试用实际可跨边界合并的微型ByteLevel BPE与CPU Nano验证Shift/Mask、均值归一化、无共享上下文、规范化上下文变化、类别传递和不兼容比较。
