# PR #7 CPU Nano 实测与自查

实现提交：`6ec1b77adf9d424578b88c31890095099fc3e401`。两次完整工作流均在该提交的干净工作区运行。此后的提交只补充本结果说明与独立JSON证据，不改实验代码或冻结资产。基线是已合并PR #6的 `aa39210`。

完整结果位于新目录 `reports/diagnostics/pr7-cpu-nano-final/`，复现目录为 `reports/diagnostics/pr7-cpu-nano-repeat/`。已提交的[结构化证据摘要](pr7-diagnostic-evidence.json)包含Manifest、训练正确性结果、完整记忆结果/曲线、分领域候选与生成摘要、全部结果文件哈希、历史文件保护与两次运行的对比。逐题Token ID/NLL/输出保留在对应本地完整结果；PR审查者可按[工作流文档](training-and-chinese-diagnostics.md)独立重现。

## 实际命令与结果

以下python命令使用项目 `.venv/bin/python`（或把 `.venv/bin` 加入PATH），无在线API测试。

| 命令 | 结果 |
|---|---|
| `python -m pytest -q -o addopts=''` | **459 passed in 145.99s**，含46项新增诊断测试 |
| `./scripts/smoke_test.sh` | PASS，读取现有语料的Nano路径，新临时输出目录 |
| `SYNTHETIC=1 ./scripts/smoke_test.sh` | PASS，纯合成语料路径 |
| `python scripts/check_training_reliability.py` | 原11项检查全部PASS；新增完整AdamW/RNG对比也PASS |
| `make diagnose-nano DIAGNOSTIC_OUT=reports/diagnostics/pr7-cpu-nano-final` | PASS |
| `make diagnose-nano DIAGNOSTIC_OUT=reports/diagnostics/pr7-cpu-nano-repeat` | PASS；两次轨迹/评分/生成精确一致 |
| `python scripts/run_memorization_probe.py --out reports/diagnostics/pr7-memorization-cli --max-steps 120 --max-effective-tokens 30000 --max-wall-seconds 60` | PASS，独立记忆入口 |
| `python scripts/evaluate_chinese_foundation.py --ckpt reports/diagnostics/pr7-resource-check/memorization/corpus/model.pt --tokenizer reports/diagnostics/pr7-resource-check/memorization/tokenizer.json --out reports/diagnostics/pr7-evaluate-cli --max-wall-seconds 60` | PASS，独立只读评测入口 |

全量测试覆盖原413项与新增46项。新增边界包括策略/实际BIN、Shift/Ignore/因果信息、非零参数梯度有限差分、训练/留出隔离、固定Seed、同词表异Tokenizer拒绝、候选参考答案不参与输入构造、BPE边界、未审查汇总、不可评分分母、重复/非法题目、解码重复/EOS/特殊Token、预算与墙钟、内存限制恢复、完整Manifest哈希和历史产物拒绝覆盖。

## CPU Nano实验

- 369,280参数，现有Nano结构，诊断词表512；CPU FP32、单线程、seed42、PyTorch `2.14.1+cu126`。
- 记忆预算：200次更新合计、最多50,000有效目标；实际200次更新、29,232有效目标。单批次/语料各100次更新。
- 全阶段墙钟上限120秒。CPU虚拟内存余量2048 MiB；本机起始虚拟空间5,137,252,352 bytes，RLIMIT_AS绝对上限7,284,736,000 bytes，**不是RSS上限**。两项训练测得峰值RSS分别795.7/806.0 MiB。
- 训练160条，留出32条；Tokenizer只拟合训练文本。续写训练子集在完整冻结行序上等距取16条，覆盖类别，留出取8条。

| 观察 | 初始 | 最终 |
|---|---:|---:|
| 固定单批次Loss | 6.390465 | 0.002002 |
| 固定语料打包Loss | 6.312129 | 1.128549 |
| 单批次原文teacher-forced准确率 | — | 7/7 |
| 全训练样本teacher-forced准确率 | — | 1150/1749（65.75%） |
| 训练前缀续写Token准确率 | — | 53/238（22.27%） |
| 训练前缀Exact（包括EOS） | — | 8/16 |
| 独立留出teacher-forced准确率 | — | 32/1520（2.11%） |
| 独立留出前缀Exact | — | 0/8 |

单批次Loss下降**99.9687%**，超过事先声明的50%诊断门槛；语料Loss下降82.12%。梯度/参数/AdamW状态有限，参数更新成立，无非法Token。teacher forcing与自回归续写差异明显，不能用前者替代后者。留出文本只有单一新模板，不是有代表性的泛化测试。这些数据只支持 **LEARNING_DEMONSTRATED**，不支持“学会了世界知识”。

完整Loss曲线为 `single-loss.csv`、`corpus-loss.csv`，逐步loss/LR/有效Token也保留在JSON。两次运行的曲线、样本输出、候选NLL与全部解码输出精确一致；参数和AdamW最大差均为0，RNG精确相等。不同运行的UUID、路径、Checkpoint时间戳不要求相同。

## 训练正确性

两种输入策略的Tokenizer回环、生产文档编码/BIN/打包、特殊Token监督语义与Checkpoint身份检查通过。CPU有限差分使用实际Nano嵌入参数3个非零梯度，epsilon0.005，atol0.0002/rtol0.02，最大绝对差0.00016245。改变未来输入的前缀logits最大差0。

固定7步连续训练，对照在第1/2步中断后恢复到7步：参数最大差0，完整AdamW最大差0，RNG/LR/样本顺序一致；双方41样本、451有效目标。保留原恢复/目录所有权/配置与数据不兼容/损坏主备等检查。

没有发现新的训练正确性P0/P1问题，没有修改训练核心。已有打包窗口仅监督L−1个目标、跨EOS仍可注意的语义已明确记录。开发时，小样本不够拟合1024词表而被原Tokenizer严格检查拒绝（仅产744）；最终诊断使用既有Nano512词表，没有放宽检查，也没有重训正式24576词Tokenizer。

## 中文候选与生成

120题全部 `REVIEW_PENDING`，因此状态为 **EVALUATION_PROVISIONAL**，正式分母为0。下面仅验证小样本Nano评测链路，不能推断真实预训练87M能力。候选sum NLL有长度/频率/措辞偏差，答案位置平衡不能消除这些偏差。

| 领域 | separate正确/可评分 | Wilson 95% |
|---|---:|---|
| 中文词语/语法 | 9/30 | 16.66%–47.88% |
| 基础世界知识 | 12/40 | 18.07%–45.43% |
| 日常常识 | 5/20 | 11.19%–46.87% |
| 数学/逻辑 | 5/20 | 11.19%–46.87% |
| 英文/代码 | 4/10 | 16.82%–68.73% |

`joint_shared_prefix` 对照分开保留，不与主口径混合排名或解释为能力提升。不可评分项按协议独立排除分母；本次主口径无不可评分项。

30条Raw前缀、最多24新Token、EOS停止、cache开启、seed+固定题号：

| 模式 | 空输出 | 提前EOS（内容<4Token） | 平均生成长度 | 平均Token四元重复率 |
|---|---:|---:|---:|---:|
| Greedy（T0,k0） | 11/30 | 13/30 | 4.30 | 0.030159 |
| Temperature（T0.8,k0） | 7/30 | 16/30 | 4.33 | 0.014286 |
| Top-k（T0.8,k20） | 10/30 | 18/30 | 3.77 | 0.014286 |

三种模式均无异常特殊Token；长度/重复分布、Distinct-2、周期循环、逐题文本与Token保留。输出有明显提前停止/空输出现象，尚待人工可读性复核。三种模式没有合成一个能力总分。

## 历史保护与PR8

对开发前333个已有文件做SHA256比对，仅README、`.gitignore`与可靠性检查脚本发生预期修改；其中**225个受保护历史文件哈希全部未变**，包括模型、Tokenizer、训练数据、PR6冻结评测资产及已有证据。新结果只写全新目录，不覆盖任何历史报告。

未执行87M训练或正式v4 Tokenizer训练：本任务要求实际执行的是CPU Nano，87M必须另行明确选择、提供现有Tokenizer与显存预算。本PR既不包含1亿Token训练，也不以随机初始化87M分数推断能力。

结论为 **ENGINEERING_PASS / LEARNING_DEMONSTRATED / EVALUATION_PROVISIONAL**。尚不满足进入正式PR8训练的条件；可以据此开展PR8准备，但需要完成题库与Raw人工审查，冻结v4语料/独立留出/Tokenizer候选方案与污染审计，并在目标硬件显式验证有预算的87M负载与恢复。评测范围窄、候选偏差、留出模板单一、跨硬件数值差异、人工审查真实性无法由本地签名链保证，都是保留限制。

本PR修改范围：六个新增诊断模块、三个CLI、四个测试文件、两组冻结资产、Makefile/README/忽略规则、可靠性脚本增强，以及工作流文档、本实测说明和证据摘要。训练/评测历史模块、权重与原资产均未修改。
