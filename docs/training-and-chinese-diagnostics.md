# PR #7：训练正确性与中文诊断基线

基线为已合并 PR #6 的 `aa39210`。本 PR 不训练正式 v4、不重训正式 Tokenizer、不改结构、Loss、AdamW、Scheduler、梯度累积、游标或恢复语义。默认只在 CPU 上使用 Nano；实验结果不是预训练 87M 能力测量。

## 一条命令运行

```bash
make diagnose-nano
# 或指定新目录：已存在目录（即使为空）也拒绝复用
python scripts/diagnose_training.py \
  --out reports/diagnostics/my-nano-run --seed 42 \
  --max-steps 200 --max-effective-tokens 50000 \
  --max-wall-seconds 120 --max-memory-mib 2048
```

使用仓库 `.venv/bin/python`，或先激活虚拟环境。无需联网、下载语料、GPU、绘图库或 Web 前端。Python 3.10+，Linux/POSIX 环境；墙钟 watchdog 使用 SIGALRM，内存限制使用 Linux `/proc` 与 RLIMIT_AS。异常退出返回非零并写 `failure.json`，不发布成功结论。预算不足但数值正常时返回 `LEARNING_INCONCLUSIVE`，仍保留诊断报告；未训练到语料 Checkpoint 时评测标记 NOT_RUN。

`--max-steps` 和 `--max-effective-tokens` 是**两项记忆实验合计**的训练预算。步骤大致对半分配；Token 用每次更新的最坏批次大小保守预留，只在完整更新后计数。正常训练使用原 `fit()`，没有新训练循环或为迎合门槛改 Loss。Nano 默认 batch=1/4、sequence=64、FP32、线程=1、诊断 Tokenizer=512，允许显式 Tokenizer 时词表上限4096。全部阶段（包括数据准备、数值检查、恢复、候选和生成）共用墙钟 watchdog。它可在更新中中断；原有原子 Checkpoint 只保存已完成更新，不把部分更新当成恢复点。

数值检查和恢复是**另外的固定 CPU Nano 开销**，不计入记忆预算：一次小批次反向传播/AdamW更新、有限差分、7步连续轨迹与两次恢复及1步备份恢复（合计15次恢复检查更新，最多15×88个有效目标），均受同一墙钟与内存上限约束。不能把这些Token记入v4训练量。

CPU `--max-memory-mib` 是**相对已导入 PyTorch 的虚拟地址空间新增余量**，默认2048 MiB，上限8192 MiB；通过 RLIMIT_AS 约束，恢复子进程继承绝对上限。它不是 RSS 上限。Manifest记录起始虚拟空间和绝对限制，训练 summary 另记峰值 RSS。步数硬上限1000，记忆有效Token上限200万，墙钟上限600秒，生成每项最多64个新Token、最多150条提示，中文候选最多150题。这些限制都不能用于启动1亿Token训练。

## 检查的含义

- Tokenizer：两种现有输入策略，中文段落、标点/数字/英文、数学、Python/C++、全角字符、明确代码类别和特殊Token回环。预期值是输入策略处理后的文本。使用生产 `tokenize_split()` 编码，逐Token对比文档、EOS与实际BIN，再检查 `PackedPretrainDataset` 窗口。
- Loss：手算交叉熵求和与因果Shift、有效目标数量、右侧Padding/Ignore Index、首标签不参与Loss、被Ignore目标的logit梯度为0、未来后缀改变不能影响此前logits。对实际Nano嵌入参数的3个非零梯度分量做中央有限差分，并验证有限梯度、有限参数及AdamW状态、参数更新。
- 特殊Token：文档EOS、助手end_of_turn是合法监督；BOS、PAD与Chat角色标记在相应模板/批处理位置被Ignore。字面特殊Token的无损回环用 `skip_special_tokens=False`。诊断训练短文不包含保留标记，不把它们伪装成普通文本。
- 身份：复用严格 `load_evaluation_model()`，即使词表大小相同，Tokenizer指纹不同也拒绝。非identity策略写入Checkpoint，并核对继承策略；已有冲突拒绝逻辑有回归覆盖。
- 恢复：调用原 `check_training_reliability.py`，沿用PR #5 `compare_tree()`。检查参数、完整AdamW状态与param groups、精确RNG、LR序列、样本消费顺序、Token与游标计数；使用Dropout与真实SIGINT/SIGTERM中断。原故障注入检查保留。`recovery/` 下最终损坏文件是显式故障注入产物，不应当成可继续训练的Checkpoint。

现有打包语义是连续文档+EOS，注意力可以跨文档；每个L长度窗口内部Shift得到L−1个目标，跨窗口的下一Token目标不计Loss。这是已有行为，诊断予以固定和说明，本PR没有修改它。

## 记忆与独立对照

```bash
python scripts/run_memorization_probe.py --out reports/diagnostics/memory-only
```

冻结资产 `data/diagnostics/memorization-v1/`：160条训练短文（80中文叙述、32基础事实、16数字/数学、16中英混合、16代码/结构化文本），另有32条独立留出文本。资产为原创CC0短文；不来自外部训练/考试数据集。文件SHA256和条数写入版本声明，加载前核验。ID与全文去重跨Split检查。Nano Tokenizer仅拟合训练短文，留出文本不参与拟合或模型更新。

单批次实验重复同一64Token样本；语料实验反复训练固定160条短文，复用生产打包和 `fit()`。两项从同一初始化独立开始。记录初末Loss、逐步曲线/LR/Token计数、有限梯度与参数、更新和AdamW状态、所有训练/留出样本的teacher-forced Token准确率。续写使用固定的、在冻结行顺序上等距抽取的16个训练样本和8个留出样本，实际ID保存在JSON，不按输出好坏挑选。

续写采用实际文档联合编码与前缀编码共有的Token前缀；BPE重新切分的边界尾部与EOS纳入目标，保存前缀/目标/输出ID。生成采用greedy，最多64Token。额外或缺失输出Token都计入准确率分母；Exact要求连EOS完全一致。完整文档的teacher-forced准确率与打包窗口Loss是不同观测口径。

单批次Loss下降至少50%只是固定CPU Nano诊断门槛。未达到不改门槛、不篡改样本或评分，输出有限性、实际预算与Loss变化用于定位。`LEARNING_DEMONSTRATED`仅表示这套参数能学习这批短文。留出文本使用单一未见叙述模板，既不代表中文通用泛化，也不代表世界知识；训练集Exact尤其不能支持这种结论。

显式87M探测示例（本PR未执行）：

```bash
python scripts/run_memorization_probe.py --model 87m --enable-87m \
  --device cuda --tokenizer /path/to/existing/tokenizer.json \
  --gpu-memory-budget-mib 3072 --max-steps 20 \
  --max-effective-tokens 4000 --max-wall-seconds 60 \
  --seed 42 --out reports/diagnostics/explicit-87m-probe
```

必须提供现有24576词Tokenizer，不允许自动训练正式Tokenizer；87M探测只支持显式CUDA。检查参数/优化器保守预算（20 bytes/parameter+768 MiB）不超过指定预算和实际空闲显存，再设置CUDA进程allocator配额。估算不保证所有硬件上成功；OOM记录失败。模型从零初始化，结果必须与Nano分开解释；没有强制87M Exact通过率，也不能推断已有预训练87M能力。

## 中文评测与人工审查

`data/evaluation/chinese-foundation-v1/`冻结120道原创四选一（30中文词语语法、40稳定基础知识、20日常常识、20数学逻辑、10英文代码）与30条Raw自然前缀（故事、说明、科普、日常、基础知识、逻辑各5条）。答案位置各30道，尽量使用同类、相近长度候选；逐题字长审计与实际Token长度均保留。长度、频率、表述偏差仍可能存在。所有题目的初始状态均为 `REVIEW_PENDING`，稳定事实也需要人工验证，不因自动生成而被自动批准。

```bash
make evaluate-chinese-foundation CKPT=/path/model.pt TOKENIZER=/path/tokenizer.json
# 同时提交独立人工决定文件：
python scripts/evaluate_chinese_foundation.py \
  --ckpt /path/model.pt --tokenizer /path/tokenizer.json \
  --reviews /path/human-decisions.jsonl --seed 42 \
  --out reports/diagnostics/reviewed-run --max-wall-seconds 120
```

可以直接运行逐题终端审核，无需手动编辑JSONL：

```bash
.venv/bin/python scripts/review_chinese_foundation.py
# 也可指定审核人及结果文件；重复运行同一命令会继续未审核题目：
.venv/bin/python scripts/review_chinese_foundation.py \
  --reviewer 你的名字 --out reports/reviews/chinese-foundation-v1.jsonl
```

脚本逐题显示题目、四个候选、参考答案和来源。输入 `1` 回车通过，`2` 回车拒绝（可填写原因），`s` 跳过，`q` 保存退出；直接回车不会通过。每题决定立即原子保存，Ctrl+C/输入结束后可按原命令继续。已审核条目自动跳过，保留原审核人和决定；跳过项保持待审，下次重新显示。结果直接传给评测器 `--reviews reports/reviews/chinese-foundation-v1.jsonl`。脚本默认只审核120道候选题，不自动批准30条Raw前缀；所有选择均需由人工输入。题库版本/哈希不符、冻结目录输出或并发写同一结果文件时拒绝。

也可复制 `review-template.jsonl` 到新路径编辑，**不要修改冻结文件**。人工逐项检查题意、唯一标准答案、事实/逻辑、候选偏差与文本类别：

- `ACCEPT`：reviewer、reason非空，`human_review=true`，题号/版本/对应资产SHA256严格匹配。
- `REJECT`：排除题目，并保留人工理由。
- `REVISE`：当前版本排除；修改题目应建立新冻结版本并重新审查，不在原文件上修答案。
- `REVIEW_PENDING`：允许探索性运行，不进入正式分母。

重复/未知ID、哈希/版本不匹配、缺失人工声明均拒绝。这个机制记录人工责任，不是身份认证系统，不能验证reviewer是否真正独立。全部候选完成ACCEPT/REJECT且至少一题ACCEPT后，候选状态才是 `EVALUATION_REVIEWED`；部分审查仍为 `EVALUATION_PROVISIONAL`。只对ACCEPT子集给正式计数；Pending仅出现在独立探索性栏。Raw Prompt的审查决定不替代生成文本的人工可读性审查，报告另生成 `generation-review.jsonl`。

候选评分直接调用PR #6 `candidate_prediction()`，主口径固定 `separate` + sum NLL，保留 `joint_shared_prefix` 的逐题对照。Prompt不拼候选列表或参考答案；teacher forcing仅将每个候选作为待评分续接输入，参考答案只用于评分后比较。记录全部Token ID、NLL、选择、平局、BPE边界和不可评分理由。Joint尾部可能包含重新分词的Prompt文字，不能将两种口径择优或混合。

分领域输出正确数/可评分分母、不可评分数、平局和Wilson 95%区间。区间只是题目采样不确定性的近似，不包含题目设计偏差。Manifest固定题库/Review/Tokenizer/输入策略/模型/解码参数及实现哈希；`require_comparable()`拒绝跨协议直接计算能力增量。本次Nano候选评测是评测链路验证，不能替代87M实测。

生成直接复用 `model.generate()`：greedy(T=0,k=0)、temperature(T=0.8,k=0)、top-k(T=0.8,k=20)，每题最多24Token、KV cache、EOS停止、独立局部Generator，seed+题目序号。分模式报告空输出、EOS/提前EOS（内容<4Token）、触及长度上限、异常特殊Token、Token四元重复、Distinct-2、最长连续重复、可观察周期循环，含长度/重复分布。内容不足2Token时Distinct-2为null，不能当作丰富度0或1。周期标记不是语义“无意义”的自动判定；开放续写不做字符串语义打分。

## 产物与下一阶段

新目录包含 `diagnostic-manifest.json`、`training-diagnostics.json`、`memorization-results.json`、`candidate-results.json`、`generation-results.json`、`diagnostic-report.md`、曲线CSV、可读性审查表与PR8条件表。单独记忆/评测入口仅生成对应结果，其他维度记NOT_RUN。Manifest记录Git commit/dirty状态、逐实现文件哈希、数据/版本/Review/Checkpoint哈希、模型配置/参数量、Tokenizer指纹、seed、硬件/PyTorch、预算和递归产物哈希。没有一个合并总分。运行中断的目录留作失败证据，不能被新运行覆盖。

结论使用独立的 `ENGINEERING_PASS/FAIL`、`LEARNING_DEMONSTRATED/INCONCLUSIVE`、`EVALUATION_PROVISIONAL/REVIEWED`。PR8准入评估与PR7诊断完成状态独立，`scope=PR8_INFORMATIONAL_ONLY`，不参与诊断命令的成功/失败退出判断。尤其是v4语料/Tokenizer方案准备属于PR8工作，不能用来阻止PR7完成。

`pr8-readiness.json`（及Manifest内的同一对象）的 `schema_version=2` 使用显式三态条件：`PASS`、`FAIL`、`NOT_ASSESSED`。未执行的训练/评测为NOT_ASSESSED；尚未完成的候选人工审查是已知未满足的条件。当前工具没有采集Raw人工可读性验收及v4语料/Tokenizer方案审查证据，因此这两项明确为NOT_ASSESSED，不假定通过或宣称审查失败。记录单独的 `failed_criteria` 和 `unassessed_criteria` 便于后续阶段接收证据。

整体 `status` 为：存在明确未满足条件时NOT_READY，`ready_for_pr8=false`；没有明确未满足条件但仍有未评估项时NOT_ASSESSED，`ready_for_pr8=null`；只有全部条件明确PASS才为READY/true。消费报告时必须读取显式状态，不能对非空状态字符串执行 `all()` 来判断通过。此版本未新增人工验收证据输入入口，不能通过重跑诊断自动认证外部人工决定；相关验收应由后续独立、可追溯的证据流程处理。旧版报告和证据不回写。

进入正式v4训练前还需要：独立完成题库与Raw人工审查；确定新语料来源/中文比例/质量与污染排查、可靠留出集；另立Tokenizer候选与压缩率/边界对比；冻结数据与Tokenizer身份；在目标硬件显式运行有预算的87M负载/恢复检查；审查时间、Token和存储预算，再决定PR8范围。不要通过增加训练量掩盖数值/恢复问题。

回归命令与本次实测结果见 `docs/pr7-diagnostic-results.md`。历史Checkpoint、Tokenizer、训练数据、PR6冻结评测资产和既有证据均不应被本工作流写入。
