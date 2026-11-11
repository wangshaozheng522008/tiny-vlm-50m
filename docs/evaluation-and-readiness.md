# MiniLLM 评测与长训练验收

本工具只读 Checkpoint、数据与训练目录，将结果写入独立报告目录；不会启动正式训练、改变学习率、提前停止或选择 Test Loss 最优模型。新报告目录必须不存在或为空。结果先在同目录临时目录完成、同步并原子发布，失败清理半成品，不覆盖旧结果。

## 统一评测

```bash
python scripts/evaluate_model.py \
  --ckpt runs/minillm/model.pt --tokenizer runs/minillm/tokenizer.json \
  --dataset data/processed/minillm-pretrain-v1 --split val \
  --config configs/evaluation.example.json --out reports/baseline
```

也可显式选择 `--split test` 做阶段最终评估，或用 `--bin path/to/heldout.bin` 做自定义评估。后者只证明文件 SHA/Token 范围，不能独立证明原 Tokenizer 与数据来源，报告会注明 `custom`。不同时指定 Dataset 与 BIN。省略两者时只运行能力题和固定生成。

配置支持 `seq_len/batch_size/token_budget/max_new_tokens/seed/temperature/top_k/system/threads`；除 system 外也可用 CLI 同名横线参数覆盖。默认 Greedy，Token 预算 0 表示全部有效 Loss Token。`--benchmark '' --prompts ''` 可显式关闭对应生成集合；也可提供自定义 JSONL。`--sft-data ...` 加入已知 SFT 数据污染扫描。数据和模型均须来自可信的本地实验。

输出固定为 `evaluation.json`、`evaluation.md`、`predictions.jsonl`、`evaluation-manifest.json`。Manifest 绑定 Checkpoint SHA、完整 Model Config、Tokenizer 指纹、数据 SHA/Content ID、题集/生成提示/答案哈希、评分实现与生成配置，并记录环境和产物校验和。缺失或不兼容的 Checkpoint 指纹、架构/版本、词表或特殊 ID 会拒绝评测。

INT4/INT8 沿用已有反量化实现，加载为 FP32 后评测；这不是原生低位算子，速度和内存不能当作低位部署基准。预训练与 SFT 使用相同 Loss 实现，但阶段、Raw/Chat 分别报告。旧 `minillm.evaluate` 作为历史入口保留；其旧格式结果不参与本系统的等价比较。

## Loss、PPL 与固定窗口

使用 `model.eval()`、`torch.inference_mode()` 和现有 `loss_sum/num_loss_tokens`，标签与输入对齐，只在模型内部 Shift。Padding mask、ignore_index 与无有效前驱的位置不计入 Loss；各批次按实际有效 Token 求和。预算最后一批只屏蔽多余目标，不缩短或移动窗口。

保持 PR #2 的 Packed 口径：`N=floor((文件Token数-1)/L)` 个窗口，每窗 L 个输入、L-1 个目标，窗口之间的跨界预测与尾部未进入窗口的 Token 不计分。读取所需的最后一个 look-ahead Token 也不作为额外目标。报告记录完整窗口、尾部 Token、实际计分 Token；不自动调整 L。它与连续滑窗 PPL 的数字不可直接混用。

PPL 为 `exp(总Loss/有效Token)`，不截断真实 Loss。指数溢出时 JSON 写 `null` 与 `overflow_unrepresentable`，展示与实际计算不会混淆。空文件、无完整窗口、无有效目标和非有限 Loss 均失败。

## 固定题集与生成观察

`data/evaluation/minillm-zh-benchmark.jsonl` v1 共 160 条：语言 30、常识 25、数学 25、逻辑 25、指令 25、英文 15、代码 15；Raw 64、Chat 96。另有 `fixed-prompts.jsonl` 40 条，Raw/Chat 各 20。`benchmark-version.json` 固定文件 SHA、数量、来源和许可。

题目为本仓库原创、可逐条人工检查，未宣称经过独立人工验证或具备大型标准考试的效度。没有下载外部题库或调用大模型裁判；建议人工审阅并在后续题集版本记录审核结论。v1 固定后，修改题目/答案需新版本，加载器会拒绝与版本文件不符的哈希。

目录存在版本声明即启用严格冻结：必须为 schema_version=1、非空 version 和合法 files 登记表，当前文件名必须登记，SHA256 与题数必须匹配。重命名为未登记文件、无效声明或悬空版本文件链接均拒绝；不会自动改写声明。Benchmark 与 Fixed Prompts 的版本分别核验、绑定到评测身份。无版本声明的显式自定义资产允许加载，但标为 CUSTOM_UNFROZEN，以实际文件 SHA256 为身份，不能冒充冻结版本。

- Exact Match 比较完整原始解码答案，不做隐式 strip。
- Normalized Match 仅执行题目声明的 strip、空白折叠、casefold、末尾标点处理；不任意删除内部字符。
- Numeric Match 要求整段回答是有限十进制数，使用 Decimal 比较；拒绝单位、解释、NaN/Inf，支持明确整数类型与 atol/rtol。
- Multiple Choice 只接受完整选项字母，选项已在固定 Prompt 中；解释文本中偶然出现字母不会判对。
- Manual Review 的 correct 为 null，排除自动正确率分母。非预期特殊 Token 会使自动答案判为无效。

评分参考答案只在生成完成后使用。Raw 与 Chat 按模式/维度统计，不计算混合总分。Raw 长续写也可能因整答案规则得低分；不能据此断言模型没有相关知识。预训练模型的 Chat 低分同样不等于知识缺失。

每个 Prompt 独立使用固定 Seed，Greedy 默认；采样也用独立 Generator，不因题目顺序或模型加载 RNG 改变。Chat 停在 EOS/End-of-Turn，Raw 仅 EOS。保存 Prompt/生成 Token ID、清理后文本、实际预算、停止原因、时长及重复率。非停止特殊 Token 保存在原始 ID 和异常列表中，不泄漏进展示文本。

Prompt 加生成预算超过 Checkpoint 上下文会整体拒绝，不静默截断。Nano 默认上下文 128，不能直接装下全部中文 Chat 题；可为已有 Nano 选短题/明确更小生成预算，或在新的测试实验中显式保存 context512 的 Nano 配置。87M 配置保持 2048。

空输出率、提前结束率、重复 Token/字符 N-gram、重复句子、长度和特殊 Token 异常仅是退化诊断，不是质量总分。随机 ByteLevel 模型可能输出不可解码的 UTF-8 字节，示例会如实保留替换字符。

## 污染检查与覆盖范围

评测时对已知 PR #3 Train/Val/Test 发布文本流式扫描；显式传入的 SFT JSONL 同样检查 text、独立 message 和串联对话。精确检查类别无关折叠空白后的完整 Prompt、Prompt+答案两种边界形式；不将单独短答案相同当作污染。

近重复使用全文字符 4-gram、SimHash64 Hamming≤4、Jaccard≥0.85。只对 ≤20,000 字符记录建立有界特征，更长记录仍做全文精确检查并计数近检跳过。报告列出文件 SHA、扫描数量、命中题 ID、未支持记录与跳过数。

这是已知来源和完整记录层面的近似审计，不检查任意子串、语义改写或未知历史训练源，不能证明完全无污染。未提供来源时标记 NOT_RUN；命中仅作为披露，不自动删除题目或抬高分数。

## 多 Checkpoint 对比

```bash
python scripts/compare_checkpoints.py \
  --checkpoints runs/stage-1/model.pt runs/stage-2/model.pt \
  --tokenizer runs/stage-1/tokenizer.json \
  --dataset data/processed/minillm-pretrain-v1 \
  --config configs/evaluation.example.json --out reports/comparison

# 复用已完成且哈希验证通过的报告
python scripts/compare_checkpoints.py \
  --evaluations reports/baseline reports/later --out reports/comparison-existing
```

产出 `comparison.json/md`，列出 Loss/PPL、Step/训练 Token、各模式维度与基线差值、固定生成并排文本。只有词表、完整配置、数据身份、题集、评分实现、生成策略相同才计算差值。显式 `--allow-incomparable` 只展示独立实验，delta 为 null。

吞吐基于实际同步的生成耗时；Loss Token/s 单独记录。进程 RSS 是导入以来峰值，CUDA 值是 allocated 峰值；每份报告保留硬件、PyTorch/CUDA 与线程信息，不按跨硬件速度排名，不承诺跨 GPU/版本逐位一致。

## 训练曲线分析

```bash
python scripts/analyze_training.py --run runs/pilot-10m --out reports/analysis
```

读取 PR #2 的日志/摘要/Manifest，校验 state.pt 结构及与摘要/Manifest 的一致性。JSON 保留原始记录；CSV 输出 `curves/training.csv`，仅有真实验证观察时才输出 `curves/validation.csv`，Token 为主轴，Step 为辅助轴。稀疏预训练验证点的 Token 位置可从固定计划准确推导；无法推导的 SFT 点写 UNKNOWN，不伪造横轴。

Validation 独立报告 NOT_RUN / COMPLETED / ERROR，并注明未配置或已配置但未执行。无验证集的合法任务不要求 metrics.json，只生成 Train 曲线。配置验证集且正常结束时，PR #2 总会最终验证（包括 val_every=0）；中断会跳过最终验证，已完成的历史验证仍需保留。分析器交叉核对追加日志、metrics.json、Checkpoint history 和最终摘要，不因文件缺失就忽略已执行或应执行的验证。损坏/丢失产物明确报错。运行时 val_every 不在固定 Manifest 中，不凭该缺失信息推测周期性验证次数。

按追加顺序检查：相同重复记录计数并去重，冲突重复、倒退、累计 Token 与 loss_tokens 覆盖缺口、非有限数值、摘要/Checkpoint 不一致均报错，不通过排序掩盖问题。同一步重新验证仅用最新观察参与趋势，原记录仍保留。检查分析期间文件未变化；分析运行中的目录应使用一致快照。

`--config trends.json` 可配置 window（默认 3）、smoothing（1）、loss_delta（0.02）、plateau_delta（0.01）、gradient_limit（100）、throughput_drop（0.4）、jump_loss（2）。两个观察窗口才判断验证下降/平台、Train 降而 Val 升；持续梯度、吞吐下降与 Loss 跳变分别报告。阈值为诊断规则，不自动判断过拟合或修改训练策略。没有足够数据时结论不充分。

PR #2 的 summary.seconds 仅是最后一个进程段；日志区间 Token/s 包含部分验证/保存开销，不能冒充纯算子或完整进程寿命。报告会区分这些口径。

## 启动验收

```bash
python scripts/check_training_readiness.py \
  --dataset data/processed/minillm-pretrain-v1 \
  --tokenizer data/processed/minillm-pretrain-v1/tokenizer.json \
  --config minillm-87m --training-config configs/training.example.json \
  --out reports/readiness
```

训练配置采用 PR #2 TrainingOptions 名称（例如 lr/min_lr），额外要求 seq_len；允许 `target_tokens`（目标下限，最多舍入一个更新）、`planned_effective_tokens`（精确断言）、`allow_data_repetition`、`disk_margin_factor` 和 `gpu_memory_limit_mib`。它是验收配置，不是训练启动器；正式训练 CLI 需要与其计划参数一致。

估算使用固定窗口的真实 L-1 目标，精确计算末尾不足梯度累积组及跨 Epoch 更新。默认拒绝预算超出一遍数据；要复用数据达到 max_steps 必须显式 allow_data_repetition。示例 611 次更新只是每组完整时达到约 1000 万有效 Token 的起点；不同数据长度的 Epoch 尾组可能使预算校验失败，须按报告调整。

验收包括 PR #3 全量审计/词表/Train-Val 窗口、Model Config/参数/特殊 ID、合法 LR/Warmup/步数、目录所有权、可写临时探测、保守磁盘预算和 Resume 固定计划/游标校验。磁盘估算为三份完整状态（含原子替换临时文件）加 model/best，默认 1.2 倍余量；不涵盖新的数据集增长。

模型执行实际配置的 batch1、最多16 Token Forward/Backward，检查 Loss/梯度；Resume 还恢复模型、AdamW、RNG 并检查非有限优化器状态。CUDA 可用时 CPU 任务也运行非必需的 Nano 算子探测；CUDA 任务运行配置模型的小批探测。记录模型配置、实际/请求批量与长度、精度、激活检查点、是否执行优化器更新，以及显存峰值。

算子兼容性与显存预算分开判定。配置 gpu_memory_limit_mib 后，短探测即使低于上限也只能 UNKNOWN，最终 INCOMPLETE / 退出码2；本版本没有实现代表正式负载的显存验收，不提供 GPU 预算 PASS。只有探测可归因的分配增量峰值已超过上限才 FAIL / BLOCKED。报告保留当前进程的绝对 allocated 峰值、探测前基线和增量，不把预先分配的 Tensor、缓存或其他进程的显存计为模型增量。Nano 算子结果从不用于正式模型预算判定。

Short forward/backward probe does not validate full training VRAM requirements.

- READY → 退出码 0：所有强制检查通过，仍受探测范围限制。
- BLOCKED → 退出码 1：明确必需检查失败。
- INCOMPLETE → 退出码 2：必要硬件/检查未完成。

无 CUDA 的 CPU 任务可 READY（CUDA 为可选 NOT_RUN）；请求 CUDA 但不可用时是 INCOMPLETE。报告本身不放入未来训练输出目录。

## 已完成试训练的验收

```bash
python scripts/evaluate_training_pilot.py \
  --run runs/pilot-10m --baseline-eval reports/baseline \
  --final-eval reports/after-pilot --out reports/pilot-acceptance
```

要求同条件且验证 Split 的初始/最终评测，Test 不参与收敛决策。校验模型/状态权重、AdamW 恢复、RNG、游标/Token/目标、数据版本与词表，记录日志数值/恢复异常。实际 Token 和分维度变化均来自文件；不硬编码“已经训练1000万”。初始权重谱系由调用者提供，PR #2 未加密记录初始权重来源，报告不会伪称已证明这一点。

工程通过后，固定评测 Loss 改善且多点验证趋势支持下降才给 CONVERGENCE_IMPROVING；能力变化按各维度观察，没有充分证据给 INCONCLUSIVE，不生成一个主观总分。不会自动建议开展10亿 Token 训练。

## 可复现的 Nano 示例

```bash
python scripts/check_evaluation_workflow.py --out /tmp/minillm-evaluation-demo
```

仅生成120篇合成文档、Train-only 264词表、显式 context512 的 Nano、8次 CPU 更新；包括全部160题、40固定提示、比较、曲线、试训练报告及三种验收。BLOCKED 是损坏的独立测试副本，INCOMPLETE 是 CPU 配置没有执行生产 GPU 上限验证，不是假装硬件不可用。

实际结果见 [示例 JSON](evaluation-example.json) 和 [曲线 CSV](evaluation-curves-example.csv)。固定256个有效验证 Token：Loss 5.5810 → 5.0763，PPL 265.33 → 160.17；训练992有效Token、8 Step。Raw 64题与Chat 96题前后均0正确，能力结论不充分，样例包含真实乱码/空格，没有美化。

本机 NVIDIA T1200 的 Nano CUDA 算子探测实际执行；记录峰值仅适用于该探测。未运行87M正式批量/长度显存验收、1000万Token试训练、1亿～10亿Token训练，也没有对真实87M能力作结论。平台差异和完整依赖环境均在实际报告记录。

## 初版实际验证（复审前）

| 命令或检查 | 结果 |
|---|---|
| `.venv/bin/python -m pytest -q -o addopts=''`（只取消重复 quiet，以显示计数） | 288 passed in 89.46s，新增65项；本机可选CUDA等价检查实际执行 |
| `./scripts/smoke_test.sh` | PASS |
| `SYNTHETIC=1 ./scripts/smoke_test.sh` | PASS |
| `python scripts/check_training_reliability.py` | 原有11项PASS，连续训练/两次恢复参数差0 |
| `python scripts/check_evaluation_workflow.py --out /tmp/minillm-pr4-final-evidence` | PASS，CPU Nano全流程与三状态 |
| 五个新CLI的真实调用 | PASS；验收脚本分别返回0、1、2 |
| Nano同一256有效Token、CPU/CUDA FP32 | Loss绝对差1.0431×10⁻⁷，非跨硬件逐位保证 |
| 示例数据审计与已知污染扫描 | 三对跨Split比较键重叠0；已知Train/Val/Test无题集完整记录或算法近重复命中，不外推未知来源 |
| 受约束文件 | 模型/配置/Tokenizer/SFT/训练/量化/Web/PR3流水线源码未改，原有87M参数与量化预算测试继续通过 |

## PR #4 复审修复验证

本次只修改三个评测/验收模块、对应测试及本说明。没有修改固定160题、40提示与答案，也没有修改训练过程、Checkpoint机制、PR3流水线、模型或量化格式。

显存回归使用CPU可运行的Mock，表中均指配置模型的短探测，不是完整训练基准：

| 条件 | 原逻辑 | 修复后实测状态 / 退出码 |
|---|---|---|
| 短探测64MiB，上限128MiB | PASS / READY | UNKNOWN / INCOMPLETE / 2 |
| 短探测可归因增量512MiB，上限128MiB | FAIL / BLOCKED | FAIL / BLOCKED / 1 |
| 已有分配4096MiB，峰值4160MiB，增量64MiB，上限128MiB | 绝对峰值可能错误归因并BLOCKED | UNKNOWN / INCOMPLETE / 2 |
| CUDA算子异常 | 不可验收 | CUDA兼容性FAIL / BLOCKED / 1 |

本机另外实际运行了配置Nano的CUDA短探测：请求Batch2、Seq32，实际Batch1、Seq16、FP32、无AdamW Step，allocated峰值19.2261MiB、基线0MiB，上限1024MiB。结果仍是 UNKNOWN / INCOMPLETE / 退出码2。该结果只证明算子可运行，未测试87M正式GPU训练负载。

无验证集的真实CPU Nano训练：2次更新、62有效训练Token，未生成metrics.json。分析输出摘录：

```json
{
  "status": "OK",
  "validation": {
    "configured": false,
    "status": "NOT_RUN",
    "reason": "not_configured",
    "observations": 0,
    "metrics_file": "MISSING",
    "final_status": "NOT_RUN"
  },
  "curve_files": ["training.csv"]
}
```

另有真实SIGTERM回归：首次验证前中断可NOT_RUN；已有验证再中断，历史必须保留；删除其metrics.json仍报ERROR。val_every=0正常结束时真实最终验证仍被要求并核验。

未登记冻结题集测试（原始题集与版本声明均不修改，只在临时目录重命名副本）：

```text
PASS unregistered frozen asset rejected:
filename 'renamed-benchmark.jsonl' is not registered in benchmark-version.json
```

修复后完整执行结果：

| 命令 | 结果 |
|---|---|
| `.venv/bin/python -m pytest -q -o addopts=''` | 321 passed in 84.21s；新增33项：显存6、验证状态10、冻结资产17 |
| `./scripts/smoke_test.sh` | PASS |
| `SYNTHETIC=1 ./scripts/smoke_test.sh` | PASS |
| `.venv/bin/python scripts/check_training_reliability.py --out /tmp/minillm-pr4-review-reliability` | 原有11项PASS |
| `.venv/bin/python scripts/check_evaluation_workflow.py --out /tmp/minillm-pr4-review-workflow` | PASS，CLI原有必需out参数显式提供 |
| 三个相关测试模块 | 98 passed in 24.77s |
| Nano CUDA显存验收CLI | 实际INCOMPLETE，退出码2 |
| `git diff --check` | PASS |

没有执行87M完整序列/批量/AdamW GPU训练显存验收或正式长训练；未用短探测伪造显存预算PASS。
