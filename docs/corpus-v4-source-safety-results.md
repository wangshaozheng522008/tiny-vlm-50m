# PR10 来源审核门禁修复记录

审查基准：`e3bf50e25cb9a3fdedb19836e4458c4f69b67d76`。本次修复两个 P1 并完成 P2 来源元数据优化。
实际来源保持 7 个 REVIEW_REQUIRED、0 个 APPROVED；每个 `v4_source` 与基准逐项一致，没有自动审批或 GPU 训练。

## P1 根因与修复

P1-1：旧 `build_registry()` 允许仅凭基础 APPROVED 字段准入；Inventory 只检查已经带有标记的来源，
Audit/Freeze/PR9 没有要求并重算完整条目，因此裸内联来源可避开新增审核规则。
现使用同一个 `validate_reviewed_source()`，由 `validate_source_inputs()` / `validate_source_plan()`
在各阶段调用，完整验证来源、审核记录、数据版本和实际输入，而非信任标记或布尔值。

P1-2：原校验不要求非空许可证据引用，且占位值没有统一 strip/大小写处理。
现在 APPROVED 的发布者、版本、商业使用、再分发、模型训练用途、许可义务及人工审批字段均须类型正确、
非空且非占位；只规范化检查值，不改写保存内容。引用必须是非空对象列表，具有 URI 和人工核验说明。
本地版权证据必须存在并匹配 SHA256；HTTP(S) 引用不做实时联网，也不因可打开就视为已获授权。
不完整的 REVIEW_REQUIRED 候选仍可读取。

新增 `model_training`、`license_obligations`、`corpus_versions` 审核字段；真实清单分别保留 PENDING/PENDING/空列表，
必须由人工明确填写，不能默认批准。

## 验证链路

1. 加载统一清单时复用严格条目校验；保存来源级 schema 2 完整记录。
2. 重算 `entry_sha256 = identity(原始完整条目)`；审核记录 SHA256 还绑定选定 corpus_version 和原始条目。
3. 来源基础内容与审核条目的规范结果必须完全一致，包括 source_id、许可、审批人/日期、路径与历史。
4. Inventory 的实际文件集合、SHA256、字节数必须与审核清单精确相同；构建前及 Ingest 后再验原始文件。
5. 来源清单摘要进入生成配置和 Plan，`source_review_policy` 与 Plan key 绑定实际 Ingest 阶段。
6. Audit、Finalize、verify_frozen 和 PR9 再准入重新验证同一证据链；Manifest 继续绑定来源摘要、阶段和文件。

正式审核范围使用配置的有效监督 Token 目标和批准的生产策略：Pilot 的 1,020,000 下限及以上强制审核，
PR9 正式入口还无条件要求该证据。旧小规模 CPU/legacy 内联接口保留；修改名称或传入低预算 Fixture
不能满足 PR9 固定生产策略。新审核流程的版本通过不可变 Plan 中的审核策略保持要求，删除记录不能降级。
严审版本再次验证时需要保留原始输入和本地版权证据；丢失文件不会默认通过。

哈希只证明身份与一致性，不能证明许可证合法或审核人真实身份；人工仍负责训练许可及所有履约结论。

## P2 存储优化

完整审核记录及大型文件清单保留在来源级 Plan/Registry/Manifest。
每篇文档只保存 source_id、source_review_sha256、input_file_sha256 和简短历史引用，可按摘要追溯来源记录。
文档 ID、父文档家族、去重簇、Split 分配、质量规则及监督 Token 算法没有修改。

使用相同证据的两种 SQLite 布局做原创合成对照（64 个输入文件、256 篇文档）：

| 指标 | 每文档复制完整审核记录 | 来源级记录 + 文档引用 | 减少 |
|---|---:|---:|---:|
| 文档 metadata 字节 | 3,489,536 | 142,080 | 95.93% |
| SQLite 文件字节 | 3,710,976 | 397,312 | 89.29% |

对照仅衡量该合成布局，不能当作 100M 真实语料的实测收益。
独立测试确认同一来源文档引用同一审核摘要、不重复文件清单，且相同输入的文档/Family/去重结果及
`split-assignment.jsonl` 与旧布局一致。原有 Demo/1k 固定 Split 快照断言保留并通过。

## 验证结果

运行环境为 `.venv/bin/python`，常规测试固定 CPU（CUDA_VISIBLE_DEVICES 为空）。

| 检查 | 结果 |
|---|---|
| 全量 `python -m pytest -q -o addopts=''` | 822 passed、1 个可选 CUDA skip，348.27 秒 |
| 用户指定的五个重点测试文件 | 215 passed，65.84 秒 |
| 相对审查基准新增回归 | 107 项，未删除或降低原有严格断言 |
| 两种 smoke_test | PASS |
| check_training_reliability | PASS，模型/Optimizer 差异 0，RNG、样本顺序及 LR 一致 |
| PR8 冻结/阶段哈希/旧快照 | PASS；旧接口与缺少新来源协议的小规模 Fixture 兼容 |
| PR9 策略及 CPU Nano/Resume | PASS；生产入口拒绝旧式内联审批，失败路径不调用 launch_pilot |
| 分阶段 Tokenizer/人工决定 | PASS；核心 Tokenizer 算法未修改 |
| 正式/独立 Pilot status-only | NOT_READY |
| 实际 Tokenizer | NOT_ASSESSED / TOKENIZER_NOT_FINALIZED |
| 实际 PR9 Preflight | BLOCKED_DATA_NOT_READY / NOT_RUN |
| CPU Demo | DEMO_ONLY / NOT_READY，无正式 FROZEN |
| GPU/100M 训练及大型性能压测 | NOT_RUN |

新增用例覆盖占位值的空格/大小写/空值/错误类型、空/错误引用、本地证据篡改、源/版本/文件身份不一致、
假标记、重算自身摘要后挪用其他来源或文件、冻结后证据缺失、Audit/Finalize 再校验、PR9 拒绝晋升旧 Fixture，
以及 P2 来源引用、实际 SQLite 存储与文档分组不变性。

## 变更范围与剩余条件

- `src/minillm/v4_sources.py`：共用严格校验、版本/文件/来源身份绑定、来源级审核记录。
- `src/minillm/corpus_v4.py`：构建、Ingest、Audit/Freeze/verify_frozen 接入；文档改用审核引用。
- `src/minillm/v4_admission.py`、`src/minillm/v4_training_pilot.py`：复用验证链，生产入口强制证据。
- `scripts/plan_corpus_v4.py`：向 Inventory 传递配置以确定审核等级。
- `configs/corpus_v4_sources.review.json`：新增待人工填写的审核字段，既有审批字段不变。
- `tests/test_v4_source_onboarding.py`、`tests/test_v4_corpus_readiness.py`：来源/字段/冻结/PR9/P2 回归。
- `README.md`、`docs/corpus-v4-source-review.md`、本文件：审核模板、链路与兼容性说明。
- `reports/pr10-source-safety-validation.json`：本次结构化验证记录。

模型、Loss、Optimizer、Scheduler、Tokenizer 算法、固定 1M 训练预算、Checkpoint/Resume、冻结评测资产、
质量/去重阈值和正式 100M 目标均未修改。来源证据协议和文档引用改变新版本的数据身份，未跳过哈希校验或覆盖历史产物。

上述两个 P1 和 P2 已修复；已验证范围内未发现新增 P0/P1/P2。
真实来源人工审批、足量独立语料、正式 Tokenizer 训练与人工决定、完整冻结及真实 GPU 验收仍未完成。
工程 CODE_READY 不代表 DATA_READY、TOKENIZER_FINALIZED 或正式 Pilot 已就绪。
