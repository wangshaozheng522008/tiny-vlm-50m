# PR #5：谱系与恢复一致性修复

本文记录082c544之后的代码修复和CPU Nano回归，与已完成87M试训练的历史测量分开。没有重跑87M训练、迁移实际611步任务或修改其权重、数据、历史谱系及`pilot-evidence.json`。

## 首次更新前持久化身份

`fit()`只新增可选`on_run_prepared`回调：在原输出目录保护通过、真实run_id生成后调用，在GPU模型/优化器初始化及任何参数更新前完成。传递Manifest的深拷贝，回调不能改变实际训练计划。未提供回调的入口行为不变，Loss、AdamW、LR、梯度累积和游标算法均未修改。

启动器先原子发布包含初始化类型/SHA、Tokenizer、数据、训练计划与request_id的启动记录。身份回调再原子写入启动报告目录的`run-identity.json`以及运行目录的`pilot-lineage.json`，绑定实际run_id。每次写入沿用现有临时文件、flush/fsync、原子替换和目录同步；写入失败在任何训练更新前中止。SIGKILL/断电可留下临时文件，但不会把它当成有效身份或Checkpoint。

执行阶段持有输出目录对应的POSIX advisory lock；内核会在SIGKILL后释放锁，不使用可能遗留的PID锁猜测。原训练目录所有权检查保持原样，已有目录不能被新任务覆盖。

## 恢复与兼容规则

`load_for_resume()`仅验证，不在检查模式中写文件。必须满足Checkpoint结构/有限状态、游标、当前完整Manifest、磁盘Manifest与run_id一致；输出目录必须是该Checkpoint所属目录。随后核对初始化来源、原始权重SHA/配置/Tokenizer、原始启动记录和所有已记录Resume请求。训练计划、数据身份及设备/精度策略的原严格比较不变。

- **v2谱系完整**：验证checksum、初始run_id绑定和每条历史记录后恢复。
- **谱系丢失，新启动绑定仍完整**：在训练配置添加`"lineage_report": "/absolute/path/to/original-launch-report"`，只接受原始、可验证的run-identity和完整匹配的Checkpoint。
- **旧v1谱系存在**：验证原始启动记录、源权重和完整运行身份，合法执行Resume时原子迁移到v2。
- **旧v1引用已被Resume覆盖**：必须显式提供原始首次启动报告。保留旧Resume引用和request_id，并明确标注旧格式未记录的起始Step不可重建，不能编造历史。
- **旧任务连谱系也丢失**：原始v1启动报告不足以单独猜测来源；还必须存在与state同Step/Token/权重完全一致、带匹配pilot_request_id与初始化元数据的model.pt或best.pt。无足够证据则拒绝。
- 已存在但损坏的谱系不能以“缺失”为由回退或重建。来源不一致、外来目录、缺少run_id、无原始报告或无法确认来源时均Fail Closed。

校验/迁移需要保留原始来源Checkpoint和引用的报告。Pilot入口不自动搬迁训练目录；其他原始训练入口的现有行为不变。显式选择`state.prev.pt`仍受相同身份校验，备份起点记录在历史中。

## 保留首启与Resume历史

v2保留不可被Resume替换的`initial_launch`，后续请求追加`resume_history`。每条记录包含request_id、report_path和起始checkpoint_step；对应启动记录还保存Checkpoint SHA及run_id。历史表示已验证、准备执行的Resume请求，不能据此声称它已经完成参数更新。

相同请求/同一证据再次追加是幂等的；相同ID对应不同证据会拒绝。CLI复用非空报告目录仍按原规则拒绝，合法的新请求使用新的报告目录与独立request_id。损坏、重复或来源不一致的历史会被拒绝；原子替换失败保留此前有效记录。

## AdamW、RNG及继续训练

`check_pilot_recovery.py`保留20步连续训练与第5/12步两次中断恢复，并增加：

- 比较完整AdamW参数ID/状态结构、逐参数step/exp_avg/exp_avg_sq，以及全部Param Group配置和LR。
- step与结构精确相等，浮点模型/动量使用atol1e-6、rtol1e-5，同时报告实际最大差。
- Python、NumPy、Torch CPU及可用CUDA RNG逐项精确相等。Resume前和比较前均不重新设置Seed；仅两个独立新任务设置初始Seed。
- 将第12步Checkpoint复制到独立测试目录，再继续8步到同一20步目标，比较参数、优化器、RNG、样本顺序、LR与全部计数。不延长训练计划。

新增字段：`optimizer_state_equal`、`rng_state_equal`、`optimizer_max_difference`、`continuation_equal`。任何比较失败都不发布PASS报告。历史87M的recovery.json没有补写这些字段。

## 本次独立CPU Nano实测

新报告：`reports/pr5-reliability-fix/nano-recovery-final/recovery.json`，与`reports/pr5-pilot-v3/`隔离。

| 检查 | 实测结果 |
|---|---|
| 原20步/两次恢复 | PASS，参数最大差0 |
| AdamW | optimizer_state_equal=true，optimizer_max_difference=0 |
| RNG | rng_state_equal=true |
| 第12步后独立继续8步 | continuation_equal=true |
| 样本/LR/进度 | 顺序、LR及计数完全一致，20步/80样本/5040有效Token |

另外的定向测试使用CPU Nano Dropout 0.2，并在每个更新后消耗Python、NumPy和Torch随机数，防止“未使用随机性”掩盖RNG恢复错误。故意修改动量、LR或RNG会导致失败；CUDA RNG结构的精确比较使用合成状态测试，没有执行新增真实CUDA训练。

新增34项回归（谱系25项、状态/继续训练9项）；连同原28项workload测试，定向覆盖真实SIGKILL子进程、启动前持久化、可信来源恢复、v1迁移、外部目录/损坏状态拒绝、两次实际中断Resume历史、重复请求、并发锁、回调隔离及原子写入失败。

| 实际命令 | 结果 |
|---|---|
| `.venv/bin/python -m pytest -q -o addopts=''` | 386 passed in114.13s |
| `./scripts/smoke_test.sh` | PASS |
| `SYNTHETIC=1 ./scripts/smoke_test.sh` | PASS |
| `.venv/bin/python scripts/check_training_reliability.py` | 11项PASS |
| 新版check_pilot_recovery.py：独立CPU Nano、20步、两次恢复及继续8步 | PASS |

## 历史证据与限制

修复前后核对13个受保护文件的SHA256（611步主备状态、model/best、原初始化模型、数据BIN/Manifest/Tokenizer、历史谱系及历史证据JSON），全部未变；历史87M恢复报告与历史证据中的原记录一致。实际611步任务没有被执行Resume或自动迁移。

启动证据是同一可信本地工作区的校验链，不是数字签名，不能抵抗能同时重写所有文件的攻击者。文件锁依赖POSIX文件系统语义，不保证分布式存储行为。删除原始来源或报告、无法重建旧首启身份等场景会拒绝恢复，需要补足可信证据。

旧GPU负载报告仍属于旧实现哈希。新代码不会放宽验证来复用它；未来启动真实CUDA任务需重新提供与当前实现匹配的负载证据。本次按约束没有执行新的87M/GPU训练，也未将CPU新字段冒充历史实机测试。
