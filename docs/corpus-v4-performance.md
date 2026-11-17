# Corpus v4 性能与规模基准

本文件记录 P2 规模基准的设计、运行方式和**实际执行**数据。表格里的数字只来自本机
实测；没有执行的阶段明确写 `NOT_RUN`，不用外推值或估算值填充。

## 1. 测试环境

| 项 | 值 |
| --- | --- |
| 主机 | Linux 6.8.0-149-generic，x86_64，16 vCPU |
| Python | 3.11.17 |
| SQLite | 3.53.1 |
| 内存 | 15 GiB（运行前 available 约 8.4 GiB） |
| GPU / 网络 | 未使用 |
| 数据集 | `corpus_benchmark.generate_synthetic_corpus` 生成的合成 JSONL，CC0-1.0，`DEMO_ONLY` |
| 配置 | `configs/corpus_v4.yaml`；报告记录 `config_sha256` |
| 默认 seed | 42 |

合成输入不是真实语料，不用于训练或对外声明语料质量。它只覆盖：中文/long-form、
英文、代码、数学；精确重复、近似重复、family 连接；可重复的输入哈希。

## 2. 运行方式

默认只跑 1,000 篇，避免普通测试/CI 意外执行大规模任务。100k 和 1M 必须显式指定
`--documents`，且输出目录必须不存在（不会覆盖已有产物）。

```bash
# 小规模冒烟；默认 1,000 篇
python scripts/benchmark_corpus_v4.py \
    --documents 1000 --out reports/corpus-v4-benchmark-1000 --seed 42

# 100k 全阶段：Ingest -> Quality -> Dedup -> Split；不训练 Tokenizer
python scripts/benchmark_corpus_v4.py \
    --documents 100000 --out reports/corpus-v4-benchmark-100k --seed 42

# 1M 可按资源只跑部分阶段；例如仅 Ingest
python scripts/benchmark_corpus_v4.py \
    --documents 1000000 --out reports/corpus-v4-benchmark-1m \
    --seed 42 --stages ingest
```

`--stages` 接受 `ingest,quality,dedup,split` 的子集；请求 `split` 时会自动执行其上游
依赖并在报告中分别记录实际执行的阶段。未执行阶段在 JSON 中为 `null`，并列入
`metrics.not_run`；Markdown 摘要显示 `NOT_RUN`。基准不训练 Tokenizer，也不加载
真实数据。

每次运行的输入目录和输出目录分开；输入目录包含 `sources.json`、JSONL 源文件和
`generation.json`。报告同时记录：

- 生成器 seed、文档数、各阶段耗时；
- 每个源文件的 SHA256 和聚合 `input_sha256`；
- 配置 `config_sha256`、Python/SQLite 版本、CPU 数；
- 峰值 RSS、SQLite 文件大小、临时文件峰值、最终输出大小；
- 去重/保留文档数、group 数、跨 split 泄漏。

## 3. 100k 全阶段实测

> 以下数据在 2026-11-16 本机实际执行；`reports/` 下的报告文件已 gitignore，
> 不随仓库提交，但命令和输入 seed/哈希可复现同一工作负载。

输入：`corpus_benchmark.generate_synthetic_corpus`，seed 42，100,000 篇；
`input_sha256 = 1984d6292812b95594332d3115a0ac191cd1d7b97eb213db91b4fb55fd3e3b78`。

| 指标 | 实测值 |
| --- | ---: |
| 总处理耗时 | 954.721 s |
| Ingest / Quality / Dedup / Split 耗时 | 5.069 / 51.463 / 759.104 / 139.082 s |
| Split 耗时（上列拆分） | 139.082 s |
| 峰值 RSS | 667,279,360 B (636.4 MiB) |
| SQLite 占用 | 2,384,961,536 B (2274.5 MiB) |
| 临时文件峰值 | 1,811,529,728 B (1727.6 MiB) |
| 最终输出体积 | 2,569,359,156 B (2450.3 MiB) |
| 保留文档数 | 99,740 |
| 去重文档数 | 260 |
| group 数 | 36,656 |
| 跨 Split 泄漏（normalized train/val, train/test, val/test） | 0 / 0 / 0 |
| 跨 Split 泄漏（raw train/val, train/test, val/test） | 0 / 0 / 0 |
| family/cluster 违规 | 0 |
| near-duplicate 跨 split | hits 0，max_similarity 0.064838（CHECKED） |

阶段统计：quality kept 100,000 / review 0 / rejected 0；dedup 精确重复 100 篇、近似重复 160 篇；split train/val/test 文档数为 95,742 / 1,988 / 2,010。Dedup 耗时主要来自 MinHash/LSH 候选比较（2,886,389 次候选比较），不是 Split 优化能消除的部分。

## 4. 1M 实测

这是**允许的局部阶段**执行，不是 1M 全流程验收：只运行 Ingest；Quality / Dedup / Split
在报告和本表中明确为 `NOT_RUN`。seed 42；`input_sha256 = 78573c4ea2ae1a18c1901a4b9668f468825549cf1a7a3b2e1fc493f0460d66a5`。

| 指标 | 实测值 |
| --- | ---: |
| 总处理耗时 | 69.223 s（仅 Ingest） |
| Split 耗时 | `NOT_RUN` |
| 峰值 RSS | 540,069,888 B (515.1 MiB) |
| SQLite 占用 | 1,452,101,632 B (1384.8 MiB)（仅 `ingested.sqlite`） |
| 临时文件峰值 | 1,452,101,632 B (1384.8 MiB) |
| 保留文档数 | `NOT_RUN` |
| 去重文档数 | `NOT_RUN` |
| 跨 Split 泄漏 | `NOT_RUN` |
| Ingest 吞吐 | 14,447 docs/s（69.219 s） |

未运行原因：100k 实测已经消耗约 16 分钟，其中 Dedup 759 s、Split 139 s；本轮在
共享开发机的时间预算内只完成了 1M Ingest。没有把外推时间写入表格，也不把
“1M Ingest 通过”说成 1M 全阶段通过。

## 5. Split 旧/新实现对比

对比方法：用同一个 deduplicated SQLite 数据库、同一个配置和 seed，分别调用旧实现
（commit `7c2aec4` 中的 `_build_split`）和新实现；只测 Split 阶段，避免 ingest/dedup
噪声。输入相同、输出文件逐字节比较 SHA256。

| 规模 | 旧实现 Split 耗时 | 旧实现峰值 RSS | 新实现 Split 耗时 | 新实现峰值 RSS | 输出 SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| 947 kept docs（固定 seed 42） | 0.759 s | 534,843,392 B (510.1 MiB) | 0.684 s | 530,227,200 B (505.7 MiB) | 5 个文件全部相同 |
| 99,740 kept docs（100k 基准 dedup DB） | FAILED | — | 139.082 s（基准 Split 阶段） | 667,279,360 B（整个进程高水位，非 Split-only） | 未比较（旧实现先失败） |

100k 的旧实现失败信息是 `sqlite3.OperationalError: too many SQL variables`，发生在旧
`_split_leakage` 用 `IN (...)` 构造 kept document 列表时；这不是新实现引入的回归，而是
旧实现在该规模下的实际限制。新实现改用 SQLite `INTERSECT`/join，因此 100k 能完成。
1M 没有做旧实现对比；旧实现首先会在同一处遇到变量上限，且其 Python 对象内存会
随 kept 文档数线性增长。

结论：

- 在 947 kept docs 的同一 dedup DB 上，旧/新输出完全一致（5 个 Split 文件的 SHA256
  相同）；100k 旧实现没有完成，因此不能把“100k 也已逐字节对比”写成事实；
- 新实现不再把所有 kept 文档的 metadata、DSU 字符串 map 和 leakage set 同时放进
  Python 对象；文档 metadata 落在 SQLite scratch 文件，DSU 是 `array('i')`，group
  摘要按 root 流式 SHA256；
- 临时 SQLite 使用 `temp_store=FILE`、`cache_size=-16384`（16 MiB）和
  `journal_mode=OFF`，因此大语料下内存增长主要由 Python 解释器/torch 基线和流式
  缓冲区决定，而不是文档数线性复制。

旧实现到新实现没有改变：group id 的成员集合哈希、score、group 排序、split 分配、
文本文件的 `(category, document_id)` 排序、assignment 的
`(split, group_id, document_id)` 排序、字段和 JSON 键顺序。

## 6. 正确性与失败原子性

对应测试在 `tests/test_corpus_v4_scale.py`（默认 pytest 可执行；不包含 100k/1M）：

- 固定小 fixture 和 1,000 篇合成输入的 Split 输出与旧实现 SHA256 一致；
- 全 family/cluster 连通分量只落到一个 split；
- 数据行物理顺序反转后 Split 输出不变；
- 5,000 篇边界覆盖批处理 flush，assignment 无丢失/重复；
- `_split_leakage` 注入异常后没有完整的 split 文件、manifest 或隐藏临时文件；
- 相同输入重复运行输出 SHA256 相同；
- 基准报告可被 `json.loads` 读取，partial 阶段报告 `NOT_RUN`。

Split 发布顺序是：三个文本文件先写隐藏 `.tmp`，再写 assignment `.tmp`，最后
`os.replace` 发布；manifest 最后写。异常时删除已发布文件和临时文件；成功后才由
stage wrapper 写 `split.stage.json`，因此中途失败不会留下一个“看起来完整”的
Split 目录。

## 7. 100M Token 正式构建的剩余风险

1. **真实语料规模未验证。** 合成 generator 的重复分布、文本长度和 family 密度
   与真实来源不同；100k 实测证明流水线可跑，不等于真实 1.27M 记录可达到 100M token。
2. **Split 不是唯一内存来源。** 本 PR 优化了 Split；Dedup 的 MinHash/LSH、Quality
   和 Tokenizer 训练仍有各自的内存/磁盘峰值。基准不能替代 dedup 的候选质量评估。
3. **临时磁盘压力。** 新 Split 会把 kept 文档 metadata 写入 `work/split-scratch.sqlite`，
   并在 near cross-split 审计时另建索引；1M 级别需要为 `.v4/work` 留出数 GiB 磁盘。
   这次 100k 的临时峰值是实测值；1M 的临时峰值未测。
4. **RSS 口径。** `ru_maxrss`/`VmHWM` 是进程高水位，包含 Python 和 torch 导入基线；
   不是 Split 阶段单独的分配量。临时文件峰值采样间隔为 0.25s，可能漏掉更短的尖峰。
5. **1M 只做了允许的局部阶段或标记 NOT_RUN。** 没有在 1M 上完成全阶段，
   因此不能用本文件声称 1M 全流程已验收；正式构建前应在目标机器上用
   `--documents 1000000` 跑一次全阶段并保存报告。
6. **正式语料仍被许可和预算阻断。** 当前没有 APPROVED 来源，真实 100M Token
   预算还差约 84M；本轮没有训练 Tokenizer、没有写 `FROZEN`，也没有合并 PR。
7. **旧实现对比的适用范围。** 旧/新逐字节对比覆盖固定 fixture 和 1,000 篇；100k
   的旧实现没有完成（先在 `_split_leakage` 的 SQLite 变量上限处失败），因此 100k
   只实际验证了新实现。1M 未做旧实现对比，也没有把未执行结果填成成功。
