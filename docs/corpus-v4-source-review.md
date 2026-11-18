# V4 来源接入与人工许可审核

PR10 当前状态：7 个候选、0 个 APPROVED、7 个 REVIEW_REQUIRED、0 个 REJECTED。
没有下载新语料，没有改变任何真实来源的批准状态。公开许可说明仅是候选证据，不能代替人工决定。

审批记录是 [corpus_v4_sources.review.json](../configs/corpus_v4_sources.review.json)
的完整来源条目，不能只填写 `v4_source` 中的 APPROVED/审核人等基础字段。
正式与 Pilot 配置引用同一份清单；不允许同时声明另一份 `sources`。
发布者、版本、商业使用、再分发、模型训练用途及署名/ShareAlike 等义务必须有明确人工结论。
完整记录保存于来源级 `source_review`（证据协议版本 2），绑定原始条目摘要和选定语料版本。
Plan/来源清单/Manifest 保存完整来源证据；文档 metadata 仅保存 source_id、审核 SHA256、
输入文件 SHA256 和简短 history 引用，不再复制大型文件清单。
运行时盘点是观测结果，不会回写审批清单。
审批时还须在 `local_inventory.files` 记录实际输入文件的绝对路径、SHA256 与字节数；
可从转换 Manifest 的输出文件身份获取。正式 Inventory 比较实际文件清单与该审核快照，
缺证据或替换内容均拒绝，不能仅凭相同文件名准入。

APPROVED 条目还须人工填写 `model_training`、`license_obligations`，并在 `corpus_versions`
显式列出审批适用的数据版本。模板中的 PENDING/空列表不构成默认批准。
审核字符串只在验证时 strip/大小写归一化，拒绝 PENDING、NOT_PINNED、UNKNOWN、UNVERIFIED、
TBD、TODO、N/A 等占位值；保存内容和摘要不因验证而改写。
`license_references` 必须是非空对象列表，每条含有效 `uri` 和非占位的人工 `verification`。
HTTP(S) 引用不要求离线时访问网络；本地证据使用绝对路径或 file URI，并提供 SHA256，验证文件存在及内容。
例如本地证据：`{"uri":"/absolute/COPYRIGHT.txt","sha256":"实际文件摘要","verification":"人工审核说明"}`。
URL 可打开或哈希匹配均不能证明许可合法，真实审核责任仍由所署审核人承担。

生产审核范围使用已有有效监督 Token 目标和 PR9 批准策略确定：Pilot 下限 1,020,000 及以上的
正式配置强制完整来源审核，不使用 corpus_version 的字符串猜测。
显式 CPU/旧版小规模 Fixture 继续支持裸内联来源，但不能满足 PR9 固定生产策略；
PR9 正式入口无条件要求完整审核。任一新审核流程的数据版本还通过不可变
`source_review_policy` 和 Ingest 阶段键保持该要求，删除记录不能退回旧接口。

同一 `validate_reviewed_source()` 在 Inventory/构建前、Audit/Finalize、verify_frozen 和 PR9 再准入时复用。
它重算原始条目与版本摘要，比较条目和实际来源的完整规范内容、实际 Ingest 文件集合与 SHA256/大小，
并重验本地版权证据。来源清单、生成配置、Plan 与 Manifest 绑定也须一致。
伪造标记、复用其他来源/版本的记录、重算自身哈希但引用其他输入文件都不能替代有效证据。
已有缺证据的生产规模产物必须补齐真实审核并创建新版本，不能自动迁移为 READY/FROZEN。

| 类别 | 候选 | 需补齐的证据 |
|---|---|---|
| zh_general 40% | 中文维基文库 | 逐作品版本、作者、版权期限、适用法域及编辑内容许可 |
| zh_encyclopedic 25% | 中文维基百科 | 固定转储版本、导入内容例外、署名及相同方式共享义务 |
| zh_education 15% | 开放教材待选；中文维基教科书 | 教材具体授权、版本、第三方图片/引用等例外 |
| en_general 10% | 英文维基文库 | 逐作品来源及法域、再分发和归属说明 |
| math_logic_code 10% | 数学/代码待选；Python 文档 | 按上游区分许可；文档与代码示例、第三方例外分别审核 |

候选参考：[Wikimedia 使用条款](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)、
[中文维基文库版权说明](https://zh.wikisource.org/wiki/Wikisource:版权信息/全文)、
[Python 文档许可](https://docs.python.org/3/license.html)。项目条款不能证明某个未下载版本的所有内容均满足训练用途；
条目中的 `license_references` 与人工 `license_evidence` 分开保存。
维基教科书项目版权页的独立远程核验未完成，清单没有声称已核验该页。
CC-BY-NC、署名、ShareAlike、引用内容和具体来源限制都必须形成明确审核结论。

审核人须填写具体发布者/版本、许可名称、可验证证据、商业使用限制、再分发条件、
`license_reviewer`、ISO 日期和 `provenance_status`，才能人工决定是否 APPROVED。
`LICENSE_UNVERIFIED` 永不准入；REVIEW_REQUIRED/REJECTED 不进入正式 Ingest。
不能把历史 MiniMind 混合语料直接批准，也不能用测试中的合成许可记录代替真实审批。

## Wikimedia Dump 抽取

```bash
python scripts/extract_wiki_dump.py /read-only/pages-articles.xml.bz2 \
  --source-id YOUR_SOURCE_ID --out /staging/NEW_VERSION/pages.jsonl
```

抽取支持带命名空间的 Wikimedia XML 及无根 multistream 页面片段。解析按页释放 XML 节点，
保留单页内容直至提取完毕。文本超过 8 MiB 时按 `oversized_page` 记账，大小检查在清洗前完成。
保留页数为零时，命令报错并显示扫描和跳过计数。抽取前后校验输入文件的完整 SHA256 和字节数，
并核对读取句柄与输入路径的文件标识。抽取期间请保持原始 dump 稳定。

`--out` 和 `--manifest` 需要指向新的文件路径。Manifest 默认写入 `<out>.manifest.json`。
抽取在各目标所在目录独占创建随机命名的 `.partial` 文件，写完并验证后用 `os.link()` 原子创建正式文件。
目标文件系统须支持同目录硬链接，例如常见的 Linux 文件系统及 Windows NTFS。不支持硬链接时，
命令报错退出。普通异常和 Ctrl-C 会清理本次创建的暂存文件。进程被强制终止可能留下隐藏的 `.partial` 文件，
确认所属进程退出后可清理。

CLI 先发布已验证的 JSONL，再发布 Manifest。若第二步失败，命令返回非零，已验证的 JSONL 保留，
可选择一对新路径重新抽取。用于来源审核的转换证据应包含成对产物，并核对 Manifest 记录的 JSONL 摘要。

`--max-pages N` 在保留第 N 页时停止解析，Manifest 记录 `PARTIAL`、`complete_input_scan: false`
及该上限。即使第 N 页恰好是最后一页，达到上限仍标记为部分抽取。完整文件的 SHA256 和字节数仍会复验。
`COMPLETE` 表示扫描完传入文件。若传入的是从上游 dump 切出的样本，审核时仍需记录其抽样来源与范围。
正式准入继续按 V4 的人工许可和文件审核规则执行。

## 本地暂存

```bash
python scripts/prepare_v4_sources.py /read-only/source.jsonl \
  --source-id YOUR_REVIEWED_SOURCE_ID --category zh_general --language zh \
  --out data/staged/YOUR_NEW_VERSION
```

转换允许暂存尚未审批的本地文件，但不授予正式准入权限。输出保持 TXT 段落/JSONL 文本，
检查 UTF-8、结构与父文档类型；格式错误进入 `errors.jsonl`，原件保持只读。
稳定 ID 使用来源、原文、父文档身份；TXT 保持文件家族，JSONL 保留显式 `parent_id`。
未提供父文档的 JSONL 使用文本身份。重复文档保留次数，留给 V4 去重处理。
SQLite 排序使 JSONL 行顺序变化后正文产物一致，`provenance.jsonl` 则保留真实原始 SHA256 与行号变化。
`conversion-manifest.json` 保存输入/输出哈希、统计、转换版本、吞吐、峰值 RSS 和磁盘占用。
转换前后重验原件；仅完整成功才原子发布新目录，拒绝覆盖非空目录。
转换不会按内容质量自行删文，接入后仍须执行全部 V4 规则。

转换完成后，由人工把选定转换文件路径和原始身份/转换 Manifest 证据写入审核清单。
不得仅因转换成功就更改许可状态。输出目录与原始文件应独立管理。

## 分阶段构建

以下命令仅在真实来源获批且本地输入存在后执行；当前环境不会运行正式构建。

```bash
python scripts/plan_corpus_v4.py --config configs/corpus_v4_pilot_1m.yaml --status-only
python scripts/plan_corpus_v4.py --config configs/corpus_v4_pilot_1m.yaml --stage split
python scripts/plan_corpus_v4.py --config configs/corpus_v4_pilot_1m.yaml --stage evaluate
# 停下，由人工填写决定；不会自动写入推荐值。
python scripts/plan_corpus_v4.py --config configs/corpus_v4_pilot_1m.yaml \
  --stage select --decision /reviewed/tokenizer-decision.json
python scripts/plan_corpus_v4.py --config configs/corpus_v4_pilot_1m.yaml --stage freeze
```

`--stage split` 自动完成 Ingest/Quality/Dedup/Split；`--stage evaluate` 训练 Train-only 候选并测评。
来源合格但尚未选择时报告 `READY_FOR_TOKENIZER_EVALUATION`，`selected: null` 不再阻断前置阶段。
有有效人工选择证据时报告 `READY_FOR_FREEZE`，含义是可以继续 Tokenize/Audit/Freeze，
最终发布仍必须通过全部九项审核。没有选择时不能 Tokenize/Freeze。
`--stage all` 在候选评估后等待人工决定，不自动使用推荐值。

原始 Plan 配置不在运行中修改。人工 `select` 阶段在 `tokenizer-selection.json` 写入生效的
`tokenizer.selected` 和 Fingerprint，再用独立阶段键绑定决定、评估阶段及原 Plan。
手动修改已建目录对应配置的 selected 会改变 Plan 身份，应使用独立选择阶段或新的数据版本。
完整候选按指纹、文件和训练身份校验后可用于中断恢复；候选文件不会覆盖。

Pilot 默认目录 `artifacts/corpus/minillm-corpus-v4-pilot-1m-v1`；正式默认目录
`artifacts/corpus/minillm-corpus-v4-draft`。候选目录也分别隔离。切勿手工指定相同输出目录。
最终冻结后必须再运行 `verify_frozen()` 和 `validate_v4_pilot_policy()`；PR9 GPU 训练不由此流程启动。
PR9 的历史默认数据路径保留；使用本次默认独立版本目录时显式传入 `--dataset`：

```bash
python scripts/run_v4_training_pilot.py \
  --dataset artifacts/corpus/minillm-corpus-v4-pilot-1m-v1 \
  --out reports/YOUR_NEW_PREFLIGHT
```

该命令只做就绪审计，没有 `--execute` 不训练。实际训练仍需另行取得 PR9 全部准入与 GPU 负载证据。
