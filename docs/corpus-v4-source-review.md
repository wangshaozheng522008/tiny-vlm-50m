# V4 来源接入与人工许可审核

PR10 当前状态：7 个候选、0 个 APPROVED、7 个 REVIEW_REQUIRED、0 个 REJECTED。
没有下载新语料，没有改变任何真实来源的批准状态。公开许可说明仅是候选证据，不能代替人工决定。

唯一审批入口是 [corpus_v4_sources.review.json](../configs/corpus_v4_sources.review.json)
的 `v4_source` 字段。正式与 Pilot 配置引用同一份清单；不允许同时声明另一份 `sources`。
发布者、版本、商业使用和再分发义务也须审核。加载配置时完整审核元数据及清单条目哈希进入
`transformation_history`，随后进入原有 Plan、来源清单、阶段哈希和 Manifest 身份链。
运行时盘点是观测结果，不会回写审批清单。
审批时还须在 `local_inventory.files` 记录实际输入文件的绝对路径、SHA256 与字节数；
可从转换 Manifest 的输出文件身份获取。正式 Inventory 比较实际文件清单与该审核快照，
缺证据或替换内容均拒绝，不能仅凭相同文件名准入。

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
