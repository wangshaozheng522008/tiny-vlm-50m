# 数学／代码类别合规接入方案

本文件记录 PR #14 对 `math_logic_code` 候选来源的本地核查结果和后续接入步骤。当前结论为
`MATH_CODE_SOURCE_REQUIRED`：本机没有可准入的真实文件，也没有足够的人工许可记录。因此该类别保持
缺失，不用测试数据、生成题目或其他类别的文本补齐。

## 本次核查范围

| 候选 Source ID | 配置路径 | 本地文件 | 许可状态 | 可准入 |
|---|---|---|---|---|
| `candidate-math-code` | `data/raw/candidate-math-code` | 不存在 | `REVIEW_REQUIRED` / `LICENSE_UNVERIFIED` | 否 |
| `candidate-python-docs` | `data/raw/candidate-python-docs` | 不存在 | `REVIEW_REQUIRED`；Python 文档为 PSF-2.0，示例涉及 0BSD 及其他例外 | 否 |

同时检查了现有转换暂存目录（`data/staged/`）和已提交的样本 Manifest。未发现这两个 Source ID 的
JSONL、转换清单、样本 Manifest 或人工批准记录。现有四份已批准 Wikimedia 样本均不属于
`math_logic_code`，不能跨类别复用。

`candidate-python-docs` 的许可字符串只是候选登记信息，不构成人工批准；Python 文档中的示例、第三方
贡献和再分发条款仍需逐项审核。Codex 不作出法律许可结论，也不自动将 `REVIEW_REQUIRED` 改为
`APPROVED`。

## 目标与当前缺口

- Pilot 有效监督 Token 目标：1,020,000。
- `math_logic_code` 目标配比：10%，即 102,000 个有效监督 Token。
- 当前准入文档：0；当前估算 Train Token：0；估算缺口：102,000。
- 供给估算使用 `chars_per_token`，只能用于规划。正式缺口以选定 Tokenizer 编码后的 packing 监督位置
  为准；在该 Tokenizer 被人工作出正式选择前，实测 Token 字段保持 `null`。

## 合规接入步骤

1. **确定来源和版本**：选择发布者、具体版本/日期、上游 URI 和许可。保留不可变的上游快照或校验清单。
   不在未确认授权的情况下自动下载大型数据集。
2. **完成许可审核**：逐项记录商业使用、再分发、模型训练用途、署名、ShareAlike/Notice 及其他义务。
   对代码示例、第三方材料和数据文件分别审核，不能只看项目主页的许可名称。
3. **本地只读暂存**：把原始文件放到受控的本地只读目录，计算并记录 SHA256 和字节数。原始文件不提交到
   Git，也不写进公开报告。
4. **转换为 JSONL**：复用 `scripts/prepare_v4_sources.py` 或现有转换流程，保存转换 Manifest、输入身份、
   错误队列和来源字段。不得用生成内容填充缺失记录。
5. **确定性抽样**：使用 `scripts/prepare_v4_pilot_sample.py` 生成哈希绑定的样本 Manifest 和审核草稿。
   抽样参数、输入 SHA256、输出 SHA256 和家族统计必须可复跑。
6. **独立人工复核**：在本地审核 Registry 中填写完整来源信息、许可证据、审核人、ISO 日期和
   `corpus_versions=["minillm-corpus-v4-pilot-1m-v1"]`，由人标记 `APPROVED`。Codex 只能准备草稿。
7. **重新准入**：用 `scripts/plan_corpus_v4.py --status-only` 确认新来源被纳入，再运行
   Ingest → Quality → Dedup → Split。既有四类数据仍需保持原有阶段身份；新输入或新配置使用新的输出目录。
8. **重新分析**：运行 `scripts/analyze_v4_pilot_supply.py`，确认来源/类别汇总一致、Val/Test 未计入 Train、
   缺口按实测口径更新。只有在所有类别达标、Tokenizer 人工选定、污染/预算/最终审计全部通过后才可考虑 Freeze。

## 不得执行的操作

- 不把测试夹具、合成数学题或代码片段计入真实 Pilot 供给。
- 不把 `REVIEW_REQUIRED` 自动改为 `APPROVED`。
- 不下载许可未确认的数据集。
- 不通过重复语料、跨类别搬运或降低 Quality/Dedup 阈值来填补 102,000 Token 缺口。
- 不在缺少人工 Tokenizer 决定时宣称 `READY_FOR_FREEZE` 或 `FROZEN_VERIFIED`。
