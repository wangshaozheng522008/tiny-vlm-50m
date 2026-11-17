# V4 Tokenizer 人工定版记录

状态：**TOKENIZER_NOT_FINALIZED**。没有真实人工选择；没有正式 V4 24,576/32,768 候选。
下列空模板不能通过审核，Codex 不填写审核人/日期，也不把推荐结果写成正式选择。

```json
{
  "candidate": "",
  "fingerprint": "",
  "vocab_size": 24576,
  "training_data_content_id": "",
  "train_split_sha256": "",
  "evaluation_report_sha256": "",
  "reason": "",
  "reviewer": "",
  "reviewed_at": "",
  "model_budget_conclusion": ""
}
```

人工根据候选和测评报告填写独立 JSON 文件，再显式运行 `--stage select --decision FILE`。
`training_data_content_id` 是 `v4_tokenizer_decision.training_data_id(dataset_dir)` 返回的文本准备身份，
绑定 Train 与 Split Manifest SHA256，避免循环依赖最终含 Tokenizer 的数据集 Content ID。
最终 Manifest 同时绑定这个决定文件和全部文本/候选/评估阶段。

正式决定必须匹配精确 24,576 词表、特殊 ID、真实 Train-only 候选及 Round-trip/EOS 结果，
并包含理由、审核人、真实 ISO 日期及模型预算结论。32,768 仅作研究对照。
校验器不能证明填写人身份，因此审批文件必须由项目真实审核流程提供，禁止自动生成审批。
`tokenizer-selection.json` 一旦发布，变更须使用新的版本/目录；不静默替换文件。

既有小词表 CPU Fixture/Demo 兼容路径保留；Demo 永远不能正式冻结。
目标达到 Pilot 规模或词表达到 24,576 的正式构建均要求人工决定，缩小词表不能绕过。
新 Tokenizer 对应新模型必须随机初始化，不能把旧 Tokenizer Checkpoint Resume/Init-from 为新身份。
PR9 原有数据/Tokenizer/Checkpoint 兼容性规则继续执行。
