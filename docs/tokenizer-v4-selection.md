# V4 Tokenizer 评估与 87M 预算

实际状态：新 V4 24,576 与 32,768 候选均 **NOT_ASSESSED**，训练阻断为无正式获批 Train Split。
未用合成或 Val/Test 数据补齐真实候选；测试中训练的精确 24,576 词表仅验证软件行为。
历史 24,576 Tokenizer 只作参考，没有自动成为 V4 选择。

固定测试文本为原创 [tokenizer-v4-evaluation.jsonl](../configs/tokenizer-v4-evaluation.jsonl)，16 条，
与 PR6/7 下游冻结能力评测资产分开。覆盖常用词、长句、标点、数字日期单位、中英混合、
实体、数学、Python/C++、Unicode/全角及特殊符号；每条检查 Round-trip 和文档结尾 EOS。
知识主题包含地理国家、历史、基础科学、生活常识、数学逻辑、技术。
这些是分词效率与可逆性测量，不是模型世界知识或答题能力。

对每个类别输出字符/Token、Token/字符、英文 Token/单词、平均 Token/文档及 P50/P90/P99，
另列代码 Token/字符、特殊字符开销、中文汇总、各类别相对历史参考变化。
Tokenizer 比较报告升级为 schema_version=2；候选元数据仍为 schema_version=1，
新增独立人工选择阶段/文件，原有 Manifest schema 保持，新增文件与阶段自然进入身份哈希。
历史报告缺少必要选择或 Split 证据不能升级为生产规模冻结依据。

| 词表 | 总参数 | FP32 权重字节 | INT4 张量负载估算字节 | 87M 适配 |
|---|---:|---:|---:|---|
| 24,576 | 86,528,640 | 346,114,560 | 44,647,680 | 是；尚未定版 |
| 32,768 | 91,771,520 | 367,086,080 | 47,351,040 | 否；仅研究对照 |

结构保持 16 层、640 hidden、10Q/2KV、1792 FFN、RoPE、Weight Tying、上下文 2048。
INT4 估算遵循现有 group-128、FP16 scales/norms、共享 embedding/head 的格式，
24,576 的张量负载低于 50,000,000 字节，但这里没有导出正式权重，序列化文件大小未实测。
报告里的 T1200 预算是既有参数公式估算，未进行 PR10 GPU 工作负载探测或训练。
不能据此宣布显存验收通过；后续仍须 PR9 硬件绑定代表性负载证明。

本次历史参考在固定集上 Round-trip/EOS 均通过；中文 386 字符、292 Token，
1.321918 字符/Token，0.756477 Token/字符。新候选相对变化无数据，保持 NOT_ASSESSED。
完整逐类别数值、固定集 SHA256、历史指纹和预算公式保存在结构化报告。

```bash
python scripts/report_v4_readiness.py --out reports/NEW_READINESS_VERSION \
  --excluded-local data/minimind/pretrain_t2t_mini.jsonl
```

命令只读生成新目录的 `pr10-corpus-readiness.json`、`pr10-tokenizer-comparison.json`，拒绝覆盖。
实际审核后的候选比较还在各独立数据版本的 `tokenizer-evaluation.json` 中，
保留训练选择摘要、候选元数据 Content ID、Train/Split SHA256 与固定集身份。
