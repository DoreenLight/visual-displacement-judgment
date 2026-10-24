# Generating Long Sequences with Sparse Transformers

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever |
| arXiv 首次提交 | 2019 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `1904.10509v1` |
| arXiv 分类 | cs.LG, stat.ML |
| 发展线阶段 | S1 |
| 发展线线程 | T2 |
| 康复迁移价值 | 低 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T2_效率与长序列/2019_Child_Generating_Long_Sequences_with_Sparse_Transformers.pdf` |

## 一句话定位（人工判断）

> 稀疏注意力先驱

## 摘要（arXiv 原文照录，未翻译）

> Transformers are powerful sequence models, but require time and memory that grows quadratically with the sequence length. In this paper we introduce sparse factorizations of the attention matrix which reduce this to $O(n \sqrt{n})$. We also introduce a) a variation on architecture and initialization to train deeper networks, b) the recomputation of attention matrices to save memory, and c) fast attention kernels for training. We call networks with these changes Sparse Transformers, and show they can model sequences tens of thousands of timesteps long using hundreds of layers. We use the same architecture to model images, audio, and text from raw bytes, setting a new state of the art for density modeling of Enwik8, CIFAR-10, and ImageNet-64. We generate unconditional samples that demonstrate global coherence and great diversity, and show it is possible in principle to use self-attention to model sequences of length one million or more.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/1904.10509v1>
- PDF：<https://arxiv.org/pdf/1904.10509v1>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

