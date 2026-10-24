# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Albert Gu, Tri Dao |
| arXiv 首次提交 | 2023 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2312.00752v2` |
| arXiv 分类 | cs.LG, cs.AI |
| 发展线阶段 | S3 |
| 发展线线程 | T7 |
| 康复迁移价值 | 高 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T7_状态空间与混合架构/2023_Gu_Mamba_Linear_Time_Sequence_Modeling_with_Selective.pdf` |

## 一句话定位（人工判断）

> 选择性状态空间，线性时间长序列，实时预测潜力

## 摘要（arXiv 原文照录，未翻译）

> Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time architectures such as linear attention, gated convolution and recurrent models, and structured state space models (SSMs) have been developed to address Transformers' computational inefficiency on long sequences, but they have not performed as well as attention on important modalities such as language. We identify that a key weakness of such models is their inability to perform content-based reasoning, and make several improvements. First, simply letting the SSM parameters be functions of the input addresses their weakness with discrete modalities, allowing the model to selectively propagate or forget information along the sequence length dimension depending on the current token. Second, even though this change prevents the use of efficient convolutions, we design a hardware-aware parallel algorithm in recurrent mode. We integrate these selective SSMs into a simplified end-to-end neural network architecture without attention or even MLP blocks (Mamba). Mamba enjoys fast inference (5$\times$ higher throughput than Transformers) and linear scaling in sequence length, and its performance improves on real data up to million-length sequences. As a general sequence model backbone, Mamba achieves state-of-the-art performance across several modalities such as language, audio, and genomics. On language modeling, our Mamba-3B model outperforms Transformers of the same size and matches Transformers twice its size, both in pretraining and downstream evaluation.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2312.00752v2>
- PDF：<https://arxiv.org/pdf/2312.00752v2>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

