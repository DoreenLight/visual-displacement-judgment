# Big Bird: Transformers for Longer Sequences

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula 等 11 人 |
| arXiv 首次提交 | 2020 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2007.14062v2` |
| arXiv 分类 | cs.LG, cs.CL, stat.ML |
| 发展线阶段 | S1 |
| 发展线线程 | T2 |
| 康复迁移价值 | 中 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T2_效率与长序列/2020_Zaheer_Big_Bird_Transformers_for_Longer_Sequences.pdf` |

## 一句话定位（人工判断）

> 块稀疏注意力

## 摘要（arXiv 原文照录，未翻译）

> Transformers-based models, such as BERT, have been one of the most successful deep learning models for NLP. Unfortunately, one of their core limitations is the quadratic dependency (mainly in terms of memory) on the sequence length due to their full attention mechanism. To remedy this, we propose, BigBird, a sparse attention mechanism that reduces this quadratic dependency to linear. We show that BigBird is a universal approximator of sequence functions and is Turing complete, thereby preserving these properties of the quadratic, full attention model. Along the way, our theoretical analysis reveals some of the benefits of having $O(1)$ global tokens (such as CLS), that attend to the entire sequence as part of the sparse attention mechanism. The proposed sparse attention can handle sequences of length up to 8x of what was previously possible using similar hardware. As a consequence of the capability to handle longer context, BigBird drastically improves performance on various NLP tasks such as question answering and summarization. We also propose novel applications to genomics data.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2007.14062v2>
- PDF：<https://arxiv.org/pdf/2007.14062v2>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

