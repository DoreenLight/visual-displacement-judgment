# Longformer: The Long-Document Transformer

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Iz Beltagy, Matthew E. Peters, Arman Cohan |
| arXiv 首次提交 | 2020 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2004.05150v2` |
| arXiv 分类 | cs.CL |
| 发展线阶段 | S1 |
| 发展线线程 | T2 |
| 康复迁移价值 | 中 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T2_效率与长序列/2020_Beltagy_Longformer_The_Long_Document_Transformer.pdf` |

## 一句话定位（人工判断）

> 滑窗+全局token，长序列稀疏模式

## 摘要（arXiv 原文照录，未翻译）

> Transformer-based models are unable to process long sequences due to their self-attention operation, which scales quadratically with the sequence length. To address this limitation, we introduce the Longformer with an attention mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or longer. Longformer's attention mechanism is a drop-in replacement for the standard self-attention and combines a local windowed attention with a task motivated global attention. Following prior work on long-sequence transformers, we evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks. Our pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on WikiHop and TriviaQA. We finally introduce the Longformer-Encoder-Decoder (LED), a Longformer variant for supporting long document generative sequence-to-sequence tasks, and demonstrate its effectiveness on the arXiv summarization dataset.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2004.05150v2>
- PDF：<https://arxiv.org/pdf/2004.05150v2>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

