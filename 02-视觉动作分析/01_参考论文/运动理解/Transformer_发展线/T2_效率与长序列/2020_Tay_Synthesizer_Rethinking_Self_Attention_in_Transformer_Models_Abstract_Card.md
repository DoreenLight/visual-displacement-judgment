# Synthesizer: Rethinking Self-Attention in Transformer Models

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Yi Tay, Dara Bahri, Donald Metzler, Da-Cheng Juan, Zhe Zhao, Che Zheng |
| arXiv 首次提交 | 2020 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2005.00743v3` |
| arXiv 分类 | cs.CL, cs.IR, cs.LG |
| 发展线阶段 | S1 |
| 发展线线程 | T2 |
| 康复迁移价值 | 低 |
| 标题匹配相似度 | 0.946（arXiv 检索自动比对） |
| 本地 PDF | `T2_效率与长序列/2020_Tay_Synthesizer_Rethinking_Self_Attention_in_Transformer_Models.pdf` |

## 一句话定位（人工判断）

> FFN 替代注意力的实验

## 摘要（arXiv 原文照录，未翻译）

> The dot product self-attention is known to be central and indispensable to state-of-the-art Transformer models. But is it really required? This paper investigates the true importance and contribution of the dot product-based self-attention mechanism on the performance of Transformer models. Via extensive experiments, we find that (1) random alignment matrices surprisingly perform quite competitively and (2) learning attention weights from token-token (query-key) interactions is useful but not that important after all. To this end, we propose \textsc{Synthesizer}, a model that learns synthetic attention weights without token-token interactions. In our experiments, we first show that simple Synthesizers achieve highly competitive performance when compared against vanilla Transformer models across a range of tasks, including machine translation, language modeling, text generation and GLUE/SuperGLUE benchmarks. When composed with dot product attention, we find that Synthesizers consistently outperform Transformers. Moreover, we conduct additional comparisons of Synthesizers against Dynamic Convolutions, showing that simple Random Synthesizer is not only $60\%$ faster but also improves perplexity by a relative $3.5\%$. Finally, we show that simple factorized Synthesizers can outperform Linformers on encoding only tasks.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2005.00743v3>
- PDF：<https://arxiv.org/pdf/2005.00743v3>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

