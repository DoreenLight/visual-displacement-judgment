# Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y. X. Wei 等 15 人 |
| arXiv 首次提交 | 2025 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2502.11089v2` |
| arXiv 分类 | cs.CL, cs.AI, cs.LG |
| 发展线阶段 | S5 |
| 发展线线程 | T7 |
| 康复迁移价值 | 中 |
| 标题匹配相似度 | 0.447（arXiv 检索自动比对） |
| 本地 PDF | `T7_状态空间与混合架构/2025_Yuan_Native_Sparse_Attention_Hardware_Aligned_and_Natively.pdf` |

## 一句话定位（人工判断）

> 硬件对齐稀疏注意力，长上下文效率新解

## 摘要（arXiv 原文照录，未翻译）

> Long-context modeling is crucial for next-generation language models, yet the high computational cost of standard attention mechanisms poses significant computational challenges. Sparse attention offers a promising direction for improving efficiency while maintaining model capabilities. We present NSA, a Natively trainable Sparse Attention mechanism that integrates algorithmic innovations with hardware-aligned optimizations to achieve efficient long-context modeling. NSA employs a dynamic hierarchical sparse strategy, combining coarse-grained token compression with fine-grained token selection to preserve both global context awareness and local precision. Our approach advances sparse attention design with two key innovations: (1) We achieve substantial speedups through arithmetic intensity-balanced algorithm design, with implementation optimizations for modern hardware. (2) We enable end-to-end training, reducing pretraining computation without sacrificing model performance. As shown in Figure 1, experiments show the model pretrained with NSA maintains or exceeds Full Attention models across general benchmarks, long-context tasks, and instruction-based reasoning. Meanwhile, NSA achieves substantial speedups over Full Attention on 64k-length sequences across decoding, forward propagation, and backward propagation, validating its efficiency throughout the model lifecycle.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2502.11089v2>
- PDF：<https://arxiv.org/pdf/2502.11089v2>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

