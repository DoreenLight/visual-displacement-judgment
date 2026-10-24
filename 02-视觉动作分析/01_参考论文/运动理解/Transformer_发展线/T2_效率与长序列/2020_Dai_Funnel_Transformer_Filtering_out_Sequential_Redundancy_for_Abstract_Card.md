# Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le |
| arXiv 首次提交 | 2020 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2006.03236v1` |
| arXiv 分类 | cs.LG, cs.CL, stat.ML |
| 发展线阶段 | S1 |
| 发展线线程 | T2 |
| 康复迁移价值 | 低 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T2_效率与长序列/2020_Dai_Funnel_Transformer_Filtering_out_Sequential_Redundancy_for.pdf` |

## 一句话定位（人工判断）

> 漏斗式逐层降采样

## 摘要（arXiv 原文照录，未翻译）

> With the success of language pretraining, it is highly desirable to develop more efficient architectures of good scalability that can exploit the abundant unlabeled data at a lower cost. To improve the efficiency, we examine the much-overlooked redundancy in maintaining a full-length token-level presentation, especially for tasks that only require a single-vector presentation of the sequence. With this intuition, we propose Funnel-Transformer which gradually compresses the sequence of hidden states to a shorter one and hence reduces the computation cost. More importantly, by re-investing the saved FLOPs from length reduction in constructing a deeper or wider model, we further improve the model capacity. In addition, to perform token-level predictions as required by common pretraining objectives, Funnel-Transformer is able to recover a deep representation for each token from the reduced hidden sequence via a decoder. Empirically, with comparable or fewer FLOPs, Funnel-Transformer outperforms the standard Transformer on a wide variety of sequence-level prediction tasks, including text classification, language understanding, and reading comprehension. The code and pretrained checkpoints are available at https://github.com/laiguokun/Funnel-Transformer.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2006.03236v1>
- PDF：<https://arxiv.org/pdf/2006.03236v1>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

