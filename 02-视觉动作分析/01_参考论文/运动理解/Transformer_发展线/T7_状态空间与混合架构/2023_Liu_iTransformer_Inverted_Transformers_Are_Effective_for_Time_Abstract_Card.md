# iTransformer: Inverted Transformers Are Effective for Time Series Forecasting

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long |
| arXiv 首次提交 | 2023 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2310.06625v4` |
| arXiv 分类 | cs.LG |
| 发展线阶段 | S3 |
| 发展线线程 | T7 |
| 康复迁移价值 | 高 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T7_状态空间与混合架构/2023_Liu_iTransformer_Inverted_Transformers_Are_Effective_for_Time.pdf` |

## 一句话定位（人工判断）

> 把注意力建在「变量维」而非「时间维」，正是关节角这类多变量序列的合适归纳偏置；MDPI iTransformer-DTL 即沿此路线做 sEMG/IMU 解码

## 摘要（arXiv 原文照录，未翻译）

> The recent boom of linear forecasting models questions the ongoing passion for architectural modifications of Transformer-based forecasters. These forecasters leverage Transformers to model the global dependencies over temporal tokens of time series, with each token formed by multiple variates of the same timestamp. However, Transformers are challenged in forecasting series with larger lookback windows due to performance degradation and computation explosion. Besides, the embedding for each temporal token fuses multiple variates that represent potential delayed events and distinct physical measurements, which may fail in learning variate-centric representations and result in meaningless attention maps. In this work, we reflect on the competent duties of Transformer components and repurpose the Transformer architecture without any modification to the basic components. We propose iTransformer that simply applies the attention and feed-forward network on the inverted dimensions. Specifically, the time points of individual series are embedded into variate tokens which are utilized by the attention mechanism to capture multivariate correlations; meanwhile, the feed-forward network is applied for each variate token to learn nonlinear representations. The iTransformer model achieves state-of-the-art on challenging real-world datasets, which further empowers the Transformer family with promoted performance, generalization ability across different variates, and better utilization of arbitrary lookback windows, making it a nice alternative as the fundamental backbone of time series forecasting. Code is available at this repository: https://github.com/thuml/iTransformer.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2310.06625v4>
- PDF：<https://arxiv.org/pdf/2310.06625v4>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

