# Titans: Learning to Memorize at Test Time

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Ali Behrouz, Peilin Zhong, Vahab Mirrokni |
| arXiv 首次提交 | 2024 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2501.00663v1` |
| arXiv 分类 | cs.LG, cs.AI, cs.CL |
| 发展线阶段 | S4 |
| 发展线线程 | T7 |
| 康复迁移价值 | 中 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T7_状态空间与混合架构/2024_Behrouz_Titans_Learning_to_Memorize_at_Test_Time.pdf` |

## 一句话定位（人工判断）

> 测试时记忆，长期个体化在线适应（康复个体差异）启发

## 摘要（arXiv 原文照录，未翻译）

> Over more than a decade there has been an extensive research effort on how to effectively utilize recurrent models and attention. While recurrent models aim to compress the data into a fixed-size memory (called hidden state), attention allows attending to the entire context window, capturing the direct dependencies of all tokens. This more accurate modeling of dependencies, however, comes with a quadratic cost, limiting the model to a fixed-length context. We present a new neural long-term memory module that learns to memorize historical context and helps attention to attend to the current context while utilizing long past information. We show that this neural memory has the advantage of fast parallelizable training while maintaining a fast inference. From a memory perspective, we argue that attention due to its limited context but accurate dependency modeling performs as a short-term memory, while neural memory due to its ability to memorize the data, acts as a long-term, more persistent, memory. Based on these two modules, we introduce a new family of architectures, called Titans, and present three variants to address how one can effectively incorporate memory into this architecture. Our experimental results on language modeling, common-sense reasoning, genomics, and time series tasks show that Titans are more effective than Transformers and recent modern linear recurrent models. They further can effectively scale to larger than 2M context window size with higher accuracy in needle-in-haystack tasks compared to baselines.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2501.00663v1>
- PDF：<https://arxiv.org/pdf/2501.00663v1>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

