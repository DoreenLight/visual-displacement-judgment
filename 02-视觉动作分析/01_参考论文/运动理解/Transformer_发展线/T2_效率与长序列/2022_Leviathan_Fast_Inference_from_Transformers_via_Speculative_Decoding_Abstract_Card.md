# Fast Inference from Transformers via Speculative Decoding

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Yaniv Leviathan, Matan Kalman, Yossi Matias |
| arXiv 首次提交 | 2022 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2211.17192v2` |
| arXiv 分类 | cs.LG, cs.CL |
| 发展线阶段 | S2 |
| 发展线线程 | T2 |
| 康复迁移价值 | 低 |
| 标题匹配相似度 | 0.556（arXiv 检索自动比对） |
| 本地 PDF | `T2_效率与长序列/2022_Leviathan_Fast_Inference_from_Transformers_via_Speculative_Decoding.pdf` |

## 一句话定位（人工判断）

> 投机解码，加速自回归

## 摘要（arXiv 原文照录，未翻译）

> Inference from large autoregressive models like Transformers is slow - decoding K tokens takes K serial runs of the model. In this work we introduce speculative decoding - an algorithm to sample from autoregressive models faster without any changes to the outputs, by computing several tokens in parallel. At the heart of our approach lie the observations that (1) hard language-modeling tasks often include easier subtasks that can be approximated well by more efficient models, and (2) using speculative execution and a novel sampling method, we can make exact decoding from the large models faster, by running them in parallel on the outputs of the approximation models, potentially generating several tokens concurrently, and without changing the distribution. Our method can accelerate existing off-the-shelf models without retraining or architecture changes. We demonstrate it on T5-XXL and show a 2X-3X acceleration compared to the standard T5X implementation, with identical outputs.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2211.17192v2>
- PDF：<https://arxiv.org/pdf/2211.17192v2>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

