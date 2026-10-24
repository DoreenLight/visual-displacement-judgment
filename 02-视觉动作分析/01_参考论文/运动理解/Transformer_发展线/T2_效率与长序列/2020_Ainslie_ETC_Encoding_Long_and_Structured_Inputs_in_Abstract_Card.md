# ETC: Encoding Long and Structured Inputs in Transformers

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Joshua Ainslie, Santiago Ontanon, Chris Alberti, Vaclav Cvicek, Zachary Fisher, Philip Pham, Anirudh Ravula, Sumit Sanghai 等 10 人 |
| arXiv 首次提交 | 2020 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2004.08483v5` |
| arXiv 分类 | cs.LG, stat.ML |
| 发展线阶段 | S1 |
| 发展线线程 | T2 |
| 康复迁移价值 | 中 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T2_效率与长序列/2020_Ainslie_ETC_Encoding_Long_and_Structured_Inputs_in.pdf` |

## 一句话定位（人工判断）

> 全局+局部混合，长输入

## 摘要（arXiv 原文照录，未翻译）

> Transformer models have advanced the state of the art in many Natural Language Processing (NLP) tasks. In this paper, we present a new Transformer architecture, Extended Transformer Construction (ETC), that addresses two key challenges of standard Transformer architectures, namely scaling input length and encoding structured inputs. To scale attention to longer inputs, we introduce a novel global-local attention mechanism between global tokens and regular input tokens. We also show that combining global-local attention with relative position encodings and a Contrastive Predictive Coding (CPC) pre-training objective allows ETC to encode structured inputs. We achieve state-of-the-art results on four natural language datasets requiring long and/or structured inputs.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2004.08483v5>
- PDF：<https://arxiv.org/pdf/2004.08483v5>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

