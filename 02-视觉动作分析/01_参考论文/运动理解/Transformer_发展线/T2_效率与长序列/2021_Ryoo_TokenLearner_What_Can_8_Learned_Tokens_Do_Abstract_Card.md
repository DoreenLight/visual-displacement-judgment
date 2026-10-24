# TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Michael S. Ryoo, AJ Piergiovanni, Anurag Arnab, Mostafa Dehghani, Anelia Angelova |
| arXiv 首次提交 | 2021 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2106.11297v4` |
| arXiv 分类 | cs.CV, cs.LG |
| 发展线阶段 | S2 |
| 发展线线程 | T2 |
| 康复迁移价值 | 中 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T2_效率与长序列/2021_Ryoo_TokenLearner_What_Can_8_Learned_Tokens_Do.pdf` |

## 一句话定位（人工判断）

> 自适应少量 token，动态 token 化思路

## 摘要（arXiv 原文照录，未翻译）

> In this paper, we introduce a novel visual representation learning which relies on a handful of adaptively learned tokens, and which is applicable to both image and video understanding tasks. Instead of relying on hand-designed splitting strategies to obtain visual tokens and processing a large number of densely sampled patches for attention, our approach learns to mine important tokens in visual data. This results in efficiently and effectively finding a few important visual tokens and enables modeling of pairwise attention between such tokens, over a longer temporal horizon for videos, or the spatial content in images. Our experiments demonstrate strong performance on several challenging benchmarks for both image and video recognition tasks. Importantly, due to our tokens being adaptive, we accomplish competitive results at significantly reduced compute amount. We obtain comparable results to the state-of-the-arts on ImageNet while being computationally more efficient. We also confirm the effectiveness of the approach on multiple video datasets, including Kinetics-400, Kinetics-600, Charades, and AViD. The code is available at: https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2106.11297v4>
- PDF：<https://arxiv.org/pdf/2106.11297v4>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

