---
title: "Fine-grained Motion Retrieval via Joint-Angle Motion Images and Token-Patch Late Interaction"
thread: B4
lens: M3
relevance: "中"
year: 2026
doi: ""
arxiv_id: "2603.09930"
evidence_grade: "B（单来源确认标题，>=0.90）"
confirm_sources: "arxiv"
---

# Fine-grained Motion Retrieval via Joint-Angle Motion Images and Token-Patch Late Interaction

> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**
> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**
> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。

| 项 | 内容 |
|---|---|
| 年份 | 2026 |
| 期刊/会议 | — |
| DOI | — |
| arXiv | `2603.09930` |
| 线程 | B4 离散运动学token_与表征量化 |
| 证据范式 | M3 方法与算法（学习模型、token 化、可微优化） |
| 相关度 | 中（人工裁定） |
| 证据等级 | B（单来源确认标题，>=0.90） |
| 交叉证实来源 | arxiv |
| 官方 OA | 是（arXiv 预印本（官方 OA）） |
| 入口链接 | [https://arxiv.org/abs/2603.09930](https://arxiv.org/abs/2603.09930) |

## 摘要（来源原文，未翻译）

Text-motion retrieval aims to learn a semantically aligned latent space between natural language descriptions and 3D human motion skeleton sequences, enabling bidirectional search across the two modalities. Most existing methods use a dual-encoder framework that compresses motion and text into global embeddings, discarding fine-grained local correspondences, and thus reducing accuracy. Additionally, these global-embedding methods offer limited interpretability of the retrieval results. To overcome these limitations, we propose an interpretable, joint-angle-based motion representation that maps joint-level local features into a structured pseudo-image, compatible with pre-trained Vision Transformers. For text-to-motion retrieval, we employ MaxSim, a token-wise late interaction mechanism, and enhance it with Masked Language Modeling regularization to foster robust, interpretable text-motion alignment. Extensive experiments on HumanML3D and KIT-ML show that our method outperforms state-of-the-art text-motion retrieval approaches while offering interpretable fine-grained correspondences between text and motion. The code is available in the supplementary material.

## 为什么收录（人工裁定理由）

Joint-Angle Motion Images + MaxSim token 交互 + MLM 正则，做文本-动作检索。相邻但不同：是把关节特征映成结构化**伪图像**喂 ViT，**不是离散符号词表**。⚠️ 但若本项目采用 MLM 式训练目标，须正面区分此文。
