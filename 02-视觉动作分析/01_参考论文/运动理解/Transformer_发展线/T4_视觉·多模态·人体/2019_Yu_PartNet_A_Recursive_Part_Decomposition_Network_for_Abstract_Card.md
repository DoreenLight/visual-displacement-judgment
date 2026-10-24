# PartNet: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Fenggen Yu, Kun Liu, Yan Zhang, Chenyang Zhu, Kai Xu |
| arXiv 首次提交 | 2019 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `1903.00709v5` |
| arXiv 分类 | cs.CV |
| 发展线阶段 | S0 |
| 发展线线程 | T4 |
| 康复迁移价值 | 低 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T4_视觉·多模态·人体/2019_Yu_PartNet_A_Recursive_Part_Decomposition_Network_for.pdf` |

## 一句话定位（人工判断）

> 层次化部件分割，解剖先验的早期思路

## 摘要（arXiv 原文照录，未翻译）

> Deep learning approaches to 3D shape segmentation are typically formulated as a multi-class labeling problem. Existing models are trained for a fixed set of labels, which greatly limits their flexibility and adaptivity. We opt for top-down recursive decomposition and develop the first deep learning model for hierarchical segmentation of 3D shapes, based on recursive neural networks. Starting from a full shape represented as a point cloud, our model performs recursive binary decomposition, where the decomposition network at all nodes in the hierarchy share weights. At each node, a node classifier is trained to determine the type (adjacency or symmetry) and stopping criteria of its decomposition. The features extracted in higher level nodes are recursively propagated to lower level ones. Thus, the meaningful decompositions in higher levels provide strong contextual cues constraining the segmentations in lower levels. Meanwhile, to increase the segmentation accuracy at each node, we enhance the recursive contextual feature with the shape feature extracted for the corresponding part. Our method segments a 3D shape in point cloud into an unfixed number of parts, depending on the shape complexity, showing strong generality and flexibility. It achieves the state-of-the-art performance, both for fine-grained and semantic segmentation, on the public benchmark and a new benchmark of fine-grained segmentation proposed in this work. We also demonstrate its application for fine-grained part refinements in image-to-shape reconstruction.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/1903.00709v5>
- PDF：<https://arxiv.org/pdf/1903.00709v5>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

