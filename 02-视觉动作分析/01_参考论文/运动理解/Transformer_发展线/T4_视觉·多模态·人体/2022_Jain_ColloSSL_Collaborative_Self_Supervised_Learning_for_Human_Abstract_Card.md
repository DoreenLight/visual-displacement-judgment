# ColloSSL: Collaborative Self-Supervised Learning for Human Activity Recognition

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Yash Jain, Chi Ian Tang, Chulhong Min, Fahim Kawsar, Akhil Mathur |
| arXiv 首次提交 | 2022 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2202.00758v1` |
| arXiv 分类 | cs.LG, cs.AI, cs.HC |
| 发展线阶段 | S1 |
| 发展线线程 | T4 |
| 康复迁移价值 | 中 |
| 标题匹配相似度 | 0.705（arXiv 检索自动比对） |
| 本地 PDF | `T4_视觉·多模态·人体/2022_Jain_ColloSSL_Collaborative_Self_Supervised_Learning_for_Human.pdf` |

## 一句话定位（人工判断）

> 可穿戴自监督，康复传感器表征

## 摘要（arXiv 原文照录，未翻译）

> A major bottleneck in training robust Human-Activity Recognition models (HAR) is the need for large-scale labeled sensor datasets. Because labeling large amounts of sensor data is an expensive task, unsupervised and semi-supervised learning techniques have emerged that can learn good features from the data without requiring any labels. In this paper, we extend this line of research and present a novel technique called Collaborative Self-Supervised Learning (ColloSSL) which leverages unlabeled data collected from multiple devices worn by a user to learn high-quality features of the data. A key insight that underpins the design of ColloSSL is that unlabeled sensor datasets simultaneously captured by multiple devices can be viewed as natural transformations of each other, and leveraged to generate a supervisory signal for representation learning. We present three technical innovations to extend conventional self-supervised learning algorithms to a multi-device setting: a Device Selection approach which selects positive and negative devices to enable contrastive learning, a Contrastive Sampling algorithm which samples positive and negative examples in a multi-device setting, and a loss function called Multi-view Contrastive Loss which extends standard contrastive loss to a multi-device setting. Our experimental results on three multi-device datasets show that ColloSSL outperforms both fully-supervised and semi-supervised learning techniques in majority of the experiment settings, resulting in an absolute increase of upto 7.9% in F_1 score compared to the best performing baselines. We also show that ColloSSL outperforms the fully-supervised methods in a low-data regime, by just using one-tenth of the available labeled data in the best case.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2202.00758v1>
- PDF：<https://arxiv.org/pdf/2202.00758v1>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

