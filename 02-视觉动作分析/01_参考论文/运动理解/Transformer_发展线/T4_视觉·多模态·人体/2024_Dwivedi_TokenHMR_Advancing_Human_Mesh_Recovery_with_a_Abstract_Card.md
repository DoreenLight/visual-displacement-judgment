# TokenHMR: Advancing Human Mesh Recovery with a Tokenized Pose Representation

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Sai Kumar Dwivedi, Yu Sun, Priyanka Patel, Yao Feng, Michael J. Black |
| arXiv 首次提交 | 2024 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2404.16752v1` |
| arXiv 分类 | cs.CV |
| 发展线阶段 | S4 |
| 发展线线程 | T4 |
| 康复迁移价值 | 高 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T4_视觉·多模态·人体/2024_Dwivedi_TokenHMR_Advancing_Human_Mesh_Recovery_with_a.pdf` |

## 一句话定位（人工判断）

> 姿态词表+token 预测，最贴近'词表化'翻译思想

## 摘要（arXiv 原文照录，未翻译）

> We address the problem of regressing 3D human pose and shape from a single image, with a focus on 3D accuracy. The current best methods leverage large datasets of 3D pseudo-ground-truth (p-GT) and 2D keypoints, leading to robust performance. With such methods, we observe a paradoxical decline in 3D pose accuracy with increasing 2D accuracy. This is caused by biases in the p-GT and the use of an approximate camera projection model. We quantify the error induced by current camera models and show that fitting 2D keypoints and p-GT accurately causes incorrect 3D poses. Our analysis defines the invalid distances within which minimizing 2D and p-GT losses is detrimental. We use this to formulate a new loss Threshold-Adaptive Loss Scaling (TALS) that penalizes gross 2D and p-GT losses but not smaller ones. With such a loss, there are many 3D poses that could equally explain the 2D evidence. To reduce this ambiguity we need a prior over valid human poses but such priors can introduce unwanted bias. To address this, we exploit a tokenized representation of human pose and reformulate the problem as token prediction. This restricts the estimated poses to the space of valid poses, effectively providing a uniform prior. Extensive experiments on the EMDB and 3DPW datasets show that our reformulated keypoint loss and tokenization allows us to train on in-the-wild data while improving 3D accuracy over the state-of-the-art. Our models and code are available for research at https://tokenhmr.is.tue.mpg.de.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2404.16752v1>
- PDF：<https://arxiv.org/pdf/2404.16752v1>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

