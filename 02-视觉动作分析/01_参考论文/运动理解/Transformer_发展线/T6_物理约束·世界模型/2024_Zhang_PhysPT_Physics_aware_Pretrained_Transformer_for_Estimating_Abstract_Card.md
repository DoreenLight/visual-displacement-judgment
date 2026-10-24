# PhysPT: Physics-aware Pretrained Transformer for Estimating Human Dynamics from Monocular Videos

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Yufei Zhang, Jeffrey O. Kephart, Zijun Cui, Qiang Ji |
| arXiv 首次提交 | 2024 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2404.04430v1` |
| arXiv 分类 | cs.CV |
| 发展线阶段 | S3 |
| 发展线线程 | T6 |
| 康复迁移价值 | 高 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T6_物理约束·世界模型/2024_Zhang_PhysPT_Physics_aware_Pretrained_Transformer_for_Estimating.pdf` |

## 一句话定位（人工判断）

> 物理感知预训练 Transformer，从单目视频估计人体动力学；biomechanical grammar 的最强外部先例

## 摘要（arXiv 原文照录，未翻译）

> While current methods have shown promising progress on estimating 3D human motion from monocular videos, their motion estimates are often physically unrealistic because they mainly consider kinematics. In this paper, we introduce Physics-aware Pretrained Transformer (PhysPT), which improves kinematics-based motion estimates and infers motion forces. PhysPT exploits a Transformer encoder-decoder backbone to effectively learn human dynamics in a self-supervised manner. Moreover, it incorporates physics principles governing human motion. Specifically, we build a physics-based body representation and contact force model. We leverage them to impose novel physics-inspired training losses (i.e., force loss, contact loss, and Euler-Lagrange loss), enabling PhysPT to capture physical properties of the human body and the forces it experiences. Experiments demonstrate that, once trained, PhysPT can be directly applied to kinematics-based estimates to significantly enhance their physical plausibility and generate favourable motion forces. Furthermore, we show that these physically meaningful quantities translate into improved accuracy of an important downstream task: human action recognition.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2404.04430v1>
- PDF：<https://arxiv.org/pdf/2404.04430v1>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

