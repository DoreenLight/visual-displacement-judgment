# Human Motion Diffusion Model

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, Amit H. Bermano |
| arXiv 首次提交 | 2022 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2209.14916v2` |
| arXiv 分类 | cs.CV, cs.GR |
| 发展线阶段 | S2 |
| 发展线线程 | T5 |
| 康复迁移价值 | 高 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T5_动作生成与离散token/2022_Tevet_Human_Motion_Diffusion_Model.pdf` |

## 一句话定位（人工判断）

> MDM，动作扩散生成，多样性

## 摘要（arXiv 原文照录，未翻译）

> Natural and expressive human motion generation is the holy grail of computer animation. It is a challenging task, due to the diversity of possible motion, human perceptual sensitivity to it, and the difficulty of accurately describing it. Therefore, current generative solutions are either low-quality or limited in expressiveness. Diffusion models, which have already shown remarkable generative capabilities in other domains, are promising candidates for human motion due to their many-to-many nature, but they tend to be resource hungry and hard to control. In this paper, we introduce Motion Diffusion Model (MDM), a carefully adapted classifier-free diffusion-based generative model for the human motion domain. MDM is transformer-based, combining insights from motion generation literature. A notable design-choice is the prediction of the sample, rather than the noise, in each diffusion step. This facilitates the use of established geometric losses on the locations and velocities of the motion, such as the foot contact loss. As we demonstrate, MDM is a generic approach, enabling different modes of conditioning, and different generation tasks. We show that our model is trained with lightweight resources and yet achieves state-of-the-art results on leading benchmarks for text-to-motion and action-to-motion. https://guytevet.github.io/mdm-page/ .

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2209.14916v2>
- PDF：<https://arxiv.org/pdf/2209.14916v2>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

