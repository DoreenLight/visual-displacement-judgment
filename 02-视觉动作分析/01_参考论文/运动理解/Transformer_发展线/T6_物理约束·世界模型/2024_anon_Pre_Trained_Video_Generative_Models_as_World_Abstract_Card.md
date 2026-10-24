# Pre-Trained Video Generative Models as World Simulators

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | —（非 arXiv 直取件，登记元数据未含作者） |
| arXiv 首次提交 | 2026 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2502.07825v1` |
| arXiv 分类 |  |
| DOI | `10.1609/aaai.v40i6.42465` |
| 发展线阶段 | S4 |
| 发展线线程 | T6 |
| 康复迁移价值 | 中 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T6_物理约束·世界模型/2024_anon_Pre_Trained_Video_Generative_Models_as_World.pdf` |

## 一句话定位（人工判断）

> 预训练视频生成模型作为世界模拟器。**与检索「Video generation models as world simulators」命中的近题论文是同一篇**；Sora 报告本身无 DOI、不在 Crossref，本条不作其替代。

## 摘要（arXiv 原文照录，未翻译）

> Video generative models pre-trained on large-scale internet datasets have achieved remarkable success, excelling at producing realistic synthetic videos. However, they often generate clips based on static prompts (e.g., text or images), limiting their ability to model interactive and dynamic scenarios. In this paper, we propose Dynamic World Simulation (DWS), a novel approach to transform pre-trained video generative models into controllable world simulators capable of executing specified action trajectories. To achieve precise alignment between conditioned actions and generated visual changes, we introduce a lightweight, universal action-conditioned module that seamlessly integrates into any existing model. Instead of focusing on complex visual details, we demonstrate that consistent dynamic transition modeling is the key to building powerful world simulators. Building upon this insight, we further introduce a motion-reinforced loss that enhances action controllability by compelling the model to capture dynamic changes more effectively. Experiments demonstrate that DWS can be versatilely applied to both diffusion and autoregressive transformer models, achieving significant improvements in generating action-controllable, dynamically consistent videos across games and robotics domains. Moreover, to facilitate the applications of the learned world simulator in downstream tasks such as model-based reinforcement learning, we propose prioritized imagination to improve sample efficiency, demonstrating competitive performance compared with state-of-the-art methods.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2502.07825v1>
- PDF：<https://arxiv.org/pdf/2502.07825v1>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

