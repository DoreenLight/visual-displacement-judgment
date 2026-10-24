# Temporal Difference Learning for Model Predictive Control

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Nicklas Hansen, Xiaolong Wang, Hao Su |
| arXiv 首次提交 | 2022 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2203.04955v2` |
| arXiv 分类 | cs.LG, cs.RO |
| 发展线阶段 | S2 |
| 发展线线程 | T6 |
| 康复迁移价值 | 低 |
| 标题匹配相似度 | 0.942（arXiv 检索自动比对） |
| 本地 PDF | `T6_物理约束·世界模型/2022_Hansen_Temporal_Difference_Learning_for_Model_Predictive_Control.pdf` |

## 一句话定位（人工判断）

> 可学习模型预测控制

## 摘要（arXiv 原文照录，未翻译）

> Data-driven model predictive control has two key advantages over model-free methods: a potential for improved sample efficiency through model learning, and better performance as computational budget for planning increases. However, it is both costly to plan over long horizons and challenging to obtain an accurate model of the environment. In this work, we combine the strengths of model-free and model-based methods. We use a learned task-oriented latent dynamics model for local trajectory optimization over a short horizon, and use a learned terminal value function to estimate long-term return, both of which are learned jointly by temporal difference learning. Our method, TD-MPC, achieves superior sample efficiency and asymptotic performance over prior work on both state and image-based continuous control tasks from DMControl and Meta-World. Code and video results are available at https://nicklashansen.github.io/td-mpc.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2203.04955v2>
- PDF：<https://arxiv.org/pdf/2203.04955v2>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

