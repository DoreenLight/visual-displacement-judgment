# MotionGPT: Human Motion as a Foreign Language

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, Tao Chen |
| arXiv 首次提交 | 2023 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2306.14795v2` |
| arXiv 分类 | cs.CV, cs.CL, cs.GR |
| 发展线阶段 | S3 |
| 发展线线程 | T5 |
| 康复迁移价值 | 高 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T5_动作生成与离散token/2023_Jiang_MotionGPT_Human_Motion_as_a_Foreign_Language.pdf` |

## 一句话定位（人工判断）

> 把人体运动视为'外语'，token 化运动，翻译隐喻直接对应

## 摘要（arXiv 原文照录，未翻译）

> Though the advancement of pre-trained large language models unfolds, the exploration of building a unified model for language and other multi-modal data, such as motion, remains challenging and untouched so far. Fortunately, human motion displays a semantic coupling akin to human language, often perceived as a form of body language. By fusing language data with large-scale motion models, motion-language pre-training that can enhance the performance of motion-related tasks becomes feasible. Driven by this insight, we propose MotionGPT, a unified, versatile, and user-friendly motion-language model to handle multiple motion-relevant tasks. Specifically, we employ the discrete vector quantization for human motion and transfer 3D motion into motion tokens, similar to the generation process of word tokens. Building upon this "motion vocabulary", we perform language modeling on both motion and text in a unified manner, treating human motion as a specific language. Moreover, inspired by prompt learning, we pre-train MotionGPT with a mixture of motion-language data and fine-tune it on prompt-based question-and-answer tasks. Extensive experiments demonstrate that MotionGPT achieves state-of-the-art performances on multiple motion tasks including text-driven motion generation, motion captioning, motion prediction, and motion in-between.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2306.14795v2>
- PDF：<https://arxiv.org/pdf/2306.14795v2>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

