# Visual Instruction Tuning

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee |
| arXiv 首次提交 | 2023 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2304.08485v2` |
| arXiv 分类 | cs.CV, cs.AI, cs.CL, cs.LG |
| 发展线阶段 | S3 |
| 发展线线程 | T4 |
| 康复迁移价值 | 低 |
| 标题匹配相似度 | 0.893（arXiv 检索自动比对） |
| 本地 PDF | `T4_视觉·多模态·人体/2023_Liu_Visual_Instruction_Tuning.pdf` |

## 一句话定位（人工判断）

> 视觉指令微调

## 摘要（arXiv 原文照录，未翻译）

> Instruction tuning large language models (LLMs) using machine-generated instruction-following data has improved zero-shot capabilities on new tasks, but the idea is less explored in the multimodal field. In this paper, we present the first attempt to use language-only GPT-4 to generate multimodal language-image instruction-following data. By instruction tuning on such generated data, we introduce LLaVA: Large Language and Vision Assistant, an end-to-end trained large multimodal model that connects a vision encoder and LLM for general-purpose visual and language understanding.Our early experiments show that LLaVA demonstrates impressive multimodel chat abilities, sometimes exhibiting the behaviors of multimodal GPT-4 on unseen images/instructions, and yields a 85.1% relative score compared with GPT-4 on a synthetic multimodal instruction-following dataset. When fine-tuned on Science QA, the synergy of LLaVA and GPT-4 achieves a new state-of-the-art accuracy of 92.53%. We make GPT-4 generated visual instruction tuning data, our model and code base publicly available.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2304.08485v2>
- PDF：<https://arxiv.org/pdf/2304.08485v2>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

