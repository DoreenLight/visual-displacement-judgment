# Byte Latent Transformer: Patches Scale Better Than Tokens

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu 等 14 人 |
| arXiv 首次提交 | 2024 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2412.09871v1` |
| arXiv 分类 | cs.CL |
| 发展线阶段 | S4 |
| 发展线线程 | T7 |
| 康复迁移价值 | 低 |
| 标题匹配相似度 | 0.683（arXiv 检索自动比对） |
| 本地 PDF | `T7_状态空间与混合架构/2024_Pagnoni_Byte_Latent_Transformer_Patches_Scale_Better_Than.pdf` |

## 一句话定位（人工判断）

> 字节级 latent 降推理开销

## 摘要（arXiv 原文照录，未翻译）

> We introduce the Byte Latent Transformer (BLT), a new byte-level LLM architecture that, for the first time, matches tokenization-based LLM performance at scale with significant improvements in inference efficiency and robustness. BLT encodes bytes into dynamically sized patches, which serve as the primary units of computation. Patches are segmented based on the entropy of the next byte, allocating more compute and model capacity where increased data complexity demands it. We present the first FLOP controlled scaling study of byte-level models up to 8B parameters and 4T training bytes. Our results demonstrate the feasibility of scaling models trained on raw bytes without a fixed vocabulary. Both training and inference efficiency improve due to dynamically selecting long patches when data is predictable, along with qualitative improvements on reasoning and long tail generalization. Overall, for fixed inference costs, BLT shows significantly better scaling than tokenization-based models, by simultaneously growing both patch and model size.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2412.09871v1>
- PDF：<https://arxiv.org/pdf/2412.09871v1>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

