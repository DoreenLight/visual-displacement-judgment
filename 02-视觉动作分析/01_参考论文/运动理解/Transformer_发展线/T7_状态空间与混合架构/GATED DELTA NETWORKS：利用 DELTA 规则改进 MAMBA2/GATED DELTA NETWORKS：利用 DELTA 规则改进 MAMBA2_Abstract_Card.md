# Gated Delta Networks: Improving Mamba2 with Delta Rule

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Songlin Yang, Jan Kautz, Ali Hatamizadeh |
| arXiv 首次提交 | 2024 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2412.06464v3` |
| arXiv 分类 | cs.CL, cs.LG |
| 发展线阶段 | S4 |
| 发展线线程 | T7 |
| 康复迁移价值 | 中 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T7_状态空间与混合架构/2024_Yang_Gated_Delta_Networks_Improving_Mamba2_with_Delta.pdf` |

## 一句话定位（人工判断）

> delta rule 线性注意力

## 摘要（arXiv 原文照录，未翻译）

> Linear Transformers have gained attention as efficient alternatives to standard Transformers, but their performance in retrieval and long-context tasks has been limited. To address these limitations, recent work has explored two distinct mechanisms: gating for adaptive memory control and the delta update rule for precise memory modifications. We observe that these mechanisms are complementary: gating enables rapid memory erasure while the delta rule facilitates targeted updates. Building on this insight, we introduce the gated delta rule and develop a parallel training algorithm optimized for modern hardware. Our proposed architecture, Gated DeltaNet, consistently surpasses existing models like Mamba2 and DeltaNet across multiple benchmarks, including language modeling, common-sense reasoning, in-context retrieval, length extrapolation, and long-context understanding. We further enhance performance by developing hybrid architectures that combine Gated DeltaNet layers with sliding window attention or Mamba2 layers, achieving both improved training efficiency and superior task performance.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2412.06464v3>
- PDF：<https://arxiv.org/pdf/2412.06464v3>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

