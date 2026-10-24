# RoFormer: Enhanced Transformer with Rotary Position Embedding

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu |
| arXiv 首次提交 | 2021 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `2104.09864v5` |
| arXiv 分类 | cs.CL, cs.AI, cs.LG |
| 发展线阶段 | S2 |
| 发展线线程 | T3 |
| 康复迁移价值 | 高 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T3_位置编码与长度外推/2021_Su_RoFormer_Enhanced_Transformer_with_Rotary_Position_Embedding.pdf` |

## 一句话定位（人工判断）

> 旋转位置编码，视觉时序角度建模友好，长度外推强

## 摘要（arXiv 原文照录，未翻译）

> Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. In this paper, we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding(RoPE) to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Finally, we evaluate the enhanced transformer with rotary position embedding, also called RoFormer, on various long text classification benchmark datasets. Our experiments show that it consistently overcomes its alternatives. Furthermore, we provide a theoretical analysis to explain some experimental results. RoFormer is already integrated into Huggingface: \url{https://huggingface.co/docs/transformers/model_doc/roformer}.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/2104.09864v5>
- PDF：<https://arxiv.org/pdf/2104.09864v5>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

