# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

> **工件类型：摘要级定位卡（Abstract Card）** — 由 arXiv 官方 API 元数据自动生成，**不是全文翻译**。全文翻译需另行产出 `_全文翻译.md`，两者不可互相冒充。

| 项 | 值 |
|---|---|
| 作者 | Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He |
| arXiv 首次提交 | 2019 |
| 载体 | arXiv 预印本（未见期刊/会议参考） |
| arXiv ID | `1910.02054v3` |
| arXiv 分类 | cs.LG, cs.DC, stat.ML |
| 发展线阶段 | S1 |
| 发展线线程 | T2 |
| 康复迁移价值 | 中 |
| 标题匹配相似度 | 1.000（arXiv 检索自动比对） |
| 本地 PDF | `T2_效率与长序列/2019_Rajbhandari_ZeRO_Memory_Optimizations_Toward_Training_Trillion_Parameter.pdf` |

## 一句话定位（人工判断）

> ZeRO/DeepScale 内存优化：大规模训练系统；与 Scaling Laws 配套阅读

## 摘要（arXiv 原文照录，未翻译）

> Large deep learning models offer significant accuracy gains, but training billions to trillions of parameters is challenging. Existing solutions such as data and model parallelisms exhibit fundamental limitations to fit these models into limited device memory, while obtaining computation, communication and development efficiency. We develop a novel solution, Zero Redundancy Optimizer (ZeRO), to optimize memory, vastly improving training speed while increasing the model size that can be efficiently trained. ZeRO eliminates memory redundancies in data- and model-parallel training while retaining low communication volume and high computational granularity, allowing us to scale the model size proportional to the number of devices with sustained high efficiency. Our analysis on memory requirements and communication volume demonstrates: ZeRO has the potential to scale beyond 1 Trillion parameters using today's hardware. We implement and evaluate ZeRO: it trains large models of over 100B parameter with super-linear speedup on 400 GPUs, achieving throughput of 15 Petaflops. This represents an 8x increase in model size and 10x increase in achievable performance over state-of-the-art. In terms of usability, ZeRO can train large models of up to 13B parameters (e.g., larger than Megatron GPT 8.3B and T5 11B) without requiring model parallelism which is harder for scientists to apply. Last but not the least, researchers have used the system breakthroughs of ZeRO to create the world's largest language model (Turing-NLG, 17B parameters) with record breaking accuracy.

## 来源与证据等级

- 摘要页：<https://arxiv.org/abs/1910.02054v3>
- PDF：<https://arxiv.org/pdf/1910.02054v3>
- **证据等级：官方一手（arXiv 官方 API 记录 + 作者自述原文）**；本卡内容不含任何未经核验的第三方转述。
- 采集方式：`ti:` 标题检索 → 标题相似度 ≥ 0.90 自动采信；期刊/会议归属以 arXiv `journal_ref` / `comment` 字段为准，缺失即标注为「预印本」。

