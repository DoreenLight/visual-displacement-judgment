# T7 状态空间与混合架构 — 论文清单

> 共 10 篇（PDF 入库 10 篇，需人工取件 0 篇）。迁移价值「高」= 2 篇。

| # | 年 | 论文 | 载体 | arXiv | 阶段 | 迁移价值 | PDF |
|---|---|---|---|---|---|---|---|
| 1 | 2021 | [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396v3) | arXiv 预印本 | [`2111.00396v3`](https://arxiv.org/abs/2111.00396v3) | S2 | 中 | ✅ |
| 2 | 2023 | [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866v3) | arXiv 预印本 | [`2302.10866v3`](https://arxiv.org/abs/2302.10866v3) | S3 | 低 | ✅ |
| 3 | 2023 | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752v2) | arXiv 预印本 | [`2312.00752v2`](https://arxiv.org/abs/2312.00752v2) | S3 | 高 | ✅ |
| 4 | 2023 | [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625v4) | arXiv 预印本 | [`2310.06625v4`](https://arxiv.org/abs/2310.06625v4) | S3 | 高 | ✅ |
| 5 | 2024 | [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871v1) | arXiv 预印本 | [`2412.09871v1`](https://arxiv.org/abs/2412.09871v1) | S4 | 低 | ✅ |
| 6 | 2024 | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464v3) | arXiv 预印本 | [`2412.06464v3`](https://arxiv.org/abs/2412.06464v3) | S4 | 中 | ✅ |
| 7 | 2024 | [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484v6) | arXiv 预印本 | [`2406.06484v6`](https://arxiv.org/abs/2406.06484v6) | S4 | 低 | ✅ |
| 8 | 2024 | [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663v1) | arXiv 预印本 | [`2501.00663v1`](https://arxiv.org/abs/2501.00663v1) | S4 | 中 | ✅ |
| 9 | 2024 | [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060v1) | arXiv 预印本 | [`2405.21060v1`](https://arxiv.org/abs/2405.21060v1) | S3 | 中 | ✅ |
| 10 | 2025 | [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089v2) | arXiv 预印本 | [`2502.11089v2`](https://arxiv.org/abs/2502.11089v2) | S5 | 中 | ✅ |

## 逐篇定位（人工判断）

### Efficiently Modeling Long Sequences with Structured State Spaces（2021，2111.00396v3）

- **作者**：Albert Gu, Karan Goel, Christopher Ré
- **载体**：arXiv 预印本
- **阶段/线程**：S2 / T7
- **康复迁移价值**：中
- **定位**：S4，结构化状态空间
- **来源**：<https://arxiv.org/abs/2111.00396v3> ｜ <https://arxiv.org/pdf/2111.00396v3>
- **证据等级**：官方一手（arXiv 官方 API，标题相似度 1.000）
- **本地工件**：PDF 已入库 + 摘要级定位卡

### Hyena Hierarchy: Towards Larger Convolutional Language Models（2023，2302.10866v3）

- **作者**：Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus 等
- **载体**：arXiv 预印本
- **阶段/线程**：S3 / T7
- **康复迁移价值**：低
- **定位**：隐式长卷积
- **来源**：<https://arxiv.org/abs/2302.10866v3> ｜ <https://arxiv.org/pdf/2302.10866v3>
- **证据等级**：官方一手（arXiv 官方 API，标题相似度 1.000）
- **本地工件**：PDF 已入库 + 摘要级定位卡

### Mamba: Linear-Time Sequence Modeling with Selective State Spaces（2023，2312.00752v2）

- **作者**：Albert Gu, Tri Dao
- **载体**：arXiv 预印本
- **阶段/线程**：S3 / T7
- **康复迁移价值**：高
- **定位**：选择性状态空间，线性时间长序列，实时预测潜力
- **来源**：<https://arxiv.org/abs/2312.00752v2> ｜ <https://arxiv.org/pdf/2312.00752v2>
- **证据等级**：官方一手（arXiv 官方 API，标题相似度 1.000）
- **本地工件**：PDF 已入库 + 摘要级定位卡

### iTransformer: Inverted Transformers Are Effective for Time Series Forecasting（2023，2310.06625v4）

- **作者**：Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma 等
- **载体**：arXiv 预印本
- **阶段/线程**：S3 / T7
- **康复迁移价值**：高
- **定位**：把注意力建在「变量维」而非「时间维」，正是关节角这类多变量序列的合适归纳偏置；MDPI iTransformer-DTL 即沿此路线做 sEMG/IMU 解码
- **来源**：<https://arxiv.org/abs/2310.06625v4> ｜ <https://arxiv.org/pdf/2310.06625v4>
- **证据等级**：官方一手（arXiv 官方 API，标题相似度 1.000）
- **本地工件**：PDF 已入库 + 摘要级定位卡

### Byte Latent Transformer: Patches Scale Better Than Tokens（2024，2412.09871v1）

- **作者**：Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li 等
- **载体**：arXiv 预印本
- **阶段/线程**：S4 / T7
- **康复迁移价值**：低
- **定位**：字节级 latent 降推理开销
- **来源**：<https://arxiv.org/abs/2412.09871v1> ｜ <https://arxiv.org/pdf/2412.09871v1>
- **证据等级**：官方一手（arXiv 官方 API，标题相似度 0.683）
- **本地工件**：PDF 已入库 + 摘要级定位卡

### Gated Delta Networks: Improving Mamba2 with Delta Rule（2024，2412.06464v3）

- **作者**：Songlin Yang, Jan Kautz, Ali Hatamizadeh
- **载体**：arXiv 预印本
- **阶段/线程**：S4 / T7
- **康复迁移价值**：中
- **定位**：delta rule 线性注意力
- **来源**：<https://arxiv.org/abs/2412.06464v3> ｜ <https://arxiv.org/pdf/2412.06464v3>
- **证据等级**：官方一手（arXiv 官方 API，标题相似度 1.000）
- **本地工件**：PDF 已入库 + 摘要级定位卡

### Parallelizing Linear Transformers with the Delta Rule over Sequence Length（2024，2406.06484v6）

- **作者**：Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim
- **载体**：arXiv 预印本
- **阶段/线程**：S4 / T7
- **康复迁移价值**：低
- **定位**：DeltaNet，并行 delta 规则
- **来源**：<https://arxiv.org/abs/2406.06484v6> ｜ <https://arxiv.org/pdf/2406.06484v6>
- **证据等级**：官方一手（arXiv 官方 API，标题相似度 1.000）
- **本地工件**：PDF 已入库 + 摘要级定位卡

### Titans: Learning to Memorize at Test Time（2024，2501.00663v1）

- **作者**：Ali Behrouz, Peilin Zhong, Vahab Mirrokni
- **载体**：arXiv 预印本
- **阶段/线程**：S4 / T7
- **康复迁移价值**：中
- **定位**：测试时记忆，长期个体化在线适应（康复个体差异）启发
- **来源**：<https://arxiv.org/abs/2501.00663v1> ｜ <https://arxiv.org/pdf/2501.00663v1>
- **证据等级**：官方一手（arXiv 官方 API，标题相似度 1.000）
- **本地工件**：PDF 已入库 + 摘要级定位卡

### Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality（2024，2405.21060v1）

- **作者**：Tri Dao, Albert Gu
- **载体**：arXiv 预印本
- **阶段/线程**：S3 / T7
- **康复迁移价值**：中
- **定位**：Mamba-2，SSM 与 Transformer 的对偶统一
- **来源**：<https://arxiv.org/abs/2405.21060v1> ｜ <https://arxiv.org/pdf/2405.21060v1>
- **证据等级**：官方一手（arXiv 官方 API，标题相似度 1.000）
- **本地工件**：PDF 已入库 + 摘要级定位卡

### Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention（2025，2502.11089v2）

- **作者**：Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang 等
- **载体**：arXiv 预印本
- **阶段/线程**：S5 / T7
- **康复迁移价值**：中
- **定位**：硬件对齐稀疏注意力，长上下文效率新解
- **来源**：<https://arxiv.org/abs/2502.11089v2> ｜ <https://arxiv.org/pdf/2502.11089v2>
- **证据等级**：官方一手（arXiv 官方 API，标题相似度 0.447）
- **本地工件**：PDF 已入库 + 摘要级定位卡

