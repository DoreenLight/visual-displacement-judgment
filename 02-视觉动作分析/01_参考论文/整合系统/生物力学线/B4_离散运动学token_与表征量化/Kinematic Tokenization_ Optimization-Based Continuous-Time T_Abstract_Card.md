---
title: "Kinematic Tokenization: Optimization-Based Continuous-Time Tokens for Learnable Decision Policies in Noisy Time Series"
thread: B4
lens: M3
relevance: "低"
year: 2026
doi: ""
arxiv_id: "2601.09949"
evidence_grade: "B（单来源确认标题，>=0.90）"
confirm_sources: "arxiv"
---

# Kinematic Tokenization: Optimization-Based Continuous-Time Tokens for Learnable Decision Policies in Noisy Time Series

> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**
> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**
> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。

| 项 | 内容 |
|---|---|
| 年份 | 2026 |
| 期刊/会议 | — |
| DOI | — |
| arXiv | `2601.09949` |
| 线程 | B4 离散运动学token_与表征量化 |
| 证据范式 | M3 方法与算法（学习模型、token 化、可微优化） |
| 相关度 | 低（人工裁定） |
| 证据等级 | B（单来源确认标题，>=0.90） |
| 交叉证实来源 | arxiv |
| 官方 OA | 是（arXiv 预印本（官方 OA）） |
| 入口链接 | [https://arxiv.org/abs/2601.09949](https://arxiv.org/abs/2601.09949) |

## 摘要（来源原文，未翻译）

Transformers are designed for discrete tokens, yet many real-world signals are continuous processes observed through noisy sampling. Discrete tokenizations (raw values, patches, finite differences) can be brittle in low signal-to-noise regimes, especially when downstream objectives impose asymmetric penalties that rationally encourage abstention. We introduce Kinematic Tokenization, an optimization-based continuous-time representation that reconstructs an explicit spline from noisy measurements and tokenizes local spline coefficients (position, velocity, acceleration, jerk). This is applied to financial time series data in the form of asset prices in conjunction with trading volume profiles. Across a multi-asset daily-equity testbed, we use a risk-averse asymmetric classification objective as a stress test for learnability. Under this objective, several discrete baselines collapse to an absorbing cash policy (the Liquidation Equilibrium), whereas the continuous spline tokens sustain calibrated, non-trivial action distributions and stable policies. These results suggest that explicit continuous-time tokens can improve the learnability and calibration of selective decision policies in noisy time series under abstention-inducing losses.

## 为什么收录（人工裁定理由）

⚠️ 【命名占用】Kinematic Tokenization —— 标题与本项目高度相似，**实为金融时间序列**（资产价格、样条系数、交易决策）。列入是为标记：**该方法名已被占用，论文中不可直接用作方法名**，否则检索会与此文混淆。领域不撞。
