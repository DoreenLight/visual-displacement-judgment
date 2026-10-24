---
title: "Learning Pose Grammar to Encode Human Body Configuration for 3D Pose Estimation"
thread: B4
lens: M3
relevance: "高"
year: 2017
doi: ""
arxiv_id: "1710.06513"
evidence_grade: "B（单来源确认标题，>=0.90）"
confirm_sources: "arxiv"
---

# Learning Pose Grammar to Encode Human Body Configuration for 3D Pose Estimation

> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**
> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**
> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。

| 项 | 内容 |
|---|---|
| 年份 | 2017 |
| 期刊/会议 | — |
| DOI | — |
| arXiv | `1710.06513` |
| 线程 | B4 离散运动学token_与表征量化 |
| 证据范式 | M3 方法与算法（学习模型、token 化、可微优化） |
| 相关度 | 高（人工裁定） |
| 证据等级 | B（单来源确认标题，>=0.90） |
| 交叉证实来源 | arxiv |
| 官方 OA | 是（arXiv 预印本（官方 OA）） |
| 入口链接 | [https://arxiv.org/abs/1710.06513](https://arxiv.org/abs/1710.06513) |

## 摘要（来源原文，未翻译）

In this paper, we propose a pose grammar to tackle the problem of 3D human pose estimation. Our model directly takes 2D pose as input and learns a generalized 2D-3D mapping function. The proposed model consists of a base network which efficiently captures pose-aligned features and a hierarchy of Bi-directional RNNs (BRNN) on the top to explicitly incorporate a set of knowledge regarding human body configuration (i.e., kinematics, symmetry, motor coordination). The proposed model thus enforces high-level constraints over human poses. In learning, we develop a pose sample simulator to augment training samples in virtual camera views, which further improves our model generalizability. We validate our method on public 3D human pose benchmarks and propose a new evaluation protocol working on cross-view setting to verify the generalization capability of different methods. We empirically observe that most state-of-the-art methods encounter difficulty under such setting while our method can well handle such challenges.

## 为什么收录（人工裁定理由）

★★★ 【F6 撞车】pose grammar 编码人体构型（另见 AAAI 2018, 10.1609/aaai.v32i1.12270）。定义 kinematic/symmetry/coordination 三种 grammar、共 9 条手写规则、实现为 tree-BRNN。**原文自述「Kinematic grammar describes human body movements WITHOUT CONSIDERING FORCES」** —— 明确排除力学，是本项目主张 A（原句级差异）的直接依据。⚠️ 其「用关节链约束姿态」已被占且已含 shoulder↔elbow↔wrist 上肢链，不得再作创新点宣称。⚠️ 发现途径：引文后向追溯 —— 12 条式子检索全部漏掉，因其用词是 pose grammar 而非 vocabulary/token。
