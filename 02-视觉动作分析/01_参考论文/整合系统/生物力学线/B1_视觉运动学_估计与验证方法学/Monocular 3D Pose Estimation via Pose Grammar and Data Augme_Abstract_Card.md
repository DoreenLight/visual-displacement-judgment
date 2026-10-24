---
title: "Monocular 3D Pose Estimation via Pose Grammar and Data Augmentation"
thread: B1
lens: M3
relevance: "高"
year: 2022
doi: "10.1109/tpami.2021.3087695"
arxiv_id: ""
evidence_grade: "A（≥2 个独立来源确认标题）"
confirm_sources: "crossref_doi, europepmc_doi"
---

# Monocular 3D Pose Estimation via Pose Grammar and Data Augmentation

> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**
> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**
> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。

| 项 | 内容 |
|---|---|
| 年份 | 2022 |
| 期刊/会议 | IEEE Transactions on Pattern Analysis and Machine Intelligence |
| DOI | `10.1109/tpami.2021.3087695` |
| arXiv | — |
| 线程 | B1 视觉运动学_估计与验证方法学 |
| 证据范式 | M3 方法与算法（学习模型、token 化、可微优化） |
| 相关度 | 高（人工裁定） |
| 证据等级 | A（≥2 个独立来源确认标题） |
| 交叉证实来源 | crossref_doi, europepmc_doi |
| 官方 OA | 否 |
| 入口链接 | [https://doi.org/10.1109/tpami.2021.3087695](https://doi.org/10.1109/tpami.2021.3087695) |

## 摘要（来源原文，未翻译）

In this paper, we propose a pose grammar to tackle the problem of 3D human pose estimation from a monocular RGB image. Our model takes estimated 2D pose as the input and learns a generalized 2D-3D mapping function to leverage into 3D pose. The proposed model consists of a base network which efficiently captures pose-aligned features and a hierarchy of Bi-directional RNNs (BRNNs) on the top to explicitly incorporate a set of knowledge regarding human body configuration (i.e., kinematics, symmetry, motor coordination). The proposed model thus enforces high-level constraints over human poses. In learning, we develop a data augmentation algorithm to further improve model robustness against appearance variations and cross-view generalization ability. We validate our method on public 3D human pose benchmarks and propose a new evaluation protocol working on cross-view setting to verify the generalization capability of different methods. We empirically observe that most state-of-the-art methods encounter difficulty under such setting while our method can well handle such challenges.

## 为什么收录（人工裁定理由）

★ 【= 台账 P0-2；pose grammar 线的正式期刊版；F6/F10 的关键对照】Xu Y, Wang W, Liu T, Liu X, Xie J, Zhu S。Crossref 核：**IEEE TPAMI 2022;44(10):6327-6344**。📌 **本条的杀伤力**：它明写引入 kinematics → **作废我们「grammar 后继不含力学」这条依据**（F6/F10）。**全文已由用户取回**（原 OpenAlex `closed`、arXiv 按题名 0 命中），**「Biomechanical Grammar 还能不能主张力学那一半」现已可查**。
