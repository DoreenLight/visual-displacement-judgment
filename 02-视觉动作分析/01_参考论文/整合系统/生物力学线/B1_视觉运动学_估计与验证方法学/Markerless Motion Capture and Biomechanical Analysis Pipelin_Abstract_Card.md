---
title: "Markerless Motion Capture and Biomechanical Analysis Pipeline"
thread: B1
lens: M3
relevance: "中"
year: 2023
doi: ""
arxiv_id: "2303.10654"
evidence_grade: "B（单来源确认标题，>=0.90）"
confirm_sources: "arxiv"
---

# Markerless Motion Capture and Biomechanical Analysis Pipeline

> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**
> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**
> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。

| 项 | 内容 |
|---|---|
| 年份 | 2023 |
| 期刊/会议 | — |
| DOI | — |
| arXiv | `2303.10654` |
| 线程 | B1 视觉运动学_估计与验证方法学 |
| 证据范式 | M3 方法与算法（学习模型、token 化、可微优化） |
| 相关度 | 中（人工裁定） |
| 证据等级 | B（单来源确认标题，>=0.90） |
| 交叉证实来源 | arxiv |
| 官方 OA | 是（arXiv 预印本（官方 OA）） |
| 入口链接 | [https://arxiv.org/abs/2303.10654](https://arxiv.org/abs/2303.10654) |

## 摘要（来源原文，未翻译）

Markerless motion capture using computer vision and human pose estimation (HPE) has the potential to expand access to precise movement analysis. This could greatly benefit rehabilitation by enabling more accurate tracking of outcomes and providing more sensitive tools for research. There are numerous steps between obtaining videos to extracting accurate biomechanical results and limited research to guide many critical design decisions in these pipelines. In this work, we analyze several of these steps including the algorithm used to detect keypoints and the keypoint set, the approach to reconstructing trajectories for biomechanical inverse kinematics and optimizing the IK process. Several features we find important are: 1) using a recent algorithm trained on many datasets that produces a dense set of biomechanically-motivated keypoints, 2) using an implicit representation to reconstruct smooth, anatomically constrained marker trajectories for IK, 3) iteratively optimizing the biomechanical model to match the dense markers, 4) appropriate regularization of the IK process. Our pipeline makes it easy to obtain accurate biomechanical estimates of movement in a rehabilitation hospital.

## 为什么收录（人工裁定理由）

MMC 与生物力学分析流水线 —— 工程基线。
