---
title: "Biomechanics-aware Multi-view Markerless Motion Capture of Dexterous Hand Movements"
thread: B1
lens: M3
relevance: "高"
year: 2026
doi: ""
arxiv_id: "2607.02796"
evidence_grade: "B（单来源确认标题，>=0.90）"
confirm_sources: "arxiv"
---

# Biomechanics-aware Multi-view Markerless Motion Capture of Dexterous Hand Movements

> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**
> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**
> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。

| 项 | 内容 |
|---|---|
| 年份 | 2026 |
| 期刊/会议 | — |
| DOI | — |
| arXiv | `2607.02796` |
| 线程 | B1 视觉运动学_估计与验证方法学 |
| 证据范式 | M3 方法与算法（学习模型、token 化、可微优化） |
| 相关度 | 高（人工裁定） |
| 证据等级 | B（单来源确认标题，>=0.90） |
| 交叉证实来源 | arxiv |
| 官方 OA | 是（arXiv 预印本（官方 OA）） |
| 入口链接 | [https://arxiv.org/abs/2607.02796](https://arxiv.org/abs/2607.02796) |

## 摘要（来源原文，未翻译）

Markerless motion capture (MMC) techniques have been widely beneficial in biomechanical analysis of human movement; however, application to complex motions of the hand lags other musculoskeletal systems. The primary goal of this study was to evaluate the performance of a biomechanical reconstruction method that implements a gradient-based optimization approach with a biomechanical model in the loop for tracking dexterous, unconstrained hand movements using MMC. Using a custom, 8-camera setup, we acquired 121 video recordings from 6 participants performing 11 different tasks that spanned 6 hand postures, 5 object manipulation tasks, and involved motion of the proximal upper limb joints. Performance of the proposed MMC pipeline was directly compared to a more commonly adopted two-stage reconstruction method that first triangulates 2D keypoints from computer vision pose estimation algorithms to 3D and then enforces biomechanical constraints by solving a constrained inverse kinematics problem. Relative performance was assessed qualitatively by visual inspection and quantitatively using a computer vision metric. Our method generated solutions for all 121 video recordings; the two-stage method did not converge for 15% of the recordings. Across the remaining videos, our method produced more biomechanically plausible hand kinematics than the two-stage method and was more robust to occlusion effects during tasks that involved objects. The relative robustness of the end-to-end method suggests that it is more effective in utilizing the available 2D digital keypoint information. Automatic and biomechanically meaningful tracking of hand kinematics during dexterous movements has the potential to support clinical evaluation, rehabilitation monitoring, and studies of human motor control.

## 为什么收录（人工裁定理由）

★ 生物力学感知的 dexterous hand 多视角 MMC —— 腕/手 DoF 表征的最近先行。
