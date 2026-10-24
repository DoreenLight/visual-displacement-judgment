---
title: "Differentiable Biomechanics Unlocks Opportunities for Markerless Motion Capture"
thread: B5
lens: M3
relevance: "高"
year: 2025
doi: "10.1109/icorr66766.2025.11063174"
arxiv_id: "2402.17192"
evidence_grade: "B（单来源确认标题，>=0.90）"
confirm_sources: "arxiv"
---

# Differentiable Biomechanics Unlocks Opportunities for Markerless Motion Capture

> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**
> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**
> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。

| 项 | 内容 |
|---|---|
| 年份 | 2025 |
| 期刊/会议 | 2025 International Conference On Rehabilitation Robotics (ICORR) |
| DOI | `10.1109/icorr66766.2025.11063174` |
| arXiv | `2402.17192` |
| 线程 | B5 视觉运动学到动力学_桥接 |
| 证据范式 | M3 方法与算法（学习模型、token 化、可微优化） |
| 相关度 | 高（人工裁定） |
| 证据等级 | B（单来源确认标题，>=0.90） |
| 交叉证实来源 | arxiv |
| 官方 OA | 是（arXiv 预印本（官方 OA）） |
| 入口链接 | [https://arxiv.org/abs/2402.17192](https://arxiv.org/abs/2402.17192) |

## 摘要（来源原文，未翻译）

Recent developments have created differentiable physics simulators designed for machine learning pipelines that can be accelerated on a GPU. While these can simulate biomechanical models, these opportunities have not been exploited for biomechanics research or markerless motion capture. We show that these simulators can be used to fit inverse kinematics to markerless motion capture data, including scaling the model to fit the anthropomorphic measurements of an individual. This is performed end-to-end with an implicit representation of the movement trajectory, which is propagated through the forward kinematic model to minimize the error from the 3D markers reprojected into the images. The differential optimizer yields other opportunities, such as adding bundle adjustment during trajectory optimization to refine the extrinsic camera parameters or meta-optimization to improve the base model jointly over trajectories from multiple participants. This approach improves the reprojection error from markerless motion capture over prior methods and produces accurate spatial step parameters compared to an instrumented walkway for control and clinical populations.

## 为什么收录（人工裁定理由）

★★★ 可微生物力学用于 MMC 的开创工作（arXiv 2402.17192 / ICORR 2025）—— 本项目方法学侧最重要的先行
