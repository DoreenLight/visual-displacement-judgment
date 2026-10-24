---
title: "Uncertainty-Aware Mapping from 3D Keypoints to Anatomical Landmarks for Markerless Biomechanics"
thread: B5
lens: M1
relevance: "高"
year: 2026
doi: ""
arxiv_id: "2603.26844"
evidence_grade: "B（单来源确认标题，>=0.90）"
confirm_sources: "arxiv"
---

# Uncertainty-Aware Mapping from 3D Keypoints to Anatomical Landmarks for Markerless Biomechanics

> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**
> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**
> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。

| 项 | 内容 |
|---|---|
| 年份 | 2026 |
| 期刊/会议 | — |
| DOI | — |
| arXiv | `2603.26844` |
| 线程 | B5 视觉运动学到动力学_桥接 |
| 证据范式 | M1 测量与验证（精度/信度/有效性研究） |
| 相关度 | 高（人工裁定） |
| 证据等级 | B（单来源确认标题，>=0.90） |
| 交叉证实来源 | arxiv |
| 官方 OA | 是（arXiv 预印本（官方 OA）） |
| 入口链接 | [https://arxiv.org/abs/2603.26844](https://arxiv.org/abs/2603.26844) |

## 摘要（来源原文，未翻译）

Markerless biomechanics increasingly relies on 3D skeletal keypoints extracted from video, yet downstream biomechanical mappings typically treat these estimates as deterministic, providing no principled mechanism for frame-wise quality control. In this work, we investigate predictive uncertainty as a quantitative measure of confidence for mapping 3D pose keypoints to 3D anatomical landmarks, a critical step preceding inverse kinematics and musculoskeletal analysis. Within a temporal learning framework, we model both uncertainty arising from observation noise and uncertainty related to model limitations. Using synchronized motion capture ground truth on AMASS, we evaluate uncertainty at frame and joint level through error--uncertainty rank correlation, risk--coverage analysis, and catastrophic outlier detection. Across experiments, uncertainty estimates, particularly those associated with model uncertainty, exhibit a strong monotonic association with landmark error (Spearman $ρ\approx 0.63$), enabling selective retention of reliable frames (error reduced to $\approx 16.8$ mm at 10% coverage) and accurate detection of severe failures (ROC-AUC $\approx 0.92$ for errors $&gt;50$ mm). Reliability ranking remains stable under controlled input degradation, including Gaussian noise and simulated missing joints. In contrast, uncertainty attributable to observation noise provides limited additional benefit in this setting, suggesting that dominant failures in keypoint-to-landmark mapping are driven primarily by model uncertainty. Our results establish predictive uncertainty as a practical, frame-wise tool for automatic quality control in markerless biomechanical pipelines.

## 为什么收录（人工裁定理由）

★★ 3D 关键点→解剖标志物的不确定性量化（ρ≈0.63，ROC-AUC 0.92）。离散 token 辅助头做置信度/可拒识别的直接方法学参照。
