---
title: "Monocular Biomechanical Tracking of Fingers with Inverse Kinematics to Foundation Models"
thread: B1
lens: M3
relevance: "高"
year: 2026
doi: ""
arxiv_id: "2605.09258"
evidence_grade: "B（单来源确认标题，>=0.90）"
confirm_sources: "arxiv"
---

# Monocular Biomechanical Tracking of Fingers with Inverse Kinematics to Foundation Models

> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**
> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**
> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。

| 项 | 内容 |
|---|---|
| 年份 | 2026 |
| 期刊/会议 | — |
| DOI | — |
| arXiv | `2605.09258` |
| 线程 | B1 视觉运动学_估计与验证方法学 |
| 证据范式 | M3 方法与算法（学习模型、token 化、可微优化） |
| 相关度 | 高（人工裁定） |
| 证据等级 | B（单来源确认标题，>=0.90） |
| 交叉证实来源 | arxiv |
| 官方 OA | 是（arXiv 预印本（官方 OA）） |
| 入口链接 | [https://arxiv.org/abs/2605.09258](https://arxiv.org/abs/2605.09258) |

## 摘要（来源原文，未翻译）

Accurate hand and finger tracking from video has significant clinical applications for monitoring activities of daily living and measuring range of motion, yet monocular video approaches for obtaining hand biomechanics remain under-developed. We present a method that combines the SAM 3D Body foundation model with inverse kinematics optimization in a full-body biomechanical model to extract anatomically-constrained finger joint angles from single-view video. We port SAM 3D Body from PyTorch to JAX for integration with MuJoCo-MJX, enabling GPU-accelerated optimization, and develop a novel mapping between the Momentum Human Rig (MHR) outputs and biomechanical model markers. Validation against 8-camera multiview reconstruction on 4,590 frames from 7 participants performing a variety of hand poses and object manipulation tasks shows finger joint angle errors of approximately 10 degrees and hand position errors of approximately 6 mm, after Procrustes alignment. Results were consistent across camera viewpoints and robust to different methods for producing reference values from multiview video. This work extends monocular biomechanical analysis to detailed finger tracking, expanding access to quantitative characterization of hand movement from readily available video.

## 为什么收录（人工裁定理由）

★★★ 单目 + 生物力学模型在优化环内的手/指关节角跟踪（SAM 3D Body + MuJoCo-MJX 下的 IK 优化；Shirley Ryan AbilityLab / Northwestern）。**本条由 snowball 追入** —— 它是 `2607.02796` 参考文献 [50]，而 `2607.02796` 当时被列为「最近的先行」，其引用网络此前从未追过。**为什么它比 `2607.02796` 更该警惕**：设定上**同为单目**（该文 monocular 出现 28 次），而 `2607.02796` 是 8 相机多视角。🚨 **全文级撞红已做（12 页全文，2026-10-02）**：`token` / `vocabulary` / `codebook` / `symbol` / `discrete` / `quantiz` / `embedding` / `Transformer` / `grammar` / `language model` / `autoregressive` **全部零命中**；4 次 `translation` 经逐条查上下文**全部是「root translation / 平移量」，非序列翻译**。→ **不构成机制级撞车**，但**「单目 + 生物力学约束」这个组合已被占**，写贡献时不可再当新意。📌 **可引的空白陈述（注意其限定）**：摘要原句 *"monocular video approaches for obtaining **hand** biomechanics remain **under-developed**"* —— 🚨 **它说的是 hand/finger，不可外推到上肢 ISB 角度**。
