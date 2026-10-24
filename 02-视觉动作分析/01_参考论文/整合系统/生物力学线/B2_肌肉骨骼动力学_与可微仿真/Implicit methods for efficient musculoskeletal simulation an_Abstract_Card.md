---
title: "Implicit methods for efficient musculoskeletal simulation and optimal control"
thread: B2
lens: M2
relevance: "低"
year: 2011
doi: "10.1016/j.piutam.2011.04.027"
arxiv_id: ""
evidence_grade: "A（≥2 个独立来源确认标题）"
confirm_sources: "crossref_doi, europepmc_doi, unpaywall"
---

# Implicit methods for efficient musculoskeletal simulation and optimal control

> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**
> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**
> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。

| 项 | 内容 |
|---|---|
| 年份 | 2011 |
| 期刊/会议 | Procedia IUTAM |
| DOI | `10.1016/j.piutam.2011.04.027` |
| arXiv | — |
| 线程 | B2 肌肉骨骼动力学_与可微仿真 |
| 证据范式 | M2 模型与仿真（肌肉骨骼模型、逆动力学、仿真器） |
| 相关度 | 低（人工裁定） |
| 证据等级 | A（≥2 个独立来源确认标题） |
| 交叉证实来源 | crossref_doi, europepmc_doi, unpaywall |
| 官方 OA | 是（hybrid） |
| 入口链接 | [https://doi.org/10.1016/j.piutam.2011.04.027](https://doi.org/10.1016/j.piutam.2011.04.027) |

## 摘要（来源原文，未翻译）

The ordinary differential equations for musculoskeletal dynamics are often numerically stiff and highly nonlinear. Consequently, simulations require small time steps, and optimal control problems are slow to solve and have poor convergence. In this paper, we present an implicit formulation of musculoskeletal dynamics, which leads to new numerical methods for simulation and optimal control, with the expectation that we can mitigate some of these problems. A first order Rosenbrock method was developed for solving forward dynamic problems using the implicit formulation. It was used to perform real-time dynamic simulation of a complex shoulder arm system with extreme dynamic stiffness. Simulations had an RMS error of only 0.11 degrees in joint angles when running at real-time speed. For optimal control of musculoskeletal systems, a direct collocation method was developed for implicitly formulated models. The method was applied to predict gait with a prosthetic foot and ankle. Solutions were obtained in well under one hour of computation time and demonstrated how patients may adapt their gait to compensate for limitations of a specific prosthetic limb design. The optimal control method was also applied to a state estimation problem in sports biomechanics, where forces during skiing were estimated from noisy and incomplete kinematic data. Using a full musculoskeletal dynamics model for state estimation had the additional advantage that forward dynamic simulations, could be done with the same implicitly formulated model to simulate injuries and perturbation responses. While these methods are powerful and allow solution of previously intractable problems, there are still considerable numerical challenges, especially related to the convergence of gradient-based solvers.

## 为什么收录（人工裁定理由）

隐式方法加速肌肉骨骼仿真
