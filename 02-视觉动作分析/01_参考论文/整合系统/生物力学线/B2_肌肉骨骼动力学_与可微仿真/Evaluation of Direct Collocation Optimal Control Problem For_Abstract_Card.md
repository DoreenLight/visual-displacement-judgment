---
title: "Evaluation of Direct Collocation Optimal Control Problem Formulations for Solving the Muscle Redundancy Problem"
thread: B2
lens: M2
relevance: "中"
year: 2016
doi: "10.1007/s10439-016-1591-9"
arxiv_id: ""
evidence_grade: "A（≥2 个独立来源确认标题）"
confirm_sources: "crossref_doi, europepmc_doi, unpaywall"
---

# Evaluation of Direct Collocation Optimal Control Problem Formulations for Solving the Muscle Redundancy Problem

> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**
> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**
> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。

| 项 | 内容 |
|---|---|
| 年份 | 2016 |
| 期刊/会议 | Annals of Biomedical Engineering |
| DOI | `10.1007/s10439-016-1591-9` |
| arXiv | — |
| 线程 | B2 肌肉骨骼动力学_与可微仿真 |
| 证据范式 | M2 模型与仿真（肌肉骨骼模型、逆动力学、仿真器） |
| 相关度 | 中（人工裁定） |
| 证据等级 | A（≥2 个独立来源确认标题） |
| 交叉证实来源 | crossref_doi, europepmc_doi, unpaywall |
| 官方 OA | 是（hybrid） |
| 入口链接 | [https://doi.org/10.1007/s10439-016-1591-9](https://doi.org/10.1007/s10439-016-1591-9) |

## 摘要（来源原文，未翻译）

Estimation of muscle forces during motion involves solving an indeterminate problem (more unknown muscle forces than joint moment constraints), frequently via optimization methods. When the dynamics of muscle activation and contraction are modeled for consistency with muscle physiology, the resulting optimization problem is dynamic and challenging to solve. This study sought to identify a robust and computationally efficient formulation for solving these dynamic optimization problems using direct collocation optimal control methods. Four problem formulations were investigated for walking based on both a two and three dimensional model. Formulations differed in the use of either an explicit or implicit representation of contraction dynamics with either muscle length or tendon force as a state variable. The implicit representations introduced additional controls defined as the time derivatives of the states, allowing the nonlinear equations describing contraction dynamics to be imposed as algebraic path constraints, simplifying their evaluation. Problem formulation affected computational speed and robustness to the initial guess. The formulation that used explicit contraction dynamics with muscle length as a state failed to converge in most cases. In contrast, the two formulations that used implicit contraction dynamics converged to an optimal solution in all cases for all initial guesses, with tendon force as a state generally being the fastest. Future work should focus on comparing the present approach to other approaches for computing muscle forces. The present approach lacks some of the major limitations of established methods such as static optimization and computed muscle control while remaining computationally efficient.

## 为什么收录（人工裁定理由）

直接配点法最优控制问题构造评测 —— 「优化式」路线的具体方法学
