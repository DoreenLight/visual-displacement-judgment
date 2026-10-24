---
title: "A Probabilistic Approach to Quantify the Impact of Uncertainty Propagation in Musculoskeletal Simulations"
thread: B2
lens: M2
relevance: "中"
year: 2014
doi: "10.1007/s10439-014-1181-7"
arxiv_id: ""
evidence_grade: "A（≥2 个独立来源确认标题）"
confirm_sources: "crossref_doi, europepmc_doi, unpaywall"
---

# A Probabilistic Approach to Quantify the Impact of Uncertainty Propagation in Musculoskeletal Simulations

> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**
> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**
> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。

| 项 | 内容 |
|---|---|
| 年份 | 2014 |
| 期刊/会议 | Annals of Biomedical Engineering |
| DOI | `10.1007/s10439-014-1181-7` |
| arXiv | — |
| 线程 | B2 肌肉骨骼动力学_与可微仿真 |
| 证据范式 | M2 模型与仿真（肌肉骨骼模型、逆动力学、仿真器） |
| 相关度 | 中（人工裁定） |
| 证据等级 | A（≥2 个独立来源确认标题） |
| 交叉证实来源 | crossref_doi, europepmc_doi, unpaywall |
| 官方 OA | 是（green） |
| 入口链接 | [https://doi.org/10.1007/s10439-014-1181-7](https://doi.org/10.1007/s10439-014-1181-7) |

## 摘要（来源原文，未翻译）

Uncertainty that arises from measurement error and parameter estimation can significantly affect the interpretation of musculoskeletal simulations; however, these effects are rarely addressed. The objective of this study was to develop an open-source probabilistic musculoskeletal modeling framework to assess how measurement error and parameter uncertainty propagate through a gait simulation. A baseline gait simulation was performed for a male subject using OpenSim for three stages: inverse kinematics, inverse dynamics, and muscle force prediction. A series of Monte Carlo simulations were performed that considered intrarater variability in marker placement, movement artifacts in each phase of gait, variability in body segment parameters, and variability in muscle parameters calculated from cadaveric investigations. Propagation of uncertainty was performed by also using the output distributions from one stage as input distributions to subsequent stages. Confidence bounds (5-95%) and sensitivity of outputs to model input parameters were calculated throughout the gait cycle. The combined impact of uncertainty resulted in mean bounds that ranged from 2.7° to 6.4° in joint kinematics, 2.7 to 8.1 N m in joint moments, and 35.8 to 130.8 N in muscle forces. The impact of movement artifact was 1.8 times larger than any other propagated source. Sensitivity to specific body segment parameters and muscle parameters were linked to where in the gait cycle they were calculated. We anticipate that through the increased use of probabilistic tools, researchers will better understand the strengths and limitations of their musculoskeletal simulations and more effectively use simulations to evaluate hypotheses and inform clinical decisions.

## 为什么收录（人工裁定理由）

不确定性传播的量化 —— 群体标签的误差上限论证
