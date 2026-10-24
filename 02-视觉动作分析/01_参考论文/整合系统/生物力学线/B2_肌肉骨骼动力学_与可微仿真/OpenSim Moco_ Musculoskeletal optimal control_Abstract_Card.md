---
title: "OpenSim Moco: Musculoskeletal optimal control"
thread: B2
lens: M2
relevance: "高"
year: 2020
doi: "10.1371/journal.pcbi.1008493"
arxiv_id: ""
evidence_grade: "A（≥2 个独立来源确认标题）"
confirm_sources: "crossref_doi, europepmc_doi, unpaywall"
---

# OpenSim Moco: Musculoskeletal optimal control

> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**
> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**
> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。

| 项 | 内容 |
|---|---|
| 年份 | 2020 |
| 期刊/会议 | PLoS Computational Biology |
| DOI | `10.1371/journal.pcbi.1008493` |
| arXiv | — |
| 线程 | B2 肌肉骨骼动力学_与可微仿真 |
| 证据范式 | M2 模型与仿真（肌肉骨骼模型、逆动力学、仿真器） |
| 相关度 | 高（人工裁定） |
| 证据等级 | A（≥2 个独立来源确认标题） |
| 交叉证实来源 | crossref_doi, europepmc_doi, unpaywall |
| 官方 OA | 是（gold） |
| 入口链接 | [https://doi.org/10.1371/journal.pcbi.1008493](https://doi.org/10.1371/journal.pcbi.1008493) |

## 摘要（来源原文，未翻译）

Musculoskeletal simulations are used in many different applications, ranging from the design of wearable robots that interact with humans to the analysis of patients with impaired movement. Here, we introduce OpenSim Moco, a software toolkit for optimizing the motion and control of musculoskeletal models built in the OpenSim modeling and simulation package. OpenSim Moco uses the direct collocation method, which is often faster and can handle more diverse problems than other methods for musculoskeletal simulation. Moco frees researchers from implementing direct collocation themselves-which typically requires extensive technical expertise-and allows them to focus on their scientific questions. The software can handle a wide range of problems that interest biomechanists, including motion tracking, motion prediction, parameter optimization, model fitting, electromyography-driven simulation, and device design. Moco is the first musculoskeletal direct collocation tool to handle kinematic constraints, which enable modeling of kinematic loops (e.g., cycling models) and complex anatomy (e.g., patellar motion). To show the abilities of Moco, we first solved for muscle activity that produced an observed walking motion while minimizing squared muscle excitations and knee joint loading. Next, we predicted how muscle weakness may cause deviations from a normal walking motion. Lastly, we predicted a squat-to-stand motion and optimized the stiffness of an assistive device placed at the knee. We designed Moco to be easy to use, customizable, and extensible, thereby accelerating the use of simulations to understand the movement of humans and other animals.

## 为什么收录（人工裁定理由）

★ OpenSim 官方最优控制模块，是「可微/优化路线」的事实标准工具
