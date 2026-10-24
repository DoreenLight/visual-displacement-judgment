---
title: "Computational performance of musculoskeletal simulation in <scp>OpenSim Moco</scp> using parallel computing"
thread: B2
lens: M2
relevance: "中"
year: 2023
doi: "10.1002/cnm.3777"
arxiv_id: ""
evidence_grade: "A（≥2 个独立来源确认标题）"
confirm_sources: "crossref_doi, europepmc_doi, unpaywall"
---

# Computational performance of musculoskeletal simulation in <scp>OpenSim Moco</scp> using parallel computing

> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**
> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**
> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。

| 项 | 内容 |
|---|---|
| 年份 | 2023 |
| 期刊/会议 | International Journal for Numerical Methods in Biomedical Engineering |
| DOI | `10.1002/cnm.3777` |
| arXiv | — |
| 线程 | B2 肌肉骨骼动力学_与可微仿真 |
| 证据范式 | M2 模型与仿真（肌肉骨骼模型、逆动力学、仿真器） |
| 相关度 | 中（人工裁定） |
| 证据等级 | A（≥2 个独立来源确认标题） |
| 交叉证实来源 | crossref_doi, europepmc_doi, unpaywall |
| 官方 OA | 是（hybrid） |
| 入口链接 | [https://doi.org/10.1002/cnm.3777](https://doi.org/10.1002/cnm.3777) |

## 摘要（来源原文，未翻译）

Optimal control musculoskeletal simulation is a valuable approach for studying fundamental and clinical aspects of human movement. However, the high computational demand has long presented a substantial challenge, creating a need to improve simulation performance. The OpenSim Moco software package permits musculoskeletal simulation problems to be solved in parallel on multicore processors using the CasADi optimal control library, potentially reducing the computational demand. However, the computational performance of this framework has not been thoroughly examined. Thus, we aimed to investigate the computational speed-up obtained via multicore parallel computing relative to solving problems serially (i.e., using a single core) in optimal control simulations of human movement in OpenSim Moco. Simulations were solved using up to 18 cores with a variety of temporal mesh interval densities and using two different initial guess strategies. We examined a range of musculoskeletal models and movements that included two- and three-dimensional models, tracking and predictive simulations, and walking and reaching tasks. The maximum overall parallel speed-up was problem specific and ranged from 1.7 to 7.7 times faster than serial, with most of the speed-up achieved by about 6 processor cores. Parallel speed-up was generally greater on finer temporal meshes, while the initial guess strategy had minimal impact on speed-up. Considerable speed-up can be achieved for some optimal control simulation problems in OpenSim Moco by leveraging the multicore processors often available in modern computers. However, since improvements are problem specific, achieving optimal computational performance will require some degree of exploration by the end user.

## 为什么收录（人工裁定理由）

OpenSim Moco 的计算性能 —— 可行性核算
