---
title: "Physics-Informed Deep Learning for Musculoskeletal Modeling: Predicting Muscle Forces and Joint Kinematics From Surface EMG"
thread: B2
lens: M3
relevance: "高"
year: 2022
doi: "10.1109/tnsre.2022.3226860"
arxiv_id: ""
evidence_grade: "A（≥2 个独立来源确认标题）"
confirm_sources: "crossref_doi, europepmc_doi, unpaywall"
---

# Physics-Informed Deep Learning for Musculoskeletal Modeling: Predicting Muscle Forces and Joint Kinematics From Surface EMG

> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**
> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**
> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。

| 项 | 内容 |
|---|---|
| 年份 | 2022 |
| 期刊/会议 | IEEE Transactions on Neural Systems and Rehabilitation Engineering |
| DOI | `10.1109/tnsre.2022.3226860` |
| arXiv | — |
| 线程 | B2 肌肉骨骼动力学_与可微仿真 |
| 证据范式 | M3 方法与算法（学习模型、token 化、可微优化） |
| 相关度 | 高（人工裁定） |
| 证据等级 | A（≥2 个独立来源确认标题） |
| 交叉证实来源 | crossref_doi, europepmc_doi, unpaywall |
| 官方 OA | 是（gold） |
| 入口链接 | [https://doi.org/10.1109/tnsre.2022.3226860](https://doi.org/10.1109/tnsre.2022.3226860) |

## 摘要（来源原文，未翻译）

Musculoskeletal models have been widely used for detailed biomechanical analysis to characterise various functional impairments given their ability to estimate movement variables (i.e., muscle forces and joint moments) which cannot be readily measured in vivo. Physics-based computational neuromusculoskeletal models can interpret the dynamic interaction between neural drive to muscles, muscle dynamics, body and joint kinematics and kinetics. Still, such set of solutions suffers from slowness, especially for the complex models, hindering the utility in real-time applications. In recent years, data-driven methods have emerged as a promising alternative due to the benefits in speedy and simple implementation, but they cannot reflect the underlying neuromechanical processes. This paper proposes a physics-informed deep learning framework for musculoskeletal modelling, where physics-based domain knowledge is brought into the data-driven model as soft constraints to penalise/regularise the data-driven model. We use the synchronous muscle forces and joint kinematics prediction from surface electromyogram (sEMG) as the exemplar to illustrate the proposed framework. Convolutional neural network (CNN) is employed as the deep neural network to implement the proposed framework. Simultaneously, the physics law between muscle forces and joint kinematics is used the soft constraint. Experimental validations on two groups of data, including one benchmark dataset and one self-collected dataset from six healthy subjects, are performed. The experimental results demonstrate the effectiveness and robustness of the proposed framework.

## 为什么收录（人工裁定理由）

★★ 物理信息深度学习做肌肉骨骼建模（173 引，TNSRE）—— 物理约束与学习模型的既有桥梁
