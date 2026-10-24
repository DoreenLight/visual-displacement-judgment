---
title: "Machine Learning for Optical Motion Capture-Driven Musculoskeletal Modelling from Inertial Motion Capture Data"
thread: B5
lens: M0
relevance: "中"
year: 2023
doi: "10.3390/bioengineering10050510"
arxiv_id: ""
evidence_grade: "A（≥2 个独立来源确认标题）"
confirm_sources: "crossref_doi, europepmc_doi, unpaywall"
---

# Machine Learning for Optical Motion Capture-Driven Musculoskeletal Modelling from Inertial Motion Capture Data

> ⚠️ **本文件是摘要级定位卡，不是全文翻译。**
> 全部判断仅基于标题、摘要与官方元数据。**引用任何数值、方法细节或结论前，必须回原文核对。**
> 未获取原文者，该条只能作为「线索」使用，不得写入论文结论。

| 项 | 内容 |
|---|---|
| 年份 | 2023 |
| 期刊/会议 | Bioengineering |
| DOI | `10.3390/bioengineering10050510` |
| arXiv | — |
| 线程 | B5 视觉运动学到动力学_桥接 |
| 证据范式 | M0 综述·标准·证据汇总 |
| 相关度 | 中（人工裁定） |
| 证据等级 | A（≥2 个独立来源确认标题） |
| 交叉证实来源 | crossref_doi, europepmc_doi, unpaywall |
| 官方 OA | 是（gold） |
| 入口链接 | [https://doi.org/10.3390/bioengineering10050510](https://doi.org/10.3390/bioengineering10050510) |

## 摘要（来源原文，未翻译）

Marker-based Optical Motion Capture (OMC) systems and associated musculoskeletal (MSK) modelling predictions offer non-invasively obtainable insights into muscle and joint loading at an in vivo level, aiding clinical decision-making. However, an OMC system is lab-based, expensive, and requires a line of sight. Inertial Motion Capture (IMC) techniques are widely-used alternatives, which are portable, user-friendly, and relatively low-cost, although with lesser accuracy. Irrespective of the choice of motion capture technique, one typically uses an MSK model to obtain the kinematic and kinetic outputs, which is a computationally expensive tool increasingly well approximated by machine learning (ML) methods. Here, an ML approach is presented that maps experimentally recorded IMC input data to the human upper-extremity MSK model outputs computed from ('gold standard') OMC input data. Essentially, this proof-of-concept study aims to predict higher-quality MSK outputs from the much easier-to-obtain IMC data. We use OMC and IMC data simultaneously collected for the same subjects to train different ML architectures that predict OMC-driven MSK outputs from IMC measurements. In particular, we employed various neural network (NN) architectures, such as Feed-Forward Neural Networks (FFNNs) and Recurrent Neural Networks (RNNs) (vanilla, Long Short-Term Memory, and Gated Recurrent Unit) and a comprehensive search for the best-fit model in the hyperparameters space in both subject-exposed (SE) as well as subject-naive (SN) settings. We observed a comparable performance for both FFNN and RNN models, which have a high degree of agreement (ravg,SE,FFNN=0.90±0.19, ravg,SE,RNN=0.89±0.17, ravg,SN,FFNN=0.84±0.23, and ravg,SN,RNN=0.78±0.23) with the desired OMC-driven MSK estimates for held-out test data. The findings demonstrate that mapping IMC inputs to OMC-driven MSK outputs using ML models could be instrumental in transitioning MSK modelling from 'lab to field'.

## 为什么收录（人工裁定理由）

动捕驱动的肌肉骨骼建模之机器学习综述
