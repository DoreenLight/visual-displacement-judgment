# 章隆彬 — 旧批次13篇资料介绍

> **当前入口（2026-10-08）**：[目录导航](README.md) · [现行台账](资料核查/README.md) · [链路索引](链路索引.md) · [新增缺口](PAPER_GAP_LIST.md)。
> **实际迁移更新**：论文已分入`papers/运动理解/`、`papers/康复辅助/`、`papers/理解与辅助结合/`、`papers/背景资料/`；旧检索运行已整体移至`历史归档/_research_run/`，专家名单位于[团队资料](团队资料/experts.yaml)。下文历史短路径描述当时布局，当前位置以[现行台账](资料核查/README.md)为准。
> 下文保留2026-10-01批次介绍，不是当前全部成果/原件状态。旧13份PDF中Gait & Posture条目仅1页摘要；开放补取5份PDF＋1份XML、用户补齐Z20—Z25六份PDF，当前24份PDF＋1份XML；六篇缺口关闭，仅Z26在投稿需团队分享。
> TMM应区分在线2023与卷期2024，TUG应区分在线2025与本地卷期2026；踝力矩TASE DOI已从原件核得`10.1109/TASE.2020.3033664`。旧表“总下载13/13”仅指旧批次，不表示教授文献全集已经补齐。

> **更新日期**：2026-09-30 → **2026-10-01 增补（7 篇学校库下载）**
> **作者**：Longbin Zhang（章隆彬，Hunan University, KTH PhD）
> **OpenAlex 作者 ID**：`A5002794703`（⚠️ 存在同名污染，混入真菌漆酶/肿瘤标志物等无关论文，**不可当全集用**）
> **总下载**：**13/13 篇目标论文全部已获取原文**（首批 6 篇 + 2026-10-01 增补 7 篇）
> **PDF 路径**：`09_章隆彬_合作团队/papers/`（ICARM 2023 一篇在你放置时落在 09 目录根下）
> **调研语料**：13 篇全文均已由 `pdftotext -layout` 注入 `_research_run/.../raw/` 并参与九段闭环
> **翻译路径**：各篇目录内的 `<目录名>_全文翻译.md`（**13/13 篇已全译**）
> 📌 **2026-10-03 订正**：本行原写「首批 6 篇已全译；增补 7 篇**待译**」—— **该记述是错的**。
> 实测 `papers/` 顶层本来就躺着 7 个 `*_zh.md`（8,492–59,113 字符，与已译 6 篇的全文翻译 18,165–65,363 同量级），
> **7 篇早就译完了**，只是当时用了旁路命名 `*_zh.md`，而 `verify_kb.sh` Case 7 只认 `<stem>_全文翻译.md`。
> 已全部归位到各篇目录并统一命名（见「一、论文总览」的「翻译」列）。
> 另：TASE 篇的译文中有一处 `![...](placeholder)` 死链，因文件名不叫 `_全文翻译.md` 而从未被 Case 2 扫到；
> 已从原文第 4 页抽出 Fig. 3 真图（831×604）并改写引用。

---

## 一、论文总览（13 篇）

| # | 论文 | 期刊/会议 | 年 | DOI | 原文 | 翻译 | 与 ViKiT 相关性 |
|---|---|---|---|---|---|---|---|
| 1 | Towards Clinical Application of Enhanced TUG With Markerless MoCap and ML for Balance/Gait Assessment | **IEEE JBHI** | 2025 | [10.1109/jbhi.2025.3543095](https://doi.org/10.1109/jbhi.2025.3543095) | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| 2 | **Global and Local Spatio-Temporal Encoder for 3D Human Pose Estimation** | **IEEE TMM** | 2024 | [10.1109/tmm.2023.3321438](https://doi.org/10.1109/tmm.2023.3321438) | ✅ 11p | ✅ | ⭐⭐⭐⭐⭐ **最对口** |
| 3 | Design and Evaluation of a Single-Sided Mobility Assistive Exoskeleton (SMAEXO) for Hemiplegia | **IEEE ICORR** | 2025 | [10.1109/icorr66766.2025.11063171](https://doi.org/10.1109/icorr66766.2025.11063171) | ✅ 6p | ✅ | ⭐⭐⭐⭐ |
| 4 | Muscle Activation and Postural Sway in Response to Task Complexity: A Study of Balance Control in Older Adults | **IEEE ICORR** | 2025 | [10.1109/icorr66766.2025.11063123](https://doi.org/10.1109/icorr66766.2025.11063123) | ✅ 6p | ✅ | ⭐⭐⭐⭐ |
| 5 | Knee Joint Torque Prediction with Uncertainties by a Neuromusculoskeletal Solver-informed Gaussian Process Model | **IEEE ICARM** | 2023 | [10.1109/icarm58088.2023.10218934](https://doi.org/10.1109/icarm58088.2023.10218934) | ✅ | ✅ | ⭐⭐⭐⭐ |
| 6 | Ankle Joint Torque Estimation Using an EMG-Driven NMS Model and an ANN Model | **IEEE TASE** | 2021 | [10.1109/tase.2020.3033664](https://doi.org/10.1109/tase.2020.3033664) | ✅ 10p | ✅ | ⭐⭐⭐⭐（KTH 路线起点） |
| 7 | EMG-Constrained and Ultrasound-Informed Muscle-Tendon Parameter Estimation in Post-Stroke Hemiparesis | **IEEE TBME** | 2024 | [10.1109/tbme.2024.3352556](https://doi.org/10.1109/tbme.2024.3352556) | ✅ | ✅ | ⭐⭐⭐⭐ |
| 8 | Estimation of Joint Torque by EMG-Driven NMS Models and LSTM Networks | **IEEE TNSRE** | 2023 | [10.1109/tnsre.2023.3315373](https://doi.org/10.1109/tnsre.2023.3315373) | ✅ | ✅ | ⭐⭐⭐⭐ |
| 9 | Ankle Joint Torque Prediction Using an NMS Solver Informed-ANN Model and Transfer Learning | **IEEE JBHI** | 2022 | [10.1109/jbhi.2022.3207313](https://doi.org/10.1109/jbhi.2022.3207313) | ✅ | ✅ | ⭐⭐⭐⭐ |
| 10 | Lower-Limb Joint Torque Prediction Using LSTM Neural Networks and Transfer Learning | **IEEE TNSRE** | 2022 | [10.1109/tnsre.2022.3156786](https://doi.org/10.1109/tnsre.2022.3156786) | ✅ | ✅ | ⭐⭐⭐ |
| 11 | NMS Model-Informed ML-Based Control of a Knee Exoskeleton With Uncertainties Quantification | **Front Neurosci** | 2023 | [10.3389/fnins.2023.1254088](https://doi.org/10.3389/fnins.2023.1254088) | ✅ | ✅ | ⭐⭐⭐⭐ |
| 12 | Do gait patterns normalize concurrently with the recovery of foot arches in children with flatfoot? A prospective cohort study | **J Biomech** | 2025 | [10.1016/j.jbiomech.2025.112758](https://doi.org/10.1016/j.jbiomech.2025.112758) | ✅ 9p | ✅ | ⭐⭐⭐（临床步态落地） |
| 13 | An approach to establishing the thresholds of plantar loading in obese children | **Gait & Posture** | 2023 | [10.1016/j.gaitpost.2023.07.273](https://doi.org/10.1016/j.gaitpost.2023.07.273) | ✅ 1p 摘要 | ✅ | ⭐⭐（会议摘要，1 页） |

---

## 二、2026-10-01 增补的 7 篇：要点与合作网络

### 2.1 ⭐⭐⭐⭐⭐ TMM 2024 — 3D 人体姿态估计（与本课题最对口）

**Global and Local Spatio-Temporal Encoder for 3D Human Pose Estimation**
Yong Wang¹, Hongbo Kang¹, Doudou Wu¹, Wenming Yang², Longbin Zhang³
¹School of AI, Chongqing University of Technology ｜ ²Shenzhen International Graduate School ｜ ³Hunan University

- **问题**：Transformer 做 3D 人体姿态估计时，多数只编码"全身所有关节的全局时空相关性"，**单个关节的局部时空相关性**少有人建模。
- **方法**：提出 **GLSTE**（Global and Local Spatio-Temporal Encoder）——GSE（全局空间编码器）+ 对每个关节的局部时空编码，建模两者相关性。
- **对你的价值**：这是她**唯一一篇纯计算机视觉方向**的论文，也是她与国内 CV 圈（重理工人工智能学院 + 深国际研究生院 杨文明）的接口。ViKiT 做「RGB → 关节角」时，GLSTE 的"全局骨架相关 + 单关节局部时序"双路结构是可直接对照/借鉴的架构参照；Yong Wang 与 Hongbo Kang 对本文贡献相同，通讯作者为 Kang。

### 2.2 ⭐⭐⭐⭐ ICORR 2025 — SMAEXO 单侧助行外骨骼（**严小月一作**）

**Design and Evaluation of a Single-Sided Mobility Assistive Exoskeleton (SMAEXO) for Hemiplegia**
Xiaoyue Yan¹², Wei Yun Yau², Isaac Okumura Tan², **Longbin Zhang**²⁴, Youlong Wang¹², Oliver Roberts², Seng Kwee Wee³, Wei Tech Ang¹²
¹Hunan University ｜ ²Nanyang Technological University ｜ ³（NTU 系统）

- **对象**：单侧、多关节外骨骼，面向偏瘫（hemiplegia）患者。
- **设计**：**髋+膝主动助力 + 踝关节被动柔性（防足下垂）**，覆盖整个步态周期；优化反作用力支撑与重量分配，减轻患侧负担；模块化、可定制以适配不同体型。
- **意义**：这是**她学生严小月的一作**，标志团队已能独立产出整机设计类工作（不再只是"用外骨骼做控制算法"）。

### 2.3 ⭐⭐⭐⭐ ICORR 2025 — 肌电与姿态摆动（她一作）

**Muscle Activation and Postural Sway in Response to Task Complexity: A Study of Balance Control in Older Adults**
Longbin Zhang¹²³, Ananda Sidarta², Yu Chin Lim², Carol Er², Xiaoyue Yan²³, Tsung-Lin Wu²³, Wei Tech Ang²³

- **实验**：7 名老年人，4 种站姿任务（双脚站立、并脚站立、并脚+认知任务、认知+运动复合任务 TSCM），同步记录下肢与躯干肌电 + CoP（压力中心）特征。
- **意义**：把她的 sEMG 主线与"平衡控制/跌倒风险"评估接起来，是 TUG markerless 那条线的**神经肌肉侧补充**。

### 2.4 ⭐⭐⭐⭐ ICARM 2023 — NMS 求解器 + 高斯过程（不确定性）

**Knee Joint Torque Prediction with Uncertainties by a Neuromusculoskeletal Solver-informed Gaussian Process Model**
Longbin Zhang, Xiaochen Zhang, Xueyu Zhu, Ruoli Wang, Elena M. Gutierrez-Farewik

- **动机**：外骨骼助力策略需要用户的力矩能力估计，但**力矩能力预测的不确定性直接影响人机界面安全**。
- **方法**：**NMS-GP**——以肌电 + 关节运动学为输入，用 NMS 求解器给高斯过程做信息先验，输出带不确定性的膝屈伸力矩。
- **意义**：Front Neurosci 2023 那篇 UQ 的会议版/方法前身；"物理求解器为 GP 提供先验"是很有辨识度的方法论。

### 2.5 ⭐⭐⭐⭐ TASE 2021 — NMS + ANN 踝关节力矩（KTH 阶段代表作）

**Ankle Joint Torque Estimation Using an EMG-Driven Neuromusculoskeletal Model and an Artificial Neural Network Model**
Longbin Zhang, Zhijun Li, Yingbai Hu, Christian Smith, Elena M. Gutierrez Farewik, Ruoli Wang
IEEE TASE 18(2): 564–573, 2021

- **动机**：助力外骨骼要补偿用户"残余肌肉贡献"，需要实时估计踝关节力矩。
- **方法**：EMG 驱动的 NMS 模型 + ANN 估计踝关节力矩。
- **意义**：**她"NMS + 机器学习混合模型"路线的起点**（JBHI 2022 / TNSRE 2022 / TNSRE 2023 都是它的延伸）；合作者含 Ruoli Wang（KTH，现已独立）。

### 2.6 ⭐⭐⭐ J Biomech 2025 — 扁平足足弓恢复与步态（四川大学合作）

**Do gait patterns normalize concurrently with the recovery of foot arches in children with flatfoot? A prospective cohort study**
Yihong Zhao¹, Shiyang Yan¹, **Longbin Zhang**², Bi Qing Shi¹, Luming Yang¹
¹四川大学 轻工清洁技术与工程研究中心 / 智能服装与运动生物力学实验室 ｜ ²湖南大学机器人学院

- **意义**：她把 markerless 动捕能力用到**儿童扁平足临床随访**上；这是"机器人学院 × 轻工/运动生物力学"的跨学科合作样本，通讯作者为 Luming Yang。

### 2.7 ⭐⭐ Gait & Posture 2023 — 肥胖儿童足底压力阈值（会议摘要，1 页）

**An approach to establishing the thresholds of plantar loading in obese children**
Shiyang Yan¹, Yihong Zhao¹, **Longbin Zhang**², Luming Yang¹
ESMAC 2023 Abstract Book, Gait & Posture 106: S229

- **意义**：会议摘要（1 页），可作为方法论注脚；正刊版可能另在他处，如需要可再检索。

---

## 三、这批论文揭示的合作网络（已同步进 `experts.yaml`，共 25 条目）

| 合作者 | 单位 | 关系 | 关联论文 |
|---|---|---|---|
| Hongbo Kang / Yong Wang / Doudou Wu | 重庆理工大学 人工智能学院 | 3D 姿态估计线通讯/一作 | TMM 2024 |
| Wenming Yang | 深圳国际研究生院（IEEE SM） | CV 领域高层合作人 | TMM 2024 |
| Xiaoyue Yan | 湖南大学 + NTU | **团队学生，已独立一作** | ICORR 2025 ×2、arXiv 2503.00794 |
| Seng Kwee Wee / Wei Yun Yau / Oliver Roberts | NTU | 外骨骼工程线 | ICORR 2025 SMAEXO |
| Luming Yang / Yihong Zhao / Shiyang Yan | 四川大学 轻工中心/运动生物力学实验室 | 临床步态/足底力学线 | J Biomech 2025、Gait&Posture 2023 |
| Ruoli Wang | KTH 时期 → 现独立 | 最稳定的长期合作者 | TASE 2021、TNSRE 2022/2023、Front Neurosci 2023、ICARM 2023 |
| Xueyu Zhu / Xiaochen Zhang | 章团队 / 合作 | NMS 求解 + GP 方向 | ICARM 2023 |

> 三条清晰的**能力支线**：① 生物力学建模与力矩估计（主线）；② 康复外骨骼整机与临床步态（2024→2025 明显前移）；③ **计算机视觉 3D 姿态估计**（唯一一条 CV 线，对本课题价值最高）。

---

## 四、待办

- [x] ~~增补 7 篇的中文翻译~~ —— **2026-10-03 发现这 7 篇本来就有译文**（`papers/*_zh.md`），已归位；本条原记载有误
- [ ] 13 篇的**翻译导读**（`_翻译导读.md`）：已译的 6 篇有，2026-10-03 归位的 7 篇尚缺（`FULLTEX` 已全，不阻断）
- [x] ~~ICARM 2023 的 PDF 归位~~ —— 2026-10-03 已并入 `papers/Zhang2023-ICARM：…/` 并按目录名规范化
- [ ] 复查 J Biomech 2025 / Gait&Posture 2023 是否另有更完整正刊版
- [ ] 补 OpenAlex 作者 handle（匿名 API 额度 2026-10-01 当日已用尽，UTC 午夜重置后再查）

---

## 五、来源

- 湖南大学官方主页：<https://robotics.hnu.edu.cn/info/1176/3002.htm>
- OpenAlex 作者页：<https://api.openalex.org/authors/A5002794703>（⚠️ 同名污染）
- IEEE Xplore 作者页：<https://ieeexplore.ieee.org/author/37088232279>
- 本次调研语料：`_research_run/zhang-.../raw/86..92-localpdf-*.md`（7 篇新增全文）
