# 宋嵘团队论文清单（24 篇，四段分类）

> **当前入口**：[目录导航](README.md) · [链路索引](链路索引.md) · [P0三篇精读](资料核查/P0三篇精读_阻塞复核.md) · [扩充语料](资料核查/扩充语料_官方双页核对.md) · [论文原件区](papers/README.md)
> **生成日期**：2026-10-08。
> **每条 DOI 均已独立复核**（题名精确匹配 + Crossref/PubMed/IEEE/出版方页面），可直接用于取件。
> **注意**：全项目唯一取件入口仍是 [`06_附录/待下载论文总清单.md`](../06_附录/待下载论文总清单.md)。本文件是**宋嵘专项**视图，两者不重复：前者由 `gen_acquisition_list.py` 从台账生成，本文件是团队定向补充。

---

## 一、P0 必取（3 篇）—— 能实质改变 C1/C3 判定

| # | 论文 | 期刊/年份 | DOI | 为什么必取 |
|---|---|---|---|---|
| 1 | Iterative Adjustment of Stimulation Timing and Intensity During FES-Assisted Treadmill Walking for Patients After Stroke | IEEE TNSRE, 2020, 28(6):1292-1298 | `10.1109/TNSRE.2020.2986295` | **10 名卒中患者 + 自适应 FES + 强度/时相调整 + 有对照**。全台账里唯一同时具备"患者/真实辅助/强度自适应"三者的一篇，直接决定 C3 能否解除 |
| 2 | Corticomuscular Coupling Alterations During Elbow Isometric Contraction Correlated With Clinical Scores: An fNIRS-sEMG Study in Stroke Survivors | IEEE TNSRE, 2025, 33:696-704 | `10.1109/TNSRE.2025.3535928` | 12 卒中 + 12 对照，**上肢肘部**，耦合指标与 **FMA 临床评分**相关。C1 缺的独立心理/神经参照，最对口的一篇 |
| 3 | Isometric Plantarflexion Moment Prediction Based on a Compartment-Specific HD-sEMG-Driven Musculoskeletal Model | IEEE TBME, 2024, 71(8):2311-2320 | `10.1109/TBME.2024.3368021` | 提供**不依赖机器人交互力**的力学参照路径，C1 物理参照候选。需核清部位迁移与"预测力矩≠主动贡献"的边界 |

---

## 二、P1 重要（3 篇）

| # | 论文 | 期刊/年份 | DOI | 取件目的 |
|---|---|---|---|---|
| 4 | Hybrid and adaptive control of functional electrical stimulation to correct hemiplegic gait for patients after stroke | Front Bioeng Biotechnol, 2023, 11:1246014 | `10.3389/fbioe.2023.1246014` | **开放获取**（PMC10441235），**可能无需学校库**。FES 混合自适应，C3 对照条件 |
| 5 | Practical Multiple Model Switching sEMG Decoders for Motion Intention Description in Rehabilitation Robot Control | IEEE TMRB, 2026, 8(2):785-797 | `10.1109/TMRB.2026.3682646` | 该团队**最新**方法（2026），多模型无扰切换解码。影响"运动理解"段的基线选择与查新 |
| 6 | Stroke-induced alteration in multi-layer information transmission of cortico-motor system during elbow isometric contraction modulated by myoelectric-controlled interfaces | J Neural Eng, 2021, 18(4):0460e1 | `10.1088/1741-2552/ac18ae` | 15 健康 + 11 卒中，改变肌电接口**维度**看三层信息传输。为 C1 提供"可干预参照"范式 |

---

## 三、P2 补充（4 篇）

| # | 论文 | 期刊/年份 | DOI | 取件目的 |
|---|---|---|---|---|
| 7 | An Electromyographic-Based Control Using Gaussian Mixture Model on an Upper-Limb Cable-Driven Rehabilitation Robot | Adv Intell Syst, 2025, 7(4):2400505 | `10.1002/aisy.202400505` | **上肢**、GMM 抗疲劳增量校准。C2 的"重标定触发条件"依据 |
| 8 | An Adaptive Hammerstein Model for FES-Induced Torque Prediction Based on Variable Forgetting Factor Recursive Least Squares Algorithm | IEEE TNSRE, 2024, 32:1109-1118 | `10.1109/TNSRE.2024.3371465` | 真实辅助下的在线力矩预测，为 N8 的建模基础 |
| 9 | Trajectory Deformation-Based Multi-Modal Adaptive Compliance Control for a Wearable Lower Limb Rehabilitation Robot | IEEE TNSRE, 2024, 32:314-324 | `10.1109/TNSRE.2023.3348332` | 人主导/机器人主导/软停三模式，"按能力调辅助"范式 |
| 10 | Voluntary Control of an Ankle Joint Exoskeleton by Able-Bodied Individuals and Stroke Survivors Using EMG-Based Admittance Control Scheme | IEEE TBME, 2021, 68(2):695-705 | `10.1109/TBME.2020.3012296` | 健康+卒中混编，主动控制比例。既有 Song_VAAN 的更早同线版本 |

---

## 四、P3 补覆盖密度（6 篇）

| # | 论文 | 期刊/年份 | DOI |
|---|---|---|---|
| 11 | Modeling Ankle Torque and Stiffness Induced by Functional Electrical Stimulation | IEEE TNSRE, 2020, 28(12):3013-3021 | `10.1109/TNSRE.2020.3042221` |
| 12 | Human Motion Intent Description Based on Bumpless Switching Mechanism for Rehabilitation Robot | IEEE TNSRE, 2021, 29:673-682 | `10.1109/TNSRE.2021.3066592` |
| 13 | Continuous Description of Human 3D Motion Intent Through Switching Mechanism | IEEE TNSRE, 2020, 28(1):277-286 | `10.1109/TNSRE.2019.2949203` |
| 14 | Modulating and restoring inter-muscular coordination in stroke patients using two-dimensional myoelectric computer interface | J Neural Eng, 2021, 18(3):036005 | `10.1088/1741-2552/abc29a` |
| 15 | Phase-Dependent Modulation of Muscle Activity and Intermuscular Coupling During Walking in Patients After Stroke | IEEE TNSRE, 2023, 31:1119-1130 | `10.1109/TNSRE.2023.3238758` |
| 16 | Investigating the Stroke- and Aging-Related Changes in Global and Instantaneous Intermuscular Coupling Using Cross-Fuzzy Entropy | IEEE TNSRE, 2021, 29:1573-1582 | `10.1109/TNSRE.2021.3101615` |

---

## 五、已核实 / 本地已有（不需要下载）

以下 8 篇**本地已有原件**，无需取件，详见 [LAB_PROFILE](团队资料/LAB_PROFILE.md) §7：

| ID | 本地原件 | 状态 |
|---|---|---|
| Song_TRO | PDF + 证据卡 | 已有 |
| Song_TIE | PDF + 证据卡 | 已有 |
| Song_TBCAS | PDF + 证据卡 | 已有 |
| Song_impedance | XML | 已有 |
| Song_wrist | XML | 已有 |
| Song_VAAN | PDF + 证据卡 | 已有 |
| Song_load | XML | 已有 |
| Song_HBMEhybrid（=既有第 7 篇） | PDF + 证据卡 | 已有 |

---

## 六、取件后必须做的事

取回 PDF 后，**不要**直接改写既有结论。按顺序：

1. 存放到 `01_参考论文/康复辅助/康复整合链路/`，命名沿用该目录既有风格（长题名 + `_Evidence_Card.md` 配证据卡）。
2. 逐篇补 [文献对照表](../10_研究方向/主动参与与训练强度_链路研究/文献对照表.md) 的输入/输出/反馈/参与/强度/安全分栏，并回填 [文献对照_原文定位.json](../10_研究方向/主动参与与训练强度_链路研究/文献对照_原文定位.json)。
3. 依新证据**重判** [链路映射与缺口裁决](链路索引.md) 的 C1–C3 结论，特别是：
   - N1 的耦合指标是否真能当"心理参与"参照，还是只能当神经机制指标；
   - N6 的力矩模型能否当"主动贡献"独立参照；
   - N8 的自适应 FES 是否构成"不同辅助暴露下有效主动练习量"的对照。
4. 确认 P0 三篇**不足以**支撑创新定案后，才考虑更新"0 条定案"。

---

## 七、边界

- 本清单**只含题录已核**的条目；未核 DOI 的候选一律不列，避免错误取件。
- 分层是**取件优先级**，不是**重要性/价值排名**，也不是发表价值排序。
- 官方双页仅列"代表性论文"，**本清单不穷尽**宋嵘全部论文。
- 专利不在本清单（无 DOI，需另行申请），线索见 [扩充语料 §五](资料核查/扩充语料_官方双页核对.md)。
- 清单的存在**不代表**这些论文能解除任何阻塞；那取决于全文内容。

---

## 相关文件

- [扩充语料_官方双页核对.md](资料核查/扩充语料_官方双页核对.md) — 16 篇完整题录与分组说明
- [链路映射与缺口裁决.md](链路索引.md) — 优先级依据
- [LAB_PROFILE.md](团队资料/LAB_PROFILE.md) — 团队画像
- [全项目唯一取件清单](../06_附录/待下载论文总清单.md)