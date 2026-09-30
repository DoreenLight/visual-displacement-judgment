# 章隆彬 et al. JBHI 2025 — 内容速读卡

> ## 🔔 注意：本速读卡已被原文级精读取代
>
> ⚠️ **本卡含 5 处元数据错误**：① 一作机构误写 KTH（实为湖南大学）；② 「Audrey Sidarta」应为 Ananda；③ 「Tsung-Li Wu」应为 Tsung-Lin；④ 缺 3 位作者（Aoyang Long / Xiangyu Long / Wei Tech Ang）；⑤ 未标卷期页（Vol.30 No.8 pp.7072-7081）。另「5 段 TUG 阶段」原为推测，已由原文确认。
>
> | 文件 | 权威度 |
> |---|---|
> | [`papers/Zhang2025-JBHI：无标记动捕与 ML 增强的 TUG 平衡步态评估/Zhang2025-JBHI：无标记动捕与 ML 增强的 TUG 平衡步态评估.pdf`](Zhang2025-JBHI：无标记动捕与 ML 增强的 TUG 平衡步态评估.pdf) | ✅ **原文 PDF（权威）** |
> | [`papers/Zhang2025-JBHI：无标记动捕与 ML 增强的 TUG 平衡步态评估/Zhang2025-JBHI：无标记动捕与 ML 增强的 TUG 平衡步态评估_全文翻译.md`](Zhang2025-JBHI：无标记动捕与 ML 增强的 TUG 平衡步态评估_全文翻译.md) | ✅ **基于原文的全译级精读 —— 请优先阅读这个** |
> | **本速读卡** | ⚠️ **速查用**；与上述文件冲突时，**一律以后者为准** |

---


> ✅ **状态**：**PDF 已获取**（原文已落盘于同目录）。本卡的元数据与
> 推测数值**已被原文推翻部分**——请以文件顶部的「原文级精读」为准。
> 本文档基于 OpenAlex 摘要完整重建（1899 字）+ Crossref 元数据 + 作者团队前期工作推导。
> **作用**：速查卡（原文已于 2026-09-30 获取）。**精读请以文件顶部的「原文级精读」为准**。
>
> **下载后请把 PDF 改名为 `Zhang_2025_JBHI_MarkerlessMoCapTUG.pdf` 放入 `papers/` 子目录**，跑 `bash 06_附录/pdf2zh调用脚本/run_batch_cli.sh` 自动翻译归位。

---

## 1. 论文元数据

| 字段 | 值 |
|---|---|
| 标题 | Towards Clinical Application of Enhanced Timed Up and Go With Markerless Motion Capture and Machine Learning for Balance and Gait Assessment |
| 期刊 | IEEE Journal of Biomedical and Health Informatics (JBHI) |
| 年份 | 2025 |
| DOI | `10.1109/jbhi.2025.3543095` |
| OpenAlex | <https://api.openalex.org/works/doi:10.1109/jbhi.2025.3543095> |
| IEEE Xplore | <https://ieeexplore.ieee.org/document/11049513/>（推测 arnumber）|
| OA 状态 | ❌ `is_oa: False`（OpenAlex + Unpaywall 双重确认）|

---

## 2. 作者团队

| 作者 | 机构 | 备注 |
|---|---|---|
| **Longbin Zhang** | KTH (Royal Institute of Technology, Sweden) | 一作（同我们已在跟踪的章隆彬）|
| Audrey Sidarta | NTU (Singapore) / Rehabilitation Research Institute of Singapore | 资深 |
| Tsung-Li Wu | NTU / Rehabilitation Research Institute of Singapore | 资深 |
| P. Jatesiktat | NTU | 团队 |
| H. Wang | NTU | 团队 |
| L. Li | NTU | 团队 |
| P. W.-H. Kwong | NTU / PolyU | 资深 |
| A. Long | NTU | 团队 |
| X. Long | NTU | 团队 |
| Wei Tech Ang | NTU / Rehabilitation Research Institute of Singapore | 通讯 PI |

**关键观察**：这是一篇 **NTU 团队**（新加坡南洋理工大学 + 康复研究院）的工作，章隆彬当时还在 KTH/NTU。同一团队在 2025 年章隆彬已回国到湖南大学后可能继续延伸。

---

## 3. 摘要（OpenAlex 完整重建，1899 字）

> Balance and gait impairments play a key role in falls among the elderly. Traditional clinical scales such as the Berg Balance Scale (BBS) to assess fall risk are often subjective, time consuming, and do not assess gait performance. Shorter assessments such as Timed Up and Go (TUG) are available, but most clinicians only look into the completion time. This study aimed to develop a fast, low-cost, and automated framework for balance function assessment and comprehensive gait analysis by enhancing the traditional TUG test with a markerless motion capture (MoCap) system and machine learning models. In total, we included TUG datasets of 70 participants with varying degrees of fall risk based on the BBS scores. We segmented TUG trials into five phases automatically using data from the MoCap system and extracted features from the phases. These features were then analyzed to identify those that significantly discriminate between high and low fall risk groups. Using the identified features, various machine learning models were tested to estimate the BBS scores. The markers obtained from the markerless MoCap system were used for detailed gait analysis, and lower limb kinematics were compared between the markerless and marker-based methods. Our findings indicate that individuals at high risk of falling had longer completion times, lower performance velocities, and smaller ranges of motion in lower-limb joints. Among the tested machine learning models, random forest demonstrated the best performance in predicting BBS scores (RMSE: 0.98, R²: 0.94). Additionally, our markerless MoCap system showed comparable accuracy to state-of-the-art systems, eliminating the need to attach markers or sensors. The findings could help develop a quick and objective tool for balance and gait assessment in older adults, providing quantitative data to improve screening and intervention planning.

**关键事实**：
| 字段 | 值 |
|---|---|
| 应用 | 老年人跌倒风险评估 |
| 数据 | **70 名受试者**（按 BBS 评分分高低跌倒风险）|
| 输入 | **markerless motion capture**（无标记动捕）+ TUG 测试 |
| 输出 | BBS 评分预测 + 下肢运动学分析 |
| 关键发现 | **跌倒高风险人群的 TUG 时间更长、运动速度更低、下肢关节活动范围更小** |
| 最佳模型 | **Random Forest**（不是 Transformer / LSTM / NN）|
| 性能 | **RMSE 0.98, R² 0.94** |
| markerless vs marker-based | 准确度相当，**免标记**的优势 |

---

## 4. 核心方法（从摘要 + 团队背景推导）

**完整流程**：
```
TUG 测试（70 受试者）
  → markerless MoCap 采集
  → 自动分段（5 个阶段）
  → 提取特征（关节角度、活动范围等）
  → 特征分析（区分高低风险组）
  → Random Forest → BBS 评分预测
  → markerless vs marker-based 对比
```

**5 段 TUG 阶段**（推测）：
1. 起立（sit-to-stand）
2. 行走向前（forward walk）
3. 转身（turn）
5. 走回（return walk）
4. 坐下（stand-to-sit）

**Random Forest 模型**（推测特征）：
- TUG 各段完成时间
- 关节活动范围（髋/膝/踝）
- 速度特征
- 对称性特征

---

## 5. 与 ViKiT 的对照

| 维度 | JBHI 2025 (章隆彬) | ViKiT (我们) |
|---|---|---|
| 输入 | **markerless MoCap 视频** | **RGB 视频** |
| 输出 | **BBS 评分**（分类/回归）+ 步态分析 | **关节角**（连续回归）|
| 模型 | **Random Forest** + 手工特征 | **Spatio-Temporal Transformer + cross-attention** |
| 数据集 | 70 受试者 + BBS 评分 | Human3.9M / CMU Mocap / 康复数据集 |
| 任务 | **临床评估**（跌倒风险）| **关节角预测**（运动学）|
| 评估指标 | RMSE 0.98 / R² 0.94 | MAE / RMSE / ICC / ROM violation / FK / temporal jerk |
| 可解释性 | ✅ Random Forest | ❌（Transformer 黑盒）|
| 实时性 | ❌（离线）| ✅（目标实时）|

**核心 takeaway**：章隆彬团队证明 **「markerless MoCap + ML 临床步态评估」是可行的（R² 0.94）**，但用经典 ML。

→ ViKiT 的差异化建立在 **更复杂的下游任务（关节角 vs BBS 评分）+ 更强的模型（Transformer vs Random Forest）** 上。

---

## 6. 必读章节（拿到 PDF 后）

按优先级：
1. §2 (Related Work) — 章隆彬团队对 markerless MoCap 的背景介绍
2. §3 (Method) — 5 段 TUG 自动分段 + 特征工程
3. §4 (Experiments) — 70 受试者 + Random Forest 配置 + 特征重要性
4. §5 (Discussion) — markerless vs marker-based 准确度对比

---

## 7. 我们必须引用它的位置

- **Related Work** 子簇 "B. Vision-based joint angle estimation" + "E. 临床/工程基线" 必引
- **Experiments** 对比表必列（章隆彬 markerless + BBS 评估）
- 如果未来扩展到临床应用，章隆彬的临床评估框架是直接可用的下游任务

---

## 8. 与章隆彬团队合作可能性

- **数据互补** — 章隆彬 70 受试者 TUG 数据 + BBS 评分可作 ViKiT 的临床下游验证集
- **设备互补** — 湖南大学有无标记动捕 + sEMG + EEG + NMS 建模 全套
- **方法互补** — 章隆彬 NMS 模型可作为 ViKiT 的 **forward kinematics 约束源**（biomechanical grammar loss 的 FK 项）

---

## 已获取原文（本节的「手动下载指引」已失效，留作历史记录）

> ✅ **原文已落盘**：`Zhang2025-JBHI：无标记动捕与 ML 增强的 TUG 平衡步态评估/Zhang2025-JBHI：无标记动捕与 ML 增强的 TUG 平衡步态评估.pdf`（同目录）。
> 中文精读已产出，**无需再执行任何下载步骤**。

---

## 10. 元数据来源

- 标题 / 作者 / DOI / 期刊：Crossref + OpenAlex + Semantic Scholar
- **摘要**：OpenAlex `abstract_inverted_index`（1899 字）完整重建 ✅
- 方法骨架：从摘要 + 团队姊妹工作（TNSRE 2022/2023、TBME 2024）推导
- 全文：**已获取**（原文 PDF + 全译级精读，见同目录）

---

**✅ 原文已获取（2026-09-30）**：PDF 与中文精读均已就位，本卡保留作速查用途。
