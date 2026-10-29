# 章隆彬 — 代表性论文清单

> **更新日期**：2026-09-30
> **作者**：Longbin Zhang（章隆彬，Hunan University, KTH PhD）
> **OpenAlex 作者 ID**：`A5002794703`（100 篇作品）
> **总下载**：**6/6 目标论文全部已获取原文**（JBHI 2025 原文已由用户的学校数据库取得，原误命名为 `LAB_PROFILE.pdf`）
> **PDF 路径**：`09_章隆彬_合作团队/papers/`
> **翻译路径**：`09_章隆彬_合作团队/_translation_outputs/` + `papers/*_全文翻译.md`（**6/6 全部已译**）

---

## 论文总览

| # | 论文 | 期刊 | 年份 | DOI | OA | PDF | 翻译 | 优先级 |
|---|---|---|---|---|---|---|---|---|
| 1 | Towards Clinical Application of Enhanced TUG with Markerless MoCap and ML for Balance/Gait Assessment | **IEEE JBHI** | 2025 | [10.1109/jbhi.2025.3543095](https://doi.org/10.1109/jbhi.2025.3543095) | ❌ | ✅ 2.2 MB | ✅ 全译级 | ⭐⭐⭐⭐⭐ |
| 2 | EMG-Constrained and Ultrasound-Informed Muscle-Tendon Parameter Estimation in Post-Stroke Hemiparesis | **IEEE TBME** | 2024 | [10.1109/tbme.2024.3352556](https://doi.org/10.1109/tbme.2024.3352556) | ✅ | ✅ 1.4 MB | ✅ 已译 | ⭐⭐⭐⭐ |
| 3 | Estimation of Joint Torque by EMG-Driven Neuromusculoskeletal Models and LSTM Networks | **IEEE TNSRE** | 2023 | [10.1109/tnsre.2023.3315373](https://doi.org/10.1109/tnsre.2023.3315373) | ✅ | ✅ 2.7 MB | ✅ 已译 | ⭐⭐⭐⭐ |
| 4 | Ankle Joint Torque Prediction Using an NMS Solver Informed-ANN Model and Transfer Learning | **IEEE JBHI** | 2022 | [10.1109/jbhi.2022.3207313](https://doi.org/10.1109/jbhi.2022.3207313) | ✅ | ✅ 3.6 MB | ✅ 已译 | ⭐⭐⭐⭐ |
| 5 | Lower-Limb Joint Torque Prediction Using LSTM Neural Networks and Transfer Learning | **IEEE TNSRE** | 2022 | [10.1109/tnsre.2022.3156786](https://doi.org/10.1109/tnsre.2022.3156786) | ✅ | ✅ 4.4 MB | ✅ 已译 | ⭐⭐⭐ |
| 6 | Neuromusculoskeletal Model-Informed Machine Learning-Based Control of a Knee Exoskeleton | **Frontiers in Neuroscience** | 2023 | [10.3389/fnins.2023.1254088](https://doi.org/10.3389/fnins.2023.1254088) | ✅ | ✅ 1.9 MB | ✅ 已译 | ⭐⭐⭐⭐ |

---

## 各论文详情

### 1. ⭐⭐⭐⭐⭐ JBHI 2025 — Markerless MoCap + TUG + ML

| 字段 | 值 |
|---|---|
| 标题 | Towards Clinical Application of Enhanced Timed Up and Go With Markerless Motion Capture and Machine Learning for Balance and Gait Assessment |
| 期刊 | IEEE Journal of Biomedical and Health Informatics |
| 作者 | L. Zhang, A. Sidarta, T.-L. Wu, P. Jatesiktat, H. Wang, L. Li, P. W.-H. Kwong, A. Long, X. Long, W. T. Ang |
| DOI | `10.1109/jbhi.2025.3543095` |
| **PDF 状态** | ✅ **已获取原文**（`papers/Zhang2025-JBHI：无标记动捕与 ML 增强的 TUG 平衡步态评估/Zhang2025-JBHI：无标记动捕与 ML 增强的 TUG 平衡步态评估.pdf`，2,214,260 B / **10 页**，IEEE 官方版） |
| **中文精读** | ✅ `papers/Zhang2025-JBHI：无标记动捕与 ML 增强的 TUG 平衡步态评估/Zhang2025-JBHI：无标记动捕与 ML 增强的 TUG 平衡步态评估_全文翻译.md`（**基于原文的全译级精读**） |
| 摘要（OpenAlex 重建 1899 字）| Balance and gait impairments play a key role in falls among the elderly... (70 参与者, random forest RMSE 0.98, R² 0.94) |
| 与 ViKiT 关系 | **直接对应**：markerless MoCap + ML 用于步态评估——和 ViKiT 同样的 input pipeline，目标下游应用不同（评估 vs 角度预测）|

**关键事实**：
- 70 名受试者（不同跌倒风险）
- 5 段 TUG 阶段自动分割
- markerless vs marker-based 对比：markerless 准确度与 SOTA 相当
- 提取的特征区分高低跌倒风险组
- **与 ViKiT 的差异**：
  - 章隆彬：markerless → BBS 评分（分类/回归）
  - ViKiT：RGB → 关节角（连续回归）

---

### 2. ⭐⭐⭐⭐ TBME 2024 — Muscle-Tendon Parameter Estimation (Post-Stroke)

| 字段 | 值 |
|---|---|
| 标题 | EMG-Constrained and Ultrasound-Informed Muscle-Tendon Parameter Estimation in Post-Stroke Hemiparesis |
| 期刊 | IEEE Transactions on Biomedical Engineering |
| 作者 | L. Zhang, T. V. Wouwe, S. Yan, R. Wang |
| 卷期 | vol. 71, no. 6, pp. 1798–1809, 2024 |
| DOI | `10.1109/tbme.2024.3352556` |
| **PDF 状态** | ✅ **已下载** (`Zhang2024-TBME：肌电约束与超声信息的肌腱参数估计/Zhang2024-TBME：肌电约束与超声信息的肌腱参数估计.pdf`, 1.4 MB, 12 页) |
| 与 ViKiT 关系 | **个性化肌肉肌腱参数**——可以借鉴为 biomechanical grammar loss 中的人体物理约束项 |

**关键贡献**：
- 联合 EMG + 超声数据估计肌肉肌腱参数
- 应用于中风偏瘫患者
- NMS 模型与影像融合
- 可作为 ViKiT 的 forward kinematics / muscle physics 约束源

---

### 3. ⭐⭐⭐⭐ TNSRE 2023 — Joint Torque by EMG-Driven NMS + LSTM

| 字段 | 值 |
|---|---|
| 标题 | Estimation of Joint Torque by EMG-Driven Neuromusculoskeletal Models and LSTM Networks |
| 期刊 | IEEE Transactions on Neural Systems and Rehabilitation Engineering |
| 作者 | L. Zhang, D. Soselia, R. Wang, E. M. Gutierrez-Farewik |
| 卷期 | vol. 31, pp. 3722–3731, 2023 |
| DOI | `10.1109/tnsre.2023.3315373` |
| **PDF 状态** | ✅ **已下载** (`Zhang2023-TNSRE：肌电驱动 NMS 与 LSTM 的关节力矩估计/Zhang2023-TNSRE：肌电驱动 NMS 与 LSTM 的关节力矩估计.pdf`, 2.7 MB, 10 页) |
| 与 ViKiT 关系 | **NMS + LSTM 混合模型**——和 ViKiT 的「物理先验 + 神经网络」思路一致 |

**关键贡献**：
- NMS 模型 + LSTM 融合
- 关节扭矩估计（不是角度）
- EMG 驱动
- 体现章隆彬团队的混合方法论

---

### 4. ⭐⭐⭐⭐ JBHI 2022 (Featured Article) — Ankle Torque NMS-ANN + Transfer

| 字段 | 值 |
|---|---|
| 标题 | Ankle Joint Torque Prediction Using an NMS Solver Informed-ANN Model and Transfer Learning |
| 期刊 | IEEE Journal of Biomedical and Health Informatics (Featured Article) |
| 作者 | L. Zhang, X. Zhu, E. M. Gutierrez-Farewik, R. Wang |
| 卷期 | vol. 26, no. 12, pp. 5895–5906, 2022 |
| DOI | `10.1109/jbhi.2022.3207313` |
| **PDF 状态** | ✅ **已下载** (`Zhang2022-JBHI：NMS-ANN 与迁移学习的踝关节力矩预测/Zhang2022-JBHI：NMS-ANN 与迁移学习的踝关节力矩预测.pdf`, 3.6 MB, 12 页) |
| 与 ViKiT 关系 | **ANN + transfer learning**——可借鉴为 ViKiT 的跨被试/跨数据集迁移策略 |

**关键贡献**：
- NMS solver-informed ANN（与 LSTM 不同）
- **Transfer learning** 策略
- Featured Article 级别

---

### 5. ⭐⭐⭐ TNSRE 2022 — Lower-Limb Torque via LSTM + Transfer

| 字段 | 值 |
|---|---|
| 标题 | Lower-Limb Joint Torque Prediction Using LSTM Neural Networks and Transfer Learning |
| 期刊 | IEEE Transactions on Neural Systems and Rehabilitation Engineering |
| 作者 | L. Zhang, D. Soselia, R. Wang, E. M. Gutierrez-Farewik |
| 卷期 | vol. 30, pp. 600–609, 2022 |
| DOI | `10.1109/tnsre.2022.3156786` |
| **PDF 状态** | ✅ **已下载** (`Zhang2022-TNSRE：LSTM 与迁移学习的下肢关节力矩预测/Zhang2022-TNSRE：LSTM 与迁移学习的下肢关节力矩预测.pdf`, 4.4 MB, 10 页) |
| 与 ViKiT 关系 | **LSTM 时序建模 + transfer learning**——可借鉴为 ViKiT 的 decoder 时序结构 |

**关键贡献**：
- LSTM 时序模型
- Transfer learning
- 下肢多关节扭矩

---

### 6. ⭐⭐⭐⭐ Front Neurosci 2023 — NMS-Informed ML Exoskeleton Control

| 字段 | 值 |
|---|---|
| 标题 | Neuromusculoskeletal Model-Informed Machine Learning-Based Control of a Knee Exoskeleton with Uncertainties Quantification |
| 期刊 | Frontiers in Neuroscience |
| 作者 | L. Zhang, X. Zhang, X. Zhu, R. Wang, E. M. Gutierrez-Farewik |
| 卷期 | vol. 17, pp. 1–12, 2023 |
| DOI | `10.3389/fnins.2023.1254088` |
| **PDF 状态** | ✅ **已下载** (`Zhang2023-FrontNeurosci：NMS 驱动 ML 的膝关节外骨骼控制与不确定性量化/Zhang2023-FrontNeurosci：NMS 驱动 ML 的膝关节外骨骼控制与不确定性量化.pdf`, 1.9 MB, 12 页) |
| 与 ViKiT 关系 | **NMS 指导 ML 的膝关节外骨骼控制**——直接关系到「物理先验如何融入 ML」|

**关键贡献**：
- NMS + ML 混合
- 膝关节外骨骼
- **不确定性量化**（UQ）——可借鉴为 ViKiT 的可信度评估

---

## 与 ViKiT 项目 6 篇论文交叉总结

| ViKiT 创新 | 章隆彬对应 |
|---|---|
| Anatomy-aware visual tokens | TBME 2024 个性化肌肉肌腱参数 |
| Spatio-Temporal Transformer | TNSRE 2023 LSTM + NMS / TNSRE 2022 LSTM + transfer |
| Cross-Attention Decoder | JBHI 2022 ANN + NMS-informed (transfer learning 借鉴) |
| Kinematic tokens + vocabulary | （章隆彬无离散化，但 NMS 提供连续物理信号）|
| Biomechanical Grammar (6 项 loss) | **TBME 2024 + TNSRE 2023 + Front Neurosci 2023** 提供 muscle-tendon / NMS / uncertainty 物理约束 |
| Forecasting 任务 | （章隆彬无未来预测；但 NMS 求解器可作为"快模型"近似未来）|

→ 章隆彬 6 篇论文**全部可作为 ViKiT 的方法学参照**和**潜在合作 baseline**。

---

## 下载/翻译 状态

| 论文 | PDF | 翻译 |
|---|---|---|
| #1 JBHI 2025 | ✅ 2.2 MB | ✅ 全译级 |
| #2 TBME 2024 | ✅ | ✅ 已译 |
| #3 TNSRE 2023 | ✅ | ✅ 已译 |
| #4 JBHI 2022 | ✅ | ✅ 已译 |
| #5 TNSRE 2022 | ✅ | ✅ 已译 |
| #6 Front Neurosci 2023 | ✅ | ✅ 已译 |

**总计**：**6 真 PDF + 6 篇翻译 = 6/6 全部完成**（2026-09-30）。

---

## 来源

- 湖南大学官方主页：<https://robotics.hnu.edu.cn/info/1176/3002.htm>
- OpenAlex 作者页：<https://api.openalex.org/authors/A5002794703>
- IEEE Xplore 作者页：<https://ieeexplore.ieee.org/author/37088232279>
- OpenAlex 100 篇作品：<https://api.openalex.org/works?filter=author.id:A5002794703>