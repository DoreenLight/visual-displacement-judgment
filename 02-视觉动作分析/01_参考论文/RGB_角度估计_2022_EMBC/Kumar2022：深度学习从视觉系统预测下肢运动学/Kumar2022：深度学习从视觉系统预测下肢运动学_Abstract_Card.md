# Kumar et al. EMBC 2022 — 内容速读卡

> ## 🔔 注意：本速读卡已被原文级精读取代
>
> 本卡曾自行更正 DOI 为 9871577（现已由 IEEE 官方 PDF 页眉独立确认）；方法细节原文已获取。
>
> | 文件 | 权威度 |
> |---|---|
> | [`Kumar2022：深度学习从视觉系统预测下肢运动学/Kumar2022：深度学习从视觉系统预测下肢运动学.pdf`](Kumar2022：深度学习从视觉系统预测下肢运动学.pdf) | ✅ **原文 PDF（权威）** |
> | [`Kumar2022：深度学习从视觉系统预测下肢运动学/Kumar2022：深度学习从视觉系统预测下肢运动学_全文翻译.md`](Kumar2022：深度学习从视觉系统预测下肢运动学_全文翻译.md) | ✅ **基于原文的全译级精读 —— 请优先阅读这个** |
> | **本速读卡** | ⚠️ **速查用**；与上述文件冲突时，**一律以后者为准** |

---


> ✅ **状态**：**PDF 已获取**（原文已落盘于同目录）。本卡的元数据与
> 推测数值**已被原文推翻部分**——请以文件顶部的「原文级精读」为准。
> 本文档基于 OpenAlex 公开摘要（1465 字完整重建）+ Crossref + 同团队姊妹工作推导。
> **作用**：速查卡（原文已于 2026-09-30 获取）。**精读请以文件顶部的「原文级精读」为准**。
>
> **下载后请把 PDF 改名为 `Kumar_2022_EMBC_LowerLimb.pdf` 放入本目录**，跑 `bash 06_附录/pdf2zh调用脚本/run_batch_cli.sh` 自动翻译归位。

---

## 1. 论文元数据（**注意：DOI 已修正**）

| 字段 | 值 |
|---|---|
| 标题 | Prediction of lower limb kinematics from vision-based system using deep learning approaches |
| 会议 | 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC) |
| 年份 | 2022 |
| **DOI** | **`10.1109/EMBC48229.2022.9871577`**（之前错误引用为 9871780，**已修正**）|
| IEEE Xplore | <https://ieeexplore.ieee.org/document/9871577/> |
| PubMed | <https://pubmed.ncbi.nlm.nih.gov/36086538/> |

---

## 2. 作者团队

| 作者 | 机构 | 备注 |
|---|---|---|
| **Konki Sravan Kumar** | KIST | **第一作者** |
| Ankhzaya Jamsrandorj | KIST / UST | **共同作者**（同年 Jamsrandorj 2022 EMBC 一作）|
| Jinwook Kim | KIST | 资深 PI |
| Kyung-Ryoul Mun | KIST | 资深 PI（通讯）|

**关键观察**：Kumar 与 Jamsrandorj 2022 EMBC **共享同一团队、同一年 EMBC、同一个资深 PI**。这是 KIST 团队 2022 年的「双论文」姊妹工作 —— 同一议题从两个角度切入：
- Jamsrandorj: **角度估计**（position-based）
- Kumar: **角速度估计**（velocity-based）

---

## 3. 摘要（OpenAlex 完整重建，1465 字）

> The joint angular velocity during daily life exercises is an important clinical outcome for injury risk index, rehabilitation progress monitoring and athlete's performance evaluation. Recently, wearable sensors have been widely used to monitor lower limb kinematics. However, these sensors are difficult and inconvenient to use in daily life. To mitigate these limitations, this study proposes a vision-based system for estimating lower limb joint kinematics using a deep convolution neural network with bi-directional long-short term memory and gated recurrent unit network. The normalized correlation coefficient, and the mean absolute error were computed between the ground truth obtained from the optical motion capture system and estimated joint angular velocities using proposed models. The estimated results show a highest correlation 0.93 in squat and 0.92 in walking on treadmill action. Furthermore, independent model for each joint angular velocity at the hip, knee, and ankle were analyzed and compared. Among the three joint angular velocities, knee joint has a best estimated accuracy (0.96 in squat and 0.96 in walking on the treadmill). The proposed models show higher estimation accuracy under both the lateral and the frontal view regardless of the camera positions and angles. This study proves the applicability of using sensor free vision-based system to monitor the lower limb kinematics during home workouts for healthcare and rehabilitation.

**关键事实**：
| 字段 | 值 |
|---|---|
| 任务 | RGB 视频 → hip/knee/ankle **joint angular velocity**（不是角度本身） |
| 应用场景 | 日常锻炼（深蹲 + 跑步机步行）|
| 模型 | **CNN + BiLSTM + GRU** 多种组合 |
| Ground truth | 光学动捕 |
| 评估指标 | normalized correlation coefficient + MAE |
| 总体最高相关 | squat 0.93 / 跑步机 0.92 |
| 膝关节（最优）| squat 0.96 / 跑步机 0.96 |
| 视角 | lateral + frontal 都测试 |
| 关键卖点 | 无需传感器、视角无关、家庭康复 |

---

## 4. 核心方法（从摘要 + 团队背景推导）

**输入**：
```
RGB 视频（squats + walking）
```

**模型对比**（推测）：
- CNN baseline
- **CNN + BiLSTM**
- **CNN + GRU**
- 可能是独立模型（每个关节一个）vs 共享模型

**输出**：
```
hip / knee / ankle 三轴角速度（angular velocity）
```

**创新点**：
1. **无传感器**：纯 RGB 替代 IMU/wearable
2. **多关节独立建模**：每个关节有自己的模型
3. **跨视角鲁棒**：lateral + frontal 都行

---

## 5. 与 ViKiT 的对照

| 维度 | Kumar 2022 EMBC | ViKiT (我们) |
|---|---|---|
| 输入 | RGB 视频 | **长序列 RGB 视频** |
| 输出 | **角速度**（velocity）| **角度**（position）+ velocity + acceleration |
| 模型 | CNN + BiLSTM / GRU | **Spatio-temporal Transformer + cross-attention** |
| 输出表征 | 连续角速度向量 | **Kinematic tokens + vocabulary** |
| 损失函数 | MAE + correlation loss | **6 项 grammar loss** |
| 数据集 | KIST 自采（squat + walking）| Human3.9M / CMU Mocap / 公开康复数据 |
| 关节 | hip / knee / ankle（3 轴）| hip / knee / ankle（多轴，含 flexion/extension 等）|
| 视角 | lateral + frontal | 跨视角（更严格）|
| 评估 | correlation + MAE | **MAE / RMSE / ICC / ROM violation / FK consistency / temporal jerk** |

**核心 takeaway**：Kumar 2022 EMBC 证明 **「vision → angular velocity」也是可行的（2022 年就已经做到 r ≈ 0.93）**，且**多关节独立建模**。

→ 我们的差异化建立在 **角度估计**（vs 角速度估计）+ **token 系统 + grammar** 上。

---

## 6. 与 Jamsrandorj 2022 EMBC 的关系（同团队姊妹工作）

| 维度 | Jamsrandorj 2022 | Kumar 2022 |
|---|---|---|
| 输出 | **角度**（position）| **角速度**（velocity）|
| 关节 | knee + elbow | hip + knee + ankle |
| 活动 | 4 种日常活动 | squat + walking |
| 最佳模型 | BiLSTM（单一）| CNN + BiLSTM / GRU（多模型对比）|
| 最佳相关 | 膝 0.955 / 肘 0.917 | squat 0.93 / 步行 0.92 |

→ 两篇合起来覆盖 **vision → angle** 和 **vision → angular velocity** 两条线。

---

## 7. 同团队其他论文（已下载）

| 论文 | 状态 |
|---|---|
| **Jamsrandorj 2022 EMBC** "Deep Learning Networks for View-independent Knee and Elbow Joint Angle Estimation" | ✅ **原文已获取**，详见 `Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计_Abstract_Card.md`** |
| Jamsrandorj 2022 Sensors / arXiv 2203.00503 | ✅ 已下载 + 已翻译 |
| **Chao 2026 Measurement** | ✅ **原文已获取**，详见 `../RGB_直接竞品_2026_Chao/Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计_Abstract_Card.md`** |
| Review 2021 "Video-Based Markerless Motion Capture for Clinical and Rehabilitation Biomechanics" | ✅ **新增** `../OpenCap_传统生物力学/MarkerlessReview2021：视频无标记动捕临床康复生物力学综述/MarkerlessReview2021：视频无标记动捕临床康复生物力学综述.pdf` (44 页, 3 MB) |

---

## 8. 必读章节（拿到 PDF 后）

按优先级：
1. §3 (Method) — 多模型对比 + 每个关节独立建模的细节
2. §4 (Experiments) — squat + walking 协议 + 跨视角测试
3. §5 (Discussion) — 角速度 vs 角度的取舍、临床应用场景

---

## 9. 我们必须引用它的位置

- **Related Work** 子簇 "B. Vision-based joint angle estimation" 必引
- **Experiments** baseline 表必列（Kumar 2022 BiLSTM）
- 如果用 Human3.9M / CMU Mocap / 康复数据集，可考虑与 KIST 自采数据集对比

---

## 已获取原文（本节的「手动下载指引」已失效，留作历史记录）

> ✅ **原文已落盘**：`Kumar2022：深度学习从视觉系统预测下肢运动学/Kumar2022：深度学习从视觉系统预测下肢运动学.pdf`（同目录）。
> 中文精读已产出，**无需再执行任何下载步骤**。

---

## 11. 元数据来源

- 标题 / 作者 / DOI / 会议：Crossref + OpenAlex
- **摘要**：OpenAlex `abstract_inverted_index`（1465 字）完整重建 ✅
- 方法骨架：从摘要 + 同团队姊妹工作（Jamsrandorj 2022）推导
- 全文：**已获取**（原文 PDF + 全译级精读，见同目录）

---

**✅ 原文已获取（2026-09-30）**：PDF 与中文精读均已就位，本卡保留作速查用途。
