# Chao et al. 2026 Measurement — 内容速读卡

> ## 🔔 注意：本速读卡已被原文级精读取代
>
> 本速读卡写于原文缺失期间，数值为摘要级推测（官方摘要仅给出 ICC > 0.92 一个数值）。
>
> | 文件 | 权威度 |
> |---|---|
> | [`Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计.pdf`](Chao2026：单目 RGB 相机的下肢关节角度估计.pdf) | ✅ **原文 PDF（权威）** |
> | [`Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计_全文翻译.md`](Chao2026：单目 RGB 相机的下肢关节角度估计_全文翻译.md) | ✅ **基于原文的全译级精读 —— 请优先阅读这个** |
> | **本速读卡** | ⚠️ **速查用**；与上述文件冲突时，**一律以后者为准** |

---


> ✅ **状态**：**PDF 已获取**（原文已落盘于同目录）。本卡的元数据与
> 推测数值**已被原文推翻部分**——请以文件顶部的「原文级精读」为准。
> 本文档基于 OpenAlex 元数据 + Crossref + SD API metadata + 第一作 Vanyi Chao 在 KIST 团队的姊妹工作推导。
> **作用**：速查卡（原文已于 2026-09-30 获取）。**精读请以文件顶部的「原文级精读」为准**。
>
> **下载后请把 PDF 改名为 `Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计.pdf` 放入本目录**，跑 `bash 06_附录/pdf2zh调用脚本/run_batch_cli.sh` 自动翻译归位。

---

## 1. 论文元数据

| 字段 | 值 |
|---|---|
| 标题 | Lower limb joint angle estimation using a single RGB camera |
| 期刊 | Measurement (Elsevier) |
| 卷期 | Vol. 246 (April 2026) |
| PII | S0263-2241(26)00481-1 |
| DOI | `10.1016/j.measurement.2026.120772` |
| ScienceDirect | <https://www.sciencedirect.com/science/article/pii/S0263224126004811> |
| 在线发表 | 2026-04-21 |
| Open Access | ❌ `openaccessArticle: false`（订阅制） |

---

## 2. 作者团队

| 作者 | 机构 | 备注 |
|---|---|---|
| Vanyi Chao | KIST (Korea Institute of Science and Technology) | **第一作者**，博士 |
| Sungho Park | KIST | **通讯作者**（PI） |
| Ankhzaya Jamsrandorj | KIST | **同 EMBC 2022 一作**，2022→2026 持续合作 |
| Dawoon Jung | KIST | 团队成员 |
| Jinwook Kim | KIST | **资深 PI**（Mun 等 2022 EMBC 系列同 PI）|

**关键观察**：Chao 2026 是 Jamsrandorj 2022 EMBC 的**直接续作**——同一团队 4 年后从「RGB → joint angle」升级到「RGB + Transformer → hip/knee/ankle angle」。所以读 Chao 2026 必须先读 Jamsrandorj 2022 EMBC。

---

## 3. 核心方法（从论文标题 + 团队前期工作推导）

**任务定义**：
```
RGB 单目视频 → hip/knee/ankle flexion-extension（踝）角度
```

**模型骨架**（从标题 "single RGB camera" + 团队路线推测）：
```
RGB video
  → CNN（空间特征提取）
  → Transformer（时间动态 / gait dynamics）
  → Joint angles（端到端回归，无 2D pose → 3D pose → IK 中介）
```

**关键证据**：
- "single RGB camera" — 单目
- 不经过 "2D pose → 3D pose → IK" 两阶段 pipeline
- 报告了 ICC > 0.92（之前对话中提及）

**损失函数**（之前对话提及）：
$$L = L_{\text{angle}} + \lambda L_{\text{angular\_velocity}}$$
只覆盖了 velocity 项约束——**这是我们 grammar loss 的 1/6**。

---

## 4. 与 ViKiT 的结构性差异（必须写进 Related Work）

| 维度 | Chao 2026 | ViKiT (我们) |
|---|---|---|
| 输入表征 | CNN 全局特征 | **Anatomy-aware visual tokens**（按解剖切分）|
| 序列建模 | 单层 Transformer（时间动态）| **Spatio-temporal Transformer + cross-modal cross-attention** |
| 输出表征 | 连续角度向量 | **Kinematic tokens + kinematic vocabulary**（VQ 可恢复）|
| 损失函数组成 | 2 项 | **6 项**：angle / velocity / acceleration / ROM / FK / temporal |
| 任务 | 同步估计 | **同步 + forecasting**：`V_{1:T} → A_{T+1:T+H}` |
| 物理合理性 | 仅 velocity 平滑 | **ROM 显式约束 + FK consistency** |
| 评估维度 | MAE / RMSE / ICC | **+ ROM violation rate / FK consistency / temporal jerk** |

---

## 5. 必读章节（拿到 PDF 后）

按优先级：
1. §3 (Method) — 端到端 RGB → angle 的具体 CNN/Transformer 配置
2. §4 (Experiments) — 数据集、ICC > 0.92 的具体条件
3. §5 (Discussion) — 作者承认的局限（这正是我们的机会）

---

## 6. 我们必须引用它的位置

- **Abstract**：明示 "directly inspired by Chao 2026 RGB + Transformer pipeline"
- **Introduction**：列出 "近期端到端 RGB 直接输出关节角的工作 [Chao 2026]"
- **Related Work**：作为 direct competitor 的核心对比
- **Experiments**：在 baseline 一栏必跑 Chao 2026（同数据集 / 同视角 / 同速度）

---

## 7. 同团队姊妹工作（已下载 / 可下载）

| 论文 | 状态 |
|---|---|
| Jamsrandorj 2022 EMBC "Deep Learning Networks for View-independent Knee and Elbow Joint Angle Estimation" | ✅ **原文已获取**，详见 `../RGB_角度估计_2022_EMBC/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计_Abstract_Card.md`** |
| Jamsrandorj 2022 Sensors "Gait Events Prediction..." (arXiv 2203.00503) | ✅ 已下载 + 已翻译 |
| Jamsrandorj 2023 "View-independent gait events detection using CNN-transformer hybrid network" | ✅ **已获取原文** |

---

## 已获取原文（本节的「手动下载指引」已失效，留作历史记录）

> ✅ **原文已落盘**：`Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计.pdf`（同目录）。
> 中文精读已产出，**无需再执行任何下载步骤**。

---

## 9. 元数据来源

- 标题 / DOI / 期刊 / 发表日期：Crossref + OpenAlex + ScienceDirect API (`api.elsevier.com/content/article/pii/...`)
- 作者机构：OpenAlex authorships（确认 5 位作者均 KIST）
- 方法骨架：**从标题 + 团队 Jamsrandorj 2022 路线推导**
- 摘要：**已获取**（见全译级精读）

---

**✅ 原文已获取（2026-09-30）**：PDF 与中文精读均已就位，本卡保留作速查用途。
