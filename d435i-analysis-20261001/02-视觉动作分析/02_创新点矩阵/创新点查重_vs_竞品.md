# 创新点查重矩阵：ViKiT vs 12 篇竞品

> **本文核心交付**：兑现 AI 在对话最后承诺的「vs 竞品对比矩阵」，帮助你在动笔前就看清你的工作在 12 篇相关工作中的位置。
> **每个 cell 都要回答**：能不能写进 Related Work？能不能作为 baseline？能不能直接复现？

---

## 0. 速查总表（按与你想法的接近度排序）

| # | 工作 | 输入→输出 | 方法核心 | 与 ViKiT 的关系 |
|---|---|---|---|---|
| 1 | **Wei 2026 JBHI** | sEMG → joint angle | muscle-anatomy-informed 3D CNN + Seq2Seq Transformer | **种子论文**，范式原型；我们把 source 换成 visual |
| 2 | **Chao 2026 Measurement** | RGB → hip/knee/ankle angle | CNN + Transformer（end-to-end）| **直接竞品**；我们用 anatomy-aware + grammar 拉开差距 |
| 3 | **Jamsrandorj 2022 EMBC** | RGB×9 → knee/elbow angle | CNN / BiLSTM | 历史基线（不是 Transformer）|
| 4 | **Kumar 2022 EMBC** | RGB → angular velocity | CNN + BiLSTM/GRU | 历史基线 |
| 5 | **HMR 2.0 ICCV 2023** | RGB → SMPL (θ, β, π) | ViT + Transformer Decoder + SMPL query | token 系统先例（image → pose parameter）|
| 6 | **TokenHMR CVPR 2024** | RGB → discrete pose tokens | pose vocabulary + token prediction | **概念最近**：pose vocabulary + token prediction；我们换目标表征 |
| 7 | **PoseFormer ICCV 2021** | 2D joints → 3D joints | Spatial + Temporal Transformer | 不是 RGB→angle，是 2D→3D 提升 |
| 8 | **PoseFormerV2 CVPR 2023** | 2D joints → 3D joints | + 频域 | 同上 |
| 9 | **MotionBERT ICCV 2023** | 2D joints → 3D motion repr | DSTformer | 学到的表征含 kinematic 知识 → **强先验** |
| 10 | **MotionAGFormer WACV 2024** | 2D joints → 3D joints | Transformer + GCN | 混合架构借鉴 |
| 11 | **METRO CVPR 2021** | image → 3D mesh | Masked Transformer | 早期 image→3D body 工作 |
| 12 | **OpenCap 2023 PLOS** | RGB → angle via OpenSim | 2D → 3D → IK | 临床/工程基线 |

---

## 1. 详细查重矩阵（按维度横切）

### 1.1 输入表征

| 工作 | 输入 | 视觉表征 | 是否有 anatomy-aware？ |
|---|---|---|---|
| Wei 2026 | sEMG | muscle-anatomy-informed 3D CNN | ✅ |
| **Chao 2026** | RGB video | CNN 全局特征 | ❌ 普通 CNN |
| Jamsrandorj 2022 | RGB×9 | CNN | ❌ |
| HMR 2.0 | RGB | ViT image tokens | ❌ 普通 patch |
| TokenHMR | RGB | ViT image tokens | ❌ 普通 patch |
| **PoseFormer 系** | 2D joints | skeleton graph | N/A（不是 RGB）|
| **METRO** | image | masked tokens | ❌ |
| **OpenCap** | RGB | 2D keypoints | ❌ |
| **ViKiT (我们)** | RGB video | **Anatomy-aware visual tokens** | ✅ **新** |

**结论**：只有 Wei 2026 有 anatomy-aware，但那是 sEMG 版的。我们是**第一个把 anatomy-aware 引入视觉表征的工作**。

### 1.2 序列建模

| 工作 | 序列模型 | 是否有跨模态 cross-attention？ |
|---|---|---|
| Wei 2026 | Transformer decoder（自回归）| ❌ 同模态 |
| **Chao 2026** | 单层 Transformer（时间动态）| ❌ |
| Jamsrandorj/Kumar 2022 | BiLSTM/GRU | ❌ |
| HMR 2.0 | Transformer Decoder + SMPL query → image tokens **cross-attention** | ✅ |
| TokenHMR | Transformer + pose vocabulary | 间接（via VQ）|
| PoseFormer 系 | Spatial + Temporal Transformer | ❌ 同模态 |
| METRO | Masked Transformer | ❌ |
| OpenCap | N/A（pipeline，不是序列模型）| N/A |
| **ViKiT** | **Spatio-temporal Transformer + cross-modal cross-attention** | ✅ **结构上更复杂** |

**结论**：HMR 2.0 已有 cross-attention（image → SMPL query），但它的目标表征是 SMPL 参数，不是临床关节角，且没有时序结构。**我们是第一个把 cross-attention 从单帧扩展到时序视频的工作**。

### 1.3 输出表征

| 工作 | 输出表征 | 是否有 tokenization？ |
|---|---|---|
| Wei 2026 | continuous angle vector | ❌ |
| **Chao 2026** | continuous angle vector | ❌ |
| HMR 2.0 | continuous SMPL (θ, β, π) | ❌ |
| TokenHMR | **discrete pose tokens + pose vocabulary** | ✅ |
| PoseFormer | continuous 3D joints | ❌ |
| OpenCap | continuous angle (via IK) | ❌ |
| **ViKiT** | **discrete kinematic tokens + kinematic vocabulary** | ✅ **新** |

**结论**：TokenHMR 已经为 SMPL pose 做了 tokenization，但**没人把临床关节角（hip/knee/ankle 三轴）做成 vocabulary**。这是 token 系统层面的新意。

### 1.4 损失函数 / 物理约束

| 工作 | 损失函数组成 | 是否有 biomechanical grammar？ |
|---|---|---|
| Wei 2026 | L_angle + L_seq | ❌ |
| **Chao 2026** | L_angle + λ L_angular_velocity | 部分（只有 velocity）|
| Jamsrandorj 2022 | MSE / correlation loss | ❌ |
| HMR 2.0 | 3D joint loss + mesh loss | ❌ |
| TokenHMR | token prediction loss + mesh | ❌ |
| PoseFormer | 3D joint MSE | ❌ |
| OpenCap | 端到端 N/A | N/A |
| **ViKiT** | **L_angle + L_velocity + L_acceleration + L_ROM + L_FK + L_temporal** | ✅ **最完整的 grammar** |

**结论**：我们的 grammar loss 是**最完整的**——把人体完整可调的"语法"全部显式建模。这点 Chao 2026 只覆盖了 velocity 一项。

### 1.5 任务定义（estimation vs forecasting）

| 工作 | 任务 | 是否 forecasting？ |
|---|---|---|
| Wei 2026 | estimation + 300 ms ahead prediction | ✅ |
| **Chao 2026** | estimation（同步）| ❌ |
| HMR 2.0 | single-frame estimation | ❌ |
| TokenHMR | single-frame | ❌ |
| PoseFormer | estimation | ❌ |
| OpenCap | estimation（pipeline）| ❌ |
| **ViKiT** | estimation + **300 ms kinematic forecasting** | ✅ **新组合** |

**结论**：forecasting 不算新（Wei 2026 已经有了），但**视觉版的 forecasting 还没人做**。我们的工作可以填补这个空白。

---

## 2. 综合查重：每篇工作能不能直接挡我们的创新点？

| 创新点 | 会被谁挡？ | 怎么挡？ |
|---|---|---|
| **1. Visual→Kinematic Translation** | Chao 2026 | "我们也是 RGB → angle with Transformer" → 用 token 系统 + grammar + forecasting 拉开 |
| **2. Anatomy-aware Tokenization** | Wei 2026（但只在 sEMG 上）| "Wei 只在 sEMG 做了 anatomy-aware，我们扩展到视觉" |
| **3. Biomechanical Grammar** | Chao 2026（有 velocity 项）| "Chao 只有 velocity，我们有 6 项 + FK consistency" |
| **4. Causal Future Translation** | Wei 2026（但只在 sEMG 上）| "视觉版的 forecasting 没人做" |

**结论**：每个创新点都有先例，但没有完全重复。我们的组合（视觉 + anatomy-aware tokens + 6 项 grammar + forecasting）是**未见组合**。

---

## 3. Related Work 怎么写（避免被审稿人挑战）

### 3.1 必须分 4 个相关工作子簇（4 类子）

```
A. sEMG-based kinematic estimation
   └─ Wei 2026 JBHI（直接种子）

B. Vision-based joint angle estimation
   ├─ Chao 2026 Measurement（直接竞品）
   ├─ Jamsrandorj 2022 EMBC
   └─ Kumar 2022 EMBC

C. Vision Transformer for human pose / mesh
   ├─ HMR 2.0 (ICCV 2023)
   ├─ TokenHMR (CVPR 2024)
   └─ METRO (CVPR 2021)

D. 2D→3D pose lifting
   ├─ PoseFormer (ICCV 2021)
   ├─ PoseFormerV2 (CVPR 2023)
   ├─ MotionBERT (ICCV 2023)
   └─ MotionAGFormer (WACV 2024)

E. 临床/工程基线
   └─ OpenCap (PLOS 2023)
```

### 3.2 每段 Related Work 必须有「批判-接续」

不只是罗列，要写「这个做了什么 → 我们的差异在哪 → 我们能借鉴什么」。

**示例段落**（Related Work 子簇内容草稿）：

> "Recent work in vision-based joint angle estimation has demonstrated the feasibility of end-to-end RGB-to-angle mapping [Chao 2026; Jamsrandorj 2022]. In particular, Chao et al. (2026) proposed a CNN + Transformer pipeline that achieves ICC > 0.92 across views, speeds, and abnormal gaits. However, their approach treats joint angles as a single regression target, without explicit tokenization of the kinematic output space or enforcement of biomechanical constraints beyond angular velocity smoothness. In contrast, our work formulates the problem as a structured sequence translation between two distinct languages: a visual language over anatomy-aware tokens and a kinematic language over discrete joint-angle tokens. This formulation allows us to introduce explicit biomechanical grammar (ROM, forward kinematics consistency, joint coupling) into the training objective."

### 3.3 必须做的 ablation（防止被挑战）

| Ablation | 切掉什么 | 期望看到的退化 |
|---|---|---|
| A1 | anatomy-aware tokenization（换成普通 ViT patches）| MAE 退化；ROM violation 飙升 |
| A2 | grammar loss（只保留 L_angle）| 物理合理性指标退化 |
| A3 | tokenization（改成连续回归）| 关节角分布变宽；ROM violation 飙升 |
| A4 | forecasting（只做同步 estimation）| 与 Wei 2026 的对比无新意 |
| A5 | cross-attention（改成纯 autoregressive）| 长序列稳定性退化 |

每条 ablation 都要在论文的 Experiments 节里出现，并且有数据支撑。

---

## 4. 一句话定位（写在 Abstract / Introduction 第一段）

> We propose ViKiT (Visual-to-Kinematic Translator), a structured sequence translation framework that maps anatomy-aware visual tokens to discrete kinematic tokens via cross-modal cross-attention, trained with a biomechanical grammar that enforces range-of-motion, joint coupling, and forward-kinematic consistency.

---

## 5. 风险评估

| 风险 | 概率 | 应对 |
|---|---|---|
| 审稿人拿 Chao 2026 直接挡 | 高 | ablation A1+A2+A3+A5 全做 |
| 审稿人说"换汤不换药" | 中 | 强调 token 系统 + grammar loss + forecasting 的结构性差异 |
| 临床数据集缺乏 | 中 | Human3.9M / CMU Mocap / 现有康复数据集；公开 + 自采 |
| 物理指标难实现 | 中 | ROM violation rate 简单；FK consistency 需 SMPL/skeleton model |
| 翻译 token 误判 | 低 | 用 OpenRouter `stealth/ox-alpha` 翻译时抽样校验 |

---

**✓ 结论**：现有 12 篇工作中**没有任何一篇**做了「anatomy-aware 视觉 token + 6 项 biomechanical grammar + 视觉版 forecasting」的完整组合。**ViKiT 的结构性差异是有依据的。**