# ViKiT 方法概览

> **模型全名**：An Anatomy- and Biomechanics-Informed Visual-to-Kinematic Sequence Translation Framework for Continuous Joint Kinematics Estimation
> **简称**：**ViKiT — Visual-to-Kinematic Translator**
> **核心命题**：$$\boxed{\text{Visual Language} \rightarrow \text{Kinematic Language}}$$

---

## 1. 整体架构（文字版）

```
┌────────────────────────────────────────────────────────────────────┐
│                          输入：RGB Video I_{1:T}                   │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  ① Anatomy-aware Visual Tokenizer（source language）              │
│     - 按人体运动链把每帧切成 anatomy regions                       │
│     - 每个 region 用小型 ViT 编码 → V_t = [v_{t,hip}, v_{t,thigh},│
│       v_{t,knee}, v_{t,shank}, v_{t,ankle}, ...]                 │
│     - 输出：visual token sentence V_{1:T}                          │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  ② Spatio-Temporal Transformer Encoder                            │
│     - Spatial attention：同帧 anatomy token 关系                   │
│     - Temporal attention：跨帧时间动态                             │
│     - 输出：时序视觉表征 H_{1:T}                                  │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  ③ Cross-Modal Kinematic Decoder（target language）               │
│     - Query: kinematic tokens K = [k_θhip_flex, k_θhip_add, ...]  │
│     - Cross-attention: K attends to H                             │
│     - 输出：kinematic token sentence K_{1:T}                       │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  ④ Kinematic Token Vocabulary + VQ Decode                         │
│     - 每个 kinematic token ∈ {token_1, ..., token_N}              │
│     - 通过 VQ-VAE 把离散 token 解码为连续角度                     │
│     - 输出：joint angle trajectory â_{1:T}                        │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  ⑤ Biomechanical Grammar Loss（多目标训练）                       │
│     L = L_angle + λ_1 L_velocity + λ_2 L_acceleration +           │
│         λ_3 L_ROM + λ_4 L_FK + λ_5 L_temporal                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. 四个核心组件

### 2.1 Anatomy-aware Visual Tokenizer

**目标**：把"普通 ViT patches"换成"按解剖切分的视觉 token"。

| 项 | 内容 |
|---|---|
| 切分依据 | 人体运动链（hip→thigh→knee→shank→ankle 等）|
| 切分方法 | (1) 用人体关键点引导 attention；(2) 用 semantic segmentation mask；(3) 用可学习的 soft assignment |
| 编码器 | 局部小型 ViT（每个 region 一个 ViT-S/16 或类似）|
| 输出 | $V_t = [v_{t,\text{hip}}, v_{t,\text{thigh}}, v_{t,\text{knee}}, v_{t,\text{shank}}, v_{t,\text{ankle}}, \ldots]$，每个 $v_{t,i} \in \mathbb{R}^d$ |

**对应原 Wei 2026**：他们用"muscle anatomy informed representation"指导 sEMG 端的 3D CNN。我们把这个思路搬到视觉端。

### 2.2 Spatio-Temporal Transformer Encoder

**目标**：把视觉 token 序列编码成时序表征。

| 子模块 | 作用 |
|---|---|
| Spatial attention | 同帧 anatomy token 之间的关系（髋和膝的耦合）|
| Temporal attention | 跨帧时间动态（步态周期）|
| Positional encoding | 加帧号 + anatomy region id |

**借鉴**：PoseFormer 的 Spatial/Temporal 分离设计。

### 2.3 Cross-Modal Kinematic Decoder

**目标**：把视觉表征"翻译"为运动学表征。

| 项 | 内容 |
|---|---|
| Query 类型 | kinematic tokens（每个对应一个关节自由度）|
| Cross-attention | $K = \text{softmax}(Q K^T / \sqrt{d}) V$，$Q$ 来自 kinematic tokens，$K, V$ 来自 visual encoder 输出 |
| 自回归 | 可选（像 NLP 翻译那样从左到右生成 kinematic tokens）|

**借鉴**：HMR 2.0 的 SMPL query + image tokens cross-attention。我们把"SMPL query"换成"kinematic query"。

### 2.4 Kinematic Token Vocabulary + VQ Decode

**目标**：把"翻译"做成真正意义上的"翻译"（不是回归）。

```python
# 伪代码
class KinematicVocabulary:
    def __init__(self, rom_range=(-120, 120), n_bins=256):
        self.bins = torch.linspace(*rom_range, n_bins)

    def encode(self, angle_continuous):
        # 连续角度 → 离散 token
        token_id = torch.bucketize(angle_continuous, self.bins)
        return token_id

    def decode(self, token_id):
        # 离散 token → 连续角度
        angle = self.bins[token_id]
        return angle
```

| 项 | 内容 |
|---|---|
| Vocabulary 大小 | 256（每个关节自由度）|
| 编码方式 | —（bucketize）或 VQ-VAE（learned）|
| 解码方式 | bin 中点 或 VQ decode |

**借鉴**：TokenHMR 的 pose vocabulary + token prediction。我们把"pose vocabulary"换成"kinematic vocabulary"。

---

## 3. Biomechanical Grammar Loss

**核心思想**：生物力学也有"grammar"——人体不能违反的物理规则。把这些规则写进损失函数。

| 损失项 | 物理含义 | 数学形式 |
|---|---|---|
| $L_{\text{angle}}$ | 角度误差 | $\sum_t \|a_t - \hat{a}_t\|^2$ |
| $L_{\text{velocity}}$ | 角速度连续性 | $\sum_t \|(\hat{a}_{t+1} - \hat{a}_t) - (\dot{a}_{t+1} - a_t)\|^2$ |
| $L_{\text{acceleration}}$ | 角加速度平滑 | $\sum_t \|(\hat{\ddot{a}}_t - \ddot{a}_t)\|^2$ |
| $L_{\text{ROM}}$ | 关节活动范围 | $\sum_t \sum_j \max(0, \hat{\theta}_{t,j} - \theta^{\max}_j)^2 + \max(0, \theta^{\min}_j - \hat{\theta}_{t,j})^2$ |
| $L_{\text{FK}}$ | 正运动学一致性 | $\|F(\hat{\theta}_{1:T}) - F(\theta_{1:T})\|^2$，其中 $F$ 是 skeleton FK |
| $L_{\text{temporal}}$ | 时序连续性 | $\sum_t \|\hat{a}_{t+1} - \hat{a}_t\|^2$ |

**总损失**：

$$L = L_{\text{angle}} + \lambda_1 L_{\text{velocity}} + \lambda_2 L_{\text{acceleration}} + \lambda_3 L_{\text{ROM}} + \lambda_4 L_{\text{FK}} + \lambda_5 L_{\text{temporal}}$$

**对比 Chao 2026**：

| 维度 | Chao 2026 | ViKiT |
|---|---|---|
| 物理约束数量 | 1（velocity）| **5 + FK** |
| 显式 ROM 限制 | ❌ | ✅ |
| FK 一致性 | ❌ | ✅ |
| 关节耦合 | ❌ | ✅（via FK）|

---

## 4. 任务定义（两种）

### 4.1 Synchronous Estimation（同步估计）

$$V_{1:T} \rightarrow A_{1:T}$$

### 4.2 Causal Future Kinematic Forecasting（因果未来预测）

$$V_{1:T} \rightarrow A_{T+1:T+H}$$

例如：1 秒视频 → 接下来 100/200/300 ms 关节角。

**两个任务都评估**，并对比。

---

## 5. 训练策略

| 项 | 内容 |
|---|---|
| 视觉 backbone | 预训练 ViT（来自 HMR 2.0 或 ImageNet）|
| Anatomy tokenizer | 随机初始化 + 弱监督（用现成的 2D keypoints）|
| Kinematic decoder | 随机初始化 |
| VQ | 用 ground-truth 角度先训 VQ-VAE |
| 总训练 | 两阶段：先训 VQ，再 end-to-end 训整个翻译模型 |
| 优化器 | AdamW，lr=1e-4，cosine schedule |
| 批大小 | 32-64 |
| 数据集 | Human3.9M / CMU Mocap / 现有康复数据集 |

---

## 6. 评估指标（建议全覆盖）

| 类别 | 指标 |
|---|---|
| **精度** | MAE°, RMSE°, R², ICC, Pearson r |
| **物理合理性** | ROM violation rate, FK consistency, joint coupling error |
| **时序质量** | angular velocity error, angular acceleration error, temporal jerk |
| **鲁棒性** | cross-subject / cross-view / cross-speed / occlusion |
| **泛化** | pathological gait, compensatory movement |

---

## 7. 与现有方法的对比（一句话）

| 方法 | 一句话 |
|---|---|
| Wei 2026 | sEMG 版的 Seq2Seq 翻译原型 |
| Chao 2026 | RGB 版的 CNN+Transformer 角度回归 |
| **ViKiT** | **RGB 版的结构化翻译 + anatomy-aware tokens + biomechanical grammar + 可选 forecasting** |

---

## 8. 论文写作建议

### 8.1 标题候选

> "ViKiT: An Anatomy- and Biomechanics-Informed Visual-to-Kinematic Sequence Translation Framework for Continuous Joint Kinematics Estimation"

### 8.2 Abstract 第一句

> "We formulate visual-to-kinematic mapping as a structured sequence translation between two languages: a visual language over anatomy-aware tokens and a kinematic language over discrete joint-angle tokens."

### 8.3 Introduction 三段式

1. 第 1 段：动机（sEMG 不能用了，RGB 是非侵入的，2026 年已有人做 RGB→angle，但还停留在 regression）
2. 第 2 段：我们的视角（升级为翻译范式，需要 token 系统 + grammar）
3. 第 3 段：贡献清单（4 项）

---

## 9. 详细方法设计 ✅ 已就位

Phase 2 的 5 篇详细方法设计文档**已全部写入** [`../03_方法设计/`](../03_方法设计/)，索引见其 `README.md`：

| 文档 | 内容 |
|---|---|
| [`Anatomy_aware_Tokenizer.md`](../03_方法设计/Anatomy_aware_Tokenizer.md) | anatomy region 怎么切（S1/S2/S3 三方案）、token 维度、ViT backbone 选型、分配温度 τ |
| [`Kinematic_Tokens_与_Grammar.md`](../03_方法设计/Kinematic_Tokens_与_Grammar.md) | vocabulary 设计（K=256）、**6 项 loss 的具体超参与权重**、两阶段训练策略 |
| [`Spatio_Temporal_Encoder.md`](../03_方法设计/Spatio_Temporal_Encoder.md) | encoder 结构（分解式时空注意力）、双轴 position encoding、sequence length 选型 |
| [`Cross_Modal_Decoder.md`](../03_方法设计/Cross_Modal_Decoder.md) | decoder 结构、cross-attention 配置、**解剖路由**、autoregressive vs parallel 预登记决策规则 |
| [`Forecasting_Task.md`](../03_方法设计/Forecasting_Task.md) | forecasting horizon (100/200/300 ms)、RGB vs sEMG 信息带宽约束、评估协议 |

**三条全局最脆弱假设**（必须在 Phase 3 早期检验，详见 `../03_方法设计/README.md` §4）：
H1 解剖先验有效（S1/S2 > S3）· H2 cross-attention 非冗余（视觉消融必须劣化）· H3 长序列带来增益（T=81 > T=27）

> ⚠️ **文件名更正**：本节此前的清单列作 `Kinematic_Vocabulary_与_Grammar.md` 与 `Spike_Temporal_Encoder.md`，
> 与实际落盘文件名不符，现已更正为 **`Kinematic_Tokens_与_Grammar.md`** 与 **`Spatio_Temporal_Encoder.md`**。