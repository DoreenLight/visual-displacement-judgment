# 03 · Spatio-Temporal Transformer Encoder（时空编码器）

> **所属**：ViKiT 组件 ③（source language 的时序编码）
> **上游**：`Anatomy_aware_Tokenizer.md`（消费 `(B,T,R,d)`）
> **下游**：`Cross_Modal_Decoder.md`（提供 memory）
> **一致性基准**：`../02_创新点矩阵/ViKiT_方法概览.md` §2.2
> **本文档定位**：Phase 2 方法设计的**实质设计文档**，含结构 / 超参 / 选型依据 / 失败模式

---

## 1. 输入契约与设计问题

```python
V: (B, T, R, d)   # T=81, R=8 (r0..r7), d=256
```

**核心设计问题**：$V$ 是**两个轴的张量**——
- **R 轴（region）**：解剖单元，**本质是集合**（无内在顺序，但有解剖层级）
- **T 轴（frame）**：时间，**严格有序**

> **拿竞品对比**：KIST 的 CNN-Transformer（EMBC 2023，真实全文）输入是
> **ResNet18 输出的一维特征序列**——**只有时间轴一个维度**，没有 region 轴。
> ViKiT 多出的 **region 轴**正是"anatomy-aware"在编码器层的体现，
> **也是我方 space-time 分解设计的必要性来源。**

---

## 2. 三种编码器结构（按优先级）

### 2.1 结构 A · 分解式时空注意力（**主方案**）

```
V: (B, T, R, d)
   │
   ├─ [空间块] 对 R 轴做 self-attention（同一帧内跨 region 交互）
   │   + Region Positional Encoding (RPE)
   │
   ├─ [时间块] 对 T 轴做 self-attention（同一 region 内跨帧交互）
   │   + Temporal Positional Encoding (TPE)
   │
   重复 L 层
   │
   ▼
Memory: (B, T, R, d)
```

**动机**：region 轴与时间轴的**统计性质完全不同**（集合 vs 序列），
用同一套参数处理会引入归纳偏置错配。分解后：

| 复杂度 | 值 |
|---|---|
| 全联合注意力 | $O((TR)^2) = O((81\times8)^2) \approx O(4.2\times10^5)$ |
| **分解式** | $O(T^2R + R^2T) \approx O(81^2\cdot8 + 8^2\cdot81) \approx O(5.7\times10^4)$ |
| **节省** | **≈ 7.4×** |

> **可引用的对照**：KIST EMBC 2023 给出的"为何用 Transformer 而非 BiLSTM"理由是
> **"计算成本只有一半"**（原文 §IV）。我方分解式设计把这条论证**推进到时空二维**。

### 2.2 结构 B · 联合注意力 + 结构化 mask（对照）

对 $(T \times R)$ 展平后做一次 self-attention，但用 mask 限制跨轴连接强度。

| | 优点 | 缺点 |
|---|---|---|
| | 表达能力上限更高（无分解假设） | 复杂度高 7.4×；需调 mask 模式 |

**用途**：验证"分解假设"是否损失了表达能力。若 B 显著优于 A，说明时空耦合比预期更强。

### 2.3 结构 C · 纯时间注意力（**基线下限**）

把 R 轴直接 flatten 或 mean-pool，**只保留时间轴注意力**。

> ⚠️ **结构 C 不是废案，而是必备对照**：它就是**竞品（KIST CNN-Transformer）的同构形式**。
> 若 C ≈ A，则说明 **region 轴没有带来增益** → attribute 该发现会触发 §5 R1 风险。
> **必须做，且必须如实报告。**

---

## 3. Positional Encoding 设计

### 3.1 双轴编码（本设计的核心）

$$h_{t,r} = v_{t,r} + \mathrm{TPE}(t) + \mathrm{RPE}(r)$$

| 编码 | 形式 | 依据 |
|---|---|---|
| **TPE（时间）** | 正弦/余弦（Vaswani 原始式）或 **RoPE** | $T$ 轴严格有序、长度可能外推 |
| **RPE（region）** | **可学习嵌入**（$R=8$ 固定） | region 是**固定的小集合**，学一个 8 向量即可；无需泛化到未见 region |

> **RPE 必须可学习而非正弦**：因为 region **没有内在顺序**。
> 用正弦编码会**人为引入 r3 与 r5 比 r3 与 r4 更"远"的伪结构**——
> 而 `r0 torso` 与 `r5 wrist` 的解剖关联度并不由索引距离决定。

### 3.2 Region 层级关系的显式注入（可选增强）

解剖链存在**已知的树状层级**（torso → shoulder → upper_arm → elbow → forearm → wrist → hand）。
可在 RPE 之外加 **相对解剖距离偏置（anatomy-relative bias）**：

$$A_{r,r'} = -\kappa \cdot d_{\text{chain}}(r, r')$$

其中 $d_{\text{chain}}$ 为解剖链上的跳数。$\kappa$ 可学习。

> **这是把"Biomechanical Grammar"从损失层**前移到**注意力层**的一种形式——
> 若有效，会强化我方"约束注入的深度"这一差异论据（见 `Kinematic_Tokens_与_Grammar.md` §3.2）。

### 3.3 序列长度 $T$ 的选型依据

| $T$ | 时长 @30fps | 显存 | 对手参照 | 用途 |
|---|---|---|---|---|
| **9** | 0.3 s | 极低 | **Jamsrandorj 2022 EMBC 的窗口** | 下界对照 |
| **27** | 0.9 s | 低 | **Kumar 2022 / Jamsrandorj 2023 的窗口** | **关键对照：直接对标 KIST** |
| **45** | 1.5 s | 中 | — | 中间点 |
| **81** | 2.7 s | 中高 | — | **基线（主方案）** |
| **121** | 4.0 s | 高 | — | 上限测试 |

> 🔴 **必须给出 $T$ 的消融曲线**。
> 三个对手的窗口都是 **9 / 27 帧的短窗**，且都用于**周期性下肢步态**。
> 而上肢代偿是**非周期、事件驱动**的——因果链可能跨越数秒。
> **若 $T \in \{27, 45, 81, 121\}$ 的曲线是平的，则"长序列"是成本而非收益，必须如实报告并撤回该主张。**

---

## 4. 超参汇总

| 超参 | 基线值 | 搜索范围 | 说明 |
|---|---|---|---|
| 结构 | **A（分解式）** | A / B / C | C 是对标竞品的下限 |
| 层数 $L$ | **6** | 2 / 4 / 6 / 8 | **KIST EMBC 2023 用 2 层**——我方需说明为何更深 |
| 注意力头数 | **8** | 4 / 8 / 16 | KIST 用 **4 头** |
| $d$ | **256** | 192 / 256 / 384 | 必须等于 tokenizer 输出维度 |
| FFN 维度 | **4d = 1024** | 2d / 4d | 标准 |
| Dropout | 0.1 | 0–0.3 | 小数据时上调 |
| **Temporal PE** | RoPE | 正弦 / RoPE / 可学习 | 见 §3.1 |
| **Region PE** | **可学习** | 可学习 / 无 / 正弦（对照） | 见 §3.1；"无"是消融项 |
| 注意力类型 | 全程 full | full / windowed | 长序列（$T\ge121$）时切 windowed |
| 权重初始化 | 前 2 层用 KIST 同规模配置作对照 | — | 便于"同规模对比" |

### 4.1 层数与头数的选型论证（必须能回答审稿人）

> **"为什么你用 6 层 8 头，而 KIST 用 2 层 4 头？"**

| 理由 | 论证 |
|---|---|
| 任务难度 | KIST 输出**单一步长信号**；ViKiT 输出**6 DOF 关节角序列**，输出维度和耦合度都更高 |
| 输入维度 | KIST 输入是**一维特征序列**；ViKiT 输入是 **(T,R,d) 二维张量**，需要更多层来混合两个轴 |
| **但** | 以上都是**先验论证**，**必须**用 $L \in \{2,4,6,8\}$ 的消融来**证明** 6 层确实优于 2 层，否则应降层以诚实对齐 |

> ⚠️ **若 $L=2$ 就够**，则应采用 2 层并**明确报告**"更小的模型已足够"——
> 这比硬撑一个更大的模型更可能通过审稿。

---

## 5. 失败模式与对策

| # | 失败模式 | 症状 | 根因 | 对策 |
|---|---|---|---|---|
| **F1** | **分解假设不成立** | 结构 A 显著劣于结构 B | 时空耦合强（如"肘-腕"跨帧耦合） | ① 退到结构 B + 高效 attention（FlashAttention / Linformer）；② 混合：每 2 层分解 + 1 层联合 |
| **F2** | **Region PE 学到无语义模式** | 打乱 region 顺序后性能不变 | region 轴实际未被使用 | ① **region 打乱测试**（permutation test）作为必做诊断；② 若打乱无影响 → 触发 R1 |
| **F3** | **长序列注意力退化** | $T$ 增大性能反而下降 | $O(T^2)$ 稀释；远距离注意力噪声 | ① windowed attention；② 相对位置偏置（ALiBi）；③ 层次化下采样（先粗后细） |
| **F4** | **RoPE 与变长推理冲突** | 训练 $T=81$、推理 $T=27$ 时崩坏 | RoPE 的频率外推性质 | ① 训练时随机采样多种 $T$（长度增强）；② 或改用可学习 PE 并固定 $T$ |
| **F5** | **过深导致小数据过拟合** | 训练 loss 极低，验证 loss 上升 | 6 层参数 ≫ 数据量 | ① 降层；② 更强 dropout / stochastic depth；③ 先用 CMU Mocap 预训练 |
| **F6** | **与 tokenizer 的联合不稳定** | 训练后期 loss 震荡 | 两个可学习模块互相追逐（tokenizer 在变，encoder 在追） | ① 前 5 epoch 冻结 encoder（配合 `Anatomy_aware_Tokenizer.md` §3 的冻结策略）；② 降低 encoder lr（0.5×） |
| **F7** | **Memory 维度与 decoder 不匹配** | cross-attention 报错或对齐错误 | $R$ 轴在 decoder 处如何被消费未定义 | 见 §6 契约；decoder 侧必须显式声明是**展平**还是**保留 R 轴** |

---

## 6. 与下游组件的接口契约

```python
# 契约（不得被下游打破）
memory = encoder(V)         # memory: (B, T, R, d)   —— 保留 R 轴
# Cross_Modal_Decoder 必须显式声明消费方式：
#   M1: 展平 (B, T*R, d)  → cross-attention 的 key/value 长度 = T*R = 648
#   M2: 保留 (B, T, R, d) → 先对 R 做 reshape，再跨 T 做 cross-attention
```

| 契约项 | 值 | 违反后果 |
|---|---|---|
| memory 形状 | **(B, T, R, d)，R 轴不得在 encoder 内被 flatten** | decoder 无法做 region-aware cross-attention |
| $d$ | 必须等于 tokenizer 与码本维度（256） | 维度不匹配 |
| 是否返回 attention | **可选但必须可开启** | 无法做可解释性分析（代偿定位需要 attention 可视化） |

---

## 7. 交付与验收

| # | 交付物 | 验收标准 |
|---|---|---|
| 1 | 结构 A/B/C 三实现 | 三者可切换；参数量与 FLOPs 报告 |
| 2 | $T$ 消融曲线 | $T \in \{9,27,45,81,121\}$ 的 MAE + jerk + 显存曲线 |
| 3 | $L$ 消融曲线 | $L \in \{2,4,6,8\}$；**必须能回答"为何不用 KIST 的 2 层"** |
| 4 | **Region 打乱诊断** | 输出打乱 region 顺序前后的性能差（若 ≈0 则触发 R1） |
| 5 | attention 可视化 | 能导出"腕角预测时注意力落在哪些 region/帧"的热图 |
| 6 | 单元测试 | 形状契约、因果性（temporal 轴不得看到未来）、mask 正确性 |

---

## 8. 风险登记

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| **R1** | **结构 C（纯时间，无 region 轴）与结构 A 表现相当** | region 轴无增益 → "anatomy-aware"在 encoder 层落空 | 与 `Anatomy_aware_Tokenizer.md` R1 联动。**缓解**：论证增益应出现在**物理合法性指标**（ROM/FK）与**可解释性**上，而非仅 MAE |
| **R2** | 分解式注意力被审稿人指为"早已有之"（如 Factorized Attention、TimeSformer） | novelty 被削弱 | **主动引用** Factorized/Separable attention 文献，把 novelty 锚定在**"R 轴是解剖单元而非空间 patch"**上 |
| **R3** | 6 层过深，实际收益低于成本 | 无谓的算力 | 用 $L$ 消融**主动降层**并如实报告——诚实的小模型比虚胖的大模型更可信 |
