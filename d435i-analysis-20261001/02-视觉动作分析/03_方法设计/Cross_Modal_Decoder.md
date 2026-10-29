# 04 · Cross-Modal Decoder（跨模态解码器）

> **所属**：ViKiT 组件 ④（target language 的生成）
> **上游**：`Spatio_Temporal_Encoder.md`（消费 memory `(B,T,R,d)`）、`Kinematic_Tokens_与_Grammar.md`（消费码本）
> **下游**：`Forecasting_Task.md`
> **一致性基准**：`../02_创新点矩阵/ViKiT_方法概览.md` §2.3
> **本文档定位**：Phase 2 方法设计的**实质设计文档**，含结构 / 超参 / 选型依据 / 失败模式

---

## 1. 设计目标：为什么叫 "Cross-Modal"

ViKiT 的核心命题是**翻译**：Visual Language → Kinematic Language。翻译成立的前提是**两种语言有各自的编码器**：

| | source language | target language |
|---|---|---|
| **模态** | 视觉（anatomy-aware visual tokens） | 运动学（kinematic tokens） |
| **编码器** | `Anatomy_aware_Tokenizer` + `Spatio_Temporal_Encoder` | **本解码器** |
| **"词"** | anatomy region token（$R=8$） | 关节角码字（$K=256$/DOF） |
| **"句法"** | 空间-时间注意力 | **Kinematic grammar（6 项损失）** |

> **与竞品的结构性差异**：KIST 的 CNN-Transformer（EMBC 2023，真实全文）的"解码"是
> **ResNet18 特征 → Transformer → 2 个全连接层**，即**同模态内的回归头**。
> 它没有独立的 target-language 编码器——**没有"两种语言"，只有"一个特征序列到一个标量"**。
> ViKiT 的 cross-attention 是**真跨模态**：视觉 memory 与运动学 token 序列是**两个不同的符号系统**。

---

## 2. 三种解码方案（按优先级）

### 2.1 方案 P1 · 并行解码（**基线主方案**）

```
Memory:  (B, T, R, d)   ← 视觉侧
   │
   ▼  一次性生成整段运动学序列
Kinematic tokens: (B, T, J)     J=6 DOF
   │
   ▼  VQ 码本查表 → 连续角
θ̂: (B, T, J)
```

| 优点 | 缺点 |
|---|---|
| **一次前向**，推理最快 | 无自回归结构，失去"翻译生成"的叙事 |
| 无 exposure bias | 难以做**未见过序列**的外推 |
| 训练稳定、易并行 | 长程依赖靠 encoder 承担 |

**实现**：用 $J$ 个可学习的 **query token**（每 DOF 一个）+ cross-attention 到 memory，
输出 $T \times J$ 的码字 logits。

### 2.2 方案 P2 · 自回归解码（**叙事主方案**）

```
Kinematic prefix:  k_{1:t-1}
        │
        ▼  逐 token 生成
   ┌──────────────────────────────────┐
   │ Masked self-attention (target 侧) │ ← 运动学语法在此生效
   ├──────────────────────────────────┤
   │ Cross-attention → 视觉 Memory     │ ← 跨模态对齐在此发生
   ├──────────────────────────────────┤
   │ FFN → 码字 logits                 │
   └──────────────────────────────────┘
        │
        ▼
   k_t  →  连续角 θ̂_t
```

| 优点 | 缺点 |
|---|---|
| **真正的 Seq2Seq**，与 Wei 2026（种子论文）的范式一致 | 推理慢（$T$ 步串行） |
| 支持**prefix 条件**（如给定起始姿态） | exposure bias |
| 可做 teacher forcing → 自由运行的课程学习 | 训练需 teacher forcing 调度 |

> ⚠️ **Wei 2026 的对照事实**：种子论文的 Transformer 模块明确是
> **"autoregressively generates the next joint angle"**（自回归生成下一关节角）。
> → **我方若采用 P1（并行），就与种子论文的范式不一致**，需要在 paper 中解释为什么改。
> → **建议：P1 作主方案（工程可行性），P2 作"与种子论文对齐的对照实验"**，两者都跑，用数据说话。

### 2.3 方案 P3 · 混合（diffusion / iterative refinement）

先用 P1 出粗解，再用少量迭代（2–4 步）精修。

| 优点 | 缺点 |
|---|---|
| 兼顾速度与精度 | 引入额外超参（步数、噪声调度） |
| 天然支持不确定性估计 | 实现复杂度高 |

**用途**：仅在 P1 精度不足且 P2 太慢时启用。

### 2.4 方案对比速查

| | P1 并行 | P2 自回归 | P3 混合 |
|---|---|---|---|
| 推理速度 | **最快** | 最慢 | 中 |
| 精度上限 | 中 | **高** | 高 |
| 与 Wei 2026 范式一致性 | ❌ | ✅ | 中 |
| 实现复杂度 | 低 | 中 | **高** |
| **建议** | **主方案** | **必做对照** | 备选 |

---

## 3. Cross-Attention 配置

### 3.1 memory 消费方式

`Spatio_Temporal_Encoder.md` §6 规定 memory 保留 R 轴 `(B,T,R,d)`。解码器有两种消费方式：

| 方式 | 机制 | query 数 | 优点 | 缺点 |
|---|---|---|---|---|
| **M1 · 展平** | flatten 成 `(B, T*R, d)` = `(B, 648, d)` | $J$ 个 DOF query × $T$ | 简单；与标准 Transformer 一致 | 丢失 region 结构；648 个 K/V 偏长 |
| **M2 · 保留 R 轴（**主方案**）** | 先对 R 做一次 pooling/attention 到 `(B,T,d)`，或按 DOF 选择相关 region | 每 DOF 选择**解剖相关的 region 子集** | **解剖先验进入解码器** | 实现稍复杂 |

#### 3.1.1 M2 的解剖路由（本设计的亮点）

利用已知的运动链关系，为每个 DOF **限定注意力范围**：

| 输出 DOF | 允许 attending 的 region 集合 | 理由 |
|---|---|---|
| 肘屈伸 | `r1 shoulder`, `r2 upper_arm`, `r3 elbow`, `r4 forearm` | 肘角由肩-肘-前臂链决定 |
| 腕屈伸/偏移 | `r3 elbow`, `r4 forearm`, `r5 wrist`, `r6 hand` | 腕角由前臂-腕-手链决定 |
| 肩（3 DOF） | `r0 torso`, `r1 shoulder`, `r2 upper_arm` | 肩角需躯干参照系 |
| **全部 DOF** | **`r7 global` 始终可用** | 保留全局上下文（见 `Anatomy_aware_Tokenizer.md` §2.2） |

> ⚠️ **这同时是一个可检验的假设**：如果解剖路由有效，则**打乱每 DOF 的允许集合**应导致性能下降。
> **该诊断必须做**（与 `Spatio_Temporal_Encoder.md` 的 region 打乱诊断配套）。

### 3.2 注意力超参

| 超参 | 基线值 | 搜索范围 | 说明 |
|---|---|---|---|
| cross-attention 头数 | **8** | 4 / 8 / 16 | 与 encoder 对齐 |
| decoder 层数 | **4** | 2 / 4 / 6 | **KIST 用 2 层**——需消融证明 |
| **解码器是否共享 K/V 投影** | **否** | 是 / 否 | 共享省参数但可能损失表达 |
| target self-attention | P2 才有 | — | P1 无此项 |
| attention dropout | 0.1 | 0–0.3 | — |
| 位置编码（target 侧） | 与 encoder 的 TPE 一致 | — | 保证两侧时间轴对齐 |

---

## 4. Autoregressive vs Parallel：决策矩阵

| 判据 | 倾向 P1（并行） | 倾向 P2（自回归） |
|---|---|---|
| 推理延迟要求（实时 AAN 类） | ✅ | ❌ |
| 与种子论文范式一致性 | ❌ | ✅ |
| 训练数据量 | 大 → 都可以 | 小 → P2 需更多正则 |
| 是否需要 prefix 条件 | ❌ | ✅ |
| **"翻译"叙事的说服力** | 弱 | **强** |
| 论文审稿风险 | "这不还是回归头？" | "为何回退到 RNN 式串行？" |

> 🔴 **这是一个必须由实验回答的问题，不能靠偏好决定。**
> **决策规则（预先登记，避免事后挑结果）**：
> 1. 两者都实现并训练
> 2. 若 P1 与 P2 在 MAE° 上差异 **< 5%（相对）** → **采用 P1**（选快的），并在 paper 中说明
> 3. 若 P2 显著更优（**≥ 5%**） → **采用 P2**，并用推理延迟数据论证可接受性
> 4. 无论选哪个，**另一个必须作为消融/对照出现在论文中**

---

## 5. 失败模式与对策

| # | 失败模式 | 症状 | 根因 | 对策 |
|---|---|---|---|---|
| **F1** | **Cross-attention 退化为恒等/忽略** | 去掉视觉输入后输出不变 | decoder 直接从 target 侧学到先验（角度分布本身可预测） | ① **必做诊断：视觉消融（输入置零/打乱帧序）**，输出必须显著劣化；② 加视觉-运动学对比损失；③ 检查 attention 熵 |
| **F2** | **解剖路由限制了必要连接** | 腕角精度上不去 | 真实代偿会让远端角度依赖躯干（**近端代偿**） | ① `r7 global` 兜底；② 路由改为**软偏置**（bias 而非硬 mask）；③ 用 attention 可视化验证 |
| **F3** | **自回归 exposure bias（P2）** | teacher forcing 好，自由运行崩 | 训练/推理分布不一致 | ① scheduled sampling；② 训练时混入自由运行片段；③ 或直接选 P1 |
| **F4** | **$J$ 个 DOF query 互相干扰** | 某关节角被另一关节拖累 | query 之间无结构约束 | ① 加 DOF 级解耦损失；② 用解剖路由（§3.1.1）天然解耦；③ 分关节 report 指标 |
| **F5** | **`(B,T,R,d)` memory 过长导致 cross-attention OOM** | 显存溢出 | M1 展平后 K/V 长度 648 | ① 用 M2；② 对 R 先 pooling；③ memory 下采样 |
| **F6** | **码本 logits 过于尖锐/平坦** | 训练不收敛或输出恒定 | temperature 设置不当 | ① 引入可学习 temperature；② 用 label smoothing；③ 监控码字熵 |
| **F7** | **teacher forcing 泄漏未来帧** | 验证指标虚高 | target 侧 mask 错误，或 encoder 看到了未来 | ① **因果性单元测试**（见 §7）；② 显式检查 mask |

---

## 6. 训练调度（若采用 P2）

| 阶段 | teacher forcing 比例 | epoch | 说明 |
|---|---|---|---|
| 1 | **1.0**（全 teacher forcing） | 20 | 先学对齐 |
| 2 | 线性降到 **0.5** | 30 | scheduled sampling |
| 3 | 固定 **0.5**，加入 grammar loss（见 `Kinematic_Tokens_与_Grammar.md` §4 的 Stage 2） | 50 | 联合微调 |

> ⚠️ 与 `Kinematic_Tokens_与_Grammar.md` §4 的两阶段训练**需要合并成统一的训练计划**，
> 不能各自两个阶段（会变成四阶段，调参空间爆炸）。
> **统一后**：Stage 1 = 码本预热（P1/P2 共用）+ teacher forcing 1.0；Stage 2 = grammar 注入 + TF 降至 0.5。

---

## 7. 交付与验收

| # | 交付物 | 验收标准 |
|---|---|---|
| 1 | P1 / P2 两实现 | 可切换；参数量、FLOPs、**推理延迟（ms/序列）**报告 |
| 2 | **视觉消融诊断** | 视觉输入置零 / 打乱帧序后，输出必须显著劣化（否则 F1 成立） |
| 3 | **解剖路由打乱诊断** | 打乱每 DOF 的允许 region 集合后性能变化 |
| 4 | attention 可视化 | 输出"predicted 肘角 ↔ 各 region/帧"的注意力热图 |
| 5 | **因果性单元测试** | 断言：改变 $t'>t$ 的输入**不得**影响 $t$ 时刻输出（P1 与 P2 均需） |
| 6 | 决策记录 | 按 §4 的**预先登记规则**记录 P1 vs P2 的最终选择与依据数据 |

---

## 8. 风险登记

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| **R1** | **Cross-attention 并非必要**（视觉消融后性能不显著下降） | "跨模态翻译"叙事被击穿——**这是全项目最大的单点风险** | 必须在**项目早期**做该诊断，而不是论文定稿前；若成立，需重新定位贡献（退到"结构化输出 + grammar"） |
| **R2** | 自回归（P2）推理延迟不可接受 | 实时应用主张失效 | 按 §4 规则选 P1；或 P2 + KV cache + 蒸馏 |
| **R3** | 审稿人指"cross-attention decoder"是标准组件 | novelty 被削弱 | **主动承认** cross-attention 本身不新；novelty 锚定在**解剖路由**（§3.1.1）与**六项 grammar 约束**上 |
| **R4** | P1 被指为"就是个回归头" | 与 Chao 2026 的区别被抹平 | 论证：P1 的 query 是**词表上的符号分布**（$K=256$ logits/DOF），而非单一标量回归；推理时虽然输出连续值，**但训练目标是语言建模目标** |
