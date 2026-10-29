# 03_方法设计 — ViKiT 方法设计文档索引

> **状态**：✅ **已就位**（2026-09-30 补齐 5 篇实质设计文档）
> **最后更新**：2026-09-30
> **父文档**：[`../02_创新点矩阵/ViKiT_方法概览.md`](../02_创新点矩阵/ViKiT_方法概览.md)（**唯一一致性基准**）

---

## 1. 本目录 5 篇文档

| # | 文档 | 对应 ViKiT 组件 | 核心内容 | 行数 |
|---|---|---|---|---|
| 1 | [`Anatomy_aware_Tokenizer.md`](Anatomy_aware_Tokenizer.md) | ① source 编码 | 三种切分方案（S1 关键点引导／S2 语义 mask／S3 可学习）、8 个 anatomy region 定义、token 维度与分配温度 $\tau$、与竞品空间表征的对照 | — |
| 2 | [`Kinematic_Tokens_与_Grammar.md`](Kinematic_Tokens_与_Grammar.md) | ②④⑤ target 词表 + 解码 + 约束 | 码本设计（$K$=256，ROM $(-120°,120°)$，3 种初始化）、**6 项 grammar loss** 的完整公式与权重、两阶段训练、消融映射 | — |
| 3 | [`Spatio_Temporal_Encoder.md`](Spatio_Temporal_Encoder.md) | ③ 时序编码 | 分解式时空注意力（vs 联合 vs 纯时间）、**双轴位置编码**（TPE 时间 + RPE region）、序列长度 $T$ 的选型与对手对标 | — |
| 4 | [`Cross_Modal_Decoder.md`](Cross_Modal_Decoder.md) | ④ 跨模态生成 | 并行（P1）／自回归（P2）／混合（P3）三方案、**解剖路由**（按 DOF 限定注意力 region）、autoregressive vs parallel 的预登记决策规则 | — |
| 5 | [`Forecasting_Task.md`](Forecasting_Task.md) | ⑥ 加分项 | 因果未来翻译（$h$ = 100/200/300 ms）、**诚实定位**（300 ms 非我方首创）、RGB vs sEMG 的信息带宽约束、评估协议与反数据泄漏要求 | — |

---

## 2. 组件依赖顺序

```
RGB Video
    │
    ▼
① Anatomy_aware_Tokenizer        → (B, T, R, d)   R=8, d=256
    │
    ▼
③ Spatio_Temporal_Encoder        → memory (B, T, R, d)   保留 R 轴
    │
    ├────────────────────────────────────────┐
    ▼                                        ▼
④ Cross_Modal_Decoder              ② Kinematic Tokens + ⑤ Grammar
   （cross-attention + 解剖路由）       （256 词表/DOF + 6 项 loss）
    │
    ▼
⑥ Forecasting_Task（可选加分项）  → Kinematics_{t : t+h}
```

---

## 3. 贯穿 5 篇的三条硬约束

| # | 约束 | 出处 | 违反后果 |
|---|---|---|---|
| **C1** | **数据维度 $d = 256$ 在全链路上必须一致**（tokenizer 输出 = encoder 输入输出 = 码本维度） | `Anatomy_aware_Tokenizer.md` §5 | 码本无法对齐 |
| **C2** | **R 轴（anatomy region）在 encoder 内不得被 flatten**，保留到 decoder | `Spatio_Temporal_Encoder.md` §6 | decoder 无法做 region-aware 解剖路由 |
| **C3** | **R 轴顺序固定为 `[r0..r7]`**（torso/shoulder/upper_arm/elbow/forearm/wrist/hand/global） | `Anatomy_aware_Tokenizer.md` §2.2 | region positional encoding 失效 |

---

## 4. 三条最脆弱的设计假设（必须在 Phase 3 用实验检验）

这 5 篇文档都在不同位置标注了失败模式与风险，其中**三条是全局性的**：

| # | 假设 | 若被证伪 | 检验方式 | 登记位置 |
|---|---|---|---|---|
| **H1** | **解剖先验有效**（S1/S2 优于 S3） | "anatomy-aware"命名被质疑为无用归纳偏置 → 动摇核心创新点 2 | $\tau$ 扫描 + region 打乱诊断 | `Anatomy_aware_Tokenizer.md` R1；`Spatio_Temporal_Encoder.md` R1 |
| **H2** | **Cross-attention 并非冗余**（视觉消融后输出显著劣化） | "跨模态翻译"叙事被击穿 → **全项目最大单点风险** | 视觉输入置零／打乱帧序诊断 | `Cross_Modal_Decoder.md` R1 |
| **H3** | **长序列带来增益**（$T$=81 优于 $T$=27） | "长上下文"是成本而非收益 | $T \in \{9,27,45,81,121\}$ 消融曲线 | `Spatio_Temporal_Encoder.md` §3.3 F5 |

> ⚠️ **H1–H3 必须在项目早期检验**，不能在论文定稿前才发现。
> 三个假设各自都有预设的"若被证伪怎么办"（见各文档风险登记），保证项目不会被单点证伪卡死。

---

## 5. 与竞品的可数差异汇总（论文差异论证的弹药库）

| 维度 | Chao 2026 Measurement（直接竞品） | KIST CNN-Transformer（EMBC 2023，真实全文） | Wei 2026 JBHI（种子，sEMG） | **ViKiT（我方）** |
|---|---|---|---|---|
| 空间表征 | CNN 特征图 | ResNet18 特征 | 3DCNN（肌肉解剖） | **Anatomy-aware tokens（运动链）** |
| 显式 token 化 | ❌ | ❌ | ⚠️ 隐式 | ✅ **显式** |
| 时序模块 | Transformer | Transformer（2 层 4 头） | Transformer（自回归） | **分解式时空 Transformer（6 层）** |
| 物理约束项数 | **2** | **1**（MAE） | 摘要未提 | **6** |
| 约束作用层 | 仅损失层 | 仅损失层 | — | **损失层 + 表征层（离散化）** |
| 输出结构 | 点估计 | 点估计（逐步伐标量） | 自回归点估计 | **序列级联合翻译** |
| Forecasting | ❌ | ❌ | ✅ 300 ms | ✅ 100/200/300 ms（**带物理约束**） |
| 姿态机依赖 | 端到端（不依赖） | **明确不依赖**（原文强调） | — | 设计目标为**不依赖**（S1 有依赖风险，见 F4） |

> **最强的一条**：**物理约束项数 1–2 vs 6**，且**约束同时作用于离散表征层**。
> **这在数量级上可审计、可复现，不是命名差异。**

---

## 6. 变更记录

| 日期 | 变更 | 说明 |
|---|---|---|
| 2026-09-30 | **目录就位** | 由仅有占位 README 的状态补成 **5 篇实质设计文档（约 57 KB）**；README 改写为索引 |
