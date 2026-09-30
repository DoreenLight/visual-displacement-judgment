# 02 · Kinematic Tokens 与 Biomechanical Grammar

> **所属**：ViKiT 组件 ②（target language 词表）与组件 ⑤（物理约束）
> **上游**：`Anatomy_aware_Tokenizer.md`
> **下游**：`Cross_Modal_Decoder.md`（消费码本）、`../04_实验设计/Ablation设计.md` A2/A3
> **一致性基准**：`../02_创新点矩阵/ViKiT_方法概览.md` §2.4 / §3
> **本文档定位**：Phase 2 方法设计的**实质设计文档**，含结构 / 超参 / 选型依据 / 失败模式

---

## 1. 设计目标

ViKiT 的核心命题是把关节运动学当作**一种语言**。这要求两件事同时成立：

1. **词表（Vocabulary）**：连续角度必须能被**离散化**为符号 → 构成 target language 的"词"
2. **语法（Grammar）**：符号序列必须受**生物力学约束** → 不是任意符号串都合法

> **二者缺一不可**：只有词表 → 是"量化回归"（quantized regression），不是语言；
> 只有语法 → 没有语言的可组合性。**这正是回应"换名字的 regression"质疑的技术论证。**

---

## 2. Kinematic Vocabulary 设计

### 2.1 词表构造

每个关节自由度 $j$ 独立构造码本（**不共享码本**）。

$$\theta_j \in [\theta_j^{\min}, \theta_j^{\max}] \xrightarrow{\ \text{VQ}\ } k_j \in \{1,\dots,K\}$$

| 参数 | 基线值 | 依据 |
|---|---|---|
| 每自由度码本大小 $K$ | **256** | 与 `ViKiT_方法概览.md` 一致；$K=256 \Rightarrow$ 角度分辨率 $\approx$ ROM/256 |
| ROM 范围 $[\theta^{\min},\theta^{\max}]$ | **$(-120°, +120°)$** | 与概览一致；覆盖主要关节极限 |
| 每自由度分辨率 | $\approx 0.94°$（$240°/256$） | **优于临床可接受误差（3–5°）**，故量化不是精度瓶颈 |
| 关节自由度（初始） | 腕（2 DOF）+ 肘（1）+ 肩（3） = **6 DOF** | 与 `08_三维动作分析展示台` 的腕肘肩能力对齐 |
| 词表总量 | $6 \times 256 = 1536$ 个关节级符号 | — |

#### 2.1.1 码本初始化（关键超参）

| 方案 | 机制 | 优点 | 缺点 | 建议 |
|---|---|---|---|---|
| **U1 · 均匀网格** | 等间隔划分 ROM | 简单、可复现、**无偏** | 不匹配角度分布（多数日常动作集中在 ROM 中部） | **基线** |
| **U2 · k-means on 训练集角度分布** | 按数据分布聚类 | 分辨率匹配实际用法 | 引入数据偏置，跨被试泛化存疑 | **主方案（对照 U1）** |
| **U3 · 可学习码本（VQ-VAE 风格）** | 端到端学习 | 与下游任务最匹配 | 码本崩塌风险（见 F2） | **消融项** |

> ⚠️ **U1 vs U2 是必做的对照**：若 U2 显著优于 U1，说明**数据分布先验**比**物理均匀性**更有用——
> 这会影响我们对"language"的解释（是物理语言还是统计语言）。**这是概念层面的实验，不只是工程调参。**

### 2.2 码本与连续角的桥接

训练时使用**直通估计（straight-through estimator, STE）**：

$$\hat{\theta}_j = \theta_{k_j} + \mathrm{sg}(\theta_j - \theta_{k_j})$$

其中 $\mathrm{sg}(\cdot)$ 为 stop-gradient。**推理时输出连续值**（不是符号），
使 ViKiT 与回归基线在**同一输出空间**比较——**这是公平对比的前提**。

---

## 3. Biomechanical Grammar：6 项损失

$$L_{\text{total}} = \underbrace{L_{\text{angle}}}_{\lambda_0} + \underbrace{\lambda_1 L_{\text{vel}} + \lambda_2 L_{\text{acc}} + \lambda_3 L_{\text{ROM}} + \lambda_4 L_{\text{FK}} + \lambda_5 L_{\text{temporal}}}_{\text{biomechanical grammar}}$$

### 3.1 逐项定义

| # | 项 | 公式 | 物理含义 | 基线权重 |
|---|---|---|---|---|
| 1 | **$L_{\text{angle}}$** | $\frac{1}{TJ}\sum_{t,j}\lvert \hat\theta_{t,j}-\theta_{t,j}\rvert$ | 角度保真（主任务） | $\lambda_0 = 1$ |
| 2 | **$L_{\text{vel}}$** | $\frac{1}{(T-1)J}\sum_{t,j}\lvert \Delta\hat\theta_{t,j}-\Delta\theta_{t,j}\rvert$ | 角速度一致性 | $\lambda_1 = 0.5$ |
| 3 | **$L_{\text{acc}}$** | $\frac{1}{(T-2)J}\sum_{t,j}\lvert \Delta^2\hat\theta_{t,j}-\Delta^2\theta_{t,j}\rvert$ | 角加速度一致性（抑制抖动） | $\lambda_2 = 0.1$ |
| 4 | **$L_{\text{ROM}}$** | $\frac{1}{TJ}\sum_{t,j}\max(0,\ \hat\theta_{t,j}-\theta_j^{\max}) + \max(0,\ \theta_j^{\min}-\hat\theta_{t,j})$ | **解剖不可能角度**（铰链越界） | $\lambda_3 = 1.0$ |
| 5 | **$L_{\text{FK}}$** | $\lVert \mathrm{FK}(\hat\theta_t) - \mathrm{FK}(\theta_t)\rVert_1$ 于**末端执行器位置** | 正向运动学一致性（骨长不失真） | $\lambda_4 = 0.5$ |
| 6 | **$L_{\text{temporal}}$** | $\lVert \nabla_t \hat\theta\rVert_1$（平滑先验，**无标签**） | 时序连贯性（半监督） | $\lambda_5 = 0.01$ |

### 3.2 与竞品的约束数量对照（核心差异论据）

| 方法 | 物理约束项数 | 具体 |
|---|---|---|
| **Chao 2026 Measurement**（直接竞品） | **2** | 角度误差 + 角速度误差（$L = L_{\text{angle}} + \lambda L_{\text{angular\_velocity}}$） |
| **Jamsrandorj 2023 EMBC**（同团队，真实全文） | **1** | 仅 MAE |
| **Kumar 2022 EMBC**（真实全文） | **1** | 仅 MAE |
| **ViKiT（我方）** | **6** | 上表 |

> **一句话差异**：竞品把生物力学先验压成 **1–2 个软惩罚项**；
> ViKiT 把生物力学**结构**拆成 **6 项可独立审计**的 grammar loss，
> 且约束**同时作用于表征层（token 离散化）而非仅损失层**。
> **这是可数的差异（1–2 vs 6），不是命名差异。**

### 3.3 约束是否"越界"的立场

$L_{\text{ROM}}$ 与 $L_{\text{FK}}$ 都是**硬物理先验**。必须诚实说明：

- 它们**不是**"从数据学到的规律"，而是**人体解剖学的既有事实**
- 因此**不能声称这是 ML 创新**；创新在于**如何把它们组合成可审计的 grammar**
- 若审稿人问"这不就是加了正则项吗" → 回答：**是，但竞品只加了 1–2 项，
  且没有把约束作用于离散表征层**。差异在于**约束的深度与可审计性**，不在于是否使用约束。

---

## 4. 两阶段训练策略

| 阶段 | 目标 | 冻结项 | 损失 | epoch |
|---|---|---|---|---|
| **Stage 1 · 码本预热** | 让码本收敛到合理的角度分布 | 冻结 tokenizer + decoder 主干 | 仅 $L_{\text{angle}} + 0 \cdot L_{\text{ROM}}$ | 20 |
| **Stage 2 · 联合微调** | 注入 grammar，全网络微调 | 无冻结 | $L_{\text{total}}$（全 6 项） | 80 |

> **为什么要两阶段**：若一开始就开全部 grammar，$L_{\text{ROM}}$ 会与 $L_{\text{angle}}$ **对抗**——
> 当预测严重偏离 ROM 时，$L_{\text{ROM}}$ 的梯度会把输出拉回边界，
> 而真实角度可能恰好接近边界（如肘全屈）。**这会让 Stage 1 根本无法收敛**（见 F4）。

### 4.1 优化器与调度

| 项 | 值 | 依据 |
|---|---|---|
| 优化器 | **AdamW** | 与概览一致 |
| 初始 lr | **1e-4** | 与概览一致；跨模态对齐不宜过大 |
| 权重衰减 | 0.01 | — |
| 调度 | **cosine** with warmup 5 epoch | — |
| Batch size | **32–64** | 与概览一致 |
| 梯度裁剪 | 1.0 | grammar 项梯度尺度不一，必须裁剪 |

---

## 5. 失败模式与对策

| # | 失败模式 | 症状 | 根因 | 对策 |
|---|---|---|---|---|
| **F1** | **量化损失不可恢复** | MAE 始终高于回归基线 | $K=256$ 的量化误差在**高频运动**段被放大（角度变化率 > 码间距离/帧） | ① 提高 $K$ 到 512/1024；② 按关节自适应 $K$（运动幅度大的自由度给更大 $K$）；③ 残差量化（residual VQ） |
| **F2** | **码本崩塌（codebook collapse）** | 实际只用了少数几十个码字 | U3 可学习码本 + STE 的经典问题 | ① 码本 EMA 更新；② commitment loss；③ 定期重置死码字；④ **优先用 U1 均匀网格**（无崩塌风险） |
| **F3** | **Grammar 项互相抵消** | 总 loss 下降但各指标不改善 | $\lambda$ 尺度失配（$L_{\text{FK}}$ 量纲为长度，$L_{\text{angle}}$ 为弧度） | ① **量纲归一化**：所有项先除以自身的初始 loss 值；② 逐项 log 监控；③ 用不确定性加权（learned uncertainty weighting） |
| **F4** | **$L_{\text{ROM}}$ 与 $L_{\text{angle}}$ 对抗** | Stage 1 收敛但 Stage 2 反弹 | 真值本身贴近 ROM 边界时，ROM 惩罚把预测拉离真值 | ① 两阶段训练（本文 §4）；② 把 $L_{\text{ROM}}$ 改为**只在越界时才生效的单边惩罚**（已如此设计）；③ 真值先做 ROM 合法性检查 |
| **F5** | **$L_{\text{FK}}$ 需要准确的骨长参数** | FK 项贡献噪声 | 不同被试骨长不同，统一 FK 模型有偏 | ① 按被试归一化肢体长度；② FK 用**相对**末端位置误差而非绝对；③ 若骨长不可得则降级为 $L_{\text{ROM}}$ only |
| **F6** | **$L_{\text{temporal}}$ 过度平滑** | 运动"变糊"，快速动作被抹平 | 平滑先验与真实高频运动冲突 | ① $\lambda_5$ 取**极小值 0.01**；② 用 jerk 的 **Huber** 而非 L1；③ 只在低置信区间启用 |
| **F7** | **两阶段切换点敏感** | 结果对 Stage 1 epoch 数敏感 | 切换时码本尚未稳定 | ① 用码本利用率作为切换判据（而非固定 epoch）；② 报告多种切换点的敏感性分析 |

---

## 6. 消融映射（与 `../04_实验设计/Ablation设计.md` 对齐）

| 消融 | 对应本设计 | 假设 | 预期结论 |
|---|---|---|---|
| **A2 · 去 grammar loss** | 只留 $L_{\text{angle}}$，$\lambda_{1..5}=0$ | 物理合法性显著变差 | **ROM violation rate 与 FK consistency 应显著劣化**，而 MAE 可能几乎不变（**这正是我方要证明的**：精度不是唯一目标） |
| **A3 · 去 tokenization** | 去掉 VQ，直接回归连续角 | 量化不是精度瓶颈，但 token 化带来看不见的收益 | 预期 MAE 差异 **不显著**，差异应出现在 **temporal jerk**（token 序列更平滑） |
| **A4 · 去 forecasting** | 只做同步翻译 | 见 `Forecasting_Task.md` | — |
| **A5 · 去 cross-attention** | 见 `Cross_Modal_Decoder.md` | — | — |

> ⚠️ **A3 的风险**：若去 tokenization 后**所有指标都不变**，
> 则"Kinematic Language"这一命名的必要性会被击穿（**这是我方最脆弱的一环**）。
> → **缓解**：必须预先设计能体现 token 化收益的指标（temporal jerk、长序列外推、零样本关节组合），
> 并**在实验设计阶段就锁定**，而不是事后挑指标。

---

## 7. 交付与验收

| # | 交付物 | 验收标准 |
|---|---|---|
| 1 | 码本构建模块（U1/U2/U3） | 三种初始化可切换；码本利用率可报告（死码字比例） |
| 2 | 6 项 loss 实现 | 逐项可单独开关；量纲归一化后各项初始量级一致（±10×） |
| 3 | 两阶段训练脚本 | 切换点可配置；能输出切换点敏感性曲线 |
| 4 | 消融可运行 | A2/A3 可一键运行并输出全部 8 项指标 |
| 5 | 单元测试 | ROM 边界、STE 梯度流通、FK 数值正确性（与解析解对比） |
