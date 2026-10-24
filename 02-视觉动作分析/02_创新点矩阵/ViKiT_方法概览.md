# ViKiT 方法概览

> **模型全名**：An Anatomy- and Biomechanics-Informed Visual-to-Kinematic Sequence Translation Framework for Continuous Joint Kinematics Estimation
> **简称**：**ViKiT — Visual-to-Kinematic Translator**
> **核心命题**：$$\boxed{\text{Visual Language} \rightarrow \text{Kinematic Language}}$$

---

## 1. 整体架构（文字版）

> 🚨 **2026-10-02 修订（G1 / G3 / G7）**：本节原为**下肢口径**（hip/thigh/knee/shank/ankle）、
> **把离散当唯一输出路径**、且**未声明串联链拓扑**。三者均已改。
> 📌 **上一版架构见文末「附录 A（原架构，已作废）」**，保留仅为对照，不得再引用。

```
┌────────────────────────────────────────────────────────────────────┐
│                          输入：RGB Video I_{1:T}                   │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  ① Anatomy-aware Region Encoder（source side）                    │
│     - 按【上肢】运动链把每帧切成 8 个 anatomy regions             │
│       r0 torso · r1 shoulder · r2 upper_arm · r3 elbow ·          │
│       r4 forearm · r5 wrist · r6 hand · r7 global                 │
│     - 每个 region 用小型 ViT 编码 → V_t = [v_{t,r0}, ..., v_{t,r7}]│
│     - 输出：visual region embedding sequence V_{1:T}              │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  ② Spatio-Temporal Transformer Encoder                            │
│     - Spatial attention：同帧 region 间关系（肘-腕耦合）          │
│     - Temporal attention：跨帧时间动态                             │
│     - 输出：时序视觉表征 H_{1:T}                                  │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  ③ Cross-Modal Kinematic Decoder                                  │
│     - Query: kinematic queries（每个对应一个【上肢】关节自由度）   │
│     - Cross-attention: Q attends to H                             │
│     - 输出：per-DoF 隐状态 + 离散符号 logits                      │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  ④a 离散运动学词表 Kinematic Vocabulary + VQ Decode              │
│     - 每个 kinematic token ∈ {token_1, ..., token_N}              │
│     - 角色：**可读、可审计、供 grammar 约束的中间表示**          │
│     🚨 不再是唯一输出路径（见 ④b）                                │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  ④b 连续回归头 Continuous Regression Head  ←【G1 新增，必做】    │
│     - 直接输出 ISB 角度 â_{1:T}（肩 3 / 肘 2 / 腕 3 DoF）         │
│     - 依据：Pohl 2026 表 S2 —— 临床优先级前 10 名里              │
│       **只有 4 项是关节角**（榜首 Task duration 100%）            │
│     - 离散词表与连续量【并存并同时暴露】，不是二选一               │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  ④c 肌肉骨骼第二输出头  ←【G5 新增，可选，依赖未核实】          │
│     - 与角度头并行输出一个肌肉骨骼量（关节力矩 / 激活粗估计）    │
│     - 作用：给 grammar 真正的力学量可约束（L_dyn）               │
│     ⚠️ 数据可得性未核实 → 未验证前不得写进贡献清单               │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  ⑤ Biomechanical Grammar Loss（多目标训练）                       │
│     L = L_angle + λ_1 L_velocity + λ_2 L_acceleration +           │
│         λ_3 L_ROM + λ_4 L_FK + λ_5 L_temporal                    │
└────────────────────────────────────────────────────────────────────┘
```

### 📌 串联链拓扑声明（G7 · 2026-10-02 研究者裁定，V6c = A）

> **本项目采用 ISB 串联链口径：前臂旋前/旋后记在【腕】。**
>
> | 关节 | DoF | 内容 |
> |---|---|---|
> | 肩 shoulder | **3** | 屈伸 · 外展内收 · 轴向旋转 |
> | 肘 elbow | **2** | 屈伸 · 内收（对应腕的旋前旋后轴）|
> | 腕 wrist | **3** | 屈伸 · 外展内收 · **旋前旋后** |
>
> 📌 **依据**：ISB Part II（`10.1016/j.jbiomech.2004.05.042`，库内全文）定义上肢为肩 3 / 肘 2 / 腕 3。
> 🚨 **文献无共识**：Wang 等 2019（`10.1177/0954411919872400`）原文 *"no agreement has been reached on
> the most appropriate modelling approach of the forearm"* —— **这是设计裁定，不是文献结论，必须自己论证。**
>
> **⚠️ 本项目的诚实边界（必须写进 Limitations）**：
> 词表**按 ISB 3-DoF 腕的结构定义**，但**公开动捕数据集不提供前臂旋前旋后**（裁定 V8 = A 的直接后果），
> 因此**腕的第三个 DoF 槽位显式留空并记录**，标为 future extension point，
> **而不是静默降为 2 DoF**。
> ✅ 可写：*"Our vocabulary is structured for the full ISB 3-DoF wrist, yet public motion-capture
> datasets do not provide forearm rotation; we therefore leave the third wrist slot explicitly
> unfilled and documented."*
> 🚨 不可写："our vocabulary covers forearm rotation" / "clinicians need pronation/supination measured"
> —— 后者被 Pohl 2026 表 S2 **直接证伪**（临床师把它排末位并剔出最终 17 项运动学指标）。

---

## 2. 四个核心组件

### 2.1 Anatomy-aware Region Encoder

**目标**：把"普通 ViT patches"换成"按【上肢】解剖链切分的 region embedding"。

| 项 | 内容 |
|---|---|
| 切分依据 | **上肢**代偿链（torso → shoulder → upper_arm → elbow → forearm → wrist → hand）+ global |
| region 数 | **8**（`r0`–`r7`），定义与依据见 `../03_方法设计/Anatomy_aware_Region_Encoder.md` §2.2 |
| 切分方法 | (1) 用人体关键点引导 attention；(2) 用 semantic segmentation mask；(3) 用可学习的 soft assignment |
| 编码器 | 局部小型 ViT（每个 region 一个 ViT-S/16 或类似），**跨 region 共享权重** |
| 输出 | $V_t = [v_{t,\text{torso}}, v_{t,\text{shoulder}}, \ldots, v_{t,\text{global}}]$，每个 $v_{t,r} \in \mathbb{R}^d$ |

> 🚨 **G3 修订说明**：本节原通篇 `hip / thigh / knee / shank / ankle`（**下肢**），
> 而本项目的目标域是**上肢康复**、竞品与数据也都在上肢。**口径必须统一，否则方法章与实验章对不上。**
> 详细 region 语义表（`r0`–`r7` 及各自临床意义）**唯一规范路径**是
> [`Anatomy_aware_Region_Encoder.md`](../03_方法设计/Anatomy_aware_Region_Encoder.md) §2.2，本文档不重复定义。

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

**目标**：提供**可读、可审计、供 grammar 约束的中间表示**。

> 🚨 **G2 修订（2026-10-02）**：本节原以 `torch.bucketize` 标量分箱为方案，**并把"分箱"等同于"语言"**。
> **这两件事不等价。** 分箱只是量化 —— 它满足"连续→离散"，但**不满足"这是一个可复用的符号系统"**。
> 下面先说清判据，再给方案。

#### 📌 判据：什么才算「词表」，而不是「量化」

| 判据 | 说明 | 证据锚点 |
|---|---|---|
| **① 逐符号独立可指定** | 一个符号必须能被单独指定，且语义不依赖别处 —— 否则它没有词义，只是编码 | **Schlegel 等 2024**（F24，`2406.17443`）第 4 条优势原文：*"Individual movements are decoupled, the angles at each joint are independent of the angles elsewhere"* |
| **② 符号集要小到能复用** | 太大则无法被人记住和复用，词表就失去意义 | **F20**（Nishimura 等 2021，IEICE Trans.）原文：*"the number of 'action concepts' is saturated and reused"*、*"could represent various case studies by using a small number of action concepts"* |
| **③ 但不能小到饱和** | ②的对面风险：**人工/规则定义的符号表会迅速饱和** | **F20** 同文 + **DigitCode**（`2608.03127`）实测：26 个方向字母表中 **68.1% 骨方向落在单一 +Y 面** |
| **④ 边界须有临床/生物力学意义** | 分箱边界若纯等距，则符号不对应任何可解释的运动学事件 | ① ISB 角度是临床事实标准（**F24**）；② 临床指标清单见 Pohl 2026 表 S2（**17 个运动学指标**）；③ 🆕 **测量精度论据（E12 / Cleffken 2007）** —— 肘关节测试者重测 SDD 为 **4.2°–10.8°**（主动 ROM **8.5°**），在 ±5–10° 的噪声下**等距细边界会被噪声反复跨界**，而 ROM 锚定边界的分类结论在噪声下仍成立 → **输出粒度必须匹配测量精度**；④ 🆕 **可推导的粒度上限（M11）** —— 由 SDD 8.5° 换算单次测量 σ ≈ 3.07°，**本节下方 baseline 的 0.94°/符号在重测时约 83% 会变**，**要达 90% 重测稳定性需符号宽 ≥ 7.1°**（95% 需 ≥ 8.5°）→ **baseline 过细 8–9 倍** |

🚨 **由此得出的方案约束（本节新增，不可省）**：
1. **等距分箱只能当 baseline**，不能当主方案 —— 它必然违反判据 ④。
2. **必须同时报告「符号使用统计」**，且**作为一级指标**，不能只报 MAE：
   **实际用到的不同符号数 / 词表容量**、**每符号熵**、**跨受试者符号一致性**。
   → **若符号表大面积闲置，词表主张即被自己的实验证伪。**
   📌 **这条直接对应消融 `../04_实验设计/Ablation设计.md` 的 ABL-A1 族（region / 词表打乱）**。
3. **分箱边界应锚定有意义的角度事件**（ROM 边界、任务阶段切换点），或改用 **learned codebook（VQ-VAE）** 并加**使用率约束**。

```python
# 伪代码 —— 【baseline】，仅用于对照，不是推荐主方案
class KinematicVocabulary:
    def __init__(self, rom_range=(-120, 120), n_bins=256):
        # ⚠️ 等距分箱：违反判据 ④（边界无生物力学意义）
        # ⚠️ 逐 DoF 独立分箱：违反判据 ① 的耦合要求
        # 🚨🚨 粒度也过细（2026-10-03 新增，E12/M11 推导）：
        #    255 区间跨 240° → 每符号仅 0.94°；
        #    肘关节单次测量 σ ≈ 3.07°（由 SDD 8.5° 换算）→
        #    P(重测落同一符号) ≈ 0.17，即约 83% 的符号会在重测时改变。
        #    要达 90% 稳定性需符号宽 ≥ 7.1°，本设置细了约 8 倍。
        #    —— 即便只作 baseline，也应同时报 M11 以量化这一点。
        self.bins = torch.linspace(*rom_range, n_bins)

    def encode(self, angle_continuous):
        token_id = torch.bucketize(angle_continuous, self.bins)
        return token_id

    def decode(self, token_id):
        return self.bins[token_id]
```

| 项 | 内容 | 状态 |
|---|---|---|
| Vocabulary 大小 | 256（每个关节自由度）| 待由 A/B 实测确定，**不得当作既定** |
| 编码方式 | (a) 等距 `bucketize` —— **baseline**；(b) **ROM 锚定边界**；(c) **learned VQ codebook + 使用率约束** —— 推荐 | (a) 已实现但证据不足；(b)(c) 待做 |
| 解码方式 | bin 中点 或 VQ decode | —— |
| 🚨 **腕第三 DoF** | **槽位保留但公开数据填不满 → 显式留空并记录**（见 §1 链拓扑声明） | 已裁定 |
| 🚨 **输出是否只走离散** | **否** —— 连续回归头并存（见 §1 的 ④b） | G1 已改 |

**借鉴**：TokenHMR 的 pose vocabulary + token prediction。我们把"pose vocabulary"换成"kinematic vocabulary"。
📌 **但注意 F9 的区分**：TokenHMR 的 codebook 是**学出来的、不带解剖结构**（符号无解剖含义、不可检视不可编辑）；
我们的主张是**符号有生物力学结构、可被人读**。**这个差别必须靠上面的「符号使用统计 + 边界锚定」兑现，否则只是措辞差别。**


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
| 🆕 $L_{\text{dyn}}$ | **动力学一致性** | 预测角度 → FK → 与第二头预测的肌肉骨骼量做**逆动力学闭合**残差 |

**总损失**：

$$L = L_{\text{angle}} + \lambda_1 L_{\text{velocity}} + \lambda_2 L_{\text{acceleration}} + \lambda_3 L_{\text{ROM}} + \lambda_4 L_{\text{FK}} + \lambda_5 L_{\text{temporal}} + \lambda_6 L_{\text{dyn}}$$

> 🚨 **G5 修订（2026-10-02）**：原式**六项全是运动学**，`L_FK` 也是运动学约束 ——
> **也就是说「Biomechanical Grammar」这个名字当时名不副实：一条力学约束都没有。**
> 现补入 `L_dyn`，并要求**第二肌肉骨骼输出头**（见 §1 的 ④ 之后），
> 否则 `L_dyn` 只能靠外部模型间接约束，grammar 里用不上任何力学量。
>
> 📌 **`L_dyn` 不是我们的原创主张，须正面处理 Related Work**：
> - **Fang AAAI 2018**（`10.1609/aaai.v32i1.12270`）原文 **"WITHOUT CONSIDERING FORCES"**；
> - **Xu TPAMI 2022**（`10.1109/tpami.2021.3087695`）原文**已明确引入 kinematics** ——
>   🚨 **这作废了「后继 grammar 都不含力学」这条旧依据**（F6/F10 原判）。
>   **现在能站住的只有 Fang 的单点明文**，**不可写「我们首次把力学引入 pose grammar」**。

**对比参照（原写 Chao 2026，但其为【下肢】hip/knee/ankle）**：

| 维度 | Chao 2026（**下肢**） | ViKiT（**上肢**） |
|---|---|---|
| 身体部位 | hip / knee / ankle | **肩 3 / 肘 2 / 腕 3（ISB Part II）** |
| 物理约束数量 | 1（velocity）| **6 + FK + 动力学项** |
| 显式 ROM 限制 | ❌ | ✅（上肢 ROM 边界） |
| FK 一致性 | ❌ | ✅ |
| 关节耦合 | ❌ | ✅（via FK）|
| 🆕 离散词表 | ❌ | ✅（F19：**pose grammar 线上为空**）|

> ⚠️ **这张表只能当"约束条数"对照，不能当主要竞品对照** ——
> **两者身体部位不同。** 上肢的直接竞品见
> [`创新点查重_vs_竞品.md`](创新点查重_vs_竞品.md) **§0.5（18 篇 A/B 级上肢竞品）**。


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
| Region encoder | 随机初始化 + 弱监督（用现成的 2D keypoints）|
| Kinematic decoder | 随机初始化 |
| VQ | 用 ground-truth 角度先训 VQ-VAE |
| 总训练 | 两阶段：先训 VQ，再 end-to-end 训整个翻译模型 |
| 优化器 | AdamW，lr=1e-4，cosine schedule |
| 批大小 | 32-64 |
| 数据集 | **公开数据集为主（裁定 V8 = A）** —— 见下方「数据集口径」 |

---

### 📌 数据集口径（2026-10-02 裁定 V8 = A：以公开数据集为主）

> 📌 **原写 `Human3.9M / CMU Mocap / 现有康复数据集` 是拍脑袋的**，未核实其腕/手可用性。
> 下表**逐条来自官方来源直读**，落 `../00_研究背景/DoF覆盖核查_公开数据集.md` §2.3。
> **凡未核实者不得写进论文。**

| 数据集 | 权威标识 | 腕/手可用性（**官方原文**） | 对本项目的含义 |
|---|---|---|---|
| **Human3.6M** | `10.1109/cvpr.2018.00551` | 32 关节，手/腕各**只保留 1 个关节** → *"leaving only one joint for each hand and each foot"* | 腕为**末端关节，无前臂旋转** |
| **AMASS** | `10.1109/iccv.2019.00554` | 主参数化是 **SMPL**（wrist 末端）；官方所称 *"realistic hand motion"* 是**表面/软组织层面**，**不是额外腕部自由度** | 🚨 **不可据官方措辞声称有手部 DoF** |
| **CMU MoCap** | 官方站定性说明 | 官方原文：*"the 'toe' and **'hand' joints tend to be noisy**"*；且 *"the 'finger' and 'thumb' joints are **added for editing convenience — we do not actually capture** these joints' motions"* | **手指/拇指根本没采集**；腕为噪声单点 |
| **SMPL / SMPL-X** | `10.1145/2816795.2818013` / `10.1109/cvpr.2019.01123` | `SMPL_JOINT_NAMES` 共 **24** 关节，`left/right_wrist` = **20 / 21**，**其后无任何关节**；SMPL-X（144）加手指 | 骨架层面确认**腕是末端** |

🚨 **由此产生的三条硬约束（已写进 §1 链拓扑声明）**：
1. **公开数据全部不提供前臂旋前旋后** → 腕第三 DoF **显式留空并记录**，不得静默降为 2 DoF。
2. **不同数据集的腕角不可直接比较**（坐标系与单位归一方案各异）→ 必须经 **ISB Part II**
   （`10.1016/j.jbiomech.2004.05.042`，shoulder/elbow/wrist/hand）归一后才能跨库评测。
   📌 注意 **ISB Part I 只管下肢（ankle/hip/spine）**，上肢必须用 Part II —— 两者不可混引。
3. **手指级任务在公开数据上做不了** → 若论文涉及手部细粒度，**须换用含手部采集的康复数据集并说明其不可复现性**。

> 📌 **未决提醒**：逐子集 marker 规格仍缺（台账 **V4**，AMASS `extra.php`/`download.php` 需注册，**按纪律未注册**）。
> **在 V4 清零前，不得对「AMASS 哪些子集带手」下任何结论。**

---

## 6. 评估指标（建议全覆盖）

| 类别 | 指标 |
|---|---|
| **精度** | MAE°, RMSE°, R², ICC, Pearson r |
| **物理合理性** | ROM violation rate, FK consistency, joint coupling error |
| **时序质量** | angular velocity error, angular acceleration error, temporal jerk |
| **鲁棒性** | cross-subject / cross-view / cross-speed / occlusion |
| **泛化** | **上肢**代偿动作（compensatory upper-limb movement）—— 🚨 原写 *pathological gait* 是**下肢口径**，已改 |
| 🆕 **词表可读性（一级指标）** | **实际使用的不同符号数 / 词表容量** · 每符号熵 · **跨受试者符号一致性** · 符号→角度的边界是否落在有意义的角度事件 |

---

## 7. 与现有方法的对比（一句话）

| 方法 | 一句话 |
|---|---|
| Wei 2026 | sEMG 版的 Seq2Seq 翻译原型 |
| Chao 2026 | RGB 版的 CNN+Transformer 角度回归 |
| **ViKiT** | **RGB 版的结构化翻译 + anatomy-aware region embeddings + biomechanical grammar + 可选 forecasting** |
| Pohl 2026 | 临床侧指标共识（**Table S2 已逐格转录**）—— 用于论证「词表不能只编码角度」 |

---

## 8. 论文写作建议

### 8.1 标题候选

> 🚨 **2026-10-02 对齐**：`../00_研究背景/论文定位陈述_收窄版.md` §4 已裁定**采用「保守」档**，
> 因为 **"Translating" 已被 ACM MM 2024 *Translating Motion to Notation* 占用（F11）**，
> 且 **"Kinematics as a Language" 是近百年旧说（F12：1994 年临床期刊即写 "written language of movement"）**。
> **本节原列的标题把两个高风险词放在同一句里，与该裁定冲突。**
> 
> ✅ **推荐（保守档，措辞与定位陈述一致）**：
> *"A Biomechanical Grammar over a Discrete Kinematic Vocabulary for Upper-Limb Joint Angle Estimation from Monocular RGB-D"*
> 
> ⚠️ 若仍要用 "ViKiT" 作方法名，**标题里不要出现 Translating / Language**；
> "translation" 只保留在**方法内部**（我们确实做 Seq2Seq 式的结构化映射），不放在标题的卖点位。

### 8.2 Abstract 第一句

> "We formulate visual-to-kinematic mapping as a structured sequence translation between two vocabularies: an anatomy-partitioned visual vocabulary over region embeddings, and a discrete kinematic vocabulary over biomechanically structured joint-angle symbols, on which a biomechanical grammar is enforced."
>
> 📌 **改了什么**：`two languages / a visual language / a kinematic language` → `two vocabularies`。
> 🚨 **理由**：**「运动是一种语言」不是我们的新发现**（F12：1994 年临床期刊已如此表述，
> Barbacci 2002 称 Labanotation 为 *"a universal movement notation language"*）。
> 把它当卖点等于**主动送出一个可被一句话反驳的 framing**。
> ✅ **可以说的**是「**我们把映射实现为两个词表之间的结构化映射，并给输出侧词表加上生物力学结构**」——
> 这个区别落在**符号是否有解剖含义、是否可被人读**，而不在比喻本身。

### 8.3 Introduction 三段式

1. 第 1 段：动机（sEMG 不能用了，RGB 是非侵入的，2026 年已有人做 RGB→angle，但还停留在 regression）
2. 第 2 段：我们的视角（升级为翻译范式，需要 token 系统 + grammar）
3. 第 3 段：贡献清单（4 项）

---

## 9. 详细方法设计 ✅ 已就位

Phase 2 的 5 篇详细方法设计文档**已全部写入** [`../03_方法设计/`](../03_方法设计/)，索引见其 `README.md`：

| 文档 | 内容 |
|---|---|
| [`Anatomy_aware_Region_Encoder.md`](../03_方法设计/Anatomy_aware_Region_Encoder.md) | **上肢 8 region** 怎么切（S1/S2/S3）、region 维度、ViT backbone 选型、分配温度 τ |
| [`Kinematic_Tokens_与_Grammar.md`](../03_方法设计/Kinematic_Tokens_与_Grammar.md) | vocabulary 设计（K=256）、**grammar loss 各项**（含新增 `L_dyn`）的具体超参与权重、两阶段训练策略 |
| [`Spatio_Temporal_Encoder.md`](../03_方法设计/Spatio_Temporal_Encoder.md) | encoder 结构（分解式时空注意力）、双轴 position encoding、sequence length 选型 |
| [`Cross_Modal_Decoder.md`](../03_方法设计/Cross_Modal_Decoder.md) | decoder 结构、cross-attention 配置、**解剖路由**、autoregressive vs parallel 预登记决策规则 |
| [`Forecasting_Task.md`](../03_方法设计/Forecasting_Task.md) | forecasting horizon (100/200/300 ms)、RGB vs sEMG 信息带宽约束、评估协议 |

**三条全局最脆弱假设**（必须在 Phase 3 早期检验，详见 `../03_方法设计/README.md` §4）：
H1 解剖先验有效（S1/S2 > S3）· H2 cross-attention 非冗余（视觉消融必须劣化）· H3 长序列带来增益（T=81 > T=27）

> ⚠️ **文件名更正**：本节此前的清单列作 `Kinematic_Vocabulary_与_Grammar.md` 与 `Spike_Temporal_Encoder.md`，
> 与实际落盘文件名不符，现已更正为 **`Kinematic_Tokens_与_Grammar.md`** 与 **`Spatio_Temporal_Encoder.md`**。

---

## 附录 A（原架构，已作废 · 2026-10-02 留痕）

> 🚨 **以下为 2026-10-02 修订前的架构，不得再引用**；保留仅为对照。
> **三处与现行裁定冲突**：
> 1. **下肢口径** —— `hip / thigh / knee / shank / ankle`；本项目目标域是**上肢**（G3）。
> 2. **离散为唯一输出路径** —— 架构图 ④ 由 VQ 直接解码为角度，**没有连续回归头**（G1）。
>    → 与 Pohl 2026 表 S2 冲突：**临床优先级前 10 名里只有 4 项是关节角**。
> 3. **未声明串联链拓扑** —— 把前臂旋前/旋后当成可直接加进词表的 DoF，
>    **未说明它挂在哪个关节**（G7）→ 已由 V6c 裁定为「记在腕（ISB 口径）」。

| 原模块（原名） | 现行对应 |
|---|---|
| ① Anatomy-aware Visual Tokenizer | **① Anatomy-aware Region Encoder**（上肢 region） |
| ② Spatio-Temporal Transformer Encoder | 不变 |
| ③ Cross-Modal Kinematic Decoder | 不变，query 改为**上肢** DoF |
| ④ Kinematic Token Vocabulary + VQ Decode（唯一输出） | **④a 词表（中间表示） + ④b 连续回归头（并存）** |
| ⑤ Biomechanical Grammar Loss | 不变 |
| 数据集：Human3.9M / CMU Mocap | 改为**已核实的公开数据集口径**（见 §5） |
