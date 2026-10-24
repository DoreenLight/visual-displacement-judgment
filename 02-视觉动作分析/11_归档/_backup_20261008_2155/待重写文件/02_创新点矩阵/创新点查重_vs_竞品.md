# 创新点查重矩阵：ViKiT vs 竞品

> 🚨 **2026-10-02 重大修订（G4）**：本文件原标题为「vs **12 篇**竞品」，
> 而那 12 篇是**在语料三建成之前**挑的，**其中 RGB→角度类的直接竞品只有 1 篇且是【下肢】**（Chao 2026）。
> → **这是一个系统性漏检：我们把「下肢竞品」当成了主要反驳对象，而本项目做的是上肢。**
> **语料三（106 篇，A 77 / B 29）建成后已补上 18 篇 A/B 级上肢直接竞品，见 §0.5。**

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

## 0.4 ⭐ 四篇已完成**全文级**撞红判定（2026-10-02，优先级高于下方摘要级判定）

> 📌 **为什么单列**：下方 §0.5 的 18 篇是**摘要级**判定；
> 而这 **4 篇的 PDF 我已全部取到并逐词统计过全文**，**结论强度完全不同**，必须排在前面。

| arXiv | 标题（简称） | 页数/字符 | `token`/`vocabulary`/`codebook`/`symbol`/`discrete`/`quantiz`/`embedding`/`Transformer`/`grammar` | 方法本质 | 判定 |
|---|---|---|---|---|---|
| `2605.09258` | **Monocular Biomechanical Tracking of Fingers with IK to Foundation Models**（Cotton 等，AbilityLab + Northwestern，2026-05-10）| 12 页 / 37,857 | **全部零命中** | **单目** + SAM 3D Body 基础模型 + MuJoCo-MJX 里的 **IK 优化** | 🚨 **设定最接近**：单目。但**零离散符号** → 不构成机制撞车 |
| `2607.02796` | Biomechanics-aware Multi-view MMC of Dexterous Hand（2026-07-02）| 9 页 / 44,084 | **全部零命中** | **8 相机多视角** + 梯度优化 + 生物力学模型在环，**方法非学习** | 🟡 命名撞车（`biomechanics-aware`），机制不撞 |
| `2602.13176` | Monocular MMC → Upper Extremity Reachable Workspace（2026-02-13）| 8 页 / 51,209 | **全部零命中** | 单目 AI-MMC，产出 **UERW 整体指标**（`joint angle` 全文仅 2 次）| 🟢 任务不同 |
| `2607.23608` | MMC in Routine Clinical Upper Limb Assessments（ARAT，2026-07-26）| 23 页 / 81,906 | 零命中（`symbol`×2 = 图例 θ/ω 记号；`discrete`×1 = ARAT 的 0–3 序数评分）| 临床流程嵌入，产出**连续**运动学指标 | 🟢 输出表征不同 |

### ✅ 全文级结论（这是目前证据最强的一条）

> **四篇全文逐词统计，输出侧的离散符号表命中数为 0。**
> → **F19（pose grammar 线上无离散运动学词表）在【全文层面】成立**，
> 且这四篇恰好覆盖了**最接近我们的四种设定**：单目 / biomechanics-aware / 上肢临床 / 康复评估。

### ✅✅ 该结论的加固与收窄（2026-10-03 第十九轮）

**为什么必须重做**：F19 是主张 C 死后**唯一的差异化支点**，
而它的证据基础（五篇全文的「离散/符号类词全零」）
**与第十八轮把 F29 漏掉的是同一套词** —— K26 已证明**零命中的是用词，不是所指**。
**不反查等于把刚犯的错留在最要命的位置上。**

| 项 | 反查前 | 反查后 |
|---|---|---|
| 探针词表 | 1 簇 16 词（全在本领域内） | **7 簇 68 词** |
| 领域外簇数 | **0** | **5 簇 53 词** |
| 领域外簇举例 | — | 时间序列挖掘 **SAX / PAA** · 图形学 **motion primitive** · 压缩感知 **dictionary** · 机器人 **skill library** · 医学影像 **atlas** |
| 审计对象 | 4 篇 | **5 篇**（+ Nakatsuka 2021） |
| SAX / motion primitive / 技能库 | 未测 | **五篇全部零命中** ✅ |
| 其余命中 | 未测 | **逐条回上下文确认无害**，多数是**反向证据** |
| 附加跨域检索 | 无 | 3 轮，**真撞红 0 条**（污染记 K28） |

**两处反向证据（比零命中更有说服力）**

| 论文 | 原文 | 含义 |
|---|---|---|
| `2605.09258` | *"there is **not** a dictionary directly mapping keypoints and vertices onto the marker locations"* | **该文明文说没有这样的映射** |
| `2607.23608` | *"continuous quantity on [0, 3] rather than the **discrete** ordinal values 0–3"* | **它从离散走向连续，与词表方向相反** |

**🚨 射程收窄（比加固更重要）**

- ❌ **不得写**「学出来的离散运动学词表是空白」/「尚无离散运动学表征」
- ✅ **只能写**「在 **pose grammar 线上**没有离散运动学词表」
- 📌 **理由**：「离散运动学词表」在**手部**（DigitCode / F5、ACM MM 2024 / F11）
  与**动作检索**（`2603.09930`）上**已经存在**

**⚠️ 残余风险（已具名）**：① 五篇是**抽样不是普查**（K24）；
② 未预料领域（运动剪辑 / 编舞 / 体育分析 / 手语）的同义词仍可能落在探针表外 → **脆弱假设 H5**。

### 🚨 F30：「小而固定、可人工转写」这条原则已被写出，且**不属于我们**

**DigitCode**（F5 同组，Zhang Xiao-Ping 亦为 F11 作者）全文 27 页 / 133,760 字符，
原文（`pdftotext` 机器抽取）：

> *"HL-26 … was not designed to minimize angular error: **like the Labanotation tradition it comes
> from, its alphabet was chosen to be small, fixed, and humantranscribable**."*
> *"It is **the axis the notation line itself adopted — the grid's value rests on bidirectional
> conversion** … and a code that cannot reconstruct cannot be converted back."*

| 约束 | 内容 |
|---|---|
| 🚫 **不得** | 把「小而固定、可人工转写」写成**我们的洞察** —— 它已被**归给 Labanotation 记谱传统** |
| ✅ **可以** | 引用它作为**既有的设计原则**（并说明我们沿用它） |
| 🚫 **不得** | 写「首个可转写的运动学字母表」（F11 + F5 + F30 **三重占位**） |

**📌 但这条同时让 F19 更锋利，不是更弱**：

| 已有工作 | 其离散化的**目标轴** | 有无离散词表 | 有无 biomechanical grammar |
|---|---|---|---|
| **DigitCode / F5、ACM MM 2024 / F11** | **rate–distortion（压缩）** ／ 记谱双向互转 | ✅ 有 | ❌ 无 |
| **pose grammar 线**（F6 / F10 / TPAMI 2022 / TBME 2023） | 姿态参数精度（连续回归） | ❌ **无** | ✅ 有 |
| **记谱传统本身**（Labanotation，见 F30） | **小而固定、可人工转写**（已被明确写出，**非任何一篇的原创**） | ✅ 有 | ❌ 无 |
| 🆕 **我们要占的交集** | **① 生物力学一致性** + **② 输出粒度匹配测量精度** | ✅ 有 | ✅ 有 |

**✅ 2026-10-03 F11 全文到手，上表第一行由「题名级推断」升级为「全文级实证」**
（9 页 / 48,498 字符；13 组关键词 + 7 组跨领域同义簇 + 逐处回上下文）：

| F11 全文实测 | 计数 | 对上表的影响 |
|---|---|---|
| `grammar` / `vocabulary` / `codebook` | **全 0 次** | 「无 biomechanical grammar」**坐实** |
| `joint angle` / `kinematic` | **全 0 次** | 🚨 **它连关节角都不说** —— 它的离散化对象是**记谱符号**，不是运动学 DoF |
| `shoulder` / `elbow` | **全 0 次**；`wrist` 仅 2 次且均为**手部骨架坐标原点** | 设定不同（上肢腕肘肩 vs 手） |
| `rehabilitat*`/`clinical`/`patient*`/`stroke`/`therap*` | **全 0 次**；设备 12 次**逐条确认全是 robotic hands** | 🚨 **与临床康复零交集**（此前是领域推断，现为全文零命中） |
| `pronation`/`supination`/`radioulnar` | **全 0 次** | 前臂旋转轴无人占 |
| `ROM` 3 次 / `dynamics` 2 次 | 全是**拇指手指在自定 XYZ 系的几何范围** / **Labanotation 四要素之一、且指「情绪成分」** | 反向证据：**均非生物力学** |
| `translat*` 30 · `intuitive` 17 · Transformer 6（MHLFormer） · 记谱 181 | — | 🚨 **威胁四条全部坐实**，「Translating」与「坐标不直观」**不可用** |

📌 **这张表现在是本项目最硬的一条差异化论证**：三条已有路线**各自都只占一半**
（压缩的、grammar 的、记谱的），**没有一条同时具备「生物力学一致性」与「离散词表」**。

**🆕 第二条轴的证据（E12 / Cleffken 2007，2026-10-03）**

*Test-retest reproducibility of elbow goniometric measurements…*，J Shoulder Elbow Surg 2007;16(6):788–794，
`10.1016/j.jse.2007.02.134`（Crossref + Europe PMC 双源一致）。
**肘关节测试者重测 SDD：主动 ROM 0±8.5°、被动 ROM 0±10.8°**（Bland–Altman 修正法，42 人 × 144 次）。

| 命题 | 说明 |
|---|---|
| **论证** | 在 ±5–10° 的测量噪声下，**连续角度输出在该带内不可解释**；**ROM 锚定的符号**能给出**噪声下仍成立**的陈述 → **输出粒度必须与测量精度匹配** |
| **与压缩轴正交** | DigitCode 要的是 rate–distortion；我们要的是**「这个数位的精度是不是真的」** |
| **对判据 ④ 的作用** | 给「符号边界须有生物力学意义」补上**第二条独立依据**（第一条是 F24 的逐关节独立可指定） |
| 🚨 **三条限制** | ① **SDD ≠ accuracy**（precision 是 accuracy 的一个分量）；② **量角器不是实验室级参照** → 8.5° 是**临床参照标准的下限，不是视觉方法的下限**；③ 🚨 **仅肘关节**，肩与腕未找到同规格 SDD |
| **实验设计后果** | 🚫 **不得以人工量角器为参照声称 5° 级精度** —— 参照自身就波动 8.5°，差异无法归因于算法；**应优先用光学动捕做主验证** |

### 🚨 但有两条边界已经不能当新意写

1. **「单目 + 生物力学约束」这个组合已被占**（`2605.09258`，`monocular` 全文 28 次）。
2. **「单目 + 上肢」本身不新**（`2602.13176`，`monocular` 34 次）→ **不得写进贡献点**。
3. **「biomechanics-aware」一词被占**（`2607.02796`）→ **必须主动区分**：
   他们的 = **模型在优化循环里**；我们的 = **力学约束在神经网络的训练目标里 + 离散符号表**。

### 📌 `2605.09258` 的空白陈述（🚨 限定极强，慎引）

> *"monocular video approaches for obtaining **hand** biomechanics remain **under-developed**"*

🚨 **它说的是 hand / finger，不是上肢 ISB 角度** —— **不可外推**（K10：规则类结论不得外推）。
📌 **可安全引用的用法**：作为「**单目手部生物力学仍在起步**」的证据，
**不能**用作「上肢关节角在视觉侧无人研究」的依据。

### 📌 `2605.09258` 给我们的一条有利证据

`2607.02796` 报告：**其对照的两阶段法（2D 关键点→3D→约束 IK）有 15% 的视频不收敛**；
`2605.09258` 同样依赖 **优化收敛**（Levenberg–Marquardt 多阶段）。
→ **纯重建路线存在优化收敛问题** —— 而我们走**直接翻译/回归**路线，**恰好绕开这一整类失败模式**。
📌 **这可以写进方法动机**：「我们选择直接序列翻译而非基于优化的重建，理由是后者存在已知的收敛失败（15%）。」

### 0.4.1 🚨🚨 E/F 级：前臂旋前旋后的精度**已被单目流水线报告过**（全文级，2026-10-03）

| 项 | 内容 |
|---|---|
| **文献** | Nakatsuka C, Xu J, Tasaka K（KDDI Research），*Learning Joint Twist Rotation for 3D Human Pose Estimation from a Single Image*，**VISAPP 2021**（SCITEPRESS），pp. 379–386 |
| **标识** | `10.5220/0010222703790386`（**Crossref 反查**：题名/作者/年份/会议/页码与 PDF 全部一致） |
| **判定级别** | **全文级** —— 8 页 / 29,791 字符，PDF 在库 `B4_离散运动学token_与表征量化/`，已逐词统计 |
| **撞的是哪一条** | **主张 C 的措辞**（不是差异化轴 F19） |
| **为什么它算前臂旋前旋后** | 原文 *"Let R be the rotation of joint J (j = 0, …, 3 for right/left shoulder and right/left elbow)"* + twist 轴 = *"direction vector from joint J to its child joint J′ on T pose"* → **肘关节的 J′ 是腕，轴即前臂长轴** |
| **它报了数** | **Table 1 按父关节名**：LSho 0.06 / RSho 0.17 / **LElb 0.18** / **RElb 0.15** rad（623 pose 评测集，均值 0.14） |
| **是否撞 F19** | ❌ **不撞**。`token`/`vocabulary`/`codebook`/`symbol`/`discrete`/`discretiz`/`quantiz`/`embedding`/`Transformer`/`grammar`/`language`/`translation` **全部零命中** → **F19 在第五篇全文级竞品上再次成立** |
| **三条限制** | ① **真值是伪标签**（Pavlakos 2019 SMPLify-x 预测生成）；② 评测仅 **623 pose**（大 twist 子集 96）；③ **输入喂了真值 3D 关节点** → 是**精化模块**，非完整流水线 |
| **应用语境** | CG character control / sports / surveillance（摘要与引言共 3 处）；`clinical`/`rehab`/`patient`/`injury`/`therap` **全文 0 次** |
| **附带正面作用** | ① 作者自陈 *"There is no enough dataset which includes 3D joint rotations on images"* → **数据缺口的外部佐证**；② 增益全部来自**关节邻域外观** → 与 PAHMT「twist 不能由关节点确定」**自洽（分工，非矛盾）** |
| **台账** | **F29**（撞红）/ **K26**（方法论：词汇零命中 ≠ 语义零命中） |

## 0.5 上肢竞品：18 篇已完成**模态级 + 摘要级**核实（2026-10-02）

> 📌 **为什么必须核实**：原 12 篇早于语料三，RGB→角度类直接竞品只有 1 篇且是**下肢**（Chao 2026）。
> 下列 18 篇出自 [`../01_参考论文/生物力学线/index.json`](../01_参考论文/生物力学线/index.json)（A/B 级可引用范围），
> **本轮已逐条反查模态与摘要**（Crossref / Europe PMC / OpenAlex / arXiv API），**核实结果见下表**。
>
> 🚨 **核实过程中发现我自己的表错了两处，已更正**（记台账 **K20**）：
> `10.1142/s0219519417500312` 原被列为「最直接的上肢角度估计前作」——
> **实为 IMU + 光学传感器，不是视觉**；`10.1186/s12984-016-0130-0` 原被列入视觉竞品 ——
> **实为上肢假肢腕单元研究，与视觉动捕无关**。两处均已移出「视觉竞品」。

| # | 工作（简称） | 权威标识 | **模态（已核实）** | **含离散符号？** | 与 ViKiT 的关系 |
|---|---|---|---|---|---|
| 1 | **Biomechanics-aware Multi-view MMC of Dexterous Hand**（2026）| `2607.02796` | **视觉 · 8 相机多视角** | ❌ 无 | 🚨 **「biomechanics-aware」一词已被占**（见下 F26）|
| 2 | **Monocular MMC → Upper Extremity Reachable Workspace**（2026）| `2602.13176` | **视觉 · 单目** | ❌ 无 | 🔴 **「单目 + 上肢」不新**；但任务是 **UERW 一个整体指标**，非逐关节角符号 |
| 3 | **MMC in Routine Clinical Upper Limb Assessments（ARAT）**（2026）| `2607.23608` | **视觉 · AI MMC** | ❌ 无 | 🟡 临床流程嵌入，输出**连续**运动学指标；无词表 |
| 4 | **Vision-based Detection of Compensatory Postures（卒中）**（2020）| `10.1109/icaretech49914.2020.00019` | **视觉** | ❌ 无 | 🟡 **代偿检测已有前作**（摘要含 "vision-based detection… compensatory postures"）|
| 5 | Theia 3D markerless vs marker-based（肩，2025）| `10.1016/j.jbiomech.2025.112905` | **视觉（题名级）** | ❌ 摘要无 | 🟡 Theia 3D 是绕不开的对照系统 |
| 6 | MMC for upper extremity kinematics（2022）| `10.3389/fspor.2022.939980` | **视觉** | ❌ 无 | 🟢 精度对照 |
| 7 | Two markerless MMC systems, upper extremity（2020）| `10.1002/hfm.20840` | **判不出**（工业作业姿态）| ❌ 无 | 🟢 精度对照；⚠️ 模态未确认 |
| 8 | Kinect skeleton for shoulder joint angles（2015）| `10.1016/j.physio.2015.02.002` | **视觉 · RGB-D（Kinect）** | ❌ 无 | 🟢 早期上肢角度基线，**深度相机非 RGB** |
| 9 | Single-camera markerless joint angles（2014）| `10.1016/j.jbiomech.2013.11.031` | **视觉（题名级）** | ❌ 摘要无 | 🟢 单目基线 |
| 10 | Color-Depth / Polymer 光学传感器对比（2019）| `10.3390/electronics8020173` | **视觉 + 光学传感器** | ❌ 无 | 🟢 传感器对照 |
| 11 | Accuracy of MMC + CV for clinical & rehab biomechanics（2023）| `10.1111/sms.14319` | **视觉** | ❌ 无 | 🟢 综述级 |
| 12 | Markerless Camera-Based 3D MMC（2024）| `10.3390/s24113686` | **视觉** | ❌ 无 | 🟢 综述级 |
| 13 | SWOT: Portable & Low-Cost MMC（2022）| `10.3389/fspor.2021.809898` | **视觉** | ❌ 无 | 🟢 临床可用性约束 |
| 14 | Review: 3D pose estimation for MMC（2020）| `2010.06449` | 综述 | ❌ 无 | 🟢 方法综述 |
| 15 | ~~Shoulder & Elbow Angle Estimation~~（2017）| `10.1142/s0219519417500312` | 🚨 **惯性 IMU + 光学** | ❌ 无 | ❌ **已移出视觉竞品**（模态不符，我方原表标错）|
| 16 | ~~Flexible/static wrist units in prosthesis~~（2016）| `10.1186/s12984-016-0130-0` | 🚨 **假肢腕，非视觉动捕** | ❌ 无 | ❌ **已移出**（主题不符，我方原表标错）|

### ✅ 撞红判定结果（摘要级，逐条已核）

**🎯 最重要的一条：18 篇中「含离散符号」命中 = 0。**
逐篇检查 `token` / `vocabulary` / `symbol` / `codebook` / `discrete`，**无一命中**。
→ **F19 那条轴（pose grammar 线上无离散运动学词表）经 18 篇上肢竞品复核后仍然成立。**

**F26 ·「biomechanics-aware」一词被占，但机制完全不同（`2607.02796`，2026-07-02）**

| | 该文（`2607.02796`） | 我们 |
|---|---|---|
| 「biomechanics-aware」的含义 | **生物力学模型在优化循环里**（gradient-based optimization, model-in-the-loop）| **力学约束在神经网络的训练目标里** |
| 方法 | **非学习**：多视角三角化 + 约束 IK 优化 | **学习式**：ViT + Transformer 翻译 |
| 相机 | **8 相机多视角** | 单目 |
| 输出 | 连续关节角 | **离散符号 + 连续量并存** |
| 部位 | **手（精细）/近端上肢** | 上肢（肩/肘/腕） |

📌 **同形异义，但审稿人会直接指认这个词** → **必须在 Related Work 主动区分**，不能装作没看见。
🚨 **且该文的对照基线恰好是「两阶段法：2D 关键点 → 3D → 约束 IK」**，并指出两阶段法 **15% 的视频不收敛** ——
**这条对我们有用**：它说明**纯重建路线存在收敛问题**，而我们走**直接回归/翻译**路线，恰好绕开优化收敛。
（登记台账 **F26**）

### 🚨 四条必须写进论文的诚实结论

1. **「上肢 RGB → 关节角」不是空白。** 至少 `2602.13176`（单目 + 上肢）、
   `10.1016/j.physio.2015.02.002`（Kinect 肩角）、`10.1016/j.jbiomech.2013.11.031`（单目）已做。
   → **差异化不能建立在「上肢没人做」上。**
2. **「单目 + 上肢」这个设定本身不新**（`2602.13176` 明确用 monocular）。
   → **不能再把「单目」写进贡献点**。
3. **代偿检测已有前作**（`10.1109/icaretech49914.2020.00019`）。
   → 我们的 `r0 torso` 代偿轴**不是新发现**，其价值在于**作为词表的一部分被结构化**。
4. ✅ **但「离散运动学词表 + biomechanical grammar」在 18 篇中零命中。**
   → **这才是可写进论文的那一条差异**，且与 F19 独立互证。

### ⚠️ 本节仍未做到的事（诚实清单）

- 上述判定**是摘要级，不是全文级**。`2607.02796`、`2602.13176`、`2607.23608` 三篇**正文未取**。
- `10.1002/hfm.20840` 的**模态仍未确认**（Europe PMC 0 命中，OpenAlex 摘要未明）。
- `10.1016/j.jbiomech.2025.112905`、`10.1016/j.jbiomech.2013.11.031` **Crossref 无摘要**，模态仅题名级。
- 📌 **因此**：在补做全文级判定前，**不得写「与以上工作不同」**，只能写
  **「与以上工作在输出表征上不同（离散词表 vs 连续回归/IK 重建）」** —— 后者已由本表「含离散符号」列支撑。

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
| **ViKiT (我们)** | RGB video | **Anatomy-aware region embeddings** | ✅ **新** |

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
| **VQ-HPS (ECCV 2024)** 🚨 | **discrete VQ latent + 分类任务** | ✅ **范式级占位** |
| **HoT (CVPR 2024)** 🚨 | 连续；"tokenizer" 指**输入侧时间剪枝** | 同形异义，见下 |
| **PAHMT (CVPR 2022)** 🚨 | **连续关节旋转**（Transformer 直出） | ❌ 但**任务撞车** |
| PoseFormer | continuous 3D joints | ❌ |
| OpenCap | continuous angle (via IK) | ❌ |
| **ViKiT** | **discrete kinematic tokens + kinematic vocabulary** | ✅（但**新颖性已被大幅收窄**） |

> 🚨 **2026-10-01 第二代引文追溯后，本节原结论已作废。**
> 原结论写的是「TokenHMR 已为 SMPL pose 做了 tokenization，但没人把临床关节角做成 vocabulary」——
> 这句话**漏掉了 VQ-HPS**，而且它的范围划错了。

**改正后的结论（三层，逐层收窄）**：

1. ❌ **不能再主张「首次把姿态离散化并当分类任务」** ——
   VQ-HPS（ECCV 2024, `2312.08291` / `10.1007/978-3-031-72943-0_27`）已把该范式用于人体网格，
   且原文自称 *"novel paradigm"*。
2. ❌ **不能再用 "tokenizer" 这个词而不加区分** ——
   HoT（CVPR 2024, `2311.12028`）已占据该词，但**同形异义**：
   它剪的是**输入侧**冗余时间帧（省 FLOPs），我们量化的是**输出侧**符号（为语法约束）。
   ✅ **2026-10-02 已执行改名**：`Anatomy_aware_Tokenizer.md` → **`Anatomy_aware_Region_Encoder.md`**，模块名 **Anatomy-aware Visual Tokenizer** → **Anatomy-aware Region Encoder**（15 个文件的引用已同步）。
   📌 **为什么改成 "Region Encoder" 而不是 F8 当初建议的 "codebook / vocabulary"**：
   ① **F8 的建议其实混了两件事** —— 撞名的是**输入侧**的模块，而 codebook/vocabulary 说的是**输出侧**；
      输出侧本来就叫 **Kinematic Token Vocabulary**，从无冲突。
   ② 输入侧这个东西**本来就不是 tokenizer** —— 它把连续视觉切分成 region embedding，**没有离散符号表、没有编码-解码对**。叫 Tokenizer 属于**名不副实**，改名同时修掉了撞名和命名不实两个问题。
   ③ §5 风险表随之更新（见下）。
3. ✅ **仍成立的差异（这才是可写进论文的那一条）**：
   > **词表的「可解释与可约束性」，而不是「有没有离散化」。**
   > VQ-HPS / TokenHMR 的码本是**学出来的、无解剖语义、不可人工检查也不可编辑**；
   > 我们的词表**受 ROM 边界约束、受关节耦合约束、有语法、人工可读**。
   > 换句话说：**离散化是手段，符号系统是贡献。**

> ⚠️ 本节整体仍是**下肢口径**（hip/knee/ankle），与上肢决策冲突，属 **G3 / G4** 待修。
> 且**上表的竞品矩阵尚未纳入** PAHMT / VQ-HPS / HoT / TPAMI 2022 等本轮新查实的工作。

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
| **2. Anatomy-aware Region Encoding** | Wei 2026（但只在 sEMG 上）| "Wei 只在 sEMG 做了 anatomy-aware，我们扩展到视觉" |
| **3. Biomechanical Grammar** | Chao 2026（有 velocity 项）| "Chao 只有 velocity，我们有 6 项 + FK consistency" |
| **4. Causal Future Translation** | Wei 2026（但只在 sEMG 上）| "视觉版的 forecasting 没人做" |

**结论**：每个创新点都有先例，但没有完全重复。我们的组合（视觉 + anatomy-aware region embeddings + 6 项 grammar + forecasting）是**未见组合**。

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

> "Recent work in vision-based joint angle estimation has demonstrated the feasibility of end-to-end RGB-to-angle mapping [Chao 2026; Jamsrandorj 2022]. In particular, Chao et al. (2026) proposed a CNN + Transformer pipeline that achieves ICC > 0.92 across views, speeds, and abnormal gaits. However, their approach treats joint angles as a single regression target, without explicit tokenization of the kinematic output space or enforcement of biomechanical constraints beyond angular velocity smoothness. In contrast, our work formulates the problem as a structured sequence translation between two distinct languages: a visual language over anatomy-aware region embeddings and a kinematic language over discrete joint-angle tokens. This formulation allows us to introduce explicit biomechanical grammar (ROM, forward kinematics consistency, joint coupling) into the training objective."

> 🚨 **2026-10-01 判定：上面这段草稿不能直接用，缺 4 篇关键工作，评审点到即 fatal。**
> 必须改用 `00_研究背景/论文定位陈述_收窄版.md` §3 的重写版（已含下列全部引用）。
>
> **漏掉的四篇**：
> | 漏掉的工作 | 层级 | 后果 |
> |---|---|---|
> | **PAHMT, CVPR 2022**（`2203.16202`）| **方法级撞车** | 单目视频 → 上肢 arm+hand 关节旋转 → Transformer。不引 = 未做功课 |
> | **VQ-HPS, ECCV 2024**（`2312.08291`）| **范式级撞车** | 离散隐表征 + 分类任务，自称 novel paradigm |
> | **Hourglass Tokenizer, CVPR 2024**（`2311.12028`）| **命名撞车** | "tokenizer" 一词已被占（同形异义） |
> | **TPAMI 2022 / TBME 2023** | **谱系撞车** | pose grammar 线已延续，只引 AAAI 2018 会被指漏引 |
>
> ⚠️ 另：本段**只做 RGB**，未处理 RGB-D；且**全文仍是下肢口径**（Chao 2026 核心反驳对象），
> 与 G3/G4 待修项一致。**引用前须回原文核对页码**。

### 3.3 必须做的 ablation（防止被挑战）

| Ablation | 切掉什么 | 期望看到的退化 |
|---|---|---|
| DUP-A1 | anatomy-aware region encoding（换成普通 ViT patches）| MAE 退化；ROM violation 飙升 |
| DUP-A2 | grammar loss（只保留 L_angle）| 物理合理性指标退化 |
| DUP-A3 | tokenization（改成连续回归）| 关节角分布变宽；ROM violation 飙升 |
| DUP-A4 | forecasting（只做同步 estimation）| 与 Wei 2026 的对比无新意 |
| DUP-A5 | cross-attention（改成纯 autoregressive）| 长序列稳定性退化 |

每条 ablation 都要在论文的 Experiments 节里出现，并且有数据支撑。

---

## 4. 一句话定位（写在 Abstract / Introduction 第一段）

> We propose ViKiT (Visual-to-Kinematic Translator), a structured sequence translation framework that maps anatomy-aware region embeddings to discrete kinematic tokens via cross-modal cross-attention, trained with a biomechanical grammar that enforces range-of-motion, joint coupling, and forward-kinematic consistency.

---

## 5. 风险评估

| 风险 | 概率 | 应对 |
|---|---|---|
| 审稿人拿 Chao 2026 直接挡 | 高 | ablation DUP-A1+DUP-A2+DUP-A3+DUP-A5 全做 |
| 审稿人说"换汤不换药" | 中 | 强调 token 系统 + grammar loss + forecasting 的结构性差异 |
| 临床数据集缺乏 | 中 | Human3.9M / CMU Mocap / 现有康复数据集；公开 + 自采 |
| 物理指标难实现 | 中 | ROM violation rate 简单；FK consistency 需 SMPL/skeleton model |
| 离散符号误判 / 词表被证伪 | **中高** | 🚨 **原第 5 行写的是「用 OpenRouter `stealth/ox-alpha` 翻译时抽样校验」—— 那是**文本翻译模型的路由名，与生物力学无关，属生成残留，已删除。** 正确应对：见下 |
| 🆕 词表大面积闲置（F20 饱和风险）| **高** | **必须报告符号使用统计**（用到的不同符号数/容量、每符号熵、跨受试者一致性）；若闲置严重，**主动收窄词表容量**并把它写成发现，而非辩解（§2.4 判据 ②③）|
| 🆕 腕第三 DoF 交不出数据 | **高** | **公开数据无前臂旋前旋后**（已实测）→ 槽位**显式留空并记录**，写进 Limitations；**不得静默降为 2 DoF，也不得声称覆盖旋前旋后** |

---

**✓ 结论**：现有 12 篇工作中**没有任何一篇**做了「anatomy-aware 视觉 token + 6 项 biomechanical grammar + 视觉版 forecasting」的完整组合。**ViKiT 的结构性差异是有依据的。**