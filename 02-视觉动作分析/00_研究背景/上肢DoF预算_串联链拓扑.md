# 上肢运动学 DoF 预算：串联链拓扑与五套带出处的约定（2026-10-01）

> **本文解决什么**：词表（kinematic vocabulary）该放哪些 DoF，取决于**上肢运动学模型怎么切分关节与自由度**。
> 此前我只查到「SMPL 的 wrist 是末端关节」，不足以定稿 —— 因为**运动捕捉系统与数据集是两套不同约定**。
> 本文给出**五套可引用的、带原文出处**的上肢 DoF 预算，并给出**文献对拓扑争议的明确表态**。
>
> **本文全部片段由 `mine_fulltext.py` 从语料三已取得的全文中抽取**
> （Europe PMC JATS XML 的 `<body>` + 30 份官方 OA PDF 的 `pdftotext` 文本），
> **原文照录、未改写**。所有 DOI / arXiv ID 均经 Crossref 与 arXiv 机器反查。
> **引用进论文前仍须回原文核对。**
> **最后更新**：2026-10-01

---

## 0. 一句话结论

> **解剖上，肘-前臂复合体由四个关节构成；文献通行做法却把前臂当刚体、只在肘给 2 个 DoF，
> 而且明确承认「对前臂该用哪种建模方式尚无共识」。**
>
> 五套已发表约定把「前臂旋前旋后」挂在**不同关节**上：
> 三段前臂记在**腕**、刚性前臂记在**肘**、还有一套把它做成**独立关节**。
>
> 而**通用数据集参数化（SMPL / Human3.6M / CMU）里 wrist 是末端关节 —— 五套都没有**。
>
> → 词表能不能含旋前旋后，**取决于你用哪一套作监督来源**；
> → 就算能含，**必须先声明采用的链拓扑与所引文献的拓扑**，否则同一物理旋转会被计两遍；
> → 且**这是领域内公开的无共识问题**，论文里选任一套都必须论证，不能默认。

---

## 1. 先例 A：无标记动捕系统约定（Theia3D 上肢多体模型）

**来源**：语料三 B1，*Accuracy of a markerless motion capture system in estimating upper extremity
kinematics during boxing*（Europe PMC 全文 XML，`<body>` · Data processing and analysis · Multibody models）

> "Theia3D embedded multibody kinematic model consists of **two separate kinematic chains**: one for
> the lower extremity and one for the upper extremity, and a **separate head segment with six degrees
> of freedom (DoFs)**. In this study, we will only adhere to the upper extremity model in the
> following descriptions.
>
> The upper extremity chain comprises the **thorax as root segment with six DoFs** with respect to the
> ground, followed by the **clavicle, upper arm, forearm, and hand** segments bilaterally. The clavicle,
> at its proximal end, is connected to the thorax with a **two rotational DoFs** constraint, while
> distally connected to the upper arm with a **three rotational DoFs at the shoulder joint**. The elbow
> and the wrist joints are constrained to have **two DoFs**, restricting abduction/adduction (Abd/Add)
> at the elbow and **internal/external (Int/Ext) rotation at the wrist**."

**同一文献给出的关节中心定义**（原文）：

> "The shoulder, cervical, lumbar, and thoracic joint centers were defined based on the regression
> equations adopted from the study of Dumas and Wojtusch (2018). The midpoint between the **medial and
> lateral humeral epicondyles** was defined as the **elbow joint center** and the midpoint between the
> **ulnar and radius styloid processes** as the **wrist joint center**."

**整理成表**：

| 节段 | 与上一节段的连接 | DoF |
|---|---|---|
| thorax（root） | 相对地面 | **6** |
| clavicle | 近端接 thorax | **2**（旋转）|
| upper arm | 远端接 clavicle（= 肩关节）| **3**（旋转）|
| forearm | 肘关节 | **2**（屈伸 + 外展/内收 Abd/Add）|
| hand | 腕关节 | **2**（屈伸 + **内/外旋 Int/Ext**）|

> ⚠️ 注意该文献中的 Abd/Add 归属：原文是 *"restricting abduction/adduction (Abd/Add) at the elbow
> and internal/external (Int/Ext) rotation at the wrist"* —— 即 **Abd/Add 归肘、Int/Ext 归腕**。
> 这与「肘 2 DoF = 屈伸+旋转」的常见写法不同，**引用时必须照原文归属，不要套用通行印象**。

> **对照标记法**：该文说明标记法模型 *"was created in Visual3D … to have **identical body segments and
> joint constraints** as that of the Theia3D model"*，并以 Dumas & Wojtusch (2018) 的回归方程定关节中心。
> → **无标记与标记法在这套约定下是对齐的**，这正是该文能做精度对比的前提。

---

## 2. 先例 B：肌肉骨骼模型约定（上肢综合模型，17 DoF）

**来源**：语料三 B2，*Development of a comprehensive musculoskeletal model of the shoulder and arm for
clinical applications*（Europe PMC 全文 XML，`<body>` · Kinematics）

> "The total number of DOF of the model is **17**, namely **six for the thorax** (which is considered as
> the moving base), **three for the shoulder girdle**, **three for the glenohumeral-joint**, **two for
> the elbow**, and **three for the wrist**."

| 部位 | DoF |
|---|---|
| thorax（活动基座） | 6 |
| shoulder girdle | 3 |
| glenohumeral joint | 3 |
| elbow | 2 |
| **wrist** | **3** |
| **合计** | **17** |

---

## 2.5 先例 C：把前臂旋转做成**独立关节**（Holzbaur 7-DoF 上肢模型）

**这是 2026-10-01 新挖到的第三套拓扑，也是与本项目最接近的一篇。**

**来源**：语料三 B1，*Biomechanics-aware Multi-view Markerless Motion Capture of Dexterous Hand
Movements*（arXiv `2607.02796` · 官方 OA PDF · §III-C "Arm and Hand Biomechanical Model and Simulation"）
—— **标识已机器反查**：Firouzabadi, Peiffer, Shah, Sobinov, Miller, **Cotton** · 2026-07-02。

> "Specifically, we implemented a **21 degree of freedom (DoF) model of the hand** [13] combined with a
> **7-DoF model of the upper-extremity** [14], both originally developed in OpenSim [32].
> The combined model includes kinematic definitions of all finger and thumb movements (including coupled
> flexion for the carpometacarpal joints of the ring and little finger), **wrist flexion/extension and
> deviation, forearm rotation (pronation/supination), elbow flexion, and shoulder DoFs** that include the
> scapulohumeral rhythm. We include an additional 6 DoFs at the ground segment (the thorax) to represent
> its 3D position and orientation in the global frame."

**7 个 DoF 的构成（由上句逐项拆出，合计正好 7）**：

| 部位 | DoF | 明细 |
|---|---|---|
| shoulder | 3 | 含 scapulohumeral rhythm（肩肱节律）|
| elbow | **1** | **仅屈伸** |
| **forearm rotation** | **1** | **旋前/旋后，独立成关节** |
| wrist | 2 | 屈伸 + 偏斜（deviation）|
| **合计** | **7** | |

> 📎 **7-DoF 上肢模型出处（已反查）**：Holzbaur KR, Murray WM, Delp SL, *A model of the upper extremity
> for simulating musculoskeletal surgery and analyzing neuromuscular control*, Ann Biomed Eng 2005,
> **33(6):829–840, doi `10.1007/s10439-005-3320-7`**（Crossref 反查通过）。
> **该文此前不在语料三内 —— 是本轮新命中，建议补入 B2 线程。**

> ⚠️ **注意先例 C 与 A/B 的关键差别**：
> 先例 A（Theia3D）把旋转挂在**腕**（腕 2 DoF = 屈伸 + 内/外旋，肘 2 DoF = 屈伸 + 外展内收）；
> 先例 C 把旋转挂在**肘与腕之间的独立关节**（肘只剩 1 个屈伸，腕是屈伸 + 偏斜）。
> **三者对同一个物理量给出三种归属。**

---

## 2.8 决定性原文：领域对拓扑争议的明确表态

**来源**：语料三 B1，Wang W, Wang D, Wesseling M, Xue B, Li F, *Comparison of modelling and tracking
methods for analysing elbow and forearm kinematics*, **Proc IMechE Part H** 2019, **233:1113–1121,
doi `10.1177/0954411919872400`**（Crossref 反查通过 · 官方 OA PDF · 全文可得）

### 2.8.1 解剖上应该是**四个**关节

> "**The elbow-forearm complex is composed of four different joints: the humeroulnar and humeroradial
> joints at the elbow and the proximal and distal radioulnar joints within the forearm.**"

| # | 关节 | 位置 | 承担的运动 |
|---|---|---|---|
| 1 | **humeroulnar**（肱尺）| 肘 | 屈伸 |
| 2 | **humeroradial**（肱桡）| 肘 | 屈伸（+ 桡骨头随动）|
| 3 | **proximal radioulnar**（近端桡尺）| 前臂近端 | **旋前/旋后** |
| 4 | **distal radioulnar**（远端桡尺）| 前臂远端 | P/S 过程中的**桡尺相对位移** |

### 2.8.2 通行做法是**把它简化成 2 个 DoF**，而文献明说这是简化

> "However, in most studies that used a marker-based elbow model, **the forearm was treated as a rigid
> segment**, and the elbow was commonly defined as a joint of **two independent degrees of freedom: elbow
> flexion-extension (F/E) and pronation/supination (P/S)**, **ignoring the fact that relative motion exists
> between ulna and radius**. … This kind of simplification may fail to estimate motion characteristics
> aside from the primary motion, while **other motions like the radioulnar motion during P/S are also
> crucial** for the biomechanical analysis of the forearm."

### 2.8.3 **而且领域明确承认「尚无共识」**

> "**Still, no agreement has been reached on the most appropriate modelling approach of the forearm for
> estimating elbow and forearm kinematics.**"

> 🎯 **这三条合起来就是 V6c 的答案**：
> **V6c 不是一个可以靠"选个更标准的"来解决的技术细节 —— 它是文献明确记载的开放争议。**
> 论文里无论选哪套，都必须 (a) 声明所选拓扑；(b) 引用 Wang 2019 说明该选择有争议；
> (c) 论证为何本场景下选它（并说明代价：选刚性前臂会丢掉桡尺相对运动）。
> **不能说"按 ISB 标准"来回避 —— ISB 提供的是坐标系定义，前臂如何切分恰恰是 ISB 之外的无共识部分。**

### 2.8.4 顺带给 V7 一个量化警告：追踪方式改变 P/S 估计达 **14.9%–43.2%**

> "Multiple comparisons demonstrated statistical differences between tracking methods: compared to using
> only anatomical markers, **tracking using clusters reduced the estimated range of pronation/supination
> by 14.9%–43.2%, while it estimated increased flexion/extension by 5.3%–9.1%.**"

> ⚠️ **适用边界**：该文是**光学标记法（OMC）**、12 名健康男性、**仪器化测量**。
> **不能外推为"D435i 单目一定不行"**。但它证明了一件事：
> **前臂 P/S 的估计值对"你怎么测"极度敏感 —— 在有实体标记的条件下，
> 仅换一种追踪方式就能让 P/S 范围变化近一半。**
> 这是 V7（单目能否可靠估前臂旋转）**必须在论文里正面回答**的量化理由。

---

## 3. 五套约定的差异汇总（这才是真正的设计问题）

### 3.1 第二轮挖掘发现的**更深分歧**：旋前旋后到底算在哪个关节

> 2026-10-01 补：`mine_fulltext.py --preset wrist` 进一步取到先例 B 的**肘关节功能轴表**（Table 3），
> 原文照录：

> "Table 3 The position ([P_x P_y P_z]) and orientation ([d_x d_y d_z]) of the functional axes of rotation of the elbow
> | Rotation axis | Bony structure | P_x | P_y | P_z | d_x | d_y | d_z |
> |---|---|---|---|---|---|---|---|
> | **Flexion–extension** | Humerus-ulna | 19.36 | −30.80 | 9.02 | 0.8531 | 0.0183 | −0.5108 |
> | **Pronation–supination** | Ulna-radius | 20.11 | −32.27 | 6.87 | −0.0604 | 0.9874 | 0.1465 |"

**先例 B 的肘 2 DoF = 屈伸 + 旋前旋后。而先例 A（Theia3D）的肘 2 DoF = Abd/Add。**
也就是说 —— **两套先例不只是「腕 2 DoF vs 3 DoF」的差别，
它们对「前臂那条长轴旋转挂在哪个关节上」给出了不同的答案。**

这对应一个经典的**串联链拓扑**选择，不是命名差异：

| 链拓扑 | 建模方式 | 出现在 |
|---|---|---|
| **两段前臂**（humerus→forearm→hand，旋转记在肘）| 肘 = 屈伸 + 旋前旋后 | 先例 B（DSEM 肌肉骨骼模型）|
| **三段前臂**（humerus→radius→ulna→hand，旋转记在腕）| 肘 = 屈伸 + 外展内收；腕 = 屈伸 + 内/外旋 | 先例 A（Theia3D）|

> ⚠️ **这直接决定词表怎么切**：
> 若走「两段前臂」，旋前旋后是**肘**的第二个自由度 → 词表里腕只剩屈伸（甚至与肘重复计入）；
> 若走「三段前臂」，旋前旋后是**腕**的第二个自由度 → 词表里腕含 Int/Ext。
>
> **不能同时两处都算** —— 那会把同一个物理旋转计两遍。

### 3.3 表面差异汇总：**五套约定对「旋前旋后挂哪里」给出五种答案**

| 约定 | 链拓扑 | **旋前/旋后记在哪** | 腕 DoF | 出处（全部已机器反查）|
|---|---|---|---|---|
| **A** 无标记动捕 Theia3D | 三段前臂 | **腕**（Int/Ext）| 2（屈伸 + 内/外旋）| 语料三 B1 拳击精度论文 |
| **B** 肌肉骨骼模型 17-DoF | 刚性前臂 | **肘**（与屈伸并列）| 3 | 语料三 B2，Table 3 功能轴表 |
| **C** Holzbaur 7-DoF 上肢 | 前臂旋转**独立成关节** | **独立前臂旋转关节**（肘与腕之间）| 2（屈伸 + 偏斜）| arXiv `2607.02796`（Cotton 等 2026）引 Holzbaur 2005 `10.1007/s10439-005-3320-7` |
| **D** 完整解剖（四关节）| 肱尺+肱桡+近端桡尺+远端桡尺 | **近端桡尺关节** | 该文未设腕 | Wang 等 2019 `10.1177/0954411919872400` |
| **E** 通行简化 | 刚性前臂 | **肘**（2 DoF：F/E + P/S）| — | 同上，Wang 2019 明指这是**简化**且**无共识** |
| **数据集参数化** | SMPL / Human3.6M / CMU | **无** —— wrist 是末端关节 | — | 见 [`DoF覆盖核查_公开数据集.md`](DoF覆盖核查_公开数据集.md) |

> 🎯 **同一个物理量，五种归属。这就是 V6c 的实质。**
>
> - 选 A → 词表出现「腕 Int/Ext」，**不能再报肘的 P/S**
> - 选 B / E → 词表出现「肘 P/S」，**腕只剩屈伸**（若同时报腕 Int/Ext 即为重复计数）
> - 选 C → 词表出现独立的「前臂旋转」token，**肘退化为 1 DoF**
> - 选 D → 最完整，但 **DoF 数最多、且要求可分辨桡尺相对运动**（对单目几乎不可能）
>
> **不能同时两处都算** —— 那会把同一个物理旋转计两遍。
> 而 Wang 2019 已明确：**这个选择领域内尚无共识**（§2.8.3），因此**必须论证，不能默认**。

### 3.4 与上一版的差异修正

先例 A 与 B 的表（保留，供逐项对照）：

| | 先例 A（无标记动捕）| 先例 B（肌肉骨骼模型）|
|---|---|---|
| 胸廓 root | 6 | 6 |
| 锁骨/肩带 | 2 | 3（shoulder girdle）|
| 肩关节 | 3 | 3 |
| 肘 | **2（Abd/Add）** | **2（屈伸 + 旋前旋后）** |
| **腕** | **2（屈伸 + 内/外旋）** | **3**（原文未命名三个角度）|
| 合计（不含 root）| 9 | 11 |

> 上一版核查里「三个公开数据集都没有旋前旋后」的表述**需要精确化**：
> **数据集参数化（SMPL / Human3.9M / CMU）没有；运动捕捉与肌肉骨骼模型约定有 —— 但有五种归属。**
> Theia3D 这类商用系统**是估得到腕内/外旋的**。

---

## 3.5 ISB 权威标识的第三重佐证（本轮升级为 A 级）

ISB Part II 的题名 / 卷 / 页 / DOI 现在有**四个相互独立的来源**一致：

| # | 来源 | 类型 |
|---|---|---|
| 1 | Crossref 权威记录 | DOI 直查（`10.1016/j.jbiomech.2004.05.042`，vol 38, p981-992）|
| 2 | 语料三 B1 拳击精度论文的参考文献表 | 引文内 |
| 3 | 语料三 B1 假肢腕单元论文的参考文献表 | 引文内 |
| 4 | 语料三 B2 肌肉骨骼模型论文的参考文献表 | 引文内 |

三处语料内引文的写法完全一致（首版）：

> "Wu G., Van Der Helm F. C. T., Veeger H. E. J., Makhsous M., Van Roy P., Anglin C., et al. (2005).
> ISB recommendation on definitions of joint coordinate systems of various joints for the reporting of
> human joint motion – part II: shoulder, elbow, wrist and hand. J. Biomech. 38, 981–992.
> **10.1016/j.jbiomech.2004.05.042**"

> 同行 Part I（踝、髋、脊柱）同样三处一致：`10.1016/S0021-9290(01)00222-6`，J Biomech 35, 543–548。
>
> **Part II 的正式题名里明确含 shoulder, elbow, wrist and hand** —— 这就是「Part II 才是腕角权威」的直接文本依据。
>
> 同一参考文献群中还有 Woltring (1994) `10.1016/0021-9290(94)90191-0`（3-D 姿态表示标准化提案），
> 是 ISB 坐标系的前身工作，可作为方法学溯源。

---

## 3.6 与 V7 相关的先验证据（**注意其适用边界**）

`mine_fulltext.py` 在语料三 B5 抽到一条与「单目能否估前臂旋前旋后」直接相关的陈述。
**来源**：Dasgupta 等 2023, *Machine Learning for Optical Motion Capture-Driven Musculoskeletal
Modelling from Inertial Motion Capture Data*, **Bioengineering** 10(5):510, doi `10.3390/bioengineering10050510`
（Crossref 已反查：Dasgupta 2023, vol 10, p510）· 4. Discussion：

> "…which often leads to differences in pose estimation accuracy between the measurement systems
> (e.g., **tracking long-axis rotations like shoulder internal/external rotation or elbow
> pronation/supination**). … Soft tissue artefacts are typically task-dependent and joint-dependent
> and **vary based on the joint's degree of freedom**."

> 🔍 **这条证据的适用边界（必须如实标注）**：
> 该文比较的是 **OMC（光学标记法）vs IMC（惯性动捕）** —— **两者都是在人体上贴了实体标记物/IMU 的仪器化测量**。
> **D435i 单目视频既无标记物也无传感器**，成像条件完全不同。
> 因此这条证据只能说明「**长轴旋转是动捕领域公认的精度薄弱环节，且误差随关节 DoF 数变化**」，
> **不能**用来论证「单目一定能/不能估出旋前旋后」。**V7 仍必须靠本机的工程验证清零。**
>

---

## 4. 对词表设计的直接后果

词表能不能含「前臂旋前/旋后」，取决于**监督来源**；含了以后挂在哪个关节，取决于**串联链拓扑**。
两条判断互相独立，必须分别定：

| 路径 | 能否含旋前旋后 | 挂在哪个关节 | 代价 |
|---|---|---|---|
| **A** 公开数据集为主（SMPL-H / AMASS / Human3.9M / CMU）| ❌ **不能** —— 这些参数化的 wrist 是末端关节 | — | 词表只到肩/肘/腕屈伸 |
| **B** 商用无标记系统的输出（如 Theia3D 上肢模型）| ✅ **能** —— 其 wrist 2 DoF 含 Int/Ext | **腕**（三段前臂，先例 A）| 需要该系统的输出做标签；且与 B 路线同源，**不构成独立验证** |
| **C** 自采 D435i + 自建上肢多体模型 | ⚠️ **取决于 V7**（单目能否估前臂旋转）| **自选**（五套里挑一套）| 需先验证 V7；**且必须同时定链拓扑并论证** |
| **C′** 自采 D435i + **直接复用先例 C 的 Holzbaur 7-DoF 上肢模型**（arXiv `2607.02796` 的做法）| ⚠️ 同上，取决于 V7 | **独立前臂旋转关节** | ⚠️ **该路径与 arXiv `2607.02796` 高度同构**（生物力学模型在环 + 关节角优化），见定位文档撞车节 |

> **新增的一个关键约束**：若走路径 B，用 Theia3D 的上肢输出当标签，
> 那么**你的模型就是在拟合 Theia3D 的运动学约定**——
> 这既是标签来源，也是它的上限，**不能同时把它当独立验证集**。
> 论文里必须显式声明这一点，否则会被指为循环论证。

> **路径 B/C/C′ 共用的第二个约束**：
> 选了三段前臂（旋转记腕）就**不能**同时报「肘的旋前旋后」；
> 选了两段前臂（旋转记肘）就**不能**同时报「腕的内/外旋」；
> 选了独立前臂旋转关节就**不能**再在腕或肘重复计入。
> **词表里这个 DoF 只能出现一次**，否则是同一物理量的重复计数，
> 会在评测里被直接指为构造错误。

---

## 4.5 给研究者的待裁定项：串联链拓扑（V6c）

**先说清楚性质**：这**不是一个"选个更标准的就能解决"的技术细节**。
Wang 等 2019（`10.1177/0954411919872400`）在原文里明确写：

> "**Still, no agreement has been reached on the most appropriate modelling approach of the forearm for
> estimating elbow and forearm kinematics.**"

所以正确做法不是"找到标准然后照做"，而是**声明选择 + 引用争议 + 论证本场景下的取舍**。

| 你的目标倾向 | 建议拓扑 | 理由 | 代价 |
|---|---|---|---|
| **与商用无标记系统对齐、复用其标签** | A（三段前臂，旋转记腕）| 与 Theia3D 等标签源同口径 | 若同源标签也当验证 → 循环论证 |
| **与主流肌肉骨骼模型对齐、便于接 OpenSim 生态** | B / E（刚性前臂，旋转记肘）| 通行做法，OpenSim/Holzbaur 生态多用 | 丢掉桡尺相对运动；且与 A 口径不可直接比 |
| **想要独立 token 表达前臂旋转、且贴近康复评估需求** | C（前臂旋转独立成关节）| 语义最干净，腕不被"借用"承担长轴旋转 | DoF 更多；**且与 arXiv `2607.02796` 高度同构，须在查重中正面区分** |
| **追求解剖完整** | D（四关节）| 最完整 | **对单目几乎不可行**（需分辨桡尺相对运动）—— 与 V7 冲突 |

**这一项我不替你决定。** 但必须提醒：选 C 时与 `2607.02796` 的区分度会明显变小，
若选 C 就要准备好回答"你和 Firouzabadi 等 2026 到底差在哪"。

---

## 5. 待核实（本轮无法从已取得全文闭合）

| # | 项 | 状态 | 说明 |
|---|---|---|---|
| **V6** | ISB Part II 权威标识 | ✅ **本轮升级为 A 级** | 题名/卷/页/DOI 有 4 个独立来源一致（Crossref + 语料内 3 处引文）；**但全文仍为 Elsevier 付费闭取，坐标定义本身尚未读到** |
| **V6a** | Theia3D 上肢模型中肘 2 DoF 的**具体角度名** | [ ] 未闭合 | 原文只写 Abd/Add；**本轮已取得两套反例对照**（先例 B/E 记肘、先例 C 记独立关节），Theia 侧官方命名仍须回官方模型文档 |
| **V6b** | Dumas & Wojtusch (2018) 回归方程的**系数** | [ ] 未闭合 | 该文以回归方程定义关节中心；未取到系数前，关节中心不可复现 |
| **V6c** | **串联链拓扑**的选用依据 | ✅ **本轮已获得答案，但答案本身是「无共识」** | Wang 2019 原文：**"no agreement has been reached on the most appropriate modelling approach of the forearm"**。故该项**不是可被文献关闭的技术缺口，而是需要研究者选定并论证的设计决策**。选项与代价见 §4.5 |
| **V6d** | 先例 B 腕 3 DoF 的**三个角度名** | [ ] 未闭合 | 原文只给总数 3，未命名（检索 radioulnar / ulnar deviation 等均无命中）|
| **V6e** | **Holzbaur 2005 全文** | [ ] **新命中，未入库** | 7-DoF 上肢模型的原始定义处；`10.1007/s10439-005-3320-7` 已 Crossref 反查，但**语料三内无此条目，建议经取件后补入 B2** |
| **V7** | D435i 单目能否可靠估计前臂旋前/旋后 | [ ] **仍未闭合** | 两条证据都指向"难"：①动捕领域先验弱项（§3.6）②Wang 2019 量化：**仅换追踪方式，P/S 范围就变 14.9%–43.2%**（§2.8.4）。**但两者均为仪器化测量，不能外推到单目**；仍须本机工程验证 |
| **V2** | SMPL-H / MANO 是否携带额外腕旋转参数 | [ ] 未闭合 | 本轮全文中未找到相关陈述 |
| **V3** | MANO 权威标识 | [ ] 未闭合 | Crossref 无记录 |
