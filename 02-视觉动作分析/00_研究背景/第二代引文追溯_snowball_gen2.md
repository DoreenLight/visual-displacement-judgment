# 第二代引文追溯（snowball_gen2）— 4 条新撞车 + 1 条比原证据更强的空白支撑

> **这份文档解决什么**：F5/F6 撞红之后，我担心「只追了一层引文」会漏掉后续工作。
> 于是从第一代新捞出的 5 篇撞车工作再滚一代，拿到 **369 条引用记录 / 368 个唯一 DOI**。
> 结果是：**新坐实 4 条撞车**（其中 2 条直接压住「离散词表」和「Tokenizer 命名」），
> 同时**意外挖到一条 A 级证据，比原来支撑主张 C 的任何一条都强**。
>
> **最后更新**：2026-10-01
> **执行工具**：`06_附录/生物力学语料工具/snowball_gen2.py` → `snowball_gen2.json`

---

## 0. 一句话结论

> **坏消息**：连续两代追溯确认「离散符号化人体姿态」和「Tokenizer 命名」在 CV 顶会均已被占，
> 且 pose grammar 线**已从 AAAI 2018 延续到 TPAMI 2022 与 TBME 2023**（不是一篇孤立旧文）。
> **好消息**：CVPR 2022 的上肢-手 SOTA 方法**自己写明前臂 twist 无法从关节点确定**，
> 并在评估中把腕旋转置零 —— 这把主张 C 从「检索未命中」升级为「**顶会作者自陈的局限**」。

---

## 1. 覆盖范围与口径

| 项 | 值 |
|---|---|
| 前向通道 | **OpenCitations COCI**（唯一在本机可用的前向引文源；只吃有 DOI 的） |
| 后向通道 | arXiv PDF → 抽参考文献（**纯 arXiv 无 DOI，只能走后向**） |
| 种子数 | 5 |
| 前向记录 | 369 条 → **368 个唯一 DOI**（1 条无 DOI 已被丢弃） |
| 定向标记命中 | **294** 个唯一 DOI |
| 其中「纯视觉」 | **274**（3D 人体姿态估计方法与综述，**引用 grammar 论文但不构成撞车**） |
| 其中「非纯视觉」 | **20** ← 真正要人工看的 |
| 后向参考文献命中 | 4 个 arXiv 种子的参考文献里**定向命中 0** |
| 入库状态 | **全部不入 `index.json`**（摘要/全文级判定 ≠ 入库，见 §5） |

> ⚠️ **274 : 20 这个比例本身是结论**：
> pose grammar 那条线被引 220 次，绝大多数是 CV 的 3D HPE 方法在顺手引用，**不是「有人做了同样的事」**。
> 撞车必须靠**逐条看标题+摘要**，不能靠计数。

### 5 个种子

| 种子 | 是什么 | 前向引用数 |
|---|---|---|
| `10.1609/aaai.v32i1.12270` | Learning Pose Grammar（AAAI 2018，F6） | 220 |
| `10.1109/cvpr46437.2021.00847` | PoseAug 可微姿态增广（CVPR 2021） | 150（解析 124） |
| `10.3390/bioengineering10050510` | Dasgupta 2023 OMC/IMC 建模 | 14 |
| `10.1109/TMRB.2025.3605962` | 可微生物力学 TMRB 2026（F2） | 6 |
| `10.1007/978-3-031-72943-0_27` | **VQ-HPS 向量量化人体姿态**（ECCV 2024） | 5 |

---

## 2. 🚨 新坐实的 4 条撞车

> 以下 4 条的 DOI、题名、刊会、年份、卷期页码、作者**全部经 Crossref 逐条机器反查**，
> 三条有 arXiv 版的另经 arXiv API 反查。**没有一条来自记忆。**

### F7 ⭐⭐ 「离散符号化人体姿态」范式已被 ECCV 2024 占

| 字段 | 值（Crossref 核） |
|---|---|
| 题名 | *VQ-HPS: Human Pose and Shape Estimation in a Vector-Quantized Latent Space* |
| 作者 | Fiche Guénolé, Leglaive Simon, Alameda-Pineda Xavier, Agudo Antonio 等 5 人 |
| 出处 | **ECCV 2024**（LNCS，`10.1007/978-3-031-72943-0_27`，pp. 471–490，Crossref 登记年 2025） |
| 预印本 | **arXiv `2312.08291`**（arXiv API 反查一致，v4） |

摘要原文（机器取得）：

> "This work introduces a novel paradigm to address the HPSE problem, involving a
> **low-dimensional discrete latent representation** of the human mesh and framing HPSE as a
> **classification task**. Instead of predicting body model parameters or 3D vertex coordinates,
> we focus on **predicting the proposed discrete latent representation**…"

**为什么这条最致命**：我们打算写的核心机制是
「把连续关节角离散成符号 token，**当作分类任务**预测」。
VQ-HPS 已经把**这套范式**用在了人体网格上，并明确自称 "novel paradigm"。

- ❌ 不能再写「首次将姿态离散化为符号/隐空间并做分类」。
- ✅ 仍开放的差异：**目标量不同**（人网格 vs 临床关节角）、
  **词表语义不同**（VQ-HPS 是学出来的隐码本，**没有可解释的 biomechanical 符号**）、
  **无 grammar、无物理可实现性约束、无临床**。
- 📌 关键论点可升级：VQ-HPS 的码本是**无监督学出来的**，不可解释、不可编辑；
  我们的词表是**有语法、有 ROM 边界、可人工检查**的符号系统。
  **这才是真差异**，不是「我们也做了 VQ」。

### F8 ⭐⭐ 「Tokenizer」这个词在姿态估计里已被 CVPR 2024 占

| 字段 | 值（Crossref 核） |
|---|---|
| 题名 | *Hourglass Tokenizer for Efficient Transformer-Based 3D Human Pose Estimation* |
| 作者 | Li Wenhao, Liu Mengyuan, Liu Hong, Wang Pichao 等 6 人 |
| 出处 | **CVPR 2024**, pp. 604–613（`10.1109/cvpr52733.2024.00064`） |
| 预印本 | **arXiv `2311.12028`**（v2） |

摘要原文（机器取得）：

> "we present a plug-and-play pruning-and-recovering framework, called **Hourglass Tokenizer
> (HoT)**, for efficient transformer-based 3D human pose estimation from videos. Our HoT begins
> with **pruning pose tokens of redundant frames** and ends with recovering full-length tokens…"

**注意两者的 "tokenizer" 是同形异义**：

| | Hourglass Tokenizer（CVPR 2024） | 我们设计的 tokenizer |
|---|---|---|
| 干什么 | **剪掉冗余时间帧的 token**，省 FLOPs | 把连续关节角**量化为符号** |
| 在哪一侧 | **输入**（帧/时间） | **输出**（运动学符号） |
| 目的 | 效率 | 离散语法约束 |
| 产物 | 稀疏 token 序列 | **有语法的 codebook** |

- ❌ 不能写「首次提出 pose tokenizer」。
- ✅ 差异真实存在且可辩护，但**必须主动区分**：
  HoT 的 tokenizer 是**时间维的稀疏化**，我们的是**符号维的量化**。
- 📌 **这直接影响 `03_方法设计/Anatomy_aware_Region_Encoder.md` 的命名**（G 类待修）。
  文档里「解剖感知 tokenizer」的「tokenizer」与 HoT 撞名，建议改称
  **codebook / vocabulary / symbol table** 以免评审直接撞脸。

### F9 ⭐⭐ 「Transformer 从单目视频回归上肢关节旋转」已被 CVPR 2022 占

| 字段 | 值（Crossref 核） |
|---|---|
| 题名 | *Spatial-Temporal Parallel Transformer for Arm-Hand Dynamic Estimation* |
| 作者 | Liu Shuying, Wu Wenbin, Wu Jiaxian, Lin Yue（4 人） |
| 出处 | **CVPR 2022**, pp. 24091–24100（`10.1109/cvpr52688.2022.01987`） |
| 预印本 | **arXiv `2203.16202`**（v1） |

摘要原文（机器取得）：

> "We propose an approach to estimate **arm and hand dynamics from monocular video** by
> utilizing the relationship between arm and hand… recovering accurate and plausible
> **arm twists** and hand gestures from in-the-wild videos still remains a challenge…"

这是**方法层面最直接的撞车**：单目视频 → 上肢关节旋转 → Transformer → CVPR。
而且它明确点名 **arm twists**（前臂扭转），正是我们的 V6c/V7 目标域。

> 🔴 **但它同时给了我们一条 A 级证据 —— 见 §3。**

### F10 pose grammar 线已延续到 TPAMI / TBME（不是一篇孤立旧文）

| 字段 | 值（Crossref 核） |
|---|---|
| ① | *Monocular 3D Pose Estimation via Pose Grammar and Data Augmentation*，Xu Yuanlu, Wang Wenguan 等 6 人，**IEEE TPAMI 2022, pp. 6327–6344**（`10.1109/tpami.2021.3087695`） |
| ② | *Progressive Direction-Aware Pose Grammar for Human Pose Estimation*，Zhou Lu, Chen Yingying, Wang Jinqiao，**IEEE TBME 2023, pp. 593–605**（`10.1109/tbiom.2023.3315509`） |

⚠️ 两条**均无 arXiv 版**（arXiv 按题名反查 0 命中），**无摘要**（Crossref 未提供），本轮为**标题级判定**。

- 📌 **含义**：F6 的撞红不是「一篇 2018 年旧文」，而是一条**持续到 2023 年的活跃研究线**。
  Related Work **不能只引 AAAI 2018 就当交代完**，必须同时引 TPAMI 2022 与 TBME 2023。
- ⚠️ 标题级判定的限度：尚不能断言它们**也明确排除力学**。若要写进论文并做差异辩护，
  须回这两篇全文核实其损失函数/约束定义。**列为待办 F10a。**

---

## 3. ✅ 意外收获：主张 C 从「检索未命中」升级为「顶会作者自陈」

PAHMT（CVPR 2022，`2203.16202`）的**全文**里（已下载 `arXiv PDF` 并 `pdftotext` 抽取核对），
有三句话直接构成 A 级证据：

**① Related Work（§2 开头，机器抽取原文）**

> "However, a common problem with these methods is that 3D skeletal rotations cannot be
> accurately recovered from 3D joint locations when the goal is to capture human skeletal
> motion, i.e., **the twist component cannot be determined by joint locations.**"

**② §4.3 评估指标节**

> "It is noteworthy that accurate hand rotations can be recovered from accurate 3D hand joints
> by inverse kinematic algorithms, while **arm rotations cannot be accurately recovered from
> 3D arm joints.** Therefore, we use 1) the overall MPJPE; 2) MPJPE of hands; **3) MPJRE of arms**…"

**③ Figure 3 说明**

> "To compare the rotations of arms, **we set the rotations of wrists to zeros** in the bottom row
> of (b) and (c)."

### 由此可以确证的三件事

| # | 事实 | 证据位置 |
|---|---|---|
| 1 | **前臂扭转（twist）是公认的可辨识性难题**，不是我们臆想的 | PAHMT §2 原文自述 |
| 2 | 该 SOTA 方法**从未单独报告腕/前臂旋转的角度精度** | 唯一旋转指标是 lumped `MPJRE(arms)`；**无逐关节分解** |
| 3 | 它**在可视化对比中把腕旋转置零**，即把腕从旋转比较里排除 | Figure 3 说明原文 |

其报告的旋转数字（Table 1/3，机器读取）：`MPJRE(arms)` = **0.0375**（自采 mocap 集）、
**0.1614**（MIXAMO 渲染集，对比 FrankMocap 0.2779 / ExPose 0.2984）。
**训练与评估数据均为自采动捕或渲染合成，不是临床 MMC 基准。**

> 📌 **这直接加强了主张 C**：原表述依赖「我们检索没找到」，
> 现在有了「**领域内 SOTA 方法自己承认做不到、并且绕开了**」的直接出处。
> 证据等级从「检索未命中（C 级推断）」升到 **A 级**。

### 🔴 同时，它也**证伪了主张 C 的现有措辞**

`论文定位陈述_收窄版.md` §2 主张 C 原文写的是：

> "no markerless pipeline has reported the accuracy of **wrist or forearm-rotation** kinematics"

**这句现在不成立。** 因为本轮同时查到
**Francia Carlalberto, Donno Lucia, Motta Filippo, Cimolin Veronica 等，
*Accuracy and Reliability of Markerless Human Pose Estimation for Upper Limb Kinematic
Analysis Across Full and Partial Range of Motion Tasks*，
Applied Sciences 2026, art. 1202（`10.3390/app16031202`）**，
其摘要明确：

> "a markerless pose estimation system for **shoulder, elbow and wrist flexion–extension**
> analysis under full and partial ROM tasks… **ICC values above 0.91 for the elbow and 0.94 for
> the shoulder** in full ROM trials. **Wrist tracking presented the lowest RMSE values**…"

→ **腕屈伸的精度已经被报告了**（而且是最好的那个）。
→ **主张 C 必须收窄为：只讲「前臂旋前/旋后（radioulnar / pronation-supination）」。**

已同步修正到 `论文定位陈述_收窄版.md` §2 与 `V7_单目前臂旋转可行性_证据链.md`（新增 E10）。

> 🚨🚨 **2026-10-03 补注：上面这个「收窄后的版本」也被证伪了。**
> 收窄后主张 C 写的是 *"no markerless pipeline has reported the accuracy of forearm
> pronation/supination (radioulnar) kinematics"* ——
> **Nakatsuka et al. 2021（VISAPP, `10.5220/0010222703790386`）报了**：
> 它对 SMPL 运动链上 **肩/肘 4 关节**做 swing-twist 分解，
> **肘关节的 twist 轴（肘→腕方向向量）即前臂长轴**，Table 1 报 **LElb 0.18 / RElb 0.15 rad**。
> → 台账 **F29**（全文级），见 `V7_单目前臂旋转可行性_证据链.md` §1.11。
> 📌 **两轮证伪的方法论教训是同一条**：「没人报过 X」这种句子，
> **每被一个先例推翻一次就要重写一次，而每写一次就再押一次注**。
> **正确做法是不要写全称否定**，改写「我们检索到的那些报了 X 的方法，它们的 Y 是怎样的」。

---

## 4. 其余 16 条「非纯视觉」命中：判定为不撞

以下均已逐条看过标题+标记，无一构成撞车，**登记备查**（防同一批人再翻出来）：

| 类别 | 条目 | 判定 |
|---|---|---|
| **人体解析**（人体解析 ≠ 姿态估计） | Blended Grammar Network `10.1007/978-3-030-58586-0_12`；Grammar-Induced Wavelet Network `10.1109/tip.2022.3181486` | 用 attribute grammar 做**像素级人体解析/分割**，非关节角度。F6 血统的同门，不撞 |
| **手部姿态估计**（3D hand pose，非临床） | Monocular Total Capture `10.1109/cvpr.2019.01122`；3D Hand Pose FIM `10.1109/tcsvt.2022.3142787`；GCN 混合分类回归 `10.1016/j.image.2021.116564`；图方法单图 3D 手 `10.1007/s10489-022-03390-x`；ArtiBoost `10.1109/cvpr52688.2022.00277`；遮挡/分类加权 `10.1016/j.patcog.2022.109217`；扰动位置编码 `10.1007/978-3-031-72624-8_25`；弱监督归一化 `10.3390/app14093578` | **手部 21 关键点 MANO 回归**，非临床关节角、无 ROM/语法约束。相邻不撞 |
| **遮挡鲁棒性评测** | Quantification of Occlusion Handling `10.1109/tmm.2022.3158068` | 评遮挡鲁棒性，非上肢临床 |
| **外科 activity grammar**（ICRA 2025） | `10.1109/icra55743.2025.11127338` | 「activity grammar」用于**腹腔镜意图预测**，与人体关节 grammar 不同源。**但可作为「grammar 概念已扩散到机器人/外科」的旁证**，写论文引言时可用 |
| **上肢动捕验证** | Markerless vs Marker-based **shoulder** angle `10.1109/tencon66050.2025.11375551`（TENCON 2025, pp.1135–1139） | **只做肩**，会议为区域性 TENCON，**不构成对 CVPR/ICCV 级工作的威胁**。相邻 |
| **上肢 markerless 精度** | `10.3390/app16031202` | **不撞车但证伪主张 C 措辞**（见 §3），必须引用 |
| **临床数字指标视角** | ~~From Metrics to Meaning `10.2196/preprints.87339`（JMIR 预印本）~~ → **已升级闭合，见 §6** | 3 组焦点小组、11 名临床医生，**质性研究**。🚨 **本轮追查发现它已正式发表**（JMIR Rehabil Assist Technol 2026，`10.2196/87339`），且取到了**发表版全文** → 提炼为**主张 E**，是本轮最重要的非撞车发现 |
| **康复临床试验** | Armeo X-over 方案 `10.3390/mps9020031` | 脊髓损伤 OT 试验方案，无视觉方法论 |
| **人体工效学视频监控** | `10.1016/j.bspc.2026.109810` | 视频 + 物理负荷风险评估，非关节角翻译 |
| **智能服装 digital twin** | `10.1109/jsen.2025.3601187` | 相邻 |
| **NeRF BoW 无人机** | `10.3390/rs16163038` | **明显误命中**（关键词碰撞，标记器误报） |

---

## 5. 我没做的事（诚实边界）

- ❌ **这 368 个 DOI 一条都没进 `index.json`** —— 摘要级/标题级判定不等于入库。
  真要引用必须走 `curate → select_candidates → resolve_multisource` 链，
  否则「可引用索引」会混入未经交叉证实的条目。
- ❌ **TPAMI 2022 / TBME 2023 只做了标题级判定** —— 尚未核实它们是否同样明确排除力学（F10a）。
- ❌ **前向只覆盖 5 个种子** —— OpenCitations 只吃有 DOI 的，
  `2608.03127`（DigitCode）、`2601.09949`（金融占名）**两个纯 arXiv 种子没有前向网络**。
  它们的引文只能靠后向抽 PDF，而本轮后向**命中 0**（`2609.18667` 抽到 28 条参考文献，无定向命中）。
- ❌ **没有做第三代追溯** —— 若 F10a 需要，应从 TPAMI 2022 再滚一代。
- ❌ **检索仍为英文单语** —— 中文与非英语文献未覆盖。

---

## 6. 变更记录

| 日期 | 变更 | 说明 |
|---|---|---|
| 2026-10-01 | 建档 | 第二代引文追溯完成：369 记录 / 368 唯一 DOI / 294 定向命中 / 20 条人工判定；**新坐实 F7–F10 四条撞车**；**主张 C 措辞被 `10.3390/app16031202` 证伪并已收窄为「前臂旋前旋后」**；PAHMT 全文三条原句升为主张 C 的 A 级支撑 |
| 2026-10-01 | **§6 补记（本轮追查）** | **F7b 从「预印本待取件」升级为「已发表 + 全文已入库」**：Pohl 等已刊于 **JMIR Rehabil Assist Technol 2026**（`10.2196/87339`）。`preprints.jmir.org` 直连返回 **HTTP 202 空响应 = 反爬**（按纪律未绕过），改由 **Europe PMC 取得发表版 JATS XML**（`PMC13293569`），走完整链路入库，**A 级双源**。全文给出三条对本项目有直接约束的事实 → 已提炼为**主张 E「临床侧可读性约束」**（见 `论文定位陈述_收窄版.md` §2）：临床只看 smoothness/efficiency/compensation 三属性，**"joint angle" 全文 0 次**，且**不熟 jerk 与 interjoint coordination**。⚠️ 残留子项：补充表 S2/S3 未取（登记 **V9g**） |

---

## 7. 一条本轮才意识到的系统性盲区

> **我这一路撞红，几乎全在查「CV / 机器人侧有没有人做过」。**

F5、F6、F7、F8、F9、F10 —— 六条全是 CV 侧的。
本轮读到 Pohl 等那篇才反应过来：**临床侧是一个完全独立的撞红面，而我从头到尾没查过。**

而且它给出的结论是**反直觉的**：
我一直默认「离散符号词表 + 可解释语法」对临床端更友好，
**事实是临床不熟 jerk、不熟 interjoint coordination，连 joint angle 都没出现过。**

→ 下一步如果继续做撞红，**应该专门补一轮「临床/康复侧」检索式**，
而不是继续在 CV 侧加深度。这一轮已登记为待办方向（尚未做成检索式）。
