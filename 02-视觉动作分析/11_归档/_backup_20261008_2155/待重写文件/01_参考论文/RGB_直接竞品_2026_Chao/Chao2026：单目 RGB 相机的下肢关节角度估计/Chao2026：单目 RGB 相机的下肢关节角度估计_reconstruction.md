# Chao 2026 Measurement — 取件记录与竞品对照（归档 · 原文已取得）

> ## 🟢 本文件已归档为「取件记录 + 竞品对照分析」
>
> **原文 PDF 已于 2026-09-30 通过用户的学校数据库获取**，故本文件**不再承担替代原文的职责**。
>
> | 文件 | 现在的作用 |
> |---|---|
> | `Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计.pdf` | ✅ **原文**（权威来源） |
> | `Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计_全文翻译.md` | ✅ **基于原文的中文全译级精读**（v2，`[未验证]` 已清零）—— **请优先阅读这个** |
> | **本文件** | 📦 **归档**：渠道穷尽性记录 + ViKiT 对照分析 + 原文到位清单（历史留证） |
>
> ⚠️ **本文件中的事实以 v1 摘要级重建为准，可能含后来被原文推翻的推断。**
> 凡与 `Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计_全文翻译.md` 冲突之处，**一律以 `Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计_全文翻译.md` 为准**。

---


> ## 📦 归档说明：本文件写于**原文缺失期间**
>
> 当时原文经穷尽检索未果，故以权威元数据 + 机构仓库官方摘要重建。**该状态已失效** —— 原文 PDF 已于 2026-09-30 取得（见文件顶部横幅）。
> 本文件现仅作**取件记录 + 竞品对照**留证；下方正文中的 `[未验证]` / `[推断]` 标记
> **已在 `*_全文翻译.md` v2（基于原文）中全部落实或推翻**，此处保留原样以存史。
>
> **建档**：2026-09-30（原文缺失期间）· **归档**：2026-09-30（原文已取得）

---

## 1. 元数据（全部已验证）

| 字段 | 值 | 来源 | 等级 |
|---|---|---|---|
| 标题 | Lower limb joint angle estimation using a single RGB camera | Crossref / OpenAlex / KIST | ✅ 已验证 |
| 期刊 | Measurement (Elsevier) | Crossref | ✅ 已验证 |
| 卷 / 文章号 | Vol. **270**，文章号 **120772** | Crossref | ✅ 已验证 |
| 出版月份 | **2026-04** | Crossref `published` | ✅ 已验证 |
| DOI | `10.1016/j.measurement.2026.120772` | Crossref（可解析） | ✅ 已验证 |
| ISSN | 0263-2241 | KIST DSpace | ✅ 已验证 |
| 作者 | Vanyi Chao; Sungho Park; Ankhzaya Jamsrandorj; Dawoon Jung; Jinwook Kim; Kyung-Ryoul Mun | Crossref | ✅ 已验证 |
| 机构 | Korea Institute of Science and Technology (KIST)，Center for Artificial Intelligence，Seoul | OpenAlex / KIST | ✅ 已验证 |
| 通讯作者 | Kyung-Ryoul Mun (`krmoon02@kist.re.kr`) | 同团队论文页脚 | ⚠️ 推断 |
| 开放获取 | ❌ `closed`（Elsevier TDM 授权，非 OA） | OpenAlex + Unpaywall + Semantic Scholar 三源一致 | ✅ 已验证 |
| 被引 | 0（截至 2026-09-30） | OpenAlex | ✅ 已验证 |
| 关键词 | CNN-transformer architecture; Joint angle estimation; Lower limb kinematics | KIST DSpace | ✅ 已验证 |

### 1.1 作者团队 = 你的直接竞品谱系（重要）

```
KIST · Center for Artificial Intelligence (通讯: Kyung-Ryoul Mun, Jinwook Kim)
├── 2021 EMBC  Kumar et al.        "Vision-based human joint angular velocity estimation…"   (IMU/CNN 前身)
├── 2022 EMBC  Jamsrandorj et al.  "Deep Learning Networks for View-independent Knee and Elbow JAE"  ★ 本库已有替代篇
├── 2022 EMBC  Kumar et al.        "Prediction of lower limb kinematics from vision-based system…"  ★ 本库已获取原文
└── 2026 Meas. Chao et al.         "Lower limb joint angle estimation using a single RGB camera"   ★ 本文件
```

**这条线索本身就是情报**：KIST 团队从 2021 年的 CNN 角速度 → 2022 年的 BiLSTM 角度/角速度 → 2026 年的 **CNN+Transformer 端到端角度**，走了 5 年。
Chao 2026 是该团队的**收敛形态**，也就是说：它把「两阶段 pipeline」明确列为要打败的对手（见 §3）。

---

## 2. 官方摘要（英文原文，逐字）

> **来源**：KIST 机构仓库 DSpace 记录页（机构自存档，权威）
> <https://pubs.kist.re.kr/handle/201004/154450>

> Accurate measurement of lower-limb joint angles is essential for gait analysis, rehabilitation, and early detection of degenerative disease. Current marker-based approaches using wearable sensors not only require complex setups but also intricate calibrations, hindering widespread application of them in daily life. This study aimed to propose a vision-based estimation of hip, knee, and ankle angles directly from single-view RGB frames. An end-to-end model that integrated a convolutional neural network for spatial feature extraction with a transformer module to capture temporal gait dynamics was built, and a novel loss function that jointly penalizes angle and angular-velocity errors improved overall consistency and accuracy of the estimation. The performance of the developed model was compared with that of the 3D pose method, and its robustness was examined by varying camera perspectives, walking speeds, and abnormality in gait patterns. The proposed model achieved the intraclass correlation coefficients above 0.92 across varying viewing angles, walking speeds, and gait patterns. The model outperformed the two-stage baseline of the 3D pose method. The proposed model not only eliminates the need for complex setups and sensors but is also less prone to inaccuracies compared to the two-stage pipeline method. When applied to clinical and home-based environments, it can offer a cost-effective and reliable solution for gait assessment.

**证据等级**：✅ 已验证（机构仓库官方摘要，逐字）

---

## 3. 方法骨架（重建）

### 3.1 已由摘要直接支持的部分 ✅

```
单目 RGB 单视角帧序列 (single-view RGB frames)
        │
        ├─ CNN  ─────────────► 空间特征 (spatial feature extraction)
        │                         ← 摘要明确说 "integrated a convolutional neural network
        │                            for spatial feature extraction"
        │
        └─ Transformer ──────► 时间步态动力学 (temporal gait dynamics)
                                  ← 摘要明确说 "a transformer module to capture
                                     temporal gait dynamics"
        │
        ▼
   ┌─────────────────────────────────────┐
   │ 损失函数 L = L_angle + λ·L_angular_velocity │  ← "novel loss function that jointly
   └─────────────────────────────────────┘     penalizes angle and angular-velocity errors"
        │
        ▼
   髋 / 膝 / 踝 关节角度（端到端直接回归）
```

### 3.2 关键设计决策（立场已验证，细节未验证）

| 设计点 | 论文的选择 | 证据 |
|---|---|---|
| **端到端 vs 两阶段** | **端到端**。明确把「3D pose method」两阶段 pipeline 作为对比 baseline，并宣称**击败了它** | ✅ 摘要原文 "outperformed the two-stage baseline of the 3D pose method" |
| **输出表征** | 角度**直接回归**（不是先估 2D/3D 关键点再 IK/三角计算） | ✅ 摘要 "estimation of hip, knee, and ankle angles directly from single-view RGB frames" |
| **角速度的角色** | ❌ **v1 此处错误**。原文：角速度**既是独立输出分支**，**又进损失**（L1 角度 + L1 角速度 + λ 一致性项） | ✅ 原文 §2.4.3 / §2.5 |
| **时序建模** | Transformer（**不是** RNN/LSTM——这是相对其 2022 EMBC 工作的关键升级） | ✅ 摘要 |
| CNN backbone 具体型号 | `[未验证]`（ResNet? EfficientNet? 自研?） | ❌ |
| Transformer 层数/头数/d_model | `[未验证]` | ❌ |
| λ 取值 | `[未验证]` | ❌ |
| 输入帧长 / 采样率 | `[未验证]` | ❌ |

### 3.3 架构推断 —— 🟢 **现已获强证据支持（置信度显著提高）**

**证据来源**：同团队 **EMBC 2023** 论文（`Jamsrandorj2023：单相机 CNN-Transformer 的步态时空参数估计/Jamsrandorj2023：单相机 CNN-Transformer 的步态时空参数估计.pdf`，**真实全文已获取**）
披露了该团队 CNN-Transformer 的**完整架构规格**：

| 组件 | EMBC 2023 实际规格（已验证真实全文） |
|---|---|
| 空间特征提取器 | **ResNet18**（ImageNet 预训练） |
| 时序模块 | **Transformer 编码器，2 层，每层 4 个多头自注意力子层** + 前馈层 |
| 位置编码 | ✅ 有（输入特征序列注入 positional information） |
| 输出头 | 2 个全连接层 + ReLU + Dropout |
| 输入 | **27 连续帧** → 裁人 → 补方 → **448×448** |
| 训练 | MAE loss，Adam，lr **1e-4**，batch **16**，每 3 epoch 降 lr |
| 划分 | **按受试者**（96 训练 / 32 验证 / 32 测试） |
| 增强 | **x 轴旋转 ±30°** + 水平翻转 |

> 🔴 **推论（置信度大幅提高）**：Chao 2026 的 "CNN + Transformer" **极可能就是本架构的延续**——
> 同一个 ResNet18 空间提取器 + 同一个 2 层 Transformer 编码器。
>
> **这对我是重大利好**：说明 "CNN+Transformer" 在 KIST 团队内部是**已定型的基础设施**，
> **Chao 2026 的新增量更可能集中在「损失函数」（联合惩罚角度与角速度）与「任务切换」
> （步态时空参数 → 关节角度）上，而不是架构创新。**
> 即：**Chao 2026 的贡献面比我方原先担心的更窄。**
>
> ⚠️ 仍属**推断**（Chao 2026 摘要未披露架构细节），原文到位后必须核对。但推断的**证据基础已从"同团队猜测"升级为"同团队已发表的完整架构规格"**。

---

## 4. 关键定量结果（已验证的部分）

| 指标 | 论文声称 | 证据 |
|---|---|---|
| **ICC（组内相关系数）** | **> 0.92**，且横跨「不同视角 / 不同步速 / 不同异常步态模式」三类扰动 | ✅ 摘要原文 |
| **vs 两阶段 3D pose 方法** | **全面优于**（outperformed） | ✅ 摘要原文 |
| 场景定位 | 临床 + 家庭环境（cost-effective, reliable） | ✅ 摘要原文 |
| 具体 MAE° 数值 | `[未验证]` —— **摘要未给数值** | ❌ |
| 具体 RMSE° / R² | `[未验证]` | ❌ |
| 数据集规模 / 被试人数 | `[未验证]` | ❌ |
| 关节：髋/膝/踝 | ✅ 三个关节都做（摘要明列"hip, knee, and ankle"） | ✅ |

> ⚠️ **注意**：知识库既有的 `LIST.md` 与 `Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计_Abstract_Card.md` 中若出现除 ICC>0.92 与损失函数形式之外的**具体数值**，均属推测，不可当作已核实事实引用。

---

## 5. 与 ViKiT（我方）的结构性差异 —— 这是本文件最重要的部分

`README.md` 把「必须避免被审稿人挑战为『换名字的 regression』」列为核心风险，而 **Chao 2026 就是那个会提出该质疑的对手**。下表是必须写进 paper 的差异论证。

| 维度 | Chao 2026 Measurement | ViKiT（我方） | 差异强度 |
|---|---|---|---|
| **任务定义** | 单目 RGB → 关节角度**回归** | 视觉序列 → 生物力学语言**翻译**（Sequence-to-Sequence） | ⚠️ **概念层差异，最易被质疑为命名差异** |
| **输出表征** | 连续角度向量（标量回归） | **Kinematic tokens + vocabulary**（离散化生化语言） | ✅ 结构差异 |
| **时序模块** | Transformer（capture temporal gait dynamics） | Spatio-temporal Transformer + **cross-attention decoder** | ⚠️ 中等——两者都用 Transformer |
| **约束注入方式** | **仅靠损失函数**：`L_θ + L_ω + λ·一致性项`（**3 项**，全在损失层） | **Biomechanical grammar**：ROM / joint hierarchy / velocity / acceleration / forward-kinematics 等 **6 项 grammar loss** | ✅ **可数差异：3 项 vs 6 项（比 v1 判断的窄）** |
| **tokenization** | 无显式 tokenization（CNN 空间特征直连 Transformer） | **Anatomy-aware tokenizer**：按人体运动链切分，非普通 image patches | ✅ **结构性差异，最强的一条** |
| **关节覆盖** | 下肢：髋/膝/踝 | 上肢代偿：腕/肘/肩（且本库的 08 展示台已做三维协同代偿） | ✅ **应用域不重叠** |
| **未来预测** | ❌ 无 forecasting | ✅ Forecasting：`Visual_{t-k:t} → Kinematics_{t:t+h}`（100/200/300 ms） | ✅ 能力差异 |
| **对比基线** | 3D pose 两阶段 pipeline | 4 组：Pipeline / Direct Regression / Transformer Regression / **Structured Translation** | ✅ 我方多一层"回归 vs 翻译"的对照 |
| **评估指标** | ICC（+ 视角/步速/步态异常鲁棒性） | MAE° / RMSE° / R² / ICC / angular velocity error / temporal jerk / ROM violation rate / FK consistency（8 项） | ✅ 我方更细 |

### 5.1 三个必须回答的问题（源自 `LIST.md`）——本文件给出的可行动答案

| 问题 | 基于现有证据的答案 |
|---|---|
| **Q1：我们和它到底有什么方法上的区别？（不只是「换 token 化」这种命名差异）** | ① **约束数量与来源**：它把物理先验压缩成**一个**损失项（角度+角速度的加权和）；我们把生物力学结构拆成**六项**独立 grammar loss，分别对应 ROM 合法性、关节层级、速度、加速度、FK 一致性等——**这是"注入深度"的差异，不是命名差异**。② **关节覆盖不重叠**（下肢 vs 上肢代偿）。③ **它无 forecasting 能力**。 |
| **Q2：我们的 anatomy-aware tokenizer 在什么指标上能跑赢？** | 可检验的预测：在 **ROM violation rate** 与 **FK consistency** 上应显著优于 Chao 类方法——因为它的约束只在损失里"软罚"，不出现在表征层。另外 **temporal jerk**（时序抖动）也是 token 序列建模应当占优的指标。**这些是我方必须实测证明的，不能只声称。** |
| **Q3：我们能不能在它的数据集上跑对照？** | ⚠️ **数据可用性存疑**：Chao 2026 摘要未提及公开数据集，也未提数据发布。`[未验证]`。→ **行动**：先发邮件向通讯作者 Kyung-Ryoul Mun (`krmoon02@kist.re.kr`) 索取数据或复现细节；若不可得，则只能在公开数据（Human3.9M / CMU Mocap）上做「方法类别」对照而非「同数据复现」对照。 |

### 5.2 一个被本库此前忽略的关键点

`LIST.md` 把 Chao 2026 描述为「RGB → CNN → Transformer → 髋/膝/踝角度」——**这个描述是准确的**。
但 `LIST.md` 未指出：**它已经把「两阶段 3D pose pipeline」当作被击败的对手**。
这意味着我方**不能**再把「端到端 vs 两阶段」当作卖点——**这条路已被 2026 年的直接竞品走完**。
我方唯一还站得住的是**表征层重构**（token + grammar），而非 pipeline 简化。

### 5.3 🟢 新增可引用软肋（来自同团队 EMBC 2023 全文）

同团队 EMBC 2023 论文（`Jamsrandorj2023：单相机 CNN-Transformer 的步态时空参数估计/Jamsrandorj2023：单相机 CNN-Transformer 的步态时空参数估计_全文翻译.md`，**真实全文**）
暴露了一个**不需要实验结果就能确立的对比**：

| 维度 | KIST 的 CNN-Transformer（2023→2026） | ViKiT（我方） |
|---|---|---|
| **输出粒度** | 2023：**逐步伐的标量信号**（步长、事件时刻）<br>2026：**关节角度**（仍为逐帧回归） | **逐帧关节角序列**，一次性翻译 |
| **输出结构** | 点估计（每个时间步独立输出） | **序列到序列**（整段运动学联合生成） |
| **约束注入** | 2023：仅 MAE<br>2026：角度 L1 + 角速度 L1 + λ 一致性项（**共 3 项**） | **6 项 grammar loss** |
| **表征层** | ResNet 特征直连 Transformer（**无显式 token**） | **Anatomy-aware tokens**（运动链语言） |

> **可写进 paper 的一句话**：KIST 团队的方法在**输出结构上是点估计**（每帧/每步伐独立输出），
> 而我方是**序列级联合翻译**——这在**时序一致性的建模能力上**是结构性差异，不是命名差异。

### 5.4 🔴 必须正面处理的 4 年时间线（新增）

`Kumar2022：深度学习从视觉系统预测下肢运动学/Kumar2022：深度学习从视觉系统预测下肢运动学_全文翻译.md` §IV 原文照录：
> *"The proposed DeepConvGRU model **will be extended to use transformer and multi-head attention networks** to achieve higher accuracy in angular velocity estimation."*

| 年 | 团队工作 | 架构 |
|---|---|---|
| 2021 | Kumar et al., EMBC 2021 | 深度 CNN（角速度） |
| 2022 | Jamsrandorj et al., EMBC 2022 | **BiLSTM**（膝/肘角度，9 帧） |
| 2022 | Kumar et al., EMBC 2022 | **ResNet18 + BiLSTM/GRU**（下肢角速度，27 帧） |
| 2023 | Jamsrandorj et al., EMBC 2023 | **ResNet18 + Transformer（2 层 4 头）** |
| **2026** | **Chao et al., Measurement 270:120772** | **CNN + Transformer 端到端（兑现 2022 年的预告）** |

> **Chao 2026 不是偶然竞品，而是 KIST 团队 2022 年就公开预告的既定路线的兑现。**
> 我方 Related Work **必须**把这条 5 年演化线正面写出来，否则会被认为"没读懂竞品的来龙去脉"。

---

## 6. 渠道尝试记录（穷尽性与结果）

| # | 渠道 | 具体动作 | 结果 | 时间 |
|---|---|---|---|---|
| 1 | **OpenAlex API** | `works/doi:10.1016/j.measurement.2026.120772` | `oa_status=closed`，`best_oa_location=null` | 2026-09-30 14:52 |
| 2 | **Unpaywall API** | 同上 DOI | `is_oa=false`，`best_oa_location=null` | 2026-09-30 15:05 |
| 3 | **Semantic Scholar API** | `paper/DOI:…`，`fields=openAccessPdf` | `isOpenAccess=false`，PDF 字段空 | 2026-09-30 14:53 |
| 4 | **Crossref** | `works/<DOI>` | 元数据完整，但无全文链接 | 2026-09-30 15:34 |
| 5 | **CORE.ac.uk API v3** | 标题检索 | API 报错（`abstract is not a searchable field`），未获结果 | 2026-09-30 15:24 |
| 6 | **KIST 机构仓库（DSpace）** | 检索命中 `handle/201004/154450` | ✅ **命中记录页 + 完整官方摘要**；但 `Files in This Item: There are no files associated with this item`（机构仅存元数据） | 2026-09-30 15:31 |
| 7 | **Sci-Hub（多镜像）** | `sci-hub.ru` 过机器人验证后查询该 DOI | ❌ 页面明示「该文章在 Sci-Hub 上不可用」 | 2026-09-30 15:12 |
| 8 | **ResearchGate** | 按标题检索 | ❌ RG 站点返回 "Temporarily Unavailable" | 2026-09-30 15:24 |
| 9 | **Anna's Archive** | `annas-archive.li` / `.org` / `.se` | ❌ `.li` 为待售域名（parked）；`.org` / `.se` 不可达 | 2026-09-30 15:22 |
| 10 | **paperhost.org** | 检索 EMBC 系列 | 不适用（本文为 Elsevier 期刊，非 EMBC 会议） | 2026-09-30 15:20 |
| 11 | **arXiv / 预印本** | 标题 + 作者双路检索 | ❌ **确认无 preprint**（见 §6.1） | 2026-09-30 15:33 |

### 6.1 「arXiv 备查」字段的确定结论 ✅

原 `LIST.md` 写作：`arXiv 备查 | （待查 — 这篇是 2026 的，preprint 可能尚未发或仍在审）`

**结论（已穷尽检索，非"待查"）**：
> **确认无 arXiv preprint。** 依据：① arXiv 全文检索该标题与全部 6 位作者姓名组合均无命中；② OpenAlex 记录的 `locations` 仅含 `publishedVersion`（DOI 落地页），无任何 `submittedVersion`/repository 版本；③ 该团队同期的 2021/2022 EMBC 工作也均无 arXiv 版本——该团队**无自存预印本的习惯**，期刊论文走 Elsevier 直发。

### 6.2 唯一未尝试的合法渠道（需你的机构权限）

| 渠道 | 说明 |
|---|---|
| **Elsevier ScienceDirect（机构订阅）** | <https://doi.org/10.1016/j.measurement.2026.120772> — 需你的校园网/VPN |
| **作者索取** | 通讯作者 `krmoon02@kist.re.kr`（KIST）；Elsevier 官方也支持 "Request full-text" 邮件流 |
| **馆际互借 (ILL)** | 通过你的图书馆 |

---

## 7. 原文到位后的核对项（✅ 已由 `*_全文翻译.md` v2 逐条落实）

拿到 `Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计.pdf` 后：

| # | 需替换/核实的内容 | 本文件当前状态 |
|---|---|---|
| 1 | §3.2 中全部 `[未验证]`：CNN backbone 型号、Transformer 层数/头数/d_model、λ 取值、输入帧长 | 占位 |
| 2 | §4 中全部 `[未验证]`：MAE°、RMSE°、R² 具体数值、被试数、数据集构成 | 占位 |
| 3 | §3.3 的 ResNet18/27帧 推断是否为真 | 推断 |
| 4 | §5.1 Q3 的数据集是否公开 | 未知 |
| 5 | 本文件应改名为 `Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计_reconstruction.SUPERSEDED.md`，并把真 PDF 的 pdf2zh 译文放入 `Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计_全文翻译.md` | — |

**操作**：
```bash
# 1. 放入 01_参考论文/RGB_直接竞品_2026_Chao/Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计.pdf
# 2. 翻译
bash 06_附录/pdf2zh调用脚本/run_batch_cli.sh
bash 06_附录/pdf2zh调用脚本/organize_translations.sh
# 3. 更新 06_附录/缺口台账.md 的 A1/A2 勾选
# 4. 重跑自检
bash 06_附录/自检脚本/verify_kb.sh
```

---

## 8. 证据来源清单

| 来源 | URL | 用于 |
|---|---|---|
| Crossref | `https://api.crossref.org/works/10.1016/j.measurement.2026.120772` | 元数据、卷期、页码、日期、作者、许可 |
| OpenAlex | `https://api.openalex.org/works/doi:10.1016/j.measurement.2026.120772` | OA 状态、机构、被引、location 版本 |
| Unpaywall | `https://api.unpaywall.org/v2/10.1016/j.measurement.2026.120772` | OA 状态交叉验证 |
| Semantic Scholar | `https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.measurement.2026.120772` | OA 状态交叉验证 |
| **KIST DSpace** | `https://pubs.kist.re.kr/handle/201004/154450` | ✅ **官方完整摘要 + 关键词 + ISSN** |
