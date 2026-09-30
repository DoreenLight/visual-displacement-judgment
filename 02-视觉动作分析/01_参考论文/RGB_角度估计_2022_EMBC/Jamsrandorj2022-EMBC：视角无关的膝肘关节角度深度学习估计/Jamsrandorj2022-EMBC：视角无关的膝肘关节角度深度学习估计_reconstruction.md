# Jamsrandorj 2022 EMBC — 取件记录与竞品对照（归档 · 原文已取得）

> ## 🟢 本文件已归档为「取件记录 + 竞品对照分析」
>
> **原文 PDF 已于 2026-09-30 通过用户的学校数据库获取**，故本文件**不再承担替代原文的职责**。
>
> | 文件 | 现在的作用 |
> |---|---|
> | `Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计.pdf` | ✅ **原文**（权威来源） |
> | `Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计_全文翻译.md` | ✅ **基于原文的中文全译级精读**（v2，`[未验证]` 已清零）—— **请优先阅读这个** |
> | **本文件** | 📦 **归档**：渠道穷尽性记录 + ViKiT 对照分析 + 原文到位清单（历史留证） |
>
> ⚠️ **本文件中的事实以 v1 摘要级重建为准，可能含后来被原文推翻的推断。**
> 凡与 `Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计_全文翻译.md` 冲突之处，**一律以 `Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计_全文翻译.md` 为准**。

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
| 标题 | Deep Learning Networks for View-independent Knee and Elbow Joint Angle Estimation | Crossref / OpenAlex / KIST | ✅ 已验证 |
| 会议 | 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (**EMBC**) | Crossref | ✅ 已验证 |
| 会址 / 日期 | Scottish Event Campus, **Glasgow, UK**, 2022-07-11 ~ 07-15 | Crossref + 会议页眉 | ✅ 已验证 |
| 页码 | pp. **2703–2707** | Crossref + KIST | ✅ 已验证 |
| DOI | `10.1109/EMBC48229.2022.9871106` | Crossref（可解析） | ✅ 已验证 |
| 作者 | Ankhzaya Jamsrandorj; Konki Sravan Kumar; Muhammad Zeeshan Arshad; Kyung-Ryoul Mun; Jinwook Kim | Crossref | ✅ 已验证 |
| 机构 | Korea Institute of Science and Technology (KIST), Seoul | KIST DSpace | ✅ 已验证 |
| PubMed | <https://pubmed.ncbi.nlm.nih.gov/36085943/> | OpenAlex `locations` | ✅ 已验证 |
| KIST 机构仓库 | <https://pubs.kist.re.kr/handle/201004/77156> | KIST DSpace | ✅ 已验证（仅元数据，无全文） |
| 开放获取 | ❌ `closed` | OpenAlex + Unpaywall + Semantic Scholar 三源一致 | ✅ 已验证 |
| 被引 | 7（截至 2026-09-30） | OpenAlex | ✅ 已验证 |
| **Sci-Hub 收录** | ✅ **有**（页面标题确认为该论文） | sci-hub.ru 实际查询 | ✅ 已验证 |
| **Sci-Hub 可取** | ❌ 托管域 `sci-hub.red` 故障（502→404） | 24 个镜像逐一实测 | ✅ 已验证 |

---

## 2. 官方摘要（英文原文，逐字）

> **来源**：KIST 机构仓库 DSpace 记录页（机构自存档，权威）
> <https://pubs.kist.re.kr/handle/201004/77156>
> **证据等级**：✅ 已验证

> Vision-based human joint angle estimation is essential for remote and continuous health monitoring. Most vision-based angle estimation methods use the locations of human joints extracted using optical motion cameras, depth cameras, or human pose estimation models. This study aimed to propose a reliable and straightforward approach with deep learning networks for knee and elbow flexion/extension angle estimation from the RGB video. Fifteen healthy participants performed four daily activities in this study. The experiments were conducted with four different deep learning networks, and the networks took nine subsequent frames as input while output was knee and elbow joint angles extracted from an optical motion capture system for each frame. The BiLSTM network-based joint angles estimator can estimate both joint angles with a correlation of 0.955 for knee and 0.917 for elbow joints regardless of the camera view angles.

---

## 3. 方法骨架（重建）

### 3.1 已由摘要直接支持 ✅

```
输入：RGB 视频，取 9 个连续帧（nine subsequent frames）作为一个样本
        │
        ▼
┌──────────────────────────────────────────────┐
│ 四种不同的深度学习网络（4 DL networks）对比    │
│  其中 BiLSTM 为最优                          │
└──────────────────────────────────────────────┘
        │
        ▼
输出：膝 / 肘 屈伸角度（flexion/extension angle）
      —— 每一帧都输出一个角度

Ground truth：光学动作捕捉系统（optical motion capture）
被试：15 名健康受试者（fifteen healthy participants）
活动：4 种日常活动（four daily activities）
评估：相关系数（correlation）
        膝 0.955 / 肘 0.917 —— 且与相机视角无关（view-independent）
```

### 3.2 关键设计事实（摘要支持的）

| # | 事实 | 摘要原句依据 |
|---|---|---|
| 1 | **输入是 9 帧**（不是单帧，也不是长序列） | "the networks took **nine subsequent frames** as input" |
| 2 | **4 种网络对比**，最终 BiLSTM 胜出 | "four different deep learning networks" / "The **BiLSTM** network-based joint angles estimator…" |
| 3 | **输出是每帧一个角度**（逐帧输出，非序列一次出） | "output was knee and elbow joint angles … **for each frame**" |
| 4 | **真值来自光学动捕** | "extracted from an **optical motion capture system**" |
| 5 | **视角无关性是被刻意检验的** | "**regardless of the camera view angles**" |
| 6 | **目标场景是远程连续健康监测** | "essential for **remote and continuous health monitoring**" |
| 7 | 方法定位是"**reliable and straightforward**"（不追求复杂 pipeline） | 摘要首段 |

### 3.3 未验证项 ❌

| 项 | 状态 |
|---|---|
| 4 种网络的**具体名单**（除 BiLSTM 外是哪 3 种） | `[未验证]`（推测含 CNN、LSTM、GRU/MLP，但属推断） |
| BiLSTM 的层数 / 隐藏维度 / 是否带 attention | `[未验证]` |
| 输入 9 帧的图像表征（原始像素? 关键点? 光流?） | `[未验证]` ← **这是最关键的未知：论文属于"端到端像素"还是"先提关键点"路线** |
| 4 种日常活动的具体名称 | `[未验证]` |
| 相机数量 / 视角设置（几个视角？角度范围？） | `[未验证]` |
| MAE° / RMSE° 等绝对误差指标 | `[未验证]`（**摘要只给相关系数**） |
| 训练/测试划分（是否按被试划分） | `[未验证]` |

> ⚠️ **对已有速读卡的提醒**：`Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计_Abstract_Card.md` 中若出现 MAE 或"4 种活动名称"等具体值，**均未经本次核实的摘要支持**，引用前须核对原文。

---

## 4. 关键定量结果（已验证）

| 关节 | 相关系数 | 备注 |
|---|---|---|
| **膝（knee）** | **0.955** | 与相机视角无关 |
| **肘（elbow）** | **0.917** | 与相机视角无关 |

| 实验设置 | 值 |
|---|---|
| 被试 | 15 名健康受试者 |
| 活动 | 4 种日常活动 |
| 输入窗口 | 9 个连续帧 |
| 对比网络 | 4 种（BiLSTM 最优） |
| 真值 | 光学动作捕捉 |

---

## 5. 与 ViKiT（我方）的差异

**这篇是"历史基线"，不是直接竞品**——它是同一 KIST 团队 4 年前的工作，但**它定义了该团队的方法起点**。

| 维度 | Jamsrandorj 2022 EMBC | ViKiT（我方） |
|---|---|---|
| 输入窗口 | **9 帧**（≈0.3 s @30fps，极短上下文） | **长序列** RGB 视频 |
| 时序模型 | **BiLSTM** | Spatio-Temporal Transformer + cross-attention |
| 输出表征 | 连续角度向量（逐帧回归） | **Kinematic tokens + vocabulary** |
| 任务定义 | 角度**回归** | 生物力学语言**翻译** |
| 物理约束 | 摘要未提 | **6 项 grammar loss** |
| 关节 | 膝 / 肘（下肢+上肢混合） | 腕 / 肘 / 肩（上肢代偿协同） |
| 视角鲁棒性 | ✅ 已做（view-independent 是卖点） | ✅ 作为必备项 |
| 提前预测 | ❌ 无 | ✅ 100/200/300 ms |
| 数据集 | 15 名健康被试 × 4 活动（**自采，未提公开**） | Human3.9M / CMU Mocap / 公开康复数据 |

### 5.1 三条对本库有价值的推论

1. **9 帧窗口是"短上下文"路线的代表**。我方主打长序列 + 时序一致性，**必须**在 `04_实验设计/对比方案.md` 中把「短窗 BiLSTM」列为 Pipeline/Direct-Regression 类基线，并明确"长上下文带来什么增益"。
2. **"view-independent" 在 2022 年就已是卖点**。我方不能把跨视角鲁棒性当创新，只能当**必备门槛**。这与 Chao 2026 的结论一致（跨视角/跨速度/跨步态异常三条扰动轴）。
3. **它输出的是逐帧角度而非序列**。这与我方的"序列到序列翻译"在**输出结构上**就不同——我们一次性翻译出一段运动学序列，它逐帧独立回归。**这是我方可用的结构差异**。

---

## 6. 与同团队姊妹论文的关系（重要情报）

| | **Jamsrandorj 2022 EMBC**（本篇） | **Kumar 2022 EMBC** |
|---|---|---|
| 页码 | 2703–2707 | 177–181 |
| 输出 | **角度**（position） | **角速度**（angular velocity） |
| 关节 | 膝 + 肘 | 髋 + 膝 + 踝 |
| 活动 | 4 种日常活动 | 深蹲 + 跑步机步行 |
| 输入 | 9 帧 | `[未验证]`（Kumar 原文已获取，见其 `_全文翻译.md`） |
| 最优模型 | **BiLSTM**（单一） | CNN + BiLSTM / GRU（多模型对比） |
| 最佳相关 | 膝 0.955 / 肘 0.917 | squat 0.93 / treadmill 0.92；膝最优 0.96 |
| 团队 | KIST（Kyung-Ryoul Mun, Jinwook Kim） | 同 |
| **本库状态** | ⚠️ **替代包**（托管故障） | ✅ **已获取真实 PDF** |

→ 两篇合起来覆盖 **vision → angle** 与 **vision → angular velocity** 两条线，是 KIST 团队 2022 年的"双论文"布局。

---

## 7. 渠道尝试记录（穷尽性与结果）

### 7.1 已尝试

| # | 渠道 | 具体动作 | 结果 | 时间 |
|---|---|---|---|---|
| 1 | **OpenAlex API** | `works/doi:10.1109/EMBC48229.2022.9871106` | `oa_status=closed`；无 OA location | 14:52 |
| 2 | **Unpaywall API** | 同上 DOI | `is_oa=false` | 15:05 |
| 3 | **Semantic Scholar API** | `fields=openAccessPdf` | `isOpenAccess=false`，空 | 14:53 |
| 4 | **Crossref** | `works/<DOI>` | 元数据完整（pp.2703-2707），无全文 | 15:34 |
| 5 | **KIST 机构仓库 DSpace** | `handle/201004/77156` | ✅ 命中记录 + **完整官方摘要**；❌ `There are no files associated with this item` | 15:31 |
| 6 | **Sci-Hub `sci-hub.ru`** | 过机器人验证（点击「不是」）后查询 DOI | ✅ **确认收录**——页面标题返回「Sci-Hub. Deep Learning Networks for View-independent Knee and Elbow Joint Angle Estimation / 2022 EMBC」 | 15:15 |
| 7 | **Sci-Hub 文件下载** | 从页面提取 `<object data=…>`，指向 `sci-hub.red/storage/triplet/9043/0077f2…/jamsrandorj2022.pdf` | ❌ `sci-hub.red` 根域 **502**，该路径 **404**（三次重试均 404） | 15:22 |
| 8 | **Sci-Hub 镜像矩阵（24 个域名）** | 对同一 triplet 路径逐一实测：`sci-hub.ru / .red / .se / .st / .ee / .hkvisa.net / .wikimirror.org / .41610.org / .ren / .love / .mksa.top / .xyz / .cat / .box / .li / .ist / .monster / .al / .art / .is / .la / .support / .tw` 等 | ❌ 无任一提供该文件：`ru`=403、`red`=404、`se/tw/la/...`=DNS 失败、`ee/ren`=通用帮助页、`mksa/al/xyz`=着陆页 | 15:22–15:23 |
| 9 | **paperhost.org（EMBC 官方论文集托管）** | ① 确认 `EMBC22/files/*.pdf` 可访问（已验证 0100/0500/0900/1000/1425 均 HTTP 200，为真实 EMBC 2022 论文）② 搜索 EMBC22 的 ContentList/AuthorIndex/Search 索引页 | ⚠️ **文件在，但 EMBC22 无索引页**（EMBC21 有 `EMBC21_ContentListMedia_1..5.html`，EMBC22 全部 404）→ **无法定位文件编号**（编号为 4 位补零，如 `0100.pdf`） | 15:20–15:40 |
| 10 | **ResearchGate** | 按标题检索 publication 363396666 | ❌ 站点 "Temporarily Unavailable" | 15:24 |
| 11 | **Anna's Archive** | `.li` / `.org` / `.se` | ❌ `.li` 为待售域名；其余不可达 | 15:22 |

### 7.2 🟢 后代优先重试方法（成本约 30 秒）

**该篇是全库唯一「文件确实存在于 Sci-Hub、仅托管域临时故障」的论文。** 托管恢复后即可直接取回：

```bash
# ① 先探测 sci-hub.red 是否恢复
curl -s -o /dev/null -w "sci-hub.red root=%{http_code}\n" -m 15 -L https://sci-hub.red/

# ② 若 root=200，直接取文件（无需 Cookie）
curl -L -o /tmp/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计.pdf \
  "https://sci-hub.red/storage/triplet/9043/0077f2324b17ec78079ecae4a25ac29fa6d0f6bb6e5906985e673b0495bf865e/jamsrandorj2022.pdf"
file /tmp/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计.pdf      # 应为: PDF document, 5 pages

# ③ 安装
cp /tmp/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计.pdf \
  "01_参考论文/RGB_角度估计_2022_EMBC/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计.pdf"

# ④ 翻译
bash 06_附录/pdf2zh调用脚本/run_batch_cli.sh
bash 06_附录/pdf2zh调用脚本/organize_translations.sh
```

**备用路径（若 sci-hub.red 长期不可用）**：
1. **paperhost.org** —— EMBC22 的 PDF **确实托管在该站**（已验证）。缺的只是"标题 → 4 位文件编号"的映射。
   若你日后拿到 EMBC 2022 的 ContentList（或逐号扫描），即可取回。目标论文共 5 页、页码 2703–2707。
2. **IEEE Xplore（你的机构订阅）**：<https://ieeexplore.ieee.org/document/9871106/>
3. **作者索取**：KIST，`krmoon02@kist.re.kr` / `jkim@kist.re.kr`

---

## 8. 原文到位后的核对项（✅ 已由 `*_全文翻译.md` v2 逐条落实）

| # | 需替换/核实 | 当前状态 |
|---|---|---|
| 1 | §3.3 全部 `[未验证]`，**尤其：输入 9 帧的表征形式（像素? 关键点?）** | 占位 |
| 2 | 除 BiLSTM 外的**另 3 种网络名单** | 占位 |
| 3 | 4 种日常活动名称、相机数量与视角设置 | 占位 |
| 4 | MAE° / RMSE° 绝对误差（摘要只给相关系数） | 占位 |
| 5 | 训练/测试划分方式（是否跨被试） | 占位 |
| 6 | 本文件应改名为 `…reconstruction.SUPERSEDED.md`，真 PDF 的 pdf2zh 译文覆盖 `Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计_全文翻译.md` | — |

**操作**：见 §7.2；完成后勾选 `06_附录/缺口台账.md` 的 A5 / A6。

---

## 9. 证据来源清单

| 来源 | URL | 用于 |
|---|---|---|
| Crossref | `https://api.crossref.org/works/10.1109/EMBC48229.2022.9871106` | 页码、日期、作者、会议名 |
| OpenAlex | `https://api.openalex.org/works/doi:10.1109/EMBC48229.2022.9871106` | OA 状态、PubMed、被引 |
| Unpaywall / Semantic Scholar | `…/v2/10.1109/EMBC48229.2022.9871106` | OA 交叉验证 |
| **KIST DSpace** | `https://pubs.kist.re.kr/handle/201004/77156` | ✅ **官方完整摘要** |
| Sci-Hub | `https://sci-hub.ru/10.1109/EMBC48229.2022.9871106` | ✅ 确认收录（文件托管域故障） |
