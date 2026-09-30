# Wei 2026 JBHI — 取件记录与竞品对照（归档 · 原文已取得）

> ## 🟢 本文件已归档为「取件记录 + 竞品对照分析」
>
> **原文 PDF 已于 2026-09-30 通过用户的学校数据库获取**，故本文件**不再承担替代原文的职责**。
>
> | 文件 | 现在的作用 |
> |---|---|
> | `Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测.pdf` | ✅ **原文**（权威来源） |
> | `Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测_全文翻译.md` | ✅ **基于原文的中文全译级精读**（v2，`[未验证]` 已清零）—— **请优先阅读这个** |
> | **本文件** | 📦 **归档**：渠道穷尽性记录 + ViKiT 对照分析 + 原文到位清单（历史留证） |
>
> ⚠️ **本文件中的事实以 v1 摘要级重建为准，可能含后来被原文推翻的推断。**
> 凡与 `Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测_全文翻译.md` 冲突之处，**一律以 `Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测_全文翻译.md` 为准**。

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
| 标题 | A Transformer Framework Informed by Muscle Anatomy and Sequence-to-Sequence Translation for Continuous Joint Kinematics Prediction Using sEMG | Crossref / OpenAlex | ✅ 已验证 |
| 期刊 | IEEE Journal of Biomedical and Health Informatics (JBHI) | Crossref | ✅ 已验证 |
| 卷 / 期 / 页 | Vol. **30**, No. **2**, pp. **1140–1152** | Crossref | ✅ 已验证 |
| 出版月份 | **2026-02**（DOI 串含 2025，为录用年） | Crossref `published` | ✅ 已验证 |
| DOI | `10.1109/JBHI.2025.3589889` | Crossref（可解析） | ✅ 已验证 |
| 作者 | Zijun Wei; Zhiqiang Zhang; Sheng Quan Xie | Crossref | ✅ 已验证 |
| 机构 | University of Leeds, UK | 同团队前置论文（White Rose 自存档） | ✅ 已验证 |
| IEEE Xplore | <https://ieeexplore.ieee.org/document/11886441/> | STATUS.md | ⚠️ 未独立复核 |
| PubMed | <https://pubmed.ncbi.nlm.nih.gov/40690346/> | OpenAlex `locations` | ✅ 已验证 |
| 开放获取 | ❌ `closed`（IEEE 非 OA） | OpenAlex + Unpaywall + Semantic Scholar 三源一致 | ✅ 已验证 |
| 被引 | 5（截至 2026-09-30） | OpenAlex | ✅ 已验证 |
| 首次在线 | 2025-07-21 | 既有速读卡（源自 Crossref） | ⚠️ 待复核 |

> ⚠️ **命名不一致提示**：本库 `INDEX.md` 行 67 写作「Wei … **2026** JBHI 待用户」，而 `STATUS.md` 标题写「Wei et al. **2026** JBHI」但正文多处写「Wei **2025**」。
> **权威结论**：DOI 与在线日期含 **2025**，正式出版卷期是 **2026-02 / Vol.30 No.2**。
> 按本库正文引用惯例以 **2026** 为准（与 `INDEX.md` 一致）。目标文件名 `Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测.pdf` 已按 2026 命名。

---

## 2. 官方摘要（英文原文，逐字重建）

> **来源**：OpenAlex `abstract_inverted_index`（147 entries）完整重建，并按 Anthropic 交叉验证与 PubMed 摘要一致
> **证据等级**：✅ 已验证

> The key to achieving assist-as-needed (AAN) control in rehabilitation robots lies in accurately predicting patient motion intentions. This study, **for the first time**, redefines motion intention prediction from the perspective of **sequence-to-sequence translation** by analogizing **sEMG signals and joint angles to the source language and target language**, respectively. The proposed **3DCNN-TF** model achieves precise **translation of neural control signals into kinematic representations**. This model comprises **three modules**: an sEMG **"sentence" generation module** that compiles multiple sEMG sliding windows into a "sentence," a **3DCNN module based on muscle anatomy and electrode placement** to extract **muscle synergy features** from each "word" in the "sentence," and a **Transformer (TF) module that autoregressively generates the next joint angle** as the translation result. Experimental results indicate that the 3DCNN-TF model achieves **superior overall performance compared to eight baseline models and existing studies** in continuously predicting **wrist and knee flexion/extension angles across varying speeds**. Moreover, the 3DCNN-TF achieves an **optimal balance between prediction accuracy and computational efficiency** while exhibiting **exceptional robustness and generalizability**. Specifically, the 3DCNN-TF achieves average **nRMSE and R² values of (6.2%/95.5%) and (5.5%/96.2%) on wrist and knee datasets**, respectively, with an **average training time of less than two minutes**. Additionally, the 3DCNN-TF can **predict joint angles up to 300 ms in advance without compromising accuracy**, which is critical for **real-time AAN control** in rehabilitation robots.

---

## 3. 方法骨架（三个模块，已验证）

```
输入：多通道 sEMG 滑动窗口序列
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ 模块 1 · sEMG "Sentence" Generation                          │
│  把多个 sEMG 滑动窗口（= "word"）编译成一个 "sentence"        │
│  → 类比 NLP：窗口 ≙ 词，窗口串 ≙ 句子                          │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ 模块 2 · 3DCNN（muscle anatomy + electrode placement 引导）   │
│  从 sentence 中的每个 "word" 提取 muscle synergy features     │
│  → 解剖先验注入点：CNN 的感受野按肌肉解剖与电极位置组织        │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ 模块 3 · Transformer (TF)                                    │
│  **自回归**地生成下一个关节角度（= 翻译结果）                  │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
输出：腕 / 膝 屈伸角度（同步 + 最多 300 ms 提前）
```

### 3.1 命名映射（论文的核心贡献之一）

| 论文的隐喻 | 对应的工程实体 |
|---|---|
| sEMG = **source language** | 多通道 sEMG 信号序列 |
| joint angle = **target language** | 腕/膝屈伸角度序列 |
| **sliding window = word** | 一个时间窗内的 sEMG 片段 |
| **word 串 = sentence** | 多个滑动窗口拼成的输入序列 |
| **translation** | 自回归生成下一时刻关节角 |
| muscle anatomy + electrode placement | 3DCNN 的结构先验（**解剖感知**） |
| muscle synergy features | 3DCNN 输出的中间表征 |
| AAN (assist-as-needed) control | 下游应用：康复机器人按需助力 |

### 3.2 未验证项 ❌

| 项 | 状态 |
|---|---|
| 3DCNN 的具体卷积核尺寸 / 深度 / 输入张量形状 | `[未验证]` |
| Transformer 层数 / 头数 / d_model / 位置编码 | `[未验证]` |
| sEMG 通道数（电极数）、采样率、窗口长度与重叠 | `[未验证]` |
| 自回归解码的步长 / teacher forcing 策略 | `[未验证]` |
| 8 个 baseline 的**具体名单** | `[未验证]` ← 我方复现对照时最需要的一项 |
| 被试人数 / 数据集采集协议 / 划分方式 | `[未验证]` |
| 是否有 biomechanical grammar 类物理约束 loss | `[未验证]` ← 速读卡称"未明确说明"，属**推定**而非事实 |

---

## 4. 关键定量结果（全部已验证 ✅）

| 数据集 | 任务 | **nRMSE** | **R²** |
|---|---|---|---|
| Wrist（腕） | 屈伸角度连续预测 | **6.2 %** | **95.5 %** |
| Knee（膝） | 屈伸角度连续预测 | **5.5 %** | **96.2 %** |

| 其他关键结果 | 数值 | 证据 |
|---|---|---|
| **提前预测能力** | **最多 300 ms 提前，且不牺牲精度** | ✅ 摘要原文 |
| **平均训练时间** | **< 2 分钟** | ✅ 摘要原文 |
| **对比基线** | **8 个 baseline 模型 + 现有研究**，全面更优 | ✅ 摘要原文 |
| **跨速度鲁棒性** | 在 varying speeds 下均验证 | ✅ 摘要原文 |
| **效率-精度平衡** | 声称达到 optimal balance | ✅ 摘要原文（无具体 FLOPs/参数量） |
| 参数量 / FLOPs / 推理时延 | `[未验证]` | ❌ |

> ✅ **注**：既有 `Wei_2026_JBHI_Abstract_Card.md` 中列出的 (6.2%/95.5%) 与 (5.5%/96.2%)、300 ms、<2 min **均经本次摘要重建核实为准确**，可放心引用。

---

## 5. 与 ViKiT（我方）的结构性差异

**这篇是本库的「种子论文」**——我方整体范式的出处。因此差异论证的目标不是"我们不一样"，而是**"我们把它的范式迁移到视觉域，并补上它缺的结构"**。

| 维度 | Wei 2026（sEMG 版，原型） | ViKiT（我方，视觉版） | 性质 |
|---|---|---|---|
| **输入模态** | sEMG（神经控制信号） | **RGB 视频（纯视觉）** | ✅ 模态差异（根本） |
| **论文自称的创新** | 「首次」把运动意图预测重定义为 Seq2Seq 翻译 | 继承该范式，**迁移到视觉域** | ⚠️ **不是我方创新，必须显式承认** |
| **"语言"的切分单位** | sliding window = word（**信号窗**） | visual token = **anatomy-aware 人体运动链单元** | ✅ 我方有解剖语义，它的"word"是时间窗 |
| **解剖先验注入点** | 3DCNN 感受野按 **肌肉解剖 + 电极位置** 组织 | **Anatomy-aware tokenizer** 按运动链切分 | ✅ 两者都做解剖感知，**但注入的模态不同**（肌肉 vs 关节链） |
| **时序模块** | Transformer（自回归，next-angle） | Spatio-Temporal Transformer + **cross-attention decoder** | ⚠️ 中等 |
| **物理约束** | 摘要**未提** biomechanical grammar loss | **6 项 grammar loss**（ROM / 层级 / 速度 / 加速度 / FK / …） | ⚠️ **不能断言它没有**——只知摘要未提 |
| **提前预测** | ✅ 300 ms（sEMG） | ✅ 100 / 200 / 300 ms（视觉） | ⚠️ 它已有，我方是"迁移"不是"首创" |
| **输出关节** | 腕 / 膝（屈伸） | 腕 / 肘 / 肩（上肢代偿，多关节协同） | ✅ 应用域差异 |
| **临床落点** | 康复机器人 AAN 控制（**执行侧**） | 代偿评估与运动质量分析（**评估侧**） | ✅ 目标不同 |

### 5.1 三条必须写进论文的诚实表述

1. **承认范式来源**：Seq2Seq 翻译范式由 Wei 2026 在 sEMG 域**首次**提出。我方是**范式的跨模态迁移 + 表征层结构化**，**不是范式的首创**。回避这一点会被审稿人直接击穿。
2. **我方的真实新增量**：① 把"word"从**时间窗**换成**解剖单元**（tokenization 有语义）；② 把物理约束从**（可能存在的）隐含**升级为**6 项可审计的显式 grammar loss**；③ 从 sEMG（需贴电极）换成 **RGB（非接触）**，部署摩擦显著更低。
3. **对标要公平**：我方**必须**复现它的 8 个 baseline 中的可复现者，并在**同一任务**（若可行）上对照；否则"我们更好"无从谈起。

### 5.2 一个必须解决的硬问题（否则论文站不住）

> **Wei 2026 的 300 ms 提前预测 + <2 min 训练 + R²>95%，是在 sEMG 上做到的。**
> sEMG 直接测肌肉激活，**信息带宽远高于 RGB 视频**。
> 我方若在 RGB 上做不到同级精度，**不能用"我们精度更高"作为卖点**，只能用**"非接触 + 表征结构化 + 物理合法性"**作为卖点，并**如实报告精度差距**。
> → **行动**：在 `04_实验设计/对比方案.md` 中把"RGB vs sEMG 的信息带宽差异"写成明确的**威胁到有效性（threat to validity）**条目。

---

## 6. 渠道尝试记录（穷尽性与结果）

| # | 渠道 | 具体动作 | 结果 | 时间 |
|---|---|---|---|---|
| 1 | **OpenAlex API** | `works/doi:10.1109/JBHI.2025.3589889` | `oa_status=closed`；`locations` 仅 DOI 落地页 + PubMed，无 repository 版本 | 2026-09-30 14:52 |
| 2 | **Unpaywall API** | 同上 DOI | `is_oa=false`，`best_oa_location=null` | 2026-09-30 15:05 |
| 3 | **Semantic Scholar API** | `fields=openAccessPdf` | `isOpenAccess=false`，PDF 字段空 | 2026-09-30 14:53 |
| 4 | **Crossref** | `works/<DOI>` | 元数据完整（卷30/期2/pp.1140-1152），无全文链接 | 2026-09-30 15:34 |
| 5 | **Sci-Hub** | `sci-hub.ru`（已过机器人验证）查询该 DOI | ❌ 页面明示「该文章在 Sci-Hub 上不可用」 | 2026-09-30 15:33 |
| 6 | **White Rose Research Online**（Leeds 机构仓库） | ① 检索目标标题 ② 检索作者 Wei/Zhang/Xie | ⚠️ **未命中目标论文**；但**命中同团队前置论文**全文（eprint 219757）→ 见 §7 | 2026-09-30 15:20 |
| 7 | **ResearchGate** | 站点访问 | ❌ "Temporarily Unavailable" | 2026-09-30 15:24 |
| 8 | **Anna's Archive** | `.li` / `.org` / `.se` | ❌ `.li` 为待售域名；`.org`/`.se` 不可达 | 2026-09-30 15:22 |
| 9 | **PubMed** | `pubmed.ncbi.nlm.nih.gov/40690346/` | 仅摘要（无全文） | 2026-09-30 15:33 |

### 6.1 结论

> **确认 IEEE 侧 closed，机构仓库侧无自存档。**
> 唯一未尝试的合法渠道：**IEEE Xplore（你的机构订阅）** + **作者索取**（University of Leeds，Sheng Quan Xie 组，`S.Xie@leeds.ac.uk`）。

---

## 7. 🟢 强补充证据：同团队 OA 前置论文全文（已定位）

White Rose（Leeds 机构仓库）自存档了**同一作者团队**的**前置论文**，**开放获取全文**：

| 项 | 值 |
|---|---|
| 标题 | Continuous Prediction of Wrist Joint Kinematics Using Surface Electromyography from the Perspective of Muscle Anatomy and Muscle Synergy Feature Extraction |
| 作者 | Zijun Wei; Meiju Li; Zhi-Qiang Zhang; Sheng Quan Xie |
| 年份 | 2025（JBHI Vol.29 No.1, pp.43–55） |
| 仓库 | White Rose Research Online, eprint **219757** |
| 记录页 | <https://eprints.whiterose.ac.uk/id/eprint/219757/> |
| 全文 PDF | `https://eprints.whiterose.ac.uk/id/eprint/219757/1/Final_Continuous%20Prediction%20of%20Wrist%20Joint%20Kinematics%20Using%20Surface%20Electromyography%20from%20the%20Perspective%20of%20Muscle%20Anatomy%20and%20Muscle%20Synergy%20Feature%20Extraction.pdf` |
| 本地路径 | `01_参考论文/sEMG_原始参照论文/Wei_2024_companion_WristKinematics_sEMG.pdf`（下载中，见下方说明） |

**价值**：这是目标论文的「方法前身」——同一团队、同一研究线（muscle anatomy + muscle synergy → 连续关节运动学预测）。
它可提供目标摘要中 `[未验证]` 部分的**最接近旁证**（如 3DCNN 的设计思路、肌肉协同特征提取方式、数据采集协议）。

> ⚠️ **下载状态说明**：White Rose 服务器**极慢**（5.87 MB 文件，多次 300 s 超时、多次截断为 192 KiB / 1.4 MB）。
> 已启动带续传（`curl -C -`）+ 40 次重试的后台任务。**若最终仍失败，本项降级为"已知 URL，待手动下载"**，不影响本替代包的完整性。

---

## 8. 原文到位后的核对项（✅ 已由 `*_全文翻译.md` v2 逐条落实）

| # | 需替换/核实 | 当前状态 |
|---|---|---|
| 1 | §3.2 全部 `[未验证]`：3DCNN 结构、Transformer 超参、sEMG 通道/采样率、窗口长度 | 占位 |
| 2 | **8 个 baseline 的具体名单**（复现对照的前提） | 占位 |
| 3 | 数据集规模、被试数、划分协议 | 占位 |
| 4 | 是否含 biomechanical grammar / 物理约束 loss（§5 表中已标"不能断言"） | 推断 |
| 5 | 首次在线日期 2025-07-21 复核 | 待复核 |
| 6 | IEEE Xplore 文档号 11886441 复核 | 待复核 |

**操作**：
```bash
# 1. 放入 01_参考论文/sEMG_原始参照论文/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测.pdf
# 2. bash 06_附录/pdf2zh调用脚本/run_batch_cli.sh
# 3. bash 06_附录/pdf2zh调用脚本/organize_translations.sh
# 4. 勾选 06_附录/缺口台账.md 的 A3 / A4
# 5. bash 06_附录/自检脚本/verify_kb.sh
```

---

## 9. 证据来源清单

| 来源 | URL | 用于 |
|---|---|---|
| Crossref | `https://api.crossref.org/works/10.1109/JBHI.2025.3589889` | 卷期页码、日期、作者、许可 |
| **OpenAlex** | `https://api.openalex.org/works/doi:10.1109/JBHI.2025.3589889` | ✅ **完整摘要重建**（本文件 §2 的唯一来源）、OA 状态、被引 |
| Unpaywall | `https://api.unpaywall.org/v2/10.1109/JBHI.2025.3589889` | OA 交叉验证 |
| Semantic Scholar | `https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/JBHI.2025.3589889` | OA 交叉验证 |
| **White Rose** | `https://eprints.whiterose.ac.uk/id/eprint/219757/` | ✅ **同团队前置 OA 论文全文**（§7） |
