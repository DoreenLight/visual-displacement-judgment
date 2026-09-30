# Wei et al. 2026 JBHI — 内容速读卡

> ## 🔔 注意：本速读卡已被原文级精读取代
>
> 本速读卡的 (6.2%/95.5%)、(5.5%/96.2%)、300 ms、<2 min 已核实为准确；但 8 个基线名单等细节原文未给。
>
> | 文件 | 权威度 |
> |---|---|
> | [`Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测.pdf`](Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测.pdf) | ✅ **原文 PDF（权威）** |
> | [`Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测_全文翻译.md`](Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测_全文翻译.md) | ✅ **基于原文的全译级精读 —— 请优先阅读这个** |
> | **本速读卡** | ⚠️ **速查用**；与上述文件冲突时，**一律以后者为准** |

---


> ✅ **状态**：**PDF 已获取**（原文已落盘于同目录）。本卡的元数据与
> 推测数值**已被原文推翻部分**——请以文件顶部的「原文级精读」为准。
> 本文档基于 OpenAlex 公开摘要（147 entries inverted index）+ Crossref 元数据 + 作者团队前期工作推导。
> **作用**：速查卡（原文已于 2026-09-30 获取）。**精读请以文件顶部的「原文级精读」为准**。
>
> **下载后请把 PDF 改名为 `Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测.pdf` 放入本目录**，跑 `bash 06_附录/pdf2zh调用脚本/run_batch_cli.sh` 自动翻译归位。

---

## 1. 论文元数据

| 字段 | 值 |
|---|---|
| 标题 | A Transformer Framework Informed by Muscle Anatomy and Sequence-to-Sequence Translation for Continuous Joint Kinematics Prediction Using sEMG |
| 作者 | Zijun Wei, Zhiqiang Zhang, Sheng Quan Xie（顺序） |
| 机构 | University of Leeds, UK |
| 期刊 | IEEE Journal of Biomedical and Health Informatics (JBHI) |
| 卷期 | Vol. 30, No. 2 (2026)，pp. 1140–1152 |
| DOI | `10.1109/JBHI.2025.3589889` |
| IEEE Xplore | <https://ieeexplore.ieee.org/document/11886441/> |
| PubMed | <https://pubmed.ncbi.nlm.nih.gov/40690346/> |
| 在线发表 | 2025-07-21 |

**作者团队前期工作**（用户已下载的同团队论文）：
- 2025 JBHI vol.29 no.1: "Continuous Prediction of Wrist Joint Kinematics Using Surface Electromyography from the Perspective of Muscle Anatomy and Muscle Synergy Feature Extraction"（Wei 等，eprint 219757 已下载但损坏）— **本论文的前置工作**，3DCNN 提取肌肉协同特征

---

## 2. 核心命题

> "We, for the first time, redefine motion intention prediction from the perspective of sequence-to-sequence translation by analogizing sEMG signals and joint angles to the source language and target language, respectively."

把 **sEMG → joint angle** 的映射重新定义为 **sequence-to-sequence translation** 任务：
- **sEMG = source language**
- **joint angle = target language**
- 这是论文最重要的"命名"贡献——把肌肉电信号到运动学表征的映射视为「翻译」问题。

---

## 3. 模型架构：3DCNN-TF

```
┌─────────────────────────────────────────────────────────────┐
│  输入：sEMG 序列（多通道滑动窗口）                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  模块 1：sEMG "Sentence" Generation                        │
│     把多个滑动窗口（words）组成 sentence                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  模块 2：3DCNN（基于 muscle anatomy + electrode placement）│
│     从每个 word 提取肌肉协同特征                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  模块 3：Transformer (TF) Decoder                          │
│     自回归生成下一时刻关节角度                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  输出：joint angle（wrist / knee 屈伸角度）                │
└─────────────────────────────────────────────────────────────┘
```

**与原 sEMG 版相关的命名映射**：
| 我们方案 | Wei 2026 的命名 |
|---|---|
| 视觉帧序列 | sEMG sliding window (word) |
| 连续帧序列 | sEMG sentence（多个 word 串起来）|
| anatomy-aware visual tokens | muscle-anatomy-informed 3DCNN features |
| Spatio-Temporal Transformer | Transformer (TF) module |
| 关节角预测 | joint angle（自回归生成下一时刻）|
| anatomy-aware tokenization | muscle-anatomy-informed representation |

---

## 4. 关键性能

| 数据集 | 任务 | nRMSE | R² | 训练时间 |
|---|---|---|---|---|
| Wrist 数据集 | 屈伸角度连续预测 | 6.2% | 95.5% | <2 min |
| Knee 数据集 | 屈伸角度连续预测 | 5.5% | 96.2% | <2 min |

**特别亮点**：
- **300 ms 提前预测**：不损失精度
- **训练时间 <2 分钟**（高效）
- **跨速度鲁棒**（varying speeds）
- **baseline 8 个模型 + 现有研究对比，全面领先**

---

## 5. 与我们方案 (ViKiT) 的对照

| 维度 | Wei 2026 (sEMG 版) | ViKiT (我们，视觉版) |
|---|---|---|
| 输入信号 | sEMG 多通道 | RGB 视频 |
| 命名框架 | Seq2Seq translation | 升级：Visual Language → Kinematic Language |
| Token 系统 | 隐式（3DCNN 特征）| **显式**：anatomy-aware visual tokens + kinematic tokens |
| 时间建模 | Transformer（自回归）| Spatio-Temporal Transformer + Cross-Attention |
| 物理约束 loss | 未明确说明 | **6 项 grammar**：angle / velocity / acceleration / ROM / FK / temporal |
| 任务 | 同步 + 300ms 提前 | 同步 + 300ms kinematic forecasting |
| 评估数据集 | 自采 wrist/knee 数据集 | Human3.9M / CMU Mocap / 公开康复数据 |

**关键差异（论文写作卖点）**：
1. **显式 token 系统**：他们隐式，**我们显式**
2. **Biomechanical grammar**：他们未明确，我们 6 项
3. **视觉版的 forecasting**：他们做 sEMG，我们做视觉

---

## 6. 必读章节（拿到 PDF 后）

按优先级：
1. §3 (Methodology) — 3DCNN-TF 的三个模块如何拼装
2. §4 (Experiments) — 数据集 + baseline + 性能表
3. §5 (Discussion) — 与 Seq2Seq NLP 的类比是否站得住
4. §1 (Introduction) — "Seq2Seq translation" 命名的论证

---

## 7. 我们的 Related Work 必须引用

文中已提及的关键概念（用于做"我们和它不一样"的对比）：
- "source language = sEMG, target language = joint angle" 的命名映射
- "muscle anatomy informed" 的 3D CNN 设计
- "Transformer (TF) module" 自回归解码
- 8 个 baseline + 现有研究（这 8 个我们也要尽量复现）

---

## 已获取原文（本节的「手动下载指引」已失效，留作历史记录）

> ✅ **原文已落盘**：`Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测.pdf`（同目录）。
> 中文精读已产出，**无需再执行任何下载步骤**。

---

## 9. 元数据来源

- 标题 / 作者 / DOI / 期刊 / 发表日期：Crossref API + OpenAlex API + Semantic Scholar
- 摘要：OpenAlex `abstract_inverted_index`（147 entries）重建
- 全文：**已获取**（原文 PDF + 全译级精读，见同目录）

---

**✅ 原文已获取（2026-09-30）**：PDF 与中文精读均已就位，本卡保留作速查用途。
