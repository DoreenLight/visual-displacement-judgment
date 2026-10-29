# 基于视频的无标记动作捕捉用于临床与康复生物力学：经验证架构、临床就绪度与新兴方法的 PRISMA-ScR 范围综述

| 项 | 内容 |
|---|---|
| 方向判定 | 临床康复无标记动作捕捉（MMC）范围综述 |
| 推荐度 | ⭐⭐⭐⭐ |
| arXiv 编号 | - |
| DOI | 10.21203/rs.3.rs-10431606/v1 |
| 原文 PDF |`01_参考论文/OpenCap_传统生物力学/MarkerlessReview2021：视频无标记动捕临床康复生物力学综述/MarkerlessReview2021：视频无标记动捕临床康复生物力学综述.pdf`|
| 作者 | Florian Delaplace, Elodie Piche, Frédéric Chorin, Raphael Zory（Laboratoire Motricité Humaine Éducation Sport Santé, Université Nice / CHU Nice） |

## 一、方向判定

属于**临床生物力学 × 计算机视觉**交叉的**二次研究（范围综述）**：以 PRISMA-ScR 框架系统盘点 2015–2026 年间 117 项经光电金标准验证的视频无标记运动捕捉（MMC）流水线，划分五大架构家族，并对照 2024–2026 年 arXiv/CVPR/ICCV/ECCV 等会议的前沿 CV 模块，给出"经验现状 vs 技术前沿"的桥接图谱。

## 二、解决什么问题

回答两个研究问题：

- **RQ1（MACRO）**：当前已发表的 MMC 流水线在**架构、验证结果与临床适用性**上处于什么水平？尤其关注关节角度误差是否达到临床可接受阈值。
- **RQ2（MICRO）**：2024–2026 年 CV 文献中涌现的**技术构件**（如 foundation-model mesh recovery、可微分 IK、video-based kinetics）有哪些？哪些尚未进入同行评审的生物医学文献？

并据此提炼面向临床可部署 MMC 流水线的**循证设计指南**（覆盖单目野外 + 多目实验室两种部署模式）与前瞻性研究议程。

## 三、使用方法

- **PRISMA-ScR + JBI 框架范围综述**：方案在 OSF 前瞻性注册（doi: 10.17605/OSF.IO/TBYPC），检索 PubMed/Scopus/IEEE Xplore（2015.01–2026.02），CV 扫描更新至 2026.07。
- **双层语料设计（MACRO + MICRO）**：MACRO 为经光电/测力板/IMU 金标准验证的生物医学研究（n=117）；MICRO 为策展性、非穷举的新兴 CV 工作，仅定性使用。
- **PCC 纳入标准 + 八项排除标准（A–H）**：明确排除黑箱系统（Theia3D、KinaTrax 等）、RGB-D 主通道（Kinect）、无现代 CV、仅中间指标 MPJPE、无外部金标准等。
- **架构五家族分类法**：① 2D monocular、② 3D multi-view triangulation、③ monocular 2D→3D lifting、④ human mesh recovery / SMPL、⑤ end-to-end direct；附五个非互斥叠加变体（+IK/MSK、+marker augmentation、+temporal DL、+physics、geometry-only）。
- **LLM 辅助数据提取**：模板 7 工作表 / 90+ 字段，Claude Opus 4.7 首次填充 → Gemini 独立交叉提取 → 作者人工裁定（10,500 字段中 156 处分歧，原始一致率 98.5%；最终 385 处单元格级更正 + 20% 随机抽样复检）。
- **临床可接受度基准**：McGinley 误差区间（≤2° acceptable、2–5° reasonable、>5° potentially misleading）。
- **MACRO–MICRO 桥接分析**：针对五家族逐一标注 MICRO 中对应的新兴构建模块与差距。

## 四、核心内容概述

**结论先行**：基于视频的无标记采集**尚不能与有标记系统互换用于临床关节运动学**——矢状面下肢一致性集中在 **5°–6°**，未达临床可接受阈值；离面（额状面/水平面）运动学、动力学、老年人群与病理性人群验证严重不足。

**关键证据图谱**：

- **流程结构**：共纳入 117 项研究，约六成发表于 2024 年及以后。绝大多数研究对象为健康成人，临床/病理性人群（脑卒中、帕金森、脑瘫、脊髓损伤）仅占约 19%；样本量中位数 17；步态为主要任务（>50%）；117 项中仅 1 项在家庭/户外采集。
- **技术配置**：45% 单摄像头、29% 双摄像头；53% 未标定；近半数依赖 OpenPose/OpenCap 系列（OpenCap 默认 OpenPose 二维后端，常未报告）；33 项使用 OpenSim（其中 13 项 OpenCap 即用型流程）；仅 7% 报告实时运行。
- **验证基线**：约 83% 以光电子为金标准；多数报告**未经生物力学精修**的原始关节角度；离线后处理是常态。
- **关键缺口**：foundation-model mesh recovery、可微分逆运动学、video-based kinetics 等新兴 CV 模块几乎未在已验证研究中出现；预印本与同行评审生物医学文献之间存在结构性时滞。
- **设计指南（假设生成性，未验证）**：① 单目野外模式（病理/老年/远程康复可及性优先）；② 多目实验室模式（精度优先，可融入 differentiable IK / physics-informed 约束）。

## 五、与 ViKiT（视觉位移判断）项目的关联

| ViKiT 模块 | 本综述的启发 |
|---|---|
| **解剖感知 tokenizer** | 综述指出 MMC 流水线普遍未做解剖学先验编码（多数仅 2D keypoint → 三角化/提升），且基础模型网格恢复（SMPL 系）在已验证研究中"几乎缺席"——为 ViKiT 的解剖学 token 化提供了**空白定位**与差异化空间。 |
| **生物力学 grammar** | 综述明确点出 5–6° 矢状面误差未达 McGinley 临床阈值，根因之一是"most reported raw joint angles without biomechanical refinement"。ViKiT 将 ROM / joint hierarchy / velocity / FK 约束**融入 decoder 与 loss**，正是综述呼吁的"生物力学精修"路径。 |
| **Seq2Seq 翻译范式** | 综述五大架构家族中"端到端直接"家族与 ViKiT 的 visual language → biomechanical language 范式最契合；"2D→3D lifting"与"end-to-end"的边界正是 ViKiT 切入的角度。 |
| **sEMG 300ms 因果预测** | 综述的 MICRO 模块中"运动预测（motion forecasting）"被列为 17 个流水线组件之一，与 ViKiT 的因果时序窗口（Visual_{t-k:t} → Kinematics_{t:t+h}）直接呼应；综述提及的时序深度学习叠加变体（+temporal DL）可作为基线对比。 |
| **腕关节康复与代偿检测** | 综述的核心痛点——**离面（额状面/水平面）误差系统性地低于矢状面**，而这恰是代偿检测（肩外展代偿、腕尺偏代偿等）所依赖的平面——为 ViKiT 切入腕/手康复提供了**临床刚需论证**。 |
| **D435i RGB-D 采集** | 综述排除标准 C 明确将 RGB-D（Kinect 类）排除在主分析之外，但承认其在康复诊所仍常见；ViKiT 借 D435i 的对齐深度做腕关节尺度与遮挡鲁棒性，是相对 MACRO 主流的差异化硬件策略。 |
| **竞品 Chao 2026** | Chao 2026（RGB + Transformer 端到端，ICC>0.92）在"端到端直接"家族中；本综述给出 5–6° 的矢状面基线 + >5° 离面未验证的现状，可作为**ViKiT 上肢/腕关节子任务**对照 Chao 上肢精度的参考基准。 |

**整体判断**：本综述是 ViKiT 立题的**外部证据锚点**——既量化了竞品基线（5–6° 未达临床阈值）、也指明了 ViKiT 三大创新（解剖 token、生物力学 grammar、因果预测）所对应的**结构性缺口**。**关联度极高**，但属二次研究（综述）而非方法原研，故推荐 ⭐⭐⭐⭐ 而非满分。

## 六、阅读建议

1. **先读 Abstract + Section 1 末段 + Section 3.1**：快速建立"5–6° / 离面 / 病理人群"三大缺口与 MACRO–MICRO 双层结构的认知框架。
2. **重点读 Section 2.6（数据综合）与 Section 3.5（MACRO–MICRO 桥接）**：这是本综述对 ViKiT 最有方法论价值的部分——五家族分类法 + 17 个 MICRO 模块映射。
3. **对照库内论文**：
   - 与 **OpenCap 原研论文**（同目录其他文件）对照：OpenCap 在 MACRO 中被列为 13 项 OpenSim 即用型流程之一，ViKiT 应在文中明确"相对于 OpenCap 的临床可达性 + ViKiT 解剖 grammar"的差异。
   - 与 **Chao 2026 Measurement** 竞品论文对照：把 Chao 的 ICC>0.92 放入本综述 5–6° 矢状面基线框架中重新审视离面表现。
   - 与库内 **2D→3D lifting / mesh recovery** 类原研（如 MotionBERT、BioPose）对照：理解 MICRO 模块的现状与可借鉴构件。
4. **次读 Section 3.2–3.4**：获取 2D 检测器分布（OpenPose 49%）、标定实践（53% 未标定）、实时能力（7%）等具体数字，便于在 ViKiT 论文的 Related Work 中引用为"领域现状"。
5. **跳过 Section 2.3 排除标准细节**：与 ViKiT 系统设计无直接相关，仅需了解 A 黑箱、H 无金标准等关键条目即可。

---

*本导读由论文全文翻译自动生成，精读前请以原文 PDF 为准。*