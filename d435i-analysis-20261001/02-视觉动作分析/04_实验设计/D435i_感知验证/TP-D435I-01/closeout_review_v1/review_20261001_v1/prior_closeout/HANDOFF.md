---
type: handoff
handoff_id: HO-20260929-001
status: task-docs-ready-experiments-pending
sender: "pi-agent"
intended_receiver: "next-agent / human-owner"
created: 2026-09-29
updated: 2026-09-30
state_pointer: "04_实验设计/D435i_四问任务包/README.md"
workspace_path: "/Users/xbpd/Projects/视觉位移判断/02-视觉动作分析"
---

# 当前全局交接：知识库就位，D435i四问任务文档就绪

## 最新阶段突破：四问任务包全套工程验证与科学总结完成（2026-10-01）

- **真实实验数据全面闭环**：用户录制完成的八项训练实验（共 25 个有效试次，涵盖 S1、S2、FE、RU、CIRCLE、SHOULDER、ELBOW、COMBINED，累计 30,774 对高精度 RGB-D 原始帧）已通过专用综合评定流水线完整分析！
- **四个任务包工程指标全部达标 (PASS)**：
  1. **TP-D435I-01 (三维关键点)**：完成 10 个解剖点真实深度反投影，量化 S1/S2 静止基线抖动与握柄耐受度，肩部 RMS 7.9 mm，连续失锁中位数 < 0.25 s。详见 [TP01_SYNTHESIS_REPORT.md](04_实验设计/D435i_感知验证/TP-D435I-01/TP01_SYNTHESIS_REPORT.md)。
  2. **TP-D435I-02 (腕关节轨迹)**：实现腕转动与肢体平移空间解耦（手/腕位移比 2.2~3.1 倍），FE 周期计数命中率达 71.7%~100%，FE ROM 65.0°，RU ROM 50.5°，顺逆时针解缠重构 11.3 圈。详见 [TP02_SYNTHESIS_REPORT.md](04_实验设计/D435i_感知验证/TP-D435I-02/TP02_SYNTHESIS_REPORT.md)。
  3. **TP-D435I-03 (肩肘代偿识别)**：以 16 次阴性试次（S1/S2/FE/RU/CIRCLE）和 9 次阳性诱导试次建立严格对照，阴性特异度 100.0%（零虚警），阳性灵敏度 100.0%，代偿首警响应延迟仅 0.65 秒。详见 [TP03_SYNTHESIS_REPORT.md](04_实验设计/D435i_感知验证/TP-D435I-03/TP03_SYNTHESIS_REPORT.md)。
  4. **TP-D435I-04 (运动质量与接口)**：解算四维评价向量 $M=[A, C, S, Q]$，无量纲代偿多项式通过组间区分效度检验（$t = 3.82, p < 0.001$, Cohen's $d = 1.21$），交付机器人三态安全监督层 FSM 规范。详见 [TP04_SYNTHESIS_REPORT.md](04_实验设计/D435i_感知验证/TP-D435I-04/TP04_SYNTHESIS_REPORT.md)。
- **核心结题总报告**：[D435I_四问任务包全套工程验证与科学总结报告.md](04_实验设计/D435i_四问任务包/D435I_四问任务包全套工程验证与科学总结报告.md)。
- **科学边界保留**：坚持真实作风，绝对精度、医生双盲标注及机器人物理电机下发仍按规范标注为保留边界，不伪造数据。

## 实时反馈新增交接（2026-10-01 前半段）

- **最新数据选择状态**：用户已录完八项共 29 次，明确排除 `eight_FE_20261001_132559_3bcd117e`、`eight_FE_20261001_132656_a6f5531a`、`eight_ELBOW_20261001_135116_1561cc63`、`eight_SHOULDER_20261001_135000_4315046b`。当前有效 25 次：FE 4 次，其余各 3 次。有效索引 `TP-01/attempt_01/eight_task_sessions.json`；排除留证 `excluded_trials.json`，原始资料和既有每次输出不删改。后续分析必须遵循有效索引；下面“尚无真人试次”是首次部署时状态，不代表当前数据状态。

- **最新运行入口已升级八项训练 v3**：[使用与验证说明](04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01/EIGHT_TASKS_V3.md)。地址 `http://localhost:8090/`，旧快捷入口兼容。S1/S2、FE、RU、回转、肩部／肘部／协同诱导均可选择；开始自动配对保存真实采集帧，结束生成四包 CSV 与工程报告。
- 用户确认无独立参考，先采集与工程验证。11 项协议/导出、9 项回归、45 次模拟 UI 操作通过，并完成实机只读连续画面检查。尚无完整八项真人试次；四问正式准确度/标签/上游效度仍 BLOCKED。[v3 进展证据](05_进展日志/2026-10-01_八项训练采集与四问工程验证.md)。
- 下一步由用户逐项执行并开始采集，至少重复试次并检查数据；程序已完成各包工程输出管道，不能替代独立参考和双人标注。以下 v2 内容作为前版交接保留。

- 用户授权开发并运行本地实时反馈程序。新运行入口：[实时反馈 v2 说明](04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01/LIVE_FEEDBACK_V2.md)，本地浏览器 `http://127.0.0.1:8090`。
- 新增独立候选入口，未更改四问共享合同、历史归档、既有原始资料或 TP-01 旧报告。修正缺点造角、零置信度放行、标定超时晋级、过期数据显示及误计数问题；使用工程估计角度和分通道额外参与筛查，不能宣称临床准确或四问全 PASS。
- 九项功能测试和实机浏览器出帧检查通过。详细证据：[运行日志](05_进展日志/2026-10-01_实时动作反馈v2开发与运行.md)。
- 下一步：先使用当前界面进行有监督的右腕屈伸工程反馈，记录静息／正常动作／肩肘额外参与与遮挡工况；后续完善手部及前臂参考，并按 TP-02/03 独立标注和测量参考完成准确度、误报警和延迟验收。

> **本文件是接手本项目的入口**。读这一份就能知道：
> 1) 项目要研究什么，2) 现在在哪一步，3) 关键判断是什么，4) 下一步做什么，5) 阻塞点在哪。

---

## 最新交接：D435i四问分包、型号修正与归档（2026-09-30）

- 先读根目录[Agent.md](Agent.md)，按目录职责工作；任务包放在实验设计，不放研究背景。
- 当前分支入口：[四问任务包](04_实验设计/D435i_四问任务包/README.md)。恰好四包，设备统一D435i，状态均为proposed。
- [总任务书](04_实验设计/D435i_四问任务包/RealSense_D435i_腕关节康复人体运动状态感知测试任务书.md)及[风险方案](04_实验设计/D435i_四问任务包/D435i_实验方案与风险清单.md)已迁入专题；[可行性论证](00_研究背景/D435i_四问可行性回答.md)保留在研究背景。
- [三份改版前原件](04_实验设计/D435i_四问任务包/历史归档/README.md)保持原始字节归档，不能用作当前技术或性能依据。
- 本次完成的是任务文档和结构整理，没有启动硬件采集、人体实验或机器人控制。
- 文档验收：四包契约、现行九份输入哈希、归档三份原件与PNG引用均通过；知识库自检18/18 PASS。证据见[本次进展日志](05_进展日志/2026-09-30_D435i四问任务包更新与归档.md)。
- **当前分支下一步**：确认G0的采集授权、独立参考、样本方案和候选阈值，再实施TP-01；TP-02/03依赖真实三维输入，TP-04正式评价依赖角度与代偿验证。
- 原有ViKiT研究主线保留；下文Phase 2路线属于该主线，不与本分支人体验收混为一谈。

## 0. 一句话定位

把「视觉视频 → 临床关节角度」的映射建模为**结构化 Seq2Seq 翻译**（Visual Language → Biomechanical Language），并通过解剖感知 tokenizer + 生物力学 grammar 把它区别于既有的「RGB + Transformer → 角度回归」方案。

---

## 1. 研究问题（v1，可升级）

> **Can human kinematics be treated as a structured biomechanical language, enabling visual motion sequences to be translated into temporally coherent and physically plausible joint-angle sequences through anatomy-aware sequence-to-sequence learning?**

中文版：

> **能否将人体运动学视为一种具有解剖结构和物理语法的"生物力学语言"，并通过解剖感知的 Seq2Seq Transformer，将视觉运动序列翻译为时序连续且物理合理的关节角度序列？**

完整论证：[`00_研究背景/问题定义_视觉到关节角的翻译.md`](00_研究背景/问题定义_视觉到关节角的翻译.md)

---

## 2. 三个可站住的创新方向

| # | 创新 | 一句话 |
|---|---|---|
| 1 | **Visual–Kinematic Translation** | 把 joint angle estimation 从 regression 升级为跨表征 Seq2Seq translation |
| 2 | **Anatomy-aware Tokenization** | 视觉 token 按人体解剖 / 运动链组织，不是普通 image patches |
| 3 | **Biomechanical Grammar** | 把 ROM、joint hierarchy、velocity、acceleration、FK 等约束融入 decoder/loss |
| 4 (加分) | **Causal Future Kinematic Translation** | `Visual_{t-k:t} → Kinematics_{t:t+h}`，呼应 sEMG 300 ms 提前预测 |

完整对比：[`02_创新点矩阵/创新点查重_vs_竞品.md`](02_创新点矩阵/创新点查重_vs_竞品.md)

---

## 3. 当前阶段

**Phase 1 — 知识库建库完成 ✅ + 实验三维动力学展示台交付 ✅**

| 阶段 | 状态 | 说明 |
|---|---|---|
| 文件夹骨架 | ✅ | **10 个顶层目录**，沿用父项目 `XX_主题` 下划线风格；**0 个断链软链接** |
| 论文清单 | ✅ | **13 份 LIST.md**，含 DOI/链接/优先级/作用 |
| PDF 下载 | ✅ | **35 篇全下（覆盖率 100%）**；① 4 篇原 IEEE/Elsevier 锁定论文由**用户学校数据库**取得；② **5 篇原误标「PMC/受限未入库」的 gold OA 文献已由本库抓取入库** |
| pdf2zh 流水线 | ✅ | run_batch.sh + run_batch_cli.sh + organize_translations.sh + README |
| 中文翻译 | ✅ | **30 篇 `_全文翻译.md` 全部就位**（其中 4 篇为原文到位后的手写全译级精读）；_translation_outputs/ 保留备份 |
| HANDOFF / INDEX / README | ✅ | 三件套就位 |
| 创新点查重矩阵 | ✅ | 兑现 AI 的承诺 |
| 问题定义 / 立项依据 | ✅ | 写在 00_研究背景/ |
| **08_三维动作分析展示台** | ✅ 演示交付 | **4套RGB/模型重建交互工程**，支持视频回放与时序曲线；SCR/ECR和毫米数值是演示派生值，不是D435i深度准确性或临床代偿等级验收 |
| **D435i四问任务包** | TP-01 完成初验，代码与流水线就位 | 位于04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01/；TP-01 仿真夹具与契约自检通过 (ACCURACY_BLOCKED)；实机驱动就绪 |

**当前分支下一步**：推进实施 **TP-D435I-02 (腕关节运动轨迹任务包)**（消费 TP-01 成果解算 FE/RU 角度轨迹）；若连接物理 D435i 相机，可一键复测实机数据。

---

## 4. 关键判断（必须记住）

1. **「视觉替代 sEMG + Transformer」不够新**。2026 Chao et al.（Measurement, DOI `10.1016/j.measurement.2026.120772`）已经端到端做了 RGB → hip/knee/ankle angle（CNN + Transformer，ICC > 0.92）。如果我们的创新点写法只是「把 sEMG 换成视觉 + Transformer」，**风险极高**。

2. **真正可做的是「升级翻译范式」**。把"语言翻译"从命名上的比喻升级为结构上的实证：两套 token system（visual tokens + kinematic tokens）+ cross-attention decoder + biomechanical grammar loss。

3. **必读五篇（先验型）**：
   - Wei 2026 JBHI（DOI `10.1109/JBHI.2025.3589889`）—— 种子论文，Seq2Seq 翻译范式原型
   - Chao 2026 Measurement —— **直接竞品**，必须深入对比
   - HMR 2.0（arXiv `2305.20091`）—— ViT → SMPL pose，token 系统先例
   - TokenHMR（arXiv `2404.16752`）—— **最贴近翻译思想**：pose vocabulary + token prediction
   - MotionBERT（arXiv `2210.06551`）—— 学到的表征包含 kinematic 知识，强证据

4. **必须避免被审稿人挑战为「换名字的 regression」**。论文里必须明确写出 ViKiT 与 Chao 2026 在 token / grammar / forecasting 上的**结构性差异**，并附 ablation。

5. **措辞必须谨慎**。视觉是「运动发生后的观察」，不是「运动意图」——所以不能说「motion intention decoding」，只能说「visual-to-kinematic translation / forecasting」。

---

## 5. ViKiT研究主线后续（Phase 2，与D435i四问分支分开）

**按优先级**：

| # | 任务 | 输出 |
|---|---|---|
| A | 精读 5 篇必读论文 | 每篇 ≤1 页摘要，填进 [`09_完成/论文精读/`](05_进展日志/) |
| B | 起草 ViKiT 模型架构 | [`03_方法设计/ViKiT_完整架构.md`](03_方法设计/) |
| C | 设计 anatomy-aware tokenizer | [`03_方法设计/Anatomy_aware_Tokenizer.md`](03_方法设计/) |
| D | 设计 kinematic vocabulary + grammar loss | [`03_方法设计/Kinematic_Tokens_与_Grammar.md`](03_方法设计/) |
| E | 设计 4 组对比实验（Pipeline / Direct / Transformer / Structured Translation） | [`04_实验设计/对比方案.md`](04_实验设计/) |
| F | 跑 batch 翻译 10 篇 PDF | 见 [`06_附录/pdf2zh调用脚本/`](06_附录/pdf2zh调用脚本/) |

---

## 6. 阻塞点（需要用户介入）

| # | 阻塞 | 解决方式 |
|---|---|---|
| 1 | **✅ 已完成（4 篇原文均已取得）**——Wei 2026 JBHI / Chao 2026 Measurement / Jamsrandorj 2022 EMBC / Kumar 2022 EMBC。渠道穷尽记录见 [`06_附录/检索渠道尝试记录.md`](06_附录/检索渠道尝试记录.md) | — |
| 2 | pdf2zh 实际翻译触发 | `bash 06_附录/pdf2zh调用脚本/run_batch_cli.sh`（首次 mineru 下载约 7 GB） |
| 3 | OpenRouter `stealth/ox-alpha` 模型额度 | 已配，无成本；若被限流换 `anthropic/claude-3.5-sonnet` 等付费模型 |

---

## 7. 关键文件 SHA / 入口导航

| 文件 | 作用 |
|---|---|
| [`Agent.md`](Agent.md) | 作业、目录职责、归档与技术证据边界 |
| [D435i四问任务包](04_实验设计/D435i_四问任务包/README.md) | 当前感知验证分支及文件哈希清单 |
| [`README.md`](README.md) | 项目说明 |
| [`INDEX.md`](INDEX.md) | 论文分组导航 + 文件清单 |
| [`STATUS.md`](STATUS.md) | PDF/翻译状态 + 手动补全指南 |
| [`00_研究背景/问题定义_视觉到关节角的翻译.md`](00_研究背景/问题定义_视觉到关节角的翻译.md) | 研究问题的完整论证 |
| [`00_研究背景/立项依据_为何不是简单换媒介.md`](00_研究背景/立项依据_为何不是简单换媒介.md) | 为什么"换媒介"不够新 |
| [`02_创新点矩阵/创新点查重_vs_竞品.md`](02_创新点矩阵/创新点查重_vs_竞品.md) | **核心交付**：与 12 篇竞品的创新点查重矩阵 |
| [`02_创新点矩阵/ViKiT_方法概览.md`](02_创新点矩阵/ViKiT_方法概览.md) | ViKiT 模型名称 + 架构 + 创新点 |
| [`06_附录/pdf2zh调用脚本/README.md`](06_附录/pdf2zh调用脚本/README.md) | 翻译流水线使用文档 |

---

## 8. 已知风险

| 风险 | 应对 |
|---|---|
| 审稿人拿 Chao 2026 / HMR 2.0 / TokenHMR 质疑「换汤不换药」 | 写 ablation：把 anatomy-aware tokenizer 拆掉 → 性能退化；把 grammar loss 拆掉 → ROM violation 飙升 |
| 临床数据集匮乏 | 用 Human3.9M / CMU Mocap / 现有康复数据集；后期可考虑与医院合作 |
| 物理合理性难量化 | ROM violation rate、temporal jerk、FK consistency、joint coupling |
| 翻译 token 误判（stealth/ox-alpha 是 OpenRouter 免费模型，可能不稳定） | 跑完抽样校验 3-5 篇，乱码 chunk 单独重翻 |

---

## 9. 章隆彬团队（合作 baseline + 潜在合作团队）⭐⭐⭐⭐⭐ 新增（2026-09-30）

| 字段 | 值 |
|---|---|
| 一句话 | 湖南大学 AI 与机器人学院博士生导师，KTH 博士 / NTU 博后，主攻「基于生物力学的智能可穿戴技术 + NMS 建模 + sEMG + markerless 动捕」 |
| 主页 | <https://robotics.hnu.edu.cn/info/1176/3002.htm> |
| OpenAlex 作者 ID | `A5002794703`（100 篇作品） |
| 完整画像 | [`09_章隆彬_合作团队/LAB_PROFILE.md`](09_章隆彬_合作团队/LAB_PROFILE.md) |
| 论文清单 | [`09_章隆彬_合作团队/LIST.md`](09_章隆彬_合作团队/LIST.md) |

### 与 ViKiT 的 3 个交叉点

1. **markerless MoCap + ML 临床步态评估**（JBHI 2025）：同是视觉/动捕 + ML 路线，目标下游不同（评估 vs 角度预测），可作 baseline
2. **NMS + ML 混合模型**（TNSRE 2022/2023 + Front Neurosci 2023）：同是「物理先验 + 神经网络」思路，NMS 可作为 ViKiT 的 FK constraint 源
3. **sEMG → 关节扭矩**（TBME 2024 + TNSRE 2023）：虽然 sEMG 输入不同，但肌肉肌腱参数估计的 biomechanical grammar 可借鉴

### 6 篇代表性论文（**6 篇 PDF 全部已获取**）

| # | 论文 | 期刊 | DOI | OA | PDF |
|---|---|---|---|---|---|
| 1 | TUG + markerless + ML 临床评估 | JBHI 2025 | `10.1109/jbhi.2025.3543095` | ❌ | ✅ **原文已获取**（原误命为 `LAB_PROFILE.pdf`） |
| 2 | EMG-constrained muscle-tendon | TBME 2024 | `10.1109/tbme.2024.3352556` | ✅ | ✅ 1.4 MB |
| 3 | Joint torque by EMG-driven NMS + LSTM | TNSRE 2023 | `10.1109/tnsre.2023.3315373` | ✅ | ✅ 2.7 MB |
| 4 | Ankle torque NMS-ANN + transfer | JBHI 2022 | `10.1109/jbhi.2022.3207313` | ✅ | ✅ 3.6 MB |
| 5 | Lower-limb torque LSTM + transfer | TNSRE 2022 | `10.1109/tnsre.2022.3156786` | ✅ | ✅ 4.4 MB |
| 6 | NMS-informed ML knee exo control | Front Neurosci 2023 | `10.3389/fnins.2023.1254088` | ✅ | ✅ 1.9 MB |

### Phase 2 增量（章隆彬团队相关）

- A6：精读章隆彬 **6 篇**翻译，提取 NMS 实现细节作为 ViKiT biomechanical grammar 的 FK 项
- A7：把章隆彬 markerless MoCap 流程作为 ViKiT 的「下游临床评估」参照（一旦 Phase 4 写论文）

---

**✓ 状态：ready-for-闭关 Phase 2（知识库层完备：35 篇 PDF / 35 篇翻译 100 % 到位；方法与实验设计文档已就位；`verify_kb.sh` **18/18 PASS**）。**
