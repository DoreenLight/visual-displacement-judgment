# D4 AAN辅助控制与介入决策

> 8 篇 ｜ ✅ 已入库 8 ｜ 需人工取件 0
> 本文件由 `build_jev.py` 生成。数字口径见 `../index.json`。

| # | 年 | 论文题名 | 载体 | 链接 | 证据等级 | 编者结论 | 本地 |
|---|---|---|---|---|---|---|---|
| 1 | 2017 | Effects of Assist-As-Needed Upper Extremity Robotic Therapy after Incomplete Spinal Cord Injury: A Parallel-Group Controlled Trial | Neural Engineering and Rehabilitation（PMC5469353，开放全文） | [DOI](https://pmc.ncbi.nlm.nih.gov/articles/PMC5469353/) | 第三方实测 | 条件可行 | ✅ |
| 2 | 2023 | Assistance level quantification-based human-robot interaction space reshaping for rehabilitation training | Frontiers in Neurorobotics 17: 1161007（开放获取） | [DOI](https://journal.frontiersin.org/article/10.3389/fnbot.2023.1161007) | 第三方实测 | 可行主路线 | ✅ |
| 3 | 2023 | Performance-based assistance control for robot-mediated upper-limbs rehabilitation | Mechatronics 89: 102919 | [DOI](https://doi.org/10.1016/j.mechatronics.2022.102919) | 官方一手 | 关键工程参照：介入时机用跟踪误差而非意图预测 | ✅ |
| 4 | 2024 | Effects of an assist-as-needed equipped Tenodesis-Induced-Grip Exoskeleton Robot (TIGER) on upper limb function in patients with chronic stroke | Journal of NeuroEngineering and Rehabilitation 21: 3（开放获取，PMC38173006） | [DOI](https://doi.org/10.1186/s12984-023-01298-2) | 第三方实测 | 反面基线：加装 AAN 反而不如原型机 | ✅ |
| 5 | 2026 | When to Act: Calibrated Confidence for Reliable Human Intention Prediction in Assistive Robotics | arXiv 预印本（cs.RO） | [2601.04982](https://arxiv.org/abs/2601.04982) | 第三方实测 | 可行主路线 | ✅ |
| 6 | 2026 | A Human-in-the-Loop Confidence-Aware Failure Recovery Framework for Modular Robot Policies | ACM/IEEE HRI 2026, pp. 346-355（DOI 10.1145/3757279.3788668） | [2602.10289](https://arxiv.org/abs/2602.10289) | 第三方实测 | 可行主路线 | ✅ |
| 7 | 2026 | Model Predictive Control-Based Assist-as-Needed Strategy for Reducing Motor Slacking in Robot-Assisted Rehabilitation | Sensors (MDPI) 26(9): 2740（Gold OA CC-BY） | [DOI](https://www.mdpi.com/1424-8220/26/9/2740) | 第三方实测 | 可行主路线 | ✅ |
| 8 | 2026 | Velocity field assist-as-needed controller for upper limb rehabilitation with sEMG-based spasticity detection | Mechatronics 114: 103450 | [DOI](https://www.sciencedirect.com/science/article/pii/s095741582500159x) | 第三方实测 | 可行主路线 | ✅ |

---

### Effects of Assist-As-Needed Upper Extremity Robotic Therapy after Incomplete Spinal Cord Injury: A Parallel-Group Controlled Trial（2017，Neural Engineering and Rehabilitation（PMC5469353，开放全文））

- **作者**：Frullo, John Michael、Elinger, Jared、Pehlivan, Ali Utku、Fitle, Kyle
- **来源**：<https://pmc.ncbi.nlm.nih.gov/articles/PMC5469353/>
- **关键数字（A）**：17 例不完全脊髓损伤（AIS C/D）平行组平衡对照试验：主要终点 ARAT 评分前后差值**两组均无统计学显著变化**；次要终点显示两组动作质量均渐进改善，但 AAN 控制器较 subject-triggered 控制器带来更大的动作质量提升
- **证据等级**：第三方实测
- **编者结论**：条件可行
- **本地工件**：PDF 全文已在库：`01_参考论文/决策模型线_JEV/D4_AAN辅助控制与介入决策/Effects of Assist-As-Needed Upper Extremity Robotic Therapy.pdf`
- **备注**：★★ **这是本语料最该被记住的一篇「泼冷水」文献。**
它用 RCT 级别的方法学给出：**AAN 控制器在主要功能终点上并未显著优于简单的主动触发式固定辅助。**
作者自陈是 stage-two 概念验证研究、未证明功能获益。
**对本项目的含义**：介入决策层不能只报「我的 AAN 更好」，必须预先定义主要终点并诚实报告阴性结果。
另注：作者之一 Fabrizio Sergi 亦为 Neurorobotics 那篇的共同作者。

### Assistance level quantification-based human-robot interaction space reshaping for rehabilitation training（2023，Frontiers in Neurorobotics 17: 1161007（开放获取））

- **作者**：Li, Xiangyun、Chen, Pengcheng、Gong, Shan、Yu, Xi
- **来源**：<https://journal.frontiersin.org/article/10.3389/fnbot.2023.1161007>
- **关键数字（A）**：按恢复阶段设计七套实验协议区分训练进程；用 PSO-SVM 分类模型 + LSTM-KF 回归模型，从 sEMG 与运动学数据识别患者运动能力，驱动 AAN 控制与交互空间重塑
- **证据等级**：第三方实测
- **编者结论**：可行主路线
- **本地工件**：PDF 全文已在库：`01_参考论文/决策模型线_JEV/D4_AAN辅助控制与介入决策/Assistance level quantification-based human-robot interactio.pdf`
- **备注**：**已有人把 sEMG 接到 AAN 决策上**，用的正是本项目关心的 sEMG 通路。
其局限：SVM/KF 不是校准过的概率，分类器输出不能直接当安全阈值用 —— 这正是 D4 主张「必须自己校准」的现实佐证。

### Performance-based assistance control for robot-mediated upper-limbs rehabilitation（2023，Mechatronics 89: 102919）

- **作者**：Leigang Zhang、Shuai Guo、Fengfeng Xi
- **来源**：<https://doi.org/10.1016/j.mechatronics.2022.102919>
- **关键数字（A（同行评审，Elsevier 期刊））**：三模式由**轨迹跟踪误差**切换：ZIF（零交互力，偷懒时给固定 1 N）/ AAN（按误差调刚度系数，5 名被试实测 1.3–1.4 N）/ RIR（限制交互区，出安全区时给 12.7–26.7 N 大力）。双 KUKA LBR iiwa7 末端执行器式，7 DOF + 1 冗余轴。仅 5 名健康被试（男，均龄 26.6）。
- **证据等级**：官方一手
- **编者结论**：关键工程参照：介入时机用跟踪误差而非意图预测
- **本地工件**：PDF 全文已在库：`01_参考论文/决策模型线_JEV/D4_AAN辅助控制与介入决策/Performance-based assistance control for robot-mediated uppe.pdf`
- **备注**：**这是 AAN 介入时机最直接的工程实现，且揭示了一个关键事实：现行控制律用的是「表现（performance）」而非「意图（intent）」。**

机制：单控制器覆盖三个康复阶段，靠跟踪误差这一单一标量做模式切换；作者自陈参数需按被试个性化调整。

**对本项目的意义**：
1. R1 想做的「介入时机判断」，工程上已有朴素基线 —— 阈值判据。**任何复杂决策模型都必须先打败它**，否则只是把一个 if-else 换成神经网络。
2. 它只用**单一标量误差**，不预测意图、不建模阶段 —— 留出的正是 R1 的空间。
3. 但也给出反面提醒：5 名健康被试、无临床终点，作者明说结论仅为 preliminary。
**这是工程可行性证据，不是临床有效性证据**，不能与 Kuo 2024 的临床结论并列引用。
4. 印证了 R1 的边界：介入决策是控制问题，不是表征学习问题 —— JEV 式语言决策模型在这里没有输入接口。

### Effects of an assist-as-needed equipped Tenodesis-Induced-Grip Exoskeleton Robot (TIGER) on upper limb function in patients with chronic stroke（2024，Journal of NeuroEngineering and Rehabilitation 21: 3（开放获取，PMC38173006））

- **作者**：Li-Chieh Kuo 等（成功大学物理医学与复健科）
- **来源**：<https://doi.org/10.1186/s12984-023-01298-2>
- **关键数字（A（同行评审 + 已在 ClinicalTrials.gov 注册））**：16 例慢性期卒中完成全程。FMA-UE 总分前后 34.6→38.5（+3.9），组内显著（GEE p<0.05）；BBT 5.9→9.5（p=0.004）；AOU 0.35→0.48（p=0.02）。**但与原型机组相比，FMA-UE 改善反而更差（GEE p=0.008）。**试验注册 NCT03713476。
- **证据等级**：第三方实测
- **编者结论**：反面基线：加装 AAN 反而不如原型机
- **本地工件**：XML 全文已在库：`01_参考论文/决策模型线_JEV/D4_AAN辅助控制与介入决策/Effects of an assist-as-needed equipped Tenodesis-Induced-Gr.xml`
- **备注**：⚠️ **2026-10-02 核实：原「FMA-UE 多改善 4.5 分」的二手引述方向有误，已按原始研究改写。**
本文是**单臂**前瞻性干预研究（无平行对照组），16 例慢性期卒中患者完成 9 周训练。
组内前后对比看起来是阳性（+3.9 分），但作者自己与「原始 TIGER」的既有数据做了跨研究比较，结论是 **AAN 版并未优于原型机，反而更差（p=0.008）**。
对照组来自 Kuo 等 2021 的单盲 RCT（Disability and Rehabilitation 44(23):7078–7086，DOI 10.1080/09638288.2021.1980915，NCT03713476 同源注册号）：34 例入组、32 例完成，TIGER 组 FMA-UE 总分 ANCOVA p=0.006、腕 p=0.037、手 p=0.006 —— 那是**原型机本身有效**的证据。

**对本项目的意义（比原记录重要得多）**：
1. 「AAN 能改善 FMA-UE」这个前提**不成立**。原型机有效 ≠ AAN 模式有效。
2. 原记录把「+4.5 分」当成 AAN 的收益，实际是跨研究比较口径不严产生的幻影。
3. 这与 Frullo 2017（AAN vs 常规训练，ARAT 两组均无显著变化）**方向一致**，两个独立 RCT 体系都指向「AAN 的临床增益未被证实」。
4. 因此 R1 不能承诺「更好的 AAN 决策 → 更好的临床结局」，而应把主要终点定在**决策层自身的可测指标**（介入时机准确率、误报率、协助剂量、患者主观负担），并把功能量表作为次要终点。

**[2026-10-02 全文已取得]**：经 Europe PMC 官方 OA 通道（PMC10765635）取回 JATS 全文 XML（111 KB），
原文逐字核验通过：FMA-UE 34.6±11.5→38.5±13.4、single-arm prospective interventional study、
`the original TIGER exhibited greater improvements ... (GEE, p = 0.008)`、9 周训练、BBT 5.9→9.5 (p=0.004)、
AOU 0.35→0.48 (p=0.02)、注册号 NCT03713476。
**本文所有数字已从「摘要级」升级为「全文级」核验。**
（出版商 Springer 直链返回 Client Challenge 挑战页，未绕过；改走 Europe PMC 官方聚合通道。）

### When to Act: Calibrated Confidence for Reliable Human Intention Prediction in Assistive Robotics（2026，arXiv 预印本（cs.RO））

- **作者**：Gaus, Johannes A.、Ilg, Winfried、Haeufle, Daniel
- **来源**：<https://arxiv.org/abs/2601.04982>
- **关键数字（A）**：EGTEA Gaze+ 数据集：Temperature Scaling / Platt Scaling / Isotonic Regression 把 ECE 从 **≈0.40 降到 0.04**（约一个数量级）且不损失准确率。校准置信度驱动 ACT/HOLD 规则：仅在可靠性高时动作，否则拒绝介入。多模态 GRU + 后验校准 + 滞回安全门。文中明确处理震颤/共济失调等病理信号与主动运动的区分
- **证据等级**：第三方实测
- **编者结论**：可行主路线
- **本地工件**：PDF 全文已在库：`01_参考论文/决策模型线_JEV/D4_AAN辅助控制与介入决策/When to Act_ Calibrated Confidence for Reliable Human Intent.pdf`
- **备注**：★★ **与本项目新方向几乎完全对应的论文，标题就是「何时行动」。**
核心主张：「错误的辅助比没有辅助更糟」（Wrong assistance can be worse than no assistance）。
**它把置信度阈值变成可解释、可调节的定量安全参数**，这正是 AAN 决策层缺的东西。
注意其自陈局限：需真实用户闭环验证；校准在病理运动等域偏移下的稳定性仍需检验 —— **对卒中患者恰好就是域偏移**。

### A Human-in-the-Loop Confidence-Aware Failure Recovery Framework for Modular Robot Policies（2026，ACM/IEEE HRI 2026, pp. 346-355（DOI 10.1145/3757279.3788668））

- **作者**：Banerjee, Rohan、Palempalli, Krishna、Yang, Bohan、Fang, Jiaying
- **来源**：<https://arxiv.org/abs/2602.10289>
- **关键数字（A）**：在机器人辅助进食系统上部署，真实行动障碍用户家庭研究：提升失效恢复成功率同时降低用户查询负荷。两个决策分离：模块选择器（该问哪个模块）+ 查询算法（是否该问人）
- **证据等级**：第三方实测
- **编者结论**：可行主路线
- **本地工件**：PDF 全文已在库：`01_参考论文/决策模型线_JEV/D4_AAN辅助控制与介入决策/A Human-in-the-Loop Confidence-Aware Failure Recovery Framew.pdf`
- **备注**：★★ **本语料证据等级最高的一篇**（同行评审会议论文，非预印本）。
**关键设计是「两个决策分离」**：问哪个模块 vs 是否要问人。
迁移到 AAN：把「是否介入」与「介入哪个自由度/哪个方向」拆成两个独立校准决策 —— 这比单阈值更贴合上肢腕-肘-肩的多关节结构。

### Model Predictive Control-Based Assist-as-Needed Strategy for Reducing Motor Slacking in Robot-Assisted Rehabilitation（2026，Sensors (MDPI) 26(9): 2740（Gold OA CC-BY））

- **作者**：Kim, Choonggun、Moon, Youngjin、Choi, Jaesoon
- **来源**：<https://www.mdpi.com/1424-8220/26/9/2740>
- **关键数字（A）**：参与比从 0.103（不足贡献）升到 0.879（恰当贡献）；在人为将任务方向贡献压到 45% 的偷懒条件下，本文控制器的整体人类贡献比 **0.282**，显著低于误差型 AAN 基线 0.595 与遗忘因子基线 0.535
- **证据等级**：第三方实测
- **编者结论**：可行主路线
- **本地工件**：PDF 全文已在库：`01_参考论文/决策模型线_JEV/D4_AAN辅助控制与介入决策/Model Predictive Control-Based Assist-as-Needed Strategy for.pdf`
- **备注**：★ **指出了误差型 AAN 的结构性缺陷**：助力和用户主动贡献被单一误差系数耦合，「主动出力减少被辅助通道吸收，伪装成小误差」。
**这正是「用置信度而非误差做介入决策」的论据** —— 误差可以伪装，参与度不能。
双通道导纳 + 凸 MPC + 能量罐保证无源性。

### Velocity field assist-as-needed controller for upper limb rehabilitation with sEMG-based spasticity detection（2026，Mechatronics 114: 103450）

- **作者**：Li, Ye、Song, Aiguo、Lai, Jianwei、Lu, Ye
- **来源**：<https://www.sciencedirect.com/science/article/pii/s095741582500159x>
- **关键数字（A）**：变导纳速度场 AAN + LSTM 网络建模肘部痉挛模式；健康被试与卒中患者在平面上肢康复机器人上验证，轨迹跟踪精度与 AAN 表现优于传统力场控制器与阻抗控制器
- **证据等级**：第三方实测
- **编者结论**：可行主路线
- **本地工件**：PDF 全文已在库：`01_参考论文/决策模型线_JEV/D4_AAN辅助控制与介入决策/Velocity field assist-as-needed controller for upper limb re.pdf`
- **备注**：**「什么时候介入」不只有一层**：除了「患者能力不足要帮」，还有「检测到痉挛要减速保护」。
这支持把介入决策拆成**多个原子问题**（能力不足 / 疲劳 / 痉挛 / 代偿出现），每个独立校准 —— 正是 JEV 式原语该干的事。
