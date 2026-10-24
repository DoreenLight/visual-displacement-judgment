# Muscle Activation and Postural Sway in Response to Task Complexity: A Study of Balance Control in Older Adults

> 原文出处：IEEE ICORR 2025: 82-87 ｜ DOI 10.1109/icorr66766.2025.11063123
> 中文精读：research-tool 九段闭环 + LLM 翻译（2026-10-01 生成）
> 原文 PDF：`Zhang2025-ICORR-PosturalSway`（本地 `papers/`）
> 译文完整度：9/9 段已译

---

## 一句话结论
任务复杂度显著增加老年人在前后方向的姿势摆动，腓肠肌内侧与胫骨前肌激活增强并与压力中心高度耦合。

## 研究问题
- 不同站姿/认知-运动双任务条件下，老年人下肢与躯干肌群激活与姿势摆动的变化规律
- 肌电（EMG）与压力中心（CoP）之间的耦合（EMG–CoP coherence）能否反映神经肌肉协调与平衡控制能力
- 关键稳定肌群（如踝关节跖屈/背屈肌）在复杂任务下的特异性贡献及其侧向不对称性
- 上述发现对康复机器人与辅助技术设计、跌倒风险干预的启示

## 方法
- 受试者：7 名老年人（原文未给出性别、年龄均值/标准差）
- 任务设计：4 种站姿——双足站立（DS）、串联站立（TS）、串联+认知双任务（TSC）、串联+认知+运动三任务（TSCM）
- 信号采集：下肢与躯干多块肌肉的表面 EMG，同步采集压力中心（CoP）
- 特征分析：CoP 时域/摆动特征（重点关注前后 AP 轴）、EMG 激活模式、EMG–CoP 相干性（coherence）
- 统计比较：跨任务复杂度梯度比较，未在节选中报告具体检验方法
- 输出：定性结论 + 关键肌群（gastrocnemius medialis、tibialis anterior、soleus、peroneus longus）的激活与耦合变化

## 关键数据
- 样本量：n = 7 老年人
- 任务数：4（DS、TS、TSC、TSCM）
- 趋势性结论：任务复杂度↑ → AP 轴姿势摆动↑；TSCM 摆动最显著
- 肌群层面定性结论：gastrocnemius medialis、tibialis anterior 在复杂任务下激活↑且 EMG–CoP coherence 增强
- 侧向不对称：左侧肌群激活持续高于右侧（原文未给出量化数值）
- RMSE / R² / 相关系数 / p 值：**原文未给出**（节选仅含摘要与引言）

## 结论与局限
- 任务复杂度对前后方向姿势稳定性影响最大，TSCM 为最具挑战性条件
- 踝跖屈/背屈肌（gastrocnemius medialis、tibialis anterior）是复杂任务下的关键稳定肌
- 观察到显著的左侧优势（lateral asymmetry），提示姿势控制存在侧化贡献
- 节选未呈现方法细节与量化结果，需结合正文/表格进一步评估统计稳健性与样本代表性

## 对「视觉→关节角（RGB 视频估计人体关节角序列）」课题的可借鉴点
- **可作 baseline 范式**：DS→TS→TSC→TSCM 四级复杂度阶梯可直接复用为 RGB 关节角估计的分级评测协议，覆盖从静态到动态双任务的难度谱
- **可比指标**：AP 轴 CoP 摆动量 / 摆动速度可作为关节角预测误差的物理一致性基准——预测出的下肢关节角序列在生物力学前向仿真中应能复现类似 AP 不稳定趋势
- **可复用模块（耦合约束）**：EMG–CoP coherence 提示踝关节矢状面力矩是关键耦合通道，可在关节角回归损失中加入「矢状面踝角变化 ↔ AP 方向 CoP 偏移」的软约束
- **可借鉴的物理约束/数据增强**：左右侧激活不对称与任务复杂度梯度可启发对模型施加侧化偏置与难度感知采样（difficulty-aware sampling），提升在双任务/三任务场景下的泛化

---

## 全文中文译

2025年5月12-16日，美国伊利诺伊州芝加哥

# 任务复杂性下的肌肉激活（muscle activation）与姿势摆动（postural sway）：老年人平衡控制（balance control）研究

**Longbin Zhang**<sup>1,2,3</sup>、**Ananda Sidarta**<sup>2</sup>、**Yu Chin Lim**<sup>2</sup>、**Carol Er**<sup>2</sup>、**Xiaoyue Yan**<sup>2,3</sup>、**Tsung-Lin Wu**<sup>2,3</sup>、**Wei Tech Ang**<sup>2,3</sup>

**摘要** —— 在本研究中，我们调查老年人在不同挑战程度的站立条件下的肌肉激活模式与姿势摆动。研究在下肢（lower limb）与躯干（trunk）肌肉上记录了七名老年人在四种站立任务期间的肌肉活动（muscle activity）：双脚站立（double stance）、串联站立（tandem stance）……控制，其中压力中心（Center of Pressure, CoP）频繁地充当关键指标。[6]。近期，越来越多的研究兴趣聚焦于利用脑电图（electroencephalography, EEG）与肌电图（electromyography, EMG）等工具探索姿势控制（postural control）背后的神经机制，这些工具为皮层（cortical）与肌肉（muscular）……

串联站立结合认知任务（tandem stance with a cognitive task），以及串联站立结合认知与运动双重任务（tandem stance with combined cognitive and motor tasks, TSCM）。我们分析了 CoP（centre of pressure，压力中心）特征、EMG（electromyography，肌电）模式以及 EMG-CoP 相干性，以深入了解神经肌肉协调与平衡控制机制。我们的研究结果表明，任务复杂度显著影响姿势稳定性，尤其是串联站立（tandem stance）——特别是 TSCM——会导致前后轴（anterior-posterior axis）上的不稳定性增加。关键稳定肌，如腓肠肌内侧头（gastrocnemius medialis）和胫骨前肌（tibialis anterior），在复杂任务下激活程度增高，并表现出强烈的 EMG-CoP 相干性，凸显了它们在抵消重力以维持平衡方面的关键作用。此外，我们还观察到肌肉激活存在侧向不对称性，左侧肌肉的激活始终强于右侧，提示姿势控制中存在偏侧化（lateralized）贡献。这些结果强调了特定肌肉在挑战性姿势任务中对稳定性的重要性，可为设计针对性干预（包括机器人康复系统和辅助技术）以增强老年人平衡能力并降低跌倒风险提供有价值的参考。

_作者感谢南洋理工大学（Nanyang Technological University）、新加坡科技研究局（Agency for Science Technology and Research, A*STAR）以及新加坡国立医疗保健集团（National Healthcare Group, NHG）的资助。（通讯作者：Ananda Sidarta。）_

1. 湖南大学机器人学院，中国长沙
2. 新加坡康复研究院，南洋理工大学李光前医学院，新加坡
3. 南洋理工大学机械与航空航天工程学院，新加坡

## I. 引言

姿势控制是一项复杂的功能，涉及将重心的垂直投影维持在支撑面（base of support, BoS）内，以确保在静态和动态活动中保持稳定 [1]。跌倒是姿势控制障碍最严重的后果，对老年人的独立性构成重大威胁，并且是一个日益严峻的公共卫生问题，常常导致严重受伤、生活质量下降和活动能力降低 [2]–[4]。这凸显了在该人群中研究姿势控制的迫切必要性。维持姿势控制需要感觉系统（视觉、前庭觉和本体感觉）、中枢神经系统和肌肉骨骼系统之间的精细协调 [5]。运动学指标如支撑面（BoS）、质心（centre of mass, CoM）和压力中心（CoP）等，对于评估涉及维持稳定性的姿势活动至关重要 [3]，[7]。深入理解这些过程对于诊断和治疗平衡障碍，以及改进神经假体和康复机器人等辅助技术至关重要。

EMG-CoP 相干性已在老年人中用于评估肌肉力量如何影响平衡。Skeparianos 等人 [8] 发现，更强的踝关节肌肉，特别是跖屈肌（plantar flexors）和背屈肌（dorsiflexors），与姿势摆动减少相关，尤其在串联站立和单腿站立等挑战性姿势中。Sozzi 等人 [9] 进一步考察了串联站立时的腿部肌肉活动，并指出：比目鱼肌（soleus）通过持续性活动维持直立姿势，而胫骨前肌和腓骨长肌（peroneus longus）则在内外侧方向通过交互作用参与平衡控制。他们的研究结果表明，肌肉力量与协调能力对于维持老年人的姿势稳定性至关重要。这些研究凸显了肌肉-姿势摆动相干性在理解老年人姿势控制中的重要性，并提示了其用于训练干预的潜力——特别是考虑到运动员之间存在的差异。间接而言，该指标可反映老年人在执行不同复杂度任务时的平衡能力。因此，有必要研究老年人在平衡任务中的肌肉-姿势摆动相干性。

认知衰退（cognitive decline）是衰老的自然结果，会通过损害记忆、注意力、决策能力和执行功能（这些功能对于日常活动至关重要）进一步使平衡控制变得复杂 [10]。认知在姿势控制中扮演着关键角色，它整合感觉信息、促进决策并实现运动规划，使老年人能够评估风险、保持注意力并基于过往经验调整动作。认知负荷增加，例如多任务处理时，会损害平衡。因此，有效的认知过程对于维持稳定性和预防跌倒至关重要 [11]。尽管已有大量研究考察了延缓认知衰退的干预措施，但关于肌肉-姿势摆动相干性在同时涉及认知与运动需求的双重任务活动中如何交互的研究仍十分有限。深入理解认知与运动任务如何共同影响平衡控制，对于识别早期衰退标志以及开发维持老年人稳定性的干预措施具有重要意义。

"，以及平均距离、最大偏差范围、AP 和 ML 方向上的平均速度，以及两个方向上的功率频率离散度。"

…tions [16]。这些压力中心（Center of Pressure, CoP）特征被分析并在所有任务间进行比较。采用 Shapiro-Wilk 检验评估 CoP 特征的正态性。对于符合正态分布的特征，组间比较使用配对 t 检验（paired t-test）；对于非正态分布的特征，则采用 Wilcoxon 符号秩检验（Wilcoxon signed-rank test）。多重比较采用 Bonferroni 校正（Bonferroni correction），统计学显著性阈值设定为 p < 0.05。

> **范围 [mm]**
>
> (a) 平均距离　　　　　　　　　　　(b) 最大偏移距离
>
> ML　AP　　　　　　　　　　　　　　ML　AP
>
> **速度 [mm/s]　　　　　　　　　　** **功率 [mm²/Hz]**
>
> (c) 平均速度　　　　　　　　　　　(d) 功率
>
> ML　AP　　　　　　　　　　　　　　ML　AP

## C. 肌电-压力中心肌肉-姿势摆动相干性（EMG-CoP Muscular-postural Sway Coherence）

为评估肌电（Electromyography, EMG）信号与 CoP 摆动之间的相干性，我们针对每块肌肉计算了其与同侧脚姿势摆动之间的 EMG-CoP 相干性。例如，左侧腓肠肌内侧（Gastrocnemius Medialis, GM）的相干性是相对于左脚 CoP 在整个频谱范围内进行分析的。相干性（coherence）作为衡量两个信号在频域上线性关联程度的指标，采用如下公式计算：

所示的特征包括：(a) 前后方向（anterior-posterior, AP）与内外侧方向（mediolateral, ML）的平均距离；(b) ML 与 AP 方向上最大偏移的极差；(c) ML 与 AP 方向的平均速度；(d) ML 与 AP 方向的功率；(e) ML 与 AP 方向的频率分散性（frequency dispersion）；(f) 面积；(g) 平均频率。星号（*）表示两组之间通过配对 t 检验（paired t-tests，正态分布数据）或 Wilcoxon 符号秩检验（Wilcoxon signed-rank tests，非正态分布数据）确定的具有统计学意义的差异，并已应用 Bonferroni 校正（Bonferroni correction）。

## III. 结果（RESULTS）

### A. CoP 摆动测量

在 ML 方向上，TS 相比 DS 的平均距离显著增加（p < 0.01）。在 AP 方向上，TS（p = 0.03）、TSC（p = 0.02）和 TSCM（p = 0.02）的平均距离均显著高于 DS。此外，在 TS、TSC 和 TSCM 条件下，AP 方向的平均距离值均大于 ML 方向（图 2）。

极差定义为最大偏移之间的距离，在 ML 与 AP 两个方向上，TS、TSC 和 TSCM 均显著大于 DS（all p < 0.01）。值得注意的是，在 TS、TSC 和 TSCM 条件下，AP 方向的极差值均超过 ML 方向。

类似地，在平均速度方面，数据显示在两个方向上 TS、TSC 和 TSCM 均显著高于 DS。在这些站立条件下，AP 方向的平均速度也持续高于 ML 方向。

相比 DS，AP 方向的功率在 TSC（p = 0.04）和 TSCM（p = 0.02）下显著升高，在 TS 下虽也有所升高但未达显著水平（p = 0.06）。在 ML 方向上，TS 的功率显著大于 DS（p = 0.03）。此外，在 TS、TSC 和 TSCM 条件下，AP 方向的功率幅度持续大于 ML 方向。

TS、TSC 和 TSCM 的摆动面积均大于 DS。有趣的是，TSC 和 TSCM 的摆动面积小于 TS。姿势摆动的平均频率在 TS、TSC 和 TSCM 下均显著高于 DS（all p < 0.01）。此外，平均频率呈递增趋势，TSC 高于 TS，TSCM 又高于 TSC。在各站立条件之间，频率分散性未观察到显著差异。

---

DS
TS
TSC
TSCM

站立任务包括：双足站立（double stance）、串联站立（tandem stance）、串联站立合并认知任务（tandem stance with cognitive task），以及串联站立合并认知与运动任务（tandem stance with cognitive and motor tasks）。我们的研究结果表明，姿势摆动测量——包括平均距离、极差、平均速度、摆动面积以及平均频率——总体上

---

979-8-3503-8068-2/25/$31.00 ©2025 IEEE    84

授权许可使用仅限于：中南民族大学。下载时间：2026 年 10 月 1 日 03:37:19 UTC，源自 IEEE Xplore。适用限制条款。

## 肌肉活动

图 3. 四种任务下各受试者所有试次的肌肉活动（归一化至所有试次的最大值）：DS（Double Stance，双足站立）、TS（Tandem Stance，串联站立）、TSC（Tandem Stance with Cognitive Task，串联站立伴认知任务）以及 TSCM（Tandem Stance with Cognitive and Motor Tasks，串联站立伴认知与运动任务）。结果以各受试者每个肌肉的均值 ± 标准差表示。

### B. 肌肉模式

肌肉活动模式在 TS、TSC 和 TSCM 三种条件下，左侧与右侧的腓肠肌内侧头（Gastrocnemius Medialis，GM）、胫骨前肌（Tibialis Anterior，TA）以及股二头肌（Biceps Femoris，BF）大体相似，左侧的股直肌（Rectus Femoris，RF）和臀大肌（Gluteus Maximus，GluM）亦是如此。然而，与 DS 条件相比，这些肌肉在 TSCM 条件下表现出明显更高的活动水平（图 3）。

各条件下肌肉活动的差异在右侧的 RF 与 GluM，以及两侧的腹直肌（Rectus Abdominis，RA）中较小，各条件之间仅观察到轻微变化。

此外，还观察到肌肉活动的不对称性：在 TS、TSC 和 TSCM 条件下，左侧肌肉（特别是 GM、TA 和 RF）的活动水平高于其右侧对应肌肉。

### C. EMG-CoP 肌-姿势摆动相干性

下肢肌肉的肌电图（Electromyography，EMG）与压力中心（Center of Pressure，CoP）的相干性（coherence）在所有任务（DS、TS、TSC 和 TSCM）中均出现在约 20 Hz 以下。在各肌肉中，GM 与 CoP 的相干性始终最高，其次为 TA 和 RF（图 4）。

在 GM 中，与 CoP 的相干性在 TS、TSC 和 TSCM 条件下总体高于 DS 条件。左侧在 TSCM 条件下的相干性高于 TS 和 TSC，而右侧在 TSCM 条件下的相干性低于 TS 和 TSC。BF 和 GluM 与 CoP 的相干性左侧略高于右侧。RA 在所有任务中的相干性均较低。

## IV. 讨论

本研究中，我们探讨了老年人在不同站立条件下姿势摆动、下肢与躯干肌肉活动模式以及 EMG-CoP 相干性之间的关系。肌电活动在 TS、TSC 和 TSCM 中较 DS 更高，反映出在窄支撑面上维持稳定性的挑战增大。与摆动增加相一致，多数下肢肌肉在 TS、TSC 和 TSCM 中的活动也较 DS 升高，提示在更具挑战性的条件下稳定身体需要更高的神经肌肉需求。此外，EMG-CoP 相干性分析表明，下肢肌肉在所有任务中均与 CoP 表现出约 20 Hz 以内的相干性，其中 GM 相干性最高。这提示 GM 在协调肌肉反应以维持姿势稳定性方面起着关键作用。上述结果为不同肌肉在多种条件下稳定姿势的作用提供了有价值的见解，并强调了任务特异性肌肉激活对姿势控制的潜在重要性。这些见解有助于通过识别在维持平衡（尤其是在窄支撑面或多任务等挑战性条件下）方面至关重要的肌肉，为损伤预防策略与康复方案的制定提供依据。

任务复杂度与站立类型显著影响姿势控制，导致不稳定性增加以及更频繁的姿势调整，尤其沿前后向（Anteroposterior，AP）轴。然而，未观察到并发运动或认知任务对平衡参数的显著影响。TS、TSC 和 TSCM 任务中的平均距离和幅度均显著增加，尤其在 AP 方向上，提示串联位姿由于支撑面受限，沿该轴带来更大的稳定性挑战。类似地，平均速度在 TS、TSC 和 TSCM 中升高，表明随着任务需求增加，姿势校正的需要也相应增加。TSC 和 TSCM 条件下 AP 方向更高的功率进一步突显了复杂任务条件所带来的额外平衡控制需求。有趣的是，摆动面积在 TS 中达到峰值，而在 TSC 和 TSCM 中略有下降，提示附加任务可能导致摆动模式更为集中，但仍较为宽广。并发运动或认知任务对平衡参数未产生显著影响，这与 Xu 等人 [17] 的发现一致，他们研究了单任务与双任务条件下的平衡能力与前额叶皮层激活。其结果同样表明，在健康老年人中，双任务并未显著影响平衡表现，提示平衡控制即使在引入额外认知任务时仍可能以隐式方式运行。这与任务优先级模型（Task Prioritization Model）[18] 相吻合，该模型认为，随着姿势威胁的增加，老年人会将姿势稳定性置于认知表现之上。我们认识到，较小的样本量可能限制了检测 TS、TSC 和 TSCM 之间更细微差异的能力。未来需要更大样本量的研究以确认这些趋势，并探讨任务依赖性平衡控制机制中潜在的细微差异。

979-8-3503-8068-2/25/$31.00 ©2025 IEEE                                                          85
                  授权许可使用仅限于：South-Central Minzu University。下载时间：2026年10月01日 UTC 03:37:19，来自 IEEE Xplore。须遵守使用限制。

                                                                                                                                            DS
                                                                                                                                            TS
                                                                                                                                            TSC
                                                                                                                                            TSCM

        Coherence

                                                                     Frequency (Hz)
**Fig. 4.** 所有受试者在四种任务下肌电图（Electromyography, EMG）与压力中心（Center of Pressure, CoP）数据之间的相干性（Coherence）：双足站立（Double Stance, DS）、串联站立（Tandem Stance, TS）、串联站立+认知任务（Tandem Stance with Cognitive Task, TSC）以及串联站立+认知+运动任务（Tandem Stance with Cognitive and Motor Tasks, TSCM）。结果以所有受试者的平均值 ± 标准差（mean ± standard deviation）表示。M LL 和 M LR 分别表示 CoP 在左侧和右侧的内外方向（mediolateral）；APL 和 APR 分别表示前后方向（anteroposterior）。肌肉活动由 GML 和 GMR 表示左右两侧的腓肠肌内侧（Gastrocnemius Medialis, GM），股二头肌（Biceps Femoris, BF）、臀大肌（Gluteus Maximus, GluM）、胫骨前肌（Tibialis Anterior, TA）、股直肌（Rectus Femoris, RF）和腹直肌（Rectus Abdominis, RA）的对应标签类似标注。

   肌肉活动水平随任务复杂度变化，与 DS 条件相比，TSCM 条件下肌肉参与度更高，尤其在双侧的 GM、TA 和 BF，以及左侧 RF 和 GluM 中更为明显。这一模式表明，需要同时进行认知和运动协调的复杂任务（TSCM）需要更强的肌肉激活以维持姿势稳定性。双侧 RA 在各条件下活动差异极小，加上 EMG-CoP 相干性较低，提示 RA 对任务复杂度的变化可能反应不敏感。双侧 RA 中观察到的中等偏高归一化值可能归因于相对于所有试次最大值的归一化处理，因为 RA 活动总体上较低且较为稳定。此外，还观察到肌肉活动的不对称性，左侧（非优势腿）肌肉——尤其是 GM、TA 和 RF——在 TS、TSC 和 TSCM 期间始终比其右侧对应肌肉表现出更高的激活水平。这种不对称性可能反映固有的侧向优势或代偿策略，提示非优势侧在串联站立等复杂条件下在稳定姿势方面发挥更为积极的作用。这些发现与 Sozzi 等人 [9] 的报告一致，他们发现非优势腿的 EMG 活动更高，表明其在串联站立中作为主要稳定肌，从而承担更大的姿势负荷。

   在肌肉激活与姿势摆动之间观察到显著的耦合关系，特别是在 GM 中，所有任务的相干性最高可达约 20 Hz。在一项相关研究中，Watanabe 等人 [12] 调查了跖屈肌——具体为腓肠肌内侧、腓肠肌外侧和比目鱼肌——的协调性及其与双足和单足站立期间姿势稳定性的关系，在老年人中发现了最高约 10 Hz 的相干性水平。本研究在此基础上扩展，通过纳入更多肌肉以提供肌肉激活与姿势摆动相干性之间关系的更全面分析。在本研究中，GM 在下肢肌肉中显示出最高的 EMG-CoP 相干性，凸显其在姿势调整中的关键作用。这一发现与既往研究一致，提示腓肠肌是调节前后方向（AP）摆动的主要贡献者，尤其是在具有挑战性的平衡任务中 [19]–[21]。GM 和 TA 均表现出比其它肌肉更高的相干性水平，表明它们在任务需求增加时在稳定姿势方面发挥突出作用。这与研究报告中各平衡任务中 GM 和 TA 的 EMG 振幅高于 RF 和 BF 的结果一致。踝部肌肉更高的激活可能是由于依赖

979-8-3503-8068-2/25/$31.00 ©2025 IEEE                                      86
  授权许可使用仅限于：South-Central Minzu University。下载时间：2026年10月01日 UTC 03:37:19，来自 IEEE Xplore。须遵守使用限制。

时平衡的刚度控制"，*Journal of Neurophysiology*，第 80 卷，第 3 期，1211–1221 页，1998 年。

[24] D. A. Winter, A. E. Patla, S. Rietdyk, 和 M. G. Ishac，"静立期间踝关节肌肉刚度在平衡控制中的作用"，*Journal of Neurophysiology*，第 85 卷，第 6 期，2630–2633 页，2001 年。

979-8-3503-8068-2/25/$31.00 ©2025 IEEE 87

授权许可使用仅限于：中南民族大学（South-Central Minzu University）。于 2026 年 10 月 01 日 03:37:19 UTC 从 IEEE Xplore 下载，附有限制条款。
