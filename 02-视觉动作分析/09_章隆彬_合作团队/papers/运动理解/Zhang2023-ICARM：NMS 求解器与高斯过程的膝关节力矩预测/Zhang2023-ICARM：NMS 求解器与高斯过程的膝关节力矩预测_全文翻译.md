# Knee Joint Torque Prediction with Uncertainties by a Neuromusculoskeletal Solver-informed Gaussian Process Model

> 原文出处：IEEE ICARM 2023: 1035-1040 ｜ DOI 10.1109/icarm58088.2023.10218934
> 中文精读：research-tool 九段闭环 + LLM 翻译（2026-10-01 生成）
> 原文 PDF：`Zhang2023-ICARM-KneeTorqueGP`（本地 `papers/`）
> 译文完整度：8/8 段已译

---

## 一句话结论
NMS-GP 模型融合神经肌肉骨骼求解器与高斯过程，以 EMG 与关节运动学为输入，在多种日常活动中显著降低膝关节力矩预测误差，并给出预测不确定度。

## 研究问题
- 如何在获得准确膝关节屈伸力矩预测的同时量化预测不确定度，以提升外骨骼人机接口安全性？
- 如何将物理 NMS 模型的可解释性与机器学习的拟合能力结合，弥补各自短板？
- 在不同日常活动（慢走、自选速度走、快走、坐起、起坐）中，预测误差与不确定度如何随运动相位变化？

## 方法
- 模型：Neuromusculoskeletal Solver-informed Gaussian Process（NMS-GP），把 NMS 求解器的输出特征整合进 GP 模型，兼具物理可解释性与数据驱动拟合能力
- 输入：肌电（EMG）信号 + 关节运动学（joint kinematics）
- 输出：膝关节屈伸力矩预测值 + GP 预测方差（不确定度）
- 数据：5 类日常活动 — slow walking、self-selected speed walking、fast walking、sit-to-stand、stand-to-sit
- 参考力矩：由逆动力学（inverse dynamics）计算得到的实测力矩
- 评估：对比 NMS-GP 与 NMS 模型的预测误差，并分析不确定度在运动各相位的分布

## 关键数据
- 具体样本量、RMSE/R²、相关系数、p 值等数字：原文未给出（摘要未披露）
- 定性结论：NMS-GP 预测误差显著低于 NMS 模型（原文表述为"significantly lower"）
- 不确定度高发区段：每个运动的起始阶段；自选速度走步态的 terminal stance 与 swing 相位
- 比较基线：仅 NMS 模型；未与纯 GP 或其他机器学习方法做并列定量对比（原文表述）

## 结论与局限
- NMS-GP 在保持 NMS 物理意义的同时显著降低力矩预测误差，并能输出不确定度，适用于康复外骨骼"AAN"控制策略设计
- 运动起始阶段以及自选速度走的支撑末/摆动相预测不确定度偏高，控制策略需对这些时段做安全处理
- 局限（作者自陈）：机器学习方法在训练分布外数据上可能不可靠；摘要未披露被试数、传感器配置、跨被试泛化能力等细节

## 对「视觉→关节角（RGB 视频估计人体关节角序列）」课题的可借鉴点
- 不确定度输出：可复用 GP（或 BNN、Deep Ensemble）作为基线或模块，为 RGB→关节角预测附带置信区间，对下游机器人/外骨骼控制的安全关键决策至关重要
- 物理约束嵌入：借鉴"solver-informed"思路，将生物力学先验（关节角范围、角速度限幅、运动链约束、地面反作用力一致性）作为软约束或额外特征融入深度回归网络，弥补纯数据驱动模型的外推短板
- 可比指标：除 RMSE/平均绝对误差外，建议补充分相位误差（对应本文的 stance/swing 子组分析）、预测区间覆盖率（PI coverage），用于评估困难时段可靠性
- 三层 baseline 设计：可设定"运动学/解剖学解析模型 baseline（对应 NMS）+ 纯数据驱动 baseline（对应 GP/LSTM）+ 物理约束融合模型（对应 NMS-GP）"三层比较范式，清晰展示融合策略的增量价值

---

## 全文中文译

2023 年第 8 届先进机器人与机电一体化国际会议
2023 年 7 月 8–10 日，中国三亚

基于神经肌肉骨骼求解器驱动的高斯过程模型的膝关节力矩不确定性预测
Longbin Zhang、Xiaochen Zhang、Xueyu Zhu、Ruoli Wang 和 Elena M. Gutierrez-Farewik

摘要—结合用户力矩能力的外骨骼辅助策略的研究兴趣正在迅速，以及何处需要它，从而提升患者执行特定任务和动作的能力，并已证明

```markdown
2023 International Conference on Advanced Robotics and Mechatronics (ICARM) | 979-8-3503-0017-8/23/$31.00 ©2023 IEEE | DOI: 10.1109/ICARM58088.2023.10218934

…不断增长，但预测力矩能力中的不确定性会显著影响用户—外骨骼界面的安全性。在本文中，我们使用神经肌肉骨骼（NMS）求解器指导的高斯过程（GP）模型（以下简称 NMS-GP 模型），以肌肉肌电信号和关节运动学作为模型输入，估计膝关节屈/伸力矩。NMS-GP 模型通过将 NMS 求解器中的有价值特征整合到 GP 模型中，结合了 NMS 和 GP 模型。NMS-GP 模型被用于在日常活动中预测膝关节力矩，并进行不确定性量化。活动包括慢速行走、自选速度行走、快速行走、坐起和起坐。分析了 NMS-GP 模型与 NMS 模型的性能，性能定义为模型预测力矩与基于逆向动力学计算所得测量力矩之间的低预测误差。我们发现，NMS-GP 模型的预测误差显著低于 NMS 模型。我们观察到，在每个运动开始时，特别是在自选速度行走中，预测的膝关节力矩存在相对较高的不确定性。在自选速度行走的支撑末期和摆动期也发现了高不确定性。与其他力矩预测方法相比，所提出的 NMS-GP 模型不仅提供了准确的关节力矩预测，还提供了不确定性度量。我们的研究表明，NMS-GP 模型在康复外骨骼的控制策略设计以及增强整体用户体验方面具有巨大潜力。

## I. 引言

机器外骨骼越来越多地用于运动障碍患者的康复，例如脑卒中、脑瘫和脊髓损伤等病症[1]–[7]。这些病症常导致肌肉无力、协调障碍和活动受限，使个人难以进行日常活动。机器外骨骼可在康复训练期间提供帮助，使运动障碍患者能够进行日常活动和/或完成否则具有挑战性或不可能完成的锻炼[8]–[10]。在外骨骼辅助康复训练期间，将用户的生物信号，例如肌电图（EMG），纳入控制策略已引起广泛关注，因为它能主动使用户参与，从而促进运动再学习[11]，[12]。这种类型的康复训练策略，称为按需辅助控制策略，旨在仅在[…]时提供辅助，具有改善康复效果的潜力[13]–[16]。一种常见的按需辅助策略从 EMG 信号估计用户的生理关节力矩，并将其用作控制器输入。

基于 EMG 驱动的神经肌肉骨骼（NMS）模型方法和机器学习是基于 EMG 传感器肌肉激发信号预测用户生理关节力矩的两种流行技术[17]–[19]。通常，NMS 模型需要领域专业知识来建立各种变量之间的关系，例如肌肉激发与激活、肌肉力与长度、肌肉力与速度，以及肌肉—肌腱运动学，并且可能涉及复杂的校准程序[19]–[22]。相反，机器学习无需显式描述变量之间的复杂关系即可估计关节力矩。然而，机器学习通常被视为一种黑箱方法，因为它不提供对物理系统底层机制的理解。这可能使其在预测训练分布之外的数据时变得不可靠，尤其是在训练数据有限的情况下[23]。基于物理的 NMS 方法和机器学习方法各有优缺点。我们最近提出了一种混合模型，将 NMS 求解器的物理特征与标准人工神经网络（ANN）相结合[24]，以提高单独使用 ANN 预测步态和等速运动期间踝关节力矩的精度。我们发现，与 NMS 模型和 ANN 模型相比，混合模型具有更优的性能，从而突出了将 NMS 模型中的信息性特征整合到 ANN 中的优势。然而，这一发现存在一个例外；我们观察到，我们的混合 ANN 模型在自选速度行走期间表现差于标准 ANN。这可能是因为来自 NMS 模型的纳入输入特征具有误导性，原因可能是校准和测试动作之间的肌肉协调模式差异或 EMG 输入信号中的不确定性。因此，对模型预测的不确定性进行量化十分重要，特别是当其用于外骨骼控制时。

高斯过程（GPs）已被证明是在各种应用中做出不确定性预测的强大工具，例如关节运动学预测[25]、天气预报[26]和机器人运动规划[27]。GP 是一种概率模型，可以提供其预测中不确定性的度量[28]。Liang 等人
```

---

\*本工作得到 Promobilia 基金会（A22078、18014、18200 和 21302）、瑞典研究委员会（2018-00750 和 2018-04902）以及 Simons 基金会（504054）的资助。

L. Zhang、X. Zhang、R. Wang 和 E.M. Gutierrez-Farewik 就职于瑞典斯德哥尔摩 KTH 皇家理工学院工程力学系 KTH MoveAbility 实验室（电子邮箱：lanie@kth.se）。

X. Zhu 就职于美国爱荷华州爱荷华市爱荷华大学数学系。

979-8-3503-0017-8/23/$31.00 ©2023 IEEE                                    1035

[25] 提出了一种 GP 模型，根据 EMG（Electromyography）信号预测行走和跑步时的膝关节角度（Joint Angle）。他们证明了 GP 模型在预测膝关节角度时具有较高精度，并能提供不确定性信息。同样地，Yun 等人 [29] 应用 GP 模型，仅以身体参数作为输入来预测人体步态模式。他们发现 GP 模型是一种成本效益高且具有个体特异性的步态运动学估算方法。GP 模型有助于全面理解人体测量参数与步态运动学之间的相关性及其不确定性。据我们所知，高斯过程（Gaussian Process, GP）尚未被用于以实测 EMG 信号作为输入来预测膝关节力矩。然而，这似乎是一种用于量化关节力矩预测相关不确定性的有前景的技术。

   我们的研究目的是，使用一种融合了实测肌电图信号和膝关节角力矩作为模型输入的 NMS 求解器驱动的 NMS 求解器驱动的 GP 模型（NMS-GP, Neuromusculoskeletal Solver-informed Gaussian Process），估计不同日常运动中的膝关节屈曲／伸展（F/E, Flexion/Extension）力矩。我们假设 NMS-GP 模型在膝关节力矩预测方面比单独的 NMS 模型更为准确。第二个研究目的是量化该模型在不同运动中预测膝关节力矩的不确定性。

                            II. 方法（METHODS）
**A. 数据采集与处理（Data Collection and Processing）**

   本研究招募了八名健康受试者（性别：4 女／4 男；身高：168.1±9.4 cm；体重：65.2±17.8 kg；年龄：29±4 岁）。瑞典伦理审查委员会批准了本研究（Dnr. 2020-02311），所有受试者均提供了书面知情同意书。所有受试者均佩戴三维运动采集传感器，并被要求执行五种不同的日常运动，具体包括慢速行走、按自选步速的正常行走、快速行走、坐起立转换和站立坐下转换。每种记录的运动至少重复十次。

   表面肌电图（Surface EMG）信号（Myon aktos/Cometa 系统，意大利米兰）以 1000 Hz 频率采集自随机选定的腿的股直肌（RF, Rectus Femoris）、股内侧肌（VM, Vastus Medialis）、股外侧肌（VL, Vastus Lateralis）、股二头肌（BF, Biceps Femoris）、半腱肌（ST, Semitendinosus）、腓肠肌内侧头（GM, Gastrocnemius Medialis）和腓肠肌外侧头（GL, Gastrocnemius Lateralis）。原始 EMG 信号的后处理流程如下：首先进行带通滤波（30–300 Hz），然后整流，再进行低通滤波（6 Hz），最后对所有运动试验中的最大 EMG 值进行归一化处理 [2]、[20]、[30]。

   使用三维运动捕捉系统（100 Hz，V16，Vicon，英国牛津）记录标记点轨迹，标记点放置遵循 CGM2.3 模型 [31]。使用三块测力板（1000 Hz，AMTI，美国马萨诸塞州）测量地面反作用力。运动学（关节角度）通过逆运动学（IK, Inverse Kinematics）方法获得，该方法通过最小化建模标记轨迹与实测标记轨迹之间的差异实现。逆动力学（ID, Inverse Dynamics）方法用于计算满足运动方程的关节力矩 [32]、[33]。使用零相位四阶巴特沃斯低通滤波器（6 Hz）对关节运动学和动力学进行滤波 [34]–[36]。

**B. NMS 求解器驱动的高斯过程回归模型（Neuromusculoskeletal Solver-informed Gaussian Process Regression Model）**

   在本研究中，我们使用由 Pizzolato 等人 [21] 开发的开源 EMG 驱动的 NMS 模型，该模型由四个元素组成（图 1）：肌肉激活动力学（muscle activation dynamics）、肌腱运动学（musculotendon kinematics）、肌肉收缩动力学（muscle contraction dynamics）和关节动力学（joint dynamics）[37]。肌肉激活动力学元素根据 EMG 信号（肌肉激励）计算肌肉激活。肌腱运动学元素计算肌腱长度和力臂（moment arms）。肌肉收缩动力学元素使用 Hill 型肌肉模型（Hill-type Muscle Model）计算肌腱力（muscle forces），其输入为肌肉激活和肌腱长度。最后，关节动力学元素以肌腱力和力臂为输入，计算关节力矩。

   NMS 驱动的 GP 模型输入 **x** = [x₁, x₂, . . . , xₙ]ᵀ（n = 21）通过两类特征进行增强（图 1）：（1）实验测量的肌肉信号和三维关节运动学（kinematics）；

---

**图 1 说明（Fig. 1）**：NMS 模型结构示意图。肌肉激活动力学元素根据 EMG 信号（肌肉激励）计算肌肉激活。肌腱运动学元素计算肌腱长度和力臂。然后使用肌肉收缩动力学元素和 Hill 型肌肉模型计算肌腱力，其输入为肌肉激活和肌腱长度。最后，关节动力学元素以肌腱力和力臂为输入计算关节力矩。

**图 2 说明（Fig. 2）**：使用 NMS 驱动的 GP 模型进行关节力矩估计的概述。NMS 驱动的 GP 模型的输入通过两类特征进行增强：（1）实验测量的肌肉信号和三维关节运动学；（2）来自底层 NMS 求解器的信息性物理特征，具体为肌力和 NMS 力矩。

…以及来自三维运动捕捉系统的关节运动学数据，以及（2）来自底层神经肌肉骨骼求解器的信息性物理特征——肌肉力与基于神经肌肉骨骼（Neuromusculoskeletal, NMS）模型的关节力矩。这些特征通过提供关于所建模系统的更多信息，被用于提升模型的性能表现。NMS-GP 模型 $f(\mathbf{x})$ 由均值函数 $\mu(\mathbf{x})$ 和协方差函数 $k(\mathbf{x}, \mathbf{x}')$ 表征，如式 (1) [38] 所示。

$$
f(\mathbf{x}) \sim \mathcal{GP}\left(\mu(\mathbf{x}),\, k(\mathbf{x}, \mathbf{x}')\right) \tag{1}
$$

均值函数 $\mu(\mathbf{x})$ 用于估计给定输入处函数的期望值，可由式 (2) 定义：

$$
\mu(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})] \tag{2}
$$

协方差函数（亦称核函数）表示两个输入变量之间的相似性，用于估计预测的不确定性，由式 (3) [39] 定义：

$$
k(\mathbf{x}, \mathbf{x}') = \mathbb{E}\left[(f(\mathbf{x}) - \mu(\mathbf{x}))(f(\mathbf{x}') - \mu(\mathbf{x}'))\right] \tag{3}
$$

高斯过程模型中有多种核函数可供选择。最常用的函数之一是径向基函数核（Radial Basis Function, RBF），它是一个光滑且无限可微的函数，能够很好地捕捉数据的潜在结构，由式 (4) 定义：

$$
k(\mathbf{x}, \mathbf{x}') = \exp\left(-\frac{|\mathbf{x} - \mathbf{x}'|^2}{2l^2}\right) \tag{4}
$$

其中 $l$ 是控制核函数长度尺度的超参数，$|\mathbf{x} - \mathbf{x}'|$ 为两个输入 $\mathbf{x}$ 与 $\mathbf{x}'$ 之间的欧氏距离。

另一个常用的核函数是 Matern 核，它是 RBF 核的推广，由式 (5) 定义：

$$
k(\mathbf{x}, \mathbf{x}') = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}}{l}\right)^{\nu} |\mathbf{x} - \mathbf{x}'|^{\nu} K_{\nu}\!\left(\frac{\sqrt{2\nu}}{l}\,|\mathbf{x} - \mathbf{x}'|\right) \tag{5}
$$

其中 $\nu$ 为决定核函数光滑性的正参数，$K_{\nu}$ 为修正贝塞尔函数，$\Gamma(\nu)$ 为伽马函数，$l$ 为控制核函数尺度的长度尺度参数。

白噪声核（White Noise Kernel）是另一种常用核，此处用于建模数据中的噪声，如式 (6) 所示：

$$
k(\mathbf{x}, \mathbf{x}') = \sigma^{2}\, \delta(\mathbf{x} - \mathbf{x}') \tag{6}
$$

其中 $\sigma^{2}$ 为控制噪声幅度的噪声方差参数，$\delta(\mathbf{x} - \mathbf{x}')$ 为狄拉克 $\delta$ 函数（Dirac delta function），当 $\mathbf{x} = \mathbf{x}'$ 时其值为 1，否则为 0。它确保核函数仅在输入空间的对角线（即 $\mathbf{x} = \mathbf{x}'$ 处）非零。

线性核（Linear Kernel）是另一种常用核，用于建模输入与输出之间的线性关系，如式 (7) 所示：

$$
k(\mathbf{x}, \mathbf{x}') = \sigma^{2}\, \mathbf{x}^{T}\mathbf{x}' \tag{7}
$$

其中 $\sigma^{2}$ 为方差参数。

实践中，通常将不同的核函数组合以提升高斯过程模型的性能。这被称为核函数组合（kernel composition），可通过将不同核函数相加或相乘实现。该方法使模型能够捕捉数据中各种潜在结构，提升预测的整体精度。在对多种核函数进行试验后，我们基于试验测试在 GP 模型中采用了 Matern 核、白噪声核与线性核的组合；该组合针对本数据集给出了最精确的预测结果。其中，Matern 核使我们能够捕捉数据中任何潜在的非线性模式，白噪声核有助于解释测量误差或其他不确定性来源，而线性核则有助于建模输入与输出之间的线性关系。

高斯过程回归在 Python 中完成，采用 scikit-learn 包中高斯过程模块的 `GaussianProcessRegressor` 库及核函数实现。在训练过程中，对 GP 模型相关的超参数（包括长度尺度、信号方差、噪声方差等）进行调整以优化其性能。具体而言，在 Matern 核中，正参数 $\nu$ 被设为 $3/2$；长度尺度的取值范围为 $(0.01,\, 200)$，方差取值范围为 $(10^{-3},\, 10^{5})$。在白噪声核中，噪声方差取值范围为 $(0.03,\, 100)$；在线性核中，方差取值范围为 $(10^{-3},\, 10^{5})$。

### C. 评估协议（Evaluation Protocol）

我们基于 NMS 和 NMS-GP 模型对膝关节力矩预测精度进行了评估——此处定义为预测力矩与逆向动力学（inverse dynamics）所得测量力矩之间的误差，并分析了 NMS-GP 模型的不确定性量化结果。带不确定性量化的预测精度在五种情形下进行了评估：Gaitslow、Gaitself、Gaitfast、SitToSta 与 StaToSit；其中为每种运动分别训练一个独立的 NMS-GP 模型。在所有情形下，NMS-GP 模型使用 80% 的记录数据进行训练，并使用剩余数据进行后续评估；而 NMS 模型使用每种运动的三个试次进行标定，并在与 NMS-GP 测试数据相同的试次上进行测试。

每个模型的力矩估计误差通过归一化均方根误差（Normalized Root Mean Square Error, NRMSE）$E_{NRMS}$ 来评估，较低的估计误差意味着较高的预测精度。NRMSE 定义为均方根误差 $E_{RMS}$（预测力矩与测量力矩之间）除以逆向动力学在相应运动中观测到的关节力矩的极差：

$$
E_{RMS} = \sqrt{\frac{1}{N}\sum_{n=1}^{N}(y_{p,n} - y_{n})^{2}} \tag{8}
$$

$$
E_{NRMS} = \frac{E_{RMS}}{y_{max} - y_{min}} \times 100\% \tag{9}
$$

---

**图（Figure）** 膝关节 F/E 预测误差（Prediction Error of Knee F/E）

| NMS | NMS-GP |
|:---:|:---:|
| \* | \* |

（左图：NMS、NMS-GP；右图：NMS、NMS-GP）

```markdown
                                              *                                                                                  *
                                                          *                                                                              *           *
                                                                  *

        力矩，归一化均方根误差（NRMSE, %）                                                                       力矩，均方根误差（RMSE, %）

                                                                                       图 4. 使用 NMS 与 NMS-GP 模型在受试者间预测膝关节
 图 3. 使用 NMS 与 NMS-GP 模型在受试者间预测膝关节力矩的归一化均方根误差（Normalized Root Mean Square Error，NRMSE）。小提琴图（violin plots）通过核密度图展示了 NRMSE 的概率分布，箱线图（box plots）展示了 NRMSE 在五种日常活动中的最小值、下四分位数、中位数、上四分位数和最大值。星号（*）表示两种模型之间存在显著差异。
                                                                                       力矩的均方根误差（Root Mean Square Error，RMSE）。小提琴图通过核密度图展示了 RMSE 的概率分布，箱线图展示了在五种日常活动中 RMSE 的最小值、下四分位数、中位数、上四分位数和最大值。星号（*）表示两种模型之间存在显著差异。

其中 $y_n$ 与 $y_{p,n}$ 分别为测量（实际）力矩与预测力矩；$y_{\text{min}}$ 与 $y_{\text{max}}$ 分别为相应运动中的最小与最大测量力矩。计算每位受试者的 NRMSE，再确定八名参与者间的数值中位数。采用 Shapiro-Wilk 检验评估数据分布的正态性（$p < 0.05$ 显著性水平）。随后对两种方法所得力矩估计的 NRMSE 进行成对比较（pairwise comparisons），对服从正态分布的数据采用配对 t 检验（paired t-tests），对非正态分布数据采用 Wilcoxon 符号秩检验（Wilcoxon signed-rank tests）（$p < 0.05$）。

使用 95% 置信水平（Confidence Level，CL）对预测力矩的不确定性进行量化，表示被建模函数的真实值有 95% 的概率落在预测区间内。不确定性值较高表示对预测值的置信度较低。

## III. 结 果（RESULTS）

总体而言，与 NMS 模型相比，NMS-GP 模型的膝关节力矩预测 NRMSE 显著更低（$Gait_{slow}$：$p < 0.01$，$Gait_{self}$：$p < 0.01$，$Gait_{fast}$：$p < 0.01$，$SitToSta$：$p < 0.01$，以及 $StaToSit$：$p = 0.03$；图 3）。在 RMSE 中也观察到类似的结果（$Gait_{slow}$：$p < 0.01$，$Gait_{self}$：$p < 0.01$，$Gait_{fast}$：$p < 0.01$，$SitToSta$：$p < 0.01$，以及 $StaToSit$：$p = 0.04$；图 4）。与 NMS-GP 模型相比，NMS 模型预测的力矩与测量力矩的一致性较差（图 5(b)），且在每个运动开始时存在相对较大的偏差。

在五种运动中，使用 NMS 模型时，步态运动的膝关节力矩预测 NRMSE（$Gait_{slow}$：$\leq 18.3\%$，$Gait_{self}$：$\leq 17.5\%$，$Gait_{fast}$：$\leq 14.2\%$）高于其他运动（$SitToSta$：$\leq 9.7\%$ 与 $StaToSit$：$\leq 7.2\%$）（图 3）。NMS-GP 模型预测的 NRMSE 在全部五种运动中普遍较低（$\leq 5.9\%$）。在 RMSE 中也观察到类似的趋势（图 4）。

值得注意的是，在每个运动开始时，预测膝关节力矩的不确定性相对较高，尤其是在自选速度步态中（图 5(a)）。在自选速度步态的支撑末期（terminal stance）与摆动相（swing）中也观察到较高的不确定性。

## IV. 讨论（DISCUSSION）

我们提出了一种 NMS-GP 模型，利用测量的肌肉肌电（electromyography，EMG）信号和关节运动学（joint kinematics）来估计膝关节屈/伸力矩。结果显示，NMS-GP 模型的预测误差显著低于传统 NMS 模型，支持了将 NMS 模型特征融入机器学习模型可发挥各自模型优势的假说。此外，NMS-GP 模型对力矩预测中的不确定性进行了量化。力矩预测不确定性在每个运动开始时相对较高，尤其是在自选速度步态中。在自选速度步态的支撑末期（terminal stance）与摆动相（swing）中也观察到较高的不确定性。能够不仅准确地预测关节力矩、还能计算这些预测的不确定性，在若干场景中具有非常重要的潜在意义，例如同时保障外骨骼（exoskeletons）的有效性与安全性。通过将这些不确定性纳入控制策略，可以更好地管理风险，确保用户–外骨骼界面（user-exoskeleton interface）的安全与舒适性。

具备不确定性量化能力的 NMS-GP 模型

---

```
        预测膝关节力矩的不确定性                正常步态          自选速度步态          快速步态          坐起             起坐
        量化（Uncertainties Quantification）  Normal Walking  Self Selected Speed Walking  Fast Walking  Sit-to-stand  Stand-to-sit

        预测膝关节力矩（95% CL）                  %Gait            %Gait                   %Gait          %Cycle         %Cycle
                                                                                                (a)

        膝关节力矩（Nm/Kg）

                                                伸 Ext

                                                屈 Flex

                                                %Gait            %Gait                   %Gait          %Cycle         %Cycle
                                                                ID                       NMS                          NMS-GP
                                                                                                (b)
```

图 5. (a) NMS-GP 模型在五种日常活动中跨受试者预测膝关节力矩的所计算不确定性，以所有受试者与试验的均值 ± 1 标准差表示。预测不确定性以 95% 置信水平（CL，由彩色阴影表示）进行量化，表示被建模函数的真实值有 95% 的概率落在预测区间内。不确定性值较高表示对预测值的置信度较低。(b) 在五种日常活动中，使用逆动力学（inverse dynamics，ID）测量的以及使用 NMS 与 NMS-GP 模型预测的膝关节屈/伸力矩（Nm/kg）的一个示例。NMS-GP 模型中的阴影区域表示期望均值的不确定性。

---

*Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:45:09 UTC from IEEE Xplore. Restrictions apply.*
```

经发现能够以相对较低的误差准确预测膝关节力矩（均方根误差（RMSE）≤ 0.05 Nm/kg，归一化均方根误差（NRMSE）≤ 5.9%），并且性能优于传统的神经肌肉骨骼（NMS）模型（RMSE ≤ 0.14 Nm/kg，NRMSE ≤ 18.3%）。这些发现印证了我们先前研究 [22] 的结论，在该研究中我们应用了一种混合的神经肌肉骨骼-人工神经网络（NMS-ANN）模型来预测步态和等速运动期间的踝关节力矩。在该研究中，我们将来自 NMS 模型的物理特征与标准人工神经网络（ANN）相结合，发现 NMS-ANN 模型在大多数情况下表现良好，仅有少数例外。然而，该混合 NMS-ANN 模型仅在踝关节进行了评估。在本研究中，我们进一步探索了其他机器学习模型，具体来说是包含预测中不确定性量化的高斯过程（GP）模型。我们观察到 NMS-GP 模型在每个运动开始时倾向于具有较高的预测不确定性。这可能归因于每个运动开始时从 NMS 模型所采用的物理特征，在这些位置也观察到了较大的偏移。NMS 模型需要来自前两个时间步的神经激活来计算每个肌腱单元（MTU）的肌肉神经激活 [22]。然而，这些数据在运动周期开始时并不可得。它们转而使用前两个时间步的肌电图（EMG）进行近似。这种近似可能会在预测力矩中引入初始偏移。此外，在自选速度步行中的站立末期和摆动末期观察到了相对较高的不确定性。这可能归因于步态站立相与摆动相之间的过渡期，在该过渡期中可能存在不同的肌肉协调模式。这些高不确定性时段应在康复外骨骼（exoskeleton）控制器的设计中予以考虑。

我们在本研究中仅使用了单一的机器学习模型和 NMS 模型；需要进行涉及其他机器学习方法和 NMS 模型的进一步研究以推广我们的发现。此外，未来的研究应着重于将具有不确定性量化的关节力矩预测应用于外骨骼控制，这对于实现真实可行的应用至关重要。我们的方法在不同速度的步态、坐起（sit-to-stand）和坐下（stand-to-sit）动作上进行了测试；需要进一步的测试和验证以确定其在更广泛活动中的适用性。本研究的其他局限性在于，所招募的受试者在年龄和整体健康状况方面相对同质化。为了将这些发现推广到其他研究人群，必须开展涉及具有更广泛年龄范围和神经肌肉功能的多样化受试者的进一步研究。

## V. 结论（CONCLUSION）

在本研究中，我们提出了一个 NMS-GP 模型，用于使用测量的肌肉肌电图（EMG）信号和关节运动学数据来估计膝关节屈/伸力矩。NMS-GP 模型将传统的 NMS 模型与包含不确定性量化的现代机器学习技术相结合。所提出的 NMS-GP 模型不仅提供了更高的预测精度，还提供了预测中的不确定性度量。该方法可应用于提升康复外骨骼的性能，并在外骨骼控制和干预措施的实时评估等各种应用中展现出巨大潜力。

[1] Z. Li、Y. Yuan、L. Luo、W. Su、K. Zhao、C. Xu、J. Huang 和 M. Pi，"脑/肌电混合信号驱动的可穿戴行走外骨骼以增强爬楼梯活动中的运动能力"，IEEE Transactions on Medical Robotics and Bionics，第 1 卷，第 4 期，第 218–227 页，2019 年。

[2] M. Sartori、D. G. Lloyd 和 D. Farina，"面向个性化神经康复技术的神经数据驱动肌肉骨骼建模"，第 879–893 页，2016 年。

[3] L. Li、M. J. Foo、J. Chen、K. Y. Tan、J. Cai、R. Swaminathan、K. S. G. Chua、S. K. Wee、C. W. K. Kuah、H. Zhuo 等，"移动机器人平衡辅助装置（Mobile Robotic Balance Assistant, MRBA）：用于日常生活的步态辅助与跌倒干预机器人"，Journal of NeuroEngineering and Rehabilitation，第 20 卷，第 1 期，第 1–17 页，2023 年。

[4] L. Luo、M. J. Foo、M. Ramanathan、J. K. Er、C. H. Chiam、L. Li、W. Y. Yau 和 W. T. Ang，"用于步态辅助的下肢外骨骼轨迹生成与控制"，Journal of Intelligent & Robotic Systems，第 106 卷，第 3 期，第 64 页，2022 年。

[5] Z. Li、G. Li、X. Wu、Z. Kan、H. Su 和 Y. Liu，"基于人机协作操作的双臂外骨骼非对称协同控制"，IEEE Transactions on Industrial Electronics，第 68 卷，第 11 期，第 12 126–12 139 页，2021 年。

[6] S. Yang、N. P. Garg、R. Gao、M. Yuan、B. Noronha、W. T. Ang 和 D. Accoto，"基于学习的运动意图预测用于上肢辅助机器人的末端控制"，Sensors，第 23 卷，第 6 期，第 2998 页，2023 年。

[7] Z. Li、K. Zhao、L. Zhang、X. Wu、T. Zhang、Q. Li、X. Li 和 C.-Y. Su，"用于稳定动态行走的可穿戴下肢外骨骼人在回路控制"，IEEE/ASME Transactions on Mechatronics，第 26 卷，第 5 期，第 2700–2711 页，2020 年。

[8] B. Chen、H. Ma、L.-Y. Qin、F. Gao、K.-M. Chan、S.-W. Law、L. Qin 和 W.-H. Liao，"下肢外骨骼的最新进展与挑战"，Journal of Orthopaedic Translation，第 5 卷，第 26–37 页，2016 年。

[9] Z. Li、B. Huang、Z. Ye、M. Deng 和 C. Yang，"基于导纳控制的机器人外骨骼物理人机交互"，IEEE Transactions on Industrial Electronics，第 65 卷，第 12 期，第 9614–9624 页，2018 年。

[10] M. Zhang、A. McDaid、A. J. Veale、Y. Peng 和 S. Q. Xie，"具有关节空间力分布的并联踝关节康复机器人自适应轨迹跟踪控制"，IEEE Access，第 7 卷，第 85 812–85 820 页，2019 年。

[11] S. Yao、Y. Zhuang、Z. Li 和 R. Song，"基于肌电驱动肌肉骨骼模型的踝关节外骨骼自适应导纳控制"，Frontiers in Neurorobotics，第 12 卷，第 1–12 页，2018 年。

[12] G. Durandau、D. Farina 和 M. Sartori，"肌电信号驱动的鲁棒实时肌肉骨骼建模"，IEEE Transactions on Biomedical Engineering，第 65 卷，第 3 期，第 556–564 页，2017 年。

[13] K. Fujii、T. Abe、S. Kubota、A. Marushima、H. Kawamoto、T. Ueno、A. Matsushita、K. Nakai、K. Saotome、H. Kadone 等，"主动驱动外骨骼混合辅助肢体（Hybrid Assistive Limb, HAL）用于胸椎后纵韧带骨化症的术后训练：病例报告"，The Journal of Spinal Cord Medicine，第 40 卷，第 3 期，第 361–367 页，2017 年。

[14] L. Marchal-Crespo 和 D. J. Reinkensmeyer，"神经损伤后机器人运动训练控制策略综述"，Journal of Neuroengineering and Rehabilitation，第 6 卷，第 1 期，第 1–15 页，2009 年。

[15] Y. Huang、R. Song、A. Argha、A. V. Savkin、B. G. Celler 和 S. W. Su，"通过切换机制对人体三维运动意图的连续描述"，IEEE Transactions on Neural Systems and Rehabilitation Engineering，第 28 卷，第 1 期，第 277–286 页，2019 年。

[16] S. Y. A. Mounis、N. Z. Azlan 和 F. Sado，"基于受试者功能能力的上肢康复按需辅助控制策略"，Measurement and Control，第 52 卷，第 9-10 期，第 1354–1361 页，2019 年。

[17] C. Pizzolato、D. J. Saxby、D. Palipana、L. E. Diamond、R. S. Barrett、Y. D. Teng、D. G. Lloyd、B. J. Fregly 和 M. Reggiani，"基于神经肌肉骨骼建模的脊髓损伤后康复假肢"，Frontiers in（第 48 卷期刊文献引文跨栏延续）… ，第 879–893 页，2016 年。

[18] M. Ezati、B. Ghannadi 和 J. McPhee，"以步态为重点的人体运动动力学仿真方法综述"，Multibody System Dynamics，第 47 卷，第 3 期，第 265–292 页，2019 年。

[19] L. Zhang、Z. Li、Y. Hu、C. Smith、E. M. G. Farewik 和 R. Wang，"基于肌电驱动神经肌肉骨骼模型与人工神经网络模型的踝关节力矩估计"，IEEE Transactions on Automation Science and Engineering，第 18 卷，第 2 期，第 564–573 页，2020 年。

[20] H. X. Hoang、C. Pizzolato、L. E. Diamond 和 D. G. Lloyd，"神经肌肉参数受试者特异性校准使神经肌肉骨骼模型能够估计健康成人中生理上合理的髋关节接触力"，Journal of Biomechanics，第 80 卷，第 111–120 页，2018 年。

[21] （作者列表跨栏缺失，续接）B. J. Fregly 和 M. Reggiani，"CEINMS：用于研究不同神经控制方案对动态运动任务中肌肉激发与关节力矩预测影响的工具箱"，Journal of Biomechanics，第 48 卷，第 14 期，第 3929–3936 页，2015 年。

[22] L. Zhang、X. Zhu、E. M. Gutierrez-Farewik 和 R. Wang，"基于神经肌肉骨骼求解器驱动人工神经网络（ANN）模型与迁移学习的踝关节力矩预测"，IEEE Journal of Biomedical and Health Informatics，第 26 卷，第 12 期，第 5895–5906 页，2022 年。

[23] E. Soleimani 和 E. Nazerfard，"基于生成对抗网络的人体活动识别跨受试者迁移学习"，Neurocomputing，第 426 卷，第 26–34 页，2021 年。

[24] L. Zhang、X. Zhu、E. M. G. Farewik 和 R. Wang，"基于（肌电信号 EMG 驱动）NN 模型的（关节力矩）估计"，2021 年第 6 届 IEEE 先进机器人与机电一体化国际会议（ICARM）。IEEE，2021 年，第 75–80 页。

[25] J. Liang、Z. Shi、F. Zhu、W. Chen、X. Chen 和 Y. Li，"基于表面肌电（sEMG）信号的高斯过程自回归关节角度预测"，Frontiers in Public Health，第 9 卷，第 1–15 页，2021 年。

[26] N. Chen、Z. Qian、I. T. Nabney 和 X. Meng，"基于高斯过程与数值天气预报的风电功率预测"，IEEE Transactions on Power Systems，第 29 卷，第 2 期，第 656–665 页，2013 年。

[27] H. Guo、Z. Meng、Z. Huang、L. W. Kang、Z. Chen、M. Meghjani、M. Ang 和 D. Rus，"基于高斯过程风险图的安全路径规划"，2019 年 IEEE/RSJ 国际智能机器人与系统会议（IROS）。IEEE，2019 年，第 2044–2051 页。

[28] J. Maritz、F. Lubbe 和 L. Lagrange，"贝叶斯框架下能耗测量与验证的高斯过程回归实践指南"，Energies，第 11 卷，第 4 期，第 1–12 页，2018 年。

[29] Y. Yun、H.-C. Kim、S. Y. Shin、J. Lee、A. D. Deshpande 和 C. Kim，"基于高斯过程回归的步态运动学统计预测方法"，Journal of Biomechanics，第 47 卷，第 1 期，第 186–192 页，2014 年。

[30] C. Pizzolato、M. Reggiani、L. Modenese 和 D. Lloyd，"基于 OpenSim 的下肢实时逆运动学与逆动力学"，Computer Methods in Biomechanics and Biomedical Engineering，第 20 卷，第 4 期，第 436–445 页，2017 年。

[31] F. Leboeuf、R. Baker、A. Barré、J. Reay、R. Jones 和 M. Sangeux，"常规步态模型——一个复现过去、面向未来的开源实现"，Gait & Posture，第 69 卷，第 235–241 页，2019 年。

[32] M. G. Pandy，"人体运动的计算机建模与仿真"，Annual Review of Biomedical Engineering，第 3 卷，第 1 期，第 245–273 页，2001 年。

[33] R. Wang，"踝足部步态损伤的生物力学后果：损伤、力线异常与协同收缩"，博士学位论文，瑞典皇家理工学院（KTH Royal Institute of Technology），2012 年。

[34] D. A. Winter、H. G. Sidwall 和 D. A. Hobson，"人体运动学中的噪声测量与降噪"，Journal of Biomechanics，第 7 卷，第 2 期，第 157–159 页，1974 年。

[35] A. Mantoan、C. Pizzolato、M. Sartori、Z. Sawacha、C. Cobelli 和 M. Reggiani，"MotoNMS：用于神经肌肉骨骼建模与仿真的运动数据处理 MATLAB 工具箱"，Source Code for Biology and Medicine，第 10 卷，第 1 期，第 1–14 页，2015 年。

[36] T. R. Derrick、A. J. van den Bogert、A. Cereatti、R. Dumas、S. Fantozzi 和 A. Leardini，"国际生物力学学会（ISB）关于人体运动分析中节段间力与力矩报告的建议"，Journal of Biomechanics，第 99 卷，第 1–10 页，2020 年。

[37] M. Sartori、M. Reggiani、D. G. Lloyd 和 E. Pagello，"人体下肢神经肌肉骨骼模型：面向动力外骨骼中多关节的肌电驱动驱动"，2011 年 IEEE 国际机器人与仿生学会议（ROBIO）。IEEE，2011 年，第 1–6 页。

[38] C. E. Rasmussen，《机器学习中的高斯过程》，Springer，2004 年。

[39] T. Beckers，"高斯过程模型简介"，arXiv 预印本 arXiv:2102.05497，2021 年。

经授权许可使用，仅限于：中南民族大学。下载于 2026 年 10 月 1 日 03:45:09 UTC，来自 IEEE Xplore。使用受限。
