# 4D 中的人体：基于 Transformer 的人体重建与跟踪

Shubham Goel Georgios Pavlakos Jathushan Rajasegaran Angjoo Kanazawa∗ Jitendra Malik∗ shubham-goel, pavlakos, jathushan, kanazawa @berkeley.edu, malik@eecs.berkeley.edu 加州大学伯克利分校

![](images/f12843d9d3312d476f7bc049d8603b239166841f16b0a8f20e25caee9daf14f4.jpg)

![](images/0e97f95d34b671a6febf05402161289549a055ef8ba6103b880835627ad111c9.jpg)  
图 1：人体网格恢复的“Transformer 化”视图。我们介绍了 HMR 2.0，这是一套完全基于 Transformer 的方法，用于从单张图像进行三维人体姿态和形状重建。除了在多种姿态和视角下表现出卓越的性能外，HMR 2.0 还充当了改进系统的关节，用于在 4D（4DHumans）中联合重建和跟踪人体。此处展示了 HMR 2.0 对左侧图像中每个二维检测的输出重建结果。

## 摘要

我们提出了一种重建人体并在时间维度上对其进行跟踪的方法。我们方法的核心是提出了一个完全基于 Transformer 的人体网格恢复网络。该网络 HMR 2.0 推进了技术水平，并展示了对过去难以从单张图像重建的非典型姿态进行分析的能力。为了分析视频，我们使用 HMR 2.0 的三维重建结果作为在三维空间中运行的跟踪系统的输入。这使我们能够处理多人场景，并在遮挡事件中保持身份一致性。我们的完整方法 4DHumans 在单目视频人体跟踪任务上取得了最先进的结果。此外，我们展示了 HMR 2.0 在动作识别下游任务上的有效性，相较于之前基于姿态的动作识别方法取得了显著改进。我们的代码和模型可在项目网站上获取：https://shubham-goel.github.io/4dhumans/。

## 1. 引言

在本文中，我们提出了一种完全基于 Transformer 的方法，用于从单张图像恢复人体的三维网格，并在视频中随时间对其进行跟踪。即使对于之前方法难以处理的非典型姿态，我们也获得了前所未有的单张图像三维重建精度（见图 1）。在视频中，我们通过三维跟踪将这些重建结果在时间上进行关联，在此过程中弥合由于遮挡或检测失败造成的间隙。这些四维重建结果可在项目网页上查看。

我们的问题建模与方法可以视为对人体网格恢复（human mesh recovery）领域先前工作 HMR [30] 以及三维跟踪领域先前工作 PHALP [65] 的"Transformer 化"。自开创性的 ViT 论文 [15] 以来，"Transformer 化"这一过程，即把模型从 CNN 或 LSTM 转换到 Transformer 主干网络，已在多个计算机视觉任务中取得快速进展，例如 [8, 16, 24, 40, 61, 77]。在二维姿态（二维人体关键点）方面，ViTPose [81] 已经完成了这种转换。我们以此为出发点，开发了一个新版本的 HMR，称之为 HMR 2.0，以彰显其承袭关系。

我们利用 HMR 2.0 构建了一个能够从视频中同时重建并跟踪人体的系统。我们基于最近的三维跟踪系统 PHALP [65]，借助我们的姿态恢复方法对其进行了简化和改进。该系统能够在四维空间中重建人体，这也正是我们方法名称 4DHumans 的由来。4DHumans 可以部署于任意视频，并能够联合地对视频中的人物进行跟踪与重建。为每一个人创建一个跟踪实体这一功能，是分析并理解视频中人体的基础。除了在 PoseTrack 数据集 [1] 上取得最优的跟踪结果之外，我们还将 HMR 2.0 应用于动作识别这一下游任务。我们沿用近期工作 [63] 的系统设计，并展示了使用 HMR 2.0 能够在 AVA v2.2 数据集上对动作识别目前最优水平带来显著提升。

本文毫不掩饰地是一篇系统论文。我们所做的设计选择，旨在打造在真实场景下（in the wild）用于三维人体重建与跟踪的最佳系统。我们的模型已在项目网页上公开发布。计算机视觉与自然语言处理领域都出现了一种新兴趋势：大规模预训练模型能够找到广泛的下游应用，从而证明其规模扩展的合理性。HMR 2.0 正是这样一种大规模预训练模型，它不仅在计算机视觉中有潜在用途，在机器人学 [54, 62, 73]、计算机图形学 [76]、生物力学 [60] 以及其他需要从图像或视频中分析人体形态与运动的领域，同样具有广阔的应用前景。

我们的贡献可总结如下：

1. 我们提出了一种用于人体网格恢复的端到端"Transformer 化"架构 HMR 2.0。在不依赖领域专属设计的前提下，我们在三维人体姿态重建上优于现有方法。

2. 在 HMR 2.0 的基础上，我们设计了 4DHumans，能够在视频中联合重建与跟踪人体，并在跟踪任务上取得最优结果。

3. 我们证明了 HMR 2.0 所提供的更精确的三维姿态能够提升动作识别这一下游任务的性能，最终在 AVA 基准上取得了 42.3 mAP 的最优结果。

## 2. 相关工作

## 单图像人体网格恢复

虽然已有许多方法通过迭代优化来估计三维人体姿态与形状，例如 SMPLify [7] 及其变体 [22, 38, 56, 66, 72, 85]，但在本分析中我们将聚焦于直接从单张图像回归人体形状的方法。这类方法的典型代表是 HMR [30]，它使用 CNN 回归 SMPL [45] 参数。自其提出以来，已有许多工作对原始方法进行了改进。值得注意的是，许多研究提出了替代的伪真值生成方法，包括利用时序信息 [3]、多视角 [39] 或迭代优化 [35, 29, 57] 等。SPIN [35] 提出了一种循环内优化方法，将 SMPLify [7] 整合进 HMR 的训练过程中。本文同样依赖伪真值拟合用于训练，并采用 [37] 进行离线拟合。

最近，一些工作提出了针对 HMR 架构的更具专门化的设计。PyMAF [89, 88] 引入了网格对齐模块用于回归 SMPL 参数。PARE [34] 提出了一种基于身体部位引导的注意力机制以更好地处理遮挡问题。HKMR [20] 在预测时利用了已知的 SMPL 层次结构信息。Holo-Pose [23] 提出了一种跟随各身体关节点二维位置的池化策略。与这些方法不同，本文采用了一种不包含任何领域特定决策的设计，并证明其性能优于所有先前方法。

许多相关方法采用非参数化的预测方式，即不再估计 SMPL 模型的参数，而是直接回归网格的顶点。GraphCMR [36] 使用图神经网络进行预测，METRO [42] 和 FastMETRO [10] 使用 Transformer，而 Mesh Graphormer [43] 则采用了二者的混合架构。由于我们回归的是 SMPL 模型参数而非网格顶点的位置，因此与这些方法无法直接比较。不过，本文展示了如何将一种完全"Transformer 化"的设计应用于 HMR。

## 从视频中恢复人体网格与运动

为了将人体网格恢复扩展到时间维度，大多数方法以 HMR [30] 的基础骨干网络为起点，并提出用于融合逐帧特征的时序编码器设计。HMMR [31] 在 HMR [30] 提取的特征上使用卷积编码器。VIBE [33]、MEVA [48] 和 TCMR [11] 使用循环时序编码器。DSD [71] 结合了卷积层与自注意力层，而 MAED [75] 和 t-HMMR [57] 则采用基于 Transformer 的时序编码器。Baradel 等人 [5, 4] 也使用 Transformer 进行时序姿态预测，但其直接在 SMPL 姿态上操作。这些方法的一个关键局限在于，它们通常运行在跟踪较为简单的场景中 [31, 90]，例如只包含单人或遮挡极少的视频。与之相反，本文所提出的完整 4DHumans 方法同时还解决了跟踪问题。

视频中的人体追踪。最近，出现了一些通过依赖 HMR 模型进行 3D 人体重建来实现最先进追踪性能的方法，例如 T3DP [64] 和 PHALP [65]。在这些方法中，每个行人检测结果都使用 HMR 网络 [57] 提升到 3D，然后使用从提升 [64] 和预测 [65] 得到的 3D 表示在视频中进行人体追踪。实验结果表明，PHALP 在多个追踪基准测试上表现非常好（主要要求是图像具有足够的空间分辨率以允许将行人提升到 3D）。我们使用这些追踪流程，特别是 PHALP，作为评估人体网格恢复方法的任务。

![](images/6ae63cbaa75636c279a47a0e8a59b1a2d720f15edad110b93f315dc1327670f6.jpg)  
图 2：我们方法的概览。左：HMR 2.0 是一个完全"Transformer 化"的人体网格恢复网络版本。右：我们将 HMR 2.0 作为 4DHumans 系统的骨干网络，该系统基于 PHALP [65]，用于在 4D 中联合重建和追踪人体。

动作识别。动作识别通常使用来自原始视频输入的外观特征来执行。此类方法的典型例子包括 SlowFast [18] 和 MViT [16]。同时，也有一些方法使用从人体姿态信息中提取的特征，例如 Po-Tion [12] 和 JMRN [68]。最近的方法 LART [63] 通过融合基于视频的特征和来自 3D 人体姿态估计的特征，展示了动作识别的最先进性能。我们使用该方法的流程，并将动作识别作为评估人体网格恢复方法的下游任务。

## 3. 人体重建

## 3.1. 预备知识

身体模型。SMPL 模型 [46] 是一个低维参数化的人体模型。给定姿态 $(\theta \in \mathbb{R}^{24 \times 3 \times 3})$ 和形状 $(\beta \in \mathbb{R}^{10})$ 的输入参数，它输出一个具有 $N = 6890$ 个顶点的网格 $\bar{M} \in \mathbb{R}^{3 \times N}$。身体关节点 $X \in \mathbb{R}^{3 \times k}$ 被定义为顶点的线性组合，可以通过 $X = MW$ 计算得到，其中权重 $W \in \mathbb{R}^{N \times k}$ 是固定的。注意，姿态参数 $\theta$ 包括身体姿态参数 $\theta_b \in \mathbb{R}^{23 \times 3 \times 3}$ 和全局朝向 $\theta_g \in \mathbb{R}^{3 \times 3}$。

相机。我们使用具有固定焦距和内参 $K$ 的透视相机模型。每个相机 $\pi = (R, t)$ 由全局朝向 $R \in \mathbb{R}^{3\times3}$ 和平移 $t \in \mathbb{R}^{3}$ 组成。给定这些参数，SMPL 空间中的点（例如关节 $X$）可以投影到图像中，表示为 $x = \pi(X) = \Pi(K(RX + t))$，其中 $\Pi$ 是使用相机内参 $K$ 的透视投影。由于 $\theta$ 已经包含了全局朝向，在实践中我们假设 $R$ 为单位矩阵，仅预测相机平移 $t$。

HMR（人体网格重建）。人体网格重建（Human Mesh Reconstruction, HMR）任务的目标是学习一个预测器 $f(I)$，给定单张图像 $I$，通过 MLP 预测其 3D 姿态和形状参数来重建图像中的人物。遵循典型的参数化方法 [30, 35]，我们将 $f$ 建模为预测 $\Theta = [\theta, \beta, \pi] = f(I)$，Pose 其中 $\theta$ 和 $\beta$ 分别是 SMPL 姿态参数和形状参数，$\pi$ 是相机平移。

## 3.2. 架构

我们将 HMR [30] 重新构想为一个端到端的 Transformer 架构，不包含任何领域特定的设计选择。然而，它的性能优于所有采用高度定制化架构和精心设计决策的现有方法。如图 2 所示，我们使用 (i) 一个 ViT [15] 来提取图像 token，以及 (ii) 一个标准的 Transformer 解码器，对图像 token 进行交叉注意力以输出 $\Theta$。

ViT。Vision Transformer，即 ViT [15]，是为处理图像而改装的 Transformer [74]。输入图像首先被分块为输入 token，然后通过 Transformer 得到输出 token。这些输出 token 随后被送入 Transformer 解码器。我们使用 ViT-H/16，即"Huge"（巨大）变体，采用 $16\times16$ 的输入块大小。更多细节请参见补充材料。

Transformer 解码器。我们使用带有多头自注意力的标准 Transformer 解码器 [74]。它通过交叉注意力处理单个（零）输入 token 与输出图像 token 的关系，最终通过线性读出层得到 $\Theta$。我们遵循 [35] 并使用 [91] 的表示方法回归 3D 旋转。

## 3.3. 损失函数

遵循 HMR 文献 [30, 35] 中的最佳实践，我们使用 2D 损失、3D 损失和判别器的组合来训练预测器 $f$。由于我们使用多数据集混合进行训练，每个数据集的标注类型不同，因此对小批量中的每张图像采用这些损失的一个子集。即使使用伪真值标注时也使用相同的损失。给定输入图像 $I$，模型预测 $\Theta = [\theta, \beta, \pi] = f(I)$。每当我们能够获取真值 SMPL 姿态参数 $\theta^*$ 和形状参数 $\beta^*$ 时，我们使用 MSE 损失对模型预测进行自举：

$$
\mathcal{L}_{\mathrm{smpl}} = ||\theta - \theta^*||_2^2 + ||\beta - \beta^*||_2^2 .
$$

当图像具有准确的 3D 关键点真值标注 $X ^ { * }$ 时，我们额外使用 L1 损失对预测的 3D 关键点 $X$ 进行监督：

$$
{ \mathcal { L } } _ { \mathrm { k p 3 D } } = | | X - X ^ { * } | | _ { 1 } .
$$

当图像具有 2D 关键点标注 $x ^ { * }$ 时，我们使用 L1 损失对预测 3D 关键点的投影 $\pi ( X )$ 进行监督：

$$
\mathcal L _ { \mathrm { k p 2 D } } = | | \pi ( X ) - x ^ { * } | | _ { 1 } .
$$

此外，我们希望确保模型预测出有效的 3D 姿态，并采用 HMR [30] 中的对抗先验。它将模型参数分解为：(i) 身体姿态参数 $\theta _ { b }$，(ii) 形状参数 $\beta$，以及 (iii) 每个部件的相对旋转 $\theta _ { i }$，即 SMPL 模型中 23 个关节点的每一个对应一个 3D 旋转。我们为身体模型的每个因子训练一个判别器 $D _ { k }$，生成器损失可表示为：

$$
\mathcal { L } _ { \mathrm { a d v } } = \sum _ { k } ( D _ { k } ( \theta _ { b } , \beta ) - 1 ) ^ { 2 } .
$$

## 3.4. 伪真值拟合

我们通过计算伪真值标注来扩展到无标注数据集（即 InstaVariety [31]、AVA [21]、AI Challenger [78]）。给定任意图像，我们首先使用现成的检测器 [40] 和身体关键点估计器 [81] 获取边界框和对应的 2D 关键点。然后，我们使用 ProHMR [37] 将 SMPL 网格拟合到这些 2D 关键点，以获得伪真值 SMPL 参数 $\theta ^ { * }$ 和 $\beta ^ { * }$，以及相机参数 $\pi ^ { * }$。

## 4. 人物跟踪

在多人视频中，我们需要具备跨时间关联人物的能力，即执行跟踪。为此，我们基于 PHALP [65] 进行构建，它是一个基于 HMR 风格 3D 重建特征的当前最先进跟踪器。其基本思想是在单帧中检测人物，并将其"提升"到 3D，提取其 3D 姿态、3D 空间中的位置（由估计的相机推导得出）以及 3D 外观（由纹理图推导得出）。针对每个独立人物，随时间逐步构建一个轨迹片段（tracklet）表示。递归步骤是为每个轨迹片段预测下一帧中人物的姿态、位置和外观，全部在 3D 中进行，然后在将这些自顶向下的预测与该帧中自底向上检测到的人物在提升到 3D 后进行最佳匹配。每个轨迹片段所表示的状态随后根据传入的观测进行更新，并迭代此流程。由于轨迹片段的 3D 表示会基于历史信息持续更新，因此可以实现在遮挡情况下的跟踪。

![](images/22a14dcdfd53a7372b6c769eb76e9a5742b1d4a6e221f50e62b438527c60fbc0.jpg)  
图 3：姿态预测：我们在从文献 [63] 获取的超过 100 万条轨迹上训练了一个 BERT 风格 [13] 的 Transformer 模型。这使得我们能够使用同一个模型进行未来预测和缺失检测的非模态补全（amodal completion）。为了预测未来姿态（$t+1$，$t+2$，…），我们使用相应的位置嵌入查询模型，查询时使用一个掩码 token（mask-token）。类似地，对于非模态补全，我们将缺失的检测替换为掩码 token。

我们认为，一个鲁棒的姿态预测器在跟踪这个下游任务中进行评估时也应该表现良好，因此我们使用跟踪指标作为评估 3D 重建质量的代理。但首先我们需要修改 PHALP 框架，以允许对不同的姿态预测模型进行公平比较。最初，PHALP 使用基于 HMR 网络最后一层的姿态特征，即一个 2048 维的嵌入空间。这限制了 PHALP 与不同姿态模型（例如 HMR 2.0、PARE、PyMAF 等）配合使用的能力。为了创建更通用的 PHALP 版本，我们进行如下修改：用 SMPL 姿态参数来表示姿态，并相应地优化 PHALP 代价函数以利用新的姿态距离。类似地，我们使姿态预测器适配在 SMPL 参数空间中运行。更具体地说，我们按照图 3 所示的方式，通过掩码随机的姿态 token 来训练一个 vanilla Transformer 模型 [74]。这使我们能够预测未来的姿态随时间的演变，以及缺失检测的非模态补全。通过这些修改，我们可以插入任何网格恢复方法并在任何视频上运行它们。我们将这个修改后的版本称为 PHALP<sup>′</sup>。

**4DHumans。** 我们最终的跟踪系统 4DHumans 使用了一个基于采样的无参数外观头（appearance head）和一个新的姿态预测器（图 3）。为了建模外观，我们将网格上的可见点通过投影到输入图像上，并从对应像素处采样颜色，来进行纹理化处理。

为了在视频中跟踪人物，以往的方法依赖于现成的跟踪方法，并使用其输出来重建视频中的人物（例如，从跟踪输出中获取边界框，然后重建人物）。例如，PHD [90]、HMMR [31] 只能在场景中只有单人的视频上运行。在本工作中，我们将重建和跟踪合并为一个单一系统，并证明更好的姿态重建会带来更好的跟踪效果，而这个组合系统现在可以在野外任何视频上运行。

## 5. 实验

本节中，我们定性和定量地评估我们的重建与跟踪系统。首先，我们展示 HMR 2.0 在标准 2D 和 3D 姿态精度指标上优于先前的方法（第 5.2 节）。其次，我们展示 4DHumans 是一个多功能的跟踪器，能够达到最先进的性能（第 5.3 节）。第三，我们通过在动作识别下游应用上的优异性能，进一步证明我们恢复的姿态具有鲁棒性和准确性（第 5.4 节）。最后，我们讨论在设计 HMR 2.0 时的实验研究，并对一系列设计选择进行消融实验（第 5.5 节）。

## 5.1. 设置

数据集。遵循先前的工作，我们使用典型的数据集进行训练，即 Human3.6M [27]、MPI-INF-3DHP [49]、COCO [44] 和 MPII [2]。此外，我们使用 InstaVariety [31]、AVA [21] 和 AI Challenger [78] 作为额外数据，并在这些数据上生成伪真值拟合结果。

基线方法。我们在能够与许多先前工作进行比较的基准上报告性能（第 5.2 节），但我们也与近期最先进的方法进行了更详细的对比，即 PyMAF [89]、CLIFF [41]、HMAR [65]、PARE [34] 和 PyMAF-X [88]。为了公平比较，我们仅评估 PyMAF-X 的仅人体（body-only）性能。

## 5.2. 姿态精度

3D 指标。对于 3D 姿态精度，我们遵循先前工作（例如 [35]）的典型协议，在 3DPW 测试划分和 Human3.6M 验证划分上展示结果，在表 1 中报告 MPJPE 和 PA-MPJPE。请注意，我们仅与不使用 3DPW 训练集进行训练的方法进行比较（与我们类似）。我们观察到，使用我们的 HMR 2.0a 模型（仅在典型数据集上训练），我们能够在所有指标上优于所有先前的基线。然而，我们认为这些基准已经非常饱和，姿态指标上较小的差异往往并不十分显著。事实上，我们观察到，通过对 3DPW 性能做出小幅让步，我们的 HMR 2.0b 模型（在更多数据 AVA [21]、AI Challenger [78] 和 InstaVariety [31] 上训练更长时间）能够在比 Human3.6M 和 3DPW 中所能找到的更不常见的姿态上取得更好的结果。我们从定性的角度以及从 2D 姿态重投影所评估的性能（表 2）中观察到了这一点。此外，我们观察到 HMR 2.0b 是一个更为鲁棒的模型，并在本文余下部分使用它进行评估。

<table><tr><td rowspan="2">方法</td><td colspan="2">3DPW</td><td colspan="2">Human3.6M</td></tr><tr><td>MPJPE PA-MPJPE|MPJPE PA-MPJPE</td><td></td><td></td><td></td></tr><tr><td>Kanazawa et al. [31]</td><td>116.5</td><td>72.6</td><td>=</td><td>56.9</td></tr><tr><td>Doersch et al. [14]</td><td>=</td><td>74.7</td><td></td><td></td></tr><tr><td>Temmpooral Arnab et al. [3]</td><td></td><td>72.2</td><td>77.8</td><td>54.3</td></tr><tr><td>DSD [71]</td><td></td><td>69.5</td><td>59.1</td><td>42.4</td></tr><tr><td>VIBE [33]</td><td>93.5</td><td>56.5</td><td>65.9</td><td>41.5</td></tr><tr><td>Pavlakos et al. [59]</td><td></td><td></td><td></td><td>75.9</td></tr><tr><td>HMR [30]</td><td>130.0</td><td>76.7</td><td>88.0</td><td>56.8</td></tr><tr><td>NBF [53]</td><td></td><td></td><td></td><td>59.9</td></tr><tr><td>GraphCMR [36]</td><td></td><td>70.2</td><td></td><td>50.1</td></tr><tr><td>HoloPose [23]</td><td></td><td></td><td>60.3</td><td>46.5</td></tr><tr><td>DenseRaC [82]</td><td></td><td></td><td>76.8</td><td>48.0</td></tr><tr><td>SPIN [35]</td><td>96.9</td><td>59.2</td><td>62.5</td><td>41.1</td></tr><tr><td>DecoMR [86]</td><td></td><td>61.7†</td><td></td><td>39.3†</td></tr><tr><td>DaNet [87]</td><td></td><td>56.9</td><td>61.5</td><td>48.6</td></tr><tr><td>Fraq-sed Song et al. [69]</td><td></td><td>55.9</td><td></td><td>56.4</td></tr><tr><td>I2L-MeshNet [51]</td><td>100.0</td><td>60.0</td><td>55.7†</td><td>41.1†</td></tr><tr><td>HKMR [20]</td><td></td><td></td><td>59.6</td><td>43.2</td></tr><tr><td>PyMAF [89]</td><td>92.8</td><td>58.9</td><td>57.7</td><td>40.5</td></tr><tr><td>PARE [34]</td><td>82.0</td><td>50.9</td><td>76.8</td><td>50.6</td></tr><tr><td>PyMAF-X [88]</td><td>78.0</td><td>47.1</td><td>54.2</td><td>37.2</td></tr><tr><td>HMR 2.0a</td><td>70.0</td><td>44.5</td><td>44.8</td><td>33.6</td></tr><tr><td>HMR 2.0b</td><td>81.3</td><td>54.3</td><td>50.0</td><td>32.4</td></tr></table>

表 1：3D 重建评估：在 3DPW 和 Human3.6M 数据集上的重建误差（单位：毫米）。<sup>†</sup> 表示在非参数化结果上评估的数值。数值越低越好。详情请参阅正文。

<table><tr><td rowspan=1 colspan=4>LSP-Extended  COCO   PoseTrack方法@0.05@0.1 @0.05 @0.1@0.05 @0.1</td></tr><tr><td rowspan=1 colspan=2>PyMAF [89]     一      1</td><td rowspan=1 colspan=1>0.680.86</td><td rowspan=1 colspan=1>0.770.92</td></tr><tr><td rowspan=1 colspan=1>CLIFF [41]</td><td rowspan=1 colspan=1>0.32 0.66</td><td rowspan=1 colspan=1>0.640.88</td><td rowspan=1 colspan=1>0.750.92</td></tr><tr><td rowspan=1 colspan=1>PARE [34]</td><td rowspan=1 colspan=1>0.27 0.60</td><td rowspan=1 colspan=1>0.720.91</td><td rowspan=1 colspan=1>0.790.93</td></tr><tr><td rowspan=1 colspan=1>PyMAF-X [88]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.790.93</td><td rowspan=1 colspan=1>0.850.95</td></tr><tr><td rowspan=1 colspan=1>HMR 2.0a</td><td rowspan=1 colspan=1>0.38 0.72</td><td rowspan=1 colspan=1>0.790.95</td><td rowspan=1 colspan=1>0.86 （0.97</td></tr><tr><td rowspan=1 colspan=1>HMR 2.0b</td><td rowspan=1 colspan=1>0.53 0.82</td><td rowspan=1 colspan=1>0.86 0.96</td><td rowspan=1 colspan=1>0.900.98</td></tr></table>

表 2：以 2D 方式评估重建结果。在 LSP-Extended、COCO 和 PoseTrack 数据集上，不同阈值下投影关键点的 PCK（Percentage of Correct Keypoints，正确关键点比例）得分。数值越高越好。

**2D 指标。** 我们通过报告在 LSP-Extended [28]、COCO 验证集 [44] 和 PoseTrack 验证集 [1] 上不同阈值下重投影关键点的 PCK 来评估生成姿态的 2D 图像对齐程度。由于 PyMAF(-X) [89, 88] 是使用 LSP-Extended 训练的，因此我们不报告该部分表格中的数据。请注意表 2 中，HMR 2.0b 始终优于所有先前方法。在包含非常规姿态的 LSP-Extended 上，HMR 2.0b 的 PCK@0.05 达到 0.53，比第二名（CLIFF）的 0.32 高出 1.6 倍。在 COCO 和 PoseTrack 等姿态不太极端的较简单数据集上，对于 PCK@0.05，HMR 2.0b 仍然优于次优方法，但优势分别收窄至 9% 和 6%。HMR 2.0a 也优于所有基线方法，但表现不及 HMR 2.0b，尤其是在 LSP-Extended 中较困难的姿态上。

**定性结果。** 我们在图 4 中展示了 HMR 2.0 的定性结果。我们对极端姿态和部分遮挡具有鲁棒性。我们的重建结果与图像对齐良好，并且从新视角观察时仍然有效。此外，我们在图 5 中与我们最接近的竞争对手进行了比较。我们观察到 PyMAF-X，尤其是 PARE，在处理较为非常规的姿态时常常表现不佳，而 HMR 2.0 能给出更为忠实的重建结果。

## 5.3. 跟踪

在跟踪任务中，我们首先展示 PHALP<sup>′</sup> 所引入改进的多功能性，这使我们能够在跟踪的下游任务上评估 3D 姿态估计器。然后，我们针对当前最先进的方法评估我们的完整系统 4DHumans。

## 评估设置

遵循先前工作 [64, 65]，我们基于 ID 切换次数（ID switches）、MOTA [32]、IDF1 [67] 和 HOTA [47] 在 Posetrack 验证集上按照 [65] 的协议报告结果，检测结果来自 Mask R-CNN [25]。

## PHALP<sup>′</sup> 的通用性

### .1 评估不同姿态估计器

有了 PHALP<sup>′</sup> 的这些修改，我们放弃了 [65] 中模型特定的隐空间，转而在 SMPL 空间中进行操作，该空间是大多数网格恢复系统所共享的。这使得 PHALP<sup>′</sup> 更加通用，并允许我们插入不同的 3D 姿态估计器，根据它们在跟踪任务上的性能进行比较。我们在表 3 中进行了这一比较，其中使用了来自最先进 3D 姿态估计器的姿态和位置线索（外观仍使用 HMAR [65]）。我们观察到 HMR 2.0 表现最佳，PARE [34]、HMAR [65] 和 PyMAF-X [88] 在 Posetrack 数据集上紧随其后，彼此之间差异很小。请注意，跟踪性能通常最容易受预测准确中出现的 3D 位置错误的影响，而人体姿态对性能的影响较小 [65]。这意味着良好的跟踪性能可以表明对遮挡具有鲁棒性，因此考虑这一指标是有帮助的，但对于区分姿态的细粒度差异帮助不大。因此，PARE [34]、HMAR [65] 和 PyMAF-X [88] 的竞争性结果表明它们能优雅地处理遮挡，但它们的姿态估计可能仍然不够准确（如表 2 所示）。更多定性比较请参见图 5 和补充材料。

### 4DHumans

表 4 评估了我们完整系统 4DHumans 在 PoseTrack 数据集上的跟踪性能。使用与 [64, 65] 相同的边界框检测器，4DHumans 在所有指标上都优于现有方法，ID 切换次数改善了 22%。使用改进的 ViT-Det 检测器 [40] 可以进一步提升性能。作为我们时序预测模型（图 3）的副产品，我们可以执行非模态补全（amodal completion）并将姿态归因于缺失的检测。我们在补充材料中展示了相关示例。

<table><tr><td rowspan="2">Tracker</td><td rowspan="2">Pose Engine</td><td colspan="3">Posetrack</td></tr><tr><td>HOTA↑</td><td>IDs↓ MOTA↑</td><td>IDF1↑</td></tr><tr><td rowspan="4">PHALP&#x27;</td><td>PARE [34]</td><td>53.6</td><td>510 59.4 472</td><td>76.8</td></tr><tr><td>PyMAF-X [88]</td><td>53.7 551</td><td>59.2 58.7</td><td>76.9 76.5</td></tr><tr><td>CLIFF [41] PyMAF [89]</td><td>53.5 53.0 623</td><td>58.6</td><td>76.1</td></tr><tr><td>HMAR [65]</td><td>53.6 482</td><td>59.3</td><td>77.1</td></tr><tr><td></td><td>HMR 2.0</td><td>54.1</td><td>456 59.4</td><td>77.4</td></tr></table>

表 3：使用不同 3D 姿态估计器的跟踪。通过对 PHALP′ 的修改，我们获得了一个通用跟踪器，可将不同的 3D 姿态估计器插入其中。HMR 2.0、PARE 和 PyMAF-X 在此设置下表现最佳。
<table><tr><td>Method</td><td colspan="2">Posetrack HOTA↑ IDs↓ MOTA↑ IDF1↑</td></tr><tr><td>Trackformer [50]</td><td>46.7 1263</td><td>33.7 64.0</td></tr><tr><td>Tracktor [6]</td><td>38.5 702</td><td>42.4 65.2</td></tr><tr><td>AlphaPose [17]</td><td>37.6 2220</td><td>36.9 66.9</td></tr><tr><td>Pose Flow [79]</td><td>38.0 1047</td><td>15.4 64.2</td></tr><tr><td>T3DP [64]</td><td>50.6 655</td><td>55.8 73.4</td></tr><tr><td>PHALP [65]</td><td>52.9 541</td><td>58.9 76.4</td></tr><tr><td>4DHumans</td><td>54.3 421</td><td>59.8 77.9</td></tr><tr><td>4DHumans + ViTDet</td><td>57.8 382</td><td>61.4 79.1</td></tr></table>

表 4：4DHumans 与 Posetrack 数据集上最先进方法的比较。4DHumans 在所有指标上均达到了最先进的跟踪性能。引入更好的检测系统 [40] 可带来进一步的性能提升。

## 5.4. 动作识别

评估设置。[63] 中的方法是视频动作识别的最先进技术。给定一个视频作为输入，作者提出使用逐帧的 3D 姿态和位置估计（使用现成的 HMR 模型 [65]）作为预测动作标签的附加特征。他们还展示了一个"仅姿态"（pose-only）基线的结果，该基线仅使用 3D 姿态和位置估计来预测动作标签。我们使用此设置，在 AVA 数据集 [21] 的下游动作识别任务上将我们的模型与基线进行比较。在 [63] 中，作者训练了一个以 SMPL 姿态作为输入并预测动作标签的 Transformer。按照他们的设置，我们为每个基线训练了一个单独的动作分类 Transformer。

输入

前视图

侧视图

顶视图

输入

前视图

侧视图

顶视图

![](images/fe1595f8664b56dfb4830d10a2ca9c655233243e3e7ead6c75a88233c4be5977.jpg)  
图 4：HMR 2.0 的定性评估。对于每个示例，我们展示：a) 输入图像，b) 重建叠加，c) 侧视图，d) 顶视图。为了展示 HMR 2.0 的鲁棒性，我们可视化了各种设置下的结果——针对异常姿态（第 1-4 行）、针对异常视角（第 5 行）以及针对具有较差可见性、极端截断和极端遮挡的图像（第 6-8 行）。

比较。在表 5 中比较结果时，我们观察到 HMR 2.0（Human Mesh Recovery 2.0，人体网格恢复 2.0）在不同类别（OM、PI、PM）以及整体上均优于基线方法。它在 AVA 测试集上达到了 22.3 的 mAP（mean Average Precision，平均精度均值），比次优基线高出 14%。由于从姿态进行准确的动作识别需要精细的姿态估计，这一结果强有力地证明了 HMR 2.0 预测的姿态比现有方法更为准确。事实上，当与外观特征结合时，文献 [63] 表明 HMR 2.0 在 AVA 动作识别上取得了 42.3 mAP 的当前最优性能，比次优的 39.5 mAP 高出 7%。

![](images/17eb7863da959ad09d79b505934955de8713fd35c4e0f98f0e8346b4b06caeef.jpg)  
图 5：当前最优网格恢复方法的定性比较。与最接近的竞争对手 PyMAF-X [88] 和 PARE [34] 相比，HMR 2.0 对非常规姿态能给出更忠实的重建结果。

![](images/23803f6ce7573202ba265b9cffda24252a0b28a44404ff8e5f078582faa0302b.jpg)  
图 6：4DHumans 的定性跟踪结果。我们使用了头部掩码（帧号位于左上角）。第一行：在一段长达一分钟的视频中，尽管存在挑战性的姿态和严重的遮挡，我们依然持续跟踪在冰上滑行的人员，且未发生身份切换。第二行：主要人物在与多名其他运动员多次互动中保持被跟踪。第三行：感兴趣的人物在长时间遮挡下仍被持续跟踪。

<table><tr><td>动作模型</td><td>姿态引擎</td><td>OM</td><td>PI</td><td>PM</td><td>mAP</td></tr><tr><td rowspan="5">[63]</td><td>PyMAF [89]</td><td>7.3</td><td>16.9</td><td>34.7</td><td>15.4</td></tr><tr><td>CLIFF [41]</td><td>9.2</td><td>20.0</td><td>40.3</td><td>18.6</td></tr><tr><td>HMAR [65]</td><td>8.7</td><td>20.1</td><td>40.3</td><td>18.3</td></tr><tr><td>PARE [34]</td><td>9.2</td><td>20.7</td><td>41.5</td><td>19.1</td></tr><tr><td>PyMAF-X [88] HMR 2.0</td><td>10.2 11.9</td><td>21.4 24.6</td><td>40.8 45.8</td><td>19.6 22.3</td></tr></table>

表 5：AVA 数据集上的动作识别结果。我们在基于姿态的动作识别这一下游任务上对不同的网格恢复方法进行了基准测试。其中，OM：对象操作（Object Manipulation），PI：人与人物交互（Person Interactions），PM：人物移动（Person Movement）。

## 5.5. HMR 2.0 模型设计

在开发 HMR 2.0 的过程中，我们研究了一系列设计决策。图 7 简要展示了这一探索过程。我们实验了超过 100 种设置，并可视化每次运行中 100 个检查点的性能。在可视化中，我们使用每个检查点在 3DPW 和 LSP-Extended 数据集上的表现。

我们的研究聚焦于模型的若干特定方面，此处将其记录为一系列可供未来研究参考的"经验教训"。在接下来的段落中，我们将经常引用表 $^{6}$，该表使用 3DPW、Human3.6M 和 LSP-Extended 数据集，在 3D 和 2D 指标上对这些方面进行了评估。

![](images/b3756cd2b951a2e574b25232459d747a53fd95a02f802c36e1d234e55e92de64.jpg)  
图7：广泛的模型搜索。每个点表示一个检查点在 3DPW 和 LSP-Extended 上的评估性能。颜色表示不同的运行。我们探索了超过 100 种设置，并可视化每次运行的 100 个检查点。

**骨干网络的影响。** 与大多数之前使用 ResNet 骨干网络的人体网格恢复工作不同，我们的 HMR 2.0 方法依赖于 ViT 骨干网络。为了直接比较骨干网络的影响，模型 B1 使用 ResNet-50 骨干网络和基于 MLP 的头部来实现 IEF（迭代误差反馈 [9, 30]）。相比之下，模型 B2 使用 transformer 骨干网络（ViT-H），同时保持其他设计决策不变。通过更新骨干网络，我们观察到 3D 和 2D 指标都有显著提升，这证明了"transformer 化"步骤的合理性。

**训练数据的影响。** 除了架构之外，我们还研究了训练数据的影响。模型 D1 在典型数据集（H3.6M、MPII、COCO、MPI-INF）上进行训练，这些数据集是大多数先前工作所使用的。作为比较，模型 D2 在训练集中加入了 AVA，遵循 [21]。最终，我们还使用 AI-Challenger 和 Insta-Variety（模型 B2）进行训练，以进一步扩展训练集。可以看到，增加更多的训练数据可以在所有报告的指标上带来提升，但相对于骨干网络的更新带来的好处较小。

**ViT 预训练。** 另一个对模型性能有显著影响的因素是 ViT 骨干网络的预训练。使用随机初始化权重（模型 P1）会导致收敛缓慢且性能较差。如果我们的骨干网络在 Imagenet 上使用 MAE [24] 进行预训练（P2），结果会有所改善。最终，我们选择的模型（HMR 2.0b）首先在 ImageNet 上使用 MAE 进行预训练，然后在 2D 关键点预测任务 [81] 上进行预训练，实现了最佳性能。

**SMPL 头部。** 我们还研究了预测 SMPL 参数的头部架构的影响。与传统的具有 IEF 步骤的基于 MLP 的头部（B2）相比，我们提出的 transformer 解码器（HMR 2.0b）在图像-模型对齐（即 2D 指标）方面提升了性能。

**数据集质量。** 与之前的工作类似，例如 [35]，保持训练数据的高质量至关重要；我们剔除低质量的伪真值拟合（高拟合误差）并去除低置信度的 2D 检测图像。

<table><tr><td rowspan="2">Models</td><td colspan="2">3DPW</td><td colspan="2">Human3.6M</td><td colspan="2">LSP-Extended</td></tr>
<tr><td>MPJPE</td><td>PA-MPJPE</td><td>MPJPE</td><td>PA-MPJPE</td><td>PCK@0.05</td><td>PCK@0.1</td></tr>
<tr><td>HMR 2.0b</td><td>81.3</td><td>54.3</td><td>50.0</td><td>32.4</td><td>0.53</td><td>0.82</td></tr>
<tr><td>B1</td><td>85.2</td><td>56.8</td><td>58.9</td><td>41.4</td><td>0.35</td><td>0.66</td></tr>
<tr><td>B2</td><td>79.7</td><td>53.4</td><td>51.4</td><td>34.4</td><td>0.48</td><td>0.81</td></tr>
<tr><td>D1</td><td>84.1</td><td>54.8</td><td>54.5</td><td>35.1</td><td>0.45</td><td>0.79</td></tr>
<tr><td>D2</td><td>80.2</td><td>53.3</td><td>52.4</td><td>34.9</td><td>0.46</td><td>0.79</td></tr>
<tr><td>P1</td><td>61.7</td><td>89.9</td><td colspan="2">—</td><td>0.52</td><td>—</td></tr>
<tr><td></td><td>98.9</td><td>82.7</td><td colspan="2">58.7</td><td>0.24</td><td>—</td></tr>
<tr><td>P2</td><td>55.6</td><td>49.3</td><td>32.4</td><td>—</td><td>0.52</td><td>0.81</td></tr>
</table>

表 6：消融实验：在 3DPW、Human3.6M 和 LSP-Extended 数据集上对不同模型设计的评估。

## 6. 结论

我们研究了从图像和视频中重建与跟踪人体的问题。首先，我们提出了 HMR 2.0，这是用于人体网格恢复（Human Mesh Recovery）[30] 问题的一个完全"Transformer 化"的版本。HMR 2.0 在常规的 2D/3D 姿态指标上取得了强劲的性能，同时充当了我们改进的视频跟踪器的骨干网络。整个系统 4DHumans 联合重建并跟踪视频中的人物，在跟踪任务上取得了最先进的结果。为了进一步展示我们 3D 姿态估计器 HMR 2.0 的优势，我们将其应用于动作识别任务，并在该任务上展示了对先前基于姿态的方法基线的显著改进。

我们的工作推进了可被 3D 人体重建技术分析的视频的边界。同时，改进后的结果也揭示了未来需要解决的某些局限性。例如，使用 SMPL 模型 [45] 带来了一定的限制，采用更优的模型将使我们能够建模手部姿态和面部表情 [56]，甚至捕捉更大的年龄变化，例如婴儿 [26] 和儿童 [55, 70]。此外，由于我们将每个人独立考虑，我们的重建在捕捉近距离人物的细粒度特性方面表现较差，例如人体接触 [19, 52]。除此之外，我们的重建结果"生活"在相机坐标系中，因此为了正确理解视频中的动作，我们需要通过推断相机运动 [58, 83, 84]，将所有人放在一个统一的全局世界坐标系中。最后，输入分辨率较低会影响我们重建的质量，这可以通过分辨率增强 [80] 来改善。

**致谢** 我们感谢 BAIR 社区成员富有成效的讨论，以及 StabilityAI 慷慨的计算资源资助。本工作得到了 BAIR/BDD 赞助商、ONR MURI（N00014-21-1-2801）以及 DARPA MCS 项目的支持。

## 参考文献

[1] Mykhaylo Andriluka, Umar Iqbal, Eldar Insafutdinov, Leonid Pishchulin, Anton Milan, Juergen Gall, and Bernt Schiele. PoseTrack: A benchmark for human pose estimation and tracking. In CVPR, 2018.

[2] Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler, and Bernt Schiele. 2D human pose estimation: New benchmark and state of the art analysis. In CVPR, 2014.

[3] Anurag Arnab, Carl Doersch, and Andrew Zisserman. Exploiting temporal context for 3D human pose estimation in the wild. In CVPR, 2019.

[4] Fabien Baradel, Romain Bregier, Thibault Groueix, Philippe´ Weinzaepfel, Yannis Kalantidis, and Gregory Rogez. Pose-´ BERT: A generic transformer module for temporal 3D human modeling. PAMI, 2022.

[5] Fabien Baradel, Thibault Groueix, Philippe Weinzaepfel, Romain Bregier, Yannis Kalantidis, and Gr´ egory Rogez.´ Leveraging MoCap data for human mesh recovery. In 3DV, 2021.

[6] Philipp Bergmann, Tim Meinhardt, and Laura Leal-Taixe. Tracking without bells and whistles. In ICCV, 2019.

[7] Federica Bogo, Angjoo Kanazawa, Christoph Lassner, Peter Gehler, Javier Romero, and Michael J Black. Keep it SMPL: Automatic estimation of 3D human pose and shape from a single image. In ECCV, 2016.

[8] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In ECCV, 2020.

[9] Joao Carreira, Pulkit Agrawal, Katerina Fragkiadaki, and Jitendra Malik. Human pose estimation with iterative error feedback. In CVPR, 2016.

[10] Junhyeong Cho, Kim Youwang, and Tae-Hyun Oh. Crossattention of disentangled modalities for 3D human mesh recovery with transformers. In ECCV, 2022.

[11] Hongsuk Choi, Gyeongsik Moon, Ju Yong Chang, and Kyoung Mu Lee. Beyond static features for temporally consistent 3D human pose and shape from a video. In CVPR, 2021.

[12] Vasileios Choutas, Philippe Weinzaepfel, Jer´ ome Revaud,ˆ and Cordelia Schmid. PoTion: Pose motion representation for action recognition. In CVPR, 2018.

[13] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

[14] Carl Doersch and Andrew Zisserman. Sim2real transfer learning for 3D human pose estimation: Motion to the rescue. NeurIPS, 2019.

[15] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021.

[16] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. In ICCV, 2021.

[17] Hao-Shu Fang, Shuqin Xie, Yu-Wing Tai, and Cewu Lu. RMPE: Regional multi-person pose estimation. In ICCV, 2017.

[18] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, Kaiming He. 用于视频识别的 Slowfast 网络. ICCV, 2019.

[19] Mihai Fieraru, Mihai Zanfir, Elisabeta Oneata, Alin-Ionut Popa, Vlad Olaru, Cristian Sminchisescu. 人体交互的三维重建. CVPR, 2020.

[20] Georgios Georgakis, Ren Li, Srikrishna Karanam, Terrence Chen, Jana Kosecka, Ziyan Wu. 分层运动学人体网格恢复. ECCV, 2020.

[21] Chunhui Gu, Chen Sun, David A Ross, Carl Vondrick, Caroline Pantofaru, Yeqing Li, Sudheendra Vijayanarasimhan, George Toderici, Susanna Ricco, Rahul Sukthankar, Cordelia Schmid, Jitendra Malik. AVA：一个时空定位原子级视觉动作的视频数据集. CVPR, 2018.

[22] Peng Guan, Alexander Weiss, Alexandru O Balan, Michael J Black. 从单张图像估计人体形状与姿态. ICCV, 2009.

[23] Riza Alp Guler, Iasonas Kokkinos. HoloPose：野外场景的整体式三维人体重建. CVPR, 2019.

[24] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, Ross Girshick. 掩码自编码器是可扩展的视觉学习器. CVPR, 2022.

[25] Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross Girshick. Mask R-CNN. ICCV, 2017.

[26] Nikolas Hesse, Sergi Pujades, Michael J Black, Michael Arens, Ulrich G Hofmann, A Sebastian Schroeder. 从 RGB-D 序列中学习与跟踪自由运动婴儿的三维身体形状. PAMI, 2019.

[27] Catalin Ionescu, Dragos Papava, Vlad Olaru, Cristian Sminchisescu. Human3.6M：面向自然环境中三维人体感知的大规模数据集与预测方法. PAMI, 2013.

[28] Sam Johnson, Mark Everingham. 从不精确标注中学习有效的人体姿态估计. CVPR, 2011.

[29] Hanbyul Joo, Natalia Neverova, Andrea Vedaldi. 面向野外三维人体姿态估计的样例微调三维人体模型拟合. 3DV, 2021.

[30] Angjoo Kanazawa, Michael J Black, David W Jacobs, Jitendra Malik. 端到端的人体形状与姿态恢复. CVPR, 2018.

[31] Angjoo Kanazawa, Jason Y Zhang, Panna Felsen, Jitendra Malik. 从视频中学习三维人体动力学. CVPR, 2019.

[32] Rangachar Kasturi, Dmitry Goldgof, Padmanabhan Soundararajan, Vasant Manohar, John Garofolo, Rachel Bowers, Matthew Boonstra, Valentina Korzhova, Jing Zhang. 视频中人脸、文本和车辆检测与跟踪的性能评估框架：数据、指标与协议. PAMI, 2008.

[33] Muhammed Kocabas, Nikos Athanasiou, Michael J Black. VIBE：面向人体姿态与形状估计的视频推断. CVPR, 2020.

[34] Muhammed Kocabas, Chun-Hao P Huang, Otmar Hilliges, Michael J Black. PARE：面向三维人体估计的部位注意力回归器. ICCV, 2021.

[35] Nikos Kolotouros, Georgios Pavlakos, Michael J Black, and Kostas Daniilidis. 通过循环模型拟合学习重建 3D 人体姿态与形状. In ICCV, 2019.

[36] Nikos Kolotouros, Georgios Pavlakos, and Kostas Daniilidis. 用于单图像人体形状重建的卷积网格回归. In CVPR, 2019.

[37] Nikos Kolotouros, Georgios Pavlakos, Dinesh Jayaraman, and Kostas Daniilidis. 人体网格恢复的概率建模. In ICCV, 2021.

[38] Christoph Lassner, Javier Romero, Martin Kiefel, Federica Bogo, Michael J Black, and Peter V Gehler. 联合人群：连接 3D 与 2D 人体表示. In CVPR, 2017.

[39] Vincent Leroy, Philippe Weinzaepfel, Romain Bregier,´ Hadrien Combaluzier, and Gregory Rogez. SMPLy 基准测试野外 3D 人体姿态估计. In 3DV, 2020.

[40] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. 探索用于目标检测的纯视觉 Transformer 主干网络. In ECCV, 2022.

[41] Zhihao Li, Jianzhuang Liu, Zhensong Zhang, Songcen Xu, and Youliang Yan. CLIFF：将全帧中的位置信息引入人体姿态与形状估计. In ECCV, 2022.

[42] Kevin Lin, Lijuan Wang, and Zicheng Liu. 基于 Transformer 的端到端人体姿态与网格重建. In CVPR, 2021.

[43] Kevin Lin, Lijuan Wang, and Zicheng Liu. Mesh Graphormer. In ICCV, 2021.

[44] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar,´ and C Lawrence´ Zitnick. Microsoft COCO：上下文中的常见对象. In ECCV, 2014.

[45] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. SMPL：一种蒙皮多人体线性模型. ACM Transactions on Graphics (TOG), 34(6):1–16, 2015.

[46] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL：一种蒙皮多人体线性模型. ACM Trans. Graphics (Proc. SIGGRAPH Asia), 34(6):248:1–248:16, Oct. 2015.

[47] Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixe, and Bastian Leibe.´ HOTA：用于评估多目标跟踪的高阶指标. IJCV, 2021.

[48] Zhengyi Luo, S Alireza Golestaneh, and Kris M Kitani. 通过运动压缩与精炼的 3D 人体运动估计. In ACCV, 2020.

[49] Dushyant Mehta, Helge Rhodin, Dan Casas, Pascal Fua, Oleksandr Sotnychenko, Weipeng Xu, and Christian Theobalt. 使用改进 CNN 监督的野外单目 3D 人体姿态估计. In 3DV, 2017.

[50] Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, and Christoph Feichtenhofer. TrackFormer：基于 Transformer 的多目标跟踪. In CVPR, 2022.

[51] Gyeongsik Moon and Kyoung Mu Lee. I2L-MeshNet：用于从单张 RGB 图像精确估计 3D 人体姿态与网格的图像到像素单元预测网络. In ECCV, 2020.

[52] Lea Muller, Vickie Ye, Georgios Pavlakos, Michael Black,¨ and Angjoo Kanazawa. 生成式空间关系学：来自图像的 3D 社交交互先验 (Generative proxemics: A prior for 3D social interaction from images). arXiv 预印本 arXiv:2306.09337, 2023.

[53] Mohamed Omran, Christoph Lassner, Gerard Pons-Moll, Peter Gehler, and Bernt Schiele. 神经人体拟合：统一深度学习与基于模型的人体姿态和形状估计 (Neural body fitting: Unifying deep learning and model based human pose and shape estimation). In 3DV, 2018.

[54] Austin Patel, Andrew Wang, Ilija Radosavovic, and Jitendra Malik. 从互联网视频中学习模仿物体交互 (Learning to imitate object interactions from internet videos). arXiv 预印本 arXiv:2211.13225, 2022.

[55] Priyanka Patel, Chun-Hao P Huang, Joachim Tesch, David T Hoffmann, Shashank Tripathi, and Michael J Black. AGORA：面向回归分析的地理优化虚拟人 (AGORA: Avatars in geography optimized for regression analysis). In CVPR, 2021.

[56] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black. 富有表现力的人体捕捉：从单张图像重建 3D 双手、面部和身体 (Expressive body capture: 3D hands, face, and body from a single image). In CVPR, 2019.

[57] Georgios Pavlakos, Jitendra Malik, and Angjoo Kanazawa. 从多帧快照中恢复人体网格 (Human mesh recovery from multiple shots). In CVPR, 2022.

[58] Georgios Pavlakos, Ethan Weber, Matthew Tancik, and Angjoo Kanazawa. 他们在电视剧中重建 3D 人体和环境 (The one where they reconstructed 3D humans and environments in TV shows). In ECCV, 2022.

[59] Georgios Pavlakos, Luyang Zhu, Xiaowei Zhou, and Kostas Daniilidis. 从单张彩色图像学习估计 3D 人体姿态和形状 (Learning to estimate 3D human pose and shape from a single color image). In CVPR, 2018.

[60] Owen Pearl, Soyong Shin, Ashwin Godura, Sarah Bergbreiter, and Eni Halilaj. 通过生物力学模型的动态优化融合视频与惯性传感数据 (Fusion of video and inertial sensing data via dynamic optimization of a biomechanical model). Journal of Biomechanics, 155:111617, 2023.

[61] William Peebles and Saining Xie. 基于 Transformer 的可扩展扩散模型 (Scalable diffusion models with transformers). arXiv 预印本 arXiv:2212.09748, 2022.

[62] Xue Bin Peng, Angjoo Kanazawa, Jitendra Malik, Pieter Abbeel, and Sergey Levine. SFV：从视频中强化学习物理技能 (SFV: Reinforcement learning of physical skills from videos). ACM Transactions On Graphics (TOG), 37(6):1–14, 2018.

[63] Jathushan Rajasegaran, Georgios Pavlakos, Angjoo Kanazawa, Christoph Feichtenhofer, and Jitendra Malik. 3D 跟踪与姿态对人体动作识别的益处 (On the benefits of 3D tracking and pose for human action recognition). In CVPR, 2023.

[64] Jathushan Rajasegaran, Georgios Pavlakos, Angjoo Kanazawa, and Jitendra Malik. 使用 3D 表示进行人体跟踪 (Tracking people with 3D representations). In NeurIPS, 2021.

[65] Jathushan Rajasegaran, Georgios Pavlakos, Angjoo Kanazawa, and Jitendra Malik. 通过预测 3D 外观、位置和姿态进行人体跟踪 (Tracking people by predicting 3D appearance, location and pose). In CVPR, 2022.

[66] Davis Rempe, Tolga Birdal, Aaron Hertzmann, Jimei Yang, Srinath Sridhar, and Leonidas J Guibas. HuMoR：用于鲁棒姿态估计的 3D 人体运动模型 (HuMoR: 3D human motion model for robust pose estimation). In ICCV, 2021.

[67] Ergys Ristani, Francesco Solera, Roger Zou, Rita Cucchiara, and Carlo Tomasi. 多目标多相机跟踪的性能度量与数据集 (Performance measures and a data set for multi-target, multi-camera tracking). In ECCV, 2016.

[68] Anshul Shah, Shlok Mishra, Ankan Bansal, Jun-Cheng Chen, Rama Chellappa, and Abhinav Shrivastava. 姿态与关节感知的动作识别 (Pose and joint-aware action recognition). In WACV, 2022.

[69] Jie Song, Xu Chen, and Otmar Hilliges. Human body model fitting by learned gradient descent. In ECCV, 2020.

[70] Yu Sun, Wu Liu, Qian Bao, Yili Fu, Tao Mei, and Michael J Black. Putting people in their place: Monocular regression of 3D people in depth. In CVPR, 2022.

[71] Yu Sun, Yun Ye, Wu Liu, Wenpeng Gao, Yili Fu, and Tao Mei. Human mesh recovery from monocular images via a skeleton-disentangled representation. In ICCV, 2019.

[72] Garvita Tiwari, Dimitrije Antic, Jan Eric Lenssen, Nikolaos Sarafianos, Tony Tung, and Gerard Pons-Moll. Pose-NDF: Modeling human pose manifolds with neural distance fields. In ECCV, 2022.

[73] Vasileios Vasilopoulos, Georgios Pavlakos, Sean L Bowman, J Diego Caporale, Kostas Daniilidis, George J Pappas, and Daniel E Koditschek. Reactive semantic planning in unexplored semantic environments using deep perceptual feedback. IEEE Robotics and Automation Letters, 5(3):4455–4462, 2020.

[74] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, 2017.

[75] Ziniu Wan, Zhengjia Li, Maoqing Tian, Jianbo Liu, Shuai Yi, and Hongsheng Li. Encoder-decoder with multi-level attention for 3D human shape and pose estimation. In ICCV, 2021.

[76] Chung-Yi Weng, Brian Curless, Pratul P Srinivasan, Jonathan T Barron, and Ira Kemelmacher-Shlizerman. HumanNeRF: Free-viewpoint rendering of moving people from monocular video. In CVPR, 2022.

[77] Chao-Yuan Wu, Justin Johnson, Jitendra Malik, Christoph Feichtenhofer, and Georgia Gkioxari. Multiview compressive coding for 3D reconstruction. In CVPR, 2023.

[78] Jiahong Wu, He Zheng, Bo Zhao, Yixin Li, Baoming Yan, Rui Liang, Wenjia Wang, Shipei Zhou, Guosen Lin, Yanwei Fu, Yizhou Wang, and Yonggang Wang. AI Challenger: A large-scale dataset for going deeper in image understanding. arXiv preprint arXiv:1711.06475, 2017.

[79] Yuliang Xiu, Jiefeng Li, Haoyu Wang, Yinghong Fang, and Cewu Lu. Pose Flow: Efficient online pose tracking. In BMVC, 2018.

[80] Xiangyu Xu, Hao Chen, Francesc Moreno-Noguer, Laszlo A Jeni, and Fernando De la Torre. 3D human pose, shape and texture from low-resolution images and videos. PAMI, 2021.

[81] Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao. ViTPose: Simple vision transformer baselines for human pose estimation. In NeurIPS, 2022.

[82] Yuanlu Xu, Song-Chun Zhu, and Tony Tung. DenseRaC: Joint 3D pose and shape estimation by dense render-and-compare. In ICCV, 2019.

[83] Vickie Ye, Georgios Pavlakos, Jitendra Malik, and Angjoo Kanazawa. Decoupling human and camera motion from videos in the wild. In CVPR, 2023.

[84] Ye Yuan, Umar Iqbal, Pavlo Molchanov, Kris Kitani, and Jan Kautz. GLAMR: Global occlusion-aware human mesh recovery with dynamic cameras. In CVPR, 2022.

[85] Andrei Zanfir, Elisabeta Marinoiu, and Cristian Sminchisescu. Monocular 3D pose and shape estimation of multiple people in natural scenes - The importance of multiple scene constraints. In CVPR, 2018.

[86] Wang Zeng, Wanli Ouyang, Ping Luo, Wentao Liu, and Xiaogang Wang. 3D human mesh regression with dense correspondence. In CVPR, 2020.

[87] Hongwen Zhang, Jie Cao, Guo Lu, Wanli Ouyang, and Zhenan Sun. DaNnet: Decompose-and-aggregate network for 3D human shape and pose estimation. In Proceedings of the 27th ACM International Conference on Multimedia, 2019.

[88] Hongwen Zhang, Yating Tian, Yuxiang Zhang, Mengcheng Li, Liang An, Zhenan Sun, and Yebin Liu. PyMAF-X: Towards well-aligned full-body model regression from monocular images. PAMI, 2023.

[89] Hongwen Zhang, Yating Tian, Xinchi Zhou, Wanli Ouyang, Yebin Liu, Limin Wang, and Zhenan Sun. PyMAF: 3D human pose and shape regression with pyramidal mesh alignment feedback loop. In ICCV, 2021.

[90] Jason Y Zhang, Panna Felsen, Angjoo Kanazawa, and Jitendra Malik. Predicting 3D human dynamics from video. In ICCV, 2019.

[91] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In CVPR, 2019.

# 补充材料：“Humans in 4D: Reconstructing and Tracking Humans with Transformers”（4D 人体：基于 Transformer 的人体重建与追踪）

Shubham Goel　Georgios Pavlakos　Jathushan Rajasegaran　Angjoo Kanazawa∗　Jitendra Malik∗

shubham-goel, pavlakos, jathushan, kanazwa @berkeley.edu, malik@eecs.berkeley.edu
加利福尼亚大学伯克利分校（University of California, Berkeley）

我们提供了关于 HMR 2.0 的更多细节，包括我们所使用的架构（第 S.1 节）、数据（第 S.2 节）以及训练流程（第 S.3 节）。此外，我们还描述了姿态预测方面的内容（第 S.4 节），并讨论了用于评估的指标（第 S.5 节）。然后，我们讨论了用于追踪的实验设置（第 S.6 节）和动作识别（第 S.7 节）。最后，我们提供了额外的定性结果（第 S.8 节）。

## S.1. HMR 2.0 架构细节

我们的 HMR 2.0 模型架构基于 ViT（Vision Transformer，视觉 Transformer）图像编码器和一个 Transformer 解码器。我们使用了一个在 2D 关键点定位任务 [25] 上预训练的 ViT-H/16（"huge"，超大版本）。它包含 50 个 Transformer 层，以尺寸为 $2 5 6 \times 1 9 2$ 的图像作为输入，并输出 $1 6 \times 1 2$ 个图像标记（image tokens），每个标记的维度为 1280。我们的 Transformer 解码器采用标准的 Transformer 解码器架构 [23]，共 6 层，每层包含多头自注意力、多头交叉注意力以及前馈块，并使用层归一化 [2]。其隐藏维度为 2048，自注意力和交叉注意力均采用 8 个（64 维）头，前馈 MLP 块中的隐藏维度为 1024。它以单个可学习的 2048 维 SMPL 查询标记（query token）作为输入，并与 $1 6 \times 1 2$ 个图像标记进行交叉注意力。最后，在 Transformer 解码器输出标记上通过一个线性读出层（linear readout）得到姿态 $\theta$、形状 $\beta$，以及相机参数 $\pi$。

## S.2. 数据细节

在训练中，我们沿用先前工作 [10] 的训练数据约定，使用来自 Human3.6M [4]、COCO [13]、MPII [1] 和 MPI-INF-3DHP [18] 的图像。这些构成了我们在正文中所称 HMR 2.0a 版本所采用的训练集。对于最终的 HMR 2.0b 版本，我们额外为来自 AVA [3]、InstaVariety [6] 和 AI Challenger [24] 的图像生成了 SMPL [14] 伪真值拟合。由于 AVA 和 InstaVariety 包含视频，我们分别以 1fps 和 5fps 的采样率采集帧。在伪真值生成中，我们使用 ViTDet [11] 进行边界框检测，ViTPose [25] 进行关键点检测，拟合则通过 ProHMR [10] 完成。我们丢弃检测到的 2D 关键点过少（少于 5 个）且检测置信度较低（阈值 0.5）的检测结果。我们同样丢弃具有不自然身体形状（即身体形状参数超出 $[-3, 3]$）、不自然身体姿态（通过在 AMASS [17] 上构建的逐关节姿态直方图进行判断）以及较大拟合误差（即表明重建未成功）的拟合结果。在训练 HMR 2.0b 模型时，我们以不同概率从各数据集中采样，即 Human3.6M：0.1、MPII：0.1、MPI-INF-3DHP：0.1、AVA：0.15、AI Challenger：0.15、InstaVariety：0.2、COCO：0.2。

## S.3. 训练细节

我们使用 8 张 A100 GPU 训练主模型，有效批大小为 $8 \times 48 = 384$。我们使用 AdamW 优化器 [15]，学习率为 1e-5，$\beta_1 = 0.9$，$\beta_2 = 0.999$，权重衰减为 1e-4。训练持续 1M 次迭代，大约需要六天时间。对于我们的主模型 HMR 2.0b，我们对网络进行端到端训练。然而，对于 HMR 2.0a 变体，ViT 编码器保持冻结，从而可以使用更大的有效批大小 $8 \times 512 = 4096$，学习率为 1e-4，以及更少的训练迭代次数 100K（即大致等价的训练轮数）。

在训练过程中，我们对不同损失项进行加权。$\mathcal{L}_{\mathrm{kp3D}}$、$\mathcal{L}_{\mathrm{kp2D}}$ 和 $\mathcal{L}_{\mathrm{adv}}$ 的权重分别为 0.05、0.01 和 0.0005。$\mathcal{L}_{\mathrm{smpl}}$ 中各项也具有不同权重，$\theta$ 和 $\beta$ 项的权重分别为 0.001 和 0.0005。

## S.4. 姿态预测

## S.5. 评价指标

在我们的评估中，我们使用文献中常用的指标：

3D 姿态（3D Pose）：我们遵循文献 [5]，使用 MPJPE 和 PA-MPJPE。MPJPE 表示平均关节位置误差（Mean Per Joint Position Error），即在以根节点对齐后，所有关节的平均 L2 误差。PA-MPJPE 类似，但在使用普氏分析（Procrustes Alignment）将预测姿态与真实姿态对齐后计算。

2D 姿态（2D Pose）：我们使用文献 [26] 中定义的 PCK，即正确关键点定位百分比（Percentage of Correctly localized Keypoints）。如果某关键点到真实关键点的 L2 距离小于阈值 $t$，则视为正确定位。我们使用不同的阈值（图像尺寸的 @0.05 和 @0.1）报告结果。

跟踪（Tracking）：遵循文献 [20, 21]，我们使用标准的跟踪指标，包括 ID 切换次数（IDs）、MOTA [7]、IDF1 [22] 和 HOTA [16]。

动作识别（Action Recognition）：我们使用 AVA 数据集 [3] 中定义的 mAP 指标报告结果。此外，我们还提供了更细粒度的分析，报告不同动作类别上的结果：涉及物体操作（Object Manipulation, OM）的动作、涉及人物交互（Person Interactions, PI）的动作，以及涉及人体运动（Person Movement, PM）的动作。这些类别的结果同样以 mAP 报告。

## S.6. 使用 PHALP′ 进行跟踪

在正文中，我们将不同的人体网格恢复系统在跟踪这一下游问题上进行了比较（正文表 3）。为此，我们修改了 PHALP 方法 [21]，使得姿态距离在所有模型共享的 SMPL 空间上计算。为了保证比较公平，我们使其他变量与原始 PHALP 保持一致（例如，相同的外观嵌入）。需要注意的是，这种比较对于那些自身不建模外观的基线方法是有利的。最终，我们的 4DHumans 系统采用了基于采样的外观建模头部和我们新的姿态预测方法，从而在 PoseTrack 数据集上取得了最先进的跟踪性能（正文表 4）。为了建模外观，我们通过将网格上可见的点投影到输入图像上，并从对应像素处采样颜色，来为其赋予纹理。

## S.7. 动作识别

作为评估 3D 人体重建质量的另一种方式，我们在 AVA 数据集上的动作识别下游任务中对各种人体网格恢复系统进行了评估（关于任务定义的更多细节，请参阅 [19]）。具体而言，我们采用 [19] 中的跟踪片段，这些片段是通过在 Kinetics [8] 和 AVA [3] 数据集上运行 PHALP [21] 生成的。然后，我们替换来自不同人体网格恢复模型（即 PyMAF [28]、PyMAF-X [27]、PARE [9]、CLIFF [12]、HMAR [21]、HMR 2.0）的姿态，并在动作识别任务上评估它们的性能。在这种仅使用姿态的设定下，动作识别模型只能访问 3D 姿态（以 SMPL 格式表示）和 3D 位置，并被训练用于预测每个人的动作。为了进行公平的比较并使每个 3D 姿态回归器达到最佳性能，我们针对每种 3D 姿态方法专门重新训练了动作识别模型。

## S.8. 额外的定性结果

我们已经在正文以及项目网页的视频中提供了大量的 HMR 2.0 定性结果。在这里，我们提供额外的实验结果，包括与我们最接近的竞争方法的比较（图 S.1），以及在各种具有挑战性的情况下展示我们的结果，包括成功案例（图 S.2）和失败案例（图 S.3）。

## 参考文献

[1] Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler, and Bernt Schiele. 2D human pose estimation: New benchmark and state of the art analysis. In CVPR, 2014.

[2] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

[3] Chunhui Gu, Chen Sun, David A Ross, Carl Vondrick, Caroline Pantofaru, Yeqing Li, Sudheendra Vijayanarasimhan, George Toderici, Susanna Ricco, Rahul Sukthankar, Cordelia Schmid, and Jitendra Malik. AVA: A video dataset of spatio-temporally localized atomic visual actions. In CVPR, 2018.

[4] Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu. Human3.6M: Large scale datasets and predictive methods for 3D human sensing in natural environments. PAMI, 2013.

[5] Angjoo Kanazawa, Michael J Black, David W Jacobs, and Jitendra Malik. End-to-end recovery of human shape and pose. In CVPR, 2018.

Input

PyMAF-X

PARE

CLIFF

HMR 2.0

![](images/221ed005cd2473f45345b9f0661129cd3ec5baed588f157835e9dc7b86fd3342.jpg)  
图 S.1：我们方法与最先进方法的定性比较。我们将 HMR 2.0 与我们最接近的竞争对手 PyMAF-X [27]、PARE [9] 和 CLIFF [12] 进行比较。对于每个示例，我们展示了输入图像以及每个方法的结果（包括正面视图和侧面视图）。HMR 2.0 在多种场景下都显著更加鲁棒，包括具有异常姿态、异常视角以及人物之间严重重叠的图像。

Input

Front view

Side view

Top view

Input

Front view

Side view

Top view

![](images/0ed9c6838d4cbdcd7a531d459aa30e168657760ed94caffe0cfad9a861b62df4.jpg)  
图 S.2：我们方法在具有挑战性示例上的定性结果。对于每个示例，我们展示了输入图像、重建叠加结果、侧面视图和顶部视图。这些示例包括异常姿态、异常视角、紧密交互的人物、极端截断与遮挡，还有模糊的图像。

Front view

Side view

![](images/929255fcdc139e1a4b3563f53f3feba3e4d12cef76c51df491a8ef8c4c639765.jpg)

Top view

Input

Front view

![](images/ab83b9508ff6fbcc470a0cf9fa3de7a2d4ac7b32466acd8f4c4b6d9ca03d325f.jpg)

Side view

Top view

![](images/c5dbd2c40e7a4bb5f0bc6f900c6ead5d48d98b62d6b98128a94c23320f3d156c.jpg)  
图 S.3：使用 HMR 2.0 进行单帧三维人体重建的失败案例。尽管我们的方法增强了鲁棒性，但我们观察到 HMR 2.0 在以下情况下偶尔会恢复出错误的重建：非常异常的关节姿态（第一行）、严重的人物交互（第二行），以及对身体不同部位而言极具挑战性的深度排序（第三行）。

[6] Angjoo Kanazawa, Jason Y Zhang, Panna Felsen, and Jitendra Malik. Learning 3D human dynamics from video. In CVPR, 2019.

[7] Rangachar Kasturi, Dmitry Goldgof, Padmanabhan Soundararajan, Vasant Manohar, John Garofolo, Rachel Bowers, Matthew Boonstra, Valentina Korzhova, and Jing Zhang. Framework for performance evaluation of face, text, and vehicle detection and tracking in video: Data, metrics, and protocol. PAMI, 2008.

[8] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, Mustafa Suleyman, and Andrew Zisserman. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.

[9] Muhammed Kocabas, Chun-Hao P Huang, Otmar Hilliges, and Michael J Black. PARE: Part attention regressor for 3D human body estimation. In ICCV, 2021.

[10] Nikos Kolotouros, Georgios Pavlakos, Dinesh Jayaraman, and Kostas Daniilidis. Probabilistic modeling for human mesh recovery. In ICCV, 2021.

[11] Yanghao Li、Hanzi Mao、Ross Girshick 和 Kaiming He。Exploring plain vision transformer backbones for object detection（《探索用于目标检测的纯视觉 Transformer 主干网络》）。发表于 ECCV，2022 年。

[12] Zhihao Li、Jianzhuang Liu、Zhensong Zhang、Songcen Xu 和 Youliang Yan。CLIFF: Carrying location information in full frames into human pose and shape estimation（《CLIFF：将完整帧中的位置信息引入人体姿态与形状估计》）。发表于 ECCV，2022 年。

[13] Tsung-Yi Lin、Michael Maire、Serge Belongie、James Hays、Pietro Perona、Deva Ramanan、Piotr Dollar 和 C Lawrence´ Zitnick。Microsoft COCO: Common objects in context（《Microsoft COCO：上下文中的常见物体》）。发表于 ECCV，2014 年。

[14] Matthew Loper、Naureen Mahmood、Javier Romero、Gerard Pons-Moll 和 Michael J Black。SMPL: A skinned multiperson linear model（《SMPL：一种蒙皮的多人线性模型》）。ACM transactions on graphics (TOG)，34(6)：1–16，2015 年。

[15] Ilya Loshchilov 和 Frank Hutter。Decoupled weight decay regularization（《解耦的权重衰减正则化》）。arXiv 预印本 arXiv:1711.05101，2017 年。

[16] Jonathon Luiten、Aljosa Osep、Patrick Dendorfer、Philip Torr、Andreas Geiger、Laura Leal-Taixe´ 和 Bastian Leibe。HOTA: A higher order metric for evaluating multi-object tracking（《HOTA：用于评估多目标跟踪的高阶指标》）。IJCV，2021 年。

[17] Naureen Mahmood、Nima Ghorbani、Nikolaus F Troje、Gerard Pons-Moll 和 Michael J Black。AMASS: Archive of motion capture as surface shapes（《AMASS：以表面形状形式存档的动作捕捉数据库》）。发表于 ICCV，2019 年。

[18] Dushyant Mehta、Helge Rhodin、Dan Casas、Pascal Fua、Oleksandr Sotnychenko、Weipeng Xu 和 Christian Theobalt。Monocular 3D human pose estimation in the wild using improved CNN supervision（《利用改进的 CNN 监督在野外进行单目 3D 人体姿态估计》）。发表于 3DV，2017 年。

[19] Jathushan Rajasegaran、Georgios Pavlakos、Angjoo Kanazawa、Christoph Feichtenhofer 和 Jitendra Malik。On the benefits of 3D tracking and pose for human action recognition（《3D 跟踪与姿态对人类行为识别的好处》）。发表于 CVPR，2023 年。

[20] Jathushan Rajasegaran、Georgios Pavlakos、Angjoo Kanazawa 和 Jitendra Malik。Tracking people with 3D representations（《使用 3D 表征进行人员跟踪》）。发表于 NeurIPS，2021 年。

[21] Jathushan Rajasegaran、Georgios Pavlakos、Angjoo Kanazawa 和 Jitendra Malik。Tracking people by predicting 3D appearance, location and pose（《通过预测 3D 外观、位置和姿态进行人员跟踪》）。发表于 CVPR，2022 年。

[22] Ergys Ristani、Francesco Solera、Roger Zou、Rita Cucchiara 和 Carlo Tomasi。Performance measures and a data set for multi-target, multi-camera tracking（《多目标多摄像头跟踪的性能度量与数据集》）。发表于 ECCV，2016 年。

[23] Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N Gomez、Łukasz Kaiser 和 Illia Polosukhin。Attention is all you need（《注意力即你所需的一切》）。发表于 NIPS，2017 年。

[24] Jiahong Wu、He Zheng、Bo Zhao、Yixin Li、Baoming Yan、Rui Liang、Wenjia Wang、Shipei Zhou、Guosen Lin、Yanwei Fu、Yizhou Wang 和 Yonggang Wang。AI Challenger: A large-scale dataset for going deeper in image understanding（《AI Challenger：用于深入图像理解的大规模数据集》）。arXiv 预印本 arXiv:1711.06475，2017 年。

[25] Yufei Xu、Jing Zhang、Qiming Zhang 和 Dacheng Tao。ViTPose: Simple vision transformer baselines for human pose estimation（《ViTPose：用于人体姿态估计的简单视觉 Transformer 基线方法》）。发表于 NeurIPS，2022 年。

[26] Yi Yang 和 Deva Ramanan。Articulated human detection with flexible mixtures of parts（《使用灵活部件混合的铰接式人体检测》）。PAMI，2012 年。

[27] Hongwen Zhang、Yating Tian、Yuxiang Zhang、Mengcheng Li、Liang An、Zhenan Sun 和 Yebin Liu。PyMAF-X: Towards well-aligned full-body model regression from monocular images（《PyMAF-X：基于单目图像的全身模型回归对齐》）。PAMI，2023 年。

[28] Hongwen Zhang, Yating Tian, Xinchi Zhou, Wanli Ouyang, Yebin Liu, Limin Wang, and Zhenan Sun. PyMAF：基于金字塔网格对齐反馈循环的 3D 人体姿态与形状回归. In *ICCV* (国际计算机视觉会议), 2021.