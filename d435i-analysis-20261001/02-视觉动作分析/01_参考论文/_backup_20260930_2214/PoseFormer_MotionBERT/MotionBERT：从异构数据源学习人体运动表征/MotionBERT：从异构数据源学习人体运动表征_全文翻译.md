# MotionBERT：学习人体运动表征的统一视角

Wentao Zhu<sup>1</sup> Xiaoxuan Ma<sup>1</sup> Zhaoyang Liu<sup>2</sup> Libin Liu<sup>1</sup> Wayne Wu<sup>2</sup> Yizhou Wang<sup>1†</sup>

<sup>1</sup> 北京大学 <sup>2</sup> 上海人工智能实验室

{wtzhu,maxiaoxuan,libin.liu,yizhou.wang}@pku.edu.cn

{zyliumy,wuwenyan0503}@gmail.com

## 摘要

我们提出了一种统一的视角，通过从大规模且异构的数据资源中学习人体运动表征来处理各种以人为中心的视频任务。具体而言，我们提出了一个预训练阶段，在该阶段中训练一个运动编码器，从含有噪声的部分二维观测中恢复出底层的三维运动。通过这种方式获得的运动表征融合了关于人体运动的几何、运动学以及物理知识，并且可以便捷地迁移到多个下游任务中。我们使用双流时空 Transformer（DSTformer）神经网络来实现该运动编码器。它能够全面且自适应地捕捉骨骼关节之间长距离的时空关系，并以从零训练的方式在三维姿态估计任务上取得了迄今为止最低的误差。此外，我们提出的框架仅需通过一个简单的回归头（1-2 层）对预训练的运动编码器进行微调，即可在全部三个下游任务上达到最先进（state-of-the-art）的性能，这表明所学习到的运动表征具有高度的通用性。代码与模型已在 https://motionbert.github.io/ 开源。

## 1. 引言

感知并理解人类活动长期以来都是机器智能领域的核心追求。为实现这一目标，研究者们定义了多种任务，从视频中估计以人为中心的语义标签，例如骨骼关键点 [14, 35]、动作类别 [64, 123] 以及表面网格 [46, 71]。尽管每个任务都取得了显著进展，但这些任务往往是被彼此孤立地建模的，而非被视为相互关联的问题。例如，时空图卷积网络（ST-GCN）已被应用于在三维姿态估计 [13, 116] 与动作识别 [96, 123] 中建模人体关节的时空关系，但它们之间的联系尚未被充分挖掘。直观上，这些模型理应都已学会了识别典型的人体运动模式，即便它们是为不同问题所设计的。然而，当前方法未能充分挖掘并利用这些跨任务之间的共性。理想情况下，我们应当能够开发出一种统一的、以人为中心的视频表征，从而可以在所有相关任务之间共享。

![](images/2667e0cd5fae1cacf4ff392a8452bf486a1df647537aa2697f1697dd8b78c350.jpg)  
图 1. 框架概览。我们利用一个运动编码器，通过从被损坏的二维骨骼序列中恢复三维人体运动来学习人体运动表征。为了适配不同的下游任务，我们使用一个线性层或一个简单的 MLP（多层感知机）对预训练的运动表征进行微调。

数据资源的异构性是开发此类表征的一个重大挑战。动作捕捉（Mocap）系统 [38, 76] 能够提供由标记点和传感器获取的高保真 3D 运动数据，但采集到的视频外观通常局限于简单的室内场景。动作识别数据集提供了动作语义的标注，但它们要么不包含人体姿态标签 [16, 95]，要么仅包含有限的日常活动动作 [63, 64, 93]。相比之下，野外（in-the-wild）人体视频提供了丰富多样的外观和运动。然而，获取精确的 2D 姿态标注需要耗费大量人力 [3]，而获取真值（ground-truth, GT）3D 关节位置几乎不可能。因此，现有研究大多聚焦于使用单一类型的人体运动数据来完成特定任务，因而无法享受其他数据资源带来的优势。

在本工作中，我们为学习人体运动表征提供了一种新视角。其核心思想是：我们能够以统一的方式从异构数据资源中学习一种通用的人体运动表征，并利用该表征以统一的方式应对不同的下游任务。我们提出了一个包含预训练和微调两阶段的框架，如图 1 所示。在预训练阶段，我们从多样化的运动数据源中提取 2D 骨架序列，并对其施加随机掩码（mask）和噪声进行破坏。随后，我们训练运动编码器从被破坏的 2D 骨架中恢复出 3D 运动。这一具有挑战性的前置任务本质上要求运动编码器能够 i) 从时序运动中推断潜在的 3D 人体结构；ii) 恢复出错误和缺失的观测。通过这种方式，运动编码器隐式地捕捉了人体运动的常识，例如关节连接关系、解剖学约束以及时序动态。实际实现中，我们提出了双流时空 Transformer（Dual-stream Spatio-temporal Transformer, DSTformer）作为运动编码器，以捕捉骨架关键点之间的长程关系。我们认为，从大规模、多样化数据资源中学习到的运动表征可以共享应用于不同的下游任务，并提升它们的性能。因此，对于每个下游任务，我们使用任务特定的训练数据和监督信号，通过一个简单的回归头对预训练的运动表征进行适配。

综上所述，本工作的贡献可归纳为三个方面：1) 我们提供了一个新的视角，即通过学习人体运动表示这一共享框架来解决各种以人为中心的视频任务。2) 我们提出了一种预训练方法，以利用大规模但异构的人体运动资源，并学习可泛化的人体运动表示。我们的方法能够同时利用三维动捕数据的精度和野外 RGB 视频的多样性。3) 我们设计了一个具有级联时空自注意力块的双流 Transformer 网络，可作为人体运动建模的通用骨干网络。实验表明，上述设计使得所学到的人体运动表示具有高度通用性，可迁移到多个下游任务，性能优于针对特定任务的当前最优方法。

## 2. 相关工作

**人体运动表示学习。** 早期工作使用隐马尔可夫模型 [53, 108] 和图模型 [51, 99] 来建模人体运动。Kanazawa 等人 [42] 设计了时间编码器和幻觉器来学习三维人体动态的表示。Zhang 等人 [132] 以自监督方式预测未来的三维动态。Sun 等人 [102] 进一步结合动作记忆库引入动作标签。从动作识别的角度来看，研究者们设计了多种自监督预训练任务来学习运动表示，包括未来预测 [100]、拼图 [60]、骨骼对比学习 [107]、速度变化 [101]、跨视角一致性 [62] 以及对比-重建 [117]。类似的技术也被探索应用于运动评估 [33, 85] 和运动重定向 [119, 139] 等任务。这些方法利用同构的运动数据，设计相应的预训练任务，并将其应用于特定的下游任务。本工作中，我们提出了一种统一的预训练-微调框架，以整合异构数据资源，并展示了其在各种下游任务中的通用性。

**三维人体姿态估计。** 从单目 RGB 视频中恢复三维人体姿态是一个经典问题，相关方法可分为两大类。第一类是直接利用 CNN 从图像中估计三维姿态 [82, 104, 136]。然而，这些方法的局限性在于，由于当前数据采集技术的限制，三维姿态精度与外观多样性之间存在权衡。第二类方法是先提取二维姿态，再通过独立的神经网络将估计的二维姿态提升到三维。该提升过程可通过全连接网络 [29, 78]、时间卷积网络 (TCN) [22, 89]、图卷积网络 (GCN) [13, 28, 116] 以及 Transformer [56, 94, 134, 135] 实现。我们的框架基于第二类方法，因为我们使用所提出的 DSTformer 来完成二维到三维的姿态提升。

Skeleton-based Action Recognition（基于骨骼的动作识别）。开创性工作 [74, 115, 127] 指出了动作识别与人体姿态估计之间的内在联系。为了建模人体关节之间的时空关系，先前的研究主要采用 LSTM [98, 138] 和 GCN [21, 55, 68, 96, 123]。最近，PoseConv3D [32] 提出将 3D-CNN 应用于堆叠的 2D 关节热力图，并取得了改进的效果。除了全监督动作识别任务外，NTU-RGB+D-120 [64] 还引起了人们对具有挑战性的单样本（one-shot）动作识别问题的关注。为此，SL-DML [81] 将深度度量学习应用于多模态信号。Sabater 等人 [92] 使用 TCN 探索治疗场景中的单样本识别问题。我们证明预训练的运动表示可以很好地泛化到动作识别任务，预训练-微调框架是解决单样本挑战的合适方案。

Human Mesh Recovery（人体网格恢复）。基于参数化人体模型（如 SMPL [71]），许多研究工作 [41, 75, 83, 122, 133] 专注于从单张图像回归人体网格。SPIN [48] 还在训练循环中

![](images/060922f440e32944b5b4bd9338d8b222b83916ea3ffc1a6455f2e4ad46247f92.jpg)  
图 2. 模型架构。我们提出 Dual-stream Spatio-temporal Transformer（DSTformer）作为人体运动建模的通用骨干网络。DSTformer 由 N 个双流融合模块组成。每个模块包含两个分支，分别用于空间或时间 MHSA 和 MLP。空间 MHSA 建模同一时间步内不同关节之间的连接，而时间 MHSA 建模单个关节的运动。

## 3.2. Network Architecture

中加入了将人体模型拟合到 2D 关节。尽管这些方法在单帧结果上表现良好，但当应用于视频时会产生抖动和不稳定的结果 [46, 130]。为了改善其时间连贯性，PoseBERT [8] 和 SmoothNet [130] 提出对单帧预测采用去噪和平滑模块。一些工作 [24, 42, 46, 106] 以视频片段作为输入以利用时间线索。另一个常见问题是，配对的图像和 GT 网格大多在受控场景中采集，这限制了上述方法的泛化能力。为此，Pose2Mesh [25] 提出首先使用现成的姿态估计器提取 2D 骨骼，然后将其提升到 3D 网格顶点。我们的方法与最先进的人体网格恢复方法互补，可以通过预训练的运动表示进一步改善它们的时间连贯性。

## 3. Method

## 3.1. Overview

如第 1 节所述，我们的方法包含两个阶段，即统一预训练和任务特定微调。在第一阶段，我们训练一个运动编码器来完成 2D 到 3D 的提升任务，其中我们使用所提出的 DSTformer 作为骨干网络。在第二阶段，我们在下游任务上微调预训练的运动编码器和一些新增的层。我们在预训练和微调阶段都采用 2D 骨架序列作为输入，因为它们可以从各种运动来源中可靠地提取 [3, 10, 76, 86, 103]，并且对变化具有更强的鲁棒性 [19, 32]。已有研究表明，在不同下游任务中使用 2D 骨架序列是有效的 [25, 32, 89, 109]。我们将首先介绍 DSTformer 的架构，然后详细描述训练方案。

图 2 展示了 2D 到 3D 提升的网络架构。给定输入的 2D 骨架序列 $\mathbf { x } \in \mathbb { R } ^ { T \times J \times C _ { \mathrm { i n } } }$，我们首先将其投影到高维特征 ${ \bf F } ^ { 0 } \in \mathbb { R } ^ { T \times J \times C _ { 1 } }$，然后为其添加可学习的空间位置编码 $\mathbf { P } _ { \mathrm { p o s } } ^ { \mathrm { S } } \in \mathbb { R } ^ { 1 \times J \times C _ { \mathrm { f } } }$ 和时间位置编码 $\mathbf { P } _ { \mathrm { p o s } } ^ { \mathrm { T } } \bar { \in } \mathbb { R } ^ { \sum _ { T \times 1 \times C _ { \mathrm { f } } } }$。随后，我们使用序列到序列模型 DSTformer 来计算 $\mathbf { F } ^ { i } \in \bar { \mathbb { R } } ^ { T \times J \times C _ { \mathrm { f } } } ( i \mathbf { \theta } = 1 , \dots , N )$，其中 N 为网络深度。我们对 $\mathbf { F } ^ { N }$ 应用带有 tanh 激活函数 [30] 的线性层以计算运动表示 $\mathbf { E } \in \mathbb { R } ^ { \bar { T } \times J \times C _ { \mathrm { e } } }$。最后，我们对 E 应用线性变换以估计 3D 运动 $\hat { \hat { \mathbf { X } } } \in \mathbb { R } ^ { T \times J \times C _ { \mathrm { o u t } } }$。其中，$T$ 表示序列长度，J 表示身体关节点的数量。$C _ { \mathrm { i n } } , C _ { \mathrm { f } } , C _ { \mathrm { e } } ,$ 和 $C _ { \mathrm { { o u t } } }$ 分别表示输入、特征、嵌入和输出的通道数。我们首先介绍 DSTformer 的基本构建块，即带有多头自注意力（MHSA）的空间块和时间块，然后解释 DSTformer 的架构设计。

空间块。空间多头自注意力（S-MHSA）旨在建模同一时间步内各关节点之间的关系。其定义如下

$$
\begin{array} { r } { \mathrm { S } \mathrm { - } \mathbf { M } \mathrm { H S A } ( \mathbf { Q } _ { \mathrm { S } } , \mathbf { K } _ { \mathrm { S } } , \mathbf { V } _ { \mathrm { S } } ) = [ \mathrm { h e a d } _ { 1 } ; \dots ; \mathrm { h e a d } _ { h } ] \mathbf { W } _ { \mathrm { S } } ^ { P } , } \\ { \mathrm { h e a d } _ { i } = \mathrm { s o f t m a x } ( \frac { \mathbf { Q } _ { \mathrm { S } } ^ { i } ( \mathbf { K } _ { \mathrm { S } } ^ { i } ) ^ { \prime } } { \sqrt { d _ { K } } } ) \mathbf { V } _ { \mathrm { S } } ^ { i } , } \end{array}\tag{1}
$$

其中 $\mathbf { W } _ { \mathrm { S } } ^ { P }$ 是投影参数矩阵，$h$ 是注意力头的数量，$i \in { 1 , \dots , h }$，$\prime$ 表示矩阵转置。我们利用自注意力从每个头的每帧空间特征 $\mathbf { F } _ { \mathrm { S } } ^ { \mathrm{ } } \in \mathbb { R } ^ { J \times C _ { \mathrm { e } } }$ 中获得查询 $\mathbf { Q } _ { \mathrm { S } }$、键 $\mathbf { K } _ { \mathrm { S } }$ 和值 ${ \bf V } _ { \mathrm { S } }$，

$$
\mathbf { Q } _ { \mathrm { s } } ^ { i } = \mathbf { F } _ { \mathrm { s } } \mathbf { W } _ { \mathrm { s } } ^ { ( Q , i ) } , \mathbf { K } _ { \mathrm { s } } ^ { i } = \mathbf { F } _ { \mathrm { s } } \mathbf { W } _ { \mathrm { s } } ^ { ( K , i ) } , \mathbf { V } _ { \mathrm { s } } ^ { i } = \mathbf { F } _ { \mathrm { s } } \mathbf { W } _ { \mathrm { s } } ^ { ( V , i ) } ,\tag{2}
$$

其中 $\mathbf { W } _ { \mathrm { S } } ^ { ( Q , i ) } , \mathbf { W } _ { \mathrm { S } } ^ { ( K , i ) } , \mathbf { W } _ { \mathrm { S } } ^ { ( V , i ) }$ 是投影矩阵，$d _ { K }$ 是 $\mathbf { K } _ { \mathrm { S } }$ 的特征维度。我们将 S-MHSA 并行地应用于不同时间步的特征。对 S-MHSA 的结果使用残差连接和层归一化（LayerNorm），其输出再送入多层感知机（MLP），随后再进行残差连接和层归一化，遵循文献 [112] 的做法。我们将包含 MHSA、LayerNorm、MLP 以及残差连接的整个空间块记为 。

**时间块。** 时间多头自注意力（T-MHSA）旨在对单个人体关节在不同时间步之间的关系进行建模。其计算过程与 S-MHSA 类似，不同之处在于 MHSA 应用于每关节时间特征 $\mathbf { F } _ { \mathrm { T } } \in \mathbb { R } ^ { T \times C _ { \mathrm { e } } }$，并在空间维度上进行并行计算。

$$
\begin{array} { r } { \mathrm { T } \mathbf { \cdot } \mathbf { M } \mathrm { H S A } ( \mathbf { Q } _ { \mathrm { T } } , \mathbf { K } _ { \mathrm { T } } , \mathbf { V } _ { \mathrm { T } } ) = [ \mathrm { h e a d } _ { 1 } ; \dots ; \mathrm { h e a d } _ { \mathrm { h } } ] \mathbf { W } _ { \mathrm { T } } ^ { P } , } \\ { \mathrm { h e a d } _ { i } = \operatorname { s o f t m a x } ( \frac { \mathbf { Q } _ { \mathrm { T } } ^ { i } ( \mathbf { K } _ { \mathrm { T } } ^ { i } ) ^ { \prime } } { \sqrt { d _ { K } } } ) \mathbf { V } _ { \mathrm { T } } ^ { i } , } \end{array}\tag{3}
$$

其中 $i \in { 1 , \ldots , h , \mathbf { Q } _ { \mathrm { { T } } } , \mathbf { K } _ { \mathrm { { T } } } , \mathbf { V } _ { \mathrm { { T } } } }$ 的计算方式与公式 2 类似。我们将整个时序块记为 $\tau$

**双流时空 Transformer（Dual-stream Spatio-temporal Transformer）**。鉴于空间多头自注意力和时序多头自注意力能够分别建模帧内和帧间的人体关节交互，我们将这些基本构建模块组合起来以融合流（flow）中的空间与时序信息。我们基于以下假设设计了一种双流架构：1) 两个分支都应具备建模完整时空上下文的能力；2) 每个分支应专注于不同的时空特征；3) 两个分支应当被融合在一起，且融合权重应根据输入的时空特征动态平衡。

因此，我们将空间和时序多头自注意力块以不同的顺序堆叠，形成两条并行的计算分支。两条分支的输出特征通过一个注意力回归器（attention regressor）预测的自适应权重进行融合。该双流融合模块重复 $N$ 次：

$$
\mathbf { F } ^ { i } = \alpha _ { \mathrm { S T } } ^ { i } \circ \mathcal { T } _ { 1 } ^ { i } ( S _ { 1 } ^ { i } ( \mathbf { F } ^ { i - 1 } ) ) + \alpha _ { \mathrm { T S } } ^ { i } \circ S _ { 2 } ^ { i } ( \mathcal { T } _ { 2 } ^ { i } ( \mathbf { F } ^ { i - 1 } ) ) , \quad i \in 1 , \dots , N ,\tag{4}
$$

其中 $\mathbf { F } ^ { i }$ 表示第 $i$ 层的特征嵌入，$\circ$ 表示逐元素乘积。$S$ 和 $\tau$ 块的顺序如图 $^ { 2 , }$ 所示，不同的块之间不共享权重。自适应融合权重 $\alpha _ { \mathrm { S T } } , \alpha _ { \mathrm { T S } } \in \mathbb { R } ^ { N \times T \times J }$ 由下式给出：

$$
\alpha _ { \mathrm { S T } } ^ { i } , \alpha _ { \mathrm { T S } } ^ { i } = \mathrm { s o f t m a x } ( \mathcal { W } ( [ \mathcal { T } _ { 1 } ^ { i } ( S _ { 1 } ^ { i } ( \mathbf { F } ^ { i - 1 } ) ) , S _ { 2 } ^ { i } ( \mathcal { T } _ { 2 } ^ { i } ( \mathbf { F } ^ { i - 1 } ) ) ] ) ) ,\tag{5}
$$

其中 $\mathcal{W}$ 是一个可学习的线性变换，$[,]$ 表示拼接操作。

## 3.3. 统一预训练（Unified Pretraining）

我们解决了设计统一预训练框架时的两个关键挑战：1) 如何通过一个通用的预训练任务（pretext task）来学习强大的运动表征。2) 如何利用大规模但异构的、格式各异的人体运动数据。

对于第一个挑战，我们借鉴语言 [12, 30, 90] 和视觉 [7, 36] 建模中的成功实践来构建监督信号，即遮蔽部分输入并利用编码后的表征来重建整个输入。需要注意的是，这种"完形填空"任务在人体运动分析中天然存在——从二维视觉观测中恢复丢失的深度信息，即三维人体姿态估计。受此启发，我们利用大规模的三维动作捕捉（mocap）数据 [76] 并设计了一个二维到三维提升的预训练任务。我们首先通过正交投影三维运动来提取二维骨架序列 x，然后通过对 x 随机遮蔽并添加噪声来生成被破坏的二维骨架序列，这也与二维检测结果相似，因为后者包含遮挡、检测失败和误差。我们以一定概率同时施加关节级和帧级的遮蔽。我们使用上述运动编码器获得运动表征 E 并重建三维运动 $\hat{\mathbf{X}}$。然后我们计算 $\hat{\mathbf{X}}$ 与真实值（GT）三维运动 $\mathbf{X}$ 之间的关节损失 $\mathcal{L}_{\mathrm{3D}}$。我们还遵循已有工作 [89, 134] 加入速度损失 $\mathcal{L}_{0}$。三维重建损失由下式给出：

$$
\mathcal{L}_{\mathrm{3D}} = \sum_{t=1}^{T} \sum_{j=1}^{J} \| \hat{\mathbf{X}}_{t,j} - \mathbf{X}_{t,j} \|_{2}, \quad \mathcal{L}_{0} = \sum_{t=2}^{T} \sum_{j=1}^{J} \| \hat{\mathbf{O}}_{t,j} - \mathbf{O}_{t,j} \|_{2},
$$

其中 $\begin{array}{r} \hat{\mathbf{O}}_{t} = \hat{\mathbf{X}}_{t} - \hat{\mathbf{X}}_{t-1}, \mathbf{O}_{t} = \mathbf{X}_{t} - \mathbf{X}_{t-1}. \end{array}$

(6)

对于第二个挑战，我们注意到二维骨架可以作为一种通用媒介，因为它们可以从各种运动数据源中提取。我们进一步将野外 RGB 视频纳入二维到三维提升框架中进行统一预训练。对于 RGB 视频，二维骨架 x 可以通过人工标注 [3] 或二维姿态估计器 [14, 103] 获得，并且所提取二维骨架的深度通道天然就是"被遮蔽"的。类似地，我们加入额外的遮蔽和噪声来退化 x（如果 x 已经包含检测噪声，则仅施加遮蔽）。由于这些数据没有三维运动真实值 $\mathbf{X}$，我们使用加权的二维重投影损失，其计算如下

$$
\mathcal { L } _ { \mathrm { 2 D } } = \sum _ { t = 1 } ^ { T } \sum _ { j = 1 } ^ { J } \delta _ { t , j } \| \hat { \mathbf { x } } _ { t , j } - \mathbf { x } _ { t , j } \| _ { 2 } ,\tag{7}
$$

其中 $\hat { \bf x }$ 是估计的三维运动 $\hat { \bf X }$ 的二维正交投影，$\pmb { \delta } \in \mathbf { \bar { \mathbb { R } } } ^ { \bar { T } \times J }$ 由可见性标注或二维检测置信度给出。

总预训练损失由下式计算：

$$
\small \mathcal { L } = \underbrace { \mathcal { L } _ { \mathrm { 3 D } } + \lambda _ { \mathrm { 0 } } \mathcal { L } _ { \mathrm { 0 } } } _ { \mathrm{3 D \ 数 \ 据 } } + \underbrace { \mathcal { L } _ { \mathrm { 2 D } } } _ { \mathrm{2 D \ 数 \ 据 } } ,\tag{8}
$$

其中 $\lambda _ { \mathrm { 0 } }$ 是用于平衡各项损失的常数系数。

## 3.4. 任务特定微调

所学到的特征嵌入 E 作为一种具备三维感知与时序感知能力的人体运动表示。对于下游任务，我们采用极简设计理念，即实现一个浅层的下游网络，并在不使用任何额外技巧的情况下进行训练。实际中，我们使用一个额外的线性层或一个具有单隐层的 MLP（多层感知机）。随后，我们对整个网络进行端到端微调。

**三维姿态估计。** 由于我们采用二维到三维提升作为预训练任务，因此直接复用整个预训练网络即可。在微调阶段，输入的二维骨架从视频中估计得到，且不使用额外的掩码或噪声。

**基于骨架的动作识别。** 我们直接对不同人物与时间步进行全局平均池化，所得结果输入到一个具有单隐层的 MLP 中。该网络使用交叉熵分类损失进行训练。对于单样本学习（one-shot learning），我们在池化后的特征之后接入一个线性层以提取片段级（clip-level）动作表示。4.4 节将详细介绍单样本学习的设置。

## 人体网格恢复

我们使用 SMPL [71]（Skinned Multi-Person Linear model）模型来表示人体网格并回归其参数。SMPL 模型由姿态参数 $\theta \in \mathbb{R}^{72}$ 和体型参数 $\beta \in \mathbb{R}^{10}$ 组成，并按下式计算三维网格：

$$
\mathcal{M}(\boldsymbol{\theta}, \boldsymbol{\beta}) \in \mathbb{R}^{6890 \times 3}.
$$

为了对每一帧的**姿态**参数进行回归，我们将运动嵌入 $E$ 输入到一个具有一层隐藏层的 MLP（多层感知机）中，得到 $\hat{\boldsymbol{\theta}} \in \mathbb{R}^{T \times 72}$。对于**体型**参数的估计，考虑到视频序列中人体体型应保持一致性，我们首先沿时间维度对 $E$ 进行平均池化（average pooling），然后将其输入到另一个 MLP 中，回归得到一个单一的 $\hat{\beta}$，再将其扩展到整个序列，得到 $\hat{\boldsymbol{\beta}} \in \mathbb{R}^{T \times 10}$。体型 MLP 与姿态回归 MLP 具有相同的网络结构，并且它们分别以平均体型和平均姿态进行初始化。总损失函数计算如下：

$$
\mathcal{L} = \lambda_{3\mathrm{D}}^{\mathrm{m}} \mathcal{L}_{3\mathrm{D}}^{\mathrm{m}} + \lambda_{\theta} \mathcal{L}_{\theta} + \lambda_{\beta} \mathcal{L}_{\beta} + \lambda_{\mathrm{norm}} \mathcal{L}_{\mathrm{norm}} + \lambda_{\mathrm{O}}^{\mathrm{m}} \mathcal{L}_{\mathrm{O}}^{\mathrm{m}}, \tag{9}
$$

其中各项的计算方式为：

$$
\begin{aligned}
\mathcal{L}_{3\mathrm{D}}^{\mathrm{m}} &= \|\hat{\mathbf{X}}^{\mathrm{m}} - \mathbf{X}^{\mathrm{m}}\|_1, \quad \mathcal{L}_{\boldsymbol{\theta}} = \|\hat{\boldsymbol{\theta}} - \boldsymbol{\theta}\|_1, \quad \mathcal{L}_{\boldsymbol{\beta}} = \|\hat{\boldsymbol{\beta}} - \boldsymbol{\beta}\|_1, \\
\mathcal{L}_{\mathrm{norm}} &= \|\hat{\boldsymbol{\theta}}\|_2 + \|\hat{\boldsymbol{\beta}}\|_2, \quad \mathcal{L}_{\mathrm{O}}^{\mathrm{m}} = \|\hat{\mathbf{O}}^{\mathrm{m}} - \mathbf{O}^{\mathrm{m}}\|_2.
\end{aligned} \tag{10}
$$

注意，运动中每个 3D 姿态 $\mathbf { X } ^ { \mathrm { m } }$ 在第 t 帧由网格顶点回归得到 $\mathbf { X } _ { t } ^ { \mathrm { m } } = \mathbf { J } \mathcal { M } ( \theta _ { t } , \beta _ { t } )$，其中 $\mathbf { J } \in \mathbb{R}^{J \times 6890}$ 是一个预定义矩阵 [10]。$\mathbf { O } ^ { \mathrm { m } } = \mathbf { X } _ { t + 1 } ^ { \mathrm { m } } - \mathbf { X } _ { t } ^ { \mathrm { m } }$，$\hat { \mathbf { O } } ^ { \mathrm { m } } = \hat { \mathbf { X } } _ { t + 1 } ^ { \mathrm { m } } - \hat { \mathbf { X } } _ { t } ^ { \mathrm { m } }$。$\lambda _ { 3 \mathrm { D } } ^ { \mathrm { m } }$、$\lambda _ { \theta }$、$\lambda _ { \beta }$、$\lambda _ { \mathrm { n } }$ 和 $\lambda _ { 0 } ^ { \mathrm { m } }$ 是用于平衡训练损失的常数系数。

## 4. 实验

## 4.1. 实现细节

我们实现了所提出的运动编码器 DSTformer（动态时空 Transformer），其深度 $N = 5$，多头注意力头数 $h = 8$，特征维度 $C _ { \mathrm { f } } = 5 1 2$，嵌入维度 $C _ { \mathrm { e } } = 5 1 2$。在预训练阶段，我们使用序列长度 $T = 2 4 3$。由于采用了基于 Transformer 的骨干网络，预训练模型能够处理不同长度的输入。在微调阶段，我们将骨干网络的学习率设置为新增层学习率的 0.1 倍。我们将在后续章节分别介绍实验所用的数据集。更多实验细节请参阅附录。

## 4.2. 预训练

我们从两个数据集 Human3.6M [38] 和 AMASS [76] 中收集多样化且真实的 3D 人体运动数据。Human3.6M [38] 是一个常用的室内 3D 人体姿态估计数据集，包含专业演员执行日常动作的 360 万视频帧。遵循先前工作 [78, 89] 的设置，我们使用受试者 1、5、6、7、8 进行训练，使用受试者 9、11 进行测试。AMASS [76] 整合了大多数现有的基于标记的动作捕捉（Mocap）数据集 [1,2,5,11,15,18,34,37,52,69,72,77,84,97,110,111]，并使用统一的表示形式对其进行参数化。由于动作捕捉数据集通常不提供原始视频，因此在预训练过程中我们不使用这两个数据集的图像或 2D 检测结果，而是使用正交投影获取未受污染的 2D 骨架。此外，我们还引入了两个野外 RGB 视频数据集 PoseTrack [3]（带标注）和 InstaVariety [42]（无标注），以提高运动的多样性。我们将人体关键点的定义与 Human3.6M 对齐，并按照 [27] 的方法将相机坐标校准为像素坐标。我们随机将 15% 的关节置零，并从高斯分布和均匀分布的混合分布中采样噪声 [17]。我们首先仅使用 3D 数据训练 30 个 epoch，然后同时使用 3D 数据和 2D 数据训练 60 个 epoch，遵循课程学习的做法 [9, 118]。

## 4.3. 3D 姿态估计

我们在 Human3.6M [38] 上评估 3D 姿态估计性能，并报告平均关节位置误差（mean per joint position error，MPJPE），单位为毫米，该指标在根关节对齐后度量预测关节点与真值（GT）之间的平均距离。我们还计算了平均关节速度误差（mean per-joint velocity error，MPJVE），以按照先前工作 [134, 135] 的方式评估时序平滑性。我们使用堆叠沙漏（Stacked Hourglass，SH）网络 [86] 从视频中提取 2D 骨架，并在 Human3.6M [38] 训练集上对整个网络进行微调。此外，我们还训练了一个相同架构但采用随机初始化（而非预训练权重）的独立模型。如表 1（上）所示，从头训练的模型优于包括其他基于 Transformer 并结合时空建模的方法在内的先前方法。这表明所提出的 DSTformer 在学习 3D 几何结构和时序动态方面的有效性。为进一步评估模型能力的上界，我们比较了使用 2D 真值姿态序列作为输入时的性能，从而排除了不同 2D 检测器的影响。如表 1（中）所示，我们的模型显著优于所有先前方法。表 1（下）显示，我们两个模型的 MPJVE 也超越了先前工作，表明具有更好的时序一致性。我们将从头训练模型的性能优势归因于所提出的 DSTformer 设计。我们将在第 4.6 节和补充材料中提供更多对比和分析，以展示 DSTformer 相对于其他时空架构的优势。此外，我们的方法在所提出的预训练阶段下取得了更低的误差。

| 方法 | T | 指示 | 讨论 | 进食 | 问候 | 打电话 | 拍照 | 摆姿势 | 购物 | 坐下 | 坐下动作 | 吸烟 | 等待 | 遛狗 | 行走 | 同行 | 平均 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Martinez et al. [78] ICCV'17 | 1 | 51.8 | 56.2 | 58.1 | 59.0 | 69.5 | 78.4 | 55.2 | 58.1 | 74.0 | 94.6 | 62.3 | 59.1 | 65.1 | 49.5 | 52.4 | 62.9 |
| Pavlakos et al. [88] CVPR'18 | 1 | 48.5 | 54.4 | 54.4 | 52.0 | 59.4 | 65.3 | 49.9 | 52.9 | 65.8 | 71.1 | 56.6 | 52.9 | 60.9 | 44.7 | 47.8 | 56.2 |
| LCN [28] ICCV'19 | 1 | 46.8 | 52.3 | 44.7 | 50.4 | 52.9 | 68.9 | 49.6 | 46.4 | 60.2 | 78.9 | 51.2 | 50.0 | 54.8 | 40.4 | 43.3 | 52.7 |
| Xu et al. [121] CVPR'21 | 1 | 45.2 | 49.9 | 47.5 | 50.9 | 54.9 | 66.1 | 48.5 | 46.3 | 59.7 | 71.5 | 51.4 | 48.6 | 53.9 | 39.9 | 44.1 | 51.9 |
| VideoPose3D [89] CVPR'19 | 243 | 45.2 | 46.7 | 43.3 | 45.6 | 48.1 | 55.1 | 44.6 | 44.3 | 57.3 | 65.8 | 47.1 | 44.0 | 49.0 | 32.8 | 33.9 | 46.8 |
| Cai et al. [13] ICCV'19 | 7 | 44.6 | 47.4 | 45.6 | 48.8 | 50.8 | 59.0 | 47.2 | 43.9 | 57.9 | 61.9 | 49.7 | 46.6 | 51.3 | 37.1 | 39.4 | 48.8 |
| Yeh et al. [129] NeurIPS'19 | 243 | 44.8 | 46.1 | 43.3 | 46.4 | 49.0 | 55.2 | 44.6 | 44.0 | 58.3 | 62.7 | 47.1 | 43.9 | 48.6 | 32.7 | 33.3 | 46.7 |
| Liu et al. [67] CVPR'20 | 243 | 41.8 | 44.8 | 41.1 | 44.9 | 47.4 | 54.1 | 43.4 | 42.2 | 56.2 | 63.6 | 45.3 | 43.5 | 45.3 | 31.3 | 32.2 | 45.1 |
| * Cheng et al. [22] AAAI'20 | 128 | 36.2 | 38.1 | 42.7 | 35.9 | 38.2 | 45.7 | 36.8 | 42.0 | 45.9 | 51.3 | 41.8 | 41.5 | 43.8 | 33.1 | 28.6 | 40.1 |
| * UGCN [116] ECCV'20 | 96 | 38.2 | 41.0 | 45.9 | 39.7 | 41.4 | 51.4 | 41.6 | 41.4 | 52.0 | 57.4 | 41.8 | 44.4 | 41.6 | 33.1 | 30.0 | 42.6 |
| † PoseFormer [135] ICCV'21 | 81 | 41.5 | 44.8 | 39.8 | 42.5 | 46.5 | 51.6 | 42.1 | 42.0 | 53.3 | 60.7 | 45.5 | 43.3 | 46.1 | 31.8 | 32.2 | 44.3 |
| * Wehrbein et al. [119] ICCV'21 | 200 | 38.5 | 42.5 | 39.9 | 41.7 | 46.5 | 51.6 | 39.9 | 40.8 | 49.5 | 56.8 | 45.3 | 46.4 | 46.8 | 37.8 | 40.4 | 44.3 |
| † MHFormer [56] CVPR'22 | 351 | 39.2 | 43.1 | 40.1 | 40.9 | 44.9 | 51.2 | 40.6 | 41.3 | 53.5 | 60.3 | 43.7 | 41.1 | 43.8 | 29.8 | 30.6 | 43.0 |
| *† MixSTE [134] CVPR'22 | 243 | 36.7 | 39.0 | 36.5 | 39.4 | 40.2 | 44.9 | 39.8 | 36.9 | 47.9 | 54.8 | 39.6 | 37.8 | 39.3 | 29.7 | 30.6 | 39.8 |
| † P-STMO [94] ECCV'22 | 243 | 38.4 | 42.1 | 39.8 | 40.2 | 45.2 | 48.9 | 40.4 | 38.3 | 53.8 | 57.3 | 43.9 | 41.6 | 42.2 | 29.3 | 29.3 | 42.1 |
| † Ours (scratch) | 243 | 36.3 | 38.7 | 38.6 | 33.6 | 42.1 | 50.1 | 36.2 | 35.7 | 50.1 | 56.6 | 41.3 | 37.4 | 37.7 | 25.6 | 26.5 | 39.2 |
| † Ours (finetune) | 243 | 36.1 | 37.5 | 35.8 | 32.1 | 40.3 | 46.3 | 36.1 | 35.3 | 46.9 | 53.9 | 39.5 | 36.3 | 35.8 | 25.1 | 25.3 | 37.5 |

| 方法 | T | 指示 | 讨论 | 进食 | 问候 | 打电话 | 拍照 | 摆姿势 | 购物 | 坐下 | 坐下动作 | 吸烟 | 等待 | 遛狗 | 行走 | 同行 | 平均 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Martinez et al. [78] ICCV'17 | 1 | 37.7 | 44.4 | 40.3 | 42.1 | 48.2 | 54.9 | 44.4 | 42.1 | 54.6 | 58.0 | 45.1 | 46.4 | 47.6 | 36.4 | 40.4 | 45.5 |
| LCN [28] ICCV'19 | 1 | 36.3 | 38.8 | 29.7 | 37.8 | 34.6 | 42.5 | 39.8 | 32.5 | 36.2 | 39.5 | 34.4 | 38.4 | 38.2 | 31.3 | 34.2 | 36.3 |
| Xu et al. [121] CVPR'21 | 1 | 35.8 | 38.1 | 31.0 | 35.3 | 35.8 | 43.2 | 37.3 | 31.7 | 38.4 | 45.5 | 35.4 | 36.7 | 36.8 | 27.9 | 30.7 | 35.8 |
| UGCN [116] ECCV'20 | 96 | 23.0 | 25.7 | 22.8 | 22.6 | 24.1 | 30.6 | 24.9 | 24.5 | 31.1 | 35.0 | 25.6 | 24.3 | 25.1 | 19.8 | 18.4 | 25.6 |
| † PoseFormer [135] ICCV'21 | 81 | 30.0 | 33.6 | 29.9 | 31.0 | 30.2 | 33.3 | 34.8 | 31.4 | 37.8 | 38.6 | 31.7 | 31.5 | 29.0 | 23.3 | 23.1 | 31.3 30.5 |
| † MHFormer [56] CVPR'22 | 351 | 27.7 | 32.1 | 29.1 | 28.9 | 30.0 | 33.9 | 33.0 | 31.2 | 37.0 | 39.3 | 30.0 | 31.0 | 29.4 | 22.2 | 23.0 |  |
| † MixSTE [134] CVPR'22 | 243 | 21.6 | 22.0 | 20.4 | 21.0 | 20.8 | 24.3 | 24.7 | 21.9 | 26.9 | 24.9 | 21.2 | 21.5 | 20.8 | 14.7 | 15.6 |  |

表格 1. Human3.6M 上 3D 人体姿态估计的定量比较。（上）使用检测到的 2D 姿态序列的 MPJPE（毫米）。（中）使用 GT 2D 姿态序列的 MPJPE（毫米）。（下）使用检测到的 2D 姿态序列的 MPJVE（毫米）。T 表示该方法使用的片段长度。我们选取每篇工作中报告的最佳结果。<sup>∗</sup> 表示使用 HRNet [103] 进行 2D 检测。<sup>†</sup> 表示采用时空 Transformer 设计实现。最佳和次佳结果分别以加粗和下划线格式突出显示。

## 4.4. 基于骨架的动作识别

我们进一步探索利用预训练的人体运动表示来学习动作语义的可能性。我们使用人体动作数据集 NTU-RGB+D [93]，该数据集包含 60 个动作类别的 57K 视频，并遵循 Cross-subject（X-Sub）和 Cross-view（X-View）的数据划分方式。该数据集有一个扩展版本 NTU-RGB+D-120 [64]，包含 120 个动作类别的 114K 视频。我们遵循 NTU-RGB+D-120 上建议的 One-shot 动作识别协议。对于这两个数据集，我们按照 [32] 的方法，使用 HRNet [103] 提取 2D 骨架。同样，我们训练一个随机初始化的基线模型进行比较。如表 2（左）所示，我们的方法与最先进的方法相当或更优。值得注意的是，预训练阶段带来了显著的性能提升。

此外，我们深入研究具有重要实际意义的 one-shot 设置。现实应用通常需要在教育、体育和医疗等特定领域进行细粒度的动作识别。然而，这些场景中的动作类别通常在公开数据集中没有定义。因此，这些新动作类别的标注数量有限，这使得准确识别成为一项困难的任务。按照 [64] 中提出的方法，我们在 20 个新类别的评估集上报告结果，每个类别仅使用 1 个标注视频。辅助集包含其余 100 个类别，可以使用这些类别的所有样本。我们使用监督对比学习技术 [44] 在辅助集上训练模型。对于一批辅助数据，同类别的样本被拉近，而不同类别的样本在动作嵌入空间中被推远。在评估过程中，我们计算测试样本与示例之间的余弦距离，并使用 1-最近邻来确定类别。表 2（右）表明，所提出的模型以显著的优势优于最先进的方法。此外，值得注意的是，我们的预训练模型仅需 1-2 个 epoch 的微调即可达到最佳性能。我们的结果表明，预训练阶段能够有效地学习鲁棒的运动表示，即使在标注数据有限的情况下，也能很好地泛化到新的下游任务。

| 方法 | X-Sub | X-View |
| --- | --- | --- |
| ST-GCN [123] AAAI'18 | 81.5 | 88.3 |
| 2s-AGCN [96] CVPR'19 | 88.5 | 95.1 |
| MS-G3D [68] CVPR'20 | 91.5 | 96.2 |
| Shift-GCN [21] CVPR'20 | 90.7 | 96.5 |
| CrossSCLR [62] CVPR'21 | 86.2 | 92.5 |
| MCC（微调） [101] ICCV'21 | 89.7 | 96.3 |
| SCC（微调） [125] ICCV'21 | 88.0 | 94.9 |
| UNIK（微调） [124] BMVC'21 | 86.8 | 94.4 |
| CTR-GCN [20] ICCV'21 | 92.4 | 96.8 |
| PoseConv3D [32] CVPR'22 | 93.1 | 95.7 |
| 本文方法（从头训练） | 87.7 | 94.1 |
| 本文方法（微调） | 93.0 | 97.2 |

| 方法 | 准确率 |
| --- | --- |
| ST-LSTM + AvgPool [65] | 42.9 |
| ST-LSTM + FC [66] | 42.1 |
| ST-LSTM + Attention [66] | 41.0 |
| APSR [64] | 45.3 |
| TCN OneShot [92] | 46.5 |
| SL-DML [81] | 50.9 |
| Skeleton-DML [80] | 54.2 |
| 本文方法（从头训练） | 61.0 |
| 本文方法（微调） | 67.4 |

Table 2. 基于骨架的动作识别准确率的定量比较。（左）NTU-RGB+D 上的跨受试者和跨视角识别准确率。为保证比较的公平性，所有方法均仅使用"joint"模态并采用 1-clip 采样进行评估。（右）NTU-RGB+D-120 上的 one-shot 识别准确率。所有结果均为 top-1 准确率（%）。

| Method | Input | $T$ | Human3.6M MPVE↓ | Human3.6M MPJPE↓ | Human3.6M PA-MPJPE↓ | 3DPW MPVE↓ | 3DPW MPJPE↓ | 3DPW PA-MPJPE↓ |
|---|---|---|---|---|---|---|---|---|
| HMR [41] CVPR'18 | image | 1 |  | 88.0 | 56.8 |  | 130.0 | 81.3 |
| † SPIN [48] ICCV'19 | image | 1 | 82.3 | 59.4 | 39.3 | 129.1 | 100.9 | 59.1 |
| Pose2Mesh [25] ECCV'20 | 2D pose | 1 | 85.3 | 64.9 | 48.7 | 109.3 | 91.4 | 60.1 |
| I2L-MeshNet [83] ECCV'20 | image | 1 |  | 55.7 | 41.7 | 110.1 | 93.2 | 58.6 |
| † HybrIK [54] CVPR'21 | image | 1 | 58.1 | 47.4 | 30.1 | 82.4 | 71.3 | 41.9 |
| METRO [58] CVPR'21 | image | 1 |  | 54.0 | 36.7 | 88.2 | 77.1 | 47.9 |
| Mesh Graphormer [59] ICCV'21 | image | 1 |  | 51.2 | 34.5 | 87.7 | 74.7 | 45.6 |
| PARE [47] ICCV'21 | image | 1 |  |  |  | 88.6 | 74.5 | 46.5 |
| ROMP [105] ICCV'21 | image | 1 |  |  |  | 108.3 | 91.3 | 54.9 |
| PyMAF [133] ICCV'21 | image | 1 |  | 57.7 | 40.5 | 110.1 | 92.8 | 58.9 |
| ProHMR [50] ICCV'21 | image | 1 |  |  | 41.2 |  |  | 59.8 |
| OCHMR [43] CVPR'22 | image | 1 |  |  |  | 107.1 | 89.7 | 58.3 |
| 3DCrowdNet [26] CVPR'22 | image | 1 |  |  |  | 98.3 | 81.7 | 51.5 |
| CLIFF [57] ECCV'22 | image | 1 |  | 47.1 | 32.7 | 81.2 | 69.0 | 43.0 |
| FastMETRO [23] ECCV'22 | image | 1 |  | 52.2 | 33.7 | 84.1 | 73.5 | 44.6 |
| VisDB [128] ECCV'22 | image | 1 |  | 51.0 | 34.5 | 85.5 | 73.5 | 44.9 |
| TemporalContext [6] CVPR'19 | video | 32 |  | 77.8 | 54.3 | - |  | 72.2 |
| HMMR [42] CVPR'19 | video | 20 |  |  | 56.9 | 139.3 | 116.5 | 72.6 |
| DSD-SATN [106] ICCV'19 | video | 9 |  | 59.1 | 42.4 |  |  | 69.5 |
| VIBE [46] CVPR'20 | video | 16 |  | 65.6 | 41.4 | 99.1 | 82.9 | 51.9 |
| TCMR [24] CVPR'21 | video | 16 |  | 62.3 | 41.1 | 102.9 | 86.5 | 52.7 |
| † MAED [114] ICCV'21 | video | 16 | 84.1 | 60.4 | 38.3 | 93.3 | 79.0 | 45.7 |
| MPS-Net [120] CVPR'22 | video | 16 |  | 69.4 | 47.4 | 99.7 | 84.3 | 52.1 |
| * PoseBERT [8] TPAMI'22 (+SPIN [48]) | video | 16 |  |  |  | - |  | 57.3↓2.3 |
| * SmoothNet [130] ECCV'22 (+SPIN [48]) | video | 32 |  | 67.5↓1.0 | 46.3↓0.2 | - | 86.7↓0.9 | 52.7↓0.6 |
| Ours (scratch) | 2D motion | 16 | 75.7 | 62.8 | 41.0 | 99.1 | 85.5 | 50.2 |
| Ours (finetune) | 2D motion | 16 | 65.5 | 53.8 | 34.9 | 88.1 | 76.9 | 47.2 |
| Ours (finetune) + SPIN [48] | video | 16 | 63.7↓18.6 | 52.2↓7.2 | 35.7↓3.6 | 92.8↓36.3 | 79.6↓21.3 | 48.2↓10.9 |
| Ours (finetune) + MAED [114] | video | 16 | 66.8↓17.3 | 54.8↓5.6 | 36.4↓1.9 | 84.4↓8.9 | 72.3↓6.7 | 42.3↓3.4 |
| Ours (finetune) + HybrIK [54] | video | 16 | 52.6↓5.5 | 43.1↓4.3 | **27.8↓2.3** | **79.4↓3.0** | **68.8↓2.5** | **40.6↓1.3** |

表 3. 在 Human3.6M 和 3DPW 数据集上人体网格恢复的定量比较。T 表示该方法使用的片段长度。<sup>†</sup> 表示使用官方模型权重得到的结果。其余均为官方报告结果。<sup>∗</sup> 中的增益对应不同重新实现的 SPIN [48] 结果。

## 4.5. 人体网格恢复

我们在 Human3.6M [38] 和 3DPW [113] 数据集上进行实验，并在训练时额外加入 COCO [61] 数据集，遵循 [46, 58, 114] 的做法。我们对两个数据集都保持与 [78]（第 4.2 节）相同的训练和测试划分

![](images/4d86d3f94a3a07c4a343700db80f9d6b038f4cc84b9e309ddd754b4adebf0200.jpg)

![](images/a4677945c5211b4e33f869a09a0b272a6758b908e8c56c5ccb81b680869a07fa.jpg)

![](images/860450794e7623f5bf6e3d2abb99f077575793fa7971030a4267aa5d7da16b3d.jpg)  
图 3. 微调与从头训练的学习曲线。

<table><tr><td>骨干网络（冻结）</td><td>MPJPE↓ (3D 姿态)</td><td>MPVE↓ (网格)</td><td>准确率 ↑ (action x-view)</td><td>准确率↑ (action 1-shot)</td></tr>
<tr><td>随机<br>预训练</td><td>404.4mm<br>40.3mm</td><td>114.4mm<br>72.1mm</td><td>47.6%<br>87.3%</td><td>46.8%<br>60.7%</td></tr></table>

表 4. 部分微调的比较。

以及 [46, 58, 114]。遵循通用做法 [41,46,49,114]，我们报告由 J(θ, β) 得到的 14 个关节点的 MPJPE (mm) 和 PA-MPJPE (mm)。PA-MPJPE 在与 GT 进行平移、旋转和尺度对齐后计算 MPJPE。我们进一步报告网格 (θ, β) 的平均每顶点误差 (MPVE) (mm)，该指标在根关节点对齐后度量估计顶点与 GT 顶点之间的平均距离。需要注意的是，大多数先前工作 [24, 41, 46, 48, 54, 58, 73, 114] 在训练时使用了除 COCO [61] 之外的更多数据集，例如 LSP [39]、MPI-INF-3DHP [79] 等，而我们没有使用这些数据集。表 3 表明，我们的微调模型在 Human3.6M 和 3DPW 数据集上都取得了具有竞争力的结果，超越了所有最先进的基于视频的方法，包括 MAED [114]，尤其在 MPVE 误差上。然而，我们注意到仅从稀疏 2D 关键点估计全身网格 [10,25] 是一个不适定问题，因为它缺少人体形状信息。鉴于此，我们提出了一种混合方法，结合了我们框架（运动一致性）和基于 RGB 方法（精确形状）的优势。我们引入了一个可轻松与现有图像/视频方法集成的优化器（refiner）模块，类似于 [8,130]。具体而言，我们的 refiner 模块是一个 MLP，将我们预训练的运动表示与初始预测相结合，回归出关节点旋转的残差。我们的方法有效地提升了当前最先进方法 [48,54,114] 的性能，达到了迄今为止最低的误差。

## 4.6. 消融实验

# 微调与从头训练。 我们比较了对预训练模型进行微调与从头开始训练的进展。如图 3 所示，使用预训练权重初始化的模型在所有三个任务上均表现出更优的性能和更快的收敛速度。这一观察结果表明，预训练模型学习到了关于人体运动的迁移知识，从而促进多个下游任务的学习。

<table><tr><td>预训练</td><td>噪声</td><td>2D 掩码</td><td>MPJPE↓ (3D 姿态)</td><td>MPVE↓ (网格)</td><td>准确率↑ (动作 x-sub)</td></tr><tr><td>-</td><td></td><td></td><td>39.2mm</td><td>75.7mm</td><td>87.7%</td></tr><tr><td>√</td><td>-</td><td></td><td>38.8mm</td><td>70.6mm</td><td>89.4%</td></tr><tr><td>√</td><td>√</td><td>一 1</td><td>38.1mm</td><td>68.4mm</td><td>90.7%</td></tr><tr><td>√</td><td>√</td><td>√ –</td><td>37.4mm</td><td>67.8mm</td><td>91.9%</td></tr><tr><td>√</td><td>√</td><td>√ √</td><td>37.5mm</td><td>65.5mm</td><td>93.0%</td></tr></tr>

表 5. 预训练策略的比较。

# 部分微调。 除了端到端微调外，我们冻结运动编码器骨干网络，仅为每个下游任务训练回归头。为了验证预训练运动表征的有效性，我们将预训练的运动编码器与随机初始化的运动编码器进行了比较。我们在 Human3.6M 上报告 3D 姿态和网格的结果，在 NTU-RGB+D 和 NTU RGB+D-120 上报告动作识别结果（下表相同）。从表 4 可以看出，基于冻结的预训练运动表征，我们的方法在多个下游任务上仍取得了具有竞争力的性能，并相较基线有大幅提升。预训练与部分微调使得所有下游任务能够共享同一骨干网络，显著降低了多任务推理应用所需的计算开销。

# 预训练策略。 我们评估了不同预训练策略对下游任务性能的影响。从从头训练的基线开始，我们逐一应用所提出的策略。如表 5 所示，原生的 2D 到 3D 预训练阶段为所有下游任务带来了收益。引入扰动也额外改善了所学到的运动嵌入。结合野外视频（w. 2D）的统一预训练带来了更高的运动多样性，进一步提升了下游任务的表现。

# 不同骨干网络下的预训练。 我们进一步研究所提出的预训练方法的通用性。我们将运动编码器骨干网络替换为两个变体：TCN [89] 和 PoseFormer [135]。这些模型被稍作修改为 seq2seq 版本，而预训练和微调的所有配置均保持不变。表 6 表明，所提出的方法在不同任务的不同骨干模型上均能持续带来收益。

| 设置 | MPJPE↓（3D 姿态） | MPVE↓（网格） | Accuracy↑（动作 x-view） | Accuracy↑（动作 1-shot） |
|---|---|---|---|---|
| TCN（从头训练） | 50.1mm | 92.6mm | 91.5% | 52.4% |
| TCN（微调） | 47.9mm | 86.3mm | 92.8% | 59.9% |
| PoseFormer（从头训练） | 44.8mm | 85.9mm | 94.2% | 57.4% |
| PoseFormer（微调） | 41.5mm | 80.5mm | 95.9% | 60.7% |

表 6. 不同骨干网络的比较。

| 架构设计 | (a) S-T | (b) T-S | (c) S+T | (d) ST-MHSA | (e) S-T + T-S（平均） | (f) S-T + T-S（自适应） |
|---|---|---|---|---|---|---|
| MPJPE↓ | 40.58±0.31 | 41.05±0.24 | 41.76±0.22 | 41.54±0.35 | 39.87±0.32 | 39.25±0.27 |

表 7. 模型架构变体的比较。所有方法均在 Human3.6M 上从头训练，运行 5 次，通过 MPJPE（mm）的均值和标准差进行测量。

**模型架构。** 最后，我们研究 DSTformer 的设计选择。在表 8 中，从 (a) 到 (f)，我们比较了基础 Transformer 模块的不同结构设计。(a) 和 (b) 是不同顺序的单流版本。(a) 在概念上类似于 PoseFormer [135]、MH-Former [56] 和 MixSTE [134]。(c) 在融合前将每个流限制为时序或空间建模，类似于 MAED [114]。(d) 直接连接 S-MHSA 和 T-MHSA 而中间不使用 MLP，类似于 MAED [114] 中的 MSA-T 变体。(e) 用两个流上的平均池化代替自适应融合。(f) 是所提出的 DSTformer 设计。结果在统计上证实了我们的设计原则，即两个流都应当具备能力且同时互补，如第 3.2 节所述。此外，我们发现将每个自注意力块与 MLP 配对至关重要，因为它可以投影学习到的特征交互并带来非线性。总体而言，我们为 3D 姿态估计任务设计了模型架构，并将其应用于所有其他任务，无需额外调整。

## 5. 结论

在本工作中，我们提供了一个统一的视角来处理各种以人为中心的视频任务。我们开发了一种预训练方法，从大规模和异构数据源中学习人体运动表示。我们还提出了 DSTformer 作为通用人体运动编码器。在多个基准上的实验结果证明了所学运动表示的多功能性。未来的工作可以探索将所学运动表示与通用视频架构相融合，作为以人为中心的语义特征，并将其应用于更多任务（例如动作评估、分割）。

致谢 本工作部分由 MOST-2022ZD0114900 资助。我们感谢 Hai Ci 和 Jiefeng Li 提供的卓越支持。

## 参考文献

[1] Advanced Computing Center for the Arts and Design. ACCAD MoCap Dataset. 5

[2] Ijaz Akhter 和 Michael J. Black. 用于三维人体姿态重建的姿态条件关节角度限制. 载于 CVPR, 页 1446–1455, 2015 年 6 月. 5

[3] Mykhaylo Andriluka、Umar Iqbal、Eldar Insafutdinov、Leonid Pishchulin、Anton Milan、Juergen Gall 和 Bernt Schiele. PoseTrack：人体姿态估计与跟踪基准. 载于 CVPR, 页 5167–5176, 2018. 1, 3, 4, 5, 16

[4] Mykhaylo Andriluka、Leonid Pishchulin、Peter Gehler 和 Bernt Schiele. 人体姿态估计：新基准与最先进技术分析. 载于 CVPR, 2014. 16

[5] Andreas Aristidou、Ariel Shamir 和 Yiorgos Chrysanthou. 数字舞蹈民族志：组织大型舞蹈集合. JOCCH, 12(4):1–27, 2019. 5

[6] Anurag Arnab、Carl Doersch 和 Andrew Zisserman. 利用时间上下文进行野外三维人体姿态估计. 载于 CVPR, 页 3395–3404, 2019. 7

[7] Hangbo Bao、Li Dong、Songhao Piao 和 Furu Wei. BEiT：图像 Transformer 的 BERT 预训练. 载于 ICLR, 2022. 4

[8] Fabien Baradel、Romain Brégier、Thibault Groueix、Philippe Weinzaepfel、Yannis Kalantidis 和 Grégory Rogez. PoseBERT：用于时序三维人体建模的通用 Transformer 模块. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022. 3, 7, 8

[9] Yoshua Bengio、Jérôme Louradour、Ronan Collobert 和 Jason Weston. 课程学习. 载于 ICML, 页 41–48, 2009. 5

[10] Federica Bogo、Angjoo Kanazawa、Christoph Lassner、Peter Gehler、Javier Romero 和 Michael J Black. 保持 SMPL：从单张图像自动估计三维人体姿态与形状. 载于 ECCV, 页 561–578, 2016. 3, 5, 8, 16

[11] Federica Bogo、Javier Romero、Gerard Pons-Moll 和 Michael J Black. Dynamic FAUST：配准运动中的人体. 载于 CVPR, 页 6233–6242, 2017. 5

[12] Tom Brown、Benjamin Mann、Nick Ryder、Melanie Subbiah、Jared D Kaplan、Prafulla Dhariwal、Arvind Neelakantan、Pranav Shyam、Girish Sastry、Amanda Askell 等. 语言模型是少样本学习器. NeurIPS, 33:1877–1901, 2020. 4

[13] Yujun Cai、Liuhao Ge、Jun Liu、Jianfei Cai、Tat-Jen Cham、Junsong Yuan 和 Nadia Magnenat Thalmann. 通过图卷积网络利用时空关系进行三维姿态估计. 载于 ICCV, 页 2272–2281, 2019. 1, 2, 6

[14] Zhe Cao、Tomas Simon、Shih-En Wei 和 Yaser Sheikh. 使用部分亲和场的实时多人二维姿态估计. 载于 CVPR, 页 7291–7299, 2017. 1, 4, 16

[15] Carnegie Mellon University. CMU MoCap Dataset. 5

[16] Joao Carreira 和 Andrew Zisserman. 动作识别何去何从？一个新模型与 Kinetics 数据集. 载于 CVPR, 页 6299–6308, 2017. 1

[17] Ju Yong Chang, Gyeongsik Moon, and Kyoung Mu Lee. Poselifter: Absolute 3d human pose lifting network from a single noisy 2d human pose. arXiv preprint arXiv:1910.12029, 2019. 5, 16

[18] Anargyros Chatzitofis, Leonidas Saroglou, Prodromos Boutis, Petros Drakoulis, Nikolaos Zioulis, Shishir Subramanyam, Bart Kevelham, Caecilia Charbonnier, Pablo Cesar, Dimitrios Zarpalas, 等. Human4d: A human-centric multimodal dataset for motions and immersive media. IEEE Access, 8:176241– 176262, 2020. 5

[19] Xipeng Chen, Kwan-Yee Lin, Wentao Liu, Chen Qian, and Liang Lin. Weakly-supervised discovery of geometry-aware representation for 3d human pose estimation. In CVPR, 2019. 3

[20] Yuxin Chen, Ziqi Zhang, Chunfeng Yuan, Bing Li, Ying Deng, and Weiming Hu. Channel-wise topology refinement graph convolution for skeleton-based action recognition. In ICCV, 2021. 7

[21] Ke Cheng, Yifan Zhang, Xiangyu He, Weihan Chen, Jian Cheng, and Hanqing Lu. Skeleton-based action recognition with shift graph convolutional network. In CVPR, 2020. 2, 7

[22] Yu Cheng, Bo Yang, Bo Wang, and Robby T Tan. 3d human pose estimation using spatio-temporal networks with explicit occlusion training. In AAAI, 2020. 2, 6

[23] Junhyeong Cho, Kim Youwang, and Tae-Hyun Oh. Cross-attention of disentangled modalities for 3d human mesh recovery with transformers. In European Conference on Computer Vision (ECCV), 2022. 7

[24] Hongsuk Choi, Gyeongsik Moon, Ju Yong Chang, and Kyoung Mu Lee. Beyond static features for temporally consistent 3d human pose and shape from a video. In CVPR, pages 1964–1973, 2021. 3, 7, 8

[25] Hongsuk Choi, Gyeongsik Moon, and Kyoung Mu Lee. Pose2mesh: Graph convolutional network for 3d human pose and mesh recovery from a 2d human pose. In ECCV, pages 769–787, 2020. 3, 7, 8, 16

[26] Hongsuk Choi, Gyeongsik Moon, JoonKyu Park, and Kyoung Mu Lee. Learning to estimate robust 3d human mesh from in-the-wild crowded scenes. In CVPR, pages 1475–1484, June 2022. 7

[27] Hai Ci, Xiaoxuan Ma, Chunyu Wang, and Yizhou Wang. Locally connected network for monocular 3d human pose estimation. IEEE TPAMI, pages 1–1, 2020. 5, 16

[28] Hai Ci, Chunyu Wang, Xiaoxuan Ma, and Yizhou Wang. Optimizing network structure for 3d human pose estimation. In ICCV, pages 2262–2271, 2019. 2, 6

[29] Hai Ci, Mingdong Wu, Wentao Zhu, Xiaoxuan Ma, Hao Dong, Fangwei Zhong, and Yizhou Wang. Gfpose: Learning 3d human pose prior with gradient fields. CVPR, 2023. 2

[30] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019. 3, 4, 16

[31] Haodong Duan, Jiaqi Wang, Kai Chen, and Dahua Lin. Pyskl: Towards good practices for skeleton action recognition, 2022. 16

[32] Haodong Duan, Yue Zhao, Kai Chen, Dian Shao, Dahua Lin, and Bo Dai. Revisiting skeleton-based action recognition. In CVPR, 2022. 2, 3, 6, 7, 16

[33] Mark Endo、Kathleen L. Poston、Edith V. Sullivan、李飞飞、Kilian M. Pohl 和 Ehsan Adeli。Gaitformer：通过人体运动预测对 Transformer 进行自监督预训练以用于少样本步态损伤严重程度估计。发表于 MICCAI，第 130–139 页，2022 年。2

[34] Saeed Ghorbani、Kimia Mahdaviani、Anne Thaler、Konrad Kording、Douglas James Cook、Gunnar Blohm 和 Nikolaus F. Troje。MoVi：一个大型多用途运动和视频数据集，2020 年。5

[35] Rıza Alp Güler、Natalia Neverova 和 Iasonas Kokkinos。Densepose：野外密集人体姿态估计。发表于 CVPR，第 7297–7306 页，2018 年。1

[36] Kaiming He、Xinlei Chen、Saining Xie、Yanghao Li、Piotr Dollár 和 Ross B. Girshick。掩码自编码器是可扩展的视觉学习器。发表于 CVPR，2022 年。4, 18

[37] Ludovic Hoyet、Kenneth Ryall、Rachel McDonnell 和 Carol O'Sullivan。手法巧妙：从缩减标记集合感知手指运动。发表于 ACM SIGGRAPH 交互式 3D 图形与游戏研讨会论文集，2012 年。5

[38] Catalin Ionescu、Dragos Papava、Vlad Olaru 和 Cristian Sminchisescu。Human3.6M：用于自然环境中三维人体感知的大规模数据集及预测方法。IEEE TPAMI，2014 年。1, 5, 7, 16

[39] Sam Johnson 和 Mark Everingham。用于人体姿态估计的聚类姿态和非线性外观模型。发表于 BMVC，第 2 卷，第 5 页，2010 年。8

[40] Hanbyul Joo、Natalia Neverova 和 Andrea Vedaldi。用于野外三维人体姿态估计的范例微调三维人体模型拟合。发表于 3DV，第 42–52 页。IEEE，2021 年。16

[41] Angjoo Kanazawa、Michael J Black、David W Jacobs 和 Jitendra Malik。人体形状和姿态的端到端恢复。发表于 CVPR，第 7122–7131 页，2018 年。2, 7, 8, 16

[42] Angjoo Kanazawa、Jason Y Zhang、Panna Felsen 和 Jitendra Malik。从视频学习三维人体动态。发表于 CVPR，第 5614–5623 页，2019 年。2, 3, 5, 7, 16

[43] Rawal Khirodkar、Shashank Tripathi 和 Kris Kitani。遮挡情况下的人体网格恢复。发表于 CVPR，第 1715–1725 页，2022 年 6 月。7

[44] Prannay Khosla、Piotr Teterwak、Chen Wang、Aaron Sarna、Yonglong Tian、Phillip Isola、Aaron Maschinot、Ce Liu 和 Dilip Krishnan。有监督对比学习。发表于 NeurIPS，2020 年。7, 16

[45] Diederik P Kingma 和 Jimmy Ba。Adam：一种随机优化方法。arXiv 预印本，arXiv:1412.6980，2014 年。16

[46] Muhammed Kocabas、Nikos Athanasiou 和 Michael J Black。VIBE：用于人体姿态与形状估计的视频推理。发表于 CVPR，第 5253–5263 页，2020 年。1, 3, 5, 7, 8, 16

[47] Muhammed Kocabas、Chun-Hao P. Huang、Otmar Hilliges 和 Michael J. Black。PARE：用于三维人体估计的部分注意力回归器。发表于 IEEE/CVF 国际计算机视觉会议 (ICCV) 论文集，第 11127–11137 页，2021 年 10 月。7

[48] Nikos Kolotouros、Georgios Pavlakos、Michael J Black 和 Kostas Daniilidis。通过循环模型拟合学习重建三维人体姿态和形状。发表于 ICCV，第 2252–2261 页，2019 年。2, 7, 8, 16

[49] Nikos Kolotouros、Georgios Pavlakos 和 Kostas Daniilidis. 用于单张图像人体形状重建的卷积网格回归. 发表于 CVPR，第 4501–4510 页，2019 年. 8

[50] Nikos Kolotouros、Georgios Pavlakos、Dinesh Jayaraman 和 Kostas Daniilidis. 人体网格恢复的概率建模. 发表于 ICCV，第 11605–11614 页，2021 年 10 月. 7

[51] Björn Krüger、Anna Vögele、Tobias Willig、Angela Yao、Reinhard Klein 和 Andreas Weber. 运动数据的高效无监督时间分割. IEEE 多媒体汇刊，2017 年. 2

[52] Bio Motion Lab. BMLhandball 运动捕捉数据库. 5

[53] Andreas Lehrmann、Peter V. Gehler 和 Sebastian Nowozin. 用于人体运动的高效非线性马尔可夫模型. 发表于 CVPR，2014 年. 2

[54] Jiefeng Li、Chao Xu、Zhicun Chen、Siyuan Bian、Lixin Yang 和 Cewu Lu. Hybrik：一种用于 3D 人体姿态与形状估计的解析-神经混合逆运动学方法. 发表于 CVPR，第 3383–3393 页，2021 年. 7, 8

[55] Maosen Li、Siheng Chen、Xu Chen、Ya Zhang、Yanfeng Wang 和 Qi Tian. 用于基于骨架动作识别的动作-结构图卷积网络. 发表于 CVPR，2019 年. 2

[56] Wenhao Li、Hong Liu、Hao Tang、Pichao Wang 和 Luc Van Gool. Mhformer：用于 3D 人体姿态估计的多假设 Transformer. 发表于 IEEE/CVF 计算机视觉与模式识别会议 (CVPR) 论文集，第 13147–13156 页，2022 年. 2, 6, 9, 16

[57] Zhihao Li、Jianzhuang Liu、Zhensong Zhang、Songcen Xu 和 Youliang Yan. Cliff：将完整帧中的位置信息带入人体姿态与形状估计. 发表于 ECCV，2022 年. 7

[58] Kevin Lin、Lijuan Wang 和 Zicheng Liu. 基于 Transformer 的端到端人体姿态与网格重建. 发表于 CVPR，第 1954–1963 页，2021 年. 7, 8

[59] Kevin Lin、Lijuan Wang 和 Zicheng Liu. Mesh Graphormer. 发表于 ICCV，第 12939–12948 页，2021 年. 7

[60] Lilang Lin、Sijie Song、Wenhan Yang 和 Jiaying Liu. MS2L：用于基于骨架动作识别的多任务自监督学习. 发表于 ACM 国际多媒体会议，第 2490–2498 页，2020 年. 2, 17

[61] Tsung-Yi Lin、Michael Maire、Serge Belongie、James Hays、Pietro Perona、Deva Ramanan、Piotr Dollár 和 C Lawrence Zitnick. Microsoft COCO：上下文中的常见物体. 发表于 ECCV，第 740–755 页，2014 年. 7, 8, 16

[62] Li Linguo、Wang Minsi、Ni Bingbing、Wang Hang、Yang Jiancheng 和 Zhang Wenjun. 通过跨视角一致性学习实现 3D 人体动作表征学习. 发表于 CVPR，2021 年. 2, 7, 17

[63] Chunhui Liu、Yueyu Hu、Yanghao Li、Sijie Song 和 Jiaying Liu. PKU-MMD：用于连续多模态人体动作理解的大规模基准. arXiv 预印本 arXiv:1703.07475，2017 年. 1

[64] Jun Liu、Amir Shahroudy、Mauricio Perez、Gang Wang、Ling-Yu Duan 和 Alex C Kot. NTU RGB+D 120：用于 3D 人体活动理解的大规模基准. IEEE TPAMI，42(10):2684–2701，2019 年. 1, 2, 6, 7

[65] Jun Liu, Amir Shahroudy, Dong Xu, Alex C Kot 和 Gang Wang. 基于骨架的动作识别：使用带信任门的时空长短期记忆网络. IEEE TPAMI, 40(12):3007–3021, 2017. 7

[66] Jun Liu, Gang Wang, Ping Hu, Ling-Yu Duan 和 Alex C Kot. 用于三维动作识别的全局上下文感知注意力长短期记忆网络. 载于 CVPR, 第 1647–1656 页, 2017. 7

[67] Ruixu Liu, Ju Shen, He Wang, Chen Chen, Sen-ching Cheung 和 Vijayan Asari. 注意力机制利用时间上下文：实时三维人体姿态重建. 载于 CVPR, 2020. 6

[68] Ziyu Liu, Hongwen Zhang, Zhenghao Chen, Zhiyong Wang 和 Wanli Ouyang. 解耦与统一图卷积用于基于骨架的动作识别. 载于 CVPR, 第 143–152 页, 2020. 2, 7

[69] Matthew Loper, Naureen Mahmood 和 Michael J. Black. MoSh：从稀疏标记点捕获运动与形状. ACM Transactions on Graphics, 33(6), 2014 年 11 月. 5

[70] Matthew Loper, Naureen Mahmood 和 Michael J Black. Mosh：从稀疏标记点捕获运动与形状. TOG, 33(6):1–13, 2014. 16

[71] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll 和 Michael J Black. SMPL：一种蒙皮多人线性模型. ACM Transactions on Graphics, 34(6):1–16, 2015. 1, 2, 5

[72] Eyes JAPAN Co. Ltd. Eyes Japan MoCap 数据集. 5

[73] Zhengyi Luo, S Alireza Golestaneh 和 Kris M Kitani. 通过运动压缩与细化估计三维人体运动. 载于 ACCV, 2020. 8

[74] Diogo C. Luvizon, David Picard 和 Hedi Tabia. 使用多任务深度学习的二维/三维姿态估计与动作识别. 载于 CVPR, 第 5137–5146 页, 2018. 2

[75] Xiaoxuan Ma, Jiajun Su, Chunyu Wang, Wentao Zhu 和 Yizhou Wang. 基于虚拟标记点的三维人体网格估计. 载于 IEEE/CVF 计算机视觉与模式识别会议论文集 (CVPR), 第 534–543 页, 2023 年 6 月. 2

[76] Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll 和 Michael J. Black. AMASS：以表面形状形式存储的动作捕捉数据集. 载于 ICCV, 第 5441–5450 页, 2019 年 10 月. 1, 3, 4, 5, 16

[77] Christian Mandery, Ömer Terlemez, Martin Do, Nikolaus Vahrenkamp 和 Tamim Asfour. KIT 全身体人运动数据库. 载于 ICAR, 第 329–336 页. IEEE, 2015. 5

[78] Julieta Martinez, Rayat Hossain, Javier Romero 和 James J Little. 三维人体姿态估计的一种简单而有效的基线方法. 载于 ICCV, 第 2640–2649 页, 2017. 2, 5, 6, 7

[79] Dushyant Mehta, Helge Rhodin, Dan Casas, Pascal Fua, Oleksandr Sotnychenko, Weipeng Xu 和 Christian Theobalt. 使用改进卷积神经网络监督的单目野外三维人体姿态估计. 载于 3DV, 第 506–516 页. IEEE, 2017. 8

[80] Raphael Memmesheimer, Simon Häring, Nick Theisen 和 Dietrich Paulus. Skeleton-DML：基于骨架的单样本动作识别的深度度量学习. 载于 WACV, 第 3702–3710 页, 2022. 7

[81] Raphael Memmesheimer, Nick Theisen 和 Dietrich Paulus. SL-DML：用于多模态单样本动作识别的信号级深度度量学习. 载于 ICPR, 2021. 2, 7

[82] Gyeongsik Moon, Juyong Chang, and Kyoung Mu Lee. Camera distance-aware top-down approach for 3d multi-person pose estimation from a single rgb image. In ICCV, 2019. 2

[83] Gyeongsik Moon and Kyoung Mu Lee. I2l-meshnet: Image-to-lixel prediction network for accurate 3d human pose and mesh estimation from a single rgb image. In ECCV, pages 752–768, 2020. 2, 7

[84] Meinard Müller, Tido Röder, Michael Clausen, Bernhard Eberhardt, Björn Krüger, and Andreas Weber. Documentation mocap database HDM05. Technical Report CG-2007-2, Universität Bonn（波恩大学）, June 2007. 5

[85] Mahdiar Nekoui and Li Cheng. Enhancing human motion assessment by self-supervised representation learning. In BMVC, 2021. 2

[86] Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked hourglass networks for human pose estimation. In ECCV, 2016. 3, 5

[87] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS. 2019. 16

[88] Georgios Pavlakos, Xiaowei Zhou, and Kostas Daniilidis. Ordinal depth supervision for 3d human pose estimation. In CVPR, pages 7307–7316, 2018. 6

[89] Dario Pavllo, Christoph Feichtenhofer, David Grangier, and Michael Auli. 3d human pose estimation in video with temporal convolutions and semi-supervised training. In CVPR, pages 7753–7762, 2019. 2, 3, 4, 5, 6, 9, 16

[90] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 2020. 4

[91] Javier Romero, Dimitrios Tzionas, and Michael J. Black. Embodied hands: Modeling and capturing hands and bodies together. ACM Transactions on Graphics, (Proc. SIGGRAPH Asia), 2017. 16

[92] Alberto Sabater, Laura Santos, Jose Santos-Victor, Alexandre Bernardino, Luis Montesano, and Ana C. Murillo. One-shot action recognition in challenging therapy scenarios. In CVPR Workshop, 2021. 2, 7

[93] Amir Shahroudy, Jun Liu, Tian-Tsong Ng, and Gang Wang. Ntu rgb+ d: A large scale dataset for 3d human activity analysis. In CVPR, pages 1010–1019, 2016. 1, 6

[94] Wenkang Shan, Zhenhua Liu, Xinfeng Zhang, Shanshe Wang, Siwei Ma, and Wen Gao. P-stmo: Pretrained spatial temporal many-to-one model for 3d human pose estimation. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part V, pages 461–478. Springer, 2022. 2, 6

[95] Dian Shao, Yue Zhao, Bo Dai, and Dahua Lin. Finegym: A hierarchical video dataset for fine-grained action understanding. In CVPR, pages 2616–2625, 2020. 1

[96] Lei Shi, Yifan Zhang, Jian Cheng, and Hanqing Lu. 基于骨架的双流自适应图卷积网络用于动作识别. In CVPR, 2019. 1, 2, 7

[97] Leonid Sigal, Alexandru O Balan, and Michael J Black. Humaneva：用于评估关节人体运动的同步视频和运动捕获数据集及基线算法. IJCV, 87(1):4–27, 2010. 5

[98] Sijie Song, Cuiling Lan, Junliang Xing, Wenjun Zeng, and Jiaying Liu. 用于骨架数据人体动作识别的端到端时空注意力模型. In AAAI, page 4263–4270, 2017. 2

[99] Yang Song, Luis Goncalves, and Pietro Perona. 人体运动模型的无监督学习. In T. Dietterich, S. Becker, and Z. Ghahramani, editors, NeurIPS, volume 14. MIT Press, 2001. 2

[100] Kun Su, Xiulong Liu, and Eli Shlizerman. 预测与聚类：无监督骨架动作识别. In CVPR, pages 9631–9640, 2020. 2

[101] Yukun Su, Guosheng Lin, and Qingyao Wu. 基于运动一致性和连续性的自监督三维骨架动作表示学习. In ICCV, 2021. 2, 7, 17

[102] Jiangxin Sun, Zihang Lin, Xintong Han, Jian-Fang Hu, Jia Xu, and Wei-Shi Zheng. 动作引导的三维人体运动预测. In NeurIPS, volume 34, pages 30169–30180. Curran Associates, Inc., 2021. 2

[103] Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. 用于人体姿态估计的深度高分辨率表示学习. In CVPR, 2019. 3, 4, 6, 16

[104] Xiao Sun, Bin Xiao, Fangyin Wei, Shuang Liang, and Yichen Wei. 积分人体姿态回归. In ECCV, 2018. 2

[105] Yu Sun, Qian Bao, Wu Liu, Yili Fu, Michael J Black, and Tao Mei. 单目单阶段多人三维回归. In ICCV, pages 11179–11188, 2021. 7

[106] Yu Sun, Yun Ye, Wu Liu, Wenpeng Gao, Yili Fu, and Tao Mei. 通过骨架解耦表示从单目图像恢复人体网格. In ICCV, pages 5349–5358, 2019. 3, 7

[107] Fida Mohammad Thoker, Hazel Doughty, and Cees GM Snoek. 骨架对比三维动作表示学习. In ACM International Conference on Multimedia, pages 1655–1663, 2021. 2, 17

[108] Dorra Trabelsi, Samer Mohammed, Faicel Chamroukhi, Latifa Oukhellou, and Yacine Amirat. 基于隐马尔可夫模型回归的自动活动识别无监督方法. IEEE Transactions on Automation Science and Engineering, 2013. 2

[109] Shashank Tripathi, Siddhant Ranade, Ambrish Tyagi, and Amit K. Agrawal. Posenet3d：通过知识蒸馏学习时间一致的三维人体姿态. In 8th International Conference on 3D Vision, 2020. 3

[110] Nikolaus F. Troje. 生物运动的分解：人类步态模式分析与合成框架. Journal of Vision, 2(5):2–2, Sept. 2002. 5

[111] Matthew Trumble, Andrew Gilbert, Charles Malleson, Adrian Hilton, and John P. Collomosse. 全捕获：融合视频和惯性传感器的三维人体姿态估计. In BMVC, 2017. 5

[112] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 4

[113] Timo von Marcard, Roberto Henschel, Michael J Black, Bodo Rosenhahn, and Gerard Pons-Moll. Recovering accurate 3d human pose in the wild using imus and a moving camera. In ECCV, pages 601–617, 2018. 7, 16

[114] Ziniu Wan, Zhengjia Li, Maoqing Tian, Jianbo Liu, Shuai Yi, and Hongsheng Li. Encoder-decoder with multi-level attention for 3d human shape and pose estimation. In ICCV, pages 13033–13042, 2021. 7, 8, 9, 16

[115] Chunyu Wang, Yizhou Wang, and Alan L. Yuille. An approach to pose-based action recognition. In CVPR, pages 915–922, 2013. 2

[116] Jingbo Wang, Sijie Yan, Yuanjun Xiong, and Dahua Lin. Motion guided 3d pose estimation from videos. In European Conference on Computer Vision, pages 764–780. Springer, 2020. 1, 2, 6

[117] Peng Wang, Jun Wen, Chenyang Si, Yuntao Qian, and Liang Wang. Contrast-reconstruction representation learning for self-supervised skeleton-based action recognition. arXiv preprint arXiv:2111.11051, 2021. 2

[118] Xin Wang, Yudong Chen, and Wenwu Zhu. A survey on curriculum learning. IEEE TPAMI, 2022. 5

[119] Tom Wehrbein, Marco Rudolph, Bodo Rosenhahn, and Bastian Wandt. Probabilistic monocular 3d human pose estimation with normalizing flows. In ICCV, 2021. 6

[120] Wen-Li Wei, Jen-Chun Lin, Tyng-Luh Liu, and Hong-Yuan Mark Liao. Capturing humans in motion: Temporal-attentive 3d human pose and shape estimation from monocular video. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2022. 7

[121] Tianhan Xu and Wataru Takano. Graph stacked hourglass networks for 3d human pose estimation. In CVPR, 2021. 6

[122] Yuanlu Xu, Song-Chun Zhu, and Tony Tung. Denserac: Joint 3d pose and shape estimation by dense render-and-compare. In ICCV, pages 7760–7770, 2019. 2

[123] Sijie Yan, Yuanjun Xiong, and Dahua Lin. Spatial temporal graph convolutional networks for skeleton-based action recognition. In AAAI, 2018. 1, 2, 7, 16

[124] Di Yang, Yaohui Wang, Antitza Dantcheva, Lorenzo Garattoni, Gianpiero Francesca, and Francois Bremond. Unik: A unified framework for real-world skeleton-based action recognition. In BMVC, 2021. 7

[125] Siyuan Yang, Jun Liu, Shijian Lu, Meng Hwa Er, and Alex C. Kot. Skeleton cloud colorization for unsupervised 3d action representation learning. In ICCV, 2021. 7, 17

[126] Zhuoqian Yang, Wentao Zhu, Wayne Wu, Chen Qian, Qiang Zhou, Bolei Zhou, and Chen Change Loy. Transmomo: Invariance-driven unsupervised video motion retargeting. In CVPR, 2020. 2

[127] Angela Yao, Juergen Gall, and Luc Gool. Coupled action recognition and pose estimation from multiple views. Int. J. Comput. Vision, page 16–37, 2012. 2

[128] Chun-Han Yao, Jimei Yang, Duygu Ceylan, Yi Zhou, Yang Zhou, and Ming-Hsuan Yang. Learning visibility for robust dense human body estimation. In *European conference on computer vision (ECCV)*, 2022. 7

[129] Raymond Yeh, Yuan-Ting Hu, and Alexander Schwing. Chirality nets for human pose regression. In *NeurIPS*, 2019. 6

[130] Ailing Zeng, Lei Yang, Xuan Ju, Jiefeng Li, Jianyi Wang, and Qiang Xu. Smoothnet: A plug-and-play network for refining human poses in videos. In *European Conference on Computer Vision*. Springer, 2022. 3, 7, 8

[131] Feng Zhang, Xiatian Zhu, Hanbin Dai, Mao Ye, and Ce Zhu. Distribution-aware coordinate representation for human pose estimation. In *CVPR*, June 2020. 16

[132] Feng Zhang, Xiatian Zhu, and Mao Ye. Fast human pose estimation. In *CVPR*, 2019. 2

[133] Hongwen Zhang, Yating Tian, Xinchi Zhou, Wanli Ouyang, Yebin Liu, Limin Wang, and Zhenan Sun. Pymaf: 3d human pose and shape regression with pyramidal mesh alignment feedback loop. In *ICCV*, pages 11446–11456, 2021. 2, 7

[134] Jinlu Zhang, Zhigang Tu, Jianyu Yang, Yujin Chen, and Junsong Yuan. Mixste: Seq2seq mixed spatiotemporal encoder for 3d human pose estimation in video. In *CVPR*, 2022. 2, 4, 5, 6, 9, 16

[135] Ce Zheng, Sijie Zhu, Matias Mendieta, Taojiannan Yang, Chen Chen, and Zhengming Ding. 3d human pose estimation with spatial and temporal transformers. *ICCV*, 2021. 2, 5, 6, 9, 16

[136] Kun Zhou, Xiaoguang Han, Nianjuan Jiang, Kui Jia, and Jiangbo Lu. Hemlets pose: Learning part-centric heatmap triplets for accurate 3d human pose estimation. In *ICCV*, 2019. 2

[137] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In *CVPR*, 2019. 16

[138] Wentao Zhu, Cuiling Lan, Junliang Xing, Wenjun Zeng, Yanghao Li, Li Shen, and Xiaohui Xie. Cooccurrence feature learning for skeleton based action recognition using regularized deep lstm networks. In *AAAI*, 2016. 2

[139] Wentao Zhu, Zhuoqian Yang, Ziang Di, Wayne Wu, Yizhou Wang, and Chen Change Loy. Mocanet: Motion retargeting in-the-wild via canonicalization networks. In *AAAI*, 2022. 2

## A. 附录

## A.1. 实验细节

**设置**。我们使用 PyTorch [87] 实现所提出的模型。所有实验均在配备 8 块 NVIDIA V100 GPU 的 Linux 机器上进行，这些 GPU 用于加速预训练。微调和推理通常只需单块 GPU 即可完成。

**预训练。**

对于 AMASS [76]，我们首先渲染参数化人体模型 SMPL+H [91]，然后使用预定义的回归矩阵 [10] 提取 3D 关键点。我们通过相机投影 [27] 从 Human3.6M [38] 中提取 3D 关键点。我们对 3D 动捕数据采样长度为 $T = 243$ 的运动片段。对于 2D 数据，我们利用 PoseTrack [3] 提供的标注。我们进一步纳入来自未标注视频数据集 InstaVariety [42] 的 2D 运动，由 OpenPose [14] 提取。由于野外视频的有效序列长度要短得多，我们使用 $T = 30$（PoseTrack）和 $T = 81$（InstaVariety）。我们按照先前工作的方法，使用置换和插值将 2D 数据集（COCO 和 OpenPose 格式）的关键点转换为 Human3.6M 格式。我们参考 [32, 123] 设置输入通道数 $C_{\mathrm{in}} = 3$（$x$、$y$ 坐标和置信度）。我们采用随机水平翻转作为数据增强。整个网络使用 Adam [45] 优化器训练 90 个 epoch，学习率为 0.0005，批量大小为 64。损失项的权重为 $\lambda_0 = 20$。我们将 2D 骨架掩码比例（masking ratio）设为 $15\%$，与 BERT [30] 相同。更具体地说，我们使用 10% 的帧级掩码和 5% 的关节点级掩码。我们改变两种类型掩码的比例，仅观察到微小的差异。我们按照 [17] 的方法拟合混合分布，并从中采样每个关节点的噪声。为保持噪声平滑并避免严重抖动，我们首先为 $T_K = 27$ 个关键帧采样噪声 $\boldsymbol{z} \in \mathbb{R}^{T_K \times J}$，然后将其上采样为 $z' \in \mathbb{R}^{T \times J}$，并添加一个较小的 Gaussian 噪声 $\mathcal{N}(0, 0.002^2)$。

**3D 姿态估计。**

2D 骨架由在 MPII [4] 和 Human3.6M [38] 上训练的 2D 姿态估计器提供，遵循常见做法 [27, 89]。对于从头训练，我们训练 60 个 epoch，学习率为 0.0002，批量大小为 32。对于微调，我们加载预训练权重并训练 30 个 epoch。

## 基于骨骼的动作识别

我们使用由 PYSKL [31] 提供的、采用 HRNet [103] 提取的 2D 骨骼序列。随后，我们将骨骼序列上采样至统一长度 $T = 243$。对于 NTU-RGB+D，DSTformer 经全局平均池化后的输出被送入一个 MLP，其中包含 dropout $p = 0.5$、一个具有 2048 个通道的隐藏层、BatchNorm 与 ReLU。我们从零开始训练 200 个 epoch，学习率为 0.001，批量大小为 32。对于骨干网络，我们将学习率设为 0.0001；对于下游 MLP，则将学习率设为 0.001 进行微调。对于 NTU-RGB+D-120 上的一次性（one-shot）识别任务，我们采用 dropout $p = 0.1$，并使用一个线性层得到大小为 2048 的动作表征。训练时的批量大小为 16，每个 batch 包含 8 个动作对。同一动作类别的样本被视为正样本，其余样本则为负样本。我们采用温度系数为 0.1 的有监督对比损失 [44]。

## 人体网格恢复

对于 Human3.6M 数据集，我们采用与 3D 姿态估计任务相同的 2D 骨骼序列。其 SMPL 真值（GT）参数由 MoSh [70] 拟合物理动作捕捉标记点得到。对于 3DPW 数据集，其检测得到的 2D 骨骼序列由 [25] 提供的 DarkPose [131] 得到。3DPW 的 SMPL 真值参数通过 IMU [113] 获取。遵循先前工作 [41, 46, 114]，我们额外加入 COCO [61] 数据集用于训练，并按照 [25] 的方式进行设置。COCO 的伪真值 SMPL 参数由 EFT [40] 提供。依照 [46, 48]，在估计姿态参数时，我们使用 6D 连续旋转表示 [137] 而非原始轴角表示。形状（shape）和姿态（pose）MLP 结构相同，均包含一个具有 512 个通道的隐藏层、BatchNorm 与 ReLU。对于两个数据集，我们以长度 $T = 16$、步长 8 截取运动片段。各损失项的权重分别为 $\lambda_{\mathrm{3D}}^{\mathrm{m}} = 0.5$、$\lambda_{\theta} = 1000$、$\lambda_{\beta} = 1$、$\lambda_{\mathrm{n}} = 20$ 以及 $\lambda_{0}^{\mathrm{m}} = 10$。我们对整个网络进行端到端训练，共 60 个 epoch，批量大小为 128，其中骨干网络的学习率为 0.00005，下游网络的学习率为 0.0005。

## A.2. 附加实验

我们首先对所提出的模型架构进行更深入的分析，随后给出基于骨骼的动作识别上的更多结果。我们还包含一项关于预训练掩码比例的消融实验以及定性结果。

## A.2.1 DSTformer 的分析

我们对所提出的 DSTformer 模型进行探究，以解读其性能提升的原因，并重新评估第 3.2 节中的设计假设。DSTformer 与现有工作 [56,134,135] 之间的主要区别在于双流设计和自适应融合机制。因此，我们探查注意力回归器产生的融合权重 $\alpha _ {  { \mathrm { S } } \mathrm { T } }$ 和 $\alpha _ { \mathrm { T S } }$，以了解 DSTformer 如何平衡两个流并适应不同输入。

我们首先考察不同输入动作如何影响双流融合权重，结果如图 4（左）所示。注意 $\alpha _ { \mathrm { S } } \mathrm { T } } + \alpha _ { \mathrm { T } } \mathrm { S } } = 1$，因此为简便起见，我们仅分析 $\alpha _ {  { \mathrm { S } } \mathrm { T } } }$。我们发现运动程度较高的动作（例如步行）会导致 S-T 流的融合权重较小，这意味着模型倾向于 T-S 流。相反，当输入来自较为静态的动作（例如坐和吃）时，模型倾向于增大 S-T 流的权重。这些结果与我们的假设一致，即两个流可以关注特征的不同时空方面，并且模型可以根据输入特征重新调整两个流的权重。直观上，S-T 分支更专注于空间建模，而 T-S 分支更专注于时间建模。

![](images/2299231489af578781352e29d151384299721cbff9af7bfdf8ccc6372a468167.jpg)

![](images/53588678bc20dfca06f9595cb1c7b46b3eb4d59010da38e06f39bc3063ae7316.jpg)  
Figure 4. 按动作（左）和关节（右）分组的融合权重统计。我们在 Human3.6M 测试集上绘制了 S-T 流的平均融合权重。使用在 Human3.6M 上从头训练的 3D 姿态估计模型的第三层进行探查。

<table><tr><td colspan="2">深度(N) 头数(h)</td><td>通道数(Ce)</td><td>MPJPE</td></tr><tr><td>4</td><td>8</td><td>512</td><td>39.7</td></tr><tr><td>6</td><td>8</td><td>512</td><td>40.4</td></tr><tr><td>5</td><td>6</td><td>512</td><td>39.6</td></tr><tr><td>5</td><td>10</td><td>512</td><td>39.5</td></tr><tr><td>5</td><td>8</td><td>256</td><td>40.4</td></tr><tr><td>5</td><td>8</td><td>512</td><td>39.2</td></tr></table>

Table 8. 模型架构设计对比。所有方法均在 Human3.6M 上从头训练，并以 MPJPE (mm) 进行度量。

我们进一步研究了融合权重在身体关节点上的分布情况。如图 4（右）所示，不同关节点的融合权重差异很大。具体而言，对于时间变化较大的关键点（肢体关节点），模型更倾向于 T-S 分支；而对于较稳定的关键点（躯干关节点），模型则赋予 S-T 分支更高的权重。这一结果也与我们对两个流所采用的"分工与合作"设计相对应。此外，我们还分析了不同身体关节点融合权重的相关性。图 5 表明，来自同一功能部位（例如手臂、腿、头部、躯干）的关节点的融合权重天然地具有相关性，这表明模型隐式地学习到了关节点之间的连接关系，并针对输入变化以相互关联的方式调整它们的权重。

![](images/cb635fb597f1c162f8cb709c840d092e48939ac822db68dcf145834aefca6ce9.jpg)  
图 5. 各关节点融合权重的相关性矩阵。结果在 Human3.6M 测试集上计算得出。我们使用一个在 Human3.6M 上从头训练的 3D 姿态估计模型的第三层进行探测。

## A.2.2 架构超参数分析

我们探讨了不同架构超参数对 DSTformer 性能的影响。在表 8 中，我们改变了堆叠模块的数量（深度）、自注意力头数（heads）和嵌入维度（channels），并在 3D 人体姿态估计任务上测量它们的性能。然后我们选取最优组合，并将其设计不加调整地应用于所有任务。

<table><tr><td rowspan="2">Method</td><td colspan="3">X-Sub</td><td colspan="3">X-View</td></tr>
<tr><td>scratch</td><td>finetune</td><td>diff</td><td>scratch</td><td>finetune</td><td>diff</td></tr>
<tr><td>MS2L [60]</td><td>78.4</td><td>78.6</td><td>+0.2</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>CrosSCLR [62]</td><td>85.2</td><td>86.2</td><td>+1.0</td><td>91.4</td><td>92.5</td><td>+1.1</td></tr>
<tr><td>Skel-Con [107]</td><td>72.9</td><td>79.3</td><td>+6.4</td><td>87.8</td><td>90.4</td><td>+2.6</td></tr>
<tr><td>MCC [101]</td><td>88.5</td><td>89.7</td><td>+1.2</td><td>95.1</td><td>96.3</td><td>+1.2</td></tr>
<tr><td>SCC [125]</td><td>-</td><td>88.0</td><td>-</td><td>-</td><td>94.9</td><td>-</td></tr>
<tr><td>Ours</td><td>87.7</td><td>93.0</td><td>+5.3</td><td>94.1</td><td>97.2</td><td>+3.1</td></tr>
</table>

表 9. 与自监督动作识别方法的定量比较。我们在 NTU-RGB+D 上测量 top-1 准确率。diff 度量了从头训练与微调之间的性能差距。

## A.2.3 与自监督动作识别方法的比较

我们的框架与基于自监督骨架的动作识别研究方向 [60, 62, 101, 107, 125] 也存在一些共同点。这些方法在 3D 骨架序列上设计各种预训练任务，并在不使用动作标签的情况下学习 3D 运动表征（即自监督学习）。据我们所知，我们的方法是首个将 2D 到 3D 提升（2D-to-3D lifting）作为基于骨架的动作识别的预训练任务的方法。通过这种方式，它在 2D 骨架序列上学习具有 3D 感知能力的运动表征。

![](images/4fe66109a057b27547399eab45d8f6dc2dfd0e8dd2b82e446da0f28127d52c51.jpg)  
图 6. 一次性识别（one-shot recognition）的动作表征可视化。我们在通过对比学习在其他 100 个辅助类上训练后，展示了 20 个新类的动作表征。我们计算成对的余弦距离并应用 t-SNE。菱形表示用于一次性识别的有标注样本。

先前的方法学习的是以 3D 骨架序列作为输入的 3D 运动表征（通常通过 RGB-D 设备（如 Azure Kinect）获取），而 2D 运动表征可以更方便地应用于野外视频（来自移动设备或互联网）。我们进行了定量比较，结果如表 9 所示。微调相对于从头训练的精度提升可视为所学运动表征的贡献。尽管我们的网络并非专为动作识别任务而设计（scratch），但在微调性能（finetune）上超越了先前的方法，并表现出显著的性能提升（diff）。

## A.2.4 一次性动作识别（One-shot Action Recognition）

图 6 展示了使用我们微调模型学习到的新类动作表征。可以看到，尽管模型从未见过这些动作，但仍出现了有意义的聚类。我们使用 1-最近邻（1-NN）结合样本进行一次性动作识别，达到了 67.4% 的 top-1 准确率。

## A.2.5 掩码率分析

我们研究不同掩码比例对预训练和下游微调的影响。我们通过 Human3.6M 测试集上的 MPJPE 来衡量性能。帧级掩码和关节点级掩码固定为 2:1。如图 7 所示，当掩码比例小于 45% 时，预训练模型可以很好地学习二维到三维的提升任务，且预训练能够提升微调性能（优于从头训练基线）。随着我们继续增加掩码比例，预训练模型在二维到三维提升任务上的表现显著变差，且预训练对微调性能不再带来收益。我们的观察与 MAE [36] 略有不同，后者在高掩码比例（75%）下效果良好。一种可能的解释是，骨骼是人体运动的高度抽象表示。高掩码比例留下的线索过少，难以补全整个序列，因此模型容易过拟合。还值得注意的是，在我们的预训练任务（二维到三维提升）中，深度通道（原始数据的 $\frac 1 3$）本质上是被掩码的。

![](images/e182c8738d33d61f0a308d017a78c8b39298f479f55164c32ed1dccf21f4985b.jpg)  
图 7. 预训练掩码比例的消融研究。对于每个掩码比例，我们分别测试预训练模型在有/无微调情况下的 Human3.6M MPJPE。

## A.3. 局限性

本研究的主要局限性在于其聚焦于单人骨骼序列，导致运动表示对外观、环境和交互等因素不敏感。为了克服这些局限性，未来工作可以探索将学习到的运动表示与通用视频特征或场景表示相结合。