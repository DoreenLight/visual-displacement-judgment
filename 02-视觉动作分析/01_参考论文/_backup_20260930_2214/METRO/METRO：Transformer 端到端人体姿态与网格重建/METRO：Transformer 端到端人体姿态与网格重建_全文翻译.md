# 基于 Transformer 的端到端人体姿态与网格重建

Kevin Lin Lijuan Wang Zicheng Liu

Microsoft

{keli, lijuanw, zliu}@microsoft.com

## 摘要

我们提出了一种新方法，称为 MEsh TRansfOrmer（METRO，网格 Transformer），用于从单张图像重建 3D 人体姿态和网格顶点。我们的方法使用 Transformer 编码器联合建模顶点-顶点和顶点-关节之间的交互，并同时输出 3D 关节坐标和网格顶点。与现有的回归姿态和形状参数的方法相比，METRO 不依赖于任何参数化网格模型（如 SMPL（一种参数化人体网格模型）），因此可以轻松扩展到其他物体，例如手部。我们进一步放宽了网格拓扑结构，允许 Transformer 自注意力机制在任意两个顶点之间自由地进行注意力计算，从而能够学习网格顶点和关节之间的非局部关系。通过所提出的掩码顶点建模（masked vertex modeling），我们的方法在处理部分遮挡等具有挑战性的场景时更加鲁棒和有效。METRO 在公开的 Human3.6M 和 3DPW 数据集上取得了人体网格重建的最新最优结果。此外，我们展示了 METRO 在野外（in the wild）3D 手部重建任务上的泛化能力，在 FreiHAND 数据集上优于现有的最新最优方法。代码和预训练模型可在 https://github.com/microsoft/MeshTransformer 获取。

## 1. 引言

从单张图像进行 3D 人体姿态与网格重建受到了广泛关注，因为它在虚拟现实、体育动作分析、神经退行性疾病诊断等诸多领域有着广泛应用。由于人体复杂的关节运动以及遮挡问题，这是一项具有挑战性的任务。

该领域的近期工作大致可分为两类。第一类方法使用诸如 SMPL [29] 之类的参数化模型，并学习预测形状和姿态系数 [14, 26, 39, 22, 24, 34, 44, 23]。这一方法取得了巨大的成功。参数化模型中编码的强先验信息增强了对环境变化的鲁棒性。然而该方法的一个缺点是，姿态和形状空间受限于构建参数化模型时所用的有限样本。为了克服这一局限，第二类方法不使用任何参数化模型 [25, 8, 32]。这些方法要么使用图卷积神经网络建模邻域内的顶点-顶点交互 [25, 8]，要么使用 1D 热力图（heatmap）回归顶点坐标 [32]。这类方法的一个局限是，它们在建模非局部的顶点-顶点交互时效率不高。

![](images/0b0d84a2585f85139a71009d94b0b68c5aaf0baa507f7ab00701e222e36cf658.jpg)  
(a)

![](images/bc636c9fa8ad1610828b2823cbcfe3c5980c670beca1c2d5cda566a49bf5733f.jpg)  
(b)

![](images/0961f3c661787d29ba4ec3855469280d6b9f3362c6c6d9bbe3db2cfc1f205456.jpg)  
(c)  
图 1：METRO 学习人体关节点与网格顶点之间的非局部交互以进行人体网格重建。给定 (a) 中的输入图像，METRO 通过考虑非局部交互来预测人体网格。(b) 展示了被遮挡的手腕关节点与网格顶点之间的注意力，颜色越亮表示注意力越强。(c) 是重建后的网格。

研究人员已经证明，属于不同身体部位（例如手和脚）的非局部顶点之间存在强相关性[55]。在计算机图形学和机器人学中，已经开发出逆向运动学技术[2]（inverse kinematics），用于根据末端执行器（例如指尖）的位置估计关节铰接图形的内部关节点位置。我们认为，学习人体关节点和网格顶点之间的相关性——包括短距离和长距离的相关性——对于处理身体形状重建中具有挑战性的姿态和遮挡具有重要价值。在本文中，我们提出了一种简单而有效的框架来建模全局的顶点-顶点交互。我们框架的核心组成部分是一个 transformer。

最近的研究表明，transformer[53]显著提升了自然语言处理中各种任务的性能[4, 9, 40, 41]。这一成功主要归功于 transformer 的自注意力（self-attention）机制，它在建模输入和输出中与距离无关的依赖关系（或交互）方面特别有效。基于这些依赖关系，transformer 能够软搜索相关的词元（token），并根据重要特征进行预测[4, 53]。

在本工作中，我们提出 METRO——一种具有渐进式维度缩减的多层 Transformer 编码器——以从给定输入图像同时重建 3D 人体关节点和网格顶点。我们设计了一个掩码顶点建模（Masked Vertex Modeling）目标，并采用 Transformer 编码器架构，以增强关节点与顶点之间的交互。如图 1 所示，METRO 学习发现人体关节点和网格顶点之间的短距离和长距离交互，这有助于更好地重建具有大幅姿态变化和遮挡的 3D 人体形状。

在多个公共数据集上的实验结果表明，METRO 在学习顶点-顶点和顶点-关节点交互方面是有效的，因此在人体网格重建上以很大幅度优于以往工作。据我们所知，METRO 是首个利用 Transformer 编码器架构从单张输入图像联合学习 3D 人体姿态和网格重建的方法。此外，METRO 是一个通用框架，可以轻松应用于预测不同的 3D 网格，例如从输入图像重建 3D 手部。

总结而言，我们做出以下贡献。

• 我们提出了一种新的基于 Transformer 的方法，名为 METRO，用于从单张图像重建 3D 人体姿态和网格。

• 我们设计了掩码顶点建模（Masked Vertex Modeling）目标函数，并采用多层 Transformer 编码器来同时建模顶点-顶点（vertex-vertex）和顶点-关节（vertex-joint）之间的交互，从而实现更好的重建效果。

• METRO 在大规模基准数据集 Human3.6M 和具有挑战性的 3DPW 数据集上取得了新的当前最优性能。

• METRO 是一个通用的多任务框架，可以方便地扩展用于预测不同类型的 3D 网格，例如在实验中所展示的 3D 手部网格。METRO 在论文投稿时在 FreiHAND 排行榜上取得了第一名。

## 2. 相关工作

人体网格重建（Human Mesh Reconstruction, HMR）：HMR 是一项重建 3D 人体形状的任务，是近年来一个活跃的研究课题。尽管早期工作已经展示了使用各种传感器（如深度传感器 [33, 48] 或惯性测量单元 [20, 54]）进行重建的出色效果，研究人员正在探索使用更高效便捷的单目相机设置。然而，由于复杂的姿态变化、遮挡以及有限的 3D 训练数据，从单张图像进行 HMR 仍然十分困难。

已有研究建议采用预训练的参数化人体模型，即 SMPL [29]、STAR [35]、MANO [43]，并估计参数化模型的姿态和形状系数以进行 HMR。由于直接从输入图像回归姿态和形状系数具有挑战性，最近的研究进一步提出利用各种人体先验，例如人体骨骼 [26, 39] 或分割图 [34]，并探索不同的优化策略 [24, 22, 51, 14] 以及时间信息 [23] 来改善重建效果。

另一方面，研究人员还提出了不采用参数化人体模型、而是直接从输入图像回归 3D 人体形状的方法。例如，已有研究探索了使用 3D 网格 [25, 8]、体素空间 [52] 或占据场（occupancy field）[46, 47] 来表示人体。在这些已有工作中，每项研究都针对其特定的应用场景采用了一种特定的输出表示。其中，与本文相关的研究是 GraphCMR [25]，它旨在使用图卷积神经网络（Graph Convolutional Neural Networks, GCNNs）来回归 3D 网格顶点。此外，最近提出的 Pose2Mesh [8] 是一个使用 GCNNs 的级联模型。Pose2Mesh 根据给定的人体姿态表示重建人体网格。

尽管基于 GCNN 的方法 [8, 25] 被设计为基于预定义的网格拓扑来建模邻接顶点-顶点之间的交互，但在建模长距离交互方面效率较低。相比之下，METRO 不受任何网格拓扑的限制，能够建模关节和网格顶点之间的全局交互。此外，我们的方法采用自注意力机制进行学习，这不同于已有研究 [8, 25]。

Attentions and Transformers: 最近的研究 [36, 28, 53] 表明，注意力机制能够提升在各种语言任务上的性能。其核心思想是学习对相关输入的注意力，以便软搜索（soft-search）出对预测输出重要的部分 [4]。Vaswani 等人 [53] 进一步提出了一种完全基于注意力机制的 Transformer 架构。Transformer 利用多头自注意力机制实现了高度并行化，从而能够高效地进行训练和推理，并在大规模语言建模任务中取得了优异的性能，如 BERT [9] 和 GPT [40, 41, 5] 所展示的。

受神经语言领域近期成功的启发，越来越多的研究开始探索将 Transformer 架构应用于各种视觉任务，例如学习图像生成的像素分布 [7, 37] 和图像分类 [7, 10]，或将目标检测简化为集合预测问题 [6]。然而，在 3D 人体重建方向上，尚未有沿着该思路的探索。

![](images/a439e6b5db67e8d2e3d0fc56b70510976c5d771454e504f5d3adc06ae0b7ee40.jpg)  
图 2：所提框架概览。给定一张输入图像，我们使用卷积神经网络（CNN）提取图像特征向量。我们通过将图像特征与每个人体关节点 $i$ 的 3D 坐标 $( x _ { i } , y _ { i } , z _ { i } )$ 以及每个网格顶点 $j$ 的 3D 坐标 $( x _ { j } , y _ { j } , z _ { j } )$ 进行拼接，并在拼接结果上叠加一个模板人体网格，从而实现位置编码。给定一组关节点查询（joint queries）和网格顶点查询（vertex queries），我们通过 Transformer 编码器的多层结构进行自注意力运算，并行地回归出人体关节点和网格顶点的 3D 坐标。我们采用渐进式维度递减架构（右侧所示），在层与层之间逐步降低隐藏嵌入维度。最终每一层的每个 token 输出一个关节点或网格顶点的 3D 坐标。每个编码器块包含 4 层和 4 个注意力头。H 表示图像特征向量的维度。

在本研究中，我们提出了一种带有渐进式维度递减的多层 Transformer 架构，用于回归关节点和顶点的 3D 坐标。

## 3. 方法

图 2 展示了所提框架的概览。该框架以一张 224 × 224 的图像作为输入，预测出一组人体关节点 J 和网格顶点 V。所提框架由两个模块组成：卷积神经网络和多层 Transformer 编码器。首先，我们使用 CNN 从输入图像中提取图像特征向量。然后，多层 Transformer 编码器以该特征向量作为输入，并行地输出人体关节点和网格顶点的 3D 坐标。下面我们将详细描述每个模块。

## 3.1. 卷积神经网络

在框架的第一个模块中，我们使用卷积神经网络（Convolutional Neural Network, CNN）进行特征提取。CNN 在 ImageNet 分类任务上进行了预训练 [45]。具体而言，我们从最后一个隐藏层提取特征向量 $X$。提取的特征向量 $X$ 的维度通常为 2048。我们将特征向量 $X$ 输入到 Transformer 中用于回归任务。

通过这种通用化设计，它支持人体姿态与网格重建的端到端训练。此外，Transformer 能够方便地受益于大规模预训练的 CNN，例如 HRNets [56]。在实验中，我们对输入特征进行了分析，发现高分辨率图像特征有助于 Transformer 回归人体关节点与网格顶点的 3D 坐标。

## 3.2. 具有渐进式维度约减的多层 Transformer 编码器

由于我们需要输出 3D 坐标，因此不能直接套用现有的 Transformer 编码器架构 [10, 6]，因为它们在所有 Transformer 层中都使用恒定维度的隐藏嵌入。受 [18] 通过多个模块逐步执行维度约减的启发，我们设计了一种采用渐进式维度约减策略的新架构。如图 2 右所示，我们使用线性投影在每个编码器层之后降低隐藏嵌入的维度。通过堆叠多个编码器层，模型可被视为以交替方式执行自注意力与维度约减。我们 Transformer 编码器的最终输出向量即为关节点与网格顶点的 3D 坐标。

如图 2 左所示，Transformer 编码器的输入是人体关节点与网格顶点的查询（query）。受位置编码（positional encoding）[53, 25, 13] 的启发，我们使用一个模板人体网格来保留输入序列中每个查询的位置信息。具体而言，我们将图像特征向量 $X \in \mathbb { R } ^ { 2 \bar { 0 } 4 8 \times 1 }$ 与每个关节点 $i$ 的 3D 坐标 $( x _ { i } , y _ { i } , z _ { i } )$ 进行拼接，由此形成一组关节点查询 $\dot { Q _ { J } } = \{ q _ { 1 } ^ { J } , q _ { 2 } ^ { J } , . . . , q _ { n } ^ { J } \}$ ，其中 $q _ { i } ^ { J } \in \mathbb { R } ^ { 2 0 5 1 \times 1 }$ 。类似地，我们对每个网格顶点 $j$ 进行相同的位置编码，并形成一组顶点查询 $\mathsf { \bar { Q } } _ { V } = \{ q _ { 1 } ^ { \dot { V } } , q _ { 2 } ^ { V } , \hdots , q _ { m } ^ { V } \}$ ，其中 $q _ { j } ^ { V } \in \mathbb { R } ^ { 2 0 5 1 \times 1 }$ 。

## 3.3. 掩码顶点建模（Masked Vertex Modeling）

以往的工作 [9, 49] 使用掩码语言建模（Masked Language Modeling, MLM）来学习训练语料的语义特性。然而，MLM 旨在恢复输入内容，这无法直接应用于我们的回归任务。

为了充分激活我们 Transformer 编码器中的双向注意力，我们为回归任务设计了掩码顶点建模（Masked Vertex Modeling，MVM）。我们随机遮蔽一定比例的输入查询（query）。不同于像掩码语言模型（MLM）[9] 那样恢复被遮蔽的输入，我们让 Transformer 对所有关节和顶点进行回归。

为了预测与缺失查询对应的输出，模型将不得不借助其他相关查询。这在精神上类似于模拟遮挡，即部分身体部位不可见。因此，MVM 强制 Transformer 通过考虑其他相关顶点和关节来回归 3D 坐标，而不考虑它们之间的距离和网格拓扑结构。这促进了关节和顶点之间短程和长程的交互，有利于更好的人体建模。

## 3.4. 训练

为了训练 Transformer 编码器，我们在 Transformer 输出的基础上应用损失函数，并最小化预测值与真实值之间的误差。给定数据集 $D = \{ I ^ { i } , \bar { V } _ { 3 D } ^ { i } , \bar { J } _ { 3 D } ^ { i } , \bar { J } _ { 2 D } ^ { i } \} _ { i = 1 } ^ { T }$，其中 T 是训练图像的总数。$\boldsymbol { I } \in \mathbb { R } ^ { w \times h \times 3 }$ 表示一张 RGB 图像。$\bar { V } _ { 3 D } \in \mathbf { \mathbb { R } } ^ { M \times \bar { 3 } }$ 表示网格顶点的真实 3D 坐标，M 是顶点的数量。$\bar { J } _ { 3 D } \in \mathbb { R } ^ { K \times 3 }$ 表示人体关节的真实 3D 坐标，K 是一个人的关节数量。类似地，$\bar { J } _ { 2 D } \in \mathbb { R } ^ { K \times 2 }$ 表示人体关节的真实 2D 坐标。

设 $V _ { 3 D }$ 表示输出的顶点位置，$J _ { 3 D }$ 表示输出的关节位置，我们使用 $L _ { 1 }$ 损失来最小化预测值与真实值之间的误差：

$$
\mathcal { L } _ { V } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \big | \big | V _ { 3 D } - \bar { V } _ { 3 D } \big | \big | _ { 1 } ,\tag{1}
$$

$$
\mathcal { L } _ { J } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \big | \big | J _ { 3 D } - \bar { J } _ { 3 D } \big | \big | _ { 1 } .\tag{2}
$$

值得注意的是，3D 关节也可以从预测的网格中计算得出。遵循文献 [8, 22, 25, 24] 中的常见做法，我们使用预定义的回归矩阵 $G \in \mathbb { R } ^ { K \times M }$，并通过 $J _ { 3 D } ^ { r e g } = G V _ { 3 D }$ 得到回归的 3D 关节。类似于先前的工作，我们使用 $L _ { 1 }$ 损失来优化 ${ { J } _ { 3 D } ^ { r e g } }$

$$
\mathcal { L } _ { J } ^ { r e g } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \left| \left| J _ { 3 D } ^ { r e g } - \bar { J } _ { 3 D } \right| \right| _ { 1 } .\tag{3}
$$

2D 重投影已被广泛用于增强图像-网格对齐 [22, 25, 24]。此外，它还有助于在图像中可视化重建结果。受先前工作启发，我们使用估计的相机参数将 3D 关节点投影到 2D 空间，并最小化 2D 投影与 2D 真值之间的误差：

$$
\mathcal { L } _ { J } ^ { p r o j } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \left| \left| J _ { 2 D } - \bar { J } _ { 2 D } \right| \right| _ { 1 } ,\tag{4}
$$

其中相机参数是通过在 transformer 编码器输出之上使用一个线性层来学习的。

为了进行大规模训练，充分利用 2D 和 3D 训练数据集以获得更好的泛化能力是非常理想的。正如文献 [34, 22, 25, 24, 23, 8, 32] 中所探讨的，我们采用混合训练策略，利用不同的训练数据集，无论其是否包含配对的图像-网格标注。我们的总体目标函数如下：

$$
\mathcal { L } = \alpha \times ( \mathcal { L } _ { V } + \mathcal { L } _ { J } + \mathcal { L } _ { J } ^ { r e g } ) + \beta \times \mathcal { L } _ { J } ^ { p r o j } ,\tag{5}
$$

其中 $\alpha$ 和 $\beta$ 是针对每个样本的二值标志，分别表示该样本是否具有 3D 和 2D 真值标注。

## 3.5. 实现细节

我们的方法能够处理任意大小的网格。然而，受限于当前硬件的内存约束，我们的 transformer 处理的是一个粗糙网格：（1）我们使用一个粗糙模板网格（431 个顶点）进行位置编码，由 transformer 输出该粗糙网格；（2）我们使用可学习的多层感知器（Multi-Layer Perceptrons，MLPs）将粗糙网格上采样到原始网格（SMPL 人体网格拓扑为 6890 个顶点）；（3）transformer 和 MLPs 以端到端方式进行训练；请注意，粗糙网格通过子采样算法 [42] 对原始网格进行两次子采样得到 431 个顶点。正如文献 [25] 中所讨论的，先学习粗糙网格再上采样的实现方式有助于减少计算量。它还有助于避免原始网格中由于顶点的空间局部性而产生的冗余，从而提高训练效率。

## 4. 实验结果

我们首先展示我们的方法在 Human3.6M 和 3DPW 数据集上优于之前最先进的人体网格重建方法。然后，我们提供消融

<table><tr><td rowspan="2">Method</td><td colspan="3">3DPW</td><td colspan="2">Human3.6M</td></tr><tr><td>MPVE↓</td><td>MPJPE↓</td><td>PA-MPJPE↓</td><td>MPJPE↓</td><td>PA-MPJPE↓</td></tr><tr><td>HMR [22]</td><td>—</td><td>—</td><td>81.3</td><td>88.0</td><td>56.8</td></tr><tr><td>GraphCMR [25]</td><td>—</td><td>—</td><td>70.2</td><td>—</td><td>50.1</td></tr><tr><td>SPIN [24]</td><td>116.4</td><td>—</td><td>59.2</td><td>—</td><td>41.1</td></tr><tr><td>Pose2Mesh [8]</td><td>—</td><td>89.2</td><td>58.9</td><td>64.9</td><td>47.0</td></tr><tr><td>I2LMeshNet [32]</td><td>—</td><td>93.2</td><td>57.7</td><td>55.7</td><td>41.1</td></tr><tr><td>VIBE [23]</td><td>99.1</td><td>82.0</td><td>51.9</td><td>65.6</td><td>41.4</td></tr><tr><td>METRO（Ours）</td><td>88.2</td><td>77.1</td><td>47.9</td><td>54.0</td><td>36.7</td></tr></table>

表 1：在 3DPW 和 Human3.6M 数据集上与最先进方法的性能比较。

研究和对非局部交互以及模型设计的洞察。最后，我们展示了我们的模型在手部重建上的泛化能力。

## 4.1. 数据集

参考文献 [34, 22, 25, 24, 23, 8, 32]，我们使用 3D 和 2D 训练数据进行混合训练。下面我们描述每个数据集。

Human3.6M [21] 是一个包含 2D 和 3D 标注的大规模数据集。每张图像中都有一个受试者执行不同的动作。由于许可证问题，真值 3D 网格无法获得。因此，我们使用 [8, 32] 中提供的伪 3D 网格进行训练。伪标签是通过 SMPLify-X [38] 模型拟合创建的。为了公平比较，我们使用 Human3.6M 中提供的真值 3D 姿态标签进行评估。按照常见设置 [50, 25, 22]，我们使用受试者 S1、S5、S6、S7 和 S8 来训练模型，使用受试者 S9 和 S11 来测试模型。

3DPW [54] 是一个包含 2D 和 3D 标注的户外图像数据集。训练集包含 22K 张图像，测试集包含 35K 张图像。按照先前的最先进方法 [23]，我们在 3DPW 上进行实验时使用 3DPW 训练数据。

UP-3D [26] 是一个户外图像数据集。其 3D 标注是通过模型拟合创建的。训练集包含 7K 张图像。

MuCo-3DHP [31] 是基于 MPI-INF-3DHP 数据集 [30] 的合成数据集。它使用各种真实世界背景图像合成训练数据，包含 200K 张训练图像。

COCO [27] 是一个包含 2D 标注的大规模数据集。我们还使用 [24] 中提供的伪 3D 网格标签，这些标签是通过 SMPLify-X [38] 拟合得到的。

MPII [1] 是一个包含 2D 姿态标签的户外图像数据集。训练集包含 14K 张图像。

FreiHAND [61] 是一个 3D 手部数据集。训练集包含 130K 张图像，测试集包含 4K 张图像。我们在该数据集上展示了模型的泛化能力，使用其提供的数据集进行训练，并在其在线服务器上进行评估。

## 4.2. 评估指标

我们使用以下三个标准指标来报告结果。三个指标的单位均为毫米（mm）。

MPJPE：均关节位置误差（Mean-Per-Joint-Position-Error, MPJPE）[21] 是用于评价人体三维姿态的指标 [22, 24, 8]。MPJPE 衡量真实关节点与预测关节点之间的欧氏距离。

PA-MPJPE：PA-MPJPE，又称重建误差 [60]，是该任务的另一个评价指标。它首先使用普氏分析（Procrustes Analysis, PA）[12] 进行三维对齐，然后计算 MPJPE。PA-MPJPE 常用于评估三维重建 [60]，因为它衡量的是重建结构的误差，而不考虑尺度与刚性姿态（即平移和旋转）。

MPVE：均顶点误差（Mean-Per-Vertex-Error, MPVE）[39] 衡量真实顶点与预测顶点之间的欧氏距离。

## 4.3. 主要结果

我们在 3DPW 和 Human3.6M 数据集上，将 METRO 与之前的最先进方法进行比较。遵循文献 [23, 24, 22, 25] 的做法，我们使用三维和二维训练数据进行混合训练。结果如表 1 所示。我们的方法在两个数据集上均优于已有工作。

首先，我们关注 Transformer 在 3DPW 野外场景三维重建中的表现。如表 1 左侧三列所示，我们的方法优于 VIBE [23]，后者曾是该数据集上的最先进方法。值得注意的是，VIBE 是基于视频的方法，而我们的方法是基于图像的方法。

此外，我们在 Human3.6M 的室内场景下评估性能。我们遵循已有工作 [24, 32] 的设置，在训练中不使用 3DPW 数据。

![](images/d2034d4763365bcd72546a68a1ee019374f9c9ee0dab91974bc67f3a9c8eb2b2.jpg)  
图 3：我们方法的定性结果。给定输入图像（左），METRO 考虑关节点与顶点之间的非局部交互来进行人体网格重建（右）。我们将指定关节点与所有其他顶点之间的自注意力进行了可视化，其中更亮的颜色表示更强的注意力。可以观察到，METRO 在关节点与顶点之间发现了丰富的、依赖于输入的交互。

结果如表 1 右侧两列所示。我们的方法取得了更好的重建性能，尤其是在 PA-MPJPE 指标上。

Human3.6M 与 3DPW 这两个数据集面临不同的挑战。3DPW 中的场景遮挡更为严重。Human3.6M 中的场景较为简单，其挑战更多在于如何精确估计人体形状。METRO 在两个数据集上均表现良好，这表明它既对遮挡具有鲁棒性，又能够实现精确的人体形状回归。

## 4.4. 消融实验

# 掩码顶点建模（Masked Vertex Modeling）的有效性

由于我们为 Transformer 设计了一个掩码顶点建模目标，一个有趣的问题是该目标是否有用。表 2 展示了在 Human3.6M 上的消融实验。我们观察到 Masked Vertex Modeling 显著提升了结果。此外，我们研究了应该掩码多少百分比的查询 token（token）。我们对最大掩码百分比进行了调整，表 3 展示了对比结果。随着训练中掩码查询数量的增加，性能有所提升。然而，当掩码超过 30% 的输入查询时，提升效果变得不再显著。这是因为大量缺失的查询会使训练变得更加困难。

<table><tr><td></td><td>MPJPE↓</td><td>PA-MPJPE↓</td></tr><tr><td>w/o MVM</td><td>61.0</td><td>39.1</td></tr><tr><td>w/MVM</td><td>54.0</td><td>36.7</td></tr></table>

表 2：在 Human3.6M 上评估的掩码顶点建模（MVM）目标的消融实验。

<table><tr><td>Max Percentage</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td><td>50%</td></tr><tr><td>PA-MPJPE</td><td>39.1</td><td>37.6</td><td>37.5</td><td>36.7</td><td>38.2</td><td>37.3</td></tr></table>

表 3：在 Human3.6M 上评估的掩码顶点建模目标使用不同掩码查询百分比的消融实验。变量 n% 表示我们随机掩码 0% 到 n% 的输入查询。

<table><tr><td>Method</td><td>PA-MPVPE↓</td><td>PA-MPJPE↓</td><td>F@5 mm ↑</td><td>F@15 mm ↑</td></tr><tr><td>Hasson et al [22]</td><td>13.2</td><td></td><td>0.436</td><td>0.908</td></tr><tr><td>Boukhayma et al. [25]</td><td>13.0</td><td></td><td>0.435</td><td>0.898</td></tr><tr>
<td>FreiHAND [24]</td><td>10.7</td><td></td><td>0.529</td><td>0.935</td></tr><tr><td>Pose2Mesh [8]</td><td>7.8</td><td>7.7</td><td>0.674</td><td>0.969</td></tr><tr><td>I2LMeshNet [32]</td><td>7.6</td><td>7.4</td><td>0.681</td><td>0.973</td></tr><tr><td>METRO (Ours)</td><td>6.3</td><td>6.5</td><td>0.731</td><td>0.984</td></tr></tr></table>

表 4：在 FreiHAND 在线服务器上评估的与最先进方法的性能比较。METRO 以较大优势超越了之前的最先进方法。

**非局部交互**：为了进一步理解 METRO 在学习关节和网格顶点之间交互方面的效果，我们对 Transformer 中的自注意力进行了分析。

图 3 展示了自注意力（self-attention）与网格重建的可视化结果。对于图 3 中的每一行，我们展示了输入图像，以及指定关节点与所有网格顶点之间的自注意力。颜色越亮表示注意力越强。在第一行中，被试被严重遮挡，右侧身体部位不可见。在预测右手腕位置时，METRO 会关注相关的非局部顶点，尤其是头部和左手上的顶点。在最下面一行中，被试身体大幅弯曲。在预测头部位置时，METRO 会关注脚部和手部（最下面一行的第 6 列）。这在直觉上是合理的，因为手和脚的位置为身体姿态提供了重要线索，进而有助于推断头部位置。此外，我们观察到模型会根据输入图像进行自注意力。如图 3 的第二行所示，在预测右手腕位置时，METRO 更关注右脚，这与其他三行中的注意力模式有所不同。

我们还对非局部交互进行了定量分析。我们从 3DPW 测试集中随机采样 5000 张图像，并估计一个整体自注意力图。该图为最后一个 Transformer 层中所有注意力头的平均注意力权重。我们在图 4 中可视化了 14 个身体关节点与 431 个网格顶点之间的交互。每个像素表示自注意力的强度，颜色越深表示注意力越强。请注意，前 14 列代表身体关节点，其余代表网格顶点。我们观察到 METRO 对前臂和小腿上的顶点赋予较强的注意力。这与逆运动学（inverse kinematics）文献 [2] 中的结论一致，即连杆机构的内部关节点可以从末端执行器的位置进行估计。

输入表征（Input Representations）：我们通过使用不同的 CNN 主干网络来研究 Transformer 架构的行为。在此实验中，我们使用了 ResNet50 [16] 和 HRNet [56] 的变体。所有主干网络均在 ImageNet [45] 的 1000 类图像分类任务上进行了预训练。对于每个主干网络，我们提取一个全局图像特征向量 $X \in \mathbb{R}^{2048 \times 1}$，并将其输入到 Transformer 中。在表 5 中，我们观察到当使用 ResNet50 主干网络时，我们的 Transformer 取得了具有竞争力的性能。随着我们增加 HRNet 中高分辨率特征图的通道数，我们观察到性能进一步提升。

![](images/9e1133a9b7b21ca973295d618fbc46eedac810f40511be9ea7a40ba421d1c13b.jpg)

Figure 4: 身体关节和网格顶点之间自注意力的可视化。x 轴和 y 轴分别对应查询（queries）和预测输出（predictions）。从左侧起的前 14 列对应身体关节，其余列对应网格顶点。每一行展示了第 j 个查询对应第 i 个输出的注意力权重 $w _ { i , j }$。颜色越深表示注意力越强。

<table><tr><td>骨干网络（Backbone）</td><td>MPJPE↓</td><td>PA-MPJPE↓</td></tr></tr>
<tr><td>ResNet50</td><td>56.5</td><td>40.6</td></tr>
<tr><td>HRNet-W40</td><td>55.9</td><td>38.5</td></tr>
<tr><td>HRNet-W64</td><td>54.0</td><td>36.7</td></tr></table>

Table 5: 在 Human3.6M 上对不同骨干网络（backbone）的评估。所有骨干网络均在 ImageNet 上进行了预训练。我们观察到，增加 HRNet 高分辨率特征图中的滤波器数量对网格回归是有益的。

![](images/c3ea72f5015b495c0ad9f0734a2b96e7e6c75420c6a1e0ab046df64e4b87991d.jpg)  
Figure 5: 我们的方法在 FreiHAND 测试集上的定性结果。我们可视化了指定关节与所有网格顶点之间的自注意力，其中更亮的颜色表示更强的注意力。METRO 是一个通用的框架，可以轻松扩展到 3D 手部重建。

Generalization to 3D Hand in-the-wild: METRO is capable of predicting arbitrary joints and vertices, without the dependencies on adjacency matrix and parametric coefficients. Thus, METRO is highly flexible and general for mesh reconstruction of other objects. To demonstrate this capability, we conduct experiment on FreiHAND [61]. We train our model on FreiHAND from scratch, and evaluate results on FreiHAND online server. Table 4 shows the comparison with the prior works. METRO outperforms previous state-of-the-art methods by a large margin. Without using any external training data, METRO achieved the first place on FreiHAND leaderboard at the time of paper sub-

mission<sup>1</sup>.

Figure 5 shows our qualitative results with non-local interactions. In the appendix, we provide further analysis on the 3D hand joints, and show that the self-attentions learned in METRO are consistent with inverse kinematics [2].

## 5. Conclusion

We present a simple yet effective mesh transformer framework to reconstruct human pose and mesh from a single input image. We propose the Masked Vertex Modeling objective to learn non-local interactions among body joints and mesh vertices. Experimental results show that, our method advances the state-of-the-art performance on 3DPW, Human3.6M, and FreiHAND datasets.

A detailed analysis reveals that the performance improvements are mainly attributed to the input-dependent non-local interactions learned in METRO, which enables predictions based on important joints and vertices, regardless of the mesh topology. We further demonstrate the generalization capability of the proposed approach to 3D hand reconstruction.

## References

（对野生环境下的 3D 手部泛化：METRO 能够预测任意的关节和顶点，而不依赖于邻接矩阵和参数化系数。因此，METRO 对于其他物体的网格重建具有高度的灵活性和通用性。为了展示这一能力，我们在 FreiHAND [61] 上进行了实验。我们从头开始训练我们的模型，并在 FreiHAND 在线服务器上评估结果。表 4 展示了与之前工作的对比。METRO 以较大优势超越了之前的最先进方法。在不使用任何外部训练数据的情况下，METRO 在论文投稿时取得了 FreiHAND 排行榜的第一名<sup>1</sup>。

图 5 展示了带有非局部交互的定性结果。在附录中，我们对 3D 手部关节进行了进一步分析，并表明 METRO 中学到的自注意力与逆运动学 [2] 一致。

## 5. 结论

我们提出了一个简单而有效的网格 Transformer 框架，用于从单张输入图像重建人体姿态和网格。我们提出了 Masked Vertex Modeling 目标，用于学习身体关节和网格顶点之间的非局部交互。实验结果表明，我们的方法在 3DPW、Human3.6M 和 FreiHAND 数据集上推进了最先进的性能。

详细的分析表明，性能提升主要归功于 METRO 中学到的输入依赖的非局部交互，这使得无论网格拓扑如何，预测都能基于重要的关节和顶点进行。我们进一步展示了所提方法对 3D 手部重建的泛化能力。

## 参考文献）

[1] Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler, and Bernt Schiele. 2d human pose estimation: New benchmark and state of the art analysis. In CVPR, 2014. 5

[2] Andreas Aristidou, Joan Lasenby, Yiorgos Chrysanthou, and Ariel Shamir. Inverse kinematics techniques in computer graphics: A survey. In Computer Graphics Forum, 2018. 1, 7, 8, 12

[3] Anurag Arnab, Carl Doersch, and Andrew Zisserman. Exploiting temporal context for 3d human pose estimation in the wild. In CVPR, 2019. 11

[4] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In ICLR, 2015. 2

[5] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020. 2

[6] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In ECCV, 2020. 3

[7] Mark Chen, Alec Radford, Rewon Child, Jeff Wu, Heewoo Jun, Prafulla Dhariwal, and Ilya Sutskever. Generative pretraining from pixels. In ICML, 2020. 2

[8] Hongsuk Choi, Gyeongsik Moon, and Kyoung Mu Lee. Pose2mesh: Graph convolutional network for 3d human pose and mesh recovery from a 2d human pose. In ECCV, 2020. 1, 2, 4, 5, 7, 11, 17

[9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019. 2, 4, 11

[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2, 3

[11] Georgios Georgakis, Ren Li, Srikrishna Karanam, Terrence Chen, Jana Kosecka, and Ziyan Wu. Hierarchical kinematic human mesh recovery. ECCV, 2020. 11

[12] John C Gower. Generalized procrustes analysis. Psychometrika, 1975. 5, 13

[13] Thibault Groueix, Matthew Fisher, Vladimir G Kim, Bryan C Russell, and Mathieu Aubry. 3d-coded: 3d correspondences by deep deformation. In ECCV, 2018. 3

[14] Peng Guan, Alexander Weiss, Alexandru O Balan, and Michael J Black. Estimating human shape and pose from a single image. In ICCV, 2009. 1, 2

[15] Riza Alp Guler and Iasonas Kokkinos. Holopose: Holistic 3d human reconstruction in-the-wild. In CVPR, 2019. 11

[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016. 7

[17] Yihui He, Rui Yan, Katerina Fragkiadaki, and Shoou-I Yu. Epipolar transformers. In CVPR, 2020. 11

[18] Geoffrey E Hinton and Ruslan R Salakhutdinov. Reducing the dimensionality of data with neural networks. Science, 2006. 3, 12

[19] Lin Huang, Jianchao Tan, Ji Liu, and Junsong Yuan. Handtransformer：用于三维手部姿态估计的非自回归结构化建模. In ECCV, 2020. 11

[20] Yinghao Huang, Manuel Kaufmann, Emre Aksan, Michael J. Black, Otmar Hilliges, and Gerard Pons-Moll. Deep inertial poser：从稀疏惯性测量实时学习重建人体姿态. ACM Transactions on Graphics, 37(6):185:1–185:15, Nov. 2018. 2

[21] Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu. Human3.6M：自然环境下三维人体感知的大规模数据集与预测方法. IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(7):1325–1339, 2014. 5

[22] Angjoo Kanazawa, Michael J Black, David W Jacobs, and Jitendra Malik. 人体形状与姿态的端到端恢复. In CVPR, 2018. 1, 2, 4, 5, 7, 11

[23] Muhammed Kocabas, Nikos Athanasiou, and Michael J. Black. VIBE：人体姿态与形状估计的视频推理. In CVPR, 2020. 1, 2, 4, 5, 11

[24] Nikos Kolotouros, Georgios Pavlakos, Michael J Black, and Kostas Daniilidis. 通过环路中模型拟合学习重建三维人体姿态与形状. In ICCV, 2019. 1, 2, 4, 5, 7, 11

[25] Nikos Kolotouros, Georgios Pavlakos, and Kostas Daniilidis. 单张图像人体形状重建的卷积网格回归. In CVPR, 2019. 1, 2, 3, 4, 5, 7, 11, 12, 14

[26] Christoph Lassner, Javier Romero, Martin Kiefel, Federica Bogo, Michael J Black, and Peter V Gehler. Unite the people：闭合三维与二维人体表示之间的环路. In CVPR, 2017. 1, 2, 5

[27] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. Microsoft COCO：上下文中的常见物体. In ECCV, 2014. 5

[28] Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. 基于实例选择性注意力机制的神经关系抽取. In ACL, 2016. 2

[29] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. SMPL：一种蒙皮多人线性模型. ACM Transactions on Graphics, 34(6):248, 2015. 1, 2

[30] Dushyant Mehta, Helge Rhodin, Dan Casas, Pascal Fua, Oleksandr Sotnychenko, Weipeng Xu, and Christian Theobalt. 使用改进 CNN 监督的野外单目三维人体姿态估计. In 3DV, 2017. 5

[31] Dushyant Mehta, Oleksandr Sotnychenko, Franziska Mueller, Weipeng Xu, Srinath Sridhar, Gerard Pons-Moll, and Christian Theobalt. 基于单目 RGB 的单次多人三维姿态估计. In 3DV, 2018. 5

[32] Gyeongsik Moon and Kyoung Mu Lee. I2L-MeshNet：用于从单张 RGB 图像精确估计三维人体姿态与网格的图像到体素预测网络. In ECCV, 2020. 1, 4, 5, 7, 11, 12, 13, 14

[33] Richard A Newcombe, Shahram Izadi, Otmar Hilliges, David Molyneaux, David Kim, Andrew J Davison, Pushmeet Kohli, Jamie Shotton, Steve Hodges, and Andrew W Fitzgibbon. Kinectfusion: Real-time dense surface mapping and tracking. In ISMAR, 2011. 2

[34] Mohamed Omran, Christoph Lassner, Gerard Pons-Moll, Peter Gehler, and Bernt Schiele. Neural body fitting: Unifying deep learning and model based human pose and shape estimation. In 3DV, 2018. 1, 2, 4, 5

[35] Ahmed A A Osman, Timo Bolkart, and Michael J. Black. STAR: A spare trained articulated human body regressor. In ECCV, 2020. 2

[36] Ankur P Parikh, Oscar Tackstr¨ om, Dipanjan Das, and Jakob¨ Uszkoreit. A decomposable attention model for natural language inference. arXiv preprint arXiv:1606.01933, 2016. 2

[37] Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In ICML, 2018. 2

[38] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3d hands, face, and body from a single image. In CVPR, 2019. 5

[39] Georgios Pavlakos, Luyang Zhu, Xiaowei Zhou, and Kostas Daniilidis. Learning to estimate 3d human pose and shape from a single color image. In CVPR, 2018. 1, 2, 5

[40] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. Technical report, 2018. 2

[41] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. Technical report, 2019. 2

[42] Anurag Ranjan, Timo Bolkart, Soubhik Sanyal, and Michael J Black. Generating 3d faces using convolutional mesh autoencoders. In ECCV, 2018. 4

[43] Javier Romero, Dimitrios Tzionas, and Michael J. Black. Embodied hands: Modeling and capturing hands and bodies together. SIGGRAPH Asia, 2017. 2

[44] Yu Rong, Ziwei Liu, Cheng Li, Kaidi Cao, and Chen Change Loy. Delving deep into hybrid annotations for 3d human recovery in the wild. In ICCV, 2019. 1

[45] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 2015. 3, 8

[46] Shunsuke Saito, Zeng Huang, Ryota Natsume, Shigeo Morishima, Angjoo Kanazawa, and Hao Li. Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization. arXiv preprint arXiv:1905.05172, 2019. 2

[47] Shunsuke Saito, Tomas Simon, Jason Saragih, and Hanbyul Joo. Pifuhd: Multi-level pixel-aligned implicit function for high-resolution 3d human digitization. In CVPR, 2020. 2

[48] Daeyun Shin, Zhile Ren, Erik B Sudderth, and Charless C Fowlkes. 3d scene reconstruction with multi-layer depth and epipolar transformers. In ICCV, 2019. 2

[49] Wilson L Taylor. "cloze procedure": A new tool for measuring readability. Journalism quarterly, 1953. 4

[50] Bugra Tekin, Artem Rozantsev, Vincent Lepetit, and Pascal Fua. Direct prediction of 3d body poses from motion compensated sequences. In CVPR, 2016. 5

[51] Hsiao-Yu Tung, Hsiao-Wei Tung, Ersin Yumer, and Katerina Fragkiadaki. Self-supervised learning of motion capture. In NeurIPS, 2017. 2

[52] Gul Varol, Duygu Ceylan, Bryan Russell, Jimei Yang, Ersin Yumer, Ivan Laptev, and Cordelia Schmid. Bodynet: Volumetric inference of 3d human body shapes. In ECCV, 2018. 2

[53] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 1, 2, 3, 12

[54] Timo von Marcard, Roberto Henschel, Michael Black, Bodo Rosenhahn, and Gerard Pons-Moll. Recovering accurate 3d human pose in the wild using imus and a moving camera. In ECCV, 2018. 2, 5

[55] Jiang Wang, Zicheng Liu, Ying Wu, and Junsong Yuan. Mining actionlet ensemble for action recognition with depth cameras. In CVPR, 2012. 1

[56] Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, and Bin Xiao. Deep high-resolution representation learning for visual recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019. 3, 7

[57] Yuanlu Xu, Song-Chun Zhu, and Tony Tung. Denserac: Joint 3d pose and shape estimation by dense render-and-compare. In ICCV, 2019. 11

[58] Wang Zeng, Wanli Ouyang, Ping Luo, Wentao Liu, and Xiaogang Wang. 3d human mesh regression with dense correspondence. In CVPR, 2020. 11

[59] Tianshu Zhang, Buzhen Huang, and Yangang Wang. Object-occluded human shape and pose estimation from a single color image. In CVPR, 2020. 11

[60] Xiaowei Zhou, Menglong Zhu, Georgios Pavlakos, Spyridon Leonardos, Konstantinos G Derpanis, and Kostas Daniilidis. Monocap: Monocular human motion capture using a cnn coupled with a geometric prior. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018. 5

[61] Christian Zimmermann, Duygu Ceylan, Jimei Yang, Bryan Russell, Max Argus, and Thomas Brox. Freihand: A dataset for markerless capture of hand pose and shape from single rgb images. In ICCV, 2019. 5, 8

## APPENDICES

<table><tr><td rowspan="2">Method</td><td colspan="3">3DPW</td><td colspan="2">Human3.6M</td></tr>
<tr><td>MPVE↓</td><td>MPJPE↓</td><td>PA-MPJPE↓</td><td>MPJPE↓</td><td>PA-MPJPE↓</td></tr>
<tr><td>HMR [22]</td><td>—</td><td>—</td><td>81.3</td><td>88.0</td><td>56.8</td></tr>
<tr><td>GraphCMR [25]</td><td></td><td>—</td><td>70.2</td><td>—</td><td>50.1</td></tr>
<tr><td>SPIN [24]</td><td>116.4</td><td>—</td><td>59.2</td><td>—</td><td>41.1</td></tr>
<tr><td>Pose2Mesh [8]</td><td>—</td><td>89.2</td><td>58.9</td><td>64.9</td><td>47.0</td></tr>
<tr><td>I2LMeshNet [32]</td><td></td><td>93.2</td><td>57.7</td><td>55.7</td><td>41.1</td></tr>
<tr><td>VIBE [23]</td><td>99.1</td><td>82.0</td><td>51.9</td><td>65.6</td><td>41.4</td></tr>
<tr><td>HoloPose [15]</td><td>—</td><td>—</td><td>—</td><td>60.2</td><td>46.5</td></tr>
<tr><td>Arnab et al. [3]</td><td>—</td><td>—</td><td>72.2</td><td>77.8</td><td>54.3</td></tr>
<tr><td>DenseRaC [57]</td><td>—</td><td>—</td><td></td><td>—</td><td>48.0</td></tr>
<tr><td>Zhang et al. [59]</td><td>—</td><td>—</td><td>72.2</td><td>—</td><td>41.7</td></tr>
<tr><td>Zeng et al. [58]</td><td>—</td><td>—</td><td>—</td><td>60.6</td><td>39.3</td></tr>
<tr><td>HKMR [11]</td><td>—</td><td>—</td><td>—</td><td>59.6</td><td>43.2</td></tr>
<tr><td>METRO (Ours)</td><td>88.2</td><td>77.1</td><td>47.9</td><td>54.0</td><td>36.7</td></tr>
</table>

Figure 6: 在 3DPW 和 Human3.6M 数据集上的对比中添加参考文献（HKMR [11]、Zeng et al. [58]、Zhang et al. [59]、DenseRaC [57]、Arnab et al. [3]、HoloPose [15]）。

## A. 补充参考文献

我们希望补充相关参考文献（HKMR [11]、Arnab et al. [3]、Zeng et al. [58]、Zhang et al. [59]、DenseRaC [57]、HoloPose [15]）。在新增的参考文献中，HKMR [11] 通过利用预定义的层次化运动学结构来回归 SMPL 参数，该结构由一个根链和对应于 5 个末端执行器（头部、左/右臂、左/右腿）的五条子链组成。HoloPose [15] 估计人体关节的旋转角度，并将其作为先验来指导基于部位的人体网格重建。Zeng et al. [58] 设计了连续 UV 图以保持网格顶点之间的邻接关系。Zhang et al. [59] 将遮挡场景下的任务建模为 UV 图修复问题。由于 3DPW 是一个相对较新的基准数据集，大多数文献仅报告了在 Human3.6M 上的结果，而未报告 3DPW 上的结果。我们已将它们在 Human3.6M 上的结果添加到表 6 中。可以看到，我们的方法以较大的幅度优于所有先前工作。

最近，研究人员正在将 Transformer 模型拓展到其他 3D 视觉任务，例如多视角人体姿态估计 [17] 和基于点云的手部姿态估计 [19]。我们鼓励读者开展这些研究以进行进一步探索。

## B. 实现细节与计算资源

我们使用 PyTorch 和 Huggingface transformer 库开发 METRO。我们在配备 8 块 NVIDIA V100 GPU（32GB 显存）的机器上进行训练，批量大小（batch size）设为 32。每个 epoch 耗时 32 分钟，我们共训练 200 个 epoch。总体而言，训练过程耗时 5 天。我们使用 Adam 优化器和阶梯式学习率衰减策略。我们将 transformer 和 CNN 主干网络的初始学习率均设为 $1 \times 1 0 ^ { - 4 }$。学习率在第 100 个 epoch 时按 10 的因子衰减。我们的多层 transformer 编码器采用随机初始化，CNN 主干网络则使用 ImageNet 预训练权重进行初始化。参照文献 [25, 24]，我们在训练过程中采用标准的数据增强方法。

我们使用配备单块 NVIDIA P100 GPU（16GB 显存）的机器来评估模型的运行速度。使用批量大小 1 时，运行速度约为 12 fps。将批量大小增至 32 后，运行速度可加速至约 24 fps。表 7 展示了 METRO 中各模块的详细信息。

关于我们的掩码顶点建模（masked vertex modeling），参照 BERT [9]，我们使用一个预定义的特殊 [MASK] 标记（本例中为 2051 维的浮点值向量）来替换随机选中的输入查询。

<table><tr><td>Model</td><td>Dimensionality Reduction Scheme</td><td>PA-MPJPE↓</td></tr>
<tr><td>Transformer [53]</td><td> $( H + 3 )  3$ </td><td>208.7</td></tr>
<tr><td>METRO</td><td> $( H + 3 )  H / 2  3$ </td><td>192.1</td></tr>
<tr><td>METRO</td><td> $( H + 3 )  H / 2  H / 4  3$ </td><td>43.8</td></tr>
<tr><td>METRO</td><td> $( H + 3 )  H / 2  H / 4  H / 8  3$ </td><td>36.7</td></tr>
</table>

Table 6: Performance comparison of different dimentionality reduction schemes, evaluated on Human3.6M validation set. Please note that all the transformer variants have the same total number of hidden layers (12 layers) for fair comparison. $H { = } 2 0 4 8$

<table><tr><td colspan="2">CNN (HRNet-W64)</td><td>Transformer</td></tr>
<tr><td># Parameters</td><td>128M</td><td>102M</td></tr>
<tr><td>Inference time</td><td>52.05 ms</td><td>28.22 ms</td></tr>
</table>

Table 7: Number of parameters and inference time per image. The runtime speed is estimated by using batch size 1.

<table><tr><td>Positional Encoding</td><td>PA-MPJPE↓</td></tr>
<tr><td>Sinusoidal [53]</td><td>37.5</td></tr>
<tr><td>Ours</td><td>36.7</td></tr>
</table>

Table 8: Comparison of different positional encoding schemes, evaluated on Human3.6M validation set.

## C. 渐进式降维

由于我们在 Transformer 架构中逐步减小隐藏层的维度，一个有趣的问题是这种渐进式降维方案是否有用。我们针对不同方案进行了消融实验，表 6 给出了对比结果。在表 6 中，"(H+3)→3" 行对应使用一个线性投影 H+3 到 3 的基线方法。其结果较差。"$(H+3)\rightarrow H/2\rightarrow 3$" 行是另一个在整个网络中保持较小维度的基线方法，结果同样不佳。我们的发现是，对于三维网格回归，大跨度（陡峭）的降维效果并不理想。我们的渐进式方案受到文献 [18] 的启发，该方案通过多个模块逐步进行降维。

## D. 位置编码

由于我们的位置编码与传统的不同，有人可能会问如果使用正弦函数 [53] 而非模板网格会怎样。我们将我们的方法与使用正弦函数的传统位置编码进行了比较，表 8 给出了结果。我们看到使用正弦函数的效果略差。这可能是因为直接对坐标进行编码使学习三维坐标回归更为高效。

## E. 定性结果

图 8 展示了在具有挑战性的场景下与此前基于图像的最先进方法 [32, 25] 的定性对比。这些方法仅使用单帧作为输入。在第一行中，受试者大幅度弯腰。此前的工作难以重建出受试者正确的人体形状。相比之下，我们的方法重建出了一个姿态正确且合理的人体网格。在第二行中，受试者被车辆遮挡。我们看到此前的方法对遮挡较为敏感，未能生成正确的人体网格。相比之下，我们的方法在这种遮挡场景下表现得更为鲁棒。在最后一行中，受试者坐在椅子上。与此前最先进的方法相比，我们的方法重建出了更好的人体网格。

图 9 展示了我们方法在三维手部重建上的定性结果。在不对网络架构做任何修改的情况下，我们的方法在手部上同样表现良好，并且对遮挡具有鲁棒性。这体现了我们方法的优势，即可以轻松扩展到其他类型的物体。

## F. 手部关节点的非局部交互

我们进一步对模型学习到的手部关节之间的非局部交互进行定量分析。我们从 FreiHAND 测试集中随机抽取 1000 个样本，并估计一个整体的自注意力图。图 10 展示了 21 个手部关节之间的交互，共有 21 行和 21 列。像素 (i, j) 表示手部关节 i 对关节 j 的注意力大小，颜色越深表示注意力越强。我们可以看到，腕关节（第 0 列）接收到了来自所有关节的较强注意力。直观来看，腕关节在手的运动学树中扮演着类似"根节点"的角色。此外，第 4、8、12 和 16 列也从许多其他关节获得了较强的注意力，这些列分别对应拇指、食指、中指和无名指的指尖。这些指尖是末端执行器 [2]，在逆运动学中可以用来估计内部关节的位置。另一方面，小指的指尖仅接收来自无名指关节的注意力。这可能是因为小指不如其他手指灵活，其运动相对于其他手指来说与无名指更为相关。

<table><tr><td>方法</td><td>PA-MPVPE↓</td><td>PA-MPJPE↓</td><td>F@5 mm↑</td><td>F@15 mm ↑</td></tr><tr><td>I2LMeshNet [32]</td><td>7.6</td><td>7.4</td><td>0.681</td><td>0.973</td></tr><tr><td>METRO</td><td>6.7</td><td>6.8</td><td>0.717</td><td>0.981</td></tr><tr><td>METRO + 测试时增强</td><td>6.3</td><td>6.5</td><td>0.731</td><td>0.984</td></tr></table>

表 9：测试时增强在 FreiHAND 测试集上的有效性。

## G. FreiHAND 上的测试时增强

我们在 FreiHAND 实验中探索了测试时增强，而在 Human3.6M 和 3DPW 实验中没有使用测试时增强。给定一张测试图像，我们对测试图像应用不同的旋转和缩放；然后将这些变换后的图像输入到我们的模型中，并对结果取平均以得到最终的输出网格。为了计算一个平均的 3D 网格，我们执行 3D 对齐（即 Procrustes 分析 [12]）来归一化输出网格。在表 9 中，我们通过实验观察到，这种实现方式有助于在 FreiHAND 测试集上将 PA-MPVPE 提升 0.4。

## H. 局限性

由于 METRO 是一种数据驱动的方法，当测试样本与训练数据差异较大时，其表现可能不佳。我们在图 7 中展示了一些失败的示例，其中测试图像是从互联网上下载的。首先，如图 7(a) 所示，我们观察到如果目标体型与现有训练数据（即 SMPL 标准库风格的数据）差异很大，我们的方法可能无法忠实地重建被试者的肌肉。其次，如图 7(b) 所示，由于训练集中没有手套数据，我们的模型无法正确重建手部网格。最后，本文提出的方法是一种针对特定网格（mesh-specific）的方法。如果我们把预训练的右手模型应用于左手图像，如图 7(c) 所示，模型将无法很好地工作。如何为不同的三维物体开发一个统一的模型是一项有意义的未来工作。

(a) 示例一  
![](images/33865c36048ba60cdbdeecac59817fff7975a2251ad6e3053d671d15cebf3e3d.jpg)

(b) 示例二  
![](images/5c434bf2816a63be8b6db616edbadfa40d73a9382aa361c9ca88a452857aab21.jpg)

(c) 示例三  
![](images/3373feab3ac2f2eb173eebefc204edf73b01d2e5c1a7c236b6ab12e1c8b91f53.jpg)  
图 7：失败案例。当测试样本与训练数据差异较大时，METRO 可能表现不佳。

![](images/f827870fed8d55a27cc9d134cc03c78ced212bc5d53694156c761e4d8887ced6.jpg)  
输入  
GraphCMR [25]  
I2L-M [32]  
本文方法

图 8：本文方法与其他基于单帧方法之间的定性比较。本文方法对具有挑战性的姿态和遮挡更加鲁棒。

![](images/9b1f6afc5b45f95e2484c98a3bdba695d8b683ae9936cbe685100400f6e0f4e5.jpg)  
图 9：本文方法在 FreiHAND 测试集上的定性结果。本文方法可以轻松扩展到三维手部网格的重建。

![](images/089ebcda466f7f6f6605131f174f296d78b74b6e1df86bcec159acc98800d9ce.jpg)  
图 10：手部关节点之间自注意力的可视化。其中 21 行和 21 列对应 21 个手部关节点。像素 (i, j) 表示关节点 i 对关节点 j 的注意力大小。颜色越深表示注意力越强。21 个关节点的定义如图 11 所示。

![](images/e869f1c4be2aefdbcdc75adceeee5c1d715fe697db72fcda055616ec87c30ca4.jpg)  
图 11：手部关节点的定义。该示意图改编自文献 [8]。