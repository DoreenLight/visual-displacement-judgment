# MotionAGFormer：使用 Transformer-GCNFormer 网络增强三维人体姿态估计

Soroush Mehraban<sup>1,2</sup>，Vida Adeli<sup>1,3</sup>，Babak Taati<sup>1,2,3</sup>

<sup>1</sup>KITE Research Institute，<sup>2</sup>Institute of Biomedical Engineering，University of Toronto，<sup>3</sup>Department of Computer Science，University of Toronto

{soroush.mehraban, vida.adeli}@mail.utoronto.ca，babak.taati@uhn.ca

## 摘要

近年来，基于 Transformer 的方法在三维人体姿态估计中表现出优异的性能。然而，这些方法具有整体视角，通过编码所有关节之间的全局关系，并不能精确地捕捉局部依赖关系。本文提出了一种新颖的 Attention-GCNFormer（AGFormer）块，通过两个并行的 Transformer 和 GCNFormer 流来分割通道数。我们提出的 GCN-Former 模块利用相邻关节之间的局部关系，输出与 Transformer 输出互补的新表示。通过以自适应方式融合这两种表示，AGFormer 展现出更好地学习底层三维结构的能力。通过堆叠多个 AGFormer 块，我们以四种不同变体形式提出了 Motion-AGFormer，可根据速度-精度权衡进行选择。我们在两个流行的基准数据集 Human3.6M 和 MPI-INF-3DHP 上评估了我们的模型。MotionAGFormer-B 取得了最先进的结果，P1 误差分别为 38.4 mm 和 16.2 mm。值得注意的是，它仅使用了之前 Human3.6M 数据集上领先模型四分之一的参数，计算效率提高了三倍。代码和模型可在 https://github.com/TaatiTeam/MotionAGFormer 获取。

## 1. 引言

三维空间中的人体姿态估计是一个活跃的研究领域，对众多应用具有重要意义，从增强现实 [21] 和虚拟现实 [26]，到自动驾驶车辆 [2, 9, 43]、人机交互 [27] 等等。鉴于如此广泛的应用范围，对更准确且计算效率更高的姿态估计模型的需求持续增长。在大多数真实场景中，姿态序列以二维形式捕获，这主要归因于标准 RGB 相机的普遍使用。因此，该领域的关键挑战之一是将这些二维序列有效地提升到三维空间。准确的三维人体姿态估计能够提取关于人体运动的丰富时空信息，并深入理解活动和交互行为。最近的三维提升模型利用人体运动固有的时空一致性来提高三维姿态预测的精度。尽管取得了显著进展，但仍有一些重大挑战需要关注。

![](images/06e85691d69a1f01161b546d1468a440d49e5c3491eae9c95b90844ce87f46e2.jpg)  
图 1. 最近的姿态提升方法在 Human3.6M [14] 上的比较（数值越低越好）。MACs/frame 表示每个输出帧的乘累加运算次数。所提出的 MotionAGFormer 展示了不同的变体，并在保持计算效率的同时取得了更优的结果。

Transformer 架构 [41] 最初是为 NLP 任务设计的，现已被应用于各种计算机视觉问题，包括姿态估计。其捕获长程依赖的能力及其固有的自注意力机制使其成为该领域一个有前景的候选方案。然而，单纯依赖全局注意力机制（如标准 Transformer 中所采用的）可能并非姿态估计任务的最优选择。人体运动本质上具有局部空间和时间依赖的结构性特征。

一个主要的关注点在于对随时间变化的骨骼关系进行建模。现有方法主要依赖 Transformer 架构或基于图的模型。虽然 Transformer 擅长捕获长期依赖，但图模型擅长捕获局部依赖。因此，存在将 Transformer 的全局视角与图模型的局部精确性相统一的机会。

此外，对实现 SOTA 精度的追求往往导致开发参数数量越来越多的复杂模型。这些模型尽管精度良好，但在计算效率和快速响应时间至关重要的实际应用中往往变得不切实际。此外，最近模型中的一种主流方法是从帧序列中仅预测中心帧的单个 3D 姿态。这种方法虽然看似高效，但由于需要重新处理多个重叠序列，会导致计算冗余。因此，我们采用了一种简化的推理策略，以最优方式利用序列数据。该方法通过单次前向传播预测输入的完整 3D 序列，从而最大限度地减少冗余。

在本文中，我们介绍了 MotionAGFormer，这是一种为 3D 人体姿态估计量身定制的新型 Transformer-图混合架构。其核心在于，MotionAGFormer 利用 Transformer 的强大能力来捕获全局信息，同时采用图卷积网络（GCNs, Graph Convolutional Networks）来整合局部空间和时间关系。我们使用自适应融合来聚合来自 Transformer 流和图流的特征。通过这样做，我们确保了人体运动的平衡且全面的表示，从而提高了 3D 姿态估计的精度（见图 1）。

总之，我们论文的主要贡献如下：

• 全新设计：我们提出了 MotionAGFormer 模型，其中引入了一个新的 GCNFormer 模块，该模块擅长表示人体姿态序列中固有的局部依赖。

• **效率与灵活性**：i) 我们的 MotionAGFormer 以其轻量化设计和更少参数带来的更快速度而脱颖而出，同时在准确性方面毫不妥协，优于以往的最先进方法。ii) 考虑到多样化的需求，我们提供了 MotionAGFormer 的不同变体，使用户能够根据自身具体需求在准确性与速度之间做出灵活的权衡选择。

• **最先进性能**：MotionAGFormer 在两个具有挑战性的数据集 Human3.6M 和 MPI-INF-3DHP 上均达到了当前最优水平（SOTA）。

## 2. 相关工作

**三维人体姿态估计**。当前解决该问题的方法可从两个角度进行理解。从一个角度看，模型可以根据输入视频进行分类，可以是多视角或单目的。依赖多视角输入的模型 [7, 15, 34, 35, 48] 需要同时使用来自不同视角的多个摄像头，在真实世界场景中可行性较低。从另一个角度看，根据方法论的不同，这些模型可以分为直接三维估计方法或二维到三维提升方法。直接三维估计方法 [31, 32, 39, 52] 直接从视频帧中推理出三维坐标下的关点，无需任何中间步骤。受到高精度二维姿态估计模型的快速发展和广泛可用性的启发，近期更多研究采用二维到三维提升方法，即首先使用现成的二维姿态检测器 [4, 28, 38]，然后将二维坐标提升到三维空间 [40, 47, 50, 51, 53]。在本文中，我们采用二维到三维提升方法，以单目视频作为输入。

基于 Transformer 的方法。Transformer [41] 在不同的视觉任务中已展现出有前景的结果 [10, 37, 45, 54]。在 3D 人体姿态估计领域，PoseFormer [51] 是首个纯粹基于 Transformer 的模型。PoseFormerV2 [50] 通过采用频域表示提升了其计算效率，同时也使其对噪声数据中的突然运动具有鲁棒性。MHFormer [20] 通过提出一个学习多个合理姿态假设（plausible pose hypotheses）的模块来解决自遮挡和深度模糊问题。P-STMO [36] 提出了掩码姿态建模（masked pose modeling），通过对模型进行自监督预训练来降低最终误差。Enfalt 等人 [11] 通过利用掩码令牌建模（masked token modeling）来降低计算复杂度。StridedFormer [19] 将 Transformer 编码器前馈网络中的全连接层替换为跨步卷积（strided convolution），以逐步缩减序列长度并高效地提升中心帧的 3D 姿态。与上述仅对序列中心帧估计 3D 姿态的模型不同，MixSTE [47] 对输入序列中的每一帧都提供 3D 估计。STCFormer [40] 通过将相关性学习分解为空间和时间分量来降低计算复杂度。HSTFormer [33] 提出了层次化 Transformer 编码器，以更好地捕获时空相关性。除了常用的关节-关节注意力（joint-joint attention）之外，HDFormer [3] 还引入了骨骼-关节（bone-joint）和超骨骼-关节（hyperbone-joint）交互。一些工作利用输出的运动表示来完成各种任务。MotionBERT [53] 对为 3D 人体姿态估计任务所学习的模型进行微调，以用于动作识别和 3D 人体网格恢复等任务，而 UPS [12] 则同时训练了一个用于动作识别、3D 姿态估计和早期动作预测的统一模型。

**图卷积网络（Graph Convolutional Network, GCN）。** 基于 GCN 的方法在基于骨骼的动作识别领域取得了显著成功 [5, 18, 22]。尽管它们在 3D 人体姿态估计中具有计算效率 [6, 8, 42, 49]，但通常无法在误差上与基于 Transformer 的对应方法相竞争。这主要是因为它们仅关注局部关节。最近，GLA-GCN [44] 引入了一种自适应 GCN 方法以利用全局表示。通过采用跨步设计来缩减其时间范围，他们在保持较轻内存负载的同时，在 3D 人体姿态估计上与各种基于 Transformer 的模型取得了有竞争力的结果。然而，所提出模块在提取全局表示方面的有效性仍无法与注意力模块相媲美。

混合方法。这些方法使用不同的模块来捕捉输入序列的不同方面，目前尚未得到广泛探索。最近，DC-GCT [16] 提出了基于 GCN 的局部约束模块，以及基于自注意力的全局约束模块，以同时利用输入序列的局部和全局依赖关系。然而，由于其模型被设计为同时处理单个输入帧和帧序列，因此不区分时间维度与空间维度。因此，与基于 Transformer 的方法相比，它难以取得具有竞争力的结果。

## 3. 方法

## 3.1. 预备知识

我们首先回顾 MetaFormer [46] 的概念，它是我们编码器的核心。MetaFormer 可以被视为 Transformer 架构 [41] 的一种泛化形式，其中注意力模块被任何能够在词元（token）之间进行信息转换的混合器所替代。具体而言，对于输入 $\check{X} \in \mathbb{R}^{N \times C}$，其中 $N$ 表示词元数量，$C$ 表示嵌入维度，词元混合器可以形式化地表示为：

$$
Y = \mathrm{TokenMixer}(\mathrm{Norm}(X)) + X, \tag{1}
$$

其中 $\mathrm{Norm}(\cdot)$ 表示归一化方法，例如批归一化或层归一化 [1,13]，$\mathrm{TokenMixer}(\cdot)$ 表示在词元之间组合信息的模块。我们的方法使用两个并行的词元混合器：多头自注意力（Multi-head Self-Attention, MHSA）和图卷积网络（Graph Convolutional Networks, GCNs）[17]，它们各自为信息转换过程做出独特的贡献。

## 3.2. 整体架构

我们的目标是将二维（像素坐标）骨骼序列提升为精确的三维姿态序列。为此，我们提出了 MotionAGFormer 架构，它同时使用注意力（Transformer）和图卷积（GC-NFormer）来提升运动序列。该架构的概览如图 2a 所示。

该模型接受一个带置信度分数的二维输入序列 $\mathbf{X} \in \mathbb{R}^{T \times J \times 3}$，其中 $T$ 和 $J$ 分别表示帧数和关节数。随后，通过一个线性投影层将每个时间帧中的每个关节映射为一个 $d$ 维特征 $\mathbf{F}^{(0)} \in \mathbb{R}^{T \times J \times d}$。然后，将空间位置嵌入 $\mathbf{P}_{pos}^{s} \in \mathbb{R}^{1 \times J \times d}$ 加到词元上。这里需要特别强调的是，我们的模型并未忽略时间词元的顺序，因为该信息在 GCNFormer 流中得以保留（详见 4.5 节消融研究中的进一步讨论）。

经过位置嵌入后，我们使用 N 个 AGFormer 块（第 3.3 节）来计算 $\mathbf { F } ^ { ( i ) } ~ \in ~ \mathbb { R } ^ { T \times J }$ ×d $( i \ = \ 1 , . . . , N )$，以有效捕捉骨骼序列的底层三维结构。最后，我们通过应用线性层和 tanh 激活函数将 $\bar { \mathbf { F } } ^ { ( N ) }$ 映射到更高维度，以计算运动语义 $\mathbf { M } \in \dot { \mathbb { R } } ^ { T \times J \times d ^ { \prime } }$，并使用回归头来估计三维姿态 $\hat { \textbf { P } } \in \mathbb { R } ^ { T \times J \times 3 }$。提升损失包含位置项 $\left( L _ { 3 D } \right)$ 和速度项 $( L _ { \Delta \mathbf { P } } )$，定义如下：

$$
\begin{array} { r l } & { L _ { 3 D } = \Sigma _ { t = 1 } ^ { T } \Sigma _ { j = 1 } ^ { J } \Vert \hat { \mathbf { P } } _ { t , j } - \mathbf { P } _ { t , j } \Vert , } \\ & { L _ { \Delta \mathbf { P } } = \Sigma _ { t = 2 } ^ { T } \Sigma _ { j = 1 } ^ { J } \Vert \Delta \hat { \mathbf { P } } _ { t , j } - \Delta \mathbf { P } _ { t , j } \Vert , } \end{array}\tag{2}
$$

其中 $\Delta \hat { \mathbf P } _ { t } = \hat { \mathbf P } _ { t } - \hat { \mathbf P } _ { t - 1 } , \Delta \mathbf P _ { t } = \mathbf P _ { t } - \mathbf P _ { t - 1 }$。则总提升损失定义为

$$
{ \cal L } = { \cal L } _ { 3 D } + \lambda _ { \Delta \mathbf { P } } { \cal L } _ { \Delta \mathbf { P } } ,\tag{3}
$$

其中常数系数 $\lambda _ { \Delta \mathbf { P } }$ 用于平衡位置精度和运动平滑性。

## 3.3. AGFormer 块

AGFormer 块采用双流架构。每个流包含两个组件：空间 MetaFormer（图 2b）和时间 MetaFormer（图 2c）。空间 MetaFormer 将各个身体关节点作为不同的 token 进行处理，有效捕捉单帧内的帧内关系。而时间 MetaFormer 则将每一帧视为单个 token，从而捕捉随时间变化的帧间关系。两个流之间的关键区别在于其 token 混合器的类型。其中一个流采用 Transformer，而另一个流则采用 GCNFormer。

**Transformer 流。** 该流采用空间多头自注意力（S-MHSA）来捕捉空间关系，随后采用时间多头自注意力（T-MHSA）来捕捉时间关系。S-MHSA 定义为

$$
\begin{array} { r } { \mathrm { S } \mathrm { - } \mathbf { M H S A } ( \mathbf { Q _ { s } } , \mathbf { K _ { s } } , \mathbf { V _ { s } } ) = \mathrm { C o n c a t } ( \mathrm { h e a d _ { i } } , . . . , \mathrm { h e a d _ { h } } ) { \mathbf { W _ { s } } } ^ { ( O ) } , } \\ { \mathrm { h e a d _ { i } } = \mathrm { s o f t m a x } ( \frac { \mathbf { Q _ { s } } ^ { ( i ) } ( \mathbf { K _ { s } } ^ { ( i ) } ) ^ { T } } { \sqrt { d _ { k } } } ) \mathbf { V _ { s } } ^ { ( i ) } , } \end{array}\tag{4}
$$

![](images/9040ca4ff143fc96416114afe51105046bf3dfb82106fc1057dab2fdfd25fe37.jpg)
Figure 2. (a) MotionAGFormer 是一种新颖的架构，包含 N 个双流时空模块，其中一个流使用 Transformers，另一个流利用 GCNFormers。(b) 空间 MetaFormer。每个输入 token 表示人体的一个关节点。(c) 时间 MetaFormer。输入 token 是姿态序列的帧。

其中 $\mathbf{W_s}^{(O)}$ 是投影参数矩阵，h 是并行注意力头的数量，$d_k$ 是 $\mathbf{K_s}$ 的特征维度。为了计算查询矩阵 $\mathbf{Q_s}$、键矩阵 $\mathbf{K_s}$ 和值矩阵 $\mathbf{V_s}$，我们有

$$
\mathbf{Q_s}^i = \mathbf{F_s}\mathbf{W_s}^{(Q,i)}, \mathbf{K_s}^i = \mathbf{F_s}\mathbf{W_s}^{(K,i)}, \mathbf{V_s}^i = \mathbf{F_s}\mathbf{W_s}^{(V,i)},\tag{5}
$$

其中 $\mathbf{F_s} \in \mathbb{R}^{BT \times J \times d}$ 是空间特征，$\mathbf{W_s}^{(Q,i)}$、$\mathbf{W_s}^{(K,i)}$、$\mathbf{W_s}^{(V,i)}$ 是投影矩阵，B 是批大小（batch size）。S-MHSA 的结果随后被送入多层感知机（MLP），然后经过残差连接和层归一化（LayerNorm）。这完成了第一个 MetaFormer，即空间 Transformer。

接下来，我们将 $\mathbf{F_s}$ 重塑为 $\mathbf{F_T} \in \mathbb{R}^{BJ \times T \times d}$，以为 T-MHSA 准备每个关节点的时间特征作为输入。这里我们有

$$
\begin{array}{r} \mathrm{T\text{-}MHSA}(\mathbf{Q_T}, \mathbf{K_T}, \mathbf{V_T}) = \mathrm{Concat}(\mathrm{head_i}, ..., \mathrm{head_h})\mathbf{W_T}^{(O)}, \\ \mathrm{head_i} = \mathrm{softmax}\left(\frac{\mathbf{Q_T}^{(i)}(\mathbf{K_T}^{(i)})^T}{\sqrt{d_k}}\right)\mathbf{V_T}^{(i)}, \end{array}\tag{6}
$$

其中 $\mathbf{Q_T}$、$\mathbf{K_T}$ 和 $\mathbf{V_T}$ 的计算方式与公式 (5) 类似。

GCNFormer 流。与聚合全局信息的 Transformer 流不同，GCNFormer 流专注于骨架序列中存在的局部空间和时间关系。虽然 Transformers 也可以获取局部信息，但加入这一并行流使得模型能够更有效地平衡局部与全局信息的整合（见消融分析 4.5）。我们 GCNFormer 中使用的定制化 GCN 模块 [24] 定义为：

$$
G C N ( \mathbf { F } ^ { ( \mathbf { i } ) } ) = \sigma ( V ^ { l } + \mathrm { N o r m } ( \tilde { D } ^ { - \frac { 1 } { 2 } } \tilde { A } \tilde { D } ^ { - \frac { 1 } { 2 } } \mathbf { F } ^ { ( \mathbf { i } ) } W _ { 1 } + \mathbf { F } ^ { ( \mathbf { i } ) } W _ { 2 } ) .\tag{7}
$$

其中，$\tilde { A } = A + I _ { N }$ 表示添加了自连接的邻接矩阵，$I _ { N }$ 为单位矩阵，$\tilde { D } i i = \Sigma j \tilde { A } j j$ 定义为 $\tilde { A }$ 对角元素的求和，$W _ { 1 } , W _ { 2 }$ 表示每一层特有的可训练权重矩阵。激活函数 $\sigma(\cdot)$（如 ReLU）与批归一化（Batch Normalization）[13] 共同应用。GCN 的输出再经过残差连接和 LayerNorm。

空间 GCNFormer 与时间 GCNFormer 的区别在于它们的邻接矩阵和输入特征。输入特征与 Transformer 流的相似。在空间 GCNFormer 中，邻接矩阵表示人体拓扑结构（图 3a）。而在时间 GCNFormer 中，我们使用 $\begin{array} { r } { { S i m } ( { { \bf \ddot { F } _ { T } } ^ { t _ { i } } } , { \bf F _ { T } } ^ { t _ { j } } ) = ( { \bf \ddot { F } _ { T } } ^ { t _ { i } } ) ^ { T } { \bf F _ { T } } ^ { t _ { j } } } \end{array}$ 计算单个关节在不同时间帧之间的相似度，并选取 K 个最近邻作为图中相连的节点（图 3b）。因此，时间 GCNFormer 中的图拓扑由学习到的节点特征决定。

自适应融合。与 MotionBERT [53] 类似，我们使用自适应融合来聚合 Transformer 流和 GCNFormer 流提取的特征。定义如下：

$$
\mathbf { F } ^ { ( i ) } = \alpha _ { T F } { } ^ { ( i ) } \circ \mathbf { F } _ { \mathbf { T F } } { } ^ { ( i - 1 ) } + \alpha _ { G F } { } ^ { ( i ) } \circ \mathbf { F } _ { \mathbf { G F } } { } ^ { ( i - 1 ) } ,\tag{8}
$$

其中，$\mathbf { F } ^ { ( i ) }$ 表示在第 $i$ 层提取的特征嵌入，$^ { \mathrm { ~ O ~ } }$ 表示逐元素乘法，$\mathbf { F _ { T F } } ^ { \prime } ( i \mathrm { - } 1 ) , ~ \mathbf { F _ { G F } } ^ { ( i \mathrm { - } 1 ) }$ 分别表示第 $i-1$ 层提取的 Transformer 流和 GCNFormer 流特征。自适应融合权重 $\alpha _ { T F }$ 和 $\alpha _ { G F }$ 是

![](images/6a33e70d9192fd5beca3835f68e907f609f67a5248b8064b2241a9e044c1e474.jpg)  
Figure 3. GCNFormer 模块拓扑结构。(a) 空间 GCN-Former 以人体骨架作为其底层拓扑结构。(b) 时间 GCNFormer 使用 K 近邻（K-NN）算法，通过考虑每个关节（例如图中的左腕和右踝）在整个时间帧范围内的最高相似度来确定连接边。经过 K-NN 处理后，每一行与 K 列相连。

定义为

$$
\alpha _ { T F } { } ^ { ( i ) } , \alpha _ { G F } { } ^ { ( i ) } = \mathrm { s o f t m a x } ( W \cdot \mathrm { C o n c a t } ( { \bf F } _ { { \bf T F } } { } ^ { ( i - 1 ) } , { \bf F } _ { { \bf G F } } { } ^ { ( i - 1 ) } ) ) ,\tag{9}
$$

其中 $W$ 是一个可学习的线性变换。

## 4. 实验

我们在两个大规模 3D 人体姿态估计数据集上评估所提出的 Motion-AGFormer 的性能，即 Human3.6M [14] 和 MPI-INF-3DHP [25]。

## 4.1. 数据集与评价指标

Human3.6M 是一个广泛使用的 3D 人体姿态估计室内数据集。它包含 360 万帧视频，涉及 11 位受试者执行 15 种不同的日常活动。为确保公平评估，我们遵循标准方法，使用来自受试者 1、5、6、7 和 8 的数据训练模型，然后在受试者 9 和 11 的数据上进行测试。遵循先前的工作 [40, 50]，我们使用两种协议进行评估。第一种协议（称为 P1）使用平均每关节位置误差（MPJPE），即在对齐根关节（骶骨）后，以毫米为单位衡量估计姿态与真实姿态之间的差异。第二种协议（称为 P2）衡量 Procrustes-MPJPE，即通过对刚体变换对齐真实姿态与估计姿态。

MPI-INF-3DHP 是另一个大规模数据集，在三种不同环境下采集：绿幕、非绿幕和室外环境。遵循先前的工作 [40, 50]，报告 MPJPE、关键点正确百分比（PC）在 150 毫米范围内以及曲线下面积（AUC）作为评价指标。

## 4.2. 实现细节

模型变体。我们构建了四种不同配置的模型，如表 1 所示。我们的基础模型，称为 MotionAGFormer-B，在精确估计和计算成本之间取得了平衡。其余变体根据其参数量和计算需求进行命名，可以根据应用需求选择每个变体，例如在实时处理或更精确的估计之间进行选择。在所有实验中，运动语义维度为 $d ^ { \prime } = 5 1 2$，每个 MLP 的扩展层为 $\alpha = 4$，注意力头数为 $h = 8$，GCNformer 流中的时间邻居数为 $k = 2$。

Table 1. MotionAGFormer 模型变体的详细信息。N：层数。d：隐藏层维度。T：输入帧数。
<table><tr><td>Method</td><td>N</td><td>d</td><td>T</td><td>Params</td><td>MACs</td></tr>
<tr><td>MotionAGFormer-XS</td><td>12</td><td>64</td><td>27</td><td>2.2 M</td><td>1.0 G</td></tr>
<tr><td>MotionAGFormer-S</td><td>26</td><td>64</td><td>81</td><td>4.8M</td><td>6.6 G</td></tr>
<tr><td>MotionAGFormer-B</td><td>16</td><td>128</td><td>243</td><td>11.7 M</td><td>48.3 G</td></tr>
<tr><td>MotionAGFormer-L</td><td>26</td><td>128</td><td>243</td><td>19.0 M</td><td>78.3 G</td></tr>
</table>

Experimental settings. Our model is implemented using PyTorch [30] and executed on a setup with two NVIDIA A40 GPUs. We apply horizontal flipping augmentation for both training and testing following [50, 53]. For model training, we set each mini-batch as 16 sequences. The network parameters are optimized using AdamW [23] optimizer over 90 epochs with a weight decay of 0.01. The initial learning rate is set to 5e-4 with an exponential learning rate decay schedule and the decay factor is 0.99. We use the Stacked Hourglass [29] 2D pose detection results and 2D ground truths on Human3.6M, following [53]. For MPI-INF-3DHP, ground truth 2D detection is used following a similar approach used in comparison baselines [40, 50].

## 4.3. Performance comparison on Human3.6M

We compare our MotionAGFormer with other models on Human3.6M. For a fair comparison, only results of models without extra pre-training on additional data is included. The results, outlined in Table 2, demonstrate that MotionAGFormer-L attains a P1 error of 38.4 mm for estimated 2D pose and 17.3 mm for ground truth 2D pose. Remarkably, this is achieved with approximately half the computational requirements and parameters compared to the previous SOTA (MotionBERT [53]), while being 0.8 mm and 0.5 mm more accurate, respectively. When comparing the Base and Large variants, MotionAGFormer-L shares the same P1 error as MotionAGFormer-B in the presence of noisy data, while exhibiting a 2.1 mm reduction in error when using ground truth 2D data. Furthermore,

(a)  
(b)  
(c)  
![](images/9edeb83edb3f78d75fe02bc377889d44f61122c898cf37390107b84bb54e1824.jpg)  
Figure 4. Qualitative comparisons of MotionAGFormer with STCformer [40], PoseFormerV2 [50], and MotionBERT [53]. The transparent gray skeleton is the ground-truth 3D pose. Blue represents the torso and left part, and Red indicates the right part of the estimated body.

表 2. 在 Human3.6M 上的定量比较。T：输入帧数。CE：仅估计中间帧。P1：MPJPE 误差（毫米）。P2：P-MPJPE 误差（毫米）。P1<sup>†</sup>：基于 2D 真值的 P1 误差。(\*) 表示使用 HRNet [38] 进行 2D 姿态估计。最优和次优分数分别以加粗和下划线标示。各动作的详细结果请参见补充材料。

<table><tr><td>方法</td><td>T</td><td>CE</td><td>参数量</td><td>MACs</td><td>MACs/帧</td><td>P1↓/P2↓</td><td>P1†↓</td></tr><tr><td>*MHFormer [20] CVPR&#x27;22</td><td>351</td><td>√</td><td>30.9 M</td><td>7.0 G</td><td>7,096M</td><td>43.0/34.4</td><td>30.5</td></tr><tr><td>MixSTE [47] CVPR’22</td><td>243</td><td>X</td><td>33.6 M</td><td>139.0 G</td><td>572M</td><td>40.9/32.6</td><td>21.6</td></tr><tr><td>P-STMO [36] ECCV’22</td><td>243</td><td>√</td><td>6.2 M</td><td>0.7 G</td><td>740 M</td><td>42.8/34.4</td><td>29.3</td></tr><tr><td>Stridedformer [19] TMM&#x27;22</td><td>351</td><td>√</td><td>4.0 M</td><td>0.8 G</td><td>801 M</td><td>43.7/35.2</td><td>28.5</td></tr><tr><td>Einfalt et al. [11] WACV&#x27;23</td><td>351</td><td>√</td><td>10.4 M</td><td>0.5 G</td><td>498M</td><td>44.2/35.7</td><td></td></tr><tr><td>STCFormer [40] CVPR’23</td><td>243</td><td>X</td><td>4.7 M</td><td>19.6 G</td><td>80 M</td><td>41.0/32.0</td><td>21.3</td></tr><tr><td>STCFormer-L [40] CVPR’23</td><td>243</td><td>X</td><td>18.9 M</td><td>78.2 G</td><td>321 M</td><td>40.5/31.8</td><td></td></tr><tr><td>PoseFormerV2 [50] CVPR’23</td><td>243</td><td>√</td><td>14.3 M</td><td>0.5 G</td><td>528M</td><td>45.2/35.6</td><td></td></tr><tr><td>UPS [12] CVPR’23</td><td>243</td><td>√</td><td></td><td></td><td></td><td>40.8/32.5</td><td></td></tr><tr><td>GLA-GCN [44] ICCV’23</td><td>243</td><td>√</td><td>1.3 M</td><td>1.5 G</td><td>1,556 M</td><td>44.4/34.8</td><td>21.0</td></tr><tr><td>MotionBERT [53] ICCV&#x27;23</td><td>243</td><td>X</td><td>42.5 M</td><td>174.7 G</td><td>719 M</td><td>39.2/32.9</td><td>17.8</td></tr><tr><td>HDFormer [3] IJCAI&#x27;23</td><td>96</td><td>X</td><td>3.7 M</td><td>0.6 G</td><td>6M</td><td>42.6/33.1</td><td>21.6</td></tr><tr><td>HSTFormer [33] arXiv&#x27;23</td><td>81</td><td>X</td><td>22.7 M</td><td>1.0 G</td><td>13M</td><td>42.7/33.7</td><td>27.8</td></tr><tr><td>DC-GCT [16] arXiv&#x27;23</td><td>81</td><td>√</td><td>3.1 M</td><td>41 M</td><td>41 M</td><td>44.71-</td><td></td></tr><tr><td>MotionAGFormer-XS</td><td>27</td><td>X</td><td>2.2 M</td><td>1.0 G</td><td>37M</td><td>45.1/36.9</td><td>28.1</td></tr><tr><td>MotionAGFormer-S</td><td>81</td><td>X</td><td>4.8M</td><td>6.6 G</td><td>81 M</td><td>42.5/35.3</td><td>26.5</td></tr><tr><td>MotionAGFormer-B</td><td>243</td><td>X</td><td>11.7 M</td><td>48.3 G</td><td>198 M</td><td>38.4/32.6</td><td>19.4</td></tr><tr><td>MotionAGFormer-L</td><td>243</td><td>X</td><td>19.0 M</td><td>78.3 G</td><td>322 M</td><td>38.4/32.5</td><td>17.3</td></tr></tr>

MotionAGFormer-S 仅使用基线方法三分之一的输入帧数，却能在 P1 误差上超越大多数基线方法。类似地，MotionAGFormer-XS 取得了与 PoseTransformerV2 相当的性能，尽管其输入信息量仅为后者的九分之一（27 帧 vs. 243 帧），且参数量约为后者的七分之一。

## 4.4. 在 MPI-INF-3DHP 上的性能比较

在 MPI-INF-3DHP 数据集上评估我们的方法时，由于视频序列较短，我们将基础（base）和大型（large）变体修改为采用 81 帧输入。在所有变体中，我们的方法在 MPJPE 指标上均一致优于其他方法。值得注意的是，我们的大型变体取得了显著的结果，AUC 为 85.3%，P1 误差为 16.2 mm。该结果在 AUC 上以 1.4% 的显著优势、在 P1 误差上以 6.9 mm 的显著优势超越了此前最佳的模型。然而，其 PCK 为 98.2%，比对比模型的 PCK 性能低 0.5%（表 3）。

Table 3. MPI-INF-3DHP 上的定量比较。T：输入帧数。最佳和次佳分数分别以**粗体**和<u>下划线</u>标注。
<table><tr><td>方法（Method）</td><td>T</td><td>PCK↑ AUC↑</td><td>MPJPE↓</td></tr>
<tr><td>MHFormer [20]</td><td>9</td><td>93.8 94.4</td><td>63.3 58.0</td></tr>
<tr><td>MixSTE [47] P-STMO [36]</td><td>27 81</td><td>66.5 75.8</td><td>54.9 32.2</td></tr>
<tr><td>Einfalt 等 [11]</td><td>81</td><td>97.9 95.4 67.6</td><td>46.9</td></tr>
<tr><td>STCFormer [40]</td><td>81</td><td>98.7 83.9</td><td>23.1</td></tr>
<tr><td>PoseFormerV2 [50]</td><td></td><td></td><td></td></tr>
<tr><td></td><td>81</td><td>97.9 78.8</td><td>27.8</td></tr>
<tr><td>GLA-GCN [44]</td><td>81</td><td>98.5 79.1</td><td>27.7</td></tr>
<tr><td>HSTFormer [33]</td><td>81</td><td>97.3 71.5</td><td>41.4</td></tr>
<tr><td>HDFormer [3]</td><td>96</td><td>98.7 72.9</td><td>37.2</td></tr>
<tr><td>MotionAGFormer-XS</td><td>27</td><td>98.2 83.5</td><td>19.2</td></tr>
<tr><td>MotionAGFormer-S</td><td>81</td><td>98.3 84.5</td><td>17.1</td></tr>
<tr><td>MotionAGFormer-B</td><td>81</td><td>98.3 84.2</td><td>18.2</td></tr>
<tr><td>MotionAGFormer-L</td><td>81</td><td>98.2 85.3</td><td>16.2</td></tr></table>

## 4.5. 消融实验（Ablation Studies）

我们在 Human3.6M 数据集上进行了一系列消融实验，以探究 AGFormer 模块的不同设计选择。

首个消融实验研究了 AGFormer 模块数量与模型宽度对 P1 误差的影响。如表 4 所示，整体趋势倾向于构建更深但更窄的模型。有趣的是，采用 16 个 AGFormer 模块且宽度为 128 的模型，与采用 12 个 AGFormer 模块且宽度为 256 的模型性能相当，但前者在显存占用和计算资源上约为后者的三分之一。

消融研究的第二部分探讨了 AGFormer 块中图流（graph stream）的替代模块。结果如表 5 所示。将 GCNFormer 替换为 GCN 块会使 P1 误差增加 0.7 毫米。类似地，使用 GCN 时，将 Temporal GCNFormer 替换为 TCN 会使 P1 误差增加 0.2 毫米；使用 GCNFormer 时，会使 P1 误差增加 0.9 毫米。最后，将 GCN 替换为 CTR-GCN [5] 会使 P1 误差增加 2.5 毫米。我们推测这是由于训练过程中 CTR-GCN 对每个通道拓扑结构的持续细化所致。因此，每个关节不仅从其直接相邻的关节获取信息，还从所有其他关节获取信息。这阻碍了为变换器流（transformer stream）生成互补信息。

为探索位置嵌入对最终性能的影响，我们进行了一系列实验，如表 6 所示。令人惊讶的是，在空间位置嵌入之外加入时间位置嵌入会使 P1 误差增加 0.6 毫米。如第 3.2 节所述，这一结果源于 GCNFormer 流的非置换等变性（non-permutation equivariant）。与变换器不同，我们的网络本质上保持着帧的时间序列。然而，如图 5 所示，与使用空间位置嵌入（0.94 毫米）相比，加入时间位置嵌入带来了更优的加速度误差（0.88 毫米）。

最后，为验证所提出的 AG-Former 块的有效性，表 7 给出了替代块。使用 GCNFormer 时，观察到 P1 误差为 57.5 毫米，说明其在准确捕捉底层 3D 序列结构方面的能力有限。然而，结合 GCNFormer 和变换器的混合方法取得了显著改进，相较于单独使用变换器，P1 误差减少了 5.2 毫米。此外，这两个模块的串行融合不如它们的并行集成有效。

表4. 通过改变AG-Former块的数量和通道数进行P1误差对比。d：每个AGFormer块中的通道数。d′：回归头之前的通道数。在所有实验中T保持为243。

<table><tr><td>层数</td><td> $d$ </td><td> $\overline { { d ^ { \prime } } }$ </td><td>参数量</td><td>MACs</td><td>P1</td></tr>
<tr><td>5</td><td>512</td><td>512</td><td>58M</td><td>252.8 G</td><td>39.2</td></tr>
<tr><td>4</td><td>256</td><td>512</td><td>11.7 M</td><td>54.G G</td><td>40.2</td></tr>
<tr><td>6</td><td>256</td><td>512</td><td>17.5 M</td><td>81.64 G</td><td>39.6</td></tr>
<tr><td>8</td><td>256</td><td>512</td><td>23.3 M</td><td>108.6 G</td><td>39.5</td></tr>
<tr><td>10</td><td>256</td><td>512</td><td>29.1 M</td><td>135.7 G</td><td>38.6</td></tr>
<tr><td>12</td><td>256</td><td>512</td><td>34.9 M</td><td>162.7 G</td><td>38.4</td></tr>
<tr><td>14</td><td>256</td><td>512</td><td>40.7 M</td><td>189.7 G</td><td>38.6</td></tr>
<tr><td>16</td><td>128</td><td>512</td><td>11.7 M</td><td>64.7 G</td><td>38.4</td></tr>
<tr><td>16</td><td>128</td><td>256</td><td>11.7 M</td><td>64.5 G</td><td>38.7</td></tr>
<tr><td>16</td><td>128</td><td>1024</td><td>11.7 M</td><td>64.9 G</td><td>39.2</td></tr>
<tr><td>26</td><td>64</td><td>512</td><td>4.8M</td><td>39.5 G</td><td>39.5</td></tr>
</table>

表5. 在AGFormer块的图流中使用替代模块时的P1误差对比。

<table><tr><td>GCN（图卷积网络）</td><td>TCN（时序卷积网络）</td><td>GCNFormer</td><td>CTR-GCNFormer</td><td>P1</td></tr>
<tr><td>√</td><td>-</td><td>1</td><td>1</td><td>39.1</td></tr>
<tr><td>√</td><td>√</td><td>-</td><td>1</td><td>39.3</td></tr>
<tr><td>1</td><td>√</td><td>√</td><td>-</td><td>39.7</td></tr>
<tr><td>-</td><td>1</td><td>-</td><td>√</td><td>40.9</td></tr>
<tr><td>-</td><td>一</td><td>√</td><td>-</td><td>38.4</td></tr>
</table>

## 4.6. 定性分析

使用时序GCNFormer的邻接矩阵以及三维人体姿态估计的可视化结果对MotionAGFormer-B进行验证。用于验证的实例从Human3.6M的评估集中随机选取。

![](images/420be680ed62a311c53a6fb02f6447b988fdea8c55d9b19cd20eafb5a4837c4d.jpg)  
图5. 使用不同类型位置嵌入时的加速度误差。使用时序位置嵌入产生的误差为0.88 mm，且相比空间位置嵌入、无嵌入以及同时使用两种嵌入的方式具有更快的收敛速度，后三者分别为0.94 mm、0.90 mm和0.97 mm。

定性比较。图 4 将 MotionAGFormer-B 与近期方法进行了比较，包括 STCFormer [40]、PoseFormerV2 [50] 和 Motion-BERT [53]。设计上，2D 检测器的置信度分数仅被纳入 MotionBERT 和 MotionAGFormer-B。总体而言，MotionAGFormer-B 在所有三个样本上都比 PoseFormerV2 和 STCFormer 展现出更好的重建结果，同时保持与 MotionBERT 相媲美的性能。具体而言，在图 4a 中，MotionAG-Former 与其他方法相比表现出更好的对齐效果。在图 4b 中，它显示出略微更优的对齐，而在图 4c 中，其对齐效果略逊于 Motion-BERT。

时间邻接可视化。时间 GCNFormer 的邻接矩阵如图 6 所示。在初始层中，各关节点具有不同的邻接矩阵，这些矩阵会因不同序列中关节位置随时间的变化而有所不同。然而，随着模型深度的推进，它似乎学习到了一种表示，其中每个关节最接近其在相邻帧中的自身状态，从而形成与相邻帧的连接。

## 5. 结论

我们提出了 MotionAGFormer（运动 AG 聚合 Transformer），一种利用 GCNFormer 捕获复杂局部关节关系、并与有效捕获全局关节相互依赖关系的 Transformer 相结合的新方法。这种融合增强了模型理解输入 2D 序列中固有 3D 结构的能力。此外，Motion-AGFormer 提供了多种可适应的变体，允许在速度和精度之间选择最优平衡。经验评估表明，我们的方法在 Human3.6M 和 MPI-INF-3DHP 上超越了其他方法。

![](images/24cd5bacd6136fe50c16c84bf76d35db6d70d7341d9bc742046482955abf04c9.jpg)  
图 6. 在 Human3.6M 数据集上，从随机序列的三个不同关节、在第一层（左）、中间层（中）和最后一层（右）可视化的时间邻接矩阵。K-NN 中的 K 设为 2。

表 6. 使用不同位置嵌入时的 P1 误差比较。
<table><tr><td>时间嵌入</td><td>空间嵌入</td><td>P1</td></tr><tr><td>1</td><td>-</td><td>39.3</td></tr><tr>-</td><td>√</td><td>38.4</td></tr><tr><td>√</td><td>-</td><td>38.9</td></tr><tr><td>√</td><td>√</td><td>40.5</td></tr></table>

表 7. 不同 MetaFormer 集成的比较。所有模型均使用我们的 MotionAGFormer-B 设置在 Human3.6M 上训练。
<table><tr><td>方法</td><td>P1</td></tr><tr><td>仅 GCNFormer</td><td>57.5</td></tr><tr><td>仅 Transformer</td><td>43.6</td></tr><tr><td>GCNFormer → Transformer（顺序）</td><td>39.1</td></tr><tr><td>Transformer → GCNFormer（顺序）</td><td>38.9</td></tr><tr><td>Transformer - GCNFormer（并行）</td><td>38.4</td></tr></table>

## 参考文献

[1] Jimmy Lei Ba、Jamie Ryan Kiros 和 Geoffrey E Hinton。Layer normalization。arXiv 预印本 arXiv:1607.06450，2016 年。3

[2] Peter Bauer、Arij Bouazizi、Ulrich Kressel 和 Fabian B Flohr。Weakly supervised multi-modal 3d human body pose estimation for autonomous driving。发表于 2023 年 IEEE 智能车辆研讨会 (IV)，第 1–7 页。IEEE，2023 年。1

[3] Hanyuan Chen、Jun-Yan He、Wangmeng Xiang、Wei Liu、Zhi-Qi Cheng、Hanbing Liu、Bin Luo、Yifeng Geng 和 Xuansong Xie。HDFormer: High-order directed transformer for 3d human pose estimation。arXiv 预印本 arXiv:2302.01825，2023 年。2, 6, 7, 12

[4] Yilun Chen、Zhicheng Wang、Yuxiang Peng、Zhiqiang Zhang、Gang Yu 和 Jian Sun。Cascaded pyramid network for multi-person pose estimation。发表于 IEEE 计算机视觉与模式识别会议论文集，第 7103–7112 页，2018 年。2

[5] Yuxin Chen、Ziqi Zhang、Chunfeng Yuan、Bing Li、Ying Deng 和 Weiming Hu。Channel-wise topology refinement graph convolution for skeleton-based action recognition。发表于 IEEE/CVF 国际计算机视觉会议论文集，第 13359–13368 页，2021 年。3, 7

[6] Hongsuk Choi、Gyeongsik Moon 和 Kyoung Mu Lee。Pose2Mesh: Graph convolutional network for 3d human pose and mesh recovery from a 2d human pose。发表于计算机视觉–ECCV 2020：第 16 届欧洲会议，英国格拉斯哥，2020 年 8 月 23–28 日，会议论文集 第七部分 16，第 769–787 页。Springer，2020 年。3

[7] Sungho Chun、Sungbum Park 和 Ju Yong Chang。Learnable human mesh triangulation for 3d human pose and shape estimation。发表于 IEEE/CVF 计算机视觉应用冬季会议论文集，第 2850–2859 页，2023 年。2

[8] Hai Ci、Chunyu Wang、Xiaoxuan Ma 和 Yizhou Wang。Optimizing network structure for 3d human pose estimation。发表于 IEEE/CVF 国际计算机视觉会议论文集，第 2262–2271 页，2019 年。3

[9] Phillip Czech、Markus Braun、Ulrich Kreßel 和 Bin Yang。On-board pedestrian trajectory prediction using behavioral features。发表于 2022 年第 21 届 IEEE 机器学习与应用国际会议 (ICMLA)，第 437–443 页。IEEE，2022 年。1

[10] Alexey Dosovitskiy、Lucas Beyer、Alexander Kolesnikov、Dirk Weissenborn、Xiaohua Zhai、Thomas Unterthiner、Mostafa Dehghani、Matthias Minderer、Georg Heigold、Sylvain Gelly 等。An image is worth 16x16 words: Transformers for image recognition at scale。arXiv 预印本 arXiv:2010.11929，2020 年。2

[11] Moritz Einfalt、Katja Ludwig 和 Rainer Lienhart。Uplift and upsample: Efficient 3d human pose estimation with uplifting transformers。发表于 IEEE/CVF 计算机视觉应用冬季会议 (WACV) 论文集，2023 年 1 月。2, 6, 7, 12

[12] Lin Geng Foo、Tianjiao Li、Hossein Rahmani、Qiuhong Ke 和 Jun Liu。Unified pose sequence modeling。发表于 IEEE/CVF 计算机视觉与模式识别会议论文集，第 13019–13030 页，2023 年。2, 6, 12

[13] Sergey Ioffe 和 Christian Szegedy。批量归一化：通过减少内部协变量偏移加速深度网络训练。载于 International conference on machine learning，第 448–456 页。pmlr，2015。3，4

[14] Catalin Ionescu、Dragos Papava、Vlad Olaru 和 Cristian Sminchisescu。Human3.6m：用于自然环境中 3D 人体感知的大规模数据集与预测方法。IEEE transactions on pattern analysis and machine intelligence，36(7)：1325–1339，2013。1，5

[15] Karim Iskakov、Egor Burkov、Victor Lempitsky 和 Yury Malkov。可学习的人体姿态三角测量。载于 Proceedings of the IEEE/CVF international conference on computer vision，第 7718–7727 页，2019。2

[16] Hongbo Kang、Yong Wang、Mengyuan Liu、Doudou Wu、Peng Liu 和 Wenming Yang。用于图像和视频中 3D 人体姿态估计的双链约束。arXiv preprint arXiv:2308.05298，2023。3，6

[17] Thomas N Kipf 和 Max Welling。基于图卷积网络的半监督分类。arXiv preprint arXiv:1609.02907，2016。3

[18] Jungho Lee、Minhyeok Lee、Dogyoon Lee 和 Sangyoun Lee。用于基于骨架动作识别的层次分解图卷积网络。arXiv preprint arXiv:2208.10741，2022。3

[19] Wenhao Li、Hong Liu、Runwei Ding、Mengyuan Liu、Pichao Wang 和 Wenming Yang。利用跨步 Transformer 探索时间上下文进行 3D 人体姿态估计。IEEE Transactions on Multimedia，25：1282–1293，2022。2，6，12

[20] Wenhao Li、Hong Liu、Hao Tang、Pichao Wang 和 Luc Van Gool。MHFormer：用于 3D 人体姿态估计的多假设 Transformer。载于 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)，第 13147–13156 页，2022。2，6，7，12

[21] Huei-Yung Lin 和 Ting-Wen Chen。基于单目 3D 姿态估计的人体交互增强现实。载于 International Conference on Advanced Concepts for Intelligent Vision Systems，第 321–331 页。Springer，2010。1

[22] Ziyu Liu、Hongwen Zhang、Zhenghao Chen、Zhiyong Wang 和 Wanli Ouyang。解耦并统一基于骨架动作识别中的图卷积。载于 Proceedings of the IEEE/CVF conference on computer vision and pattern recognition，第 143–152 页，2020。3

[23] Ilya Loshchilov 和 Frank Hutter。解耦权重衰减正则化。arXiv preprint arXiv:1711.05101，2017。5

[24] Cheng Luo、Siyang Song、Weicheng Xie、Linlin Shen 和 Hatice Gunes。学习基于多维边缘特征的面部动作单元关系图用于面部动作单元识别。arXiv preprint arXiv:2205.01782，2022。4

[25] Dushyant Mehta、Helge Rhodin、Dan Casas、Pascal Fua、Oleksandr Sotnychenko、Weipeng Xu 和 Christian Theobalt。利用改进的 CNN 监督进行野外单目 3D 人体姿态估计。载于 2017 international conference on 3D vision (3DV)，第 506–516 页。IEEE，2017。5

[26] Dushyant Mehta、Srinath Sridhar、Oleksandr Sotnychenko、Helge Rhodin、Mohammad Shafiei、Hans-Peter Seidel、

[26] Weipeng Xu, Dan Casas, and Christian Theobalt. VNect: Real-time 3d human pose estimation with a single rgb camera. *ACM Transactions on Graphics (TOG)*, 36(4):1–14, 2017. 1

[27] Tewodros Legesse Munea, Yalew Zelalem Jembre, Halefom Tekle Weldegebriel, Longbiao Chen, Chenxi Huang, and Chenhui Yang. The progress of human pose estimation: A survey and taxonomy of models applied in 2d human pose estimation. *IEEE Access*, 8:133330–133348, 2020. 1

[28] Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked hourglass networks for human pose estimation. In *Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VIII 14*, pages 483–499. Springer, 2016. 2

[29] Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked hourglass networks for human pose estimation. In *Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VIII 14*, pages 483–499. Springer, 2016. 5

[30] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017. 5

[31] Georgios Pavlakos, Xiaowei Zhou, and Kostas Daniilidis. Ordinal depth supervision for 3d human pose estimation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 7307–7316, 2018. 2

[32] Georgios Pavlakos, Xiaowei Zhou, Konstantinos G Derpanis, and Kostas Daniilidis. Coarse-to-fine volumetric prediction for single-image 3d human pose. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 7025–7034, 2017. 2

[33] Xiaoye Qian, Youbao Tang, Ning Zhang, Mei Han, Jing Xiao, Ming-Chun Huang, and Ruei-Sung Lin. HSTFormer: Hierarchical spatial-temporal transformers for 3d human pose estimation. *arXiv preprint arXiv:2301.07322*, 2023. 2, 6, 7, 12

[34] N Dinesh Reddy, Laurent Guigues, Leonid Pishchulin, Jayan Eledath, and Srinivasa G Narasimhan. TesseTrack: End-toend learnable multi-person articulated 3d pose tracking. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 15190–15200, 2021. 2

[35] Edoardo Remelli, Shangchen Han, Sina Honari, Pascal Fua, and Robert Wang. Lightweight multi-view 3d pose estimation through camera-disentangled representation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6040–6049, 2020. 2

[36] Wenkang Shan, Zhenhua Liu, Xinfeng Zhang, Shanshe Wang, Siwei Ma, and Wen Gao. P-STMO: Pre-trained spatial temporal many-to-one model for 3d human pose estimation. In *Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part V*, pages 461–478. Springer, 2022. 2, 6, 7, 12

[37] Weijie Su, Xizhou Zhu, Chenxin Tao, Lewei Lu, Bin Li, Gao Huang, Yu Qiao, Xiaogang Wang, Jie Zhou, and Jifeng Dai. Towards all-in-one pre-training via maximizing multi-modal mutual information. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15888–15899, 2023. 2

[38] Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. Deep high-resolution representation learning for human pose estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5693–5703, 2019. 2, 6, 12

[39] Xiao Sun, Bin Xiao, Fangyin Wei, Shuang Liang, and Yichen Wei. Integral human pose regression. In Proceedings of the European conference on computer vision (ECCV), pages 529–545, 2018. 2

[40] Zhenhua Tang, Zhaofan Qiu, Yanbin Hao, Richang Hong, and Ting Yao. 3D human pose estimation with spatiotemporal criss-cross attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4790–4799, 2023. 2, 5, 6, 7, 8, 12, 13

[41] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 1, 2, 3

[42] Jingbo Wang, Sijie Yan, Yuanjun Xiong, and Dahua Lin. Motion guided 3d pose estimation from videos. In European Conference on Computer Vision, pages 764–780. Springer, 2020. 3

[43] Julian Wiederer, Arij Bouazizi, Ulrich Kressel, and Vasileios Belagiannis. Traffic control gesture recognition for autonomous vehicles. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 10676–10683. IEEE, 2020. 1

[44] Bruce X. B. Yu, Zhi Zhang, Yongxu Liu, Sheng hua Zhong, Yan Liu, and Chang Wen Chen. GLA-GCN: Global-local adaptive graph convolutional network for 3d human pose estimation from monocular video, 2023. 3, 6, 7, 12

[45] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. CoCa: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022. 2

[46] Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, and Shuicheng Yan. MetaFormer is actually what you need for vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10819–10829, 2022. 3

[47] Jinlu Zhang, Zhigang Tu, Jianyu Yang, Yujin Chen, and Junsong Yuan. MixSTE: Seq2seq mixed spatio-temporal encoder for 3d human pose estimation in video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13232–13242, June 2022. 2, 6, 7, 12

[48] Zhe Zhang, Chunyu Wang, Weichao Qiu, Wenhu Qin, and Wenjun Zeng. Adafuse: Adaptive multiview fusion for accurate human pose estimation in the wild. International Journal ofComputer Vision, 129:703–718, 2021. 2

[49] Long Zhao、Xi Peng、Yu Tian、Mubbasir Kapadia 和 Dimitris N Metaxas。用于 3D 人体姿态回归的语义图卷积网络。发表于 IEEE/CVF 计算机视觉与模式识别会议论文集，第 3425–3435 页，2019 年。3

![](images/96a6acffdd1502a2b8907a8d2f85cabf6914ff0c6a07caab8e18da79254d06a7.jpg)  
图 7. 在 Human3.6M 数据集上与其他模型在 MPJPE（毫米）方面的逐关节误差比较。'L' 和 'R' 分别表示左和右。

[50] Qitao Zhao、Ce Zheng、Mengyuan Liu、Pichao Wang 和 Chen Chen。PoseFormerV2：探索频域以实现高效且鲁棒的 3D 人体姿态估计。发表于 IEEE/CVF 计算机视觉与模式识别会议（CVPR）论文集，第 8877–8886 页，2023 年 6 月。2、5、6、7、8、13

[51] Ce Zheng、Sijie Zhu、Matias Mendieta、Taojiannan Yang、Chen Chen 和 Zhengming Ding。基于空间与时间 Transformer 的 3D 人体姿态估计。发表于 IEEE/CVF 国际计算机视觉会议论文集，第 11656–11665 页，2021 年。2

[52] Kun Zhou、Xiaoguang Han、Nianjuan Jiang、Kui Jia 和 Jiangbo Lu。HEMlets pose：学习以部位为中心的热图三元组以实现精确的 3D 人体姿态估计。发表于 IEEE/CVF 国际计算机视觉会议论文集，第 2344–2353 页，2019 年。2

[53] Wentao Zhu、Xiaoxuan Ma、Zhaoyang Liu、Libin Liu、Wayne Wu 和 Yizhou Wang。MotionBERT：学习人体运动表示的统一视角。发表于 IEEE/CVF 国际计算机视觉会议论文集，2023 年。2、4、5、6、8、12、13

[54] Zhuofan Zong、Guanglu Song 和 Yu Liu。具有协作混合分配的 DETRs。arXiv 预印本 arXiv:2211.12860，2022 年。2

## 附录

## A. Human3.6M 上的额外结果

## A.1. 逐动作结果

为了评估我们所提出的方法在每个动作上的有效性，表 8 给出了我们的方法与替代方法在 Human3.6M 数据集上的 P1 和 P2 误差比较。与其他现有方法相比，我们所提出的方法在 Direction（指示）、Discuss（讨论）、Eating（进食）、Greet（问候）、Purchase（购物）、Sitting（坐下）、Sitting Down（坐下-起立）、Smoke（吸烟）、Wait（等待）和 Walk Dog（遛狗）等各种动作上在 P1 误差方面表现出优越的性能。对于其余动作，我们的方法取得了第二佳的结果。

![](images/84dceb1b06aec64a89443daa9c8586762ea1e33b69acc7560ad844432cdc75f9.jpg)  
图 8. 不同替代模块在捕获局部时间依赖性方面的加速度误差比较。

## A.2. 逐关节误差比较

图 7 展示了 MotionAGFormer-B、MotionBERT、STCFormer 和 StridedFormer 在 Human3.6M 基准数据集上的逐关节比较结果。STC-Former 在髋部关节上表现出轻微的性能优势，而在肩部关节上，MotionAGFormer 略优于 MotionBERT，并显著超越其他模型。MotionBERT 在自由度（DoF）为 2 的关节（包括膝部和肘部）上总体优于 MotionAGFormer，并且与其他模型之间存在显著差距。然而，对于自由度为 3 的关节（如踝部和腕部），MotionAGFormer 始终优于所有其他模型，这一点尤为重要，因为这些关节对整体误差的贡献更大，并且在手/物交互和步态分析等应用中具有重要意义。

## B. 时序 GCNFormer 的消融实验

在主文中，我们考察了时序 GCNFormer 的多种替代方案，并报告了它们对应的 P1 误差。然而，我们尚未研究每种替代方案所对应的加速度误差。加速度误差定义如下：

$$
\begin{array} { r l r } & { } & { A c c ( { \bf P } , \hat { \bf P } ) = \Sigma _ { t = 1 } ^ { T - 1 } L ( { \bf P } , t ) - L ( \hat { \bf P } , t ) , } \\ & { } & { L ( { \bf X } , t ) = { \bf X } _ { t - 1 } - 2 { \bf X } _ { t } + { \bf X } _ { t + 1 } , \quad } \end{array}\tag{10}
$$

其中 $\mathbf{P}$ 和 $\hat{\mathbf{P}}$ 分别表示真实三维姿态和估计三维姿态。

图 8 展示了用于捕捉时序局部依赖的不同模型的加速度误差。有趣的是，尽管 CTR-GCN 和 TCN 导致更高的 P1 误差（如主文所述），但训练过程中它们的加速度误差收敛更快。此外，最终使加速度误差改善了约 0.1 mm。

表 8. 在 Human3.6M 数据集上按动作类别进行的 3D 人体姿态估计定量比较。（上）使用检测到的 2D 姿态序列的 MPJPE（毫米）。（下）使用检测到的 2D 姿态序列的 P-MPJPE（毫米）。（\*）表示使用 HRNet [38] 进行 2D 姿态估计。（†）表示使用其提供的评估代码进行人工评估。

| MPJPE | T | 指示 | 讨论 | 进食 | 打招呼 | 打电话 | 拍照 | 摆姿势 | 购物 | 坐 | 坐下 | 吸烟 | 等待 | 遛狗 | 行走 | 同行 | 平均 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *MHFormer [20] | 351 | 39.2 | 43.1 | 40.1 | 40.9 | 44.9 | 51.2 | 40.6 | 41.3 | 53.5 | 60.3 | 43.7 | 41.1 | 43.8 | 29.8 | 30.6 | 43.0 |
| MixSTE [47] | 243 | 37.6 | 40.9 | 37.3 | 39.7 | 42.3 | 49.9 | 40.1 | 39.8 | 51.7 | 55.0 | 42.1 | 39.8 | 41.0 | 27.9 | 27.9 | 40.9 |
| P-STMO [36] | 243 | 38.9 | 42.7 | 40.4 | 41.1 | 45.6 | 49.7 | 40.9 | 39.9 | 55.5 | 59.4 | 44.9 | 42.2 | 42.7 | 29.4 | 29.4 | 42.8 |
| StridedFormer [19] | 351 | 40.3 | 43.3 | 40.2 | 42.3 | 45.6 | 52.3 | 41.8 | 40.5 | 55.9 | 60.6 | 44.2 | 43.0 | 44.2 | 30.0 | 30.2 | 43.7 |
| Einfalt et al. [11] | 351 | 39.6 | 43.8 | 40.2 | 42.4 | 46.5 | 53.9 | 42.3 | 42.5 | 55.7 | 62.3 | 45.1 | 43.0 | 44.7 | 30.1 | 30.8 | 44.2 |
| STCFormer [40] | 243 | 39.6 | 41.6 | 37.4 | 38.8 | 43.1 | 51.1 | 39.1 | 39.7 | 51.4 | 57.4 | 41.8 | 38.5 | 40.7 | 27.1 | 28.6 | 41.0 |
| STCFormer-L [40] | 243 | 38.4 | 41.2 | 36.8 | 38.0 | 42.7 | 50.5 | 38.7 | 38.2 | 52.5 | 56.8 | 41.8 | 38.4 | 40.2 | 26.2 | 27.7 | 40.5 |
| UPS [12] | 243 | 37.5 | 39.2 | 36.9 | 40.6 | 39.3 | 46.8 | 39.0 | 41.7 | 50.6 | 63.5 | 40.4 | 37.8 | 44.2 | 26.7 | 29.1 | 40.8 |
| GLA-GCN [44] | 243 | 41.3 | 44.3 | 40.8 | 41.8 | 45.9 | 54.1 | 42.1 | 41.5 | 57.8 | 62.9 | 45.0 | 42.8 | 45.9 | 29.4 | 29.9 | 44.4 |
| † MotionBERT [53] | 243 | 36.6 | 39.3 | 37.8 | 33.5 | 41.4 | 49.9 | 37.0 | 35.5 | 50.4 | 56.5 | 41.4 | 38.2 | 37.3 | 26.2 | 26.9 | 39.2 |
| HDFormer [3] | 96 | 38.1 | 43.1 | 39.3 | 39.4 | 44.3 | 49.1 | 41.3 | 40.8 | 53.1 | 62.1 | 43.3 | 41.8 | 43.1 | 31.0 | 29.7 | 42.6 |
| HSTFormer [33] | 81 | 39.5 | 42.0 | 39.9 | 40.8 | 44.4 | 50.9 | 40.9 | 41.3 | 54.7 | 58.8 | 43.6 | 40.7 | 43.4 | 30.1 | 30.4 | 42.7 |
| MotionAGFormer-XS | 27 | 42.7 | 46.7 | 41.1 | 39.4 | 47.0 | 56.3 | 44.4 | 41.0 | 52.4 | 66.3 | 46.7 | 43.2 | 43.4 | 32.4 | 34.0 | 45.1 |
| MotionAGFormer-S | 81 | 41.9 | 42.7 | 40.4 | 37.6 | 45.6 | 51.3 | 41.0 | 38.0 | 54.1 | 58.8 | 45.5 | 40.4 | 39.8 | 29.4 | 31.0 | 42.5 |
| MotionAGFormer-B | 243 | 36.4 | 38.4 | 36.8 | 32.9 | 40.9 | 48.5 | 36.6 | 34.6 | 51.7 | 52.8 | 41.0 | 36.4 | 36.5 | 26.7 | 27.0 | 38.4 |
| MotionAGFormer-L | 243 | 36.8 | 38.5 | 35.9 | 33.0 | 41.1 | 48.6 | 38.0 | 34.8 | 49.0 | 51.4 | 40.3 | 37.4 | 36.3 | 27.2 | 27.2 | 38.4 |
| P-MPJPE | T | 指示 | 讨论 | 进食 | 打招呼 | 打电话 | 拍照 | 摆姿势 | 购物 | 坐 | 坐下 | 吸烟 | 等待 | 遛狗 | 行走 | 同行 | 平均 |
| *MHFormer [20] | 351 | 31.5 | 34.9 | 32.8 | 33.6 | 35.3 | 39.6 | 32.0 | 32.2 | 43.5 | 48.7 | 36.4 | 32.6 | 34.3 | 23.9 | 25.1 | 34.4 |
| MixSTE [47] | 243 | 30.8 | 33.1 | 30.3 | 31.8 | 33.1 | 39.1 | 31.1 | 30.5 | 42.5 | 44.5 | 34.0 | 30.8 | 32.7 | 22.1 | 22.9 | 32.6 |
| P-STMO [36] | 243 | 31.3 | 35.2 | 32.9 | 33.9 | 35.4 | 39.3 | 32.5 | 31.5 | 44.6 | 48.2 | 36.3 | 32.9 | 34.4 | 23.8 | 23.9 | 34.4 |
| StridedFormer [19] | 351 | 32.7 | 35.5 | 32.5 | 35.4 | 35.9 | 41.6 | 33.0 | 31.9 | 45.1 | 50.1 | 36.3 | 33.5 | 35.1 | 23.9 | 25.0 | 35.2 |
| Einfalt et al. [11] | 351 | 32.7 | 36.1 | 33.4 | 36.0 | 36.1 | 42.0 | 33.3 | 33.1 | 45.4 | 50.7 | 37.0 | 34.1 | 35.9 | 24.4 | 25.4 | 35.7 |
| STCFormer [40] | 243 | 29.5 | 33.2 | 30.6 | 31.0 | 33.0 | 38.0 | 30.4 | 29.4 | 41.8 | 45.2 | 33.6 | 29.5 | 31.6 | 21.3 | 22.6 | 32.0 |
| STCFormer-L [40] | 243 | 29.3 | 33.0 | 30.7 | 30.6 | 32.7 | 38.2 | 29.7 | 28.8 | 42.2 | 45.0 | 33.3 | 29.4 | 31.5 | 20.9 | 22.3 | 31.8 |
| UPS [12] | 243 | 30.3 | 32.2 | 30.8 | 33.1 | 31.1 | 35.2 | 30.3 | 32.1 | 39.4 | 49.6 | 32.9 | 29.2 | 33.9 | 21.6 | 24.5 |  |

## C. 序列上的定性比较

除了使用 Human3.6M 数据集中的随机帧进行定性比较外，我们还分析了单个序列内的连续帧。这使我们能够观察对比模型在一段序列上的行为表现，而非仅基于单帧进行定性评估。图 9 将 MotionAGFormer（运动聚合图Transformer）-B 与 STCFormer（时空并发Transformer）[40]、PoseFormerV2 [50] 和 MotionBERT [53] 进行了对比。总体而言，MotionAGFormer 和 MotionBERT 在与真值的对齐效果上优于另外两个模型。在比较 MotionAGFormer 与 MotionBERT 时，MotionAGFormer 在具有自由度（DoF）的关节（例如手腕和脚踝）上表现出略优于 MotionBERT 的性能。此外，也存在一些具有挑战性的序列（见图 10），所有模型都未能准确估计这些关节。

## D. 野外（In-the-wild）评估

为了验证所提出的 MotionAGFormer 的泛化能力，我们在一些未见过的野外视频上测试了基础（base）变体模型。我们的估计结果表明，即使在推断的二维姿态存在一定噪声的情况下（如图 11 中第二个示例的最后一帧所示），相应的三维姿态估计仍然保持合理，并对二维序列中突发的噪声运动具有较强的鲁棒性。此外，当被试者距离摄像机较远时，三维姿态估计仍展现出很高的精度（如图 11 的最后一个示例所示）。

![](images/83f0974fc3244db8329c6f16a5a2b560994de9439be2172a80bdf6b8d8b57f4e.jpg)

Time

MotionAGFormer

图 9. 在 Human3.6M 数据集的随机序列上，将 MotionAGFormer 与 STCformer [40]、PoseFormerV2 [50] 和 MotionBERT [53] 的定性比较。  
![](images/34207cb83284243909e9ea726c50d5fe9f5f177a5ab66f0df97c480fdffb990e.jpg)

![](images/67c4cfd92ef4e6ecda93683c98aa4afd12cd3a738b82017029d82563f4df4e67.jpg)

Time

![](images/733e8be5afb0def96804cb7252e73dd69fe90c8d0aa6dce479f569ca12f9838e.jpg)

![](images/1c53222141fa42783b69e75eaeee5fae8ed766f795aa3cc1f5aaac34cb9fe8b7.jpg)

![](images/4f76a42cd6620371844905b04ca56d5b85c4ee5e0c9063b26e5842d440cb0991.jpg)

STCFormer

![](images/7188d9780303debc205803f7f029a2ebaf65bf2743a9d5b6fd9f05aa446758ba.jpg)

![](images/578a8d6bdd788d35ba9583813b190480d0925901b41cfbd983abab349f4bb4f6.jpg)

![](images/496a12d146a0dc75f27f0346d502908076317699cd2183e28cf1edee7407533e.jpg)

![](images/9201da1d3190d2541836d87f3b9afc1401be80fa965dc3ff0f4e51f213cb54bd.jpg)

![](images/60209afdd7ed4d51e6adb9ed859fc6ddddd977a76485bb3b321268b458529b39.jpg)

PoseFormerV2

![](images/fe36c2fd8788e8f25c1fcbf72d665340592b9fca9cf6d42100947c75102ef21b.jpg)

![](images/6c60374bd68c2a08378814abc05a38d3ddb488b8ef126f8b75708120a62eb9c4.jpg)

![](images/b8bf9d2a300a3353d69b8187ddb8c81030f35d0db8528031c0189986c4d02918.jpg)

![](images/5e14566f45efb9fd4bf12cebf04a106d61ff0f0d7e9c5e2f7261ea742b7e66a9.jpg)

![](images/93e50d575d32932210de653943a923c510e664becf76c6ff3a968208703dce9f.jpg)

![](images/436e37fa9b844db1c88d9364bcd65dbaa3402b1172fe6be215bcba26970a7ab7.jpg)

MotionBERT

![](images/e4491a93da569732848482c4e54637c1e52d954678c220b5522caf9fdf14235f.jpg)

![](images/410489a2a26876287e3c503bed3d2914d0e32b46867c1df43e4cab8e3900072c.jpg)

![](images/2e04026dd08848d08ca6eb67752159e0e5e150f3bcd92645c781179484de422f.jpg)

![](images/4fa37ff4d7a960a839f4f2e4c20c68b03d49e760a1d647d5efae4e1ac5c31ed7.jpg)  
图10. 各模型在 Human3.6M 数据集某随机序列上的失败案例。

![](images/a5b8956f4315c16fafe0ed6c5ce86543103a29342d2ae079f3b13caa92fd911c.jpg)  
图11. MotionAGFormer 在野外视频上的评估结果。估计结果基于我们的 base（基础）变体。