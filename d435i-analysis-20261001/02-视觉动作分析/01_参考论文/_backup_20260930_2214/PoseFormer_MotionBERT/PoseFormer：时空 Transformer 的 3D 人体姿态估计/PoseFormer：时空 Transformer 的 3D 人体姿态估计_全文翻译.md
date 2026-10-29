# 基于时空Transformer的3D人体姿态估计

Ce Zheng<sup>1</sup>, Sijie Zhu<sup>1</sup>, Matias Mendieta<sup>1</sup>, Taojiannan Yang<sup>1</sup>, Chen Chen<sup>1</sup>, Zhengming Ding<sup>2</sup>

<sup>1</sup>Center for Research in Computer Vision, University of Central Florida, USA（中佛罗里达大学计算机视觉研究中心）

<sup>2</sup>Department of Computer Science, Tulane University, USA（杜兰大学计算机科学系）

{cezheng,sizhu,mendieta,taoyang1122}@knights.ucf.edu;

chen.chen@crcv.ucf.edu;zding1@tulane.edu

## 摘要

Transformer架构已成为自然语言处理领域的首选模型，如今正被引入到计算机视觉任务中，如图像分类、目标检测和语义分割。然而，在人体姿态估计领域，卷积架构仍然占据主导地位。在本工作中，我们提出了PoseFormer，一种纯粹基于Transformer的3D人体姿态估计方法，在视频中实现端到端的姿态估计，且不涉及任何卷积架构。受近期视觉Transformer发展的启发，我们构建了一种时空Transformer结构，能够全面建模每一帧内人体关节的关系以及跨帧的时间相关性，从而输出准确的中间帧3D人体姿态。我们在两个流行且标准的基准数据集Human3.6M和MPI-INF-3DHP上对我们的方法进行了定量和定性评估。大量实验表明，PoseFormer在这两个数据集上均达到了最先进的性能。代码已开源，地址为 https://github.com/zczcwh/PoseFormer

## 1. 引言

人体姿态估计（Human Pose Estimation，HPE）旨在从图像和视频等输入数据中定位关节点并构建人体表征（例如骨骼位置）。HPE 提供了人体的几何与运动信息，可广泛应用于多种场景（如人机交互、运动分析、医疗健康）。现有方法通常可分为两类：(1) 直接估计方法，(2) 2D 到 3D 提升方法。直接估计方法 [31, 29] 直接从 2D 图像或视频帧推断 3D 人体姿态，无需中间估计 2D 姿态表征。2D 到 3D 提升方法 [25, 5, 43, 38] 则基于已估计的 2D 姿态推断 3D 人体姿态。得益于当前最先进的 2D 姿态检测器的优异性能，2D 到 3D 提升方法通常优于直接估计方法。然而，将这些 2D 姿态映射到 3D 空间并非易事；由于深度模糊和遮挡，同一 2D 姿态可能对应多种潜在的 3D 姿态。为缓解这些问题并保持自然的连贯性，许多最新工作已将视频中的时间信息融入其方法中。例如，[25, 5] 利用时间卷积神经网络（CNN）捕捉相邻帧之间的全局依赖关系，而 [33] 则使用循环架构以达到类似效果。然而，这两种架构的时间相关窗口均有限。基于 CNN 的方法通常依赖扩张（dilation）技术，其时间连接性本质上受限；而循环网络则主要局限于简单的序列相关性。

近年来，Transformer [37] 凭借其高效性、可扩展性以及强大的建模能力，已成为自然语言处理（NLP）领域的主流模型。得益于 Transformer 的自注意力机制，可以清晰地建模长输入序列上的全局关联。这使得它特别适合处理序列数据问题，因此可自然地扩展到 3D HPE 任务。凭借其全面的连接性与表达能力，Transformer 为跨帧学习更强的时间表征提供了可能。然而，最新研究 [12, 36] 表明，Transformer 需要特定的设计才能在视觉任务上达到与 CNN 对应方法相当的性能。具体而言，它们通常需要大规模的训练数据 [12]，或者在应用于较小数据集时需要增强的数据增强与正则化策略 [36]。此外，现有的视觉 Transformer 主要局限于图像分类 [12, 36]、目标检测 [4, 50] 和分割 [41, 47] 等任务，而如何将 Transformer 的强大能力应用于 3D HPE 仍不清楚。

要开始回答这个问题，我们首先将 Transformer 直接应用于二维到三维的提升式人体姿态估计（HPE）。在这种情况下，我们将给定序列中每一帧的整个二维姿态视为一个 token（图 1(a)）。虽然这种基线方法在某种程度上是可行的，但它忽略了空间关系（关节与关节之间）的天然区别，从而留下了潜在的改进空间。该基线方法的一个自然扩展是将每个二维关节坐标视为一个 token，并提供由整个序列所有帧中的这些关节组成的输入（图 1(b)）。然而，在这种情况下，当使用较长的帧序列时，token 的数量会变得非常大（在三维 HPE 中，243 帧和每帧 17 个关节是常见的，token 的数量将达到 243×17=4131）。由于 Transformer 计算每个 token 与另一个 token 之间的直接注意力，模型的内存需求将达到不合理的水平。

![](images/0ab2a32ec86d750b2a1a7c585e0f752904a17aa5a7ac0f59ceec0330a8f26c13.jpg)  
图 1. 两种基线方法。

因此，作为应对这些挑战的有效解决方案，我们提出了 PoseFormer，这是首个用于视频中二维到三维提升式 HPE 的纯 Transformer 网络。PoseFormer 直接利用不同的 Transformer 模块对空间和时间两个维度进行建模。PoseFormer 不仅能在空间和时间元素上产生强大的表示，而且不会在长输入序列中引入巨大的 token 数量。在高层次上，PoseFormer 简单地接收来自现成的二维姿态估计器检测到的二维姿态序列，并输出中心帧的三维姿态。更具体地说，我们构建了一个空间 Transformer 模块来编码每一帧中二维关节之间的局部关系。空间自注意力层考虑二维关节的位置信息，并返回该帧的潜在特征表示。接着，我们的时间 Transformer 模块分析每个空间特征表示之间的全局依赖关系，并生成准确的三维姿态估计。

在两个流行的三维 HPE 基准数据集 Human3.6M [16] 和 MPI-INF-3DHP [27] 上的实验评估表明，PoseFormer 在两个数据集上都取得了最先进的性能。我们将估计的三维姿态与最先进的方法进行了可视化比较，发现 PoseFormer 产生了更平滑、更可靠的结果。此外，在消融研究中，我们还提供了 PoseFormer 注意力图的可视化和分析，以理解模型的内部工作机制并证明其有效性。我们的贡献主要有三点：

• 我们提出了首个基于纯 Transformer 的模型 PoseFormer，用于二维到三维提升类别的三维 HPE。

• 我们设计了一个有效的时空 Transformer 模型，其中空间 Transformer 模块编码人体关节之间的局部关系，时间 Transformer 模块捕获整个序列中跨帧的全局依赖关系。

• 没有花哨的技巧，我们的 PoseFormer 模型在 Human3.6M 和 MPI-INF-3DHP 数据集上均取得了最先进的结果。

## 2. 相关工作

在这里我们专门总结 3D 单人单视角人体姿态估计方法。直接估计方法直接从二维图像推断三维人体姿态，而不经过中间二维姿态表示的步骤。2D 到 3D 提升方法利用二维姿态作为输入来生成对应的三维姿态，这在当前该领域的最先进方法中更为流行。任何现成的二维姿态估计器都可以与这些方法有效兼容。我们提出的方法 PoseFormer 也遵循 2D 到 3D 提升的流程，因此本节我们将主要关注此类方法。

**2D 到 3D 提升的人体姿态估计。** 2D 到 3D 提升方法利用从输入图像或视频帧中估计出的二维姿态。OpenPose [3]、CPN [6]、AlphaPose [13] 和 HR-Net [35] 已被广泛用作二维姿态检测器。基于这种中间表示，三维姿态可以通过多种方法生成。Martinez 等人 [26] 提出了一种简单而有效的全连接残差网络，用于仅从单帧的二维关节位置回归三维关节位置。然而，相比于从单张图像估计三维人体姿态，视频能够提供时序信息以提高准确性和鲁棒性 [49, 10, 32, 8, 2, 44, 38]。Hossain 和 Little [33] 提出了一种使用长短期记忆（LSTM）单元的循环神经网络来利用输入序列中的时序信息。若干工作 [10, 2, 21] 利用了时空关系和约束（如骨骼长度和左右对称性）来提升性能。Pavllo 等人 [32] 引入了一个时序卷积网络，从连续的二维关键点序列估计三维姿态。基于 [32]，Chen 等人 [5] 增加了骨骼方向模块和骨骼长度模块以确保视频帧间的时序一致性，而 Liu 等人 [25] 则利用注意力机制来识别重要帧。然而，先前的最先进方法（例如 [25, 5]）依赖扩张时序卷积来捕捉全局依赖，这在时序连通性上存在固有的局限性。此外，这些工作中的多数 [25, 5, 33, 32] 通过简单的操作将关节坐标投影到隐空间，没有考虑人体关节的运动学关联。

GNN 在 3D 人体姿态估计中的应用。人体姿态天然可以用图来表示，其中关节点作为节点，骨骼作为边。图神经网络（GNN）也被应用于 2D 到 3D 姿态提升问题，并取得了优异的性能 [9, 45, 24]。Ci 等人 [9] 提出了一种名为局部连接网络（LCN）的框架，该框架利用全连接网络和 GNN 操作来编码局部关节点邻域之间的关系。Zhao 等人 [45] 解决了图卷积网络 [19]（GCN）操作的一个局限性，即权重矩阵在所有节点间共享的问题。他们引入了语义图卷积操作，以学习逐通道的边权重。

![](images/594280047e913891aef1948cca96b6db28c83b8c00ca3f9a305b3b112660b1d8.jpg)  
图 2. (a) 时间 Transformer 基线。(b) 时空 Transformer（PoseFormer）架构，由三个模块组成。一个空间 Transformer 模块，用于在考虑每个个体骨架的关节点相关性的同时提取特征。一个时间 Transformer 模块，用于学习整个序列的全局依赖关系。一个回归头模块，用于回归得到中心帧的最终 3D 姿态。Transformer 编码器的示意图遵循 ViT [12] 的设计。

对于我们的 PoseFormer，Transformer 可以被视为一种具有独特且通常具有优势的图操作的图神经网络。具体而言，Transformer 编码器模块本质上构成一个全连接图，其中边权重通过以输入为条件的多头自注意力机制计算得出。该操作还包括节点特征的归一化、对注意力头输出的前馈聚合以及残差连接，这些组件使其能够在堆叠多层时有效地扩展。相较于其他图操作，这种操作具有明显的优势。例如，节点之间连接的强度由 Transformer 的自注意力机制决定，而不是像该任务中典型的基于 GCN 的方法那样通过预定义的邻接矩阵来确定。这使得模型能够根据每个输入姿态灵活地调整关节点之间的相对重要性。此外，Transformer 中全面的缩放和归一化组件在缓解许多 GNN 操作变体在多层堆叠时所面临的过度平滑问题方面可能具有优势 [48]。

## 视觉 Transformer

近年来，将 Transformer 应用于视觉任务引起了越来越多的关注 [17, 14]。Carion 等人 [4] 提出了用于目标检测和全景分割的 DEtection TRansformer (DETR)。Dosovitskiy 等人 [12] 提出了一种纯 Transformer 架构——Vision Transformer (ViT)，在图像分类任务上取得了最先进的性能。然而，ViT 是在大规模数据集 ImageNet-21k 和 JFT-300M 上训练的，这需要巨大的计算资源。随后，提出了一种基于 ViT 并结合知识蒸馏的数据高效图像 Transformer (DeiT) [36]。对于 HPE (人体姿态估计) 等回归问题，Yang 等人 [40] 提出了一个 Transformer 网络 Transpose，仅用于从图像估计 2D 姿态。Lin 等人 [23] 在其方法 METRO (MEsh TRansfOrmer) 中结合了 CNN 和 Transformer 网络，从单张图像重建 3D 姿态和网格顶点。与我们的方法不同，METRO 属于直接估计的范畴。此外，METRO 忽略了时间一致性，这限制了其估计的鲁棒性。我们的时空 Transformer 架构利用了每一帧中的关键点相关性，并在视频中保持了自然的时间连贯性。

## 3. 方法

我们沿用与 [26, 32, 25, 5] 相同的视频 3D HPE 中的 2D 到 3D 提升流程。每帧的 2D 姿态由现成的 2D 姿态检测器获得，然后将连续帧的 2D 姿态序列作为输入，用于估计中间帧的 3D 姿态。与之前基于 CNN 的最先进模型相比，我们提出了一个极具竞争力的无卷积 Transformer 网络。

## 3.1. 时间 Transformer 基线

作为 Transformer 在 2D 到 3D 提升中的基线应用，我们将每个 2D 姿态视为一个输入 token，并采用 Transformer 来捕获输入之间的全局依赖关系，如图 2(a) 所示。我们将每个输入 token 称为 patch（块），与 ViT [12] 中的术语类似。对于输入序列 $X \in \mathbb { R } ^ { f \times ( J \cdot 2 ) }$ <sup>)</sup>，其中 f 是输入序列的帧数，J 是每个 2D 姿态的关节数，2 表示关节在 2D 空间中的坐标。$\{ \mathbf { x } ^ { i } \overset { \cdot } { \in } \mathbb { R } ^ { 1 \times ( J \cdot 2 ) } | i = 1 , 2 , \overset { \cdot } { \dots } f \}$ 表示每一帧的输入向量。块嵌入（patch embedding）是一个可训练的线性投影层，用于将每个块嵌入到高维特征中。Transformer 网络利用位置嵌入来保留序列的位置信息。该过程可以形式化为：

$$
Z _ { 0 } = [ \mathbf { x } ^ { 1 } E ; \mathbf { x } ^ { 2 } E ; \ldots ; \mathbf { x } ^ { f } E ] + E _ { p o s } .\tag{1}
$$

通过线性投影矩阵 $E \in \mathbb{R}^{(J \cdot 2) \times C}$ 进行嵌入，并与位置嵌入 $E_{pos} \in \mathbb{R}^{f \times C}$ 相加后，输入序列 $\dot{X} \in \mathbb{R}^{f \times (J \cdot 2)}$ 变为 $\dot{Z_0} \in \mathbb{R}^{f \times C}$，其中 $C$ 是嵌入维度。$Z_0$ 被送入时序 Transformer 编码器。

作为 Transformer 的核心功能，自注意力机制旨在关联输入序列中不同位置的嵌入特征。我们的 Transformer 编码器由多头自注意力块和多层感知机（MLP）块组成，结构与文献 [12] 中相同。每个块之前应用层归一化（LayerNorm），每个块之后应用残差连接（residual connections）[39, 1]。

缩放点积注意力（Scaled Dot-Product Attention）可以描述为一个映射函数，将查询矩阵 $Q$、键矩阵 $K$ 和值矩阵 $V$ 映射到输出注意力矩阵。$Q, K, V \in \mathbb{R}^{N \times d}$，其中 $N$ 是序列中的向量数量，$d$ 是该维度。在该注意力操作中使用缩放因子 $\frac{1}{\sqrt{d}}$ 进行适当的归一化，以防止当 $d$ 较大时点积的幅值过大导致梯度极小。因此，缩放点积注意力的输出可表示为：

$$
\operatorname{Attention}(Q, K, V) = \operatorname{Softmax}(Q K^{\top} / \sqrt{d}) V .\tag{2}
$$

在我们的时序 Transformer 中，$d = C$，$N = f$。$Q$、$K$ 和 $V$ 由嵌入特征 $Z \in \mathbb{R}^{f \times C}$ 通过线性变换 $W_Q$、$W_K$ 和 $W_V \in \mathbb{R}^{C \times C}$ 计算得到：

$$
Q = Z W_Q , \quad K = Z W_K , \quad V = Z W_V .\tag{3}
$$

多头自注意力层（Multi-head Self Attention Layer, MSA）利用多个头从不同位置的不同表示子空间中联合建模信息。每个头并行应用缩放点积注意力。MSA 的输出将是 $h$ 个注意力头输出的拼接。

$$
\mathrm{MSA}(Q, K, V) = \mathrm{Concat}(H_1, H_2, \dots, H_h) W_{out}\tag{4}
$$

$$
\mathrm{where} \quad H_i = \mathrm{Attention}(Q_i, K_i, V_i) , \quad i \in [1, \ldots, h]\tag{5}
$$

给定嵌入特征 $Z_0 \in \mathbb{R}^{f \times C}$，$L$ 层的时序 Transformer 编码器结构可表示如下：

$$
Z_{\ell}^{'} = \mathrm{MSA}(\mathrm{LN}(Z_{\ell-1})) + Z_{\ell-1}, \quad \ell = 1, 2 \dots L\tag{6}
$$

$$
Z_{\ell} = \mathrm{MLP}(\mathrm{LN}(Z_{\ell}^{'})) + Z_{\ell}^{'}, \quad \ell = 1, 2, \ldots L\tag{7}
$$

$$
Y = \mathrm{LN}(Z_L) ,\tag{8}
$$

其中 LN(·) 表示层归一化算子（与 ViT 中相同）。模型骨干由 L 个相同的层组成，编码器输出 $Y \in \mathbb { R } ^ { f \times C }$ 与编码器输入 $Z _ { 0 } \in \bar { \mathbb { R } ^ { f \times C } }$ 保持相同的尺寸

为了预测中间帧的 3D 姿态，编码器输出 $\bar { Y } \in \mathbb { R } ^ { f \times C }$ 通过在帧维度上取平均值被压缩为向量 $\mathbf { y } \in \mathbb { R } ^ { 1 \times C }$。最后，一个 MLP 模块将输出回归到 $\mathbf { y } \in \mathbb { R } ^ { 1 \times ( J \cdot 3 ) }$，即中间帧的 3D 姿态。

## 3.2. PoseFormer：时空 Transformer

我们观察到，时序 Transformer 基线主要关注输入序列中帧之间的全局依赖关系。Patch embedding，即一个线性变换，被用于将关节坐标投影到隐藏维度。然而，时序 Transformer 基线并未充分表征局部关节坐标之间的运动学信息，因为简单的线性投影层无法学习注意力信息。一个可行的替代方案是将每个关节坐标视为一个独立的 patch，并将所有帧的关节作为输入送入 Transformer（参见图 1(b)）。然而，patch 的数量会急剧增加（帧数 f 乘以关节数 J），导致模型计算复杂度为 $O ( ( f \cdot J ) ^ { 2 } )$。例如，如果我们使用 81 帧且每个 2D 姿态包含 17 个关节，则 patch 数量将达到 1377（而 ViT 模型使用 576 个 patch（输入尺寸 $\mathrm { s i z e = 3 8 4 \times 3 8 4 }$，patch $\mathrm { s i z e } = 1 6 \times 1 6$））。

为了高效地学习局部关节相关性，我们分别采用两个独立的 Transformer 分别处理空间信息和时间信息。如图 2(b) 所示，PoseFormer 由三个模块组成：空间 Transformer 模块、时序 Transformer 模块和回归头模块。

# 空间变换器模块

空间变换器模块旨在从单帧中提取高维特征嵌入。给定包含 J 个关节点的二维姿态，我们将每个关节点（即两个坐标）视为一个块（patch），并遵循通用的视觉变换器（Vision Transformer）流程在所有块之间执行特征提取。首先，我们通过可训练的线性投影将每个关节点的坐标映射到高维空间，这被称为空间块嵌入（spatial patch embedding）。我们将其与可学习的空间位置嵌入 [12] $E _ { S P o s } ~ \in ~ \mathbb { R } ^ { J \times c }$ 相加，因此第 i 帧的输入 $\mathbf { x } _ { i } \in \mathbb { R } ^ { 1 \times \left( J \cdot 2 \right) }$ 变为 $\boldsymbol { z } _ { 0 } ^ { i } \in \mathbb { R } ^ { J \times c }$，其中 2 表示每帧中的二维坐标，c 为空间嵌入维度。得到的关节点特征序列 $z _ { 0 } ^ { i }$ 被送入空间变换器编码器（spatial transformer encoder），该编码器应用自注意力（self-attention）机制来整合所有关节点之间的信息。对于第 i 帧，经过 L 层空间变换器编码器后的输出为 $\boldsymbol { z } _ { L } ^ { i } \in \mathbb { R } ^ { J \times c }$。

# 时间变换器模块

由于空间变换器模块为每个单独的帧编码了高维特征，时间变换器模块的目标是对跨帧序列的依赖关系进行建模。对于第 i 帧，空间变换器的输出 $z _ { L } ^ { i } \in \mathbb { R } ^ { J \times c }$ 被展平为向量 $\mathbf { z } ^ { i } \in \mathbb { R } ^ { 1 \times ( J \cdot c ) }$。然后，我们将来自 f 个输入帧的这些向量 $\{ \mathbf { z } ^ { 1 } , \mathbf { z } ^ { 2 } , \hdots , \mathbf { z } ^ { f } \}$ 拼接为 $Z _ { 0 } ~ \in ~ \mathbb { R } ^ { f \times ( J \cdot c ) }$。在进入时间变换器模块之前，我们添加可学习的时间位置嵌入 [12] $E _ { T P o s } \in \mathbb { R } ^ { f \times ( J \cdot c ) }$ 以保留帧位置信息。对于时间变换器编码器，我们采用与空间变换器编码器相同的架构，由多头自注意力（multihead self-attention）模块和多层感知机（MLP）模块组成。时间变换器模块的输出为 $Y \in \mathbb { R } ^ { f \times ( J \cdot c ) }$。

# 回归头

由于我们使用一序列帧来预测中心帧的三维姿态，时间变换器模块的输出 $Y \in \mathbb { R } ^ { f \times ( J \cdot c ) }$ 需要被归约为 $\mathbf { y } \in \mathbb { R } ^ { 1 \times ( J \cdot c ) }$。我们在帧维度上应用加权平均操作（使用可学习的权重）来实现这一归约。最后，一个简单的 MLP 模块（包含层归一化（Layer norm）和一个线性层）返回输出 $\mathbf { y } \in \mathbb { R } ^ { 1 \times ( J \cdot 3 ) }$，即中心帧的预测三维姿态。

**损失函数。** 为了训练我们的时空 Transformer 模型，我们采用标准的 MPJPE（Mean Per Joint Position Error，平均每关节位置误差）损失来最小化预测姿态与真实姿态之间的误差：

$$
\mathcal { L } = \frac { 1 } { J } \Sigma _ { k = 1 } ^ { J } \Vert p _ { k } - \hat { p } _ { k } \Vert _ { 2 } ,\tag{9}
$$

其中 $p _ { k }$ 和 $\hat { p } _ { k }$ 分别是第 k 个关节的真实 3D 位置和估计 3D 位置。

## 4. 实验

### 4.1. 数据集与评价指标

我们在两个常用的 3D HPE（Human Pose Estimation，人体姿态估计）数据集上评估我们的模型：Human3.6M [16] 和 MPI-INF-3DHP [27]。

**Human3.6M [16]** 是 3D 单人 HPE 任务中使用最广泛的室内数据集。其中有 11 位专业演员执行 17 种动作，例如坐着、行走和打电话等。每个受试者的视频在室内环境中从 4 个不同视角进行录制。该数据集包含 360 万个视频帧，并带有通过精确的基于标记的运动捕捉系统采集的 3D 真实标注。遵循先前工作 [32, 25, 5]，我们采用相同的实验设置：使用全部 15 个动作进行训练和测试，模型在五个受试者子集（S1、S5、S6、S7、S8）上进行训练，并在两个受试者（S9 和 S11）上进行测试。

**MPI-INF-3DHP [27]** 是一个更具挑战性的 3D 姿态数据集。它既包含受限的室内场景，也包含复杂的室外场景。有 8 位演员从 14 个相机视角执行 8 种动作，涵盖了更多样化的姿态。MPI-INF-3DHP 提供了一个包含 6 个受试者在不同场景下的测试集。我们遵循 [22, 5, 38] 中的设置。

对于 Human3.6M 数据集，我们使用最常用的评估指标（MPJPE 和 P-MPJPE）[46] 来根据真实 3D 姿态评估我们估计结果的性能。MPJPE（Mean Per Joint Position Error，平均每关节位置误差）以毫米为单位，计算为估计关节与真实关节之间的平均欧氏距离；我们将 MPJPE 称为 Protocol 1。P-MPJPE 是估计的 3D 姿态与真实姿态之间经过后处理刚性对齐后的 MPJPE，对单个关节预测失败更具鲁棒性。我们将 P-MPJPE 称为 Protocol 2。

对于 MPI-INF-3DHP 数据集，我们使用 MPJPE、150mm 范围内的关键点正确率（Percentage of Correct Keypoint，PCK）[22, 5, 38] 以及曲线下面积（Area Under Curve，AUC）。

### 4.2. 实现细节

我使用 PyTorch [30] 实现了所提出的方法。训练和测试均使用了两块 NVIDIA RTX 3090 GPU。在进行实验时，我们选择了三种不同的帧序列长度，即 $f = 9, f = 27, f = 81$。关于帧数及对应结果的详细信息将在消融实验（第 4.4 节）中讨论。我们按照 [32, 25, 5] 的方法，在训练和测试中均采用姿态水平翻转作为数据增强。我们使用 Adam [18] 优化器训练模型 130 个 epoch，权重衰减为 0.1。我们采用指数学习率衰减策略，初始学习率为 2e-4，每个 epoch 的衰减因子为 0.98。我们将批量大小设置为 1024，并对 transformer 编码器层使用速率为 0.1 的随机深度（Stochastic Depth）[15]。对于 2D 姿态检测器，在 Human3.6M 数据集上我们按照 [32, 25, 5] 使用级联金字塔网络（CPN）[7]，而在 MPI-INF-3DHP 数据集上按照 [28, 22] 使用真值 2D 姿态作为输入。

## 4.3. 与当前最优方法的比较

Human3.6M。我们在表 1 中报告了测试集（S9 和 S11）的全部 15 个动作的结果。最后一列给出了整个测试集上的平均性能。按照 2D 到 3D 提升方法，我们使用 CPN 网络作为 2D 姿态检测器，然后将检测到的 2D 姿态用作训练和测试的输入。PoseFormer 在协议 1 和协议 2 下分别比我们的基线（即第 3.1 节中的时序 transformer 基线）有大幅提升（6.1% 和 6.4%）。这清楚地展示了使用空间 transformer 来对每帧中关节之间的相关性进行建模的优势。如表 1（上）所示，PoseFormer 在协议 1 下取得了最低的平均 MPJPE，为 44.3mm。与基于 transformer 的方法 METRO [23]（该方法由于从单张图像估计 3D 姿态而忽略了时序一致性）相比，PoseFormer 将 MPJPE 降低了约 18%。对于协议 2，我们同样取得了最好的总体结果，如表 1（下）所示。此外，PoseFormer 在 Photo、SittingDown、WalkDog 和 Smoke 等困难动作上实现了更准确的姿态预测。与其他简单动作不同，这些动作中的姿态变化更快，且一些远距离帧之间具有很强的相关性。在这种情况下，全局依赖关系起着重要作用，而 transformer 的注意力机制尤为有利。

Table 1. 在 Human3.6M 数据集上，使用检测得到的 2D 姿态作为输入时，估计 3D 姿态与真实 3D 姿态之间平均每关节位置误差（Mean Per Joint Position Error, MPJPE）的定量对比（协议 1 & 2）。上表：协议 1（MPJPE）下的结果。下表：协议 2（P-MPJPE）下的结果。f 表示各方法使用的输入帧数，∗ 表示输入的 2D 姿态由级联金字塔网络（CPN）检测得到，† 表示基于 Transformer 的模型。（红色：最佳；蓝色：次佳）

<table><tr><td>Protocol i</td><td></td><td>Dir.</td><td>Disc.</td><td>Eat.</td><td>Greet</td><td>Phone</td><td>Photo</td><td>Pose</td><td>Purch.</td><td>Sit</td><td>SitD.</td><td>Somke</td><td>Wait</td><td>WalkD.</td><td>Walk</td><td>WalkT.</td><td>Average</td></tr><tr><td>Dabral et al. [11]</td><td>ECCV&#x27;18 ICCV&#x27;19</td><td>44.8</td><td>50.4</td><td>44.7</td><td>49.0</td><td>52.9</td><td>61.4</td><td>43.5</td><td>45.5</td><td>63.1</td><td>87.3</td><td>51.7</td><td>48.5</td><td>52.2 51.3</td><td>37.6 37.1</td><td>41.9 39.4</td><td>52.1</td></tr><tr><td>Cai et al.  $[ 2 ] \overset { \cdot } { ( } f \overset { \cdot } { = } 7 )$ </td><td></td><td>44.6</td><td>47.4</td><td>45.6</td><td>48.8</td><td>50.8</td><td>59.0</td><td>47.2</td><td>43.9</td><td>57.9</td><td>61.9</td><td>49.7</td><td>46.6</td><td></td><td></td><td></td><td>48.8</td></tr><tr><td>Pavllo et  $a l . \left[ 3 2 \right] ( f = 2 4 3 ) ^ { * }$ </td><td>CVPR&#x27;19</td><td>45.2</td><td>46.7</td><td>43.3</td><td>45.6</td><td>48.1</td><td>55.1</td><td>44.6</td><td>44.3</td><td>57.3</td><td>65.8</td><td>47.1</td><td>44.0</td><td>49.0</td><td>32.8</td><td>33.9</td><td>46.8</td></tr><tr><td>Lin et al.  $[ 2 2 ] ( f = 5 0 )$ </td><td>BMVC&#x27;19</td><td>42.5</td><td>44.8</td><td>42.6</td><td>44.2</td><td>48.5</td><td>57.1</td><td>52.6</td><td>41.4</td><td>56.5</td><td>64.5</td><td>47.4</td><td>43.0</td><td>48.1</td><td>33.0</td><td>35.1</td><td>46.6</td></tr><tr><td>Yeh et al. [42]</td><td>NIPS&#x27;19</td><td>44.8</td><td>46.1</td><td>43.3</td><td>46.4</td><td>49.0</td><td>55.2</td><td>44.6</td><td>44.0</td><td>58.3</td><td>62.7</td><td>47.1</td><td>43.9</td><td>48.6</td><td>32.7</td><td>33.3</td><td>46.7</td></tr><tr><td>Liu et al. [25] (f = 243)*</td><td>CVPR&#x27;20</td><td>41.8</td><td>44.8</td><td>41.1</td><td>44.9</td><td>47.4</td><td>54.1</td><td>43.4</td><td>42.2</td><td>56.2</td><td>63.6</td><td>45.3</td><td>43.5</td><td>45.3</td><td>31.3</td><td>32.2</td><td>45.1</td></tr><tr>
tr><td>SRNet [43] *</td><td>ECCV&#x27;20</td><td>46.6</td><td>47.1</td><td>43.9</td><td>41.6</td><td>45.8</td><td>49.6</td><td>46.5</td><td>40.0</td><td>53.4</td><td>61.1</td><td>46.1</td><td>42.6</td><td>43.1</td><td>31.5</td><td>32.6</td><td>44.8</td></tr>
<tr><td>UGCN [38] (f = 96)</td><td>ECCV&#x27;20</td><td>41.3</td><td>43.9</td><td>44.0</td><td>42.2</td><td>48.0</td><td>57.1</td><td>42.2</td><td>43.2</td><td>57.3</td><td>61.3</td><td>47.0</td><td>43.5</td><td>47.0</td><td>32.6</td><td>31.8</td><td>45.6</td></tr>
<tr><td>Chen et al. [5] (f = 81)* METRO [23] (f = 1) †</td><td>TCSVT21</td><td>42.1</td><td>43.8</td><td>41.0</td><td>43.8</td><td>46.1</td><td>53.5</td><td>42.4</td><td>43.1</td><td>53.9</td><td>60.5</td><td>45.7</td><td>42.1</td><td>46.2</td><td>32.2</td><td>33.8</td><td>44.6</td></tr>
<tr><td>Baseline (f = 81)*†</td><td>CVPR&#x27;21</td><td></td><td>47.9</td><td></td><td></td><td></td><td>-</td><td>-</td><td>-</td><td>-</td><td></td><td>-</td><td></td><td></td><td></td><td></td><td>54.0</td></tr>
<tr><td>PoseFormer (f = 81)*†</td><td></td><td>43.8 41.5</td><td>44.8</td><td>43.8 39.8</td><td>45.5 42.5</td><td>49.7 46.5</td><td>55.7 51.6</td><td>44.3 42.1</td><td>45.8</td><td>57.7 53.3</td><td>66.3 60.7</td><td>47.4 45.5</td><td>45.4</td><td>48.6 46.1</td><td>32.5</td><td>33.8 32.2</td><td>47.2 44.3</td></tr>
<tr><td>Protocol 2</td><td></td><td>Dir.</td><td>Disc.</td><td>Eat.</td><td></td><td></td><td></td><td></td><td>42.0</td><td>Sit</td><td>SitD.</td><td></td><td>43.3</td><td>WalkD.</td><td>31.8</td><td></td><td></td></tr>
<tr><td>Pavlakos et al. [31]</td><td>CVPR&#x27;18</td><td>34.7</td><td>39.8</td><td>41.8</td><td>Greet</td><td>Phone 42.5</td><td>Photo 47.5</td><td>Pose</td><td>Purch. 36.6</td><td>50.7</td><td>56.8</td><td>Somke 42.6</td><td>Wait</td><td>43.9</td><td>Walk</td><td>WalkT. 36.5</td><td>Average 41.8</td></tr>
<tr><td></td><td>ECCV&#x27;18</td><td>35.7</td><td>39.3</td><td>44.6</td><td>38.6</td><td></td><td></td><td>38.0</td><td></td><td></td><td>61.3</td><td>46.5</td><td>39.6</td><td></td><td>32.1</td><td>39.4</td><td>44.1</td></tr>
<tr><td>Hossain et al. [33]</td><td></td><td></td><td></td><td></td><td>43.0</td><td>47.2</td><td>54.0</td><td>38.3</td><td>37.5</td><td>51.6</td><td></td><td></td><td>41.4</td><td>47.3</td><td>34.2</td><td></td><td></td></tr>
<tr><td>Cai et al.  $\left[ 2 \right] \left( f = 7 \right)$ </td><td>ICCV&#x27;19</td><td>35.7</td><td>37.8 35.3</td><td>36.9</td><td>40.7</td><td>39.6</td><td>45.2</td><td>37.4</td><td>34.5</td><td>46.9</td><td>50.1</td><td>40.5</td><td>36.1</td><td>41.0</td><td>29.6</td><td>32.3</td><td>39.0 36.8</td></tr>
<tr><td>Lin et al.  $\left[ 2 2 \right] \left( f = 5 0 \right)$ </td><td>BMVC&#x27;19 CVPR&#x27;19</td><td>32.5</td><td></td><td>34.3</td><td>36.2</td><td>37.8</td><td>43.0</td><td>33.0</td><td>32.2</td><td>45.7</td><td>51.8</td><td>38.4</td><td>32.8</td><td>37.5</td><td>25.8</td><td>28.9</td><td>36.5</td></tr>
<tr><td>Pavllo et al.  $[ 3 2 ] ( f = 2 4 3 ) ^ { \ast }$ </td><td>CVPR&#x27;20</td><td>34.1</td><td>36.1 35.2</td><td>34.4</td><td>37.2</td><td>36.4</td><td>42.2</td><td>34.4</td><td>33.6</td><td>45.0</td><td>52.5</td><td>37.4</td><td>33.8</td><td>37.8</td><td>25.6</td><td>27.3 27.2</td><td>35.6</td></tr>
<tr><td>Liu et al. [2  $5 ] ( f = 2 4 3 ) ^ { \ast }$ </td><td></td><td>32.3</td><td>35.2</td><td>33.3</td><td>35.8</td><td>35.9</td><td>41.5</td><td>33.2</td><td>32.7</td><td>44.6</td><td>50.9</td><td>37.0</td><td>32.4</td><td>37.0</td><td>25.2</td><td></td><td>35.5</td></tr>
<tr><td>UGCN [38  $\displaystyle { \mathrm { ~ l ~ } ( f = 9 6 ) }$ </td><td>ECCV&#x27;20 TCSVT&#x27;21</td><td>32.9</td><td>35.3</td><td>35.6</td><td>34.4</td><td>36.4</td><td>42.7</td><td>31.2</td><td>32.5</td><td>45.6</td><td>50.2</td><td>37.3</td><td>32.8</td><td>36.3</td><td>26.0</td><td>23.9</td><td></td></tr>
<tr><td>Chen et al.  $[ 5 ] ( f = \mathring 8 1 ) ^ { * }$ </td><td></td><td>33.1</td><td></td><td>33.4</td><td>35.9</td><td>36.1</td><td>41.7</td><td>32.8</td><td>33.3</td><td>42.6</td><td>49.4</td><td>37.0</td><td>32.7</td><td>36.5</td><td>25.5</td><td>27.9</td><td>35.6</td></tr>
<tr><td>Baseline  $\overline { { f = 8 1 ) ^ { * } \dag } }$ </td><td></td><td>33.6</td><td>37.1</td><td>35.4</td><td>36.7</td><td>37.8</td><td>42.2</td><td>33.9</td><td>34.7</td><td>47.0</td><td>53.4</td><td>38.2</td><td>34.3</td><td>37.6</td><td>25.3</td><td>27.8</td><td>37.0</td></tr>
<tr><td>PoseFormer  $( f = 8 1 ) ^ { * } \dag$ </td><td></td><td>32.5</td><td>34.8</td><td>32.6</td><td>34.6</td><td>35.3</td><td>39.5</td><td>32.1</td><td>32.0</td><td>42.8</td><td>48.5</td><td>34.8</td><td>32.4</td><td>35.3</td><td>24.5</td><td>26.0</td><td>34.6</td></tr>
</table>

为了进一步研究我们方法的下界，我们直接使用真值二维姿态作为输入，以减轻噪声二维姿态数据引起的误差。结果如表 2 所示。通过使用干净的二维姿态数据，MPJPE 从 44.3mm 降低到 31.3mm，降幅约为 29.7%。PoseFormer 在 9 个动作中取得了最佳成绩，在 6 个动作中取得了次佳成绩。平均得分比 SRNet [43] 提高了约 2%。

在图 3 中，我们还比较了 Human3.6M 测试集 S11 中 Photo 动作下某些误差最大的单个关节的 MPJPE。在这些困难关节上，PoseFormer 的性能优于 [32, 5]。

![](images/89d3bd6c415b46fc37329455a72b7cde2376b38a14406ae71040bfa33e568e74.jpg)  
图 3. 在 Human3.6M 测试集 S11 中 Photo 动作的所有帧上各关节的平均误差对比。

MPI-INF-3DHP。表 3 报告了 PoseFormer 与其他方法在 MPI-INF-3DHP 上的定量结果。与 Human3.6M 相比，该数据集的训练样本较少，且部分样本采集自室外场景。由于该数据集的序列长度通常较短，我们使用 9 帧的二维姿态作为模型输入。我们的方法在所有三个评估指标（PCK、AUC 和 MPJPE）上均再次取得了最佳性能。

定性结果。我们还提供了三维估计姿态与真值之间的可视化对比。我们在 Human3.6M 测试集 S11 的 Photo 动作上评估 PoseFormer，这是最具挑战性的动作之一（所有方法都具有较高的 MPJPE）。与最先进的方法 [5] 相比，我们的 PoseFormer 实现了更准确的预测，如图 4 所示。

## 4.4. 消融实验

为了验证 PoseFormer 各组成部分的贡献以及超参数对性能的影响，我们在协议 1 下的 Human3.6M 数据集上进行了广泛的消融实验。

## PoseFormer 的设计

我们在表 4 中探究了空间 Transformer 的影响，以及空间和时间 Transformer 的位置嵌入的影响。我们输入 9 帧由 CPN 检测到的 2D 姿态 $( J ~ = ~ 1 7 )$ 来预测 3D 姿态。为了公平地比较每个模块的影响，我们固定了所有架构参数；空间 Transformer 的嵌入维度为 $1 7 \times 3 2 = 5 4 4$，空间 Transformer 编码器层数为 4。对于时间 Transformer，嵌入维度与空间 Transformer 保持一致（即 544），我们同样使用 4 层时间 Transformer。为了验证我们时空设计的影响，我们首先与第 3.1 节中采用的 Transformer 基线进行比较。表 4 中的结果表明，我们的时空 Transformer 带来了显著的影响（MPJPE 从 52.5 降到 49.9），因为关节间的相关性得到了更强的建模。这与表 1 中 $f = 8 1$ 时的结果（Baseline vs. PoseFormer）一致。接下来，我们评估位置嵌入的影响。我们探索了四种可能的组合：不使用位置嵌入、仅使用空间位置嵌入、仅使用时间位置嵌入、以及同时使用空间和时间位置嵌入。通过比较这些组合的结果，可以明显看出位置嵌入能提升性能。同时在空间和时间模块上应用位置嵌入，可获得最佳的整体结果。

表 2. 在 Human3.6M 数据集上使用真实 2D 姿态作为输入时，估计的 3D 姿态与真实 3D 姿态之间平均关节位置误差（Mean Per Joint Position Error，MPJPE）的定量比较，依据协议 1（MPJPE）。（红色：最佳；蓝色：次佳）

<table><tr><td>真实 协议 1</td><td colspan="10">指</td><td>吸</td><td></td><td>等</td><td>遛狗</td><td>行走</td><td>同行</td><td>平均</td></tr><tr><td>Hossain 等人 [33]</td><td>ECCV'18</td><td>35.2</td><td>讨 40.8</td><td>食 37.2</td><td>招 37.4</td><td>电话 43.2</td><td>拍照 44.0</td><td>摆姿 38.9</td><td>购 35.6</td><td>坐 42.3</td><td>坐下 44.6</td><td>39.7</td><td>39.7</td><td>40.2</td><td>32.8</td><td>35.5</td><td>39.2</td></tr><tr><td>Pavllo 等人 [32] (f = 243)</td><td>CVPR'19</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>37.2</td></tr><tr><td>Liu 等人 [25] (f = 243)</td><td>CVPR'20</td><td>34.5</td><td>37.1</td><td>33.6</td><td>34.2</td><td>32.9</td><td>37.1</td><td>39.6</td><td>35.8</td><td>40.7</td><td>41.4</td><td>33.0</td><td>33.8</td><td>33.0</td><td>26.6</td><td>26.9</td><td>34.7</td></tr><tr><td>SRNet [43]</td><td>ECCV'20</td><td>34.8</td><td>32.1</td><td>28.5</td><td>30.7</td><td>31.4</td><td>36.9</td><td>35.6</td><td>30.5</td><td>38.9</td><td>40.5</td><td>32.5</td><td>31.0</td><td>29.9</td><td>22.5</td><td>24.5</td><td>32.0</td></tr><tr><td>Chen 等人 [5] (f = 243)</td><td>TCSVT'21</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>32.3</td></tr><tr><td>PoseFormer (f = 81)</td><td></td><td>30.0</td><td>33.6</td><td>29.9</td><td>31.0</td><td>30.2</td><td>33.3</td><td>34.8</td><td>31.4</td><td>37.8</td><td>38.6</td><td>31.7</td><td>31.5</td><td>29.0</td><td>23.3</td><td>23.1</td><td>31.3</td></tr></table>

![](images/54abe0cfc5293f64391648c881f26527b2d423477c34da91711017a26b044191.jpg)  
图 4. 在 Human3.6M 测试集 S11 的拍照动作上，我们的方法（PoseFormer）与当前最优方法 Chen 等人 [5] 的定性比较。绿色箭头突出显示了 PoseFormer 明显具有更优结果的位置。

Table 3. 在 MPI-INF-3DHP 上与以往方法的定量比较。最佳分数以粗体标注。

<table><tr><td></td><td></td><td>PCK↑</td><td>AUC↑</td><td>MPJPE↓</td></tr>
<tr><td>Mehta et al. [27]</td><td>3DV&#x27;17</td><td>75.7</td><td>39.3</td><td>117.6</td></tr>
<tr><td>Mehta et al. [28] Pavllo et al. [32] (81 frames)</td><td>ACM ToG&#x27;17 CVPR&#x27;19</td><td>76.6 86.0</td><td>40.4</td><td>124.7</td></tr>
<tr><td>Pavllo et al. [32] (243 frames)</td><td>CVPR&#x27;19</td><td>85.5</td><td>51.9 51.5</td><td>84.0 84.8</td></tr>
<tr><td>Lin et al. [22] (25 frames)</td><td>BMVC&#x27;19</td><td>83.6</td><td>51.4</td><td>79.8</td></tr>
<tr><td>Li et al. [20]</td><td>CVPR&#x27;20</td><td>81.2</td><td>46.1</td><td>99.7</td></tr>
<tr><td>Chen et al. [5] (81 frames)</td><td>TCSVT’21</td><td>87.9</td><td>54.0</td><td>78.8</td></tr>
<tr><td>PoseFormer (9 frames)</td><td></td><td><b>88.6</b></td><td><b>56.4</b></td><td><b>77.1</b></td></tr>
</table>

Table 4. PoseFormer 中不同组件的消融实验。评估在 Human3.6M（协议 1）上进行，使用检测得到的 2D 姿态作为输入。（T：仅时间；S-T：时空）

<table><tr><td>输入长度 (f)</td><td>T</td><td>S-T</td><td>空间位置编码</td><td>时间位置编码</td><td>MPJPE</td></tr>
<tr><td>9</td><td>√</td><td>x</td><td>x</td><td>√</td><td>52.5</td></tr>
<tr><td>9</td><td>x</td><td>√</td><td>x</td><td>x</td><td>51.6</td></tr>
<tr><td>9</td><td>×</td><td>√</td><td>√</td><td>x</td><td>50.7</td></tr>
<tr><td>9</td><td>x</td><td>v√</td><td>x</td><td>√</td><td>50.5</td></tr>
<tr><td>9</td><td>x</td><td></td><td>√</td><td>√</td><td>49.9</td></tr>
</table>

Table 5. PoseFormer 中不同架构参数的消融实验。评估在 Human3.6M（协议 1）上进行，使用检测得到的 2D 姿态作为输入。c 为空间 Transformer 的 patch 嵌入维度。$L _ { S }$ 和 $L _ { T }$ 分别表示空间 Transformer 和时间 Transformer 的层数。

<table><tr><td>C</td><td>16</td><td>16</td><td>16</td><td>32</td><td>32</td><td>32</td><td>48</td><td>48</td><td>48</td></tr>
<tr><td> $\overline { { L _ { S } } }$ </td><td>2</td><td>4</td><td>6</td><td>2</td><td>4</td><td>6</td><td>2</td><td>4</td><td>6</td></tr>
<tr><td> $L _ { T }$ </td><td>2</td><td>4</td><td>6</td><td>2</td><td>4</td><td>6</td><td>2</td><td>4</td><td>6</td></tr>
<tr><td>MPJPE</td><td>52.8</td><td>51.7</td><td>50.4</td><td>52.4</td><td><b>49.9</b></td><td>50.3</td><td>51.7</td><td>50.4</td><td>50.5</td></tr>
</table>

架构参数分析。我们在表 5 中探索了各种参数组合以寻找最优的网络架构。c 表示空间变换器中的嵌入特征维度，L 表示变换器编码器中使用的层数。在 Pose-Former 中，空间变换器的输出被展平并与时间位置嵌入相加，形成时间变换器编码器的输入。因此，时间变换器编码器中的嵌入特征维度为 c × J。我们模型的最优参数为 $c = 3 2 \AA$ $L _ { S } = 4 ,$ 和 $L _ { T } = 4$

表 6. 计算复杂度、MPJPE 和推理速度（每秒帧数 (FPS)）的比较。评估在 Human3.6M 上使用 Protocol 1，以检测到的 2D 姿态作为输入进行。FPS 基于单个 GeForce GTX 2080 Ti GPU。

<table><tr><td rowspan=1 colspan=4></td><td></td><td rowspan=1 colspan=1>Parameters (M)</td><td rowspan=1 colspan=1>FLOPs (M)</td><td rowspan=1 colspan=1>MPJPE</td><td rowspan=1 colspan=6>FPS</td></tr><tr><td rowspan=6 colspan=4>Hossain and Little [33]Pavllo et al. [32]Pavllo et al. [32]Pavllo et al. [32]Chen et al. [5]Chen et al. [5]Chen et al. [5]</td><td></td><td rowspan=1 colspan=1>16.95</td><td rowspan=1 colspan=1>33.88</td><td rowspan=1 colspan=1>58.3</td><td rowspan=1 colspan=6>=</td></tr><tr><td rowspan=1 colspan=2></td><td></td><td rowspan=1 colspan=1>8.56</td><td rowspan=1 colspan=1>17.09</td><td rowspan=1 colspan=1>48.8</td><td rowspan=1 colspan=6>1492</td></tr><tr><td rowspan=1 colspan=2>et al. [32]</td><td></td><td rowspan=1 colspan=1>12.79</td><td rowspan=1 colspan=1>25.48</td><td rowspan=1 colspan=1>47.7</td><td rowspan=1 colspan=6>1121</td></tr><tr><td></td><td rowspan=1 colspan=1>16.95</td><td rowspan=1 colspan=1>33.87</td><td rowspan=1 colspan=1>46.8</td><td rowspan=1 colspan=6>863</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>31.88</td><td rowspan=1 colspan=1>61.7</td><td rowspan=1 colspan=1>45.3</td><td rowspan=1 colspan=6>410</td><td rowspan=1 colspan=1>81</td><td rowspan=1 colspan=1>45.53</td><td rowspan=1 colspan=1>88.9</td><td rowspan=1 colspan=1>44.6</td></tr><tr><td></td><td rowspan=1 colspan=1>59.18</td><td rowspan=1 colspan=1>116</td><td rowspan=1 colspan=1>44.1</td><td rowspan=1 colspan=6>264</td></tr><tr><td rowspan=2 colspan=4>PoseFormerPoseFormerPoseFormer</td><td></td><td rowspan=1 colspan=1>9.58</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>49.9</td><td rowspan=1 colspan=6>320</td></tr><tr><td></td><td rowspan=1 colspan=1>9.599.60</td><td rowspan=1 colspan=1>4521358</td><td rowspan=1 colspan=1>47.044.3</td><td rowspan=1 colspan=6>297269</td></tr></table>

计算复杂度分析。我们在表 6 中报告了模型性能、总参数量以及每帧估计的浮点运算次数（FLOPs），并给出了不同输入序列长度（f）下的输出帧率（FPS）。当序列长度增加时，我们的模型实现了更好的精度，且总参数量没有显著增加。这是因为帧数仅影响时间位置嵌入层，而该层不需要大量参数。与其他模型相比，我们的模型以更少的总参数量获得了具有竞争力的性能。我们参照 [5] 中的相同设置，在单块 GeForce RTX 2080 Ti GPU 上报告了不同模型的推理 FPS。虽然我们的模型推理速度并非绝对最快，但该速度对于实时推理仍然是可以接受的。对于完整的 3D HPE 处理流程，首先由 2D 姿态检测器检测 2D 姿态，然后由我们的方法估计 3D 姿态。常用 2D 姿态检测器的 FPS 通常低于 80，这意味着我们模型的推理速度不会成为瓶颈。

![](images/6ea09982e55e9fc3405fd015acd423d63dde77eacb84ecd192051606795a8f9b.jpg)  
图 5. 空间 Transformer 中自注意力的可视化。x 轴（水平方向）和 y 轴（垂直方向）分别对应查询和预测输出。像素 $w_{i,j}$（i：行，j：列）表示第 j 个查询对第 i 个输出的注意力权重。红色表示更强的注意力。注意力输出被归一化到 0 到 1 的范围。

![](images/7c0563b552957a340d898ed82231d0651a3779f86f5c68f2f992093c92cc10bf.jpg)  
图 6. 时间 Transformer 中自注意力的可视化。x 轴（水平方向）和 y 轴（垂直方向）分别对应查询和预测输出。像素 $w_{i,j}$（i：行，j：列）表示第 j 个查询对第 i 个输出的注意力权重。红色表示更强的注意力。注意力输出被归一化到 0 到 1 的范围。

注意力可视化。为了通过多头自注意力模块展示注意力机制，我们在 Human3.6M 测试集 S11 上针对特定动作（SittingDown，坐下）评估我们的模型，并分别可视化空间和时间 Transformer 的自注意力图，如 Fig. 5 和 Fig. 6 所示。对于空间自注意力图，横轴对应于 17 个关节的查询，纵轴表示注意力输出。如 Fig. 5 所示，不同注意力头返回不同的注意力强度，这反映了输入关节之间学习到的各种局部关系。我们发现 Head 3 聚焦于关节 15 和 16，即右肘和右腕。Head 5 建立了左腿到左臂的连接（关节 4、5、6 与关节 11、12、13）。这些关节可以归为身体的左侧部分，而 Head 7 则集中关注右侧（关节 1、2、3 与关节 12、13、14）。

对于 Fig. 6 中的时间自注意力图，横轴对应于 81 帧的查询，纵轴表示注意力输出。不同注意力头学习了长程全局依赖关系。Head 3 的注意力与中心帧右侧的某些帧（例如第 58、62 和 69 帧）高度相关。Head 7 捕捉了第 1、20、22、42、78 帧之间的依赖关系，尽管它们距离较远。空间和时间注意力图表明，PoseFormer 成功地建模了关节之间的局部关系，并捕捉了整个输入序列的长程全局依赖。

表 7. 在 HumanEva 测试集上的 MPJPE 评估。FT 表示使用在 Human3.6M 上预训练的模型进行微调。

<table><tr><td></td><td colspan="3">walk（行走）</td><td colspan="3">jog（慢跑）</td></tr><tr><td></td><td>S1</td><td>S2</td><td>S3</td><td>S1</td><td>S2</td><td>S3</td></tr><tr><td>PoseFormer (f = 43)</td><td>16.3</td><td>11.0</td><td>47.1</td><td>25.0</td><td>15.2</td><td>15.1</td></tr>
<tr><td>PoseFormer (f = 43) FT</td><td>14.4</td><td>10.2</td><td>46.6</td><td>22.7</td><td>13.4</td><td>13.4</td></tr></table>

对小型数据集的泛化能力。先前的工作如 [12] 认为 Transformer 在数据量不足时泛化能力较差。我们使用模型在小型数据集 HumanEva [34] 上进行实验，研究 Transformer 的学习能力。与 Human3.6M（>1M 帧）相比，HumanEva 是一个小得多的数据集（<50K 帧）。表 7 展示了从零训练以及使用在 Human3.6M 上预训练的模型进行微调的结果。我们发现微调可以带来大幅度的性能提升，这与之前的观察 [12, 36] 一致，即 Transformer 在大规模数据集上进行预训练后可以表现良好。

## 5. 结论

本文中，我们提出了 PoseFormer（姿态变换器），一种基于纯Transformer的从二维视频进行三维姿态估计的方法。空间Transformer模块对二维关节点之间的局部关系进行编码，时间Transformer模块则捕获跨越任意帧的全局依赖关系，不受帧间距离的限制。大量的实验表明，我们的模型在两个流行的三维姿态数据集上取得了最先进的性能。

致谢：本工作部分由国家科学基金会（National Science Foundation, NSF）资助，资助编号 1910844。

## 参考文献

[1] Alexei Baevski and Michael Auli. Adaptive input representations for neural language modeling. arXiv preprint arXiv:1809.10853, 2018.

[2] Y. Cai, L. Ge, J. Liu, J. Cai, T. Cham, J. Yuan, and N. M. Thalmann. Exploiting spatial-temporal relationships for 3d pose estimation via graph convolutional networks. In ICCV, 2019.

[3] Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh. Realtime multi-person 2d pose estimation using part affinity fields. In CVPR, 2017.

[4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European Conference on Computer Vision, pages 213–229. Springer, 2020.

[5] Tianlang Chen, Chen Fang, Xiaohui Shen, Yiheng Zhu, Zhili Chen, and Jiebo Luo. Anatomy-aware 3d human pose estimation with bone-based pose decomposition. IEEE Transactions on Circuits and Systemsfor Video Technology, 2021.

[6] Yilun Chen, Zhicheng Wang, Yuxiang Peng, Zhiqiang Zhang, Gang Yu, and Jian Sun. Cascaded pyramid network for multi-person pose estimation. In CVPR, 2018.

[7] Yilun Chen, Zhicheng Wang, Yuxiang Peng, Zhiqiang Zhang, Gang Yu, and Jian Sun. Cascaded pyramid network for multi-person pose estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7103–7112, 2018.

[8] Y. Cheng, B. Yang, B. Wang, Y. Wending, and R. Tan. Occlusion-aware networks for 3d human pose estimation in video. In ICCV, 2019.

[9] H. Ci, C. Wang, X. Ma, and Y. Wang. Optimizing network structure for 3d human pose estimation. In ICCV, 2019.

[10] Rishabh Dabral, Anurag Mundhada, Uday Kusupati, Safeer Afaque, Abhishek Sharma, and Arjun Jain. Learning 3d human pose from structure and motion. In ECCV, 2018.

[11] Rishabh Dabral, Anurag Mundhada, Uday Kusupati, Safeer Afaque, Abhishek Sharma, and Arjun Jain. Learning 3d human pose from structure and motion. In Proceedings of the European Conference on Computer Vision (ECCV), pages 668–683, 2018.

[12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021.

[13] Hao-Shu Fang, Shuqin Xie, Yu-Wing Tai, and Cewu Lu. Rmpe: Regional multi-person pose estimation. In ICCV, 2017.

[14] Kai Han, Yunhe Wang, Hanting Chen, Xinghao Chen, Jianyuan Guo, Zhenhua Liu, Yehui Tang, An Xiao, Chunjing Xu, Yixing Xu, et al. A survey on visual transformer. arXiv preprint arXiv:2012.12556, 2020.

[14] Kai Han、Yunhe Wang、Hanting Chen、Xinghao Chen、Jianyuan Guo、Zhenhua Liu、Yehui Tang、An Xiao、Chunjing Xu、Yixing Xu 等. 视觉 Transformer 综述. arXiv 预印本 arXiv:2012.12556, 2020.

[15] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In European conference on computer vision, pages 646–661. Springer, 2016.

[15] Gao Huang、Yu Sun、Zhuang Liu、Daniel Sedra 和 Kilian Q Weinberger. 具有随机深度的深度网络. 发表于 欧洲计算机视觉国际会议 (ECCV), 第 646–661 页. Springer, 2016.

[16] C. Ionescu, D. Papava, V. Olaru, and C. Sminchisescu. Human3.6m: Large scale datasets and predictive methods for 3d human sensing in natural environments. IEEE TPAMI, 2014.

[16] C. Ionescu、D. Papava、V. Olaru 和 C. Sminchisescu. Human3.6m：自然环境中 3D 人体感知的大规模数据集与预测方法. IEEE TPAMI, 2014.

[17] Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, and Mubarak Shah. Transformers in vision: A survey. arXiv preprint arXiv:2101.01169, 2021.

[17] Salman Khan、Muzammal Naseer、Munawar Hayat、Syed Waqas Zamir、Fahad Shahbaz Khan 和 Mubarak Shah. 视觉中的 Transformer：综述. arXiv 预印本 arXiv:2101.01169, 2021.

[18] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[18] Diederik P Kingma 和 Jimmy Ba. Adam：一种随机优化方法. arXiv 预印本 arXiv:1412.6980, 2014.

[19] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations (ICLR), 2017.

[19] Thomas N. Kipf 和 Max Welling. 基于图卷积网络的半监督分类. 发表于 国际学习表征会议 (ICLR), 2017.

[20] Shichao Li, Lei Ke, Kevin Pratama, Yu-Wing Tai, Chi-Keung Tang, and Kwang-Ting Cheng. Cascaded deep monocular 3d human pose estimation with evolutionary training data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

[20] Shichao Li、Lei Ke、Kevin Pratama、Yu-Wing Tai、Chi-Keung Tang 和 Kwang-Ting Cheng. 使用进化训练数据的级联深度单目 3D 人体姿态估计. 发表于 IEEE/CVF 计算机视觉与模式识别会议 (CVPR) 论文集, 2020 年 6 月.

[21] Z. Li, X. Wang, F. Wang, and P. Jiang. On boosting single-frame 3d human pose estimation via monocular videos. In ICCV, 2019.

[21] Z. Li、X. Wang、F. Wang 和 P. Jiang. 通过单目视频提升单帧 3D 人体姿态估计. 发表于 ICCV, 2019.

[22] Jiahao Lin and Gim Hee Lee. Trajectory space factorization for deep video-based 3d human pose estimation. In BMVC, 2019.

[22] Jiahao Lin 和 Gim Hee Lee. 基于深度视频的 3D 人体姿态估计中的轨迹空间分解. 发表于 BMVC, 2019.

[23] Kevin Lin, Lijuan Wang, and Zicheng Liu. End-to-end human pose and mesh reconstruction with transformers. arXiv preprint arXiv:2012.09760, 2020.

[23] Kevin Lin、Lijuan Wang 和 Zicheng Liu. 基于 Transformer 的端到端人体姿态与网格重建. arXiv 预印本 arXiv:2012.09760, 2020.

[24] Kenkun Liu, Rongqi Ding, Zhiming Zou, Le Wang, and Wei Tang. A comprehensive study of weight sharing in graph networks for 3d human pose estimation. In ECCV, 2020.

[24] Kenkun Liu、Rongqi Ding、Zhiming Zou、Le Wang 和 Wei Tang. 3D 人体姿态估计中图网络权重共享的综合研究. 发表于 ECCV, 2020.

[25] Ruixu Liu, Ju Shen, He Wang, Chen Chen, Sen-ching Cheung, and Vijayan Asari. Attention mechanism exploits temporal contexts: Real-time 3d human pose reconstruction. In CVPR, 2020.

[25] Ruixu Liu、Ju Shen、He Wang、Chen Chen、Sen-ching Cheung 和 Vijayan Asari. 注意力机制利用时间上下文：实时 3D 人体姿态重建. 发表于 CVPR, 2020.

[26] Julieta Martinez, Rayat Hossain, Javier Romero, and James J. Little. A simple yet effective baseline for 3d human pose estimation. In ICCV, 2017.

[26] Julieta Martinez、Rayat Hossain、Javier Romero 和 James J. Little. 一个简单而有效的 3D 人体姿态估计基线方法. 发表于 ICCV, 2017.

[27] Dushyant Mehta, Helge Rhodin, Dan Casas, Pascal Fua, Oleksandr Sotnychenko, Weipeng Xu, and Christian Theobalt. Monocular 3d human pose estimation in the wild using improved cnn supervision. In 2017 international conference on 3D vision (3DV), pages 506–516. IEEE, 2017.

[27] Dushyant Mehta、Helge Rhodin、Dan Casas、Pascal Fua、Oleksandr Sotnychenko、Weipeng Xu 和 Christian Theobalt. 使用改进的 CNN 监督在野外进行单目 3D 人体姿态估计. 发表于 2017 年国际 3D 视觉会议 (3DV), 第 506–516 页. IEEE, 2017.

[28] Dushyant Mehta, Srinath Sridhar, Oleksandr Sotnychenko, Helge Rhodin, Mohammad Shafiei, Hans-Peter Seidel, Weipeng Xu, Dan Casas, and Christian Theobalt. Vnect: Real-time 3d human pose estimation with a single rgb camera. ACM Transactions on Graphics (TOG), 36(4):1–14, 2017.

[28] Dushyant Mehta、Srinath Sridhar、Oleksandr Sotnychenko、Helge Rhodin、Mohammad Shafiei、Hans-Peter Seidel、Weipeng Xu、Dan Casas 和 Christian Theobalt. VNect：使用单个 RGB 摄像头进行实时 3D 人体姿态估计. ACM 图形学汇刊 (TOG), 36(4):1–14, 2017.

[29] Gyeongsik Moon and Kyoung Mu Lee. I2l-meshnet: Image-to-lixel prediction network for accurate 3d human pose and mesh estimation from a single rgb image. In ECCV, 2020.

[29] Gyeongsik Moon 和 Kyoung Mu Lee. I2L-MeshNet：从单张 RGB 图像进行精确 3D 人体姿态与网格估计的图像到线素（lixel）预测网络. 发表于 ECCV, 2020.

[30] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017.

[31] Georgios Pavlakos, Xiaowei Zhou, and Kostas Daniilidis. Ordinal depth supervision for 3D human pose estimation. In CVPR, 2018.

[32] Dario Pavllo, Christoph Feichtenhofer, David Grangier, and Michael Auli. 3d human pose estimation in video with temporal convolutions and semi-supervised training. In CVPR, 2019.

[33] Mir Rayat Imtiaz Hossain and James J. Little. Exploiting temporal information for 3d human pose estimation. In The European Conference on Computer Vision (ECCV), 2018.

[34] L. Sigal, A. Balan, and M. J. Black. HumanEva: Synchronized video and motion capture dataset and baseline algorithm for evaluation of articulated human motion. IJCV, 2010.

[35] Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. Deep high-resolution representation learning for human pose estimation. In CVPR, 2019.

[36] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. arXiv preprint arXiv:2012.12877, 2020.

[37] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. 2017.

[38] Jingbo Wang, Sijie Yan, Yuanjun Xiong, and Dahua Lin. Motion guided 3d pose estimation from videos. In European Conference on Computer Vision, pages 764–780. Springer, 2020.

[39] Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F Wong, and Lidia S Chao. Learning deep transformer models for machine translation. arXiv preprint arXiv:1906.01787, 2019.

[40] Sen Yang, Zhibin Quan, Mu Nie, and Wankou Yang. Transpose: Towards explainable human pose estimation by transformer. arXiv preprint arXiv:2012.14214, 2020.

[41] Linwei Ye, Mrigank Rochan, Zhi Liu, and Yang Wang. Cross-modal self-attention network for referring image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10502–10511, 2019.

[42] Raymond A Yeh, Yuan-Ting Hu, and Alexander G Schwing. Chirality nets for human pose regression. 2019.

[43] Ailing Zeng, Xiao Sun, Fuyang Huang, Minhao Liu, Qiang Xu, and Stephen Lin. Srnet: Improving generalization in 3d human pose estimation with a split-and-recombine approach. In ECCV, 2020.

[44] Tianshu Zhang, Buzhen Huang, and Yangang Wang. Object-occluded human shape and pose estimation from a single color image. In CVPR, 2020.

[45] Long Zhao, Xi Peng, Yu Tian, Mubbasir Kapadia, and Dimitris N. Metaxas. Semantic graph convolutional networks for 3d human pose regression. In CVPR, 2019.

[46] Ce Zheng, Wenhan Wu, Taojiannan Yang, Sijie Zhu, Chen Chen, Ruixu Liu, Ju Shen, Nasser Kehtarnavaz, and Mubarak Shah. Deep learning-based human pose estimation: A survey, 2020.

[47] Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip HS Torr, et al. Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers. arXiv preprint arXiv:2012.15840, 2020.

[48] Kaixiong Zhou, Xiao Huang, Yuening Li, Daochen Zha, Rui Chen, and Xia Hu. Towards deeper graph neural networks with differentiable group normalization. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 4917–4928. Curran Associates, Inc., 2020.

[49] Xingyi Zhou, Qixing Huang, Xiao Sun, Xiangyang Xue, and Yichen Wei. Towards 3d human pose estimation in the wild: A weakly-supervised approach. In ICCV, 2017.

[50] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020.

## 附录

在本附录中，我们提供以下内容：

• 空间注意力图与时间注意力图的可视化展示。

• 逐帧比较以追踪所有关节点在帧间的平均 MPJPE。

• 更多三维姿态估计的定性比较结果。

• 使用所提出的 PoseFormer 在从 YouTube 上收集的野外视频上进行三维姿态估计的结果。

我们还提供了演示视频，以展示所提出的 PoseFormer 在三维人体姿态估计上的结果。更多详情，请访问 https://github.com/zczcwh/PoseFormer

## A. 注意力可视化

我们展示了在 Human3.6M 测试集 S11 的 SittingDown（坐下）动作上评估 PoseFormer 模型时，全部 8 个注意力头的空间注意力图与时间注意力图的更多可视化样例。对于图 7 中的空间自注意力图，横轴对应 17 个关节点的查询，纵轴表示注意力输出。不同的注意力头返回不同的注意力强度，这反映了在输入关节点之间学习到的多种局部关系。对于图 8 中的时间自注意力图，横轴对应 81 帧的查询，纵轴表示注意力输出。不同的注意力头捕捉到了长程全局依赖关系。空间注意力图和时间注意力图均表明，PoseFormer 不仅成功编码了二维关节点之间的局部关系，还建模了跨任意帧（无论距离远近）的全局依赖。

## B. 逐帧分析

我们通过对每一帧中所有估计关节点计算平均 MPJPE，进行逐帧姿态估计分析。如图 9 所示，我们在 Human3.6M [16] 测试集 S11 的 Eating（吃东西）和 Photo（拍照）动作上测量了逐帧 MPJPE。我们的 PoseFormer（红线）在两个动作的大多数帧上都产生了更低的 MPJPE，相比于我们的基线（仅使用时间 Transformer）以及当前最优方法 [5]。

## C. 更多定性结果

我们提供了更多 3D 估计姿态与真实值之间的可视化对比。我们在 Human3.6M 测试集 S11 上的 Greeting 和 Walk-Dog 动作上对 PoseFormer 进行了评估。与当前最先进方法 [5] 以及我们的基线方法相比，PoseFormer 实现了更精确的估计结果，如图 7 所示。

## D. 在野外视频上的性能表现

我们的模型在室内数据集 Human3.6M 上进行训练，该数据集背景静态且相机拍摄设置已知。从野外视频中估计 3D 人体姿态更具挑战性，原因在于动态环境以及未知的相机设置。在非受限环境下，前景/背景物体的外观常常存在较大变化，并且存在严重的遮挡。我们也在一些来自 YouTube 的在线视频上评估了 PoseFormer 的性能，如图 8 所示。我们首先使用 AlphaPose [13] 作为 2D 姿态检测器从视频帧中生成 2D 姿态，然后应用 PoseFormer 进行 3D 姿态估计。我们观察到，PoseFormer 在大多数帧中都能达到可接受的性能，但由于 2D 姿态检测不准确、遮挡以及快速运动，仍存在一些失败案例（见图 8）。由于 PoseFormer 是一种 2D 到 3D 的提升方法，任何检测错误的 2D 姿态都可能导致 3D 姿态估计不准确。遮挡是 3D 人体姿态估计（HPE）中仍然存在的一个关键挑战，因为信息缺失。此外，极端快速运动下的估计可能会受到运动模糊帧的影响。

![](images/d9c75798c7df8bdef8e5a7367ad77262533edc676304abf34b0d49d5c5a22a17.jpg)

![](images/29fba13b400655a324052bef92140f8570e3ac8445627396d6661eefa6e03fa5.jpg)

![](images/d6a30201e5b62d0af3ad9e3059e9ba1b40553bac47e87e92e44ad73b69d47582.jpg)

![](images/bb6cd8a7b06e31d6008b80a715d0ef6bcfac53a50969f8eab716e2b3fe36c49b.jpg)

![](images/1b176bd638b175ff0cb8506a236ddfee9c3286590f231f7dcbb588c654450335.jpg)

![](images/e40d9ae34a25a21291657cbf1d6f67021dcc906185bc7f554bcb60f38d6d2394.jpg)

![](images/8e49368baef137878d01be00a80a9e5b75c487bc96786bff5518e0d59b1b3fee.jpg)

![](images/d42c0087dd196843c8eb4eacaf166541132dc77a83bd2f668c2b0cc5e7d8d367.jpg)  
图 7. 空间 Transformer 中自注意力的可视化。x 轴（水平方向）和 y 轴（垂直方向）分别对应查询（queries）和预测输出。像素 w (i: 行, j: 列) 表示第 j 个查询对第 i 个输出的注意力权重。红色表示更强的注意力。注意力输出从 0 到 1 进行归一化。

![](images/2e83b319c602b930dbacc3433a64ae3101f30fa7e1f314197790910bb8d2d753.jpg)

![](images/b466b8bf91890f1031ed2905be0df29cc6eae39f5a4ecf3bf9426995e5e54902.jpg)

![](images/db99f287463f92d122ebdd8d5c1a258465f42fa87e8599814864330e07104584.jpg)

![](images/20ef6c5e98b25a5983259b0d40650f3bf0bdfd11c67b769718987e25f921ee0e.jpg)

![](images/5923ab9659997de6541ebace3461d2f8ef72455fc1acd756dca3502575b3833f.jpg)

Figure 8. 时序 Transformer 中自注意力的可视化。横轴（水平方向）对应查询（queries），纵轴（垂直方向）对应预测输出。像素 $w_{i,j}$（i：行，j：列）表示第 j 个查询对第 i 个输出的注意力权重。红色表示更强的注意力。注意力输出归一化到 0 到 1 之间。

Figure 9. 在 Human3.6M 测试集上，我们的方法 (PoseFormer)、基线方法与 SOTA 方法 Chen 等人 [5] 的逐帧比较。上图：S11 的 Eating 动作。下图：S11 的 Photo 动作。

Figure 10. 在 Human3.6M 测试集 S11 的 Greeting 和 WalkDog 动作上，我们的方法 (PoseFormer)、基线方法与 SOTA 方法 Chen 等人 [5] 的定性比较。绿色箭头突出显示了 PoseFormer 结果明显更好的位置。

合理的情况

不佳的情况

Figure 11. 野外视频上的定性结果：原始帧序列及其检测到的 2D 关节和 PoseFormer 恢复的 3D 姿态。