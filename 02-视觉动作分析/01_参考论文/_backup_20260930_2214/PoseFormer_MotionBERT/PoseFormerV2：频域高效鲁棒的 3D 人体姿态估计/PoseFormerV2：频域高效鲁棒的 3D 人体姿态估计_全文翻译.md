# PoseFormerV2：探索频域以实现高效且鲁棒的3D人体姿态估计

Qitao Zhao<sup>1\*</sup> Ce Zheng<sup>2</sup> Mengyuan Liu<sup>3</sup> Pichao Wang<sup>4</sup> Chen Chen<sup>2</sup>

<sup>1</sup>山东大学 <sup>2</sup>中佛罗里达大学计算机视觉研究中心 <sup>4</sup>亚马逊Prime Video

<sup>3</sup>北京大学深圳研究生院机器感知重点实验室

qitaozhao@mail.sdu.edu.cn cezheng@knights.ucf.edu nkliuyifang@gmail.com

pichaowang@gmail.com

chen.chen@crcv.ucf.edu

## 摘要

近期，基于Transformer的方法在序列式2D到3D提升的人体姿态估计中取得了显著成功<sup>0.9</sup>。作为先驱性工作，PoseFormer通过级联的Transformer层捕获每一帧中人体的空间关系以及跨帧的人体动态，并取得了令人瞩目的性能。然而，在实际场景中，PoseFormer及其后续方法的性能受限于两个因素：（a）输入关节点序列的长度；（b）2D关节点检测的质量。现有方法通常对输入序列的所有帧应用自注意力，当帧数增加以获得更高的估计精度时，这会带来巨大的计算负担，并且这些方法对2D关节点检测器能力有限所天然引入的噪声并不具备鲁棒性。本文提出PoseFormerV2，利用频域中对长骨架序列的紧凑表征，在有效扩大感受野的同时提升对噪声2D关节点检测的鲁棒性。在对PoseFormer进行最小改动的情况下，所提出的方法能够有效融合时域与频域的特征，相较于其前身实现了更好的速度-精度权衡。在两个基准数据集（即Human3.6M和MPI-INF-3DHP）上进行的广泛实验表明，所提出的方法显著优于原始的PoseFormer以及其他基于Transformer的变体。代码已发布于 https://github.com/QitaoZhao/PoseFormerV2。

## 1. 引言

3D人体姿态估计（Human Pose Estimation, HPE）旨在基于单目视频（不经过中间2D表征）[25,28]或2D人体关节点序列（称为2D到3D提升方法）[5,18,37,43]，在三维空间中定位人体关节点。随着2D人体姿态检测器[6, 26]的大量普及以及人体2D骨架表征的轻量化特性，基于提升的方法如今在3D人体姿态估计中占据主导地位。与原始的单目视频相比，每一视频帧中人体的2D关节点坐标在内存方面更为友好，这使得基于提升的方法能够利用长的关节点序列来提升姿态估计精度。

![](images/ba656e9713e79b8b8e967b4abf6b9ef69031adf5a923af0bcfd8559bd31a32e6.jpg)  
**图 1.** PoseFormerV2 与 PoseFormerV1 [46] 在 Human3.6M [13] 上的对比。RF 表示感受野（Receptive Field），k×RF 表示完整序列长度与输入 PoseFormerV2 空间编码器的帧数之比为 k，即空间编码器的感受野通过完整序列的少量低频系数扩展了 k×。所提方法在速度-精度权衡方面大幅优于 PoseFormerV1，且更大的 k 带来更显著的提升，例如当 k 为 27 时可获得 4.6× 的加速。

Transformer [36] 最早在自然语言处理（NLP）领域 [3, 8] 取得巨大成功，随后将其能力扩展到计算机视觉社区，成为多项视觉任务的事实标准方法，例如图像分类 [9, 19, 35]、目标检测 [4, 47] 和视频识别 [1, 2, 42]。人体关节表示的离散性以及在骨架序列中对长程时序依赖建模的需求，使得 Transformer 成为基于提升（lifting）的人体姿态估计的理想选择。已有工作 [15, 16, 32, 44, 46] 已将 Transformer 作为 3D 人体姿态估计的骨干网络，并展示了良好的效果。

![](images/0458882d3d801664b33f3f32f00757ecf61973d1bf24277e3d51f0084f118123.jpg)  
**图 2.** PoseFormerV1 概览。PoseFormerV1 主要由两个模块组成：空间 Transformer 编码器和时间 Transformer 编码器。PoseFormerV1 的时间编码器对给定 2D 关节序列的所有帧应用自注意力机制以进行人体运动建模。

**表 1.** 将真值 2D 检测替换为 CPN [6] 2D 姿态检测后，所带来的计算开销和性能下降。性能下降在 Human3.6M 数据集（协议 1）[13] 上报告。RF：感受野（Receptive Field），含义与图 1 相同。

<table><tr><td colspan="2">方法</td><td>序列长度</td><td>GFLOPs</td><td>性能下降 (mm)</td></tr><tr><td colspan="2">PoseFormerV1 [46]</td><td>81</td><td>1.36</td><td>13.0</td></tr><tr><td rowspan="3">StridedTransformer [15] MixSTE [44]</td><td>ICCV&#x27;21 TMM&#x27;22</td><td>243</td><td>1.37</td><td>15.2</td></tr><tr><td>CVPR&#x27;22</td><td>81</td><td>92.46</td><td>16.5</td></tr><tr><td>CVPR&#x27;22 ECCV&#x27;22</td><td>81</td><td>3.12</td><td>11.8</td></tr><tr><td colspan="2">P-STMO [32] PoseFormerV2 (9×RF) PoseFormerV2 (27× RF)</td><td>243 81</td><td>1.74 0.35</td><td>13.5 8.2</td></tr></tr>

作为基于 Transformer 方法的开创性工作，PoseFormer [46] 将关节序列特征提取分解为两个阶段（见图 2），并优于传统的基于卷积的方法。首先，将每一帧内的所有关节线性投影到高维向量（即关节 token），作为空间 Transformer 编码器的输入。空间编码器通过自注意力机制在单帧内建立关节间的依赖关系。在第二阶段，将每一帧的关节 token 组合为一个帧 token，作为时间编码器的输入，以建模序列中跨帧的人体运动。更多细节见第 3.1 节。

尽管 PoseFormer（以及其他基于 Transformer 的方法）能力强大，但其性能受到两个关键因素的限制。**(a) 输入二维骨架序列的长度（帧数）**。最先进的基于 Transformer 的方法通常使用极长的序列以获得先进性能，例如 PoseFormer [46] 使用 81 帧，P-STMO [15] 使用 243 帧，MHFormer [16] 使用 351 帧。然而，对如此长的序列密集应用自注意力机制在计算上非常昂贵，例如，在 RTX 3090 GPU 上，3 帧 PoseFormer 单个 epoch 的实际训练耗时约为 5 分钟，而 81 帧 PoseFormer 的耗时则激增至约 1.5 小时。**(b) 二维关节检测的质量**。二维关节检测器不可避免地会引入噪声，这是由于其训练数据集中的偏差以及单帧估计范式带来的时间不一致性。例如，PoseFormer 在 Human3.6M 数据集 [13] 上使用真实（ground-truth）二维检测结果时达到 31.3 mm 的 MPJPE（平均关节位置误差，Mean Per Joint Position Error）。当干净的输入被 CPN [6] 的二维姿态检测结果取代后，该结果显著下降至 44.3 mm。在实际应用中，长序列推理对于 AR/VR 头显等资源受限设备的硬件部署而言可能成本过高，并且高质量的二维检测也难以获得。有关现有基于 Transformer 的方法处理长序列的效率以及对噪声二维关节检测的鲁棒性的更多定量结果，请参见表 1。

基于这些实际考量，我们提出两个重要的研究问题：

- **Q1：如何高效利用长关节序列以获得更高的估计精度？**

- **Q2：如何提高模型对不可靠二维姿态检测的鲁棒性？**

很少有工作尝试通过引入手工设计的模块来回答这两个问题中的任意一个，例如，仅处理部分视频帧以提高效率的下采样-上提升模块（downsampling-and-uplifting module）[10]，用于建模人体部位深度歧义和二维检测器不确定性的多假设模块（multi-hypothesis module）[16]。然而，这些方法均未能同时找到解决这两个问题的单一方案。更糟糕的是，上述两个问题的解决方案之间似乎存在矛盾，例如，多假设方法 [16] 提升了鲁棒性，但却带来了额外的计算开销（另见表 1）。

在本文中，我们进行了"一石二鸟"的初步尝试。在对先前最优方法 PoseFormer 进行适度修改的基础上，我们表明输入序列的合适表示形式可能是同时回答上述两个问题的关键。具体而言，我们聚焦于三维人体姿态估计（Human Pose Estimation, HPE）文献中几乎未被探索的频域，并提出将输入骨骼序列编码为低频系数。这一表示背后的洞见出奇地简单：一方面，低频分量足以表示整个视觉身份 [38, 41]（例如图像压缩中的二维图像以及本文中的关节轨迹），从而无需使用开销巨大的全帧自注意力；另一方面，骨骼序列本身的低频表示能够滤除检测到的关节轨迹中所包含的高频噪声（抖动和异常值）[21, 22]。

我们沿用了 PoseFormer 的时空架构，但强制空间 Transformer 编码器仅"观察"长序列中的少数中心帧。然后，我们用完整序列低频分量所蕴含的全局特征，来补充"短视"的帧级特征（空间编码器的输出）。在不依赖对所有时间步进行高开销的帧间自注意力的情况下，时间 Transformer 编码器被重新构建为时频特征融合模块（Time-Frequency Feature Fusion module）。

在两个人体姿态估计基准数据集（即 Human3.6M [13] 和 MPI-INF-3DHP [23]）上的大量实验表明，本文所提出的方法（称为 PoseFormerV2）在速度-精度权衡以及对二维关节检测噪声的鲁棒性方面，显著优于其前身（见图 1）以及其他基于 Transformer 的变体。本文的贡献包含三个方面：

• 据我们所知，我们是首个将输入关节序列的频域表示用于二维到三维提升式人体姿态估计的研究。我们发现该表示非常适合同时解决该领域中两个重要问题（即处理长序列的效率以及对不可靠关节检测的鲁棒性），并且实验证据表明该方法可轻松推广到其他模型。

- 我们设计了一种有效的时频特征融合（Time-Frequency Feature Fusion）模块，以缩小时域特征与频域特征之间的差距，使我们能够在速度和精度之间实现灵活的平衡。

- 我们的 PoseFormerV2 在速度-精度权衡和 Human3.6M 数据集上的鲁棒性方面优于其他基于 Transformer 的方法，并在 MPI-INF-3DHP 数据集上达到了最先进水平。

## 2. 相关工作

我们的方法建立在概念上简洁的 PoseFormer [46] 之上，旨在从频域视角提升其处理长序列的效率，以及其对噪声关节检测的鲁棒性。因此，这里我们主要聚焦于 2D 到 3D 提升人体姿态估计（HPE）中这一研究方向（基于 Transformer 的方法），并介绍计算机视觉文献中频域表示的应用，尤其是与基于提升的三维 HPE 最相关的基于骨骼的任务。

## 2.1. 基于 Transformer 的三维人体姿态估计

PoseFormer [46] 是首个在基于提升的三维人体姿态估计中采用视觉 Transformer 作为骨干网络的工作，并且以较大优势超越了之前的基于 CNN 的方法。尽管具有竞争力，Zhang 等人 [44] 指出 PoseFormer 的"先空间-后时间"范式可能忽略了每个关节的独特时间模式，并提出采用交替的空间-时间 Transformer 层来实现细粒度的、针对特定关节的特征提取。MHFormer [16] 进一步将任务相关的先验知识融入 Transformer 中用于三维 HPE。具体而言，2D 到 3D 提升是一个逆问题，可能存在多个合理的解，因此他们生成多个假设来建模模糊的身体部位和关节检测器中的不确定性，从而取得了先进的性能。受到掩码图像建模（Masked Image Modeling, MIM）在图像分类中进展 [12, 39, 40] 的启发，P-STMO [32] 将掩码关节建模应用于具有自监督学习的三维 HPE。

另一类工作 [10, 15] 提升了基于 Transformer 的方法的效率。利用 2D 关节序列中的时间冗余性，StridedTransformer [15] 用步幅卷积替代了参数繁重的全连接层。Einfalt 等人 [10] 声称逐帧的 2D 关节检测甚至比提升模型本身计算成本更高，因此提出以固定间隔对输入视频帧进行下采样，并仅在这些采样帧上应用 2D 关节检测器和提升模型。尽管比先前的工作更高效，但上述方法 [10, 15] 仅利用相邻视频帧之间的一致性来减少沿时间维度的自注意力中的参与元素，而非从全局视角出发，因此可能会出现较大的性能下降。

## 2.2. 视觉中的频率表示

由于人类视觉系统对图像的低频成分更敏感，传统的图像压缩算法（如 JPEG [30] 和 JPEG 2000 [33]）通过为图像的低频离散余弦变换（DCT）系数分配更多的存储预算，来降低存储 2D 图像的内存开销。基于同样的逻辑，[41] 提出自适应地去除 DCT 成分中无信息量的通道，以提高 2D 图像分类的效率。最近，一些工作 [11, 31] 提出用频域变换来替代计算昂贵的自注意力机制，这些变换可以通过其快速算法进行加速。GFN [31] 提出利用可学习的频域滤波器来高效地混合视觉 token，而 AFNO [11] 则通过算子学习进一步提升了频域中 token mixer 的性能。此外，Wang 等人 [38] 利用低频快速傅里叶变换（FFT）成分来压缩视觉 Transformer。

基于骨架的任务与我们的工作更为相关，因为我们以 2D 骨架序列作为输入。在人体运动预测领域，先前的工作 [21, 22] 将骨架序列从时域变换到 DCT 系数，以对人体动态进行编码，相比于静态关节坐标效果更好。他们观察到，丢弃少量高频系数不一定会导致性能下降，甚至可以提高所预测未来运动的平滑度。然而，在基于提升的 3D 人体姿态估计中，2D 关节序列的频域表示尚未被探索。

我们的方法受到了这些将频域变换应用于视觉任务的前期尝试的启发，但视角有所不同。我们在第 3.2.1 节中详细阐述了选择 DCT 系数表示的动机。

## 3. 方法

PoseFormer [46] 通过将序列特征提取分解为两个阶段（即空间编码器和时间编码器）来促进 3D 人体姿态估计，并已被证明是有效的。然而，当输入序列的长度增加时，它的计算负担很大，并且对噪声关节检测敏感。在本节中，我们介绍 PoseFormerV2 的细节，它利用输入序列的频域表示来克服上述两个问题。

## 3.1. PoseFormerV1 [46] 的预备知识

我们首先简要概述 PoseFormerV1（见图 2），为其后续改进的讨论奠定基础。PoseFormerV1 包含两个主要模块：用于单帧关节点相关性建模的空间编码器（spatial encoder）和用于跨帧人体运动建模的时间编码器（temporal encoder）。给定输入的二维骨架序列 $\mathbf { x } \in \mathbb { R } ^ { F \times J \times 2 }$，其中 $F$ 表示序列长度，$J$ 表示人体表示的关节点数量。首先，每一帧中人体所有关节点的坐标被线性投影到一个 $c$ 维向量（即关节点嵌入 joint embedding），记为 ${ \bf z } _ { 0 } \in \mathbb { R } ^ { F \times J \times c }$。将一个可学习的空间位置嵌入 $\mathbf { E } _ { S P o s } \in \mathbb { R } ^ { 1 \times J \times c }$ [9] 加到 $\mathbf { z } _ { 0 }$ 上，以编码与关节点相关的信息。

**空间 Transformer 编码器（Spatial transformer encoder）** 利用自注意力机制独立地为每一帧的关节点嵌入 $\mathbf { z } _ { 0 } ^ { i } \in \mathbb { R } ^ { 1 \times J \times c }$ 建立空间依赖关系。在此阶段，输入到每个 Transformer 块的 token 数量为 $J$。第 $i$ 帧经过 $L$ 层空间 Transformer 编码器后的输出记为 $\mathbf { z } _ { L } ^ { i } \in \mathbb { R } ^ { 1 \times J \times c }$。然后，将每帧的表示展平并拼接，作为输入 $\mathbf { Z } _ { 0 } \in \bar { \mathbb { R } } ^ { F \times ( J \cdot c ) }$ 送入时间 Transformer 编码器。

**时间 Transformer 编码器（Temporal transformer encoder）。** 类似地，输入 $\mathbf { Z } _ { 0 }$ 加上一个可学习的时间位置嵌入 $\mathbf { E } _ { T P o s } \in \mathbb { R } ^ { F \times ( J \cdot c ) }$，以编码每一帧与索引相关的信息。具有 $M$ 个 Transformer 层的时间编码器密集建模整个序列中帧与帧之间的依赖关系，其输出记为 $\mathbf { Z } _ { M } \in \mathbb { R } ^ { F \times ( J \cdot c ) }$。在此阶段，每个 Transformer 层的 token 数量为 $F$，即输入序列的长度。

**回归头（Regression head）。** 为了估计序列中中间帧的三维姿态，采用一个简单的一维卷积来收集时间信息，并通过线性投影得到最终的姿态表示 $\mathbf { y } \in \mathbb { R } ^ { 1 \times ( J \cdot 3 ) }$。

![](images/2cc303397f35a8fd019593356f654f5071f490d5c252aa5702fcdd332f505b5f.jpg)

![](images/2cfc5f6dfb263d825376ef7741f68f751a99d0abe821cbb579a797ede72784d6.jpg)  
图 3. Human3.6M [13] 中由 CPN [6] 检测到的关节点轨迹的一个随机选取示例，及其使用前 3、9 和 27 个 DCT 系数（共 81 个）进行重建的结果。注意，即使仅使用前 3 个系数，重建曲线（橙色）也能捕捉到原始输入的整体特征，并且更加平滑。

# 局限性与方法动机

**PoseFormerV1 的局限性。** 使用 Transformer 层建模每一帧内关节之间的依赖关系以及跨帧的人体运动十分直接。尽管这种密集建模带来了更高的估计精度，但由于自注意力相对于 token 数量的二次计算复杂度，它在计算上并不友好（在空间编码器中 token 数量指的是关节数量，在时间编码器中指的是序列长度），尤其是在输入序列长度增加时。虽然空间 Transformer 层的 token 数量（即关节数量）独立于帧数，但值得注意的是，在实际场景中，由于 GPU 并行能力的限制，序列长度会隐式地影响空间编码器的计算开销。除了效率问题之外，PoseFormerV1 对输入二维关节检测的质量较为敏感（实验证据见表 1 和第 4.3 节）。

在下文中，我们提出一种替代方案，通过输入序列的频域表示来克服 PoseFormerV1 的局限性。

## 3.2. PoseFormerV2

## 3.2.1 骨骼序列的频域表示

**动机。** 我们提出使用离散余弦变换（DCT）将输入骨骼序列变换到频域，并仅利用其中一部分低频系数。DCT 系数对输入时间序列的多层次时间信息进行编码：低频系数编码其大致轮廓，而高频系数编码其细节，例如抖动或急剧变化。为了更好地说明我们选择这种表示的动机，我们在图 3 中展示了一个来自 Human3.6M [13] 数据集中"Directions"动作的 81 帧 CPN 检测 [6] 关节轨迹示例，以及分别用前 3、9 和 27 个 DCT 系数重建的结果。随着保留的 DCT 系数数量的增加，重建轨迹越来越接近原始输入，但平滑度降低。注意，仅使用 3 个 DCT 系数（橙色曲线表示）即可捕捉原始轨迹的整体趋势，而使用 9 个和 27 个系数（粉色和绿色曲线）则在去除高频噪声（锯齿状抖动）的同时，更好地保留了原始序列的特征。这些观察结果促使我们在输入关节序列中利用少数信息量大的低频 DCT 分量作为我们工作中的紧凑且去噪的序列表示。通过这种表示，我们显著缩短了作为输入的序列有效长度，并增强了模型对二维关节检测中噪声的鲁棒性。我们在补充材料中给出 DCT 的正式介绍。

图 4. (a) PoseFormerV2 概览。(b) 空间 Transformer 和时频特征融合 Transformer。(c) FreqMLP（频率多层感知机）。作为示例，在 (a) 中我们使用 3 个中心帧（索引 -1、0 和 1）进行细粒度的逐帧特征提取，并使用完整 9 帧序列的前 3 个 DCT 系数进行全局频域特征提取。因此，与 PoseFormerV1 相比，输入到空间编码器和时间编码器的有效帧数有所减少（分别为 3、6，而 PoseFormerV1 中为 9、9）。

## 3.2.2 架构

本节介绍所提方法 PoseFormerV2 的架构（概览见图 4）。

**空间 Transformer 编码器。** 给定一个二维骨架序列 $\mathbf{x} \in \mathbb{R}^{F \times J \times 2}$（最好是一个较长的序列，例如 $F=81$），我们首先围绕序列中心（图 4(a) 中索引为 0 的帧）采样 $F'$（通常 $F' \ll F$）帧，记作 $\mathbf{x}' \in \mathbb{R}^{F' \times J \times 2}$，作为空间编码器的输入。空间编码器的输出记作 $\mathbf{z}^{Time} \in \mathbb{R}^{F' \times (J \cdot c)}$（时域中的逐帧特征）。空间编码器的设计直接沿用 PoseFormerV1。

**低频 DCT 系数。** $\mathbf{z}^{Time}$ 被称作"短视"特征，因为其感受野（$F'$）相对于整个序列长度（$F$）是受限的。为了高效地利用原始序列的局部人体动态信息，我们借助其频域表示。我们首先将完整序列 $\mathbf{x} \in \mathbb{R}^{F \times J \times 2}$ 转换为 DCT 系数，记作 $\mathbf{C} \in \mathbb{R}^{F \times J \times 2}$。然后对每个关节点轨迹，使用低通滤波器仅保留前 $N$（$\ll F$）个系数 $\mathbf{C}' \in \mathbb{R}^{N \times J \times 2}$，其中原始序列的时间信息被大部分保留，而高频噪声被去除。

**低频系数** $\mathbf{C}'$ 被展平并线性投影到 $\mathbf{z}^{Freq} \in \mathbb{R}^{N \times (J \cdot c)}$（频率系数的嵌入）。$\mathbf{z}^{Freq}$ 与一个可学习的频率位置嵌入 $\mathbf{E}_{FPos}$（类似于 PoseFormerV1 中的 $\mathbf{E}_{TPos}$）相加。来自时域和频域的特征被拼接在一起，表示为

$$
\mathbf{z} = [\mathbf{z}^{Time}; \mathbf{z}^{Freq}], \tag{1}
$$

输入到时频特征融合模块中。

时频特征融合。我们采用 Transformer 层进行跨帧时间依赖建模，方法与 PoseFormerV1 相同。与完全在时域中提取特征的 PoseFormerV1 不同，所提方法融合了时域和频域两方面的特征。为了缩小这两个域之间的差距，我们对原始 Transformer 层进行了简单修改。(1) 时域和频域特征共享自注意力机制，但使用独立的前馈网络；(2) 我们在时域特征 $\mathbf { z } ^ { Time }$ 的前馈网络中应用 FreqMLP（Frequency Multi-Layer Perceptron，频率多层感知器）（参见图 4 (b)(c)）。在我们的 FreqMLP 中，在原始 MLP 前馈网络前后分别利用 DCT 和 IDCT。该方法背后的直觉是：频域特征通过低通滤波器可滤除高频噪声，但细节性的人体运动特征（例如快速的局部运动）也可能作为噪声被一并滤除。为解决这一问题，FreqMLP 充当一个可训练的频域滤波器，使我们能够自适应地调整 2D 关节点坐标嵌入（即时域特征）中每个频率分量的权重，从而作为频域特征的补充。这些模块表述为：

$$
\mathbf { z } _ { k } ^ { ' } = \mathrm { M S A } ( \mathbf { z } _ { k } ),\tag{2}
$$

$$
\begin{array} { r } { \mathbf { z } _ { k } ^ { T i m e } , \mathbf { z } _ { k } ^ { F r e q } = \mathbf { z } _ { k } ^ { ' } [ : F ^ { \prime } ] , \mathbf { z } _ { k } ^ { ' } [ F ^ { \prime } : ] , } \end{array}\tag{3}
$$

$$
\mathbf { z } _ { k + 1 } = \mathrm { C o n c a t } ( \mathrm { F r e q M L P } ( \mathbf { z } _ { k } ^ { T i m e } ) , \mathrm { M L P } ( \mathbf { z } _ { k } ^ { F r e q } ) ),\tag{4}
$$

其中 MSA 代表多头自注意力（Multi-head Self-Attention），$F ^ { \prime }$ 为采样的中心帧数量。上述修改的有效性在第 4.4 节中得到验证。需要认识到的是，拼接操作会导致 Transformer 的 token 数量增多。然而，通过限制空间编码器仅观察有限数量的中心帧，并结合少量低频 DCT 系数以扩展其感受野，我们能够灵活地降低整体计算量。该方法不仅降低了计算成本，还相较于 PoseFormerV1 增强了模型对噪声的抵抗能力。

回归头与损失函数。沿用 PoseFormerV1，我们使用 1D 卷积层来聚合时间信息，并采用线性投影得到序列中心帧的最终 3D 姿态 $\mathbf { y } \in \mathbb { R } ^ { 1 \times ( J \cdot 3 ) }$。我们采用与 PoseFormerV1 相同的标准 MPJPE（Mean Per Joint Position Error，平均每关节位置误差）损失来训练模型。

## 4. 实验

## 4.1. 数据集与评价指标

我们在两个 3D 人体姿态估计数据集上进行了实验，即 Human3.6M [13] 和 MPI-INF-3DHP [23]，以证明我们所提方法的有效性。关于这两个数据集及其各自评估指标的更详细描述见补充材料。

## 4.2. 实现细节与分析

所提方法包含三个重要的、与实验设置密切相关的超参数。这些参数包括：在空间编码器中作为输入使用的帧数 (f)、代表扩大感受野的整个输入序列的长度 (F)，以及用于引入长程时序信息的所保留的 DCT 系数数量 (n)。除非特别说明，为简便起见，我们直接设置 $n = f$。在实践中，可以进一步调节这些超参数以灵活地权衡速度与精度。当 f 等于 1 时，n 被设置为 3，因为单个 DCT 系数可能不足以对来自长输入序列的时序信息进行编码。由于 $f$ 和 n 是固定的，模型的计算复杂度也随之确定（即空间编码器的 token 数量和特征融合模块的 token 数量是固定的）。我们可以通过改变 $F$ 来有效地将模型的感受野从一个有限的 f 扩展到任意值，且不会带来额外的计算开销。这使我们能够高效地利用长序列来提升精度。模型架构与训练的超参数细节见补充材料。

## 4.3. 与最先进方法的比较

Human3.6M。我们在 Human3.6M 数据集上将我们的方法与 Pose-FormerV1 以及其他基于 Transformer 的方法进行比较（表 2）。我们通过变化 $f$ 的值和序列长度来展示模型的灵活性。当扩展比（即完整序列长度与 f 之比）较大时，我们的方法尤其高效。例如，在扩展比为 81 的情况下，我们的方法仅用 77.2 MFLOPs 就达到了 47.6 mm 的 MPJPE，而 MHFormer [16] 用 342.9 MFLOPs 仅达到 47.8 mm 的 MPJPE（速度慢 4.4×）。此外，在相近的计算开销（约 350 MFLOPs）和相同的完整序列长度（81）的条件下，我们的方法取得了 46.0 mm 的 MPJPE，而 StridedTransformer [15] 取得 47.5 mm 的 MPJPE（提升 3.2%）。图 3 给出了更清晰的对比，表明所提方法在速度-精度权衡方面优于其他基于 Transformer 的方法。需要注意的是，那些带有额外预训练阶段的方法以及计算量庞大的 MixSTE [44]（仅 3 帧输入就需要 3420 MFLOPs）未被纳入比较。Pose-FormerV2 相较于 PoseFormerV1 的提升见图 1。

表 2. 在 Human3.6M 数据集上与此前基于 Transformer 的方法的定量比较（单位：mm）。$f \colon$ 输入模型的帧数，Seq. Len.：整个输入序列的长度（即有效感受野）。最佳结果以粗体标注。(\*) 表示使用了额外的预训练阶段，(†) 表示为本论文的重新实现。

| Method（方法） | $f$ | Seq.Len.（序列长度） | MFLOPs | MPJPE↓/P-MPJPE↓ |
|---|---|---|---|---|
| PoseFormerV1 [46] ICCV'21 | 27 | 27 | 542.1 | **47.0**/- |
| StridedTrans. [15] TMM'22 | 81 | 81 | 342.5 | 47.5/- |
| MixSTE [44](†) CVPR'22 | 39 | 39 | 3420 | 49.6/38.9 |
| MHFormer [16] CVPR'22 | 27 | 27 | 342.9 | 47.8/- |
| MHFormer [16] CVPR'22 | 27 | 81 | 434.9 | 45.9/- |
| P-STMO [32](*) ECCV'22 | 81 | 81 | 1634 | 46.8/- |
| P-STMO [32](*) ECCV'22 | 81 | 81 | 9354 | 45.6/- |
| Einfalt et al. [10] WACV'23 | 9 | 81 | 34 | 47.9/- |
| PoseFormerV2 | 1 | 9 | 77.2 | 49.9/38.7 |
| PoseFormerV2 | 1 | 27 | 77.2 | 48.7/37.8 |
| PoseFormerV2 | 1 | 81 | 77.2 | 47.6/37.3 |
| PoseFormerV2 | 3 | 9 | 117.3 | 49.5/38.5 |
| PoseFormerV2 | 3 | 27 | 117.3 | 47.9/37.4 |
| PoseFormerV2 | 3 | 81 | 117.3 | 47.1/37.3 |
| PoseFormerV2 | 9 | 27 | 351.7 | 47.6/37.1 |
| PoseFormerV2 | 9 | 81 | 351.7 | 46.0/36.1 |
| PoseFormerV2 | 27 | 243 | 1054.8 | **45.2**/**35.6** |

![](images/e6f83f756239388cfcde477e74fcac1b5aa23d88eac452c18078b0b126f5a212.jpg)  
图 5. PoseFormerV2 与其他当前最先进的基于 Transformer 的方法在 Human3.6M 数据集上的比较（单位：mm）。RF：感受野，$k { \times } \mathrm { R F }$ 表示 PoseFormerV2 的感受野通过完整序列的少量低频 DCT 系数扩展了 $k \times$。所提出的方法在速度-精度权衡方面优于其他方法，且 $k$ 越大，相对于其他方法的提升也越大。（建议在彩色下查看）

![](images/d960e8f3a09ecce37349bb023ede13022d7fe8d6d5e7a39c19ae0f1c0eaa72bf.jpg)  
图 6. PoseFormerV2 与其他基于 Transformer 的方法 [16, 44, 46] 在 Human3.6M 上对噪声鲁棒性的比较。将标准差为 sigma 的零均值高斯噪声加到真值 2D 检测上，并展示它们的性能下降（△MPJPE，单位 mm）随 sigma 增加的变化。标记的大小表示模型的计算成本。

表 3. 在 MPI-INF-3DHP 上与之前方法的定量比较。T：整个序列长度，默认为 1。最佳得分以粗体标出。(\*) 表示使用了额外的预训练阶段，(†) 表示我们的重新实现。

| 方法 | | PCK↑ AUC ↑ MPJPE↓ |
| --- | --- | --- |
| Mehta et al. [23] | 3DV'17 | 75.7  39.3  117.6 |
| Mehta et al. [24] | ACM ToG'17 | 76.6  40.4  124.7 |
| Pavllo et al. [29] (T=81) | CVPR'19 | 86.0  51.9  84.0 |
| Pavllo et al. [29] (T=243) | CVPR'19 | 85.5  51.5  84.8 |
| Lin et al. [17] (T=25) | CVPR'19 | 83.6  51.4  79.8 |
| Li et al. [14] | BMVC'19 | 81.2  46.1  99.7 |
| Chen et al. [5] (T=81) | CVPR'20 | 87.9  54.0  78.8 |
| PoseFormerV1 [46] (T=9)(†) | TCSVT'21 | 95.4  63.2  57.7 |
| MHFormer [16] (T=9) | ICCV'21 | 93.8  63.3  58.0 |
| MixSTE [44] (T=27) | CVPR'22 | 94.4  66.5  54.9 |
| P-STMO [32] (T=81)(*) | CVPR'22 ECCV'22 | **97.9**  75.8  32.2 |
| PoseFormerV2 (T=81) | | **97.9**  **78.8**  **27.8** |

为了证明引入低频 DCT（离散余弦变换）系数有助于提高所提方法的鲁棒性，我们通过向 Human3.6M 数据集 [13]（图 6）上真值 2D 检测结果添加零均值高斯噪声，使基于提升（lifting）策略的姿态估计任务更具挑战性。为了保证公平比较，我们对所有方法保持相同的输入序列长度（此处为 27 帧）。对于我们的方法，$f = n = 3$。实验结果表明，随着高斯噪声标准差（$\sigma$）的增大，PoseFormerV2 的性能下降更少，同时效率更高。我们观察到，当 $\sigma$ 从 8 增加到 10 时，PoseFormerV1 的性能急剧下降。相比之下，所提方法呈现出更稳定的趋势。此外，我们的方法甚至优于将 2D 检测器的不确定性纳入模型设计的 MHFormer [16]。有趣的是，我们发现较小的噪声可能会提高 3D 姿态估计的精度（$\sigma = 3$）。

表 4. 对 PoseFormerV1 若干修改的消融研究。我们展示了如何将 9 帧的 PoseFormerV1 逐步转换为 PoseFormerV2（使用来自 81 帧序列的 9 个 DCT 系数）。评估在 Human3.6M（协议 1，单位 mm）上进行。RF 表示感受野（Receptive Field）。
<table><tr><td>步骤</td><td>描述</td><td>RF</td><td>MPJPE↓</td></tr><tr><td>(0)</td><td>原始 9 帧 PoseFormerV1。</td><td>9</td><td>49.9</td></tr><tr><td>(1)</td><td>从更长的序列中采样帧。</td><td>9</td><td>49.9</td></tr><tr><td>(2)</td><td>附加 DCT 系数的嵌入。</td><td>81</td><td>47.1 (2.8↓)</td></tr><tr><td>(3)</td><td>将普通 MLP 替换为 FreqMLP。</td><td>81</td><td>46.0 (3.9↓)</td></tr></table>

表 5. 关于 PoseFormerV2 输入的帧数与 DCT 系数数量的消融研究。评估在 Human3.6M（协议 1，单位 mm）上进行。
<table><tr><td>帧数 (f)</td><td>系数数量 (n)</td><td>完整长度</td><td>MFLOPs</td><td>MPJPE</td></tr><tr><td>1</td><td>1</td><td>27</td><td>39.2</td><td>51.1</td></tr><tr><td>1</td><td>3</td><td>27</td><td>77.2</td><td>48.7 (2.4↓)</td></tr><tr><td>3</td><td>1</td><td>27</td><td>79.4</td><td>50.1 (1.0↓)</td></tr><tr><td>3</td><td>3</td><td>27</td><td>117.3</td><td>47.9 (3.2↓)</td></tr><tr><td>9</td><td>9</td><td>27</td><td>351.7</td><td>47.6 (3.5↓)</td></tr></table>

MPI-INF-3DHP。我们还在 MPI-INF-3DHP [23]（表 3）上将我们的方法与其他方法进行了比较。我们使用来自 81 帧输入序列的 9 个中间帧及其前 9 个 DCT 系数。所提方法优于包括采用了掩码关节预训练的 P-STMO [32] 在内的其他方法。该结果验证了我们方法的有效性。我们的实现遵循文献 [32]。

定性比较。我们在图 7 中展示了我们的方法与具有竞争力的 MHFormer [16] 和 PoseFormerV1 [46] 的定性比较。所有方法均使用 81 帧的 2D 关节序列作为输入。为了进一步说明我们方法的鲁棒性，我们通过向随机选择的关节（例如"左手腕"、"右脚"）的连续 2D 检测添加高斯噪声来增加姿态估计任务的难度。所提出的方法即使在 2D 检测显著偏离（由浅黄色箭头指示）的情况下也能获得可靠的 3D 人体姿态。注意，我们的模型比 MHFormer 高效约 9 倍（3.12 GFLOPs vs. 0.35 GFLOPs），比 PoseFormerV1 高效约 4 倍（1.36 GFLOPs vs. 0.35 GFLOPs）。

## 4.4. 消融研究

在本节中，我们展示了如何通过对 PoseFormerV1 的一些修改以循序渐进的方式带来显著的性能提升。此外，为了更深入地了解输入序列的频域表示，我们揭示了输入帧数量和保留的 DCT 系数数量对我们方法的影响。

将 PoseFormerV1 转换为 PoseFormerV2。我们继承了 PoseFormerV1 的整体时空架构，并对其时间 Transformer 进行了克制的修改，以实现更好的多域特征融合。作为示例，我们展示了如何将 9 帧的 PoseFormerV1 逐步转换为 PoseFormerV2：（1）输入（即 9 帧）从更长序列（例如 81 帧）的序列中心采样。由于模型的实际输入并未改变，此步骤不会带来性能提升或感受野的增加。（2）PoseFormerV1 空间编码器的输出 z<sup>T</sup> <sup>ime</sup> 被附加到完整序列（本例中为 81 帧）的前 n 个 DCT 系数的嵌入（记为 z<sup>F</sup> <sup>req</sup>）之后，作为时间编码器的输入。为方便起见，我们将 n 设为 9。（3）我们将时间编码器中用于 z<sup>T</sup> <sup>ime</sup> 的 vanilla MLP（在替换之前，z<sup>T</sup> <sup>ime</sup> 和 z<sup>F</sup> <sup>req</sup> 已分别使用各自的 vanilla MLP）替换为 FreqMLP（详见第 3.2.2 节）。经过这些步骤后，PoseFormerV1 被转换为 PoseFormerV2，并具有更大的感受野（从 9 扩展到 81）。我们在表 4 中展示了每个步骤带来的改进。值得注意的是，通过引入来自更长序列（即 81 帧）的 9 个 DCT 系数，9 帧 PoseFormerV1 的 MPJPE 降低了 7.8%（49.9mm vs. 46.0mm），这验证了所提出的输入关节序列 DCT 表示的有效性。

Figure 7. PoseFormerV2 与 MHFormer [16] 和 PoseFormerV1 [46] 的定性对比。我们在特定关节的 2D 检测上随机加入高斯噪声。我们用浅黄色箭头标出偏差较大的 2D 检测，用橙色箭头标出相应的 3D 姿态估计。PoseFormerV2 在面对高噪声输入时比现有方法展现出更强的鲁棒性。

输入帧数与 DCT 系数个数。在表 5 中，我们研究了输入到空间编码器的帧数（f）以及保留的 DCT 系数个数（n）的影响。这里我们保持整个关节序列的长度固定，即 27 帧。基线模型仅使用一个中心帧和一个 DCT 系数（$f = n = 1$）。同时增大这两个参数会带来持续的性能提升，并且 n 的增大带来更大的误差降低（n = 3 时下降 2.4，而 f = 3 时仅下降 1.0），因为仅需少数 DCT 系数即可捕获整个序列的全局特征。我们通过实验发现，以 9 倍扩展率匹配的 f 和 n（即 f = n = 3）能够在速度与精度之间取得令人满意的折衷。

## 4.5. 泛化能力

所提出的频域方法可以泛化到其他方法，例如 MixSTE [44] 和 MHFormer [16]，因为它们同样使用 Transformer 进行时序建模。我们通过引入低频 DCT 系数来改进这两种方法。详情见补充材料。

## 5. 结论

我们从几乎未被探索的频域视角出发，针对基于提升方法的 3D 人体姿态估计中两个看似不相干甚至冲突的问题——长序列输入的处理效率与对噪声关节检测的鲁棒性——给出了一个统一的处理方案。该方法 PoseFormerV2 利用长 2D 关节序列的紧凑频域表示，在高效扩大模型感受野的同时提升其鲁棒性。实验结果表明，我们的方法在 Human3.6M 和 MPI-INF-3DHP 数据集上优于以往基于 Transformer 的方法。

## 补充材料

## A. 概述

补充材料包含以下章节：

- 章节 B：离散余弦变换的形式化介绍。

- 章节 C：数据集与评价指标。

- 章节 D：更多实现细节。

- 章节 E：PoseFormerV2 与一个纯频域简单基线模型的对比。

- 章节 F：所提方法到更多模型的泛化。

- 章节 G：可视化与分析。

- 章节 H：更广泛的影响与局限性。

## B. 离散余弦变换

现在我们对 DCT 进行正式介绍。给定一个二维关节点序列，记为 $\mathbf { x } \in \mathbb { R } ^ { F \times J \times 2 }$，其中 $F$ 为序列长度，$J$ 为每帧中的关节点数量。第 $j$ 个关节点的 x（或 y）坐标轨迹记为 $\mathbf { x } _ { j , 0 } \in \mathbb { R } ^ { F }$（或 $\mathbf { x } _ { j , 1 } \in \mathbb { R } ^ { F }$，为方便起见统一记为 $\hat { \mathbf { x } } _ { j }$），是一个一维时间序列，我们对每条轨迹（共 $J * 2$ 条轨迹）单独施加 DCT。

对于轨迹 $\hat { \mathbf { x } } _ { j }$，第 $i$ 个 DCT 系数计算如下：

$$
\begin{array} { r } { C _ { j , i } = \sqrt { \frac { 2 } { F } } \sum _ { f = 1 } ^ { F } x _ { j , f } \frac { 1 } { \sqrt { 1 + \delta _ { i 1 } } } \cos \left( \frac { \pi } { 2 F } ( 2 f - 1 ) ( i - 1 ) \right) } \end{array}\tag{5}
$$

其中当 $i = 1$ 时 $\delta _ { i 1 } = 1$，否则 $\delta _ { i 1 } = 0$。轨迹中的每个时间步产生一个 DCT 系数，即 $i \in \{ 1 , 2 , \cdots , F \}$。DCT 系数对输入时间序列中的多层级时间信息进行编码。具体而言，低频系数（即 $i$ 较小时）编码了输入序列的整体轮廓，而高频系数（即 $i$ 较大时）编码了细节信息，例如输入序列中的抖动或剧烈变化。可以使用逆离散余弦变换（Inverse Discrete Cosine Transform, IDCT）将时域中的原始输入序列恢复出来，公式如下：

$$
\begin{array} { r } { x _ { j , f } = \sqrt { \frac { 2 } { F } } \sum _ { i = 1 } ^ { F } C _ { j , i } \frac { 1 } { \sqrt { 1 + \delta _ { i 1 } } } \cos \left( \frac { \pi } { 2 F } ( 2 f - 1 ) ( i - 1 ) \right) } \end{array}\tag{6}
$$

其中 $f \in \{ 1 , 2 , \cdots , F \}$。如果我们保留 DCT 的所有系数，则 DCT 是无损的。实际应用中，我们仅使用少数低频系数即可对输入序列进行近似恢复（存在轻微的有损），而将其余系数置零。值得注意的是，由于我们丢弃了部分高频系数，恢复出的曲线相比原始曲线会更加平滑。 DCT 的这一性质是非常理想的——仅需少量 DCT 系数即可（以更清晰的方式）表示整个输入序列。这促使我们采用这种表示方式来高效处理长序列，同时提升模型对低质量二维检测的鲁棒性，而低质量检测中往往会出现高频噪声。

## C. 数据集与评估指标

Human3.6M 是三维人体姿态估计中使用最广泛的基准数据集。该数据集采集了 4 个不同地点摄像机的超过 360 万帧室内视频帧，包含 11 位受试者执行 15 种不同动作，例如"行走"（Walking）和"打电话"（Phoning）。我们遵循文献 [5, 18, 29, 46] 的做法，在 5 位受试者（S1, S5, S6, S7, S8）上训练模型，并在另外 2 位受试者（S9, S11）上进行测试。

MPI-INF-3DHP 数据集采集于受控的室内环境和具有挑战性的室外环境，包含与 Human3.6M 类似的来自多摄像机视角的不同受试者和动作。

**评估指标。** 我们在 Human3.6M 上报告两个常用指标 MPJPE 和 P-MPJPE [45]。MPJPE（Mean Per Joint Position Error，平均关节位置误差，称为 Protocol 1）度量估计的三维姿态与真实三维姿态之间的平均欧几里得距离。P-MPJPE（Protocol 2）对估计的三维姿态施加刚性变换，然后计算对齐后的估计三维姿态与真实三维姿态之间的距离。

对于 MPI-INF-3DHP 数据集，我们按照文献 [5, 17, 37] 的方法报告 MPJPE、150 mm 范围内的正确关键点百分比（Percentage of Correct Keypoint, PCK）以及曲线下面积（Area Under Curve, AUC）。

## D. 更多实现细节

我们的方法基于 PoseFormerV1 [46] 构建。为了更好地展示输入序列的 DCT 系数表示的有效性，并与 PoseFormerV1 进行公平比较，我们直接采用来自 PoseFormerV1 的最优模型架构超参数，尽管进一步的探索可能带来额外的性能提升。

**模型架构超参数。** 空间变换器（spatial transformer）中的嵌入特征维度 $c$ 为 32，空间变换器和特征融合变换器（feature-fusion transformer）的层数为 4，均遵循文献 [46]。此外，时空位置嵌入（Spatial-Temporal Positional Embedding）的设计也沿用自文献 [46]。

**实验设置。** 我们的实验使用 PyTorch [27] 在单块 NVIDIA RTX 3090 上进行。在训练和测试阶段，我们遵循文献 [5, 18, 29, 46] 应用水平翻转数据增强。我们使用 AdamW [20] 优化器训练 80 个 epoch，权重衰减为 0.1。初始学习率设为 $8 \times 10^{-4}$，采用指数学习率衰减调度，衰减因子为 0.99。在 Human3.6M 上，我们遵循文献 [5, 18, 29] 使用 CPN [7] 进行二维姿态检测。对于 MPI-INF-3DHP 数据集，我们遵循文献 [17, 24] 使用真实二维检测结果。

## E. 简单基线

在我们提出的方法中，PoseFormerV1 [46] 的时间编码器被重构为时频特征融合模块，我们证明输入序列的低频系数有助于提升模型处理长序列的效率，以及对关节检测噪声的鲁棒性。鉴于这种表示的有效性，读者可能会提出一个问题：为什么不直接从输入序列的 DCT 系数中提取特征，而是将它们与时域中的特征进一步结合？为此，我们设计了一个基线模型，在该模型中，我们简单地将 PoseFormerV1 [46] 的输入（时域中的关节坐标）替换为输入序列的低频 DCT 系数。基线模型和我们方法中的完整序列长度与保留的 DCT 系数个数（记为 n）保持一致。为方便起见，输入到 PoseFormerV2 空间编码器的帧数（f）设置为 n。我们提供了定量结果表明，这种直接的方法效果不佳，特别是当完整序列长度与 n 的比值增大时（见表 6）。仅使用序列中少数几个中心帧的特征就能显著提升精度，$e.g.$，在长度为 81 的完整输入序列中，使用 3 个中心帧时，MPJPE 从 49.7mm 降低到 47.1mm（降低 5.2%，见表 6 第 3 行）。

表 6. PoseFormerV2 与简单基线模型的对比。评估在 Human3.6M（协议 1，MPJPE）[13] 上进行，帧数（f）仅适用于 PoseFormerV2。

<table><tr><td>帧数 (f)</td><td>系数个数 (n)</td><td>完整长度</td><td>基线</td><td>PoseFormerV2</td></tr><tr><td>3</td><td>3</td><td>9</td><td>50.2</td><td>49.5 (↓0.7)</td></tr><tr><td>3</td><td>3</td><td>27</td><td>48.7</td><td>47.9 (↓0.8)</td></tr><tr><td>3</td><td>3</td><td>81</td><td>49.7</td><td>47.1 (↓2.6)</td></tr><tr><td>9</td><td>9</td><td>27</td><td>48.8</td><td>47.6 (↓1.2)</td></tr><tr><td>9</td><td>9</td><td>81</td><td>47.8</td><td>46.0 (↓1.8)</td></tr></table>

直观上，PoseFormerV2 的空间编码器对时域中少数中心帧的关节坐标进行编码，有助于捕获细粒度的人体运动，从而有利于对序列中心帧进行三维姿态估计。相比之下，输入序列的低频系数会过滤掉高频噪声和人体运动细节 $(e.g.$，快速运动），而这些细节对于人体姿态估计可能是具有信息量的（即过平滑问题）。因此，来自时域和频域的特征，即中心帧的关节坐标与序列的低频系数，承载了互补的语义。这些考虑因素使得我们有必要提出时频特征融合设计。

## F. 推广到更多模型

在正文中，我们专注于从一个鲜有探索的频域视角改进 PoseFormerV1 [46]。在这一部分，我们展示了所提出的频域方法也能很好地推广到其他现有的最先进方法，例如 MixSTE [44] 和 MHFormer [16]。由于这些方法 [16, 44] 也像 PoseFormerV1 [46] 一样沿时间维度对所有帧应用自注意力机制，因此所提出的方法可以轻松地融入它们的模型中，无需对模型架构进行复杂的重新设计。为了进行公平的比较，我们直接采用这些原始方法的最优超参数（例如，层数、通道维度）。进一步调整超参数可能会带来额外的提升。

![](images/83d2f5dd387d4a4117380c10d2d1507256539fa10397d5e70a0781864d115066.jpg)  
图 8. 在 Human3.6M [13] 上 MixSTE [44] 及其使用序列频域表示的改进版本之间的比较。RF：感受野（Receptive Field），k×RF 表示 MixSTE 的感受野通过完整序列的少量低频 DCT（离散余弦变换）系数扩展了 k 倍。所提出的方法有助于 MixSTE 在速度与精度之间获得更好的权衡。（建议在彩色下查看）

![](images/1bb4c0cffec9d4ad83397ca30b1d4b6313841fece6c81d10de2381c399f95d81.jpg)  
图 9. 在 Human3.6M [13] 上 MixSTE [44] 及其使用序列低频 DCT 系数的改进版本在抗噪声鲁棒性方面的比较。将标准差为 $\sigma$ 的零均值高斯噪声添加到真实 2D 检测中，我们展示了随 $\sigma$ 增大时其性能下降情况（△MPJPE，即平均每关节位置误差的增量）。标记的大小表示模型的计算成本。

MixSTE [44] 采用了与 PoseFormerV1 [46] 相同的时空架构。与 PoseFormerV1 的"先空间后时间"范式不同，MixSTE 交替使用空间和时间 Transformer 编码器。类似地，我们从一个较长的视频序列中中心采样少量视频帧，输入到 MixSTE 的空间编码器中。对于时间编码器，我们将完整输入序列的低频系数嵌入拼接到时域特征（即空间编码器的输出）上。MixSTE 及其改进版本的对比见图 8 和图 9。原始的 MixSTE 计算开销极大，而我们的方法同时提升了其效率和精度，例如 MixSTE 达到 46.2mm MPJPE 需 30.8 GFLOPs，而其改进版本以 15.4 GFLOPs 达到 45.3mm MPJPE（速度提升 2 倍且误差降低 1.9%，见图 8 中的亮红色曲线）。我们还展示了我们的方法提升了 MixSTE 对含噪 2D 关节点检测的鲁棒性（图 9）。具体而言，我们在 Human3.6M [13] 上对 27 帧的真实 2D 关节点序列添加零均值高斯噪声。改进的 MixSTE（记为 MixSTE+）使用 3 个中心帧作为其空间编码器的输入，并将前 3 个 DCT 系数作为更干净的完整序列全局表示。MixSTE+ 在性能下降更少的同时，效率提升了 6 倍（30.8 GFLOPs vs. 5.1 GFLOPs，以标记大小表示）。

![](images/f734e1d043bbb1c8e2a74a59a37952250f04eaa7948dcad212432740081b1040.jpg)  
图 10. MHFormer [16] 及其使用序列低频 DCT 系数的改进版本在 Human3.6M [13] 上针对噪声鲁棒性的对比。实验设置沿用图 9。标记的大小表示模型的计算开销。

MHFormer [16] 将多假设引入其架构，以建模身体部位的深度模糊性和关节点检测器的不确定性，因此具有相对较好的鲁棒性（实验结果见主论文）。此外，MHFormer 也包含与 PoseFormerV1 [46] 相同的时空 Transformer 模块。为了进一步验证我们方法的普适性，我们按照 MixSTE+ 的方式类似地改进 MHFormer。实验证据表明，所提出的方法在提升 MHFormer 鲁棒性的同时降低了其计算开销（见图 10），即使 MHFormer 已经具备了对含噪关节点检测的先验知识。因此，这一结果表明，在提升模型鲁棒性方面，我们的方法与其他方法是兼容的。

We have so far generalized our approach to other two transformer-based methods, i.e., MixSTE [44] and MH-Former [16]. We may explore the generalization of the proposed method to a wider range of model architectures in the future, such as CNN-based and GNN-based methods in 3D human pose estimation. Moreover, we believe our method can also be utilized in other tasks, especially skeleton-based ones where the computational cost of long-sequence processing and the quality of human skeleton representations can become problems.

## G. Visualizations and Analysis

In this section, we provide a series of qualitative results on challenging in-the-wild images to showcase the robustness of PoseFormerV2 in real scenarios.

Fig. 11 presents several representative hard cases with HRNet [34] 2D joint detection: (a) Occlusions where joints overlap with each other; (b)(c) Missed joints; (d) Switched joints. Specifically, the right arm of the person in the 4th image of (b) and the left arm of the person in the 3rd image of (c) are missed. Moreover, in the 2nd image of (d), two legs of the person are switched (highlighted with lightyellow circles). Despite the imperfect 2D joint input, Pose-FormerV2 still infers correct positions for these joints in 3D space (marked with orange circles).

Analysis. The robustness of PoseFormerV2 is attributed to the usage of an appropriate representation – lowfrequency DCT coefficients – of the input joint sequence, instead of hand-crafted modules that may bring additional computational cost such as the multi-hypothesis generation module in [16]. Low-frequency DCT coefficients provide a global vision of the input sequence and therefore the noise contained in individual video frames is dwarfed. This utilization of DCT coefficients also brings an extra advantage to PoseFormerV2, the temporal consistency of the estimated 3D pose between adjacent frames. We provide a video demo to illustrate that the proposed method keeps an excellent consistency (i.e., temporal stability) under extremely corrupted 2D joint detection.

## H. Broader Impacts and Limitations

Broader impacts. In this paper, we attempt to reconcile two critical issues in real-scenario applications of 3D HPE, i.e., the efficiency of models to process long sequences for improved precision and their robustness against noisy 2D detection as high-quality joint sequences are hard to obtain. To encourage more real-world applications, we may shift our research focus from marginal improvements on carefully controlled datasets to overcoming the drawbacks of existing approaches in practical use. We expect more research to follow this line.

![](images/182193826ce1c603545cc4320658a9369dadadde3e0ee64364dbc6bb19686a97.jpg)

![](images/e15afc600afae65f3fd975e0cc2bf11caded96344cd15b71a8f2ae0c39276707.jpg)

![](images/16489bac2c530db24cf2389623257d73077f258dbfc9d15692ba505ba31d9853.jpg)

![](images/234b8b05a13cfddd407d4bbe467c31636a62e66c5b194b2dd814ba9723e01244.jpg)  
(a)

(b)

![](images/e49ec90610852bba009ef2d6fb504793637ec3a8d68191885f6bb7261bac3e71.jpg)  
(c)  
(d)  
图 11. PoseFormerV2 在具有挑战性的真实场景图像下的定性结果：(a) 遮挡；(b)(c) 漏检的 2D 关节点；(d) 错位的 2D 关节点。我们用浅黄色圆圈突出显示不可靠的 2D 检测，并用橙色圆圈突出显示对应的 3D 姿态估计。PoseFormerV2 对不完美的 2D 关节点检测表现出很强的鲁棒性。

另一方面，本工作基于一种鲜有研究的频域方法——离散余弦变换（DCT），该方法在传统图像压缩算法中发挥着重要作用。我们希望这项研究能够启发更多研究重新审视传统信号处理技术，因为在深度学习时代我们处理的各种数据实际上都是不同形式的信号。这些精心设计的技术与最新深度学习方法的有效结合可能会带来意想不到的优势。

局限性。我们的方法包含两个重要的超参数——采样的中心帧数量以及完整输入序列保留的 DCT 系数数量。目前，它们是根据实验结果或人工经验在速度与精度之间权衡选择的。未来，我们可以将它们重塑为可从输入数据中自动学习的可学习参数，或者进一步从理论上推导出它们的最优选择，从而消除参数搜索的需要。

## 参考文献

[1] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Luciˇ c, and Cordelia Schmid. Vivit: A video vi-´ sion transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 6836– 6846, October 2021. 1

[2] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In Proceedings of the International Conference on Machine Learning (ICML), July 2021. 1

[3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 1

[4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 1

[5] Tianlang Chen, Chen Fang, Xiaohui Shen, Yiheng Zhu, Zhili Chen, and Jiebo Luo. Anatomy-aware 3d human pose estimation with bone-based pose decomposition. IEEE Transactions on Circuits and Systems for Video Technology, 2021. 1, 7, 9

[6] Yilun Chen, Zhicheng Wang, Yuxiang Peng, Zhiqiang Zhang, Gang Yu, and Jian Sun. Cascaded pyramid network for multi-person pose estimation. In CVPR, 2018. 1, 2, 4

[7] Yilun Chen, Zhicheng Wang, Yuxiang Peng, Zhiqiang Zhang, Gang Yu, Jian Sun. Cascaded pyramid network for multi-person pose estimation. 载于 IEEE 计算机视觉与模式识别会议论文集，第 7103–7112 页，2018 年。9

[8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv 预印本 arXiv:1810.04805，2018 年。1

[9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR，2021 年。1, 4

[10] Moritz Einfalt, Katja Ludwig, Rainer Lienhart. Uplift and upsample: Efficient 3d human pose estimation with uplifting transformers. 载于 IEEE/CVF 冬季计算机视觉应用会议 (WACV) 论文集，2023 年 1 月。2, 3, 6

[11] John Guibas, Morteza Mardani, Zongyi Li, Andrew Tao, Anima Anandkumar, Bryan Catanzaro. Adaptive fourier neural operators: Efficient token mixers for transformers. arXiv 预印本 arXiv:2111.13587，2021 年。3

[12] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, Ross Girshick. Masked autoencoders are scalable´ vision learners. 载于 IEEE/CVF 计算机视觉与模式识别会议论文集，第 16000–16009 页，2022 年。3

[13] C. Ionescu, D. Papava, V. Olaru, C. Sminchisescu. Human3.6m: Large scale datasets and predictive methods for 3d human sensing in natural environments. IEEE 模式分析与机器智能汇刊，2014 年。1, 2, 3, 4, 6, 7, 10, 11

[14] Shichao Li, Lei Ke, Kevin Pratama, Yu-Wing Tai, Chi-Keung Tang, Kwang-Ting Cheng. Cascaded deep monocular 3d human pose estimation with evolutionary training data. 载于 IEEE/CVF 计算机视觉与模式识别会议 (CVPR) 论文集，2020 年 6 月。7

[15] Wenhao Li, Hong Liu, Runwei Ding, Mengyuan Liu, Pichao Wang, Wenming Yang. Exploiting temporal contexts with strided transformer for 3d human pose estimation. IEEE 多媒体汇刊，2022 年。2, 3, 6

[16] Wenhao Li, Hong Liu, Hao Tang, Pichao Wang, Luc Van Gool. Mhformer: Multi-hypothesis transformer for 3d human pose estimation. 载于 IEEE/CVF 计算机视觉与模式识别会议 (CVPR) 论文集，第 13147–13156 页，2022 年 6 月。2, 3, 6, 7, 8, 10, 11

[17] Jiahao Lin, Gim Hee Lee. Trajectory space factorization for deep video-based 3d human pose estimation. 载于 BMVC，2019 年。7, 9

[18] Ruixu Liu, Ju Shen, He Wang, Chen Chen, Sen-ching Cheung, Vijayan Asari. Attention mechanism exploits temporal contexts: Real-time 3d human pose reconstruction. 载于 CVPR，2020 年。1, 9

[19] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. 载于

[20] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 9

[21] Wei Mao, Miaomiao Liu, and Mathieu Salzmann. History repeats itself: Human motion prediction via motion attention. In *European Conference on Computer Vision*, pages 474–489. Springer, 2020. 2, 3

[22] Wei Mao, Miaomiao Liu, Mathieu Salzmann, and Hongdong Li. Learning trajectory dependencies for human motion prediction. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 9489–9497, 2019. 2, 3

[23] Dushyant Mehta, Helge Rhodin, Dan Casas, Pascal Fua, Oleksandr Sotnychenko, Weipeng Xu, and Christian Theobalt. Monocular 3d human pose estimation in the wild using improved cnn supervision. In *2017 international conference on 3D vision (3DV)*, pages 506–516. IEEE, 2017. 3, 6, 7

[24] Dushyant Mehta, Srinath Sridhar, Oleksandr Sotnychenko, Helge Rhodin, Mohammad Shafiei, Hans-Peter Seidel, Weipeng Xu, Dan Casas, and Christian Theobalt. Vnect: Real-time 3d human pose estimation with a single rgb camera. *ACM Transactions on Graphics (TOG)*, 36(4):1–14, 2017. 7, 9

[25] Gyeongsik Moon and Kyoung Mu Lee. I2l-meshnet: Image-to-lixel prediction network for accurate 3d human pose and mesh estimation from a single rgb image. In *ECCV*, 2020. 1

[26] Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked hourglass networks for human pose estimation. In *European conference on computer vision*, pages 483–499. Springer, 2016. 1

[27] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017. 9

[28] Georgios Pavlakos, Xiaowei Zhou, and Kostas Daniilidis. Ordinal depth supervision for 3D human pose estimation. In *CVPR*, 2018. 1

[29] Dario Pavllo, Christoph Feichtenhofer, David Grangier, and Michael Auli. 3d human pose estimation in video with temporal convolutions and semi-supervised training. In *CVPR*, 2019. 7, 9

[30] William B Pennebaker and Joan L Mitchell. *JPEG: Still image data compression standard*. Springer Science & Business Media, 1992. 3

[31] Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, and Jie Zhou. Global filter networks for image classification. *Advances in Neural Information Processing Systems*, 34:980–993, 2021. 3

[32] Wenkang Shan, Zhenhua Liu, Xinfeng Zhang, Shanshe Wang, Siwei Ma, and Wen Gao. P-stmo: Pre-trained spatial temporal many-to-one model for 3d human pose estimation. arXiv preprint arXiv:2203.07628, 2022. 2, 3, 6, 7

[33] Athanassios Skodras, Charilaos Christopoulos, and Touradj Ebrahimi. The jpeg 2000 still image compression standard. *IEEE Signal processing magazine*, 18(5):36–58, 2001. 3

[34] Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. Deep high-resolution representation learning for human pose estimation. In *CVPR*, 2019. 11

[35] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. arXiv preprint arXiv:2012.12877, 2020. 1

[36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. 2017. 1

[37] Jingbo Wang, Sijie Yan, Yuanjun Xiong, and Dahua Lin. Motion guided 3d pose estimation from videos. In European Conference on Computer Vision, pages 764–780. Springer, 2020. 1, 9

[38] Zhenyu Wang, Hao Luo, Pichao WANG, Feng Ding, Fan Wang, and Hao Li. VTC-LFC: Vision transformer compression with low-frequency components. In Thirty-Sixth Conference on Neural Information Processing Systems, 2022. 2, 3

[39] Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan Yuille, and Christoph Feichtenhofer. Masked feature prediction for self-supervised visual pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14668–14678, 2022. 3

[40] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9653–9663, 2022. 3

[41] Kai Xu, Minghai Qin, Fei Sun, Yuhao Wang, Yen-Kuang Chen, and Fengbo Ren. Learning in the frequency domain. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020. 2, 3

[42] Shen Yan, Xuehan Xiong, Anurag Arnab, Zhichao Lu, Mi Zhang, Chen Sun, and Cordelia Schmid. Multiview transformers for video recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3333–3343, 2022. 1

[43] Ailing Zeng, Xiao Sun, Fuyang Huang, Minhao Liu, Qiang Xu, and Stephen Lin. Srnet: Improving generalization in 3d human pose estimation with a split-and-recombine approach. In ECCV, 2020. 1

[44] Jinlu Zhang, Zhigang Tu, Jianyu Yang, Yujin Chen, and Junsong Yuan. Mixste: Seq2seq mixed spatio-temporal encoder for 3d human pose estimation in video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13232–13242, 2022. 2, 3, 6, 7, 8, 10, 11

[45] Ce Zheng, Wenhan Wu, Taojiannan Yang, Sijie Zhu, Chen Chen, Ruixu Liu, Ju Shen, Nasser Kehtarnavaz, and Mubarak Shah. Deep learning-based human pose estimation: A survey, 2020. 9

[46] Ce Zheng, Sijie Zhu, Matias Mendieta, Taojiannan Yang, Chen Chen, and Zhengming Ding. 3d human pose estimation with spatial and temporal transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 11656–11665, October 2021. 1, 2, 3, 4, 6, 7, 8, 9, 10, 11

[47] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020. 1