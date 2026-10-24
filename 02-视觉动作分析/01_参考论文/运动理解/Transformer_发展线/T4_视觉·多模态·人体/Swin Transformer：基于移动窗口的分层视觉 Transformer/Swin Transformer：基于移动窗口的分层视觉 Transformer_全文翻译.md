# Swin Transformer：基于移动窗口的分层视觉 Transformer

Ze Liu<sup>†\*</sup> Yutong Lin<sup>†\*</sup> Yue Cao<sup>\*</sup> Han Hu<sup>\*‡</sup> Yixuan Wei<sup>†</sup>

Zheng Zhang Stephen Lin Baining Guo

Microsoft Research Asia

{v-zeliu1,v-yutlin,yuecao,hanhu,v-yixwe,zhez,stevelin,bainguo}@microsoft.com

## 摘要

本文提出了一种新型的视觉 Transformer，称为 Swin Transformer，它能够作为计算机视觉的通用骨干网络（backbone）。将 Transformer 从语言领域迁移到视觉领域所面临的挑战源于两个领域之间的差异，例如视觉实体尺度的巨大变化，以及与文本中的单词相比图像中像素的高分辨率。为了解决这些差异，我们提出了一种分层 Transformer，其表示是通过移动窗口（Shifted windows）计算的。移动窗口方案通过将自注意力（self-attention）计算限制在不重叠的局部窗口内，同时允许跨窗口连接，从而带来了更高的效率。这种分层架构具有在不同尺度下建模的灵活性，并且计算复杂度关于图像尺寸呈线性关系。Swin Transformer 的这些特性使其能够广泛适用于各种视觉任务，包括图像分类（在 ImageNet-1K 上达到 87.3 的 top-1 准确率）以及密集预测任务，如目标检测（在 COCO testdev 上达到 58.7 box AP 和 51.1 mask AP）和语义分割（在 ADE20K val 上达到 53.5 mIoU）。其性能大幅超越了先前的最先进技术（state-of-the-art），在 COCO 上提升了 +2.7 box AP 和 +2.6 mask AP，在 ADE20K 上提升了 +3.2 mIoU，展示了基于 Transformer 的模型作为视觉骨干网络的巨大潜力。分层设计和移动窗口方法也被证明对全 MLP（all-MLP）架构大有裨益。代码和模型已开源于 https://github.com/microsoft/Swin-Transformer 。

## 1. 引言

长期以来，计算机视觉中的建模一直由卷积神经网络（CNNs）主导。从 AlexNet [39] 及其在 ImageNet 图像分类挑战赛上的革命性表现开始，CNN 架构通过更大的规模 [30, 76]、更广泛的连接 [34] 以及更复杂的卷积形式 [70, 18, 84] 不断演进，变得日益强大。随着 CNN 成为各种视觉任务的骨干网络，这些架构上的进步带来了性能的提升，并在广泛范围内推动了整个领域的发展。

![](images/9d63efa9eab1255916e42050e70fb8e0da76646e3fd0fe6aeec6a3aa6c739094.jpg)  
(a) Swin Transformer（本文方法）  
Figure 1. (a) 所提出的 Swin Transformer 通过在更深层合并图像块（以灰色显示）来构建层次化特征图，并且由于仅在每个局部窗口（以红色显示）内计算自注意力，因而其计算复杂度与输入图像大小呈线性关系。因此，它可以作为图像分类和密集识别任务的通用主干网络。(b) 相比之下，先前的视觉 Transformer [20] 生成单一低分辨率的特征图，并且由于在全局范围内计算自注意力，其计算复杂度与输入图像大小呈二次方关系。

另一方面，自然语言处理（NLP）中网络架构的演进走了一条不同的道路，如今主流的架构则是 Transformer [64]。Transformer 专为序列建模和转换任务而设计，因其利用注意力机制对数据中的长距离依赖关系进行建模而著称。它在语言领域的巨大成功促使研究人员探索其在计算机视觉中的应用，近期它在某些任务上展现出了极具前景的结果，特别是图像分类 [20] 和视觉-语言联合建模 [47]。

在本文中，我们力求扩展 Transformer 的适用性，使其能够像在 NLP 领域以及 CNN 在视觉领域那样，作为计算机视觉的通用主干网络（general-purpose backbone）。我们观察到，将其在语言领域的高性能迁移到视觉领域所面临的重大挑战，可以用这两种模态之间的差异来解释。其中一个差异涉及尺度。与语言 Transformer 中作为基本处理单元的词 token 不同，视觉元素在尺度上可能存在巨大差异，这一问题在目标检测 [42, 53, 54] 等任务中备受关注。在现有的基于 Transformer 的模型 [64, 20] 中，token 均为固定尺度，这一特性并不适用于这些视觉应用。另一个差异在于，与文本段落中的词相比，图像中像素的分辨率要高得多。存在许多诸如语义分割等需要在像素级进行密集预测的视觉任务，这对于高分辨率图像上的 Transformer 来说是难以处理的，因为其自注意力机制的计算复杂度与图像大小呈二次方关系。为了克服这些问题，我们提出了一种名为 Swin Transformer 的通用 Transformer 主干网络，它构建了分层特征图（hierarchical feature maps），且计算复杂度与图像大小呈线性关系。如图 Figure 1(a) 所示，Swin Transformer 通过从小尺寸的 patch（灰色轮廓所示）开始，并在更深的 Transformer 层中逐步合并相邻 patch 来构建分层表示。借助这些分层特征图，Swin Transformer 模型可以方便地利用先进的密集预测技术，如特征金字塔网络（FPN）[42] 或 U-Net [51]。其线性计算复杂度是通过在划分图像的不重叠窗口（红色轮廓所示）内局部计算自注意力来实现的。由于每个窗口中的 patch 数量是固定的，因此复杂度与图像大小呈线性关系。与以往生成单一分辨率特征图且具有二次方复杂度的基于 Transformer 的架构 [20] 相比，这些优势使得 Swin Transformer 适合作为各种视觉任务的通用主干网络。

Swin Transformer 的一个关键设计要素是在连续的自注意力（self-attention）层之间移动窗口划分，如图 2 所示。移位窗口桥接了前一层的窗口，在它们之间提供了连接，从而显著增强了建模能力（见 Table 4）。该策略在实际延迟方面也是高效的：窗口内的所有查询 patch（query patch）共享相同的键集合<sup>1</sup>，这有利于硬件中的内存访问。相比之下，早期基于滑动窗口的自注意力方法 [33, 50] 由于不同查询像素对应不同的键集合<sup>2</sup>，在通用硬件上面临较高的延迟。我们的实验表明，所提出的移位窗口方法比滑动窗口方法的延迟低得多，但在建模能力上相当（见 Table 5 和 Table 6）。移位窗口方法也被证明对全 MLP（all-MLP）架构有益 [61]。

![](images/02619979390754b22771bff22043cdcdab4391449795c738c0cdeb1a7aa0373a.jpg)  
Figure 2. 所提出的 Swin Transformer 架构中用于计算自注意力的移位窗口方法示意图。在层 l（左）中，采用规则的窗口划分方案，并在每个窗口内计算自注意力。在下一层 l + 1（右）中，窗口划分发生移位，从而产生新的窗口。新窗口中的自注意力计算跨越了层 l 中先前窗口的边界，在它们之间提供了连接。

所提出的 Swin Transformer 在图像分类、目标检测和语义分割等识别任务上取得了强大的性能。在三个任务上以相近的延迟显著超越了 ViT / DeiT [20, 63] 和 ResNe(X)t 模型 [30, 70]。它在 COCO test-dev 集上取得了 58.7 box AP 和 51.1 mask AP，分别超越此前最先进的结果 +2.7 box AP（不使用外部数据的 Copy-paste [26]）和 +2.6 mask AP（DetectoRS [46]）。在 ADE20K 语义分割任务中，它在验证集上获得了 53.5 mIoU，相比此前最先进的结果（SETR [81]）提升了 +3.2 mIoU。它在 ImageNet-1K 图像分类上也取得了 87.3% 的 top-1 准确率。

我们坚信，贯穿计算机视觉和自然语言处理的统一架构将使两个领域共同受益，因为这将促进视觉和文本信号的联合建模，并使来自两个领域的建模知识能够更深入地共享。我们希望 Swin Transformer 在各种视觉问题上的强劲表现能够加深社区中的这一信念，并促进视觉和语言信号的统一建模。

## 2. 相关工作 (Related Work)

**CNN 及其变体** CNN 在整个计算机视觉领域充当标准网络模型。尽管 CNN 已经存在了几十年 [40]，但直到 AlexNet [39] 的提出，CNN 才得以迅速发展并成为主流。自那时起，人们提出了更深、更有效的卷积神经网络架构，以进一步推动计算机视觉中的深度学习浪潮，例如 VGG [52]、GoogleNet [57]、ResNet [30]、DenseNet [34]、

HRNet [65] 和 EfficientNet [58]。除了这些架构上的进展之外，还有许多关于改进单个卷积层的工作，例如深度卷积（depthwise convolution）[70] 和可变形卷积（deformable convolution）[18, 84]。虽然 CNN 及其变体仍然是计算机视觉应用的主要骨干架构，但我们强调类 Transformer 架构在视觉和语言统一建模方面的强大潜力。我们的工作在几个基础视觉识别任务上取得了优异的性能，我们希望这将有助于推动建模方式的转变。

**基于自注意力的骨干架构** 同样受到自注意力层和 Transformer 架构在 NLP 领域成功的启发，一些工作采用自注意力层来替代流行 ResNet 中的部分或全部空间卷积层 [33, 50, 80]。在这些工作中，自注意力是在每个像素的局部窗口内计算的，以加快优化速度 [33]，并且与对应的 ResNet 架构相比，它们取得了略好的精度/FLOPs 权衡。然而，其昂贵的内存访问导致其实际延迟显著大于卷积网络 [33]。与使用滑动窗口不同，我们提出在连续层之间移动窗口，这允许在通用硬件上实现更高效的计算。

**用于补充 CNN 的自注意力/Transformer** 另一类工作是通过自注意力层或 Transformer 来增强标准 CNN 架构。自注意力层可以通过提供编码长距离依赖或异构交互的能力，来补充骨干网络 [67, 7, 3, 71, 23, 74, 55] 或头部网络 [32, 27]。最近，Transformer 中的编码器-解码器设计已被应用于目标检测和实例分割任务 [8, 13, 85, 56]。我们的工作探索了 Transformer 在基础视觉特征提取中的应用，并且是对这些工作的补充。

基于 Transformer 的视觉骨干网络 与我们工作最相关的是 Vision Transformer (ViT) [20] 及其后续工作 [63, 72, 15, 28, 66]。ViT 的开创性工作直接将 Transformer 架构应用于互不重叠的中等尺寸图像块（patch）上进行图像分类。与卷积网络相比，它在图像分类上取得了令人瞩目的速度-精度权衡。尽管 ViT 需要大规模训练数据集（即 JFT-300M）才能表现出色，但 DeiT [63] 引入了几种训练策略，使得 ViT 在使用较小的 ImageNet-1K 数据集时也能保持有效。ViT 在图像分类上的成果令人鼓舞，但由于其低分辨率特征图以及计算复杂度随图像尺寸呈二次方增长，其架构不适合用作稠密视觉任务或输入图像分辨率较高时的通用骨干网络。有一些工作通过直接上采样或反卷积将 ViT 模型应用于目标检测和语义分割等稠密视觉任务，但性能相对较低 [2, 81]。与我们同时期的一些工作修改了 ViT 架构 [72, 15, 28] 以实现更好的图像分类。实验证明，尽管我们的工作侧重于通用性能而非专门针对分类，但我们的 Swin Transformer 架构在这些方法中实现了最佳的图像分类速度-精度权衡。另一项同时期的工作 [66] 探索了类似的思路，在 Transformer 上构建多分辨率特征图。其计算复杂度相对于图像尺寸仍呈二次方增长，而我们的复杂度是线性的，并且是在局部进行操作，这已被证明有利于对视觉信号中的高度相关性进行建模 [36, 25, 41]。我们的方法兼具高效性与有效性，在 COCO 目标检测和 ADE20K 语义分割上均取得了最先进的精度（state-of-the-art accuracy）。

## 3. Method

## 3.1. Overall Architecture

Figure 3 展示了 Swin Transformer 架构的概览，图中展示的是其微型版本（Swin-T）。它首先通过一个 patch 分割模块将输入的 RGB 图像分割成互不重叠的 patch，类似于 ViT。每个 patch 被视为一个“token”，其特征由原始像素的 RGB 值拼接而成。在我们的实现中，采用的 patch 大小为 $4 \times 4$，因此每个 patch 的特征维度为 $4 \times 4 \times 3 = 4 8$ 。在该原始数值特征上应用一个线性嵌入层，将其投影到任意维度（记为 C）。

随后，在这些 patch token 上应用了几个经过修改自注意力计算的 Transformer 块（Swin Transformer 块）。这些 Transformer 块保持了 token 的数量 $\begin{array} { r } { ( \frac { H } { 4 } \times \frac { W } { 4 } ) } \end{array}$ ，并与线性嵌入层一起被称为“Stage $1 ^ { \circ }$”。

为了生成分层表示（hierarchical representation），随着网络深度的增加，token 的数量通过 patch 合并层（patch merging layers）逐渐减少。第一个 patch 合并层将每组 $2 \times 2$ 个相邻 patch 的特征进行拼接，并在拼接后的 4C 维特征上应用一个线性层。这将 token 的数量减少为原来的 $2 \times 2 = 4$ 倍（分辨率下采样 $2 \times$），且输出维度设置为 $2 C$。随后应用 Swin Transformer 块进行特征变换，分辨率保持在 ${ \frac { H } { 8 } } \times { \frac { W } { 8 } }$。这第一个由 patch 合并和特征变换构成的组合被称为“Stage $2 ^ { \circ }$”。该过程重复两次，分别称为“Stage $3 ^ { \circ }$”和“Stage $4 ^ { \circ }$”，输出分辨率分别为 $\frac { H } { 1 6 } \times \frac { W } { 1 6 }$ 和 $\frac { H } { 3 2 } \times \frac { W } { 3 2 }$。这些阶段共同生成了分层表示，并具有与典型卷积网络（例如 VGG [52] 和 ResNet [30]）相同的特征图分辨率。因此，所提出的架构可以方便地替换各种视觉任务中现有方法的骨干网络（backbone networks）。

![](images/c29a9246214dec1ea2cedf034842053b505c0e0f719e214c7750a31b8a8dce2b.jpg)  
Figure 3. (a) Swin Transformer (Swin-T) 的架构；(b) 两个连续的 Swin Transformer 块（符号表示见公式 (3)）。W-MSA 和 SW-MSA 分别是具有常规和移动窗口配置的多头自注意力模块。

Swin Transformer 块 Swin Transformer 是通过将 Transformer 块中的标准多头自注意力（MSA）模块替换为基于移动窗口的模块（在第 3.2 节中描述）构建的，其他层保持不变。如图 3(b) 所示，一个 Swin Transformer 块由一个基于移动窗口的 MSA 模块组成，随后是一个中间带有 GELU 非线性的 2 层 MLP。在每个 MSA 模块和每个 MLP 之前应用 LayerNorm (LN) 层，并在每个模块之后应用残差连接。

## 3.2. 基于移动窗口的自注意力

标准的 Transformer 架构 [64] 及其用于图像分类的变体 [20] 都采用全局自注意力，即计算一个 token 与所有其他 token 之间的关系。全局计算会导致关于 token 数量的二次复杂度，使其不适用于许多需要大量 token 进行密集预测或表示高分辨率图像的视觉问题。

非重叠窗口中的自注意力 为了进行高效建模，我们提出在局部窗口内计算自注意力。这些窗口以非重叠的方式均匀划分图像。假设每个窗口包含 $M \times M$ 个 patch，对于包含 $h \times w$ 个 patch 的图像，全局 MSA 模块与基于窗口的模块的计算复杂度分别为 $\mathrm { a r e } ^ { 3 }$ ：

$$
\Omega (\mathrm{MSA}) = 4 h w C ^ {2} + 2 (h w) ^ {2} C,\tag{1}
$$

$$
\Omega (\mathbf {W - M S A}) = 4 h w C ^ {2} + 2 M ^ {2} h w C,\tag{2}
$$

其中前者与 patch 数量 $hw$ 呈二次方关系，而当 $M$ 固定时（默认设为 7），后者呈线性关系。对于较大的 $hw$，全局自注意力的计算通常是难以承受的，而基于窗口的自注意力则是可扩展的。

连续块中的移动窗口分区 基于窗口的自注意力模块缺乏跨窗口连接，这限制了其建模能力。为了在保持非重叠窗口高效计算的同时引入跨窗口连接，我们提出了一种移动窗口分区方法，该方法在连续的 Swin Transformer 块中交替使用两种分区配置。

如图 Figure 2 所示，第一个模块采用常规窗口分区策略，从左上角像素开始，将 $8 \times 8$ 的特征图均匀划分为大小为 $4 \times 4 ( M = 4 )$ 的 $2 \times 2$ 个窗口。接着，下一个模块采用从前一层移动而来的窗口配置，即从常规分区的窗口位移 $\begin{array} { r } { \big ( \big \lfloor \frac { M } { 2 } \big \rfloor , \big \lfloor \frac { M } { 2 } \big \rfloor \big ) } \end{array}$ 个像素。

采用移动窗口分区方法后，连续的 Swin Transformer 块的计算方式如下：

$$
\begin{array}{l} \hat {\mathbf {z}} ^ {l} = \mathrm{W-MSA} \left(\mathrm{LN} \left(\mathbf {z} ^ {l - 1}\right)\right) + \mathbf {z} ^ {l - 1}, \\ \mathbf {z} ^ {l} = \mathrm{MLP} \left(\mathrm{LN} \left(\hat {\mathbf {z}} ^ {l}\right)\right) + \hat {\mathbf {z}} ^ {l}, \\ \hat {\mathbf {z}} ^ {l + 1} = \mathrm{SW-MSA} \left(\mathrm{LN} \left(\mathbf {z} ^ {l}\right)\right) + \mathbf {z} ^ {l}, \\ \mathbf {z} ^ {l + 1} = \mathrm{MLP} \left(\mathrm{LN} \left(\hat {\mathbf {z}} ^ {l + 1}\right)\right) + \hat {\mathbf {z}} ^ {l + 1}, \end{array}\tag{3}
$$

其中 $\hat { \mathbf { z } } ^ { l }$ 和 $\mathbf { z } ^ { l }$ 分别表示块 $l$ 的 (S)W-MSA 模块和 MLP 模块的输出特征；

![](images/1a0153b71a5bcb291627acdb9f1f237f5fd6e1a969e861dcebb08cf349c942a0.jpg)  
Figure 4. 移动窗口分区中自注意力的高效批计算方法示意图。

W-MSA 和 SW-MSA 分别表示采用常规窗口分区配置和移动窗口分区配置的基于窗口的多头自注意力（Multi-Head Self-Attention）。

移动窗口分区方法在上一层相邻的非重叠窗口之间引入了连接，如 Table 4 所示，该方法在图像分类、目标检测和语义分割中被证明是有效的。

针对移动配置的高效批处理计算 移动窗口分区的一个问题是它会导致窗口数量增加，在移动配置下从 $\begin{array} { r } { \lceil \frac { \mathbf { \bar { h } } } { M } \rceil \times \lceil \frac { w } { M } \rceil ^ { \mathbf { \bar { l } } } \mathrm { t o } \left( \lceil \frac { h } { M } \rceil + 1 \right) \succ } \end{array}$ × $\left( \lceil \frac { w } { M } \rceil + 1 \right)$，并且其中一些窗口将小于 $M \times M ^ { 4 }$。一种朴素的解决方案是将较小的窗口填充到 $M \times M$ 的大小，并在计算注意力时将填充的值掩码（mask）掉。当规则分区中的窗口数量较少时，例如 $2 \times 2 .$，使用这种朴素方案增加的计算量是相当可观的（$2 \times 2  3 \times 3$，增加了 2.25 倍）。在此，我们提出了一种更高效的批处理计算方法，即向左上方向进行循环移位（cyclic-shifting），如 Figure 4 所示。完成移位后，一个批次窗口可能由特征图中不相邻的若干子窗口组成，因此采用掩码机制将自注意力计算限制在每个子窗口内部。通过循环移位，批次窗口的数量与规则窗口分区的数量保持一致，因而同样高效。该方法的低延迟特性如 Table 5 所示。

相对位置偏差 在计算自注意力时，我们遵循 [49, 1, 32, 33]，在计算相似度时为每个头引入相对位置偏差 $\boldsymbol { B } \in \mathbb { R } ^ { M ^ { 2 } \times M ^ { 2 } }$：

$$
\operatorname{Attention} (Q, K, V) = \operatorname{SoftMax} \left(Q K ^ {T} / \sqrt {d} + B\right) V,\tag{4}
$$

其中 $Q , K , V \in \mathbb { R } ^ { M ^ { 2 } \times d }$ 分别是查询（query）、键（key）和值（value）矩阵；$d$ 是查询/键的维度，$M ^ { 2 }$ 是窗口中 patch 的数量。由于沿每个轴的相对位置均在 $\left[ - M + 1 , M - 1 \right]$ 范围内，我们参数化了一个尺寸较小的偏差矩阵 $\hat { B } \in \mathbb { R } ^ { ( 2 M - 1 ) \times ( \hat { 2 M } - 1 ) }$，并且 $B$ 中的值取自 $\hat { B } .$

如 Table 4 所示，与不含该偏差项或使用绝对位置嵌入的对应结构相比，我们观察到了显著的提升。像 [20] 中那样在输入中进一步添加绝对位置嵌入会使性能略微下降，因此在我们的实现中未予采用。

在预训练中学到的相对位置偏差，还可以通过双三次插值（bi-cubic interpolation）[20, 63] 用于初始化在不同窗口大小下进行微调的模型。

## 3.3. 架构变体

我们构建的基础模型称为 Swin-B，其模型大小和计算复杂度与 ViT-B/DeiT-B 相似。我们还引入了 Swin-T、Swin-S 和 Swin-L，它们分别是模型大小和计算复杂度约为 0.25×、0.5× 和 2× 的版本。请注意，Swin-T 和 Swin-S 的复杂度分别与 ResNet-50 (DeiT-S) 和 ResNet-101 相似。在所有实验中，默认窗口大小设置为 $M = 7$。每个头的查询维度为 $d = 3 2$，每个 MLP 的扩展层为 $\alpha = 4$。这些模型变体的架构超参数为：

• Swin-T: C = 96, layer numbers = {2, 2, 6, 2}

• Swin-S: C = 96, layer numbers ={2, 2, 18, 2}

• Swin-B: C = 128, layer numbers ={2, 2, 18, 2}

$$
\bullet \text { Swin - L: } C = 1 9 2, \text { layer   numbers } = \{2, 2, 1 8, 2 \}
$$

其中 C 是第一阶段隐藏层的通道数。用于 ImageNet 图像分类的模型变体的模型大小、理论计算复杂度（FLOPs）和吞吐量列于 Table 1 中。

## 4. 实验

我们在 ImageNet-1K 图像分类 [19]、COCO 目标检测 [43] 和 ADE20K 语义分割 [83] 上进行了实验。在下文中，我们首先在这三项任务上将所提出的 Swin Transformer 架构与以往的最优方法进行比较。然后，我们对 Swin Transformer 的重要设计元素进行消融实验。

## 4.1. ImageNet-1K 图像分类

设置 对于图像分类，我们在 ImageNet-1K [19] 上对所提出的 Swin Transformer 进行基准测试，该数据集包含来自 1,000 个类别的 128 万张训练图像和 5 万张验证图像。报告了单裁剪（single-crop）下的 top-1 准确率。我们考虑了两种训练设置：

• 常规 ImageNet-1K 训练。该设置主要遵循 [63]。我们使用 AdamW [37] 优化器训练 300 个 epoch，采用余弦衰减学习率调度器和 20 个 epoch 的线性预热（warm-up）。使用 1024 的批大小（batch size）、0.001 的初始学习率和 0.05 的权重衰减。我们在训练中包含了 [63] 中的大部分数据增强和正则化策略，除了重复增强（repeated augmentation）[31] 和 EMA [45]，因为它们并未提升性能。需要注意的是，这与 [63] 相反，在 [63] 中重复增强对于稳定 ViT 的训练至关重要。

• 在 ImageNet-22K 上预训练并在 ImageNet-1K 上微调。我们还在更大的 ImageNet-22K 数据集上进行了预训练，该数据集包含 1420 万张图像和 22K 个类别。我们使用 AdamW 优化器训练 90 个 epoch，采用线性衰减学习率调度器和 5 个 epoch 的线性预热。使用 4096 的批大小、0.001 的初始学习率和 0.01 的权重衰减。在 ImageNet-1K 微调中，我们使用 1024 的批大小、恒定的 $1 0 ^ { - 5 }$ 学习率以及 $1 0 ^ { - 8 }$ 的权重衰减训练模型 30 个 epoch。

常规 ImageNet-1K 训练结果 Table 1(a) 展示了在使用常规 ImageNet-1K 训练时与其他骨干网络（包括基于 Transformer 和基于 ConvNet）的对比。

与先前最先进的基于 Transformer 的架构（即 DeiT [63]）相比，Swin Transformer 明显超越了复杂度相近的对应 DeiT 架构：在使用 224<sup>2</sup> 输入时，Swin-T（81.3%）比 DeiT-S（79.8%）高出 +1.5%；在使用 224<sup>2</sup>/384<sup>2</sup> 输入时，Swin-B（83.3%/84.5%）分别比 DeiT-B（81.8%/83.1%）高出 +1.5%/1.4%。

与最先进的 ConvNet（即 RegNet [48] 和 EfficientNet [58]）相比，Swin Transformer 取得了稍优的速度-准确率权衡（speed-accuracy trade-off）。值得注意的是，尽管 RegNet [48] 和 EfficientNet [58] 是通过详尽的架构搜索获得的，但所提出的 Swin Transformer 是基于标准 Transformer 改编而来，具有进一步提升的巨大潜力。

基于 ImageNet-22K 预训练的结果 我们还在 ImageNet-22K 上对更大容量的 Swin-B 和 Swin-L 进行了预训练。在 ImageNet-1K 图像分类上微调的结果如 Table 1(b) 所示。对于 Swin-B，ImageNet-22K 预训练相比于在 ImageNet-1K 上从头训练带来了 1.8%∼1.9% 的增益。与以往 ImageNet-22K 预训练的最佳结果相比，我们的模型取得了显著更优的速度-准确率权衡：Swin-B 取得了 86.4% 的 top-1 准确率，在具有相似推理吞吐量（84.7 对比 85.9 images/sec）且 FLOPs 略低（47.0G 对比 55.4G）的情况下，比 ViT 高出 2.4%。更大的 Swin-L 模型达到了 87.3% 的 top-1 准确率，比 Swin-B 模型高出 +0.9%。

<table><tr><td colspan="6">(a) 常规 ImageNet-1K 训练模型</td></tr><tr><td>方法</td><td>图像尺寸</td><td>参数量</td><td>FLOPs</td><td>吞吐量 (图像 / 秒)</td><td>ImageNet top-1 准确率</td></tr><tr><td>RegNetY-4G [48]</td><td> $224^2$ </td><td>21M</td><td>4.0G</td><td>1156.7</td><td>80.0</td></tr><tr><td>RegNetY-8G [48]</td><td> $224^2$ </td><td>39M</td><td>8.0G</td><td>591.6</td><td>81.7</td></tr><tr><td>RegNetY-16G [48]</td><td> $224^2$ </td><td>84M</td><td>16.0G</td><td>334.7</td><td>82.9</td></tr><tr><td>EffNet-B3 [58]</td><td> $300^2$ </td><td>12M</td><td>1.8G</td><td>732.1</td><td>81.6</td></tr><tr><td>EffNet-B4 [58]</td><td> $380^2$ </td><td>19M</td><td>4.2G</td><td>349.4</td><td>82.9</td></tr><tr><td>EffNet-B5 [58]</td><td> $456^2$ </td><td>30M</td><td>9.9G</td><td>169.1</td><td>83.6</td></tr><tr><td>EffNet-B6 [58]</td><td> $528^2$ </td><td>43M</td><td>19.0G</td><td>96.9</td><td>84.0</td></tr><tr><td>EffNet-B7 [58]</td><td> $600^2$ </td><td>66M</td><td>37.0G</td><td>55.1</td><td>84.3</td></tr><tr><td>ViT-B/16 [20]</td><td> $384^2$ </td><td>86M</td><td>55.4G</td><td>85.9</td><td>77.9</td></tr><tr><td>ViT-L/16 [20]</td><td> $384^2$ </td><td>307M</td><td>190.7G</td><td>27.3</td><td>76.5</td></tr><tr><td>DeiT-S [63]</td><td> $224^2$ </td><td>22M</td><td>4.6G</td><td>940.4</td><td>79.8</td></tr><tr><td>DeiT-B [63]</td><td> $224^2$ </td><td>86M</td><td>17.5G</td><td>292.3</td><td>81.8</td></tr><tr><td>DeiT-B [63]</td><td> $384^2$ </td><td>86M</td><td>55.4G</td><td>85.9</td><td>83.1</td></tr><tr><td>Swin-T</td><td> $224^2$ </td><td>29M</td><td>4.5G</td><td>755.2</td><td>81.3</td></tr><tr><td>Swin-S</td><td> $224^2$ </td><td>50M</td><td>8.7G</td><td>436.9</td><td>83.0</td></tr><tr><td>Swin-B</td><td> $224^2$ </td><td>88M</td><td>15.4G</td><td>278.1</td><td>83.5</td></tr><tr><td>Swin-B</td><td> $384^2$ </td><td>88M</td><td>47.0G</td><td>84.7</td><td>84.5</td></tr></table>

(b) ImageNet-22K 预训练模型

<table><tr><td>方法</td><td>图像尺寸</td><td>参数量</td><td>FLOPs</td><td>吞吐量 (图像 / 秒)</td><td>ImageNet top-1 准确率</td></tr><tr><td>R-101x3 [38]</td><td> $384^2$ </td><td>388M</td><td>204.6G</td><td>-</td><td>84.4</td></tr><tr><td>R-152x4 [38]</td><td> $480^2$ </td><td>937M</td><td>840.5G</td><td>-</td><td>85.4</td></tr><tr><td>ViT-B/16 [20]</td><td> $384^2$ </td><td>86M</td><td>55.4G</td><td>85.9</td><td>84.0</td></tr><tr><td>ViT-L/16 [20]</td><td> $384^2$ </td><td>307M</td><td>190.7G</td><td>27.3</td><td>85.2</td></tr><tr><td>Swin-B</td><td> $224^2$ </td><td>88M</td><td>15.4G</td><td>278.1</td><td>85.2</td></tr><tr><td>Swin-B</td><td> $384^2$ </td><td>88M</td><td>47.0G</td><td>84.7</td><td>86.4</td></tr><tr><td>Swin-L</td><td> $384^2$ </td><td>197M</td><td>103.9G</td><td>42.1</td><td>87.3</td></tr></table>

Table 1. 不同骨干网络在 ImageNet-1K 分类任务上的比较。吞吐量遵循 [63]，使用 [68] 的 GitHub 仓库并在 V100 GPU 上进行测试。

## 4.2. COCO 上的目标检测

设置 目标检测与实例分割实验在 COCO 2017 上进行，该数据集包含 118K 张训练图像、5K 张验证图像和 20K 张 test-dev 图像。消融研究在验证集上进行，系统级对比在 test-dev 上报告。对于消融研究，我们考虑了 mmdetection [10] 中的四种典型目标检测框架：Cascade Mask R-CNN [29, 6]、ATSS [79]、RepPoints v2 [12] 和 Sparse RCNN [56]。对于这四种框架，我们采用相同的设置：多尺度训练 [8, 56]（调整输入大小使短边在 480 到 800 之间，长边至多为 1333）、AdamW [44] 优化器（初始学习率为 0.0001，权重衰减为 0.05，批大小为 16）以及 3x 训练策略（36 个 epoch）。对于系统级对比，我们采用了改进的 HTC [9]（记为 HTC++），并结合了 instaboost [22]、更强的多尺度训练 [7]、6x 训练策略（72 个 epoch）、soft-NMS [5]，并使用 ImageNet-22K 预训练模型作为初始化。

我们将 Swin Transformer 与标准卷积网络（即 ResNe(X)t）以及先前的 Transformer 网络（例如 DeiT）进行了比较。对比实验在仅更换主干网络且保持其他设置不变的情况下进行。需要注意的是，由于 Swin Transformer 和 ResNe(X)t 具有分层特征图，它们可以直接应用于上述所有框架，而 DeiT 仅生成单一分辨率的特征图，无法直接应用。为了公平比较，我们遵循 [81]，使用反卷积层为 DeiT 构建分层特征图。

<table><tr><td colspan="8">(a) 各种框架</td></tr><tr><td>方法</td><td>主干网络</td><td> $AP^{box}$ </td><td> $AP_{50}^{box}$ </td><td> $AP_{75}^{box}$ </td><td>#param.</td><td>FLOPs</td><td>FPS</td></tr><tr><td>Cascade</td><td>R-50</td><td>46.3</td><td>64.3</td><td>50.5</td><td>82M</td><td>739G</td><td>18.0</td></tr><tr><td>Mask R-CNN</td><td>Swin-T</td><td>50.5</td><td>69.3</td><td>54.9</td><td>86M</td><td>745G</td><td>15.3</td></tr><tr><td rowspan="2">ATSS</td><td>R-50</td><td>43.5</td><td>61.9</td><td>47.0</td><td>32M</td><td>205G</td><td>28.3</td></tr><tr><td>Swin-T</td><td>47.2</td><td>66.5</td><td>51.3</td><td>36M</td><td>215G</td><td>22.3</td></tr><tr><td rowspan="2">RepPointsV2</td><td>R-50</td><td>46.5</td><td>64.6</td><td>50.3</td><td>42M</td><td>274G</td><td>13.6</td></tr><tr><td>Swin-T</td><td>50.0</td><td>68.5</td><td>54.2</td><td>45M</td><td>283G</td><td>12.0</td></tr><tr><td>Sparse</td><td>R-50</td><td>44.5</td><td>63.4</td><td>48.2</td><td>106M</td><td>166G</td><td>21.0</td></tr><tr><td>R-CNN</td><td>Swin-T</td><td>47.9</td><td>67.3</td><td>52.3</td><td>110M</td><td>172G</td><td>18.4</td></tr></table>

(b) 各种主干网络结合 Cascade Mask R-CNN

<table><tr><td></td><td> $AP^{box}$ </td><td> $AP_{50}^{box}$ </td><td> $AP_{75}^{box}$ </td><td> $AP^{mask}$ </td><td> $AP_{50}^{mask}$ </td><td> $AP_{75}^{mask}$ </td><td>paramFLOPs</td><td>FPS</td></tr><tr><td>DeiT-S $^{\dagger}$ </td><td>48.0</td><td>67.2</td><td>51.7</td><td>41.4</td><td>64.2</td><td>44.3</td><td>80M</td><td>889G 10.4</td></tr><tr><td>R50</td><td>46.3</td><td>64.3</td><td>50.5</td><td>40.1</td><td>61.7</td><td>43.4</td><td>82M</td><td>739G 18.0</td></tr><tr><td>Swin-T</td><td>50.5</td><td>69.3</td><td>54.9</td><td>43.7</td><td>66.6</td><td>47.1</td><td>86M</td><td>745G 15.3</td></tr><tr><td>X101-32</td><td>48.1</td><td>66.5</td><td>52.4</td><td>41.6</td><td>63.9</td><td>45.2</td><td>101M</td><td>819G 12.8</td></tr><tr><td>Swin-S</td><td>51.8</td><td>70.4</td><td>56.3</td><td>44.7</td><td>67.9</td><td>48.5</td><td>107M</td><td>838G 12.0</td></tr><tr><td>X101-64</td><td>48.3</td><td>66.4</td><td>52.3</td><td>41.7</td><td>64.0</td><td>45.1</td><td>140M</td><td>972G 10.4</td></tr><tr><td>Swin-B</td><td>51.9</td><td>70.9</td><td>56.5</td><td>45.0</td><td>68.4</td><td>48.7</td><td>145M</td><td>982G 11.6</td></tr></table>

(c) 系统级比较

<table><tr><td rowspan="2">方法</td><td colspan="2">mini-val</td><td colspan="2">test-dev</td><td rowspan="2" colspan="2">#param. FLOPs</td></tr><tr><td> $AP^{box}$ </td><td> $AP^{mask}$ </td><td> $AP^{box}$ </td><td> $AP^{mask}$ </td></tr><tr><td>RepPointsV2* [12]</td><td>-</td><td>-</td><td>52.1</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GCNet* [7]</td><td>51.8</td><td>44.7</td><td>52.3</td><td>45.4</td><td>-</td><td>1041G</td></tr><tr><td>RelationNet++* [13]</td><td>-</td><td>-</td><td>52.7</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SpineNet-190 [21]</td><td>52.6</td><td>-</td><td>52.8</td><td>-</td><td>164M</td><td>1885G</td></tr><tr><td>ResNeSt-200* [78]</td><td>52.5</td><td>-</td><td>53.3</td><td>47.1</td><td>-</td><td>-</td></tr><tr><td>EfficientDet-D7 [59]</td><td>54.4</td><td>-</td><td>55.1</td><td>-</td><td>77M</td><td>410G</td></tr><tr><td>DetectoRS* [46]</td><td>-</td><td>-</td><td>55.7</td><td>48.5</td><td>-</td><td>-</td></tr><tr><td>YOLOv4 P7* [4]</td><td>-</td><td>-</td><td>55.8</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Copy-paste [26]</td><td>55.9</td><td>47.2</td><td>56.0</td><td>47.4</td><td>185M</td><td>1440G</td></tr><tr><td>X101-64 (HTC++)</td><td>52.3</td><td>46.0</td><td>-</td><td>-</td><td>155M</td><td>1033G</td></tr><tr><td>Swin-B (HTC++)</td><td>56.4</td><td>49.1</td><td>-</td><td>-</td><td>160M</td><td>1043G</td></tr><tr><td>Swin-L (HTC++)</td><td>57.1</td><td>49.5</td><td>57.7</td><td>50.2</td><td>284M</td><td>1470G</td></tr><tr><td>Swin-L (HTC++)*</td><td>58.0</td><td>50.4</td><td>58.7</td><td>51.1</td><td>284M</td><td>-</td></tr></table>

Table 2. COCO 目标检测与实例分割结果。<sup>†</sup>表示使用额外的反卷积层来生成分层特征图。\* 表示多尺度测试。

与 ResNe(X)t 的比较 Table 2(a) 列出了 Swin-T 和 ResNet-50 在四种目标检测框架上的结果。相比 ResNet-50，我们的 Swin-T 架构带来了持续 +3.4∼4.2 box AP 的提升，同时模型大小、FLOPs 和延迟仅略有增加。

Table 2(b) 比较了在使用 Cascade Mask R-CNN 的不同模型容量下 Swin Transformer 与 ResNe(X)t 的表现。Swin Transformer 取得了 51.9 box AP 和 45.0 mask AP 的高检测精度，相较于具有相似模型大小、FLOPs 和延迟的 ResNeXt101-64x4d，分别带来了 +3.6 box AP 和 +3.3 mask AP 的显著提升。在使用改进的 HTC 框架达到的 52.3 box AP 和 46.0 mask AP 这一更高基线上，Swin Transformer 带来的提升同样显著，分别达到了 +4.1 box AP 和 +3.1 mask AP（见 Table 2(c)）。关于推理速度，ResNe(X)t 是通过高度优化的 Cudnn 函数构建的，而我们的架构则是使用未完全优化的内置 PyTorch 函数实现的。彻底的算子核心（kernel）优化超出了本文的研究范围。

<table><tr><td colspan="2">ADE20K</td><td>验证集</td><td>测试集</td><td rowspan="2">参数量</td><td rowspan="2">FLOPs</td><td rowspan="2">FPS</td></tr><tr><td>方法</td><td>骨干网络</td><td>mIoU</td><td>得分</td></tr><tr><td>DANet [23]</td><td>ResNet-101</td><td>45.2</td><td>-</td><td>69M</td><td>1119G</td><td>15.2</td></tr><tr><td>DLab.v3+ [11]</td><td>ResNet-101</td><td>44.1</td><td>-</td><td>63M</td><td>1021G</td><td>16.0</td></tr><tr><td>ACNet [24]</td><td>ResNet-101</td><td>45.9</td><td>38.5</td><td>-</td><td></td><td></td></tr><tr><td>DNL [71]</td><td>ResNet-101</td><td>46.0</td><td>56.2</td><td>69M</td><td>1249G</td><td>14.8</td></tr><tr><td>OCRNet [73]</td><td>ResNet-101</td><td>45.3</td><td>56.0</td><td>56M</td><td>923G</td><td>19.3</td></tr><tr><td>UperNet [69]</td><td>ResNet-101</td><td>44.9</td><td>-</td><td>86M</td><td>1029G</td><td>20.1</td></tr><tr><td>OCRNet [73]</td><td>HRNet-w48</td><td>45.7</td><td>-</td><td>71M</td><td>664G</td><td>12.5</td></tr><tr><td>DLab.v3+ [11]</td><td>ResNeSt-101</td><td>46.9</td><td>55.1</td><td>66M</td><td>1051G</td><td>11.9</td></tr><tr><td>DLab.v3+ [11]</td><td>ResNeSt-200</td><td>48.4</td><td>-</td><td>88M</td><td>1381G</td><td>8.1</td></tr><tr><td>SETR [81]</td><td>T-Large $^{\dagger}$ </td><td>50.3</td><td>61.7</td><td>308M</td><td>-</td><td>-</td></tr><tr><td>UperNet</td><td>DeiT-S $^{\dagger}$ </td><td>44.0</td><td>-</td><td>52M</td><td>1099G</td><td>16.2</td></tr><tr><td>UperNet</td><td>Swin-T</td><td>46.1</td><td>-</td><td>60M</td><td>945G</td><td>18.5</td></tr><tr><td>UperNet</td><td>Swin-S</td><td>49.3</td><td>-</td><td>81M</td><td>1038G</td><td>15.2</td></tr><tr><td>UperNet</td><td>Swin-B $^{\ddagger}$ </td><td>51.6</td><td>-</td><td>121M</td><td>1841G</td><td>8.7</td></tr><tr><td>UperNet</td><td>Swin-L $^{\ddagger}$ </td><td>53.5</td><td>62.8</td><td>234M</td><td>3230G</td><td>6.2</td></tr></table>

Table 3. ADE20K 验证集和测试集上的语义分割结果。<sup>†</sup> 表示使用额外的反卷积层来生成分层特征图。‡ 表示该模型在 ImageNet-22K 上进行了预训练。

与 DeiT 的比较 采用 Cascade Mask R-CNN 框架的 DeiT-S 性能如 Table 2(b) 所示。在模型大小相似的情况下（86M 对比 80M），Swin-T 的结果比 DeiT-S 高出 +2.5 边界框 AP（box AP）和 +2.3 掩码 AP（mask AP），并且推理速度明显更快（15.3 FPS 对比 10.4 FPS）。DeiT 较低的推理速度主要是由于其对输入图像大小具有二次复杂度。

与以往最先进技术的比较 Table 2(c) 将我们的最佳结果与以往最先进的模型进行了比较。我们的最佳模型在 COCO test-dev 上取得了 58.7 边界框 AP 和 51.1 掩码 AP，超过了以往的最佳结果 +2.7 边界框 AP（无外部数据的 Copy-paste [26]）和 +2.6 掩码 AP（DetectoRS [46]）。

## 4.3. ADE20K 上的语义分割

设置 ADE20K [83] 是一个广泛使用的语义分割数据集，涵盖了 150 个广泛的语义类别。它总共包含 25K 张图像，其中 20K 用于训练，2K 用于验证，另有 3K 用于测试。鉴于其高效性，我们采用 mmseg [16] 中的 UperNet [69] 作为基础框架。更多细节见附录。

<table><tr><td rowspan="2"></td><td colspan="2">ImageNet</td><td colspan="2">COCO</td><td>ADE20k</td></tr><tr><td>top-1</td><td>top-5</td><td> $AP^{box}$ </td><td> $AP^{mask}$ </td><td>mIoU</td></tr><tr><td>w/o shifting</td><td>80.2</td><td>95.1</td><td>47.7</td><td>41.5</td><td>43.3</td></tr><tr><td>shifted windows</td><td>81.3</td><td>95.6</td><td>50.5</td><td>43.7</td><td>46.1</td></tr><tr><td>no pos.</td><td>80.1</td><td>94.9</td><td>49.2</td><td>42.6</td><td>43.8</td></tr><tr><td>abs. pos.</td><td>80.5</td><td>95.2</td><td>49.0</td><td>42.4</td><td>43.2</td></tr><tr><td>abs.+rel. pos.</td><td>81.3</td><td>95.6</td><td>50.2</td><td>43.4</td><td>44.0</td></tr><tr><td>rel. pos. w/o app.</td><td>79.3</td><td>94.7</td><td>48.2</td><td>41.9</td><td>44.1</td></tr><tr><td>rel. pos.</td><td>81.3</td><td>95.6</td><td>50.5</td><td>43.7</td><td>46.1</td></tr></table>

Table 4. 在三个基准上使用 Swin-T 架构对移动窗口方法和不同位置编码方法进行的消融研究。w/o shifting：所有自注意力模块均采用常规窗口划分，无移动；abs. pos.：ViT 的绝对位置编码项；rel. pos.：带有额外相对位置偏置项的默认设置（见公式 (4)）；app.：公式 (4) 中的第一个缩放点积项。

结果 Table 3 列出了不同方法/骨干网络对的 mIoU、模型大小（#param）、FLOPs 和 FPS。从这些结果可以看出，在相似的计算成本下，Swin-S 比 DeiT-S 高出 +5.3 mIoU（49.3 对比 44.0）。它还比 ResNet-101 高出 +4.4 mIoU，比 ResNeSt-101 [78] 高出 +2.4 mIoU。我们带有 ImageNet-22K 预训练的 Swin-L 模型在验证集上达到了 53.5 mIoU，比之前的最佳模型高出 +3.2 mIoU（SETR [81] 为 50.3 mIoU，且其模型尺寸更大）。

## 4.4. 消融研究

在本节中，我们利用 ImageNet-1K 图像分类、COCO 目标检测上的 Cascade Mask R-CNN 以及 ADE20K 语义分割上的 UperNet，对所提出的 Swin Transformer 中的重要设计元素进行消融实验。

移动窗口 移动窗口方法在三项任务上的消融实验结果如 Table 4 所示。在每个阶段中，采用移动窗口划分的 Swin-T 性能优于基于单一窗口划分的对应模型，在 ImageNet-1K 上 top-1 准确率提高 +1.1%，在 COCO 上提高 +2.8 box AP/+2.2 mask AP，在 ADE20K 上提高 +2.8 mIoU。结果表明使用移动窗口在前面各层的窗口之间建立连接的有效性。如 Table 5 所示，移动窗口带来的延迟开销也很小。

相对位置偏差（Relative position bias） Table 4 展示了不同位置嵌入方法的比较。相较于无位置编码和具有绝对位置嵌入的变体，带有相对位置偏差的 Swin-T 在 ImageNet-1K 上的 Top-1 准确率分别提升了 +1.2%/+0.8%，在 COCO 上的 box AP 分别提升了 +1.3/+1.5、mask AP 分别提升了 +1.1/+1.3，在 ADE20K 上的 mIoU 分别提升了 +2.3/+2.9，这表明了相对位置偏差的有效性。还要注意，虽然引入绝对位置嵌入提高了图像分类准确率（+0.4%），但它损害了目标检测和语义分割的性能（在 COCO 上 box/mask AP 分别 -0.2，在 ADE20K 上 mIoU -0.6）。

<table><tr><td rowspan="2">方法</td><td colspan="4">单阶段 MSA (ms)</td><td colspan="3">架构 (FPS)</td></tr><tr><td>S1</td><td>S2</td><td>S3</td><td>S4</td><td>T</td><td>S</td><td>B</td></tr><tr><td>滑动窗口 (朴素)</td><td>122.5</td><td>38.3</td><td>12.1</td><td>7.6</td><td>183</td><td>109</td><td>77</td></tr><tr><td>滑动窗口 (kernel)</td><td>7.6</td><td>4.7</td><td>2.7</td><td>1.8</td><td>488</td><td>283</td><td>187</td></tr><tr><td>Performer [14]</td><td>4.8</td><td>2.8</td><td>1.8</td><td>1.5</td><td>638</td><td>370</td><td>241</td></tr><tr><td>窗口 (无移位)</td><td>2.8</td><td>1.7</td><td>1.2</td><td>0.9</td><td>770</td><td>444</td><td>280</td></tr><tr><td>移位窗口 (填充)</td><td>3.3</td><td>2.3</td><td>1.9</td><td>2.2</td><td>670</td><td>371</td><td>236</td></tr><tr><td>移位窗口 (循环移位)</td><td>3.0</td><td>1.9</td><td>1.3</td><td>1.0</td><td>755</td><td>437</td><td>278</td></tr></table>

Table 5. 不同自注意力计算方法和实现在 V100 GPU 上的实际速度。

虽然近期的 ViT/DeiT 模型在图像分类中放弃了平移不变性（尽管长期以来平移不变性已被证明对视觉建模至关重要），但我们发现，引入一定平移不变性的归纳偏置对于通用视觉建模仍然更为适用，特别是对于目标检测和语义分割等密集预测任务。

不同的自注意力方法（Different self-attention methods） Table 5 比较了不同自注意力计算方法和实现的实际速度。我们的循环移位实现在硬件效率上优于朴素填充，在更深的阶段尤其明显。总体而言，它在 Swin-T、Swin-S 和 Swin-B 上分别带来了 13%、18% 和 18% 的加速。

在四个网络阶段中，基于所提出的移位窗口方法构建的自注意力模块比朴素/kernel 实现的滑动窗口自注意力模块效率分别高出 40.8×/2.5×、20.2×/2.5×、9.3×/2.1× 和 7.6×/1.8×。总体而言，基于移位窗口构建的 Swin Transformer 架构比基于滑动窗口构建的变体在 Swin-T、Swin-S 和 Swin-B 上分别快 4.1/1.5、4.0/1.5 和 3.6/1.5 倍。Table 6 比较了它们在三项任务上的准确率，表明它们在视觉建模中具有相当的准确率。

与最快的 Transformer 架构之一 Performer [14] 相比（参见 [60]），所提出的基于移动窗口的自注意力计算以及整体 Swin Transformer 架构略微更快（参见 Table 5），同时在使用 Swin-T 时在 ImageNet-1K 上相比 Performer 取得了 +2.3% 的 top-1 准确率（参见 Table 6）。

## 5. 结论

本文提出了 Swin Transformer，一种新型的视觉 Transformer，它能够生成分层的特征表示，并且计算复杂度与输入图像大小呈线性关系。Swin Transformer 在 COCO 目标检测和 ADE20K 语义分割上取得了最先进的（state-of-the-art）性能，显著超越了以往的最佳方法。我们希望 Swin Transformer 在各种视觉问题上的强劲表现将促进视觉和语言信号的统一建模。

<table><tr><td rowspan="2"></td><td rowspan="2">骨干网络</td><td rowspan="2">ImageNet top-1</td><td rowspan="2">top-5</td><td colspan="2">COCO</td><td rowspan="2">ADE20k mIoU</td></tr><tr><td> $AP^{box}$ </td><td> $AP^{mask}$ </td></tr><tr><td>滑动窗口</td><td>Swin-T</td><td>81.4</td><td>95.6</td><td>50.2</td><td>43.5</td><td>45.8</td></tr><tr><td>Performer [14]</td><td>Swin-T</td><td>79.0</td><td>94.2</td><td>-</td><td>-</td><td>-</td></tr><tr><td>移动窗口</td><td>Swin-T</td><td>81.3</td><td>95.6</td><td>50.5</td><td>43.7</td><td>46.1</td></tr></table>

Table 6. 在三个基准上使用不同自注意力计算方法的 Swin Transformer 的准确率。

作为 Swin Transformer 的核心要素，基于移动窗口的自注意力被证明在视觉问题上是高效且有效的，我们期待未来对其在自然语言处理中的应用展开研究。

## 致谢

我们感谢微软（Microsoft）众多同事的帮助，特别是 Li Dong 和 Furu Wei 的有益讨论；以及 Bin Xiao、Lu Yuan 和 Lei Zhang 在数据集方面的协助。

## A1. 详细架构

详细的架构规格如 Table 7 所示，其中所有架构均假设输入图像尺寸为 224×224。“Concat $n \times n ^ { \prime \prime }$”表示将一个 patch 内的 n × n 个相邻特征进行拼接。该操作将特征图下采样 n 倍。“96-d”表示输出维度为 96 的线性层。“win. sz. $7 \times 7 ^ { 5 }$”表示窗口大小为 $7 \times 7$ 的多头自注意力模块。

## A2. 详细实验设置

## A2.1. ImageNet-1K 上的图像分类

图像分类是通过在最后一个阶段（stage）输出的特征图上应用全局平均池化层，随后接一个线性分类器来执行的。我们发现该策略与 ViT [20] 和 DeiT [63] 中使用额外类别标记（class token）的方法一样准确。在评估中，报告了使用单裁剪（single crop）的 top-1 准确率。

常规 ImageNet-1K 训练 训练设置基本遵循 [63]。对于所有模型变体，我们采用默认的 $2 2 4 ^ { 2 }$ 输入图像分辨率。对于其他分辨率（如 $3 8 4 ^ { 2 }$），我们在 $2 2 4 ^ { 2 }$ 分辨率下训练好的模型上进行微调，而不是从头开始训练，以减少 GPU 消耗。

在使用 $2 2 4 ^ { 2 }$ 输入从头训练时，我们使用 AdamW [37] 优化器训练 300 个周期，采用带有 20 个周期线性预热（warm-up）的余弦衰减学习率调度器。批次大小为 1024，初始学习率为 0.001，权重衰减为 0.05，并采用最大范数为 1 的梯度裁剪。我们在训练中包含了 [63] 中的大部分增强和正则化策略，包括 RandAugment [17]、Mixup [77]、Cutmix [75]、随机擦除 [82] 和随机深度 [35]，但没有采用无法提升性能的重复增强 [31] 和指数移动平均（EMA）[45]。需要注意的是，这与 [63] 相反，在 [63] 中重复增强对于稳定 ViT 的训练至关重要。对于更大的模型，采用了递增程度的随机深度增强，即 Swin-T、Swin-S 和 Swin-B 分别为 0.2、0.3、0.5。

对于在更大分辨率输入上的微调，我们使用 AdamW [37] 优化器训练 30 个周期，采用恒定学习率 $1 0 ^ { - 5 }$、权重衰减 $1 0 ^ { - 8 }$，以及与第一阶段相同的数据增强和正则化，除了将随机深度比例设置为 0.1。

ImageNet-22K 预训练 我们还在更大的 ImageNet-22K 数据集上进行了预训练，该数据集包含 1420 万张图像和 2.2 万个类别。训练分两个阶段进行。对于输入为 $2 2 4 ^ { 2 }$ 的第一阶段，我们使用 AdamW 优化器训练 90 个周期，采用带有 5 个周期线性预热的线性衰减学习率调度器。批次大小为 4096，初始学习率为 0.001，权重衰减为 0.01。在输入为 $2 2 4 ^ { 2 } / 3 8 4 ^ { 2 }$ 的第二阶段 ImageNet-1K 微调中，我们以 1024 的批次大小、恒定学习率 $1 0 ^ { - 5 }$ 以及权重衰减 $1 0 ^ { - 8 }$ 对模型训练 30 个周期。

## A2.2. COCO 上的目标检测

对于消融实验，我们考虑了 mmdetection [10] 中的四种典型目标检测框架：Cascade Mask R-CNN [29, 6]、ATSS [79]、RepPoints v2 [12] 和 Sparse RCNN [56]。对于这四种框架，我们采用相同的设置：多尺度训练 [8, 56]（调整输入尺寸使短边在 480 到 800 之间，长边最大为 1333）、AdamW [44] 优化器（初始学习率为 0.0001，权重衰减为 0.05，批次大小为 16）以及 3x 调度（36 个周期，在第 27 和 33 个周期学习率衰减 10 倍）。

为了进行系统级比较，我们采用了一种改进的 HTC [9]（记为 HTC++），其结合了 instaboost [22]、更强的数据多尺度训练 [7]（调整输入尺寸使短边在 400 至 1400 之间，长边最多为 1600）、6x 训练计划（72 个 epoch，在第 63 和 69 个 epoch 时学习率衰减 0.1 倍）、soft-NMS [5]，并在最后一个阶段的输出处添加了额外的全局自注意力层，同时采用 ImageNet-22K 预训练模型作为初始化。对于所有 Swin Transformer 模型，我们均采用比率为 0.2 的随机深度（stochastic depth）。

<table><tr><td></td><td>下采样率 (输出尺寸)</td><td colspan="2">Swin-T</td><td colspan="2">Swin-S</td><td colspan="2">Swin-B</td><td colspan="2">Swin-L</td></tr><tr><td rowspan="2">阶段 1</td><td rowspan="2">4×(56×56)</td><td colspan="2">concat 4×4, 96-d, LN</td><td colspan="2">concat 4×4, 96-d, LN</td><td colspan="2">concat 4×4, 128-d, LN</td><td colspan="2">concat 4×4, 192-d, LN</td></tr><tr><td colspan="2">[win. sz. 7×7, dim 96, head 3] × 2</td><td colspan="2">[win. sz. 7×7, dim 96, head 3] × 2</td><td colspan="2">[win. sz. 7×7, dim 128, head 4] × 2</td><td colspan="2">[win. sz. 7×7, dim 192, head 6] × 2</td></tr><tr><td rowspan="2">阶段 2</td><td rowspan="2">8×(28×28)</td><td colspan="2">concat 2×2, 192-d, LN</td><td colspan="2">concat 2×2, 192-d, LN</td><td colspan="2">concat 2×2, 256-d, LN</td><td colspan="2">concat 2×2, 384-d, LN</td></tr><tr><td colspan="2">[win. sz. 7×7, dim 192, head 6] × 2</td><td colspan="2">[win. sz. 7×7, dim 192, head 6] × 2</td><td colspan="2">[win. sz. 7×7, dim 256, head 8] × 2</td><td colspan="2">[win. sz. 7×7, dim 384, head 12] × 2</td></tr><tr><td rowspan="2">阶段 3</td><td rowspan="2">16×(14×14)</td><td colspan="2">concat 2×2, 384-d, LN</td><td colspan="2">concat 2×2, 384-d, LN</td><td colspan="2">concat 2×2, 512-d, LN</td><td colspan="2">concat 2×2, 768-d, LN</td></tr><tr><td colspan="2">[win. sz. 7×7, dim 384, head 12] × 6</td><td colspan="2">[win. sz. 7×7, dim 384, head 12] × 18</td><td colspan="2">[win. sz. 7×7, dim 512, head 16] × 18</td><td colspan="2">[win. sz. 7×7, dim 768, head 24] × 18</td></tr><tr><td rowspan="2">阶段 4</td><td rowspan="2">32×(7×7)</td><td colspan="2">concat 2×2, 768-d, LN</td><td colspan="2">concat 2×2, 768-d, LN</td><td colspan="2">concat 2×2, 1024-d, LN</td><td colspan="2">concat 2×2, 1536-d, LN</td></tr><tr><td colspan="2">[win. sz. 7×7, dim 768, head 24] × 2</td><td colspan="2">[win. sz. 7×7, dim 768, head 24] × 2</td><td colspan="2">[win. sz. 7×7, dim 1024, head 32] × 2</td><td colspan="2">[win. sz. 7×7, dim 1536, head 48] × 2</td></tr></table>

Table 7. 详细的架构规格说明。

## A2.3. ADE20K 上的语义分割

ADE20K [83] 是一个广泛使用的语义分割数据集，涵盖了 150 个广泛的语义类别。它总共包含 25K 张图像，其中 20K 张用于训练，2K 张用于验证，另外 3K 张用于测试。鉴于其高效率，我们采用 mmsegmentation [16] 中的 UperNet [69] 作为基础框架。

在训练中，我们采用 AdamW [44] 优化器，初始学习率为 $6 \times 1 0 ^ { - 5 }$，权重衰减为 0.01，使用线性学习率衰减调度器，以及 1,500 次迭代的线性预热（warmup）。模型在 8 块 GPU 上训练，每块 GPU 2 张图像，共训练 160K 次迭代。在数据增强方面，我们采用 mmsegmentation 中的默认设置，包括随机水平翻转、比例范围在 [0.5, 2.0] 内的随机缩放以及随机光学畸变（photometric distortion）。所有 Swin Transformer 模型均应用比率为 0.2 的随机深度（stochastic depth）。Swin-T 和 Swin-S 按照之前方法的标准设置进行训练，输入尺寸为 512×512。带有 ‡ 的 Swin-B 和 Swin-L 表示这两个模型在 ImageNet-22K 上进行了预训练，并以 640×640 的输入尺寸进行训练。

在推理阶段，采用多尺度测试，使用的分辨率为训练分辨率的 [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] 倍。在报告测试分数时，按照通用做法 [71]，同时使用训练集图像和验证集图像进行训练。

## A3. 更多实验

## A3.1. 不同输入尺寸的图像分类

Table 8 列出了 Swin Transformers 在输入图像尺寸从 $2 2 4 ^ { 2 }$ 到 384<sup>2</sup> 下的性能表现。通常，更大的输入分辨率会带来更好的 top-1 准确率，但推理速度会有所降低。

<table><tr><td rowspan="2">输入尺寸</td><td colspan="2">Swin-T</td><td colspan="2">Swin-S</td><td colspan="2">Swin-B</td></tr><tr><td>top-1 准确率</td><td>吞吐量 (image / s)</td><td>top-1 准确率</td><td>吞吐量 (image / s)</td><td>top-1 准确率</td><td>吞吐量 (image / s)</td></tr><tr><td> $224^2$ </td><td>81.3</td><td>755.2</td><td>83.0</td><td>436.9</td><td>83.3</td><td>278.1</td></tr><tr><td> $256^2$ </td><td>81.6</td><td>580.9</td><td>83.4</td><td>336.7</td><td>83.7</td><td>208.1</td></tr><tr><td> $320^2$ </td><td>82.1</td><td>342.0</td><td>83.7</td><td>198.2</td><td>84.0</td><td>132.0</td></tr><tr><td> $384^2$ </td><td>82.2</td><td>219.5</td><td>83.9</td><td>127.6</td><td>84.5</td><td>84.7</td></tr></table>

Table 8. 不同输入图像尺寸的 Swin Transformers 在 ImageNet-1K 分类任务上的表现。

<table><tr><td>骨干网络</td><td>优化器</td><td> $AP^{box}$ </td><td> $AP_{50}^{box}$ </td><td> $AP_{75}^{box}$ </td><td> $AP^{mask}$ </td><td> $AP_{50}^{mask}$ </td><td> $AP_{75}^{mask}$ </td></tr><tr><td rowspan="2">R50</td><td>SGD</td><td>45.0</td><td>62.9</td><td>48.8</td><td>38.5</td><td>59.9</td><td>41.4</td></tr><tr><td>AdamW</td><td>46.3</td><td>64.3</td><td>50.5</td><td>40.1</td><td>61.7</td><td>43.4</td></tr><tr><td rowspan="2">X101-32x4d</td><td>SGD</td><td>47.8</td><td>65.9</td><td>51.9</td><td>40.4</td><td>62.9</td><td>43.5</td></tr><tr><td>AdamW</td><td>48.1</td><td>66.5</td><td>52.4</td><td>41.6</td><td>63.9</td><td>45.2</td></tr><tr><td rowspan="2">X101-64x4d</td><td>SGD</td><td>48.8</td><td>66.9</td><td>53.0</td><td>41.4</td><td>63.9</td><td>44.7</td></tr><tr><td>AdamW</td><td>48.3</td><td>66.4</td><td>52.3</td><td>41.7</td><td>64.0</td><td>45.1</td></tr></table>

Table 9. 在使用 Cascade Mask R-CNN 框架的 COCO 目标检测任务中，ResNe(X)t 骨干网络使用 SGD 与 AdamW 优化器的对比。

## A3.2. ResNe(X)t 在 COCO 上的不同优化器

Table 9 对比了 ResNe(X)t 骨干网络在 COCO 目标检测任务中使用 AdamW 和 SGD 优化器的效果。该对比基于 Cascade Mask R-CNN 框架进行。虽然 SGD 是 Cascade Mask R-CNN 框架的默认优化器，但我们普遍观察到将其替换为 AdamW 优化器能够提升准确率，尤其是对于较小的骨干网络。因此，在与本文提出的 Swin Transformer 架构进行对比时，我们对 ResNe(X)t 骨干网络采用了 AdamW。

## A3.3. Swin MLP-Mixer

我们将提出的分层设计和移动窗口方法应用于 MLP-Mixer 架构 [61]，称之为 Swin-Mixer。Table 10 展示了 Swin-Mixer 与原始 MLP-Mixer 架构 MLP-Mixer [61] 以及后续方法 ResMLP [61] 的性能对比。Swin-Mixer 在使用略小的计算预算（10.4G 对比 12.7G）的情况下，性能显著优于 MLP-Mixer（81.3% 对比 76.4%）。与 ResMLP [62] 相比，它也具有更好的速度-准确率权衡。这些结果表明，本文提出的分层设计和移动窗口方法具有良好的泛化性。

<table><tr><td>方法</td><td>图像尺寸</td><td>参数量</td><td>FLOPs</td><td>吞吐量（图像 / 秒）</td><td>ImageNet Top-1 准确率</td></tr><tr><td>MLP-Mixer-B/16 [61]</td><td> $224^2$ </td><td>59M</td><td>12.7G</td><td>-</td><td>76.4</td></tr><tr><td>ResMLP-S24 [62]</td><td> $224^2$ </td><td>30M</td><td>6.0G</td><td>715</td><td>79.4</td></tr><tr><td>ResMLP-B24 [62]</td><td> $224^2$ </td><td>116M</td><td>23.0G</td><td>231</td><td>81.0</td></tr><tr><td>Swin-T/D24 (Transformer)</td><td> $256^2$ </td><td>28M</td><td>5.9G</td><td>563</td><td>81.6</td></tr><tr><td>Swin-Mixer-T/D24</td><td> $256^2$ </td><td>20M</td><td>4.0G</td><td>807</td><td>79.4</td></tr><tr><td>Swin-Mixer-T/D12</td><td> $256^2$ </td><td>21M</td><td>4.0G</td><td>792</td><td>79.6</td></tr><tr><td>Swin-Mixer-T/D6</td><td> $256^2$ </td><td>23M</td><td>4.0G</td><td>766</td><td>79.7</td></tr><tr><td>Swin-Mixer-B/D24 (no shift)</td><td> $224^2$ </td><td>61M</td><td>10.4G</td><td>409</td><td>80.3</td></tr><tr><td>Swin-Mixer-B/D24</td><td> $224^2$ </td><td>61M</td><td>10.4G</td><td>409</td><td>81.3</td></tr></table>

Table 10. Swin MLP-Mixer 在 ImageNet-1K 分类任务上的性能。D 表示每个头（head）的通道数。吞吐量遵循 [63]，使用 [68] 的 GitHub 代码仓库和一块 V100 GPU 测得。

## 参考文献

[1] Hangbo Bao, Li Dong, Furu Wei, Wenhui Wang, Nan Yang, Xiaodong Liu, Yu Wang, Jianfeng Gao, Songhao Piao, Ming Zhou, et al. Unilmv2: Pseudo-masked language models for unified language model pre-training. In International Conference on Machine Learning, pages 642–652. PMLR, 2020. 5

[2] Josh Beal, Eric Kim, Eric Tzeng, Dong Huk Park, Andrew Zhai, and Dmitry Kislyuk. Toward transformer-based object detection. arXiv preprint arXiv:2012.09958, 2020. 3

[3] Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens, and Quoc V. Le. Attention augmented convolutional networks, 2020. 3

[4] Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. Yolov4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934, 2020. 7

[5] Navaneeth Bodla, Bharat Singh, Rama Chellappa, and Larry S. Davis. Soft-nms – improving object detection with one line of code. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), Oct 2017. 6, 9

[6] Zhaowei Cai and Nuno Vasconcelos. Cascade r-cnn: Delving into high quality object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6154–6162, 2018. 6, 9

[7] Yue Cao, Jiarui Xu, Stephen Lin, Fangyun Wei, and Han Hu. Gcnet: Non-local networks meet squeeze-excitation networks and beyond. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, Oct 2019. 3, 6, 7, 9

[8] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-

end object detection with transformers. In European Conference on Computer Vision, pages 213–229. Springer, 2020. 3, 6, 9

[9] Kai Chen, Jiangmiao Pang, Jiaqi Wang, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jianping Shi, Wanli Ouyang, et al. Hybrid task cascade for instance segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4974– 4983, 2019. 6, 9

[10] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, et al. Mmdetection: Open mmlab detection toolbox and benchmark. arXivpreprint arXiv:1906.07155, 2019. 6, 9

[11] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European conference on computer vision (ECCV), pages 801–818, 2018. 7

[12] Yihong Chen, Zheng Zhang, Yue Cao, Liwei Wang, Stephen Lin, and Han Hu. Reppoints v2: Verification meets regression for object detection. In NeurIPS, 2020. 6, 7, 9

[13] Cheng Chi, Fangyun Wei, and Han Hu. Relationnet++: Bridging visual representations for object detection via transformer decoder. In NeurIPS, 2020. 3, 7

[14] Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J Colwell, and Adrian Weller. Rethinking attention with performers. In International Conference on Learning Representations, 2021. 8, 9

[15] Xiangxiang Chu, Bo Zhang, Zhi Tian, Xiaolin Wei, and Huaxia Xia. Do we really need explicit position encodings for vision transformers? arXiv preprint arXiv:2102.10882, 2021. 3

[16] MMSegmentation Contributors. MMSegmentation: Openmmlab semantic segmentation toolbox and benchmark. https://github.com/open-mmlab/ mmsegmentation, 2020. 8, 10

[17] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data augmentation with a reduced search space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 702–703, 2020. 9

[18] Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, and Yichen Wei. Deformable convolutional networks. In Proceedings ofthe IEEE International Conference on Computer Vision, pages 764–773, 2017. 1, 3

[19] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 5

[20] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is

worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. 1, 2, 3, 4, 5, 6, 9

[21] Xianzhi Du, Tsung-Yi Lin, Pengchong Jin, Golnaz Ghiasi, Mingxing Tan, Yin Cui, Quoc V Le, and Xiaodan Song. Spinenet: Learning scale-permuted backbone for recognition and localization. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11592–11601, 2020. 7

[22] Hao-Shu Fang, Jianhua Sun, Runzhong Wang, Minghao Gou, Yong-Lu Li, and Cewu Lu. Instaboost: Boosting instance segmentation via probability map guided copypasting. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 682–691, 2019. 6, 9

[23] Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang, and Hanqing Lu. Dual attention network for scene segmentation. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 3146– 3154, 2019. 3, 7

[24] Jun Fu, Jing Liu, Yuhang Wang, Yong Li, Yongjun Bao, Jinhui Tang, and Hanqing Lu. Adaptive context network for scene parsing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6748–6757, 2019. 7

[25] Kunihiko Fukushima. Cognitron: A self-organizing multilayered neural network. Biological cybernetics, 20(3):121– 136, 1975. 3

[26] Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-Yi Lin, Ekin D Cubuk, Quoc V Le, and Barret Zoph. Simple copy-paste is a strong data augmentation method for instance segmentation. arXiv preprint arXiv:2012.07177, 2020. 2, 7

[27] Jiayuan Gu, Han Hu, Liwei Wang, Yichen Wei, and Jifeng Dai. Learning region features for object detection. In Proceedings of the European Conference on Computer Vision (ECCV), 2018. 3

[28] Kai Han, An Xiao, Enhua Wu, Jianyuan Guo, Chunjing Xu, and Yunhe Wang. Transformer in transformer. arXiv preprint arXiv:2103.00112, 2021. 3

[29] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Gir-´ shick. Mask r-cnn. In Proceedings ofthe IEEE international conference on computer vision, pages 2961–2969, 2017. 6, 9

[30] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 1, 2, 4

[31] Elad Hoffer, Tal Ben-Nun, Itay Hubara, Niv Giladi, Torsten Hoefler, and Daniel Soudry. Augment your batch: Improving generalization through instance repetition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8129–8138, 2020. 6, 9

[32] Han Hu, Jiayuan Gu, Zheng Zhang, Jifeng Dai, and Yichen Wei. Relation networks for object detection. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 3588–3597, 2018. 3, 5

[33] Han Hu, Zheng Zhang, Zhenda Xie, and Stephen Lin. Local relation networks for image recognition. In Proceedings of

the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3464–3473, October 2019. 2, 3, 5

[34] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4700–4708, 2017. 1, 2

[35] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In European conference on computer vision, pages 646–661. Springer, 2016. 9

[36] David H Hubel and Torsten N Wiesel. Receptive fields, binocular interaction and functional architecture in the cat’s visual cortex. The Journal of physiology, 160(1):106–154, 1962. 3

[37] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 5, 9

[38] Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby. Big transfer (bit): General visual representation learning. arXiv preprint arXiv:1912.11370, 6(2):8, 2019. 6

[39] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pages 1097–1105, 2012. 1, 2

[40] Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner,´ et al. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998. 2

[41] Yann LeCun, Patrick Haffner, Leon Bottou, and Yoshua Ben-´ gio. Object recognition with gradient-based learning. In Shape, contour and grouping in computer vision, pages 319– 345. Springer, 1999. 3

[42] Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017. 2

[43] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence´ Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 5

[44] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. 6, 9, 10

[45] Boris T Polyak and Anatoli B Juditsky. Acceleration of stochastic approximation by averaging. SIAM journal on control and optimization, 30(4):838–855, 1992. 6, 9

[46] Siyuan Qiao, Liang-Chieh Chen, and Alan Yuille. Detectors: Detecting objects with recursive feature pyramid and switchable atrous convolution. arXiv preprint arXiv:2006.02334, 2020. 2, 7

[47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 1

[48] Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollar. Designing network design´ spaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10428– 10436, 2020. 6

[49] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020. 5

[50] Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, and Jon Shlens. Stand-alone selfattention in vision models. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. 2, 3

[51] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015. 2

[52] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In International Conference on Learning Representations, May 2015. 2, 4

[53] Bharat Singh and Larry S Davis. An analysis of scale invariance in object detection snip. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3578–3587, 2018. 2

[54] Bharat Singh, Mahyar Najibi, and Larry S Davis. Sniper: Efficient multi-scale training. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. 2

[55] Aravind Srinivas, Tsung-Yi Lin, Niki Parmar, Jonathon Shlens, Pieter Abbeel, and Ashish Vaswani. Bottleneck transformers for visual recognition. arXiv preprint arXiv:2101.11605, 2021. 3

[56] Peize Sun, Rufeng Zhang, Yi Jiang, Tao Kong, Chenfeng Xu, Wei Zhan, Masayoshi Tomizuka, Lei Li, Zehuan Yuan, Changhu Wang, et al. Sparse r-cnn: End-to-end object detection with learnable proposals. arXiv preprint arXiv:2011.12450, 2020. 3, 6, 9

[57] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1–9, 2015. 2

[58] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International Conference on Machine Learning, pages 6105–6114. PMLR, 2019. 3, 6

[59] Mingxing Tan, Ruoming Pang, and Quoc V Le. Efficientdet: Scalable and efficient object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10781–10790, 2020. 7

[60] Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. Long range arena : A bench-

mark for efficient transformers. In International Conference on Learning Representations, 2021. 8

[61] Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, and Alexey Dosovitskiy. Mlp-mixer: An all-mlp architecture for vision, 2021. 2, 10, 11

[62] Hugo Touvron, Piotr Bojanowski, Mathilde Caron, Matthieu Cord, Alaaeldin El-Nouby, Edouard Grave, Gautier Izacard, Armand Joulin, Gabriel Synnaeve, Jakob Verbeek, and Herve J´egou. Resmlp: Feedforward networks for image classification with data-efficient training, 2021. 11

[63] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herve J´egou. Training data-efficient image transformers & distillation through attention. arXiv preprint arXiv:2012.12877, 2020. 2, 3, 5, 6, 9, 11

[64] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998–6008, 2017. 1, 2, 4

[65] Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, et al. Deep high-resolution representation learning for visual recognition. IEEE transactions on pattern analysis and machine intelligence, 2020. 3

[66] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. arXiv preprint arXiv:2102.12122, 2021. 3

[67] Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, 2018. 3

[68] Ross Wightman. Pytorch image models. https://github.com/rwightman/ pytorch-image-models, 2019. 6, 11

[69] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In Proceedings of the European Conference on Computer Vision (ECCV), pages 418–434, 2018. 7, 8, 10

[70] Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1492–1500, 2017. 1, 2, 3

[71] Minghao Yin, Zhuliang Yao, Yue Cao, Xiu Li, Zheng Zhang, Stephen Lin, and Han Hu. Disentangled non-local neural networks. In Proceedings of the European conference on computer vision (ECCV), 2020. 3, 7, 10

[72] Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Francis EH Tay, Jiashi Feng, and Shuicheng Yan. Tokens-to-token vit: Training vision transformers from scratch on imagenet. arXiv preprint arXiv:2101.11986, 2021. 3

[73] Yuhui Yuan, Xilin Chen, and Jingdong Wang. Object-contextual representations for semantic segmentation. In

16th European Conference Computer Vision (ECCV 2020), August 2020. 7

[74] Yuhui Yuan and Jingdong Wang. OCNet: 用于场景解析的对象上下文网络（Object context network for scene parsing）. arXiv preprint arXiv:1809.00916, 2018. 3

[75] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. CutMix: 训练具有可定位特征的强分类器的正则化策略（Regularization strategy to train strong classifiers with localizable features）. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6023–6032, 2019. 9

[76] Sergey Zagoruyko and Nikos Komodakis. 宽残差网络（Wide residual networks）. In BMVC, 2016. 1

[77] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: 超越经验风险最小化（Beyond empirical risk minimization）. arXiv preprint arXiv:1710.09412, 2017. 9

[78] Hang Zhang, Chongruo Wu, Zhongyue Zhang, Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun, Tong He, Jonas Mueller, R Manmatha, et al. ResNeSt: 分裂注意力网络（Split-attention networks）. arXiv preprint arXiv:2004.08955, 2020. 7, 8

[79] Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, and Stan Z Li. 通过自适应训练样本选择缩小基于锚框与无锚框检测之间的差距（Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection）. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9759–9768, 2020. 6, 9

[80] Hengshuang Zhao, Jiaya Jia, and Vladlen Koltun. 探索用于图像识别的自注意力机制（Exploring self-attention for image recognition）. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10076–10085, 2020. 3

[81] Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip HS Torr, et al. 从序列到序列的视角利用 Transformer 重新思考语义分割（Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers）. arXiv preprint arXiv:2012.15840, 2020. 2, 3, 7, 8

[82] Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang. 随机擦除数据增强（Random erasing data augmentation）. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 13001–13008, 2020. 9

[83] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. 通过 ADE20K 数据集对场景进行语义理解（Semantic understanding of scenes through the ade20k dataset）. International Journal on Computer Vision, 2018. 5, 7, 10

[84] Xizhou Zhu, Han Hu, Stephen Lin, and Jifeng Dai. 可变形卷积网络 v2：更形变，更优结果（Deformable convnets v2: More deformable, better results）. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 9308–9316, 2019. 1, 3

[85] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable DETR: 用于端到端目标检测的可变形 Transformer（Deformable transformers for end-to-end object detection）. In International Conference on Learning Representations, 2021. 3