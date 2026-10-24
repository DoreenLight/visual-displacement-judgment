# Transformer 即 SSM：基于结构化状态空间对偶性的广义模型与高效算法

Tri Dao<sup>∗1</sup> 和 Albert Gu<sup>∗2</sup>

普林斯顿大学计算机科学系（Department of Computer Science, Princeton University）

卡内基梅隆大学机器学习系（Machine Learning Department, Carnegie Mellon University） tri@tridao.me, agu@cs.cmu.edu

## 摘要

尽管 Transformer 一直是深度学习在语言建模领域取得成功背后的核心架构，但最近以 Mamba 为代表的状态空间模型（State-Space Models, SSM）已被证明在中小规模下能够媲美甚至超越 Transformer。我们证明了这两类模型族实际上存在着非常紧密的联系，并通过对一类经过深入研究的结构化半可分矩阵（semiseparable matrices）的多种分解，构建了连接 SSM 与各类注意力机制（Attention）变体的丰富理论框架。我们的状态空间对偶性（State Space Duality, SSD）框架使我们能够设计出一种全新的架构（Mamba-2）。其核心层是对 Mamba 选择性 SSM 的精简与改进，计算速度提升了 2 到 8 倍，同时在语言建模任务上持续保持对 Transformer 的强劲竞争力。

## 1 引言

Transformer（特别是以因果方式处理输入序列的仅解码器（decoder-only）模型，例如 GPT (Brown et al. 2020)、Llama (Touvron, Lavril, et al. 2023)）是现代深度学习取得成功的主要推动力之一。许多方法尝试近似核心注意力层以解决其效率问题 (Tay et al. 2022)，例如训练期间计算量随序列长度呈二次方增长，以及自回归生成期间需要与序列长度呈线性大小的缓存。与此同时，一类替代的序列模型——结构化状态空间模型（structured state-space models, SSM）脱颖而出，其在训练期间随序列长度呈线性扩展，且在生成期间保持恒定的状态大小。它们在长程任务中表现出强劲的性能（例如 S4 (Gu, Goel, and Ré 2022)），最近更是在中小规模的语言建模任务上达到或超越了 Transformer（例如 Mamba (Gu and Dao 2023)）。然而，SSM 的发展似乎与社区改进 Transformer 的集体努力脱节，例如在理论层面上理解 Transformer 以及在现代硬件上对其进行优化。因此，与 Transformer 相比，理解 SSM 并开展实验更加困难，且从算法和系统层面来看，实现与 Transformer 同样高效的 SSM 训练仍具挑战性。

我们的主要目标是在结构化 SSM（structured SSMs）与各种注意力机制变体之间建立丰富的理论联系。这将使我们能够把最初为 Transformer 开发的算法和系统级优化迁移到 SSM 上，从而实现构建性能超越 Transformer 且在序列长度上扩展效率更高的基础模型这一目标。该方向上的一项里程碑式贡献是线性注意力（Linear Attention, LA）框架（Katharopoulos et al. 2020），该框架通过展示二次核化注意力与特定线性循环之间的“对偶形式”等价性，推导出了自回归注意力与线性 RNN 之间的联系。这种对偶性带来了新的能力，例如兼具高效的可并行化训练和高效的自回归推理。基于相同的理念，本文提供了将线性复杂度 SSM 与二次复杂度形式相联系的多种视角，以结合 SSM 与注意力的优势。<sup>1</sup>

状态空间对偶（State Space Duality）。我们将连接结构化 SSM 与注意力变体的框架称为结构化状态空间对偶（structured state space duality, SSD），该框架是通过结构化矩阵的抽象建立的：即具有次二次（sub-quadratic）参数量和乘法复杂度的矩阵。我们开发了两个用于表示序列模型的广泛框架，一个基于矩阵变换，另一个基于张量缩并（tensor contractions），两者分别揭示了该对偶性的不同视角。我们的技术贡献包括：

• 我们证明了状态空间模型与一类被深入研究的结构化矩阵——半可分矩阵（semiseparable matrices）之间的等价性（第 3 节）。这种联系是我们框架的核心，揭示了 SSM 的新特性与新算法。本文的一个核心观点是，计算状态空间模型的不同方法可以被重构为结构化矩阵上的各种矩阵乘法算法。

• 我们显著改进了线性注意力（linear attention）（Katharopoulos et al. 2020）的理论。我们首先通过张量缩并的语言对其循环形式给出了深刻的证明，随后将其推广为一种全新的结构化掩码注意力（structured masked attention, SMA）族（第 4 节）。

• 我们将 SSM 与 SMA 联系起来，表明它们存在很大的交集且互为对偶，兼具类似 SSM 的线性形式和类似注意力的二次形式（第 5 节）。我们还证明了任何具有快速循环形式的核注意力方法都必然是 SSM。

![](images/603e0b6b6b59173d215c52f709bda358c8010c4342f961682b825a91d1d746fc.jpg)  
Figure 1: （结构化状态空间对偶性。）本文通过结构化矩阵这一桥梁，阐述了状态空间模型与注意力机制之间的关系。

除了其内在的理论价值之外，我们的框架还为理解和改进序列模型开辟了广泛的研究方向。

高效算法。首先也是最重要的一点，我们的框架揭示了用于计算 SSM 的新型高效且易于实现的算法（第 6 节）。我们提出了一种基于半可分矩阵块分解的新型 SSD 算法，该算法同时利用了线性 SSM 循环和二次对偶形式的优势，在所有主要效率维度（例如训练和推理计算量、内存占用以及在现代硬件上利用矩阵乘法单元的能力）上均取得了最优权衡。SSD 的专用实现比 Mamba 经过优化的选择性扫描实现快 2 至 8 倍，同时允许使用更大的循环状态尺寸（可达 Mamba 的 8 倍甚至更高，且速度几乎没有下降）。SSD 与经过优化的 softmax 注意力实现（FlashAttention-2 (Dao 2024)）相比极具竞争力，在序列长度为 2K 时性能实现交汇反超，并在序列长度为 16K 时提速达 6 倍。

架构设计。采用诸如 SSM 等新架构的一个主要障碍在于专为 Transformer 量身定制的生态系统，例如针对大规模训练的硬件高效优化和并行化技术。我们的框架允许利用已确立的注意力机制惯例和技术，为 SSM 构建架构设计选择的词汇表，并进一步改进它们（第 7 节）。例如，我们将多头注意力（MHA）中的“头”概念类比引入到 SSM 中。我们展示了 Mamba 架构是一种多输入 SSM（MIS），它在结构上类似于多值注意力（MVA），并对比了具有不同头结构的其他 Mamba 变体。

我们还利用这些思想对 Mamba 块进行了轻微修改，从而能够实现张量并行（例如采用 Megatron 风格（Shoeybi et al. 2019））。主要思想包括引入分组值注意力（GVA）头结构，并将所有数据依赖的投影移至块的起始位置并行进行。

修改后的并行 Mamba 块与使用 SSD 作为内部 SSM 层相结合，构成了 Mamba-2 架构。我们在与 Mamba 相同的设置下研究了 Mamba-2 的 Chinchilla 缩放法则，发现它在困惑度和实际运行时间（wall-clock time）两方面均帕累托占优（Pareto dominates）于 Mamba 和 Transformer++。我们还额外训练了一个系列的模型

不同规模的 Mamba-2 模型在 The Pile 数据集上的结果表明，它在标准下游评估中达到或超越了 Mamba 以及开源 Transformer 模型。例如，在 The Pile 上基于 300B token 训练的 2.7B 参数 Mamba-2，其表现超越了在同一数据集上训练的 Mamba-2.8B、Pythia-2.8B 甚至 Pythia-6.9B。

系统优化（Systems Optimizations）。SSD 框架将 SSM 与 Transformer 联系起来，使我们能够利用大量专为 Transformer 开发的系统优化成果（第 8 节）。

• 例如，张量并行（Tensor Parallelism, TP）是一种重要的模型并行技术，它通过将每一层拆分到同一节点内的多个 GPU 上来训练大型 Transformer 模型。我们将 Mamba-2 设计为对 TP 友好，使每个模块（block）的同步点数量减少了一半。

• 对于激活值无法放入单个设备中的超长序列，业界已针对注意力模块开发了序列并行（sequence parallelism）。我们阐述了如何通过在设备之间传递循环状态（recurrent states），对通用 SSM 以及特别是 Mamba-2 进行序列并行训练。

• 对于包含不同长度样本的微调任务，为了实现最佳效率，Transformer 需要复杂的技术来移除填充标记（padding tokens）并在变长序列上执行注意力计算。我们展示了 Mamba-2 如何高效地使用可变序列长度进行训练，且无需任何填充标记。

第 9 节在语言建模、训练效率以及一项困难的多查询联想记忆（multi-query associative recall）任务（Arora, Eyuboglu, Zhang, et al. 2024）上对 Mamba-2 进行了实证验证。最后，在第 10 节中，我们提供了扩展的相关工作，并讨论了我们的框架所开辟的潜在研究方向。

模型代码和预训练检查点开源于 https://github.com/state-spaces/mamba 。

## 2 背景与概述

## 2.1 结构化状态空间模型

结构化状态空间序列模型（S4）是近年来用于深度学习的一类序列模型，与 RNN、CNN 以及经典状态空间模型广泛相关。它们受到特定连续系统 (1) 的启发，该系统通过隐式潜在状态 $\bar { h } \in \mathbb { R } ^ { ( \top , \mathsf { N } ) }$ 将一维序列 $x \in \mathbb { R } ^ { \intercal } \mapsto y \in \mathbb { R } ^ { \intercal }$ 进行映射。

结构化 SSM 的通用离散形式如下公式 (1) 所示。

$$
\begin{array}{l} h _ {t} = A h _ {t - 1} + B x _ {t} \\ y _ {t} = C ^ {\top} h _ {t} \end{array}\tag{1a}
$$

(1b)

$$
h _ {t} = A _ {t} h _ {t - 1} + B _ {t} x _ {t}\tag{2a}
$$

$$
y _ {t} = C _ {t} ^ {\top} h _ {t}\tag{2b}
$$

其中 $A \in \mathbb { R } ^ { ( \mathsf { N } , \mathsf { N } ) } , B \in \mathbb { R } ^ { ( \mathsf { N } , 1 ) } , C \in \mathbb { R } ^ { ( \mathsf { N } , 1 ) }$。结构化 SSM 之所以得名，是因为控制时间动态特性的 $A$ 矩阵必须具备特定的结构，才能足够高效地计算这种序列到序列的变换，从而应用于深度神经网络中。最初引入的结构是对角加低秩（DPLR）（Gu, Goel, and Ré 2022）和对角结构（Gu, Gupta, et al. 2022; Gupta, Gu, and Berant 2022; J. T. Smith, Warrington, and Linderman 2023），后者至今仍然是最常用的结构。

在这项工作中，我们使用术语状态空间模型（state space model, SSM）来指代结构化 SSM。这类 SSM 有多种形式，与连续时间、循环和卷积模型等几种主要的神经序列模型范式有着深厚的联系（Gu, Johnson, Goel, et al. 2021）。我们在下文提供了一个简要概述，更多背景和细节请参见先前的研究（Gu 2023; Gu and Dao 2023）。

连续时间模型（Continuous-time Models）。原始的结构化 SSM 最初是作为函数 $x ( t ) \in \mathbb { R } \mapsto$ <sub></sub>() ∈ R 上的连续时间映射，而不是直接对序列进行操作。在连续时间的视角下，公式 (1a) 中的矩阵 $( A , B )$ 并不是直接学习得到的，而是由底层参数 $( { \mathring \cal A } , { \mathring \cal B } )$ 以及参数化的步长 Δ 生成的。“连续参数” $( \Delta , \mathring { A } , \mathring { B } )$ 通过固定公式 $A = f _ { A } ( \Delta , \mathring { A } )$ 和 $B = f _ { B } ( \Delta , \mathring { B } )$ 转换为“离散参数” (, )，其中配对 $( f _ { A } , f _ { B } )$ 被称为离散化规则。

注 1（Remark 1）。虽然我们的主要模型采用了与先前工作相同的参数化和离散化步骤（详见 Gu and Dao (2023)），但为了简化阐述和符号表示，我们在本文的其余部分中将其省略。我们注意到，先前关于结构化 SSM 的工作将连续参数 $( { \mathring \cal A } , { \mathring \cal B } )$ 和离散参数 (, ) 分别记为 (, ) 和 (,<sup>¯</sup> <sup>¯</sup>)；我们更改了记号以简化表述，并直接聚焦于主导 SSM 主要递推过程的离散参数。

循环模型（Recurrent Models）。公式 (1) 和 (2) 呈现为其输入  上的线性递推形式。因此，结构化 SSM 可以被视为循环神经网络（recurrent neural networks, RNN）的一种形式，其中线性赋予了它们额外的性质，并使它们能够避免传统 RNN 的顺序计算。相反，尽管进行了这种简化，SSM 作为序列变换（在通用近似的意义上）仍然具有完全的表达能力（Kaul 2020; Orvieto et al. 2023; Shida Wan<sub>g</sub> and Xue 2023）。

卷积模型。当 SSM 的动力学过程随时间保持恒定时（如公式 (1) 所示），该模型被称为线性时不变（LTI）模型。在这种情况下，它们等价于卷积。因此，SSM 也可以被视为某种类型的 CNN，但其中 (i) 卷积核通过 SSM 参数 $( A , B , C )$ 进行隐式参数化，且 (ii) 卷积核通常是全局的而非局部的。反之，根据经典信号处理理论，所有性质足够良好的卷积都可以表示为 SSM。

通常，以往的 LTI SSM 会使用卷积模式来实现高效的可并行化训练（此时整个输入序列是预先已知的），并切换到循环模式 (1) 来实现高效的自回归推理（此时每次只输入一个步长）。

选择性状态空间模型。形式 (2) 中参数 $(A, B, C)$ 也可以随时间变化，这在 Mamba 中作为选择性 SSM（selective SSM）被引入。与标准的 LTI 公式 (1) 相比，该模型可以在每个时间步选择性地关注或忽略输入。研究表明，在语言等信息密集型数据上，该模型的表现远优于 LTI SSM，尤其是随着其状态维度 $N$ 的增大允许了更大的信息容量。然而，它只能以循环模式而非卷积模式进行计算，并且需要精心的硬件感知（hardware-aware）实现才能达到高效。即便如此，它仍然不如 CNN 和 Transformer 等硬件友好型模型高效，因为它没有利用现代加速器（如 GPU 和 TPU）所专门针对的矩阵乘法单元。

虽然时不变 SSM 与连续、循环和卷积序列模型密切相关，但它们与注意力机制（attention）并没有直接关联。在本文中，我们揭示了选择性 SSM 与注意力机制之间更深层次的联系，并利用这种联系显著提升了 SSM 的训练速度，同时允许使用大得多的状态维度 N。

## 作为序列变换的结构化 SSM

定义 2.1。我们使用术语序列变换（sequence transformation）来指代序列上的参数化映射 $Y = f _ { \theta } ( X )$，其中 $X , Y \in \mathbb { R } ^ { ( \mathsf { T } , \mathsf { P } ) }$，且  为任意参数集合。T 表示序列或时间轴；下标索引第一维，例如 $X _ { t } , Y _ { t } \in \mathbb { R } ^ { \mathsf { P } }$。

序列变换（例如 SSM 或自注意力机制）是深度序列模型的基石，它们被融入到神经网络架构（例如 Transformer）中。式 (1) 或 (2) 中的 SSM 是一个 ${ \mathsf { P } } = 1 ;$ 的序列变换；通过简单地沿该维度进行广播，它可以推广到 $\mathsf { P } > 1$（换言之，将输入视为 P 个独立的序列并对每个序列应用 SSM）。人们可以将 P 视为头维度（head dimension），我们将在第 7 节中对此进行详细阐述。

定义 2.2。我们将 SSM 算子 $\mathsf { S S M } ( A , B , C ) \ : = \ : \mathsf { S S M } ( A _ { 0 : T } , B _ { 0 : T } , C _ { 0 : T } )$ 定义为由方程 (2) 所给出的序列变换 $X \in$ $\mathbb { R } ^ { ( \mathsf { T } , \mathsf { P } ) } \mapsto Y \in \mathbb { R } ^ { ( \mathsf { T } , \mathsf { P } ) }$。

在 SSM 中，N 维度是一个自由参数，称为状态大小（state size）或状态维度（state dimension）。我们还称其为状态扩展因子（state expansion factor），因为它将输入/输出的大小扩展了  倍，这对这些模型的计算效率具有重要影响。

最后，我们指出，许多类型的序列变换（例如注意力机制）都可以表示为沿序列维度的单个矩阵乘法。

定义 2.3。如果序列变换 $Y = f _ { \theta } ( X )$ 可以写成 $Y = M _ { \theta } X$ 的形式，其中  是一个依赖于参数  的矩阵，则我们称其为矩阵变换（matrix transformation）。我们将该序列变换与矩阵  等同起来，并且在上下文中含义明确时，通常会省略对  的依赖。

## 2.2 注意力机制 (Attention)

广义上的注意力（Attention）是指一种为序列中每一对位置分配分数的计算类型，它允许每个元素“关注”（attend）其余元素。迄今为止，最常见且最重要的注意力变体是 softmax 自注意力（softmax self-attention），其可以定义为

$$
Y = \operatorname{softmax} (Q K ^ {\top}) \cdot V
$$

其中 $Q , K , V \in \mathbb { R } ^ { ( \mathsf { T } , \mathsf { P } ) }$。这种成对比较的机制（由具象化/实例化 $Q K ^ { \top }$ 产生）导致了注意力机制特有的二次方训练成本。

人们已经提出了注意力机制的许多变体，但它们都共享这些注意力分数的底层核心，并伴有各种近似（Tay et al. 2022）。对本工作而言最重要的变体是线性注意力（linear attention, Katharopoulos et al. 2020）。大致来说，这类方法通过将 softmax 折叠进核特征映射中来省去 softmax，并利用矩阵乘法的结合律将其重写为 $( Q K ^ { \top } ) \cdot { \bar { V } } = Q \cdot ( K ^ { \top } V )$。此外，在因果（自回归）注意力的重要情形中，他们表明当因果掩码作为 $( L \circ Q K ^ { \intercal } ) \cdot V$ 融入左侧时（其中 $L$ 为全 1 下三角矩阵），右侧可以展开为一个递推形式。近期的多项相关及同期工作，如 RetNet（Y. Sun et al. 2023）和 GateLoop（Katsch 2023），将此推广到了 $L$ 的更一般形式（第 10 节）。在本工作中，我们对结构化掩码注意力（structured masked attention）的形式化表述将极大地泛化这些思想。

## 2.3 结构化矩阵

一般矩阵 $M \in \mathbb { R } ^ { ( \mathsf { T } , \mathsf { T } ) }$ 需要 ${ \mathsf { T } } ^ { 2 }$ 个参数来表示，并需要 $O ( \mathsf { T } ^ { 2 } )$ 的时间来执行如矩阵-向量乘法等基本操作。结构化矩阵是指那些

(i) 能通过压缩表示以次二次（subquadratic，理想情况下为线性）参数量进行表示，且

(ii) 通过直接在此压缩表示上进行操作，拥有快速算法（最重要的是矩阵乘法）。

也许最典型的结构化矩阵族是稀疏矩阵和低秩矩阵。然而，还存在许多其他矩阵族，例如托普利茨（Toeplitz）、柯西（Cauchy）、范德蒙（Vandermonde）和蝶形（butterfly）矩阵，它们都已在机器学习中被用于构建高效模型（Dao, Gu, et al. 2019; D. Fu et al. 2024; Gu, Gupta, et al. 2022; Thomas et al. 2018）。结构化矩阵是实现高效表示与算法的强大抽象工具。在这项工作中，我们将展示 SSM 等价于另一类此前未在深度学习中使用过的结构化矩阵，并利用这种联系推导出高效的方法和算法。

## 2.4 概述：结构化状态空间对偶性

尽管本文构建了状态空间模型（SSMs）、注意力机制与结构化矩阵之间更为丰富的联系框架，我们在此简要总结主要方法，该方法实际上是相当自成体系的，且在算法上十分简单。

循环（线性）形式。状态空间对偶（SSD）层可以被定义为选择性 SSM（2）的一个特例。可以应用 SSM 作为循环（或并行扫描）的标准计算，其在序列长度上具有线性复杂度。与 Mamba 中使用的版本相比，SSD 有两个微小差异：

• $A$ 上的结构进一步从对角矩阵简化为标量乘单位矩阵结构。在这种情况下，每个 $A_t$ 也可以仅用一个标量来表示。

• 与 Mamba 中使用的 $\mathsf { P } = 1$ 相比，我们使用了更大的头维度 $\mathsf { P }$。通常选择 $\mathsf { P } = \{ 6 4 , 1 2 8 \}$，这与现代 Transformer 的惯例相似。

与原始的选择性 SSM 相比，这些改动可以被视为略微降低了表达能力，以换取训练效率的显著提升。特别是，我们的新算法将允许在现代加速器上使用矩阵乘法单元。

对偶（二次）形式。SSD 的对偶形式是与注意力密切相关的二次计算，定义为

$$
(L \circ Q K ^ {\top}) \cdot V \qquad L _ {i j} = \left\{ \begin{array}{l l} a _ {i} \times \dots \times a _ {j + 1} & i \geq j \\ 0 & i <   j \end{array} \right.
$$

其中 $a _ { i }$ 是有界于 [0, 1] 的输入相关标量。

与标准 softmax 注意力相比，存在两个主要差异：

• 舍弃了 softmax。

• 注意力矩阵与一个额外的掩码矩阵 $L$ 进行逐元素相乘。

这两项改动都可以看作是在解决标准注意力（vanilla attention）中的问题。例如，最近观察到 softmax 会在注意力分数中引发问题，比如“注意力汇聚”（attention sink）现象（Darcet et al. 2024; Xiao et al. 2024）。更重要的是，掩码矩阵 $L$ 可以被视为用一种不同的、依赖于数据的位置掩码取代了 Transformer 的启发式位置嵌入，该掩码控制着跨时间传递的信息量。

更广泛地说，这种形式是我们在线性注意力基础上推广的结构化掩码注意力（structured masked attention）的一个特例，在第 4 节中进行了定义。

矩阵形式与 SSD 算法。通过展示 SSM 对于依赖于 $\theta = ( A , B , C )$ 的矩阵 $M _ { \theta } \in \bar { \mathbb { R } } ^ { ( \top , \top ) }$ 具有矩阵变换形式 $Y = M X$，SSD 的各种形式通过统一的矩阵表示联系在一起。特别地，SSD 的对偶形式等价于与矩阵 $M$ 的朴素（平方时间复杂度）相乘，而循环形式则是一种利用 $M$ 中的结构的特定高效（线性时间复杂度）算法。

除这些之外，任何与 $M$ 相乘的算法都可以应用。我们提出的硬件高效 SSD 算法（第 6 节）是一种新的结构化矩阵乘法方法，涉及 $M$ 的块分解，相比纯线性或平方形式获得了更好的效率权衡。与通用选择性 SSM（Gu and Dao 2023）相比，它相对简单且易于实现；Listing 1 用几行代码提供了一个完整的实现。

Figure 1 简要展示了本文中提出的概念之间关系的路线图。

## 2.5 符号表示

在整篇论文中，我们倾向于使用能够直接映射到代码的精确符号表示。

矩阵与向量。我们通常使用小写字母表示向量（即具有单个轴的张量），使用大写字母表示矩阵（即具有多个轴的张量）。在本文中，我们不对矩阵进行加粗。有时，如果一个矩阵沿着某个轴绑定或重复（因此也可以视为向量），我们可能会使用大写或小写字母来表示 $\mathrm { i t . } ^ { 2 }$。· 表示标量乘法或矩阵乘法，而 ◦ 表示 Hadamard（逐元素）乘法。

索引。我们使用 Python 风格的索引，$\mathbf { e . g . } i : j$ 在 $i < j$ 时表示范围 $( i , i + 1 , \ldots , j - 1 )$，在 $i > j$ 时表示 $( i , i - 1 , \ldots , j + 1 )$。例如，对于任意符号 ，我们令 $j \geq i$ 时的 $\boldsymbol { v } _ { j : i }$ 表示序列 $\big ( v _ { j } , \ldots , v _ { i + 1 } \big )$。[] 等价于 $0 : i = ( 0 , \ldots , i - 1 )$。为简便起见，我们还令 $\boldsymbol { v } _ { j : i } ^ { \times }$ 表示乘积 $v _ { j } \times \cdots \times v _ { i + 1 } . \stackrel { ? } { \cdot }$。

维度。为了与矩阵和张量进行区分，我们通常使用打字机字体的大写字母 $( \mathbf { e . g . \ D , N , \top } )$ 来表示维度和张量形状。我们经常使用 $M \in \mathbb { R } ^ { ( \mathsf { T } , \mathsf { T } ) }$ 来反映代码中的张量形状，而不是传统的符号 $\boldsymbol { M } \in \mathbb { R } ^ { T \times T }$。

张量缩并。为了清晰起见，并作为阐述和证明我们结果的核心工具，我们将大量依赖张量缩并（tensor contraction）或 einsum 记号。我们假定读者熟悉这种符号，它在 numpy 等现代张量库中被广泛使用。例如，我们可以使用 contract(MN, NK → MK) 来表示矩阵-矩阵乘法运算符，并且在我们的记号中，contract(MN, NK → MK)(, )（等价于 $X \cdot Y )$）可以转换为代码 numpy.einsum(<sup>′</sup>mn, nk → mk<sup>′</sup>, X, Y)。

附录 A 中包含了一份详尽的符号词汇表。

## 3 状态空间模型即结构化矩阵

本节探讨了状态空间模型作为序列变换的不同视角，并概述了此类映射的性质与算法。本节的主要结果涉及状态空间模型与一类被称为半可分矩阵（semiseparable matrices）的结构化矩阵之间的等价性，这带来了新的效率结果（定理 3.5 和 3.7）。

## 3.1 状态空间模型的矩阵变换形式

回顾我们对 SSM 的定义，它是通过 (2) 定义的参数化映射。我们的理论框架首先是将这一变换简单地写作矩阵乘法，将向量 $x \in \mathbb { R } ^ { \mathsf { T } } \mapsto y \in \mathbb { R } ^ { \mathsf { T } }$ 进行映射。

根据定义，$h _ { 0 } = B _ { 0 } x _ { 0 }$。通过数学归纳法，

$$
\begin{array}{l} h _ {t} = A _ {t} \dots A _ {1} B _ {0} x _ {0} + A _ {t} \dots A _ {2} B _ {1} x _ {1} + \dots + A _ {t} A _ {t - 1} B _ {t - 2} x _ {t - 2} + A _ {t} B _ {t - 1} x _ {t - 1} + B _ {t} x _ {t} \\ = \sum_ {s = 0} ^ {t} A _ {t: s} ^ {\times} B _ {s} x _ {s}. \end{array}
$$

两边同乘以 $C _ { t }$ 以得到 $y _ { t }$，并在 $t \in [ \top ]$ 上对该等式进行向量化，我们便推导出了 SSM 的矩阵变换形式。

$$
\begin{array}{c} y _ {t} = \sum_ {s = 0} ^ {t} C _ {t} ^ {\top} A _ {t: s} ^ {\times} B _ {s} x _ {s} \\ y = \mathsf {S S M} (A, B, C) (x) = M x \\ M _ {j i} := C _ {j} ^ {\top} A _ {j} \dots A _ {i + 1} B _ {i} \end{array}\tag{3}
$$

## 3.2 半可分矩阵

公式 (3) 中的  是一类被称为半可分矩阵（semiseparable matrices）的特殊表示。半可分矩阵是一种基本的矩阵结构。我们首先定义这些矩阵及其性质。

定义 3.1。如果一个（下三角）矩阵  包含在下三角部分（即对角线上或下方）中的每个子矩阵的秩至多为 $N$，则该矩阵是 $N$-半可分的（N-semiseparable）。我们称 $N$ 为该半可分矩阵的阶（order）或秩（rank）。

定义 3.1 以及其他形式的相关“可分”结构（例如准可分矩阵（quasiseparable matrices）和其他半可分矩阵的定义）有时被称为结构秩矩阵（structured rank matrices，或秩结构矩阵 rank-structured matrices），因为它们的特征在于其子矩阵上的秩条件。半可分矩阵具有许多结构化表示，包括分层半可分（hierarchical semiseparable, HSS）、顺序半可分（sequentially semiseparable, SSS）以及 Bruhat 形式（Pernet and Storjohann 2018）。我们将主要使用 SSS 形式。

## 3.2.1 顺序半可分（SSS）表示

定义 3.2。如果一个下三角矩阵 $M \in \mathbb { R } ^ { ( \mathsf { T } , \mathsf { T } ) }$ 可以写成如下形式，则它具有 $N$-顺序半可分（sequentially semiseparable, SSS）表示：

$$
M _ {j i} = C _ {j} ^ {\top} A _ {j} \cdot \cdot \cdot A _ {i + 1} B _ {i}\tag{4}
$$

其中向量 $B _ { 0 } , \dots , B _ { \intercal - 1 } , C _ { 0 } , \dots , C _ { \intercal - 1 } \in \mathbb { R } ^ { \mathsf { N } }$，矩阵 $A _ { 0 } , \dots , A _ { \intercal - 1 } \in \mathbb { R } ^ { ( \mathsf { N } , \mathsf { N } ) }$。

我们定义算子 SSS，使得 $M = { \sf S S S } ( A _ { 0 : \mathrm { T } } , B _ { 0 : \mathrm { T } } , C _ { 0 : \mathrm { T } } )$。

半可分矩阵的一个基本结果是，它们与具有 SSS 表示的矩阵完全等价。其中一个方向可以通过简单的构造性证明推导得出。

引理 3.3。具有表示 (4) 的 $N$-SSS 矩阵  是 $N$-半可分的。

证明。考虑任意非对角块 $M _ { j : j ^ { \prime } , i ^ { \prime } : i }$，其中 $j ^ { \prime } > j \ge i > i ^ { \prime }$。它具有显式的秩-N（rank-N）分解：

$$
\left[ \begin{array}{c c c} C _ {j} ^ {\top} A _ {j: i ^ {\prime}} ^ {\times} B _ {i ^ {\prime}} & \ldots & C _ {j} ^ {\top} A _ {j: i - 1} ^ {\times} B _ {i - 1} \\ \vdots & & \vdots \\ C _ {j ^ {\prime} - 1} ^ {\top} A _ {j ^ {\prime} - 1: i ^ {\prime}} ^ {\times} B _ {i ^ {\prime}} & \ldots & C _ {j ^ {\prime} - 1} ^ {\top} A _ {j ^ {\prime} - 1: i - 1} ^ {\times} B _ {i - 1} \end{array} \right] = \left[ \begin{array}{c} C _ {j} ^ {\top} A _ {j: j} ^ {\times} \\ \vdots \\ C _ {j ^ {\prime} - 1} ^ {\top} A _ {j ^ {\prime} - 1: j} ^ {\times} \end{array} \right] A _ {j: i - 1} ^ {\times} \left[ \begin{array}{c c c} A _ {i - 1: i ^ {\prime}} ^ {\times} B _ {i ^ {\prime}} & \dots & A _ {i - 1: i - 1} ^ {\times} B _ {i - 1} \end{array} \right].\tag{5}
$$

公式 (5) 将在推导我们针对序列模型的快速算法中被广泛使用。另一个方向在半可分矩阵（semiseparable matrices）的文献中已有充分论证。

命题 3.4。每个 N-半可分矩阵都具有 N-SSS 表示。

此外，需要注意的是，尽管定义 3.2 在该表示中涉及 $O ( \mathsf { N } ^ { 2 } \mathsf { T } )$ 个参数（特别是用于存储相关矩阵），但它实际上可以压缩至 $O(\mathrm{NT})$ 个参数，这在渐近意义上是紧致的（Pernet, Signargout, and Villard 2023）。因此，在本文的其余部分中，我们将混用这种结构化矩阵类（定义 3.1）与其特定的表示形式（定义 3.2）；我们将始终使用这种表示形式，而非其他候选形式。相应地，我们将使用 N-SSS 来指代 SSS 形式的 N-半可分矩阵。

半可分矩阵是一种基本的矩阵结构，具有许多重要性质。它们在宏观上与递推关系密切相关，并且可以通过多种表征（例如定义 3.1 和 3.2）进行定义，这些表征揭示了针对它们的不同联系和高效算法。我们在附录 C.1 中提及了它们的其他一些性质。

注 2。半可分性（semiseparability）的概念非常广泛，文献中出现了许多相似但略有不同的定义；我们的定义可能与其他惯例略有出入。首先，由于本文主要关注因果（causal）或自回归（autoregressive）设定，我们将半可分性的定义限制在三角矩阵情形；定义 3.1 在某些作者那里可能被更正式地称为 (N, 0)-半可分性。一些作者也可能将其称为拟可分性（quasiseparability）的一种形式（Eidelman and Gohberg 1999; Pernet 2016）。简要综述参见 Vandebril et al. (2005)。

## 3.2.2 1-半可分矩阵：标量 SSM 递推

我们将特别讨论 1-SS 矩阵这一特殊情况。请注意，在这种情况下，$C _ { j }$ 和 $B _ { i }$ 均为标量，可以从 SSS 表示 (4) 中分解提取出来（在此我们也使用小写字母以强调该情况下的参数为标量）

$$
\operatorname{SSS} (a, b, c) = \operatorname{diag} (c) \cdot M \cdot \operatorname{diag} (b) \quad \text { where } \quad M _ {j i} = a _ {j: i} ^ {\times}.
$$

由于对角矩阵易于处理（例如，与对角矩阵相乘等价于逐元素的标量乘法），我们可以忽略这些项。因此，我们对 1-SS 矩阵的基本表示为 ${ M _ { j i } = a _ { j : i } }$ 或

$$
M = 1 \mathrm{SS} (a _ {0: T}) := \left[ \begin{array}{c c c c c} 1 & & & & \\ a _ {1} & 1 & & & \\ a _ {2} a _ {1} & a _ {2} & 1 & & \\ \vdots & \vdots & \ddots & \ddots & \\ a _ {T - 1} \ldots a _ {1} & a _ {T - 1} \ldots a _ {2} & \ldots & a _ {T - 1} & 1 \end{array} \right].\tag{6}
$$

1-SS 矩阵的重要性在于它们等价于标量递推的最小形式——即状态维度 N = 1 且无 $( B , C )$ 投影的退化 SSM 情形。请注意，乘法 $y = M x$ 可以通过以下递推计算

$$
\begin{array}{r l} & y _ {t} = a _ {t: 0} x _ {0} + \dots + a _ {t: t} x _ {t} \\ & \quad = a _ {t} (a _ {t - 1: 0} x _ {0} + \dots + a _ {t - 1: t - 1} x _ {t - 1}) + a _ {t: t} x _ {t} \\ & \quad = a _ {t} y _ {t - 1} + x _ {t}. \end{array}\tag{7}
$$

![](images/18c8f37e0f96d63d39783d3c53347c1f7be79f50ac0555e7d473817605fa955e.jpg)  
Figure 2：（状态空间模型是半可分矩阵。）作为序列变换，状态空间模型可以表示为作用于序列维度 T 上的矩阵变换 $M \in \overline { { \mathbb { R } } } ^ { ( \mathsf { T } , \mathsf { T } ) }$，在一个头（head）内的每个通道之间共享同一个矩阵（左图）。该矩阵是一个半可分矩阵（右图），即一种秩结构矩阵，其中包含在对角线及其下方的每个子矩阵（蓝色）的秩至多为 N，等于 SSM 的状态维度。

因此，我们也把与 1-SS 矩阵的矩阵乘法称为标量 SSM 递推或 cumprodsum（累积积和，cumulative product sum；即累积积和累积和的推广）算子。作为递推的基本形式，与 1-SS 矩阵的乘法是我们核心算法的重要构建块（building block）。

我们强调，本文的核心主题之一是序列模型上的许多算法都可以归约为结构化矩阵乘法算法。1-SS 矩阵充分体现了这种联系：计算原始标量递推或 cumprodsum 算子存在许多快速算法，而它们都被证明等价于 1-SS 矩阵的不同结构化分解。我们在附录 B（Appendix B）中专门讨论了这些用于 1-SS 矩阵乘法的算法。

## 3.3 状态空间模型是半可分矩阵

回想一下，我们对 SSM 的定义是通过定义 2.1（Definition 2.1）给出的参数化映射。SSM 与半可分矩阵之间的联系，只需直接将该变换写成映射向量 $x \mapsto y \in \mathbb { R } ^ { \mathsf { T } }$ 的矩阵乘法即可得出。

公式 (3) 直接建立了状态空间模型与顺序半可分（sequentially semiseparable）表示之间的联系，而后者通常又等价于半可分矩阵（引理 3.3 与命题 3.4）。

定理 3.5。状态大小为 N 的状态空间模型变换 $y = { \mathsf { S S M } } ( A , B , C ) ( x )$ 等同于在顺序半可分表示下与 N-SS 矩阵的矩阵乘法 $y = { \mathrm { S S S } } ( A , B , C ) \cdot x .$

换句话说，序列变换算子 SSM（定义 2.2）与矩阵构造算子 SSS（定义 3.2）是一致的，我们可将它们互换使用（有时也简写为 SS）。此外——命运弄人的是——结构化状态空间模型（structured state space models）与顺序半可分矩阵（sequentially semiseparable matrices）具有相同的首字母缩写，这进一步强调了它们的等价性！方便的是，我们可以互换使用这些缩写词中的任意一个：SSM（状态空间模型或半可分矩阵）、SSS（结构化状态空间或顺序半可分）或 SS（状态空间或半可分），均可明确地指代任一概念。然而，我们通常遵循以下约定：SSM 指状态空间模型，SS 指半可分，而 SSS 指顺序半可分。

图 2 说明了将状态空间模型视为半可分矩阵的序列变换视角。

## 3.4 通过结构化矩阵算法计算状态空间模型

定理 3.5 之所以重要，是因为它使我们能够将高效计算 SSM（以及其他序列模型）的问题，规约为结构化矩阵乘法的高效算法。我们在此简要提供一个概述，并将在第 4 节和第 5 节展示 SSM 与其他序列模型的等价性之后，于第 6 节给出我们的主要新算法。

如前所述，半可分矩阵（即秩结构矩阵）是一类经典的结构化矩阵：

(i) 它们具有压缩表示，例如 SSS 形式仅有 $O(T)$ 个参数，而非 $O ( \mathsf { T } ^ { 2 } )$ 个参数。

(ii) 它们具有直接在压缩表示上操作的快速算法。

此外，参数化和矩阵乘法成本在半可分阶数上可以是紧凑的。

命题 3.6 (Pernet, Signargout, and Villard (2023))。大小为 T 的 N-SS 矩阵可以用 (NT) 个参数表示，且具有在时间和空间复杂度为 (NT) 的矩阵-向量乘法。

例如，1-SS 矩阵阐明了这种联系的本质。矩阵 $ { M } = 1  { \mathrm { S } }  { \mathrm { S } } (  { a } )$ 正好由 $\top - 1$ 个参数 $a _ { 0 : \mathsf { T } - 1 } = a _ { 1 } , \dotsc , a _ { \mathsf { T } - 1 }$ 定义，并且可以通过遵循标量递推 (7) 在 (T) 时间内计算。

## 3.4.1 线性（循环）模式

在对角结构化 SSM（S4D (Gu, Gupta, et al. 2022)）的情况下，只需利用状态空间模型表述 (2) 并展开递推，即可轻松看出命题 3.6。我们在 (8) 中提供了形式化的张量缩并算法，其中维度 S 等于 T<sup>4</sup>。

$$
Z = \operatorname{contract} (\mathrm{SP}, \mathrm{SN} \rightarrow \mathrm{SPN}) (X, B)\tag{S, P, N}
$$

$$
H = \operatorname{contract} (\mathrm{TSN}, \mathrm{SPN} \rightarrow \mathrm{TPN}) (L, Z)\tag{T, P, N}
$$

(8b)

$$
Y = \operatorname{contract} (\mathrm{TN}, \mathrm{TPN} \rightarrow \mathrm{TP}) (C, H)\tag{T, P}
$$

(8c)

这里，$L \in \mathbb { R } ^ { ( \mathsf { T } , \mathsf { T } ) }$ 定义为 1SS()，换句话说，对于 $i \in [ \mathsf { N } ]$，有 $L _ { 0 : \mathsf { T } , 0 : \mathsf { T } } = 1 \mathsf { S S } ( A _ { 0 : \mathsf { T } } )$。该算法涉及与 (2) 对应的三个步骤：

(i) 通过输入矩阵  扩展输入  (8a)，

(ii) 展开独立的标量 SSM 递推 (8b)，以及

(iii) 通过输出矩阵  缩并隐藏状态  (8c)。

注意，我们在步骤 (8b) 中使用了标量 SSM 与 1-SS 矩阵之间的等价性。

注 3。我们注意到 (8) 是 Mamba (S6) 模型的一个特例。然而，朴素实现会很慢，因为存在大小为 (T, P, N) 的扩展张量  和 ；Gu and Dao (2023) 引入了一种硬件感知实现来避免具体实例化这些张量。

令人惊讶的是，定理 3.5 和命题 3.6 直接表明所有 SSM 都具有与算法 (8) 相同的渐近效率。

定理 3.7。对于序列长度为 $T$、状态大小为 $N$ 的任意状态空间模型（定义 2.2），均可在 $O(TN)$ 时间内计算完成（不考虑潜在的预处理）。

我们注意到，这一结果在结构化 SSM 文献中是全新的。特别是，给定稠密非结构化的 $A _ { t }$ 矩阵，仅总表示大小似乎就达到 $O(TN^2)$。因此，定理 3.7 得出了一个非平凡的结论：通过预处理步骤，即使是非结构化 SSM 也能以最优效率进行计算，其上限与由 $B$ 和 $C$ 的大小所给出的下限 $O(TN)$ 相匹配。

备注 4。鉴于 $\mathbb { R } ^ { ( \mathsf { N } , \mathsf { N } ) }$ 上的几乎所有稠密矩阵在 $\mathbb{C}$ 上都是可对角化的，从而导致几乎所有稠密实数 SSM 都等价于对角复数 SSM 这一事实，定理 3.7 或许并不太令人意外。这一事实构成了对角 SSM 成为最受欢迎的结构化 SSM 形式的底层原因 (Gu, Gupta, et al. 2022; Gupta, Gu, and Berant 2022; J. T. Smith, Warrington, and Linderman 2023)。然而，定理 3.7 对所有实数 SSM（而不仅仅是可对角化的 SSM）以及其他数域（包括 $\mathbb{C}$ 本身）上的稠密 SSM 蕴含了强得多的结果。

在实践中，高效可计算的 SSM 仍然需要对 $A$ 施加额外的结构，特别是为了避免开销高昂的预处理步骤（该步骤既需要 $N$ 阶的额外 FLOPs，又涉及诸如奇异值分解等硬件效率低下的操作）。这些结构是以往关于结构化 SSM 的工作（例如 S4(D) 和 Mamba）以及我们新算法的研究重点。特别地，当对 $A$ 施加稍强的结构时，我们将在第 6 节中通过 SSM 矩阵 $M = { \mathrm { S S S } } ( A , B , C )$ 的分块分解来设计极具硬件效率的算法。

## 3.4.2 二次（朴素）模式

我们注意到，由我们新的矩阵视角所揭示的，还有另一种计算 SSM 的方法。矩阵 SSM 表示 (3) 的朴素计算仅涉及显式实例化（materializing）序列变换矩阵 $M = { \mathrm { S S S } } ( A , B , C )$。这是一个 (T, T) 矩阵，因此该朴素算法在序列长度上呈二次方扩展。然而，当序列长度 T 较短时，由于常数因子以及计算模式对硬件的友好性（例如，利用矩阵-矩阵乘法），该方法实际上可能比线性算法更高效。事实上，对于结构化 SSM 的某种特定情况，这看起来与二次复杂度注意力计算非常相似（第 5 节）。

## 3.4.3 总结

许多序列模型都是明确地基于矩阵序列变换而被构想或定义的——最显著的是 Transformer，其中矩阵混合器（matrix mixer）即为注意力矩阵。另一方面，RNN 和 SSM 此前尚未以这种方式被描述过。通过提供状态空间模型的显式矩阵变换形式，我们揭示了理解和使用它们的新方法。从计算的角度来看，计算状态空间模型前向传播的任何方法都可以被视为半可分矩阵（semiseparable matrices）上的矩阵乘法算法。半可分矩阵视角为状态空间对偶性（state space duality, SSD）提供了一个视角，其中对偶模式分别指线性时间半可分矩阵乘法算法与二次时间朴素矩阵乘法。

此外，利用半可分矩阵的丰富结构可以带来更优秀的算法和更多见解（例如第 6 节和附录 B）。在附录 C.1 中，我们描述了半可分矩阵的一些其他性质。

## 4 结构化掩码注意力：利用结构化矩阵推广线性注意力

在本节中，我们从第一性原理重新审视线性注意力（linear attention）框架。本节的主要结果是基于张量缩并（tensor contraction）对线性注意力的简单证明（命题 4.1），以及我们在定义 4.2 中对结构化掩码注意力（structured masked attention）的广义抽象。我们指出，本节从不同于状态空间模型（state space models）的方向推导了主要的对偶性结果，并且完全可以独立于第 3 节进行阅读。

• 第 4.1 节建立了我们针对各类注意力变体的框架，重点关注核注意力（kernel attention）和掩码核注意力（masked kernel attention）。

• 第 4.2 节给出了我们的第一个主要注意力结果：从张量缩并的视角对线性注意力给出的一个简单证明。

• 第 4.3 节定义了结构化掩码注意力，这是我们通过结构化矩阵对先前注意力变体所作的推广。

## 4.1 注意力框架

## 4.1.1 注意力

（单头）注意力的基本形式是作用于三个向量序列上的映射 $( Q , K , V ) \mapsto Y$

$$
\begin{array}{l l} Q = \text {input} & (\mathsf {T}, \mathsf {N}) \\ K = \text {input} & (\mathsf {S}, \mathsf {N}) \\ V = \text {input} & (\mathsf {S}, \mathsf {P}) \\ G = Q K ^ {\top} & (\mathsf {T}, \mathsf {S}) \\ M = f (G) & (\mathsf {T}, \mathsf {S}) \\ Y = G V & (\mathsf {T}, \mathsf {P}) \end{array}\tag{9}
$$

我们使用“形状注释”来表示张量的维度，例如 $Q \in \mathbb { R } ^ { ( \mathsf { T } , \mathsf { N } ) }$。在这一通用形式中，$\mathsf{S}$ 和 $\mathsf{T}$ 分别表示源序列和目标序列的长度，$\mathsf{N}$ 表示特征维度，而 $\mathsf{P}$ 表示头维度（head dimension）。

softmax 注意力最常见的变体使用 softmax 激活函数 $f = { \tt s o f t m a x }$ 来归一化  矩阵的各行。

## 4.1.2 自注意力（Self-Attention）

我们的论述源于自注意力最重要的情况，其中

(i) 源序列和目标序列相同 $( \mathrm { i } . \mathrm { e } . \mathsf { S } = \mathsf { T } )$

(ii) 通常特征维度与头维度相同 $( \mathrm { i . e . } \ N = \mathsf { P } )$

(iii) 且 , ,  是通过对同一输入向量进行线性投影生成的 $( Q = W _ { Q } \cdot X , K = W _ { K } \cdot X , V = W _ { V } \cdot X ) .$

然而，我们的表述抽象掉了这些选择，直接从 , ,  矩阵出发。

Remark 5. 我们的重点是头维度和特征维度相等的自注意力情况（即 S = T 且 N = P），这应当作为贯穿全文的示例。我们定义注意力的通用形式，不仅是为了使我们的框架能够涵盖交叉注意力（cross-attention）等变体，还因为将维度的符号（例如 S 和 T）分开，能够使本节中关于主要结果的张量缩并符号证明更加清晰。

Remark 6. 虽然注意力通常被表述为对这三个被对称看待的输入 , ,  进行的操作，但 (9) 中的输入和输出维度表明事实并非如此。特别是，输出中并不存在特征维度 N；因此在 S = T 的情况下（例如自注意力），我们将  视为主要输入，从而使 (9) 定义了一个良定义的序列变换  ↦→ （定义 2.1）。

## 4.1.3 核注意力（Kernel Attention）

将 softmax 函数应用于 Gram 矩阵  的步骤可以分解为两个部分：

1. 对  矩阵进行指数运算。

2. 在 S 轴上归一化  矩阵。

我们目前可以忽略归一化项，因为这仅相当于传入  = 1 并进行除法运算（我们将在 7.3 节重新讨论这一点）。指数运算项可以视为核变换：存在一个（无限维的）特征映射 ，使得 $\exp ( { \cal Q K } ^ { \top } ) = \varphi ( { \cal Q } ) \varphi ( { \cal K } ) ^ { \top }$ 。通过将特征映射抽象到  和  本身的定义中（即定义 ,  为变换后的版本），我们可以忽略 softmax 变换，并假设 ,  是由核特征映射任意生成的，且可能存在 N ≠ P。

已经提出了核注意力（kernel attention）的许多具体实现，包括：

• 原始的 Linear Attention（Katharopoulos et al. 2020）将核特征映射定义为任意逐点激活函数，例如 $x \mapsto 1 + \mathsf { e l u } ( x )$。

• Random Feature Attention（RFA）（H. Peng et al. 2021）通过高斯核的随机傅里叶特征近似（Rahimi and Recht 2007）来选择核特征映射以近似 softmax 注意力（即 $\exp$ 特征映射）。这涉及随机投影（即通过随机投影矩阵乘以  和  并应用激活函数 $x \mapsto ( \cos ( x ) , \sin ( x ) )$）。

• Performer（Choromanski et al. 2021）提出了基于正正交随机特征的快速注意力（FAVOR+）。其中正随机特征（PRF）部分选择的核特征映射为先进行随机投影，随后应用特征映射  ↦→ $2 ^ { - 1 / 2 } ( \exp ( x ) , \exp ( - x ) )$。做出这种选择的出发点是使核元素取正值，并在理论上可证明能够近似 softmax 注意力。[该方法还提议在正交方向上选择随机投影，对此我们不予考虑。]

• cosFormer（Qin, Weixuan Sun, et al. 2022）通过余弦重加权机制对 RFA 进行了增强，该机制结合了位置信息以强调局部性。这实际上是将 $Q _ { t } , K _ { t }$ 输入特征映射  ↦→ $( x \cos ( \pi t / 2 T ) , \sin ( \pi t / 2 T ) )$。

• Linear Randomized Attention（Zheng, C. Wang, and Kong 2022）从重要性采样的角度推广了 RFA，并对其进行泛化以提供对完整 softmax 核（而不仅仅是经 exp 变换的分子）更好的估计。

其他相关的注意力变体包括 Linformer（Sinong Wang et al. 2020）和 Nyströformer（Xiong et al. 2021），二者分别通过随机投影（Johnson-Lindenstrauss）和核近似（Nyström 方法）来使用注意力矩阵  的低秩近似（因此与公式 (9) 兼容）。

## 4.1.4 掩码（核）注意力

设 $L$ 为一个形状为 (T, S) 的掩码。最常见的情况是，在 $S = T$ 的自回归自注意力场景中，$L$ 可以是一个由 1 组成的下三角矩阵，表示因果掩码（causal mask）。除了强制因果性之外，还可以应用许多其他类型的掩码——特别是出于降低稠密注意力复杂度而采用的各种稀疏模式，例如带状（banded）、空洞/扩张（dilated）或块对角（block diagonal）掩码。

掩码注意力通常用矩阵形式表示为

$$
y = \left(L \circ (Q K ^ {\top})\right) \cdot V.\tag{10}
$$

更精确地，附带形状标注并将其分解为精确的计算序列如下：

$$
\begin{array}{c c} G = Q K ^ {\top} & (\mathsf {T}, \mathsf {S}) \\ M = G \circ L & (\mathsf {T}, \mathsf {S}) \\ Y = M V & (\mathsf {T}, \mathsf {P}) \end{array}\tag{11}
$$

我们在本节中对注意力变体的改进推导始于注意到该公式可以写为一个单独的张量缩并（contraction）：

$$
Y = \text { contract } (\text { TN }, \text { SN }, \text { SP }, \text { TS } \to \text { TP }) (Q, K, V, L)\tag{12}
$$

并且 (11) 中的算法可以重新表述为通过两两缩并的特定顺序来计算 (12)

$$
G = \operatorname{contract} (\mathrm{TN}, \mathrm{SN} \rightarrow \mathrm{TS}) (Q, K)\tag{T, S}
$$

(13a)

$$
M = \operatorname{contract} (\mathrm{TS}, \mathrm{TS} \rightarrow \mathrm{TS}) (G, L)\tag{T, S}
$$

(13b)

$$
Y = \operatorname{contract} (\mathrm{TS}, \mathrm{SP} \rightarrow \mathrm{TP}) (M, V)\tag{T, P}
$$

(13c)

## 4.2 线性注意力

线性注意力（Linear attention）以及许多其他高效注意力（efficient attention）变体，通常是通过改变核心注意力计算中的矩阵结合律顺序 $( Q K ^ { \top } ) V = Q ( K ^ { \top } V )$ 来实现的。然而，当加入掩码（mask）时，其推导就变得不那么显而易见了（例如，原始论文 (Katharopoulos et al. 2020) 及其变体 (Y. Sun et al. 2023) 均在没有给出证明的情况下直接给出了公式）。

大致来说，线性注意力方法声称以下公式与 (10) 等价，而这必须通过展开求和并仔细追踪下标索引来进行验证。

$$
Y = Q \cdot \operatorname{cumsum} (K ^ {\top} V)\tag{14}
$$

命题 4.1 ((Katharopoulos et al. 2020))。自回归核注意力（Autoregressive kernel attention），即带有因果掩码的掩码核注意力，可以通过每步耗费常数时间的递推在 () 时间内计算完成。

## 4.2.1 线性注意力的张量缩约证明

我们给出了线性注意力的一种简单且严格的推导，这也将直接揭示如何对其进行推广。其核心思想是以另一种顺序执行缩约 (12)。我们避免使用容易混淆的矩阵表示法，而是直接使用张量缩约表示法：

$$
Z = \operatorname{contract} (\mathrm{SP}, \mathrm{SN} \rightarrow \mathrm{SPN}) (V, K)\tag{S, P, N}
$$

(15a)

$$
H = \operatorname{contract} (\mathrm{TS}, \mathrm{SPN} \rightarrow \mathrm{TPN}) (L, Z)\tag{T, P, N}
$$

(15b)

$$
Y = \operatorname{contract} (\mathrm{TN}, \mathrm{TPN} \rightarrow \mathrm{TP}) (Q, H)\tag{T, P}
$$

(15c)

直观上，我们对该缩约顺序作如下解释。

第一步 (15a) 执行向更多特征的“扩张”（expansion），扩张的倍数为特征维度 N。第三步 (15c) 则将扩张的特征维度缩约消除。如果将  视为输入（Remark 6），那么  和  则分别执行扩张与缩约。

第二步是最关键的，它解释了线性注意力（linear attention）的线性部分。首先注意到 (15b) 只是通过  进行的直接矩阵乘法（因为 (P, N) 轴可以被展平）。还要注意，这是唯一同时涉及 T 和 S 轴的项，因此应具有 $\Omega(TS)$ 复杂度（即在序列长度上呈二次方关系）。然而，当掩码  是标准因果注意力掩码（下三角全为 1）时，与  的矩阵-向量乘法等价于特征维度的累加和（cumulative sum）

$$
y = \left[ \begin{array}{c c c} 1 & & \\ \vdots & \ddots & \\ 1 & \ldots & 1 \end{array} \right] x \quad \Longleftrightarrow \quad \begin{array}{l} y _ {0} = x _ {0} \\ y _ {t} = y _ {t - 1} + x _ {t} \end{array} .
$$

## 4.3 结构化掩码注意力

从掩码注意力 (15) 的张量缩约视角来看，我们立刻可以看出原始线性注意力的关键在于：与因果掩码的矩阵-向量乘法等价于累加和算子。

然而，我们观察到并没有任何理由要求注意力掩码必须全为 1。线性注意力要实现快速计算，所必需的只是让  成为一个结构化矩阵，根据定义，结构化矩阵即那些具有快速矩阵乘法的矩阵（第 2.3 节）。特别地，我们可以使用任何具有次二次（理想情况下为线性）矩阵-向量乘法的掩码矩阵 ，通过加速瓶颈方程 (15b)，它将具有与标准线性注意力相同的复杂度。

定义 4.2。结构化掩码注意力（Structured Masked Attention，SMA，或简称为结构化注意力）被定义为一个作用于查询/键/值（queries/keys/values）, ,  以及任意结构化矩阵 （即具有次二次矩阵乘法）上的函数，通过 4 路张量缩约表示为

$$
Y = \operatorname{contract} (\mathrm{TN}, \mathrm{SN}, \mathrm{SP}, \mathrm{TS} \rightarrow \mathrm{TP}) (Q, K, V, L).
$$

SMA 二次方模式算法是由 (13) 定义的成对缩约序列，对应于标准（掩码）注意力计算。

SMA 线性模式算法是由 (15) 定义的成对缩约序列，其中步骤 (15b) 通过次二次结构化矩阵乘法进行了优化。

我们可以将结构化掩码注意力实例化为任意给定的矩阵结构类别。一些示例包括（Figure 3）：

• 线性注意力使用因果掩码。

• RetNet (Y. Sun et al. 2023) 采用衰减掩码 $L _ { i j } = \gamma ^ { i - j } \cdot \mathbb { I } [ j \geq i ]$，其中 $\gamma \in \left[ 0 , 1 \right]$ 为某个衰减因子。
• 该衰减掩码可以推广为托普利茨矩阵（Toeplitz matrix）$L _ { i j } = \alpha _ { i - j }$，其中 $\alpha \in \mathbb { R } ^ { \mathsf { T } }$ 是一组可学习（或依赖于输入）的参数。这可以解释为一种相对位置编码形式，类似于 AliBi (Press, N. Smith, and Lewis 2022) 等其他方法，但这里是乘性而非加性的。

![](images/6379c24a6f3c1076a53edd9720d3d07f4b14ff80b1a0d7392096defc913e162b.jpg)  
Figure 3: （结构化掩码注意力。）SMA 针对任意结构化矩阵 $L$ 构造掩码注意力矩阵 $M = \boldsymbol { Q } \boldsymbol { K } ^ { \intercal } \circ \boldsymbol { L }$，从而定义了矩阵序列变换 $Y = M V$。通过不同的缩并顺序，并结合与 $L$ 的高效结构化矩阵乘法，所有 SMA 实例都具有对偶的亚二次形式。先前的范例包括线性注意力（Linear Attention, Katharopoulos et al. 2020）和 RetNet (Y. Sun et al. 2023)。除了本文重点研究的 SSD（1-半可分 SMA）之外，结构化注意力还存在许多其他潜在的实例化形式。

• 另一种变体可以使用傅里叶矩阵（Fourier matrix）$L _ { i j } = \omega ^ { i j / \top }$ 以不同的方式编码位置结构。

在第 5 节中，我们考虑半可分 SMA（semiseparable SMA），它定义了我们主要的 SSD 模型。

## 4.3.1 总结：掩码注意力的对偶形式

标准（掩码核）注意力机制往往在函数与算法之间被混淆。理清这一区别提供了一种理解注意力机制不同变体的清晰途径。

• 我们将掩码注意力视为一个特定的函数 (12)。

• 标准的二次注意力计算 (13) 可以被视为计算该函数的一种算法。

• 线性注意力 (15) 则是计算同一函数的另一种替代算法。

此外，在这种情况下

• 掩码注意力函数仅仅是针对四个项的一个特定缩并（contraction）。

• 二次和线性注意力算法仅仅是执行该缩并的两种不同顺序。

众所周知，缩并顺序会对计算复杂度产生巨大差异，从而导致了二次与线性的分立。正如状态空间模型（State Space Models）是一种可以通过多种方式计算的变换，并具有二次与线性的对偶形式（第 3.4 节）一样，线性注意力也具有源自两种缩并顺序的类似对偶性。事实上，它们是对同一个底层对偶性的不同视角，我们将在第 5 节中对其进行明确阐述。

## 5 状态空间对偶性

在第 3 节和第 4 节中，我们定义了结构化状态空间模型（structured state space models）和结构化注意力（structured attention），讨论了它们的性质，并表明它们都具有二次算法和线性算法。本节将它们联系起来。我们的主要结果是证明了结构化状态空间模型的一个特例与结构化注意力的一个特例相重合，并且线性时间 SSM 算法与二次时间核注意力（kernel attention）算法互为对偶形式。

• 第 5.1 节将状态空间模型特化为标量结构，其中朴素的二次计算可被视为核注意力的一个实例。

• 第 5.2 节将结构化掩码注意力（SMA）特化为半可分 SMA，刻画了具有高效自回归特性的掩码注意力。

• 第 5.3 节总结了结构化掩码注意力与结构化状态空间模型之间的联系，称之为结构化状态空间对偶性（structured state space duality）。

## 5.1 标量-恒等结构化状态空间模型

在第 3 节中，我们证明了状态空间模型等价于半可分矩阵变换，从而同时具有线性循环形式和二次朴素形式。

回想一下，SSM 定义为 $y = { \mathsf { S S M } } ( A , B , C ) ( x )$，并且 SSM 的矩阵形式采用 SSS（时序半可分，sequentially semiseparable）表示 $M = { \cal S } { \cal S } { \cal S } ( A , B , C )$，其中 $M _ { j i } = C _ { j } ^ { \top } A _ { j : i } B _ { i }$（公式 (3)）。

现在让我们考虑 $A_j$ 仅为一个标量的情况；换句话说，这是结构化 SSM 的一种实例化，其中 $A$ 矩阵具有极强的结构：对于标量 $a$ 和单位矩阵 $I$，有 $A = a I$。那么我们可以将其重排为

$$
M _ {j i} = A _ {j: i} \cdot (C _ {j} ^ {\top} B _ {i}).
$$

并且这可以被向量化为

$$
\begin{array}{c} L := 1 \text {SS} (a) \\ M = L \circ (C B ^ {\top}) \end{array}
$$

其中 $B , C \in \mathbb { R } ^ { ( \mathsf { T } , \mathsf { N } ) }$

使用这种表述形式，完整输出 $Y = M X$ 的精确计算方式为

$$
G = \operatorname{contract} (\mathsf {T N}, \mathsf {S N} \to \mathsf {T S}) (C, B)\tag{T, S}
$$

$$
M = \operatorname{contract} (\mathrm{TS}, \mathrm{TS} \rightarrow \mathrm{TS}) (G, L)\tag{T, S}
$$

(16)

$$
Y = \operatorname{contract} (\mathrm{TS}, \mathrm{SP} \rightarrow \mathrm{TP}) (M, X)\tag{T, P}
$$

其中 ${ \mathsf S } = { \mathsf T }$。但这与掩码核注意力（masked kernel attention）的原始定义 (13) 完全相同！

因此，正如第 3.4 节所提及的，朴素地计算标量结构化 SSM——即显式实例化（materialize）半可分矩阵 $M$ 并执行二次复杂度的矩阵-向量乘法——与二次掩码核注意力完全相同。

## 5.2 1-半可分结构化掩码注意力

结构化掩码注意力（structured masked attention）允许使用任何结构化掩码 $M$。当 $M$ 为因果掩码（causal mask）时，它就是标准的线性注意力（linear attention）。请注意，因果掩码为 $L = { \sf S S } ( 1 _ { T } )$，即 1-SS 掩码在定义 (6) 中由 $a _ { t } = 1$ 生成。这启发我们将 $M$ 推广到 1-半可分掩码（1-semiseparable masks）类，即 1-半可分结构化掩码注意力（1-SS SMA），其中线性注意力循环中的累积和（cumsum）被更一般的循环——标量 SSM 扫描（scalar SSM scan），即 1-半可分矩阵乘法（第 3.2.2 节）所取代。

最后，我们考虑 1-半可分 SMA 最重要的原因是计算它的线性形式是对角状态空间模型（diagonal state space model）的一个特例。SMA 的线性形式为算法 (15)，其中瓶颈步骤 (15b)

可以视为与 1-SS 掩码的矩阵乘法。在第 3 节中，我们还写出了对角 SSM (8) 的计算过程，其中瓶颈步骤 (8b) 是等价于 1-SS 乘法的标量 SSM 循环。唯一的区别在于，由于矩阵 $A$ 是大小为 $N$ 的对角矩阵，(8b) 中的 $A$ 多出了一个 $N$ 维度。如果 $A$ 的所有对角元都相同，则该 $N$ 维度将会消失，从而得出推论 5.1。

推论 5.1。 1-SS SMA（具有 1-半可分结构化矩阵 $M$ 的掩码注意力）(15) 是对角 SSM (8) 的一个特例，其中对角矩阵是单位矩阵的标量倍数。

虽然推论 5.1 表明 1-SS SMA 具有高效的循环形式，但我们也可以展示一个逆向结果，以刻画 SMA 的哪些实例具有高效的自回归（autoregression）形式。

定理 5.2。对于作为具有有界阶数的自回归过程的结构化掩码注意力（Structured Masked Attention，Definition 4.2）的任意实例化，其结构化掩码  必然是一个半可分矩阵（semiseparable matrix）。

换句话说，高效自回归注意力就是一般的半可分 SMA。定理 5.2 的证明见附录 C.2。

注 7。虽然 1-半可分 SMA 是状态空间模型（state space model）的一个特例，但一般的半可分 SMA 严格比 1-SS SMA 具有更强的表达能力，且无法由标准 SSM 描述。然而，与  的半可分乘法以及 SMA 的线性形式（公式 (15a)）各自都包含一个展开和收缩步骤，并且可以通过单次（更大的）展开被吸纳进一个类似的 1-SS SMA 实例中。

综上所述，1-半可分结构化注意力是 SMA 中最重要的情形，因为它是：

• 具有输入依赖循环（input-dependent recurrence）的线性注意力的自然推广。

• 一般半可分注意力的最简情形，等价于高效自回归注意力。

• 对角状态空间模型（diagonal state space model）的一个特例。

## 5.3 结构化状态空间对偶性（Structured State-Space Duality，SSD）

总结我们的结果：

• 结构化状态空间模型（第 3 节）通常是通过线性时间循环定义的模型。然而，通过展开表征其线性序列到序列变换的矩阵公式，可以推导出一种二次形式。

• 注意力变体（第 4 节）是通过二次时间成对交互定义的模型。然而，通过将其视为四路张量收缩并以不同的顺序进行规约，可以推导出一种线性形式。

• 每种模型的一个自然特例——更确切地说，在 $A$ 矩阵上具有标量-单位阵结构的状态空间模型（state space models），以及在其 $M$ 掩码上具有 1-半分离结构（1-semiseparable structure）的结构化掩码注意力（structured masked attention）——彼此互为对偶，具有完全相同的线性形式和二次形式。

Figure 4 总结了这两种表示之间的对偶性。

扩展的相关工作与讨论（第 10 节）更详细地描述了 SSD 与通用 SSM / 注意力机制之间的关系。

## 6 面向 SSD 模型的硬件高效算法

在 SSM、注意力机制和结构化矩阵之间建立理论 SSD 框架的好处在于，可以利用这些关联来改进模型和算法。在本节中，我们展示了如何从计算结构化矩阵乘法的各种算法中推导出高效计算 SSD 模型的各种算法。

我们的主要计算成果是一种用于计算 SSD 模型的算法，该算法结合了线性（循环）模式和二次（注意力）模式。该算法既具有与 SSM 相当的计算效率（在序列长度上呈线性扩展），又具备与注意力机制相当的硬件友好性（主要采用矩阵乘法）。

Theorem 6.1. 考虑一个状态扩展因子为 $N$ 且头维度为 $P = N$ 的 SSD 模型。存在一种在任意输入 $X \in \mathbb { R } ^ { ( \mathsf { T } , \mathsf { P } ) }$ 上计算该模型的算法，该算法仅需要 $\mathcal{O}(TN^2)$ 训练 FLOPs、$\mathcal{O}(TN)$ 推理 FLOPs、$\mathcal{O}(N^2)$ 推理内存，且其计算主要由矩阵乘法主导。

<table><tr><td colspan="2">结构化状态空间模型 (Structured State Space Model)</td><td colspan="2">结构化掩码注意力 (Structured Masked Attention)</td></tr><tr><td>C</td><td>(收缩矩阵)</td><td>Q</td><td>(查询 / queries)</td></tr><tr><td>B</td><td>(扩展矩阵)</td><td>K</td><td>(键 / keys)</td></tr><tr><td>X</td><td>(输入序列)</td><td>V</td><td>(值 / values)</td></tr><tr><td> $A_{j:i}$ </td><td>(状态矩阵)</td><td> $L_{ji}$ </td><td>(掩码)</td></tr><tr><td>N</td><td>(状态扩展维度)</td><td>N</td><td>(核特征维度)</td></tr><tr><td>H</td><td>(隐藏状态 (8b))</td><td></td><td>SMA 线性对偶 (15)</td></tr><tr><td>=L·XB</td><td>(线性模式)</td><td></td><td></td></tr><tr><td rowspan="2" colspan="2">SSM 二次对偶 (16)</td><td>G</td><td>(Gram 矩阵 (13a))</td></tr><tr><td>=Q·K $^{\top}$ </td><td>(二次模式)</td></tr></table>

![](images/a916596294ba8749a8ad99b94e902819aea89d14dbe2fead5dbfc294f3187a51.jpg)  
Figure 4: (结构化状态空间对偶性。) 状态空间对偶性描述了状态空间模型与掩码注意力之间的密切关系。$( L e f t )$ 通用 SSM 与 SMA 均具有线性形式和二次形式，且在符号表示上具有直接对应关系。(Right) SSM 与 SMA 在一大类状态空间对偶模型（SSD）中相交，该模型将许多序列模型作为特例涵盖在内。

请注意，所有这些界限都是紧致的，因为状态扩展为 N 且在头大小为 N 上运行的状态空间模型具有 $\mathsf { N } ^ { 2 }$ 的总状态大小（从而分别给出了 $O ( \mathsf { T N } ^ { 2 } )$ 和 $O ( { \mathsf N } ^ { 2 } )$ 的训练和推理 FLOPs 下界）。此外，输入  本身包含 TN 个元素，从而给出了内存下界。

定理 6.1 背后的核心思想，是再次将计算状态空间模型的问题视为半可分矩阵乘法，但以一种全新的方式利用其结构。我们不再是在循环模式或注意力模式下计算整个矩阵，而是对矩阵进行分块分解。对角块可以使用对偶注意力模式进行计算，这可以通过矩阵乘法高效完成；而非对角块则可以利用半可分矩阵的秩结构进行因子分解，并约简为更小规模的循环。我们重点指出，Listing 1 提供了 SSD 算法的自包含实现。与 Gu and Dao (2023) 的通用选择性 SSM（selective SSM）相比，该实现要简单得多，并且即使在原生 PyTorch 中无需特殊的底层算子（low-level kernels）也相对高效。

首先，对于某个块大小 Q，我们将矩阵划分为大小为 ${ \sf Q } \times { \sf Q }$ 的子矩阵构成的 $\textstyle { \frac { \intercal } { 0 } } \times { \frac { \intercal } { 0 } }$ 网格。注意，根据半可分矩阵的定义性质（定义 $3 . 1 ) . ^ { 5 }$），非对角块是低秩的。

$$
\begin{array}{l} \text {(Block Decomposition)} \quad M = \left[ \begin{array}{c c c c} M ^ {(0, 0)} & & & \\ M ^ {(1, 0)} & M ^ {(1, 1)} & & \\ \vdots & \vdots & \ddots & \\ M ^ {(\mathsf {T} / \mathsf {Q} - 1, 0)} & M ^ {(\mathsf {T} / \mathsf {Q} - 1, 1)} & \ldots & M ^ {(\mathsf {T} / \mathsf {Q} - 1, \mathsf {T} / \mathsf {Q} - 1)} \end{array} \right] \\ \text {(Diagonal Block)} \quad M ^ {(j, j)} = \operatorname{SSM} (A _ {j \mathbb {Q}: (j + 1) \mathbb {Q}}, B _ {j \mathbb {Q}: (j + 1) \mathbb {Q}}, C _ {j \mathbb {Q}: (j + 1) \mathbb {Q}}) \\ \text {(Low - Rank Block)} \quad M ^ {(j, i)} = \left[ \begin{array}{c} C _ {j \mathbb {Q}} ^ {\top} A _ {j \mathbb {Q}: j \mathbb {Q} - 1} \\ \vdots \\ C _ {(j + 1) \mathbb {Q} - 1} ^ {\top} A _ {(j + 1) \mathbb {Q} - 1: j \mathbb {Q} - 1} \end{array} \right] A _ {j \mathbb {Q} - 1: (i + 1) \mathbb {Q} - 1} \left[ \begin{array}{c} B _ {i \mathbb {Q}} ^ {\top} A _ {(i + 1) \mathbb {Q} - 1: i \mathbb {Q}} \\ \vdots \\ B _ {(i + 1) \mathbb {Q} - 1} ^ {\top} A _ {(i + 1) \mathbb {Q} - 1: (i + 1) \mathbb {Q} - 1} \end{array} \right] ^ {\top} \end{array}
$$

这通过一个示例最容易说明，例如对于 ${\mathsf T} = 9$ 并分解为长度为 $Q = 3$ 的块。阴影

单元格是半可分矩阵（semiseparable matrix）非对角块的低秩分解。

<table><tr><td rowspan="5"> $M =$ </td><td> $\begin{bmatrix} C_{0}^{\top} A_{0:0} B_{0} \\ C_{1}^{\top} A_{1:0} B_{0} & C_{1}^{\top} A_{1:1} B_{1} \\ C_{2}^{\top} A_{2:0} B_{0} & C_{2}^{\top} A_{2:1} B_{1} & C_{2}^{\top} A_{2:2} B_{2} \end{bmatrix}$ </td><td colspan="3"></td><td colspan="3"></td></tr><tr><td rowspan="2"> $\begin{bmatrix} C_{3}^{\top} A_{3:0} B_{0} & C_{3}^{\top} A_{3:1} B_{1} & C_{3}^{\top} A_{3:2} B_{2} \\ C_{4}^{\top} A_{4:0} B_{0} & C_{4}^{\top} A_{4:1} B_{1} & C_{4}^{\top} A_{4:2} B_{2} \\ C_{5}^{\top} A_{5:0} B_{0} & C_{5}^{\top} A_{5:1} B_{1} & C_{5}^{\top} A_{5:2} B_{2} \end{bmatrix}$ </td><td> $C_{3}^{\top} A_{3:3} B_{3}$ </td><td colspan="2"></td><td colspan="3"></td></tr><tr><td> $C_{4}^{\top} A_{4:3} B_{3} & C_{4}^{\top} A_{4:4} B_{4} \\ C_{5}^{\top} A_{5:3} B_{3} & C_{5}^{\top} A_{5:4} B_{4} & C_{5}^{\top} A_{5:5} B_{5} \end{bmatrix}$ </td><td colspan="2"></td><td colspan="3"></td></tr><tr><td rowspan="2"> $\begin{bmatrix} C_{6}^{\top} A_{6:0} B_{0} & C_{6}^{\top} A_{6:1} B_{1} & C_{6}^{\top} A_{6:2} B_{2} \\ C_{7}^{\top} A_{7:0} B_{0} & C_{7}^{\top} A_{7:1} B_{1} & C_{7}^{\top} A_{7:2} B_{2} \\ C_{8}^{\top} A_{8:0} B_{0} & C_{8}^{\top} A_{8:1} B_{1} & C_{8}^{\top} A_{8:2} B_{2} \end{bmatrix}$ </td><td rowspan="2"> $C_{6}^{\top} A_{6:3} B_{3} & C_{6}^{\top} A_{6:4} B_{4} & C_{6}^{\top} A_{6:5} B_{5} \\ C_{7}^{\top} A_{7:3} B_{3} & C_{7}^{\top} A_{7:4} B_{4} & C_{7}^{\top} A_{7:5} B_{5} \\ C_{8}^{\top} A_{8:3} B_{3} & C_{8}^{\top} A_{8:4} B_{4} & C_{8}^{\top} A_{8:5} B_{5} \end{bmatrix}$ </td><td> $C_{6}^{\top} A_{6:6} B_{6}$ </td><td colspan="3"></td><td></td></tr><tr><td> $C_{7}^{\top} A_{7:6} B_{6} & C_{7}^{\top} A_{7:7} B_{7} \\ C_{8}^{\top} A_{8:6} B_{6} & C_{8}^{\top} A_{8:7} B_{7} & C_{8}^{\top} A_{8:8} B_{8} \end{bmatrix}$ </td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5"> $=$ </td><td> $\begin{bmatrix} C_{0}^{\top} A_{0:0} B_{0} \\ C_{1}^{\top} A_{1:0} B_{0} & C_{1}^{\top} A_{1:1} B_{1} \\ C_{2}^{\top} A_{2:0} B_{0} & C_{2}^{\top} A_{2:2} B_{1} & C_{2}^{\top} A_{2:2} B_{2} \end{bmatrix}$ </td><td colspan="3"></td><td colspan="3"></td></tr><tr><td rowspan="2"> $\begin{bmatrix} C_{3}^{\top} A_{3:2} \\ C_{4}^{\top} A_{4:2} \\ C_{5}^{\top} A_{5:2} \end{bmatrix} A_{2:2} \begin{bmatrix} B_{0}^{\top} A_{2:0} \\ B_{1}^{\top} A_{2:1} \\ B_{2}^{\top} A_{2:2} \end{bmatrix} ^{\top}$ </td><td> $C_{3}^{\top} A_{3:3} B_{3}$ </td><td colspan="2"></td><td colspan="3"></td></tr><tr><td> $C_{4}^{\top} A_{4:3} B_{3} & C_{4}^{\top} A_{4:4} B_{4} \\ C_{5}^{\top} A_{5:3} B_{3} & C_{5}^{\ top } A_{5:4} B_{4} & C_{5}^{\top } A_{5:5} B_{5} \end{bmatrix}$ </td><td colspan="2"></td><td colspan="3"></td></tr><tr><td rowspan="2"> $\begin{bmatrix} C_{6}^{\top} A_{6:5} \\ C_{7}^{\top} A_{7:5} \\ C_{8}^{\top} A_{8:5} \end{bmatrix} A_{5:2} \begin{bmatrix} B_{0}^{\top} A_{2:0} \\ B_{1}^{\top} A_{2:1} \\ B_{2}^{\top} A_{2:2} \end{bmatrix} ^{\top}$ </td><td rowspan="2"> $\begin{bmatrix} C_{6}^{\top} A_{6:5} \\ C_{7}^{\top} A_{7:5} \\ C_{8}^{\top} A_{8:5} \end{bmatrix} A_{5:5} \begin{bmatrix} B_{3}^{\top} A_{5:3} \\ B_{4}^{\top} A_{5:4} \\ B_{5}^{\top } A_{5:5} \end{bmatrix} ^{\top}$ </td><td> $C_{6}^{\top} A_{6:6} B_{6}$ </td><td colspan="3"></td><td></td></tr><tr><td> $C_{7}^{\top} A_{7:6} B_{6} & C_{7}^{\top }A_{7:7}B_{7} \\ C_{8}^{\top }A_{8:6}B_{6} & C_{8}^{\top }A_{8:7}B_{7} & C_{8}^{\top }A_{8:8}B_{8} \end{bmatrix}$ </td><td></td><td></td><td></td><td></td></tr></table>

由此，我们可以将该问题简化为这两个部分。这也可以解释为将一个“块（chunk）”的输出 $y _ { j 0 : ( j + 1 )  Q }$ 分解为两个组成部分：块内输入 $x _ { j 0 : ( j + 1 ) \ Q } \ /$ 的影响，以及块之前输入 $x _ { 0 : j \ell } .$ 的影响。

## 6.1 对角块

对角块很容易处理，因为它们仅仅是更小尺寸的自相似问题。第  块表示计算范围 $R = j \boldsymbol { \mathbb { Q } } : ( j + 1 ) \boldsymbol { \mathbb { Q } } = ( j \boldsymbol { \mathbb { Q } } , j \boldsymbol { \mathbb { Q } } + 1 , . . . , j \boldsymbol { \mathbb { Q } } + \boldsymbol { \mathbb { Q } } - 1 )$ 的结果 $\mathsf { S S M } ( A _ { R } , B _ { R } , C _ { R } ) ( x _ { R } )$。关键在于该块可以使用任何所需的方法进行计算。特别是，对于较小的块长度 $Q$，使用对偶二次 SMA 形式可以更高效地计算该问题。此外，各块可以并行计算。

这些子问题可以解释为：假设（该块的）初始状态为 0，每个块的输出是什么。换句话说，对于块 ，这仅在考虑块输入 $x _ { j  Q : ( j + 1 ) Q }$ 的情况下计算正确的输出。

## 6.2 低秩块

低秩分解由 3 项组成，相应地计算也有三个部分。在此分解中，我们将使用术语

- 形如 $\begin{array} { r } { \left[ \begin{array} { l } { B _ { 0 } ^ { \top } A _ { 2 : 0 } } \\ { B _ { 1 } ^ { \top } A _ { 2 : 1 } } \\ { B _ { 2 } ^ { \top } A _ { 2 : 2 } } \end{array} \right] ^ { \top } } \end{array}$ ⊤ 的项被称为右因子（right factors）或 -块因子（-block-factors）。

- 形如 $A _ { 5 : 2 }$ 的项被称为中心因子（center factors）或 -块因子（-block-factors）。

- 形如 $\begin{array} { r } { \left[ C _ { 6 } ^ { \top } A _ { 6 : 5 } \right] } \\ { \left. C _ { 7 } ^ { \top } A _ { 7 : 5 } \right| } \\ { \left. C _ { 8 } ^ { \top } A _ { 8 : 5 } \right] } \end{array}$ 的项被称为左因子（left factors）或 -块因子（-block-factors）。

<table><tr><td> $\begin{bmatrix} C_{0}^{\top} A_{0:0} B_{0} \\ C_{1}^{\top} A_{1:0} B_{0} & C_{1}^{\top} A_{1:1} B_{1} \\ C_{2}^{\top} A_{2:0} B_{0} & C_{2}^{\top} A_{2:1} B_{1} & C_{2}^{\top} A_{2:2} B_{2} \end{bmatrix}$ </td><td></td><td></td></tr><tr><td></td><td> $\begin{bmatrix} C_{3}^{\top} A_{3:2} \\ C_{4}^{\top} A_{4:2} \\ C_{5}^{\top} A_{5:2} \end{bmatrix} \quad A_{2:2} \quad \begin{bmatrix} B_{0}^{\top} A_{2:0} \\ B_{1}^{\top} A_{2:1} \\ B_{2}^{\top} A_{2:2} \end{bmatrix} ^{\top}$ </td><td> $C_{3}^{\top} A_{3:3} B_{3}$  $C_{4}^{\top} A_{4:3} B_{3} \quad C_{4}^{\top} A_{4:4} B_{4}$  $C_{5}^{\top} A_{5:3} B_{3} \quad C_{5}^{\top} A_{5:4} B_{4} \quad C_{5}^{\top} A_{5:5} B_{5}$ </td></tr><tr><td></td><td> $\begin{bmatrix} C_{6}^{\top} A_{6:5} \\ C_{7}^{\top} A_{7:5} \\ C_{8}^{\top} A_{8:5} \end{bmatrix} \quad A_{5:2} \quad \begin{bmatrix} B_{0}^{\top} A_{2:0} \\ B_{1}^{\top} A_{2:1} \\ B_{2}^{\top} A_{2:2} \end{bmatrix} ^{\top}$ </td><td> $\begin{bmatrix} C_{6}^{\top} A_{6:5} \\ C_{7}^{\top} A_{7:5} \\ C_{8}^{\top} A_{8:5} \end{bmatrix} \quad A_{5:5} \quad \begin{bmatrix} B_{3}^{\top} A_{5:3} \\ B_{4}^{\top} A_{5:4} \\ B_{5}^{\top} A_{5:5} \end{bmatrix} ^{\top}$ </td></tr></table>

![](images/1e68377f9a8979e2afa84521a318dd0d2d59ab5e0d8c911367e8a167c13ebb97.jpg)  
Figure 5: （SSD 算法。）通过利用状态空间模型的矩阵变换视角将其表示为半可分矩阵（第 3 节），我们通过分块分解矩阵乘法算法为 SSD 模型开发了一种对硬件更高效的计算方式。该矩阵乘法也可以解释为一个状态空间模型，其中各块表示对输入和输出序列进行分块。对角块表示块内计算，非对角块表示块间计算，并通过 SSM 的隐藏状态进行因子分解。

右因子（Right Factors）。该步骤计算与低秩分解中右侧块因子的乘法。请注意，对于每个分块，这是一个 $(N, Q)$ 乘 $(Q, P)$ 的矩阵乘法，其中 $N$ 是状态维度，$P$ 是头维度。结果是每个分块的一个 $(N, P)$ 张量，它与扩展隐藏状态 $h$ 具有相同的维度。

这可以解释为：假设（分块的）初始状态为 0，每个分块的最终状态是什么。换句话说，这在假设 $x _ { 0 : j \ell } = 0$ 的情况下计算 $h _ { j 0 + \ell - 1 }$。

中心因子（Center Factors）。该步骤计算低秩分解中中心块因子项的影响。在上一步中，每个分块的最终状态的总形状为 $(T/Q, N, P)$。现在将其乘以由 $A _ { 2 \mathbb { Q } - 1 : \mathbb { Q } - 1 } ^ { \times } , A _ { 3 \mathbb { Q } - 1 : 2 \mathbb { Q } - 1 } ^ { \times } , \mathscr { \cdot } \mathscr { \cdot } , A _ { \intercal - 1 : \intercal - \mathbb { Q } - 1 } ^ { \times }$ 生成的 1-SS 矩阵。

该步骤可以通过任何用于计算 1-SS 乘法的算法（也称为标量 SSM 扫描或 cumprodsum 算子）来计算。

这可以解释为：在考虑所有先前输入的情况下，每个分块（chunk）的实际最终状态是什么；换句话说，这计算了考虑所有 $x _ { 0 : ( j + 1 )  Q }$ 时的真实隐藏状态 $h _ { j { 0 } }$。

左因子（Left Factors）。此步骤计算与低秩分解的左侧块因子的乘法。对于每个分块，这可以用矩阵乘法收缩 contract(QN, NP → QP) 来表示。

这可以解释为：在考虑正确的初始状态 $h _ { j 0 - 1 }$ 且假设输入 $x _ { j  Q : ( j + 1 ) Q }$ 为 0 的情况下，每个分块的输出是什么。换句话说，对于分块 $j$，这计算了仅考虑先前输入 $x _ { 0 : j \ell }$ 时的正确输出。

## 6.3 计算成本

我们定义记号 BMM(B, M, N, K) 来表示批大小维度为 B 的批矩阵乘法 contract(MK, KN → MN)。从该记号中我们可以推断出效率的三个方面：

• 计算成本：总计 $O(\text{BMNK})$ FLOPs。

• 内存成本：总计 $O ( \mathsf { B } ( \mathsf { M K } + \mathsf { K N } + \mathsf { M N } ) )$ 空间。

```python
Listing 1 Full PyTorch example of the state space dual (SSD) model.

def segsum(x):
    """Naive segment sum calculation. exp(segsum(A)) produces a 1-SS matrix,
    which is equivalent to a scalar SSM."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[... , :, None] - x_cumsum[... , None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.mask_fill(~mask, -torch.inf)
    return x_segsum

def ssd(X, A, B, C, block_len=64, initial_states=None):
    """
    Arguments:
    X: (batch, length, n_heads, d_head)
    A: (batch, length, n_heads)
    B: (batch, length, n_heads, d_state)
    C: (batch, length, n_heads, d_state)
    Return:
    Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

# Rearrange into blocks/chunks
X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

A = rearrange(A, "b c l h -> b h c l")
A_cumsum = torch.cumsum(A, dim=-1)

# 1. Compute the output for each intra-chunk (diagonal blocks)
L = torch.exp(segsum(A))
Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

# 2. Compute the state for each intra-chunk
# (right term of low-rank factorization of off-diagonal blocks; B terms)
decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

# 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
# (middle term of factorization of off-diag blocks; A terms)
if initial_states is None:
    initial_states = torch.zeros_like(states[:, :1])
states = torch.cat([initial_states, states], dim=1)
decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
states, final_state = new_states[:, :-1], new_states[:, -1]

# 4. Compute state -> output conversion per chunk
# (left term of low-rank factorization of off-diagonal blocks; C terms)
state_decay_out = torch.exp(A_cumsum)
Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

# Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
return Y, final_state
```

• 并行化：更大的 M、N、K 项可以充分利用现代加速器上的专用矩阵乘法单元。

中心块（Center Blocks）。二次 SMA 计算的开销包含三个步骤（公式 (16)）：

• 计算核矩阵 <sup>⊤</sup>，其开销为 BMM(T/Q, Q, Q, N)。

• 乘以掩码矩阵，这是在形状为 (T/Q, Q, Q) 的张量上进行的逐元素操作。

• 乘以  值，其开销为 BMM(T/Q, Q, P, N)。

低秩块：右因子（Low-Rank Blocks: Right Factors）。该步骤是一次矩阵乘法，开销为 BMM(T/Q, N, P, Q)。

低秩块：中心因子（Low-Rank Blocks: Center Factors）。该步骤是在 (N, P) 个独立通道上进行的长度为 T/Q 的标量 SSM 扫描（或 1-SS 乘法）。此扫描的计算量为 TNP/Q，与其他因子相比可以忽略不计。

注意，由于分块将序列长度从 T 缩减到了 T/Q，该扫描的开销比纯 SSM 扫描（例如 Mamba 的选择性扫描）小 Q 倍。因此我们观察到，在大多数问题长度下，其他算法（附录 B）可能更高效或更容易实现，且不会带来显著的减速。例如，通过 1-SS 矩阵乘法进行的朴素实现开销为 BMM(1, T/Q, NP, T/Q)，它比朴素的递推/扫描实现更容易实现，并且可能更加高效。

低秩块：左因子（Low-Rank Blocks: Left Factors）。该步骤是一次矩阵乘法，开销为 BMM(T/Q, Q, P, N)。

总开销。如果我们设置 ${ \mathsf { N } } = { \mathsf { P } } = { \mathsf { Q } }$（换句话说，状态维度、头维度和分块长度相等），那么上述所有 BMM 项都变为 BMM(T/N, N, N, N)。其计算特征如下：

• 总 FLOP 次数为 (TN<sup>2</sup>)。

• 总内存为 (TN)。

• 计算工作主要由形状为 (N, N) 的矩阵上的矩阵乘法组成。

注意到内存消耗是紧凑的；输入和输出 ,  的形状为 $( { \sf T } , { \sf P } ) = ( { \sf T } , { \sf N } )$。同时，FLOP 次数反映了一个额外的 N 因子，这是由自回归状态大小带来的开销，在所有模型中都是共通的。

除矩阵乘法外，在 ${ \mathsf { N P } } = { \mathsf { N } } ^ { 2 }$ 个特征和序列长度 T/Q 上还存在一个标量 SSM 扫描（scan）。这需要 $O ( \mathsf { T } / \mathsf { Q N } ^ { 2 } )$ FLOPs 的计算量以及 $O(\log(\text{T/Q}))$ 的深度。尽管它没有使用矩阵乘法，但它仍然是可并行化的，并且与其他步骤相比，所完成的总工作量可以忽略不计；在我们的 GPU 实现中，这部分开销微乎其微。

与纯 SSM 及注意力模型的比较。二次注意力（Quadratic attention）仅通过利用矩阵乘法也具有很高的硬件效率，但具有 $T^{2}N$ 的总 FLOPs。它在训练和推理时较慢的计算速度，可以直接归因于拥有更大的状态大小——标准注意力机制的状态大小随序列长度 T 增长，因为它缓存了历史记录且不对其状态进行压缩。

线性 SSM 的总 FLOPs 为 ${ \mathsf { T N P } } = { \mathsf { T N } } ^ { 2 }$，与 SSD 相同。然而，朴素实现需要进行状态扩展 (15a)，从而实体化占用额外的内存/显存，并且包含一个无法利用矩阵乘法的标量操作 (15b)。

<table><tr><td></td><td>Attention</td><td>SSM</td><td>SSD</td></tr><tr><td>状态大小</td><td>T</td><td>N</td><td>N</td></tr><tr><td>训练 FLOPs</td><td> $T^{2}N$ </td><td> $TN^{2}$ </td><td> $TN^{2}$ </td></tr><tr><td>推理 FLOPs</td><td>TN</td><td> $N^{2}$ </td><td> $N^{2}$ </td></tr><tr><td>（朴素）内存占用</td><td> $T^{2}$ </td><td> $TN^{2}$ </td><td>TN</td></tr><tr><td>矩阵乘法</td><td>√</td><td></td><td>√</td></tr></table>

我们注意到，许多其他的矩阵分解也是可能的（例如，参见附录 B，其中汇总了通过不同结构化矩阵分解进行 1-SS 乘法的算法纲要），这可能会引出更多适用于其他特定场景的更优 SSD 算法。更广泛地讲，我们注意到半可分矩阵拥有丰富的文献资料，除了我们所采用的 SSS 形式（定义 3.2）之外，还存在许多其他表示形式，因此可能存在更加高效的算法。

## 7 Mamba-2 架构

通过将 SSM 与注意力机制联系起来，SSD 框架使我们能够为两者建立共享的词汇表与技术库。在本节中，我们讨论利用最初为 Transformer 开发的思想来理解和修改 SSD 层的若干示例。我们讨论了若干设计选择，最终形成了 Mamba-2 架构。这些变化维度在第 9.4 节中进行了消融实验。

![](images/8d457e3d1f065604b5e0168e2611acd01e4a8fa98fcc09fd41e25c68a669e5b2.jpg)  
Figure 6：（Mamba-2 架构。）Mamba-2 块通过移除顺序线性投影简化了 Mamba 块；SSM 参数 $A$、$B$、$C$ 是在块的起始处生成的，而不是作为 SSM 输入 $X$ 的函数。与 NormFormer（Shleifer, Weston, and Ott 2021）一样添加了一个额外的归一化层，提高了稳定性。$A$ 和 $B$ 投影仅具有一个在各个 $C$ 头之间共享的单头，类似于多值注意力（MVA）。

## 7.1 块设计

我们首先讨论与内部序列混合层（sequence mixing layer）无关的神经网络块（neural network block）修改（即在核心 SSD 层之外）。

并行参数投影。Mamba-1 源于以 SSM 为中心的视角，其中选择性 SSM 层被视为从  ↦→  的映射。SSM 参数 , ,  被视为辅助参数，且是 SSM 输入  的函数。因此，定义 (, , ) 的线性投影发生在生成  的初始线性投影之后。

在 Mamba-2 中，SSD 层被视为从 , , ,  ↦→  的映射。因此，在块的起始位置通过单个投影并行生成 , , ,  是合理的。注意这与标准注意力架构（attention architectures）的相似之处，其中 , ,  对应于并行创建的 , ,  投影。

注意，对 SSM 的 , , ,  输入采用并行投影略微减少了参数量，并且更重要的是，通过使用标准的 Megatron 分片模式（Shoeyb et al. 2019），它更易于在较大模型中应用张量并行（tensor parallelism）。

额外归一化。在初步实验中，我们发现在较大的模型中容易出现不稳定性。我们通过在最终输出投影之前的块中添加一个额外的归一化层（例如 LayerNorm、GroupNorm 或 RMSNorm）缓解了这一问题。这种归一化的用法与 NormFormer 架构（Shleifer, Weston, and Ott 2021）最为直接相关，该架构也在 MLP 和 MHA 块的末端添加了归一化层。

我们还注意到，这一改变与近期从线性注意力（linear attention）视角导出的其他 Mamba-2 相关模型类似。原始的线性注意力公式通过一个分母项进行归一化，以模拟标准注意力机制中 softmax 函数的归一化效果。TransNormerLLM（Qin, Dong, Li, et al. 2023）和 RetNet（Y. Sun et al. 2023）发现这种归一化方式不稳定，并在经过线性注意力层之后添加了一个额外的 LayerNorm 或 GroupNorm。我们的额外归一化层与这些方法略有不同，它位于乘法门控（multiplicative gate）分支之后，而不是之前。

## 7.2 序列变换的多头模式

回想一下，SSM 被定义为一种序列变换（定义 2.1），其中：

• $A, B, C$ 参数具有状态维度 $N$。

• 它们定义了一个序列变换 $\mathbb { R } ^ { \mathsf { T } } \to \mathbb { R } ^ { \mathsf { T } }$，例如可以表示为一个矩阵 $M \in \mathbb { R } ^ { ( \top , \top ) }$。

• 该变换在输入序列 $X \in \mathbb { R } ^ { ( \mathsf { T } , \mathsf { P } ) }$ 上操作，且在 $P$ 轴上独立进行。

可以将这视为定义了序列变换的一个头（head）。

定义 7.1（多头模式 / Multihead patterns）。多头序列变换由 $H$ 个独立的头组成，总模型维度为 $D = \text{d\_model}$。参数可以在不同的头之间共享（tied），从而形成一种头模式（head pattern）。

状态大小 $N$ 和头维度 $P$ 分别类似于注意力机制的  头维度和  头维度。正如在现代 Transformer 架构（Chowdhery et al. 2023; Touvron, Lavril, et al. 2023）中一样，在 Mamba-2 中，我们通常将它们选为 64 或 128 左右的常数；当模型维度 $D$ 增加时，我们在保持头维度 $N$ 和 $P$ 固定的同时增加头的数量。为了描述如何做到这一点，我们可以迁移并推广多头注意力（multihead attention）的思想，为 SSM 或任何通用的序列变换定义类似的模式。

<table><tr><td colspan="2">多头 SSM（多头注意力）</td><td colspan="2">多收缩 SSM（多查询注意力）</td><td colspan="2">多扩展 SSM（多键注意力）</td><td colspan="2">多输入 SSM（多值注意力）</td></tr><tr><td>X</td><td>(T,H,P)</td><td>(X</td><td>(T,1,P)</td><td>(X</td><td>(T,1,P)</td><td>(X</td><td>(T,H,P)</td></tr><tr><td>A</td><td>(T,H)</td><td>A</td><td>(T,H)</td><td>A</td><td>(T,H)</td><td>A</td><td>(T,H)</td></tr><tr><td>B</td><td>(T,H,N)</td><td>B</td><td>(T,1,N)</td><td>B</td><td>(T,H,N)</td><td>B</td><td>(T,1,N)</td></tr><tr><td>C</td><td>(T,H,N)</td><td>C</td><td>(T,H,N)</td><td>C</td><td>(T,1,N)</td><td>C</td><td>(T,1,N)</td></tr></table>

多头 SSM（MHS）/ 多头注意力（MHA）模式。经典的 MHA 模式假设头维度 $P$ 能整除模型维度 $D$。头的数量定义为 $H = D/P$。然后，通过为每个参数创建 $H$ 个独立副本，生成核心序列变换的 $H$ 个副本。请注意，虽然 MHA 模式最初是针对注意力序列变换描述的，但它可以应用于任何符合定义 2.1 的变换。例如，多头 SSD 层将接收符合公式 (17) 形状的输入，其中 SSD 算法在 $H = \text{n\_heads}$ 维度上进行广播。

多收缩 SSM（MCS）/ 多查询注意力（MQA）模式。多查询注意力（Shazeer 2019）是针对注意力机制的一种巧妙优化，它可以极大地提高自回归推理的速度，而自回归推理依赖于缓存  和  张量。该技术仅避免为  和  赋予额外的头维度，换言之，是将单头的 (, ) 广播到  的所有头上。

利用状态空间对偶性（state space duality），我们可以将 MQA 的等价 SSM 版本定义为公式 (18)。在此， $A$ 和 $B$（注意力机制中 $M$ 与 $K$ 的 SSM 对应项）在 $H$ 个头之间共享。我们亦将其称为多收缩 SSM（multi-contract SSM, MCS）头模式，因为控制 SSM 状态收缩的 $C$ 参数在每个头中都具有独立的副本。

我们类似地可以定义多键注意力（multi-key attention, MKA）或多扩展 SSM（multi-expand SSM, MES）头模式，其中控制 SSM 扩展的 $B$ 在每个头之间相互独立，而 $A$ 与 $C$ 则在各个头之间共享。

多输入 SSM（MIS）/ 多值注意力（MVA）模式。虽然 MQA 由于其 KV 缓存（KV cache）的特性而在注意力机制中非常合理，但它并非 SSM 的自然选择。相反，在 Mamba 中， $X$ 被视为 SSM 的主要输入，因此 $B$ 和 $C$ 是在各个输入通道之间共享的参数。我们在公式 (20) 中定义了一种新的多值注意力（multi-value attention, MVA）或多输入 SSM（multi-input SSM, MIS）模式，该模式同样可以应用于任何序列变换，例如 SSD。

借助这套术语，我们可以更精确地刻画原始的 Mamba 架构。

命题 7.2（Proposition 7.2）。Mamba 架构（Gu and Dao 2023）的选择性 SSM（S6）层可以被视为具有：

• 头维度 $P = 1$：每个通道具有独立的 SSM 动力学参数 $A$。

• 多输入 SSM（MIS）或多值注意力（MVA）头结构：$B, C$ 矩阵（对应注意力对偶中的 $K, Q$）在输入 $X$（对应注意力中的 $V$）的所有通道之间共享。

我们还可以在应用于 SSD 时对这些头模式变体进行消融实验（第 9.4.3 节）。有趣的是，尽管控制了参数量和总状态维度，下游性能仍存在显著差异。我们在经验上发现，Mamba 最初采用的 MVA 模式表现最佳。

分组头模式（Grouped Head Patterns）。多查询注意力（multi-query attention）的思想可以扩展为分组查询注意力（grouped-query attention，Ainslie et al. 2023）：不再是 1 个 K 和 V 头，而是可以创建 G 个独立的 K 和 V 头，其中 $1 \ < \ 6$ 且 G 整除 H。这一动机既是为了弥合多查询注意力与多头注意力之间的性能差距，也是为了通过将 G 设置为分片（shard）数量的倍数来实现更高效的张量并行（第 8 节）。

类似地，Mamba-2 中使用的多输入 SSM 头模式可以很容易地扩展为分组输入 SSM（grouped-input SSM, GIS），或同义的分组值注意力（grouped-value attention, GVA）。该泛化非常直接，为简便起见我们省略其细节。

## 7.3 来自线性注意力的其他 SSD 扩展

在此我们描述一个受线性注意力启发的 SSD 架构修改示例。我们在第 9.4.3 节中对这些进行了消融实验，作为一种负面结果，发现它们并没有显著提高性能到足以将其采纳为默认设置的程度。尽管如此，这些例子说明了如何融合大量关于注意力的文献来定义 SSD 的变体。我们将核特征映射（kernel feature map）的选择视为 Mamba-2 架构中的一个超参数，并预期受注意力启发的其他简单修改也是可行的。

Softmax 注意力的核注意力近似（Kernel Attention Approximations to Softmax Attention）。线性注意力或核注意力的许多变体都源于将注意力分数 softmax $( Q K ^ { \top } )$ 视为由

1. 指数核 $Z = \exp ( Q K ^ { \top } )$，可以通过某种核特征映射近似为 $Z = \psi ( Q ) \psi ( K ) ^ { \intercal }$。

2. 通过 $M = G / G 1 1 ^ { \top }$ 对核进行归一化，使得行和为 1，其中除法是逐元素进行的，1 是全 1 向量。

指数核特征映射。在 Mamba-2 中，我们引入了一种灵活的核特征映射，并将其应用于  和  分支（对应于注意力机制中的  和  分支）。为了简便和对称，该特征映射也可以选择性地应用于  () 分支。在 Figure 6 中，这由任意非线性函数表示。默认情况下，我们直接选择  为逐元素的 Swish / SiLU 函数（Hendrycks and Gimpel 2016; Ramachandran, Zoph, and Le 2017）。我们在 Section 9.4.3 的消融实验中探索了其他选项，包括 Linear Attention、Performer、Random Feature Attention 和 cosFormer（Section 4.1.3）所使用的特征映射。

引入归一化（分母）项。为了求得分母项，我们只需计算 1。但回想一下，模型的最终输出就是  = （公式 (16)）。因此，只需用一个额外的全 1 列扩充 ，即可得到归一化项，从而得到一个形状为 (T, P + 1) 的张量。

注意在这种情况下，核特征映射 $\psi$ 必须为正，以保证其和为正。

## 8 SSM 的系统优化

我们描述了针对 SSM（特别是 Mamba-2 架构）的若干系统优化，以实现大规模高效训练和推理。具体而言，我们重点关注用于大规模训练的张量并行（tensor parallel）和序列并行（sequence parallel），以及用于高效微调和推理的可变长序列。

## 8.1 张量并行

张量并行（Tensor parallelism, TP）（Shoeybi et al. 2019）是一种模型并行技术，它将每个层（$( \mathbf { e . g . }$，注意力机制、MLP）拆分到多个加速器（如 GPU）上运行。该技术被广泛应用于在 GPU 集群上训练大多数大模型（Brown et al. 2020; Chowdhery et al. 2023; Touvron, Lavril, et al. 2023; Touvron, L. Martin, et al. 2023），其中每个节点通常包含 4-8 个 GPU，并配备诸如 NVLink 等高速网络连接。TP 最初是为 Transformer 架构开发的，将其调整适配到其他架构并非显而易见。我们首先展示将 TP 应用于 Mamba 架构时的挑战，然后展示 Mamba-2 架构是如何通过设计来实现高效 TP 的。

回顾 Mamba 架构，对于单个输入 $u ~ \in ~ \mathbb { R } ^ { L \times d }$（为简便起见不进行批处理），输入投影矩阵 $W ^ { ( x ) } , W ^ { ( z ) } \in \mathbb { R } ^ { d \times e d }$，其中 $e$ 是扩展因子（通常为 2），以及输出投影矩阵 $\bar { W ^ { ( o ) } } \bar { \in } \bar { \mathbb { R } ^ { e d \times d } }$：

$$
\begin{array}{l} x = u W ^ {(x) ^ {\top}} \in \mathbb {R} ^ {L \times e d} \\ z = u W ^ {(z) ^ {\top}} \in \mathbb {R} ^ {L \times e d} \\ x _ {c} = \operatorname{conv1d} (x) \in \mathbb {R} ^ {L \times e d} \quad (\text {depthwise, independent along } d) \\ \Delta , B, C = \text {low - rank projection} (x _ {c}) \\ y = S S M _ {A, B, C, \Delta} (x _ {c}) \in \mathbb {R} ^ {L \times e d} \quad (\text {independent along } d) \\ y _ {g} = y \cdot \phi (z) \quad (\text {gating, e.g., with} \phi \text {being SiLU}) \\ \text {out} = y _ {g} W ^ {(o) ^ {\top}} \in \mathbb {R} ^ {L \times d}. \end{array}
$$

在使用 TP（张量并行，Tensor Parallelism）时，假设我们希望将计算切分到 2 个 GPU 上。将输入投影矩阵 $W ^ { ( x ) }$ 和 $\bar { W } ^ { ( \bar { z } ) }$ 拆分为两个大小均为 $\bar { d } \times \frac { e d } { 2 }$ 的分区是非常容易的。这样每个 GPU 将持有大小为 $\begin{array} { r } { L \times \frac { e d } { 2 } } \end{array}$ 的 $x _ { c }$ 的一半。然而，我们发现由于 $\Delta , B , C$ 是 $x _ { c }$ 的函数，因此在计算 $\Delta , B , C$ 之前，我们需要在 GPU 之间执行一次额外的 all-reduce 操作以获取完整的 $x _ { c }$。在此之后，两个 GPU 可以并行计算 SSM，因为它们沿 $d$ 是相互独立的。在最后，我们可以将输出投影矩阵 $W ^ { \overline { { ( o ) } } }$ 拆分为两个大小均为 $\begin{array} { r } { \frac { e d } { 2 } \times d , } \end{array}$ 的分区，并在末尾执行一次 all-reduce。与 Transformer 相比，我们将产生两次 all-reduce 而不是一次，从而使通信耗时翻倍。对于大规模 Transformer 训练，通信可能已经占据了相当大比例的时间（例如 10-20%），通信量的翻倍将导致 Mamba 在大规模训练中的效率降低。

对于 Mamba-2，我们的目标是每个块仅进行一次 all-reduce，类似于 Transformer 中的注意力（attention）或 MLP 块。因此，我们直接从 $u$ 而不是从 $x_c$ 投影得到 $\Delta , B , C$，从而允许我们对这些投影矩阵进行切分。这意味着我们在不同的 GPU 上拥有不同的 $\Delta , B , C$ 集合，这等价于在一个更大的“逻辑 GPU”上拥有多“组”（groups）$\Delta , B , C$。此外，我们在每个块内使用 GroupNorm，其组数可被 TP（张量并行）度整除，这样 TP 组内的各个 GPU 就不需要在块内部进行通信：

$$
\begin{array}{r l} & x = u W ^ {(x) ^ {\top}} \in \mathbb {R} ^ {L \times e d} \\ & z = u W ^ {(z) ^ {\top}} \in \mathbb {R} ^ {L \times e d} \\ \Delta , B, C = \text {projection} (u) \quad \text {(one or more groups of} \Delta , B, C \text {per GPU)} \\ & x _ {c} = \text {conv1d} (x) \in \mathbb {R} ^ {L \times e d} \quad \text {(depthwise, independent along} d) \\ & y = S S M _ {A, B, C, \Delta} (x _ {c}) \in \mathbb {R} ^ {L \times e d} \quad \text {(independent along} d) \\ & y _ {g} = y \cdot \phi (z) \quad \text {(gating, e.g., with} \phi \text {being SiLU)} \\ & y _ {n} = \text {groupnorm} (y _ {g}) \quad \text {(number of groups divisible by degree of tensor parallel)} \\ & \text {out} = y _ {g} W ^ {(o) ^ {\top}} \in \mathbb {R} ^ {L \times d}. \end{array}
$$

我们可以看到，我们只需要切分输入投影矩阵和输出投影矩阵，并且只需要在块（block）末尾执行一次 all-reduce。这与注意力机制（attention）和 MLP 层的张量并行（Tensor Parallelism，TP）设计类似。具体而言，如果 TP 度数为 2，我们将切分 $W ^ { ( x ) } = [ W _ { 1 } ^ { ( x ) } , W _ { 2 } ^ { ( x ) } ]$（其中 $\overset { \vartriangle } { W _ { i } ^ { ( x ) } } \in \mathbb { R } ^ { d \times e d / 2 }$）、$W ^ { ( z ) } = [ W _ { 1 } ^ { ( z ) } , W _ { 2 } ^ { ( \overset { \cdot } { z } ) } ]$（其中 $W _ { i } ^ { ( z ) } \in \mathbb { R } ^ { d \times e d / 2 }$）以及 $W ^ { ( o ) } = \left[ \begin{array} { l } { W _ { 1 } ^ { ( o ) } } \\ { W _ { 2 } ^ { ( o ) } } \end{array} \right]$（其中 $W _ { i } ^ { ( o ) } \in \mathbb { R } ^ { e d / 2 \times d }$）。对于 $i = 1, 2$，TP Mamba-2 层可以表示为：

![](images/092e16c85af32d264ef8e11750d4e08d52649d2d95dc125b9612b216b0907ec6.jpg)  
Figure 7: （Mamba-2 块的并行化。）（左：张量并行）我们切分输入投影矩阵 $W ^ { ( x ) } , W ^ { ( z ) }$ 和输出投影矩阵 $W ^ { ( o ) }$。每个 SSM 头 $( A , B , C , X ) \mapsto Y$ 位于单个设备上。最终归一化层选择 GroupNorm 可以避免额外的通信。与 Transformer 中的 MLP 或注意力块一样，我们每层只需要一次 all-reduce。（右：序列/上下文并行）与 SSD 算法类似，在多设备情况下，我们可以沿着序列维度进行切分。每个设备计算其序列的状态，然后将该状态传递给下一个 GPU。

$$
x ^ {(i)} = u W _ {i} ^ {(x) ^ {\top}} \in \mathbb {R} ^ {L \times e d / 2}
$$

$$
\boldsymbol {z} ^ {(i)} = \boldsymbol {u} W _ {i} ^ {(z) ^ {\top}} \in \mathbb {R} ^ {L \times e d / 2}
$$

$\Delta^{(i)}, B^{(i)}, C^{(i)} = \text{projection}(u)$（每个 GPU 一组或多组 $\Delta, B, C$）

$$
x _ {c} ^ {(i)} = \mathrm{conv1d} (x ^ {(i)}) \in \mathbb {R} ^ {L \times e d / 2}
$$

$$
y ^ {(i)} = S S M _ {A, B, C, \Delta} (x _ {c} ^ {(i)}) \in \mathbb {R} ^ {L \times e d / 2}
$$

$$
y _ {g} ^ {(i)} = y ^ {(i)} \cdot \phi (z ^ {(i)})
$$

$y_n^{(i)} = \mathrm{groupnorm}(y_g^{(i)})$（组数需可被张量并行度整除）

$$
\mathsf {o u t} ^ {(i)} = y _ {g} ^ {(i)} W _ {i} ^ {(o) ^ {\top}} \in \mathbb {R} ^ {L \times d / 2}
$$

$\mathrm { o u t } = \sum _ { i } \mathrm { o u t } ^ { ( i ) }$ 。（通过 all-reduce 对来自所有 GPU 的输出求和）

我们在 Figure $7 \ ( L e f t )$ 中展示了 Mamba-2 的张量并行。

## 8.2 序列并行（Sequence Parallelism）

对于极长序列，我们可能需要沿序列长度维度将输入和激活值切分到不同的 GPU 上。主要有两种技术：

1. 用于残差和归一化操作的序列并行（Sequence parallelism, SP）：由 Korthikanti et al. (2023) 首次提出，该技术将 TP 中的 all-reduce 分解为 reduce-scatter 和 all-gather。注意到对于同一 TP 组中的所有 GPU，残差和归一化操作是在相同的输入上重复执行的，SP 通过执行以下操作沿序列长度维度切分激活值：reduce-scatter、残差与归一化，然后是 all-gather。

由于 Mamba-2 架构采用了相同的残差和归一化结构，因此 SP 可以直接应用而无需修改。

2. 用于标记混合（token-mixing）操作（注意力机制或 SSM）的序列并行（Sequence parallelism），也称为“上下文并行”（context parallelism, CP）。针对注意力层已经开发了若干技术（例如 Ring attention (Liu, Yan, et al. 2024; Liu, Zaharia, and Abbeel 2023)），并配备了复杂的负载均衡技术 (Brandon et al. 2023)。注意力机制中序列并行的难点在于，虽然我们可以将查询（queries）和键（keys）切分成块，但每个查询块都需要与键块进行交互，导致通信带宽需求随工作节点（worker）数量呈二次方增长。

![](images/cd61ae528404eda8a9945b37ab3669544175e7627c4c1cf55d8a66cd7986ef52.jpg)  
Figure 8: （多查询关联召回（Multi-Query Associative Recall, MQAR））。关联召回任务对 SSM 而言极具挑战性，因为 SSM 必须将所有相关信息记忆到其循环状态中。SSD 层结合改进的架构使得 Mamba-2 能够使用大得多的状态维度，其性能显著优于 Mamba-1，甚至超越了原生注意力机制（vanilla attention）。

对于 SSM，我们可以以一种简单的方式切分序列：每个 worker 接收一个初始状态，针对其输入计算 SSM，返回最终状态，并将该最终状态传递给下一个 worker。通信带宽与 worker 的数量呈线性关系。这种分解方式与 SSD 算法（Figure 5）中切分为块（blocks / chunks）的块分解完全相同。我们在 Figure 7（右图）中展示了这种上下文并行。

## 8.3 可变长度

虽然预训练通常在批次（batch）中使用相同的序列长度，但在微调或推理过程中，模型可能需要处理不同长度的不同输入序列。处理这种情况的一种朴素方法是将批次中的所有序列右侧填充（right-pad）至最大长度，但如果序列长度差异极大，这种方法可能会效率低下。针对 Transformer，已经开发出了复杂的技术来避免填充并在 GPU 之间进行负载均衡（Zeng et al. 2022; Y. Zhai et al. 2023），或者将多个序列打包在同一个批次中并调整注意力掩码（attention mask）（Ding et al. 2024; Pouransari et al. 2024）。对于 SSM，特别是 Mamba，我们只需将整个批次视为一个长序列，并避免在各个独立序列之间传递状态，即可处理可变序列长度。这等价于在某个序列末尾的 token $t$ 处简单地设置 $A _ { t } = 0$，以防止其向属于不同序列的 token $t + 1$ 传递信息。

## 9 实证验证

我们在循环模型一直难以应对的合成回忆任务（第 9.1 节），以及标准语言模型预训练和下游评估（第 9.2 节）中对 Mamba-2 进行了实证评估。我们验证了我们的 SSD 算法比 Mamba-1 高效得多（第 9.3 节），且在中等序列长度下可与优化后的注意力机制相媲美。最后，我们对 Mamba-2 架构中的各种设计选择进行了消融实验（第 9.4 节）。

## 9.1 合成任务：关联回忆

合成关联回忆（Associative Recall）任务在测试语言模型从其上下文中检索信息的能力方面一直很受欢迎。广义上讲，它们包括向自回归模型输入成对的键值关联，然后在向模型展示之前见过的键时，提示模型给出正确的补全。多查询关联回忆（Multi-Query Associative Recall, MQAR）任务是该任务的一种特定形式，它要求模型记住多个关联（Arora, E ubo lu, Timalsina, et al. 2024）。最初的 Mamba 论文报告了相关合成任务的结果，特别是选择性复制（Selective Copying，Gu and Dao 2023）和归纳头（Induction Heads，Olsson et al. 2022），这些可以看作是更简单的关联回忆任务。MQAR 任务还与“电话簿查找”（phonebook look-up）任务密切相关，由于有限的状态容量，这类任务已被证明对状态空间模型（SSM）等循环模型极具挑战性（De et al. 2024; Jelassi et al. 2024）。

The Pile 上的缩放定律（序列长度 8192）  
![](images/5a3af6669fc2667c5615bf24e812553e3fd92b39c634cea8d46872bcbf9bfebe.jpg)  
Figure 9：（缩放定律。）在 The Pile 上训练的大小约为 125 到 ≈ 1.3 参数量的模型。Mamba-2 达到或超过了 Mamba 以及强“Transformer++”方案的性能。与我们的 Transformer 基线相比，Mamba-2 在性能（困惑度）、理论 FLOPs 和实际运行耗时上均占据帕累托最优（Pareto dominant）。

Table 1：（零样本评估。）加粗表示每种尺寸下的最优结果，次优结果带下划线。我们与使用不同分词器（tokenizer）、训练 token 量高达 300B 的开源语言模型（LM）进行了对比。Pile 指的是验证集划分，仅与使用相同数据集和分词器（GPT-NeoX-20B）训练的模型进行比较。在每个模型尺寸下，Mamba-2 的表现均优于 Mamba，并且大体上能媲美两倍于其模型尺寸的 Pythia。完整结果见 Table 10。

<table><tr><td>模型</td><td>分词器</td><td>PILE PPL ↓</td><td>LAMBADA PPL ↓</td><td>LAMBADA ACC ↑</td><td>HELLASWAG ACC ↑</td><td>PIQA ACC ↑</td><td>ARC-E ACC ↑</td><td>ARC-C ACC ↑</td><td>WINOGRANDE ACC ↑</td><td>OPENBOOKQA ACC ↑</td><td>平均 ACC ↑</td></tr><tr><td>Pythia-1B</td><td>NeoX</td><td>7.82</td><td>7.92</td><td>56.1</td><td>47.2</td><td>70.7</td><td>57.0</td><td>27.1</td><td>53.5</td><td>31.4</td><td>49.0</td></tr><tr><td>Mamba-790M</td><td>NeoX</td><td>7.33</td><td>6.02</td><td>62.7</td><td>55.1</td><td>72.1</td><td>61.2</td><td>29.5</td><td>56.1</td><td>34.2</td><td>53.0</td></tr><tr><td>Mamba-2-780M</td><td>NeoX</td><td>7.26</td><td>5.86</td><td>61.7</td><td>54.9</td><td>72.0</td><td>61.0</td><td>28.5</td><td>60.2</td><td>36.2</td><td>53.5</td></tr><tr><td>Hybrid H3-1.3B</td><td>GPT2</td><td>—</td><td>11.25</td><td>49.6</td><td>52.6</td><td>71.3</td><td>59.2</td><td>28.1</td><td>56.9</td><td>34.4</td><td>50.3</td></tr><tr><td>Pythia-1.4B</td><td>NeoX</td><td>7.51</td><td>6.08</td><td>61.7</td><td>52.1</td><td>71.0</td><td>60.5</td><td>28.5</td><td>57.2</td><td>30.8</td><td>51.7</td></tr><tr><td>RWKV4-1.5B</td><td>NeoX</td><td>7.70</td><td>7.04</td><td>56.4</td><td>52.5</td><td>72.4</td><td>60.5</td><td>29.4</td><td>54.6</td><td>34.0</td><td>51.4</td></tr><tr><td>Mamba-1.4B</td><td>NeoX</td><td>6.80</td><td>5.04</td><td>65.0</td><td>59.1</td><td>74.2</td><td>65.5</td><td>32.8</td><td>61.5</td><td>36.4</td><td>56.4</td></tr><tr><td>Mamba-2-1.3B</td><td>NeoX</td><td>6.66</td><td>5.02</td><td>65.7</td><td>59.9</td><td>73.2</td><td>64.3</td><td>33.3</td><td>60.9</td><td>37.8</td><td>56.4</td></tr><tr><td>Hybrid H3-2.7B</td><td>GPT2</td><td>—</td><td>7.92</td><td>55.7</td><td>59.7</td><td>73.3</td><td>65.6</td><td>32.3</td><td>61.4</td><td>33.6</td><td>54.5</td></tr><tr><td>Pythia-2.8B</td><td>NeoX</td><td>6.73</td><td>5.04</td><td>64.7</td><td>59.3</td><td>74.0</td><td>64.1</td><td>32.9</td><td>59.7</td><td>35.2</td><td>55.7</td></tr><tr><td>RWKV4-3B</td><td>NeoX</td><td>7.00</td><td>5.24</td><td>63.9</td><td>59.6</td><td>73.7</td><td>67.8</td><td>33.1</td><td>59.6</td><td>37.0</td><td>56.4</td></tr><tr><td>Mamba-2.8B</td><td>NeoX</td><td>6.22</td><td>4.23</td><td>69.2</td><td>66.1</td><td>75.2</td><td>69.7</td><td>36.3</td><td>63.5</td><td>39.6</td><td>59.9</td></tr><tr><td>Mamba-2-2.7B</td><td>NeoX</td><td>6.09</td><td>4.10</td><td>69.7</td><td>66.6</td><td>76.4</td><td>69.6</td><td>36.4</td><td>64.0</td><td>38.8</td><td>60.2</td></tr></table>

SSD、Scan、Convolution 与 Attention 耗时对比（A100 80GB PCIe）  
![](images/dd57a7f384226224b1e217764cee9520f0b36170a3c14dd768bb947fc85ddcac.jpg)

![](images/295c9939056f5dbffe0bc16e483b15a5ee3764ce80dec17d549fa4d3eff1b89c.jpg)  
Figure 10: （效率基准测试。）（左）对于较大的状态扩展（ = 64），我们的 SSD 比 Mamba 融合扫描快 2 − 8 倍，并且在序列长度为 2k 及以上时比 FlashAttention-2 更快。（右）序列长度 4K：增加状态扩展会使 Mamba 优化扫描实现的运行速度呈线性下降。SSD 则可以处理大得多的状态扩展因子而不会发生大幅减速。

我们在 (Arora, Eyuboglu, Zhang, et al. 2024) 的 MQAR 设置的高难度版本上进行了比较，采用了更难的任务、更长的序列以及更小的模型。我们的基线包括标准多头 softmax 注意力（multi-head softmax attention），以及结合了卷积、局部注意力（local attention）和线性注意力变体的 Based 架构。

结果如 Figure 8 所示。虽然 Mamba-1 在该任务上面临困难，但 Mamba-2 在所有设置下均表现良好。令人惊讶的是，即使在控制状态大小相同的情况下（$N = 16$），它也显著优于 Mamba-1。（我们尚不确定架构的哪一方面是主导因素，这仍是一个留待未来工作中探索的问题。）此外，该任务验证了状态大小的重要性：将状态从 $N = 16$ 增加到 $N = 64$ 和 $N = 256$ 能够持续提升在 MQAR 上的表现，因为更大的状态允许记忆更多的信息（键值对）。

## 9.2 语言建模

遵循大语言模型（LLM）的标准实验流程，我们在标准自回归语言建模任务上训练并评估了 Mamba-2 架构，并与其他架构进行了对比。我们对比了预训练指标（困惑度/perplexity）以及零样本（zero-shot）评估结果。模型规模（深度和宽度）遵循 GPT3 规范，参数量从 125M 到 2.7B。我们使用了 The Pile 数据集（L. Gao, Biderman, et al. 2020），并遵循 Brown et al. (2020) 中描述的训练配方。该设置与 Mamba（Gu and Dao 2023）中报告的设置相同；训练细节见附录 D。

## 9.2.1 标度律（Scaling Laws）

作为基线，我们对比了 Mamba 及其 Transformer++ 配方（Gu and Dao 2023），后者基于 PaLM 和 LLaMa 架构（例如旋转位置编码/rotary embedding、SwiGLU MLP、以 RMSNorm 代替 LayerNorm、无线性偏置以及更高的学习率）。由于 Mamba 已经证明其性能优于标准 Transformer 架构（GPT3 架构）以及近期提出的亚二次方架构（H3 (Dao, D. Y. Fu, et al. 2023)、Hyena (Poli et al. 2023)、RWKV-4 (B. Peng, Alcaide, et al. 2023)、RetNet (Y. Sun et al. 2023)），为了图表清晰起见，我们在图中省略了这些架构（对比详见 Gu and Dao (2023)）。

Figure 9 展示了在标准 Chinchilla（Hoffmann et al. 2022）协议下，参数量从约 125M 到约 1.3B 的模型标度律。

## 9.2.2 下游任务评估

Table 1 展示了 Mamba-2 在一系列主流下游零样本评估任务上的表现，并与同等规模下最知名的开源模型进行了对比，其中最重要的是 Pythia（Biderman et al. 2023），它与我们的模型使用了相同的分词器（tokenizer）、数据集和训练长度（300B token）。

## 9.2.3 混合模型：将 SSD 层与 MLP 及 Attention 相结合

最近及同期的工作（Dao, D. Y. Fu, et al. 2023; De et al. 2024; Glorioso et al. 2024; Lieber et al. 2024）表明，同时包含 SSM 层和注意力（attention）层的混合架构相比于 Transformer 或纯 SSM（例如 Mamba）模型能够提升模型质量，特别是在上下文学习（in-context learning）方面。我们探索了将 SSD 层与注意力和 MLP 结合的不同方式，以理解各自的优势。实证发现，使注意力层占总层数的大约 10% 时表现最佳。将 SSD 层、注意力层和 MLP 相结合，其效果也优于纯 Transformer++ 或 Mamba-2。

**SSD 与注意力** 我们发现 SSD 层与注意力层是互补的：单独使用时（例如在 Mamba-2 架构与 Transformer++ 的对比中），它们的性能（以困惑度衡量）几乎相同，但 SSD 与注意力层的混合架构优于纯 Mamba-2 或 Transformer++ 架构。我们展示了在 Pile 数据集上使用 GPT-2 分词器训练至 7B token 的 350M 模型（48 层）的部分结果（Table 2）（相同的参数量、相同的超参数、相同的训练集和验证集）。仅加入极少数注意力层就已带来显著提升，并在质量与效率之间取得了最佳平衡。我们假设 SSM 层作为通用的序列到序列映射表现良好，而注意力层则充当检索机制，用于快速引用序列中先前的 token，而不是强迫模型将所有上下文全部压缩到其记忆（SSM 状态）中。

Table 2:（结合 SSD 与 Attention 块。）具有 48 层的 350M 模型在不同 Attention 层数下的困惑度（Perplexity）。拥有约 10% 比例的 Attention 层表现最佳。

<table><tr><td>ATTN 块数量</td><td>0 (Mamba-2)</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>9</td><td>11</td><td>15</td><td>24</td><td>Transformer++</td></tr><tr><td>困惑度 ↓</td><td>8.60</td><td>8.38</td><td>8.32</td><td>8.29</td><td>8.29</td><td>8.28</td><td>8.26</td><td>8.27</td><td>8.28</td><td>8.30</td><td>8.34</td><td>8.50</td><td>8.68</td></tr></table>

结合 SSD、MLP 和 Attention 的混合模型 我们比较了将 SSD 与（门控）MLP 以及 Attention 层结合的不同方式，并在 2.7B 规模（64 层）下进行了评估，在 the Pile 上训练了 300B tokens（相同的参数量、相同的超参数、相同的训练集与验证集、相同的数据顺序）：

1. Transformer++：32 个 Attention 层与 32 个门控 MLP 层交替排列。

2. Mamba-2：64 个 SSD 层。

3. Mamba-2-MLP：32 个 SSD 层与 32 个门控 MLP 层交替排列。

4. Mamba-2-Attention：58 个 SSD 层与 6 个 Attention 层（位于索引 9, 18, 27, 36, 45, 56 处）<sup>6</sup>。

5. Mamba-2-MLP-Attention：28 个 SSD 层与 4 个 Attention 层，并与 32 个门控 MLP 层交替排列。

我们在 Table 3 中报告了 the Pile 上的验证困惑度以及零样本（zero-shot）评估结果。总体而言，Transformer++ 与 Mamba-2 模型的质量大致相当。我们可以看到，仅添加 6 个 Attention 层便能显著优于纯 Mamba-2 模型（同时也优于 Transformer++）。添加 MLP 层虽然会降低模型质量，但可以 (i) 由于 MLP 层的简洁性和硬件高效性而加速训练与推理 (ii) 通过将 MLP 层替换为混合专家（mixture-of-experts），更容易向上复用升级（up-cycle）为 MoE 模型。

Table 3:（零样本评估。）各规模的最佳结果以粗体显示。我们比较了将 SSD、MLP 和注意力（attention）层相结合的不同方式，在 The Pile 上训练 300B token 的 2.7B 规模下进行了评估。

<table><tr><td>模型</td><td>分词器</td><td>PILE PPL ↓</td><td>LAMBADA PPL ↓</td><td>LAMBADA ACC ↑</td><td>HELLASWAG ACC ↑</td><td>PIQA ACC ↑</td><td>ARC-E ACC ↑</td><td>ARC-C ACC ↑</td><td>WINOGRANDE ACC ↑</td><td>OPENBOOKQA ACC ↑</td><td>平均准确率 ↑</td></tr><tr><td>Transformer++</td><td>NeoX</td><td>6.13</td><td>3.99</td><td>70.3</td><td>66.4</td><td>75.2</td><td>67.7</td><td>37.8</td><td>63.9</td><td>40.4</td><td>60.2</td></tr><tr><td>Mamba-2</td><td>NeoX</td><td>6.09</td><td>4.10</td><td>69.7</td><td>66.6</td><td>76.4</td><td>69.6</td><td>36.4</td><td>64.0</td><td>38.8</td><td>60.2</td></tr><tr><td>Mamba-2-MLP</td><td>NeoX</td><td>6.13</td><td>4.18</td><td>69.3</td><td>65.0</td><td>76.4</td><td>68.1</td><td>37.0</td><td>63.1</td><td>38.2</td><td>59.6</td></tr><tr><td>Mamba-2-Attention</td><td>NeoX</td><td>5.95</td><td>3.85</td><td>71.1</td><td>67.8</td><td>75.8</td><td>69.9</td><td>37.8</td><td>65.3</td><td>39.0</td><td>61.0</td></tr><tr><td>Mamba-2-MLP-Attention</td><td>NeoX</td><td>6.00</td><td>3.95</td><td>70.0</td><td>66.6</td><td>75.4</td><td>70.6</td><td>38.6</td><td>64.6</td><td>39.2</td><td>60.7</td></tr></table>

## 9.3 速度基准测试

我们将 SSD 算法的速度与 Mamba 的扫描（scan）实现以及 FlashAttention-2 进行了基准对比（Figure 10）。得益于将其重新表述为以矩阵乘法作为子例程，SSD 可以充分利用 GPU 上的专用矩阵乘法（matmul）单元，即 Tensor Core。因此，它比未利用 matmul 单元的 Mamba 融合结合扫描（fused associative scan）快 2-8 倍。由于其关于序列长度呈线性扩展，SSD 从序列长度 2K 开始便比 FlashAttention-2 更快。

然而，我们注意到整个 Mamba-2 模型在短序列长度（例如 2k）下的训练效率可能不如 Transformer，因为一个具有 $L$ 层的 Transformer 会拥有 $\begin{array} { l } { { \frac { L } { 2 } } } \end{array}$ 个 MLP 层和 $\begin{array} { l } { { \frac { L } { 2 } } } \end{array}$ 个注意力层（attention layers），而在相同参数量下 Mamba-2 模型将拥有 $L$ 个 SSD 层。通常 MLP 层的硬件效率非常高，因为它们仅由简单的矩阵乘法和逐点线性/非线性变换组成。正如第 9.2.3 节所示，也可以将 $\begin{array} { l } { { \frac { L } { 2 } } } \end{array}$ 个 SSD 层与 $\begin{array} { l } { { \frac { L } { 2 } } } \end{array}$ 个 MLP 层相结合，以加速短序列长度下的训练。

Table 4: （消融实验：Mamba-2 块。）我们对 Mamba-2 和 Mamba-1 神经网络块之间的主要差异进行了消融实验（Figure $^ { 6 , }$，Section 7.1）。注意，这些组件独立于内部序列混合层；在这些消融实验中，内部 SSM 层均使用 SSD（不同于 Mamba-1 的 S6 层）。

<table><tr><td>块</td><td>ABCX 投影</td><td>额外归一化</td><td>PARAMETERS</td><td>PERPLEXITY</td></tr><tr><td rowspan="3">Mamba-1</td><td>Sequential</td><td>✕</td><td>129.3M</td><td>11.76</td></tr><tr><td>Sequential</td><td>✕</td><td>129.3M</td><td>11.54</td></tr><tr><td>Parallel</td><td>✕</td><td>126.5M</td><td>11.66</td></tr><tr><td>Mamba-2</td><td>Parallel</td><td>✕</td><td>126.5M</td><td>11.49</td></tr></table>

## 9.4 架构消融实验

## 9.4.1 块设计

第 7.1 节介绍了 Mamba-2 模块，该模块对 Mamba-1 模块进行了细微修改，这些修改部分是受到与注意力机制（attention）联系的启发，同时也是为了提高 Mamba-2 的可扩展性。表 4 对模块的这些架构改动进行了消融实验，这些改动均发生在核心 SSM 层之外。

消融实验证实，用于生成 (, , , ) 的并行投影不仅节省了参数，而且性能略优于 Mamba 的顺序投影。更重要的是，这种修改更易于在更大的模型规模下应用张量并行（tensor parallelism，第 8 节）。此外，额外的归一化层也略微提升了性能。更重要的是，在大规模下的初步实验观察到，它还有助于提高训练稳定性。

## 9.4.2 头结构

第 7.2 节阐述了如何将 , ,  投影的维度视为类似于多头注意力（multi-head attention）和多查询注意力（multi-query attention）概念的超参数。我们还展示了原始 Mamba 架构如何类比于多值注意力（multi-value attention，命题 7.2），这是一种从状态空间模型（state-space model）角度自然发展而来的选择，此前未进行过消融实验。

表 5 消融了 Mamba-2 架构多头结构的选择。引人注目的是，尽管 multi-value（多值）模式与 multi-query（多查询）或 multi-key（多键）头模式看似非常相似，但我们发现它们之间存在巨大差异。请注意，这无法用总状态大小来解释，因为对所有这些模式而言总状态大小都是相同的（等于 $HPN$，即头数、头维度和状态维度的乘积）。

我们还与 , , （类似于 , , ）头数相等的多头模式进行了比较。我们对比了标准多头模式，以及一种它们都仅有 1 个头的激进共享模式。请注意，在后一种情况下，模型仍然具有 $H$ 个不同的序列混合器 $A$，因为每个头仍然具有不同的 $A$。在参数匹配时，这些多头模式的表现彼此相似，介于 MVA 与 MQA/MKA 模式之间。

## 9.4.3 注意力核近似

第 7.3 节指出了 SSD 如何与线性注意力（linear attention）文献中的思想相结合，例如各种形式的核近似（kernel approximations）。我们在表 6 中消融了先前工作提出的几种变体。其中包括 cosFormer (Qin, Weixuan Sun, et al. 2022)、Random Feature Attention (H. Peng et al. 2021) 以及 Positive Random Features (Performer) (Choromanski et al. 2021)。

我们还消融了添加归一化项（类似于标准注意力（standard attention）中 softmax 函数的分母）的效果。我们发现这给大多数变体引入了不稳定性，但略微提升了 ReLU 激活函数  的性能。

Table 7 还测试了近期通过扩展特征维度来改进线性注意力的方案（Based (Arora, Eyuboglu, Zhang, et al. 2024) 和 ReBased (Aksenov et al. 2024)）。这些线性注意力扩展旨在通过二次近似来拟合 exp 核函数。ReBased 还提议用层归一化（layer normalization）替代 QK 激活函数；从以 SSM 为中心的视角来看，我们在应用 SSM 函数之前，在 (, ) 之上应用了归一化。

Table 5: (消融实验：多头结构。) 所有模型的状态扩展因子均为  = 64，头大小为  = 64，并按照 Chinchilla 缩放律（Chinchilla scaling law）的 token 数量进行训练。 头的数量始终等于总头数 H，即每个头都有一个独立的、依赖于输入的  衰减因子。（上）125M 模型，2.5B tokens（下）360M 模型，7B tokens

<table><tr><td>SSM 头模式</td><td>注意力对应结构</td><td>A 头数</td><td>B 头数</td><td>C 头数</td><td>X 头数</td><td>层数</td><td>参数量</td><td>PPL.</td></tr><tr><td>多输入 (MIS)</td><td>多值 (MVA)</td><td>24</td><td>1</td><td>1</td><td>24</td><td>24</td><td>126.5M</td><td>11.66</td></tr><tr><td>多收缩 (MCS)</td><td>多查询 (MQA)</td><td>24</td><td>1</td><td>24</td><td>1</td><td>24</td><td>126.5M</td><td>12.62</td></tr><tr><td>多扩展 (MES)</td><td>多键 (MKA)</td><td>24</td><td>24</td><td>1</td><td>1</td><td>24</td><td>126.5M</td><td>12.59</td></tr><tr><td>多头 (MHS)</td><td>多头 (MHA)</td><td>24</td><td>24</td><td>24</td><td>24</td><td>15</td><td>127.6M</td><td>12.06</td></tr><tr><td>多状态 (MSS)</td><td>-</td><td>24</td><td>1</td><td>1</td><td>1</td><td>36</td><td>129.6M</td><td>12.00</td></tr><tr><td>多输入 (MIS)</td><td>多值 (MVA)</td><td>32</td><td>1</td><td>1</td><td>32</td><td>48</td><td>361.8M</td><td>8.73</td></tr><tr><td>多收缩 (MCS)</td><td>多查询 (MQA)</td><td>32</td><td>1</td><td>32</td><td>1</td><td>48</td><td>361.8M</td><td>9.33</td></tr><tr><td>多扩展 (MES)</td><td>多键 (MKA)</td><td>32</td><td>32</td><td>1</td><td>1</td><td>48</td><td>361.8M</td><td>9.36</td></tr><tr><td>多头 (MHS)</td><td>多头 (MHA)</td><td>32</td><td>1</td><td>1</td><td>1</td><td>70</td><td>361.3M</td><td>9.01</td></tr><tr><td>多状态 (MSS)</td><td>-</td><td>32</td><td>32</td><td>32</td><td>32</td><td>29</td><td>357.3M</td><td>9.04</td></tr></table>

Table 7: (消融实验：核近似。) 我们测试了用于线性注意力近似的 (Re)Based 方法，这些方法涉及扩展特征图。(上) 130M 模型。(下)  = 256 的 380M 模型。

Table 6: (消融实验：核近似。) 我们测试了关于核激活函数  的各种方案，包括旨在近似标准 softmax 注意力的 exp 核的线性注意力变体。

<table><tr><td>核激活函数 φ</td><td>困惑度</td></tr><tr><td>无</td><td>11.58</td></tr><tr><td>Swish</td><td>11.66</td></tr><tr><td>Exp</td><td>11.62</td></tr><tr><td>ReLU</td><td>11.73</td></tr><tr><td>ReLU + 归一化</td><td>11.64</td></tr><tr><td>cosFormer</td><td>11.97</td></tr><tr><td>Random Feature Attention</td><td>11.57</td></tr><tr><td>Positive Random Features (Performer)</td><td>12.21</td></tr></table>

<table><tr><td>核激活函数 φ</td><td>困惑度</td></tr><tr><td>Swish</td><td>11.67</td></tr><tr><td>Swish + Taylor (Based)</td><td>12.19</td></tr><tr><td>LayerNorm</td><td>11.50</td></tr><tr><td>LayerNorm + Square (ReBased)</td><td>11.84</td></tr><tr><td>Swish</td><td>8.58</td></tr><tr><td>Swish + Taylor (Based)</td><td>8.71</td></tr><tr><td>LayerNorm</td><td>8.61</td></tr><tr><td>LayerNorm + Square (ReBased)</td><td>8.63</td></tr></table>

我们注意到，该技术已被独立提出作为 Softmax 注意力的“QK-Norm”（Team 2024）以及 Mamba 的“内部归一化（internal normalization）”（Lieber et al. 2024）。

总体而言，表 6 和表 7 发现，我们尝试的核近似方法相比于针对 $\phi$ 的简单逐点非线性激活函数似乎并没有带来提升。因此，我们针对 Mamba-2 的默认设置采用了 $\phi(x) = \text{Swish}(x)$ 以沿用 Mamba-1，但我们认为完全移除该激活函数可能是一个更简单的选择，不过我们尚未对此进行充分测试。

然而我们强调，SSD 与标准线性注意力（vanilla linear attention）的区别在于包含了 1-半可分掩码（1-semiseparable mask）$M$，而文献中的各种线性注意力方法是在没有该项的情况下推导用于近似 Softmax 注意力的；因此，我们得到负面结果或许并不令人意外。

## 10 相关工作与讨论

状态空间对偶（State Space Duality，SSD）框架架起了状态空间模型（SSM）、结构化矩阵与注意力机制之间的桥梁。我们在此更深入地探讨 SSD 与这些更广泛概念之间的关系。借鉴各个视角的思想，我们还提出了未来拓展 SSD 框架的一些可能方向。

## 10.1 状态空间模型

结构化状态空间模型可以从以下几个维度进行表征：

(i) 是时不变的还是时变的。

(ii) 系统的维度。

(iii) 循环转移上的结构。

SSD 可以被描述为一种具有 SISO（单输入单输出）维度和标量恒等结构的非时变/选择性 SSM。

**时变性（选择性）**。最初的结构化 SSM（S4）是线性时不变（LTI）系统（Gu 2023; Gu, Goel, and Ré 2022），其受到连续时间在线记忆机制的启发（Gu, Dao, et al. 2020; Gu, Johnson, Goel, et al. 2021; Gu, Johnson, Timalsina, et al. 2023）。目前已提出了许多结构化 SSM 的变体（Dao, D. Y. Fu, et al. 2023; Gu, Gupta, et al. 2022; Gupta, Gu, and Berant 2022; Ma et al. 2023; J. T. Smith, Warrington, and Linderman 2023），其中包括几种舍弃了循环结构并侧重于 LTI SSM 卷积表示的变体（D. Y. Fu et al. 2023; Y. Li et al. 2023; Poli et al. 2023; Qin, Han, Weixuan Sun, B. He, et al. 2023）。

SSD 是一种时变结构化状态空间模型（time-varying structured SSM），也被称为在 Mamba（Gu and Dao 2023）中引入的选择性 SSM（selective SSM）。选择性 SSM 与循环神经网络（RNN）的门控机制密切相关，包括经典 RNN 如 LSTM（Hochreiter and Schmidhuber 1997）和 GRU（J. Chung et al. 2014），以及更现代的变体如 QRNN（Bradbury et al. 2016）、SRU（Lei 2021; Lei et al. 2017）、RWKV（B. Peng, Alcaide, et al. 2023）、HGRN（Qin, Yang, and Zhong 2023）和 Griffin（Botev et al. 2024; De et al. 2024）。这些 RNN 在参数化方式上存在各种差异，其中最重要的是缺少状态扩展（state expansion）。

维度与状态扩展。SSD 的一个重要特征（也是其谱系中先前的 SSM 如 S4、H3、Mamba 所共有的）是它是一个单输入单输出（SISO）系统，其中输入通道是独立处理的。这导致了更大的有效状态大小 $ND$，其中 $N$ 是 SSM 状态大小（也称为状态扩展因子），$D$ 是标准模型维度。传统 RNN 要么 $N = 1$，要么是具有稠密 $A, B$ 矩阵的多输入多输出（MIMO）系统，这两种情况都会导致较小的状态。尽管 MIMO SSM 已被证明在某些领域表现良好（Lu et al. 2023; Orvieto et al. 2023; J. T. Smith, Warrington, and Linderman 2023），但 Mamba 表明状态扩展对于语言等信息密集型领域至关重要。SSD 的主要优势之一是允许在不降低模型速度的情况下实现更大的状态扩展因子。此后的许多后续工作都采用了状态扩展（第 10.4 节）。

结构。与以往的结构化 SSM 相比，SSD 的主要限制在于状态转移  的表达能力。我们注意到，更一般的 SSM（例如对角  的情况）具有与 SSD 相同的理论效率，但对硬件不够友好。这是因为其对偶二次型失去了类似注意力机制的解释，并且计算变得更加困难。因此，与 Mamba 相比，SSD 的不同之处仅在于采用了形式稍加受限的对角 <sub></sub>，并通过权衡这种表达能力换取了更高的硬件效率（以及实现的简易性）。

我们推测，或许可以通过改进我们的结构化矩阵算法，进一步拓展到一般的对角 SSM 情况。

## 10.2 结构化矩阵

状态空间对偶性的第一个视角将这些模型视作矩阵序列变换或“矩阵混合器”（matrix mixers）：即可以表示为沿序列维度 T 进行矩阵乘法（乘以一个 T × T 矩阵）的序列变换（定义 2.1）。

此前已有若干此类矩阵混合器被提出，其主要变化维度在于矩阵的表示形式。其中包括 MLP-Mixer (Tolstikhin et al. 2021)（非结构化矩阵）、FNet (Lee-Thorp et al. 2021)（傅里叶变换矩阵）、M2 (Dao, B. Chen, et al. 2022; Dao, Gu, et al. 2019; Dao, Sohoni, et al. 2020; D. Fu et al. 2024)（蝶形/monarch 矩阵）、Toeplitz 矩阵 (Poli et al. 2023; Qin, Han, Weixuan Sun, B. He, et al. 2023) 以及更为特殊的结构 (De Sa et al. 2018; Thomas et al. 2018)。

一个重要的特征是，高效（次二次，sub-quadratic）矩阵序列变换恰恰是那些拥有结构化矩阵混合器的变换。SSD 框架的一个核心成果是将 SSM 视为具有特定结构的矩阵混合器——半可分矩阵（第 3 节）。线性与二次的对偶性进而表现为结构化矩阵乘法与朴素矩阵乘法的对偶。

结构化矩阵表示通过对特定半可分（semiseparable）矩阵进行分块分解，引出了我们高效的 SSD 算法（第 6 节）。我们注意到，半可分矩阵在科学计算文献中已有充分研究，结合这些思想可能是进一步改进状态空间模型（state space models）的一条极具前景的途径。我们还建议，聚焦于矩阵混合器（matrix mixer）的视角可以为序列模型带来更多富有成效的研究方向，例如设计原则性的非因果 Mamba 变体，或者通过分析其矩阵变换结构来刻画并弥合 softmax 注意力与次二次（sub-quadratic）模型之间的差距。

## 10.3 （线性）注意力

与标准（因果）注意力相比，SSD 仅有两个主要区别。

首先，SSD 没有使用标准注意力中的 softmax 激活函数（Bahdanau, Cho, and Bengio 2015; Vaswani et al. 2017），而这正是使注意力具有二次复杂度的原因。当去掉 softmax 时，可以通过线性注意力框架（Katharopoulos et al. 2020）以线性复杂度计算序列。

其次，SSD 将 logits 矩阵乘以一个依赖于输入的 1-半可分掩码。因此，该掩码可以被视为替代了标准注意力中的 softmax。

这种半可分掩码（semiseparable mask）也可以被视为提供了位置信息。元素 $a _ { t }$ 在 RNN 意义上充当“门控”（gates），或者一种“选择”机制（参见 Mamba 论文中的讨论），它们的累积乘积 $a _ { j : i }$ 控制着位置  和  之间允许发生多大程度的交互。位置嵌入（例如正弦位置编码（sinusoidal，Vaswani et al. 2017）、AliBi (Press, N. Smith, and Lewis 2022) 和 RoPE (Su et al. 2021)）是 Transformers 的重要组成部分，通常被视为一种启发式设计，而 SSD 的 1-SS 掩码可以被看作是一种更具原则性的相对位置嵌入形式。我们注意到，这一观点同时也被 GateLoop (Katsch 2023) 提出。

状态空间对偶性的第二个视角是我们更为通用的结构化掩码注意力（structured masked attention，SMA）框架的一个特例，其中对偶性体现为一个简单的 4 向张量收缩上的不同收缩顺序。SMA 是线性注意力的强力推广，同时也远比 SSD 更加通用；其他形式的结构化掩码可能会带来相比 SSD 具有不同特性的更多高效注意力变体。

除了衍生出新模型之外，这些与注意力的联系还可以为理解 SSMs 提供其他方向。例如，我们很好奇注意力汇聚（attention sinks，Darcet et al. 2024; Xiao et al. 2024）现象是否存在于 Mamba 模型中，以及更广泛地，可解释性技术是否可以迁移到 SSMs 上 (Ali, Zimerman, and Wolf 2024)。

最后，目前已经提出了许多其他线性注意力的变体 (Arora, Eyuboglu, Timalsina, et al. 2024; Arora, Eyuboglu, Zhang, et al. 2024; Choromanski et al. 2021; H. Peng et al. 2021; Qin, Han, Weixuan Sun, Dongxu Li, et al. 2022; Qin, Weixuan Sun, et al. 2022; Schlag, Irie, and Schmidhuber 2021; Zhang et al. 2024; Zheng, C. Wang, and Kong 2022)（其中几种的描述见第 4.1.3 节），我们预期许多技术都可以迁移到 SSMs 中（例如第 7.3 节）。

我们强调，SSD 并不能泛化标准 softmax 注意力（softmax attention），也不能泛化注意力核矩阵上任何没有有限特征映射 $\phi$ 的其他变换。与通用注意力相比，SSD 的优势在于拥有一个可控的状态扩展因子 $N$ 来压缩历史信息，而二次方注意力的整个历史缓存则随着序列长度 $T \gg N$ 扩展。同时期的工作已经开始研究这些表示的权衡，例如在复制（copying）和上下文学习（in-context learning）任务上（Akyürek et al. 2024; Grazzi et al. 2024; Jelassi et al. 2024; Park et al. 2024）。我们注意到 Mamba-2 在其中某些能力上相比 Mamba 有显著提升（例如第 9.1 节中的 MQAR 结果所示），但仍有更多内容有待进一步理解。

## 10.4 相关模型

最后，我们重点介绍近期和同期涌现的一系列与 Mamba 和 Mamba-2 非常相似的序列模型工作。

• RetNet (Y. Sun et al. 2023) 和 TransNormerLLM (Qin, Dong Li, et al. 2023) 使用衰减项代替累加和来推广线性注意力（Linear Attention），并提出了双重并行/循环算法以及混合的“分块（chunkwise）”模式。这些算法可以看作是 SSD 的一个特例，其中 $A_t$ 是时不变的（对所有 $t$ 均为常数）；在 SMA 的解释下，掩码矩阵 $M$ 将是一个衰减矩阵 $L _ { i , j } = \gamma ^ { i - j }$。这些模型在架构上也有诸多不同。例如，由于它们源于以注意力为中心的视角，因此保留了多头注意力（MHA）模式；而 Mamba-2 源于以 SSM 为中心的模式，因此保留了多值注意力（MVA）或多扩展 SSM（MES）模式，我们证明了该模式表现更优（第 9.4 节）。

• GateLoop (Katsch 2023) 同时提出使用输入相关的衰减因子，并推导出了与 SSD 中相同的对偶二次型形式，他们称之为“代理注意力（surrogate attention）”形式。

• Gated Linear Attention (GLA) (Yang et al. 2024) 提出了一种带有数据相关门控的线性注意力变体，以及用于计算分块模式的高效算法和硬件感知实现。

• HGRN (Qin, Yang, and Zhong 2023) 引入了一种具有输入相关门控的 RNN，并在 HGRN2 (Qin, Yang, Weixuan Sun, et al. 2024) 中通过结合状态扩展技术对其进行了改进。

• Griffin (De et al. 2024) 和 RecurrentGemma (Botev et al. 2024) 表明，结合局部注意力的输入相关门控 RNN 能够与强大的现代 Transformer 极具竞争力。Jamba 还表明，将 Mamba 与少数几层注意力相结合在语言建模上表现非常出色 (Lieber et al. 2024)。

• xLSTM (Beck et al. 2024) 通过采用状态扩展思想以及其他门控、归一化和稳定化技术对 LSTM 进行了改进。

• RWKV(-4) (B. Peng, Alcaide, et al. 2023) 是一种基于不同线性注意力近似（无注意力 Transformer (S. Zhai et al. 2021)）的 RNN。最近，通过采用选择性和状态扩展的思想，它已被改进为 RWKV-5/6 (Eagle 和 Finch) 架构 (B. Peng, Goldstein, et al. 2024)。

## 11 结论

我们提出了一个基于深入研究的结构化矩阵类的理论框架，弥合了状态空间模型（SSM）与注意力变体之间的概念鸿沟。该框架深入揭示了近期 SSM（例如 Mamba）如何在语言建模任务上取得与 Transformer 相当的性能。此外，我们的理论工具通过连接两者的算法与系统层面的进展，为改进 SSM（乃至潜在的 Transformer）提供了新思路。作为实例展示，该框架指导了我们处于 SSM 与结构化注意力交叉领域的新架构（Mamba-2）的设计。

## 致谢

我们感谢 Angela Wu 对如何以数值稳定的方式高效计算 $\Delta$ 的梯度所提出的建议。感谢 Sukjun Hwang 和 Aakash Lahoti 在 MQAR 实验中提供的协助。

## 参考文献

[1] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. “GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints”. In: arXiv preprint arXiv:2305.13245 (2023).

[2] Yaroslav Aksenov, Nikita Balagansky, Sofia Maria Lo Cicero Vaina, Boris Shaposhnikov, Alexey Gorbatovski, and Daniil Gavrilov. “Linear Transformers with Learnable Kernel Functions are Better In-Context Models”. In: arXiv preprint arXiv:2402.10644 (2024).

[3] Ekin Akyürek, Bailin Wang, Yoon Kim, and Jacob Andreas. “In-Context Language Learning: Architectures and Algorithms”. In: The International Conference on Machine Learning (ICML). 2024.

[4] Ameen Ali, Itamar Zimerman, and Lior Wolf. The Hidden Attention of Mamba Models. 2024. arXiv: 2403.01590 [cs.LG].

[5] Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, and Christopher Ré. “Zoology: Measuring and Improving Recall in Efficient Language Models”. In: The International Conference on Learning Representations (ICLR). 2024.

[6] Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, Dylan Zinsley, James Zou, Atri Rudra, and Christopher Ré. “Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff”. In: The International Conference on Machine Learning (ICML). 2024.

[7] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. “Neural Machine Translation by Jointly Learning to Align and Translate”. In: The International Conference on Learning Representations (ICLR). 2015.

[8] George A Baker, George A Baker Jr, Peter Graves-Morris, and Susan S Baker. Pade Approximants: Encyclopedia of Mathematics and It’s Applications, Vol. 59 George A. Baker, Jr., Peter Graves-Morris. Vol. 59. Cambridge University Press, 1996.

[9] Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. “xLSTM: Extended Long Short-Term Memory”. In: arXiv preprint arXiv:2405.04517 (2024).

[10] Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. “Pythia: A Suite for Analyzing Large Language Models across Training and Scaling”. In: The International Conference on Machine Learning (ICML). PMLR. 2023, pp. 2397–2430.

[11] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. “PIQA: Reasoning about Physical Commonsense in Natural Language”. In: Proceedings of the AAAI conference on Artificial Intelligence. Vol. 34. 2020.

[12] Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, et al. “Gpt-NeoX-20B: An Open-source Autoregressive Language Model”. In: arXiv preprint arXiv:2204.06745 (2022).

[13] Guy E Blelloch. “Prefix Sums and Their Applications”. In: (1990).

[14] Aleksandar Botev, Soham De, Samuel L Smith, Anushan Fernando, George-Cristian Muraru, Ruba Haroun, Leonard Berrada, Razvan Pascanu, Pier Giuseppe Sessa, Robert Dadashi, et al. “RecurrentGemma: Moving Past Transformers for Efficient Open Language Models”. In: arXiv preprint arXiv:2404.07839 (2024).

[15] George EP Box, Gwilym M Jenkins, Gregory C Reinsel, and Greta M Ljung. Time Series Analysis: Forecasting and Control. John Wiley & Sons, 2015.

[16] James Bradbury, Stephen Merity, Caiming Xiong, and Richard Socher. “Quasi-recurrent Neural Networks”. In: arXiv preprint arXiv:1611.01576 (2016).

[17] William Brandon, Aniruddha Nrusimha, Kevin Qian, Zachary Ankner, Tian Jin, Zhiye Song, and Jonathan Ragan-Kelley. “Striped attention: Faster ring attention for causal transformers”. In: arXiv preprint arXiv:2311.09431 (2023).

[18] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. “Language Models are Few-shot Learners”. In: Advances in Neural Information Processing Systems (NeurIPS) 33 (2020), pp. 1877–1901.

[19] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. “Rethinking Attention with Performers”. In: The International Conference on Learning Representations (ICLR). 2021.

[20] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. “PaLM: Scaling Language Modeling with Pathways”. In: Journal of Machine Learning Research 24.240 (2023), pp. 1–113. url: http://jmlr.org/papers/v24/22-1144.html.

[21] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. “Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling”. In: arXiv preprint arXiv:1412.3555 (2014).

[22] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. “Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge”. In: arXiv preprint arXiv:1803.05457 (2018).

[23] Tri Dao. “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning”. In: The International Conference on Learning Representations (ICLR). 2024.

[24] Tri Dao, Beidi Chen, Nimit S Sohoni, Arjun Desai, Michael Poli, Jessica Grogan, Alexander Liu, Aniruddh Rao, Atri Rudra, and Christopher Ré. “Monarch: 用于高效且准确训练的高表现力结构化矩阵 (Monarch: Expressive structured matrices for efficient and accurate training)”. In: International Conference on Machine Learning (ICML). PMLR. 2022, pp. 4690–4721.

[25] Tri Dao, Daniel Y Fu, Khaled K Saab, Armin W Thomas, Atri Rudra, and Christopher Ré. “Hungry Hungry Hippos: 迈向基于状态空间模型的语言建模 (Hungry Hungry Hippos: Towards Language Modeling with State Space Models)”. In: The International Conference on Learning Representations (ICLR). 2023.

[26] Tri Dao, Albert Gu, Matthew Eichhorn, Atri Rudra, and Christopher Ré. “利用蝶形因子分解学习线性变换的快速算法 (Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations)”. In: The International Conference on Machine Learning (ICML). 2019.

[27] Tri Dao, Nimit Sohoni, Albert Gu, Matthew Eichhorn, Amit Blonder, Megan Leszczynski, Atri Rudra, and Christopher Ré. “Kaleidoscope: 适用于所有结构化线性映射的高效可学习表示 (Kaleidoscope: An Efficient, Learnable Representation for All Structured Linear Maps)”. In: The International Conference on Learning Representations (ICLR). 2020.

[28] Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. “视觉 Transformer 需要寄存器 (Vision Transformers Need Registers)”. In: The International Conference on Learning Representations (ICLR). 2024.

[29] Soham De, Samuel L Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru, Albert Gu, Ruba Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, et al. “Griffin: 结合门控线性循环与局部注意力的高效语言模型 (Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models)”. In: arXiv preprint arXiv:2402.19427 (2024).

[30] Christopher De Sa, Albert Gu, Rohan Puttagunta, Christopher Ré, and Atri Rudra. “结构化稠密矩阵向量乘法的双管齐下进展 (A Two-Pronged Progress in Structured Dense Matrix Vector Multiplication)”. In: Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms. SIAM. 2018, pp. 1060–1079.

[31] Hantian Ding, Zijian Wang, Giovanni Paolini, Varun Kumar, Anoop Deoras, Dan Roth, and Stefano Soatto. “更少的截断可提升语言建模能力 (Fewer truncations improve language modeling)”. In: arXiv preprint arXiv:2404.10830 (2024).

[32] Yuli Eidelman and Israel Gohberg. “论一类新的结构化矩阵 (On a new class of structured matrices)”. In: Integral Equations and Operator Theory 34.3 (1999), pp. 293–324.

[33] Dan Fu, Simran Arora, Jessica Grogan, Isys Johnson, Evan Sabri Eyuboglu, Armin Thomas, Benjamin Spector, Michael Poli, Atri Rudra, and Christopher Ré. “Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture”. In: Advances in Neural Information Processing Systems 36 (2024).

[34] Daniel Y Fu, Elliot L Epstein, Eric Nguyen, Armin W Thomas, Michael Zhang, Tri Dao, Atri Rudra, and Christopher Ré. “Simple Hardware-Efficient Long Convolutions for Sequence Modeling”. In: The International Conference on Machine Learning (ICML) (2023).

[35] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. “The Pile: An 800GB Dataset of Diverse Text for Language Modeling”. In: arXiv preprint arXiv:2101.00027 (2020).

[36] Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A Framework for Few-shot Language Model Evaluation. Version v0.0.1. Sept. 2021. doi: 10.5281/zenodo.5371628. url: https://doi.org/10.5281/zenodo.5371628.

[37] Paolo Glorioso, Quentin Anthony, Yury Tokpanov, James Whittington, Jonathan Pilault, Adam Ibrahim, and Beren Millidge. “Zamba: A Compact 7B SSM Hybrid Model”. In: arXiv preprint arXiv:2405.16712 (2024).

[38] Riccardo Grazzi, Julien Siems, Simon Schrodi, Thomas Brox, and Frank Hutter. “Is Mamba Capable of In-Context Learning?” In: arXiv preprint arXiv:2402.03170 (2024).

[39] Albert Gu. “Modeling Sequences with Structured State Spaces”. 博士学位论文. Stanford University, 2023.

[40] Albert Gu and Tri Dao. “Mamba: Linear-Time Sequence Modeling with Selective State Spaces”. In: arXiv preprint arXiv:2312.00752 (2023).

[41] Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Ré. “HiPPO: Recurrent Memory with Optimal Polynomial Projections”. In: Advances in Neural Information Processing Systems (NeurIPS). 2020.

[42] Albert Gu, Karan Goel, and Christo<sub>p</sub>her Ré. “Eficientl<sub>y</sub> Modelin<sub>g</sub> Lon<sub>g</sub> Se<sub>q</sub>uences with Structured State S<sub>p</sub>aces”. In: The International Conference on Learning Representations (ICLR). 2022.

[43] Albert Gu, Ankit Gu<sub>p</sub>ta, Karan Goel, and Christo<sub>p</sub>her Ré. “On the Parameterization and Initialization of Dia<sub>g</sub>onal State Space Models”. In: Advances in Neural Information Processing Systems (NeurIPS). 2022.

[44] Albert Gu, Is<sub>y</sub>s Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, and Christo<sub>p</sub>her Ré. “Combinin<sub>g</sub> Recurrent, Convolutional, and Continuous-time Models with the Linear State Space Layer”. In: Advances in Neural Information Processing Systems (NeurIPS). 2021.

[45] Albert Gu, Is<sub>y</sub>s Johnson, Aman Timalsina, Atri Rudra, and Christo<sub>p</sub>her Ré. “How to Train Your HIPPO: State S<sub>p</sub>ace Models with Generalized Basis Projections”. In: The International Conference on Learning Representations (ICLR). 2023.

[46] Ankit Gu<sub>p</sub>ta, Albert Gu, and Jonathan Berant. “Dia<sub>g</sub>onal State S<sub>p</sub>aces are as Efective as Structured State S<sub>p</sub>aces”. In: Advances in Neural Information Processing Systems 35 (2022), pp. 22982–22994.

[47] Dan Hendrycks and Kevin Gimpel. “Gaussian Error Linear Units (GELUs)”. In: arXiv preprint arXiv:1606.08415 (2016).

[48] W Daniel Hillis and Guy L Steele Jr. “Data Parallel Algorithms”. In: Communications of the ACM 29.12 (1986), <sub>pp</sub>. 1170–1183.

[49] Sepp Hochreiter and Jürgen Schmidhuber. “Long Short-Term Memory”. In: Neural Computation 9.8 (1997), pp. 1735– 1780.

[50] Jordan Hofmann, Sebastian Bor<sub>g</sub>eaud, Arthur Mensch, Elena Buchatska<sub>y</sub>a, Trevor Cai, Eliza Rutherford, Die<sub>g</sub>o de Las Casas<sub>,</sub> Lisa Anne Hendricks<sub>,</sub> Johannes Welbl<sub>,</sub> Aidan Clark<sub>,</sub> et al. “An Em<sub>p</sub>irical Anal<sub>y</sub>sis of Com<sub>p</sub>ute-

Optimal Large Language Model Training”. In: Advances in Neural Information Processing Systems (NeurIPS) 35 (2022), <sub>pp</sub>. 30016–30030.

[51] Sam<sub>y</sub> Jelassi, David Brandfonbrener, Sham M Kakade, and Eran Malach. “Re<sub>p</sub>eat After Me: Transformers Are Better Than State Space Models at Copying”. In: The International Conference on Machine Learning (ICML). 2024.

[52] An<sub>g</sub>elos Katharo<sub>p</sub>oulos, A<sub>p</sub>oorv V<sub>y</sub>as, Nikolaos Pa<sub>pp</sub>as, and François Fleuret. “Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention”. In: International Conference on Machine Learning. PMLR. 2020, <sub>pp</sub>. 5156–5165.

[53] Tobias Katsch. “GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling”. In: arXiv preprint arXiv:2311.01927 (2023).

[54] Shiva Kaul. “Linear Dynamical Systems as a Core Computational Primitive”. In: Advances in Neural Information Processing Systems 33 (2020), pp. 16808–16820.

[55] Vija<sub>y</sub> Anand Korthikanti, Jared Cas<sub>p</sub>er, San<sub>g</sub>ku<sub>g</sub> L<sub>y</sub>m, Lawrence McAfee, Michael Andersch, Mohammad Shoe<sub>y</sub>bi, and Bryan Catanzaro. “Reducing activation recomputation in large transformer models”. In: Proceedings of Machine Learning and Systems 5 (2023).

[56] James Lee-Thor<sub>p</sub>, Joshua Ainslie, Il<sub>y</sub>a Eckstein, and Santia<sub>g</sub>o Ontanon. “Fnet: Mixin<sub>g</sub> tokens with fourier transforms”. In: arXiv preprint arXiv:2105.03824 (2021).

[57] Tao Lei. “When Attention Meets Fast Recurrence: Training Language Models with Reduced Compute”. In: Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing. 2021, pp. 7633–7648.

[58] Tao Lei, Yu Zhan<sub>g</sub>, Sida I Wan<sub>g</sub>, Hui Dai, and Yoav Artzi. “Sim<sub>p</sub>le Recurrent Units for Hi<sub>g</sub>hl<sub>y</sub> Parallelizable Recurrence”. In: arXiv preprint arXiv:1709.02755 (2017).

[59] Yuhon<sub>g</sub> Li, Tianle Cai, Yi Zhan<sub>g</sub>, Demin<sub>g</sub> Chen, and Debadee<sub>p</sub>ta De<sub>y</sub>. “What Makes Convolutional Models Great on Long Sequence Modeling?” In: The International Conference on Learning Representations (ICLR). 2023.

[60] O<sub>p</sub>her Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Ita<sub>y</sub> Dalmedi<sub>g</sub>os, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, et al. “Jamba: A Hybrid Transformer-Mamba Language Model”. In: arXiv preprint arXiv:2403.19887 (2024).

[61] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. “World Model on Million-Len<sub>g</sub>th Video And Lan<sub>g</sub>ua<sub>g</sub>e With RingAttention”. In: arXiv preprint arXiv:2402.08268 (2024).

[62] Hao Liu, Matei Zaharia, and Pieter Abbeel. “Rin<sub>g</sub> attention with blockwise transformers for near-infinite context”. In: arXiv preprint arXiv:2310.01889 (2023).

[63] Chris Lu, Yannick Schroecker, Albert Gu, Emilio Parisotto, Jakob Foerster, Satinder Sin<sub>g</sub>h, and Fer<sub>y</sub>al Behbahani. “Structured State Space Models for In-Context Reinforcement Learning”. In: Advances in Neural Information Processing Systems (NeurIPS). 2023.

[64] Xuezhe Ma, Chuntin<sub>g</sub> Zhou, Xian<sub>g</sub> Kon<sub>g</sub>, Junxian He, Lian<sub>g</sub>ke Gui, Graham Neubi<sub>g</sub>, Jonathan Ma<sub>y</sub>, and Luke Zettlemoyer. “Mega: Moving Average Equipped Gated Attention”. In: The International Conference on Learning Representations (ICLR). 2023.

[65] Eric Martin and Chris Cundy. “Parallelizing Linear Recurrent Neural Nets Over Sequence Length”. In: The International Conference on Learning Representations (ICLR). 2018.

[66] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. “Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering”. In: arXiv preprint arXiv:1809.02789 (2018).

[67] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. “In-context Learning and Induction Heads”. In: Transformer Circuits Thread (2022). https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

[68] Antonio Orvieto, Samuel L Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, and Soham De. “Resurrecting Recurrent Neural Networks for Long Sequences”. In: The International Conference on Machine Learning (ICML). 2023.

[69] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. “The LAMBADA Dataset: Word Prediction Requiring a Broad Discourse Context”. In: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics. 2016, pp. 1525–1534.

[70] Jongho Park, Jaeseung Park, Zheyang Xiong, Nayoung Lee, Jaewoong Cho, Samet Oymak, Kangwook Lee, and Dimitris Papailiopoulos. “Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks”. In: The International Conference on Machine Learning (ICML). 2024.

[71] Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, et al. “RWKV: Reinventing RNNs for the Transformer Era”. In: arXiv preprint arXiv:2305.13048 (2023).

[72] Bo Peng, Daniel Goldstein, Quentin Anthony, Alon Albalak, Eric Alcaide, Stella Biderman, Eugene Cheah, Teddy Ferdinan, Haowen Hou, Przemysław Kazienko, et al. “Eagle and Finch: RWKV with matrix-valued states and dynamic recurrence”. In: arXiv preprint arXiv:2404.05892 (2024).

[73] Hao Peng, Nikolaos Pappas, Dani Yogatama, Roy Schwartz, Noah A Smith, and Lingpeng Kong. “Random Feature Attention”. In: The International Conference on Learning Representations (ICLR). 2021.

[74] Clément Pernet. “Computing with Quasiseparable Matrices”. In: Proceedings of the ACM on International Symposium on Symbolic and Algebraic Computation. 2016, pp. 389–396.

[75] Clément Pernet, Hippolyte Signargout, and Gilles Villard. “Exact computations with quasiseparable matrices”. In: arXiv preprint arXiv:2302.04515 (2023).

[76] Clément Pernet and Arne Storjohann. “Time and space eficient generators for quasiseparable matrices”. In: Journal of Symbolic Computation 85 (2018), pp. 224–246.

[77] Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, and Christopher Ré. “Hyena Hierarchy: Towards Larger Convolutional Language Models”. In: The International Conference on Machine Learning (ICML). 2023.

[78] Hadi Pouransari, Chun-Liang Li, Jen-Hao Rick Chang, Pavan Kumar Anasosalu Vasu, Cem Koc, Vaishaal Shankar, and Oncel Tuzel. “Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum”. In: arXiv preprint arXiv:2405.13226 (2024).

[79] Ofir Press, Noah Smith, and Mike Lewis. “Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation”. In: International Conference on Learning Representations. 2022.

[80] Zhen Qin, Xiaodong Han, Weixuan Sun, Bowen He, Dong Li, Dongxu Li, Yuchao Dai, Lingpeng Kong, and Yiran Zhong. “Toeplitz Neural Network for Sequence Modeling”. In: The International Conference on Learning Representations (ICLR). 2023.

[81] Zhen Qin, Xiaodon<sub>g</sub> Han, Weixuan Sun, Don<sub>g</sub>xu Li, Lin<sub>gp</sub>en<sub>g</sub> Kon<sub>g</sub>, Nick Barnes, and Yiran Zhon<sub>g</sub>. “The devil in linear transformer”. In: arXiv preprint arXiv:2210.10340 (2022).

[82] Zhen Qin, Don<sub>g</sub> Li, Wei<sub>g</sub>ao Sun, Weixuan Sun, Xu<sub>y</sub>an<sub>g</sub> Shen, Xiaodon<sub>g</sub> Han, Yunshen Wei, Baohon<sub>g</sub> Lv, Xiao Luo, Yu Qiao, et al. <sup>“</sup>TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer”. In: arXiv preprint arXiv:2307.14995 (2023).

[83] Zhen Qin, Weixuan Sun, Hui Deng, Dongxu Li, Yunshen Wei, Baohong Lv, Junjie Yan, Lingpeng Kong, and Yiran Zhong. “CosFormer: Rethinking Softmax in Attention”. In: The International Conference on Learning Representations (ICLR). 2022.

[84] Zhen Qin, Son<sub>g</sub>lin Yan<sub>g</sub>, Weixuan Sun, Xu<sub>y</sub>an<sub>g</sub> Shen, Don<sub>g</sub> Li, Wei<sub>g</sub>ao Sun, and Yiran Zhon<sub>g</sub>. “HGRN2: Gated Linear RNNs with State Expansion”. In: arXiv preprint arXiv:2404.07904 (2024).

[85] Zhen Qin, Son<sub>g</sub>lin Yan<sub>g</sub>, and Yiran Zhon<sub>g</sub>. “Hierarchicall<sub>y</sub> Gated Recurrent Neural Network for Se<sub>q</sub>uence Model ing”. In: Advances in Neural Information Processing Systems 36 (2023).

[86] Ali Rahimi and Benjamin Recht. “Random Features for Large-Scale Kernel Machines”. In: Advances in Neural Information Processing Systems (NeurIPS) 20 (2007).

[87] Prajit Ramachandran, Barret Zoph, and Quoc V Le. “Swish: A Self-gated Activation Function”. In: arXiv preprint arXiv:1710.05941 7.1 (2017), p. 5.

[88] Keisuke Saka<sub>g</sub>uchi, Ronan Le Bras, Chandra Bha<sub>g</sub>avatula, and Yejin Choi. “Wino<sub>g</sub>rande: An Adversarial Wino<sub>g</sub>rad Schema Challenge at Scale”. In: Communications of the ACM 64.9 (2021), pp. 99–106.

[89] Imanol Schla<sub>g</sub>, Kazuki Irie, and Jür<sub>g</sub>en Schmidhuber. “Linear Transformers are Secretl<sub>y</sub> Fast Wei<sub>g</sub>ht Pro<sub>g</sub>rammers”. In: The International Conference on Machine Learning (ICML). PMLR. 2021, pp. 9355–9366.

[90] Noam Shazeer. “Fast Transformer Decoding: One Write-head is All You Need”. In: arXiv preprint arXiv:1911.02150 (2019).

[91] Sam Shleifer, Jason Weston, and M<sub>y</sub>le Ott. “NormFormer: Im<sub>p</sub>roved Transformer Pretrainin<sub>g</sub> with Extra Normalization”. In: arXiv preprint arXiv:2110.09456 (2021).

[92] Mohammad Shoe<sub>y</sub>bi, Mostofa Patwar<sub>y</sub>, Raul Puri, Patrick LeGresle<sub>y</sub>,Jared Cas<sub>p</sub>er, and Br<sub>y</sub>an Catanzaro. “Me<sub>g</sub>atron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism”. In: arXiv preprint arXiv:1909.08053 (2019).

[93] Jimmy TH Smith, Andrew Warrington, and Scott W Linderman. “用于序列建模的简化状态空间层（Simplified State Space Layers for Sequence Modeling）”. In: The International Conference on Learning Representations (ICLR). 2023.

[94] Jianlin Su, Yu Lu, Shenfen Pan, Ahmed Murtadha, Bo Wen, and Yunfen Liu. “Roformer: 基于旋转位置嵌入的增强型 Transformer（Roformer: Enhanced Transformer with Rotary Position Embedding）”. In: arXiv preprint arXiv:2104.09864 (2021).

[95] Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. “Retentive network: 大语言模型中 Transformer 的继任者（Retentive network: A successor to transformer for large language models）”. In: arXiv preprint arXiv:2307.08621 (2023).

[96] Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. “高效 Transformer 综述（Efficient Transformers: A Survey）”. In: ACM Computing Surveys 55.6 (2022), pp. 1–28.

[97] Chameleon Team. “Chameleon: 混合模态早期融合基础模型（Chameleon: Mixed-Modal Early-Fusion Foundation Models）”. In: arXiv preprint arXiv:2405.09818 (2024).

[98] Anna Thomas, Albert Gu, Tri Dao, Atri Rudra, and Christopher Ré. “学习具有低位移秩的压缩变换（Learning Compressed Transforms with Low Displacement Rank）”. In: Advances in Neural Information Processing Systems (NeurIPS). 2018, pp. 9052–9060.

[99] Ilya O Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, et al. “MLP-Mixer: 一种用于视觉的全 MLP 架构（MLP-Mixer: An All-MLP Architecture for Vision）”. In: Advances in Neural Information Processing Systems (NeurIPS) 34 (2021), pp. 24261–24272.

[100] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. “Llama: 开源且高效的基础语言模型（Llama: Open and Efficient Foundation Language Models）”. In: arXiv preprint arXiv:2302.13971 (2023).

[101] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. “Llama 2: 开源基础与微调对话模型（Llama 2: Open Foundation and Fine-Tuned Chat Models）”. In: arXiv preprint arXiv:2307.09288 (2023).

[102] Raf Vandebril, M Van Barel, Gene Golub, and Nicola Mastronardi. “半可分矩阵文献综述（A bibliography on semiseparable matrices）”. In: Calcolo 42 (2005), pp. 249–270.

[103] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. “Attention Is All You Need”. In: Advances in Neural Information Processing Systems (NeurIPS). 2017.

[104] Shida Wan<sub>g</sub> and Beichen Xue. “State-s<sub>p</sub>ace Models with La<sub>y</sub>er-wise Nonlinearit<sub>y</sub> are Universal A<sub>pp</sub>roximators with Exponential Decaying Memory”. In: arXiv preprint arXiv:2309.13414 (2023).

[105] Sinon<sub>g</sub> Wan<sub>g</sub>, Belinda Z Li, Madian Khabsa, Han Fan<sub>g</sub>, and Hao Ma. “Linformer: Self-attention with Linear Complexity”. In: arXiv preprint arXiv:2006.04768 (2020).

[106] Guan<sub>g</sub>xuan Xiao, Yuandon<sub>g</sub> Tian, Beidi Chen, Son<sub>g</sub> Han, and Mike Lewis. “Eficient Streamin<sub>g</sub> Lan<sub>g</sub>ua<sub>g</sub>e Models with Attention Sinks”. In: The International Conference on Learning Representations (ICLR). 2024.

[107] Yun<sub>y</sub>an<sub>g</sub> Xion<sub>g</sub>, Zhan<sub>p</sub>en<sub>g</sub> Zen<sub>g</sub>, Rudrasis Chakrabort<sub>y</sub>, Min<sub>g</sub>xin<sub>g</sub> Tan, Glenn Fun<sub>g</sub>, Yin Li, and Vikas Sin<sub>g</sub>h. “Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention”. In: Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. 2021.

[108] Son<sub>g</sub>lin Yan<sub>g</sub>, Bailin Wan<sub>g</sub>, Yikan<sub>g</sub> Shen, Rameswar Panda, and Yoon Kim. “Gated Linear Attention Transformers with Hardware-Eficient Training”. In: The International Conference on Machine Learning (ICML). 2024.

[109] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. “HellaSwa<sub>g</sub>: Can a Machine Reall<sub>y</sub> Finish Your Sentence?” In: Proceedings ofthe 57th Annual Meeting ofthe Association for Computational Linguistics. 2019.

[110] Jinle Zeng, Min Li, Zhihua Wu, Jiaqi Liu, Yuang Liu, Dianhai Yu, and Yanjun Ma. “Boosting distributed training performance of the unpadded bert model”. In: arXiv preprint arXiv:2208.08124 (2022).

[111] Shuan<sub>g</sub>fei Zhai, Walter Talbott, Nitish Srivastava, Chen Huan<sub>g</sub>, Hanlin Goh, Ruixian<sub>g</sub> Zhan<sub>g</sub>, and Josh Susskind. “An Attention Free Transformer”. In: arXiv preprint arXiv:2105.14103 (2021).

[112] Yujia Zhai, Chengquan Jiang, Leyuan Wang, Xiaoying Jia, Shang Zhang, Zizhong Chen, Xin Liu, and Yibo Zhu. “Bytetransformer: A high-performance transformer boosted for variable-length inputs”. In: 2023 IEEE International Parallel and Distributed Processing Symposium (IPDPS). IEEE. 2023, pp. 344–355.

[113] Michael Zhan , Kush Bhatia, Hermann Kumbon , and Christo her Ré. “The Hed eho & the Porcu ine: Ex ressive Linear Attentions with Softmax Mimicry”. In: The International Conference on Learning Representations (ICLR). 2024.

[114] Lin Zheng, Chong Wang, and Lingpeng Kong. “Linear complexity randomized self-attention mechanism”. In: International Conference on Machine Learning. PMLR. 2022, pp. 27011–27041.

## A 术语表

Table 8: 符号与术语表；助记符加粗。（上）常用张量维度。（下）状态空间模型或结构化掩码注意力中使用的矩阵和张量。

<table><tr><td>符号</td><td>描述</td><td>定义</td></tr><tr><td>T</td><td>时间轴或目标序列轴</td><td>定义 2.1</td></tr><tr><td>S</td><td>源序列轴（在注意力机制中）</td><td>公式 (9)</td></tr><tr><td>D</td><td>模型维度或 d_model</td><td>定义 7.1</td></tr><tr><td>N</td><td>状态/特征维度或 d_state</td><td>公式 (2) 和 (9)</td></tr><tr><td>P</td><td>注意力头维度或 d_head</td><td>定义 2.1</td></tr><tr><td>H</td><td>注意力头数或 n_head</td><td>定义 7.1</td></tr><tr><td>M</td><td>序列变换矩阵</td><td>定义 2.3</td></tr><tr><td>A</td><td>离散 SSM 循环（状态）矩阵</td><td>公式 (2)</td></tr><tr><td>B</td><td>状态空间模型输入投影（扩展）矩阵</td><td>公式 (2)</td></tr><tr><td>C</td><td>状态空间模型输出投影（收缩）矩阵</td><td>公式 (2)</td></tr><tr><td>X</td><td>输入矩阵（形状 (T,P)）</td><td>公式 (2) 和 (9)</td></tr><tr><td>Y</td><td>输出矩阵（形状 (T,P)）</td><td>公式 (2) 和 (9)</td></tr><tr><td>Q</td><td>注意力查询（Query）矩阵</td><td>公式 (9)</td></tr><tr><td>K</td><td>注意力键（Key）矩阵</td><td>公式 (9)</td></tr><tr><td>V</td><td>注意力值（Value）矩阵</td><td>公式 (9)</td></tr><tr><td>G</td><td>注意力格拉姆（Gram）矩阵</td><td> $QK^{\top}$  （或  $CB^{\top}$ ）</td></tr><tr><td>L</td><td>（结构化）掩码矩阵（在因果设定下为下三角矩阵）</td><td>定义 4.2</td></tr></table>

## B 标量 SSM 扫描（1-SS 乘法）的高效算法

在本节中，我们从结构化矩阵分解的视角详细阐述用于计算标量 SSM 扫描（scalar SSM scan）的各种算法。标量 SSM 扫描被定义为在 $N = 1$（即 $A$ 为标量）的情况下计算离散 SSM (7) 的循环部分。这通常用于以循环方式计算 SSM；特别地，对于如 S5 (J. T. Smith, Warrington, and Linderman 2023) 和 S6 (Gu and Dao 2023) 模型中 $A$ 为对角结构的情况，结构化 SSM 均可归约为该操作。

本节的目的是支持本文的一个核心主题：序列模型的高效算法可以被视为结构化矩阵乘法算法。我们在此展示的各种矩阵分解思想既与用于推导快速 SSM 算法的思想相关（第 6 节），也可以直接用作子例程。

## B.1 问题定义

设 $a$ 与 $b$ 为标量序列。标量 SSM 扫描定义为

$$
h _ {t} = a _ {t} h _ {t - 1} + b _ {t}.\tag{21}
$$

这里 $h _ { - 1 }$ 可以是表示 SSM 循环的前一个隐藏状态的任意值；除非另有说明，我们假设 $h _ { - 1 } = 0$。

我们也将式 (21) 称为 cumprodsum（累积乘积和）。请注意，当 $b = 0$（加法单位元）时，cumprodsum 退化为 cumprod（累积乘积）；当 $a = 1$（乘法单位元）时，它退化为 cumsum（累积和）。

最后，请注意在向量化形式下我们可以写为

$$
\begin{array}{c} h = M b \\ M = \left[ \begin{array}{c c c c c} 1 & & & & \\ a _ {1} & 1 & & & \\ a _ {2} a _ {1} & a _ {2} & 1 & & \\ \vdots & \vdots & \ddots & \ddots & \\ a _ {T - 1} \ldots a _ {1} & a _ {T - 1} \ldots a _ {2} & \ldots & a _ {T - 1} & 1 \end{array} \right] \end{array}
$$

换句话说，这仅仅是与 1-SS 矩阵 $M$ 的矩阵-向量乘积。

因此，我们有三种看待这一基本原语操作的视角，它们都是等价的：

• （标量）SSM 扫描。

• cumprodsum。

• 1-SS 矩阵-向量乘法。

## B.2 经典算法

我们首先介绍计算 SSM 扫描 (21) 的两种经典方法，这也是先前工作所采用的方法。

## B.2.1 串行递推

递推模式只需一次计算一个时间步 (21)。从 1-SS 乘法的角度来看，这在 3.4.1 节中也已有描述。

## B.2.2 并行结合扫描

其次，一个重要的观察是，这种递推可以转化为结合扫描 (E. Martin and Cundy 2018; J. T. Smith, Warrington, and Linderman 2023)。这一事实并非完全显而易见。例如，S5 定义了正确的结合扫描算子，然后通过死记硬背式的推导计算证明了该算子的结合律。

要理解这可以通过结合扫描来计算，一种更为清晰的方式是将多项递推转换为隐藏状态维度为 2 而非 1 的单项递推：

$$
\begin{array}{c} h _ {t} = a _ {t} h _ {t - 1} + b _ {t} \\ \left[ \begin{array}{c} h _ {t} \\ 1 \end{array} \right] = \left[ \begin{array}{c c} a _ {t} & b _ {t} \\ 0 & 1 \end{array} \right] \left[ \begin{array}{c} h _ {t - 1} \\ 1 \end{array} \right]. \end{array}
$$

那么计算所有的 $h _ { t }$ 就等价于计算这些 $2 \times 2$ 矩阵的累积乘积（前缀积）。由于矩阵乘法满足结合律，这可以通过结合扫描来计算。该结合二元算子实际上就是作用在这些特定矩阵上的矩阵乘法：

$$
\left[ \begin{array}{c c} a _ {t} & b _ {t} \\ 0 & 1 \end{array} \right] \left[ \begin{array}{c c} a _ {s} & b _ {s} \\ 0 & 1 \end{array} \right] = \left[ \begin{array}{c c} a _ {t} a _ {s} & a _ {t} b _ {s} + b _ {t} \\ 0 & 1 \end{array} \right].
$$

使顶行相等即可得到与 S5 所定义的相同的结合扫描算子（associative scan operator）：

$$
(a _ {t}, b _ {t}) \otimes (a _ {s}, b _ {s}) = (a _ {t} a _ {s}, a _ {t} b _ {s} + b _ {t}).\tag{22}
$$

结合扫描之所以重要，是因为它们可以通过分治算法进行并行化（Blelloch 1990）。我们省略了该算法的细节，而是展示整个结合 SSM 扫描算法可以通过矩阵分解从头推导出来（附录 B.3.5）。

## B.3 基于结构化矩阵分解的高效算法

我们讨论了几种用于计算 SSM 扫描的算法，均从寻找 1-SS 矩阵  的结构化矩阵分解的角度出发。这些算法或计算模式包括：

• 膨胀模式（dilated mode）：信息一次传播 1, 2, 4, 8, . . . 步。

• 状态传递模式（state-passing mode）：信息按块向前传播。

• 完全循环模式（fully recurrent mode）：一次递增一步，这是状态传递模式的特例。

• 分块分解并行模式（block decomposition parallel mode）：将  划分为层次化分块。

• 扫描模式（scan mode）：将  划分为大小相等的块并进行递归归约。

## B.3.1 膨胀模式

该模式以一种涉及增加“步幅”（strides）的特定方式分解 1-SS 矩阵。这可以通过一个具体示例得到最好的说明：

$$
\begin{array}{l} M = \left[ \begin{array}{c c c c c c c c c} a _ {0: 0} & & & & & & & \\ a _ {1: 0} & a _ {1: 1} & & & & & & \\ a _ {2: 0} & a _ {2: 1} & a _ {2: 2} & & & & & \\ a _ {3: 0} & a _ {3: 1} & a _ {3: 2} & a _ {3: 3} & & & & \\ a _ {4: 0} & a _ {4: 1} & a _ {4: 2} & a _ {4: 3} & a _ {4: 4} & & \\ a _ {5: 0} & a _ {5: 1} & a _ {5: 2} & a _ {5: 3} & a _ {5: 4} & a _ {5: 5} & \\ a _ {6: 0} & a _ {6: 1} & a _ {6: 2} & a _ {6: 3} & a _ {6: 4} & a _ {6: 5} & a _ {6: 6} \\ a _ {7: 0} & a _ {7: 1} & a _ {7: 2} & a _ {7: 3} & a _ {7: 4} & a _ {7: 5} & a _ {7: 6} & a _ {7: 7} \end{array} \right] \\ = \left[ \begin{array}{c c c c c c c c c} a _ {0: 0} & & & & & & & \\ & a _ {1: 1} & & & & \\ & & a _ {2: 2} & & & \\ & & & a _ {3: 3} & & \\ a _ {4: 0} & & & & a _ {4: 4} & \\ & a _ {5: 1} & & & & a _ {5: 5} \\ & & a _ {6: 2} & & & & a _ {6: 6} \\ & & & a _ {7: 3} \end{array} \right] \left[ \begin{array}{c c c c c c c c c} a _ {0: 0} & & & & & \\ & a _ {1: 1} & & & \\ a _ {2: 0} & & a _ {2: 2} & \\ & a _ {3: 1} & & a _ {3: 3} \\ & & a _ {4: 2} & & a _ {4: 4} \\ & & a _ {5: 3} & & a _ {5: 5} \\ & & & a _ {6: 4} & & a _ {6: 6} \\ & & & & a _ {7: 5} \end{array} \right] \left[ \begin{array}{c c c c c c c c c} a _ {0: 0} & & & & \\ a _ {1: 0} & a _ {1: 1} & \\ & a _ {2: 1} & a _ {2: 2} \\ & & a _ {3: 2} & a _ {3: 3} \\ & & a _ {4: 3} & a _ {4: 4} \\ & & a _ {5: 4} & a _ {5: 5} \\ & & a _ {6: 5} & a _ {6: 6} \\ & & \texttt \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit\textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \textit \textbf \tag
$$

注意，这与空洞卷积（dilated convolutions）的计算非常相似。

我们还注意到，这种因式分解表明 1-SS 矩阵是蝶形矩阵（butterfly matrices）的一个特例，蝶形矩阵是另一类广泛且基础的结构化矩阵（Dao, Gu, et al. 2019; Dao, Sohoni, et al. 2020）。

Remark 8. 这种算法有时被称为“工作量低效但更具可并行性”的前缀和算法（Hillis and Steele Jr 1986），因为它使用 $O(T \log(T))$ 次操作，但其深度/跨度仅为工作量高效的结合律扫描算法（associative scan algorithm）的一半。

## B.3.2 状态传递（分块）模式

该模式可视为标准循环模式的推广：我们不再一次向前传递一步循环状态 $h$，而是在任意长度 $Q$ 的分块上计算结果，并通过该分块传递状态。这也可以通过 1-SS 矩阵的简单分块分解推导得出。

Remark 9. 虽然我们称之为“状态传递”（state-passing）以指代状态如何从一个局部片段传递到另一个局部片段，但这与相关模型中提出的“分块”（chunkwise）算法密切相关（Y. Sun et al. 2023; Yang et al. 2024）。

考虑以“分块”形式计算 $h = M b$：对于某个索引 $k \in [ T ]$，我们希望计算 $h _ { 0 : k }$ 即截至索引 $k$ 的输出，并有一种方法将该问题规约为索引 $[k : T]$ 上的较小子问题。

我们将 $M$ 写作

$$
M = \left[ \begin{array}{c c c c c c c} a _ {0: 0} & & & & & \\ a _ {1: 0} & a _ {1: 1} & & & & \\ \vdots & & \ddots & & & \\ a _ {k - 1: 0} & \ldots & \ldots & a _ {k - 1: k - 1} & & \\ a _ {k: 0} & \ldots & \ldots & a _ {k: k - 1} & a _ {k: k} & & \\ \vdots & & & \vdots & \vdots & \ddots & \\ a _ {T - 1: 0} & \ldots & \ldots & a _ {T - 1: k - 1} & a _ {T - 1: k} & \ldots & a _ {T - 1: T - 1} \end{array} \right]
$$

令左上三角为 $M _ { L }$，右下三角为 $M _ { R }$（左子问题和右子问题），左下部分为 $M _ { C }$。以相同的方式将 $b$ 划分为 $b _ { L } = b _ { 0 : k }$ 和 $b _ { R } = b _ { k : T }$。注意

$$
M b = \left[ \begin{array}{c} M _ {L} b _ {L} \\ M _ {R} b _ {R} + M _ {C} b _ {L} \end{array} \right]
$$

此外，$M _ { C }$ 具有秩 1 分解（这本质上是半可分矩阵（semiseparable matrices）的定义性质）

$$
M _ {C} = \left[ \begin{array}{c} a _ {k: k} \\ \vdots \\ a _ {T - 1: k} \end{array} \right] a _ {k} \left[ \begin{array}{c c c} a _ {k - 1: 0} & \dots & a _ {k - 1: k - 1} \end{array} \right]
$$

因此

$$
M _ {C} b _ {L} = \left[ \begin{array}{c} a _ {k: k} \\ \vdots \\ a _ {T - 1: k} \end{array} \right] a _ {k} \cdot (M b) _ {k - 1}.
$$

这里我们将 $( M b ) _ { k - 1 } = h _ { k - 1 }$ 视为左侧块的“最终 $s \mathrm { t a t e } ^ { \boldsymbol { \mathbf { \mathit { \phi } } } }$”，因为 $\vec { M _ { C } { ' s } }$ 分解中的行向量与 $M _ { L }$ 的最后一行相同。此外，注意到 $\vec { M _ { C } { ' } s }$ 分解中的列向量与 $M _ { R } . ^ { 7 }$ 的第一列相同。因此

$$
M _ {R} b _ {R} + M _ {C} b _ {L} = M _ {R} \left[ \begin{array}{c} a _ {k} h _ {k - 1} + b _ {k} \\ b _ {k + 1} \\ \vdots \\ b _ {T - 1} \end{array} \right]
$$

最后，我们利用 $M _ { L }$ 和 $M _ { R }$ 与原始矩阵 $M$ 自相似的观察；这两个规模更小的 1-SS 矩阵乘法的结果可以使用任意算法进行计算。总体而言，该算法的执行步骤如下：

1. 使用任意所需的方法（即本节中用于 1-SS 乘法的任意方法）计算前半部分结果 $h _ { 0 : k }$。

2. 计算最终状态 $h _ { k - 1 }$。

3. 将状态推进一步以修改 $b _ { k }$。

4. 使用任何所需的方法计算答案的右半部分 $h _ { k : T }$。

换句话说，我们将左子问题作为一个黑盒进行计算，将其最终状态传递给右问题，并将右子问题也作为一个黑盒进行计算。

该方法的效用来自于更复杂的设置，例如在一般的半可分（semiseparable）情况下，以及当输入具有额外的“批次”（batch）维度时（换句话说，这是矩阵-矩阵乘法而非矩阵-向量乘法）。在这种情况下，我们可以对分块（chunks）使用另一种算法（对应于通过 $M _ { L }$ 和 $M _ { R }$ 进行矩阵乘法），该算法不会实体化（materialize）全部的隐藏状态 $h$。相反，我们跳过隐藏状态，以另一种方式直接计算最终状态 $h _ { k - 1 }$，然后将该状态“传递”给下一个分块。

复杂度。该方法可以做到非常高的工作效率（work-efficient），因为步骤 2-3 仅需常数时间。因此，假设两个子问题（步骤 1 和 4）是在线性时间内完成的，整个方法的时间复杂度也是线性的。

缺点在于它也是串行的。

## B.3.3 完全循环模式

请注意，完全循环模式（其中递推每次演化一步 (21)）仅仅是分块大小为 1 的状态传递模式的一个实例。

## B.3.4 （并行）块分解模式

这使用了与状态传递模式（state-passing mode）相同的矩阵分解，但以不同的顺序计算子问题，从而在计算量与并行度之间进行权衡。

通常，我们将 $M$ 写作

$$
M = \left[ \begin{array}{c c c c c} 1 & & & & \\ a _ {1} & 1 & & & \\ a _ {2} a _ {1} & a _ {2} & 1 & & \\ \vdots & \vdots & \ddots & \ddots & \\ a _ {T - 1} \ldots a _ {1} & a _ {T - 1} \ldots a _ {2} & \ldots & a _ {T - 1} & 1 \end{array} \right] = \left[ \begin{array}{c c c c c} 1 & & & & \\ - a _ {1} & 1 & & \\ 0 & - a _ {2} & 1 & & \\ \vdots & \vdots & \ddots & \ddots & \\ 0 & 0 & \ldots & - a _ {T - 1} & 1 \end{array} \right] ^ {- 1}
$$

关键的观察仍然是 $M$ 的左下象限是秩 1（rank-1）矩阵。除了直接观察外，另一种理解该性质的方法是利用右式（RHS），注意到其左下象限是一个平凡的秩 1 矩阵（除右上角为 $\mathrm { i } s - a _ { T / 2 }$ 外全为 0），并利用伍德伯里求逆公式（Woodbury inversion formula）可知左式（LHS）的左下角也必然是秩 1。这也提供了一种推导秩 1 分解的方法，该分解可以通过直接观察来验证：

$$
\begin{array}{l} M _ {\text {lower - left - quadrant}} = \left[ \begin{array}{c c c} (a _ {T / 2} \ldots a _ {1}) & \ldots & a _ {T / 2} \\ \vdots & \ddots & \vdots \\ (a _ {T - 1} \ldots a _ {T / 2} a _ {T / 2 - 1} \ldots a _ {1}) & \ldots & (a _ {T - 1} \ldots a _ {T / 2}) \end{array} \right] \\ = \left[ \begin{array}{c} a _ {T / 2} \\ \vdots \\ a _ {T - 1} \ldots a _ {T / 2} \end{array} \right] \left[ \begin{array}{c c c c} (a _ {T / 2 - 1} \ldots a _ {1}) & \ldots & a _ {T / 2 - 1} & 1 \end{array} \right]. \end{array}
$$

第二个观察是该矩阵具有自相似性：任何主子矩阵都具有相同的形式。特别是，左上和右下象限都是 1-SS 矩阵（1-半可分矩阵）。

这提供了一种执行矩阵乘法（乘以 ）的简便方法：在两半部分（即左上和右下）上并行递归，然后处理左下子矩阵。分治算法中的这一“合并”步骤非常简单，因为该子矩阵的秩为 1。这就产生了一种并行算法。

复杂度。与状态传递算法类似，该方法使用了秩结构半可分矩阵的相同分块分解。不同之处在于我们在两个子问题上并行递归，而状态传递算法则是先处理左侧子问题，再处理右侧子问题。这使得算法的深度/跨度（depth/span）从线性降低到了 $\log()$。权衡之处在于，合并步骤（处理秩为 1 的左下子矩阵）需要线性的工作量而不是常数工作量，因此总工作量是 $O( \log())$ 而不是线性的。

另外需要注意，在递归中，我们可以随时停止并以任何其他方式计算子问题。这是 SSD 算法（第 6 节）背后的一个主要思想，我们在小子问题上切换到了对偶二次注意力公式。

## B.3.5 结合扫描模式

状态传递（分块）算法具有线性的计算工作量，但也涉及串行操作。

分块矩阵归约和空洞模式是可并行的：它们具有 $\log()$ 的深度/跨度。然而，它们会产生额外的工作量（$O( \log())$）。

正如附录 B.2.2（Appendix B.2.2）中所述，存在一种算法，通过利用结合扫描（associative scan，也称为前缀扫描 prefix scan）算法（Baker et al. 1996），可以同时实现 $O(\log N)$ 深度和 $O(N)$ 工作量。该算法从 SSM 扫描或 cumprodsum 的视角最容易理解，但即便如此也并不显然：它需要单独推导出一个结合运算符 (22)，然后将并行/结合/前缀扫描算法作为一个黑盒来利用（Blelloch 1990）。

在此，我们展示了实际上可以通过利用一种不同的矩阵分解来推导出这种并行扫描：

<table><tr><td rowspan="7"> $M =$ </td><td> $\begin{bmatrix} a_{0:0} & & \\ a_{1:0} & a_{1:1} \end{bmatrix}$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="6"> $\begin{array}{c|cc}a_{2:0} & a_{2:1} & a_{2:2}\\ a_{3:0} & a_{3:1} & a_{3:2} & a_{3:3}\\ \hline a_{4:0} & a_{4:1} & a_{4:2} & a_{4:3}\\ a_{5:0} & a_{5:1} & a_{5:2} & a_{5:3}\\ \hline a_{6:0} & a_{6:1} & a_{6:2} & a_{6:3}\\ a_{7:0} & a_{7:1} & a_{7:2} & a_{7:3}\\ \end{array}$ </td><td> $a_{2:2}$ </td><td></td><td></td><td></td><td></td></tr><tr><td> $a_{3:2} & a_{3:3}$ </td><td></td><td></td><td></td><td></td></tr><tr><td> $a_{4:4}$ </td><td></td><td></td><td></td><td></td></tr><tr><td> $a_{5:4} & a_{5:5}$ </td><td></td><td></td><td></td><td></td></tr><tr><td> $a_{6:4} & a_{6:5} & a_{6:6}$ </td><td></td><td></td><td></td><td></td></tr><tr><td> $a_{7:4} & a_{7:5} & a_{7:6} & a_{7:7}$ </td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="6"> $=$ </td><td> $\begin{bmatrix} a_{0:0} & & \\ a_{1:0} & a_{1:1} \end{bmatrix}$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2"> $\begin{bmatrix} a_{2:2} & a_{2:1} & [a_{1:0}]^{\top}\\ a_{3:2} & a_{4:1} & [a_{1:0}]^{\top}\\ \end{bmatrix}$ </td><td> $a_{2:2}$ </td><td></td><td></td><td></td><td></td></tr><tr><td> $a_{3:2} & a_{3:3}$ </td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2"> $\begin{bmatrix} a_{4:4} & a_{4:1} & [a_{1:0}]^{\top}\\ a_{5:4} & a_{6:1} & [a_{1:0}]^{\top}\\ \end{bmatrix}$ </td><td rowspan="2"> $\begin{bmatrix} a_{4:4} & a_{4:3} & [a_{3:2}]^{\top}\\ a_{5:4} & a_{6:3} & [a_{3:3}]^{\top}\\ \end{bmatrix}$ </td><td> $a_{4:4}$ </td><td></td><td></td><td></td></tr><tr><td> $a_{5:4} & a_{5:5}$ </td><td></td><td></td><td></td></tr><tr><td> $\begin{bmatrix} a_{6:6} & a_{6:1} & [a_{1:0}]^{\top}\\ a_{7:6} & a_{7:1} & [a_{1:1}]^{\top}\\ \end{bmatrix}$ </td><td> $\begin{bmatrix} a_{6:6} & a_{6:3} & [a_{3:2}]^{\top}\\ a_{7:6} & a_{6:1} & [a_{5:4}]^{\top}\\ \end{bmatrix}$ </td><td> $\begin{bmatrix} a_{6:6} & a_{6:1} & [a_{5:4}]^{\top}\\ a_{7:6} & a_{7:7} \end{bmatrix}$ </td><td> $a_{6:6}$ </td><td></td><td></td></tr></table>

现在我们分三个阶段进行。

阶段 1。首先我们计算乘法  中每个对角块的结果。这会产生两个数，但第一个元素保持不变。例如，第二个块将计算 $b _ { 2 }$ 和 $a _ { 3 } b _ { 2 } + b _ { 3 }$

阶段 2。现在考虑在矩阵的严格下三角部分中被分解为秩 1 矩阵的每个 $2 \times 2$ 块。请注意，每个右侧行向量都与其所在列的对角块中的底端行向量相同：特别是 $[ a _ { 1 : 0 } \ a _ { 1 : 1 } ] , [ a _ { 3 : 2 } \ a _ { 3 : 3 } ]$ 以及 $\left[ a _ { 5 : 4 } ~ a _ { 5 : 5 } \right]$ 行。

因此，我们已经从阶段 1 中获得了这些的结果，即阶段 1 中所有 $T / 2$ 个子问题的第二个元素。如果我们将该元素数组称为 $b ^ { \prime }$（大小为 $b$ 的一半），那么我们需要将 $b ^ { \prime }$ 乘以由 $a _ { 3 : - 1 } , a _ { 3 : 1 } , a _ { 5 : 3 } , a _ { 7 : 5 }$ 生成的 1-SS 矩阵。

阶段 3。最后，通过乘以左侧列向量，阶段 2 的每个结果都可以广播为两个最终结果：特别是 $[ a _ { 2 : 2 } \ a _ { 3 : 2 } ] ^ { \top } , [ a _ { 4 : 4 } \ a _ { 5 : 4 } ] ^ { \top }$ 和 $[ a _ { 6 : 6 } \ a _ { 7 : 6 } ] ^ { \top }$ 向量。

请注意，通过对索引进行一些差一（off-by-one）偏移，可以对该过程进行轻微修改。看待该算法的另一种等价视角是将其视为三步矩阵分解。

$$
\begin{array}{l} M = \left[ \begin{array}{c c c c c c c c c} a _ {0: 0} & & & & & & & \\ a _ {1: 0} & a _ {1: 1} & & & & & & \\ a _ {2: 0} & a _ {2: 1} & a _ {2: 2} & & & & & \\ a _ {3: 0} & a _ {3: 1} & a _ {3: 2} & a _ {3: 3} & & & & \\ a _ {4: 0} & a _ {4: 1} & a _ {4: 2} & a _ {4: 3} & a _ {4: 4} & & & \\ a _ {5: 0} & a _ {5: 1} & a _ {5: 2} & a _ {5: 3} & a _ {5: 4} & a _ {5: 5} & & \\ a _ {6: 0} & a _ {6: 1} & a _ {6: 2} & a _ {6: 3} & a _ {6: 4} & a _ {6: 5} & a _ {6: 6} \\ a _ {7: 0} & a _ {7: 1} & a _ {7: 2} & a _ {7: 3} & a _ {7: 4} & a _ {7: 5} & a _ {7: 6} & a _ {7: 7} \end{array} \right] \\ = \left[ \begin{array}{c c c c c c c c c} a _ {0: 0} & & & & & & \\ & a _ {1: 1} & & & & \\ & a _ {2: 1} & a _ {2: 2} & & \\ & & & a _ {3: 3} & \\ & & & a _ {4: 3} & a _ {4: 4} \\ & & & & & a _ {5: 5} \\ & & & & & a _ {6: 5} & a _ {6: 6} \\ & & & & & & & a _ {7: 7} \end{array} \right] \left[ \begin{array}{c c c c c c c c c} a _ {0: 0} & & & & & \\ & a _ {1: 1} & & \\ & & a _ {2: 2} & \\ & a _ {3: 1} & & a _ {3: 3} \\ & & & a _ {4: 4} \\ a _ {5: 1} & & a _ {5: 3} & & a _ {5: 5} \\ & & & & & a _ {6: 6} \\ a _ {7: 1} & & a _ {7: 3} & & a _ {7: 5} \end{array} \right] \left[ \begin{array}{c c c c c c c c c} a _ {0: 0} & & & & \\ a _ {1: 0} & a _ {1: 1} & \\ & a _ {2: 2} & \\ & a _ {3: 2} & a _ {3: 3} \\ & & a _ {4: 4} \\ a _ {5: 4} & a _ {5: 5} \\ a _ {6: 6} \\ a _ {7: 6} \end{array} \right] \end{array}
$$

注意，阶段 1 和阶段 3 需要 $O(N)$ 的工作量，而阶段 2 则归约为一个规模减半的自相似问题。很容易验证，这需要 $O(N)$ 的总工作量以及 $O(\log N)$ 的深度/跨度（depth/span）。

Remark 10. 事实上，可以看出该算法的计算图与附录 B.2.2 中描述的结合扫描（associative scan）算法完全相同。关键结论是，与其遵循（1）认识到 $M$ 定义了一个递推关系，（2）观察到该递推关系可以用结合二元运算符来定义的步骤；存在一个完全不同的视角，即只需为 $M$ 寻找一种结构化矩阵分解算法。

## C 理论细节

## C.1 补充：SSM 的闭包性质

我们在此展示半可分矩阵（semiseparable matrices）的一些附加性质，以阐明其灵活性和实用性。本节内容对于理解我们的核心结果并不是必需的。

Proposition C.1（半可分闭包性质）。半可分矩阵在若干基本运算下是封闭的。

• 加法：一个 $p$-SS 矩阵与一个 $q$-SS 矩阵的和至多为 $(p + q)$-SS。

• 乘法：一个 $p$-SS 矩阵与一个 $q$-SS 矩阵的乘积为 $(p + q)$-SS。

• 求逆：一个 $p$-SS 矩阵的逆至多为 $(p + 1)$-SS。

加法和乘法性质显而易见。求逆性质有多种证明；其中一种方法可直接由 Woodbury 逆矩阵恒等式得出，该恒等式在结构化 SSM 文献中也占据了重要地位 (Gu, Goel, and Ré 2022)。

反过来，这些性质蕴含了状态空间模型的闭包性质。

例如，加法性质表明两个并行 SSM 模型相加仍是一个 SSM。乘法性质表明对两个 SSM 进行顺序复合或链式连接仍可视为一个 SSM，其总状态维度是可加的——这是一个并非显而易见的事实。

最后，求逆性质可以让我们将 SSM 与其他类型的模型联系起来。例如，可以注意到带状矩阵是半可分的（semiseparable），因此它们的逆矩阵也是半可分的。（事实上，半可分结构族通常是通过对带状矩阵求逆来引入的 (Vandebril et al. 2005)）。此外，半可分矩阵的快速递推性质可以看作是其逆矩阵为带状矩阵的结果。

注 11。1-SS 矩阵是简单递推 (7) 这一事实，等价于 1-SS 矩阵的逆矩阵是一个 2-带状矩阵：

$$
M = \left[ \begin{array}{c c c c c} 1 & & & & \\ a _ {1} & 1 & & & \\ a _ {2} a _ {1} & a _ {2} & 1 & & \\ \vdots & \vdots & \ddots & \ddots & \\ a _ {T - 1} \ldots a _ {1} & a _ {T - 1} \ldots a _ {2} & \ldots & a _ {T - 1} & 1 \end{array} \right] = \left[ \begin{array}{c c c c c} 1 & & & & \\ - a _ {1} & 1 & & \\ 0 & - a _ {2} & 1 & & \\ \vdots & \vdots & \ddots & \ddots & \\ 0 & 0 & \ldots & - a _ {T - 1} & 1 \end{array} \right] ^ {- 1}
$$

因此 $y = M x  M ^ { - 1 } y = x ,$ 或者

$$
\left[ \begin{array}{c c c c c} 1 & & & & \\ - a _ {1} & 1 & & & \\ 0 & - a _ {2} & 1 & & \\ \vdots & \vdots & \ddots & \ddots & \\ 0 & 0 & \ldots & - a _ {T - 1} & 1 \end{array} \right] y = x.
$$

或者按元素展开，

$$
\begin{array}{l} y _ {t} - a _ {t} y _ {t - 1} = x _ {t} \\ y _ {t} = a _ {t} y _ {t - 1} + x _ {t}. \end{array}
$$

反之，我们还利用这些闭包结果证明了自回归结构化注意力（在某些假设下）必定是 SSM（状态空间模型），从而使我们能够表明，包括注意力变体在内的更广泛的高效序列模型族都可以归约为状态空间模型（附录 C.2）。

## C.2 自回归掩码注意力是半可分结构化注意力

我们证明第 5.2 节中的定理 5.2。在第 4.3 节中，我们将结构化注意力定义为掩码注意力的广泛推广，其中高效性这一性质（即核注意力的线性时间形式）被抽象为结构化矩阵乘法的高效性。然而，除了计算效率之外，标准线性注意力（Katharopoulos et al. 2020）还具有两个重要性质。首先，它是因果的，这是自回归建模等场景所必需的。此外，它具有高效的自回归生成能力。换言之，一个自回归步骤的成本——即在已经观察并预处理了 $x _ { 0 : T }$ 的情况下，观察到 $x _ { T }$ 后计算输出 <sub></sub> 的增量成本——仅需要常数时间。

在此，我们刻画了 SMA 的哪些实例具有高效的自回归能力。

在 SMA 的框架中，因果性等价于掩码  为下三角矩阵这一约束。

表征具有高效自回归特性的 $L$ 矩阵空间更为困难。我们将遵循时间序列文献中的经典定义（例如 ARIMA 过程 (Box et al. 2015)）的精神，采用自回归过程的一个严格技术性定义。

定义 C.2。我们将阶数为 $k$ 的自回归变换 $x \in \mathbb { R } ^ { T } \mapsto y \in \mathbb { R } ^ { T }$ 定义为：每个输出 $y_t$ 仅依赖于当前输入和前 $k$ 个输出：

$$
y _ {t} = \mu_ {t} x _ {t} + \ell_ {t 1} y _ {t - 1} + \dots + \ell_ {t k} y _ {t - k}.\tag{23}
$$

注意，$L$ 为累加（cumsum）矩阵的情况是 $k = 1$ 且 $y _ { t } = x _ { t } + y _ { t - 1 }$ 的特例。有了这个定义，表征高效自回归线性变换的空间便可由半可分矩阵（semiseparable matrices）的性质推导得出。定理 C.3 形式化并证明了定理 5.2。

定理 C.3。设 $L \in \mathbb { R } ^ { T \times T }$ 为阶数为 $k$ 的高效自回归变换。则 $L$ 是一个阶数为 $k + 1$ 的状态空间模型（state space model）。

证明。设 $( x , y )$ 为输入和输出序列，使得 $y = L x$。重排定义 (23)，

$$
y _ {t} - \ell_ {t 1} y _ {t - 1} - \dots - \ell_ {t k} y _ {t - k} = \mu_ {t} x _ {t}.
$$

对 $t$ 进行向量化，这可以表示为一个矩阵变换：

$$
\left[ \begin{array}{c c c c c c} 1 & & & & & \\ - \ell_ {t 1} & 1 & & & & \\ \vdots & \ddots & \ddots & & \\ - \ell_ {t k} & \ldots & - \ell_ {t 1} & 1 & & \\ \vdots & \ddots & \vdots & \ddots & \ddots & \\ 0 & \ldots & - \ell_ {T - 1, k} & \ldots & - \ell_ {T - 1, 1} & 1 \end{array} \right] \left[ \begin{array}{c} y _ {0} \\ y _ {1} \\ \vdots \\ y _ {k} \\ \vdots \\ y _ {T - 1} \end{array} \right] = \left[ \begin{array}{c c c c c c} \mu_ {0} & & & & & \\ & \mu_ {1} & & & & \\ & & \ddots & & \\ & & & \mu_ {k} & & \\ & & & & \ddots & \\ & & & & & \mu_ {T - 1} \end{array} \right] \left[ \begin{array}{c} x _ {0} \\ x _ {1} \\ \vdots \\ x _ {k} \\ \vdots \\ x _ {T - 1} \end{array} \right].
$$

对角矩阵可以移至左侧并合并到 $\ell$ 系数矩阵中，该矩阵仍然是一个 $k +$ 1-带状下三角矩阵。但我们同时有 $L ^ { - 1 } y = x ,$ ，因此该矩阵的逆即为所求矩阵。

其次，根据半可分性的秩表征（定义 3.1），$k+1$-带状矩阵是 $k+1$-半可分的。根据命题 $\mathrm { C . 1 }$ ，其逆矩阵因此至多是 $k+2$-半可分的。由于带状矩阵的额外结构，可以得到略微更紧的界 $k+1$。最后，将其表征为阶数为 $k+1$ 的状态空间模型由定理 3.5 得出。□

换句话说，高效自回归注意力就是半可分 SMA。

## D 实验细节

## D.1 MQAR 细节

我们使用了 Based（Arora, Eyuboglu, Zhang, et al. 2024）中引入的任务的更难版本，其中非查询/键/值（query/key/value）标记均被替换为随机标记。与先前工作使用的通常 MQAR 变体相比，我们还使用了更多的键值对、更长的序列以及更小的模型尺寸，所有这些都使得该任务更具挑战性。

对于每个序列长度 $T \in$ {256, 512, 1024}，我们使用 $T/4$ 个键值对。词表总大小为 8192。

我们采用了一种课程训练（curriculum training）形式，训练循环遍历使用 $( T / 3 2 , T / 1 6 , T / 8 , T / 4 )$ 个键值对的数据集，其中每个数据集包含 $2 ^ { 1 8 } \approx 2 5 0 0 0 0$ 个样本，每个数据集共训练 8 个 epoch（总计约 $\cdot 2 ^ { 2 8 } \approx 2 7 0 M$ 个样本）。总批次大小为 $2 ^ { 1 8 } \approx 0 . 2 5 M$ 个标记（例如，对于 $T = 1 0 2 4$ ，批次大小为 256）。

所有方法均使用默认设置下的 2 层网络；注意力基线（attention baseline）额外引入了位置嵌入（positional embeddings）。对于每种方法，我们在模型维度 $D = \{32, 64, 128, 256\}$ 以及学习率 $\{ 1 0 ^ { - 3 . 5 } , 1 0 ^ { - 2 } , 1 0 ^ { - 2 . 5 } \}$ 上进行参数遍历。我们采用在每个 epoch 均下降的线性衰减调度策略（例如最后一个 epoch 的学习率为最大/初始学习率的 $1 / 8$）。

## D.2 缩放定律（Scaling Law）细节

所有模型均在 the Pile 上进行训练。对于缩放定律实验，我们使用 GPT2 分词器（tokenizer）。

模型规模。表 9（Table 9）列出了我们遵循 GPT3 (Brown et al. 2020) 用于缩放定律的模型规模。首先，为了统一规范，我们将 1.3B 模型的批大小（batch size）从 1M tokens 调整为 0.5M tokens。其次，我们调整了训练步数和总 token 数量，以大致匹配 Chinchilla 缩放定律 (Hoffmann et al. 2022)，该定律规定训练 token 数应与模型规模成比例增加。

训练方案。所有模型均采用 AdamW 优化器，具体配置如下：

• 梯度裁剪值 1.0

• 权重衰减 0.1

• 不使用 dropout

• 带有余弦衰减的线性学习率预热（warmup）

Table 9: (缩放定律模型规模。) 我们用于缩放实验的模型规模与超参数。（模型维度和注意力头数仅适用于 Transformer 模型。）

<table><tr><td>参数量</td><td>n_layers</td><td>d_model</td><td>n_heads / d_head</td><td>训练步数</td><td>学习率</td><td>批大小</td><td>Token数</td></tr><tr><td>125M</td><td>12</td><td>768</td><td>12 / 64</td><td>4800</td><td>6e-4</td><td>0.5M tokens</td><td>2.5B</td></tr><tr><td>350M</td><td>24</td><td>1024</td><td>16 / 64</td><td>13500</td><td>3e-4</td><td>0.5M tokens</td><td>7B</td></tr><tr><td>760M</td><td>24</td><td>1536</td><td>16 / 96</td><td>29000</td><td>2.5e-4</td><td>0.5M tokens</td><td>15B</td></tr><tr><td>1.3B</td><td>24</td><td>2048</td><td>32 / 64</td><td>50000</td><td>2e-4</td><td>0.5M tokens</td><td>26B</td></tr></table>

默认情况下，峰值学习率采用 GPT3 的设定规范。

与 GPT3 训练配方相比，我们采用了一种“改进配方”，其灵感来源于 PaLM (Chowdhery et al. 2023) 和 LLaMa (Touvron, Lavril, et al. 2023) 等主流大型语言模型所采用的改动。这些改动包括：

• 线性学习率预热（warmup）并伴随余弦衰减至 1e−5，峰值为 GPT3 设定值的 5 倍

• 无线性偏置项（linear bias terms）

• 使用 RMSNorm 代替 LayerNorm

• AdamW 超参数采用 $\beta = ( . 9 , . 9 5 )$（GPT3 设定值），而非 PyTorch 默认的 $\boldsymbol { \beta } = ( . 9 , . 9 9 9 )$

## D.3 下游评估细节

为了评估充分训练下的下游性能，我们使用 GPT-NeoX (Black et al. 2022) 分词器，在 Pile 数据集上训练了 300B token 的 Mamba-2。

除了 1.3B 和 2.7B 模型的批大小设为 1M 之外，我们使用了与扩展实验相同的超参数。对于 2.7B 模型，我们同样遵循 GPT3 规范（32 层，维度 2560）。

对于所有模型，我们均采用对应 GPT3 模型 5 倍的学习率。

对于下游评估，我们使用了 EleutherAI 的 LM evaluation harness (L. Gao, Tow, et al. 2021)，评估任务与 Mamba (Gu and Dao 2023) 相同，并额外增加了一项：

• LAMBADA (Paperno et al. 2016)

• HellaSwag (Zellers et al. 2019)

• PIQA (Bisk et al. 2020)

• ARC-challenge (P. Clark et al. 2018)

• ARC-easy：ARC-challenge 的简单子集

• WinoGrande (Sakaguchi et al. 2021)

• OpenBookQA (Mihaylov et al. 2018)

## D.4 消融实验细节

(Re)Based 细节。我们在第 9.4.3 节中的消融实验考虑了 Based (Arora, Eyuboglu, Zhang, et al. 2024) 和 Re-Based (Aksenov et al. 2024) 模型。

Based 使用二次泰勒展开 $\exp(x) \approx 1 + x + x ^ { 2 } / 2 ,$ 来近似 $\exp$ 核，这可以通过以下特征映射实现

$$
\psi_ {\text { Taylor }} (x) = \text { concatenate } (1, x, 1 / \sqrt {2} x \otimes x).
$$

ReBased 建议使用更简单的特征映射 $\psi _ { \mathrm { Q u a d r a t i c } } ( x ) = x \otimes x$，对应于核变换 $x ^ { 2 } ,$，但在此之前还会应用层归一化（layer normalization）。我们将层归一化视为我们默认 Swish 激活函数的一种替代非线性激活，并对它们的组合进行了消融。

Table 10: （零样本评估。）每种尺寸的最佳结果以粗体显示，次优结果加下划线。我们与使用各种分词器、训练量高达 300B token 的开源语言模型进行对比。Pile 指验证集划分，仅与在相同数据集和分词器（GPT-NeoX-20B）上训练的模型进行对比。对于每种模型尺寸，Mamba-2 的表现均优于 Mamba，并且通常能达到两倍于自身模型尺寸的 Pythia 的水平。

<table><tr><td>模型</td><td>分词器</td><td>PILE PPL ↓</td><td>LAMBADA PPL ↓</td><td>LAMBADA 准确率 ↑</td><td>HELLASWAG 准确率 ↑</td><td>PIQA 准确率 ↑</td><td>ARC-E 准确率 ↑</td><td>ARC-C 准确率 ↑</td><td>WINOGRANDE 准确率 ↑</td><td>OPENBOOKQA 准确率 ↑</td><td>平均准确率 ↑</td></tr><tr><td>Hybrid H3-130M</td><td>GPT2</td><td>-</td><td>89.48</td><td>25.8</td><td>31.7</td><td>64.2</td><td>44.4</td><td>24.2</td><td>50.6</td><td>27.0</td><td>38.2</td></tr><tr><td>Pythia-160M</td><td>NeoX</td><td>29.64</td><td>38.10</td><td>33.0</td><td>30.2</td><td>61.4</td><td>43.2</td><td>24.1</td><td>51.9</td><td>29.2</td><td>39.0</td></tr><tr><td>Mamba-130M</td><td>NeoX</td><td>10.56</td><td>16.07</td><td>44.3</td><td>35.2</td><td>64.5</td><td>48.0</td><td>24.2</td><td>51.9</td><td>28.8</td><td>42.4</td></tr><tr><td>Mamba-2-130M</td><td>NeoX</td><td>10.48</td><td>16.86</td><td>43.9</td><td>35.3</td><td>64.9</td><td>47.4</td><td>24.2</td><td>52.1</td><td>30.6</td><td>42.6</td></tr><tr><td>Hybrid H3-360M</td><td>GPT2</td><td>-</td><td>12.58</td><td>48.0</td><td>41.5</td><td>68.1</td><td>51.4</td><td>24.7</td><td>54.1</td><td>31.6</td><td>45.6</td></tr><tr><td>Pythia-410M</td><td>NeoX</td><td>9.95</td><td>10.84</td><td>51.4</td><td>40.6</td><td>66.9</td><td>52.1</td><td>24.6</td><td>53.8</td><td>30.0</td><td>45.6</td></tr><tr><td>Mamba-370M</td><td>NeoX</td><td>8.28</td><td>8.14</td><td>55.6</td><td>46.5</td><td>69.5</td><td>55.1</td><td>28.0</td><td>55.3</td><td>30.8</td><td>48.7</td></tr><tr><td>Mamba-2-370M</td><td>NeoX</td><td>8.21</td><td>8.02</td><td>55.8</td><td>46.9</td><td>70.5</td><td>54.9</td><td>26.9</td><td>55.7</td><td>32.4</td><td>49.0</td></tr><tr><td>Pythia-1B</td><td>NeoX</td><td>7.82</td><td>7.92</td><td>56.1</td><td>47.2</td><td>70.7</td><td>57.0</td><td>27.1</td><td>53.5</td><td>31.4</td><td>49.0</td></tr><tr><td>Mamba-790M</td><td>NeoX</td><td>7.33</td><td>6.02</td><td>62.7</td><td>55.1</td><td>72.1</td><td>61.2</td><td>29.5</td><td>56.1</td><td>34.2</td><td>53.0</td></tr><tr><td>Mamba-2-780M</td><td>NeoX</td><td>7.26</td><td>5.86</td><td>61.7</td><td>54.9</td><td>72.0</td><td>61.0</td><td>28.5</td><td>60.2</td><td>36.2</td><td>53.5</td></tr><tr><td>GPT-Neo 1.3B</td><td>GPT2</td><td>-</td><td>7.50</td><td>57.2</td><td>48.9</td><td>71.1</td><td>56.2</td><td>25.9</td><td>54.9</td><td>33.6</td><td>49.7</td></tr><tr><td>Hybrid H3-1.3B</td><td>GPT2</td><td>-</td><td>11.25</td><td>49.6</td><td>52.6</td><td>71.3</td><td>59.2</td><td>28.1</td><td>56.9</td><td>34.4</td><td>50.3</td></tr><tr><td>OPT-1.3B</td><td>OPT</td><td>-</td><td>6.64</td><td>58.0</td><td>53.7</td><td>72.4</td><td>56.7</td><td>29.6</td><td>59.5</td><td>33.2</td><td>51.9</td></tr><tr><td>Pythia-1.4B</td><td>NeoX</td><td>7.51</td><td>6.08</td><td>61.7</td><td>52.1</td><td>71.0</td><td>60.5</td><td>28.5</td><td>57.2</td><td>30.8</td><td>51.7</td></tr><tr><td>RWKV4-1.5B</td><td>NeoX</td><td>7.70</td><td>7.04</td><td>56.4</td><td>52.5</td><td>72.4</td><td>60.5</td><td>29.4</td><td>54.6</td><td>34.0</td><td>51.4</td></tr><tr><td>Mamba-1.4B</td><td>NeoX</td><td>6.80</td><td>5.04</td><td>65.0</td><td>59.1</td><td>74.2</td><td>65.5</td><td>32.8</td><td>61.5</td><td>36.4</td><td>56.4</td></tr><tr><td>Mamba-2-1.3B</td><td>NeoX</td><td>6.66</td><td>5.02</td><td>65.7</td><td>59.9</td><td>73.2</td><td>64.3</td><td>33.3</td><td>60.9</td><td>37.8</td><td>56.4</td></tr><tr><td>GPT-Neo 2.7B</td><td>GPT2</td><td>-</td><td>5.63</td><td>62.2</td><td>55.8</td><td>72.1</td><td>61.1</td><td>30.2</td><td>57.6</td><td>33.2</td><td>53.2</td></tr><tr><td>Hybrid H3-2.7B</td><td>GPT2</td><td>-</td><td>7.92</td><td>55.7</td><td>59.7</td><td>73.3</td><td>65.6</td><td>32.3</td><td>61.4</td><td>33.6</td><td>54.5</td></tr><tr><td>OPT-2.7B</td><td>OPT</td><td>-</td><td>5.12</td><td>63.6</td><td>60.6</td><td>74.8</td><td>60.8</td><td>31.3</td><td>61.0</td><td>35.2</td><td>55.3</td></tr><tr><td>Pythia-2.8B</td><td>NeoX</td><td>6.73</td><td>5.04</td><td>64.7</td><td>59.3</td><td>74.0</td><td>64.1</td><td>32.9</td><td>59.7</td><td>35.2</td><td>55.7</td></tr><tr><td>RWKV4-3B</td><td>NeoX</td><td>7.00</td><td>5.24</td><td>63.9</td><td>59.6</td><td>73.7</td><td>67.8</td><td>33.1</td><td>59.6</td><td>37.0</td><td>56.4</td></tr><tr><td>Mamba-2.8B</td><td>NeoX</td><td>6.22</td><td>4.23</td><td>69.2</td><td>66.1</td><td>75.2</td><td>69.7</td><td>36.3</td><td>63.5</td><td>39.6</td><td>59.9</td></tr><tr><td>Mamba-2-2.7B</td><td>NeoX</td><td>6.09</td><td>4.10</td><td>69.7</td><td>66.6</td><td>76.4</td><td>69.6</td><td>36.4</td><td>64.0</td><td>38.8</td><td>60.2</td></tr><tr><td>GPT-J-6B</td><td>GPT2</td><td>-</td><td>4.10</td><td>68.3</td><td>66.3</td><td>75.4</td><td>67.0</td><td>36.6</td><td>64.1</td><td>38.2</td><td>59.4</td></tr><tr><td>OPT-6.7B</td><td>OPT</td><td>-</td><td>4.25</td><td>67.7</td><td>67.2</td><td>76.3</td><td>65.6</td><td>34.9</td><td>65.5</td><td>37.4</td><td>59.2</td></tr><tr><td>Pythia-6.9B</td><td>NeoX</td><td>6.51</td><td>4.45</td><td>67.1</td><td>64.0</td><td>75.2</td><td>67.3</td><td>35.5</td><td>61.3</td><td>38.0</td><td>58.3</td></tr><tr><td>RWKV4-7.4B</td><td>NeoX</td><td>6.31</td><td>4.38</td><td>67.2</td><td>65.5</td><td>76.1</td><td>67.8</td><td>37.5</td><td>61.0</td><td>40.2</td><td>59.3</td></tr></table>

请注意，由于这些方法扩展了特征维度，我们必须投影到更小的 $q$、$k$ 维度；在 Table 7 中，对于 130M 模型使用状态大小 $d = 64$，对于 380M 模型使用 $d = 256$。对于 (Re)Based 方法，我们在应用其特征映射（feature maps）之前分别投影到 8 维和 16 维；在 130M 模型的情况下，这导致 ReBased 的总状态大小为 $8 ^ { 2 } = 6 4$，Based 的总状态大小为 $1 + 8 + 8 ^ { 2 } = 7 3 $。由于 $q$ 和 $k$ 投影较小，这些方法使用的参数量较少，因此我们相应地调整了层数。