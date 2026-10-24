# Funnel-Transformer：滤除序列冗余以实现高效语言处理

Zihang Dai<sup>∗12</sup>, Guokun Lai<sup>∗1</sup>, Yiming Yang<sup>1</sup>, Quoc V. Le<sup>2</sup> <sup>1</sup>Carnegie Mellon University（卡内基梅隆大学）, <sup>2</sup>Google AI Brain Team（谷歌 AI Brain 团队） {dzihang,guokun,yiming}@cs.cmu.edu, qvl@google.com

## 摘要

随着语言预训练的成功，迫切需要开发具备良好可扩展性且更加高效的架构，从而能够以更低的成本利用海量的无标注数据。为了提高效率，我们审视了在维持全长词元级（token-level）表示中常被忽视的冗余现象，尤其是对于那些仅需要序列单一向量表示的任务。基于这一直觉，我们提出了 Funnel-Transformer，该模型逐步将隐藏状态序列压缩为更短的序列，从而降低计算成本。更重要的是，通过将长度缩减所节省的 FLOPs 重新投入到构建更深或更宽的模型中，我们进一步提升了模型容量。此外，为了执行常见预训练目标所要求的词元级预测，Funnel-Transformer 能够通过解码器从缩减后的隐藏序列中为每个词元恢复出深层表示。实证表明，在消耗相当或更少 FLOPs 的情况下，Funnel-Transformer 在包括文本分类、语言理解和阅读理解在内的各种序列级预测任务上均优于标准 Transformer。

## 1 引言

随着近期无监督语言预训练的成功 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]，神经自注意力模型（又称 Transformer）[13] 的能力被推向了新的高度，带动了机器学习和自然语言处理（NLP）领域的飞跃式发展。更重要的是，已有研究观察到，通过在更长时间的预训练和/或更大规模的模型上投入更多 FLOPs，预训练 Transformer 模型的性能会持续提升。然而，预训练乃至仅仅微调这些最先进的自注意力模型的成本都极其高昂，因为与 NLP 中的传统模型相比，它们需要多得多的 FLOPs 和内存资源。这在很大程度上限制了它们在更多领域的应用与推广。

鉴于这一挑战，已有越来越多的工作致力于降低自注意力（self-attention）模型的预训练和微调成本。从预训练后处理的角度来看，典型方法包括各种蒸馏、剪枝和量化，这些方法试图利用大模型中更丰富的信号，或通过学习移除次要操作，从而从充分预训练的模型中衍生出一个更轻量的模型。另一项研究方向旨在设计一种架构，它不仅具有更低的资源与性能之比（即更高的效率），而且至少在某些领域中能够具备与 Transformer 相当的可扩展性。此类方法大多数建立在 Transformer 骨干网络之上，并专注于重新设计其构建模块。代表性方案包括搜索更好的微观操作或宏观模块设计 [14, 15]，用卷积 [16] 和动态卷积 [17] 等局部操作替代完全成对注意力（full pairwise attention），以及优化现有模块的隐藏层维度组合 [18]。

在上述广泛的思路中，一个共同的策略是识别冗余的操作或表征，并用更高效的操作或表征进行替换。受此思路启发，在本文中，我们将聚焦于因在 Transformer 所有层中始终维持全长隐藏表征序列而导致的潜在冗余。直观上，对于文本分类和排序等许多序列级 NLP 任务，最常见的用例是从整个序列中提取单个向量，这并不一定需要保留细化到词元（token）粒度的所有信息。因此，对于此类任务，隐藏状态的全长序列可能包含显著的冗余。这类似于图像识别中的情况，即随着神经网络的加深，卷积神经网络会逐渐降低特征图的空间分辨率/尺寸。此外，语言学先验也支持将相邻词元（单词）逐步合并为更大的语义单元（短语），这自然会导致更短的表征序列。

具体而言，我们提出在自注意力模型中逐步降低隐藏表示的序列分辨率（即长度）。序列长度的缩减能够立竿见影地显著节省 FLOPs 和内存。更重要的是，节省下来的计算资源可以直接重新投入到构建更深（或更宽）的模型中，从而在不增加额外计算负担的情况下提升模型容量。此外，为了解决掩码语言建模（MLM）[2] 等常见预训练目标需要每个标记（token）单独表示的挑战，我们设计了一种简单的策略，从缩减长度的隐藏状态中解码出全长序列的深度表示。因此，所提出的模型无需修改预训练目标即可直接进行训练，并且适用于需要标记级表示的下游任务。

实验表明，在相当甚至更少的 FLOPs 下，通过用序列分辨率换取深度，我们提出的模型在包括文本分类、语言理解和阅读理解在内的多种序列级预测任务上，均取得了优于标准 Transformer 的性能。

## 2 方法

## 2.1 背景

Transformer 架构 Transformer 架构 [13] 是一种高度模块化的神经网络，其中每个 Transformer 层由两个子模块组成，即多头自注意力（S-Attn）和逐位置前馈网络（P-FFN）。两个子模块均由残差连接和层归一化（Layer Normalization）包裹。示意性地，给定长度为 T 的隐藏状态序列 $\mathbf { h } = [ h _ { 1 } , \ldots , h _ { T } ]$ ，单个 Transformer 层的计算可以表示为

$$
\mathbf {h} \leftarrow \text { LayerNorm } (\mathbf {h} + \text { S - Attn } (Q = \mathbf {h}, \text { KV } = \mathbf {h})),\tag{1}
$$

$$
h _ {i} \leftarrow \operatorname{LayerNorm} (h _ {i} + \mathrm{P-FFN} (h _ {i})), \quad \forall i = 1, \dots , T.\tag{2}
$$

预训练目标 最常用的预训练目标是 BERT [2] 提出的掩码语言建模（MLM）。对于从大型无标签集合 D 中采样的长度为 T 的自然语言序列 x，MLM 目标首先通过将 x 中 15% 的标记随机替换为特殊标记 [mask] 来构建一个受损序列 xˆ，然后训练 Transformer 模型 [2] 基于 xˆ 重构原始的 x，即：

$$
\max _ {\boldsymbol {\theta}} \mathcal {J} _ {\mathrm{MLM}} (\boldsymbol {\theta}) = \mathbb {E} _ {\mathbf {x} \sim \mathcal {D}} \mathbb {E} _ {\mathcal {I}} \sum_ {i \in \mathcal {I}} \log P _ {\boldsymbol {\theta}} (x _ {i} \mid \hat {\mathbf {x}} _ {\mathcal {I}}) = \mathbb {E} _ {\mathbf {x} \sim \mathcal {D}} \mathbb {E} _ {\mathcal {I}} \sum_ {i \in \mathcal {I}} \log \frac {\exp (e (x _ {i}) ^ {\top} h _ {i} (\hat {\mathbf {x}} _ {\mathcal {I}}))}{\sum_ {x ^ {\prime}} \exp (e (x ^ {\prime}) ^ {\top} h _ {i} (\hat {\mathbf {x}} _ {\mathcal {I}}))},
$$

其中 $\mathcal{I}$ 为被掩码 token 的位置，$\hat { \mathbf { x } } _ { \mathcal { T } }$ 中的下标强调了其对 $\tau$ 的依赖，$e ( x )$ 表示 token $x$ 的嵌入，而 $h _ { i } ( \hat { \mathbf { x } } _ { \mathcal { T } } )$ 表示由 Transformer 模型在位置 $i$ 处生成的最后一层隐藏状态。预训练后，整个模型将在下游任务中进行微调。

为了展示我们所提出模型的通用性，我们还对另一个预训练目标 ELECTRA [5] 进行了实验。与 MLM 不同，ELECTRA 依赖于一对联合训练的生成器和判别器。具体而言，生成器的尺寸通常较小（为判别器的 1/4），并直接通过 MLM 目标进行训练，即 $\operatorname* { m a x } _ { \theta _ { G } } \mathcal { I } _ { \mathrm { M L M } } ( \theta _ { G } )$。然后，对于每个被掩码的位置，从生成器的重构分布中采样一个 token 来替换 [mask] token，从而形成一个新的序列 $\tilde{\mathbf{x}}$，即如果 $i \in \mathcal { T } , \tilde { x } _ { i } \sim { P _ { \theta _ { G } } ( x _ { i } \mid \hat { \mathbf { x } } _ { \mathcal { T } } ) }$ 否则 $\tilde { x } _ { i } = x _ { i }$。给定新序列 $\tilde{\mathbf{x}}$ 后，判别器通过二分类进行训练，以区分 $\tilde{\mathbf{x}}$ 中的每个 token 是真实的（与 $\mathbf{x}$ 相同）还是伪造的（与 $\mathbf{x}$ 不同）。预训练完成后，在微调过程中仅使用判别器，而生成器则直接被舍弃。

讨论 注意，上述介绍的两个预训练目标都需要能够为每个输入 token 生成一个隐藏状态，即 $h _ { i } ( \hat { \mathbf { x } } _ { \mathcal { T } } )$ 和 $h _ { i } ( \tilde { \mathbf { x } } )$。由于这一需求，保留完整的隐藏状态序列似乎是很自然的。然而，相比之下，许多序列级下游任务（如分类或排序）只需要整个序列的单向量摘要。从根本上说，这表明在微调过程中通常需要某种压缩来消除不必要的冗余。这一观察立即引出了以下两个问题：

• 我们能否通过将完整的隐藏状态序列压缩为更紧凑的形式，设计出一个表达能力相当但更高效的通用模型？

• 在使用压缩表示的情况下，模型如何保持为预训练生成 token 级别（token-level）表示的能力？

为了回答这两个问题，接下来我们将介绍所提出的架构。

## 2.2 提出的架构

![](images/de926e657d0832983a403240e84eff9cea82043a922f00545e3c561a08af4d40.jpg)  
Figure 1: 所提出的 Funnel-Transformer 的高层可视化。

为了继承 Transformer 架构的高容量和优化优势，所提出的模型保留了由残差连接和层归一化包装的交替 S-Attn 与 P-FFN 子模块的总体骨架。但不同之处在于，为了实现表示压缩和减少计算量，我们的模型采用了一个编码器，该编码器随着层数的加深逐渐缩减隐藏状态的序列长度。此外，对于涉及逐 token 预测的任务（如预训练），使用了一个简单的解码器从压缩的编码器输出中重建出完整的 token 级别表示序列。

编码器 如图 1 左侧所示，编码器由若干个包含连续 Transformer 层的块（block）组成。在每个块内部，隐藏状态的序列长度始终保持不变。但当从低级块转到高级块时，通过沿序列维度执行某种类型的池化来缩减隐藏序列的长度，即：

$$
\mathbf {h} ^ {\prime} \leftarrow \operatorname{Pooling} (\mathbf {h}),\tag{3}
$$

其中 $\mathbf{h} \in \mathbb { R } ^ { T \times D }$ 且 $\mathbf { h } ^ { \prime } \in \mathbb { R } ^ { T ^ { \prime } \times D }$，满足某个 $T ^ { \prime } < T$。重要的是，我们并不是直接将池化后的序列 $\mathbf{h}'$ 输入到新块的第一个 S-Attn 层中，而是仅使用池化序列 $\mathbf { \bar { h } } ^ { \prime }$ 来构建自注意力的查询向量（以及残差信号），而未池化的序列 $\mathbf{h}$ 则充当键向量和值向量的角色，即：

$$
\mathbf {h} \leftarrow \text { LayerNorm } \big (\mathbf {h} ^ {\prime} + \text { S - Attn } \big (Q = \mathbf {h} ^ {\prime}, \text { KV } = \mathbf {h} \big) \big).\tag{4}
$$

注意，这个特殊 S-Attn 模块的输出序列长度与池化序列 $\mathbf{h}'$ 相同。为了理解这种特殊设计的优势，将所提出的“仅查询池化（pool-query-only）”变体与同时将 $\mathbf { h } ^ { \prime }$ 用于查询和键值向量的朴素替代方案（即 ${ \displaystyle { \bf S } { \bf - } \dot { \bf A t t n } \big ( } { \bf Q } = { \bf h } ^ { \prime } , { \bf K V } = { \bf h } ^ { \prime } \big ) $）进行比较是有帮助的：

• 在朴素方法下，压缩完全由池化操作控制，而该操作在注意力模块之前就已完成。因此，相对简单的池化方法（如平均池化）将无法实现良好的压缩效果。

• 在仅池化查询（pool-query-only）变体下，压缩不仅取决于池化的执行方式，还取决于自注意力机制（self-attention）如何通过对未池化序列进行加权求和来形成每个池化向量。实际上，这里的特定注意力机制可以被看作一种线性压缩，将 T 个基底组合成数量更少的 $T ^ { \prime } ^ { \mathrm { ~  ~ } }$ 个“压缩基底”。因此，在产生极小计算开销的情况下，该变体使压缩操作更具表现力。

采用这种特殊的仅池化查询设计后，我们发现应用于序列每个滑动窗口的最简单的步长平均池化（strided mean pooling）在实践中效果非常好。为简便起见，我们在本文中仅使用步长为 2 和窗口大小为 2 进行实验。因此，池化操作将把序列长度减半，并且每个池化后的隐藏状态对应于 2 个未池化隐藏向量构成的窗口。从直觉上看，这种类型的池化大致遵循了语言学先验，即相邻的 token 可以逐渐合并（或压缩）成更大的语义成分。一旦在池化和仅池化查询注意力之后序列长度减半，编码器的其余计算就直接遵循公式 (2) 和 (1) 中的标准更新。

最后，作为一个额外的实现细节，回顾一下语言预训练中的一个特殊设计是在原始输入序列的开头添加一个特殊 token [cls]，并将与 [cls] 对应的最后一层隐藏状态 $( \mathrm { i . e . , } h _ { 1 } )$ 作为序列的表示。为了防止池化破坏这一特殊结构，我们首先将 [cls] 隐藏状态与其他隐藏状态分离开来，仅对其余隐藏状态应用池化。有关某些实际实现问题及高效解决方案，读者可参阅附录 A.1。

Decoder（解码器） 为了从长度缩减的编码器输出中恢复完整的隐藏状态序列，一个自然的想法是进行某种形式的上采样（up-sampling）。例如，在图像生成或超分辨率中，反卷积（转置卷积）或带双线性插值的无参数缩放通常用于提高特征图的空间分辨率。因此，我们可以简单地将这些来自 2D 处理的思想应用到我们的 1D 场景中，并对编码器输出进行适当的上采样。

然而，与图像领域中执行多次小倍率上采样（例如每次将序列长度增加 $2 \mathbf { x }$）不同，我们在此选择采用单次大倍率上采样，如图 Fig. 1 右侧部分所示。具体而言，给定来自包含 M 个块的编码器的输出序列 $\mathbf { \bar { h } } ^ { \mathbf { M } }$（长度为 $T _ { M } = T / 2 ^ { M - 1 }$），我们通过将每个隐藏向量重复 $2 ^ { M - 1 }$ 次，直接将其上采样为全长序列 $\mathbf { \bar { h } } ^ { \mathsf { u p } } = \left\lceil h _ { 1 } ^ { \mathsf { u p } } , \cdot \cdot \cdot , h _ { T } ^ { \mathsf { u p } } \right\rceil$：

$$
\forall i = 1, \dots , T, h _ {i} ^ {\mathrm{up}} = h _ {i / / 2 ^ {M - 1}} ^ {M},\tag{5}
$$

其中 $\cdot / / \cdot$ 表示向下取整除法。然而，需要注意的是，$\mathbf { h } ^ { \mathrm { u p } }$ 中每连续 $2 ^ { M - 1 }$ 个向量都是完全相同的，因此不包含细粒度的标记级别（token-level）信息。为此，我们进一步从编码器的第一个块中提取最后一层的隐藏状态 ${ \bf h } ^ { 1 }$，该状态仍具有完整长度 T，并且包含未压缩的标记级别信息。然后，将较低层级表示 ${ \bf h } ^ { 1 }$ 与上采样的较高层级表示 $\mathbf { h } ^ { \mathrm { u p } }$ 相加，以形成深度标记级别表示 $\mathbf { g } = \mathbf { h } ^ { 1 } + \mathbf { h } ^ { \mathsf { u p } }$。实际上，这形成了一个残差/跳跃连接（residual/skip connection），既能够提供详细的标记信息，又可能使优化变得更容易。此外，我们在 $\mathbf { g }$ 之上堆叠了若干额外的 Transformer 层，以实现低层级和高层级特征之间更好的深度融合。在本研究中，我们在解码器中始终使用 2 个 Transformer 层。

需要重点强调的是，仅当任务需要标记级别预测（例如在标准预训练或序列标注中）时，才会使用解码器。对于仅需要序列的单一向量表示的任务（如分类），解码器在预训练后会被丢弃，并且仅对编码器进行微调。最后，为了强调编码器的过滤/压缩特性及其形状，我们将所提出的模型命名为 Funnel-Transformer (F-TFM)。

## 2.3 复杂度与容量分析

在明确了架构设计之后，我们现在分析序列压缩如何影响所提出模型的复杂度和容量，特别是与标准 Transformer 相比。首先，对于一个具有隐藏大小为 D 的 S-Attn 和 P-FFN 模块的 Transformer 层，处理长度为 T 的序列的复杂度为 $O ( T ^ { 2 } D + T D ^ { 2 } )$。<sup>2</sup> 因此，在编码器中每次将序列长度减半，我们都能获得超线性（超过一半）的复杂度下降。在实践中，由于 $O ( T D ^ { 2 } )$ 项具有较大的常数系数，通常更常观察到近似线性的加速。当序列长度相对较长时（例如在预训练中），超线性效应更为明显。

因此，在给定相同 FLOPs 的情况下，我们至少可以用第 1 个块中的 1 个全长层换取第 m 个块中的 $2 ^ { m - 1 }$ 个层，这提供了一种经济的网络深度增加方式。

另一方面，压缩长度层的容量显然以普通全长层的容量为其上限。在大多数压缩是有损的情况下，减少序列长度将不可避免地导致容量下降。好消息是，单层的容量下降可以通过将节省下来的 FLOPs 重新投资于堆叠更多更经济的缩短长度层或增加模型宽度来得到充分补偿。

举一个具体的例子，对于具有 $\mathbf { B E R T _ { B a s e } }$ 大小的 Transformer，即 12 层且隐藏大小为 768（L12H768），我们可以构建一个包含 3 个块的 Funnel-Transformer，其中每个块有 6 层且隐藏大小为 768（B6-6-6H768）。尽管总共有 18 层，但在针对分类任务进行微调时，B6-6-6H768 架构的 FLOPs 最多仅相当于 $6 + 6 / 2 + 6 / 4 = 1 0 . 5$ 个全长层，明显少于 L12H768。更重要的是，正如我们在实验中所展示的，B6-6-6H768 显著优于 L12H768。尽管这种深度与长度的权衡很直观，但在给定这种权衡的情况下如何构建最优的块布局仍然是一个开放性挑战。在本文中，我们仅考虑相对规则的布局，并将更系统的研究留待未来工作。

最后，需要注意的是，牺牲序列分辨率来换取深度或宽度会带来增加参数总量的副作用。例如，与 L12H768 相比，B6-6-6H768 拥有的 Transformer 参数量是其 1.5 倍。在实践中，更多的参数可能会增加分布式训练中的通信开销，以及内存消耗和内存访问时间。一个简单的补救措施是执行某种参数共享（如 ALBERT 中所采用的方法），以恢复相同的参数量。以 B6-6-6H768 为例，可以将第 2 和第 3 个模块中的参数每两层绑定一次，记为 B6-3x2-3x2H768，从而恢复到与 L12H768 相同的参数量。然而，参数共享可能会导致性能损失。从根本上说，这为我们带来了使用更多参数的增益（模型容量）与成本（内存和通信开销）之间的另一种权衡，而这可能高度依赖于具体设备。

## 3 相关工作

据我们所知，以往的研究中尚未有在语言预训练中通过压缩隐藏状态的序列长度来获得性能提升的。同时，我们提出的模型与同期一项针对因果语言建模的研究 [19] 中提出的自底向上模型非常相似。主要区别包括用于下采样的 pool-query-only 设计、上采样的执行方式以及我们的相对注意力参数化。另一个密切相关的构想是 Power-BERT [20]，它在微调期间学习软消除不太“重要”的词向量。因此，在微调后的推理阶段，可以通过减少序列长度来实现加速。更广泛地讲，我们的工作还与以往关于分层循环神经网络 [21] 和 Transformer 模型 [22, 23] 的研究相关。与这些方法不同的是，我们的模型不依赖任何预定义的层次结构或语义边界，并且始终利用注意力机制捕获全长输入依赖。

相比之下，我们的工作从计算机视觉领域汲取了许多灵感。带有残差连接的收缩编码器和扩张解码器框架在概念上类似于用于图像分割的 ResUNet [24]。步长池化（strided pooling）也被广泛用于构建现代图像识别网络 [25]。尽管有这些相似之处，除了数据领域和计算模块上的明显差异外，我们的编码器采用了特殊的 pool-query-only 设计来改善压缩效果，并且我们的解码器仅需要一次具有大扩张率的单次上采样。

此外，图神经网络领域的一系列研究尝试通过不同方式逐渐减少节点数量，并为监督分类获取单一的向量表示 [26, 27, 28]。尽管这些方法有可能作为替代压缩操作接入我们的模型中，但为监督图分类开发的压缩技术是否能够扩展到大规模语言预训练中，仍然是一个悬而未决的问题。

## 4 实验

在本节中，我们通过先对其进行预训练，然后在下游任务中进行微调，来实证评估所提出的 F-TFM。遵循先前的研究工作，在预训练方面，我们考虑两种常见设置：

• Base 规模：在 Wikipedia + Book Corpus 上以 256 的批大小（batch size）预训练模型 100 万步（1M steps）。这是原始 BERT [2] 所使用的设置。我们将依赖此设置在 F-TFM 与标准 Transformer 之间进行公平比较，并开展一些消融实验。

• Large 规模：在 XLNet [3] 和 ELECTRA [5] 所使用的五个数据集（Wikipedia + Book Corpus + ClueWeb + Gigaword + Common Crawl）上以 8K 的批大小预训练模型 50 万步（500K steps）。我们将把在该规模下训练的 F-TFM 与先前的最先进方法进行比较。

对于微调，我们主要关注仅需要输入序列的单一向量表示的序列级任务，因为 F-TFM 在设计时就考虑到了这一目的。具体而言，此类任务包括用于语言理解的 GLUE 基准 [29]、7 个广泛使用的文本（情感/主题）分类任务（IMDB、AD、DBpedia、Yelp-2、Yelp-5、Amazon-2、Amazon-5）[30] 以及 RACE 阅读理解数据集 [31]。此外，为了观察 F-TFM 在需要词元级（token-level）预测时的表现，我们还考虑了 SQuAD 问答任务，该任务要求模型从上下文段落中选择一个词元跨度（token span）作为答案。有关实验设置的更多详细信息，请读者参阅附录 B（Appendix B）。

最后，对于本工作中实现的所有模型，包括第 4.1 节 Base 规模比较中的 Transformer 基线，我们始终使用 Transformer-XL [32] 提出的相对位置注意力参数化（有关 Transformer-XL 的部分实现细节，请参见附录 A.2）。

## 4.1 Base 规模结果

首先，我们评估了在相似计算量（即 FLOPs）下，F-TFM 与标准 Transformer 相比的表现。为此，我们考虑了标准 Transformer 的三种常用模型大小，即 Large（L24H1024）、Base（L12H768）和小模型 Small（L6H768）。然后，针对每个 Transformer 基线，我们构建了具有不同块布局和参数的 F-TFM，同时确保 F-TFM 始终具有更少或相似的 FLOPs。基于 MLM 预训练目标，GLUE 基准和文本分类上的结果如 Table 1 所示，我们在其中还包含了相对 FLOPs 和 #Params。在此，我们可以得出几点关键观察：

<table><tr><td>模型大小</td><td>CoLA</td><td>SST-2</td><td>MRPC</td><td>STS-B</td><td>QQP</td><td>MNLI</td><td>QNLI</td><td>RTE</td><td>GLUE-AVG</td></tr><tr><td>L24H1024</td><td>63.2</td><td>94.8</td><td>91.8/88.5</td><td>91.1</td><td>88.7/91.7</td><td>88.7</td><td>94.0</td><td>80.5</td><td>86.6</td></tr><tr><td>B10-10-10</td><td>64.8</td><td>95.0</td><td>92.5/89.5</td><td>90.7</td><td>88.6/91.5</td><td>88.9</td><td>94.0</td><td>81.5</td><td>87.0</td></tr><tr><td>B8-8-8</td><td>63.5</td><td>94.7</td><td>92.2/89.0</td><td>90.7</td><td>88.9/91.7</td><td>88.8</td><td>93.6</td><td>81.2</td><td>86.7</td></tr><tr><td>L12H768</td><td>60.5</td><td>93.0</td><td>92.2/89.0</td><td>89.4</td><td>88.1/91.2</td><td>86.0</td><td>92.2</td><td>73.6</td><td>84.4</td></tr><tr><td>B6-6-6</td><td>62.5</td><td>94.0</td><td>92.2/89.0</td><td>89.5</td><td>88.4/91.4</td><td>87.0</td><td>92.7</td><td>76.5</td><td>85.3</td></tr><tr><td>B6-3x2-3x2</td><td>60.5</td><td>93.6</td><td>92.4/89.2</td><td>89.4</td><td>88.2/91.3</td><td>86.4</td><td>92.5</td><td>75.0</td><td>84.7</td></tr><tr><td>B4-4-4</td><td>59.1</td><td>92.7</td><td>91.8/88.7</td><td>89.1</td><td>88.2/91.3</td><td>85.5</td><td>92.0</td><td>73.2</td><td>83.9</td></tr><tr><td>L6H768</td><td>55.2</td><td>91.5</td><td>91.1/87.8</td><td>88.1</td><td>87.2/90.6</td><td>82.7</td><td>90.0</td><td>64.6</td><td>81.3</td></tr><tr><td>B3-4-4</td><td>59.0</td><td>92.8</td><td>91.8/88.5</td><td>88.5</td><td>87.8/90.9</td><td>84.8</td><td>91.8</td><td>73.2</td><td>83.7</td></tr><tr><td>模型大小</td><td>IMDB</td><td>AG</td><td>DBpedia</td><td>Yelp2</td><td>Yelp5</td><td>Amazon2</td><td>Amazon5</td><td>FLOPs</td><td>#Params</td></tr><tr><td>L24H1024</td><td>4.440</td><td>4.987</td><td>0.646</td><td>1.758</td><td>28.73</td><td>2.409</td><td>32.78</td><td>1.00x</td><td>1.00x</td></tr><tr><td>B10-10-10</td><td>4.404</td><td>5.026</td><td>0.617</td><td>1.734</td><td>28.52</td><td>2.400</td><td>32.65</td><td>0.73x</td><td>1.22x</td></tr><tr><td>B8-8-8</td><td>4.552</td><td>5.079</td><td>0.664</td><td>1.713</td><td>28.84</td><td>2.438</td><td>32.87</td><td>0.58x</td><td>1.00x</td></tr><tr><td>L12H768</td><td>5.328</td><td>5.184</td><td>0.663</td><td>2.013</td><td>29.35</td><td>2.571</td><td>33.14</td><td>1.00x</td><td>1.00x</td></tr><tr><td>B6-6-6</td><td>4.908</td><td>5.079</td><td>0.654</td><td>1.939</td><td>29.03</td><td>2.518</td><td>32.91</td><td>0.88x</td><td>1.39x</td></tr><tr><td>B6-3x2-3x2</td><td>5.144</td><td>5.342</td><td>0.649</td><td>1.892</td><td>29.03</td><td>2.570</td><td>33.01</td><td>0.88x</td><td>1.00x</td></tr><tr><td>B4-4-4</td><td>5.348</td><td>5.250</td><td>0.670</td><td>1.979</td><td>29.37</td><td>2.596</td><td>33.16</td><td>0.58x</td><td>1.00x</td></tr><tr><td>L6H768</td><td>6.252</td><td>5.421</td><td>0.697</td><td>2.203</td><td>30.33</td><td>2.801</td><td>33.69</td><td>1.00x</td><td>1.00x</td></tr><tr><td>B3-4-4</td><td>5.520</td><td>5.342</td><td>0.670</td><td>2.042</td><td>29.51</td><td>2.603</td><td>33.16</td><td>1.00x</td><td>1.53x</td></tr></table>

Table 1: base 规模下的 MLM 预训练结果：上半部分为 GLUE 验证集性能（越高越好），下半部分为文本分类错误率（越低越好）。FLOPs 和 #Params 均指仅包含编码器的微调设置。包含解码器的对应数值见附录 C.2。FLOPs 是假设关于序列长度呈线性复杂度的粗略估算。#Params 为包含嵌入矩阵的精确值。

• 在相似或更少 FLOPs 的情况下，通过牺牲序列分辨率以换取更多层数，F-TFM 在除 STS-B 之外的大多数任务中均优于标准 Transformer，对于较小的模型尤为明显。

• 当我们仅压缩序列长度而不增加深度（以及 #Params）时，F-TFM 在 GLUE 数据集的某些设置下可能会承受一定的性能损失。然而，随着模型规模的增大，此类性能差距会变小甚至消失。

• 此外，我们发现部分参数共享往往会损害性能。因此，应根据实际任务和计算设备进行实际权衡。

为了进一步测试 F-TFM 的通用性，我们还额外考虑了采用 ELECTRA 进行预训练。结果总结在 Table 2 中。总体而言，我们观察到了类似的趋势，尽管在 GLUE 基准上的增益略小。这可能是由于直接复用了针对 Transformer 调优的两个关键超参数（判别器损失系数和生成器大小乘数）来训练 F-TFM，而未做任何调整。

<table><tr><td>模型尺寸</td><td>CoLA</td><td>SST-2</td><td>MRPC</td><td>STS-B</td><td>QQP</td><td>MNLI</td><td>QNLI</td><td>RTE</td><td>GLUE-AVG</td></tr><tr><td>L24H1024</td><td>66.5</td><td>94.3</td><td>92.8/90.0</td><td>91.5</td><td>89.6/92.2</td><td>89.4</td><td>94.1</td><td>84.5</td><td>87.8</td></tr><tr><td>B10-10-10</td><td>68.6</td><td>95.0</td><td>93.0/90.0</td><td>91.0</td><td>88.9/91.7</td><td>89.1</td><td>93.6</td><td>84.5</td><td>87.9</td></tr><tr><td>B8-8-8</td><td>66.6</td><td>94.8</td><td>92.6/89.7</td><td>90.7</td><td>88.8/91.7</td><td>89.0</td><td>93.6</td><td>82.1</td><td>87.3</td></tr><tr><td>L12H768</td><td>64.3</td><td>93.1</td><td>92.1/89.2</td><td>90.8</td><td>88.7/91.7</td><td>86.4</td><td>92.1</td><td>75.4</td><td>85.4</td></tr><tr><td>B6-6-6</td><td>64.3</td><td>94.2</td><td>92.8/89.7</td><td>90.1</td><td>88.7/91.6</td><td>87.4</td><td>92.5</td><td>78.3</td><td>86.0</td></tr><tr><td>B6-3x2-3x2</td><td>63.9</td><td>94.2</td><td>93.0/90.2</td><td>89.5</td><td>88.4/91.4</td><td>87.0</td><td>92.2</td><td>77.6</td><td>85.7</td></tr><tr><td>B4-4-4</td><td>62.8</td><td>93.6</td><td>92.5/89.2</td><td>89.2</td><td>88.4/91.3</td><td>86.0</td><td>91.6</td><td>74.3</td><td>84.8</td></tr><tr><td>L6H768</td><td>62.1</td><td>91.1</td><td>90.8/86.8</td><td>88.9</td><td>88.2/91.3</td><td>83.9</td><td>89.7</td><td>66.7</td><td>82.6</td></tr><tr><td>B3-4-4</td><td>59.0</td><td>93.1</td><td>90.8/87.5</td><td>88.7</td><td>88.1/91.0</td><td>85.8</td><td>91.1</td><td>72.5</td><td>83.6</td></tr><tr><td>模型尺寸</td><td>IMDB</td><td>AG</td><td>DBpedia</td><td>Yelp2</td><td>Yelp5</td><td>Amazon2</td><td>Amazon5</td><td>FLOPs</td><td>参数量</td></tr><tr><td>L24H1024</td><td>4.724</td><td>5.053</td><td>0.653</td><td>1.874</td><td>28.84</td><td>2.425</td><td>32.85</td><td>1.00x</td><td>1.00x</td></tr><tr><td>B10-10-10</td><td>4.324</td><td>5.250</td><td>0.639</td><td>1.789</td><td>28.68</td><td>2.419</td><td>32.72</td><td>0.73x</td><td>1.22x</td></tr><tr><td>B8-8-8</td><td>4.364</td><td>5.408</td><td>0.651</td><td>1.729</td><td>28.76</td><td>2.447</td><td>32.85</td><td>0.58x</td><td>1.00x</td></tr><tr><td>L12H768</td><td>5.248</td><td>5.355</td><td>0.657</td><td>1.953</td><td>29.24</td><td>2.596</td><td>33.04</td><td>1.00x</td><td>1.00x</td></tr><tr><td>B6-6-6</td><td>4.792</td><td>5.237</td><td>0.650</td><td>1.850</td><td>28.73</td><td>2.499</td><td>32.79</td><td>0.88x</td><td>1.39x</td></tr><tr><td>B6-3x2-3x2</td><td>4.924</td><td>5.342</td><td>0.671</td><td>1.913</td><td>29.00</td><td>2.523</td><td>32.85</td><td>0.88x</td><td>1.00x</td></tr><tr><td>B4-4-4</td><td>5.152</td><td>5.382</td><td>0.659</td><td>2.032</td><td>29.33</td><td>2.566</td><td>33.03</td><td>0.58x</td><td>1.00x</td></tr><tr><td>L6H768</td><td>6.220</td><td>5.395</td><td>0.674</td><td>2.287</td><td>30.16</td><td>2.759</td><td>33.57</td><td>1.00x</td><td>1.00x</td></tr><tr><td>B3-4-4</td><td>5.396</td><td>5.342</td><td>0.653</td><td>2.000</td><td>29.60</td><td>2.591</td><td>33.09</td><td>1.00x</td><td>1.53x</td></tr></table>

Table 2: base 规模下的 ELECTRA 预训练结果。

运行时间对比 虽然 FLOPs 计数能够大致反映模型速度，但它与实际运行时间仍存在差异，特别是当存在其他开销时。因此，为了完整起见，我们在附录 C.2 中展示了 F-TFM 在实际运行时间方面的加速效果。我们还在附录 C.2 中对比了 F-TFM 和 TFM 的实际内存占用。

## 4.2 大规模结果

鉴于 F-TFM 在 base 规模下的优异结果，我们接下来考虑在大规模设置下训练 F-TFM，并将其与在类似设置下预训练的以往模型进行比较。由于 ELECTRA 的性能略优于 MLM，我们在所有大规模实验中均采用 ELECTRA 目标。针对不同大小的预训练 F-TFM，我们首先在 Table 3 中比较了它们在 GLUE 基准上的微调性能。与 base 规模的结果相似，在具有更少或相当的 FLOPs 情况下，F-TFM 在大多数任务上均优于相应的基准模型，这表明 F-TFM 具有良好的可扩展性。我们还在 7 个文本分类任务上测试了这些模型。但由于篇幅限制，请读者参阅附录 C.1。

接下来，我们考虑与 GLUE 基准截然不同的 RACE 数据集。从核心来看，RACE 是一个需要复杂推理的多项选择阅读理解任务，不过它可以被形式化为对正确选项进行分类。此外，RACE 中的段落要长得多。对于 F-TFM 而言，这既是一个挑战（因为它需要细致的推理），也是一个压缩长段落的契机。如 Table 4 所示，与以往所有模型相比，F-TFM 取得了更优的性能。特别是在 base 模型组中，性能提升非常显著。这表明 F-TFM 同样擅长处理涉及长文本和推理的序列级任务。

最后，尽管 F-TFM 主要是针对仅需要序列级表示的任务而设计的，但通过额外微调解码器，也可以将 F-TFM 应用于 token 级任务。为了测试这种能力，我们在 SQuAD 数据集上微调了 F-TFM，并在 Table 5 中将其与以往模型进行了比较。虽然在 base 组中 F-TFM 大幅超越了以往模型，但在 large 模型组中，具有约 83% FLOPs 的 F-TFM（B10-10-10）仍然落后于始终保持全长 token 级表示的标准 Transformer。这表明当细致的 token 级信息至关重要时，序列压缩可能会损害性能。另一方面，与在 SQuAD1.1 上的结果相比，F-TFM 在 SQuAD2.0 上的表现相对更好，

<table><tr><td>模型</td><td>CoLA</td><td>SST-2</td><td>MRPC</td><td>STS-B</td><td>QQP</td><td>MNLI</td><td>QNLI</td><td>RTE</td><td>WNLI</td><td>AVG</td></tr><tr><td colspan="11">开发集结果（单模型）</td></tr><tr><td> $ROBERTA_{Large}$  [4]</td><td>68.0</td><td>96.4</td><td>-/90.9</td><td>92.4</td><td>-/92.2</td><td>90.2</td><td>94.7</td><td>86.6</td><td>-</td><td>88.9</td></tr><tr><td>XLNetLarge [3]</td><td>69.0</td><td>97.0</td><td>-/90.8</td><td>92.5</td><td>-/92.3</td><td>90.8</td><td>94.9</td><td>85.9</td><td>-</td><td>89.2</td></tr><tr><td> $ELECTRA_{Large}$  [5]</td><td>69.1</td><td>96.9</td><td>-/90.8</td><td>92.6</td><td>-/92.4</td><td>90.9</td><td>95.0</td><td>88.0</td><td>-</td><td>89.5</td></tr><tr><td>B10-10-10H1024</td><td>72.4</td><td>96.8</td><td>93.5/90.9</td><td>92.1</td><td>89.8/92.4</td><td>91.1/-</td><td>95.1</td><td>89.5</td><td>-</td><td>90.0</td></tr><tr><td>B8-8-8H1024</td><td>71.3</td><td>96.8</td><td>93.1/90.7</td><td>91.7</td><td>89.8/92.4</td><td>90.8/-</td><td>94.7</td><td>89.2</td><td>-</td><td>89.7</td></tr><tr><td> $ROBERTA_{Base}$  [4]</td><td>63.6</td><td>94.8</td><td>-/90.2</td><td>91.2</td><td>-/91.9</td><td>87.6/-</td><td>92.8</td><td>78.7</td><td>-</td><td>86.4</td></tr><tr><td>MPNetBase [12]</td><td>65.0</td><td>95.4</td><td>-/91.5</td><td>90.9</td><td>-/91.9</td><td>88.5/-</td><td>93.3</td><td>85.2</td><td>-</td><td>87.7</td></tr><tr><td>B6-6-6H768</td><td>70.1</td><td>96.3</td><td>93.2/90.4</td><td>91.1</td><td>89.2/92.0</td><td>89.7/-</td><td>93.7</td><td>83.4</td><td>-</td><td>88.3</td></tr><tr><td>B6-3x2-3x2H768</td><td>68.5</td><td>95.6</td><td>92.5/89.5</td><td>91.0</td><td>89.3/92.0</td><td>89.1/-</td><td>93.0</td><td>83.4</td><td>-</td><td>87.8</td></tr><tr><td>B4-4-4H768</td><td>68.2</td><td>95.0</td><td>92.8/90.2</td><td>90.3</td><td>89.0/91.8</td><td>88.6/-</td><td>92.6</td><td>79.1</td><td>-</td><td>87.0</td></tr><tr><td colspan="11">排行榜测试集结果（单任务 &amp; 单模型）</td></tr><tr><td> $ELECTRA_{Large}$  [5]</td><td>68.1</td><td>96.7</td><td>89.2/92.0</td><td>92.1/91.7</td><td>74.8/90.4</td><td>90.7/90.2</td><td>95.5</td><td>86.1</td><td>65.1</td><td>85.2</td></tr><tr><td>B10-10-10H1024</td><td>68.9</td><td>97.2</td><td>89.4/92.1</td><td>91.6/91.3</td><td>74.3/90.2</td><td>90.9/90.9</td><td>95.5</td><td>86.5</td><td>65.1</td><td>85.4</td></tr><tr><td>B8-8-8H1024</td><td>68.3</td><td>96.9</td><td>89.2/92.0</td><td>91.5/91.1</td><td>73.8/90.1</td><td>90.7/90.7</td><td>95.1</td><td>85.3</td><td>65.1</td><td>85.0</td></tr><tr><td> $ELECTRA_{Base}$  [5]</td><td>64.6</td><td>96.0</td><td>88.1/91.2</td><td>91.0/90.2</td><td>73.2/89.5</td><td>88.5/88.0</td><td>93.1</td><td>75.2</td><td>65.1</td><td>82.7</td></tr><tr><td>B6-6-6H768</td><td>68.3</td><td>96.5</td><td>89.1/91.9</td><td>90.6/89.9</td><td>73.3/89.9</td><td>89.7/89.4</td><td>94.0</td><td>80.4</td><td>65.1</td><td>84.0</td></tr><tr><td>B6-3x2-3x2H768</td><td>65.9</td><td>96.0</td><td>87.8/91.0</td><td>90.0/89.6</td><td>73.3/89.8</td><td>88.9/88.7</td><td>93.8</td><td>79.9</td><td>65.1</td><td>83.4</td></tr><tr><td colspan="11">排行榜测试集结果（多任务 &amp; 集成）</td></tr><tr><td> $ROBERTA_{Large}$  [4]</td><td>67.8</td><td>96.7</td><td>89.8/92.3</td><td>92.2/91.9</td><td>74.3/90.2</td><td>90.8/90.2</td><td>95.4</td><td>88.2</td><td>89.0</td><td>88.1</td></tr><tr><td> $ELECTRA_{Large}$  [5]</td><td>71.7</td><td>97.1</td><td>90.7/93.1</td><td>92.9/92.5</td><td>75.6/90.8</td><td>91.3/90.8</td><td>95.8</td><td>89.8</td><td>91.8</td><td>89.4</td></tr><tr><td>B10-10-10H1024</td><td>70.5</td><td>97.5</td><td>91.2/93.4</td><td>92.6/92.3</td><td>75.4/90.7</td><td>91.4/91.1</td><td>95.8</td><td>90.0</td><td>94.5</td><td>89.7</td></tr></table>

Table 3: 在大规模预训练下与以往方法在 GLUE 基准上的对比。

<table><tr><td rowspan="2">模型</td><td colspan="3">RACE</td></tr><tr><td>总分</td><td>高中</td><td>初中</td></tr><tr><td> $ROBERTA_{Large}$  [4]</td><td>83.2</td><td>81.3</td><td>86.5</td></tr><tr><td>XLNetLarge [3]</td><td>85.4</td><td>84.0</td><td>88.6</td></tr><tr><td>B10-10-10</td><td>85.7</td><td>84.4</td><td>88.8</td></tr><tr><td>B8-8-8</td><td>85.2</td><td>83.9</td><td>88.4</td></tr><tr><td> $ALBERT_{Base}$  [6]</td><td>66.0</td><td>-</td><td>-</td></tr><tr><td>MPNetBase [12]</td><td>72.0</td><td>76.3</td><td>70.3</td></tr><tr><td>B6-6-6</td><td>79.7</td><td>78.2</td><td>83.4</td></tr><tr><td>B6-3x2-3x2</td><td>78.8</td><td>77.5</td><td>82.0</td></tr><tr><td>B4-4-4</td><td>76.2</td><td>74.6</td><td>80.0</td></tr></table>

Table 4: RACE 测试集性能对比。

<table><tr><td rowspan="2">模型</td><td colspan="2">SQuAD2.0</td><td colspan="2">SQuAD1.1</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td> $ROBERTA_{Large}$  [4]</td><td>86.5</td><td>89.4</td><td>88.9</td><td>94.6</td></tr><tr><td> $ELECTRA_{Large}$  [5]</td><td>88.0</td><td>90.6</td><td>89.7</td><td>94.9</td></tr><tr><td>B10-10-10</td><td>87.6</td><td>90.4</td><td>89.0</td><td>94.7</td></tr><tr><td>B8-8-8</td><td>87.1</td><td>89.8</td><td>88.7</td><td>94.4</td></tr><tr><td> $ROBERTA_{Base}$  [4]</td><td>80.5</td><td>83.7</td><td>84.6</td><td>91.5</td></tr><tr><td> $MPNet_{Base}$  [21]</td><td>80.5</td><td>83.3</td><td>86.8</td><td>92.5</td></tr><tr><td>B6-6-6</td><td>85.1</td><td>87.7</td><td>87.4</td><td>93.3</td></tr><tr><td>B6-3x2-3x2</td><td>84.2</td><td>87.0</td><td>87.0</td><td>93.0</td></tr><tr><td>B4-4-4</td><td>82.6</td><td>85.5</td><td>85.9</td><td>92.2</td></tr></table>

Table 5: SQuAD 验证集性能对比。  
这额外要求模型在序列级别对问题是否可回答做出预测。这再次表明了 F-TFM 在序列级任务中的通用有效性。

## 4.3 消融实验

<table><tr><td>ID</td><td>布局</td><td>(FLOPs / 参数量)</td><td>池化操作</td><td>仅池化查询</td><td>分离 [cls]</td><td>相对注意力</td><td>GLUE-AVG</td></tr><tr><td>(1)</td><td>B6-6-6</td><td>(1.00x / 1.00x)</td><td>Mean</td><td>√</td><td>√</td><td>√</td><td>83.5</td></tr><tr><td>(2)</td><td></td><td></td><td>Mean</td><td>√</td><td></td><td>√</td><td>82.9</td></tr><tr><td>(3)</td><td></td><td></td><td>Mean</td><td></td><td>√</td><td>√</td><td>83.0</td></tr><tr><td>(4)</td><td></td><td></td><td>Mean</td><td>√</td><td>√</td><td></td><td>81.4</td></tr><tr><td>(5)</td><td></td><td></td><td>Max</td><td>√</td><td>√</td><td>√</td><td>83.4</td></tr><tr><td>(6)</td><td></td><td></td><td>Top-Attn</td><td>√</td><td>√</td><td>√</td><td>75.8</td></tr><tr><td>(7)</td><td>B8-8</td><td>(1.14x / 0.91x)</td><td>Mean</td><td>√</td><td>√</td><td>√</td><td>83.4</td></tr><tr><td>(8)</td><td>B5-5-5-5</td><td>(0.89x / 1.08x)</td><td>Mean</td><td>√</td><td>√</td><td>√</td><td>82.9</td></tr></table>

Table 6: 具有不同设计的 F-TFM 的消融研究。

最后，基于 GLUE 基准，我们针对 F-TFM 中各项设计的重要性进行了一系列消融研究，包括块布局设计、池化操作类型、仅池化查询技术、保留独立的 [cls] 向量以及使用 Transformer-XL 参数化。

• 池化操作：包括我们最终在 F-TFM 中采用的均值池化在内，我们实际测试了两种类型的池化操作。

(1) 第一种类型即如第 2 节中所述的步幅均值/最大值池化。

(2) 第二种类型旨在选择一组“枢纽（hub）”状态子集，即在前一个 S-Attn 层中受到最多注意力的那些隐层向量，因此它们可能携带了有关序列的最关键信息。具体而言，给定前一个 S-Attn 层的注意力图，我们沿头数和查询长度维度对分数进行求和约简，得到每个位置的一个分数。然后，我们只需选择前 50% 的状态即可达到相同的压缩率。请注意，这种类型的池化操作本质上与 Power-BERT [20] 中的重要状态选择过程相同。

• 仅池化查询设计

• 在池化操作中分离 [cls]

• 块布局设计：在我们的实验中，所有模型实际上都采用了 3 块设计。在此，我们将 3 块设计与 2 块和 4 块设计进行了比较。

• Transformer-XL [32] 中提出的相对注意力参数化。我们将这种参数化与 BERT [2] 中使用的可学习绝对位置嵌入进行了比较。

消融实验结果列于 Table 6 中。为了节省计算资源，Table 6 中模型隐层状态的大小设置为 512。从消融实验结果中，我们可以得出以下观察结论：

• 比较不同的池化操作（(1)、(5) 和 (6)），我们发现均值池化和最大池化操作的性能相似。但它们都显著优于利用注意力分数（Top-Attn 池化）来选择“中心”（hub）状态的方法。

• 分别对比 (1) 与 (2) 和 (3)，我们发现两个特殊设计（即“仅池化查询（pool-query-only）”和维护一个单独的非池化 [cls]）都能为所提出的模型带来明显的提升。

• 对比 (1) 和 (4)，我们发现相对位置参数化对所提出的 F-TFM 的性能至关重要。我们推测池化操作可能会破坏绝对位置编码所携带的位置信息，而绝对位置编码仅在输入嵌入层被注入到模型中。因此，更高层级的块可能没有足够的位置信息来学习到足够好的注意力模式。相比之下，在相对位置注意力方案下，位置信息被注入到了每一层。因此，为了使基于绝对位置嵌入的 F-TFM 取得良好效果，可以将绝对位置嵌入注入到每个注意力层中。事实上，Transformer 在计算机视觉目标检测问题中的一项同期应用表明，将位置嵌入注入到每一层是非常重要的 [33]。

• 最后，我们研究了本框架中块布局（block layout）设计的影响。以 B6-6-6 作为 3 块基准，我们考虑了另外两种具有相似 FLOPs 和参数量的布局设计。具体而言，针对 2 块设计我们考虑了 B8-8，针对 4 块设计考虑了 B5-5-5-5。对比 (1)、(7) 和 (8) 中的结果，我们发现 3 块（B6-6-6）设计取得了最佳性能，显著优于 4 块设计，且略优于 2 块设计。然而，如果进一步将 FLOPs/#Params 纳入考量，3 块设计的优势则更为明显。因此，在正文中我们始终采用 3 块设计。

## 5 结论与讨论

在本工作中，在预训练-微调范式下，我们研究了语言处理中一个在很大程度上被忽视的复杂度维度。通过提出的 Funnel-Transformer，我们展示了如何以简单的形式压缩序列分辨率以节省计算量，以及如何将节省下来的 FLOPs 重新投入到提升模型容量从而提升性能上。未来研究的开放性挑战包括探索更好的方法来改进压缩方案、优化块布局设计以及重新分配所节省的 FLOPs。此外，将 Funnel-Transformer 与知识蒸馏（knowledge distillation）和量化（quantization）等模型压缩技术相结合，将是提升其实际应用价值的一个重要方向。

## 参考文献

[1] Matthew E Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. arXiv preprint arXiv:1802.05365, 2018.

[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

[3] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. Xlnet: Generalized autoregressive pretraining for language understanding. In Advances in neural information processing systems, pages 5754–5764, 2019.

[4] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.

[5] Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christopher D Manning. Electra: Pre-training text encoders as discriminators rather than generators. arXiv preprint arXiv:2003.10555, 2020.

[6] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942, 2019.

[7] Lingpeng Kong, Cyprien de Masson d’Autume, Wang Ling, Lei Yu, Zihang Dai, and Dani Yogatama. A mutual information maximization perspective of language representation learning. arXiv preprint arXiv:1910.08350, 2019.

[8] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683, 2019.

[9] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461, 2019.

[10] Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. Mass: Masked sequence to sequence pre-training for language generation. arXiv preprint arXiv:1905.02450, 2019.

[11] Xiaodong Liu, Pengcheng He, Weizhu Chen, and Jianfeng Gao. Multi-task deep neural networks for natural language understanding. arXiv preprint arXiv:1901.11504, 2019.

[12] Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. Mpnet: Masked and permuted pre-training for language understanding. arXiv preprint arXiv:2004.09297, 2020.

[13] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008, 2017.

[14] David R So, Chen Liang, and Quoc V Le. The evolved transformer. arXiv preprint arXiv:1901.11117, 2019.

[15] Daoyuan Chen, Yaliang Li, Minghui Qiu, Zhen Wang, Bofang Li, Bolin Ding, Hongbo Deng, Jun Huang, Wei Lin, and Jingren Zhou. Adabert: Task-adaptive bert compression with differentiable neural architecture search. arXiv preprint arXiv:2001.04246, 2020.

[16] Zhanghao Wu, Zhijian Liu, Ji Lin, Yujun Lin, and Song Han. Lite transformer with long-short range attention. arXiv preprint arXiv:2004.11886, 2020.

[17] Felix Wu, Angela Fan, Alexei Baevski, Yann N Dauphin, and Michael Auli. Pay less attention with lightweight and dynamic convolutions. arXiv preprint arXiv:1901.10430, 2019.

[18] Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, and Denny Zhou. Mobilebert: a compact task-agnostic bert for resource-limited devices. arXiv preprint arXiv:2004.02984, 2020.

[19] Sandeep Subramanian, Ronan Collobert, Marc’Aurelio Ranzato, and Y-Lan Boureau. Multiscale transformer language models. arXiv preprint arXiv:2005.00581, 2020.

[20] Saurabh Goyal, Anamitra Roy Choudhary, Venkatesan Chakaravarthy, Saurabh ManishRaje, Yogish Sabharwal, and Ashish Verma. Power-bert: Accelerating bert inference for classification tasks. arXiv preprint arXiv:2001.08950, 2020.

[21] Rui Lin, Shujie Liu, Muyun Yang, Mu Li, Ming Zhou, and Sheng Li. Hierarchical recurrent neural network for document modeling. In Proceedings ofthe 2015 Conference on Empirical Methods in Natural Language Processing, pages 899–907, 2015.

[22] Xingxing Zhang, Furu Wei, and Ming Zhou. Hibert: Document level pre-training of hierarchical bidirectional transformers for document summarization. arXiv preprint arXiv:1905.06566, 2019.

[23] Vikas K Garg, Inderjit S Dhillon, and Hsiang-Fu Yu. Multiresolution transformer networks: Recurrence is not essential for modeling hierarchical structure. arXiv preprint arXiv:1908.10408, 2019.

[24] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015.

[25] Dominik Scherer, Andreas Müller, and Sven Behnke. Evaluation of pooling operations in convolutional architectures for object recognition. In International conference on artificial neural networks, pages 92–101. Springer, 2010.

[26] Zhitao Ying, Jiaxuan You, Christopher Morris, Xiang Ren, Will Hamilton, and Jure Leskovec. Hierarchical graph representation learning with differentiable pooling. In Advances in neural information processing systems, pages 4800–4810, 2018.

[27] Hongyang Gao and Shuiwang Ji. Graph u-nets. arXiv preprint arXiv:1905.05178, 2019.

[28] Junhyun Lee, Inyeop Lee, and Jaewoo Kang. Self-attention graph pooling. arXiv preprint arXiv:1904.08082, 2019.

[29] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.

[30] Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. In Advances in neural information processing systems, pages 649–657, 2015.

[31] Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. Race: Large-scale reading comprehension dataset from examinations. arXiv preprint arXiv:1704.04683, 2017.

[32] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860, 2019.

[33] Nicolas Carion, F. Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander M Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. ArXiv, abs/2005.12872, 2020.

[34] Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, and Quoc V Le. Unsupervised data augmentation for consistency training. 2019.

## A 实现优化

## A.1 分离 [cls] 技巧的序列截断

如第 2.2 节所述，为了避免破坏预训练中常用的 [cls] 结构，我们不对 [cls] 应用池化操作，并保持对应于 [cls] 的隐藏状态不变。尽管在概念上很简单，但朴素的实现可能会由于该操作导致的“不规则”序列长度而使计算速度降低 15%。具体而言，假设输入样本的序列长度是 2 的幂次，即 $2 ^ { p }$，在预训练阶段通常为 512。在保持 [cls] 不变的情况下进行一次池化操作后，池化后序列的长度变为 $2 ^ { p - 1 } + 1$，不再是 2 的幂次。因此，这可能导致内存不对齐以及加速器中并行计算能力的浪费，从而导致大幅度的速度损失。

为了解决这个问题，我们采用了一种简单的策略，即在池化后截断最后一个标记（token）。形式上，将池化后的隐藏状态表示为 $\mathbf { h } = \left\{ h _ { [ \mathsf { c 1 s } ] } \ , h _ { 1 } , \cdot \cdot \cdot \ , h _ { 2 ^ { p - 1 } } \right\}$，截断可以表示为

$$
\hat {\mathbf {h}} = \operatorname{truncate} (\mathbf {h}) = \left[ h _ {[ \mathrm{cls} ]}, h _ {1}, \dots , h _ {2 ^ {p - 1} - 1} \right]\tag{6}
$$

通过这个简单技巧，我们可以始终保持序列长度为 2 的幂次，从而避免了因维护独立的 [cls] 隐藏状态而导致的速度减慢。

## A.2 相对位置注意力实现

在这项工作中，我们使用了 Transformer-XL [32] 中提出的相对位置注意力参数化方法。为了便于进一步讨论，我们首先回顾该参数化方法的细节。以单头注意力为例。设 T、D 分别为序列长度和隐藏维度。那么，位置 i 和 j 之间在 Softmax 之前的注意力分数 $A _ { i j }$ 由两项组成：

$$
A _ {i j} = \underbrace {(W _ {Q} h _ {i} + v) ^ {\top} (W _ {K} h _ {j})} _ {\text {content term}} + \underbrace {(W _ {Q} h _ {i} + u) ^ {\top} (W _ {R} r _ {i - j})} _ {\text {position term}}.\tag{7}
$$

其中 $v , u \in \mathbb { R } ^ { D }$ 是两个可训练的偏置向量，$W _ { Q } , W _ { K } , W _ { R } \in \mathbb { R } ^ { D \times D }$ 是三个可训练的投影矩阵，而 $r _ { i - j } \in \mathbb { R } ^ { D }$ 是表示两个位置之间相对距离 $i - j$ 的正弦位置编码。

为了计算整个注意力分数矩阵 A，内容项（content term）可以通过两次头投影（head projections）以及复杂度为 $O ( T D ^ { 2 } + T ^ { 2 } D )$ 的外积轻松获得：

$$
\mathbf {A} ^ {\text { content }} = (\mathbf {H} W _ {Q} + v) (\mathbf {H} W _ {K}) ^ {\top},
$$

其中 $\mathbf { H } = [ h _ { 1 } , \cdots , h _ { T } ] \in \mathbb { R } ^ { T \times D }$ 将所有隐藏状态收集为一个矩阵。然而，由于每个 $A _ { i j } ^ { \mathsf { p o s i t i o n } }$ 都对应不同的 $r _ { i - j }$，我们无法以相同的方式计算位置项。因此，一种朴素的解决方案是将 $T ^ { 2 }$ 对位置编码堆叠为一个张量 $\hat { \mathbf { R } } \in \mathbb { R } ^ { T \times T \times D }$（其中 $\hat { \bf R } _ { i j } = r _ { i - j }$），然后执行以下张量积：

$$
\mathbf {A} ^ {\text { position }} = \operatorname{einsum} \left(" \mathrm{id}, \mathrm{ijd} - > \mathrm{ij}", \mathbf {H} W _ {Q} + u, \hat {\mathbf {R}} W _ {R}\right).
$$

请注意，头投影 $\mathbf { R } W _ { K }$ 此时的复杂度为 $O ( T ^ { 2 } D ^ { 2 } )$，内存占用为 $O ( T ^ { 2 } D )$，在所有计算中占据主导地位。

## A.2.1 标准解法：Gather / Shift

为了解决上述计算负担，一种常用的技术是改为收集一个矩阵 $\mathbf { R } \in$ $\mathbb { R } ^ { 2 T - 1 \times D }$，其中

$$
\mathbf {R} = [ r _ {T - 1}, \ldots , r _ {0}, \dots , r _ {1 - T} ]
$$

它包含了从最大可能距离值 $T - 1$ 到最小可能距离值 $1 - T$ 排列的所有可能位置编码。请注意，完整的 $\bar { \hat { \mathbf { R } } }$ 可以通过形状为 $[ T \times T ]$ 的索引矩阵 I 从 R 中收集（gather）特定元素来构成，即：

$$
\hat {\mathbf {R}} = \operatorname{gather} (\mathbf {R}, \mathbf {I}), \quad I _ {i j} = T + i - j.
$$

在数学上，这等价于使用置换张量（permutation tensor）$\mathbf { P } \in \mathbb { R } ^ { T \times T \times 2 T - 1 }$ 与 R 相乘，即 $\hat { \mathbf { R } } = \mathbf { P } \mathbf { R }$，其中 $\mathbf { P } _ { i j } \in \mathbb { R } ^ { 2 T - 1 }$ 是用于选择/收集 R 中单个位置的独热（one-hot）向量。由于注意力分数的计算仅涉及线性运算，我们可以将位置项的计算重排如下

$$
\begin{array}{l} \mathbf {A} ^ {\text {position}} = \operatorname{einsum} \bigl (" \mathrm{id}, \mathrm{ijd-} > \mathrm{ij}", \mathbf {H} W _ {Q} + u, (\mathbf {P R}) W _ {R} \bigr) \\ \qquad = \operatorname{einsum} \Bigl (" \mathrm{ijk}, \mathrm{jk-} > \mathrm{ij}", \mathbf {P}, \bigl [ (\mathbf {H} W _ {Q} + v) (\mathbf {R} W _ {R}) ^ {\top} \bigr ] \Bigr) \\ \qquad = \operatorname{gather} \Bigl ((\mathbf {H} W _ {Q} + v) (\mathbf {R} W _ {R}) ^ {\top}, \mathbf {I} \Bigr) \end{array}
$$

注意，假设 gather（收集）$T ^ { 2 }$ 个元素的复杂度仅为 $O ( T ^ { 2 } )$（这在 CPU/GPU 上成立），该技巧将计算复杂度降低回 $\dot { O } ( 2 T D ^ { 2 } + 2 T ^ { 2 } D )$。在实践中，gather 操作可以通过更巧妙的 reshape（重塑）操作来实现，其开销甚至更低。

## A.2.2 针对 TPU 的优化：因子化相对位置注意力

然而，在 $\mathrm { T P U s }$ 上，收集 $T ^ { 2 }$ 个元素仅具有 $O ( T ^ { 2 } )$ 复杂度的假设并不成立。相反，我们发现在 TPU 上此类 gather 操作显著更慢。因此，我们在此考虑另一种在 TPU 上明显更快的实现方式。

首先，我们将位置项重写如下：

$$
\begin{array}{r l} A _ {i j} ^ {\text {position}} & = (W _ {Q} h _ {i} + u) ^ {\top} (W _ {R} r _ {i - j}) \\ & = \Big [ \underbrace {W _ {R} ^ {\top} (W _ {Q} h _ {i} + u)} _ {q _ {i}} \Big ] ^ {\top} r _ {i - j} \\ & = q _ {i} ^ {\top} r _ {i - j}. \end{array}\tag{8}
$$

为了便于推导，我们引入了符号 $q _ { i }$。接着，回顾 $r _ { i - j }$ 是由正弦和余弦分量组成的正弦曲线编码（sinusoidal encoding）$\begin{array} { r } { r _ { i - j } = { \tt c a t } ( \sin _ { i - j } , \cos _ { i - j } ) } \end{array}$，其中

$$
\begin{array}{l} \sin_ {t} = \Big [ \sin \Big (t / 1 0 0 0 0 ^ {2 / D} \Big), \sin \Big (t / 1 0 0 0 0 ^ {4 / D} \Big), \dots , \sin \Big (t / 1 0 0 0 0 ^ {D / D} \Big) \Big ] \in \mathbb {R} ^ {D / 2}, \\ \cos_ {t} = \Big [ \cos \Big (t / 1 0 0 0 0 ^ {2 / D} \Big), \cos \Big (t / 1 0 0 0 0 ^ {4 / D} \Big), \dots , \cos \Big (t / 1 0 0 0 0 ^ {D / D} \Big) \Big ] \in \mathbb {R} ^ {D / 2}. \end{array}
$$

因此，我们类似地将上面定义的 $q _ { i }$ 分为两部分，即

$$
q _ {i} = \mathsf {c a t} (q _ {i} ^ {\sin}, q _ {i} ^ {\cos}).
$$

根据这些定义，我们可以进一步将式 (8) 分解为两项：

$$
A _ {i j} ^ {\text { position }} = q _ {i} ^ {\top} r _ {i - j} = q _ {i} ^ {\sin^ {\top}} \sin_ {i - j} + q _ {i} ^ {\cos^ {\top}} \cos_ {i - j}.
$$

现在，利用三角恒等式 $\sin ( a - b ) = \sin ( a ) \cos ( b ) - \cos ( a ) \sin ( b )$ 和 $\cos ( a - b ) = \cos ( a ) \cos ( b ) + \sin ( a ) \sin ( b )$，这两项可以分别重新表示为

$$
\begin{array}{r l} & {q _ {i} ^ {\sin^ {\top}} \sin_ {i - j} = q _ {i} ^ {\sin^ {\top}} [ \sin_ {i} \odot \cos_ {j} - \cos_ {i} \odot \sin_ {j} ]} \\ & {\quad = q _ {i} ^ {\sin^ {\top}} (\sin_ {i} \odot \cos_ {j}) - q _ {i} ^ {\sin^ {\top}} (\cos_ {i} \odot \sin_ {j})} \\ & {\quad = \left[ q _ {i} ^ {\sin} \odot \sin_ {i} \right] ^ {\top} \cos_ {j} + \left[ q _ {i} ^ {\sin} \odot (- \cos_ {i}) \right] ^ {\top} \sin_ {j}} \end{array}
$$

以及

$$
\begin{array}{r l} & {q _ {i} ^ {\cos^ {\top}} \cos_ {i - j} = q _ {i} ^ {\cos^ {\top}} [ \cos_ {i} \odot \cos_ {j} + \sin_ {i} \odot \sin_ {j} ]} \\ & {\quad = q _ {i} ^ {\cos^ {\top}} (\cos_ {i} \odot \cos_ {j}) + q _ {i} ^ {\cos^ {\top}} (\sin_ {i} \odot \sin_ {j})} \\ & {\quad = [ q _ {i} ^ {\cos} \odot \cos_ {i} ] ^ {\top} \cos_ {j} + [ q _ {i} ^ {\cos} \odot \sin_ {i} ] ^ {\top} \sin_ {j}} \end{array}
$$

因此，将这两部分结合在一起，可得

$$
\begin{array}{r l} & q _ {i} ^ {\top} r _ {i - j} = q _ {i} ^ {\sin^ {\top}} \sin_ {i - j} + q _ {i} ^ {\cos^ {\top}} \cos_ {i - j} \\ & \quad = \left[ q _ {i} ^ {\sin} \odot \sin_ {i} \right] ^ {\top} \cos_ {j} + \left[ q _ {i} ^ {\sin} \odot (- \cos_ {i}) \right] ^ {\top} \sin_ {j} + [ q _ {i} ^ {\cos} \odot \cos_ {i} ] ^ {\top} \cos_ {j} + [ q _ {i} ^ {\cos} \odot \sin_ {i} ] ^ {\top} \sin_ {j} \\ & \quad = \left\{\left[ q _ {i} ^ {\sin} \odot \sin_ {i} \right] ^ {\top} \cos_ {j} + [ q _ {i} ^ {\cos} \odot \cos_ {i} ] ^ {\top} \cos_ {j} \right\} + \left\{\left[ q _ {i} ^ {\sin} \odot (- \cos_ {i}) \right] ^ {\top} \sin_ {j} + [ q _ {i} ^ {\cos} \odot \sin_ {i} ] ^ {\top} \sin_ {j} \right\} \\ & \quad = \left[ \underbrace {\mathsf {c a t} (q _ {i} ^ {\sin} , q _ {i} ^ {\cos})} _ {= q _ {i}} \odot \underbrace {\mathsf {c a t} (\sin_ {i} , \cos_ {i})} _ {:= \phi_ {i}} \right] ^ {\top} \underbrace {\mathsf {c a t} (\cos_ {j} , \cos_ {j})} _ {:= \psi_ {j}} \\ & \quad + \left[ \underbrace {\mathsf {c a t} (q _ {i} ^ {\sin} , q _ {i} ^ {\cos})} _ {= q _ {i}} \odot \underbrace {\mathsf {c a t} (- \cos_ {i} , \sin_ {i})} _ {:= \pi_ {i}} \right] ^ {\top} \underbrace {\mathsf {c a t} (\sin_ {j} , \sin_ {j})} _ {:= \omega_ {j}} \\ & \quad = [ q _ {i} \odot \phi_ {i} ] ^ {\top} \psi_ {j} + [ q _ {i} \odot \pi_ {i} ] ^ {\top} \omega_ {j}, \end{array}
$$

其中，上述 $\phi _ { i } , \psi _ { j } , \pi _ { i } , \omega _ { j }$ 仅是通过以不同方式拼接对应 $i$ 和 $j$ 的余弦与正弦向量而形成的 4 个位置编码。注意到，最后一行的每一项都具有可通过外积计算的因式分解形式，就像标准内容项一样。因此，通过将所有位置（即 $i = 1 , \dots , T$ 和 $j = 1 , \dots , T$）的 $\phi _ { i } , \psi _ { j } , \pi _ { i } , \omega _ { j }$ 分别堆叠到对应的 $\Phi, \Psi, \Pi, \pmb { \Omega } \in \mathbb { R } ^ { T \times D }$ 中，完整的位置项可以表示为一种简单的形式

$$
\mathbf {A} ^ {\text { position }} = \left\{\left[ (\mathbf {H} W _ {Q} + u) W _ {R} ^ {\top} \right] \odot \boldsymbol {\Phi} \right\} \boldsymbol {\Psi} ^ {\top} + \left\{\left[ (\mathbf {H} W _ {Q} + u) W _ {R} ^ {\top} \right] \odot \boldsymbol {\Pi} \right\} \boldsymbol {\Omega} ^ {\top}
$$

这导致了 $O ( 2 T D ^ { 2 } + 4 T ^ { 2 } D )$ 的复杂度，与内容项相当。

## A.3 潜在的模型扩展

在本节中，我们讨论 Funnel-Transformer 的一些潜在模型扩展。如第 2 节所述，Funnel-Transformer 可以分为具有压缩功能的编码器和恢复全长词元级（token-level）表示的解码器。为了进一步扩展所提出的模型，首先注意到编码器-解码器框架可以表述为一种更通用的形式：

$$
\begin{array}{l} \mathbf {h} _ {\text {enc}} = \text {Encoder} (\mathbf {x} _ {\text {enc}}), \\ \mathbf {h} _ {\text {dec}} = \text {Decoder} (\mathbf {h} _ {\text {enc}}, \mathbf {x} _ {\text {dec}}), \end{array}
$$

其中 $\mathbf { x } _ { \mathrm { e n c } }$ 和 $\mathbf { x } _ { \mathrm { d e c } }$ 分别是编码器输入序列以及可选且特定于问题的解码器输入。编码器的目标是将输入序列 $\mathbf { x } _ { \mathrm { e n c } }$ 压缩为长度缩短的隐藏表示 $\mathbf { h } _ { \mathrm { e n c } }$。然后，以解码器输入 $\mathbf { h } _ { \mathrm { e n c } }$（若有）为条件，解码器将从 $\mathbf { h } _ { \mathrm { e n c } }$ 中提取相关信息/表示，以解决手头具体的 NLP 问题。接下来，我们将介绍 Funnel-Transformer 的通用形式如何被实例化为特定形式以解决相应的 NLP 问题。

序列级预测 这本质上是我们在大多数实验中所考虑的情况，即我们希望获得输入序列的向量表示，例如文本分类。在这种情况下，我们实际上并不需要解码器 $\mathbf { x } _ { \mathrm { d e c } } \left( \mathrm { i . e . \ x _ { \mathrm { d e c } } = } \emptyset \right)$，解码器仅需从 $\mathbf { h } _ { \mathrm { e n c } }$ 中提取对应于 [cls] 词元（token）的隐藏表示，并将其送入特定于任务的结构（例如分类器）中。

词元级预测（Token-level prediction）在诸如 MLM 预训练、SQuAD 和序列标注等词元级预测任务中，我们需要一个解码器从压缩序列 $\mathbf { h } _ { \mathrm { e n c } } .$ 中恢复词元级表示。在许多情况下，$\mathbf { x } _ { \mathrm { d e c } }$ 可以仅为原始序列或其词元级隐藏表示，以提供每个词元的细粒度低层信息，从而简化优化过程。在本文中，我们利用第 1 个块（在第一次池化操作之前）最后一层的隐藏状态作为额外的解码器输入。

但对于利用额外输入信号的问题，例如 XLNet [3] 中用于排列语言建模的排列顺序。该额外信息可以通过解码器输入 $\mathbf { x } _ { \mathrm { d e c } }$ 注入到 Funnel-Transformer 中，以（近似地）恢复注意力机制中某些更复杂的控制。

序列到序列问题（Sequence-to-sequence problems）NLP 任务的另一大重要类别是序列到序列问题，包括机器翻译、文本摘要和对话生成，其最先进的解决方案是传统的编码器-解码器（encoder-decoder）框架。因此，Funnel-Transformer 自然地契合这些任务，其中解码器输入 $\mathbf { X } _ { \mathrm { d e c } }$ 对应目标文本序列，而编码器输入 $\mathbf { X } _ { \mathrm { e n c } }$ 对应源文本序列。这样一来，与传统模型相比的关键区别在于 Funnel-Transformer 所提供的源端压缩。

总的来说，我们总结了将第 2.2 节中介绍的 Funnel-Transformer 拓展到 NLP 任务中的一些潜在方向。最后，尽管本文重点讨论 NLP 任务，但 Funnel-Transformer 也可以应用于处理序列数据的任何任务，例如时间序列和视频流分析。

## B 实验设置与超参数

## B.1 预处理与词元化（Preprocessing & Tokenization）

在本工作进行的所有实验中，我们直接采用了 BERT [2] 最初使用的“uncased”（不区分大小写）WordPiece 模型，其词表大小约为 30K。除了转为小写以及 WordPiece 词元分析器中包含的默认预处理之外，我们进行的唯一定制预处理是在 7 个文本分类任务中去除了一些 http 符号（例如 <b>）。

## B.2 预训练

<table><tr><td>超参数</td><td>Base 规模</td><td>Large 规模</td></tr><tr><td>隐藏层 Dropout</td><td>0.1</td><td></td></tr><tr><td>GeLU Dropout</td><td>0.0</td><td></td></tr><tr><td>注意力 Dropout</td><td>0.1</td><td></td></tr><tr><td>最大序列长度</td><td>512</td><td></td></tr><tr><td>批大小</td><td>256</td><td>8192</td></tr><tr><td>学习率</td><td>1e-4</td><td>2e-4</td></tr><tr><td>步数</td><td>1M</td><td>500K</td></tr><tr><td>预热步数</td><td>10K</td><td>30K</td></tr><tr><td>优化器</td><td colspan="2">Adam 权重衰减</td></tr><tr><td>学习率衰减</td><td colspan="2">线性</td></tr><tr><td>Adam epsilon</td><td colspan="2">1e-6</td></tr><tr><td>权重衰减</td><td colspan="2">0.01</td></tr></table>

Table 7: 预训练超参数。

两种不同预训练设置所使用的超参数汇总于 Table 7 中。一个例外是 B10-10-10H1024 在 Base 规模下使用的学习率。具体而言，我们发现在 Base 规模下，当深度超过 24 层时（如 B10-10-10H1024 的情况），训练可能会变得不稳定，尤其是对于 MLM 目标。因此，在 Base 规模预训练期间，我们将 B10-10-10H1024 F-TFM 的学习率降低至 8e-5。这带来的副作用是训练速度变慢，且微调性能可能会略有下降。然而，当增大批大小（如在 Large 规模设置下）时，我们并未观察到这种不稳定性。

对于 ELECTRA，有两个额外的关键超参数，即判别器损失系数和生成器的相对大小倍数。在本工作中，我们完全没有调整这两个超参数，而是直接采用原论文中的数值，即对于所有使用 ELECTRA 训练的架构，判别器损失系数均设为 50，大小倍数均设为 1/4。此外，在 ELECTRA 训练中，每当使用 F-TFM 作为判别器时，生成器也同样使用 F-TFM。

此外，在所有实验中，我们仅标注了隐藏状态的大小，其余模型尺寸均可由此推导得出：

• 嵌入大小 = 隐藏层大小

• P-FFN 的内部状态大小为“4 × 隐藏层大小”。

• 注意力头维度始终为 64。

• 注意力头数量为“隐藏层大小/64”。

最后，预训练中的另一个重要因素是掩码采样策略。对于 MLM 训练，遵循以往的工作，我们始终采用完整词跨度（最多 5 个完整词）采样。然而，对于 ELECTRA 训练，我们注意到一个奇怪的现象：在 Base 规模设置下，如果使用词跨度采样而不是单 token 采样，Transformer 和 F-TFM 的性能都会显著下降。另一方面，在 Large 规模设置下，使用词跨度采样则表现良好。因此，我们对 Base 规模的 ELECTRA 训练使用单 token 采样，对 Large 规模的 ELECTRA 训练使用词跨度采样。

## B.3 微调

<table><tr><td>超参数</td><td>RTE</td><td>MRPC</td><td>STS-B</td><td>CoLA</td><td>SST-2</td><td>QNLI</td><td>MNLI</td><td>QQP</td></tr><tr><td>隐藏层 dropout</td><td></td><td></td><td></td><td>0.1</td><td></td><td></td><td></td><td></td></tr><tr><td>GeLU dropout</td><td></td><td></td><td></td><td>0.0</td><td></td><td></td><td></td><td></td></tr><tr><td>注意力 dropout</td><td></td><td></td><td></td><td>0.1</td><td></td><td></td><td></td><td></td></tr><tr><td>最大序列长度</td><td></td><td></td><td></td><td>128</td><td></td><td></td><td></td><td></td></tr><tr><td>批大小</td><td>16</td><td>16</td><td>16</td><td>16</td><td>32</td><td>32</td><td>64</td><td>64</td></tr><tr><td>轮数</td><td>10</td><td>10</td><td>10</td><td>10</td><td>5</td><td>3</td><td>3</td><td>5</td></tr><tr><td>学习率衰减</td><td></td><td></td><td></td><td>线性</td><td></td><td></td><td></td><td></td></tr><tr><td>权重衰减</td><td></td><td></td><td></td><td>0.01</td><td></td><td></td><td></td><td></td></tr><tr><td>预热比例</td><td></td><td></td><td></td><td>0.1</td><td></td><td></td><td></td><td></td></tr><tr><td>Adam epsilon</td><td></td><td></td><td></td><td>1e-6</td><td></td><td></td><td></td><td></td></tr><tr><td>超参数</td><td>IMDB</td><td>AG</td><td>DBpedia</td><td>Yelp-2</td><td>Yelp-5</td><td colspan="2">Amazon-2</td><td>Amazon-5</td></tr><tr><td>隐藏层 dropout</td><td></td><td></td><td></td><td>0.1</td><td></td><td></td><td></td><td></td></tr><tr><td>GeLU dropout</td><td></td><td></td><td></td><td>0.0</td><td></td><td></td><td></td><td></td></tr><tr><td>注意力 dropout</td><td></td><td></td><td></td><td>0.1</td><td></td><td></td><td></td><td></td></tr><tr><td>最大序列长度</td><td>512</td><td>128</td><td>128</td><td>512</td><td>512</td><td colspan="2">512</td><td>512</td></tr><tr><td>批大小</td><td>32</td><td>32</td><td>64</td><td>128</td><td>128</td><td colspan="2">128</td><td>128</td></tr><tr><td>轮数</td><td>5</td><td>3</td><td>3</td><td>3</td><td>3</td><td colspan="2">3</td><td>3</td></tr><tr><td>学习率衰减</td><td></td><td></td><td></td><td>线性</td><td></td><td></td><td></td><td></td></tr><tr><td>权重衰减</td><td></td><td></td><td></td><td>0.01</td><td></td><td></td><td></td><td></td></tr><tr><td>预热比例</td><td></td><td></td><td></td><td>0.1</td><td></td><td></td><td></td><td></td></tr><tr><td>Adam epsilon</td><td></td><td></td><td></td><td>1e-6</td><td></td><td></td><td></td><td></td></tr></table>

Table 8：在 GLUE 基准和 7 个文本分类数据集上进行微调的超参数。

对于所有微调实验，我们基本上沿用了 XLNet [3] 所使用的超参数。报告的所有性能数据均在配备 TensorFlow 2.2 的 TPU 上获得。

## B.3.1 GLUE 与文本分类

对于 GLUE 和文本分类数据集，我们首先固定表 8 中显示的大多数超参数的值。然后，我们仅在集合 [1e-5, 2e-5, 3e-5] 中搜索学习率，并根据验证集选择最佳学习率。

遵循先前的研究 [3, 4, 5]，所有 GLUE 性能分别对应于 base 设置下使用不同随机种子运行 5 次的中位数结果，以及 large 设置下运行 15 次的中位数结果。

对于文本分类，base 规模的结果是使用不同随机种子运行 5 次的中位数性能。然而，对于 large 规模的实验，为了与先前的研究 [34, 3] 保持一致，结果是 5 次随机运行中的最佳性能。

## B.3.2 阅读理解

同样遵循 XLNet [3]，在 RACE 和 SQuAD 数据集上进行微调所使用的超参数汇总在表 9 中。“逐层衰减（Layer-wise decay）”是指自顶向下以指数方式衰减各个层的学习率。例如，假设第 24 层使用学习率 l，且逐层衰减率为 α，那么第 m 层的学习率为 $l \alpha ^ { \dot { 2 } 4 - m }$。此外，对于 SQuAD 的两个版本，在 SQuAD v1.1 上评估时，我们直接复用了在 SQuAD v2.0 上训练的模型。

<table><tr><td>超参数</td><td>RACE</td><td>SQuAD</td></tr><tr><td>Dropout</td><td>0.1</td><td></td></tr><tr><td>Attention dropout</td><td>0.1</td><td></td></tr><tr><td>最大序列长度</td><td>512</td><td>512</td></tr><tr><td>训练周期/步数</td><td>5 epochs</td><td>8000 steps</td></tr><tr><td>预热比例/步数</td><td>0.1</td><td>1000 steps</td></tr><tr><td>批大小</td><td>[16, 32]</td><td>48</td></tr><tr><td>学习率</td><td>[1e-5, 2e-5]</td><td>3e-5</td></tr><tr><td>学习率衰减</td><td colspan="2">linear</td></tr><tr><td>权重衰减</td><td colspan="2">0.01</td></tr><tr><td>Adam epsilon</td><td colspan="2">1e-6</td></tr><tr><td>逐层学习率衰减</td><td>1.0</td><td>0.75</td></tr></table>

Table 9: RACE 和 SQuAD 的超参数。

## C 附加实验结果

## C.1 大规模文本分类

<table><tr><td>模型</td><td>IMDB</td><td>AG</td><td>DBpedia</td><td>Yelp-2</td><td>Yelp-5</td><td>Amazon-2</td><td>Amazon-5</td></tr><tr><td>BERT-Large</td><td>4.51</td><td>-</td><td>0.64</td><td>1.89</td><td>29.32</td><td>2.63</td><td>34.17</td></tr><tr><td>ROBERTA-Large</td><td>3.50</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>XLNet-Large</td><td>3.20</td><td>4.45</td><td>0.64</td><td>1.37</td><td>27.05</td><td>2.11</td><td>31.67</td></tr><tr><td>B10-10-10H1024</td><td>3.36</td><td>4.66</td><td>0.60</td><td>1.33</td><td>27.14</td><td>2.10</td><td>31.64</td></tr><tr><td>B8-8-8H1024</td><td>3.42</td><td>4.96</td><td>0.63</td><td>1.39</td><td>27.20</td><td>2.14</td><td>31.74</td></tr><tr><td>MPNet</td><td>4.40</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>B6-6-6H768</td><td>3.72</td><td>5.00</td><td>0.64</td><td>1.50</td><td>27.73</td><td>2.27</td><td>32.11</td></tr><tr><td>B6-3x2-3x2H768</td><td>3.82</td><td>5.12</td><td>0.64</td><td>1.58</td><td>27.96</td><td>2.32</td><td>32.23</td></tr><tr><td>B4-4-4H768</td><td>4.12</td><td>5.09</td><td>0.67</td><td>1.70</td><td>28.40</td><td>2.35</td><td>32.46</td></tr></table>

Table 10: 大规模预训练下的文本分类性能对比。

Table 10 包含了在大规模训练设定下 7 个文本分类任务上的性能对比。与 GLUE 基准测试结果类似，相较于以往基于 Transformer 的结果，所提出的 F-TFM 在使用更少 FLOPs 的情况下取得了相当的性能。

## C.2 训练成本对比

在本节中，我们在 TPU 和 GPU 平台上测试了 F-TFM 与标准 Transformer 相比的预训练和微调速度。对于预训练速度评估，我们使用 TensorFlow 在 TPU v3-16（16 核 x 16Gb）上测试了 F-TFM。对于微调速度评估，我们使用 TensorFlow 在 TPU v2-8（8 核 x 8Gb）上以及使用 PyTorch 在 Nvidia-V100（16Gb）GPU 上测试了 F-TFM。TensorFlow 版本为 2.2.0，PyTorch 版本为 1.5.0。对于 GPU 实验，我们使用了 Google Cloud Platform 上的一个 8-GPU 节点。所有运行速度均基于 FP16 优化器进行报告。在 PyTorch 实现中，我们使用 apex<sup>3</sup> 包中 AMP 管理器的“O2”选项来处理 FP16 优化。对于微调，我们考虑了三种不同的序列长度，即 128、256 和 512。对于预训练，我们仅考虑序列长度 512。在每种情况下，我们都选择了设备显存允许的最大可能批次大小（batch size）。我们通过对固定长度的随机输入序列执行 1000 步梯度下降来测量模型的实际运行时间。

首先，我们对比了微调阶段的模型速度。需要注意的是，在此设定下不使用解码器。Table 11 和 12 分别总结了在 GPU 和 TPU 上的微调运行时间对比。

• 在基础模型（L12H768）组中，我们观察到 B6-6-6H768 的运行速度与基础 Transformer 模型相当甚至更快，尽管 B6-6-6 的层数更深且拥有更多的参数。

<table><tr><td>序列长度</td><td colspan="3">128</td><td colspan="3">256</td><td colspan="2">512</td><td></td></tr><tr><td rowspan="2">指标</td><td colspan="2">运行时间</td><td rowspan="2">显存</td><td colspan="2">运行时间</td><td rowspan="2">显存</td><td colspan="2">运行时间</td><td rowspan="2">GLUE</td></tr><tr><td>1 GPU</td><td>8 GPUs</td><td>1 GPU</td><td>8 GPUs</td><td>8 GPUs</td><td>显存</td></tr><tr><td>每 GPU 批大小</td><td colspan="3">64</td><td colspan="3">32</td><td colspan="2">16</td><td></td></tr><tr><td>L12H768</td><td>1.00x</td><td>1.00x</td><td>9.2G</td><td>1.00x</td><td>1.00x</td><td>11.0G</td><td>1.00x</td><td>14.3G</td><td>84.40</td></tr><tr><td>B6-6-6</td><td>0.97x</td><td>0.99x</td><td>9.1G</td><td>0.95x</td><td>0.97x</td><td>10.3G</td><td>0.94x</td><td>12.5G</td><td>85.37</td></tr><tr><td>B6-3x2-3x2</td><td>0.93x</td><td>0.93x</td><td>8.4G</td><td>0.91x</td><td>0.92x</td><td>9.5G</td><td>0.90x</td><td>11.8G</td><td>84.78</td></tr><tr><td>B4-4-4</td><td>0.67x</td><td>0.67x</td><td>6.6G</td><td>0.65x</td><td>0.66x</td><td>7.5G</td><td>0.64x</td><td>9.0G</td><td>83.99</td></tr><tr><td>每 GPU 批大小</td><td colspan="3">32</td><td colspan="3">12</td><td colspan="2">4</td><td></td></tr><tr><td>L24H1024</td><td>1.00x</td><td>1.00x</td><td>14.8G</td><td>1.00x</td><td>1.00x</td><td>14.4G</td><td>1.00x</td><td>13.9G</td><td>86.62</td></tr><tr><td>B10-10-10</td><td>0.87x</td><td>0.92x</td><td>14.0G</td><td>0.90x</td><td>0.93x</td><td>13.0G</td><td>0.96x</td><td>12.7G</td><td>87.03</td></tr><tr><td>B8-8-8</td><td>0.70x</td><td>0.73x</td><td>11.6G</td><td>0.73x</td><td>0.75x</td><td>10.8G</td><td>0.78x</td><td>10.5G</td><td>86.70</td></tr></table>

Table 11: F-TFM 与标准 Transformer 在 GPU 上的运行时间及显存消耗对比。在每个模型组中，标准 Transformer（首个模型）均作为其余 F-TFM 模型的基准。注意，在每块 GPU 的批大小相同时，1 块 GPU 与 8 块 GPU 的显存消耗大致相同。

<table><tr><td>序列长度</td><td>128</td><td>256</td><td>512</td><td></td></tr><tr><td>指标</td><td colspan="3">在 8 个 TPU 核心（TPUv2-8）上的运行时间</td><td>GLUE</td></tr><tr><td>批大小 / TPU 核心</td><td>64</td><td>32</td><td>16</td><td></td></tr><tr><td>L12H768</td><td>1.00x</td><td>1.00x</td><td>1.00x</td><td>84.40</td></tr><tr><td>B6-6-6</td><td>0.99x</td><td>0.88x</td><td>0.81x</td><td>85.37</td></tr><tr><td>B6-3x2-3x2</td><td>0.97x</td><td>0.87x</td><td>0.77x</td><td>84.78</td></tr><tr><td>B4-4-4</td><td>0.69x</td><td>0.62x</td><td>0.55x</td><td>83.99</td></tr><tr><td>批大小 / TPU 核心</td><td>16</td><td>8</td><td>4</td><td></td></tr><tr><td>L24H1024</td><td>1.00x</td><td>1.00x</td><td>1.00x</td><td>86.62</td></tr><tr><td>B10-10-10</td><td>0.89x</td><td>0.81x</td><td>0.73x</td><td>87.03</td></tr><tr><td>B8-8-8</td><td>0.66x</td><td>0.60x</td><td>0.56x</td><td>86.70</td></tr></table>

Table 12: TPU v2-8 上 F-TFM 与标准 Transformer 之间的运行时间对比。在每个模型组中，标准 Transformer（首个模型）用作其余 F-TFM 模型的基准。

此外，与基础 Transformer 模型相比，B6-6-6H768 取得了更好的结果。类似的结论也适用于 B6-3x2-3x2 模型，该模型与基础模型具有相同数量的参数。B4-4-4 模型与基础模型具有相同的深度和模型参数，能够在不损失过多性能的情况下提供 30%-50% 的加速。

• 在大模型（L24H1024）组中，结论类似。较大模型 B10-10-10 的速度与大模型几乎相同，而 B8-8-8 的速度明显快于大模型。此外，当序列长度等于 512 时，F-TFM 在 TPU 上的加速效果比 GPU 更为明显。

• 在这两个组中，所有测试的 F-TFM 变体与标准 TFM 模型相比都具有更小的显存/内存占用，展示了 F-TFM 的内存效率。

接下来，我们在表 13 中比较了在 MLM 目标下预训练期间的模型速度，由于解码器的存在，该目标会产生额外的开销。结果表明，与标准 Transformer 相比，所提出的方法仍然可以大幅提高预训练速度，尽管速度提升幅度略小于微调阶段。总之，本研究表明，所提出的方法在现代并行计算平台的微调和预训练阶段都更为高效。

<table><tr><td>序列长度</td><td colspan="2">512</td></tr><tr><td></td><td>运行时间</td><td>FLOPs</td></tr><tr><td>#TPU 核心数 / 总批大小</td><td colspan="2">16 / 512</td></tr><tr><td>L12H768</td><td>1.00x</td><td>1.00x</td></tr><tr><td>B6-6-6H768D2</td><td>0.99x</td><td>1.04x</td></tr><tr><td>B6-3x2-3x2H768D2</td><td>0.97x</td><td>1.04x</td></tr><tr><td>B4-4-4H768D2</td><td>0.79x</td><td>0.75x</td></tr><tr><td>#TPU 核心数 / 总批大小</td><td colspan="2">16 / 128</td></tr><tr><td>L24H1024</td><td>1.00x</td><td>1.00x</td></tr><tr><td>B10-10-10H1024D2</td><td>0.83x</td><td>0.81x</td></tr><tr><td>B8-8-8H1024D2</td><td>0.71x</td><td>0.66x</td></tr></table>

Table 13: TPU 预训练速度比较。后缀“D2”表示 F-TFM 模型具有 2 个解码器层。