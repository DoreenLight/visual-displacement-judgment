# Hydra Attention: 多种头数的高效注意力

Daniel Bolya<sup>1,2⋆</sup>, Cheng-Yang Fu<sup>2</sup>, Xiaoliang Dai<sup>2</sup>, Peizhao Zhang<sup>2</sup>, 和 Judy Hofman<sup>1</sup>

<sup>1</sup> 佐治亚理工学院（Georgia Tech）

{dbolya,judy}@gatech.edu

<sup>2</sup> Meta AI

{chengyangfu,xiaoliangdai,stzpz}@fb.com

**摘要.** 尽管 Transformer 已经开始在许多视觉任务中占据主导地位，但将其应用于大尺寸图像仍然在计算上十分困难。一个重要的原因在于，自注意力（self-attention）的计算量与 token（词元）数量成二次方关系，而 token 数量本身又与图像尺寸成二次方关系。在较大的图像（例如 1080p）上，仅用于创建和应用注意力矩阵的计算就占网络总计算量的 60% 以上。为了解决这一问题，我们提出了 Hydra Attention，一种面向视觉 Transformer（ViTs）的高效注意力操作。颇具反讽意味的是，这种效率源自于将多头注意力（multi-head attention）推向其极端：通过使用与特征数一样多的注意力头，Hydra Attention 在 token 和特征维度上的计算复杂度均为线性，且不包含隐藏常数项，这使其在现成的（off-the-shelf）ViT-B/16 上比标准自注意力快出一个 token 数量的倍数。此外，Hydra Attention 在 ImageNet 上保持了高精度，在某些情况下甚至有所提升。

**关键词：** 视觉 Transformer，注意力，token 效率

## 1 引言

由于其通用性和从大量数据中学习的高容量，Transformer [32] 在过去几年中已成为自然语言处理（NLP）领域的主导力量 [17,25,6]。如今，随着视觉 Transformer（Vision Transformers，ViTs）[10] 的提出，同样的变革也正在视觉领域上演。

然而，与 NLP 不同的是，无论是 NLP 中 BERT [17] 还是视觉中 ViT [10] 这种纯粹的 Transformer 形式，实际上并不是主导计算机视觉任务的力量。相反，更多针对视觉任务专门设计的、基于注意力的架构（如 Swin [21] 或 MViT [11,20]），或是注意力与卷积混合的架构（如 LeViT [13]），正在被广泛采用。

![](images/38ba4d8edb39d7434cdad03c279fbf0438444e2d88a8d563d8887085788d39f7.jpg)  

图 1：Hydra 注意力。标准注意力 [32] 的计算复杂度与 token 数量 T 的平方成正比。通过使用可分解核（decomposable kernel），我们可以按照文献 [16] 的方式重新组织运算顺序，使注意力的复杂度与特征维度 D 的平方成正比。而我们的 Hydra 注意力更进一步，通过最大化注意力头数量 H，在空间和时间上均实现了 O(TD) 的运算复杂度。

这种差异背后的主要原因是计算效率：专用的视觉Transformer可以用更少的计算量实现更好的性能——无论是通过加入卷积层、通过使用面向视觉的局部窗口注意力，还是通过其他一些低成本地添加视觉归纳偏置的方式。虽然纯视觉Transformer（pure ViT）在大规模下可以表现良好（在ImageNet [38]上达到90.45%的top-1准确率），但纯Transformer的核心机制——多头自注意力（multihead self-attention）[32]——在多个下游任务所需的、在大尺寸图像上应用模型时，会成为一个极大的瓶颈。

事实上，在1080p图像上应用现成的ViT（这在诸如CityScapes [7]等分割基准任务中很常见）时，神经网络中总计算量的60%（见表4）都消耗在自注意力的注意力矩阵的构建与应用上，相比之下，在224×224的ImageNet [9]图像上，这一比例仅为4%。在纯Transformer中，这些注意力矩阵的计算量随token数量的平方增长，仅这一点就已经让计算成本高得难以承受（例如在NLP中处理长句子时）。但在ViT中，问题进一步被放大，因为token数量随图像尺寸的平方增长，这意味着图像尺寸翻倍会使注意力中的计算量增加16倍。

在NLP领域，已经有大量技术被探索用于应对该问题。一些工作通过使用"核技巧"（kernel trick）[5,28,16,24]重新排列计算顺序，或投影到与token无关的低秩空间[34,5,24]（有些工作两者兼用），从而引入了"线性"注意力（就token数量而言）。然而，这些"线性"注意力方法大多以特征维度的计算量替换了token维度的计算量，使它们的开销仍然相当大。事实上，最近Flash Attention [8]已经证明，即使token数量达到数千级别，一种IO高效的多头自注意力实现也能胜过大多数这类"线性"注意力方法。

在视觉领域，也有一些工作尝试了高效注意力，但都没有在传统的ViT框架中独立进行探索。PolyNL [2]将注意力视为一种高效的三阶多项式，但尚未在ViT架构中进行探索。Attention Free Transformer [37]有一个同样高效的AFT-Simple变体，但在纯ViT中表现较差，并且需要额外的卷积和位置编码支持。我们在标准的DeiT [31]框架中对这两种方法都进行了测试（见表1），发现这两种方法虽然高效，但都带来了显著的准确率下降。因此，文献中仍存在一个空缺，需要一种真正高效、准确且通用的多头自注意力替代方案。

为此，我们引入了 Hydra Attention（参见图 1）。我们的方法源于线性注意力中一种看似矛盾的行为：在标准多头自注意力中，向模型添加更多的头并不会改变计算量。然而，在改变线性注意力中的运算顺序之后，添加更多的头实际上会降低该层的计算成本。我们将这一观察推到极致，将模型中头的数量设置为等于特征数量，从而构建出一种在 token 和特征两个维度上都呈线性复杂度的注意力模块。

Hydra Attention 不仅是对先前高效注意力工作（参见第 3.5 节）的一种更具一般性的形式化表达，而且在选用合适的核函数时，能够显著提升精度（参见表 1）。事实上，当与标准多头注意力混合使用时，Hydra Attention 能够在加速的同时提升 DeiT-B 基线模型的精度（参见图 4）。由于我们的方法源自多头注意力，因此保留了注意力的若干优良特性，例如可解释性（参见图 3）以及对不同任务的通用性。

然而，尽管 Hydra Attention 对大图像具有通用性和高效性，但本文仅专注于使用 DeiT-B [31] 进行 ImageNet [9] 分类，该任务传统上使用较小的 224 × 224 和 384 × 384 图像。虽然在此场景下效率提升并不显著（根据图像大小在 10% 至 27% 之间），但其他高效注意力方法（例如 [37, 2]）在该设定下已经出现巨大的精度下降（参见表 1），而 Hydra Attention 则不存在这一问题。我们希望 Hydra Attention 能成为未来通用、纯粹的、面向大量 token 的 Transformer 模型的基石。

我们的贡献如下：我们开展了一项研究，验证一个 Transformer 能够拥有多少个头（图 2），并发现 softmax 注意力的上限为 12，但使用合适的核函数时，任意数量的头都是可行的。随后，我们利用这一观察，针对纯 Transformer 模型引入了 Hydra Attention（第 3 节），方法是在多头自注意力中增加头的数量。接着，我们对 Hydra Attention 的作用进行了数学分析（第 3.4 节），并提出了一种可视化其注意力焦点的方法（图 3）。最后，我们发现，通过将特定的注意力层替换为 Hydra Attention（图 4），在使用 DeiT-B [31] 应用于 ImageNet-1k [9] 时，我们既可以将精度提升 1%，也可以在严格更快的模型上达到与基线相当的精度。

## 2 相关工作

本文的目标是去除多头自注意力中关于 token 的平方级计算瓶颈，从而加速 Transformer 的推理时间。

**高效注意力（Efficient Attention）** 多头自注意力 [32] 是一种出了名的缓慢操作，已有大量工作尝试在各个领域中解决其计算上的不足。

在自然语言处理（NLP）中，一些研究工作使用可分解核函数来近似注意力机制 [5,28,16,24]。这种"核技巧"使它们能够重新组织矩阵乘法，使其相对于特征维度是二次的，而不是相对于标记维度。其中一些方法更进一步，通过投影到低秩空间来降低该矩阵乘法的维度 [34,5,24]。然而，这些"线性"注意力方法将跨标记的计算换成了跨特征的计算，这会使它们变得昂贵。事实上，在本文的研究领域（ImageNet 分类）中，标记数量不足以证明这些方法的合理性，并且它们中的大多数会产生更慢的模型。并且即使有数千个标记，Flash Attention [8] 已经表明，具有 IO 感知实现的多头自注意力实际上可以超越这些方法中最快的那些。

但是，重新组织操作并不是加速注意力的唯一方法。事实上，在视觉中将注意力"线性化"的最常见方法是使用局部窗口注意力（例如，[21,3,19]）。这在计算上确实相对于标记数量是线性的，但局部窗口注意力可能难以计算（特别是在 Swin [21] 的情况下），并且这只能在密集的、具有空间顺序的模态（如图像和视频）中实现。

我们的目标反而是提出一种线性注意力方法，它是高效的、计算快速的，并且在多种不同模态上具有通用性。

## 高效 Transformer

替换注意力模块并不是加速 Transformer 推理时间的唯一方法。事实上，根据任务和标记数量的不同，其他高效的 Transformer 方法可能更受欢迎。例如，在 ImageNet [9] 分类任务上，注意力仅占总网络计算的 4%，这意味着如果只修改注意力模块，最大可获得的加速比为 4%。

有几种高效的视觉 Transformer 将卷积和注意力混合在一起，以创建更高效的最终产品，例如 LeViT [13]、MobileViT [22]、Mobile-Former [4] 和 LVT [35]。所有这些对于图像来说都是有效的策略，我们将它们视为相邻的技术。其他视觉特定的注意力论文，例如 [37,2]，在高效注意力之外还使用卷积，这使得难以辨别改进是来自注意力方法还是卷积的引入。

在本文中，我们不对底层 ViT 架构做任何修改，只是将多头自注意力替换为 Hydra Attention，以便清晰地隔离其对性能的影响。

多头注意力 Hydra Attention 依赖于增加多头注意力（multihead attention）中使用的头数。有趣的是，自从 [32] 引入多头注意力以来，头数的使用并未得到深入探索。已经有一些关于剪枝注意力头的研究 [33,23]，但所有研究都集中在减少头数的方向上。事实上，即使在 [38] 中所探索的最大的 ViT 模型 ViT-G 中，作者也只使用了 16 个注意力头。因此，我们在图 2 中自行开展了这项研究。

## 3 Hydra Attention

标准的（standard）多头自注意力 [32] 与图像中的 token 数量呈二次方缩放。更具体地说，如果 T 是 token 的数量，D 是特征维数，那么创建和应用注意力矩阵的复杂度均为 $O(T^2D)$。当 T 较大时（对于大图像正是这种情况），这就构成了一个问题，因为该操作会很快变得在计算上不可行。

## 3.1 核技巧（The Kernel Trick）

如第 2 节所述，已有诸多工作 [5,28,16,24] 尝试通过引入"线性"（linear）注意力来应对这一问题。给定 $\mathbb{R}^{T \times D}$ 中的查询（query）Q、键（key）K 和值（value）V，标准的 softmax 自注意力计算如下

$$
A (Q, K, V) = \mathrm{softmax} \left(\frac {Q K ^ {T}}{\sqrt {D}}\right) V\tag{1}
$$

计算 $Q K ^ { T }$ 的复杂度为 $O ( T ^ { 2 } D )$，并生成一个 $T \times T$ 的矩阵，因此随着 T 的增大其扩展性较差。正如文献 $[1 6]$ 所述，我们可以通过将 softmax(·) 视为 $Q$ 与 K 之间的逐对相似度来推广这一操作。也就是说，对于某个相似度函数 sim(·)，我们可以写出

$$
A (Q, K, V) = \mathrm{sim} (Q, K) V\tag{2}
$$

如果随后选择一个具有特征表示 $\phi ( \cdot )$ 的可分解核（decomposable kernel），使得 sim $( x , y ) = \phi ( x ) \phi ( y ) ^ { T }$，则可以得到

$$
A (Q, K, V; \phi) = (\phi (Q) \phi (K) ^ {T}) V\tag{3}
$$

那么根据结合律，我们可以改变计算的顺序，使得

$$
A (Q, K, V; \phi) = \phi (Q) (\phi (K) ^ {T} V)\tag{4}
$$

这允许我们先计算 $\phi(K)^T V$，这样操作的复杂度为 $O(TD^2)$，并且生成的是一个 $D^2$ 的矩阵，而不是 $T^2$ 的矩阵。注意，该公式与文献 [16] 略有不同：我们将归一化留给了相似度函数处理，而没有显式地完成它。

## 3.2 多头注意力（Multi-Head Attention）

尽管相对于 $T$ 是线性的，但公式 4 的结果仍然不理想：D 通常较大（$\geq 768$），因此构建一个 $D \times D$ 的矩阵并执行 $O(TD^2)$ 量级的运算仍然代价不菲。然而，公式 1 至公式 4 假设我们只构建一个注意力矩阵，即只有一个"头"。

在实际应用中，大多数视觉 Transformer 使用 H 个头（通常在 6 到 16 之间），其中每个头都构建并应用自己的注意力矩阵。参考文献 [32]，每个头分别作用于来自 $Q$、$K$ 和 $V$ 的各自 $D/H$ 维的特征子集。因此，公式 1 变为

$$
A (Q _ {h}, K _ {h}, V _ {h}) = \mathrm{softmax} \left(\frac {Q _ {h} K _ {h} ^ {T}}{\sqrt {D}}\right) V _ {h} \qquad \forall h \in \{1, \ldots , H \}\tag{5}
$$

其中 $Q _ { h } , K _ { h } , V _ { h } \in \mathbb { R } ^ { T \times \frac { D } { H } }$。这保持了总运算量不变：

$$
O (H T ^ {2} D / H) = O (T ^ {2} D)\tag{6}
$$

然而，对于线性注意力而言，情况并非如此。式 (4) 变为

$$
A (Q _ {h}, K _ {h}, V _ {h}; \phi) = \phi (Q _ {h}) \left(\phi (K _ {h}) ^ {T} V _ {h}\right) \quad \forall h \in \{1, \dots , H \}\tag{7}
$$

通过以这种方式计算注意力，添加注意力头实际上会减少运算次数：

$$
O (H T (D / H) ^ {2}) = O (T D ^ {2} / H)\tag{8}
$$

## 3.3 增加头数

给定公式 8，我们向网络中增加的头数越多，多头线性注意力（multi-head linear attention）的速度就越快。这就引出一个问题：我们到底能合理地增加多少个头？实际中大多数 Transformer 使用 6 到 16 个头 [32,17,10,38]，其数量取决于特征维度 $D_:$，但如果将头的数量增加到超出这个范围会发生什么呢？

为了找出答案，我们在 ImageNet-1k [9] 上训练 DeiT-B [31]，并使用标准多头自注意力（公式 5，MSA）配合 softmax，或多头线性注意力（公式 7，MLA）配合余弦相似度（cosine similarity）来变化头数 H，将结果绘制在图 2 中。在内存使用方面，MSA 在 $H > 96$ 时内存耗尽，MLA 在 $H < 3$ 时内存耗尽。

在性能方面，虽然 MSA 的准确率在 $H > 12$ 时急剧下降，但带余弦相似度的 MLA 的准确率一直保持相当稳定，直至 H = 768。令人惊讶的是，在这么大的头数下，H 等于 $D_:$，这意味着每个头仅有标量特征可用！

![](images/c8d7421fcd823e9c9b43457568fae9253df3d086119d7b6c6eab3fc4d08ed276.jpg)  

图 2：变化的头数。我们在 ImageNet-1k 上使用不同数量的头训练 DeiT-B 模型，分别采用使用 softmax 的标准自注意力（蓝色）和使用余弦相似度（cosine similarity）的多头线性注意力（红色）。标准自注意力的结果在 $H > 9 6$ 时出现内存不足，多头线性注意力的结果在 $H < 3$ 时出现内存不足。随着头数的增加，softmax 注意力在准确率上似乎出现崩溃，而多头线性注意力则保持稳定。注意 H 必须整除 $D = 7 6 8$。

## 3.4 Hydra 技巧

如图 2 所示，只要相似函数 $\text{sim}(x, y)$ 不是 softmax，就可以将 $H$ 任意放大。为了利用这一点，我们引入了"Hydra 技巧"，其中设定 $H = D$

$$
A (Q _ {h}, K _ {h}, V _ {h}; \phi) = \phi (Q _ {h}) \left(\phi (K _ {h}) ^ {T} V _ {h}\right) \quad \forall h \in \{1, \dots , D \}\tag{9}
$$

在这种情况下，每个 $Q _ { h } , K _ { h } , V _ { h }$ 都是 $\mathbb { R } ^ { T \times 1 }$ 中的一个列向量。如果随后我们沿各个注意力头对该操作进行向量化，那么最终得到

$$
\operatorname{Hydra} (Q, K, V; \phi) = \phi (Q) \odot \sum_ {t = 1} ^ {T} \phi (K) ^ {t} \odot V ^ {t}\tag{10}
$$

其中 ⊙ 表示逐元素乘法。注意此向量化形式与公式 9 之间存在一个细微的差异：$\phi$ 作用于 $Q$ 和 $K$ 的整体，而不是作用于单独的列向量 $Q _ { h }$ 和 $K _ { h }$。这一点很重要，因为对于每个 token，$Q _ { h }$ 和 $K _ { h }$ 都是标量，而对两个标量取相似度是非常受限的（例如，余弦相似度只能输出 $\cdot 1 , 0 , \mathrm { o r } + 1$）。

此外，虽然 $\operatorname { E q . }$ 10 的推导源自多头注意力（multihead attention），但它实际上执行的操作相当不同：它首先创建一个全局特征向量 $\bar { \sum _ { t = 1 } ^ { T } } \phi ( K ) ^ { \breve { t } } \odot V ^ { t }$，该向量聚合了图像中所有 token 的信息。然后，每个 $\phi ( Q )$ 对该全局特征在每个输出 token 上的重要性进行门控。因此，Hydra Attention 通过一个全局瓶颈来混合信息，而不是像标准自注意力那样进行显式的 token-to-token 混合。

这导致了如下计算复杂度

$$
O (T D (D / H)) = O (T D)\tag{11}
$$

最终我们得到一个高效的 token 混合模块，其复杂度在 token 数量和特征维度上均为线性，并且不像其他线性注意力方法（如 [5,34,16]）那样存在额外的常数项。需要注意的是，该技术的空间复杂度同样为 $O ( T D )$，这对实际运行速度至关重要，因为许多操作受 IO 限制（参见 [8]）。

## 3.5 与其他工作的关系

在文献中还存在一些其他的 $O ( T D )$ 注意力候选方法：Attention-Free Transformer [37]（具体为 AFT-Simple）和 PolyNL [2]。在本节中，我们将探讨式 10 中所述的 Hydra Attention 与上述各方法之间的关联。

AFT-Simple [37] 被描述为

$$
\mathrm{AFT-Simple} (Q, K, V) = \sigma (Q) \odot \sum_ {t = 1} ^ {T} \operatorname{softmax} (K) ^ {t} \odot V ^ {t}\tag{12}
$$

其中 $\sigma(\cdot)$ 表示 sigmoid。如果我们允许 $\phi$ 在 Q 和 K 之间变化，这就是公式 10 的一个直接特例，其中 $\phi(Q) = \sigma(Q)$ 且 $\phi(K) = \mathrm{softmax}(K)$。

另一方面，PolyNL [2] 被描述为

$$
\operatorname{PolyNL} (X; W _ {1}, W _ {2}, W _ {3}) = \left(X \odot \frac {1}{T} \sum_ {t = 1} ^ {T} X W _ {1} \odot X W _ {2}\right) W _ {3}\tag{13}
$$

如果我们记 $K = X W _ { 1 }$ 并且 $V = X W _ { 2 }$，并令 $\phi _ { \mathrm { m e a n } } ( x ) = x / \sqrt { T }$，我们可以写出

$$
\operatorname{PolyNL} (X; W _ {1}, W _ {2}, W _ {3}) = \operatorname{Hydra} (X, K, V; \phi_ {\text {mean}}) W _ {3}\tag{14}
$$

因此，Hydra 注意力可以看作是其他 $O ( T D )$ 注意力方法的一种更一般化形式。

## 4 实验

对于所有实验，除非另有说明，我们使用 DeiT-B [31] 的默认设置，在 ImageNet-1k [9] 上进行训练，并以验证集上的 Top-1 准确率作为报告指标。若未另行指定，式 (10) 中 $\phi ( \cdot )$ 所使用的函数为 L2 归一化，从而使 $\mathrm{sim}( \cdot , \cdot )$ 为余弦相似度。为了计算吞吐量，我们在多个批次大小上进行扫描，并在丢弃 10 次预热迭代后，报告 30 个批次中的最高平均吞吐量。

<table><tr><td>Method</td><td>Kernel</td><td> $\phi(Q)$ </td><td> $\phi(K)$ </td><td>Accuracy</td></tr><tr><td>Hydra</td><td>Cosine Similarity</td><td colspan="2"> $x/||x||_2$ </td><td>76.37</td></tr><tr><td>Hydra</td><td>Mean</td><td colspan="2"> $x/\sqrt{T}$ </td><td>75.95</td></tr><tr><td>Hydra</td><td>Tanh Softmax</td><td> $\tanh(x)$ </td><td> $softmax(x)$ </td><td>74.18</td></tr><tr><td>AFT-Simple [37]</td><td>Sigmoid Softmax</td><td> $\sigma(x)$ </td><td> $softmax(x)$ </td><td>74.02</td></tr><tr><td>PolyNL [2]</td><td>Mean</td><td colspan="2"> $x/\sqrt{T}$ </td><td>73.96</td></tr></table>

表 1：核函数选择。此处我们通过公式 10 中的特征表示 $\phi(\cdot)$ 来改变核函数的选择。此外，我们还在此处与第 3.5 节中提到的 AFT 和 PolyNL 进行了比较。需要注意的是，某些核函数可以是非对称的，其 $\phi(Q)$ 和 $\phi(K)$ 可以不同。更多核函数请参见附录。

## 4.1 核函数的选择

在大多数实验中，我们沿用文献 [16]，使用余弦相似度作为公式 10 中的核函数。在表 1 中，我们探索了其他可能的核函数，包括第 3.5 节中讨论的其他候选注意力替代方法所使用的核函数。然而，在我们测试的所有核函数中，没有一种能超越简单的余弦相似度。

这可能是因为余弦相似度改变了注意力的本质。在 MSA（公式 5）中，注意力仅混合 $V$ 中包含的信息，因为混合权重 $\text{sim}(Q, K)$ 必须和为 1。但在使用余弦相似度或其他不受限的点积核（如均值）时，情况并非如此。事实证明，权重和为 1 这一性质一开始可能并不是我们想要的：文献 [37] 中描述的 AFT-Simple（如公式 12 所示）设定 $\phi(Q) = \sigma(Q)$、$\phi(K) = \text{softmax}(K)$，这更接近于严格混合 $V$，但性能因此受到影响（见表 1）。

我们还测试了使用 $\text{tanh}(Q)$ 替代 $\sigma(Q)$，以探究余弦相似度允许结果为负是否是原因，但其性能仅略优于 AFT-Simple。因此，在这种计算受限的环境中，让核函数尽可能不受限制，同时以某种方式对其进行归一化，似乎非常重要。我们测试了其他几种核函数，并将其记录在附录中，但没有一种能超越这种简单方法。

## 4.2 可视化 Hydra 注意力

自注意力最受赞誉的特性之一是其可解释性：可视化基于注意力的模型所关注的内容（例如使用注意力展开图（attention rollout）[1]）通常非常直接。但对于 Hydra 注意力而言，这一点则不太适用。

为了可视化 Hydra 注意力模块的关注点，我们可以构造注意力矩阵 $\phi(Q)_h \phi(K)_h^T$（其中 $h \in \{1, \ldots, D\}$），但每个矩阵的秩都为 1，并且当每个矩阵负责不同的特征维度时，如何组合 $D$ 个不同的注意力矩阵并不清楚。简单地将各个头平均在一起会产生无意义的结果，因为每个特征维度编码了不同的信息。

![](images/d86445641d29aead0f728e1cda1b75e90ccd1480061bfbf9441b47da71d1d821.jpg)  

图 3：Hydra 注意力可视化。如第 4.2 节所述，对最后一层中类别标记（class token）的 Hydra 注意力进行可视化。左侧 4 张图像被正确预测，而右侧两个示例被错误分类。在右上角的图像中，网络聚焦于错误狗的头部，从而猜测了错误的品种。而在右下角，网络则完全忽略了鸟的存在。更多示例请参见附录。

我们换个角度，看看每个 token 对类别 token 输出的贡献信息。如果只从公式 10 中采样类别 token c 的输出，则得到

$$
\phi (Q) ^ {c} \odot \sum_ {t = 1} ^ {T} \phi (K) ^ {t} \odot V ^ {t} = \sum_ {t = 1} ^ {T} \phi (Q) ^ {c} \odot \phi (K) ^ {t} \odot V ^ {t}\tag{15}
$$

因此，每个 token t 对 class token c 的输出都有一个贡献，由下式给出：

$$
\phi (Q) ^ {c} \odot \phi (K) ^ {t} \odot V ^ {t}\tag{16}
$$

为了说明这与最终预测之间的关系，我们可以采用一种与 Grad-CAM [27] 类似的方法：将损失设为预测类别的 logit，然后获取相对于 Hydra 注意力层（Hydra attention layer）输出的梯度 g。随后，每个 token 沿该梯度方向的贡献为

$$
(\phi (Q) ^ {c} \odot \phi (K) ^ {t} \odot V ^ {t}) ^ {T} g\tag{17}
$$

我们在图 3 和附录中针对若干不同的图像绘制了该量。对于这些可视化，我们沿 token 方向对公式 17 进行归一化，并展示正值。这些聚焦图表明，尽管数学形式可能有所不同，Hydra 注意力所执行的功能与标准自注意力基本相同。

## 4.3 哪些层可以被替换？

如第 3.4 节和第 4.1 节所述，使用余弦相似度（cosine similarity）核的 Hydra 注意力以一种与标准 MSA [32] 不同的方式在 token 之间混合信息。因此，将网络中的每一个注意力层都替换为 Hydra 注意力可能是不合理的。事实上，Hydra 注意力从 token 中生成一个全局特征，并按 Q 加权后施加到每个 token 上。由于这是一个全局操作，将其放在网络的较后层会更为合理，因为在到达这些层之前信息已经在局部完成了混合。我们在图 4 中对此进行了测试，按照不同的策略逐步将 DeiT-B 中的 MSA 注意力层替换为 Hydra 注意力。

![](images/be2e4a819ca2d902d142313c671a4a0222c88282b31b3a6d6f376bf96d74754b.jpg)  

图 4：哪些层可以被替换？我们采用不同的替换策略，将 softmax 自注意力（softmax self-attention）替换为 Hydra 注意力（Hydra attention）：从头开始替换、从尾开始替换、或交错替换各层。在所有情况下，0 表示没有替换任何层（即基线），12 表示所有层都被替换。令人惊讶的是，采用合适的层替换策略，Hydra 注意力实际上可以将 ImageNet 上的准确率提升 1%，同时速度更快。或者，我们可以替换多达 8 层而不造成准确率下降。

在本实验中，我们观察到，如果从网络的第一层开始替换，模型的性能会迅速下降。然而，事实证明，如果从最后一层开始反向替换，我们实际上可以提高模型的准确率。而且这种提升幅度相当大，以至于我们可以替换网络最后 8 层，仍然能达到基线 DeiT-B 模型的准确率。

那么，如果 Hydra 注意力能够与标准的 softmax 注意力互补，也许将二者结合的最佳方式就是将它们交错排列。在图 4 中，我们也尝试按照"Hydra 注意力层应跟在 MSA 层之后"的原则，交替使用 MSA 层和 Hydra 层。然而，我们并没有观察到这种交错策略相比从后端起始替换有什么明显的优势，这表明重要的因素是 Hydra 层的数量，而不是它们的位置。

值得注意的是，其他高效注意力方法如 AFT [37] 和 UFO-ViT [29] 添加的是卷积层，而非穿插常规的注意力层。添加这些卷积层的目的与使用自注意力进行局部混合大体相同，但目前尚不清楚这些先前方法带来的收益究竟是来自卷积层，还是来自它们所提出的注意力层。在本例中，我们明确地证明 Hydra 注意力不仅能提升模型的速度，还能提升其性能。未来的工作可以尝试使用卷积层来替代。

<table><tr><td>Method</td><td>Accuracy (%)</td><td>FLOPs (G)</td><td>Speed (im/s)</td></tr><tr><td>Standard Attention [32]</td><td>79.57</td><td>17.58</td><td>314.8</td></tr><tr><td>AFT-Simple [37]</td><td>74.02 (-5.55)</td><td>16.87 (-4.0%)</td><td>346.1 (+9.9%)</td></tr><tr><td>PolyNL [2]</td><td>73.96 (-5.61)</td><td>16.87 (-4.0%)</td><td>353.8 (+12.4%)</td></tr><tr><td>Hydra (2 layers)</td><td>80.64 (+1.1)</td><td>17.46 (-0.7%)</td><td>321.9 (+2.3%)</td></tr><tr><td>Hydra (8 layers)</td><td>79.45 (-0.12)</td><td>17.11 (-2.7%)</td><td>334.8 (+6.3%)</td></tr><tr><td>Hydra (12 layers)</td><td>76.40 (-3.17)</td><td>16.87 (-4.0%)</td><td>346.8 (+10.2%)</td></tr></table>

表 2：结果。在 DeiT-B [31] 框架中使用不同注意力方法在 ImageNet-1k [9] 验证集上的结果（以 224px 图像训练），以及在 V100 上测得的吞吐量。Hydra 注意力相较于其他 O(T D) 注意力方法（AFT-Simple [37] 和 PolyNL [2]）造成的精度下降更小。此外，如果不替换网络中的每一个注意力层，Hydra 注意力可以在降低 FLOPs 并提升吞吐量的同时，提高精度或保持精度不变。

<table><tr><td>Method</td><td>Accuracy (%)</td><td>FLOPs (G)</td><td>Speed (im/s)</td></tr><tr><td>Standard Attention [32]</td><td>81.33</td><td>55.54</td><td>92.5</td></tr><tr><td>Hydra (2 layers)</td><td>81.92 (+0.59)</td><td>54.52 (-1.8%)</td><td>96.3 (+4.1%)</td></tr><tr><td>Hydra (7 layers)</td><td>81.26 (-0.07)</td><td>51.96 (-6.4%)</td><td>106.8 (+15.4%)</td></tr><tr><td>Hydra (12 layers)</td><td>77.85 (-3.48)</td><td>49.40 (-11.0%)</td><td>117.6 (+27.1%)</td></tr></table>

表 3：384 像素微调。对表 2 中的模型使用 384 像素图像微调 30 个 epoch 的结果。即便 token 更多，Hydra 注意力在使用 2 层时仍能将准确率较基线提升 0.59%，并在 7 层时将吞吐量提升 15.4%，同时保持与基线相当的准确率。

## 4.4 结果

我们在表 2 中展示了在 ImageNet-1k 上使用 Hydra 注意力与标准 $O ( T ^ { 2 } D )$ 注意力以及其他 O(TD) 方法相比的最终准确率与 FLOPs。当替换全部层时，Hydra 注意力比其他 O(T D) 方法的准确率高出 2.4%。而当替换较少的层时，Hydra 注意力能够严格优于基线标准注意力模型：使用 2 层时，准确率提升 1.1%，FLOPs 减少 0.7%，吞吐量提升 2.3%；使用 8 层时，准确率保持不变，FLOPs 减少 2.7%，吞吐量提升 6.3%。颇为有趣的是，实际吞吐量的提升幅度显著超过了 FLOPs 的下降幅度。这可能源于文献 [8] 中的观察：注意力是内存受限（memory-bound）的，而 Hydra 注意力比标准注意力使用更少的内存。

<table><tr><td rowspan="2">Image Size</td><td rowspan="2">T</td><td colspan="2">Baseline</td><td colspan="2">Hydra</td><td colspan="2">Local Window</td></tr><tr><td>GFLOPs</td><td>Attn</td><td>GFLOPs</td><td>Attn</td><td>GFLOPs</td><td>Attn</td></tr><tr><td>224</td><td>197</td><td>17.6</td><td>4.10%</td><td>16.8</td><td>0.02%</td><td>17.6</td><td>4.10%</td></tr><tr><td>384</td><td>577</td><td>55.1</td><td>11.13%</td><td>49.0</td><td>0.02%</td><td>51.1</td><td>4.10%</td></tr><tr><td>448</td><td>785</td><td>78.0</td><td>14.56%</td><td>66.7</td><td>0.02%</td><td>69.5</td><td>4.10%</td></tr><tr><td>1024</td><td>4097</td><td>657.3</td><td>47.06%</td><td>348.1</td><td>0.02%</td><td>362.8</td><td>4.10%</td></tr><tr><td>1280</td><td>6401</td><td>1298.9</td><td>58.14%</td><td>543.8</td><td>0.02%</td><td>566.9</td><td>4.10%</td></tr></table>

表 4：FLOP 数量与图像尺寸。ViT-B/16 模型在不同注意力方法下随图像尺寸增大的 FLOP 数量扩展情况。我们还列出了创建和应用注意力矩阵所占总计算量的比例。虽然 Hydra 注意力在大尺寸图像上显著改善了模型的 FLOP 数量，但局部窗口注意力同样如此，后者已被证明在大尺寸图像上有效 [19]。Hydra 注意力的一大局限是，它仅比局部窗口注意力快 4%，尽管它更具通用性，并且可以带来按比例更高的吞吐量。

更大尺寸图像 为了探究 Hydra 注意力在更多 token 数量下是否仍保持这些优势，在表 3 中，我们使用 [31] 中建议的超参数，在 384px 分辨率下对图 4 中的反向替换模型进行了 30 个 epoch 的微调。这得到的模型几乎拥有 3 倍数量的 token，这既应凸显 $O(TD)$ 与 $O ( T ^ { 2 } D )$ 注意力之间的差异，又应表明 Hydra 注意力的全局信息传播策略在这些更高 token 数量下是否有效。事实上，在表 3 中，我们看到了与 224px 图像相同的趋势：Hydra 注意力在使用 2 层时可以将准确率提高 0.59%、吞吐量提高 4.1%，而使用 7 层时可以保持准确率不变并将吞吐量提高 15.4%。

局限性 好的，但是 Hydra 注意力比标准注意力快 197 倍（当 T = 197 时），那么为什么最大 FLOP 减少量仅为 4%？结果表明，对于 $\mathrm { V i T { - } B / 1 6 ~ 2 2 4 ~ \times ~ 2 2 4 }$ 图像 $( T = 1 9 7 , D = 7 6 8 )$，模型总 FLOP 中仅有 4.10% 位于创建和应用注意力矩阵的部分。使用 Hydra 注意力后，这一比例降至 0.02%，基本上消除了模型中注意力的开销。虽然这确实带来了最高 10.2% 的原始吞吐量提升（见表 2），但我们可以做得更好。

当然，随着图像尺寸的增大，情况会发生变化：在表 4 中，我们针对不同图像尺寸重复了此计算，对于 1280px 图像，标准注意力的计算量激增至 58%，而 Hydra 注意力仍可忽略不计，仅为 0.02%。我们在表 3 中自行测试了 384px 图像，Hydra 注意力的加速效果更加显著（吞吐量最高提升 27.1%）。然而，仍需进一步工作来在更多 token 的任务（例如实例分割 [15]）上验证 Hydra 注意力。但在这些任务中，我们将与 ViTDet [19] 中使用的局部窗口注意力进行比较，后者已被证明对图像中的大 token 场景有效。与局部窗口注意力相比，Hydra 注意力在任何图像尺寸下仅少用 4% 的 FLOP，不过由于内存占用更少，其吞吐量可能会按比例更高。

总而言之，Hydra 注意力机制的优势在于其通用性。局部窗口注意力是密集图像预测的一种强力方案，但在 token 稀疏的情况下（例如掩码预训练 [14,12,30] 或 token 剪枝 [26,18,36]）会迅速变得笨拙。我们将这一方向留给未来的工作来探索。

## 5 结论与未来工作

在本文中，我们提出了 Hydra Attention，一种具有多头结构的高效注意力模块。我们在表 1 中展示了 Hydra Attention 优于其他 O(T D) 注意力方法，并且它甚至可以与传统多头自注意力协同工作，从而在图 4 中提升基线 DeiT-B 模型的精度。然而，尽管 Hydra Attention 在 ImageNet 分类任务上表现良好（表 2、表 3），其真正的加速潜力在于更大的图像（表 4）。

我们已经迈出了第一步，证明 Hydra Attention 是可行的，并希望未来的工作能够探索它在其他更加 token 密集的领域中的应用，例如检测、分割或视频。此外，Hydra Attention 是一种通用技术，对 token 之间的关系不作任何假设，因此可以应用于进一步提升 token 稀疏应用的速度，例如掩码预训练 [14,12,30] 或 token 剪枝 [26,18,36]。我们希望 Hydra Attention 能够作为迈向未来更强大、更高效、更通用的 Transformer 的一步。

致谢 我们要感谢 Alexander Kirillov、Christoph Feichtenhofer 和 Sean Bell 的许多深入讨论和宝贵意见！

## 参考文献

1. Abnar, S., Zuidema, W.: 《量化 Transformer 中的注意力流》. arXiv:2005.00928 [cs.LG] (2020) 9

2. Babiloni, F., Marras, I., Kokkinos, F., Deng, J., Chrysos, G., Zafeiriou, S.: 《Poly-NL：具有三阶多项式的线性复杂度非局部层》. 载于：ICCV (2021) 3, 4, 8, 12

3. Chen, B., Wang, R., Ming, D., Feng, X.: 《ViT-P：从局部性角度重新思考数据高效的视觉 Transformer》. arXiv:2203.02358 [cs.CV] (2022) 4

4. Chen, Y., Dai, X., Chen, D., Liu, M., Dong, X., Yuan, L., Liu, Z.: 《Mobile-Former：连接 MobileNet 与 Transformer》. 载于：CVPR (2022) 4

5. Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., Hawkins, P., Davis, J., Mohiuddin, A., Kaiser, L., 等：《与 Performers 一起重新思考注意力机制》. arXiv:2009.14794 [cs.LG] (2020) 2, 4, 5, 8

7. Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B.: 《Cityscapes 数据集：用于语义城市场景理解》. 载于：CVPR (2016) 2

8. Dao, T., Fu, D.Y., Ermon, S., Rudra, A., Ré, C.: 《FlashAttention：具有 IO 感知的快速且内存高效的精确注意力》. arXiv:2205.14135 [cs.LG] (2022) 2, 4, 8, 13

9. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: ImageNet：大规模层次化图像数据库. 发表于：CVPR（2009）2, 3, 4, 6, 8, 12

10. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., 等：一张图像相当于 16×16 个词：用于大规模图像识别的 Transformer. 发表于：ICLR（2020）1, 6

11. Fan, H., Xiong, B., Mangalam, K., Li, Y., Yan, Z., Malik, J., Feichtenhofer, C.: 多尺度视觉 Transformer. 发表于：ICCV（2021）1

12. Feichtenhofer, C., Fan, H., Li, Y., He, K.: 作为时空学习器的掩码自编码器. arXiv:2205.09113 [cs.CV]（2022）14

13. Graham, B., El-Nouby, A., Touvron, H., Stock, P., Joulin, A., Jégou, H., Douze, M.: LeViT：穿着 ConvNet 外衣以实现更快推理的视觉 Transformer. 发表于：ICCV（2021）1, 4

14. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: 掩码自编码器是可扩展的视觉学习器. 发表于：CVPR（2022）14

15. He, K., Gkioxari, G., Dollár, P., Girshick, R.: Mask R-CNN. 发表于：ICCV（2017）14

16. Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F.: Transformer 就是 RNN：具有线性注意力的快速自回归 Transformer. 发表于：ICML（2020）2, 4, 5, 8, 9

17. Kenton, J.D.M.W.C., Toutanova, L.K.: BERT：面向语言理解的深度双向 Transformer 预训练. 发表于：NAACL-HLT（2019）1, 6

18. Kong, Z., Dong, P., Ma, X., Meng, X., Niu, W., Sun, M., Ren, B., Qin, M., Tang, H., Wang, Y.: SPViT：通过软 token 剪枝实现更快的视觉 Transformer. arXiv:2112.13890 [cs.CV]（2021）14

19. Li, Y., Mao, H., Girshick, R., He, K.: 探索用于目标检测的纯视觉 Transformer 主干网络. arXiv:2203.16527 [cs.CV]（2022）4, 13, 14

20. Li, Y., Wu, C.Y., Fan, H., Mangalam, K., Xiong, B., Malik, J., Feichtenhofer, C.: MViTv2：用于分类与检测的改进多尺度视觉 Transformer. 发表于：CVPR（2022）1

21. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin Transformer：使用移位窗口的层次化视觉 Transformer. 发表于：ICCV（2021）1, 4

22. Mehta, S., Rastegari, M.: MobileViT：轻量级、通用且对移动端友好的视觉 Transformer. arXiv:2110.02178 [cs.CV]（2021）4

23. Michel, P., Levy, O., Neubig, G.: 十六个注意力头真的比一个更好吗？发表于：NeurIPS（2019）5

24. Peng, H., Pappas, N., Yogatama, D., Schwartz, R., Smith, N.A., Kong, L.: 随机特征注意力. arXiv:2103.02143 [cs.CL]（2021）2, 4, 5

25. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., 等：语言模型是无监督多任务学习器. OpenAI 博客（2019）1

26. Rao, Y., Zhao, W., Liu, B., Lu, J., Zhou, J., Hsieh, C.J.: DynamicViT：通过动态 token 稀疏化实现高效的视觉 Transformer. 发表于：NeurIPS（2021）14

27. Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.: Grad-CAM：通过基于梯度的定位从深度网络生成视觉解释. 发表于：ICCV（2017）10

28. Shen, Z., Zhang, M., Zhao, H., Yi, S., Li, H.: 高效注意力：具有线性复杂度的注意力（Eficient attention: Attention with linear complexities）. 载于：WACV（2021） 2, 4, 5

29. Song, J.g.: Ufo-vit：无 Softmax 的高性能线性视觉 Transformer（Ufo-vit: High performance linear vision transformer without softmax）. arXiv:2109.14382 [cs.CV]（2021） 11

30. Tong, Z., Song, Y., Wang, J., Wang, L.: VideoMAE：掩码自编码器是自监督视频预训练的数据高效学习器（VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training）. arXiv:2203.12602 [cs.CV]（2022） 14

31. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., Jégou, H.: 通过注意力进行训练数据高效的图像 Transformer 与蒸馏（Training data-efficient image transformers & distillation through attention）. arXiv:2012.12877 [cs.CV]（2020） 3, 6, 8, 12, 13

32. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need（注意力即你所需要的一切）. NeurIPS（2017） 1, 2, 4, 5, 6, 11, 12

33. Voita, E., Talbot, D., Moiseev, F., Sennrich, R., Titov, I.: 分析多头自注意力：专门的头部承担主要工作，其余可以被剪枝（Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned）. arXiv:1905.09418 [cs.CL]（2019） 5

34. Wang, S., Li, B.Z., Khabsa, M., Fang, H., Ma, H.: Linformer：具有线性复杂度的自注意力（Linformer: Self-attention with linear complexity）. arXiv:2006.04768 [cs.LG]（2020） 2, 4, 8

35. Yang, C., Wang, Y., Zhang, J., Zhang, H., Wei, Z., Lin, Z., Yuille, A.: 具有增强自注意力的轻量视觉 Transformer（Lite vision transformer with enhanced self-attention）. 载于：CVPR（2022） 4

36. Yin, H., Vahdat, A., Alvarez, J., Mallya, A., Kautz, J., Molchanov, P.: AdaViT：用于高效视觉 Transformer 的自适应 Token（AdaViT: Adaptive tokens for efficient vision transformer）. CVPR（2022） 14

37. Zhai, S., Talbott, W., Srivastava, N., Huang, C., Goh, H., Zhang, R., Susskind, J.: 无注意力的 Transformer（An attention free transformer）. arXiv:2105.14103 [cs.LG]（2021） 3, 4, 8, 9, 11, 12

38. Zhai, X., Kolesnikov, A., Houlsby, N., Beyer, L.: 视觉 Transformer 的规模化（Scaling vision transformers）. arXiv:2106.04560 [cs.CV]（2021） 2, 5, 6

## A 其他核函数

在表 5 中，我们列出了我们在 Hydra Attention 中尝试过的所有核函数（即 $\phi ( \cdot )$ 的实例化）。在选择这些核函数时，我们有三个主要的考量。即 $\phi ( \cdot )$ 是否应当：1.）是无界的，2.）允许负值，或 3.）是线性的。

例如，用于余弦相似度的 L2 归一化（即我们在论文中所使用的）是有界的，允许负值，并且是线性的。另一方面，Sigmoid 是有界的，只取正值，并且是非线性的。

从我们的实验中我们观察到，虽然用于 Q 的函数影响并不大，但 K 则显著受益于同时具备线性和允许负值的特性。与 L2 归一化相比，对 K 进行 softmax 处理会显著降低性能。而在各种归一化技术中，我们发现 L2 效果最佳（优于 L1 或常数归一化）。

最后，由于 ϕ(K)（在不经过 softmax 时）之和并不为 1，将其与 V 相乘可能产生比标准注意力更大的数值量级。因此，我们认为对注意力层的输出进行归一化可能是有用的。我们在 $^ { 6 6 } + \mathrm { L N } ^ { 5 }$ 上测试了两种核函数，其中注意力操作之后接一个层归一化（位于投影之前）。然而，这似乎并没有带来帮助，因此这里似乎更好的做法是保持核函数不进行归一化。

## B 更多可视化

在图 5 中，我们展示了更多的图像可视化结果，以补充正文中的内容。这些图像是从 ImageNet-1k 验证集的 12 个不同类别（每类 4 张）中随机选取的，唯一的选择标准是图像内容适宜观看。该网络对其中大多数图像都给出了正确的预测结果。

<table><tr><td>Kernel</td><td> $\phi(Q)$ </td><td> $\phi(K)$ </td><td>Accuracy</td></tr><tr><td>Cosine Similarity</td><td colspan="2"> $x/||x||_2$ </td><td>76.37</td></tr><tr><td>Tanh-L2</td><td> $\tanh(x)$ </td><td> $x/||x||_2$ </td><td>76.17</td></tr><tr><td>Mean</td><td colspan="2"> $x/\sqrt{T}$ </td><td>75.95</td></tr><tr><td>CosSim + LN</td><td colspan="2"> $x/||x||_2$ </td><td>75.74</td></tr><tr><td>Tanh-L2 + LN</td><td> $\tanh(x)$ </td><td> $x/||x||_2$ </td><td>75.22</td></tr><tr><td>Tanh-Softmax</td><td> $\tanh(x)$ </td><td> $softmax(x)$ </td><td>74.18</td></tr><tr><td>Sigmoid-Softmax</td><td> $\sigma(x)$ </td><td> $softmax(x)$ </td><td>74.02</td></tr><tr><td>L1 Normalization</td><td colspan="2"> $x/||x||_1$ </td><td>70.75</td></tr></table>

表 5：更多核函数（More Kernels）。我们在此列出了测试过的其他核函数。由于大多数核函数都是非对称的，我们分别列出 Q 和 K 对应的 ϕ。

![](images/633bf7117df146033fe4934dd153fcf1f9ba1cc9ba7cda921a6071af97cb1d86.jpg)  

图 5：更多可视化结果。使用本文所述方法对模型关注区域的更多可视化结果。

## 代码

Hydra 注意力机制的实现极为简单。我们在此给出基于余弦相似度核的 PyTorch 参考实现：

```python
def hydra(q, k, v):
    """
    q, k, and v should all be tensors of shape
    [batch, tokens, features]
    """
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)

    kv = (k * v).sum(dim=-2, keepdim=True)
    out = q * kv

    return out
```