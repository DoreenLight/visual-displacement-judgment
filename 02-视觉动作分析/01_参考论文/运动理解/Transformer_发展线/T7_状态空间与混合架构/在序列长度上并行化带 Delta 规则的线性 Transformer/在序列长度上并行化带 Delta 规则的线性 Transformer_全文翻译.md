# 在序列长度上并行化带 Delta 规则的线性 Transformer

Songlin Yang<sup>⋄</sup> Bailin Wang<sup>⋄</sup> Yu Zhang<sup>†</sup> Yikang Shen<sup>‡</sup> Yoon Kim<sup>⋄</sup>

<sup>⋄</sup>麻省理工学院 (Massachusetts Institute of Technology) <sup>†</sup>苏州大学 (Soochow University) <sup>‡</sup>MIT-IBM Watson AI Lab

yangsl66@mit.edu

## 摘要

具有线性注意力机制的 Transformer（即线性 Transformer）和状态空间模型（state-space models）最近被提出作为具有 Softmax 注意力机制的 Transformer 的可行线性时间替代方案。然而，这些模型的表现仍然不如 Transformer，特别是在需要上下文检索（in-context retrieval）的任务上。尽管通过用 Delta 规则替代线性 Transformer 中的加法更新这一更具表达力的变体 [DeltaNet; 101] 已被发现在联想回忆（associative recall）上更加有效，但现存用于训练此类模型的算法无法在序列长度上进行并行化，因此在现代硬件上的训练效率低下。本工作提出了一种用于训练带 Delta 规则的线性 Transformer 的硬件高效算法，该算法利用了一种用于计算 Householder 矩阵乘积的内存高效表示 [11]。该算法使我们能够将 DeltaNet 扩展到标准语言建模设置中。我们在 100B token 上训练了一个 1.3B 参数的模型，并发现它在困惑度（perplexity）和下游任务的零样本（zero-shot）性能方面优于最近的线性时间基准模型，如 Mamba [31] 和 GLA [124]。我们还对两种混合模型进行了实验，这些模型将 DeltaNet 层与 (1) 每隔一层交替使用的滑动窗口注意力（sliding-window attention）层或 (2) 两个全局注意力（global attention）层相结合，并发现这些混合模型的表现优于强劲的 Transformer 基准。

## 1 引言

注意力机制 [8, 116] 已被证明是实现精确序列建模的重要原语。此外，注意力机制在训练期间非常高效，因为它包含大量的矩阵乘法，因而能够充分利用现代 GPU 上的高度并行处理能力和专用加速器。然而，注意力机制的复杂度与序列长度呈二次方关系，因此它从根本上说是一种计算昂贵（expensive）的原语。尽管最近的技术通过对中间计算进行硬件感知的重构，使得注意力机制能够扩展到更长的序列 [20, 18, 59, 14]，但这些方法仍然需要存储先前元素的键/值（key/value）向量，而对于长序列而言，管理这一大小呈线性增长的“KV 缓存”（KV cache）可能会非常棘手。

线性注意力 Transformer [48] 用（可能经过变换的）键向量和查询向量之间的点积取代了 softmax 注意力中的指数核。这使得将线性注意力表示为具有矩阵值隐藏状态的线性 RNN 成为可能，从而消除了对 KV 缓存的需求，并实现了常数显存推理。尽管线性注意力的早期变体在语言建模上的表现普遍逊于 softmax 注意力，但最近的研究表明，结合了数据依赖型门控因子的线性注意力门控变体能够与强大的 Transformer 基线相媲美 [124, 92, 9, 79]。这些门控线性 Transformer，连同诸如 Mamba [31, 19] 之类的时变状态空间模型（其可被重参数化为门控线性 Transformer [124]），已被提出作为普通 Transformer 的潜在替代方案。然而，尽管在语言建模方面具有极具竞争力的性能，但这些模型在召回密集型任务上的表现被证明逊于 Transformer [6, 7]，而这类能力对于许多受关注的实际下游任务（例如检索增强生成 [53]）至关重要。

为了增强长上下文中的联想召回（associative recall），Schlag 等人 [101] 提出了 DeltaNet，这是一种线性 Transformer 变体，它使用类似 delta 规则的更新 [121] 来检索并更新与当前键关联的值向量。DeltaNet 被发现在合成任务和小规模语言建模/机器翻译中非常有效。然而，最初的工作采用的是无法沿序列长度并行化的串行算法，从而导致硬件训练效率低下，并且目前尚不清楚如何将 DeltaNet 扩展到更大的模型和数据集。

本文描述了一种针对 DeltaNet 的硬件高效训练算法，该算法在序列长度维度上并行化前向/反向传播过程。我们将 DeltaNet 重新参数化为一个矩阵值循环神经网络（RNN），其循环关系由广义豪斯霍尔德变换（generalized Householder transformation）给出。这种重新参数化使得针对豪斯霍尔德矩阵乘积使用紧凑 WY 表示（compact WY representation）[11] 成为可能，从而避免了在并行训练期间在每个时间步实例化矩阵大小的隐藏状态，否则会导致高昂的 I/O 开销。这种内存高效的表示使得将用于训练线性注意力模型的分块并行策略（chunkwise parallel strategy）[34, 108, 124] 直接扩展到 DeltaNet 成为可能。我们将 DeltaNet 扩展到中等规模的语言建模基准测试（在 100B token 上训练的 1.3B 模型），发现 DeltaNet 比诸如 Mamba [31] 和 GLA [124] 等强线性循环模型取得了更好的语言建模和零样本（zero-shot）下游任务性能。在上下文检索和学习评估方面，我们在合成和真实基准测试 [4, 2, 84, 6] 上对 DeltaNet 进行了评估，再次发现其相比线性循环基线表现优异。最后，我们尝试了一种混合方法，将 DeltaNet 层与滑动窗口注意力（sliding attention）层或全局注意力（global attention）层相结合，并发现这些混合模型不仅优于普通 Transformer，也优于纯 DeltaNet Transformer。

## 2 背景

## 2.1 线性 Transformer：具有线性注意力的 Transformer

给定一个 $d$ 维输入向量序列 $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { L }$，Transformer 使用 softmax 注意力机制来关注过去的全部信息，

$$
\pmb {q} _ {t}, \pmb {k} _ {t}, \pmb {v} _ {t} = \pmb {W} _ {Q} \pmb {x} _ {t}, \pmb {W} _ {K} \pmb {x} _ {t}, \pmb {W} _ {V} \pmb {x} _ {t},
$$

$$
\pmb {o} _ {t} = \sum_ {i = 1} ^ {t} \frac {\exp (\pmb {k} _ {i} ^ {\top} \pmb {q} _ {t})}{\sum_ {j = 1} ^ {t} \exp (\pmb {k} _ {j} ^ {\top} \pmb {q} _ {t})} \pmb {v} _ {i},
$$

其中 $\mathbf { W } _ { Q } , \mathbf { W } _ { K } , \mathbf { W } _ { V } \in \mathbb { R } ^ { d \times d } , \mathbf { q } _ { t } , \mathbf { k } _ { t } , \mathbf { v } _ { t } , \mathbf { o } _ { t } \in \mathbb { R } ^ { d }$。（为简便起见，此处假设为单注意力头）。线性注意力 [48] 将指数核函数 $\exp ( { k _ { i } ^ { \intercal } { q _ { t } } } )$ 替换为点积 $\phi { ( k _ { i } ) } ^ { \mathsf { T } } \phi ( \pmb q _ { t } )$，其中 $\phi : \mathbb { R } ^ { d }  \mathbb { R } ^ { n }$ 是一个特征映射。这使得重新排列计算从而将线性注意力表示为具有矩阵值隐藏状态的线性 RNN 成为可能，

$$
\boldsymbol {o} _ {t} = \sum_ {i = 1} ^ {t} \frac {\phi (\boldsymbol {k} _ {i}) ^ {\top} \phi (\boldsymbol {q} _ {t})}{\sum_ {j = 1} ^ {t} \phi (\boldsymbol {k} _ {j}) ^ {\top} \phi (\boldsymbol {q} _ {t})} \boldsymbol {v} _ {i} = \frac {\left(\sum_ {i = 1} ^ {t} \boldsymbol {v} _ {i} \phi (\boldsymbol {k} _ {i}) ^ {\top}\right) \phi (\boldsymbol {q} _ {t})}{\left(\sum_ {j = 1} ^ {t} \phi (\boldsymbol {k} _ {j}) ^ {\top}\right) \phi (\boldsymbol {q} _ {t})} = \frac {\mathbf {S} _ {t} \phi (\boldsymbol {q} _ {t})}{\mathbf {z} _ {t} ^ {\top} \phi (\boldsymbol {q} _ {t})},
$$

其中 $\begin{array} { r } { \mathbf { S } _ { t } = \sum _ { i = 1 } ^ { t } { \pmb { v } } _ { i } \phi ( { \pmb { k } } _ { i } ) ^ { \top } \in \mathbb { R } ^ { d \times n } } \end{array}$ 且 $\begin{array} { r } { \mathbf { z } _ { t } = \sum _ { i = 1 } ^ { t } \phi ( \pmb { k } _ { i } ) \in \mathbb { R } ^ { n } } \end{array}$。如果我们允许 $n$ 趋向于无穷大，线性注意力（linear attention）可以使用与多项式核相关的特征映射，通过点积计算指数核的多项式近似，从而能够任意好地逼近 softmax 注意力 [6]。分母 $\mathbf { z } _ { t } ^ { \top } \phi ( \pmb q _ { t } ) \in \mathbb { R }$ 可能会导致数值不稳定 [86]，在近期的工作中已被移除 [101, 63]。对 $\phi$ 采用恒等映射也很常见 [63, 108]，这会得到如下简化形式的线性 Transformer：$\mathbf { S } _ { t } = \mathbf { S } _ { t - 1 } + v _ { t } k _ { t } ^ { \top } , o _ { t } = \mathbf { S } _ { t } \mathbf { q } _ { t }$

高效训练。令 $\mathbf { Q } , \mathbf { K } , \mathbf { V } \in \mathbb { R } ^ { L \times d }$ 分别为堆叠后的查询（query）、键（key）、值（value）向量，例如 $\mathbf { Q } _ { i } =$ $\pmb q _ { i }$。随后我们可以通过 $\mathbf { O } = \left( \mathbf { Q } \mathbf { K } ^ { \mathsf { T } } \odot \mathbf { M } _ { L } \right) \mathbf { V } .$ 并行计算输出 $\mathbf { O } \in \mathbb { R } ^ { L \times d }$，其中 $\mathbf { M } _ { L } \in \mathbb { R } ^ { L \times L }$ 为因果掩码。这种完全的“并行形式”与上述“循环形式”具有不同的 FLOPs 和并行化权衡。并行形式的计算复杂度为 $O ( L ^ { 2 } d + L d ^ { 2 } )$，因此比循环形式需要更多的 FLOPs（循环形式的计算复杂度为 $O ( L d ^ { 2 } )$）。然而，在实际应用中，对于中等长度的序列，并行形式通常要快得多，因为它可以以 $O ( 1 )$ 步完成。这种序列级并行性还实现了高 GPU 占用率。循环形式虽然需要更少的

FLOPs，但无法在序列长度 $\mathrm { l e n g t h } ^ { \mathrm { 1 } }$ 上进行并行化，此外循环中涉及的逐元素操作也无法利用专门的矩阵乘法加速器（例如 Tensor Core）。

分块并行形式（Chunkwise parallel form）。分块并行形式 [34, 108, 124] 在并行形式与循环形式之间取得了平衡，相较于并行形式所需的 FLOPs 更少，且比循环形式具有更高的序列级并行度。具体而言，假设查询（query）/键（key）/值（value）向量被分割为 $\begin{array} { c } { { \frac { \texttt { L } } { C } } } \end{array}$ 个块，其中每个块的长度为 $C .$ 令 $\dot { { \bf Q } } _ { [ t ] } \in \mathbb { R } ^ { \dot { C } \times d }$ 为块 t 的所有查询向量，并令 ${ \pmb q } _ { [ t ] } ^ { i } = { \pmb q } _ { t C + i }$ 为第 t 个块内的第 i 个查询向量；键/值块的定义类似。注意 $t \in [ 0 , L / C ) , i \in [ 1 , C ]$ 。状态矩阵也进行了重新索引，使得 ${ \bf S } _ { [ t ] } ^ { i } = { \bf S } _ { t C + i }$ ，并且我们额外定义 ${ \bf S } _ { [ t ] } ^ { 0 ^ { \scriptstyle * } } = { \bf \dot { S } } _ { [ t - 1 ] } ^ { C }$ ，即一个块的初始状态是前一个块的最终状态。由此我们可以得到第 t 个块内第 r 个元素的隐藏状态和输出向量的如下恒等式：

$$
\mathbf {S} _ {[ t ]} ^ {r} = \mathbf {S} _ {[ t ]} ^ {0} + \sum_ {i = 1} ^ {r} \boldsymbol {v} _ {[ t ]} ^ {i} \boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}}, \qquad \boldsymbol {o} _ {[ t ]} ^ {r} = \mathbf {S} _ {[ t ]} ^ {0} \boldsymbol {q} _ {[ t ]} ^ {r} + \sum_ {i = 1} ^ {r} \boldsymbol {v} _ {[ t ]} ^ {i} \left(\boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}} \boldsymbol {q} _ {[ t ]} ^ {r}\right).
$$

通过基于并行形式进一步重写块内计算，我们得到如下公式：

$$
\mathbf {S} _ {[ t + 1 ]} = \mathbf {S} _ {[ t ]} + \mathbf {V} _ {[ t ]} ^ {\top} \mathbf {K} _ {[ t ]}
$$

$$
\in \mathbb {R} ^ {d \times d},\tag{1}
$$

$$
\mathbf {O} _ {[ t ]} = \mathbf {Q} _ {[ t ]} \mathbf {S} _ {[ t ]} ^ {\top} + \left(\mathbf {Q} _ {[ t ]} \mathbf {K} _ {[ t ]} ^ {\top} \odot \mathbf {M} _ {C}\right) \mathbf {V} _ {[ t ]}
$$

$$
\in \mathbb {R} ^ {C \times d}\tag{2}
$$

其中我们令 ${ \mathbf S } _ { [ t ] } = { \mathbf S } _ { [ t ] } ^ { 0 }$ 以简化符号表示。在这种“分块并行形式”下，信息通过 ${ \bf S } _ { [ t ] }$ 在块与块之间传递，而 $i \in [ 1 , C ]$ 的块内状态 ${ \bf S } _ { [ t ] } ^ { i }$ 无需实体化，从而节省了内存。

分块并行形式的复杂度为 $O ( L C d + L d ^ { 2 } )$，且步数（不含块级并行扫描）为 $O ( \textstyle { \frac { L } { T } } )$。因此，$C = L$ 可还原为全并行形式，而 $C = 1$ 则退化为循环形式。分块并行形式允许我们在两种形式之间进行插值权衡，其本质是在顺序计算步数与序列级并行度之间做出取舍。在实践中，$C$ 通常被设为一个较小的常数（通常为 64 或 128），从而实现次二次复杂度的训练。正如 FLASHLINEARATTENTION [124, 123] 所展示的，即使在中等长度的序列上，这种分块形式也能带来相比仅采用并行形式的 Softmax 注意力切实可行的实际加速。

## 2.2 DeltaNet：基于 Delta 更新规则的线性 Transformer

上述线性 Transformer 采用了一种简单的线性循环：$\mathbf { S } _ { t } = \mathbf { S } _ { t - 1 } + v _ { t } \mathbf { } \mathbf { } k _ { t } ^ { \mathsf { T } }$。这可以被视为在每个时间步使用新的键-值关联对记忆 $\mathbf { S } _ { t - 1 }$ 进行加法更新。然而，正如 Schlag et al. [101] 所指出的，纯加法的更新规则使得释放过去的键-值关联变得十分困难，当 $L > d ,$ 时最终会导致键“冲突”。理想情况下，模型应该学会移除不太重要的键-值关联以为新的关联腾出空间，并且这种移除应当取决于新键与记忆内容之间的交互。

从快速权重编程（fast weight programming）[102] 的视角来看，线性注意力的循环隐藏状态是快速权重，它通过类 Hessian 更新规则将输入 $\pmb q _ { t }$ 映射到输出 $\mathbf { } _ { o _ { t } }$，但其记忆容量有限 [68]。另一方面，DeltaNet 使用 delta 更新规则或 Widrow-Hoff 学习规则 [121] 进行快速权重更新：$\mathbf { S } _ { t } = \mathbf { S } _ { t - 1 } - \beta _ { t } \big ( \mathbf { S } _ { t - 1 } k _ { t } - \bar { v } _ { t } \big ) k _ { t } ^ { \top }$，其中 $\beta _ { t }$ 为学习率，$\mathbf { S } _ { t - 1 } \mathbf { \delta } _ { k _ { t } }$ 表示当前预测，${ \mathbf { } } v _ { t }$ 为目标值。该方法的命名源于其基于预测值 $\mathbf { S } _ { t - 1 } \mathbf { \delta } _ { k _ { t } }$ 与目标值 ${ \mathbf { } } v _ { t } .$ 之间的“delta”（差值）来更新权重的核心原理，该机制已被证明具有更好的记忆容量 [28, 85, 56, 101]。这一过程也可以被视为使用单步 SGD 优化在线回归损失，<sup>2</sup>

$$
\mathcal {L} _ {t} (\mathbf {S}) = \frac {1}{2} \| \mathbf {S} \pmb {k} _ {t} - \pmb {v} _ {t} \| ^ {2}, \quad \mathbf {S} _ {t} = \mathbf {S} _ {t - 1} - \beta_ {t} \nabla_ {\mathbf {S} _ {t - 1}} \mathcal {L} _ {t} (\mathbf {S} _ {t - 1}) = \mathbf {S} _ {t - 1} - \beta_ {t} (\mathbf {S} _ {t - 1} \pmb {k} _ {t} - \pmb {v} _ {t}) \pmb {k} _ {t} ^ {\top},
$$

这在最近的几项工作中已有讨论 [110, 58, 125, 10]。相比之下，线性 Transformer

可以看作是在优化一个在线线性（负内积）损失 $\mathcal { L } _ { t } = - \langle \mathbf { S } \pmb { k } _ { t } , \pmb { v } _ { t } \rangle . ^ { 3 }$

对 DeltaNet 的另一种解释是从键值检索的角度来看的。它首先使用当前键检索旧值，$v _ { t } ^ { \mathrm { o l d } } = \mathbf { S } _ { t - 1 } \bar { k _ { t } }$。然后，它通过在旧值与当前值 ${ \mathbf { } } v _ { t } .$ 之间进行插值来获得新值 ${ \boldsymbol { v } } _ { t } ^ { \mathrm { n e w } }$，该新值在记忆中替换 ${ \mathbf { } } { \mathbf { } } v _ { t } ^ { \mathrm { { o l d } } }$：

$$
\boldsymbol {v} _ {t} ^ {\text { new }} = \beta_ {t} \boldsymbol {v} _ {t} + (1 - \beta_ {t})   \boldsymbol {v} _ {t} ^ {\text { old }}, \quad \mathbf {S} _ {t} = \mathbf {S} _ {t - 1} \underbrace {- \boldsymbol {v} _ {t} ^ {\text { old }} \boldsymbol {k} _ {t} ^ {\top}} _ {\text { remove }} \underbrace {+ \boldsymbol {v} _ {t} ^ {\text { new }} \boldsymbol {k} _ {t} ^ {\top}} _ {\text { write }}
$$

这里 $\beta _ { t } = \sigma ( \mathbf { W } _ { \beta } \pmb { x } _ { t } ) \in ( 0 , 1 )$ 是一个软“写入强度”：当 $\beta _ { t } = 1$ 时，旧值被完全移除且 $\begin{array} { r } { \pmb { v } _ { t } ^ { \mathrm { n e w } } = \pmb { v } _ { t } ; } \end{array}$；当 $\beta _ { t } = 0$ 时，记忆保持不变，我们有 $\mathbf { S } _ { t } = \mathbf { S } _ { t - 1 }$。输出计算与标准线性注意力相同，即 $o _ { t } = \mathbf { S } _ { t } \pmb { q } _ { t }$。该循环形式的复杂度与标准线性注意力相同，即 $\mathrm { i } . \mathrm { e } . , { \mathcal { O } } ( L d ^ { 2 } )$。

Schlag 等人 [101] 证明了 DeltaNet 在小规模语言建模和合成上下文检索任务上优于普通线性 Transformer。然而，正如 [124, §3.2] 中所讨论的，他们的训练算法（线性 Transformer 内存高效循环实现 [48, §3.3.1] 的一种扩展）是严格顺序执行的，因此在硬件上效率低下，这促使我们推导出一个等价的分块算法以在大规模下训练 DeltaNet，如下文所介绍。

## 3 沿序列维度并行化 DeltaNet

## 3.1 一种内存高效的重参数化

我们首先观察到，对于 $u _ { i } , k _ { i } \in \mathbb { R } ^ { d }$，$\mathbf { S } _ { t }$ 具有形如 $\begin{array} { r } { \mathbf { S } _ { t } = \sum _ { i = 1 } ^ { t } \mathbf { { u } } _ { i } \mathbf { { k } } _ { i } ^ { \top } } \end{array}$ 的纯加性表示，因为我们只需令 $\pmb { u } _ { i } = \pmb { v } _ { i } ^ { \mathrm { n e w } } - \pmb { v } _ { i } ^ { \mathrm { o l d } } = \beta _ { i } ( \pmb { v } _ { i } - \pmb { v } _ { i } ^ { \mathrm { o l d } } )$。回顾 $\ S 2 . 1$ 可知，简单线性注意力（linear attention）的形式为 $\begin{array} { r } { \mathbf { S } _ { t } = \sum _ { i = 1 } ^ { t } v _ { i } \pmb { k } _ { i } ^ { \top } } \end{array}$。因此，DeltaNet 仅仅是将线性注意力中的值向量 ${ \mathbf { } } v _ { i }$ 替换为了“伪”值向量 $\mathbf { \Delta } \mathbf { u } _ { i }$。一旦构建好 $\mathbf { \chi } _ { \mathbf { \chi } _ { i } ^ { \prime } \mathbf { \tilde { s } } }$，后续计算就可以像普通线性注意力一样进行，即 $\mathbf { O } = \left( \mathbf { Q } \mathbf { K } ^ { \mathsf { T } } \odot \mathbf { M } \right)$ U，其中 $\mathbf { U } \in \mathbb { R } ^ { L \times d }$ 是 $\mathbf { \Delta } \mathbf { u } _ { i }$ 向量按行拼接而成的矩阵。

然而，朴素地计算 $\mathbf { \Delta } \mathbf { u } _ { t }$ 需要显式实例化 $\mathbf { S } _ { t - 1 }$ 以计算 $\pmb { v } _ { t } ^ { \mathrm { o l d } }$，这将需要 $\mathcal { O } ( d ^ { 2 } )$ ) 的显存/内存。我们现在展示，可以在 $\mathcal O ( d )$ 显存下获得 $\mathbf { \chi } _ { \mathbf { \chi } _ { t } } \mathbf { \chi } _ { \mathbf { \chi } }$ s，而无需显式实例化 $\mathbf { S } _ { t - 1 }$。我们简单的归纳证明依赖于 Householder 矩阵乘积的 WY 表示 [11] 的应用。基础情况显而易见，因为我们有 $\mathbf { S } _ { 1 } = \beta _ { 1 } \pmb { v } _ { 1 } \pmb { k } _ { 1 } ^ { \top }$，所以 $\pmb { u } _ { 1 } = \beta _ { 1 } \pmb { v } _ { 1 }$。对于归纳步骤，我们首先观察到 DeltaNet 的更新形式为：

$$
\mathbf {S} _ {t} = \mathbf {S} _ {t - 1} - \boldsymbol {v} _ {t} ^ {\text {old}} \boldsymbol {k} _ {t} ^ {\top} + \boldsymbol {v} _ {t} ^ {\text {new}} \boldsymbol {k} _ {t} ^ {\top} = \mathbf {S} _ {t - 1} - \beta_ {t} (\mathbf {S} _ {t - 1} \boldsymbol {k} _ {t}) \boldsymbol {k} _ {t} ^ {\top} + \beta_ {t} \boldsymbol {v} _ {t} \boldsymbol {k} _ {t} ^ {\top} = \mathbf {S} _ {t - 1} (\mathbf {I} - \beta_ {t} \boldsymbol {k} _ {t} \boldsymbol {k} _ {t} ^ {\top}) + \beta_ {t} \boldsymbol {v} _ {t} \boldsymbol {k} _ {t} ^ {\top},
$$

这可以看作是对前一个状态应用了广义 Householder 变换 $( \mathrm { i . e . }$ .，即与单位矩阵加秩一矩阵进行矩阵乘法）。随后归纳步骤如下给出：

$$
\mathbf {S} _ {t} = \mathbf {S} _ {t - 1} (\mathbf {I} - \beta_ {t} \boldsymbol {k} _ {t} \boldsymbol {k} _ {t} ^ {\top}) + \beta_ {t} \boldsymbol {v} _ {t} \boldsymbol {k} _ {t} ^ {\top} = \sum_ {i = 1} ^ {t - 1} \boldsymbol {u} _ {i} \boldsymbol {k} _ {i} ^ {\top} + \underbrace {\beta_ {t} \left(\boldsymbol {v} _ {t} - \sum_ {i = 1} ^ {t - 1} \boldsymbol {u} _ {i} \left(\boldsymbol {k} _ {i} ^ {\top} \boldsymbol {k} _ {t}\right)\right)} _ {\text { defined   as } \boldsymbol {u} _ {t}} \boldsymbol {k} _ {t} ^ {\top} = \sum_ {i = 1} ^ {t} \boldsymbol {u} _ {i} \boldsymbol {k} _ {i} ^ {\top}\tag{3}
$$

注意到 $\mathbf { \Delta } \mathbf { u } _ { t }$ 不需要实体化任何隐藏状态，且计算需要 $\mathcal O ( d )$ 的内存，从而完成了证明。尽管我们避免了实体化 $\mathbf { S } _ { t } \mathbf { \bar { s } } .$ ，但计算所有 $L$ 的 $\mathbf { \Delta } \mathbf { \em u } _ { t } \mathrm { : \partial }$ （即 U）需要 $\bar { \mathcal { O } ( L ^ { 2 } d ) }$ ，而且不能完全并行化，这与线性注意力机制（linear attention）不同——在线性注意力机制中，我们可以在 $\mathcal { O } ( 1 )$ 步内并行计算所有的值向量 V。我们现在表明，上述技巧仍然可以为 DeltaNet 实现一种高效的分块并行形式。

## 3.2 DeltaNet 的分块并行形式

为了推导分块并行形式，我们首先展开递推关系，

$$
\mathbf {S} _ {t} = \mathbf {S} _ {t - 1} (\mathbf {I} - \beta_ {t} \boldsymbol {k} _ {t} \boldsymbol {k} _ {t} ^ {\top}) + \beta_ {t} \boldsymbol {v} _ {t} \boldsymbol {k} _ {t} ^ {\top} = \sum_ {i = 1} ^ {t} \beta_ {i} (\boldsymbol {v} _ {i} \boldsymbol {k} _ {i} ^ {\top}) \left(\prod_ {j = i + 1} ^ {t} (\mathbf {I} - \beta_ {j} \boldsymbol {k} _ {j} \boldsymbol {k} _ {j} ^ {\top})\right).\tag{4}
$$

然后我们定义以下变量：$\begin{array} { r l r } { \mathbf { P } _ { i } ^ { j } } & { = } & { \prod _ { t = i } ^ { j } ( \mathbf { I } \ - \ \beta _ { t } \pmb { k } _ { t } \pmb { k } _ { t } ^ { \top } ) \in \mathbb { R } ^ { d \times d } , \mathbf { H } _ { i } ^ { j } = } \end{array}$ $\begin{array} { r } { \sum _ { t = i } ^ { j } \beta _ { t } ( \pmb { v } _ { t } \pmb { k } _ { t } ^ { \top } ) \mathbf { P } _ { t + 1 } ^ { j } \in \mathbb { R } ^ { d \times d } } \end{array}$ ，其中当 $i > j$ 时，我们令 $\mathbf { P } _ { i } ^ { j } = \mathbf { I }$ 。直观上，$\mathbf { P } _ { i } ^ { j }$ 是应用于 $\mathbf { S } _ { i }$ 以获得 $\mathbf { S } _ { j }$ 的“衰减因子”，而 $\mathbf { H } _ { i } ^ { j }$ 表示从标记 i 开始对 $\mathbf { S } _ { j }$ 的贡献。（因此 $\mathbf { S } _ { t } = \mathbf { H } _ { 1 } ^ { t } ,$ ）。分块递推关系随后可以写作，

$$
\mathbf {S} _ {[ t ]} ^ {r} = \mathbf {S} _ {[ t ]} ^ {0} \mathbf {P} _ {[ t ]} ^ {r} + \mathbf {H} _ {[ t ]} ^ {r}\tag{5}
$$

其中我们定义了分块变量 $\mathbf { S } _ { [ t ] } ^ { i } = \mathbf { S } _ { t C + i } , \mathbf { P } _ { [ t ] } ^ { r } = \mathbf { P } _ { t C + 1 } ^ { t C + r } , \mathbf { H } _ { [ t ] } ^ { r } = \mathbf { H } _ { t C + 1 } ^ { t C + r }$ 。此处我们有 $\textstyle { \frac { L } { C } }$ 个大小为 $C$ 的块。这里的技巧是使用 §3.1 中描述的类似方法来高效表示矩阵 $\mathbf { P } _ { [ t ] } ^ { r } , \mathbf { H } _ { [ t ] } ^ { r } \in \mathbb { R } ^ { d \times d }$，从而使这些矩阵可以存储在 $\mathcal O ( d )$ 的内存中，

$$
\mathbf {P} _ {[ t ]} ^ {r} = \mathbf {I} - \sum_ {i = 1} ^ {r} \boldsymbol {w} _ {[ t ]} ^ {i} \boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}}, \qquad \mathbf {H} _ {[ t ]} ^ {r} = \sum_ {i = 1} ^ {r} \boldsymbol {u} _ {[ t ]} ^ {i} \boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}} \quad \in \mathbb {R} ^ {d \times d}\tag{6}
$$

$$
\boldsymbol {w} _ {[ t ]} ^ {r} = \beta_ {[ t ]} ^ {r} \left(\boldsymbol {k} _ {[ t ]} ^ {r} - \sum_ {i = 1} ^ {r - 1} \boldsymbol {w} _ {[ t ]} ^ {i} (\boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}} \boldsymbol {k} _ {[ t ]} ^ {r})\right), \quad \boldsymbol {u} _ {[ t ]} ^ {r} = \beta_ {[ t ]} ^ {r} \left(\boldsymbol {v} _ {[ t ]} ^ {r} - \sum_ {i = 1} ^ {r - 1} \boldsymbol {u} _ {[ t ]} ^ {i} (\boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}} \boldsymbol {k} _ {[ t ]} ^ {r})\right) \quad \in \mathbb {R} ^ {d}\tag{7}
$$

上述内容的推导见附录。随后，基于 Eq. 5，我们可以得到隐藏状态和输出的块级递推关系为：

$$
\mathbf {S} _ {[ t ]} ^ {r} = \mathbf {S} _ {[ t ]} ^ {0} - \left(\mathbf {S} _ {[ t ]} ^ {0} \sum_ {i = 1} ^ {r} \boldsymbol {w} _ {[ t ]} ^ {i} \boldsymbol {k} _ {[ t ]} ^ {i ^ {\intercal}}\right) + \sum_ {i = 1} ^ {r} \boldsymbol {u} _ {[ t ]} ^ {i} \boldsymbol {k} _ {[ t ]} ^ {i ^ {\intercal}} = \mathbf {S} _ {[ t ]} ^ {0} + \sum_ {i = 1} ^ {r} \left(\boldsymbol {u} _ {[ t ]} ^ {i} - \mathbf {S} _ {[ t ]} ^ {0} \boldsymbol {w} _ {[ t ]} ^ {i}\right) \boldsymbol {k} _ {[ t ]} ^ {i ^ {\intercal}},
$$

$$
\boldsymbol {o} _ {[ t ]} ^ {r} = \mathbf {S} _ {[ t ]} ^ {r} \boldsymbol {q} _ {[ t ]} ^ {r} = \mathbf {S} _ {[ t ]} ^ {0} \boldsymbol {q} _ {[ t ]} ^ {r} + \sum_ {i = 1} ^ {r} \left(\boldsymbol {u} _ {[ t ]} ^ {i} - \mathbf {S} _ {[ t ]} ^ {0} \boldsymbol {w} _ {[ t ]} ^ {i}\right) \left(\boldsymbol {k} _ {[ t ]} ^ {i ^ {\intercal}} \boldsymbol {q} _ {[ t ]} ^ {i}\right).
$$

令 ${ \bf S } _ { [ t ] } = { \bf S } _ { [ t ] } ^ { 0 }$，类似于 Eq.1-2，上述公式可以简化为矩阵形式：

$$
\mathbf {S} _ {[ t + 1 ]} = \mathbf {S} _ {[ t ]} + \left(\mathbf {U} _ {[ t ]} - \mathbf {W} _ {[ t ]} \mathbf {S} _ {[ t ]} ^ {\top}\right) ^ {\top} \mathbf {K} _ {[ t ]},\tag{8}
$$

$$
\mathbf {O} _ {[ t ]} = \mathbf {Q} _ {[ t ]} \mathbf {S} _ {[ t ]} ^ {\top} + (\mathbf {Q} _ {[ t ]} \mathbf {K} _ {[ t ]} ^ {\top} \odot \mathbf {M}) \left(\mathbf {U} _ {[ t ]} - \mathbf {W} _ {[ t ]} \mathbf {S} _ {[ t ]} ^ {\top}\right)\tag{9}
$$

其中 $\begin{array} { r } { \bigsqcup _ { [ t ] } = \bigsqcup _ { [ t ] } ^ { 1 : C } \in \mathbb { R } ^ { C \times d } \mathrm { f o r } \bigsqcup \in \{ \mathbf { Q } , \mathbf { K } , \mathbf { V } , \mathbf { O } , \mathbf { U } , \mathbf { W } \} } \end{array}$ 定义了由堆叠 ${ \bf q } _ { t } , { \bf k } _ { t } , { \bf v } _ { t } , o _ { t } , { \bf u } _ { t } , { \bf w } _ { t }$ 向量所形成的分块矩阵（chunkwise matrices）。

实际考量。在上述内容中，公式 7（Eq. 7）是完全循环的，因此按原样书写无法使用张量核心（tensor cores）。为了解决这个问题，我们进一步利用 UT 变换（UT transform）[44, 23]（推导见 §B.2）：

$$
\mathbf {T} _ {[ t ]} = \left(\mathbf {I} + \operatorname{tril} (\operatorname{diag} (\beta_ {[ t ]}) \mathbf {K} _ {[ t ]} \mathbf {K} _ {[ t ]} ^ {\top}, - 1)\right) ^ {- 1} \operatorname{diag} \left(\beta_ {[ t ]}\right)\tag{10}
$$

$$
\mathbf {W} _ {[ t ]} = \mathbf {T} _ {[ t ]} \mathbf {K} _ {[ t ]}, \qquad \mathbf {U} _ {[ t ]} = \mathbf {T} _ {[ t ]} \mathbf {V} _ {[ t ]}\tag{11}
$$

来将大多数操作重写为矩阵乘法（matmuls）。下三角矩阵的逆可以通过前向替换（forward substitution）高效求解。一旦计算完成，隐藏状态更新（公式 8）和输出计算（公式 9）在很大程度上与标准线性注意力（vanilla linear attention）相同。我们调整了 FLASHLINEARATTENTION [124] 来实现 $\operatorname { E q . }$ 8 和 9，并在反向传播期间重新计算隐藏状态以节省 GPU 显存。前向传播的 $\mathrm { P y } .$ Torch 伪代码如 Listing 8 所示。

![](images/56bdcf0102cd47998f1c5c129536efffea3721392e549d412bf044dd1f0586e3.jpg)

速度对比。我们在 Triton [112] 中实现了纯循环形式<sup>4</sup>和分块并行形式，并在右图中展示了不同序列长度（L）和头维度 $( d _ { \mathrm { h e a d } } )$ 下的加速比，其中模型维度 d 为

Figure 1: 分块并行形式对比循环形式的加速比。

2048，并且我们改变批次大小（batch size）和序列长度，使它们的乘积为 $1 6 3 8 4 . ^ { 5 }$ 随着序列长度 L 和头维度 $d _ { \mathrm { h e a d } }$ 的增加，我们的分块算法实现了更大的加速比，此时使用序列级并行（以实现高 GPU 占用率）和张量核心（用于快速矩阵乘法）变得更加重要 [124, §3]。

DeltaNet 的全并行形式。为了完整起见，我们还讨论了 DeltaNet 的全并行形式。虽然我们使用了“伪”值（pseudo value）的概念，但避免修改值也是可能的。根据公式 4，可以直接计算注意力矩阵 A：当 $j \leq i$ 时，$\mathbf { A } _ { i j } = \boldsymbol { k } _ { j } ^ { \intercal } \mathbf { P } _ { j + 1 } ^ { i } \pmb { q } _ { i }$，否则为 0。值得注意的是，结合公式 3 和公式 11，A 具有矩阵形式 $\mathbf { A } = \left( \mathbf { Q } \mathbf { K } ^ { T } \odot \mathbf { M } \right)$ T。然而，计算 T 需要进行矩阵求逆（公式 10），在没有进一步算法改进的情况下，其复杂度随序列长度呈三次方向上扩展。基于上述原因，我们避免使用全并行形式来训练 DeltaNet；然而，从该形式派生出的“注意力”矩阵可能对 RNN 的可解释性研究具有参考价值 [3, 134]。

## 3.3 DeltaNet Transformer

我们描述了如何使用 DeltaNet 层原语结合标准模块构建类 Transformer 模型。我们主要遵循 LLaMA 架构 [Transformer++, 114]，并直接将自注意力层替换为 DeltaNet 层。我们还在输出投影之前应用了归一化以实现稳定训练 [86, 69]。由于用于计算标量 $\beta _ { t }$ 项的额外参数微不足道，因此参数分配与 Transformer++ 大致相同，即 DeltaNet 层为 $4 d ^ { 2 }$，SwiGLU FFN 层为 $8 d ^ { 2 }$ [103]。

![](images/ba357c35310e488dffacf5cbfcb8863582b89d4854342851e7bac91eaa0787a9.jpg)  
Figure 2: DeltaNet 神经架构示意图。

特征映射与归一化。我们的键/查询向量由 $\begin{array} { r } { \pmb { k } _ { t } = \frac { \mathrm { S i L U } ( \mathbf { W } _ { K } \pmb { x } _ { t } ) } { \| \mathrm { S i L U } ( \mathbf { W } _ { K } \pmb { x } _ { t } ) \| _ { 2 } } , \pmb { q } _ { t } = } \end{array}$ $\frac { \mathrm { S i L U } ( \mathbf { W } _ { Q } \pmb { x } _ { t } ) } { \| \mathrm { S i L U } ( \mathbf { W } _ { Q } \pmb { x } _ { t } ) \| _ { 2 } }$ 给出。Schlag et al. [101] 最初遵循 Katharopoulos et al. [48]，应用 $^ { 6 6 } \mathrm { E L U } + 1 ^ { 5 }$ [17] 对键/查询向量进行非线性变换。我们改用 SiLU 激活函数 [24]，它被发现具有更好的表现 [88, 19]。为了保证稳定性，确保转移矩阵每个特征值的模长不超过 1 至关重要。$\mathbf { I } - \beta _ { t } \pmb { k } _ { t } \pmb { k } _ { t } ^ { \top }$ 的特征值为重数为 $d - 1$ 的 1 以及重数为 1 的 $1 - \beta _ { t } \| \pmb { k } _ { t } \| _ { 2 }$。Schlag et al. [101] 使用 $L _ { 1 }$ 范数对查询/键向量进行归一化，以确保 $0 \leq 1 - \bar { \beta } _ { t } \| \boldsymbol k _ { t } \| _ { 2 } \leq 1$。我们则采用 $L _ { 2 }$ 归一化，我们发现它的表现更好，并且提供了一种更直观的解释：当 $\beta _ { t } = 1$ 时，$\mathbf { I } - k _ { t } \pmb { k } _ { t } ^ { \top }$ 变为一个投影矩阵，在擦除一个子空间中信息的同时保留其他 $d - 1$ 个子空间。这有利于在保留信息的同时实现更有针对性的遗忘。

## 3.4 混合模型

借鉴近期将次二次（subquadratic）标记混合层与现有神经网络原语相结合的研究工作 [6, 21, 55]，我们还对 DeltaNet 模型的混合化进行了实验。

卷积层。近期的线性循环模型通常在查询/键/值投影之后引入一个轻量级的深度可分离卷积层 [31, 9, 19]。这种“短卷积”层 [83] 泛化了移位 SSM [26]，且在参数量和计算开销方面都非常高效。我们同样在查询/键/值投影之后添加了一个短卷积层。

局部滑动窗口与全局注意力。线性注意力（Linear attention）主要采用基于内容的寻址机制 [29]，且缺乏位置信息 [129]。Arora et al. [6] 亦指出，线性注意力缺乏执行精确局部 token 移位和比较的能力，因而在检索密集型任务上面临困难。受此启发，我们尝试了两种融合 softmax 注意力的不同混合架构。我们首先探索了滑动窗口注意力（sliding window attention, SWA），该方法已被证明能显著改善线性注意力 [86, 6, 57, 75]；我们遵循 Griffin [21] 和 Samba [95] 的做法，将 DeltaNet 层与 SWA 层交替堆叠。我们还尝试了全局注意力（global attention），研究发现即便仅用全局注意力替换少数几个循环层也是大有裨益的 [51, 35, 55]。我们遵循 Fu et al. [26] 的做法，仅将两层替换为全局注意力：第二层以及第 $\begin{array} { r } { ( \frac { N } { 2 } + 1 ) } \end{array}$ 层，其中 N 为总层数。

## 4 实证研究

我们在合成任务和真实世界的语言建模设置中，将 DeltaNet 与强基线模型进行了对比。我们的主要基线包括：LLaMA 架构的 Transformer++ [114]；RetNet [108]，一种具有非数据依赖型指数衰减和大头维度的线性注意力 Transformer；

GLA [124]，一种具有数据依赖型衰减的线性注意力 Transformer；以及 Mamba [31]，一种具有数据依赖型衰减的选择性状态空间模型。

## 4.1 合成基准测试

我们在三个合成基准测试上进行了评估：多查询关联回忆 [MQAR; 4]、机制化架构设计 [MAD; 84] 以及上下文语言学习 [RegBench; 2]。

<table><tr><td>模型</td><td>压缩</td><td>模糊回忆</td><td>上下文回忆</td><td>记忆</td><td>噪声回忆</td><td>选择性复制</td><td>平均</td></tr><tr><td>Transformer</td><td>51.6</td><td>29.8</td><td>94.1</td><td>85.2</td><td>86.8</td><td>99.6</td><td>74.5</td></tr><tr><td>Hyena</td><td>45.2</td><td>7.9</td><td>81.7</td><td>89.5</td><td>78.8</td><td>93.1</td><td>66.0</td></tr><tr><td>Multihead Hyena</td><td>44.8</td><td>14.4</td><td>99.0</td><td>89.4</td><td>98.6</td><td>93.0</td><td>73.2</td></tr><tr><td>Mamba</td><td>52.7</td><td>6.7</td><td>90.4</td><td>89.5</td><td>90.1</td><td>86.3</td><td>69.3</td></tr><tr><td>GLA</td><td>38.8</td><td>6.9</td><td>80.8</td><td>63.3</td><td>81.6</td><td>88.6</td><td>60.0</td></tr><tr><td>DeltaNet</td><td>42.2</td><td>35.7</td><td>100</td><td>52.8</td><td>100</td><td>100</td><td>71.8</td></tr></table>

Figure 3: 合成 MAD 基准测试上的结果。除 DeltaNet 之外的结果均直接引用自 Poli et al. [84]。(Multi-head) Hyena、DeltaNet 和 Mamba 使用了卷积，而 GLA 则没有。

![](images/316054db7ca964ab652f2b7a19df93fc5cad2b976573075db1bdea6da4186788.jpg)  
Figure 4: MQAR 上的准确率（%）。

MQAR 评估语言模型在面对多个召回查询时，在上下文中（in-context）召回信息的能力。我们采用 Arora 等人 [4] 的训练设置，对于 DeltaNet，我们使用 2 个头。在这些实验中我们没有使用卷积。Figure 4 显示，DeltaNet 在最困难的设置下表现完美（即使没有卷积），并且在低维设置下优于 Mamba（使用了卷积）。接下来，我们考虑 MAD 基准测试 [84]，这是一套旨在探究模型架构能力的合成标记操作任务。结果如 Table 3 所示。与其他架构（包括 MHA）相比，DeltaNet 在召回任务上表现更好，正如预期的那样，特别是在 Fuzzy Recall 任务上，尽管它在“Memorize”任务上略有吃力。由于篇幅限制，我们将 RegBench 的结果放在了 §A.2。

## 4.2 语言建模

实验设置。遵循先前的工作 [31, 124]，我们在 Wikitext 困惑度以及零样本常识推理任务上进行评估，包括 LAMBADA [LMB.; 77]、PiQA [12]、HellaSwag [Hella.; 127]、WinoGrande [Wino.; 99]、ARC-easy (ARC-e) 和 ARC-challenge (Arc-c) [16]。遵循 Arora 等人 [6] 的方法，我们还在真实世界的密集召回任务上对模型进行了评估，包括 FDA [5]、SWDE [60] 和 SQUAD [93]。SWDE 和 FDA 都侧重于提取结构化信息：SWDE 从原始 HTML 中识别半结构化关系，而 FDA 从 PDF 中检索键值对。SQUAD 通过提供一段文本和一个相关问题来评估语言模型的阅读理解能力。超参数设置见 §A.1。

结果。我们的主要语言建模结果如 Table 1 所示。由于 Mamba 默认使用卷积而 GLA 则不使用，我们重新训练了带卷积的 GLA，并训练了不带卷积的 DeltaNet。对于 1.3B 设置，由于计算资源有限，我们仅训练了带卷积的 DeltaNet。总体而言，我们发现 DeltaNet 在困惑度（perplexity）和下游任务性能方面均优于强大的 Mamba/GLA 基线。对于召回密集型任务（即 SWDE、SQuAD、FDA），我们发现在 340M 规模及相同状态大小下，DeltaNet 优于 GLA，证实了 delta 规则的有效性。然而，在 1.3B 规模下，DeltaNet 的表现逊于 GLA，这是由于其较差的状态大小可扩展性（参见 §5.3），因为状态大小在召回密集型任务中起着重要作用。最后，我们证实了混合架构 [21, 55] 的优势：滑动窗口和全局注意力混合模型均表现良好，超越了强大的 Transformer++ 基线。我们还将 DeltaNet 扩展到了 3B 参数规模，并使用与 Shen 等人 [104] 相同的设置在 1T token 上进行训练。结果如 Table 5 所示，其中 3B DeltaNet 的表现略逊于在相同设置下训练的 Transformer 架构（PowerLM-3B），但优于 2B–3B 范围内的其他 RNN 基线（尽管这些基线是在不同数量的 token 上训练的，因此不完全具有可比性）。

消融实验。在 Table 1（底部）中，我们对特征图（feature map）和归一化的选择进行了消融实验。我们发现，仅通过将 L -norm 替换为 L -norm 就能极大地提升性能。对于特征图，我们对 {ReLU, 1 + ELU, SiLU} 进行了实验，发现 SiLU 表现最好，这与先前的工作 [88] 一致。

训练吞吐量。Figure 6 比较了不同 1.3B 模型在不同训练长度和批次大小（batch size）设置下的训练吞吐量。DeltaNet 的训练速度接近 GLA，且显著快于 Mamba。在更长序列训练中，所有线性时间模型的表现均优于 Transformer。

<table><tr><td>模型</td><td>Wiki. ppl ↓</td><td>LMB. ppl ↓</td><td>LMB. acc ↑</td><td>PIQA acc ↑</td><td>Hella. acc_n ↑</td><td>Wino. acc ↑</td><td>ARC-e acc ↑</td><td>ARC-c acc_n ↑</td><td>平均</td><td>SWDE acc ↑</td><td>SQuAD acc ↑</td><td>FDA acc ↑</td><td>状态扩展</td></tr><tr><td colspan="14">340M 参数 / 15B tokens</td></tr><tr><td>Transformer++</td><td>28.39</td><td>42.69</td><td>31.0</td><td>63.3</td><td>34.0</td><td>50.4</td><td>44.5</td><td>24.2</td><td>41.2</td><td>42.2</td><td>22.1</td><td>21.4</td><td>N/A</td></tr><tr><td>RetNet（无卷积）</td><td>32.33</td><td>49.19</td><td>28.6</td><td>63.5</td><td>33.5</td><td>52.5</td><td>44.5</td><td>23.4</td><td>41.0</td><td>13.3</td><td>27.6</td><td>2.9</td><td>512x</td></tr><tr><td>Mamba（含卷积）</td><td>28.39</td><td>39.66</td><td>30.6</td><td>65.0</td><td>35.4</td><td>50.1</td><td>46.3</td><td>23.6</td><td>41.8</td><td>12.4</td><td>23.0</td><td>2.1</td><td>64x</td></tr><tr><td>GLA（无卷积）</td><td>28.65</td><td>43.35</td><td>30.3</td><td>64.8</td><td>34.5</td><td>51.4</td><td>45.1</td><td>22.7</td><td>41.5</td><td>18.6</td><td>27.2</td><td>8.1</td><td>128x</td></tr><tr><td>（含卷积）</td><td>29.47</td><td>45.53</td><td>31.3</td><td>65.1</td><td>33.8</td><td>51.6</td><td>44.4</td><td>24.6</td><td>41.8</td><td>24.0</td><td>24.7</td><td>7.3</td><td>128x</td></tr><tr><td>DeltaNet（无卷积）</td><td>29.08</td><td>50.87</td><td>30.0</td><td>63.6</td><td>33.6</td><td>51.7</td><td>46.0</td><td>23.0</td><td>41.3</td><td>24.6</td><td>26.9</td><td>4.5</td><td>128x</td></tr><tr><td>DeltaNet（含卷积）</td><td>28.24</td><td>37.37</td><td>32.1</td><td>64.8</td><td>34.3</td><td>52.2</td><td>45.8</td><td>23.5</td><td>42.1</td><td>26.4</td><td>28.9</td><td>12.8</td><td>128x</td></tr><tr><td>+ 滑动注意力</td><td>27.06</td><td>38.17</td><td>33.4</td><td>64.0</td><td>35.3</td><td>50.9</td><td>45.9</td><td>23.2</td><td>42.1</td><td>39.3</td><td>32.5</td><td>18.8</td><td>N/A</td></tr><tr><td>+ 全局注意力（2 层）</td><td>27.51</td><td>35.04</td><td>33.5</td><td>64.0</td><td>34.5</td><td>51.7</td><td>46.0</td><td>23.3</td><td>42.1</td><td>42.9</td><td>32.1</td><td>23.1</td><td>N/A</td></tr><tr><td colspan="14">1.3B 参数 / 100B tokens</td></tr><tr><td>Transformer++</td><td>16.85</td><td>13.44</td><td>48.9</td><td>70.8</td><td>49.6</td><td>53.6</td><td>56.0</td><td>26.5</td><td>50.9</td><td>66.6</td><td>31.5</td><td>27.4</td><td>N/A</td></tr><tr><td>RetNet（无卷积）</td><td>18.64</td><td>17.27</td><td>43.3</td><td>70.0</td><td>47.3</td><td>52.5</td><td>54.8</td><td>25.6</td><td>48.9</td><td>42.8</td><td>34.7</td><td>14.3</td><td>512x</td></tr><tr><td>Mamba（含卷积）</td><td>17.06</td><td>13.89</td><td>46.2</td><td>72.2</td><td>40.1</td><td>54.1</td><td>59.0</td><td>28.2</td><td>50.0</td><td>41.4</td><td>35.2</td><td>6.2</td><td>64x</td></tr><tr><td>GLA（无卷积）</td><td>17.22</td><td>14.47</td><td>46.9</td><td>71.8</td><td>49.8</td><td>53.9</td><td>57.2</td><td>26.6</td><td>51.0</td><td>50.6</td><td>42.6</td><td>19.9</td><td>256x</td></tr><tr><td>（含卷积）</td><td>17.25</td><td>14.92</td><td>46.2</td><td>70.6</td><td>49.9</td><td>53.0</td><td>55.3</td><td>27.0</td><td>50.4</td><td>52.4</td><td>37.4</td><td>22.3</td><td>256x</td></tr><tr><td>DeltaNet（含卷积）</td><td>16.87</td><td>12.21</td><td>48.9</td><td>71.2</td><td>50.2</td><td>53.6</td><td>57.2</td><td>28.3</td><td>51.6</td><td>49.5</td><td>37.4</td><td>17.2</td><td>128x</td></tr><tr><td>+ 滑动注意力</td><td>16.56</td><td>11.74</td><td>49.2</td><td>71.8</td><td>51.1</td><td>52.8</td><td>58.9</td><td>28.8</td><td>52.1</td><td>53.3</td><td>43.3</td><td>22.3</td><td>N/A</td></tr><tr><td>+ 全局注意力（2 层）</td><td>16.55</td><td>12.40</td><td>48.8</td><td>70.8</td><td>50.7</td><td>54.2</td><td>58.4</td><td>28.1</td><td>51.8</td><td>71.0</td><td>43.0</td><td>29.8</td><td>N/A</td></tr><tr><td colspan="14">DeltaNet 消融实验（340M）</td></tr><tr><td>采用 $L_1$ 范数与 1+ELU</td><td>31.12</td><td>55.96</td><td>26.3</td><td>63.9</td><td>33.0</td><td>50.9</td><td>44.3</td><td>21.8</td><td>40.1</td><td>14.5</td><td>23.9</td><td>6.2</td><td>128x</td></tr><tr><td>采用 $L_2$ 范数与 1+ELU</td><td>28.03</td><td>37.62</td><td>32.2</td><td>65.7</td><td>34.7</td><td>51.8</td><td>45.4</td><td>22.5</td><td>42.1</td><td>23.8</td><td>28.6</td><td>13.1</td><td>128x</td></tr><tr><td>采用 $L_2$ 范数与 ReLU</td><td>28.75</td><td>43.53</td><td>30.2</td><td>64.0</td><td>33.9</td><td>48.9</td><td>45.6</td><td>22.8</td><td>40.9</td><td>27.2</td><td>26.7</td><td>9.0</td><td>128x</td></tr></table>

Table 1: 与 Transformer++、RetNet [108]、Mamba [31] 和 GLA [124] 对比的主要语言建模结果。所有模型均在 SlimPajama 数据集的相同子集上使用 Mistral 分词器进行训练。Transformer++、RetNet、Mamba、GLA（无卷积）的结果取自 Yang 等人 [124]。对于混合模型，“Sliding Attn”表示每隔一层交替使用滑动窗口注意力，“Global Attn”表示在两层上使用全全局注意力。340M/1.3B 模型分别训练了 15B/100B 个 token。所有结果均通过 lm-evaluation-harness [27] 获得。最后一列表示循环状态大小相对于层数与模型维度乘积的扩展比例（参见 Zhang 等人 [131, 附录 C]）。

<table><tr><td>模型</td><td>ARC</td><td>HellaSwag</td><td>OBQA</td><td>PIQA</td><td>WinoGrande</td><td>MMLU</td><td>平均值</td></tr><tr><td>Llama-3.2-3B [111]</td><td>59.1</td><td>73.6</td><td>43.4</td><td>77.5</td><td>69.2</td><td>54.1</td><td>62.8</td></tr><tr><td>PowerLM-3B [104]</td><td>60.5</td><td>74.6</td><td>43.6</td><td>79.9</td><td>70.0</td><td>45.0</td><td>62.3</td></tr><tr><td>DeltaNet-3B</td><td>60.4</td><td>72.8</td><td>41.0</td><td>78.5</td><td>65.7</td><td>40.7</td><td>59.8</td></tr><tr><td>RecurrentGemma-2B [30]</td><td>57.0</td><td>71.1</td><td>42.0</td><td>78.2</td><td>67.6</td><td>31.8</td><td>57.9</td></tr><tr><td>RWKV-6-3B [79]</td><td>49.5</td><td>68.6</td><td>40.6</td><td>76.8</td><td>65.4</td><td>28.4</td><td>54.9</td></tr><tr><td>Mamba-2.7B [31]</td><td>50.3</td><td>65.3</td><td>39.4</td><td>75.8</td><td>63.1</td><td>26.1</td><td>53.3</td></tr></table>

Figure 5: 3B 模型在选定基准测试中的零样本（zero-shot）性能。Llama-3.2-3B 和 PowerLM-3B 为 Transformer 模型，其余为循环模型。ARC 结果为 ARC-Easy 和 ARC-Challenge 归一化准确率的平均值。

![](images/80f283283617ebafeb0ef06685e5dd95afc1f187ef56bc7339b113d333b3fdbe.jpg)  
Figure 6: 1.3B 模型在单张 H100 上的训练吞吐量。

## 5 讨论与相关工作

## 5.1 DeltaNet 对比状态空间模型 / 线性 RNN

为了讨论 DeltaNet 与现有线性 RNN（包括状态空间模型）的区别，我们首先引入一类具有矩阵值隐状态的通用联想式 RNN。给定矩阵值隐状态 $\mathbf { S } _ { t } \in \mathbb { R } ^ { d \times n }$ 和当前输入 $\pmb { x } _ { t } \in \mathbb { R } ^ { d }$，这些模型具有如下形式：

$$
\begin{array}{l l} \mathbf {S} _ {t} = \mathbf {S} _ {t - 1} \bullet \mathbf {M} _ {t} + \boldsymbol {v} _ {t} \boldsymbol {k} _ {t} ^ {\top}, & \text {(recurrence)} \\ \boldsymbol {o} _ {t} = \mathbf {S} _ {t} \boldsymbol {q} _ {t}, & \text {(memory read - out)} \end{array}
$$

其中 • 是一个结合算子（例如，Hadamard 积、矩阵乘法等）。矩阵 M<sub>t</sub> 和向量 ${ \pmb v } _ { t } , { \pmb k } _ { t } , { \pmb q } _ { t }$ 是当前输入 $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ 的（可能是非线性的）函数。

正如在向量值线性 RNN [64, 106] 中的情况一样，如果给定输入 $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { L }$，使用结合算子便能够利用并行扫描（parallel scan）[13] 在 O(log $L )$ 步和 $O ( L )$ 计算量内（忽略与结合操作相关的项）计算出 $\mathbf { S } _ { 1 } , \ldots , \mathbf { S } _ { L }$（不过参见我们在脚注 1 中的讨论）。因此，只要结合算子的计算开销不过高，训练就可以非常高效。然而，由于某些结合算子的开销过大，仅靠并行扫描本身不足以在实际应用规模下训练语言模型。因此，诸如 Mamba [31] 和门控线性注意力 Transformer（gated linear attention Transformers）[108, 124, 92, 79, 9] 等近期模型采用了开销较低的逐元素循环更新，特别是 Hadamard 积，$\mathrm { i . e . , } \bullet = \odot$。有关近期模型如何转换为该形式的内容，请参见 Table 2。

<table><tr><td>模型</td><td>循环递推</td><td>记忆读出</td></tr><tr><td>Linear Attention（线性注意力）[48, 47]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} + \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t}$ </td></tr><tr><td>+ 核函数</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} + \boldsymbol{v}_{t} \phi(\boldsymbol{k}_{t})^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \phi(\boldsymbol{q}_{t})$ </td></tr><tr><td>+ 归一化</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} + \boldsymbol{v}_{t} \phi(\boldsymbol{k}_{t})^{\top}, \boldsymbol{z}_{t} = \boldsymbol{z}_{t-1} + \phi(\boldsymbol{k}_{t})$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \phi(\boldsymbol{q}_{t}) / (\boldsymbol{z}_{t}^{\top} \phi(\boldsymbol{q}_{t}))$ </td></tr><tr><td>DeltaNet [101]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} (\mathbf{I} - \beta_{t} \boldsymbol{k}_{t} \boldsymbol{k}_{t}^{\top}) + \beta_{t} \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t}$ </td></tr><tr><td>Gated RFA [81]</td><td> $\mathbf{S}_{t} = g_{t} \mathbf{S}_{t-1} + (1 - g_{t}) \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}, \boldsymbol{z}_{t} = g_{t} \boldsymbol{z}_{t-1} + (1 - g_{t}) \boldsymbol{k}_{t}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t} / (\boldsymbol{z}_{t}^{\top} \boldsymbol{q}_{t})$ </td></tr><tr><td>S4 [32, 106]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} \odot \exp(-(\boldsymbol{\alpha} \mathbf{1}^{\top}) \odot \exp(\boldsymbol{A})) + \boldsymbol{B} \odot (\boldsymbol{v}_{t} \mathbf{1}^{\top})$ </td><td> $\boldsymbol{o}_{t} = (\mathbf{S}_{t} \odot \boldsymbol{C}) \mathbf{1} + \boldsymbol{d} \odot \boldsymbol{v}_{t}$ </td></tr><tr><td>ABC [82]</td><td> $\mathbf{S}_{t}^{k} = \mathbf{S}_{t-1}^{k} + \boldsymbol{k}_{t} \phi_{t}^{\top}, \mathbf{S}_{t}^{v} = \mathbf{S}_{t-1}^{v} + \boldsymbol{v}_{t} \phi_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \text{softmax} (\mathbf{S}_{t}^{k} \boldsymbol{q}_{t})$ </td></tr><tr><td>DFW [63]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} \odot (\boldsymbol{\beta}_{t} \boldsymbol{\alpha}_{t}^{\top}) + \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t}$ </td></tr><tr><td>RetNet [108]</td><td> $\mathbf{S}_{t} = \gamma \mathbf{S}_{t-1} + \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t}$ </td></tr><tr><td>Mamba [31]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} \odot \exp(-(\boldsymbol{\alpha}_{t} \mathbf{1}^{\top}) \odot \exp(\boldsymbol{A})) + (\boldsymbol{\alpha}_{t} \odot \boldsymbol{v}_{t}) \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t} + \boldsymbol{d} \odot \boldsymbol{v}_{t}$ </td></tr><tr><td>GLA [124]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} \odot (\mathbf{1} \boldsymbol{\alpha}_{t}^{\top}) + \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top} = \mathbf{S}_{t-1} \text{Diag}(\boldsymbol{\alpha}_{t}) + \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t}$ </td></tr><tr><td>RWKV-6 [79]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} \text{Diag}(\boldsymbol{\alpha}_{t}) + \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = (\mathbf{S}_{t-1} + (\boldsymbol{d} \odot \boldsymbol{v}_{t}) \boldsymbol{k}_{t}^{\top}) \boldsymbol{q}_{t}$ </td></tr><tr><td>HGRN-2 [92]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} \text{Diag}(\boldsymbol{\alpha}_{t}) + \boldsymbol{v}_{t} (\mathbf{1} - \boldsymbol{\alpha}_{t})^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t}$ </td></tr><tr><td>mLSTM [9]</td><td> $\mathbf{S}_{t} = f_{t} \mathbf{S}_{t-1} + i_{t} \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}, \boldsymbol{z}_{t} = f_{t} z_{t-1} + i_{t} \boldsymbol{k}_{t}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t} / \max\{1, |\boldsymbol{z}_{t}^{\top} \boldsymbol{q}_{t}| \}$ </td></tr><tr><td>Mamba-2 [19]</td><td> $\mathbf{S}_{t} = \gamma_{t} \mathbf{S}_{t-1} + \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t}$ </td></tr><tr><td>GSA [131]</td><td> $\mathbf{S}_{t}^{k} = \mathbf{S}_{t-1}^{k} \text{Diag}(\boldsymbol{\alpha}_{t}) + \boldsymbol{k}_{t} \phi_{t}^{\top}, \mathbf{S}_{t}^{v} = \mathbf{S}_{t-1}^{v} \text{Diag}(\boldsymbol{\alpha}_{t}) + \boldsymbol{v}_{t} \phi_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t}^{v} \text{softmax} (\mathbf{S}_{t}^{k} \boldsymbol{q}_{t})$ </td></tr><tr><td>Gated DeltaNet [125]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} \left( \alpha_{t} (\mathbf{I} - \beta_{t} \boldsymbol{k}_{t} \boldsymbol{k}_{t}^{\top}) \right) + \beta_{t} \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t}$ </td></tr></table>

Table 2: 近期被提出并应用于自回归语言建模的线性循环模型概览（大致按时间顺序排列）。这些工作利用了通过结合性循环后接基于外积的加法来更新的矩阵值隐藏状态 $\breve { \mathbf { S } _ { t } } \in \mathbb { R } ^ { d \times n }$（或两个矩阵值隐藏状态 $\mathbf { S } _ { t } ^ { k } , \mathbf { S } _ { t } ^ { v } , \mathbf { e . g . } , [ 8 2 , 1 3 1 ] )$。这里的 ⊙ 表示阿达马积（Hadamard product）。某些模型利用带有隐藏状态向量 $\scriptstyle { z _ { t } }$ 的额外线性 RNN，用于对查询向量 $\mathbf { \nabla } _ { \mathbf { \eta } } \mathbf { q } _ { t }$ 进行归一化。带有下标 $t \ ( \mathbf { e } . \mathbf { g } . , \ v _ { t } , \alpha _ { t } , f _ { t } , \gamma _ { t } )$ 的变量是当前输入 $\mathbf { \Delta } \mathbf { x } _ { t } .$ 的（潜在非线性）函数。非时变参数 $( \mathrm { e . g . } , A , d , \gamma )$ 在表示时不带下标；这些参数要么是通过学习得到的，要么被设为固定值。矩阵用粗体大写字母表示，向量用粗体小写字母表示，标量用斜体字母表示。许多模型使用了核函数 φ（例如 [101, 81]），但为了简化符号表示，我们将它们归入了键/值向量中。

另一方面，标准矩阵乘法 $( \mathrm { i . e . , } \ \mathbf { S } _ { t - 1 } \bullet \mathbf { M } _ { t } = \mathbf { S } _ { t - 1 } \mathbf { M } _ { t } )$ 能够建模超越逐元素循环的更丰富的交互。然而，若不对 ${ { \bf { M } } _ { t } }$ 作出任何结构性假设，这些操作在每次更新时将耗费 $O ( d n ^ { 2 } )$ 的复杂度（相比之下逐元素乘积为 $O ( d n )$），这将带来过高的计算开销。因此，DeltaNet 对 $\mathbf { M } _ { t } = \mathbf { I } - \beta _ { t } \pmb { k } _ { t } \pmb { k } _ { t } ^ { \top }$ 的使用可以看作是利用结构化矩阵来高效建模超越逐元素循环的交互。我们的分块算法（chunkwise algorithm）可以推广到更广泛的对角加低秩（Diagonal-Plus-Low-Rank, DPLR）形式矩阵 $\mathbf { M } _ { t } = \mathbf { D } - \mathbf { a } _ { t } \mathbf { b } _ { t } ^ { \mathsf { T } }$，该形式已在 S4 [32] 中被探讨，尽管它们的 DPLR 转移矩阵是与数据无关的。我们在本工作中采用了 DeltaNet 的参数化形式 $( \mathrm { i . e . , D } = \mathbf { I } , \pmb { a } _ { t } = \beta _ { t } \pmb { k } _ { t } , \pmb { b } _ { t } = \pmb { k } _ { t } )$，因为我们主要关注在保持参数效率的同时（通过 DeltaNet 的键值更新规则）提升召回能力。我们将对更通用参数化形式的探索留待未来的工作。

## 5.2 迈向高效自回归序列变换的统一框架

尽管上述模型类别使得统一近期模型成为可能，但我们并不断言这是审视形如 $\{ \pmb { x } _ { t } \} _ { t = 1 } ^ { L } \ \mapsto \ \{ \pmb { o } _ { t } \} _ { t = 1 } ^ { L }$（其中 $\mathbf { } _ { o _ { t } }$ 不能依赖于任何 $x _ { j } ~ \mathrm { i f } ~ j ~ > ~ t .$）的（自回归）序列变换的 $\mathbf { \ddot { r i g h t } } ^ { \mathbf { \vec { r } } }$ 层面。例如，这种构想框架很难（整洁地）涵盖已被证明有效的其他亚二次方模型 [126, 49, 98, 83]。另一种统一框架可能是将上述序列变换视为连续状态空间模型（state space model, SSM）的离散化 [32, 106, 31]，或是视为与掩码结构化矩阵的矩阵乘法 [76, 87, 46, 19]。然而，看似至关重要的一点是，一个框架在理想情况下应能展现出高效的训练算法，并且该算法应具备硬件高效性；在现代 GPU 的语境下，这意味着它应当富含矩阵乘法运算。从这个角度来看，Dao 和 Gu [19] 最近提出的状态空间对偶（state-space duality, SSD）框架在基于 SSM 的序列变换与带有半可分矩阵的结构化矩阵乘法之间建立了联系，似乎是一个很有前景的候选方案。然而，该框架可能无法涵盖一类重要模型，$\mathrm { e . g . }$，结合律递推涉及与非结构化矩阵进行矩阵乘法的模型，或者利用了更为奇特的结合律算子的模型（例如 Peng et al. [80] 中的模型）。

最后，我们观察到近期提出了许多据称能够匹敌或超越经典 Transformer 的线性时间模型。如 Table 2 所示，这些工作中的“序列混合”（sequence mixing）组件彼此密切相关。然而，利用词元混合（token-mixing）原语构建类 Transformer 模型的方式却大相径庭。例如，虽然大多数近期工作采用了深度可分离卷积层（未在 Table 2 中展示）[31, 9, 19, 15, 125]，但早期工作通常没有采用 [48, 101, 81]。用于“通道混合”（channel mixing）组件的前馈层在参数化方式上也存在差异。在断定某一个模型层优于另一个模型层之前，应当充分考虑这些差异。

## 5.3 局限性与未来工作

我们的工作存在一些局限性。首先，在计算方面，尽管我们提出了一种新的硬件高效算法，但训练速度仍落后于 GLA。这正如上文所述，是由于建模状态间依赖关系（state-to-state dependencies）所带来的开销导致的，它需要在核函数（kernel）内部对头维度（head dimension）进行“边缘化”（marginalizing），类似于 Softmax 注意力机制的情况。然而，对于 GLA 而言，由于不存在状态内依赖关系（所有计算都是逐元素的），因此很容易利用分块（tiling）技术来支持任意大小的头维度，正如 Yang 和 Zhang [123] 所实现的那样。这一局限性可能会限制 DeltaNet 的记忆大小，从而降低其在召回密集型任务上的性能，正如我们在 §4.2 中所观察到的。不过，采用块大小适合 GPU SRAM（例如 128）的分块对角广义 Householder 变换矩阵，同时保持总体较大的头维度（从而获得较大的循环状态大小），可能是可行的。

我们还发现 DeltaNet 的长度泛化能力有限，而 GLA 和 RetNet（以及在一定程度上的 Mamba）已被发现能够外推到训练长度之外 [124]。我们推测这是因为 DeltaNet 缺乏显式的衰减因子。正如 Yang 等人 [125] 最近的工作所示，这可以通过在循环中引入门控项来进行改进。

## 6 相关工作

我们在此简要讨论相关工作，并在附录 C 中进行更深入的讨论。

线性 Transformer 可以看作是一种迭代 Hopfield 网络（iterated Hopfield networks）[72]，这种联系可以为线性注意力 Transformer 的局限性与改进提供视角。例如，原生线性 Transformer 使用类似赫布（Hebbian-like）的更新规则，该规则已被证明记忆容量有限 [68]。Hopfield 网络后续的工作使用高阶多项式 [22] 和指数核 [94, 50] 来提升记忆容量，这也与带有多项式核的线性注意力相关 [45, 6, 1]。另一方面，Delta 规则（delta rule）已被证明具有更好的记忆容量 [28, 85, 56, 101]。从这个意义上说，在给定固定大小的循环状态下，使用 Delta 规则能够在召回率-记忆权衡曲线（recall-memory tradeoff curve）上实现更优的前沿边界 [6]，并且最近已被应用于增强现实世界的检索任务 [74, 96]。此外，它在多个领域均优于原生线性 Transformer 中所使用的加法规则 [101, 38, 40, 36, 42]。

尽管有这些优势，Irie 等人 [42] 揭示了 delta 更新规则在表达能力方面的理论局限性。学界提出了 DeltaNet 的循环增强版本，如 Recurrent DeltaNet [37]、Modern Self-Referential Weight Matrix [41] 以及 mesa-layer [117]，并发现它们表现更优。然而，这些模型超越了线性循环神经网络（linear RNNs）的范畴，无法在序列长度维度上进行并行化。这表明并行性与表达能力之间存在根本性的权衡 [70]。如何在不牺牲并行性的前提下进一步增强 DeltaNet 仍然是一个开放性问题，而 TTT [110] 中采用的块间非线性与块内线性的混合策略可能会提供一个合适的折中方案。最后，我们指出 delta 规则与通过梯度下降进行的元学习或在线学习密切相关 [73, 39]，这一方向在 Longhorn [58] 和 TTT [110] 等近期工作中被重新审视。最近，Titans [10] 通过引入动量项和权重衰减项对 TTT 进行了改进。

## 7 结论

我们描述了一种在序列长度维度上并行化 DeltaNet 训练的算法，在现代硬件上相比现有实现取得了显著的加速。这使得将 DeltaNet 扩展到中等规模语言建模设置成为可能，在这些设置中，我们发现它与近期的线性循环基线模型相比表现良好。

## 致谢

本研究得到了 MIT-IBM Watson AI Lab 和 MIT-Google 计算创新计划（MIT-Google Program for Computing Innovation）的资助。我们感谢 Mayank Mishra 在训练和评估 3B 模型方面提供的帮助，感谢 Kazuki Irie 对初稿提出的宝贵反馈，以及 Simran Arora、Liliang Ren 和 Eric Alcaide 富有见地的讨论。我们还要感谢 Michael Poli 和 Armin Thomas 分享了 MAD 基准实验的原始结果，以及 Fan Zhou 指出了 Eq. 10 中的一处错误。

## 参考文献

[1] Y. Aksenov, N. Balagansky, S. M. L. C. Vaina, B. Shaposhnikov, A. Gorbatovski, and D. Gavrilov. Linear Transformers with Learnable Kernel Functions are Better In-Context Models, 2024. URL https://arxiv.org/abs/2402.10644.

[2] E. Akyürek, B. Wang, Y. Kim, and J. Andreas. In-context language learning: Arhitectures and algorithms. ArXiv preprint, abs/2401.12973, 2024. URL https://arxiv.org/abs/ 2401.12973.

[3] A. Ali, I. Zimerman, and L. Wolf. The hidden attention of mamba models, 2024.

[4] S. Arora, S. Eyuboglu, A. Timalsina, I. Johnson, M. Poli, J. Zou, A. Rudra, and C. Ré. Zoology: Measuring and improving recall in efficient language models. ArXiv preprint, abs/2312.04927, 2023. URL https://arxiv.org/abs/2312.04927.

[5] S. Arora, B. Yang, S. Eyuboglu, A. Narayan, A. Hojel, I. Trummer, and C. Ré. Language Mod els Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes, 2023. URL https://arxiv.org/abs/2304.09433.

[6] S. Arora, S. Eyuboglu, M. Zhang, A. Timalsina, S. Alberti, D. Zinsley, J. Zou, A. Rudra, and C. Ré. Simple linear attention language models balance the recall-throughput tradeoff. ArXiv preprint, abs/2402.18668, 2024. URL https://arxiv.org/abs/2402.18668.

[7] S. Arora, A. Timalsina, A. Singhal, B. Spector, S. Eyuboglu, X. Zhao, A. Rao, A. Rudra, and C. Ré. Just read twice: closing the recall gap for recurrent language models, 2024. URL https://arxiv.org/abs/2407.05483.

[8] D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and translate. In Y. Bengio and Y. LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. URL http://arxiv.org/abs/1409.0473.

[9] M. Beck, K. Pöppel, M. Spanring, A. Auer, O. Prudnikova, M. Kopp, G. Klambauer, J. Brandstetter, and S. Hochreiter. xlstm: Extended long short-term memory. ArXiv preprint, abs/2405.04517, 2024. URL https://arxiv.org/abs/2405.04517.

[10] A. Behrouz, P. Zhong, and V. Mirrokni. Titans: Learning to memorize at test time, 2024. URL https://arxiv.org/abs/2501.00663.

[11] C. H. Bischof and C. V. Loan. The WY representation for products of householder matrices. In SIAM Conference on Parallel Processingfor Scientific Computing, 1985. URL https:// api.semanticscholar.org/CorpusID:36094006.

[12] Y. Bisk, R. Zellers, R. LeBras, J. Gao, and Y. Choi. PIQA: reasoning about physical commonsense in natural language. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications ofArtificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 7432–7439. AAAI Press, 2020. URL https://aaai.org/ojs/index.php/AAAI/article/view/6239.

[13] G. E. Blelloch. Prefix sums and their applications. 1990.

[14] W. Brandon, A. Nrusimha, K. Qian, Z. Ankner, T. Jin, Z. Song, and J. Ragan-Kelley. Striped Attention: Faster Ring Attention for Causal Transformers. ArXiv preprint, abs/2311.09431, 2023. URL https://arxiv.org/abs/2311.09431.

[15] Y. Chou, M. Yao, K. Wang, Y. Pan, R.-J. Zhu, J. Wu, Y. Zhong, Y. Qiao, B. XU, and G. Li. MetaLA: Unified optimal linear approximation to softmax attention map. In The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024. URL https:// openreview.net/forum?id=Y8YVCOMEpz.

[16] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. ArXiv preprint, abs/1803.05457, 2018. URL https://arxiv.org/abs/1803.05457.

[17] D. Clevert, T. Unterthiner, and S. Hochreiter. Fast and accurate deep network learning by exponential linear units (elus). In Y. Bengio and Y. LeCun, editors, 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings, 2016. URL http://arxiv.org/abs/1511.07289.

[18] T. Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ArXiv preprint, abs/2307.08691, 2023. URL https://arxiv.org/abs/2307.08691.

[19] T. Dao and A. Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv: 2405.21060, 2024.

[20] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/ 67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html.

[21] S. De, S. L. Smith, A. Fernando, A. Botev, G. Cristian-Muraru, A. Gu, R. Haroun, L. Berrada, Y. Chen, S. Srinivasan, G. Desjardins, A. Doucet, D. Budden, Y. W. Teh, R. Pascanu, N. De Freitas, and C. Gulcehre. Griffin: Mixing Gated Linear Recurrences with Local Atten tion for Efficient Language Models, 2024. URL https://arxiv.org/abs/2402.19427.

[22] M. Demircigil, J. Heusel, M. Löwe, S. Upgang, and F. Vermet. On a model of associative memory with huge storage capacity. ArXiv preprint, abs/1702.01929, 2017. URL https:// arxiv.org/abs/1702.01929.

[23] A. E. T. Dominguez and E. S. Q. Orti. Fast blocking of householder reflectors on graphics processors. 2018 26th Euromicro International Conference on Parallel, Distributed and Networkbased Processing (PDP), pages 385–393, 2018. URL https://api.semanticscholar. org/CorpusID:46960439.

[24] S. Elfwing, E. Uchibe, and K. Doya. Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning, 2017. URL https://arxiv.org/ abs/1702.03118.

[25] M. Fathi, J. Pilault, P.-L. Bacon, C. Pal, O. Firat, and R. Goroshin. Block-state transformer. ArXiv preprint, abs/2306.09539, 2023. URL https://arxiv.org/abs/2306.09539.

[26] D. Y. Fu, T. Dao, K. K. Saab, A. W. Thomas, A. Rudra, and C. Ré. Hungry hungry hippos: Towards language modeling with state space models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/pdf?id=COZDy0WYGg.

[27] L. Gao, J. Tow, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, K. McDonell, N. Muennighoff, J. Phang, L. Reynolds, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou. A framework for few-shot language model evaluation, 2021.

[28] E. Gardner. The space of interactions in neural network models. Journal of Physics A, 21: 257–270, 1988.

[29] A. Graves, G. Wayne, and I. Danihelka. Neural Turing Machines, 2014. URL http:// arxiv.org/abs/1410.5401. arXiv:1410.5401 [cs].

[30] R. Griffin and G. Teams. Recurrentgemma: Moving past transformers for efficient open language models. ArXiv preprint, abs/2404.07839, 2024. URL https://arxiv.org/abs/ 2404.07839.

[31] A. Gu and T. Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. In Proceedings of COLM, 2023.

[32] A. Gu, K. Goel, and C. Ré. Efficiently modeling long sequences with structured state spaces. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum? id=uYLFoz1vlAC.

[33] K. Helfrich, D. Willmott, and Q. Ye. Orthogonal recurrent neural networks with scaled cayley transform. In J. G. Dy and A. Krause, editors, Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pages 1974–1983. PMLR, 2018. URL http://proceedings.mlr.press/v80/helfrich18a.html.

[34] W. Hua, Z. Dai, H. Liu, and Q. V. Le. Transformer quality in linear time. In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvári, G. Niu, and S. Sabato, editors, International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 9099–9117. PMLR, 2022. URL https://proceedings.mlr.press/v162/hua22a.html.

[35] F. Huang, K. Lu, Y. Cai, Z. Qin, Y. Fang, G. Tian, and G. Li. Encoding recurrence into transformers. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview. net/pdf?id=7YfHla7IxBJ.

[36] K. Irie and J. Schmidhuber. Images as weight matrices: Sequential image generation through synaptic learning rules. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https:// openreview.net/pdf?id=ddad0PNUvV.

[37] K. Irie, I. Schlag, R. Csordás, and J. Schmidhuber. Going beyond linear transformers with recurrent fast weight programmers. In M. Ranzato, A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 7703–7717, 2021. URL https://proceedings.neurips.cc/paper/ 2021/hash/3f9e3767ef3b10a0de4c256d7ef9805d-Abstract.html.

[38] K. Irie, I. Schlag, R. Csordás, and J. Schmidhuber. Going beyond linear transformers with recurrent fast weight programmers. In M. Ranzato, A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 7703–7717, 2021. URL https://proceedings.neurips.cc/paper/ 2021/hash/3f9e3767ef3b10a0de4c256d7ef9805d-Abstract.html.

[39] K. Irie, R. Csordás, and J. Schmidhuber. The dual form of neural networks revisited: Connecting test time predictions to training patterns via spotlights of attention. In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvári, G. Niu, and S. Sabato, editors, International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 9639–9659. PMLR, 2022. URL https://proceedings.mlr.press/v162/irie22a.html.

[40] K. Irie, F. Faccio, and J. Schmidhuber. Neural differential equations for learning to program neural nets through continuous learning rules. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Process-

ing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/ fc09b26b85ab3abb2832bd555a2e4215-Abstract-Conference.html.

[41] K. Irie, I. Schlag, R. Csordás, and J. Schmidhuber. A modern self-referential weight matrix that learns to modify itself. In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvári, G. Niu, and S. Sabato, editors, International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings ofMachine Learning Research, pages 9660–9677. PMLR, 2022. URL https://proceedings.mlr.press/v162/ irie22b.html.

[42] K. Irie, R. Csordás, and J. Schmidhuber. Practical computational power of linear transform ers and their recurrent and self-referential extensions. In H. Bouamor, J. Pino, and K. Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9455–9465, Singapore, 2023. Association for Computational Linguis tics. doi: 10.18653/v1/2023.emnlp-main.588. URL https://aclanthology.org/2023. emnlp-main.588.

[43] L. Jing, C. Gulcehre, J. Peurifoy, Y. Shen, M. Tegmark, M. Soljacic, and Y. Bengio. Gated Orthogonal Recurrent Units: On Learning to Forget. Neural Computation, 31(4):765–783, 2019. ISSN 0899-7667, 1530-888X. doi: 10.1162/neco\_a\_01174. URL https://direct. mit.edu/neco/article/31/4/765-783/8458.

[44] T. Joffrain, T. M. Low, E. S. Quintana-Ortí, R. A. van de Geijn, and F. G. V. Zee. Accumu lating householder transformations, revisited. ACM Trans. Math. Softw., 32:169–179, 2006. URL https://api.semanticscholar.org/CorpusID:15723171.

[45] P. Kacham, V. Mirrokni, and P. Zhong. Polysketchformer: 通过多项式核草图实现的快速 Transformer. ArXiv 预印本, abs/2310.01655, 2023. URL https://arxiv.org/ abs/2310.01655.

[46] Y. Kang, G. Tran, and H. De Sterck. 快速多极子注意力：一种面向长序列的分治注意力机制. ArXiv 预印本, abs/2310.11960, 2023. URL https:// arxiv.org/abs/2310.11960.

[47] J. Kasai, H. Peng, Y. Zhang, D. Yogatama, G. Ilharco, N. Pappas, Y. Mao, W. Chen, and N. A. Smith. 将预训练 Transformer 微调为 RNN. 载于 M.-F. Moens, X. Huang, L. Specia, and S. W.-t. Yih 编, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 第 10630–10643 页, 线上及多米尼加共和国蓬塔卡纳, 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.830. URL https://aclanthology.org/2021.emnlp-main.830.

[48] A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret. Transformer 即 RNN：基于线性注意力的快速自回归 Transformer. 载于 Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 2020年7月13-18日, 线上会议, 第 119 卷 Proceedings of Machine Learning Research, 第 5156–5165 页. PMLR, 2020. URL http:// proceedings.mlr.press/v119/katharopoulos20a.html.

[49] N. Kitaev, L. Kaiser, and A. Levskaya. Reformer: 高效 Transformer. 载于 8th International Conference on Learning Representations, ICLR 2020, 埃塞俄比亚亚的斯亚贝巴, 2020年4月26-30日. OpenReview.net, 2020. URL https://openreview.net/forum? id=rkgNKkHtvB.

[50] D. Krotov and J. J. Hopfield. 神经生物学与机器学习中的大容量联想记忆问题. 载于 9th International Conference on Learning Representations, ICLR 2021, 线上会议, 奥地利, 2021年5月3-7日. OpenReview.net, 2021. URL https://openreview.net/ forum?id=X4y\_10OX-hX.

[51] T. Lei. 当注意力遇上快速循环：以更少的计算量训练语言模型. 载于 M.-F. Moens, X. Huang, L. Specia, and S. W.-t. Yih 编, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 第 7633–7648 页, 线上及多米尼加共和国蓬塔卡纳, 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.602. URL https://aclanthology.org/2021. emnlp-main.602.

[52] T. Lei, R. Tian, J. Bastings, and A. P. Parikh. 简单循环改进掩码语言模型. ArXiv 预印本, abs/2205.11588, 2022. URL https://arxiv.org/abs/ 2205.11588.

[53] P. S. H. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6- 12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 6b493230205f780e1bc26945df7481e5-Abstract.html.

[54] B. Liao, X. Wang, L. Zhu, Q. Zhang, and C. Huang. Vig: Linear-complexity visual sequence learning with gated linear attention. ArXiv, abs/2405.18425, 2024. URL https://api. semanticscholar.org/CorpusID:270068351.

[55] O. Lieber, B. Lenz, H. Bata, G. Cohen, J. Osin, I. Dalmedigos, E. Safahi, S. Meirom, Y. Belinkov, S. Shalev-Shwartz, et al. Jamba: A hybrid transformer-mamba language model. ArXiv preprint, abs/2403.19887, 2024. URL https://arxiv.org/abs/2403.19887.

[56] K. C. Lingashetty. Delta learning rule for the active sites model. arXiv preprint arXiv:1007.0417, 2010.

[57] L. D. Lingle. Transformer-vq: Linear-time transformers via vector quantization. ArXiv preprint, abs/2309.16354, 2023. URL https://arxiv.org/abs/2309.16354.

[58] B. Liu, R. Wang, L. Wu, Y. Feng, P. Stone, and Q. Liu. Longhorn: State space models are amortized online learners. ArXiv preprint, abs/2407.14207, 2024. URL https://arxiv. org/abs/2407.14207.

[59] H. Liu, M. Zaharia, and P. Abbeel. Ring Attention with Blockwise Transformers for Near-Infinite Context. ArXiv preprint, abs/2310.01889, 2023. URL https://arxiv.org/abs/ 2310.01889.

[60] C. Lockard, P. Shiralkar, and X. L. Dong. OpenCeres: When open information extraction meets the semi-structured web. In J. Burstein, C. Doran, and T. Solorio, editors, Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3047– 3056, Minneapolis, Minnesota, 2019. Association for Computational Linguistics. doi: 10. 18653/v1/N19-1309. URL https://aclanthology.org/N19-1309.

[61] X. Ma, C. Zhou, X. Kong, J. He, L. Gui, G. Neubig, J. May, and L. Zettlemoyer. Mega: Moving average equipped gated attention. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/pdf?id=qNLe3iq2El.

[62] X. Ma, X. Yang, W. Xiong, B. Chen, L. Yu, H. Zhang, J. May, L. Zettlemoyer, O. Levy, and C. Zhou. Megalodon: Efficient llm pretraining and inference with unlimited context length. ArXiv preprint, abs/2404.08801, 2024. URL https://arxiv.org/abs/2404.08801.

[63] H. H. Mao. Fine-tuning pre-trained transformers into decaying fast weights. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10236–10242, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.697. URL https://aclanthology.org/2022.emnlp-main.697.

[64] E. Martin and C. Cundy. Parallelizing linear recurrent neural nets over sequence length. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. URL https://openreview.net/forum?id=HyUNwulC-.

[65] E. Martin and C. Cundy. Parallelizing linear recurrent neural nets over sequence length. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. URL https://openreview.net/forum?id=HyUNwulC-.

[66] A. Mathiasen, F. Hvilshoj, J. R. Jørgensen, A. Nasery, and D. Mottin. What if neural networks had svds? ArXiv, abs/2009.13977, 2020. URL https://api.semanticscholar.org/ CorpusID:221995651.

[67] A. Mathiasen, F. Hvilshøj, J. R. Jørgensen, A. Nasery, and D. Mottin. Faster orthogonal parameterization with householder matrices. In ICML, Workshop Proceedings, 2020.

[68] R. J. McEliece, E. C. Posner, E. R. Rodemich, and S. S. Venkatesh. The capacity of the hopfield associative memory. IEEE Trans. Inf. Theory, 33:461–482, 1987.

[69] J. Mercat, I. Vasiljevic, S. Keh, K. Arora, A. Dave, A. Gaidon, and T. Kollar. Linearizing large language models. ArXiv preprint, abs/2405.06640, 2024. URL https://arxiv.org/ abs/2405.06640.

[70] W. Merrill, J. Petty, and A. Sabharwal. The Illusion of State in State-Space Models, 2024. URL https://arxiv.org/abs/2404.08819.

[71] Z. Mhammedi, A. D. Hellicar, A. Rahman, and J. Bailey. Efficient orthogonal parametrisation of recurrent neural networks using householder reflections. In D. Precup and Y. W. Teh, editors, Proceedings ofthe 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings ofMachine Learning Research, pages 2401–2409. PMLR, 2017. URL http://proceedings.mlr.press/v70/ mhammedi17a.html.

[72] B. Millidge. Linear Attention as Iterated Hopfield Networks. URL http://www.beren.io/ 2024-03-03-Linear-Attention-as-Iterated-Hopfield-Networks/.

[73] T. Munkhdalai, A. Sordoni, T. Wang, and A. Trischler. Metalearned neural memory. In H. M. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché-Buc, E. B. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 13310–13321, 2019. URL https://proceedings.neurips.cc/paper/ 2019/hash/182bd81ea25270b7d1c2fe8353d17fe6-Abstract.html.

[74] T. Munkhdalai, M. Faruqui, and S. Gopal. Leave no context behind: Efficient infinite context transformers with infini-attention. ArXiv preprint, abs/2404.07143, 2024. URL https:// arxiv.org/abs/2404.07143.

[75] Y. Nahshan, J. Kampeas, and E. Haleva. Linear Log-Normal Attention with Unbiased Concentration, 2023. URL https://arxiv.org/abs/2311.13541.

[76] T. M. Nguyen, V. Suliafu, S. J. Osher, L. Chen, and B. Wang. Fmmformer: Efficient and flexible transformer via decomposed near-field and far-field attention. In M. Ranzato, A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 29449–29463, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ f621585df244e9596dc70a39b579efb1-Abstract.html.

[77] D. Paperno, G. Kruszewski, A. Lazaridou, N. Q. Pham, R. Bernardi, S. Pezzelle, M. Baroni, G. Boleda, and R. Fernández. The LAMBADA dataset: Word prediction requiring a broad discourse context. In K. Erk and N. A. Smith, editors, Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1525–1534, Berlin, Germany, 2016. Association for Computational Linguistics. doi: 10.18653/v1/P16-1144. URL https://aclanthology.org/P16-1144.

[78] J. Park, J. Park, Z. Xiong, N. Lee, J. Cho, S. Oymak, K. Lee, and D. Papailiopoulos. Can mamba learn how to learn? a comparative study on in-context learning tasks. ArXiv preprint, abs/2402.04248, 2024. URL https://arxiv.org/abs/2402.04248.

[79] B. Peng, D. Goldstein, Q. Anthony, A. Albalak, E. Alcaide, S. Biderman, E. Cheah, X. Du, T. Ferdinan, H. Hou, P. Kazienko, K. K. GV, J. Kocon, B. Koptyra, S. Krishna, R. McClel-´ land Jr., N. Muennighoff, F. Obeid, A. Saito, G. Song, H. Tu, S. Wo´zniak, R. Zhang, B. Zhao, Q. Zhao, P. Zhou, J. Zhu, and R.-J. Zhu. Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence, 2024. URL https://arxiv.org/abs/2404.05892.

[80] H. Peng, R. Schwartz, S. Thomson, and N. A. Smith. Rational recurrences. In E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii, editors, Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1203–1214, Brussels, Belgium, 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1152. URL https:// aclanthology.org/D18-1152.

[81] H. Peng, N. Pappas, D. Yogatama, R. Schwartz, N. A. Smith, and L. Kong. Random feature attention. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/ forum?id=QtTKTdVrFBB.

[82] H. Peng, J. Kasai, N. Pappas, D. Yogatama, Z. Wu, L. Kong, R. Schwartz, and N. A. Smith. ABC: Attention with bounded-memory control. In S. Muresan, P. Nakov, and A. Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7469–7483, Dublin, Ireland, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.515. URL https:// aclanthology.org/2022.acl-long.515.

[83] M. Poli, S. Massaroli, E. Nguyen, D. Y. Fu, T. Dao, S. Baccus, Y. Bengio, S. Ermon, and C. Ré. Hyena hierarchy: Towards larger convolutional language models. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings ofMachine Learning Research, pages 28043–28078. PMLR, 2023. URL https://proceedings.mlr.press/v202/poli23a.html.

[84] M. Poli, A. W. Thomas, E. Nguyen, P. Ponnusamy, B. Deiseroth, K. Kersting, T. Suzuki, B. Hie, S. Ermon, C. Ré, C. Zhang, and S. Massaroli. Mechanistic Design and Scaling of Hybrid Architectures, 2024. URL https://arxiv.org/abs/2403.17844.

[85] D. Prados and S. Kak. Neural network capacity using delta rule. Electronics Letters, 3(25): 197–199, 1989.

[86] Z. Qin, X. Han, W. Sun, D. Li, L. Kong, N. Barnes, and Y. Zhong. The devil in linear transformer. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 7025–7041, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics. doi: 10. 18653/v1/2022.emnlp-main.473. URL https://aclanthology.org/2022.emnlp-main. 473.

[87] Z. Qin, X. Han, W. Sun, B. He, D. Li, D. Li, Y. Dai, L. Kong, and Y. Zhong. Toeplitz neural network for sequence modeling. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/pdf?id=IxmWsm4xrua.

[88] Z. Qin, D. Li, W. Sun, W. Sun, X. Shen, X. Han, Y. Wei, B. Lv, F. Yuan, X. Luo, et al. Scaling transnormer to 175 billion parameters. ArXiv preprint, abs/2307.14995, 2023. URL https://arxiv.org/abs/2307.14995.

[89] Z. Qin, W. Sun, K. Lu, H. Deng, D. Li, X. Han, Y. Dai, L. Kong, and Y. Zhong. Linearized Relative Positional Encoding, 2023. URL https://arxiv.org/abs/2307.09270.

[90] Z. Qin, S. Yang, and Y. Zhong. Hierarchically gated recurrent neural network for sequence modeling. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ 694be3548697e9cc8999d45e8d16fe1e-Abstract-Conference.html.

[91] Z. Qin, W. Sun, D. Li, X. Shen, W. Sun, and Y. Zhong. Lightning attention-2: A free lunch for handling unlimited sequence lengths in large language models. 2024.

[92] Z. Qin, S. Yang, W. Sun, X. Shen, D. Li, W. Sun, and Y. Zhong. HGRN2: Gated Linear RNNs with State Expansion. 2024. URL https://api.semanticscholar.org/ CorpusID:269043328.

[93] P. Rajpurkar, R. Jia, and P. Liang. Know what you don’t know: Unanswerable questions for SQuAD. In I. Gurevych and Y. Miyao, editors, Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 784–789, Melbourne, Australia, 2018. Association for Computational Linguistics. doi: 10.18653/v1/ P18-2124. URL https://aclanthology.org/P18-2124.

[94] H. Ramsauer, B. Schäfl, J. Lehner, P. Seidl, M. Widrich, L. Gruber, M. Holzleitner, T. Adler, D. P. Kreil, M. K. Kopp, G. Klambauer, J. Brandstetter, and S. Hochreiter. Hopfield networks is all you need. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview. net/forum?id=tL89RnzIiCd.

[95] L. Ren, Y. Liu, Y. Lu, Y. Shen, C. Liang, and W. Chen. Samba: Simple hybrid state space models for efficient unlimited context language modeling. ArXiv preprint, abs/2406.07522, 2024. URL https://arxiv.org/abs/2406.07522.

[96] I. Rodkin, Y. Kuratov, A. Bulatov, and M. Burtsev. Associative recurrent memory transformer. ArXiv preprint, abs/2407.04841, 2024. URL https://arxiv.org/abs/2407.04841.

[97] S. Rogers. Adaptive filter theory. Control Engineering Practice, 4:1629–1630, 1996. URL https://api.semanticscholar.org/CorpusID:109523942.

[98] A. Roy, M. Saffar, A. Vaswani, and D. Grangier. Efficient content-based sparse attention with routing transformers. Transactions of the Association for Computational Linguistics, 9:53– 68, 2021. doi: 10.1162/tacl\_a\_00353. URL https://aclanthology.org/2021.tacl-1. 4.

[99] K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi. Winogrande: An adversarial winograd schema challenge at scale. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications ofArtificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 8732–8740. AAAI Press, 2020. URL https://aaai.org/ojs/index.php/AAAI/article/view/6399.

[100] Y. Schiff, C.-H. Kao, A. Gokaslan, T. Dao, A. Gu, and V. Kuleshov. Caduceus: Bidirectional equivariant long-range dna sequence modeling. ArXiv, abs/2403.03234, 2024. URL https://api.semanticscholar.org/CorpusID:268253280.

[101] I. Schlag, K. Irie, and J. Schmidhuber. Linear transformers are secretly fast weight programmers. In M. Meila and T. Zhang, editors, Proceedings ofthe 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 9355–9366. PMLR, 2021. URL http://proceedings. mlr.press/v139/schlag21a.html.

[102] J. Schmidhuber. Learning to control fast-weight memories: An alternative to dynamic recurrent networks. Neural Computation, 4(1):131–139, 1992.

[103] N. Shazeer. Glu variants improve transformer. ArXiv preprint, abs/2002.05202, 2020. URL https://arxiv.org/abs/2002.05202.

[104] Y. Shen, M. Stallone, M. Mishra, G. Zhang, S. Tan, A. Prasad, A. M. Soria, D. D. Cox, and R. Panda. Power scheduler: A batch size and token number agnostic learning rate scheduler. ArXiv preprint, abs/2408.13359, 2024. URL https://arxiv.org/abs/2408.13359.

[105] J. T. H. Smith, A. Warrington, and S. W. Linderman. Simplified state space layers for sequence modeling. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https:// openreview.net/pdf?id=Ai8Hw3AXqks.

[106] J. T. H. Smith, A. Warrington, and S. W. Linderman. Simplified state space layers for sequence modeling. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https:// openreview.net/pdf?id=Ai8Hw3AXqks.

[107] W. Sun, Z. Qin, D. Li, X. Shen, Y. Qiao, and Y. Zhong. Linear attention sequence parallelism. ArXiv preprint, abs/2404.02882, 2024. URL https://arxiv.org/abs/2404.02882.

[108] Y. Sun, L. Dong, S. Huang, S. Ma, Y. Xia, J. Xue, J. Wang, and F. Wei. Retentive network: A successor to transformer for large language models. ArXiv preprint, abs/2307.08621, 2023. URL https://arxiv.org/abs/2307.08621.

[109] Y. Sun, L. Dong, Y. Zhu, S. Huang, W. Wang, S. Ma, Q. Zhang, J. Wang, and F. Wei. You only cache once: Decoder-decoder architectures for language models. ArXiv preprint, abs/2405.05254, 2024. URL https://arxiv.org/abs/2405.05254.

[110] Y. Sun, X. Li, K. Dalal, J. Xu, A. Vikram, G. Zhang, Y. Dubois, X. Chen, X. Wang, O. Koyejo, T. Hashimoto, and C. Guestrin. Learning to (learn at test time): Rnns with expressive hid den states. ArXiv preprint, abs/2407.04620, 2024. URL https://arxiv.org/abs/2407. 04620.

[111] L. Team. The llama 3 herd of models. ArXiv preprint, abs/2407.21783, 2024. URL https:// arxiv.org/abs/2407.21783.

[112] P. Tillet, H. Kung, and D. D. Cox. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, MAPL@PLDI 2019, pages 10–19. ACM, 2019. doi: 10.1145/3315508.3329973.

[113] J. M. Tomczak and M. Welling. Improving Variational Auto-Encoders using Householder Flow, 2016. URL https://arxiv.org/abs/1611.09630.

[114] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient foundation language models. ArXiv preprint, abs/2302.13971, 2023. URL https://arxiv.org/abs/2302.13971.

[115] R. van den Berg, L. Hasenclever, J. M. Tomczak, and M. Welling. Sylvester normalizing flows for variational inference. In A. Globerson and R. Silva, editors, Proceedings of the Thirty-Fourth Conference on Uncertainty in Artificial Intelligence, UAI 2018, Monterey, California, USA, August 6-10, 2018, pages 393–402. AUAI Press, 2018. URL http://auai.org/uai2018/proceedings/papers/156.pdf.

[116] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. In I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.

[117] J. von Oswald, E. Niklasson, M. Schlegel, S. Kobayashi, N. Zucchet, N. Scherrer, N. Miller, M. Sandler, B. A. y Arcas, M. Vladymyrov, R. Pascanu, and J. Sacramento. Uncovering mesaoptimization algorithms in transformers. CoRR, abs/2309.05858, 2023. URL https://doi.org/10.48550/arXiv.2309.05858.

[118] E. Vorontsov, C. Trabelsi, S. Kadoury, and C. Pal. On orthogonality and learning recurrent networks with long term dependencies. In D. Precup and Y. W. Teh, editors, Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia,

6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pages 3570–3578. PMLR, 2017. URL http://proceedings.mlr.press/v70/vorontsov17a.html.

[119] C. X. Wang, O. Tsepa, J. Ma, and B. Wang. Graph-mamba: Towards long-range graph sequence modeling with selective state spaces. ArXiv, abs/2402.00789, 2024. URL https://api.semanticscholar.org/CorpusID:267364853.

[120] J. Wang, J. N. Yan, A. Gu, and A. Rush. Pretraining without attention. In H. Bouamor, J. Pino, and K. Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 58–69, Singapore, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.5. URL https://aclanthology.org/2023.findings-emnlp.5.

[121] B. Widrow, M. E. Hoff, et al. Adaptive switching circuits. In IRE WESCON convention record, volume 4, pages 96–104. New York, 1960.

[122] J. N. Yan, J. Gu, and A. M. Rush. Diffusion models without attention. 2023.

[123] S. Yang and Y. Zhang. FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism, 2024. URL https://github.com/sustcsonglin/ flash-linear-attention. original-date: 2023-12-20T06:50:18Z.

[124] S. Yang, B. Wang, Y. Shen, R. Panda, and Y. Kim. Gated linear attention transformers with hardware-efficient training. ArXiv preprint, abs/2312.06635, 2023. URL https://arxiv. org/abs/2312.06635.

[125] S. Yang, J. Kautz, and A. Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule, 2024. URL https://arxiv.org/abs/2412.06464.

[126] M. Zaheer, G. Guruganesh, K. A. Dubey, J. Ainslie, C. Alberti, S. Ontañón, P. Pham, A. Ravula, Q. Wang, L. Yang, and A. Ahmed. Big bird: Transformers for longer sequences. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ c8512d142a2d849725f31a9a7a361ab9-Abstract.html.

[127] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi. HellaSwag: Can a machine really finish your sentence? In A. Korhonen, D. Traum, and L. Màrquez, editors, Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791– 4800, Florence, Italy, 2019. Association for Computational Linguistics. doi: 10.18653/v1/ P19-1472. URL https://aclanthology.org/P19-1472.

[128] J. Zhang, Q. Lei, and I. S. Dhillon. Stabilizing gradients for deep neural networks via efficient SVD parameterization. In J. G. Dy and A. Krause, editors, Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pages 5801– 5809. PMLR, 2018. URL http://proceedings.mlr.press/v80/zhang18g.html.

[129] J. Zhang, S. Jiang, J. Feng, L. Zheng, and L. Kong. Linear Attention via Orthogonal Memory, 2023. URL https://arxiv.org/abs/2312.11135.

[130] Q. Zhang, D. Ram, C. Hawkins, S. Zha, and T. Zhao. Efficient long-range transformers: You need to attend more, but not necessarily at every layer. In H. Bouamor, J. Pino, and K. Bali, ed itors, Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 2775– 2786, Singapore, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. findings-emnlp.183. URL https://aclanthology.org/2023.findings-emnlp.183.

[131] Y. Zhang, S. Yang, R. Zhu, Y. Zhang, L. Cui, Y. Wang, B. Wang, F. Shi, B. Wang, W. Bi, P. Zhou, and G. Fu. Gated slot attention for efficient linear-time sequence modeling. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[132] L. Zhu, Z. Huang, B. Liao, J. H. Liew, H. Yan, J. Feng, and X. Wang. Dig: Scalable and efficient diffusion models with gated linear attention. ArXiv, abs/2405.18428, 2024. URL https://api.semanticscholar.org/CorpusID:270068366.

[133] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang. Vision mamba: Efficient visual representation learning with bidirectional state space model. ArXiv, abs/2401.09417, 2024. URL https://api.semanticscholar.org/CorpusID:267028142.

[134] I. Zimerman, A. Ali, and L. Wolf. A unified implicit attention formulation for gated-linear recurrent sequence models. ArXiv preprint, abs/2405.16504, 2024. URL https://arxiv. org/abs/2405.16504.

## A 实验补充

## A.1 超参数

我们在 340M 和 1.3B 语言建模实验中使用了 8 块 H100 GPU。每个模型均使用 AdamW 进行优化，峰值学习率为 $3 \times 1 0 ^ { - 4 }$。340M 模型使用 150 亿个 token 进行训练，批大小为 0.5M token；而 1.3B 模型使用 1000 亿个 token 进行训练，批大小为 2M token。我们采用余弦学习率调度，340M 模型的预热阶段为 5 亿个 token，1.3B 模型为 10 亿个 token。两种配置的初始和最终学习率均设置为 $3 \times 1 0 ^ { - 5 }$。我们应用了 0.01 的权重衰减，并将梯度裁剪的最大值设为 1.0。DeltaNet 的头维度设为 128，卷积层的卷积核大小设为 4。

## A.2 合成任务

我们还在 RegBench [2] 上进行了补充实验，这是一个旨在评估不同模型架构的上下文语言学习能力的合成数据集。该基准测试中的每个输入序列由来自特定语言的 10 到 20 个字符串组成，该语言由概率有限自动机（PFA）定义，因此模型需要实时从上下文中推断出底层语言。在测试过程中，通过评估模型预测由保留的 PFA 生成的测试序列的下一个 token 来进行评估。在这里我们再次发现，如图 7 所示，与基线模型相比，DeltaNet 表现出了强劲的性能。

![](images/a1d5d763a644c0a62844d9e618859200beb4c5af883000de5bd1023c155d6d3a.jpg)  
Figure 7: RegBench 上的准确率（%）。

## B 方法补充

## B.1 WY 表示的推导

为了简化符号表示，我们在此仅讨论第一个块（chunk）。

我们首先通过数学归纳法证明 $\begin{array} { r } { \mathbf { P } _ { n } = \mathbf { I } - \sum _ { t = 1 } ^ { n } w _ { t } \pmb { k } _ { t } ^ { \top } } \end{array}$，

$$
\begin{array}{l} \mathbf {P} _ {n} = \prod_ {t = 1} ^ {n} (\mathbf {I} - \beta_ {t} \boldsymbol {k} _ {t} \boldsymbol {k} _ {t} ^ {\intercal}) \\ \quad = \mathbf {P} _ {n - 1} (\mathbf {I} - \beta_ {n} \boldsymbol {k} _ {n} \boldsymbol {k} _ {n} ^ {\intercal}) \\ \quad = (\mathbf {I} - \sum_ {t = 1} ^ {n - 1} \boldsymbol {w} _ {t} \boldsymbol {k} _ {t} ^ {\intercal}) (\mathbf {I} - \beta_ {n} \boldsymbol {k} _ {n} \boldsymbol {k} _ {n} ^ {\intercal}) \\ \quad = \mathbf {I} - \sum_ {t = 1} ^ {n - 1} \boldsymbol {w} _ {t} \boldsymbol {k} _ {t} ^ {\intercal} - \beta_ {n} \boldsymbol {k} _ {n} \boldsymbol {k} _ {n} ^ {\intercal} + (\sum_ {t = 1} ^ {n - 1} \boldsymbol {w} _ {t} \boldsymbol {k} _ {t} ^ {\intercal}) \beta_ {n} \boldsymbol {k} _ {n} \boldsymbol {k} _ {n} ^ {\intercal} \\ \quad = \mathbf {I} - \sum_ {t = 1} ^ {n - 1} \boldsymbol {w} _ {t} \boldsymbol {k} _ {t} ^ {\intercal} - \underbrace {\left(\beta_ {n} \boldsymbol {k} _ {n} - \beta_ {n} \sum_ {t = 1} ^ {n - 1} \left(\boldsymbol {w} _ {t} (\boldsymbol {k} _ {t} ^ {\intercal} \boldsymbol {k} _ {n})\right)\right)} _ {\boldsymbol {w} _ {n}} \boldsymbol {k} _ {n} ^ {\intercal} \\ \quad = \mathbf {I} - \sum_ {t = 1} ^ {n} \boldsymbol {w} _ {t} \boldsymbol {k} _ {t} ^ {\intercal} \end{array}
$$

类似地，我们通过数学归纳法证明 $\begin{array} { r } { \mathbf { S } _ { n } = \sum _ { t = 1 } ^ { n } { \mathbf { u } } _ { t } { \mathbf { k } } _ { n } ^ { \top } } \end{array}$，

$$
\begin{array}{l} \mathbf {S} _ {n} = \mathbf {S} _ {n - 1} (\mathbf {I} - \beta_ {n} \boldsymbol {k} _ {n} \boldsymbol {k} _ {n} ^ {\top}) + \beta_ {n} \boldsymbol {v} _ {n} \boldsymbol {k} _ {n} ^ {\top} \\ \qquad = \left(\sum_ {t = 1} ^ {n - 1} \boldsymbol {u} _ {t} \boldsymbol {k} _ {t} ^ {\top}\right) (\mathbf {I} - \beta_ {n} \boldsymbol {k} _ {n} \boldsymbol {k} _ {n} ^ {\top}) + \beta_ {n} \boldsymbol {v} _ {n} \boldsymbol {k} _ {n} ^ {\top} \\ \qquad = \sum_ {t = 1} ^ {n - 1} \boldsymbol {u} _ {t} \boldsymbol {k} _ {t} ^ {\top} - \left(\sum_ {t = 1} ^ {n - 1} \boldsymbol {u} _ {t} \boldsymbol {k} _ {t} ^ {\top}\right) \beta_ {n} \boldsymbol {k} _ {n} \boldsymbol {k} _ {n} ^ {\top} + \beta_ {n} \boldsymbol {v} _ {n} \boldsymbol {k} _ {n} ^ {\top} \\ \qquad = \sum_ {t = 1} ^ {n - 1} \boldsymbol {u} _ {t} \boldsymbol {k} _ {t} ^ {\top} + \underbrace {\left(\beta_ {n} \boldsymbol {v} _ {n} - \beta_ {n} \sum_ {t = 1} ^ {n - 1} \boldsymbol {u} _ {t} \left(\boldsymbol {k} _ {t} ^ {\top} \boldsymbol {k} _ {n}\right)\right)} _ {\boldsymbol {u} _ {n}} \boldsymbol {k} _ {n} ^ {\top} \\ \qquad = \sum_ {t = 1} ^ {n} \boldsymbol {u} _ {t} \boldsymbol {k} _ {n} ^ {\top} \end{array}
$$

## B.2 UT 变换推导

在此，我们提供 §3.2 中所使用的 UT 变换（UT transform）的详细推导。具体而言，我们证明了使用 UT 变换的矩阵形式（正文中的公式 10 和公式 11）等价于递归更新方程（公式 7）。

我们首先从公式 7 中给出的用于计算 $\pmb { w } _ { [ t ] } ^ { r }$ 和 $\pmb { u } _ { [ t ] } ^ { r }$ 的递归公式开始：

$$
\begin{array}{l} \boldsymbol {w} _ {[ t ]} ^ {r} = \beta_ {[ t ]} ^ {r} \left(\boldsymbol {k} _ {[ t ]} ^ {r} - \sum_ {i = 1} ^ {r - 1} \boldsymbol {w} _ {[ t ]} ^ {i} (\boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}} \boldsymbol {k} _ {[ t ]} ^ {r})\right) \\ \boldsymbol {u} _ {[ t ]} ^ {r} = \beta_ {[ t ]} ^ {r} \left(\boldsymbol {v} _ {[ t ]} ^ {r} - \sum_ {i = 1} ^ {r - 1} \boldsymbol {u} _ {[ t ]} ^ {i} (\boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}} \boldsymbol {k} _ {[ t ]} ^ {r})\right) \end{array}
$$

我们的目标是证明这些递归更新可以使用 UT 变换重写为矩阵形式：

$$
\begin{array}{r l} & {\mathbf {T} _ {[ t ]} = \left(\mathbf {I} + \mathrm{tril} (\mathrm{diag} (\beta_ {[ t ]}) \mathbf {K} _ {[ t ]} \mathbf {K} _ {[ t ]} ^ {\intercal}, - 1)\right) ^ {- 1} \mathrm{diag} \left(\beta_ {[ t ]}\right)} \\ & {\mathbf {W} _ {[ t ]} = \mathbf {T} _ {[ t ]} \mathbf {K} _ {[ t ]}} \\ & {\mathbf {U} _ {[ t ]} = \mathbf {T} _ {[ t ]} \mathbf {V} _ {[ t ]}} \end{array}
$$

首先，让我们考虑 $\pmb { w } _ { [ t ] } ^ { r }$ 的计算。对于任意行 r，我们可以写成：

$$
\mathbf {W} _ {[ t ]} [ r,: ] = \beta_ {[ t ]} ^ {r} \mathbf {K} _ {[ t ]} [ r,: ] - \beta_ {[ t ]} ^ {r} \sum_ {i = 1} ^ {r - 1} \mathbf {W} _ {[ t ]} [ i,: ] (\mathbf {K} _ {[ t ]} [ i,: ] \mathbf {K} _ {[ t ]} [ r,: ] ^ {\top})
$$

该方程组可以写成矩阵形式。让我们定义：

$$
\begin{array}{c} \mathbf {B} _ {[ t ]} = \mathrm{diag} (\beta_ {[ t ]}) \\ \mathbf {L} _ {[ t ]} = \mathrm{tril} (\mathbf {B} _ {[ t ]} \mathbf {K} _ {[ t ]} \mathbf {K} _ {[ t ]} ^ {\intercal}, - 1) \end{array}
$$

那么该方程组变为：

$$
\mathbf {W} _ {[ t ]} + \mathbf {L} _ {[ t ]} \mathbf {W} _ {[ t ]} = \mathbf {B} _ {[ t ]} \mathbf {K} _ {[ t ]}
$$

因此，我们可以求解 $\mathbf { W } _ { [ t ] }$：

$$
\mathbf {W} _ {[ t ]} = (\mathbf {I} + \mathbf {L} _ {[ t ]}) ^ {- 1} \mathbf {B} _ {[ t ]} \mathbf {K} _ {[ t ]} = \mathbf {T} _ {[ t ]} \mathbf {K} _ {[ t ]}
$$

其中

$$
\mathbf {T} _ {[ t ]} = (\mathbf {I} + \mathbf {L} _ {[ t ]}) ^ {- 1} \mathbf {B} _ {[ t ]}
$$

同样的推导也适用于 $\mathbf { U } _ { [ t ] }$，只需在最后一步将 $\mathbf { K } _ { [ t ] }$ 替换为 $\mathbf { V } _ { [ t ] }$，从而得到：

$$
\mathbf {U} _ {[ t ]} = \mathbf {T} _ {[ t ]} \mathbf {V} _ {[ t ]}\tag{12}
$$

## C 相关工作（续）

分块线性注意力。Hua 等人 [34] 首次提出了线性注意力的分块形式（chunkwise form）；然而，他们使用的是类似于 Munkhdalai 等人 [74] 的线性与非线性混合注意力模型。正如 Sun 等人 [108] 和 Yang 等人 [124] 所示，可以对他们的算法进行适配，以计算纯线性注意力的精确输出。分块线性注意力算法也在多项工作中被独立发现 [108, 45, 19]。Yang 等人 [124] 和 Qin 等人 [91] 探讨了针对分块线性注意力的 I/O 感知硬件优化，Sun 等人 [107] 则将其推广至多节点分布式训练。受分块形式的启发，我们提出了一种用于硬件高效 DeltaNet 训练的新算法，显著提高了训练效率并使大规模实验成为可能。

混合模型。线性循环模型——包括状态空间模型（state-space models）[32, 105, 31, 120]、门控线性 RNN [65, 90] 和线性注意力机制 [48, 124]——在不同领域中作为传统 softmax 注意力的替代方案展现出了显著的潜力 [122, 132, 54, 133, 119, 100]。鉴于它们互补的优势，近期的研究越来越侧重于开发将线性循环层与局部块注意力（local chunk attention）[61, 130, 25, 62, 74]、滑动窗口注意力 [130, 6, 21, 95] 或全局注意力 [51, 52, 35, 26, 55, 78, 109] 相结合的混合架构。Poli 等人 [84] 系统地研究了混合模型的缩放法则（scaling law）。我们同样表明，将 DeltaNet 与经典注意力相结合是一种有效的策略。

Householder 矩阵。以保持范数著称的 Householder 矩阵是一类在机器学习中广泛使用的正交矩阵 [67, 71, 128, 113, 89, 115]。这些矩阵可以高效计算逆矩阵，且其雅可比行列式为 1，这使得它们特别适用于归一化流（normalizing flows）[67, 115] 中的应用。值得注意的是，Mathiasen 等人 [67] 和 Mathiasen 等人 [66] 利用 WY 表示（WY representation），开发了一种用于计算归一化流中 Householder 矩阵累积乘积的分块快速算法。我们的方法虽然共享了相同的高层概念，但解决的是不同的问题，并且可以说是更加通用的。

使用正交矩阵来参数化 RNN 的转移矩阵 [71, 43, 118, 33] 以缓解梯度消失，也引起了广泛的关注。Mhammedi 等人 [71] 使用 WY 表示来减少在训练具有 Householder 转移矩阵的非线性 RNN 时的内存占用。

## D 伪代码

```python
def chunk_delta_rule_forward(Q, K, V, beta, C):
    ' '
    Q/K/V: query, key, value of shape [L, d]
    beta: beta of shape [L]
    C: chunk size
    ' '
    # L: sequence length, d: head dimension
    L, d = Q.shape
    # chunking
    Q, K, V = map(lambda x: x.reshape(-1, C, d), [Q, K, V])
    beta = beta.reshape(-1, C)
    K_beta = K * beta.unsqueeze(-1)
    V_beta = V * beta.unsqueeze(-1)

    # compute eq. 10 with vectorized forward substitution for fast inverse
    T = -(K_beta @ K.t()).tril(-1)
    for i in range(1, C):
    T[i, :i] = T[i, :i] + (T[i, :, None] * T[:, :i]).sum(-2)
    T += torch.eye(C)
    # compute Eq. 11
    W = T @ K_beta
    U = T @ V_beta
    # chunkwise parallel. Eq. 8-9
    S = torch.zeros(d, d)
    O = torch.empty_like(V)
    for i in range(L//C):
    q_i, k_i, w_i = Q[i], K[i], W[i]
    u_i = U[i] - w_i @ S
    o_inter = q_i @ S
    A_i = (q_i @ k_i.t()).tril()
    o_intra = A_i @ u_i
    S += k_i.t() @ u_i
    O[i] = o_intra + o_inter
    return 0.reshape(L, d)
```

Figure 8: 用于训练 DeltaNet 的分块算法前向传播的类 PyTorch 代码片段。为清晰起见，我们省略了批大小（batch size）和头数（number of heads）的维度。