# Hyena 层次结构：迈向更大的卷积语言模型

Michael Poli<sup>∗,1</sup>, Stefano Massaroli<sup>∗,2</sup>, Eric Nguyen<sup>1,∗</sup>, Daniel Y. Fu<sup>1</sup>, Tri Dao<sup>1</sup>, Stephen Baccus<sup>1</sup>, Yoshua Bengio<sup>2</sup>, Stefano Ermon<sup>1,†</sup>, Christopher Ré<sup>1,†</sup>

版本：提交草稿，最后编译时间：2023年4月21日

## 摘要

深度学习近期的进展在很大程度上依赖于大型 Transformer 的使用，这是由于它们具备大规模学习的能力。然而，Transformer 的核心构建模块——注意力算子（attention operator），其计算成本与序列长度呈二次方增长，从而限制了可访问的上下文长度。现有基于低秩和稀疏近似的亚二次（subquadratic）方法需要与稠密注意力层结合才能匹敌 Transformer，这表明其能力上存在差距。在这项工作中，我们提出了 Hyena，这是一种通过交错隐式参数化长卷积和数据控制门控构建的注意力亚二次即插即用替代算子。在涉及数千到数十万个 token 的序列召回和推理任务中，Hyena 相比依赖状态空间（statespaces）以及其他隐式和显式方法的算子将准确率提高了 50 点以上，达到了与基于注意力的模型相当的水平。我们在标准数据集（WikiText103 和 The Pile）的语言建模上树立了无稠密注意力架构的新最先进水平（state-of-the-art），在序列长度为 2K 时达到了 Transformer 的质量，同时所需的训练计算量减少了 20%。在序列长度为 8K 时，Hyena 算子的速度是高度优化注意力的两倍，在序列长度为 64K 时则快 100 倍。

## 1 引言

大型 Transformer 在语言、视觉、音频、生物学及其他众多领域的建模中带来了一系列突破性进展 (Vaswani et al., 2017), (Dosovitskiy et al., 2020), (Radford et al., 2022), (Cramer, 2021)。由注意力算子提供支持的 Transformer 的巨大成功 (Vaswani et al., 2017)，很大程度上依赖于它们的扩展特性（scaling properties）(Hofmann et al., 2022) 以及上下文学习（in-context learning）的涌现 (Garg et al., 2022)，这使得它们能够在给定上下文作为输入的情况下泛化到未见过的数据和任务。Transformer 块是用于序列建模的强大工具，但它并非没有局限性。其中最显著的局限性之一是计算成本，它随着输入序列长度的增加而迅速增长。具体而言，该成本与序列长度 L 成二次方比例增长，这对模型能够考虑的上下文量施加了严格限制。打破二次方壁垒是迈向深度学习全新可能性的关键一步，例如将整本教科书用作上下文、生成长篇音乐或处理十亿像素（gigapixel）级的图像。

降低模型中注意力机制计算成本的努力主要涉及使用线性化、低秩和稀疏近似（Child et al., 2019; Wang et al., 2020; Kitaev et al., 2020; Zhai et al., 2021; Roy et al., 2021; Schlag et al., 2021; Tu et al., 2022）。这些方法在表达能力和速度之间引入了权衡，需要与标准注意力层进行混合才能达到 Transformer 的质量（Mehta et al., 2022; Dao et al., 2022c）。

越来越多的证据表明，注意力机制在语言处理中仅利用了其二次方能力的一小部分（Olsson et al., 2022; Dao et al., 2022c），这促使我们质疑其作为大规模深度学习黄金标准算子的地位。具体来说，我们提出疑问：

是否存在能够在规模化应用中匹敌注意力质量的次二次方算子？

![](images/637c3d5a747a409ce141e4a9a27ef14c1b877801f3e5753058b653f19509fe4d.jpg)  
Figure 1.1: Hyena 算子被定义为两种高效次二次原语的递推：隐式长卷积 h（即由前馈网络参数化的 Hyena 滤波器）以及（投影后的）输入的逐元素乘法门控。递推的深度决定了算子的大小。Hyena 也可以等价地表示为与数据控制（受输入 u 条件约束）的对角矩阵 $\mathsf { D } _ { x }$ 和 Toeplitz 矩阵 $\mathsf { S } _ { h }$ 的乘法。此外，Hyena 表现出次线性的参数扩展（随序列长度）和无限制的上下文，类似于注意力机制，同时具有更低的时间复杂度。

基于高效次二次原语的组合——例如逐元素乘法（门控）和长卷积（即滤波器大小与输入等长的卷积），我们得到了肯定的回答。我们依托一组有针对性的推理任务，这些任务基于近期关于机理可解释性（mechanistic interpretability）的研究（Elhage et al., 2021; Power et al., 2022; Olsson et al., 2022; Zhang et al., 2022），如回忆（recall）和归纳（induction），提炼出与注意力机制性能相关的三项特性，这也解释了其与现有次二次方法之间的质量差距：

a. 数据控制（Data control）：注意力实现了一个富有表达力的数据控制（Massaroli et al., 2020）线性算子<sup>1</sup>，在单个模块中编码了一整个线性函数族。

b. 次线性参数扩展（Sublinear parameter scaling）：注意力层的参数量与序列长度解耦，允许 Transformer 将更多参数分配到其他地方，例如注意力层之间的前馈神经网络（FFN）。

c. 无限制上下文（Unrestricted context）：对于给定的输入，注意力具有无限制的上下文，即它可以逼近任意两个输入之间的依赖关系，而没有诸如局部性之类的任意限制（在使用掩码的情况除外，例如自回归模型）。

**Hyena 层级结构（The Hyena hierarchy）** 在这些发现的指导下，我们引入了 Hyena 层级结构，这是一种由两个高效的次二次（subquadratic）原语循环递归定义的算子：长卷积（long convolution）和逐元素乘法门控（element-wise multiplicative gating）（参见 Figure 1.1）。该循环递归的指定深度（即步数）控制了算子的大小。对于较短的递归，现有模型可以作为特例被恢复（Mehta et al., 2022; Dao et al., 2022c）。通过将 Hyena 递归中的每一步映射到其对应的矩阵形式，我们揭示了 Hyena 算子在等价意义上被定义为数据控制矩阵（即其元素是输入的函数的矩阵）的分解。此外，我们展示了如何利用快速卷积算法（Selesnick and Burrus, 2017），在无需具象化完整矩阵的情况下高效计算 Hyena 算子。在实证上，Hyena 算子能够显著缩小在大规模场景下与注意力机制的质量差距，以更小的计算预算（Section 4.2）且在不混合注意力机制的情况下达到相近的困惑度（perplexity）和下游任务性能。

**缩小能力差距（Narrowing the capabilities gap）** Hyena 的设计源于标准稠密注意力机制与替代性次二次算子之间的质量差距，我们通过关注与大规模语言建模性能相关的推理任务发现了这一差距。我们用额外的任务扩展了基本机制可解释性基准测试套件（归纳与回忆），以探究当任务复杂度增加（例如词表大小增大）时模型性能下降的速度。此外，我们研究了 Hyena 中长卷积的最优参数化方案。在包含数十万个标记（tokens）的最具挑战性的设置中，我们的隐式参数化方案相比于利用状态空间（Gu et al., 2021）、频域参数化（Li et al., 2020）或标准卷积的其他算子，准确率提升了 50% 以上。

**在语言和视觉领域的规模扩展（Scaling in language and vision）** 接下来，我们旨在验证推理基准套件中的排名是否能够预测大规模下的模型质量。我们在十亿以下（sub-billion）参数规模的自回归语言建模上对 Hyena 进行了测试，在标准数据集（WikiText103 和 The $\mathrm { P I L E } )$ 上为无稠密注意力架构设立了新的最高水平（state-of-the-art），并达到了与 Transformer 相当的质量。在 335M 参数规模的 The Pile 数据集上，我们以浮点运算总次数（FLOPs）减少 20% 的代价达到了与 Transformer 相同的困惑度。作为扩展，我们通过在大规模图像识别上进行测试，替换 Vision Transformer (ViT)（Dosovitskiy et al., 2020）中的注意力机制，来探究 Hyena 算子的通用性。在图像分类中，从零开始在 ImageNet-1k 上训练时，Hyena 在准确率上能够与注意力机制相媲美。

走向更长的上下文 最后，我们在长序列上对 Hyena 的效率进行了基准测试。在长度为 8192 时，相比稠密自注意力（dense self-attention），我们测得了 5 倍的加速——比高度优化的 FlashAttention<sup>2</sup> (Dao et al., 2022b) 快 2 倍——并且在序列长度为 64k 时，相比 FlashAttention 实现了 100 倍的加速，而此时 PyTorch 中的标准注意力实现已耗尽显存。

## 2 预备知识与相关工作

离散卷积是包含两个参数的函数：长度为 $L$ 的输入信号 $u$ 和可学习滤波器 $h$。（可能无限长的）可测<sup>3</sup>滤波器 $h$ 与长度为 $L$ 的输入信号 $u$ 之间的线性（非周期）卷积定义为

$$
y _ {t} = (h * u) _ {t} = \sum_ {n = 0} ^ {L - 1} h _ {t - n} u _ {n}.\tag{1}
$$

通常，$u _ { t } \in \mathbb { R } ^ { D }$，其中 $D$ 是信号的宽度，或者用深度学习术语来说是通道数。不失一般性，我们将分析专门针对单输入单输出（SISO）层，即 $D = 1$。在标准卷积层中很典型的多输入多输出（MIMO）情况可以直接推广得出。

在这种情况下，输入信号可以表示为向量 $u \in \mathbb { R } ^ { L }$，卷积可以表示为输入与由滤波器 $h$ 导出的托普利茨（Toeplitz）核矩阵 $\mathsf { S } _ { h } \in \mathbb { R } ^ { L \times L }$ 之间的矩阵-向量乘积：

$$
(h * u) = \left[ \begin{array}{c c c c} h _ {0} & h _ {- 1} & \dots & h _ {- L + 1} \\ h _ {1} & h _ {0} & \dots & h _ {- L + 2} \\ \vdots & \vdots & \ddots & \vdots \\ h _ {L - 1} & h _ {L - 2} & \dots & h _ {0} \end{array} \right] \left[ \begin{array}{c} u _ {0} \\ u _ {1} \\ \vdots \\ u _ {L - 1} \end{array} \right]\tag{2}
$$

## 2.1 显式与隐式卷积

参数化并优化卷积滤波器 $h _ { t }$ 是深度学习以及更广泛的信号处理领域的标准流程。卷积神经网络（CNN）（Fukushima and Miyake, 1982; LeCun et al., 1998; Ronneberger et al., 2015; He et al., 2016）的经典方法是直接优化滤波器在 $M$ 个指定步长上的响应值 $h _ { t }$，我们称这种参数化方式为显式的。$M$ 被称为滤波器大小，并且通常远短于输入序列长度 $M \ll L$。这类滤波器在信号处理中被称为有限冲激响应（FIR）。

FIR 滤波器具有局部性，最多只能捕获相隔 M 个时间步的输入之间的依赖关系。它们的主要优势在于速度快，复杂度为 (ML)。然而，FIR 滤波器的参数量随滤波器大小呈线性增长，这在计算上可能会带来过高的开销。为了将参数量与滤波器大小解耦，我们可以转而将滤波器 $h _ { t }$ 表示为时间步 t 的参数化函数，即 $h _ { t } = \gamma _ { \theta } ( t )$，其中 $\theta$ 是函数 $\gamma _ { \theta }$ 的参数。这种参数化方式被称为隐式参数化。函数族 $\gamma _ { \theta }$ 的选取属于一种设计选择，对该层的表达能力和计算复杂度具有重大影响。

隐式参数化的一种选择是将 h 选取为线性状态空间模型（State-Space Model, SSM）（Chen, 1984）的响应函数，该模型由以下一阶差分方程描述：

$$
x _ {t + 1} = \mathsf {A} x _ {t} + \mathsf {B} u _ {t} \qquad \mathrm{stateequation}
$$

$$
y _ {t} = \mathsf {C} x _ {t} + \mathsf {D} u _ {t} \quad \mathrm{outputequation}
$$

在此，设定 $x _ { 0 } = 0$ 这一便利的选择将输入-输出映射转化为一个简单的卷积

$$
y _ {t} = \sum_ {n = 0} ^ {t} \left(\mathsf {C A} ^ {t - n} \mathsf {B} + \mathsf {D} \delta_ {t - n}\right) u _ {n}
$$

其中 $\delta _ { t }$ 表示克罗内克 delta 函数（Kronecker delta）。由此我们可以将滤波器 h 确定为

$$
t \mapsto h _ {t} = \left\{ \begin{array}{l l} 0 & t <   0 \\ \mathsf {C A} ^ {t} \mathsf {B} + \mathsf {D} \delta_ {t} & t \geq 0 \end{array} \right.
$$

其中 A、B、C 和 D 的元素即为该滤波器的学习参数。在层设计方面，SSM 的自由度在于状态维度和矩阵的结构。SSM 是展示具有次线性参数量的长卷积如何改进长序列深度学习模型的典型范例（Gu et al., 2020, 2021）。其他隐式方法包括将滤波器参数化为从 t（或 t 的位置编码）到滤波器响应的映射，即 $\gamma _ { \theta } : t \mapsto h _ { t } = \gamma _ { \theta } ( t )$，例如采用前馈神经网络（Romero et al., 2021b,a）。

长卷积与记忆：衡量单个计算单元记忆的一个粗略指标是，为了在某一时间步生成输出，它能在多大程度上访问过去的信息。这可以通过 $n = 0 , \ldots , t$ 时非零项 $\partial y _ { t } / \partial u _ { t - n }$ 的数量来粗略量化。CNN 滤波器的记忆等价于滤波器大小 M，因为 $\partial y _ { t } / \partial u _ { t - n } = h _ { n }$。因此，全卷积 CNN 的总记忆容量与模型的参数量成正比。另一方面，隐式参数化使我们能够将每个滤波器的记忆与参数量解耦，其中滤波器的长度由学习到的参数隐式控制。在 SSM（状态空间模型，State Space Model）中，$\partial y _ { t } / \partial u _ { t - n } = \mathsf { C A } ^ { n } \mathsf { B }$，记忆范围仅由 A 的谱半径决定，并可通过训练过程进行精细微调<sup>a</sup>。另一方面，参数数量控制着记忆单元的表达能力，例如构成 $h _ { t }$ 的基函数数量。

<sup>a</sup>参见例如 Gu et al. (2020, 2021)

卷积的快速方法 Cooley-Tukey 快速傅里叶变换（FFT）算法的最早应用之一就是实现比直接计算 (1) 更快的卷积。乍看之下，(1) 具有 $O ( L ^ { 2 } )$ 的渐进时间复杂度。在次二次（subquadratic）时间内实现快速长卷积的常用方法是通过 FFT 算法。该方法首先通过对输入和滤波器序列进行适当的零填充，将非周期卷积转化为循环卷积 Selesnick and Burrus (2017)。所得的核 $\hat { \mathsf { S } } _ { h }$ 是一个循环矩阵，并由离散傅里叶基对角化

$$
\hat {\mathsf {S}} _ {h} = \mathsf {W} ^ {- 1} \mathsf {D} _ {H} \mathsf {W}
$$

其中 W 是 DFT 矩阵，$\mathsf { W } _ { t t ^ { \prime } } = z ^ { - t } , z = e ^ { i 2 \pi t ^ { \prime } / L }$，而 H 是填充后滤波器 $h$ 的 DFT，即 $H = \mathsf { W p a d } ( h )$。因此，此类卷积的计算如下进行

$$
\begin{array}{r l} \mathsf {p a d} (y) & = \hat {S} _ {h} \mathsf {p a d} (u) \\ & = W ^ {- 1} D _ {H} W \mathsf {p a d} (u) \\ & = \mathsf {i F F T} (D _ {H} \mathsf {F F T} (\mathsf {p a d} (u))) \end{array}
$$

其中 $\mathsf { D } _ { H }$ 是对角线上为 Wh 的矩阵。上述内容被称为 DFT 的卷积定理（Oppenheim et al., 1997）。在这种 FFTConv 形式下，卷积可以在无需显式具象化算子 $\mathsf { S } _ { h }$ 的情况下进行，其渐进成本与 FFT 相同，均为 $O ( L \log _ { 2 } L )$。

![](images/bb7df1fa7e029a4c5c0e957b60c03e18245ded66b0dc62af68cd8579962a26a5.jpg)  
Figure 2.1: 数据控制矩阵之间的比较：SelfAttention 与 Hyena。

## 2.2 自注意力算子

Transformer 的核心是多头注意力（multi-head attention，MHA）机制。给定一个长度为 L 的序列 $\boldsymbol { u } \in \mathbb { R } ^ { L \times D }$，缩放自注意力（Vaswani et al., 2017）的每个头都是一个从 $\mathbb { R } ^ { L \times D }$ 到 $\mathbb { R } ^ { L \times D }$ 的映射，执行以下操作

$$
\begin{array}{c} \mathsf {A} (u) = \mathsf {S o f t M a x} \left(\frac {1}{\sqrt {D}} u \mathsf {M} _ {q} \mathsf {M} _ {k} ^ {\top} u ^ {\top}\right) \\ y = \mathsf {S e l f A t t e n t i o n} (u) \\ = \mathsf {A} (u) u \mathsf {M} _ {v}, \end{array}\tag{3}
$$

其中 $\mathsf { M } _ { q } , \mathsf { M } _ { k } , \mathsf { M } _ { v } \in \mathbb { R } ^ { D \times D }$ 是可学习的线性投影，且 SoftMax 旨在按行应用。注意力机制参数化了一族稠密线性算子，对于输入 $u$ ，通过 u 的投影即 $\mathsf { A } ( u )$ 对其进行索引。我们将此类算子称为数据控制（data-controlled）算子，因为它们编码了一个线性变换 $u \mapsto y$，然而该变换却由 u 非线性地定义。这种方法在 u 中产生了表达能力强的非线性算子，并且我们推测，它与其他机制（Olsson et al., 2022）共同促成了某些算子进行上下文学习（in-context learning）的能力，即通过利用上下文来适应未见任务。在深度学习中，这些投影具有特定的名称：查询（query）$q = u \mathsf { M } _ { q }$、键（key）$k = u \mathsf { M } _ { k }$ 和值（value）$v = u \mathsf { M } _ { v }$。我们通常将注意力算子重写为 $y = \mathsf { A } ( q , k ) v$

注 2.1（Remark 2.1）。与隐式卷积类似，SelfAttention 获取远距离信息的能力并不与参数量相绑定：它以 $\mathcal { O } ( L ^ { 2 } )$ 次操作为代价审视整个序列。

次二次算子（Subquadratic Operators） 现有替代注意力的次二次方法可以总结为改变数据控制的实现方式，即该算子如何由 $u$ 非线性定义，然后应用于 v。例如，一层 Attention-Free Transformers（AFT）（Zhai et al., 2021）通过门控与 SoftMax 的组合（AFT full）或门控与单个显式卷积的组合（AFT conv）来构建该算子。Gated State Spaces（GSS）则通过门控以及由 SSM 参数化的长卷积来组合该算子。沿着这一思路更进一步，受 GSS 在关联回忆（associative recall）上缺陷的启发，Hungry Hungry Hippo（H3）（Dao et al., 2022c）扩展了该机制，加入了一个额外的门控以及通过移位 SSM 获得的短卷积。Hyena 通过引入门控和隐式长卷积的高效计算递推，推广了这一系列工作。

## 3 Hyena：定义与性质

在本节中，我们定义了 Hyena，这是一类由乘法门控交互和长卷积的递推组成的数据控制算子（data-controlled operators）。我们并非寻求对注意力机制（attention）的近似，而是通过有意融入注意力机制的关键计算特性来指导我们的设计，包括序列长度与参数量的解耦。

## 3.1 Hyena 递推

在高层次上，Hyena 包含以下步骤（为清晰起见，设 D = 1）：

i. 与注意力机制类似，计算输入的一组 N + 1 个线性投影。投影的数量 $( v _ { t } , \bar { x _ { t } ^ { 1 } } , \ldots , x _ { t } ^ { N } )$ 不一定为 3。其中一个投影充当值（value）的角色，使得线性输入-输出函数可以针对某个 H(u) 定义为 $y = \mathsf { H } ( u ) v$。

ii. 矩阵 $\mathsf { H } ( u )$ 是通过交替进行隐式长卷积和逐元素与一个投影 $x ^ { i }$ 相乘来定义的，每次处理一个投影，直至用尽所有投影。$\mathsf { H } ( u ) v$ 的计算是在无需显式实例化 H(u) 的情况下高效完成的。通过这种方式，我们将数据控制算子隐式定义为矩阵的分解。构成 H(u) 的长卷积采用隐式参数化，以在序列长度上保持次线性的参数扩展规模。

接下来，我们从其计算模型开始，正式定义 Hyena。我们将其数据控制矩阵形式的分析留到本节的后半部分。

定义 3.1（N 阶 Hyena 算子）。设 $( v , x ^ { 1 } , \cdot \cdot \cdot , x ^ { N } )$ 为输入的投影，并设 $h ^ { 1 } , \ldots , h ^ { N }$ 为一组可学习滤波器。Hyena<sub>N</sub> 算子由以下递推式定义：

$$
\begin{array}{c} {z _ {t} ^ {1} = v _ {t}} \\ {z _ {t} ^ {n + 1} = x _ {t} ^ {n} (h ^ {n} * z ^ {n}) _ {t} \quad n = 1, \ldots , N} \\ {y _ {t} = z _ {t} ^ {N + 1}} \end{array}\tag{4}
$$

注 3.1。Hyena 递推的时间复杂度为 $\mathcal { O } ( N L \log _ { 2 } L )$。输入-输出映射可以重写为

$$
y = x ^ {N} \cdot (h ^ {N} * (x ^ {N - 1} \cdot (h ^ {N - 1} * (\dots))))
$$

其中每个卷积都是通过傅里叶域以 $\mathcal { O } ( L \log _ { 2 } L )$ 的复杂度进行的

有趣的是，时域中的逐元素乘积对应于频域中的卷积，即

$$
x _ {t} u _ {t} = (\hat {x} * \hat {u}) _ {t},
$$

其中 x,ˆ uˆ 分别表示 x 和 $u ,$ 的 DFT。因此，Hyena 交替在时域和频域中应用卷积（或者说交替在时域和频域中应用逐元素相乘）。对于该过程有效性的一种可能解释是：时域中的卷积（频域中的逐元素乘法）增加了记忆长度，从而能够考虑更广泛的上下文。另一方面，时域中的逐元素乘法（频域中的卷积）允许对信号的特定频率成分进行更细粒度的选择。

## 3.2 Hyena 矩阵

Hyena 算子建立在 (Dao et al., 2022c) 提出的 H3 机制之上。为了表述清晰，我们再次考虑单输入单输出（SISO）情况 $( D = 1 )$ 。令 $\mathsf { D } _ { q }$ 和 $\mathsf { D } _ { k }$ 为 $L \times L$ 对角矩阵，其主对角线元素分别为 $q$ 和 k 的对应元素。H3 通过包含四项的数据受控、参数化分解来实现代理注意力矩阵：

$$
\begin{array}{c} \mathsf {A} (q, k) = \mathsf {D} _ {q} \mathsf {S} _ {\psi} \mathsf {D} _ {k} \mathsf {S} _ {\varphi} \\ \mathsf {H 3} (q, k, v) = \mathsf {A} (q, k) v \end{array}\tag{5}
$$

其中 $\mathsf { S } _ { \varphi } , \mathsf { S } _ { \psi }$ 是通过 $\mathrm { S S M s ^ { 4 } }$ 参数化的可学习因果滤波器 $\varphi ,$ ψ 的托普利茨（Toeplitz）矩阵。这些滤波器与 qkv 投影一起构成了我们在层设计中的自由度。该分解允许仅在 $\mathcal { O } ( L \log _ { 2 } L )$ 时间内（两次 FFT 卷积和两次逐元素乘积）计算 (8)，即

$$
\begin{array}{c} {z _ {t} = k _ {t} (\varphi * v) _ {t}} \\ {y _ {t} = q _ {t} (\psi * z) _ {t}} \end{array}\tag{6}
$$

Hyena 代表了对 $( 8 )$ 的泛化，支持任意数量的投影——不局限于三个——并在卷积中使用隐式自由形式的长滤波器。所得到的递推关系 $\left( 4 \right)$ 也可以表示为矩阵形式 $y = \mathsf { H } ( u ) v$ 。令 ${ \sf D } _ { x } ^ { n } = \mathrm { d i a g } ( x ^ { n } ) \in \mathbb { R } ^ { L \times L }$，且令 $\mathsf { S } _ { h } ^ { n }$ 为对应于滤波器 $h ^ { n }$ 的托普利茨矩阵。最终得到的 Hyena 递推关于 v 是线性的，可以重写为矩阵形式：

$$
y = \mathsf {H} (u) v = \mathsf {D} _ {x} ^ {N} \mathsf {S} _ {h} ^ {N} \dots \mathsf {D} _ {x} ^ {2} \mathsf {S} _ {h} ^ {2} \mathsf {D} _ {x} ^ {1} \mathsf {S} _ {h} ^ {1} v
$$

Figure 2.1 可视化了一个分解示例。

![](images/e58a987861fc4b87513edfe5f5292699e0801546cd8156c2491ab969cb3a82ac.jpg)  
Figure 3.1: [上]：Hyena 算子的长卷积参数化示例，带有衰减窗 $\mathbf { \rho } ( t ) \mathbf { \rho } = \exp \{ - \alpha t \}$。参数 $\alpha$ 在 Hyena 的各个独立通道间进行调整，以正则化滤波器使其具有不同的长度。在实践中，我们为窗口添加了一个偏置项，从而使滤波器在超过由衰减率决定的长度后不会被强制限制为全零。

Remark 3.2 (Hyena 泛化了 H3 与 GSS。). H3 机制（Dao et al., 2022c）对应于 Hyena，而 GSS（Mehta et al., 2022）也是 Hyena，只是长卷积（SSM）采用了特定的参数化选择。

将 H3 机制分析为其代理注意力矩阵<sup>5</sup>的分解 $\mathsf { D } _ { q } \mathsf { S } _ { \psi } \mathsf { D } _ { k } \mathsf { S } _ { \varphi }$，阐明了其与矩阵-向量乘法快速求值算法之间的联系。特别是，将 (8) 推广到任意阶是受到基于蝶形分解（butterfly decomposition）的结构化稠密矩阵快速求值算法（Li et al., 2015; Dao et al., 2019, 2022a）的启发，其中分解的长度与其表达能力（在其可表示的矩阵类别方面）紧密相关。Hyena 算子将数据控制与蝶形分解的一种特例相融合。

Remark 3.3. Hyena 算子具有无界上下文。即它们不会被人为地限制（例如受局部性限制），并且可以通过长卷积学习 $v$ 中任意元素之间的长程依赖关系，我们将在接下来讨论这一点。

## 3.3 Hyena 滤波器

在此，我们提供关于卷积参数化的详细信息。我们将每个 Hyena 算子的滤波器表示为从时间（或空间）域 $t$ 到值 $h _ { t } .$ 的映射，并使用浅层前馈神经网络（FFN）对其进行学习：

$$
h _ {t} = \text { Window } (t) \cdot (\text { FFN } \circ \text { PositionalEncoding }) (t)\tag{7}
$$

该方法建立在神经隐式表示（neural implicit representation）相关文献（Mildenhall et al., 2021; Sitzmann et al., 2020）的基础之上，此类方法已在长卷积层中得到应用（Romero et al., 2021b,a）。(7) 式的一个优势在于实现了滤波器长度与参数开销的解耦。

Hyena 中的滤波器特化 窗函数和位置编码函数用于特化 Hyena 算子中的滤波器，引导它们偏向特定类型。Figure 3.1 给出了一个重要示例：我们选择 Hyena 中至少一个卷积塑造成指数衰减的形式，这呼应了 (Li et al., 2022) 在其他应用中的发现。有趣的是，我们发现长指数衰减滤波器与高频滤波器具有协同效应，因为它们使算子能够在特定步长选择特定输入<sup>6</sup>。与 (Romero et al., 2021b) 类似，我们在 FFN 中使用了高频周期性激活函数（sine）。这使得 (7) 能够学习包含高频成分的滤波器，解决了神经网络的低频偏置问题（Basri et al., 2020）。得益于 FFN，(7) 中的参数化可以近似通过其他方式获得的滤波器，例如 S4 (Gu et al., 2020, 2021)、CKConv (Romero et al., 2021b)、SGConv (Li et al., 2022) 和傅里叶神经算子（Fourier Neural Operator, FNO）(Li et al., 2020)。

保持因果性 因果性对于训练自回归语言模型是必不可少的，目的是使给定位置的输出仅依赖于过去的信息。例如，Transformer 将注意力矩阵掩码为下三角矩阵。对于 Hyena，可以通过参数化因果卷积来保证因果性：

命题 3.1（因果 Hyena）。如果每个滤波器 $h ^ { n }$ , $n = 1 , \ldots , N$ 都是因果的，那么对应的 Hyena $N$ 算子也是因果的。

在实践中，我们不需要约束滤波器 (7) 的学习来确保其数值因果性。如果使用基于 FFT 的卷积算法，我们只需要在 $t = 0 , \ldots , L - 1$ 处评估滤波器，并在进行 FFT 之前将输入和滤波器序列补零至 2L 1 即可。

效率 长卷积模型的一个瓶颈可能是硬件加速器的利用率较低，特别是当它们涉及通过迭代数值方法来具现化滤波器时<sup>7</sup>。对 $7$ 的求值速度很快，因为它仅涉及 FFN 的单次前向传播，并且可以如 Algorithm 2 所示在整个序列长度和 Hyena 算子的所有阶数上并行执行，从而提高了硬件利用率。导致低利用率的另一个来源是 FFT，这也是其他长卷积层共同面临的问题。这一瓶颈可以通过分块（blocking）(Selesnick and Burrus, 2017) 以及优化底层例程 (Dao et al., 2022c) 来部分解决。我们在第 4.5 节对运行时间进行了基准测试。

## 3.4 Hyena 算法

Hyena 的前向传播总结如下。

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
算法 1 投影 (Projection)
输入：输入序列 $u \in R^{L \times D}$ 
1. 在 L 维度上并行：$\hat{z} = \text{Linear}(u)$ , Linear : $R^{D} \to R^{(N+1)D}$ 
2. 在 D 维度上并行：z = DepthwiseConv1d(h, $\hat{z}$ )，h 为短卷积核
3. 重塑并切分 z 为 $x^{1}, \ldots, x^{N}, v$ 。单个元素的维度为 $x^{n} \in R^{D \times L}$ 
输出 $x^{1}, \ldots, x^{N}, v, x^{n}$
</div>

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
算法 2 Hyena 滤波器 (Hyena Filter)
输入：序列长度 L，位置嵌入维度 $D_{e}$ 
1. $t = \text{PositionalEncoding}(L)$ , $t \in R^{L \times D_{e}}$ 
2. 在 N, L 维度上并行：$\hat{h} = \mathbb{F}\mathbb{F}\mathbb{N}(t)$ , $FFN : R^{D_{e}} \to R^{ND}$ , $\hat{h} \in R^{L \times ND}$ 
3. 重塑为 $\hat{h} \in R^{N \times D \times L}$ 
4. $h = \hat{h} \cdot \text{Window}(t)$ , $h \in R^{N \times D \times L}$ 
5. 切分 h 为 $h^{1}, \ldots, h^{N}$ 
输出 $h^{1}, \ldots, h^{N}$
</div>

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
算法 3 Hyena 前向传播 (Forward pass of Hyena)
输入：输入序列 $u \in R^{L \times D}$ ，阶数 N，模型宽度 D，序列长度 L，位置嵌入维度 $D_{e}$ 
1. $x^{1}, \ldots, x^{N}, v = \text{Projection}(u)$ 
2. $h^{1}, \ldots, h^{N} = \text{HyenaFilter}(L, D_{e})$ 
for $n = 1, \ldots, N$ do
    3. 在 $D$ 维度上并行：$v_{t} \leftarrow x_{t}^{n} \cdot \text{FFTConv}(h^{n}, v)_{t}$ 
end for
输出 y = v
</div>

命题 3.2（计算复杂度）。使用 N 阶 Hyena 算子处理输入 $\boldsymbol { u } \in \mathbb { R } ^ { L \times D }$ 的计算开销为

$$
\mathcal {O} (N D L (\log_ {2} L + D))
$$

![](images/db6698012fd54958de72ae3f02367ce35d9b914e033ad8bb76eeba2956a6a9fd.jpg)  
Figure 4.1：在 2 阶 Hyena 算子中，长卷积参数化在关联回忆（associative recall）（%）任务上的基准测试。我们的结果表明，隐式参数化在词表大小（输入中 token 的可能取值数量）和序列长度上的扩展性更具优势。

## 4 实验

## 4.1 缩小在上下文学习（in-context learning）上的差距

我们首先从实证角度阐述 Hyena 设计的动机，包括长卷积参数化（long convolution parametrization）的选择。我们考虑了表 4.1（Table 4.1）中描述的任务套件。我们的评估基于近期关于 Transformer 机制可解释性（mechanistic interpretability）的研究（Elhage et al., 2021; Power et al., 2022; Olsson et al., 2022; Zhang et al., 2022）。近来，联想回忆（associative recall）尤其被成功用于指导 H3 的设计（Dao et al., 2022c）。我们扩展了这些研究中的任务套件，并加入了对每个任务更具挑战性的版本的基准测试。例如，在词表大小仅为 10 的情况下解决联想回忆，可以揭示模型在结构上是否具备执行回忆的能力。而在更长的序列和更大的词表上进行测试，则揭示了其他原本被掩盖的性能差距。

如何参数化长卷积 我们比较了在 2 阶 Hyena 中 $S ^ { 1 }$ 和 $S ^ { 2 }$ 的以下长卷积参数化方法的性能：

• Conv1d：显式卷积（具有固定滤波器大小的常规卷积层）。

• FNO：在频域中显式参数化的滤波器（Li et al., 2020）。

• H3：使用状态空间模型（SSMs）的隐式参数化，特别是标准 S4（Gu et al., 2021）。

• TransferFunc：通过传递函数（transfer functions）进行隐式参数化，这是 SSM 的一种经典系统理论推广<sup>8</sup>

• CKConv：使用 FFN 进行隐式参数化（Romero et al., 2021b）。

<sup>8</sup>传递函数大致对应于 SSM 的频域表示。

Table 4.1: 我们机制设计基准的精选集合。

<table><tr><td>任务</td><td>提示</td><td>目标</td></tr><tr><td>联想回忆（Associative Recall）</td><td>a, 1, b, e, 3, f, b</td><td>e</td></tr><tr><td>多数判别（Majority）</td><td>a, g, g, g, e, f, g</td><td>g</td></tr><tr><td>计数（Counting）</td><td>a, b, b, b, a, c, b</td><td>4</td></tr><tr><td>函数上下文学习（ICL of Functions）</td><td> $x_0$ ,  $f(x_0)$ ,  $\ldots x_n$ </td><td> $f(x_n)$ </td></tr><tr><td>算术（Arithmetic）</td><td>1, 3, 5, +, 6, 8, 3</td><td>8, 1, 8</td></tr></table>

Table 4.2: 在更长序列、词表大小为 30 的联想回忆测试准确率（%）。符号 ✗ 用于标记模型超出内存/显存限制的设置。

<table><tr><td>序列长度</td><td>Hyena</td><td>FlashTransformer</td><td>Transformer</td><td>GSS</td><td>H3</td><td>AFT</td><td>RWKV</td></tr><tr><td>30k</td><td>100.0</td><td>32.4</td><td> $\times$ </td><td>5.3</td><td>8.4</td><td>2.3</td><td>12.4</td></tr><tr><td>64k</td><td>100.0</td><td>26.7</td><td> $\times$ </td><td>2.1</td><td>4.3</td><td>1.2</td><td>6.5</td></tr><tr><td>131k</td><td>97.2</td><td> $\times$ </td><td> $\times$ </td><td>0.1</td><td>0.6</td><td>0.8</td><td>2.3</td></tr></table>

• Hyena：通过 FFN 的隐式参数化（带有如图 3.1（Figure 3.1）所示的指数衰减调制）与短显式滤波器的组合。

所有模型均具有相同的宽度和 2 个层。Figure 4.1 显示，基于 FFN 的隐式方法优于其他长卷积，且在更长的序列和更大的词表规模下差距进一步扩大。我们在每个序列长度和词表规模的设置下分别训练了不同的模型。这一排名与将序列长度与参数量解耦的能力（Hyena、CKConv、TransferFunc、H3）以及表达能力（Hyena、CKConv）密切相关。我们在其他任务上也观察到了类似的趋势。

将序列长度推向极限 接下来，我们评估了在长度为 131k 的超长序列上的关联召回（associative recall）性能。据我们所知，这是首次在如此长度的序列上实证展示无注意力（attention-free）的上下文学习（in-context learning）。如 Appendix A 所示，参数化方案之间的差距进一步扩大，Hyena 相比 CKConv 提升了 80 个百分点。

算子对比 我们重复了关联召回实验，这次是基准测试不同的 2 层模型，而非改变卷积参数化方案：二阶 Hyena、GSS (Mehta et al., 2022)、H3 (Dao et al., 2022c)、AFT-conv (Zhai et al., 2021)、RWKV (Peng, 2021) 以及使用 FlashAttention (Dao et al., 2022b) 的标准 GPT (Brown et al., 2020)。如 Table 4.2 所示，Hyena 是唯一能够解决该任务的算子。我们的结果对“仅有 Transformer 能够完成具有挑战性的上下文学习”这一观点提出了质疑。令人惊讶的是，在 The Pile 上固定序列长度下的模型性能排名与在我们的合成数据集上的综合得分排名一致（Appendix C）。

Hyena 算子与滤波器的通用性 Hyena 算子和滤波器也可以成功应用于语言任务之外。我们在 sequential CIFAR 上进行了实验（其中像素被展平为序列），并使用了为语言定义的相同算子。我们在相同模型大小下达到了标准 S4 (Gu et al., 2021) 的准确率（91%）。在 Section 4.5 和 Appendix A 中，我们讨论了使用 Hyena 进行的更大规模图像分类实验。

## 4.2 语言建模

接下来，我们验证 Hyena 在自回归语言建模上的扩展性（scaling）。我们评估了在 WikiText103 (Table 4.3) 和 The Pile (Table 4.4) 上的困惑度（perplexity）。在 The Pile 上，我们针对 50 亿、100 亿、150 亿 token 训练了不同的模型（不同的运行），并调整了学习率调度器。Hyena 是首个在达到 GPT 质量的同时将总 FLOPs 降低 20%<sup>9</sup> 的无注意力卷积架构。初步的缩放定律（scaling laws）展示在 Figure 4.2 中，汇总了 50 亿、100 亿、150 亿 token 的训练运行结果。每条曲线代表一次不同的训练运行。在 Appendix A 中，我们提供了在 PG-19 长程基准（Rae et al., 2019）上的结果。

## 4.3 下游评估

我们在 SuperGLUE (Wang et al., 2019) 任务上进行了下游评估。我们将 Hyena（训练了 1370 亿个 token）与当前可用的最佳预训练无注意力机制模型 RWKV (Peng, 2021)（训练了 3320 亿个 token）以及同等规模的参考模型 GPTNeo (Black et al., 2021)（训练了 3000 亿个 token）进行了比较。表 4.5 和表 4.6 总结了结果。尽管训练的总 token 数量不到其他模型的一半，Hyena 的表现仍与其他模型相近。我们观察到 Hyena 展现出标准 Transformer 的典型少样本（few-shot）能力，在某些任务（例如 MultiRC）中，当为模型提供额外的提示作为上下文时，其准确率相比零样本（zero-shot）提升了 20% 以上。这种改进在生成任务中更为显著，在这些任务中，额外的提示可以指导模型应该如何回答问题。我们在附录 A 中报告了在 LAMBADA 任务 (Paperno et al., 2016) 上的额外下游评估。

![](images/4d05d1ba23584c52c8405f2e0e43f27202018bd8e85ff76cc4d44a844ca517ee.jpg)  
Figure 4.2: The Pile 上语言模型的初步“标度律”（scaling law）。基于长卷积和门控的本文方法（Hyena，红色）与标准 GPT (Brown et al., 2020)（蓝色）的比较。我们在更小的训练 FLOP 预算下达到了 GPT 的困惑度。

Table 4.3: WikiText103 上的困惑度（使用相同分词器）。∗ 表示来自 (Dao et al., 2022c) 的结果。更深且更瘦的模型（Hyena-slim）取得了更低的困惑度。

<table><tr><td>模型</td><td>困惑度</td></tr><tr><td>Transformer (125M)</td><td>18.6</td></tr><tr><td>Hybrid H3 (125M)</td><td>18.5*</td></tr><tr><td>Performer (125M)</td><td>26.8*</td></tr><tr><td>Reformer (125M)</td><td>25.6*</td></tr><tr><td>AFT-conv (125M)</td><td>28.2</td></tr><tr><td>Linear Attention (125M)</td><td>25.6*</td></tr><tr><td>Hyena-3 (125M)</td><td>18.6</td></tr><tr><td>Hyena-3-slim (125M)</td><td>18.5</td></tr></table>

Table 4.4: 训练至特定总 token 数（例如 50 亿）的模型在 The Pile 上的困惑度（每个 token 总数对应不同的运行）。所有模型均使用相同的分词器（GPT2）。FLOP 计算针对 150 亿 token 的运行。

<table><tr><td>模型</td><td>5B</td><td>10B</td><td>15B</td><td>FLOPs ( $10^{19}$ )</td></tr><tr><td>GPT (125M)</td><td>13.3</td><td>11.9</td><td>11.2</td><td>1.88</td></tr><tr><td>Hyena-2 (153M)</td><td>13.3</td><td>11.8</td><td>11.1</td><td>1.87</td></tr><tr><td>GPT (355M)</td><td>11.4</td><td>9.8</td><td>9.1</td><td>4.77</td></tr><tr><td>Hyena-2 (355M)</td><td>11.3</td><td>9.8</td><td>9.2</td><td>3.93</td></tr></table>

Table 4.5: 小型模型在 SuperGLUE 任务上的零样本准确率（%）。

<table><tr><td>模型</td><td>WSC</td><td>WIC</td><td>RTE</td><td>CB</td><td>MULTIRC</td><td>RECoRD</td><td>BOOLQ</td><td>COPA</td><td>平均值</td></tr><tr><td>GPTNeo (Black et al., 2021)</td><td>27.9</td><td>50.0</td><td>45.1</td><td>41.1</td><td>0.0</td><td>61.7</td><td>62.2</td><td>62.0</td><td>43.8</td></tr><tr><td>RWKV (Peng, 2021)</td><td>13.4</td><td>52.3</td><td>46.9</td><td>25.0</td><td>0.0</td><td>58.5</td><td>59.2</td><td>66.0</td><td>40.2</td></tr><tr><td>Hyena</td><td>21.2</td><td>50.5</td><td>46.6</td><td>39.3</td><td>1.1</td><td>59.4</td><td>51.8</td><td>70.0</td><td>41.5</td></tr></table>

Table 4.6: 小型模型在 SuperGLUE 任务上的少样本（3-shot）准确率（%）。

<table><tr><td>模型</td><td>WSC</td><td>WIC</td><td>RTE</td><td>CB</td><td>MULTIRC</td><td>RECoRD</td><td>BOOLQ</td><td>COPA</td><td>平均值</td></tr><tr><td>GPTNeo (Black et al., 2021)</td><td>38.5</td><td>50.0</td><td>53.8</td><td>42.9</td><td>22.4</td><td>61.4</td><td>61.0</td><td>63.0</td><td>49.1</td></tr><tr><td>RWKV (Peng, 2021)</td><td>32.7</td><td>49.4</td><td>47.2</td><td>37.5</td><td>0.0</td><td>58.3</td><td>55.0</td><td>64.0</td><td>43.0</td></tr><tr><td>Hyena</td><td>39.4</td><td>50.1</td><td>47.6</td><td>46.4</td><td>26.7</td><td>58.1</td><td>56.0</td><td>70.0</td><td>49.3</td></tr></table>

![](images/98e7a7dba6b809e7404341747f846afbfa7cb38cb0504bcec5e0e787a0b64ab6.jpg)  
Figure 4.3: 不同序列长度下 Hyena、Attention 与 FlashAttention 的运行时间基准测试。批大小设为 64。右图为左图局部放大图。

## 4.4 基准测试

我们将 2 阶 Hyena 算子的运行时间与注意力（Attention）层以及 FlashAttention 层（Dao et al., 2022b）进行了基准测试对比。Hyena 使用融合 CUDA 核函数来执行 FFTConv（Dao et al., 2022c）。我们将批大小设为 64 并测量运行时间（以毫秒为单位）。结果如 Figure 4.3 所示。在序列长度为 64K 时，Hyena 的加速比达到了 100 倍。Hyena 与注意力的交叉点位于序列长度 2048 处，而 Hyena 与 FlashAttention 的交叉点在 4096 到 8196 之间。尽管 FLOPs 得到了绝对减少，但只有在差距变得足够大时的更长序列上才能实现加速。这是因为 Hyena 的硬件利用率低于 FlashAttention。我们预期，随着 FFTConv 实现的改进以及专用硬件的出现，与理论最大加速比之间的差距将会缩小。

## 4.5 大规模图像分类

最后，我们通过将 Hyena 应用于图像分类，展示了其作为通用深度学习算子的潜力。在 ImageNet 上，我们直接用 Hyena 算子（与其语言版本完全相同，未作修改）即插即用地替换了 Vision Transformer (ViT) (Dosovitskiy et al., 2020) 中的注意力层，并达到了与 ViT 相当的性能。我们还表明，使用更小的图像块（patches）可以同时提升注意力和 Hyena 的性能。由于这会导致更长的序列长度，我们预期随着图像块更加细粒度并接近像素级别，Hyena 将在速度上展现出更大的优势。在 CIFAR-2D 上，我们在标准卷积架构中测试了 Hyena 长卷积滤波器的 2D 版本，相比于 2D 长卷积模型 S4ND (Nguyen et al., 2022)，在获得 8% 速度提升和减少 25% 参数量的同时提高了准确率。有关其他视觉架构和训练过程的详细信息，请参见附录 A.4。

Table 4.7: 图像分类 Top-1 准确率。

<table><tr><td>模型</td><td>图像块大小</td><td>序列长度</td><td>数据集</td><td>ACC (%)</td></tr><tr><td>ViT (87M)</td><td>16x16</td><td>196</td><td>ImageNet-1k</td><td>78.5</td></tr><tr><td>Hyena-ViT (88M)</td><td>16x16</td><td>196</td><td>ImageNet-1k</td><td>78.5</td></tr><tr><td>ViT (87M)</td><td>8x8</td><td>1024</td><td>ImageNet-1k</td><td>80.0</td></tr><tr><td>Hyena-ViT (88M)</td><td>8x8</td><td>1024</td><td>ImageNet-1k</td><td>79.8</td></tr><tr><td>S4ND-ISO (268k)</td><td>-</td><td>-</td><td>CIFAR-10</td><td>89.9</td></tr><tr><td>Hyena-ISO (202k)</td><td>-</td><td>-</td><td>CIFAR-10</td><td>91.2</td></tr></table>

## 5 讨论与结论

在这项工作中，我们为许多大规模语言模型的核心构建块引入了一种无注意力的即插即用替代方案。Hyena 算子是门控机制与隐式参数化长卷积的递归组合，可以在次二次（subquadratic）时间内高效计算，并能在极长序列上进行上下文学习（in-context learning）。在 The Pile 数据集上，深层堆叠的 Hyena 算子构成了首批能够在显著降低训练计算量的同时，匹配 Transformer 困惑度（perplexity）和下游任务性能的无注意力卷积架构之一。我们在十亿以下参数规模上展现出的良好结果表明，注意力机制可能并非我们所需要的一切（attention may not be all we need），而在简单的指导原则启发下并通过机制可解释性基准评估的更简练的次二次设计（如 Hyena），可能会成为构建高效大模型的基础。随着进一步扩展这些模型的规模并优化其推理速度，我们对其所能开辟的新能力感到十分期待。

## 致谢

我们感谢 Karan Goel、Albert Gu、Avanika Narayan、Khaled Saab、Michael Zhang、Elliot Epstein 和 Sabri Eyuboglu 对早期草稿的有益讨论和反馈，并感谢 Together Computer 和 Crusoe 为本论文中模型的训练提供算力支持。我们衷心感谢 NIH（美国国家卫生研究院）项目编号 U54EB020405 (Mobilize)；NSF（美国国家科学基金会）项目编号 CCF1763315 (Beyond Sparsity)、CCF1563078 (Volume to Velocity) 和 1937301 (RTML)；US DEVCOM ARL 项目编号 W911NF-21-2-0251 (Interactive Human-AI Teaming)；ONR（美国海军研究办公室）项目编号 N000141712266 (Unifying Weak Supervision)；ONR N00014-20-1-2480: Understanding and Applying Non-Euclidean Geometry in Machine Learning；N000142012275 (NEPTUNE) 的支持；感谢 NXP、Xilinx、LETI-CEA、Intel、IBM、Microsoft、NEC、Toshiba、TSMC、ARM、Hitachi、BASF、Accenture、Ericsson、Qualcomm、Analog Devices、Google Cloud、Salesforce、Total、HAI-GCP Cloud Credits for Research 项目、Stanford Data Science Initiative (SDSI)、美国国防部（DoD）通过国家防务科学与工程研究生奖学金（NDSEG）项目，以及 Stanford DAWN 项目成员：Facebook、Google 和 VMWare 的支持。本研究得到了 NSF (1651565)、AFOSR (FA95501910024)、ARO (W911NF-21-1-0125)、ONR、DOE (DE-SC0022222)、CZ Biohub 以及 Sloan Fellowship 的资助。尽管本文载有版权声明，美国政府仍获授权出于政府目的复制和分发重印本。本文中所表达的任何观点、发现、结论或建议仅代表作者本人，并不一定反映 NIH、ONR 或美国政府的观点、政策或立场。

## 参考文献

S. Arora, A. Narayan, M. F. Chen, L. J. Orr, N. Guha, K. Bhatia, I. Chami, F. Sala, and C. Ré. Ask me anything: A simple strategy for prompting language models. arXiv preprint arXiv:2210.02441, 2022.

R. Basri, M. Galun, A. Geifman, D. Jacobs, Y. Kasten, and S. Kritchman. Frequency bias in neural networks for input of non-uniform density. In International Conference on Machine Learning, pages 685–694. PMLR, 2020.

S. Black, L. Gao, P. Wang, C. Leahy, and S. Biderman. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow, Mar. 2021. URL https://doi.org/10.5281/zenodo.5297715. If you use this software, please cite it using these metadata.

T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

C.-T. Chen. Linear system theory and design. Saunders college publishing, 1984.

R. Child, S. Gray, A. Radford, and I. Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.

P. Cramer. Alphafold2 and the future of structural biology. Nature structural & molecular biology, 28(9): 704–705, 2021.

E. D. Cubuk, B. Zoph, J. Shlens, and Q. V. Le. Randaugment: Practical automated data augmentation with a reduced search space. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 702–703, 2020.

T. Dao, A. Gu, M. Eichhorn, A. Rudra, and C. Ré. Learning fast algorithms for linear transforms using butterfly factorizations. In International conference on machine learning, pages 1517–1527. PMLR, 2019.

T. Dao, B. Chen, N. S. Sohoni, A. Desai, M. Poli, J. Grogan, A. Liu, A. Rao, A. Rudra, and C. Ré. Monarch: Expressive structured matrices for eficient and accurate training. In International Conference on Machine Learning, pages 4690–4721. PMLR, 2022a.

T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré. Flashattention: Fast and memory-eficient exact attention with io-awareness. arXiv preprint arXiv:2205.14135, 2022b.

T. Dao, D. Y. Fu, K. K. Saab, A. W. Thomas, A. Rudra, and C. Ré. Hungry hungry hippos: Towards language modeling with state space models. arXiv preprint arXiv:2212.14052, 2022c.

A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

N. Elhage, N. Nanda, C. Olsson, T. Henighan, N. Joseph, B. Mann, A. Askell, Y. Bai, A. Chen, T. Conerly, et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021.

K. Fukushima and S. Miyake. Neocognitron: A self-organizing neural network model for a mechanism of visual pattern recognition. In Competition and cooperation in neural nets, pages 267–285. Springer, 1982.

L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster, J. Phang, H. He, A. Thite, N. Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

S. Garg, D. Tsipras, P. Liang, and G. Valiant. What can transformers learn in-context? a case study of simple function classes. arXiv preprint arXiv:2208.01066, 2022.

A. Gu, T. Dao, S. Ermon, A. Rudra, and C. Ré. Hippo: Recurrent memory with optimal polynomial projections. Advances in Neural Information Processing Systems, 33:1474–1487, 2020.

A. Gu, K. Goel, and C. Ré. Eficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021.

K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

D. Hendrycks, N. Mu, E. D. Cubuk, B. Zoph, J. Gilmer, and B. Lakshminarayanan. Augmix: A simple data processing method to improve robustness and uncertainty. arXiv preprint arXiv:1912.02781, 2019.

J. Hofmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

G. Huang, Y. Sun, Z. Liu, D. Sedra, and K. Q. Weinberger. Deep networks with stochastic depth. In European conference on computer vision, pages 646–661. Springer, 2016.

N. Kitaev, Ł. Kaiser, and A. Levskaya. Reformer: The eficient transformer. arXiv preprint arXiv:2001.04451, 2020.

Y. LeCun, L. Bottou, Y. Bengio, and P. Hafner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

Y. Li, H. Yang, E. R. Martin, K. L. Ho, and L. Ying. Butterfly factorization. Multiscale Modeling & Simulation, 13(2):714–732, 2015.

Y. Li, T. Cai, Y. Zhang, D. Chen, and D. Dey. What makes convolutional models great on long sequence modeling? arXiv preprint arXiv:2210.09298, 2022.

Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar. Fourier neural operator for parametric partial diferential equations. arXiv preprint arXiv:2010.08895, 2020.

P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga, Y. Zhang, D. Narayanan, Y. Wu, A. Kumar, et al. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110, 2022.

S. Massaroli, M. Poli, J. Park, A. Yamashita, and H. Asama. Dissecting neural odes. Advances in Neural Information Processing Systems, 33:3952–3963, 2020.

H. Mehta, A. Gupta, A. Cutkosky, and B. Neyshabur. Long range language modeling via gated state spaces. arXiv preprint arXiv:2206.13947, 2022.

B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021.

E. Nguyen, K. Goel, A. Gu, G. W. Downs, P. Shah, T. Dao, S. A. Baccus, and C. Ré. S4nd: Modeling images and videos as multidimensional signals using state spaces. arXiv preprint arXiv:2210.06583, 2022.

C. Olsson, N. Elhage, N. Nanda, N. Joseph, N. DasSarma, T. Henighan, B. Mann, A. Askell, Y. Bai, A. Chen, et al. In-context learning and induction heads. arXiv preprint arXiv:2209.11895, 2022.

A. V. Oppenheim, A. S. Willsky, S. H. Nawab, and J.-J. Ding. Signals and systems, volume 2. Prentice hall Upper Saddle River, NJ, 1997.

D. Paperno, G. Kruszewski, A. Lazaridou, Q. N. Pham, R. Bernardi, S. Pezzelle, M. Baroni, G. Boleda, and R. Fernández. The lambada dataset: Word prediction requiring a broad discourse context. arXiv preprint arXiv:1606.06031, 2016.

B. Peng. RWKV-LM, 8 2021. URL https://github.com/BlinkDL/RWKV-LM.

B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging. SIAM journal on control and optimization, 30(4):838–855, 1992.

A. Power, Y. Burda, H. Edwards, I. Babuschkin, and V. Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv preprint arXiv:2201.02177, 2022.

A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever. Robust speech recognition via large-scale weak supervision. arXiv preprint arXiv:2212.04356, 2022.

J. W. Rae, A. Potapenko, S. M. Jayakumar, C. Hillier, and T. P. Lillicrap. Compressive transformers for long-range sequence modelling. arXiv preprint, 2019. URL https://arxiv.org/abs/1911.05507.

D. W. Romero, R.-J. Bruintjes, J. M. Tomczak, E. J. Bekkers, M. Hoogendoorn, and J. C. van Gemert. Flexconv: Continuous kernel convolutions with diferentiable kernel sizes. arXiv preprint arXiv:2110.08059, 2021a.

D. W. Romero, A. Kuzina, E. J. Bekkers, J. M. Tomczak, and M. Hoogendoorn. Ckconv: Continuous kernel convolution for sequential data. arXiv preprint arXiv:2102.02611, 2021b.

O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234– 241. Springer, 2015.

A. Roy, M. Safar, A. Vaswani, and D. Grangier. Eficient content-based sparse attention with routing transformers. Transactions of the Association for Computational Linguistics, 9:53–68, 2021.

I. Schlag, K. Irie, and J. Schmidhuber. Linear transformers are secretly fast weight programmers. In International Conference on Machine Learning, pages 9355–9366. PMLR, 2021.

I. W. Selesnick and C. S. Burrus. Fast convolution and filtering. In The Digital Signal Processing Handbook, pages 8–1. CRC Press, 2017.

V. Sitzmann, J. N. Martel, A. W. Bergman, D. B. Lindell, and G. Wetzstein. Implicit neural representations with periodic activation functions. arXiv preprint arXiv:2006.09661, 2020.

C. Szegedy, V. Vanhoucke, S. Iofe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2818–2826, 2016.

Z. Tu, H. Talebi, H. Zhang, F. Yang, P. Milanfar, A. Bovik, and Y. Li. Maxvit: Multi-axis vision transformer. arXiv preprint arXiv:2204.01697, 2022.

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008, 2017.

A. Wang, Y. Pruksachatkun, N. Nangia, A. Singh, J. Michael, F. Hill, O. Levy, and S. Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. Advances in neural information processing systems, 32, 2019.

S. Wang, B. Z. Li, M. Khabsa, H. Fang, and H. Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.

L. Yuan, Y. Chen, T. Wang, W. Yu, Y. Shi, Z.-H. Jiang, F. E. Tay, J. Feng, and S. Yan. Tokens-to-token vit: Training vision transformers from scratch on imagenet. In Proceedings of the IEEE/CVF international conference on computer vision, pages 558–567, 2021.

S. Yun, D. Han, S. J. Oh, S. Chun, J. Choe, and Y. Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6023–6032, 2019.

S. Zhai, W. Talbott, N. Srivastava, C. Huang, H. Goh, R. Zhang, and J. Susskind. An attention free transformer. arXiv preprint arXiv:2105.14103, 2021.

H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017.

Y. Zhang, A. Backurs, S. Bubeck, R. Eldan, S. Gunasekar, and T. Wagner. Unveiling transformers with lego: a synthetic reasoning task. arXiv preprint arXiv:2206.04301, 2022.

Z. Zhong, L. Zheng, G. Kang, S. Li, and Y. Yang. Random erasing data augmentation. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 13001–13008, 2020.

## Hyena Hierarchy 补充材料

## 目录

1 引言 1   
2 预备知识与相关工作 3   
2.1 显式与隐式卷积 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   
3 Hyena：定义与性质 5   
3.1 Hyena 递推 5   
3.2 Hyena 矩阵 6   
3.3 Hyena 滤波器 7   
3.4 Hyena 算法 8   
4 实验 9   
4.1 缩小上下文学习差距 9   
4.2 语言建模 10   
4.3 下游任务评估 10   
4.4 基准测试 12   
4.5 大规模图像分类 12   
5 讨论与结论 13   
A 实验细节 18   
A.1 机制设计合成基准 18   
A.2 语言建模 19   
A.3 下游任务评估 21   
A.4 图像分类 21   
B 理论结果与细节 21   
B.1 证明 21   
B.2 数据控制机制分析 22   
C 讨论与额外结果 24   
C.1 学习算术 25   
D 样例与可视化 26   
D.1 Hyena 矩阵 26   
D.2 Hyena 滤波器 31   
D.3 位置编码与滤波器初始化 31   
D.4 下游示例 35

## A 实验细节

Hyena 的实现可以在此链接中找到。

## A.1 机制设计合成基准

我们的合成推理受到机制可解释性（mechanistic interpretability）(Elhage et al., 2021)、上下文学习（in-context learning, ICL）(Garg et al., 2022) 以及语言模型基准测试（Liang et al., 2022）研究的启发。评估围绕 4 个主要任务展开：

• 关联回忆（Associative recall）：每个字符串都是通过拼接来自不同随机字典的键值元组生成的。该测试验证模型在给定键作为提示时，是否能够提取出正确的值，从而有效地应用数据控制的位移（延迟）。

• 多数表决与计数（Majority voting and counting）：测试模型是否能够稠密激活其数据控制矩阵，即通过多个非零项（考虑字符串 'a a a a a a a a a a $b  a ^ { \prime } )$ ）。

• 线性函数的上下文学习（ICL of linear functions）：验证模型是否能对实值输入执行上下文学习（ICL）。提示生成形式为 $x _ { 1 } , w ^ { k } x _ { 1 } , \ldots , x _ { n } \to w ^ { k } x _ { n }$，其中 $x _ { k }$ 和 $w ^ { k } \in R ^ { n _ { o } }$ 均采样自正态分布。

## • 算术：基础能力检验。

针对每个任务，我们使用 Table A.1 中所示的超参数训练模型。我们考虑了由序列长度控制的递增难度设置，跨越 1024、2048、4098、8196、16392、32784、65568、131136 等数值，以及 10、20、30、40 的词表大小。对于函数的 ICL，我们改为改变维度 $n _ { o }$。

注意，对于更长序列上的关联召回（associative recall），提示中会出现键值元组的多个副本。为了理解这一点，可以考虑在词表大小为 40 的情况下，采样特定键值对的多个副本以构成 100k 字符的序列的可能性有多大。能够回溯更长序列距离的模型实际上能看到更多数据，从而能够解决更具挑战性的上下文学习任务版本。增大词表大小会增加每个提示中相同键值对实例之间的平均距离，从而凸显出不同方法之间的性能差距。

Table A.1: （推理与上下文学习任务的超参数设置。）

<table><tr><td>优化器</td><td>AdamW</td></tr><tr><td>优化器动量</td><td> $\beta_1, \beta_2 = 0.9, 0.98$ </td></tr><tr><td>基础学习率</td><td>0.0005</td></tr><tr><td>权重衰减</td><td>0.1</td></tr><tr><td>Dropout</td><td>无</td></tr><tr><td>批大小</td><td>32</td></tr><tr><td>训练轮数</td><td>200</td></tr><tr><td>样本数</td><td>2000</td></tr><tr><td>学习率调度</td><td>余弦衰减</td></tr><tr><td>预热轮数</td><td>10</td></tr><tr><td>预热调度</td><td>线性</td></tr><tr><td>层数</td><td>2</td></tr><tr><td>宽度</td><td>64</td></tr></table>

长卷积对比：我们对比了不同的卷积参数化方法，并将它们嵌入到一个 2 阶 Hyena 算子中。所有卷积均分别应用于输入通道（在信号处理中称为单输入单输出 (SISO)，或在其他机器学习语境中称为深度可分离 (depthwise)）。

• Conv1d：显式卷积（具有固定滤波器大小的常规卷积层）。我们使用大小为 64 的固定滤波器，以匹配其他方法的参数。

• FNO：在频域中显式参数化的滤波器（Li et al., 2020）。我们将模态数设置为 64。

• H3：使用状态空间模型（SSMs）的隐式参数化，特别是标准 S4（Gu et al., 2021）。我们将状态维度设置为 64。

• TransferFunc：通过传递函数（transfer function）进行的隐式参数化，这是状态空间模型（SSM）在系统理论上的经典推广。传递函数由多项式之比定义（我们对系数进行参数化，并通过 FFT 高效评估多项式）。我们将阶数设为 64。

• CKConv：使用 FFN 的隐式参数化（Romero et al., 2021b）。

• item Hyena：通过 FFN 的隐式参数化（带有如图 3.1 所示的指数衰减调制）与短显式滤波器的组合。

CKConv 和 Hyena 使用相同大小的 FFN（宽度为 32，以匹配参数量）。

在 Table A.1 中，我们报告了在序列长度 131072 和词表大小 30 这一具有挑战性的设置下的额外结果。卷积的隐式参数化在关联回忆（associative recall）上优于显式参数化，CKConv 和 Hyena 极大地提高了从不同输入中提取正确键-值（key, value）关系的能力。在 Appendix C 中，我们讨论了合成任务上的结果如何能预示更大规模下的性能。

Table A.2：在序列长度 131072、词表大小 30 的关联回忆任务上的测试准确率（%）。

<table><tr><td>Hyena</td><td>CKConv</td><td>TransferFunc</td><td>H3</td><td>FNO</td><td>Conv1d</td></tr><tr><td>97.2</td><td>14.3</td><td>0.5</td><td>0.6</td><td>0.3</td><td>0.5</td></tr></table>

算子比较：我们在相同的关联回忆任务上比较不同模型，使用 Table A.1 中的超参数。Hyena 对长卷积采用带衰减加窗的滤波器参数化，并在稠密输入投影后采用大小为 3 的短显式卷积。所有其他模型均使用其最大规模实验中的默认设置，同时保持层数为 2 层、宽度为 64。

关于 Transformer 性能的说明：只要序列长度未导致显存溢出，并且训练数据中存在足够的样本，Transformer 就可以解决较长序列的关联回忆任务。在我们的所有实验中，我们将样本数量保持固定（2000），在这种情况下 Transformer 很难找到具有泛化能力的解（见 Table A.1）。

对于较短的序列（见 Appendix C），即使在数据有限的情况下，Transformer 也能轻松解决该任务，表现与 Hyena 相当。

更广泛地说，注意力机制与无注意力（attention-free）词元混合层（token-mixing layer）的这些不同特性，或许可以解释当它们在混合架构中结合时能带来性能提升（Dao et al., 2022c）。本工作的重点是寻找一种无需注意力机制即可保持良好性能的架构，这对于应对普遍存在长序列的领域是必不可少的。然而，当使用较短序列（最多 8k）进行训练时，如果最终的下游性能是唯一关心的指标，那么通过类似 H3（Dao et al., 2022c）的方式对我们的模型进行混合，可以获得更好的结果。

## A.2 语言建模

WikiText103：我们在 WikiText103 上训练了 125M 参数的模型，并将困惑度（perplexity）与 Transformers、诸如 H3（Dao et al., 2022c）等混合模型以及其他次二次注意力（subquadratic attention）变体进行了比较。所有模型均使用相同的 GPT2 分词器（tokenizer），词表大小为 50257。我们测试了 3 阶 Hyena，其中对两个长卷积采用了我们提出的滤波器参数化方案，并在第三个卷积上采用了较短的显式卷积。我们还考虑了比 Transformers 深 1.5 倍（12 层对比 18 层）的 Hyena (slim)，其 FFN 的宽度乘数设为 2。我们发现，用宽度换取深度通常是有利的。这些修改之所以可行，是因为与自注意力相比，Hyena 算子的整体 FLOPs 有所减少，特别是非参数 FLOPs，其中包括注意力矩阵的实例化、softmax 的应用以及矩阵-值约简。

Table A.3: The $\mathrm { P I L E }$（125M）的超参数设置。

<table><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Optimizer momentum</td><td> $\beta_{1},\beta_{2}=0.9,0.98$ </td></tr><tr><td>Peak learning rate</td><td>0.0006</td></tr><tr><td>Warmup learning rate init</td><td>0.000001</td></tr><tr><td>Learning rate min</td><td>0.00006</td></tr><tr><td>Weight decay</td><td>0.1</td></tr><tr><td>Dropout</td><td>None</td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Learning rate schedule</td><td>cosine decay</td></tr><tr><td>Warmup schedule</td><td>linear</td></tr></table>

The Pile：我们遵循相同的流程，在 The Pile（Gao et al., 2020）上训练了 125M 和 355M 规模的模型。超参数如 Table A.3 所示。除了将峰值学习率降低到 $4 \cdot 1 0 ^ { - 4 } $ 之外，355M 的超参数与 125M 相同。对于更大的模型（1.3B），我们将学习率设置为 $2 . 2 \cdot 1 0 ^ { - 4 }$。

我们针对每种模型类型和大小进行了三次实验，在序列长度为 2024、全局批大小为 256 的条件下分别训练了 50 亿、100 亿和 150 亿个 token。所有模型均在包含 8 块 A100 80GB GPU 的单节点上进行训练。我们使用 2 阶 Hyena，并采用了上述针对 WikiText103 所描述的相同架构考量。除了在 50 亿、100 亿和 150 亿 token 上的数据扩展实验外，我们还提供了 1.3B 参数规模模型的初步结果（在 50 亿 token 后困惑度达到 10.8），并训练了一个 153M 的模型（1300 亿 token），困惑度达到 9.8。该 153M 模型与我们在 SuperGLUE 下游评估中使用的模型相同。

训练超参数与标准 GPT 训练流程相匹配，因此对于像 Hyena 这样新的无注意力架构而言可能并非最优。我们进行了一些初步实验，发现例如对目前包含线性预热和余弦衰减的学习率调度进行某些修改，可以提高 Hyena 模型收敛时的困惑度（我们建议与类似规模的 GPT 相比，Hyena 模型采用略低的学习率）。尽管有这些发现，我们对 GPT 和 Hyena 均采用了标准的 GPT 超参数。

PG-19 我们还报告了在其他数据集上的额外训练运行结果。我们在标准 PG-19 长距离语料库（Rae et al., 2019）上训练了一个 Hyena 153M 模型，上下文长度为 16k tokens，在 8 个 epoch 内达到了 14.6 的测试困惑度（使用标准 GPT2 分词器）。

架构 Hyena 的架构超参数如 Table A.4 所示。我们使用正弦（sine）函数作为 Hyena 滤波器的 FFN 激活函数。

Table A.4: Hyena 架构超参数。

<table><tr><td>大小</td><td>深度</td><td>宽度</td><td>FFN 宽度</td><td>滤波器 FFN 宽度</td><td>滤波器 FFN 深度</td><td>正弦频率</td></tr><tr><td>125M</td><td>12</td><td>768</td><td>3072</td><td>64</td><td>4</td><td>14</td></tr><tr><td>125M-slim</td><td>18</td><td>768</td><td>1536</td><td>64</td><td>4</td><td>14</td></tr><tr><td>153M</td><td>18</td><td>864</td><td>1728</td><td>64</td><td>4</td><td>14</td></tr><tr><td>355M</td><td>36</td><td>1024</td><td>2048</td><td>64</td><td>4</td><td>14</td></tr><tr><td>1.3B</td><td>36</td><td>2048</td><td>4096</td><td>64</td><td>4</td><td>14</td></tr></table>

FLOP 计算 正文中报告的浮点运算次数（FLOPs）采用与 (Hofmann et al., 2022) 相同的策略进行计算。对于 GPT，我们不使用近似计算，而是采用基于单层 FLOP 计数的更精确公式。对于 Hyena，FLOPs 使用相同的方法计算，但将注意力层替换为：

i. 投影（Projections）：order  d\_model $\times \mathrm { ~ d ~ } _ { - }$ \_model  seq\_len。

ii. 投影上的短卷积（Short conv on projections）：order $\times \mathrm { ~ d ~ } _ { - }$ \_model  seq\_len  filter\_len（通常为 3）。

<sup>iii.</sup> <sup>FFTConv:</sup> <sup>5</sup> × <sup>(order</sup> <sup>-</sup> <sup>1)</sup> × <sup>d\_model</sup> × <sup>log(seq\_len)</sup> × <sup>seq\_len.</sup>

iv. 输出（Output）：d\_model d\_model seq\_len。

其中乘以前置系数 2，以同时计入加法和乘法。

## A.3 下游评估

SuperGLUE：我们使用 (Arora et al., 2022) 的解析流程在 SuperGLUE（Wang et al., 2019）上评估模型。除 WIC、CB 和 BoolQ 外的所有任务，我们都使用贪婪解码生成回复，然后对照真实标签进行检查。WIC、CB 和 BoolQ 使用 logit 评分而非生成式评估。

模型 所考虑的模型包括在 The Pile 上训练了 300B tokens 的 GPTNeo 125M 开源检查点，以及在 The Pile 上训练了 332B tokens 的 RWKV-v4 169M 检查点。Hyena 是一个在 The Pile 上训练了 137B tokens 的 153M 模型。

LAMBADA：我们在 LAMBADA（Paperno et al., 2016）任务上评估 Hyena。我们应用了停用词过滤器，并检查对应最后一个单词的所有 token 的预测是否与真实标签一致。在 137B tokens 上训练的小型 Hyena 模型达到了 44.64% 的准确率。

## A.4 图像分类

a ImageNet：我们使用由 1000 个类别和 130 万张图像组成的 ImageNet-1k，并在无外部数据的情况下使用 8 块 Nvidia A100 GPU 从头开始训练。在我们的 ViT 基准测试中，我们将注意力层替换为语言实验中定义的 Hyena 算子，并移除了类别标记（class token）和位置嵌入（positional embeddings），类似于 S4ND（Nguyen et al., 2022）。参数量保持相近，ViT-B（base）为 87M，而 Hyena-ViT 为 88M。我们采用了 T2T-ViT（Yuan et al., 2021）的训练流程，包括 RandAugment（Cubuk et al., 2020）、Mixup（Zhang et al., 2017）和 AugMix（Hendrycks et al., 2019）等数据增强技术。所使用的超参数设置参见 Table A.5。

CIFAR-10：我们在序列实验和 2D 实验中使用了 CIFAR-10。对于序列实验，我们使用在语言任务中定义的 Hyena 算子，并通过替换残差块中的层，与同等规模的 S4 模型（Gu et al., 2021）进行比较。在 2D 实验中，我们学习与输入形状大小相同的 Hyena 滤波器（在 x 和 y 维度上），并舍弃了语言实验中的门控机制。我们通过衰减项对 Hyena 滤波器进行加窗（即在空间上应用软掩码）。衰减速率在不同通道间各不相同，以确保滤波器在初始化时具有不同的大小。我们通过将模型层替换为 2D Hyena 滤波器，与另一种隐式 2D 卷积 S4ND（Nguyen et al., 2022）进行了比较。“等距”（isometric）模型由 4 个模型维度为 128 的残差块组成。我们使用了基本图像增强、0.1 的 dropout、0.03 的权重衰减，并使用 Nvidia T4 GPU 训练了 100 个 epoch。

## B 理论结果与细节

## B.1 证明

## 命题 3.1 的证明

证明。一个离散的 L-by-L 算子如果是下三角的，则该算子是因果的（causal），即未来输入信息不会泄漏到输出中。Hyena 算子 H 是交替出现的对角矩阵和 Toeplitz 矩阵的乘积。因此，如果所有 Toeplitz 矩阵 $\mathsf { S } _ { h } ^ { n }$ 都是下三角的，那么 H 就是下三角的。反过来，当且仅当滤波器 h 是因果的时，每个 S<sup>n</sup> 才是下三角的，证明完毕。□

Table A.5: ImageNet-1k 的 ViT 与 ViT-Hyena 设置）。

<table><tr><td>图像尺寸</td><td> $224^2$ </td></tr><tr><td>优化器</td><td>AdamW</td></tr><tr><td>优化器动量</td><td> $\beta_1, \beta_2 = 0.9, 0.999$ </td></tr><tr><td>权重初始化</td><td>截断正态分布 (std=0.02)</td></tr><tr><td>ViT 基础学习率</td><td> $1e^{-3}$ </td></tr><tr><td>Hyena-ViT 基础学习率</td><td> $2e^{-4}$ </td></tr><tr><td>ViT 权重衰减</td><td>0.05</td></tr><tr><td>Hyena-ViT 权重衰减</td><td>0.01</td></tr><tr><td>Dropout</td><td>无</td></tr><tr><td>批大小</td><td>1024</td></tr><tr><td>训练轮数</td><td>300</td></tr><tr><td>学习率调度策略</td><td>余弦衰减</td></tr><tr><td>预热轮数</td><td>10</td></tr><tr><td>预热调度策略</td><td>线性</td></tr><tr><td>Randaugment (Cubuk et al., 2020)</td><td>(9,0.5,layers=2)</td></tr><tr><td>Mixup (Zhang et al., 2017)</td><td>0.8</td></tr><tr><td>Cutmix (Yun et al., 2019)</td><td>1.0</td></tr><tr><td>随机擦除 (Zhong et al., 2020)</td><td>0.25</td></tr><tr><td>标签平滑 (Szegedy et al., 2016)</td><td>0.1</td></tr><tr><td>随机深度 (Huang et al., 2016)</td><td>0.1</td></tr><tr><td>指数移动平均 (EMA) (Polyak and Juditsky, 1992)</td><td>无</td></tr></table>

## B.2 数据控制机制分析

我们讨论 Hyena-2 的代理注意力机制（surrogate attention mechanism）：$q , k , v \mapsto y \colon$

$$
\begin{array}{l} z _ {t} = k _ {t} (\varphi * v) _ {t} \\ y _ {t} = q _ {t} (\psi * z) _ {t} \end{array}\tag{8}
$$

如果 $\varphi$ 和 $\psi$ 是通过状态空间模型（SSM）参数化的卷积，上述形式类似于 H3 机制 (Dao et al., 2022c)。我们研究卷积核 $\varphi$ 和 $\psi$ 对注意力层的影响。我们首先引入该层的矩阵表示，并分离出注意力矩阵 $\mathsf { A } _ { \varphi } ^ { \psi } ( q , k )$，使得

$$
y = \mathsf {A} _ {\varphi} ^ {\psi} (q, k) v.\tag{9}
$$

分离代理注意力矩阵 在长度为 L 的离散序列情况下

$$
\begin{array}{l} {z _ {t} = k _ {t} \sum_ {m = 0} ^ {L - 1} \varphi_ {t - m} v _ {m}} \\ {y _ {t} = q _ {t} \sum_ {m = 0} ^ {L - 1} \psi_ {t - m} z _ {m}} \end{array}\tag{10}
$$

因此我们可以将 (8) 重写为

$$
\begin{array}{l l} y _ {t} = q _ {t} \sum_ {m = 0} ^ {L - 1} \psi_ {t - m} k _ {m} \sum_ {n = 0} ^ {L - 1} \varphi_ {m - n} v _ {n} \\ = q _ {t} \sum_ {m = 0} ^ {L - 1} \sum_ {n = 0} ^ {L - 1} \psi_ {t - m} k _ {m} \varphi_ {m - n} v _ {n} & \text {Move \psi , k inside inner sum} \\ = q _ {t} \sum_ {n = 0} ^ {L - 1} \sum_ {m = 0} ^ {L - 1} \psi_ {t - m} k _ {m} \varphi_ {m - n} v _ {n} & \text {Index shift} \\ = \sum_ {n = 0} ^ {L - 1} q _ {t} \sum_ {m = 0} ^ {L - 1} \psi_ {t - m} k _ {m} \varphi_ {m - n} v _ {n} \end{array}\tag{11}
$$

于是我们可以定义代理注意力矩阵 $\mathsf { A } _ { \varphi } ^ { \psi } ( q , k )$

$$
[ \mathsf {A} _ {\varphi} ^ {\psi} (q, k)) ] _ {t, t ^ {\prime}} = q _ {t} \sum_ {m = 0} ^ {L - 1} \psi_ {t - m} k _ {m} \varphi_ {m - t ^ {\prime}}.\tag{12}
$$

连续信号：我们还可以考虑群 $G$ 上的连续信号情况。在连续情况下，我们可以将 (8) 中的卷积展开为

$$
(\varphi * v) _ {t} = \int_ {G} \varphi_ {t - g} v _ {g} \mathrm{d} g, \qquad (\psi * z) _ {t} = \int_ {G} \psi_ {t - g} z _ {g} \mathrm{d} g\tag{13}
$$

这使我们能够将 (8) 重写为

$$
\begin{array}{l l} y _ {t} = q _ {t} (\psi * k (\varphi * v)) _ {t} \\ \quad = q _ {t} \int_ {G} \psi_ {t - g} \left[ k _ {g} \int_ {G} \varphi_ {g - \tau} v _ {\tau} \mathrm{d} \tau \right] \mathrm{d} g \\ \quad = q _ {t} \int_ {G} \left[ \int_ {G} \psi_ {t - g} k _ {g} \varphi_ {g - \tau} v _ {\tau} \mathrm{d} \tau \right] \mathrm{d} g \\ \quad = q _ {t} \int_ {G} \left[ \int_ {G} \psi_ {t - g} k _ {g} \varphi_ {g - \tau} v _ {\tau} \mathrm{d} g \right] \mathrm{d} \tau & \text { Variable   swap } \\ \quad = \int_ {G} \left[ q _ {t} \int_ {G} \psi_ {t - g} k _ {g} \varphi_ {g - \tau} v _ {\tau} \mathrm{d} g \right] \mathrm{d} \tau & \text { Pull   } q _ {t} \text {   in   } \tau \text {   integral } \\ \quad = \int_ {G} \left[ q _ {t} \int_ {G} \psi_ {t - g} k _ {g} \varphi_ {g - \tau} \mathrm{d} g \right] v _ {\tau} \mathrm{d} \tau & \text { Pull   } v _ {\tau} \text {   out   of   } g \text {   integral }. \end{array}\tag{14}
$$

存在一个线性算子 $\mathcal { A } : v \mapsto y = \mathcal { A } v$，我们将其解释为替代注意力算子（surrogate attention operator）。$\mathcal { A }$ 以查询 $q$、键 $k$ 以及滤波器 $\varphi$ 和 $\psi$ 为条件，即 $\mathcal { A } = \mathcal { A } _ { \varphi } ^ { \psi } ( q , k )$。该算子的核由下式给出

$$
\mathcal {K} (t, t ^ {\prime}) = q _ {t} \int_ {G} \psi_ {t - g} k _ {g} \varphi_ {g - t ^ {\prime}} \mathrm{d} g\tag{15}
$$

替代注意力矩阵的算子分解 我们可以将线性映射 $v \mapsto y ; \ y = \mathsf { A } _ { \varphi } ^ { \psi } ( q , k ) v$ 分解为一系列因子，每个因子都依赖于输入的投影 $\mathsf { A } _ { \varphi } ^ { \psi } ( q , k ) =$ $\mathsf { A } ^ { \psi } ( q ) \mathsf { A } _ { \varphi } ( k )$。令 $\mathsf { D } _ { q }$ 和 $\mathsf { D } _ { k }$ 为 $L { \mathrm { - b y - } } L$ 对角矩阵，其主对角线元素分别为 $q$ 和 $k$ 的对应元素。那么，我们有

$$
\begin{array}{l l} \mathsf {A} ^ {\psi} (q) = \mathsf {D} _ {q} \mathsf {S} _ {\psi}, & \mathsf {D} _ {q} = \mathrm{diag} (q), \\ \mathsf {A} _ {\varphi} (k) = \mathsf {D} _ {k} \mathsf {S} _ {\varphi}, & \mathsf {D} _ {k} = \mathrm{diag} (k). \end{array}\tag{16}
$$

该矩阵被分解为两项 $\mathsf { A } ^ { \psi } ( q )$ 和 $\mathsf { A } _ { \varphi } ( k )$，它们分别由对角矩阵 $\mathsf { D } _ { q }$ 和 $\mathsf { D } _ { k }$ 与 Toeplitz（托普利茨）矩阵 $\mathsf { S } _ { \psi }$ 和 $\mathsf { S } _ { \varphi }$ 相乘构建而成。$\mathsf { S } _ { \psi }$ 和 $\mathsf { S } _ { \varphi }$ 分别是滤波器冲激响应为 $\psi$ 和 $\varphi$ 的卷积算子核。在当前感兴趣的应用中，$\psi$ 和 $\varphi$ 被选取为因果的，即当 $t < 0$ 时 $\psi [ t ] = 0$，当 $t < 0$ 时 $\varphi [ t ] = 0$。这使得 $\mathsf { S } _ { \psi }$ 和 $\mathsf { S } _ { \varphi }$ 成为下三角矩阵

$$
\mathsf {S} _ {\psi} = \left[ \begin{array}{c c c c} \psi_ {0} & 0 & \dots & 0 \\ \psi_ {1} & \psi_ {0} & \dots & 0 \\ \vdots & \ddots & \ddots & \vdots \\ \psi_ {L - 1} & \psi_ {L - 2} & \dots & \psi_ {0} \end{array} \right], \qquad \mathsf {S} _ {\varphi} = \left[ \begin{array}{c c c c} \varphi_ {0} & 0 & \dots & 0 \\ \varphi_ {1} & \varphi_ {0} & \dots & 0 \\ \vdots & \ddots & \ddots & \vdots \\ \varphi_ {L - 1} & \varphi_ {L - 2} & \dots & \varphi_ {0} \end{array} \right].\tag{17}
$$

替代注意力矩阵随后可表示为

$$
\mathsf {A} _ {\varphi} ^ {\psi} (q, k) = \mathsf {D} _ {q} \mathsf {S} _ {\psi} \mathsf {D} _ {k} \mathsf {S} _ {\varphi}\tag{18}
$$

在因果滤波器 $\varphi$ 和 $\psi$ 的情况下，我们可以将 (16) 中的矩阵乘法展开为

$$
\begin{array}{c} \mathsf {D} _ {q} \\ \left[ \begin{array}{c c c c} q _ {0} & & & \\ & q _ {1} & & \\ & & \ddots & \\ & & & q _ {L - 1} \end{array} \right] \left[ \begin{array}{c c c c} \psi_ {0} & & & \\ \psi_ {1} & \psi_ {0} & & \\ \vdots & \ddots & \ddots & \\ \psi_ {L - 1} & \psi_ {L - 2} & \dots & \psi_ {0} \end{array} \right] \left[ \begin{array}{c c c c} k _ {0} & & & \\ & k _ {1} & & \\ & & \ddots & \\ & & & k _ {L - 1} \end{array} \right] \left[ \begin{array}{c c c c} \varphi_ {0} & & & \\ \varphi_ {1} & \varphi_ {0} & & \\ \vdots & \ddots & \ddots & \\ \varphi_ {L - 1} & \varphi_ {L - 2} & \dots & \varphi_ {0} \end{array} \right] \\ = \left[ \begin{array}{c c c c} q _ {0} \psi_ {0} & & & \\ q _ {1} \psi_ {1} & q _ {1} \psi_ {0} & & \\ \vdots & \ddots & \ddots & \\ q _ {L - 1} \psi_ {L - 1} & q _ {L - 1} \psi_ {L - 2} & \dots & q _ {L - 1} \psi_ {0} \end{array} \right] \\ \mathsf {A} _ {\psi} (q) \\ \mathsf {A} _ {\varphi} (k) \end{array}\tag{19}
$$

卷积算子的傅里叶分解：卷积算子的核 $\mathsf { S } _ { \psi }$ 和 $\mathsf { S } _ { \varphi }$ 由傅里叶变换矩阵 $\mathsf { W } \in \mathbb { C } ^ { L \times L } , \ \mathsf { W } _ { n m } = z ^ { m } , \ z = e ^ { \hat { j } 2 \pi n / L }$ 对角化。卷积算子 $\mathsf { S } _ { \psi }$ 的傅里叶变换由下式给出

$$
S _ {\psi} = W ^ {*} D _ {\Psi} W, \quad S _ {\Phi} = W ^ {*} D _ {\Phi} W\tag{20}
$$

其中 $\mathsf { D } _ { \Psi } , \mathsf { D } _ { \Phi } \in \mathbb { C } ^ { L \times L }$ 分别是由频率响应（离散傅里叶变换）$\Psi = \mathsf { W } \psi , \Phi = \mathsf { W } \varphi$ 构建的对角矩阵。该分解可用于简化 (19) 中的矩阵乘法：

$$
\mathsf {A} = \mathsf {D} _ {q} \mathsf {S} _ {\psi} \mathsf {D} _ {k} \mathsf {S} _ {\varphi} = \mathsf {D} _ {q} \mathsf {W} ^ {*} \mathsf {D} _ {\Psi} \mathsf {W} \mathsf {D} _ {k} \mathsf {W} ^ {*} \mathsf {D} _ {\Phi} \mathsf {W}\tag{21}
$$

上述公式的一个重要性质是 $\mathsf { D } _ { q }$ 和 $\mathsf { S } _ { k }$ 与 W 的不可交换性。如果这两个算子可交换，我们将得到

$$
\left| \mathsf {A} = \mathsf {D} _ {q} \mathsf {W} ^ {*} \mathsf {D} _ {\Psi} \mathsf {W} \mathsf {D} _ {k} \mathsf {W} ^ {*} \mathsf {D} _ {\Phi} \mathsf {W} = \mathsf {W} ^ {*} \mathsf {D} _ {q} \mathsf {D} _ {\Psi} \mathsf {D} _ {k} \mathsf {D} _ {\Phi} \mathsf {W} \right|\tag{22}
$$

这将使整个层退化为一个简单的卷积。门控项的不可交换性在卷积算子链中起到了非线性的作用。

## C 讨论与补充结果

词表大小扩展性 Table C.1 展示了不同词表大小下的关联回忆（associative recall）性能与 The Pile 上的损失之间有趣的关联。在此实验中，我们将关联回忆的序列长度固定为 2048，这与在 The Pile 上训练所有模型时所使用的序列长度相同。

我们在机制设计基准测试的其他任务切片上也观察到了类似现象，这表明有可能基于 1 或 2 层模型在合成任务上的快速实验，推导出用于预测大规模性能的规律。令人惊讶的是，在我们的语言合成任务上的性能似乎还与在其他领域作为注意力替代方案时的性能进一步相关联（图像分类结果见 Appendix A.4）。

Table C.1: 在词表大小分别为 10、20、30、40 时 Hyena 的关联回忆准确率，及其与在 The Pile 上训练 50 亿 token 后的测试损失之间的关系。我们注意到这两个性能指标之间存在关联，这表明机制设计合成任务的切片可能对大规模性能具有潜在的预测能力。

<table><tr><td>模型</td><td>Acc @ 10</td><td>Acc @ 20</td><td>Acc @ 30</td><td>Acc @ 40</td><td>THE PILE 上的 Loss @ 5B</td></tr><tr><td>Conv1d</td><td>32</td><td>11</td><td>10</td><td>8</td><td>4.21</td></tr><tr><td>AFT-conv</td><td>55</td><td>21</td><td>12</td><td>10</td><td>3.57</td></tr><tr><td>H3</td><td>92</td><td>60</td><td>13</td><td>10</td><td>2.69</td></tr><tr><td>Transformer</td><td>100</td><td>100</td><td>92</td><td>82</td><td>2.59</td></tr><tr><td>Hyena</td><td>100</td><td>100</td><td>98</td><td>85</td><td>2.59</td></tr></table>

![](images/223403a8da2afaf98e46d13a926f2311ffa206de2dd6788bb94893fa3a7f9fe3.jpg)  
Figure C.1: Hyena 在不同位数加法和不同模型深度下的测试损失与准确率。每个子图报告了一个不同实验的结果，曲线追踪了训练期间的测试结果。

单层召回（Single layer recall）我们在合成任务上的所有实验默认使用 2 层模型。我们选择 2 是因为它是基于电路对 Transformer 进行机制分析（Elhage et al., 2021）的标准层数。有趣的是，单层 Hyena（宽度为 64）就能够执行联想召回（associative recall），即使在词表大小为 40 这一具有挑战性的设置下也能完全解决该任务。对单个 Hyena 算子究竟如何能够执行召回进行逆向工程留待未来工作完成。

## C.1 学习算术

我们在机制设计基准中展示了一个额外的任务：学习算术。我们在 $D _ { n } { \mathrm { - d i g i t } }$ 位加法数据集上训练了深度递增（1、2 和 3 层）的 Hyena 模型。例如，一个 3 位数加法输入样本由以下序列给出：

## 1, 2, 3, 9, 5, 4, 1, 0, 7, 7

其中前 6 位数字包含要相加的两个 3 位数，后 4 位是结果。由于我们的模型是因果的，因此使用标准的自回归训练（即预测下一个标记）进行优化。具体而言，我们优化模型以学习一个映射 $x \mapsto y$，其中 x 是不含最后一个元素的原始提示，y 等于 x 向右平移一位。我们对每个序列损失中的前 $2 D _ { n } - 1$ 个元素进行了掩码，因为它们包含对加数的预测而非结果。

我们在 Figure C.1 中报告了结果。单层 Hyena 能够学会执行最多 4 位的加法。更长的数字需要更深的模型。在我们的实验中，AFT-conv 等替代架构难以学会算术，这表明其能力存在上限。

## D 样本与可视化

## D.1 Hyena 矩阵

我们提供了由测试字符串激活的注意力矩阵和 Hyena 矩阵的可视化。在 D.1、D.2 中，我们将 GPTNeo（Black et al., 2021）注意力矩阵与我们预训练的小型 Hyena 模型提取的 Hyena 矩阵进行了比较。在 D.3 和 D.4 中，我们为 355M 模型提供了由不同长度的测试字符串激活的其他 Hyena 矩阵。

对于注意力机制，我们可视化了 Softmax 后的原始矩阵。对于 Hyena 矩阵，我们绘制了 H(u) 的（逐元素）绝对值：

$$
\begin{array}{c} \mathsf {H} (u) = \mathsf {D} _ {x} ^ {N} \mathsf {S} _ {h} ^ {N} \dots \mathsf {D} _ {x} ^ {2} \mathsf {S} _ {h} ^ {2} \mathsf {D} _ {x} ^ {1} \mathsf {S} _ {h} ^ {1} \\ \hat {\mathsf {H}} (u) _ {i j} = | \mathsf {H} (u) _ {i j} | \end{array}
$$

由于 Hyena 没有（例如通过 softmax）对其矩阵的元素进行归一化，因此它与注意力机制（attention）存在显著差异：(1) $\mathsf { H } ( u )$ 的元素可正可负，以及 (2) 幅度不受约束。我们观察到预训练 Hyena 模型中的矩阵幅度大约在 $1 0 ^ { - 3 }$ 左右。

![](images/c9aa164d84b8e26c8e248ee9e69ef8d65ff32423bf5321885f7418207885882b.jpg)  
Figure D.1: 来自 GPTNeo 小型模型的注意力矩阵。我们使用的测试字符串为“Attention is all you need. Attention is”。

![](images/0b61bcacf3ef28fe289c52381581a0e52b10b5b8d4958442abc6f98fb9975719.jpg)  
Figure D.2: 来自 Hyena 小型模型（与用于 SuperGLUE 下游评估的模型相同）的 Hyena 矩阵。我们使用的测试字符串为“Attention is all you need. Attention is”。我们注意到，由于 Hyena 不使用注意力头（heads），因此对于每个通道（即其宽度中的每个维度），Hyena 都有一个不同的数据控制矩阵。

![](images/d9ab70d147cd5687c3a33d2e6351a66d3cff8cdc26ca235d157f5d7815c0e2e4.jpg)  
Figure D.3: 数据控制的 Hyena 矩阵（355M 模型），由字符串“When a doctor doctors a doctor, does the doctor doing the doctoring doctor as the doctor being doctored wants to be doctored or does the doctor doing the doctoring doctor as they want to doctor? ”激活。图中的行表示来自不同层的矩阵，列表示来自不同通道的矩阵。该算子在没有注意力机制的情况下展示了注意力矩阵的特征模式。

![](images/4bdde30d0d84f4e6b091f6e97771a881fdc92df05af207fc239bff0d34df22d5.jpg)  
Figure D.4: 数据控制的 Hyena 矩阵（355M 模型），由来自 Causal scrubbing: results on induction heads 的字符串“Mrs. Dursley, Mr. Dursley, Dudley Dursley”激活。图中的行表示来自不同层的矩阵，列表示来自不同通道的矩阵。

## D.2 Hyena 滤波器

Figure D.5 提供了 Hyena 长卷积滤波器在初始化时以及在 The Pile 上完成训练后的可视化结果。

我们发现不同初始化方案之间存在显著的性能差异（困惑度差异高达 5%）。如果初始化时的滤波器过于平滑（参见附录 D.3 关于位置编码和激活函数的讨论），模型会找到较差的解且需要更长的时间收敛。此外，我们观察到将滤波器正则化为收敛时学习到的典型滤波器的初始化方案会降低性能。这些观察结果与正文和附录 A.1 中讨论的卷积参数化方案之间的性能差距相一致。特别是，通过 Hyena 滤波器获得的性能提升可能是由于在卷积滤波器空间中更容易进行优化。

在收敛时，Hyena 学习到了一组具有相似结构的低阶滤波器，这可用于在训练后进一步加速推理。

## D.3 位置编码与滤波器初始化

Hyena 滤波器所选用的位置编码是截断复指数基。具体而言，对于 $k = 0 , \ldots K - 1$，设 $\overline { { \rho _ { k } ( t ) } } = e ^ { i 2 \pi k t / L }$，位置编码被定义为从 R 到 $\bar { \mathbb { R } ^ { 2 K + 1 } }$ 的映射，使得

$$
\text { PositionalEncoding } (t) = \left[ t \quad \Re [ \rho_ {0} ] (t) \quad \dots \quad \Re [ \rho_ {K - 1} ] (t) \quad \Im [ \rho_ {0} ] (t) \quad \dots \quad \Im [ \rho_ {K - 1} ] (t) \right]
$$

其中 R[ ]、I[ ] 分别表示其参数的实部和虚部。在正文中，我们使用 $D _ { e } = 2 K + 1$ 来表示具有 K 个特征的位置编码的大小。位置编码的特征数量对滤波器的初始化和训练性能有影响。特别是，我们展示了 K 如何在初始化时导致滤波器频谱的预条件化（preconditioning）。图 D.6、D.7、D.8 展示了在 $L = 1 2 8$ 且正弦激活函数 $\sigma ( \cdot ) = \sin ( \omega _ { a } \cdot )$ 的频率 $\omega _ { a }$ 设为 1 时，不同 $K \left( \{ 8 , 3 2 , 6 4 \} \right)$ 值的初始化滤波器（无窗函数）。我们注意到，K 的选择如何在初始化时对建模频率产生偏差。具体来说，这些滤波器类似于截止频率约为 $2 K + 1$ 的低通滤波器。

该截止频率与滤波器的平滑度密切相关；如前所述，我们通过经验观察到，初始化为非平滑（即具有丰富的高频内容）的滤波器具有更好的训练动态。虽然我们可以通过增加 K 来获得良好的初始化，但这会导致更大的 FFN（其输入维度为 2K + 1，即位置编码特征的数量），从而带来更高的参数量。一个更高效的解决方案是增加正弦激活函数的频率 $\omega _ { a }$。图 D.9 展示了在 $K = 8$ 时，我们如何仅通过设置 $\omega _ { a } = 1 0$ 就能覆盖完整频谱。

![](images/1663e21213e4f13dfbec01636046549d9ed8e22b15114488ce0c35ada909f746.jpg)  
Figure D.5: [上]：初始化时的长卷积 Hyena 滤波器（153M 参数，18 层模型）。[下]：在 The Pile 上训练 1300 亿（130 billion）token 后的滤波器。

脉冲响应 h  
![](images/fa4ad5c1620f6e0ba433d52321ad024f0414b90abd78fc95ac3c92de14f4b0fa.jpg)

![](images/93571d6ac3034aeccbba6057348abe17ce68f7b4e1aee408d35a356681d7c979.jpg)

幅度响应 FFT[h]  
![](images/63556d81e304c65d62d39d4f31aad58d9999bebac57bc70b294b9e49f16a1622.jpg)

相位响应 ̸ FFT[h]  
![](images/c9902280d41c01971a79f726678508453eb27431ce34bf99bf04d12e2cf64d1f.jpg)  
Figure D.6: 初始化时具有 17 个位置编码特征 $K = 8 .$ 的 Hyena 滤波器。

![](images/c7f96cab8053494f2b78dda4c7843c28417f41f9b3de95b4ffbaf4554ecc9d33.jpg)

![](images/ffd9fe21f2f71665a52fddad74ca93265c56485d1455cc01e53d26392a12392f.jpg)

![](images/1a75118340cd1c535b70e8cd24b8ea8cceb034b36c2788ecd4a816aac7909997.jpg)

![](images/6cd733384dfa604d317bad0a5adcbc762d39260567283f3d2bff446b42932027.jpg)  
Figure D.7: 具有 65 个位置编码特征 $K = 32$ 时的 Hyena 滤波器初始化状态。

![](images/5cb1f2831d67044f6b572bfe300d5fdebf7ec146ce48c7f78e8184985617c129.jpg)

![](images/0fd4634ac0a560078a25f568b655cb7fa3cc3710e1adccce237268c7d111ddbf.jpg)

![](images/caea2947d2e187b96da33890e4812092d5bf8beed153f11f533a42e0d989d441.jpg)  
Figure D.8: 具有 65 个位置编码特征 $K = 6 4$ 时的 Hyena 滤波器初始化状态

相位响应 ̸ FFT[h]  
![](images/1b30af5bb69a6ef485ca446d685daf04c3e6ed6f96b73a2a8cb7532676e7da77.jpg)

![](images/c7d151b31162a9cf45a955d4436acf2dd93f66f18b699dea435b762a4f35486f.jpg)

![](images/a8bc02fbbf3098ab2302635fa0408150b2fe96fbad1e83b0ac6b54e1f8ff82fb.jpg)

![](images/9de2f3c4e46e9abb254f41a5182903e9a4d61d72a46aabe9ad00f387606ffbbc.jpg)

相位响应 ̸ FFT[h]  
![](images/f39a179f260de99ea90053293393d657301cfc19817bfdc0e2f73a84f0077ae7.jpg)  
Figure D.9: 具有 17 个位置编码特征 $K = 8$ 且正弦激活频率设置为 10 时的 Hyena 滤波器初始化状态。

## D.4 下游示例

MultiRC 我们展示了小型模型在 MultiRC 问答任务上的下游评估示例。我们报告了小型 Hyena（153M，在 The Pile 上训练了 130B 词元）和公开检查点 RWKV-v4 (Peng, 2021)（169M，在 The Pile 上训练了 332B 词元）的回答。我们选取了索引为 2 的幂的随机示例。通过在提示中提供少样本（few-shot）示例，Hyena 的回答与任务格式的对齐程度得到了大幅提升，这对于基于所提出无注意力（attention-free）架构的更大规模模型而言可能是一个令人期待的迹象。每个示例展示了验证集中特定样本的回答（样本索引列在每个示例的开头）。

```txt
MultiRC 索引：2, 4
文章：是什么引起了运动的改变？力的施加。任何时候物体改变运动状态，都施加了力。这可以通过哪些方式发生？力可以使静止的物体开始运动。力可以使物体加速或减速。力可以使运动的物体停下来。力也可以引起方向的改变。简而言之，力引起运动的改变。运动的物体可以改变其速度、方向，或两者皆改变。我们知道运动的改变需要力。我们知道力的大小决定了运动的改变。当施加力时，物体的运动改变多少取决于两件事。它取决于力的大小（强度）。它还取决于物体的质量。想一想你平时经常做的一些简单任务。你可能会捡起一个棒球。这只需要很小的力。
问题：根据前文，棒球的质量会影响你捡起它所需施加力的大小吗？“质量越小，施加的力越小”是一个正确的答案吗？
答案：是
RWKV zero-shot：棒球的质量取决于施加的力。如果棒球的质量很小，施加的力就很小。
Hyena zero-shot：是的，它是。
RWKV few-shot：质量越小施加的力越小，是一种更重要的力。它是改变棒球质量的力。
Hyena few-shot：是的。
问题：根据前文，你对物体施加什么能使其运动或停止？“强度（Strength）”是一个正确的答案吗？
答案：否
RWKV zero-shot：施加在物体上的力就是施加在物体上的力。
Hyena zero-shot：是。
RWKV few-shot：施加在棒球上的力是一种力。它是一种施加在棒球上的力。
Hyena few-shot：是。
```

```txt
MultiRC index: 32, 64
Passage: The film opens with Sunita, a medical student, and her friends working on a project about the human brain. She wants to investigate the curious case of Sanjay Singhania, a notable city businessman, who is reported to have anterograde amnesia. Her professor denies access to Sanjay's records as it is currently under criminal investigation. Sunita, nonetheless, decides to investigate the matter herself. Sanjay is introduced as he brutally murders a man. He takes a Polaroid picture of the man, and writes on it "done". It is revealed that Sanjay has anterograde amnesia where he loses his memory every 15 minutes. Sanjay uses a system of photographs, notes, and tattoos on his body to recover his memory after each cycle. It is revealed that Sanjay is ultimately out to avenge the death of his sweetheart Kalpana, and that he is systematically killing the people who were responsible for it. His main target is "Ghajini", a notable social personality in the city. Police Inspector Arjun Yadav, on the case of the serial murders, tracks Sanjay down to his flat and attacks and disables him. Yadav finds two diaries where Sanjay has chronicled the events of 2005 and 2006 . The film flashes back to 2005 as Yadav reads the diary. Sanjay Singhania is shown as the owner of the Air Voice mobile telephone company. In the course of his business, Sanjay sends his men to meet Kalpana, a struggling model, about putting up a billboard above her apartment. The owner of Kalpana's advertising firm misinterprets this as a romantic advance, and in view of a possible lucrative Air Voice ad campaign and other benefits, encourages Kalpana to accept the overture.
Question: Based on the previous passage, Why did Sanjay murder a man? Is "Sanjay" a correct answer?
Answer: Yes
RWKV zero-shot: The answer is "Sanjay".
Hyena zero-shot: The answer is
RWKV few-shot: The question is not about Sanjay's death.
Hyena few-shot: No.
Question: Based on the previous passage, Every 15 minutes, Sanjay goes through what process, Which frustrates his attempts to avenge the death of his sweetheart? Is "He forgets about facts" a correct answer?
Answer: Yes
RWKV zero-shot: Yes, it is a correct answer..
Hyena zero-shot: The answer is.
RWKV few-shot: The answer is "yes".
Hyena few-shot: The answer is "yes."
```

## MultiRC 索引：128

文章：1863 年，亚历山大二世（Alexander II）重新召集了芬兰议会（Diet of Finland），并启动了多项改革，增强了芬兰脱离俄罗斯的自治权，包括建立其本国货币芬兰马克（markka）。商业自由化促进了外国投资的增加和工业发展。芬兰还拥有了第一批在芬兰行政管理下单独建立的铁路。最后，芬兰语从平民语言提升为与瑞典语平等的国家语言，为更大比例的社会群体打开了机遇之门。在芬兰，亚历山大二世至今仍被视为“仁慈的沙皇”（"The Good Tsar"）。这些改革可被视为一种真诚信念的产物，即改革在一个地广人稀、同质化的国家中比在整个俄罗斯更容易进行测试。它们也可以被视为对其在克里米亚战争（Crimean War）和波兰起义（Polish uprising）期间相对亲西方的民众忠诚的奖赏。鼓励芬兰民族主义和芬兰语也可以被视为淡化其与瑞典关系的企图。

问题：根据前文，自 1863 年以来，亚历山大二世在芬兰被称作什么？“仁慈的沙皇”（"The good tsar"）是正确答案吗？

答案：是

RWKV zero-shot：正确答案是“yes”。

Hyena zero-shot：是的，问题是正确的。

RWKV few-shot：芬兰语是人民的语言。

Hyena few-shot：是的。

MultiRC index: 1024

文章：爱因斯坦（Einstein）与马里奇（Maric）于 1903 年 1 月结婚。1904 年 5 月，夫妇俩的长子汉斯·阿尔伯特·爱因斯坦（Hans Albert Einstein）在瑞士伯尔尼出生。他们的次子爱德华（Eduard）于 1910 年 7 月在苏黎世出生。1914 年，夫妇俩分居；爱因斯坦移居柏林，其妻子与儿子们则留在苏黎世。在分居五年后，他们于 1919 年 2 月 14 日离婚。被父亲称为“泰特”（"Tete"，源自 petit）的爱德华在大约 20 岁时精神崩溃，并被诊断患有精神分裂症。他的母亲照顾他，他也在精神病院待过几段时期，包括在她去世后全托。与马里奇的婚姻似乎并不十分幸福。在 2015 年公开的信件中，爱因斯坦写信给他早期的恋人玛丽·温特勒（Marie Winteler），谈及自己的婚姻以及对玛丽依然强烈的感情。1910 年，在妻子怀第二个孩子期间，他写信给她道：“我每分每秒都在由衷地爱着你，并且感到只有男人才能体会到的极度痛苦”。关于对玛丽的爱，爱因斯坦称之为“错误的爱”和“错过的生活”。在自 1912 年便与埃尔莎·洛文塔尔（Elsa Lowenthal）保持关系后，爱因斯坦于 1919 年 6 月 2 日与她结婚。她是他的母系表姐和父系堂姐。1933 年，他们移民到美国。1935 年，埃尔莎·爱因斯坦被诊断出患有心脏和肾脏疾病；她于 1936 年 12 月去世。

问题：根据前文，阿尔伯特·爱因斯坦的父亲与母亲离婚后，过了多长时间再婚？“3 个月零 18 天”（"3 months and 18 days"）是正确答案吗？

答案：是

RWKV 零样本（zero-shot）：Albert Einstein 于 1916 年 3 月 3 日出生在德国 Gerlin。他是一位德国医生和一位德国妇女的儿子。

Hyena 零样本（zero-shot）：“3 个月零 18 天”

RWKV 少样本（few-shot）：这不是一个正确的答案。确切日期未知。

Hyena 少样本（few-shot）：是的，根据前文所述。