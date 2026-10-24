# 基于 BERT 的跨受试者表面肌电信号连续手部运动学估计方法

Chuang Lin, Xingjian Chen, Graduate Student Member IEEE, Weiyu Guo, Ning Jiang, Dario Farina , Fellow<sub>,</sub> IEEE, and Jingyong Su

— 基于表面肌电（surface electromyographic, sEMG）信号对手部运动学进行估计，提供了一种非侵入式人机接口。该方法通常是针对特定受试者的（subject-specific），因此在单个个体上的训练无法泛化到不同受试者。在本文中，我们提出了一种基于来自 Transformer 的双向编码器表示（Bidirectional Encoder Representation from Transformers, BERT）结构的方法，利用经过 µ 律归一化（µ-law normalization）后的 sEMG 信号均方根（RMS）特征来预测手部运动。该方法在受试者内和跨受试者条件下均进行了测试。我们使用两种难样本挖掘方法——梯度均衡机制（Gradient Harmonizing Mechanism, GHM）和在线难样本挖掘（Online Hard Sample Mining, OHEM）对模型进行了训练。所提出的方法与经典方法（包括长短期记忆网络 LSTM 和时序卷积网络 TCN）以及一种名为长曝光卷积记忆网络（Long Exposure Convolutional Memory Network, LE-ConvMN）的最新方法进行了对比。相关系数（CC）、归一化均方根误差（NRMSE）和时间开销被用作性能评估指标。我们的方法（sBERT-OHEM）在 Ninapro 数据集的跨受试者评估中达到了最先进的性能，同时在特定受试者测试中也取得了优异的性能。上述测试均基于相同随机选择的 10 名受试者。通常情况下，在跨受试者情形中，随着受试者数量的增加，性能不可避免地会下降。然而，在跨受试者条件下，我们的方法在 38 名受试者上的性能显著高于其他方法在 10 名受试者上的性能，这进一步验证了我们方法的优势。

— sEMG，手部运动学，BERT，难样本挖掘，跨受试者。

## I. 引言

随着互联网的发展以及提高老龄化人群生活质量的日益增长的需求，智能设备与人类之间的交互正在成为我们社会的重要组成部分。因此，迫切需要能够以自然、准确且稳健的方式实现这种交互的技术。例如，主动假肢、机器人辅助手术、无人机侦察等领域都需要人机交互（human-robot interaction）。此外，高效、精准且用户友好的人机交互（human-computer interaction, HCI）以及人机协作（human-machine collaboration, HMC）也备受关注。为了实现高精度和高效率，从生物信号中提取精确特征并将其转换为控制指令的技术在 HCI 和 HMC 领域正发挥着重要作用。表面肌电（sEMG）信号可以通过可穿戴设备轻松记录，几十年来一直被用于解码人类运动意图 [1], [2]。

人手通过提供超过 20 个自由度（DoF）[3] 的复杂结构，使人类能够完成极其复杂的动作。通过可穿戴系统解码手部运动将提供一个高信息传输接口。

近年来，深度学习方法已被广泛用于自动提取 sEMG 特征，并在分类任务中表现出优异性能 [4], [5], [6]。目前，研究重点主要集中在连续运动回归而非分类上。

当前用于连续运动估计的方法大致可分为两类：基于模型（model-based）和无模型（model-free）方法 [7]。基于模型的方法依赖于物理模型，通常包括运动学模型 [8]、肌肉骨骼模型 [9], [10] 以及动力学模型。然而，这些模型可能非常复杂，因此其参数辨识具有很大挑战性。因此，研究人员目前更倾向于使用无模型方法。例如，[11] 提出了一种同步连续运动学估计方法，使用单个人工神经网络（ANN）估计跨越肩关节和肘关节的 4 个自由度。在 [12] 中，提出了一种利用循环神经网络（RNN）结构从 sEMG 估计手部姿态的方法。然而，据我们所知，目前尚无方法可应用于跨被试（cross-subject）情况。

EMG 信号因人而异，甚至同一个人在不同记录时间下也会存在差异。迁移学习方法 [13] 可用于调整特定被试的模型以适用于其他被试。例如，Fan 等人 [14] 提出了一种基于迁移学习的手势识别方法，利用健全手部的模型进行学习以适配截肢者。然而，迁移学习会增加占用内存的额外结构 [15]，且输出的模型只能从一个被试迁移到另一个被试，而生成一个能够同时适用于两个或更多被试的模型仍然十分困难。

作为另一项挑战，归一化在深度学习方法中得到了广泛应用。在基于 sEMG 的估计领域，最小-最大归一化（min-max normalization）是使用最广泛的归一化方法。然而，不同被试的 sEMG 信号处于不同的范围且分布各异，当对多个被试进行联合分析时，最小-最大归一化会干扰它们各自原有的特征。此外，时域中 sEMG 的大量有效信息都集中在零值附近 [16]，无法通过线性方法进行有效识别。

最后，现有的回归方法往往忽略了对包含有用信息的困难样本（hard samples）的研究，而这些样本可能会连续集中在若干时间段内 [17]。

在此，我们提出了一种基于来自 Transformer 的双向编码器表示（BERT，Bidirectional Encoder Representation from Transformers）结构的方法，该方法结合了 $\mu$-律（$\mu$-law）归一化，这是一种能够更好地放大低幅值并保持较大数值尺度的非线性归一化方法。应用了梯度均衡机制（GHM，Gradient Harmonizing Mechanism）和在线难例挖掘（OHEM，Online Hard Samples Mining）以使模型更好地从困难样本中学习，并应用了平滑层来减少由 BERT 引起的波动。我们的方法在 Ninapro 数据集上进行了验证，并与两种经典方法 LSTM [18] 和 TCN [19]，以及近期提出的一种名为长曝光卷积记忆网络（LE-ConvMN，Long Exposure Convolutional Memory Network）[17] 的新方法进行了对比，该方法在连续手部运动学估计中应达到最先进的性能。总而言之，本文的主要贡献如下：

• 首次提出了基于 BERT 的连续手部运动回归方法。

• 应用了难例挖掘策略，以从 sEMG 中实现更好且稳定的估计。

• 据我们所知，我们的方法可应用于跨受试者场景，这是以往工作尚未解决的问题，实验结果表明我们的方法达到了最先进的性能。

## II. 相关工作

在本节中，将介绍用于连续运动估计的两种经典模型（包括 LSTM 和 TCN）以及近期的一种名为 LE-ConvMN 的方法。由于无模型（model-free）方法在实践中应用更为广泛，因此本文仅考虑此类方法。

## A. 长短期记忆网络（LSTM）

LSTM [18] 是由 RNN 结构发展而来的，具备极强的解决时间序列问题的能力。RNN 允许保留在前次迭代中收集的上下文信息，以使后续迭代受益。然而，随着数据复杂度和序列长度的增加，短时循环回路在处理长时间序列时表现较弱。LSTM 旨在通过记忆与遗忘的结合来解决这一问题。由于基本的 RNN 结构，LSTM 能够自然地进行记忆，并通过应用遗忘门结构实现了遗忘能力。作为一种经典的无模型方法，LSTM 因其长时记忆特性而被广泛应用于连续运动估计中。

## B. 时间卷积网络（TCN）

TCN [19] 是一种广泛用于在没有 RNN 结构的情况下提取时间信息特征的神经网络。由于卷积核尺寸的限制，经典卷积网络不适用于处理时间序列问题。

TCN 建立在两个基本原则之上：(1) 网络产生的输出长度与输入相同。(2) 不能存在从未来到过去的泄漏。对于原则 (1)，TCN 设计了一维全卷积网络（FCN，1D fully-convolutional network）[20] 架构，其每个隐藏层的长度与输入层相同。对于原则 (2)，TCN 利用了因果卷积（causal convolution），该卷积仅使用网络正在预测的时间点之前的信息。

TCN 应用空洞卷积（dilated convolutions）[21] 来扩大感受野以处理长历史时间序列，从而避免堆叠过多的 CNN 层。作为一种经典的无模型（model-free）方法，TCN 是连续运动估计中另一种广泛应用的模型。

## C. LE-ConvMN

长曝光卷积记忆网络（Long Exposure Convolutional Memory Network，LE-ConvMN）由 Guo 等人 [17] 提出，旨在更好地利用 sEMG 数据中的时空信息，是一种用于连续手部运动估计的新颖方法。

长曝光是一种 sEMG 数据处理方法。传统上，RMS 特征是通过按窗口大小进行步进的滑动窗口提取的。长曝光方法通过减小步长来提取特征，这可以作为一种增加数据量的独特方法。本文中的步长固定为 1。

随后将 ConvLSTM 模型 [22] 应用于长曝光数据。ConvLSTM 是一种由 LSTM 衍生而来的 RNN 结构模型。为了处理高维数据，该模型将门控的完全连接矩阵运算扩展为卷积运算。在 [17] 中，LE-ConvMN 被认为在该领域达到了最先进的（state-of-the-art）结果。

## III. 结合难样本挖掘的平滑 BERT A. BERT

尽管上述所有模型在我们的领域都是可行的，但仍存在一些缺陷。对于以往的深度学习方法，LSTM 由于其 RNN 结构使得训练依赖于先前的数据，因此必须保持数据的顺序，从而导致耗时较长。TCN 不依赖 RNN 结构，而是基于 CNN 结构，这可能导致估计的不稳定性和波动。LE-ConvMN 虽然能够更好地利用时空信息，但更加耗时，且需要占用大量 GPU 显存。训练这种模型的代价巨大，硬件要求也相对较高。此外，这些方法都源于单向结构，可能会忽略 sEMG 的上下文信息。

![](images/326c6a5b1b76f3a4bb61783712b5b69d627b1d58b3b4f94a5dee47e0a07d2013.jpg)  
Fig. 1. 基于 BERT 的 sEMG 信号连续手部运动学估计方法模型结构。原始 sEMG 信号被提取为长曝光 RMS 特征，这可以看作是一种数据增强的方法。随后，将特征划分为长度为 100ms（200 个采样点）的切片。模型在嵌入（embedding）后对每个 RMS 特征切片进行估计。最后，所有估计结果由平滑层进行平滑。

基于 Transformer 的双向编码器表示（Bidirectional Encoder Representation from Transformers，BERT）神经网络 [23] 是近年来基于 Transformer [24] 结构提出的一种新颖方法。BERT 自提出以来便备受关注。与 RNN 和 TCN 结构不同，BERT 双向提取并学习序列中的特征，能够从小尺度的时空序列信息中提取特征，使其性能优于其他模型。

BERT 在多个受试者上也同样可行，这归功于注意力机制和残差跳跃连接赋予其从小规模序列数据中提取特征的强大能力。BERT 可以同时接收未来信号和先前信号，因此能够从整个序列中提取特征，而不仅限于先前信号，这使得 BERT 在多个受试者上的表现优于经典的 TCN 和 RNN 模型。BERT 在 sEMG 上的性能表现见实验部分。

此外，BERT 是一种优秀的预训练方法。[23] 中指出，训练良好的 BERT 模型在微调后能够在下游任务中达到最先进（state-of-the-art）的性能，这有助于该领域的迁移学习。

Transformer 是一种用于解决语言序列的新型语言模型。Transformer 强大的特征提取能力依赖于其注意力机制。BERT 可以被视为 Transformer 编码器的复杂堆叠。我们对 BERT 进行了多项修改，使其更适用于基于 sEMG 的手部运动学序列估计问题。整个过程如图 Figure 1 所示，具体描述如下。

Transformer 编码器由两部分组成：多头自注意力机制（Multihead Self-Attention Mechanism, MSA）和多层感知机（MLP）模块。为了更好地描述该结构，我们将一维输入表示为 $X = [ x _ { 0 } , x _ { 1 } , \cdot \cdot \cdot , x _ { t } ]$。第 $i$ 个编码器层的输出序列表示为 $Z _ { i } ( i = 1 , \cdots , L )$，$L$ 为编码器层的数量。在将输入送入这些编码器层之前，我们对其进行嵌入操作并记为 $Z _ { 0 } { \mathrm { : } }$

$$
Z _ {0} = [ x _ {0} ^ {p}, x _ {1} ^ {p}, \dots , x _ {t} ^ {p}, \dots ] + E _ {t i m e}\tag{1}
$$

其中 $x _ { i } ^ { p }$ 表示输入 $X$ 经过投影嵌入后的结果，$E _ { t i m e }$ 为时间嵌入向量，它使模型具备捕获 sEMG 时间信息的能力。采用可训练的一维向量作为时间嵌入，并使用线性层（linear layer, LL）作为线性投影。因此，

$$
X ^ {p} = [ x _ {0} ^ {p}, x _ {1} ^ {p}, \dots , x _ {t} ^ {p}, \dots ] = L L _ {(s \times c _ {i}, s \times d _ {h})} (X)\tag{2}
$$

其中 $d _ { h }$ 为嵌入维度，自然地，$E _ { t i m e } \in \mathcal { R } ^ { s \times h }$。编码器模块中应用了层归一化（Layer normalization）以及残差跳跃连接以解决退化问题，因此，一个编码器层可以描述如下：

$$
\begin{array}{l} Z _ {l} ^ {\prime} = M S A (L a y e r N o r m (Z _ {l - 1})) + Z _ {l - 1} \\ Z _ {l} = M L P (L a y e r N o r m (Z _ {l} ^ {\prime})) + Z _ {l} ^ {\prime} \end{array}\tag{3}
$$

其中 $l \in \{ 1 , 2 , \cdots , L \}$，最后，我们应用一个 LL 从 $Z _ { L } \mathrm { : }$ 中提取结果：

$$
Z = L L _ {(s \times h, 1 \times c _ {o})} (Z _ {L})\tag{4}
$$

随后我们将 MSA 模块和 MLP 模块介绍如下。

SA（自注意力）可以被视为寻找输入 $Z _ { i }$ 中不同采样点之间关系的过程，这是通过分别记为查询矩阵 $Q$、键矩阵 $K$ 以及值矩阵 $V$ 的三个矩阵实现的。它们通过线性变换计算得到：

$$
[ Q _ {i}, K _ {i}, V _ {i} ] = Z _ {i - 1} W _ {i} ^ {Q K V}\tag{5}
$$

其中下标 i 表示参数是在编码器层 i 中计算的，$\dot { W } _ { i } ^ { Q K V }$ 是每层中的可学习权重矩阵。Q 和 K 经过缩放后用于计算 $V$ 的权重，最后对 $V$ 的所有值进行加权求和以得到结果：

$$
S A (Z _ {i - 1}) = \mathrm{softmax} (\frac {Q _ {i} K _ {i} ^ {T}}{\sqrt {d _ {h}}}) V\tag{6}
$$

其中 $S A ( Z _ { i - 1 } ) ~ \in ~ \mathcal { R } ^ { s \times d _ { h } }$。该方法使模型能够关注 sEMG 输入序列中的重要部分。

我们在 MSA 中应用 h 个注意力头来计算 Q、K、V，这意味着 MSA 允许模型通过不同的注意力头对输入 sEMG 序列的不同部分给予不同的关注。MSA 将不同注意力头计算出的所有 SA 输出进行拼接，然后将其投影得到结果。MSA 可描述如下：

$$
\begin{array}{l} M S A (Z _ {i - 1}) \\ = [ S A _ {1} (Z _ {i - 1}), S A _ {2} (Z _ {i - 1}), \dots , S A _ {h} (Z _ {i - 1}) ] W _ {i} ^ {M S A} \end{array}\tag{7}
$$

其中每个 SA 都有其唯一的 Q、K、V，$W _ { i } ^ { M S A }$ 是每个编码器层中的可学习权重矩阵。

此外，MLP 模块可以表示如下：

$$
\begin{array}{l} Z _ {1} ^ {\prime} = L L _ {(s \times d _ {h}, s \times d _ {f h})} (Z ^ {\prime}) \\ Z _ {2} ^ {\prime} = \operatorname{GELU} (Z _ {2} ^ {\prime}) \\ Z _ {o} ^ {\prime} = L L _ {(s \times d _ {f h}, s \times d _ {h})} \end{array}\tag{8}
$$

其中 $d _ { f h }$ 是 MLP 模块的隐藏层大小，GELU 是高斯误差线性单元（Gaussian Error Linear Unit）激活函数。

BERT 被提出用于处理语言序列问题，它输出离散张量。然而，运动中的关节角度应当是连续值。因此，尽管 BERT 在两个评估指标上表现良好，但它会导致预测值出现严重的波动。为了解决这一问题，我们在 BERT 预测之后应用了一种平滑方法，以获得更平滑的结果。

通过应用滑动窗口对结果进行平滑处理。对于每个滑动步长，我们计算窗口内所有采样点的平均值：

$$
\operatorname{AvgSmooth} (Y) = \left[ \sum_ {i = 1} ^ {w} y _ {i} / w, \sum_ {i = 2} ^ {w + 1} y _ {i} / w, \dots , \sum_ {i = k} ^ {w + k - 1} y _ {i} / w, \dots \right]\tag{9}
$$

其中 w 代表滑动窗口的大小，$Y \ =$ $[ y _ { 0 } , y _ { 1 } , \cdot \cdot \cdot , y _ { k } , \cdot \cdot \cdot ]$ 是输入序列。

## B. μ 律归一化 (μ-Law Normalization)

在将sEMG的RMS特征输入模型之前对其应用了 $\mu$-law 归一化（$\mu$-law normalization）[16], [25]，已有研究证明采用 $\mu$-law 方法对sEMG信号进行归一化可以提升性能 [6], [16], [26]。$\mu$-law 归一化的定义如下：

$$
F (x _ {t}) = \operatorname{sign} (x _ {t}) \frac {\ln (1 + \mu | x _ {t} |)}{\ln (1 + \mu)}\tag{10}
$$

其中 $x _ { t }$ 表示第 $t$ 个采样点的输入，超参数 $\mu$ 决定归一化后的范围。

sEMG信号具有许多有用信息集中在零附近的特征，$\mu$-law 归一化能够以对数形式放大较小幅度的传感器输出，这种非线性归一化的表现优于线性归一化。$\mu$-law 归一化对回归性能的提升在实验部分中展示。

## C. 难样本挖掘（Hard Sample Mining）

难样本挖掘是数据挖掘中的一个重要问题。难样本是指估计值与真实值之间的损失适度偏大的样本，相比简单样本，它们对模型训练的贡献更大。相比之下，简单样本的估计值与真实值之间的偏差很小。此外，数据中不可避免地会存在一些不良数据，称为异常值（outliers）。数据采集过程中产生的误差也会影响结果。设备误差不可避免地会导致sEMG采集过程中产生异常值，并且数据集中始终存在难样本。GHM使模型能够更多地从这些难样本中获益，同时尽可能少地受简单样本和异常值的影响，从而增强了模型的鲁棒性和稳定性。

目前有几种常用的难样本处理方法。最简单的方法是增加数据集的规模，但在我们的领域中，人体连续运动数据很难采集，即使是 NinaPro 数据库的数据量也是有限的。还有一些低成本的方法，例如裁剪、翻转和旋转。然而，由于鲜明的个体特征差异，这些方法无法有效增强sEMG信号。在线难例挖掘（Online Hard Example Mining, OHEM）[27]是另一种相关方法。在 [27] 中，OHEM 通过按损失对输入特征进行排序，并选取当前网络表现最差的若干样本作为难例。网络仅根据选出的难样本进行更新。

在本文中，通过OHEM按MSE损失对输出采样点进行排序，并选取当前网络表现最差的若干样本作为难例。我们仅使用选出的难样本来更新模型。

考虑到异常值，采用了专注于难样本挖掘（hard sample mining）的损失计算策略。梯度均衡机制（Gradient Harmonizing Mechanism, GHM）是一种损失计算机制 [28]，它可以为难样本分配更大的权重，使模型尽可能从中获益。GHM 定义了梯度范数来衡量真实值与估计值之间的偏差。我们将梯度范数表示为 g。g 的取值范围为 0 到 1，当 g 趋近于 $0 ,$ 时，估计值与真实值几乎相同；而 $g$ 趋近于 1 则意味着两个值之间存在巨大偏差。

GHM 定义了梯度密度（Gradient Density, GD）来通过梯度范数 g 描述数据的分布。该关系可以表示如下：

$$
G D (g) = \frac {1}{l _ {\epsilon} (g)} \sum_ {k = 1} ^ {N} \delta_ {\epsilon} (g _ {k}, g)\tag{11}
$$

其中 $g$ 的 GD 表示落在以 $g$ 为中心、长度为 $\epsilon$ 的区域内的采样点数量。我们将一个子帧中第 k 个采样点的梯度范数记为 g<sub>k</sub>。$l _ { \epsilon } ( g )$ 表示实际区域的长度，可定义如下：

$$
l _ {\epsilon} (g) = \min (g + \epsilon , 1) - \max (g - \epsilon , 0)\tag{12}
$$

$\delta _ { \epsilon } ( g _ { k } , g )$ 是一个用于计数的判断函数，它对梯度范数落在范围 $l _ { \epsilon } ( g )$ 内的所有采样点进行计数。公式如下：

$$
\delta_ {\epsilon} (g _ {k}, g) = \left\{ \begin{array}{l l} 1 & \text { if } g _ {k} - \frac {\epsilon}{2} \leq g \leq g _ {k} + \frac {\epsilon}{2} \\ 0 & \text { otherwise } \end{array} \right.\tag{13}
$$

随后，梯度密度均衡参数定义为：

$$
\beta_ {i} = \frac {N}{G D (g _ {i})}\tag{14}
$$

其中 N 是一个子帧中的采样点总数，$\beta _ { i }$ 是该子帧中第 i 个采样点的梯度密度均衡参数。

本研究中使用 MSE 损失作为基础损失，定义如下：

$$
L _ {M S E} = \frac {\sum_ {i = 1} ^ {N} (\theta_ {i} - \hat {\theta} _ {i}) ^ {2}}{N}\tag{15}
$$

由于该领域数据的匮乏，在 GHM-MSE 损失中 $g _ { i }$ 定义如下，以使损失分别呈稀疏分布：

$$
g _ {i} = \sigma (\frac {2 (\theta_ {i} - \hat {\theta} _ {i})}{N})\tag{16}
$$

其中 $\sigma$ 为 Sigmoid 激活函数。随后我们将 $L _ { G H M - M S E }$ 定义为：

$$
L _ {G H M - M S E} = \frac {\sum_ {i = 1} ^ {N} \beta_ {i} (\theta_ {i} - \hat {\theta} _ {i}) ^ {2}}{N}\tag{17}
$$

## IV. 实验

## A. 数据集

Ninapro [29] 是一个广泛使用的数据集，代表了 sEMG 领域针对健全手或截肢手最大规模的数据采集工作。在 Ninapro 数据集中，sEMG 的原始信号使用包含 12 个电极的 Delsys Trigno 无线系统进行采样。手部运动学通过 22 个关节角度进行测量，并使用 CyberGlove II 数据手套进行采样。sEMG 的采样率为 2 kHz。手部运动学的采样率为 20 Hz，随后重采样至 2 kHz。

![](images/4f2d5cdde9116532b56e6d147f93f4f46a30ba9563f50f22048cfa8e44aebee7.jpg)  
Fig. 2. 图左侧展示了 CyberGlove 通道，其中红点表示待估计的十个关节。右侧展示了所选取的六种手部动作。

Ninapro 包含分为 10 个子数据集的 300 多次数据采集，提供了肌电图、运动学等数据。实验选用从 Ninapro DB2 中挑选出的 10 名具有代表性的受试者展开，该数据集包含 40 名健全受试者进行的 49 种不同动作的 6 次重复。我们的筛选保证了性别和利手分布相对均匀，以确保该方法对不同受试者具有普遍有效性，他们的身高范围为 150-187cm，体重范围为 52-87kg。每个受试者选取了六种用于抓握不同物体的动作，这些动作具有相对较好的数据质量。此外，我们将估计重点集中在 10 个手指关节上。我们的选择如图 2 所示。

我们使用步长为 0.5 ms、滑动窗口为 100 ms 计算均方根（RMS，Root Mean Square）作为特征。RMS 可以降低数据采集所产生的噪声。经过简短的超参数搜索实验后，为了获得更好的性能，特征采用 $\mu = 2 ^ { 2 0 }$ 的 $\mu$-law 归一化进行处理。

在对 sEMG 进行处理后，$\begin{array} { r l r l } { X } & { { } \in } & { \mathcal { R } ^ { s \times c _ { i } } , X } & { { } = } \end{array}$ $[ x _ { 0 } , x _ { 1 } , \cdots , x _ { s } ]$ 表示处理结果，其中 s 为输入子帧的采样点数，$c _ { i }$ 代表 sEMG 通道数。在此，$s = 2 0 0 , c _ { i } = 1 2$。类似地，$Z \in \mathcal { R } ^ { 1 \times c _ { o } }$ 表示最终输出，其中 $c _ { o }$ 为手部运动学通道数。选取 10 个手指关节作为 10 个典型的运动学通道，因此 $c _ { o } = 1 0$。

在所有特定受试者（subject-specific）和跨受试者（cross-subject）的情况下，每个受试者数据的 7/10 用于训练，3/10 用于测试。具体而言，在特定受试者情况下，每个受试者都是单独进行训练和测试的。而在跨受试者情况下，我们同时基于 10 个个体的训练数据训练模型，但单独评估每个受试者的测试数据以获得平均性能。

## B. 参数评估

为了评估我们的方法并将其与其他方法进行比较，引入了以下两个评价指标。

皮尔逊相关系数（Pearson correlation coefficient, CC）是衡量两个变量之间线性相关程度的常用标准。CC 的取值范围为 −1 到 1。CC 值越大，预测动作与估计动作之间的相似度越高，这意味着获得了更好的估计结果。

TABLE I  
采用 Min-Max 归一化时 10 位不同受试者在 10 个关节和 6 种动作上不同模型的平均性能

<table><tr><td>模型</td><td>CC</td><td>NRMSE</td><td> $\kappa$ </td><td>时间开销/epoch(s)</td></tr><tr><td>LE-LSTM</td><td>0.8428</td><td>0.0835</td><td>1.1456</td><td>25.779</td></tr><tr><td>LE-TCN</td><td>0.8021</td><td>0.0934</td><td>1.4033</td><td>15.469</td></tr><tr><td>LE-ConvMN</td><td>0.8872</td><td>0.0705</td><td>0.6345</td><td>13.446</td></tr><tr><td>BERT</td><td>0.8523</td><td>0.0813</td><td>1.5133</td><td>4.915</td></tr><tr><td>sBERT</td><td>0.8522</td><td>0.0811</td><td>0.5385</td><td>4.918</td></tr><tr><td>BERT-GHM</td><td>0.8532</td><td>0.0807</td><td>1.5480</td><td>4.922</td></tr><tr><td>BERT-OHEM</td><td>0.8531</td><td>0.0806</td><td>1.5282</td><td>4.924</td></tr><tr><td>sBERT-GHM</td><td>0.8533</td><td>0.0804</td><td>0.5382</td><td>4.929</td></tr><tr><td>sBERT-OHEM</td><td>0.8536</td><td>0.0802</td><td>0.5438</td><td>4.931</td></tr><tr><td>LE-LSTM*</td><td>0.4950</td><td>0.1661</td><td>-</td><td>255.057</td></tr><tr><td>LE-TCN*</td><td>0.5398</td><td>0.1467</td><td>-</td><td>152.045</td></tr><tr><td>LE-ConvMN*</td><td>0.5797</td><td>0.1517</td><td>-</td><td>130.467</td></tr><tr><td>BERT*</td><td>0.4504</td><td>0.1927</td><td>-</td><td>44.019</td></tr><tr><td>sBERT*</td><td>0.4574</td><td>0.1866</td><td>-</td><td>44.017</td></tr><tr><td>BERT-GHM*</td><td>0.4195</td><td>0.1966</td><td>-</td><td>45.646</td></tr><tr><td>BERT-OHEM*</td><td>0.5275</td><td>0.1899</td><td>-</td><td>44.111</td></tr><tr><td>sBERT-GHM*</td><td>0.4297</td><td>0.1912</td><td>-</td><td>45.933</td></tr><tr><td>sBERT-OHEM*</td><td>0.4539</td><td>0.1751</td><td>-</td><td>44.192</td></tr></table>

1. '\*’ 表示该模型在所有选定的受试者上进行训练。

均方根误差（Root Mean Square Error, RMSE）是衡量预测值与实际观测值之间偏差的典型指标。对于同一个关节角度，RMSE 越小，估计效果越好。然而，我们无法直接比较不同关节角度的 RMSE。为此，采用对 RMSE 进行 Min-Max 归一化来解决这一问题。因此，归一化均方根误差（Normalized RMSE, NRMSE）定义为：

$$
N R M S E = \frac {R M S E}{\theta_ {m a x} - \theta_ {m i n}}\tag{18}
$$

其中 $\theta _ { m a x } , \theta _ { m i n }$ 分别表示某一关节角度的最大和最小真实值。

无偏标准差（记为 σ）是衡量一组数据离散程度的常用指标。计算每个受试者 10 个关节的 σ，以衡量各模型的稳定性。该值越小，离散程度越低，估计的稳定性越好。在我们的工作中，基于每个受试者 10 个预测关节的结果采用了这一指标。

采用每个关节所有点的平均曲率（记为 κ）来衡量估计曲线的平滑度。曲率越小，曲线越平滑。

## C. Experimental Results

在连续手部运动估计任务上，验证了我们方法的效率与准确性，并与以往的模型进行了对比。所有模型均在 Pytorch 框架 [30] 上运行。

所有模型均在同一块 GPU（NVIDIA GeForce RTX 3090）上进行训练，除 LE-ConvMN 训练了 1000 个 epoch 以获取更多 epoch 达到收敛外，其余每个模型均训练了 400 个 epoch。所有模型均采用了长时间曝光方法（Long exposure method）[17]，以排除数据处理的影响。实验在两种情形下展开：受试者特异性（subject-specific）情形和跨受试者（cross-subject）情形。在受试者特异性情形下，模型在单个受试者的数据上进行训练和验证。相比之下，模型在跨受试者情形下同时在多个受试者的数据上进行训练，并在每个单一受试者上进行验证。在每个受试者上分别计算各模型的 NRMSE 和 CC，以详细展示模型性能。同样，我们统计了每个训练操作中每个 epoch 的平均训练时间，并估计了收敛时间。推理时间是实际应用中估计的时间开销，该过程在相同的 CPU（Intel i7-10875H）上进行模拟，以比较不同模型的性能。针对这些指标，应用 Friedman 检验（Friedman test）和 Wilcoxon 符号秩检验（Wilcoxon signed-rank test）来评估我们方法的显著性，并采用 Bonferroni 校正（Bonferroni correction）对结果进行了校正。

TABLE II  
不同模型在 10 名不同受试者的 10 个关节和 6 种运动上在 10 名不同受试者的 10 个关节和 6 种运动上采用 µ-律归一化的平均性能

<table><tr><td>模型</td><td>CC</td><td>NRMSE</td><td> $\kappa$ </td><td>时间开销/epoch(s)</td></tr><tr><td>LE-LSTM</td><td>0.7685</td><td>0.0960</td><td>0.5812</td><td>26.360</td></tr><tr><td>LE-TCN</td><td>0.8329</td><td>0.0875</td><td>1.5332</td><td>15.623</td></tr><tr><td>LE-ConvMN</td><td>0.9017</td><td>0.0648</td><td>0.6883</td><td>13.468</td></tr><tr><td>BERT</td><td>0.8678</td><td>0.0773</td><td>1.5708</td><td>4.948</td></tr><tr><td>sBERT</td><td>0.8675</td><td>0.0772</td><td>0.5348</td><td>4.951</td></tr><tr><td>BERT-GHM</td><td>0.8679</td><td>0.0770</td><td>1.5272</td><td>4.956</td></tr><tr><td>BERT-OHEM</td><td>0.8683</td><td>0.0768</td><td>1.5857</td><td>4.957</td></tr><tr><td>sBERT-GHM</td><td>0.8689</td><td>0.0766</td><td>0.5435</td><td>4.962</td></tr><tr><td>sBERT-OHEM</td><td>0.8693</td><td>0.0764</td><td>0.5323</td><td>4.960</td></tr><tr><td>LE-LSTM*</td><td>0.5536</td><td>0.1495</td><td>-</td><td>252.660</td></tr><tr><td>LE-TCN*</td><td>0.6695</td><td>0.1292</td><td>-</td><td>151.598</td></tr><tr><td>LE-ConvMN*</td><td>0.7410</td><td>0.1229</td><td>-</td><td>130.060</td></tr><tr><td>BERT*</td><td>0.8346</td><td>0.0881</td><td>1.6098</td><td>44.001</td></tr><tr><td>sBERT*</td><td>0.8427</td><td>0.0850</td><td>0.6011</td><td>44.426</td></tr><tr><td>BERT-GHM*</td><td>0.8308</td><td>0.0877</td><td>1.5815</td><td>46.386</td></tr><tr><td>BERT-OHEM*</td><td>0.8299</td><td>0.0878</td><td>1.6019</td><td>44.039</td></tr><tr><td>sBERT-GHM*</td><td>0.8441</td><td>0.0846</td><td>0.6012</td><td>45.569</td></tr><tr><td>sBERT-OHEM*</td><td>0.8430</td><td>0.0835</td><td>0.6263</td><td>44.449</td></tr><tr><td>sBERT-OHEM**</td><td>0.8065</td><td>0.0911</td><td>-</td><td>165.454</td></tr></table>

1. “*”表示该模型在选定的全部10名受试者上进行训练。2. 此外，作为我们的方法，sBERT-OHEM在跨受试者情形下对Ninapro DB2数据集中的全部38名受试者进行了验证（两名可能损坏的受试者数据除外），用上标“**”表示。

在尝试了几组参数后，我们选择了一组相对较优的参数来进行实验。所有模型的训练学习率均设为0.0001，并在200个epoch后减半；仅LE-ConvMN是在每300个epoch后减半。

为了验证BERT的优越性，我们在10名受试者和10个通道上对RNN和TCN模型进行了训练。结果如Table I所示。模型名称的前缀$\bf \dot { s } - \bf \dot { \theta }$表示应用了平滑层，模型名称后面的后缀“-GHM”或“-OHEM”表示该模型在当前及后续实验中结合了GHM或OHEM进行训练。前缀“LE-”用于区分我们的模型与原始经典模型。

![](images/a87d4293d03ad8f4ba29cb6afa1d810cc3979745a1610144429c2c2e1877f8e9.jpg)  
(a) LSTM

![](images/4be5bff01ccd881e70529f2625d37582ad99d380ea6318811c5c7bccdfd3689f.jpg)  
(c) LE-ConvMN

![](images/98ebee102a6819ebecfc0385938702b8cf1d08b36f3e9a87d9a9c55cbe6656ef.jpg)  
(b) TCN

![](images/73dc2de9f5c02b531887ddf01a5454e7df29a052efd3824a6386ec8c2a08530d.jpg)  
(d) BERT

![](images/264d95bac98de245938c65c0b623e87e01d8b1a678df4693b4a75320e08a9910.jpg)  
(e) BERT-OHEM

![](images/c640738185558b62651a1a7511b2602e6503857ff82b9a550d19a0ff9654bf01.jpg)  
(f) sBERT-OHEM  
Fig. 3. 跨受试者情况下基于 BERT 的方法与其他方法的估计结果。我们的方法使用 µ-law 归一化进行训练，而其他方法使用经典的 Min-Max 归一化进行训练。红色曲线为真实值，蓝色曲线为估计值。图中仅展示了两个关节以更好地呈现估计细节，这可以代表全部 10 个关节的估计质量。

在 10 名受试者中，受试者 S3 的表现最好。我们在图 3 中展示了 S3 的两个代表性关节角度的结果，作为我们方法的代表，使用 μ-law 归一化的 BERT、BERT-OHEM 和 sBERT-OHEM 的结果如图 3 所示。

我们方法在单受试者上的平均 CC 和 NRMSE（最佳性能 $0 . 8 7 \ : \pm \ : 0 . 0 5 ; \ : 0 . 0 7 \ : \pm \ : 0 . 0 1 )$ 显著优于 LE-TCN $( 0 . 8 0 ~ \pm ~ 0 . 0 6 ,$ $p \ =$ $0 . 0 0 5 ; 0 . 0 9 \pm 0 . 0 2 , p = 0 . 0 0 5 )$ ，略优于 LE-LSTM $( 0 . 8 4 ~ \pm ~ 0 . 1 1$ , p = 0.262; 0.08 ± 0.03, $p \ = \ 0 . 2 4 1 )$ ，但略逊于 LE-ConvMN $( 0 . 8 9 \pm 0 . 1 2 , \ p = 0 . 0 7 4$ $0 . 0 7 \pm 0 . 0 3 , p = 0 . 0 5 9 )$ 。由于其他方法的特殊结构导致其无法并行训练，我们方法的平均训练时间成本显著低于其他方法。BERT 达到收敛所需的平均时间（约 0.5 小时）少于 LE-LSTM（约 2.5 小时）、LE-TCN（约 1.5 小时）和 LE-ConvMN（约 1.8 小时）。此外，验证了基于 BERT 的方法从 sEMG 序列中提取特征的强大能力。

为了比较 μ-law 归一化和 Min-Max 归一化，将它们分别应用于我们选择的每个受试者的每个模型。在我们的研究中，μ 设置为 $2 ^ { 2 0 }$。结果见表 I 和表 II，表明在我们的研究中 μ-law 归一化的表现优于 Min-Max 归一化，带来了显著的改进。

基于采用 μ-law 归一化训练的 BERT，我们在模型上应用了 GHM 或 OHEM 以及平滑层，从而设计了一些变体，其结果如表 II 所示。此外，在跨受试者实验中验证了采用 μ-law 归一化的 LE-ConvMN、LE-LSTM 和 LE-TCN，以排除归一化带来的影响。

结果表明，GHM、OHEM 和平滑层（smooth layer）在单受试者和多受试者情况下均带来了轻微的提升。尽管在评价指标上的提升较小，但 OHEM 和平滑层提高了估计的稳定性，并减少了预测运动曲线的波动，如图 3 所示。带有平滑层的 OHEM 在基于 BERT 的模型变体中表现最佳。GHM 则导致稳定性下降。LE-ConvMN 在单受试者上仍具有最先进的（state-of-the-art）性能，但我们的方法在估计中可以达到更好的稳定性，这将在下文段落中展示。两个指标的图表显示在图 4 中，以展示在每个单独受试者上的估计表现。此外，当在跨受试者情况下增加受试者数量时，指标出现了可接受的下降。尽管基于 BERT 的方法受波动的困扰更大，但如表 I、表 II 和图 3 所示，平滑层可以有效地排除这种波动。我们的方法即使在 38 位受试者上的性能，也优于其他方法仅在 10 位受试者上的性能，这表明了在跨受试者情况下提取特征的强大能力。

如图 4 所示，引入了 10 个关节估计的无偏标准差来评估这些方法估计的稳定性，并表示为 σ（$\sigma _ { c }$ 和 $\sigma _ { n }$ 分别对应 CC 和 NRMSE）。作为代表，我们的方法在特定受试者（subject-specific）情况下，sBERT-OHEM（$\sigma _ { c } = 0 . 0 6 1 4 ;$ $\sigma _ { n } = 0 . 0 2 4 2 ;$）在 $\sigma _ { c }$ 上显著优于 TCN（$\sigma _ { c } = 0 . 0 8 5 6$ $p = 0 . 0 4 7 ;$ $\sigma _ { n } = 0 . 0 2 6 8$ $p = 0 . 7 6 6$），在 $\sigma _ { n }$ 上略好；在两个指标上均略低于 LSTM（$\sigma _ { c } = 0 . 0 7 1 8 .$ $p \ = \ 0 . 3 8 4 ;$ $\sigma _ { n } ~ = ~ 0 . 0 2 5 4 .$ $p = 0 . 8 5 9$），且在 $\sigma _ { c }$ 和 $\sigma _ { n }$ 上均略低于 LE-ConvMN（$\sigma _ { c } = 0 . 0 8 0 2$ $p = 0 . 6 8 3 \mathrm { ; }$ $\sigma _ { n } = 0 . 0 3 0 1$ $p = 0 . 5 7 4$）。在跨受试者情况下，作为代表的我们方法 sBERT-OHEM（$\sigma _ { c } = 0 . 0 7 5 3 ;$ $\sigma _ { n } = 0 . 0 2 6 5$）在两个指标上的 σ 均显著低于 TCN（$\sigma _ { c } = 0 . 1 5 7 0$ $p = 0 . 0 0 7 ;$ $\sigma _ { n } = 0 . 0 9 2 9$ $p = 0 . 0 0 5$）、LSTM（$\sigma _ { c } ~ = ~ 0 . 1 7 8 5$ $p = 0 . 0 0 5 ;$ $\sigma _ { n } = 0 . 0 8 2 6 .$ $p = 0 . 0 0 5$）以及 LE-ConvMN（$\sigma _ { c } = 0 . 1 5 8 8$ $p = 0 . 0 0 7 ;$ $\sigma _ { n } = 0 . 0 6 2 8$ $p =$ 0.009）。表 III 显示了每种模型在两种情况下的平均无偏标准差。结果表明，我们的方法比其他模型具有更好的稳定性。

![](images/5217f912a60a097e40d595d332ab5ee96b781bc71d16f3e0a26d834761d02473.jpg)  
(a) CC

![](images/eb628f6e0ec8686a44bcd274f398b2fb76234cedc1206b46de0051d8ebc36e71.jpg)  
(b) NRMSE  
Fig. 4. 跨受试者情况下10名手部健全的健康受试者的性能指标汇总。结果表明，我们的方法在所有受试者中均优于其他无模型（model-free）方法。我们方法的 CC 和 NRMSE 显著优于其他模型。此外，从偏差棒可以看出，我们的方法比其他方法更加稳定。

由于不同受试者具有不同的特征，EMG（肌电）信号具有受试者特异性和非平稳特性，这使得以往的方法一直难以找到一种通用的方法来适应跨受试者的情况。因此，研究人员通常针对特定受试者训练并定制一套独特的模型参数。然而，由于 BERT 从小规模数据中提取特征的强大能力，它在多个受试者上的表现优于其他方法。在接下来的实验中，所选全部10名受试者的 sEMG 信号被拼接在一起。我们在受试者组合数据集上将我们的方法与 LE-LSTM、LE-TCN 和 LE-ConvMN 方法进行了比较，以验证不同模型在多个受试者上的性能。结果如 Table I 和 Table II 所示。

TABLE III  
不同模型在10个关节间的无偏标准差

<table><tr><td>模型</td><td> $\sigma_c$ </td><td> $\sigma_n$ </td><td> $\sigma_{c}^{*}$ </td><td> $\sigma_{n}^{*}$ </td></tr><tr><td>LE-LSTM</td><td>0.0718</td><td>0.0254</td><td>0.1785</td><td>0.0826</td></tr><tr><td>LE-TCN</td><td>0.0856</td><td>0.0268</td><td>0.1570</td><td>0.0929</td></tr><tr><td>LE-ConvMN</td><td>0.0802</td><td>0.0301</td><td>0.1588</td><td>0.0628</td></tr><tr><td>sBERT-OHEM</td><td>0.0614</td><td>0.0242</td><td>0.0753</td><td>0.0265</td></tr></table>

1. 基于 BERT 的模型采用 $\mu$ 律归一化，而 TCN、LSTM 和 LE-ConvMN 采用最大最小归一化。  
$2 , \ ^ { 6 } \ast ^ { 9 }$ 表示结果是在跨受试者情况下验证的。

TABLE IV  
不同模型在 INTEL I7-10875H CPU 上的推理时间（IT）

<table><tr><td>模型</td><td>TCN</td><td>LSTM</td><td>LE-ConvMN</td><td>BERT-based</td><td>sBERT-based</td></tr><tr><td>IT(ms)</td><td>3.99</td><td>48.54</td><td>18.23</td><td>31.78</td><td>32.22</td></tr></table>

1. OHEM 和 GHM 不影响推理，而平滑层会导致额外的时间消耗。我们在表中根据是否应用平滑层，将 BERT 的变体分为基于 BERT（BERT-based）和基于 sBERT（sBERT-based）两部分。

表 Table I 和 II 中的结果表明，在跨被试（cross-subject）场景下，我们的方法显著优于其他模型。当应用 Min-Max 归一化时，基于 BERT 的模型与 LE-ConvMN 表现相当。然而，尽管 $\mu$-law 归一化为基于 BERT 的模型带来了显著提升，但对 LE-ConvMN 的性能几乎没有影响。因此，我们的方法显著优于所有其他模型，并在跨被试场景下取得了最先进（state-of-the-art）的性能。GHM 和 OHEM 在多个被试上表现近乎相当且略有提升，但它们对不同单个被试的性能影响并不稳定。采样点越多，GHM 表现更好的可能性就越高。与特定被试（subject-specific）模型相比，跨被试模型不可避免地会导致性能下降，但这种下降在可接受范围内。然而，除了用于训练的被试之外，跨被试模型在更多被试上的表现仍然不够理想。

推理时间是指模型从 sEMG 中估计运动所需的时间。我们在相同的 CPU（Intel i7-10875H）上针对被试 S1 的测试数据集运行了所有方法。采用平均值来评估性能。结果如表 Table IV 所示。

根据结果，TCN 是最高效的方法，而基于 BERT 的方法仅优于 LSTM，这是因为推理操作无法并行执行，从而使 BERT 在时间开销上失去了优势。LE-ConvMN 的参数量少于 LSTM，这使得它比 LSTM 和基于 BERT 的方法更快。

总之，带有平滑层并使用 OHEM 机制训练的 BERT 在基于 BERT 的变体中表现最佳。在特定被试和跨被试场景下，我们的方法在质量和稳定性方面均优于 TCN。尽管 TCN 推理更快，但在质量上有所欠缺。此外，在特定被试和跨被试场景下，我们的方法在所有评估指标上均优于 LSTM，包括质量、效率以及质量和效率的稳定性。在特定被试场景下，我们的方法在质量上与 LE-ConvMN 表现相当，但在推理效率上较低。然而，基于 BERT 的方法可以并行训练，这使我们能够更快地获得特定被试模型并具备更高的估计稳定性。我们的方法与 LE-ConvMN 各有优势。然而，在跨被试场景下，除推理之外，我们的方法在所有方面均优于 LE-ConvMN。我们的方法在特定被试场景下表现出色，并在跨被试场景下取得了最先进的性能。

## V. 讨论 (DISCUSSION)

提出了基于 BERT 的模型来从表面肌电信号（sEMG）中估计手指关节运动，并与 LSTM、TCN 和 LE-ConvMN 进行了对比。采用相关系数（CC）和归一化均方根误差（NRMSE）来衡量估计的质量，并使用每轮（epoch）平均耗时、收敛时间和推理时间来衡量效率。结果表明，在所选的 10 名单一受试者中，基于 BERT 的模型性能优于经典模型，但在特定受试者（subject-specific）的情况下，LE-ConvMN 仍然是最先进（state-of-the-art）的模型。然而，所提出的新方法在同时面向多个受试者时显著优于所有其他模型，也就是说，我们的方法具有更强的泛化能力。

此外，基于 BERT 的模型训练效率更高且收敛更快，因为其结构允许模型进行并行训练。但由于其独特的自注意力机制，模型在实际应用的推理效率方面很难获益。难例挖掘（hard sample mining）和平滑层策略带来了一定程度的轻微改善。在 Ninapro 数据集上没有进一步的提升，这可能是因为 Ninapro DB2 中的数据量较小且已经过良好处理，几乎没有噪声，且单个个体的特征具有同质性。然而，当我们将 GHM 或 OHEM 应用于多个受试者时，由于受试者特征数量的增加，难例的数量也随之增多，此时难例挖掘能够带来性能的提升，且时间成本在可接受范围内。平滑层则使得估计结果更加平滑和稳定，更贴近真实情况。

尽管已有基于迁移学习的方法可以让模型适应不同的受试者，但要找到一个能够同时拟合多个受试者的模型依然十分困难。不过，本研究提出的已训练模型无法直接应用于新的受试者，除非新受试者参与了训练阶段。也就是说，当我们需要已训练的模型适用于新受试者时，必须将新受试者的训练数据与先前受试者的数据合并，而这有时是非常耗时的。将我们的方法推广到新受试者的更高效途径仍然是迁移学习。鉴于 BERT 本身就是一个高质量的预训练模型 [24]，基于 BERT 的方法有望为迁移学习方法做出贡献，这也将是我们未来的工作方向。目前，设计一种能够适应通用个体的普适方法仍然是一项挑战，而我们的方法在这一方面取得了重要进展。

我们的工作存在一些局限性。本文仅选择了手部健全的受试者，导致对我们方法的通用性缺乏充分的验证。在选择受试者、通道和动作时，我们刻意避开了由 Ninapro 采集误差导致的不合理或不良数据，从而缺少了对模型鲁棒性的验证。推理时间应当进一步缩短以满足实际应用的需求。至于基于 BERT 的结构，它们基于 Transformer 并受益于注意力机制，但这导致了较高的推理延迟。最近有许多关于高效 Transformer [31] 的研究以提高基于 Transformer 结构的效率，这可以作为后续改进的方向。

## VI. 结论与未来工作

在本文中，我们提出了一种基于 BERT 的手部连续运动估计方法。为了从 sEMG 信号中提取时空信息，设计了一种基于 BERT 的结构，以更好地满足临床使用中跨受试者场景的需求。同时引入了 μ-law 归一化，以更好地利用 sEMG 小幅度信号中蕴含的隐藏信息。在训练阶段应用了 GHM 和 OHEM，以更稳定地从 sEMG 中估计手部运动。随后，将两种经典算法和一种近期提出的基于 sEMG 信号的连续运动估计算法与我们基于 BERT 的方法进行了对比。结果表明，我们的方法在单受试者上取得了相当高的准确性和稳定性。此外，所提出的方法首次能够同时应用于多个不同的受试者，并在该场景下优于其他模型，取得了最先进（state-of-the-art）的结果。

在未来的工作中，可以考虑纳入更多的受试者、动作和通道，以评估模型的鲁棒性和稳定性。尽管 BERT 在训练时效率较高，但全注意力机制的计算在推理时非常耗时。由于连续运动估计对高效性有较高要求，也可以使用高效 Transformer 对该方法进行改进。尽管 GHM 和 OHEM 在我们的工作中被验证为仅有轻微帮助，但鉴于 GHM 和 OHEM 的独特机制，随着未来数据量的增加，难样本挖掘（hard sample mining）有望发挥巨大潜力。预计基于 Transformer、基于注意力的模型以及难样本挖掘策略将在人机交互与协作领域发挥越来越重要的作用。

## 参考文献

[1] P. D. Lawrence and W.-C. Lin, “Statistical decision making in the realtime control of an arm aid for the disabled,” IEEE Trans. Syst., Man, Cybern., vols. SMC-2, no. 1, pp. 35–42, Jan. 1972.

[2] M. Ahsan, M. I. Ibrahimy, and O. O. Khalifa, “EMG signal classification for human computer interaction: A review,” Eur. J. Sci. Res., vol. 33, no. 3, pp. 480–501, 2009.

[3] I. Kapandji, “The physiology of the joints, volume I, upper limb,” Amer. J. Phys. Med. Rehabil., vol. 50, no. 2, p. 96, 1971.

[4] M. Längkvist, L. Karlsson, and A. Loutfi, “A review of unsupervised feature learning and deep learning for time-series modeling,” Pattern Recognit. Lett., vol. 42, pp. 11–24, Jun. 2014.

[5] M. Zanghieri, S. Benatti, A. Burrello, V. Kartsch, F. Conti, and L. Benini, “Robust real-time embedded EMG recognition framework using temporal convolutional networks on a multicore IoT processor,” IEEE Trans. Biomed. Circuits Syst., vol. 14, no. 2, pp. 244–256, Apr. 2020.

[6] E. Rahimian, S. Zabihi, A. Asif, D. Farina, S. F. Atashzar, and A. Mohammadi, “TEMGNET: Deep transformer-based decoding of upperlimb sEMG for hand gestures recognition,” 2021, arXiv:2109.12379.

[7] L. Bi, A. G. Feleke, and C. Guan, “A review on EMG-based motor intention prediction of continuous human upper limb motion for human–robot collaboration,” Biomed. Signal Process. Control, vol. 51, pp. 113–127, May 2019.

[8] S. Duprey, A. Naaim, F. Moissenet, M. Begon, and L. Chèze, “Kinematic models of the upper limb joints for multibody kinematics optimisation: An overview,” J. Biomech., vol. 62, pp. 87–94, Sep. 2017.

[9] M. Sartori, G. Durandau, S. Došen, and D. Farina, “Robust simultaneous myoelectric control of multiple degrees of freedom in wrist-hand prostheses by real-time neuromusculoskeletal modeling,” J. Neural Eng., vol. 15, no. 6, Dec. 2018, Art. no. 066026.

[10] L. Pan, D. L. Crouch, and H. Huang, “Myoelectric control based on a generic musculoskeletal model: Toward a multi-user neural-machine interface,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 26, no. 7, pp. 1435–1442, Jul. 2018.

[11] Q. Zhang, R. Liu, W. Chen, and C. Xiong, “Simultaneous and continuous estimation of shoulder and elbow kinematics from surface EMG signals,” Frontiers Neurosci., vol. 11, p. 280, May 2017.

[12] F. Quivira, T. Koike-Akino, Y. Wang, and D. Erdogmus, “Translating sEMG signals to continuous hand poses using recurrent neural networks,” in Proc. IEEE EMBS Int. Conf. Biomed. Health Informat. (BHI), Mar. 2018, pp. 166–169.

[13] S. J. Pan and Q. Yang, “A survey on transfer learning,” IEEE Trans. Knowl. Data Eng., vol. 22, no. 10, pp. 1345–1359, Oct. 2009.

[14] J. Fan et al., “Improving semg-based motion intention recognition for upper-limb amputees using transfer learning,” Neural Comput. Appl., early access, pp. 1–11, Jul. 2021.

[15] U. Côté-Allard et al., “Deep learning for electromyographic hand gesture signal classification using transfer learning,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 27, no. 4, pp. 760–771, Apr. 2019.

[16] E. Rahimian, S. Zabihi, S. F. Atashzar, A. Asif, and A. Mohammadi, “XceptionTime: Independent time-window xceptiontime architecture for hand gesture classification,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), May 2020, pp. 1304–1308.

[17] W. Guo et al., “用于从表面肌电信号中精确估计手指运动学的长曝光卷积记忆网络,” J. Neural Eng., vol. 18, no. 2, 2021, Art. no. 026027.

[18] S. Hochreiter and J. Schmidhuber, “长短期记忆,” Neural Comput., vol. 9, no. 8, pp. 1735–1780, 1997.

[19] S. Bai, J. Z. Kolter, and V. Koltun, “用于序列建模的通用卷积与循环网络的实证评估,” CoRR, vol. abs/1803.01271, pp. 1–14, Mar. 2018.

[20] J. Long, E. Shelhamer, and T. Darrell, “用于语义分割的全卷积网络,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2015, pp. 3431–3440.

[21] F. Yu and V. Koltun, “基于空洞卷积的多尺度上下文聚合,” in Proc. 4th Int. Conf. Learn. Represent. (ICLR), Y. Bengio and Y. LeCun, Eds., San Juan, PR, USA, May 2016, pp. 1–10.

[22] X. Shi, Z. Chen, H. Wang, D.-Y. Yeung, W.-K. Wong, and W.-C. Woo, “卷积 LSTM 网络：一种用于降水临近预报的机器学习方法,” in Proc. Adv. Neural Inf. Process. Syst., vol. 28, 2015, pp. 1–9.

[23] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT：用于语言理解的深度双向 Transformer 预训练,” in Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics, Hum. Lang. Technol., vol. 1. Stroudsburg, PA, USA: Association for Computational Linguistics, Jun. 2019, pp. 4171–4186.

[24] A. Vaswani et al., “注意力就是你所需要的一切 (Attention is all you need),” in Proc. Adv. Neural Inf. Process. Syst., vol. 30, 2017, pp. 1–11.

[25] 话音频率的脉冲编码调制 (PCM), ITU, Geneva, Switzerland, 1988.

[26] E. Rahimian, S. Zabihi, A. Asif, D. Farina, S. F. Atashzar, and A. Mohammadi, “FS-HGR：基于肌电信号的手势识别少样本学习,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 29, pp. 1004–1015, 2021.

[27] A. Shrivastava, A. Gupta, and R. Girshick, “基于在线困难样本挖掘训练基于区域的目标检测器,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2016, pp. 761–769.

[28] B. Li, Y. Liu, and X. Wang, “梯度均衡单阶段检测器,” in Proc. AAAI Conf. Artif. Intell., 2019, vol. 33, no. 1, pp. 8577–8584.

[29] M. Atzori et al., “用于无创自然控制机械手假肢的肌电数据,” Sci. Data, vol. 1, no. 1, pp. 1–13, 2014.

[30] N. Ketkar and J. Moolayil, “PyTorch 简介,” in Deep Learning With Python. Berkeley, CA, USA: Apress, 2021, pp. 27–91.

[31] Y. Tay, M. Dehghani, D. Bahri, and D. Metzler, “高效 Transformer：综述,” ACM Comput. Surv., Apr. 2022.