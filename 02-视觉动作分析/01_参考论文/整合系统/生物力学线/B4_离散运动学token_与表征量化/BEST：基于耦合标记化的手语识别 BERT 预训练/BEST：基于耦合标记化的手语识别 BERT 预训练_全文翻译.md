# BEST：基于耦合标记化的手语识别 BERT 预训练

Weichao Zhao<sup>1</sup>, Hezhen Hu<sup>1</sup>\*, Wengang Zhou<sup>1,</sup> <sup>2†</sup>, Jiaxin Shi<sup>3</sup>, Houqiang Li<sup>1,</sup> <sup>2†</sup>

<sup>1</sup>CAS Key Laboratory of GIPAS, EEIS Department, University of Science and Technology of China (USTC) <sup>2</sup>Institute of Artificial Intelligence, Hefei Comprehensive National Science Center <sup>3</sup>Huawei Inc. {saruka, alexhu}@mail.ustc.edu.cn, {zhwg, lihq}@ustc.edu.cn, shijiaxin3@huawei.com

## 摘要

在本文中，我们致力于利用 BERT 预训练的成功经验，并对领域特定的统计特征进行建模，以赋能手语识别（Sign Language Recognition, SLR）模型。考虑到手部和身体在手语表达中的主导地位，我们将它们组织为姿态三元组单元，并以逐帧的方式输入到 Transformer 骨干网络中。预训练通过从受损的输入序列中重构被掩蔽的三元组单元来执行，从而学习三元组单元内部和外部之间的分层相关上下文线索。值得注意的是，与 BERT 中高度语义化的词标记（word token）不同，姿态单元是原本位于连续空间中的低级信号，这阻碍了直接采用 BERT 的交叉熵目标函数。为此，我们通过三元组单元的耦合标记化（coupling tokenization）来弥合这一语义鸿沟。它自适应地从姿态三元组单元中提取离散伪标签，用于表示语义手势/身体状态。预训练完成后，我们将预训练编码器与新添加的任务特定层一起，在下游 SLR 任务上进行微调。大量的实验验证了所提方法的有效性，在全部四个基准数据集上均取得了显著提升，达到了新的最先进性能。

## 1 引言

手语是聋人群体的主要交流工具。它具有独特的语法和词汇，非手语使用者往往难以理解。为了弥合这一交流鸿沟，自动手语识别（SLR）得到了广泛的研究，具有深远的社会影响力。作为一项基础任务，孤立词手语识别（isolated SLR）旨在进行词元级（gloss-level）的识别，是一个细粒度的分类问题。在本文中，我们重点关注这一任务。

由于高昂的标注成本，目前已标注的手语数据资源十分有限。由于常见的基于深度学习的方法极度依赖数据，它们在手语识别（Sign Language Recognition, SLR）任务上容易发生过拟合。为此，研究人员在 SLR 方面做出了若干尝试。例如，考虑到手部的主导作用，一些方法 (Hu, Zhou, and Li 2021; Albanie et al. 2020) 将裁剪后的手部序列用作辅助信息。HMA (Hu, Zhou, and Li 2021) 提出以手部网格（hand mesh）作为中间约束，在模型感知（model-aware）范式下识别人类手语。然而，这些方法直接在目标基准上进行优化，未能利用手语领域的通用统计规律。

值得注意的是，以 BERT (Devlin et al. 2018) 为代表的自监督预训练技术在自然语言处理（Natural Language Processing, NLP）领域取得了巨大的成功。BERT 建立在强大的 Transformer (Vaswani et al. 2017) 主干网络之上，并设计了一个巧妙的预训练任务（pretext task），即掩码语言建模（masked language modeling, MLM）。其目标是从受损的输入序列中重构被掩码的词标记（word tokens），其优化目标是通过交叉熵来实现联合词概率分布的最大化。然而，在视频手语识别（video SLR）中借鉴其成功经验的主要障碍在于输入信号的不同特性。在 NLP 中，输入的词标记是离散且预先定义的，具有高度的语义性。相比之下，手语的视频信号在空间和时间维度上都是连续的。这种信号属于相当底层的特征，使得原有的 BERT 目标无法直接适用。此外，由于手语视频的主要特征是手部和身体动作，直接采用 BERT 框架可能并非最佳选择。

为解决上述问题，我们提出了一种专为手语设计的自监督可预训练框架，命名为 BEST。聚焦于手语的主要特征，我们将手部和身体组织为姿态三元组单元（pose triplet unit）。该三元组单元经过嵌入后被输入到 Transformer 主干网络中。基本上，我们的框架包含两个阶段，即自监督预训练和下游微调。在预训练期间，我们提出了掩码单元建模（masked unit modeling, MUM）预训练任务以捕获上下文线索。输入的手部或身体单元嵌入被随机掩码，随后框架从该受损的输入序列中重构被掩码的单元。与 BERT 类似，自重构是通过交叉熵目标来进行优化的。为此，我们将姿态三元组单元联合分词（tokenize）为伪标签，用以表示手势/身体状态。预训练完成后，预训练的 Transformer 编码器与新添加的预测头一起进行微调，以执行 SLR 任务。

我们的贡献总结如下：

• 我们提出了一个自监督可预训练框架。该框架借鉴了 BERT 的成功经验，并结合了针对手语领域的专门设计。

• 我们将主要的手部和身体动作组织为姿态三元组单元（pose triplet unit），并提出了掩码单元建模（masked unit modeling，MUM）预训练任务。为了利用 BERT 目标，我们通过对姿态三元组单元进行耦合分词（coupling tokenization）来为该任务生成伪标签。

• 在下游 SLR 上的大量实验验证了我们所提方法的有效性，在四个基准测试集上均取得了显著的性能提升，达到了新的最先进水平。

## 2 相关工作

在本节中，我们将简要回顾几个相关主题，包括手语识别和自监督预训练。

## 2.1 手语识别

近年来，手语识别受到了广泛关注 (Koller et al. 2018; Momeni et al. 2022; Niu and Mak 2020; Jin and Zhao 2021; Hu, Zhou, and Li 2021; Li et al. 2020b)。通常，根据输入模态，这些研究工作可以分为两类，即基于 RGB 的方法和基于姿态的方法。

基于 RGB 的方法。早期关于 SLR 的工作 (Farhadi, Forsyth, and White 2007; Fillbrandt, Akyol, and Kraiss 2003; Starner 1995) 专注于针对手形变化和身体运动计算的手工特征。随着卷积神经网络（CNNs）在计算机视觉领域的普及，SLR 中的许多工作采用 CNN 作为骨干网络 (Selvaraj et al. 2022; Sincan and Keles 2020; Joze and Koller 2018; Koller et al. 2018; Hu, Zhou, and Li 2021)。例如，由于 3D CNN 对时空依赖性的表征能力，它被广泛采用 (Huang et al. 2018; Joze and Koller 2018; Albanie et al. 2020; Li et al. 2020b)。

基于姿态的方法。姿态模态是人体动作的一种紧凑且高级的表示，包含骨骼关节之间的物理连接 (Li et al. 2018; Ng et al. 2021; Yan, Xiong, and Lin 2018)。为了提取姿态数据的语义表示，一些工作 (Li et al. 2018; Du, Wang, and Wang 2015) 探索了将图卷积网络（GCNs）作为骨干网络。这些基于 GCN 的方法 (Camgoz et al. 2018; Li et al. 2018; Yan, Xiong, and Lin 2018; Min et al. 2020) 在动作识别中展现出了令人瞩目的性能。Tunga 等人 (Tunga, Nuthalapati, and Wachs 2021) 结合了 GCN 和 Transformer，以基于手语姿态序列捕获时空信息用于手语识别。在这项工作中，鉴于姿态数据的紧凑性，我们将其用作我们的输入模态。

## 2.2 自监督预训练

自监督预训练方法在自然语言处理（NLP）和计算机视觉（CV）领域取得了显著成功，该方法充分利用大规模无标注数据来学习通用特征表示，以用于广泛的下游任务。在 NLP 中，凭借 Transformer (Vaswani et al. 2017) 强大的建模能力，许多工作提出在此骨干网络上进行预训练以获得通用表示 (Devlin et al. 2018; Lewis et al.

2019; Conneau and Lample 2019）。BERT 是最流行的方法之一，它巧妙地设计了掩码语言建模（MLM）预训练任务。MLM 通过受损的输入句子预测被掩码的词元（word token），旨在捕获文本语料库中的上下文线索。

受 BERT 的启发，一些研究尝试将其成功经验迁移到 CV 任务中（Su et al. 2019; Sun et al. 2019; Zhu and Yang 2020; Bao, Dong, and Wei 2021; He et al. 2022）。NLP 和 CV 之间的输入信号存在不同的特征。与具有语义的离散词元不同，CV 任务中的信号通常是低级且连续的，这使得原本的 BERT 目标函数不再适用。解决该问题的一种方法是将其目标转变为回归。He 等人（He et al. 2022）提出了掩码自编码器（MAE），使用回归目标来重建被掩码图像块的缺失像素。Hu 等人（Hu et al. 2021a）提出了一种基于手部姿态数据自重建的手语预训练模型。Jiang 等人（Jiang, Camgoz, and Bowden 2021）尝试将 BERT 风格的精细化姿态用于手语。根据（Ramesh et al. 2021）的研究，前一种方法可能会过多关注短程依赖，从而损害下游性能。因此，一些方法转向对输入信号进行离散化分词（tokenizing）以提供离散的伪标签。BEiT（Bao, Dong, and Wei 2021）提出了以离散化图像块为监督的掩码图像建模（MIM）。它最初是针对基于图像的任务设计的，由于任务特征和输入模态的不同，不能直接应用到基于视频的手语领域。

## 3 方法

我们提出的方法包括两个阶段，即自监督预训练和下游任务微调。如图 Figure 1 所示，在预训练期间，我们首先利用离散变分自编码器（discrete variational autoencoder）来学习上半身、左手和右手的码本（codebook）。它对姿态三元组单元进行耦合分词（coupling tokenization），作为后续预训练任务的伪标签。然后，我们通过设计的掩码单元建模（MUM）预训练任务对 BEST 模型进行预训练，以捕获上下文线索。最后，我们添加特定于任务的层，并在下游 SLR 任务上对预训练参数进行微调。

## 3.1 预训练中的分词

在预训练期间，分词过程为我们设计的借口任务（pretext task）提供伪标签。受 VQ-VAE (Van Den Oord, Vinyals et al. 2017) 的启发，我们利用离散变分自编码器（d-VAE）将姿态三元组单元联合转换为三元组词元（tokens）（身体、左手和右手）。我们所使用的姿态三元组单元 $J _ { s i g n }$ 由两个手部姿态 $J _ { l e f t } , J _ { r i g h t }$ 以及一个上半身姿态 ${ J } _ { b o d y }$ 组成。d-VAE 包含三个部分，即编码器、量化器和解码器。编码器将姿态三元组单元映射为中间隐向量 $z = \operatorname { E n c } ( J _ { s i g n } )$。量化器负责将每个向量转换为来自手部码本 $\mathcal { V } _ { h a n d } = \{ h _ { k } \} _ { k = 1 } ^ { M _ { 1 } }$ 和上半身码本 $\mathcal { V } _ { b o d y } = \{ d _ { k } \} _ { k = 1 } ^ { M _ { 2 } }$ 的码字（codewords）。量化后的向量 $z _ { \mathbf { q } }$ 的计算

![](images/9cd4b7f655ed5b2e3598418f00d4e75f1c10a7c1a589dd384ef85587be1ea188.jpg)  
Figure 1: 我们提出的 BEST 框架在预训练阶段的示意图。BEST 主要包含四个组件，即预训练分词器、姿态嵌入层、Transformer 编码器和解码器。输入的姿态三元组单元由身体姿态、左手姿态和右手姿态组成。给定一个姿态三元组单元序列，我们利用所提出的 MUM 借口任务来预训练我们的框架。[M] 表示输入序列的可学习掩码词元（masked token）。同时，预训练分词器耦合将三元组单元离散化为伪三元组单元标签，以监督预训练过程。

方式如下：

$$
\begin{array}{l} z _ {\mathbf {q}} = \text {Concat} (h _ {k ^ {l}}, h _ {k ^ {r}}, d _ {k ^ {b}}), z = \text {Concat} (z _ {l}, z _ {r}, z _ {b}), \\ k ^ {l} = \text {Q\_hand} (z _ {l}) = \underset {k \in \{1, 2, \dots , M _ {1} \}} {\arg \min} | | z _ {l} - h _ {k} | |, \\ k ^ {r} = \text {Q\_hand} (z _ {r}) = \underset {k \in \{1, 2, \dots , M _ {1} \}} {\arg \min} | | z _ {r} - h _ {k} | |, \\ k ^ {b} = \text {Q\_body} (z _ {b}) = \underset {k \in \{1, 2, \dots , M _ {2} \}} {\arg \min} | | z _ {b} - d _ {k} | |, \end{array}\tag{1}
$$

其中 $l , r$ 和 b 分别表示“左（left）”、“右（right）”和“上半身（upper $\mathbf { b o d y } ^ { \prime }$）”的缩写。在接下来的介绍中，我们依然采用缩写 $l , r$ 和 b。隐向量 z 由姿态三元组单元的三个局部表征组成，即 $z _ { l } , z _ { r } ,$ 和 z<sub>b</sub>。Q hand(·) 是手部量化编码器，它将向量映射到手部码本的索引，该索引满足输入向量与对应码字之间的最小语义距离。与手部量化器类似，Q body(·) 表示用于上半身的身体量化编码器。量化后的向量 $z _ { \mathbf { q } }$ 由来自两个码本的三个码字级联而成。基于该量化向量，解码器旨在重建输入的姿态三元组单元。解码器的输出计算如下：

$$
\hat {J} _ {s i g n} = \mathrm{Dec} (z _ {\mathbf {q}}),\tag{2}
$$

其输出为重建后的手部和身体姿态 $( \hat { J } _ { l e f t } , \hat { J } _ { r i g h t }$ 和 $\hat { J } _ { b o d y } )$。

该分词器在自监督预训练之前进行训练。然而，分词过程是不可微的。为了将梯度从解码器反向传播到编码器，我们利用直通估计器（straight-through estimator，Bengio, Leonard, and Courville 2013）将梯度从解码器复制到编码器。d-VAE 的训练目标函数定义如下：

$$
\begin{array}{r l} & {\mathcal {L} _ {d - V A E} = \mathcal {L} _ {h a n d} + \beta_ {1} \mathcal {L} _ {b o d y} + \beta_ {2} | | \mathbf {s g} [ z ] - z _ {\mathbf {q}} | | _ {2} ^ {2}} \\ & {\qquad + \beta_ {3} | | \mathbf {s g} [ z _ {\mathbf {q}} ] - z | | _ {2} ^ {2},} \\ & {\mathcal {L} _ {h a n d} = | | \hat {J} _ {l e f t} - J _ {l e f t} | | + | | \hat {J} _ {r i g h t} - J _ {r i g h t} | |,} \\ & {\mathcal {L} _ {b o d y} = | | \hat {J} _ {b o d y} - J _ {b o d y} | |,} \end{array}\tag{3}
$$

其中 $\mathcal { L } _ { h a n d }$ 和 $\mathcal { L } _ { b o d y }$ 分别表示 2D 手部和上半身姿态的重建损失。$\mathbf { s g } [ \cdot ]$ 表示停止梯度算子，$\beta _ { 1 } , \bar { \beta } _ { 2 }$ 和 $\beta _ { 3 }$ 是用于平衡这四个损失影响的权重因子。

## 3.2 预训练阶段的框架架构

在预训练期间，该框架包含四个组件，即姿态嵌入层、预训练姿态分词器、Transformer 编码器和解码器。

姿态嵌入层（Pose Embedding Layer）。该嵌入层旨在从输入的姿态三元组单元中提取特征嵌入。由于姿态具有物理连接关系，我们利用 (Cai et al. 2019) 中提出的图卷积网络（GCN）作为姿态嵌入层。具体而言，给定包含 $T$ 帧的姿态序列 $V _ { s i g n } = \{ J _ { s i g n , t } \} _ { t = 1 } ^ { T }$，它以逐帧方式从身体、左手和右手提取姿态三元组单元 $f _ { s i g n , t } \in \mathbb { R } ^ { D }$，并将它们拼接以供后续模块使用。三元组单元的每个部分具有相同的特征维度 $\begin{array} { r } { D _ { p a r t } \dot { = } \frac { 1 } { 3 } D } \end{array}$。

分词器（Tokenizer）。如第 3.1 节所述，给定姿态序列 $\{ J _ { s i g n , t } \} _ { t = 1 } ^ { \hat { T } }$，我们将其分词为用于预训练的伪标签 $\{ k _ { t } \} _ { t = 1 } ^ { T }$，其中 $k _ { t }$ 表示按照公式 (1) 计算得到的 $\mathbf { \bar { \boldsymbol { k } } } _ { t } ^ { l } , k _ { t } ^ { \bar { r } }$ 和 $k _ { t } ^ { b }$ 的拼接。分词器仅用于伪标签推理，其所有参数在预训练期间均被冻结。

Transformer 编码器（Transformer Encoder）。给定来自姿态嵌入层的特征序列 $\mathrm { F } _ { s i g n } = \{ f _ { s i g n , t } \} _ { t = 1 } ^ { T }$，我们进一步添加由位置编码（Vaswani et al. 2017）实现的时间嵌入 $f _ { t e m p , t } \in \mathbb { R } ^ { D }$。输入序列 $\mathrm { { \bar { F } _ { 0 } } }$ 的计算方式如下：

$$
\begin{array}{c} \mathrm{F} _ {0} = [ f _ {i n, 1}, \dots , f _ {i n, T} ], \\ f _ {i n, t} = f _ {m a s k, t} + f _ {t e m p, t}, \quad t \in \{1, \dots , T \}, \\ \mathrm{F} _ {m} = \mathrm{Mask} (\mathrm{F} _ {s i g n}) = [ f _ {m a s k, 1}, \dots , f _ {m a s k, T} ], \end{array}\tag{4}
$$

其中 Mask(·) 表示用于嵌入序列的掩码操作符，被掩码的帧位置表示为 $\breve { \mathcal { M } } \in \{ 1 , \cdot \cdot \cdot , T \} ^ { \alpha \cdot T }$，其中 α 表示掩码率。我们将在第 3.3 节中解释掩码建模。输入序列 $\mathrm { F _ { 0 } }$ 被输入到 Transformer 编码器中。Transformer 编码器包含 N 层 Transformer 块 $\mathrm { F } _ { l } = B l o c k ( \mathrm { F } _ { l - 1 } )$，其中 $l = 1 , \cdots , N$。最后一层的输出序列 $\mathrm { F } _ { N } = [ f _ { o u t , 1 } , \cdot \cdot \cdot , f _ { o u t , T } ]$ 被用作姿态三元组单元的编码表示，其中 $f _ { o u t , t }$ 对应于第 t 帧。

解码器（Decoder）。给定输出序列 $\mathrm { F } _ { N } ,$、掩码帧位置 M 以及伪标签 $\{ k _ { t } \} _ { t = 1 } ^ { T } ,$ ，我们利用解码器来重构姿态三元组单元。由于每个输出特征 $f _ { o u t , t }$ 包含三个部分，即 $f _ { o u t , t } ^ { l } , f _ { o u t , t } ^ { r }$ 和 $f _ { o u t , t } ^ { b } ,$ ，我们分别将各部分的被掩码位置记录到 $\mathcal { M } ^ { l } , \mathcal { M } ^ { r }$ 和 $\mathcal { M } ^ { b }$ 中。$\mathcal { M } = \mathcal { M } ^ { l } \cup \mathcal { M } ^ { r } \cup \mathcal { M } ^ { b }$ 。对于每个特征 $f _ { o u t , t }$ ，我们使用 softmax 分类器来预测各部分的对应标签，

$$
\begin{array}{r l} & p _ {\mathrm{hand}} (k _ {t} ^ {l} | f _ {o u t, t}) = \mathrm{softmax} _ {k _ {t} ^ {l}} (W _ {1} f _ {o u t, t} ^ {l} + b _ {1}), \quad t \in \mathcal {M} ^ {l}, \\ & p _ {\mathrm{hand}} (k _ {t} ^ {r} | f _ {o u t, t}) = \mathrm{softmax} _ {k _ {t} ^ {r}} (W _ {1} f _ {o u t, t} ^ {r} + b _ {1}), \quad t \in \mathcal {M} ^ {r}, \\ & p _ {\mathrm{body}} (k _ {t} ^ {b} | f _ {o u t, t}) = \mathrm{softmax} _ {k _ {t} ^ {b}} (W _ {2} f _ {o u t, t} ^ {b} + b _ {2}), \quad t \in \mathcal {M} ^ {b}, \end{array}\tag{5}
$$

其中 $l , r$ 和 b 分别表示左（left）、右（right）和上半身（upper body）的缩写。$W _ { 1 } \ \in \ \mathbb { R } ^ { | \mathcal { V } _ { h a n d } | \times D _ { p a r t } }$ 和 $W _ { 2 } \in$ $\mathbb { R } ^ { | \dot { \nu } _ { b o d y } | \times \dot { D } _ { p a r t } }$ 表示投影矩阵。$b _ { 1 } \in \mathbb { R } ^ { | \nu _ { i } | }$ <sub>hand</sub>| 和 $b _ { 2 } \in \mathbb { R } ^ { | \mathcal { V } _ { b o d y } | }$ 表示偏置。

## 3.3 预训练中的前置任务与目标函数

我们设计的代理任务（pretext task）是 MUM，旨在挖掘内部和外部三元姿态单元之间的层次相关上下文。给定一个长度为 T 的包含三元姿态单元的姿态序列，我们首先随机选择 $\alpha \cdot T$ 帧进行掩码操作。为了明确说明，我们将姿态三元单元的三个部分分别定义为 $f _ { s i g n , t } ^ { l } , f _ { s i g n , t } ^ { r }$ 和 $f _ { s i g n , t } ^ { b }$。如果一个单元被掩码，则以 50% 的概率使用一个可学习的掩码标记（masked token）$\mathbf { e } _ { m a s k } \ \in \ \mathbb { R } ^ { D _ { p a r t } }$ 来替换该三元单元的每个部分。因此，被掩码的三元单元包括三种掩码情况：仅手部掩码、仅身体掩码以及手部-身体掩码。手部掩码的情况旨在赋予我们的模型捕获局部手部上下文的能力。与手部对应情况相比，身体掩码的情况旨在挖掘全局身体动作的上下文。手部-身体掩码的情况意味着身体和手部特征同时被掩码。由于手语是通过手部和身体的协同作用来传达完整含义的，因此我们期望在这种情况下挖掘手部和身体之间的相关上下文线索。因此，我们利用 MUM 来预训练我们的方法，并进行了消融实验以验证我们的构想。

整体预训练目标是在给定损坏序列的情况下最大化正确标签的对数似然，其计算方式如下：

$$
\mathcal {L} _ {\mathrm{pre-train}} = \max \sum_ {V _ {s i g n} \in \mathcal {D}} \mathbb {E} _ {\mathcal {M}} \left[ \sum_ {t \in \mathcal {M}} \log p (k _ {t} | V _ {s i g n}) \right]
$$

$$
\begin{array}{r l} & V _ {s i g n} \in \mathcal {D} \qquad \lfloor t \in \mathcal {M} \\ \sum_ {t \in \mathcal {M}} \log p (k _ {t} | V _ {s i g n}) = \sum_ {t \in \mathcal {M} ^ {l}} \log p _ {\text {hand}} (k _ {t} ^ {l} | f _ {o u t, t} ^ {l}) \\ & + \sum_ {t \in \mathcal {M} ^ {r}} \log p _ {\text {hand}} (k _ {t} ^ {r} | f _ {o u t, t} ^ {r}) \\ & + \sum_ {t \in \mathcal {M} ^ {b}} \log p _ {\text {body}} (k _ {t} ^ {b} | f _ {o u t, t} ^ {b}), \end{array}\tag{6}
$$

其中 $\mathcal { D }$ 为训练语料库，$\mathcal { M }$ 表示被掩码的帧位置，$f _ { o u t , t }$ 是被掩码的三元单元。$\mathcal { M } ^ { l }$、$\mathcal { M } ^ { r }$ 和 $\mathcal { M } ^ { b }$ 分别表示左手、右手和身体的被掩码位置。

## 3.4 下游微调

预训练完成后，我们直接在下游 SLR 任务下微调参数。我们用一个简单的 MLP 替换解码器作为预测头。在微调过程中，我们不对任何三元姿态单元进行掩码，并使用真实标签监督预测头的输出。此外，仅凭姿态信息不足以传达手语的完整含义。我们采用后融合策略将姿态数据与 RGB 数据的预测结果相加。在我们的实验中，仅使用姿态数据以及融合 RGB 数据的方法分别表示为 Ours 和 Ours (+R)。

## 4 实验

## 4.1 数据集与评估指标

数据集。我们在四个公开手语数据集上进行实验，即 NMFs-CSL (Hu et al. 2021b)、SLR500 (Huang et al. 2018)、WLASL (Li et al. 2020a) 和 MSASL (Joze and Koller 2018)。所有数据集的训练集均参与了预训练阶段。表 1 呈现了上述数据集的概览。

NMFs-CSL 是一个大规模中国手语（Chinese sign language, CSL）数据集，词汇量为 1,067。所有样本分为 25,608 个训练样本和 6,402 个测试样本。SLR500 是另一个 CSL 数据集，包含由 50 位手语者表演的 500 个日常词汇。它共包含 125,000 个样本，其中分别有 90,000 个和 35,000 个样本用于训练和测试。上述两个数据集均在受控的实验室场景中采集样本。

WLASL 是一个大规模美国手语（American sign language, ASL）数据集，包含由 100 多位手语者表演的 2000 个词汇。它总共由 21,083 个样本组成。特别地，它选取了出现频率最高的前 K 个词，其中 $\textit { K } ~ = ~ \{ 1 0 0 , 3 0 0 \}$，并将它们组织为两个子集，即 WLASL100 和 WLASL300。MSASL 是另一个流行的 ASL 数据集，共有 25,513 个样本，词汇量为 1000。与 WLASL 类似，它也提供了两个子集，分别命名为 MSASL100 和 MSASL200。与 NMFs-CSL 和 SLR500 不同，WLASL 和 MSASL 从网络上收集数据，由于其处于无限制的真实生活场景中，带来了更多挑战。

<table><tr><td>名称</td><td>语言</td><td>词汇量</td><td>视频数</td><td>手语者</td><td>来源</td></tr><tr><td>WLASL</td><td>ASL</td><td>2000</td><td>21.1k</td><td>119</td><td>网络、词典</td></tr><tr><td>NMF-CSL</td><td>CSL</td><td>1067</td><td>32.0k</td><td>10</td><td>实验室</td></tr><tr><td>MSASL</td><td>ASL</td><td>1000</td><td>25.5k</td><td>222</td><td>网络、词典</td></tr><tr><td>SLR500</td><td>CSL</td><td>500</td><td>125k</td><td>50</td><td>实验室</td></tr></table>

Table 1: 公开孤立词 SLR 数据集统计信息。ASL 表示美国手语（American Sign Language），CSL 表示中国手语（Chinese Sign Language）。

评估指标。在评估中，我们报告下游 SLR 任务的分类准确率，即 Top-1（T-1）和 Top-5（T-5）。遵循（Li et al. 2020a; Joze and Koller 2018），我们同时采用每实例（per-instance，P-I）和每类别（per-class，P-C）准确率指标。每实例准确率在整个测试数据上计算。每类别准确率是测试集中存在的手语类别的平均值。对于 WLASL 和 MSASL，由于这两个数据集每个类别的样本不均衡，我们同时报告每实例和每类别准确率。对于 NMFs-CSL 和 SLR500，由于每个类别的样本数量相同，我们仅报告每实例准确率。

## 4.2 实验设置

数据准备与处理。我们提出的方法利用姿态模态来表示手部和身体动作。由于手语数据集中未提供可用的姿态标注，我们使用现成的姿态检测器 MMPose 来提取 2D 姿态关键点。在每一帧中，提取的 2D 姿态包括 49 个关节点，包含 7 个上半身关节点和 42 个手部关节点。此外，考虑到三元组单元之间的不同尺度，我们分别裁剪了身体、左手和右手的姿态，并将它们缩放到 $2 5 6 ~ \times$ 256。

模型超参数。对于分词器（tokenizer），手部码本 $M _ { 1 }$ 和身体码本 $M _ { 2 }$ 的词表大小分别为 1000 和 500。两个码本中每个码字的维度均为 512。公式 3 中的权重因子 $\beta _ { 1 }$、$\beta _ { 2 }$ 和 $\beta _ { 3 }$ 分别设置为 0.1、1.0 和 0.9。在预训练期间，Transformer 编码器包含 8 个头，Transformer 编码器的输入大小 D 为 1536，逐位置前馈网络维度为 2048。

训练设置。我们在实验中采用了 Adam（Kingma and Ba 2014）优化器。对于分词器训练，我们将初始学习率设置为 0.001，并且每 10 个 epoch 按 0.1 的因子进行衰减。对于预训练，权重衰减和动量分别设置为 0.01 和 0.9。学习率设置为 0.0001，包含 6 个 epoch 的预热（warmup），并采用线性学习率衰减。对于下游 SLR 任务，学习率初始化为 0.0001，每 10 个 epoch 按 0.1 的因子衰减。我们在训练期间使用随机扰动矩阵来扰动输入姿态序列的坐标，以缓解过拟合。遵循（Hu et al. 2021a），我们在训练和测试期间分别使用随机采样和中心时间采样在时间上选取 32 帧。所有实验均由 PyTorch 实现并在 NVIDIA RTX 3090 上进行训练。

## 4.3 与最先进方法的比较

在本节中，我们在四个公开数据集上将我们的方法与先前的最先进（state-of-the-art）方法进行比较。遵循（Hu et al. 2021a），我们按输入模态对先前的方法进行分组，即基于姿态的方法和基于 RGB 的方法。

NMFs-CSL 数据集。如 Table 5 所示，我们与以往的方法进行了对比。SignBERT (Hu et al. 2021a) 是一种利用手部先验对手机序列进行建模的自监督预训练方法。与 SignBERT 相比，我们的方法在不同设置下的 Top-1 准确率依然取得了最佳性能。GLE-Net (Hu et al. 2021b) 是当前最先进的方法，它利用了来自全局和局部视角的判别性线索。与 GLE-NET 相比，Ours (+R) 超越了它，取得了 79.2% 的 Top-1 准确率。

MSASL 数据集。如 Table 2 所示，ST-GCN (Yan, Xiong, and Lin 2018) 的表现逊于基于 RGB 的方法，这可能是由于姿态检测失败所导致的。与 ST-GCN (Yan, Xiong, and Lin 2018) 相比，自监督学习方法，即 SignBERT (Hu et al. 2021a) 和我们提出的方法，通过利用预训练期间存储的统计数据缓解了这一问题。我们的方法优于 SignBERT (Hu et al. 2021a)，在 MSASL100、MSASL200 和 MSASL1000 上的 Top-1 单实例（per-instance）准确率分别提升了 4.89%、5.96% 和 9.28%。值得注意的是，我们的方法甚至取得了与基于 RGB 的方法相当的性能。当与 RGB 方法融合时，性能得到了进一步提升。

WLASL 数据集。与 MSASL 类似，WLASL 是一个具有非受限录制条件和不平衡样本分布的极具挑战性的数据集。如 Table 3 所示，Pose-TGCN (Li et al. 2020a) 和 PSLR (Tunga, Nuthalapati, and Wachs 2021) 由于姿态估计错误而表现欠佳。HMA (Hu, Zhou, and Li 2021) 利用手部统计模型来精炼姿态并提高性能。BSL (Albanie et al. 2020) 和 TCK (Li et al. 2020b) 利用外部 RGB 数据来增强模型的鲁棒性并提升性能。与它们相比，我们的方法在所有子集上都取得了最佳性能。

SLR500 数据集。如 Table 4 所示，与基于手工设计特征的 STIP (Laptev 2005) 和 GMM-HMM (Tang et al. 2015) 相比，基于深度学习的方法 (Yan, Xiong, and Lin 2018; Qiu, Yao, and Mei 2017; Hu et al. 2021a) 取得了显着的性能提升。在基于姿态和基于 RGB 的方法中，我们的方法依然取得了最新的最先进性能。

## 4.4 消融实验

在本节中，我们进行了消融实验，以验证所提出方法的有效性，并为我们的框架选择合适的超参数。为了公平比较，实验在 MSASL 数据集上进行，我们报告了单实例（per-instance）和单类别（per-class）指标下的 Top-1 准确率作为评估指标。

<table><tr><td rowspan="3">方法</td><td colspan="4">MSASL100</td><td colspan="4">MSASL200</td><td colspan="4">MSASL1000</td></tr><tr><td colspan="2">P-I</td><td colspan="2">P-C</td><td colspan="2">P-I</td><td colspan="2">P-C</td><td colspan="2">P-I</td><td colspan="2">P-C</td></tr><tr><td>T-1</td><td>T-5</td><td>T-1</td><td>T-5</td><td>T-1</td><td>T-5</td><td>T-1</td><td>T-5</td><td>T-1</td><td>T-5</td><td>T-1</td><td>T-5</td></tr><tr><td colspan="13">基于姿态</td></tr><tr><td>ST-GCN (Yan, Xiong, and Lin 2018)</td><td>59.84</td><td>82.03</td><td>60.79</td><td>82.96</td><td>52.91</td><td>76.67</td><td>54.20</td><td>77.62</td><td>36.03</td><td>59.92</td><td>32.32</td><td>57.15</td></tr><tr><td>SignBERT (Hu et al. 2021a)</td><td>76.09</td><td>92.87</td><td>76.65</td><td>93.06</td><td>70.64</td><td>89.55</td><td>70.92</td><td>90.00</td><td>49.54</td><td>74.11</td><td>46.39</td><td>72.65</td></tr><tr><td>Ours</td><td>80.98</td><td>95.11</td><td>81.24</td><td>95.44</td><td>76.60</td><td>91.54</td><td>76.75</td><td>91.95</td><td>58.82</td><td>81.18</td><td>54.87</td><td>80.05</td></tr><tr><td colspan="13">基于 RGB</td></tr><tr><td>I3D (Carreira and Zisserman 2017)</td><td>-</td><td>-</td><td>81.76</td><td>95.16</td><td>-</td><td>-</td><td>81.97</td><td>93.79</td><td>-</td><td>-</td><td>57.69</td><td>81.05</td></tr><tr><td>TCK (Li et al. 2020b)</td><td>83.04</td><td>93.46</td><td>83.91</td><td>93.52</td><td>80.31</td><td>91.82</td><td>81.14</td><td>92.24</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BSL (Albanie et al. 2020)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>64.71</td><td>85.59</td><td>61.55</td><td>84.43</td></tr><tr><td>HMA (Hu, Zhou, and Li 2021)</td><td>73.45</td><td>89.70</td><td>74.59</td><td>89.70</td><td>66.30</td><td>84.03</td><td>67.47</td><td>84.03</td><td>49.16</td><td>69.75</td><td>46.27</td><td>68.60</td></tr><tr><td>Ours (+R)</td><td>89.56</td><td>96.96</td><td>90.08</td><td>97.07</td><td>86.83</td><td>95.66</td><td>87.45</td><td>95.72</td><td>71.21</td><td>88.85</td><td>68.24</td><td>87.98</td></tr></table>

Table 2: 在 MSASL 数据集上与最先进方法的比较。我们提出的方法与另一种基于 RGB 的方法 I3D (Carreira and Zisserman 2017) 融合后表示为 Ours (+R)。

<table><tr><td rowspan="3">方法</td><td colspan="4">WLASL100</td><td colspan="4">WLASL300</td><td colspan="4">WLASL2000</td></tr><tr><td colspan="2">P-I</td><td colspan="2">P-C</td><td colspan="2">P-I</td><td colspan="2">P-C</td><td colspan="2">P-I</td><td colspan="2">P-C</td></tr><tr><td>T-1</td><td>T-5</td><td>T-1</td><td>T-5</td><td>T-1</td><td>T-5</td><td>T-1</td><td>T-5</td><td>T-1</td><td>T-5</td><td>T-1</td><td>T-5</td></tr><tr><td colspan="13">基于姿态</td></tr><tr><td>ST-GCN (Yan, Xiong, and Lin 2018)</td><td>50.78</td><td>79.07</td><td>51.62</td><td>79.47</td><td>44.46</td><td>73.05</td><td>45.29</td><td>73.16</td><td>34.40</td><td>66.57</td><td>32.53</td><td>65.45</td></tr><tr><td>Pose-TGCN (Li et al. 2020a)</td><td>55.43</td><td>78.68</td><td>-</td><td>-</td><td>38.32</td><td>67.51</td><td>-</td><td>-</td><td>23.65</td><td>51.75</td><td>-</td><td>-</td></tr><tr><td>PSLR (Tunga, Nuthalapati, and Wachs 2021)</td><td>60.15</td><td>83.98</td><td>-</td><td>-</td><td>42.18</td><td>71.71</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SignBERT (Hu et al. 2021a)</td><td>76.36</td><td>91.09</td><td>77.68</td><td>91.67</td><td>62.72</td><td>85.18</td><td>63.43</td><td>85.71</td><td>39.40</td><td>73.35</td><td>36.74</td><td>72.38</td></tr><tr><td>Ours</td><td>77.91</td><td>91.47</td><td>77.83</td><td>92.50</td><td>67.66</td><td>89.22</td><td>68.31</td><td>89.57</td><td>46.25</td><td>79.33</td><td>43.52</td><td>77.65</td></tr><tr><td colspan="13">基于RGB</td></tr><tr><td>I3D (Carreira and Zisserman 2017)</td><td>65.89</td><td>84.11</td><td>67.01</td><td>84.58</td><td>56.14</td><td>79.94</td><td>56.24</td><td>78.38</td><td>32.48</td><td>57.31</td><td>-</td><td>-</td></tr><tr><td>TCK (Li et al. 2020b)</td><td>77.52</td><td>91.08</td><td>77.55</td><td>91.42</td><td>68.56</td><td>89.52</td><td>68.75</td><td>89.41</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BSL (Albanie et al. 2020)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>46.82</td><td>79.36</td><td>44.72</td><td>78.47</td></tr><tr><td>HMA (Hu, Zhou, and Li 2021)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>37.91</td><td>71.26</td><td>35.90</td><td>70.00</td></tr><tr><td>Ours (+R)</td><td>81.01</td><td>94.19</td><td>81.63</td><td>94.67</td><td>75.60</td><td>92.81</td><td>76.12</td><td>93.07</td><td>54.59</td><td>88.08</td><td>52.12</td><td>87.28</td></tr></table>

Table 3: 在 WLASL 数据集上与最先进方法的比较。我们提出的方法与另一种基于 RGB 的方法 I3D (Carreira and Zisserman 2017) 融合后表示为 Ours (+R)。

<table><tr><td>方法</td><td>准确率</td></tr><tr><td colspan="2">基于姿态</td></tr><tr><td>ST-GCN (Yan, Xiong, and Lin 2018)</td><td>90.0</td></tr><tr><td>SignBERT (Hu et al. 2021a)</td><td>94.5</td></tr><tr><td>Ours</td><td>95.4</td></tr><tr><td colspan="2">基于RGB</td></tr><tr><td>STIP (Laptev 2005)</td><td>61.8</td></tr><tr><td>GMM-HMM (Tang et al. 2015)</td><td>56.3</td></tr><tr><td>3D-R50 (Qiu, Yao, and Mei 2017)</td><td>95.1</td></tr><tr><td>HMA (Hu, Zhou, and Li 2021)</td><td>95.9</td></tr><tr><td>GLE-Net (Hu et al. 2021b)</td><td>96.8</td></tr><tr><td>Ours (+R)</td><td>97.7</td></tr></table>

Table 4: 在 SLR500 数据集上与最先进方法的对比。3D-R50 (Qiu, Yao, and Mei 2017) 用于与我们的方法进行融合。

不同的分词器（Different Tokenizers）。在 Table 6 中，我们对比了不同分词器对下游 SLR 任务的影响。K-Means 方法是一种简单的无模型聚类算法。与可学习的分词器不同，K-Means 分别将手部和身体的 2D 姿态数据迭代转换为一系列聚类中心。然后，我们直接将聚类中心的索引作为伪标签。为了公平对比，聚类中心的数量与我们所使用码本（codebook）的词表大小相同。VQ 和 Ours 分别代表独立和耦合的矢量量化器。VQ 分别学习手部和身体的码本，而我们提出的分词器则对手部和身体码本进行联合优化。可以看出，我们提出的分词器表现出最佳性能，验证了我们的假设。由于身体和手部强相关，我们的耦合分词利用了这一线索以实现更好的码本学习。此外，我们在 Figure 2 中展示了我们提出的耦合分词器的定性结果。在不受约束的设置下（即个体姿态差异和不准确的姿态检测），我们的分词器能够成功对相似的姿态三元组单元进行聚类。

掩码策略（Masking Strategy）。如 Table 7 所示，我们对比了所提出的 MUM 上的三种不同设置，即仅掩码手部、仅掩码身体以及同时掩码两个部位（手部和身体）。第一行表示未进行预训练的基线。由于手语表达中手部的主导地位，仅掩码手部的设置比仅掩码身体的设置表现更好。我们的 MUM 采用了第三种掩码设置，并取得了最佳性能。与基线相比，我们提出的 MUM 取得了性能提升，即每个实例的 Top-1 准确率分别提高了 7.54%、5.81% 和 6.86%。预训练数据规模（Pre-Training Data Scale）。在 Table 8 中，我们研究了预训练数据规模的影响。第一行表示我们提出的框架在无预训练情况下的方法。可以明显观察到，随着预训练数据比例的增加，性能逐渐提升。该结果表明，我们提出的方法适用于大规模数据的预训练。

<table><tr><td rowspan="2">方法</td><td>总计</td><td>易混淆</td><td>正常</td></tr><tr><td>T-1</td><td>T-1</td><td>T-1</td></tr><tr><td colspan="4">基于姿态 (Pose-based)</td></tr><tr><td>ST-GCN (Yan, Xiong, and Lin 2018)</td><td>59.9</td><td>42.2</td><td>83.4</td></tr><tr><td>SignBERT (Hu et al. 2021a)</td><td>67.0</td><td>46.4</td><td>94.5</td></tr><tr><td>Ours</td><td>68.5</td><td>49.0</td><td>94.6</td></tr><tr><td colspan="4">基于 RGB (RGB-based)</td></tr><tr><td>3D-R50 (Qiu, Yao, and Mei 2017)</td><td>62.1</td><td>43.1</td><td>87.4</td></tr><tr><td>DNF (Cui, Liu, and Zhang 2019)</td><td>55.8</td><td>51.9</td><td>86.3</td></tr><tr><td>I3D (Carreira and Zisserman 2017)</td><td>64.4</td><td>47.3</td><td>87.1</td></tr><tr><td>TSM (Lin, Gan, and Han 2019)</td><td>64.5</td><td>42.9</td><td>93.3</td></tr><tr><td>Slowfast (Feichtenhofer et al. 2019)</td><td>66.3</td><td>47.0</td><td>92.0</td></tr><tr><td>GLE-Net (Hu et al. 2021b)</td><td>69.0</td><td>50.6</td><td>93.6</td></tr><tr><td>HMA (Hu, Zhou, and Li 2021)</td><td>64.7</td><td>42.3</td><td>94.6</td></tr><tr><td>Ours (+R)</td><td>79.2</td><td>65.5</td><td>97.5</td></tr></table>

Table 5: 在 NMFs-CSL 数据集上与最先进（SOTA）方法的对比。我们提出的方法与另一种基于 RGB 的方法 3D-R50 (Qiu, Yao, and Mei 2017) 融合后表示为 Ours (+R)。  
![](images/2f82761b0a7768e08437546b2b3849a040f700cc82ca294b4c0b1c1caa8a8526.jpg)  
Figure 2: 我们提出的耦合分词（coupling tokenization）的定性结果。每相邻两列为一组。在每组的同一行中，姿态与 RGB 图像相对应。每组属于相同的三元组单元标签。

## 5 结论

在本文中，我们提出了一个名为 BEST 的自监督预训练框架，该框架利用了 BERT 的成功经验，并针对手语领域进行了专门设计。针对手语表达过程中的主要特征，我们将手部和身体动作组织为姿态三元组单元（pose triplet unit）。在预训练期间，我们提出了掩码单元建模（masked unit modeling, MUM）前置任务，以挖掘内部和外部三元组单元之间的分层上下文线索。为了使原有的 BERT 目标适用，我们尝试通过对三元组单元进行耦合分词来弥合伪标签的语义鸿沟。大量的实验验证了我们方法的有效性，在所有四个基准数据集上均取得了新的最先进性能。

<table><tr><td rowspan="2">量化器</td><td colspan="2">MSASL100</td><td colspan="2">MSASL200</td><td colspan="2">MSASL1000</td></tr><tr><td>P-I</td><td>P-C</td><td>P-I</td><td>P-C</td><td>P-I</td><td>P-C</td></tr><tr><td>K-Means</td><td>74.63</td><td>74.03</td><td>72.62</td><td>74.20</td><td>54.29</td><td>50.74</td></tr><tr><td>VQ</td><td>77.54</td><td>77.44</td><td>74.47</td><td>75.47</td><td>55.87</td><td>52.16</td></tr><tr><td>Ours</td><td>80.98</td><td>81.24</td><td>76.60</td><td>76.75</td><td>58.82</td><td>54.87</td></tr></table>

Table 6: 在 MSASL 数据集上与不同姿态分词器（pose tokenizer）的比较。K-Means 表示一种无模型聚类算法。VQ 和 Ours 分别代表分离式和耦合式向量分词器。

<table><tr><td colspan="2">掩码</td><td colspan="2">MSASL100</td><td colspan="2">MSASL200</td><td colspan="2">MSASL1000</td></tr><tr><td>手部</td><td>身体</td><td>P-I</td><td>P-C</td><td>P-I</td><td>P-C</td><td>P-I</td><td>P-C</td></tr><tr><td></td><td></td><td>73.44</td><td>73.75</td><td>70.79</td><td>71.46</td><td>51.96</td><td>48.50</td></tr><tr><td>√</td><td></td><td>76.75</td><td>77.22</td><td>71.37</td><td>72.07</td><td>54.60</td><td>51.63</td></tr><tr><td></td><td>√</td><td>68.82</td><td>68.82</td><td>67.11</td><td>67.84</td><td>47.72</td><td>43.87</td></tr><tr><td>√</td><td>√</td><td>80.98</td><td>81.24</td><td>76.60</td><td>76.75</td><td>58.82</td><td>54.87</td></tr></table>

Table 7: 在 MSASL 数据集上掩码策略的影响。第一行表示未经预训练的基线方法。“手部（Hand）”和“身体（Body）”分别表示仅对手部和身体部位进行掩码。

<table><tr><td rowspan="2">比例</td><td colspan="2">MSASL100</td><td colspan="2">MSASL200</td><td colspan="2">MSASL1000</td></tr><tr><td>P-I</td><td>P-C</td><td>P-I</td><td>P-C</td><td>P-I</td><td>P-C</td></tr><tr><td>0%</td><td>73.44</td><td>73.75</td><td>70.79</td><td>71.46</td><td>51.96</td><td>48.50</td></tr><tr><td>25%</td><td>74.50</td><td>75.20</td><td>71.89</td><td>72.69</td><td>53.67</td><td>50.81</td></tr><tr><td>50%</td><td>77.54</td><td>78.06</td><td>73.43</td><td>75.03</td><td>54.69</td><td>51.47</td></tr><tr><td>75%</td><td>78.47</td><td>78.77</td><td>74.54</td><td>75.32</td><td>56.06</td><td>53.16</td></tr><tr><td>100%</td><td>80.98</td><td>81.24</td><td>76.60</td><td>76.75</td><td>58.82</td><td>54.87</td></tr></table>

Table 8: 在 MSASL 数据集上预训练期间数据规模的影响。“比例（Percent）”表示预训练数据的占比。

## 致谢

本研究得到了中国国家自然科学基金（项目资助号：U20A20183 和 62021001）的资助。同时得到了中国科学技术大学信息科学技术学院 MCC 实验室搭建的 GPU 集群的支持。

Albanie, S.; Varol, G.; Momeni, L.; Afouras, T.; Chung, J. S.; Fox, N.; and Zisserman, A. 2020. BSL-1K: Scaling up co-articulated sign language recognition using mouthing cues. In ECCV, 35–53.

Bao, H.; Dong, L.; and Wei, F. 2021. BEiT: BERT pretraining of image transformers. In ICLR, 1–18.

Bengio, Y.; Leonard, N.; and Courville, A. 2013. Estimat-´ ing or propagating gradients through stochastic neurons for conditional computation. arXiv, 1–12.

Cai, Y.; Ge, L.; Liu, J.; Cai, J.; Cham, T.-J.; Yuan, J.; and Thalmann, N. M. 2019. Exploiting spatial-temporal relationships for 3D pose estimation via graph convolutional networks. In ICCV, 2272–2281.

Camgoz, N. C.; Hadfield, S.; Koller, O.; Ney, H.; and Bowden, R. 2018. Neural sign language translation. In CVPR, 7784–7793.

Carreira, J.; and Zisserman, A. 2017. Quo vadis, action recognition? a new model and the kinetics dataset. In CVPR, 6299–6308.

Conneau, A.; and Lample, G. 2019. Cross-lingual language model pretraining. In NeurIPS, 1–11.

Cui, R.; Liu, H.; and Zhang, C. 2019. A deep neural framework for continuous sign language recognition by iterative training. IEEE TMM, 21(7): 1880–1891.

Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2018. BERT: Pre-training of deep bidirectional transformers for language understanding. In ACL, 4171–4186.

Du, Y.; Wang, W.; and Wang, L. 2015. Hierarchical recurrent neural network for skeleton based action recognition. In CVPR, 1110–1118.

Farhadi, A.; Forsyth, D.; and White, R. 2007. Transfer learning in sign language. In CVPR, 1–8.

Feichtenhofer, C.; Fan, H.; Malik, J.; and He, K. 2019. Slowfast networks for video recognition. In ICCV, 6202–6211.

Fillbrandt, H.; Akyol, S.; and Kraiss, K.-F. 2003. Extraction of 3D hand shape and posture from image sequences for sign language recognition. In ISO4, 181–186.

He, K.; Chen, X.; Xie, S.; Li, Y.; Dollar, P.; and Girshick, R. 2022. Masked autoencoders are scalable vision learners. In CVPR, 16000–16009.

Hu, H.; Zhao, W.; Zhou, W.; Wang, Y.; and Li, H. 2021a. SignBERT: Pre-Training of Hand-Model-Aware Representation for Sign Language Recognition. In ICCV, 11087– 11096.

Hu, H.; Zhou, W.; and Li, H. 2021. Hand-model-aware sign language recognition. In AAAI, 1558–1566.

Hu, H.; Zhou, W.; Pu, J.; and Li, H. 2021b. Global-local enhancement network for NMF-aware sign language recognition. ACM TOMM, 17(3): 1–19.

Huang, J.; Zhou, W.; Li, H.; and Li, W. 2018. Attentionbased 3D-CNNs for large-vocabulary sign language recognition. IEEE TCSVT, 29(9): 2822–2832.

Jiang, T.; Camgoz, N. C.; and Bowden, R. 2021. Skeletor: Skeletal transformers for robust body-pose estimation. In CVPR, 3394–3402.

Jin, T.; and Zhao, Z. 2021. Contrastive Disentangled Meta-Learning for Signer-Independent Sign Language Translation. In ACM MM, 5065–5073.

Joze, H. R. V.; and Koller, O. 2018. MS-ASL: A large-scale data set and benchmark for understanding american sign language. In BMVC, 1–16.

Kingma, D. P.; and Ba, J. 2014. Adam: A method for stochastic optimization. arXiv, 1–15.

Koller, O.; Zargaran, S.; Ney, H.; and Bowden, R. 2018. Deep sign: Enabling robust statistical continuous sign language recognition via hybrid CNN-HMMs. IJCV, 126(12): 1311–1325.

Laptev, I. 2005. On space-time interest points. IJCV, 64(2): 107–123.

Lewis, M.; Liu, Y.; Goyal, N.; Ghazvininejad, M.; Mohamed, A.; Levy, O.; Stoyanov, V.; and Zettlemoyer, L. 2019. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In ACL, 7871–7880.

Li, C.; Zhong, Q.; Xie, D.; and Pu, S. 2018. Co-occurrence feature learning from skeleton data for action recognition and detection with hierarchical aggregation. In IJCAI, 786– 792.

Li, D.; Rodriguez, C.; Yu, X.; and Li, H. 2020a. Word-level deep sign language recognition from video: A new largescale dataset and methods comparison. In WACV, 1459– 1469.

Li, D.; Yu, X.; Xu, C.; Petersson, L.; and Li, H. 2020b. Transferring cross-domain knowledge for video sign language recognition. In CVPR, 6205–6214.

Lin, J.; Gan, C.; and Han, S. 2019. TSM: Temporal shift module for efficient video understanding. In ICCV, 7083– 7093.

Min, Y.; Zhang, Y.; Chai, X.; and Chen, X. 2020. An efficient pointlstm for point clouds based gesture recognition. In CVPR, 5761–5770.

Momeni, L.; Bull, H.; Prajwal, K.; Albanie, S.; Varol, G.; and Zisserman, A. 2022. Automatic dense annotation of large-vocabulary sign language videos. In ECCV, 671–690.

Ng, E.; Ginosar, S.; Darrell, T.; and Joo, H. 2021. Body2hands: Learning to infer 3D hands from conversational gesture body dynamics. In CVPR, 11865–11874.

Niu, Z.; and Mak, B. 2020. Stochastic fine-grained labeling of multi-state sign glosses for continuous sign language recognition. In ECCV, 172–186.

Qiu, Z.; Yao, T.; and Mei, T. 2017. Learning spatio-temporal representation with pseudo-3D residual networks. In ICCV, 5533–5541.

Ramesh, A.; Pavlov, M.; Goh, G.; Gray, S.; Voss, C.; Radford, A.; Chen, M.; and Sutskever, I. 2021. Zero-shot textto-image generation. In ICML, 8821–8831.

Selvaraj, P.; Nc, G.; Kumar, P.; and Khapra, M. 2022. Open-Hands: Making Sign Language Recognition Accessible with Pose-based Pretrained Models across Languages. In ACL, 2114–2133.

Sincan, O. M.; and Keles, H. Y. 2020. Autsl: A large scale multi-modal turkish sign language dataset and baseline methods. IEEE Access, 8: 181340–181355.

Starner, T. E. 1995. Visual Recognition of American Sign Language Using Hidden Markov Models. Technical report, MIT.

Su, W.; Zhu, X.; Cao, Y.; Li, B.; Lu, L.; Wei, F.; and Dai, J. 2019. VL-BERT: Pre-training of generic visual-linguistic representations. In ICLR, 1–16.

Sun, C.; Myers, A.; Vondrick, C.; Murphy, K.; and Schmid, C. 2019. VideoBERT: A joint model for video and language representation learning. In ICCV, 7464–7473.

Tang, A.; Lu, K.; Wang, Y.; Huang, J.; and Li, H. 2015. A real-time hand posture recognition system using deep neural networks. ACM TIST, 6(2): 1–23.

Tunga, A.; Nuthalapati, S. V.; and Wachs, J. 2021. Pose-Based Sign Language Recognition Using GCN and BERT. In WACV, 31–40.

Van Den Oord, A.; Vinyals, O.; et al. 2017. Neural discrete representation learning. In NeurIPS, 1–11.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. In NeurIPS, 6000–6010.

Yan, S.; Xiong, Y.; and Lin, D. 2018. Spatial temporal graph convolutional networks for skeleton-based action recognition. In AAAI, 1–10.

Zhu, L.; and Yang, Y. 2020. ActBERT: 学习全局-局部视频-文本表征 (ActBERT: Learning global-local video-text representations). In CVPR, 8746–8755.