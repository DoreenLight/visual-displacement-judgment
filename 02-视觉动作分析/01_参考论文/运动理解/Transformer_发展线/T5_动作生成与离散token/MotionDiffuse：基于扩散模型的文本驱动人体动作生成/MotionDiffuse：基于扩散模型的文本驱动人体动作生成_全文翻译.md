# MotionDiffuse：基于扩散模型的文本驱动人体动作生成

Mingyuan Zhang\* · Zhongang Cai\* · Liang Pan · Fangzhou Hong · Xinying Guo · Lei Yang · Ziwei Liu

接收日期：date / 接受日期：date

摘要 人体动作建模对于许多现代图形学应用至关重要，而这些应用通常需要专业技能。为了消除非专业人员的技能门槛，近期的动作生成方法可以直接以自然语言为条件生成人体动作。然而，针对多样化的文本输入实现多样且细粒度的动作生成仍然具有挑战性。为了解决这一问题，我们提出了 MotionDiffuse，这是首个基于扩散模型的文本驱动动作生成框架，相较于现有方法展现出若干理想特性：1）概率映射。MotionDiffuse 并非采用确定性的语言-动作映射，而是通过一系列注入了变化因子的去噪步骤来生成动作。2）逼真合成。MotionDiffuse 擅长建模复杂的数据分布并生成生动的动作序列。3）多层级操控。MotionDiffuse 能够响应针对身体部位的细粒度指令，并能够利用随时间变化的文本提示合成任意长度的动作。我们的实验表明，MotionDiffuse 在文本驱动动作生成和动作类别条件动作生成任务上均以显著优势超越现有的 SoTA 方法。定性分析进一步证明了 MotionDiffuse 在综合动作生成方面的可控性。主页：https://mingyuan-zhang.github. io/projects/MotionDiffuse.html

关键词 动作合成 · 条件动作生成 · 扩散模型 · 文本驱动生成

## 1 引言

人体动作建模是驱动虚拟角色以模仿生动丰富的人类动作的关键组成部分，它已成为电影制作、游戏开发和虚拟主播（virtual YouTuber）动画等众多应用中的核心课题。为了模仿人类动作，虚拟角色应当能够自然地四处移动、对环境刺激做出反应，并在此过程中表达复杂的情感。尽管经历数十载激动人心的技术突破，生成生动真实的人体动作仍然需要精密的设备（例如昂贵的动作捕捉系统）和领域专家。为了消除非专业用户的技能门槛并有望推广至大众受众，构建一个能够生成多样化且易于操控的动作序列的通用人体动作生成模型至关重要。

![](images/02d9db4a2fb689b356e16d9df43d0d967d362dcc2f3892e87bcc5ff8b03df932.jpg)  
Fig. 1 MotionDifuse 是一种基于扩散模型的文本驱动动作生成方法，其特点包括：1) 概率映射；2) 逼真合成，能够生成如 a)-d) 所示的高度多样化且高保真度的动作；3) 多层次操控，支持生成复杂的动作，例如涉及多个身体部位的 c) 以及给定随时间变化的提示词的 d)。

以往的人体动作生成方法利用了多种条件信号，包括预定义的动作类别（Guo et al., 2020; Petrovich et al., 2021; Cervantes et al., 2022）、音乐片段（Huang et al., 2020; Li et al., 2020, 2021; Zhuang et al., 2020; Siyao et al., 2022）以及自然语言（Lin et al., 2018; Ahuja and Morency, 2019; Ghosh et al., 2021; Petrovich et al., 2022）。其中，自然语言可以说是用于动作序列合成中最具用户友好性且最便捷的输入形式，因此我们在本研究中专注于文本驱动的动作生成。最近，TEMOS（Petrovich et al., 2022）利用 KIT Motion-Language MoCap 数据集（Plappert et al., 2016）展示了细粒度的轨迹合成。然而，它不支持对生成的动作进行风格化，因此无法实现高度的多样性。MotionCLIP（Tevet et al., 2022）可以生成风格化的动作，但它仍然局限于简短的文本输入，无法处理复杂的动作描述。此外，这些方法（Petrovich et al., 2022; Tevet et al., 2022）通常仅接受单一文本提示，这极大地限制了用户的创造力。

为了应对上述挑战，我们提出了 MotionDifuse，一个通用且可控的动作生成框架，能够根据复杂的文本生成多样化的动作。受到文本条件图像生成近期进展（Dhariwal and Nichol, 2021; Nichol and Dhariwal, 2021; Nichol et al., 2021; Ramesh et al., 2022）的启发，我们提出将去噪扩散概率模型（Denoising Diffusion Probabilistic Models，DDPM）（Ho et al., 2020）引入到动作生成中。与仅能进行固定尺寸生成的经典 DDPM 不同，我们提出了一种跨模态线性 Transformer（Cross-Modality Linear Transformer），以根据动作时长实现任意长度的动作合成。与学习文本空间和动作空间之间的直接映射（Tevet et al., 2022）不同，我们提出使用输入文本软性地引导生成流程，这可以显著增加生成结果的多样性。为了在去噪过程中保持不确定性，我们在每个去噪步骤中通过多个 Transformer 解码器层来处理以输入文本为条件的噪声项。通过这种方式，文本条件不会以确定性的方式主导动作生成，从而有助于从驱动文本中生成多样化的动作序列。

此外，MotionDifuse 能够通过细粒度文本实现身体部位独立控制。具体而言，为了契合人体结构，MotionDifuse 将全身动作划分为若干近似独立的部位（例如上半身和下半身）。基于细粒度的身体部位定义，我们提出了“噪声插值”（noise interpolation）方法，在单独控制不同身体部位的同时充分考虑它们之间的相关性。此外，为了合成任意长度的动作序列，我们提出了一种新的采样方法，用于同时对多个重叠序列进行去噪。具体来说，MotionDifuse 首先独立获取每个序列的结果，然后结合修正项将它们混合。与通常需要大量长动作序列进行训练的自回归推理方案不同，MotionDifuse 能够在不引入额外训练成本的情况下建模连续动作之间的相关性。

我们在流行基准数据集上进行了广泛的定性实验，并对全面的动作生成进行了定量评估。首先，在 HumanML3D 和 KIT-ML 上，我们展示了文本驱动动作生成相比当前现有技术的显著改进。其次，为了说明 MotionDifuse 的优越性，我们直接将其应用于动作类别条件下的动作生成任务中，它在 HumanAct12 数据集和 UESTC 数据集上的表现均优于所有现有工作。此外，我们还通过以混合控制信号为条件进行建模，展示了 MotionDifuse 的更多可能性，从而实现了身体部位级别的操控和长动作生成。

总而言之，与现有技术相比，我们提出的 MotionDifuse 具有若干理想的特性：

– 概率映射（Probabilistic Mapping）。得益于我们融合了 DDPM 的全新动作生成形式，MotionDifuse 能够以文本描述为条件，以概率方式生成动作，这自然带来了高度的多样性。

逼真合成（Realistic Synthesis）。精心的架构设计使 MotionDifuse 能够合成高保真度的动作序列，并在两个条件动作生成任务上达到了领先水平（state-of-the-art）。– 多级操控（Multi-Level Manipulation）。通过扩展设计，MotionDifuse 能够处理调动全身的细粒度文本描述（例如“一个人边走边喝水”）以及随时间变化的信号（例如“一个人在走，然后跑起来”）。

## 2 相关工作

## 2.1 动作生成模型

动作生成（Motion Generation）已被研究了数十年。一些早期工作侧重于无条件动作生成（Rose et al., 1998; Ikemoto et al., 2009; Mukai and Kuriyama, 2005）。而另一些工作则尝试在给定初始姿态或起始动作序列的情况下预测未来的动作（Futrelle and Speckert, 1978; Gavrila, 1999; O’rourke and Badler, 1980）。PCA（Ormoneit et al., 2005）、动作图（Motion Graph）（Min and Chai, 2012）等统计模型被应用于这些目的。

随着深度学习（Deep Learning，DL）技术的飞速发展，越来越多的生成式架构涌现并蓬勃发展。先前的研究大致可以分为四类：1）变分自编码器（Variational Auto Encoder，VAE）；2）生成对抗网络（Generative Adversarial Networks，GAN）；3）规范化流网络（Normalization Flow Network）；4）隐式神经表示（Implicit Neural Representations）。

VAE（Kingma and Welling, 2013）是动作合成中最常用的生成模型之一。Yan et al. (2018) 和 Aliakbarian et al. (2020) 将动作生成任务视为在给定当前小段动作序列的情况下预测未来的一小段动作序列。他们使用 VAE 对当前序列和未来序列对进行编码，进而重构未来序列。ACTOR（Petrovich et al., 2021）提出了一种基于 Transformer 的编码器和解码器架构。Transformer 编码器层和 Transformer 解码器层（Vaswani et al., 2017）是构建动作编码器和动作解码器的基础模块。该架构在后续研究中也被广泛采用（Tevet et al., 2022; Hong et al., 2022; Petrovich et al., 2022）。

GAN（Goodfellow et al., 2014）引入了一个辅助模块——判别器网络，用于评判生成样本的质量和真实性。一些工作致力于为动作生成提出合适的判别器网络以提升合成质量（Barsoum et al., 2018; Harvey et al., 2020; Wang et al., 2020）。HP-GAN（Barsoum et al., 2018）尝试在没有特定真实值（ground truth）的情况下监督动作预测结果。为此，引入了一个数据驱动的判别器来学习动作先验，并用于评判预测质量。Harvey et al. (2020) 旨在解决动作补帧（Motion In-between）任务中预测动作模糊的问题，提出了分别用于短期评估和长期评估的两个判别器。Wang et al. (2020) 构建了一个循环管线（cyclic pipeline）。在判别器的辅助下，该管线既能生成特定类别的动作序列，也能生成混合类别的动作序列。

规范化流网络（Normalization Flow Network）（Dinh et al., 2014）历史悠久，并在图像合成领域得到了广泛研究（Dinh et al., 2016; Kingma and Dhariwal, 2018）。该类架构构建了一个可逆神经网络，将输入数据映射到多维高斯分布中。因此，我们可以从该分布中生成初始随机向量，并将其输入反向网络以生成动作样本。受 GLOW（Kingma and Dhariwal, 2018）成功的启发，MoGlow（Henter et al., 2020）提出了一种自回归规范化流网络来对动作序列进行建模。来自 LSTM 模型（Hochreiter and Schmidhuber, 1997）的历史特征作为流网络的条件输入，用于预测下一个姿态。

最近，随着 NeRF（Mildenhall et al., 2020; Jain et al., 2021）在渲染逼真图像方面取得的巨大成功，另一种生成模型也受到了广泛关注。隐式神经表示（Implicit Neural Representations，INR）是一系列通过优化其参数来拟合单个样本而非整个分布的神经网络。其主要优势之一是该技术在空间或时间维度上具有极佳的泛化能力。例如，Cervantes et al. (2022) 提出了一种隐式方案，可同时对动作类别和时间戳进行建模。与原始 NeRF 类似，时间戳通过正弦值进行表示。经过监督训练后，所提出的方法能够为每个动作类别生成可变长度的动作序列。

本文提出了一种基于去噪扩散概率模型（Denoising Diffusion Probabilistic Model，DDPM）（Ho et al., 2020）的新型动作生成管线。DDPM 的主要优势之一在于可以保留原始动作序列的形态。这意味着我们可以在去噪过程中轻松施加更多约束。在后续章节中，我们将探索 DDPM 在不同条件类型下的更多潜力。此外，得益于这种特性，DDPM 能够生成更多样化的样本。

## 2.2 条件动作生成

各种生成模型的日益成熟激发了研究人员探索条件动作生成的热情。例如，一些工作（Guo et al., 2020; Petrovich et al., 2021; Cervantes et al., 2022）致力于合成若干特定类别的动作序列。Action2Motion（Guo et al., 2020）构建了一个用于动作生成的循环条件 VAE。在给定历史记忆的情况下，该模型在动作类别的约束下预测下一姿态。ACTOR（Petrovich et al., 2021）同样使用 VAE 进行随机采样。与 Action2Motion 不同，ACTOR 将整个动作序列嵌入到潜在空间中。这种设计避免了循环方案中的累积误差。此外，ACTOR 提出了一种基于 Transformer 的动作编码器和解码器架构。该结构的性能显著优于循环方法。Cervantes et al. (2022) 尝试使用隐式函数对动作序列进行建模，能够生成不同长度的动作序列。

另一个重要的条件动作生成任务是音乐驱动舞蹈（music to dance）。该任务要求生成的动作具有节拍连贯性、属于特定舞种，或能够表达与音乐相似的内容。许多工作尝试将音乐特征和动作特征嵌入到联合空间中（Lee et al., 2019; Sun et al., 2020; Li et al., 2020, 2021）。与直接的特征嵌入不同，Bailando（Siyao et al., 2022）提出了一种两阶段舞蹈生成器。它首先学习一个包含有意义舞蹈片段的量化码本（codebook），然后尝试利用码本中的一系列元素生成完整序列。

与音乐生成舞蹈（music-to-dance）类似，文本驱动的动作生成可以被视为学习文本特征空间与动作特征空间的联合嵌入。这其中存在两个主要差异。第一个差异在于语言指令与人体的关联性更强。因此，我们期望能够精确控制每个身体部位。第二个差异在于文本驱动的动作生成涵盖了极其广泛的动作类型。有些描述是对特定身体部位的直接指令，例如“摸头”（touch head）；有些则描述了像“拉小提琴”（playing the violin）这样的任意概念。如此巨大的动作复杂性给架构设计带来了诸多困难。最近，许多工作提出了文本驱动动作生成的流程。其中大部分是确定性生成（Ahuja and Morency, 2019; Ghosh et al., 2021; Tevet et al., 2022），这意味着它们只能从给定的文本中生成单一结果。TEMOS (Petrovich et al., 2022) 将 VAE 架构引入到该任务中。它可以在给定一个文本描述的情况下生成不同的动作序列。然而，这些方法试图获取动作与自然语言的联合嵌入空间。这种设计显著压缩了来自文本的信息。因此，这些工作很难根据详细的描述生成正确的动作序列。Guo et al. (2022) 提出了一个自回归流程。它首先将语言描述编码为特征，然后以文本特征为条件自回归地生成动作帧。然而，由于自回归机制，该方法难以捕捉全局关系。此外，生成质量也较差。相反，我们提出的 MotionDiffuse（原文作 MotionDifuse）将文本特征柔和地融合到生成过程中，并能同时生成整个序列。实验结果证明了我们设计的优越性。

## 2.3 动作数据集

人体动作建模一直是计算机视觉和计算机图形学中长期存在的研究问题。随着深度学习的出现，数据对于训练感知、理解和生成人体动作的神经网络变得越来越重要。

包含人体受试者视频的数据集的一种常见形式是记录有各种标注，例如 2D 关键点（Jhuang et al., 2013; Andriluka et al., 2018）、3D 关键点（Ionescu et al., 2013; Joo et al., 2015; Mehta et al., 2017; Trumble et al., 2017; Li et al., 2021）和统计模型参数（Yu et al., 2020; Patel et al., 2021; Cao et al., 2020; Cai et al., 2021, 2022）。动作标签也是人体动作理解数据集的一个常用属性，涵盖以人为中心的动作（Kuehne et al., 2011; Soomro et al., 2012; Karpathy et al., 2014; Gu et al., 2018; Shao et al., 2020; Chung et al., 2021）、交互（Carreira et al., 2019; Monfort et al., 2019; Zhao et al., 2019）、细粒度动作理解（Gu et al., 2018; Shao et al., 2020; Chung et al., 2021）以及 3D 数据（Liu et al., 2019）。对于动作条件动作生成任务，HumanAct12（Guo et al., 2020）、UESTC（Ji et al., 2018）和 NTU RGB+D（Liu et al., 2019）是三个常用的基准。然而，上述数据集并未为动作序列提供成对的复杂语义标签。

KIT（Plappert et al., 2016）包含带有详细描述标注的动作捕捉数据。Zhang et al. (2020) 招募男女演员来记录表达情感的身体动作。最近，BABEL（Punnakkal et al., 2021）和 HumanML3D（Guo et al., 2022）使用英语语言标签对大规模动作捕捉数据集 AMASS（Mahmood et al., 2019）进行了重新标注。

在本文中，我们使用 HumanML3D 数据集和 KIT 数据集来评估所提出的文本驱动动作生成任务方法。HumanAct12 和 UESTC 用于证明所提出流程的广泛适用性。此外，我们还将 BABEL 数据集用于其他应用。

## 3 方法

我们提出了一个基于扩散模型的框架 MotionDiffuse，用于高保真且可控的文本驱动动作生成。我们首先在第 3.1 节中给出原始文本驱动动作生成的问题定义和设置。随后，我们在第 3.2 节中对所提出的 MotionDiffuse 进行整体阐述，接着在第 3.3 节介绍扩散模型，并在第 3.4 节介绍基于 Transformer 的架构。最后，在第 3.5 节中针对细粒度生成场景阐述推理策略。

## 3.1 预备知识

动作序列 $\Theta$ 是一个数组 $( \theta _ { i } ) , \ i \in$ $\{ 1 , 2 , \ldots , F \}$，其中 $\theta _ { i } \in \mathbb { R } ^ { D }$ 表示第 $i$ 帧中的姿态状态，$F$ 是帧数。在不同的数据集中，每个姿态状态 $\theta _ { i }$ 的表示方式各不相同。它通常包含关节旋转、关节位置、关节速度以及脚部接触条件。我们提出的 MotionDiffuse 对各种动作表示形式均具有鲁棒性。因此，我们在本节中不具体指定 $\theta _ { i }$ 的组成部分，而将详细内容保留在第 4 节中。

对于标准的文本驱动动作生成（Text-driven Motion Generation），训练数据集由 $( \theta _ { i } , \mathrm { t e x t } _ { i } )$ 对组成，其中 text 是动作序列 $\theta _ { i } .$ 的语言描述。在推理过程中，给定一组描述 $\{ \mathrm { t e x t } _ { i } \}$，我们需要生成以给定描述为条件的动作序列。该任务也可以被视为文本到动作生成（text-to-motion translation, T2M）任务。我们在下文中将使用这一缩写。

一个相关的任务是动作类别条件的动作生成（Action-conditioned Motion Generation）。给定一个预定义的动作类别集合，模型需要拟合数据分布并合成每个类别的动作序列。该任务中的标注数据可以表示为 $( y _ { i } , \theta _ { i } )$，其中 $y _ { i }$ 是第 $i$ 个数据的类别索引，$\theta _ { i }$ 是第 $i$ 个数据的动作序列。在本文中，我们将 $y _ { i }$ 替换为其语义描述 $\mathrm { t e x t } _ { i }$。这样我们就可以使用与 T2M 任务相同的流程。

## 3.2 流程概述

遵循图像合成领域扩散模型（difusion model）的相关文献 (Ho et al., 2020)，我们首先使用去噪扩散概率模型（DDPM）构建了一个以文本为条件的动作生成流程。该模型是我们提出的 MotionDifuse 的基础。对于去噪过程，我们提出了一个跨模态线性 Transformer（Cross-Modality Linear Transformer），以处理以给定文本提示为条件的输入序列。除了直接应用于文本驱动的动作生成之外，我们更进一步探索了在去噪过程中以动作表示为条件的方法。具体而言，我们实验了两种类型的额外信号：部位感知的文本控制（part-aware text controlling）和随时间变化的控制（time-varied controlling）。前者将不同的文本条件分配给不同的身体部位，使我们能够精确控制身体的各个部位并生成更复杂的动作序列。后者将整个序列划分为若干部分，并为每个区间分配独立的文本条件。因此，我们可以合成包含多种动作的任意长度的动作序列。这两种条件显著扩展了 MotionDifuse 的能力。整体流程如 Figure 2 所示。我们将在以下小节中介绍该架构的各个部分。

## 3.3 用于动作生成的扩散模型

生成对抗网络（Generative Adversarial Networks，GANs）引入判别器，以对抗方式提升生成质量。GANs 的训练通常极具挑战性，尤其是在条件动作生成任务中。隐式函数（Implicit Functions）使用多层感知机（Multi-Layer Perceptron，MLP）来拟合动作序列。这种简洁的架构在少量数据上易于训练，但在面对复杂条件时泛化能力往往较差。自编码器（Auto-Encoder，AE）和变分自编码器（Variational Auto-Encoder，VAE）是文本驱动动作生成中最广泛使用的方法（Ghosh et al., 2021; Petrovich et al., 2022）。以往的工作学习动作序列与语言的联合嵌入，在确定性的语言-动作映射中显式应用文本条件。然而，高层级文本特征通常缺乏足够的细粒度细节来指导生成具有微妙差异的动作。因此，直接将文本嵌入与动作嵌入关联会导致生成动作的多样性受限。

![](images/84c7a9ad6c9b7502ddd7c0205998754542a83b81acb1e4887b3f9bcd688cb2a3.jpg)  
Fig. 2 所提出的 MotionDiffuse 整体流程管线。箭头的颜色表示不同阶段：蓝色代表训练，红色代表推理，黑色代表训练与推理兼有。

为了解决该问题，我们基于扩散模型构建了文本驱动的动作生成管线。扩散模型（Diffusion Models）（Ho et al., 2020; Dhariwal and Nichol, 2021; Nichol and Dhariwal, 2021; Nichol et al., 2021）是一类新型生成模型。通过学习一个概率模型，逐步对高斯噪声进行去噪以生成目标输出，例如二维图像或三维点云。形式上，扩散模型公式化表示为 $\begin{array} { r } { p _ { \theta } ( \mathbf { x } _ { 0 } ) : = \int p _ { \theta } ( \mathbf { x } _ { 0 : T } ) d \mathbf { x } _ { 1 : T } } \end{array}$ ，其中 $\mathbf { x } _ { 0 } \sim q ( \mathbf { x } _ { 0 } )$ 为真实数据，$\mathbf { x } _ { 1 } , \cdots , \mathbf { x } _ { T }$ 为潜数据。它们通常包含扩散过程和逆过程。为了逼近后验分布 $q \big ( \mathbf { x } _ { 1 : T } | \mathbf { x } _ { 0 } \big )$ ，扩散过程遵循马尔可夫链，根据由 $\beta _ { t }$ 给定的方差调度（variance schedules）逐步向数据添加高斯噪声，直至其分布接近潜分布 $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ ：

$$
\begin{array}{l} q (\mathbf {x} _ {1: T} | \mathbf {x} _ {0}) := \prod_ {t = 1} ^ {T} q (\mathbf {x} _ {t} | \mathbf {x} _ {t - 1}), \\ q (\mathbf {x} _ {t} | \mathbf {x} _ {t - 1}) := \mathcal {N} (\mathbf {x} _ {t}; \sqrt {1 - \beta_ {t}} \mathbf {x} _ {t - 1}, \beta_ {t} \mathbf {I}). \end{array}\tag{1}
$$

逆过程 $p _ { \theta } ( \mathbf { x } _ { 0 : T } )$ 也是一个马尔可夫链，它从 $p ( \mathbf { x } _ { T } ) = \mathcal { N } ( \mathbf { x } _ { T } ; \mathbf { 0 } , \mathbf { I } )$ 开始，利用学习到 Gaussian 转移来预测并消除噪声

$$
\begin{array}{r l} & p _ {\theta} (\mathbf {x} _ {0: T}) := p (\mathbf {x} _ {T}) \prod_ {t = 1} ^ {T} p _ {\theta} (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t}), \\ & p _ {\theta} (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t}) := \mathcal {N} (\mathbf {x} _ {t - 1}; \mu_ {\theta} (\mathbf {x} _ {t}, t), \Sigma_ {\theta} (\mathbf {x} _ {t}, t)). \end{array}\tag{2}
$$

为了完成扩散模型的逆过程，我们需要构建并优化一个神经网络。在训练期间，首先我们为每个真实动作 $\mathbf { x } _ { \mathrm { 0 } }$ 均匀采样时间步 t，然后从 $q ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } )$ 生成一个样本。Ho et al. (2020) 没有在 $\mathbf { x } _ { \mathrm { 0 } }$ 上重复添加噪声，而是将扩散过程形式化为

$$
q (\mathbf {x} _ {t} | \mathbf {x} _ {0}) = \sqrt {\bar {\alpha_ {t}}} \mathbf {x} _ {0} + \epsilon \sqrt {1 - \bar {\alpha_ {t}}}, \epsilon \sim \mathcal {N} (\mathbf {0}, \mathbf {I}),\tag{3}
$$

![](images/47195fd95167963d94767aa4d7b43b37d11e3b96d96caca3caade404ce60ba73.jpg)  
Fig. 3 我们的线性自注意力（Linear Self-attention）架构。

其中 $\begin{array} { r } { \alpha _ { t } = 1 - \beta _ { t } , \bar { \alpha _ { t } } = \prod _ { s = 0 } ^ { t } \alpha _ { s } } \end{array}$。因此，我们可以简单地采样一个噪声，然后通过该公式直接生成 $\mathbf { x } _ { t }$。与预测 $\mathbf { x } _ { t - 1 }$ 不同，这里我们遵循 GLIDE (Nichol et al., 2021) 并预测噪声项。这意味着我们需要构建一个网络来拟合 $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t , \mathrm { t e x t } )$。我们优化模型参数以减小均方误差，如下所示：

$$
\mathcal {L} = \mathrm{E} _ {t \in [ 1, T ], \mathbf {x} _ {0} \sim q (\mathbf {x} _ {0}), \epsilon \sim \mathcal {N} (\mathbf {0}, \mathbf {I})} [ \| \epsilon - \epsilon_ {\theta} (\mathbf {x} _ {t}, t, \text { text }) \| ].\tag{4}
$$

这是我们在模型训练中使用的唯一损失。为了从给定的文本描述生成样本，我们从 $p ( \mathbf { x } _ { T } ) = \mathcal { N } ( \mathbf { x } _ { T } ; \mathbf { 0 } , \mathbf { I } )$ 开始对序列进行去噪。公式 2 表明我们需要估计 $\mu _ { \theta } ( \mathbf { x } _ { t } , t ,$ text) 和 $\textstyle \sum _ { \theta } ( \mathbf { x } _ { t } , t , \mathrm { t e x t } )$。为了简化问题，我们将 $\varSigma _ { \theta } ( \mathbf { x } _ { t } , t , \mathrm { t e x t } )$ 设为常数 $\beta _ { t }$。$\mu _ { \theta } ( \mathbf { x } _ { t } , t , \mathrm { t e x t } $ ) 可以估计为

$$
\mu_ {\theta} (\mathbf {x} _ {t}, t, \text {text}) = \frac {1}{\sqrt {\mathbf {x} _ {t}}} (\mathbf {x} _ {t} - \frac {1 - \alpha_ {t}}{\sqrt {1 - \bar {\alpha} _ {t}}} \epsilon_ {\theta} (\mathbf {x} _ {t}, t, \text {text})).\tag{5}
$$

因此，我们可以逐步对动作序列进行去噪，最终获得以给定文本为条件的清晰动作序列。

## 3.4 跨模态线性 Transformer

在第 3.3 节中，我们将扩散模型作为动作生成器进行了阐述，其中神经网络 $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t , \mathrm { t e x t } )$ 对去噪步骤至关重要。在本节中，我们将介绍所提出的 MotionDiffuse 中 $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t , \mathrm { t e x t } )$ 的设计。

以往的工作（Ho et al., 2020; Dhariwal and Nichol, 2021; Nichol and Dhariwal, 2021; Nichol et al., 2021）主要采用类 UNet 结构作为去噪模型。然而，在动作生成任务中，目标动作序列是变长的，这使得基于卷积的网络不再适用。因此，我们提出了一种跨模态线性 Transformer（Cross-Modality Linear Transformer），如图 Figure 2 所示。与机器翻译任务类似，我们提出的模型包含一个文本编码器和一个动作解码器。为了满足扩散模型的要求，我们进一步对动作解码器的每一层进行了定制。

文本编码器（Text Encoder） 在这里我们直接使用经典的 Transformer（Vaswani et al., 2017）来提取文本特征。具体而言，输入数据首先通过嵌入层从原始文本中获取嵌入特征，随后由一系列 Transformer 块进一步处理。每个块包含两个组件：多头注意力模块（MHA）和前馈网络（FFN）。假设输入特征为 $\mathbf { X } \in \mathbb { R } ^ { n \times d }$（n 表示元素数量，d 表示元素特征维度），MHA 提取查询特征向量 $\mathbf { Q } \in \mathbb { R } ^ { n \times d }$、键特征向量 K $\in \mathbb { R } ^ { n \times d }$ 以及值特征向量 $\mathbf { V } \in \mathbb { R } ^ { n \times d }$ 如下：

$$
\mathbf {Q} = W _ {q} \mathbf {X}, \quad \mathbf {K} = W _ {k} \mathbf {X}, \quad \mathbf {V} = W _ {v} \mathbf {X},\tag{6}
$$

其中 $W _ { q } , \ W _ { k }$ 和 $W _ { v }$ 分别是用于生成 Q、K 和 V 的相应线性投影。随后，值特征通过注意力权重 $\mathbf { A } \in \mathbb { R } ^ { n \times n }$ 进行聚合：

$$
\mathbf {A} = \mathrm{softmax} (\frac {\mathbf {Q} \otimes \mathbf {K} ^ {\top}}{\sqrt {d}}), \quad \mathbf {Y} = \mathbf {A} \otimes \mathbf {V},\tag{7}
$$

其中 $\mathbf { Y } \in \mathbb { R } ^ { n \times d }$ 是 MHA 模块的输出，d 是 X 中每个元素的维度，⊗ 表示矩阵乘法。多头机制将输入向量分成若干部分，并独立地通过公式 6 和 7 中的过程。各输出被拼接在一起，使得维度保持不变。在 MHA 模块的输入和输出之间应用了残差连接。该特征随后由 FFN 进一步处理，该网络包含三次线性变换以及夹在它们之间的两个 GELU (Hendrycks and Gimpel, 2016) 层。

为了提高泛化能力，我们使用 CLIP (Radford et al., 2021) 中的参数权重来初始化前几层。这部分参数被冻结，在后续训练中不会被优化。

线性自注意力（Linear Self-attention）。该模块旨在通过建模不同帧之间的相关性来增强动作特征。自注意力的主要优势在于获取输入序列的全局概览，从而有利于估计注入的噪声。然而，计算注意力权重 $\mathbf { A } \in \mathbb { R } ^ { n \times n }$ 的时间复杂度为 $\mathcal { O } ( n ^ { 2 } d )$。当目标动作序列长度增加时，时间开销呈二次方增长。在 T2M 任务中，长度可达数百，这导致运行速度变慢。因此，我们采用高效注意力（Efficient Attention）(Shen et al., 2021) 来加速自注意力模块。高效注意力不再计算成对（pair-wise）的注意力权重，而是生成全局特征图 $\mathbf { F } _ { g } \in \mathbb { R } ^ { d _ { k } \times d _ { k } }$，其中 $d _ { k }$ 是多头切分后的特征维度：

$$
\mathbf {F} _ {g} = \operatorname{softmax} (\mathbf {K} ^ {\top}) \otimes \mathbf {V}, \quad \mathbf {Y} = \operatorname{softmax} (\mathbf {Q}) \otimes \mathbf {F} _ {g}.\tag{8}
$$

这两个步骤的时间复杂度为 $\mathcal { O } ( d _ { k } ^ { 2 } n k ) =$ $\mathcal { O } ( d d _ { k } n )$，其中 n 是序列中的元素数量，k 是自注意力的头数。

高效注意力对于扩散模型的另一个优势是，特征图 $\mathbf { F } _ { g }$ 显式地聚合了全局信息，而经典自注意力则更侧重于成对关系。相比于成对关系，全局信息为动作序列的语义含义提供了更多线索。实验结果也证实了这一结论。

线性交叉注意力（Linear Cross-attention）。交叉注意力在计算 K 和 V 时将 X 替换为文本特征。其他公式与线性自注意力相同。在此过程中，文本特征被注入到动作序列中，以生成以给定文本为条件的动作。

风格化块（Stylization Block）。在每个去噪步骤中，输出以给定的文本和时间戳 $t$ 为条件。线性交叉注意力（Linear Cross-attention）将文本特征融合到动作序列中。我们需要另一个风格化块组件将时间戳 $t$ 引入生成过程。该块应用在每个线性自注意力块、线性交叉注意力块和 FFN 块之后。

类似于 GLIDE (Nichol et al., 2021)，我们首先通过对文本特征进行线性变换获得文本嵌入 $e _ { \mathrm { t e x t } }$，并通过位置嵌入（Vaswani et al., 2017）获得时间戳嵌入 $\mathbf { e _ { t } }$。这两项相加为一个单一向量 $\mathbf{e}$。给定来自其他块的原始输出 $\mathbf{Y}$，风格化块将如下处理该特征：

$$
\mathbf {B} = \psi_ {b} (\phi (\mathbf {e})), \quad \mathbf {W} = \psi_ {w} (\phi (\mathbf {e})), \quad \mathbf {Y} ^ {\prime} = \mathbf {Y} \cdot \mathbf {W} + \mathbf {B},\tag{9}
$$

其中 $(\cdot)$ 表示哈达玛积（Hadamard product），$\mathbf { Y } ^ { \prime }$ 是风格化块的输出。$\psi _ { b } , \psi _ { w } , \phi$ 表示三个不同的线性投影。在经典的 Transformer 中，每个块的输出作为残差连接添加到原始输入中，如图 Figure 2 所示。在 Motion-Diffuse 中，这些输出通过风格化块并添加到信息中。这种修改使我们提出的方法能够感知时间戳 $t$。

## 3.5 细粒度控制

为了丰富 MotionDiffuse 的能力，我们探索了动作表示以及 DDPM 去噪过程的特性。与 VAE 不同，生成的动作序列采用显式形式，而不是被压缩在潜空间中。基于 DDPM 的动作生成的这一特性允许对该动作序列应用更多操作，以提高可操控性。

身体部位独立控制。由于文本描述缺乏多样性，我们无法仅凭文本描述实现对每个身体部位的精确动作控制。例如，提示词“一个人正在奔跑并挥动左手”对模型极具挑战性，因为预期的动作序列与训练分布相去甚远。即使我们手动将原始描述拆分为两个独立的描述：针对下肢的“一个人正在奔跑”，以及针对上肢的“$ { \mathbf { \hat { \rho } } } _ { \mathrm { a } }$ 个人正在挥动左手”，模型仍然难以生成正确的动作。针对这种情况，一个直观的解决方案是分别生成两个动作序列，并将第一个序列的上肢动作与第二个序列的下肢动作结合起来。这种简单的解决方案在一定程度上缓解了该问题。然而，它忽略了这两个部位之间的相关性。具体到“奔跑并挥动左手”这一动作，这两个动作的频率应当保持匹配。否则，通过这种朴素方法生成的动作会显得不自然。为了更好地解决这一问题，我们提出了一种身体部位独立控制方案。

回想一下，在去噪过程中，我们的扩散模型（diffusion model）预测噪声项 $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t , \mathrm { t e x t } ) \ \in$ $\mathbb { R } ^ { F \times D }$，其中 $F$ 表示帧数，$D$ 表示每个姿态状态的维度，包括身体关节的平移和旋转。该噪声项决定了全身的去噪方向。

受到潜在编码插值（latent code interpolation）应用的启发，在此我们提出了“噪声插值”（noise interpolation）来分别控制人体的不同部位。假设我们针对不同的身体部位 $\{ s _ { i } \}$ 有 $n$ 个文本描述 $\left\{ \mathrm { t e x t } _ { i } \right\}$。我们希望计算噪声项 $\epsilon = \{ \epsilon _ { i } ^ { \mathrm { j o i n t } } \} , i \in [ 1 , m ]$，其中 $\epsilon _ { i } ^ { \mathrm { j o i n t } }$ 表示第 $i$ 个身体部位的噪声项，$m$ 表示划分的数量。我们首先估计噪声 $\epsilon _ { i } ^ { \mathrm { p a r t } } =$ $\epsilon _ { \theta } \big ( \mathbf { x } _ { t } , t , \mathrm { t e x t } _ { i } \big ) , \epsilon _ { i } ^ { \mathrm { p a r t } } \ \in \ \mathbb { R } ^ { F \times D }$。组合这些项的一个直观方法是 $\begin{array} { r } { \epsilon ^ { \mathrm { p a r t } } = \sum _ { i = 1 } ^ { m } \epsilon _ { i } ^ { \mathrm { p a r t } } \cdot M _ { i } . } \end{array}$，其中 $M _ { i } \in \{ 0 , 1 \} ^ { D }$ 是一个用于指示我们应当关注哪个身体部位的二进制向量。$(\cdot)$ 表示阿达玛积（Hadamard product），为了简单起见，这里我们在计算中忽略广播机制。尽管这种方法在某种程度上取得了成功，但在 $\epsilon _ { i } ^ { \mathrm { p a r t } }$ 中直接忽略某些部位会增加组合的难度并导致低质量的生成结果。因此，我们添加了一个校正项用于平滑插值：

$$
\bar {\epsilon} ^ {\text { part }} = \sum_ {i = 1} ^ {m} \epsilon_ {i} ^ {\text { part }} \cdot M _ {i} + \lambda_ {1} \cdot \nabla (\sum_ {1 \leq i, j \leq m} \| \epsilon_ {i} ^ {\text { part }} - \epsilon_ {j} ^ {\text { part }} \|),\tag{10}
$$

其中 $\nabla$ 表示梯度计算，$\lambda _ { 1 }$ 是平衡这两项的超参数。该校正项强制施加了更平滑的去噪方向，从而使不同身体部位的运动更加自然。

时变控制（Time-varied Controlling） 长期动作生成在实际应用中起着至关重要的作用。以往的工作主要聚焦于单一动作类型的动作生成。自回归方法（Henter et al., 2020; Guo et al., 2022）以令人满意的性能解决了这一问题。然而，它们都无法以连续的方式合成不同的动作。得益于 DDPM 的特性，在此我们提出了另一种采样方法来满足这一需求。

回想一下，给定一个数组 $\{ \mathrm { t e x t } _ { i , j } , [ l _ { i , j } , r _ { i , j } ] \}$ $i \in [ 1 , m ]$ ，其中 $m$ 是时间区间的数量。类似于我们在上一段中提出的方法，我们首先独立估计第 $i$ 个区间的噪声项 $\epsilon _ { i } ^ { \mathrm { t i m e } }$。假设目标序列的总长度为 $F ^ { \prime } .$。通过填充零，我们将每个噪声项扩展为相同的维度 $F ^ { \prime } \times D$。然后，我们使用一个修正项对这些噪声进行插值：

$$
\bar {\epsilon} ^ {\text {time}} = \sum_ {i = 1} ^ {m} \bar {\epsilon} _ {i} ^ {\text {time}} + \lambda_ {2} \cdot \nabla (\sum_ {1 \leq i, j \leq m} \| \bar {\epsilon} _ {i} ^ {\text {time}} - \bar {\epsilon} _ {j} ^ {\text {time}} \|),\tag{11}
$$

其中 $\overline { { \epsilon } } _ { j } ^ { \mathrm { t i m e } }$ 是由 $\epsilon _ { i } ^ { \mathrm { t i m e } }$ 填充后的项，$\lambda _ { 2 }$ 是一个超参数。

## 4 实验

我们通过三类实验评估 MotionDifuse：文本驱动的动作生成（第 4.1 节）、动作类别条件的动作生成（第 4.2 节）以及动作编辑（第 4.3 节）。在所有评估的基准测试中，MotionDifuse 均显著优于先前的 SOTA（最先进）方法。

## 4.1 文本驱动的动作生成

数据集。KIT Motion Language 数据集 (Plappert et al., 2016) 提供了 3911 个动作序列和 6353 条序列级的自然语言描述。HumanML3D (Guo et al., 2022) 重新标注了 AMASS 数据集 (Mahmood et al., 2019) 和 HumanAct12 数据集 (Guo et al., 2020)。它在 14616 个动作序列上提供了 44970 条标注。KIT 和 HumanML3D 是文本驱动动作生成任务的两个重要基准。遵循 Guo et al. (2022)，我们使用了预训练的文本-动作对比模型。

评估指标。我们使用五种不同的指标来评估所有方法。1) Frechet Inception Distance (FID)。通过预训练的动作编码器从生成结果和真实动作序列中提取特征。计算这两个分布之间的 FID 以衡量相似度。2) R Precision。对于每对生成的序列和描述文本，从测试集中随机选择 31 个其他提示词。预训练的对比模型计算平均 top-k 准确率。本节报告 top-1、top-2 和 top-3 准确率。3) 多样性（Diversity）。由所有测试文本生成的序列被随机分成数对。然后计算每对序列之间的平均关节点差异，以此作为多样性指标。4) 多模态性（Multimodality）。针对单个文本描述，我们随机生成 32 个动作序列。多模态性衡量这些同质动作序列之间关节点位置的差异。5) 多模态距离（Multimodal Distance）。在预训练对比模型的辅助下，我们可以计算给定描述的文本特征与生成结果的动作特征之间的差异，称为多模态距离。

在本节中，R Precision 和 FID 是我们得出大部分结论的主要指标。此外，为了公平比较，我们对每次评估运行 20 次，并报告具有 95% 置信度的统计区间。

实现细节。对于 HumanML3D 和 KIT-ML 数据集，我们构建了一个 8 层的 Transformer 作为动作解码器。对于文本编码器，我们首先直接使用 CLIP ViT-B/32 (Radford et al., 2021) 中的文本编码器，然后额外添加 4 个 Transformer 编码器层。文本编码器和动作解码器的潜在维度分别为 256 和 512。对于扩散模型，扩散步数为 1000，方差 $\beta _ { t }$ 从 0.0001 线性增加至 0.02。我们选择 Adam 作为优化器，以 0.0002 的学习率训练模型。我们使用 8 块 Tesla V100 进行训练，每块 GPU 上有 128 个样本，因此总批次大小为 1024。KIT-ML 的总迭代次数为 40K，HumanML3D 为 100K。

遵循 Guo et al. (2022) 的方法，本系列实验中的姿态状态主要包含七个不同部分：$( r ^ { v a } , r ^ { v x } , r ^ { v z } , r ^ { h } , \mathbf { j } ^ { p } , \mathbf { j } ^ { v } , \mathbf { j } ^ { r } )$。此处 Y 轴垂直于地面。$r ^ { v a } , r ^ { v x } , r ^ { v z } \in \mathbb { R }$ 分别表示根关节绕 Y 轴的角速度、沿 X 轴和 Z 轴的线速度。$r ^ { h } \in \mathbb { R }$ 为根关节的高度。$\mathbf { j } ^ { p } , \mathbf { j } ^ { v } \in \mathbb { R } ^ { J \times 3 }$ 为每个关节的位置和线速度，其中 J 为关节数量。$\mathbf { j } ^ { r } \in \mathbb { R } ^ { J \times 6 }$ 为每个关节的 6D 旋转表示（Zhou et al., 2019）。具体而言，在 HumanML3D 中 J 为 22，在 KIT-ML 中为 21。

Table 1 HumanML3D 测试集上的定量结果。所有方法均使用来自真实值（ground truth）的真实动作长度。“→”表示该指标越接近真实动作结果越好。我们将所有评估运行 20 次，± 表示 95% 置信区间。最佳结果以粗体表示。

<table><tr><td rowspan="2">方法</td><td colspan="3">R Precision↑</td><td rowspan="2">FID↓</td><td rowspan="2">MultiModal Dist↓</td><td rowspan="2">Diversity→</td><td rowspan="2">MultiModality</td></tr><tr><td>Top 1</td><td>Top 2</td><td>Top 3</td></tr><tr><td>真实动作</td><td> $0.511^{\pm .003}$ </td><td> $0.703^{\pm .003}$ </td><td> $0.797^{\pm .002}$ </td><td> $0.002^{\pm .000}$ </td><td> $2.974^{\pm .008}$ </td><td> $9.503^{\pm .065}$ </td><td>-</td></tr><tr><td>Language2Pose</td><td> $0.246^{\pm .002}$ </td><td> $0.387^{\pm .002}$ </td><td> $0.486^{\pm .002}$ </td><td> $11.02^{\pm .046}$ </td><td> $5.296^{\pm .008}$ </td><td> $7.676^{\pm .058}$ </td><td>-</td></tr><tr><td>Text2Gesture</td><td> $0.165^{\pm .001}$ </td><td> $0.267^{\pm .002}$ </td><td> $0.345^{\pm .002}$ </td><td> $7.664^{\pm .030}$ </td><td> $6.030^{\pm .008}$ </td><td> $6.409^{\pm .071}$ </td><td>-</td></tr><tr><td>MoCoGAN</td><td> $0.037^{\pm .000}$ </td><td> $0.072^{\pm .001}$ </td><td> $0.106^{\pm .001}$ </td><td> $94.41^{\pm .021}$ </td><td> $9.643^{\pm .006}$ </td><td> $0.462^{\pm .008}$ </td><td> $0.019^{\pm .000}$ </td></tr><tr><td>Dance2Music</td><td> $0.033^{\pm .000}$ </td><td> $0.065^{\pm .001}$ </td><td> $0.097^{\pm .001}$ </td><td> $66.98^{\pm .016}$ </td><td> $8.116^{\pm .006}$ </td><td> $0.725^{\pm .011}$ </td><td> $0.043^{\pm .001}$ </td></tr><tr><td>Guo et al.</td><td> $0.457^{\pm .002}$ </td><td> $0.639^{\pm .003}$ </td><td> $0.740^{\pm .003}$ </td><td> $1.067^{\pm .002}$ </td><td> $3.340^{\pm .008}$ </td><td> $9.188^{\pm .002}$ </td><td> $2.090^{\pm .083}$ </td></tr><tr><td>Ours（本文方法）</td><td> $0.491^{\pm .001}$ </td><td> $0.681^{\pm .001}$ </td><td> $0.782^{\pm .001}$ </td><td> $0.630^{\pm .001}$ </td><td> $3.113^{\pm .001}$ </td><td> $9.410^{\pm .049}$ </td><td> $1.553^{\pm .042}$ </td></tr></table>

Table 2 KIT-ML 测试集上的定量结果。所有方法均使用来自真实值（ground truth）的真实动作长度。

<table><tr><td rowspan="2">方法</td><td colspan="3">R Precision↑</td><td rowspan="2">FID↓</td><td rowspan="2">多模态距离↓</td><td rowspan="2">多样性→</td><td rowspan="2">多模态性</td></tr><tr><td>Top 1</td><td>Top 2</td><td>Top 3</td></tr><tr><td>真实动作</td><td> $0.424^{\pm .005}$ </td><td> $0.649^{\pm .006}$ </td><td> $0.779^{\pm .006}$ </td><td> $0.031^{\pm .004}$ </td><td> $2.788^{\pm .012}$ </td><td> $11.08^{\pm .097}$ </td><td>-</td></tr><tr><td>Language2Pose</td><td> $0.221^{\pm .005}$ </td><td> $0.373^{\pm .004}$ </td><td> $0.483^{\pm .005}$ </td><td> $6.545^{\pm .072}$ </td><td> $5.147^{\pm .030}$ </td><td> $9.073^{\pm .100}$ </td><td>-</td></tr><tr><td>Text2Gesture</td><td> $0.156^{\pm .004}$ </td><td> $0.255^{\pm .004}$ </td><td> $0.338^{\pm .005}$ </td><td> $12.12^{\pm .183}$ </td><td> $6.964^{\pm .029}$ </td><td> $9.334^{\pm .079}$ </td><td>-</td></tr><tr><td>MoCoGAN</td><td> $0.022^{\pm .002}$ </td><td> $0.042^{\pm .003}$ </td><td> $0.063^{\pm .003}$ </td><td> $82.69^{\pm .242}$ </td><td> $10.47^{\pm .012}$ </td><td> $3.091^{\pm .043}$ </td><td> $0.250^{\pm .009}$ </td></tr><tr><td>Dance2Music</td><td> $0.031^{\pm .002}$ </td><td> $0.058^{\pm .002}$ </td><td> $0.086^{\pm .003}$ </td><td> $115.4^{\pm .240}$ </td><td> $10.40^{\pm .016}$ </td><td> $0.241^{\pm .004}$ </td><td> $0.062^{\pm .002}$ </td></tr><tr><td>Guo et al.</td><td> $0.370^{\pm .005}$ </td><td> $0.569^{\pm .007}$ </td><td> $0.693^{\pm .007}$ </td><td> $2.770^{\pm .109}$ </td><td> $3.401^{\pm .008}$ </td><td> $10.91^{\pm .119}$ </td><td> $1.482^{\pm .065}$ </td></tr><tr><td>Ours</td><td> $0.417^{\pm .004}$ </td><td> $0.621^{\pm .004}$ </td><td> $0.739^{\pm .004}$ </td><td> $1.954^{\pm .062}$ </td><td> $2.958^{\pm .005}$ </td><td> $11.10^{\pm .143}$ </td><td> $0.730^{\pm .013}$ </td></tr></table>

Table 3 预训练 CLIP 与高效注意力机制的消融实验。所有结果均在 KIT-ML 测试集上报告。

<table><tr><td rowspan="2">CLIP</td><td rowspan="2">EFF</td><td colspan="3">R Precision↑</td></tr><tr><td>Top 1</td><td>Top 2</td><td>Top 3</td></tr><tr><td>N</td><td>N</td><td> $0.288^{\pm .004}$ </td><td> $0.440^{\pm .004}$ </td><td> $0.539^{\pm .004}$ </td></tr><tr><td>N</td><td>Y</td><td> $0.136^{\pm .003}$ </td><td> $0.233^{\pm .003}$ </td><td> $0.309^{\pm .003}$ </td></tr><tr><td>Y</td><td>N</td><td> $0.357^{\pm .004}$ </td><td> $0.555^{\pm .004}$ </td><td> $0.679^{\pm .005}$ </td></tr><tr><td>Y</td><td>Y</td><td> $0.417^{\pm .004}$ </td><td> $0.621^{\pm .004}$ </td><td> $0.739^{\pm .004}$ </td></tr></table>

定量结果。我们将提出的 MotionDifuse 与五个基线模型进行比较：Language2Pose (Ahuja and Morency, 2019)、Text2Gesture (Bhattacharya et al., 2021)、MoCo-GAN (Tulyakov et al., 2018)、Dance2Music (Lee et al., 2019) 和 Guo et al. (2022)。所有基线模型的表现均引用自 Guo et al. (2022)。Table 1 和 Table 2 展示了在 HumanML3D 数据集和 KIT-ML 数据集上的定量比较。我们提出的 MotionDifuse 在精度（precision）、FID、多模态距离（MultiModal Distance）以及多样性（Diversity）方面均以显著优势超越了所有现有方法。MotionDifuse 的精度甚至接近真实动作，这表明我们生成的动作序列具有令人满意的保真度和真实感。

为了进一步理解 CLIP 初始化和高效注意力机制（efficient attention）的作用，我们在 Table 3 中报告了消融实验结果。未使用预训练 CLIP 的模型性能出现严重下降，这表明预训练语言模型对于文本生成动作（T2M）任务的必要性。对于高效注意力机制，当我们同时使用 CLIP 时，它会带来显著的收益。然而，在没有 CLIP 的情况下，该模块也会限制模型的性能。对这一现象的一个合理解释是，当来自给定文本的语义信息不足时，高效注意力机制中的全局关系可能会产生误导。

Guo et al. (2022) 指出 MultiModality（多模态性）指标的结果应当尽可能大。然而，在基于动作条件的动作生成任务的相关文献（Guo et al., 2020; Petrovich et al., 2021; Cervantes et al., 2022）中认为，该指标应当接近真实动作。在 T2M 任务中，计算真实动作的该指标较为困难。因此，我们仅报告这些结果，而不进行比较。

我们探讨了网络架构规模对性能的影响。Table 4 表明潜变量维度（latent dimension）起着更为重要的作用。潜变量维度为 512 的模型显著优于潜变量维度为 256 的模型。相反，当潜变量维度为 128 或 256 时，增加网络层数可以提升性能，但当维度为 512 时，增加层数的影响甚微。

定性结果 Figure 4 展示了我们的方法与作为基线的 Guo et al. (2022) 之间的比较。我们强调，MotionDifuse 在多样性与真实性之间取得了平衡。例如，对于提示词（prompt）“一个人挥动高尔夫球杆并击球”，我们生成的动作更为忠实地刻画了所描述的动作。相比之下，基线方法以牺牲准确性为代价换取了较高的多模态性。此外，当给定复杂的提示词（例如“一个人被用力推向左侧，然后恢复到站立姿势”）时，MotionDifuse 能够生成反映详细描述的高质量动作，而基线方法则未能生成任何有意义的动作。

![](images/f2e2999ac9e2a4da18e3259c6ca531a79adf6a5fd53e01225d2b8eb5532d502a.jpg)  
Fig. 4 HumanML3D 数据集上的定性结果。我们将我们的方法与 Guo et al. (2022) 进行了对比，并针对每个给定的提示词可视化了两个样本。MotionDiffuse 能够兼顾准确性与多样性。

Table 4 隐层维度和 Transformer 层数的消融实验。所有结果均在 KIT-ML 测试集上汇报。

<table><tr><td rowspan="2">#层数</td><td rowspan="2">维度</td><td colspan="3">R Precision↑</td></tr><tr><td>Top 1</td><td>Top 2</td><td>Top 3</td></tr><tr><td>4</td><td>128</td><td>0.033±.002</td><td>0.066±.003</td><td>0.097±.003</td></tr><tr><td>4</td><td>256</td><td>0.095±.002</td><td>0.166±.003</td><td>0.227±.003</td></tr><tr><td>4</td><td>512</td><td>0.405±.005</td><td>0.620±.005</td><td>0.743±.004</td></tr><tr><td>8</td><td>128</td><td>0.025±.002</td><td>0.053±.002</td><td>0.086±.002</td></tr><tr><td>8</td><td>256</td><td>0.198±.003</td><td>0.335±.004</td><td>0.441±.004</td></tr><tr><td>8</td><td>512</td><td>0.417±.004</td><td>0.621±.004</td><td>0.739±.004</td></tr><tr><td>12</td><td>128</td><td>0.031±.002</td><td>0.063±.003</td><td>0.091±.002</td></tr><tr><td>12</td><td>256</td><td>0.209±.003</td><td>0.348±.004</td><td>0.452±.003</td></tr><tr><td>12</td><td>512</td><td>0.412±.006</td><td>0.616±.004</td><td>0.741±.004</td></tr></table>

## 4.2 动作条件动作生成

数据集。HumanAct12 数据集（Guo et al., 2020）提供了 12 种动作序列。该数据集改编自包含 1191 个视频的 PHSPD 数据集（Zou et al., 2020）。HumanAct12 进一步将这些视频整理为经过裁剪的动作片段。UESTC 数据集（Ji et al., 2018）也是动作条件动作生成任务的一个重要基准，其中包含涵盖 40 个不同动作类别的 25K 个动作序列。Petrovich et al. (2021) 进一步使用预训练的 VIBE（Kocabas et al., 2020）从 UESTC 数据集中提取 SMPL（Loper et al., 2015）序列，并提供了预训练的动作识别模型用于评估。

评估指标。该任务采用了四项评估指标：FID、Accuracy、Diversity 和 Multimodality。预训练的动作识别模块可以直接计算所有动作类别的平均准确率，而无需构建小批量数据。该指标与 R Precision 具有相似的作用。其他三项指标已在第 4.1 节中介绍。HumanAct12 没有官方划分，我们在整个数据集上汇报 FID。UESTC 具有测试集划分，因此我们在其上汇报 FID，这比训练集划分更具代表性。在本节中，FID 和 Accuracy 是两个主要指标。我们的结论主要基于这两者。

Table 5 动作条件动作生成的定量结果。对于 UESTC 数据集，我们汇报在测试集划分上的 FID。MM：MultiModality。

<table><tr><td rowspan="2">方法</td><td colspan="4">HumanAct12</td><td colspan="4">UESTC</td></tr><tr><td>FID↓</td><td>Accuracy↑</td><td>Diversity→</td><td>MM→</td><td>FID↓</td><td>Accuracy↑</td><td>Diversity→</td><td>MM→</td></tr><tr><td>真实动作</td><td> $0.020^{\pm .010}$ </td><td> $0.997^{\pm .001}$ </td><td> $6.850^{\pm .050}$ </td><td> $2.450^{\pm .040}$ </td><td> $2.79^{\pm .29}$ </td><td> $0.988^{\pm .001}$ </td><td> $33.34^{\pm .320}$ </td><td> $14.16^{\pm .06}$ </td></tr><tr><td>Action2Motion</td><td> $0.338^{\pm .015}$ </td><td> $0.917^{\pm .003}$ </td><td> $6.879^{\pm .066}$ </td><td> $2.511^{\pm .023}$ </td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ACTOR</td><td> $0.12^{\pm .00}$ </td><td> $0.955^{\pm .008}$ </td><td> $6.84^{\pm .03}$ </td><td> $2.53^{\pm .02}$ </td><td> $23.43^{\pm 2.20}$ </td><td> $0.911^{\pm .003}$ </td><td> $31.96^{\pm .33}$ </td><td> $14.52^{\pm .09}$ </td></tr><tr><td>INR</td><td> $0.088^{\pm .004}$ </td><td> $0.973^{\pm .001}$ </td><td> $6.881^{\pm .048}$ </td><td> $2.569^{\pm .040}$ </td><td> $15.00^{\pm .09}$ </td><td> $0.941^{\pm .001}$ </td><td> $31.59^{\pm .19}$ </td><td> $14.68^{\pm .07}$ </td></tr><tr><td>Ours</td><td> $0.07^{\pm .00}$ </td><td> $0.992^{\pm .13}$ </td><td> $6.85^{\pm .02}$ </td><td> $2.46^{\pm .02}$ </td><td> $9.10^{\pm .437}$ </td><td> $0.950^{\pm .000}$ </td><td> $32.42^{\pm .214}$ </td><td> $14.74^{\pm .07}$ </td></tr></table>

实现细节。除了学习率、迭代次数和动作表示之外，所有设置均与文本驱动动作生成任务相同。在这一系列实验中，我们在 HumanAct12 数据集上训练 100K 次迭代，在 UESTC 数据集上训练 500K 次迭代，两者的学习率均为 0.0001。

本任务中的动作表示与 T2M（文本到动作）任务略有不同。对于 HumanAct12 数据集，每个姿态状态可以表示为 $( \mathbf { j } ^ { x } , \mathbf { j } ^ { y } , \mathbf { j } ^ { z } )$，其中 $\mathbf { j } ^ { x } , \mathbf { j } ^ { y } , \mathbf { j } ^ { z } \in \mathbb { R } ^ { 2 4 \times 3 }$ 为 24 个关节点的坐标。对于 UESTC 数据集，我们使用 $( r ^ { x } , r ^ { y } , r ^ { z } , \mathbf { j } ^ { r } )$ 作为姿态表示，其中 $r ^ { x } , r ^ { y } , r ^ { z } \in \mathbb { R }$ 为根关节点的坐标，$\mathbf { j } ^ { r } \in \mathbb { R } ^ { 2 4 \times 6 }$ 为采用 6D 表示的每个关节点的旋转角度。

定量结果。遵循 Cervantes et al. (2022)，我们选取了三个基线模型：Action2Motion (Guo et al., 2020)、ACTOR (Petrovich et al., 2021)、INR (Cervantes et al., 2022)。Table 5 展示了在 HumanAct12 数据集和 UESTC 数据集上的定量结果。与现有的其他工作相比，我们提出的 MotionDifuse 在 FID 和准确率（Accuracy）方面均取得了最佳性能。我们想强调的是，在 HumanAct12 数据集上，我们的结果在所有四个指标上都明显接近真实动作。

Table 6 潜空间维度与 Transformer 层数的消融实验。所有结果均在 HumanAct12 数据集上报告。

<table><tr><td>层数</td><td>维度</td><td>FID↓</td><td>Accuracy↑</td></tr><tr><td>4</td><td>128</td><td> $0.29^{\pm 0.00}$ </td><td> $0.892^{\pm 1.97}$ </td></tr><tr><td>4</td><td>256</td><td> $0.14^{\pm 0.00}$ </td><td> $0.958^{\pm 0.51}$ </td></tr><tr><td>4</td><td>512</td><td> $0.09^{\pm 0.00}$ </td><td> $0.984^{\pm 0.21}$ </td></tr><tr><td>8</td><td>128</td><td> $0.22^{\pm 0.00}$ </td><td> $0.929^{\pm 1.04}$ </td></tr><tr><td>8</td><td>256</td><td> $0.09^{\pm 0.00}$ </td><td> $0.983^{\pm 0.23}$ </td></tr><tr><td>8</td><td>512</td><td> $\mathbf{0.07^{\pm 0.00}}$ </td><td> $0.992^{\pm 0.13}$ </td></tr><tr><td>12</td><td>128</td><td> $0.11^{\pm 0.00}$ </td><td> $0.954^{\pm 0.67}$ </td></tr><tr><td>12</td><td>256</td><td> $0.10^{\pm 0.00}$ </td><td> $0.988^{\pm 0.21}$ </td></tr><tr><td>12</td><td>512</td><td> $0.08^{\pm 0.00}$ </td><td> $\mathbf{0.996^{\pm 0.08}}$ </td></tr></table>

在此我们还尝试了潜空间维度与层数的不同组合，如 Table 6 所示。与我们在第 4.1 节中得出的结论类似，潜空间维度比层数更为重要。

## 4.3 动作操控

为了更好地评估文本驱动动作生成模型的能力，我们设计了两种任务变体。首先是空间多样性 T2M 任务（Spatially-diverse T2M task，T2M-S）。T2M-S 要求生成的动作序列在不同身体部位包含多种动作$( e . g . \ ^ { \circ } \mathrm { \dot { a } }$ 个人在跑步的同时喝水”）。具体而言，T2M-S 任务中的第 $i$ 个测试样本可以表示为一组文本-掩码对 $\{ ( \mathrm { t e x t } _ { i , j } , \mathrm { M } _ { i , j } ) \}$ }，其中 $\mathrm { M } _ { i , j } \in \{ 0 , 1 \} ^ { D }$ 是一个 $D$ 维二进制向量。它指示了在给定文本描述 $\mathrm { t e x t } _ { i , j } .$ 时我们应该关注哪个身体部位。其次是时间多样性 T2M 任务（Temporally-diverse T2M task，T2M-T）。T2M-T 期望模型生成一个长动作序列，其中按特定顺序包含跨越不同时间区间的多种动作（例如“一个人在走路，然后跑步”）。第 $i$ 个测试样本是一个文本-时长对数组 $\{ \mathrm { t e x t } _ { i , j } , [ l _ { i , j } , r _ { i , j } ] \} , l _ { i , j } < r _ { i , j }$。这意味着从第 $l _ { i , j } \mathrm { - t h }$ 帧到第 $r _ { i , j }$ 帧的动作片段应该包含动作 tex $^ { ; } i , j ^ { \cdot }$。

实现细节。我们在 BABEL 数据集（Punnakkal et al., 2021）上对所提出的 MotionDiffuse 进行了 50K 次迭代训练。每个姿态状态由 $( r ^ { x } , r ^ { y } , r ^ { z } , \mathbf { j } ^ { r } )$ 表示，这与 UESTC 数据集的设置相同。其他设置保持不变。可视化中使用 $\lambda _ { 1 } = \lambda _ { 2 } = 0 . 0 1$。

定性结果。如图 Fig 5 所示，MotionDiffuse 具备处理高度综合性提示词的能力，这些提示词为多个身体部位分配动作（例如需要上下身协调的“踢击并出拳”以及“跳跃并举起手臂”）。此外，MotionDiffuse 能够根据包含多个动作的复杂指令生成长序列（例如包含一系列截然不同动作的“系鞋带、站起来然后向前走”）。

![](images/b6ded070c760e45eb3c4bdafb851187d5413daab63acd9057a82784ff218bff5.jpg)

![](images/e5757fbb65523a17eab251b023b95f52ee2f7ef0c47f1f0d921c8672cd02ba72.jpg)  
d) 系鞋带、站起来然后向前走

Fig. 5 BABEL 数据集上的定性结果。MotionDiffuse 能够根据涉及多个身体部位或动作的复杂提示词生成动态序列。

## 5 结论、局限性与未来工作

我们提出了 MotionDiffuse，这是首个基于扩散模型（diffusion model）的文本驱动动作生成方法。MotionDiffuse 展现出三大主要优势：增强多样性的概率映射（Probabilistic Mapping）、确保动作序列合理性的逼真合成（Realistic Synthesis），以及支持局部操控和长序列生成的多层次操控（Multi-Level Manipulation）。定量和定性评估均表明，MotionDiffuse 在文本驱动动作生成和动作条件驱动动作生成等多种任务上均优于现有技术，并展示了卓越的动作操控能力。

尽管 MotionDiffuse 推进了动作生成任务的性能边界，但仍存在一些问题。首先，扩散模型在推理过程中需要大量的扩散步骤，实时生成动作序列仍具有挑战性。其次，当前的流程仅接受单一形式的动作表征。一个能够同时适应所有数据集的更通用的流程将对各种场景更具通用性。

致谢 本研究由 NTU NAP、MOE AcRF Tier 2（T2EP20221-0033）以及 RIE2020 产业协调基金——产业合作项目（IAF-ICP）资助计划资助，并获得产业合作伙伴的现金和实物支持。通讯作者：Ziwei Liu (ziwei.liu@ntu.edu.sg)。

## 参考文献

Ahuja C, Morency LP (2019) Language2pose: Natural language grounded pose forecasting. In: 2019 International Conference on 3D Vision (3DV), IEEE, pp 719–728

Aliakbarian S, Saleh FS, Salzmann M, Petersson L, Gould S (2020) A stochastic conditioning scheme for diverse human motion prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 5223–5232

Andriluka M, Iqbal U, Insafutdinov E, Pishchulin L, Milan A, Gall J, Schiele B (2018) Posetrack: A benchmark for human pose estimation and tracking. In: Proceedings of the IEEE conference on computer vision and pattern recognition, pp 5167–5176

Barsoum E, Kender J, Liu Z (2018) Hp-gan: Probabilistic 3d human motion prediction via gan. In: Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pp 1418–1427

Bhattacharya U, Rewkowski N, Banerjee A, Guhan P, Bera A, Manocha D (2021) Text2gestures: A transformer-based network for generating emotive body gestures for virtual agents. In: 2021 IEEE Virtual Reality and 3D User Interfaces (VR), IEEE, pp 1–10

Cai Z, Zhang M, Ren J, Wei C, Ren D, Li J, Lin Z, Zhao H, Yi S, Yang L, et al. (2021) Playing for 3d human recovery. arXiv preprint arXiv:211007588

Cai Z, Ren D, Zeng A, Lin Z, Yu T, Wang W, Fan X, Gao Y, Yu Y, Pan L, et al. (2022) Humman: Multimodal 4d human dataset for versatile sensing and modeling. arXiv preprint arXiv:220413686

Cao Z, Gao H, Mangalam K, Cai QZ, Vo M, Malik J (2020) Long-term human motion prediction with scene context. In: European Conference on Computer Vision, Springer, pp 387–404

Carreira J, Noland E, Hillier C, Zisserman A (2019) A short note on the kinetics-700 human action dataset. arXiv preprint arXiv:190706987

Cervantes P, Sekikawa Y, Sato I, Shinoda K (2022) Implicit neural representations for variable length human motion generation. arXiv preprint arXiv:220313694

Chung J, Wuu Ch, Yang Hr, Tai YW, Tang CK (2021) Haa500: Human-centric atomic action dataset with curated videos. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 13465–13474

Dhariwal P, Nichol A (2021) Difusion models beat gans on image synthesis. Advances in Neural Information Processing Systems 34

Dinh L, Krueger D, Bengio Y (2014) Nice: Non-linear independent components estimation. arXiv preprint arXiv:14108516

Dinh L, Sohl-Dickstein J, Bengio S (2016) Density estimation using real nvp. arXiv preprint arXiv:160508803

Futrelle R, Speckert G (1978) Extraction of motion data by interactive processing. In: Proc. 1978 IEEE Conf. Pattern Recog. and Image Processing, Chicago, IL, pp 405–408

Gavrila DM (1999) The visual analysis of human movement: A survey. Computer vision and image understanding 73(1):82–98

Ghosh A, Cheema N, Oguz C, Theobalt C, Slusallek P (2021) Synthesis of compositional animations from textual descriptions. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 1396–1406

Goodfellow I, Pouget-Abadie J, Mirza M, Xu B, Warde-Farley D, Ozair S, Courville A, Bengio Y (2014) Generative adversarial nets. Advances in neural information processing systems 27

Gu C, Sun C, Ross DA, Vondrick C, Pantofaru C, Li Y, Vijayanarasimhan S, Toderici G, Ricco S, Sukthankar R, et al. (2018) Ava: A video dataset of spatio-temporally localized atomic visual actions. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp 6047–6056

Guo C, Zuo X, Wang S, Zou S, Sun Q, Deng A, Gong M, Cheng L (2020) Action2motion: Conditioned gen-

eration of 3d human motions. In: Proceedings of the 28th ACM International Conference on Multimedia, pp 2021–2029

Guo C, Zou S, Zuo X, Wang S, Ji W, Li X, Cheng L (2022) Generating diverse and natural 3d human motions from text. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 5152–5161

Harvey FG, Yurick M, Nowrouzezahrai D, Pal C (2020) Robust motion in-betweening. ACM Transactions on Graphics (TOG) 39(4):60–1

Hendrycks D, Gimpel K (2016) Gaussian error linear units (gelus). arXiv preprint arXiv:160608415

Henter GE, Alexanderson S, Beskow J (2020) Moglow: Probabilistic and controllable motion synthesis using normalising flows. ACM Transactions on Graphics (TOG) 39(6):1–14

Ho J, Jain A, Abbeel P (2020) Denoising difusion probabilistic models. Advances in Neural Information Processing Systems 33:6840–6851

Hochreiter S, Schmidhuber J (1997) Long short-term memory. Neural computation 9(8):1735–1780

Hong F, Zhang M, Pan L, Cai Z, Yang L, Liu Z (2022) Avatarclip: Zero-shot text-driven generation and animation of 3d avatars. arXiv preprint arXiv:220508535

Huang R, Hu H, Wu W, Sawada K, Zhang M, Jiang D (2020) Dance revolution: Long-term dance generation with music via curriculum learning. arXiv preprint arXiv:200606119

Ikemoto L, Arikan O, Forsyth D (2009) Generalizing motion edits with gaussian processes. ACM Transactions on Graphics (TOG) 28(1):1–12

Ionescu C, Papava D, Olaru V, Sminchisescu C (2013) Human3.6m: Large scale datasets and predictive methods for 3d human sensing in natural environments. IEEE transactions on pattern analysis and machine intelligence 36(7):1325–1339

Jain A, Tancik M, Abbeel P (2021) Putting nerf on a diet: Semantically consistent few-shot view synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 5885–5894

Jhuang H, Gall J, Zufi S, Schmid C, Black MJ (2013) Towards understanding action recognition. In: Proceedings of the IEEE international conference on computer vision, pp 3192–3199

Ji Y, Xu F, Yang Y, Shen F, Shen HT, Zheng WS (2018) A large-scale rgb-d database for arbitraryview human action recognition. In: Proceedings of the 26th ACM international Conference on Multimedia, pp 1510–1518

Joo H, Liu H, Tan L, Gui L, Nabbe B, Matthews I, Kanade T, Nobuhara S, Sheikh Y (2015) Panoptic studio: A massively multiview system for social mo-

tion capture. In: Proceedings of the IEEE International Conference on Computer Vision, pp 3334–3342

Karpathy A, Toderici G, Shetty S, Leung T, Sukthankar R, Fei-Fei L (2014) Large-scale video classification with convolutional neural networks. In: Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp 1725–1732

Kingma DP, Dhariwal P (2018) Glow: Generative flow with invertible 1x1 convolutions. Advances in neural information processing systems 31

Kingma DP, Welling M (2013) Auto-encoding variational bayes. arXiv preprint arXiv:13126114

Kocabas M, Athanasiou N, Black MJ (2020) Vibe: Video inference for human body pose and shape estimation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp 5253–5263

Kuehne H, Jhuang H, Garrote E, Poggio T, Serre T (2011) Hmdb: a large video database for human motion recognition. In: 2011 International conference on computer vision, IEEE, pp 2556–2563

Lee HY, Yang X, Liu MY, Wang TC, Lu YD, Yang MH, Kautz J (2019) Dancing to music. Advances in Neural Information Processing Systems 32

Li J, Yin Y, Chu H, Zhou Y, Wang T, Fidler S, Li H (2020) Learning to generate diverse dance motions with transformer. arXiv preprint arXiv:200808171

Li R, Yang S, Ross DA, Kanazawa A (2021) Ai choreographer: Music conditioned 3d dance generation with aist++. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 13401– 13412

Lin AS, Wu L, Corona R, Tai K, Huang Q, Mooney RJ (2018) Generating animated videos of human activities from natural language descriptions. In: NeurIPS Workshop

Liu J, Shahroudy A, Perez M, Wang G, Duan LY, Kot AC (2019) Ntu rgb+ d 120: A large-scale benchmark for 3d human activity understanding. IEEE transactions on pattern analysis and machine intelligence 42(10):2684–2701

Loper M, Mahmood N, Romero J, Pons-Moll G, Black MJ (2015) Smpl: A skinned multi-person linear model. ACM transactions on graphics (TOG) 34(6):1–16

Mahmood N, Ghorbani N, Troje NF, Pons-Moll G, Black MJ (2019) Amass: Archive of motion capture as surface shapes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 5442–5451

Mehta D, Rhodin H, Casas D, Fua P, Sotnychenko O, Xu W, Theobalt C (2017) Monocular 3d human pose estimation in the wild using improved cnn supervi-

sion. In: 2017 international conference on 3D vision (3DV), IEEE, pp 506–516

Mildenhall B, Srinivasan PP, Tancik M, Barron JT, Ramamoorthi R, Ng R (2020) Nerf: Representing scenes as neural radiance fields for view synthesis. In: European conference on computer vision, Springer, pp 405–421

Min J, Chai J (2012) Motion graphs++ a compact generative model for semantic motion analysis and synthesis. ACM Transactions on Graphics (TOG) 31(6):1–12

Monfort M, Andonian A, Zhou B, Ramakrishnan K, Bargal SA, Yan T, Brown L, Fan Q, Gutfreund D, Vondrick C, et al. (2019) Moments in time dataset: one million videos for event understanding. IEEE transactions on pattern analysis and machine intelligence 42(2):502–508

Mukai T, Kuriyama S (2005) Geostatistical motion interpolation. In: ACM SIGGRAPH 2005 Papers, pp 1062–1070

Nichol A, Dhariwal P, Ramesh A, Shyam P, Mishkin P, McGrew B, Sutskever I, Chen M (2021) Glide: Towards photorealistic image generation and editing with text-guided difusion models. arXiv preprint arXiv:211210741

Nichol AQ, Dhariwal P (2021) Improved denoising diffusion probabilistic models. In: International Conference on Machine Learning, PMLR, pp 8162–8171

Ormoneit D, Black MJ, Hastie T, Kjellstr¨om H (2005) Representing cyclic human motion using functional analysis. Image and Vision Computing 23(14):1264– 1276

O’rourke J, Badler NI (1980) Model-based image analysis of human motion using constraint propagation. IEEE Transactions on Pattern Analysis and Machine Intelligence (6):522–536

Patel P, Huang CHP, Tesch J, Hofmann DT, Tripathi S, Black MJ (2021) AGORA: Avatars in geography optimized for regression analysis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 13468–13478

Petrovich M, Black MJ, Varol G (2021) Actionconditioned 3d human motion synthesis with transformer vae. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 10985– 10995

Petrovich M, Black MJ, Varol G (2022) Temos: Generating diverse human motions from textual descriptions. arXiv preprint arXiv:220414109

Plappert M, Mandery C, Asfour T (2016) The kit motion-language dataset. Big data 4(4):236–252

Punnakkal AR, Chandrasekaran A, Athanasiou N, Quiros-Ramirez A, Black MJ (2021) Babel: Bodies,

action and behavior with english labels. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 722–731

Radford A, Kim JW, Hallacy C, Ramesh A, Goh G, Agarwal S, Sastry G, Askell A, Mishkin P, Clark J, et al. (2021) Learning transferable visual models from natural language supervision. arXiv preprint arXiv:210300020

Ramesh A, Dhariwal P, Nichol A, Chu C, Chen M (2022) Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:220406125

Rose C, Cohen MF, Bodenheimer B (1998) Verbs and adverbs: Multidimensional motion interpolation. IEEE Computer Graphics and Applications 18(5):32–40

Shao D, Zhao Y, Dai B, Lin D (2020) Finegym: A hierarchical video dataset for fine-grained action understanding. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 2616–2625

Shen Z, Zhang M, Zhao H, Yi S, Li H (2021) Eficient attention: Attention with linear complexities. In: Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp 3531–3539

Siyao L, Yu W, Gu T, Lin C, Wang Q, Qian C, Loy CC, Liu Z (2022) Bailando: 3d dance generation by actorcritic gpt with choreographic memory. arXiv preprint arXiv:220313055

Soomro K, Zamir AR, Shah M (2012) Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:12120402

Sun G, Wong Y, Cheng Z, Kankanhalli MS, Geng W, Li X (2020) Deepdance: music-to-dance motion choreography with adversarial learning. IEEE Transactions on Multimedia 23:497–509

Tevet G, Gordon B, Hertz A, Bermano AH, Cohen-Or D (2022) Motionclip: Exposing human motion generation to clip space. arXiv preprint arXiv:220308063

Trumble M, Gilbert A, Malleson C, Hilton A, Collomosse JP (2017) Total capture: 3d human pose estimation fusing video and inertial sensors. In: BMVC, vol 2, pp 1–13

Tulyakov S, Liu MY, Yang X, Kautz J (2018) Mocogan: Decomposing motion and content for video generation. In: Proceedings of the IEEE conference on computer vision and pattern recognition, pp 1526– 1535

Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser L, Polosukhin I (2017) Attention is all you need. Advances in neural information processing systems 30

Wang Z, Yu P, Zhao Y, Zhang R, Zhou Y, Yuan J, Chen C (2020) Learning diverse stochastic human-action generators by learning smooth latent transitions. In: Proceedings of the AAAI conference on artificial intelligence, vol 34, pp 12281–12288

Yan X, Rastogi A, Villegas R, Sunkavalli K, Shechtman E, Hadap S, Yumer E, Lee H (2018) Mt-vae: Learning motion transformations to generate multimodal human dynamics. In: Proceedings of the European conference on computer vision (ECCV), pp 265–281

Yu Z, Yoon JS, Lee I, Venkatesh P, Park J, Yu J, Park H (2020) Humbi：人体表情的大型多视图数据集. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp 2987–2997

Zhang M, Yu L, Zhang K, Du B, Zhan B, Chen S, Jiang X, Guo S, Zhao J, Wang Y, et al. (2020) 演员表达情感的运动学数据集. Scientific data 7(1):1–8

Zhao H, Torralba A, Torresani L, Yan Z (2019) Hacs：用于识别与时序定位的人体动作片段及分段数据集. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 8668–8678

Zhou Y, Barnes C, Lu J, Yang J, Li H (2019) 论神经网络中旋转表示的连续性. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 5745–5753

Zhuang W, Wang C, Xia S, Chai J, Wang Y (2020) Music2dance：用于音乐驱动舞蹈生成的 Dancenet. arXiv preprint arXiv:200203761

Zou S, Zuo X, Qian Y, Wang S, Xu C, Gong M, Cheng L (2020) 基于偏振图像的三维人体形状重建. In: European Conference on Computer Vision, Springer, pp 351–368