# InternVideo：通过生成式与判别式学习构建通用视频基座模型

Yi Wang<sup>1∗</sup>, Kunchang Li<sup>5,1∗</sup>, Yizhuo Li<sup>3,1∗</sup>, Yinan He<sup>1∗</sup>, Bingkun Huang<sup>2,1∗</sup>, Zhiyu Zhao<sup>2,1∗</sup>, Hongjie Zhang<sup>1∗</sup> Jilan Xu<sup>4,1</sup>, Yi Liu<sup>5,1</sup>, Zun Wang<sup>8,1</sup>, Sen Xing<sup>6,1</sup>, Guo Chen<sup>2,1</sup>, Junting Pan<sup>7,1</sup>, Jiashuo Yu<sup>1</sup> Yali Wang<sup>5,1∗</sup>, Limin Wang<sup>2,1∗</sup>, Yu Qiao<sup>1†</sup>

<sup>1</sup>Shanghai AI Laboratory, <sup>2</sup>Nanjing University, <sup>3</sup>The University of Hong Kong, <sup>4</sup>Fudan University <sup>5</sup>Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences <sup>6</sup>Tsinghua University, <sup>7</sup>The Chinese University of Hong Kong, <sup>8</sup>The Australia National University

https://github.com/OpenGVLab/InternVideo

![](images/b9eb57ddc6cc054c7dbf7d3acbde638d66a2e2ad93932012db2dedad6edd414c.jpg)

![](images/14b9de827ddd5013070cf6d7fb566d44df780a9cc651256fedab97e6925f80d8.jpg)

![](images/ecf8fd58f9baa04ed0786e1dbe94a74322fd9aec6d9329c00997c3b894f61a0e.jpg)

![](images/054604e2ce2514d381c5fe81d47ff7d93ffd6113c6cbe8ed89760805e30ec601.jpg)  
Figure 1: 与最先进的方法（包括专用模型 [1–5] 和基座模型 [6–9]）相比，InternVideo 在广泛的视频相关任务中取得了最佳性能。对比细节见第 4.3 节。v2t 和 t2v 分别代表视频到文本和文本到视频检索。STA、FHP、NLQ、SCOD 和 MQ 分别是短期物体交互预测（Short-term Object Interaction Anticipation）、未来手部预测（Future Hand Prediction）、自然语言查询（Natural Language Queries）、状态改变物体检测（State Change Object Detection）以及时刻查询（Moment Queries）的缩写。

## 摘要

基础模型（foundation models）最近在计算机视觉的各种下游任务中展现出了优异的性能。然而，大多数现有的视觉基础模型仅仅专注于图像级别的预训练和适配，这对于动态且复杂的视频级别理解任务而言存在局限性。为了填补这一空白，我们结合生成式和判别式自监督视频学习的优势，提出了通用视频基础模型 InternVideo。具体而言，InternVideo 高效地探索了掩码视频建模（masked video modeling）和视频-语言对比学习作为预训练目标，并以可学习的方式选择性地协调这两种互补框架的视频表征，以提升各种视频应用的能力。无需繁杂的设计，InternVideo 在涵盖视频动作识别/检测、视频-语言对齐以及开放世界视频应用等广泛任务的 39 个视频数据集上取得了最先进的性能。特别是，我们的方法在具有挑战性的 Kinetics-400 和 Something-Something V2 基准测试上分别获得了 91.1% 和 77.2% 的 top-1 准确率。所有这些结果都有效证明了 InternVideo 在视频理解方面的通用性。代码将发布于 https://github.com/OpenGVLab/InternVideo。

## 1 引言

基础模型在研究界受到了越来越多的关注 [10–12]，因为它们提供了一种扩展到众多感知任务并取得令人瞩目效果的实用范式。通过简单的适配或零样本/少样本学习，基础模型利用从网络规模数据中学习到的通用表征以及高容量的强大骨干网络，极大地降低了下游的设计和训练成本。人们期望通过开发基础模型能够从感知中培养认知，从而获得通用视觉能力。

尽管已有一系列视觉基础模型（vision foundation models）被提出 [7, 13–21]，但与图像任务相比，视频理解及其相关任务的探索相对较少，主要用于验证这些模型所提取的视觉特征同样有利于时空表征。我们推测学术界关注相对较少的原因在于：1）视频处理带来的高昂计算负担；2）相当一部分现有的视频基准可以通过利用图像主干网络（image backbones）的外观特征并结合相应的时间建模来解决。具体而言，就效率而言，当空间分辨率接近且时间采样率通常为 16 时，视频处理中额外的时间维度所带来的复杂度至少比图像处理高出一个数量级。对于目前的一些视频数据集，仅凭图像特征或辅以侧向时间模块就足以取得不错的效果，尤其是随着多模态模型 CLIP [13] 的兴起。其各种时间变体在若干核心任务中取得了具有竞争力或领先（state-of-the-art）的性能 [5, 22]。鉴于此，同时进行时空学习的模型在研发成本与回报之间似乎并非一个理想的平衡点。

此外，考虑到视频应用的广泛性，当前视觉基础模型的迁移能力仍显得较为有限。这些模型 [6, 8, 23, 24] 要么专注于动作理解任务（例如动作识别、时空动作定位等），要么专注于视频-语言对齐任务（例如视频检索、视频问答等）。我们认为这是由它们的学习方案以及缺乏用于衡量视频理解能力的综合基准所导致的。因此，这些工作 [6, 8, 23, 24] 集中于少数特定任务来展示其时空感知能力。学术界迫切需要一个能够赋能更广泛应用领域的通用基础模型。

在本文中，我们通过一种高性价比且通用的模型 InternVideo 推进了视频基础模型的研究。为了建立一种可行且有效的时空表征，我们研究了当前流行的视频掩码建模 [23, 25] 和多模态对比学习 [13, 26]。需要注意的是，视频掩码建模擅长动作理解，但由于当前解码器的限制导致其模型规模受限，这一方向仍值得进一步探索。对于多模态对比学习，它将丰富的语义嵌入到视频表征中，但忽略了具体的时空建模。为了应对这些挑战，我们通过模块化设计使这两种自监督方法能够高效地进行大规模学习。为了显著拓宽当前视频基础模型的泛化能力，我们提出了一种结合这两种自监督训练方式的统一表征学习方法。为了验证这种通用表征，我们提出了一个系统的视频理解基准。它包含了对动作理解、视频-语言对齐以及开放世界视频应用的评估，我们认为这是通用视频感知的三个核心能力。我们没有向该系统引入新的数据或标注，而是初步选择了涵盖 39 个公开数据集的 10 个代表性视频任务，并将它们归入上述三类。据我们所知，InternVideo 是首个在所有这三类不同视频任务中均展现出极具前景的迁移能力以及最先进性能的视频基础模型。

在 InternVideo 中，我们设计了一种统一视频表征（Unified Video Representation, UVR）学习范式。它同时探索了基于自编码器的掩码视频建模（MAE）和多模态对比学习以获得两种类型的表征，通过监督动作分类对它们进行增强，并基于两者之间的跨表征学习生成更通用的表征。UVR 不仅在经验上证明了具备时间捕捉能力的视频表征在核心视频任务上显著优于图像表征，而且具有很高的训练效率。其中的 MAE 利用了视频中的高度冗余性，仅使用少量的可见 token 进行训练。同时，InternVideo 中的多模态学习扩展了现有的图像预训练骨干网络，用于视频对比训练。在对这两个视频编码器进行监督训练之后，我们精心设计了跨模型注意力机制，在这两个几乎冻结的编码器之间进行特征对齐。

除了提出统一的视频表征学习范式外，我们还为以易于处理且高效的方式训练大规模视频基础模型（video foundation models）提供了实践方案与指导准则。我们的工作包括但不限于：1）使 VideoMAE 具备可扩展性，并探索其在模型与数据规模上的可扩展性；2）关于如何利用现有图像预训练骨干网络的高效且有效的多模态架构设计与训练方案；3）通过实验发现来自 VideoMAE 和多模态模型的特征具有互补性，并研究如何通过协同不同的现有模型来推导出更强大的视频表征。具体而言：

• 在 VideoMAE 的可扩展性研究方面，我们表明训练视频中适当的多样性以及规模扩充能够提高所使用视频编码器的可扩展性。在掩码自编码器训练设置下采用新的预训练数据集，通过微调，ViT 在 Kinetics-400 [27] 上的动作识别性能在模型配置从 base 增加到 large 时从 81.01% 提升至 85.35%，并在 huge 配置下进一步达到 86.9%，以显著优势超越了 [23] 中报告的性能。VideoMAE 的可扩展性使其能够应用于视频基础模型的开发。

• 在复用现有基础模型进行多模态学习方面，我们将图像预训练的视觉 Transformer [28] 扩展至视频表征学习。这种迁移学习需要大量的结构与优化定制，或是使用局部和全局时空模块进行多模态预训练。局部模块通过连续且独立的空间和时间注意力计算来解耦时空建模；与此同时，全局模块计算跨越空间和时间的 token 交互。实验表明，这种复用设计在时空表征学习中十分有效。

• 除了自监督预训练之外，我们还采用有监督动作识别来进一步增强视频表征。结果表明，动作识别是一个适合迁移至各类下游应用的良好源任务。

• 为了协调基础模型，我们通过跨表征学习将掩码视频编码器与多模态编码器统一起来，而不是在单一框架下联合训练它们。鉴于 MAE 与多模态学习（MML）的优化目标可能相互冲突，如何在不损害各自优势的情况下将它们结合仍是一个开放性问题 [29]。更重要的是，基于对比学习的 MML 需要巨大的批大小（batch size）以实现更好的对比优化。在此基础上引入 MAE 势必会导致诸多棘手的工程实现问题。考虑到它们之间潜在的训练冲突，我们分别对 MAE 和 MML 进行独立训练。在它们训练收敛后，我们再通过提出的跨模型注意力（CMA）模块动态融合它们的表征。该模块在 MAE 和 MML 的中层特征之间实现交叉注意力，并自适应地融合它们的高层特征以进行预测。在模型级表征交互阶段，我们冻结分别由 MAE 和 MML 训练的骨干网络（backbones），仅在监督学习下用极少数轮次（epochs）更新 CMA。实验表明，这是一种在计算上可行且高效地利用 MAE 和 MML 特征的方法。

我们在 10 项任务的 39 个数据集上（包括动作识别、时空动作定位、视频问答、视频检索等核心任务）验证了我们提出的视频基础模型，且在各项任务中均显著超越了所有当前最先进的方法（state-of-the-art）。我们认为，本方法取得的这些全面领先的结果以及相关观察与分析，为视频理解领域树立了新的基准（baseline）。本文提供的实证依据增强了人们的信心，表明视频感知任务以及部分高阶任务（形式化为感知形式）能够被视频基础模型很好地解决，从而在一系列应用中作为决定性能的关键方法发挥作用。

总结而言，我们在视频基础模型方面的贡献如下：

• 我们探索了一种同时结合掩码建模和对比建模的通用视频表征范式，并通过在监督下进行轻量级模型交互学习来统合它们的表征，从而实现了这一设计。我们在实验中证实，生成式训练和对比式训练所学习到的特征是互补的，并且能够取得优于两者独立训练的效果。

• 我们发现，经过适当调优，掩码视频编码器在模型和数据规模上均具备可扩展性。我们设计了可插拔的局部时序和全局时空交互模块，以复用基于图文数据预训练的 ViT 进行多模态学习，从而减轻了训练负担并带来了更优的下游性能。

• 我们在构建系统性视频理解基准方面进行了初步尝试。我们的通用视频基础模型在涵盖该基准中多个核心任务的 39 个数据集上取得了最先进的性能（state-of-the-art），例如动作识别中的 Kinetics-400 和 Something-Something v2。我们通过实证发现，我们所学习到的视频表征优于竞争对手，在视觉-语言任务中大幅领先，尤其是在某些基于图像的任务上。这表明通用视频表征将在视频任务中扮演核心角色。我们相信，所提出方法和模型的开源开放将为研究界提供获取基础模型及其特征的便捷工具。

## 2 相关工作

图像基础模型（Image Foundation Models）。目前大多数视觉模型仅适用于特定的任务和领域，并且需要人工标注的数据集进行训练。针对这一问题，近期的工作提出了视觉基础模型。CLIP [13] 和 ALIGN [14] 准备了网络规模的带噪声图像-文本对，利用对比学习训练双编码器模型，从而获得了用于强大零样本迁移的鲁棒图像-文本表征。INTERN [12] 将自监督预训练扩展到多个学习阶段，利用了大量的图像-文本对以及人工标注的图像。与 CLIP 相比，INTERN 取得了更好的线性探测（linear probe）性能，并提升了下游图像任务中的数据效率。Florence [15] 通过统一对比学习 [16] 和精心设计的适配模型对它们进行了扩展，在不同的迁移设置下支持了广泛的视觉任务。SimVLM [17] 和 OFA [18] 利用生成式目标训练编码器-解码器模型，并在多项多模态任务上展现出具有竞争力的性能。此外，CoCa [7] 统一了类似 CLIP 的对比学习和类似 SimVLM 的生成式学习。最近，BeiT-3 [19] 引入了多路 Transformer（Multiway Transformers）与统一的 BeiT [20] 预训练，在多项视觉和图像-语言任务上取得了最先进的迁移结果。

视频基座模型（Video Foundation Models）。先前的图像基座模型 [7, 15] 仅在视频识别任务上表现出良好性能（尤其是在 Kinetics 上）。对于视频多模态任务，VIOLET [30] 结合了掩码语言建模和掩码视频建模，All-in-one [24] 提出了基于共享骨干网络的统一视频-语言预训练，而 LAVENDER [31] 则将各项任务统一为掩码语言建模。尽管它们在多模态基准测试中表现良好，但它们是在有限的视频-文本数据上进行训练的，在纯视频任务（例如动作识别）上表现较为吃力。相比之下，MERLOT Reserve [32] 收集了 20M 视频-文本-音频对，通过对比跨度匹配训练联合视频表征，从而在视频识别和视觉常识推理任务上取得了最先进的性能。与图像基座模型相比，当前的视频基座模型所支持的视频和视频-语言任务有限，尤其是对于时序定位等细粒度时序辨别任务。

自监督预训练（Self-supervised Pretraining）。自监督学习近年来发展迅速。它专注于为预训练设计不同的代理任务 [33–37]，主要可分为对比学习和掩码建模。对比学习采用多种数据增强来生成图像的不同视图，然后拉近正样本对并推开负样本对。为了维持足够且富含信息的负样本，以往的方法依赖于大型记忆库（memory banks）或大批次大小 [38–40]。BYOL [41] 和 SimSiam [42] 消除了对负样本的需求，设计了精巧的技术来避免模型崩溃。至于掩码建模，它通过基于可见上下文的掩码预测来学习丰富的视觉表征。iGPT [43] 首次提出了掩码图像建模（MIM）。BeiT [20] 提出利用预训练的分词器进行视觉标记预测 [44]，MaskFeat [6] 预测手工设计的图像描述符，而 MAE [25] 则直接重建原始像素。在时空表征学习方面，VideoMAE [23] 和 BEVT [45] 分别将 MAE 和 BeiT 拓展到了时空领域。

多模态预训练（Multimodal Pretraining）。自图像-文本预训练发展以来，大规模视频-文本预训练结合特定下游任务微调已成为视频-语言领域的标准范式 [26, 30, 32, 46–51]。开创性方法 [52, 53] 使用预训练的视觉和语言编码器来提取离线视频和文本特征，而近期的方法 [9, 24, 26, 46, 54, 55] 证明了端到端训练的可行性。此外，主流方法通常包含两到三个预训练任务，例如掩码语言建模 [31]、视频-文本匹配 [24]、视频-文本对比学习 [47] 和视频-文本掩码建模 [30]。

## 3 InternVideo

InternVideo 是一个通用的视频基座模型（foundation model），其训练和内部协作如图 Figure 2 所示。在结构上，InternVideo 采用了视觉 Transformer（Vision Transformer, ViT）[28] 及其变体 UniformerV2 [56]，并结合额外的局部时空建模模块以实现多层次的表征交互。在学习机制上，InternVideo 逐步提升其表征能力，融合了自监督学习（掩码建模和多模态学习）与监督训练。此外，由于我们探索了两种类型的自监督学习，我们进一步整合了它们的优势：InternVideo 通过可学习的交互动态地从这两个 Transformer 中派生出新特征，兼取了生成式和对比式预训练的各自所长。通过新聚合的特征，InternVideo 在 10 个主流视频任务的 34 个基准测试中刷新了性能纪录，并在近期的 Ego4D 竞赛 [57] 中斩获了五个赛道的冠军。

![](images/d2456d9063e7aa6c27e77a29ba12c16047b381bf50c10225abecd9a32d775a04.jpg)  
Figure 2: InternVideo 的整体框架。

## 3.1 自监督视频预训练

InternVideo 在无监督的情况下同时进行掩码和对比训练以进行表征学习。根据 [13, 23]，视频掩码建模生成的特征在动作判别方面表现优异，例如动作识别和时序动作定位；而视频-语言对比学习能够在没有标注的情况下借助文本语义理解视频。我们采用了两个具有不同结构的 Transformer，以便更好地利用这两个优化目标。最终的表征是通过自适应聚合这两种类型的表征来构建的。

## 3.1.1 视频掩码建模

我们遵循所提出的 VideoMAE [23] 工作中的大部分惯例，训练一个原生视觉 Transformer（Vision Transformer, ViT）作为用于时空建模的视频编码器，如图 Figure 3 (a) 所示。VideoMAE 采用非对称编码器-解码器架构，在极高掩码率的视频输入下执行视频重建任务。所使用的编码器和解码器均为 ViT。解码器的通道数是编码器的一半，默认包含 4 个块（blocks）。具体而言，我们将经过时间步长下采样的视频输入划分为不重叠的 3D patch（视频块），并将其线性投影为立方体嵌入（cube embeddings）。然后，我们对这些嵌入应用极高比例（例如 90%）的管道掩码（tube masking），并将它们输入到非对称编码器-解码器架构中以执行掩码视频建模预训练。为了在全局范围内表征时空交互，我们在 ViT 中采用了联合时空注意力（joint space-time attention）[58, 59]，使所有可见 token 彼此之间进行全局交互。由于仅保留了少量 token 用于计算，这在计算上是完全可行的。

## 3.1.2 视频-语言对比学习

我们同时进行视频/图像-文本对比学习和视频描述（video captioning）任务来进行预训练，如图 3 (b) 所示。为了提高训练效率，我们基于预训练的 CLIP [13] 构建多模态结构。我们没有直接采用原生的 ViT，而是使用我们提出的 UniFormerV2 [56] 作为视频编码器，以实现更好且更高效的时序建模。此外，我们采用了一个额外的 Transformer 解码器进行跨模态学习。具体而言，我们遵循如 [7, 60] 中给出的典型的先对齐后融合（align-before-fuse）范式。首先，分别对视频和文本进行编码。然后利用对比损失来对齐视频和文本特征的嵌入空间。在融合阶段，我们应用一个描述解码器（caption decoder）作为跨模态融合器，它使用交叉注意力进行描述前置任务（captioning pretext）。这种先对齐后融合的范式不仅确保了不同模态能够对齐到同一个嵌入空间中（这对检索等任务有利），还赋予了模型组合不同模态的能力，从而对问答等任务大有裨益。描述解码器的引入既扩展了原始 CLIP 的潜力，又提高了多模态特征的鲁棒性。

![](images/95d1b61bb47d8985e33953ea00ef56a8ff82276b9ffdbae425f61f604149098b.jpg)  
Figure 3: 预训练阶段掩码学习和多模态学习的整体框架。

![](images/77f58e4e70355000f8113146141311240bf7e1183476388bffd43afa035cd16f.jpg)  
Figure 4: 使用跨模型注意力机制的模型交互图示。

## 3.2 监督式视频后预训练

从经验上看，动作识别在视频下游应用中表现出良好的元任务特性，这在 [61, 62] 中得到了广泛验证。因此，我们分别使用有监督动作分类对掩码视频编码器和多模态编码器进行训练，以此作为后预训练（post-pretraining）步骤，以在多样化任务中获得更好的性能。为了提升这些编码器的学习能力，我们提出了一个统一的视频基准 Kinetics-710（K710，在第 4.1 节中描述），用于微调我们的视频编码器。

掩码视频编码器（Masked Video Encoder）。我们在 K710 上使用 32 块 GPU 微调掩码视频编码器。我们根据基础学习率和批次大小线性调整学习率，lr = base learning rate × <sup>batch</sup> <sup>size</sup> 。我们采用 DeepSpeed<sup>1</sup> 256 框架以节省显存占用并加速训练。我们将基础学习率设置为 0.001，drop path 率设置为 0.2，head dropout 率设置为 0.5，重复采样（repeated sampling）[63] 设置为 2，层衰减（layer decay）设置为 0.8，共训练 40 个 epoch。

多模态视频编码器。我们遵循 UniFormer [64] 中的大部分训练配方。为了获得最佳结果，鉴于 CLIP-ViT [13] 通过视觉-语言对比学习预训练出的鲁棒表征，我们默认采用其作为骨干网络（backbone）。我们在 ViT-B/L 的最后 4 层插入全局 UniBlocks 以执行多阶段融合。我们将基础学习率设置为 1e − 5，重复采样次数设为 1，批次大小（batch size）设为 512，并训练 40 个 epoch。对于所有数据集，我们均采用分辨率为 224 的稀疏采样（sparse sampling）[65]。在后预训练（post-pretraining）阶段，我们使用 UniformerV2 [56] 作为视觉编码器，并以使输出与原始 CLIP 模型相同的方式初始化附加参数，我们发现这对于获得良好的零样本（zero-shot）性能至关重要。视频字幕生成模块是一个标准的 6 层 Transformer 解码器（$c = 768$），后接一个双层 MLP。其他设置保持 CLIP Large/14 不变。

Table 1: InternVideo 预训练过程中使用的数据集汇总。大规模数据库对通用视觉预训练至关重要。我们的预训练数据由来自 5 个不同领域的 1200 万个视频片段组成。

<table><tr><td>预训练数据集</td><td>领域</td><td>样本片段数</td><td>帧数 × 采样率</td></tr><tr><td>Kinetics-400 [27]</td><td>Youtube 视频</td><td>240k</td><td>16 × 4</td></tr><tr><td>WebVid2M [55]</td><td>网络视频</td><td>250k</td><td>16 × 4</td></tr><tr><td>WebVid10M [55]</td><td>网络视频</td><td>10M</td><td>16 × 4</td></tr><tr><td>HowTo100M [66]</td><td>Youtube 视频</td><td>1.2M</td><td>16 × 4</td></tr><tr><td>AVA [67]</td><td>电影</td><td>21k</td><td>16 × 4</td></tr><tr><td>Something-Something V2 [68]</td><td>依脚本拍摄</td><td>169k</td><td>16 × 2</td></tr><tr><td>自收集视频</td><td>Youtube, Instagram</td><td>250k</td><td>16 × 4</td></tr><tr><td>Kinetics-710 [56]</td><td>Youtube 视频</td><td>680k</td><td>16 × 4</td></tr></table>

## 3.3 跨模型交互

为了基于视频掩码建模和视频-语言对比学习来学习统一的视频表征，我们通过添加跨模型注意力模块来进行跨表征学习，如图 Figure 4 所示。

考虑到同时优化两个模型计算开销巨大，除分类层和多模态视频编码器中的查询词元（query token）外，我们冻结了两个骨干网络，仅更新新添加的组件。我们添加了一些精心设计的可学习模块（跨模型注意力，cross-model attention）用于对齐通过不同方法学习到的表征。跨模型注意力（CMA）由标准多头跨注意力（MHCA）以及前馈网络（FFN）组成。它将来自多模态视频编码器的中间词元用作键（keys）和值（values），同时将来自掩码视频编码器的词元用作查询（queries）。从 CMA 计算得到的新词元被视为与来自多模态视频编码器的表征逐步对齐的表征。该过程主要将多模态知识迁移至掩码视频编码器中的 CMA。一个设计上的例外是，对于最后一个 CMA 模块，其键和值来自掩码视频编码器的词元，而查询来自多模态视频编码器的类别词元（class token）。因此，类别词元根据来自掩码编码器的词元进行更新。它将单模态知识迁移至多模态视频编码器中的 CMA。从这个角度来看，在动作识别的监督下，掩码视频编码器所有阶段的特征与多模态视频编码器最终阶段的特征均得到增强，从而彼此相互协同。最后，我们利用可学习的线性组合动态融合这两个预测得分。

## 4 实验

我们首先详细介绍实验配置（第 4.1 节），然后在第 4.3 节中展示 Intern-Video 在所提出的包含三类任务（动作理解、视频-语言对齐和开放理解）的视频理解基准上的下游性能。

## 4.1 预训练数据

通用视频基础模型预训练需要来自不同领域的大规模数据。为实现具有多样性的数据分布，我们采用了 6 个公开数据集以及我们自收集的视频片段，如 Table 1 所示。

Kinetics-710。我们采用了一个新的定制 kinetics 动作数据集 Kinetics-710 [56] 进行监督训练（包括独立训练和联合训练）。它包含 650K 个视频，涵盖 710 个独特的动作标签。它融合了来自 Kinetics 400/600/700 [27, 69, 70] 的所有不重复的训练数据。为了避免训练数据泄露，特定版本 Kinetics 测试集中已存在的某些训练数据被舍弃。

UnlabeledHybrid。UnlabeledHybrid 数据集用于掩码视频预训练，由 Kinetics-710 [56]、Something-Something V2 [68]、AVA [67]、WebVid2M [55] 以及我们自收集的视频组成。对于 AVA，我们按 300 帧裁剪 15 分钟的训练视频，获得了 21k 个视频片段。我们分别从自收集视频和 WebVid2M 中随机抽取了 250k 个视频。更多细节可见 Table. 1。

Table 2: Kinetics 与 Something-Something 上的动作识别结果。我们报告了各对比方法在每个数据集上的 Top-1 准确率。InternVideo-D 表示由掩码视频编码器 ViT-H 与 CLIP 预训练的 UniFormerV2-L 进行模型集成构建而成，而 InternVideo-T 则表示基于 InternVideo-D 与多模态预训练的 UniFormerV2-L 计算得出。

<table><tr><td>方法</td><td>参数量</td><td>K400</td><td>K600</td><td>K700</td></tr><tr><td>MaskFeat-L [6]</td><td>218M</td><td>87.0</td><td>88.3</td><td>80.4</td></tr><tr><td>CoCa [7]</td><td>1B+</td><td>88.9</td><td>89.4</td><td>82.7</td></tr><tr><td>MTV-H [8]</td><td>1B+</td><td>89.9</td><td>90.3</td><td>83.4</td></tr><tr><td>MerlotReserve-L [9]</td><td>644M</td><td>-</td><td>90.1</td><td>-</td></tr><tr><td>MerlotReserve-L (+Audio) [9]</td><td>644M</td><td>-</td><td>91.1</td><td>-</td></tr><tr><td>InternVideo-D</td><td>1.0B</td><td>90.9</td><td>91.1</td><td>83.8</td></tr><tr><td>InternVideo-T</td><td>1.3B</td><td> $\mathbf{91.1}_{(+1.2)}$ </td><td> $\mathbf{91.3}_{(+1.0)}$ </td><td> $\mathbf{84.0}_{(+0.6)}$ </td></tr></table>

Table 3: Something-Something、ActivityNet、HACS 与 HMDB51 上的动作识别结果。我们报告了各对比方法在每个数据集上的 Top-1 准确率。

<table><tr><td>方法</td><td>SthSthV1</td><td>SthSthV2</td><td>ActivityNet</td><td>HACS</td><td>HMDB51</td></tr><tr><td>Previous SOTA</td><td>60.9 [73]</td><td>75.6 [6]</td><td>90.2 [1]</td><td>91.9 [58]</td><td>87.6 [3]</td></tr><tr><td>InternVideo</td><td> $\mathbf{70.0}_{(+9.1)}$ </td><td> $\mathbf{77.2}_{(+1.6)}$ </td><td> $\mathbf{94.3}_{(+4.1)}$ </td><td> $\mathbf{95.5}_{(+3.6)}$ </td><td> $\mathbf{89.3}_{(+1.7)}$ </td></tr></table>

## 4.2 实现细节

## 4.2.1 多模态训练

借助来自 CLIP 的初始化，我们使用 WebVid2M、WebVid10M 和 HowTo100M 对多模态模型进行后预训练。由于视频-文本数据集的训练语料不如 CLIP-400M [13] 丰富，我们使用图像-文本数据集（LAION-400M [71] 的一个包含 100M 图像-文本对的子集）与视频模型进行联合训练。我们在每次迭代中交替使用图像和视频。视频-文本的批大小为 14,336，图像-文本的批大小为 86,016。我们在 128 块 NVIDIA A100 GPU 上训练了 400k 步，耗时 2 周，学习率为 $8 \times 1 0 ^ { - 5 }$，权重衰减为 0.2，采用余弦退火调度策略，并包含 4k 步的预热步骤。

## 4.2.2 掩码视频训练

我们在 UnlabeledHybrid 数据集上使用 64 块 80G-A100 GPU 对 VideoMAE-Huge 进行了 1200 个 epoch 的训练。该模型采用余弦退火学习率调度策略，并预热总 epoch 的 10%。学习率设置为 $2 . 5 e \mathrm { ~ - ~ } 4$。数据增强仅使用 MultiScaleCrop。

## 4.2.3 模型交互

如图 4 所示，除了分类层和多模态视频编码器中的查询标记（query token）外，我们冻结了两个骨干网络（backbone）。为了保持原始输出，我们像 Flamingo [72] 一样在额外的 MHCA 和 FFN 中添加了 tanh 门控层，并将动态加权求和中的参数初始化为零。我们使用 64 的批次大小（batch size）、$5 \times 1 0 ^ { 5 } .$ 的学习率、0.001 的权重衰减（weight decay）、0.9 的 dropout 率以及 0.9999 的 EMA 率来训练协同模型。此外，我们使用了包含 1 个预热轮次（warmup epoch）的 5 轮余弦退火调度（cosine annealing schedule）。所使用的所有数据增强方法均与 UniFormerV2 [56] 相同。

## 4.3 下游任务

我们在多项下游任务上进行了广泛的实验以评估 InternVideo。所采用的任务分为三类，分别涵盖动作理解、视频-语言对齐以及开放式理解。由于 InternVideo 包含了专精于时空变化表征的掩码视频编码器（masked video encoder）和融合多模态视频编码器，因此它能够显著提升动作理解（第 4.3.1 节）和视频-语言对齐（第 4.3.2 节）任务的性能。大规模训练数据带来的泛化能力也赋予了它在相关任务上令人瞩目的零样本（zero-shot）和开放集（open-set）能力（第 4.3.3 节）。即使迁移到以自我为中心（ego-centric）的任务中，InternVideo 配合简单的预测头（head）仍能取得压倒性的优异性能 [57]。详细内容如下。

表 4：在 THUMOS-14、ActivityNet-v1.3、HACS 和 FineAction 上的时序动作定位结果。我们报告了各对比方法在每个数据集上的平均 mAP。

<table><tr><td>骨干网络</td><td>预测头</td><td>THUMOS-14</td><td>ActivityNet-v1.3</td><td>HACS</td><td>FineAction</td></tr><tr><td>I3D [27]</td><td>ActionFormer [4]</td><td>66.80</td><td>-</td><td>-</td><td>13.24</td></tr><tr><td>SlowFast [74]</td><td>TCANet [75]</td><td>-</td><td>-</td><td>38.71</td><td>-</td></tr><tr><td>TSP [76]</td><td>ActionFormer [4]</td><td>-</td><td>36.60</td><td>-</td><td>-</td></tr><tr><td>InternVideo</td><td>ActionFormer [4]</td><td> $71.58_{(+4.78)}$ </td><td> $39.00_{(+2.40)}$ </td><td>41.32</td><td> $17.57_{(+4.33)}$ </td></tr><tr><td>InternVideo</td><td>TCANet [75]</td><td>-</td><td>-</td><td> $41.55_{(+2.84)}$ </td><td>-</td></tr></table>

表 5：在 AVA2.2 和 AVA-Kinetics (AK) 上的时空动作定位结果。我们报告了所评估方法在各数据集上的 mAP。

<table><tr><td>方法</td><td>预测头</td><td>AVA2.2</td><td>AVA-Kinetics</td></tr><tr><td>ACAR [77]（集成）</td><td>ACAR [77]</td><td>33.30</td><td>40.49</td></tr><tr><td>RM [78]（集成）</td><td>RM [78]</td><td>-</td><td>40.97</td></tr><tr><td>MaskFeat [6]</td><td>-</td><td>38.80</td><td>-</td></tr><tr><td>InternVideo</td><td>Linear</td><td> $\mathbf{41.01}_{(+2.21)}$ </td><td> $\mathbf{42.51}_{(+1.54)}$ </td></tr></table>

## 4.3.1 动作理解任务

动作识别（Action Recognition）。动作源于时空模式。InternVideo 旨在学习合适时空特征的表征以及动态模式的建模。我们在 8 个动作识别基准上对 InternVideo 进行了评估，包括常用的 Kinetics 和 Something-Something。

我们在 Kinetics-400 [27]、Kinetics-600 [69]、Kinetics-700 [70]、Something-in-Something-V1 [68]、Something-in-Something-V2 [68]、ActivityNet [79]、HACS [80] 和 HMDB51 [81] 上对 InternVideo 中的 VideoMAE 和 UniFormerV2 进行了评估。我们使用 Top-1 准确率作为对比指标。在 Table 2 和 3 中，InternVideo 在所有这些动作识别基准上均展现出极具前景的性能。我们的 InternVideo 在几乎所有基准上都显著超越了先前的 SOTA 方法，并在 ActivityNet 上达到了 SOTA 结果。额外融合模型带来的准确率提升（InternVideo-D 对比 InternVideo-T）表明，探索广泛的技术路线是必要的，因为不同路线在性能上能够互相促进。

时序动作定位（Temporal Action Localization）。该任务（TAL）旨在从包含完全观测的整段未修剪视频中定位动作片段的起始点和终止点。我们在四个经典的 TAL 数据集上评估了 InternVideo：THUMOS-14 [82]、ActivityNet-v1.3 [79]、HACS Segment [80] 和 FineAction [83]。与先前的时序动作定位任务一致，我们使用平均精度均值（mAP）进行定量评估。每个动作类别都会计算平均精度（AP），以评估针对各个动作类别的候选提议（proposals）。AP 是在不同的 tIoU 阈值下计算的。我们报告了开源的最先进 TAL 方法的性能，包括用于 THUMOS-14、ActivityNet-v1.3 和 FineAction 的 ActionFormer [4] 方法，以及用于 HACS Segment 的 TCANet [75] 方法。

我们使用 InternVideo 中的 ViT-H 作为特征提取的主干网络。在实验中，ViT-H 模型是在 Hybrid 数据集上进行预训练的。如 Table 4 所示，我们的 InternVideo 在这四个 TAL 数据集上的表现优于以往的所有方法。值得注意的是，InternVideo 在时序动作定位方面取得了巨大提升，特别是在诸如 THUMOS-14 和 FineAction 等细粒度 TAL 数据集上。

时空动作定位（Spatiotemporal Action Localization）。该任务（STAL）旨在预测视频关键帧中人物的边界框及其对应动作。我们在两个经典的 STAL 数据集 AVA2.2 [67] 和 AVA-Kinetics [84] 上对 InternVideo 进行了评估。在 AVA2.2 [67] 中，每个视频时长 15 分钟，每秒给出一个关键帧。标注仅针对关键帧而非所有帧提供。在此，我们采用经典的两阶段方法来处理此任务。我们应用在 MS-COCO [85] 上充分训练的 Mask-RCNN [86] 来检测每个关键帧中的人体，关键帧检测框由 Alphaction [87] 项目提供。在第二阶段中，以关键帧为中心提取一定数量的帧，并输入到我们的视频骨干网络中。类似地，在训练中，我们使用真实标注框（ground truth box）进行训练 [87]，在测试中使用第一阶段预测的框。

我们在 InternVideo 中使用 ViT-Huge 进行了实验。具体结果见 Table 5。分类头采用简单的线性头，在两个数据集上均取得了 SOTA（当前最佳）性能。需要注意的是，使用 ViT-H 模型并在 AVA-Kinetics 数据集上进行训练，不仅提高了整体 mAP，而且显著提高了仅在 AVA 上测试获得的 mAP。这表明向 AVA 中引入部分 Kinetics 视频将提高模型在 AVA 上的泛化能力；另一方面，观察 AVA 数据集的各类分布，我们发现 AVA 呈现出明显的长尾分布。引入 Kinetics 视频将缓解这一问题，从而获得更好的结果。由于在 AVA-Kinetics 数据集上验证的模型数量较少，Table 5 中仅选取了来自 paperswithcode 网站的结果。

Table 6：在 MSR-VTT、MSVD、LSMDC、ActivityNet、DiDeMo 和 VATEX 上的视频检索结果。我们报告了文本到视频（T2V）和视频到文本（V2T）检索任务上的 R@1。

<table><tr><td rowspan="2">方法</td><td colspan="2">MSR-VTT</td><td colspan="2">MSVD</td><td colspan="2">LSMDC</td><td colspan="2">ActivityNet</td><td colspan="2">DiDeMo</td><td colspan="2">VATEX</td></tr><tr><td>T2V</td><td>V2T</td><td>T2V</td><td>V2T</td><td>T2V</td><td>V2T</td><td>T2V</td><td>V2T</td><td>T2V</td><td>V2T</td><td>T2V</td><td>V2T</td></tr><tr><td>CLIP4Clip [5]</td><td>45.6</td><td>45.9</td><td>45.2</td><td>48.4</td><td>24.3</td><td>23.8</td><td>40.3</td><td>41.6</td><td>43.0</td><td>43.6</td><td>63.0</td><td>78.3</td></tr><tr><td>TS2Net [88]</td><td>49.4</td><td>46.6</td><td>-</td><td>-</td><td>23.4</td><td>-</td><td>41.0</td><td>-</td><td>41.8</td><td>-</td><td>59.1</td><td>-</td></tr><tr><td>X-CLIP [22]</td><td>49.3</td><td>48.9</td><td>50.4</td><td>66.8</td><td>26.1</td><td>26.9</td><td>46.2</td><td>46.4</td><td>47.8</td><td>47.8</td><td>-</td><td>-</td></tr><tr><td>InternVideo</td><td>55.2</td><td>57.9</td><td>58.4</td><td>76.3</td><td>34.0</td><td>34.9</td><td>62.2</td><td>62.8</td><td>57.9</td><td>59.1</td><td>71.1</td><td>87.2</td></tr></table>

Table 7: MSRVTT、MSVD 和 TGIF 上的视频问答。我们报告 Top-1 准确率。

<table><tr><td>方法</td><td>MSRVTT</td><td>MSVD</td><td>TGIF</td></tr><tr><td>ClipBERT [54]</td><td>37.4</td><td>-</td><td>60.3</td></tr><tr><td>All-in-one [24]</td><td>42.9</td><td>46.5</td><td>64.2</td></tr><tr><td>MERLOT [9]</td><td>43.1</td><td>-</td><td>69.5</td></tr><tr><td>VIOLET [30]</td><td>43.9</td><td>47.9</td><td>68.9</td></tr><tr><td>InternVideo</td><td> $\mathbf{47.1}_{(+3.2)}$ </td><td> $\mathbf{55.5}_{(+7.6)}$ </td><td> $\mathbf{72.2}_{(+3.3)}$ </td></tr></table>

## 4.3.2 视频-语言对齐任务

视频检索。我们在视频检索任务上评估了 InternVideo。给定一组视频和相关的自然语言描述，该任务要求从候选集中检索出与其跨模态对应项相匹配的视频或描述。我们遵循通用范式，通过视觉编码器 $f _ { v } ( \cdot )$ 和文本编码器 $f _ { t } ( \cdot )$ 捕获视觉和文本语义，然后计算跨模态相似度矩阵作为检索指导。我们将多模态视频编码器用作 $f _ { v } ( \cdot )$，并将预训练的 ViT-L/14 [28] 作为基础 CLIP [13] 架构中的 $f _ { t } ( \cdot )$，并在每个检索数据集上微调整个模型。训练方案和大部分超参数设置均遵循 CLIP4Clip [5]，包括训练计划、学习率、批次大小、视频帧数、最大文本长度等。为了进一步提升模型性能，我们还采用了双 softmax 损失（dual softmax loss）[91] 作为后处理操作。

我们的模型在六个公开基准上进行了评估：MSR-VTT [92]、MSVD [93]、LSMDC [94]、DiDeMo [95]、ActivityNet [79] 和 VATEX [96]，遵循以往的工作，我们报告了在标准划分上的结果。我们在文本到视频和视频到文本任务上均采用 rank-1 (R@1) 指标来衡量检索结果，如 Table 6 所示。结果表明，我们的模型以较大优势显著优于以往的所有方法，展示了 InternVideo 在视频-语言相关任务上的优越性。更详细的检索结果，包括 rank-5 (R@5) 和 rank-10 (R@10)，可以在补充材料中找到。

视频问答（Video Question Answering）。为了进一步展示 InternVideo 的视觉-语言能力，我们在视频问答（VQA）任务上对 InternVideo 进行了评估。给定一个视频和问题对，VQA 的目标是预测该问题的答案。与没有跨模态融合的原始 CLIP 模型不同，我们的多模态视频编码器能够通过所提出的字幕解码器（caption decoder）捕捉模态之间的交互。生成 VQA 分类器所需特征有三种潜在途径：拼接视频编码器和文本编码器的特征、仅使用字幕解码器的特征，以及拼接来自视频编码器、文本编码器和字幕解码器的所有特征。经过对比，我们选择使用所有这三种来源的特征以提升性能。VQA 分类器为一个三层 MLP。

我们在三个常用的公开基准上进行了评估：MSR-VTT [92]、MSVD [97] 和 TGIF [98]。我们主要遵循 [24] 中的做法。结果如 Table 7 所示，我们的模型超越了以往所有的 SOTA，证明了我们跨模态学习器的有效性。

视觉语言导航（Visual Language Navigation）。视觉语言导航 [99] 要求智能体根据自然语言指令，基于其视觉感知在未知的逼真环境中进行导航。导航智能体应具备从导航历史中捕获时空信息（例如物体的相对运动）的能力，尤其是在智能体于连续空间中以较小步长导航时。为了验证我们模型这种能力的有效性，我们在要求智能体在连续环境中运行的 VLN-CE 基准 [100] 上进行了实验。

Table 8: VLN-CE 数据集的结果。

<table><tr><td rowspan="2">Agent</td><td rowspan="2">Backbone</td><td colspan="5">val-unseen</td></tr><tr><td>NE↓</td><td>PL</td><td>SR↑</td><td>NDTW↑</td><td>SPL↑</td></tr><tr><td>CWP-VLNBERT [89]</td><td>ResNet50</td><td>5.52</td><td>11.85</td><td>45.19</td><td>54.20</td><td>39.91</td></tr><tr><td>CWP-HEP [90]</td><td>CLIP-ViT-B/16</td><td>5.21</td><td>10.29</td><td>50.24</td><td>60.39</td><td>45.71</td></tr><tr><td>CWP-HEP [90]</td><td>InternVideo</td><td> $\mathbf{4.95}_{(-0.26)}$ </td><td>10.44</td><td> $\mathbf{52.90}_{(+2.66)}$ </td><td> $\mathbf{61.55}_{(+1.16)}$ </td><td> $\mathbf{47.70}_{(+0.99)}$ </td></tr></table>

Table 9: MSR-VTT、MSVD、LSMDC、ActivityNet、DiDeMo 和 VATEX 上的零样本（zero-shot）视频检索结果。我们报告了文本到视频（T2V）和视频到文本（V2T）检索任务上的 R@1。

<table><tr><td rowspan="2">方法</td><td colspan="2">MSR-VTT</td><td colspan="2">MSVD</td><td colspan="2">LSMDC</td><td colspan="2">ActivityNet</td><td colspan="2">DiDeMo</td><td colspan="2">VATEX</td></tr><tr><td>T2V</td><td>V2T</td><td>T2V</td><td>V2T</td><td>T2V</td><td>V2T</td><td>T2V</td><td>V2T</td><td>T2V</td><td>V2T</td><td>T2V</td><td>V2T</td></tr><tr><td>CLIP [13]</td><td>35.0</td><td>32.3</td><td>39.2</td><td>63.3</td><td>17.0</td><td>10.5</td><td>25.2</td><td>20.7</td><td>29.8</td><td>24.3</td><td>45.2</td><td>59.2</td></tr><tr><td>InternVideo</td><td>40.7</td><td>39.6</td><td>43.4</td><td>67.6</td><td>17.6</td><td>13.2</td><td>30.7</td><td>31.4</td><td>31.5</td><td>33.5</td><td>49.5</td><td>69.5</td></tr></table>

我们使用 [90](CWP-HEP) 中提出的方法进行实验。历史增强规划器（history-enhanced planner）是 HAMT [101] 的定制变体，它使用深度嵌入（depth embedding）与 RGB 嵌入的拼接作为输入嵌入。请注意，由于 VLN-CE 设置允许滑动，我们在此处没有使用试探控制器（tryout controller）。这是一个强大的基线，已经超越了此前最先进的方法 CWP-VLNBERT [89]。在每个决策循环中，我们收集最新的 16 帧观测以构建全景导航视频，然后使用 InternVideo 中的 ViT-L 对视频进行编码。将视频嵌入与 RGB 嵌入及深度嵌入拼接，作为最终的图像嵌入。关于评估，我们参考 [90] 中的详细指标。InternVideo 在成功率（Success Rate，SR）上将我们的基线从 50.2% 提升至 52.9%（Table 8）。

## 4.3.3 视频开放理解任务

零样本动作识别。零样本识别是原始 CLIP 模型的非凡能力之一。借助我们设计的多模态视频编码器，我们无需进一步优化即可实现显著的零样本动作识别性能。我们在 Kinetics-400 数据集上对模型进行了评估，取得了 64.25% 的准确率，大幅超越了先前的 SOTA 56.4% [102]。零样本视频检索。我们在零样本文本到视频以及视频到文本检索任务上将 InternVideo 与 CLIP 进行了比较。为公平起见，我们使用了带有 CLIP 预训练权重 <sup>1</sup> 的 ViT-L/14 模型。采用 Wise-finetuning [103] 和模型集成来进一步提升模型在零样本视频检索上的性能。我们通过实验发现，用于零样本检索的最佳视频帧数在 4 到 8 之间，并通过网格搜索获得了在每个基准数据集上表现最佳的帧数。如 Table 9 所示，InternVideo 在所有六个基准数据集上均展现出卓越的检索能力。此外，Florence [15] 使用了 900M 图文对进行预训练，并在 MSR-VTT 上取得了 37.6 的 R@1 文本到视频检索准确率。相比之下，我们的模型在使用少得多的训练数据（14.35M 视频 + 100M 图像 vs. 900M 图像）的情况下，比 Florence 高出 4.1%。这些结果揭示了我们的方法在预训练期间学习联合视频-文本特征空间的有效性。

零样本多项选择。零样本多项选择是另一项能够展现模型通用性的零样本任务。多项选择任务旨在从给定的选项中找到正确答案，这些选项通常是一个较小的子集（例如 5 个词）。我们发现，结合图文对的协同训练、wise-finetuning 和模型集成对于零样本多项选择任务的性能至关重要。我们在 Table 10 中报告了在 MSR-VTT 和 LSMDC 数据集上的零样本多项选择结果。我们将零样本性能作为训练中通用性的便捷指标，结果表明我们的模型是稳健且有效的。

开放集动作识别。在开放集动作识别（OSAR）中，模型需要识别来自训练类别的已知动作样本，并拒绝属于训练类别之外的未知样本。与图像相比，由于不确定的时间动态性以及人类动作的静态偏差 [109]，视频动作在 OSR 设置下的识别更具挑战性。我们的 InternVideo 在训练类别之外的未知类别上具有良好的泛化能力，并且在无需任何模型校准的情况下超越了现有方法 [109]。

Table 10: MSR-VTT 和 LSMDC 上的零样本多项选择。灰色表示采用监督训练的方法。

<table><tr><td colspan="2">MSR-VTT</td><td colspan="2">LSMDC</td></tr><tr><td>方法</td><td>准确率</td><td>方法</td><td>准确率</td></tr><tr><td>JSFusion [104]</td><td>83.4</td><td>JSFusion [104]</td><td>73.5</td></tr><tr><td>ActBERT [53]</td><td>85.7</td><td>MERLOT [9]</td><td>81.7</td></tr><tr><td>ClipBERT [54]</td><td>88.2</td><td>VIOLET [30]</td><td>82.9</td></tr><tr><td>All-in-one [24]</td><td>80.3</td><td>All-in-one [24]</td><td>56.3</td></tr><tr><td>InternVideo</td><td> $\mathbf{93.4}_{(+13.1)}$ </td><td>InternVideo</td><td> $\mathbf{77.3}_{(+21.0)}$ </td></tr></table>

Table 11: 在两个不同开集上的开集动作识别结果，其中未知类别的样本分别来自 HMDB-51 和 MiT-v2。我们汇报了在确保 95% 训练视频（UCF101）被识别为已知类别所确定的阈值下的开集 AUC。

<table><tr><td rowspan="2">方法</td><td colspan="2">开集 AUC (%)</td><td rowspan="2">闭集准确率 (%)</td></tr><tr><td>UCF-101 + HMDB-51</td><td>UCF-101 + MiT-v2</td></tr><tr><td>OpenMax [105]</td><td>78.76</td><td>80.62</td><td>62.09</td></tr><tr><td>MC Dropout [106]</td><td>75.41</td><td>78.49</td><td>96.75</td></tr><tr><td>BNN SVI [107]</td><td>74.78</td><td>77.39</td><td>96.43</td></tr><tr><td>SoftMax</td><td>79.16</td><td>82.88</td><td>96.70</td></tr><tr><td>RPL [108]</td><td>74.23</td><td>77.42</td><td>96.93</td></tr><tr><td>DEAR [109]</td><td>82.94</td><td>86.99</td><td>96.48</td></tr><tr><td>InternVideo</td><td> $\mathbf{85.48}_{(+2.54)}$ </td><td> $\mathbf{91.85}_{(+4.86)}$ </td><td> $\mathbf{97.89}_{(+1.41)}$ </td></tr></table>

我们使用 InternVideo 的 ViT-H/16 模型作为骨干网络，并在 UCF-101 [110] 训练集上使用一个简单的线性分类头对其进行微调。为了使 InternVideo 能够“知其未知”，我们遵循 [109] 中提出的 DEAR 方法，利用证据深度学习（Evidential Deep Learning, EDL）将其构建为一个不确定性估计问题，EDL 提供了一种联合构建多类别分类与不确定性建模的方法。具体而言，给定一个视频作为输入，位于 InternVideo 骨干网络之上的证据神经网络（Evidential Neural Network, ENN）头会预测类别证据，该证据构建了一个狄利克雷分布（Dirichlet distribution），从而可以确定输入的多分类概率和预测不确定性。在开集推理过程中，高不确定性的视频可被视为未知动作，而低不确定性的视频则根据学习到的类别概率进行分类。

InternVideo 不仅能够准确识别已知的动作类别，还能够识别未知类别。Table 11 汇报了 InternVideo 和其他基线方法在闭集（闭集准确率）和开集（开集 AUC）上的性能结果。结果表明，在未知样本分别来自 HMDB-51 [81] 和 MiT-v2 [111] 的两个开集数据集上，我们的 InternVideo 均持续且显著地优于其他基线方法。

## 5 结语

在本文中，我们提出了一个通用且训练高效的视频基础模型 InternVideo。据我们所知，InternVideo 是现有研究中首个在动作理解（action understanding）、视频-语言对齐（video-language alignment）以及视频开放理解（video open understanding）所有任务上均取得最佳表现的工作。与以往的相关工作 [6, 8, 9] 相比，它在涵盖 10 个不同任务的近 40 个数据集上取得了最先进的性能，将视频基础模型的通用性大幅提升到了一个全新的高度。该模型利用了基于掩码视频学习（VideoMAE）与视频-语言对比建模之间的跨模型学习，并结合了监督训练，从而构建了一种统一的视频表征。与以往的基础模型相比，它的训练效率极高。借助简单的 ViT 及其对应变体，我们仅耗费 64.5K GPU 小时（A100-80G）就实现了通用的视频表征，而 CoCa [7] 则需要 245.76K TPU 小时（v4）。我们在广泛的应用场景中验证了这种通用的时空表征。通过简单的任务头（甚至是线性头）和恰当的下游适配微调，我们的视频表征在所有使用的数据集上都取得了打破纪录的结果。即使在零样本（zero-shot）和开放集（open-set）设定下，我们的模型体系仍然实现了持续且显著的性能提升，进一步证明了其泛化与适配能力。

## 5.1 局限性

我们的研究展示了视频基础模型的有效性与可行性，而非提出了全新的公式表述或模型设计。它专注于当前主流的视频感知任务，并以视频片段（clips）的形式处理视频。其设计几乎无法处理长视频任务以及更高阶的任务，例如根据电影已观看的部分预测后续剧情。具备解决这些任务的能力对于进一步拓展视频表征学习的通用性至关重要。

## 5.2 未来工作

为了进一步扩展视频基础模型的通用性，我们认为在其研究中引入模型协同（model coordination）与认知（cognition）是十分必要的。具体而言，如何系统地协同来自不同模态、预训练任务甚至不同架构的基础模型以获得更好的表征，仍是一个开放且具有挑战性的问题。解决该问题有多条技术路线，例如模型蒸馏、统一不同的预训练目标以及特征对齐等。通过利用先前学到的知识，我们能够可持续地加速视频基础模型的发展。

从长远来看，基础模型（foundation models）有望掌握超越单纯感知能力的认知能力。考虑到其可行性，我们认为其研究趋势之一是从开放世界中的基础动态感知出发，实现大规模时空分析（长周期与大场景），从而达成本质的认知理解。此外，将基础模型与决策相结合以构建智能体（agents）来探索新任务已掀起一股浪潮。在这种交互中，数据收集和模型训练也实现了自动化。随着交互结果不断调整智能体的策略与行为，整个过程进入了一个闭环。我们在视觉-语言导航（vision-language navigation）上的初步实验（第 4.3.2 节）展示了将视频基础模型融入具身智能（Embodied AI）的广阔前景。

## 6 更广泛的影响

我们提出了视频基础模型谱系 InternVideo。它能够在约 40 个数据集上取得最先进的性能（SOTA），具备动作辨识、视频-语言对齐以及开放理解的能力。除了公开数据外，我们还利用了从互联网自主采集的数据。用于收集数据的检索词以及整理后的数据均经过了伦理和法律问题的审查。训练 InternVideo 的能耗远低于 CoCa [7]，仅占 CoCa 的 23.19 %。对于更深远的影响研究，我们还需要探讨偏见、风险、公平性、平等以及更多社会层面的议题。

## 参考文献

[1] Boyang Xia, Wenhao Wu, Haoran Wang, Rui Su, Dongliang He, Haosen Yang, Xiaoran Fan, and Wanli Ouyang. Nsnet: Non-saliency suppression sampler for efficient video recognition. In ECCV, 2022.

[2] Alexandros Stergiou and Ronald Poppe. Learn to cycle: Time-consistent feature discovery for action recognition. Pattern Recognition Letters, 141:1–7, 2021.

[3] Lei Wang and Piotr Koniusz. Self-supervising action recognition by statistical moment and subspace descriptors. In ACM International Conference on Multimedia, 2021.

[4] Chenlin Zhang, Jianxin Wu, and Yin Li. Actionformer: Localizing moments of actions with transformers. In eccv, 2022.

[5] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. Neurocomputing, 2022.

[6] Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan Yuille, and Christoph Feichtenhofer. Masked feature prediction for self-supervised visual pre-training. In CVPR, 2022.

[7] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022.

[8] Shen Yan, Xuehan Xiong, Anurag Arnab, Zhichao Lu, Mi Zhang, Chen Sun, and Cordelia Schmid. Multiview transformers for video recognition. In CVPR, 2022.

[9] Rowan Zellers, Ximing Lu, Jack Hessel, Youngjae Yu, Jae Sung Park, Jize Cao, Ali Farhadi, and Yejin Choi. Merlot: Multimodal neural script knowledge models. NeurIPS, 2021.

[10] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.

[11] Hassan Akbari, Liangzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, and Boqing Gong. Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text. NeurIPS, 2021.

[12] Jing Shao, Siyu Chen, Yangguang Li, Kun Wang, Zhenfei Yin, Yinan He, Jianing Teng, Qinghong Sun, Mengya Gao, Jihao Liu, et al. Intern: A new learning paradigm towards general vision. arXiv preprint arXiv:2111.08687, 2021.

[13] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.

[14] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In ICML, 2021.

[15] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision. arXiv preprint arXiv:2111.11432, 2021.

[16] Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Bin Xiao, Ce Liu, Lu Yuan, and Jianfeng Gao. Unified contrastive learning in image-text-label space. In CVPR, 2022.

[17] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. SimVLM: Simple visual language model pretraining with weak supervision. In ICLR, 2022.

[18] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In ICML, 2022.

[19] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for all vision and vision-language tasks. arXiv preprint arXiv:2208.10442, 2022.

[20] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. BEit: BERT pre-training of image transformers. In ICLR, 2022.

[21] Paul Barham, Aakanksha Chowdhery, Jeff Dean, Sanjay Ghemawat, Steven Hand, Daniel Hurt, Michael Isard, Hyeontaek Lim, Ruoming Pang, Sudip Roy, et al. Pathways: Asynchronous distributed dataflow for ml. Proceedings of Machine Learning and Systems, 2022.

[22] Yiwei Ma, Guohai Xu, Xiaoshuai Sun, Ming Yan, Ji Zhang, and Rongrong Ji. X-clip: 用于视频-文本检索的端到端多粒度对比学习. In ACM International Conference on Multimedia, 2022.

[23] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. VideoMAE: 掩码自编码器是用于自监督视频预训练的数据高效学习器. In NeurIPS, 2022.

[24] Alex Jinpeng Wang, Yixiao Ge, Rui Yan, Yuying Ge, Xudong Lin, Guanyu Cai, Jianping Wu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. All in one: 探索统一的视频-语言预训练. arXiv preprint arXiv:2203.07303, 2022.

[25] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. 掩码自编码器是可扩展的视觉学习器. In CVPR, 2022.

[26] Tianhao Li and Limin Wang. 通过视频和文本对判别学习时空特征. CoRR, abs/2001.05691, 2020.

[27] Joao Carreira and Andrew Zisserman. 动作识别向何处去？一种新模型与 Kinetics 数据集. In CVPR, 2017.

[28] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 一张图片等价于 16x16 的词：用于大规模图像识别的 Transformer. In ICLR, 2021.

[29] Roman Bachmann, David Mizrahi, Andrei Atanov, and Amir Zamir. MultiMAE: 多模态多任务掩码自编码器. arXiv preprint arXiv:2204.01678, 2022.

[30] Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu. Violet: 具有掩码视觉标记建模的端到端视频-语言 Transformer. arXiv preprint arXiv:2111.12681, 2021.

[31] Linjie Li, Zhe Gan, Kevin Lin, Chung-Ching Lin, Zicheng Liu, Ce Liu, and Lijuan Wang. Lavender: 将视频-语言理解统一为掩码语言建模. arXiv preprint arXiv:2206.07160, 2022.

[32] Rowan Zellers, Jiasen Lu, Ximing Lu, Youngjae Yu, Yanpeng Zhao, Mohammadreza Salehi, Aditya Kusupati, Jack Hessel, Ali Farhadi, and Yejin Choi. Merlot Reserve: 通过视觉、语言与声音获取神经脚本知识. In CVPR, 2022.

[33] Carl Doersch, Abhinav Gupta, and Alexei A Efros. 通过上下文预测进行无监督视觉表征学习. In ICCV, 2015.

[34] Xiaolong Wang and Abhinav Gupta. 利用视频进行视觉表征的无监督学习. In ICCV, 2015.

[35] Mehdi Noroozi and Paolo Favaro. 通过求解拼图进行视觉表征的无监督学习. In ECCV, 2016.

[36] Richard Zhang, Phillip Isola, and Alexei A Efros. 彩色图像着色. In ECCV, 2016.

[37] Christoph Feichtenhofer, Haoqi Fan, Yanghao Li, and Kaiming He. 掩码自编码器作为时空学习器. arXiv preprint arXiv:2205.09113, 2022.

[38] Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. 通过非参数实例判别进行无监督特征学习. In CVPR, 2018.

[39] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020.

[40] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020.

[41] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. NeurIPS, 2020.

[42] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In CVPR, 2021.

[43] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, 2020.

[44] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021.

[45] Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Yu-Gang Jiang, Luowei Zhou, and Lu Yuan. Bevt: Bert pretraining of video transformers. In CVPR, 2022.

[46] Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman. End-to-end learning of visual representations from uncurated instructional videos. In CVPR, 2020.

[47] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding. arXiv preprint arXiv:2109.14084, 2021.

[48] Xiaowei Hu, Zhe Gan, Jianfeng Wang, Zhengyuan Yang, Zicheng Liu, Yumao Lu, and Lijuan Wang. Scaling up visionlanguage pre-training for image captioning. In CVPR, 2022.

[49] Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang Zhu, Pengchuan Zhang, Lu Yuan, Nanyun Peng, et al. An empirical study of training end-to-end vision-and-language transformers. In CVPR, 2022.

[50] Sheng Shen, Liunian Harold Li, Hao Tan, Mohit Bansal, Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, and Kurt Keutzer. How much can clip benefit vision-and-language tasks? arXiv preprint arXiv:2107.06383, 2021.

[51] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive language-image pre-training. arXiv preprint arXiv:2111.07783, 2021.

[52] Chen Sun, Austin Myers, Carl Vondrick, Kevin P. Murphy, and Cordelia Schmid. Videobert: A joint model for video and language representation learning. ICCV, 2019.

[53] Linchao Zhu and Yi Yang. Actbert: Learning global-local video-text representations. CVPR, 2020.

[54] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, Mohit Bansal, and Jingjing Liu. Less is more: Clipbert for video-and-language learning via sparse sampling. In CVPR, 2021.

[55] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In ICCV, 2021.

[56] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Limin Wang, and Yu Qiao. Uniformerv2: Spatiotemporal learning by arming image vits with video uniformer. arXiv preprint arXiv:2211.09552, 2022.

[57] Guo Chen, Sen Xing, Zhe Chen, Yi Wang, Kunchang Li, Yizhuo Li, Yi Liu, Jiahao Wang, Yin-Dong Zheng, Bingkun Huang, et al. Internvideo-ego4d: A pack of champion solutions to ego4d challenges. arXiv preprint arXiv:2211.09529, 2022.

[58] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Luciˇ c, and Cordelia Schmid. Vivit: A video vision´ transformer. In ICCV, 2021.

[59] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In CVPR, 2022.

[60] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. NeurIPS, 2021.

[61] Tianwei Lin, Xu Zhao, Haisheng Su, Chongjing Wang, and Ming Yang. Bsn: Boundary sensitive network for temporal action proposal generation. In Proceedings of the European conference on computer vision (ECCV), pages 3–19, 2018.

[62] Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, and Shilei Wen. Bmn: Boundary-matching network for temporal action proposal generation. In ICCV, 2019.

[63] Elad Hoffer, Tal Ben-Nun, Itay Hubara, Niv Giladi, Torsten Hoefler, and Daniel Soudry. Augment your batch: Improving generalization through instance repetition. CVPR, 2020.

[64] Kunchang Li, Yali Wang, Gao Peng, Guanglu Song, Yu Liu, Hongsheng Li, and Yu Qiao. Uniformer: Unified transformer for efficient spatial-temporal representation learning. In ICLR, 2022.

[65] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool. Temporal segment networks: Towards good practices for deep action recognition. In ECCV, 2016.

[66] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. ICCV, 2019.

[67] Chunhui Gu, Chen Sun, David A Ross, Carl Vondrick, Caroline Pantofaru, Yeqing Li, Sudheendra Vijayanarasimhan, George Toderici, Susanna Ricco, Rahul Sukthankar, et al. Ava: A video dataset of spatio-temporally localized atomic visual actions. In CVPR, 2018.

[68] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The" something something" video database for learning and evaluating visual common sense. In ICCV, 2017.

[69] Joao Carreira, Eric Noland, Andras Banki-Horvath, Chloe Hillier, and Andrew Zisserman. A short note about kinetics-600. arXiv preprint arXiv:1808.01340, 2018.

[70] Joao Carreira, Eric Noland, Chloe Hillier, and Andrew Zisserman. A short note on the kinetics-700 human action dataset. arXiv preprint arXiv:1907.06987, 2019.

[71] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.

[72] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. arXiv preprint arXiv:2204.14198, 2022.

[73] Yuan Tian, Yichao Yan, Guangtao Zhai, Guodong Guo, and Zhiyong Gao. Ean: event adaptive network for enhanced action recognition. IJCV, 2022.

[74] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In ICCV, 2019.

[75] Zhiwu Qing, Haisheng Su, Weihao Gan, Dongliang Wang, Wei Wu, Xiang Wang, Yu Qiao, Junjie Yan, Changxin Gao, and Nong Sang. Temporal context aggregation network for temporal action proposal refinement. In CVPR, 2021.

[76] Humam Alwassel, Silvio Giancola, and Bernard Ghanem. Tsp: Temporally-sensitive pretraining of video encoders for localization tasks. In ICCV, 2021.

[77] Junting Pan, Siyu Chen, Mike Zheng Shou, Yu Liu, Jing Shao, and Hongsheng Li. Actor-context-actor relation network for spatio-temporal action localization. In CVPR, 2021.

[78] Yutong Feng, Jianwen Jiang, Ziyuan Huang, Zhiwu Qing, Xiang Wang, Shiwei Zhang, Mingqian Tang, and Yue Gao. Relation modeling in spatio-temporal action localization. arXiv preprint arXiv:2106.08061, 2021.

[79] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In CVPR, 2015.

[80] Hang Zhao, Antonio Torralba, Lorenzo Torresani, and Zhicheng Yan. Hacs: Human action clips and segments dataset for recognition and temporal localization. In ICCV, 2019.

[81] Hildegard Kuehne, Hueihan Jhuang, Estíbaliz Garrote, Tomaso Poggio, and Thomas Serre. Hmdb: a large video database for human motion recognition. In ICCV, 2011.

[82] Haroon Idrees, Amir R Zamir, Yu-Gang Jiang, Alex Gorban, Ivan Laptev, Rahul Sukthankar, and Mubarak Shah. The thumos challenge on action recognition for videos “in the wild”. CVIU, 2017.

[83] Yi Liu, Limin Wang, Yali Wang, Xiao Ma, and Yu Qiao. Fineaction: A fine-grained video dataset for temporal action localization. IEEE Transactions on Image Processing, 2022.

[84] Ang Li, Meghana Thotakuri, David A Ross, João Carreira, Alexander Vostrikov, and Andrew Zisserman. The ava-kinetics localized human actions video dataset. arXiv preprint arXiv:2005.00214, 2020.

[85] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014.

[86] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Girshick. Mask r-cnn. In ICCV, 2017.

[87] Jiajun Tang, Jin Xia, Xinzhi Mu, Bo Pang, and Cewu Lu. Asynchronous interaction aggregation for action detection. In ECCV, 2020.

[88] Yuqi Liu, Pengfei Xiong, Luhui Xu, Shengming Cao, and Qin Jin. Ts2-net: Token shift and selection transformer for text-video retrieval. In ECCV, 2022.

[89] Yicong Hong, Zun Wang, Qi Wu, and Stephen Gould. Bridging the gap between learning in discrete and continuous environments for vision-and-language navigation. In CVPR, 2022.

[90] Dong An, Zun Wang, Yangguang Li, Yi Wang, Yicong Hong, Yan Huang, Liang Wang, and Jing Shao. 1st place solutions for rxr-habitat vision-and-language navigation competition (cvpr 2022). arXiv preprint arXiv:2206.11610, 2022.

[91] Xing Cheng, Hezheng Lin, Xiangyu Wu, Fan Yang, and Dong Shen. Improving video-text retrieval by multi-stream corpus alignment and dual softmax loss. arXiv preprint arXiv:2109.04290, 2021.

[92] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In CVPR, 2016.

[93] Zuxuan Wu, Ting Yao, Yanwei Fu, and Yu-Gang Jiang. Deep learning for video classification and captioning. In Frontiers of Multimedia Research. 2017.

[94] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In ICCV, 2017.

[95] Anna Rohrbach, Marcus Rohrbach, Niket Tandon, and Bernt Schiele. A dataset for movie description. In CVPR, 2015.

[96] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, high-quality multilingual dataset for video-and-language research. In ICCV, 2019.

[97] David Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In ACL, 2011.

[98] Yuncheng Li, Yale Song, Liangliang Cao, Joel Tetreault, Larry Goldberg, Alejandro Jaimes, and Jiebo Luo. Tgif: A new dataset and benchmark on animated gif description. In CVPR, 2016.

[99] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In CVPR, 2018.

[100] Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph: Vision-and-language navigation in continuous environments. In ECCV, 2020.

[101] Shizhe Chen, Pierre-Louis Guhur, Cordelia Schmid, and Ivan Laptev. History aware multimodal transformer for vision-andlanguage navigation. NeurIPS, 2021.

[102] Mengmeng Wang, Jiazheng Xing, and Yong Liu. Actionclip: A new paradigm for video action recognition. ArXiv, abs/2109.08472, 2021.

[103] Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, et al. 零样本模型的鲁棒微调（Robust fine-tuning of zero-shot models）. In CVPR, 2022.

[104] Youngjae Yu, Jongseok Kim, and Gunhee Kim. 用于视频问答与检索的联合序列融合模型（A joint sequence fusion model for video question answering and retrieval）. In ECCV, 2018.

[105] Abhijit Bendale and Terrance E Boult. 迈向开放集深度网络（Towards open set deep networks）. In CVPR, 2016.

[106] Yarin Gal and Zoubin Ghahramani. 作为贝叶斯近似的 Dropout：在深度学习中表示模型不确定性（Dropout as a bayesian approximation: Representing model uncertainty in deep learning）. In ICML, 2016.

[107] Ranganath Krishnan, Mahesh Subedar, and Omesh Tickoo. BAR：基于变分推断的贝叶斯行为识别（Bar: Bayesian activity recognition using variational inference）. arXiv preprint arXiv:1811.03305, 2018.

[108] Guangyao Chen, Limeng Qiao, Yemin Shi, Peixi Peng, Jia Li, Tiejun Huang, Shiliang Pu, and Yonghong Tian. 利用判别性倒数点学习开放集网络（Learning open set network with discriminative reciprocal points）. In ECCV, 2020.

[109] Wentao Bao, Qi Yu, and Yu Kong. 面向开放集动作识别的证据深度学习（Evidential deep learning for open set action recognition）. In ICCV, 2021.

[110] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. 真实场景视频中的 101 类人类动作数据集（A dataset of 101 human action classes from videos in the wild）. Center for Research in Computer Vision, 2012.

[111] Mathew Monfort, Bowen Pan, Kandan Ramakrishnan, Alex Andonian, Barry A McNamara, Alex Lascelles, Quanfu Fan, Dan Gutfreund, Rogerio Feris, and Aude Oliva. Multi-Moments in Time：用于多动作视频理解的模型学习与解释（Multi-moments in time: Learning and interpreting models for multi-action video understanding）. TPAMI, 2021.