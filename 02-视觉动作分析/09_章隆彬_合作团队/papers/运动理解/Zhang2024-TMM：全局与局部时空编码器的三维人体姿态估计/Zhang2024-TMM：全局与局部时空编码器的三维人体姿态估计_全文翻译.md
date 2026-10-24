# Global and Local Spatio-Temporal Encoder for 3D Human Pose Estimation

> 原文出处：IEEE Transactions on Multimedia 26: 4039-4049, 2024 ｜ DOI 10.1109/tmm.2023.3321438
> 中文精读：research-tool 九段闭环 + LLM 翻译（2026-10-01 生成）
> 原文 PDF：`Zhang2023-TMM-3DPoseEstimator`（本地 `papers/`）
> 译文完整度：10/10 段已译

---

## 一句话结论
提出全局-局部时空编码器 GLSTE，同时建模全身全局与单关节局部时空关联，用 27 帧输入即在 Human3.6M / MPI-INF-3DHP 上超越多数使用 81/243 帧的 SOTA。

## 研究问题
- 单目 3D 人体姿态估计存在 2D→3D 多义性，单帧图像对关节位置和外观的微小变化敏感，需要时序信息消歧。
- 现有基于 Transformer 的方法（如 PoseFormer）侧重全身所有关节的全局时空关联，忽略每个关节自身的局部时空上下文。
- 长时序输入存在大量相邻帧冗余，带来高算力开销。
- 如何以更少的输入帧获得更高精度的中间帧 3D 姿态。

## 方法
- 任务定位：2D→3D lifting pipeline，先用现成 2D 姿态检测器得到 2D 关键点序列，再估计序列中间帧的 3D 姿态。
- 整体架构：global-to-local 串行级联，先全局后局部；空间→时序顺序建模。
- Global Spatial Encoder (GSE)：采用"简单但有效的全连接残差网络"融合单帧内所有关节信息，模拟全局空间相关性。
- Global Temporal Encoder (GTE)：使用 vanilla Transformer encoder，将每帧的空间特征作为一个 patch，建模跨帧全局时序相关性。
- Local Spatio-Temporal Encoder (LSTE)：对每个关节单独建模其在局部 N 帧内的时空关联。
- Parallel Attention Module（权重共享）：在同一节点上并行融合空间与时序信息，兼顾参数效率。
- 实验数据集：Human3.6M 与 MPI-INF-3DHP；输入窗口对比 27 / 81 / 243 帧。

## 关键数据
- 原文未给出（节选未列出具体 MPJPE / RMSE / 参数量 / FLOPs 数字）。
- 仅陈述性结论：以 **27 帧**输入的结果"优于绝大多数使用 **81 帧**和 **243 帧**输入的近期 SOTA 方法"。
- 模型规模描述："fewer parameters and less computational overhead"，具体倍数原文未给出。
- 单位/显著性/p 值：原文未给出。

## 结论与局限
- 在 Human3.6M 与 MPI-INF-3DHP 两个基准上，GLSTE 以更少参数和更低算力超越 SOTA。
- 27 帧即可达到多数 SOTA 用 81/243 帧才能达到的水平，说明模型能从更小输入中提取更有用信息。
- 局限：节选中未明确陈述（原文未给出）。

## 对「视觉→关节角（RGB 视频估计人体关节角序列）」课题的可借鉴点
- **可作 baseline** 的 global-to-local 解耦思路：将"全身所有关节全局时空相关"与"单关节局部轨迹相关"分模块建模，再融合；可直接套用到关节角序列回归中分离"多关节耦合"与"单关节运动平滑"两路分支。
- **可复用模块**：GSE 的"帧内全连接残差融合所有关节"可作为关节角回归中的帧内关节耦合编码器；LSTE 的"局部 N 帧单关节时空编码"可作为连续帧同关节角度演化（角速度/角加速度）特征提取块。
- **可比指标**：输入帧数 vs. 精度的 trade-off（27 vs. 81 vs. 243）可作为关节角序列回归里"时序窗口长度"对比的统一参考维度。
- **可借鉴的物理约束思路**：将 LSTE 的局部时空注意力直接作用于单关节轨迹，天然隐含关节运动平滑性先验，可作为对关节角序列做平滑性/连续性约束的轻量替代方案。

---

## 全文中文译

# 用于三维人体姿态估计的全局与局部时空编码器

王勇, 康洪波, 吴豆豆, 杨文明（IEEE 高级会员）, 张龙斌（IEEE 会员）

**摘要**——Transformer 已应用于三维人体姿态估计并取得了优异的性能；然而，大多数 Transformer 专注于对整个人体所有关节的全局时空相关性进行编码，而对人体中每个关节的局部时空相关性的研究较少。在本文中，我们提出了一种全局与局部时空编码器（Global and Local Spatio-Temporal Encoder, GLSTE）来建模时空相关性。具体而言，构建了全局空间编码器（Global Spatial Encoder, GSE）和全局时间编码器（Global Temporal Encoder, GTE），分别用于捕获单帧内所有关节的全局空间信息以及所有帧的全局时间信息。构建了局部时空编码器（Local Spatio-Temporal Encoder, LSTE）来捕获局部 N 帧内每个关节的空间与时间信息。此外，我们提出了一种带权重共享的并行注意力模块，以更好地将空间和时间信息同时整合到每个节点中。大量实验表明，在两个具有挑战性的数据集 Human3.6M 和 MPI-INF-3DHP 上，GLSTE 以更少的参数和更低的计算开销优于现有最先进（SOTA）的方法。特别是在 Human3.6M 数据集上的评估中，我们的方法以 27 帧作为输入的结果优于绝大多数以 81 和 243 帧作为输入的近期 SOTA 方法，这表明该模型能够以更小的输入学习到更有用的信息。

**索引词**——三维人体姿态估计，Transformer，时空编码器，并行注意力。

## I. 引言

单目三维人体姿态估计旨在通过图像或视频估计人体的三维关节位置。与二维人体姿态估计 [1]、行人重识别 [2]、[3]、[4]、[5] 等任务类似，这是计算机视觉中一个长期且具有挑战性的任务。其应用包括人机交互 [6]、动作识别 [7]、[8]、[9]、增强现实（AR）、虚拟现实（VR）[10]、医疗辅助等，近年来引起了大量关注。现有工作通常分为：1）直接从二维图像或视频帧推断三维人体姿态；2）通过任何现成的二维姿态估计器从单目图像或视频推断二维姿态，然后通过二维到三维提升（2D-to-3D lifting）流程从估计的二维姿态推断三维人体姿态。最先进的二维姿态检测器的优异性能使得二维到三维提升 [11]、[12]、[13] 表现更佳。然而，由于多个三维姿态可能对应几乎相同的二维投影，关节位置和外观的细微变化在这种情况下都可能影响预测结果。

为了缓解该问题，视频可以提供时间信息以提高三维人体姿态估计（3DHPE）的准确性和鲁棒性，而非从单目图像估计三维人体姿态。一些近期方法 [12]、[14]、[15]、[16] 通过聚合来自多帧的二维姿态序列信息来提高当前（中间）帧的三维姿态预测，从而获得更准确的三维表示并降低二维到三维的不确定性。我们的工作属于此类方法。Transformer [17] 被开发用于利用长程依赖关系，并在自然语言处理 [18]、[19] 和计算机视觉 [20]、[21]、[22]、[23] 中取得了巨大成功。由于 Transformer 的自注意力机制（self-attention），可以清晰地捕获长输入序列的全局相关性。它是序列数据问题的一个特别合适的架构，是对三维人体姿态估计（3DHPE）这一基本视觉任务的一种自然扩展。

在本文中，我们研究了 Transformer 在三维人体姿态估计（3DHPE）中的近期工作 [14]、[15]、[16]。Zheng 等人 [14] 提出的 PoseFormer 利用 Transformer 的自注意力机制来描述每帧内关节之间的空间依赖关系以及序列中帧之间的时间依赖关系。然而，由于视频中相邻帧的人体姿态具有相似性，整个时间序列中存在大量冗余信息，并且需要大量计算资源。此外，PoseFormer 忽略了局部上下文，但局部信息的使用在计算机视觉任务中至关重要。为了解决上述问题，Li 等人 [15] 提出对长程依赖关系（全局时间相关性）进行建模

---

稿件于 2022 年 10 月 8 日收到；分别于 2023 年 1 月 23 日、2023 年 6 月 27 日和 2023 年 9 月 20 日修订；2023 年 9 月 20 日被接受。发表日期为 2023 年 10 月 2 日；当前版本日期为 2024 年 2 月 23 日。本工作部分得到了中国国家自然科学基金（资助号 62171251）的支持，部分得到了深圳市战略性新兴产业发展专项资金（资助号 JCYJ20200109143035495、CJGJZD20210408092804011 和 JSGG20211108092812020）的支持。负责审稿并批准本稿件发表的副主编为 Jianguo Zhang 教授。（王勇与康洪波对本工作贡献相同。）（通讯作者：康洪波。）

王勇、康洪波和吴豆豆隶属于重庆理工大学人工智能学院，重庆 400054，中国（电子邮件：ywang@cqut.edu.cn；khb1698@163.com；doudouwu@stu.cqut.edu.cn）。

杨文明隶属于清华大学深圳国际研究生院电子工程系，深圳 518055，中国（电子邮件：yang.wenming@sz.tsinghua.edu.cn）。

张龙斌隶属于南洋理工大学机械与航空航天工程学院及新加坡康复研究所，新加坡 639798（电子邮件：longbin.zhang@ntu.edu.sg）。

---

1520-9210 © 2023 IEEE。允许个人使用，但再版/再分发需获得 IEEE 许可。
更多信息请参阅 https://www.ieee.org/publications/rights/index.html。

经授权使用许可。仅限于中南民族大学使用。下载时间：2026 年 10 月 1 日 03:36:18 UTC，来自 IEEE Xplore。使用受限。

授权许可使用范围限于：中南民族大学（South-Central Minzu University）。下载时间：2026 年 10 月 01 日 03:36:18 UTC，来源 IEEE Xplore。限制适用。

WANG 等人：全局与局部时空编码器（GLOBAL AND LOCAL SPATIO-TEMPORAL ENCODER） 4041

---

所提出的 GLSTE 是一种全局到局部的架构，分别通过不同模块捕获全局与局部信息。本文中的全局信息指的是人体所有关节点（joints）在所有帧（frames）中的时空相关性。根据文献 [14]、[16]，我们也采用由空间到时间的顺序建模方式，使用全局空间编码器（Global Spatial Encoder, GSE）与全局时间编码器（Global Temporal Encoder, GTE）模块。在 GSE 中，我们使用一种简单但有效的全连接残差网络（fully connected residual network）来整合每帧内所有关节点的信息，以模拟全局空间相关性。在 GTE 中，我们使用标准的 Transformer 编码器（vanilla Transformer encoder）[17]，它将每一帧的空间信息作为一个 patch（块），以模拟全局时间相关性。根据我们的观察，人体每个关节点的运动是不同的，且同一关节点在局部若干帧中的关系更为明显。因此，本文中的局部信息指的是同一关节点在局部 N 帧中的时空相关性。为了更好地捕获局部信息，在局部时空编码器（Local Spatio-Temporal Encoder, LSTE）中，我们提出了一种多头空间与时间自注意力（Multi-head Spatial and Temporal Self-Attention, MSA-ST），采用并行注意力模块（parallel attention module）来模拟局部时空相关性，并设计了一个空间与时间注意力融合模块（Spatial and Temporal Attention Fusion, STAF）来融合这两种注意力。由于任务目标是预测中间帧的三维人体姿态，我们设计了一个时间聚合器（Temporal Aggregator, TA），通过多层信息学习持续聚合局部信息，以获得最终的三维姿态。

我们的贡献总结如下：

- 我们提出了一种新的全局到局部的三维人体姿态估计架构，命名为 GLSTE，通过建模全局与局部时空相关性，从二维关节点位置高效地推断三维姿态。
- 在我们的 LSTE 中，创新性地提出了一种采用权重共享并行注意力的 MSA-ST 模块，用于建模局部时空相关性，其中空间注意力与时间注意力通过 STAF 模块进行融合。
- 我们的方法在两个基准（benchmarks）上达到了最先进的（state-of-the-art）性能，与同类方法相比，参数更少、计算开销（computational budgets）更小，并且模型能以更小的输入学习到更多有用的信息。

## II. 相关工作（RELATED WORK）

### A. 三维人体姿态估计（3D Human Pose Estimation）

三维人体姿态估计（3D Human Pose Estimation, 3D HPE）可分为直接估计方法（direct estimation approaches）和二维到三维提升方法（2D to 3D lifting approaches）。直接估计方法直接从二维图像推断三维人体姿态，无需中间估计二维人体姿态表示，例如文献 [24]、[25]、[26]、[27]。二维到三维提升方法首先估计二维人体姿态，然后从中间估计的二维人体姿态推断三维人体姿态。由于二维人体姿态估计算法的快速发展 [1]、[2]，第二种方法也带来了更好的性能。Martinez 等人 [11] 提出了一种全连接残差网络来完成二维到三维 HPE 任务。在一些后续工作 [28]、[29]、[30] 中，也尝试从底层二维关键点（2D keypoints）生成准确的三维姿态估计。尽管如此，从单张图像生成准确的三维姿态仍然是一个不适定问题（ill-posed problem）。最近的工作 [31]、[32]、[33] 利用视频中的时间信息来提高三维 HPE 的准确性与鲁棒性。

**图 2.** 我们所提出的全局与局部时空编码器（Global and Local Spatio-Temporal Encoder, GLSTE）概览，其输入为经过时间下采样策略（Temporal Downsampling Strategy, TDS）后的二维姿态序列（2D pose sequence），输出为完整序列尺度和单一目标帧尺度下的三维关节点位置。该模型需要在微调（fine-tuning）前加载预训练权重。在我们的模型中，全局特征通过全局空间编码器（Global Spatial Encoder, GSE）获取所有关节点的空间信息，随后通过全局时间编码器（Global Temporal Encoder, GTE）获取全部帧的时间信息。局部特征通过局部时空编码器（Local Spatio-Temporal Encoder, LSTE）获取每个关节点在局部帧中的时空信息，并通过时间聚合器（Temporal Aggregator, TA）对局部特征进行聚合，生成新的局部特征。

Hossain 等人 [34] 提出了一种由带层归一化（layer-normalized）的长短期记忆（Long Short-Term Memory, LSTM）单元组成的序列到序列（seq-to-seq）网络。Pavllo 等人 [12] 提出了一种基于膨胀时间卷积（dilated temporal convolutions）的网络，用于从连续的二维（2D）序列回归三维（3D）姿态。Chen 等人 [35] 提出了一种网络来预测骨骼方向（bone direction）和骨骼长度（bone length），以保证视频中人体的时序一致性。Liu 等人 [36] 提出了一种注意力机制（attentional mechanism），自适应地识别重要帧和每个深度神经网络层的张量输出，以获得更好的估计。Wang 等人 [33] 提出了一种 U 形图卷积网络（Graph Convolution Network, GCN），以捕获短期和长期运动信息。Hu 等人 [37] 进一步提出了时空条件有向图卷积（spatial-temporal conditional directed graph convolution），通过将人体骨骼表示为有向图（directed graph）来利用不同姿态的变长非局部依赖关系。

这些工作大多基于传统的神经网络结构，如 LSTM、CNN 和 GCN。然而，这些方法在捕获长期依赖关系方面仍存在一定局限。在本文中，我们提出使用具有更好全局相关性的 Transformer 来建模全局时间相关性。此外，我们还改进了 Transformer 的自注意力机制（self-attention mechanism），以更好地捕获局部时空相关性。

### B. 自注意力机制与 Transformer

近年来，Transformer [17] 在各个领域受到了广泛关注。因此，许多方法 [38]、[39]、[40]、[41] 已被提出采用自注意力机制来解决视觉任务。作为开创性工作，视觉 Transformer（Vision Transformer, ViT）[21] 将 Transformer [17] 架构直接应用于图像分类。DeiT [42] 通过采用一种新颖的蒸馏方法进一步扩展了 ViT。DETR [22] 被提出用于目标检测和全景分割。还有许多基于 Transformer 的三维人体姿态估计工作 [43]、[44]、[45]。PoseFormer [14] 使用纯 ViT 模型来捕获人体关节点之间的关系以及整个时间序列中跨帧的全局依赖，但忽略了时间上的局部信息。Strided Transformer [15] 将局部时间上下文整合进标准 Transformer。在前者的基础上，P-STMO [16] 加强了对全局空间信息的建模，并提出了时间下采样策略以缓解数据冗余，以及两阶段训练方法以降低模型训练难度，从而也取得了更好的效果。然而，上述方法将所有关节点聚合在一起，并捕获整体的时间上下文，而未考虑每个关节点的运动差异。这导致了在学习局部时空信息方面的不足。通过结合我们所设计的不同自注意力机制，我们的方法可以同时考虑人体内部的全局时空关系以及每个关节点在多个局部帧中的局部时空关系。由于我们考虑了每个关节点的运动，因此能够更全面地捕获信息，并实现更好的性能和泛化能力。

*授权许可使用范围有限：中南民族大学。下载时间：2026 年 10 月 01 日 03:36:18 UTC，来源：IEEE Xplore。限制适用。*

**图 3.** 用于三维姿态估计的 GLSTE 具体实现。我们通过 TDS 采样 27 帧作为输入。张量维度标注于圆括号中。全局模块对所有关节点建模，即 (27,34) 表示 27 帧和 34 的输入维度（17 个关节点的二维坐标）。局部模块对每个关节点单独建模，即 (17,27,2) 表示 17 个关节点、27 帧，每个关节点的输入维度为 2。(a) 全局空间编码器。(b) 全局时间编码器。(c) 局部时空编码器。

## 三、提出的方法

所提 GLSTE（Global and Local Spatio-Temporal Encoder）的整体结构如图 2 所示。这是一个全局到局部（global-to-local）的双通道输入与输出架构，我们采用预训练与微调两阶段策略来优化模型，如文献 [16] 所建议。在第一阶段，GLSTE 模型的全局部分（GSE 和 GTE）以自监督方式通过求解掩码体预测任务进行预训练，其主要目的是从受损的 2D 姿态中恢复出完整的 2D 姿态序列，从而学习 2D 时空相关性以及固有的全局特征。在第二阶段，我们加载全局部分的预训练权重对整个模型进行微调。在该阶段，全局部分首先用于建模全局时空信息，并由完整序列尺度进行监督以强制实现时序平滑。随后，局部嵌入模块引导从全局到局部的映射。最终，通过局部部分（LSTE 与 TA）得到中间帧的输出，该输出由单目标帧尺度监督，以产生更精确的估计。在两个阶段中，经过时间下采样（Temporal Downsampling, TDS）后的 2D 人体姿态坐标被用作输入序列 $\hat{P} \in \mathbb{R}^{T \times N \times 2}$，其中 $T$ 为帧数，$N$ 为每个 2D 姿态的关节点数，2 表示 2D 空间中的关节点坐标 $(x, y)$。GLSTE 的详细结构如图 3 所示。输入序列将依次经过 GSE、GTE、LSTE 和 TA 以获得最终输出。这些模块的细节将在以下各小节中描述。

### A. 预备知识

本节介绍一些关于 Transformer [17] 的背景知识，包括多头自注意力（Multi-head Self-Attention, MSA）和多层感知机（Multi-Layer Perceptron, MLP）。

**MSA：** 缩放点积注意力（scaled dot-product attention）是 MSA 的单个注意力头。它以注意力分数的形式隐式地关联输入块。输入为查询矩阵 $Q$、键矩阵 $K$ 和值矩阵 $V$。$Q, K, V \in \mathbb{R}^{n \times d_m}$，其中 $n$ 为输入块的数量，$d_m$ 为每个块的维度。缩放点积注意力定义如下：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_m}}\right)V \quad (1)$$

MSA 将查询、键和值拆分为多个头（head），然后对每个头应用缩放点积注意力。MSA 的输出被拼接（concatenate）后进行投影（projection）以得到最终输出。

**MLP：** MLP 由两个线性变换及中间的 ReLU 激活函数构成：

$$\text{MLP}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2 \quad (2)$$

其中 $W_1 \in \mathbb{R}^{d_m \times d_f}$、$W_2 \in \mathbb{R}^{d_f \times d_m}$、$b_1 \in \mathbb{R}^{d_f}$ 和 $b_2 \in \mathbb{R}^{d_m}$ 为参数矩阵，$d_f$ 为中间层的维度。

### B. 全局空间编码器

GSE 的目的是将人体所有关节点的空间信息融合到单帧之中，这需要从 2D 姿态中学习足够的空间信息并映射到更高的维度。Martinez 等人 [11] 提出使用全连接残差网络来建模单姿态的空间信息，该方法同样适用于本文中全局空间信息的融合。通过将序列中的每个 2D 姿态独立输入，即可获得每一帧的全局空间信息，如图 3(a) 所示。同时，GSE 也起到全局嵌入（Global embedding）的作用。2D 姿态序列中每帧所有关节点的 2 维坐标合并作为输入 $\hat{X} \in \mathbb{R}^{T \times (N \cdot 2)}$，并通过 GSE 的线性层将每帧的空间信息嵌入

---

*Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:36:18 UTC from IEEE Xplore. Restrictions apply.*

WANG 等：**全球与局部时空编码器** 4043

全局到局部块（Global-to-Local Block，GLB）用于处理全
                                                                                   局特征 $Z_g^M \in \mathbb{R}^{N \times d_g}$。由于全局特征
                                                                                   是融合所有节点的特征，而局部特征是单个节点的特征，我
                                                                                   们的 GLB 通过投影与分割（Proj&Split）实现全局到局部
                                                                                   的操作，其定义如下：
                                                                                                           投影与分割
                                                                                                     $Z_g^M \quad \longrightarrow \quad \hat{Z}_1 \in \mathbb{R}^{N \times T \times d_l}$                (7)
                                                                                      将二维姿态序列 $P$ 的时空通道进行交换，得到局部嵌入
                                                                                   的输入 $X \in \mathbb{R}^{N \times T \times 2}$。然后，局部嵌
                                                                                   入模块通过线性投影将每个节点的二维特征映射到更高维度
                                                                                   $d_l$。该过程还结合 $\hat{Z}_1$ 指导 GLB 从全局到局部的投
                                                                                   影，定义如下：
                                                                                                       $Z_1 = [\ldots; x_{i,j} E; \ldots] + \hat{Z}_1$                  (8)
                                                                                   其中 $x_{i,j}$ 表示 $X$ 的第 $j$ 帧的第 $i$ 个节点，$i = 1, 2, \ldots, N$，
                                                                                   $j = 1, 2, \ldots, T$，$E \in \mathbb{R}^{2 \times d_l}$ 是局部嵌入
Fig. 4. (a) 全局时间编码器（Global Temporal Encoder，GTE）中的多头自注意力（Multi-head Self-Attention，MSA）。（b）局部时空编码器（Local Spatio-Temporal Encoder，LSTE）中的多头时空自注意力（Multi-head Spatial and Temporal Self-Attention，MSA-ST），其中左侧为 $\text{Attn}_s$，右侧为 $\text{Attn}_t$。虚线框为注意力图矩阵。
                                                                                   的参数矩阵。

                                                                                   E. 局部时空编码器（Local Spatio-Temporal Encoder，LSTE）
                                                                                      根据我们的观察，人体每个关节的运动是不同的，同一
                                                                                   关节在局部帧中的关系更为显著。因此，我们设计了 LSTE
                                                                                   （见图 3(c)），用于建模每个关节在局部帧中的时空相关性。
                                                                                   在 LSTE 中，我们创新性地提出一种带并行注意力模块的
                                                                                   MSA-ST 来模拟局部时空相关性，并设计了一个时空注意力
                                                                                   融合块（Spatial and Temporal Attention Fusion Block，STAF）
                                                                                   来融合这两种注意力。在本节中，$d_l$ 是局部模块（LSTE
                                                                                   和 TA）的关键参数。
                                                                                      多头时空自注意力（Multi-head Spatio-Temporal Self-Attention，MSA-ST）：
                                                                                   在 MSA-ST 中，主要部分是权重共享级联注意力模块，如
                                                                                   图 4(b) 所示，它包含空间注意力（$\text{Attn}_s$）和时间注意力
                                                                                   （$\text{Attn}_t$）。三个映射得到的可学习矩阵 $Q, K, V \in \mathbb{R}^{N \times T \times d_l}$
                                                                                   被这两种注意力共享，但需要相应地变换：$Q_s, K_s, V_s$
                                                                                   通过交换空间维度与时间维度得到，$Q_t, K_t, V_t$ 通过将
                                                                                   完整序列 $T$ 划分为 $S$ 个局部帧数为 $\bar{T}$ 的序列得到，定义
                                                                                   如下：
                                                                                                           重塑（Reshape）
                                                                                            $\{Q, K, V\} \quad \longrightarrow \quad \{Q_s, K_s, V_s\} \in \mathbb{R}^{T \times N \times d_l}$                (9)
                                                                                                           分割（Split）
                                                                                            $\{Q, K, V\} \quad \longrightarrow \quad \{Q_t, K_t, V_t\} \in \mathbb{R}^{N \times S \times \bar{T} \times d_l}$                (10)
                                                                                      空间注意力 $\text{Attn}_s$ 用于模拟单帧中每个关节之间的空
                                                                                   间关系，时间注意力 $\text{Attn}_t$ 用于模拟同一关节在 $\bar{T}$ 个局部
                                                                                   帧之间的时间关系。$\text{Attn}_s$ 和 $\text{Attn}_t$ 的计算如下：
                                                                                       $\text{Attn}_s(Q, K, V) = \text{Reshape}(\text{Attention}(Q_s, K_s, V_s))$                (11)
                                                                                       $\text{Attn}_t(Q, K, V) = \text{Concat}(\text{Attention}(Q_t, K_t, V_t))$                (12)
                                                                                   其中 $\text{Attention}(\cdot)$ 是公式 (1) 中的缩放点积注意力。
                                                                                   $\text{Attn}_s$ 和 $\text{Attn}_t$ 的注意力图矩阵大小分别为
                                                                                   $T \times N \times N$ 和 $N \times S \times \bar{T} \times \bar{T}$。假设 MSA-ST

     Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:36:18 UTC from IEEE Xplore. Restrictions apply.

                                                                                 将该过程应用于输入 $Z_l^0 \in \mathbb{R}^{N \times T \times d_l}$ 以获得输出
                                                                                 $Z_l^N \in \mathbb{R}^{N \times 1 \times d_l}$，过程如下：
                                                                                                  $\hat{Z}_l^n = Z_l^{n-1} + E_t^n$                                (19)
                                                                                                  $\hat{Z}_l^n = \text{MSA-ST}(\text{LN}(\hat{Z}_l^n)) + \hat{Z}_l^n$                (20)

**Fig. 5.** 所提出的空间与时间注意力融合（Spatial and Temporal Attention Fusion, STAF）模块，用于融合空间与时间注意力。

通过将所有头拼接来实现，如下：

$$MSA_{ST} = F(\text{Concat}(Attn_s^i(\cdot), \text{Concat}(Attn_t^i(\cdot)))) \cdot W \quad (13)$$

其中 $i = 1, 2, \ldots, h$，$W \in \mathbb{R}^{dl \times dl}$ 是一个可学习矩阵，$F(\cdot)$ 是 (17) 中的融合函数，$Attn_s^i(\cdot)$ 和 $Attn_t^i(\cdot)$ 分别为 $Attn_s(\cdot)$ 和 $Attn_t(\cdot)$ 的第 $i$ 个头。

**空间与时间注意力融合（Spatial and Temporal Attention Fusion, STAF）** 为了更好地整合 $Attn_s$ 和 $Attn_t$ 两条支路的信息，我们不采用简单相加的方式，而是设计 STAF 来动态调整空间与时间注意力信息，如图 5 所示。我们首先在特征通道维度上拼接 $Attn_s$ 和 $Attn_t$，得到的维度为 $\mathbb{R}^{N \times T \times (2 \cdot dl)}$。然后使用平均池化（Average Pooling, AvgPool）和最大池化（Max Pooling, MaxPool）操作压缩输入特征的时间维度。该过程生成两个维度为 $\mathbb{R}^{N \times 1 \times (2 \cdot dl)}$ 的特征，并通过相加将平均特征和最大特征进行融合。最后，通过全连接（Fully Connected, FC）网络和 Sigmoid 操作生成权重 $\alpha_s, \alpha_t \in \mathbb{R}^{N \times 1 \times dl}$。$Attn_s$ 与 $Attn_t$ 的融合定义如下：

$$f_{cat} = \text{Concat}(Attn_s, Attn_t) \quad (14)$$

$$f_{avg+max} = FC(\text{AvgPool}(f_{cat}) + \text{MaxPool}(f_{cat})) \quad (15)$$

$$\alpha_s, \alpha_t = \text{Split}(\text{Sigmoid}(f_{avg+max})) \quad (16)$$

$$F(Attn_s, Attn_t) = \alpha_s \cdot Attn_s + \alpha_t \cdot Attn_t \quad (17)$$

**LSTE 的网络结构：** LSTE 的输入 $Z_1$ 通过由全局到局部的投影得到，见 (7) 和 (8)。为了定位 patch 的空间位置，我们向输入添加位置编码以注入序列信息，即空间位置编码 $E_s \in \mathbb{R}^{N \times 1 \times dl}$。它是一个可训练的序列矩阵，被加到输入中，可表示为：

$$Z_l^0 = Z_1 + E_s \quad (18)$$

LSTE 的网络结构如图 3(c) 所示。局部时空编码器（Local Spatio-Temporal Encoder, LSTE）和时间聚合器（在 III-F 节中）组合建模局部特征，由 $N$ 层构成。由于每层的时间维度会发生变化，模型在每层之前使用可训练的时间位置编码作为前缀，第一个时间位置编码为 $E_t^1 \in \mathbb{R}^{1 \times T \times dl}$，之后每层的时间编码将随时间维度 $T$ 的变化而变化。

$$\hat{Z}_l^n = MLP(LN(\hat{Z}_l^n)) + \hat{Z}_l^n \quad (21)$$

$$Z_l^n = TA(\hat{Z}_l^n, \bar{T}_n) \quad (22)$$

其中 $n = 1, 2, \ldots, N$，$TA(\cdot)$ 为时间聚合器（T̄n 是第 $n$ 层中的局部帧数，$E_t^n$ 表示第 $n$ 层之前的时间位置编码）。上述公式可表示为：

$$Z_l^n = TA(LSTE(Z_l^{n-1} + E_t^n), \bar{T}_n) \quad (23)$$

**F. 时间聚合器（Temporal Aggregator, TA）** 本工作的任务是从整个序列中回归出单帧（中间帧）的姿态。如果仅学习时间序列而不进行聚合，模型将获得大量冗余信息并增加计算开销。因此，我们提出时间聚合器，以更好地回归中间帧的三维姿态并聚合局部信息。具体而言，我们通过一维卷积分别聚合每个节点的时间信息，输入的时间序列从 $T$ 变为 $S$，其中 $T$ 和 $S$ 分别表示当前层的完整序列数和当前层划分后的序列数。上述方法可表示为函数 $TA(\cdot, \bar{T})$，其中 $\bar{T}$ 是当前层划分序列的局部帧数，我们需结合 LSTE 使用，如 (23) 所示。

**G. 回归头（Regression Head）** 由于我们使用帧序列来预测中心帧的三维姿态，仅监督单个姿态会损失时间的平滑性。因此，我们通过在训练期间从 GTE 输出处添加回归头，施加额外的时间平滑性约束，并获得三维姿态序列的完整序列预测输出 $\hat{Y} \in \mathbb{R}^{T \times N \times 3}$。最后，在 LSTE 的输出处添加回归头，以获得中间帧三维关节位置的预测输出 $Y \in \mathbb{R}^{N \times 3}$。

**H. 损失函数（Loss Function）** 由于 GTE 的输出是所有帧的特征信息，我们需要对完整序列尺度进行监督以实现序列上的时间一致性。因此，我们采用序列损失 [15] 来最小化预测值与真实值之间的误差：

$$L_{global} = \frac{1}{N \cdot T} \sum_{i=1}^{N} \sum_{j=1}^{T} \|P_{i,j} - \hat{Y}_{i,j}\|^2 \quad (24)$$

其中 $\hat{Y}_{i,j}$ 与 $P_{i,j}$ 分别表示第 $j$ 帧第 $i$ 个关节的估计与真实三维位置序列。

鉴于 LSTE 对局部帧进行建模，TA 聚合时间信息以最终获得中间帧的特征，因此必须对单个姿态进行监督。我们应用标准的平均每关节位置误差（Mean Per Joint Position Error, MPJPE）损失来最小化预测姿态与真实姿态之间的误差，如下：

$$L_{local} = \frac{1}{N} \sum_{i=1}^{N} \|P_i - Y_i\|^2 \quad (25)$$

数据集上将 GLSTE 与之前的现有最佳方法进行比较。我们使用 CPN [1] 作为 2D 关键点检测器，然后在检测得到的 2D 姿态序列上训练网络。如表 I 所示，展示了我们的模型在 243 帧下的性能。我们的方法在 Protocol 1（42.2 mm）和 Protocol 2（33.8 mm）下表现良好。我们还使用与 [15]、[16]、[31]、[33] 相同的细化模块来训练模型。该方法在 Protocol 1 下达到 41.9 mm，优于所有其他方法。

如表 II 所示，当使用真实 2D 姿态（ground truth）作为输入时，我们的方法取得了最佳结果（MPJPE 为 27.9 mm），优于所有其他方法。除了 Directions 动作外，我们方法的检测结果优于之前的工作，这也证明了我们模型的鲁棒性。

在表 III 中，我们针对不同输入帧数 (T)，从参数总量、每帧估计浮点运算次数（FLOPs）以及模型性能等方面与现有最佳方法进行了对比。可以看出，我们方法的参数量最少，但性能最佳。与基于全卷积架构的方法 [12]、[35] 相比，基于 Transformer 架构的方法 [16] 更有效，我们的方法优于同样采用预训练和下采样策略的方法 [16]。值得注意的是，我们的模型在 27 帧、MPJPE 为 43.7 mm 时的性能超过了 Shan 等人 [16] 在 81 帧以及 Chen 等人 [35] 在 243 帧下的结果，且我们的参数量和 FLOPs 远少于它们。同时，图 6 也展示了表中不同输入帧下的 MPJPE 指标比较。

**表 I**

在来自级联金字塔网络（Cascaded Pyramid Network，CPN）[1] 的检测 2D 姿态条件下，于 Human3.6M 数据集上 Protocol 1 与 Protocol 2 下与现有最佳方法的定量比较

**表 II**

以真实（Ground Truth，GT）2D 关节位置作为输入时，于 Human3.6M 数据集上 Protocol 1 下 MPJPE（以毫米为单位）的定量比较

**表 III**

在 Human3.6M 数据集上不同输入帧数条件下与现有最佳方法的定量比较

**表 IV**

在 MPI-INF-3DHP 上三种评估指标下的结果

---

经授权许可使用，仅限于：中南民族大学。下载时间：2026 年 10 月 1 日 03:36:18 UTC，来自 IEEE Xplore。适用限制条件。

再次在所有三项评估指标（PCK、AUC 和 MPJPE）上达到最先进水平（state-of-the-art）。这凸显了我们的 GLSTE 在提升室外场景性能方面的有效性。

   **MPI-INF-3DHP 上的结果**：为评估所提出模型的泛化能力，表 IV 报告了 GLSTE 与其他方法在 MPI-INF-3DHP 上的定量结果。沿用文献 [14]、[33]、[37]，由于该数据集的序列长度相比 Human3.6M 更短，我们采用长度为 81 帧的二维姿态序列作为输入。表中显示，我们的方法在所有评估指标上均达到最先进水平，凸显了 GLSTE 在室外场景下提升性能的有效性。

   **定性结果**：我们在三维估计姿态与真实标签之间提供了可视化对比。我们在 Human3.6M 测试集 S9 与 S11 上，针对三个具有挑战性的动作（Photo、Sitting 与 SittingDown）对 GLSTE 进行了评估。与最先进方法 [16] 相比，如图 7 所示，所提方法能够获得更准确的预测结果。

WANG 等：全局与局部时空编码器（Global and Local Spatio-Temporal Encoder）

                                                                                                                                4047

Fig. 7. 在 Human3.6M 上，针对拍照（photo）、坐下（sitting down）和坐姿（sitting）动作，我们的方法（GLSTE）与文献 [16] 的定性对比。红色箭头指示我们方法具有更好结果的位置。

                              TABLE V                                                                            TABLE VI
        GLSTE 中不同组件的消融研究（ABLATION STUDY）                  HUMAN3.6M 在协议1下模型超参数的消融研究（ABLATION STUDY）

                                                                                          TABLE VII
                                                              GLSTE 中各层局部帧（LOCAL FRAMES）配置的消融研究（ABLATION STUDY）

D. 消融研究（Ablation Study）

   为验证所提出模型的有效性，我们进行了多项消融实验。消融实验在协议 1 下基于 Human3.6M 数据集进行，输入选取为 27 帧。实验中我们均采用预训练与降采样策略，其中降采样率为 9。

   **模型组件**：我们在表 V 中验证 GLSTE 模型中不同模块的贡献。仅由 GSE 构成的网络在 MPJPE 指标下达到 51.8 毫米。为评估 GTE 的效果，我们以 Transformer 作为骨干网络，得到 47.0 毫米的结果。将 GSE 与 GTE 组合后，MPJPE 降至 45.1 毫米。然而，将 GSE、GTE 与 LSTE 组合后，最终的 MPJPE 增加至 47.5 毫米。可能的原因在于网络学习到了大量的冗余时序信息，因此模型无法良好地获取中间帧。因此，我们在每一层 LSTE 之后添加 TA 模块以获得所提的 GLSTE 模型，在 MPJPE 指标下达到 43.7 毫米，并且计算成本降低（FLOPs 从 76.55 降至 70.78）。

   我们还评估了不同的全局到局部连接方式。与简单的直接投影方法（GLSTE∗）相比，如表中所示，我们的 GLSTE 通过局部嵌入（Local Embedding）模块来指导全局到局部的投影，能够获得更小的 MPJPE，这表明我们的连接方法是有效的。

   **模型超参数**：我们在表 VI 中探索了多种超参数组合以寻找最佳的网络架构。dg 表示全局时空编码器（global spatio-temporal encoder, GSE）与全局时序编码器（global temporal encoder, GTE）的中间维度，dl 表示局部时空编码器（local spatio-temporal encoder, LSTE）的中间维度。随着 dg 的增大，参数数量与计算量也呈指数增长，但 MPJPE 并未取得更优的结果。而 dl 的增大对参数数量与计算量的影响较小。在我们的模型中，dg = 256 与 dl = 32 是最佳的超参数配置，且参数数量与计算成本也相对较低。

   **各层的局部帧**：表 VII 展示了不同的局部帧配置对模型性能的影响。由于 LSTE 模块模拟的是单节点在局部帧中的时空相关性，因此每一层 LSTE 可以通过不同的局部帧进行配置。由于 TA 模块在每一层 LSTE 之后执行时序聚合并合并局部信息，我们需要确保每一层中局部帧的数量

## 表 VIII
**关于 LSTE 不同多头自注意力（MSA）设计的消融研究**

等于时间序列的总数。对于 27 帧，共有四种配置，其中当每一层的局部帧数为 (27) 时，MPJPE 最大。可能的原因是网络模拟了所有帧之间单个节点的时空信息，而长距离的时间信息无法对当前节点起到积极作用，因此结果并不理想。当每一层的局部帧数为 (3, 3, 3) 时，这是最佳配置，MPJPE 为 43.7 mm。

**LSTE 的不同多头自注意力（MSA）设计**：在表 VIII 中，我们进行实验以验证不同 MSA 设计对所提模型的影响。在没有时空注意力融合模块（STAF）的情况下，模型的参数量和计算量保持不变，因为空间注意力（Attns）和时间注意力（Attnt）的权重是共享的，即权重的大小不会改变。当多头注意力模块中仅有空间注意力时，模型的 MPJPE 为 44.0 mm，而仅有时间注意力时，模型的 MPJPE 为 45.0 mm。这两种策略的结果均优于文献 [16] 的结果，证明了这两个模块的有效性。为了获得更好的结果，我们提出了两种融合方法。第一种方法直接将 Attns 和 Attnt 元素相加，其 MPJPE 为 44.1 mm，效果不如仅使用空间注意力的模块。第二种方法使用 STAF 模块进行融合，其 MPJPE 为 43.7 mm，在参数量和计算量略有增加的情况下，比前一种方法高 0.4 mm。这是因为 STAF 能够通过时间序列中每个关节的最大特征和平均特征动态调整时空注意力信息。

## V. 结论（CONCLUSION）

本文中，我们提出了一种用于单目视频三维人体姿态估计的全局与局部时空编码器（Global and Local Spatio-Temporal Encoder），该编码器带有并行注意力模块。我们的结果表明，所提编码器能够在两个流行的公开三维姿态数据集 Human3.6M 和 MPI-INF-3DHP 上更好地捕捉人体所有关节的全局时空信息和不同身体关节的局部时空信息，同时具有更少的参数量和更低的计算开销。具体而言，我们的模型在使用更少输入帧的情况下获得了更有用的信息，实现了最先进的性能。所提算法在人机交互（HCI）和虚拟现实（VR）等应用中具有广阔的前景。

## 参考文献（REFERENCES）

[1] Y. Chen 等人，"用于多人姿态估计的级联金字塔网络"，载于 Proc. IEEE Conf. Comput. Vis. Pattern Recognit.，2018，第 7103–7112 页。

[2] K. Sun、B. Xiao、D. Liu 和 J. Wang，"用于人体姿态估计的深度高分辨率表示学习"，载于 Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.，2019，第 5693–5703 页。

[3] L. Wu 等人，"用于行人重识别中相对表示学习的基于深度协同注意力的比较器"，IEEE Trans. Neural Netw. Learn. Syst.，第 32 卷，第 2 期，第 722–735 页，2021 年 2 月。

[4] L. Wu 等人，"用于无监督行人重识别的基于伪对的自相似性学习"，IEEE Trans. Image Process.，第 31 卷，第 4803–4816 页，2022。

[5] L. Wu、Y. Wang、L. Shao 和 M. Wang，"3-D PersonVLAD：基于视频的行人重识别的深度全局表示学习"，IEEE Trans. Neural Netw. Learn. Syst.，第 30 卷，第 11 期，第 3347–3359 页，2019 年 11 月。

[6] J. Preece 等人，《人机交互》。Boston, MA, USA: Addison-Wesley，1994。

[7] M. Liu、H. Liu 和 C. Chen，"用于视角无关人体动作识别的增强骨骼可视化"，Pattern Recognit.，第 68 卷，第 346–362 页，2017。

[8] M. Liu 和 J. Yuan，"将人体动作识别为姿态估计图的演化"，载于 Proc. IEEE Conf. Comput. Vis. Pattern Recognit.，2018，第 1159–1168 页。

[9] P. Wang、W. Li、Z. Gao、C. Tang 和 P. O. Ogunbona，"基于深度池化的大规模三维动作识别与卷积神经网络"，IEEE Trans. Multimedia，第 20 卷，第 5 期，第 1051–1061 页，2018 年 5 月。

[10] D. Mehta 等人，"VNect：使用单个 RGB 摄像头的实时三维人体姿态估计"，ACM Trans. Graph.，第 36 卷，第 4 期，第 1–14 页，2017。

[11] J. Martinez、R. Hossain、J. Romero 和 J. J. Little，"用于三维人体姿态估计的简单而有效的基线方法"，载于 Proc. IEEE Int. Conf. Comput. Vis.，2017，第 2640–2649 页。

[12] D. Pavllo、C. Feichtenhofer、D. Grangier 和 M. Auli，"视频中带时间卷积和半监督训练的三维人体姿态估计"，载于 Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.，2019，第 7753–7762 页。

[13] G. Hua、W. Li、Q. Zhang、R. Ding 和 H. Liu，"使用跨视图 U 形图卷积网络的弱监督三维人体姿态估计"，IEEE Trans. Multimedia，第 1832–1843 页，2022。

[14] C. Zheng 等人，"使用空间和时间 Transformer 的三维人体姿态估计"，载于 Proc. IEEE/CVF Int. Conf. Comput. Vis.，2021，第 11656–11665 页。

[15] W. Li 等人，"利用带步幅 Transformer 的时间上下文进行三维人体姿态估计"，IEEE Trans. Multimedia，第 25 卷，第 1282–1293 页，2023。

[16] W. Shan 等人，"P-STMO：用于三维人体姿态估计的预训练时空多对一模型"，载于 Proc. Eur. Conf. Comput. Vis.，2022，第 461–478 页。

[17] A. Vaswani 等人，"Attention is all you need"，载于 Proc. Annu. Conf. Neural Inf. Process. Syst.，第 30 卷，2017，第 5998–6008 页。

[18] Y. Tay、M. Dehghani、D. Bahri 和 D. Metzler，"高效 Transformer：一项综述"，ACM Comput. Surv.，第 55 卷，第 1–28 页，2020。

[19] Z. Dai、G. Lai、Y. Yang 和 Q. Le，"Funnel-transformer：过滤序列冗余以实现高效语言处理"，载于 Proc. Annu. Conf. Neural Inf. Process. Syst.，2020，第 4271–4288 页。

[20] K. Han 等人，"视觉 Transformer 综述"，IEEE Trans. Pattern Anal. Mach. Intell.，第 45 卷，第 1 期，第 87–110 页，2022。

[21] A. Dosovitskiy 等人，"一幅图像胜过 16×16 个词：用于大规模图像识别的 Transformer"，载于 Proc. Int. Conf. Learn. Representations，2021。[在线]。Available: https://openreview.net/forum?id=YicbFdNTTy

[22] N. Carion 等人，"使用 Transformer 的端到端目标检测"，载于 Proc. Eur. Conf. Comput. Vis.，2020，第 213–229 页。

[23] H. Bao、L. Dong 和 F. Wei，"BEiT：图像 Transformer 的 BERT 预训练"，载于 Proc. Int. Conf. Learn. Representations，2022。[在线]。Available: https://openreview.net/forum?id=p-BhZSz59o4

[24] G. Pavlakos、X. Zhou、K. G. Derpanis 和 K. Daniilidis，"单张图像三维人体姿态的由粗到细体素预测"，载于 Proc. IEEE Conf. Comput. Vis. Pattern Recognit.，2017，第 7025–7034 页。

[25] X. Sun、B. Xiao、F. Wei、S. Liang 和 Y. Wei，"整体人体姿态回归"，载于 Proc. Eur. Conf. Comput. Vis.，2018，第 529–545 页。

[26] L. Zhao、X. Peng、Y. Tian、M. Kapadia 和 D. N. Metaxas，"用于三维人体姿态回归的语义图卷积网络"，载于 Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.，2019，第 3425–3435 页。

授权许可使用，仅限于：中南民族大学。下载时间：2026年10月01日 03:36:18 UTC，来自 IEEE Xplore。受限制使用。

WANG 等人：用于三维人体姿态估计的全局与局部时空编码器                                                                                                                4049

[27] J. Liu 等人，"用于三维姿态估计的特征增强网络（Feature boosting network for 3D pose estimation），" *IEEE 模式分析与机器智能汇刊（IEEE Trans. Pattern Anal. Mach. Intell.）*，第 42 卷，第 2 期，第 494–501 页，2020 年 2 月。

[28] B. Wandt 和 B. Rosenhahn，"RepNet：用于三维人体姿态估计的对抗重投影网络弱监督训练（RepNet: Weakly supervised training of an adversarial reprojection network for 3D human pose estimation），" 见于 *IEEE/CVF 计算机视觉与模式识别会议论文集（Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.）*，2019 年，第 7782–7791 页。

[29] C. Li 和 G. H. Lee，"使用混合密度网络生成三维人体姿态估计的多个假设（Generating multiple hypotheses for 3D human pose estimation with mixture density network），" 见于 *IEEE/CVF 计算机视觉与模式识别会议论文集（Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.）*，2019 年，第 9887–9895 页。

[30] W. Li、H. Liu、T. Guo、H. Tang 和 R. Ding，"GraphMLP：用于三维人体姿态估计的图 MLP 类架构（GraphMLP: A graph MLP-like architecture for 3D human pose estimation），" 2022 年，arXiv:2206.06420。

[31] Y. Cai 等人，"通过图卷积网络利用时空关系进行三维姿态估计（Exploiting spatial-temporal relationships for 3D pose estimation via graph convolutional networks），" 见于 *IEEE/CVF 国际计算机视觉会议论文集（Proc. IEEE/CVF Int. Conf. Comput. Vis.）*，2019 年，第 2272–2281 页。

[32] Y. Cheng、B. Yang、B. Wang、W. Yan 和 R. T. Tan，"视频中三维人体姿态估计的遮挡感知网络（Occlusion-aware networks for 3D human pose estimation in video），" 见于 *IEEE/CVF 国际计算机视觉会议论文集（Proc. IEEE/CVF Int. Conf. Comput. Vis.）*，2019 年，第 723–732 页。

[33] J. Wang、S. Yan、Y. Xiong 和 D. Lin，"视频中运动引导的三维姿态估计（Motion guided 3D pose estimation from videos），" 见于 *欧洲计算机视觉会议论文集（Proc. Eur. Conf. Comput. Vis.）*，2020 年，第 764–780 页。

[34] M. R. I. Hossain 和 J. J. Little，"利用时间信息进行三维人体姿态估计（Exploiting temporal information for 3D human pose estimation），" 见于 *欧洲计算机视觉会议论文集（Proc. Eur. Conf. Comput. Vis.）*，2018 年，第 68–84 页。

[35] T. Chen 等人，"基于骨骼姿态分解的解剖学感知三维人体姿态估计（Anatomy-aware 3D human pose estimation with bone-based pose decomposition），" *IEEE 电路与系统视频技术汇刊（IEEE Trans. Circuits Syst. Video Technol.）*，第 32 卷，第 1 期，第 198–209 页，2022 年 1 月。

[36] R. Liu 等人，"注意力机制利用时间上下文：实时三维人体姿态重建（Attention mechanism exploits temporal contexts: Real-time 3D human pose reconstruction），" 见于 *IEEE/CVF 计算机视觉与模式识别会议论文集（Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.）*，2020 年，第 5064–5073 页。

[37] W. Hu、C. Zhang、F. Zhan、L. Zhang 和 T.-T. Wong，"用于三维人体姿态估计的条件有向图卷积（Conditional directed graph convolution for 3D human pose estimation），" 见于 *第 29 届 ACM 国际多媒体会议论文集（Proc. 29th ACM Int. Conf. Multimedia）*，2021 年，第 602–611 页。

[38] L. Yuan 等人，"Tokens-to-token ViT：在 ImageNet 上从头训练视觉 Transformer（Tokens-to-token ViT: Training vision transformers from scratch on ImageNet），" 见于 *IEEE/CVF 国际计算机视觉会议论文集（Proc. IEEE/CVF Int. Conf. Comput. Vis.）*，2021 年，第 558–567 页。

[39] X. Chu、B. Zhang、Z. Tian、X. Wang 和 C. Shen，"视觉 Transformer 真的需要显式位置编码吗（Do we really need explicit position encodings for vision transformers），" 见于 *第 11 届国际学习表征会议（Proc. 11th Int. Conf. Learn. Representations）*，2023 年。[在线]。获取地址：https://openreview.net/forum?id=3KWnuT-R1bh

[40] K. Han 等人，"Transformer in Transformer，" 见于 *神经信息处理系统年度会议论文集（Proc. Annu. Conf. Neural Inf. Process. Syst.）*，2021 年，第 15908–15919 页。

[41] W. Wang 等人，"金字塔视觉 Transformer：无卷积的密集预测通用骨干网络（Pyramid vision transformer: A versatile backbone for dense prediction without convolutions），" 见于 *IEEE/CVF 国际计算机视觉会议论文集（Proc. IEEE/CVF Int. Conf. Comput. Vis.）*，2021 年，第 568–578 页。

[42] H. Touvron 等人，"通过注意力进行数据高效图像 Transformer 训练与蒸馏（Training data-efficient image transformers & distillation through attention），" 见于 *国际机器学习会议论文集（Proc. Int. Conf. Mach. Learn.）*，2021 年，第 10347–10357 页。

[43] J. Zhang、Z. Tu、J. Yang、Y. Chen 和 J. Yuan，"MixSTE：用于视频中三维人体姿态估计的 Seq2seq 混合时空编码器（MixSTE: Seq2seq mixed spatio-temporal encoder for 3D human pose estimation in video），" 见于 *IEEE/CVF 计算机视觉与模式识别会议论文集（Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.）*，2022 年，第 13232–13242 页。

[44] W. Li、H. Liu、H. Tang、P. Wang 和 L. V. Gool，"MHFormer：用于三维人体姿态估计的多假设 Transformer（MHFormer: Multi-hypothesis transformer for 3D human pose estimation），" 见于 *IEEE/CVF 计算机视觉与模式识别会议论文集（Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.）*，2022 年，第 13147–13156 页。

[45] M. Hassanin、A. Khamiss、M. Bennamoun、F. Boussaid 和 I. Radwan，"CrossFormer：用于三维人体姿态估计的交叉时空 Transformer（CrossFormer: Cross spatio-temporal transformer for 3D human pose estimation），" 2022 年，arXiv:2203.13387。

[46] J. L. Ba、J. R. Kiros 和 G. E. Hinton，"层归一化（Layer normalization），" 2016 年，arXiv:1607.06450。

[47] C. Ionescu、D. Papava、V. Olaru 和 C. Sminchisescu，"Human3.6M：自然环境中三维人体感知的大规模数据集与预测方法（Human3.6M: Large scale datasets and predictive methods for 3D human sensing in natural environments），" *IEEE 模式分析与机器智能汇刊（IEEE Trans. Pattern Anal. Mach. Intell.）*，第 36 卷，第 7 期，第 1325–1339 页，2014 年 7 月。

[48] D. Mehta 等人，"使用改进 CNN 监督的野外单目三维人体姿态估计（Monocular 3D human pose estimation in the wild using improved CNN supervision），" 见于 *IEEE 国际三维视觉会议论文集（Proc. IEEE Int. Conf. 3D Vis.）*，2017 年，第 506–516 页。

[49] J. Xu 等人，"单目三维人体姿态估计的深度运动学分析（Deep kinematics analysis for monocular 3D human pose estimation），" 见于 *IEEE/CVF 计算机视觉与模式识别会议论文集（Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.）*，2020 年，第 899–908 页。

[50] A. Zeng 等人，"SRNet：通过分离与重组方法改进三维人体姿态估计的泛化能力（SRNet: Improving generalization in 3D human pose estimation with a split-and-recombine approach），" 见于 *第 16 届欧洲计算机视觉会议论文集（Proc. 16th Eur. Conf. Comput. Vis.）*，2020 年，第 507–523 页。

[51] A. Zeng 等人，"学习骨骼图神经网络用于困难三维姿态估计（Learning skeletal graph neural networks for hard 3D pose estimation），" 见于 *IEEE/CVF 国际计算机视觉会议论文集（Proc. IEEE/CVF Int. Conf. Comput. Vis.）*，2021 年，第 11436–11445 页。

[52] A. Paszke 等人，"PyTorch：命令式风格的高性能深度学习库（PyTorch: An imperative style, high-performance deep learning library），" 见于 *神经信息处理系统年度会议论文集（Proc. Annu. Conf. Neural Inf. Process. Syst.）*，第 32 卷，第 8024–8035 页，2019 年。

[53] D. P. Kingma 和 J. Ba，"Adam：一种随机优化方法（Adam: A method for stochastic optimization），" 见于 *第三届国际学习表征会议论文集（Proc. 3rd Int. Conf. Learn. Representations）*，美国加利福尼亚州圣迭戈，2015 年 5 月 7–9 日。

[54] W. Shan、H. Lu、S. Wang、X. Zhang 和 W. Gao，"通过相对信息编码改进三维人体姿态估计的鲁棒性与准确性（Improving robustness and accuracy via relative information encoding in 3D human pose estimation），" 见于 *第 29 届 ACM 国际多媒体会议论文集（Proc. 29th ACM Int. Conf. Multimedia）*，2021 年，第 3446–3454 页。

[55] J. Lin 和 G. H. Lee，"基于深度视频的三维人体姿态估计的轨迹空间分解（Trajectory space factorization for deep video-based 3D human pose estimation），" *BMVC*，2019 年。

   授权许可使用，仅限于：中南民族大学。下载时间：2026年10月01日 03:36:18 UTC，来自 IEEE Xplore。受限制使用。
