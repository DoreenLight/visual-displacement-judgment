# TokenHMR：基于令牌化姿态表示推进人体网格恢复

Sai Kumar Dwivedi<sup>1,\*</sup> Yu Sun<sup>2,∗</sup> Priyanka Patel<sup>1</sup> Yao Feng<sup>1,2,3</sup> Michael J. Black<sup>1</sup> <sup>1</sup>Max Planck Institute for Intelligent Systems, 图宾根，德国 <sup>2</sup>Meshcapade <sup>3</sup>ETH Zurich

![](images/17dfb28783440a33baf877a52ada3d96afb026531472c0b7ce404787690a5a17.jpg)  
图 1. 现有从图像中回归 3D 人体姿态与形状（HPS）的方法（如 HMR2.0 [12]）所估计的人体要么与图像对齐，要么具有准确的 3D 姿态，但无法兼得。我们表明这对现有方法是一个基本的权衡。为解决此问题，我们的方法 TokenHMR 提出了一种新颖的损失——阈值自适应损失缩放（Threshold-Adaptive Loss Scaling, TALS），以及一种基于离散令牌的 3D 姿态表示。凭借这些，TokenHMR 在多个真实场景下的 3D 基准上达到了最先进的准确率。

## 摘要

我们研究从单张图像回归 3D 人体姿态与形状的问题，重点关注 3D 准确性。当前最优方法利用大规模的 3D 伪真值（pseudo-ground-truth, p-GT）和 2D 关键点数据集，取得了强健的性能。然而，我们观察到一种矛盾的现象：随着 2D 准确度的提升，3D 姿态准确度反而下降。这是由 p-GT 中的偏差以及近似相机投影模型的使用所导致的。我们量化了当前相机模型所引入的误差，并表明精确拟合 2D 关键点和 p-GT 会导致错误的 3D 姿态。我们的分析定义了使 2D 和 p-GT 损失最小化会产生不利影响的无效距离范围。基于此，我们提出了一种新的损失——"阈值自适应损失缩放"（TALS），它对较大的 2D 和 p-GT 误差进行惩罚，但忽略较小的误差。然而，在这种损失下，可能存在许多 3D 姿态同样能解释 2D 证据。为了减少这种歧义，我们需要一个关于有效人体姿态的先验，但这样的先验可能引入不必要的偏差。为应对此挑战，我们利用人体姿态的令牌化表示，将问题重新表述为令牌预测。这将估计的姿态限制在有效姿态的空间内，有效提升了对遮挡的鲁棒性。在 EMDB 和 3DPW 数据集上的大量实验表明，我们重新设计的损失和令牌化方法使我们能够在真实场景数据上进行训练，同时在 3D 准确度上超越现有最先进水平。我们的模型和代码已在 https://tokenhmr.is.tue.mpg.de 开放以供研究使用。

## 1. 引言

我们研究从单张图像回归 3D 人体姿态与形状（HPS）的问题。现有方法 [3, 12, 32, 49] 在该任务上的准确度不断提高。

所谓准确，我们指的是两个方面：方法应当能够正确回归 3D 姿态，同时应当与图像证据保持一致。不幸的是，当前的方法无法同时兼顾这两点。我们观察到一个看似矛盾的现象：方法在拟合 2D 关键点上越准确，其对 3D 姿态的预测精度反而越低。我们对这一问题进行了研究，并识别出常见的弱透视相机假设是主要症结所在。该相机模型与采集图像时所使用的真实相机不匹配，因此投影得到的 3D 关节点与检测出的 2D 关节点之间存在偏差。由于目前尚无可靠的方法能够从单张图像中估计相机参数，我们对这一影响进行了研究和量化，并提出了两种解决方案来应对该问题。

具体而言，我们使用了合成的 BEDLAM [3] 数据集，该数据集具有完美的 3D 和 2D 真值（GT）。我们使用 [12] 中的相机模型将 3D 数据投影到 2D，以量化最佳情况下的 2D 误差，结果表明该误差很大。随后，我们还从反方向验证了较小的 2D 误差也可能导致较大的 3D 误差。即使采用 [32] 中那样的完整透视模型也无法解决问题，因为我们缺少精确的相机内参和外参。

该分析凸显了使用 2D 关键点损失来监督 3D 姿态回归所存在的问题。然而，这类损失能够让我们利用大规模数据集，从而带来泛化性和鲁棒性。不幸的是，伪真值（pseudo ground-truth，p-GT）训练数据也存在同样的问题，因为它是使用近似的相机模型，通过优化将 3D 人体拟合到 2D 数据上生成的。我们如何才能在缓解 3D 精度下降的同时，利用大规模、真实场景数据集中丰富的信息？我们对这一问题的回答是 TokenHMR，这是一种新的人体姿态（HPS）回归方法，它在有效利用 2D 关键点的同时保持了 3D 姿态的准确性，从而能够在相机参数未知的情况下利用现有的互联网数据。

TokenHMR 主要包含两个组件。第一个组件基于我们的核心洞察：来自 2D 关键点的监督虽然存在缺陷，但对于防止出现严重错误的预测具有重要价值。然而，过度依赖 2D 线索会引入偏差。为解决这一问题，我们定义了一种新的损失，称为阈值自适应损失缩放（Threshold-Adaptive Loss Scaling，TALS），该损失会对较大的 2D 和 p-GT 误差进行惩罚，而对较小的误差仅施加极轻微的惩罚。我们利用在 BEDLAM 数据集上的分析来定义该损失，从而避免网络在拟合 2D 关键点时追求超出相机模型所能支撑的精度。

然而，这又产生了一个新问题。从二维关键点预测三维姿态从根本上来说具有歧义性。当放宽关键点匹配约束时，会有更多三维姿态与二维数据相吻合。为了控制这一情况，需要引入一个先验，使网络倾向于输出有效姿态。遗憾的是，现有的基于高斯混合模型（mixture of Gaussians）[4] 或 VAE [42] 的姿态先验，会偏向于训练数据中频繁出现的姿态。相反，我们寻求的是一种无偏的先验，它仅限制网络输出有效姿态，但不偏向于任何特定姿态。

这引出了 TokenHHR 的第二个关键组件，这也是该方法名称的由来。具体而言，我们通过将人体姿态进行分词，将连续姿态回归问题转化为 token 预测问题。我们使用向量量化 VAE（VQ-VAE）[53] 在 AMASS [37] 和 MOYO [52] 等大规模运动捕捉数据集上进行预训练，将连续人体姿态离散化。这种分词表示为回归器提供了一个有效姿态的"词典"，从而将姿态先验有效地表示为一个知识库，即码本（codebook）。由于 VQ-VAE 被设计为表示均匀先验，我们认为这能够减少之前姿态先验所带来的偏差。

TokenHMR 通过分类生成离散 token，而非回归连续姿态。当我们使用一个当前最优（SOTA）的 HPS 方法，并将连续姿态替换为我们的分词姿态方法时，在三维精度上观察到稳定的提升（其他条件保持不变）。

我们进行了大量实验，以评估不同的姿态分词方式及其对精度的影响。任何对姿态的离散化都会在一定程度上造成精度损失。在我们的方法中，这导致了约 2.5 毫米的三维精度损失，相较于现有最优 HPS 回归器在真实数据上的精度，这一损失小 20 倍；也就是说，由分词带来的精度损失可以忽略不计。

最后，我们将这两个组件结合起来，发现它们具有协同效应。我们的新损失函数不会扭曲三维姿态以过拟合关键点或 p-GT，而分词化则防止网络为追求二维精度而扭曲三维姿态。通过这一组合，我们在三维精度方面取得了新的当前最优水平。我们在 EMDB [23] 和 3DPW [55]（这两个数据集具有准确的三维真值）上对 TokenHMR 及其他近期方法进行了广泛的评估。使用相同的数据和骨干网络，在具有挑战性的 EMDB 数据集上，TokenHMR 相比 HMR2.0 [12] 将三维误差降低了 7.6%。定性结果表明，TokenHMR 对模糊的图像证据具有鲁棒性，且估计出的姿态不存在使用 p-GT 和二维关键点的方法所常见的"弯膝"偏差（见图 1）。

总之，我们做出了以下重要贡献：(1) 三维精度退化分析：我们分析并量化了当前人体姿态与形状估计（Human Pose and Shape，HPS）方法在使用二维损失时，在三维与二维精度之间所面临的权衡关系。(2) 阈值自适应损失缩放：为了缓解该问题，我们提出了一种新颖的损失函数，能够减小低于因相机模型错误所导致的预期误差的二维误差和伪真值标签 (p-GT) 误差的影响。(3) 基于令牌（Token）的姿态表示：我们引入了一种基于令牌的姿态表示方法，并证明其能够产生更精确的姿态估计结果。我们的模型与代码已开源，可供研究使用。

## 2. 相关工作

## 2.1. HPS 回归

从单张图像中估计三维人体姿态与形状已得到广泛研究，从基于优化的方法到最新的基于 Transformer 的回归器均有所涉及。优化方法将参数化人体模型 [36, 42, 59] 拟合到二维图像线索上，包括但不限于关键点 [4, 42, 59]、轮廓 [41] 和部件分割 [30]。一些基于学习的方法直接从图像 [6, 8, 9, 21, 25, 32, 48, 49, 61] 和视频 [22, 24] 中估计参数化人体模型，另一些则以无模型方式估计人体形状，输出形式包括顶点 [28, 34, 46] 或隐式形状 [39, 45, 58]。近期方法 [12, 33] 采用 Transformer 估计三维人体，并取得了当前最佳的精度。为了应对泛化方面的挑战，EFT [20]、NeuralAnnot [40]、HMR2.0 [12] 和 CLIFF [32] 等近期方法在训练损失中使用了二维关键点和伪真值标签 (p-GT)，以在投影后的人体与图像之间产生良好的对齐效果。HuManiFlow [47] 和 POCO [9] 等方法对人体姿态与形状估计 (HPS) 进行概率建模，以显式地处理姿态模糊性问题。在追求更好的二维对齐过程中出现的三维精度退化问题已被注意到，但在我们的工作之前尚未得到广泛的量化分析。我们的统计分析揭示了现有 HMR 方法中的这一偏差，为该问题的训练策略提供了新的视角。

一些方法 [26, 32, 56] 通过从单张图像估计相机参数来解决三维到二维的投影误差问题。SPEC [26] 使用网络预测相机参数，但泛化能力较差；而 CLIFF [32] 则通过向网络提供图像中人物的包围框坐标信息来近似相机参数。由于从单张图像估计相机参数是高度不适定的（ill-posed），这仍然是一个具有挑战性且尚未解决的问题。我们的方法减小了使用错误相机模型的影响，并且可应用于任何 HPS 回归方法。

## 2.2. 姿态先验

人体姿态先验在多种应用中发挥着关键作用，例如将二维姿态提升到三维 [4, 42]，以及从图像/视频中估计人体姿态 [20, 29]。早期的姿态先验侧重于学习关节限位 [1] 以避免出现不可能的姿态。高斯混合模型（Gaussian Mixture Models，GMMs）[4] 和生成对抗网络（Generative Adversarial Networks，GANs）[11, 21] 也被用于在训练期间施加先验知识。一些近期方法使用变分自编码器（VAE）[42] 和归一化流（normalizing flows）[29] 作为先验。这些方法中的许多都偏向于常见出现的姿态，并将这种偏差传递给回归器。Pose-NDF [51] 等方法学习由神经隐式函数的零水平集所表示的合理姿态流形。从无效姿态到有效姿态的映射涉及梯度下降，将其集成到人体姿态与形状估计（HPS）训练中时计算开销较大。与之前的工作相比，我们学习了一个基于离散令牌（token）的有效 SMPL 姿态先验，减少了姿态偏差并提高了对遮挡的鲁棒性，同时易于集成到 HPS 训练中。

![](images/21b1922141d58f0d6c396b6960cd665fb7d04a4062dca24dcb9ae5222eb4ed71.jpg)  
图 2. 相机/姿态偏差问题的可视化。（a）焦距不正确导致 HMR2.0 等方法将透视缩短的腿估计为弯曲的。（b）用真值姿态替换预测的身体姿态可揭示相机偏差；（c）在保持二维对齐的情况下，三维姿态究竟能偏差多大？详见第 3 节。

我们使用 VQ-VAE [53]（VAE 的一种变体），通过量化三维训练姿态来学习一个离散化先验，这一过程称为"令牌化"（**Tokenization**），由此构建一个知识库，即码本（codebook）。令牌化在多种应用中被广泛使用，如图像合成 [44, 53]、文本到图像生成 [43]、二维人体姿态估计 [10] 以及学习运动先验 [18, 62]。在人体姿态估计的背景下，令牌化仍相对未被充分探索，尽管它在人体运动生成中已被广泛采用 [14, 62]。当然，图像和语言的令牌化表示已广泛用于众多视觉和语言任务中。我们方法的新颖之处在于，将回归问题重新表述为姿态令牌分类问题，从而利用令牌化来表示有效姿态，提供了一种有效的姿态先验。

## 3. 相机/姿态偏差

估计 3D 人体姿态与形状 (HPS) 的方法通常试图满足两个目标：准确的 3D 姿态和与 2D 图像特征的精确对齐。不幸的是，我们在所有实验中观察到一个趋势——方法在 2D 误差上表现越好，在 3D 上表现就越差，反之亦然。其关键原因在于，包括本文所测试的方法在内的现有方法，都不估计相机内参（例如焦距）或相机外参（旋转和平移）。相反，现有方法使用缩放正交投影或使用固定且不正确的相机参数进行透视投影，在相机坐标系中估计人体。这导致真实的 3D 关节点与其 2D 投影之间不匹配。具体而言，由于照片通常从大致与眼睛齐平的高度拍摄，因此腿部比上半身更远，从而产生透视收缩效应。训练模型以最小化 2D 误差会迫使它们在 3D 中生成不正确的姿态；如图 2 (a) 所示。2D 姿态数据集的伪真值 (p-GT) 是通过使用有问题的相机参数最小化 2D 误差获得的。完全信任此类 p-GT 并追求对此类标注的精确学习，会使问题更加突出。请注意，透视收缩使腿部在图像中显得更短。唯一使人体适应这种情况的方法是在膝部弯曲腿部或在 3D 中倾斜身体，使腿部距离更远。这会产生不自然或不稳定的姿态。

这是所有现有方法的一个根本性问题，在不知道相机参数的情况下，无法同时获得较低的 3D 和 2D 误差。为了定量评估这种不匹配的影响，我们采用 BEDLAM [3]，一个合成数据集，其中 3D 和 2D 数据以及相机参数都是已知的。这消除了任何可能的噪声，并使我们能够观察使用错误相机对 2D 投影误差的影响。具体而言，如图 2 (b) 所示，我们使用 HMR2.0 [12] 的相机将真值 BEDLAM 人体投影到图像中。

我们使用标准度量——正确关键点百分比 (PCK) 来评估错误相机对 2D 的影响，并针对 BEDLAM 验证集的一个序列进行计算。HMR2.0b 计算的 3D 人体在 PCK0.5 上有 0.78 的误差，在 PCK1.0 上有 0.88 的误差。相比之下，当我们使用 HMR2.0b 的相机配合真值 3D 人体时，PCK 分数下降到 PCK0.5 为 0.66，PCK1.0 为 0.86。理想情况下，使用正确的相机模型，PCK0.5 和 PCK1.0 都应达到 1.0。HMR2.0b 取得比真值更低的误差这一事实表明，其输出由于相机偏差而偏离了真实的 3D 姿态和形状。这证明，像 HMR2.0b 这样的方法虽然获得了较高的 PCK 值，但这是以牺牲 3D 精度为代价的。总之，除非拥有正确的相机模型，否则追求高 PCK 值对 3D 精度是有害的。

我们进一步设计实验来探究在保持良好 2D 对齐的前提下，3D 误差可以大到什么程度。我们修改了 SMPLify [4] 的损失函数，使预测的 2D 关键点与真值 2D 关键点 $\scriptstyle J _ { 2 D _ { g } }$ 之间的距离保持接近，同时加入一个新的损失项以增大预测的 3D 关键点 $J _ { 3 D }$ 与真实 3D 关键点 $J _ { 3 D _ { g } }$ 之间的距离，如下式所示：

$$
w _ { 2 D } | | \Pi ( J _ { 3 D } , { \cal T } ) - J _ { 2 D _ { g } } | | _ { 2 } - w _ { 3 D } | | J _ { 3 D } - J _ { 3 D _ { g } } | | _ { 2 } + m\tag{1}
$$

其中 $\Pi$ 表示使用 HMR2.0 相机进行的 3D 到 2D 投影，$m = 20$ 是裕量值，$w _ { 2 D } = 4$ 和 $w _ { 3 D } = 40.5$ 是标量权重。经过 100 次迭代优化后，平均关节点位置误差（MPJPE）达到 146 mm。如图 2 (c) 所示，即便在深度方向上存在显著误差，投影后的 3D 姿态仍能保持较高程度的 2D 对齐。当优化进行 200 次迭代时，MPJPE 超过 300 mm，并且随着进一步优化，误差持续增大。

由于该领域目前还没有一种从单张图像可靠估计相机参数的方法，下面我们探究所提新方法（TALS 和 tokenization）是否有助于缓解由近似相机模型所引发的问题。图 2 (a) 比较了 HMR2.0 与 TokenHMR 的结果。可以看出，TokenHMR 的姿态受透视缩短（foreshortening）效应的影响较小。

## 4. TokenHMR

## 4.1. 预备知识

我们的方法 TokenHMR 接受输入图像 $I$，并输出身体姿态 $\theta$、形状 $\beta$ 以及透视相机参数 $T$。我们使用 SMPL [36]，一种可微的参数化人体模型。其输入参数包括姿态 $\theta \in \mathbb { R } ^ { 7 2 }$ 和形状 $\beta \in \mathbb { R } ^ { 1 0 }$。作为输出，它生成人体网格 $M$ 以及顶点 $V \in \mathbb { R } ^ { N \times 3 }$，其中 $N = 6 8 9 0$ 是顶点数。3D 关节点 $J _ { 3 D }$ 通过使用预训练的关节点回归器对网格顶点进行线性组合得到。

## 4.2. 阈值自适应损失缩放：TALS

在第 3 节中，我们的分析揭示了使用伪真值（pseudo-ground-truth）和 2D 关键点进行有效学习的一个显著障碍——相机/姿态偏差。尽管存在这一挑战，这些标注所提供的尺度信息对于实现最佳的泛化性和鲁棒性仍然是不可或缺的。我们断言，当得到恰当利用且不过拟合时，这些标注显著增强了模型鲁棒估计姿态的能力。从我们的观察中得出一个关键见解：确立一个有效阈值对于甄别哪些误差水平作为训练信号已无额外助益是必要的。当损失超过该阈值时，传统的学习机制引导姿态估计；反之，当损失低于该有效阈值时，我们将其影响最小化，以防止对相机/姿态偏差的过拟合。

为确定该有效阈值，我们利用真值（GT）3D 姿态和一个标准的（不正确的）相机模型分析所得误差。我们再次利用 BEDLAM [3] 中的 3D

![](images/48ce85fed892eadd5454f60aaaf9d65a34d855c64822fdbb56e43515b67ec2a4.jpg)  
图 3. 框架概览。我们的方法包含两个阶段。(a) 在分词化（tokenization）步骤中，编码器学习将连续姿态映射为离散的姿态标记（token），而解码器则尝试重构原始姿态。(b) 为训练 TokenHMR，我们使用预训练的解码器提供的"有效姿态词汇表"，将回归替换为分类。

## 4.3. 分词化（Tokenization）

真值，这次分别为 2D 关键点和 SMPL 伪真值确立有效阈值。对于 2D 关键点，我们用 BEDLAM 中的真值替换预测的 SMPL 参数，以获得在 HMR2.0 相机下真实 3D 人体的 2D 关键点投影。然后，我们计算这些投影与 GT 2D 关键点之间的平均 L1 范数，并以此作为 2D 关键点监督的阈值 $\varepsilon J_{2D}$。我们相对于图像宽度对这些 2D 关键点进行归一化，并将数值缩放至 -0.5 至 0.5 之间，以减轻尺度相关的差异。

类似地，为 SMPL p-GT 确立有效监督阈值，我们开展了额外的实验。借助 p-GT，我们将姿态损失以关节角度误差的形式加以表述。为对这些误差设定合适的阈值，我们评估了在 BEDLAM 上 HMR2.0 的预测与 SMPL 模型中每个关节的真值之间的关节角度差异。具体而言，我们计算 HMR2.0 预测的 SO(3) 旋转流形上的 3D 关节旋转与 BEDLAM 中的真值旋转之间的平均测地距离<sup>\*</sup>。具体的阈值数值请参见补充材料（Sup. Mat.）的 B.2 节。

在为 2D 关键点和 SMPL p-GT 确立有效阈值之后，我们引入了一种称为阈值自适应损失缩放（Threshold-Adaptive Loss Scaling，TALS）的新损失函数。它仅在损失低于阈值时对其进行缩放。具体而言，用于 p-GT 姿态和 2D 关节的 TALS 损失项定义为

$$
\mathcal { L } _ { \theta _ { p G T } } = \left\{ \begin{array} { l l } { \Vert \pmb { \theta } - \pmb { \theta _ { g } } \Vert ^ { 2 } } & { \mathrm { i f } \ \mathcal { L } _ { \theta _ { p G T } } > \varepsilon _ { \theta } } \\ { \alpha _ { \theta } \cdot \Vert \pmb { \theta } - \pmb { \theta _ { g } } \Vert ^ { 2 } } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{2}
$$

$$
\mathcal { L } _ { J _ { 2 D _ { p G T } } } = \left\{ \begin{array} { l l } { | J _ { 2 D } - J _ { 2 D _ { g } } | } & { \mathrm { i f ~ } \mathcal { L } _ { J _ { 2 D _ { p G T } } } > \varepsilon _ { J _ { 2 D } } } \\ { \alpha _ { J _ { 2 D } } \cdot | J _ { 2 D } - J _ { 2 D _ { g } } | } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{3}
$$

其中 $\alpha _ { J _ { 2 D } }$ 和 $\alpha _ { \theta }$ 是小的标量乘子，$\varepsilon _ { \theta }$ 是为每个姿态参数单独计算的阈值，$\varepsilon _ { J _ { 2 D } }$ 是 2D 关节点的阈值。

我们使用 VQ-VAE [53]，它学习 3D 姿态的离散表示。具体来说，我们为 SMPL 身体参数学习一个离散表示，$\theta = [ \theta _ { 1 } , \theta _ { 2 } , \ldots , \theta _ { 2 1 } ]$，其中 $\theta _ { i }$ 表示 $\mathbb { R } ^ { 6 }$ 中每个关节的姿态参数。该过程涉及使用自编码器架构和可学习码本对姿态参数进行编码和解码，记为 $C B = \{ c _ { k } \} _ { k = 1 } ^ { K }$，每个码 $c _ { k } \in \mathbb { R } ^ { d _ { c } }$，其中 $d _ { c }$ 是码的维度。姿态 VQ-VAE 的整体架构如图 3 (a) 所示。自编码器的编码器和解码器分别用 $E$ 和 $D$ 表示。编码器负责生成离散姿态标记，而解码器将这些标记重构回 SMPL 姿态。隐特征 $z$ 可以通过 $z = E ( \theta }$ 计算，得到 $z = [ z _ { 1 } , z _ { 2 } , \dots , z _ { M } ]$，其中 $z _ { i } \in \mathbb { R } ^ { d _ { c } }$，$M$ 是标记的数量。每个隐特征 $z _ { i }$ 通过在码本 $C B$ 中寻找最相似的码元素进行量化，如下式所示：

$$
\hat { z } _ { i } = \underset { c _ { k } \in C B } { \arg \operatorname* { m i n } } \| z _ { i } - c _ { k } \| _ { 2 } .\tag{4}
$$

在训练姿态分词器时，我们采用与先前工作 [53] 相似的策略，该策略涉及三个主要的损失函数来优化分词器：重构损失 $( \mathcal { L } _ { \mathcal { R } \mathcal { E } } )$、嵌入损失 $( \mathcal { L } _ { \mathcal { E } } )$ 和承诺损失 $( \mathcal { L } _ { \mathcal { C } } )$。总体损失 $( \mathcal { L } _ { \mathcal { V } \mathcal { Q } } )$ 定义为

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathcal { V } \mathcal { Q } } = \lambda _ { \mathcal { R } \mathcal { E } } \mathcal { L } _ { \mathcal { R } \mathcal { E } } + \lambda _ { \mathcal { E } } \mathcal { L } _ { \mathcal { E } } + \lambda _ { \mathcal { C } } \mathcal { L } _ { \mathcal { C } } } \\ & { \quad \quad \quad = \lambda _ { \mathcal { R } \mathcal { E } } \mathcal { L } _ { \mathcal { R } \mathcal { E } } + \lambda _ { \mathcal { E } } \vert \vert s g [ z ] - e \vert \vert _ { 2 } + \lambda _ { \mathcal { C } } \vert \vert z - s g [ e ] \vert \vert _ { 2 } } \end{array}\tag{5}
$$

其中 $sg$ 是停止梯度算子（stop gradient operator），$e$ 是来自码本（codebook）的嵌入，$\lambda _ { \mathcal { R } \mathcal { E } } , \lambda _ { \mathcal { E } } , \lambda _ { \mathcal { C } }$ 是各项的超参数。对于重建任务，我们使用 $\mathcal { L } _ { \mathcal { 1 } }$ 损失来衡量真实姿态 $\theta _ { g }$ 与预测姿态 $\theta$ 之间的误差，以及 SMPL 真实 3D 关节点 $J _ { 3 D _ { g } }$ 与预测关节点 $J _ { 3 D }$ 之间的误差。因此，$\mathcal { L } _ { \mathcal { R } \mathcal { E } }$ 损失定义如下：

$$
\begin{array} { r } { \mathcal { L } _ { \mathcal { R E } } = \mathcal { L } _ { 1 } ( \theta _ { g } , \theta ) + \mathcal { L } _ { 1 } ( J _ { 3 D _ { g } } , J _ { 3 D } ) . } \end{array}\tag{6}
$$

原始的 VQ-VAE 存在码本坍缩（codebook collapse）问题，即码本未被充分利用。参照先前工作 [62]，我们采用指数移动平均（exponential moving average，EMA）和码本重置（Code Reset）的训练策略，以实现更好的利用率。

## 4.4. 架构

我们的架构利用了 Vision Transformer（ViT）[7]，与 HMR2.0 [12] 类似。输入图像 $I$ 首先被转换为输入 token，随后由 Transformer 处理以生成输出 token。这些输出 token 随后在 Transformer 解码器中进一步处理。Transformer 解码器具有多头自注意力机制，通过将零输入 token 与图像输出 token 进行交叉注意力，从而从 Transformer 模块中获取特征。与使用三个线性层将 Transformer 模块的特征映射到 SMPL 姿态 $\theta$、形状 $\beta$ 和相机 $T$ 的 HMR2.0 [12] 不同，我们提出了一种新颖的方法。我们的目标是利用在大量运动捕捉（mocap）数据上训练得到的 tokenizer，特别侧重于人体姿态。为此，我们将 SMPL 姿态参数划分为身体姿态和全局朝向。我们使用单独的线性层从 tokenizer 预测全局朝向和身体姿态。

一种直接的分词器（tokenizer）集成方法将直接从 ViT 变换器骨干网络估计码本索引（code index），并基于该索引从码本 CB 中选取嵌入 $e$。然而，这一做法存在挑战，因为从码本中选取嵌入的过程是不可微的。为解决该问题，我们采用基于 logit 的方法。我们不为每个 token（词元）直接估计码本索引，而是输出 logit 值 $Q$。这些 logit 与码本相乘，得到加权嵌入。因此，近似量化特征 $\hat{z} = [\hat{z}_1, \hat{z}_2, ..., \hat{z}_M]$ 可由下式计算得到：

$$
\bar{z} = \sigma(Q_{\mathcal{M} \times \mathcal{K}}) \times CB_{\mathcal{K} \times \mathcal{D}} \approx \hat{z} \tag{7}
$$

其中，$Q$ 是由骨干网络估计的 logit 值，$\sigma$ 是 softmax 操作，CB 是预训练的码本，$M$ 是 token 的数量，$K$ 是码本中的条目数，$D$ 是每个码本条目的维度。该操作使过程变得可微。随后，将得到的近似量化特征 $\bar{z}$ 输入到分词器解码器中，该过程输出最终的姿态。在 TokenHMR 的训练过程中，学习到的分词器解码器被冻结，以利用其从动捕（mocap）数据中学习到的先验知识。

## 4.5. 损失函数

沿用先前工作 [12, 25] 的做法，我们在 2D 和 3D 关节以及 SMPL 姿态和形状参数上进行损失计算，即分别在 $J_{2D}$、$J_{3D}$、$\theta$、$\beta$ 上计算。然而，依据第 4.2 节的分析，我们对 2D 和 3D 数据集的数据区别对待。对于带有 3D 真值标注的数据集，我们将标准损失定义为

$$
\begin{array}{r} \mathcal{L}_{GT} = \lambda_{\theta} \mathcal{L}_{\theta}(\theta, \theta_g) + \lambda_{\beta} \mathcal{L}_{\beta}(\beta, \beta_g) + \\ \lambda_{3D} \mathcal{L}_{3D}(\mathbf{J}_{3D}, \mathbf{J}_{3D_g}) + \lambda_{2D} \mathcal{L}_{2D}(\mathbf{J}_{2D}, \mathbf{J}_{2D_g}) \end{array} \tag{8}
$$

其中，$\mathcal{L}_{\beta}$ 是 SMPL 形状损失，$\mathcal{L}_{J_{3D}}$ 是 3D 关节损失，$\mathcal{L}_{J_{2D}}$ 是关节重投影损失。$\lambda_{\beta}$、$\lambda_{3D}$ 和 $\lambda_{2D}$ 是各项的调控权重。为了从 SMPL 伪真值标注中学习，我们采用阈值自适应损失缩放（Threshold-Adaptive Loss Scaling, TALS），即根据第 4.2 节（公式 2 和 3）计算得到的阈值来缩放损失。因此，总损失定义为

$$
\mathcal{L}_{Total} = \mathcal{L}_{GT} + \mathcal{L}_{\theta_{pGT}} + \mathcal{L}_{J_{2D_{pGT}}}. \tag{9}
$$

## 5. 实验

## 5.1. 实现细节

TokenHMR 的训练包含两个阶段：首先，我们训练一个分词器（tokenizer），利用 AMASS [37] 和 MOYO [52] 动作捕捉数据来学习离散的姿态表示。然后，我们将分词器的预训练解码器用作回归身体姿态的额外预测头。在 TokenHMR 的训练过程中，分词器被冻结以利用其先验。

我们的分词器架构灵感来自 T2M-GPT [62]，但我们学习的是 SMPL 姿态参数的姿态标记，而不是 3D 关节的运动标记。我们在编码器和解码器中都使用 1 个 ResNet [15] 块和 4 个一维卷积。控制权重 $\lambda _ { \mathcal { R } \mathcal { E } } , \lambda _ { \mathcal { E } } , \lambda _ { \mathcal { C } }$ 分别设置为 50.0、1.0、1.0。模型训练了 150K 次迭代，批量大小为 256，学习率为 $2 e ^ { - 4 }$。为了训练一个鲁棒的模型，我们从 $1 e ^ { - 3 }$ 开始为随机关节添加噪声，并在每 5K 次迭代后逐步增加噪声。我们根据验证集上的重建误差，为 TokenHMR 选择了包含 160 个标记、码本（codebook）大小为 2048 × 256 的最佳分词器模型。

对于 TokenHMR，我们使用 ViT-H/16 [7] 作为骨干网络，并遵循 HMR2.0 [12] 使用标准的 Transformer 解码器 [54]。我们使用 4 个独立的线性层，将 Transformer 解码器输出的 1024 维特征分别映射到 SMPL 的全局朝向、手部姿态和身体形状，以及一个用于相机。然而，对于身体姿态，我们将 1024 维特征通过 4 个线性层块进行处理，每个块包含 2 个 MLP 和一个 GELU 激活函数 [16]。这给出了最终的 logits Q，大小为 160 × 2048，用于与大小为 $2 0 4 8 \times 2 5 6$ 的码本相乘，从而得到近似的量化特征 z¯；参见公式 4。我们使用 ViT-Pose [60] 作为预训练骨干网络。我们在 4 块 Nvidia RTX 6000 GPU 上训练 100K 次迭代，批量大小为 256，学习率为 $1 e ^ { - 5 }$，耗时约一天。控制权重 $\lambda _ { \theta } , \lambda _ { \beta } , \lambda _ { J _ { 2 D } } , \lambda _ { J _ { 3 D } }$ 分别设置为 $1 e ^ { - 3 } , 5 e ^ { - 4 }$，$1 e ^ { - 2 } , \bar { 5 } e ^ { - 2 }$。TALS 的损失权重对姿态 $\alpha \theta$ 和 2D 关键点 $\alpha _ { J _ { 2 D } }$ 均设置为 1%。

<table><tr><td rowspan="2">训练数据集</td><td rowspan="2">方法</td><td colspan="3">EMDB [23]</td><td colspan="3">3DPW [55]</td></tr><tr><td>MVE</td><td>MPJPE</td><td>PA-MPJPE</td><td>MVE</td><td>MPJPE</td><td>PA-MPJPE</td></tr><tr><td>SD</td><td>HybrIK [31]</td><td>122.2</td><td>103.0</td><td>65.6</td><td>94.5</td><td>80.0</td><td>48.8</td></tr><tr><td>SD</td><td>CLIFF [32]</td><td>122.9</td><td>103.1</td><td>68.8</td><td>81.2</td><td>69.0</td><td>43.0</td></tr><tr><td>SD</td><td>HMR2.0 [12]</td><td>120.1</td><td>97.8</td><td>61.5</td><td>84.1</td><td>70.0</td><td>44.5</td></tr><tr><td>BL</td><td>BEDLAM-CLIFF [3]</td><td>113.2</td><td>97.1</td><td>61.3</td><td>85.0</td><td>72.0</td><td>46.6</td></tr><tr><td>BL</td><td>HMR2.0</td><td>106.6</td><td>90.7</td><td>51.3</td><td>88.4</td><td>72.2</td><td>45.1</td></tr><tr><td>BL</td><td>TokenHMR</td><td>104.2</td><td>88.1</td><td>49.8</td><td>86.0</td><td>70.5</td><td>43.8</td></tr><tr><td>SD + ITW</td><td>HMR2.0 [12]</td><td>140.6</td><td>118.5</td><td>79.3</td><td>94.4</td><td>81.3</td><td>54.3</td></tr><tr><td>SD + ITW</td><td>TokenHMR</td><td>124.4</td><td>102.4</td><td>67.5</td><td>88.1</td><td>76.2</td><td>49.3</td></tr><tr><td>SD + ITW + BL</td><td>HMR2.0</td><td>120.7</td><td>99.3</td><td>62.8</td><td>88.4</td><td>77.4</td><td>47.4</td></tr><tr><td>SD + ITW + BL</td><td>HMR2.0 + TALS</td><td>115.7</td><td>96.7</td><td>58.5</td><td>89.6</td><td>73.5</td><td>46.8</td></tr><tr><td>SD + ITW + BL</td><td>HMR2.0 + Token</td><td>116.1</td><td>95.6</td><td>62.2</td><td>86.6</td><td>75.0</td><td>48.0</td></tr><tr><td>SD + ITW + BL</td><td>HMR2.0 + TALS + VPoser [42]</td><td>116.8</td><td>97.9</td><td>56.4</td><td>87.1</td><td>73.7</td><td>45.7</td></tr><tr><td>SD + ITW + BL</td><td>TokenHMR</td><td>109.4</td><td>91.7</td><td>55.6</td><td>84.6</td><td>71.0</td><td>44.3</td></tr></table>

表 1. EMDB 和 3DPW 数据集上的 3D 人体网格与姿态误差。详见正文。
<table><tr><td rowspan="2">方法</td><td colspan="3">裁剪 30%</td><td colspan="3">裁剪 50%</td></tr><tr><td>MVE</td><td>MPJPE</td><td>PA-MPJPE</td><td>MVE</td><td>MPJPE</td><td>PA-MPJPE</td></tr><tr><td>HMR2.0 [12]</td><td>135.24 (+14.98)</td><td>113.39 (+14.13)</td><td>70.68 (+7.86)</td><td>166.71 (+46.45)</td><td>137.88 (+38.59)</td><td>90.30 (+27.48)</td></tr><tr><td>TokenHMR</td><td>124.09 (+14.71)</td><td>104.72 (+13.01)</td><td>62.13 (+6.52)</td><td>150.29 (+40.91)</td><td>125.99 (+34.28)</td><td>78.88 (+23.27)</td></tr></table>

表 2. 在 EMDB 数据集上，从图像边界按不同比例均匀裁剪对 3D HPS 精度的影响。括号中的数字表示相对于未裁剪场景的性能变化；数值越小越好。此处比较的所有模型均采用相同的骨干网络，并在相同的数据上进行训练。

<table><tr><td>1</td><td>一</td><td colspan="2">AMASS [37]</td><td colspan="2">MOYO [52]</td></tr>
<tr><td></td><td>Method</td><td>MVE↓|</td><td>MPJPE↓</td><td>MVE↓</td><td>MPJPE↓</td></tr>
<tr><td rowspan="3">CB</td><td>1024 × 256</td><td>11.5</td><td>4.6</td><td>27.1</td><td>15.7</td></tr>
<tr><td>2048 × 128</td><td>9.4</td><td>3.1</td><td>22.5</td><td>12.3</td></tr>
<tr><td>2048 × 256</td><td>8.3</td><td>2.2</td><td>19.9</td><td>10.4</td></tr>
<tr><td rowspan="3">Tokekens</td><td>80</td><td>12.5</td><td>4.1</td><td>24.4</td><td>16.7</td></tr>
<tr><td>160</td><td>8.3</td><td>2.2</td><td>19.9</td><td>10.4</td></tr>
<tr><td>320</td><td>8.1</td><td>1.9</td><td>19.0</td><td>10.1</td></tr>
<tr><td rowspan="3">Noise</td><td>Yes</td><td>8.3</td><td>2.2</td><td>19.9</td><td>10.4</td></tr>
<tr><td>No</td><td>7.9</td><td>1.9</td><td>21.0</td><td>11.5</td></tr>
<tr><td colspan="2">AMASS + MOYO*</td><td>8.7</td><td>2.6</td><td>16.5</td><td>7.6</td></tr></table>

Table 3. Tokenizer 消融实验。除最后一行<sup>⋆</sup>（使用 MOYO 训练集进行训练）外，所有方法均在 AMASS [37] 的标准训练集上训练，并在 AMASS 的测试集和 MOYO [52] 的验证集上进行评估。最后一个模型被用作 TokenHMR 中的 tokenizer。

Training Data（训练数据）：对于 tokenizer 的训练，我们使用 AMASS [37] 的标准训练划分和 MOYO [52] 的训练数据。有关训练数据准备的更多细节，请参阅补充材料。按照先前的方法 [12, 25, 27]，我们使用标准数据集（SD）进行训练，包括 Human3.6M [17]、MPI-INF-3DHP [38]、COCO [35] 和 MPII [2]。此外，与 HMR2.0b 类似，我们也使用野外 2D 数据集（ITW），如 InstaVariety [22]、AVA [13] 和 AI Challenger [57] 数据集及其 p-GT（伪真值）进行训练。我们还引入了 BEDLAM (BL) [3]，这是一个具有精确 3D 真值的合成数据集。为了公平比较，我们使用 SD、ITW 和 BL 数据集的组合重新训练了 HMR2.0b。我们选择 HMR2.0b 作为基线模型，因为其代码是开源的，我们可以复现其结果。

Evaluation and Metrics（评估与指标）：对于 tokenizer 的精度，我们报告平均顶点误差（MVE）和平均关节位置误差（MPJPE），并在 AMASS 的标准测试划分和 MOYO 的验证集上进行评估。对于 TokenHMR，我们报告预测值与真值之间的平均顶点误差（MVE）、平均关节位置误差（MPJPE）以及 Procrustes 对齐后的平均关节位置误差（PA-MPJPE）。我们在 3DPW [55] 和 EMDB [23] 数据集的测试集上进行评估。前者是一个标准的 3D 数据集，后者是最近发布的更具挑战性的数据集，包含变化的相机运动和多样的 3D 姿态。

## 5.2. 如何缓解 3D 退化问题？

表 1 展示了仅使用 SD 数据集训练的 HMR2.0 模型的性能，以及同时使用 SD 和 ITW 数据集训练时的性能。我们观察到，在加入 ITW 数据后，模型在 EMDB 数据集上的 3D 准确率显著降低（超过 17%）。与此同时，根据 HMR2.0 [12] 论文中的表 2，ITW 数据提升了该方法的 2D 性能。应对这一趋势的一种直接方法是整合更多具有精确 3D 标注的数据，例如 BEDLAM [3]。然而，如表 1 所示，即使加入了 BEDLAM，HMR2.0（SD+ITW+BL）仍然遭受明显的 3D 指标退化。这一观察结果构成了我们进一步研究的基线。

![](images/ca47fabd852e2547842d778a54f18b8d9f0fb69647707a7365ed728e7b1993c1.jpg)  
Figure 4. Qualitative comparisons on challenging poses from the LSP [19] dataset.

采用我们新颖的损失函数形式（TALS），在 EMDB 和 3DPW 数据集上都带来了显著的性能提升，表明其在防止对含噪声的 p-GT 数据过拟合方面的有效性。尽管 TALS 带来了一定的改进，我们进一步深入探索姿态先验，以补偿削弱的监督信号。我们对 HPS 中广泛使用的 VAE 先验 VPoser [42] 的评估仅产生了边际改进，这表明需要一种更为强健的替代方案。我们基于 VQ-VAE 的姿态 token 化方法则带来了更大的提升。TokenHMR 显著优于 HMR2.0，在 EMDB 上 MVE 提升了 9%，MPJPE 提升了 7.6%，PA-MPJPE 提升了 11.5%。在 3DPW 上也观察到了类似的趋势。

## 5.3. How does the Token-based Prior Help?

除了如上所述促进更有效的学习之外，我们还考察了基于离散 token 的先验在图像信息不明确（例如截断）的场景中的有效性。我们在 EMDB 数据集上对图像施加不同程度的裁剪来评估我们的方法。具体而言，我们从图像边界分别裁剪 30% 和 50%。如表 2 所示，与 HMR2.0 [12] 相比，我们的方法在具有挑战性的截断设置下（50% 对比 30%）性能下降幅度更小。此外，如图 4 所示的定性结果进一步凸显了嵌入在我们基于 token 的姿态表示中的先验的鲁棒性。这种鲁棒性对于处理图像截断常见的真实场景至关重要。

## 5.4. Ablation Study of Tokenizer

表 3 展示了我们在 AMASS 标准测试集和 MOYO 验证集上对不同分词器（tokenizer）设计选择的消融研究。为了理解设计选择对分布外（out-of-distribution）MOYO 数据的影响，我们仅使用 AMASS 进行训练，并进行了多种消融实验。然而，最终的分词器模型（表 3 最后一行）用于 TokenHMR 中，并在 AMASS 和 MOYO 数据集上联合训练。我们的研究结果表明，码本（codebook）条目数的影响比编码维度更为显著。尽管 token 数量对于精确表示至关重要，但我们观察到性能存在平台期，因此在最终模型中选择了 160 个 token。该数量在 TokenHMR 的网络规模与重建精度之间取得了平衡。对姿态参数进行随机加噪增强能够构建出更加鲁棒的分词器，虽然略微降低了分布内数据的性能，但对分布外（OOD）数据有积极影响。

## 6. 结论

在本文中，我们提出了一种从单张图像进行三维人体姿态估计的新方法。我们首先识别并量化了使用不正确的相机模型配合二维关键点损失所带来的问题。这导致了当前方法面临的一个根本性权衡——要么具有较高的三维精度，要么具有较高的二维精度，但难以两者兼得。我们的方法 TokenHMR 通过两项易于被其他方法采用的贡献解决了该问题。TokenHMR 采用了一种基于回归人体姿态离散化表示的全新姿态估计范式。我们将其与一种新的损失 TALS 相结合，该损失缓解了由相机投影误差以及带偏 p-GT 所引起的一些偏差，同时仍允许使用野外（in-the-wild）训练数据。我们在 EMDB 和 3DPW 数据集上的实验表明，即使在存在严重遮挡的情况下，TokenHMR 在三维精度方面也显著优于 HMR2.0 等现有模型。

## 7. 致谢

我们衷心感谢感知系统部门（Perceiving Systems）以及 Meshcapade GmbH 的机器学习团队提出的深刻见解和宝贵反馈。我们感谢国际马普智能系统研究学校（IMPRS-IS）对 Sai Kumar Dwivedi 的支持。我们感谢 Meshcapade GmbH 对 Yu Sun 的支持以及所提供的 GPU 资源。本工作得到了德国联邦教育与研究部（BMBF）的部分资助：图宾根人工智能中心，项目编号 FKZ: 01IS18039B。利益冲突声明：https://files.is.tue.mpg.de/black/CoI CVPR 2024.txt

## 参考文献

[1] Ijaz Akhter and Michael J. Black. Pose-conditioned joint angle limits for 3D human pose reconstruction. In Computer Vision and Pattern Recognition (CVPR), 2015. 3

[2] Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler, and Bernt Schiele. 2d human pose estimation: New benchmark and state of the art analysis. In Computer Vision and Pattern Recognition (CVPR), 2014. 7, 1

[3] Michael J Black, Priyanka Patel, Joachim Tesch, and Jinlong Yang. **Bedlam**: A synthetic dataset of bodies exhibiting detailed lifelike animated motion. In *CVPR*, pages 8726–8737, 2023. 1, 2, 4, 5, 7, 8

[4] Federica Bogo, Angjoo Kanazawa, Christoph Lassner, Peter Gehler, Javier Romero, and Michael J. Black. Keep it **SMPL**: Automatic estimation of 3D human pose and shape from a single image. In *European Conference on Computer Vision (ECCV)*, 2016. 2, 3, 4

[5] Z. Cao, G. Hidalgo Martinez, T. Simon, S. Wei, and Y. A. Sheikh. **Openpose**: Realtime multi-person 2d pose estimation using part affinity fields. *IEEE Transaction on Pattern Analysis and Machine Intelligence (TPAMI)*, 2019. 1

[6] Hongsuk Choi, Gyeongsik Moon, JoonKyu Park, and Kyoung Mu Lee. Learning to estimate robust 3d human mesh from in-the-wild crowded scenes. In *Computer Vision and Pattern Recognition (CVPR)*, pages 1475–1484, 2022. 3

[7] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In *International Conference on Learning Representations (ICLR)*, 2021. 6

[8] Sai Kumar Dwivedi, Nikos Athanasiou, Muhammed Kocabas, and Michael J. Black. Learning to regress bodies from images using differentiable semantic rendering. In *International Conference on Computer Vision (ICCV)*, 2021. 3

[9] Sai Kumar Dwivedi, Cordelia Schmid, Hongwei Yi, Michael J. Black, and Dimitrios Tzionas. **POCO**: 3D pose and shape estimation using confidence. In *International Conference on 3D Vision (3DV)*, 2024. 3

[10] Zigang Geng, Chunyu Wang, Yixuan Wei, Ze Liu, Houqiang Li, and Han Hu. Human pose as compositional tokens. In *Computer Vision and Pattern Recognition (CVPR)*, 2023. 3

[11] G. Georgakis, Ren Li, S. Karanam, Terrence Chen, J. Kosecka, and Ziyan Wu. Hierarchical kinematic human mesh recovery. *European Conference on Computer Vision (ECCV)*, 2020. 3

[12] Shubham Goel, Georgios Pavlakos, Jathushan Rajasegaran, Angjoo Kanazawa, and Jitendra Malik. Humans in 4D: Reconstructing and tracking humans with transformers. In *International Conference on Computer Vision (ICCV)*, 2023. 1, 2, 3, 4, 6, 7, 8

[13] Chunhui Gu, Chen Sun, David A Ross, Carl Vondrick, Caroline Pantofaru, Yeqing Li, Sudheendra Vijayanarasimhan, George Toderici, Susanna Ricco, Rahul Sukthankar, et al. **Ava**: A video dataset of spatio-temporally localized atomic visual actions. In *Computer Vision and Pattern Recognition (CVPR)*, pages 6047–6056, 2018. 7

[14] Chuan Guo, Xinxin Zuo, Sen Wang, and Li Cheng. **Tm2t**: Stochastic and tokenized modeling for the reciprocal generation of 3d human motions and texts. In *European Conference on Computer Vision (ECCV)*, 2022. 3

[15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Computer Vision and Pattern Recognition (CVPR)*, 2016. 6

[16] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv: 1606.08415, 2016. 6

[17] Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu. Human3.6M: Large scale datasets and predictive methods for 3D human sensing in natural environments. Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2014. 7, 1

[18] Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. Motiongpt: Human motion as a foreign language. arXiv preprint arXiv: 2306.14795, 2023. 3

[19] Sam Johnson and Mark Everingham. Learning effective human pose estimation from inaccurate annotation. In Computer Vision and Pattern Recognition (CVPR), 2011. 8, 1

[20] Hanbyul Joo, Natalia Neverova, and Andrea Vedaldi. Exemplar fine-tuning for 3D human pose fitting towards in-the-wild 3D human pose estimation. In International Conference on 3D Vision (3DV), 2020. 3

[21] Angjoo Kanazawa, Michael J. Black, David W. Jacobs, and Jitendra Malik. End-to-end recovery of human shape and pose. In Computer Vision and Pattern Recognition (CVPR), 2018. 3

[22] Angjoo Kanazawa, Jason Y. Zhang, Panna Felsen, and Jitendra Malik. Learning 3D human dynamics from video. In Computer Vision and Pattern Recognition (CVPR), 2019. 3, 7

[23] Manuel Kaufmann, Jie Song, Chen Guo, Kaiyue Shen, Tianjian Jiang, Chengcheng Tang, Juan Jose Zárate, and Otmar Hilliges. EMDB: The Electromagnetic Database of Global 3D Human Pose and Shape in the Wild. In International Conference on Computer Vision (ICCV), 2023. 2, 7

[24] Muhammed Kocabas, Nikos Athanasiou, and Michael J. Black. VIBE: Video inference for human body pose and shape estimation. In Computer Vision and Pattern Recognition (CVPR), 2020. 3

[25] Muhammed Kocabas, Chun-Hao P. Huang, Otmar Hilliges, and Michael J. Black. PARE: Part attention regressor for 3D human body estimation. In International Conference on Computer Vision (ICCV), 2021. 3, 6, 7

[26] Muhammed Kocabas, Chun-Hao P. Huang, Joachim Tesch, Lea Muller, Otmar Hilliges, and Michael J. Black. SPEC: Seeing people in the wild with an estimated camera. In International Conference on Computer Vision (ICCV), 2021. 3

[27] Nikos Kolotouros, Georgios Pavlakos, Michael J. Black, and Kostas Daniilidis. Learning to reconstruct 3D human pose and shape via model-fitting in the loop. In International Conference on Computer Vision (ICCV), 2019. 7

[28] Nikos Kolotouros, Georgios Pavlakos, and Kostas Daniilidis. Convolutional mesh regression for single-image human shape reconstruction. In Computer Vision and Pattern Recognition (CVPR), 2019. 3

[29] Nikos Kolotouros, Georgios Pavlakos, Dinesh Jayaraman, and Kostas Daniilidis. Probabilistic modeling for human mesh recovery. In International Conference on Computer Vision (ICCV), 2021. 3

[30] Christoph Lassner, Javier Romero, Martin Kiefel, Federica Bogo, Michael J. Black, and Peter Gehler. Unite the people: Closing the loop between 3D and 2D human representations. In Computer Vision and Pattern Recognition (CVPR), 2017. 3

[31] 李杰锋（Jiefeng Li）、徐超（Chao Xu）、陈志存（Zhicun Chen）、卞思远（Siyuan Bian）、杨立新（Lixin Yang）和卢策吾（Cewu Lu）。HybrIK: A hybrid analytical-neural inverse kinematics solution for 3D human pose and shape estimation（《HybrIK：用于 3D 人体姿态与形状估计的混合解析-神经逆运动学方法》）。发表于 Computer Vision and Pattern Recognition (CVPR)，2021 年。7

[32] 李志豪（Zhihao Li）、刘建庄（Jianzhuang Liu）、张振松（Zhensong Zhang）、徐松岑（Songcen Xu）和严有良（Youliang Yan）。CLIFF: Carrying location information in full frames into human pose and shape estimation（《CLIFF：将完整帧中的位置信息带入人体姿态与形状估计》）。发表于 European Conference on Computer Vision (ECCV)，2022 年。1, 2, 3, 7

[33] 林静（Jing Lin）、曾爱玲（Ailing Zeng）、王浩乾（Haoqian Wang）、张磊（Lei Zhang）和李煜（Yu Li）。One-stage 3D whole-body mesh recovery with component aware transformer（《基于组件感知 Transformer 的单阶段 3D 全身体网格恢复》）。发表于 Computer Vision and Pattern Recognition (CVPR)，2023 年。3

[34] Kevin Lin, Lijuan Wang, and Zicheng Liu. Mesh graphormer（《网格图 Transformer》）。发表于 International Conference on Computer Vision (ICCV)，2021 年。3

[35] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C. Lawrence Zitnick. Microsoft coco: Common objects in context（《Microsoft COCO：上下文中的常见物体》）。发表于 European Conference on Computer Vision (ECCV)，2014 年。7

[36] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multiperson linear model（《SMPL：一种蒙皮多人线性模型》）。发表于 Transactions on Graphics (TOG)，2015 年。3, 4

[37] Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll, and Michael J. Black. AMASS: Archive of motion capture as surface shapes（《AMASS：以表面形状形式存储的动作捕捉数据库》）。发表于 International Conference on Computer Vision (ICCV)，2019 年。2, 6, 7, 1

[38] Dushyant Mehta, Helge Rhodin, Dan Casas, Pascal V. Fua, Oleksandr Sotnychenko, Weipeng Xu, and Christian Theobalt. Monocular 3d human pose estimation in the wild using improved cnn supervision（《使用改进的 CNN 监督的野外单目 3D 人体姿态估计》）。发表于 International Conference on 3D Vision (3DV)，2017 年。7

[39] Marko Mihajlovic, Shunsuke Saito, Aayush Bansal, Michael Zollhoefer, and Siyu Tang. COAP: Compositional articulated occupancy of people（《COAP：人体的组合式关节占据表示》）。发表于 Computer Vision and Pattern Recognition (CVPR)，2022 年。3

[40] Gyeongsik Moon, Hongsuk Choi, and Kyoung Mu Lee. Neuralannot: Neural annotator for 3d human mesh training sets（《NeuralAnnot：用于 3D 人体网格训练集的神经标注器》）。发表于 Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition，第 2299–2307 页，2022 年。3

[41] Mohamed Omran, Christoph Lassner, Gerard Pons-Moll, Peter Gehler, and Bernt Schiele. Neural body fitting: Unifying deep learning and model based human pose and shape estimation（《神经人体拟合：统一深度学习与基于模型的人体姿态与形状估计》）。发表于 International Conference on 3D Vision (3DV)，2018 年。3

[42] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3D hands, face, and body from a single image（《富有表现力的人体捕捉：从单张图像重建 3D 手部、面部与身体》）。发表于 Computer Vision and Pattern Recognition (CVPR)，2019 年。2, 3, 7, 8, 1

[43] A. Ramesh, Mikhail Pavlov, Gabriel Goh, S. Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. 零样本文本到图像生成. 国际机器学习会议 (ICML), 2021. 3

[44] Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. 使用 vq-vae-2 生成多样化高保真图像. 神经信息处理系统会议 (NeurIPS), 2019. 3

[45] Shunsuke Saito, Tomas Simon, Jason Saragih, and Hanbyul Joo. PIFuHD：用于高分辨率 3D 人体数字化的多层级像素对齐隐式函数. 计算机视觉与模式识别会议 (CVPR), 2020. 3

[46] Istvan Sárándi, Timm Linder, Kai O. Arras, and Bastian Leibe. 面向 3D 人体姿态估计的度量尺度截断鲁棒热力图. IEEE 国际自动面部与手势识别会议 (FG), 2020. 3

[47] Akash Sengupta, Ignas Budvytis, and Roberto Cipolla. Hu-ManiFlow：基于 SO(3) 流形上祖先条件归一化流的人体姿态与形状分布估计. 计算机视觉与模式识别会议 (CVPR), 2023. 3

[48] Yu Sun, Qian Bao, Wu Liu, Yili Fu, Black Michael J., and Tao Mei. 单目单阶段多 3D 人体回归. 国际计算机视觉会议 (ICCV), 2021. 3

[49] Yu Sun, Wu Liu, Qian Bao, Yili Fu, Tao Mei, and Michael J Black. 将人物置于其位：深度空间中 3D 人体的单目回归. 计算机视觉与模式识别会议 (CVPR), 2022. 1, 3

[50] Guy Tevet, Sigal Raab, Brian Gordon, Yoni Shafir, Daniel Cohen-or, and Amit Haim Bermano. 人体运动扩散模型. 第十一届国际学习表征会议 (ICLR), 2023. 2

[51] Garvita Tiwari, Dimitrije Antic, Jan Eric Lenssen, Nikolaos Sarafianos, Tony Tung, and Gerard Pons-Moll. Pose-ndf：使用神经距离场建模人体姿态流形. 欧洲计算机视觉会议 (ECCV), 2022. 3

[52] Shashank Tripathi, Lea Muller, Chun-Hao P. Huang, Taheri Omid, Michael J. Black, and Dimitrios Tzionas. 基于直观物理的 3D 人体姿态估计. 计算机视觉与模式识别会议 (CVPR), 2023. 2, 6, 7, 1

[53] Aaron Van Den Oord, Oriol Vinyals, et al. 神经离散表示学习. 神经信息处理系统会议 (NeurIPS), 2017. 2, 3, 5

[54] Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 注意力即所需. 神经信息处理系统会议 (NeurIPS), 2017. 6

[55] Timo von Marcard, Roberto Henschel, Michael J. Black, Bodo Rosenhahn, and Gerard Pons-Moll. 使用 IMU 和移动相机在野外恢复准确的 3D 人体姿态. 欧洲计算机视觉会议 (ECCV), 2018. 2, 7

[56] Wenjia Wang, Yongtao Ge, Haiyi Mei, Zhongang Cai, Qingping Sun, Yanjun Wang, Chunhua Shen, Lei Yang, and Taku Komura. Zolly: Zoom focal length correctly for perspectivedistorted human mesh reconstruction. In International Conference on Computer Vision (ICCV), 2023. 3

[57] Jiahong Wu, He Zheng, Bo Zhao, Yixin Li, Baoming Yan, Rui Liang, Wenjia Wang, Shipei Zhou, Guosen Lin, Yanwei Fu, et al. Ai challenger: A large-scale dataset for going deeper in image understanding. arXiv, 2017. 7

[58] Yuliang Xiu, Jinlong Yang, Xu Cao, Dimitrios Tzionas, and Michael J. Black. ECON: Explicit clothed humans optimized via normal integration. In Computer Vision and Pattern Recognition (CVPR), 2023. 3

[59] Hongyi Xu, Eduard Gabriel Bazavan, Andrei Zanfir, William T. Freeman, Rahul Sukthankar, and Cristian Sminchisescu. GHUM & GHUML: Generative 3D human shape and articulated pose models. In Computer Vision and Pattern Recognition (CVPR), 2020. 3

[60] Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao. ViTPose: Simple vision transformer baselines for human pose estimation. In Conference on Neural Information Processing Systems (NeurIPS), 2022. 6

[61] Hongwen Zhang, Yating Tian, Xinchi Zhou, Wanli Ouyang, Yebin Liu, Limin Wang, and Zhenan Sun. Pymaf: 3d human pose and shape regression with pyramidal mesh alignment feedback loop. In International Conference on Computer Vision (ICCV), pages 11446–11456, 2021. 3

[62] Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Shaol Huang, Yong Zhang, Hongwei Zhao, Hongtao Lu, and Xi Shen. T2m-gpt: Generating human motion from textual descriptions with discrete representations. In Computer Vision and Pattern Recognition (CVPR), 2023. 3, 6

# TokenHMR: Advancing Human Mesh Recovery with a Tokenized Pose Representation

Supplementary Material

## A. Introduction

In this supplemental document, we provide more implementation details and discuss limitations of TokenHMR. Please refer to the supplemental video for a brief review of the paper and more qualitative results.

## B. More Implementation Details

## B.1. Data Preparation for Tokenizer

For pose tokenization, we use 21 body pose parameters following Vposer [42]. As shown in Tab. 3 of main paper, we evaluate our tokenization in two settings: indistribution and out-of-distribution. For in-distribution, we train on the training set of AMASS [37] and evaluate on the test set of AMASS. To show the efficacy of tokenization, we also evaluate on an out-of-distribution yoga dataset, MOYO [52]. For training, we use the following datasets: {CMU, KIT, BMLrub, DanceDB, BMLmovi, EyesJapan, BMLhandball, TotalCapture, EKUT, ACCAD, TCDHands, MPI-Limits} with a weighting of {0.14, 0.14, 0.14, 0.06, 0.06, 0.06, 0.06, 0.06, 0.04, 0.04, 0.04, 0.16}, respectively.

## B.2. Joint-wise Thresholds for TALS

为了为 TALS（第 4.2 节）建立有效的逐关节阈值，我们对 BEDLAM [3] 数据集的 20221018 3- 8 250 batch01hand 6fps 验证子集进行了详尽的统计分析，该子集包含超过 34k 个样本，涵盖了多样化的人体 3D 姿态、形状以及相机视角。表 S.1 展示了 TALS 所使用的每个关节的阈值距离。

## B.3. 数据增强

数据增强在提升人体姿态与形状（HPS）回归器的鲁棒性与泛化能力方面起着关键作用。因此，沿用 HMR2.0 的做法，我们执行了多种数据增强操作。其中包括在 x 和 y 方向上的随机平移（系数为 0.02）、缩放（系数为 0.3）以及旋转（30 度）。其他增强手段还包括水平翻转与颜色重缩放。我们观察到，极端裁剪（即随机移除人体的部分肢体）同样能够提升模型对遮挡的鲁棒性。

## C. 讨论

## C.1. 姿态空间分析

我们通过评估训练集中未出现的分布外（OOD）姿态的重构效果来分析姿态空间。具体做法是在 AMASS 上训练，并在 MOYO 上进行测试。定性结果如

| 2D 关节 | 阈值 | SMPL 关节 | 阈值 |
| --- | --- | --- | --- |
| OP Nose（鼻子） | 0.00850 | Pelvis（骨盆） | 0.46 |
| OP Neck（颈部） | 0.00649 | LHip（左髋） | 0.22 |
| OP RShoulder（右肩） | 0.00748 | RHip（右髋） | 0.21 |
| OP RElbow（右肘） | 0.01103 | Spine（脊柱） | 0.15 |
| OP RWrist（右腕） | 0.01356 | LKnee（左膝） | 0.33 |
| OP LShoulder（左肩） | 0.00742 | RKnee（右膝） | 0.30 |
| OP LElbow（左肘） | 0.01097 | Thorax（胸腔） | 0.17 |
| OP LWrist（左腕） | 0.01414 | LAnkle（左踝） | 0.20 |
| OP MidHip（髋部中点） | 0.00974 | RAnkle（右踝） | 0.27 |
| OP RHip（右髋） | 0.01127 | Thorax（胸腔） | 0.12 |
| OP RKnee（右膝） | 0.01663 | LToe（左脚趾） | 0.29 |
| OP RAnkle（右踝） | 0.00565 | RToe（右脚趾） | 0.28 |
| OP LHip（左髋） | 0.01126 | Neck（颈部） | 0.24 |
| OP LKnee（左膝） | 0.01616 | LCollar（左锁骨） | 0.26 |
| OP LAnkle（左踝） | 0.00533 | RCollar（右锁骨） | 0.26 |
| OP REye（右眼） | 0.00830 | Jaw（下颌） | 0.28 |
| OP LEye（左眼） | 0.00831 | LShoulder（左肩） | 0.29 |
| OP REar（右耳） | 0.00737 | RShoulder（右肩） | 0.32 |
| OP LEar（左耳） | 0.00743 | LElbow（左肘） | 0.35 |
| OP LBigToe（左大脚趾） | 0.00544 | RElbow（右肘） | 0.35 |
| OP LSmallToe（左小脚趾） | 0.00551 | LWrist（左腕） | 0.62 |
| OP LHeel（左脚跟） | 0.00536 | RWrist（右腕） | 0.59 |
| OP RBigToe（右大脚趾） | 0.00565 | LHand（左手） | 0.20 |
| OP RSmallToe（右小脚趾） | 0.00582 | RHand（右手） | 0.20 |
| OP RHeel（右脚跟） | 0.00573 |  |  |
| LSP RAnkle（右踝） | 0.00554 |  |  |
| LSP RKnee（右膝） | 0.01515 |  |  |
| LSP RHip（右髋） | 0.00986 |  |  |
| LSP LHip（左髋） | 0.00998 |  |  |
| LSP LKnee（左膝） | 0.01520 |  |  |
| LSP LAnkle（左踝） | 0.00511 |  |  |
| LSP RWrist（右腕） | 0.01288 |  |  |
| LSP RElbow（右肘） | 0.01106 |  |  |
| LSP RShoulder（右肩） | 0.00711 |  |  |
| LSP LShoulder（左肩） | 0.00710 |  |  |
| LSP LElbow（左肘） | 0.01092 |  |  |
| LSP LWrist（左腕） | 0.01388 |  |  |
| LSP Neck（颈部） | 0.00648 |  |  |
| LSP Head Top（头顶） | 0.00766 |  |  |
| MPII Pelvis（骨盆） | 0.00931 |  |  |
| MPII Thorax（胸腔） | 0.00647 |  |  |
| H36M Spine（脊柱） | 0.00677 |  |  |
| H36M Jaw（下颌） | 0.00744 |  |  |
| H36M Head（头部） | 0.00752 |  |  |

表 S.1. 44 个 2D 关节和 24 个 SMPL 关节的阈值。2D 关节名称以骨骼原点开头，其中 OP 代表 OpenPose [5]。LSP [19]、MPII [2] 和 H36M [17] 为数据集。

所示，其展示了对分布外（out-of-distribution）的 MOYO [52] 瑜伽姿态具有良好的泛化能力。相比之下，我们发现噪声较大的测试姿态无法被很好地恢复。

![](images/0fd964cc854b0e895e9d6b072b4e739ddd58a19410ad94c1ef1973f1d99da6ac.jpg)  
图 S.1. 由仅在 AMASS 上训练的 tokenizer 重建的未见姿态（3D 身体关节）的 t-SNE 可视化。我们能够重建 MOYO 中分布外的瑜伽姿态。GT 表示真实姿态，PR 表示预测姿态。

## C.2. TALS 损失 vs 过滤策略

与 HMR2.0 类似，我们采用过滤策略来确保 p-GT（伪真实标注）的高质量 2D 图像对齐。然而，过滤策略是"全有或全无"的；即数据样本要么被拒绝，要么被接受。我们的 TALS 损失则不同，它使用阈值内的所有过滤后的伪真实样本，超过阈值后监督强度会被缩小。这超越了标准的过滤和数据清理流程。

## D. 局限性讨论

## D.1. 弱透视相机模型下的 2D 对齐较差

第 3 节中的实验分析表明，使用现有的有缺陷的相机投影模型会导致对 2D 关键点的过拟合，并导致学习到有偏的姿态。为避免此问题，我们设计了一种宽松的 TALS 监督训练策略，并通过基于 token 的姿态表示融入先验知识。如图 S.2 a) 所示，结合使用 TALS 的宽松 2D 监督和表示中内置的先验，TokenHMR 能够估计出合理的 3D 姿态，但当存在透视缩短（foreshortening）时，这些姿态在 2D 图像中并不总能很好地对齐。正如在弱透视相机模型下所预期的，透视畸变越明显，2D 对齐效果越差。

## D.2. 失败案例

在本工作中，我们引入 TokenHMR 以减少相机/姿态偏差，并通过 token 化的姿态先验来缓解歧义。然而，TokenHMR 仍然存在一些局限性，可在未来工作中进一步探索。

如图 S.2 b) 所示，在没有更好的相机模型的情况下，透视缩短仍然具有挑战性。在如图 S.2 c) 所示的情况下，仅考虑身体线索时，全局朝向是模糊的。我们可能需要利用来自面部和脚部的更多线索来确定正确的全局朝向。未来工作可以尝试将 TokenHMR 扩展到全身姿态估计（即 SMPL-X）以解决此问题。

## E. 未来工作

显然，未来工作应当直接解决相机投影问题，通过恢复更准确的相机估计。

![](images/c2153c52761eeb0961ccaf9e6c9a65d10ac6445c964be52ed4e8385099eab95e.jpg)

a) 由于 TALS (Token-Aligned Supervision, token 对齐监督) 的监督较为宽松，我们的预测在弱透视相机下无法在二维空间良好对齐。  
![](images/803b370b4a8b745368d52c170c16cc7175d9363af1a926b57d02c8cb259b0d7d.jpg)  
b) 深度方向的歧义性仍然极具挑战性。

![](images/305319639da87fd2c58b35c184ee67e2c6c2d66a4d708088a6fae5922119d915.jpg)

![](images/a5d0c3c9ae93c0e59c0af36a331350583886162a24bceb5cacec67900488f78e.jpg)  
c) 全局朝向估计有时会失败，因为面部与脚部线索尚未被充分挖掘。  
图 S.2. 二维对齐问题及失败案例。

即便有上述改进，我们仍认为 token 表示具有重要价值，因为它在多种测试场景下都能稳定提升性能。一个有前景的下一步方向是将 token 化扩展到时间维度上。近期从文本生成人体动作的工作利用了人体动作的 token 化表示 [50]。放眼未来，一个引人入胜的研究方向是探索将我们基于 token 的姿态表示应用于大语言模型 (LLMs)。我们为三维人体姿态估计所设计的姿态 token 具有离散且鲁棒的特性，这为弥合计算机视觉与自然语言处理之间的鸿沟提供了机会。