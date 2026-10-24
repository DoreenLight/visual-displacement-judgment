# 运动学标记化（Kinematic Tokenization）：基于优化的连续时间标记用于噪声时间序列中的可学习决策策略

Grifin M. Kearney, Ph.D.

Syracuse, New York, USA gkearneyengineering@gmail.com

2026年1月21日

## 摘要

Transformer 最初是为离散标记（tokens）设计的，然而许多现实世界中的信号是通过噪声采样观测到的连续过程。离散标记化（原始数值、补丁/patches、有限差分）在低信噪比环境下可能表现脆弱，特别是当下游目标施加不对称惩罚并合理地促使模型选择放弃行动（abstention）时更是如此。我们提出了运动学标记化（Kinematic Tokenization），这是一种基于优化的连续时间表征方法，它从带噪测量中重构出显式样条（spline），并将局部样条系数（位置、速度、加速度、加加速度/jerk）标记化。该方法应用于以资产价格结合成交量分布图形式呈现的金融时间序列数据中。在多资产日度股票测试平台上，我们使用风险厌恶的不对称分类目标作为可学习性的压力测试。在此目标下，多个离散基线方法崩溃为吸收性现金策略（清算均衡，Liquidation Equilibrium），而连续样条标记则维持了经过良好校准的非平凡动作分布和稳定策略。这些结果表明，在诱发放弃行动的损失函数下，显式连续时间标记可以提高噪声时间序列中选择性决策策略的可学习性和校准能力。

## 1 引言

将最初为离散自然语言标记设计的基础模型（Foundation Models）适配到连续时间序列数据，面临着根本性的表征挑战。虽然文本本质上是离散的，但物理和经济信号是透过离散采样时钟观察到的连续系统。当前的方法通常依赖于分块（patching）或对原始数值的直接量化。这些方法未能考虑测量不确定性，也未能考虑控制样本之间系统演化的潜在动力学规律（即“物理规律”）。尽管金融市场本质上缺乏自然物理系统中所具有的严格第一性原理控制，但在我们的实验中，我们对交易参数的演化施加了类牛顿运动方程，利用运动学一致性作为表征学习的鲁棒归纳偏置。

## 1.1 物理启发标记化框架

在本文中，我们通过将时间序列问题置于物理启发人工智能（Physics-Informed AI）的框架内，弥合了离散测量样本与底层连续过程之间的这一差距。我们不再将数据点视为静态数值，而是将其视为受随机外力作用的潜在动力系统的带噪观测值。我们提出了一种基于优化数据丰富（Optimization-Based Data Enrichment）[1] 的方法。如图 Figure 1 所示，我们的框架从根本上通过两个随机通道和两个耦合流形融合了物理特性：

1. 动力系统模型（Dynamical Systems Model）：描述系统状态（例如价格/Price）和系统通量（例如成交量/Volume）在随机过程噪声的影响下如何随时间连续演变。

2. 测量模型（Measurement Model）：描述我们如何观测这些状态，并对快照测量（例如价格）和聚合测量（例如成交量）进行区分。

![](images/fae459e887633005f5a73e9c41042114423712f24dfac57afc5e30089be836c5.jpg)  
Figure 1: 物理感知连续分词（Physics-Informed Continuous Tokenization）概念框架。该方法整合了系统动力学模型和测量协议，两者均受随机不确定性影响。与通过隐式表示产生离散估计的传统滤波（例如 Kalman 滤波）不同，我们基于优化的方法生成显式连续样条表示。这种连续统对于为基础模型（foundation model）提取高阶运动学标记（tokens）至关重要。

## 1.2 显式表示与隐式表示

我们方法的一个关键区别在于所生成表示的本质。传统方法（如 Kalman 滤波）是隐式的：它们在预先指定的离散滤波时间生成最佳估计，从而实现离散数据校正和保真度更高（但仍为离散）的采样。Kalman 方法的连续极限确实存在 [2]，但这些处理方式未能生成适合在 Transformer 架构中使用的紧凑离散表示（例如样条系数）；这些方法通常是隐式且递归的。相比之下，我们的优化框架通过直接紧凑的表示求解显式连续样条，该样条定义了区间内所有时间 $t$ 的状态 $x(t)$ 及其所有导数。

这种显式特性对于分词（tokenization）至关重要。它使我们能够解析计算高阶导数，即速度、加速度、加加速度（jerk）和体积流率，这些在离散方法中通常缺失，但包含有关驱动系统的底层动力学的重要信息。这些运动学参数构成了我们连续时间标记（tokens）的基础，为 Transformer 提供了信号耦合演化的物理基础表示。

## 1.3 作为随机测试平台的金融

尽管本工作中的实验验证侧重于金融市场的日度数据，但所提出的框架是领域无关的（domain-agnostic）。金融时间序列代表了一类极具挑战性的随机动态系统，其特征在于低信噪比和时变波动性。然而，底层的最优样条形式是通用的。通过重构动力学、过程噪声和测量模型，该策略可以适应不同的物理领域，例如惯性导航、机器人控制或其他各种信号处理应用，在这些应用中，保持信号的连续导数至关重要。

## 2 相关工作

## 2.1 时间序列基础模型

大语言模型（Large Language Models）的成功推动了大观测模型（Large Observation Models, LOMs）的发展。当前最先进的架构如 PatchTST [3] 和 iTransformer [4] 利用分块（patching）来捕获局部语义信息。2025 年的最新研究，例如 Google 的 TimesFM-ICF [5]，引入了上下文学习以适应非平稳数据。然而，这些模型在物理上仍是不可知的（physically agnostic）；它们将时间序列分块视为统计向量，忽略了信号底层的运动学连续性。

## 2.2 新兴连续架构（SSMs & LNNs）

为了解决 Transformer 在长序列上的计算效率低下问题，以 Mamba 为代表的状态空间模型（State Space Models, SSMs）受到了广泛关注。近期的应用如 MambaStock [6] 和 TimeMachine [7] 利用 Mamba 的选择性扫描机制，以线性复杂度学习离散化的潜在动力学。类似地，液体神经网络（Liquid Neural Networks）[8] 和物理信息 Koopman 网络（Physics-Informed Koopman Networks）[9] 通过微分方程显式建模神经元激活或潜在状态，以处理不规则性。

尽管这些方法有效，但它们通常在未经处理的嘈杂离散输入上运行，并且必须从头开始学习系统动力学（矩阵 A 或 ODE 参数）。SplineGPT 的不同之处在于将物理特征提取与模式识别解耦。我们不要求模型去提取物理特征；而是通过事前的样条优化显式计算连续状态（速度、加速度等），为 Transformer 提供预先计算好的、具有物理依据的连续物理离散表示，同时保留标准注意力机制的训练效率。

## 2.3 物理信息金融

物理信息神经网络（Physics-Informed Neural Networks, PINNs）[10] 在金融领域的应用主要集中在求解已知的偏微分方程（Partial Differential Equations, PDEs）。Socaciu 等人（2025）[11] 的最新工作展示了 PINNs 在求解用于期权定价的 Black-Scholes 和 Heston 方程方面的实用价值。限价订单簿中的深度学习方法通常依赖于离散的空间或时间结构 [12, 13]。然而，这些方法大多是演绎式的——将已知定律强加于数据之上。我们的方法则是归纳式的：我们使用基于优化的丰富方法（optimization-based enrichment）[1] 从资产价格数据流中推导出潜在的运动学状态，从而在一般支配定律未知的交易场景中实现 Alpha 生成（alpha generation）。

需要指出的是，在当前方法中，在缺乏基础物理学（例如反映牛顿第二定律的随机受力二阶运动）的情况下预先规定了局部“定律”，这些定律用于金融数据标记化（tokenization）。我们强调，通过修改控制律和测量模型，此处描述的标记化方案可以适用于其他存在已验证的控制系统行为的丰富物理框架的数据领域。正如我们在 (1) 和 (2) 中所示，当具备严谨物理学时，同样可以应用相同的通用处理方法来调整等效的数据富集公式。

## 3 方法

我们提出的流程将原始金融数据转化为适合高维嵌入的丰富潜在表示。该过程由三个不同的阶段组成：(1) 信号预处理，(2) 基于优化的样条富集，以及 (3) 平稳性与归一化。我们在 Figure 2 中展示了所提出的机器学习系统架构。

![](images/3f7637cf11aff1bebfdee79c8556f47d6c105ff93d83e7ab958ae3ec232b3d87.jpg)  
Figure 2: 连续标记化流程。原始离散数据经过变换，然后使用基于优化的随机样条生成方法 [1] 进行富集。提取代表位置（Position）、速度（Velocity）、加速度（Acceleration）和加加速度（Jerk）的系数，锚定至窗口起点，并输入到因果 Transformer（Causal Transformer）中。

## 3.1 标记化前变换

金融时间序列，特别是资产价格 $P ( t )$ ，定义在非负实数线 $\mathbb { R } ^ { + }$ 上。直接分析原始价格通常会引入尺度不稳定性。为了将状态空间映射到更适合标准随机过程的域，我们应用对数变换 $y _ { k } = \ln ( P ( t _ { k } ) )$ 。这使得数据与富集阶段中使用的对称随机过程（例如高斯过程）保持一致。

对数变换也应用于日交易量数据 $V ( t )$ ，得到 $\tilde { y } _ { k } = \ln ( V ( t _ { k } ) )$ 。与前面一样，这使数据与对称过程保持一致，并通过压缩动态范围来提高数值稳定性；超大盘股的日交易量可达数亿，而小盘股的日交易量仅在数万左右。

## 3.2 基于优化的样条富集

核心创新依赖于我们之前的工作 [1]。我们将变换后的序列 $\left\{ y _ { k } \right\}$ 视为由随机微分方程（SDE）控制的潜在连续状态 x(t) 的带噪测量值。我们构建了在区间 $[ t _ { 0 } , t _ { N } ]$ 上使似然最大化的样条

$$
\begin{array}{r l} \min _ {v, w, x} & \int_ {t _ {0}} ^ {t _ {N}} \frac {1}{2} v ^ {2} d t + \frac {\alpha^ {2}}{2} \sum_ {k} w _ {k} ^ {2} \\ \text {s.t.} & \ddot {x} (t) = v (t) \\ & y _ {k} = x (t _ {k}) + w _ {k}, k = 0, 1,..., N \end{array}\tag{1}
$$

利用变分法，最优解 $x ^ { * } ( t )$ 是一个由过程噪声与测量噪声之比 $\begin{array} { r } { ( \alpha ^ { 2 } = \frac { \sigma _ { p } ^ { 2 } } { \sigma _ { m } ^ { 2 } } ) } \end{array}$ 所决定的连续样条。参数 $\alpha ^ { 2 }$ 起到滤波参数的作用，将潜在的市场趋势与高频微观结构噪声（例如买卖报价弹跳）分离开来，确保 Token（标记）表征的是信号而非抖动。在本文所述的原型模型中，我们采用 $\alpha = 5$ 的值进行概念验证，但未来的详细统计研究可用于优化该随机滤波参数以提升性能。

问题 (1) 为资产价格的 Token 化提供了严格的运动学基础。对于交易量，我们采用了一种不同的思路。由于交易量是一个聚合度量（在某一区间内交易的股票数量）而非连续状态变量，我们采用了一种启发式交易量平滑器，旨在稳定 Transformer 的输入。我们将交易量提取公式化为：

$$
\begin{array}{r l} \min _ {\tilde {v}, \tilde {w}, \tilde {x}} & \int_ {t _ {0}} ^ {t _ {N}} \frac {1}{2} \tilde {v} ^ {2} d t + \frac {\alpha^ {2}}{2} \sum_ {k} \tilde {w} _ {k} ^ {2} \\ \mathrm{s.t.} & \ddot {\tilde {x}} (t) = \tilde {v} (t) \\ \mathrm{s.t.} & \tilde {y} _ {k} = \int_ {t _ {k - 1}} ^ {t _ {k}} \tilde {x} (t) d t + \tilde {w} _ {k}, k = 1, \dots , N. \end{array}\tag{2}
$$

我们明确区分了 (2) 与 (1) 的解释。(1) 代表物理运动方程，而 (2) 则充当现象学滤波器 [14]。测量方程使用对数变换后的聚合交易量 $( \tilde { y } _ { k } = \ln ( V _ { k } ) )$。从数学上讲，将状态 ${ \tilde { x } } ( t )$ 的积分视为与和的对数一致是一种已知的近似。然而，在深度学习输入的背景下，这种启发式方法具有关键的工程用途：它压缩了交易量激增（跨越数个数量级）的动态范围，同时通过样条约束强制实现时间连续性。我们发现该近似在经验上对于稳定重尾交易量数据的 Token 化十分稳健，从而防止了异常梯度破坏 Transformer 主干网络的稳定性。

求解 (1) 可以生成对数价格的分段最优（极大似然）估计，由下式给出：

$$
x (t) = c _ {0} (t _ {k}) + c _ {1} (t _ {k}) (t - t _ {k}) + c _ {2} (t _ {k}) \frac {(t - t _ {k}) ^ {2}}{2} + c _ {3} (t _ {k}) \frac {(t - t _ {k}) ^ {3}}{6},\tag{3}
$$

其中 $t \in [ t _ { k } , t _ { k + 1 } ]$，且对应于该区间的常数由最优解给出。系数 $c _ { j } ( t _ { k } )$ 表示从 $t _ { k }$ 开始的区间上的第 $j$ 个运动学项（例如位置、速度等）。类似地，求解 (2) 会生成成交量强度函数（Volume Intensity Function）${ \tilde { x } } ( t )$ 的分段最优估计，定义为

$$
\tilde {x} (t) = \tilde {c} _ {0} (t _ {k}) + \tilde {c} _ {1} (t _ {k}) (t - t _ {k}) + \tilde {c} _ {2} (t _ {k}) \frac {(t - t _ {k}) ^ {2}}{2} + \tilde {c} _ {3} (t _ {k}) \frac {(t - t _ {k}) ^ {3}}{6} + \tilde {c} _ {4} (t _ {k}) \frac {(t - t _ {k}) ^ {4}}{2 4},\tag{4}
$$

对于 $t \in [ t _ { k } , t _ { k + 1 } ]$。在此，系数 $\tilde{c}$ 并不代表严格的物理导数，而是局部对数成交量率分布的形状参数，使模型能够检测到需求激增（速度 $\tilde { c } _ { 1 } )$ 和衰竭形态（加速度 $\tilde { c } _ { 2 } )$。

## 3.3 连续时间 Token

式 (1) 的解为三次样条，因此在每个区间上，该样条完全由 4 个系数确定。类似地，式 (2) 的解为四次样条，因此在每个区间上完全由 5 个系数确定。从最优样条 $x ^ { * } ( t )$ 中，我们提取定义每个区间信号段的系数，用作数据 token。与离散点不同，我们的 token $\mathbf { t } _ { k }$ 和 $\tilde { \mathbf { t } } _ { k }$ 是表示该区间上市场运动学状态的向量：

$$
\mathbf {t} _ {k} = [ c _ {0}, c _ {1}, c _ {2}, c _ {3} ] ^ {T} \in \mathbb {R} ^ {4},\tag{5}
$$

以及

$$
\tilde {\mathbf {t}} _ {k} = [ \tilde {c} _ {0}, \tilde {c} _ {1}, \tilde {c} _ {2}, \tilde {c} _ {3}, \tilde {c} _ {4} ] ^ {T} \in \mathbb {R} ^ {5}.\tag{6}
$$

在 (5) 中，$c _ { 0 }$ 为“位置”（对数价格），$c _ { 1 }$ 为“速度”（收益率），$c _ { 2 }$ 为“加速度”（凸性），$c _ { 3 }$ 为 $\mathrm { \bar { \Psi } j e r k } ^ { \prime \prime }$（急动度/趋势不稳定性）。在 (6) 中，这一解释被修正为将交易量解释为“能量”等价物；交易的股票越多，标的资产的能量就越强。在此类比下，$\tilde { c } _ { 0 }$ 为“功率”（对数成交量率），$\tilde { c } _ { 1 }$ 为“功率”的一阶时间导数，依此类推。当拼接 $\mathbf { t } _ { k }$ 和 $\tilde { \mathbf { t } } _ { k }$ 进行处理时，这使得架构在每个时间区间上产生一个净 9 维的 token。拼接后的 token 向量同时包含价格和成交量信息，使 Transformer 能够关注两者之间的耦合动态。

## 3.4 平稳性与归一化

在金融数据上训练基础模型（foundation models）的一个关键挑战是平稳性。原始价格水平 $\left( c _ { 0 } \right)$ 和成交量强度 $\left( \tilde { c } _ { 0 } \right)$ 是非平稳的，这会导致“均值回归偏差”（mean reversion bias），即模型会错误地预测价格和成交量回归到数据集均值，而不是遵循局部趋势。

为了解决这一问题，我们实现了双窗口锚定（Dual Window Anchoring）。对于每个上下文窗口 W，我们从序列中的所有 token 中减去初始价格位置 $c _ { 0 } ( t _ { 0 } )$ 和初始成交量强度 $\tilde { c } _ { 0 } ( t _ { 0 } )$：

$$
c _ {0} ^ {\prime} (t _ {k}) = c _ {0} (t _ {k}) - c _ {0} (t _ {0})\tag{7}
$$

$$
\tilde {c} _ {0} ^ {\prime} (t _ {k}) = \tilde {c} _ {0} (t _ {k}) - \tilde {c} _ {0} (t _ {0})\tag{8}
$$

这强制每个训练序列从相对原点 (0.0, 0.0) 开始，使模型能够更好地学习价格和成交量的相对动态。

此外，高阶导数表现出巨大的尺度差异（例如，价格速度 $c _ { 1 } \approx 1 0 ^ { - 2 }$ 与成交量急动度变化率 $\tilde { c } _ { 4 } \approx 5 . 7 )$。我们计算了训练语料库中未锚定系数的全局统计量 $( \sigma \in \mathbb { R } ^ { 7 } )$，并对导数 $( c _ { 1 } , c _ { 2 } , c _ { 3 } )$ 及成交量形状参数 $( \tilde { c } _ { 1 } , \tilde { c } _ { 2 } , \tilde { c } _ { 3 } , \tilde { c } _ { 4 } )$ 应用了 Z-score 归一化 [15]。

## 4 模型架构与训练

## 4.1 SplineGPT 架构规范

我们采用了一种轻量级的、仅包含解码器（decoder-only）的因果 Transformer（Causal Transformer），旨在实现高吞吐量推理和快速适应。该架构被配置为“矮宽型”（Short $\&$ Wide）模型，优先考虑高带宽特征提取（宽度）而非深度逻辑推理（深度），以捕获 9 个耦合物理状态变量之间的复杂相关性。具体超参数详见 Table 1。

<table><tr><td>超参数</td><td>值</td></tr><tr><td>层数 (L)</td><td>4</td></tr><tr><td>注意力头数 (H)</td><td>8</td></tr><tr><td>嵌入维度 ( $d_{model}$ )</td><td>512</td></tr><tr><td>前馈网络维度 ( $d_{ff}$ )</td><td>2048</td></tr><tr><td>上下文窗口 (T)</td><td>64</td></tr><tr><td>输入通道数</td><td>9 (4 价格 + 5 成交量)</td></tr><tr><td>Dropout</td><td>0.1</td></tr><tr><td>激活函数</td><td>GELU</td></tr><tr><td>位置编码</td><td>RoPE</td></tr></table>

Table 1: SplineGPT 超参数

可训练参数总量约为 1260 万。尽管大于典型的“Nano”模型，但这种容量有利于解析 9 维运动学状态空间。尽管宽度较大，但较浅的深度（4 层）支持模型保持足够小的体积，以实现快速的 CPU 推理 $/$ 低延迟部署。原型模型在一台配备 64GB 内存（RAM）、拥有 8GB 板载显存的 NVIDIA GeForce RTX 2080 SUPER GPU 以及 Intel i9-9900 CPU 的个人台式机上进行训练。

## 4.2 训练与测试数据策略

该模型在大量的美国股票（US Equities）数据语料库上进行了预训练。经过数据集整合和质量保证筛选——基于低流动性、数据缺失等原因剔除缺陷股票代码——最终的训练语料库包含 2250 万个连续样条窗口。每个窗口的上下文长度为 64 个区间，这相当于约 14.4 亿个连续时间 token。

为确保严格的科学验证并防止前瞻偏差（look-ahead bias，数据泄漏），我们实施了严格的时间截断：

• 训练集（基础模型与适配器 / Foundation & Adapter）：2000 年 1 月 1 日至 2022 年 12 月 31 日的数据。

• 测试集（仅用于评估）：2023 年 1 月 1 日至今的数据。

基础模型（Foundation Model）对测试期完全不可见，以确保在近期市场机制上的任何性能表现均源自泛化的运动学学习，而非机械记忆。

## 4.3 训练稳定性

在高方差导数 token $\left( \mathrm { e . g . } c _ { 3 } , \tilde { c } _ { 4 } \right)$ 上进行训练会带来稳定性挑战。为确保收敛，我们采用了余弦退火学习率调度器（Cosine Annealing Learning Rate Scheduler），在 5 个 epoch 内将学习率从 $6 e ^ { - 4 }$ 衰减至 0。此外，我们实施了梯度裁剪（Gradient Clipping，范数 $\leq 1 . 0 )$，以防止由波动率曲面中的异常值引起的梯度爆炸。

## 4.4 通过低秩适应（LoRA）进行微调

在通用预训练阶段之后，我们采用低秩适应（Low-Rank Adaptation, LoRA）对模型进行专业化微调。LoRA 冻结预训练权重 W，并引入可训练的秩分解矩阵 A 和 B，使得 $W ^ { \prime } = W + B A$。我们针对 4 层架构的输入和输出处理阶段，策略性地注入了 LoRA 适配器：

1. 感知适应（Perception Adaptation，第 1 层与第 2 层）：微调底层特征提取 $( \mathrm { e . g . }$ ，检测量价背离）。

2. 推理适应（Reasoning Adaptation，第 4 层）：微调用于策略制定的最终决策边界。

微调头针对分类目标进行训练：买入（Buy）、卖出（Sell）或持有（Hold）。

## 4.5 训练目标与标签构建

为了使模型的运动学推理与可操作的交易信号相匹配，我们基于“动量逻辑（Momentum Logic）”阈值采用了三分类目标（买入、卖出、持有 / Buy, Sell, Hold）。对于离散类别标签，我们采用符号 $g _ { j }$，以将其与丰富化阶段（enrichment phase）中使用的测量观测值 $y _ { k }$ 区分开来。

对于在时间 $t _ { j }$ 结束的给定上下文窗口，标签 $g _ { j }$ 由目标标记（token）$c _ { 0 } ( t _ { j + 1 } )$ 相对于最后一个输入标记 $c _ { 0 } ( t _ { j } )$ 的状态确定：

$$
g _ {j} = \left\{ \begin{array}{l l} 0 (\text {Buy}) & \text {if r_{j} >\tau} \\ 1 (\text {Sell}) & \text {if r_{j} <  - \tau} \\ 2 (\text {Hold}) & \text {otherwise} \end{array} \right.\tag{9}
$$

其中 $r _ { j } = c _ { 0 } ( t _ { j + 1 } ) - c _ { 0 } ( t _ { j } )$ 是下一时间间隔内的对数收益率。在本工作中，我们将波动率阈值设为 $\tau = 0 . 0 1 \left( 1 \% \right)$。这过滤掉了微小的噪声，迫使模型仅在运动学信号表明存在显著价格偏移时预测“可操作的”走势。

为了诱导“主动风险管理”行为，我们使用非对称加权交叉熵损失（Weighted Cross-Entropy Loss）进行训练：

$$
\mathcal {L} = - \sum_ {i \in \{0, 1, 2 \}} w _ {i} \mathbb {I} (g = i) \log (\hat {g} _ {i})\tag{10}
$$

在此，$\hat { g } _ { i }$ 表示模型分配给类别 i 的预测概率，$\mathbb { I } ( g = i )$ 是真实标签 $g .$ 的指示函数。我们赋予权重 $\mathbf { w } = [ 2 . 0 , 1 0 . 0 , 1 . 0 ]$。对“卖出”（Sell）类别的极高惩罚 $( w _ { s e l l } = 1 0 . 0 )$ 促使模型对下行运动学特征（如负加加速度 / negative jerk）极度敏感，从而将资金保全置于准确率之上。

## 4.6 标记化消融实验与外部基线

为了分离出连续运动学标记化的效果，我们进行了两组互补的对比。(i) 受控内部消融实验：我们保持 Transformer 主干、参数量、训练目标和微调适配方法固定，仅改变输入表征（样条 Spline vs. 离散原始值 Raw vs. 离散有限差分）。(ii) 外部架构基线：我们还与 PatchTST [3] 进行了对比，作为基于分块（patching）的时间序列基础模型的代表。第二组对比匹配了参数规模，但并非严格的其他条件均同（ceteris paribus），因此被解释为行为鲁棒性参考，而非受控消融实验。

<table><tr><td>模型</td><td>输入表征</td><td>物理特性？</td><td>主干网络</td><td>参数量</td><td>适配方法</td></tr><tr><td>SplineGPT</td><td>连续样条系数</td><td>显式（精确）</td><td>Transf. (4L, 8H)</td><td>12.6M</td><td>LoRA</td></tr><tr><td>RawGPT-FD</td><td>离散有限差分</td><td>显式（含噪声）</td><td>Transf. (4L, 8H)</td><td>12.6M</td><td>LoRA</td></tr><tr><td>RawGPT</td><td>原始离散值</td><td>无（消融）</td><td>Transf. (4L, 8H)</td><td>12.6M</td><td>LoRA</td></tr><tr><td>PatchTST [3]</td><td>离散分块 (P=16,S=8)</td><td>隐式（潜在）</td><td>Transf. (4L, 4H)</td><td>≈12M</td><td>线性探测 (Linear Probe)</td></tr></table>

Table 2: 模型规格。受控消融实验（SplineGPT、RawGPT-FD、RawGPT）共享相同的 Transformer 主干网络和训练目标；差异仅局限于输入表征。PatchTST 作为外部基于分块（patching）的参考模型被纳入，其参数规模相当，但在架构和适配方法上有所不同。

## 4.6.1 内部消融实验：平滑性的价值

我们定义了两个内部变体，它们使用与 SplineGPT 完全相同的 Transformer 主干网络，但缺少样条丰富步骤。

RawGPT-FD（离散物理基线）：该模型用于测试样条的平滑性和准确性是否优于标准离散微分算子。它被提供了完整的运动学状态向量（位置、速度、加速度），但这些特征是通过显式有限差分推导得出的：

$$
c _ {n, k} \approx \Delta^ {n} y _ {k} = \sum_ {i = 0} ^ {n} (- 1) ^ {n - i} {\binom {n} {i}} y _ {k - (n - i)}.\tag{11}
$$

虽然在理论上这提供了与样条 $( c _ { 2 } , c _ { 3 } )$ 相同的信息，但离散微分会放大测量噪声 $( \mathrm { V a r } ( \Delta y ) \approx 2 \mathrm { V a r } ( y ) )$。如果 SplineGPT 优于 RawGPT-FD，则证实了基于优化的去噪相对于算术近似的价值。

RawGPT（消融物理基线）：该模型代表了一种信息匮乏的情况。它用于测试高阶运动学是否完全必要。我们将离散输入直接映射到通道维度，但将高阶导数 $( c _ { 2 } , c _ { 3 } )$ 和交易量形状参数 $( \tilde { c } _ { 2 } - \tilde { c } _ { 4 } )$ 用零掩码遮蔽。这迫使模型仅依赖原始方向性变动和标量波动率，从而有效地从输入中移除了“物理学”特征。

## 4.6.2 外部基准：PatchTST

为了验证我们的结果并非特定主干网络的人为产物，我们与多元时间序列领域的最先进基础模型 PatchTST [3] 进行了基准对比。

• 匹配容量：我们将 PatchTST 主干网络（通道独立，Channel-Independent）配置为 $d _ { m o d e l } = 2 5 6$ 和 4 个头。这使得总参数量（≈ 12M）与 SplineGPT 相当，从而避免基准模型参数不足。

• 隐式平滑：PatchTST 依赖于基于分块的标记化（patch-based tokenization）$( P = 1 6 , S = 8 )$ 来捕获局部语义。这构成了隐式与显式平滑的对比测试：模型是在自行聚合噪声点（分块）时学习得更好，还是在我们通过数学方法将点预处理为平滑曲线（样条）时学习得更好？

因此，PatchTST 的结果应被解读为外部参考基准，而非严格受控的消融实验。

## 5 实验验证

我们在包含 6 支主要美股（INTC、NVDA、JPM、XOM、PFE、TSLA）的测试集上进行了回测实验，时间跨度为 2023 年 1 月 1 日至今。

## 5.1 实验设置与协议

有效时间序列回测的关键要求是严格防止前视偏差（look-ahead bias）。为了确保这一点，我们实现了滚动样条标记化（Rolling Spline Tokenization）协议。我们不对整个数据集拟合单一的全局样条。相反，对于每个推理步骤 $t$：

1. 窗口提取（Window Extraction）：我们严格提取时间窗口 $[ t - T , t ]$ 内的原始离散价格/成交量历史数据。

2. 局部增强（Local Enrichment）：我们在该独立窗口上求解优化问题 (1) 和 (2)，以生成连续函数 $x ^ { * } ( t )$。

3. 标记化（Tokenization）：我们从该局部样条中提取系数 $\mathbf { t } _ { t - T : t }$。

4. 推理（Inference）：Transformer 处理这些标记（tokens）以预测 $t + 1$ 时刻的状态。

参数 $T$ 表示上下文窗口长度。这确保了时间 $t$ 处的“运动学状态”完全仅由截至时间 $t$ 可获得的信息导出。

## 5.1.1 回测约束与局限性

模拟假设初始资金为 \$10,000。我们通过在每 252 个交易日的一年结束时对已实现利润征收 32% 的短期资本利得税来执行税收惩罚。

• 执行（Execution）：交易在时间 $t$ 的每日收盘价进行模拟，与推理步骤重合。我们采用“收盘市价单”（Market-On-Close, MOC）执行机制，其中模型推理使用临近收盘的数据完成，订单在收盘集合竞价中成交。鉴于所选大盘股标的池（large-cap universe）的高流动性，以及本实验侧重于表示学习而非高换手率套利，我们将执行延迟视为可忽略不计。

• 局限性（Limitations）：本研究孤立地考察了表示的预测能力；因此，目前并未对可变交易成本（滑点、佣金）或市场冲击进行建模。尽管该模型显著的盈利表现为这些成本提供了可观的缓冲，但未来的工作将纳入完整的订单簿动态。

• 基准（Baselines）：主要的金融基准是买入并持有（Buy & Hold）策略，代表投资者的标准机会成本。

## 5.1.2 交易策略与状态机

为了将三元模型输出 $( g _ { t } ~ \in ~ \{ \mathrm { B u y } , \mathrm { S e l l } , \mathrm { H o l d } \} )$ 转化为投资组合状态，我们实现了一个严格的有限状态机（Finite State Machine, FSM）。时间 $t$ 处的投资组合状态 $S _ { t }$ 是二元的：$S _ { t } \in \{ 0 \ ( \mathrm { C a s h } ) , 1 \ ( \mathrm { L o n g } ) \}$。做空和杠杆被显式禁用，以将信号质量与杠杆效应隔离开来。

状态转换遵循以下逻辑：

• 入场（Cash → Long）：如果 $S _ { t } = 0$ 且模型预测 $g _ { t } = \mathrm { B u y }$，则以收盘价 $P _ { t }$ 执行买单。全部可用资金（扣除交易成本）将转换为股票。

• 退出 $( \mathbf { L o n g } \to \mathbf { C a s h } )$：若 $S _ { t } = 1$ 且模型预测 $g _ { t } = \mathrm { S e l l }$，则在 $P _ { t }$ 处执行卖单。所有持有的股票均被清算为现金。

• 维持：若 $g _ { t } = \mathrm { H o l d }$，或者模型发出无效转移信号（例如，在已处于多头时发出 $^ { 6 6 } \mathrm { B u y } ^ { 5 9 }$ 信号），则状态保持不变 $( S _ { t + 1 } = S _ { t } )$。

定义（清算均衡，Liquidation Equilibrium）。我们将动作空间输出与投资组合状态结果区分开来。模型可能会崩溃至退化的动作策略（例如，预测单一标签）。在我们的有限状态机（FSM）下，任何未能产生有效入场（在持币时买入）的退化策略都会产生敞口为零的吸收态现金状态 $S _ { t } = 0$。我们将这种投资组合层面的吸收行为称为清算均衡（Liquidation Equilibrium）。在我们的实验中，离散基线所观察到的清算均衡具体是通过单调的“始终卖出”预测 $( g _ { t } = \mathrm { S e l l } )$ 实现的，该预测会立即（或持续）推动并维持 $S _ { t } = 0$。这一定义明确了基线中所观察到的“清算均衡”（第 6.1 节）对应于吸收态 $S _ { t } = 0$，当模型无论输入为何都单调地预测 $g _ { t } = \mathrm { S e l l }$ 时便会达到该状态。

## 5.1.3 评估指标

为了严格量化所学表征的风险调整后表现，我们在总回报率之外还报告了以下指标：

• 夏普比率（Sharpe Ratio）：衡量单位总波动率所获得的超额回报。定义为

$$
S = \frac {E [ R _ {p} - R _ {f} ]}{\eta},
$$

其中 $R _ { p }$ 为年化投资组合回报率，$R _ { f }$ 为无风险利率（对于 2023–2025 年期间假定为 $4 \%$），而 η 为日回报率的年化标准差。

• 索提诺比率（Sortino Ratio）：夏普比率的一种变体，仅对下行波动进行惩罚。定义为

$$
S _ {o r t} = \frac {E [ R _ {p} - R _ {f} ]}{\sigma_ {d}},
$$

其中 $\sigma _ { d }$ 为负资产回报率的标准差（下行偏差）。该指标特别契合我们的“主动风险管理”目标，该目标旨在最小化资本损失，而非方差本身。

• 最大回撤（Maximum Drawdown, MaxDD）：测试期内投资组合权益曲线从峰值到谷值的最大观察损失：

$$
\mathrm{MaxDD} = \min _ {t} \left(\frac {C _ {t} - \max _ {s \leq t} C _ {s}}{\max _ {s \leq t} C _ {s}}\right),
$$

其中 $C _ { t }$ 为在时间 t 的投资组合价值。这量化了持有该策略的投资者所面临的最坏情况。

## 5.2 结果

我们分三个阶段展示我们的研究结果：（1）行为诊断（Behavioral Diagnostics），用于证实模型确实在学习某种策略而非发生崩溃；（2）财务表现（Financial Performance），量化该策略的价值；以及（3）鲁棒性（Robustness），验证跨阈值和成本的稳定性。

## 5.2.1 行为诊断：摆脱清算均衡

在评估收益之前，我们严格评估该任务的可学习性。鉴于非对称损失 $( w _ { s e l l } = 1 0 . 0 )$ ，信号置信度较低的模型会在理性上收敛至“清算均衡（Liquidation Equilibrium）”（始终卖出/持有现金）以最小化风险。

Figure 3 可视化了 NVDA 交易研究中的这种差异。离散基线（PatchTST）崩溃为单调分布（动作比率 → 100% 卖出），而 SplineGPT 则保持了平衡的三元分布。这证实了连续样条表示提供了足够的信噪比来识别可操作的信息，并证明承担入场风险是合理的。

![](images/c971830cfcd770fbb3845a9148d6a4e982fec479abfe73bbe402732b81f82daf.jpg)

![](images/958e5db985aa35781a0d46a8844f8dba84af4e20d7b1cb695878b18abbbcf545.jpg)

![](images/22e5c91743a67df6b8648428d26f0e7dc71ffe31b7ed23708c945f1cce45d222.jpg)  
Figure 3: 行为差异与校准（NVDA）。(A) 动作分布显示离散基线（PatchTST，红色）崩溃为狭窄的密度峰值（清算均衡），而 SplineGPT（蓝色）保持了宽泛且活跃的策略分布。(B) 混淆矩阵证实 SplineGPT 在持有（2）和动作 $( 0 / 1 )$ 状态之间实现了有意义的区分，而不是默认归为单一类别。(C) 校准图表明模型对“买入”信号的概率估计跟踪了经验胜率，紧贴对角线（完美校准）。

## 5.2.2 财务表现

在确立了行为有效性之后，我们在 Table 3 和 4 中报告了财务结果，并在 Figure 6 中可视化了累积资产净值曲线。SplineGPT 在下行风险（最大回撤，Max Drawdown）方面稳定优于市场基准，同时在动量机制（NVDA、TSLA）中捕获了显著的上行收益。值得注意的是，在 PFE 暴跌场景中，该模型识别负运动学特征（加加速度，Jerk）的能力使其得以提早退出，在该资产下跌 ≈ 40% 的情况下保持了平坦的资产净值曲线。

<table><tr><td></td><td colspan="5">总回报率（累计）</td><td colspan="5">最大回撤（风险）</td></tr><tr><td>Ticker</td><td>Spline</td><td>Patch</td><td>FD</td><td>Raw</td><td>Mkt</td><td>Spline</td><td>Patch</td><td>FD</td><td>Raw</td><td>Mkt</td></tr><tr><td>INTC</td><td>+240.4%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>+71.6%</td><td>-34.7%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>-48.2%</td></tr><tr><td>NVDA</td><td>+579.2%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>+1185.7%</td><td>-32.1%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>-56.4%</td></tr><tr><td>JPM</td><td>+82.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>+159.9%</td><td>-12.5%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>-22.1%</td></tr><tr><td>XOM</td><td>+48.4%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>+28.5%</td><td>-13.5%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>-26.3%</td></tr><tr><td>PFE</td><td>+3.6%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>-41.2%</td><td>-18.8%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>-44.5%</td></tr><tr><td>TSLA</td><td>+699.8%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>+306.2%</td><td>-33.4%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>-63.5%</td></tr></table>

Table 3: 绝对性能指标。所有非样条（non-spline）基线模型均陷入清算均衡（0.0% 回报/风险）。SplineGPT 生成了具有卓越资金保护能力（MaxDD）的主动策略。

<table><tr><td></td><td colspan="5">夏普比率</td><td colspan="5">索提诺比率</td></tr><tr><td>Ticker</td><td>Spline</td><td>Patch</td><td>FD</td><td>Raw</td><td>Mkt</td><td>Spline</td><td>Patch</td><td>FD</td><td>Raw</td><td>Mkt</td></tr><tr><td>INTC</td><td>1.24</td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.85</td><td>1.13</td><td>N/A</td><td>N/A</td><td>N/A</td><td>1.12</td></tr><tr><td>NVDA</td><td>2.31</td><td>N/A</td><td>N/A</td><td>N/A</td><td>2.80</td><td>2.16</td><td>N/A</td><td>N/A</td><td>N/A</td><td>3.95</td></tr><tr><td>JPM</td><td>0.98</td><td>N/A</td><td>N/A</td><td>N/A</td><td>1.60</td><td>0.88</td><td>N/A</td><td>N/A</td><td>N/A</td><td>2.25</td></tr><tr><td>XOM</td><td>0.61</td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.55</td><td>0.61</td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.78</td></tr><tr><td>PFE</td><td>-0.19</td><td>N/A</td><td>N/A</td><td>N/A</td><td>-0.80</td><td>-0.18</td><td>N/A</td><td>N/A</td><td>N/A</td><td>-1.05</td></tr><tr><td>TSLA</td><td>2.20</td><td>N/A</td><td>N/A</td><td>N/A</td><td>1.15</td><td>2.23</td><td>N/A</td><td>N/A</td><td>N/A</td><td>1.65</td></tr></table>

Table 4: 风险调整后效率。处于清算均衡的策略其风险调整指标未定义。在非对称目标下，PatchTST、FD 和 Raw 均收敛至该均衡（通过单调的“始终卖出”预测实现），导致夏普/索提诺比率为 N/A。SplineGPT 则产生了非平凡的操作行为和可衡量的效率。

## 5.2.3 稳健性与敏感性分析

为了进一步验证这些结果并非特定阈值（τ = 1%）或理想化执行的人为产物，我们以 NVDA 为案例研究进行了全面的参数扫描。图 4 表明，SplineGPT 在较宽范围的交易成本（高达 30bps）和波动率阈值下均能保持正期望。相反，即使将阈值 τ 降低至 0.25%，离散基线模型仍未能激活，这表明存在根本性的表征缺陷，而非调参不匹配所致。

## 5.2.4 运动学响应性与市场状态自适应

最后，我们考察了导致表 3 中观察到的卓越下行保护的机制。图 5 可视化了 XOM 中高频“洗盘”（whipsaw）的场景。

至关重要的是，尽管模型在突破尝试时触发了动量入场（绿色三角），但在该走势失败时并未盲目持有。如子图 B 和 C 所示，加速度（Acceleration）token $\left( c _ { 2 } \right)$ 的迅速衰减充当了趋势衰竭的先行指标。模型检测到了这种运动学崩溃，并在主要价格崩盘发生之前切换到了“卖出”状态（红色三角）。这证实了高阶导数 token 使得策略能够比价格滞后指标更快地区分健康的动量与停滞的价格走势。

![](images/f028e15f3dfb9db7d56612920d052def754ed6a96f30c3f039570db2457465cf.jpg)

![](images/505804b4d8cc079344604b3286514130996286b2b6204740da278cce12737324.jpg)  
Figure 4: 参数敏感性分析（NVDA）。(A) SplineGPT 表现出理性选择性：随着波动率阈值（τ）的增加（使任务更具挑战性），模型单调地降低其操作率，从而避免假阳性。(B) 即使在高摩擦机制下（高达 30bps），性能依然保持稳健（夏普比率 > 2.0），验证了由运动学 token 产生的 alpha 不仅仅是微观结构噪声套利。

## 6 讨论

实验结果为在此日度股票测试平台中支持物理信息分词（Physics-Informed Tokenization）假说提供了证据。通过将 SplineGPT 与层级基线（物理消融（RawGPT）、离散有限差分近似（RawGPT-FD）和基于补丁的聚合（PatchTST））进行基准对比，我们观察到了一致的定性分离：在不对称操作目标下，基线模型收敛于弃权（“持有”）均衡，而 SplineGPT 则学习到了稳定的选择性交易策略。这种分离是我们论点的核心：其贡献不仅在于预测精度，更在于当弃权属于理性行为的状态下，非平凡行为的可学习性与稳定性。

## 6.1 信号不足与清算均衡

一个关键发现是离散基线模型的行为崩溃。尽管绝对收益指标（表 3）显示 PatchTST 为 0.0%，但诊断分析（图 3）揭示了这是由于收敛至清算均衡（100% 卖出/现金）所致。

这一现象是对非对称损失函数 $( w _ { s e l l } = 1 0 . 0 )$ 的理性响应。为了触发“买入”动作，模型对价格上涨的后验概率必须超过该惩罚所隐含的风险调整阈值。这种崩溃表明标准离散标记化（tokenization）施加了信号置信度壁垒：离散输入特征无法提供足够的信息增益来克服先验损失概率，从而迫使优化器默认采用全局风险最小化状态（现金，Cash）。相比之下，SplineGPT 保持了平衡的主动策略。这意味着连续运动学 Token（特别是速度和加速度）成功提取了足够的可操作信号以证明承担风险敞口的合理性，从而有效地解析了对离散表示而言仍不透明的潜在结构。

(B) 运动学 Token：速度

![](images/bb5ce181141a0829e5d5b408bf0bcecb7c1e1c9bff33cde28f00821b06ec97b3.jpg)

![](images/60820779f4455331dd75eaaf8f54c2c8f1e40a4c583e1a1b2d8e3ba36b069212.jpg)

![](images/0e86d5c56df7258384ed7378eec71bf1ec73271cf402fc960f8e4b438710c664.jpg)  
Figure 5: 运动学响应性（XOM）。这次假突破非但不是一次失败，反而展示了该模型的快速适应能力。尽管高速度 $\left( c _ { 1 } \right)$ 触发了初始入场，但加速度的立即反转（$( c _ { 2 } ,$ 图 C）及时提醒了模型突破失败。该策略在数日内翻转为“卖出”（红色三角形），从而保护资金免受随后约 10% 下跌的影响。这种迅速的反应速度解释了其相对于市场极低的最大回撤指标。

跨行业性能验证（2023 – 至今）  
![](images/47a6fb9e9ef6f2b8fbfa939e3177ab2e7af7eb7eef6b0d67e80708c63c12fafc.jpg)  
Figure 6: 跨行业性能验证（2023 – 至今）。左图：SplineGPT 策略使用主动风险管理权重生成的净值曲线。右图：相同资产的买入持有（Buy & Hold）基准。请注意该策略平滑波动的能力（TSLA），以及在 PFE 暴跌期间（红线）严格保存资本的能力——在资产下跌 40% 的情况下保持了平稳的收益曲线。消融基线已被略去，因为它们收敛到了清算均衡（Liquidation Equilibrium），在此体现为单调的“始终卖出”动作策略；参见 Table $^ { 3 , }$ 4 和 5。

<table><tr><td rowspan="2">模型</td><td colspan="3">动作分布</td><td colspan="3">性能</td></tr><tr><td>买入</td><td>卖出</td><td>持有</td><td>动作率</td><td>收益率</td><td>夏普比率</td></tr><tr><td>PatchTST</td><td>0</td><td>762</td><td>0</td><td>100.0%</td><td>0.0%</td><td>N/A*</td></tr><tr><td>SplineGPT</td><td>294</td><td>228</td><td>240</td><td>68.5%</td><td>+579.2%</td><td>2.31</td></tr></table>

Table 5: 离散Token的清算均衡（Liquidation Equilibrium）。我们在NVDA的标准波动率阈值（$\tau = 1.0\%$）下，对基线模型（PatchTST）与SplineGPT的具体动作分布进行了诊断。虽然SplineGPT保持了平衡的交易策略，但离散基线模型却崩溃为单调的“始终卖出”（Always Sell）策略。这表明离散表示未能生成足够的信号置信度来克服非对称风险惩罚（$w_{sell} = 10.0$），迫使模型进入最安全的状态（100%现金）。  
<sup>∗</sup>对于静态现金持仓（零波动率），夏普比率（Sharpe ratio）未定义。

## 6.1.1 敏感性扫描分析

为了验证基线模型的失效不仅仅是由于阈值过于保守所致，我们进行了敏感性扫描，将波动率阈值降低至 $\tau = 0.5\%$ 以鼓励模型采取行动。如Table 6所示，离散基线模型即使在这种激进的设定下也未能恢复，仍然困在清算均衡中。相反，SplineGPT成功实现了自适应，将其动作率提升至90.8%，同时将其夏普比率提高到2.48，并通过更频繁地进行主动投资，更好地捕捉了标的NVDA股票的极端看涨特性。这表明性能差距在于表征层面：在此标准PatchTST架构下，无论阈值如何，离散测量token（相比于离散样条token）都缺乏支持非平凡决策边界所需的信噪比。

<table><tr><td rowspan="2">条件(τ 阈值)</td><td colspan="2">PatchTST</td><td colspan="2">SplineGPT</td><td colspan="2">差异</td></tr><tr><td>动作率</td><td>夏普比率</td><td>动作率</td><td>夏普比率</td><td>行为</td><td>结果</td></tr><tr><td>极激进(τ = 0.25%)</td><td rowspan="4">100.0% (卖出)</td><td rowspan="4">N/A</td><td>100.0%</td><td>2.73</td><td>积极</td><td>高收益</td></tr><tr><td>激进(τ = 0.50%)</td><td>90.8%</td><td>2.48</td><td>积极</td><td>高收益</td></tr><tr><td>保守(τ = 1.00%)</td><td>67.1%</td><td>2.20</td><td>选择性</td><td>稳健</td></tr><tr><td>极保守(τ = 2.00%)</td><td>38.3%</td><td>1.74</td><td>精准</td><td>防御性</td></tr></table>

Table 6: 敏感性扫描下的行为差异（NVDA）。我们将波动率阈值（$\tau$）从0.25%变化到2.0%。虽然SplineGPT理性地调整了其活动规模——从激进设定下的100%活动率到保守设定下的38%精准度——但在所有阈值下，离散基线模型（PatchTST）都困在清算均衡（100%卖出）中。在最激进的设定下，SplineGPT同样表现出100%的动作率，但这包含了买入和卖出决策的混合。这证实了基线模型的失效是根本性的信号不足，而非调参所致的偶然现象。

## 6.2 显式平滑与隐式平滑

PatchTST 主要是针对 MSE 目标下的连续预测进行优化的，而我们的下游任务是一个离散的、非对称的动作分类问题；因此，该比较应被解释为行为鲁棒性测试，而非纯粹的预测基准测试。一个合理的驱动因素是通道独立性（Channel Independence, CI）：PatchTST 将价格（Price）和成交量（Volume）作为孤立的单变量序列进行处理，而 SplineGPT 则将它们耦合为一个统一的 9 维运动学标记（token）。

## 6.3 目标诱导的保守性

在抛物线式行情（例如 NVDA、JPM）中，SplineGPT 获得了正收益，但落后于买入持有（Buy & Hold）基准。这种分歧并非预测失败，而是非对称风险目标 (10) 的结构性结果。标准的趋势跟踪针对总回报（R）进行优化，隐式地接受了无限的回撤风险。相反，我们的目标函数是一个约束优化问题：在严格的下行惩罚约束下最大化 R。因此，该模型表现出凸性削减（convexity shedding）——在中间的均值回归事件中退出仓位以重置风险敞口。这一行为将该模型表征为一个“状态切换”（regime-switching）智能体，而非纯粹的动量因子，即牺牲绝对收益上限以换取更高的风险调整后下限（Sharpe/Sortino）。

## 6.4 对执行成本与换手率的敏感性

为了评估生成信号的实际可实现性，我们针对每笔交易从 0 到 20 个基点（bps）的不同交易成本区间进行了敏感性分析。Table 7 汇总了每种资产的风险调整后收益（夏普比率，Sharpe Ratio）和年化换手率。

我们观察到较高的年化换手率（36x–89x），这表明该模型学到的是短期主动交易策略，而非被动持有策略。尽管存在这种高换手率行为，该策略仍对交易摩擦表现出显著的抵御能力：

• 高波动性资产（NVDA、TSLA）：即使在严苛的 20bps 成本假设下，仍保持强劲的风险调整后表现（Sharpe > 0.49），证实了运动学信号捕捉到的价格变动幅度足以克服巨大的摩擦。

• 较低波动性资产（XOM、JPM）：表现出更高的敏感性。虽然 JPM 依然保持稳健，但 XOM 的优势在接近 10bps 阈值时发生衰减。这符合以下预期：在“物理”力量（动量、加速度）显著的市场环境中，基于样条的导数能够提供最大的价值。

<table><tr><td>股票代码</td><td>Sharpe (0 bps)</td><td>Sharpe (5 bps)</td><td>Sharpe (10 bps)</td><td>Sharpe (20 bps)</td><td>年化换手率</td></tr><tr><td>INTC</td><td>1.31</td><td>1.19</td><td>1.06</td><td>0.83</td><td>77.7x</td></tr><tr><td>NVDA</td><td>2.39</td><td>2.26</td><td>2.12</td><td>1.87</td><td>70.3x</td></tr><tr><td>JPM</td><td>1.15</td><td>1.07</td><td>0.99</td><td>0.82</td><td>36.0x</td></tr><tr><td>XOM</td><td>0.73</td><td>0.62</td><td>0.50</td><td>0.27</td><td>47.6x</td></tr><tr><td>PFE</td><td>-0.31</td><td>-0.43</td><td>-0.56</td><td>-0.83</td><td>44.6x</td></tr><tr><td>TSLA</td><td>2.05</td><td>1.90</td><td>1.76</td><td>1.48</td><td>89.7x</td></tr></table>

Table 7: 交易成本敏感性。我们报告了在不同执行成本假设（每笔交易 0–20 个基点）下的夏普比率（Sharpe Ratios）。尽管换手率较高（> 35x），但动量资产（NVDA、TSLA、INTC）即使在 20bps 下仍保持强劲的风险调整后收益，这证实了运动学信号捕捉到了超过典型摩擦成本的显著价格变动。

## 6.5 Alpha 生成机制

鉴于观察到 SplineGPT 与表现出清算均衡（Liquidation Equilibrium）的基准模型之间的差异，我们重点指出学习策略似乎受益于平滑耦合导数的两种常见状态机制：

1. 动量凸性（Momentum Convexity）（TSLA, INTC）：在高贝塔（high-beta）资产中，价格走势在突破期间往往表现出显著的凸性（加速度）。SplineGPT 成功捕捉到了与这些上涨行情开始相关的正向“加加速度”（Jerk）(c )。通过根据平滑导数而非充满噪声的价格差采取行动，该模型能够更早建仓，并在轻微波动中保持持仓，而这种波动可能会促使不够复杂的策略平仓卖出。

2. 崩盘规避（Crash Avoidance）（PFE）：对“主动风险管理”（Active Risk Management）目标最关键的验证体现在辉瑞（Pfizer, PFE）的案例中。尽管该资产贬值超过 40%，但该模型成功保护了本金（最大回撤 MaxDD 为 -18.8%）。如图 Figure 7 所示，崩盘期间的运动学特征非常复杂，其特点是高阶导数中的波动率聚集。该模型应对此次崩盘的能力证明了其在 9 维流形中将有意义的结构性断裂与瞬态噪声解耦的能力。

## 7 结论

本研究提出了 SplineGPT，这是一种建立在新型连续分词（continuous tokenization）流水线之上的基础模型架构。通过将金融时间序列视为连续随机过程的物理可观测变量，我们能够提取标准离散分词器无法捕捉的高阶运动学状态（速度、加速度、加加速度）。

我们通过跨多个不同市场板块的严谨且考虑税费的回测对该模型进行了验证。结果表明，该模型学习到了可迁移的运动学规律，这些规律对下游决策规则十分有效，使其能够：

1. 运动学动量捕获（Kinematic Momentum Capture）：利用正凸性 $( c _ { 2 } > 0 )$ 来识别高贝塔（high-beta）趋势的启动（例如 TSLA）。

2. 结构性断裂识别（Structural Break Identification）：利用高阶导数衰减在价格投降式变动之前检测趋势耗尽（例如 PFE）。

3. 跨领域泛化（Cross-Domain Generalization）：学习可在不相关板块（能源、科技、医药）之间迁移的鲁棒特征，而无需针对特定资产进行参数微调。

未来的工作将集中在扩大模型参数量，并将诸如宏观经济指标等其他数据模态集成到连续 Token 流中。此外，尽管“主动风险管理”（Active Risk Management）策略起到了概念验证的作用，但还可以探索替代的损失加权方案，以针对不同的风险偏好训练适配器。例如，激进的“风险偏好”（risk-on）微调可能更适合捕捉抛物线形态行情的全部上行空间，例如 NVDA 所表现出的行情。此外，本文所采用的三元买入/卖出/持有（Buy/Sell/Hold）分类仅代表一种潜在的下游应用。SplineGPT 可以针对多种目标进行微调，包括用于波动率预测的回归头或用于截面板块轮动的排序头。未来的研究将利用动作率和校准诊断（例如混淆矩阵和 $\tau$ 扫描）来进一步将目标诱导的观望与表征不足区分开来。

归根结底，这项工作的意义超越了金融领域。我们建立了一个通用的物理信息 Token 化（Physics-Informed Tokenization）框架，适用于观测值源自连续隐状态的任何领域。与金融不同，当物理第一性原理存在时（例如力学、电磁学、流体力学等），基础定律会被纳入随机微分方程（SDE）中，从而针对每个领域自适应地修改样条结构。通过在 Token 化层本身正式编码控制动力学，而不是依赖模型从头开始学习它们，该方法解决了离散 AI 架构与连续物理世界之间的根本性脱节。这一范式转变有望显著加速鲁棒的、以物理为基础的 AI 系统的开发。

![](images/2ef2e0461f8584c6622eb626bf0fffdb575d7d416e37251ad6842c546752dfcb.jpg)

![](images/143fdfe5664f1302ce5c9b3fb6c26b25cb37284913954678e3486c7bf4931a01.jpg)

![](images/0add466378d4781e9232db6b09b590cf5b9840a1afff26cabaddc05b0401df31.jpg)

![](images/c581b01f2c988f3b0813ceedb4e0f18b3d352a9988954a13334842c30119bef0.jpg)

![](images/f2975e6daf44717af7c4b417809dc06d93fa92392fdccb5ca43a0eacf33a7408.jpg)

![](images/fdcff001956dd6230441bc1e5ac77c3a94a1f90cac05781a720b701ec2c61d17.jpg)

![](images/ce14a65c84cca931d8858acbcd0ff01afa7ca42311add85bb7a825b354f8c1e6.jpg)

![](images/9048641ba9d08d598a93824fd4247308609dd53168388f269207bfcb59e9f9be.jpg)

![](images/5f57c14d611002df9099eb96cbe2a009d8502b21e777276e0d6db7c6d5fdea9c.jpg)

![](images/5fcc6786aa5d2d338212002a9033b544fd4686c97767e82e0a250f7d06dd98df.jpg)  
Figure 7: 9 维运动学流形与真实值对比。顶部面板：辉瑞（Pfizer, PFE）的标准 OHLC K线图与成交量图，记录了投降式抛售事件（12 月 13 日）以及随后的复苏过程。下方面板：由 SplineGPT 提取的显式运动学标记（tokens）。左列：价格导数 $\left( c _ { 0 } – c _ { 3 } \right)$。右列：成交量导数 $\left( \tilde { c } _ { 0 } – \tilde { c } _ { 4 } \right)$。尽管“真实值”（顶部）呈现出充满噪声且模棱两可的筑底过程，但下方面板揭示了模型可获取的丰富高频信号结构。高阶项（速度 $c _ { 1 }$ 与成交量流加速度 $\tilde { c } _ { 2 } )$ 中显著的波动率聚集现象突显了信号的复杂性。

## 参考文献

[1] Grifin M. Kearney and Makan Fardad. “Optimization based data enrichment using stochastic dynamical system models”. In: PLOS ONE 19.9 (Sept. 2024), e0310504. doi: 10.1371/journal.pone.0310504. url: https://doi.org/10.1371/journal.pone. 0310504.

[2] Simo S¨arkk¨a and Lennart Svensson. Bayesian Filtering and Smoothing. 2nd ed. Cambridge University Press, 2023.

[3] Yuqi Nie et al. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. 2023. arXiv: 2211.14730 [cs.LG]. url: https://arxiv.org/abs/2211.14730.

[4] Yong Liu et al. iTransformer: Inverted Transformers Are Efective for Time Series Forecasting. 2024. arXiv: 2310.06625 [cs.LG]. url: https://arxiv.org/abs/2310.06625.

[5] Matthew Faw et al. “In-Context Fine-Tuning for Time-Series Foundation Models”. In: Proceedings of the 42nd International Conference on Machine Learning (ICML). 2025. url: https://icml.cc/virtual/2025/poster/43707.

[6] Zhuangwei Shi. MambaStock: Selective state space model for stock prediction. 2024. arXiv: 2402.18959 [cs.CE]. url: https://arxiv.org/abs/2402.18959.

[7] Md Atik Ahamed and Qiang Cheng. TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting. 2024. arXiv: 2403.09898 [cs.LG]. url: https://arxiv.org/ abs/2403.09898.

[8] Ramin Hasani et al. Liquid Time-constant Networks. 2020. arXiv: 2006.04439 [cs.LG]. url: https://arxiv.org/abs/2006.04439.

[9] Hang Liu et al. “Physics-Informed Koopman Networks for Industrial Process Time-Series Prediction”. In: Industrial & Engineering Chemistry Research 64.37 (2025), pp. 18328– 18346. doi: 10.1021/acs.iecr.5c02217.

[10] M. Raissi, P. Perdikaris, and G.E. Karniadakis. “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations”. In: Journal of Computational Physics 378 (2019), pp. 686– 707. issn: 0021-9991. doi: https://doi.org/10.1016/j.jcp.2018.10.045. url: https://www.sciencedirect.com/science/article/pii/S0021999118307125.

[11] Tibor Socaciu and Paul Pa¸scu. “Physics-Informed Neural Networks in Pricing Financial Options”. In: Broad Research in Artificial Intelligence and Neuroscience 16.2 (2025), p. 474.

[12] Justin A. Sirignano. “Deep learning for limit order books”. In: Quantitative Finance 19.4 (2019), pp. 549–570. doi: 10.1080/14697688.2018.1546053.

[13] Zihao Zhang, Stefan Zohren, and Stephen Roberts. “DeepLOB: Deep Convolutional Neural Networks for Limit Order Books”. In: IEEE Transactions on Signal Processing 67.11 (June 2019), pp. 3001–3012. issn: 1941-0476. doi: 10.1109/tsp.2019.2907260. url: http://dx.doi.org/10.1109/TSP.2019.2907260.

[14] J. O. Ramsay and B. W. Silverman. Functional Data Analysis. 2nd. New York: Springer, 2005. doi: 10.1007/b98888.

[15] Yann A. LeCun et al. “Eficient BackProp”. In: Neural Networks: Tricks of the Trade. Ed. by Gr´egoire Montavon, Genevieve B. Orr, and Klaus-Robert M¨uller. Vol. 7700. Lecture Notes in Computer Science. Springer, 2012, pp. 9–48. doi: 10.1007/978-3-642-35289- 8\_3.