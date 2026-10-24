# 基于时间卷积网络的低功耗边缘微控制器 sEMG 手部运动学回归

Marcello Zanghieri<sup>1</sup>, Simone Benatti<sup>1,2</sup>, Alessio Burrello<sup>1</sup>, Victor Javier Kartsch Morinigo<sup>1</sup>, Roberto Meattini<sup>1</sup>, Gianluca Palli<sup>1</sup>, Claudio Melchiorri<sup>1</sup>, Luca Benini<sup>1,3</sup>

摘要— 基于手势控制的人机界面（Human-Machine Interfaces, HMI）是一个非常活跃的研究领域，旨在实现与物体的自然交互。如今，用于机械手控制的最有前景的最先进（State-of-the-Art, SoA）方法之一依赖于表面肌电（surface electromyographic, sEMG）信号，当与基于深度学习（Deep Learning, DL）的解码算法相结合时，这种非侵入式方法可以提供精确且直观的控制。然而，到目前为止，绝大多数方法都侧重于 sEMG 分类，导致控制系统将手势限制在预定义的一组姿态中。相比之下，sEMG 回归仍是一个新兴领域，它提供了一种更自然、更完整的控制方法，能够返回完整的手部运动学信息。本工作提出了一种基于 TEMPONet 的回归框架，TEMPONet 是一种用于 sEMG 解码的 SoA 时间卷积网络（Temporal Convolutional Network, TCN），我们对其进行了面向部署的进一步优化。我们在 NinaPro DB8 数据集上测试了该方法，针对执行 9 组对侧动作的 12 名受试者（10 名健全人和 2 名经桡骨截肢者）估计 5 个连续自由度。我们的模型实现了 6.89° 的平均绝对误差（Mean Absolute Error），比 SoA 提升了 0.15°。得益于 int8 量化，我们的 TCN 在仅占用 70.9 kB 内存的情况下达到了这一精度。这是非常显著的，因为如果不采用量化或剪枝等面向部署的精简技术，用于 sEMG 的高精度 SoA 神经网络模型大小可达数十 MB。我们将模型部署在 GAP8 边缘微控制器上，获得了 4.76 ms 的执行延迟和每次推理 0.243 mJ 的能耗，表明我们的方案非常适合在资源受限设备上实现实时控制。

## I. 引言与相关工作

用于手势控制的人机界面（HMI）正日益受到关注，因为它们能够利用自然手势实现与物体的智能交互 [1]。实现手势控制的一种常用方法依赖于利用机器学习（Machine Learning, ML）[3] 以及近期的深度学习（DL）技术 [4], [5]，将表面肌电图（Surface electromyography, sEMG）信号 [2] 映射到一组预期手势上。

传统的 ML 算法（如 LDA、SVM 和 ANN）可实现 80% 以上的分类准确率 [3], [6]，不过它们的表现逊色于 DL 模型；DL 模型能够利用更大的数据集、利用特征学习能力，并直接处理 sEMG 信号的时间加窗，而无需预先提取特征 [7], [8]。

尽管使用 sEMG（表面肌电）信号使得传统的 ML/DL（机器学习/深度学习）方法能够以相对自然的程度控制手势和抓握动作，但上述系统受限于有限的预定义静态姿势集合（例如握拳、捏取、食指指向等），这无法实现完全自然且多功能的控制。

因此，利用基于 sEMG 的回归来针对手部运动学建模是一个极具前景的研究方向。已有几种 DL 方法可以解决回归问题。然而，由于模型尺寸庞大，它们中的大多数计算密集且需要占用大量内存 [4], [5], [9]，从而阻碍了在具有实时执行约束的边缘设备上的可部署性。

因此，针对高能效平台的可靠 DL 框架需要精细的多模态软硬件协同设计（HW-SW codesign）。具体而言，在找到高精度的深度模型之后，有必要最小化其尺寸（例如通过量化 [10]），然后确定模型在目标嵌入式设备上运行推理时的延迟和功耗。值得注意的是，大多数针对 sEMG 提出的卷积网络仅以高达数十 MB 的模型大小为代价达到最先进的（State-of-the-Art, SoA）精度 [4]，缺乏面向部署的优化，例如步幅（stride）与膨胀（dilation）、量化 [10] 或剪枝，而这些优化可以在精度损失极小的情况下将模型尺寸缩减 10 倍以上 [8]。

sEMG 回归任务已在若干启发性研究中得到探讨，其区别在于所针对的目标变量：运动学变量（关节角度、关节速度）或动力学变量（手指力）。关节角度的位置是手部运动学的最佳指标，因此它是手势回归中最常用的参数。例如，在 [11] 中，回归目标是数据手套（dataglove）的关节角度，使用维纳滤波器（Wiener filter）达到了 0.63 的中位数 $R ^ { 2 }$。然而，由于该工作更侧重于用户感知的控制质量，因此并未测试其他算法以提高回归得分。在 [12] 中，长短期记忆（Long Short-Term Memory, LSTM）深度网络被应用于同一数据集，获得了 7.04° 的平均绝对误差（Mean Absolute Error, MAE）。该工作的局限性在于未探索纯卷积网络，而纯卷积网络更易于并行化，因此更适用于嵌入式控制。

针对手部运动学的另一种替代方法侧重于手部关节运动的速度。例如，[13] 将速度作为目标，采用了一种混合分类-回归架构，将速度阈值化为 3 个级别，因此仍将预测限制在离散类别中。sEMG 回归的一种完全正交的方法侧重于手部动力学而非运动学。文献 [14] 的工作针对多自由度（multiple-Degrees-of-Freedom, multi-DoF）力估计，然而，力估计是一个应用范围更受限的领域，因为它仅局限于抓握动作。

上述所有工作的一个主要缺点是，它们都没有解决在嵌入式控制设备上部署的问题。特别是，它们没有讨论其模型如何应对资源受限的平台，也没有探索诸如模型搜索（model search）、量化（quantization）和剪枝（pruning）等技术。

在本文中，我们致力于解决基于 sEMG 回归以解码手部运动学的挑战。我们提出了一个基于时间卷积网络（Temporal Convolutional Network, TCN）的回归框架，这是一种用于时间序列建模的深度学习（DL）模型，适用于在资源受限设备上进行实时运行。我们提出了以下贡献：

‚ 我们在 NinaPro DB8（一个包含经前臂截肢者的基准 sEMG 回归数据集）上应用了最先进的（SoA）TEMPONet TCN 架构，获得了低至 6.89° 的平均绝对误差（Mean Absolute Error），即使我们模型的位宽降低到 int8，而 SoA 网络采用 float32，该结果仍比该数据集的 SoA 好 0.15°。

‚ 我们进一步优化了我们的 TCN 框架，确定了在回归误差与内存占用及乘累加（MAC）操作之间取得最佳权衡的模型大小；我们在保持回归精度的同时，将模型大小限制在 70.9 kB，操作次数限制在 3.16 MMACs。

‚ 我们将解决方案部署在 GAP8 边缘微控制器 [15] 上，测量了延迟和功耗方面的性能。我们获得了每次推理 4.76 ms 的延迟和 0.243 mJ 的能耗，证明了我们的回归 TEMPONet 适用于实时运行的边缘低功耗节点。

## II. 材料与方法

## A. 表面肌电信号

肌电（EMG）信号 [16], [17], [18] 是肌肉活动的极佳指标，因为它起源于穿过肌纤维膜的离子电流，由中枢神经系统通过运动神经元传播的电刺激触发。EMG 的典型幅度和带宽分别为 10 µV´1 mV 和最高 2 kHz。使 EMG 信号具有挑战性的是其固有的信号变异性和噪声源，主要由浮地、串扰、工频干扰和运动伪影引起 [6], [3]。

sEMG 是通过放置在肌肉上方皮肤表面的导电电极采集的，并由仪器差分放大器进行收集。凭借其非侵入性，sEMG 是成功实现人机接口（Human-Machine Interfaces, HMIs）最流行的方法之一。

## B. NinaPro 数据库 8

非侵入式自适应手部假肢数据库 8（Non-Invasive Adaptive hand Prosthetics Database 8, NinaPro DB8）[11], [19] 是一个用于手指位置解码的公开 sEMG 数据库，旨在作为运动学估计/重建而非手势或抓握分类的基准。特别是，对侧运动被用作 sEMG 回归的目标。

NinaPro DB8 包含 10 名身体健全的受试者和 2 名右侧经桡骨截肢者。所有参与者重复 9 种<sup>1</sup>双侧镜像运动，每次持续约 6 s - 9 s（即特意放慢以用于瞬态建模），中间穿插约 3 s 的休息。肌肉电位采用 16 个有源双差分传感器（来自 Delsys Trigno IM Wireless EMG 系统）进行记录，传感器分两排、每排 8 个单元布置在受试者右前臂对应肱桡关节处。手部运动学通过佩戴在左手（与 sEMG 电极所在前臂对侧）的 18 自由度（Degree of Freedom, DoF）Cyberglove 2 采集，用于测量 18 个数据手套关节的角度。所有信号均上采样至 2 kHz 并进行了后同步处理。

在此 $\mathbf { X } ( t ) \ \mapsto \ \mathbf { Y } ( t )$ 多变量回归表述中，输入信息为 16 通道 sEMG 信号，目标由 5 个驱动自由度（Degree of Actuations, DoA）表示，这些 DoA 定义为手套 18 个 DoF 的线性组合 [11]。这种 DoF 到 DoA 的降维旨在 (i) 舍弃或缩放不相关的 DoF，以及 (ii) 直接针对定义为相关手部运动的 DoA。

NinaPro DB8 上的 SoA（最先进水平）代表是 [12] 中的 LSTM<sup>2</sup>，其达到了 7.04° 的 MAE。然而，由于采用 float32 数值格式，该 LSTM 不适合部署在低功耗嵌入式平台上。此外，作者并未报告所使用的 LSTM 的隐藏单元数量、参数量以及 MACs 等关键信息。此外，LSTM 通常比卷积模型更难训练 [21]，这一额外缺点促使我们探索时序卷积网络（Temporal Convolutional Networks，如接下来的第 II-C 小节所述）。

## C. Temporal Convolutional Networks

在这项工作中，我们将信号视为时间序列来解决 sEMG 到运动学的回归问题，探索了 TEMPONet（Temporal Embedded Muscular Processing Online Network）拓扑结构 [7] 的变体，这是一种专为 sEMG 分类设计的时序卷积网络（Temporal Convolutional Network, TCN）。

![](images/bba7474e23093c2f4e30836a2d06c1edc93361ab846775501677a5ee47517b7d.jpg)  
Fig. 1. TEMPONet TCN 架构，适配于回归任务，并与其最初提出的版本 [7] 相比进行了进一步优化。

1) TCN 背景：TCN 是一类新型 DL 模型，已在多项时间序列任务中达到 SoA 水平，在训练调优和精度方面均优于循环神经网络（Recurrent Neural Networks, RNN）[22], [21], [9]。TCN 的两个具体特性体现在其沿时间维度的 1D 卷积中：(i) 因果性：卷积核仅覆盖每个样本的左邻域，因此不会窥探任何未来样本；(ii) 膨胀：滤波器输入之间插入固定步长 d，从而在模型大小恒定的情况下扩大感受野。因此，膨胀因果卷积的工作方式如下：

$$
\mathbf {y} _ {t} ^ {c _ {\text { out }}} = \operatorname{Conv} (\mathbf {x}) = \sum_ {c _ {\text { in }} = 1} ^ {C _ {\text { in }}} \sum_ {k = 0} ^ {K - 1} \mathbf {W} _ {k} ^ {c _ {\text { in }}, c _ {\text { out }}} \mathbf {x} _ {t - d \cdot k} ^ {c _ {\text { in }}}\tag{1}
$$

其中 $\mathbf{x}$ 和 $\mathbf{y}$ 分别为输入和输出特征图，$t$ 为时间索引，$c _ { \mathrm { i n } }$ 和 $c _ { \mathrm { o u t } }$ 分别为输入和输出通道，$K$ 为卷积核大小，$\mathbf { W } \in \mathbb { R } ^ { K \times ^ { \bullet } C _ { \mathrm { i n } } \times C _ { \mathrm { o u t } } }$ 为滤波器张量，$d$ 为膨胀率（dilation）。

2) TEMPONet 架构：我们在本文中提出的网络是对 SOTA TEMPONet [7] TCN 的进一步发展。除了对其进行修改以用于回归任务之外，我们还进行了一项模型探索，旨在确定准确率与网络规模之间的最佳权衡。该网络架构如图 Figure 1 所示，包含 3 个卷积块（Convolutional Block），每个卷积块堆叠了：

- 2 个扩张因果卷积（dilated causal convolution），卷积核大小为 3，具有可变膨胀率 $d$ 和全填充（full padding）；

- 1 个卷积核大小为 5、具有可变步长 $s$ 的卷积，后接一个平均池化层（卷积核大小为 2，步长为 2）。

这 3 个卷积块的膨胀率 $d$ 分别为 2、4、8，步长分别为 $s = 1 , 2 , 4$。在我们对更高效 TEMPONet 的创新探索中，与 [7] 的原始模型相比，我们减少了这 3 个卷积块的通道数。我们将块 I（Block I）的通道数减半（从 32 降至 16），将块 II（Block II）的通道数减半（从 64 降至 32），并对块 III（Block III）探索了不同的通道数（即 32、48、64、96、128 和 192），以寻找回归误差与部署指标（即模型大小、MAC、延迟和能耗）之间的最佳权衡。在 TEMPONet 上进行此项模型搜索是具有创新性的，因为最初提出它的论文 [7] 中并未进行过该探索。

在卷积块之后，3 个全连接（Fully Connected, FC）层执行分类。FC I 的单元数是块 III 通道数的 4 倍（如上文所述的可变数量），FC II 具有 32 个单元，FC III 具有 5 个单元，对应于回归目标的 5 个 DoA。除 FC III 外，所有层均采用线性整流单元（Rectified Linear Unit, ReLU）非线性激活函数，并配备批归一化（Batch-Normalization, BN）以应对内部协变量偏移（internal covariate shift）[23]。FC I 和 FC II 采用 $p _ { \mathrm { d r o p } } ~ = ~ 0 . 5 .$ 的 Dropout 进行训练，以辅助正则化 [24]。

我们提出的优化 TEMPONet 处理一个 256 个样本的输入窗口（128 ms @ 2 kHz），参数量少于 500 k。它直接输入原始信号，因此没有任何预处理或特征提取开销。与原始 TEMPONet 相比，最终选定架构在大小和计算量方面的改进详见 Section III。

## D. 实验设置细节

1) 数据集划分（Dataset Split）：对于 NinaPro DB8 的 12 名受试者中的每一位，均提供了 3 次采集会话（session）。按照数据集作者的建议，第 3 次会话（每个动作重复 2 次）被用作测试集。第 1 次和第 2 次会话（每个动作重复 10 次）被合并并用于 2 折交叉验证（2-fold cross-validation）设置中。

2) 预处理（Preprocessing）：[19] 中提供的 NinaPro DB8 信号已经使用 4 阶巴特沃斯（Butterworth）滤波器在 10 Hz 至 500 Hz 之间进行了带通滤波。经典的 ML 模型，即支持向量机（Support Vector Machine, SVM）和多层感知机（Multi-Layer Perceptron, MLP），需要在每个信号窗口上进行特征提取，它们在从每个通道的 60 ms 窗口中提取的波形长度（Waveform Length, WL）特征上进行训练，SVM 的滑动步长为 100 ms（计算上可承受的最大训练集大小），MLP 的滑动步长为 25 ms。对于 SVM，使用了径向基函数（Radial Basis Function, RBF）核，应用于缩放至单位方差的数据，并且针对每个目标 DoA 分别调整了 C 系数。TCN 则直接在原始 sEMG 信号上进行训练，使用 128 ms 的时间窗口（即 2 kHz 下的 256 个样本）。

3) 机器学习设置（Machine Learning Setup）：模型在 Python 3.8 中实现，SVM 和 MLP 使用 Scikit-learn 0.23，TCN 使用 PyTorch 1.6。所使用的 SVM 是 RBF 核 SVM，MLP 则实现了 3 个隐藏层。

![](images/14039a502c1842e17ab91df8bce764daabfabe3a925d2fff2941469b28c2467b.jpg)  
Fig. 2. 我们的 TEMPONet 在 GAP8 [15] 上的部署指标，作为卷积块 III（Convolutional Block III）通道数的函数。

![](images/8ac663c6d589f0d429c4e78aa820e2a70c1701873195e45dff3123d47ae20fe3.jpg)  
Fig. 3. 我们的 TEMPONet 生成的回归示例：受试者 1 的测试会话，DoA 角度 5；显示来自传感器 1 的 sEMG 信号以供参考。

TCN 使用 MAE 损失、AdaM 优化器、初始学习率 $1 \cdot 1 0 ^ { - 4 }$ 以及小批量（minibatch）大小 64 进行训练。首先，以 float32 格式运行 19 个 epoch；然后，执行 8 位（即 int8）训练后量化，并运行最后一个 epoch 的量化感知训练（quantization-aware training），使用了由 NEMO（NEural Minimizer for tOrch，[25], [26]）实现的参数化裁剪激活（PArameterized Clipping acTivation, PACT）[10]，NEMO 是一个针对在高度内存受限的超低功耗设备上部署的 CNN 最小化开源库。

4) 模型输出后处理（Model output post-processing）：所有模型的输出均通过指数移动平均（Exponential Moving Average, EMA）进行了后处理：

$$
y _ {t} ^ {\prime} = \alpha_ {\mathrm{EMA}} \cdot y _ {t - 1} ^ {\prime} + (1 - \alpha_ {\mathrm{EMA}}) \cdot y _ {t},\tag{2}
$$

其中 $y$ 和 $y ^ { \prime }$ 分别表示未滤波和滤波后的信号，$t$ 为时间索引，$\alpha _ { \mathrm { E M A } } ~ \in ~ [ 0 , 1 ]$ 为衰减因子。衰减因子 α<sub>EMA</sub> 在每个模型（模型训练完成后）针对每个受试者、每个 DoA 分别进行调优，仅使用训练数据并通过网格搜索（grid search）<sup>3</sup>进行优化。这种优化与机器学习（ML）设置保持一致（因为未使用测试数据），旨在权衡 EMA 有益的平滑效果与来自历史值权重的惯性。值得注意的是，EMA 仅使用当前和过去的输出 $y _ { t }$ 进行计算：因此，它不会给系统增加延迟。衰减因子 $\alpha _ { \mathrm { E M A } }$ 所获得的经验值见实验结果（第 III 节）。

## III. EXPERIMENTAL RESULTS

## A. Evaluation metrics

评估回归框架在机械手端到端控制中的有效性并非易事。MAE 的评估固然重要，但必须考虑到，例如，估计角度与真实值之间的差异在运动阶段发生时，相比于在静态阶段对控制的影响更大。

因此，我们通过 MAE（以度为单位度量）以及定义为 MAE 低于容差 $\Theta _ { \mathrm { t o l } }$ 的频率的回归准确率来衡量模型的回归质量：

$$
\Theta_ {\mathrm{tol}} \text {-accuracy} \triangleq \frac {1}{T} \sum_ {t = 1} ^ {T} \mathbb {I} _ {\Theta_ {\mathrm{tol}}} (\mathrm{MAE} (t))\tag{3}
$$

其中 I 表示指示函数

$$
\mathbb {I} _ {\Theta_ {\mathrm{tol}}} (\mathbf {M A E} (t)) \triangleq \left\{ \begin{array}{l l} 1 & \quad \text { if } \mathbf {M A E} (t) <   \Theta_ {\mathrm{tol}} \\ 0 & \quad \text { otherwise }. \end{array} \right.\tag{4}
$$

我们通过经验选取了回归准确率阈值 $\Theta _ { \mathrm { t o l } } =$ $1 0 ^ { \circ } , 1 5 ^ { \circ }$。

这些指标之所以能够可靠地衡量端到端控制质量，原因有以下几点：(i) 它们与角位置的实际尺度相关；(ii) 它们具有统计代表性，因为平均值是在关节、动作和受试者（包括两名经桡骨截肢者）之间计算的；(iii) MAE 是一阶的，因此受异常值的影响小于回归 $R ^ { 2 }$。而基于阈值的回归准确率则更具鲁棒性。

TABLE I  
所探索模型的回归质量，并与 NINAPRO DB8 的 SoA 进行对比。

<table><tr><td>模型</td><td>格式</td><td>MAE</td><td> $10^{\circ}$ 准确率</td><td> $15^{\circ}$ 准确率</td></tr><tr><td>SVM</td><td>fp32</td><td> $7.28^{\circ}$ </td><td>0.795</td><td>0.883</td></tr><tr><td>MLP</td><td>fp32</td><td> $7.14^{\circ}$ </td><td>0.799</td><td>0.889</td></tr><tr><td>LSTM [12]</td><td>fp32</td><td> $7.04^{\circ}$ </td><td>n.a. $^{1}$ </td><td>n.a. $^{1}$ </td></tr><tr><td>TEMPONet $^{2}$ </td><td>int8</td><td> $6.89^{\circ}$ </td><td>0.814</td><td>0.900</td></tr></table>

<sup>1</sup>[12] 的正确性得分（Correctness Scores, CS）是准确率指标，但在某种程度上是经过挑选的，因为每个关节的容差在动态范围裁剪后被固定为百分位数。  
<sup>2</sup>选出的最佳模型：具有 64 个通道的 Block III。

## B. 模型对比

Table I 和 Figure 2 中展示的结果给出了测试的 ML（机器学习，SVM 和 MLP）和 DL（深度学习，TEMPONet）算法，以及用作基线的 [12] 中的 LSTM。特别是，Figure 2 描绘了回归 TEMPONet 最优尺寸的搜索过程，其中 Block III 的通道数在 32、48、64、96、128、196 这些数值之间变化（如 Subsection II-C.2 中所述）。

从 Table I 可以看出，传统的 ML 框架无法达到最先进的（state-of-the-art, SoA）准确率：SVM 达到了 $7 . 2 8 ^ { \circ } ~ ( \mathrm { i . e . , ~ + 0 . 2 4 ^ { \circ } }$ 相比于 SoA），而 MLP 获得的 MAE 为 $7 . 1 4 ^ { \circ } ( \mathrm { i . e . , ~ + 0 . 1 0 ^ { \circ } }$ 相比于 SoA）。此外，SVM 还存在另外两个局限性：(i) 由于训练时间发散，试图通过增加训练集规模来提高 SVM 的准确率被证明是不可行的；(ii) 即使调整了调节每个 DoA（自由度）偏差-方差权衡的 $C ^ { \bullet } ,$，SVM 平均仍将 98.3% 的训练样本作为支持向量（16 维），占用 1.15 MB 内存，这对具有严格内存限制的嵌入式设备而言要求过高。需要注意的是，后一个问题是 SVM 固有的方法学局限，其模型大小无法在训练前预先固定。

TCN（时间卷积网络）是唯一被证明能够超越 SoA MAE 的模型。所报告的所有 TEMPONet 结果均指量化为 int8 格式的网络，与 fp32 相比，其内存占用减少了 4 倍。从 Figure 2 可以看出，实验中最小和最大的 TEMPONet 显示出更高的误差，表明它们分别产生了欠拟合和过拟合，从而确定了最佳的偏差-方差权衡处于介于中间的区间 [64, 96, 128] 内。当卷积块 Block III 具有 64、96 或 128 个通道时，TEMPONet 的回归精度相当。值得注意的是，即使在较低精度下运行，该模型也超越了基线。在 Table I 中，我们报告了我们的 TEMPONet-64 与 [12] 的 LSTM 以及本工作实现的 SVM 和 MLP 相比的 MAE 和回归准确率。

特别是，在 Block III 采用 64 通道时，我们的 TEMPONet 在模型大小和 MACs 曲线中出现拐点（elbow），从而代表了最佳的 MAE 与部署权衡。该 64 通道 TEMPONet 的内存占用仅为 70.9 kB，计算量为 3.16 MMACs，与文献 [7] 中提出的原始 TEMPONet 相比，内存减少了 6.5ˆ，计算量减少了 5.3ˆ。

在 Figure 3 中，我们展示了该网络给出的回归输出示例。特别是可以观察到，无论对于窄幅还是宽幅运动，输出都是迅速且准确的。典型误差主要分为两类。第一类误差是偏移量（offset），在每次运动期间保持平稳；由于输出也是平稳的，这种恒定差异预计不会影响用户感知到的准确度；并且即使被感知到，偏移量也可以通过特定会话的重新校准轻松补偿。第二类误差是快速不稳定片段，可以通过增强 EMA 后处理来进行平滑；如 II-D.3 小节所述，对 EMA 平滑的最佳程度进行了优化，以针对 MAE 调整平滑度与延迟之间的最佳权衡；获得的平均衰减因子为 $\alpha _ { \mathrm { E M A } } = 0 . 8 6 2$，跨受试者和 DoA 的标准差为 0.044。

最后，我们在商用微控制器 GAP8 [15] 上实现了 64 通道 TEMPONet，以测量每次推理的延迟和能耗开销。在部署方面，我们使用了开源工具 DORY（Deployment Oriented to memoRY，[27]），并对后端进行了扩展以支持空洞卷积。计算延迟和能耗是通过在 GAP8 上运行推理进行实验测量的。能耗 $E _ { \mathrm { e x p } }$ 是通过实验测量消耗的电流 $i _ { \mathrm { e x p } } ( t )$，然后在计算时间区间内进行积分确定的：

$$
E _ {\mathrm{exp}} = V _ {\mathrm{DD}} \int_ {0} ^ {T _ {\mathrm{exp}}} i _ {\mathrm{exp}} (t) \mathrm{d} t,\tag{5}
$$

其中 $V _ { \mathrm { D D } }$ 是供电电压，$[ 0 , T _ { \mathrm { e x p } } ]$ 是实验测得的推理所需延迟时间区间。

当运行在 $V _ { \mathrm { D D } } ~ = ~ 1 \mathrm { V } , ~ f _ { \mathrm { C L K } } ~ = ~ 1 0 0 \mathrm { M H z }$（能效最高的配置）时，GAP8 的功耗为 51.0 mW。这使得每次推理的延迟仅为 4.76 ms，单次推理能耗仅为 0.243 mJ。这些数值表明，所选的 64 通道 TEMPONet 能够满足资源受限控制器和实时操作的严格约束。关于延迟，假手控制实时性要求的共识为 300 ms [28]。考虑到 128 ms 的输入窗口长度加上 4.76 ms 的计算延迟，我们的应用以极大的裕度满足了实时性要求，证明了其有能力在没有明显感知延迟的情况下提供流畅的控制。

## IV. 结论

在这项工作中，我们提出了一种基于时间卷积网络（Temporal Convolutional Network）的 sEMG 手部运动学估计方案，在数值精度从 float32 降低到 int8 的情况下，实现了低于目标数据集当前最优（SoA）水平的回归误差。我们将模型部署在低功耗边缘微控制器上，展现出较低的内存占用、计算开销和单次推理能耗，从而证明了我们的解决方案适用于在实时工作的资源受限嵌入式控制器上实现。

## 参考文献

[1] R. Meattini, S. Benatti, U. Scarcia, L. Benini, and C. Melchiorri, “Experimental evaluation of a sEMG-based human-robot interface for human-like grasping tasks,” in 2015 IEEE International Conference on Robotics and Biomimetics (ROBIO). IEEE, 2015, pp. 1030–1035.

[2] T. Scott Saponas et al., “Making muscle-computer interfaces more practical,” in Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, 2010, pp. 851–854.

[3] B. Milosevic et al., “Exploring Arm Posture and Temporal Variability in Myoelectric Hand Gesture Recognition,” Int. Conf. on BioRob, 2018.

[4] Y. Hu, Y. Wong, W. Wei, Y. Du, M. Kankanhalli, and W. Geng, “A novel attention-based hybrid cnn-rnn architecture for semg-based gesture recognition,” PloS one, vol. 13, no. 10, p. e0206049, 2018.

[5] P. Tsinganos et al., “Improved gesture recognition based on semg signals and tcn,” in ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019, pp. 1169–1173.

[6] B. Milosevic, S. Benatti, and E. Farella, “Design challenges for wearable emg applications,” in Design, Automation & Test in Europe Conference & Exhibition (DATE), 2017. IEEE, 2017, pp. 1432–1437.

[7] M. Zanghieri et al., “Robust real-time embedded emg recognition framework using temporal convolutional networks on a multicore iot processor,” IEEE transactions on biomedical circuits and systems, vol. 14, no. 2, pp. 244–256, 2019.

[8] M. Zanghieri, S. Benatti, F. Conti, A. Burrello, and L. Benini, “基于时间卷积网络的表面肌电手部抓握识别时间变异性分析,” in 2020 2nd IEEE International Conference on Artificial Intelligence Circuits and Systems (AICAS). IEEE, 2020, pp. 228–232.

[9] J. L. Betthauser et al., “使用时间卷积网络在动作转换期间实现稳定的肌电序列预测,” in Int. Conf. NER. IEEE, 2019.

[10] J. Choi et al., “PACT：用于量化神经网络的参数化截断激活,” 2018.

[11] A. Krasoulis et al., “使用者练习对直观肌电解码器假手手指控制的影响,” Frontiers in neuroscience, vol. 13, p. 891, 2019.

[12] P. Koch et al., “基于循环神经网络从表面肌电数据回归手部动作,” in 2020 Int. Conf. EMBC, 2020.

[13] A. Krasoulis et al., “基于多输出、多类别分类的肌电手指动作解码：离线分析,” Scientific reports, vol. 10, no. 1, pp. 1–10, 2020.

[14] C. Castellini and P. Van Der Smagt, “先进手部假肢中的表面肌电,” Biological cybernetics, vol. 100, no. 1, pp. 35–47, 2009.

[15] https://greenwaves-technologies.com/gap8 gap9/.

[16] J. Cacioppo et al., “骨骼运动系统,” 1990.

[17] C. J. De Luca et al., “表面肌电在生物力学中的应用,” Journal of applied biomechanics, 1997.

[18] R. M. Rangayyan, 生物医学信号分析 (Biomed. signal analysis). John Wiley & Sons, 2015.

[19] http://ninaweb.hevs.ch/DB8.

[20] T. Bao, Y. Zhao, S. A. R. Zaidi, S. Xie, P. Yang, and Z. Zhang, “用于表面肌电手部运动学估计的深度卡尔曼滤波网络,” Pattern Recognition Letters, vol. 143, pp. 88–94, 2021.

[21] S. Bai et al., “用于序列建模的通用卷积与循环网络的实证评估,” 2018.

[22] C. Lea et al., “用于动作分割与检测的时间卷积网络,” in IEEE Conf. CVPR, 2017.

[23] S. Ioffe and C. Szegedy, “批量归一化：通过减少内部协变量偏移加速深度网络训练,” in International conference on machine learning. PMLR, 2015, pp. 448–456.

[24] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout：防止神经网络过拟合的一种简便方法,” The journal of machine learning research, vol. 15, no. 1, pp. 1929–1958, 2014.

[25] https://github.com/pulp-platform/nemo.

[26] F. Conti, “技术报告：用于部署模型的 NEMO 深度神经网络量化,” 2020.

[27] A. Burrello, A. Garofalo, N. Bruschi, G. Tagliavini, D. Rossi, and F. Conti, “DORY：在低成本物联网微控制器上自动端到端部署实际深度神经网络,” 2020.

[28] B. Hudgins, P. Parker, and R. N. Scott, “多功能肌电控制的一种新策略,” IEEE transactions on biomedical engineering, vol. 40, no. 1, pp. 82–94, 1993.