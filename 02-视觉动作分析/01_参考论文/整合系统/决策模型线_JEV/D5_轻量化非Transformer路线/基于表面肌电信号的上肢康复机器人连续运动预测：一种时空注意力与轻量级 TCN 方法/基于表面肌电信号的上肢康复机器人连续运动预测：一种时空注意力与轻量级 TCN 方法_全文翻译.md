# 基于表面肌电信号的上肢康复机器人连续运动预测：一种时空注意力与轻量级 TCN 方法

Weiyu Chen<sup>1</sup>, Pengcheng Li<sup>1\*</sup>, Shuxiang Guo<sup>1,2\*</sup>, Chunying Li<sup>1</sup>, Jun Leng<sup>1</sup>

<sup>1</sup>南方科技大学电子与电气工程系

中国 广东 深圳 518055

<sup>2</sup>北京理工大学生命学院、工信部融合医工系统与健康工程重点实验室，航天中心医院

中国 北京 100081

通讯作者：guo.shuxiang@sustech.edu.cn, lipc@sustech.edu.cn

摘要— 表面肌电信号（sEMG）在控制上肢康复外骨骼方面展现出巨大的应用前景。然而，sEMG 信号显著的个体间差异性对开发能够跨个体有效泛化的模型提出了挑战。为应对这一挑战并实现受试者无关的连续运动预测，本文提出了一种融合时空注意力机制与轻量级时间卷积网络（STA-TCN）的新方法。通过引入双通路（空间和时间）注意力机制，模型有选择地增强了与 sEMG 信号相关的空间（肌肉通道）和时间（运动阶段）特征，从而消除了针对特定受试者重新训练的需求。本研究设计了一种绳驱动上肢康复外骨骼（CURE），并将其用于连续运动预测。包含 6 名受试者的实验表明，STA-TCN 在连续预测肘关节角度方面优于传统模型，在跨用户测试中实现了更低的均方根误差（RMSE）（14.77<sup>◦</sup> 对比 15.25<sup>◦</sup>）以及接近当前最高结果的 R<sup>2</sup> 值（0.91 对比 0.93）。该方法每次预测的延迟仅为 1.5 ms，满足实时处理要求，非常适合部署在实时康复机器人系统中。所提出的方法可为个性化和自适应机器人康复系统提供有效的解决方案，在不依赖个体校准的情况下提高了运动预测精度。

索引词— 连续运动预测，时空注意力，时间卷积网络（TCN），上肢康复外骨骼，表面肌电信号（sEMG）

## I. 引言

脑卒中是全球范围内导致残疾的主要原因之一，通常会导致上肢偏瘫，严重影响患者的日常生活活动和生活质量 [1]。早期强化康复对于恢复至关重要 [2]，但传统疗法需要长期且有专业人员监督的训练，给患者、家庭和治疗师带来了经济与身体上的双重负担 [3]。这凸显了对创新康复方法的需求，即在减少对持续专业监督依赖的同时，提供有效且个性化的治疗。

诸如外骨骼等机器人康复设备已成为应对这些挑战极具前景的解决方案，因为它们有助于开展重复性、精准且可量化评估的康复训练 [4]。患者与这些康复系统之间的有效交互，关键取决于能否准确解码神经意图，以将生理信号实时转化为可执行的机器人控制指令。表面肌电信号（sEMG）因其采集简便且具有较高的时间分辨率，已成为外骨骼设备中广泛使用的一种生理信号，促进了用户与机器人系统之间的有效控制与交互。

神经信号解码方法已经取得了显著进展，近期的研究进展解决了在准确性和泛化性方面的关键挑战 [5]。早期的研究通常依赖生物力学建模或传统的机器学习技术，而这些方法由于复杂的参数估计以及对个体差异的敏感性而面临局限性 [6]。近年来，基于深度学习的方法在该领域取得了重大进展，为建模表面肌电信号（sEMG）数据中固有的复杂、高维和时序关系提供了有效工具，从而提高了运动意图解码的准确性和鲁棒性 [7] [8] [9] [10]。Bao 等人 [11] 提出了一种 CNN-LSTM 混合模型，该模型首先使用 CNN 从 sEMG 信号中提取深层空间特征，随后通过基于 LSTM 的序列回归来捕获长期时序依赖关系，与单独的 CNN 或 LSTM 模型相比，显著提高了手腕运动学估计的准确性。Zanghieri 等人 [12] 提出了一种用于基于 sEMG 的手部运动学回归的轻量级时间卷积网络（Temporal Convolutional Network, TCN），在保持极小内存占用的同时实现了高精度，在运动估计方面优于循环模型。Zabihi 等人 [13] 提出了 TraHGR 框架，这是一种基于 Transformer 的混合架构，用于利用 sEMG 信号进行手势识别。通过整合并行的时序和特征 Transformer 网络，他们的方法相比现有的深度学习模型提升了分类准确率，并增强了对信号变异性的鲁棒性。Zhang 等人 [14] 将多通道 EMG 信号特征与卡尔曼滤波器相结合，用于估计连续的前臂旋前-旋后运动。他们的混合方法在实时外骨骼控制中实现了高精度，展示了将深度学习与传统滤波技术相结合的潜力。Wu 等人 [15] 使用强化学习结合 sEMG 来估计关节力矩，提高了准确性并减少了对大量训练数据的依赖。

尽管取得了这些多样化的进展，但由于神经肌肉模式的显著个体间差异、解剖学差异以及电极放置位置的变化，基于深度学习（deep learning, DL）的方法在被试间泛化和实时自适应性方面仍存在局限性 [7]。研究人员已经探索了多种策略来提升基于 DL 方法的泛化能力。Li 等人 [8] 提出了一种基于两阶段遗传算法（genetic algorithm, GA）的特征选择（TS-GAFS）方法，该方法应用最小冗余最大相关（minimum redundancy maximum relevance, mRMR）准则来选择不变特征，提高了基于 sEMG 的运动估计的独立性。然而，该方法仍然依赖预定义的特征集，限制了其对个体间肌肉激活动态变化的适应能力。Long 等人 [16] 引入了迁移学习策略，包括利用新被试数据对预训练模型进行微调，这虽然增强了泛化能力，但需要来自每个新用户的带标签数据，从而限制了实际应用性。Li 等人 [9] 应用了一种多源域适应（multi-source domain adaptation, MDA）方法，通过学习来自多个用户的不变特征来减少域偏移（domain shift），从而提高了泛化能力。然而，该方法需要大量的计算资源，导致延迟过大，制约了实时预测。

综上所述，现有的 sEMG 解码研究面临以下挑战：

1) 实时效率：许多模型需要高计算资源，限制了便携性和即时应用性。

2) 跨被试变异性：sEMG 信号的高变异性需要针对特定用户进行校准，限制了对新用户的泛化能力。

本文针对上肢康复中基于 sEMG 的连续运动预测中被试间泛化的挑战，提出了一种不依赖目标用户数据且能满足实时应用要求的新型时空注意力机制。受视频字幕生成 [17] 和位置推荐任务 [18] 近期成功的启发，该注意力机制独立计算并整合时空注意力，在空间上（跨肌肉通道）和时间上（跨时间片段）选择性地突出相关的信号特征。所提出的方法无需对每个个体进行预先校准或微调，即可从 sEMG 信号中提取用户不变特征。模型中集成的轻量级 TCN（时间卷积网络）架构确保了计算效率，在保持高解码精度的同时实现了无缝的实时处理。

本文的主要贡献如下：

1) 提出了一种专为基于 sEMG 的连续运动预测设计的时空注意力机制，以提高跨被试泛化能力。

2) 开发并将该注意力机制与轻量级 TCN 模型相集成，在无需针对特定受试者重新训练的情况下，跨多用户实现了准确、实时的连续角度估计。

本文其余部分的结构安排如下：第 II 节详细介绍相关方法，包括系统设置与所提出的模型架构；第 III 节展示实验结果；第 IV 节讨论研究发现及其意义；第 V 节对所提出的 STA-TCN 方法进行总结。

## II. 提出的方法

## A. 上肢康复外骨骼概述

表面肌电（sEMG）信号已广泛应用于实时关节角度估计，有助于实现上肢康复设备的直观控制。在本文中，所提出的轨迹规划与康复策略通过如图 1 所示的名叫 CURE 的上肢机器人进行验证。

![](images/81d732e4cb703241e65a5c689cdc7b7954b589e657d6ce3bbc3d172048eb8e07.jpg)  
Fig. 1 上肢外骨骼设备。

CURE 被设计为一种便携式居家康复系统，能够辅助三个自由度（DoF）：肘部屈曲/伸展、腕部屈曲/伸展以及腕部旋后/旋前，同时还兼容肩关节处的三个被动自由度。该系统采用绳驱动机构以利用其轻量化和柔顺性，允许电机远离外骨骼的运动部件布置，从而最大限度地减轻用户手臂的直接重量负担。四个紧凑型电机固定在支撑背板上，用户可通过腰带和肩带舒适地穿戴。具体而言，一个电机驱动肘部屈曲/伸展，而三个伺服电机协助腕部运动。

为增强基于用户意图的直观机器人辅助，我们开发了一个实时 sEMG 控制系统，包含四个关键组件：(1) 使用嵌入式 Myo 设备滤波器（巴特沃斯高通、陷波和归一化）结合 RMS 滑动窗口滤波的信号预处理；(2) 带有空时注意力机制的轻量级时间卷积网络（Lite Temporal Convolutional Network）；(3) 将 256 通道特征映射到运动输出的回归层；(4) 通过 MAE、RMSE 和 $R^2$ 指标进行的定量评估。该工作流程按顺序从原始 sEMG 采集推进到实时性能验证。

## B. 实验方案

1) 参与者：本研究共招募了六名健康成年人——三名男性和三名女性，年龄在 22 至 25 岁之间。所有人均无神经肌肉疾病史报告。每位志愿者分别标记为 A 至 F。所有参与者均听取了实验流程的说明并签署了知情同意书。

2) 实验设置：数据采集平台如图 2 所示，采用了一个配备有八个均匀分布的 sEMG（表面肌电）传感器的 Myo 臂环（Thalmic Labs Inc.）。图 2(b) 展示了臂环上这些电极的放置位置。Myo 设备通过低功耗蓝牙（BLE）以无线方式将 sEMG 信号传输到计算机，并使用 Myo SDK 以 200 Hz 的频率实时采样数据。将惯性测量单元（IMU）（JY901，WIT Motion，中国深圳；如图 2(c) 所示）固定在前臂上，以 20 Hz 的采样频率记录肘部运动，用于验证基于 sEMG 的预测。

![](images/2161e01b9d4bfe332b134e4b1b765e9e691671be2b948c3e5ba2bf3fe1377a24.jpg)

![](images/bbf0e4270b71461563461c94706fb42d48ba51c86f840cc413589ab81dfcc9cb.jpg)

(b) 电极布置。  
(a) 实验数据采集原理图。  
![](images/01d0d16c52ac4c27845480905587b14f99d9f39d3b3c6946f42117ce7208e49e.jpg)  
(c) IMU 传感器。  
Fig. 2 (a) 实验数据采集原理图。(b) 电极布置。(c) IMU 传感器。

3) Myo 臂环的通道选择：根据文献 [19] 中的建议，三个通道被认为是最优的。因此，选择了通道 4、7 和 1，分别对应于肱二头肌、肱三头肌（长头）和肱三头肌（外侧头）。

4) 数据采集：受试者进行了同步动作，并通过触觉反馈和常亮的 LED 指示灯进行确认，以确保电极与肌肉之间的稳定接触。每次试验持续一分钟，包括静息、肘部屈伸以及恢复静息，重复五次，并在两次试验之间间隔两分钟的休息时间以防止疲劳。

5) sEMG 信号预处理：原始 sEMG 信号微弱且易受噪声干扰，需要进行预处理以实现可靠的分析。Myo 臂环内置了 50 Hz 陷波滤波器以消除工频干扰，且 Myo SDK 提供了在 $[−1, 1]$ 范围内的预归一化信号。应用了截止频率为 20 Hz 的四阶巴特沃斯（Butterworth）高通滤波器来去除直流偏移和低频噪声。针对 sEMG 的非平稳特性，采用滑动窗口均方根（RMS）滤波器提取肌肉激活强度。对于每个选定的通道（1、4、7），信号在一个 100 样本（0.5 s）的窗口内进行平滑处理并开方，从而获得用于后续分析的 RMS 包络。

## C. CURE 的建模与角度估计

为了解决基于 sEMG 的关节角度估计中的跨受试者变异性，我们提出了一种将双重注意力通路与时序卷积操作相结合的分层架构（Fig. 3）。主要创新点包括：

• 解耦的时空注意力通路，用于捕获空间和时间依赖性

• 针对实时处理进行优化的轻量级时序卷积骨干网络。

1) 时空注意力机制：

a) 空间注意力通路（Spatial Attention Pathway）：空间注意力通路通过可学习的通道交互对神经肌肉激活模式进行建模。空间注意力权重 $\alpha _ { s } \in \mathbb { R } ^ { C }$ 通过多尺度时域卷积以及随后的通道瓶颈投影计算得出：

$$
\alpha_ {s} = \operatorname{Sigmoid} \left(\mathcal {P} _ {b} \left(\operatorname{Conv1D} _ {1 5} (X)\right)\right),\tag{1}
$$

其中 Conv $\mathrm { . D _ { 1 5 } }$ 表示带填充的 15 抽头时域卷积，$\mathcal { P } _ { b } : \mathbb { R } ^ { 6 4 \times T }  \mathbb { R } ^ { \bar { C } \times T }$ 表示通过 $1 \times 1$ 卷积实现的瓶颈投影。该设计捕获了通道间的肌肉协同作用。

b) 时间注意力通路（Temporal Attention Pathway）：时间注意力通路采用相对位置编码 $P _ { t } \in \mathbb { R } ^ { T \times T }$ 来对肢体运动中的相位连续性进行建模。注意力计算公式为：

$$
\mathbf {A} _ {t} = \mathrm{Softmax} \Bigl (\frac {Q _ {t} K _ {t} ^ {\top}}{\sqrt {d _ {k}}} + P _ {t} \Bigr) V _ {t},\tag{2}
$$

其中查询 $Q _ { t } ,$、键 $K _ { t } ,$ 和值 $V _ { t }$ 衍生自经层归一化的输入特征。位置编码矩阵 $P _ { t }$ 通过梯度下降进行学习，从而能够在没有显式运动学约束的情况下自动发现运动相位关系。

时间特征通过值聚合获得：

$$
X _ {t} = \mathbf {A} _ {t} V.\tag{3}
$$

![](images/e47054e7a36b4d7ff4ffd48ec62c0e3500b3dd01b7d08687596f3cf33ebcdb21.jpg)  
Fig. 3 具有分层时域卷积的双通路时空注意力网络。

c) 自适应特征融合（Adaptive Feature Fusion）：空间和时间注意力特征通过带有残差连接的参数化融合相结合：

$$
X _ {\mathrm{fusion}} = \gamma \cdot (\alpha_ {s} \odot X _ {s}) + (1 - \gamma) \cdot X _ {t} + \mathcal {P} _ {r} (X)\tag{4}
$$

其中动态系数 $\gamma \in [ 0 , 1 ]$ 自动平衡空间肌肉协同特征与时间相位特征。

2) 轻量级时间网络（Lightweight Temporal Network）：

a) 具有几何膨胀的深度可分离卷积：

$$
\operatorname{DSConv} (k = 1 5, d = 2 ^ {l - 1}), \quad \mathrm{RF} = 1 + \sum_ {i = 1} ^ {L} 1 4 \cdot 2 ^ {i - 1}\tag{5}
$$

在卷积核大小 $k = 1 5$ 且膨胀率 $d = [ 2 ^ { 0 } , 2 ^ { 1 } , 2 ^ { 2 } ]$ 的情况下，这实现了 425 ms 的时间窗口（在 200 Hz 采样率下），从而能够对完整的运动周期进行有效建模。

b) 瓶颈架构（Bottleneck Architecture）：通道扩展 32 → $6 4  1 2 8  2 5 6$ 在保持特征抽象的同时防止过拟合。

3) 实现细节：该模型在 PyTorch 中实现，采用混合精度训练（FP16/FP32），并使用 AdamW 优化器 $( \beta _ { 1 } ~ = ~ 0 . 9 , ~ \beta _ { 2 } ~ = ~ 0 . 9 9 9 )$ 和余弦退火学习率调度策略 $( T _ { m a x } = 1 0 0 )$ 进行优化。时间卷积块采用 dropout $( p = 0 . 1 )$ 和批归一化（batch normalization）进行正则化。

4) 评估标准：对于定量评估，所选的标准包括均方根误差（RMSE）、平均绝对误差（MAE）以及决定系数（R<sup>2</sup>）。

## III. 实验结果

本节通过离线实验展示对所提出的 STA-TCN 模型性能的评估。以基线 TCN 模型作为主要基准，并与其他现有方法进行对比，以进行全面的性能分析。所有实验均在运行 Ubuntu 系统且配备 NVIDIA GeForce RTX 4090 GPU 的 Linux 服务器上进行。模型训练和推理使用 Python 基于 PyTorch 框架实现，而来自 Myo 臂环的原始 sEMG 数据通过低功耗蓝牙（BLE）传输，以便与外骨骼控制系统集成。

## A. 与基线模型的对比性能

我们对所提出的 STA-TCN 方法与基线 TCN 模型进行了直接对比，以评估从 sEMG 信号估计关节角度的预测性能。如表 I 所示，STA-TCN 取得了显著更低的平均绝对误差（MAE）和均方根误差（RMSE），同时获得了更高的决定系数 $( R ^ { 2 } )$ ，证明了其在降低误差和解释方差两方面的优越性能。

TABLE I  
所提出的 STA-TCN 与基线 TCN 之间的性能对比

<table><tr><td>方法</td><td>MAE</td><td>RMSE</td><td> $R^2$ </td></tr><tr><td>STA-TCN</td><td>11.53°</td><td>14.77°</td><td>0.91</td></tr><tr><td>TCN</td><td>17.04°</td><td>21.27°</td><td>0.89</td></tr></table>

图 4 进一步说明了对比结果及其潜在机制。列 (a, d, g)、(b, e, h) 和 (c, f, i) 分别对应来自三名不同受试者的数据。子图 (a)–(c) 展示了 STA-TCN 和 TCN 在代表性运动序列下的预测角度和真实角度。子图 (d)–(f) 量化了各方法之间的误差差异。子图 (g)–(i) 表示由 STA 模块生成的注意力权重热力图，突显了该网络如何在有效淡化受试者个体差异的同时强调一致的运动模式。

## B. 与现有研究的对比

除了基线评估外，我们还与以往基于 sEMG 驱动的关节角度估计研究进行了进一步对比，展示了本方法如何超越现有经典方法（表 II）。所提出的模型表现出更低的 RMSE 和更优的 $\bar { R } ^ { 2 }$ 值，突显了其在多通道 sEMG 解码方面的有效性。

值得注意的是，我们的模型取得了 14.77<sup>◦</sup> 的 RMSE，在预测精度和稳定性方面均优于多种对比方法。此外，0.91 的 $R ^ { 2 }$ 值表明估计角度与真实角度之间存在强相关性，突显了我们的模型性能在不同受试者间的鲁棒性。

![](images/451207842ea74abaec3ed3e888f82ea30dead8bdb8fb7bce52795152bec8766f.jpg)  
(a)

![](images/70769d8c2cb8b1dcb7dc5a52c31eaca5ff679aec38e15b8eb639978886071524.jpg)

![](images/0552c98c9e1ba0a96995026b6a8affefa9dc358df116b83ff944106ed8b640f1.jpg)

![](images/18e06f1d2c4c18c5415242411f64cbb43ae852bcd7d5d7c9ca9dfbe60e9fe5d9.jpg)  
(b)  
(c)

(d)  
![](images/cf3cd9c5e3e5dd64cd8cdffa2981ce9c11b63b83b5511b190faba7bdda807be8.jpg)

![](images/c2885ac61688556fe3a721baf4986c8734b95066158c0c0030c78eb1a53dab74.jpg)

![](images/d6c9ed92655b932ee9d99e976e701085f21efc5e2ba8e03653fe5fa506a0bcf1.jpg)

(g)  
(e)  
(f)  
![](images/7eb1910bde4e257451fcec02ce20e8cf508582332393b103d4c42c152f30806c.jpg)  
(h)

![](images/7a8e86df39b095945e15261b1b09422a9cf0e1501336f208a166fe40d9788a5e.jpg)  
(i)  
Fig. 4 来自三名受试者的结果对比与注意力模式。子图 (a)–(c)：STA-TCN 与 TCN 预测的关节角度与真实数据的对比。子图 (d)–(f)：两种方法的误差分布。子图 (g)–(i)：STA 注意力热图。

TABLE II  
与其他方法的对比结果

<table><tr><td>方法</td><td>RMSE</td><td> $R^2$ </td></tr><tr><td>Yang [20]</td><td>20.44°</td><td>0.89</td></tr><tr><td>Li [8]</td><td>17.40°</td><td>0.90</td></tr><tr><td>Li [10]</td><td>15.25°</td><td>0.93</td></tr><tr><td>Zhao [21]</td><td>17.59°</td><td>0.91</td></tr><tr><td>本文方法</td><td>14.77°</td><td>0.91</td></tr></table>

## IV. 讨论

本研究提出了一种结合轻量级时间卷积网络（STA-TCN）的新型时空注意力机制，用于基于表面肌电（sEMG）信号进行实时的跨受试者关节角度估计。实验结果表明，所提出的方法在 RMSE 和 $R ^ { 2 }$ 方面均优于基线时间卷积网络（TCN）以及现有的多种方法

STA-TCN 模型与基线 TCN 模型之间的直接对比表明，关节角度估计精度得到了显著提升。如 Table I 所示，STA-TCN 将平均绝对误差（MAE）从 $1 7 . 0 4 ^ { \circ }$ 降低至 11.53<sup>◦</sup>，并将均方根误差（RMSE）从 21.27<sup>◦</sup> 降低至 14.77<sup>◦</sup>。此外，STA-TCN 的 $R ^ { 2 }$ 值为 0.91，优于基线 TCN 的 $R ^ { 2 }$ 值（0.89）。这些结果突出了所提出的时空注意力机制在捕获 sEMG（表面肌电）信号内部复杂、高维关系方面的优越性能。该注意力机制有助于模型在空间维度（跨肌肉通道）和时间维度（跨时间片段）上选择性地聚焦于相关特征，从而提高了预测关节角度的精度。

除了若干预测精度指标外，该方法在泛化性和实时性能方面也优于现有的部分方法。Yang 等人的方法需要单独训练，限制了其在不同用户之间的泛化能力。Li 等人 [10] 没有明确解决跨不同受试者泛化的挑战，因为其仅在小规模且固定的参与者群体上进行了评估。相比之下，所提出的 STA-TCN 方法在多个受试者之间表现出强大的泛化能力，我们采用的留一法实验对此进行了验证，该实验涵盖了六名不同的参与者，且无需针对每个个体进行特定的重新训练。Li 等人 [9] 引入了一种多源域自适应（MDA）方法来增强跨受试者的泛化能力。尽管他们的方法展现出良好的前景，但由于计算复杂度较高，导致严重的延迟，无法满足 HRI（人机交互）的要求。这一局限性使得他们的方法不适用于对低延迟有严格要求的实时康复应用。相比之下，我们的方法在 NVIDIA GeForce RTX 4090 GPU 上运行时，对包含 1500 个样本的输入序列进行每次 IMU 角度预测的延迟仅为 1.5 ms，实现了实时性能。这种低延迟处理满足了假肢控制的实时性要求，使我们的方法非常适合实际的康复系统。

尽管取得了良好的结果，但所提出的方法目前仍存在一些局限性。该模型在 GPU 上进行了验证和运行，尚未在便携式硬件（例如嵌入式系统或边缘设备）上实现。将模型部署在轻量级、便携式设备上将使其更适用于居家康复应用。此外，该方法仅在涉及相对较少肌肉的肘关节简单周期性运动中得到了验证。在涉及具有更多且冗余肌肉的其他关节的情况下，或者当运动变得随机或结构性较弱时，该方法是否能取得相当的性能仍不明确。探索该模型在这些更复杂场景中的有效性是未来研究的一个重要方向。

## V. 结论

本文提出了时空注意力与轻量级时间卷积网络（spatial temporal attention and light-weight temporal convolutional network, STA-TCN）模型，以改善上肢康复系统中的关节角度估计。所提出的方法有效解决了受试者间个体差异和实时估计的挑战，在预测精度以及跨多个受试者的泛化能力方面均展现出优越的性能。通过利用时空注意力机制，该模型能够选择性地增强 sEMG 信号中的相关特征，从而实现更精确的运动估计，且无需针对特定受试者进行重新训练。具体而言，STA-TCN 模型实现了 14.77<sup>◦</sup> 的 RMSE，优于仅使用 TCN 模型的 21.27<sup>◦</sup> RMSE。轻量级 TCN 架构的集成确保了计算效率，每次角度预测仅需 1.5 ms 延迟，使得该模型非常适合机器人康复中的实时应用。未来的工作将侧重于其在便携式嵌入式设备上的部署，以及研究该模型在更复杂关节和较弱结构化运动中的性能表现。

## 致谢

本研究得到了中国深圳南方科技大学高水平专项资金（项目编号 G03034K003）的部分资助，以及中国深圳“医工融合”专项支持计划项目（项目编号 F-2024-Z99-502365）的部分资助。

## 参考文献

[1] V. L. Feigin, M. Brainin, B. Norrving et al., ”World Stroke Organization (WSO): Global Stroke Fact Sheet 2022,” Int. J. Stroke, vol. 17, no. 1, pp. 18-29, 2022.

[2] J. Shahid, A. Kashif and M. K. Shahid, ”A Comprehensive Review of Physical Therapy Interventions for Stroke Rehabilitation: Impairment-Based Approaches and Functional Goals,” Brain Sci., vol. 13, no. 5, p. 717, 2023.

[3] R. Raghavan, E.E. Fitzsimmons-Craft, R.R. Welch, et al., “Costeffectiveness of train-the-trainer versus expert consultation training models for implementing interpersonal psychotherapy in college mental health settings: evidence from a national cluster randomized trial,” Implement. Sci., vol. 19, Art. no. 55, 2024.

[4] L. Song, C. Ju, H. Cui 等，“上肢外骨骼机器人控制策略技术研究：综述”，Machines，卷 13，期 3，论文编号 3，2025年3月。

[5] S. P. Sitole 和 F. C. Sup，“基于肌电信号的人体关节力学连续预测：基于模型和无模型方法的综述”，IEEE Trans. Med. Robot. Bionics，卷 5，页码 528-546，2023。

[6] Z. Wei, Z.Q. Zhang 和 S.Q. Xie，“利用表面肌电（sEMG）进行上肢康复的连续运动意图预测：基于模型和无模型方法的系统综述”，IEEE Trans. Neural Syst. Rehabil. Eng.，卷 32，页码 1487-1504，2024。

[7] H. Li, S. Guo, D. Bu 和 H. Wang，“使用BP神经网络（BPNN）评估基于表面肌电（sEMG）的连续肘关节角度的主体间变异性”，收录于 Proc. IEEE Int. Conf. Mechatron. Autom. (ICMA)，中国广西，页码 388-393，2022。

[8] H. Li, S. Guo, D. Bu 和 H. Wang，“一种基于遗传算法（GA）的两阶段表面肌电（sEMG）特征选择方法，用于独立于用户的肘关节角度连续估计”，IEEE Trans. Instrum. Meas.，卷 72，页码 1-9，2023。

[9] H. Li, S. Guo, H. Wang 和 D. Bu，“结合多源域适应和BP神经网络的独立于受试者的基于表面肌电（sEMG）关节角度连续估计”，IEEE Trans. Instrum. Meas.，卷 72，页码 1-10，2023。

[10] H. Li, S. Guo, D. Bu, H. Wang 和 M. Kawanishi，“用于家庭上肢康复系统的基于CNN-LSTM的独立于受试者连续运动估计”，IEEE Robot. Autom. Lett.，卷 8，页码 6403-6410，2023。

[11] T. Bao, S.A.R. Zaidi, S. Xie, P. Yang 和 Z.-Q. Zhang，“利用表面肌电估计腕部运动学的CNN-LSTM混合模型”，IEEE Trans. Instrum. Meas.，卷 70，论文编号 2503809，页码 1-9，2021。

[12] M. Zanghieri 等，“在低功耗边缘微控制器上利用时间卷积网络实现基于表面肌电（sEMG）的手部运动学回归”，收录于 Proc. IEEE Int. Conf. Omni-Layer Intell. Syst. (COINS)，西班牙巴塞罗那，页码 1-6，2021。

[13] S. Zabihi, E. Rahimian, A. Asif 和 A. Mohammadi，“TraHGR：基于肌电图的手势识别Transformer”，IEEE Trans. Neural Syst. Rehabil. Eng.，卷 31，页码 4211-4224，2023。

[14] L. Zhang, J. Long, R.G. Zhao, H. Cao 和 K. Zhang，“利用多通道肌电信号特征与卡尔曼滤波器估计连续旋前-旋后运动：在外骨骼控制中的应用”，Front. Bioeng. Biotechnol.，卷 9，论文编号 771255，2022。

[15] W. Wu, K.R. Saul 和 H. Huang，“利用强化学习从肌电图或关节运动学估计人体关节力矩：基于肌肉骨骼生物力学的一种替代方案”，J. Biomech. Eng.，卷 143，论文编号 044502，2021。

[16] Y. Long, Y. Geng, C. Dai 和 G. Li，“一种基于迁移学习的跨受试者通用模型，用于新用户手指关节角度的连续估计”，IEEE J. Biomed. Health Inform.，卷 27，页码 1914-1925，2023。

[17] C. Yan et al., “STAT: Spatial-Temporal Attention Mechanism for Video Captioning,” IEEE Trans. Multimedia, vol. 22, no. 1, pp. 229-241, 2020.

[18] Y. Luo, Q. Liu, and Z. Liu, ”STAN: Spatio-Temporal Attention Network for Next Location Recommendation,” in Proc. Web Conf. 2021, Ljubljana Slovenia, pp. 2177-2185, 2021.

[19] S. Y. Shirazi, J. Welzel, S. Jeung and L. Godbersen, “A standardized framework for sensor placement in human motion capture and wearable applications,” arXiv preprint arXiv:2412.21159, 2024.

[20] Z. Yang, S. Guo, Y. Liu, H. Hirata and T. Tamiya, “An intention-based online bilateral training system for upper limb motor rehabilitation,” Microsyst. Technol., vol. 27, pp. 211-222, 2021.

[21] Y. Zhao, Z. Zhang, Z. Li, Z. Yang, A. A. Dehghani-Sanij and S. Xie, “An EMG-driven musculoskeletal model for estimating continuous wrist motion,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 28, pp. 3113-3120, 2020.