# 使用基于单个腰戴式可穿戴传感器的混合CNN-RNN深度学习模型进行步态事件预测

Muhammad Zeeshan Arshad, Ankhzaya Jamsrandorj, Jinwook Kim, and Kyung-Ryoul Mun

摘要—— 老年步态蕴含着丰富的身体和精神健康状况信息。作为下肢多个传感器的替代方案，将单个传感器佩戴在骨盆（腰部）位置具有位置优势，并且可获取大量信息。本研究旨在探索一种通过使用腰部单个传感器和深度学习模型来提高老年人步态事件检测准确性的方法。数据采集于配备三个IMU（惯性测量单元）传感器步行时的老年受试者。输入仅采用腰部传感器的数据，用于训练16个深度学习模型，包括CNN（卷积神经网络）、RNN（循环神经网络）以及带有或不带有双向机制和注意力机制的CNN-RNN混合模型。真值标签从足部IMU传感器中提取。在±6TS（±6毫秒）和±1TS（±1毫秒）的容差窗口下，CNN-BiGRU-Att模型分别实现了99.73%和93.89%的较高准确率。相比以往探索步态事件检测的研究，该模型在预测误差方面有了显著改善，在±1TS容差窗口下，HS（脚跟着地）和TO（脚尖离地）事件的MAE（平均绝对误差）分别为6.239毫秒和5.24毫秒。结果表明，使用结合注意力机制和双向机制的CNN-RNN混合模型，对于利用单个腰部传感器进行精确步态事件检测具有广阔的应用前景。本研究有助于减轻步态检测的负担，并提高其在未来可穿戴设备中的适用性，从而用于远程健康监测（RHM）或基于此的健康诊断。

关键词—— 步态分析、可穿戴设备、深度学习

## I. 引言

老年人步态中所蕴含的信息有助于评估他们所面临的健康风险[1]。除了身体状况外，即使是认知障碍和痴呆等精神状况，也可以在他们的步态中被发现[2, 3, 4, 5]。对老年人步态进行精确测量，使我们能够在早期阶段预测和检测其健康危机，并建立积极的策略以防止不必要的疾病进展。

测量人体步态这一具有挑战性的工作，从传统的视觉观察[6]发展到了目前使用阈值、峰值检测、手工特征和基于规则的方法[7, 8, 9]。IMU（惯性测量单元）等更小、更轻、更廉价传感器的出现，使得我们能够摆脱使用运动捕捉系统和测力台的繁琐且昂贵的测量方式，这些方式曾局限于严格的临床环境中[10, 11]。基于机器学习的方法，如隐马尔可夫模型（HMM）[12, 13, 14]、支持向量机（SVM）[15]、深度CNN[16]和循环神经网络（RNN）[17]的出现，进一步减轻了步态测量的负担，并开启了准确步态评估的新视野。

在测量步态时，可通过时间特征对其进行量化，其中精确检测每只脚的足跟着地（HS）和足尖离地（TO）至关重要。根据检测到的 HS 和 TO，可计算出步态相位。尽管现有方法在检测方面已取得了相当不错的性能，但使用多种传感器以及复杂的特征提取预处理过程被认为是主要的障碍。此外，传感器大多放置在下肢部位，会干扰自然行走，并限制其在日常生活中的应用。

作为下肢多个传感器的替代方案，可在骨盆上放置单一传感器，因为其具有位置优势且可获取丰富的信息。骨盆可能是可穿戴传感器的理想放置位置，因为它通常是人们系腰带的部位，且对日常活动的影响最小。同时，骨盆是宝贵的信息来源，因为单个传感器即可检测左右脚，并且它与步态六个决定因素中的三个相关联，即骨盆旋转、骨盆倾斜和骨盆的横向位移 [18]。它还与身体质心（COM）处的垂直中线对齐，本质上连接下肢与上身，从而能够在两者之间传递力并控制全身平衡。利用骨盆的信号，可以识别活动，甚至可以估计坐姿和躺姿等姿势。骨盆的信号富含日常活动模式的信息，并携有关于步态和运动的综合信息。然而，由于使用骨盆信号的准确性不及使用下肢部位信号的准确性，目前很少尝试使用骨盆信号 [19, 20]。

因此，本研究旨在探索一种利用骨盆上的单一传感器和深度学习模型来提高老年人步态事件检测准确性的方法。研究招募了健康或患有健康问题的老年人，并对他们的步态进行了测量。多种深度学习模型学习了骨盆信号中存在的步态事件信息之间的相互关系，并预测步态事件。然后，将预测结果与脚上传感器的真值进行比较。本研究提出了一种可靠的使用骨盆单一传感器进行步态检测的方法，有望减轻步态检测的负担，并提高其在未来可用于远程健康监测（RHM）或基于此进行诊断的可穿戴设备中的适用性。

![](images/f83a0c849d4b2e47cb7a4cd646d24a742a931703b0b959628b1a4152d7ab3d23.jpg)  
图 1：所提出的相位检测方法及用于数据采集的实验概述

## II. 方法

## A. 数据采集

本研究招募了 169 名年龄介于 60 至 80 岁之间的社区居住老年人。本研究获得了庆熙大学医学中心机构审查委员会的批准（IRB 编号：2017-04-001）。所有参与者在参与研究之前均获得了书面知情同意。根据参与时的健康状况，受试者被分为健康组和患者组。患者组包括衰弱受试者（n=47）、认知障碍受试者（n=8）、有跌倒史受试者（n=11）以及上述情况组合的受试者（n=9）。衰弱的定义是 5 项 FRAIL 量表（FRAIL scale）测试得分大于等于 1 [21]，认知障碍的定义是简易精神状态检查量表（mini-mental state examination, MMSE）[22] 得分低于 24 分。在问卷中回答在过去一年内有过跌倒史并因此接受过医院治疗者被纳入患者组。根据这些标准，75 名受试者被纳入患者组，其余 94 名被纳入健康组。所有受试者在数据采集时均能够在无他人帮助或辅助设备的情况下行走。表 I 总结了受试者的人口统计学详细信息。

本研究使用了三个商用 IMU 惯性测量单元传感器：一个固定于骨盆处，两个固定于足部（Xsens MVN，荷兰恩斯赫德）。IMU 传感器通过腰带固定于受试者腰部，另外两个传感器分别绑于左右足部，如图 1 所示。在佩戴传感器后，要求受试者以自选速度和比平时更快的速度各行走 10 米路径三次，因此每位受试者总共行走路径六次。三轴平移和三轴旋转惯性数据以 100 Hz 的采样率进行采集，并通过 0.5–6 Hz 的带通滤波器以去除高频噪声。

图 2(a) 和 (b) 显示了一名健康受试者骨盆部位的加速度和角速度信号。为了检测实际的 HS（heel-strike，足跟着地）和 TO（toe-off，脚尖离地）事件，使用了矢状面内左右足屈伸的角速度信号（图 2(c)）。脚尖离地事件被检测为反向高幅峰值，在图中以方形标记；足跟着地事件被检测为反向低幅峰值之前的过零点，在同一图中以三角形标记 [33]。利用每只脚的四个 HS 和 TO 事件，生成了真实标签数据（图 2(d)）。步态周期中足部接触地面的阶段（HS 到 TO）称为站立相，而足部离开地面的阶段（TO 到 HS）称为摆动相 [11]。图 2(d) 显示了右足相位信号（以红色虚线表示）和左足相位信号（以黑色实线表示）。站立相赋值为 1，摆动相赋值为 -1。

Fig. 3 展示了如何为单步超前预测（one-step-ahead prediction）训练生成输入-输出数据对，其中输入 x 指滑动窗口中的骨盆惯性测量单元（IMU）数据，输出 y 指滑动窗口下一时间步的右脚和左脚相位信号值。第一对数据以输入 $x _ { 1 } , x _ { 2 } , . . . , x _ { w }$ 开始，其中 $w$ 为窗口长度，对应的输出为 $y _ { w + 1 }$。然后将窗口向后移动一个时间步。因此，对于时间步 $t$ 的每一对数据，输入为 $x _ { t - w } , x _ { t - w + 1 } , . . . , X _ { t - 1 }$，输出为 $y _ { t }$。对于时间步 $n$ 的输入数据，共生成 $n - w$ 个输入-输出数据对。

**TABLE I: 受试者人口统计学信息**

<table><tr><td>特征</td><td>所有受试者 n=169</td><td>健康受试者 n=94</td><td>患者 n=75</td></tr>
<tr><td>年龄（岁），$\mathrm { M e a n } \pm \mathrm { S D } ( \mathrm { R a n g e } )$</td><td>$7 4 . 8 9 \pm 5 . 0 8 ( 6 0 - 8 7 )$</td><td>$7 4 . 6 6 \pm 4 . 7 5 ( 6 4 - 8 7 )$</td><td>$7 5 . 1 7 \pm 5 . 4 8 ( 6 0 - 8 7 )$</td></tr>
<tr><td>身高（cm），$\mathrm { M e a n } \pm \mathrm { S D } ( \mathrm { R a n g e } )$</td><td>$1 5 9 . 6 7 \pm 7 . 2 3 ( 1 4 1 . 9 \mathrm { { - } } 1 7 1 )$</td><td>$1 6 0 . 5 \pm 7 . 0 1 ( 1 4 1 . 9 – 1 7 1 )$</td><td>$1 5 5 . 6 5 \pm 7 . 5 5 ( 1 5 1 . 3 \ – 1 7 0 . 4 )$</td></tr>
<tr><td>体重（kg），$\mathrm { M e a n } \pm \mathrm { S D } ( \mathrm { R a n g e } )$</td><td>$6 1 . 1 \pm 9 . 3 4 ( 4 2 . 3 - 9 1 )$</td><td>$6 2 . 0 1 \pm 9 . 6 9 ( 4 2 . 3 - 9 1 )$</td><td>$5 9 . 9 2 \pm 8 . 7 9 ( 4 2 . 5 { - } 8 0 )$</td></tr>
<tr><td colspan="4">性别</td></tr>
<tr><td>- 男性 n (%)</td><td>68 (40.24%)</td><td>42 (44.68%)</td><td>26 (34.67%)</td></tr>
<tr><td>- 女性 n (%)</td><td>101 (59.77%)</td><td>52 (55.32%)</td><td>49 (65.34%)</td></tr>
</table>

![](images/9d5de048c5a14aa25b703a81004c696dd5db0856451599821bf394399f1892ed.jpg)  
Fig. 2: 输入信号包括：(a) 加速度信号，(b) 来自骨盆的角速度信号。(c) 通过从足部角速度信号中识别事件来提取真值标签（Groundtruth）。(d) 为右脚和左脚生成的支撑相（stance）与摆动相（swing）信号，作为基于回归的模型的两个连续真值标签信号。

## B. 深度学习模型

共训练并测试了 16 个深度学习模型。这些模型包括经典的卷积神经网络（CNN）模型、循环神经网络（RNN）模型，以及这些模型在引入或不引入双向（Bidirectional）机制或注意力（Attention）机制下的变体，还有将 CNN 与 RNN 模型相结合（无论是否引入双向或注意力机制）的若干混合模型。经典模型包括卷积神经网络（CNN）和多层感知器（MLP）模型；RNN 模型则包括长短期记忆网络（LSTM）、门控循环单元（GRU）、双向 LSTM（BiLSTM）、双向 GRU（BiGRU）、堆叠 LSTM（Stacked LSTM）、堆叠 GRU（Stacked GRU）、带注意力机制的堆叠 LSTM（stacked-LSTM-Att）以及带注意力机制的堆叠 GRU（stacked-GRU-Att）。混合模型包括 CNN 结合 LSTM（CNN-LSTM）、CNN 结合 GRU（CNN-GRU）、CNN 结合双向 LSTM（CNN-BiLSTM）、CNN 结合双向 GRU（CNN-BiGRU）、CNN 结合双向 LSTM 与注意力机制（CNN-BiLSTM-Att），以及 CNN 结合双向 GRU 与注意力机制（CNN-BiGRU-Att）。

对于所有模型而言，训练所用的原始数据均以骨盆部位的六路 IMU 信号作为输入，并以双脚各自的站立与摆动相位信号作为输出。这些模型利用所生成的输入与输出数据样本进行提前一步（one-step-ahead）预测。所有模型的超参数均经过优化，以使每个模型达到最佳精度。最终的输出层采用 Dense 层，并随后接一个线性激活函数（AF）。

深度学习模型所采用的架构如图 4 所示。图 4(a) 展示了 MLP 的架构。该模型对六路输入信号中的每一路都设置了一个包含 100 个神经元的 Dense 层，并使用一个拼接（concatenate）层将它们合并，然后再连接到一个 Dense 层。对于 CNN 模型，输入首先被重塑为 [样本数，时间步长，特征数] 的维度，以兼容一维卷积层 Conv1D（图 4(b)）。Conv1D 层之后接一个一维池化层 MaxPool1D。在展平（flatten）后，输出进入一个包含 50 个神经元的 Dense 层，最后的 Dense 层再将该序列经线性 AF 连接到最终输出。对于采用单层或堆叠 LSTM/GRU（无论是否带注意力机制）的 RNN 模型，单层的 vanilla LSTM 和 GRU 网络均具有 100 个隐藏单元，其后接一个 Dense 层和一个线性 AF（图 4(c)）。堆叠 LSTM 和堆叠 GRU 则由两层堆叠组成，每层均包含 100 个隐藏单元。首层将其隐藏状态（hidden state）传递给第二层，并由第二层用于输出预测。

对于结合 CNN 和 RNN 模型的混合模型，具有维度 [样本数, 窗口长度, 特征数] 的输入通过将窗口长度维度拆分为两段来进行重塑。当窗口长度为 80 个时间步时，输入维度 [样本数, 80, 6] 被变换为 [样本数, 2, 40, 6]。重塑后的输入被送入 Conv1D。卷积层之后接 MaxPooling1D，最后进行展平（图 4(d)）。对卷积块进行包装的 TimeDistributed 层使得可以将同一组卷积块应用于输入的所有时间切片 [23]。该 CNN 的输出进入具有 600 个隐藏单元的 LSTM 或 GRU 单层。Dense 层随后使用线性激活函数，与其他模型相同。

至于具有双向（Bidirectional）和注意力（Attention）机制的模型，双向机制将输入的前向和后向序列提供给两个不同的

![](images/6eef0f12fab54a9478b3eac07a57dbc2723bc265a8faa6364756bf6ad0ed55bb.jpg)  
yt = t 时刻的相位信号值  
w = 窗口长度（时间步数）  
w = 窗口长度（时间步数）

| x1 | x2 | … | $x_{w-1}$ | xw | … | xt-w | xt-w+1 | xt-w+2 | … | $x_{t-1}$ | xt | $x_{t\pm 1}$ | $x_{t+2}$ | … | $\boxed{\begin{array}{rl} x_{n-w} & \left| x_{n-w+1} \end{array}}$ | … | $\underline{x_{n-1}}$ | xn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| y1 | y2 | … | yw-1 | yw | … | yt-w | yt-w+1 | yt-w+2 | … | $x_{t-1}$ | yt | $y_{t+1}$ | yt+2 | … | yn−w | Yn-w+1 | … | yn-1 | yn |
| ** |  |  |  |  |  | ： |  |  |  | ： |  |  | ： |  | ： |  |  | ： |  |
| x1 | x2 | … | xw-1 | xw | … | xt-w | xt-w+1 | xt-w+2 | . | xt-1 | xt | $x_{t\pm 1}$ | xt+2 | … | xn−w | xn-w+1 | … | xn-1 | xn |
| y1 | y2 | … | yw-1 | yw | … | yt-w | yt-w+1 | yt-w+2 | … | yt-1 | yt | yt+1 | yt+2 | … | yn-w | yn-w+1 | … | yn-1 | yn |
| x1 | x2 | … | xw-1 | xw | … | xt-w | xt-w+1 | xt-w+2 | … | xt-1 | xt | $x_{t+1}$ | xt+2 | … | xn−w | xn-w+1 | … | xn−1 | xn |
| y1 | y2 | … | yw-1 | yw | … | yt-w | yt-w+1 | yt-w+2 | … | yt-1 | yt | yt+1 | yt+2 | … | yn-w | yn-w+1 | … | yn-1 | yn |
|  |  |  |  |  |  |  | ： |  |  |  | … |  |  |  | ： |  |  | ： |  |
| x1 | x2 | … | xw-1 | xw | … | xt-w | xt-w+1 | xt-w+2 | … | $\underline{x_{t-1}}$ | xt | $\underline{\boldsymbol{x}}_{t+1}$ | xt+2 | … | x |  |  | xn-1 | xn |
| y1 | y2 | … | yw-1 | yw | … | yt-w | yt-w+1 | yt-w+2 | … | yt-1 | yt | yt+1 | yt+2 | … | yn-w | yn-w+1 | … | yn-1 | yn |

xt = 时刻 t 的 IMU 数据  
n = 数据中的时间步数

图 3：训练数据被准备为输入-输出对，其中输入由滑动窗口内骨盆 IMU 信号的先前值组成，输出为窗口之后的下一时刻的真实相位信号（groundtruth phase signal）。

RNN 层，使网络能够利用每个输入点的过去和未来上下文进行预测 [24]。自注意力（Self-attention）机制使模型能够对输入序列的特定部分赋予更大的权重 [25]。在本研究中，Attention 使模型能够在滑动窗口的特定时段中找到时间依赖关系，而不是依赖所有时间步来做出更准确的预测。图 4(e) 展示了在 RNN 中加入双向（Bidirectional）机制，并在 RNN 网络输出端连接 Attention 机制的 CNN-RNN 混合模型。最终输出层中，Attention 层之后连接 Dense 层和线性激活函数（Linear AF），与其他模型一致。

所有模型均使用 Adam 优化器训练，并以均方误差作为损失度量。采用早停（early stopping）机制，以验证集损失最小化为标准回溯最佳模型，耐心值（patience）设为 10 个 epoch。

## C. 输出后处理

对输出信号进行后处理，以去除原始输出中的噪声尖峰并提高预测精度（图 5）。第一步，从输出信号中提取所有跨越零线的过渡点。接着，对这些过渡点进行滤波，以消除可能被误认为相位过渡的扰动。由于输出信号本质上是一个脉冲序列，因此通过区分噪声尖峰和真实脉冲来识别有效的相位过渡。一个脉冲被判定为有效需要满足三个条件：(1) 脉冲的最大值必须大于 0.5；(2) 脉冲的均值必须大于 0.6；(3) 脉冲宽度必须大于三个时间步。处理后的输出与真实值匹配得更好（图 5(b)）。

![](images/cdad8f2e2abca389c637fb4854f843207a481d6495ebeb74bb6ee3a3437b8edb.jpg)  
图 4：各模型的简要架构细节，(a) MLP (b) CNN (c) 基于 LSTM 和 GRU 的模型 (d) CNN-RNN 混合模型 (e) 加入双向和 Attention 机制的 CNN-RNN 混合模型

![](images/28178c7c4df13229c9138d6d73a4262c12ca8124d7fe65d971ff33e34c5de8d5.jpg)

![](images/8930c8bc12f2425a84c56f2fd20d4d1b7cc3b91a5283de69492eb50e8db55f3d.jpg)  
图 5：真实值与 (a) 预处理前输出信号和 (b) 后处理后输出信号的对比。

## D. 精度评估

准确率通过不同的容差窗口（tolerance window）进行测量，以验证检测的精度。当输出转换发生在容差窗口 W 内时，事件被定义为成功检测到（如图 6 所示）。各模型的性能在容差窗口 ±1TS、±2TS、……、±6TS 下进行测量，其中 1TS 表示一个 1ms 的时间步长。总体准确率定义为正确检测到的事件数占事件总数的百分比。

为进一步研究，对包含患者组数据的模型准确率进行了考察。模型的设置包括：用健康组数据进行训练和测试、用健康组数据训练但用患者组数据测试、以及用两组数据共同进行训练和测试。

![](images/904d9f4f708c723c07fa96bc4e4e7d7feb6f4463fed8093cc7d3f6d30b0ed90b.jpg)  
图 6：准确率通过不同精度的容差窗口 ±1TS、±2TS、……、±6TS 进行测量，其中 1TS 为一个 1ms 的单时间步长

表 II：所有模型的准确率结果
<table><tr><td>模型</td><td>±1TS</td><td>±2TS</td><td>±3TS</td><td>±TS</td><td>±5TS</td><td>±6TS</td></tr><tr><td>CNN-BiGRU-Att</td><td>93.89</td><td>98.29</td><td>99.02</td><td>99.46</td><td>99.64</td><td>99.73</td></tr><tr><td>CNN-BiLSTM</td><td>93.68</td><td>98.28</td><td>99.14</td><td>99.48</td><td>99.68</td><td>99.76</td></tr><tr><td>CNN-BiLSTM-Att</td><td>93.52</td><td>98.21</td><td>99.02</td><td>99.46</td><td>99.64</td><td>99.70</td></tr><tr><td>CNN-BiGRU</td><td>93.27</td><td>98.19</td><td>99.10</td><td>99.41</td><td>99.56</td><td>99.73</td></tr><tr><td>stacked-LSTM-Att</td><td>92.71</td><td>97.60</td><td>98.91</td><td>99.31</td><td>99.49</td><td>99.59</td></tr><tr><td>CNN-GRU</td><td>92.33</td><td>97.82</td><td>98.93</td><td>99.40</td><td>99.57</td><td>99.65</td></tr><tr><td>CNN-LSTM</td><td>91.77</td><td>97.89</td><td>98.94</td><td>99.44</td><td>99.61</td><td>99.76</td></tr><tr><td>stacked-GRU-Att</td><td>91.67</td><td>97.54</td><td>98.68</td><td>99.30</td><td>99.59</td><td>99.65</td></tr><tr><td>BiGRU</td><td>89.99</td><td>97.08</td><td>98.76</td><td>99.35</td><td>99.55</td><td>99.64</td></tr><tr><td>stacked-GRU</td><td>88.86</td><td>96.51</td><td>98.53</td><td>99.24</td><td>99.51</td><td>99.60</td></tr><tr><td>stacked-LSTM</td><td>88.04</td><td>96.64</td><td>98.53</td><td>99.18</td><td>99.47</td><td>99.59</td></tr><tr><td>BiLSTM</td><td>86.54</td><td>95.98</td><td>98.17</td><td>99.02</td><td>99.35</td><td>99.51</td></tr><tr><td>GRU</td><td>78.98</td><td>92.70</td><td>96.72</td><td>98.32</td><td>99.13</td><td>99.47</td></tr><tr><td>LSTM</td><td>76.83</td><td>91.61</td><td>96.21</td><td>98.18</td><td>98.89</td><td>99.24</td></tr><tr><td>MLP</td><td>69.55</td><td>86.57</td><td>92.88</td><td>95.73</td><td>97.05</td><td>97.53</td></tr><tr><td>CNN</td><td>68.58</td><td>87.23</td><td>94.25</td><td>96.76</td><td>98.01</td><td>98.56</td></tr></table>

## III. 结果

表 II 总结了健康组在不同大小容差窗口下训练和测试模型的平均准确率。使用 ±6TS 容差窗口的 CNN-BiGRU-Att 模型达到了 99.73 的最高准确率。所有使用双向机制（无论是否带有注意力机制）的混合模型，其准确率均与表现最佳的 CNN-BiGRU-Att 相当。

准确率随容差窗口大小的增大而提高。使用更宽的容差窗口时，检测率提高，但精确率降低。每个模型均在 ±6TS 容差窗口下取得了其最佳准确率。由于所有模型在大于 ±3TS 的容差窗口下偏差较小，因此模型之间的比较是基于 ±1TS 容差窗口下的准确率进行的，这表示最精确的检测。

图 7 显示了所有事件（图 7(a)）、HS 事件（图 7(b)）和 TO 事件（图 7(c)）的准确率曲线。九个关键模型被选为各类事件的最佳表现模型。三个图中均对左右脚事件取了平均值。在所有三类事件中，准确率均随着容差窗口的增加而提高。在大多数模型中，TO 事件相比 HS 事件取得了更高的准确率。

为了观察左右脚事件在性能上的差异，计算了预测事件与真实事件之间的时间步绝对误差。图 8 显示了 CNN-BiGRU-Att 的事件预测误差。所有事件的平均绝对误差（MAE）为 5.77 ms，其中 HS 和 TO 事件的平均绝对误差分别为 6.239 ms 和 5.24 ms。TO 事件的误差少于 HS 事件。在使用右脚事件和左脚事件之间未发现显著差异。

对 6 个输入信号的子集进行了消融实验，以研究预测准确率与输入信号数量之间的关系。表 III 列出了 CNN-BiGRU-Att 模型中表现最佳的输入组合列表。使用 ML 或 ROT，以及同时使用两者，均能提高准确率。当使用两个输入信号时，AP 与 ML 的组合以及 AP 与 ROT 的组合在 ±1TS 容差窗口下达到了约 88% 的准确率。使用 AP、ML、V 和 ROT 四个输入信号时，取得了超过 94% 的最高准确率。当使用全部六个输入信号时，准确率达到了 93.89%。

表 IV 展示了使用相同或不同受试者组进行训练和测试时 CNN-BiGRU-Att 模型的准确率。当使用全部六个输入信号时，使用健康组进行训练和测试的模型在 ±1TS 容差窗口下取得了 93.89% 的准确率。当使用健康组训练但使用患者组测试时，准确率为 63.10%。当使用两组同时进行训练和测试时，模型的准确率达到了 93.63%。当使用 AP、ML、V 和 ROT 这四个输入信号时，使用健康组进行训练和测试的模型在 ±1TS 容差窗口下取得了最高准确率 94.11%，高于使用全部六个信号时的准确率。当模型使用两组同时训练时，使用全部六个输入信号在 ±1TS 和 ±2TS 容差窗口下均取得了更高的准确率。为了观察这些结果中 HS（脚跟着地）和 TO（脚尖离地）事件的准确率，准确率曲线图如图 9 所示。与之前的观察一致，TO 事件的识别准确率高于 HS 事件。

## IV. 讨论（DISCUSSION）

本研究旨在探索一种利用骨盆部位的单个传感器结合深度学习技术来提高老年人步态事件检测准确率的方法。共训练并测试了 16 个模型来预测老年人的步态事件，并将其预测结果与通过脚部采集的真实值进行了比较。CNN-BiGRU-Att 模型在 ±6TS 容差窗口下取得了最高的 99.73% 的准确率，该准确率与在下肢多个部位使用多个传感器的结果相当。消融实验表明，使用 AP、ML、V 和 ROT 这四个信号集可在 ±1TS 容差窗口下取得超过 94% 的准确率。本研究首次将深度学习技术应用于基于骨盆数据的步态事件预测，并提出了一种可靠的方法，即利用骨盆部位的单个传感器进行老年人步态事件检测。这些发现有望减轻步态测量的负担，并增加未来各种技术与老年人步态监测相结合应用的潜力。

(a)  
![](images/fec6bddf03ab417757f4fb2e30eb1b8b3ab7ecc4f3ea19d4e66ea39844be32eb.jpg)

(b)  
![](images/514d6b6cd82e02b3f25d86f2a8b055df538be818a86c5215c98f0b59218e35e9.jpg)

(c)  
![](images/458ed61b53e0c09041fc4b12ada8b07e30904e619cab0c46188abbc1f9c73818.jpg)  
图 7：关键模型的准确率曲线图，分别为 (a) 所有事件、(b) HS（脚跟着地）事件、(c) TO（脚尖离地）事件

表 III：使用输入信号子集的消融研究结果

<table><tr><td>输入信号数量</td><td>输入</td><td>±1TS</td><td>±2TS</td><td>±3TS</td><td>±4TS</td><td>±5TS</td><td>±6TS</td></tr><tr><td rowspan="5">1</td><td>[AP]</td><td>20.57</td><td>27.75</td><td>32.12</td><td>34.65</td><td>36.26</td><td>37.52</td></tr><tr><td>[ML]</td><td>60.12</td><td>79.12</td><td>88.20</td><td>92.57</td><td>94.80</td><td>96.14</td></tr><tr><td>[V]</td><td>15.37</td><td>21.60</td><td>25.12</td><td>27.47</td><td>28.83</td><td>29.74</td></tr><tr><td>[TIL]</td><td>24.52</td><td>33.52</td><td>38.65</td><td>41.89</td><td>44.03</td><td>45.42</td></tr><tr><td>[OBL]</td><td>15.52</td><td>24.05</td><td>31.13</td><td>36.47</td><td>40.53</td><td>43.18</td></tr><tr><td rowspan="3">2</td><td>[ROT]</td><td>58.00</td><td>75.96</td><td>84.42</td><td>88.45</td><td>90.72</td><td>92.09</td></tr><tr><td>[AP, ML]</td><td>88.16</td><td>97.12</td><td>98.66</td><td>99.14</td><td>99.45</td><td>99.65</td></tr><tr><td>[AP, ROT]</td><td>87.23</td><td>95.45</td><td>97.61</td><td>98.66</td><td>99.27</td><td>99.45</td></tr><tr><td rowspan="2">3</td><td>[AP, ML, TIL]</td><td>90.15</td><td>97.28</td><td>98.68</td><td>99.21</td><td>99.60</td><td>99.70</td></tr><tr><td>[AP, ML, ROT]</td><td>90.28</td><td>97.34</td><td>98.86</td><td>99.18</td><td>99.50</td><td>99.68</td></tr><tr><td rowspan="2">4</td><td>[AP, ML, V, TIL]</td><td>93.40</td><td>98.25</td><td>98.95</td><td>99.30</td><td>99.60</td><td>99.68</td></tr><tr><td>[AP, ML, V, ROT]</td><td>94.11</td><td>98.48</td><td>99.13</td><td>99.46</td><td>99.68</td><td>99.79</td></tr><tr><td rowspan="3">5</td><td>[AP, ML, V, TIL, OBL]</td><td>87.97</td><td>96.57</td><td>98.31</td><td>99.14</td><td>99.49</td><td>99.62</td></tr><tr><td>[AP, ML, V, TIL, ROT]</td><td>93.89</td><td>98.04</td><td>98.97</td><td>99.53</td><td>99.73</td><td>99.84</td></tr><tr><td>[AP, ML, V, OBL, ROT]</td><td>93.99</td><td>98.28</td><td>98.94</td><td>99.38</td><td>99.68</td><td>99.81</td></tr><tr><td>6</td><td>[AP, ML, V, TIL, OBL, ROT]</td><td>93.89</td><td>98.29</td><td>99.02</td><td>99.46</td><td>99.64</td><td>99.73</td></tr></table>

表 IV：使用患者数据时的准确率结果
<table><tr><td>输入信号</td><td>训练集</td><td>测试集</td><td>±1TS</td><td>±2TS</td><td>±3TS</td><td>±4TS</td><td>±5TS</td><td>±6TS</td></tr><tr><td rowspan="3">[AP, ML, V, TIL, OBL, ROT]</td><td>HS</td><td>HS</td><td>93.89</td><td>98.29</td><td>99.02</td><td>99.46</td><td>99.64</td><td>99.73</td></tr><tr><td>HS</td><td>P</td><td>63.10</td><td>84.09</td><td>93.30</td><td>96.94</td><td>98.44</td><td>99.05</td></tr><tr><td>混合</td><td>混合</td><td>93.63</td><td>98.04</td><td>98.85</td><td>99.21</td><td>99.44</td><td>99.59</td></tr><tr><td rowspan="3">[AP, ML, V, ROT]</td><td>HS</td><td>HS</td><td>94.11</td><td>98.48</td><td>99.13</td><td>99.46</td><td>99.68</td><td>99.79</td></tr><tr><td>HS</td><td>P</td><td>62.78</td><td>83.63</td><td>92.40</td><td>96.25</td><td>97.97</td><td>98.69</td></tr><tr><td>混合</td><td>混合</td><td>92.80</td><td>97.91</td><td>98.95</td><td>99.30</td><td>99.52</td><td>99.66</td></tr></table>

所提出的方法。

将 CNN（卷积神经网络）与 RNN（循环神经网络）模型结合使用提高了预测准确率，因为 CNN 首先提取有效的特征，然后随后的 RNN 模型可以按顺序处理这些特征。所加入的双向（Bidirectional）机制同时考虑了前向和后向的时间视角，从而进一步提高了准确率。在 CNN-BiGRU 中加入 Attention（注意力）机制进一步提升了准确率，因为 Attention 机制能够聚焦于与输出预测更为相关的时间步。由于 MLP（多层感知机）和 CNN 无法利用滑动窗口中信息的时间特性，因此在容差窗口达到 ±4TS 之前，它们都无法准确预测输出信号中的相位跃迁，尽管随着容差窗口尺寸的增大，它们的准确率有所提升。

![](images/25932c6a0c8e6187db863741f898cf7498f4a5c9aed3e219826e37905ee81d82.jpg)

![](images/a07446dc173a6aad1b38949f60e2138db8f4974e14f583743de52d9d1db15760.jpg)  
图 8：左脚和右脚跟着地（heel-strike）与脚尖离地（toe-off）事件的事件预测误差

许多先前的研究已经探讨了使用多个传感器来分析步态事件。Lin 等人 [17] 使用了基于 LSTM 的回归模型，该模型使用五个 IMU（两个位于大腿上，两个位于小腿上，一个位于左脚上）。他们对足跟着地（HS）实现了 2ms 的平均误差（ME），然而脚尖离地（TO）的误差较大，ME 为 -18ms。在文献 [16] 中，Hannink 等人使用了基于深度 CNN 的网络，输入来自放置在脚上的两个惯性传感器，并预测了不同的步态参数，包括作为输出的 HS 和 TO 事件，他们报告的误差分别为 ±70ms 和 ±120ms。Sarshar 等人 [26] 也使用 RNN 来训练附着在小腿上的两个 IMU 传感器，报告了 HS 和 TO 事件的准确率均为 0.9977，但他们没有计算预测延迟中的误差。利用基于规则的方法，Fadillioglu 等人 [27] 提出了一种使用附着在右小腿上的陀螺仪的自动步态事件检测方法，报告了 HS 和 TO 事件的平均绝对误差（MAE）分别为 7ms 和 19ms。最近，Yu 等人 [28] 也使用位于右脚的单个传感器，采用 LSTM-HMM 混合模型进行步态事件检测，仅报告了准确率而未提及延迟。他们报告的 HS 和 TO 事件的准确率分别为 0.9679 和 0.9846。但由于这两项工作仅在右侧使用了单个传感器，因此无法获取左侧的事件。相比之下，本文提出的方法仅在腰部这一准确度相对较低的位置使用了单个传感器，却表现出了更优越的整体性能。使用多个传感器的研究不仅会阻碍自然步态，而且也远离这些方法在日常生活中的实际应用。如果将单个传感器附着在肢体的一侧，则无法从另一侧提取步态事件，因此除了腰部以外，对于下肢任何位置的完整步态事件信息，都需要两个传感器。尽管与下肢相比，腰部是一个准确度较低的位置，但本研究中所提出的方法仍然仅使用单个传感器就实现了更准确的步态事件检测。

与之前仅使用腰部单一传感器检测步态事件的方法相比，所提出的 CNN-BiGRU-Att 模型取得了显著更高的精度。根据最近一项关于使用安装在腰部的 IMU（惯性测量单元）传感器进行步态事件检测方法的综述，Gonzalez 等人 [29] 使用基于规则的方法，分别在 HS（脚跟着地）和 TO（脚尖离地）事件上达到了最低的 MAE（平均绝对误差）15 ms 和 9 ms。McCamley 等人 [30] 提出了一种基于高斯 CWT（连续小波变换）的步态事件估计方法，使用腰部单一惯性传感器，分别在 HS 和 TO 上报告了 19 ms 和 32 ms 的 MAE。Soaz 等人 [31] 同样使用基于规则的方法配合腰部单一传感器，在其实验中评估老年人的步态，并报告了 HS 的误差为 20 ms。除了相比所提出方法具有相对更高的误差外，这些研究也没有足够的受试者来证明其方法的泛化能力。

研究发现 TO 事件的误差小于 HS 事件。骨盆信号在这些事件附近出现极值可能是一种可能的解释。如图 2 所示，TO 事件与骨盆信号（如 V、TIL 和 ROT）中的峰值更为对齐，这些峰值之后便出现 TO 事件。这些特征可能有助于 TO 事件相比 HS 事件表现更佳。

就所使用的单一输入信号而言，ML 和 ROT 由于富含左右方向的信息，因此表现出色。AP 和 V 信号最为清晰但缺乏左右方向的信息，因此当仅基于其中之一进行预测时，表现出最低的性能。将 AP 与 ML 或 ROT 中的任一信号相结合，可能将前后方向的运动信息与左右方向的运动信息融合在一起，从而显著提高了精度。使用 AP、ML、V 和 ROT 的整体精度高于使用全部六个信号的精度，这可能是由于在老年人群中 TILT 和 OBL 存在更大的变异 [32]。AP、ML 和 V 均为加速度信号，这可能也起到了积极作用，因为与陀螺仪相比，加速度计的输出通常不易受传感器位置误差的影响。

当模型使用相同和不同组进行训练和测试时，使用健康组训练但使用患者组测试的模型，在使用全部六个输入信号时，于±1TS处的准确率为63.10。使用健康组训练的模型对患者组所具有的变异和异常并不熟悉，但其准确率随着容差窗口大小的增大而提升。在健康组中加入TILT和OBL并未显著提高模型的准确率，而在患者组中加入它们则提高了准确率，这可能是因为患者组中所有信号的变异更大。建议对信号表现出较大变异的患者（即使是被认为是相对清晰的信号）考虑使用全部六个信号。

为了研究在使用全部六个输入信号与仅限于四个信号时注意力权重的变化，使用stacked-LSTM-Att模型绘制了每个时间步的平均注意力权重值（图10）。当模型使用全部六个输入信号时，它对跟着地（HS）和趾离地（TO）事件之间的双腿支撑相（DLS）给予了比单腿支撑相（SLS）更高的注意力。当使用四个信号时，对SLS相给予了略多的注意力，且其DLS相的峰值注意力低于使用全部六个信号时的峰值。

![](images/99df02db00bb0dadf7ed8b9ca6b419e59948bb18023f0d6b3c5cfb5a374aeb16.jpg)

![](images/3e497d91ab568cd965e7d67b28353e4bffea7ebfb3aeda6394cfb1fbdb02e918.jpg)

(c)  
![](images/523d326b655296c61f4433aaa177a5c39a449477e26cf551e657d6e84f63d95a.jpg)  
图9：使用患者数据工作时，(a)所有事件、(b)跟着地（heel-strike）事件和(c)趾离地（toe-off）事件的准确率图。图中包含使用全部6个骨盆信号（AllSig）和限制4个信号（LimSig）训练的模型的结果。训练和测试包括健康组训练—患者测试（HS-P）、健康组训练—健康组测试（HS-HS）以及混合训练—混合测试（M-M）。

![](images/cf02204dc3cb70299973a7dffb0ad7173e63256a8b6158bbc891f79fa6820215.jpg)  
图10：使用全部6个输入骨盆信号和限制4个输入信号的stacked-LSTM模型的平均注意力图。与右TO和右HS之间的SLS相比，模型对右HS和左TO之间的DLS相给予了更多注意力。

本研究的局限之一在于，所提出的方法尚未在其他身体损伤上进行评估，例如骨关节炎、骨骼畸形以及神经系统疾病（如偏瘫、帕金森病、亨廷顿病和阿尔茨海默病）。此外，步态数据并非通过在自然环境和日常生活中对受试者进行长期或连续监测所采集。因此，我们未来的工作将包括在非传统环境中以及在非受控和不受约束的条件下，针对现实世界行走的步态事件检测。

## V. 结论

本研究提出了基于深度学习的步态检测方法，作为一种新颖且可靠的方式，仅使用骨盆处的单个传感器即可检测老年人的步态。共训练并测试了 16 个模型，包括 CNN、RNN 以及带有或不带有双向（Bidirectional）和注意力（Attention）机制的 CNN-RNN 混合模型，其中 CNN-BiGRU-Att 模型在 ±6TS 和 ±1TS 容差窗口下分别达到了 99.73% 和 93.89% 的较高准确率。在前人步态事件检测研究的基础上进一步推进，所提模型在预测误差方面表现出显著改善，HS 和 TO 事件的 MAE 分别为 6.239 毫秒和 5.24 毫秒。对于健康受试者，使用三个加速度信号加上 ROT 作为输入信号比使用全部六个信号表现出更好的性能；而对于患者，使用全部六个信号的性能更优。该研究表明，从骨盆处的单个传感器进行可靠的步态检测是可行的，预计将有助于降低步态检测的负担，并扩展步态检测在未来可穿戴设备中的适用性。

## 致谢

本研究得到了韩国科学技术研究院机构项目（项目编号 2E31051）以及由韩国政府（科学技术信息通信部、贸易工业能源部、卫生福利部、食品药品安全部）资助的韩国医疗器械开发基金资助项目（项目编号 1711139131）的支持。

## 参考文献

[1] Stephanie Studenski et al. "Gait speed and survival in older adults". In: Jama 305.1 (2011), pp. 50–58.

[2] Muhammad Zeeshan Arshad et al. "Gait-based Frailty Assessment using Image Representation of IMU Signals and Deep CNN". In: 2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). IEEE. 2021, pp. 1874–1879.

[3] Dawoon Jung et al. "Classifying the Risk of Cognitive Impairment Using Sequential Gait Characteristics and Long Short-Term Memory Networks". In: IEEE Journal of Biomedical and Health Informatics (2021).

[4] Joe Verghese et al. "Gait dysfunction in mild cognitive impairment syndromes". In: Journal of the American Geriatrics Society 56.7 (2008), pp. 1244–1251.

[5] Michelle M Mielke 等. "评估认知与步态之间的时间关系：缓慢步态可预测梅奥诊所老龄化研究中的认知衰退". 见于: *Journals of Gerontology Series A: Biomedical Sciences and Medical Sciences* 68.8 (2013), 第 929–937 页.

[6] David E Krebs, Joan E Edelstein, 及 Sidney Fishman. "观察性运动学步态分析的可靠性". 见于: *Physical Therapy* 65.7 (1985), 第 1027–1033 页.

[7] Myeongkyu Kim 及 Donghun Lee. "基于 IMU 的足地接触检测（FGCD）算法的开发". 见于: *Ergonomics* 60.3 (2017), 第 384–403 页.

[8] Laurent Oudre 等. "基于模板的惯性测量单元步态检测". 见于: *Sensors* 18.11 (2018), 第 4033 页.

[9] Hyo-Ki Lee 等. "使用单个三轴加速度计评估偏瘫步态的新型算法". 见于: *2009 Annual International Conference of the IEEE Engineering in Medicine and Biology Society*. IEEE. 2009, 第 3964–3966 页.

[10] Susan Barker 等. "时空步态分析系统的准确性、可靠性和有效性". 见于: *Medical engineering & physics* 28.5 (2006), 第 460–467 页.

[11] David A Winter. *人体运动的生物力学与运动控制*. John Wiley & Sons, 2009.

[12] Andrea Mannini 及 Angelo Maria Sabatini. "基于陀螺仪足部运动数据应用隐马尔可夫模型进行步态相位检测及步行–慢跑活动的区分". 见于: *Gait & posture* 36.4 (2012), 第 657–661 页.

[13] Juri Taborri 等. "基于可穿戴惯性传感器网络检测步态相位的新型 HMM 分布式分类器". 见于: *Sensors* 14.9 (2014), 第 16212–16234 页.

[14] Joonbum Bae 及 Masayoshi Tomizuka. "基于隐马尔可夫模型的步态相位分析". 见于: *Mechatronics* 21.6 (2011), 第 961–970 页.

[15] Andrea Mannini 等. "基于惯性传感器的步态分类机器学习框架：应用于老年人、中风后及亨廷顿病患者". 见于: *Sensors* 16.1 (2016), 第 134 页.

[16] Julius Hannink 等. "基于深度卷积神经网络的传感器步态参数提取". 见于: *IEEE journal of biomedical and health informatics* 21.1 (2016), 第 85–93 页.

[17] Po-Hsin Lin 等. "基于腿部和鞋载 IMU 与深度学习的步态参数分析". 见于: *2021 International Symposium on VLSI Design, Automation and Test (VLSI-DAT)*. IEEE. 2021, 第 1–4 页.

[18] Verne T Inman, Howard D Eberhart 等. "正常与病理步态的主要决定因素". 见于: *JBJS* 35.3 (1953), 第 543–558 页.

[19] Wiebren Zijlstra 及 At L Hof. "基于躯干加速度评估人类行走过程中的时空步态参数". 见于: *Gait & posture* 18.2 (2003), 第 1–10 页.

[20] Roel De Ridder 等. "商用无线躯干三轴加速度计系统用于步态分析的同期效度". 见于: *Journal of sport rehabilitation* 28.6 (2019).

[21] John E Morley, TK Malmstrom, and DK Miller. "A simple frailty questionnaire (FRAIL) predicts outcomes in middle aged African Americans". In: *The journal of nutrition, health & aging* 16.7 (2012), pp. 601–608. （中译："一项简易衰弱问卷（FRAIL 量表）可预测中年非裔美国人的结局"）

[22] Marshal F Folstein, Susan E Folstein, and Paul R McHugh. ""Mini-mental state": a practical method for grading the cognitive state of patients for the clinician". In: *Journal of psychiatric research* 12.3 (1975), pp. 189–198. （中译："简易精神状态检查表"：一种用于临床评估患者认知状态的实用方法"）

[23] Martín Abadi 等. "Tensorflow: A system for largescale machine learning". In: *12th {USENIX} symposium on operating systems design and implementation ({OSDI} 16)*. 2016, pp. 265–283. （中译："TensorFlow：面向大规模机器学习的系统"）

[24] Alex Graves. "Supervised sequence labelling". In: *Supervised sequence labelling with recurrent neural networks*. Springer, 2012, pp. 5–13. （中译："监督序列标注"）

[25] Ashish Vaswani 等. "Attention is all you need". In: *Advances in neural information processing systems*. 2017, pp. 5998–6008. （中译："注意力即一切"）

[26] Mustafa Sarshar, Sasanka Polturi, and Lutz Schega. "Gait phase estimation by using LSTM in IMU-based gait analysis—Proof of concept". In: *Sensors* 21.17 (2021), p. 5749. （中译："基于 IMU 步态分析中使用 LSTM 进行步态相位估计——概念验证"）

[27] Cagla Fadillioglu 等. "Automated gait event detection for a variety of locomotion tasks using a novel gyroscope-based algorithm". In: *Gait & Posture* 81 (2020), pp. 102–108. （中译："基于新型陀螺仪算法的多种运动任务步态事件自动检测"）

[28] Zhipeng Yu 等. "Gait Phase Detection Based on a Foot-Mounted Inertial Sensor Using Long Short-Term Memory Enhanced by Hidden Markov Model". In: *2021 26th International Conference on Automation and Computing (ICAC)*. IEEE. 2021, pp. 1–5. （中译："基于足戴式惯性传感器使用长短期记忆网络结合隐马尔可夫模型增强的步态相位检测"）

[29] Rafael C Gonzalez 等. "Real-time gait event detection for normal subjects from lower trunk accelerations". In: *Gait & posture* 31.3 (2010), pp. 322–325. （中译："基于下躯干加速度的正常受试者实时步态事件检测"）

[30] John McCamley 等. "An enhanced estimate of initial contact and final contact instants of time using lower trunk inertial sensor data". In: *Gait & posture* 36.2 (2012), pp. 316–318. （中译："利用下躯干惯性传感器数据对初始着地时刻与最终离地时刻的增强估计"）

[31] Cristina Soaz and Klaus Diepold. "Step detection and parameterization for gait assessment using a single waist-worn accelerometer". In: *IEEE Transactions on Biomedical Engineering* 63.5 (2015), pp. 933–942. （中译："使用单一腰部佩戴加速度计进行步态评估的步态检测与参数化"）

[32] Jacquelin Perry and Judith M Burnfield. "Gait analysis. Normal and pathological function 2nd ed". In: *California: Slack* (2010). （中译："步态分析：正常与病理功能 第2版"）