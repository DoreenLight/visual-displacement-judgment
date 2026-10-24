# 使用运动学方法和 LSTM 模型检测足跟着地和足尖离地事件

Longbin Zhang, Zhizhang Li<sup>\*</sup>, Xinyi Fu, Yi Xie, Zekun Liu, Xiaoyue Yan, Suiyuan Wang, Te Zhang, Hui Zhang, Kailun Yang, Tsung-Lin Wu, Prayook Jatesiktat, Ananda Sidarta, and Wei Tech Ang

摘要— 准确的步态事件检测对于步态分析、康复和辅助技术至关重要，尤其是在外骨骼控制中，精确识别支撑相和摆动相更是必不可少。本研究评估了七种基于运动学的方法和一种长短期记忆（Long Short-Term Memory, LSTM）模型在来自 588 名健全受试者的 4363 个步态周期中检测足跟着地（heel strike）和足尖离地（toe-off）事件的性能。结果表明，尽管 Zeni et al. 方法在基于运动学的方法中取得了最高的准确率，但其他方法表现出系统性偏差或需要针对特定数据集进行微调。LSTM 模型的表现与 Zeni et al. 相当，提供了一种没有系统性偏差的数据驱动替代方案。这些发现突显了基于深度学习的方法在步态事件检测中的潜力，同时强调了在临床人群和多种步态条件下进行进一步验证的必要性。未来的研究将探索这些方法在病理人群（例如中风后患者和膝骨关节炎患者）中的泛化能力，以及它们在各种步态条件和数据采集环境下的鲁棒性，以增强其在康复和外骨骼控制中的适用性。

## I. 引言

外骨骼正作为一种强大的辅助技术崭露头角，旨在康复和日常环境中增强行动能力、恢复步态功能并减轻使用者的身体负担 [1]–[3]。外骨骼控制中的一个关键挑战是施加辅助的精确时机，这依赖于准确的步态事件检测来区分站立相和摆动相 [4], [5]。对这些事件的错误识别会导致辅助时机不当，从而降低效率、增加代谢消耗，并可能危及使用者的安全。因此，步态事件检测是步态分析的基础组成部分，用以确保精确识别步态周期内的关键时刻，例如足跟着地（heel strike, HS）和足尖离地（toe off, TO）[6]。准确的检测对于计算时空参数和运动学变量至关重要，而这些参数对优化外骨骼控制策略具有关键作用。步态事件识别中的误差会在系统中传播，导致关节角度估计不准确以及运动同步受损 [7], [8]。在临床环境中，精确的步态事件检测对于诊断步态异常、监测康复进展以及制定个性化干预措施至关重要 [9]。它还在向患者提供实时反馈、预防继发性并发症以及定制辅助设备方面发挥着关键作用 [10]。除临床应用外，准确的步态事件检测还能支持假肢和矫形器的优化，提升运动表现，并实现跌倒风险的预测建模 [11]。鉴于其在康复和辅助机器人领域的重要意义 [12]，改进步态事件检测对于推进外骨骼技术、确保流畅的人机交互以及增强步态障碍人群的行动能力至关重要。

在步态分析中，识别 HS（脚跟着地）和 TO（脚趾离地）的金标准是在实验室环境中利用预设的垂直地面反作用力（GRF）阈值（通常在 5 至 20 N 之间 [13], [14]）通过测力台进行测量，此时受试者走过嵌入有测力台的步道 [15]–[17]。然而，测力台的可用性以及获取足够数量的有效单足接触可能构成重大挑战，尤其是在存在拖步、小步幅或使用助行器等病理特征的情况下 [7]。此外，使用测力台通常需要力敏步道或多块测力台，这不仅昂贵且不切实际，还要求每一步都落在单个测力台上，并确保每次只有一只脚接触测力台 [18]。便携式替代方案（如压力鞋垫或足底开关）虽然提供了一定的灵活性，但需要额外的设备并对受试者的鞋具进行改装，从而限制了其在异常步态人群中的使用，特别是那些在摆动相期间缺乏足够抬足高度的人群。此外，这些便携式方法的准确性极大地依赖于传感器的精确放置 [8]。尽管面临诸多挑战，测力台阈值法仍然是金标准，但仍持续需要改进步态事件检测方法，以提升其在各类临床和研究环境中的准确性与适用性。

基于运动学的模型和深度学习方法广泛应用于健康人群和病理人群的步态事件检测 [6], [8], [16], [19]–[23]。这些模型通常依赖于通过动作捕捉系统或可穿戴传感器（如 IMUs）捕获的足部或骨盆线段 [17], [24], [25]。French et al. [24] 发现水平踝-跟距离在平地行走过程中检测足跟着地（heel strike, HS）最为准确，而骶骨-足跟距离最适合检测足尖离地（toe off, TO）。Voisard et al. [25] 使用惯性测量单元（inertial measurement units, IMUs）检测步态事件，在健康受试者中表现出高精度（TO：8 ms，HS：7 ms），在多发性硬化症患者中表现出合理的精度（TO：23 ms，HS：15 ms）。Bonci et al. [17] 结合了标记点位置和速度，实现了超过 99% 的灵敏度，但受到小样本量和标记点要求的限制。包括长短期记忆（Long Short-Term Memory, LSTM）网络在内的深度学习模型已成为替代方案。Lempereur et al. [16] 使用 LSTM 检测步态障碍儿童的步态事件，实现了高精度检测，其中 HS 在 5.5 ms 内检出，TO 在 10.7 ms 内检出。Kim et al. [6] 在脑瘫儿童中实现了 HS 89.7% 和 TO 71.6% 的检测率。Filtjens et al. [26] 将时间卷积网络（Temporal Convolutional Networks）应用于帕金森病，获得的 F1 分数超过 0.99。Dumphart et al. [7] 使用 LSTM 检测健康个体和病理个体的事件，报告了较高的检测率，但强调了标记点放置和实验室设置变异性的影响。尽管取得了进展，但很少有研究在不同数据中心和多样化受试人群之间对基于运动学的模型和深度学习模型进行比较。

本研究的目的是使用基于运动学的方法和 LSTM 模型来检测足跟着地和足尖离地事件，并评估它们的预测精度。我们假设 LSTM 模型将达到与传统基于运动学的方法相当或更优的精度，同时在各试验之间保持一致性。

## II. 方法

## A. 实验设置

从新加坡康复研究院（Rehabilitation Research Institute of Singapore）共收集了来自 588 名健全受试者（年龄：$4 5 ~ \pm ~ 1 6 . 2$ 岁；性别：317女/271男；体重：62.7 ± 13.5 kg；身高：164.4 ± 9.1 cm）的 4363 个步态周期。该研究获得了南洋理工大学机构审查委员会（Nanyang Technological University Institutional Review Board）（IRB-2022-382）的批准。在新加坡使用基于标记点的 Miqus M3 Qualisys 系统（200 Hz）记录标记点轨迹。同时测量地面反作用力，并使用 20 N 的阈值来识别足跟着地和足尖离地事件，作为基于运动学模型和机器学习模型的真实基准（Ground truth）（Fig. 1）。

## B. 步态事件识别模型

## (a) 基于运动学的模型

为了识别足跟着地（heel strike, HS）和足尖离地（toe off, TO）步态事件，我们评估了源自同行评审文献的七种基于运动学的方法。这些方法采用了不同的生物力学原理，利用基于位置、速度或加速度的标准，应用于包括放置在足跟、足尖和骨盆上的关键解剖学标记点（Table I）。每种方法都具有独特的优势和局限性，特别是在对步态变异的敏感性以及在不同行走条件下的鲁棒性方面。

## • 基于位置的方法：

Zeni et al. (2008) 提出了一种广泛使用的方法，用于基于足跟和足尖标记点相对于骨盆位置的局部极值来检测 HS 和 TO。具体而言，当足跟标记点达到其最大水平位移时识别为 HS，而当足尖标记点达到其最小水平位移时识别为 TO，两者均相对于骨盆进行测量（Table I）。

Desailly et al. (2009) 引入了一种基于滤波的方法，其中对足跟和足尖标记点的水平轨迹应用高通滤波，以消除低频漂移。HS 在经高通滤波（0.5 \* 步频）的足跟标记点水平轨迹的第一个极大值处被检测到，而 TO 则在经高通滤波（1.1 \* 步频）的足尖标记点水平轨迹的最后一个极小值处被识别。该方法旨在通过减轻标记点噪声和软组织伪影的影响来提高准确性。

## • 基于速度的方法：

O’Connor et al. (2007) 采用了一种基于速度的检测方法，根据垂直速度极值来定义 HS 和 TO。具体而言，在足跟和足尖标记点之间中点的垂直速度局部极小值处识别为 HS，在局部极大值处识别为 TO。

Ghoussayni et al. (2004) 引入了一种基于阈值的速度方法，其中当足跟标记点的矢状面速度降至 5 cm/s 以下时检测为 HS，当足尖标记点的矢状面速度超过 5 cm/s 时识别为 TO。然而，Ghoussayni et al. 推荐的速度阈值后来被发现过低，Bruening et al. [15] 将其提高至 50 cm/s。在本研究中，我们采用修订后的 50 cm/s 阈值。

## • 基于加速度的方法：

Hreljac et al. (2000) 使用加速度曲线来检测 HS 和 TO，结合垂直和水平加速度分量来捕捉步态事件动力学特征。在足跟标记点的垂直加速度局部极大值处识别为 HS，前提是加速度的导数（加加速度 / jerk）为零。同样，在足尖标记点的水平加速度局部极大值处检测到 TO，条件也是相应的加加速度为零。

Hsue 等人（2007）通过引入加加速度约束（jerk constraints，加速度的变化率）扩展了基于加速度的检测方法。HS（足跟着地）被确定在足跟标记点水平加速度的局部极小值处，而 TO（足尖离地）则在足尖标记点水平加速度的局部极大值处被检测到。

## • 混合方法：

Bonci 等人（2022）通过结合相对于行走速度的 3D 速度阈值，对 Zeni 的方法进行了改进。HS 最初使用 Zeni 等人（2008）的方法进行检测，随后根据足跟标记点的 3D 速度进行调整：对于后足着地低于 0.5 \* 行走速度，对于前足着地低于 0.8 \* 行走速度。TO 首先使用 Zeni 等人（2008）的方法进行识别，然后在足尖标记点的 3D 速度超过 0.8 \* 行走速度时进行修正，并在足跟标记点的 3D 速度达到局部峰值后下降时进行进一步调整。

![](images/81b83919f51e27cae4e005cc0a3e72825b79a9b924c0a7bc4072164a4d8756c4.jpg)  
Fig. 1. 使用基于运动学或机器学习模型识别步态事件的示意图。这些模型的输入包括使用 3D 动作捕捉系统采集并后处理的足尖、足跟或骨盆的标记点轨迹。

TABLE I  
用于检测 HS 和 TO 的基于运动学的方法

<table><tr><td>方法</td><td>标记点</td><td>触地（HS）检测特征</td><td>离地（TO）检测特征</td></tr><tr><td>相对于骨盆的足部轨迹 (Zeni et al. 2008)</td><td>脚跟、脚尖和骨盆</td><td>脚跟标记点相对于骨盆水平位置的局部极大值</td><td>脚尖标记点相对于骨盆水平位置的局部极小值</td></tr><tr><td>高通滤波足部标记点 (Desailly et al. 2009)</td><td>脚跟和脚尖</td><td>经高通滤波（0.5 * 步态频率）后的脚跟标记点水平轨迹中的首个极大值</td><td>经高通滤波（1.1 * 步态频率）后的脚尖标记点水平轨迹中的最后一个极小值</td></tr><tr><td>足部标记点速度极值 (O&#x27;Connor et al. 2007)</td><td>脚跟和脚尖</td><td>脚跟与脚尖标记点中点垂直速度的局部极小值</td><td>脚跟与脚尖标记点中点垂直速度的局部极大值</td></tr><tr><td>足部标记点速度阈值 (Ghoussayni et al. 2004)</td><td>脚跟和脚尖</td><td>脚跟标记点的矢状面速度低于 50 cm/s</td><td>脚尖标记点的矢状面速度高于 50 cm/s</td></tr><tr><td>足部标记点加速度极值 (Hreljac et al. 2000)</td><td>脚跟和脚尖</td><td>脚跟标记点垂直加速度的局部极大值，约束条件为加速度的导数（加加速度，jerk）为零</td><td>脚尖标记点水平加速度的局部极大值，同样约束相关加加速度为零</td></tr><tr><td>足部标记点加速度极值 (Hsue et al. 2007)</td><td>脚跟和脚尖</td><td>脚跟标记点水平加速度的局部极小值</td><td>脚尖标记点水平加速度的局部极大值</td></tr><tr><td>位置与速度结合 (Bonci et al. 2022)</td><td>脚跟、脚尖和骨盆</td><td>初步通过 Zeni et al. (2008) 的方法检测，随后基于脚跟标记点的 3D 速度进行优化修正：后足着地时低于 0.5 * 步行速度，前足着地时低于 0.8 * 步行速度</td><td>初步通过 Zeni et al. 的方法检测，当脚尖标记点的 3D 速度超过 0.8 倍步行速度时进行优化修正，并在脚跟标记点的 3D 速度在其局部峰值后下降时进一步优化修正</td></tr></table>

## (b) LSTM 网络 (LSTM Networks)

开发了 LSTM 模型，利用骨盆、足跟和足趾标记点的位置与速度来识别步态事件。这些标记点包括放置在左髂前上棘（LASIS）、左髂后上棘（LPSIS）、左足跟（LFCC）、左第二跖骨头（LFMT2）、右髂前上棘（RASIS）、右髂后上棘（RPSIS）、右足跟（RFCC）和右第二跖骨头（RFMT2）上的标记点。此外，足跟相对于骨盆的位置也被用作输入。该模型首先使用一个掩码层（Masking layer）以忽略输入中的填充值（零）。它包含一个具有 128 个单元和 30% 丢弃率（dropout rate）的单层 LSTM，随后是一个具有 32 个神经元并使用 ReLU 激活函数的全连接层（Dense layer），以捕获高级特征表示。最后，一个采用 softmax 激活函数的时间分布全连接层（time-distributed Dense layer）输出五个节点上的概率，分别对应无步态事件、左脚 HS/TO 以及右脚 HS/TO 事件。如 Lempereur 等人 [16] 所提出的，采用阈值为 0.01 的简单峰值检测算法来检测步态事件。

超参数优化采用 Keras Tuner 的 RandomSearch 方法进行。搜索空间包括关键参数，如 LSTM 单元数（32–128，步长为 32）、丢弃率（0.2–0.5，步长为 0.1）、批次大小（batch size，16、32、64、128）以及优化器类型（Adam 或 RMSprop）。共评估了 20 次试验，每次测试一组独特的超参数组合，并以验证集准确率作为优化指标。应用了早停机制（early stopping mechanism），以确保高效调优且不发生过拟合。完成搜索后，使用性能最佳的超参数来构建和训练最终模型。

![](images/43a85911990a609b3dc6695dcde6539c565366059811edd049153fc7655ddca9.jpg)

![](images/68ee84483a0cd5c49ade2bd95408be5d66b339e60283a60da684628274d93c6c.jpg)  
Fig. 2. 基于各种运动学方法和 LSTM 模型测得与计算出的足跟触地（左）和足趾离地（右）事件之间预测误差（时间差）的箱线图。

## C. 评估方案

通过计算基于运动学和基于机器学习的模型所估计的事件时间与实际事件时间之间的时间误差（时间差），来评估步态事件识别的准确性。分析所得差异以评估每种方法在检测关键步态事件方面的性能。

TABLE II  
基于运动学的方法与 LSTM 模型在足跟触地和足趾离地事件中时间差的均值和标准差（STD）。

<table><tr><td rowspan="2">方法</td><td colspan="2">足跟着地（Heel Strike）</td><td colspan="2">足尖离地（Toe off）</td></tr><tr><td>均值 (ms)</td><td>标准差 (ms)</td><td>均值 (ms)</td><td>标准差 (ms)</td></tr><tr><td>Zeni et al.</td><td>4.78</td><td>9.56</td><td>-8.66</td><td>6.05</td></tr><tr><td>Desailly et al.</td><td>22.17</td><td>8.66</td><td>7.45</td><td>7.62</td></tr><tr><td>O&#x27;Connor et al.</td><td>46.73</td><td>7.72</td><td>-16.27</td><td>8.35</td></tr><tr><td>Ghoussayni et al.</td><td>50.49</td><td>11.28</td><td>-61.96</td><td>8.67</td></tr><tr><td>Hreljac et al.</td><td>32.28</td><td>31.56</td><td>4.70</td><td>8.14</td></tr><tr><td>Hsue et al.</td><td>-11.86</td><td>11.50</td><td>4.70</td><td>8.14</td></tr><tr><td>Bonci et al.</td><td>36.31</td><td>9.15</td><td>-16.63</td><td>10.34</td></tr><tr><td>LSTM</td><td>-0.72</td><td>9.56</td><td>-6.52</td><td>10.34</td></tr></table>

## III. 结果

在基于运动学的方法中，Zeni et al. 方法在足跟着地检测上的时间差最小 $( 4 . 7 8 \pm 9 . 5 6 ~ \mathrm { m s } ,$ ，表 II），最接近零时间差线，其次是 Hsue et al. 和 Desailly et al. 的方法（图 2 和图 3）。相比之下，Desailly et al.、O’Connor et al.、Ghoussayni et al. 以及 Bonci et al. 的方法检测到的足跟着地时间晚于实测值，而 Hreljac et al. 的方法则同时出现了提前和滞后检测。Hreljac et al. 方法的变异性最高（31.56 ms），表明在不同试验中的一致性较差（图 2）。

对于足尖离地检测，Desailly et al.、Hreljac et al. 和 Bonci et al. 的方法的时间差接近于零，而 Ghoussayni et al. 方法检测到的足尖离地事件明显提前 $( - 6 1 . 9 6 \pm 8 . 6 7 ~ \mathrm { m s } )$ ，表明存在偏向提前检测的系统偏差。O’Connor et al. 和 Bonci et al. 也表现出提前检测的趋势，但偏差较小（分别为 -16.27 ms 和 -16.63 ms）。各方法之间的变异性也十分明显，部分方法（如 O’Connor et al.）表现出较低的变异性（8.35 ms），而其他方法（如 Hreljac et al.）在足跟着地检测中表现出更宽的离散度。

基于 LSTM 的方法在足跟着地 $( - 0 . 7 2 \pm 9 . 5 6$ ms) 和足尖离地 $( - 6 . 5 2 \pm 1 0 . 3 4 ~ \mathrm { m s } )$ 检测中均表现出较低的平均时间差，其数值集中在零附近。然而，其性能与 Zeni et al. 相当，后者在足跟着地检测中也显示出相对较小的偏差。

## IV. 讨论

在本研究中，我们评估了使用七种基于运动学的计算模型和一种 LSTM（长短期记忆网络）模型进行的步态事件检测。我们的结果表明，Zeni et al. 方法在基于运动学的方法中达到了最高的准确度，而其他方法则表现出系统性偏差或需要针对不同数据集进行特定调整。LSTM 模型的表现与 Zeni et al. 方法相似，提供了一种避免系统性偏差的数据驱动型替代方案。这些发现对步态分析研究以及辅助外骨骼等应用具有潜在意义，在这些应用中，准确识别支撑相和摆动相对于在各种环境中控制算法至关重要。通过提高步态事件检测的准确性和鲁棒性，这些进展可能在支持康复工作、改进临床评估以及推进辅助技术的发展方面发挥至关重要的作用。

![](images/c9e98e40f2273c24ef151ce2c808b5aa54768b407f4efb9a0ad6833f07b27f7d.jpg)  
Fig. 3. 使用各种基于运动学的方法以及 LSTM 模型测得与计算得出的脚跟着地（左）和足尖离地（右）事件之间时间差的直方图。

我们的结果表明，尽管某些基于运动学的方法（例如 Zeni et al.）提供了相对准确的步态事件检测，但其他方法则表现出系统性偏移或更大的变异性。基于运动学的方法广泛用于步态分析，尤其是在测力台不可用的情况下。在评估的方法中，Zeni et al. 在脚跟着地方面表现出最小的时间差 $( 4 . 7 8 ~ \pm ~ 9 . 5 6$ ms)，这可能是由于其参考数据集与我们的研究人群相吻合。基于速度阈值的方法，例如 O’Connor et al. 和 Ghoussayni et al. 的方法，需要针对不同数据集仔细调整阈值以保持准确度。基于加速度的方法，包括 Hreljac et al. 和 Hsue et al. 的方法，对水平分量表现出良好的预测准确度，但对垂直分量可靠性较差，后者似乎受到了个体行走模式的影响。有趣的是，Bonci et al. 提出的基于位置和速度的方法在我们的研究中并没有达到最高准确度。这可能是由于其最初验证中使用的数据集所致。Bonci et al. 最初使用来自健康年轻成年人（n = 20）的数据对比现有算法测试了其方法，随后在一个由来自五组人群（老年人、慢性阻塞性肺疾病患者、多发性硬化症患者、帕金森病患者以及股骨近端骨折患者）的 10 名受试者组成的小队列中对其进行了评估。相比之下，我们的研究仅包含常模数据，并证实了 Zeni et al. 的方法无需进行大量针对特定数据集的调整即可达到很高的准确度。

LSTM 模型表现出了令人鼓舞的结果，具有较低的平均时间差且无系统偏差。其性能与 Zeni et al. 的方法相当，表明基于深度学习的方法可以作为传统基于规则方法的有效替代方案。与依赖预定义阈值的传统方法不同，LSTM 模型利用其学习步态运动学中时间依赖性的能力，捕捉运动阶段之间的复杂关系 [27]–[29]。然而，尽管 LSTM 模型在同一数据集内的未见数据上表现良好，但其在更广泛人群（尤其是具有病理性步态模式的人群）中的泛化能力仍不确定。深度学习模型通常在与其训练数据相似的数据集上评估时表现出色，仍需进一步研究以评估它们在多样化运动模式下的鲁棒性。将评估扩展到步态受损个体，将为基于 LSTM 的步态事件检测的适应性和潜在临床应用价值提供有价值的见解。

本研究存在若干局限性。评估仅在标准数据集上进行，限制了其对步态模式改变人群的适用性。未来的工作应纳入患有膝骨关节炎或中风后功能障碍等病症的个体，以评估所测试方法的泛化能力。此外，研究仅评估了一种机器学习模型（LSTM）。探索其他深度学习架构（例如 Transformer）可以为模型在不同数据集上的适应性提供进一步的见解。最后，利用来自多个研究机构的数据进行跨中心验证，将有助于确定这些方法在不同临床和现实世界环境中的可重复性与鲁棒性。

## V. 结论

本研究评估了七种基于运动学的方法和一种基于 LSTM 的方法在步态事件检测中的性能。尽管一些基于运动学的方法（特别是 Zeni et al.）展现出极高的准确性，但其他方法表现出系统性偏差或需要针对特定数据集进行调优。LSTM 模型展现出良好的前景，其性能与最准确的基于运动学的方法相当，且无系统偏差。然而，其对步态受损个体的适用性仍有待探索。未来的研究应将这些评估拓展至临床人群，并探索其他机器学习技术（如 Transformer 模型），以增强在多样化步态条件和数据采集环境下的鲁棒性与泛化能力。

## 参考文献

[1] Z. Li, B. Huang, Z. Ye, M. Deng, and C. Yang, “Physical human– robot interaction of a robotic exoskeleton by admittance control,” IEEE Transactions on Industrial Electronics, vol. 65, no. 12, pp. 9614–9624, 2018.

[2] Z. Li, B. Huang, A. Ajoudani, C. Yang, C.-Y. Su, and A. Bicchi, “Asymmetric bimanual control of dual-arm exoskeletons for humancooperative manipulations,” IEEE Transactions on Robotics, vol. 34, no. 1, pp. 264–271, 2017.

[3] L. Zhang, X. Zhu, E. M. Gutierrez-Farewik, and R. Wang, “Ankle joint torque prediction using an nms solver informed-ann model and transfer learning,” IEEE Journal ofBiomedical and Health Informatics, vol. 26, no. 12, pp. 5895–5906, 2022.

[4] Z. Li, Y. Yuan, L. Luo, W. Su, K. Zhao, C. Xu, J. Huang, and M. Pi, “Hybrid brain/muscle signals powered wearable walking exoskeleton enhancing motor ability in climbing stairs activity,” IEEE Transactions on Medical Robotics and Bionics, vol. 1, no. 4, pp. 218–227, 2019.

[5] L. Luo, M. J. Foo, M. Ramanathan, J. K. Er, C. H. Chiam, L. Li, W. Y. Yau, and W. T. Ang, “Trajectory generation and control of a lower limb exoskeleton for gait assistance,” Journal of Intelligent & Robotic Systems, vol. 106, no. 3, p. 64, 2022.

[6] Y. K. Kim, R. M. Visscher, E. Viehweger, N. B. Singh, W. R. Taylor, and F. Vogl, “A deep-learning approach for automatically detecting gait-events based on foot-marker kinematics in children with cerebral palsy—which markers work best for which gait patterns?” Plos One, vol. 17, no. 10, pp. 1–16, 2022.

[7] B. Dumphart, D. Slijepcevic, M. Zeppelzauer, A. Kranzl, F. Unglaube, A. Baca, and B. Horsak, “Robust deep learning-based gait event detection across various pathologies,” Plos One, vol. 18, no. 8, pp. 1–16, 2023.

[8] J. Zeni Jr, J. Richards, and J. Higginson, “Two simple methods for determining gait events during treadmill and overground walking using kinematic data,” Gait & Posture, vol. 27, no. 4, pp. 710–714, 2008.

[9] R. Baker and H. M. Hart, Measuring walking: a handbook of clinical gait analysis. Mac Keith Press London, 2013, vol. 1.

[10] S. R. Simon, “Quantification of human motion: gait analysis—benefits and limitations to its application to clinical problems,” Journal of Biomechanics, vol. 37, no. 12, pp. 1869–1880, 2004.

[11] D. H. Sutherland, “The evolution of clinical gait analysis: Part ii kinematics,” Gait & Posture, vol. 16, no. 2, pp. 159–179, 2002.

[12] L. Zhang, Y. Liu, R. Wang, C. Smith, and E. M. Gutierrez-Farewik, “Modeling and simulation of a human knee exoskeleton’s assistive strategies and interaction,” Frontiers in Neurorobotics, vol. 15, pp. 1– 12, 2021.

[13] N. Zahradka, K. Verma, A. Behboodi, B. Bodt, H. Wright, and S. C. Lee, “An evaluation of three kinematic methods for gait event detection compared to the kinetic-based ‘gold standard’,” Sensors, vol. 20, no. 18, pp. 1–15, 2020.

[14] O. Tirosh and W. Sparrow, “Identifying heel contact and toe-off using forceplate thresholds with a range of digital-filter cutoff frequencies,” Journal of Applied Biomechanics, vol. 19, no. 2, pp. 178–184, 2003.

[15] D. A. Bruening and S. T. Ridge, “Automated event detection algorithms in pathological gait,” Gait & Posture, vol. 39, no. 1, pp. 472– 477, 2014.

[16] M. Lempereur, F. Rousseau, O. Remy-Néris, C. Pons, L. Houx, G. Quellec, and S. Brochard, “A new deep learning-based method for the detection of gait events in children with gait disorders: Proofof-concept and concurrent validity,” Journal of Biomechanics, vol. 98, pp. 1–4, 2020.

[17] T. Bonci, F. Salis, K. Scott, L. Alcock, C. Becker, S. Bertuletti, E. Buckley, M. Caruso, A. Cereatti, S. Del Din et al., “An algorithm for accurate marker-based gait event detection in healthy and pathological populations during complex motor tasks,” Frontiers in Bioengineering and Biotechnology, vol. 10, pp. 1–14, 2022.

[18] A. H. Hansen, D. S. Childress, and M. R. Meier, “A simple method for determination of gait events,” Journal of Biomechanics, vol. 35, no. 1, pp. 135–138, 2002.

[19] E. Desailly, Y. Daniel, P. Sardain, and P. Lacouture, “Foot contact event detection using kinematic data in cerebral palsy children and normal adults gait,” Gait & Posture, vol. 29, no. 1, pp. 76–80, 2009.

[20] S. Ghoussayni, C. Stevens, S. Durham, and D. Ewins, “Assessment and validation of a simple automated method for the detection of gait events and intervals,” Gait & Posture, vol. 20, no. 3, pp. 266–272, 2004.

[21] C. M. O’Connor, S. K. Thorpe, M. J. O’Malley, and C. L. Vaughan, “Automatic detection of gait events using kinematic data,” Gait & Posture, vol. 25, no. 3, pp. 469–474, 2007.

[22] B. Hsue, F. Miller, F. Su, J. Henley, and C. Church, “Gait timing event determination using kinematic data for the toe walking children with cerebreal palsy,” Journal of Biomechanics, vol. 40, no. 2, p. S529, 2007.

[23] A. Hreljac and R. N. Marshall, “Algorithms to determine event timing during normal walking using kinematic data,” Journal of Biomechanics, vol. 33, no. 6, pp. 783–786, 2000.

[24] M. A. French, C. Koller, and E. S. Arch, “Comparison of three kinematic gait event detection methods during overground and treadmill walking for individuals post stroke,” Journal of Biomechanics, vol. 99, pp. 1–7, 2020.

[25] C. Voisard, N. de l’Escalopier, D. Ricard, and L. Oudre, “Automatic gait events detection with inertial measurement units: healthy subjects and moderate to severe impaired patients,” Journal of NeuroEngineering and Rehabilitation, vol. 21, no. 1, pp. 1–17, 2024.

[26] B. Filtjens, A. Nieuwboer, N. D’cruz, J. Spildooren, P. Slaets, and B. Vanrumste, “A data-driven approach for detecting gait events during turning in people with parkinson’s disease and freezing of gait,” Gait & Posture, vol. 80, pp. 130–136, 2020.

[27] L. Zhang, D. Soselia, R. Wang, and E. M. Gutierrez-Farewik, “Lowerlimb joint torque prediction using lstm neural networks and transfer learning,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 30, pp. 600–609, 2022.

[28] Z. Xia, B. M. Cornish, D. Devaprakash, R. S. Barrett, D. G. Lloyd, A. H. Hams, and C. Pizzolato, “基于无标记视频预测常见运动任务中的跟腱力,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2024.

[29] L. Xiang, Y. Gu, Z. Gao, P. Yu, V. Shim, A. Wang, and J. Fernandez, “结合 LSTM 框架利用惯性传感器预测步态期间的踝关节生物力学,” Computers in Biology and Medicine, vol. 170, p. 108016, 2024.