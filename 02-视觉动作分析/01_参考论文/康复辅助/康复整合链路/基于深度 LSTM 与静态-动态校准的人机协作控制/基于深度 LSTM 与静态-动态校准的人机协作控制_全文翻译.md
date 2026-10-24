# 基于深度 LSTM 与静态-动态校准的人机协作控制

Chenglin Xie，IEEE 学生会员，Zhijun Li，IEEE 会士，以及 Rong Song，IEEE 高级会员

摘要—利用肌电（electromyography, EMG）信号估计人体三维（3-D）上肢运动意图是人机协作系统中一项极具前景的技术。由于三维上肢神经肌骨系统内部的非线性成分以及人机耦合系统中错综复杂的动态在线环境，该技术极具挑战性。在本文中，提出了一种深度长短期记忆（deep long short-term memory, Deep-LSTM）网络，用于为上肢绳驱机器人（upper limb cable-driven robot, UL-CDR）建立非线性肌电-三维力（EMG-to-3-D-force）模型。为了增强模型对动态环境的适应性，我们提出了一种静态-动态校准方法，该方法首先利用静态测量数据训练网络，随后利用动态数据对其进行微调。招募了受试者在 UL-CDR 的辅助下进行 160 次跟踪参考轨迹试验。结果表明，与仅进行静态校准的方法相比，所提出的方法获得了更好的跟踪精度，证明了该方法在静态和动态环境之间过渡的有效性，并具有提升人机协作性能的潜力。

手稿收稿日期为 2023 年 12 月 8 日；修回日期为 2024 年 2 月 24 日和 2024 年 4 月 9 日；录用日期为 2024 年 4 月 26 日。出版日期为 2024 年 6 月 6 日；当前版本日期为 2024 年 10 月 11 日。本工作部分受到中国国家重点研发计划（项目资助号 2022YFE0201900 和 2021YFF0501600）、ANSO 合作研究项目（项目资助号 ANSO-CR-PP-2020-03）、广东省科技计划项目（项目资助号 2020B1212060077）以及深圳市科技计划项目（项目资助号 SGDX20210823103405040）的资助。（通信作者：Rong Song。）

Chenglin Xie 和 Rong Song 供职于 Key Laboratory of Sensing Technology and Biomedical Instrument of Guangdong Province, School of Biomedical Engineering, Shenzhen Campus of Sun Yat-sen University, Shenzhen 518107, China（中山大学深圳校区生物医学工程学院广东省传感技术与生物医疗仪器重点实验室），亦供职于 Guangdong Provincial Engineering and Technology Center of Advanced and Portable Medical Devices, School of Biomedical Engineering, Sun Yat-sen University, Guangzhou, Guangdong 510006, China（中山大学生物医学工程学院广东省先进与便携式医疗器械工程技术研究中心），以及 Shenzhen Research Institute, Sun Yat-sen University, Shenzhen, Guangdong 510810, China（中山大学深圳研究院）（电子邮箱：xiechlin@mail2.sysu.edu.cn；songrong@mail.sysu.edu.cn）。

Zhijun Li 供职于 School of Mechanical Engineering, Translational Research Center, and Shanghai Yangzhi Rehabilitation Hospital, Tongji University, Shanghai 200070, China（同济大学机械工程学院、转化医学研究中心及上海市养志康复医院）（电子邮箱：zjli@ieee.org）。

本文中一个或多个插图的彩色版本可访问 https://doi.org/10.1109/TIE.2024.3398675 获取。

数字对象唯一标识符（Digital Object Identifier）10.1109/TIE.2024.3398675

索引词——深度长短期记忆网络（DLN）、静态-动态校准、人体运动意图、人机协作系统、肌电控制。

## I. 引言

人机协作（human–robot collaboration）是一个快速发展的领域，彻底改变了人类与机器人互动及协作的方式。人机协作中的一个关键要素是机器人理解并响应人体运动意图的能力 [1]。通过准确且连续地感知人体运动意图，机器人可以与人类实现无缝协作。与人类并肩工作的机器人能够利用人类导师的技能，从而提高灵活性、安全性和生产力 [2]。人机协作的潜力和重要性使其在制造业 [2]、假肢 [3]、辅助技术 [4] 以及康复 [5], [6] 等诸多领域得到了广泛应用。

由肌细胞产生的肌电（EMG）信号与人体运动意图密切相关，因此可应用于上述人机协作系统中 [2], [3], [7], [8]。在过去的几十年中，研究人员对 EMG 与人体运动/力之间的关系进行了深入探索。众多学者致力于利用支持向量机（SVM）[9]、线性判别分析（LDA）[10] 和深度学习（DL）[11] 等 EMG 模式识别技术进行离散运动控制。连续运动控制是肌电控制中的另一个热点问题 [1]。基于 Hill 的肌肉骨骼模型已被广泛应用于机器人控制，以利用 EMG 信号连续获取人体运动意图 [12], [13]。然而，该模型实际上局限于低维力或关节力矩的估计 [14]，且由于所涉及运动的复杂性，在估计复杂的上肢三维末端力时面临挑战 [15], [16]。先前的研究探讨了使用线性状态空间模型（SSM）连续估计上肢三维末端力作为上肢康复机器人的输入，并完成伸手抓取等简单人机协作任务的可能性 [17]。

为了在更复杂的人机协作（human–robot collaboration）任务中实现肌电控制，研究人员探索了具有非线性拟合能力的新模型。Aung 和 Al-Jumaily 建立了一个反向传播神经网络（back propagation neural network, BPNN），利用肌电（EMG）信号来预测上肢关节角度 [18]。Zhang 等人训练了一个人工神经网络（artificial neural network, ANN）来构建 EMG 到关节角度的模型 [19]。Ding 等人提出了一种非线性状态空间模型来描述 EMG 与关节角速度之间的关系，其中状态模型为自回归网络，测量模型为 BPNN [16]。近年来，深度学习（DL）因其在处理复杂非线性问题方面的优势以及相比传统机器学习方法具备更好的泛化能力而受到越来越多的关注 [20]。Nahid 等人采用 AlexNet 对手势进行分类，并将其应用于假肢手 [11]。Ozdemir 等人研究了多种深度网络，发现 ResNet-50 获得了最高的准确率 [21]。Nasr 等人探索了包括 CNN、RNN、ANN 和 RCNN 在内的深度网络，以映射 EMG 与关节角度、速度、加速度及力矩之间的关系 [22]。长短期记忆网络（Long short-term memory network, LSTM）分别利用隐藏状态和记忆状态来形成短期和长期记忆 [23]，因此 LSTM 可能是处理 EMG 等时序数据的潜在有效模型 [24]。在我们之前的会议论文中，建立了一种基于深度 LSTM 网络（deep LSTM network, DLN）的 EMG 到三维力模型，且该模型仅使用静态测量数据进行训练 [25]。然而，当将其应用于人机协作系统以跟踪参考轨迹时，尽管在静态测量数据集中的估计精度足够高，但在动态环境中仍观察到了不可忽略的跟踪误差。这一现象表明，仅使用静态测量数据训练的模型无法有效适应在线动态环境。

尽管肌电控制领域已经取得了进展，但现有的非线性肌电-三维力（EMG-to-3-D-force）研究通常是在静态恒定姿态约束下 [25], [26], [27], [28] 或未与机器人交互的离线数据集 [16], [18], [19], [22] 中进行校准/训练的，这在应用于人机耦合系统中的动态在线环境时可能会导致次优性能。本研究提出了一种静-动态校准方法，该方法首先利用静态测量数据训练基于 DLN 的三维末端力估计模型，然后利用动态数据对网络进行更新。训练/更新后的 DLN 模型与导纳控制器相结合，为上肢绳驱动机器人（UL-CDR）生成期望轨迹。整个控制方案通过多速率并行计算解决方案构建在名为 dSpace 的机器人控制平台中，从而最大化了系统的高频实时性能。招募了三名受试者，通过其肌电（EMG）信号控制 UL-CDR 跟踪矩形和圆形轨迹。将所提出方法的跟踪误差与在动态环境中未更新 DLN 模型的方法进行了比较。我们假设，采用静-动态校准的 DLN 能够适应从静态到动态环境的转变，并产生比仅采用静态校准更好的轨迹跟踪性能。

本文的主要创新点在于：1）构建了一个 DLN 网络以拟合肌电、关节角和三维力之间的关系，并将其嵌入硬件中以实现人机协作中的高频实时肌电控制；2）证明了使用静态数据集训练 DLN 无法完美应用于动态人机交互，而解决该问题需要所提出的静-动态校准方法。

![](images/c507e4fb0c43aa5b8bb5a3705c56d2d1290a8741b61fa37716a3146b2bbbe8ac.jpg)  
(a)

![](images/cb15b05c79cfd1c0cfb2d076ba1bd1b4795b9600bf65b895b4d1dc2c4fa509aa.jpg)  
(b)

![](images/684a8762723e9278130d1f432e97c504233a5dbb49d4e54974a9a8fd4909c6bb.jpg)  
(c)  
Fig. 1. 实验原型。(a) UL-CDR 的机械结构。(b) 附着在八块选定肌肉皮肤表面的肌电电极。(c) 带有屏幕视觉反馈的 UL-CDR 图像。

## II. 基于 DLN 控制器的 UL-CDR

实验原型如图 1 所示。UL-CDR 包括四台电机（DM1B045G, Yokogawa）、四根绳索、一个带有六维力传感器（D45MM-F400N, Sunrise Instruments）的末端执行器（end-effector, EE）、一个带有四个摄像头的动作捕捉系统（OptiTrack2, Natural-Point）、一个实时平台（RTI1202, dSpace，配备 2 GHz Qorl0 P5020 处理器），以及一个为轨迹跟踪提供视觉反馈的屏幕。与受试者手部进行交互的末端执行器由四根绳索驱动，每根绳索由一台电机驱动。肘部、肩部和末端执行器上固定有五个光学标记点。动作捕捉系统用于获取光学标记点的实时位置。

UL-CDR 的控制框图如图 2 所示。构建了一个 DLN，根据 8 个通道的肌肉激活度和 3 个关节角信息来估计三维人体主动发力 $F _ { h }$。导纳滤波器根据 $F _ { h }$ 生成末端执行器的初步期望位置 $P _ { d }$。为了安全起见，设计了一个工作空间限制模块来对 $P _ { d } ,$ 进行约束，形成期望位置 $\boldsymbol { P } _ { d } ^ { * }$。采用绳索轨迹规划器将任务空间中的 $\boldsymbol { P } _ { d } ^ { * }$ 转换为关节空间中四根绳索的期望长度向量 $P _ { l d }$。PID 控制器与绳索张力控制器共同确定四台电机的控制信号，驱动末端执行器跟踪期望位置 $\boldsymbol { P } _ { d } ^ { * }$。整个控制系统以多速率并行计算模式运行，从而最大化高频实时性能。肌肉激活度计算、PID 控制器和绳索张力控制器的工作控制频率为 1000 Hz，而其他组件（如计算密集型的 DLN）则在 100 Hz 下运行。

![](images/775d172fb4673c20d898122e683b8cc6e156bd6273b63b6e7cec10d6301ff8e6.jpg)  
Fig. 2. UL-CDR 总体控制框图。

## A. 肌肉激活度计算

每个选定肌肉的表面肌电（sEMG）信号[如图 1(b) 所示]均通过 Ag-AgCl 电极获取，电极根据 SENIAM [29], [30] 的推荐贴附在皮肤表面。肌电信号的采样频率为 1 kHz。然后，肌电信号通过放大器放大 5000 倍，该放大器还包含一个内置陷波滤波器（50 Hz）和一个带通滤波器（20–450 Hz）。滤波后的信号经过全波整流，并通过截止频率为 4-Hz 的四阶低通巴特沃斯（Butterworth）滤波器进行处理 [14]（上述 sEMG 处理的功能见 [31]）。对于第 k 个 $( k = 1 , 2 , \ldots , 8 )$ 肌肉，使用最大自主等长收缩（MVC）方法 [32], [33] 将处理后的信号归一化为 $X _ { k }$。接下来，根据二阶线性模型获得神经激活度 $N _ { k }$

$$
N _ {k} (t) = \alpha X _ {k} (t - t _ {d}) - \beta_ {1} N _ {k} (t - 1) - \beta_ {2} N _ {k} (t - 2)\tag{1}
$$

其中 $\beta _ { 1 } = - 0 . 0 5 2 0 , \beta _ { 2 } = 0 . 0 0 0 6 , \alpha = 0 . 9 4 8 6$，机电时间延迟 $t _ { d } = 1 0 0$ 根据先前的研究 [17] 确定。

最后，根据以下模型计算肌肉激活度 $a _ { k } ( t )$ [34]：

$$
a _ {k} (t) = \left\{ \begin{array}{l l} \tau \mathrm{ln} (\omega N _ {k} (t) + 1), & 0 \leq N _ {k} (t) <   N _ {0} \\ \varphi N _ {k} (t) + \delta , & N _ {0} \leq N _ {k} (t) \leq 1 \end{array} \right.\tag{2}
$$

其中 $\tau = 0 . 2 8 8 2 , \omega = 1 0 . 4 7 6 0 , \varphi = 0 . 8 3 9 1 , \delta = 0 . 1 6 0 9$ 以及 $N _ { 0 } = 0 . 2 4 8 0$ 是通过单参数法结合牛顿-拉弗森法（Newton–Raphson method）获得的 [17], [34]。

## B. 关节角度计算

五个光学标记点被固定在上肢皮肤和末端执行器（EE）上。两个标记点附着于肘部 $( E ( x _ { E } , y _ { E } , z _ { E } ) )$ 和肩部 $( S ( x _ { S } , y _ { S } , z _ { S } ) )$，而另外三个标记点固定在 EE 上。EE 上标记点的中心位置代表受试者手部的位置 $( H ( x _ { H } , y _ { H } , z _ { H } ) )$，这也代表了 EE 的位置。这些位置可用于计算以下关节角度：

$$
\theta_ {\mathrm {S\_ {E} F}} = \arctan 2 (z _ {S} - z _ {E}, y _ {S} - y _ {E}).\tag{3}
$$

$$
\theta_ {S \_ A A} = \arctan 2 (x _ {S} - x _ {E}, y _ {S} - y _ {E}).\tag{4}
$$

$$
\theta_ {\mathrm {E\_EF}} = \arccos \left(\frac {H E ^ {2} + E S ^ {2} - H S ^ {2}}{2 H E E S}\right).\tag{5}
$$

其中 $\theta _ { \mathrm { S \_ E F } } , \quad \theta _ { S \_ A A }$ 和 $\theta _ { \mathrm { E } \_ \mathrm { E F } }$ 分别表示肩部屈伸角、肩部外展-内收角和肘部屈伸角；k∙k 代表欧几里得距离。

## C. DLN

在时刻 $t$，DLN 的输入向量 $U ( t )$ 包含三个关节角度和八个通道的肌肉激活度，而输出向量 $F _ { h } ( t )$ 为估计的手部三维自主末端力

$$
\boldsymbol {U} (t) = [ \theta_ {\mathrm {S\_EF}} (t); \theta_ {S \_ A A} (t); \theta_ {\mathrm {E\_EF}} (t); a _ {1} (t);...; a _ {8} (t) ]\tag{6}
$$

$$
\boldsymbol {F} _ {h} (t) = \left[ F _ {h \_ x} (t); F _ {h \_ y} (t); F _ {h \_ z} (t) \right].\tag{7}
$$

DLN 原理图如图 3 所示。为了处理十一维序列输入 U，采用了序列输入层。随后级联了三层 LSTM。每个 LSTM 层后接一个概率为 0.50 的 dropout 层。接着是一个包含 200 个神经元的全连接层。之后，应用了一个回归层用于输出估计的三维力 $F _ { h } .$。对于每个 LSTM 层，构建了 200 个单元。每个单元由一个输入门 $i ,$、一个遗忘门 $f ,$、一个候选值 ${ \tilde { c } } _ { \cdot }$、一个记忆状态 $^ { c , }$、一个输出门 $^ { \bullet , }$ 以及一个隐藏状态 H 组成，公式如下：

$$
\boldsymbol {i} (t) = \sigma (\boldsymbol {Q} _ {i} \boldsymbol {U} (t) + \boldsymbol {R} _ {i} \boldsymbol {H} (t - 1) + \varepsilon_ {i})\tag{8}
$$

![](images/8d9e123ae6e08017029fc5e3329d71d9c5d0335a1ed652c32e98fc22570514e0.jpg)  
Fig. 3. DLN 的架构。

$$
\boldsymbol {f} (t) = \sigma (\boldsymbol {Q} _ {f} \boldsymbol {U} (t) + \boldsymbol {R} _ {f} \boldsymbol {H} (t - 1) + \boldsymbol {\varepsilon} _ {f})\tag{9}
$$

$$
\tilde {\boldsymbol {C}} (t) = \operatorname{Tanh} \left(\boldsymbol {Q} _ {\tilde {\boldsymbol {C}}} \boldsymbol {U} (t) + \boldsymbol {R} _ {\tilde {\boldsymbol {C}}} \boldsymbol {H} (t - 1) + \varepsilon_ {\tilde {\boldsymbol {C}}}\right)\tag{10}
$$

$$
\boldsymbol {C} (t) = \boldsymbol {f} (t) \odot \boldsymbol {C} (t - 1) + \boldsymbol {i} (t) \odot \tilde {\boldsymbol {C}} (t)\tag{11}
$$

$$
\boldsymbol {o} (t) = \sigma (\boldsymbol {Q} _ {o} \boldsymbol {U} (t) + \boldsymbol {R} _ {o} \boldsymbol {H} (t - 1) + \boldsymbol {\varepsilon} _ {o})\tag{12}
$$

$$
\boldsymbol {H} (t) = \boldsymbol {o} (t) \odot \operatorname{Tanh} (\boldsymbol {C} (t))\tag{13}
$$

其中 $Q _ { i } , Q _ { f } , Q _ { \tilde { C } } , Q _ { o } , R _ { i } , R _ { f } , R _ { \tilde { C } }$ 和 $\pmb { R } _ { o }$ 为权重矩阵，$\varepsilon _ { i } , ~ \varepsilon _ { f } , ~ \varepsilon _ { \tilde { C } }$ 和 $\pmb { \varepsilon } _ { o }$ 为偏置。DLN 的详细信息可见 Table I。

## D. 导纳滤波器

采用导纳滤波器根据 DLN 的输出即估计力 $F _ { h }$ 来生成 EE 的初步期望位置 $P _ { d }$

$$
\boldsymbol {P} _ {d} = \boldsymbol {P} + \boldsymbol {F} _ {h} (K + B s + M s ^ {2}) ^ {- 1}\tag{14}
$$

其中 P 为通过四台相机实时测量的实际位置；s 为拉普拉斯算子（Laplacian operator）；K、B 和 M 分别为刚度、阻尼和质量参数。

## E. 工作空间限制

为了防止实验过程中的危险，EE 的位置应被限制在工作空间内。范围

TABLE I DLN 结构参数

<table><tr><td>参数</td><td>LSTM 层 #1</td><td>LSTM 层 #2</td><td>LSTM 层 #3</td></tr><tr><td>输入大小</td><td>11</td><td>200</td><td>200</td></tr><tr><td>隐藏单元数</td><td>200</td><td>200</td><td>200</td></tr><tr><td>输入权重</td><td> $\mathcal{Q}_{i}, \mathcal{Q}_{f}, \mathcal{Q}_{\bar{\mathcal{C}}},$  $\mathcal{Q}_{o} \in \mathbf{R}^{200 \times 11}$ </td><td> $\mathcal{Q}_{i}, \mathcal{Q}_{f}, \mathcal{Q}_{\bar{\mathcal{C}}},$  $\mathcal{Q}_{o} \in \mathbf{R}^{200 \times 200}$ </td><td> $\mathcal{Q}_{i}, \mathcal{Q}_{f}, \mathcal{Q}_{\bar{\mathcal{C}}},$  $\mathcal{Q}_{o} \in \mathbf{R}^{200 \times 200}$ </td></tr><tr><td>循环权重</td><td> $\mathcal{R}_{i}, \mathcal{R}_{f}, \mathcal{R}_{\bar{\mathcal{C}}},$  $\mathcal{R}_{o} \in \mathbf{R}^{200 \times 200}$ </td><td> $\mathcal{R}_{i}, \mathcal{R}_{f}, \mathcal{R}_{\bar{\mathcal{C}}},$  $\mathcal{R}_{o} \in \mathbf{R}^{200 \times 200}$ </td><td> $\mathcal{R}_{i}, \mathcal{R}_{f}, \mathcal{R}_{\bar{\mathcal{C}}},$   $\mathcal{R}_{o} \in \mathbf{R}^{200 \times 200}$ </td></tr><tr><td>偏置</td><td> $\varepsilon_{i}, \varepsilon_{f}, \varepsilon_{\bar{\mathcal{C}}},$  $\varepsilon_{o} \in \mathbf{R}^{200 \times 1}$ </td><td> $\varepsilon_{i}, \varepsilon_{f}, \varepsilon_{\bar{\mathcal{C}}},$  $\varepsilon_{o} \in \mathbf{R}^{200 \times 1}$ </td><td> $\varepsilon_{i}, \varepsilon_{f}, \varepsilon_{\bar{\mathcal{C}}},$  $\varepsilon_{o} = \mathbf{R}^{200 \times 1}$ </td></tr><tr><td>隐藏状态</td><td> $\mathcal{H} \in \mathbf{R}^{200 \times 1}$ </td><td> $\mathcal{H} \in \mathbf{R}^{200 \times 1}$ </td><td> $\mathcal{H} \in \mathbf{R}^{200 \times 1}$ </td></tr><tr><td>记忆状态</td><td> $\mathcal{C} \in \mathbf{R}^{200 \times 1}$ </td><td> $\mathcal{C} \in \mathbf{R}^{200 \times 1}$ </td><td> $\mathcal{C} \in \mathbf{R}^{200 \times 1}$ </td></tr><tr><td>权重数量</td><td>168,800</td><td>320,000</td><td>320,000</td></tr></table>

工作空间被定义为 A

$$
\boldsymbol {A} = \{(x, y, z) \mid \text { distance } ((x, y, z), (x _ {l}, y _ {l}, z _ {l})) \leq r \}\tag{15}
$$

其中 x<sub>l ¼</sub> 0, y<sub>l ¼</sub> 0:99 m, $z _ { l } = - 0 . 7 6$ m, $r = 0 . 2 5$ m，且 distanceð<sub> </sub>, <sub> </sub>Þ 表示两点之间的欧几里得距离。

当超出 A 的范围时，期望位置 $P _ { d } ^ { * }$ 等于同期的实际位置，否则等于导纳滤波器输出的初步期望位置 $P _ { d }$

$$
\boldsymbol {P} _ {d} ^ {*} = \left\{ \begin{array}{c} P, \boldsymbol {P} _ {d} \notin A \\ \boldsymbol {P} _ {d}, \text { otherwise. } \end{array} \right.\tag{16}
$$

## F. 缆索轨迹规划器

为了将任务空间中的期望位置转换为关节空间中上部三根缆索的期望长度，采用了缆索轨迹规划器

$$
\boldsymbol {P} _ {l d} = \left[ P _ {l d \_ 1}, P _ {l d \_ 2}, P _ {l d \_ 3} \right] ^ {T}\tag{17}
$$

$$
P _ {l d \_ j} = \text { distance } ((P _ {d \_ x} ^ {*}, P _ {d \_ y} ^ {*}, P _ {d \_ z} ^ {*}), (c _ {j \_ x}, c _ {j \_ y}, c _ {j \_ z}))\tag{18}
$$

其中 $( c _ { j \_ x } , \ c _ { j \_ y } , \ c _ { j \_ z } )$ 为第 $j$ 个滑轮的位置（参见 $[ 1 7 ]$）；$P _ { d _ { - } x } ^ { * } , P _ { d _ { - } y } ^ { * },$ 以及 $P _ { d _ { - } } ^ { * }$ z 分别表示 $\pmb { P } _ { d } ^ { * }$ 的 $x , y ,$ 和 z 分量。

## G. PID 控制器

上方三根绳索的驱动力通过 PID 控制器确定

$$
\boldsymbol {F} _ {l} = k _ {P} \left(\boldsymbol {P} _ {l d} - \boldsymbol {P} _ {l}\right) + k _ {I} \int_ {1} ^ {t} \left(\boldsymbol {P} _ {l d} - \boldsymbol {P} _ {l}\right) d t + k _ {D} \frac {d \left(\boldsymbol {P} _ {l d} - \boldsymbol {P} _ {l}\right)}{d t}\tag{19}
$$

其中 $F _ { l } = [ F _ { l 1 } , \ F _ { l 2 } , \ F _ { l 3 } ] ^ { T }$ 为驱动力向量；$k _ { P } , k _ { I }$ 和 $k _ { D }$ 为 PID 系数；$P _ { l }$ 表示三根绳索的实际长度。

## H. 绳索张力控制器

下方绳索的驱动力根据实时绳索张力确定，从而保持适当的张力范围以形成四绳全约束并联结构 [17], [35]

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
静态-动态标定算法
1 获取数据集 S：输入：$U \in R^{11 \times T} = [\theta_{joint angle} \in R^{3 \times T}; a_{activation} \in R^{8 \times T}]$；输出：$F_{h} \in R^{3 \times T} = F_{interaction force (static)} \in R^{3 \times T}$ 
2 随机初始化 (8)-(13) 中的偏置和权重矩阵；
3 标准化数据集 S；
4 for  $E \leftarrow 1$  to Max_epochs do
5    for  $I \leftarrow 1$  to [S_Size/batch_size] do
6    $F_{h\_estimate} = forward(U)$ ;
7    $loss \leftarrow 1/M \cdot \Sigma ||F_{h\_target} - F_{h\_estimate}||^2 + 0.0001/N \cdot \Sigma weight^2;$ 
8    使用 ADAM 算法更新网络参数；
9    end for
10    保存网络参数至 $\theta(E)$；
11    保存验证损失至 L(E)；
12 end for
13 根据早停策略选择第 E' 轮；
14 网络参数 $\leftarrow \theta(E')$；
15 将网络嵌入机器人以进行动态标定；
16 更新数据集 S：输入数据：$U \in R^{11 \times T} = [\theta_{joint angle} \in R^{3 \times T}; a_{activation} \in R^{8 \times T}]$；输出数据：$F_{h} \in R^{3 \times T} \leftarrow F_{interaction force (dynamic)} \in R^{3 \times T} + F_{estimated force} \in R^{3 \times T};$ 
17 将预训练的最优参数作为初始参数：$\theta(0) \leftarrow \theta(E')$；
18 重复步骤 3~14 并更新网络；
③ 19 将网络嵌入机器人以进行动态验证实验；
</div>

Fig. 4. 静态-动态标定算法。‹ 静态标定；› 动态标定；以及 fi 动态验证。

$$
F _ {l 4} = \left\{ \begin{array}{l l} - \mu (f _ {1} - f _ {l 4}), & \quad f _ {l 4} <   f _ {1} \\ 0, & \quad f _ {1} \leq f _ {l 4} \leq f _ {2} \\ \mu (f _ {l 4} - f _ {2}), & \quad f _ {l 4} > f _ {2} \end{array} \right.\tag{20}
$$

其中 $\mu { = } 2 7 , f _ { 1 } { = } 1 0 \ \mathrm { N } , f _ { 2 } { = } 1 5 \ \mathrm { N } .$，且 $f _ { l 4 }$ 表示由传感器测得的下部缆索张力。

## III. 静态-动态标定方法

所提出的静态-动态标定方法的算法（如图 4 所示）包括静态标定步骤（DLN 训练）、动态标定步骤（DLN 更新）以及动态验证步骤。

## A. 静态标定

在静态标定步骤中，每位受试者被要求在六个随机方向上对末端执行器（End-Effector, EE）施加自主作用力，共进行 21 次试验。每次试验的持续时间约为 15 s。在该步骤中，UL-CDR 的电机被锁定（无法移动），从而使受试者的肢体姿态保持固定。集成在 EE 中的六轴力传感器测量并记录了机器人与受试者手部之间的静态交互力。在静态阶段，测得的交互力等于受试者施加的自主作用力。交互力和标记点位置以 100 Hz 进行记录，而肌电（EMG）信号则以 1000 Hz 记录。为防止混叠，利用 EMG 信号计算出的肌肉激活度被降采样至 100 Hz，以与其他信号的频率相匹配。在静态标定数据集中，输入数据 U [通过式 (1)–(6) 获得] 包含肌肉激活度与关节角度信号，而输出数据 $F _ { h }$ 由三维交互力的三个分量组成（由力传感器记录，在静态阶段可视为自主作用力）。在 21 次试验中，9 次分配至训练集，3 次用作验证集，其余 9 次分配至测试集。来自训练集的数据用于 DLN 训练。式 (8)–(13) 中的网络偏置和权重矩阵采用随机数进行初始化。文献 [36] 中描述的 ADAM 算法被选用于网络训练，相关参数见表 II 第二列。为了防止过拟合，我们监测验证损失并在验证损失开始增加之前停止训练（称为早停策略）。随后，训练好的 DLN 被整合到系统中，用于在线 UL-CDR 控制。

TABLE II  
DLN 训练与更新过程参数

<table><tr><td>参数</td><td>数值（训练过程）</td><td>数值（更新过程）</td></tr><tr><td>梯度阈值方法</td><td>L2 范数（L2 norm）</td><td> $\sim$ </td></tr><tr><td>梯度阈值</td><td>1.0</td><td> $\sim$ </td></tr><tr><td>梯度衰减因子</td><td>0.9</td><td> $\sim$ </td></tr><tr><td>初始学习率</td><td> $5 \times 10^{-3}$ </td><td> $\sim$ </td></tr><tr><td>学习率下降周期</td><td>125</td><td>75</td></tr><tr><td>学习率下降因子</td><td>0.2</td><td> $\sim$ </td></tr><tr><td>小批量大小（Mini-batch size）</td><td>128</td><td> $\sim$ </td></tr><tr><td>Epsilon</td><td> $1 \times 10^{-8}$ </td><td> $\sim$ </td></tr><tr><td>最大轮数（Max epochs）</td><td>250</td><td>150</td></tr></table>

注：“ $\sim$ ”表示该数值与左侧单元格中的数值相同

## B. 动态校准（Dynamic Calibration）

在动态校准步骤中，三名受试者分别与 EE（末端执行器）进行交互，EE 通过受试者的 EMG 信号进行控制，在屏幕视觉反馈下跟踪形状为 0.25 0.25 m 矩形的参考轨迹，共进行 13 次试验 [Fig. 1(c)]。每次试验的平均持续时间约为 46 s。在此步骤中，UL-CDR 为受试者提供运动辅助（称为动态环境）。记录了 EMG 信号、标记点位置以及 EE 与受试者手部之间的动态交互力。值得注意的是，来自力传感器的反馈信号随后用于校准 DLN，而非用于控制机器人。在 13 次试验中，7 次用作 DLN 更新的训练集，3 次分配给验证集，其余分配给测试集。输入数据仍为关节角度和肌肉激活度，而输出数据则更改为估计力与动态交互力的合力（即矢量和）（Fig. 4，第 18 行）。式 (8)–(13) 中的初始偏置和权重矩阵设置为前一步骤中预训练 DLN 的参数。再次采用 ADAM 算法来更新 DLN。更新/再训练过程的参数如 Table II 的第二列所示。在此过程中还采用了早停（Early stopping）策略。更新后的 DLN 随后重新嵌入到系统中，替代预训练的 DLN 以进行动态验证。

## C. 动态验证（Dynamic Validation）

在动态验证步骤中，对每位受试者进行了两组实验（采用静态-动态校准的 DLN 以及仅采用静态校准的 DLN）。受试者与 EE 发生交互，EE 由受试者的 EMG 信号控制，在视觉反馈下跟踪两种参考轨迹 [Fig. 1(c)]。对于 subject1，两组实验均进行了 40 次（20 次用于 $0 . 2 5 \times 0 . 2 5$ m 矩形参考轨迹，20 次用于直径为 0.3 m 的圆形参考轨迹，每次试验平均耗时约 43 s），以便进行对比和统计分析。对于 subject2 和 subject3，两组实验均仅进行了 20 次（10 次用于矩形参考轨迹，10 次用于圆形参考轨迹）。这是由于他们要求实验时间不要过长，我们尊重了他们的要求。

总计有约 273, 900（<sub>¼</sub> 3 subjects   21 trials $\times$ $1 5 \mathrm { ~ s ~ } \times 1 0 0 \mathrm { ~ H z } + 3$ subjects   13 trials $\times 4 6 \mathrm { ~ s ~ } \times 1 0 0 \mathrm { ~ H z } )$ 对输入和输出用于 DLN 训练和更新过程。此外，共有 160（<sub>¼</sub> 1 subject   2 experiments $\times 4 0$ trials þ 2 subjects   2 experiments   20 trials）次试验用于统计分析。所有实验均已获得中山大学医学院机构审查委员会（School of Medicine, Sun Yat-sen University Institutional Review Board, IRB/SM-14/2021）的批准。

## IV. 信号处理与分析

采用估计的 3-D 末端力的均方根误差（RMSE）来评估 EMG-to-3-D-force 模型的估计精度

$$
F _ {\text { err } \_ j} = \sqrt {\frac {1}{T} \int_ {0} ^ {T} (F _ {h \_ \text { estimate } \_ j} - F _ {h \_ \text { actual } \_ j}) ^ {2}}\tag{21}
$$

$$
F _ {\text { err\_3D }} = \sqrt {\frac {1}{T} \int_ {0} ^ {T} \| \boldsymbol {F} _ {h \_ \text { estimate }} - \boldsymbol {F} _ {h \_ \text { actual }} \| ^ {2}}\tag{22}
$$

其中 $j { = } 1 , 2 ,$ 和 3 分别表示 $\mathbf { X } , \ \mathbf { y } ;$ 和 z 轴上的数据；T 为数据集的持续时间。

位置跟踪的 RMSE 和 RMSE 相对误差 $( \mathrm { R E _ { R M S E } } )$ 也被用于评估模型的精度和跟踪性能

$$
P _ {\text { RMSE } \_ j} = \sqrt {\frac {1}{T} \int_ {0} ^ {T} (P _ {\text { Reference } \_ j} - P _ {\text { actual } \_ j}) ^ {2}}\tag{23}
$$

$$
P _ {\text { RMSE } \_ 3 D} = \sqrt {\frac {1}{T} \int_ {0} ^ {T} \| P _ {\text { Reference }} - P _ {\text { actual }} \| ^ {2}}\tag{24}
$$

$$
P _ {\mathrm {RE\_RMSE\_j}} = \frac {P _ {\mathrm {RMSE\_j}}}{D _ {\text { Reference }}}\tag{25}
$$

其中 $D _ { \mathrm { R e f e r e n c e } }$ 为矩形/圆形参考轨迹的边长/直径。

使用 SPSS 26 软件进行配对 t 检验（paired t-tests），以检验两组实验（采用静动态标定的 DLN 与仅采用静态标定的 DLN）之间位置跟踪的 RMSE 和 MRE 差异。

![](images/615e3ff9de0f31d190c0da3b93140b266267050638687132942e06a7da12a6d9.jpg)  
(a)

![](images/c12b0be18278a3718fb8e01a82f083546e67bda89d77d2f87d2b7333a931c64a.jpg)  
(b)  
Fig. 5. 受试者 1（subject1）训练集和测试集中的一个代表性试验（目标力在静态测量中记录）。(a) 训练集单次试验中的力拟合结果。(b) 测试集单次试验中的力估计结果。

![](images/ec2d9c9eaeb069baacfd33ac87799722bd7d41d18e63f85bc8cc0968f2aa8f22.jpg)  
Fig. 6. 静态测量数据在训练集、验证集和测试集中三维末端力估计误差的平均 RMSE。

## V. 结果

在静态标定步骤中，使用 RTX 3070 GPU 训练 DLN 大约耗时 19 min。Fig. 5 展示了受试者 1（subject1）在训练集中的一个代表性试验和测试集中的一个代表性试验。训练集中的拟合结果和测试集中的估计结果分别展示在 Fig. 5(a) 和 5(b) 中。橙色曲线代表估计力，黑色曲线代表在静态测量中由六轴力传感器测得的目标力。所有受试者静态测量数据在训练集、验证集和测试集中的力估计平均 RMSE 如 Fig. 6 所示。尽管测试集的估计误差不可忽略，但正如 Fig. 5 所示，训练集和测试集中的估计力均趋近于目标力。

在动态标定步骤中，嵌入在 dSpace 中的 DLN 每步大约耗时 $0 . 0 0 3 \mathrm { ~ s ~ } ( < 0 . 0 1 \mathrm { ~ s ~ } )$ 进行力估计，这意味着 DLN 能够在硬件上以控制系统的频率（100 Hz）运行。受试者1（subject1）某一选定试验中的位置跟踪结果如图 7 所示。显然，位置跟踪误差并不小，这表明 DLN 仍有进一步优化的空间。因此，动态环境下的交互力被用于进一步优化。动态交互力加上估计力被作为 DLN 更新/进一步训练数据集的输出数据。图 8(a) 显示了训练集中的关节角度，图 8(b) 描绘了测试集中的关节角度。动态数据为网络训练提供了变化的关节角度信息 [图 8(a) 和 8(b)]，这在静态数据集中是无法获得的。图 8(c) 描绘了更新步骤中训练集一个样本的拟合结果，图 8(d) 描绘了测试集一个样本的估计结果。训练集、验证集和测试集中的力估计误差如图 9 所示。与图 5 相比，更新后的 DLN 学习到了输入与输出之间更详细/复杂的关系。

![](images/a87c09180d0a821c2abfcf00b8f7e4d06ee3534dec541c8cc9169806168b7359.jpg)  
(a)

![](images/0b33e03259f4e3891bb839055e21a9c116c63e2730a6c7dfba24f27d2804d73c.jpg)  
(b)

Fig. 7. 受试者1（subject1）代表性试验中的位置跟踪结果：末端执行器（EE）的实际轨迹与参考轨迹：(a) 在 x、y 和 z 轴上；以及 (b) 在三维视图中。  
![](images/e1cbb69c09b769336487b2fc04568f4261dbe5ecf2ddf8ac5a5c9abb208b031c.jpg)  
(a)

![](images/aa3d57fcbadc34e122cda88ea4343a54d9dcdffb539c687c02f09160ac276c9b.jpg)

![](images/426e776c1b367bf52f3e47edc441c7a231367330887f04b53efa88805ae8b994.jpg)  
(c)

(b)  
![](images/e4c82db5c37ef79fa74a63656986c121f78613ac1f513a8e05445591c1f72173.jpg)  
(d)  
Fig. 8. 更新过程中训练集和测试集各一个样本（目标数据记录于动态测量步骤）。(a) 训练集一次试验中的三个关节角度。(b) 测试集一次试验中的三个关节角度。(c) 训练集的拟合结果。(d) 测试集的拟合结果。

图 10 展示了在跟踪矩形参考轨迹时，采用静态-动态校准与仅采用静态校准的动态验证位置跟踪结果。图 11 描绘了跟踪圆形参考轨迹时动态验证的位置跟踪结果。使用带有静态-动态校准和仅带有静态校准的 DLN 在动态验证中的位置跟踪误差平均 RMSE 如图 12 所示。用于比较的具体跟踪误差（包括 RMSE 和 ${ \mathrm { R E } } _ { \mathrm { R M S E } } )$ 以及用于显著性分析的 p 值如表 III 所示。可以观察到，对于矩形和圆形轨迹跟踪任务，采用静态-动态校准的控制器均比仅采用静态校准的控制器获得了显著更好的轨迹跟踪性能 $( \mathtt { p } < 0 . 0 5 )$ ）。

![](images/aaf8688942f39c24ce7f265ca51ab54e8341cf2eb1bd44c5453bc52ffc4f22a8.jpg)  
Fig. 9. 动态校准过程（即更新过程）中动态数据的训练集、验证集和测试集的三维末端力估计误差平均 RMSE

![](images/68d9e624c0ee5abe5af8f5a2de1b0bf05c1b0ae096826914042335ecfabe8928.jpg)

![](images/aaf437864c2422445d2ea826e8574aa3f57440fccecde159236c181ed5927a49.jpg)

![](images/6a25294abc284b69a101acaf0a88954d0c7b8eb6dd9858023df6509ea5dcb1ab.jpg)  
(c)

(b)  
![](images/62fb53cc59f194056dc804cc1c08c70d67d4aeffd50e2ecb17cec69b6305fb9d.jpg)  
(d)  
Fig. 10. 跟踪矩形参考轨迹时采用静态-动态校准与仅采用静态校准的动态验证结果：EE 的实际轨迹和参考轨迹：(a) 采用静态-动态校准 DLN 时在 x、y 和 z 轴上的轨迹；(b) 仅采用静态校准 DLN 时在 x、y 和 z 轴上的轨迹；(c) 采用静态-动态校准 DLN 时的三维视图；以及 (d) 仅采用静态校准 DLN 时的三维视图。

## VI. 讨论

所提出的结合静态-动态校准方法的 DLN 不仅通过将关节角度纳入输入，消除了关节角度与人体主动发力耦合效应的影响，还通过二次校准补偿了人机耦合系统不确定动力学特性的影响。交互力仅在静态情况下等于人体主动发力。然而，在动态人机协作环境中，人体主动发力并不等于动态交互力，而是机器人助力与测量得到的交互力的矢量和 [图 8(c) 和 8(d)]。在动态校准步骤中，测量得到的动态交互力与先前的估计力的矢量和被用于校正从原始 DLN 模型获得的估计力，随后更新数据集以进行二次校准。已有其他研究尝试解决肌电-力（EMG-to-force）建模的问题，其中考虑了可变关节角度的影响。Guoxin 等人通过在三维空间的 72 个位置测量 6 个方向上的等长力，考虑了上肢姿态对肌电-力关系的影响 [37]。该方法需要耗费大量时间来采集数据集（$6 \times 7 2 = 4 3 2$ 次测量）。与 [37] 相比，我们提出的方法更为便捷，因为 DLN 可以利用动态交互力和估计力通过二次校准轻松完成更新。Pesenti 等人利用线性变参数模型估计一维力，其中参数是在变化的手肘关节角度下辨识得到的 [38]。然而，由于上肢神经肌骨系统中复杂的非线性，线性模型可能无法准确估计三维主动发力。与 [38] 相比，所提出的 DLN 更适合于复杂的非线性拟合，尤其是三维上肢运动解码。也有研究探索了基于网络的非线性模型，如 CNN [22]、ANN [19]、BPNN [18]、RNN [22]、带有 BPNN 的自回归网络 [16] 以及 RCNN [22]，用于运动意图估计并在离线数据集上对这些模型进行了测试。在本研究中，所提出的结合静态-动态校准方法的 DLN 被应用于机器人平台以实现实时肌电控制，并纳入了人机耦合系统的不确定动力学特性。

![](images/8b6fcd33aa2e770151bdb2b6fcdacf15e4b1963928c248226970e5af7cae6392.jpg)  
(a)

![](images/e9a11b5bcab5514666e8a1694079a0a72bfd26bee525ff3eda68fa6a4df61c61.jpg)  
(b)

![](images/343b5f84bd370bb1b37c1263299feeb0b1c9d76d19e90de98743162f36e0e245.jpg)  
(c)

![](images/a095c679c1ab87ec515e900eba7b082f183ec786471350dff4dc7a8adc9a5ca0.jpg)  
(d)  
Fig. 11. 跟踪圆形参考轨迹时采用静态-动态标定与仅采用静态标定的动态验证结果：末端执行器（EE）的实际轨迹与参考轨迹：(a) 采用静态-动态标定的 DLN 在 x、y 和 z 轴上的轨迹；(b) 仅采用静态标定的 DLN 在 x、y 和 z 轴上的轨迹；(c) 采用静态-动态标定的 DLN 的三维视图；以及 (d) 仅采用静态标定的 DLN 的三维视图。

![](images/347e299e2cf189d95738595ce69ecc347d4280ccad8301b289f936727d308135.jpg)  
(a)

![](images/2a34f87a70f8fb262c2ba84227a5e2675a10f85c29d054916f7a1263147f8254.jpg)  
(b)  
Fig. 12. 采用静态-动态标定和仅采用静态标定的 DLN 在动态验证期间跟踪误差的平均 RMSE：(a) 矩形参考轨迹跟踪实验；以及 (b) 圆形参考轨迹跟踪实验。

TABLE III  
采用静态-动态标定与仅采用静态标定的 DLN 控制器在动态验证期间跟踪误差的平均 RMSE 和 ${ \mathsf { R E } } _ { { \mathsf { R M S E } } }$

<table><tr><td rowspan="2">指标</td><td colspan="2">控制器 (Mean ± STD)</td><td rowspan="2">p 值</td></tr><tr><td>静态-动态标定</td><td>仅静态标定</td></tr><tr><td>Rectangle, RMSE_x</td><td>12.8 ± 3.9 mm</td><td>28.1 ± 13.7 mm</td><td>&lt;0.001</td></tr><tr><td>Rectangle, RMSE_y</td><td>19.2 ± 8.9 mm</td><td>36.4 ± 15.6 mm</td><td>&lt;0.001</td></tr><tr><td>Rectangle, RMSE_z</td><td>15.0 ± 8.3 mm</td><td>36.0 ± 13.7 mm</td><td>&lt;0.001</td></tr><tr><td>Rectangle, RMSE_3-D</td><td>28.7 ± 10.1 mm</td><td>59.8 ± 21.1 mm</td><td>&lt;0.001</td></tr><tr><td>Circle, RMSE_x</td><td>11.4 ± 3.7 mm</td><td>25.4 ± 7.1 mm</td><td>&lt;0.001</td></tr><tr><td>Circle, RMSE_y</td><td>22.8 ± 16.9 mm</td><td>38.8 ± 18.8 mm</td><td>0.004</td></tr><tr><td>Circle, RMSE_z</td><td>13.2 ± 5.4 mm</td><td>36.6 ± 14.0 mm</td><td>&lt;0.001</td></tr><tr><td>Circle, RMSE_3-D</td><td>29.4 ± 16.8 mm</td><td>60.4 ± 21.0 mm</td><td>&lt;0.001</td></tr><tr><td>Rectangle, RE $_{RMSE\_x}$ </td><td>5.12 ± 1.56%</td><td>11.24 ± 5.48%</td><td>/</td></tr><tr><td>Rectangle, RE $_{RMSE\_y}$ </td><td>7.68 ± 3.56%</td><td>14.56 ± 6.24%</td><td>/</td></tr><tr><td>Rectangle, RE $_{RMSE\_z}$ </td><td>6.00 ± 3.32%</td><td>14.40 ± 5.48%</td><td>/</td></tr><tr><td>Circle, RE $_{RMSE\_x}$ </td><td>3.80 ± 1.23%</td><td>8.47 ± 2.37%</td><td>/</td></tr><tr><td>Circle, RE $_{RMSE\_y}$ </td><td>7.60 ± 5.63%</td><td>12.93 ± 6.27%</td><td>/</td></tr><tr><td>Circle, RE $_{RMSE\_z}$ </td><td>4.40 ± 1.80%</td><td>12.20 ± 4.67%</td><td>/</td></tr></table>

尽管如 [25] 中所报道的，仅采用静态校准的 DLN 可用于构建 EMG 到三维力（EMG-to-3-D-force）模型并应用于 UL-CDR，但由于动态应用环境与训练集中的静态测量数据之间存在差异，运动意图估计的准确率并不高。跟踪误差较大的平均 RMSE 表明人机协作性能有所下降。我们的结果表明，仅使用静态测量数据训练的模型在应用于人机耦合系统时，在在线动态环境中可能表现不佳。原因之一是关节角度与人体主动发力之间存在耦合效应，正如 [39] 中所报道的，关节角度的变化 [图 8(a) 和 8(b)] 会影响肌电到力（EMG-to-force）的估计。另一个原因是仅使用静态测量数据训练的模型忽略了人机耦合系统的不确定动力学。相比之下，采用静态-动态校准方法的 DLN 不仅能够解耦关节角度和人体三维力，还可以通过根据动态数据更新网络来补偿人机耦合系统不确定动力学带来的影响。所提方法的这些优势使得实际跟踪轨迹更平滑（抖动更少），且跟踪误差的 RMSE 较低，如图 12 所示。尽管位置跟踪误差既取决于个体灵活性，也取决于三维力估计的精度，但我们指导所有参与者尽其所能跟随参考轨迹，并且每位参与者在所提方法和基线方法上都进行了相同数量的跟踪实验。我们没有告知参与者在每次试验中使用了哪种方法。这些设置可以消除个体灵巧度对对比实验的影响。本研究的一个局限性在于图 10(a)、10(c) 以及图 11(a)、11(c) 中的跟踪性能并不完美，这可能是由于网络训练数据有限所致，这也是机器人应用中的一个普遍挑战。在未来的工作中，静态-动态校准方法可以与更先进的网络模型相结合。

TABLE IV  
DLN (STATIC)、DLN (DYNAMIC)、RESIDUAL-DLN、RESIDUAL-BI-LSTM、RESIDUAL-GRU、RESIDUAL-BI-GRU、DLN-NO-DROPOUT、MUSCLENET-CNN 以及 MUSCLENET-RCNN 在三维力估计误差、权重数量和计算时间方面的比较

<table><tr><td rowspan="2">网络</td><td colspan="4">估计误差的 RMSE（均值 ± 标准差）</td><td rowspan="2">参数量</td><td rowspan="2">每步计算时间 (ms)</td></tr><tr><td>x (N)</td><td>y (N)</td><td>z (N)</td><td>3-D(N)</td></tr><tr><td>DLN (静态)</td><td>6.012 ± 0.780</td><td>2.760 ± 0.534</td><td>6.716 ± 0.331</td><td>9.457 ± 0.608</td><td>809 k</td><td>3.109</td></tr><tr><td>DLN (动态)</td><td>3.009 ± 0.512</td><td>3.803 ± 0.621</td><td>3.970 ± 0.205</td><td>6.278 ± 0.693</td><td>809 k</td><td>3.115</td></tr><tr><td>Residual-DLN</td><td>7.160 ± 1.399</td><td>3.753 ± 0.461</td><td>7.360 ± 0.937</td><td>10.980 ± 1.370</td><td>809 k</td><td>3.143</td></tr><tr><td>Residual-Bi-LSTM</td><td>9.518 ± 1.166</td><td>3.355 ± 0.324</td><td>8.704 ± 0.970</td><td>13.355 ± 1.599</td><td>818 k</td><td>4.851</td></tr><tr><td>Residual-GRU</td><td>7.501 ± 3.682</td><td>3.162 ± 0.543</td><td>7.009 ± 2.296</td><td>10.839 ± 4.166</td><td>809 k</td><td>7.483</td></tr><tr><td>Residual-Bi-GRU</td><td>6.845 ± 0.685</td><td>2.760 ± 0.316</td><td>6.004 ± 0.419</td><td>9.551 ± 0.619</td><td>810 k</td><td>9.395</td></tr><tr><td>DLN-no-Dropout</td><td>7.565 ± 1.030</td><td>3.489 ± 0.273</td><td>6.032 ± 1.204</td><td>10.338 ± 1.281</td><td>809 k</td><td>3.112</td></tr><tr><td>MuscleNET-CNN*</td><td>9.902 ± 1.194</td><td>4.236 ± 0.496</td><td>12.078 ± 0.496</td><td>16.216 ± 2.047</td><td>70 k</td><td>3.926</td></tr><tr><td>MuscleNET-RCNN*</td><td>9.476 ± 0.987</td><td>4.137 ± 0.576</td><td>10.080 ± 1.586</td><td>14.463 ± 1.756</td><td>86 k</td><td>4.404</td></tr></table>

\*注：文献 [22] 中的 MuscleNET-CNN 和 MuscleNET-RCNN 是为一维连续运动预测设计的，我们将其最后一层（$1 \times 1$ 回归输出尺寸）改为 $3 \times 1$ 回归输出尺寸，以匹配我们的三维末端力数据。

具有静动态校准的 DLN 控制器与 SSM 控制器在三维跟踪误差和每步总处理时间上的对比  
TABLE V

<table><tr><td rowspan="2">指标</td><td colspan="2">控制器</td></tr><tr><td>具有静动态校准的 DLN</td><td>SSM</td></tr><tr><td>矩形,  $RE_{RMSE\_x}$ </td><td>5.12±1.56 %</td><td>27.37±10.57</td></tr><tr><td>矩形,  $RE_{RMSE\_y}$ </td><td>7.68±3.56 %</td><td>22.26±43.74</td></tr><tr><td>矩形,  $RE_{RMSE\_z}$ </td><td>6.00±3.32 %</td><td>48.52±62.15</td></tr><tr><td>圆形,  $RE_{RMSE\_x}$ </td><td>3.80±1.23 %</td><td>22.88±5.89</td></tr><tr><td>圆形,  $RE_{RMSE\_y}$ </td><td>7.60±5.63 %</td><td>8.12±2.15</td></tr><tr><td>圆形,  $RE_{RMSE\_z}$ </td><td>4.40±1.80%</td><td>24.36±6.15</td></tr><tr><td>每步总处理时间（包括硬件 I/O 和软件处理时间）</td><td>3.1230 ms</td><td>0.0116 ms</td></tr></table>

选择嵌入硬件中用于实时肌电人机交互的模型需要权衡准确性和计算成本。我们还将所提出的 DLN 与 Table IV 中列出的其他代表性网络进行了比较，这些网络具有相似的权重总数或相似的计算时间。为了说明三层 LSTM 架构相较于单层或双层架构的重要性，我们引入了 residual-DLN 网络。该网络在各层之间采用恒等映射（identity mapping），以应对随着网络层数增加而出现的网络退化问题。它允许低级特征绕过中间的 LSTM 层直接传递至更高级的层。然而，我们的实验结果表明，residual-DLN 并未降低我们所提出的 DLN 的误差，从而证实了采用至少三层 LSTM 网络的有效性。此外，我们发现 residual-DLN、三层 residual-bi-LSTM、三层 residual-GRU、三层 residual-bi-GRU、DLN-no-dropout、MuscleNET-CNN [22] 以及 MuscleNET-RCNN [22] 网络的 3-D 估计误差均高于我们所提出的 DLN (static) 和 DLN (dynamic)。诸如 residual-Bi-LSTM 等双向模型在降低力估计误差方面的效果有限，这可能归因于神经肌肉骨骼系统的固有特性。当前的自主发力可能会受到肢体过去状态的影响，但由于人类中枢神经系统能够随意切换指令，力与未来的输入关联有限，从而限制了双向模型在捕获输入元素未来上下文关系方面的优势。在我们先前的研究中，一种基于线性状态空间模型（SSM）的肌电控制器也被嵌入到了上肢康复机器人中 [17]。尽管先前的控制器具有更快的计算性能且可用于伸展（reaching）等简单任务，但在应用于更复杂的任务（本研究中的矩形和圆形轨迹跟踪，Table V）时，其准确性逊于本研究所提出的控制器。本研究对 DLN 的选择实现了准确性与计算效率之间的良好平衡，确保了我们的模型在 100-Hz 人机交互应用中的可行性。

## VII. 结论

在本文中，提出了一种结合静态-动态校准方法的 DLN，用于建立上肢绳驱动康复机器人（UL-CDR）控制系统的非线性 EMG-to-3-Dforce（肌电至三维力）模型。与仅采用静态校准的方法相比，所提出的方法能够适应从静态测量数据到动态环境的变化，从而在人机协作应用中实现更优的估计准确度。

结合静态-动态校准方法的 DLN 网络使机器人能够通过其 EMG 信号在线准确解码人体的运动意图，具有在多个方面促进人机协作的潜力，包括：1）在重物搬运、表面抛光和材料锯切等任务中提高制造效率；2）辅助运动功能障碍人群并提升其生活质量与能力；3）实现主动康复训练并鼓励中风后患者的主动参与。采用二次校准方法的 DLN 网络还可用于进一步解决跨被试应用的问题。如果将利用健康被试数据训练的 DLN 网络进行微小重校准后应用于中风后患者，将是一个极具前景的研究方向。

![](images/9a18a4c3fd720c44eefa389b3beca5806024edc2803f7c28e02efcd5ab03cf99.jpg)  
(a)

![](images/bec221e0264c7ff5f24a64c1c182b8541764a635a4e31e6be12b418c0c7b81d7.jpg)  
(b)  
Fig. 13. (a) 静态校准和 (b) 动态校准中在特定轮次的早停。

## 附录

静态校准和动态校准中在特定轮次的早停情况如图 13 所示。

## 参考文献

[1] L. Bi, A. G. Feleke, and C. Guan, “A review on EMG-based motor intention prediction of continuous human upper limb motion for human–robot collaboration,” Biomed. Signal Process. Control, vol. 51, pp. 113–127, May 2019.

[2] T. Zhang, H. Sun, and Y. Zou, “An electromyography signals-based human–robot collaboration system for human motion intention recognition and realization,” Robot. Comput. Integr. Manuf., vol. 77, Oct. 2022, Art. no. 102359.

[3] A. Furui et al., “A myoelectric prosthetic hand with muscle synergy based motion determination and impedance model-based biomimetic control,” Sci. Robot., vol. 4, no. 31, Jun. 2019, Art. no. eaaw6339.

[4] J. Furukawa and J. Morimoto, “Composing an assistive control strategy based on linear bellman combination from estimated user’s motor goal,” IEEE Robot. Automat. Lett., vol. 6, no. 2, pp. 1051–1058, Apr. 2021.

[5] Q. Ai, Z. Liu, W. Meng, Q. Liu, and S. Q. Xie, “Machine learning in robot assisted upper limb rehabilitation: A focused review,” IEEE Trans. Cogn. Develop. Syst., vol. 15, no. 4, pp. 2053–2063, Dec. 2023.

[6] B. Rodriguez-Tapia, I. Soto, D. M. Marinez, and N. Candolfi Arballo, “Myoelectric interfaces and related applications: Current state of EMG signal processing—A systematic review,” IEEE Access, vol. 8, pp. 7792–805, 2020.

[7] G. Kai, U. X. Tan, L. Honghai, and Z. Dingguo, “Electromyography driven progressive assist-as-needed control for lower limb exoskeleton,” IEEE Trans. Robot. Bionics, vol. 2, no. 1, pp. 50–58, Feb. 2020.

[8] J. Wang, J. Liu, G. Zhang, and S. Guo, “Periodic event-triggered sliding mode control for lower limb exoskeleton based on human–robo cooperation,” ISA Trans., vol. 123, pp. 87–97, Apr. 2022.

[9] A. Alkan 和 M. Gunay，“使用判别分析和 SVM 分类器识别 EMG 信号，” Expert Syst. Appl., vol. 39, no. 1, pp. 44–47, Jan. 2012.

[10] S. Pancholi 和 A. M. Joshi，“用于改进对抗肌力变化的基于 EMG-PR 假肢控制的高级能量核特征提取方案，” IEEE Trans. Cybern., vol. 52, no. 5, pp. 3819–3828, May 2022.

[11] N. Nahid, A. Rahman, 和 M. A. R. Ahad，“用于低成本肌电假手基于深度学习的表面肌电（sEMG）手势分类，” in Proc. Joint 9th Int. Conf. Inform., Electron. Vis. (ICIEV); 4th Int. Conf. Image, Vis. Pattern Recognit. (ICIVPR), 2020, pp. 1–8.

[12] G. Durandau 等，“轻瘫患者通过神经力学建模对穿戴式机器人外骨骼的主动控制，” J. NeuroEng. Rehabil., vol. 16, no. 1, pp. 1–18, Jul. 2019.

[13] Y. Zhuang, Y. Leng, J. Zhou, R. Song, L. Li, 和 S. Su，“健全人与脑卒中幸存者采用基于 EMG 导纳控制方案对踝关节外骨骼的主动控制，” IEEE Trans. Biomed. Eng., vol. 68, no. 2, pp. 695–705, Feb. 2021.

[14] P. K. Artemiadis 和 K. J. Kyriakopoulos，“利用低维嵌入实现基于 EMG 的机械臂控制，” IEEE Trans. Robot., vol. 26, no. 2, pp. 393–398, Apr. 2010.

[15] J. Han, Q. Ding, A. Xiong, 和 X. Zhao，“用于估计连续关节运动的状态空间 EMG 模型，” IEEE Trans. Ind. Electron., vol. 62, no. 7, pp. 4267–4275, Jul. 2015.

[16] Q. Ding, J. Han, 和 X. Zhao，“利用状态空间模型基于 sEMG 连续估计人体多关节角度，” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 25, no. 9, pp. 1518–1528, Sep. 2017.

[17] C. Xie, Q. Yang, Y. Huang, S. Su, T. Xu, 和 R. Song，“配备基于 EMG 导纳控制器的臂手复合康复机器人，” IEEE Trans. Biomed. Circuits Syst., vol. 15, no. 6, pp. 1332–1342, Dec. 2021.

[18] Y. M. Aung 和 A. Al-Jumaily，“利用表面 EMG 信号估计上肢关节角度，” Int. J. Adv. Robot. Syst., vol. 10, pp. 1–8, Oct. 2013.

[19] Q. Zhang, R. F. Liu, W. B. Chen, 和 C. H. Xiong，“基于表面 EMG 信号同时且连续估计肩部和肘部运动学，” Frontiers Neurosci., vol. 11, pp. 1–12, May 2017.

[20] S. Pancholi, A. Giri, A. Jain, L. Kumar, 和 S. Roy，“利用 EEG 信号重建手部运动学的源感知深度学习框架，” IEEE Trans. Cybern., vol. 53, no. 7, pp. 4094–4106, Jul. 2023.

[21] M. A. Ozdemir, D. H. Kisa, O. Guren, 和 A. Akan，“基于时频图像和 CNN 迁移学习的手势分类，” Biomed. Signal Process. Control, vol. 77, 2022, Art. no. 103787.

[22] A. Nasr 等，“MuscleNET：通过机器学习将肌电图映射到运动学和动力学生物力学变量，” J. Neural Eng., vol. 18, no. 4, pp. 1–21, Aug. 2021.

[23] T. T. Dao, “From deep learning to transfer learning for the prediction of skeletal muscle forces,” Med. Biol. Eng. Comput., vol. 57, no. 5, pp. 1049–1058, May 2019.

[24] Y. Chen et al., “A continuous estimation model of upper limb joint angles by using surface electromyography and deep learning method,” IEEE Access, vol. 7, pp. 174940–174950, 2019.

[25] C. Xie, T. Xu, and R. Song, “A deep LSTM based sEMG-to-force model for a cable-driven rehabilitation robot,” in Proc. 7th IEEE Int. Conf. Adv. Robot. Mechatron. (ICARM), Guilin, China, 2022, pp. 660–665.

[26] J. Hashemi, E. Morin, P. Mousavi, and K. Hashtrudi-Zaad, “Surface EMG force modeling with joint angle based calibration,” J. Electromyogr. Kinesiol., vol. 23, no. 2, pp. 416–424, Apr. 2013.

[27] Y. Zhuang, S. Yao, C. Ma, and R. Song, “Admittance control based on EMG-driven musculoskeletal model improves the human–robot synchronization,” IEEE Trans. Ind. Inform., vol. 15, no. 2, pp. 1211–1218, Feb. 2019.

[28] L. L. Menegaldo, L. F. de Oliveira, and K. K. Minato, “EMGD-FE: An open source graphical user interface for estimating isometric muscle forces in the lower limb using an EMG-driven model,” Biomed. Eng., vol. 13, Apr. 2014, Art. no. 37.

[29] H. J. Hermens, B. Freriks, C. Disselhorst-Klug, and G. Rau, “Development of recommendations for SEMG sensors and sensor placement procedures,” J. Electromyogr. Kinesiol., vol. 10, no. 5, pp. 361–374, Oct. 2000.

[30] M. Hakonen, H. Piitulainen, and A. Visala, “Current state of digital signal processing in myoelectric interfaces and related applications,” Biomed. Signal Process. Control, vol. 18, pp. 334–359, Apr. 2015.

[31] T. S. Buchanan, D. G. Lloyd, K. Manal, and T. F. Besier, “Neuromusculoskeletal modeling: Estimation of muscle forces and joint moments and movements from measurements of neural command,” J. Appl. Biomech., vol. 20, no. 4, pp. 367–395, Nov. 2004.

[32] F. E. Zajac, “Muscle and tendon: Properties, models, scaling, and application to biomechanics and motor control,” Crit. Rev. Biomed. Eng., vol. 17, no. 4, pp. 359–411, 1989.

[33] Y. Huang, R. Song, A. Argha, A. V. Savkin, B. G. Celler, and S. W. Su, “Continuous description of human 3D motion intent through switching mechanism,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 28, no. 1, pp. 277–286, Jan. 2020.

[34] K. Manal and T. S. Buchanan, “A one-parameter neural activation to muscle activation model: Estimating isometric joint moments from electromyograms,” J. Biomech., vol. 36, no. 8, pp. 1197–1202, Aug. 2003.

[35] O. Bo and S. Weiwei, “Rapid optimization of tension distribution for cable-driven parallel manipulators with redundant cables,” Chin. J. Mech. Eng., vol. 29, no. 2, pp. 231–238, Mar. 2016.

[36] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proc. 3rd Int. Conf. Learn. Representations, San Diego, USA, 2015, pp. 1–15.

[37] L. Guoxin, L. Zhijun, L. Junjun, L. Yueyue, and Q. Hong, “Musclesynergy-based planning and neural-adaptive control for a prosthetic arm,” IEEE Trans. Artif. Intell., vol. 2, no. 5, pp. 424–436, Jun. 2021.

[38] M. Pesenti, Z. Alkhoury, M. Bednarczyk, H. Omran, and B. Bayle, “Linear parameter-varying identification of the EMG-force relationship of the human arm,” in Proc. 28th IEEE Int. Conf. Robot Hum. Interactive Commun. (RO-MAN), New Delhi, India, 2019, pp. 1–6.

[39] E. P. Doheny, M. M. Lowery, D. P. FitzPatrick, and M. J. O’Malley, “Effect of elbow joint angle on force-EMG relationships in human elbow flexor and extensor muscles,” J. Electromyogr. Kinesiol., vol. 18, no. 5, pp. 760–770, Oct. 2008.

![](images/07c8025925255fd0a287ed8828a28212c35cbc51878c982b5769e78dc42335b8.jpg)

Chenglin Xie（IEEE 学生会员）于 1996 年出生于中国汕头。他于 2019 年在广东中山大学（Sun Yat-sen University）获得生物医学工程学士学位，目前正在该校攻读生物医学工程博士学位。

他的研究兴趣包括康复机器人、肌电控制、人机交互、神经网络应用和生物医学信号处理。

Xie 博士曾获 2021 年 IEEE-EMB 港澳联合分会学生项目竞赛杰出表现奖（Outstanding Performance Award），以及 IEEE ARM 2020 最佳论文入围奖（Best Paper Finalist Award）。

![](images/abc5dcbef3676eda432e57d18c8d44900248a8b780a058f3e59aea8c7bd3428c.jpg)

Zhijun Li（IEEE 会士）于 2002 年在上海交通大学（Shanghai Jiao Tong University）获得机电一体化博士学位。

2003 年至 2006 年，他先后在日本东京电气通信大学（University of Electro-Communications）和新加坡国立大学（National University of Singapore）担任博士后研究员。目前，他是中国上海同济大学（Tongji University）讲席教授，并自 2024 年起担任机械与能源工程学院院长。

Li 博士已发表论文 400 余篇，主要贡献集中在可穿戴机器人和生物机电系统领域。他曾入选 IEEE RAS 杰出讲师（Distinguished Lecturer）、Web of Science 全球高被引科学家（2019–2023 年）、2018 年国家万人计划领军人才、2016 年国家杰出青年科学基金（NSFC）。他是 IEEE 和 AAIA 会士。他目前的研究兴趣包括可穿戴机器人、生物机电系统和智能控制。

![](images/822251bfa130c38ecc9f470f395a719f1360740c063b7e173e3de696fd13a30b.jpg)

Rong Song（IEEE 高级会员）于 1975 年出生于中国湖北。他于 1999 年在清华大学（Tsinghua University）获得电气工程学士学位，2002 年在汕头大学（Shantou University）获得电子工程硕士学位，2006 年在香港理工大学（The Hong Kong Polytechnic University）获得生物医学工程博士学位。

他目前是中山大学（Sun Yat-sen University）生物医学工程学院教授，

中国广东。他的研究兴趣包括肌肉骨骼建模、生物医学信号处理和康复机器人。

宋博士曾担任 IEEE 先进机器人与机电一体化会议（IEEE Conference on Advanced Robotics and Mechatronics, IEEE-ARM）的专题分会主席。他一直担任多本 IEEE Transactions 的审稿人。