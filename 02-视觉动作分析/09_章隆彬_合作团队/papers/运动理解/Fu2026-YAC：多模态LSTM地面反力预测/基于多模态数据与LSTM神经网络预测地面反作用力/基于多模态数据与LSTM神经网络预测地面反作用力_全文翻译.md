# 基于多模态数据与LSTM神经网络预测地面反作用力

1<sup>st</sup> Xinyi Fu School of Artificial Intelligence and Robotics, Hunan University（湖南大学人工智能与机器人学院） Changsha, China foxn@hnu.edu.cn

4<sup>th</sup> Suiyuan Wang Fourth Hospital of Changsha, Changsha Hospital of Hunan Normal University（长沙市第四医院，湖南师范大学附属长沙医院） Changsha, China wangsuiyuan@hunnu.edu.cn

7<sup>th</sup> Longbin Zhang \* School of Artificial Intelligence and Robotics, Hunan University（湖南大学人工智能与机器人学院） Changsha, China longbin@hnu.edu.cn

2<sup>nd</sup> Zhaoyuan Wan Department of Engineering Mechanics KTH Royal Institute of Technology（瑞典皇家理工学院工程力学系） Stockholm, Sweden zhaoyuan@kth.se

5<sup>th</sup> Te Zhang Department of Stomatology Hunan Provincial People’s Hospital（湖南省人民医院口腔科） Changsha, China 286414895@qq.com

8<sup>th</sup> Xueyu Zhu Department of Mathematics University of Iowa（爱荷华大学数学系） Iowa, USA xueyu-zhu@uiowa.edu

3<sup>rd</sup> Zhizhang Li School of Artificial Intelligence and Robotics, Hunan University（湖南大学人工智能与机器人学院） Changsha, China zhizhangli@hnu.edu.cn

6<sup>th</sup> Hui Zhang School of Artificial Intelligence and Robotics, Hunan University（湖南大学人工智能与机器人学院） Changsha, China zhanghui1983@hnu.edu.cn

9<sup>th</sup> Ruoli Wang<sup>\*</sup> Department of Engineering Mechanics KTH Royal Institute of Technology（瑞典皇家理工学院工程力学系） Stockholm, Sweden ruoli@kth.se

摘要—在非实验室环境中准确估计三维地面反作用力对于生物力学分析和康复监测至关重要。尽管可穿戴传感器提供了一种便携式解决方案，但仅依赖惯性测量单元（Inertial Measurement Units, IMU）或压力鞋垫的单一模态方法往往面临信息缺失问题，导致在复杂动态任务中的估计精度受限。为解决这一局限性，本研究提出了一种基于双向长短期记忆（BiLSTM）网络的多模态融合框架。该框架整合了由 IMU 获取的全身逆运动学（Inverse Kinematics, IK）数据与来自智能鞋垫的足底压力分布数据。我们在一个包含 9 名健康受试者的数据集上，通过四种不同的评估方案验证了该模型，涵盖了包括步行、深蹲、坐站转换、过渡动作和垂直跳跃在内的多种日常功能性活动。实验结果表明，多模态融合方法始终优于单模态基线模型。此外，该模型在多种运动任务中展现出良好的适应性，即使在涉及快速质心调节的复杂瞬态活动中也能保持一致的性能。这些结果表明，结合 IK 和足底压力数据能够提供互补的生物力学信息，从而克服单模态估计的局限性，并提高自由生活环境中步态分析的泛化能力。

关键词—ground reaction forces（地面反作用力）, kinetics（动力学）, machine learning（机器学习）, wearable sensors（可穿戴传感器）, sensing insole（传感鞋垫）

## I. 引言

地面反作用力（Ground Reaction Force, GRF）为康复外骨骼提供了感知人-环境交互并确保稳定运动控制所需的重要动力学信息。作为生物力学动力学分析和姿态控制研究中的一个重要指标，它不仅反映了人体的动态力特性，还广泛应用于临床步态分析 [1]–[5]、康复评估 [6]–[9] 以及下肢损伤预防 [10], [11]。此外，GRF 还是量化运动动力学和评估各种功能活动中运动表现的基本指标。精确获取 GRF 对于阐明人体运动控制机制以及评估运动过程中的机械负荷 [12], [13] 至关重要。

目前，GRF 测量的金标准依赖于地面嵌入式测力台与光学动作捕捉系统的联合使用。然而，该方法局限于实验室环境，设备昂贵且固定，严格限制了其在户外环境或日常生活活动中的应用。因此，研究人员探索了基于可穿戴传感器的技术，以实现自由生活环境下的 GRF 测量 [14], [15]。惯性测量单元（Inertial Measurement Units, IMU）作为集成了加速度计、陀螺仪和磁力计的小型可穿戴传感器，可测量线加速度、角速度和磁场，已成为兼顾精度与易用性的工具。Sun 等人 [16] 利用可穿戴 IMU 结合模块化 LSTM 网络对跳下着地进行动力学监测，在单腿任务中垂直 GRF 的决定系数（R<sup>2</sup>）达到了 0.88 0.12。与此同时，智能鞋垫 [17] 利用阵列式压力传感器来监测足底压力分布、压力中心轨迹和接触时序。这些设备无需外部接线或外围附属装置，在提供便利性的同时减少了对自然步态的潜在干扰。Zhang 等人 [18] 探索了低成本智能鞋垫在 GRF 预测中的应用，实现了 10−<sup>3</sup> 数量级的最小均方误差损失。然而，尽管鞋垫系统集成了 IMU，但其传感器布局仅局限于足部，缺乏捕获整个下肢动力学链全面运动学信息的能力。

传统的 GRF 估计方法主要依赖于基于物理的肌肉骨骼模型结合逆动力学分析 [19]。尽管这些方法提供了清晰的生物力学可解释性，但它们计算量大，且对建模不确定性高度敏感，例如环节长度估计中的误差以及关于质量分布的假设。近年来，随着深度学习的快速发展，数据驱动的方法被相继提出。Marion Mundt 等人 [20] 尝试将运动时间序列转换为图像，并使用卷积神经网络根据可穿戴加速度计数据预测 3D GRF 和力矩。Komaris 等人 [21] 利用人工神经网络和下肢运动学预测了跑步过程中的 3D GRF。然而，他们的模型是在稳态跑步数据上训练的，并未涵盖深蹲或跳跃等动态任务。这些方法通过利用人工神经网络在传感器数据与 GRF 之间建立非线性映射，绕过了复杂的物理建模，在计算效率和泛化能力方面展现出显著优势。

尽管单模态输入在某些应用中取得了良好的效果，但在复杂的动态场景中，仅基于单一类型传感器的 GRF 估计仍可能面临局限性。基于 IMU 的方法有时难以准确捕捉足-地接触动力学和快速的压力变化 [16]，而从鞋垫传感器获取的足底压力仅能提供垂直分量，前后向（AP）和内外侧向（ML）GRF 则需要通过其他手段推导获得 [22]。这些局限性凸显了对融合互补生物力学信息的多模态传感器融合策略的需求。

本研究的目标是开发并验证一个用于预测日常功能性活动中 3D GRF 的多模态框架。为实现这一目标，本研究联合使用可穿戴 IMU 和智能鞋垫分别采集全身运动学和足底压力信息。在双向长短期记忆（BiLSTM）网络中融合源自 IMU 数据的逆向运动学（IK）和足底压力特征，从而实现对一系列日常运动任务中 3D GRF 的可靠估计，同时对未见受试者表现出良好的适应性。

![](images/8b95353b7e473ee0a33d2ad22e261a09843f1ab0652b59be742630699853a237.jpg)  
Fig. 1. 用于预测地面反作用力的多模态融合框架概览。该流程包含数据采集、跨模态时空对齐、任务相关的混合分割策略以及 BiLSTM 深度回归网络的构建。

## II. 方法

本研究的总体架构如图 1（Fig. 1）所示。工作流程主要包含三个阶段：数据采集、数据处理和模型训练。首先，系统从多模态传感器同步采集数据。随后，通过数据处理流水线构建对齐的时间序列样本。最后，将预处理后的多模态特征输入 BiLSTM 网络以预测 3D 地面反作用力（GRF）。

![](images/8d81d5abdc0d33acae4c42dc905d0e586e19ff9f64d80ea2e5e70c6c69ad1494.jpg)  
Fig. 2. 鞋垫的 2D 示意图。左右鞋垫均配备有传感器阵列

## A. 数据采集

本研究招募了九名健康受试者（年龄：32.5 ± 6.3 岁）。使用与实验室金标准设备同步的多传感器可穿戴系统，同时记录了多种活动期间的运动数据——包括不同速度的行走、深蹲、坐立转换（sit-to-stand, STS）和垂直跳跃。对于每项活动，受试者需完成两次试验，每次试验包含 3 到 5 次连续的动作重复。该研究已获得瑞典伦理审查局（Swedish Ethical Review Authority, 2020-02311）的批准。可穿戴系统由八个采样率为 100 Hz 的无线 IMU（x-io Technologies Limited, 英国）和一双嵌入阵列式压力传感器的智能鞋垫组成。八个 IMU 分别固定在受试者的躯干、骨盆、股骨、胫骨和距骨上，以捕捉躯干和下肢的运动姿态。同时，智能鞋垫（Moticon ReGoAG, 德国）记录了双脚的压力中心分布和总接触力（Fig. 2）。金标准参考系统由光学动作捕捉系统（VICON Vantage V16, Vicon, 英国）和 AMTI 测力系统（AMTI, 美国）组成，用于记录解剖标记点轨迹和真实 GRF。行走试验中的 GRF 通过仪器化跑步机（AMTI Tandem）记录，其他活动中的 GRF 则通过四个 3D 测力台（AMTI Optima）记录。3D 动作捕捉系统的标记点按照 CGM2.3 标记集协议 [22], [23] 进行粘贴。存在严重标记点遮挡或与测力台接触不完全的采集数据均已从最终数据集中剔除。

## B. 数据处理

由于可穿戴设备与实验室金标准数据流之间不可避免地存在起始时间偏移和微小的采样频率漂移，直接使用原始数据会引入显著的预测误差。为了解决这种时间未对齐问题，本研究通过峰值匹配实施了跨模态对齐策略，并将所有数据流重采样至统一的 100 Hz 频率，以构建一致且等长的时间序列。对于金标准数据，测力台信号使用四阶零相位巴特沃斯（Butterworth）低通滤波器进行滤波，处理后的动力学数据作为监督学习的真实标签（ground truth）[24]。为了最大程度减少人体测量学差异并促进模型收敛，所有地面反作用力（GRF）数据均按受试者的体重（BW）进行了归一化。最后，采用任务相关的混合分割策略将连续数据流切分为独立的动作样本，随后统一重采样至固定序列长度 $T = 201$。

## C. 神经网络架构与训练

本研究采用了一种基于双层 BiLSTM 网络的深度回归模型。该模型架构主要由三个部分组成：输入层、核心 BiLSTM 模块以及 MLP 回归解码层。输入张量 $\mathbf { X } \in \mathbb { R } ^ { T \times D _ { i n } }$ 被配置为仅包含源自 IMU 的逆运动学（IK）数据的单模态输入（$D _ { i n } \ = \ 1 9$，包括骨盆的 3 个平移量、倾斜、倾倒和旋转；腰椎的伸展、侧屈和旋转；双侧髋关节屈曲、内收和旋转；双侧膝关节屈曲；以及双侧踝关节背屈），以及在此基础上整合了智能鞋垫动力学数据的多模态输入 $( D _ { i n } = 2 2 )$。输入数据首先通过全连接层映射到潜在空间以进行初步的特征融合。核心网络由两个堆叠的 BiLSTM 单元组成，层间引入了 Dropout 机制以防止过拟合。随后，由两个全连接层组成的多层感知机（MLP）充当解码器，将 BiLSTM 输出的 256 维高级时序特征回归为三维力向量。第一个全连接层包含 128 个神经元，并应用 ReLU 激活函数引入非线性；第二个全连接层包含 3 个神经元，直接输出归一化后的三维 GRF 预测值 $( F _ { x } , F _ { y } , F _ { z } )$。

## D. 评估方案

为了全面评估模型在不同应用场景下的预测精度、任务适应性和泛化能力，本研究设计了四种实验方案，定义如下：

受试者内单任务（Intra-Subject Single-Task, Intra-ST）：在该模式下，针对每位受试者的每个特定运动任务独立训练一个专用模型。在对特定受试者和任务的时间序列样本进行打乱后，数据集按 70% : 15% : 15% 的比例划分为训练集、验证集和测试集。测试集仅用于最终的性能评估，不参与梯度更新。

受试者内多任务（Intra-Subject Multi-Task, Intra-MT）：在该模式下，将单个受试者执行的所有运动任务的数据汇集在一起，构建一个统一的个人数据库。通过训练一个能够同时处理多种动作的统一模型，我们评估了其在不同动力学模式之间切换和适应的能力。与单任务模式类似，该协议采用了随机划分策略，其中 70% 用于训练，15% 用于验证，15% 用于测试。

跨受试者单任务（Inter-Subject Single-Task, Inter-ST）：该模式模拟了模型在未见受试者的特定任务上的应用。数据划分采用了严格的留一受试者交叉验证（Leave-One-Subject-Out Cross-Validation, LOSO）策略 [25]。在每一轮中，从总共 N 名受试者中选出一名受试者的完整数据作为独立测试集，而其余 N - 1 名受试者的数据汇集作为训练集。该过程重复 N 次，以确保每位受试者都作为新用户被测试过。

跨受试者多任务（Inter-Subject Multi-Task, Inter-MT）：在此模式下，模型使用包含来自 N - 1 名受试者所有运动任务的海量数据进行联合训练，并在剩余一名受试者的全任务数据上进行验证。

为了量化模型的预测性能，本研究采用均方根误差（Root Mean Square Error, RMSE）和归一化均方根误差（Normalized Root Mean Square Error, NRMSE）作为主要评估指标。RMSE 评估了预测 GRF 与真实测量值之间的绝对偏差，并通过 BW 进行了归一化，而 NRMSE 则提供了相对误差的评估。此外，利用 Wilcoxon 符号秩检验比较了单模态和多模态配置之间的配对误差分布。

统计显著性阈值设定为 $p \ < \ 0 . 0 5$，在随后的比较可视化图表中用星号（*）表示。

![](images/3f5b171a811d2ada84636b28b69f37a5ef39ddfcc4a7dbd5f8682f0bcb0daf28.jpg)  
(a) Intra-ST

![](images/6c8554ec8ecb0afdd003b79e4eafb5f563b84dd6979426615ab18dd73bc6dd63.jpg)

![](images/9ce36f3e01c8e49ae0132843b3765f9411ba0d3a1723d7104b55db889ebe9c09.jpg)  
(c) Inter-ST

![](images/e36e815b366cdbd455c3525653cee1620caffb5eda4c7bbf30ef1479cec6822f.jpg)  
(d) Inter-MT  
Fig. 3. 描绘七种运动下 3D GRFs 估计的 RMSE 分布小提琴图，对比了仅基于 IK 数据与结合 IK 和鞋垫数据的预测结果。每个小提琴图内含一个箱线图，表示 RMSE 值的下四分位数、中位数和上四分位数。该分布在四种实验方案下进行评估：(a) 被试内单任务（Intra-ST）、(b) 被试内多任务（Intra-MT）、(c) 被试间单任务（Inter-ST）以及 (d) 被试间多任务（Inter-MT）。\* 表示两种输入模态之间存在统计学显著性差异 $( p < 0 . 0 5 )$。

## III. 结果

单模态与多模态配置之间的对比分析表明，多模态融合持续提升了预测精度。如 RMSE（Fig. 3）和 NRMSE（Fig. 4）的分布所示，对于具有较低幅值特征的水平剪切力，模型在大多数任务中 ML 和 AP 方向上的回归能力均表现出小幅改善。各方案间的平均 NRMSE 分别降低了 0.5%-1% 和 1%-1.5%，相应的平均 RMSE 降低了约 0.02 N/kg 和 0.04 N/kg。对于跨被试方案，在 Inter-ST 下的正常行走（ML 方向）以及正常行走、快速行走、斜坡行走和 STS（AP 方向）中观察到了显著改善；在 Inter-MT 下的斜坡行走和 STS（ML 方向）以及正常行走和慢速行走（AP 方向）中也观察到了显著改善。多模态融合后，垂直方向上的整体预测误差有所降低，平均 NRMSE 降低了约 5%，平均 RMSE 降低了约 0.45 N/kg。值得注意的是，这种误差降低在 7 个运动任务中的绝大多数都具有统计学显著性 $( p < 0 . 0 5 )$（Intra-ST：6 个；Inter-ST：7 个；Intra-MT：6 个；Inter-MT：5 个），而其余任务的误差虽然没有显著性差异，但也有所减小。

![](images/23896ecf7566b117ed1248db8bf8ff91fee9b1c78c9bb412555315576e70d1a5.jpg)  
(a) Intra-ST

![](images/bbc750e8ff3232e925a95e94dbb4a603af5c254acacdd87a0291d530c0e45abd.jpg)  
(b) Intra-MT

![](images/399a8ee35d21a428923bc3e05c86a165695b64fdfe9ffd59282ce893fa05f58e.jpg)  
(c) Inter-ST

![](images/5bc4894e6fa69fffda063c36a5f2dcf6ad55aeac301577a8e9acd193d4f04114.jpg)  
(d) 跨被试多任务（Inter-MT）  
图 4. 展示七种动作下 3D GRF 估计的 NRMSE 分布小提琴图，对比了仅基于 IK 数据的预测与结合 IK 及鞋垫数据的预测。每个小提琴图内均包含一个内部箱线图，表示 NRMSE 值的下四分位数、中位数和上四分位数。分布在四种实验方案下进行评估：(a) 被试内单任务（Intra-ST），(b) 被试内多任务（Intra-MT），(c) 跨被试单任务（Inter-ST），以及 (d) 跨被试多任务（Inter-MT）。\* 表示两种输入模态之间存在统计学显著性差异 $( p < 0 . 0 5 )$。

本研究还评估了 LSTM 模型在跨任务和跨被试预测 3D GRF 时的泛化能力。在被试内方案中，该模型表现出较高的拟合精度。特别是对于自然行走等周期性任务，NRMSE 的上四分位数始终保持在 10% 以内，在 ML、AP 和垂直方向上对应的 RMSE 上四分位数分别约为 0.3 N/kg、0.4 N/kg 和 1.5 N/kg。非周期性动态任务的总体预测误差高于自然行走等平稳的周期性任务。归一化误差结果显示，深蹲（squat）的 NRMSE 最高，其次是 STS（起坐动作），而跳跃（jump）的误差相对最低。

与 Intra-MT 模型相比，跨被试评估的平均 NRMSE 增加了约 5.18%（同时平均 RMSE 增加了约 0.24 N/kg）。此外，误差分布变得明显更加离散，高误差离群值与中位数的偏离程度更大。

## IV. 讨论

本研究构建了一个 BiLSTM 网络，通过融合 IK 数据和足底压力信息来估计七种动作中的 GRF。四种评估协议下的实验结果表明，无论是采用单任务训练还是统一的多任务方法，该模型均能可靠地预测动力学指标。本研究的一个重要发现是揭示了足底压力数据作为物理约束在 GRF 预测中所起到的关键作用。水平剪切力预测精度的提升可能归因于智能鞋垫提供的压力中心位移，这为模型提供了关键的接触点定位和物理上下文信息。在 GRF 预测的垂直维度上，这种多模态融合同样表现出了校正效应。在涉及质心快速上升和下降的动作（如 STS 和跳跃）中，仅依赖 IK 数据的模型往往会低估垂直力峰值，这可能是由于其完全依赖运动学特征所致。通过结合直接测量垂直反作用力的鞋垫数据，模型能够更敏感地捕捉到力学极值。这些结果表明，虽然 IK 数据提供了丰富的身体姿态信息，但足底压力数据在减少运动学解的不确定性以及校正 GRF 幅值偏差方面发挥了至关重要的互补作用。

值得注意的是，在 Inter-ST 深蹲协议中，某些受试者的组合输入预测精度在 NRMSE 指标上并未优于单输入基准。这可能是由于在某些深蹲试验中采集到的鞋垫数据存在系统性偏移。然而，这种差异在 RMSE 中并不明显。一种可能的解释是，NRMSE 通过真实值的动态范围对误差进行归一化，而深蹲动作中真实值变化的幅度相对较小，从而放大了传感器偏移的影响。因此，由 IK 提供的关节角度信息不受这些偏移的影响，与 GRF 保持了相对稳定的映射关系，从而获得了更低的归一化预测误差。

即使在更具挑战性的多任务混合模式下，该模型仍保持了良好的性能，证明了其能够有效捕捉单个受试者在不同运动模式之间切换时的动力学特征。然而，模型的跨受试者泛化能力表现出方向上的差异。尽管在预测垂直 GRF 时保持了合理的泛化性能，但其在水平面上的表现存在一定局限性。此外，本研究的数据集仅包含九名健康受试者。未来的研究可以考虑纳入更多样化的人群，包括具有不同身高、体重、性别和健康状况的个体，以进一步验证并提升该模型的通用适用性。

## V. 结论

本研究提出了一种基于 BiLSTM（双向长短期记忆）网络的数据驱动方法，缓解了传统生物力学模型的局限性。此外，研究表明，融合 IK（逆运动学）数据与足底压力特征的多模态框架能够提高日常功能活动中 3D GRF（三维地面反作用力）的预测精度。结果表明，该多模态融合模型在各类复杂的动态任务中均展现出稳定的预测能力。与单模态基线相比，该框架降低了垂直方向上的预测误差，并在估计 AP（前后向）和 ML（内外向）剪切力方面实现了不同程度的精度提升。即使在最具挑战性的 Inter-MT 协议下，模型的预测误差仍保持在可接受的范围内，展现出一定程度的泛化能力和适应性。该多模态框架有助于弥补单一传感器的信息局限性，为非实验室环境下的生物力学监测提供了可靠的解决方案。

## 参考文献

[1] Masato Murakami, Tsuneo Okada, ”Gait analysis using threedimensional motion and ground reaction force systems in patients with

hemiplegia treated with botulinum toxin type A in ankle plantar flexors,” Journal of rural medicine, vol. 19, no. 3, pp. 174-180, 2024.

[2] Md Nazmul Islam Shuzan, Muhammad E.H. Chowdhury, Mamun Bin Ibne Reaz, et al., ”Machine learning-based classification of healthy and impaired gaits using 3D GRF signals,” Biomedical Signal Processing and Control, vol. 81, pp. 104448, 2023.

[3] Joel Kearney, Henrike Greaves, Mark A. Robinson, et al., ”Comparison of Theia3D and the conventional gait model in typically developing children and adults in a clinical gait laboratory,” Journal of Biomechanics, vol. 193, pp. 112995, 2025.

[4] Guillaume M. Meurisse, Dierick, et al., ”Determination of the vertical ground reaction forces acting upon individual limbs during healthy and clinical gait,” Gait & Posture, vol. 43, no. 1, pp. 245-250, 2016.

[5] Taylor P. Trentadue, Daniel Schmitt, ”Fourier Analysis of the Vertical Ground Reaction Force During Walking: Applications for Quantifying Differences in Gait Strategies,” Journal of applied biomechanics, vol. 40, no. 3, pp. 250-258, 2024.

[6] H. Debelle, C. Maganaris, T. O’Brien, ”Controlling the ground reaction force angle characterises better recovery following a single forward falling slip in young and older adults,” Gait & Posture, vol. 81, no. 1, pp. 77-78, 2020.

[7] R. R. Holt, D. Simpson, J. R. Jenner, et al., ”Ground reaction force after a sideways push as a measure of balance in recovery from stroke,” Clinical Rehabilitation, vol. 14, no. 1, pp. 88-95, 2000.

[8] Annchristin Andres, Michael Roland, Marcel Orth, et al., ”From Injury to Full Recovery: Monitoring Patient Progress Through Advanced Sensor and Motion Capture Technology,” Sensors (Basel, Switzerland), vol. 25, no. 13, pp. 3853, 2025.

[9] Kylee North, Grange Simpson, Walt Geiger, et al., ”Predicting the Healing of Lower Extremity Fractures Using Wearable Ground Reaction Force Sensors and Machine Learning,” Sensors (Basel, Switzerland), vol. 24, no. 16, pp. 5321, 2024.

[10] Ryan M. Nixon, Sharareh Sharififar, Matthew Martenson, et al., ”Ground Reaction Forces and Impact Loading Among Runners with Different Acuity of Tibial Stress Injuries: Advanced Waveform Analysis for Running Mechanics,” Bioengineering (Basel, Switzerland), vol. 12, no. 8, pp. 802, 2025.

[11] Laurent Malisoux, Paul Gette, Nicolas Delattre, et al., ”Spatiotemporal and Ground-Reaction Force Characteristics as Risk Factors for Running-Related Injury: A Secondary Analysis of a Randomized Trial Including 800+ Recreational Runners,” The American journal of sports medicine, vol. 50, no. 2, pp. 537-544, 2022.

[12] N. Hori, R. U. Newton, N. Kawamori, M. R. McGuigan, W. J. Kraemer, and K. Nosaka, ”Reliability of Performance Measurements Derived From Ground Reaction Force Data During Countermovement Jump and the Influence of Sampling Frequency,” J. Strength Cond. Res., vol. 23, no. 3, pp. 874-882, 2009.

[13] A. A. Nikooyan and A. A. Zadpoor, ”Effects of Muscle Fatigue on the Ground Reaction Force and Soft-Tissue Vibrations During Running: A

Model Study,” IEEE Trans. Biomed. Eng., vol. 59, no. 3, pp. 797-804, 2012.

[14] Thibault Marsan, Sacha Clauzade, Xiang Zhang, et al., ”REEV SENSE IMUs for Gait Analysis in Stroke: A Clinical Study on Lower Limb Kinematics,” Sensors (Basel, Switzerland), vol. 25, no. 16, pp. 5123, 2025.

[15] Yeqing Zhu, Xia, et al., ”Using Wearable Sensors to Estimate Vertical Ground Reaction Force Based on a Transformer,” Applied Sciences Basel, vol. 13, no. 4, pp. 2136, 2023.

[16] Tao Sun, Dongxuan Li, Bingfei Fan, et al., ”Real-time ground reaction force and knee extension moment estimation during drop landings via modular LSTM modeling and wearable IMUs,” IEEE journal of biomedical and health informatics, vol. 27, no. 7, pp. 1-12, 2023.

[17] Jianlin Ou, Yifeng Yan, Hui Ouyang, et al., ”Test-retest reliability and usability evaluation of a wearable plantar pressure monitoring system for linear and curved walking,” Scientific Reports, vol. 15, no. 1, pp. 37232, 2025.

[18] Dinghuang Zhang, Ying Liu, Timothy Exell, et al., ”Estimation of three-dimensional ground reaction forces using low-cost smart insoles,” Intelligent Sports and Health, vol. 1, no. 1, pp. 40-50, 2025.

[19] R. Fluit, M.S. Andersen, S. Kolk, et al., ”Prediction of ground reaction forces and moments during various activities of daily living,” Journal of Biomechanics, vol. 47, no. 10, pp. 2321-2329, 2014.

[20] Marion Mundt, William R. Johnson, Wolfgang Potthast, et al., ”A Comparison of Three Neural Network Approaches for Estimating Joint Angles and Moments from Inertial Measurement Units,” Sensors (Basel, Switzerland), vol. 21, no. 13, pp. 4535, 2021.

[21] Dimitrios-Sokratis Komaris, Eduardo Perez-Valero, Luke Jordan, et al., ”Predicting Three-Dimensional Ground Reaction Forces in Running by Using Artificial Neural Networks and Lower Body Kinematics,” IEEE Access, vol. 7, pp. 156779-156786, 2019.

[22] Aurelien Patoz, Thibault Lussiana, Bastiaan Breine, et al., ”Co´ mparison of different machine learning models to enhance sacral accelerationbased estimations of running stride temporal variables and peak vertical ground reaction force,” Sports Biomechanics, vol. 24, no. 4, pp. 825-841, 2025.

[23] Britney Kerr, Jessica Pascoe, Pam Thomason, et al., ”Effect of conventional gait model II variants on gait kinematics and kinetics,” Gait & Posture, vol. 122, pp. 190-198, 2025.

[24] F. Leboeuf, R. Baker, A. Barre, et al., ”Publisher note: The conventional´ gait model, an open-source implementation that reproduce the past but prepares for the future,” Gait & Posture, vol. 69, pp. 126, 2019.

[25] David A. Winter, Biomechanics and motor control of human movement, John Wiley, 2009.

[26] Eni Halilaj, Apoorva Rajagopal, Madalina Fiterau, et al., ”Machine learning in human movement biomechanics: Best practices, common pitfalls, and new opportunities,” Journal of Biomechanics, vol. 81, no. 1, pp. 1-11, 2018.