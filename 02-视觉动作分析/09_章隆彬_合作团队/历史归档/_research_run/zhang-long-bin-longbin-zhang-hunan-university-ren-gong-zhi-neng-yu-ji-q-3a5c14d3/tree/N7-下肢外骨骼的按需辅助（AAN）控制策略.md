# N7 下肢外骨骼的按需辅助（AAN）控制策略

> 核心问题：个体化关节力矩预测如何转化为下肢可穿戴外骨骼的 assist-as-needed 控制器，实现卒中/脊髓损伤患者的神经康复？

## S1 原始资料依据

### 来源01：Zhang2024-TBME（IEEE Transactions on Biomedical Engineering）——标定型 EMG-informed 神经肌肉骨骼建模

**模型理论基础**：Hill-type 肌肉模型由收缩元件（contractile element, CE）、串联弹性元件（series elastic element, SE）和平行弹性元件（parallel elastic element, PE）构成，用于描述肌肉的力–长关系（force-length relationship）、力–速关系（force-velocity relationship）以及肌腱力–长关系（tendon force-length relationship）（来源01）。其四个核心参数为：最大等长力 F_max（maximum isometric force）、最优纤维长度 l_M,opt（optimal fiber length）、肌腱刚度 k_T（tendon stiffness）、肌腱松弛长度 l_TS（tendon slack length）（来源01）。该模型由 De Groote 等人提出（来源01）。

**肌冗余问题与求解器**：人体肌肉–骨骼系统存在典型的肌冗余问题（muscle redundancy problem），需要通过直接配点最优控制方法（direct collocation optimal control methods）求解（来源01）。该研究采用"肌冗余求解器"（muscle redundancy solver / muscle redundancy solver algorithm），输入为 EMG 和肌纤维长度（muscle fiber length），可同时估计 GAS（腓肠肌）、SOL（比目鱼肌）和 TA（胫骨前肌）三个踝关节肌肉的参数（来源01）。该求解器在性能上优于传统的基于身高体重的解剖缩放法（anthropometric scaling）（来源01），可估计受试者特异的肌腱参数（来源01）。

**标定型 EMG-informed 神经肌肉骨骼建模框架（calibrated EMG-informed neuromusculoskeletal modelling）**：将 EMG 驱动建模（EMG-driven modeling）与超声（Ultrasound）成像相结合（来源01）。其中：
- EMG 由 Noraxon Inc.（位于美国 AZ）制造（来源01），用于测量肌肉激活（来源01）；
- 超声使用深圳迈瑞 Mindray M9（来源01），结合 UltraTrack 软件（来源01）追踪肌纤维长度；
- 运动学采用瑞典哥德堡 Qualisys 动捕系统采集（来源01）；
- 等速测力计（Dynamometer）测量踝关节角度与关节力矩（来源01）。
整体框架在通用肌肉骨骼模型 gait10dof18musc（或称 gait10musc18，部署于 OpenSim）基础上进行缩放与个体化标定（来源01）。

**最优控制问题的代价函数结构**：最优控制问题（optimal control problem）以肌肉激活（muscle activations）和肌腱力（muscle tendon force）为状态量（来源01），以肌肉激励（muscle excitations）和储备致动器力矩（reserve actuator torques）为控制量（来源01）。代价函数 J 由四项加权组成（来源01）：
1. **EMG 项**：度量肌肉激励与归一化 EMG 信号的差；
2. **纤维长度项**：度量仿真与实测肌纤维长度的差；
3. **努力项（effort term）**：正则化求解过程；
4. **储备致动器项（reserve actuator term）**：约束踝关节的额外力矩贡献。
权重因子（weight factors）通过权重调整方法（weight-tuning methodology）设定为：EMG 项 = 1，纤维长度项 = 1，努力项 = 0.1，储备致动器项 = 10（来源01）。该权重配置体现"尽量让 EMG 解释运动，让储备动器填补残余力矩"的设计思路，即"按需"的本质——只补充肌肉不足的部分。

**储备致动器（reserve actuator）机制**：在踝关节额外添加储备致动器（来源01），通过力臂矩阵 R（moment-arm matrix R）描述肌腱动力学（muscle-tendon dynamics）（来源01），R 取决于关节角度（来源01），并与力臂矩阵相乘得到关节力矩估计（来源01）。储备致动器提供额外关节力矩（来源01），其权重（10）远高于其他项，说明求解器倾向于最小化外骨骼/外力介入，仅在必要时介入。

**直接配点转录与求解**：直接配点（direct collocation）将最优控制问题转录（transcribe）为非线性规划（NLP）（来源01），采用梯形积分方案（trapezoidal integration scheme）（来源01），通过 IPOPT 求解 NLP（来源01），并借助 CasADi 进行自动微分（automatic differentiation）（来源01）。

**实验采集与数据处理**：使用 Butterworth 滤波器处理信号（来源01）；以最大自主收缩（Maximum Voluntary Contraction, MVC）归一化 EMG（来源01）；踝关节力矩在 5° 背屈、0° 跖屈、10° 跖屈、20° 跖屈四个角度下测量（来源01）；采用 Shapiro-Wilk 检验估计参数的正态性（来源01）；通过交叉验证（cross-validation）评估优化性能（来源01）。

**卒中后肌肉改变与个体化模型意义**：卒中后个体具有更低的 F_max、k_T 与 l_M,opt（来源01），其肌肉–骨骼力学特性与健康对照差异显著（来源01）。该研究通过比较个体化模型（individualized model）与通用模型（generic model）发现：在踝关节力矩估计上，个体化模型具有更低的 RMSE（来源01）。这表明仅靠通用模型难以准确预测卒中患者的关节力矩——必须依赖个体化标定。这也是外骨骼 AAN 控制器所需关节力矩预测准确性的前提。

### 来源02：Zhang2023-TNSRE（IEEE Transactions on Neural Systems and Rehabilitation Engineering）——基于可穿戴传感器与 LSTM/NMS 模型的关节力矩预测

**研究动机**：按需辅助外骨骼控制器（assist-as-needed exoskeleton controllers）的设计目标为辅助肌肉产生的力矩（assist muscle-generated torque），同时确保任务成功完成（ensure successful task performance）（来源02）。其依赖的核心信息为关节力矩（joint torque），由神经肌肉骨骼（NMS）模型与长短期记忆网络（LSTM）两类方法预测（来源02）。

**多关节覆盖**：研究覆盖下肢 8 个关节力矩：踝背屈/跖屈（ankle dorsiflexion/plantarflexion）、膝屈/伸（knee flexion/extension）、髋屈/伸（hip flexion/extension）、髋外展/内收（hip abduction/adduction）（来源02），适用于多关节外骨骼控制器设计。

**方法学组合**：关节力矩的获取途径包括逆动力学（inverse dynamics）（来源02）、人工神经网络（ANN）、多层感知器（MLP）与 LSTM（来源02）。逆动力学提供参考真值标签，LSTM 与 NMS 模型作为可部署于外骨骼在线控制器中的预测器。

**聚类分析改进估计精度**：聚类分析（cluster analysis）基于特征相似度（characteristic similarity）将踝背屈/跖屈的运动模式分组（来源02）。研究对比了三种建模策略（来源02）：
- **Pooled models（池化模型）**：使用全部 10 种运动数据训练，对所有运动模式过于通用（too generic），估计精度受限；
- **Individual models（个体模型）**：对每种运动单独建模，估计性能良好（good estimation performance）；
- **Clustered models（聚类模型）**：对每个聚类单独建模，估计性能良好。
聚类分析同时改进了 NMS 模型与 LSTM 网络的估计精度（来源02）。

**临床应用场景**：研究明确面向脊髓损伤（spinal cord injury）与卒中后（post-stroke）患者的神经康复（neurorehabilitation）与神经肌肉恢复（neuromuscular recovery）（来源02），覆盖运动障碍（motor disorders）人群（来源02），所使用的可穿戴传感器（wearable sensors）使在线、临床环境部署成为可能（来源02）。

## S2 核心观点提炼

1. **AAN 控制器的物理本质是"残差补偿"**：储备致动器项权重（10）远高于其他项（EMG=1、纤维长度=1、努力=0.1），意味着模型优先让 EMG 解释关节力矩、仅用储备项填补残余——这正是按需辅助"只补不足"的数学体现（来源01）。

2. **个体化肌腱参数标定是 AAN 控制器的前提**：卒中患者 F_max、k_T、l_M,opt 显著低于健康人，使用通用模型估计关节力矩会产生较大 RMSE 误差；个体化模型的 RMSE 显著降低（来源01）。若 AAN 控制器依据通用模型给出的力矩预测，患者实际能力将被高估或低估，辅助量将偏离"按需"原则。

3. **个体化 NMS 模型 + LSTM 构成 AAN 控制器的双预测路径**：关节力矩可由 NMS 模型（物理可解释、依赖个体化参数）或 LSTM（数据驱动、依赖可穿戴传感器）预测（来源02）。两条路径为 AAN 控制器提供了"模型驱动 + 数据驱动"互补的工程实现基础。

4. **聚类建模策略在 AAN 控制器中具有关键作用**：池化模型"对所有运动模式过于通用"，而个体模型与聚类模型均显著提升估计性能（来源01中个体化模型表现、来源02中 clustered models 表现）。对于外骨骼在线运行，可在离线阶段为患者建立聚类专属模型，运行时按当前运动模式匹配——这是把"个体化"从单一个体扩展到"个体–运动模式"二维细化的关键。

5. **多关节 AAN 控制需要多关节力矩预测支撑**：控制器需同时驱动踝、膝、髋多个关节（来源02），力矩预测必须多关节化。研究通过 LSTM + NMS 的多输出建模实现 8 个关节力矩的同时预测，为多关节 AAN 控制器提供完整力矩曲线（来源02）。

6. **AAN 控制器的最终临床目标**：保证任务成功完成的同时（来源02），最大化患者主动肌肉贡献（通过储备致动器最小介入实现，来源01），以促进神经肌肉恢复（来源02）——这是卒中、脊髓损伤神经康复中"运动再学习"理论的工程化落地。

## S3 与其他节点的关系

- **与"神经肌肉骨骼建模的理论基础与 Hill-type 肌肉模型"节点**：AAN 控制器的物理基础直接依赖 Hill-type 肌肉模型（CE/SE/PE 三元件、力–长/力–速关系，来源01），后者为个体化模型提供数学骨架。
- **与"基于 EMG + 超声的个体化肌肉-肌腱参数标定"节点**：该节点为 AAN 控制器提供校准后的 F_max、l_M_opt、k_T、l_TS 四参数（来源01）。标定的精度通过代价函数中的 EMG 项与纤维长度项约束（来源01），并通过交叉验证评估（来源01）。
- **与"卒中后踝关节肌-腱力学改变的量化与对照研究"节点**：该节点揭示卒中后 F_max、k_T、l_M,opt 下降（来源01），为 AAN 控制器"为什么必须个体化"提供临床依据。
- **与"基于 LSTM 与 NMS 模型的多关节关节力矩预测"节点**：该节点直接产出控制器所需的关节力矩时间序列，并通过聚类分析提升精度（来源02），是 AAN 控制器的"感知–决策"层输入。
- **与"基于聚类分析的个体化运动模式建模"节点**：聚类模型是 AAN 控制器在多运动模式下的鲁棒性保障，避免池化模型"过于通用"导致的力矩预测偏差（来源02）。
- **与"章隆彬与湖南大学人工智能与机器人学院的实验室定位"节点**：AAN 控制策略的工程化（外骨骼软硬件、可穿戴传感器）依托实验室平台，是学术成果转化的出口。
- **与"多模态实验采集与跨地域合作网络"节点**：EMG（Noraxon，美国 AZ）、超声（Mindray M9，深圳）、动捕（Qualisys，瑞典哥德堡）的多模态采集（来源01）以及跨地域合作网络，为 AAN 控制器的数据采集与算法验证提供基础设施。

## S4 在主线中的位置

本节点是知识树"个体化建模 → 力矩预测 → 控制器 → 临床康复"主线的**集成层（控制层）**，处于力学建模/参数标定（上游 N2、N4）与临床康复目标（下游主线终点）之间的关键转化节点：

- **上游依赖**：吸收 Hill-type 建模理论（N2）、EMG + 超声参数标定（N4）、卒中后肌-腱力学改变（N5）的成果，将其封装为可在线运行的力矩预测器。
- **下游输出**：将 LSTM/NMS 多关节力矩预测（N6）+ 聚类建模（N7'）的输出转化为"残差补偿型"控制指令，使外骨骼在最小介入原则下辅助卒中/脊髓损伤患者完成行走，并触发神经可塑性。
- **闭环定位**：AAN 控制器既是"建模-标定-预测"研究链的工程终点，也是临床康复的起点；其"按需"思想既是 EMG-informed 个体化建模的内在逻辑（储备致动器最小化），也是神经康复"主动参与促进恢复"原则的工程体现。

简言之，本节点回答了核心问题：如何把"个体化力矩预测"转化为"按需辅助"——通过**储备致动器残差补偿机制 + 个体化肌腱参数 + 聚类/多关节 LSTM/NMS 力矩预测**，最终实现"任务完成"与"主动肌肉贡献最大化"的双重目标，支撑卒中/脊髓损伤患者的神经康复。