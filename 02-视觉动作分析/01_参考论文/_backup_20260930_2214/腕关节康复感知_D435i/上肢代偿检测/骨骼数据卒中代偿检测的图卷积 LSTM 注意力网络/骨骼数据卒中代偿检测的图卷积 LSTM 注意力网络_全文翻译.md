# 基于骨骼数据的脑卒中后代偿运动检测的图卷积长短期记忆注意力网络

Jiaxing Fan  
控制科学与工程学院  
山东大学  
中国济南  
202334915@mail.sdu.edu.cn  

Yang Zhang\*  
康复与物理治疗科  
山东中医药大学附属医院  
中国济南  
zhangyang982003@163.com  

Jiaojiao Liu  
康复与物理治疗科  
山东中医药大学附属医院  
中国济南  
1142365430@qq.com  

Xin Ma\*  
控制科学与工程学院  
山东大学  
中国济南  
maxin@sdu.edu.cn  

Wenkong Wang  
控制科学与工程学院  
山东大学  
中国济南  
202320760@mail.sdu.edu.cn  

Jichen Zhang\*  
山东浪潮科学研究院有限公司  
中国济南  
zhangjichen@inspur.com  

摘要——大多数脑卒中患者存在上肢运动功能障碍。代偿运动在康复训练中普遍存在，对患者的长期康复不利。因此，检测代偿运动具有重要意义。本研究提出了一种基于骨骼数据的图卷积长短期记忆注意力网络（Graph Convolutional Long Short-Term Memory Attention Network，GCN-LSTM-ATT），用于检测脑卒中后的代偿运动。研究选取了十六名脑卒中患者，使用 Kinect 深度相机采集患者进行特定康复运动时的骨骼数据。经过数据处理后，分别使用 GCN-LSTM-ATT 模型、支持向量机（Support Vector Machine，SVM）、K 近邻算法（K-Nearest Neighbor algorithm，KNN）和随机森林（Random Forest，RF）构建检测模型。结果表明，GCN-LSTM-ATT 模型的检测准确率达到 0.8580，显著高于传统机器学习算法。消融实验表明，模型的各个组件对性能的提升均有显著贡献。这些发现为脑卒中后代偿运动的检测提供了一种更精确且更有力的工具，有望促进脑卒中患者康复训练策略的优化。

关键词——脑卒中、骨骼数据、代偿运动检测、GCN-LSTM-ATT。

## I. 引言

中风是死亡率最高的疾病之一。高达80%的中风患者存在不同程度的上肢运动功能障碍，严重影响他们的日常生活活动[1]。中风后的康复训练对于患者恢复上肢运动功能至关重要。在康复过程中，为了实现预期的动作，患者经常改变运动模式，并使用其他关节或身体部位来代偿肢体运动功能的丧失，这是产生代偿运动的一个重要原因[2][3]。上肢的运动极其复杂。代偿运动主要包括躯干前倾（Trunk Leanforward, TLF）、躯干旋转（Trunk Rotation, TR）和肩部上抬（Shoulder Elevation, SE）[4]。随着康复的进行，这些代偿运动不利于患侧的正确和有效康复。在日常生活中，代偿运动有助于改善患者的短期预后。然而，长期的代偿运动会阻碍患侧恢复到中风前的运动状态，甚至可能导致其他疾病的发生[5]。因此，在患者的康复过程中检测这些代偿运动，并提供指导和提醒以减少此类代偿运动具有重要意义。

目前，临床上减少代偿运动的方法主要依赖于专业康复医生的监督、指导和纠正，这种方式既耗时又费力[6]。从医生的角度来看，他们工作负担重，时间和精力有限。迫切需要在康复过程中自动检测代偿运动，从而消除医生一对一人工辅助的必要性。从患者的角度来看，同样迫切需要在康复过程中自动检测代偿运动，尤其是对于那些正在进行居家康复治疗的患者而言。此类信息可以帮助医生和患者制定更具策略性的康复方案。

在当前研究中，先进设备和算法的应用已成为上肢代偿运动检测领域的一个重要方向。目前，该领域的主流检测方法依赖于惯性测量单元(IMU)[5][7]、压力垫传感器[8]、表面肌电(sEMG)传感器[9][10]、深度相机[11]以及Vicon光学跟踪系统[12]等设备。其中，基于压力的检测方法易受外部环境干扰，导致压力采集不稳定[8]。基于可穿戴设备的检测方案存在安装繁琐、稳定性差等问题[4]。此外，基于Vicon光学跟踪系统的检测方法面临成本高昂、需要标记工件以及操作流程复杂等挑战[12]。

相比之下，Kinect深度相机传感器因其独特的优势而脱颖而出。作为一种非侵入式、非穿戴式且更具成本效益的平台，Kinect传感器在人体运动检测领域受到了广泛关注[13]。它能够实现实时三维人体骨架跟踪，有效降低人体外观差异和光照变化对检测结果的影响。同时，深度学习算法能够准确分析Kinect采集的骨架数据，并精确识别相应的人体动作，为上肢代偿运动的检测提供了一种高效可靠的技术途径[14]。因此，基于骨架数据的代偿运动检测方法已成为一种可行的解决方案。本文的主要贡献如下：

1 、 提出了一种基于非接触式骨架数据的代偿运动检测方法，即图卷积长短期记忆注意力网络（Graph Convolutional Long Short-Term Memory Attention Network）。

2、通过对比实验验证，在代偿运动检测任务中，相较于现有的三种常用机器学习方法，本文所提深度学习算法具有更高的准确率，有效证明了所提方法的有效性。

## II. 方法

本研究提出了一种基于骨骼数据的代偿运动检测方法，该方法的具体流程结构如图1所示。首先，要求受试者在 Kinect 深度相机的监测下完成康复训练动作，以采集受试者的原始运动数据。随后，将采集到的数据依次进行预处理、骨骼点提取和特征提取等操作。最后，分别采用本研究提出的深度学习算法和三种已有的机器学习算法构建代偿运动检测模型，实现对代偿运动的准确识别与分析。在图1中，NC=非代偿（Noncompensation），TLF=躯干前倾（Trunk lean-forward），TR=躯干旋转（Trunk rotation），SE=肩部上提（Shoulder elevation）。

![](images/72c99f4fecb6000cfe914b53cfc79943c640b7b6c06709d7f8769a77979ebd19.jpg)  
图1. 工作流程结构

## A. 受试者

受试者为从山东中医药大学附属医院（Affiliated Hospital of Shandong University of Traditional Chinese Medicine）招募的15名脑卒中患者。实验前所有受试者均签署了书面知情同意书，相关流程和实验方案经山东中医药大学附属医院机构伦理委员会（Institutional Ethics Committee）批准。

脑卒中患者的纳入标准如下：1）符合脑卒中诊断标准，并经CT或MRI确诊；2）年龄在20至80岁之间；3）首次脑卒中发病超过15天且在6个月以内的患者；4）处于 Brunnstrom 分期 Ⅱ 期及以上；5）对研究方案有清晰理解并能够遵守执行（MMSE评分>24分）；6）自愿参与并签署知情同意书。

排除标准如下：1）存在严重认知障碍以至于无法完成 Fugl-Meyer 评定量表（Fugl-Meyer Assessment, FMA）测试；2）其他身体损伤，如骨折、严重关节炎等；3）心、肝、肾功能严重不全；4）研究人员认为不适合参与本实验的其他情况。

## B. 实验系统与流程

在数据集采集系统中，使用三台放置在不同位置的 Kinect 相机采集受试者的三维骨骼点数据。Kinect 以 320×288 像素的分辨率记录深度视频，工作频率为 30Hz。Kinect 相机的工作范围为 0.5-5.46m。三维骨骼数据通过相应的软件开发工具包（Software Development Kit, SDK）从深度视频中获取。数据采集设备布置的俯视图如图2所示。

![](images/4c7ede8dc7d39f3a813c1f06c7bccaaa800554b5d55b7f33c8688b65f0a60126.jpg)  
图2. 数据采集设备布置俯视图

采用多视角数据采集策略更符合代偿运动检测的实际需求。这种部署方式能够全面捕捉受试者的运动信息，有效避免单视角下可能出现的信息缺失问题。此外，实验要求每位脑卒中受试者在进行康复训练时达到其最大可能的活动范围，以便充分暴露潜在的代偿运动行为。关于受试者是否表现出代偿运动，本研究采用双专家独立评估机制。两位具有丰富临床经验的专家按照专业标准进行评估，通过交叉核对与讨论达成最终判断，从而确保评估结果的准确性与可靠性。

(1) 实验任务设计：基于临床康复医生的建议以及相关研究工作，本研究精心选取了三种在脑卒中患者日常康复训练中容易诱发代偿运动的标准康复动作。同时，确定了与之对应的三种常见代偿动作。这些动作的具体细节与图示分别如表 1 和图 3 所示，以明确展示实验设计的具体安排。所有动作均源自临床康复医生为脑卒中患者制定的常规康复训练方案。选取这些动作的原因在于它们在临床中具有较高的代偿运动发生率。

表 1. 标准康复动作与对应代偿动作
<table><tr><td></td><td>康复动作</td><td>对应代偿动作</td></tr><tr><td>1</td><td>触摸嘴巴</td><td>躯干前倾</td></tr><tr><td>2</td><td>向后伸展</td><td>躯干旋转</td></tr><tr><td>3</td><td>手臂外展</td><td>肩部上抬</td></tr></table>

![](images/524c151f1708039071abdd457c37cfd7abc3ba10a43663422dec9a3e5f0bfafc.jpg)  
图 3. 标准康复动作与对应代偿动作

(2) 实验设置：在本实验中，要求受试者坐于固定的圆形凳子上，通过三台 Kinect 深度相机从不同视角同步采集运动数据。受试者需依次完成三种康复动作，即触摸嘴巴、上肢外展 90 度以及向后伸展 45 度，以舒适的运动速度在其能够达到的最大活动范围内完成动作。在实验开始前，受试者将在专业医生的指导下充分熟悉实验流程与动作规范。

在实验过程中，每位受试者需将每个康复动作重复六次。单个动作之间间隔 8 秒，相邻动作之间设置 1 分钟休息时间，以避免疲劳因素对实验数据真实性的干扰。整个实验过程中，两位经验丰富的临床康复医生密切观察受试者，并实时记录三种常见代偿动作的发生情况：躯干前倾、躯干旋转和肩胛骨上提。

实验结束后，两位资深医生独立使用 Fugl-Meyer 运动功能评定量表的上肢部分（FMA-UE）对每位受试者的上肢运动功能进行评估。最终，取两位医生所给分数的平均值作为受试者上肢运动功能的量化评分。

## C. 骨骼数据与预处理

（1）骨骼数据：本研究使用 Azure Kinect Body Tracking SDK 从深度视频中提取 32 个骨骼关节点的三维运动数据（X、Y、Z）。在基于骨骼的上肢动作分析中，许多方法仅使用上肢关节的坐标作为输入。该策略旨在防止引入来自其他节点的冗余信息[15][16]。本研究中，骨骼数据由上肢 20 个关节的坐标组成。在空间维度层面，每一帧内的关节按照人体解剖结构有序连接，从而形成完整的上肢骨骼拓扑结构。具体连接方式如图 4 所示。

![](images/19b267ac62f67a18f1147899d4bdeca4c18f4bf44afdc619357748834cb9a5fd.jpg)  
图 4. 空间骨骼示意图

（2）数据预处理：首先，对原始动作序列的关键帧进行截取。计算相邻帧之间骨骼点位置变化的欧氏距离，并保留距离超过设定阈值的帧。随后，使用滑动窗口算法进行进一步处理。设定窗口大小和步长，去除窗口内相似且冗余的帧，从而保留完整有效的动作片段。针对动作序列时长差异的问题，本研究提出了基于最长序列的时间轴归一化策略，并结合三次样条插值技术，实现不同样本的精确时间对齐，确保所有数据在时间维度上的一致性。同时，通过对三维坐标进行 Z-score 归一化处理，进一步增强数据的可靠性。

## D. 图卷积长短期记忆注意力网络（GCN-LSTM-ATT）

本研究提出的图卷积长短期记忆注意力网络方法（Graph Convolutional Long Short-Term Memory Attention Network）是一种深度学习模型，有机融合了图卷积网络（Graph Convolutional Network, GCN）、长短期记忆网络（Long Short-Term Memory network, LSTM）以及注意力机制（Attention mechanism, ATT）的优势特性。该模型的概览如图5所示。

![](images/f959c460b56ac2bc8259182d1c0c27c8f486f3be4eae738b4279cfbc46116a1e.jpg)  
图5. 图卷积长短期记忆注意力网络

在结构原理方面，该模型将人体骨骼视为一种独特的图结构，并充分利用了图卷积网络（GCN）强大的图数据处理能力。在人体骨骼关节点的建模中，GCN能够有效地捕获和分析各个关节点之间复杂的空间依赖关系。通过对图结构中节点和边的运算，GCN能够提取关节点之间的空间关联特征，从而准确描述人体动作在空间维度上的特征分布[17]。

长短期记忆网络（LSTM）在该模型处理时间序列数据方面发挥着重要作用。在人体动作分析场景中，动作序列呈现出明显的时间动态变化特征。LSTM通过其独特的门控机制，能够有效地学习和记忆动作序列在不同时间步上的依赖关系[18]。无论是动作的起始、持续还是结束阶段，LSTM都能准确捕获时间序列中的关键信息，避免了传统循环神经网络（RNNs）中常见的梯度消失和梯度爆炸问题，从而能够对动作的时间动态进行稳定的建模。

注意力机制（ATT）的引入进一步增强了模型的性能。在处理动作序列时，每个时间步上的信息并非具有同等的重要性。ATT能够根据输入数据的特征自适应地调整对不同时间步的关注程度。具体而言，它能够智能地识别动作序列中的关键帧，并为这些关键帧分配更高的权重，从而突出关键信息的意义。同时，对于可能包含噪声或相对不重要信息的时间步，ATT会抑制其权重以降低噪声干扰的影响。通过这种方式，注意力机制显著提升了模型的判别能力，使其能够更准确地对动作进行分类和识别。

总之，通过将 GCN（Graph Convolutional Network，图卷积网络）、LSTM（Long Short-Term Memory，长短期记忆网络）和 ATT（Attention Mechanism，注意力机制）三者相结合，图卷积长短期记忆注意力网络（GCN-LSTM-ATT）不仅能够准确建模人体骨骼关节点之间的空间依赖关系，而且能够有效捕捉动作序列的时间动态变化。此外，该模型还可以自适应地增加关键帧的权重，抑制噪声干扰。它在人体动作分析领域展现出了强大的性能和潜力。

## E. 机器学习算法

总之，通过将 GCN、LSTM 和 ATT 三者相结合，图卷积长短期记忆注意力网络（GCN-LSTM-ATT）不仅能够准确建模人体骨骼关节点之间的空间依赖关系，而且能够有效捕捉动作序列的时间动态变化。此外，该模型还可以自适应地增加关键帧的权重，抑制噪声干扰。它在人体动作分析领域展现出了强大的性能和潜力。

## 1）SVM

支持向量机（Support Vector Machine，SVM）是一种经典的监督式机器学习算法。其核心思想在于寻找能够使类别间隔最大化的最优超平面，从而实现数据分类[19]。当数据线性可分时，可以直接求解该超平面；当数据线性不可分时，则利用核函数将数据映射到高维空间中进行处理。其优点包括：对于小样本具有较强的学习能力，对高维数据具有较好的适应性，以及出色的泛化性能。然而，其局限性如下：在处理大规模数据时计算复杂度较高；该算法最初是为二分类问题设计的，将其扩展到多分类问题相当繁琐；并且对数据的分布较为敏感。

## 2）KNN

K 最近邻算法（K-Nearest Neighbors，KNN）是一种简单直观的监督式机器学习方法。其核心逻辑如下：通过计算待分类数据点与训练集中每个点之间的距离（通常使用欧氏距离等度量方式），找出 K 个最近邻，并取这 K 个最近邻中出现频率最高的类别作为待分类数据点的类别。KNN 的优点是无需复杂的训练过程，且能够适应各种数据分布。然而，其缺点在于计算过程中需要遍历训练集，在数据量较大的场景中效率较低。此外，该算法极度依赖于 K 值的选择，不恰当的 K 值将显著影响分类精度[20]。

## 3）RF

随机森林（Random Forest，RF）是一种基于决策树构建的集成学习算法。它通过对训练集进行自助采样（bootstrap sampling）生成子数据集，为每个子数据集构建一棵决策树，在构建过程中随机选取部分特征来确定分裂点。在分类任务中，结果通过投票决定；在回归任务中，则取均值。其优点包括高精度、强鲁棒性、能够处理高维数据，并具有一定程度的可解释性。然而，其缺点是计算复杂度高且难以理解[21]。

## III. 实验结果

本节基于从真实受试者处采集的数据，首先将本研究提出的代偿运动检测方法与现有代表性算法在性能上进行对比。通过多指标量化评估，揭示所提方法在脑卒中后代偿运动检测方面的优势。此外，采用控制变量法对模型进行消融实验，分析各模块对性能的贡献，验证架构设计的科学性。

## A. 评价指标

本研究将 15 名脑卒中患者的实验数据按 8:2 的比例划分为训练集和测试集。为评估分类模型的性能，选取了四个关键指标进行量化分析：准确率（Accuracy）、精确率（Precision）、召回率（Recall）和 F1 分数（F1-score）。这些指标能够直观地揭示模型分类器的预测结果与真实标签之间的对应关系，从而对模型性能进行全面而精确的评估。具体计算公式如下：

$$
A c c u r a c y = { \frac { T P + T N } { T P + T N + F P + F N } }
$$

$$
P r e c i s i o n = \frac { T P } { T P + F P }
$$

$$
R e c a l l = \frac { T P } { T P + F N }
$$

$$
F 1 = \frac { 2 * P r e c i s i o n * R e c a l l } { P r e c i s i o n + R e c a l l }
$$

其中，TP、TN、FP、FN 分别表示分类标签的真正例（True Positive）、真负例（True Negative）、假正例（False Positive）和假负例（False Negative）。

## B. 对比实验

本研究设计的三个康复动作任务中各模型对代偿运动的检测性能详见表2。实验数据表明，基于传统机器学习（Machine Learning, ML）方法的检测准确率介于0.6481至0.7901之间。其中，支持向量机（Support Vector Machine, SVM）凭借其在非线性分类方面的优势取得了最佳性能。然而，本文提出的GCN-LSTM-ATT融合模型展现出了显著的性能优势，其检测准确率高达0.858。该结果充分验证了图卷积网络（Graph Convolutional Network, GCN）对关节空间拓扑结构的建模能力、长短期记忆网络（Long Short-Term Memory network, LSTM）对时间特征的捕获能力以及注意力机制（Attention mechanism, ATT）对关键时序特征的关注能力三者协同作用的有效性。该模型能够精确捕捉标准康复动作及其对应代偿动作中关节之间的动态拓扑关系，显著提升了模型对复杂代偿运动模式的识别准确率。

表2 各模型性能
<table><tr><td>模型</td><td>准确率</td><td>精确率</td><td>召回率</td><td>F1值</td></tr><tr><td>SVM</td><td>0.7901</td><td>0.8267</td><td>0.7901</td><td>0.7887</td></tr><tr><td>KNN</td><td>0.6481</td><td>0.6562</td><td>0.6481</td><td>0.6459</td></tr><tr><td>RF</td><td>0.6852</td><td>0.7599</td><td>0.6852</td><td>0.6903</td></tr><tr><td>GCN-LATM-ATT</td><td>0.8580</td><td>0.8695</td><td>0.8580</td><td>0.8603</td></tr></tr>

## C. 消融实验

基于实际采集的数据，对所提出的GCN-LSTM-ATT各模块进行消融研究，以评估三个核心组件——即GCN（作为基线）、LSTM和ATT——对模型性能的影响。首先，对GCN的输出特征进行时空平均（spatiotemporal averaging）操作，并将结果输入分类器，将其视为基线。相关实验结果如表3所示。

如表3所示，与基线GCN相比，GCN-LSTM-ATT在准确率上取得了显著提升，验证了所提模型在脑卒中后（post-stroke）代偿运动检测任务中对关节空间拓扑结构进行建模、捕获时间特征以及聚焦关键时间特征方面的巨大价值。此外，与基线GCN相比，GCN-LSTM模型在准确率上也取得了相当大的提升，这验证了将空间研究与时间研究相结合对于提高代偿运动检测准确率的重要性。

表3 消融实验结果

<table><tr><td>模型</td><td>准确率</td><td>精确率</td><td>召回率</td><td>F1值</td></tr><tr><td>GCN</td><td>0.5679</td><td>0.6841</td><td>0.5679</td><td>0.5607</td></tr><tr><td>GCN-LSTM</td><td>0.8457</td><td>0.8592</td><td>0.8457</td><td>0.8462</td></tr><tr><td>GCN-LATM-ATT</td><td>0.8580</td><td>0.8695</td><td>0.8580</td><td>0.8603</td></tr></table>

## IV. 结论

本研究基于骨骼数据构建了一种新颖的深度学习模型，用于检测脑卒中后的代偿性运动。首先，设计了三项创新性的康复运动任务，并采集了脑卒中患者在执行这些任务时的骨骼数据，为后续研究提供了基础数据支持。随后，将本研究提出的深度学习模型与三种现有的机器学习算法进行了对比实验，建立了代偿性运动检测模型，以实现对代偿性运动的高精度识别。

实验结果表明，GCN-LSTM-ATT 模型在基于骨骼数据的脑卒中后代偿性运动检测任务中显著优于其他对比模型。尽管当前研究的数据集规模相对有限，但这一成果为后续研究提供了重要的理论基础和技术参考。同时，它增强了模型在临床应用中的可解释性。

未来的研究将着重于扩大数据集规模、优化模型架构，并进一步探索脑卒中患者各种代偿性运动模式的检测。旨在开发更加高效、智能的辅助诊断工具，有效减轻临床医生的工作负担，推动脑卒中康复诊疗领域的智能化进程。

## 参考文献

[1] Beilei L ,Zhenxiang Z ,Yongxia M , et al.Cumulative risk of stroke recurrence over the last 10 years: a systematic review and metaanalysis.[J].Neurological Sciences : Official Journal of the Italian Neurological Society and of the Italian Society of Clinical Neurophysiology,2020,42(1):61-71.

[2] A B V ,N M S G ,Machiel F H L D V .Trunk Compensation During Bimanual Reaching at Different Heights by Healthy and Hemiparetic Adults.[J].Journal of Motor Behavior,2017,49(5):580-592.

[3] Levin F M ,Liebermann G D ,Parmet Y , et al.Compensatory Versus Noncompensatory Shoulder Movements Used for Reaching in Stroke[J].Neurorehabilitation and Neural Repair,2016,30(7):635-646.

[4] Xu P, Xia D, Zheng B, et al. A novel compensatory motion detection method using multiple signals and machine learning[J]. IEEE Sensors Journal, 2022, 22(17): 17162-17172.

[5] Ding K, Wang J, Wang X, 等. 基于可穿戴传感器和机器学习算法的中风后患者代偿运动检测与定量评估系统[J]. *IEEE Sensors Journal*, 2024.

[6] Cai S, Li G, Su E, 等. 实时检测中风患者代偿模式以减少机器人康复治疗中的代偿[J]. *IEEE Journal of Biomedical and Health Informatics*, 2020, 24(9): 2630-2638.

[7] Ranganathan R, Wang R, Dong B, 等. 使用可穿戴传感器系统识别上肢代偿运动模式[J]. *Physiological Measurement*, 2017, 38(12): 2222.

[8] Cai S, Li G, Zhang X, 等. 使用压力分布数据和机器学习算法检测中风幸存者的代偿运动[J]. *Journal of Neuroengineering and Rehabilitation*, 2019, 16: 1-11.

[9] Chen Y, Ma K, Yang L, 等. 使用生成对抗网络的躯干代偿肌电特征净化与分类模型[J]. *Biomedical Signal Processing and Control*, 2021, 65: 102345.

[10] Ma K, Chen Y, Zhang X, 等. 康复训练中基于表面肌电信号（sEMG）的躯干代偿检测[J]. *Frontiers in Neuroscience*, 2019, 13: 1250.

[11] Luo Z, Lim A E P, Durairaj P, 等. 面向社区居住中风老年人上肢康复的代偿感知虚拟康复系统开发[J]. *Journal of NeuroEngineering and Rehabilitation*, 2023, 20(1): 56.

[12] Van Remoortel H, Giavedoni S, Raste Y, 等. 健康与慢性病中活动监测仪的有效性：系统综述[J]. *International Journal of Behavioral Nutrition and Physical Activity*, 2012, 9: 1-23.

[13] Tian H, Li H, Jiang W, 等. 基于骨骼的帕金森步态 MDS-UPDRS 评分估计的跨时空图卷积网络[J]. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 2024, 32: 412-421.

[14] Moon S, Park Y, Ko D W, 等. 使用卡尔曼滤波的多 Kinect 传感器融合用于人体骨骼追踪[J]. *International Journal of Advanced Robotic Systems*, 2016, 13(2): 65.

[15] Yan S, Xiong Y, Lin D. 用于基于骨骼的动作识别的时空图卷积网络[C]. *Proceedings of the AAAI Conference on Artificial Intelligence*. 2018, 32(1).

[16] Zhou J, Cui G, Hu S, 等. 图神经网络：方法与应用综述[J]. *AI Open*, 2020, 1: 57-81.

[17] Wu Z, Pan S, Chen F, 等. 图神经网络全面综述[J]. *IEEE Transactions on Neural Networks and Learning Systems*, 2020, 32(1): 4-24.

[18] Liu J, Shahroudy A, Xu D, 等. 使用带信任门的时空 LSTM 网络进行基于骨骼的动作识别[J]. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2017, 40(12): 3007-3021.

[19] Sunnetci K M, Alkan A. 基于双相多数投票的胸部 X 光图像 COVID-19 比较诊断[J]. Expert Systems with Applications（专家系统与应用）, 2023, 216: 119430.

[20] Cunningham P, Delany S J. K 近邻分类器——教程[J]. ACM Computing Surveys (CSUR)（ACM 计算综述）, 2021, 54(6): 1-25.

[21] Biau G, Scornet E. 随机森林导览[J]. Test, 2016, 25(2): 197-227.