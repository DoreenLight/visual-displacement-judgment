# OPENCAP MONOCULAR：从单段智能手机视频中获取人体运动动力学

Scott D. Uhlrich<sup>1,2\*</sup>、Shardul Sapkota<sup>1</sup>、Antoine Falisse<sup>1</sup>、Scott L. Delp<sup>1</sup> <sup>1</sup>斯坦福大学（Stanford University），加利福尼亚州斯坦福；<sup>2</sup>犹他大学（University of Utah），犹他州盐湖城 \*通讯作者邮箱：suhlrich@stanford.edu

引言：能够便捷地测量人体运动的运动学与动力学特征，有助于改善神经系统和肌肉骨骼疾病的预防与治疗。基于标记点的动作捕捉所需的时间和专业门槛，限制了其在临床以及大规模研究中的应用。我们近期开发了 OpenCap——一款利用两台智能手机视频来估计运动学与动力学的开源软件 [1]。OpenCap 及其他无标记动作捕捉技术 [2] 将运动分析在时间和成本上的门槛降低了数个数量级 [1]；然而，这些方法均需要多台相机。如果我们能够基于单段视频达到相近的精度，那么全球超过 40 亿的智能手机用户都将能够使用三维动作捕捉。计算机视觉领域在从单段视频估计人体网格的全局姿态方面已取得诸多进展 [3]，但这些算法并非针对其在生物医学中的应用而设计，也未在生物医学效用上加以评估。本研究旨在开发并评估一套从单段视频估计常见人体运动（行走与下蹲）运动学与动力学的流程。

方法：在此我们介绍 OpenCap Monocular（OpenCap 单目方案），它结合了三维人体姿态估计、生物力学建模以及动态肌肉骨骼仿真，能够从单目视频中估计运动学与动力学参数（图 1a）。我们在一个被试（一名 33 岁女性）上，针对行走与下蹲试验对这一流程进行了评估，试验中同步采集了视频（与正前方成 45°）、基于标记点的动作捕捉数据以及测力台数据 [1]。首先，我们使用 WHAM [3]——一种当前最先进的基于回归的三维姿态估计算法——估计足部接触概率以及 SMPL 模型 [4] 的全局姿态。以 WHAM 的结果作为初始猜测，我们进行两阶段优化以求解相机参数与人体全局姿态。第一阶段优化在固定相机且已知被试身高的前提下，通过最小化重投影误差来求解相机参数与 SMPL 形状参数（即人体测量学参数）。在固定相机与 SMPL 参数之后，第二阶段优化通过最小化重投影误差、接触期间足部位置方差以及关节速度来求解 SMPL 全局姿态。随后，我们从 SMPL 顶点中提取解剖学标志点，对肌肉骨骼模型 [1] 进行缩放，并使用 OpenSim 4.4 [5] 运行逆向运动学。我们生成一个肌肉驱动的动态仿真来跟踪运动学以估计地面反作用力 [1,6]。为评估精度，我们计算了 OpenCap Monocular 与基于标记点的动作捕捉及测力台之间的平均绝对差，并在各试验间取平均。

图 1：（a）用于从视频中估计运动学与动力学参数的 OpenCap Monocular 流程。（b）在基于回归的姿态估计之后进行姿态优化，可减少五次重复下蹲过程中的漂移（平移误差由 96 mm 降低至 46 mm）。

![](images/33cf4b02423e9a29f52ffb49a20e4df349e8276718f6e05c4dd8050e7b33fafe.jpg)  

结果与讨论：对于行走和下蹲动作，OpenCap Monocular 相对于基于标记点的运动捕捉系统和测力台，在运动学参数上的平均绝对差异分别为 6.6° 和 37 mm（骨盆位移），反作用力方面则为 7.7% 体重（表 1）。行走时的关节角度差异（5.6°）与双相机 OpenCap 系统（5.2°）、八相机无标记点运动捕捉系统（5°）[2] 以及 17 传感器 IMU 套件（5°）[7] 相当；然而，位移差异（27 mm）大于双相机 OpenCap（7 mm）。行走时的地面反作用力精度（内外侧方向：1.3% 体重，前后方向：4.9%，垂直方向：21.5%）劣于双相机 OpenCap（1.5–11.1% 体重）[1] 和 IMU（1.7–9.3% 体重）[8]，这可能是由于骨盆位移误差较大所致。姿态优化减少了基于回归的姿态估计模型所引起的漂移（图 1b）。

表格 1：OpenCap 与基于标记点的运动捕捉之间的平均绝对差异。运动学差异在 24 个旋转自由度和 3 个平移自由度上取平均值，地面力在 3 个方向上取平均值。

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Kinematics (rotations, °)</td><td rowspan=1 colspan=1>Kinematics (translations, mm)</td><td rowspan=1 colspan=1>Ground forces (% bodyweight)</td></tr><tr><td rowspan=1 colspan=1>Opencap Monocular</td><td rowspan=1 colspan=1>6.6</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>7.7</td></tr><tr><td rowspan=1 colspan=1>OpenCap two-camera</td><td rowspan=1 colspan=1>4.9</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>4.2</td></tr></table>

**意义（Significance）** OpenCap 单目（Monocular）在两项活动中对单个个体的运动学（kinematics）与动力学（kinetics）参数进行估计，其准确度与双摄像头 OpenCap 相似但略差；仍需在不同个体与不同活动之间进行进一步验证。基于单一视频的生物力学分析将极大地扩展我们在家庭、临床与现场中量化运动的频率与便捷程度。在生物力学评估的背景下进行严谨的分析，是这一扩展过程中的必要环节。

**参考文献（References）** [1] Uhlrich et al. (2023), Plos Comput Biol 19(10). [2] Kanko et al. (2021), J Biomech 127. [3] Shin et al. (2024), Proc CVPR. [4] Loper et al, (2015), Proc SIGGRAPH Asia. [5] Seth et al. (2018) Plos Comput Biol 14(7). [6] Falisse et al. (2019) J R Soc Interface 16. [7] Al Borno et al. (2022), J Neuroeng Rehabil 19(11). [8] Karatsidis et al. (2019), Med Eng Phys 65.