# 使用单个 RGBD 相机进行实时、准确、开源的上肢肌肉骨骼分析

Amedeo Ceglia<sup>1,\*</sup>, Kael Facon<sup>2</sup>, Mickael Begon ¨ <sup>1,3</sup>, and Lama Seoud<sup>4</sup>

<sup>1</sup>蒙特利尔大学生物医学工程研究所，加拿大 魁北克省 蒙特利尔

<sup>2</sup>巴黎信息工程师学校，法国 Le Kremlin-Bic ´ etre ˆ

<sup>3</sup>蒙特利尔大学运动机能学与人体动力学学院，加拿大 魁北克省 蒙特利尔

<sup>4</sup>蒙特利尔理工学院计算机工程与软件工程系，加拿大 魁北克省 蒙特利尔

通讯作者：Amedeo Ceglia, amedeo.ceglia@umontreal.ca

## 摘要

生物力学反馈可促进康复，并为临床医生提供更客观的任务评估。这类反馈通常依赖于昂贵的运动捕捉系统，从而限制了其广泛应用，并推动了基于计算机视觉的方法的发展。然而，这些方法在上肢方面存在较大的关节角度误差，且在分析中忽略了肩胛骨和锁骨的运动。本文提出了一种开源方法，使用单个低成本的 RGBD（彩色图像加深度图像）相机，提供易于使用的高保真上肢运动学分析方案，并包含半自动的皮肤标记点标注。为展示该方法在上肢分析中的适用性，我们对八名受试者在执行手摇车运动时进行了实时生物力学分析（涵盖从运动学到肌肉力估计）。分别使用 RGBD 相机和光电（光电子）相机系统采集标记点数据，其中光电系统作为参考；使用八个表面肌电（EMG）通道和带有仪器测量的手踏板分别记录肌肉活动和外部载荷。Bland-Altman 分析显示，两种运动捕捉方法在三维标记点位置上具有显著一致性，平均误差为 3.3±3.9 mm。在生物力学分析方面，一致性水平对是否使用相同的标记点集较为敏感。例如，使用相同标记点集时关节角度差异平均为 2.3±2.8°，而使用不同标记点集时则为 4.5 pm 2.9°。RGBD 相机的生物力学反馈帧率可达 63 Hz。本研究提出了一种将 RGBD 相机用作低成本运动捕捉解决方案的新方法，强调了其在精确运动学重建和综合上肢生物力学研究方面的潜力。

**关键词** —— 运动捕捉、计算机视觉、RGBD 相机、上肢生物力学

## 1 引言

实时生物力学反馈对改善或指导康复过程具有高度相关性 [GPC13]。反馈内容可包括关节运动学 [BBC<sup>+</sup>16]、关节动力学 [FMH<sup>+</sup>04] 或肌肉活动 [Bol03]。关节运动学通常是后续生物力学分析所必需的，并依赖于基于运动捕捉数据的快速而直接的算法（例如逆运动学）。

动作捕捉通常基于光电摄像系统，用于追踪放置在骨性标志点上的反光或主动标记物。该技术精度很高，标记位置误差小于一毫米 [TR20]，因此在临床应用中具有很强的鲁棒性，并被用作评估新兴动作捕捉技术的参考。然而，由于成本高昂和设置复杂，在临床或运动环境中使用光电系统仍然具有挑战性。因此，研究人员开发了经济实用的替代系统以解决这些局限性。在这些替代系统中，有三种技术最为常用。

第一种技术基于放置在身体节段预定位置上的惯性测量单元（IMU）[NWK20]。IMU 的优势在于几乎可以与任何衣物兼容使用，并且适用于在有限空间的环境中应用。然而，陀螺仪在长时间任务中可能出现漂移，磁力计的电磁场也可能受到干扰，导致与基于标记物的系统相比，下肢关节角度估计误差约为 10° [SDLC22]，上肢可达 30° [HKWE22]。另一个局限在于上肢模型过于简化，仅采用单一的胸肱关节，而未对胸锁关节、肩锁关节和盂肱关节进行建模。因此，具有上肢康复重要意义的肩胛骨 [VT00] 被忽略。

第二种技术采用一组经过同步和标定的 RGB 摄像机，结合深度学习算法来无标记地识别三维人体姿态。该算法通过从每个标定摄像机提供的二维姿态推断三维姿态，检测代表关节中心的一系列关键点 $[ \mathrm{LMD}^+22]$。随后利用这些关键点的三维位置计算关节角度。这类无标记系统减少了实验时间，且侵入性较低，因为可以在穿着衣物的状态下使用。然而，与基于标记物的系统类似，基于摄像头的系统实施成本较高，需要较大空间，并且难以实现精确的标定。所有摄像机必须进行同步和联合标定，因为最终的三维姿态精度依赖于多视角几何。此外，姿态估计对计算资源要求较高，需要高性能 GPU 硬件，因此不适合实时生物反馈或嵌入式系统。该方法虽前景良好，但依赖于标注图像，而用户对这些标注的可靠性了解有限。根据具体应用场景，其精度可能不足，特别是对于手臂的内/外旋，误差可达 $23^{\circ}$ $[\mathrm{LMD}^+22]$。此外，与基于 IMU 的技术类似，模型仅包含简化的胸肱关节（排除了锁骨和肩胛骨）。

最后一种技术基于单个相机（是否具备深度测量均可）结合机器学习，通过预训练数据来定位关节中心并计算关节角度，如文献 [BDMM<sup>+</sup>24] 所示。由于仅使用一台相机，因此无需外部相机标定或同步，使其成为一种经济实惠且用户友好的方法。该技术与前述方法存在类似的缺陷，包括标注图像的不准确性以及关节中心定位的误差，可能导致上肢运动的误差高达 $2 5 ^ { \circ }$ [SBS<sup>+</sup>19]。此外，当无法获取深度测量时，输入信息（二维图像）十分有限，算法必须做出诸多假设来推断三维姿态估计。与前两种技术一样，该模型同样对肩关节进行了简化。鉴于上述情况，一些方法（例如使用光深度相机（RGBD）的方法）已展现出提供低成本实时基于标记的运动捕捉系统的良好前景。值得注意的是，据我们所知，这些方法尚未应用于上肢的生物力学分析。在 [YYS<sup>+</sup>16] 中，观察到使用红外图像和反光标记会降低标记周围深度信息的精度；而在 [TCF<sup>+</sup>18] 中，算法的鲁棒性要求使用大半径的球形标记，限制了其在上肢运动捕捉中的适用性。本研究中，我们提出了一种开源且用户友好的方法，使用一台低成本 RGBD 相机和 13 个扁平白色皮肤标记来记录上肢运动学。我们的主要目标是开发一套简单、快速且经济实惠的系统，用于在临床环境中获取运动学数据。我们实现了半自动标记标定算法，并基于肌肉骨骼模型构建了生物力学框架。我们将所提方法与基于标记的方法（视为参考标准）进行了对比。评估内容涵盖标记的三维位置、生物力学结果以及时间性能。目的在于为上肢运动捕捉提供一个综合解决方案，在精度与生态可用性之间取得平衡。

## 2 方法

## 2.1 数据采集

8 名健康受试者（3 名女性；年龄：$2 5 . 9 \pm 1 2 . 2$ 岁；身高：$1 7 5 . 8 \pm 8 . 5$ 厘米；体重：$7 0 . 2 \pm 8 . 4 \mathrm { k g }$）在签署知情同意书后参与本实验。本实验已获得蒙特利尔大学（Université de Montréal）伦理委员会的批准（#2023-4743）。使用单个立体视觉 RGBD 相机（RealSense D455，Intel，美国）以 60 Hz 的频率采集 848×480 像素分辨率的彩色（RGB）和深度图像帧。相机放置在距受试者胸部约一米处，大致与受试者颈部对齐。在躯干和左上肢的预定位置放置了十三个方形白色标记（约 $2 \times 2$ 厘米），用于检测关键骨性标志点并减小软组织伪影 [BDLB17]（图 1）。使用 14 相机运动捕捉系统（T40s，Vicon，Oxford，英国）以 120 Hz 记录 13 个放置于白色标记中心的半球形（1 毫米）反光标记，以及 5 个放置于躯干背面和手臂内侧的 10 毫米反光标记（图 1）。为了公平比较两种方法，我们放置了更多的反光标记以进一步评估标记冗余度的影响，因为光电系统可以轻松采集大量标记。"minimal-Vicon-based"（基于最简标记的 Vicon 方法）一词指的是使用与 RGBD 方法相同的标记集合，而"redundant-Vicon based"（基于冗余标记的 Vicon 方法）表示使用额外的标记。由于精简标记集合中未包含背部标记，因此使用人体测量卡尺在第 7 颈椎（C7）与胸骨柄之间进行手动测量，以用于后续的模型缩放。受试者佩戴了经过验证的定制肩峰标记集群 [CMB24]，无需进行光学校准即可从正面视角估计肩胛骨的骨性标志点（肩峰角、肩胛冈和下角）。使用八个表面肌电传感器（Trigno EMG Wireless System，Delsys，美国）以 2160 Hz 记录肌肉活动，电极放置位置包括胸大肌（Pectoralis major）、肱二头肌（长头）、肱三头肌（长头）、上斜方肌、三角肌（前束、中束、后束）和背阔肌，放置前对皮肤进行了剃毛和清洁处理。手摇车配备了仪器化手踏板（Sensix，法国），以 100 Hz 记录三维力和力矩。实验装置如图 2 所示。EMG 和反光标记数据使用 Nexus 2.11（Vicon，Oxford，英国）进行同步。在 Vicon 上记录触发信号，并与 Biosiglive [CVB23] 同时使用，以同步启动 RGBD 和手踏板的记录。受试者被要求完成两次 5 秒的最大努力手摇车测试，保持 60 RPM 的转速，间隔 2 分钟休息。所采集的信号用于后续肌肉力估计的 EMG 归一化。最后，受试者执行一个解剖姿势（用于肌肉骨骼模型缩放）以及在 60 RPM 下进行的 2 分钟手踏板测试，负荷分别为 15、20、30 和 40 W。

![](images/74d0f4145e4f7de1f41422e69aa2567adbbf8e3ec9f63786533d93ed4f1d89ae.jpg)
Figure 1: 参与者佩戴白色标记点、反光标记点（方块中心）以及左臂上的肌电传感器。右侧展示了肩峰簇 [CMB24] 的手动校准。设备上的刻度便于通过附带的 Python 代码从 3D 簇的标记点位置提取解剖学骨性标志。

## 2.2 从 RGBD 相机提取标记点轨迹

RGBDMocap Python 软件包 [CFBS24] 被开发用于在 RGB 图像中检测并标记标记点（x<sub>c</sub> 和 y<sub>c</sub> 标记点位置），然后获取其对应的深度（z）位置。彩色图像和深度图像使用预实现的 RealSense® 算法进行对齐，以为每个彩色像素建立对应的深度值。标记点的标记通过一个四步工作流程执行，以确保在每一帧中进行检测（图 3），具体描述如下：1) 初始化（仅在第一帧执行一次）；2) 标记点二维预测；3) 标记点二维校正；4) 标记点三维校正。

## 2.2.1 初始化

为了提供稳健、快速且资源高效的检测，图像被分割为三个不同的区域：胸部、肩部和手臂。我们使用 Python multiprocessing 包 [Hun19] 并行处理这三个区域。所有后续的初始化过程均独立应用于每个区域。开发了一个图形用户界面（GUI）以方便操作过程（图 4）。对图像应用滤波器以改善身体上白色标记的检测效果，这反映了针对每个光电相机的初始参数调优。具体而言，首先使用对齐的深度图像去除背景。然后应用模糊滤波器，通过平滑边缘来增强图像。接着使用 OpenCV 函数 [How13] 应用对比度受限的自适应直方图均衡化（对比度限制的阈值以及行和列中的瓦片大小是根据每个区域的经验选择的）。最后，应用二值化阈值以提高白色标记的可见性。使用 OpenCV 库中的斑点检测器（即图像中某些属性近似恒定的区域）自动检测白色方形标记。该检测器使用白色范围颜色、面积、圆度和凸度等准则来识别连接的像素组。这些准则在 GUI 中针对每个区域和参与者进行了手动调整（图 4-a)），以提供最佳的斑点检测。一旦相机定位完成并为第一位参与者确定了参数，这些参数将为后续参与者产生相当好的结果。尽管如此，仍建议针对每位参与者进行微调。随后，用户对每个白色标记进行一次标注，形成后续预测步骤的初始值（图 4-b)）。初始化完成后，对每帧图像使用三步方法进行处理。

## 2.2.2 标记 2D 预测

由于运动引起的噪声和图像中的光照变化，斑点检测器可能会检测出假阳性斑点或遗漏图像中的某些白色标记。因此，仅通过检测斑点并寻找最近邻来跟踪轨迹是不够的。使用标记先前位置的预测步骤来估计标记的后续位置。为了提高算法的稳健性，独立使用了两种预测方法：

- 一个光流检测器通过分析先前图像与当前图像之间的差异来跟踪每个先前标注的标记。使用金字塔中 Lucas-Kanade 稀疏迭代版本的算法 [B<sup>+</sup>01] 在

![](images/118a56472d76cce964871d9e15af83862d00b9880cc24e16e93bbaec59b674ac.jpg)  
图 2：实验设置，显示参与者在进行手摇车运动时 RGBD 相机和一些 Vicon 相机的布置位置。

![](images/e9905416eed6f586f5ec5156e760ad56d828dad9e3f2d32181f561a06b146206.jpg)  
Figure 3: 来自 RGBD 摄像头的标记点跟踪的软件架构。

![](images/c348a5eb8a578b65e2792fcb55bf74a6298ba80471cfa5a589265303936796ba.jpg)  
Figure 4: 图形用户界面，分别用于 a) 手臂区域的斑点检测（blob detection）和 b) 手动标记点标注（manual markers labeling）。

使用了 OpenCV（开源计算机视觉库）。该算法通过跟踪像素强度随时间的变化来估计特定像素的位置。初始已标注的标记点位置被用作初始位置。

• Kalman 滤波器（卡尔曼滤波器）预测标记点的 2D 位置，利用其之前的位置和速度。该方法仅依赖于检测到的斑点位置。通过使用协方差矩阵，滤波器预测当前位置。Kalman 滤波器的实现通过 OpenCV 完成，将标记点位置和速度作为状态。已标注标记点的初始位置用作滤波器的初始状态。

## 2.2.3 标记点 2D 修正

将估计的标记点位置与图像中检测到的斑点位置进行比较。如果预测位置与检测到的斑点之间的距离小于 10 像素，则该标记点被分类为可见，并将其位置替换为斑点位置。否则，估计位置保持不变，且该标记点被分类为不可见。在距离摄像头 1 米处，标记点的平均尺寸约为 10×10 像素。两种跟踪方法生成的标记点位置通过对可见标记点的位置取平均值进行合并。当两种方法均未检测到可见标记点时，保留其最后已知位置以用于下一步。虽然该方法在标记点可见时表现良好，但当斑点未被持续检测到时会出现问题，从而导致不确定性。因此，在 3D 空间中利用所有标记点实施了额外的修正步骤，以缓解这些问题。

## 2.2.4 标记点 3D 修正

前述流程（第 2.2.2、2.2.3 节）仅通过二维彩色图像实现标记追踪。然而，本节通过在三维相机空间中操作，充分发挥 RGBD 相机的完整能力。为此，基于初始的三维标记位置构建了一个三维铰接模型，该模型由四个刚体段（胸、肩膀、上臂、前臂）通过球窝关节相互连接构成，而胸段在空间中可自由运动。利用前述步骤（第 2.2.2–2.2.3 节）中得到的二维标记位置，结合深度图像，通过 RGBD 相机的默认内参，将标记位置反投影到相机坐标系下的三维位置。随后，基于扩展卡尔曼滤波（EKF）[FCBL10] 的逆运动学算法被应用于三维标记位置，以估计模型的关节角度，所用工具为 Biorbd 库 [MB21]。EKF 是一种常见的传感器融合方法，其目标是：在给定含噪声测量值（此处为标记）的前提下，通过最小化误差协方差矩阵的迹，估计非线性系统的状态（此处为关节角度与角速度）。这些关节角度给出了模型及各刚体段在相机三维空间中的位姿估计。随后利用正运动学从关节角度反推标记位置。这些三维位置通过 RealSense® 重投影函数及内参再次投影至图像坐标。最后，执行第 2.2.3 节所述的相同的二维校正步骤，基于斑点检测从正运动学得到的标记位置中提取校正后的二维标记位置。此后，这些最终的二维标记位置被用于卡尔曼校正及光流输入位置，为下一帧的标记预测提供校正后的初始值。

## 2.3 RGBD – Vicon 配准

第 2.2 节允许我们在相机坐标系 $(x_c, y_c, z_c)$ 中获取标记位置。为了实现通过 RGBD 获取的三维位置与基于 Vicon（一种光学动作捕捉系统）的动作捕捉所提供的三维位置之间的有意义比较，有必要将两个系统的三维位置表示在同一坐标系中。标记的三维位置评估在相机坐标系 $(x_c, y_c, z_c)$ 内进行，以评估沿每个相机轴的误差，因为彩色图像 $(x_c, y_c)$ 和深度图像 $(z_c)$ 来自不同的传感器。对于生物力学分析，位置在 Vicon 坐标系 $(x_v, y_v, z_v)$ 内定义，该坐标系由垂直方向的 z 轴定义。坐标系变换通过一个最优变换矩阵完成，该矩阵旨在最小化初始同步帧对上标记的误差。随后，将 RGBD 到 Vicon 的旋转-平移矩阵应用于变换所有帧中所有 RGBD 标记的三维位置。类似地，该矩阵的逆矩阵用于将每个 Vicon 标记变换到 RGBD 坐标系中。此外，如相机提供的不连续相机帧编号所示，RGBD 记录可能无法始终保持 60 Hz 的帧率。这些帧编号中的间隙被用于以 60 Hz 插值数据以填补缺失的帧。然后，使用线性插值以匹配 Vicon 系统的帧率 (120 Hz)。这种方法促进了 RGBD 和 Vicon 系统之间时间序列数据的比较。接下来的章节（第 2.4 节）使用在 Vicon 坐标系中表示的修改后的 RGBD 标记位置。

## 2.4 实时生物力学框架

开发了一个实时生物力学框架，并将其应用于使用两套动作捕捉系统收集的数据，考虑了两套标记集（即 RGBD、最小化-Vicon 和冗余-Vicon）。该框架包括从标记位置计算关节角度的逆运动学、计算关节力矩的逆动力学，以及估计肌肉力量的静态优化。

## 2.4.1 肌肉骨骼模型

## 2.4.2 逆运动学

该肌肉骨骼模型基于 Wu 等人提出的上肢模型 $[\mathrm{WLB^{+}16}]$，包含五个环节（胸廓、锁骨、肩胛骨、上臂和前臂），具有 10 个自由度（两个胸锁关节旋转、三个肩锁关节旋转、三个盂肱关节旋转，以及肘部屈伸和旋前/旋后）。每个运动捕捉方法都使用静态试验对模型进行缩放，生成三个独立的缩放模型。胸廓宽度（前后径）通过胸骨柄与第七颈椎（C7）之间的测量值进行缩放。该模型由 31 条 Hill 型肌肉驱动，包含并联的收缩元和被动元元件。为了加快计算速度，肌腱被视为刚体 $[\mathrm{MUSD13}]$。此外，由于关注的运动速度较慢（60 转/分），肌肉直接由其激活水平控制（忽略具有机电延迟的神经激励动力学 $[\mathrm{NK01}]$）。对于肌肉 $m$，由激活水平 $(a)$ 和肌肉运动学（归一化肌纤维长度 $\tilde{l}_m$ 和速度 $\widetilde{\nu}_m$）计算得到的肌力 $(f^m)$ 表示为：

$$
f^m = f_o^m ( a f^{act}(\tilde{l}_m) f^{\nu}(\widetilde{\nu}_m) + f^{pas}(\tilde{l}_m) ) \cos \alpha ,\tag{1}
$$

其中 $f_o^m$ 是肌肉的最大等长肌力，$f^{act}(\tilde{l}_m)$、$f^{\nu}(\widetilde{\nu}_m)$ 和 $f^{pas}(\tilde{l}_m)$ 分别是主动力-长度关系、力-速度关系和被动力-长度关系；$\alpha$ 是肌肉的羽状角 $[\mathrm{DGKRF16}]$。

## 2.4.2 逆运动学

首先，通过应用跨 14 帧（延迟 7 帧）的滑动平均对标记轨迹进行平滑处理。然后，使用在 Biorbd 软件中实现的 EKF $[\mathrm{FCBL10}]$ 进行逆运动学，其中关节角度 $q$、速度 $\dot{q}$ 和加速度 $\ddot{q}$ 作为状态变量。有关使用 EKF 进行逆运动学的更多信息，读者可参考 $[\mathrm{JLWK20}]$。

## 2.4.3 逆动力学

关节力矩 $(\tau)$ 由关节运动学 $(q, \dot{q}, \ddot{q})$（由 EKF 输出）、外部力 $(F_{ext})$（由踏板力传感器测得）和环节惯性参数（包含在模型中）计算得出：

$$
\tau = M(q)\ddot{q} + N(q, \dot{q}) + G(q) + C^T(q) F_{ext} \tag{2}
$$

其中 $M(q)$、$N(q, \dot{q})$、$G(q)$ 和 $C(q)$ 分别是质量矩阵、科里奥利力和离心力向量、重力效应向量以及手部力作用点的雅可比矩阵。

## 2.4.4 EMG 驱动的静态优化

静态优化用于估计肌肉力，对应于一个最小二乘问题，旨在根据通过逆动力学获得的关节力矩以及实验性 EMG 找到最优肌肉激活（a）。离线状态下，在手摇自行车测试之前，最大手摇自行车测试的 EMG 信号经带通滤波（10–425 Hz）、整流，并使用四阶 Butterworth 滤波器进行低通滤波（5 Hz）。然后，通过计算最大用力测试期间每块记录肌肉在 1 s 内最高值的均值来获得其激活值。静态优化定义如下：

$$
\begin{array} { r } { \displaystyle \operatorname* { m i n } _ { a , \tau ^ { r e s } } \sum _ { k = 1 } ^ { n _ { e m g } } \omega _ { e m g } \| e m g _ { k } - a _ { k } ^ { i n f } \| ^ { 2 } + \sum _ { i = 1 } ^ { n _ { m u s c l e s } - n _ { e m g } } \omega _ { a } \| a _ { i } ^ { n i n f } \| ^ { 2 } } \\ { + \sum _ { j = 1 } ^ { n _ { \mathrm { j o i n t s } } } \omega _ { \tau } \| \tau _ { j } - ( \tau _ { j } ^ { m } + \tau _ { j } ^ { r e s } ) \| ^ { 2 } + \omega _ { r e s } \| \tau _ { j } ^ { r e s } \| ^ { 2 } } \end{array}\tag{3a}
$$

$$
\forall t , \pmb { a } \in \mathcal { A } , \pmb { \tau } ^ { r e s } \in \mathcal { T }\tag{3b}
$$

其中 $a ^ { i n f }$ 和 $a ^ { n i n f }$ 分别为已知（informed）和未知（non-informed）肌肉激活。τ 对应于通过逆动力学（式 2）获得的关节力矩，$\tau ^ { r e s }$ 为建模关节内刚度的残余关节力矩，$\tau ^ { m }$ 为通过式 4 计算的肌肉力矩。emg 代表所记录的八个 EMG 信号。$\omega _ { a }$、$\omega _ { e m g }$、$\omega _ { \tau }$ 和 $\omega _ { r e s }$ 分别代表用于最小化肌肉激活、追踪实验性 EMG、追踪逆动力学力矩以及最小化残余力矩所分配的权重。式 3b 确保了肌肉激活在 0 到 1 之间的生理边界，并将残余力矩限制在 −5 N·m 到 5 N·m 之间。跨过关节 j 的肌肉 m 产生的关节力矩为：

$$
\tau _ { j } ^ { m } = \sum _ { i = 1 } ^ { n _ { \mathrm { m u s c l e s } } } \Gamma _ { i } \cdot f _ { i } ^ { m }\tag{4}
$$

其中 $\Gamma _ { i }$ 为肌肉 i 的力臂，由肌肉长度的雅可比矩阵计算得到。尽管已设定上述边界，但为了确保成功求解，优化过程中未纳入额外的约束条件。然而，必须将关节力矩追踪的权重设定得足够高，以保证力矩被忠实地追踪。本优化使用 Acados [VFK<sup>+</sup>22] 执行，借助线性求解器 Blasfeo 和 QP 求解器 OSQP [SBG<sup>+</sup>20] 的高效实现，从而实现快速求解。

## 2.5 数据分析

对 RGBDmocap 系统的评估聚焦于两个关键方面：1）标记位置的精度以及该误差在生物力学分析结果中的传播情况，并与 Vicon 系统进行对比；2）整体生物反馈延迟。首先，通过三种不同的方法对两个系统进行了对比分析：1）计算基于 RGBD 的标记位置与基于 Vicon 的方法所获标记位置之间的绝对误差；2）考察生物力学研究结果中的误差传播情况；3）研究基于 RGBD 的方法中标记冗余度对生物力学结果的影响。通过 Bland–Altman 分析 [BA07] 对 Vicon 和 RGBD 两种不同方法获取的三维标记位置之间的一致性进行了评估，评估在 RGBD 相机坐标系内进行。具体而言，使用偏差（bias）和一致性界限（LoA, limits of agreement）来评估一致性。在分析过程中，Bland-Altman 差值沿 $x _ { c }$ 轴和 $y _ { c }$ 轴的平均值进行计算，对应于彩色图像平面内的误差；沿 $z _ { c }$ 轴的差值则对应于深度值所引入的误差。Bland-Altman 沿 $z _ { c }$ 轴的平均值被计算出来，旨在探讨标记到相机距离的影响。同样的分析也被应用于生物力学研究结果，包括关节角度、关节力矩和肌肉力。在此，一致性评估通过将基于 RGBD 的方法和基于最少标记 Vicon 的方法分别与基于冗余标记 Vicon 的方法（作为参照）进行比较，以及将基于 RGBD 的方法与基于最少标记 Vicon 的方法进行比较来实现。该方法产生了三种不同的分析，旨在评估方法之间的一致性，并考察结果对所采集标记数量的依赖性。为便于阅读，使用以下符号表示：RGBD 对比冗余 Vicon（RGBDvRED）、最少标记 Vicon 对比冗余 Vicon（MINvRED）、以及 RGBD 对比最少标记 Vicon（RGBDvMIN）。

要在临床环境中应用，反馈必须以实时且低延迟的方式提供。为了评估基于 RGBD 的方法的实时性能，对运动与结果之间的延迟进行了评估。该延迟包括每帧检索标记位置和计算生物力学结果所需的时间。同时也考虑了标记滤波（第 2.4.2 节）所引入的延迟。跟踪算法和生物力学分析在一台笔记本电脑上运行（Intel® CoreTM i7-6700HQ CPU @ 2.60 GHz），操作系统为 Ubuntu 20.04 LTS。

## 3 结果

## 3.1 三维标记位置

标记点的三维轨迹在两个动作捕捉系统中均呈现出一致的模式（图 5）。这一观察结果得到了标记点跨所有试次与受试者轨迹 RMSE 的进一步支持，其平均值（mean）为 3.32 ± 3.94 mm。Bland-Altman 分析（图 6）显示数据点在图中呈均匀分布，表明标记点深度对两个系统一致性结果的影响很小。图像平面（由 $x$ 和 $y_c$ 轴构成）内的一致性（图 6-a）优于沿深度轴 $z$ 的一致性（图 6-b），其一致性界限更窄（分别为 (-2.33 mm, 2.59 mm) 与 (-3.59 mm, 3.39 mm，需核实) ……此处按原文保留：(-2.70 mm, 2.79 mm) 与 (-3.87 mm, 3.65 mm)）。两种情况下，偏差均接近于零。

![](images/843d89be40d3e75f9d6511427c90b68da7178a70fe8390d033eae03efb861a5b.jpg)  
图 5：一名受试者 120 个运动周期三维标记轨迹的均值，分别来自 minimal-Vicon（红色）与 RGBD-based（蓝色）动作捕捉方法。

![](images/14db41f2bb14e0a5bd97f5008fb3813fbc8f12e0001f09b62a984d3f35cc315b.jpg)  
图 6：标记点位置的 Bland-Altman 图，以 RGBD 相机坐标系表示，每个标记点由一个单点表示，并按受试者颜色加以区分。差异分别沿图像平面（a）、深度轴（b）以及所有轴的平均值（c）计算。

![](images/c6fbe9caa7c88a05810e50f2a3ce7293a23af2d7f5ae4f68e2af890f79cf5b24.jpg)  
图 7：一名受试者 120 个运动周期关节角度的均值（实线）及其 ± 一倍标准差（阴影区域），分别来自 RGBD-based（蓝色）、带冗余标记的 Vicon（红色）以及 minimal（绿色）。

表 1：三种方法按部位划分的缩放因子（所有受试者所有轴的均值）

| Segment | RGBD | Vicon | |
|---|---|---|---|
| Thorax | $1.06 \pm 0.04$ | redundant $1.06 \pm 0.05$ | minimal $1.07 \pm 0.04$ |
| Clavicle | $1.10 \pm 0.09$ | $1.08 \pm 0.08$ | $1.09 \pm 0.08$ |
| Scapula | $1.21 \pm 0.04$ | $1.21 \pm 0.05$ | $1.21 \pm 0.05$ |
| Humerus | $0.98 \pm 0.08$ | $0.98 \pm 0.07$ | $0.97 \pm 0.07$ |
| Forearm | $0.99 \pm 0.06$ | $1.01 \pm 0.07$ | $0.99 \pm 0.05$ |

## 3.2 生物力学结果

三种方法在所有参与者中都生成了具有可比身体缩放因子的个性化模型（表 1）。RGBDvMIN 的关节角度 RMSD 平均差异为 $2 . 3 4 \pm 2 . 7 9 ^ { \circ }$，相比 RGBDvRED 和 MINvRED 大约低两倍。关节力矩也观察到类似趋势，RGBDvMIN 的 RMSD 达到 0.88±1.86 N.m（表 2）。此外，RGBDvMIN 的肌肉力 RMSD 平均差异为 3.55±13.68 N，相比 RGBDvRED 或 MINvRED 大约低三倍（表 2）。这些结果得到了 Bland-Altman 分析的支持，其中 RGBDvMIN 在关节角度上的一致性界限（-4.53° 至 5.01°）以及在关节力矩上的一致性界限（-1.70 N.m, 1.40 N.m）相比 RGBDvRED 和 MINvRED 大约窄两倍。同样，RGBDvMIN 在肌肉力上的一致性界限（-19.96 N, 16.79 N）比其他两个比较窄两倍以上。在所有情况下，所有比较的偏差都接近于零，其中最高的（但仍然较低）值出现在 RGBDvRED 比较中的肌肉力上（-2.12 N）。对于每个结果，将基于 RGBD 的方法和最简 Vicon 方法与冗余 Vicon 方法进行比较时，分析显示出一致性水平相当。图 7-9 全面展示了在 40 W 阻力下的某位参与者平均周期内的关节角度、力矩和肌肉力。所有方法的关节角度呈现出相同的模式，尽管冗余 Vicon 与其他方法之间存在一些偏移，特别是在胸锁关节和肩锁关节上。两种标记点较少的方法之间几乎没有偏移。基于 RGBD 的方法提供的轨迹噪声较大，特别是肩胛骨旋转和倾斜以及前臂旋前/旋后。三种方法提供的关节力矩（图 8）几乎完全重合，尽管基于 RGBD 的方法由于关节角度噪声较大而显示出较大的标准差区域。肌肉力在周期内表现出几乎相同的起始时间，但冗余 Vicon 方法与其他两种方法相比在幅度上存在明显偏移（图 9）。

![](images/c788be489801710158b287281b5f8df0e378db85dafd4f7a17fa4e3eb9824ff1.jpg)  
图 8：某位参与者 120 个周期的关节力矩平均值（实线）±一个标准差（阴影区域），分别对应基于 RGBD（蓝色）、带冗余的 Vicon（红色）和最简 Vicon（绿色）。

![](images/734bcafe3fd9f2fc21fe089d45ce6aa0da2ce692e6da0c4f6fb00658d4ab9a7c.jpg)  
图 9：某位参与者 120 个周期的九块肌肉力平均值（实线）±一个标准差（阴影区域），分别对应基于 RGBD（蓝色）、带冗余的 Vicon（红色）和最简 Vicon（绿色）。

表 2：关节角度、关节力矩和肌肉力的均方根偏差（RMSD）及一个标准差（SD），以及 Bland-Altman 一致性界限（LOA）和偏差。结果分别给出了基于 RGBD 的方法和精简 Vicon 方法与冗余 Vicon 方法的比较，以及基于 RGBD 的方法与精简 Vicon 方法之间的比较。

<table><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">RMSD</td><td rowspan="2">SD</td><td colspan="2">LOA</td><td rowspan="2">偏差</td></tr><tr><td>下限</td><td>上限</td></tr><tr><td rowspan="3">关节角度（°）</td><td>RGBD vs 冗余</td><td>4.50</td><td>2.85</td><td>-12.13</td><td>11.22</td><td>-0.46</td></tr><tr><td>精简 vs 冗余</td><td>4.46</td><td>0.84</td><td>-13.28</td><td>11.89</td><td>-0.70</td></tr><tr><td>RGBD vs 精简</td><td>2.34</td><td>2.79</td><td>-4.53</td><td>5.01</td><td>0.24</td></tr>
<tr><td></td><td>RGBD vs 冗余</td><td>2.26</td><td>2.49</td><td>-3.83</td><td>5.07</td><td>0.62</td></tr>
<tr><td rowspan="2">关节力矩（N·m）</td><td>精简 vs 冗余</td><td>2.25</td><td>1.56</td><td>-4.19</td><td>5.73</td><td>0.77</td></tr>
<tr><td>RGBD vs 精简</td><td>0.88</td><td>1.86</td><td>-1.70</td><td>1.40</td><td>-0.15</td></tr>
<tr><td rowspan="3">肌肉力（N）</td><td>RGBD vs 冗余</td><td>10.70</td><td>20.89</td><td>-51.64</td><td>47.40</td><td>-2.12</td></tr>
<tr><td>精简 vs 冗余</td><td>10.60</td><td>17.55</td><td>-51.65</td><td>50.58</td><td>-0.53</td></tr>
<tr><td>RGBD vs 精简</td><td>3.55</td><td>13.68</td><td>-19.96</td><td>16.79</td><td>-1.58</td></tr></table>

表 3：每帧各部分引入延迟的均值和标准差（SD）

<table><tr><td colspan="2"></td><td colspan="2">延迟（ms）</td></tr>
<tr><td rowspan="2">追踪</td><td>二维追踪与修正</td><td>均值 7.64</td><td>SD 1.83</td></tr>
<tr><td></td><td>2.71</td><td></td></tr>
<tr><td></td><td>三维修正</td><td>0.47</td><td>0.58</td></tr>
<tr><td rowspan="2">生物力学分析</td><td>逆运动学</td><td></td><td>0.042</td></tr>
<tr><td>逆动力学</td><td>0.31</td><td>0.051</td></tr>
<tr><td>总计</td><td>静态优化与肌肉力计算</td><td>3.94 15.07</td><td>1.54 4.04</td></tr></table>

## 3.3 时间性能

表 3 概述了每个任务所产生的延迟，范围从 RGBD 跟踪到生物力学分析。由于三个区域采用多进程处理，所有被试的 RGBD 跟踪总延迟为 10.35±2.41 ms，对应跟踪频率为 94 Hz，超过了相机帧率（60 Hz）。在获取标记点轨迹后，生物力学分析引入的延迟约为 4.72±1.63 ms，对应求解速率为 211 Hz，显著超过相机帧率。反馈延迟还受到使用实时移动平均方法对标记点进行滤波的影响，该方法引入约七帧的延迟。鉴于系统频率固定为 120 Hz，这导致每帧的总延迟为 58 ms。然而，该延迟并不影响算法频率，算法频率平均约为 63 Hz（15.07±4.04 ms，见表 3）。综合考虑跟踪、生物力学分析以及标记点滤波所引入的延迟，从记录数据得到的估计总延迟约为 73.07±4.04 ms。

## 4 讨论

我们的目标是评估使用单个 RGBD 相机进行上肢生物力学分析的可行性。我们提出了一种半自动标注算法和一种高效的 EMG 驱动的生物力学流程，基于单个 RGBD 相机提供实时反馈。我们的结果表明，我们的方法能够产生准确的实时三维标记点轨迹，从而实现全面的上肢生物力学评估。所提供的用户友好图形界面简化了跟踪算法的使用，而其单相机设置使其易于安装，适合临床环境。此外，我们的方法支持从正面进行记录，能够捕获大多数上肢部位，包括肩胛骨和锁骨。

## 4.1 标记点精度

如 Bland-Altman 分析所示，基于 RGBD 的动捕系统所提供的三维标记点位置与基于 Vicon 的动捕系统之间具有良好至较强的一致性。接近零的偏差（-0.21 mm）以及在（-3.05 mm，3.48 mm）范围内的一致性界限，均低于由标记点放置引起的误差（最高可达 9 mm [SJC<sup>+</sup>09]）或软组织伪影引起的误差（最高可达 18 mm [AADS21]）。尽管标记点与相机之间的距离对一致性水平影响不大，但在深度轴上观察到的一致性界限比图像平面上的更宽。这表明由于深度图像分辨率或其与 RGB 图像的未对齐，三维空间中可能引入了更大比例的误差。这类误差有望通过提高相机分辨率或进行一次相机深度重标定 [DTLC17] 来加以缓解。

## 4.2 较低的误差传播

众所周知，增加标记点的数量可以提高运动学估计的准确性。除了增加标记点之外，还可以通过数值技术进行评估，例如使用运动链 [BWY08]。然而，即使临床医生在标记点放置方面具有专业知识，过多的标记点放置也可能非常耗时。我们使用 RGBD（彩色图像结合深度信息）相机所采用的标记点数量与其他临床上肢标记点方案一致，例如使用 7 个 [VRD15]、8 个 [BSK<sup>+</sup>23] 或 12 个 [NFB<sup>+</sup>18] 标记点。我们的评估表明，对于每个生物力学结果，当使用相同的标记点方案时，由偏差和一致性界限所表示的一致性水平是相似的。在比较两种最简方法时，其一致性水平是观察到的最简方法与冗余方法之间一致性的两倍，这强调了与使用两种不同的运动捕捉方法相比，标记点方案对生物力学结果的影响更大。在胸廓上实施标记点簇，如 Noble 等人 [NFB<sup>+</sup>18] 所做的那样，有望减轻初始胸廓朝向的不准确性，从而提高相邻关节段的关节角度精度。尽管如此，RGBD 与最简 Vicon 方法之间的比较显示出强一致性，关节角度的一致性界限范围为 -4.53° 到 5.01°。RGBD 与冗余 Vicon 方法之间关节角度估计的误差（约 5°）低于文献中报告的使用 IMU（惯性测量单元）系统（约 30° [HKWE22]）或无标记系统（约 23° [LMD<sup>+</sup>22]）对上肢的测量结果。这些数值落在使用运动学重建方法时通常观察到的变异性范围内，在脑瘫儿童的上肢分析中，该方法在不同实验环节之间可能产生高达 10° 的误差 [JFB<sup>+</sup>11]。类似地，Uchida 等人 [US22] 报告关节力矩估计误差高达 26.6 N·m（峰值的 17%），而在我们研究中，关节力矩的一致性界限从 -1.70 N·m 到 1.40 N·m（约为关节力矩最高值的 7%，即 ±25 N·m）。我们观察到在关节力矩计算方面的一致性高于肌肉力估计方面的一致性。这可归因于关节运动学的差异，关节运动学通过力臂或肌肉长度的变化影响肌肉力的估计。Blache 和 Begon [BB17] 的研究证明了这一点。然而，尽管存在这些差异，所有方法都产生了相似的肌肉力模式，其起始时刻可能因所使用阈值的不同而相同。

## 4.3 延迟

Waltermate 等人 [WSH<sup>+</sup>16] 表明，当延迟超过 75 ms 时，运动表现和感知会受到影响。本工作的最终目标是促进实时生物反馈以引导康复运动，因此本系统所引入的延迟成为一个关键因素。我们开发了一个基于 Python multiprocessing（多进程）的高效标记点标注流水线，能够以 10.6 ms（94 Hz）的延迟获取已标注的标记点，显著超过了相机的帧率。这确保了即使使用更高帧率的相机，我们的算法也能及时提供已标注的标记点。对于生物力学流水线，我们使用了基于 C++ 的库进行肌肉骨骼建模（biorbd）和静态优化（acados），能够以 4.7 ms（213 Hz）的延迟提供结果，同样显著超过了相机的帧率。然而，对标记点轨迹进行滤波的需求会导致每帧产生 58 ms 的显著延迟。因此，反馈可以以 73.07 ms 的延迟进行传送，与 75 ms 的目标延迟处于同一量级。该频率有望通过并行执行标记点检索和生物力学分析来进一步提升。此外，标记点滤波可以使用卡尔曼滤波器（Kalman filter）来完成，这将使延迟大幅降低至接近 0 ms [SNPW08]。

## 4.4 临床意义

对于我们这种新型低成本动作捕捉系统，我们设计了一个图形化界面，以简化其在各种场景（从研究到临床应用）中的使用。跟踪算法是半自动的，便于设置流程；尽管如此，初始帧仍需进行手动标注。为了提高效率，可以集成迭代最近点（iterative closest point）方法，利用预实现的模型自动标注初始帧 [YPDQ09]。此外，我们可以引入一个标定步骤，直接通过相机建立 z 轴垂直方向上的坐标系。这可以通过使用带有白色标记点的三维形状（例如 Vicon 系统所使用的 5 标记点标定棒）或使用集成在相机中的惯性传感器来实现。本研究下一步的工作是在临床背景下、在临床医生参与的情况下对系统进行评估。

本研究存在一些局限性。首先，相机以非保守的 60 Hz 帧率运行，这可能会影响时间相关的分析。然而，这一局限性可以得到有效解决，因为相机提供图像计数，允许使用插值方法检测和处理任何不连续性。此外，由于所选相机并非面向研究用途，将其与肌电图（EMG）或测力台等其他设备进行同步可能具有挑战性。biosiglive [CVB23] 等外部工具可以协助从各种来源获取数据，但仍需进一步开发。我们低成本运动捕捉系统的一个未来目标可能涉及开发一个嵌入式系统，以更可靠地促进数据传播。未来的工作将利用我们的自动标注算法来提供已标注的深度图像，这些图像可用于训练一个新的基于骨性标志点的无标记姿态估计机器学习模型。

## 5 结论

我们证明，使用单个低成本 RGBD 相机和白色皮肤标记，可以进行与光电系统相媲美的上肢生物力学分析。通过开发半自动标注算法和基于 EMG 的生物力学流程，我们基于单个 RGBD 相机提供了实时反馈。我们的方法提供了准确的实时三维标记轨迹，促进了全面的上肢生物力学评估。凭借其用户友好的界面和单相机设置，我们的系统非常适合临床环境。在考虑相同数量的记录标记时，我们观察到 RGBD 和 Vicon 方法之间具有很强的一致性。然而，我们强调了标记冗余度不足对逆运动学的影响，导致使用 RGBD 和最小 Vicon 系统与冗余 Vicon 系统相比存在显著差异。总体而言，我们的研究结果强调了 RGBD 相机作为上肢生物力学分析经济高效替代方案的潜力，并在未来研究中具有进一步改进和增强的机会。

## 6 利益冲突声明

作者声明，他们没有已知的可能影响本文所报告工作的竞争性经济利益或个人关系。

## 7 作者贡献

AC、MB 和 LS：概念化。AC 和 KF：数据管理。AC、MB 和 LS：方法学。AC 和 KF：软件。MB 和 LS：指导。AC：撰写初稿。MB 和 LS：撰写-审阅与编辑。AC 和 MB：经费获取。所有作者均已阅读并同意发表的手稿版本。

## 8 资助

本研究获得加拿大自然科学与工程研究理事会（NSERC）通过 CREATE OPSIDIAN 项目、发现基金 RGPIN/04978-2019 以及魁北克省自然与技术研究基金（FRQNT）的资助。

## 参考文献

[AADS21] Andrea Ancillao, Erwin Aertbelien, and Joris De Schutter. Effect of the soft tissue artifact on marker measurements and on the calculation of the helical axis of the knee during a gait cycle: A study on the cams-knee data set. *Human Movement Science*, 80:102866, 2021.

[B$^{+}$01] Jean-Yves Bouguet et al. Pyramidal implementation of the affine lucas kanade feature tracker description of the algorithm. Intel corporation, 5(1-10):4, 2001.

[BA07] J Martin Bland and Douglas G Altman. Agreement between methods of measurement with multiple observations per individual. *Journal of biopharmaceutical statistics*, 17(4):571–582, 2007.

[BB17] Yoann Blache and Mickael Begon. Influence of shoulder kinematic estimate on joint and muscle mechanics predicted by musculoskeletal model. *IEEE Transactions on Biomedical Engineering*, 65(4):715–722, 2017.

[BBC$^{+}$16] Matteo Bigoni, Silvia Baudo, Veronica Cimolin, Nicola Cau, Manuela Galli, Lucia Pianta, Elena Tacchini, Paolo Capodaglio, and Alessandro Mauro. Does kinematics add meaningful information to clinical assessment in post-stroke upper limb rehabilitation? a case report. *Journal of physical therapy science*, 28(8):2408–2413, 2016.

[BDLB17] Yoann Blache, Raphael Dumas, Arian Lundberg, and Mickael Begon. Main component of soft tissue artifact of the upper-limbs with respect to different functional, daily life and sports movements. *J. Biomech.*, 62:39–46, 2017.

[BDMM$^{+}$24] Michele Boldo, Roberto Di Marco, Enrico Martini, Mauro Nardon, Matteo Bertucco, and Nicola Bombieri. On the reliability of single-camera markerless systems for overground gait monitoring. *Computers in biology and medicine*, page 108101, 2024.

[Bol03] Jeffrey E Bolek. A preliminary study of modification of gait in real-time using surface electromyography. *Applied psychophysiology and biofeedback*, 28:129–138, 2003.

[BSK$^{+}$23] Monika Błaszczyszyn, Agnieszka Szczesna, Mariusz Konieczny, Paweł Pakosz, Stefan Balko, and Zbigniew Borysiuk. Quantitative assessment of upper limb movement in post-stroke adults for identification of sensitive measures in reaching and lifting activities. *Journal of Clinical Medicine*, 12(9):3333, 2023.

[BWY08] Mickael Begon, Pierre-Brice Wieber, and Maurice Raymond Yeadon. Kinematics estimation of straddled movements on high bar from a limited number of skin markers using a chain model. *Journal of biomechanics*, 41(3):581–586, 2008.

[CFBS24] Amedeo Ceglia, Kael Facon, Mickael Begon, and Lama Seoud. Rgbd mocap, 2024. https://github.com/aceglia/rgbd\_mocap.

[CMB24] Amedeo Ceglia, Lucas Mulhaupt, and Mickael Begon. Dispositif innovant pour reconstruire la cinematique de la scapula. In *Proceedings of the XXII congr\`es de la Soci\'et\'e Francophone d'analyse du mouvement chez l'enfant et l'adulte*, 2024.

[CVB23] Amedeo Ceglia, Felipe Verdugo, 和 Mickael Begon. Biosiglive：一个用于实时生物信号处理的开源 Python 软件包. Journal of Open Source Software, 8(83):5091, 2023.

[DGKRF16] Friedl De Groote, Allison L Kinney, Anil V Rao, 和 Benjamin J Fregly. 用于求解肌肉冗余问题的直接配点最优控制问题公式评估. Annals of Biomedical Engineering, 44(10):2922–2936, 2016.

[DTLC17] Walid Darwish, Shenjun Tang, Wenbin Li, 和 Wu Chen. 一种用于商用 RGB-D 传感器的新型标定方法. Sensors, 17(6):1204, 2017.

[FCBL10] V Fohanno, F Colloud, M Begon, 和 P Lacouture. 基于扩展卡尔曼滤波算法的皮划艇三维运动学估计：一项初步研究. Computer Methods in Biomechanics and Biomedical Engineering, 13(S1):55–56, 2010.

[FMH<sup>+</sup>04] Virginie G Femery, Pierre G Moretto, Jean-Michel G Hespel, Andre Thévenon, 和 Ghislaine Lensel. 一种用于足部卸载的实时足底压力反馈装置. Archives of Physical Medicine and Rehabilitation, 85(10):1724–1728, 2004.

[GPC13] Oonagh M Giggins, Ulrik McCarthy Persson, 和 Brian Caulfield. 康复中的生物反馈. Journal of Neuroengineering and Rehabilitation, 10(1):1–11, 2013.

[HKWE22] Jakob Henschke, Hannes Kaplick, Monique Wochatz, 和 Tilman Engel. 使用商用传感器-软件系统评估惯性测量单元用于肩部运动学的有效性：一项验证性研究. Health Science Reports, 5(5):e772, 2022.

[How13] Joseph Howse. Python OpenCV 计算机视觉, 第 27 卷. Packt Publishing Birmingham, UK, 2013.

[Hun19] John Hunt. 多进程处理. 见《Python 3 编程高级指南》, 第 363–376 页. Springer, 2019.

[JFB<sup>+</sup>11] Ellen Jaspers, Hilde Feys, Herman Bruyninckx, Andrea Cutti, Jaap Harlaar, Guy Molenaers, 和 Kaat Desloovere. 偏瘫型脑瘫儿童上肢运动学的可靠性. Gait & Posture, 33(4):568–575, 2011.

[JLWK20] Vladimir Joukov, Jonathan FS Lin, Kevin Westermann, 和 Dana Kulic. 基于约束扩展卡尔曼滤波的实时无标记点姿态估计. 见《2018 年国际实验机器人研讨会论文集》, 第 762–771 页. Springer, 2020.

[LMD<sup>+</sup>22] Bhrigu K Lahkar, Antoine Muller, Raphael Dumas, Lionel Reveret, 和 Thomas Robert. 无标记点运动捕捉系统在拳击中估计上肢运动学的准确性. Frontiers in Sports and Active Living, 4:939980, 2022.

[MB21] Benjamin Michaud 和 Mickael Begon. biorbd：一个用于分析和模拟人体生物力学的 C++、Python 和 MATLAB 库. Journal of Open Source Software, 6(57):2562, 2021.

[MUSD13] Matthew Millard, Thomas Uchida, Ajay Seth, 和 Scott L Delp. 屈伸计算肌肉：肌腱动力学的建模与仿真. Journal of Biomechanical Engineering, 135(2), 2013.

[NFB<sup>+</sup>18] Jonathan J Noble、Nicola RD Fry、Carly R Bingham、Rebecca H East 和 Adam P Shortland. 实用临床的上肢运动学模型. *Proceedings of the Institution of Mechanical Engineers, Part H: Journal of Engineering in Medicine*（《机械工程师学会会报，H 辑：医学工程杂志》），32(2):207–212，2018。

[NK01] RR Neptune 和 SA Kautz. 肌肉激活与去激活动力学：快速周期性人体运动表现中的支配性特性？ *Exercise and Sport Sciences Reviews*（《运动与体育科学评论》），29(2):76–81，2001。

[NWK20] Wesley Niswander、Wei Wang 和 Kimberly Kontson. 用于下肢关节运动学测量的 IMU 传感器放置优化. *Sensors*（《传感器》），20(21):5993，2020。

[SBG<sup>+</sup>20] Bartolomeo Stellato、Goran Banjac、Paul Goulart、Alberto Bemporad 和 Stephen Boyd. OSQP：用于二次规划的算子分裂求解器. *Math. Program. Comput.*（《数学规划计算》），12(4):637–672，2020。

[SBS<sup>+</sup>19] Joe Sarsfield、David Brown、Nasser Sherkat、Caroline Langensiepen、James Lewis、Mohammad Taheri、Christopher McCollin、Cleveland Barnett、Louise Selwood、Penny Standen 等. 基于深度传感器的姿态估计算法在技术监督康复应用中的临床评估. *International Journal of Medical Informatics*（《国际医学信息学杂志》），121:30–38，2019。

[SDLC22] Zhenyu Shuai、Anqi Dong、Haoyang Liu 和 Yixiong Cui. 用于量化功能性运动中下肢关节角度的惯性测量系统的可靠性与有效性. *Sensors*（《传感器》），22(3):863，2022。

[SJC<sup>+</sup>09] Patrick Salvia、S Van Sint Jan、A Crouan、L Vanderkerken、Fedor Moiseev、Victor Sholukha、Celine Mahieu、Olivier Snoeck 和 Marcel Rooze. 两种方法进行肩部解剖学标志点标定的精度比较：类石膏协议与一种新的解剖学触诊方法. *Gait & Posture*（《步态与姿势》），29(4):587–591，2009。

[SNPW08] Pascal Spincemaille、Thanh D Nguyen、Martin R Prince 和 Yi Wang. 用于实时导航处理的卡尔曼滤波. *Magnetic Resonance in Medicine*（《磁共振医学》）：国际磁共振医学学会官方期刊，60(1):158–168，2008。

[TCF<sup>+</sup>18] Alessandro Timmi、Gino Coates、Karine Fortin、David Ackland、Adam L Bryant、Ian Gordon 和 Peter Pivonka. 基于低成本 Microsoft Kinect v2 传感器的新型标志点跟踪方法的精度. *Medical Engineering & Physics*（《医学工程与物理》），59:63–69，2018。

[TR20] Matt Topley 和 James G Richards. 当前可用的光电运动捕捉系统的比较. *Journal of Biomechanics*（《生物力学杂志》），106:109820，2020。

[US22] Thomas K Uchida 和 Ajay Seth. 结论还是错觉：量化基于标志点的运动捕捉逆分析中由于标志点配准与模型缩放误差引起的不确定性. *Frontiers in Bioengineering and Biotechnology*（《生物工程与生物技术前沿》），10:874725，2022。

[VFK<sup>+</sup>22] Robin Verschueren, Gianluca Frison, Dimitris Kouzoupis, Jonathan Frey, Niels van Duijkeren, Andrea Zanelli, Branimir Novoselnik, Thivaharan Albin, Rien Quirynen, 和 Moritz Diehl。acados——一个用于快速嵌入式最优控制的模块化开源框架。《Math. Prog. Computation》，14(1)：147–183，2022。

[VRD15] Athanasios Vanezis、Mark A Robinson 和 Nikolaos Darras。elepap 临床方案对上肢功能三维运动学评估的可靠性。《Gait & Posture》，41(2)：431–439，2015。

[VT00] Michael L Voight 和 Brian C Thomson。肩胛骨在肩部损伤康复中的作用。《Journal of athletic training》，35(3)：364，2000。

[WLB<sup>+</sup>16] Wen Wu、Peter VS Lee、Adam L Bryant、Mary Galea 和 David C Ackland。在肩部肌肉和关节功能评估中的特定受试者肌肉骨骼建模。《J. Biomech.》，49(15)：3626–3634，2016。

[WSH<sup>+</sup>16] Thomas Waltemate、Irene Senna、Felix Hulsmann、Marieke Rohde、Stefan Kopp、Marc Ernst 和 Mario¨ Botsch。虚拟现实闭环交互中延迟对感知判断和运动表现的影响。载于《第 22 届 ACM 虚拟现实软件与技术会议论文集》，第 27–35 页，2016。

[YPDQ09] Shihui Ying、Jigen Peng、Shaoyi Du 和 Hong Qiao。一种基于 ICP 的三维数据配准尺度拉伸方法。《IEEE Transactions on automation science and engineering》，6(3)：559–565，2009。

[YYS<sup>+</sup>16] Minxiang Ye、Cheng Yang、Vladimir Stankovic、Lina Stankovic 和 Andrew Kerr。用于远程康复的深度相机运动分析框架：运动捕捉与以人为中心的运动学分析。《IEEE Journal of Selected Topics in Signal Processing》，10(5)：877–887，2016。