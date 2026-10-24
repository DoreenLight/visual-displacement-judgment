文章

# 上肢与躯干运动学变异性分析：RGB-D传感器的精度与可靠性

Alessandro Scano <sup>1,</sup>\* , Robert Mihai Mira <sup>2</sup>, Pietro Cerveri <sup>3</sup>, Lorenzo Molinari Tosatti <sup>1</sup> and Marco Sacco <sup>1</sup>

1 意大利国家研究委员会（CNR）智能工业系统与先进制造技术研究所（STIIMA），Via Alfonso Corti 12, 20133 Milan, Italy; lorenzo.molinaritosatti@stiima.cnr.it (L.M.T.); marco.sacco@stiima.cnr.it (M.S.)

2 米兰理工大学（Politecnico di Milano），20133 Milano, Italy; robertmihai.mira@mail.polimi.it

3 米兰理工大学电子、信息与生物工程系，20133 Milano, Italy; pietro.cerveri@polimi.it

\* 通讯作者：alessandro.scano@stiima.cnr.it

收稿日期：2020年3月2日；录用日期：2020年4月23日；发表日期：2020年4月26日

![](images/61ecb72075b11b38445bc68f689e364da7433c59c5a6c05c6895c4715fae2745.jpg)

摘要：在运动分析领域，金标准设备是基于标记点的追踪系统（marker-based tracking systems）。尽管这类系统精度极高，但其高昂的成本、严苛的工作环境要求以及冗长的准备时间，使其不适用于小型诊所及工业应用等其他场景。随着以人为中心的方法在临床环境之外也得到了推广，对易用的人体运动追踪解决方案的需求日益迫切。在这一背景下，低成本设备（如RGB-深度（RGB-D）相机）被提出，旨在用于康复评估或工业环境中针对工人的以用户为中心的评估。在本文中，我们旨在比较用于人体运动追踪的基于标记点的系统与RGB-D相机。我们以Vicon系统（Vicon Motion Systems, Oxford, UK）作为金标准，分析了Kinect V2（Microsoft, Redmond, WA, USA）在上肢工作空间中多种手势下的精度与可靠性——针对康复和工作应用场景。该对比实验在由15名成年健康受试者组成的人群中进行。每位受试者需在康复和工业工作场景可能涉及的三个工作空间区域（中央、右侧和左侧）执行两类上肢动作（点到点运动和探索运动）。该方案旨在测试RGB-D设备的大范围视场角。我们在文中详细阐述的研究结果表明，RGB-D传感器足以用于上肢运动追踪以进行生物力学评估，尽管与基于标记点的系统相比，在评估某些特定自由度及手势时的可靠性仍存在显著局限性。

关键词：无标记采集系统；基于标记点的光学追踪系统；运动康复；人体运动追踪；Kinect V2；Vicon系统；上肢；生物力学

## 1. 引言

人体运动分析（Human Motion Analysis, HMA）是生物力学领域的一个分支，旨在定量描述人体运动，传统上通过运动跟踪技术进行采集。HMA 的应用涵盖从临床和运动康复到工业及娱乐等诸多领域 [1]。在人体运动康复领域，从运动跟踪获取的数据被用于制定医疗治疗方案和进行评估 [2]。该领域最精确的标准系统是基于标记点的光电跟踪设备。此类系统通过将标记点放置在受试者身体的解剖学标志上，能够精确重建运动学参数。许多研究描述了光电跟踪设备的各种用途 [3–9]。然而，这种高精度是通过耗时的标记点定位过程实现的，这不仅限制了其使用环境（如仅限于实验室），而且成本高昂。这些问题将此类系统的应用局限在科研实验室和医院中，且由于环境可控性较差，阻碍了其在小型诊所、家庭康复以及工业领域的应用。鉴于这些局限性，对设置时间更短、采集环境更灵活且成本更低的系统的需求应运而生 [1]。在过去的二十年里，研发出了许多能够在三维（3D）空间中提供精细人体运动学和/或动力学数据的新型仪器，例如可穿戴传感器或光学运动采集系统，其中一些已在临床环境和康复场景中进行了测试 [10,11]。惯性传感器和 RGB 深度（RGB-D）传感器的发展催生了满足这些要求的设备。RGB-D 传感器通过传感器融合机制实现这一目标，并通过结合深度分析的视频帧分割来重建人体运动 [12]。RGB-D 设备的一个典型代表是微软的 Kinect V2。它具有安装设置快捷、可采集大范围运动以及具备良好运动跟踪能力的特点 [13,14]。一些研究证实，与基于标记点的系统相比，Kinect 和 Kinect V2 等设备在记录上肢运动学方面具有良好的可靠性 [15,16]。以 Kinect 为代表的低成本 RGB-D 系统在康复和健康领域的应用涵盖了广泛的研究，包括生物力学评估 [17–27]、定量临床量表转化 [28–30]、脑卒中后患者评估 [31–34]、家庭环境适用性验证 [35–37] 以及辅助生活 [38]。

直到最近，人体运动追踪才在工业领域获得关注。近期的技术路线图 [39,40] 指出，需要采用以人为中心的方法和解决方案来改善工作场所中工人的健康并预防工伤，并指出了运动追踪在工作环境中评估用户的适用性 [41]。因此，近年来，随着运动追踪技术在标准临床环境之外变得广泛普及和适用，人们对其产生了越来越浓厚的兴趣。第一个应用领域是工伤预防：有研究表明，由于工业中的大多数体力劳动都需要在不受监控的环境中进行复杂动作，这可能会导致受伤和与工作相关的疾病 [42]。其次，运动追踪系统可以提供身体受压和应变的数据，从而改善工作站的安全性 [42]，帮助避免不必要的工作风险，并确保人机交互（human–robot interaction）的安全与成功 [43,44]。此外，追踪技术已被用于为设计人员提供适当的数据，以帮助他们改进产品，使产品更易于使用且对人们来说更加舒适 [45,46]。近期的一项工作 [47] 展示了追踪技术的离线应用，该工作提出了一种在更有可能出现人类协作者的空间区域内改变机械臂行为的方法，其中占用的空间体积是根据预先录制的人体运动数据计算得出的。

工业和康复运动捕捉的另一个重要方面涉及可能产生干涉的环境，例如在与机器人交互时。在某些人机交互中，要求机器与操作员协同工作或模仿操作员；为了避免有害或非预期的交互，运动追踪至关重要 [48]。一些研究利用了这一概念。例如，有学者提出了集成于人机交互背景下的运动追踪方法 [49]。

最后，值得注意的是，近期许多实验室研究在测试用于辅助体力劳动的设备时，采用了基于追踪系统的仿生方法来评估人机交互 [50]。这些近期的研究旨在将生物医学技术转化应用至工业环境中。

鉴于人体运动追踪在临床环境中的重要应用，以及即使在生物医学领域之外对以人为本方法的日益增长的关注，一些研究旨在评估 RGB-D 设备的可用性，并评估其相对于金标准系统的准确性和可靠性 [16,18,51–53]。在将基于标记点的系统与 Kinect $[ 1 6 , 1 8 ]$ 进行对比时，作者展示了 RGB-D 传感器良好的灵敏度，尽管他们的研究集中在小样本受试者以及仅限于伸展触及动作上。在一项研究 [51] 中，作者明确指出使用基于标记点的系统和 Kinect 设备记录的步态参数之间存在高度相关性，并得出结论认为它可以用于追踪步态参数，但对于其在临床环境中的使用，他们建议需要进一步提高其灵敏度。一项研究对 Kinect V2 在集中于步行和平衡控制的全身采集中的准确性进行了评估 [52]，得出的结论是 Kinect V2 可以达到用于运动分析的良好精度水平。另一项工作 [53] 仅集中于上肢的某些自由度。他们测试了该传感器的准确性，发现 RGB-D 传感器可以逼近基于标记点系统的结果，尽管所分析的动作范围相当有限。一项全面的评估 [54] 详细分析了 Kinect 在动作中相对于 Vicon 系统的表现。作者得出结论认为该系统适用于运动追踪，但所分析的自由度与旨在强调每个特定运动原语的特定手势相关联——这是追踪的一个有利条件。总体而言，我们发现只有有限数量的研究侧重于评估 RGB-D 传感器在多个工作空间区域内评估上肢运动的有效性，而这些区域类似于日常生活或工作活动中各种可能的任务。特别是，我们注意到该领域以往的研究主要集中于范式化的康复任务、步态或功能性动作，而不是绘制工作和现实生活应用中典型的各种上肢手势。因此，鉴于 RGB-D 传感器可以在运动评估背景下促进以人为本的视角，在这项先导研究中，我们实施了 Kinect V2（作为广泛用于运动追踪的知名 RGB-D 传感器）与光电标记点 Vicon 系统（作为金标准）之间的对比。其目的是在工作环境典型手势的扩展模拟中，评估上肢和躯干运动的准确性和可靠性，特别是针对与将康复评估扩展到工业领域相关的新型应用。

## 2. 材料与方法

## 2.1. 参与者

招募了 15 名年龄相近（19–35 岁）的年轻成年人（8 名女性，7 名男性，平均年龄 $2 4 \pm 3$ 岁，平均身高 $1 . 7 2 \pm \ : 0 . 1 0$ m，平均体重 $6 3 \pm 1 2 \mathrm { ~ k g } )$。参与者均无影响其表现的肌肉骨骼损伤。涉及人类参与者的研究已通过 CNR 伦理委员会（意大利罗马，方案编号 0044338/2018）的审查和批准。患者/参与者提供了参与本研究的书面知情同意书。已从相关个人处获得关于发表本文中包含的任何潜在可识别图像或数据的书面知情同意。

## 2.2. 设备

测试在意大利莱科的意大利国家研究委员会（CNR，Italian Council of National Research）运动采集实验室中进行。实验室配备有：

1. 一套由 10 台红外摄像机组成的 Vicon Vero 系统；一组用于与 Vicon 系统配合进行运动跟踪的反光标记点。在本实验条件下，共使用了 34 个标记点（25 个用于上肢模型，9 个用于目标）；

2. 一台用于在空间中跟踪人体的 Kinect V2.0 设备。Kinect 使用 RGB-D 摄像机以 30 Hz 的采样频率进行帧采集，并使用飞行时间红外摄像机进行深度传感。有关 Kinect 系统的更多深入信息，详尽细节见以往的研究 [55,56]。Kinect 安装在画架上，距离记录场景约 2.5 m，以获得最佳跟踪效果 [52]；

3. 两台通用计算机：第一台连接至 Vicon 系统，包含用于采集和预处理跟踪数据的软件；第二台包含一个用 C# 编写的自定义软件，可直接与 Kinect V2 设备通信。它能够生成一个包含构成 SDK Kinect 骨架的 25 个兴趣点的文件；

4. 一个直径为 60 cm 的圆形目标，带有 9 个兴趣点，分别命名为 N、NE、E、SE、S、SW、W、NW 和 O。该目标用作受试者执行点对点和工作空间探索运动的参考 [57]。and O. This target w

实验装置的示意图如图 1 所示。

![](images/5ee8f540be804ea8f20e7789645148c590b09d930486a8987caa7e558962e028.jpg)  
Figure 1. 实验装置图解示意图。

## 2.3. 动作选择

每位受试者都被要求用右臂执行两种类型的动作。所选任务为点到点运动和手部探索运动 [57]，常用于运动康复背景中。这些是从日常生活活动中推演出的动作，同时也是康复动作。这些是从日常生活活动中推演出来的动作，同时允许<sup>允许</sup> <sup>探索</sup> <sup>至</sup> <sup>上肢</sup> <sup>可用</sup> <sup>工作</sup>探索至上肢可用工作空间的极限。

<sup>点到点</sup> <sup>运动</sup> <sup>(Figure</sup> <sup>2,</sup> <sup>顶行)</sup> <sup>是</sup> <sup>临床医生</sup> <sup>用于</sup> 点到点运动（Figure 2，顶行）是临床医生用于测试患者运动能力的范式动作。选择它是因其在康复领域的广泛应用 [28]。该动作涉及多关节协调，并常用于日常生活活动中。受试者朝向圆形目标上的特定点，然后返回起始位置。受试者首先从预设的静止姿势开始，上臂贴靠身体，然后伸向目标动作，每位受试者执行的动作为伸向“O”点，然后返回圆形目标上特定点的静止位置，接着返回起始位置。每位受试者执行的第一个动作是伸向“O”点，然后返回静止位置。随后，受试者指向“NE”，返回静止位置，然后按顺时针方向继续此运动模式，直至到达“N”点，然后再逆向返回。

手部探索任务（Figure 2，底行）是另一种旨在模拟工作表面运动以及用于远端肢体协调的物体位移的动作。圆形目标再次被用作引导受试者运动的参考。每位参与者的目标是从静止位置开始，使手部沿径向穿过目标，到达圆周上的不同点。该动作的静止位置为身体直立且手指向<sup>参与者</sup> <sup>是</sup> <sup>从</sup> <sup>静止</sup> <sup>位置</sup> <sup>开始</sup> <sup>并</sup> <sup>沿径向</sup> <sup>移动</sup> <sup>手部</sup> <sup>穿过</sup> <sup>目标</sup> $^ \prime \mathrm { O ^ { \prime } }$ 点。第一个动作是朝向 $' \mathrm { N E } ^ { \prime } ,$ ，然后是 $' \mathrm { O ^ { \prime } } _ { \mathrm { \ell } }$ ，该运动模式按顺时针方向进行，直到受试者到达“N”，最后返回 $^ \prime \mathrm { O ^ { \prime } }$。

(f)  
![](images/79f4d2cc2b54756ce985b47e186cdbb6d7369b71098587e10d66645de26caa16.jpg)

![](images/7f8c249a6f871a0ea70094ccdeb7545711279c5a336f4d3e05c7fd4311265e41.jpg)

![](images/7a0e7e6975c5357130fdf3e222c47f39bb1b557b2f4cdd65e0e602118f9d83c9.jpg)

![](images/5bad32be98768111e1ca2a5d205f67f0ae914c926bb9c811a0ede398235c6e00.jpg)

![](images/16f3ce6d95365f9570bfc2ab96b2d6213cd4982ef648131bab494b2e2f78a12f.jpg)

![](images/d90b49d83eb8eedc775480ae86d2be91e48f2d944362a571f715d358b4e09b78.jpg)  
Figure 2. 工作空间扇区和运动的图形表示。上行：点对点运动：(a) 中央扇区，(b) 右侧扇区，以及 (c) 左侧扇区；下行：探索运动：(d) 中央扇区，(e) 右侧扇区，以及 (f) 左侧扇区。

## 2.4. 实验设置

设备放置在 Vicon 系统的采集空间内。受试者正对 Kinect。为了测试该设备的视场，圆形目标被放置在三个不同的位置：受试者右侧、左侧以及参与者正前方的水平面上。无论目标处于何种位置，动作始终使用右上肢进行。对于右侧和左侧扇区，目标放置在三脚架上并进行调节，使“O”形目标位于受试者的肩部高度。而在中央扇区，目标则水平放置在桌面上。工作空间扇区如图 2 所示。

工作空间内有一张桌子、目标物、两个三脚架以及 Kinect。两台分别运行 Vicon 和 Kinect 的通用计算机放置在采集区域之外，以免干扰采集过程。工作空间内仅有的物体是一张桌子、目标物、两个三脚架以及 Kinect。

## 2.5. 数据采集

在每次记录前，受试者均按照为 Vicon 系统设计的上肢模型穿戴标记点。共有 25 个标记点 [53]，其中躯干放置 5 个，双肩各 1 个，双上臂各 3 个，双肘各 2 个，双前臂和手腕各 4 个（Figure 3）。在此期间，Kinect V2 开启预热约 20 min。

数据集是在前述环境中通过 Kinect 和 Vicon 共同采集的。由于两个系统的软件运行在不同的计算机上，因此在记录期间数据的采集并未实现同步。我们选择始终先启动 Vicon 记录，随后开启 Kinect 数据流。在操作员发出信号后，受试者开始动作。

两种不同的采集系统提供了特定的数据结构。Kinect 数据集由来自 25 个解剖学关键点的 Microsoft SDK 2.0 骨骼 3D 数据组成（Figure 3）。用于与 Kinect 交互的自研软件提供了一个可用 MATLAB（Mathworks, Natick, MA, USA）读取的文件。该文件包含了 SDK 骨骼点相对于 Kinect 参考系统的 3D 坐标（与先前的一项类似研究 [27] 相仿，在第一次采集后约 2 min 进行）。

![](images/411931f85be00d318c65f50cb1c0b362b52592adf0afbcc3ec13d2b509892347.jpg)  
<sup>Figure</sup> <sup>3.</sup> <sup>Kinect</sup> <sup>V2</sup> <sup>跟踪的关节点</sup> <sup>(a)</sup> <sup>以及上肢</sup> <sup>Vicon</sup> <sup>系统中的标记点位置</sup> Figure 3. Kinect V2 跟踪的关节点 (a) 以及上肢 Vicon 系统中的标记点位置 (b)。

使用 Vicon 系统记录的数据集需要在 Nexus 虚拟环境（Vicon 软件）中进行预处理，包括标记点跟踪和标注。随后，利用上肢模型，该软件能够精确重建左右上肢的盂肱关节中心（肩部旋转中心）、肱尺关节中心（肘部旋转中心）和桡腕关节中心（腕关节中心）的位置，并估算躯干的位置 [58]。该跟踪允许对 11 个自由度进行估计（详见 2.6 节）。预处理完成后，数据以 c3d 文件格式保存并导入 MATLAB 中。

为了进行重测（test–retest）对比，每次采集均重复两次；第二次试验在第一次试验后约 2 min 进行，其方式与此前的一项类似研究相似 [27]。

## 2.6. 数据分析

所有计算得到的角度数据（Vicon 和 Kinect）均使用 5 Hz 的低通巴特沃斯（Butterworth）滤波器进行滤波 [59]。随后，由于两个系统以不同的采样频率（Vicon 为 100 Hz；Kinect 为 30 Hz）和采集时间（Vicon 比 Kinect 提前约 1 s 开始，并在 Kinect 之后 1 s 结束）采集数据，因此对这两个数据集进行了对齐。第一步是运动相位的检测：每次采集都被细分为若干子运动（手臂点对点前向相位包含 9 个子运动，工作空间探索包含 8 个前向相位和 8 个后向相位）。运动相位的划分按如下方式实现：计算手腕标记点（末端执行器）的曲线横坐标和速度曲线。随后，通过基于阈值法和速度曲线局部极小值检测的算法实现每个相位起始和结束的同步。接着，在起始点对齐后，来自每个相位（O->NE、NE->O . . . ）的数据

被重新采样，使每个相位的长度均为 100 个样本，以便进行变量间、区域间以及受试者间的比较。

为了对这两个系统进行比较，利用右上肢旋转中心的位置计算了一组包含 11 个变量的集合，即描述上肢运动的旋转自由度（DoF）。考虑了以下自由度（Figure 4）：肩部抬高、肩部沿垂直轴旋转、肩部内旋–外旋、肘部屈曲–伸展、手部屈曲–伸展、手部旋前–旋后、手部偏斜、肩胛骨抬高、躯干扭转、躯干前屈–后伸以及躯干侧屈。箭头的方向表示“正”角度。为了避免参考系配准的问题，所有变量均相对于受试者特异性参考系进行计算。

![](images/ce92d5cee336de9fa49cb99417af7122d962d0848fdf0a7d60a9fd45108969a6.jpg)  
Figure 4. 使用 Vicon 和 Kinect V2 计算的自由度的图形表示：1. Figure 4. 使用 Vicon 和 Kinect V2 计算的自由度的图形表示：肩部抬升，2. 肩部沿垂直轴旋转，3. 肩部内旋–外旋，1. 肩部抬升，2. 肩部沿垂直轴旋转，3. 肩部内旋–外旋，4. 肘部屈曲–伸展，5. 手部屈曲–伸展，6. 手部旋前–旋后，7. 手部 4. 肘部屈曲–伸展，5. 手部屈曲–伸展，6. 手部旋前–旋后，7. 手部偏斜，8. 肩胛骨抬升，9. 躯干扭转，10. 躯干前后屈曲，11. 躯干偏斜，8. 肩胛骨抬升，9. 躯干扭转，10. 躯干前后屈曲，11. 躯干 <sup>内侧–外侧</sup> <sup>屈曲，</sup> <sup>a.</sup> <sup>手部</sup> <sup>参考</sup> <sup>向量，</sup> <sup>b.</sup> <sup>受试者</sup> <sup>参考</sup> <sup>系统。</sup> <sup>箭头的</sup> <sub>内侧–外侧屈曲，a. 手部参考向量，b. 受试者参考系统。箭头的</sub> <sup>方向</sup> <sup>表示</sup> <sup>“正”</sup> <sup>角度。</sup> <sup>计算</sup> <sup>过程</sup> <sup>和</sup> <sup>投影</sup> <sub>方向表示“正”角度。计算过程和投影</sub> 平面的定义见附录 A。

## 2.7. 结果测量与统计分析

为了衡量 Kinect 的准确性，所选的比较指标是分别使用 Vicon 和 Kinect 系统逐阶段计算的相应自由度之间的角距离（angular distance, ad）。隐含的假设是将 Vicon 作为金标准。首先，生成了展示所考虑的各个自由度（DoF）的 ad 的方向饼图，其中考虑了方向以及各个扇区。对于点对点运动，方向饼图具有 8 个基本方向，对应于运动的前向阶段；对于探索运动，我们还考虑了后向阶段，因此饼图的分辨率增加了一倍。设计了热图以展示 Vicon 与 Kinect 之间 ad 的大小。

然后，将所有的 ad 数据汇总到三个矩阵中，每个扇区对应一个矩阵，每个矩阵包含每个自由度（DoF）和所计算阶段的 ad。这三个矩阵被用作双因素方差分析（two-way ANOVA）的参数，以自由度（维度：11）和扇区（3个：中央、右侧和左侧）作为因素对 ad 进行比较，从而观察 Kinect 相较于 Vicon 对各个 DoF 的追踪情况，并量化 RGB-D 传感器在每个自由度和扇区中的性能。假定原假设正确的前提下，p 值（p）确定了获得至少与测试中实际观察到的结果一样极端的结果的概率。对点对点运动和探索运动分别重复进行了测试。还执行了事后检验（MATLAB 函数 multcompare），以确定哪些自由度和/或扇区与其他存在差异。事后检验采用 Tukey 真实显著性差异（honest significant difference，HSD）准则。该方法在等方差和统计独立的假设下，成对比较感兴趣的变量。结果提供了成对的相似性置信区间。所有统计检验的显著性水平均设为 0.05。

为了衡量 Kinect 的重复性，所选的比较指标是在测试和重测实验中，由 Kinect 系统获取的相应自由度之间逐阶段计算的角距离（ad）。

为了展示这一点，绘制了用于描绘在测试和重测条件下对比 Kinect 性能的 ad 的方向性热图饼图。对于点对点运动，方向饼图包含八个基准方向，对应运动的前向阶段；对于探索运动，我们还考虑了后向阶段，因此饼图的分辨率翻倍。设计热图旨在展示测试与重测之间 ad 的大小。

为了给统计结果的信度提供一个度量指标，利用每次采集均重复两次这一条件，还进行了重测分析。为了通过测试-重测数据集量化信度，选择了数据集之间的组间相关系数（interclass correlation coefficient，ICC）[54]。还执行了事后检验（MATLAB 函数 ‘multcompare’），以确定哪些自由度和/或扇区与其他存在差异。事后检验采用 Tukey 真实显著性差异准则（HSD）。该方法在等方差和统计独立的假设下，成对比较感兴趣的变量。结果提供了成对的相似性置信区间。所有统计检验的显著性水平均设为 0.05。

## 3. 结果

## 3.1. 基于标记点的系统 vs. RGB-D 传感器

本节展示了 Kinect 相较于基于标记点系统的准确度，以及对所有受试者获得的统计结果。

一系列圆形目标的视觉表征通过热图展示了在各个特定自由度（DoF）、各区域以及相对于目标方向性方面，Kinect 相比于 Vicon 系统的性能表现。文中报告了采用本研究所建立的实验方案所获得的、在所有受试者间平均的 Kinect 设备平均精度（跨两次重复测试）。首先，报告了中央区域点对点任务的平均数据误差（所有受试者）（Figure 5a）。随后报告了右侧区域（Figure 5b）和左侧区域（Figure 5c）的相同结果。对于探索任务，同样报告了中央区域（Figure 6a）、右侧区域（Figure 6b）和左侧区域（Figure 6c）的结果。

![](images/e169514c26872ae4adedb4aad6ebeaba06691cd519fac3d29b9b852b2fe503f9.jpg)

![](images/4fbb30bf0b56d8de41f43ce2a66067bbaf0efb1f539574a7fc97f864d72114aa.jpg)  
沿垂直轴的躯干扭转

![](images/2a5e39fd28ade81bbfb587c233ea3477b2f57f3c6ec99683d703548a4cb494c0.jpg)

![](images/b840496a8e03de02614d1a6f037be67d983ac9fb5b150950ea6dd79962703c31.jpg)  
沿垂直轴的肩部旋转 肩部内旋-外旋

![](images/a6d36173f84ae47bf9d2741c7adf5fb132cd88736b16bd8be3add41317660afd.jpg)

![](images/40cc59bfb2697e9346ef8774ab38e3d7ac46e3c55d6ba981ee08aff3cd581286.jpg)

![](images/d8b1b1576667b32422c5a6ecd8b586b2c51c82d280c1192231701db22f577137.jpg)

![](images/02007c46772277dd20675c61aa895e7698b8f5b693ced1222029899ade32b887.jpg)

![](images/af3f9bc6c456260d14e83424c2400e5a7f943c925215b3b5542f064f9689e729.jpg)

![](images/f97e192d1a566d83d2d782a91ede314659264b40e44189458d7188e8edf59d7f.jpg)

![](images/8591e77619fb8132ad04e819673b180a911366c1192a02cf4c56dd26f00ee58c.jpg)

![](images/843d06cddd6fe55ea379b9e94c4d01add2da9462f738647383b74edab744bcfd.jpg)

![](images/832e7377046005efe57693b6a4897cffa0c1f1686a78198fd577bd9521fbf088.jpg)  
沿垂直轴的肩部旋转 肩部内旋-外旋

![](images/98d4ab5147e1173a81d800510cd310661d047cd42e87288c3829479c3a285812.jpg)

![](images/e46ed832062fbe2e444dbcc4e538152befff1e2a8b1f01517089ece5d5a39063.jpg)

![](images/718aeeb11580fb1c75406be734e688098a256cd332ac1303f8be5cc994b84c9d.jpg)

![](images/964ea0f207152ab27e03771d842974435ee311e9a1fcf65b827890158ac68583.jpg)

![](images/3639b78c49181eb86cad5c5517a6bb539280a207a7af55a31c4ceff5810218b4.jpg)

![](images/6aeab6668c75ce759894cc941a776c64eaaad2c5e38b2d253fca8d990f60c5f2.jpg)

![](images/31a0bc0e8d1543c4f1f00b54c466fc6fb626e98ccd2e01220779e979874095d7.jpg)

![](images/1cca395c3d16514f164961ca587c1aef65c55ae65a50291a6f7daf4e5a63284e.jpg)

![](images/cfd51de3f11a4a251b41b31de78bed8c7346eebda24826b67da137f0dac9b0d4.jpg)

![](images/925500063d4ba2c31af019e75de6f276de4dffb69f2aad284c5a46cf84d744fe.jpg)

![](images/5eda6aab4713afd26e6a3e50c78d77ae338422a65451ff29998a8814620ca7bc.jpg)

![](images/26aa67b07b892be741537a9d60b6b596ed30b97e653a0ff5466e713978050484.jpg)  
沿垂直轴的肩关节旋转 肩关节内旋-外旋

![](images/a21bdd4b5681d41470484724492f39ad1e65ef3ed2675b6e82d76ea8ec7f21a7.jpg)

![](images/01ffefa60135383cd7c83a61d88726270cc72a5fd6195b4357f523a26625ef30.jpg)

![](images/59264b8c28081d0bd32cb1a746340a055a2ad188a6b58b1ca9a3bc6c9fcda200.jpg)

![](images/51164389fca034e3ca369c81fbfc90be51856efaa003518f467b8beae2a1ecc6.jpg)

![](images/9bae5ace6761446d6b123988f710d73d60c3ee6656a03a00e92e3c743d87b04b.jpg)

![](images/d31f15fa453176d96d3c7193c4075c649c82ca84e2ac9d624555cbef756b23d9.jpg)

![](images/9c66a48a1eb930e2fc75372bbb0b0d949a7c17f2596d60a1a891a7baf272dfb8.jpg)

![](images/2153287a37b12f6e63c957726b00d8f858fa989564a8992c06562461b12dad19.jpg)

![](images/0a2d6262ac0e19fb65dd06b70ad4593cd13aa88a15d8b92385b4035b15b6896b.jpg)

![](images/804e2c4f808ecde0527ed669f07084a25cd47569eb9e9b3e0e69aa16d02594a5.jpg)  
Figure 5. 点对点运动中 Kinect V2 与 Vicon 角距离的饼图（作为测试与重测实验的平均值）。从上到下依次考虑的区域为 (a) 中间区域，(b) 右侧区域，(c) 左侧区域。方向饼图包含 8 个基本方向，对应运动的前向阶段。图形表示采用热图来描绘角距离的大小。

沿垂直轴的肩关节旋转 肩关节内旋-外旋

沿垂直轴的肩关节旋转 肩关节内旋-外旋

![](images/6c102316bc61cb4b31dccca868c8c3d9ee061c46943417d599d26112703217e8.jpg)

![](images/55bef8280e9433a805e9aa29082e2f77464878629133f6de01edb1712509456d.jpg)

![](images/98a98b9a50edcf4875bb285c7095b65044dd84fc75443e7114b4a32d58b7c0e9.jpg)

![](images/ed6fd3286bd983ab64f0c629faf8ab172e3ce6402a06ab6ba7fa24ea801ae055.jpg)

![](images/65c6c41264fa0642cec8f88fb0617ddaeef5b20a859744e209595910d4ed2ed6.jpg)

![](images/23cff556c09658c26a1e6e4a2837d0c54ac2f609498dc0b66bd26439be1cc9d5.jpg)  
沿垂直轴的躯干扭转

![](images/9c9ac38e7d413fa0d09be609d2bb90e6fc891366dbdbcfbaae3e2a3fa33ee01a.jpg)

![](images/f807489ed3e2aebf6bdf06c45c4a9ecf0886ab9af26298b4afbdd4dfa02ae30d.jpg)

![](images/d256d8a6d93dbf8f2288ef966948ad048628af2a6e3fb9d862e03418f24464e4.jpg)

![](images/5c5f9a1ae2293f1a5fe53bf0ccd7697785c6f066d2e14a4764f4b72a8bc56b9e.jpg)

![](images/3526fc1773bcbd470b54d9f9c366cc8a5634da4e2e12d46180274b481f0bce05.jpg)

![](images/eec2a644fc9ad00a6d4db4084957de0758cf19753aa0a64dae4fe892004df728.jpg)

![](images/b6be2646b0eae1fc3564dfb1f74509e07d8d32b764bdabff9e0428e812f6b169.jpg)

![](images/12c124cbbf2bb2dcd9fb2dcc1fbe763309fbb515d7e4c9f2821b2315d0e7bbd9.jpg)

![](images/dfd0276695c194fd116692efddc35f2d42c3886d2ca4bec33745215b9ce02fa8.jpg)

![](images/29fa2bf3172ca0920823b863e74ce81e8fb6cd4a6505fe7f1b204bc3b45b97f3.jpg)

![](images/67af3a1c1f793a236dcffb76ce9892f85840f20fedf0d8ba45693a6a55bbf9b6.jpg)

![](images/7806d8fb9f3f91a7e30439d7ec06a388a3115804b31c7962d0b7d15a3efc4715.jpg)

![](images/d06b20e87b788b64b4309a9fdb2f562adbad100e4eab94213b240a6bcb6b6350.jpg)

![](images/4782b7822e53ab169d4f3693bd8f9272e51c83a0263c59673b3cd505ce96d64a.jpg)

![](images/c94733565db9eb74f62201e19b025a9c5c8923b9a30cece6ed44c268e1ec9e76.jpg)

![](images/2e9ae76034211d29463a3919cb2881be687e2805d44bb2b833d72e0d3877882f.jpg)

![](images/5cadaebf87e1ce4f9256aa63ff6f52a7e17e1ad038bd89be390ec18204794cef.jpg)

![](images/1b03691f3891ce70865a14a69eae34f56889ae43d48e1b8e926a0acbab0663cd.jpg)

![](images/93eff45c5cdf55e573da555894052b9116280fa9ebd00d0e17af4112a3c171a4.jpg)

![](images/e2e568878702f93cacc2921c81404ef536c5cc67578372a7473d650ece20e95e.jpg)

![](images/cce6e03ad48abe0987f0b41f730d3bdc12948c081203060dc913999949532308.jpg)

![](images/21a7b8159aa37129ba23a1c4be997dac361509c6950d8ec4c396d79612d95d44.jpg)

![](images/fdef1f28685f2885c123b7a725a60833fad2e2198627f13337884a3a22455932.jpg)

![](images/df2e58936a8aadfbda85359a37a73025f8f4f59eaca959f06d3a1b3934bdcaf7.jpg)

![](images/edbe51bcd40ec9ca8fd532d7b944c30c0fc06f0202f1402f73e65344dd7e0785.jpg)

![](images/4bd50500343cd0e5633e82eda3987a5f5d8f931d71733cb7d9c838d4888a0a77.jpg)

![](images/a3809d3995a5dd381c235571c8a760e23dab8396cd304f5703c6f5a583fc3b55.jpg)

![](images/d8903fd389de12556743960ed62c84bb567ff0cb493373c31a7062c62c47e129.jpg)

![](images/7757a82f4cb88d11900d924b28da7e0ac3a56747320aa7681bdbe20134d17f9f.jpg)

![](images/bced62263ef9c97d79b865bb819425a857b43dcae7991591135470ec5799ac03.jpg)  
Figure 6. 探索运动中 Kinect V2 与 Vicon 之间角距离的饼图，为测试与重测实验的平均值。从上到下，考量的区域依次为 (a) 中间区域、(b) 右侧区域、(c) 左侧区域。方向饼图具有 16 个基本方位。图形表示采用热图来描绘角距离（ad）的大小。

本节展示的结果与测试数据集相关（测试与本节展示的结果与测试数据集相关（测试与重测之间的比较见第 3.2 节））。双因素方差分析（two-way Anova）表明，平均而言，七个自由度（DoF）重测见第 3.2 节）。双因素方差分析表明，平均而言，七个自由度表现出的平均误差低于表现出的平均误差低于 $1 0 ^ { \circ }$。两个自由度表现出的平均误差范围在 。两个自由度表现出的平均误差范围在 $1 0 ^ { \circ } - 2 0 ^ { \circ }$ 之间，且对于两种执行的动作（点对点运动与探索运动）其中一个具有对于两种执行的动作（点对点运动与探索运动） $> 2 0 ^ { \circ }$。此外，一个自由度的误差大于此外，一个自由度的误差大于 $2 0 ^ { \circ }$，且在点对点和在点对点和探索运动之间具有不同的均值。探索运动。

对上肢点对点运动的更深入分析表明，躯干扭转、躯干前后屈曲、躯干内外侧屈曲、肩部上抬、肩部沿垂直轴旋转、肘部伸展以及肩胛骨上提的追踪误差均小于或等于 $5 ^ { \circ }$ 。手部偏斜角和手部屈伸的追踪误差范围在 $5 ^ { \circ }$ 和 $1 5 ^ { \circ }$ 之间。肩部内旋–外旋的平均误差约为 $2 0 ^ { \circ }$ 。最后，手部旋前–旋后角的平均误差大于 $2 0 ^ { \circ }$ 。此外，从点对点运动的分析中，肩部上抬的均值与肩部沿垂直轴旋转 $( p = 0 . 9 6 )$ 、肩胛骨上提 $( p = 0 . 9 9 )$ 、躯干扭转 $( p = 0 . 7 8 )$ 、躯干内外侧屈曲 $( p = 1 )$ 、肘部伸展 $( p = 0 . 5 9 )$ 以及躯干前后屈曲 $( p = 0 . 6 1 )$ 的均值之间未发现统计学差异。而肩部内旋–外旋、手部旋前–旋后、手部屈伸和手部偏斜彼此之间以及相对于其他角度均具有统计学上显著差异的均值 $( p < 0 . 0 0 1 )$ ）。同时，考虑各扇区之间的比较，我们发现右侧扇区的平均误差小于其他两个扇区，介于 $8 ^ { \circ }$ 和 9<sup>◦</sup> 之间，且与左侧和中央扇区存在统计学差异（分别为 $p = 0 . 0 1 2$ 和 $p = 0 . 0 0 5 3 )$ 。左侧和中央扇区具有相似的均值，且不存在统计学差异 $\left( { p = 0 . 9 6 } \right) ;$ ；平均误差在 $9 ^ { \circ }$ 和 $1 0 ^ { \circ }$ 之间变化。

对探索运动进行的相同测试在各自由度（DoF）之间的平均变化方面提供了相似的结果，但具有不同的统计学显著性。我们可以识别出一些组别：肩部抬高、肩部沿垂直轴旋转和肘部伸展被归为一组；事实上，我们无法发现与肩部抬高相比的统计学差异（分别对应肩部沿垂直轴旋转，$p = 1$ 以及肘部伸展 $p = 0 . 9 9 )$ ）。这三个自由度与其余变量均表现出统计学差异，三者均为 $p < 0 . 0 1$。手部旋前–旋后与手部屈曲–伸展之间无统计学差异 $( p = 0 . 8 7 )$，且与所有其他变量均具有统计学差异 $( p < 0 . 0 0 1 )$；随后，肩胛骨抬高、躯干扭转和躯干内侧–外侧屈曲无统计学差异，分别与躯干扭转相比 $( p = 0 . 9 4 )$，同时躯干前后屈曲仅在与躯干内侧–外侧屈曲相比时均值无统计学差异 $( p = 0 . 9 9 )$，但在与躯干扭转 $( p = 0 . 0 5 )$ 和肩胛骨抬高 $( p \leq 0 . 0 0 1 )$ 相比时具有统计学差异。最后，肩部内旋–外旋与手部旋前–旋后相互比较时均值具有统计学差异 $( p < 0 . 0 0 0 1 )$ ），且与其余变量相比也具有统计学差异 $( p < 0 . 0 0 1 )$。在探索运动的情况下，扇区间的误差分布有所不同：所有扇区均无统计学差异，分别为中央扇区 $( 1 1 . 1 3 ^ { \circ } )$、右侧扇区 $( 1 1 . 3 9 ^ { \circ } )$ 和左侧扇区 (11.18<sup>◦</sup>)。比较右侧和左侧扇区，结果显示两者无统计学差异 $( p = 0 . 7 9 )$ ）。右侧与中央扇区 $( p = 0 . 6 8 )$ 以及中央与左侧扇区 $( p = 0 . 9 8 )$ 亦是如此。

## 3.2. RGB-D 传感器可靠性

在本节中，我们描述了作为 RGB-D 传感器可靠性度量方法的重测信度（test-retest）。结果的概览见 Table 1 和 Table 2。

一系列圆形目标的可视化表示通过热图展示了 Kinect 在每个扇区中针对特定自由度（DoF）关于目标方向性的重复性。首先报告了中央扇区中点对点运动的平均角距离（Figure 7a）。随后报告了右侧扇区（Figure 7b）和左侧扇区（Figure 7c）的相同结果。对于探索运动，中央扇区（Figure 8a）、右侧扇区（Figure 8b）和左侧扇区（Figure 8c）也报告了相同的结果。

为了更全面地评估可重复性，还分别针对每个区域和每种运动计算了数据集之间的组内相关系数（ICC）。在点对点运动中，右侧区域的组内相关系数为 0.81 $( p = 0 )$，中央区域为 0.82 $( p = 0 )$，左侧区域为 0.73 $( p < 0 . 0 0 1 )$。同时，对于探索运动，右侧区域的 ICC 为 0.84 $( p = 0 )$，中央区域为 $0 . 7 5 \ : ( p = 0 )$，左侧区域为 $0 . 6 2 \left( p < 0 . 0 0 1 \right)$。

躯干绕垂直轴的扭转

躯干绕垂直轴的扭转

![](images/1b68a7c4e3efbe1f1a86732134af067021c0a157d9a18aea4497cd740c30173f.jpg)  
肩关节绕垂直轴旋转 肩关节内旋-外旋

![](images/82aa53bce43365b4e4f8864f4050500c52a6133b13b6a12fc2309a2bdfd2f9af.jpg)

![](images/d173794281cf8714ca9472de5d9c10c2b061c6313bc73f440aec3f7b7d0a5082.jpg)

![](images/061555197791765e2f41d9237fb12f149342173e538b5c10be81badf50d4b0d5.jpg)

![](images/7e0564f299670389036baab73553745b3c9018ca50a1584a093a660a9353ff08.jpg)

![](images/6df5d763fad6cd0d2ad740246e0585e70dab7f694bd8d390a9770705f25af210.jpg)

![](images/053cb1c07acb8cb2204bcf7f74ab4541b80bcc2791e6d42395589e7881018c98.jpg)

![](images/a2cfcad19ebc895a27b58b595e8f196b649236cf1fa919011baee22a1da93835.jpg)

![](images/deedf1074bf88bc2bdd4d603b14dd30760608a895c8a8ab541dba0e4e0813020.jpg)

![](images/154b4a265f8653f73144dfabd1cdf99cf021c76fabb5365aee2a7d7d2ad88b01.jpg)

![](images/7016e377a172f78ff03194de80185c1836ecb1763ab79e22ab8681cd810b60fa.jpg)

![](images/f2c75f1fa1d7dec55801b197ea83941cd4a3e564000ca2ce8816640928f03a30.jpg)

![](images/2c19b1876d78d4150fb6404d09fce52aa03fd3ddb528b98ca8b546cee3a760eb.jpg)

肩关节绕垂直轴旋转 肩关节内旋-外旋  
![](images/b8ecd629f2f4f3ca71efc5b31537980e615eebf39834da4b76f4d96050f2af15.jpg)

![](images/48bc7f6f8b4fb1f9f5ccee4f153455b6cc63e69c1dea81e5dbe9cdf4e210bb63.jpg)

![](images/139602dda419094102d3faf364dea9988c8104b03d25fd4ad0912737670a3341.jpg)

![](images/de23bbd321e7c3adb006d0bf2a9e74f02da1ddcc693f2980919d523e4728ece0.jpg)

![](images/fd06abd53d80ef1b365851513858550f788fb314d72dfbb0b74f49defae06e41.jpg)

![](images/b840b744251d880e19d17eba25c9db72587cf0fc5e88a03d4de2ca419f20c1dd.jpg)

![](images/817604e1543ef1caad30251b762a4c28c806c0c9043f65426d28c801a8d45c14.jpg)

![](images/8b1947f7db9cbde05196c207720e064558f41c1dd03e31ea50871e026da34229.jpg)

![](images/96c22a5f8532ec7ca4d9cfd8919ccdf7797a4287169ba78e8b5e65596407bcf0.jpg)

![](images/f67c8b1991e08e58bebd0972be4c1673a649e7fa34eeadf8aca98062f2e1185e.jpg)

![](images/96cdb1e2cca87a52bf615fe63a937abe15fa0dfa84d447bc5d3222f83f9d875d.jpg)

![](images/0abb2c4db85b986ab6b9da0fbf0f2a3026a25df96da6afc0315c1e1ca60f2386.jpg)

肩关节绕垂直轴旋转 肩关节内旋-外旋  
![](images/84b2ac7e48887c0bad3868ec2081a5b2110bacfd9053dadbccfbdf000c39f2af.jpg)

![](images/5fac81cceb096a849ae6db8b5ba86c422c2a07a10a99f69bb8b667dac9c3e01a.jpg)

![](images/d58b4bf38b7af4d524edf6e3a0b697d91ed61a940d4a541edc1a1727c717ef51.jpg)

![](images/cff18f58f6bd48fec23a68612ed4212c86ddae5842d383de44d80d385dcd7931.jpg)

![](images/15a35c8c6a3b42a941a42cf07f3641998ee4bbce27fa99872a619ca92e74524b.jpg)

![](images/53f8e9b4da6e874ab42c5910e8796abcbbc8d3b6f1135796fb4258e3271834cb.jpg)

![](images/bc4dd9e6f22d6c47ee3a3124cc305be49a4befec247771dee6337903bd512ccf.jpg)

![](images/253646e128de9149b4641fa4fe9b80b9c132fb0c749bffce84357a21594f6099.jpg)

![](images/1ff907a9c4f4684dccb624a42f2d34168f173ec5ea841203ab96be50047b7d48.jpg)

![](images/9581dc77b7b27da113b9545c5fcd2719a66ddb6d7a9d65d27762a7fdeea79b3c.jpg)

![](images/2f22dd04be6eedbb3cc06faf373ca753da27c8a555f2ff602af7ed48db56fcf6.jpg)

Figure 7. Kinect V2 在各次重复中点对点运动的角距离饼图。从上至下的分析区域依次为 (a) 中心区域、(b) 右侧区域、(c) 左侧区域。对于点对点运动，方向饼图包含对应于运动前向阶段的八个基本方向。图形表示采用热图来展示角距离（AD）的大小。

Figure 9 和 Figure 10 分别展示了点对点运动和探索运动统计分析的可视化表示。此外，还提供了测试与重测数据集之间的统计结果。如图 9 所示以及通过 ICC 所报告的，点对点运动在均值差异和显著性方面的结果均具有一致性。图 10 表明右侧和中心区域在自由度之间保持了一致性；同时，左侧区域存在轻微差异，但与其他区域相比未呈现出统计学显著性。

![](images/bd1e8fddab9dc3e5ded443d7b01bca6ec457793b5114a55bd99257b8d03a84df.jpg)  
肩关节绕垂直轴旋转 肩关节内旋-外旋  
肩关节绕垂直轴旋转 肩关节内旋-外旋  
躯干绕垂直轴扭转  
躯干绕垂直轴扭转

![](images/ababc6c79a1b2ddb66a9f588f46784decc1b6d17562316b662a0c9155403ca93.jpg)

![](images/cbdbdeb05cebe5d10b08017cd29a0f22936ca99aa14a19658113c43512e2601d.jpg)

![](images/b971bc4cc46dba2e90a8d817de8ec6f71d109233b8eb4457b3f567b2eca4dd2c.jpg)

![](images/e2c1b6ffbaf1f2c00bd95c48ce89274a1f3da5c037a1ea4a28707c3ca1e4f46e.jpg)

![](images/a94573a00eb88a32e595b389bb7df11320fccbfa195823f238b862aa0f8c29d9.jpg)

![](images/918756b8b17274e07e3b2410dbb5507dcaff154e875cc3fa3ef0e0f5d45ef5c9.jpg)

![](images/48d8024dc777f0b990bb2e4d1092783f766bfaf68599186978cf2672db97e12f.jpg)

![](images/d48545fdff98c61ec37d19ff16907aaaf9199babae1050860fa99dbc08935192.jpg)

![](images/babea308ab3c848f57ce95042b6e1683cf5866d1802958420ba53dd6c9851890.jpg)

![](images/816baa698ff453a9a17d1b59b06bfe59a0c0d380406f52c77b9ad7c5e5ec3df2.jpg)

![](images/1baf6e7770784218ac10d999dc697d9cdadd97b4ab1335ce91f3cc4edff56aa6.jpg)

![](images/73368a0fccc95be1ef7769e9b2a8db745955d52720fbe72e110161f42a033416.jpg)

![](images/4bdc7db631ab05c91e268893952cca8172712c1e7cbfafb1a1c071710f9bbb40.jpg)  
沿垂直轴的肩部旋转 肩部内旋-外旋

![](images/867f31cfb097a2d528100e6a64b90a9e9e67c4334581160233583756672fac93.jpg)

![](images/b582bad4c95788bd62620b3a1c27b417236067831cac41ad09a01b5581b509cb.jpg)

![](images/5153f04c6100f83c398f5eab2d72fbc732fdc73b84b79297961c5aaac394a62e.jpg)

![](images/3c4d8bcf96521064168868d9a870cc73f72afc095800d7c5b35e282602c68fe5.jpg)

![](images/f57bd75671a3678b893351a7de43452d01123d5e26aae83c7a4a6ecd7530af71.jpg)

![](images/d9d7f204029065f1934d7fabef41edfe37686f78b7cd3db096ee3eacd601a772.jpg)

![](images/62d96fcecbec8d1d0ca965a7be088688cf08773a3993d85dbd4b97fdb1a3855d.jpg)

![](images/9848229428dc00f3ee63c8d8b5372b5b76b1c02778865df2626576fc85972cd6.jpg)

![](images/dab6019cde0c3fc7419203be46f1bddbb5055a9367c4cebaf78e901edaaff8d0.jpg)

![](images/0a73ba75d5e2d71d23067020008590c9c0cb2a6d2b45093ddee9ca96824d860a.jpg)

![](images/3a3af00ed2bf5a6d79095f518829609386b191b676f1b7eec8e130ce39f7b80e.jpg)

![](images/9c115b8c7f1e104ac1c2fb2b2794394f49b2d1e67246fe46370c156da527c881.jpg)

![](images/6a2fbfedbba64d12a8b6311788147be4044e966cafd0daf6b290f50755fdad14.jpg)

![](images/4112ebc01afa2da667e76c77492493338cd8b2c3f205d7635257df0e7419c50b.jpg)

![](images/d48aa45bc44d99d866246d3a3b19453585eac8c29c8f6a1703486b2fda59daf8.jpg)

![](images/a63429a605215b87fba528239c7c6fa2fac5cbbe8b0f82d567bb2e46dd0c735b.jpg)

![](images/9bd6300c6bd8b7b0db9a64b98fcaf6435b4eeb65c0223f3a45786e3d404fcfb0.jpg)

![](images/1cbb6bc94c0cb4cf62df02704e0453d49f42ff51f0a21af4f098db5421d5ddef.jpg)

![](images/a8e2a62e6accf2ac756a0c27cfabd28982ead1a3e31228a77a71806a4190dfd9.jpg)

![](images/0c9497fdc76a52cc3a70b284a62e2366136391f6976faeb23c3b3fdde94fba91.jpg)  
<sup>Figure</sup> <sup>8.</sup> <sup>探索运动中</sup> <sup>Kinect</sup> <sup>V2</sup> <sup>各次重复间</sup> <sup>角距离的</sup> <sup>饼图。</sup> Figure 8. 探索运动中各次重复间 Kinect V2 角距离的饼图。 <sup>从上到下，</sup> <sup>所考虑的区域分别为</sup> <sup>(a)</sup> <sup>中心区域，</sup> <sup>(b)</sup> <sup>右侧区域，</sup> <sup>(c)</sup> <sup>左侧区域。</sup> <sup>对于探索运动，</sup> <sup>我们还考虑了</sup> <sup>返回阶段，</sup> <sup>因此</sup> <sup>饼图的</sup> <sup>分辨率</sup> <sup>包含了</sup> <sup>16</sup> <sup>个方向。</sup> <sup>图形表示</sup> <sup>采用了</sup> <sup>热力图</sup> <sup>来描绘</sup> <sup>角距离的大小。</sup> In order from top to bottom, the considered sectors were (a) central sector, (b) right sector, (c) left sector. For exploration movements, we considered also the backward phases and thus the resolution of the pie-charts included 16 directions. The graphical representation adopted a heatmap to depict the magnitude of the ad.

点对点双因素方差分析（2-way ANOVA）1 个自由度  
![](images/e6c1e951045d982c0a6a26946fa9928672742d6477015be13f48977435d412dd.jpg)

![](images/3fcb7ac44ec8da2b18ac36c657d99abf0be376d11b19b388078f090115d6d221.jpg)

点对点双因素方差分析（2-way ANOVA）2 个自由度  
![](images/6e628a06ccff7a93398fa6c67b29dca9dfdb41222c5a16a0beacc86b90d83ffe.jpg)

![](images/c8aa57887b5dfa478bf1221d72786a1438a9a6a8fea09bc46235f93703ae8b0d.jpg)  
Figure 9. 点对点运动：测试和重测数据集的双因素方差分析（two-way ANOVA）检验结果。

Table 1. 平均角距离，按自由度和区域（测试与重测）排序，以及组内相关系数（ICC）：点对点运动。

<table><tr><td>自由度</td><td>测试均值 (deg°)</td><td>重测均值 (deg°)</td></tr><tr><td>肩关节上举</td><td>3.96</td><td>3.82</td></tr><tr><td>沿垂直轴的肩关节旋转</td><td>4.73</td><td>4.88</td></tr><tr><td>肩关节内旋-外旋</td><td>19.00</td><td>18.18</td></tr><tr><td>肘关节伸展</td><td>5.20</td><td>5.36</td></tr><tr><td>手偏展角</td><td>11.25</td><td>11.60</td></tr><tr><td>手屈伸角</td><td>8.10</td><td>7.89</td></tr><tr><td>手旋前角</td><td>36.40</td><td>38.59</td></tr><tr><td>肩胛骨上提</td><td>3.64</td><td>3.70</td></tr><tr><td>躯干扭转</td><td>3.02</td><td>3.01</td></tr><tr><td>躯干前后屈</td><td>2.41</td><td>2.61</td></tr><tr><td>躯干侧屈</td><td>4.06</td><td>4.22</td></tr></table>

<table><tr><td>扇区</td><td>测试均值 (deg°)</td><td>重测均值 (deg°)</td><td>ICC</td></tr><tr><td>右侧</td><td>8.59</td><td>8.77</td><td>0.81</td></tr><tr><td>中央</td><td>9.55</td><td>9.73</td><td>0.82</td></tr><tr><td>左侧</td><td>9.60</td><td>9.82</td><td>0.73</td></tr></table>

探索性双因素方差分析（2-way ANOVA）2 个自由度  
Table 2. 平均角距离，按自由度和扇区（测试与重测）排序，以及 ICC：探索运动。

<table><tr><td colspan="2">自由度</td><td>测试均值 (deg°)</td><td>重测均值 (deg°)</td></tr><tr><td colspan="2">肩关节上举</td><td>7.09</td><td>6.92</td></tr><tr><td colspan="2">沿垂直轴的肩关节旋转</td><td>7.44</td><td>8.67</td></tr><tr><td colspan="2">肩关节内旋-外旋</td><td>25.63</td><td>26.93</td></tr><tr><td colspan="2">肘关节伸展</td><td>7.56</td><td>7.43</td></tr><tr><td colspan="2">手偏展角</td><td>10.70</td><td>10.47</td></tr><tr><td colspan="2">手屈伸角</td><td>9.69</td><td>9.70</td></tr><tr><td colspan="2">手旋前角</td><td>43.28</td><td>44.06</td></tr><tr><td colspan="2">肩胛骨上提</td><td>4.60</td><td>4.67</td></tr><tr><td colspan="2">躯干扭转</td><td>3.70</td><td>4.16</td></tr><tr><td colspan="2">躯干前后屈</td><td>1.71</td><td>1.67</td></tr><tr><td colspan="2">躯干侧屈</td><td>2.16</td><td>2.17</td></tr><tr><td>扇区</td><td>测试均值 (deg°)</td><td>重测均值 (deg°)</td><td>ICC</td></tr><tr><td>右侧</td><td>11.39</td><td>11.81</td><td>0.84</td></tr><tr><td>中央</td><td>11.13</td><td>11.45</td><td>0.75</td></tr><tr><td>左侧</td><td>11.18</td><td>11.34</td><td>0.62</td></tr></table>

![](images/c64589a89428c95520971de8efa99215d462c51f51309ebf007fb978e3b9ed52.jpg)

![](images/c01054dce472e7e1518eb5f07a9779858ea3bfe47eba94fcee5d1030890198b6.jpg)

![](images/573b9a21826ebcf7510632a5e6558633e5e8787289af210d18b40a5faacbfce9.jpg)

![](images/0a8b268418d2f4156d6d2c51083d1661f35a2926cb3d8bb26fac550ea5a06b06.jpg)  
Figure 10. 探索运动：测试和重测数据集的双因素方差分析（two-way ANOVA）测试结果。

## 4. 讨论

## 4.1. 结果总结

总体而言，RGB-D 设备能够针对大多数所考虑的关节角度追踪上肢和躯干运动。然而，在肩关节内旋-外旋以及前臂旋前-旋后中发现的角度差异不容忽视。正如所见，如果可以接受这种误差，Kinect V2 在执行手臂点对点运动时，对于追踪盂肱关节中心运动、肱尺关节中心运动以及躯干运动（特别是在右侧和中央区域）是很有用的。相反，我们的结果建议在记录探索运动时要更加谨慎，因为每个区域的平均误差略高，尤其是相对于点对点运动而言。

## 4.1.1. 自由度

大多数自由度的跟踪误差均低于 10<sup>◦</sup>。该结果与先前的研究 [18] 一致，在这些研究中，描述肩部抬升和肘部伸展的变量跟踪误差约为 5<sup>◦</sup>，但那是在更为受限的场景（单向伸展运动）下测得的。在健康人群中针对康复相关指标进行的准确性测试表明，大多数临床参数在 RGB-D 系统与基于标记点（marker-based）的系统之间表现出绝对一致性且无系统偏差 [52]。其他类似的研究 [53] 发现，大多数推断出的参数跟踪的 RMSE 均低于 $1 0 ^ { \circ }$，这与本研究所展示方法得出的结果相当。另有研究报告了对肩部跟踪更为精确的结果，但其侧重于受限的姿态范围 [20]。唯一表现出较低准确性的自由度是与手部相关的自由度。这一发现表明，在运动变异性较大或姿态不利的背景下使用 RGB-D 相机采集数据时应格外谨慎。事实上，在本研究中，有两个角度表现出了极高的误差：肩部内旋–外旋以及手部旋前–旋后。前者尤为关键，因为它是基于前臂在上臂横截面上的投影计算的。该投影高度依赖于肘部伸展角度，这意味着当手臂与上臂之间的夹角接近 0<sup>◦</sup> 时，由于手臂接近运动学奇点，肩部内旋–外旋的计算可靠性便无法得到保证。此外，在类似的构型中，关节中心重建的微小位移就可能在角度推断中产生不可预测的偏差。在本研究中，由于肢体经常处于完全伸展的奇点姿态（尤其是在探索运动中），这种情况更加凸显。这极有可能是总体结果相较于先前研究 [20,54] 更差的原因。很可能在工作空间的大范围探索中，相较于更受约束的场景，跟踪效果会更差。

由于使用 Vicon 和 Kinect V2 重建的骨骼形态不同 [54]，旋前–旋后角（pronation–supination angle）同样非常关键。利用 Kinect SDK 骨骼，该变量是通过从手腕出发并指向大拇指的单位向量的角度变化来计算的。该向量绕前臂轴线的角度变化可被假定为手部的旋前–旋后。导致难以推算该角度的一个关键问题是 Kinect 对大拇指的跟踪质量较差。我们的结果与之前一项全面研究的发现部分一致 [54]，表明与手部相关的精细动作可以受益于专门的手部模型（ad-hoc hand-models），而本研究并未采用这些模型。此外，在评估身体四肢末端（手和脚）时，先前的研究也发现了较低的信噪比 [52]，从而导致跟踪效果不佳。再者，我们所分析的动作对于手部旋前和旋后的可靠计算要求非常高。

从更广泛的讨论来看，一个相关的发现是，对于运动范围较大的动作（例如点对点动作，其跟踪效果优于手部探索动作），跟踪性能似乎更好。可以说，Microsoft 的内置算法使得更好地重建具有更大关节运动幅度的大范围动作成为可能。在对跟踪精度要求较高的环境中，或在精细动作要求高精度的场合对人体进行跟踪时，应考虑这些结果。

## 4.1.2. 扇区

迄今为止，对诸如 Kinect V2 等 RGB-D 传感器的测试，主要考虑的是该传感器在跟踪旨在强调特定自由度的手势时的性能 [54]，或者在康复导向的任务中的性能 [18,33,36,51]。在本研究中，我们的创新贡献在于，在日常生活任务和工作活动中天然存在的连续变异性背景下 [57]，对 RGB-D 传感器在上肢运动中的性能进行了深入表征。

在所分析的两种运动之间，跟踪工作空间分区的精度存在差异。对于点到点运动（point-to-point movement），左侧和中间分区存在显著的统计学差异，相较于右侧分区，Kinect 在这些区域的跟踪性能较差。跟踪性能的这种下降可能是由于即使目标位于受试者的左侧或正前方，运动也是由右臂执行的。因此，在这些条件下执行任务期间，受试者身体的一部分在较长时间内被手臂部分遮挡，导致 Kinect 设备不得不估算被覆盖点的位置，从而可能引入了额外的误差。这种实验条件是 RGB-D 设备的常见情况，应被视为采用此类传感器相较于基于标记的系统（marker-based systems）的主要局限性之一。根据具体应用场景，可以考虑使用特定算法（ad-hoc algorithms）来解决这一问题。

对于探索运动（exploration movement），我们发现左侧和中间分区的性能相当，而右侧分区的性能略差。尽管如此，右侧和中间分区的 ICC 表现更高，因此这两个分区提供了更一致和可靠的数据。这表明，中间分区可能是记录探索运动最合适的区域，以实现最高的可靠性和最低的误差。这些结果表明，在存在遮挡的情况下，RGB-D 设备的使用可能会更加关键。然而，我们仍然必须强调，即使在性能在统计学上不具有可重复性的情况下，总体差异也局限在小于 3<sup>◦</sup> 的范围内。

我们的研究结果对于科学界在评估多个应用领域中用于精确手势识别的更可靠配置也可能具有参考价值 [60]，特别是在辅助生活、临床和工作环境中上肢的应用场景中 [38]。

## 4.1.3. RGB-D 传感器：可靠性

重测实验能够表征 RGB-D 传感器的可靠性。我们发现，除手部旋前和旋后自由度外，点对点运动和探索运动中的重复性均相当高。我们的结果总体上与先前的研究结果一致。事实上，针对健康人群的初步研究表明，RGB-D 传感器具有与基于标记的系统相同的重复性 [16]。在一项分析全身运动（尤其是步行）的综合性康复导向研究中也发现了类似的结果。作者得出结论，重复性分析显示 Kinect V2 和 Vicon 的结果相当相似 [52]。本研究的结果与先前的研究相吻合，表明无论是对于健康受试者还是中风患者，Kinect 测量的许多参数在不同测试批次间均表现出可接受的可靠性和灵敏度 [61]。所有这些研究考虑的手势都与自由度强耦合：本研究所获得的稍显保守的结果可能与选择了要求更高的实验方案（手势的选择、变异性和配置）有关，该方案对传感器在通用运动中的追踪能力施加了更大压力。各区域之间的重复性未发现显著差异，尽管正如预期的那样，左侧区域的重复性稍差，这可能是由于遮挡所致。

## 4.1.4. 真实场景中的应用

如文献所述，RGB-D 设备能够记录上肢手势，并已应用于身体评估或训练等康复应用中，例如矢状面内的二维运动 [28]、远程康复场景 [62]，甚至是上肢的三维关节活动度 [63]。该技术在这一领域的应用对临床医生和患者而言都具有巨大的优势。例如，关节活动度提供了对关节位置足够精确的近似估计，为临床医生提供了反映患者康复状况的客观指标。从患者的角度来看，它可能会使基于家庭的康复治疗更加普及。

另一方面，工业应用主要面向工作场所的人体空间占用监测、安全性以及伤害预防。因此，工作台上的动作包括许多简单动作（如伸手取物），或者更复杂的动作（如操作桌面上的物品），以及使用简单且受控的动作与机器进行交互（例如传递手势）[48–50]。在更广泛的工业背景下，它已被用于研究工人在工作环境中的安全性。在此背景下，值得一提的是，在近期启动的欧洲地平线2020（European H2020）研究项目“Mindbot”[64]的框架中，RGB-D 技术将用于在工人与协作机器人交互过程中提供监测。此外，新发布的 Kinect Azure [65] 增强了 Kinect V2 的跟踪能力，并提供了一种更先进的工具，以促进本文提出的与生活和工作环境中的人群及工人相关的概念。

## 4.2. 局限性

本研究的首要局限性在于工作空间区域的数量有限。实际应用场景可能需要使用其他区域，正如在其他模拟装配场景的上肢研究所展示的那样[57]，但由于 Kinect 系统跟踪算法的局限性，这些区域在本研究中被排除了。此外，动作的数量也可以进一步扩展以模拟真实场景，包括工业中实用的其他上肢手势[47,66,67]，并纳入对腿部等其他身体部位的跟踪[52,63]。引入全身模型将有助于考察更多数量的自由度，并比较在完整三维运动中的性能。此外，在采用的传感器和使用的算法两方面，都可以考虑其他 RGB-D 解决方案。在本研究中，我们决定参考一种文档齐全、应用广泛的商业解决方案，尽管所提出概念的进一步发展应涉及使用其他传感器或骨骼分割算法。

事实上，本研究仅考虑了受试者对跟踪产生部分遮挡的情况；在真实场景中，环境可能包含本研究未考虑的其他限制和遮挡。最后，为了提供更为可靠的统计分析，未来可以招募规模更大的受试者群体。

## 5. 结论

在本文中，我们研究了商用 RGB 传感器与金标准光电标记点系统之间的对比。我们发现，搭载嵌入式人体跟踪算法的 RGB-D 传感器能够合理地逼近基于标记点系统计算出的自由度（DoF）。然而，我们发现 RGB-D 传感器的性能在某些自由度的检测上实用性欠佳，并且相较于探索性动作，“幅度更大”的动作（如点对点移动）能被更精确地跟踪。尽管存在一些局限性，我们仍可以得出结论：在无法使用标记点系统的康复和工业环境中，RGB-D 传感器是进行运动分析的潜在候选方案。

作者贡献：概念化，A.S. 与 R.M.M.；方法学，A.S. 与 R.M.M.；软件，R.M.M.；验证，A.S. 与 R.M.M.；形式分析，A.S. 与 R.M.M.；调查研究，R.M.M. 与 A.S.；资源，A.S.、R.M.M.、L.M.T.、M.S. 与 P.C.；数据管护，R.M.M. 与 A.S.；撰写——初稿准备，A.S.、R.M.M.、P.C.、M.S. 与 L.M.T.；撰写——审阅与编辑，R.M.M.、P.C.、M.S. 与 L.M.T.；可视化，R.M.M.；监督，P.C.；项目管理，A.S.、M.S. 与 L.M.T.；资金获取，A.S.、M.S. 与 L.M.T.。所有作者均已阅读并同意已发表的手稿版本。

资金支持：本研究未获得任何外部资金资助。

利益冲突：作者声明不存在利益冲突。

附录 A

$$
\text { Shoulder   elevation } = a c o s (\hat {a} \times (- \hat {t})),\tag{A1}
$$

$$
\text { Shoulder   rotation } = a c o s \left[ \widehat {a p} \times (- \hat {f}) \right] - 9 0,\tag{A2}
$$

$$
\text { Shoulder   internal / external   rotation } = a c o s (\widehat {a f \_ p} \times \widehat {n 2}) - 9 0,\tag{A3}
$$

$$
E l b o w e x t e n s i o n = a c o s (\widehat {a} \times \widehat {a f}),\tag{A4}
$$

$$
H a n d d e v i a t i o n = a c o s (u \widehat {h a n d} _ {-} \Phi \times \widehat {w 2}) - 9 0,\tag{A5}
$$

$$
H a n d f l e x i o n a n d e x t e n s i o n = a c o s (u \_ h a n d \_ \Pi \times \widehat {w 1}) - 9 0,\tag{A6}
$$

$$
H a n d p r o n a t i o n / s u p i n a t i o n = a c o s (u \_ h a n d 2 \_ \Omega \times \widehat {w 1}) - 9 0,\tag{A7}
$$

$$
\text { Scapular   elevation } = a c o s \bigl (\hat {s ^ {\prime}} \times (\hat {s}) \bigr),\tag{A8}
$$

$$
\text { Trunk   torsion } = a c o s \left[ \hat {s} \times (- \hat {f}) \right] - 9 0,\tag{A9}
$$

$$
\text { Trunk   antero / posterior   flexion } = a c o s \bigl (\widehat {t _ {\_} s a g} \times \hat {f} \bigr),\tag{A10}
$$

$$
T r u n k m e d i o / l a t e r a l f l e x i o n = a c o s \bigl [ \widehat {t \_ f r o n t} \times (- \hat {s}) \bigr ] - 9 0,\tag{A11}
$$

注：在公式 (A2) 中，向量 $\widehat { a p }$ 是上臂向量在与向量 $\hat { t }$ 正交的平面上的投影；在公式 (A3) 中，向量 $\widehat { a f _ { - } p }$ 是前臂向量在与向量 $\hat { a }$ 正交的平面上的投影；在 (A5) 中，向量 $u \widehat { \mathrm { ~ \textmu ~ } h a n d \mathrm { ~ \textmu ~ } } \Phi$ 是 $\widehat { h 1 }$ <sub>在与向量 w</sub>c<sub>1 正交的平面上的投影；</sub>在 (A6) 中，向量 u\_hand<sup>\</sup>\_Π 是 $\widehat { h 1 }$ 在与向量 $\widehat { w 2 }$ 正交的平面上的投影；在 (A7) 中，向量 u\_hand<sup>\</sup>2\_Ω 是 $\widehat { h 2 }$ 在与向量 ${ \widehat { a f } }$ 正交的平面上的投影；在 (A8) 中，sˆ 为初始位置的肩部向量，$\hat { s ^ { \prime } }$ 为连接躯干与右盂肱关节中心（glenohumeral joint center）的向量在与向量 ${ \hat { f } }$ 正交的平面上的投影；在 (A9) 中，向量 $- { \hat { f } }$ 表示第一帧中与受试者背部正交的方向；在 (A10) 中，$\widehat { \mathbf { t } \_ s a g }$ 是向量 $\hat { t }$ 在矢状面上的投影，正交于第一帧中的向量 sˆ；在 (A11) 中，t\_<sup>\</sup>front 是向量 ˆt 在额状面上的投影，正交于第一帧中的向量 $\hat { f }$。

## 参考文献

1. Colyer, S.; Evans, M.; Cosker, D.; Salo, A.I.T. A Review of the Evolution of Vision-Based Motion Analysis and the Integration of Advanced Computer Vision Methods Towards Developing a Markerless System. Sports Med. Open 2018, 4, 24. [CrossRef] [PubMed]

2. Zhou, H.; Hu, H. Human motion tracking for rehabilitation—A survey. Biomed. Signal Process. Control. 2008, 3, 1–18. [CrossRef]

3. Lu, T.W.; O’Connor, J. Bone position estimation from skin marker co-ordinates using global optimisation with joint constraints. J. Biomech. 1999, 32, 129–134. [CrossRef]

4. Roux, E.; Bouilland, S.; Godillon-Maquinghen, A.-P.; Bouttens, D. Evaluation of the GO method within the upper limb kinematics analysis. J. Biomech. 2002, 35, 1279–1283. [CrossRef]

5. Petuskey, K.; Bagley, A.; Abdala, E.; James, M.A.; Rab, G. Upper extremity kinematics during functional activities: Three-dimensional studies in a normal pediatric population. Gait Posture 2007, 25, 573–579. [CrossRef]

6. Pontonnier, C.; Dumont, G. Inverse dynamics method using optimization techniques for the estimation of muscles forces involved in the elbow motion. Int. J. Interact. Des. Manuf. (IJIDeM) 2009, 3, 227–236. [CrossRef]

7. Nussbaum, M.A.; Zhang, X. Heuristics for locating upper extremity joint centres from a reduced set of surface markers. Hum. Mov. Sci. 2000, 19, 797–816. [CrossRef]

8. Cappozzo, A.; Catani, F.; Della Croce, U.; Leardini, A. Position and orientation in space of bones during movement: Anatomical frame definition and determination. Clin. Biomech. 1995, 10, 171–178. [CrossRef]

9. Cappozzo, A.; Catani, F.; Leardini, A.; Benedetti, M.G.; Della Croce, U. Position and orientation in space of bones during movement: Experimental artefacts. Clin. Biomech. 1996, 11, 90–100. [CrossRef]

10. Carpinella, I.; Cattaneo, D.; Ferrarin, M. Quantitative assessment of upper limb motor function in Multiple Sclerosis using an instrumented Action Research Arm Test. J. Neuroeng. Rehabil. 2014, 11, 67. [CrossRef]

11. Carpinella, I.; Lencioni, T.; Bowman, T.; Bertoni, R.; Turolla, A.; Ferrarin, M.; Jonsdottir, J. Planar robotic training versus arm-specific physiotherapy: Efects on arm function and motor strategies in post-stroke subjects. Gait Posture 2019, 74, 7. [CrossRef]

12. Dorazio, T.; Marani, R.; Renò, V.; Cicirelli, G. Recent trends in gesture recognition: How depth data has improved classical approaches. Image Vis. Comput. 2016, 52, 56–72. [CrossRef]

13. Gesto-Diaz, M.; Tombari, F.; Rodríguez-Gonzálvez, P.; González-Aguilera, D. Analysis and Evaluation Between the First and the Second Generation of RGB-D Sensors. IEEE Sens. J. 2015, 15, 6507–6516. [CrossRef]

14. Pagliari, D.; Pinto, L. Calibration of Kinect for Xbox One and Comparison between the Two Generations of Microsoft Sensors. Sensors 2015, 15, 27569–27589. [CrossRef]

15. Kutlu, M.; Freeman, C.; Spraggs, M. Functional electrical stimulation for home-based upper-limb stroke rehabilitation. Curr. Dir. Biomed. Eng. 2017, 3, 25–29.

16. Bonnechère, B.; Jansen, B.; Salvia, P.; Bouzahouene, H.; Omelina, L’.; Moiseev, F.; Sholukha, V.; Cornelis, J.; Rooze, M.; Jan, S.V.S. Validity and reliability of the Kinect within functional assessment activities: Comparison with standard stereophotogrammetry. Gait Posture 2014, 39, 593–598. [CrossRef]

17. Cruz, L.M.V.; Lucio, D.; Velho, L. Kinect and RGBD Images: Challenges and Applications. In Proceedings of the 2012 25th SIBGRAPI Conference on Graphics, Patterns and Images Tutorials, Ouro Preto, Brasil, 22–25 August 2012; pp. 36–49.

18. Scano, A.; Caimmi, M.; Malosio, M.; Tosatti, L.M. Using Kinect for upper-limb functional evaluation in home rehabilitation: A comparison with a 3D stereoscopic passive marker system. In Proceedings of the 5th IEEE RAS/EMBS International Conference on Biomedical Robotics and Biomechatronics, Ouro Preto, Brasil, 12–15 August 2014; pp. 561–566.

19. Kurillo, G.; Chen, A.; Bajcsy, R.; Han, J.J. Evaluation of upper extremity reachable workspace using Kinect camera. Technol. Heal. Care 2013, 21, 641–656. [CrossRef]

20. Lee, S.H.; Yoon, C.; Chung, S.G.; Kim, H.C.; Kwak, Y.; Park, H.-W.; Kim, K. Measurement of Shoulder Range of Motion in Patients with Adhesive Capsulitis Using a Kinect. PLoS ONE 2015, 10, e0129398. [CrossRef]

21. Huber, M.E.; Seitz, A.; Leeser, M.; Sternad, D. Validity and reliability of Kinect skeleton for measuring shoulder joint angles: A feasibility study. Physiotherapy 2015, 101, 389–393. [CrossRef]

22. Clark, R.; Pua, Y.-H.; Oliveira, C.C.; Bower, K.J.; Thilarajah, S.; McGaw, R.; Hasanki, K.; Mentiplay, B.F. Reliability and concurrent validity of the Microsoft Xbox One Kinect for assessment of standing balance and postural control. Gait Posture 2015, 42, 210–213. [CrossRef]

23. Yahya, M.; Shah, J.; Kadir, K.; Warsi, A.; Khan, S.; Nasir, H. Accurate Shoulder Joint Angle Estimation Using Single RGB camera for Rehabilitation. In Proceedings of the 2019 IEEE International Instrumentation and Measurement Technology Conference (I2MTC), Auckland, New Zealand, 20–23 May 2019; pp. 1–6.

24. Scano, A.; Chiavenna, A.; Malosio, M.; Tosatti, L.M. Kinect V2 Performance Assessment in Daily-Life Gestures: Cohort Study on Healthy Subjects for a Reference Database for Automated Instrumental Evaluations on Neurological Patients. Appl. Bionics Biomech. 2017, 2017, 1–16. [CrossRef] [PubMed]

25. Chen, Y.C.; Lee, H.J.; Lin, K.H. Measurement of body joint angles for physical therapy based on mean shift tracking using two low cost Kinect images. In Proceedings of the 2015 37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Milan, Italy, 25–29 August 2015; Volume 2015, pp. 703–706.

26. Scano, A.; Caimmi, M.; Chiavenna, A.; Malosio, M.; Tosatti, L.M. A Kinect-Based Biomechanical Assessment of Neurological Patients’ Motor Performances for Domestic Rehabilitation. Adv. Med Technol. Clin. Pract. 2016, 252–279. [CrossRef]

27. Yang, Y.; Pu, F.; Li, Y.; Li, S.; Fan, Y.; Li, D. Reliability and Validity of Kinect RGB-D Sensor for Assessing Standing Balance. IEEE Sens. J. 2014, 14, 1633–1638. [CrossRef]

28. Scano, A.; Chiavenna, A.; Malosio, M.; Tosatti, L.M.; Molteni, F. Kinect V2 implementation and testing of the reaching performance scale for motor evaluation of patients with neurological impairment. Med. Eng. Phys. 2018, 56, 54–58. [CrossRef] [PubMed]

29. Kim, W.-S.; Cho, S.; Baek, D.; Bang, H.; Paik, N.-J. Upper Extremity Functional Evaluation by Fugl-Meyer Assessment Scoring Using Depth-Sensing Camera in Hemiplegic Stroke Patients. PLoS ONE 2016, 11, e0158640. [CrossRef] [PubMed]

30. Fernández-Baena, A.; Susin, A.; Lligadas, X. Biomechanical Validation of Upper-Body and Lower-Body Joint Movements of Kinect Motion Capture Data for Rehabilitation Treatments. In Proceedings of the 2012 Fourth International Conference on Intelligent Networking and Collaborative Systems, Bucharest, Romania, 19–21 September 2012; pp. 656–661.

31. Okuyama, K.; Kawakami, M.; Tsuchimoto, S.; Ogura, M.; Okada, K.; Mizuno, K.; Ushiba, J.; Liu, M. Depth Sensor–Based Assessment of Reachable Work Space for Visualizing and Quantifying Paretic Upper Extremity Motor Function in People with Stroke. Phys. Ther. 2020. [CrossRef]

32. Latorre, J.; Colomer, C.; Alcañiz, M.; Llorens, R. Gait analysis with the Kinect v2: Normative study with healthy individuals and comprehensive study of its sensitivity, validity, and reliability in individuals with stroke. J. Neuroeng. Rehabil. 2019, 16, 11–97. [CrossRef]

33. Scano, A.; Caimmi, M.; Chiavenna, A.; Malosio, M.; Tosatti, L.M. Kinect One-based biomechanical assessment of upper-limb performance compared to clinical scales in post-stroke patients. In Proceedings of the 2015 37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Milan, Italy, 25–29 August 2015; Volume 2015, pp. 5720–5723

34. Scano, A.; Molteni, F.; Tosatti, L.M. Low-Cost Tracking Systems Allow Fine Biomechanical Evaluation of Upper-Limb Daily-Life Gestures in Healthy People and Post-Stroke Patients. Sensors 2019, 19, 1224. [CrossRef]

35. Gu, Y.; Pandit, S.; Saraee, E.; Nordahl, T.; Ellis, T.; Betke, M. Home-Based Physical Therapy with an Interactive Computer Vision System. In Proceedings of the 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), Seoul, Korea, 27–28 October 2019; pp. 2619–2628.

36. Cameirão, M.; Smailagic, A.; Miao, G.; Siewiorek, D. Coaching or gaming? Implications of strategy choice for home based stroke rehabilitation. J. Neuroeng. Rehabil. 2016, 13, 18. [CrossRef]

37. Vieira, Á.; Gabriel, J.; Melo, C.; Machado, J. Kinect system in home-based cardiovascular rehabilitation. Proc. Inst. Mech. Eng. Part H J. Eng. Med. 2016, 231, 40–47. [CrossRef]

38. Mosca, N.; Renó, V.; Marani, R.; Nitti, M.; D’Orazio, T.; Stella, E. Human Walking Behavior detection with a RGB-D Sensors Network for Ambient Assisted Living Applications. In Proceedings of the AI\* AAL@ AI\* IA, Bari, Italy, 14–17 November 2017; pp. 17–29.

39. Terkaj, W.; Tolio, T. The Italian Flagship Project: Factories of the Future. In Factories of the Future; Springer Science and Business Media LLC: Berlin, Germany, 2019; pp. 3–35.

40. Santos, C.; Mehrsai, A.; Barros, A.C.; Araújo, M.; Ares, E. Towards Industry 4.0: An overview of European strategic roadmaps. Procedia Manuf. 2017, 13, 972–979. [CrossRef]

41. Geiselhart, F.; Otto, M.; Rukzio, E. On the Use of Multi-Depth-Camera Based Motion Tracking Systems in Production Planning Environments. Procedia CIRP 2016, 41, 759–764. [CrossRef]

42. Dufy, V.G. A methodology for assessing industrial workstations using optical motion capture integrated with digital human models. Occup. Ergon. 2007, 7, 11–25.

43. Ramey, A.; González-Pacheco, V.; Salichs, M.A. Integration of a low-cost RGB-D sensor in a social robot for gesture recognition. In Proceedings of the 6th international conference on Multimodal interfaces—ICMI ’04, Alicante, Spain, 14–18 November 2011; pp. 229–230.

44. Basso, F.; Munaro, M.; Michieletto, S.; Pagello, E.; Menegatti, E. 基于 RGB-D 数据的移动机器人快速鲁棒多人跟踪. In Advances in Intelligent Systems and Computing; Springer Science and Business Media LLC: Berlin, Germany, 2013; Volume 193, pp. 265–276.

45. Colombo, G.; Regazzoni, D.; Rizzi, C. 结合人体建模的无标记动作捕捉在虚拟人机工程学中的应用. Lect. Notes Comput. Sci. 2013, V, 314–323. [CrossRef]

46. Bachynskyi, M.; Oulasvirta, A.; Palmas, G.; Weinkauf, T. 基于动作捕捉的生物力学模拟对人机交互（HCI）研究是否有效？研究与启示. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, Toronto, ON, Canada, 26 April–1 May 2014; pp. 3215–3224.

47. Pellegrinelli, S.; Moro, F.L.; Pedrocchi, N.; Tosatti, L.M.; Tolio, T. 装配任务中人机协作工作空间共享的概率方法. CIRP Ann. 2016, 65, 57–60. [CrossRef]

48. Glasauer, S.; Huber, M.; Basili, P.; Knoll, A.; Brandt, T. 在时间和空间中的交互：人与人以及人机联合行动的研究. In Proceedings of the 19th International Symposium in Robot and Human Interactive Communication, Roman, Italy, 13–15 September 2010; pp. 252–257. [CrossRef]

49. Field, M.; Stirling, D.; Naghdy, F.; Pan, Z. 机器人领域动作捕捉技术综述. In Proceedings of the IEEE International Conference on Control and Automation, Christchurch, New Zealand, 9–11 December 2009; pp. 1697–1702.

50. Kim, S.; Nussbaum, M.A.; Esfahani, M.I.M.; Alemi, M.M.; Alabdulkarim, S.A.; Rashedi, E. 评估被动式上肢外骨骼背心对需要抬臂任务的影响：第一部分——对不适感、肩部肌肉活动和工作任务表现的“预期”影响. Appl. Ergon. 2018, 70, 315–322. [CrossRef]

51. Pfister, A.; West, A.M.; Bronner, S.; Noah, J.A. Microsoft Kinect 与 Vicon 3D 动作捕捉在步态分析中的能力对比. J. Med. Eng. Technol. 2014, 38, 274–280. [CrossRef]

52. Otte, K.; Kayser, B.; Mansow-Model, S.; Verrel, J.; Paul, F.; Brandt, A.U.; Schmitz-Hübsch, T. Kinect 2.0 在运动功能临床测量中的准确性与可靠性. PLoS ONE 2016, 11, e0166532. [CrossRef]

53. Cai, L.; Ma, Y.; Xiong, S.; Zhang, Y. 使用 Microsoft Kinect V2 传感器进行上肢功能评估的有效性与可靠性. Appl. Bionics Biomech. 2019, 2019, 7175240. [CrossRef]

54. Galna, B.; Barry, G.; Jackson, D.; Mhiripiri, D.; Olivier, P.; Rochester, L. Microsoft Kinect 传感器测量帕金森病患者运动的准确性. Gait Posture 2014, 39, 1062–1068. [CrossRef]

55. Sarbolandi, H.; Lefloch, D.; Kolb, A. Kinect 距离传感：结构光与飞行时间（Time-of-Flight）Kinect 的对比. Comput. Vis. Image Underst. 2015, 139, 1–20. [CrossRef]

56. Mahmoudzadeh, A.; Golroo, A.; Jahanshahi, M.R.; Yeganeh, S.F. Estimating Pavement Roughness by Fusing Color and Depth Data Obtained from an Inexpensive RGB-D Sensor. Sensors 2019, 19, 1655. [CrossRef]

57. Scano, A.; Dardari, L.; Molteni, F.; Giberti, H.; Tosatti, L.M.; D’Avella, A. A Comprehensive Spatial Mapping of Muscle Synergies in Highly Variable Upper-Limb Movements of Healthy Subjects. Front. Physiol. 2019, 10, 1231. [CrossRef] [PubMed]

58. Moore, K.L.; Dalley, A.F.; Agur, A.M. Clinically Oriented Anatomy, 7th ed.; Taylor, C., Ed.; Lippincott Williams & Wilkins: Philadelphia, PA, USA, 2013.

59. Sinclair, J.; Taylor, P.; Hobbs, S.J. Digital Filtering of Three-Dimensional Lower Extremity Kinematics: An Assessment. J. Hum. Kinet. 2013, 39, 25–36. [CrossRef] [PubMed]

60. Cicirelli, G.; Attolico, C.; Guaragnella, C.; D’Orazio, T. A Kinect-Based Gesture Recognition Approach for a Natural Human Robot Interface. Int. J. Adv. Robot. Syst. 2015, 12, 22. [CrossRef]

61. Mobini, A.; Behzadipour, S.; Foumani, M.S. Test-retest reliability of Kinect’s measurements for the evaluation of upper body recovery of stroke patients. Biomed. Eng. Online 2015, 14, 75. [CrossRef]

62. Capecci, M.; Ceravolo, M.G.; Ferracuti, F.; Iarlori, S.; Longhi, S.; Romeo, L.; Russi, S.N.; Verdini, F. Accuracy evaluation of the Kinect v2 sensor during dynamic movements in a rehabilitation scenario. In Proceedings of the 2016 38th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Orlando, FL, USA, 16–20 August 2016; pp. 5409–5412.

63. Bonnechère, B.; Sholukha, V.; Omelina, L.; Jansen, B.; Jan, S.V.S. Three-dimensional functional evaluation of the shoulder complex using the Kinect™ sensor. In Proceedings of the 4th Workshop on ICTs for improving Patients Rehabilitation Research Techniques, Lisbon, Portugal, 13–14 October 2016; pp. 5–8.

64. Cordis European Commission. 访问网址：https://cordis.europa.eu/project/id/847926（访问日期：2020年2月15日）。

65. Microsoft Azure. 访问网址：https://azure.microsoft.com/it-it/services/kinect-dk/（访问日期：2020年2月2日）。

66. Jiang, S.; Liu, P.; Fu, D.; Xue, Y.; Luo, W.; Wang, M. A low-cost rapid upper limb assessment method in manual assembly line based on somatosensory interaction technology. In Proceedings of the 2017 5TH International Conference on Computer-Aided Design, Manufacturing, Modeling And Simulation (CDMMS 2017), Busan, Korea, 22–23 April 2017. [CrossRef]

67. Miguez, S.A.; Hallbeck, M.S.; Vink, P. Work Movements: Balance Between Freedom and Guidance on an Assembly Task in a Furniture Manufacturer. In Advances in Intelligent Systems and Computing; Springer Science and Business Media LLC: Berlin, Germany, 2016; Volume 491, pp. 503–511.

![](images/bfe5e60189847f8045b115308f900e29e4b8752ee7b0ed95f0ee118f2055415e.jpg)

© 2020 版权归作者所有。被许可人 MDPI（瑞士巴塞尔）。本文是一篇遵循知识共享署名（Creative Commons Attribution，CC BY）许可协议 (http://creativecommons.org/licenses/by/4.0/) 条款和条件的开放获取（Open Access）文章。