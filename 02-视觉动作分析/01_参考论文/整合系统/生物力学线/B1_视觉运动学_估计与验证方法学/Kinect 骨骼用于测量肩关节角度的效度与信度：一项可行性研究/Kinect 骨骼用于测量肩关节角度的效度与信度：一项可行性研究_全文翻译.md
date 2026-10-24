最终编辑版本发表于：Physiotherapy. 2015 December ; 101(4): 389–393. doi:10.1016/j.physio.2015.02.002.

# Kinect 骨骼用于测量肩关节角度的效度与信度：一项可行性研究

M. Huber<sup>a</sup>, A.L. Seitz<sup>b</sup>, M. Leeser<sup>c</sup>, and D. Sternad<sup>c,d,e</sup>

<sup>a</sup>Department of Bioengineering, Northeastern University, Boston, MA, USA

<sup>b</sup>Department of Physical Therapy and Human Movement Sciences, Northwestern University, Chicago, IL, USA

<sup>c</sup>Department of Electrical and Computer Engineering, Northeastern University, Boston, MA, USA

<sup>d</sup>Department of Biology, Northeastern University, Boston, MA, USA

<sup>e</sup>Department of Physics, Northeastern University, Boston, MA, USA

## 摘要

目的——测试用于虚拟康复的 Microsoft Kinect<sup>™</sup> 测量肩关节角度的信度与效度。

设计——重测信度与同时效度，可行性研究。

场所——动作分析实验室。

参与者——由 10 名健康成年人组成的便利样本。

方法——在四种静态姿势下进行两次试验，同时使用以下设备评估肩关节角度：(1) Kinect；(2) 三维（3D）动作分析系统；(3) 临床关节角度计。所有姿势均通过 Kinect 从正面视角进行采集。肩关节屈曲的两种姿势还通过 Kinect 从矢状面视角进行了采集。

主要结局指标——在每种姿势下，通过组内相关系数（ICC）、测量标准误和最小可检测变化量确定 Kinect 测量肩关节角度的绝对与相对重测信度。计算 Kinect 与测量肩关节角度的标准方法之间的 95% 一致性界限（LOA），以确定同时效度。

结果——虽然 Kinect 从正面视角测量肩关节角度表现出高信度（ICC 0.76 至 0.98），但在两种视角的全部姿势下，Kinect 与两种测量标准之间的 95% LOA 均大于 ±5°。

结论——在将 Kinect 用于虚拟康复应用中的动作测量之前，必须了解其在测量特定关节运动时的精密度和准确度局限性。

## 关键词

Microsoft Kinect；动作分析；骨骼追踪；虚拟现实；测量准确度；肩部康复

## 引言

将虚拟现实技术用于康复，即虚拟康复（virtual rehabilitation, VR），与传统疗法相比具有多项优势，包括提高了对运动表现进行定量测量的能力、提供实时运动表现反馈以及增强患者的康复积极性。通过利用最新的商业游戏技术 [1–4]，VR 系统的开发成本越来越低，这使其在居家康复治疗中尤为适用。

目前使用居家虚拟现实（VR）的大部分研究主要针对神经系统疾病患者 [1–5]。然而，这些系统同样具有改善肌肉骨骼疾病患者物理治疗的潜力。借助居家 VR 系统，临床医生可以确保患者正确进行锻炼，并在有助于关节正常愈合的特定术后时间范围内达到预期的功能目标。然而，尽管具有这些优势，VR 在术后关节治疗中的应用目前仍非常有限。

Kinect<sup>™</sup>（Microsoft Corp., Redmond, WA, USA）作为较受欢迎的游戏传感器之一，是专为术后关节康复设计的 VR 系统的理想传感器。然而，与所有游戏技术一样，Kinect 最初并不是为了临床使用而开发的。因此，在临床应用之前，必须针对相关动作对 Kinect 测量的准确性进行全面评估。令人担忧的是，有使用 Kinect 进行 VR 研究的文献声称该传感器的有效性和可靠性此前已被证实，并引用了评估 Kinect 不同测量指标有效性和可靠性的研究 [5]。例如，引用深度图像的准确性来确保来自 Microsoft Kinect for Windows<sup>™</sup> 软件开发套件（SDK）的骨骼数据的准确性是具有误导性的 [6]。

以开发用于肩部术后治疗的 VR 系统为最终目标，本研究旨在评估来自 Kinect for Windows SDK 的骨骼数据在精确测量肩部角度方面的可靠性和有效性。先前的研究发现，来自 Kinect SDK 的骨骼数据可用于精确测量肩关节活动度（ROM）[7]；然而，其用于测量精确肩部角度的准确性尚未得到研究。Fernández-Baena 等人 [8] 先前的一项研究检验了 Kinect 用于测量肩关节 ROM 的准确性，观察到肩部角度轨迹的平均误差在 8 到 $1 4 ^ { \circ }$ 之间。类似地，Chang 等人 [9] 观察到使用 Kinect 追踪肩部位置时存在较大误差。然而，这些研究并未对有效性和可靠性进行正式分析。此外，这些研究使用了来自 OpenNI（Primesense）的骨骼数据，这些数据不同于来自 Kinect

SDK 的骨骼数据。尽管如此，Fernández-Baena 等人 [8] 和 Chang 等人 [9] 的发现表明，有必要对使用 Kinect 骨骼数据测量精确肩部角度的准确性进行评估。

本可行性研究在参与者保持一系列静态姿势时测量了各种肩部角度。这些姿势由术后肩部康复中常用的肩部构型组成，其中包括一种肩部在 Kinect 视野中被遮挡的姿势。研究对比了来自三个数据采集系统——Kinect、三维（3D）运动分析系统（金标准；trakSTAR, Ascension Technology Corp., Shelburne, VT, USA）和测角计（临床标准）的肩部角度测量值。

## 方法

## 参与者

由 10 名无已知肩部病变的无症状成年人组成的便利样本{6 名女性和 4 名男性，平均年龄 22.1 [标准差（SD）0.9] 岁}参与了本研究。所有参与者在研究前均签署了知情同意书。研究方案经 xxxxxxxxxxx 机构审查委员会批准。

## 实验流程

在 Kinect 处于正视图位时，每位参与者以随机顺序保持以下静态姿势，每个姿势重复两次：屈曲至 $9 0 ^ { \circ }$、最大屈曲、外展至 $9 0 ^ { \circ }$ 以及外展 $0 ^ { \circ }$ 时的最大外旋（Fig. A，参见在线补充材料）。屈曲至 $9 0 ^ { \circ }$ 和最大屈曲姿势在矢状视图下用 Kinect 额外测量了两次重复。增加矢状视图姿势是因为预实验发现，在正视图位进行屈曲至 $9 0 ^ { \circ }$ 姿势期间，肩关节在 Kinect 视野中被遮挡。因此，在矢状视图下重复了这两个肩部屈曲姿势，以确定从该视角测量肩部屈曲是否可靠且有效。半数参与者使用利手进行肩部运动，另一半使用非利手；这也经过了随机分配。

$9 0 ^ { \circ }$ 姿势（屈曲至 $9 0 ^ { \circ }$ 和外展至 $9 0 ^ { \circ } )$ 使用测角计设定。对于最大姿势（最大屈曲以及外展 $0 ^ { \circ }$ 时的最大外旋），指导参与者旋转至其最大极限能力。姿势设定好后，同时使用 Kinect、3D 运动分析系统和盲法测角计记录测量数据。每次重复均重新设定姿势。在各次重复之间，最大值可能存在差异。然而，该流程与当前的临床实践一致。通过招募健康、无痛的受试者，降低了测试之间每个姿势关节活动度（ROM）发生变异的可能性。

## 数据采集与处理

Kinect——通过 Kinect for Windows SDK 为每个姿势捕获的骨骼数据由 20 个关节点的三维（3D）位置组成。肩关节和肘关节相对于躯干的位置用于测量肩关节屈曲和外展的角度（以度为单位），而肘部和手部相对于躯干的位置则用于测量外旋角度。来自 Kinect for Windows SDK 的骨骼数据使用 MATLAB（Mathworks, Natick, MA, USA）进行访问和分析。

测角仪（Goniometer）——对标准 12 英寸测角仪进行了改装，以使检查者对测量读数设盲。肩关节运动的测角测量采用标准化方法进行 [10]。当检查者将测角仪与肩部运动对齐后，由第二名检查者读取并记录测量值（以度为单位）。肩部的测角仪测量已表现出极佳的信度 [11,12]。

三维动作分析（3D motion analysis）——采用采样率为 240 Hz 的 Trakstar 基于电磁的动作分析系统配合 Motion Monitor 软件（Innovative Sports Training, Inc., Chicago, IL, USA）采集肱骨和躯干的三维运动学数据。电磁接收器用胶带固定在胸部 T3 椎骨棘突上方以及手臂远端肱骨后侧。根据国际生物力学学会（International Society of Biomechanics）的建议 [13]，利用各环节上的数字化解剖标志点建立每个环节的局部坐标轴系。使用肱骨旋转的欧拉角（Euler angle）序列来描述肱骨相对于胸部的运动。外展和屈曲方向的肩部运动使用抬高角描述，外旋则使用长轴旋转描述。为了与临床测角仪测量结果进行直接对比，屈曲、外展和外旋的所有运动均定义为正值。据报道，该系统的均方根精度小于 1° [14]。

## 统计分析

采用组内相关系数（Intraclass correlation coefficient, ICC）模型 3,2 [ICC(3,2)] 来确定 Kinect 测量肩关节角度的相对重测信度 [15]。计算了代表各个视角下每种姿势两次试验一致性的六个 ICC(3,2) 值。ICC(3,2) 值定义为：低于 0.20 为“差”（poor），0.21 至 0.40 之间为“一般”（fair），0.41 至 0.60 之间为“中等”（moderate），0.61 至 0.80 之间为“良好”（good），0.81 至 1.0 之间为“极好”（very good）[16]。计算测量标准误（SEM）和最小可检测变化量（MDC）以确定绝对信度。SEM 定义为标准差（SD）乘以 1 减去 ICC 的差值的平方根 [17]，MDC 通过将 SEM 乘以 2 的平方根计算得出 [18]。

计算了每种姿势下 Kinect 与两种肩关节测量标准之间的 95% 一致性界限（limits of agreement, LOA）以确定效度 [18]。为了获得每种姿势下的 95% LOA，计算了每种方法测得的两个肩关节角度测量值的平均值。接着，计算了（1）Kinect 与量角器之间，以及（2）Kinect 与 3D 运动分析测量值之间差异的均值和标准差（SD）。95% LOA 定义为平均差异 $\pm 1.96$ 差异的标准差，使得 95% 的差异落在此界限范围内。若 95% LOA 大于 $\pm 5^\circ$，则认为测量系统之间的差异具有临床显著性 [19]。所有统计分析均使用 IBM SPSS Statistics 软件（IBM Corp., Armonk, NY, USA）进行。

## 结果

## 重测信度

Kinect 测量肩关节角度的重测信度结果（包括 ICC、SEM 和 MDC 值）如 Table 1 所示。从正视图来看，较高的 $\mathrm { I C C } ( 3 , 2 )$ 值表明 Kinect 在所有四种姿势下测量肩关节角度均具有良好至极好的相对信度。从正视图观察，除了前屈至 $9 0 ^ { \circ }$ 姿势（此时肩关节被 Kinect 摄像头遮挡）外，所有姿势均表现出较小的 SEM 和 MDC 值，表明具有良好的绝对信度。从矢状视图来看，Kinect 在前屈至 $9 0 ^ { \circ }$ 姿势下具有极好的相对信度和绝对信度，但在最大前屈姿势下仅具有尚可的相对信度和较差的绝对信度。

## 同时效度

Kinect 与量角器之间以及 Kinect 与 3D 运动分析系统之间的 95% LOA 如 Table 2 所示。外展至 $9 0 ^ { \circ }$ 是与量角器或 3D 运动追踪系统相比，Kinect 肩关节角度测量值均具有合理准确度的唯一姿势。然而与其他姿势一样，系统间差异的 95% LOA 超出了 $\pm 5 ^ { \circ } .$，这被定义为具有临床显著性。

## 讨论

利用来自 Kinect for Windows SDK 的骨骼数据，发现 Kinect 在大多数姿势下测量肩关节角度具有高度的信度。在肩关节外展至 $9 0 ^ { \circ }$ 姿势下测量肩关节角度的准确度最高。这与以往的研究结果一致，即 Kinect 在测量肩关节外展过程中的关节活动度（ROM）方面是准确的 [7]。然而，在所有姿势下，Kinect 与两种测量标准之间的测量差异均具有临床显著性。

从正面视角来看，Kinect 在横断面（外展 $0 ^ { \circ }$ 位最大外旋）和矢状面（前屈至 $9 0 ^ { \circ }$ 及最大前屈）的肩关节角度测量结果较差。Bonnechère et al. [7] 报道了使用 Kinect 测量屈肘和屈膝（两者均为矢状面内的运动）期间关节活动度（ROM）时类似的不良结果。矢状面测量的不准确，部分原因可以解释为估算被遮挡关节位置时引入的误差；这是基于视觉的运动追踪系统的典型问题。这一点在正面视角下前屈至 $9 0 ^ { \circ }$ 姿态中最为明显，此时肩关节被手臂遮挡，导致 Kinect 测量的绝对信度和效度较差。

为了尝试解决这一问题，研究还从矢状面视角测量了前屈至 $9 0 ^ { \circ }$ 和最大前屈姿态，但这并未被证明是一个可行的解决方案。在外展 $0 ^ { \circ }$ 位最大外旋和最大前屈姿态中，关节遮挡并不是问题，但 Kinect 测量的肩关节角度仍然不准确。这一主要局限性对简单肩部运动以及复杂的全身运动的测量都构成了难题。

## 结论

尽管 Kinect for Windows SDK 的骨骼数据在商业游戏用途中可能足够准确，但本研究表明，当需要精确的肩关节角度测量时，使用这些数据来测量肩部运动存在显著的问题。本研究仅检验了有限数量的肩部姿态。然而，肩关节角度测量中的巨大差异不容忽视。事实上，这些结果表明有必要进一步评估 Kinect 在测量更大范围肩关节角度以及受损运动中的准确性。

用于术后肩部治疗的虚拟现实（VR）仅仅是需要精确肩关节角度测量的一个应用范例。事实上，大多数肌肉骨骼和神经康复方案都需要精确的肩关节角度测量 [4,8,9,20]。在将 Kinect 用于测量 VR 中的运动之前，必须充分了解其在测量特定关节运动时在精密度和准确度方面的局限性。

## 补充材料

补充材料请参阅 PubMed Central 上的网页版本。

## 致谢

资助：本研究由 The Mathworks 向 M.E. Huber 提供的种子基金以及 Northeastern University Provost Tier 1 基金资助。

## 参考文献

1. Deutsch JE, Borbely M, Filler J, Huhn K, Guarrera-Bowlby P. Use of a low-cost, commercially available gaming console (Wii) for rehabilitation of an adolescent with cerebral palsy. Phys Ther. 2008; 88:1196–207. [PubMed: 18689607]

2. Morrow, K.; Docan, C.; Burdea, G.; Merians, A. Low-cost virtual rehabilitation of the hand for patients post-stroke. IEEE 2006 International Workshop on Virtual Rehabilitation; New York, NY, USA. p. 6-10.

3. Huber M, Rabin B, Docan C, Burdea GC, AbdelBaky M, Golomb MR. Feasibility of modified remotely monitored in-home gaming technology for improving hand function in adolescents with cerebral palsy. IEEE Trans Inf Technol Biomed. 2010; 14:526–34. [PubMed: 20071262]

4. Golomb MR, McDonald BC, Warden SJ, Yonkman J, Saykin AJ, Shirley B, et al. In-home virtual reality videogame telerehabilitation in adolescents with hemiplegic cerebral palsy. Arch Phys Med Rehabil. 2010; 91:1–8. e1. [PubMed: 20103390]

5. Pompeu JE, Arduini LA, Botelho AR, Fonseca MBF, Pompeu SMAA, Torriani-Pasin C, et al. Feasibility, safety and outcomes of playing Kinect Adventures!<sup>™</sup> for people with Parkinson’s disease: a pilot study. Physiotherapy. 2014; 100:162–8. [PubMed: 24703891]

6. Shotton, J.; Fitzgibbon, A.; Cook, M.; Sharp, T.; Finocchio, M.; Moore, R., et al. Real-time human pose recognition in parts from single depth images. 2011 IEEE Conference on Computer Vision and Pattern Recognition; Colorado Springs, CO, USA. 20–25 June 2011; p. 1297-304.

7. Bonnechère B, Jansen B, Salvia P, Bouzahouene H, Omelina L, Moiseev F, et al. Validity and reliability of the Kinect within functional assessment activities: comparison with standard stereophotogrammetry. Gait Posture. 2014; 39:593–8. [PubMed: 24269523]

8. Fernández-Baena, A.; Susín, A.; Lligadas, X. Biomechanical validation of upper-body and lowerbody joint movements of Kinect motion capture data for rehabilitation treatments. 4th IEEE International Conference on Intelligent Networking and Collaborative Systems; 19–21 September 2012; Bucharest, Romania. p. 656-61.

9. Chang CY, Lange B, Zhang M, Koenig S, Requejo P, Somboon N, et al. Towards pervasive physical rehabilitation using Microsoft Kinect. Pervasive Health. 2012:159–62.

10. Jain NB, Wilcox RB, Katz JN, Higgins LD. Clinical examination of the rotator cuff. PM R. 2013; 5:45–56. [PubMed: 23332909]

11. Riddle DL, Rothstein JM, Lamb RL. Goniometric reliability in a clinical setting. Shoulder measurements. Phys Ther. 1987; 67:668–73. [PubMed: 3575423]

12. Hayes K, Walton JR, Szomor ZR, Murrell GA. Reliability of five methods for assessing shoulder range of motion. Aust J Physiother. 2001; 47:289–94. [PubMed: 11722295]

13. Wu G, van der Helm FCT, Veeger HEJ, Makhsous M, Van Roy P, Anglin C, et al. ISB recommendation on definitions of joint coordinate systems of various joints for the reporting of human joint motion. Part II: Shoulder, elbow, wrist and hand. J Biomech. 2005; 38:981–92. [PubMed: 15844264]

14. Ludewig PM, Phadke V, Braman JP, Hassett DR, Cieminski CJ, LaPrade RF. Motion of the shoulder complex during multiplanar humeral elevation. J Bone Joint Surg Am. 2009; 91:378–89. [PubMed: 19181982]

15. Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater reliability. Psychol Bull. 1979; 86:420–8. [PubMed: 18839484]

16. Landis JR, Koch GG. The measurement of observer agreement for categorical data. Biometrics. 1977; 33:159–174. [PubMed: 843571]

17. Stratford PW, Binkley JM, Riddle DL. Health status measures: strategies and analytic methods for assessing change scores. Phys Ther. 1996; 76:1109–23. [PubMed: 8863764]

18. Stratford PW, Binkley J, Solomon P, Finch E, Gill C, Moreland J. Defining the minimum level of detectable change for the Roland–Morris questionnaire. Phys Ther. 1996; 76:359–65. discussion 366–8. [PubMed: 8606899]

19. Martin Bland J, Altman D. Statistical methods for assessing agreement between two methods of clinical measurement. Lancet. 1986; 327:307–10.

20. Lange B, Chang C-Y, Suma E, Newman B, Rizzo AS, Bolas M. Development and evaluation of low cost game-based balance rehabilitation tool using the Microsoft Kinect sensor. Annu Int Conf IEEE Eng Med Biol Soc. 2011; 2011:1831–4.

ICC，组内相关系数（intraclass correlation coefficient）；SEM，测量标准误（standard error of the measure）；MDC，最小可检测变化量（minimal detectable change）。  
<sub>a</sub>b<sup>le</sup>  
Kinect 用于测量肩关节角度的重测信度结果。

<table><tr><td>Kinect 视角</td><td>姿势</td><td>ICC</td><td>均值</td><td>SEM</td><td>MDC</td></tr><tr><td rowspan="4">正面</td><td>外展至 90°</td><td>0.76</td><td>90.1°</td><td>2.5°</td><td>3.5°</td></tr><tr><td>0° 外展位最大外旋</td><td>0.98</td><td>65.8°</td><td>3.7°</td><td>5.2°</td></tr><tr><td>屈曲至 90°</td><td>0.85</td><td>73.7°</td><td>12.2°</td><td>17.3°</td></tr><tr><td>最大屈曲</td><td>0.95</td><td>162.2°</td><td>4.0°</td><td>5.6°</td></tr><tr><td rowspan="2">矢状面</td><td>屈曲至 90°</td><td>0.84</td><td>86.7°</td><td>4.4°</td><td>6.2°</td></tr><tr><td>最大屈曲</td><td>0.37</td><td>161.5°</td><td>24.2°</td><td>34.1°</td></tr></table>

<sub>a</sub>b<sup>le</sup>

使用 Kinect、量角器和三维（3D）磁跟踪系统获得的角度。肩关节测量值之间的 Bland-Altman 分析及一致性界值（计算为差值的 ±1.96 个标准差）

<table><tr><td rowspan="2">Kinect 视角</td><td rowspan="2">姿态</td><td colspan="2">Kinect 对比 测角仪</td><td colspan="2">Kinect 对比 3D 磁力追踪器</td></tr><tr><td>平均偏差</td><td>95% 一致性界限</td><td>平均偏差</td><td>95% 一致性界限</td></tr><tr><td>正面</td><td>外展至 90°</td><td>-1.5°</td><td>-7.0 至 4.1°</td><td>3.0°</td><td>-10.6 至 16.7°</td></tr><tr><td></td><td>0° 外展下外旋至最大角度</td><td>1.3°</td><td>-14.3 至 16.9°</td><td>16.0°</td><td>-36.4 至 68.4°</td></tr><tr><td></td><td>屈曲至 90°</td><td>-16.6°</td><td>-52.1 至 18.9°</td><td>-6.1°</td><td>-44.6 至 32.4°</td></tr><tr><td></td><td>屈曲至最大角度</td><td>-15.9°</td><td>-36.5 至 4.7°</td><td>10.6°</td><td>-15.4 至 36.7°</td></tr><tr><td>矢状面</td><td>屈曲至 90°</td><td>-6.9°</td><td>-17.7 至 3.8°</td><td>6.1°</td><td>-8.7 至 20.9°</td></tr><tr><td></td><td>屈曲至最大角度</td><td>-15.0°</td><td>-44.5 至 14.5°</td><td>10.6°</td><td>-25.0 至 46.3°</td></tr></table>