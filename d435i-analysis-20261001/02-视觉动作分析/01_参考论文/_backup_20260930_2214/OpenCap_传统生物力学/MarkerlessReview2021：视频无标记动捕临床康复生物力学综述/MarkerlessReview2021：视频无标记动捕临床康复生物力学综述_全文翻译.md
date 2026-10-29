# Video-Based Markerless Motion Capture for Clinical and Rehabilitation Biomechanics: A PRISMA-ScR Scoping Review of Validated Architectures, Clinical Readiness, and Emerging Methods

Florian Delaplace florian.delaplace@unice.fr

Laboratoire Motricité Humaine Éducation Sport Santé（人体运动、教育、体育与健康实验室）Elodie Piche Laboratoire Motricité Humaine Éducation Sport Santé

Frédéric Chorin

Centre Hospitalier Universitaire de Nice（尼斯大学医学中心）

Raphael Zory Laboratoire Motricité Humaine Éducation Sport Santé

## Research Article（研究论文）

Keywords: markerless motion capture, pose estimation, biomechanics, gait analysis, deep learning, scoping review, clinical translation, OpenSim

关键词：无标记点动作捕捉、姿态估计、生物力学、步态分析、深度学习、范围综述、临床转化、OpenSim

Posted Date: July 31st, 2026

DOI: https://doi.org/10.21203/rs.3.rs-10431606/v1

License:   This work is licensed under a Creative Commons Attribution 4.0 International License. Read Full License

Additional Declarations: No competing interests reported.

附加声明：无利益冲突报告。

## Abstract（摘要）

## Background.（背景）

Video-based markerless motion capture promises movement analysis without the cost, space and skin-marker constraints of optoelectronic systems, with particular potential for clinical and rehabilitation settings. Whether validated pipelines yet deliver clinically acceptable biomechanics, and how they relate to the underlying computer-vision research, remains unclear.

基于视频的无标记点动作捕捉有望摆脱光电系统在成本、空间以及皮肤标记点方面的限制，在临床和康复场景中具有尤为广阔的应用前景。然而，经过验证的处理流程是否已能提供临床上可接受的生物力学结果，以及这些流程与底层计算机视觉研究之间的关联如何，目前尚不明确。

## Methods.（方法）

We conducted a scoping review following the PRISMA extension for Scoping Reviews, with a registered protocol and searches of PubMed, Scopus and IEEE Xplore (January 2015 to February 2026; the computer-vision scan was updated to July 2026). A dual-tier design paired a primary corpus of validated biomechanical studies with a complementary, curated and deliberately non-exhaustive corpus of emerging computer-vision work, used qualitatively. We charted study characteristics, pipeline architecture, validation methods and joint-angle accuracy.

我们按照 PRISMA 范围综述扩展版开展了一项范围综述，预先注册了研究方案，并在 PubMed、Scopus 和 IEEE Xplore 中进行了检索（检索时间范围为 2015 年 1 月至 2026 年 2 月；计算机视觉部分的检索更新至 2026 年 7 月）。研究采用双层设计，将经过验证的生物力学研究作为主要语料库，并与一个经过筛选且明确非穷举的新兴计算机视觉研究语料库相配合，后者仅作定性分析使用。我们对研究特征、处理流程架构、验证方法以及关节角度精度进行了系统梳理。

## Results.（结果）

We included 117 studies, most published from 2024 onward and conducted on healthy adults walking in a laboratory. Pipelines formed five architectural families across monocular and multi-camera modalities; most reported raw joint angles without biomechanical refinement. Sagittal lower-limb agreement clustered around 5 to 6°, generally short of clinical acceptability, while out-of-plane kinematics, kinetics, and pathological or older populations were rarely validated. Emerging computer-vision building blocks (foundation-model mesh recovery, differentiable inverse kinematics, video-based kinetics) were almost absent from validated studies.

共纳入 117 项研究，其中大多数发表于 2024 年及之后，研究对象为在实验室内行走的健康成年人。处理流程在单目和多目两种模态下形成了五个架构家族；大多数研究仅报告未经生物力学精修的原始关节角度。矢状面下肢的一致性集中在 5°至 6°左右，总体上尚未达到临床可接受标准，而离面（out-of-plane）运动学、动力学以及病理人群或老年人群则鲜有验证。计算机视觉领域的新兴构建模块（如基础模型的网格恢复、可微分逆运动学、基于视频的动力学估计）几乎未出现在经过验证的研究中。

## Conclusions.（结论）

基于视频的无标记采集尚不能与有标记系统在临床关节运动学中互换使用，而在康复最需要的领域——老年人群与病理性人群、离面（out-of-plane）运动学与动力学——其验证仍然严重不足。将这一证据缺口与新兴计算机视觉进展相对应，我们提出的是假设生成性的设计指导原则（而非经过验证的方法），旨在引导下一代流水线朝着可及、临床有意义的运动分析方向发展。

## 1. 背景

三维运动采集三十多年来一直是生物力学研究与临床步态分析的基石。光电子学、有标记运动采集（marker-based motion capture, MBMC）系统（Vicon、Qualisys 和 OptiTrack）仍然是关节运动学测量的实际金标准，在受控实验室条件下具有记录在案的亚毫米级空间精度与亚度级关节角度精度 [1]。然而，其在预算充足的学术与临床研究实验室之外的推广应用受到三重障碍的制约：资本成本（每套系统数万至数十万欧元）、场地（具有受控光照的专用采集空间）以及运行开销（标记放置、软组织伪影、后期处理人力）。即便是一套设置良好的光电子系统，其精度也更多受限于软组织伪影——皮肤标记相对于骨骼的位移——而非硬件本身 [2–3]；因此，去除皮肤标记正是基于视频方法的重要吸引力所在。

与此同时，过去十年间，从单一 RGB 视频进行人体姿态估计的计算机视觉（computer vision, CV）技术快速增长，这便是所谓的"无标记运动采集"（markerless motion capture, MMC）范式 [4]。从基础的二维姿态估计器（OpenPose [5]、BlazePose [6]、HRNet [7]）到多视角三角测量流水线（Pose2Sim [8]、OpenCap [9]），再到近期通过深度神经网络进行单目三维恢复的方法（MotionBERT [10]、BioPose [11]、OpenCap Monocular [12]），该领域已产出一系列广泛的算法方法。潜在收益包括设备成本降低一至三个数量级、部署于临床或家庭环境，以及兼容研究与常规临床使用的工作流程。

然而，这一技术前景尚未转化为连贯的临床现实。当前生物力学领域关于无标记运动捕捉（MMC）的文献呈现出显著的异质性：不同的算法体系并存，但对于某一特定临床或研究问题应优先采用哪种架构尚无共识；不同研究报告的精度指标跨度达一个数量级；相较于矢状面，临床相关的关节平面（额状面、水平面）被系统性低估；而在病理人群中的验证仍然稀缺。这些不足具有直接的临床意义。最少被报告的额状面与水平面偏差——例如膝外翻、髋关节内旋、骨盆倾斜、画圈步态和剪刀步态——恰恰是定义常见神经性步态障碍的特征性体征 [13]。因此，一个仅能以五至六度精度解析矢状面的流程，可能会遗漏康复临床医生需要追踪的这些体征。此外，大量前沿计算机视觉研究几乎仅出现在预印本平台（arXiv）和计算机视觉会议论文集中。将这些新发表的算法模块整合到流程中、参照金标准进行验证以及发表结果都需要时间，因此预计最新的计算机视觉进展在进入经过验证的生物医学文献之前将存在不可忽视的延迟。本综述正是对这一预期加以考察。

该证据基础的广度、异质性及快速演进使得采用荟萃分析进行综合变得不切实际，甚至可能产生误导：各研究在研究设计、报告指标、金标准设备和目标人群方面的差异性阻碍了推论性荟萃分析（加权效应量估计）的开展；因此，我们以描述性方式——使用中位数、范围和比例——汇总证据，而非合并效应量。依据PRISMA-ScR（系统综述与荟萃分析优先报告条目——范围综述扩展版）和乔安娜·布里格斯研究所（Joanna Briggs Institute, JBI）方法学框架开展的范围综述，因此是描绘该证据图景更为合适的工具。

先前的无标记生物力学综述大多罗列了各类架构并报告了精度 [4, 14]；目前仍存在两方面不足：一是缺乏对经过临床验证的领域如何与日新月异的计算机视觉前沿相联系的阐述，而后者将塑造该领域的下一代发展；二是缺乏超越纯描述、走向具体且有据可依的设计方向的内容。我们通过刻意连接两套鲜有交集的文献——临床生物力学与计算机视觉——来同时回应这两点。在常规的范围综述之外，我们提出一种双层设计。MACRO 层级构成对生物医学数据库（PubMed、Scopus、IEEE Xplore）的正式 PRISMA 筛选，产出已对 MMC 流水线相对于外部生物力学金标准进行实证验证的研究语料。MICRO 层级构成一个互补的、经策展的语料，来源有二：在系统检索中因仅涉及单一流水线组件而非完整验证流水线而被筛除的已发表记录，以及对近期 arXiv 预印本和计算机视觉会议论文集的有针对性扫描，用于揭示支撑下一代 MMC 流水线的技术构件。MICRO 在设计上即为非穷尽性，定性使用，以识别和刻画新兴的技术族系而非估计体量，但明确弥补了预印本阶段的 CV 研究与其最终在生物医学数据库中被索引之间的结构性时滞。

本范围综述围绕两个研究问题展开。

RQ1（MACRO）：当前已发表的、基于视频的无标记运动捕捉流水线用于生物力学分析时，其架构、验证结果和临床适用性如何？

RQ2（MICRO）：计算机视觉文献（2024–2026）中涌现的技术构件有哪些——这些构件经由一次刻意非系统性的前瞻扫描加以刻画——可能塑造下一代 MMC 流水线，但尚未过渡到同行评审的生物医学出版物中？

在综合两者的基础上，我们进一步提炼出一套面向临床可部署无标记流水线的循证设计指南，涵盖两种部署模式（单目、单摄像头，可在临床或野外部署；以及多摄像头、实验室级），将 MACRO 验证研究的实证经验与 MICRO 的技术现状相结合。第三条线索具有刻意的前瞻性：它以视角与研究议程的形式呈现，提供与本综述所记录的空白相锚定的、产生假设的方向，而非一种已验证的方法。

## 2. 方法

## 2.1 研究设计与方案

本范围综述依据 PRISMA-ScR 指南 [15] 以及 JBI 范围综述方法学框架 [16] 开展。综述方案已在开放科学框架（Open Science Framework，OSF，doi: 10.17605/OSF.IO/TBYPC）前瞻性注册；完整的 PRISMA-ScR 检查清单见 Additional file 1。相较于系统综述，本研究选择范围综述设计基于两方面原因：第一，所纳入文献在研究设计、验证方法学及报告指标方面存在显著的异质性，无法进行有意义的 Meta 分析合并；第二，我们明确的研究目标是描绘并刻画一个快速演进的证据全景图，包括其新兴的技术前沿，而非估计某个汇总效应量。

## 2.2 信息来源与检索策略

我们在三个生物医学与工程数据库（PubMed、Scopus 与 IEEE Xplore）中检索了 2015 年 1 月至 2026 年 2 月期间发表的研究，最后一次数据库检索执行于 2026 年 2 月 4 日。检索策略结合了受控词汇表（PubMed 中的 MeSH 主题词）与涵盖无标记运动捕捉、基于视频的姿态估计及生物力学验证的自由词。完整检索式见 Additional file 2（Appendix A）。

为补充正式的生物医学检索，我们在系统综述之外配对了一次算法视野扫描：构建了第二个、刻意保持非穷尽性的语料库（即 MICRO 语料库），用于捕捉新兴的计算机视觉工作。其来源包括两个方面：系统检索中仅涉及单一流程环节的筛选记录，以及对近期计算机视觉文献（即 arXiv，主要为 cs.CV 分类）的有针对性扫描，并纳入主要视觉与学习会议（CVPR、ICCV、ECCV、WACV、NeurIPS）的会议论文，时间范围限定为 2024–2026 年，以纳入尚未被生物医学数据库收录的方法。其构建过程详见第 2.6 节。MICRO 语料库仅作定性使用，旨在揭示技术构件与新兴方法族群，而非估计其流行程度。计算机视觉扫描独立于系统检索运行，并更新至 2026 年 7 月，晚于 2026 年 2 月的生物医学检索截止日期，因此有少量近期预印本晚于 MACRO 检索。

## 2.3 纳入与排除标准

纳入标准按照人群-概念-情境（Population-Concept-Context，PCC）框架加以阐述，并对 MACRO 语料库明确列出了具体的纳入与排除标准。

## 纳入标准

研究纳入标准如下：(i) 评估基于视频输入（单目或多摄像机）的无标记运动捕捉流程；(ii) 报告了与外部生物力学参考标准的定量验证（Vicon、Qualisys、OptiTrack、BTS/MIQUS、Motion Analysis、Nokov 或类似的光电子系统；用于动力学的测力板；或在少数情况下使用仪器化 IMU 系统）；(iii) 报告了经验关节角、时空或动力学精度指标（RMSE、MAE、相关系数或 ICC）；(iv) 聚焦于生物力学运动分析，而非一般的姿态检测或活动识别。

## 排除标准

应用了八项预先指定的排除标准（A-H）：

1. **A. 黑箱系统**：商业或专有流程，其姿态估计模型未被识别或公开，例如 Theia3D、Captury 和 KinaTrax。排除这些系统的原因是它们的姿态估计模型无法被表征或复现，尽管 Theia3D 尤其是在临床部署最多的无标记系统之一；此处有十一项研究被排除。

2. **B. 无现代计算机视觉**：仅使用手动数字化或经典（非深度学习）计算机视觉方法。

3. **C. 深度传感器通道**：以深度或 RGB-D 模态（例如 Kinect）作为主要三维机制的研究，由于本综述关注 RGB 视频流程，因此被排除，尽管此类设备在康复诊所中仍然很常见。

4. **D. 无原始数据的综述**。

5. **E. 非无标记**：使用物理身体标记作为流程输入。

6. **F. 仅手动金标准**：仅与 Kinovea、量角器、视频手动标注或类似手动参考进行验证。

7. **G. 无生物力学输出**：仅报告中间指标（平均关节点位置误差 MPJPE，该指标评分关键点位置而非关节角度），未报告关节角度、时空参数或动力学数据。

8. **H. 无外部金标准**：仅报告内部重测信度，未与独立参考系统进行比较。

对于 MICRO 语料库，无论验证状态如何，研究均被纳入，前提是其涉及至少一个流程处理组件（2D 姿态估计、3D 提升、三角剖分、网格恢复、逆运动学、步态事件检测、时间建模、物理信息方法或信号处理）。

## 2.4 研究筛选流程

标题与摘要筛选以及全文评估均使用 Covidence 系统综述软件由两名评审员独立进行，依据纳入排除标准进行评估。任何阶段的冲突均由第三位资深评审员解决。全文排除被编码为广泛的类别（错误的技术、验证、结果、文档类型、群体或未获取报告），而通过全文审查但在数据提取验证时未通过预指定 A-H 标准的研究则在该后续阶段被排除（图 1）。

## 2.5 数据提取

我们在 Microsoft Excel 中开发了一个结构化提取模板，包含 7 个工作表和 90 多个数据字段，涵盖六个领域：

**一般特征**（研究 ID、完整标题、第一作者、年份、论文类型、样本量、受试者人口统计学信息、目标人群、运动任务、服装）。

**视频输入规格**（相机数量、位置布置、相机类型、分辨率、采集频率、标定程序、同步方式、环境）。

**计算机视觉处理**（2D 姿态算法、骨干网络架构、训练方案、训练数据集、3D 重建方法、流水线名称、遮挡处理、滤波处理、标志点增补、实时能力）。

**生物力学处理**（生物力学软件、运动学方法、骨骼模型、缩放程序、动力学/力的计算）。

**全局与时空验证**（金标准系统、已验证的结果类别、报告指标、全局运动学误差和相关性、时空误差、作者结论）。

**关节层面验证**（每个关节 × 每个平面的 RMSE 和相关性：骨盆、髋、膝、踝 × 矢状面、额状面、水平面）。

第七个工作表将每项研究的架构分类综合为分析中使用的 5 类分类法。

## LLM 辅助数据提取

数据提取由作者完成，大语言模型作为辅助性的错误浮现步骤，而非数据来源的主体。第一作者定义了结构化模板（在 117 项研究中约 90 个字段）。为了从异构来源（自由文本、表格、图表、扫描 PDF）加速模板的首次填充，每项研究由一个大语言模型（Claude Opus 4.7）处理。随后由来自不同供应商的第二模型（Google Gemini）独立重新提取表格并交叉核对，作者根据源 PDF 对两者之间的每一处分歧进行裁定。在裁定之前，两个模型在约 10,500 个字段中的 156 个字段上存在分歧（117 项研究共约 90 个字段），原始一致性约为 98.5%。由于模板混合了数值型、类别型和自由文本字段，未计算单一的经机会校正的一致性系数。在所有验证轮次中，应用了 385 处单元格级别的更正（多于 156 处模型分歧，因为某些错误为两个模型共有，仅在与源 PDF 核对时才显现出来），这些更正主要为数值转录错误以及逐关节或逐平面的错位，另有较小比例的架构或金标准的类别误分类。还识别并更正了两处模型引用了所引用的先前研究而非本文本身数值的情况。作为最终的内部一致性检查，第一作者使用计算机随机数生成器随机抽取了 20% 的纳入研究，直接从源文档重新提取；该样本中的所有字段与表格完全匹配（数值和类别标签均一致）。最终这一轮并非独立于主要提取；独立性依赖于上述的跨供应商双模型裁定。因此，这些数值是机器辅助但经过人工核实的，作者对所提取的数据集负责。

## 2.6 数据综合

综合分析按三个层级递进展开。

## 描述性分析

MACRO 语料库首先从时间趋势、人群特征、样本量、运动任务和采集环境等方面进行刻画。

架构映射。117 项 MACRO 研究被归入一个五家族的架构分类法（二维单目、三维多视图三角测量、单目二维到三维提升、人体网格恢复/SMPL、端到端直接），并附有五个非互斥的叠加变体（+ IK/MSK、+标记点增强、+时序深度学习、+物理、仅几何）。针对每个家族，我们报告技术组件（骨干网络、姿态估计器、生物力学软件）的分布以及按关节和解剖平面的验证结果。关节角度误差根据 McGinley 的误差区间（2° 或以下为可接受，2° 至 5° 为合理，高于 5° 可能产生误导）进行汇总，作为临床相关基准。

技术解构。MICRO 语料库分两步构建。第一步，在数据库筛选阶段识别出的、仅描述单一流水线组件（而非针对金标准进行验证的完整流水线）的记录（排除标准 F 和 G）被保留并归入 17 个流水线步骤模块（三维提升、网格恢复、二维姿态、物理信息方法、三角化、步态事件、深度估计、逆运动学求解器、滤波、运动预测、标记点增广、跟踪、肌肉骨骼建模、时间平滑、基础模型、动力学与标定；图 6，第 3.4 节）。第二步，我们在每个模块内进一步检索，针对近期的 arXiv 预印本与计算机视觉会议论文（2024–2026 年），以纳入生物医学筛选中未出现的最新前沿方法；随后追溯每种新兴方法可升级到的 MACRO 架构家族。MICRO 语料库由此经策展而构建，并刻意保持非穷举性。arXiv（cs.CV）与会议扫描截至 2026 年 7 月；完整记录列表、每个条目的来源以及各模块的计数（附加文件 6）见附加文件 5。该 MICRO 扫描不属于 PRISMA-ScR 范围，PRISMA-ScR 仅规范系统的 MACRO 语料库。完整的构建协议与各模块可复现的检索式见附加文件 7。

最后，MACRO-MICRO 桥接分析针对五个架构家族中的每一个，明确了 MICRO 语料库中对应的技术构建模块，以及经验证的生物医学文献与新兴计算机视觉前沿技术之间的主要分歧点（第 3.5 节）。

## 3. 结果

## 3.1 纳入语料库的描述

从三个数据库中识别出 9,346 条记录，去除 2,294 条重复后，对 7,052 条标题与摘要进行了筛选；其中 6,659 条被排除，393 篇全文报告进行了评估，其中 237 篇在全文阶段被排除，另有 39 篇在依据标准 A–H 进行数据提取核验时被排除，最终纳入 117 项研究（图 1）。纳入的 117 项研究跨越 2015 年 1 月至 2026 年 2 月；各项研究的特征见附加文件 3。发表量近期加速增长：约六成自 2024 年起发表（图 2a）。

证据基础较为狭窄。受试者绝大多数为健康成人。临床或病理性人群（脑卒中、帕金森病、脑瘫、脊髓损伤）仅出现在少数研究中（约 19%），儿童或老年队列则更为罕见（图 2b）。步态是主要的运动任务，出现于超过一半的语料库中；跑步、坐起、跳跃落地与下蹲占据了其余大部分（图 2c）。样本量较小，中位数为 17 名受试者（图 2d）。人口学偏倚与较小的样本量限制了各项验证研究的普适性。

验证几乎总是使用基于光电标记的系统作为参考（约 83% 的研究；图 2e）。其余由仪器化步道、测力板和公共数据集覆盖。数据采集仍以实验室为基础：117 项研究中有 100 项在实验室进行，16 项在临床运动分析室进行，家庭或户外采集几乎缺失（1 项研究，无研究）。因此，缺乏在光学实验室以外进行分析的验证证据。

技术配置证实了研究原型阶段（表 1）。45% 的研究使用单摄像头采集，29% 使用双摄像头，使用机器视觉摄像头（40%）或消费级智能手机（26%）。近一半的研究（53 项）未进行标定，其余使用棋盘格标定。二维姿态估计约一半研究依赖 OpenPose 系列，其中也包括 OpenCap 部署——其二维后端（默认为 OpenPose，或可选 HRNet）通常未报告。其次是 MediaPipe 和 BlazePose [6]。三维恢复在多视图三角测量和纯二维分析之间几乎均匀分布。单目提升、网格拟合和端到端回归各占一小部分。63% 的研究使用自定义代码（Python 或 Matlab）进行生物力学处理，略多于四分之一的研究（33 项）采用基于 OpenSim 的建模，其中 13 项使用了即用型 OpenCap 智能手机流程，20 项直接使用 OpenSim。117 项研究中有 8 项（7%）报告了实时运行。离线后处理是常态。

表 1  
纳入文献集的技术配置（n = 117）。

<table><tr><td>变量 / 类别</td><td>n (%)</td></tr>
<tr><td>摄像头数量</td><td></td></tr>
<tr><td>1</td><td>53 (45%)</td></tr>
<tr><td>2</td><td>34 (29%)</td></tr>
<tr><td>3–4</td><td>13 (11%)</td></tr>
<tr><td>5–9</td><td>11 (9%)</td></tr>
<tr><td>≥10</td><td>4 (3%)</td></tr>
<tr><td>基准 / 多样</td><td>2 (2%)</td></tr>
<tr><td>摄像头类型</td><td></td></tr>
<tr><td>工业 / 机器视觉</td><td>47 (40%)</td></tr>
<tr><td>智能手机</td><td>30 (26%)</td></tr>
<tr><td>网络摄像头</td><td>14 (12%)</td></tr>
<tr><td>运动相机 / 单反 / RGB</td><td>16 (14%)</td></tr>
<tr><td>其他 / 未报告</td><td>5 (4%)</td></tr>
<tr><td>平板电脑</td><td>5 (4%)</td></tr>
<tr><td>标定方式</td><td></td></tr>
<tr><td>无（单目）</td><td>53 (45%)</td></tr>
<tr><td>棋盘格</td><td>38 (32%)</td></tr>
<tr><td>其他 / 未说明</td><td>20 (17%)</td></tr>
<tr><td>束束调整 / 点阵</td><td>3 (3%)</td></tr>
<tr><td>标定杆</td><td>3 (3%)</td></tr>
<tr><td>二维姿态估计器</td><td></td></tr>
<tr><td>OpenPose / OpenCap 系列</td><td>57 (49%)</td></tr>
<tr><td>MediaPipe / BlazePose</td><td>17 (15%)</td></tr>
<tr><td>其他已命名的深度学习检测器（AlphaPose、YOLO、PifPaf、HRNet、MMPose、Detectron2、ViTPose…）</td><td>17 (15%)</td></tr>
<tr><td>自定义 / 未指定的深度学习方法</td><td>7 (6%)</td></tr>
<tr><td>DeepLabCut</td><td>6 (5%)</td></tr></table>

<table><tr><td rowspan=1 colspan=1>变量 / 类别                                                                                                              n (%)</td></tr><tr><td rowspan=1 colspan=1>无 / 端到端                                                                                                                5 (4%)</td></tr><tr><td rowspan=1 colspan=1>专有 / SDK（软件开发工具包）                                                                                              5 (4%)</td></tr><tr><td rowspan=1 colspan=1>多估计器基准                                                                                                        3 (3%)</td></tr><tr><td rowspan=1 colspan=1>架构（3D 重建）</td></tr><tr><td rowspan=1 colspan=1>2D 单目（无显式 3D）                                                                                                    47 (40%)</td></tr><tr><td rowspan=1 colspan=1>多视角三角化                                                                                                        44 (38%)</td></tr><tr><td rowspan=1 colspan=1>单目提升（2D-&gt;3D）                                                                                                     19 (16%)</td></tr><tr><td rowspan=1 colspan=1>人体网格 / SMPL                                                                                                       4 (3%)</td></tr><tr><td rowspan=1 colspan=1>端到端 / 直接                                                                                                                3 (3%)</td></tr><tr><td rowspan=1 colspan=1>生物力学软件</td></tr><tr><td rowspan=1 colspan=1>自定义代码（Python/Matlab）                                                                                             74 (63%)</td></tr><tr><td rowspan=1 colspan=1>基于 OpenSim 的建模                                                                                                 33 (28%)</td></tr><tr><td rowspan=1 colspan=1>通过 OpenCap（智能手机应用）                                                                                          13 (11%)</td></tr><tr><td rowspan=1 colspan=1>直接使用 OpenSim                                                                                                            20 (17%)</td></tr><tr><td rowspan=1 colspan=1>无（2D 角度）                                                                                                                8 (7%)</td></tr><tr><td rowspan=1 colspan=1>其他 MSK（Visual3D、CusToM、AnyBody）                                                                  2 (2%)</td></tr><tr><td rowspan=1 colspan=1>实时能力</td></tr><tr><td rowspan=1 colspan=1>否（后处理）                                                                                                                19 (2%)</td></tr><tr><td rowspan=1 colspan=1>是 / 近实时                                                                                                                  8 (7%)</td></tr><tr><td rowspan=1 colspan=1>未指定                                                                                                                            2 (2%)</td></tr></table>

## 3.2 核心流水线架构

每个流水线属于五种互斥的架构家族之一，按部署模态进行分组（图3）。全文保持以下三个术语的区分：模态（modality，从单个视角即单目重建，或从多个同步视角重建）、家族（family，五种架构之一）以及叠加层（overlay，任何家族均可添加的可选精化模块）。单目模态包含四个家族，其主要区别在于如何达到三维：二维单目直接从二维关键点计算关节角度（47项研究，40%）[17–63]；单目提升从这些检测结果回归三维姿态（19项，16%）[64–81, 86]；人体网格恢复拟合参数化的全身网格而非稀疏骨架（4项，3%）[11, 82–84]，其代表性工作 BioPose [11] 是通过 IEEE Xplore 检索到的、经同行评审的 WACV 2025 论文；端到端直接则直接从图像回归关节角度（3项，3%）[85, 87–88]。多相机模态以多视角三角化为主导，通过来自多个已标定相机的射线相交来重建三维关节位置（44项，38%）[8–9, 89–130]。两种模态在数量上（73项对比44项研究）有所不同，更重要的是在能力上也存在差异（第3.3节）。模态与实际相机数量并不等同。许多单视角流水线仅将额外相机用于对比或标定，且有一项端到端研究使用了多视角输入，因此家族按架构而非相机数量进行划分（表1）。

二维单目与三角化已较为成熟且仍在持续增长（自2024年起分别为27项和30项研究）。单目提升出现于2021年之后，目前仍在扩展。网格恢复是最新的参与者：所有基于 SMPL 的研究均出现在2024年或之后。端到端回归曾短暂出现，但未得以延续。三角化研究的近期增长得益于 Pose2Sim 和 OpenCap 等即用型多视角工具的发布。

一种架构是一个可共享的流水线骨架，可向其添加可选的细化模块，而非一种固定方法。共编码了五种互不排斥的叠加方案：基于肌骨骼建模的逆运动学、虚拟标记增补、深度学习时序平滑、基于物理的优化以及动力学输出（地面反作用力或关节力矩）。大多数流水线几乎不加细化。略低于三分之二的研究（117篇中73篇）不使用任何叠加方案，仅报告原始几何角度，仅有28篇研究组合了两种或以上的叠加方案。细化集中在多视角类别中。基于肌骨骼建模的逆运动学是最常见的叠加方案（36项研究，31%），在44篇三角测量研究中有27篇采用了该方法，这些研究同时也几乎囊括了所有虚拟标记增补流水线（OpenCap模式）。相比之下，47篇2D单目研究中有46篇仍仅使用几何方法，而基于OpenSim的应用（包括OpenCap）几乎仅限于三角测量研究。因此，严肃的肌骨骼建模位于多相机一侧，而具有最大部署潜力的单目一侧则大多止步于原始的二维角度。

## 3.3 计量学验证

## 3.3.1 参考标准

约83%的研究采用光电标记系统作为验证参照，最常用的包括Vicon、Qualisys、OptiTrack、BTS或Nokov。少数研究使用了仪器化步道（GAITRite、Zeno/PKMAS）、测力台、公开的运动捕捉数据集，或在极少数情况下使用平面X射线摄影。这些参考系统本身并非无误差：它们存在软组织伪影，即皮肤标记相对于骨骼的运动 [2]，这为任何无标记系统相对于它们所能达到的一致性设定了上限。尽管如此，光电捕捉仍然是生物力学领域的实际参考标准。一项近期的无标记研究甚至发现，其自身的不一致性至少与该伪影相当 [117]，因此这两种范式在准确性上并非能够截然区分。

## 3.3.2 关节角度精度与临床阈值

矢状面下肢的一致性在髋、膝、踝和骨盆四个关节上聚集在约 5 至 $6 ^ { \circ }$ 的中位绝对误差附近，四者彼此相差约 $1 ^ { \circ }$ 以内（表 2）。研究间离散度较大（四分位距约 3 至 8°）。我们将一致性对照 McGinley 等人针对三维步态运动学所提出的误差带进行基准化处理：误差 $2 ^ { \circ }$ 及以下被普遍视为可接受，2 至 $5 ^ { \circ }$ 视为合理但需在解读时加以考量，超过 $5 ^ { \circ }$ 则可能误导临床解读 [131]。这些是基于标记点的可靠性指标。由于参考系统本身具有可重复性，而在矢状面之外其重复性即可接近 $5 ^ { \circ }$，处于该水平的一致性部分受限于参考系统自身的噪声，因此更宜解读为尚未达到基于标记点的可靠性水平，而非临床不可用。可接受误差并非单一常数，因为它取决于临床问题以及所测量偏差的大小，故我们报告的是一个区间而非单一截止值。在矢状面下肢估计中，仅约 6% 落入 $2 ^ { \circ }$ 可接受区间，约 37% 落入 2 至 $5 ^ { \circ }$ 合理区间；而单独考察时，约一半矢状面下肢估计仅在均方根误差下即超过 $5 ^ { \circ }$（中位数 5.0°），在平均绝对误差下亦约一半超过 $5 ^ { \circ }$（中位数 5.0°）。将所有报告的指标合并统计后，结果相近（约 57%）。这一水平正是 McGinley 等人认为误差可能误导临床解读的阈值（图 5）。关键在于这一模式本身，而非具体比率。即便在其表现最佳的方面，大多数已发表流程仍超过 $5 ^ { \circ }$ 的警戒水平。在康复学层面，5 至 6° 的误差与临床医生在恢复或治疗滴定过程中试图追踪的运动学变化处于同一量级——例如脑性瘫痪患儿蹲伏步态的逐步改善，或脑卒中后僵硬膝与划圈步态的消退——因此处于该误差水平的仪器可能掩盖真实的临床变化。

## 3.3.3 按架构划分的精度

我们按架构家族绘制了矢状面误差（图 4）。两个最具代表性的家族——2D 单目（2D Monocular，n = 47）和多视图三角测量（Multi-view Triangulation，n = 44）——表现相近，下肢误差中位数约为 $4$ 至 $6^\circ$。单目提升（Monocular Lifting，n = 19）表现较差，下肢误差中位数约为 $6$ 至 $7^\circ$，且研究间离散度最大。这与深度模糊性一致：单个相机无法完全解析朝向或远离相机的运动，因此从一个视图回归三维姿态本质上比从多个已标定视图进行三角测量更难。人体网格恢复（Body Mesh Recovery）和端到端直接（End-to-End Direct）各自仅有四项或更少的研究，代表性不足，无法进行比较，仅作描述性展示。没有家族在中位数上达到 $2^\circ$ 的可接受区间，因此临床精度差距是整个领域的共性，而非某一弱架构的特性。

## 3.3.4 平面覆盖与矢状面报告偏倚

主要局限一部分在于被报告的内容，另一部分在于能够被测量的内容本身。离平面运动学几乎只来自多视图家族，原因有二：2D 单目家族根本无法恢复离平面旋转，因为其不产生三维估计；而单目三维家族和三角测量虽能恢复离平面旋转，但大多未予报告。在架构 × 关节 × 平面映射图（附加文件 4）中，额状面与水平面的数据单元主要由三角测量填充（例如髋部额状面 n = 13，髋部水平面 n = 10），而每个单目家族仅填充单一的矢状面列。报告的不平衡非常显著。对于膝关节，65 项研究报告了矢状面误差，而额状面仅 5 项、水平面仅 3 项；髋关节（57 / 17 / 12）和踝关节（49 / 4 / 1）也呈现相同模式。在额状面和水平面存在数据之处，其报告数量更少且数值更大（例如踝关节额状面约 $9^\circ$）。与若干临床问题最相关的平面——包括膝关节内外翻、髋关节旋转和足部推进方向——恰恰是验证最少的。

## 3.3.5 时空参数与动力学验证

时空参数（步行速度、步长、步时）在约三分之一（34%）的研究中进行了验证，其一致性高于关节角度，这反映了它们对关键点噪声的敏感性较低。相对于测力板进行的动力学验证（针对地面反作用力或关节力矩）仅在九项研究中被报告。该领域主要验证的是下肢矢状面运动学。离平面运动学和动力学在很大程度上仍未得到充分检验，第 3.4 节和第 3.5 节将进一步展开讨论。

Table 2  
按解剖平面的关节角度验证（与光电参考系统的绝对一致性）。

| 关节 | 平面 | 研究数量 | 中位数误差（°） | 四分位距（°） | $\leq 5^\circ \left( \% \right)$ |
| --- | --- | --- | --- | --- | --- |
| 骨盆 | 矢状面 | 5 | 5.2 | 3.4-6.1 | 40 |
| 骨盆 | 冠状面 | 3 | 4.0 | 2.0-4.9 | — |
| 骨盆 | 横断面 | 3 | 3.1 | 2.0-3.3 | — |
| 髋 | 矢状面 | 57 | 5.5 | 3.6-7.1 | 47 |
| 髋 | 冠状面 | 17 | 4.3 | 3.0-5.2 | 76 |
| 髋 | 横断面 | 12 | 5.7 | 4.4-7.0 | 33 |
| 膝 | 矢状面 | 65 | 5.6 | 3.7-7.0 | 40 |
| 膝 | 冠状面 | 5 | 4.8 | 3.1-12.8 | 60 |
| 膝 | 横断面 | 3 | 5.9 | 2.4-9.5 | — |
| 踝 | 矢状面 | 49 | 5.9 | 3.6-8.4 | 43 |
| 踝 | 冠状面 | 4 | 9.0 | 5.9-10.3 | — |
| 踝 | 横断面 | 1 | 11.1 | n/a | — — |

注：矢状面以外的报告明显欠缺（例如膝关节：矢状面 65 项 vs 冠状面 5 项 vs 横断面 3 项）。最后一列报告误差在 5° 以内的比例（McGinley 可接受至合理范围；见 3.3.2 节）。基于不足五项研究的单元仅为指示性。由于报告存在广泛异质性，所报告的一致性指标同时汇总了均方根误差（root-mean-square error）与平均绝对误差（mean absolute error）；各指标的细分见 Additional file 3。仅采用这两个主要指标并不改变上述不足：在仅采用均方根误差（中位数 5.0°）与仅采用平均绝对误差时，约一半的矢状面下肢估计值仍超过 5°。

## 3.4 技术拆解（MICRO）

为识别下一代流水线背后的计算机视觉构建模块，我们组装了互补的 MICRO 语料库，包含 572 条 arXiv 与会议记录，并将其归为 17 个流水线步骤模块。由于该语料库是策展性而非系统性的（见方法部分），图 6 以各模块的相对活跃度（低、中或高）而非原始计数来表征每个模块，且我们不计算 MACRO 与 MICRO 的比值。

一个值得注意的定性信号（需审慎对待）。我们筛选的记录集中在 2024 与 2025 年，且基础模型（foundation-model）板块几乎全部来自 2024 年之后。由于文献集是通过反向引用策划与播种的，这种近期性部分反映了我们的采样而非领域全貌，因此我们仅将其视为新计算机视觉方法与其在经过验证的生物医学工作中被采纳之间存在时滞的一个粗略、定性提示，而非一个可测量的间隔。能够更明确地证明这一差距的，是这些新兴板块在经过验证的文献集中几乎完全缺席（3.5 节）。

若将现有方法视为映射到流水线各阶段的工具箱（图 7；完整的方法清单、每个方法所针对的阶段、其贡献及其成熟度见 Additional file 5），那么几乎每个板块都存在一个候选的升级方案 [132–150]。在输入阶段，单目度量深度与三维几何模型，以及无需标定物的多相机自标定正在趋于成熟 [174–176]；通过相机运动、重力或学习得到的视场线索进行世界坐标系锚定，有助于在单相机条件下以度量尺度对被试进行定位——这一问题目前仅得到部分解决 [133, 177]。在二维姿态方面，实时与 Transformer 主干网络 [151–153] 正在取代 MACRO 文献集中占主导地位的 OpenPose 一代方法。三维重建方面则多种方法并存：单目升维 [10, 154]；基于 SMPL 系列的基础模型网格重建 [155–157, 172]，它将流水线重构为围绕稠密参数化网格展开，并可访问解剖学标志点 [158]；全身网格到关节角的逆运动学 [171]；多人与多视角网格拟合 [169, 173]；以及通向 OpenSim 的经典三角化方法 [8]。在生物力学衔接方面，虚拟标志点增强 [159] 与可微分或神经逆运动学 [160–162]，以及尺度缩放与逆动力学 [168] 正逐渐兴起。动力学板块在姿态下游估计地面反作用力与关节力矩，所采用的是物理信息或学习模型 [163]，其目标是 MACRO 文献中很少验证的输出。在重建之上，多模态生物力学基础模型 [146] 与逐自由度不确定性量化 [164] 构成的分析层开始出现，以满足临床部署对可靠性的要求。已有若干近乎完整的流水线将上述板块组合起来，而非引入新的板块：BioPose [11]、OpenCap Monocular [12] 与便携式生物力学实验室（Portable Biomechanics Laboratory）[170]，其中后者报告的与参考值的一致性被其作者描述为从手持视频即可达到临床可接受水平。

## 3.5 MACRO–MICRO 桥梁

将五个经过验证的家族映射到新兴模块上（图 8），可以逐家族地描绘转化前沿（translational frontier）。仅有多视角三角测量（multi-view triangulation）在主动引入新兴模块：通过 OpenCap 谱系，它采用了虚拟标记增强（virtual-marker augmentation）以及结合肌肉骨骼建模的逆运动学（inverse kinematics, IK），另有少量研究开始触及视频动力学（video kinetics）。四个单目（monocular）家族除 2D 姿态估计和直接几何角度计算外，几乎没有引入任何其他模块。在全部五个家族中，网格重建（mesh recovery）、可微分逆运动学、视频动力学和以人为中心的基座模型（human-centric foundation models）基本缺席。因此，该图的右侧近乎空白：那些最有可能改变临床能力的模块，恰恰是采纳最少的。

四个模块界定了这一差距。第一，基座模型网格重建（foundation-model mesh recovery）能够为单目流程提供稠密且具有解剖学可寻址性（anatomically addressable）的表征，但在经过验证的研究中完全缺失。第二，可微分与神经逆运动学本可取代脆弱的事后 IK（post-hoc IK），目前仅出现于 MICRO 中。第三，视频动力学是 MACRO 输出中服务最少的一类（第 3.3.5 节），却是 MICRO 中活跃的模块。第四，以人为中心的基座模型尚未在经过验证的研究中使用。这些模块共同构成了一条以网格为中心（mesh-centred）的单一流程的组件，并界定了第 5 节所提框架的设计空间。

## 4. 讨论

## 4.1 主要发现综述

本次范围综述（scoping review）比较了生物力学中基于视频的无标记动作捕捉的经过验证文献（MACRO 语料库）与新兴的计算机视觉前沿技术（策展后的 MICRO 语料库）。五项结论尤为突出。该领域尚处于年轻阶段且发展迅速，纳入研究中的大多数发表于过去两年。其证据基础较为狭窄：受试者多为健康成人，在实验室中接受步态（gait）研究，样本量较小，且几乎不涉及实时或实验室外应用。方法在两种模态（单目与多摄像机）下可归为五个架构家族，但大多数流程未进行任何 biomechanical refinement，仅报告原始几何角度。计量学性能（metrological performance）未能达到该领域所宣称的与基于标记方法等效的水平：矢状面（sagittal）下肢一致性通常超过 McGinley 等人（McGinley et al.）所关联的、可能引发临床误判的误差阈值，而离面（out-of-plane）运动学与动力学则鲜有验证。最有可能改变这一局面的构建模块——即基座模型网格重建、可微分逆运动学和视频动力学——在计算机视觉领域十分活跃，但在经过验证的生物力学研究中几乎缺席，这与计算机视觉创新与其经过验证的生物医学应用之间存在的预期延迟相符。

## 4.2 按模态与临床转化划分的成熟度

两种模态在部署便捷性与计量完整性之间进行权衡。多视角三角测量是通往完整三维生物力学分析中更为成熟的路径。它是唯一一类能够常规报告冠状面（额状面）和水平面（横断面）运动学的家族，并且汇集了逆运动学与肌肉骨骼建模流程，特别是 OpenCap 系列，使关节角度具有可解释性。它仍然受限于实验室环境，并需要多台经过同步与标定的摄像机。单目模态使用单一摄像机且无需标定，在临床现场部署便捷得多。然而，大多数二维单目研究仅停留在原始几何角度（第 3.2 节），而单目三维姿态提升的误差最大（下肢误差中位数约为 6° 至 7°）。因此，临床可接受性的差距属于整个领域的共性问题，而非某一薄弱架构的个别缺陷。时空步态参数（如步行速度）与参考系统高度一致，但论点更为狭窄：对基于标记的关节运动学声称可互换性，对大多数流程而言仍为时过早。

## 4.3 动力学——下一前沿

已验证文献中输出最匮乏的类别是动力学。地面反作用力与关节力矩与关节负荷、平衡以及跌倒风险联系最为直接，却仅在 9 项研究中针对测力台进行了验证。它们的临床可操作性可能比单纯运动学更强。一个新兴的 MICRO 模块正瞄准这一缺口：从视频进行物理约束驱动与数据驱动的力与力矩估计；当前的生成式基础模型已能以接近实验室的精度从运动学估计地面反作用力 [163]。首批从单摄像机到肌肉骨骼动力学的端到端流程也已出现 [12]。因此，无测力台的动力学估计是一项处于早期阶段且尚未得到充分验证的方向。与矢状面关节角度精度的渐进提升相比，它具备更高的潜在临床价值，但验证成本也显著更高，因为它在运动学误差之上又叠加了逆动力学建模误差，并且最难在稳态步态之外的场景下加以验证。

## 4.4 已验证文献中的空白

三个空白界定了转化研究的前沿。第一个是平面覆盖。平面外运动学鲜有报告（以膝关节为例：矢状面 65 项，冠状面 5 项，水平面 3 项），因此与许多临床问题最相关的测量恰恰是最少得到验证的，并且在被报告时误差更大。第二个是受试人群。临床与病理性队列研究稀少，老年与儿科队列更为罕见，尽管这些群体最能从可及的运动分析中获益。第三个是运行条件。实时分析（仅占研究的 7%）以及居家或野外采集（几乎缺失）在很大程度上仍未得到解决，尽管二者皆为在实验室之外开展分析的前提条件。弥合这些空白比进一步基准测试健康成人的矢状面步态更为优先。

## 4.5 本综述的优势与局限

The review has two main strengths. It combines a registered PRISMA-ScR protocol with a broad three-database search and a dual-tier design that places the validated literature within the trajectory of the underlying computer-vision research. Several limitations apply. By design the review excludes black-box commercial systems (Section 2.3), including Theia3D, among the most clinically deployed, and RGB-D depth sensors that remain common in rehabilitation; the findings therefore characterise the validated open-architecture evidence base, not the full deployed clinical landscape. The MICRO corpus is curated and non-exhaustive, so its block counts reflect screening and publication volume, not biomechanical importance, and we used them only qualitatively. Data extraction used a large-language-model-assisted pipeline with cross-vendor checking and adjudication, validated by manual re-extraction of a random 20% of the included studies, and we report only cleaned values, excluding signed-bias, velocity-derived and wrong-metric entries. As a scoping review, the synthesis is structured and narrative, not meta-analytic, and we did not formally appraise risk of bias. We benchmarked against McGinley et al.'s error bands, 2° or less acceptable, 2 to $5 ^ { \circ }$ reasonable, above $5 ^ { \circ }$ likely to mislead [131]; acceptable error ultimately depends on the application, so this band analysis is a reference point, not an absolute cut-off.

## 4.6 对临床实践与康复研究的启示

当前可用输出取决于测量内容（表 3）。步行速度、步长、步态周期时间等时空参数与参考系统高度一致，适用于监测、筛查与远程评估；尤其是步行速度，在老年人中被誉为第六生命体征 [165]。矢状面关节角度可用于总体定性趋势分析，但不足以支持需要 3 至 5° 以下分辨率的临床决策，例如追踪细微的康复变化。冠状面与水平面角度以及所有动力学指标仍处于研究级别，尚未获得临床使用验证。

Table 3  
无标记运动捕捉输出的临床就绪程度。
<table><tr><td>就绪程度等级</td><td>输出</td><td>适用场景</td></tr>
<tr><td>可供临床使用</td><td>时空参数（步行速度、步长、步态周期时间）</td><td>监测、筛查与远程评估</td></tr>
<tr><td>可用于筛查与趋势分析</td><td>矢状面关节角度（整体屈伸）</td><td>定性趋势追踪；不适用于需低于三至五度分辨率的临床决策</td></tr>
<tr><td>仅限研究使用</td><td>冠状面与水平面关节角度；无测力台的动力学；病理性步态画像</td><td>不可用于临床决策</td></tr></table>

一个安全性要点从该领域的结构中可得出。可部署性最强的工具（单目单相机流水线）同时也是精炼程度最低的，其中 47 个中有 46 个仅输出原始几何角度。因此，可部署性与验证就绪度呈负相关：一个廉价且易获取的单目工具可能正因为其可及性而进入临床，而它尚未针对其被用于的临床决策得到验证。

健康成人的验证结果不能直接用于患者。病理步态违反了这些流水线赖以构建的假设：助行器、矫形器和手杖等辅助设备；异常形态；震颤、痉挛和张力障碍等不自主运动；以及画圈步态和剪刀步态等严重的离面偏差。5° 的健康成人误差对偏瘫或蹲伏步态患者而言并非安全的先验。所需的是针对特定人群的验证，而非仅仅增加受试者数量。上述每一点都对应着对某项特定假设的违反。辅助设备和矫形器遮挡或位移了姿态估计所依赖的关键点。不自主运动违反了对轨迹进行去噪时所依赖的时序平滑先验。最大的病理偏差发生在额状面和水平面，而这些系统在额状面和水平面上的估计精度恰恰最差，于是临床信号与测量误差落在同一平面内。

三项实际风险值得注意。姿态估计的训练与验证数据偏向年轻、瘦削、浅肤色的成年人，这与本综述所记录的偏差一致，因此代表性最低的人群中的性能可能更差，这是一项公平性关切。临床视频和家庭视频引发了标记实验室中并不存在的隐私和知情同意问题。并且，治理必须使验证先于部署。

## 5. 面向临床可部署无标记生物力学的未来设计指南

本节实现第三个目标：将已验证的 MACRO 经验与新兴的 MICRO 模块转化为面向临床可部署流程的设计指南。这些是产生假设的方向，而非经过验证的方法，每条都对应第 3 节和第 4 节中记录的某一缺口，而非在此重述。有一条原则统领全局：对于临床目标人群——老年与病理性群体——泛化能力必须来自普遍性的东西，即解剖学、力学以及个体特异性模型，而非来自在排除了这些人群的数据上学习得到的先验。因此，该方向将已成熟的组件（个体特异性的 OpenSim 缩放与逆运动学，以及基于物理学的细化）与新兴组件（密集网格恢复；可微的、基于模型的逆运动学；逐自由度的不确定性）以及一些尚未在已验证生物力学中测试过的组件（多视图网格融合和两级网格到解剖的对应关系）相结合。任何学习得到的先验都被保留在这样的阶段：其误差可针对图像进行核查，且组件可替换，从而使最终保真度建立在个体特异性解剖学与力学之上。所有组件均来自已发表文献，且均以公开代码开放获取。本视角不提出任何专有系统。我们提炼出五条指南（图 9）。

首先，我们提出一种密集参数化网格前端——逐帧恢复密集参数化人体网格，而非稀疏关键点，并使用在已验证文献库中尚未出现的人体网格基础模型（例如 SAM 3D Body [155] 及其实时变体 [166]，建立在 Sapiens 系列 [153] 之上，MedSapiens [167] 展示了其向临床标志点的迁移能力）。密集网格携带了分段朝向与平面外旋转，这恰恰是已验证单目方法族很少报告的信息。其估计器承袭了来自健全人群的偏差，但该风险是有限的：估计器是一个可互换的组件，会随着模型的多元化而改进；并且网格误差是局部的、可针对图像核查的，不同于学习到的运动学模型内部那种静默的回归。这回应了第 3.5 节的第一个缺口：网格恢复已出现于 MICRO 中，但所有已验证的 MACRO 研究中均未采用。

其次，我们解决从网格到临床定义解剖结构的步骤。参数化网格没有临床定义的关节中心，而从固定网格位置读取虚拟标记会将通用的、可标定的网格到解剖结构的偏移与微小的个体差异混为一谈。由于一到两厘米的不精确配准可能引起五到十度的关节角度误差，这对非典型人群最为关键，因为他们的临床有意义偏差本身就较小。我们提出一种两级对应关系：一个在带注释的参考上一次标定然后冻结的通用网格到标志点映射，加上一个小的、强正则化的受试者特异性残差，其中骨性标志点比软组织标志点受到更严格的约束。神经定位场（Neural Localizer Fields）[158] 通过允许在网格上查询任何解剖点使这成为可能。这弥补了 MICRO 留下的一项空白：虚拟标记从固定位置读取，缺少非典型人群所需的标定。

第三，运动学推断应当是受试者特异性的、基于模型的，而非习得的。这可以说是临床转化中最重要的选择。存在三种方案。习得神经逆运动学由经过人群训练的网络（BioPose [11]、Fast Neural IK [162]）预测角度，其泛化能力受训练分布的制约。可微逆运动学则是同一基于模型的求解被渲染为可微的，因此缩放、配准与逆运动学通过梯度下降联合优化（Differentiable Biomechanics [160]、JAX-IK [161]，以及 AddBiomechanics [168]——后者还求解逆动力学）；可微并不等同于习得。经典的基于模型求解（OpenSim）在解剖约束下逐受试者拟合关节角度，不进行人群训练。对于老年人群与病理性人群，这一区分可能具有决定性意义：在健全人运动上学得的逆运动学可能向其训练模式回归，从而掩盖测量本应捕捉到的病理性偏差。因此，我们建议运动学推断应基于受试者特异性的、基于模型的逆运动学（可选择性地使其可微），而非基于习得映射。这弥补了第 3.5 节中的第二项空白——可微与神经逆运动学仅出现于 MICRO，而习得求解能否迁移到目标人群仍悬而未决。

第四，我们区分两种部署模式。在单目情况下（单个相机，最易于部署），通过一种时间形状锁定（temporal shapelock）将逐帧网格在每次试验中整合为单一形态，该锁定固定了肢段长度；因为软组织伪影同样会改变基于标记点的肢段长度，所以形状锁定估计值在这一维度上有望接近基于标记点采集的一致性，这是一个可检验的假设[117]。随后，一个基于物理的精修阶段（足部接触、防漂移、平滑）可对抗单目漂移，沿袭自 OpenCap Monocular [12]。在多相机情况下，不再三角化稀疏关键点，而是通过可见性加权重投影，将单一网格联合拟合到所有已标定的视图；固定拓扑使视图间对应关系成为固有属性，并允许多个视图直接观测肢段长度。这一网格融合路线，受 MAMMA [169] 启发，在已验证的生物力学研究中基本未被探索，其推测的收益（遮挡鲁棒性、面外恢复）仍有待检验（见第 3.3.4 和 4.2 节中的差距）。

第五，我们转向输出、动力学和不确定性。两种模式都以所有自由度的关节运动学为目标。不使用测力台的动力学是本领域验证最少的输出（第 3.3.5 节），且仍是一个遥远的前景：生成式基础模型已可从运动学估算地面反作用力[163]，但其对非典型临床队列的泛化能力尚未经过检验，并且，如同学习式逆运动学一样，要在这些人群中可靠应用，它们需要涵盖老年人和病理人群的训练数据。每个自由度的不确定性，在已验证文献中几乎缺失，是一个更直接且临床必需的补充，因为测量的可靠性决定着临床决策；使用归一化流的概率姿态估计（Wehrbein 等[164]）提供了一种途径。两者均属于第 3.3.5 节所识别的验证最少的输出。

## 6. 结论

基于视频的无标记运动捕捉虽已快速成熟，但已验证的证据尚不支持其与基于标记点的系统在临床关节运动学上的互换性。与临床生物力学最相关的功能，即面外运动学、动力学，以及在现实条件下对病理和老年人群的验证，是发展最不充分的，而能够解决这些问题的计算机视觉进展尚未进入已验证文献。我们将这些已记录的局限性提炼为基于证据的设计指南，每条都将一项差距映射到相应的前沿模块，作为待检验的视角而非已验证的方法。进展将依赖于标准化的、以生物力学为基础的评估——报告关节角度而非关键点位置误差，依赖于在临床相关的关键人群和场景中进行验证，并依赖于缩短计算机视觉创新与生物医学验证之间的滞后。

## 声明

**伦理审批与参与同意。**

不适用（对已发表文献的二次分析）。

**出版同意。**

不适用。

**竞争性利益。**

作者声明无竞争性利益。

**资助。**

FD 获得法国蔚蓝海岸大学（Université Côte d'Azur）人体运动科学博士院（ED SMH, Sciences du Mouvement Humain）的博士奖学金资助。本研究部分由 PRESAGE 项目（法国国家科研署，ANR-23-PAVH-0002）资助，该项目为文献审查提供了包括 Covidence 系统综述平台在内的审查资源。资助方在本综述的设计、实施、分析或报告撰写中未发挥任何作用。

**作者贡献。**

F.D.：构思、方法学、软件、形式分析、调查、数据管理、可视化、撰写——初稿。E.P.：调查（在 Covidence 上独立、盲法进行的研究筛选与遴选）、撰写——审阅与编辑。F.C.：撰写——审阅与编辑。R.Z.：验证（筛选分歧的裁决）、监督、获取资助、撰写——审阅与编辑。所有作者均阅读并批准了最终稿件。

**致谢。**

不适用。

**数据可用性。**

研究方案与完整的数据提取数据集可在开放科学框架（Open Science Framework）资料库中获取（DOI 10.17605/OSF.IO/TBYPC）；提取表以机器可读的补充数据 1 文件形式提供。

## 参考文献

1. Windolf M, Götzen N, Morlock M. Systematic accuracy and precision analysis of video motion capturing systems: exemplified on the Vicon-460 system. *J Biomech*. 2008;41(12):2776–80. https://doi.org/10.1016/j.jbiomech.2008.06.024.

2. Leardini A, Chiari L, Della Croce U, Cappozzo A. Human movement analysis using stereophotogrammetry. Part 3: soft tissue artifact assessment and compensation. *Gait Posture*. 2005;21(2):212–25. https://doi.org/10.1016/j.gaitpost.2004.05.002.

3. Baker R. Gait analysis methods in rehabilitation. *J Neuroeng Rehabil*. 2006;3(1):4. https://doi.org/10.1186/1743-0003-3-4.

4. Colyer SL, Evans M, Cosker DP, Salo AIT. *Sports Med Open*. 2018;4(1):24. https://doi.org/10.1186/s40798-018-0139-y. 基于视觉的运动分析演进综述及面向无标记点系统开发的先进计算机视觉方法集成。

5. Cao Z, Hidalgo G, Simon T, Wei SE, Sheikh Y. OpenPose: realtime multi-person 2D pose estimation using part affinity fields. *IEEE Trans Pattern Anal Mach Intell*. 2021;43(1):172–86. https://doi.org/10.1109/TPAMI.2019.2929257.

6. Bazarevsky V, Grishchenko I, Raveendran K, Zhu T, Zhang F, Grundmann M. BlazePose: on-device real-time body pose tracking. *arXiv:2006.10204*. 2020.

7. Sun K, Xiao B, Liu D, Wang J. Deep high-resolution representation learning for human pose estimation. In: *Proc IEEE/CVF Conf Comput Vis Pattern Recognit (CVPR)*. 2019. pp. 5686–5696. https://doi.org/10.1109/CVPR.2019.00584.

8. Pagnon D, Domalain M, Reveret L. Pose2Sim：三维无标记运动生物力学端到端工作流程——第1部分：鲁棒性. Sensors. 2021;21(19):6530. https://doi.org/10.3390/s21196530.

9. Uhlrich SD, Falisse A, Kidzinski L, Muccini J, Ko M, Chaudhari AS, et al. OpenCap：从智能手机视频中获取人体运动动力学. PLoS Comput Biol. 2023;19(10):e1011462. https://doi.org/10.1371/journal.pcbi.1011462.

10. Zhu W, Ma X, Liu Z, Liu L, Wu W, Wang Y. MotionBERT：学习人体运动表示的统一视角. 见：2023 IEEE/CVF 国际计算机视觉大会 (ICCV). 2023. pp. 15039–15053. https://doi.org/10.1109/iccv51070.2023.01385

11. Koleini F, Saleem MU, Wang P, Xue H, Helmy A, Fenwick A. BioPose：基于单目视频的生物力学精确三维姿态估计. 见：2025 IEEE/CVF 冬季计算机视觉应用大会 (WACV). IEEE; 2025. arXiv:2501.07800 [cs.CV].

12. Gilon S, Miller EY, Uhlrich SD. OpenCap Monocular：从单个智能手机视频获取三维人体运动学与肌肉骨骼动力学. arXiv:2603.24733 [cs.CV]. 2026.

13. Balaban B, Tok F. 卒中患者的步态障碍. PM R. 2014;6(7):635–42. https://doi.org/10.1016/j.pmrj.2013.12.017.

14. Lam WWT, Tang YM, Fong KNK. 无标记运动捕捉（MMC）技术在康复临床测量中的应用系统综述. J Neuroeng Rehabil. 2023;20(1):57. https://doi.org/10.1186/s12984-023-01186-9.

15. Tricco AC, Lillie E, Zarin W, O'Brien KK, Colquhoun H, Levac D, et al. 范围综述的 PRISMA 扩展（PRISMA-ScR）：清单与说明. Ann Intern Med. 2018;169(7):467–73. https://doi.org/10.7326/M18-0850.

16. Peters MDJ, Marnie C, Tricco AC, Pollock D, Munn Z, Alexander L, et al. 范围综述实施的方法学最新指南. JBI Evid Synth. 2020;18(10):2119–26. https://doi.org/10.11124/jbies-20-00167.

17. Young F, Mason R, Morris R, Stuart S, Godfrey A. 基于物联网的单摄像头无标记跑步步态评估. Sensors. 2023;23(2):696. https://doi.org/10.3390/s23020696.

18. Verhoeven M, Zandvoort CS, Dominici N. 从有标记到无标记：DeepLabCut 在成人与初学走路幼儿二维矢状面步态分析中的验证. J Biomech. 2025;186:112708. https://doi.org/10.1016/j.jbiomech.2025.112708.

19. Takeda I, Yamada A, Onodera H. 人工智能辅助运动捕捉在医学中的应用：无标记与被动标记运动捕捉的对比研究. Comput Methods Biomech BioMed Eng. 2020;24(8):864–73. https://doi.org/10.1080/10255842.2020.1856372.

20. Stenum J, Rossi C, Roemmich RT. 基于二维视频与姿态估计的人体步态分析. PLoS Comput Biol. 2021;17(4):e1008935. https://doi.org/10.1371/journal.pcbi.1008935.

21. Molteni LE, Andreoni G. Comparing the Accuracy of Markerless Motion Analysis and Optoelectronic System for Measuring Gait Kinematics of Lower Limb. *Bioengineering*. 2025;12(4):424. https://doi.org/10.3390/bioengineering12040424.

22. Sabo A, Gorodetsky C, Fasano A, Iaboni A, Taati B. Concurrent validity of Zeno instrumented walkway and video-based gait features in adults with Parkinson's disease. *IEEE J Translational Eng Health Med*. 2022;10:1–11. https://doi.org/10.1109/jtehm.2022.3180231.

23. Panconi G, Grasso S, Guarducci S, Mucchi L, Minciacchi D, Bravi R. DeepLabCut custom-trained model and the refinement function for gait analysis. *Sci Rep*. 2025;15(1):2364. https://doi.org/10.1038/s41598-025-85591-1.

24. Ota M, Tateuchi H, Hashiguchi T, Ichihashi N. Verification of validity of gait analysis systems during treadmill walking and running using human pose tracking algorithm. *Gait Posture*. 2021;85:290–7. https://doi.org/10.1016/j.gaitpost.2021.02.006.

25. Moro M, Marchesi G, Odone F, Casadio M. Markerless gait analysis in stroke survivors based on computer vision and deep learning: a pilot study. Proceedings of the 35th Annual ACM Symposium on Applied Computing. 2020:2097–2104. https://doi.org/10.1145/3341105.3373963

26. Menychtas D, Petrou N, Kansizoglou I, Giannakou E, Grekidis A, Gasteratos A, et al. Gait analysis comparison between manual marking, 2D pose estimation algorithms, and 3D marker-based system. *Front Rehabilitation Sci*. 2023;4:1238134. https://doi.org/10.3389/fresc.2023.1238134.

27. Mehdizadeh S, Nabavi H, Sabo A, Arora T, Iaboni A, Taati B. Concurrent validity of human pose tracking in video for measuring gait parameters in older adults: a preliminary analysis with multiple trackers, viewing angles, and walking directions. *J Neuroeng Rehabil*. 2021;18(1):139. https://doi.org/10.1186/s12984-021-00933-0.

28. Lonini L, Moon Y, Embry K, Cotton RJ, McKenzie K, Jenz S, et al. Video-Based Pose Estimation for Gait Analysis in Stroke Survivors during Clinical Assessments: A Proof-of-Concept Study. *Digit Biomarkers*. 2022;6(1):9–18. https://doi.org/10.1159/000520732.

29. Ino T, Samukawa M, Ishida T, Wada N, Koshino Y, Kasahara S, et al. Validity of AI-Based Gait Analysis for Simultaneous Measurement of Bilateral Lower Limb Kinematics Using a Single Video Camera. *Sensors*. 2023;23(24):9799. https://doi.org/10.3390/s23249799.

30. Lambricht N, Englebert A, Nguyen AP, Fisette P, Pitance L, Detrembleur C. Impact of Running Clothes on Accuracy of Smartphone-Based 2D Joint Kinematic Assessment During Treadmill Running Using OpenPifPaf. *Sensors*. 2025;25(3):934. https://doi.org/10.3390/s25030934.

31. Lambricht N, Englebert A, Pitance L, Fisette P, Detrembleur C. Quantifying performance and joint kinematics in functional tasks crucial for anterior cruciate ligament rehabilitation using smartphone video and pose detection. *Knee*. 2025;52:171–8. https://doi.org/10.1016/j.knee.2024.11.006.

32. John K, Stenum J, Chiang CC, French MA, Kim C, Manor J, 等. 视频步态分析在跑步机行走与地面行走中应用于脑卒中后患者的精度比较（基于姿态估计）. *Phys Ther*. 2023;104(2):pzad121. https://doi.org/10.1093/ptj/pzad121.

33. Ino T, Samukawa M, Ishida T, Wada N, Koshino Y, Kasahara S, 等. 基于OpenPose的动作分析在测量下蹲垂直跳跃测试中膝外翻的有效性与可靠性. *J Sports Sci Med*. 2024;515–25. https://doi.org/10.52082/jssm.2024.515.

34. Dunn M, Kennerley A, Murrell-Smith Z, Webster K, Middleton K, Wheat J. 视频帧插值在无标记、单摄像头步态分析中的应用. *Sports Eng*. 2023;26(1):22. https://doi.org/10.1007/s12283-023-00419-3.

35. Drazan JF, Phillips WT, Seethapathi N, Hullfish TJ, Baxter JR. 走出实验室：无标记运动捕捉准确量化垂直跳跃中矢状面运动学参数. *J Biomech*. 2021;125:110547. https://doi.org/10.1016/j.jbiomech.2021.110547.

36. Boldo M, Di Marco R, Martini E, Nardon M, Bertucco M, Bombieri N. 单摄像头无标记系统在地面步态监测中的可靠性研究. *Comput Biol Med*. 2024;171:108101. https://doi.org/10.1016/j.compbiomed.2024.108101.

37. Andreoni G, Molteni LE. 无标记运动分析与光电系统在测量下肢步态运动学中的精度比较. *Lect Notes Comput Sci*. 2024;3–15. https://doi.org/10.1007/978-3-031-61063-9\_1.

38. Anderson JT, Stenum J, Roemmich RT, Wilson RB. 基于姿态估计的无标记视频步态分析在神经发育障碍与正常幼儿中的验证. *Front Digit Health*. 2025;7. https://doi.org/10.3389/fdgth.2025.1542012.

39. Aleksic J, Kanevsky D, Mesaroš D, Knezevic OM, Cabarkapa D, Bozovic B, 等. 自动反向运动垂直跳跃分析的验证：无标记姿态估计与三维标记式运动捕捉系统的比较. *Sensors*. 2024;24(20):6624. https://doi.org/10.3390/s24206624.

40. Yagi K, Sugiura Y, Hasegawa K, Saito H. 使用单个RGB摄像头在家中进行步态测量. *Gait Posture*. 2020;76:136–40. https://doi.org/10.1016/j.gaitpost.2019.10.006.

41. Tahara AK, Chinaglia AG, Monteiro RLM, Bedo BLS, Cesar GM, Santiago PRP. 基于无标记、像素级机器学习的人行通道时空参数预测. *Brazilian J Motor Behav*. 2025;19(1):e462. https://doi.org/10.20338/bjmb.v19i1.462.

42. Francia C, Donno L, Chiosso S, Palotti G, Tarabini M, Galli M. 用于AI无标记三维下肢运动分析的双摄像头系统. *Lecture Notes in Computer Science*. 2025;:303–312. https://doi.org/10.1007/978-3-031-97781-7\_22

43. Yamamoto M, Shimatani K, Hasegawa M, Kurita Y, Ishige Y, Takemura H. 使用 OpenPose 测量不同矫形器步态模式下的时空间参数和下肢关节运动学参数的准确性. *IEEE Trans Neural Syst Rehabil Eng*. 2021;29:2666–75. https://doi.org/10.1109/tnsre.2021.3135879.

44. Wade L, Needham L, Evans M, McGuigan P, Colyer S, Cosker D, et al. 二维正面与矢状面无标记运动捕捉的检验：对无标记应用的意义. *PLoS ONE*. 2023;18(11):e0293917. https://doi.org/10.1371/journal.pone.0293917.

45. Kim J, Kim R, Byun K, Kang N, Park K. 使用人体姿态估计评估帕金森病患者的时空间与运动学步态参数：近正面视图与侧面视图的比较. *PLoS ONE*. 2025;20(1):e0317933. https://doi.org/10.1371/journal.pone.0317933.

46. Balci IC, Sayin I, Salturk S, Gursoy R, Ozsoy U, Dogru HC, et al. 生物力学运动分析中无标记技术的可靠性评估：性能比较. *Front Sports Act Living*. 2026;7. https://doi.org/10.3389/fspor.2025.1712332.

47. Hii CST, Gan KB, Zainal N, Mohamed Ibrahim N, Azmin S, Mat Desa SH, et al. 基于无标记姿态估计模型的自动化步态分析. *Sensors*. 2023;23(14):6489. https://doi.org/10.3390/s23146489.

48. Ge F, Wu C, Ge F, Xu S, Xiao J. OpenPose 用于测量膝骨关节炎患者动态行走视频中 HKA 角的可靠性与有效性. *Sci Rep*. 2025;15(1). https://doi.org/10.1038/s41598-025-09627-2.

49. Asaeda M, Onishi T, Ito H, Miyahara S, Mikami Y. 基于机器学习姿态估计计算单腿下落着地时膝外翻角的可靠性与有效性. *Heliyon*. 2024;10(17):e36338. https://doi.org/10.1016/j.heliyon.2024.e36338.

50. Haberkamp LD, Garcia MC, Bazett-Jones DM. 人工智能人体姿态估计模型用于测量单腿深蹲运动学的有效性. *J Biomech*. 2022;144:111333. https://doi.org/10.1016/j.jbiomech.2022.111333.

51. Washabaugh EP, Shanmugam TA, Ranganathan R, Krishnan C. 开源姿态估计方法测量步态运动学准确性的比较. *Gait Posture*. 2022;97:188–95. https://doi.org/10.1016/j.gaitpost.2022.08.008.

52. Gao X, Cheng X, Jiao Y, Reading S, Zhang Y. 基于视频的步态分析能否识别步态偏差？BlazePose 姿态估计算法的验证. *2025 第 47 届 IEEE 医学与生物工程学会国际年会 (EMBC)*. 2025:1–5. https://doi.org/10.1109/embc58623.2025.11254531

53. Kumthekar PS, Sharma A, Malla S, Richardson RT, Morales AW, Nguyen H et al. 用于上肢运动分析的移动应用与云计算开发的初步研究. *2024 IEEE Cloud Summit*. 2024:24–27. https://doi.org/10.1109/cloud-summit61220.2024.00011

54. Nishizawa K, Oba Y, Yamada K, Tanaka I, Tsumugiwa T, Yokogawa R, Watanabe T. Evaluation of the clinical utility of a gait analysis system using pose estimation techniques in physical therapy. In: 2024 SICE International Symposium on Control Systems (SICE ISCS). Higashi-Hiroshima, Japan; 2024. pp. 107–10.

55. Hii CST, Gan KB, You HW, Zainal N, Ibrahim NM, Azmin S et al. Frontal Plane Gait Analysis using Pose Estimation Models. 2023 IEEE 2nd National Biomedical Engineering Conference (NBEC). 2023:1–6. https://doi.org/10.1109/nbec58134.2023.10352623

56. Irfan M, Suryadevara NK. Reliable Gait Measurements Using Smartphone Vision Sensor. IEEE Sens J. 2025;25(1):1487–94. https://doi.org/10.1109/jsen.2024.3487541.

57. Ceriola L, Taborri J, Donati M, Rossi S, Patanè F, Mileti I. Comparative Analysis of Markerless Motion Capture Systems for Measuring Human Kinematics. IEEE Sens J. 2024;24(17):28135–44. https://doi.org/10.1109/jsen.2024.3431873.

58. Kidziński Ł, Yang B, Hicks JL, Rajagopal A, Delp SL, Schwartz MH. Deep neural networks enable quantitative movement analysis using single-camera videos. Nat Commun. 2020;11(1). https://doi.org/10.1038/s41467-020-17807-z.

59. Hwang W, Shim D, Kim J, Oh KR, Chung SG, Beom J, et al. Sit-to-Stand Power From 2D Pose Estimation as an Indicator of Muscle Strength in Older Adults. J Cachexia Sarcopenia Muscle. 2026;17(1):e70208. https://doi.org/10.1002/jcsm.70208.

60. Irfan M, Suryadevara NK, Biswas R, Gaddam A. Kinematic Gait Analysis Using Markerless System to Determine Joint Angles. Lecture Notes Networks Syst. 2024;551–9. https://doi.org/10.1007/978-981-97-2671-4\_42.

61. Talaa S, Jilbab A, El Yousfi Alaoui MH, Ihssane K. Markerless Knee Angle Measurement: Advancements in Gait Analysis Using Computer Vision. Lecture Notes in Networks and Systems. 2025;:103–112. https://doi.org/10.1007/978-3-031-94623-3\_9

62. Ang YH, Chan CK, Yap SC, Toa CK, Tran P, Goh SK. Markerless Human Motion Analysis for Telerehabilitation: A Case Study on Squat. Adv Sci Technol Innov. 2024;249–59. https://doi.org/10.1007/978-3-031-52303-8\_18.

63. Tsintzira K, Smyrli A, Mastrogeorgiou A, Papadopoulos E. Low-Cost Markerless Gait Analysis Using a Minimal Human Model. 2025 IEEE 25th International Conference on Bioinformatics and Bioengineering (BIBE). 2025:558–562. https://doi.org/10.1109/BIBE66822.2025.00099

64. Natraj S, Messmer T, Fujii Y, Suzuki K, Riener R, Eriks-Hoogland I, et al. 3D pose estimation for scalable remote gait kinematics assessment. npj Digit Med. 2025;9(1):37. https://doi.org/10.1038/s41746-025-02211-y.

65. Zhu X, Boukhennoufa I, Liew B, Gao C, Yu W, McDonald-Maier KD, et al. Monocular 3D Human Pose Markerless Systems for Gait Assessment. Bioengineering. 2023;10(6):653. https://doi.org/10.3390/bioengineering10060653.

66. Mercadal-Baudart C, Liu CJ, Farrell G, Boyne M, Gonzalez Escribano J, Smolic A, et al. 基于单摄像头视角无标记三维姿态估计的运动量化. Heliyon. 2024;10(6):e27596. https://doi.org/10.1016/j.heliyon.2024.e27596.

67. Lagomarsino B, Massone A, Odone F, Casadio M, Moro M. 颈脊髓损伤后基于视频的双侧上肢运动活动无标记评估. Comput Biol Med. 2025;196:110908. https://doi.org/10.1016/j.compbiomed.2025.110908.

68. Carrasco-Plaza J, Cerda M. 单目相机三维人体姿态估计的临床应用评估. Commun Comput Inform Sci. 2022;121–34. https://doi.org/10.1007/978-3-030-98457-1\_10.

69. Barzyk P, Zimmermann P, Stein M, Keim D, Gruber M. 反向跳中基于AI智能手机的髋、膝和踝关节运动学无标记运动捕捉. Eur J Sport Sci. 2024;24(10):1452–62. https://doi.org/10.1002/ejsc.12186.

70. Barzyk P, Boden AS, Howaldt J, Stürner J, Zimmermann P, Seebacher D, et al. 促进脑卒中患者临床步态分析应用的步骤：基于单一2D RGB智能手机视频的步态分析系统验证. Sensors. 2024;24(23):7819. https://doi.org/10.3390/s24237819.

71. Azhand A, Rabe S, Müller S, Sattler I, Heimann-Steinert A. 基于单一单目视频的算法可提供高度有效且可靠的步态参数. Sci Rep. 2021;11(1). https://doi.org/10.1038/s41598-021-93530-z.

72. Wang H, Su B, Lu L, Jung S, Qing L, Xie Z, et al. 通过单一摄像头与计算机视觉实现无标记步态分析. J Biomech. 2024;165:112027. https://doi.org/10.1016/j.jbiomech.2024.112027.

73. Ueno R. 步态中无需标定的单目视觉肌肉骨骼模拟. Heliyon. 2024;10(11):e32078. https://doi.org/10.1016/j.heliyon.2024.e32078.

74. Hulleck AA, AlShehhi A, El Rich M, Khan R, Katmah R, Mohseni M, et al. BlazePose-Seq2Seq：利用常规RGB摄像头实现稳健的步态评估. IEEE Trans Neural Syst Rehabil Eng. 2024;32:1715–24. https://doi.org/10.1109/tnsre.2024.3391908.

75. Wang XM, Smith DT, Zhu Q. 基于网络摄像头的三维关节活动度评估机器学习方法. PLoS ONE. 2023;18(10):e0293178. https://doi.org/10.1371/journal.pone.0293178.

76. He R, You Z, Zhou Y, Chen G, Diao Y, Jiang X, et al. 一种使用单目视频进行帕金森病步态检测的新型多级三维姿态估计框架. Front Bioeng Biotechnol. 2024;12. https://doi.org/10.3389/fbioe.2024.1520831.

77. Liang S, Zhang Y, Diao Y, Li G, Zhao G. 使用三维无标记姿态估计算法的步态分析系统的信度与效度. Front Bioeng Biotechnol. 2022;10. https://doi.org/10.3389/fbioe.2022.857975.

78. Lin ZY, Lyu B, van der Fernandez JC, Seth A, Zhang X. 基于生物力学模型和合成训练数据的视频三维运动学估计. 2024 IEEE/CVF 计算机视觉与模式识别会议研讨会 (CVPRW). 2024;:1441–1450. https://doi.org/10.1109/cvprw63382.2024.00151

79. Pappas MC, Boughanem DJ, Baudendistel ST, Chen S, Liu S, Acevedo GT 等. 基于视频的帕金森病临床步态分析：一种使用额状面视频和机器学习的新方法. 2024 年第 46 届 IEEE 医学与生物工程学会国际年会 (EMBC). 2024:1–4. https://doi.org/10.1109/embc53108.2024.10781504

80. Rode D, Dunkel A, Willi R, Wolf P, Xiloyannis M, Riener R. 单目人体姿态估计模型在临床运动分析中的评估. Sci Rep. 2025;15(1):38767. https://doi.org/10.1038/s41598-025-22626-7.

81. Athama A, Srivastava A, Huang S, Wang K, Li Y, Zhai X. 面向医疗应用的无线单摄像头无标记动作捕捉系统. 2025 IEEE 国际高性能计算与通信会议 (HPCC). 2025;:694–700. https://doi.org/10.1109/hpcc67675.2025.00106

82. Huang T, Ruan M, Huang S, Fan L, Wu X. 使用无标记和基于标记的动作捕捉进行步态下肢运动学与关节力矩计算的比较. Front Bioeng Biotechnol. 2024;12:1280363. https://doi.org/10.3389/fbioe.2024.1280363.

83. Horsak B, Simonlehner M, Quehenberger V, Dumphart B, Wegscheider P, Kranzl A, 等. 单目三维无标记步态分析在模拟病理步态中的有效性与可靠性：与 OpenCap 的比较研究. J Biomech. 2025;193:112986. https://doi.org/10.1016/j.jbiomech.2025.112986.

84. Hou H, Zhao S, Fan Z, Jin W, Zhu J, Ruan L 等. 基于非同步摄像机的无标记动作捕捉系统. Lecture Notes in Computer Science. 2025;:567–78. https://doi.org/10.1007/978-981-95-2098-5\_48

85. Nguyen KX, Zheng L, Hawke AL, Carey RE, Breloff SP, Li K, 等. 基于深度学习的多视图图像全身运动学估计. Comput Vis Image Underst. 2023;235:103780. https://doi.org/10.1016/j.cviu.2023.103780.

86. Li J, Wang Z, Wang C, Su W. GaitFormer：通过单个低成本 RGB 摄像头利用双流时空视觉 Transformer 进行临床步态分析. Knowl Based Syst. 2024;295:111810. https://doi.org/10.1016/j.knosys.2024.111810.

87. Kumar KS, Jamsrandorj A, Kim J, Mun KR. 基于深度学习方法利用视觉系统预测下肢运动学. 2022 年第 44 届 IEEE 医学与生物工程学会国际年会 (EMBC). 2022;:177–181. https://doi.org/10.1109/embc48229.2022.9871577

88. Bittner M, Yang WT, Zhang X, Seth A, van Gemert J. 迈向单摄像头人体三维运动学. Sensors. 2023;23(1):341. https://doi.org/10.3390/s23010341.

89. Salehi M, Taheri A, Choi S, Kim JH. 使用移动设备的无标记运动捕捉系统评估职业举升任务中三维关节运动学的测量. Appl Ergon. 2026;134:104743. https://doi.org/10.1016/j.apergo.2026.104743.

90. Verheul J, Hughes O, Hitchens L, Atherton T, Sauter T, Radwan A, et al. 跑步中的无标记运动捕捉：全身、关节和肌肉运动学的有效性和可靠性. J Biomech. 2026;195:113133. https://doi.org/10.1016/j.jbiomech.2025.113133.

91. Zago M, Luzzago M, Marangoni T, De Cecco M, Tarabini M, Galli M. 使用视觉骨骼化和立体视觉进行人体运动的三维跟踪. Front Bioeng Biotechnol. 2020;8:181. https://doi.org/10.3389/fbioe.2020.00181.

92. Turner JA, Chaaban CR, Padua DA. OpenCap 的验证：用于重返运动任务中下肢运动学的低成本无标记运动捕捉系统. J Biomech. 2024;171:112200. https://doi.org/10.1016/j.jbiomech.2024.112200.

93. Torvinen P, Ruotsalainen KS, Zhao S, Cronin N, Ohtonen O, Linnamo V. 跑步机滑行滑雪期间三维无标记运动捕捉系统精度评估. Bioengineering. 2024;11(2):136. https://doi.org/10.3390/bioengineering11020136.

94. Pagnon D, Domalain M, Reveret L. Pose2Sim：三维无标记运动学端到端工作流程——第二部分：精度. Sensors. 2022;22(7):2712. https://doi.org/10.3390/s22072712.

95. Needham L, Evans M, Cosker DP, Colyer SL. 一种用于理解精英钢架雪车运动员推启动作的新型无标记运动分析系统的开发、评估与应用. PLoS ONE. 2021;16(11):e0259624. https://doi.org/10.1371/journal.pone.0259624.

96. Martis P, Kosutska Z, Kranzl A. 进一步理解基于智能手机的无标记步态分析中的方向局限性：一项初步研究. Sensors. 2024;24(10):3091. https://doi.org/10.3390/s24103091.

97. Needham L, Evans M, Wade L, Cosker DP, McGuigan MP, Bilzon JL, et al. 全自动无标记运动捕捉工作流程的开发与评估. J Biomech. 2022;144:111338. https://doi.org/10.1016/j.jbiomech.2022.111338.

98. Moro M, Marchesi G, Hesse F, Odone F, Casadio M. 无标记与基于标记的步态分析对比：概念验证研究. Sensors. 2022;22(5):2011. https://doi.org/10.3390/s22052011.

99. Lim W. 基于网络摄像头使用 OpenPose 的无标记运动捕捉系统. Int J Hum Mov Sports Sci. 2022;10(5):900–5. https://doi.org/10.13189/saj.2022.100505.

100. Lima YL, Collings T, Hall M, Bourne MN, Diamond LE. 使用 OpenCap 无标记运动捕捉应用程序进行下蹲、跳跃、跳和侧向跨步时躯干和下肢运动学的有效性和可靠性. J Sports Sci. 2024;42(19):1847–58. https://doi.org/10.1080/02640414.2024.2415233.

101. Kakavand R, Ahmadi R, Parsaei A, Edwards WB, Komeili A. Comparison of kinematics and kinetics between OpenCap and a marker-based motion capture system in cycling. *Comput Biol Med*. 2025;192:110295. https://doi.org/10.1016/j.compbiomed.2025.110295.

102. Horsak B, Prock K, Krondorfer P, Siragy T, Simonlehner M, Dumphart B. Inter-trial variability is higher in 3D markerless compared to marker-based motion capture: Implications for data post-processing and analysis. *J Biomech*. 2024;166:112049. https://doi.org/10.1016/j.jbiomech.2024.112049.

103. Horsak B, Eichmann A, Lauer K, Prock K, Krondorfer P, Siragy T, et al. Concurrent validity of smartphone-based markerless motion capturing to quantify lower-limb joint kinematics in healthy and pathological gait. *J Biomech*. 2023;159:111801. https://doi.org/10.1016/j.jbiomech.2023.111801.

104. Harrison SM, Cohen RCZ, Starkey SC, Greenwood J, Cheong E, Nguyen K, et al. Evaluation of the Ergomechanic Markerless Motion Capture System for Lower Body Kinematics During Standing, Squatting, and Walking. *J Biomech Eng*. 2025;147(12):121003. https://doi.org/10.1115/1.4069821.

105. Emmerson J, Needham L, Williams S, Colyer S. Evaluation of a markerless motion capture system for measuring mechanical work during tennis strokes. *J Sports Sci*. 2025;44(10):1348–60. https://doi.org/10.1080/02640414.2025.2555096.

106. D'Haene M, Chorin F, Colson SS, Guérin O, Zory R, Piche E. Validation of a 3D Markerless Motion Capture Tool Using Multiple Pose and Depth Estimations for Quantitative Gait Analysis. *Sensors*. 2024;24(22):7105. https://doi.org/10.3390/s24227105.

107. de Borba EF, Storniolo JL, Cerfoglio S, Capodaglio P, Cimolin V, Peyré-Tartaruga LA, et al. Effect of Walking Speed on the Reliability of a Smartphone-Based Markerless Gait Analysis System. *Sensors*. 2025;25(20):6474. https://doi.org/10.3390/s25206474.

108. D'Antonio E, Taborri J, Palermo E, Rossi S, Patane F. A markerless system for gait analysis based on OpenPose library. 2020 IEEE International Instrumentation and Measurement Technology Conference (I2MTC). 2020;:1–6. https://doi.org/10.1109/i2mtc43012.2020.9128918

109. D'Antonio E, Taborri J, Mileti I, Rossi S, Patane F. Validation of a 3D Markerless System for Gait Analysis based on OpenPose and Two RGB Webcams. *IEEE Sens J*. 2021;21(15):17064–75. https://doi.org/10.1109/jsen.2021.3081188.

110. D'Antonio E, Taborri J, Palermo E, Rossi S, Patane F. Characterization of a low-cost markerless system for 3D gait analysis. 2020 IEEE International Symposium on Medical Measurements and Applications (MeMeA). 2020;:1–6. https://doi.org/10.1109/memea49120.2020.9137236

111. Bertozzi F, Brunetti C, Maver P, Palombi M, Santini M, Galli M, et al. Concurrent validity of IMU and phone-based markerless systems for lower-limb kinematics during cognitively-challenging landing tasks. *J Biomech*. 2025;191:112883. https://doi.org/10.1016/j.jbiomech.2025.112883.

112. Bak SY, Ahn J, Choi H, Lee S, Lim W, Kim HD. 无标记运动捕捉系统的有效性及其与髋关节活动度测量身体特征的相关性：一项初步研究. *Sens Mater*. 2025;37(3):965. https://doi.org/10.18494/sam5489.

113. Bae K, Lee S, Bak SY, Kim HS, Ha Y, You JH. 过顶深蹲过程中深度学习无标记运动捕捉系统的同时效度与测试信度. *Sci Rep*. 2024;14(1). https://doi.org/10.1038/s41598-024-79707-2.

114. Ahn J, Choi H, Lee H, Kim SW, Lee J, Kim HD. 基于新型多RGB传感器的三关节连续运动分析对关节角度测量的同时效度与测试内信度：一项初步研究. *Appl Sci*. 2023;14(1):73. https://doi.org/10.3390/app14010073.

115. Verheul J, Robinson MA, Burton S. 利用OpenCap面向场地化的地面反作用力估计与评估的跳跃研究. *J Biomech*. 2024;166:112044. https://doi.org/10.1016/j.jbiomech.2024.112044.

116. Wang J, Xu W, Wu Z, Zhang H, Wang B, Zhou Z, et al. 基于智能手机的无标记系统测量膝骨关节炎患者下肢运动学的评估. *J Biomech*. 2025;181:112529. https://doi.org/10.1016/j.jbiomech.2025.112529.

117. Bousigues S, Naaim A, Robert T, Muller A, Dumas R. 无标记不一致性的影响至少与基于标记的软组织伪影的影响一样大. *J Biomech*. 2025;182:112566. https://doi.org/10.1016/j.jbiomech.2025.112566.

118. Templin T, Riehm CD, Eliason T, Hulburt TC, Kwak ST, Medjaouri O, et al. 在大样本队列中使用三维无标记运动捕捉评估下落垂直跳跃的运动学与动力学. *Front Bioeng Biotechnol*. 2024;12. https://doi.org/10.3389/fbioe.2024.1426677.

119. Ho MY, Kuo MC, Chen CS, Wu RM, Chuang CC, Shih CS, et al. 基于半监督学习赋能的开源云平台PathoOpenGait的病理性步态分析. *IEEE J Biomedical Health Inf*. 2024;28(2):1066–77. https://doi.org/10.1109/jbhi.2023.3340716.

120. Chen L, Zheng Y, Gong Z, Wang D. 基于视频三维姿态估计的步态异常检测. *Med Biol Eng Comput*. 2025;63(9):2651–63. https://doi.org/10.1007/s11517-025-03339-5.

121. Fukushima T, Blauberger P, Guedes Russomanno T, Lames M. 人体姿态估计在体育运动捕捉中的潜力：一项验证研究. *Sports Eng*. 2024;27(1). https://doi.org/10.1007/s12283-024-00460-w.

122. Svetek A, Morgan K, Burland J, Glaviano NR. OpenCap在功能性任务中下肢运动学的验证. *J Biomech*. 2025;183:112602. https://doi.org/10.1016/j.jbiomech.2025.112602.

123. Hu B, Wang J, Xu W, Li T, Nie Y, Li K. 用于骨关节炎精确下肢运动学分析的双摄像头无标记运动捕捉系统. *Ann Biomed Eng*. 2025;53(11):2949–65. https://doi.org/10.1007/s10439-025-03859-z.

124. Vafadar S, Skalli W, Bonnet-Lebrun A, Assi A, Gajny L. Assessment of a novel deep learning-based marker-less motion capture system for gait study. *Gait Posture*. 2022;94:138–43. https://doi.org/10.1016/j.gaitpost.2022.03.008.

125. Yeom S, Jeong B, Lee J, Hyung SK, Kim H, Chang E, et al. Validation of Markerless Motion Capture System (OpenCap) for Overhead Squat Kinematics. *Meas Phys Educ Exerc Sci*. 2025;30(1):49–62. https://doi.org/10.1080/1091367x.2025.2512361.

126. Todesca D, Fabiocchi D, Farías Fuentes J, Giulietti N, Carnevale M, Giberti H. Comparison of Real-Time Marker-Less and Optoelectronic 3D Human Pose Estimation Systems for Cyclist Pose Analysis. 2025 IEEE International Conference on Metrology for eXtended Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE). 2025:417–422. https://doi.org/10.1109/metroxraine66377.2025.11340187

127. Zhang C, Li Y, Ye W, Huang G. Human Kinematics Analysis by Markerless Vision Based on OpenSim. 2024 IEEE 5th International Conference on Pattern Recognition and Machine Learning (PRML). 2024:443–450. https://doi.org/10.1109/prml62565.2024.10779779

128. Liang Y, Qi S, Xu T, Hu Y, 3D Gait Analysis for the Elderly Mobility Based on Multiple RGB Cameras. 2023 29th International Conference on Mechatronics and Machine Vision in Practice (M2VIP). 2023;:1–5. https://doi.org/10.1109/m2vip58386.2023.10413409

129. Zhang L, Sidarta A, Wu TL, Jatesiktat P, Wang H, Li L, et al. Towards Clinical Application of Enhanced Timed Up and Go with Markerless Motion Capture and Machine Learning for Balance and Gait Assessment. *IEEE J Biomedical Health Inf*. 2025;1–9. https://doi.org/10.1109/JBHI.2025.3543095.

130. Shah EEZ, Mustansar Z, Biomechanics of Knee Kinematics During Sit-Ups Using MoCap Technology. 2025 27th International Multitopic Conference (INMIC). 2025;:1–4. https://doi.org/10.1109/inmic65900.2025.11348465

131. McGinley JL, Baker R, Wolfe R, Morris ME. The reliability of three-dimensional kinematic gait measurements: A systematic review. *Gait Posture*. 2009;29(3):360–9. https://doi.org/10.1016/j.gaitpost.2008.09.003.

132. Soesilo TH, Gunaratne PN, Tamura H. Development of a Real-Time Multi-Person 3D Keypoint Detection System Using Stereoscopic Cameras and RTMPose. In: Proceedings of the 2025 International Conference on Artificial Life and Robotics (ICAROB2025). Oita, Japan: ALife Robotics; 2025. pp. 593–596.

133. Wang Y, Wang Z, Liu L, Daniilidis K. TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos. *arXiv:2403.17346 [cs.CV]*. 2024.

134. Bermuth D, Poeppel A, Reif W, RapidPoseTriangulation. Multi-view Multi-person Whole-body Human Pose Triangulation in a Millisecond. *arXiv:2503.21692 [cs.CV]*. 2025.

135. Janampa S, Pattichis M, DETRPose. Real-time end-to-end transformer model for multi-person pose estimation. *arXiv:2506.13027 [cs.CV]*. 2025.

136. Mehraban S, Iaboni A, Taati B. FastHMR: Accelerating Human Mesh Recovery via Token and Layer Merging with Diffusion Decoding. arXiv:2510.10868 [cs.CV]. 2025.

137. Cho H, Choi G, Choi JAJAHR. Amputated Joint Aware 3D Human Mesh Recovery. arXiv:2509.19939 [cs.CV]. 2025.

138. Zheng Z, Yang L, Pan J, Zhu H. Mamba-Driven Topology Fusion for Monocular 3D Human Pose Estimation. arXiv:2505.20611 [cs.CV]. 2025.

139. Bright J, Chen Y, Zelek JS. DreamPose3D: Hallucinative Diffusion with Prompt Learning for 3D Human Pose Estimation. arXiv:2511.09502 [cs.CV]. 2025.

140. Han B, Huang Y, Gao P, HyperDiff. Hypergraph Guided Diffusion Model for 3D Human Pose Estimation. arXiv:2508.14431 [cs.CV]. 2025.

141. Armstrong K, Rodrigues A, Willmott AP, Zhang L, Ye X. Validation of Human Pose Estimation and Human Mesh Recovery for Extracting Clinically Relevant Motion Data from Videos. arXiv:2503.14760 [cs.CV]. 2025.

142. Pemasiri A, Goan E, Lichtwark G, Schuster R, Kelly L, Fookes C. Biomechanically Accurate Gait Analysis: A 3D Human Reconstruction Framework for Markerless Estimation of Gait Parameters. arXiv:2603.02499 [cs.CV]. 2026.

143. Guo C, L'Erario G, Romualdi G, Leonori M, Lorenzini M, Ajoudani A 等. Physics-Informed Learning for Human Whole-Body Kinematics Prediction via Sparse IMUs. arXiv:2509.25704 [cs.CV]. 2025.

144. Kinfu KA, Vidal R. Efficient Vision Transformer for Human Pose Estimation via Patch Selection. arXiv:2306.04225 [cs.CV]. 2023.

145. Fang H, Cai J, Wang X, Yang W. Beyond Static Frames: Temporal Aggregate-and-Restore Vision Transformer for Human Pose Estimation. arXiv:2603.05929 [cs.CV]. 2026.

146. Yang R, Kennedy A, Cotton RJ. BiomechGPT: Towards a Biomechanically Fluent Multimodal Foundation Model for Clinically Relevant Motion Tasks. arXiv:2505.18465 [cs.CV]. 2025.

147. Cotton RJ, Leonard T, BiomechAgent:. AI-Assisted Biomechanical Analysis Through Code-Generating Agents. arXiv:2602.06975 [cs.CV]. 2026.

148. Katsu J 等. GRF-MV: Ground Reaction Force Estimation from Monocular Video. 见：BMVC 2024 Workshop (ANIMA). 2024.

149. Han X, Senderling B, To S, Kumar D, Whiting E, Saito J. GroundLink: A Dataset Unifying Human Body Movement and Ground Reaction Dynamics. arXiv:2310.03930 [cs.CV]. 2023.

150. Hossain MSB, Choi H, Guo Z, Yoo S, Song MK, Shin H, 等. Knowledge transfer-driven estimation of knee moments and ground reaction forces from smartphone videos via temporal-spatial modeling of augmented joint kinematics. PLoS ONE. 2025;20(11):e0335257. https://doi.org/10.1371/journal.pone.0335257.

151. Jiang T, Lu P, Zhang L, Ma N, Han R, Lyu C 等. RTMPose: real-time multi-person pose estimation based on MMPose. arXiv:2303.07399. 2023.

152. Xu Y, Zhang J, Zhang Q, Tao D. ViTPose: simple vision transformer baselines for human pose estimation. Adv Neural Inf Process Syst (NeurIPS). 2022. arXiv:2204.12484.

153. Khirodkar R, Bagautdinov T, Martinez J, Zhaoen S, James A, Selednik P 等. Sapiens: Foundation for Human Vision Models. arXiv:2408.12569 [cs.CV]. 2024.

154. Huang Y, Liu J, Xian K, Qiu RC, PoseMamba. Monocular 3D Human Pose Estimation with Bidirectional Global-Local Spatio-Temporal State Space Model. arXiv:2408.03540 [cs.CV]. 2024.

155. Yang X, Kukreja D, Pinkus D, Sagar A, Fan T, Park J 等. SAM 3D Body: Robust Full-Body Human Mesh Recovery. arXiv:2602.15989 [cs.CV]. 2026.

156. Shin S, Kim J, Halilaj E, Black MJ. WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion. arXiv:2312.07531 [cs.CV]. 2023.

157. Shen Z, Pi H, Xia Y, Cen Z, Peng S, Hu Z 等. World-Grounded Human Motion Recovery via Gravity-View Coordinates. arXiv:2409.06662 [cs.CV]. 2024.

158. Sárándi I, Pons-Moll G. Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation. arXiv:2407.07532 [cs.CV]. 2024.

159. Gozlan Y, Falisse A, Uhlrich S, Gatti A, Black M, Chaudhari A. OpenCapBench: A Benchmark to Bridge Pose Estimation and Biomechanics. arXiv:2406.09788 [cs.CV]. 2024.

160. Unger T, Sal Moslehian A, Peiffer JD, Ullrich J, Gassert R, Lambercy O 等. Differentiable Biomechanics for Markerless Motion Capture in Upper Limb Stroke Rehabilitation: A Comparison with Optical Motion Capture. arXiv:2411.14992 [cs.CV]. 2024.

161. Voss H, Kopp S. JAX-IK: Real-Time Inverse Kinematics for Generating Multi-Constrained Movements of Virtual Human Characters. arXiv:2507.00792 [cs.CV]. 2025.

162. Tolpin D, Kagarlitsky S. Fast Neural Inverse Kinematics on Human Body Motions. arXiv:2506.17996 [cs.CV]. 2025.

163. Tan T, Van Wouwe T, Werling K, Liu CK, Delp S, Hicks J 等. GaitDynamics: a generative foundation model for analyzing human walking and running. Nat Biomed Eng. 2026. https://doi.org/10.1038/s41551-025-01565-8.

164. Wehrbein T, Rudolph M, Rosenhahn B, Wandt B. Probabilistic Monocular 3D Human Pose Estimation with Normalizing Flows. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2021. arXiv:2107.13788 [cs.CV].

165. Fritz S, Lusardi M. 白皮书：步行速度：第六生命体征. J Geriatr Phys Ther. 2009;32(2):2–5. https://doi.org/10.1519/00139143-200932020-00002.

166. Yang T, He S, Jing H, Yang J, Liu Z, Zou C 等. Fast SAM 3D Body: Accelerating SAM 3D Body for Real-Time Full-Body Human Mesh Recovery. arXiv:2603.15603 [cs.CV]. 2026.

167. Elbatel M, Wang A, Liu K, Mouheb K, Almar-Munoz E, Lin L 等. MedSapiens: Taking a Pose to Rethink Medical Imaging Landmark Detection. arXiv:2511.04255 [cs.CV]. 2025.

168. Werling K, Bianco NA, Raitor M, Stingel J, Hicks JL, Collins SH 等. AddBiomechanics: Automating model scaling, inverse kinematics, and inverse dynamics from human motion data through sequential optimization. PLoS ONE. 2023;18(11):e0295152. https://doi.org/10.1371/journal.pone.0295152.

169. Cuevas-Velasquez H, Yiannakidis A, Shin S, Becherini G, Höschle M, Tesch J et al. MAMMA: Markerless & Automatic Multi-Person Motion Action Capture. arXiv:2506.13040 [cs.CV]. 2025.

170. Peiffer JD, Shah K, Djuraskovic I, Anarwala S, Abdou K, Patel R et al. Portable Biomechanics Laboratory: Clinically Accessible Movement Analysis from a Handheld Smartphone. arXiv:2507.08268 [cs.CV]. 2025.

171. Li J, Bian S, Xu C, Chen Z, Yang L, Lu C. HybrIK-X: Hybrid Analytical-Neural Inverse Kinematics for Whole-body Mesh Recovery. arXiv:2304.05690 [cs.CV]. 2023.

172. Wang Y, Sun Y, Patel P, Daniilidis K, Black MJ, Kocabas M. PromptHMR: Promptable Human Mesh Recovery. arXiv:2504.06397 [cs.CV]. 2025.

173. Fiche G, Weinzaepfel P, Bregier R, Baradel F. Multi-HMR 2: Multi-Person Camera-Centric Human Detection, Mesh Recovery and Tracking. arXiv:2606.14841 [cs.CV]. 2026.

174. Wang R, Xu S, Dong Y, Deng Y, Xiang J, Lv Z et al. MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details. arXiv:2507.02546 [cs.CV]. 2025.

175. Piccinelli L, Sakaridis C, Yang Y-H, Segu M, Li S, Abbeloos W, Van Gool L. UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler. arXiv:2502.20110 [cs.CV]. 2025.

176. Lee SE, Nishino K, Nobuhara S. SteerPose: Simultaneous Extrinsic Camera Calibration and Matching from Articulation. arXiv:2506.01691 [cs.CV]. 2025.

177. Patel P, Black MJ. CameraHMR: Aligning People with Perspective. arXiv:2411.08128 [cs.CV]. 2024.

## 图表

![](images/92772517913bc5857dc42ed64bee35c7da6d67643d215f840b99ffde2e9274c2.jpg)  
图 1

研究识别、筛选与纳入的 PRISMA-ScR（系统综述与元分析优先报告条目——范围综述扩展版）流程图。

![](images/b0fbfdc2940c26b419caac2b39cad18ad178d193e1586dae13ca01677d0c1f7f.jpg)

![](images/12c657efc86715af00587933799b94a530a710761164c4f2795db4d33aa092c7.jpg)

![](images/4ee76b41b7654965fb5da142490a5d767fd68ab1cd73d3b9eedad6a3af414545.jpg)

![](images/2d887382b2149676424145ec1ac6366e3d936b0f30e52dc8a28c572e69e93179.jpg)

![](images/2e174798d6b3ddca3e381b24f2d05d87cef213b159f4fb3dcd3565e3abc50a4d.jpg)

![](images/400a2d1676feabda72ea0c52dcee10719e4406320fb8074b42c3e71aaf3e36d8.jpg)  
图 2  
纳入语料库（n = 117）的概览：$\widehat { \mathbf { \xi } } _ { \mathbf { \alpha } }$ 出版年份；$\mathbf { { b } , }$ 研究人群；$\pmb { c } ,$ 运动任务；${ \pmb d } ,$ 样本量；e，金标准参考方法；$\pmb { f } ,$ 采集环境。

![](images/e9c7ad08591730da8d9447446d429f0f01b66255b2994866ac9be955ab408abc.jpg)  
图 3  
五种无标记运动捕捉架构及其共享的流水线阶段。

![](images/53bb425e6b4604d79afef0c77a53f5e0af1fcdf5e0b8963ee35d04a20d5793cf.jpg)

图 4

按核心架构划分的矢状面下肢关节误差。

![](images/2cad31844594bd1fc36b07810d1900f7f03f8a4893f6fac8a08b69589e466b71.jpg)  
图 5

Proportion of studies within the $5 ^ { \circ }$ error level (McGinley acceptable-to-reasonable range), by architecture, joint and plane. Each cell is computed over the studies contributing that architecture, joint and plane.

![](images/463de37823a1aca5b6d2380609f1bc4ed7bdb9a9824e56056f43829e3e3aee56.jpg)  
Figure 6

各架构、关节和平面下，误差在 $5 ^ { \circ }$ 以内（McGinley 可接受至合理区间）的研究占比。每个单元格基于同时贡献该架构、关节和平面的研究计算得出。

精选 MICRO 池在 17 个流水线步骤模块中的定性活跃度。阴影表示在非系统性精选池内的代表性相对水平（低、中、高）；它反映筛选与整理工作的投入，而非领域规模或生物力学重要性，不应被解读为可量化的度量。

## Input

## MoGe-2

## UniDepth v2

## SteerPose

## 2D pose

## RTMPose

## ViTPose

## Sapiens

## 3D recovery

## Lifting

## Mesh / SMPL

## Whole-body IK

## Multi-person

## Multi-view mesh

## SynthPose

## Biomech. bridge

## AddBiomechanics

## Differentiable IK

## Fast Neural IK

## Multi-view classic

## Output

## GaitDynamics

## Wehrbein

## BiomechGPT

## Figure 7

前沿工具箱：每个流水线阶段的候选升级方案（每个阶段选择一个代表性方案，并非穷举）。集成流水线（BioPose、OpenCap Monocular、Portable Biomechanics Laboratory）由若干上述模块组合而成，将在正文中讨论。

proposed framework (Section 5) 的设计空间
<table><tr><td></td><td>2D pose</td><td>Marker aug.</td><td>IK/ MSK</td><td>Mesh recovery</td><td>Diff./ neuraÍ IK</td><td>Kinetics (video)</td><td>Foundation</td></tr>
<tr><td>2D Monocular</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>Triangulation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>Lifting</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>Body Mesh (SMPL)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>End-to-End</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td colspan="8">established rare / emerging</td></tr>
</table>

Figure 8  
已验证架构（MACRO）与新兴构件（MICRO）之间的映射关系。

![](images/1a6a3929c1e5decb0cbeac13ef7755137b6c17b19324eb2aff0638bf364c86cc.jpg)  
Figure 9

面向临床可部署的标记点无标记生物力学研究议程，以候选流水线形式展示。单目与多相机路径共用同一参数化网格引擎，并重新汇合为单一网格，通过虚拟标记点映射到临床解剖结构，再由受试者特异的逆运动学求解。输出为具有逐自由度不确定度的关节运动学结果，并在多相机设置下加上测力台的动力学数据。

## Supplementary Files

This is a list of supplementary files associated with this preprint. Click to download.

Additionalfile1PRISMAScRchecklist.docx

Additionalfile2Searchstrategy.docx

Additionalfile7MICROmethods.docx

Additionalfile3Extractiontable.xlsx

Additionalfile4medianerrorbyarchitecture.png

您只提供了两个文件名：

- `Additionalfile6MICROblocks.png`
- `Additionalfile5MICROcorpus.xlsx`

这两个看起来是补充材料文件的引用，但并没有实际的英文 Markdown 正文内容可供翻译。请您提供需要翻译的英文 Markdown 文本内容，我会按照规则为您准确翻译。