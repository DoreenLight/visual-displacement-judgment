# 用于偏瘫的单侧移动辅助外骨骼（SMAEXO）的设计与评估

Xiaoyue Yan<sup>1,</sup> <sup>2</sup>, Wei Yun Yau<sup>3</sup>, Isaac Okumura Tan<sup>2</sup>, Longbin Zhang<sup>2,</sup> <sup>4</sup>, Youlong Wang<sup>1,</sup> <sup>2</sup>, Oliver Roberts<sup>2</sup>, Seng Kwee Wee<sup>5</sup>, Wei Tech Ang<sup>1,</sup> <sup>2</sup>

摘要— 本研究提出了单侧移动辅助外骨骼（Single-Sided Mobility Assistive Exoskeleton, SMAEXO），这是一种新型单侧多关节设备，旨在增强偏瘫患者的移动能力与康复效果。SMAEXO 在髋关节和膝关节提供主动助力，并结合用于防止足下垂的被动柔性踝关节，以确保在整个步态周期中提供全面支撑。其创新设计优化了反作用力支撑和重量分布，从而促进平衡并减轻患侧的负担。此外，其模块化、可定制的设计能够适应不同的身体尺寸和生物力学需求，而自锁机构、缓冲垫、防撞块和弹性元件等安全特性则提升了用户的安全性与舒适度。通过对八名健全受试者（包括模拟中风步态模式的物理治疗师）的试验，评估了 SMAEXO 的可用性、舒适性、安全性和助力效果。受试者表示该设备舒适、安全且有效，并突出了其在平地行走、坡道行走以及越障过程中的优势。值得注意的反馈包括需要增强透气性、减轻设备重量以及提高穿脱便利性。结果表明，SMAEXO 具有为偏瘫患者提供全面多关节助力的潜力。这项工作有助于推进中风后患者的康复与辅助技术发展，从而实现更有效、更个性化的移动支持。

## I. 引言

中风作为致残的主要原因之一，通常会导致偏瘫，影响了 65% 伴有步态受损的幸存者 [1]。常见问题包括不对称性、平衡能力差和足下垂 [2]，需要进行强化康复训练以恢复移动能力。传统疗法属于劳动密集型，每次治疗需要两名物理治疗师，限制了其可推广性 [3]。外骨骼提供了一种更高效的解决方案，能够提供持续、自动化的支持以增强康复效果，同时减轻治疗师的工作负担 [4]。

大多数下肢外骨骼是双侧的，通过针对髋关节、膝关节和踝关节来提供全面的移动支持。诸如 Ekso Indego [5]、ReWalk [6] 和 Rex [7] 等设备可辅助具有严重运动功能丧失的用户（如脊髓损伤患者）。然而，它们笨重的设计、高昂的成本以及对位置控制的依赖，使其不适合需要响应迅速、量身定制化支持的中风幸存者 [8]。

对于偏瘫患者而言，单侧外骨骼提供了一种更简单、更轻便的替代方案。与限制自然运动并破坏步态对称性 [9] 的双侧设备不同，单侧外骨骼仅辅助患肢，在减轻佩戴者负担的同时提高了行动能力。

然而，单侧方案也带来了自身的挑战。此类设备的不对称结构可能会增加维持平衡的难度，且有限的反作用力支撑限制了产生足够力矩以充分稳定和推进患肢的能力。

此外，许多单侧外骨骼仅辅助单个关节，这限制了其在中风引起的多关节损伤中的有效性。诸如密歇根大学（University of Michigan）的无源外骨骼 [10]、纽约大学（University of New York）的有源外骨骼 [11] 以及 REFLEX 系统 [12] 等膝关节设备，可辅助膝关节以实现站立支撑和摆动期足底离地。类似地，ReStore Exo-Suit [13] 和北亚利桑那大学（Northern Arizona University）的设备 [14] 等踝关节外骨骼改善了跖屈和背屈，但仅提供部分支持，忽略了其他关节的损伤。HAL [15]、AGoRA [16] 和 LitteStepper [17] 等多关节单侧设备尝试同时辅助多个关节，但往往更重、用户友好性较差，且更难以与用户的运动意图无缝集成。

在实际应用中，现有的单侧外骨骼在保持易用性、舒适度和安全性的同时，仍难以提供无缝的多关节辅助。这些局限性突显了对新一代单侧外骨骼的需求，以期为偏瘫患者提供全面的、以用户为中心的支持。

为弥补这些不足，本研究提出了单侧移动辅助外骨骼（Single-Sided Mobility Assistive Exoskeleton, SMAEXO），这是一种专为偏瘫患者设计的新型多关节设备。SMAEXO 融入了旨在优化反作用力管理和重量分布的创新特性，同时确保对髋关节、膝关节和踝关节的辅助兼具有效性与直观性。本工作的主要贡献包括：

1) 研发了 SMAEXO，实现了对髋关节、膝关节和踝关节的一体化辅助，为患肢提供全面支持。优化的重量分布和反作用力管理策略确保该设备能够有效稳定并支撑用户，且不会牺牲舒适性、安全性或运动自如度。

2) 通过一项包含 8 名受试者的用户研究验证了 SMAEXO 的性能。该评估聚焦于易用性、舒适度、安全性和有效性，为 SMAEXO 的实用性及其促进脑卒中康复进程的潜力提供了有价值的见解。

![](images/0a5eefbc686948efd1a88322ce1e441c2d1b3d7402d1a2592ee1676b9b28e6a7.jpg)  
Fig. 1. 单侧行动辅助外骨骼（SMAEXO）的架构。红色箭头指示关节运动。

通过提供满足偏瘫患者独特需求的精心设计的解决方案，SMAEXO 有望改变中风康复领域。

## II. 系统概述

## A. SMAEXO 的架构

SMAEXO 是通过与专门从事中风康复和下肢外骨骼的物理治疗师密切合作开发的。这一咨询过程确保了该系统符合偏瘫患者的独特需求，强调了可用性、安全性、舒适性和全面辅助。该设备采用模块化设计，由四个主要部件组成（Fig. 1）：

• 患侧模块：包含由准直接驱动（QDD）电机驱动的主动髋关节和膝关节，该电机将大扭矩电机与低传动比传动系统（1:6）相结合，以提供高带宽和反向驱动能力。此外，还配备了一个与脚底板配合的被动踝关节，以防止足下垂。

• 健侧模块：具有被动髋关节，有助于髋关节的屈曲/伸展和外展/内收，确保健侧的自然运动。

• 背部模块：连接所有模块并容纳控制组件和电池，策略性地将电池放置在健侧，以优化重量分布和平衡。

• 背心吊带：将设备重量和反作用力均匀分布在用户的核心躯干上，提高舒适度和稳定性。

为了安全和舒适，所有关节都结合了硬限位器和软限位器，将关节活动度（ROM）限制在用户的生物力学能力范围内。作为关节缓冲器实现的软限位器，在关节接近硬限位器时吸收冲击（Fig. 2 (b)）。关节缓冲器可以轻松更换，以适应 ROM 受限的用户。此外，所有腿杆均配备了绑带和衬垫，如 Fig. 2 (d) 所示，这些绑带可以轻松打开以便快速穿脱，并且绑带的长半轴和短半轴均可轻松调节，以适应不同腿围的用户。

为了进一步提高适应性和对齐性，SMAEXO 具有相应的调节机构，允许免工具调节腰宽、大腿和小腿杆长度以及脚部尺寸，以适应各种体型尺寸的用户（Fig. 2 和 Fig. 3 (a)）。所有调节机构均配备复位弹簧，以确保处于锁定状态以保证安全。

SMAEXO 系统的主要规格总结在 Table I 中。

TABLE I  
SMAEXO 的规格。

<table><tr><td>项目</td><td>规格</td></tr><tr><td>运动范围 (°)</td><td>髋关节: (-20, 125); 膝关节: (0, 135); 踝关节: (-40, 24)</td></tr><tr><td>长度范围 (mm)</td><td>大腿段: 400–500 (长), 300–400 (短); 小腿段: 380–480 (长), 280–380 (短); 髋宽: 335–435; 足部: 250–305</td></tr><tr><td>髋关节和膝关节电机</td><td>额定扭矩: 12Nm; 最大扭矩: 37Nm; 减速比: 1:6; 重量: 0.759kg</td></tr><tr><td>电池</td><td>锂电池组, 24V 6Ah, 重量: 0.82kg</td></tr><tr><td>重量 (kg)</td><td>总计: 7.5; 患侧模块: 3.3; 健侧模块: 0.3 背部模块: 2.1 背心背带: 1.8</td></tr></table>

B. 高效的反作用力管理与重量分布

有效的反作用力管理是最大程度减少不适感并提高可用性的关键。双侧外骨骼（如 Ekso Indego [5]、ReWalk [6] 和 Rex [7]）通过较重的自重和广泛的用户接触面积来实现这一点，从而能够提供更高的助力扭矩，但这也增加了体积并降低了活动能力。相比之下，轻量化外骨骼由于反作用力支撑较弱而面临扭矩限制。例如，Honda 的 SMA 设备提供 4 Nm 的髋关节扭矩 [18]，而 Samsung 的 GEMS 通过髋部支架提供额外支撑，扭矩可达 12 Nm [19]。像 AGoRA 这样的单侧设备通过双侧配置提高了扭矩和平衡性，但质量增加到了 17 kg [16]。

SMAEXO 通过集成用于高效反作用力管理和重量分布的创新特性来应对这些挑战。通过将符合人体工学的背心背带与连接有脚踏板的柔性踝关节相结合，SMAEXO 在保持轻量化和用户友好特性的同时，确保了有效的稳定性与全面的助力。

## • 符合人体工学的背心背带

如图 3 (a) 所示，SMAEXO 背心背带旨在紧密贴合穿戴者的身体。它由集成有背带和腰带的背心组成，具有由弹簧钢等柔性材料制成的弹性脊柱结构，可自然适应穿戴者的脊柱曲率。为了提高贴合度和接触效果，其中还包含一个可调节的气囊凸起，从而能够兼容多种体型并提升整体的用户舒适度。该配置有效地利用了穿戴者的腰部、胸部、腹部和背部肌肉来稳定反作用力，在减少不适感的同时提升了活动能力。

![](images/9f8fdcb46f5558bfcc416a109580f376c795ab5a5bcf1af8020f20b3d7fad868.jpg)  
Fig. 2. SMAEXO 的患侧模块。(a) 具有长度调节功能的模块（红色箭头）。(b) 具有受限运动范围（ROM）的关节，配备可更换的缓冲垫以吸收冲击。(c) 带有复位弹簧以实现可靠锁定的长度调节机构。(d) 具有线性（a, b）和旋转调节功能的绑带调节机构，用于适应腿围并便于穿戴（红色箭头）。

![](images/6730c8f96f6d73b358668066ad8630e50d49fe24a858684cd73b5f60081fc4be.jpg)  
Fig. 3. SMAEXO 背心背带、背部和健侧模块。(a) 背部模块与背心背带相连，包含一个用于监测躯干角度的 IMU。(b) 健侧模块配备一个带有编码器的被动髋关节，用于测量矢状面髋关节角度。红色箭头表示腰宽和髋部运动的线性调节。

![](images/3aba59067cd58469b8121e1c01ac911d7606669cecfa4df031b5cd4c81eeb896.jpg)  
Fig. 4. SMAEXO 的踝关节和足底板。(a) 踝关节通过输出杆和弹性防足下垂带与足底板连接，该弹性防足下垂带的作用类似于张力可调的弹簧，以适应不同用户。(b) 足底板通过前鞋底 I 在后跟鞋底 I 上的滑动来实现长度调节，以适应各种足部尺寸。前鞋底 I 和后跟鞋底 I 中嵌入的力传感器受到额外保护层（前鞋底 II 和后跟鞋底 II）的保护。集成的线缆通道便于走线。

## • 柔性踝关节

与大多数外骨骼（如膝踝足矫形器 Knee-Ankle-Foot Orthoses、ReWalk™ [6] 和 HAL [15]）采用固定踝关节和刚性足底板来防止足下垂并将重量传递至地面不同，SMAEXO 采用了兼顾舒适性和自然步态的多功能踝关节设计。这一创新设计可根据用户的个性化需求进行调整，从而有助于提升活动能力。

如图 Fig. 4 和 Fig. 5 所示，SMAEXO 踝关节包含以下特性：(1) 角度可调机构，既允许在受控范围内旋转，也可以固定在特定角度。(2) 柔性足底板，在足跟和前掌处配有用于检测步态事件的力传感器。(3) 在站立相期间有效地将重量转移至地面，同时不影响传感器的功能。(4) 与弹性绑带和踝部套环集成，以防止摆动相期间发生足下垂。(5) 足底板长度可调，以适应不同的足部尺寸。(6) 采用聚氨酯（PU）、橡胶和乙烯-醋酸乙烯共聚物（EVA）等高弹耐用材料制造，以实现自然的地面接触和耐久性。该构型有助于实现自然步态，同时提供个性化调节以匹配个人的步行模式和助力需求，从而提升舒适性与活动能力。

得益于在反作用力管理和重量支撑方面的进步，SMAEXO 实现了 25 Nm 的最大助力力矩。该能力在确保用户舒适度的同时，满足了偏瘫康复对大扭矩的需求，为行动助力提供了一种有效且人性化的解决方案。

![](images/67b02160e6b473fd1bb6dc584f9cdbf49d128fab1e387a56549e90a4a737f3d8.jpg)  
Fig. 5. 踝关节角度可调机构。(a) 锁定在受限运动状态。(b) 固定在特定角度。(c) 爆炸图。该机构利用锁销外壳、滑动导槽、回转块、弹簧限位块、枢轴和锁销盖，实现受控的踝关节旋转或固定定位。限位块沿导槽 II 旋转，并在弹簧锁销卡入锁销孔时锁定。解锁时释放锁销，恢复运动。回转块在导槽 I 内旋转，并在与两个限位块接触时将踝关节锁定在设定角度。

![](images/85deeccbbc7318d6888007633e3469a0695ebb109c3cf793291bbe9dbdbf258d.jpg)  
Fig. 6. SMAEXO 控制框图。  
Fig. 7. 实验装置。

## C. 控制方案

SMAEXO 采用多级控制策略，并配备可靠且具成本效益的传感器系统以增强其功能。该系统结合了躯干 IMU（惯性测量单元）以及嵌入在髋关节和膝关节中的编码器，以提供精确的关节运动学测量。来自力传感器的足底压力测量数据与运动学数据相结合，用于识别各种日常活动中的步态相位，包括从坐到站的转换以及在平地、斜坡或楼梯上的行走。控制系统利用该步态相位信息实现外骨骼透明性（例如重力补偿）和部分用户行动辅助，从而实现优化支撑并提供更好的用户体验。此外，外骨骼和用户参数可根据个体辅助需求进行定制，而指数加权移动平均（EWMA）等滤波器可确保整个步态周期内力矩曲线的平滑（Fig. 6）。

## III. 实验

为评估 SMAEXO 提供的可用性、舒适度、安全性和辅助效果，共招募了八名健全受试者，其中包括两名专攻下肢康复的物理治疗师。在测试过程中，物理治疗师根据其经验模拟了中风步态模式——包括足下垂、膝关节僵直、划圈步态和膝关节无力——以从患者的角度评估外骨骼的有效性。受试者的人口统计学和人体测量学详细信息见 Table II。所有受试者在了解研究目标和潜在风险后均签署了书面知情同意书。该研究方案获得了 Nanyang Technological University Institutional Review Board（南洋理工大学机构审查委员会）的批准和监督。

实验期间，受试者可随时休息、停止或退出测试。评估包括两个阶段（穿戴外骨骼和不穿戴外骨骼），每个阶段包含三项任务。受试者对每项任务至少重复三次：

• 10 米平地行走；

• 坡度为 8 度的 4 米斜坡上下行走；

• 跨越 10 cm 高、12.5 cm 宽的障碍物。

在第二阶段中，参与者在测试前对外骨骼进行了 30 分钟的熟悉训练。研究人员指导他们使用外骨骼并确保其充分适应。参与者被告知如有任何不适或疼痛应及时报告，相关情况均被记录并处理。

在研究结束时，参与者完成了一份问卷，用于评估外骨骼的可用性、舒适度、安全性和助力效果。所有问题的评分范围均为 1 到 5 分，其中 1 代表“强烈反对”，5 代表“强烈赞同”。

![](images/5d4b284a5b6d62fd947aed07ab3197a578ee1ee8c0e859084d316b849eb33982.jpg)  
图 8. 用户在平地行走、坡道行走以及跨越障碍任务中对 SMAEXO 的问卷反馈。

表 II  
参与者的人口统计学与人体测量学信息。

<table><tr><td>参与者</td><td>年龄</td><td>性别</td><td>身高 (cm)</td><td>体重 (kg)</td></tr><tr><td>P1</td><td>31-40</td><td>女</td><td>155</td><td>54.5</td></tr><tr><td>P2</td><td>21-30</td><td>男</td><td>171</td><td>70</td></tr><tr><td>P3</td><td>31-40</td><td>女</td><td>163</td><td>55</td></tr><tr><td>P4</td><td>31-40</td><td>男</td><td>183</td><td>76</td></tr><tr><td>P5</td><td>21-30</td><td>男</td><td>183</td><td>73</td></tr><tr><td>P6</td><td>21-30</td><td>男</td><td>181</td><td>85</td></tr><tr><td>P7</td><td>21-30</td><td>男</td><td>176</td><td>85</td></tr><tr><td>P8</td><td>41-50</td><td>男</td><td>169</td><td>71</td></tr></table>

表 III

SMAEXO 的规格参数。

<table><tr><td>设备</td><td>HAL (Cyberdyne)</td><td>LiteStepper (Angelexo)</td><td>SMAEXO</td></tr><tr><td>控制策略</td><td>基于模型的控制（放大肌肉活动）或预定义步态轨迹</td><td>复制健侧肢体步态轨迹 &amp; 拐杖手柄启动行走</td><td>混合基于模型的控制</td></tr><tr><td>执行器</td><td>2 台无刷直流电机</td><td>2 台无刷直流电机</td><td>2 台 QDD 电机</td></tr><tr><td>所用传感器</td><td>运动学传感器、足底压力传感器、肌电传感器（EMG）</td><td>运动学传感器、足底压力传感器、手杖压力传感器</td><td>运动学传感器 (4)、足底压力传感器</td></tr><tr><td>足板类型</td><td>带有刚性鞋垫的定制鞋</td><td>机械足板</td><td>柔性足板</td></tr><tr><td>重量支撑</td><td>刚性鞋垫（地面）</td><td>背心（核心） 足板（地面）</td><td>背心式挽具（核心） 足板（地面）</td></tr><tr><td>重量</td><td>9 kg</td><td>16.9 kg</td><td>7.5 kg</td></tr></table>

## IV. 结果与讨论

如图 8 所示，参与者认可 SMAEXO 的模块化设计和免工具调节功能，这便于穿脱。然而，其不平衡的结构使得保持中立姿态具有挑战性，尤其是在坐姿时。物理治疗师指出，脑卒中后患者通常需要两名助手协助以保持稳定。引入支撑架可以在穿戴过程中提供稳定性，从而解决这一问题。

参与者认为该设备的重量是可以接受的。在支撑相期间，SMAEXO 有效地将重量传递至地面，减轻了膝关节的负担。然而，物理治疗师对偏瘫用户表达了担忧，因为在摆动相期间，健侧腿需要承受设备的重量。这可能会增加健侧膝关节的受力，对于更容易出现膝关节问题（如骨关节炎）的中风后老年患者而言尤其令人担忧。建议进一步减轻外骨骼的重量以缓解这一负担。

据报告，该设备佩戴舒适，未出现疼痛、瘙痒或刺激等情况。可调节机构使其能够良好地适应不同的身体尺寸，确保紧密贴合，从而有效调动用户的核心肌群，实现最佳的反作用力管理与重量分配。然而，有两名参与者在佩戴该设备超过一小时后背部出现轻微出汗，这可能是由于背心和气囊凸起所致。在这些部件中采用透气材料可以提升长时间使用时的舒适度。

在安全性方面，参与者表示在试验期间无需担忧跌倒或其他安全问题，这表明 SMAEXO 的设计有效地解决了重量分配和平衡方面的挑战。然而，一名参与者指出了若绑带未牢固固定，脚踏板可能会碰撞障碍物或楼梯的潜在风险。这强调了确保正确调节并牢固系紧所有绑带以保障使用安全的重要性。

参与者反映，SMAEXO 在平地行走、坡道行走以及越障时均提供了充足的助力，减少了他们的体力消耗。这表明 SMAEXO 通过结合主动髋膝助力与被动柔性踝关节，在整个步态周期中提供了全面的关节支撑。

与双侧外骨骼相比，SMAEXO 在减轻重量和运动自由度方面展现出显著优势，同时比单关节设备提供了更全面的支撑。Table III 将本设备的主要特征与商用单侧外骨骼 HAL 和 LiteStepper 进行了对比。SMAEXO 在紧凑化设计、更轻的自重、直观且具成本效益的传感系统、更优的受力管理，以及有助实现自然步态的更具柔性的脚踏板方面具有优势。然而，在最大限度降低总重量以及改善穿脱便利性方面仍面临挑战，特别是由于在缺乏外部支撑时无法保持稳定的坐姿。未来的迭代应侧重于引入专用的支撑机构，以解决这一局限性并进一步提升可用性。

与双侧外骨骼相比，SMAEXO 减轻了重量并提供了更大的运动自由度，同时比单关节设备提供更全面的支撑。Table III 将其主要特征与商用单侧外骨骼 HAL [20] 和 LiteStepper [17] 进行了比较。SMAEXO 在紧凑设计、更轻的重量、直观且具成本效益的传感器系统、改进的受力管理以及用于自然步态的柔性脚踏板方面表现出色。然而，在进一步减轻重量以及改善穿脱便利性方面仍存在挑战，特别是由于在没有外部支撑的情况下缺乏稳定的坐姿。未来的迭代版本应集成专用的支撑机构以提高可用性。

## V. 结论

在本研究中，我们开发了单侧移动辅助外骨骼（Single-Sided Mobility Assistive Exoskeleton，SMAEXO），这是一种旨在增强偏瘫患者行动能力的单侧多关节下肢外骨骼。SMAEXO 主动辅助髋关节和膝关节，同时结合被动柔性踝关节以防止足下垂，确保在整个步态周期中提供动态辅助。其免工具调节、优化的重量分布和反作用力管理增强了可用性、平衡性、舒适度和安全性，同时最大限度地减少了患侧的负担。

我们在平地行走、斜坡行走和越障过程中，通过 8 名健全受试者评估了该设备的可用性、安全性、舒适性和辅助效果。受试者反馈该设备安全、舒适且有效，没有出现瘙痒或刺激等不良反应，并认为其辅助功能大有裨益。然而，研究也指出了需要改进的地方，包括更好的散热性能、减轻重量以及简化穿脱过程。

本研究的局限性在于受试者样本量较小且不够均衡，这可能会影响研究结果的普遍适用性。未来的工作将通过在更大规模、更多样化的受试者群体中进行广泛测试来解决这一问题，以实现稳健的验证。

在这些研究结果的基础上，我们将推进临床试验，以评估 SMAEXO 在脑卒中后患者中的有效性。全面的数据分析将为设计优化提供依据，以增强功能性、舒适度和用户体验。随着持续开发，SMAEXO 有望显著改善偏瘫患者的行动能力和生活质量。

[1] V. L. Feigin, M. Brainin, B. Norrving, S. Martins, R. L. Sacco, W. Hacke, M. Fisher, J. Pandian, and P. Lindsay, “World stroke organization (wso): global stroke fact sheet 2022,” International Journal of Stroke, vol. 17, no. 1, pp. 18–29, 2022.

[2] Y. Li, T. Fan, Q. Qi, J. Wang, H. Qiu, L. Zhang, X. Wu, J. Ye, G. Chen, J. Long et al., “Efficacy of a novel exoskeletal robot for locomotor rehabilitation in stroke patients: A multi-center, non-inferiority, randomized controlled trial,” Frontiers in aging neuroscience, vol. 13, p. 706569, 2021.

[3] G. Chen, C. K. Chan, Z. Guo, and H. Yu, “A review of lower extremity assistive robotic exoskeletons in rehabilitation therapy,” Critical Reviews™ in Biomedical Engineering, vol. 41, no. 4-5, 2013.

[4] E. Read, C. Woolsey, C. A. McGibbon, and C. O’Connell, “Physiotherapists’ experiences using the ekso bionic exoskeleton with patients in a neurological rehabilitation hospital: A qualitative study,” Rehabilitation research and practice, vol. 2020, no. 1, p. 2939573, 2020.

[5] “Ekso indego,” Ekso Bionics. [Online]. Available: https://eksobionics. com/ekso-indego-therapy/

[6] “Rewalk personal exoskeleton,” Lifeward. [Online]. Available: https://golifeward.com/products/rewalkpersonal-exoskeleton/

[7] “Rex,” Rex Bionics. [Online]. Available: https://www.rexbionics.com/ product-information/

[8] J. S. Lora-Millan, J. C. Moreno, and E. Rocon, “Coordination between partial robotic exoskeletons and human gait: A comprehensive review on control strategies,” Frontiers in Bioengineering and Biotechnology, vol. 10, p. 842294, 2022.

[9] J. M. Hidler and A. E. Wall, “Alterations in muscle activation patterns during robotic-assisted walking,” Clinical Biomechanics, vol. 20, no. 2, pp. 184–193, 2005.

[10] E. P. Washabaugh, T. E. Augenstein, A. M. Ebenhoeh, J. Qiu, K. A. Ford, and C. Krishnan, “Design and preliminary assessment of a passive elastic leg exoskeleton for resistive gait rehabilitation,” IEEE Transactions on Biomedical Engineering, vol. 68, no. 6, pp. 1941– 1950, 2020.

[11] J. Wang, X. Li, T.-H. Huang, S. Yu, Y. Li, T. Chen, A. Carriero, M. Oh-Park, and H. Su, “Comfort-centered design of a lightweight and backdrivable knee exoskeleton,” IEEE Robotics and Automation Letters, vol. 3, no. 4, pp. 4265–4272, 2018.

[12] J. S. Lora-Millan, F. J. Sanchez-Cuesta, J. P. Romero, J. C. Moreno, and E. Rocon, “A unilateral robotic knee exoskeleton to assess the role of natural gait assistance in hemiparetic patients,” Journal of NeuroEngineering and Rehabilitation, vol. 19, no. 1, p. 109, 2022.

[13] “Restore exo-suit,” Lifeward. [Online]. Available: https://golifeward. com/products/restore-exo-suit/

[14] G. Orekhov, Y. Fang, C. F. Cuddeback, and Z. F. Lerner, “Usability and performance validation of an ultra-lightweight and versatile untethered robotic ankle exoskeleton,” Journal of neuroengineering and rehabilitation, vol. 18, pp. 1–16, 2021.

[15] H. Kawamoto, T. Hayashi, T. Sakurai, K. Eguchi, and Y. Sankai, “Development of single leg version of hal for hemiplegia,” in 2009 Annual international conference of the IEEE engineering in medicine and biology society. IEEE, 2009, pp. 5038–5043.

[16] L. J. A. Mayag, M. Munera, and C. A. Cifuentes, “Human-in-the-´ loop control for agora unilateral lower-limb exoskeleton,” Journal of Intelligent & Robotic Systems, vol. 104, no. 1, p. 3, 2022.

[17] Y. Jin, B. Xiong, L. Chen, W. Zhao, Z. Li, C. Zhang, and X. Xu, “Efficacy and safety of using a unilateral lower limb exoskeleton combined with conventional treatment in post-stroke rehabilitation: a randomized controlled trial,” Frontiers in Bioengineering and Biotechnology, vol. 12, p. 1441986, 2024.

[18] R. Kitatani, K. Ohata, H. Takahashi, S. Shibuta, Y. Hashiguchi, and N. Yamakami, “Reduction in energy expenditure during walking using an automated stride assistance device in healthy young adults,” Archives of physical medicine and rehabilitation, vol. 95, no. 11, pp. 2128–2133, 2014.

[19] K. Seo, J. Lee, Y. Lee, T. Ha, and Y. Shim, “Fully autonomous hip exoskeleton saves metabolic cost of walking,” in 2016 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2016, pp. 4628–4635.

[20] “Hal for well-being (lower limb type),” Cyborg. [Online]. Available: https://www.cyberdyne.jp/english/products/fl05.html