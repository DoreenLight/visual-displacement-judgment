# Design and Evaluation of a Single-Sided Mobility Assistive Exoskeleton (SMAEXO) for Hemiplegia

> 原文出处：IEEE ICORR 2025: 597-602 ｜ DOI 10.1109/icorr66766.2025.11063171
> 中文精读：research-tool 九段闭环 + LLM 翻译（2026-10-01 生成）
> 原文 PDF：`Zhang2025-ICORR-SMAEXO`（本地 `papers/`）
> 译文完整度：9/9 段已译

---

## 一句话结论
SMAEXO 是一种面向偏瘫患者的单侧多关节外骨骼（髋、膝主动，踝被动），8 名健康受试者（含模拟偏瘫步态的物理治疗师）试穿后评价为舒适、安全、有效。

## 研究问题
- 单侧外骨骼能否在髋、膝、踝多关节上覆盖完整步态周期的辅助需求？
- 现有单侧外骨骼的平衡、反应力支撑不足以及多关节协同困难如何解决？
- 装置的模块化、可定制化与安全（自锁/缓冲/弹性件）能否兼顾不同体型？
- 在平地、坡道、越障三类任务下用户可用性、舒适性与辅助效果如何？

## 方法
- 机械设计：单侧（single-sided）多关节构型，髋关节 + 膝关节主动驱动，踝关节采用被动柔性关节以防止足下垂
- 力学/结构创新：优化反力支撑与重量分布；模块化尺寸可定制；自锁机构、缓冲垫、弹性件等安全设计
- 受试者：8 名健康受试者（其中含物理治疗师模拟偏瘫步态）
- 试验任务：平地行走、上/下坡行走、跨越障碍
- 评估维度：可用性、舒适性、安全性、辅助效果（结合主观反馈）
- 论文后续方法节（II. 及之后）在节选中未给出，故细节（如驱动方式、控制算法、传感器配置）原文未呈现

## 关键数据
- 样本量：8 名健康受试者
- 主动驱动关节数：2（髋 + 膝）
- 被动关节数：1（踝，柔性）
- 试验任务种类：3（平地、坡道、越障）
- 用户反馈改进点：需增强通风、减轻重量、便于穿脱
- 关节角度 RMSE/R²、跟踪误差、显著性 p 值、力矩幅值等定量指标：原文未给出

## 结论与局限
- 结论：受试者认为 SMAEXO 舒适、安全且有效，可在多任务下提供多关节辅助，具备临床转化潜力
- 局限：仅 8 名健康受试者试穿，未纳入真实偏瘫患者进行临床试验
- 局限：装置重量、通风与穿脱便利性仍需改进
- 局限：原文未提供定量性能指标（如关节轨迹跟踪误差、辅助力矩等），客观对比受限

## 对「视觉→关节角（RGB 视频估计人体关节角序列）」课题的可借鉴点
- 多关节耦合物理约束：髋–膝–踝在步态中具有强耦合运动学关系，可作为 RGB 视频关节角估计的正则化先验，保证三关节角度序列的物理一致性
- 步态相位分段：步态 stance/swing 相位划分可将连续关节角估计拆解为相位条件子任务，提高跨相位预测的鲁棒性（可作为评估指标）
- 模拟偏瘫步态测试集：本文招募物理治疗师"模拟偏瘫步态"的做法，可迁移为视觉模型的鲁棒性 / 域外泛化 benchmark
- 平衡与反应力约束：文中强调 reaction force 与重量分布对平衡的影响，提示视觉关节角下游应用（如闭环控制外骨骼）需同步关注地面反力线、CoM 等物理量，可作多模态融合的可比指标

---

## 全文中文译

2025年5月12-16日，美国伊利诺伊州芝加哥

# 面向偏瘫（Hemiplegia）的单侧移动辅助外骨骼（Single-Sided Mobility Assistive Exoskeleton, SMAEXO）设计与评估

Xiaoyue Yan¹,², Wei Yun Yau³, Isaac Okumura Tan², Longbin Zhang²,⁴, Youlong Wang¹,², Oliver Roberts², Seng Kwee Wee⁵, Wei Tech Ang¹,²

**摘要**— 本研究提出了单侧移动辅助外骨骼（Single-Sided Mobility Assistive Exoskeleton, SMAEXO），这是一种新颖的、单侧、多关节设备，旨在增强偏瘫患者的移动能力和康复效果。SMAEXO 在髋关节和膝关节提供主动辅助，并结合一种被动（方式），对位置控制的依赖使其不适合脑卒中（Stroke）幸存者，他们需要响应迅速、量身定制的支持 [8]。对于偏瘫患者而言，单侧（Unilateral）外骨骼提供了一种更简单、更轻量的替代方案。与限制自然运动并破坏步态对称性的双侧（Bilateral）设备不同 [9]，单侧——

以更有效、更个性化的移动支持。

## I. 引言

卒中（中风，stroke）是导致残疾的主要原因之一，常导致偏瘫（hemiparesis），影响 65% 的幸存者出现步态障碍 [1]。常见问题包括步态不对称、平衡能力差和足下垂 [2]，需要强化康复以恢复移动性。传统疗法劳动密集，每节次需要两名物理治疗师，限制了可扩展性 [3]。外骨骼（exoskeleton）提供了一种更高效的解决方案，通过持续、自动化的支持来增强康复效果，同时减轻治疗师的工作负担 [4]。

大多数下肢外骨骼是双侧（bilateral）型，通过针对髋、膝、踝关节提供完整的移动支持。Ekso Indego [5]、ReWalk [6] 和 Rex [7] 等装置可辅助严重运动功能丧失的用户，例如脊髓损伤患者。然而，其笨重的设计、高昂的成本以及……单侧（unilateral）型外骨骼仅辅助患肢，以较小的使用者负担提升移动性。然而，单侧方案本身也带来挑战。这类装置的非对称结构使保持平衡变得困难，且有限的反作用力支撑阻碍了产生足够力矩以充分稳定和推进患肢。

此外，许多单侧型外骨骼仅辅助单个关节，限制了其对卒中相关多关节损伤的疗效。膝关节装置如密歇根大学的被动外骨骼 [10]、纽约大学的主动外骨骼 [11] 以及 REFLEX 系统 [12] 可辅助膝关节以实现站立支撑和摆动通过。同样，踝关节外骨骼如 ReStore Exo-Suit [13] 和北亚利桑那大学的装置 [14] 可改善足底屈曲（plantarflexion）与足背屈（dorsiflexion），但仅提供部分支持，忽略了其他关节的损伤。HAL [15]、AGoRA [16] 和 LitteStepper [17] 等多关节单侧装置试图同时辅助多个关节，但往往更重、更不友好，并且更难与使用者的意图无缝集成。

实际上，现有的单侧型外骨骼仍无法在保持可用性、舒适性和安全性的同时提供无缝的多关节辅助。这些局限性凸显了对新一代单侧型外骨骼的需求，使其能够为偏瘫患者提供全面、以用户为中心的支持。

为弥补这些空白，本文介绍了单侧移动辅助外骨骼（Single-Sided Mobility Assistive Exoskeleton，SMAEXO），一种专为偏瘫患者设计的新型多关节装置。SMAEXO 集成了创新特性，旨在优化反作用力管理与重量分布，同时确保髋、膝、踝关节的辅助既有效又直观。本研究的主要贡献包括：

1) SMAEXO 的开发实现了跨髋、膝、踝关节的集成辅助，为患肢提供全面的支持。优化的重量分布与反作用力管理策略确保装置能够有效稳定和支撑使用者，同时不影响舒适性、安全性或运动便利性。

2) 通过 … 对 SMAEXO 的性能进行了验证

---

本研究由 ASTAR 在国家机器人计划（National Robotics Programme，NRP）BAU 资助下支持，项目为辅助机器人计划（项目编号：M22NBK0074）。

¹ 新加坡南洋理工大学 机械与航空航天工程学院
² 新加坡南洋理工大学 新加坡康复研究院
³ 新加坡 A*STAR 信息通信研究院
⁴ 中国长沙湖南大学 机器人学院
⁵ 新加坡 陈笃生医院

通讯作者电子邮件：{xiaoyue.yan, wtang}@ntu.edu.sg

979-8-3503-8068-2/25/$31.00 ©2025 IEEE 597

授权许可使用仅限于：中南民族大学。于2026年10月01日03:37:54 UTC从IEEE Xplore下载。限制适用。

用户的生物力学能力。作为关节缓冲器实现的软限位器，在关节接近硬限位器时吸收冲击（图2(b)）。关节缓冲器易于更换，以适应关节活动范围（Range of Motion, ROM）受限的用户。此外，所有腿部连接杆都配备护套和衬垫，如图2(d)所示，这些护套可以方便地打开，便于快速穿脱，并且护套的长半轴、短半轴可以方便地调节，以适应腿围不同的用户。

为了进一步增强适应性和对准性，SMAEXO配备了相应的可调机构，允许无需工具即可调节腰围宽度、大腿和小腿杆长度以及脚部尺寸，从而适应不同身体尺寸的用户（图2和图3(a)）。所有可调机构均配备复位弹簧，以确保锁定状态以保证安全。

图1. 单侧移动辅助外骨骼（SMAEXO）的架构。红色箭头表示关节运动方向。

<div style="text-align: left">

| Vest Harness | Battery & Control | Active Hip |
| :---: | :---: | :---: |
| | Components | |
| Back Module | | Passive |
| Unaffected-side | Active Hip | |
| Module | | Knee |
| Affected-side | Passive | |
| Module | Ankle | |

</div>

<div style="text-align: right">

**电池与控制组件**

**被动**

**主动髋**

**膝**

**被动**

**踝**

</div>

SMAEXO系统的主要规格汇总于表I中。

<div align="center">

表I  SMAEXO的规格

| 项目 | 规格 |
| :--- | :--- |
| 运动范围 (°) | 髋：(-20, 125)；膝：(0, 135)；踝：(-40, 24) |
| 长度范围 (mm) | 上肢：400–500（长），300–400（短）；<br>下肢：380–480（长），280–380（短）；<br>髋宽：335–435；脚部：250–305 |
| 髋与膝关节电机 | 额定扭矩：12Nm；<br>最大扭矩：37Nm；传动比：1:6；重量：0.759kg |
| 电池 | 锂电池组，24V 6Ah，重量：0.82kg |
| 重量 (kg) | 总计：7.5；<br>患侧模块：3.3；<br>健侧模块：0.3<br>背部模块：2.1<br>背心背带：1.8 |

</div>

## II. 系统概述（SYSTEM OVERVIEW）

### A. SMAEXO架构

SMAEXO是通过与专门从事中风康复和下肢外骨骼的物理治疗师紧密合作开发的。这一咨询过程确保了系统符合偏瘫个体的独特需求，强调可用性、安全性、舒适性以及全面辅助。该设备采用模块化设计，包括四个主要组件（图1）：

- **患侧模块（Affected-side module）**：包含由准直驱（Quasi-Direct Drive, QDD）电机驱动的主动髋关节和膝关节，这些电机结合高扭矩电机与低传动比（1:6）以提供高带宽和反向驱动能力（back-drivability）。此外，还配备一个与脚踏板配合的被动踝关节，以防止足下垂。

- **健侧模块（Unaffected-side module）**：具有一个被动髋关节，便于髋屈伸和外展/内收，确保健侧的自然运动。

- **背部模块（Back module）**：连接所有模块并容纳控制组件和电池，将电池策略性地放置在健侧以优化重量分配和平衡。

- **背心背带（Vest harness）**：将设备重量和反作用力均匀分布在用户的躯干上，增强舒适性和稳定性。

为确保安全性和舒适性，所有关节均同时配备硬限位器和软限位器，以将运动范围（Range of Motion, ROM）限制在用户生物力学能力范围内。

### B. 高效反作用力管理与重量分配

有效的反作用力管理对于最小化不适感和增强可用性至关重要。像Ekso Indego [5]、ReWalk [6]和Rex [7]这样的双侧外骨骼通过更重的自重和广泛的用户接触实现这一点，能够提供更高的辅助扭矩，但会增加体积并降低机动性。相比之下，轻量级外骨骼由于反作用力支撑较弱而面临扭矩限制。例如，本田的SMA装置提供4Nm的髋部扭矩[18]，而三星的GEMS通过髋部支撑提供额外支撑，达到12Nm[19]。像AGoRA这样的单侧装置通过双侧配置改善扭矩和平衡，但质量增加至17kg [16]。

SMAEXO通过集成创新特性来解决这些挑战，实现高效的反作用力管理和重量分配。通过人体工学背心背带和与脚踏板耦合的柔性踝关节的组合，SMAEXO在保持轻量化和用户友好特性的同时，确保有效的稳定性和全面的辅助。

- **人体工学背心背带**

如图3(a)所示，SMAEXO背心背带被设计为紧贴穿戴者的身体。它由

979-8-3503-8068-2/25/$31.00 ©2025 IEEE 598

授权许可使用仅限于：中南民族大学。于2026年10月01日03:37:54 UTC从IEEE Xplore下载。限制适用。

腰带杆连接器（Waist Bar Connector）
- 躯干 IMU（Torso IMU）
- 腰带杆Ⅱ（Waist Bar Ⅱ）
- 腰带（Belt）

(b) 髋关节模块
- 气囊（Airbag）
- 缓冲块（Bumper）
- 被动髋关节编码器（Passive Hip Encoder）
- 腰带杆Ⅰ（Waist Bar Ⅰ）
- 肢杆（Limb Bar）
- 护套（Cuff）
- 被动髋关节（Passive Hip）

由集成背带与腰带的背心构成，其弹性脊柱由弹簧钢等柔性材料制成，可自然贴合穿戴者的脊柱曲度。为增强贴合度与接触效果，设置了可调节的气囊凸块，可适配不同体型并提升整体穿戴舒适性。该结构能够有效利用穿戴者的腰、胸、腹、背部肌肉以稳定反作用力，从而降低不适感并提升活动能力。

- 柔性踝关节（Flexible Ankle Joint）

**图 3（Fig. 3）.** SMAEXO 背心式背带、背部模块以及健侧模块。（a）背部模块与背心式背带配合使用，内置惯性测量单元（Inertial Measurement Unit, IMU）用于监测躯干角度。（b）健侧模块采用被动式髋关节结构，并配备编码器以测量矢状面髋关节角度。红色箭头指示腰部宽度与髋关节运动的线性调节方向。

与大多数外骨骼（如使用固定踝关节和刚性足底板以防止足下垂并将重量传递至地面的膝踝足矫形器（Knee-Ankle-Foot Orthoses, KAFO）、ReWalk™ [6] 以及 HAL [15]）不同，SMAEXO 采用了一种通用性强的踝关节设计，兼顾舒适性与自然步态。该创新设计能够适应不同用户的个体需求，有助于改善其活动能力。

如图 4 与图 5 所示，SMAEXO 踝关节具备以下特性：(1) 一种角度可调机构，可在受控范围内自由旋转，也可固定于特定角度。(2) 一种柔性足底板，在足跟与前足处内置力传感器，用于检测步态事件。(3) 在站立相期间将重量有效地传递至地面，同时不影响力传感器的功能。(4) 与弹性绑带和踝部护具集成，在摆动相期间防止足下垂。(5) 足底板长度可调，以适应不同的脚部尺寸。(6) 采用聚氨酯（Polyurethane, PU）、橡胶及乙烯-乙酸乙烯酯共聚物（Ethylene-Vinyl Acetate, EVA）等弹性耐用材料制造，以实现自然的地面接触并保证耐用性。

**图 4（Fig. 4）.** SMAEXO 的踝关节与足底板。（a）踝关节通过输出杆与弹性防足下垂绑带（其作用类似于一根具有可调张力的弹簧，以适应不同用户）与足底板相连。（b）足底板通过前足底板Ⅰ在足跟底板Ⅰ上滑动实现长度可调，以适应不同的脚部尺寸。前足底板Ⅰ与足跟底板Ⅰ内的嵌入式力传感器由额外一层（前足底板Ⅱ与足跟底板Ⅱ）加以保护。集成的线缆通道便于布线。

**图 4 部件标签：**

- （a）小腿杆Ⅱ（Lower Leg Bar Ⅱ）、踝关节输出杆（Ankle Joint Output Rod）、踝部护具（Ankle Cuff）、防足下垂绑带（Footdrop Prevention Strap）、绑带（Straps）、足底板（Sole）、踝关节枢轴（Ankle Pivot）、角度可调机构（Angle-adjustable Mechanism）。
- （b）前足底板Ⅰ（Front Sole Ⅰ）、前足底板Ⅱ（Front Sole Ⅱ）、足跟底板Ⅰ（Heel Sole Ⅰ）、足跟底板Ⅱ（Heel Sole Ⅱ）、力传感器（Force Sensors）、足跟（Heel）、线缆通道（Cable Channels）、绑带（Straps）。

该构型在提供个性化调节以匹配个体步态模式和辅助需求的同时，有助于实现自然步态，从而提升舒适性与活动能力。

凭借在反作用力管理与重量支撑方面的技术进步，SMAEXO 实现了最大 25 Nm 的辅助扭矩。该能力在满足偏瘫康复的高扭矩需求的同时，确保用户

**图 5 部件标签：**

- （a）小腿杆Ⅱ（Lower Leg Bar Ⅱ）、锁扣盖（Latch Cover）、踝关节枢轴（Ankle Pivot）、锁扣外壳（Latch Housing）、锁扣孔（Latch Holes）、复位弹簧（Reset Springs）、弹簧加载锁扣（Spring-loaded Latch）、限位块（Limit Blocks）、踝关节输出杆（Ankle Joint Output Rod）、踝关节活动度（Ankle ROM）。
- （b）锁扣外壳（Latch Housing）、回转块（Slewing Block）、滑动导轨凹槽Ⅰ（Sliding Guide Groove Ⅰ）、滑动导轨凹槽Ⅱ（Sliding Guide Groove Ⅱ）、锁扣（Latch）、复位弹簧（Reset Springs）、限位块（Limit Blocks）、弹簧加载锁扣（Spring-loaded Latch）、滑动限位Ⅱ（Slide Limit Ⅱ）、滑动限位Ⅰ（Slide Limit Ⅰ）。
- （c）小腿杆Ⅱ（Lower Leg Bar Ⅱ）、踝关节输出杆（Ankle Joint Output Rod）、回转块（Slewing Block）、锁扣外壳（Latch Housing）、限位块（Limit Blocks）、枢轴（Pivot）、弹簧加载锁扣（Spring-loaded Latch）、锁扣盖（Latch Cover）。

图5. 踝关节的角度可调机构（angle-adjustable mechanism）。(a) 锁定状态，运动受限。(c) 爆炸视图。该机构利用卡扣壳体（latch housing）、滑动导向槽、回转块（slewing block）、弹簧加载限位块、枢轴（pivot）和卡扣盖板实现可控的踝关节旋转或固定定位。限位块沿导向槽 II（guide groove II）旋转，并在弹簧加载卡扣（spring-loaded latch）啮合卡扣孔（latch hole）时锁定。解锁后释放卡扣，恢复运动。回转块在导向槽 I（guide groove I）内旋转，并在与两个限位块同时接触时，将踝关节锁定在设定角度。

| 足底压力（Plantar Pressure） | 步态事件检测（Gait Event Detection） | 步态相位（Gait phase） | 滤波器（Filter）<br>• EWMA（指数加权移动平均） |
|---|---|---|---|
| 力传感器（Force Sensors） |  |  |  |
| 关节运动学（Joint Kinematics）<br>编码器（Encoders）、IMU（惯性测量单元） |  |  | 外骨骼透明控制算法<br>• 外骨骼重力补偿（Exo GC） | 𝜏exo_gc |
| 外骨骼参数（Exo Parameters） |  |  | • 用户辅助控制算法 | + |
| 用户参数（User Parameters） |  |  | • 部分用户移动辅助（Partial User Mobility Assistance） | 𝜏user_assist |

→ SMAEXO 关节

图6. SMAEXO 的控制框图。

图7. 实验设置。

comfort, providing an effective and user-friendly solution for mobility assistance.（舒适性，为移动辅助提供一种有效且用户友好的解决方案。）

## C. 控制方案（Control Scheme）

SMAEXO 采用多级控制策略，并配备可靠且经济高效的传感器系统以增强其功能性。该系统包含躯干 IMU 以及嵌入髋关节和膝关节的编码器（encoder），用于提供精确的关节运动学（kinematic）测量。来自力传感器的足底压力测量数据与关节运动学数据相结合，用于识别各种日常活动中的步态相位，包括坐立转换以及在平地、斜坡或楼梯上行走。控制系统利用该步态相位信息实现外骨骼透明性（例如，重力补偿（Gravity Compensation, GC））以及部分用户移动辅助，从而提供优化的支撑和更好的用户体验。此外，外骨骼参数和用户参数可根据个体辅助需求进行定制，而诸如指数加权移动平均（Exponentially Weighted Moving Average, EWMA）等滤波器可确保整个步态周期内力矩曲线的平滑（图6）。

## III. 实验（Experiments）

为了评估 SMAEXO 在可用性、舒适性、安全性和辅助效果方面的表现，研究招募了八名健康受试者，其中包括两名专门从事下肢康复的物理治疗师。在测试过程中，物理治疗师根据其经验模拟中风步态模式——包括足下垂（foot drop）、僵硬膝（stiff knee）、画圈步态（circumduction）和弱膝（weak knee）——以从患者角度评估外骨骼的有效性。受试者的人口学信息和人体测量学特征详见表 II（Table II）。所有受试者在被告知研究目标和潜在风险后，均签署了书面知情同意书。研究方案经南洋理工大学机构审查委员会（Nanyang Technological University Institutional Review Board）批准并受其监管。

实验期间，受试者可以随时休息、停止或退出测试。评估包括两个环节（穿戴外骨骼和不穿戴外骨骼），每个环节包含三项任务。受试者至少重复每项任务三次：

- 10 米水平行走；
- 4 米、8 度倾角的斜坡上、下行走；
- 跨越一个高 10 厘米、宽 12.5 厘米的障碍物。

在第二个环节中，受试者在测试前对穿戴外骨骼进行了 30 分钟的熟悉。研究人员向其讲解了外骨骼的使用方法，并确保其适应。受试者被要求报告任何不适或疼痛情况，研究人员对此进行了记录和处理。

研究结束时，受试者填写了一份问卷，从可用性、舒适性、安全性和辅助效果四个方面评估该外骨骼。所有问题均采用 1 至 5 分评分，其中 1 分代表"强烈不同意"，5 分代表"强烈同意"。

| 强烈不同意（Strongly Disagree） | 不同意（Disagree） | 中立（Neutral） | 同意（Agree） | 强烈同意（Strongly Agree） |
|---|---|---|---|---|
| 设备穿戴容易。 |  |  |  |  |
| 穿戴时感觉设备不重。 |  |  |  |  |
| 使用设备时未出现任何不适，例如疼痛、瘙痒、刺激感或异常发热。 |  |  |  |  |
| 使用设备时感到安全，未感觉到跌倒风险。 |  |  |  |  |
| 使用设备进行水平行走时感觉更省力。 |  |  |  |  |
| 使用设备进行斜坡行走时感觉更省力。 |  |  |  |  |
| 使用设备跨越障碍时感觉更省力。 |  |  |  |  |
| 总体感觉该设备提供的辅助是充分的。 |  |  |  |  |

0                      20              40             60        80               100
                                                                                                                  响应百分比（Percentage of Responses）

图8. 在水平行走、斜坡行走以及跨越障碍任务中，受试者对 SMAEXO 的问卷反馈结果。

979-8-3503-8068-2/25/$31.00 ©2025 IEEE                                                                        600
  Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:37:54 UTC from IEEE Xplore. Restrictions apply.

**表 II**

**受试者人口统计学与人体测量学数据（Demographic and Anthropometric Details of Participants）**

| 受试者 | 年龄  | 性别 | 身高（cm） | 体重（kg） |
|--------|-------|------|-----------|-----------|
| P1     | 31-40 | F    | 155       | 54.5      |
| P2     | 21-30 | M    | 171       | 70        |
| P3     | 31-40 | F    | 163       | 55        |
| P4     | 31-40 | M    | 183       | 76        |
| P5     | 21-30 | M    | 183       | 73        |
| P6     | 21-30 | M    | 181       | 85        |
| P7     | 21-30 | M    | 176       | 85        |
| P8     | 41-50 | M    | 169       | 71        |

**表 III**

**SMAEXO 的规格参数（Specifications of the SMAEXO）**

| 设备          | HAL（Cyberdyne）                                                                                  | LiteStepper（Angelexo）                                                | SMAEXO                                |
|---------------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|---------------------------------------|
| 控制策略      | 基于模型的控制（放大肌肉活动）或预定义步态轨迹                                                     | 复现健侧肢体步态轨迹，并借助拐杖手柄启动行走                          | 混合基于模型的控制                    |
| 驱动器        | 2 个无刷直流电机（Brushless DC motors）                                                            | 2 个无刷直流电机（Brushless DC motors）                                 | 2 个 QDD 电机（QDD motors）           |
| 使用的传感器  | 运动学传感器（Kinematic Sensors）；足底压力传感器（Sole Pressure Sensors）；肌电信号（EMG）       | 运动学传感器；足底压力传感器；手杖压力传感器                           | 运动学传感器（4 个）；足底压力传感器  |
| 脚踏板类型    | 带刚性鞋垫的定制鞋                                                                                  | 机械式脚踏板                                                            | 柔性脚踏板                            |
| 承重支撑      | 刚性鞋垫（地面）                                                                                    | 背心（核心）；脚踏板（地面）                                           | 背心式肩带（核心）；脚踏板（地面）    |
| 重量          | 9 kg                                                                                                | 16.9 kg                                                                 | 7.5 kg                                |

## IV. 结果与讨论

如图 8 所示，受试者赞赏 SMAEXO 的模块化设计以及无需工具的调节特性，便于穿脱。然而，其不平衡的结构使中立位的保持具有挑战性，尤其在坐姿时。物理治疗师指出，卒中后用户通常需要两名助手进行稳定。引入支撑框架可在穿脱过程中提供稳定性，从而解决此问题。

受试者认为该设备的重量可接受。在站立相，SMAEXO 有效地将重量传递至地面，减轻了膝关节的负担。然而，物理治疗师对偏瘫用户提出了担忧，因为在摆动相中，健侧腿承担着设备的重量。这可能会增加健侧膝关节的应力，对更易罹患膝关节问题（如骨关节炎）的老年卒中后个体尤其值得关注。建议进一步减轻外骨骼的重量以缓解此种负担。

据报告，该设备穿着舒适，无疼痛、瘙痒或刺激现象。可调节机构使其能够适应不同的身体尺寸，确保贴合紧密，从而有效调动用户核心肌群以实现最优的反作用力管理与重量分配。然而，两名受试者在穿戴设备超过一小时后背部出现轻微出汗，可能由背心和气囊凸起引起。在这些部件中加入透气材料可提升长时间使用的舒适性。

在安全性方面，受试者表示在试验期间对跌倒或其他安全问题没有担忧，表明 SMAEXO 的设计有效解决了重量分配与平衡挑战。然而，一名受试者指出，如果绑带未紧固好，脚踏板存在接触障碍物或楼梯的潜在风险。这强调了确保所有绑带正确调节并紧固以维持使用安全的重要性。

受试者反映，SMAEXO 在平地行走、坡道行走以及跨越障碍物时提供了充足的助力，减轻了他们的努力。这表明 SMAEXO 通过将主动的髋关节与膝关节辅助与被动的柔性踝关节相结合，在整个步态周期内提供了全面的关节支撑。

与双侧外骨骼相比，SMAEXO

...在减重和运动自由度方面展现出优势，同时比单关节装置提供更全面的支撑。表 III 将本装置的主要特性与商用单侧外骨骼 HAL（混合辅助肢体，Hybrid Assistive Limb）和 LiteStepper 进行了比较。SMAEXO 在紧凑设计、更轻的重量、直观且经济高效的传感器系统、改进的力管理以及更灵活的脚踏板以实现自然步态方面具有优势。然而，在最小化整体重量以及改善穿脱便利性方面仍面临挑战，特别是由于缺乏在无外部支撑情况下的稳定坐姿。未来的迭代版本应专注于集成专用的支撑机构，以解决这一局限性并进一步提升可用性。

   与双侧外骨骼相比，SMAEXO 重量更轻，运动自由度更大，同时比单关节装置提供更全面的支撑。表 III 将其关键特性与商用单侧外骨骼 HAL [20] 和 LiteStepper [17] 进行了比较。SMAEXO 在紧凑设计、更轻重量、直观且经济高效的传感器系统、改进的力管理以及支持自然步态的灵活脚踏板方面表现出色。然而，在进一步减重以及改善穿脱便利性方面仍面临挑战，特别是由于缺乏在无外部支撑情况下的稳定坐姿。未来的迭代版本应集成专用的支撑机构以增强可用性。

## V. 结论（CONCLUSION）

   本研究开发了单侧移动辅助外骨骼（Single-Sided Mobility Assistive Exoskeleton, SMAEXO），这是一种为增强偏瘫患者移动能力而设计的单侧多关节下肢外骨骼。SMAEXO 主动支撑髋关节和膝关节，同时集成被动式柔性踝关节以防止足下垂，确保在整个步态周期内提供动态辅助。其免工具调节、优化重量分布和反作用力管理提升了可用性、平衡性、舒适性和安全性，同时减轻患侧负担。

   我们使用 8 名健康受试者，在平地行走、斜坡行走和跨越障碍等任务中评估了可用性、安全性、舒适性和辅助效果。受试者反馈该装置安全、舒适、有效，未出现瘙痒或刺激等不良反应，并认为辅助效果良好。然而，也指出了需要改进的方面，包括更好的散热、进一步减重以及简化的穿脱过程。

   本研究的局限性在于受试者样本量小且分布不均，这可能影响研究结果的普适性。未来工作将通过更大规模、多样化的受试者群体进行全面测试以进行稳健验证，从而解决这一问题。

   基于上述发现，我们将推进临床试验，以评估 SMAEXO 在卒中后患者中的有效性。全面的数据分析将为设计改进提供依据，以增强功能、舒适性和用户体验。随着持续开发，SMAEXO 有潜力显著改善偏瘫患者的移动能力和生活质量。

## 参考文献（REFERENCES）

[1] V. L. Feigin、M. Brainin、B. Norrving、S. Martins、R. L. Sacco、W. Hacke、M. Fisher、J. Pandian 和 P. Lindsay，"世界卒中组织（WSO）：2022 年全球卒中情况说明书（World stroke organization (wso): global stroke fact sheet 2022）"，《国际卒中杂志》（International Journal of Stroke），卷 17，第 1 期，第 18–29 页，2022 年。

[2] Y. Li、T. Fan、Q. Qi、J. Wang、H. Qiu、L. Zhang、X. Wu、J. Ye、G. Chen、J. Long 等，"新型外骨骼机器人在卒中患者运动康复中的疗效：一项多中心、非劣效、随机对照试验（Efficacy of a novel exoskeletal robot for locomotor rehabilitation in stroke patients: A multi-center, non-inferiority, randomized controlled trial）"，《老化神经科学前沿》（Frontiers in Aging Neuroscience），卷 13，第 706569 页，2021 年。

[3] G. Chen、C. K. Chan、Z. Guo 和 H. Yu，"康复治疗中下肢辅助机器外骨骼综述（A review of lower extremity assistive robotic exoskeletons in rehabilitation therapy）"，《生物医学工程关键评论》（Critical Reviews™ in Biomedical Engineering），卷 41，第 4-5 期，2013 年。

[4] E. Read、C. Woolsey、C. A. McGibbon 和 C. O'Connell，"理疗师在神经康复医院使用 Ekso 仿生外骨骼的体验：一项定性研究（Physiotherapists' experiences using the ekso bionic exoskeleton with patients in a neurological rehabilitation hospital: A qualitative study）"，《康复研究与实践》（Rehabilitation Research and Practice），卷 2020，第 1 期，第 2939573 页，2020 年。

[5] "Ekso Indego"，Ekso Bionics。[在线]。可访问：https://eksobionics.com/ekso-indego-therapy/

[6] "Rewalk 个人外骨骼（Rewalk personal exoskeleton）"，Lifeward。[在线]。可访问：https://golifeward.com/products/rewalkpersonal-exoskeleton/

[7] "Rex"，Rex Bionics。[在线]。可访问：https://www.rexbionics.com/product-information/

[8] J. S. Lora-Millan、J. C. Moreno 和 E. Rocon，"部分机器外骨骼与人步态的协调：关于控制策略的全面综述（Coordination between partial robotic exoskeletons and human gait: A comprehensive review on control strategies）"，《生物工程与生物技术前沿》（Frontiers in Bioengineering and Biotechnology），卷 10，第 842294 页，2022 年。

[9] J. M. Hidler 和 A. E. Wall，"机器人辅助行走过程中肌肉激活模式的改变（Alterations in muscle activation patterns during robotic-assisted walking）"，《临床生物力学》（Clinical Biomechanics），卷 20，第 2 期，第 184–193 页，2005 年。

[10] E. P. Washabaugh、T. E. Augenstein、A. M. Ebenhoeh、J. Qiu、K. A. Ford 和 C. Krishnan，"用于阻力步态康复的被动弹性腿外骨骼的设计与初步评估（Design and preliminary assessment of a passive elastic leg exoskeleton for resistive gait rehabilitation）"，《IEEE 生物医学工程汇刊》（IEEE Transactions on Biomedical Engineering），卷 68，第 6 期，第 1941–1950 页，2020 年。

[11] J. Wang、X. Li、T.-H. Huang、S. Yu、Y. Li、T. Chen、A. Carriero、M. Oh-Park 和 H. Su，"以舒适性为中心的轻量级可回驱膝关节外骨骼设计（Comfort-centered design of a lightweight and backdrivable knee exoskeleton）"，《IEEE 机器人与自动化快报》（IEEE Robotics and Automation Letters），卷 3，第 4 期，第 4265–4272 页，2018 年。

[12] J. S. Lora-Millan、F. J. Sanchez-Cuesta、J. P. Romero、J. C. Moreno 和 E. Rocon，"用于评估偏瘫患者自然步态辅助作用的单侧机器膝外骨骼（A unilateral robotic knee exoskeleton to assess the role of natural gait assistance in hemiparetic patients）"，《神经工程与康复杂志》（Journal of NeuroEngineering and Rehabilitation），卷 19，第 1 期，第 109 页，2022 年。

[13] "Restore 外骨骼服装（Restore exo-suit）"，Lifeward。[在线]。可访问：https://golifeward.com/products/restore-exo-suit/

[14] G. Orekhov、Y. Fang、C. F. Cuddeback 和 Z. F. Lerner，"超轻量级多功能无系绳机器踝外骨骼的可用性与性能验证（Usability and performance validation of an ultra-lightweight and versatile untethered robotic ankle exoskeleton）"，《神经工程与康复杂志》（Journal of Neuroengineering and Rehabilitation），卷 18，第 1–16 页，2021 年。

[15] H. Kawamoto、T. Hayashi、T. Sakurai、K. Eguchi 和 Y. Sankai，"用于偏瘫的 HAL 单腿版本开发（Development of single leg version of HAL for hemiplegia）"，2009 年 IEEE 医学与生物工程学会国际年会论文集。IEEE，2009 年，第 5038–5043 页。

[16] L. J. A. Mayag、M. Múnera 和 C. A. Cifuentes，"Agora 单侧下肢外骨骼的人在环控制（Human-in-the-loop control for agora unilateral lower-limb exoskeleton）"，《智能与机器人系统杂志》（Journal of Intelligent & Robotic Systems），卷 104，第 1 期，第 3 页，2022 年。

[17] Y. Jin、B. Xiong、L. Chen、W. Zhao、Z. Li、C. Zhang 和 X. Xu，"在卒中后康复中联合常规治疗使用单侧下肢外骨骼的疗效与安全性：一项随机对照试验（Efficacy and safety of using a unilateral lower limb exoskeleton combined with conventional treatment in post-stroke rehabilitation: a randomized controlled trial）"，《生物工程与生物技术前沿》（Frontiers in Bioengineering and Biotechnology），卷 12，第 1441986 页，2024 年。

[18] R. Kitatani、K. Ohata、H. Takahashi、S. Shibuta、Y. Hashiguchi 和 N. Yamakami，"健康年轻成人使用自动步幅辅助装置行走时能量消耗的减少（Reduction in energy expenditure during walking using an automated stride assistance device in healthy young adults）"，《物理医学与康复档案》（Archives of Physical Medicine and Rehabilitation），卷 95，第 11 期，第 2128–2133 页，2014 年。

[19] K. Seo、J. Lee、Y. Lee、T. Ha 和 Y. Shim，"全自主髋部外骨骼节省行走代谢成本（Fully autonomous hip exoskeleton saves metabolic cost of walking）"，2016 年 IEEE 国际机器人与自动化大会 (ICRA)。IEEE，2016 年，第 4628–4635 页。

[20] "HAL 福祉型（下肢型）（HAL for well-being (lower limb type)）"，Cyborg。[在线]。可访问：https://www.cyberdyne.jp/english/products/fl05.html

979-8-3503-8068-2/25/$31.00 ©2025 IEEE 602
授权许可使用仅限于：中南民族大学（South-Central Minzu University）。于 2026 年 10 月 01 日 03:37:54 UTC 从 IEEE Xplore 下载。限制适用。
