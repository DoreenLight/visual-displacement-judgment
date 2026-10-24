# 2011年人体动力学研讨会（2011 Symposium on Human Body Dynamics）

# 人体运动模拟：基于 OpenSim 的应用

Jeffrey A. Reinbolt<sup>a,</sup>\*, Ajay Seth<sup>b</sup>, Scott L. Delp<sup>b,c</sup>

<sup>a</sup>Mechanical, Aerospace, & Biomedical Engineering, The University of Tennessee, Knoxville, TN, USA <sup>b</sup>Bioengineering and <sup>c</sup>Mechanical Engineering, Stanford University, Stanford, CA, USA

## 摘要

计算机模拟在解决复杂工程问题中发挥着越来越重要的作用，并具有彻底革新基于实验的医疗决策与治疗方案设计的潜力。仅靠实验虽能提供对运动动力学的重要认识，但这种认识是有限的。尽管我们可以测量引起运动的某些物理量（如肌肉活动和地面反作用力），但模拟可以通过估计其他重要变量（如肌肉力和关节力）来补充这些测量。模拟还使我们能够建立因果关系，从而深入洞察肌肉功能。模拟或许最令人兴奋的特性是具备开展“假设（what if）”研究的潜力，以检验科学假设、预测功能性结果并发现涌现行为。本文重点介绍了使用 OpenSim 的应用案例，包括以下项目：最小化不合理模拟的评估指标；确定新的运动方式作为降低损伤风险的体育训练工具；以及建立姿态、肌肉力与地面反作用力之间的关系。

© 2011 Published by Elsevier Ltd. Open access under CC BY-NC-ND license. Peer-review under responsibility of John McPhee and József Kövecses Keywords: 动力学模拟（Dynamic simulation）；肌肉骨骼建模（musculoskeletal modeling）；姿态（posture）；地面反作用力（ground reaction forces）；降低损伤风险（reducing injury risk）

## 1. 引言

试想一下将这份文档拿在手中，或使用计算机鼠标和键盘浏览它所必需的肌肉协调。人体运动需要众多肌肉的协同配合。神经控制与意图运动之间的转化极其复杂，涉及许多独立要素（例如：肌腱动力学、肌肉骨骼几何结构和多体动力学）。这些独立要素中有许多已通过实验得到了表征 [1-4]，但仅凭实验对运动动力学的理解十分有限。尽管我们可以测量产生运动的少数物理量（例如肌电图和地面反作用力），但要测量诸如肌肉力和关节力等重要变量却极其困难。而要建立能深入理解肌肉功能的因果关系则更是难上加难。

仿真补充了实验，并给出了通常无法测量的变量的估计值，从而为深入理解肌肉功能和人体运动控制提供了见解。基于实验的模型和仿真用于在模拟的神经控制与模拟的运动之间生成肌腱动力学、肌肉骨骼几何结构以及多体动力学变换。对人体运动各要素的仿真提供了重要变量（如肌肉力和关节力）的估计。仿真还使我们能够建立因果关系，从而深入了解肌肉功能。仿真最令人兴奋的特性也许在于其开展“假设分析（what if）”研究的潜力，以检验假设、预测功能性结果并识别涌现行为。

本文重点介绍了使用 OpenSim [5] 的应用，包括以下项目：最小化不合理仿真的度量指标；确定新的动作作为降低损伤风险的体育训练工具；以及建立姿态、肌肉力和地面反作用力之间的关系。

## 2. 最小化不合理仿真的度量指标

## 2.1. 目标

在这一使用 OpenSim [5] 的应用中，我们通过调整计算肌肉控制（computed muscle control）[6] 的输入参数，利用优化方法来最小化不合理仿真的度量指标。我们的目标是确定最佳输入参数，以生成能够紧密跟踪地面跑步实验数据且残余力/力矩有限的仿真。

## 2.2. 背景与基础

计算肌肉控制是快速生成前向动力学仿真的常用选择 [6]。基于逆动力学的结果很难生成稳定的仿真；因此，计算肌肉控制结合使用静态优化以及前馈和反馈控制来跟踪期望的运动学。该跟踪算法需要几分钟来确定蹬踏运动的肌肉兴奋模式 [6]，以及几分钟来处理半个步态周期的数据 [7]。计算肌肉控制的高效性及其在 OpenSim [5] 中的可用性提高了其普及度。描述计算肌肉控制的三篇主要文章已被引用近 100 次 [5-7]。尽管有些研究报告了计算时间，但大多数并未报告两个相关的量：

x 个人生成合理仿真所需的时间（例如，紧密跟踪实验运动学并遵循将地面反作用力与身体各节段加速度关联起来的牛顿运动方程）。

x 不合理性的定量度量指标（例如，运动学跟踪误差以及平衡牛顿运动方程所需的残余力/力矩——残余力等同于实验误差和建模误差）。

鉴于我们在开发 OpenSim 软件 [5]、亲自使用计算肌肉控制（computed muscle control）以及与他人讨论其使用过程中的经验，工程师或临床医生生成合理的仿真需要耗费极多的时间。根据个人对合理性的容忍度，生成一次合理的仿真可能需要 1 到 3 天，甚至长达数月。在所有情况下，使用者都需要花费大量时间为计算肌肉控制选择过多不必要的输入参数，以生成能够将不合理性度量最小化的前向动力学仿真。

## 2.3. 研究描述

本研究所分析的受试者来自一组 34 名澳大利亚西部业余橄榄球男运动员，他们此前曾在澳大利亚珀斯的西澳大学（The University of Western Australia）接受了预期和非预期变向跑期间的运动分析 [8]。作为变向跑研究的常规组成部分，收集了运动分析数据，包括三维反光标记点轨迹以及地面反作用力和力矩（Fig. 1a）。受试者对采集和分析其运动数据签署了知情同意书。

由 37 个执行器（actuator）驱动的三维、全身、37 自由度（degree-of-freedom, DoF）骨骼模型构成了每次仿真的基础。骨盆相对于地面的位置和方向定义为 6 个 DoF。头部和躯干表示为一个刚性节段，通过 3 个 DoF 与骨盆相连。其余肢体关节的建模如下：每个髋关节为 3 个 DoF，每个膝关节为 3 个 DoF（其屈/伸轴随膝关节角度的变化而平移），每个踝关节为 1 个 DoF，每个肩关节为 3 个 DoF，每个肘关节和腕关节为 2 个 DoF。每个执行器均被建模为理想力或力矩（在软件内部的处理方式与任何其他执行器（如肌腱单元）相同）。该模型在 OpenSim [5] 中建立，并结合运动分析数据用于构建受试者特异性的动力学仿真（Fig. 1b）。

![](images/4603558623d62ccfcdc6fe1f102078142c6f0e0ae6840a4bb3557dbacb9a4377.jpg)  
Fig. 1. 本研究中的受试者为一名澳大利亚西部业余橄榄球男运动员。(a) 采集了地面跑步时的运动分析数据，包括三维标记点轨迹以及地面反作用力和力矩。(b) 通过三步流程构建了受试者的动力学仿真：1) 将由 37 个执行器驱动的 37 自由度骨骼模型缩放至受试者的体型；2) 逆向运动学确定与标记点数据相匹配的模型广义坐标值；3) 计算肌肉控制确定产生前向动力学仿真的一组最优激励。(c) 外层优化确定了计算肌肉控制的输入参数，这些参数在内层优化中用于生成合理的地面跑步前向动力学仿真

通过三步流程创建了受试者的动力学仿真。首先，对骨骼模型进行缩放，以表征实验测得的受试者体型尺寸。其次，利用逆运动学（inverse kinematics）分析获取模型的广义坐标值，使其与实验测得的受试者运动学数据紧密匹配。第三，应用计算肌肉控制（computed muscle control）来确定一组最优作动器激励，从而生成与实验测量运动学大体一致的前向动力学仿真。该三步流程用于构建受试者在地面跑步过程中从触地前（pre-contact）到离地（toe-off）阶段的仿真。

尽管每一步都使用了优化程序，但涉及计算肌肉控制的第三步通过嵌套或双层优化方法得到了改进（Fig. 1c）。内层优化使用计算肌肉控制及其相关输入参数（包括加速度跟踪权重、最大残余力/力矩以及最大关节力矩）生成仿真。外层优化则选取这些输入参数，以生成使不合理性指标最小化的仿真。

计算肌肉控制采用内层优化来生成仿真。内层代价函数为 OpenSim [5] 的默认实现。该内层代价函数 (1) 最小化了跨 $nq$ 个自由度（DoFs）的期望加速度 $( \ddot { q } _ { i } ^ { d e s i r e d } )$ 与前向动力学仿真加速度 $( \ddot { q } _ { i } ^ { s i m u l a t e d } )$ 之间的加权 $( w _ { \ddot { q } _ { l } } )$ 平方误差、跨 6 个残差的与其驱动激励 $( x _ { j } ^ { R } )$ 成比例并经其最大值 $( R _ { j } ^ { m a x } )$ 归一化的残余力/力矩 $( R _ { j } )$ 平方，以及跨 $nT$ 个力矩的与其驱动激励 $( x _ { k } ^ { T } )$ 成比例并经其最大值 $( T _ { k } ^ { m a x } )$ 归一化的关节力矩 $\left( T _ { k } \right)$ 平方：

$$
\min_{x^{R},  x^{T}}\left[\sum_{i = 1}^{nq}w_{\ddot{q}_{i}}\big(\ddot{q}_{i}^{desired} - \ddot{q}_{i}^{simulated}\big)^{2} + \sum_{j = 1}^{6}\left(\frac{R_{j}(x_{j}^{R})}{R_{j}^{max}}\right)^{2} + \sum_{k = 1}^{nT}\left(\frac{T_{k}(x_{k}^{T})}{T_{k}^{max}}\right)^{2}\right]\Bigg|_{\substack{inner\\ level}}\tag{1}
$$

设计变量是驱动每个残差力/力矩或关节力矩执行器的一组激励 $( \pmb { x } ^ { R } , \pmb { x } ^ { T } )$。这些激励被输入到前向动力学模型中，并且通过对牛顿运动方程进行数值积分，生成了每个时间帧的动力学仿真。仿真输出的运动学、残差力/力矩以及关节力矩均取决于输入参数的选择，包括每个加速度权重 $( w _ { \ddot { q } _ { l } } )$、最大残差 $( R _ { j } ^ { m a x } )$ 和最大关节力矩 $( T _ { k } ^ { m a x } )$。计算肌肉控制（computed muscle control）的输入参数选择决定了动力学仿真是否能够在有限的残差力/力矩下紧密追踪实验运动学数据，从而生成合理的仿真。

使用外层优化而非人类直觉来确定生成合理仿真的输入参数。外层代价函数将人工定义的输入参数数量从 $n q + 6 + n T$（在我们的模型中为 74）减少到仅 2 个。外层代价函数 (2) 最小化了 6 个骨盆自由度（DoF）上实验运动学 $( q _ { i j } ^ { e x p } )$ 与仿真运动学 $( q _ { i j } ^ { s i m } )$ 之间的均匀加权 $( W _ { p e l v i s }$，在本例中为 1000）平方误差、其余 ݊ݍ 个自由度上运动学的平方误差、6 个残差上的均匀加权 $( W _ { R }$，在本例中为 500）残差力/力矩平方 $( R _ { i k } )$，以及 ݊ܶ 个力矩上的关节力矩平方 $\left( T _ { i l } \right)$，每个物理量均在 ݂݊ 个时间帧内进行计算：

$$
\min_{\boldsymbol{w}_{\bar{q}},\boldsymbol{R}^{max},\boldsymbol{T}^{max}}\sum_{i = 1}^{nf}\left[W_{pelvis}\sum_{j = 1}^{6}\big(q_{ij}^{exp} - q_{ij}^{sim}\big)^{2} + \sum_{j = 7}^{nq}\big(q_{ij}^{exp} - q_{ij}^{sim}\big)^{2} + W_{R}\sum_{k = 1}^{6}R_{ik}{}^{2} + \sum_{l = 1}^{nT}T_{il}{}^{2}\right]\Bigg|_{\substack{outer\\ level}}\tag{2}
$$

设计变量是定义计算肌肉控制输入参数的一组加速度权重 $( { \pmb w } _ { \ddot { q } } )$、最大残差力/力矩 $( R ^ { m a x } )$ 和最大关节力矩 $( T ^ { m a x } )$。这些输入参数在整个内层优化中使用，以生成在有限残差力/力矩下紧密追踪实验运动学数据的动力学仿真。

## 2.4. 结果与讨论

找到了计算肌肉控制的最优输入参数（Fig. 2），所生成的模拟能够紧密跟踪地面跑步的实验数据（摆动肢髋关节内旋的 RMS 误差小于 $3 . 9 ^ { \circ }$），且具有极小的残差力（RMS 小于 0.3N）和力矩（RMS 小于 0.4Nm）。将优化后的模拟与由具有相同目标的 OpenSim [5] 用户生成的模拟进行对比时，RMS 跟踪误差最多增加了 $3 . 1 ^ { \circ }$（同样出现在摆动肢髋关节内旋）。在某些自由度（DoF）中，RMS 跟踪误差略有下降（骨盆垂直位置减小了 2.9mm）。这些跟踪误差的变化归因于残差力/力矩的显著降低（RMS 分别降低多达 151N 和 23Nm）。至于生成每次模拟所需的总时间，OpenSim [5] 用户在该问题上花费了约 3 个研究工作日，而优化所需的挂钟时间为 11.8 小时（耗时减少了约 50%；此外，在优化运行期间用户可以专注于其他研究，这进一步放大了记录时间上的差异）。

a  
![](images/84df2ef81ea91e7c4ed80a5e8cd4102115c13c71aadffa900a99c8016f3a1599.jpg)

b  
![](images/53752a725a9c120368edd3aebf6050809d8f09fc1cbae80f28990e0d9734423e.jpg)  
C

![](images/d05faa468bd159376d93deeb0df2c9fd2045ce3268102c5dddef7e26841e7230.jpg)

![](images/16f49b9abe790e6a34623af7ea1461643f53c748a7ff3d1a028770319bcce174.jpg)

d  
![](images/10402fb87301d29d9af7907530d6f3c260740fa62de33a5729de34e16356fe9e.jpg)

e  
![](images/de607cf71c3807ad59f2d33c5ba33e8d2ff0a0b38e4d581c8b2a2b7fd7540bf5.jpg)  
f

![](images/0bca756437b339caaef3727aa9e4a2bff6326485c09e091f2d7381bffa6b2f60.jpg)  
Fig. 2. 使用计算肌肉控制生成的模拟在优化前（蓝色，由 OpenSim [5] 用户完成）和优化后（红色）的结果中，按幅度降序排列的最大差异：(a) 运动学（加速度二次积分得到位置），(b) 残差力/力矩，以及 (c) 关节力矩。这些结果基于由 OpenSim [5] 用户（优化前情况）或优化器（优化后情况）所选取的输入：(d) 加速度权重，(e) 最大残差力/力矩，以及 (f) 最大关节力矩。

我们此前未预料到跟踪误差会出现较大幅度的增加（< 3.1º），并且对关节力矩的差异（< 15Nm）感到意外。这两处变化对于大幅减小残差力/力矩而言都是必需的。这些变化的幅度由我们在跟踪骨盆自由度权重 $( W _ { p e l \nu i s } = 1 0 0 0 )$ 和降低残差力/力矩权重 $( W _ { R } = 5 0 0 )$ 上的选择所决定。如果赋予较小的权重值，外层代价函数的最小化方式将发生改变，重点将被放在骨盆以外的跟踪误差以及降低关节力矩上。

通过优化改进计算肌肉控制（computed muscle control），使工程师和临床医生能够在合理的时间内生成合理的正向动力学仿真，且无需选择额外的输入参数。

## 3. 识别降低损伤风险的新动作

## 3.1. 目标

在这项使用 OpenSim [5] 的应用中，我们利用计算肌肉控制来识别能够在跑步变向过程中降低韧带损伤风险的新动作。我们的目标是确定在非预期侧向变向的承重阶段（weight-acceptance phase）能够降低膝关节峰值外翻力矩的运动学变化。

## 3.2. 背景与基础

前交叉韧带（Anterior cruciate ligament, ACL）损伤是代价最高的膝关节损伤之一，超过一半的此类损伤发生在运动过程中 [9]。在 ACL 重建手术 3 年后，70% 至 100% 的受伤运动员无法恢复到原有的竞技水平 [10]。大多数非接触性 ACL 损伤发生在变向或侧切变向动作中，这些动作会导致膝关节发生外翻塌陷 [11-14]。具体而言，与直线奔跑相比，侧向变向承重（weight acceptance, WA）阶段的膝关节外翻力矩显著升高 [15-17]。在非预期侧向变向情况下，膝关节峰值外翻力矩会进一步增加 [18]。因此，ACL 损伤预防训练干预应重点降低非预期侧向变向 WA 阶段的膝关节外翻载荷，因为这一阶段被认为是 ACL 损伤风险最高的时候。

肌骨模型结合优化方法已被用于确定步行过程中全身运动学与膝关节峰值力矩之间的因果关系 [19]。在 OpenSim [5] 中使用计算肌肉控制（该方法利用运动数据和优化方法来创建人体运动的动力学一致性仿真），可以使用类似的计算方法来确定侧向变向过程中全身运动学与膝关节外翻力矩之间的因果关系。此外，在非预期侧向变向的 WA 阶段中影响膝关节峰值外翻力矩的运动学变化目前尚不明确。

## 3.3. 研究描述

本研究中分析的 9 名受试者（年龄 19-30 岁；身高 1.83 ± 0.04 m；体重 80.8 ± 6.66 kg）来自上文第 2 节中所述的一组 34 名西澳大利亚业余澳式橄榄球（Western Australian Amateur Football）男性运动员。运动分析数据（包括三维标记点轨迹以及地面反作用力和力矩）作为侧向变向研究的常规组成部分进行了采集（Fig. 1a）。受试者对该运动数据的采集和分析签署了知情同意书。

如上文第 2 节所述，每个模拟的基础均基于一个由 37 个执行器驱动的三维全身 37 自由度（DoF）骨骼模型。该模型在 OpenSim [5] 中生成，并与动作分析数据结合使用，以创建受试者特异性的动态模拟（Fig. 1b）。这些非预期侧切动作（unanticipated sidestepping）的模拟是使用上文第 2 节所述的三步过程创建的。

采用计算肌肉控制（Computed muscle control）来求解一种运动学解，该解在追踪实验记录的关节坐标并保持与实验记录的地面反作用力（GRF）的动态一致性的同时，将膝关节外翻力矩峰值降至最低。为了实现这一目标，降低了与膝关节内翻/外翻自由度相关的最大可达力矩值。实际上，优化问题被重新表述，以在模拟过程中避免（或阻止）使用膝关节外翻力矩。比较了运动学优化前后所有 37 个自由度的角度和力矩差异。

## 3.4. 结果与讨论

在所有 9 次模拟中，运动学优化后膝关节外翻平均峰值力矩均显著降低 $( \mathtt { p } = 0 . 0 4 5 )$，平均降低了 44.2 Nm。其他膝关节平均峰值力矩平均增加了 24.1 Nm（屈曲）和 1.1 Nm（内旋），但这些变化在统计学上并不显著（Fig. 3a）。在运动学优化前后，每次模拟都产生了独特的改变以减小膝关节外翻峰值力矩（Fig. 3b）。然而，均使用了 9 个关键关节坐标的子集（支撑侧：踝关节跖屈、髋关节外展、肩关节内收和内旋；摆动侧：膝关节伸展、髋关节屈曲和外展、肩关节伸展；以及躯干旋转）来减小这些力矩。

a  
![](images/d2b714c9100fe600ed328f5f854846d1b1fb3496dbfc9ba28b13ee213e3a7e96.jpg)

b  
![](images/b4ee85a38ffc5d50790b7e42ed85b31a13de88433152a60496fe3e3f207ffab3.jpg)  
Fig. 3. (a) 在非预期侧切的承重反应（WA）阶段计算的运动学优化前后膝关节伸展、外翻和内旋的平均峰值力矩。符号 \* 表示随时间的显著变化（Į = 0.05）。(b) 9 次模拟中的 1 次在非预期侧切模拟的 WA 阶段中，所有 37 个模型自由度（DoF）从运动学优化前到优化后的角度运动学差异。

采用了两种主要的运动学策略来减小膝关节外翻峰值力矩。第一种策略在 9 次模拟中的 6 次中使用，使运动学优化前后的平均踝关节跖屈增加了 $7 . 9 \pm 5 . 2 ^ { \mathrm { ~ o ~ } }$。第二种运动学策略在全部 9 次模拟中使用，将全身质心相对于支撑脚向内侧和前方重新定位，这正是非预期侧切动作中预期的行进方向。

从教练的角度来看，将全身质心向内侧朝向预期的行进方向重新定位这一通用运动学策略，可以被认为是一种相对容易实施的策略。这项技术建议还需要运动员发展出新的运动控制策略，以成功掌握该技术。尽管先前的文献在多项个案研究中表明，针对特定患者的步态调整在减少冠状面膝关节峰值力矩方面是有效的 [19,22,23]，但在向普通运动和临床人群提出这些建议之前，仍必须在大型异质性人群中从临床上验证侧切步（sidestepping）过程中将全身质心引向内侧的有效性。

## 4. 建立姿势、肌肉力与地面反作用力之间的关系

## 4.1. 目标

在该使用 OpenSim [5] 的应用中，我们利用肌肉骨骼建模和优化方法来研究姿势、肌肉力与地面反作用力之间的关系。我们的目标是确定在整个步态支撑相中，姿势是否会影响肌肉在横断面上产生地面反作用力的能力。

## 4.2. 背景与基础

蹲伏步态（Crouch gait）是脑瘫患儿中常见的一种运动异常，由于步态支撑相期间膝关节和髋关节屈曲增加，导致步行效率降低 [24]。步行过程中过度的膝关节屈曲会使关节退化，并可能导致慢性膝关节疼痛 [25]，如果不加以治疗，这些症状可能会随时间推移而恶化 [26]。一些因素已被证实与蹲伏步态有关，包括肌无力、痉挛、僵硬和运动控制能力下降 [27]。尽管已经研究了数十年，但由于肌肉骨骼系统的复杂性，这些因素与蹲伏步态之间的因果关系在一定程度上仍不明确 [27]。

蹲伏步态通常被认为对脑瘫患者不利；然而，蹲伏姿势可能会提供生物力学上的优势，从而促使某些患者采用蹲伏步态。例如，运动员采取蹲伏姿势以增强向任意选定方向移动的能力。同样，站在行驶列车上的乘客采取这种姿势以增强抵抗移动的能力。在这两种情况下，横断面上的运动都是通过在横断面上产生地面反作用力来实现或抵抗的。

## 4.3. 研究描述

在 OpenSim [5] 中建立了一个具有 15 个自由度和 92 个肌腱作动器（muscle-tendon actuators）的三维肌肉骨骼模型。该模型由头部、躯干、骨盆以及左、右股骨、胫骨和足部刚体段组成。支撑足（在我们的研究中为右足）与地面固联。下肢关节的建模方式如下：距下关节和踝关节为旋转关节（revolute joints），双侧膝关节各为一个平面关节（planar joint），髋关节为球窝关节（ball-and-socket joint）。头部和躯干包含在模型中，并通过球窝关节与骨盆相连。

![](images/deeb10d97632a685968576a05ba3a084cd74663b41dd1a56740e5901e9cf8c94.jpg)  
Fig. 4. 具有 15 个自由度和 92 个肌腱作动器的三维全身肌肉骨骼模型，展示了其在 4 种（共 45 种）姿态下的状态以及横断面内的最大地面反作用力轮廓：(a) 实验直立姿态 [28]，(b) 实验直立与蹲伏数据之间的插值姿态，(c) 实验蹲伏姿态 [28]，(d) 由实验直立与蹲伏数据外推得到的姿态

该肌肉骨骼模型在步态周期的单腿支撑初期、中期和末期分别置于从直立到严重蹲伏的 15 种不同姿态下，共计 45 种姿态（Fig. 4）。直立姿态定义自 83 名健全受试者的平均步态数据 [28]。蹲伏姿态定义自 100 名患有脑性瘫痪（cerebral palsy）且呈屈膝步态受试者的平均步态数据 [28]。利用这些实验数据，我们在单腿支撑的三个不同阶段，在线性上插值了直立与蹲伏之间的 9 种姿态。此外，我们还外推了屈膝程度大于蹲伏姿态的另外 4 种姿态。

针对这 45 种不同的姿态，分别进行了一系列优化计算。优化器通过修改作用于模型上的肌肉力，最大化横断面内 8 个罗盘方向上的地面反作用力。每次优化均满足以下约束条件：1）压力中心位于支撑足下方，以及 2）垂直地面反作用力大于或等于零。通过求解来自 8 个罗盘方向的作用力所围成的面积，为每种姿态生成了地面反作用力轮廓。通过比较所有姿态下反作用力轮廓的面积，对我们的假设进行了评估。

## 4.4. 结果与讨论

在支撑相期间，一系列屈膝姿态允许产生最大面积的地面反作用力分布轮廓（Fig. 5）。在支撑中期之前，介于直立和屈膝之间的轻度屈膝姿态（#4-6）允许产生最大的地面反作用力分布轮廓。这些姿态产生的力分布面积相互之间的差异在 1% 以内，其中姿态 #5 最大。相比之下，直立（#1）和屈膝（#11）姿态的力分布面积小了 12-13%，而严重屈膝（#15）则小了大约 23%。根据我们之前的研究，支撑中期的屈膝姿态产生了最大面积的力分布 [29]。这一趋势一直持续到单腿支撑相末期。在支撑末期，介于屈膝和严重屈膝之间的姿态（#12）允许产生最大的地面反作用力分布；然而，与屈膝（#11）相比，该力分布面积仅大不到 2%。在支撑末期，屈膝的力分布面积比直立高 7.3%，比严重屈膝高 4%。

![](images/50b604e60fb93a2aa5d8e762faa31239d8c69b546abe14f6b85a7c0e89a6c6c8.jpg)  
Fig. 5. 曲面图（上方）与等高线图（底部）显示了支撑相三个阶段及所有姿态下的地面反作用力分布面积（初始-中期-末期之间的中间力分布面积通过三次样条插值生成）。整个支撑相期间的力分布面积从最低（蓝色）到最高（红色）。在支撑早期，轻度屈膝姿态（#4-6）允许产生最大的力。在支撑晚期，与直立相比，屈膝姿态（#9-11）允许产生更大的力。

从轻度屈膝（用于支撑初期）到屈膝（用于支撑末期）的姿态整体上允许产生更大的力分布面积。为了最大化肌肉在横断面产生地面反作用力的能力，人们必须在整个步态支撑相中从轻度屈膝过渡到屈膝姿态（Fig. 5，红色区域）。

尽管本研究拓展了我们之前的研究 [29]，但这项工作与 Hicks et al. [28] 的研究有着根本的不同，后者考察了屈膝姿态对肌肉伸展髋关节和膝关节能力的影响。他们使用诱发加速度分析（induced acceleration analysis）来确定向模型施加 1 N 肌肉力所引起的伸展方向的关节角加速度。他们的研究表明，在屈膝步态中，几乎所有主要髋伸肌和膝伸肌的能力都有所下降。这一发现表明产生垂直地面反作用力的能力有所降低。在我们的研究中，我们使用优化算法来最大化水平地面反作用力，而不考虑垂直地面反作用力。我们的发现表明产生这些水平力的能力有所增加。

## 5. 挑战与下一步工作

尽管人体运动仿真的前景令人振奋，但其面临的挑战同样不容忽视。数十年来，实验和物理方法加深了我们对人体运动控制、关节运动、肌肉力量以及人体机能容量的理解。然而，研究进展本质上受到两个因素的限制：1）关键变量（如肌肉力）通常无法在实验中直接测量；2）因果关系（如肌肉对运动的贡献）难以通过实验来确立。肌肉驱动的动力学仿真通过估算关键变量并阐明因果关系，从而提供了对肌肉功能的深入见解，有力地补充了实验和物理方法。下文概述了在使用 OpenSim 进行与上述应用类似的仿真时所涉及的一些关键问题。

首先，尽管仿真在根本上依赖于实验，但在将实验信息整合到仿真软件中的尝试一直存在重大局限。许多公司生产并销售实验测量系统，这些系统使用各种不同格式记录体格检查和运动分析信息。许多实验室开发了自己的专用软件，以根据其自身的实验信息生成仿真，但这些专有软件并不与其他实验室共享。开发支撑这些仿真的肌肉驱动动力学模型极具挑战性，大多数实验室缺乏构建自己模型所需的资源或专业技术。因此，大多数仿真都依赖于“通用”模型，这些模型基于从数量有限、具有正常肌肉骨骼解剖结构的成年人尸体标本中测得的实验数据。为了克服这些局限性，必须开发能够处理各种实验信息的工具，生成用于解决涉及模型优化以匹配这些信息的复杂问题的新算法，并建立一个整合实验、数据和计算的通用框架以推进深入研究。

其次，目前缺乏免费、开源的计算工具来对生物力学系统进行建模与仿真，并开发稳健的设计与控制算法。建模和仿真软件使生物力学研究人员能够查看模型、编辑肌肉、生成仿真并分析生物力学系统。另一方面，数学计算软件使工程师和科学家能够开发算法、分析数据、可视化结果并执行数值计算。将这两类软件工具相结合，不仅能加深对神经肌肉生物力学的理解，还能为优化手术和康复治疗方案的制定提供预测能力，从而使人体运动研究和运动异常治疗从中极大受益。

第三，为了预测功能性结果，必须解答“如何选择合适的肌肉模式以实现行为目标”这一悬而未决的问题。通常，仿真并未纳入感觉器官或提供运动控制所必需的反馈机制。事实上，在这方面许多仿真甚至没有包含中枢神经系统的大部分结构。综合神经控制与目的性运动之间的复杂转换是一个艰巨的障碍。整合实验运动数据、新型控制系统和大规模并行算法的软件组件，对于加速人体运动控制的多学科研究是必不可少的。

建模与仿真技术需要取得显著进展，以充分发挥基于仿真的个性化医疗的全部潜力。在医师、物理治疗师和康复科学家所采用的实验与物理方法，与工程师、数学家和计算机科学家所使用的计算机仿真方法之间，目前仍存在差距。自半个多世纪前起步微末以来，计算机仿真已在各个工程领域产生了巨大的影响。若要在人体运动领域产生类似的影响，临床医生与工程师之间必须建立起共生关系。这种协同关系不仅能够产生最佳的仿真结果，而且具有彻底变革人体运动多学科研究的潜力。

## 致谢

作者感谢 Cyril Donnelly、Hoa Hoang、Ashley Clark、Samuel Hamner 和 Jennifer Hicks 提供的宝贵贡献与有益讨论。本项工作得到了 The University of Tennessee（田纳西大学）、NIH roadmap for Medical Research U54 GM072 970、通过 NHMRC 400937 资助的 PAFIX、Canadian Society of Biomechanics（加拿大生物力学学会）以及 The University of Western Australia’s Convocation Office（西澳大学校友会办公室）的支持。

## 参考文献

[1] Robertson G, Caldwell G, Hamill J, Kamen G, Whittlesey S, editors. Research Methods in Biomechanics. Champaign: Human Kinetics; 2004.

[2] Murray WM, Delp SL, Buchanan TS. Variation of muscle moment arms with elbow and forearm position. Journal of 1996; :513-25.

[3] Arnold AS, Salinas S, Asakawa DJ, Delp SL. Accuracy of muscle moment arms estimated from MRI-based musculoskeletal models of the lower extremity. Computer Aided Surgery 2000;5:108-19.

[4] Newton I. Philosophiae Naturalis Principia Mathematica. London; 1687.

[5] Delp SL, Anderson FC, Arnold AS, Loan P, Habib A, John CT, Guendelman E, and Thelen DG. OpenSim: Open-source software to create and analyze dynamic simulations of movement. IEEE Transactions on Biomedical Engineering 2007;55:1940-50.

[6] Thelen DG, Anderson FC, and Delp SL. Generating dynamic simulations of movement using computed muscle control. Journal of Biomechanics 2003;36:321-8.

[7] Thelen DG and Anderson FC. Using computed muscle control to generate forward dynamic simulations of human walking from experimental data. Journal of Biomechanics 2006;39:1107-15.

[8] Donnelly CJ, Doyle TLA, Finch CF, Elliott BC, and Lloyd DG. The influence of balance and technique training on knee loading and risk of ACL injury during sidestepping. In Proceeding of the XXII Congress of the International Society of Biomechanics, Cape Town, South Africa, July 5-9, 2009.

[9] Gianotti SM, Marshall SW, Hume PA, Bunt L. Incidence of anterior cruciate ligament injury and other knee ligament injuries: a national population-based study. Journal of Science and Medicine in Sport 2009;12:622-7.

[10] Roos H, Ornell M, Gardsell P, Lohmander LS, Lindstrand A. Soccer after anterior cruciate ligament injury--an incompatible combination? A national survey of incidence and risk factors and a 7-year follow-up of 310 players. Acta 1995; :107-12.

[11] Andersen TE, Tenga A, Engebretsen L, Bahr R. Video analysis of injuries and incidents in norwegian professional football. British Journal of Sports Medicine 2004; :626-31.

[12] Cochrane JL, Lloyd DG, Buttfield A, Seward H, McGivern J. Characteristics of anterior cruciate ligament injuries in australian football. Journal of Science and Medicine in Sport 2007;10:96-104.

[13] Koga H, Nakamae A, Shima Y, Iwasa J, Myklebust G, Engebretsen L, Bahr R, Krosshaug T. Mechanisms for noncontact anterior cruciate ligament injuries: Knee joint kinematics in 10 injury situations from female team handball and basketball. 2010; :2218-25.

[14] Krosshaug T, Nakamae A, Boden BP, Engebretsen L, Smith G, Slauterbeck JR, Hewett TE, Bahr R. Mechanisms of anterior cruciate ligament injury in basketball: Video analysis of 39 cases. American Journal of Sports Medicine2007;35:359-67.

[15] Besier TF, Lloyd DG, Cochrane JL, Ackland TR. External loading of the knee joint during running and cutting maneuvers. Medicine & Science in Sports & Exercise 2001;33:1168-75.

[16] Cochrane JL, Lloyd DG, Besier TF, Elliott BC, Doyle TL, Ackland,TR. Training affects knee kinematics and kinetics in cutting maneuvers in sport.  2010; :1535-44

[17] Dempsey AR, Lloyd DG, Elliott BC, Steele JR, Munro BJ, Russo KA. The effect of technique change on knee loads during sidestep cutting. Medicine & Science in Sports & Exercise 2007;39:1765-73.

[18] Besier TF, Lloyd DG, Ackland TR, Cochrane JL. Anticipatory effects on knee joint loading during running and cutting maneuvers.  2001; :1176-81.

[19] Fregly BJ, Reinbolt JA, Rooney KL, Mitchell KH, Chmielewski TL. Design of patient-specific gait modifications for knee osteoarthritis rehabilitation. IEEE Transactions on Biomedical Engineering 2007;54:1687-95.

[20] Donnelly CJ, Lloyd DG, Elliott BC, and Reinbolt JA. Minimizing valgus knee loading during sidestepping: implications for ACL injury risk. In Proceedings of the 6th World Congress on Biomechanics, Suntec City, Singapore, August 1-6, 2010.

[21] Hamner SR, Seth A, Delp SL. Muscle contributions to propulsion and support during running. Journal of Biomechanics 2010;43:2709-16.

[22] Schache AG, Fregly BJ, Crossley KM, Hinman RS, Pandy MG. The effect of gait modification on the external knee adduction moment is reference frame dependent. Clinical Biomechanics 2008;23:601-8.

[23] Walter JP, D'Lima DD, Colwell CW, Jr, Fregly BJ. Decreased knee adduction moment does not guarantee decreased medial contact force during gait. Journal of Orthopaedic Research 2010;28:1348-54.

[24] Wren TA, Rethlefsen S, Kay RM. Prevalence of specific gait abnormalities in children with cerebral palsy: influence of cerebral palsy subtype, age, and previous surgery. Journal of Pediatric Orthopaedics 2005;25:79-83.

[25] Rose J, Gamble JG, Burgos A, Medeiros J, Haskell WL. Energy expenditure index of walking for normal children and for children with cerebral palsy. Developmental Medicine and Child Neurology 1990;32:333-40

[26] Bell KJ, Ounpuu S, DeLuca PA, Romness MJ. Natural progression of gait in children with cerebral palsy. Journal of Pediatric Orthopaedics 2002;22:677-82.

[27] Arnold AS, Anderson FC, Pandy MG, and Delp SL. Muscular contributions to hip and knee extension during the single limb stance phase of normal gait: a framework for investigating the causes of crouch gait. Journal of Biomechanics 2005;38:2181-9.

[28] Hicks JL, Schwartz M, Arnold AS, and Delp SL. Crouched postures reduce the capacity of muscles to extend the hip and knee during the single-limb stance phase of gait. Journal of Biomechanics 2008;41:960-7

[29] Hoang HX, and Reinbolt JA. Posture influences ground reaction force: implications for crouch gait. In Proceedings of 3D Analysis of Human Movement, San Francisco, USA, July 14-16, 2010.