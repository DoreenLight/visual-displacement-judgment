![](images/2e2aa9923fa87ce29cb05032e529cb596128533350046e76df0a7e3a9c8253cc.jpg)

研究论文

# OpenSim Moco: Musculoskeletal optimal control

Christopher L. Dembia<sub>ID</sub> 1☯ , Nicholas A. Bianco<sub>ID</sub><sup>1☯</sup>, Antoine Falisse<sub>ID</sub> <sup>2,3</sup>, Jennifer L. Hicks<sub>ID</sub><sup>3</sup>, Scott L. Delp<sub>ID</sub><sup>1,3,4</sup>

![](images/47aa35eb9154dcf4d0448eeb7021374569fd1ca52786c0d5e4e158770a75954a.jpg)

## 开放获取（OPEN ACCESS）

引用格式：Dembia CL, Bianco NA, Falisse A, Hicks JL, Delp SL (2020) OpenSim Moco: Musculoskeletal optimal control. PLoS Comput Biol 16(12): e1008493. https://doi.org/10.1371/ journal.pcbi.1008493

编辑：Kenneth S. Campbell, University of Kentucky, UNITED STATES

收稿日期：2020年5月20日

录用日期：2020年11月5日

发表日期：2020年12月28日

版权所有：© 2020 Dembia et al. 这是一篇采用 Creative Commons Attribution License 授权条款发布的开放获取文章，只要注明原始作者和来源，即可在任何媒介上无限制地使用、分发和复制。

数据可用性声明：所有数据均包含在文稿中。复现结果所需的所有文件均可在 https://github.com/ stanfordnmbl/mocopaper 获取。OpenSim Moco 软件可从 https://opensim. stanford.edu/moco 免费获取。Moco 的源代码可从 https://github.com/opensim-org/ opensim-moco 获取。

资助信息：CLD、NAB、AF、JLH 和 SLD 获得了来自美国国家卫生研究院（National Institutes of Health，https://www.nih.gov）的资助，基金号为 U54 EB020405、P2C HD065690、P2C HD101913 和 P41 EB027060。

1 斯坦福大学机械工程系（Department of Mechanical Engineering, Stanford University, Stanford, California, United States of America），2 鲁汶大学运动科学系（Department of Movement Sciences, KU Leuven, Leuven, Belgium），3 斯坦福大学生物工程系（Department of Bioengineering, Stanford University, Stanford, California, United States of America），4 斯坦福大学骨外科系（Department of Orthopaedic Surgery, Stanford University, Stanford, California, United States of America）

☯ 这些作者对这项工作具有同等贡献。

\* dembia@alumni.stanford.edu

## 摘要

肌肉骨骼仿真被广泛应用于许多不同领域，从与人类交互的可穿戴机器人设计，到运动功能受损患者的分析。在此，我们介绍了 OpenSim Moco，这是一个用于优化在 OpenSim 建模与仿真软件包中构建的肌肉骨骼模型的运动与控制的软件工具包。OpenSim Moco 采用直接配点法（direct collocation method），与其他肌肉骨骼仿真方法相比，该方法通常速度更快，且能够处理更多样化的问题。Moco 使研究人员无需自行实现直接配点法——这通常需要深厚的技术专业知识——从而使他们能够专注于自己的科学问题。该软件可以处理生物力学学者感兴趣的广泛问题，包括运动追踪、运动预测、参数优化、模型拟合、肌电驱动仿真以及设备设计。Moco 是首个能够处理运动学约束的肌肉骨骼直接配点工具，这些约束使得对运动学闭环（例如自行车骑行模型）和复杂解剖结构（例如髌骨运动）进行建模成为可能。为了展示 Moco 的能力，我们首先求解了在最小化肌肉兴奋度平方和膝关节负荷的同时产生所观察到的行走运动的肌肉活动。接着，我们预测了肌肉无力如何导致偏离正常行走运动。最后，我们预测了蹲起运动（squat-to-stand motion），并优化了置于膝部的助力装置的刚度。我们将 Moco 设计为易于使用、可定制和可扩展的工具，从而加速利用仿真来理解人类和其他动物的运动。

## 作者概要

计算机仿真已成为研究肌肉骨骼系统日益普及的工具。仿真被用于研究肌肉在行走和跑步中的作用，分析神经系统疾病患者的步态，以及设计假肢和外骨骼。从历史上看，研究人员依赖实验数据来生成仿真并估计肌肉活动。基于直接

CLD 和 NAB 获得了国家科学基金会研究生研究奖学金计划（National Science Foundation Graduate Research Fellowship Program）的资助。CLD 获得了斯坦福 Bio-X 研究生奖学金（Stanford Bio-X Graduate Fellowship，https://biox.stanford.edu）的资助。NAB 获得了斯坦福研究生奖学金计划（Stanford Graduate Fellowship Program，https://vpge.stanford.edu/fellowships-funding/sgf）的资助。AF 获得了法兰德斯研究基金会（Research Foundation Flanders，https://www.fwo.be）博士奖学金 1S35416N 和差旅资助 V441717N 的资助。出资方在研究设计、数据收集与分析、发表决定或手稿准备中未参与任何工作。

利益冲突：作者声明不存在竞争性利益。

配点最优控制方法使研究人员不仅能够估算肌肉活动，而且无需实验数据即可预测新的运动。然而，直接配点法实现难度大且耗时，并需要最优控制和优化理论方面的专业知识。在此，我们介绍了 OpenSim Moco，这是一个开源软件包，使没有最优控制背景的人也能进行新运动的预测。Moco 充分利用了 OpenSim 肌肉骨骼建模软件包提供的现有建模工具，并提供了一个易于使用的接口，便于生成和共享仿真流程。Moco 具有模块化和易扩展的特点，并包含一个针对已知解问题进行求解的测试套件。我们提供了若干示例，包括预测使膝关节负荷最小化的肌肉活动、预测肌无力如何影响正常行走，以及优化膝关节外骨骼以辅助深蹲起立运动。

这是一篇 PLOS Computational Biology 软件论文。

## 引言

肌肉骨骼仿真揭示了运动障碍的机制，例如，发现了能够减少膝关节负荷的行走方式 [1]，揭示了脑瘫儿童在行走时表现出简化的运动控制 [2]，以及再现了导致复视的眼部疾病 [3]。除了人类健康与机能表现之外，研究人员还利用肌肉骨骼仿真来了解非人动物的运动方式，例如通过研究恐龙的运动 [4] 以及人类与黑猩猩力量的差异 [5]。

运动仿真通常根据运动是由实验数据规定还是由仿真预测来进行分类。研究者可以规定肌肉骨骼模型（musculoskeletal model）的运动以匹配实验中测得的运动 [6, 7]，从而估算未测量的物理量，例如肌肉层面的能量消耗 [8, 9]。最常用的规定运动方法计算速度快，但缺乏对重要模型特征（如肌肉动力学）的支持（例如静态优化 [10]）。运动预测 [11] 则可以建立因果关系（例如，发现由肌无力或挛缩引起的步态损伤 [12]），并有助于设计临床干预方案（例如假肢 [13] 和外骨骼 [14]）。然而，一些预测运动的方法（如单打靶法，single shooting）通常需要等待数小时甚至数天才能求解，因此研究人员往往会简化模型的肌肉系统或降低控制方案的维度 [12, 15]。介于规定运动和预测运动之间的第三类是跟踪运动，其中模型运动学与参考数据之间的误差被作为代价函数的一部分，而不是被精确规定 [16]。虽然一些研究人员编写了自己的代码来解决涵盖从规定到预测整个谱系的问题，但许多研究人员缺乏实现自定义算法所需的专业知识或资源。能够解决多样化问题的仿真方法，结合使所有研究人员都易于采用这些方法的工具，将加速仿真在科学和临床问题中的应用。

大多数肌肉骨骼仿真自然地被表述为最优控制问题 [17]。最优控制问题旨在寻找系统的参数和时变控制，以在受系统动力学约束的前提下最小化代价（例如能量消耗），表示为

微分代数方程。生物力学研究人员通常使用单打靶法（single shooting）来求解最优控制问题，但存在更快速的替代方法。直接多打靶法（direct multiple shooting）利用时间步进数值积分来模拟轨迹的连续区间，并约束相邻区间的端点保持一致 [18, 19]，与单打靶法相比，这提高了数值稳定性并加快了收敛速度。直接配点法（direct collocation）是一种越来越受欢迎的方法，它避免了时间步进积分的需求，并且与直接多打靶法相比，能够在精度和计算成本之间更轻松地配置权衡。采用该方法时，系统的状态和控制量在时间点网格上被近似为多项式样条，优化器求解节点以使样条遵循系统动力学 [20–23]。动力学约束是通过要求状态样条的时间导数在指定时间点与系统微分方程得出的导数相匹配来实现的。（该方法被称为“直接配点法”，因为样条导数与精确导数在这些点上“并置/配点”([24], 第 211 页; [25], 第 498 页)。）直接配点法会生成一个非线性规划问题，其中状态被作为变量引入，而系统动力学则作为约束条件强制执行。典型的肌肉骨骼模型会导致包含数千个变量的优化问题。这些大规模问题在直接配点法下是可处理的，因为在给定时间强制执行系统动力学的约束条件仅取决于该时间附近的变量。

直接配点法的优势促使生物力学研究人员将其用于规定运动 [26, 27]、跟踪运动 [16, 28–31]、预测运动 [32–44]、拟合肌肉特性 [45] 以及优化模型参数 [46]。研究人员取得了关键的方法学进展，包括通过隐式公式高效处理多体和肌肉动力学 [26, 47]、最小化能量消耗 [48, 49]，以及采用算法微分（algorithmic differentiation）与使用有限差分相比更快速地模拟复杂模型 [50]。

尽管直接配置法（direct collocation）具有诸多优势，但极少有生物力学实验室能够应用这一强大的技术。该方法需要对变量进行繁琐的记录追踪，并对基于梯度的优化算法所需的目标函数和约束函数导数进行高效计算。目前虽存在若干直接配置求解器（例如 [51, 52]），但现有的求解器要求用户手动整合其肌骨模型。OpenSim 是一款被数千名生物力学研究人员用于构建肌骨系统模型的软件包 [53–55]，但 OpenSim 目前并未采用直接配置法。一些生物力学学者慷慨地分享了他们将 OpenSim 或手写代码模型与直接配置求解器相结合的代码，但这些代码要么是针对特定模型或问题表述量身定制的，要么缺乏对具有运动学约束模型的支持 [26, 35, 36, 40]。此外，此类代码可能依赖闭源组件，从而限制了自定义扩展 [26]。最后，选择能够实现快速收敛的问题表述（例如将动力学表示为显式或隐式微分方程）和求解器设置（例如自动检测稀疏模式）需要专业知识；理想情况下，此类专业知识应通过默认设置嵌入到软件中，使用户可以通过简单的命令修改其问题表述或求解器设置。

为了提高肌骨生物力学领域中先进最优控制方法的可及性，我们推出了 OpenSim Moco（“肌骨最优控制”，musculoskeletal optimal control），这是一个易于使用、可定制且可扩展的软件工具包，用于求解基于 OpenSim 肌骨模型的最优控制问题（Fig 1）。OpenSim 使生物力学学者免于自行实现运动方程，而 Moco 则使生物力学学者免于自行实现直接配置法。Moco 不仅消除了设置基于梯度的优化的需求，还提供了一个抽象掉构建最优控制问题细节的接口。如果 Moco 没有提供所需的功能，用户还可以添加自定义的代价项或约束。

![](images/95f8e85709e40c969424e3fc6b27190f1db91b2d98aa21f8e335fb4f8fb8ea8d.jpg)  
Fig 1. Moco 概述。在给定运动过程中要达成的目标和参考数据的情况下，OpenSim Moco 可为 OpenSim 肌骨模型生成最佳运动和肌肉行为 [54]。Moco 提供了一个目标库，例如最小化能耗努力（以间接测热面罩示意）、与标记点数据（或广义坐标数据）的偏差以及关节负荷。运动参考数据（标记点或广义坐标）、外力（来自测力台）和肌肉活动（来自肌电图）是可选的。插图来源：Kai Rasmussen。

https://doi.org/10.1371/journal.pcbi.1008493.g001

本文详细介绍了 Moco 的设计与实现，并展示了三个应用。首先，我们估计了观察到的行走动作中的肌肉活动，并预测了在不同代价函数下肌肉活动的变化。其次，我们使用 Moco 预测了肌肉无力可能如何导致偏离正常行走动作。最后，为了说明 Moco 能够预测运动和模型参数，我们预测了从深蹲到站立的动作，并优化了一种无源辅助设备的刚度。

## 设计与实现

研究人员可以使用 Moco 求解通过代价和约束模块库定义的最优控制问题，这些模块是通过可配置的软件类实现的。用户使用 *MocoProblem* 类来描述其问题。（我们使用斜体表示 Moco 和 OpenSim 中的类名。）为了将问题与用于求解它的数值方法解耦，我们使用了 *MocoSolver* 类。Moco 的类可通过 C++、MATLAB、Python 和 XML 文本文件使用，其接口对于 OpenSim 用户来说十分熟悉。我们将 *MocoProblem* 和 *MocoSolver* 一起打包到 *MocoStudy*（Fig 2）中，它可以写入 XML 文本文件并从中加载。Moco 包含用于可视化和绘制研究解的实用工具，该解由 *MocoSolution* 类保存。对于某些标准的生物力学问题，Moco 提供了更简单的接口，这可能比 *MocoStudy* 的灵活性更具优势。

## 使用 MocoProblem 定义问题

*MocoProblem* 支持多种科学问题，并包含以下要素。

![](images/891cc87615349a10758e7a6eabe5c9c1e5f06cbab722933c5638af89a4627464.jpg)  
Fig 2. MocoStudy 概述。研究人员可以使用 Moco 通过代价、边界约束和路径约束模块库来求解自定义的最优控制问题。除了此处展示的内容外，Moco 还包含其他代价模块，且用户可以定义自己的自定义模块。  
https://doi.org/10.1371/journal.pcbi.1008493.g002

• 代价项（cost terms）：用户可以通过向 *MocoProblem* 追加与代价模块关联的类实例，来最小化控制量的加权平方和、与观察运动的偏差、关节反作用载荷、运动持续时间以及其他代价（例如，*MocoControlGoal* 允许最小化控制量的平方和）。

• 多体动力学、肌肉动力学和运动学约束：OpenSim 模型是用于肌骨系统建模的常用格式。Moco 使用 OpenSim 模型来获取底层的微分代数方程组，包括多体动力学、辅助动力学（例如肌肉激活动力学和肌腱顺应性）以及运动学约束；这些方程由工业级 Simbody 多体动力学库提供 [55]。Moco 采用 Posa 及其同事提出的一种方法来支持系统中存在的任何运动学约束 [56]。运动学约束对于模拟不易使用标准关节描述的解剖结构非常有用，否则这些解剖结构需要经过校准的韧带和软骨模型才能产生所需运动的显著特征。通过运动学约束实现具有复杂解剖结构的模型示例包括膝关节、肩关节和颈部模型 [57–60]。在对闭式运动链进行建模时，运动学约束也是必不可少的。例如，若双脚固定在同一个自行车曲柄上，骑行模型就会产生闭式运动链 [44]。

• 边界约束：用户可以通过关联初始和终止状态的约束来强制施加平均速度、对称性或周期性。

• 路径约束：用户可以约束任何时间函数在整个运动过程中处于指定范围内。例如，研究人员通常利用肌电图（electromyography）估计肌肉活动，而 Moco 允许通过 MocoControlBoundConstraint 类约束模拟的肌肉兴奋度接近这些测量值。

• 参数优化：用户可以优化模型属性，例如刚体的质量、肌肉的最佳肌纤维长度或外骨骼的刚度。

• 变量边界：用户可以对状态量、控制量以及初始和终止时间的取值进行边界约束。

关于 Moco 支持的问题的详细数学描述（包括 Moco 如何处理运动学约束），请参阅 S1 Appendix。

用户可以以多种方式组合 MocoProblem 的各个模块。以下示例说明了用户可以解决的问题的广泛性。

• 动力学约束下的逆运动学：在服从多体动力学的同时，最小化实验与模型标记点位置之间的误差（标记点追踪）以及控制量的平方。

• 肌电约束下的肌力估计：对于指定的实验运动，在服从多体动力学并约束肌肉兴奋度与肌电图数据之间差异的同时，最小化肌肉兴奋度的平方。

• 躯干质量校准：对于指定的实验运动，在服从多体动力学的同时调整躯干质量，以最小化骨盆处的残余力。

• 预测穿戴外骨骼行走时运动学和肌肉协调的适应性：在服从多体和肌肉动力学以及质心平均速度约束的同时，最小化肌肉兴奋度的平方。

我们在 Fig 3 中通过一个简单的 MATLAB 示例展示了 Moco 接口。设置模型和添加代价（在代码中称为“goal”）各自仅需要一条语句。

直接配点法依赖于基于梯度的优化，当问题中的所有函数都是连续且可微时，该方法收敛更快且更可靠。Moco 包含一个先前发表的连续且可微的肌肉模型，名为 DeGrooteFregly2016-Muscle [26]。我们扩展了该肌肉模型以包含阻尼，这进一步改善了收敛性，但在低激活度下肌纤维缩短时可能会引入微小的压力。为了以连续和平滑的方式表示柔顺接触力，Moco 包含一个先前发表的接触模型，名为 SmoothSphereHalfSpaceForce [61]。

希望使用 Moco 未提供的代价项、边界约束或路径约束的用户，可以按照与 OpenSim 插件相同的步骤创建 C++ 插件。通过提供代价、边界约束和路径约束模块库，允许组合这些模块，并支持用户创建自己的模块，我们实现了易用性、可定制性和可扩展性的设计目标。

## 使用 MocoSolver 求解问题

求解最优控制问题的所有细节均封装在 MocoSolver 中，它与 MocoProblem 解耦以提供灵活性。例如，用户可以在不修改求解器的情况下向其模型添加刚体或肌肉。MocoSolver 使用 CasADi 库 [62] 将 MocoProblem 定义的连续最优控制问题转录为有限维非线性规划，我们使用成熟的基于梯度的非线性规划求解器（如 IPOPT [63] 和 SNOPT [64]）来求解该规划（见 S1 Appendix）。CasADi 可以使用有限差分或算法微分提供目标函数和约束函数的导数；Moco 仅使用有限差分，以避免为了支持算法微分而调整 OpenSim 代码库带来的复杂性。

![](images/8c9ece1fe2738296a2c0d542599e8e213c0a083cc5b08e56b915dab6455547e5.jpg)

```matlab
study = MocoStudy();
problem = study.updProblem();
problem.addGoal(MocoFinalTimeGoal()); % Minimum-time problem.
problem.setModel(Model('sliding_mass.osim')); % Load an OpenSim model from a file.
problem.setTimeBounds(0, [0, 5]); % initial time = 0, final time <= 5 seconds.
problem.setStateInfo('/slider/position/value', [-5, 5], 0, 1); % Move 1 meter.
problem.setStateInfo('/slider/position/speed', [-50, 50], 0, 0); % Start and end at rest.
problem.setControlInfo('/force', [-50, 50]);
solution = study.solve();
```

Moco 提供了两种转录方案：二阶梯形格式和三阶 Hermite–Simpson 格式 [20]。多体动力学既可以用显式微分方程（“正向动力学”）表达，也可以用隐式微分方程（“逆向动力学”）表达；使用隐式微分方程时问题可能会收敛得更快 [26, 47]。Moco-Solver 还提供了约束容差、收敛容差以及用于求解 MocoProblem 的网格区间数量等设置。

求解 MocoStudy 会生成一个 MocoSolution（Fig 4），它是 MocoTrajectory 的一个子类，使用户能够方便地访问优化过程中任意迭代步骤的所有变量值。用户通过 MocoTrajectory 提供初始猜测值，并且可以将一个问题的解作为后续问题的初始猜测值；这使得用户能够通过一系列较简单的研究来构建复杂的研究。例如，预测踝关节外骨骼引起的行走运动学改变，可以受益于从正常行走的跟踪模拟中生成的初始猜测值。MocoSolution 还提供了额外的信息，包括求解器是否收敛、最终的目标函数值以及求解器的迭代次数。

在求解问题后，研究人员可以使用 Moco 将解可视化为动画，绘制状态和控制轨迹（利用 MATLAB 和 Python 中提供的实用工具），或根据解计算相关物理量。将 MocoTrajectories 和 MocoStudies 保存为文件的功能使用户能够复现彼此的结果，这对于可靠的科学研究至关重要 [65]。

![](images/b6e77d1092e90b56166c951573d6783934a7c652aa62e04364fefaecfd03f9ac.jpg)  
Fig 4. 使用轨迹迭代求解问题。优化的猜测值使用 MocoTrajectory 指定，该对象保存了优化过程中任意迭代步骤的状态、控制和参数值。MocoSolution 是 MocoTrajectory 的子类，保存了研究的解，并包含优化的成功状态、最终目标函数值以及求解器迭代次数。用户可以将一个问题的解用作后续问题的初始猜测值

![](images/85ef667afd02c93114e208874decadda86cef1448faf7bc1bc5101804d7e1905.jpg)  
Fig 5. 求解规定运动、跟踪运动和预测运动问题。Moco 提供了工具 MocoTrack 和 MocoInverse 用于求解标准问题。两者都需要模型（Model）和运动学数据作为输入，并输出控制和执行器状态，但这些工具求解的是不同的最优控制问题。MocoTrack 会生成一个新的模拟运动，而 MocoInverse 则不允许偏离所提供的运动学数据。预测运动不易标准化，需要定制的 MocoStudy。  
https://doi.org/10.1371/journal.pcbi.1008493.g005

## 标准问题工具

目前，Moco 提供了两个用于求解标准问题的工具（Fig 5）：

• MocoInverse 解决肌肉/执行器冗余问题，即在最小化控制平方或其他代价的同时，求解实现精确规定运动（见 S1 Appendix）的肌肉（或其他执行器）控制量。

• MocoTrack 解决运动追踪问题，即除了控制平方或其他代价之外，同时求解使与观测运动相比的误差最小化的运动以及肌肉（或其他执行器）控制量。

MocoTrack 适用于预测相对于运动数据的偏差（例如，预测对外骨骼的运动学适应），而当需要精确强制执行运动时（例如，估计观测运动中的弹性储能），MocoInverse 是一种更快速的选择。MocoTrack 可以使用接触模型，而 MocoInverse 可以将测量的外力应用于模型。对于这两种工具，唯一必需的输入是 OpenSim 模型和运动数据（坐标或标记点轨迹，以及相关的外力数据）。在内部，这些工具构建一个 MocoStudy，并配置在我们测试的问题上能够实现快速且可靠收敛的求解器设置。

![](images/939c7f2b57893c5f890cb32f5e7467cafc2c8dba6b2e48203fc4d4af5604ca2a.jpg)

未来版本的 Moco 可能会包含用于模型校准、肌电驱动模拟以及其他标准问题的工具。对于不符合标准形式的问题（例如预测运动），MocoStudy 提供了必要的灵活性。

## 验证

验证软件至关重要 [66]；因此，我们进行了广泛的验证测试。例如，为了验证 Moco 正确实现了直接配点法（direct collocation），我们求解了具有已知解的“线性正切转向”最优控制问题 [67]。该解与已知解相匹配，均方根误差为 $2 . 7 \times 1 0 ^ { - 5 }$

我们还确保了使用来自运动预测的控制量进行时间步进前向模拟（time-stepping forward simulation）所产生的运动与预测中的运动一致。该测试使用了一个由三块肌肉（DeGrooteFregly2016Muscle）悬挂并在重力作用下的质点模型（Fig 6）。对于这个问题，肌肉具有激活力学特性和刚性肌腱。我们首先预测了在规定的初始和终止位置之间移动质点（起始和结束均处于静止状态）的状态和控制轨迹（Fig 6，灰色带）。在此预测中，代价包括肌肉兴奋度平方和以及最终时间。然后，我们使用预测的控制量通过 OpenSim 积分器执行时间步进前向模拟（Fig 6，蓝色线）。所得质点位置轨迹与预测轨迹相匹配，均方根误差为 0.0051 m（初始与终止位置之间距离的 1.8%）。这使我们确信 Moco 执行了与 OpenSim 中时间步进前向模拟相同的多体动力学和肌肉动力学。欲了解更多

![](images/7eed3f99e1e6d4b00c8da130aa35bedfe470dee3b58bbb88daff678611911c9f.jpg)

![](images/a807f0df4fc5dc9f3ba221125867c72d9e5893e2536ac55e15896d802a2d9b7b.jpg)

![](images/5a835307f601d1b13fb6c73700759ae56570eda6deedd3797cfeae691901bacf.jpg)  
Fig 6. 时间步进与运动追踪的验证。左图：使用 Moco 模拟了一个由三根肌肉悬挂并在重力（g）作用下运动的质点的轨迹。右图：展示了不同模拟中“左”、“中”、“右”肌肉的激活程度。原始轨迹（灰色条带）是通过最小化最终时间和肌肉兴奋度平方和预测得到的。由预测控制驱动的时间步进正向模拟（蓝色）生成了我们最初预测的运动。使用 MocoTrack（橙色）追踪预测运动生成了原始的激活度。

https://doi.org/10.1371/journal.pcbi.1008493.g006

复杂问题中，使用来自 MocoSolution 的控制量进行时间步进正向模拟需要一个稳定反馈控制器来抵消数值误差。

为了增强对运动追踪问题的信心，我们确保在具有已知肌肉活动的合成运动上使用 MocoTrack 能够重现原始的肌肉活动。我们使用了相同的悬挂质点模型，并在最小化兴奋度平方的同时追踪先前的运动预测（Fig 6，灰色条带）。来自 MocoTrack 的肌肉激活度（Fig 6，橙色线）与预测的激活度相匹配，其均方根误差仅为峰值预测激活度的 0.47%。对于大多数受关注的追踪问题，我们并不知道真实的肌肉活动解；利用合成数据验证追踪问题，使我们对 Moco 估算肌肉活动的能力充满信心。

Moco 包含一个拥有 80 多个测试用例的自动化测试套件，其范围超出了此处描述的验证内容。该测试套件包含四种主要测试类型：解析测试、接口测试、算法测试和回归测试。所有测试均位于源代码中；我们采用持续集成（continuous integration），以确保在接受 GitHub 上的代码更改之前所有测试均能通过。关于我们自动化测试套件的详细描述见 S1 Appendix。

我们的自动化测试套件能够让用户确信 Moco 正确实现了直接配置法（direct collocation），但用户必须确保针对其具体问题使用合适的网格密度（单位时间内的网格区间数）和约束容差设置 [66]。尽管使用粗糙网格的问题可能会满足用户指定的约束容差，但动力学误差可能会过大，以致于模拟失去实用价值（[20]，第 4.7 节）。用户可以进行收敛性分析，以确保解不会对所选的网格密度或约束容差过度敏感。

## 结果

在此，我们展示了使用 Moco 获得的三个结果，代表了从规定运动到预测运动这一生物力学应用谱系。在第一个结果中，我们使用 MocoInverse 工具求解肌肉活动，同时精确规定模型的广义坐标值。我们比较了包含和不包含膝关节接触载荷项的目标函数的肌肉活动结果。在第二个结果中，我们使用 MocoTrack 模拟了正常行走以及髋外展肌和踝背屈肌减弱情况下的行走。在模拟肌无力时，我们在保持原始代价函数权重的同时减弱肌肉力量，直到模型无法再跟踪正常的运动学参考数据，并将生成的运动与已知的步态病理进行对比。在第三个结果中，我们使用自定义的 MocoStudy 预测了深蹲到站立（squat-to-stand）运动。随后，我们在优化膝关节辅助装置刚度的同时再次求解该问题，以评估辅助对肌肉活动的影响。这三个结果展示了 Moco 提供的最常用用户界面，并证明了 Moco 可以高效且准确地解决与生物力学研究人员相关的问题。

## 规定行走运动以估算肌肉活动

Moco 可以估算行走过程中的肌肉活动，从而能够研究正常和病理步态下的肌肉协同。我们使用了一个包含 29 个自由度和 80 块下肢肌肉的模型、广义坐标轨迹（通过逆运动学求解获得并使用 OpenSim 残差减少算法 Residual Reduction Algorithm 进行调整）以及地面反作用力，来模拟以 1.24 m/s 的自选速度行走的一个步态周期。该模型和数据基于文献 [59]，其中有更详细的描述。我们在所有肌肉中都对激活力学（activation dynamics）进行了建模，但仅在刚性肌腱假设不太适用的腓肠肌和比目鱼肌中对肌腱柔度（tendon compliance）进行了建模 [26]。我们使用能够精确规定运动学的 MocoInverse 求解了肌肉行为（参见“Tools for standard problems”获取

![](images/88617a9ca6b73115742a90ff71a3ab1620b90c7e92a8c2e6cc953ce8c1945677.jpg)  
Fig 7. 行走过程中肌肉活动的估算。除了髂肌之外，MocoInverse 生成的肌肉激活（黑色）在时序上与肌电图测量结果（灰色）的时序相似。臀大肌和髂肌的肌电图数据来自 Perry 和 Burnfield [68]；所有其他肌肉的肌电图数据均来自 Rajagopal 等人 [59]，并进行了归一化处理，使其峰值与 MocoInverse 激活的峰值相匹配。添加膝关节载荷的代价项降低了跨越膝关节的股二头肌短头和腓肠肌内侧头的活动（蓝色），并增加了其他肌肉的活动，以确保实现原始规定的运动。  
https://doi.org/10.1371/journal.pcbi.1008493.g007

details）。所有变量的初始猜测值为变量边界的中点。我们将 MocoInverse 生成的肌肉激活度与肌电图测量值进行了比较（Fig 7；黑线和灰色阴影）。

MocoInverse 生成的激活度包含了肌电图数据的一些主要特征，例如半腱肌（semitendinosus）、股二头肌短头（biceps femoris short head）、股外侧肌（vastus lateralis）、腓肠肌内侧头（medial gastrocnemius）和比目鱼肌（soleus）峰值活动的时间。髂肌（iliacus）的活动明显高于另一独立数据集中测得的数值 [68]；这可能是因为模型在站立后期产生的被动髋屈曲力矩 [59] 远低于基于称重传感器测量的被动力矩的预期值 [69]。使用 MocoInverse 时，所需的净关节力矩与肌肉生成的净关节力矩之间的差值（称为“储备”力矩，reserve moments）在所有时间和自由度上均被限制在不大于 2.5 N-m，这符合既定指南（储备力矩低于峰值净关节力矩的 5% [66]）。MocoInverse 使用 3.6 GHz 的 Intel Core i7 处理器（8 个并行线程）在 3.5 分钟内解决了该问题。我们将 MocoInverse 的解与 OpenSim 静态优化（Static Optimization）和计算肌肉控制（Computed Muscle Control）[6] 工具的解进行了比较（S1 Fig）。除腓肠肌内侧头、比目鱼肌和胫骨前肌（tibialis anterior）外，所有展示的肌肉在三种工具之间的肌肉活动都相似；这些肌肉的差异是由算法差异引起的，例如肌腱顺应性的处理方式。

为了证明 Moco 问题是可定制的，我们在 MocoInverse 中添加了一个代价项以最小化膝关节负荷（控制平方项的权重：1.0；关节负荷项的权重：0.005）。引入该额外代价后，膝关节反作用力的平均大小从 1.7 倍体重降至 1.1 倍体重。无论是否添加额外代价，膝关节反作用力的峰值均为 5.0 倍体重；通过最小化肌肉活动与肌电图数据之间的偏差 [71]，可能获得更接近体内膝关节植入物测量值（1.8–3.0 倍体重 [70]）的峰值。正如预期的那样，跨膝关节肌肉（股外侧肌、股二头肌短头和

腓肠肌内侧头）的活动有所减少（Fig 7；蓝线）。为了补偿踝关节处腓肠肌内侧头力矩的减少，比目鱼肌活动有所增加，正如先前的仿真研究所观察到的那样 [72]。Moco 在 15 分钟内解决了该问题。

## 追踪正常和肌力减弱情况下的行走动作

除了为预测优化（predictive optimizations）生成初始猜测外，追踪问题（tracking problems，其中运动作为代价函数的一部分被追踪，而不是被严格指定）对于研究模型的变化如何影响其复现参考运动的能力也非常有用。Moco 包含用于构建和求解追踪仿真的 MocoTrack 工具。我们探讨了减弱下肢步行模型中不同肌群对追踪正常步行数据能力的影响。作为 MocoTrack 的输入，我们使用了与上述 MocoInverse 问题相同的模型和运动学数据。我们没有指定地面反作用力，而是在双脚上添加了顺应性足-地接触力单元 [43]。我们移除了作为“备用”力矩（“reserve” moments）的力矩执行器（这些力矩仅在 MocoInverse 中是必需的），从而使下肢完全由肌肉驱动。与 MocoInverse 不同，MocoTrack 不指定模型广义坐标值；因此，我们在模型中施加了根据膝关节角度决定髌骨运动的运动学约束 [73]。这些运动学约束使得膝关节周围能够产生逼真的肌肉力臂，而无需对膝关节韧带和软骨进行建模。代价函数包括运动追踪项（权重：0.11）和控制平方项（权重：0.16）。我们加入了一个确保模型以与参考数据相同的平均速度行走的边界约束，以及一个防止双脚相互穿透的路径约束。我们假设了内外侧对称性，并模拟了半个步态周期。运动学变量的初始猜测取自追踪数据，而肌肉相关变量的初始猜测为变量上下界的中点。

我们首先使用正常的肌肉力量（最大等长肌力）求解了该追踪问题。MocoTrack 生成了追踪实验运动学的运动（Fig 8，顶部），其地面反作用力的时序与矢状面实验地面反作用力相匹配（Fig 8，左侧）。许多主要下肢肌肉的肌肉激活度在定性上与肌电图数据相似，包括臀大肌、股二头肌短头、股外侧肌、腓肠肌内侧头、比目鱼肌和胫骨前肌（Fig 8，激活度）。在与 MocoInverse 问题相同的计算机上，MocoTrack 在 130 分钟内求解了该问题；这一耗时短于单次打靶预测（single shooting predictions）中常见的耗时 [12]，但长于将算法微分（algorithmic differentiation）与直接配点法（direct collocation）相结合所能达到的耗时 [43]。

接下来，我们通过将最大等长肌力降低 95% 来削弱踝背屈肌（胫骨前肌、趾长伸肌和𧿹长伸肌），并再次求解了追踪问题（使用“正常”解作为初始猜测）。MocoTrack 生成了“足下垂”（drop foot）[68] 的行走解，其中模型在站立早期和摆动期无法背屈踝关节（Fig 9，左）。在恢复背屈肌力量并将髋外展肌（臀中肌、臀小肌和阔筋膜张肌）削弱 90% 后，我们再次求解了该问题（使用“正常”解作为初始猜测）。对于这两种肌力减弱的情况，我们均使用了惩罚控制变量平方和追踪误差的原始代价项。尽管此类追踪问题很少用于预测步态适应性改变，但它们在准确性与在没有追踪数据的情况下“设计”能够重现行走的代价项所必需的深入研究之间提供了一种折衷。由于主要的髋外展肌力几乎降至零（例如臀中肌），模型无法产生在实验数据和正常追踪解中观察到的髋内收角。适应后的步态

![](images/a0980b4772d72cad92740d7014a246d95856cf0676a76b7002d61717a8f8d458.jpg)  
Fig 8. 追踪三维实验运动数据。上图：MocoTrack 生成了追踪实验数据（灰色）的运动学结果（黑色）。左图：MocoTrack 生成的矢状面地面反作用力（黑色）的时间与实验测量的地面反作用力（灰色）相匹配。然而，垂直地面反作用力第一个峰值的幅度被高估了。右图：与 MocoInverse 类似，MocoTrack 生成的肌肉激活度（黑色）与肌电图测量值（灰色）的时间相匹配。臀大肌和髂肌的肌电图来自 Perry 和 Burnfield [68]；所有其他肌肉的肌电图来自 Rajagopal 等人 [59]，且信号再次进行了归一化，使其峰值与 MocoInverse 激活度的峰值相匹配，以与 Fig 7 保持一致。

https://doi.org/10.1371/journal.pcbi.1008493.g008

包含了躯干摇摆的大幅增加（Fig 9，右），这是特伦德伦伯格步态（Trendelenburg gait）[43, 68] 的特征。MocoTrack 求解背屈肌和髋外展肌减弱问题分别耗时 101 分钟和 148 分钟。

## 预测与辅助深蹲起立动作

为了展示 Moco 能够预测运动并优化模型参数，我们预测了一个蹲起动作，该动作最小化了肌肉消耗（表示为兴奋平方和，权重：1.0）与运动时长（权重：1.0）的组合。初始姿态设定为深蹲 [74]，最终姿态设定为直立站立。未跟踪任何运动数据。模型包含一个躯干和由 9 块具有激活力学特性的肌肉所驱动的单腿；其中 6 块肌肉具有顺应性肌腱力学特性（腘绳肌、股直肌、股肌群、腓肠肌、比目鱼肌和胫骨前肌）。肌肉强度取自先前的研究 [12]，并加倍以将双腿建模为单腿。我们通过使用焊接关节（weld joint）将模型的脚固定在地面上来模拟足-地接触。该模型包含与前述结果中相同的决定髌骨运动的运动学约束。运动学变量的初始猜测值为变量边界的中点；肌肉兴奋和激活度的初始猜测值为 0.05。预测的运动和肌肉激活度如图 10 所示（黑线）。正如预期的那样 [75]，臀大肌、腘绳肌和股肌群等伸肌表现出显著的活性。

![](images/e55d28bcf286f54bee67f489aad860e999676a231690a4aadae410885cb6ca9e.jpg)  
Fig 9. 背屈肌和髋外展肌减弱对运动跟踪的影响。左图：与实验踝关节角度（灰色）和正常跟踪解（黑线，白色骨骼）相比，减弱踝背屈肌导致支撑初期和摆动期的踝关节跖屈增加（橙色）。胫骨前肌力通过该肌肉的正常最大等长肌力 $F _ { \mathrm { i s o } } .$ 进行归一化。右图：与实验髋关节内收角（灰色）和正常跟踪解（黑色）相比，减弱髋外展肌导致支撑期髋关节内收角减小（绿色）。如骨骼图所示，与正常跟踪解（白色）相比，观察到躯干晃动增加（绿色）。作为主要髋外展肌的臀中肌的受力在整个步态周期中几乎降至零。  
https://doi.org/10.1371/journal.pcbi.1008493.g009

接下来，我们在膝关节处添加了一个扭簧，并求解了最佳运动、肌肉激活度和弹簧刚度。如图 10 所示，当膝关节伸展时弹簧处于平衡状态。我们使用了与无辅助情况相同的代价函数；未跟踪任何运动数据。在辅助装置的作用下，以更低的肌肉活性完成了该动作。最佳弹簧刚度为 90 N-m/rad。

![](images/8c91f142e724e85da2171c1fe5778e55e52645852c09441c6b86d481a969d55d.jpg)

![](images/76210739ccb8ff001915ddc561ce14fb0de055d1ac321b688a48f3646f895af3.jpg)  
Fig 10. 预测和辅助蹲起动作。左图：我们在规定初始和终止姿态的情况下预测了蹲起动作，该动作使肌肉兴奋度平方和以及最终时间最小化，随后在膝关节添加了扭转弹簧并求解最优弹簧刚度 k。右图：在无辅助弹簧（黑色）和有辅助弹簧（蓝色）情况下，整个动作过程中关键肌肉的激活程度表明，弹簧使臀大肌（gluteus maximus）和股肌群（vasti）的活性大幅降低。

https://doi.org/10.1371/journal.pcbi.1008493.g010

Table 1. 优化问题的求解时长与设置。

<table><tr><td>问题</td><td>时长（分钟）</td><td>网格区间数</td><td>自由度数</td><td>肌肉数</td><td>收敛容限</td><td>约束容限</td></tr><tr><td>MocoInverse，步态，正常（Fig 7，黑色）</td><td>3.5</td><td>25</td><td>29</td><td>80</td><td> $10^{-2}$ </td><td> $10^{-3}$ </td></tr><tr><td>MocoInverse，步态，膝关节负荷最小化（Fig 7，蓝色）</td><td>15</td><td>25</td><td>29</td><td>80</td><td> $10^{-2}$ </td><td> $10^{-3}$ </td></tr><tr><td>MocoTrack，步态，正常（Fig 8）</td><td>130</td><td>40</td><td>29</td><td>80</td><td> $10^{-3}$ </td><td> $10^{-3}$ </td></tr><tr><td>MocoTrack，步态，背屈肌无力（Fig 9，橙色）</td><td>101</td><td>40</td><td>29</td><td>80</td><td> $10^{-3}$ </td><td> $10^{-3}$ </td></tr><tr><td>MocoTrack，步态，髋外展肌无力（Fig 9，绿色）</td><td>148</td><td>40</td><td>29</td><td>80</td><td> $10^{-3}$ </td><td> $10^{-3}$ </td></tr><tr><td>预测性 MocoStudy，蹲起，无辅助（Fig 10，黑色）</td><td>3</td><td>50</td><td>4</td><td>9</td><td> $10^{-3}$ </td><td> $10^{-3}$ </td></tr><tr><td>预测性 MocoStudy，蹲起，有辅助（Fig 10，蓝色）</td><td>3</td><td>50</td><td>4</td><td>9</td><td> $10^{-3}$ </td><td> $10^{-3}$ </td></tr></table>

时长是在配备 8 个并行线程的 3.6 GHz Intel Core i7 处理器上获得的。所有问题均采用 Hermite–Simpson 转录方案求解；该方案中的每个网格区间包含区间边界上的两个点以及区间中点的一个配置点（即，具有 N 个网格区间的问题具有 2N + 1 个时间点）。DOFs：自由度（degrees of freedom）；收敛容限：KKT 条件的容限 [20]；约束容限：任何单项约束违背的最大允许值 [63]。  
https://doi.org/10.1371/journal.pcbi.1008493.t001

两次预测均在 3 分钟内收敛。快速预测运动和优化设备参数的能力，使 Moco 成为设计辅助设备的有力工具。

本节中介绍的各种优化的时长总结在 Table 1 中。

## 验证与确认总结

对于我们的每一项结果，我们都根据实际情况，将模拟量与实验肌电图（electromyography）、运动学（kinematics）和地面反作用力（ground reaction forces）进行了验证。为确保我们的解对网格的选择不敏感，我们进行了敏感性分析（S2 Fig）。我们的验证与确认（verification and validation）表明 Moco 能够产生有意义的结果，但并不保证用户在自己的应用中也能获得有意义的结果。研究人员必须通过将自己的结果与适当的实验数据进行对比，来验证他们使用自己的模型、代价项、约束和数据所获得的结果；我们鼓励研究人员遵循先前发表的指南 [66] 进行验证。

## 可用性与未来方向

OpenSim Moco 可在 Moco 网站（https:// opensim.stanford.edu/moco）和 SimTK（https://simtk.org/projects/opensim-moco）上免费下载 Windows 和 Mac 版本。源代码可在 GitHub（https://github.com/opensim-org/opensim-core）上获取。Moco 源代码采用宽松的 Apache License 2.0 授权许可，但某些依赖项具有更严格的许可证（例如，CasADi [62] 在 GNU 宽通用公共许可证下可用）。

Moco 的文档包含四个主要部分。“用户指南（User Guide）”解释了如何使用 Moco 并提供了提出问题的技巧。“理论指南（Theory Guide）”解释了 Moco 如何实现直接配置法（direct collocation）。“应用程序编程接口（API）参考（Application Programming Interface Reference）”描述了库中的类和函数。“开发者指南（Developer Guide）”解释了我们的软件设计。我们提供了一份两页的“速查表（cheat sheet）”，演示了常用命令。为了有效地使用 Moco，用户必须熟悉文档中未涵盖的各种主题，包括最优控制理论和直接配置法 [20]、数值方法（例如有限

差分）与优化及相关的陷阱（例如局部极小值）、多体动力学、Hill 型肌肉模型 [26] 以及 OpenSim 中的肌骨建模 [54]。

Moco 包含 MATLAB、Python 和 C++ 示例，涵盖从预测双摆的最优轨迹到预测二维肌肉驱动行走（求解耗时 30 分钟）。生成本文结果的代码可在 https:// github.com/stanfordnmbl/mocopaper 获取。

在规划仿真研究时，应考虑 Moco 和直接配置法（direct collocation method）的一些局限性。Moco 目前缺乏处理某些最优控制问题的能力，例如包含多阶段（multiple phases）[45]或单侧运动学约束（unilateral kinematic constraints）[76]的问题。将自定义模型组件（例如易疲劳肌肉）引入 Moco 需要具备 C++ 和 OpenSim 软件架构的相关知识；该软件的未来版本应允许用户在 MATLAB 或 Python 中实现自定义模型组件。Moco 中直接配置求解器的性能和易用性仍有改进空间。使用算法微分（algorithmic differentiation）而非有限差分（finite differences）来计算非线性规划导数，将极大地提高 Moco 的运行速度 [50]。支持网格细化（mesh refinement）将允许求解器在动力学变化较快的时间范围内增加网格区间数量，从而提高精度（[20]，第 4.7 节）。Moco 尝试生成缩放良好（well-scaled）的优化问题以促进快速收敛（例如，使用归一化肌腱力作为状态变量），但提供自动化问题缩放（[20]，第 4.8 节）可以进一步改善收敛性。最后，直接配置法本身不太适合交互式仿真（例如用户对模型施加扰动）或包含随机性的仿真（例如不平整地形）；此类问题更适合采用单打靶法（single shooting）[12]或强化学习方法 [77]。

展望未来，我们将重点把 Moco 应用于生物力学中具有重大影响力的课题，包括骨科手术以及肌肉力量训练等康复策略。运动预测通常采用通用模型，因为大规模的受试者特异性（subject-specific）仿真构建起来较为困难。Moco 允许研究人员轻松构建和共享受试者特异性研究，从而能够识别出哪些个体可以从某项干预措施中获益。将 Moco 与来自惯性测量单元（inertial measurement units）[78]的数据或经计算机视觉处理的视频 [79] 相结合，将使研究人员能够基于实验室外的观察来研究运动过程中的肌肉行为。

Moco 项目得益于一个由应用肌肉骨骼仿真的研究人员组成的活跃社区。我们通过在国际会议、Stanford University（斯坦福大学）以及线上举办 Moco 研讨会与该社区开展互动。我们将继续通过举办更多的研讨会，并协助他人贡献文档、示例、教学材料和代码，来持续维系和推动该社区的发展。

我们把 Moco 设计为易于使用、可定制且可扩展的软件。我们验证了该软件，并将其应用于多个肌肉骨骼问题。Moco 能够处理具有运动学约束、肌肉激活力学、顺应性肌腱和顺应性接触的模型，并且可以最小化由标记点追踪和关节反作用载荷等组成的复合复杂代价。基于这一基础，我们期望 Moco 能够通过减少研究人员在仿真工具上耗费的时间，并使我们的领域能够攻克更具挑战性的问题，从而加速科学研究。

## 支持信息

S1 Appendix. 数学细节。本附录详细描述了 Moco 如何使用直接配点法求解最优控制问题。

(PDF)

S1 Fig. MocoInverse 与静态优化及计算肌肉控制的比较。对于 Fig 7 中展示的行走动作，OpenSim Static Optimization 和 Computed Muscle Control 工具产生的肌肉激活时间与 MocoInverse 产生的结果相似。在所有工具中，除内侧腓肠肌、比目鱼肌和胫骨前肌外，所示所有肌肉的激活幅度均相似。幅度上的差异是由算法差异引起的，例如对肌腱顺应性的处理方式。

(TIF)

S2 Fig. 收敛性分析。为了确保我们的结果对所选网格区间数（与网格密度相关）不敏感，我们进行了收敛性分析：我们确保随着网格区间数的增加，目标函数值收敛于单一数值。我们对结果中的三个问题进行了收敛性分析：“MocoInverse, gait, normal”（Fig 7，黑色）、“MocoTrack, gait, normal”（Fig 8）以及“Predictive MocoStudy, squat-to-stand, unassisted”（Fig 10，黑色）。对于每个图表，目标函数值均通过使用最大网格区间数时的目标函数值进行了归一化。

(TIF)

## 致谢

我们感谢 Ajay Seth、Michael Sherman、Friedl de Groote、Antonie J. van den Bogert、Michael Posa、Joris Gillis 和 Joel Andersson 讨论了方法与实现；感谢 Bradley Humphreys、Carmichael Ong、Noah Gordon 和 Jennifer Yong 贡献了代码；感谢 Andrew Baines、Mohammad Shourijeh 和 Prasanna Sritharan 测试了软件；并感谢 Ayman Habib 审阅了手稿。

## 作者贡献

概念构思：Christopher L. Dembia, Nicholas A. Bianco.

经费获取：Jennifer L. Hicks, Scott L. Delp.

方法学：Christopher L. Dembia, Nicholas A. Bianco, Antoine Falisse.

项目管理：Jennifer L. Hicks, Scott L. Delp.

资源提供：Scott L. Delp.

软件：Christopher L. Dembia, Nicholas A. Bianco, Antoine Falisse.

监督指导：Jennifer L. Hicks, Scott L. Delp.

验证：Christopher L. Dembia, Nicholas A. Bianco.

可视化：Christopher L. Dembia, Nicholas A. Bianco.

撰写 – 原始初稿：Christopher L. Dembia, Nicholas A. Bianco.

撰写 – 审阅与编辑：Christopher L. Dembia, Nicholas A. Bianco, Antoine Falisse, Jennifer L. Hicks, Scott L. Delp.

## 参考文献

1. Fregly BJ, Reinbolt JA, Rooney KL, Mitchell KH, Chmielewski TL. Design of patient-specific gait modifi cations for knee osteoarthritis rehabilitation. IEEE Transactions on Biomedical Engineering. 2007; 54:1687–1695. https://doi.org/10.1109/TBME.2007.891934 PMID: 17867361

2. Steele KM, Rozumalski A, Schwartz MH. Muscle synergies and complexity of neuromuscular control during gait in cerebral palsy. Developmental Medicine & Child Neurology. 2015; 57:1176–1182.

3. Priamikov A, Fronius M, Shi B, Triesch J. OpenEyeSim: A biomechanical model for simulation of closed-loop visual perception. Journal of Vision. 2016; 16:25–25. https://doi.org/10.1167/16.15.25 PMID: 28006074

4. Hutchinson JR, Anderson FC, Blemker SS, Delp SL. Analysis of hindlimb muscle moment arms in Tyrannosaurus rex using a three-dimensional musculoskeletal computer model: implications for stance, gait, and speed. Paleobiology. 2005; 31:676–701. https://doi.org/10.1666/04044.1

5. O’Neill MC, Umberger BR, Holowka NB, Larson SG, Reiser PJ. Chimpanzee super strength and human skeletal muscle evolution. Proceedings of the National Academy of Sciences. 2017; 114:7343– 7348. https://doi.org/10.1073/pnas.1619071114 PMID: 28652350

6. Thelen DG, Anderson FC, Delp SL. Generating dynamic simulations of movement using computed muscle control. Journal of Biomechanics. 2003; 36:321–328. https://doi.org/10.1016/S0021-9290(02) 00432-3 PMID: 12594980

7. Lloyd DG, Besier TF. An EMG-driven musculoskeletal model to estimate muscle forces and knee joint moments in vivo. Journal of Biomechanics. 2003; 36:765–776. https://doi.org/10.1016/S0021-9290(03) 00010-1 PMID: 12742444

8. Farris DJ, Hicks JL, Delp SL, Sawicki GS. Musculoskeletal modelling deconstructs the paradoxical effects of elastic ankle exoskeletons on plantar-flexor mechanics and energetics during hopping. Jour nal of Experimental Biology. 2014; 217:4018–4028. https://doi.org/10.1242/jeb.107656 PMID: 25278469

9. Jackson RW, Dembia CL, Delp SL, Collins SH. Muscle-tendon mechanics explain unexpected effects of exoskeleton assistance on metabolic rate during walking. Journal of Experimental Biology. 2017; 220:jeb.150011. https://doi.org/10.1242/jeb.150011 PMID: 28341663

10. Prilutsky BI, Zatsiorsky VM. Optimization-Based Models of Muscle Coordination. Exercise and Sport Sciences Reviews. 2002; 30(1):32–38. https://doi.org/10.1097/00003677-200201000-00007 PMID: 11800497

11. Geijtenbeek T. SCONE: Open Source Software for Predictive Simulation of Biological Motion. Journal of Open Source Software. 2019; 4:1421. https://doi.org/10.21105/joss.01421

12. Ong CF, Geijtenbeek T, Hicks JL, Delp SL. Predicting gait adaptations due to ankle plantarflexor muscle weakness and contracture using physics-based musculoskeletal simulations. PLoS Computational Biology. 2019; 15(10):e1006993. https://doi.org/10.1371/journal.pcbi.1006993 PMID: 31589597

13. Handford ML, Srinivasan M. Robotic lower limb prosthesis design through simultaneous computer optimizations of human and prosthesis costs. Scientific Reports. 2016; 6:19983. https://doi.org/10.1038/ srep19983 PMID: 26857747

14. Millard M, Sreenivasa M, Mombaur K. Predicting the Motions and Forces of Wearable Robotic Systems Using Optimal Control. Frontiers in Robotics and AI. 2017; 4:41. https://doi.org/10.3389/frobt.2017. 00041

15. Anderson FC, Pandy MG. Dynamic Optimization of Human Walking. Journal of Biomechanical Engineering. 2001; 123(5):381–390. https://doi.org/10.1115/1.1392310 PMID: 11601721

16. Lin YC, Pandy MG. Three-dimensional data-tracking dynamic optimization simulations of human locomotion generated by direct collocation. Journal of Biomechanics. 2017; 59:1–8. https://doi.org/10.1016/ j.jbiomech.2017.04.038 PMID: 28583674

17. Todorov E, Li W. Optimal control methods suitable for biomechanical systems. Proceedings of the 25th Annual International Conference of the IEEE Engineering in Medicine and Biology Society. 2003;2:1758–1761.

18. Bock HG, Plitt KJ. A Multiple Shooting Algorithm for Direct Solution of Optimal Control Problems\*. IFAC Proceedings Volumes. 1984; 17(2):1603—1608. https://doi.org/10.1016/S1474-6670(17)61205-9

19. Leineweber DB, Bauer I, Bock HG, Schlo¨der JP. An efficient multiple shooting based reduced SQP strategy for large-scale dynamic process optimization. Part 1: theoretical aspects. Computers & Chemical Engineering. 2003; 27(2):157—166. https://doi.org/10.1016/S0098-1354(02)00158-8

20. Betts JT. Practical Methods for Optimal Control and Estimation Using Nonlinear Programming. SIAM; 2010.

21. Kelly M. An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation. SIAM Review. 2017; 59:849–904. https://doi.org/10.1137/16M1062569

22. Hargraves CR, Paris SW. Direct trajectory optimization using nonlinear programming and collocation. Journal of Guidance, Control, and Dynamics. 1987; 10(4):338–342. https://doi.org/10.2514/3.20223

23. von Stryk O. In: Bulirsch R, Miele A, Stoer J, Well K, editors. Numerical Solution of Optimal Control Problems by Direct Collocation. Basel: Birkha¨user Basel; 1993. p. 129–143. Available from: https://doi. org/10.1007/978-3-0348-7539-4\_10.

24. Hairer E, Wanner G, Nørsett SP. Solving Ordinary Differential Equations I, Nonstiff Problems. Springer; 1993.

25. Hairer E, Wanner G. Solving Ordinary Differential Equations II, Stiff and Differential-Algebraic Problems. Springer; 1996.

26. De Groote F, Kinney AL, Rao AV, Fregly BJ. Evaluation of Direct Collocation Optimal Control Problem Formulations for Solving the Muscle Redundancy Problem. Annals of Biomedical Engineering. 2016; 44:2922–2936. https://doi.org/10.1007/s10439-016-1591-9 PMID: 27001399

27. Ueno R, Navacchia A, DiCesare CA, Ford KR, Myer GD, Ishida T, et al. Knee abduction moment is predicted by lower gluteus medius force and larger vertical and lateral ground reaction forces during drop vertical jump in female athletes. Journal of Biomechanics. 2020; 103:109669. https://doi.org/10.1016/j. jbiomech.2020.109669 PMID: 32019678

28. Kaplan ML, Heegaard JH. Predictive algorithms for neuromuscular control of human locomotion. Journal of Biomechanics. 2001; 34(8):1077–1083. https://doi.org/10.1016/S0021-9290(01)00057-4 PMID: 11448699

29. Mehrabi N, Schwartz MH, Steele KM. Can Altered Muscle Synergies Control Unimpaired Gait? Journa of Biomechanics. 2019; 90:84–91. https://doi.org/10.1016/j.jbiomech.2019.04.038 PMID: 31101431

30. Koelewijn AD, van den Bogert AJ. Joint contact forces can be reduced by improving joint moment symmetry in below-knee amputee gait simulations. Gait & Posture. 2016; 49:219–225. https://doi.org/10. 1016/j.gaitpost.2016.07.007 PMID: 27459416

31. Meyer AJ, Eskinazi I, Jackson JN, Rao AV, Patten C, Fregly BJ. Muscle Synergies Facilitate Computational Prediction of Subject-Specific Walking Motions. Frontiers in Bioengineering and Biotechnology. 2016; 4:77. https://doi.org/10.3389/fbioe.2016.00077 PMID: 27790612

32. Ackermann M, van den Bogert AJ. Optimality principles for model-based prediction of human gait. Journal of Biomechanics. 2010; 43:1055–1060. https://doi.org/10.1016/j.jbiomech.2009.12.012 PMID: 20074736

33. Ackermann M, van den Bogert AJ. Predictive simulation of gait at low gravity reveals skipping as the preferred locomotion strategy. Journal of Biomechanics. 2012; 45:1293–1298. https://doi.org/10.1016/j. jbiomech.2012.01.029 PMID: 22365845

34. Miller RH, Hamill J. Optimal footfall patterns for cost minimization in running. Journal of Biomechanics. 2015; 48:2858–2864. https://doi.org/10.1016/j.jbiomech.2015.04.019 PMID: 25952545

35. Porsa S, Lin YC, Pandy MG. Direct Methods for Predicting Movement Biomechanics Based Upon Optimal Control Theory with Implementation in OpenSim. Annals of Biomedical Engineering. 2016; 44:2542–2557. https://doi.org/10.1007/s10439-015-1538-6 PMID: 26715209

36. Lee LF, Umberger BR. Generating optimal control simulations of musculoskeletal movement using OpenSim and MATLAB. PeerJ. 2016; 4:e1638. https://doi.org/10.7717/peerj.1638 PMID: 26835184

37. Umberger BR, Miller RH. In: Optimal Control Modeling of Human Movement. Cham: Springer International Publishing; 2018. p. 327–348.

38. Bobbert MF, Kistemaker DA, Vaz MA, Ackermann M. Searching for strategies to reduce the mechanical demands of the sit-to-stand task with a muscle-actuated optimal control model. Clinical Biomechanics. 2016; 37:83–90. https://doi.org/10.1016/j.clinbiomech.2016.06.008 PMID: 27380203

39. Moore JK, van den Bogert AJ. opty: Software for trajectory optimization and parameter identification using direct collocation. The Journal of Open Source Software. 2018; 3:300. https://doi.org/10.21105/ joss.00300

40. Lin YC, Walter JP, Pandy MG. Predictive Simulations of Neuromuscular Coordination and Joint-Contact Loading in Human Gait. Annals of Biomedical Engineering. 2018; 46:1216–1227. https://doi.org/10. 1007/s10439-018-2045-3 PMID: 29671152

41. Lai AKM, Biewener AA, Wakeling JM. Metabolic cost underlies task-dependent variations in motor unit recruitment. Journal of the Royal Society Interface. 2018; 15:20180541. https://doi.org/10.1098/rsif. 2018.0541 PMID: 30464057

42. Nguyen VQ, Johnson RT, S FC IV, Umberger BR. Bilevel Optimization for Cost Function Determination in Dynamic Simulation of Human Gait. IEEE Transactions on Neural Systems and Rehabilitation Engineering. 2019; 27:1426–1435. https://doi.org/10.1109/TNSRE.2019.2922942 PMID: 31199264

43. Falisse A, Serrancolı´ G, Dembia CL, Gillis J, Jonkers I, De Groote F. Rapid predictive simulations with complex musculoskeletal models suggest that diverse healthy and pathological human gaits can emerge from similar control strategies. Journal of the Royal Society, Interface. 2019; 16:20190402. https://doi.org/10.1098/rsif.2019.0402 PMID: 31431186

44. Jansen C, McPhee J. Predictive dynamic simulation of Olympic track cycling standing start using direct collocation optimal control. Multibody System Dynamics. 2020; 49:53–70. https://doi.org/10.1007/ s11044-020-09723-3

45. Falisse A, Rossom SV, Jonkers I, De Groote F. EMG-Driven Optimal Estimation of Subject-Specific Hill Model Muscle–Tendon Parameters of the Knee Joint Actuators. IEEE Transactions on Biomedical Engineering. 2017; 64:2253–2262. https://doi.org/10.1109/TBME.2016.2630009 PMID: 27875132

46. Rohani F, Richter H, van den Bogert AJ. Optimal design and control of an electromechanical transfemoral prosthesis with energy regeneration. PLoS ONE. 2017; 12:e0188266. https://doi.org/10.1371/ journal.pone.0188266 PMID: 29149213

47. van den Bogert AJ, Blana D, Heinrich D. Implicit methods for efficient musculoskeletal simulation and optimal control. Procedia IUTAM. 2011; 2:297–316. https://doi.org/10.1016/j.piutam.2011.04.027 PMID: 22102983

48. Koelewijn AD, Dorschky E, van den Bogert AJ. A metabolic energy expenditure model with a continuous first derivative and its application to predictive simulations of gait. Computer Methods in Biomechanics and Biomedical Engineering. 2018; 21:1–11. https://doi.org/10.1080/10255842.2018.1490954

49. Koelewijn AD, Heinrich D, van den Bogert AJ. Metabolic cost calculations of gait using musculoskeletal energy models, a comparison study. PLoS ONE. 2019; 14(9):e0222037. https://doi.org/10.1371/ journal.pone.0222037 PMID: 31532796

50. Falisse A, Serrancolı´ G, Dembia CL, Gillis J, De Groote F. Algorithmic differentiation improves the computational efficiency of OpenSim-based trajectory optimization of human movement. PLoS ONE. 2019; 14:e0217730. https://doi.org/10.1371/journal.pone.0217730 PMID: 31622352

51. Becerra VM. Solving Complex Optimal Control Problems at No Cost with PSOPT. 2010 IEEE International Symposium on Computer-Aided Control System Design. 2010; p. 1391–1396.

52. Patterson MA, Rao AV. GPOPS-II: A MATLAB Software for Solving Multiple-Phase Optimal Control Problems Using hp-Adaptive Gaussian Quadrature Collocation Methods and Sparse Nonlinear Programming. ACM Transactions on Mathematical Software. 2014; 41. https://doi.org/10.1145/2558904

53. Delp SL, Anderson FC, Arnold AS, Loan P, Habib A, John CT, et al. OpenSim: Open-Source Software to Create and Analyze Dynamic Simulations of Movement. IEEE Transactions on Biomedical Engineering. 2007; 54:1940–1950. https://doi.org/10.1109/TBME.2007.901024 PMID: 18018689

54. Seth A, Hicks JL, Uchida TK, Habib A, Dembia CL, Dunne JJ, et al. OpenSim: Simulating musculoskeletal dynamics and neuromuscular control to study human and animal movement. PLoS Computationa Biology. 2018; 14:e1006223. https://doi.org/10.1371/journal.pcbi.1006223 PMID: 30048444

55. Sherman MA, Seth A, Delp SL. Simbody: multibody dynamics for biomedical research. Procedia IUTAM. 2011; 2:241–261. https://doi.org/10.1016/j.piutam.2011.04.023 PMID: 25866705

56. Posa M, Tedrake R, Kuindersma S. Optimization and Stabilization of Trajectories for Constrained Dynamical Systems. 2016 IEEE International Conference on Robotics and Automation. 2016; p. 1366– 1373.

57. Seth A, Matias R, Veloso AP, Delp SL. A Biomechanical Model of the Scapulothoracic Joint to Accurately Capture Scapular Kinematics during Shoulder Movements. PLoS ONE. 2016; 11:e0141028. https://doi.org/10.1371/journal.pone.0141028 PMID: 26734761

58. Lerner ZF, DeMers MS, Delp SL, Browning RC. How tibiofemoral alignment and contact locations affect predictions of medial and lateral tibiofemoral contact forces. Journal of Biomechanics. 2015; 48:644– 650. https://doi.org/10.1016/j.jbiomech.2014.12.049 PMID: 25595425

59. Rajagopal A, Dembia CL, DeMers MS, Delp DD, Hicks JL, Delp SL. Full-Body Musculoskeletal Model for Muscle-Driven Simulation of Human Gait. IEEE Transactions on Biomedical Engineering. 2016; 63:2068–2079. https://doi.org/10.1109/TBME.2016.2586891 PMID: 27392337

60. Cazzola D, Holsgrove TP, Preatoni E, Gill HS, Trewartha G. Cervical Spine Injuries: A Whole-Body Musculoskeletal Model for the Analysis of Spinal Loading. PLoS ONE. 2017; 12:e0169329. https://doi. org/10.1371/journal.pone.0169329 PMID: 28052130

61. Serrancoli G, Falisse A, Dembia C, Vantilt J, Tanghe K, Lefeber D, et al. Subject-Exoskeleton Contact Model Calibration Leads to Accurate Interaction Force Predictions. IEEE Transactions on Neural Systems and Rehabilitation Engineering. 2019; 27:1597–1605. https://doi.org/10.1109/TNSRE.2019. 2924536 PMID: 31247556

62. Andersson JAE, Gillis J, Horn G, Rawlings JB, Diehl M. CasADi: a software framework for nonlinear optimization and optimal control. Mathematical Programming Computation. 2019; 11:1–36. https://doi. org/10.1007/s12532-018-0139-4

63. Wa¨chter A, Biegler LT. On the implementation of an interior-point filter line-search algorithm for largescale nonlinear programming. Mathematical Programming. 2006; 106:25–57. https://doi.org/10.1007/ s10107-004-0559-y

64. Gill PE, Murray W, Saunders MA. SNOPT: An SQP Algorithm for Large-Scale Constrained Optimization. SIAM Review. 2005; 47:99–131. https://doi.org/10.1137/S0036144504446096

65. Peng RD. Reproducible Research in Computational Science. Science. 2011; 334:1226–1227. https:/ doi.org/10.1126/science.1213847 PMID: 22144613

66. Hicks JL, Uchida TK, Seth A, Rajagopal A, Delp SL. Is My Model Good Enough? Best Practices for Verification and Validation of Musculoskeletal Models and Simulations of Movement. Journal of Biomechanical Engineering. 2015; 137:020905. https://doi.org/10.1115/1.4029304 PMID: 25474098

67. Bryson AE, Ho YC. Applied Optimal Control: Optimization, Estimation, and Control. Washington: Hemisphere Pub. Corp; 1975.

68. Perry J, Burnfield J. Gait Analysis: Normal and Pathological Function. 2nd ed. Slack Incorporated; 2010.

69. Whittington B, Silder A, Heiderscheit B, Thelen DG. The contribution of passive-elastic mechanisms to lower extremity joint kinetics during human walking. Gait & Posture. 2008; 27(4):628–634. https://doi. org/10.1016/j.gaitpost.2007.08.005 PMID: 17928228

70. Fregly BJ, Besier TF, Lloyd DG, Delp SL, Banks SA, Pandy MG, et al. Grand challenge competition to predict in vivo knee loads. Journal of Orthopaedic Research. 2012; 30(4):503–513. https://doi.org/10. 1002/jor.22023 PMID: 22161745

71. Kinney AL, Besier TF, D’Lima DD, Fregly BJ. Update on grand challenge competition to predict in vivo knee loads. Journal of biomechanical engineering. 2013; 135(2):021012. https://doi.org/10.1115/1. 4023255 PMID: 23445057

72. DeMers MS, Pal S, Delp SL. Changes in tibiofemoral forces due to variations in muscle activity during walking. Journal of Orthopaedic Research. 2014; 32:769–776. https://doi.org/10.1002/jor.22601 PMID: 24615885

73. Arnold EM, Ward SR, Lieber RL, Delp SL. A Model of the Lower Limb for Analysis of Human Movement. Annals of Biomedical Engineering. 2010; 38(2):269–279. https://doi.org/10.1007/s10439-009-9852-5 PMID: 19957039

74. Fregly BJ, Fregly CD, Kim BT. 国际空间站上 ARED 深蹲运动期间肌肉力矩的计算预测. Journal of Biomechanical Engineering. 2015; 137:121005. https://doi.org/10.1115/1.4031795 PMID: 26473475

75. Yavuz HU, Erdag D. 次最大与最大负荷颈后深蹲期间的运动学和肌电活动变化. Applied Bionics and Biomechanics. 2017; 2017:1–8. https://doi.org/10.1155/2017/9084725 PMID: 28546738

76. Patel A, Shield SL, Kazi S, Johnson AM, Biegler LT. 基于正交配点法的接触隐式轨迹优化. IEEE Robotics and Automation Letters. 2019; 4(2):2242–2249. https://doi.org/10.1109/LRA.2019.2900840

77. Kidziński Ł, Ong C, Mohanty SP, Hicks J, Carroll S, Zhou B, et al. 假肢人工智能：挑战赛解决方案. 载于: Escalera S, Herbrich R, 编. The NeurIPS’18 Competition. Cham: Springer International Publishing; 2020. 第 69–128 页.

78. Dorschky E, Nitschke M, Seifer AK, van den Bogert AJ, Eskofier BM. 基于肌骨模型最优控制利用惯性传感器数据估计步态运动学与动力学. Journal of Biomechanics. 2019; 95:109278. https://doi.org/10.1016/j.jbiomech.2019.07.022 PMID: 31472970

79. Cao Z, Hidalgo Martinez G, Simon T, Wei S, Sheikh YA. OpenPose：利用部分亲和场进行实时多人 2D 姿态估计. IEEE Transactions on Pattern Analysis and Machine Intelligence. 2019;. https://doi.org/10.1109/TPAMI.2019.2929257