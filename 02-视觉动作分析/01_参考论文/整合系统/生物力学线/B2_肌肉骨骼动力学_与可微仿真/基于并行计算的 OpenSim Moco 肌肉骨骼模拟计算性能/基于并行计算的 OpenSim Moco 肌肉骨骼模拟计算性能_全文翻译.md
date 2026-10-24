应 用 研 究

WILEY

# 基于并行计算的 OpenSim Moco 肌肉骨骼模拟计算性能

Alex N. Denton | Brian R. Umberger

School of Kinesiology, University of Michigan, Ann Arbor, Michigan, USA

通讯作者

Brian R. Umberger, School of Kinesiology, University of Michigan, Ann Arbor, MI, USA.

电子邮箱：umberger@umich.edu

资助信息 Rackham Graduate Student Research Grant

## 摘要

最优控制肌肉骨骼模拟是研究人体运动的基础和临床问题的一种有价值的方法。然而，高昂的计算需求长期以来一直是一项重大挑战，因此亟需提升模拟性能。OpenSim Moco 软件包允许使用 CasADi 最优控制库在多核处理器上并行求解肌肉骨骼模拟问题，从而有可能降低计算需求。然而，该框架的计算性能尚未得到彻底检验。因此，我们的目标是研究在 OpenSim Moco 中进行人体运动最优控制模拟时，通过多核并行计算相对于串行求解（即使用单核）所获得的计算加速比。模拟使用多达 18 个核心、在多种不同密度的时间网格区间下，并采用两种不同的初始猜测策略进行求解。我们考察了一系列肌肉骨骼模型和运动，包括二维和三维模型、追踪和预测模拟，以及步行和触碰任务。最大总体并行加速比因具体问题而异，相比串行求解提高了 1.7 至 7.7 倍，且大部分加速效果在约 6 个处理器核心时即可达到。在更精细的时间网格上，并行加速比通常更大，而初始猜测策略对加速比的影响极小。通过充分利用现代计算机中常见的多核处理器，OpenSim Moco 中的某些最优控制模拟问题可以实现显著的加速。然而，由于性能提升因具体问题而异，要实现最优计算性能仍需要最终用户进行一定程度的探索。

关 键 词

生物力学，肌肉骨骼模型，最优控制，优化

## 1 | 引言

肌肉骨骼仿真结合最优控制技术是一种极具价值的研究方法，能够为人体运动提供新颖的见解；然而，计算效率是限制该技术广泛应用的主要因素。<sup>1,2</sup> 虽然存在一些具有较高计算效率的仿真技术（例如静态优化<sup>3</sup>），但这些算法往往绑定于特定的实验数据集，通常不包含肌肉-肌腱动力学，且无法用于预测未曾发生过的新运动。<sup>4</sup> 通过诸如直接配点法（direct collocation）等算法的发展，最优控制仿真的巨大计算成本已得到部分降低。<sup>5–9</sup> OpenSim Moco<sup>10</sup> 是一款相对较新的软件工具，它在利用 OpenSim 肌肉骨骼建模环境的同时，简化了直接配点法的使用流程。<sup>11</sup> 利用直接配点法进行快速仿真，不仅使许多对计算能力要求极高的问题变得切实可行，还能促进更为复杂的分析（例如双层优化<sup>12</sup>和蒙特卡洛方法<sup>13</sup>），在这些分析中，算法的每一次迭代都需要求解一个完整的最优控制问题。

在最优控制仿真中提高计算效率的另一种途径是并行求解问题的各组成部分。并行计算是一种强大且应用广泛的计算技术，通过该技术可以同时求解问题的各个组件。<sup>14</sup> 数十年来，人体运动的生物力学仿真一直受益于并行计算，但这通常局限于集中式高性能计算资源（例如参考文献 2）或商用计算机集群（例如参考文献 15）的环境中。如今的现代笔记本电脑和台式计算机已配备一个或多个处理器，每个处理器包含多达数十个处理器核心。这种计算机架构非常适合 OpenSim Moco，该软件使用 CasADi 库<sup>16</sup>，能够在多核处理器上并行评估目标函数和约束方程。

尽管并行计算现已可在主流计算机上实现，但拥有超过 4–8 个处理器核心的多核处理器价格昂贵。并行计算缩短运行时间的潜力通常因具体问题而异，取决于问题规模、问题类型（例如，跟踪数据对比预测运动）、运动任务、模型复杂度以及问题离散化（即时间网格密度）等因素。将问题并行化也会产生计算开销，因此在超过特定阈值后使用更多处理器核心可能不会带来进一步的加速，甚至可能导致计算性能下降。了解这些问题有助于计算科学家在有限的资金预算下，选择与其实验仿真问题最匹配的算法和计算机架构。因此，本研究的目的是探讨在求解最优控制肌骨仿真时所使用的处理器核心数如何与肌骨模型复杂度、运动任务、初始猜测类型和时间网格密度相互作用，从而影响仿真运行时间和加速潜力。

## 2 | 方法

## 2.1 | 肌骨模型

本研究使用二维（2-D）和三维（3-D）全身肌骨模型生成行走仿真，并使用 2-D 上肢模型生成伸手触及运动仿真。我们将分析重点放在步态和上肢触及运动上，因为它们是人体生物力学中研究最广泛的运动之一，同时也是 OpenSim Moco 研究的重点（例如参考文献 17,18）。所有三个肌骨模型的肌腱作动器均基于 Hill 型肌肉模型。<sup>19</sup>

全身 2-D<sup>12</sup> 和 3-D<sup>17</sup> 肌骨模型此前已有详细描述，此处仅作简要概述。2-D 矢状面模型由 9 个环节、11 个自由度和 18 个肌腱单元作动器组成。<sup>12</sup> 足-地相互作用通过每只脚下的 8 个 Hunt-Crossley 接触单元进行建模。<sup>6</sup> 全身 3-D 肌骨模型由 18 个环节、31 个自由度和 84 个肌腱单元作动器组成。<sup>17</sup> 足-地相互作用通过每只脚下的 11 个 Hunt-Crossley 接触单元进行建模。<sup>20</sup>

上肢肌肉骨骼模型是在 OpenSim “Arm26” 模型<sup>21</sup> 的基础上修改而成，该模型基于先前发表的三维（3-D）模型。<sup>22</sup> 原始的平面 Arm26 模型<sup>21</sup> 由三个环节、两个自由度和六个肌腱执行器组成。我们通过添加来自<sup>22</sup>的六块额外肌肉对 Arm26 模型进行了扩展，以提供对盂肱关节更稳健的控制。添加到当前上肢模型中的来自<sup>22</sup>的肌肉包括：前三角肌（DELT1）、后三角肌（DELT3）、大圆肌（TMAJ）、胸大肌（PECM）、背阔肌（LAT）和喙肱肌（CORB）。二维（2-D）全身模型和修改后的二维上肢模型均可在 SimTK 网站获取（https://simtk.org/projects/mocoparallel），三维全身模型可在 https://simtk.org/projects/umocod 获取。

## 2.2 | 仿真

所有仿真均在一系列处理器核心数（1、3、6、9、12、15 和 18）下以串行和并行方式运行。半个步态周期（1.3 m/s）的 2-D 追踪仿真、2-D 预测仿真和 3-D 追踪仿真均在一台多核计算机工作站（Intel<sup>®</sup> Core(TM) i9-7980XE CPU @ 2.6 GHz，18 核处理器，64 GB 内存）上使用 OpenSim Moco 4.3<sup>10</sup> 和 MATLAB 2020a（Mathworks, Natick）生成。为了生成仿真，OpenSim Moco 依赖于 CasADi 库<sup>16</sup>（提供用于求解数值最优控制问题的框架）和 IPOPT 优化器<sup>23</sup>（采用基于梯度的方法求解非线性规划 [NLP]）。SimTK 网站（https://simtk.org/projects/mocoparallel）提供了一个完整的运行示例，其中指明了所使用的各种仿真参数设置，并突出了在不同仿真中发生更改的参数值。

在步行的 2-D 追踪仿真中，目标函数是追踪误差（即仿真与实验运动学及地面反作用力数据之间的偏差平方）与肌肉兴奋度平方和的加权和。<sup>24</sup> 追踪的数据为 8 名健康成年受试者以 1.3 m/s 进行地面行走的平均值。<sup>12</sup> 在步行的 3-D 追踪仿真中，目标函数是追踪误差（即仿真与实验运动学及地面反作用力数据之间的偏差平方）与肌肉兴奋度平方和的总和。<sup>24</sup> 追踪的数据为 14 名健康成年受试者以 1.45 m/s 进行地面行走的平均值。<sup>25</sup> 预测目标函数为肌肉兴奋度的立方和。<sup>5</sup>

半个步态周期时长（0.54 s）被划分为 $n$ 个时间网格区间，采用 Hermite-Simpson 离散化方案生成了 $2n + 1$ 个配置节点。<sup>10</sup> 问题在一系列网格区间密度（5、25、50 和 100）下进行了求解。对于 3-D 仿真，由于在 100 个网格区间下收敛不稳定，所使用的最细网格密度为 75 而非 100。每个最优控制问题均使用两种不同类型的初始猜测求解了两次：一种是动态一致性猜测（CG），其中模型已处于行走状态，但并不接近最终解；另一种是网格细化（MR）策略，即以特定网格密度（例如 5）下的解作为下一个更高网格密度（例如 25）的初始猜测。<sup>7</sup> 对于 CG 情况，每个网格区间密度均使用相同的初始猜测。行走仿真的 CG 猜测是通过求解另一种类型的行走问题生成的，即追踪问题的 CG 是预测问题的解，而预测问题的 CG 是追踪问题的解。CG 还被用于最粗糙的网格密度以启动 MR 过程。生成初始猜测的前期计算成本较低，未计入报告的运行时间内。

离散点对点前向伸展运动的预测仿真是在与行走仿真相同的计算机上求解的。期望的运动起始于手臂垂于躯干侧面且肘部微屈（肩关节抬起 0°，肘关节屈曲 20°），最终目标位置为手臂伸向躯干前方且肘部同样微屈（肩关节抬起 75°，肘关节屈曲 20°）。初始和最终位置的关节角速度均为零。期望的初始和最终关节运动学状态通过在初始和最终时间节点施加边界约束来强制执行。在初始和最终时间之间，关节运动不受其他约束。伸展任务的目标函数是以最短时间完成该动作。选取了一系列网格区间密度（10、25、50、100）来求解该问题。伸展仿真在行走仿真所用的最粗网格区间（5 个区间）下无法收敛；因此，伸展仿真的最粗网格区间密度被设为 10。与行走仿真类似，针对每个网格密度，最优控制伸展问题均分别使用 CG 和 MR 初始猜测求解了两次。伸展仿真的 CG 策略始于模型已在运动但尚未接近最终解的初始猜测。

配备多核处理器的计算机不仅提供了并行求解单个问题的机会，还提供了同时求解多个问题的机会。基于本研究获得的初步结果，我们对使用 MATLAB Parallel Computing Toolbox 中包含的并行 for 循环（parfor）同时生成多个仿真所带来的整体加速比进行了二次分析。作为该二次分析的典型案例，我们使用 6 个处理器核心和 CG 求解了 50 个网格时间间隔上的二维步行预测仿真。我们将顺序生成三次仿真所需的运行时间与在 parfor 循环中同时生成这三次仿真所需的运行时间进行了比较。

## 2.3 | 评估

在不同的处理器核心数量、时间网格密度和初始猜测类型范围内，每种仿真均运行了三次。重复求解同一问题会得到相同的结果；然而，运行时间可能会略有不同。因此，每种情况的计算运行时间取三次独立运行的平均值。使用并行计算相比于串行情况的加速比如公式 (1) 所示进行了量化。

$$
\text { Speed - up } = \frac {\text { serial   runtime }}{n \text { parallel   runtime }}\tag{1}
$$

其中 n 是并行使用的处理器核心数。总运行时间包括在 IPOPT 算法中花费的时间（例如，确定牛顿步长、求解线性系统）以及评估 NLP 函数（例如，目标函数、目标梯度、约束雅可比矩阵）所花费的时间。对于 OpenSim Moco 最终用户而言，只需设置单个 CasADi 参数，即可轻松在多个处理器核心上并行化 NLP 函数评估。原则上，IPOPT 算法中遇到的稀疏线性系统的求解也可以并行化，但实际上这对最终用户来说更为复杂，并且取决于所使用的特定线性求解器（有多种可用求解器）以及该求解器编译所针对的线性代数子程序。在本研究中，我们仅评估了 NLP 函数评估的并行化。用于评估 NLP 函数的总运行时间比例在不同问题之间可能存在差异；因此，我们方法的加速潜力也会有所不同。因此，除了报告整体加速比之外，我们还研究了仅代表 NLP 函数评估部分的加速比，以了解问题的并行化部分在问题内部的扩展规律。

## 3 | 结果

在所测试的网格区间密度、计算机处理器核心数和初始猜测策略下，步态二维追踪模拟的运行时间介于 1.3 min 到 3.4 h 之间（Figure 1A,B）。两种初始猜测策略的总运行时间相近，然而，在大小为 25 和 100（而非 50）的网格区间密度下，采用 MR 策略的模拟运行时间略长于采用 CG 策略的模拟（Figure 1A,B）。步态二维追踪模拟因并行化带来的整体加速微乎其微。无论采用哪种初始猜测策略，最大加速比仅比串行快约 1.7 倍，且大部分加速在约 6 个处理器核心时就已达到，与初始猜测策略无关（Figure 2A,B）。

取决于网格区间密度、计算机处理器核心数和初始猜测策略，步态二维预测模拟的运行时间在 54.9 s 到 33.6 min 之间（Figure 1C,D）。两种初始猜测策略的总运行时间相似，但在大小为 5、50 和 100（而非 25）的网格区间密度下，采用 CG 的模拟运行时间略长于采用 MR 策略的模拟（Figure 1C,D）。最大加速比依初始猜测策略不同在约 3.7–3.8 之间变化，且大部分加速通过 6 个处理器核心即可实现（Figure 2C,D）。

步态三维追踪模拟的运行时间介于 2 到 53 h 之间（Figure 1E,F）。与步态二维预测模拟类似，在大小为 50 和 75（而非 25）的网格区间密度下，采用 CG 的模拟运行时间始终长于采用 MR 策略的模拟（Figure 1E,F）。三维追踪模拟中两种初始猜测策略之间的运行时间差异最大，采用 CG 策略的模拟总体完成速度比采用 MR 策略的模拟慢 2.3 倍（Figure 1E,F）。采用 CG 策略的步态三维追踪模拟的最大加速比比串行快 4.1 倍，而采用 MR 策略时比串行快 3.5 倍，其中大部分加速通过 6–9 个处理器核心实现（Figure 2E,F）。

二维伸手动作预测模拟的运行时间在 1.3 min 至 1.2 h 之间。总体而言，采用 CG 策略的伸手模拟比采用 MR 策略的模拟慢 2.1 倍（Figure 1G,H）。在网格区间密度大小为 5、25 和 100 时，采用 CG 策略的运行时间始终大于采用 MR 策略的模拟运行时间，但在 50 时则并非如此（Figure 1G,H）。二维伸手动作预测模拟实现了最大的加速比，根据初始猜测策略的不同，比串行计算快 7.5–7.7 倍（Figure 2G,H）。超过 9 个核心后进一步加速的潜力取决于时间网格密度（Figure 2G,H）。二维伸手动作预测模拟从并行化中获益最多。无论采用何种初始猜测策略，当使用更细的网格区间密度和更多的处理器核心时，加速比通常都会增加（Figure 2G,H）。

![](images/c442a650c5f5866176231523d7faffdc003cbd7e36d5aae42fac742f6a6e8b38.jpg)

![](images/fdee32944ac33bd5b2b1c405793346ea0167a7958a5835ef19e2df51ed5f88a1.jpg)

(C)  
![](images/b07076c20a850c801dad98119f81c3ba42da8babd3ca5cc6887c924360295e48.jpg)

(D)  
![](images/7b8ff24d05390a8b73e535b44a4860fbcc18796a6821ef7015f21bfb6f238fb5.jpg)

![](images/a60f26ecd53aba2d0d7f5237edce17390ab8da4d789db21752a9bf5f6c440871.jpg)

![](images/cd265eff6f4121c9ea2065106e5f41036f5238300b59c93395963cc9dee56d54.jpg)

![](images/de2b76ee9620e494159e6abf84ffc4204efe4ef4463c6fd4cf1e18575602ab8c.jpg)

![](images/ae2fca480ac03ddaf27cdfb5b8d6effcab0ed4cd510b02417d28120e003e8cc3.jpg)  
FIGURE 1 在一系列计算机处理器核心和网格区间密度下，二维步行追踪模拟（A 和 B）、二维步行预测模拟（C 和 D）、三维步行追踪模拟（E 和 F）以及二维伸手预测模拟（G 和 H）的计算运行时间。请注意不同行图表之间不同的纵坐标范围。

本研究仅评估了每个模拟中 NLP（非线性规划）函数求值的并行化。NLP 函数求值的加速比通常随着网格区间密度和处理器核心数量的增加而增加（Figure 3）。对于二维步行追踪模拟、二维步行预测模拟和三维步行追踪模拟，NLP 函数求值的加速比（Figure 3）大于整体加速比（Figure 2）。对于二维伸手预测模拟，NLP 函数求值的加速比（Figure 3）与整体加速比相似（Figure 2）。

用于 NLP 函数求值的时间占总运行时间的百分比在 2-D 跟踪模拟的 35.8% 到 2-D 伸手预测模拟的 95.6% 之间不等（Table 1）。2-D 步行预测模拟和 3-D 步行跟踪模拟中用于 NLP 函数求值的时间占总运行时间的百分比相近，均略高于总运行时间的一半（Table 1）。

对于 2-D 和 3-D 步行跟踪模拟，最小目标函数值在最粗糙的网格密度下最大，而在更密集的网格上相当一致（Figure 4A,C）。对于 2-D 步行和伸手预测模拟，最小目标函数值在所有网格区间密度下均保持一致（Figure 4B,D）。初始猜测策略对最小目标函数值几乎没有影响（Figure 4）。

![](images/3405f1a695e784b30630c43daa1c33bd70128ae22039c51500ae0e90b5d3321e.jpg)

![](images/18158fcccd32cb16ba7ecc85b7cf59586731a01577faa22f5d97e4c10d24b783.jpg)

![](images/d4ec6af078852652209f53a9df87652e5e4feece34cc7db69f85f0265807766a.jpg)

![](images/38a4a8600e89a3d19eb476a7f8fd69634944ac1225f4266b84c42423607cc196.jpg)

![](images/f3405bfa2c0ec629781a9a2d24e71b94593291d9cb5ddb0b9d1b7d9e45d11a4b.jpg)

![](images/6c2d32b4e8125f0f533cc8959f2e7593b8008e5d969dfab835d390aebdb8b0f8.jpg)

10  
![](images/0c61f0e65cc8d855149dfb232251e7054a822d8fda38d802e0d323467532c573.jpg)  
F I G U R E 2 在一系列计算机处理器核心数和网格区间密度下，2-D 步行跟踪模拟（A 和 B）、2-D 步行预测模拟（C 和 D）、3-D 步行跟踪模拟（E 和 F）以及 2-D 伸手预测模拟（G 和 H）的总体计算加速比。不同模拟的最大加速比在约 1.7 到 7.7 之间变化，其中 2-D 伸手模拟的最大加速比最高，2-D 步行跟踪模拟的最低

在不同的模拟和初始猜测策略下，迭代次数在 21 到 1084 次之间变化。在不同网格区间密度下，两种初始猜测策略求解这四个最优控制问题所需的迭代次数均无一致趋势（Figure 5）。MR 策略使 3-D 步行跟踪模拟和 2-D 伸手预测模拟能够在更密集的网格密度下以显著更少的迭代次数收敛至解（Figure 5C,D）。然而，对于其他问题并非如此，尤其是在 2-D 步行跟踪模拟中（Figure 5A）。

连续三次求解二维预测性行走仿真（50 个网格时间区间，6 个处理器核心，CG）的累积运行时间为 17.8 min。在并行 for 循环中求解相同的三个问题的运行时间为 6.1 min。因此，鉴于本研究中可用的 18 个处理器核心，除了在各个仿真内部并行化 NLP 函数评估所获得的加速之外，还可以实现 2.9 倍的总体加速比。在串行情况下连续三次求解相同的二维预测性行走问题导致的累积运行时间

10  
10  
![](images/64d43b2573a32b32059adf86eaf9cb2d9afe2830f148e5845331671372d61c0e.jpg)

![](images/2d1c96904c23818a04ce2dcfd29559d737b93434406076490e28c6bad1047882.jpg)

![](images/ce66c540dd0f501b19add83aa35d87dff8976ae6ec30fd52cde5808dd91a3322.jpg)

![](images/ec8e73801a159a5b01ecb5270bf8c9c73dc3919e8629186c7a53c586772bea24.jpg)

![](images/18ec9f1f8c3b6e69566817ef3efc0eba2cd4b948e7c916cbeba2c76d3f84a2df.jpg)

![](images/329ccf9e2140f8a6dedee126177611b8c54654d8848bf8cc88b108df4ef18333.jpg)

10  
![](images/a480665fa8f81bd632237e843a9906f3836553213b58aebbf617e7514dc384c8.jpg)

![](images/b20170700621ff5e00e3531211d9de3b7170082036ec3b30f6021a506fcb77f4.jpg)  
F I G U R E 3 在不同计算机处理器核心数和网格区间密度下，最优控制问题的非线性规划（NLP）函数评估计算加速比：二维行走跟踪仿真（A 和 B）、二维行走预测仿真（C 和 D）、三维行走跟踪仿真（E 和 ${ \mathrm { F } } ) ,$ 以及二维伸手预测仿真（G 和 H）。在不同仿真中，最大加速比在约 4.4 到 8.5 之间变化，其中二维行走预测仿真最大，二维行走跟踪仿真最低。

T A B L E 1 求解最优控制问题中用于非线性规划（NLP）函数评估的平均总运行时间百分比（±1 标准差）。

<table><tr><td>仿真类型</td><td>NLP 函数评估耗时占比 (%)</td></tr><tr><td>2-D tracking walking</td><td>35.8 ± 12.8</td></tr><tr><td>2-D predictive walking</td><td>54.3 ± 13.4</td></tr><tr><td>3-D tracking walking</td><td>58.6 ± 17.1</td></tr><tr><td>2-D predictive reaching</td><td>95.6 ± 2.1</td></tr></table>

注：数据为各模型和任务在不同处理器核心数、网格区间数和初始猜测类型下的平均值。

![](images/469a54a401e44df240a08b2bc5a0ba6f6cdda0f6290d850a51422eff9011914f.jpg)

![](images/a6367512fb7ef70e8605f2dd7bad965849624178b8513e80016091824be0a788.jpg)

![](images/4a9bc72a577f99307f1d72435f93bdb7599dbaeabc2dbedba9f7d9e45f2a066e.jpg)

![](images/9d60a73b635e83fd124b207919de75c8b9481ace65db36af1ae2d943faec2b97.jpg)  
F I G U R E 4 使用两种初始猜测策略在一系列网格区间密度下的二维行走追踪模拟 (A)、二维行走预测模拟 (B)、三维行走追踪模拟 (C) 以及二维伸展预测模拟 (D) 的最小目标函数值。对于某些问题（B 和 D），最小目标函数值在不同网格区间密度下保持一致，而对于其他情况（A 和 C），粗网格上的最小目标函数值更大。在不同的网格密度下，初始猜测策略对最小目标函数值几乎没有影响。请注意各面板中纵轴的范围不同。

为 49.9 min。因此，将 NLP 函数求值的并行化与各个模拟的并行求解相结合，可实现 8.2 倍的总体加速比。

## 4 | 讨论

在本研究中，我们评估了在单个计算机工作站中跨一系列计算机处理器内核并行化时，模型复杂度、运动任务、初始猜测类型和时间网格密度如何影响人体运动最优控制模拟的加速比。并行计算在生物力学研究中已在超级计算机和计算机集群上使用了数十年<sup>2,15</sup>，但现在可以在典型的笔记本电脑或台式计算机上实现。OpenSim Moco<sup>10</sup> 中实现的最优控制算法<sup>16</sup>非常适合具有多核处理器的计算机，具有提高计算性能的潜力。对于此处考虑的所有模拟，多核并行计算均在一定程度上提高了计算性能，但加速的程度在不同模拟之间差异很大。

![](images/606f6f8ee4152e1aca64151c82a9f264fa744e64ca5afaf7b2020c2bc73bfdcd.jpg)

![](images/7f0efea044b56c26c6f97368ecff5223986e1c835f0437b252361199c5dea32e.jpg)

![](images/8fdd06cf5dc791cf3f0c3311d5901aea8a642ed39b1ff32c37aa50c7667dd5fc.jpg)

![](images/6d36a3f836d26ea35661b536ebcd8fb09ea4b9425f7848dee7c9e6194bb98b0c.jpg)  
F I G U R E 5 使用两种初始猜测策略在一系列网格区间密度下的二维行走追踪模拟 (A)、二维行走预测模拟 (B)、三维行走追踪模拟 (C) 以及二维伸展预测模拟 (D) 的迭代次数。在不同条件下，网格区间密度和初始猜测策略对收敛到解所需的迭代次数没有一致的影响。

整体加速比与用于评估 NLP（非线性规划）函数所占总运行时间的百分比密切相关。在二维伸手动作预测模拟中，该部分占总运行时间的比例最高，而在二维步行跟踪模拟中最低（Table 1），这两者分别对应最大和最小的有效并行加速比（Figure 2）。二维步行预测问题与三维步行跟踪问题在问题规模和类型上有所不同，但在 NLP 函数评估占总运行时间的百分比上相似（Table 1），从而产生了相近的整体并行加速比（Figure 2）。因此，尽管在 OpenSim Moco 中通过多核并行处理实现的整体加速比因具体问题而异，但它似乎可以通过耗费在 IPOPT 算法中的时间与耗费在评估 NLP 函数上的时间的比例来进行良好预测，这一比例可以很容易地从 IPOPT 程序提供的标准输出中针对任何特定问题确定。对于任何特定问题，精确决定 IPOPT 算法耗时相对于 NLP 函数评估耗时的确切因素很难准确获知，但会受到诸如用户函数评估的计算开销大小，以及解的约束数量和类型等因素的影响。

在大多数情况下，整体加速比在 3 到 6 个处理器核心之间达到最大，其中大部分提升在 3 个处理器核心时就已实现。仅在较精细网格上的伸手动作模拟中，超过 6 个核心后仍有可观的加速比，尽管仍低于理想的线性加速比。因此，对于此处考虑的模拟，大部分可实现的加速比都可以在仅配备 4 个处理器核心的计算机上实现，这在现代计算机中十分普遍。导致加速比低于线性的常见因素是核间通信相关的开销，以及某些核心需要等待其他核心完成后才能继续进行。<sup>26</sup> 然而，如果不进行比本研究更深入的分析，很难获知具体因素在多大程度上影响特定问题中的加速比。拥有更多处理器核心（这会带来经济成本）可以为某些问题（例如上肢伸手任务）带来额外的加速比，或者允许在同一台计算机上同时求解多个问题。我们使用了并行 for 循环来并行求解多个问题，这在 Monte Carlo（蒙特卡洛）模拟或双层优化中可能较为典型，但也可以同时启动多个互不相关的 OpenSim Moco 实例。无论哪种情况，主要考虑的都是如何决定在并行求解的多个 OpenSim Moco 问题内部以及问题之间分配可用的处理器核心。

本研究中可能达到的最大总体加速比受到了限制，因为我们仅研究了非线性规划（NLP）函数求值的并行化。将其作为研究重点，是因为 OpenSim Moco<sup>10</sup> 中使用的 CasADi 库<sup>16</sup>能够方便地选择用于该问题部分的并行线程数。因此，对 NLP 函数求值进行并行化很可能代表了典型用户的操作习惯。OpenSim Moco 中的默认设置是通过 CasADi 求解器利用所有可用核心并行评估 NLP 函数。因此，用户无需执行任何操作即可调用并行处理，仅在需要对处理器核心的分配方式进行精细控制时才需要更改设置。对 IPOPT 中可用的线性方程求解器内部的并行化进行系统研究（例如参考文献 27），可能会进一步提升计算性能，并应成为未来研究的重点。即使在被并行化的问题部分内部，所实现的实际加速比也可能存在差异。因此，我们将 NLP 函数求值所实现的加速比与总体加速比分开进行了评估。当处理器核心数量增加一倍使计算速度也提高一倍时，该加速比被认为是“理想”的。对于大多数问题，NLP 函数求值的加速比显著大于总体加速比。然而，即使在最佳情况下（二维预测性行走，Figure 3C），加速比也未达到理想状态。以往有报道称，在分布式内存并行超级计算机上求解人体行走的直接打靶法问题时，评估目标函数和约束导数获得了接近理想的加速比。<sup>2</sup> 与本研究结果的差异可能是由于采用了不同的最优控制公式构建方式（以往研究中的直接打靶法对比本研究中的直接配点法）以及不同的计算机架构（以往研究中的独立计算节点对比本研究中的单台多核处理器）。

在不同模型与任务、网格区间数与处理器核心数，以及初始猜测类型之间，各个运行时间存在显著差异（Figure 1）。运行时间的变化部分归因于问题收敛所需的迭代次数（Figure 5），而迭代次数对网格密度和初始猜测类型十分敏感。为了完整起见，我们展示了运行时间，然而，我们的主要关注点并非运行时间本身，而是通过并行计算所获得的加速比。与运行时间相比，所获得的加速比在不同网格区间和处理器核心数之间通常表现出一致的扩展性，并且对于两种不同的初始猜测类型表现相似（Figures 2 and 3）。尽管我们在此主要关注并行加速比，但运行时间无疑是一个重要的性能考量因素，它不仅受到并行化程度的影响，还受到网格密度和猜测类型等其他因素的影响（Figure 1）。我们还要指出，报告的运行时间中并未包含获取初始猜测所需的时间。当求解一大组相关问题时（如本研究中所做），少量的初始猜测即可用于所有问题，仅占总计算时间中微不足道的一部分。当从任意或随机的初始猜测开始时，情况也是如此。然而，有时可能需要为每次优化运行获取独特的初始猜测，在这些情况下，所需的时间可能会显著增加。

使用直接配点法（direct collocation）生成人体运动模拟涉及在时间网格上对问题进行离散化。使用粗网格通常会降低计算需求，但可能会损害解的质量。在更精细的网格上求解问题可能会获得更高质量的解，但代价是计算负荷更大。<sup>9</sup>在当前研究中，与在较粗网格区间密度上获得的解相比，在更细网格区间密度上获得的解在跨更多处理器核心并行化时通常会产生更大的加速比。因此，通过并行计算所实现的计算效率有助于抵消使用更精细时间网格密度所带来的计算需求。

本研究存在一些局限性。仅对问题的 NLP 函数求值部分进行并行化限制了可能达到的总体加速比。然而，值得注意的是，这种方法可能反映了典型的 OpenSim Moco 用户对人体运动最优控制模拟进行并行化的方式。此外，我们使用的行走和伸手动作在文献中很常见，但还有许多其他任务也已使用肌肉骨骼模型进行了模拟。目前的结果并不一定能推广到其他任务的模拟中。最后，OpenSim Moco 还可以求解我们在此未考虑的问题，例如确定与规定运动相对应的肌肉状态（即 Moco Inverse 问题）。无法从目前的结果推断出并行计算加速这些其他问题类型的潜力。我们选择重点关注追踪和预测性最优控制问题，因为它们通常在计算上要求最高。

在此，我们使用 OpenSim Moco 中的并行计算分析了广泛的人体运动最优控制模拟的计算需求。尽管实现的加速比并不理想，但在并行求解时，所有受检模拟的计算性能均有所提高，并且同时求解多个问题进一步提高了计算速度。综上所述，通过 CasADi 库在 OpenSim Moco 中进行多核并行计算提供了一种降低最优控制肌肉骨骼模拟计算需求的有效手段，提高了该研究技术的可行性。

## 致谢

我们感谢 Aaron Fox 博士对本文早期草稿提出的有益意见。本研究得到了 Rackham Graduate Student Research Grant 的资助，并在一定程度上受到了 Ros Miller 博士在 OpenSim Moco 论坛上关于在不同时间网格密度下获得的模拟结果的帖子的启发。

## 利益冲突声明

作者声明不存在利益冲突。

## 数据可用性声明

我们在 SimTK 网站（https://simtk.org/projects/ mocoparallel）上提供了一个完整的运行示例。目前，在我们准备和上传所有内容期间，该项目页面设置为私有。我们将在论文发表后公开发布。

## ORCID

Alex N. Denton https://orcid.org/0000-0002-4939-4819 Brian R. Umberger https://orcid.org/0000-0002-5780-2405

## 参考文献

1. Anderson FC, Pandy MG. Dynamic optimization of human walking. J Biomech Eng. 2001;23(5):381-390. doi:10.1115/1.1392310

2. Anderson FC, Ziegler JM, Pandy MG, Whalen RT. Application of high-performance computing to numerical simulation of human movement. J Biomech Eng. 1995;117(1):155-157. doi:10.1115/1.2792264

3. Crowninshield RD, Brand RA. A physiologically based criterion of muscle force prediction in locomotion. J Biomech. 1981;14(11):793- 801. doi:10.1016/0021-9290(81)90035-X

4. Umberger BR, Caldwell GE. Musculoskeletal modeling. In: Robertson DGE, Caldwell GE, Hamill J, Kamen G, Whittlesey SN, eds. Research Methods in Biomechanics. 2nd ed. Human Kinetics; 2014:247-276.

5. Ackermann M, van den Bogert AJ. Optimality principles for model-based prediction of human gait. J Biomech. 2010;43(6):1055-1060. doi:10.1016/j.jbiomech.2009.12.012

6. Porsa S, Lin Y-C, Pandy MG. Direct methods for predicting movement biomechanics based upon optimal control theory with implementation in OpenSim. Ann Biomed Eng. 2016;44(8):2542-2557. doi:10.1007/s10439-015-1538-6

7. Lee L-F, Umberger BR. Generating optimal control simulations of musculoskeletal movement using OpenSim and MATLAB. PeerJ. 2016;4:e1638. doi:10.7717/peerj.1638

8. Lin Y-C, Pandy MG. Three-dimensional data-tracking dynamic optimization simulations of human locomotion generated by direct collocation. J Biomech. 2017;59:1-8. doi:10.1016/j.jbiomech.2017.04.038

9. Kelly M. An introduction to trajectory optimization: how to do your own direct collocation. SIAM Rev. 2017;59(4):849-904. doi:10.1137/16M1062569

10. Dembia CL, Bianco NA, Falisse A, Hicks JL, Delp SL. OpenSim Moco: musculoskeletal optimal control. PLoS Comput Biol. 2020;16(12):e1008493. doi:10.1371/journal.pcbi.1008493

11. Seth A, Hicks JL, Uchida TK, et al. OpenSim: simulating musculoskeletal dynamics and neuromuscular control to study human and animal movement. PLoS Comput Biol. 2018;14(7):e1006223. doi:10.1371/journal.pcbi.1006223

12. Nguyen VQ, Johnson RT, Sup FC, Umberger BR. Bilevel optimization for cost function determination in dynamic simulation of human gait. IEEE Trans Neural Syst Rehabil Eng. 2019;27(7):1426-1435. doi:10.1109/TNSRE.2019.2922942

13. Johnson RT, Lakeland D, Finley JM. Using Bayesian inference to estimate plausible muscle forces in musculoskeletal models. J Neuroeng Rehabil. 2022;19:34. doi:10.1186/s12984-022-01008-4

14. Amdahl GM. Validity of the single processor approach to achieving large scale computing capabilities. Proceedings of the Spring Joint Computer Conference, 483–485. 1967. doi:10.1145/1465482.1465560

15. van Soest AJ, Casius LJ. The merits of a parallel genetic algorithm in solving hard optimization problems. J Biomech Eng. 2003;125(1):141-146. doi:10.1115/1.1537735

16. Andersson JAE, Gillis J, Horn G, Rawlings JB, Diehl M. CasADi: a software framework for nonlinear optimization and optimal control. Math Program Comput. 2019;11:1-36. doi:10.1007/s12532-018-0139-4

17. Miller RH, Russell Esposito E. Transtibial limb loss does not increase metabolic cost in three-dimensional computer simulations of human walking. PeerJ. 2021;9:e11960. doi:10.7717/peerj.11960

18. Fox AS, Bonacci J, Gill SD, Page RS. Simulating the effect of glenohumeral capsulorrhaphy on kinematics and muscle function. J Orthop Res. 2021;39(4):880-890. https://doi-org.proxy.lib.umich.edu/10.1002/jor.24908

19. De Groote F, Kinney AL, Rao AV, Fregly BJ. Evaluation of direct collocation optimal control problem formulations for solving the muscle redundancy problem. Ann Biomed Eng. 2016;44(10):2922-2936. doi:10.1007/s10439-016-1591-9

20. Sherman MA, Seth AJ, Delp SL. Simbody: multibody dynamics for biomedical research. Procedia IUTAM. 2011;2:241-261. doi:10.1016/j piutam.2011.04.023

21. Seth A, Sherman M, Reinbolt JA, Delp SL. OpenSim: a musculoskeletal modeling and simulation framework for in silico investigations and exchange. Procedia IUTAM. 2011;2:212-232. 10.1016/j.piutam.2011.04.021

22. Holzbaur KR, Murray WM, Delp SL. A model of the upper extremity for simulating musculoskeletal surgery and analyzing neuromuscular control. Ann Biomed Eng. 2005;33(6):829-840. doi:10.1007/s10439-005-3320-7

23. Wächter A, Biegler LT. On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. Math Program. 2006;106:25-57. doi:10.1007/s10107-004-0559-y

24. van den Bogert AJ, Blana D, Heinrich D. Implicit methods for efficient musculoskeletal simulation and optimal control. Procedia IUTAM. 2011;2:297-316. doi:10.1016/j.piutam.2011.04.027

25. Miller RH, Edwards WB, Brandon SE, Morton AM, Deluzio KJ. Why don't most runners get knee osteoarthritis? A case for perunit-distance loads. Med Sci Sports Exerc. 2014;46(3):572-579. doi:10.1249/MSS.0000000000000135

26. Shameem A, Jason R. Multi-Core Programming-Increasing Performance through Software Multithreading. Intel; 2005.

27. Tasseff B, Coffrin C, Wächter A, Laird C. Exploring benefits of linear solver parallelism on modern nonlinear optimization applications. arXiv:1909.08104. 2019 10.48550/arXiv.1909.08104