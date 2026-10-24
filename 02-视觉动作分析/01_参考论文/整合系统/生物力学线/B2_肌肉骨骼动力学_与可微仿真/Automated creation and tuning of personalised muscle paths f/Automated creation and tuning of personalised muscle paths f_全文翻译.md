1 用于膝关节 OpenSim 肌骨模型的个体化肌肉路径自动化构建与调整

2 

Killen BA<sup>1,2,3</sup>, Brito da Luz S<sup>1,2</sup>, Lloyd DG<sup>1,2</sup>, Carleton AD<sup>4</sup>, Zhang J<sup>4</sup>4 , Besier TF<sup>4</sup>, Saxby DJ<sup>1,2</sup>5

6 1. 格里菲斯生物医学与康复工程中心（Griffith Centre of Biomedical and Rehabilitation Engineering, GCORE），7 昆士兰孟席斯健康研究所（Menzies Health Institute Queensland）及先进设计与原型技术研究所（Advanced Design and Prototyping Technologies Institute, ADAPT），格里菲斯大学（Griffith University），黄金海岸，9 昆士兰州，澳大利亚

2. 专职医疗科学学院（School of Allied Health Sciences），格里菲斯大学（Griffith University），黄金海岸，昆士兰州，澳大利亚

3. 人体运动生物力学研究组（Human Movement Biomechanics Research Group），鲁汶大学（KU Leuven），鲁汶，比利时

4. 奥克兰生物工程研究所（Auckland Bioengineering Institute），奥克兰大学（University of Auckland），奥克兰，新西兰

17 通讯作者：

18 Dr Bryce A Killen

19 Human Movement Biomechanics Research Group, KU Leuven, Belgium

20 电子邮箱：bryce.killen@kuleuven.be

## 摘要

计算建模是研究人体运动和运动控制特征的宝贵工具，而这些特征通常只能通过侵入性技术才能测量。近期的研究重点集中在利用基于人群的形态变形或直接利用医学影像来构建个体化肌骨模型。尽管已经取得了进展，但对两个关键模型参数进行鲁棒定义仍然具有挑战性：(i) 完整的胫股（tibiofemoral, TF）和髌股（patellofemoral, PF）关节运动，以及 (ii) 肌腱单元（muscle tendon unit, MTU）路径和运动学（即长度与力臂）。本研究旨在开发一个自动化框架，采用基于人群的形态变形方法来构建包含个体化骨骼几何形状、TF 与 PF 关节机制以及 MTU 路径和运动学的个体化肌骨模型。基于医学影像信息，构建了个体化的刚体 TF 和 PF 关节机制。采用基于图谱和优化的方法构建了个体化 MTU 路径和运动学，旨在防止 MTU 穿透骨骼，并实现符合现有文献模式的平滑 MTU 运动学。该框架被集成到 Musculoskeletal Atlas Project Client 软件包中，为 6 名参与者构建并优化了具有逐步提高的个体化水平的模型，以期改善 MTU 运动学和路径。研究共进行了三组对比：(i) 具有通用关节机制的未优化模型（Model 1）与优化模型（Model 3）；(ii) 具有个体化关节机制的未优化模型（Model 2）与优化模型（Model 4）；以及 (iii) 两种优化模型之间（Model 3 与 Model 4）。优化后，在对比 i 和 ii 中，与尸体数据的模式相似性均表现出一致的改善。对于对比 iii，多项比较显示所对比的两个模型之间没有显著差异。重要的是，在任何情况下优化均未产生统计学上显著变差的结果。

## 引言

人体肌肉骨骼系统的计算模型使研究人员能够在无需进行侵入性和昂贵实验的情况下研究体内生物力学。刚体肌肉骨骼建模工具，例如 AnyBody Modeling Software AnyBody（AnyBody Technology，Aalborg，Denmark）和 OpenSim（Delp et al. 2007；Seth et al. 2018），已被用于研究广泛的运动、健康和工业问题，并用于估算日常活动中的肌腱单位（muscle tendon unit, MTU）力和关节接触力等体内机体力学特征（Winby et al. 2009；Ackland et al. 2011；Cleather and Bull 2011；Guess et al. 2014；Saxby et al. 2016；Konrath et al. 2017；Andersen 2018；Modenese et al. 2018）。通常，研究中会使用骨骼几何形态和 MTU 路径的通用模型，但即使经过精细缩放，这些模型也不太可能反映个体解剖结构（Kainz et al. 2017；Davico et al. 2020a）。

线性缩放的模型可能无法很好地表征 MTU 力臂，尽管尸体测量中存在差异，却在不同受试者之间产生几乎相同的值（Fick 1879；Draganich et al. 1987；Visser et al. 1990；Spoor and van Leeuwen 1992；Buford et al. 1997；Pal et al. 2007；Wilson and Sheehan 2009；Arnold et al. 2010；Navacchia et al. 2017）。此外，这些通用模型通常包含不允许 6 自由度（degree of freedom, DOF）的胫股关节（tibiofemoral joint, TFJ）和髌股关节（patellofemoral joint, PFJ），例如将外展/内收以及内旋/外旋设置为 0。因此，这些模型可能不适合通过运动仿真来准确估算常见的胫股关节变量（Gerus et al. 2013；Demers et al. 2014；Lerner et al. 2015）。为了克服这些局限性，可以使用个性化模型。

肌肉骨骼模型中的许多特征均可实现个性化，包括骨骼几何形态、环节质量与惯量、关节解剖结构与运动学，以及 MTU 内部参数和路径（Saxby et al. 2020）。先前的研究表明，纳入个性化特征对仿真中的胫股关节变量具有显著影响（例如关节力矩（Reinbolt et al. 2007）以及接触载荷（Gerus et al. 2013；Lerner et al. 2015））。尽管已有若干研究提出了纳入个性化骨骼解剖结构（Scheys et al. 2009；Wesseling et al. 2016；Valente et al. 2017；Modenese et al. 2018）和关节运动学函数（Sancisi and Parenti-Castelli 2011a, b；Brito da Luz et al. 2017；Dzialo et al. 2018；Barzan et al. 2019；Smale et al. 2019）的方法，但在具有个性化骨骼几何形态、关节解剖结构和关节运动学的模型中，鲜有研究报道定义 MTU 路径的方法（Scheys et al. 2009；Nolte et al. 2016；Modenese et al. 2018；Modenese and Kohout 2020）。许多个性化方法依赖于人工用户输入，且鲜少与尸体 MTU 运动学（即长度和力臂）进行对比。

创建个性化高保真肌骨模型需要采集大量的医学影像。统计形状建模（Statistical Shape Modelling, SSM）允许基于动作捕捉（MOCAP）数据，或结合极少量的医学图像进行基于人群的形变（Zhang et al. 2014, 2016; Nolte et al. 2016, 2020; Bahl et al. 2019; Suwarganda et al. 2019; Bakke and Besier 2020; Davico et al. 2020a）。这些方法已在肌骨图谱项目（Musculoskeletal Atlas Project, MAP）Client（Zhang et al. 2014）中实现。与线性缩放相比，该工具能够通过形变创建精准重构解剖结构的 OpenSim 模型（Zhang et al. 2016; Bahl et al. 2019; Suwarganda et al. 2019; Davico et al. 2020a），进而提高仿真的可重复性（Bakke and Besier 2020）。然而，如何从形变数据中自动生成肌肉肌腱单元（MTU）路径（即无需明确的肌肉影像数据）仍未明确。

因此，本文提出了一个基于 MAP-Client（Zhang et al. 2014）的框架，该框架可自动创建并调整个性化 OpenSim 肌骨模型，并重点关注胫股关节（TFJ）。用于创建这些模型的步骤旨在自动执行通常需要手动完成的任务，并通过算法约束来模拟人工检查。这些人工检查包括确保 MTU 不穿透骨骼表面，以及 MTU 运动学符合已有数据（即测得的长度和力臂）。它们还包括检查 MTU 是否执行正确的动作（即伸肌产生伸展力矩），我们将此称为 MTU 极性。个性化特征包括骨骼几何形状、关节轴定义、TFJ 和髌股关节（PFJ）运动学机制、MTU 起止点以及选定的跨 TFJ MTU 的中间路径。MAP-Client 结合所开发的开源软件用于生成四个具有不同个性化程度的 OpenSim 模型，随后在以下假设下进行了实施和测试。首先（H<sub>1</sub>），MTU 包络面的优化将提高 MTU 运动学与文献报道结果的相似性。其次（H<sub>2</sub>），包络面优化将防止 MTU 穿透骨骼并避免 MTU 极性错误。第三（H<sub>3</sub>），MTU 包络面的优化将提高 MTU 长度和力臂的平滑度。最后（H<sub>4</sub>），与具有优化 MTU 包络面但采用通用 TFJ 和 PFJ 机制的模型相比，具有优化 MTU 包络面以及个性化 TFJ 和 PFJ 机制的模型将产生更符合生理特性的 MTU 运动学（即防止 MTU 穿透骨骼和 MTU 极性错误，并提高 MTU 长度和力臂的平滑度）。

## 方法

## 动作捕捉与磁共振成像

数据收集于 Griffith University 作为一项正在进行的项目（PES/36/10/HREC）的一部分。从一个更大的队列中挑选了六名受试者，以涵盖年龄、身高和体重的范围（Table 1）。受试者均无肌肉骨骼损伤、创伤或下肢手术史。在进行全面的动作捕捉（MOCAP）和医学影像检查之前，每位受试者均签署了书面知情同意书。静态校准试验期间的三维（3D）标记点位置通过 MOtoNMS（Mantoan et al. 2015）从标准 MOCAP（即 c3d）格式转换为 OpenSim（trc）格式，以供 MAP-Client 使用。

## [Table 1]

每位受试者均在当地的一家放射诊所（QScan Southport, QLD, Australia）接受了下肢磁共振成像（MRI）检查，该检查在 MOCAP 当天或前一天进行。在受试者仰卧于 3 T MRI 扫描仪（Philips Medical Systems, Netherlands）中的状态下，采集双侧从髂嵴上方至脚趾下方的轴位 T<sub>1</sub> 加权 3D 快速场回波序列。使用体线圈分 5 个扫描站采集图像，每站约 245 层，全程站间重叠 10 mm，层厚 1 mm，层间距 1 mm，体素大小为 0.79 mm<sup>3</sup>，视野为 446 mm x 446 mm。

使用 Mimics v19（Materialise, Leuven, Belgium）对骨盆（不包括骶骨）、股骨、胫腓骨和髌骨进行分割。还对随机选择的单侧肢体进行了专门的 TFJ（胫股关节）和 PFJ（髌股关节）扫描（Table 1）。关节扫描包含从大腿中段至胫骨结节下方采集的 3D 质子密度 16 通道序列。图像在单个扫描站（约 440 层）采集，层厚 0.6 mm，层间距 0.3 mm，体素大小为 0.79 mm<sup>3</sup>，视野为 446 mm x 446 mm。这些图像用于分割股骨远端、胫骨近端和髌骨骨骼，以及股骨、胫骨和髌骨软骨，以及前交叉韧带、后交叉韧带和内侧副韧带附着区。

## 创建个性化 OpenSim 模型

创建了四种具有递增个性化程度的 OpenSim 模型（Table 2）。基于各模型在生成兼具生理学与解剖学合理性的肌腱单位（MTU）运动学方面的表现进行了比较（见下文及 Appendix 2）。简而言之，生理学合理运动学是指平滑且符合以往发表的尸体测量模式的运动学特征（Fick 1879; Draganich et al. 1987;

Visser et al. 1990; Spoor and van Leeuwen 1992; Buford et al. 1997; Pal et al. 2007; Wilson and Sheehan 2009; Arnold et al. 2010; Navacchia et al. 2017）。解剖学合理是指 MTU 不穿透骨骼或产生非物理路径（即包绕表面的圆周环路）。

## 个性化 OpenSim 模型创建

模型是通过结合 MOCAP（动作捕捉）和 MRI（磁共振成像），使用 MAP-Client 创建的（Zhang et al. 2016）。MAP-Client 利用图形界面以及不同的几何拟合方法来重建（Bahl et al. 2019; Suwarganda et al. 2019; Davico et al. 2020a）下肢的特定骨骼（即骨盆、股骨、胫骨、腓骨和髌骨）。简而言之，在 MOCAP 中采集的标记点位置被用于缩放包含骨盆、股骨、胫骨、腓骨和髌骨的统计形状模型（statistical shape models, SSM）。随后，缩放后的 SSM 通过迭代最近点方法配准到 MRI 骨骼分割。配准后的 SSM 接着使用主网格拟合进行变形，以紧密贴合 MRI 分割。最后，采用点对点（即局部）变形对变形后的 SSM 进行微调，以匹配 MRI 分割。这一先前经过验证的过程（Suwarganda et al. 2019）针对每块骨骼分别完成，这些变形后的骨骼随后被用于创建个性化的 OpenSim 模型。

个性化 OpenSim 模型实际上是具备了个性化骨骼几何形状、关节位置、体重和惯性特性的通用 OpenSim 模型（Delp et al. 2007）（Zhang et al. 2014, 2016）。仅对骨盆、股骨、胫骨、腓骨和髌骨进行了个性化处理，而踝足部则进行了各向同性缩放，因为 MAP-Client 尚不支持足-踝复合体的统计形状模型。

## 定义肌腱单元路径

MTU（肌腱单元）的起点和止点是使用基于模板的模型定义的（Zhang et al. 2014, 2016），即 SOMSO（Marcus Sommer SOMSO Modelle，德国索讷贝格）。SOMSO 是一种用于解剖学教学的实体模型，包含一套下肢骨骼及其相关的 MTU 附着区域。在 SOMSO 模型上，若干显著的骨骼位点以及 MTU 附着区域的质心被数字化。利用这些骨骼位点，MTU 附着质心被投影到与 SOMSO 骨骼等效的 MAP-Client 生成骨骼上的最近节点。随后，这些质心被用于定义 MTU 的起点和止点。

为了生成 MTU 路径，MAP-Client 同时使用了与通用模型一致的固定路径点和条件路径点。然而，路径点往往会在 MTU 运动学中引入不连续性（Garner and Pandy 2000; Hammer et al. 2019）以及非生理性的肌肉形状（附录 1）。为了克服这些局限性并确保与近期的 OpenSim 模型（Arnold et al. 2010; Rajagopal et al. 2016; Catelli et al. 2019）保持一致，本研究采用了包络面（wrapping surfaces）。

包络面参数，即位置、朝向和尺寸，是基于拟合到骨骼区域的解析形状以及 MTU（肌肉-肌腱单元）路径点的位置来确定的。包络面参数使用 MAP-Client 自动定义，并与全身模型（Fullbody Model，Rajagopal et al. 2016）进行了比较。然而，这种方法并不能保证 MTU 运动学（即长度和力臂）在生理学和解剖学上是合理的，因此需要进一步优化。

为了产生生理学上合理的 MTU 运动学和解剖学上合理的 MTU 路径，对包络面的几何参数进行了优化。该优化使用了 pyswarm（一个支持约束的粒子群优化 (PSO) Python 软件包，Abraham D. Lee，https://pythonhosted.org/pyswarm/），包含两个自定义目标函数和三个自定义惩罚函数（详细说明见附录 2）。先前的研究报告了尸体标本之间在 MTU 运动学幅值方面的差异（Fick 1879；Draganich et al. 1987；Visser et al. 1990；Spoor and van Leeuwen 1992；Buford et al. 1997；Pal et al. 2007；Wilson and Sheehan 2009；Arnold et al. 2010；Navacchia et al. 2017）。因此，第一个目标称为归一化梯度误差，它通过归一化梯度促使模型跟踪基于 MTU 运动学模式而非绝对幅值的尸体文献数据。后者是 MTU 长度或力臂相对于关节角度变化的幅值变化（即梯度）除以目标（即尸体）数据梯度。第二个目标称为平滑度，促使 MTU 运动学模式保持平滑。惩罚函数用于生成解剖学上合理的 MTU 路径。第一个惩罚函数降低了建模的 MTU 路径穿透骨骼的可能性。穿透是通过光线投射法（ray casting）检测的。在每个相邻的 MTU 路径点之间定义一条光线。加载 MTU 可能穿透的骨骼（基于其跨越的关节），并计算骨骼网格上每个点到光线的距离。如果该距离低于指定值，则判定为穿透骨骼。第二个惩罚函数称为极性误差，防止力臂不恰当地改变其力学作用，例如屈肌变成伸肌；第三个惩罚函数降低了非解剖学包络情况的可能性（附录 2）。一旦为每位参与者完成了包络面优化，便使用 MAP-Client 生成个性化的 TFJ（胫股关节）和 PFJ（髌股关节）运动学机制。

## 关节机制定义

创建了两套不同的 TFJ（胫股关节）和 PFJ（髌股关节）机构：缩放通用型（Model 1 和 3）和个性化型（Model 2 和 4）。胫股关节的运动范围被限制在 0（完全伸展）到 -100 度（屈曲）之间，涵盖了步行和跑步过程中的运动范围（Novacheck 1998）。缩放通用 TFJ 机构为三自由度（DOF）关节，具有一个独立的屈曲-伸展自由度和两个定义为 TFJ 屈曲-伸展函数的从属平移自由度（即前/后向和平移以及上/下向平移）。平移函数被定义为样条曲线，以中立位为参考，在整个运动范围内维持股骨和胫骨内外侧髁之间的距离（Zhang et al. 2016）。其余自由度（即内/外侧平移、外展/内收旋转和内旋/外旋）被锁定为零。随后添加了髌骨和 PFJ，髌骨原点的位置与先前的模型（Arnold et al. 2010）一致，并且其相对于胫骨的位置是固定的。

随后从 MRI 分割数据创建了个性化 TFJ 和 PFJ 机构（Sancisi and Parenti-Castelli 2011a, b; Brito da Luz et al. 2017; Barzan et al. 2019）。将分割后的解剖结构（即骨骼、软骨和韧带）导入 3-matic v10 (Materialise, Leuven, Belgium) 中，在其中识别出定义关节机构所需的表面和标志点（例如韧带附着区域）。然后使用多目标粒子群优化（Multi-Objective Particle Swarm Optimization (MOPSO) by Yarpiz, 2015）来确保生理学解，从而产生大量候选解，并通过一系列完整描述 6 个自由度 TFJ 运动的样条函数进行拟合：1 个独立自由度（即屈曲-伸展）和 5 个从属自由度（即次级运动学）。每个 TFJ 解都与基于与尸体 PFJ 运动学相关性选出的唯一个性化 PFJ 机构解相配对（Brito da Luz et al. 2017; Barzan et al. 2019）。随后将候选个性化 TFJ-PFJ 模型与 MAP-Client 生成的模型（Model 1）相结合，产生多个候选个性化 OpenSim 模型，即每个 TFJ 解对应一个 Model 2。

最终的个性化 TFJ-PFJ 方案是基于 MTU（肌腱单元）运动学评估指标选出的。使用 MTU 包绕面优化中所采用的目标函数测试了选定的 MTU（股直肌、股内侧肌、股外侧肌、半膜肌、股二头肌长头和腓肠肌内侧头）运动学（见上文和附录 2）。将每个模型的评估指标相加，并根据最小的总和值选择 TFJ-PFJ 方案，理论上该方案代表最符合生理状态的 MTU 运动学。随后运行 MTU 包绕面优化以生成最终的个性化模型（Model 4）。

## 模型比较和肌腱单元运动学评估

模型创建完成后，计算了每个模型（Table 2）的 MTU（肌腱单位）运动学评估指标：(i) 力臂极性惩罚次数，(ii) 归一化 MTU 力臂梯度误差，(iii) MTU 骨穿透惩罚次数，以及 (iv) MTU 长度与力臂平滑度。针对每个 MTU、模型和受试者分别计算了这些指标。为了进行比较，将 MTU 分为以下几组：(i) 股四头肌（左侧和右侧股直肌、股外侧肌、股中间肌和股内侧肌），(ii) 腘绳肌（左侧和右侧股二头肌长头与短头、半腱肌和半膜肌），以及 (iii) 其他肌群（左侧和右侧腓肠肌内侧头与外侧头、缝匠肌和股薄肌）。对于每个 MTU 指标，计算了频数计数（即惩罚次数）或均值 ± 标准差（即平滑度和误差）。

在每次比较中，优胜模型由评估指标的大小决定，数值越小表示 MTU 路径或运动学在生理学/解剖学上越合理。每次比较均被标记为 (i) 改善、(ii) 恶化或 (iii) 无变化，并针对每个 MTU 组及所有 MTU 汇总计数（即出现次数）。使用比例检验对指标计数进行比较，以确定统计学显著性差异（Wessa P.; 2016, Testing Population Proportion (v1.0.3)）。为了计算 z 分数（z-scores），设定了 50% 的原假设，显著性水平设为 $p<0.05$。确定了占主导地位的结果（即改善、恶化或无变化）并检验了其统计学显著性。

## 结果

在标准计算机（2.4 GHz Intel i5 处理器，8 GB 内存）上为每个受试者生成全部四套模型所需的总时间约为 13.5 小时，外加用于优化的 2 小时 HPC（高性能计算）（Table 3）。

## [Table 3]

## 具有通用关节机制的模型中微调的效果

当使用通用 TFJ（胫股关节）和 PFJ（髌股关节）模型时，与未优化模型（即 Model 1）相比，包络面优化（即 Model 3）产生了改善的结果（Table 4、Fig 1 和 2）。唯一表现出统计学显著改善的 MTU 指标是 MTU 力臂梯度误差，该指标在优化后（Model 3）得到改善。所有其余指标要么显示出无变化情况的显著比例（即力臂极性惩罚、MTU 骨穿透惩罚 [其他 MTU 组除外] 以及 MTU 长度平滑度），要么未显示出显著比例（即 MTU 力臂平滑度，以及其他 MTU 组中的 MTU 骨穿透）。尽管如此，在所有 MTU 组中，力臂极性惩罚、MTU 骨穿透惩罚和 MTU 长度平滑度的改善案例数量均高于恶化案例数量。类似地，除股四头肌组显示出改善和恶化案例分布均等外，MTU 力臂平滑度在所有 MTU 组中均显示出更多的改善案例。

## [Table 4]

[Fig 1]

## [Fig 2]

## 具有个性化关节机制的模型中微调的效果

具有个性化关节运动学模型且采用未优化（Model 2）与优化（Model 4）包络面的模型比较结果与先前的比较相似（Table 5）。同样，唯一表现出统计学显著改善的指标是 MTU 力臂梯度误差，该指标在优化模型（Model 4）中得到了改善。在力臂极性惩罚（腘绳肌 MTU 组除外）、MTU 骨穿透惩罚（额外 MTU 组除外）以及 MTU 长度平滑度中，均表现出显著比例的无变化案例。而在腘绳肌 MTU 组的力臂极性惩罚、额外 MTU 组的 MTU 骨穿透惩罚以及 MTU 力臂平滑度中，未发现占显著比例的情况。在排除无变化案例后，MTU 力臂平滑度中改善的案例数量多于恶化的案例。此外，对于全部（All）MTU 组和额外 MTU 组，在力臂极性惩罚、MTU 骨穿透惩罚和 MTU 长度平滑度方面，改善的案例数量也均多于恶化的案例。

[Table 5]

## 优化模型的比较

在具有通用关节（Model 3）和个性化关节（Model 4）的优化模型之间进行的最终比较显示出不一致的结果（Table 6；Fig 3 和 4）。在所有受试者中，对于全部 MTU、股四头肌和腘绳肌组，在模型间的 MTU 力臂极性惩罚方面均有显著比例的案例显示无变化。同样，全部 MTU、股四头肌和腘绳肌组在模型间的 MTU 骨穿透惩罚上也表现为无变化。对于所有其余的比较，未发现占主导地位的情况，在改善或恶化的案例中也均未发现明显趋势。

[Table 6]

## [Fig 3]

## [Fig 4]

## 讨论

本研究旨在开发并测试一个基于 MAP-Client 构建的框架，用于自动微调个性化 OpenSim 肌肉骨骼模型，特别是针对膝关节。所提出的工作流程将以往手动或半手动执行的任务实现了自动化，并且基于统计形状建模（statistical shape modelling），利用动作捕捉（MOCAP）和医学影像对骨骼进行形变，从而生成肌肉起点、止点及路径。这是通过将传统上手动执行的质量检查转化为表述为优化问题的数学算法来实现的。该优化过程能够自动检测误差并调整模型以最小化这些误差，而无需耗时且主观的人工干预。总体而言，经过优化后，在通用和个性化关节机制模型中，大多数 MTU 评估指标均表现出改善。重要的是，该框架提出了一种微调模型肌肉路径的方法，该方法可以从当前的应用（即膝关节）拓展到身体的其他部位。

## 未优化与优化模型的比较

第一个假设（H<sub>1</sub>）得到了证实，即与未优化模型相比，包裹面的优化能够提高模型肌腱单元（MTU）运动学与尸体测量结果的相似性。无论是否采用个性化膝关节机制，优化的包裹面均使所有 MTU 组的力臂梯度误差显著降低（表 3 和表 4）。尽管这一结果并不令人意外（因为力臂梯度误差已被作为优化目标函数包含在内），但这种持续的改善为所构建的框架提供了信心，并表明优化后的肌肉具有相似的力臂模式。

第二和第三个假设（H<sub>2</sub> 和 H<sub>3</sub>）未能得到证实，即与未优化模型相比，优化模型会减少 MTU 骨穿透和力臂极性错误的发生率，并改善 MTU 运动学平滑度。无论是通用关节还是个性化关节，在优化包裹面后，MTU 力臂极性、骨穿透以及长度和力臂平滑度的出现频率均未显示出显著变化。包裹面优化后变化有限的可能原因在于最终加权值的设计。综合考虑每个受试者和模型的评估指标（附录 3，表 6-9），尽管平滑度误差和梯度误差均已归一化，但各个目标函数的数量级差异巨大。具体而言，力臂梯度误差要大得多，这可能解释了为什么这是唯一表现出一贯且具有统计学显著改善的指标。采用不同的归一化方法，可能会对本文中难以改善的其他指标带来更为一致且具有统计学显著性的变化。此外，目标函数中的不同项可能存在相互竞争，这也可能部分解释了为何模型中各类 MTU 惩罚项（骨穿透和极性）未能被优化器有效最小化。未来的工作应研究新的归一化方法或潜在的目标权重分配，以确定是否可以取得进一步的改进。尽管缺乏统计学显著性，但在大多数情况下，优化后 MTU 的运动学和路径（图 5）均得到了改善。

## [图 5]

## 优化模型的比较

最终假设（H<sub>4</sub>），即经过优化后具有个性化关节机制的模型将产生更符合生理学和解剖学合理性的 MTU 运动学，并未得到一致支持。这些结果表明，无论采用何种关节机制，所设计的优化框架均可产生质量相近的 MTU 运动学（通过我们提出的一套评估指标进行评估）。由于纳入这些个性化关节机制需要额外的数据和处理时间，只要能够建立在生理学和解剖学上合理的 MTU 运动学，具有通用 TFJ 和 PFJ 的模型可能就足以支持肌肉驱动的模拟。尽管评估指标显示出相似的结果，但需要指出的是，MTU 运动学、关节运动学和路径的幅值存在差异（Appendix 4）。由于这些参数（MTU 长度和力臂）无法在关节整个运动范围内进行可行性测量，因此无法得出哪个模型更能代表体内（in vivo）MTU 运动学或路径的结论。

## 与既往方法的比较

我们提出的用于自动创建和调整肌腱单元（MTU）路径的框架与以往的研究（Scheys et al. 2009; Nolte et al. 2016, 2020; Modenese et al. 2018; Modenese and Renault 2020）具有相似之处，特别是在对自动化的关注上。我们结合解剖图谱数据来定义 MTU 起点和止点的方法与先前的研究（Nolte et al. 2016）一致，并且可以通过额外的医学影像进行补充（Scheys et al. 2009）。然而，我们定义中间 MTU 路径的方法与以往的方法有所不同。在过去的研究中，尸体标本被用于实验，MTU 路径通过直接测量并配准到下层骨骼上离散化为路径点。这种方法既用于通用模型（Delp et al. 2007），也用于受试者特异性模型（Scheys et al. 2009）。较新的研究选择了包络面（wrapping surfaces）而非路径点，因为它们能产生更平滑的 MTU 运动学（Garner and Pandy 2000; Hammer et al. 2019）。然而，这些包络面的尺寸调整、定位以及在非球形物体情况下的朝向确定，通常由操作人员手动完成（Garner and Pandy 2000; Rajagopal et al. 2016; Lai et al. 2017; Catelli et al. 2019; Hammer et al. 2019）。针对跨髋关节肌肉的 MTU 路径定义也已提出自动化方法（Modenese and Kohout 2020），但该方法依赖于目标肌肉的完整分割。本研究中方法的优势在于，它们不需要依赖通过昂贵的医学成像和肌肉分割所获取的关于每块肌肉形状、大小或其他形态特征的明确信息。尽管当前的框架确实依赖于骨骼的成像和分割，但这些可以通过使用不完全骨骼分割（Suwarganda et al. 2019）或光学动作捕捉标记点的 3D 位置（Nolte et al. 2016, 2020; Davico et al. 2020a）的不同变形方法来替代。

相反，我们的方法依赖于在通过标准 MAP-Client 流程生成的模型中测试 MTU 的行为，并且可以轻松扩展到其他肌肉和关节。尽管我们将优化重点放在受试者特异性模型上，但其自身特性意味着它们也可以应用于通用模型，以消除非生理性和/或非解剖学的 MTU 路径。此外，本研究中实现了包络面，但将这种优化方法扩展到由途径点（via points）构成的模型也是可行的，目前正在测试中。

本研究存在一些需要考虑的局限性。首先，尽管部分肌腱复合体（MTU，Muscle-Tendon Unit）跨越两个关节（例如股直肌），但我们仅考虑了针对单一自由度（DOF，Degree of Freedom）的 MTU 运动学，即胫股关节（TFJ，Tibiofemoral Joint）屈曲/伸展。对于所考虑的 DOF，MTU 运动学可能与文献中的模式高度吻合，然而，当涉及多个 DOF 时，MTU 运动学可能会出现误差。尽管我们承认这是一个局限性，但该框架是同类中的首创，并可在未来的工作中扩展到多个 DOF。其次，本研究所使用的个性化关节机制所产生的关节运动学尚未通过动态医学成像进行直接验证。这些个性化关节机制的调优确保了与尸体文献一致的生理学和解剖学上合理的解（Appendix 4）。在此，我们将生理学和解剖学上合理的运动学定义为符合关节几何形态（即无骨与骨穿透）和组织特性（即韧带长度变化）的运动学。然而，并不能保证所得运动学结果代表受试者特异性的 TFJ 和髌股关节（PFJ，Patellofemoral Joint）运动学。支持该方法的是，先前的研究表明在 8 名儿童的无负荷体位下具有一致性（Barzan et al. 2019），而另一项采用类似优化方法针对单一受试者的研究表明，其关节运动学与通过透视成像测量的结果相当（Nardini et al. 2020）。

在当前的实现中，我们的方法不能保证每个 MTU 的建模路径与医学成像测量的路径相匹配。尽管每位受试者都接受了下肢 MRI 扫描，但这些数据不足以分割单独的 MTU。因此，我们无法评判优化后的 MTU 路径是否与 MRI 测量的路径一致。然而，我们方法的初衷是利用 MAP-Client 中形变后的骨骼和关节来确保生理学和解剖学上合理的 MTU 运动学，而不是对每个 MTU 进行完全个性化的表征。事实上，如果能够从单一或多种姿势的 MRI 中获取分割肌肉，或从统计形状建模（statistical shape modelling，目前正在开发中）中获取重建结果，则可以实现更高程度的个性化，其中它们的三维质心可用作附加的优化标准，以强制追踪个性化的 MTU 质心。因此，当前方法中缺乏三维 MTU 表征，也缺乏通过其他方式对该方法的验证。

通过检验将优化后的 MTU（肌腱单位，muscle-tendon unit）路径模型嵌入神经肌肉骨骼模型后其预测结果（例如关节力矩和/或关节接触力）的改善情况，可以实现间接验证。尽管我们的团队 (Gerus et al. 2013; Davico et al. 2020b) 以及其他研究人员 (Serrancolí et al. 2020) 曾采用过这种方法，但这超出了本研究的范围。我们当前的主要目标是建立一套评价准则和一个自动化框架，以构建具有不同个性化程度的肌肉骨骼模型，并检验它们生成在解剖学和生理学上合理的 MTU 路径和运动学特征（MTU 长度和力臂）的能力。有趣的是，先前的研究 (Serrancolí et al. 2020) 提出了一种框架，通过优化神经肌肉骨骼模型在预测关节接触力方面的表现来直接校准 MTU 力臂。然而，该方法导致了 MTU 力臂与 MTU 长度之间在力学机制上的脱节（即在数学上，力臂是 MTU 长度对关节角度的微分），而 MTU 长度是 MTU 产力模型的输入。此外，由于肌肉骨骼模型与 MTU 模型及其设计变量之间存在相互依赖性，通过校准 MTU 力臂来获得更优的神经肌肉骨骼模型预测（例如关节接触力或关节力矩），并不能确保肌肉骨骼模型的有效性。相反，本研究所采用的方法通过优化模型中的 MTU 路径，保持了模型与所得 MTU 运动学之间的机制有效性，以及在解剖学和生理学上的合理性。尽管如此，未来的研究仍应检验采用这种方法是否能改善整个神经肌肉骨骼模型的预测表现。

本研究所开发的框架对个性化肌肉骨骼建模领域做出了重要贡献。这是首个利用 MAP-Client 自动开发和调整个性化模型，并评估其在后续肌肉骨骼仿真中适用性的研究。我们对 MAP-Client 进行了若干改进和补充，最显著的是无需明确采集单一或多个关节角度下的 MRI（磁共振成像），即可定义 MTU 中间路径。以往需要手动执行的若干流程在该框架内实现了自动化，从而降低了生成这些个性化模型时的主观性。这些流程包括手动解剖标志点定义、关节位置确定，以及 MTU 起点、止点和肌肉中间路径的定义。未来的工作将重点改进

MTU（肌肉-肌腱单元）中间路径的定义与优化。具体而言，将对优化表述进行分析，以最小化各项及其潜在的相互竞争。此外，还将开展将这些优化方法应用于路径点（via points）而非包络面（wrapping surfaces），并扩展所考虑的肌肉和关节数量的研究。

## 声明

## 资助

作者谨在此鸣谢 Griffith University 博士生奖学金的资助。

## 利益冲突

作者声明与本研究所呈现的工作不存在利益冲突。

## 代码、数据与材料可用性

现有的 MAP-Client 可在此处免费获取：https://mapclient.readthedocs.io/en/latest/ ，更多信息见 https://simtk.org/projects/map 。请注意，该框架目前仅适用于 Python 2，原开发者目前正在制作适用于 Python 3 的更新版本。有关该更新状态的最新信息将在上述 SimTK 链接中提供。

作为本研究一部分所开发的框架可在合理要求下向通讯作者索取。作为本研究一部分所生成的模型可在合理要求下索取。

## 参考文献

Ackland DC, Roshan-Zamir S, Richardson M, Pandy MG (2011) Muscle and joint-contact loading at the glenohumeral joint after reverse total shoulder arthroplasty. J Orthop Res 29:1850 1858

Andersen MS (2018) How sensitive are predicted muscle and knee contact forces to normalization factors and polynomial order in the muscle recruitment criterion formulation? Int Biomech 5:88 103

Arnold EM, Ward SR, Lieber RL, Delp SL (2010) A Model of the Lower Limb of Analysis of Human Movement. Ann Biomed Eng 38:269 279

Bahl JS, Zhang J, Killen BA, et al (2019) Statistical shape modelling versus linear scaling: effects on predictions of hip joint centre location and muscle moment arms in people with hip osteoarthritis. J Biomech 85:164 172

Bakke D, Besier T (2020) Shape model constrained scaling improves repeatability of gait data. J Biomech 107:109838. https://doi.org/https://doi.org/10.1016/j.jbiomech.2020.109838

Barzan M, Modenese L, Carty CP, et al (2019) Development and validation of subject-specific pediatric multibody knee kinematic models with ligamentous constraints. J Biomech 93:194 203. https://doi.org/https://doi.org/10.1016/j.jbiomech.2019.07.001

Brito da Luz S, Modenese L, Sancisi N, et al (2017) Feasibility of using MRIs to create subjectspecific parallel-mechanism joint models. J Biomech 53:45 55

Buford WL, Marty Ivey Jr F, Malone D, et al (1997) Muscle Balanace at the Knee - Moment Arm for the Normal and the ACL-Minus Knee. IEEE Trans Biomed Eng 5:367 379

Catelli DS, Wesseling M, Jonkers I, Lamontagne M (2019) A musculoskeletal model customized for squatting task. Comput Methods Bioemchanics Biomed Eng 22:21 24

Cleather DJ, Bull AM (2011) An optimization-based simultaneous approach to the determination of muscular, ligamentous, and joint contact forces provides insight into musculoligamentous interaction. Ann Biomed Eng 39:1925 1934

Davico G, Pizzolato C, Killen BA, et al (2020a) Best methods and data to reconstruct paediatric lower limb bones for musculoskeletal modelling. Biomech Model Mechanobiol 19:1225 1238. https://doi.org/10.1007/s10237-019-01245-y

Davico G, Pizzolato C, Lloyd DG, et al (2020b) Increasing level of neuromusculoskeletal model personalisation to investigate joint contact forces in cerebral palsy: A twin case study. Clin Biomech 72:141 149. https://doi.org/10.1016/j.clinbiomech.2019.12.011

Delp SL, Anderson FC, Arnold AS, et al (2007) OpenSim: open-source software to create and analyze dynamic simulations of movement. IEEE Trans Biomed Enginerring 54:1940 1950

Demers MS, Pal S, Delp SL (2014) Changes in tibiofemoral forces due to variations in muscle activity during walking. J Orthop Res 32:769 776. https://doi.org/10.1002/jor.22601

Draganich LF, Andriacchi TP, Andersson GB (1987) Interaction between intrinsic knee mechanics and the knee extensor mechanism. J Orthop Res 5:539 547

Dzialo CM, Pedersen PH, Simonsen CW, et al (2018) Development and validation of a subject-

specific moving-axis tibiofemoral joint model using MRI and EOS imaging during a quasi-static lunge. J Biomech 72:71 80

Fick AE (1879) Uber Zweigelenkige Muskeln. Arch Anat Physiol (Anat Abt) 201 239

Garner BA, Pandy MG (2000) The Obstacle Set Method for Representing Muslce Paths in Musculoskeletal Models. Comput Methods Bioemchanics Biomed Eng 3:1 30

Gerus P, Sartori M, Besier TF, et al (2013) Subject-specific knee joint geometry improves predictions of medial tibiofemoral contact forces. J Biomech 9:2 9. https://doi.org/10.1016/j.jbiomech.2013.09.005i

Guess TM, Stylianou AP, Kia M (2014) Concurrent prediction of muscle and tibiofemoral contact forces during treadmill gait. J Biomech Eng 136:21032

Hammer M, Gunther M, Haeufle DFB, Schmitt S (2019) Tailoring anatomical muscle paths: a sheath-like solution for muscle routing in musculoskeletal computer models. Math Biosci 311:68 81

Kainz H, Carty CP, Maine S, et al (2017) Effects of hip joint centre mislocation on gait kinematics of children with cerebral palsy calculated using patient-specific direct and inverse kinematic models. Gait Posture 57:154 160. https://doi.org/10.1016/j.gaitpost.2017.06.002

Konrath JM, Saxby DJ, Killen BA, et al (2017) Muscle contributions to medial tibiofemoral compartment contact loading following ACL reconstruction using semitendinosus and gracilis tendon grafts. PLoS One 12:. https://doi.org/https://doi.org/10.1371/journal.pone.0176016

Lai AKM, Arnold AS, Wakeling JM (2017) Why are Antagonist Muscles Co-activated in My Simulation? A Musculoskeletal Model for Analysing Human Locomotor Tasks. Ann Biomed Eng 45:2762 2774. https://doi.org/10.1007/s10439-017-1920-7

Lerner ZF, Demers MS, Delp SL, Browning RC (2015) How tibiofemoral alignment and contact locations affect predictions of medial and lateral tibiofemoral contact forces. J Biomech 48:644 650

Modenese L, Kohout J (2020) Automated Generation of Three-Dimensional Complex Muscle Geometries for Use in Personalised Musculoskeletal Models. Ann Biomed Eng 48:1793 1804. https://doi.org/10.1007/s10439-020-02490-4

Modenese L, Montefiori E, Wang A, et al (2018) Investigation of the dependence of joint contact forces on musculotendon parameters using a codified workflow for image-based modelling. J Biomech 73:108 118. https://doi.org/10.1016/j.jbiomech.2018.03.039

Modenese L, Renault J-B (2020) Automatic Generation of Personalised Models of the Lower Limb from Three-Dimensional Bone Geometries: A Validation Study. bioRxiv 2020.06.23.162727. https://doi.org/10.1101/2020.06.23.162727

Nardini F, Belvedere C, Sancisi N, et al (2020) An Anatomical-Based Subject-Specific Model of In-Vivo Knee Joint 3D Kinematics From Medical Imaging. Appl. Sci. 10

Navacchia A, Kefala V, Shelburne KB (2017) Dependence of Muscle Moment Arms on In Vivo Three-Dimensional Kinematics of the Knee. Ann Biomed Eng 45:789 798

Nolte D, Ko S-T, Bull AMJ, Kedgley AE (2020) Reconstruction of the lower limb bones from

digitised anatomical landmarks using statistical shape modelling. Gait Posture 77:269 275. https://doi.org/https://doi.org/10.1016/j.gaitpost.2020.02.010

Nolte D, Tsang CK, Zhang KY, et al (2016) Non-linear scaling of a musculoskeletal model of the lower limb using statistical shape models. J Biomech 49:3576 3581. https://doi.org/10.1016/j.jbiomech.2016.09.005

Novacheck TF (1998) The biomechanics of running. Gait Posture 7:77 95. https://doi.org/10.1016/s0966-6362(97)00038-6

Pal S, Langenderfer JE, Stowe JQ, et al (2007) Probabilistic modelling of knee muscle moment arms: effects of methods, origin-insertion, and kinematic variability. Ann Biomed Eng 35:1632 1642

Rajagopal A, Dembia CL, Demers MS, et al (2016) Full-Body Musculoskeletal Model for Muscle-Driven Simulation of Human Gait. IEEE Trans Biomed Eng 63:2068 2079

Reinbolt JA, Haftka RT, Chmielewski TL, Fregly BJ (2007) Are patient-specific joint and inertial parameters necessary for accurate inverse dynamics analyses of gait? IEEE Trans Biomed Eng 54:782 793. https://doi.org/10.1109/TBME.2006.889187

Sancisi N, Parenti-Castelli V (2011a) A novel 3D parallel mechanism for the passive motion simulation of the patella-femur-tibia complex. Meccanica 46:207 220. https://doi.org/10.1007/s11012-010-9405-x

Sancisi N, Parenti-Castelli V (2011b) A New Kinematic Model of the Passive Motion of the Knee Inclusive of the Patella. J Mech Robot 3:. https://doi.org/10.1115/1.4004890

Saxby DJ, Killen BA, Pizzolato C, et al (2020) Machine learning methods to support personalized neuromusculoskeletal modelling. Biomech Model Mechanobiol. https://doi.org/10.1007/s10237-020-01367-8

Saxby DJ, Modenese L, Bryant AL, et al (2016) Tibiofemoral Contact Forces During Walking, Running and Sidestepping. Gait Posture 49:78 85. https://doi.org/10.1016/j.gaitpost.2016.06.014

Scheys L, Loeckx D, Spaepen A, et al (2009) Atlas-based non-rigid image registration to automatically define line-of-action muscle models: a validation study. J Biomech 42:565 572. https://doi.org/10.1016/j.jbiomech.2008.12.014

Serrancolí G, Kinney AL, Fregly BJ (2020) Influence of musculoskeletal model parameter values on prediction of accurate knee contact forces during walking. Med Eng Phys 85:35 47. https://doi.org/https://doi.org/10.1016/j.medengphy.2020.09.004

Seth A, Hicks JL, Uchida TK, et al (2018) OpenSim: Simulating musculoskeletal dynamics and neuromuscular control to study human and animal movement. PLoS Comput Biol 14:e1006223

Smale KB, Conconi M, Sancisi N, et al (2019) Effect of implementing magnetic resonance imaging for patient-specific OpenSim models on lower-body kinematics and knee ligament lengths. J Biomech 83:9 15

Spoor CW, van Leeuwen JL (1992) Knee Muscle Moment Arms From MRI and From Tendon Travel. J Biomech 25:201 206

Suwarganda EK, Diamond LE, Lloyd DG, et al (2019) Minimal medical imaging can

accurately reconstruct geometric bone models for musculoskeletal models. PLoS One. https://doi.org/https://doi.org/10.1371/journal.pone.0205628

Valente G, Crimi G, Vanella N, et al (2017) nmsBuilder: Freeware to create subject-specific musculoskeletal models for OpenSim. Comput Methods Programs Biomed 152:85 92

Visser JJ, Hoogkamer JE, Bobbert MF, Huijing PA (1990) Length and moment arm of human leg muscles as a function of knee and hip-joint angles. Eur J Appl Physiol Occup Physiol 61:453 460

Wesseling M, De Groote F, Meyer C, et al (2016) Subject-specifc musculoskeletal modelling in patients before and after total hip arthroplasty. Comput Methods Biomech Biomed Engin 19:1683 1691

Wilson NA, Sheehan FT (2009) Dynamic in vivo 3-dimensional moment arms of the individual quadriceps components. J Biomech 42:1891 1897

Winby CR, Lloyd DG, Besier TF, Kirk TB (2009) Muscle and external load contribution to knee joint contact loads during normal gait. J Biomech 42:2294 2300

Zhang J, Fernandez J, Hislop-Jambrish J, Besier TF (2016) Lower limb estimation from sparse landmarks using an articulated shape model. J Biomech 49:3875 3881

Zhang J, Sorby H, Clement J, et al (2014) The MAP Client: User Friendly Musculoskeletal Modelling Workflows. In: Bello F, Cotin S (eds) Interntional Symposium on Biomedical Simulation. Springer, Strasbourg, France, pp 182 192

![](images/31d291cc1a1bedeaf4d6d8b22daf0c75145e6d08cc82b3dab124e3eb160b1211.jpg)

![](images/a8e165b6e75ada157b2b5a0e4727eb0a9e39ea750b8e3dda65a3c9f5066bc53c.jpg)

![](images/e2d89ed82bb78242f658dc2295487cd2df158c5c770d5ebcbe7985eaa3ce7f04.jpg)

![](images/bd0ba3b31da0756bfc940039325e8b02618f9a3b9fec28e39180eca732a8e28f.jpg)

![](images/429d93dc4078d6968ef9026d2c30979d19c572462996d69dae4a41ff66db51cf.jpg)

![](images/97f5c88e934335bd76f6e3df76d53bef90a5795f105ceae57a2e963116f8254f.jpg)

![](images/a6db10a7f3d4fd63b5deba0d589e18f40f3814f1b104f522f4135f03fd5a0b30.jpg)

![](images/65ce3000b7e3087e30e1e35592c3602a2bb6b93c41b4150078144955bebc2032.jpg)

![](images/1e74e63d2c19f2b2de41fe0acea7307bfa8bac3ccb8ba2c6feb9c0e4eec4f37b.jpg)

![](images/c8e3f1f80ed34007efc7ef3d92007bbfa41d26e1ae186d449caf6f6e630f3475.jpg)

Table 1: 研究受试者的人口统计学数据。

<table><tr><td>受试者</td><td>性别</td><td>测试肢</td><td>年龄（岁）</td><td>身高（cm）</td><td>体重（kg）</td></tr><tr><td>M01</td><td>M</td><td>R</td><td>24</td><td>182.0</td><td>82</td></tr><tr><td>M02</td><td>F</td><td>R</td><td>22</td><td>172.0</td><td>63</td></tr><tr><td>M03</td><td>M</td><td>R</td><td>23</td><td>180.0</td><td>88</td></tr><tr><td>M07</td><td>M</td><td>L</td><td>32</td><td>185.0</td><td>89</td></tr><tr><td>M09</td><td>M</td><td>L</td><td>31</td><td>161.0</td><td>45</td></tr><tr><td>M11</td><td>F</td><td>L</td><td>21</td><td>160.5</td><td>55</td></tr><tr><td>(n 或 均值±标准差)</td><td>4M &amp; 2F</td><td>3L &amp; 3R</td><td>25.5±4.8</td><td>173.4±10.7</td><td>70.3±18.6</td></tr></table>

M-男性，F-女性；L-左，R-右；sd-标准差。

Table 2: 作为本研究一部分所构建的四种模型的说明。  
Model 1 MAP-Client 生成的模型，包含个性化的：骨骼几何形态、关节位置、MTU 起止点以及未优化的包络面

<table><tr><td>Model 2</td><td>Model 1，但缩放后的通用 TFJ 和 PFJ 已被替换为个性化关节机制（Brito da Luz et al. 2017; Barzan et al. 2019）</td></tr><tr><td>Model 3</td><td>经 MTU 包络面优化后的 Model 1</td></tr><tr><td>Model 4</td><td>经 MTU 包络面优化后的 Model 2</td></tr></table>

MTU 肌腱单元（muscle-tendon unit）。

Table 3: 模型构建中各任务所需的时间。

<table><tr><td>任务</td><td>时间（小时）</td></tr><tr><td>MRI 分割（骨盆和双侧下肢）</td><td>3</td></tr><tr><td>MAP-Client 骨骼变形（骨盆和双侧下肢）</td><td>3</td></tr><tr><td>TFJ MRI 分割（骨骼、软骨和韧带）</td><td>3</td></tr><tr><td>MAP-Client OpenSim 创建</td><td>0.5</td></tr><tr><td>TFJ 和 PFJ 机制优化</td><td>4</td></tr><tr><td>包络面优化</td><td>2*</td></tr><tr><td>总计</td><td>15.5 小时</td></tr></table>

\*优化在高性能计算机集群上并行运行。所用集群的配置为 2-4GB RAM  
MRI 磁共振成像（magnetic resonance imaging）；TFJ 胫股关节（tibiofemoral joint）；PFJ 髌股关节（patellofemoral joint）。

三种结果类型中，单元格被留空……表示改善，红色表示恶化，黄色表示无变化。在无统计学[显著性]的情况下……样本量为 144（每组 8 个 × 3 组 × 6 名受试者）。统计学显著性差异为……为 (+) 改善、(-) 恶化或 (±) 无变化。对于每个 MTU 组，样本量为 4……面（Model 3）。展示的是每个 MTU 组以及所有组综合后的实例数量……带有拟合包络面（fitted wrapping-surfaces）的 MAP-Client 模型（Model 1）与……之间的比较

<table><tr><td colspan="3">极性惩罚</td><td colspan="3">肌肉穿透</td><td colspan="3">长度平滑度</td><td colspan="3">力臂平滑度</td><td colspan="3">力臂梯度</td></tr><tr><td>+</td><td>-</td><td>±</td><td>+</td><td>-</td><td>±</td><td>+</td><td>-</td><td>±</td><td>+</td><td>-</td><td>±</td><td>+</td><td>-</td><td>±</td></tr><tr><td colspan="15">所有 MTU</td></tr><tr><td>23,16%</td><td>5,3.5%</td><td>116,80.6%</td><td>48,33.3%</td><td>3,2.1%</td><td>93,64.6%</td><td>28,19.4%</td><td>9,6.3%</td><td>107,74.3%</td><td>62,43.1%</td><td>37,25.7%</td><td>45,31.3%</td><td>141,97.9%</td><td>3,2.1%</td><td>0,0%</td></tr><tr><td colspan="15">股四头肌 MTU 组</td></tr><tr><td>0,0%</td><td>0,0%</td><td>48,100%</td><td>5,10.4%</td><td>2,4.2%</td><td>41,85.4%</td><td>13,27.1%</td><td>0,0%</td><td>35,72.9%</td><td>14,29.2%</td><td>14,29.2%</td><td>20,41.7%</td><td>48,100%</td><td>0,0%</td><td>0,0%</td></tr><tr><td colspan="15">腘绳肌 MTU 组</td></tr><tr><td>9,18.8%</td><td>4,8.3%</td><td>35,72.9%</td><td>16,33.3%</td><td>1,2.1%</td><td>31,64.6%</td><td>9,18.8%</td><td>4,8.3%</td><td>35,72.9%</td><td>27,56.3%</td><td>13,27.1%</td><td>8,16.7%</td><td>46,95.8%</td><td>2,4.2%</td><td>0,0%</td></tr><tr><td colspan="15">其他 MTU 组</td></tr><tr><td>14,29.2%</td><td>1,2.1%</td><td>33,68.8%</td><td>27,56.3%</td><td>0,0%</td><td>21,43.8%</td><td>6,12.5%</td><td>5,10.4%</td><td>37,77.1%</td><td>21,43.8%</td><td>10,20.8%</td><td>17,35.4%</td><td>47,97.9%</td><td>1,2.1%</td><td>0,0%</td></tr><tr><td colspan="15">其中，+ 表示从模型 1 到模型 3 改善，- 表示从模型 1 到模型 3 恶化，± 表示模型 1 与模型 3 之间无变化。</td></tr></table>

三种结果类型之间均无统计学显著差异的情况……差异通过彩色单元格表示，其中绿色表示改善，红色表示……为 48（8 个 MTU × 6 名受试者）。对于所有合并的 MTU，样本量为 144（每组 8 个 × ……合并的 MTU，其中 MTU 质量评估指标为 (+) 改善、(-) 变差或……个性化包络面与个性化关节机制（Model 4）。展示的是……具有拟合包络面和个性化关节机制的 MAP-Client 模型之间的比较……

<table><tr><td colspan="3">极性惩罚</td><td colspan="3">肌肉穿透</td><td colspan="3">长度平滑度</td><td colspan="3">力臂平滑度</td><td colspan="3">力臂梯度</td></tr><tr><td>+</td><td>-</td><td>±</td><td>+</td><td>-</td><td>±</td><td>+</td><td>-</td><td>±</td><td>+</td><td>-</td><td>±</td><td>+</td><td>-</td><td>±</td></tr><tr><td colspan="15">所有 MTU</td></tr><tr><td>25, 17.4%</td><td>4, 2.8%</td><td>115, 79.9%</td><td>45, 31.3%</td><td>10, 6.9%</td><td>89, 61.8%</td><td>26, 18.1%</td><td>18, 12.5%</td><td>100, 69.4%</td><td>71, 49.3%</td><td>29, 20.1%</td><td>44, 30.6%</td><td>135, 93.8%</td><td>9, 6.3%</td><td>0, 0%</td></tr><tr><td colspan="15">股四头肌 MTU 组</td></tr><tr><td>1, 2.1%</td><td>1, 2.1%</td><td>46, 95.8%</td><td>9, 18.8%</td><td>4, 8.3%</td><td>35, 72.9%</td><td>6, 12.5%</td><td>7, 14.6%</td><td>35, 72.9%</td><td>21, 43.8%</td><td>9, 18.8%</td><td>18, 37.5%</td><td>46, 95.8%</td><td>2, 4.2%</td><td>0, 0%</td></tr><tr><td colspan="15">腘绳肌 MTU 组</td></tr><tr><td>12, 25%</td><td>2, 4.2%</td><td>34, 70.8%</td><td>15, 31.3%</td><td>1, 2.1%</td><td>32, 66.7%</td><td>7, 14.6%</td><td>7, 14.6%</td><td>34, 70.8%</td><td>21, 43.8%</td><td>11, 22.9%</td><td>16, 33.3%</td><td>43, 89.6%</td><td>5, 10.4%</td><td>0, 0%</td></tr><tr><td colspan="15">其他 MTU 组</td></tr><tr><td>12, 25%</td><td>1, 2.1%</td><td>35, 72.9%</td><td>21, 43.8%</td><td>5, 10.4%</td><td>22, 45.8%</td><td>13, 27.1%</td><td>4, 8.3%</td><td>31, 64.6%</td><td>29, 60.4%</td><td>9, 18.8%</td><td>10, 20.8%</td><td>46, 95.8%</td><td>2, 4.2%</td><td>0, 0%</td></tr><tr><td colspan="15">其中，+ 表示从模型 2 到模型 4 改善，- 表示从模型 2 到模型 4 恶化，± 表示模型 2 与模型 4 之间无变化，</td></tr></table>

[三种结果类型之]一，单元格留[空。绿色表]示改善，红色表示恶化，黄色表示无变化。在无统计学[差异的情况下……]
[样]本量为 144（每组 8 个 × 3 组 × 6 名受试者）。具有统计学显著性的差异[以……表示]
分别为（+）改善、（-）恶化或（±）无变化。对于每个 MTU 组，样本量为 48（8[……]
[个]性化关节机制（Model 4）。显示的是每个 MTU [……]的实例数量，
[具有优化包绕面的 MAP-Client 模型（Model 3）与具有……] MAP-Client [……之间的比较]

<table><tr><td colspan="3">极性惩罚</td><td colspan="3">肌肉穿透</td><td colspan="3">长度平滑度</td><td colspan="3">力臂平滑度</td><td colspan="3">力臂梯度</td></tr><tr><td>+</td><td>-</td><td>±</td><td>+</td><td>-</td><td>±</td><td>+</td><td>-</td><td>±</td><td>+</td><td>-</td><td>±</td><td>+</td><td>-</td><td>±</td></tr><tr><td colspan="15">所有 MTU</td></tr><tr><td>27,18.8%</td><td>13,9%</td><td>104,72.7%</td><td>3,2.1%</td><td>25,17.4%</td><td>116,80.6%</td><td>20,13.9%</td><td>27,18.8%</td><td>97,67.4%</td><td>43,29.9%</td><td>48,33.3%</td><td>53,36.8%</td><td>66,45.8%</td><td>78,54.2%</td><td>0,0%</td></tr><tr><td colspan="15">股四头肌 MTU 组</td></tr><tr><td>0,0%</td><td>1,2.1%</td><td>47,97.9%</td><td>0,0%</td><td>11,22.9%</td><td>37,77.1%</td><td>3,6.3%</td><td>16,33.3%</td><td>29,60.4%</td><td>14,29.2%</td><td>16,33.3%</td><td>18,37.5%</td><td>24,50%</td><td>24,50%</td><td>0,0%</td></tr><tr><td colspan="15">腘绳肌 MTU 组</td></tr><tr><td>6,12.5%</td><td>4,8.3%</td><td>38,79.2%</td><td>2,4.2%</td><td>4,8.3%</td><td>42,87.5%</td><td>6,12.5%</td><td>6,12.5%</td><td>36,75%</td><td>15,31.3%</td><td>15,31.3%</td><td>18,37.5%</td><td>19,39.6%</td><td>29,60.4%</td><td>0,0%</td></tr><tr><td colspan="15">其他 MTU 组</td></tr><tr><td>21,43.8%</td><td>8,16.7%</td><td>19,39.6%</td><td>1,2.1%</td><td>10,20.8%</td><td>37,77.1%</td><td>11,22.9%</td><td>5,10.4%</td><td>32,66.7%</td><td>14,29.2%</td><td>17,35.4%</td><td>17,35.4%</td><td>23,47.9%</td><td>25,52.1%</td><td>0,0%</td></tr><tr><td colspan="15">其中，+ 表示从模型 3 到模型 4 有所改善，- 表示从模型 3 到模型 4 有所恶化，± 表示模型 3 与模型 4 之间无变化。</td></tr></table>

Fig 1: 受试者 M02 的右腿缝匠肌 MTU 运动学曲线，其中左图为 MTU 长度，右图为 TFJ 屈曲力臂。黑线代表各向同性线性缩放的 gait2392 模型，绿线代表尸体文献综合数据，红线代表模型 1，蓝线代表模型 3。MTU：肌腱单位（muscle-tendon unit）；TFJ：胫股关节（tibiofemoral joint）。

Fig 2: 受试者 M03 的右腿半膜肌 MTU 运动学曲线，其中左图为 MTU 长度，右图为 TFJ 屈曲力臂。黑线代表各向同性线性缩放的 gait2392 模型，绿线代表尸体文献综合数据，红线代表模型 1，蓝线代表模型 3。MTU：肌腱单位；TFJ：胫股关节。

Fig 3: 受试者 M01 的左腿缝匠肌 MTU 运动学曲线，其中左图为 MTU 长度，右图为 TFJ 屈曲力臂。黑线代表各向同性线性缩放的 gait2392 模型，绿线代表尸体文献综合数据，红线代表模型 3，蓝线代表模型 4。MTU：肌腱单位；TFJ：胫股关节。

Fig 4：受试者 M01 右腿股直肌 MTU 运动学曲线，其中左图为 MTU 长度，右图为 TFJ 屈曲力臂。黑线代表各向同性线性缩放的 gait2392 模型，绿线代表综合尸体文献数据，红线代表 Model 3，蓝线代表 Model 4。MTU：肌腱单位（muscle-tendon unit）；TFJ：胫股关节（tibiofemoral joint）。

Fig 5：受试者 M09 在 Model 1（A、C）中半腱肌（A、B）和股外侧肌（C、D）的 MTU 骨穿透示例，在模型优化后即 Model 3（B、D）中该穿透被消除。

## 1 附录 1：常见的非生理性 MTU 运动学和非物理性 MTU 路径

在大多数通用 OpenSim 模型中，MTU 中间路径主要通过固定点和条件（“via”）点的组合来定义。使用经由点（via points）允许用户根据模型的行为来定义条件性 MTU 路径（例如，当模型达到特定构型时被激活）。尽管这种方法在通用模型中可能有效，但通过线性或偏度缩放将这些经由点转移到个性化模型中并非易事。先前的研究人员（Scheys et al. 2009；Modenese et al. 2018）以及 MAP-Client 开发人员已经使用非刚性形变方法将经由点拟合到个性化骨骼几何形状中。尽管这些方法可以在 MAP-Client 内直接实现，但它们常常会引入非物理路径（Fig S1）和非生理性 MTU 运动学（Fig S2）。非物理路径是指包含 $90^{\mathrm{o}}$ 转折并穿透骨骼的路径。非生理性 MTU 运动学是指 MTU 长度不平滑和/或不符合先前通过尸体实验发表的文献中的模式（表现为力臂大小随关节角度的变化）。

![](images/3645a4f3c024f41021259721fc8c399460f7b48ede35854d72c07a11b2d7b105.jpg)

B  
![](images/5bdfa1252a63523b9548a592dc99076f9f44f4439c6e7411d3278e99047d6c3c.jpg)

![](images/bb45c3d8e151c98cc93c70ce727f703ba5b2d65847bdd0d6249aba0b2a88d55c.jpg)

![](images/eafab10b7985a081b676428b5bc4dcafcf34a030141911ebdf706041c891a456.jpg)  
E

![](images/9799f790f56901e92538df8f5a9ca33fe4552caee5ffb45613e6ab0d3ae96212.jpg)

![](images/24764ab060683c7d8436665fd1fbc01727b74ceaad12a8e205a28800bccc944f.jpg)  
Fig S1：在使用形变经由点的标准 MAP-Client 生成模型中非生理性 MTU 形态（A、B、C）和 MTU 骨穿透（C、D、E、F）的示例。MTU：肌腱单位（muscle-tendon unit）。

![](images/a00e8f705e06951aa217d45c434ae754bf6f37033e431cca3c3039b556f5db09.jpg)  
Fig S2: 标准 MAP-Client 生成模型（红色）、通用 gait2392 模型（黑色）以及尸体文献（绿色）中股直肌、内侧腓肠肌和股外侧肌的 MTU 长度与力臂，其中不连续处用黑框突出显示。MTU：肌肉-肌腱单元（muscle-tendon unit）。

## 附录 2：详细优化与评估标准

优化方法使用开源 Python 软件包编写，并部署在 Griffith University 高性能计算集群（High-Performance Computing cluster，https://confers.griffith.edu.au/display/GHCD/Gowonda+HPC）上，从而实现了大规模并行化。该框架内使用的优化方法被称为粒子群优化（particle swarm optimisation，PSO）（pyswarm: A Python package for particle swarm optimization (PSO) with constraint support, Abraham D. Lee, https://pythonhosted.org/pyswarm/）。PSO 使用不同的“粒子”同时搜索解空间的不同区域。在每次迭代期间，计算 MTU 长度和力臂，并根据各种目标标准和惩罚函数进行测试。

使用 OpenSim 计算的 MTU 力臂对 MTU 长度的微小变化高度敏感。为了克服这一问题并提高计算速度，实施了先前发表的用于精确估计 MTU 力臂的方法。该实现使用三次 B 样条拟合 TFJ 屈曲/伸展运动范围内的 MTU 长度（Sartori et al. 2012）。然后，将 MTU 力臂计算为这些样条关于关节自由度的偏导数，即长度变化除以关节角度变化。为了确保该过程不会人为地增加计算出的 MTU 长度平滑度（下文详述），计算了 OpenSim 计算出的 MTU 长度与三次 B 样条拟合的 MTU 长度之间的归一化误差（splineNormErr），并在优化框架内将其最小化。

MTU（肌肉肌腱单元）运动学模式表示为相对于关节角度的梯度（MTU 力臂/长度的变化除以关节角度的变化），因此模式相似性通过建模数据（MAP-Client 个性化模型）与目标数据（尸体文献数据）梯度之间的归一化误差（maGradErr）进行评估。建立了进一步的目标函数来测量和控制曲线平滑度，以确保 MTU 运动学的平滑与连续。平滑度测量基于三个假设。首先，MTU 运动学曲线主要是单相的，没有明显的峰值或波谷，这在关节于生理活动范围内运动时是成立的。其次，曲线的梯度对于 MTU 长度而言是恒定的，对于 MTU 力臂而言则是缓慢变化的。第三也是最后一个假设是，如果 MTU 运动学曲线是一条完全平直的直线，则该直线的二阶导数将为零。因此，如果 MTU 运动学的梯度仅发生轻微变化，其二阶导数将近似为零。因此，运动学平滑度被定义为使求导后曲线的极差、均值和最大值降至预设阈值以下所需的建模曲线（MAP-Client 个性化模型）求导阶数。随后，将测试曲线的平滑度测量值相对于目标数据的平滑度测量值进行归一化。需要说明的是，仅使用了通用 OpenSim 模型数据对平滑度测量值进行归一化，其中采用了两个通用 OpenSim 模型的平均平滑度测量值（见下文）。如上所述，目标准则在优化中使用了多个目标。

目标数据取自多个来源，但可分为两个不同的类别：模型数据和文献数据。模型数据获取自两个通用 OpenSim 模型：MAP-Client 模型所基于的 gait2392 模型（Delp et al. 2007）以及较新的 Fullbody 模型（Rajagopal et al. 2016）。与模型数据不同，文献数据取自在尸体上开展的广泛不同研究（Fick 1879; Draganich et al. 1987; Visser et al. 1990; Spoor and van Leeuwen 1992; Buford et al. 1997; Pal et al. 2007; Wilson and Sheehan 2009; Arnold et al. 2010; Navacchia et al. 2017）。请注意，上述并非所有研究的选定原始尸体数据都能直接获取。然而，我们使用了在其他文献中复现并发表的尸体数据（Arnold et al. 2010; Rajagopal et al. 2016）。针对每个 MTU 对这些数值进行了合并，并计算了平均值和标准差。

采用优化准则和惩罚函数在数学上检测 MTU 运动学和路径中常见的各种错误。每个 MTU 的力臂都需要在关节 DOF（例如屈曲或伸展）上具有与目标数据一致的力学作用（Fig S3）。这被称为“力臂极性”（polarity of the moment arm），并定义了惩罚以确保其符合生理特性（力臂极性惩罚）。在每个 TFJ 屈曲角度下，比较测试模型（即 MAP-Client 个性化模型）与通用 gait2392 模型的极性。如果极性相同，则不施加惩罚；否则执行二次检验。具体而言，检查相邻的 20 个角度（即 ±10º）。如果测试模型在此范围内的极性与目标数据相符，则该差异不再被视为极性错误。当不满足此条件时，会对最终加权值施加惩罚。

![](images/3d24e3f01b2ff3447e2c5520f5d8eae851b2c92cbc1de861dbe673dc432276a3.jpg)  
Fig S3: MAP-Client 生成的模型（红色）与通用模型（黑色）及文献（绿色）数据相比，MTU 力臂极性错误的示例。

取决于 MTU 与相关包络圆柱体相交的位置，它可能会完全围绕其缠绕，即在继续延伸至止点之前绕完一个完整的圆周回路（Fig S4A）。为了避免这些非物理的 MTU 路径，对中立位时的 MTU 路径点进行了查询。这些点代表起点、止点和途经点，以及 MTU 开始和结束缠绕的点（Fig S4B）。基于每个 MTU 路径点相对于前一个路径点均处于更下侧的假设（在中立位时确实如此），测试了每个路径点的上/下坐标。如果某一路径点的上/下坐标大于前一路径点的上/下坐标，则表明存在缠绕错误。

![](images/67b7658045f9e348c1b9847e2c0914ab7120e712797217c92c4cfa0b25382fa9.jpg)  
Fig S4: (A) OpenSim 模型中 MTU 缠绕错误的示例，其中 MTU 沿圆柱体圆周缠绕；以及 (B) 按序编号的相关 MTU 路径点，其中点 3 和点 4 说明了缠绕错误。

MTU 骨穿透惩罚采用了自动检测算法。首先，确定每个 MTU 跨越的关节以及它们可能穿透的刚体（即骨骼）。与包裹误差惩罚类似，确定每个 MTU 路径点并计算相邻路径点之间的向量。由于 OpenSim 应用程序编程接口（API）的限制，无法直接确定包裹切入/切出点之间的路径。因此，该惩罚仅考虑在以下情况之间与骨骼表面相交的向量：(i) 两个固定点（例如，起点或途经点）之间，(ii) 固定点与包裹切入点之间，(iii) 包裹切出点与包裹切入点之间（在不同的包裹表面上），以及 (iv) 包裹切出点与固定点之间（Fig S5）。假设如果包裹切入/切出点没有发生穿透，则中间路径也没有发生穿透。

![](images/15d13f247a053213441a56d32ce2b1a3ef32ae6ade904e97f67a21b7efec00f1.jpg)  
Fig S5: (A) OpenSim 内胫股关节（TFJ）屈曲 $1 0 0 ^ { \mathrm { o } }$ 时股内侧肌路径的示意图，以及 (B) OpenSim API 中可用的各个路径点。其中点 1、6、7 和 8 为固定点，点 2 和 4 为包裹切入点，点 3 和 5 为包裹切出点。使用所提出的框架，测试的路径点对为：1-2、3-4、5-6、6-7 和 7-8。

一旦确定了各项惩罚，它们就会被合并为一个单一的惩罚值，其中每个惩罚函数如果返回阳性测试结果，就会计入惩罚值。惩罚函数和目标准则被合并为一个加权值，通过优化将其最小化（公式 1）。下文对各个优化函数和惩罚函数进行了详细的明确说明。

$$
w v = s p l i n e L e n N + l e n N + s p l i n e N o r m E r r + m a G r a d E r r + p e n\tag{1}
$$

式中，为加权值，为拟合到 OpenSim MTU 长度的样条长度平滑度度量，为从 OpenSim API 导出的 MTU 长度的平滑度度量，为 OpenSim 导出长度与三次 B 样条拟合 MTU 长度之间的归一化误差，为 MTU 力臂与目标数据之间的归一化误差，为总惩罚值。

splineLenN：提供了样条化 MTU 长度平滑度的估计。此处由 B 样条方法定义的每个关节角度下的 MTU 长度对关节角度求导。重复该求导过程（使用 numpy.gradient 函数），直到导数的均值、最大值、最小值和极差低于定义的阈值，该阈值比测试值小 9 个数量级。实现这一过程所需的求导次数定义了 splineLenN 的值。

lenN：与上述 splineLenN 度量相同，但它不使用所实现的 B 样条方法中的长度，而是直接使用源自

splineNormErr：旨在确保所采用的 B-spline 方法不会人为增加 MTU（肌肉-肌腱单元）长度的平滑度。其计算方式为：来自 OpenSim API 的 MTU 长度与 B-spline 估计值之间的均方根差，并归一化至来自 OpenSim 的 MTU 长度。

maGradErr：提供了尸体数据与模型数据之间相似性的估计。如上所述，尸体与模型的力臂模式均表示为梯度，即力臂变化量 / 关节角度变化量。随后将 maGradErr 计算为尸体值与模型值之间的均方根差，并归一化至尸体值。

pen：代表累加的惩罚值，如前所述，该框架中实施了三种惩罚项。第一种是包络误差惩罚，第二种是 MTU 骨骼穿透误差惩罚，第三种是 MTU 极性惩罚。这些误差若出现，每个都会以 250 的惩罚值累加到最终误差中。

## 附录 3：各模型的模型评估标准结果

本研究中使用的综合评估指标如下所示，分别对应表 S1 至 S4 中的模型 1 至 4（Models 1–4）。

Table S1：采用拟合包络面和通用关节模型的 MAP-Client 模型（Model 1）的性能标准结果。

<table><tr><td></td><td>极性惩罚</td><td>肌肉穿透</td><td>长度平滑度</td><td>力臂平滑度</td><td>力臂梯度误差</td></tr><tr><td>M01</td><td>7 (29.2%)</td><td>8 (33.3%)</td><td> $3.5 \pm 11.5$ </td><td> $1.5 \pm 0.4$ </td><td> $1270.8 \pm 1964.14$ </td></tr><tr><td>M02</td><td>4 (16.7%)</td><td>8 (33.3%)</td><td> $1.1 \pm 0.2$ </td><td> $3.1 \pm 6.5$ </td><td> $19527 \pm 72014$ </td></tr><tr><td>M03</td><td>7 (29.2%)</td><td>8 (33.3%)</td><td> $1.4 \pm 1.1$ </td><td> $1.6 \pm 1.1$ </td><td> $1891.7 \pm 4145.5$ </td></tr><tr><td>M07</td><td>6 (25.0%)</td><td>9 (37.5%)</td><td> $1.3 \pm 0.8$ </td><td> $1.42 \pm 0.9$ </td><td> $1704.2 \pm 2666.1$ </td></tr><tr><td>M09</td><td>5 (20.8%)</td><td>13 (54.2%)</td><td> $2.1 \pm 4.8$ </td><td> $6.7 \pm 20.6$ </td><td> $55325.7 \pm 182702$ </td></tr><tr><td>M11</td><td>8 (33.3%)</td><td>7 (29.2%)</td><td> $3.4 \pm 10.8$ </td><td> $6.8 \pm 23.4$ </td><td> $42438.2 \pm 20233$ </td></tr></table>

其中极性惩罚和肌肉穿透以每项惩罚出现的次数和百分比报告，其余指标以本分析所涉及的 24 块肌肉的平均值 ± 标准差报告。在所有情况下，数值越低代表结果越好。

Table S2：采用拟合包络面和个性化关节模型的 MAP-Client 模型（Model 2）的性能标准结果。

<table><tr><td></td><td>极性惩罚</td><td>肌肉穿透</td><td>长度平滑度</td><td>力臂平滑度</td><td>力臂梯度误差</td></tr><tr><td>M01</td><td>10 (41.7%)</td><td>8 (33.3%)</td><td>2.7 ± 7.6</td><td>2.8 ± 5.3</td><td>9915.8 ± 416417.7</td></tr><tr><td>M02</td><td>4 (16.7%)</td><td>12 (50%)</td><td>1.6 ± 2.1</td><td>5.7 ± 18</td><td>33246.5 ± 157718</td></tr><tr><td>M03</td><td>3 (12.5%)</td><td>10 (41.7%)</td><td>1.2 ± 0.2</td><td>1.5 ± 0.5</td><td>683.9 ± 741.9</td></tr><tr><td>M07</td><td>6 (25%)</td><td>9 (37.5%)</td><td>1.1 ± 0.2</td><td>3.7 ± 5.5</td><td>7808.9 ± 13452.1</td></tr><tr><td>M09</td><td>12 (50%)</td><td>15 (60%)</td><td>1.7 ± 1.8</td><td>7.2 ±17.3</td><td>45440.5 ± 147443</td></tr><tr><td>M11</td><td>9 (37.5%)</td><td>11 (45.8%)</td><td>1.12 ± 0.19</td><td>3.9 ± 9.7</td><td>22362.5 ± 96077.7</td></tr></table>

其中极性惩罚和肌肉穿透以每种惩罚出现的次数和百分比表示，其余指标以本分析中考虑的 24 块肌肉的平均值 ± 标准差表示。在所有情况下，数值越低代表结果越好。

Table S3. 具有优化包络面和通用关节模型的 MAP-Client 模型的性能准则结果（模型 3）。

<table><tr><td></td><td>极性惩罚</td><td>肌肉穿透</td><td>长度平滑度</td><td>力臂平滑度</td><td>力臂梯度误差</td></tr><tr><td>M01</td><td>7 (29.2%)</td><td>0 (0.0%)</td><td> $1.06 \pm 0.2$ </td><td> $1.36 \pm 0.9$ </td><td> $97.88 \pm 251.9$ </td></tr><tr><td>M02</td><td>7 (29.2%)</td><td>0 (0.0%)</td><td> $1.37 \pm 1.3$ </td><td> $1.36 \pm 0.8$ </td><td> $36.46 \pm 49.7$ </td></tr><tr><td>M03</td><td>5 (20.84%)</td><td>1 (4.2%)</td><td> $1.23 \pm 0.7$ </td><td> $1.32 \pm 0.4$ </td><td> $87.62 \pm 151.3$ </td></tr><tr><td>M07</td><td>7 (29.2%)</td><td>4 (16.7%)</td><td> $1.17 \pm 0.2$ </td><td> $1.40 \pm 0.6$ </td><td> $110.94 \pm 188.7$ </td></tr><tr><td>M09</td><td>6 (25.0%)</td><td>2 (8.3%)</td><td> $1.06 \pm 0.2$ </td><td> $1.30 \pm 0.6$ </td><td> $54.42 \pm 87.3$ </td></tr><tr><td>M11</td><td>5 (20.8%)</td><td>1 (4.2%)</td><td> $1.13 \pm 0.2$ </td><td> $1.72 \pm 0.6$ </td><td> $142.36 \pm 186.1$ </td></tr></table>

其中极性惩罚和肌肉穿透以每种惩罚出现的次数和百分比表示，其余指标以本分析中考虑的 24 块肌肉的平均值 ± 标准差表示。在所有情况下，数值越低代表结果越好。

Table S4. 具有优化包络面和个性化关节模型的 MAP-Client 模型的性能准则结果（模型 4）。

<table><tr><td></td><td>极性惩罚</td><td>肌肉穿透</td><td>长度平滑度</td><td>力臂平滑度</td><td>力臂梯度误差</td></tr><tr><td>M01</td><td>3 (12.5%)</td><td>3 (12.5%)</td><td> $1.41 \pm 0.9$ </td><td> $1.46 \pm 0.7$ </td><td> $422.44 \pm 1126.1$ </td></tr><tr><td>M02</td><td>0 (0.0%)</td><td>2 (8.3%)</td><td> $0.99 \pm 0.2$ </td><td> $1.33 \pm 0.4$ </td><td> $32.50 \pm 35.6$ </td></tr><tr><td>M03</td><td>4 (16.7%)</td><td>5 (20.8%)</td><td> $1.58 \pm 2.1$ </td><td> $1.83 \pm 1.9$ </td><td> $438.29 \pm 1983.8$ </td></tr><tr><td>M07</td><td>Fig4(16.7%)</td><td>9 (37.5%)</td><td> $1.62 \pm 1.6$ </td><td> $1.71 \pm 0.9$ </td><td> $150.03 \pm 302.1$ </td></tr><tr><td>M09</td><td>5 (20.8%)</td><td>5 (20.8%)</td><td> $1.32 \pm 0.2$ </td><td> $1.29 \pm 0.4$ </td><td> $169.25 \pm 321.1$ </td></tr><tr><td>M11</td><td>4 (16.7%)</td><td>6 (25.0%)</td><td> $1.14 \pm 0.2$ </td><td> $1.68 \pm 1.1$ </td><td> $186.06 \pm 353.7$ </td></tr></table>

其中极性惩罚和肌肉穿透以每种惩罚出现的次数和百分比表示，其余指标以本分析中考量的 24 块肌肉的平均值 ± 标准差表示。在所有情况下，数值越低代表结果越好。

## 附录 4：关节机制的比较

本研究中实现了两种不同类型的关节机制，第一种是类似于通用 gait2392 模型的通用实现，第二种是基于医学影像分割的个性化关节机制。由于缺乏金标准真值数据，未在这两种关节机制之间进行直接验证。采用个性化关节机制是因为观察到通用关节机制中存在关节运动误差，具体而言，在膝关节屈曲 90 度时，观察到胫骨向上（上侧）平移过度（Fig S6）。

![](images/ee1b183288984ef487cc3d9ae8e46552ebef1d5f309fe41e4f9b4cab9cb9089b.jpg)  
Fig S6：在通用 gait 2392 模型 (A)、采用通用关节机制的模型 (B) 和个性化关节机制 (C) 中，胫股关节屈曲 90 度时胫骨位置的比较  
除了这些看似非生理性的运动之外，在估计的次级运动学中观察到的变异在胫股关节和髌股关节运动学中均表现出高得多的变异性（Fig S8、S9），尤其是与通用关节机制中的情况相比（Fig S7）

![](images/cc5501139340795898e3a2efa7242f0ae12a7a31cc494071caf82ceff053624b.jpg)  
Fig S7：由 MAP Client 生成的具有通用关节机制的模型所测得的每位参与者在 6 个 DOF 中的胫股关节运动，其中每种颜色代表不同的参与者。平移以米为单位报告，旋转以弧度为单位报告。注意，每项运动均相对于 TFJ 屈曲角表示。

![](images/01d01c6e0580b43417d70dd6ca693a054ca2da0c21f765701a8b85cfbda671da.jpg)  
Fig S8: 每位受试者在 6 个自由度（DOF）下的个性化 TFJ（胫股关节）运动学，其中每种颜色代表不同的受试者。平移以米（metres）为单位汇报，旋转以弧度（radians）为单位汇报。请注意，各项运动均相对于 TFJ 屈曲角度表示。

![](images/24e37915736e35bebb4816b8963615d0d44bec8ae9940f7d0d98f81a5adcb475.jpg)  
Fig S9: 每位受试者在 6 个自由度（DOF）下的个性化 PFJ（髌股关节）运动，其中每种颜色代表不同的受试者。平移以米（metres）为单位汇报，旋转以弧度（radians）为单位汇报。请注意，各项运动均相对于 TFJ 屈曲角度表示。