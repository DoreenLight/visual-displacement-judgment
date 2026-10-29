# N1 EMG与超声驱动的个性化肌肉-肌腱参数估计

> 核心问题：如何将表面肌电信号与超声实测肌纤维长度整合到Hill型神经肌肉骨骼模型中，通过直接配点最优控制估计脑卒中偏瘫患者的受试者特异性肌肉-肌腱参数（F_max、l_M,opt、l_TS、k_T），解决个体化肌肉冗余问题？

## S1 原始资料依据

### 来源01｜EMG-Constrained and Ultrasound-Informed Muscle-Tendon Parameter Estimation in Post-Stroke Hemiparesis（IEEE TBME 2024）

**研究问题与动机**
- 针对脑卒中后踝关节肌肉-肌腱单元(muscle-tendon unit, MTU)的继发性形态与力学特性改变，针对**个体化肌肉冗余**(muscle redundancy)问题，结合实验性肌电(EMG)信号与实测肌纤维长度，估计受试者特异性(subject-specific)肌肉-肌腱参数（来源01）。
- 研究的科学问题定位是"中风后偏瘫患者的个体化肌肉-肌腱参数估计，通过整合EMG和超声测量数据实现个性化神经肌肉骨骼建模"（来源01）。

**建模与肌肉模型**
- 采用**Hill型三元素肌肉模型**(series elastic element SE, contractile element CE, parallel elastic element PE)（来源01）。
- 采用De Groote等人提出的**柔顺肌腱建模**（compliant tendon modeling），将OpenSim的`gait10musc18`模型修改为仅包含胫骨与足的踝关节驱动器版本（来源01）。
- 待估计的三块踝关节肌肉：腓肠肌 gastrocnemius (GAS)、比目鱼肌 soleus (SOL)、胫骨前肌 tibialis anterior (TA)（来源01）。

**参数估计的数学表述**
- 估计参数集合 $\mathbf{p} = [l_{M,\text{opt}}, l_{TS}, k_T, F_{\max}]$，分别为optimal fiber length、tendon slack length、tendon stiffness、maximum isometric force（来源01）。
- 膝关节角度在所有试验和受试者中保持恒定为30°，踝关节角度由dynamometer记录（来源01）。
- 通过muscle redundancy solver优化，**最小化肌肉激活平方和同时最大化测量与模拟的关节力矩、EMG和肌纤维长度的一致性**（来源01）。
- 通过**直接配点**(direct collocation)将最优控制问题转录为稀疏非线性规划(NLP)，采用梯形积分方案，mesh size为10ms，使用IPOPT求解器求解，依赖自动微分工具CasADi计算梯度（来源01）。

**目标函数构成**
- 目标函数 $J$ 由四项加权和构成（来源01）：
  - $J_{\text{EMG}}$：模拟兴奋 $e(t)$ 与归一化实验EMG $\hat{e}(t)$ 的差异；
  - $J_{\text{FL}}$：模拟纤维长度 $l_f(t)$ 与超声测量纤维长度 $\hat{l}_f(t)$ 的差异；
  - $J_{\text{effort}}$：肌肉激活正则化项；
  - $J_{\text{res}}$：储备致动器力矩项。
- 初始权重为 $w_{\text{EMG}}=1$、$w_{\text{FL}}=1$、$w_{\text{effort}}=0.1$、$w_{\text{res}}=10$，并采用Delabastita等人提出的个体化权重调优方法（来源01）。
- reserve actuator项 $\tau_{\text{res}}$ 被添加到踝关节，定义为 $\tau_{\text{res}}^2$ 对时间的积分(Eq.10)，用于提高数值稳定性并衡量模型与实验数据的不匹配（来源01）。

**实验装置与数据采集**
- 受试者半直立坐姿，膝关节屈曲30°，足部固定于dynamometer脚踏板（来源01）。
- 使用Noraxon表面EMG系统(3000 Hz)按欧洲SENIAM推荐方案采集GAS/SOL/TA信号，原始EMG经20–500 Hz高通、整流、6 Hz低通Butterworth零相移滤波处理，并归一化至MVC试验最大值（来源01）。
- 同步使用**Mindray M9超声系统**（6–14 MHz，38 mm宽线性探头）采集肌纤维长度，由UltraTrack软件自动追踪并人工校验（来源01）。
- 运动学使用Qualisys动作捕捉系统(Mocap)的marker位置进行模型缩放，脑卒中组则采用手动测量的人体测量学长度（来源01）；通用肌肉骨骼模型`gait10dof18musc`通过静态marker轨迹或腿长测量进行缩放以匹配个体人体测量学（来源01）。

**样本与试验协议**
- **7例脑卒中偏瘫患者(paretic side)与7名健康对照(randomized side)**（来源01）。
- 分别执行踝关节跖屈(plantarflexion)与背屈(dorsiflexion)等长最大自主收缩(MVC)试验，分为三类（来源01）：
  - PF_GAS（跖屈+实测GAS纤维长度）；
  - PF_SOL（跖屈+实测SOL纤维长度）；
  - DF（背屈+实测TA纤维长度）。

**关键发现**
- 脑卒中组的**maximum isometric force (F_max)**、**tendon stiffness (k_T)**与**optimal fiber length (l_M,opt)均显著低于健康对照组**（来源01）。
- 与基于身高/体重的anthropometric scaling通用参数相比，使用肌肉冗余求解器估计的受试者特异性参数在muscle excitation与fiber length估计上均表现更优，并使力产生行为更符合实验测量值（来源01）。
- 验证指标：计算估计值与实验值之间的**root mean square error (RMSE)**，用于评估muscle excitation和muscle fiber length的跟踪精度，结果表明所估计的参数具备生理学意义(physiologically meaningful)（来源01）。

**方法对比与定位**
- 相较于Pizzolato等人提出的**CEINMS工具箱**（标定EMG-informed neuromusculoskeletal modeling，最大等长力、最佳纤维长度、肌腱松弛长度标定），本方法额外纳入实验性纤维长度作为约束并估计tendon stiffness（来源01）。
- 相较于Delabastita等人基于健康受试者小腿肌肉的**肌肉冗余求解器**（仅估计gastrocnemius medialis纤维长度且将EMG按行走最大值归一化、不适用于运动功能受损患者），本方法拓展到脑卒中患者并对三块踝关节肌肉同时估计参数（来源01）。
- 文中引用的相关工作表明EMG驱动神经肌肉骨骼模型框架已被用于exoskeleton control strategy design [24]–[26]、joint torque prediction [17], [23]及joint kinematics prediction [27]，所提方法填补了将EMG驱动建模与超声信息结合的最优化方法应用于脑卒中患者肌肉-肌腱参数评估的空白（来源01）。
- 应用研究结果可为脑卒中后治疗与康复规划(treatment/rehabilitation planning)提供依据（来源01）。

**建模与仿真平台**
- OpenSim (gait10musc18)（来源01）；肌肉冗余求解(muscle redundancy solver)概念；与Hill-type muscle model、force-length/force-velocity/tendon force-length关系紧密相关（来源01）。

### 来源07｜湖南大学章隆彬（Longbin Zhang）个人主页

- 论文《EMG-constrained and ultrasound-informed muscle-tendon parameter estimation in post-stroke hemiparesis》发表于 **IEEE TBME 2024, vol.71, no.6, pp.1798-1809**（IF:4.6, Q2）（来源07）。
- 核心研究贡献之一：开发**全面可靠的算法量化肌肉肌腱变化，可用于制定个性化治疗计划**（来源07）。
- 实验室研究方向涵盖：康复机器人、外骨骼康复机器人、脑机接口（BCI）、智慧医疗、具身智能、下肢骨肌系统生物力学、可穿戴辅助设备设计与控制、肌电控制与深度信息融合控制、人体运动仿真、数字孪生、物理信息与机器学习协同建模、康复医学、老年人摔倒风险评估、运动平衡机制研究（来源07）。

### 来源11｜OpenAlex 作者档案（A5002794703）

- 研究者身份：Longbin Zhang（OpenAlex ID: A5002794703，ORCID: 0000-0001-8785-5885），累计 works_count=100 篇，h_index=26（来源11）。
- 研究方向主题中与主题强匹配：① "Prosthetics and Rehabilitation Robotics" 18 篇；② "Muscle activation and electromyography studies" 20 篇；③ "Stroke Rehabilitation and Recovery" 12 篇（来源11）。
- 主要合作机构包括 Hunan University（湖南大学）、Nanyang Technological University、Rehabilitation Research Institute of Singapore、KTH Royal Institute of Technology、福州大学等（来源11）。

### 来源49｜Simulation of Lower Limb Muscle Moment and Total Muscle Fiber Force Using OpenSim for Knee Osteoarthritis（参考方法学对比）

- 采用 OpenSim 神经肌肉骨骼建模软件分析关键下肢肌肉（medial gastrocnemius、vastus lateralis、rectus femoris）相对于膝关节屈曲力矩的总肌纤维力和肌肉力矩（来源49）。
- 关键发现：膝OA患者中 medial gastrocnemius 与 vastus lateralis 的总肌纤维力和肌肉力矩均高于正常人；rectus femoris 表现相反（来源49）。
- 该方法可应用于康复机器人与外骨骼设计中**肌肉-肌腱参数估计**及**关节力矩预测**（来源49）。

### 来源46｜Clinical Implications of Muscle-Tendon & Force Interplay（参考应用场景）

- 采用表面肌电（sEMG）记录技术，对股外侧肌（m. vastus lateralis）和腓肠肌（m. gastrocnemius）的肌电活动进行采集与分析，研究肌肉-肌腱与力的相互作用（来源46）。
- 将sEMG作为评估肌肉功能、肌肉-肌腱力学交互及神经肌肉控制状态的临床工具，适用于**肌腱损伤**及**慢性疾病导致的肌肉功能改变评估**，与康复机器人/神经肌肉骨骼建模中的肌肉状态估计相关（来源46）。

### 来源47｜Automated creation and tuning of personalised muscle paths for OpenSim（参考方法学）

- 提出一种自动化框架，利用 population-based morphing 与 atlas- and optimisation-based methods 创建个性化 OpenSim 肌骨模型（来源47）。
- 针对个性化肌骨建模中两个关键难点：① 完整的胫股关节与髌股关节运动学；② MTU 路径与运动学（肌肉长度与力臂）（来源47）。
- 该框架集成于 MAP Client 软件包，证明自动化个性化 MTU 路径调整方法可用于人体运动学与运动控制研究，与康复机器人中个性化神经肌骨建模及关节力矩预测相关（来源47）。

## S2 核心观点提炼

**观点1：脑卒中后肌肉-肌腱单元经历显著的结构与力学重塑，必须通过受试者特异性参数估计来替代通用人体测量学缩放。**
- 支撑证据：7例脑卒中偏瘫患者与7名健康对照的对比试验表明，脑卒中组的 maximum isometric force (F_max)、tendon stiffness (k_T) 与 optimal fiber length (l_M,opt) 均**显著低于健康对照组**（来源01）；与基于身高/体重的anthropometric scaling通用参数相比，受试者特异性参数在 muscle excitation 与 fiber length 估计上均表现更优（来源01）。

**观点2：将表面肌电信号与超声实测肌纤维长度同时作为约束引入直接配点最优控制框架，可有效求解脑卒中患者的肌肉冗余问题。**
- 支撑证据：方法核心为 muscle redundancy solver 优化，最小化肌肉激活平方和同时最大化测量与模拟的关节力矩、EMG和肌纤维长度的一致性（来源01）；目标函数 $J$ 由 $J_{\text{EMG}}$（模拟兴奋 $e(t)$ 与归一化实验EMG $\hat{e}(t)$ 的差异）、$J_{\text{FL}}$（模拟纤维长度 $l_f(t)$ 与超声测量纤维长度 $\hat{l}_f(t)$ 的差异）、$J_{\text{effort}}$（肌肉激活正则化项）、$J_{\text{res}}$（储备致动器力矩项）四项加权和构成（来源01）；相较于Pizzolato等人的CEINMS工具箱，本方法额外纳入实验性纤维长度作为约束并估计tendon stiffness；相较于Delabastita等人的健康受试者肌肉冗余求解器，本方法拓展到脑卒中患者并对三块踝关节肌肉同时估计参数（来源01）。

**观点3：Hill型三元素模型（SE+CE+PE）+ De Groote柔顺肌腱建模 + 直接配点NLP（IPOPT/CasADi, 10ms mesh）构成了脑卒中患者参数估计的可行数值管线。**
- 支撑证据：采用直接配点(direct collocation)最优控制方法，基于Hill-type三元素肌肉模型与De Groote柔顺肌腱建模，使用OpenSim gait10musc18踝关节驱动器版本（来源01）；通过直接配点将最优控制问题转录为稀疏非线性规划(NLP)，采用梯形积分方案，mesh size为10ms，使用IPOPT求解器求解，依赖自动微分工具CasADi计算梯度（来源01）；膝关节角度在所有试验和受试者中保持恒定为30°，踝关节角度由dynamometer记录（来源01）。

**观点4：超声成像（Mindray M9, 6–14 MHz, UltraTrack追踪）是捕捉活体肌纤维长度的关键技术，使参数估计具有直接形态学锚点。**
- 支撑证据：同步使用Mindray M9超声系统(6–14 MHz，38 mm宽线性探头)采集肌纤维长度，由UltraTrack软件自动追踪并人工校验（来源01）；试验分为PF_GAS、PF_SOL、DF三类，每类对应不同的肌纤维长度测量目标（来源01）。

**观点5：该参数估计方法是连接"中风后个性化治疗/康复规划"与"外骨骼按需辅助控制"的中间桥梁。**
- 支撑证据：研究结果可为脑卒中后治疗与康复规划提供依据（来源01）；EMG驱动神经肌肉骨骼模型框架已被用于 exoskeleton control strategy design [24]–[26]、joint torque prediction [17], [23] 及 joint kinematics prediction [27]（来源01）；OpenSim神经肌肉骨骼建模方法可应用于康复机器人与外骨骼设计中肌肉-肌腱参数估计及关节力矩预测（来源49）；sEMG作为评估肌肉功能、肌肉-肌腱力学交互及神经肌肉控制状态的临床工具，适用于肌腱损伤及慢性疾病导致的肌肉功能改变评估（来源46）。

**观点6：自动化个性化MTU路径调整与个性化关节机制构建是当前OpenSim建模的活跃方向，与本研究的"个性化"理念形成方法学互补。**
- 支撑证据：基于医学影像构建了个性化的胫股与髌股刚体关节机制，并通过 atlas-based 与 optimisation-based 方法生成个性化 MTU 路径与运动学；该框架集成于 MAP Client 软件包，证明自动化个性化 MTU 路径调整方法可用于人体运动学与运动控制研究（来源47）。

## S3 与其他节点的关系

**N1 ↔ 物理信息混合模型驱动的下肢多关节力矩预测**
- 来源01的方法明确引用了 EMG-driven NMS 模型框架已被用于 joint torque prediction [17], [23]，说明本节点的下游（肌肉-肌腱参数）输出可直接输入多关节力矩预测模型（来源01）。
- 来源02、03、04、05都构建了基于NMS或NMS-informed ANN/LSTM的下肢关节力矩预测方法，其底层都依赖受试者特异性肌肉-肌腱参数；本节点（N1）的参数估计为这些下游力矩预测模型提供准确的个性化输入（来源02、03、04、05）。
- 来源03的CEINMS模型沿用了Pizzolato等人的标定方法，本节点（N1）的扩展恰是"额外纳入实验性纤维长度作为约束并估计tendon stiffness"，形成方法上的递进关系（来源01、03）。

**N1 ↔ 面向外骨骼按需辅助的自适应控制与不确定性量化**
- 来源01将本研究的应用场景定位为脑卒中后治疗与康复规划，并明确引用 exoskeleton control strategy design 为下游应用场景（来源01）。
- 来源03进一步将NMS-informed ML模型与BNN/GP不确定性量化结合，应用到膝外骨骼按需辅助控制，本节点（N1）的参数估计是这类控制框架的"物理基础层"（来源03）。
- 来源17模拟了膝关节外骨骼的辅助策略与人机交互，发现交互力与生理力矩之间存在trade-off，其中肌肉冲量是主要贡献因素，这进一步说明个性化肌肉-肌腱参数估计对于准确模拟人机交互至关重要（来源17）。

**N1 ↔ 无标记运动捕捉与机器学习的临床步态与平衡评估**
- 来源01使用的运动学数据采集依赖Qualisys动作捕捉系统的marker位置进行模型缩放（标记式MoCap）；来源06、41、42、37推进了markerless MoCap与机器学习结合用于临床步态与平衡评估；未来可将本节点（N1）的参数估计流程推广至markerless MoCap环境以降低临床采集门槛（来源01、06、41、42、37）。
- 来源06的增强TUG框架中，markerless MoCap系统的精度与state-of-the-art系统相当，这为脑卒中患者接受无标记MoCap驱动的参数估计提供了技术可行性（来源06）。

**N1 ↔ 面向运动功能恢复的脑机接口系统**
- 来源44展示了MI-BCI控制的电刺激对急性脑卒中偏瘫患者下肢功能恢复的临床效果（FMA-LE显著优于对照组，p<0.05）（来源44）。
- 来源45提出RLLFC算法用于下肢MI-BCI系统解码（来源45）。
- 本节点（N1）聚焦于运动输出端的肌肉-肌腱参数个性化，可与BCI输入端的运动想象解码系统协同，形成"BCI意图解码 → NMS个性化建模 → 外骨骼辅助执行"的闭环康复链路（来源44、45）。

**N1 ↔ 实验室团队与国际合作网络**
- 来源07详细列出章隆彬（Longbin Zhang）作为国家海外青年人才项目入选者、湖南大学博士生导师，长期围绕"基于生物力学的智能可穿戴技术"展开研究（来源07）。
- 来源11显示Longbin Zhang的OpenAlex档案涵盖Prosthetics and Rehabilitation Robotics (18篇)、Muscle activation and electromyography studies (20篇)、Stroke Rehabilitation and Recovery (12篇)三大主题，与本节点（N1）的三大方向（个性化肌肉-肌腱参数、EMG测量、卒中康复）完全契合（来源11）。
- 国际合作网络覆盖Hunan University、NTU、Rehabilitation Research Institute of Singapore、KTH Royal Institute of Technology、福州大学等（来源07、11），本研究作为KTH时期博士后的延续性工作（IEEE TBME 2024），体现了团队的国际化合作基础（来源07）。

## S4 在主线中的位置

**本节点在整体研究主线中处于"物理基础层/个性化建模层"的核心位置**，是从实验测量（EMG、超声、运动学）通往下游外骨骼控制与临床康复应用的"必经通道"。

1. **价值定位**：本节点解决了脑卒中偏瘫患者个体化肌肉冗余求解这一基础性问题，为下游的关节力矩预测、外骨骼控制策略设计、康复规划提供生理学意义的受试者特异性输入参数 $F_{\max}$、$l_{M,\text{opt}}$、$l_{TS}$、$k_T$（来源01）。如果不解决此问题，所有下游应用都将退化为基于通用人体测量学的群体平均参数，丧失个性化精度（来源01）。

2. **技术承上启下**：
   - **上游（数据层）**：依赖Noraxon sEMG系统与Mindray M9超声系统同步采集，对应实验测量能力（来源01）；
   - **本节点（建模层）**：将原始测量整合进Hill型NMS模型，通过直接配点最优控制反演受试者特异性参数（来源01）；
   - **下游（应用层）**：参数化的NMS模型可输入CEINMS等工具箱进行关节力矩预测（来源03、04），或与LSTM/BNN/GP等数据驱动方法融合用于外骨骼按需辅助控制（来源03），或与BCI系统对接形成完整康复闭环（来源44）。

3. **主线叙事角色**：本节点代表团队"个性化建模+临床转化"研究路径中的核心方法学贡献，与Longbin Zhang团队在KTH时期奠定的NMS-driven ML建模路线（踝关节hybrid-ANN [来源04]、LSTM+transfer learning [来源05]）以及当前湖南大学时期拓展的临床步态/平衡评估（markerless MoCap+TUG+BBS [来源06]）形成完整的方法学-应用学闭环（来源04、05、06、07）。论文发表于IEEE TBME 2024 (IF:4.6, Q2)，是该团队在个性化神经肌肉骨骼建模方向的标志性成果之一（来源07）。

4. **未来延展方向**：
   - 将参数估计方法从踝关节（GAS/SOL/TA）推广至膝、髋关节肌群，建立全下肢个性化模型（来源01、03、04、05）；
   - 将marker-based MoCap（Qualisys）替换为markerless MoCap（来源06、41、42），降低临床部署门槛；
   - 与MI-BCI等运动意图解码系统结合，形成"脑-肌-骨"多层级个性化康复系统（来源44、45）；
   - 在病理性步态群体（post-stroke、knee osteoarthritis）中验证markerless步态事件检测的LSTM模型泛化能力（来源15）。