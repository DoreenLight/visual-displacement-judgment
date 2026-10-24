# N5 基于 LSTM 与 NMS 模型的多关节关节力矩预测

> 核心问题：LSTM 神经网络与神经肌肉骨骼模型如何融合，对踝/膝/髋多关节的关节力矩进行可穿戴实时预测？

## S1 原始资料依据

### 来源01（Zhang2024-TBME-MuscleTendon，来自 TBME 论文）— 神经肌肉骨骼建模与肌-腱参数标定

**研究主题与人群**
- 本研究关注**卒中后患者**的踝关节肌-腱力学改变，使用个体化 EMG 标定的神经肌肉骨骼建模（calibrated EMG-informed neuromusculoskeletal modelling）来量化差异（来源01）。
- 对比组包括**健康对照组（Healthy Control）**与**卒中组（Post-stroke group）**，并通过 Shapiro-Wilk 检验对估计参数进行正态性检验（来源01）。

**Hill-type 肌肉模型与关键参数**
- 肌肉建模采用 Hill-type 肌肉模型（De Groote et al. 提出），由三元件构成：收缩元件（contractile element, CE）、并联弹性元件（parallel elastic element, PE）、串联弹性元件（series elastic element, SE）（来源01）。
- 该模型描述力-长度关系（force-length relationship）、力-速度关系（force-velocity relationship）、肌腱力-长度关系（tendon force-length relationship）（来源01）。
- 关键肌-腱参数包括：最大等长力 F_max（maximum isometric force）、最优肌纤维长度 l_M,opt（optimal fiber length）、肌腱刚度 k_T（tendon stiffness）、肌腱松弛长度 l_TS（tendon slack length）（来源01）。
- 卒中后患者**降低**的 F_max、k_T、l_M,opt 等参数，被用来量化中枢神经损伤对骨骼肌的次级改变（来源01）。
- 研究聚焦的三块踝部肌肉：**腓肠肌（GAS）**、**比目鱼肌（SOL）**、**胫骨前肌（TA）**；GAS 与 SOL 共享肌腱并组成**小腿三头肌复合体（triceps surae muscle complex）**，通过**跟腱（Achilles tendon）**连接（来源01）。

**肌冗余求解器与最优控制**
- 使用**直接配点法（direct collocation）**求解最优控制问题，处理肌冗余问题（muscle redundancy problem）（来源01）。
- 最优控制问题的状态变量为肌肉激活与肌腱力，控制变量为肌肉激发与储备驱动器力矩（reserve actuator torques）（来源01）。
- 成本函数 J 由四项构成：EMG 项（权重 1）、肌纤维长度项（权重 1）、努力项（权重 0.1）、储备驱动器项（权重 10）（来源01）。
- EMG 项度量肌肉激发与归一化 EMG 信号的差异；肌纤维长度项度量仿真与测量的肌纤维长度差异；努力项正则化最优控制问题（来源01）。

**肌冗余求解器与参数估计**
- 肌冗余求解器（muscle redundancy solver）输入 EMG 与肌纤维长度，估计 GAS、SOL、TA 三块肌肉的个体特异性肌-腱参数（来源01）。
- 在踝关节上额外添加**储备驱动器（reserve actuator）**，提供额外的关节力矩以吸收动作求解误差（来源01）。
- 使用**直接配点法**将最优控制问题转录为 NLP（非线性规划），用 IPOPT 求解，借助 CasADi 的自动微分（automatic differentiation）（来源01）。
- 力臂矩阵 R 描述肌腱动力学对应的肌肉力臂，依赖于关节角度（来源01）。

**实验与模型工具链**
- EMG：Noraxon Inc.（AZ, USA）系统，用于测量肌肉激活（来源01）。
- 超声：Mindray M9（位于深圳），用 UltraTrack 跟踪肌纤维长度（来源01）。
- 运动捕捉：Qualisys（位于瑞典哥德堡 Gothenburg）（来源01）。
- 测力计（dynamometer）测量踝关节角度与实测关节力矩（measured joint torque）（来源01）。
- 通用模型：OpenSim 中的 **gait10musc18**（gait10dof18musc）（来源01）。
- 数据处理：Butterworth 滤波（来源01）。
- 最大随意收缩（MVC）用于 EMG 归一化，踝背屈与跖屈在不同角度（5° 背屈、0° 跖屈、10° 跖屈、20° 跖屈）下进行力矩测量（来源01）。

**个体化 vs 通用模型**
- 通过交叉验证（cross-validation）评估个体化模型相对于通用模型（anthropometric scaling 依赖身高/体重的标量化模型）的优化性能（来源01）。
- 肌冗余求解器**优于**传统人体测量学缩放（anthropometric scaling），其估计的个体化参数包括 F_max、l_M_opt、k_T、l_TS（来源01）。
- 个体化模型产生个体特异性肌-腱参数，进而生成**个体化模型（individualized model）**；通用模型来自缩放后的肌肉骨骼模型（来源01）。
- 在卒中组中，**个体化模型的 RMSE 更低于通用模型**，意味着更精准的关节力矩/运动学预测（来源01）。

**功能角色的生理学背景**
- 踝跖屈肌（ankle plantarflexors）对抗外部背屈力矩，在**支撑相（stance phase）**活跃；踝背屈肌（ankle dorsiflexors）实现足廓清，活跃于**摆动相（swing phase）**（来源01）。
- 卒中会影响中枢神经系统并对骨骼肌造成次级改变（来源01）。
- 在下肢外骨骼、肌肉力量估计、可穿戴外骨骼控制等场景下，Hill-type 肌肉模型被广泛用作底层动力学描述（来源01）。

**相关方法学来源**
- 综述提到 Modenese et al.、Pizzolato et al.、Delabastita et al. 等学者在**形态学缩放（morphometric scaling）**、**最小二乘优化（least squares optimization）**、**优化算法（optimization algorithms）**、**CEINMS** 等方向的工作（来源01）。
- Living Human Digital Library 作为通用模型的解剖学数据来源之一（来源01）。
- 提到了**肌肉冗余求解器算法（muscle redundancy solver algorithm）**、**动态优化（dynamic optimization）**与**EMG 驱动建模（EMG-driven modeling）**等技术（来源01）。
- 研究的应用范围还可推广至帕金森病、运动员、健全个体等人群（来源01）。

### 来源02（Zhang2023-TNSRE-JointTorqueLSTM，来自 TNSRE 论文）— LSTM 与 NMS 模型的联合用于多关节力矩预测

**研究动机与应用场景**
- 论文核心关注**可穿戴传感器**（wearable sensors）与**按需辅助外骨骼控制器**（Assist-as-needed exoskeleton controllers, AAN controllers）（来源02）。
- 目标是为运动障碍患者（如**脊髓损伤 spinal cord injury**、**卒中后 post-stroke**）提供神经康复（neurorehabilitation）与神经肌肉恢复（neuromuscular recovery）的力矩估计（来源02）。
- AAN 控制器**被设计为**辅助肌肉产生的力矩，**保证**任务的成功执行（来源02）。

**多关节覆盖范围**
- 研究的关节力矩维度覆盖：**踝背屈/跖屈（ankle dorsiflexion/plantarflexion）**、**膝屈/伸（knee flexion/extension）**、**髋屈/伸（hip flexion/extension）**、**髋外展/内收（hip abduction/adduction）**（来源02）。
- 力矩来源由**逆动力学（inverse dynamics）**提供（来源02）。

**LSTM 网络与 NMS 模型并用**
- **关节力矩**分别可由**神经肌肉骨骼 NMS 模型**或**LSTM 长短期记忆网络**预测（来源02）。
- LSTM 适合捕捉步态时序依赖，NMS 模型提供物理可解释的肌肉-肌腱力矩贡献（来源02）。

**聚类分析与三类模型策略**
- 使用**聚类分析（cluster analysis）**对踝背屈/跖屈运动进行分组，基于**特征相似性**（characteristic similarity）（来源02）。
- 聚类分析**提升** NMS 模型与 LSTM 网络的估计精度（来源02）。
- 由此衍生出三类模型：**混合模型（Pooled models）**、**个体模型（Individual models）**、**聚类模型（Clustered models）**（来源02）：
  - Pooled 模型：使用全部 10 种运动数据训练（来源02）。
  - Individual 模型：使用每种单独运动的数据训练（来源02）。
  - Clustered 模型：使用每个聚类的数据训练（来源02）。

**模型评估结论**
- **Pooled 模型**对所有运动模式**可能过于通用**（too generic），预测精度受限（来源02）。
- **个体 NMS 模型**与**聚类 NMS 模型**均具有**良好的估计性能**（good estimation performance）（来源02）。
- 由此作者推断：在 LSTM 与 NMS 模型融合架构下，对运动模式进行聚类可在保持泛化能力的同时提升关节力矩预测精度，并满足可穿戴 AAN 控制器的实时需求（来源02）。

**对比方法**
- 论文对比基线为**人工神经网络（ANN）**与**多层感知机（MLP）模型**（来源02），用以说明 LSTM 在时序建模上的优势。

---

## S2 核心观点提炼

1. **LSTM 与 NMS 模型的融合体现在功能互补**：NMS 模型基于 Hill-type 肌肉动力学提供物理可解释的肌-腱力贡献，LSTM 则捕捉步态时序依赖；二者对踝/膝/髋多关节（踝跖屈/背屈、膝屈/伸、髋屈/伸、髋外展/内收）进行关节力矩估计（来源02）。

2. **三类建模策略（Pooled / Individual / Clustered）用以平衡泛化与精度**：混合模型使用全部运动数据，可能"过于通用"；个体模型精度高但需每动作单独训练；聚类模型通过**聚类分析**基于特征相似性分组，**提升 NMS 与 LSTM 估计精度**，是折中方案（来源02）。

3. **个体化肌-腱参数标定是 NMS 模型的精度前提**：使用 EMG + 超声（Noraxon + Mindray M9 / UltraTrack）作为输入，通过**直接配点法**求解最优控制问题，估计 GAS/SOL/TA 的 F_max、l_M,opt、k_T、l_TS 等参数；卒中患者表现出更低的 F_max、k_T、l_M,opt（来源01）。

4. **储备驱动器 + 多目标成本函数**提升估计的鲁棒性：在踝关节添加储备驱动器（reserve actuator），其成本函数权重为 10，远高于努力项（0.1）、EMG 项（1）、肌纤维长度项（1），使个体化模型 RMSE 低于通用模型（来源01）。

5. **OpenSim + CasADi + IPOPT 工具组合**使 Hill-type 肌肉模型在通用 gait10musc18 模型上完成跨工具链的优化求解，并支持多关节外骨骼控制所需的实时计算需求（来源01）。

6. **面向 AAN 外骨骼的实时可穿戴力矩预测**：AAN 控制器目标为"辅助肌肉产生的力矩"以保证任务成功；关节力矩可由逆动力学提供作为标签，由 LSTM/NMS 联合预测后嵌入到控制器中，形成闭环（来源02）。

---

## S3 与其他节点的关系

- **N1 章隆彬与湖南大学人工智能与机器人学院的实验室定位**：N5 的研究产出（TBME 与 TNSRE 论文）是该实验室"康复外骨骼 + 神经肌肉骨骼建模"方向的代表性成果，节点 N1 提供实验室与团队背景，N5 是核心方法学输出。

- **N2 神经肌肉骨骼建模的理论基础与 Hill-type 肌肉模型**：N5 直接建立在 N2 的理论基础上，使用其 CEINMS/EMG-driven modeling 框架，将 Hill-type 模型中的 CE/PE/SE 三元件与 F_max、l_M,opt、k_T、l_TS 等参数化定义用于个体化标定（来源01）。

- **N3 基于 EMG + 超声的个体化肌肉-肌腱参数标定**：N5 是 N3 方法论的工程落地。N3 提供 EMG（Noraxon）+ 超声（Mindray M9 / UltraTrack）的双模态采集范式，N5 将其转化为最优控制问题，并通过 IPOPT/CasADi 完成求解（来源01）。

- **N4 卒中后踝关节肌-腱力学改变的量化与对照研究**：N5 的卒中组 vs 健康对照对比、5° 背屈/0°/10°/20° 跖屈下的力矩测量、Shapiro-Wilk 正态性检验等，都是 N4 量化研究的具体方法（来源01）。

- **N6 基于聚类分析的个体化运动模式建模**：N5 中的聚类分析直接源于 N6 的方法——对踝背屈/跖屈进行特征相似性聚类，划分出 Pooled/Individual/Clustered 三类模型（来源02）。

- **N7 下肢外骨骼的按需辅助（AAN）控制策略**：N5 是 N7 控制器的**上游关节力矩预测模块**——只有先精准预测踝/膝/髋的肌肉性关节力矩，AAN 控制器才能"按需"补充缺口（来源02）。

- **N8 多模态实验采集与跨地域合作网络**：N5 中所采用的 Noraxon（USA, AZ）、Mindray M9（中国深圳）、Qualisys（瑞典哥德堡）等设备跨地域合作网络，是 N8 节点的实证体现（来源01、来源02）。

---

## S4 在主线中的位置

N5 处于知识树的**核心方法学枢纽位置**，承担承上启下功能：

- **承上**：汇总 N1（实验室定位）的研究背景，继承 N2（NMS 理论基础 + Hill-type 模型）与 N3（EMG+超声双模态参数标定），并直接对接 N4（卒中量化研究）的实验范式。

- **启下**：通过**聚类分析**（衔接 N6）与**多关节力矩预测**直接驱动 N7 的 AAN 外骨骼控制；并通过多模态实验采集与跨地域合作（N8）实现工程闭环。

- **技术定位**：N5 是从"参数标定"走向"实时可穿戴控制"的关键**方法学桥梁**——以 Hill-type NMS 模型保证物理可解释性，以 LSTM 网络保证时序建模能力，以聚类分析与个体化标定保证多关节多任务的精度，以 OpenSim/CasADi/IPOPT 工具链保证工程可实现性，最终输出踝/膝/髋多关节力矩实时预测，馈入 N7 的 AAN 外骨骼控制器。

简言之，N5 是"建模–预测–控制"链路的中段核心节点，既是 N2/N3/N4 的方法汇合点，也是 N6/N7/N8 的方法发散点。