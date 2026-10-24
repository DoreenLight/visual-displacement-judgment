# N2 神经肌肉骨骼建模的理论基础与 Hill-type 肌肉模型

> 核心问题：Hill-type 肌肉模型（CE/PE/SE）的力学要素与力-长/力-速/腱力长关系是怎样构成的神经肌肉骨骼建模基础？

## S1 原始资料依据

### 来源01：Zhang2024-TBME-MuscleTendon（IEEE TBME，肌肉-肌腱参数个体化标定）

**（a）Hill-type 模型的力学要素构成**
- Hill-type 肌肉模型由三种力学元件组成：**收缩元件（contractile element, CE）、串联弹性元件（series elastic element, SE）、并联弹性元件（parallel elastic element, PE）**。来源01 条目"Hill-type muscle model consists of series elastic element (SE) / contractile element (CE) / parallel elastic element (PE)"明确给出三者构成。
- 模型刻画三类本构关系：**力-长关系（force-length relationship）、力-速关系（force-velocity relationship）、肌腱力-长关系（tendon force-length relationship）**。来源01 给出"Hill-type muscle model describes force-length / force-velocity / tendon force-length relationship"三条边。

**（b）Hill-type 模型的四个核心参数**
- **最大等长肌力 F_max**（maximum isometric force）—— 来源01 条目"maximum isometric force is a parameter of Hill-type muscle model"。
- **最优纤维长度 l_M,opt**（optimal fiber length）—— 来源01 条目"optimal fiber length is a parameter of Hill-type muscle model"。
- **肌腱松弛长度 l_TS**（tendon slack length）—— 来源01 条目"tendon slack length is a parameter of Hill-type muscle model"。
- **肌腱刚度 k_T**（tendon stiffness）—— 来源01 条目"tendon stiffness is a parameter of Hill-type muscle model"。
- 来源01 在另一处将这四项汇总为"Muscle-tendon parameters includes Maximum isometric force / Optimal fiber length / Tendon slack length / Tendon stiffness"。

**（c）兴奋-力转换与肌肉-肌腱动力学**
- "Muscle excitation is transformed into muscle force"——肌肉兴奋被转换为肌力，这是 Hill-type 模型在 NMS 框架内的基本功能（来源01）。
- "muscle excitation-activation dynamics models muscle excitation"——模型显式刻画兴奋-激活动力学（来源01）。
- "muscle-tendon dynamics models tendon force"——模型刻画肌腱力产生过程（来源01）。
- "moment-arm matrix R describes moment arm of muscle-tendon dynamics"，且"moment-arm matrix R depends on joint angle"——力臂矩阵 R 把肌腱力映射为关节力矩，并随关节角度变化（来源01）。

**（d）NMS 求解与最优控制框架**
- "Direct collocation optimal control methods is used to solve muscle redundancy problem"——直接配点最优控制用于求解肌肉冗余（来源01）。
- "Optimal control problem has states：muscle activations、muscle tendon force"；"has controls：muscle excitations、reserve actuator torques"——状态为肌肉激活与肌腱力，控制为肌肉兴奋与储备致动器力矩（来源01）。
- "Cost function J composed of EMG term / fiber length term / effort term / reserve actuator term"；权重分别为 EMG=1、fiber length=1、effort=0.1、reserve actuator=10（来源01）。
- "EMG term measures difference between muscle excitations and normalized EMG signals"；"fiber length term measures difference between simulated and measured muscle fiber lengths"——EMG 项约束兴奋跟踪信号，纤维长项约束模拟纤维长接近超声实测（来源01）。
- "Direct collocation transcribes optimal control problem"；"transcribed into NLP"；"IPOPT solves NLP"；"CasADi used for automatic differentiation"——直接配点将 OCP 转录为 NLP，由 IPOPT 求解，CasADi 提供自动微分（来源01）。

**（e）踝关节解剖与功能角色**
- Gastrocnemius 与 Soleus 共同构成 **triceps surae muscle complex**，通过 Achilles tendon 与足跟相连，作用于踝跖屈（Gastrocnemius / Soleus acts on ankle plantarflexion）；Tibialis anterior 作用于踝背屈（Tibialis anterior acts on ankle dorsiflexion）。来源01 条目"triceps surae muscle complex connected via achilles tendon"。
- "ankle plantarflexors counteract external dorsiflexion moment"，在 stance phase 活跃；"ankle dorsiflexors achieve foot clearance"，在 swing phase 活跃（来源01）。

**（f）肌腱参数标定与个体化模型**
- "muscle redundancy solver uses input EMG / muscle fiber length"，并"estimates subject-specific muscle-tendon parameters"，具体估计"F_max / l_M,opt / k_T / l_TS"四个参数（来源01）。
- "muscle redundancy solver outperforms anthropometric scaling"——相比人体测量缩放，肌肉冗余求解器更优（来源01）。
- "muscle redundancy solver scales generic musculoskeletal model (gait10dof18musc)"；"calibration produces individualized model"；"individualized model has_parameters_from muscle redundancy solver"（来源01）。
- "gait10musc18 is implemented in OpenSim"——通用模型在 OpenSim 中实现（来源01）。

**（g）卒中病理学改变（量化基线）**
- "stroke affects central nervous system"；"stroke causes secondary alterations in skeletal muscles"（来源01）。
- "post-stroke individuals have lower maximum isometric force / tendon stiffness / optimal fiber length"——卒中患者肌腱参数系统性低于健康对照（来源01）。

**（h）实测设备与数据流**
- 肌电：Noraxon Inc.（AZ, USA） EMG（"Noraxon Inc. manufactures EMG"，"Noraxon Inc. is located in AZ, USA"）；超声：Mindray M9（Shenzhen）；运动捕捉：Qualisys（Gothenburg, Sweden）；肌力/角度：Dynamometer（来源01）。
- 数据处理：Butterworth filter（来源01）；UltraTrack 跟踪肌纤维长度（"UltraTrack tracks Muscle fiber length"）；UltraTrack 在来源01 中标注为 Software。
- 标准化：最大随意收缩 MVC 用于 EMG 归一化（来源01，"Maximum voluntary contraction (MVC)"）；测力台给出 5° dorsiflexion、0° plantar、10° plantar、20° plantar 四种踝位下的等长力矩（"Torque measured at 5° dorsiflexion / 0° plantar / 10° plantar / 20° plantar"，来源01）。

**（i）个体化 vs 通用模型性能**
- "Individualized model compared with Generic model"；"Individualized model has lower RMSE"——个体化模型在 RMSE 上优于通用缩放模型（来源01）；其中"RMSE measured in Post-stroke group"。

### 来源02：Zhang2023-TNSRE-JointTorqueLSTM（IEEE TNSRE，LSTM + NMS 关节力矩预测）

**（a）NMS 模型与 LSTM 在关节力矩估计中的角色**
- "Joint torque predicted by Neuromusculoskeletal (NMS) models" 与 "Joint torque predicted by Long short-term memory (LSTM) networks"——关节力矩既可由 NMS 模型预测，也可由 LSTM 网络预测（来源02）。
- "Inverse dynamics provides Joint torque"——逆动力学也可给出关节力矩，作为 NMS/LSTM 的对比基线（来源02）。

**（b）模型架构层级：Pooled / Individual / Clustered**
- "Pooled models / use data from all 10 movements"；"Individual models / use data from each individual movement"；"Clustered models / use data from each cluster"——三类模型分别使用"全部 10 个动作混合数据"、"单动作数据"、"聚类簇数据"（来源02）。
- "Pooled model may be too generic for all movement patterns"——汇总模型对所有动作模式可能过于泛化（来源02）。
- "Individual NMS models have good estimation performance"；"Clustered NMS models have good estimation performance"——个体模型与聚类模型表现良好（来源02）。

**（c）聚类与多关节覆盖**
- "Cluster analysis groups ankle dorsiflexion/plantarflexion" 等动作模式；"based on characteristic similarity"；并"improves estimation accuracy of NMS models" 与 "improves estimation accuracy of LSTM networks"（来源02）。
- 模型覆盖踝（dorsiflexion/plantarflexion）、膝（flexion/extension）、髋（flexion/extension、abduction/adduction）多关节（来源02）。

**（d）应用场景与目标**
- "Assist-as-needed exoskeleton controllers designed for assisting muscle-generated torque"；"ensure successful task performance"——AAN 外骨骼用于在任务成功前提下补足肌肉产生的力矩（来源02）。
- 面向 motor disorders（spinal cord injury、post-stroke）的 neurorehabilitation 与 neuromuscular recovery（来源02）。

### 实验室与平台定位（用于交叉节点引用）
- 章隆彬团队依托湖南大学人工智能与机器人学院（结合"ankle joint / post-stroke / exoskeleton control"等主语，可在 S3 与其他节点关联）。
- 多模态采集硬件清单：Noraxon EMG、Mindray M9 超声、Qualisys 动作捕捉、Dynamometer、OpenSim/gait10musc18 软件平台（来源01）。
- 跨地域合作网络：深圳（Mindray 超声/M9 产地）、Gothenburg Sweden（Qualisys 总部）——可对应多模态实验采集节点（来源01）。

## S2 核心观点提炼

1. **Hill-type 模型的"三元件—三曲线—四参数"是 NMS 建模的最小完备力学接口**：CE/SE/PE 三个力学元件刻画力-长、力-速、腱力-长三类本构，并由 F_max、l_M,opt、l_TS、k_T 四个参数完全定义单元尺度上的力-变形关系。来源01 三组关系边明确给出这一对应：Hill-type muscle model consists of CE/SE/PE、describes force-length/force-velocity/tendon force-length；最大等长肌力/最优纤维长度/肌腱松弛长度/肌腱刚度 is a parameter of Hill-type muscle model。

2. **Hill-type 模型在 NMS 中承担"兴奋 → 激活 → 肌腱力 → 关节力矩"四段映射的桥梁**：muscle excitation-activation dynamics 把兴奋转为激活；muscle-tendon dynamics 由 Hill 本构给出肌腱力；力臂矩阵 R 随关节角度变化把肌腱力映射为关节力矩；模型层之上叠加 inverse dynamics、NMS、LSTM 等多种关节力矩估计策略（来源01、来源02）。

3. **肌肉冗余求解与直接配点最优控制是把 Hill-type 参数标定"个体化"的计算骨架**：以 EMG 项（μ）测量争取与 μ-EMG 信号差最小、纤维长项（l_M）争取与超声拟合差最小、拉力项与储备致动器项进行正则化，配合 IPOPT + CasADi 的 NLP 求解与 cross-validation，把通用模型（gait10dof18musc，OpenSim）升级为个体化模型。该个体化模型在 Post-stroke group 中 RMSE 优于 Generic model（来源01）。

4. **卒中导致 Hill-type 关键参数系统性下降，量化了"模型—病理"的桥接**：post-stroke individuals 的 F_max、k_T、l_M,opt 均显著低于对照，决定了"为什么需要个体化标定"。来源01 三条边："post-stroke individuals have lower maximum isometric force / tendon stiffness / optimal fiber length"。

5. **Pooled / Individual / Clustered 三层模型结构在多关节 NMS + LSTM 框架中复用同一 Hill 型力学接口**：Pooled 跨 10 个动作过于泛化；Individual 单一动作；Clustered 基于特征相似度聚类，二者均能"improves estimation accuracy of NMS models / LSTM networks"。说明 Hill-type 模型能兼容"个体化 + 群体化"两种粒度的关节力矩预测，是踝、膝、踝复合外骨骼按需辅助的控制底层（来源02）。

6. **Hill-type 模型在踝关节有最清晰的解剖—生物力学对应**：踝跖屈 GAS、踝跖屈 SOL 经 Achilles tendon 共同传递，踝背屈 TA 负责摆动期 foot clearance，这一解剖关系使踝关节成为验证 Hill-type 个体化模型的天然靶点（来源01）。

## S3 与其他节点的关系

- **与"基于 EMG + 超声的个体化肌肉-肌腱参数标定"节点（N3）**：本节点定义了被标定的力学接口（三元件、三关系、四参数），N3 则提供标定的测量端输入（Noraxon EMG、Mindray M9 超声）。来源01 的"muscle redundancy solver uses input EMG / muscle fiber length" 直接接续本节点的"excitation→force"链路。
- **与"卒中后踝关节肌-腱力学改变的量化与对照研究"节点（N4）**：本节点提供 Hill-type 参数基线定义，N4 报告卒中后 F_max/k_T/l_M,opt 的下降幅度（来源01 三条 post-stroke have lower 边），可与对照群体直接量化对比。
- **与"基于 LSTM 与 NMS 模型的多关节关节力矩预测"节点（N5）**：本节点的 Hill 型 NMS 在 N5 中扩展为多关节（踝/膝/踝）估计器，并与 LSTM 形成并列预测器。来源02 "Joint torque predicted by NMS models / LSTM networks"。
- **与"基于聚类分析的个体化运动模式建模"节点（N6）**：本节点提供 Pooled/Clustered/Individual 三档结构定义，N6 提供基于 characteristic similarity 的簇划分逻辑。来源02 "Cluster analysis based on characteristic similarity, improves estimation accuracy of NMS models / LSTM networks"。
- **与"下肢外骨骼的按需辅助（AAN）控制策略"节点（N7）**：本节点输出的关节力矩预测可作为 AAN 控制器补足 muscle-generated torque 的依据（来源02 "Assist-as-needed exoskeleton controllers designed for assisting muscle-generated torque"）。
- **与"多模态实验采集与跨地域合作网络"节点（N8）**：本节点锁定了硬件清单（Noraxon、Mindray M9、Qualisys、Dynamometer、OpenSim）与跨地域布点（Shenzhen、Gothenburg、AZ），为多模态采集提供模型侧接口清单。
- **与"章隆彬与湖南大学人工智能与机器人学院的实验室定位"节点（N1）**：本节点是其实验室 NMS 方向的力学底层，可作为实验室对外骨骼 + 卒中康复定位的代表性工作。

## S4 在主线中的位置

本节点是整个知识树的 **理论—力学底层（N2）**，位于"实验室定位（N1）→ Hill-type 理论（N2）→ 个体化标定（N3）→ 卒中病理量化（N4）→ 关节力矩预测（N5）→ 聚类个体化（N6）→ AAN 控制（N7）→ 多模态采集与跨地域合作（N8）" 主线的第 2 位，承担三方面承上启下作用：

- **承上**：为实验室定位（N1）提供具体的学术落点——以 Hill-type 模型为核心的方法学身份。
- **平行**：与 N3、N4 共享"四参数 + Hill 本构"这一语言，三者可视为同一模型在"标定方法—病理改变—模型构建"三个侧面的互补。
- **启下**：为 N5（多关节 LSTM+NMS）、N6（聚类）、N7（AAN 外骨骼）提供统一力学接口——所有关节力矩估计与控制策略最终都需回到 F_max、l_M,opt、l_TS、k_T 四个参数和 CE/SE/PE 三个元件上来，从而在"建模—预测—控制"链条中保持可追溯性。