# 章隆彬 Longbin Zhang Hunan University 人工智能与机器人学院 rehabilitation robotics exoskeleton neuromusculoskeletal model surface EMG markerless motion capture brain computer interface muscle tendon estimation joint torque prediction gait balance assessment 实验室团队 合作网络 — 调研报告

> 生成日期: 2026-10-01
> 报告版本: 全量版
> 数据来源: 38 篇/个
> 知识节点: 8 个

## 来源覆盖审计

>已调用后端: 6 个；有最终贡献: 4 个。

| 来源 | 调用 | 命中 | 失败 | 过滤 | 去重 | 抓取失败 | 保留 |
|---|---:|---:|---:|---:|---:|---:|---:|
| arxiv | 29 | 202 | 6 | 168 | 29 | 0 | 5 |
| crossref | 29 | 334 | 0 | 129 | 85 | 47 | 60 |
| expert_seed | 0 | 0 | 0 | 0 | 0 | 0 | 13 |
| google_news | 29 | 49 | 0 | 48 | 1 | 0 | 0 |
| pubmed | 29 | 183 | 0 | 178 | 5 | 0 | 0 |
| web | 29 | 316 | 0 | 262 | 49 | 4 | 1 |
| wikipedia | 29 | 53 | 0 | 52 | 1 | 0 | 0 |

1KTH MoveAbility Lab, Department of Engineering Mechanics, KTH Royal Institute of Technology, Stockholm, Sweden, 2Department of Mathematics, University of Iowa, Iowa City, IA, United States(来源03)

Abstract—Balance and gait impairments play a key role                         we included TUG datasets of 70 participants with varying(来源06)

章隆彬，博士生导师、硕士生导师，入选国家海外青年人才项目(来源07)

博士毕业于瑞典皇家理工学院，博士后在新加坡南洋理工大学(来源07)

长期围绕**基于生物力学的智能可穿戴技术** 展开研究，包括精确解码型运动意图预测、个性化肌肉肌腱参数的可靠估计和多场景安全型人机交互。(来源07)

**研究方向：** 外骨骼康复机器人，脑机接口，智慧医疗，数据安全，具身智能，下肢骨肌系统生物力学研究，可穿戴辅助设备设计与控制，肌电控制和深度信息融合控制，人体运动仿真，数字孪生，物理信息与机器学习协同建模，康复医学，老年人摔倒风险评估，运动平衡机制研究。(来源07)

课题组招收**博士后、博士、硕士研究生、本科生毕业设计、本科生科研助理等** ，与瑞典皇家理工学院、香港理工大学、新加坡南洋理工大学保持长期合作关系，可推荐优秀学生赴海外联合培养(来源07)

欢迎感兴趣的同学与我联系：longbin@hnu.edu.cn(来源07)

**1）运动意图预测、地面反作用力预测、力矩预测、肌肉力预测、关节接触力预测**(来源07)

[章隆彬](https://robotics.hnu.edu.cn/info/1176/3002.htm)(来源09)

Thanks: L. Zhang is with the School of Artificial Intelligence and Robotics, Hunan University, Changsha, China and the School of Mechanical and Aerospace Engineering, Nanyang Technological University, Singapore.(来源12)

[(来源13)

Nanyang Technological University, School of(来源37)

Abstract— Research interest in exoskeleton assistance strate-                and where it is needed, thus improving the patient’s ability to(来源38)

Abstract—Secondary morphological and mechanical property changes in the muscle-tendon unit at the ankle joint are often observed in post-stroke individuals. These changes may alter the force generation capacity and affect daily activities such as locomotion. This work aimed to estimate subject-specific muscle-tendon parameters in individuals after stroke by solving the muscle redundancy problem using direct collocation optimal control methods based on experimental electromyography (EMG) signals and measured muscle fiber length. Subject-specific muscle-tendon parameters of the gastrocnemius, soleus, and tibialis anterior were estimated in seven post-stroke individuals and seven healthy controls. We found that the maximum isometric force, tendon stiffness and optimal fiber length in the post-stroke group were considerably lower than in the control group. We also computed the root mean square error between estimated and experimental values of muscle excitation and fiber length. The musculoskeletal model with estimated subject-specific muscle tendon parameters (from the muscle redundancy solver), yielded better muscle excitation and fiber length estimations than did scaled generic parameters. Our findings also showed that the muscle redundancy solver can estimate muscle-tendon parameters that produce force behavior in better accordance with the experimentally-measured value. These muscle-tendon parameters in the post-stroke individuals were physiologically meaningful and may shed light on treatment and/or rehabilitation planning(来源01)

Abstract—Accurately predicting joint torque using wearable sensors is crucial for designing assist-as-needed exoskeleton controllers to assist muscle-generated torque and ensure successful task performance. In this paper, we estimated ankle dorsiflexion/plantarflexion, knee flexion/extension, hip flexion/extension, and hip abduction/adduction torques from electromyography (EMG) and kinematics during daily activities using neuromusculoskeletal (NMS) models and long short-term memory (LSTM) networks. The joint torque ground truth for model calibrating and training was obtained through inverse dynamics of captured motion data. A cluster approach that grouped movements based on characteristic similarity was implemented, and its ability to improve the estimation accuracy of both NMS and LSTM models was evaluated. We compared torque estimation accuracy of NMS and LSTM models in three cases: Pooled, Individual, and Clustered models. Pooled models used data from all 10 movements to calibrate or train one model, Individual models used data from each individual movement, and Clustered models used data from each cluster. Individual, Clustered and Pooled LSTM models all had relatively high joint torque estimation accuracy. Individual and Clustered NMS models had similarly good estimation performance whereas the Pooled model may be too generic to satisfy all movement patterns. While the cluster approach improved the estimation accuracy in NMS models in some movements, it made relatively little difference in the LSTM neural networks, which already had high estimation accuracy. Our study provides practical implications for designing assist-asneeded exoskeleton controllers by offering guidelines for selecting the appropriate model for different scenarios, and has potential to enhance the functionality of wearable exoskeletons and improve rehabilitation and assistance for individuals with motor disorders(来源02)

Longbin Zhang1, Xiaochen Zhang1, Xueyu Zhu2, Ruoli Wang1 and Elena M. Gutierrez-Farewik1\*(来源03)

Abstract—In this work, we predicted ankle joint torque by combining a neuromusculoskeletal (NMS) solver-informed artificial neural network (hybrid-ANN) model with transfer learning based on joint angle and muscle electromyography signals. The hybrid-ANN is an ANN augmented with two kinds of features: 1) experimental measurements – muscle signals and joint angles, and 2) informative physical features extracted from the underlying NMS solver, such as individual muscle force and joint torque. The hybrid-ANN model accuracy in torque prediction was studied in both intra- and inter-subject tests, and compared to the baseline models (NMS and standard-ANN). For each prediction model, seven different cases were studied using data from gait at different speeds and from isokinetic ankle dorsi/plantarflexion motion. Additionally, we integrated a transfer learning method in inter-subject models to improve joint torque prediction accuracy by transferring the learned knowledge from previous participants to a new participant, which could be useful when training data is limited. Our results indicated that better accuracy could be obtained by integrating informative NMS features into a standard ANN model, especially in inter-subject cases; overall, the hybrid-ANN model predicted joint torque with higher accuracy than the baseline models, most notably in inter-subject prediction after adopting the transfer learning technique. We demonstrated the potential of combining physics-based NMS and standard-ANN models with a transfer learning technique in different prediction scenarios. This procedure holds great promise in applications such as assistance-asneeded exoskeleton control strategy design by incorporating the physiological joint torque of the users(来源04)

— Estimation of joint torque during movement provides important information in several settings, such as effect of athletes’ training or of a medical intervention, or analysis of the remaining muscle strength in a wearer of an assistive device. The ability to estimate joint torque during daily activities using wearable sensors is increasingly relevant in such settings. In this study, lower limb joint torques during ten daily activities were predicted by long short-term memory (LSTM) neural networks and transfer learning. LSTM models were trained with muscle electromyography signals and lower limb joint angles. Hip flexion/extension, hip abduction/adduction, knee flexion/extension and ankle dorsiflexion/plantarflexion torques were predicted. The LSTM models’ performance in predicting torque was investigated in both intra-subject and inter-subject scenarios. Each scenario was further divided into intra-task and inter-task tests. We observed that LSTM models could predict lower limb joint torques during various activities accurately with relatively low error (root mean square error 0.14 Nm/kg, normalized root mean square error ≤ 8.7%) either through a uniform model or through ten separate models in intra-subjecttests. Furthermore, a transfer learning technique was adopted in the inter-task and inter-subject tests to further improve the generalizability of LSTM models by pre-training a model on multiple subjects and/or tasks and transferring the learned knowledge to a target task/subject. Particularly in the inter-subject tests, we could predict joint torques accurately in several movements after training from only a few movements from new subjects(来源05)

in falls among the elderly. Traditional clinical scales such                     degrees of fall risk based on the BBS scores. We seg-(来源06)

长期围绕**基于生物力学的智能可穿戴技术** 展开研究，包括精确解码型运动意图预测、个性化肌肉肌腱参数的可靠估计和多场景安全型人机交互(来源07)

搭建了神经肌肉骨骼和机器学习混合模型以实时预测运动意图，开创了预测模型的新方向；开发了全面的可靠的算法来量化肌肉肌腱变化，可用于制定个性化治疗计划；设计了数字孪生人机交互系统可模拟不同场景下的交互效果；研发了基于神经肌骨的自适应控制算法来提高交互安全性(来源07)

在康复机器人、生物力学建模与控制领域发表高水平论文52篇，获授权专利4项，2次获机器人领域A类会议最佳论文提名，5次获国际学术奖项(来源07)

Thanks: This work was supported by the National Natural Science Foundation of China under Grant 62503165, Hunan Provincial Research and Development Project under Grant 2026QK3021, Yuelushan Center for Industrial Innovation, and Health Commission of Hunan Province. The authors also acknowledge the support from Nanyang Technological University, the Agency for Science Technology and Research (A*STAR), and the National Healthcare Group (NHG). (Corresponding author: ZhiZhang Li.)(来源12)

We are a community of more than 103,000 authors and editors from 3,291 institutions spanning 160 countries, including Nobel Prize winners and some of the world’s most-cited researchers. Publishing on IntechOpen allows authors to earn citations and find new collaborators, meaning more people see your work not only from your own field of study, but from other related fields too(来源14)

We use cookies to personalise content and ads, to provide social media features and to analyse our traffic. We also share information about your use of our site with our social media, advertising and analytics partners who may combine it with other information that you’ve provided to them or that they’ve collected from your use of their services(来源15)

You are accessing a machine-readable page. In order to be human-readable, please install an RSS reader(来源16)

Handmaid: design and evaluation of a hand exoskeleton with active backdrive for neuromuscular rehabilitation(来源17)

Department of Mechanical Engineering, Faculty of Engineering, Chulalongkorn University, Bangkok 10330, Thailand(来源18)

deep learning, gait analysis, gait mat, markerless motion capture, spatiotemporal parameters(来源19)

To keep the service fast and reliable for everyone, this page will reload automatically in 30 seconds(来源20)

**Performance Management for Parametric Building Design Generated by a Genetic Algorithm**(来源21)

Marker-based motion capture systems provide highly accurate biomechanical measurements, but they are expensive, require reflective marker placement, and are typically limited to controlled laboratory environments. Their reliance on extensive post-processing also restricts real-time clinical applicability. Markerless pose estimation systems offer a more accessible alternative; however, their ability to reliably quantify clinically relevant dynamic gait stability metrics, such as anterior–posterior (AP) and mediolateral (ML) margin of stability (MoS), remains insufficiently validated. This study evaluated a cost-effective, multi-camera, three-dimensional, markerless motion-capture system based on MediaPipe to estimate gait stability. Ten participants walked on a treadmill at a constant speed of 1.1 m/s while data were collected simultaneously using a three-camera synchronized RGB setup and a gold-standard Vicon capture system. Because MoS relies heavily on accurate estimates of center of mass (CoM) motion and joint kinematics, the temporal trajectories of 14 anatomical landmarks were first compared across systems. The markerless system achieved a mean 3D Euclidean RMSE of 2.48 ± 0.69 cm, with median and 95th-percentile errors of 1.98 ± 0.51 cm and 4.09 ± 1.40 cm, respectively. Using a segmental CoM model, AP MoS values were − 19.6 ± 0.46 cm (marker-based) and − 18.6 ± 0.30 cm (markerless), while ML MoS values were 4.2 ± 0.1 cm and 3.9 ± 0.12 cm, respectively. Errors were larger in the AP direction (RMSE = 3.8 cm) than in the ML direction (RMSE = 0.83 cm). Similar trends were observed using pelvic CoM approximation. Paired _t_ -tests showed no statistically significant differences between the two systems (_p >_ 0.05). These findings suggest that the proposed multi-camera markerless system can reliably estimate whole-body kinematics and gait stability metrics, support scalable clinical assessment of walking stability and fall risk beyond specialized laboratory environments(来源22)

Relative reliability of raw ensemble-averaged sagittal joint angle trajectories by side after 0° normalization(来源23)

We are a community of more than 103,000 authors and editors from 3,291 institutions spanning 160 countries, including Nobel Prize winners and some of the world’s most-cited researchers. Publishing on IntechOpen allows authors to earn citations and find new collaborators, meaning more people see your work not only from your own field of study, but from other related fields too(来源24)

Comparison of Fugl-Meyer score of lower limb motor function scores before and after intervention in the two groups (− _x_ ±  _s_)(来源25)

School of Automation Science and Engineering, South China University of Technology, Guangzhou 510641, China(来源26)

We are a community of more than 103,000 authors and editors from 3,291 institutions spanning 160 countries, including Nobel Prize winners and some of the world’s most-cited researchers. Publishing on IntechOpen allows authors to earn citations and find new collaborators, meaning more people see your work not only from your own field of study, but from other related fields too(来源27)

This is a pre-print of an article published in [Biomechanics and Modeling in Mechanobiology (2020) ]. The final authenticated version is available online at: [10.1007/s10237-020-01398-1]. Computational modelling is an invaluable tool for investigating features of human locomotion and motor control which cannot be measured except through invasive techniques. Recent research has focused on creating personalised musculoskeletal models using population-based morphing or directly from medical imaging. Although progress has been made, robust definition of two critical model parameters remains challenging: (1) complete tibiofemoral (TF) and patellofemoral (PF) joint motions, and (2) muscle tendon unit (MTU) pathways and kinematics (i.e. lengths and moment arms). The aim of this study was to develop an automated framework, using population-based morphing approaches to create personalised musculoskeletal models, consisting of personalised bone geometries, TF and PF joint mechanisms, and MTU pathways and kinematics. Informed from medical imaging, personalised rigid body TF and PF joint mechanisms were created. Using atlas- and optimisation-based methods, personalised MTU pathways and kinematics were created with the aim of preventing MTU penetration into bones and achieving smooth MTU kinematics that follow patterns from existing literature. This framework was integrated into the Musculoskeletal Atlas Project Client software package to create and optimise models for 6 participants with incrementally increasing levels of personalisation with the aim of improving MTU kinematics and pathways. Three comparisons were made: (1) non-optimised (Model 1) and optimised models (Model 3) with generic joint mechanisms; (2) non-optimised (Model 2) and optimised models (Model 4) with personalised joint mechanisms; and (3) both optimised models (Model 3 and 4). Following optimisation, improvements were consistently shown in pattern similarity to cadaveric data in comparison (1) and (2). For comparison (3), a number of comparisons showed no significant difference between the two compared models. Importantly, optimisation did not produce statistically significantly worse results in any case(来源28)

Computational modelling is a tool that provide accurate readings and simulation by using a combination of trusted and tested mathematical models of materials and system dynamics. Existing study in gait analysis mostly focus on extrinsic parameter such as kinetics and kinematics measures. The analysis of gait especially for disease such as knee osteoarthritis (OA) can provides insight into muscle properties, activation pattern and potential impact on the knee stability and function. This study aims to simulate and compare muscle activation properties in normal individual and individuals with knee osteoarthritis (OA) during gait phases. This study used musculoskeletal modeling software, OpenSim, to analyze the total fiber force and moment of several key muscles (such as medial gastrocnemius, vastus lateralis, and rectus femoris) relative to knee flexion moments. The motion analysis data of three subjects (one normal subject and two knee OA patients) were used in this study. The results shows that total muscle fiber force and muscle moment for medial gastrocnemius and vastus lateralis muscle were higher among individual with knee osteoarthritis. Meanwhile, rectus femoris muscle behaved differently where both parameters were smaller among knee OA patient. These simulated findings suggest that muscle mechanism changed due to knee osteoarthritis disease. The muscles force and moment were observed in this study gave some insight on the devastating effect of the disease on the lower limb joint muscle. Thus, further study is still warranted to investigate the risk of muscle injury and alteration of gait on individual with knee osteoarthritis. It is highly recommended to repeat this study with larger number of subjects for a better data interpretation(来源30)

Sometimes you may be asked to solve the CAPTCHA if you are using advanced terms that robots are known to use, or sending requests very quickly(来源31)

Abstract—Transformers have been used for 3D human                              [2] and person re-identification [3], [4], [5], this is a longstand-(来源32)

Abstract— In this study, we investigate muscle activation                      control, with the CoP frequently acting as a key indicator(来源33)

Abstract— This study presents the Single-Sided Mobility                         reliance on position control make them unsuitable for stroke(来源34)

Abstract— In recent decades, there has been an increasing                   predicting unseen movements. Detailed comparative studies of(来源35)

Keywords:                                                 Flexible flatfoot in childhood usually undergoes a spontaneous return to normal. However, during this process, it(来源36)

ESMAC 2023 Abstract Book                                                                                                       Gait & Posture 106 (2023) 1–231(来源37)

gies that incorporate the user’s torque capacity is rapidly                     perform specific tasks and movements, and has demonstrated(来源38)

## 参考资料
- 来源01：EMG-Constrained and Ultrasound-Informed Muscle-Tendon Parameter Estimation in Post-Stroke Hemiparesis — https://doi.org/10.1109/tbme.2024.3352556
- 来源02：Estimation of Joint Torque by EMG-Driven Neuromusculoskeletal Models and LSTM Networks — https://doi.org/10.1109/tnsre.2023.3315373
- 来源03：Neuromusculoskeletal Model-Informed Machine Learning-Based Control of a Knee Exoskeleton With Uncertainties Quantification — https://doi.org/10.3389/fnins.2023.1254088
- 来源04：Ankle Joint Torque Prediction Using an NMS Solver Informed-ANN Model and Transfer Learning — https://doi.org/10.1109/jbhi.2022.3207313
- 来源05：Lower-Limb Joint Torque Prediction Using LSTM Neural Networks and Transfer Learning — https://doi.org/10.1109/tnsre.2022.3156786
- 来源06：Towards Clinical Application of Enhanced Timed Up and Go With Markerless Motion Capture and Machine Learning for Balance and Gait Assessment — https://doi.org/10.1109/jbhi.2025.3543095
- 来源07：07-robotics.hnu.edu.cn-a344c9 — https://robotics.hnu.edu.cn/info/1176/3002.htm
- 来源08：08-grzy.hnu.edu.cn-da34b8 — https://grzy.hnu.edu.cn/site/index/zhanglongbin
- 来源09：09-robotics.hnu.edu.cn-2842fc — https://robotics.hnu.edu.cn/szdw/jsjj2.htm
- 来源10：11-api.openalex.org-ef8cd0 — https://api.openalex.org/authors/A5002794703
- 来源11：12-api.openalex.org-fc55b3 — https://api.openalex.org/works?filter=author.id:A5002794703&per-page=50&sort=publication_date:desc
- 来源12：15-arxiv.org-d1dfa3 — https://arxiv.org/abs/2503.00794
- 来源13：18-loop.frontiersin.org-273865 — https://loop.frontiersin.org/people/1121302/overview
- 来源14：Upper-Limb Robotic Rehabilitation Exoskeleton: Tremor Suppression — https://doi.org/10.5772/5175
- 来源15：Lower-Limb Wearable Exoskeleton — https://doi.org/10.5772/5176
- 来源16：Lower Limb Exoskeleton for Rehabilitation with Improved Postural Equilibrium — https://doi.org/10.3390/robotics7020028
- 来源17：Handmaid: design and evaluation of a hand exoskeleton with active backdrive for neuromuscular rehabilitation — https://doi.org/10.1007/s11370-026-00697-3
- 来源18：A Refined 2D Lagrangian-Based Model for Joint Torque Estimation in Lower-Limb Exoskeleton Applications — https://doi.org/10.3390/math14132400
- 来源19：Assessment of spatiotemporal gait parameters using a deep learning algorithm-based markerless motion capture system — https://doi.org/10.31224/osf.io/j4rbg
- 来源20：A Low-Cost Markerless motion capture system to automate Functional Gait Assessment: Feasibility Study — https://doi.org/10.1101/2024.12.17.628985
- 来源21：A Review of Human Pose Estimation Methods in Markerless Motion Capture — https://doi.org/10.14733/cadaps.2024.392-423
- 来源22：Validated Multi-Camera Markerless Motion Capture with Real-Time Data Analytics for Gait Stability Assessment — https://doi.org/10.21203/rs.3.rs-9347103/v1
- 来源23：Gait assessment using a 2D video-based pose estimation app in comparison to a markerless motion capture system in subjects with osteoarthritis of the knee – a pilot study — https://doi.org/10.3389/fbioe.2026.1869878
- 来源24：Neural Networks to Control Voluntary Movements in a Lower Limb Exoskeleton Using Electromyographic Signals — https://doi.org/10.5772/intechopen.115174
- 来源25：Effects of motor imagery-based brain-computer interface-controlled electrical stimulation on lower limb function in hemiplegic patients in the acute phase of stroke: a randomized controlled study — https://doi.org/10.3389/fneur.2024.1394424
- 来源26：A Decoding Method Using Riemannian Local Linear Feature Construction for a Lower-Limb Motor Imagery Brain–Computer Interface System — https://doi.org/10.3390/electronics12224697
- 来源27：Clinical Implications of Muscle-Tendon &amp; -Force Interplay: Surface Electromyography Recordings of m. vastus lateralis in Renal Failure Patients Undergoing Dialysis and of m. gastrocnemius in Individuals with Achilles Tendon Damage — https://doi.org/10.5772/27638
- 来源28：Automated creation and tuning of personalised muscle paths for OpenSim musculoskeletal models via the Musculoskeletal Atlas Project Client — https://doi.org/10.31224/osf.io/hy36f
- 来源29：Tibiofemoral Joint Contact Force Estimation Based on OpenSim Musculoskeletal Modeling — https://doi.org/10.3724/sp.j.1329.2025.01015
- 来源30：Simulation of Lower Limb Muscle Moment and Total Muscle Fiber Force Using OpenSim for Knee Osteoarthritis — https://doi.org/10.11113/jmeditec.v1n1.31
- 来源31：Longbin Zhang - Google Scholar — https://scholar.google.com/citations?user=zfHrGd0AAAAJ&hl=en
- 来源32：Global and Local Spatio-Temporal Encoder for 3D Human Pose Estimation — https://doi.org/10.1109/tmm.2023.3321438
- 来源33：Muscle Activation and Postural Sway in Response to Task Complexity: A Study of Balance Control in Older Adults — https://doi.org/10.1109/icorr66766.2025.11063123
- 来源34：Design and Evaluation of a Single-Sided Mobility Assistive Exoskeleton (SMAEXO) for Hemiplegia — https://doi.org/10.1109/icorr66766.2025.11063171
- 来源35：Ankle Joint Torque Estimation Using an EMG-Driven Neuromusculoskeletal Model and an Artificial Neural Network Model — https://doi.org/10.1109/tase.2020.3033664
- 来源36：Do gait patterns normalize concurrently with the recovery of foot arches in children with flatfoot? A prospective cohort study — https://doi.org/10.1016/j.jbiomech.2025.112758
- 来源37：An approach to establishing the thresholds of plantar loading in obese children — https://doi.org/10.1016/j.gaitpost.2023.07.273
- 来源38：Knee Joint Torque Prediction with Uncertainties by a Neuromusculoskeletal Solver-informed Gaussian Process Model — https://doi.org/10.1109/icarm58088.2023.10218934
