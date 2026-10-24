# Trajectory Deformation-Based Multi-Modal Adaptive Compliance Control for a Wearable Lower Limb Rehabilitation Robot

Jie Zhou , Huanfeng Peng , Manxu Zheng , Zhe Wei , Tao Fan , and Rong Song , Senior Member, IEEE

Abstract— Adaptive compliance control is critical for rehabilitation robots to cope with the varying rehabilitation needs and enhance training safety. This article presents a trajectory deformation-based multi-modal adaptive compliance control strategy (TD-MACCS) for a wearable lower limb rehabilitation robot (WLLRR), which includes a high-level trajectory planner and a low-level position controller. Dynamic motion primitives (DMPs) and a trajectory deformation algorithm (TDA) are integrated into the high-level trajectory planner, generating multi-joint synchronized desired trajectories through physical human-robot interaction (pHRI). In particular, the amplitude modulation factor of DMPs and the deformation factor of TDA are adapted by a multi-modal adaptive regulator, achieving smooth switching of human-dominant mode, robot-dominant mode, and soft-stop mode. Besides, a linear active disturbance rejection controller is designed as the low-level position controller. Four healthy participants and two stroke survivors are recruited to conduct robot-assisted walking experiments using the TD-MACCS. The results show that the TD-MACCS can smoothly switch three control modes while guaranteeing trajectory tracking accuracy. Moreover, we find that appropriately increasing the upper bound of the deformation factor can enhance the average walking speed (AWS) and root mean square of trajectory deviation (RMSTD).

Index Terms— Wearable lower limb rehabilitation robot, physical human-robot interaction, trajectory deformation algorithm, dynamic motion primitives, linear active disturbance rejection control.

## I. INTRODUCTION

TROKE is a significant global health issue, resulting in S approximately 5.5 million deaths annually, and the number of stroke survivors is projected to reach 77 million globally by 2030 [1]. The substantial disability rate associated with stroke, exemplified by a one-year disability rate of 44.78% in China, often leads to lower extremity motor and functional injuries among survivors [2]. Driven by the need to accommodate the large patient population, wearable lower limb rehabilitation robots (WLLRRs), such as H2 [3], HAL [4], Ekso [5], and BEAR-H1 [6], have emerged as promising interventions for assisting the stroke survivors with lower extremity motor dysfunction in achieving gait training. Furthermore, they have demonstrated positive effects in facilitating the recovery of motor function [6], [7], [8].

Various control strategies have been proposed to enable rehabilitation robots to cope with patients with different levels of motor abilities. For patients with weak residual muscle strength, the passive control strategy is commonly employed, delivering long-endurance and high-intensity therapeutic training along predefined trajectories [9]. Many passive control strategies, such as the linear active disturbance rejection controller (LADRC) [10] and the adaptive network-based fuzzy logic controller [11], have been successfully implemented in lower limb rehabilitation robots to ensure accurate trajectory tracking. For patients with high residual muscle strength, the active control strategy has garnered considerable attention as it takes into account the patients’ active contribution, which can promote neural plasticity [12]. An impedance controller has been utilized in a lower limb rehabilitation robot (LLRR) to provide predefined assistance within an acceptable tracking error range [13]. Admittance controllers have been applied in

WLLRR to deviate the robot-imposed motions from the predefined trajectory, reducing gait trajectory mismatch and offering more comfortable walking assistance [14], [15]. A humanrobot cooperation controller based on a trajectory deformation algorithm (TDA) has been proposed and applied in a LLRR, enabling the prediction of the robot’s future trajectory through the physical human-robot interaction (pHRI) [16]. A force field controller composed of normal, tangential, damping, and feedforward compensation force has been proposed for an active leg exoskeleton [17], allowing the patients to move freely around a preset reference path. Additionally, the velocity-fieldbased controller has been proposed for a fully back-drivable WLLRR, which can provide motion guidance and assistance during over-ground gait training [18]. Although these active control strategies consider the patients’ active contribution, they overlook the significant variability in disability levels among patients and even within an individual patient during rehabilitation training.

Many research studies have focused on control strategies that adaptively switch between multiple modes of rehabilitation training. Li et al. developed a human-in-the-loop control strategy for a WLLRR based on a specific potential energy function [19]. This strategy effectively balances the capabil ities of the user and exoskeleton by transitioning between the human-voluntary and robot-constrained regions. Further more, given the close coupling between rehabilitation robots and patients, ensuring safety during rehabilitation training is paramount. To address this concern, Zhang and Cheah [20] proposed a human-robot interaction control strategy. Based on real-time position error, this strategy can adaptively switch between human-dominant mode, robot-dominant mode, and safety-stop mode. Unfortunately, the impulsive contact force or the low robot compliance will make it difficult for the patients to generate enough error signals to activate the safety-stop mode. To overcome this limitation, a multi-modal control strategy was proposed and verified on the upper and lower limb rehabilitation robots [21]. This strategy adaptively switches between human-dominant mode and robot-dominant mode based on tracking errors, and can easily activate the safety-stop mode using interaction forces. Although error signals are essential in motion learning [22], they cannot be perceived by the patients promptly during robot-assisted training without visual feedback. Additionally, the tracking error is affected by both the subjects’ motor ability and the controller’s performance. Therefore, Xu et al. proposed a multi-modal adaptive control strategy for a sitting lower limb rehabilitation robot [23]. This strategy enables smooth switching between the robot- and human-active modes by adjusting a weight factor based on interaction torque obtained using an EMG-driven impedance model. Although these multi-modal adaptive control strategies can match the varying motor abilities of patients, achieving compliant pHRI and adjusting robot compliance, particularly in robot-assisted walking, remains a significant challenge. To the best of our knowledge, how to design a multi-modal adaptive compliance control strategy for WLLRR, which can achieve smooth switching of humandominant mode, robot-dominant mode, and soft-stop mode through compliant pHRI, has not been investigated.

![](images/b6bfc141d6b3b02e6836329c35c39d71f3016928710cbc69b9a56cf24a6de54e.jpg)  
Fig. 1. System of the SYSU-REHAB1-H: (a) The self-designed WLLRR with a movable bracket; (b) The mechanical structure of the hip actuator module; (c) The exoskeleton length adjustment and the knee joint motion range limitation mechanical structure; (d) One stroke patient walking over-ground assisted by the SYSU-REHAB1-H under the guidance of the operator.

This article proposed a trajectory deformation-based multimodal adaptive compliance control strategy (TD-MACCS) for a self-designed WLLRR. Besides, robot-assisted walking experiments were performed on four healthy participants and two stroke survivors, and the results were analyzed quantitatively. The main contributions can be summarized as follows:

1) We have integrated the dynamic motion primitives (DMPs) [24] and TDA into the trajectory planner, making learning and adjusting rehabilitation task trajectory more convenient and realizing compliant pHRI. Besides, we have proposed a multi-modal adaptive regulator based on the amplitude modulation factor of DMPs and the deformation factor of TDA. It can enable the WLLRR to match the varying motor abilities of users by smoothly switching different control modes. Additionally, the LADRC is designed to ensure trajectory tracking accuracy under internal and external disturbances.

2) We have preliminarily verified the TD-MACCS on four healthy participants and two stroke survivors. The results demonstrate that TD-MACCS can adaptively and smoothly switch between human-dominant, robot-dominant, and soft-stop modes while maintaining high trajectory tracking accuracy. Furthermore, we find that appropriately increasing the upper bound of the deformation factor in TD-MACCS can enhance the average walking speed (AWS) and root mean square of trajectory deviation (RMSTD).

## II. DESIGN OF WLLRR SYSTEM

## A. Mechanical Structure of WLLRR

Intending to provide over-ground gait training for patients, we have developed a wearable lower limb rehabilitation robot, SYSU-REHAB1-H, based on the robot described in [25]. As illustrated in Fig. 1 (a), to constitute a simple and applicable system, each leg is driven with two active DoF (degrees of freedom) at the hip and knee joints (flexion/extension). The waist, thigh, calf, and foot support with Velcro straps are designed to fasten the WLLRR to the different parts of the patient’s body. The limit block in each actuator module is designed to avoid joint movement beyond the normal range, as illustrated in Fig. 1 (b) and (c). Besides, the thigh and calf exoskeletons are made adjustable by rotating the knobs to align the user and SYSU-REHAB1 joints as much as possible. From Fig. 1 (d), due to patients’ poor balance maintenance ability, the hoist with a walking aid sling is adopted to prevent the users from falling during robot-assisted walking.

![](images/9dc3e400dead2146d8bbb660708855a57aa68d6228d62b2a4d6a1663051e5e54.jpg)  
Fig. 2. The framework of the control strategy for SYSU-REHAB1-H.

## B. Driving System ofWLLRR

The driving system comprises the joint actuator modules, the data acquisition/output module, and the compact DAQ controller. A 220 W flat motor (EC 90 flat, Maxon, Sachseln, Switzerland), a harmonic reducer (LCD-20-100-C-I, Leader, Jiangsu, China), and a torque sensor (M2212A, SRI, Guangxi, China) are selected to constitute the hip joint actuator module, as illustrated in Fig. 1 (b). Besides, a 100 W flat motor (EC 60 flat, Maxon, Sachseln, Switzerland), a harmonic reducer (LCD-17-100-C-I-ST, Leader, Jiangsu, China), and a torque sensor (M2210N4, SRI, Guangxi, China) are applied to constitute the knee joint actuator module. Different motors are controlled by the same motor driver (Module 50/5, Maxon, Sachseln, Switzerland). The angular position of each active joint is sensed using an optical encoder (MILE 1024, Maxon, Sachseln, Switzerland) located inside the motors. The data acquisition/output modules (NI-9205, NI-9361, NI-9403, NI-9401, National Instruments, Texas, USA) and the compact DAQ controller (cDAQ-9136, National Instruments, Texas, USA) are adopted to collect, generate, and output signals to control the actuator module of each joint through the LabVIEW 2018 software.

## III. DESIGN OF CONTROL STRATEGY

As illustrated in Fig. 2, the TD-MACCS, including a high-level trajectory planner and a low-level position controller, is proposed for the SYSU-REHAB1-H. The high-level trajectory planner consists of a trajectory generator, a desired trajectory predictor, an interaction torque estimator, and a multi-modal adaptive regulator. The LADRC is adopted as the low-level position controller to ensure each joint of the SYSU-REHAB1-H tracks the desired trajectory under internal and external disturbances. Each part of the TD-MACCS will be comprehensively discussed in the following subsections.

It is worth noting that to avoid confusion between the reference trajectory and the desired trajectory, we provide the following explanation: 1) The reference trajectory for each active joint is generated by the trajectory generator without activating the desired trajectory predictor; 2) The desired trajectory for each active joint is generated by the trajectory generator and the desired trajectory predictor, which can be adjusted in real-time through pHRI.

## A. Trajectory Generator

Different nonlinear methods were adopted as trajectory generators for WLLRRs, including DMPs [26] and central pattern generators [27]. Since DMPs have an excellent ability for motion learning and generalizing, especially in multi-joint robot control scenarios, they are suitable for encoding and adjusting the periodic gait trajectory in real time [24]. Thus, the trajectory generator is designed based on DMPs, and it can be described as follows:

$$
\left\{ \begin{array}{l} \kappa \dot {z} _ {\mathrm{q}} = \alpha_ {\mathrm{q}} (\beta_ {\mathrm{q}} (\pmb {g} _ {\mathrm{q}} - \pmb {q} _ {\mathrm{r}}) - z _ {\mathrm{q}}) + \pmb {f} _ {\mathrm{q}} \\ \kappa \dot {\pmb {q}} _ {\mathrm{r}} = z _ {\mathrm{q}} \end{array} \right.
$$

$$
\kappa \dot {\phi} = 1,\tag{1}
$$

$$
\kappa \dot {r} = \alpha_ {\mathrm{r}} (r _ {0} - r),\tag{2}
$$

(3)

$$
\boldsymbol {f} _ {\mathrm{q}} = \frac {\sum_ {i = 1} ^ {m} \psi_ {i} (\phi) \boldsymbol {w} _ {\mathrm{q} , i}}{\sum_ {i = 1} ^ {m} \psi_ {i} (\phi)} r,\tag{4}
$$

$$
\psi_ {i} (\phi) = \exp (\frac {\cos (\phi - c _ {i}) - 1}{2 \sigma_ {i} ^ {2}}),\tag{5}
$$

where $\kappa$ is the positive temporal scaling factor; ${ \pmb g } _ { \mathbf { q } } \in \mathbb { R } ^ { 4 }$ is the position goal vector; $\alpha _ { \mathrm { q } }$ and $\beta _ { \mathfrak { q } }$ are positive constants, and the critical damping configuration $\alpha _ { \mathrm { q } } = 4 \beta _ { \mathrm { q } }$ can ensure the convergence and stability of DMPs; $f _ { \mathrm { ~ q ~ } } \in \mathrm { ~ \mathbb { R } ^ { 4 } ~ }$ is the forcing term vector obtained by supervised learning; we adopt the locally weighted regression method [24] in this article; $q _ { \mathrm { r } } \in \mathbb { R } ^ { 4 }$ and $\dot { \pmb q } _ { \mathrm { r } } \in \mathbb { R } ^ { 4 }$ are the reference angle and angular velocity vector; $z _ { \mathrm { q } } ~ \in ~ \mathbb { R } ^ { 4 }$ and $\dot { z } _ { \mathrm { q } } ~ \in ~ \mathbb { R } ^ { 4 }$ are the reference angular velocity and angular acceleration after expansion or contraction; r and $r _ { 0 }$ are the state variables of the amplitude modulation factor and the desired amplitude modulation factor respectively; $\alpha _ { r }$ is a positive constant that determines the change rate of amplitude factor; $\phi$ is the phase variable; ψ<sub>i</sub> (φ) is the $i _ { \mathrm { t h } }$ kernel function; $\sigma _ { i }$ and $c _ { i }$ are constants that determine the width and center of the $i _ { \mathrm { t h } }$ kernel function; m is the number of kernel functions; ${ \pmb w } _ { { \sf q } , i } \in \mathbb { R } ^ { 4 }$ is the weight coefficient vector corresponding to the kernel function $\psi _ { i } ( \phi )$ .

## B. Interaction Torque Estimator

The torque sensor mounted between the harmonic reducer and exoskeleton can be used to measure the interaction torque [13]. Unfortunately, it is not easy to accurately extract the interaction torque because of the inertial torque, gravity torque, and the torque caused by Coriolis and centripetal [28]. According to our previous research, an interaction torque estimator is adopted to estimate the interaction torque without using the dynamic model [25]. The interaction torque estimator based on DMPs can be described as follows:

$$
\left\{ \begin{array}{l} \kappa \dot {\boldsymbol {z}} _ {\tau} = \alpha_ {\tau} (\beta_ {\tau} (\boldsymbol {g} _ {\tau} - \boldsymbol {\tau} _ {\text { nts }}) - \boldsymbol {z} _ {\tau}) + \boldsymbol {f} _ {\tau} \\ \kappa \dot {\boldsymbol {\tau}} _ {\text { nts }} = \boldsymbol {z} _ {\tau} \end{array} \right.\tag{6}
$$

$$
\boldsymbol {f} _ {\tau} = \frac {\sum_ {i = 1} ^ {m} \psi_ {i} (\phi) \boldsymbol {w} _ {\tau , i}}{\sum_ {i = 1} ^ {m} \psi_ {i} (\phi)} r,\tag{7}
$$

$$
\hat {\pmb {\tau}} _ {\mathrm{I}} = \pmb {\tau} _ {\mathrm{Rmtw}} - \pmb {\tau} _ {\mathrm{nts}},\tag{8}
$$

where $\tau _ { \mathrm { n t s } } \in \mathbb { R } ^ { 4 }$ and $\dot { \tau } _ { \mathrm { n t s } } \in \mathbb { R } ^ { 4 }$ are the normalized joint torque vector and its first derivative; $z _ { \tau } \in \mathbb { R } ^ { 4 }$ and $\dot { z } _ { \tau } \in \mathbb { R } ^ { 4 }$ are the first and second derivative of ${ \pmb \tau } _ { \mathrm { n t s } }$ after expansion or contraction; $\alpha _ { \tau }$ and $\beta _ { \tau }$ are positive constants in the interaction torque estimator, and the critical damping configuration $\alpha _ { \tau } = 4 \beta _ { \tau }$ can ensure the convergence and stability of DMPs; ${ \pmb g } _ { \tau } \in \mathbb { R } ^ { 4 }$ is the torque goal vector; $f _ { \tau } \in \mathbb { R } ^ { 4 }$ is the forcing term vector in interaction torque estimator; $\pmb { w } _ { \tau , i } ~ \in ~ \mathbb { R } ^ { 4 }$ is the weight coefficient vector corresponding to the $i _ { \mathrm { t h } }$ kernel function in interaction torque estimator; $\tau _ { \mathrm { j t w } } \in \mathbb { R } ^ { 4 }$ and $\hat { \pmb { \tau } } _ { \mathrm { I } } \in \mathbb { R } ^ { 4 }$ are the measured torque vector and interaction torque vector during robot-assisted walking.

To avoid frequent adjustment of the reference trajectory caused by small interaction torque under high robot compliance [27], [29], it is necessary to design the interaction torque truncation function as follows:

$$
\hat {\tau} _ {\mathrm{IA}, k} = \left\{ \begin{array}{l l} \hat {\tau} _ {\mathrm{I}, k}, & \left| \hat {\tau} _ {\mathrm{I}, k} \right| > \bar {\tau} _ {k} \\ 0, & \left| \hat {\tau} _ {\mathrm{I}, k} \right| \leq \bar {\tau} _ {k} \end{array} \right.,\tag{9}
$$

where $k = 1 , 2 , 3 ,$ 4 represent different active joints, $k = 1$ and $k \ = \ 3$ represent the left and right hip joints, $k \ = \ 2$ and $k = 4$ represent the left and right knee joints; τˆ<sub>I,k</sub> represents the interaction torque of the $k _ { \mathrm { t h } }$ active joint; $\hat { \tau } _ { \mathrm { I A } , k }$ represents the interaction torque of the $k _ { \mathrm { t h } }$ active joint after truncation processing; $\bar { \tau } _ { k }$ represents the interaction torque threshold of the $k _ { \mathrm { t h } }$ active joint, which is the minimum interaction torque required for activating the desired trajectory predictor; Usually, the interaction torque threshold of the hip joint is greater than that of the knee joint, namely, $\bar { \tau } _ { 1 } > \bar { \tau } _ { 2 } , \bar { \tau } _ { 3 } > \bar { \tau } _ { 4 } , \bar { \tau } _ { 1 } > \bar { \tau } _ { 4 } , \bar { \tau } _ { 3 } >$ $\bar { \tau } _ { 2 } ,$ and their value can be adjusted by physical therapists according to actual needs [29].

## C. Desired Trajectory Predictor

Physical interaction affects not only the current state of the WLLRR but also its future behavior [30]. A trajectory planner should be able to generate smooth current and future trajectories continuously. TDA can smoothly generate the current desired position and predict the future desired trajectory based on the interaction torque [16]. Thus, the TDA is adopted as the desired trajectory predictor.

According to our previous work [16], we define $t _ { \mathrm { s } }$ and $t _ { \mathrm { f } }$ as the start and end times of the trajectory deformation at the current pHRI. $\pmb q _ { r } ( t ) \in \mathbb { R } ^ { 4 }$ and $\pmb q _ { d } ( t ) \in \mathbb { R } ^ { 4 }$ are defined as the reference and desired trajectory, respectively. $\pmb { u } \in \mathbb { R } ^ { 4 \times 4 }$ is defined as the diagonal deformation factor matrix. $\Gamma ( \pmb { u } , t )$ is defined as the deformation curve function, which can yield different smooth deformation curves over the time interval $t \in [ t _ { s } , t _ { f } ]$ by adjusting the deformation factor. The desired trajectory in the current iteration and the previous iteration are defined as follows:

$$
\boldsymbol {q} _ {d} (t) = \boldsymbol {\Gamma} (\boldsymbol {u}, t), t \in [ t _ {s}, t _ {f} ],\tag{10}
$$

$$
\begin{array}{c} \boldsymbol {\Gamma} (\boldsymbol {0}, t) = [ \boldsymbol {q} _ {d} (t _ {s}), \boldsymbol {q} _ {d} (t _ {s} + \delta), \\ \dots , \boldsymbol {q} _ {d} (t _ {s} + (N - 2) \delta), \boldsymbol {q} _ {r} (t _ {f}) ], \end{array}\tag{11}
$$

$$
N = \frac {p}{\delta} + 1,\tag{12}
$$

where $p$ is the prediction time; δ is the sample period; N is the number of waypoints.

The iterative equation of TDA for the current trajectory deformation during pHRI can be described as follows:

$$
\left\{ \begin{array}{l} \boldsymbol {\Gamma} (\boldsymbol {u}, t) = \boldsymbol {\Gamma} (\boldsymbol {0}, t) + \boldsymbol {V} (t) \boldsymbol {u}, t \in [ t _ {\mathrm{s}}, t _ {\mathrm{f}} ] \\ \boldsymbol {V} (t) = \delta \boldsymbol {H} \hat {\boldsymbol {\tau}} _ {\mathrm{IA}} ^ {T} (t _ {\mathrm{s}}) \\ \boldsymbol {H} = \frac {\boldsymbol {W} \boldsymbol {\beta}}{(p + \delta) \| \boldsymbol {W} \|} \\ \boldsymbol {W} = (\boldsymbol {I} - (\boldsymbol {Z} ^ {\mathrm{T}} \boldsymbol {Z}) ^ {- 1} \boldsymbol {C} ^ {\mathrm{T}} (\boldsymbol {C} (\boldsymbol {Z} ^ {\mathrm{T}} \boldsymbol {Z}) ^ {- 1} \boldsymbol {C} ^ {\mathrm{T}}) ^ {- 1} \boldsymbol {C}) (\boldsymbol {Z} ^ {\mathrm{T}} \boldsymbol {Z}) ^ {- 1}, \\ \boldsymbol {I} \in \mathbb {R} ^ {N \times N}, \end{array} \right.\tag{13}
$$

(14)

where $\hat { \tau } _ { \mathrm { I A } } ( t _ { \mathrm { s } } ) = [ \hat { \tau } _ { \mathrm { I A } , 1 } ( t _ { \mathrm { s } } ) , \hat { \tau } _ { \mathrm { I A } , 2 } ( t _ { \mathrm { s } } ) , \hat { \tau } _ { \mathrm { I A } , 3 } ( t _ { \mathrm { s } } ) , \hat { \tau } _ { \mathrm { I A } , 4 } ( t _ { \mathrm { s } } ) ]$ is the interaction torque vector after truncation processing at time t<sub>s</sub>; $\pmb { \beta } \in \mathbb { R } ^ { N }$ is the prediction vector of the interaction torque; $\pmb { Z } \in \mathbb { R } ^ { ( N + 3 ) \times N }$ is a finite differencing matrix; $\boldsymbol { c } _ { } \in \mathbb { R } ^ { 4 \times \hat { N } }$ is a constraint matrix; $\pmb { I } \in \mathbb { R } ^ { N \times N }$ is an identity matrix; H and $V ( t )$ are the motor primitive and vector field functions.

Moreover, it is worth mentioning that, although we focus on the current trajectory deformation, the desired trajectory can be continuously generated in response to the interaction torque through a linear iterative method during physical human-robot interaction. Furthermore, the specific derivation process of TDA is given in [16].

## D. Multi-Modal Adaptive Regulator

Since the motor ability of different patients is complex and ever-changing, it is necessary to develop a multi-modal adaptive regulator to increase the robots’ adaptability and training safety. Inspired by [21] and [23], a multi-modal adaptive regulator is proposed according to the interaction torque. The overall idea of the multi-modal adaptive regulator is as follows: 1) the deformation factor can increase smoothly and adaptively within the allowable range when the motor ability is satisfactory, allowing the patients to move with greater degrees of freedom, that is, the human-dominant mode; 2) the deformation factor can decrease adaptively to enforce small or no deviations from the reference trajectory when the motor ability is poor, which can correct the patients abnormal gait trajectory, that is, the robot-dominant mode; 3) the reference trajectory should be smoothly switched to the zero reference position to offer soft-stop protection when the interaction torque is outside the safe interaction torque region, that is, the soft-stop mode. Besides, the reference trajectory can also automatically and smoothly switch back to the original reference trajectory to restore robot-assisted walking after a specified protection time. The multi-modal adaptive regulator based on the above concept can be described as follows:

$$
h (\hat {\tau} _ {I, k}) = \frac {\hat {\tau} _ {I , k} ^ {2}}{\tau_ {n , k} ^ {2}} - 1,\tag{15}
$$

$$
u _ {k} = \left\{ \begin{array}{c} u _ {u, k}, \left| \hat {\tau} _ {I, k} \right| \leq k _ {\tau , k} \tau_ {n, k} \\ u _ {u, k} + (u _ {d, k} - u _ {u, k}) \frac {[ h (\hat {\tau} _ {I , k}) ^ {4} - (k _ {\tau , k} ^ {2} - 1) ^ {4} ] ^ {4}}{(k _ {\tau , k} ^ {2} - 1) ^ {1 6}}, \\ k _ {\tau , k} \tau_ {n, k} <   \left| \hat {\tau} _ {I, k} \right| \leq \tau_ {n, k} \\ u _ {d, k}, \left| \hat {\tau} _ {I, k} \right| > \tau_ {n, k} \end{array} \right.\tag{16}
$$

$$
\hat {r} _ {k} = \left\{ \begin{array}{l} 0, \left| \hat {\tau} _ {I, k} (t _ {a, k}) \right| > \tau_ {s, k} \cup t _ {a, k} <   t \leq t _ {a, k} + L _ {s} \\ r _ {0} ^ {*}, \left| \hat {\tau} _ {I, k} (t) \right| \leq \tau_ {s, k} \cap t > t _ {a, k} + L _ {s} \end{array} \right.\tag{17}
$$

$$
r _ {0} = \min (\min (\hat {r} _ {1}, \hat {r} _ {3}), \min (\hat {r} _ {2}, \hat {r} _ {4})),\tag{18}
$$

where $k \ = \ 1 , 2 , 3 ,$ 4 represent different active joints of WLLRR; $\hat { \tau } _ { \mathrm { n } } = [ \hat { \tau } _ { \mathrm { n , 1 } } , \hat { \tau } _ { \mathrm { n , 2 } } , \hat { \tau } _ { \mathrm { n , 3 } } , \hat { \tau } _ { \mathrm { n , 4 } } ]$ is the nominal interaction torque vector, which is set according to patients’ disability level; We define that motor ability is satisfactory when the interaction torque is less than the nominal interaction torque, otherwise it is poor. $h ( \hat { \tau } _ { \mathrm { I } , k } )$ is the interaction torque region function of the $k _ { \mathrm { t h } }$ joint, which can reflect the changes of interaction torque in a normalized form; $\pmb { k } _ { \tau } = [ k _ { \tau , 1 } , k _ { \tau , 2 } , k _ { \tau , 3 } , k _ { \tau , 4 } ]$ is the constant vector, which determines the change rate of the deformation factor; $\pmb { u } = \mathrm { d i a g } ( u _ { 1 } , u _ { 2 } , u _ { 3 } , u _ { 4 } )$ is the diagonal deformation factor matrix; ${ \pmb u } _ { \mathrm { u } } ~ = ~ \mathrm { d i a g } ( u _ { \mathrm { u } , 1 } , u _ { \mathrm { u } , 2 } , u _ { \mathrm { u } , 3 } , u _ { \mathrm { u } , 4 } )$ and $\pmb { u } _ { \mathrm { d } } = \mathrm { d i a g } ( u _ { \mathrm { d , 1 } } , u _ { \mathrm { d , 2 } } , u _ { \mathrm { d , 3 } } , u _ { \mathrm { d , 4 } } )$ are the upper and lower bound matrices of the deformation factor, respectively; $\tau _ { \mathrm { { S } } } =$ $[ \tau _ { \mathrm { s } , 1 } , \tau _ { \mathrm { s } , 2 } , \tau _ { \mathrm { s } , 3 } , \tau _ { \mathrm { s } , 4 } ]$ is the safe interaction torque vector, which should be set to avoid the safety problem caused by excessive interaction torque; $t _ { \mathrm { a } , k }$ is the time when the interaction torque of the $k _ { \mathrm { t h } }$ joint is larger than the safe interaction torque; $L _ { \mathrm { s } }$ is the specified protection time after the soft stop protection is activated; $\hat { r } _ { k }$ is the amplitude modulation factor of the $k _ { \mathrm { t h } }$ joint; $r _ { 0 } ^ { \ast }$ is the initial value of the amplitude modulation factor for each joint; $r _ { 0 }$ is the desired amplitude modulation factor in the interaction torque estimator and the trajectory generator.

According to (3), (17), and (18), the state variable of the amplitude modulation factor r will converge exponentially to zero when the desired amplitude factor $r _ { 0 }$ is switched to zero. It follows that each component of the forcing term vector $f _ { \mathrm { q } } \in$ $\mathbb { R } ^ { 4 }$ can converge to zero. Thus, the reference trajectory of each active joint will converge to the target position vector ${ \pmb g } _ { \mathfrak { q } } \in \mathbb { R } ^ { 4 }$ synchronously, realizing soft-stop protection during robot-assisted walking.

## E. LADRC

Model-free controllers [10], [31] are widely used in WLLRR because the model parameters of the human-robot coupling model are difficult to obtain accurately by system identification. In recent years, an active disturbance rejection controller has been applied in robots because it can eliminate the necessity of an accurate human-robot coupling model [32]. To simplify the structure of the controller and reduce the number of tuning parameters, the LADRC, i.e., a combination of linear extended state observer and linear state feedback, has been proposed [33]. In this part, the LADRC is designed based on a second-order error auxiliary system to ensure each joint tracks the desired trajectory under internal and external disturbances.

The dynamic model of $k _ { \mathrm { t h } }$ active joint in the human-robot coupling system can be simplified as follows:

$$
\begin{array}{c} J _ {k} \ddot {q} _ {k} + B _ {k} \dot {q} _ {k} + G _ {k} \sin (q _ {k}) + T _ {k} \mathrm{sgn} (\dot {q} _ {k}) \\ = \tau_ {\mathrm{A}, k} + \tau_ {\mathrm{u}, k} + \tau_ {\mathrm{H}, k}, \end{array}\tag{19}
$$

where $J _ { k }$ is the inertial of the robotic exoskeleton (including the subject’s leg); $B _ { k }$ is the viscous friction torque coefficient; $G _ { k }$ is the gravity torque; $T _ { k }$ is the Coulomb friction torque; $\tau _ { \mathrm { A } , k }$ is the control torque; $\tau _ { \mathrm { H } , k }$ is the active joint torque of subjects; $\tau _ { \mathrm { u } , k }$ is the disturbance torque introduced by the uncertain parameters, unknown friction characteristics, and external disturbance of the dynamic model.

Defining the tracking error as $\boldsymbol { e } _ { k } ~ = ~ \boldsymbol { q } _ { k } - \boldsymbol { q } _ { \mathrm { d } , k }$ , and a second-order error auxiliary system can be constructed as follows:

$$
\begin{array}{l} \ddot {e} _ {k} = f _ {k} + b _ {k} \tau_ {\mathrm{A}, k}, \\ f _ {k} = \frac {1}{J _ {k}} (\tau_ {\mathrm{u}, k} + \tau_ {\mathrm{H}, k} - J _ {k} \ddot {q} _ {\mathrm{d}, k} - B _ {k} \dot {q} _ {k} - G _ {k} \sin (q _ {k}) \\ \quad - T _ {k} s g n (\dot {q} _ {k}) + J _ {k} \tau_ {\mathrm{A}, k} - J _ {k} b _ {k} \tau_ {\mathrm{A}, k}), \end{array}\tag{20}
$$

(21)

where $q _ { \mathrm { d } , k }$ and $q _ { k }$ are the desired angle and actual angle; $f _ { k }$ is the total disturbance consisting of internal and external disturbances; $b _ { k }$ is the compensation factor determined by dynamic characteristics.

The state space form of (20) can be given as follows:

$$
\left\{ \begin{array}{l} \dot {\boldsymbol {x}} _ {k} = \boldsymbol {A} _ {k} \boldsymbol {x} _ {k} + \boldsymbol {B} _ {k} \tau_ {\mathrm{A}, k} + \boldsymbol {E} _ {k} \dot {f} _ {k} \\ e _ {k} = \boldsymbol {C} _ {k} \boldsymbol {x} _ {k} \end{array} \right.,\tag{22}
$$

where $\mathbf { \boldsymbol { x } } _ { k } ~ = ~ [ e _ { k } , \dot { e } _ { k } , f _ { k } ] ^ { \mathrm { T } }$ is the extended state vector, and

$$
\boldsymbol {A} _ {k} = \left[ \begin{array}{l l l} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{array} \right] \boldsymbol {B} _ {k} = \left[ \begin{array}{c} 0 \\ b _ {k} \\ 0 \end{array} \right], \boldsymbol {E} _ {k} = \left[ \begin{array}{c} 0 \\ 0 \\ 1 \end{array} \right], \boldsymbol {C} _ {k} = [ 1 0 0 ].
$$

The linear extended state observer corresponding to the above equation can be given as follows:

$$
\left\{ \begin{array}{l} \dot {\boldsymbol {z}} _ {k} = [ \boldsymbol {A} _ {k} - \boldsymbol {L} _ {k} \boldsymbol {C} _ {k} ] \boldsymbol {z} _ {k} + [ \boldsymbol {B} _ {k}, \boldsymbol {L} _ {k} ] \boldsymbol {D} _ {k} \\ \boldsymbol {y} _ {k} = \boldsymbol {z} _ {k} \end{array} \right.,\tag{23}
$$

where $\boldsymbol { z } _ { k } = [ \hat { e } _ { k } , \hat { \dot { e } } _ { k } , \hat { f } _ { k } ] ^ { \mathrm { T } }$ is the state vector; $\scriptstyle L _ { k }$ is the observer gain vector; $\mathbf { \phi } _ { { D } _ { k } } ~ = ~ [ \tau _ { \mathrm { A } , k } , e _ { k } ] ^ { \mathrm { T } }$ is the input vector; $\begin{array} { r l } { \mathfrak { y } _ { k } } & { { } = } \end{array}$ $[ \hat { e } _ { k } , \hat { \dot { e } } _ { k } , \hat { f } _ { k } ]$ is the output vector.

It is necessary to reasonably design the observer gain vector to realize the ideal state observation. The poles of the observer’s characteristic equation are set at the same position on the left side of the complex plane to ensure that the observer gain vector is uniquely related to the observer bandwidth, i.e., $\bar { \bf L } _ { k } = [ 3 \omega _ { k } , 3 \omega _ { k } ^ { 2 } , \bar { \omega } _ { k } ^ { 3 } ] ^ { \mathrm { T } }$ , and it follows that

$$
\lambda (s) = | s \pmb {I} - (\pmb {A} _ {k} - \pmb {L} _ {k} \pmb {C} _ {k}) | = (s + \omega_ {k}) ^ {3},\tag{24}
$$

where $\pmb { I } \in \mathbb { R } ^ { 3 \times 3 }$ is an identity matrix; $\omega _ { k }$ is the observer bandwidth.

For the second-order error auxiliary system, the linear extended state observer can estimate the external and internal disturbances in real time. Hence, the integrator term used to eliminate steady-state errors in classical proportional–integral– derivative is unnecessary. Thus, the linear state feedback control law for the error auxiliary system is simplified to a proportional–derivative controller. The control law for the $k _ { \mathrm { t h } }$ active joint of SYSU-REHAB1-H can be described as follows:

$$
\tau_ {A, k} = \frac {- k _ {p , k} z _ {k , 1} - k _ {d , k} z _ {k , 2} - z _ {k , 3}}{b _ {k}}, \quad k = 1, 2, 3, 4,\tag{25}
$$

where $k _ { p , k }$ and $k _ { d , k }$ are the proportional and derivative gain, respectively; $b _ { k }$ is the compensation factor determined by dynamic characteristics.

The stability analysis of the LADRC for nonlinear systems with dynamic uncertainties has been considered in [34]. Under the assumption that the total disturbance and its time derivative are bounded, the estimation error of the linear extended state observer and the tracking error of the closed-loop control system are bounded, namely bounded-input-bounded-output stable.

## IV. DESIGN OF EXPERIMENT

## A. Experimental Protocol

We conducted robot-assisted walking experiments on four healthy participants and two stroke survivors at the Department of Rehabilitation Medicine, Zhujiang Hospital of Southern Medical University, China. Our research was approved by the Ethics Committee of Zhujiang Hospital of Southern Medical University. The participant information is shown in Table I. The inclusion criteria of the stroke survivors are: 1) stroke survivors should have enough strength to maintain the body balance through upper limb support; 2) stroke survivors should have no significant limitation in the passive range of motion of the hip and knee joints; 3) stroke survivors should have the walking ability, despite the walking stability is reduced and the gait speed is significantly lower than that of person without disability; 4) stroke survivors should have no visuospatial, cognitive, or attention deficits that prevent them from following instructions. Writing informed consent was acquired from all participants before the experiments. Besides, some instructions should be given to the participants: 1) participants should maintain their balance through the hoist and complete the alternating left and right legs by shifting the center of gravity between two legs; 2) participants should walk along the yellow guideline on the ground to the end under the guidance of the operator. Moreover, similar to [25], we chose reference trajectories for the SYSU-REHAB1 through the following steps:1) we collected gait trajectories of healthy participants through the motion capture system; 2) gait trajectories were normalized and averaged, and the averaged gait trajectories were used as the reference trajectories; 3) local weighted regression method was adopted to encode the reference trajectories; 4) we adjusted the gait speed and amplitude through the temporal scaling factor and the amplitude modulation factor to obtain suitable reference trajectories.

Each participant was required to conduct robot-assisted walking experiments using the TD-MACCS with low, medium, and high upper bounds of the deformation factor (TD-MACCS-L, TD-MACCS-M, and TD-MACCS-H). Healthy participants should train for 20 minutes before the experiments, alleviating the influence of initial skill differences on the experimental results. In each trial, the healthy participants should complete a 15 m straight-line robot-assisted walking, and they were permitted to sit on a chair and rest for 5 minutes after a trial. Stroke survivors must train for two days (two groups per day, 30 minutes for each group) before the experiment. It can help patients become familiar with the SYSU-REHAB1-H and master the skill of shifting the center of gravity. Due to the large difference in motor ability between stroke survivors and healthy participants, stroke survivors only needed to complete a 5 m straight-line robot-assisted walking in each trial, and they were permitted to sit on a chair and rest for 5 minutes after three trials. Each control strategy (TD-MACCS-L, TD-MACCS-M, and TD-MACCS-H) was used randomly and discontinuously three times during robot-assisted walking experiments. In addition, the angle, deformation factor, interaction torque, control torque, and observation torque were recorded during each trial.

TABLE I  
INFORMATION OF PARTICIPANTS

<table><tr><td>Parameters</td><td>HP#1</td><td>HP#2</td><td>HP#3</td><td>HP#4</td><td>SS#1</td><td>SS#2</td></tr><tr><td>Age/year</td><td>30</td><td>22</td><td>27</td><td>23</td><td>14</td><td>54</td></tr><tr><td>Weight/kg</td><td>65</td><td>72</td><td>73</td><td>57</td><td>48</td><td>55</td></tr><tr><td>Height/cm</td><td>168</td><td>175</td><td>170</td><td>169</td><td>170</td><td>165</td></tr><tr><td>Months after stroke</td><td>-</td><td>-</td><td>-</td><td>-</td><td>16</td><td>6</td></tr><tr><td>Hemiplegic side</td><td>-</td><td>-</td><td>-</td><td>-</td><td>Left</td><td>Right</td></tr></table>

HP#1, HP#2, HP#3, and HP#4 represent four healthy participants, respectively. SS#1 and SS#2 represent two stroke survivors, respectively.

## B. Parameter Settings

Control parameters of the TD-MACCS should be set reasonably. Similar to [25], the positive constants in the trajectory generator and the active torque estimator were set as follows: $\alpha _ { q } = 2 5 , \beta _ { q } = 6 . 2 5 , \alpha _ { \tau } = 2 5 , \beta _ { \tau } = 6 . 2 5 , \alpha _ { r } = 1 2 . 5 , g _ { q } =$ $\begin{array} { r l r } { \mathbf { g } _ { \tau } } & { { } = } & { \mathbf { 0 } } \end{array}$ . Besides, the temporal scaling factor was set to $\kappa \ = \ 0 . 3$ and the initial value of the amplitude modulation factor was set to $r _ { 0 } ^ { * } ~ = ~ 0 . 8$ . For the trajectory predictor, the prediction time was set to $\textit { p } = \textit { 1 s }$ and the sample period was set to $\delta \ = \ 0 . 0 2 \mathrm { s }$ . The lower bound of the deformation factor was uniformly set to $\pmb { u } _ { d } = \pmb { 0 } \in \mathbb { R } ^ { 4 \times 4 }$ ; and the upper bounds of the deformation factor in TD-MACCS-L, TD-MACCS-M, and TD-MACCS-H were set as follows: $\pmb { u } _ { u } \ = \ \mathbf { 0 } \in \ \mathbb { R } ^ { 4 \times 4 } , \pmb { u } _ { u } \ = \ \mathrm { d i a g } \{ 0 . 0 3 , 0 . 0 3 , 0 . 0 3 , 0 . 0 3 \}$ , and $\pmb { u } _ { u } = \mathrm { d i a g } \{ 0 . 0 5 , 0 . 0 5 , 0 . 0 5 , 0 . 0 5 \}$ , respectively. The interaction torque threshold vector was set to $\bar { \tau } = [ 1 , 0 . 5 , 1 , 0 . 5 ]$ For the multi-modal adaptive regulator, the constant vector, the protection time, the nominal interaction torque vector, and the safe interaction torque vector were set as follows: $\begin{array} { r } { k _ { \tau } = \left[ 0 . 1 5 , 0 . 1 5 , 0 . 1 5 , 0 . 1 5 \right] , L _ { s } = 1 \mathrm { ~ s } , \pmb { \tau } _ { n } = \left[ 1 5 , 1 0 , 1 5 , 1 0 \right] . } \end{array}$ and $\tau _ { \mathrm { s } } = [ 2 0 , 1 5 , 2 0 , 1 5 ]$ , respectively. For the LADRC, the compensation factor vector and the observer bandwidth vector were set to b = diag{1, 1, 1, 1} and $\pmb { \omega } = [ 1 0 0 , 1 0 0 , 1 0 0 , 1 0 0 ] ;$ the proportional gain vector and the derivative gain vector were set to $k _ { \mathrm { p } } ~ = ~ \mathrm { d i a g } \{ 1 8 0 , 1 6 0 , 1 8 0 , 1 6 0 \}$ and $k _ { \mathrm { d } } ~ =$ diag{25, 15, 25, 15}. Note that all control parameters of the TD-MACCS remain unchanged for different participants to ensure the fairness of comparing experimental results.

![](images/f99ba0aa3590739e096f5b9d912f65322498f812fe7932b255e62b18d1b9680e.jpg)  
Fig. 3. Results of SS#1 when using the TD-MACCS-L. The first and second columns are the results of the right and left legs respectively. TT, RT, TE, IT, CT, and OT represent tracking trajectory, reference trajectory, tracking error, interaction torque, control torque, and observation torque, respectively.

## C. Evaluation Method

The average walking speed (AWS), root mean square of trajectory deviation (RMSTD), and normalized root mean square of tracking error (NRMSTE) were used to quantitatively evaluate the movement performance during robot-assisted walking. Each evaluation index can be described as follows:

$$
\mathrm{AWS} = \frac {L _ {w}}{T _ {w}},\tag{26}
$$

$$
\mathrm{RMSTD} = \sqrt {\frac {\sum_ {n = 1} ^ {M} \Delta d _ {k} ^ {2} (n)}{M}},\tag{27}
$$

$$
\text { NRMSTE } = \frac {\sqrt {\frac {\sum_ {n = 1} ^ {M} e _ {k} ^ {2} (n)}{M}}}{\max (q _ {d , k}) - \min (q _ {d , k})},\tag{28}
$$

where $n = 1 , 2 , 3 , 4 , \dots , M$ is the sample number; $L _ { w }$ is the straight-line distance of robot-assisted walking; $T _ { w }$ is the time taken by the participants to complete one trial; $\Delta d _ { k } ( n )$ is the deviation between the reference trajectory and desired trajectory of the $k _ { t h }$ active joint at the $n _ { t h }$ sampling point; $e _ { k } ( n )$ is the tracking error between the desired trajectory and tracking trajectory of the $k _ { t h }$ active joint at the $n _ { t h }$ sampling point; max $( q _ { d , k } )$ and min $( q _ { d , k } )$ are the maximum and minimum desired position of the $k _ { t h }$ active joint, respectively.

## V. EXPERIMENTS RESULTS

Fig. 3 (a), (b), (e), and (f) show that the trajectory generator can generate multi-joint synchronized reference trajectories for active joints of SYSU-REHAB1-H, and the LADRC can ensure that each joint tracks the reference trajectory within a small range of error. Since the upper bound of the deformation factor is set to zero to “shield” the desired trajectory predictor, there is no real-time generation of desired trajectories during pHRI. Fig. 3 (c), (d), (g), and (h) show that the interaction torque, control torque, and observation torque of each joint have obvious periodic characteristics during robot-assisted walking. The interaction torque of each joint in the right leg is within 10 Nm, less than that of the left leg (hemiplegic side). Except for the right hip joint, the control torque curve of each joint has no significant vibration. Besides, the shape and magnitude of the observation torque are consistent with those of the control torque, which shows that the linear extended state observer can effectively compensate for internal and external disturbances.

![](images/d547b1e56ca1710ca94e78f6538d20d18c9836e4c11b38370441ca5e718c47fe.jpg)  
Fig. 4. Results of SS#1 when using the TD-MACCS-M. The first and second columns are the results of the right and left legs respectively. TT, DT, RT, and TE represent tracking trajectory, desired trajectory, reference trajectory, and tracking error, respectively.

Fig. 4 shows that the TD-MACCS-M can smoothly switch the control mode of the robot among human-dominant mode, robot-dominant mode, and soft-stop mode during robotassisted walking. From the gray-shaded part, the soft-stop mode is activated when the interaction torque exceeds the safe interaction torque. The reference trajectory of each joint quickly and smoothly converges to the zero reference position, thus realizing the soft stop protection. Meanwhile, each joint of SYSU-REHAB1-H can track the desired trajectory well. From the orange-shaded part, the robot can smoothly switch to human-dominant mode when it detects the interaction torque smaller than the safe interaction torque after a specified protection time. In human-dominant mode, the robot allows the patient to adjust the reference trajectory naturally and smoothly through pHRI. From the light green shaded part, the robot-dominant mode is activated when the interaction torque exceeds the nominal interaction torque, which can enforce the desired trajectory to converge to the reference trajectory.

Fig. 5 shows that the right leg of the stroke survivor is in human-dominant mode, while the left leg (hemiplegic side) is in either robot-dominant mode or human-dominant mode when using TD-MACCS-H. As shown in Fig. 5 (a), (c), (e), and (g), the trajectory deformation factor of each joint is kept at the maximum value for most of the time, which can allow the right leg to move with greater degrees of freedom. The deviation between the desired and reference trajectories significantly increases compared to TD-MACCS-M, and the right hip and knee joints can track the desired trajectory accurately during pHRI. The interaction torque range of the right leg hip and knee joints are −13 ∼ 5 Nm and −4 ∼ 7 Nm, respectively, and have periodic characteristics. From Fig. 5 (b), (d), (f), and (h), the interaction torque of each joint has more obvious periodic characteristics than that of the right leg. The amplitude of the interaction torque increases significantly, and the deformation factor is smoothly reduced to a minimum value, thus activating the robot-dominant mode to prevent abnormal gait trajectories in the swing phase.

![](images/c704e448eaeabc39d557def9582f37fac0ee8ad3349aa7638996e561fc21d736.jpg)  
Fig. 5. Results of SS#1 when using the TD-MACCS-H. The first and second columns are the results of the right and left legs respectively. TT, DT, RT, and TE represent tracking trajectory, desired trajectory, and reference trajectory, tracking error, respectively.

![](images/4a9adecd36d2d7f9b81c249fdad19f5652003030ff42b47f82cb4759ee159caa.jpg)  
Fig. 6. Results of HP#1 when using the TD-MACCS-H. The first and second columns are the results of the right and left legs respectively. TT, DT, RT, and TE represent tracking trajectory, desired trajectory, and reference trajectory, tracking error, respectively.

![](images/6120d553e33181b6303e39ef2eb47021c339136b210e261f3969591dd9714f0d.jpg)  
Fig. 7. Kinematic data of robot-assisted walking experiments under different upper bounds of the deformation factor: (a) (d), and (g) are the trajectory profiles from 15 gait cycles of HP#1 and SS#1 in joint space; (b), (e), and (h) are the average trajectory profiles from 15 gait cycles of each participant in joint space; (c), (f), and (i) are the average trajectory profiles from 15 gait cycles of the left and right hip joints of each participant. From top to bottom are the results of the TD-MACCS-L, the TD-MACCS-M, and the TD-MACCS-H, respectively. RT represents the reference trajectory profile in joint space.

Fig. 6 shows that the left leg of the healthy participant is in human-dominant mode, while the right leg is in either robot-dominant mode or human-dominant mode when using TD-MACCS-H. Compared to the results of the stroke survivor, robot-assisted walking has less time in robot-dominant mode. From Fig. 6 (a), the healthy participant can smoothly generate desired trajectories with different amplitudes through physical interaction, and the desired trajectory can converge to the reference trajectory quickly once the robot-dominant mode is activated, which is similar to the results of the stroke survivor. Besides, as shown in Fig. 6 (d) and (h), compared with the patient’s left leg (hemiplegic side), the periodicity of the interaction torque of the healthy participant is not obvious, and the amplitude is significantly reduced.

From Fig. 7 (a), (d), and (g), the trajectory profiles of HP#1 and SS#1 in the joint space show a narrow band when using the TD-MACCS-L. Increasing the upper bound of the deformation factor can significantly expand the distribution of the trajectory profiles. Compared with SS#1, the trajectory profile of HP#1 is smoother. Fig. 7 (b), (e), and (h) show that the average trajectory profiles of healthy participants are similar to the reference trajectory profile, whereas they shift more significantly with the upper bound of the deformation factor increasing. Compared with healthy participants, the average trajectory profile of each stroke survivor changes significantly in shape and spatial position when using TD-MACCS-M and TD-MACCS-H. As shown in Fig. 7 (c), (f), and (i), the average trajectory profiles of all participants show a butterfly-like shape, and the differences in gait symmetry and spatial position increase when enlarging the upper bound of the deformation factor. Besides, compared with healthy participants, stroke survivors have the lowest gait symmetry under the high upper bound of the deformation factor.

TABLE II  
STATISTICAL RESULTS OF AWS, RMSTD AND NRMSTE

<table><tr><td rowspan="2">Control strategy</td><td rowspan="2">Participant</td><td colspan="4">RMSTD (rad)</td><td colspan="4">NRMSTE (%)</td><td rowspan="2">AWS (m/s)</td></tr><tr><td>LH</td><td>LK</td><td>RH</td><td>RK</td><td>LH</td><td>LK</td><td>RH</td><td>RK</td></tr><tr><td rowspan="3">TD-MACCS-L</td><td>HP</td><td>0</td><td>0</td><td>0</td><td>0</td><td>4.1±0.7</td><td>4.7±0.3</td><td>3.4±0.8</td><td>3.8±0.4</td><td>0.124±0.021</td></tr><tr><td>SS#1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>5.8±0.9</td><td>7.2±2.2</td><td>5.1±0.5</td><td>5.7±1.6</td><td>0.131±0.007</td></tr><tr><td>SS#2</td><td>0</td><td>0</td><td>0</td><td>0</td><td>4.2±0.8</td><td>5.7±1.2</td><td>7.3±1.6</td><td>6.1±1.9</td><td>0.108±0.006</td></tr><tr><td rowspan="3">TD-MACCS-M</td><td>HP</td><td>0.064±0.005</td><td>0.046±0.004</td><td>0.062±0.009</td><td>0.042±0.006</td><td>3.3±0.4</td><td>4.0±0.6</td><td>3.2±0.6</td><td>4.4±0.6</td><td>0.138±0.020</td></tr><tr><td>SS#1</td><td>0.069±0.004</td><td>0.037±0.009</td><td>0.069±0.003</td><td>0.046±0.045</td><td>7.4±1.2</td><td>6.0±0.1</td><td>6.4±0.8</td><td>5.4±0.8</td><td>0.147±0.002</td></tr><tr><td>SS#2</td><td>0.065±0.005</td><td>0.044±0.003</td><td>0.059±0.015</td><td>0.040±0.012</td><td>4.4±1.2</td><td>5.4±0.5</td><td>6.6±0.9</td><td>6.3±1.2</td><td>0.113±0.003</td></tr><tr><td rowspan="3">TD-MACCS-H</td><td>HP</td><td>0.111±0.011</td><td>0.078±0.004</td><td>0.114±0.006</td><td>0.071±0.006</td><td>3.4±0.6</td><td>3.9±0.5</td><td>3.3±0.4</td><td>4.2±0.5</td><td>0.145±0.019</td></tr><tr><td>SS#1</td><td>0.099±0.004</td><td>0.059±0.003</td><td>0.107±0.003</td><td>0.071±0.004</td><td>9.0±2.4</td><td>6.2±0.7</td><td>5.7±1.1</td><td>5.1±00.8</td><td>0.138±0.003</td></tr><tr><td>SS#2</td><td>0.117±0.018</td><td>0.067±0.016</td><td>0.108±0.030</td><td>0.072±0.011</td><td>7.9±4.1</td><td>6.0±0.8</td><td>5.1±0.7</td><td>6.0±1.4</td><td>0.113±0.020</td></tr></table>

Table II shows the RMSTD, NRMSTE, and AWS mean of all healthy participants and each stroke survivor when using the TD-MACCS-L, the TD-MACCS-M, and the TD-MACCS-H. Compared with the TD-MACCS-M, the RMSTD mean of all healthy participants is increased by 73.44% (LH), 69.57% (LK), 83.87% (RH), and 69.05% (RK), respectively, when the TD-MACCS-H is adopted. Notably, the trend that the RMSTD mean of each stroke survivor increases does not change when enlarging the upper bound of the deformation factor. Besides, compared with the TD-MACCS-L, except for the right knee joint, the NRMSTE mean of all healthy participants decreases when using the TD-MACCS-M and the TD-MACCS-H. The maximum NRMSTE mean of all healthy participants is less than 5.00%. Although the NRMSTE mean of each stroke survivor is larger than that of all healthy participants, the tracking accuracy of each joint is within an acceptable range. In addition, the AWS mean of all healthy participants increases obviously when increasing the upper bound of the deformation factor, and the maximum AWS mean reaches 0.145 m/s. Compared with the results of all healthy participants, a moderate increase in the upper bound of the deformation factor improves the AWS mean of each stroke survivor, but an excessive value decreases the AWS mean.

## VI. DISCUSSION

## A. Multi-Modal Adaptive Compliance Control

Adaptive compliance control strategies can make rehabilitation robots match users’ ever-changing motor abilities and improve training safety. In this article, TDA [16] and DMPs [24] are effectively integrated into the TD-MACCS, which can smoothly switch the human-dominant, robotdominant, and soft-stop modes. In the human-dominant mode, the SYSU-REHAB1-H allows the participants to adjust the reference trajectory with variable deformation factors through compliant pHRI, encouraging their active participation in robot-assisted walking. Since patients with lower extremity motor dysfunction usually have different symptoms, they are prone to abnormal gait trajectories [35]. In the robot-dominant mode, the SYSU-REHAB1-H can smoothly reduce the deformation factor to a minimum value, effectively correcting the participants’ abnormal gait trajectory by restricting their freedom of motion. Furthermore, safety protection deserves paying more attention during robot-assisted training [20]. In the soft-stop mode, the reference trajectory can smoothly switch to the zero reference position by adjusting the desired amplitude factor of DMPs. Besides, the reference trajectory can automatically and smoothly switch back to the reference trajectory after the specified protection time. Although multi-modal control strategies can be designed based on an error-based function [9], [22], [36], they may cause oscillations in the control signal and tracking trajectory when switching continuously between different modes. In comparison, the TD-MACCS can realize smooth switching of different modes through a multi-modal adaptive regulator without considering the tracking error. Furthermore, the participants can easily activate the soft-stop mode through the interaction torque, effectively solving the problem that the error-based safety stop mode is challenging to activate under impact forces or low robot compliance [21]. Compared to multi-modal control strategies based on customized position controller design [20], [21], the TD-MACCS can realize the separation of the LADRC and the multi-modal adaptive regulator design, which can avoid complicated position controller design and simplify parameter adjustment. Additionally, learning and generalizing the reference trajectory has not been considered in multi-modal control strategies [9], [19], [20], [21], [22], [23], [36]; the trajectory generator based on DMPs can help therapists obtain and adjust reference trajectories of different rehabilitation tasks [25], [37].

## B. Effects on Movement Performance

TD-MACCS with different upper bound values of the deformation factor can influence robot compliance, walking speed, and gait symmetry during robot-assisted walking. In this article, the TD-MACCS-H obtained the largest RMSTD among the three tested control strategies while maintaining trajectory tracking accuracy. The reason is that the LADRC can effectively compensate for internal and external disturbances and ensure trajectory tracking accuracy [33], [34]. Furthermore, increasing the upper bound value of the deformation factor enables the participants to adjust the reference trajectory more easily [16], enlarging the RMSTD. Although it can promote the patients’ motor exploration and learning [25], the excessive upper bound value of the deformation factor leads to a decrease in AWS for patients. The reason is that increasing robot compliance reduces the degree of inter-joint coordination kinematic guidance for the participants, which increases the difficulty of human-robot synergy movement and makes the patients improve their local dynamic stability by reducing walking speed [38]. Additionally, there is a significant decrease in gait symmetry when increasing the upper bound value of the deformation factor. The reason is that the WLLRR reduces the normal walking speed of healthy participants [39] and leads to the difference between the left and right step lengths [40]. Compared with the healthy participants, the gait asymmetry of stroke survivors caused by lower limb movement disorder can be corrected by the WLLRR by decreasing the upper bound value of the deformation factor. Moreover, compared with healthy participants, the duration of the robot-dominant mode increased significantly during patients’ robot-assisting walking. It indicates that the WLLRR using TD-MACCS can correct gait asymmetry under a predetermined upper bound value of the deformation factor by adaptively switching to a control mode that matches patients varying motor abilities.

## C. Limitations and Future Work

This article has several limitations that should be addressed in future research. First, although we have tried our best to enhance the stability of the different parts of the TD-MACCS, the stability analysis was not conducted because the dynamics of participants may not be perfectly known [41], and it is difficult to model accurately. Second, although we propose to distinguish different control modes based on the nominal and safe interaction torques, they need to be set reasonably according to the patient’s disability level and the comfort of physical interaction; besides, to ensure the patient’s safety and comfort, the nominal and safe interaction torques set in this study are too conservative. Third, we did not standardize the walking distances in our experimental paradigm, and only two stroke survivors were included in robot-assisted walking experiments. In further studies, we will attempt to construct the dynamic model of the participants and introduce new methods, such as energy tanks [41], [42], for stability analysis. Furthermore, more patients will be recruited, and the 2-min and the 10-m walk tests will be performed to assess the ambulatory function of patients [43], which can help the physiotherapist determine the appropriate robot-assisted walking speed and nominal interaction torque. Besides, an external information acquisition system will be designed to monitor the contact behavior between the exoskeleton and skin, torso motion, and total arm force during robot-assisted walking [15], [44], which helps determine the appropriate safe interaction torque and objectively and quantitatively evaluate patients’ acceptance of the TD-MACCS. Meanwhile, patient-based robot-assisted rehabilitation training will be carried out, and statistically significant results will be obtained to verify the clinical effectiveness of the TD-MACCS.

## VII. CONCLUSION

This article proposed a TD-MACCS for a self-designed WLLRR, SYSU-REHAB1-H. Robot-assisted walking experiments and quantitative analysis were performed. The experimental results have been provided to verify that the

TD-MACCS can adaptively and smoothly switch different control modes. Moreover, a moderate increase in the upper bound of the deformation factor can enhance the AWS and RMSTD while maintaining the NRMSTE within an allowable range. Therefore, TD-MACCS has the potential to be used in robot-assisted training to match the varying motor abilities of stroke survivors. It can also be applied in other fields involving human-robot cooperation based on pHRI.

## REFERENCES

[1] D. Mukherjee and C. G. Patil, “Epidemiology and the global burden of stroke,” World Neurosurg., vol. 76, no. 6, pp. S85–S90, Dec. 2011, doi: 10.1016/j.wneu.2011.07.023.

[2] G.-M. Shi et al., “Profile and 1-year outcome of ischemic stroke in East China: Nanjing first hospital stroke registry,” J. Stroke Cerebrovascular Diseases, vol. 25, no. 1, pp. 49–56, Jan. 2016, doi: 10.1016/j.jstrokecerebrovasdis.2015.08.032.

[3] M. Bortole et al., “The H2 robotic exoskeleton for gait rehabilitation after stroke: Early findings from a clinical study,” J. Neuroeng. Rehabil., vol. 12, no. 1, pp. 1–14, Jun. 2015, doi: 10.1186/s12984-015-0048-y.

[4] D. Ando et al., “Microstructural white matter changes following gait training with hybrid assistive limb initiated within 1 week of stroke onset,” J. Neurol. Sci., vol. 415, Aug. 2020, Art. no. 116939, doi: 10.1016/j.jns.2020.116939.

[5] E. Høyer, A. Opheim, and V. Jørgensen, “Implementing the exoskeleton ekso GTTM for gait rehabilitation in a stroke unit—Feasibility, functional benefits and patient experiences,” Disab. Rehabil., Assistive Technol., vol. 17, no. 4, pp. 473–479, May 2022, doi: 10.1080/17483107.2020.1800110.

[6] Y. Li et al., “Efficacy of a novel exoskeletal robot for locomotor rehabilitation in stroke patients: A multi-center, non-inferiority, randomized controlled trial,” Frontiers Aging Neurosci., vol. 13, Aug. 2021, Art. no. 706569, doi: 10.3389/fnagi.2021.706569.

[7] H. Watanabe, R. Goto, N. Tanaka, A. Matsumura, and H. Yanagi, “Effects of gait training using the hybrid assistive Limb in recoveryphase stroke patients: A 2-month follow-up, randomized, controlled study,” NeuroRehabilitation, vol. 40, no. 3, pp. 363–367, May 2017, doi: 10.3233/NRE-161424.

[8] F. Molteni et al., “Gait recovery with an overground powered exoskeleton: A randomized controlled trial on subacute stroke subjects,” Brain Sci., vol. 11, no. 1, p. 104, Jan. 2021, doi: 10.3390/brainsci11010104.

[9] Q. Wu, X. Wang, B. Chen, and H. Wu, “Development of a minimalintervention-based admittance control strategy for upper extremity rehabilitation exoskeleton,” IEEE Trans. Syst., Man, Cybern. Syst., vol. 48, no. 6, pp. 1005–1016, Jun. 2018, doi: 10.1109/TSMC.2017.2771227.

[10] Y. Long, Z. Du, L. Cong, W. Wang, Z. Zhang, and W. Dong, “Active disturbance rejection control based human gait tracking for lower extremity rehabilitation exoskeleton,” ISA Trans., vol. 67, pp. 389–397, Mar. 2017, doi: 10.1016/j.isatra.2017.01.006.

[11] Ü. Önen, F. M. Botsali, M. Kalyoncu, M. Tinkir, N. Yilmaz, and Y. Sahin, “Design and actuator selection of a lower extremity exoskeleton,” IEEE/ASME Trans. Mechatronics, vol. 19, no. 2, pp. 623–632, Apr. 2014, doi: 10.1109/TMECH.2013.2250295.

[12] Z. Warraich and J. A. Kleim, “Neural plasticity: The biological substrate for neurorehabilitation,” PM&R, vol. 2, no. 12, pp. S208–S219, Dec. 2010, doi: 10.1016/j.pmrj.2010.10.016.

[13] Z. Shen, J. Zhou, J. Gao, and R. Song, “Torque tracking impedance control for a 3DOF lower limb rehabilitation robot,” in Proc. 3rd Int. Conf. Adv. Robot. Mechatronics (ICARM), Singapore, Jul. 2018, pp. 294–299, doi: 10.1109/ICARM.2018.8610807.

[14] P. Huang, Z. Li, M. Zhou, X. Li, and M. Cheng, “Fuzzy enhanced adaptive admittance control of a wearable walking exoskeleton with step trajectory shaping,” IEEE Trans. Fuzzy Syst., vol. 30, no. 6, pp. 1541–1552, Jun. 2022, doi: 10.1109/TFUZZ.2022.3162700.

[15] B. Ugurlu, H. Oshima, E. Sariyildiz, T. Narikiyo, and J. Babic, “Active compliance control reduces upper body effort in exoskeleton-supported walking,” IEEE Trans. Human-Mach. Syst., vol. 50, no. 2, pp. 144–153, Apr. 2020, doi: 10.1109/THMS.2019.2961969.

[16] J. Zhou, Z. Li, X. Li, X. Wang, and R. Song, “Human–robot cooperation control based on trajectory deformation algorithm for a lower limb rehabilitation robot,” IEEE/ASME Trans. Mechatronics, vol. 26, no. 6, pp. 3128–3138, Dec. 2021, doi: 10.1109/TMECH.2021.3053562.

[17] S. K. Banala, S. H. Kim, S. K. Agrawal, and J. P. Scholz, “Robot assisted gait training with active leg exoskeleton (ALEX),” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 17, no. 1, pp. 2–8, Feb. 2009, doi: 10.1109/TNSRE.2008.2008280.

[18] A. Martínez, B. Lawson, C. Durrough, and M. Goldfarb, “A velocityfield-based controller for assisting leg movement during walking with a bilateral hip and knee lower limb exoskeleton,” IEEE Trans. Robot., vol. 35, no. 2, pp. 307–316, Apr. 2019, doi: 10.1109/TRO.2018.2883819.

[19] Z. Li et al., “Human-in-the-loop control of a wearable lower limb exoskeleton for stable dynamic walking,” IEEE/ASME Trans. Mechatronics, vol. 26, no. 5, pp. 2700–2711, Oct. 2021, doi: 10.1109/TMECH.2020.3044289.

[20] J. Zhang and C. C. Cheah, “Passivity and stability of human–robot interaction control for upper-limb rehabilitation robots,” IEEE Trans. Robot., vol. 31, no. 2, pp. 233–245, Apr. 2015, doi: 10.1109/TRO.2015.2392451.

[21] X. Li, Y. Pan, G. Chen, and H. Yu, “Multi-modal control scheme for rehabilitation robotic exoskeletons,” Int. J. Robot. Res., vol. 36, nos. 5–7, pp. 759–777, Feb. 2017, doi: 10.1177/0278364917691111.

[22] X. Li, Q. Yang, and R. Song, “Performance-based hybrid control of a cable-driven upper-limb rehabilitation robot,” IEEE Trans. Biomed. Eng., vol. 68, no. 4, pp. 1351–1359, Apr. 2021, doi: 10.1109/TBME.2020.3027823.

[23] J. Xu et al., “A multi-mode rehabilitation robot with magnetorheological actuators based on human motion intention estimation,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 27, no. 10, pp. 2216–2228, Oct. 2019, doi: 10.1109/TNSRE.2019.2937000.

[24] A. J. Ijspeert, J. Nakanishi, H. Hoffmann, P. Pastor, and S. Schaal, “Dynamical movement primitives: Learning attractor models for motor behaviors,” Neural Comput., vol. 25, no. 2, pp. 328–373, Feb. 2013, doi: 10.1162/NECO\_a\_00393.

[25] J. Zhou, H. Peng, S. Su, and R. Song, “Spatiotemporal compliance control for a wearable lower limb rehabilitation robot,” IEEE Trans. Biomed. Eng., vol. 70, no. 6, pp. 1858–1868, Jun. 2023, doi: 10.1109/TBME.2022.3230784.

[26] Y. Yuan, Z. Li, T. Zhao, and D. Gan, “DMP-based motion generation for a walking exoskeleton robot using reinforcement learning,” IEEE Trans. Ind. Electron., vol. 67, no. 5, pp. 3830–3839, May 2020, doi: 10.1109/TIE.2019.2916396

[27] M. Sharifi, J. K. Mehr, V. K. Mushahwar, and M. Tavakoli, “Autonomous locomotion trajectory shaping and nonlinear control for lower limb exoskeletons,” IEEE/ASME Trans. Mechatronics, vol. 27, no. 2, pp. 645–655, Apr. 2022, doi: 10.1109/TMECH.2022.3156168.

[28] K. Gui, H. Liu, and D. Zhang, “A practical and adaptive method to achieve EMG-based torque estimation for a robotic exoskeleton,” IEEE/ASME Trans. Mechatronics, vol. 24, no. 2, pp. 483–494, Apr. 2019, doi: 10.1109/TMECH.2019.2893055.

[29] M. Dong et al., “A new ankle robotic system enabling wholestage compliance rehabilitation training,” IEEE/ASME Trans. Mechatronics, vol. 26, no. 3, pp. 1490–1500, Jun. 2021, doi: 10.1109/TMECH.2020.3022165.

[30] D. P. Losey and M. K. O’Malley, “Trajectory deformations from physical human–robot interaction,” IEEE Trans. Robot., vol. 34, no. 1, pp. 126–138, Feb. 2018, doi: 10.1109/TRO.2017.2765335.

[31] M. Yang, X. Wang, Z. Zhu, R. Xi, and Q. Wu, “Development and control of a robotic lower limb exoskeleton for paraplegic patients,” Proc. Inst. Mech. Eng., C, J. Mech. Eng. Sci., vol. 233, no. 3, pp. 1087–1098, Feb. 2019, doi: 10.1177/0954406218761484.

[32] R. Fareh, S. Khadraoui, M. Y. Abdallah, M. Baziyad, and M. Bettayeb, “Active disturbance rejection control for robotic systems: A review,” Mechatronics, vol. 80, Dec. 2021, Art. no. 102671, doi: 10.1016/j.mechatronics.2021.102671.

[33] Z. Gao, “Scaling and bandwidth-parameterization based controller tuning,” in Proc. Amer. Control Conf., Denver, CO, USA, Jun. 2003, pp. 4989–4996, doi: 10.1109/ACC.2003.1242516.

[34] H. Peng, J. Zhou, and R. Song, “A triple-step controller with linear active disturbance rejection control for a lower limb rehabilitation robot,” Frontiers Neurorobotics, vol. 16, Nov. 2022, Art. no. 1053360, doi: 10.3389/fnbot.2022.1053360.

[35] N. Sánchez, A. M. Acosta, R. Lopez-Rosado, A. H. A. Stienen, and J. P. A. Dewald, “Lower extremity motor impairments in ambulatory chronic hemiparetic stroke: Evidence for lower extremity weakness and abnormal muscle and joint torque coupling patterns,” Neurorehabilitation Neural Repair, vol. 31, no. 9, pp. 814–826, Aug. 2017, doi: 10.1177/1545968317721974.

[36] R. Yang, Z. Shen, Y. Lyu, Y. Zhuang, L. Li, and R. Song, “Voluntary assist-as-needed controller for an ankle power-assist rehabilitation robot,” IEEE Trans. Biomed. Eng., vol. 70, no. 6, pp. 1795–1803, Jun. 2023, doi: 10.1109/TBME.2022.3228070.

[37] J. Nielsen, A. S. Sorensen, T. S. Christensen, T. R. Savarimuthu and T. Kulvicius, “Individualised and adaptive upper limb rehabilitation with industrial robot using dynamic movement primitives,” in Proc. ICRA Workshop Adv. Challenges Develop. Test. Assessment Assistive Rehabil. Robots, Exper. From Eng. Human Sci. Res., vol. 1, 2017, p. 40.

[38] J. B. Dingwell, J. P. Cusumano, D. Sternad, and P. R. Cavanagh, “Slower speeds in patients with diabetic neuropathy lead to improved local dynamic stability of continuous overground walking,” J. Biomech., vol. 33, no. 10, pp. 1269–1277, Oct. 2000, doi: 10.1016/S0021- 9290(00)00092-0.

[39] M. Plotnik, R. P. Bartsch, A. Zeev, N. Giladi, and J. M. Hausdorff, “Effects of walking speed on asymmetry and bilateral coordination of gait,” Gait Posture, vol. 38, no. 4, pp. 864–869, Sep. 2013, doi: 10.1016/j.gaitpost.2013.04.011.

[40] Y. Xiang, J. S. Arora, and K. Abdel-Malek, “Optimization-based prediction of asymmetric human gait,” J. Biomechanics, vol. 44, no. 4, pp. 683–693, Feb. 2011, doi: 10.1016/j.jbiomech.2010.10.045.

[41] F. Benzi, F. Ferraguti, G. Riggio, and C. Secchi, “An energybased control architecture for shared autonomy,” IEEE Trans. Robot., vol. 38, no. 6, pp. 3917–3935, Dec. 2022, doi: 10.1109/TRO. 2022.3180885.

[42] Y. Michel, C. Ott, and D. Lee, “Safety-aware hierarchical passivitybased variable compliance control for redundant manipulators,” IEEE Trans. Robot., vol. 38, no. 6, pp. 3899–3916, Dec. 2022, doi: 10.1109/TRO.2022.3174478.

[43] T. Nakajima et al., “Cybernic treatment with wearable cyborg hybrid assistive limb (HAL) improves ambulatory function in patients with slowly progressive rare neuromuscular diseases: A multicentre, randomised, controlled crossover trial for efficacy and safety (NCY-3001),” Orphanet J. Rare Diseases, vol. 16, no. 1, p. 304, Jul. 2021, doi: 10.1186/s13023-021-01928-9.

[44] X. Wan, Y. Liu, Y. Akiyama, and Y. Yamada, “Monitoring contact behavior during assisted walking with a lower limb exoskeleton,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 28, no. 4, pp. 869–877, Apr. 2020, doi: 10.1109/TNSRE.2020.2979986.