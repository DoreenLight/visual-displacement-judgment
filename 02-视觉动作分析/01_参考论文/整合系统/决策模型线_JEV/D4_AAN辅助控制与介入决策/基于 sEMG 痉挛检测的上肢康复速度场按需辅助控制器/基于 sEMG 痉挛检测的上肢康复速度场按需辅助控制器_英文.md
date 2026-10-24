# Velocity field assist-as-needed controller for upper limb rehabilitation with sEMG-based spasticity detection

![](images/4fc9b9fc14275090efb5231fe870a19f2dea2a79f10467ce1db6114c9c518bea.jpg)

Ye Li , Aiguo Song <sup>∗</sup>, Jianwei Lai, Ye Lu, Huijun Li

School of Instrument Science and Engineering, Southeast University, Nanjing 210096, China

A R T I C L E I N F O

Keywords: Rehabilitation training Assist-as-needed controller Velocity field Spasticity detection Long Short-Term Memory

## A B S T R A C T

Robot-assisted training has demonstrated significant potential in facilitating motor function recovery for stroke patients with hemiplegia. Among various rehabilitation strategies, the assist-as-needed (AAN) strategy, which promotes user participation while minimizing robotic intervention based on task performance or physiological states, has been widely adopted in rehabilitation robotics. This paper presents a novel variable admittance velocity field AAN controller integrated with spasticity detection capability. The primary objective was to develop a control strategy that enables upper limb trajectory tracking while ensuring training safety. First, we designed an admittance-controlled velocity field controller that achieves the AAN property through adaptive adiustment of admittance parameters. The velocity field design ensures precise trajectory tracking while maintaining temporal flexibility. Second, we propose an sEMG-based spasticity detection method that utilizes a Long Short-Term Memory (LSTM) network to model elbow spasticity patterns. In addition, a smooth velocity switching function is designed to implement slow stretching of the affected limb during the spasticity phase. The performance of the controller was experimentally validated on both healthy subjects and post-stroke patients using a planar upper-limb rehabilitation robotic system. The results demonstrated that the proposed controller achieved better trajectory tracking accuracy and enhanced AAN performance compared to traditional force-field controller and impedance-based controller, and was capable of adjusting velocity upon the detection of subject spasticity.

## 1. Introduction

Stroke is one of the main causes of chronic physical disability, with 70% to 80% of post-stroke patients losing the ability to perform activities of daily living [1]. Rehabilitation robots are widely used in post-stroke treatment due to their ability to provide repetitive and precise training. Numerous clinical studies have also demonstrated the positive impacts of robot-assisted rehabilitation on patient recovery, with active patient participation significantly enhancing treatment outcomes [2,3]. To ensure active participation, controllers based on the assist-as-needed (AAN) strategy have been developed [4], in which the robot controller provides the minimum required assistance to the patient.

Most AAN controllers utilize impedance control as their fundamental control strategy, which ensures precise trajectory tracking while maintaining compliance, and enables adaptive adjustment of robot impedance parameters or desired trajectories based on the patient’s training performance [5,6]. However, these controllers rely on timebased desired trajectories, requiring the patient’s affected limb to reach specific positions on the trajectory at predetermined times, which may lead to inappropriate assistance during rehabilitation training. Another common AAN controllers involve constructing force field control (FFC) methods to provide assistive forces that guide patients along desired trajectories. For instance, Pan et al. [7] partitioned the training space into human-dominated and robot-dominated areas, implementing a virtual force field channel to guide patients along the trajectory for compliant assistance training. In the research of Rossa et al. [8], a potential field function was designed to calculate the required assistance force at different positions, combined with a learning-from-demonstration strategy to enable the robot to mimic therapist-guided training trajectory. Although force field based controllers eliminate strict temporal constraints, they generally cannot achieve precise trajectory tracking and often require additional dissipative fields to stabilize the robot’s motion velocity [9].

To achieve freedom in movement timing while maintaining accurate trajectory tracking, several studies have proposed velocity field control (VFC) methods. These controllers guide patients to move along the trajectory at a desired velocity, effectively decoupling temporal factors from patient movement. They can also adaptively adjust the desired velocity based on the patient’s performance or trajectory deviation, thereby enabling AAN control based on the velocity field. For example, Guo et al. [10] proposed a task performance based adaptive velocity assist-as-needed controller for upper limb rehabilitation training. This controller applies position constraints in the normal direction of the task trajectory to limit tracking errors, while imposing velocity control in the tangential direction to preserve temporal flexibility during movement. Hernandez et al. [11] introduced an AAN training strategy centered on an adaptive velocity field, where the direction and magni tude of the velocity field vary with position to promote active patient participation.

On the other hand, spasticity is a highly prevalent symptom in individuals with neurological injuries, with estimated occurrence rates ranging from 30% to 80% after stroke [12]. This condition leads to abnormal postures and movement patterns, significantly hindering motor function recovery [13]. Therefore, spasticity detection during robot-assisted rehabilitation training is particularly crucial. Current spasticity detection methods typically rely on clinicians manually evaluating limb resistance based on rating scales (e.g., Ashworth Scale, Modified Ashworth Scale). These methods depend on clinician experience and cannot achieve real-time spasticity detection. Consequently, several studies have proposed real-time spasticity detection methods for rehabilitation training. For instance, Hu et al. [14] proposed a spasticity detection method based on Hilbert-Huang transform marginal spectrum entropy, which can accurately identify stretch reflex onset in surface electromyography (sEMG) signals and has been validated through experiments with patients. Trigili et al. [15] designed a 4 degree-of-freedom (DOF) shoulder-elbow exoskeleton for upper-limb neurorehabilitation and developed a threshold-based online spasticity detection method. When abnormal torque distribution is detected at the robot joint, it is identified as spasticity, and the robot switches to transparent mode to ensure rehabilitation training safety. Although both biomechanical and sEMG signals can be utilized for spasticity detection, sEMG-based methods are more widely adopted due to their direct measurement capability and higher precision [14,16]. However, most existing sEMG-based spasticity detection methods primarily focus on spasticity grading and offline assessment, with limited research dedicated to real-time spasticity detection during rehabilitation training [12].

In this study, we propose a novel VFC for constructing a smooth velocity field around the desired task trajectory. This method is integrated with a variable admittance-based low-level controller to track reference velocity vectors and achieve the AAN property. Since the reference velocity depends on position rather than time, the proposed method provides temporal flexibility during rehabilitation training. Additionally, to address spasticity during rehabilitation, we introduce an sEMG-based spasticity detection method that leverages Long Short-Term Memory (LSTM) network to model the time-series characteristics of sEMG signals, enabling accurate spasticity detection. The system can smoothly switch the robot’s velocity field based on the detected spasticity state, ensuring safe and stable rehabilitation training. The effectiveness of the proposed method has been validated on a 2 DOF upper limb rehabilitation robot. The main contributions of this paper b l d f ll

(1) A novel variable admittance VFC (VA-VFC) is proposed. Compared with existing VFC strategies [17,18], the proposed controller optimizes the construction of the velocity field, enabling smooth ve locity modulation according to the subject’s trajectory deviation. By integrating a variable admittance strategy, the method enhances patient active participation while preserving temporal flexibility. The effectiveness of the proposed VA-VFC has been validated in various training scenarios.

(2) An sEMG-based spasticity detection method is developed, which combines sEMG with an LSTM network to achieve accurate spasticity detection during active patient training. Furthermore, based on the detection results, a velocity smooth-switching function is designed to ensure safe and stable rehabilitation training in the presence of spasticity.

![](images/1d847caadfb482622b990a9df79bc6d21802186c3bb50a3b84eb6f541115bfb5.jpg)  
Fig. 1. (a) Training scenario of the rehabilitation system. (b) Structural sketch of the 2 DOF upper limb rehabilitation robot.

(3) Comprehensive experimental validation is conducted on both healthy subjects and post-stroke patients, demonstrating the feasibility and potential of the proposed VA-VFC with spasticity detection for clinical rehabilitation applications.

## 2. Upper limb rehabilitation system

Planar rehabilitation robots have been widely used in hospitals and research institutions due to their advantages of ease of use, low cost, and high safety [2,3]. In our previous work [19], a 2 DOF end-effector-based robot (see Fig. 1(a)) has been developed for upper limb rehabilitation training of the subject suffering from stroke or neurological impairments. The rehabilitation robot adopts a structural design that combines parallel linkages and rhombic linkages in series, enabling it to provide patients with a wide range of planar translational movements. An arm bracket is linked to the end-effector to support the affected arm. The rehabilitation robot can provide shoulder and lb h b l f h l f of 30N. A six-dimensional force/torque sensor (Mini45 Titanium, ATI Industrial Automation, USA) was mounted at the end-effector of the robot to collect interaction force data, while encoders (Encoder MR, maxon international, CHE) were mounted at the robot joints to collect <sub>angle data. For sEMG signal acquisition, sEMG electrodes (Trigno</sub>™ Wireless System, Delsys, USA) were placed on the biceps brachii (BB) and triceps brachii (TB) muscles of subjects’ upper limbs to record neuromuscular activity.

The simplified diagram of the robot is shown in Fig. 1(b), which consists of two revolute joints connected by rigid links. And denoting the angle, angular velocity, and angular acceleration of robot joints as �, $\dot { \mathbf { \zeta } } , \ddot { q } \in \mathbb { R } ^ { 2 }$ , the dynamics of the robot can be described as follows:

$$
M _ {q} (q) \ddot {q} + C _ {q} (q, \dot {q}) \dot {q} + G _ {q} (q) + D _ {q} (\dot {q}) = \tau_ {r} + J _ {r} ^ {T} F _ {h}\tag{1}
$$

where $M _ { q }$ is the inertia matrix of the robot, $C _ { q }$ is the coriolis and centrifugal matrix of the robot, $G _ { q }$ represents the gravity vector, and $D _ { q }$ represents the friction vector. $\tau _ { r }$ is the vector of torques provided by the robot joints, $F _ { h }$ is the vector of forces exerted by the patient on the robot’s end-effector (the interaction force), and � is the robot Jacobian matrix related with the robot’s structural parameters and joint angles.

Based on the simplified diagram in Fig. 1(b), the rehabilitation robot used in this study can be modeled as a two-link manipulator with a specific configuration. Accordingly, the forward kinematics formula of the upper-limb rehabilitation robot can be derived as follows:

$$
\left\{ \begin{array}{l l} x = L _ {1} \cos (\alpha) \\ y = L _ {1} \sin (\alpha) + 2 L _ {2} \sin (\beta) \end{array} \right.\tag{2}
$$

![](images/88199e646050539ed1160742b1a1d662ec2983db9fb1e67fd94faa7be1e14e82.jpg)  
Fig. 2. Velocity field of the circular trajectory training task.

By differentiating the end-effector position vector $p = [ x , y ] ^ { T }$ with respect to the joint angle vector $q = [ \alpha , \beta ] ^ { T }$ , the Jacobian matrix of the robot can be obtained:

$$
J _ {r} = \left[ \begin{array}{c c} - L _ {1} \sin (\alpha) & 0 \\ L _ {1} \cos (\alpha) & 2 L _ {2} \cos (\beta) \end{array} \right]\tag{3}
$$

By utilizing the Jacobian matrix, we can develop the robot dynamics based on the Cartesian space variables as follows:

$$
M _ {p} (p) \ddot {p} + C _ {p} (p, \dot {p}) \dot {p} + D _ {p} (\dot {p}) = J _ {r} ^ {- T} \tau_ {r} + F _ {h}\tag{4}
$$

where $p , \dot { p } , \ddot { p } \in \mathbb { R } ^ { 2 }$ represent the position, velocity and acceleration of the robot’s end-effector. And the new dynamic terms are defined as $M _ { p } ( p ) = J _ { r } ^ { - T } M _ { q } ( q ) J _ { r } ^ { - 1 } , C _ { p } ( p , \dot { p } ) = J _ { r } ^ { - T } [ C _ { q } ( q , \dot { q } ) - M _ { q } ( q ) J ^ { - 1 } \dot { J } ] J _ { r } ^ { - 1 }$ , and $D _ { p } ( \dot { p } ) = J _ { r } ^ { - T } D _ { q } ( \dot { q } )$ ). Since the robot performs planar motion, the gravity term $G _ { q }$ can be neglected in (4).

To ensure the safety of participants during rehabilitation training, the rehabilitation system is equipped with an emergency stop button, and the upper limit of the robot’s assistive force $F _ { r }$ is set to 15 N.

## 3. Controller design

In this section, a VA-VFC is proposed, which describes the desired training trajectory in terms of velocity, thereby overcoming the time dependence of traditional path controllers and providing greater flex ibility in motion timing. In the proposed controller, the velocity field vectors are used as desired inputs for the robot to guide movements toward the desired path. And the control objective is to assist patients when they cannot follow the desired velocity profile while ensuring compliant robot behavior.

## 3.1. Adaptive velocity field

Rehabilitation training trajectories are typically either graphically defined by analytical expressions or demonstration trajectories provided by rehabilitation therapists. For the analytical trajectories, the desired velocity field can be constructed based on their mathematical models. For the demonstration trajectories, the desired velocity field can be generated by sampling the demonstration trajectory, where increasing the number of sampling points results in a smoother velocity field. The proposed velocity field construction method in this study is applicable to both analytical trajectories and demonstration trajectories.

The desired path considered in this work is circular trajectory with a radius of $r = 0 .$ .15m in the Cartesian space, as shown in Fig. 2. And the center of the trajectory (represented by yellow dot) is set as $p _ { 0 } =$ [0, 0] to simplify the calculation. During rehabilitation training, visual feedback is provided to subjects through a display interface showing: the target trajectory, the current position $p ,$ and the guide position $p _ { g }$ projected � degrees ahead of the $p .$ In the experimental protocol, subjects consistently initiated movements from the starting point $p _ { s }$ on the left side of the circular trajectory and performed training exercises in a counterclockwise direction. The velocity field $v _ { d e s }$ (represented by blue arrows) is designed to guide the end-effector along the reference trajectory, parameterized by position � rather than time �. And the velocity field can be constructed by the normal vector ̂� directed toward the path and the tangent vector �̂ along the path. This can be formulated as follows:

$$
v _ {d e s} (p, \dot {p}) = k (p, \dot {p}) \hat {n} + c (p, \dot {p}) \hat {t}\tag{5}
$$

The contribution of unit vectors ̂� and �̂ in $v _ { d e s }$ is weighted according to the nonlinear functions �(�, ̇�) and �(�, ̇�), respectively. To achieve compliant rehabilitation training, it is desired that as the end-effector approaches the desired trajectory, the velocity field is predominantly guided by the tangential vector �̂, ensuring motion along the desired trajectory. Conversely, as the end-effector deviates further from the desired trajectory, the velocity field is increasingly dominated by the normal vector $\hat { n } ,$ assisting the patient to correct the motion deviation. To balance these two effects, we improved the weight parameter design based on [11,20], yielding $\begin{array} { r } { k ( p , \dot { p } ) \stackrel { \star } { = } \frac { k _ { 0 } t a n h ( k _ { 1 } d ) } { \| \dot { p } \| + \varepsilon _ { 0 } } , \stackrel { \smile } { c ( p , \dot { p } ) } \stackrel { \star } { = } \frac { c _ { 0 } ( 1 - t a n h ( c _ { 1 } d ) ) } { \| \dot { p } \| + \varepsilon _ { 0 } } } \end{array}$ and $d \ = \ \left\| p - p _ { d } \right\|$ is the distance between the current position and the desired path. Here, $k _ { 0 }$ and $c _ { 0 }$ are scalar parameters used to adjust the magnitude of the velocity field, while $k _ { 1 }$ and $c _ { 1 }$ are parameters that adjust the shape of the velocity field, and $\varepsilon _ { 0 }$ is a small constant introduced to prevent division by zero issue.

For analytical trajectories such as circles and lines, the smooth velocity field can be constructed using (5). And for demonstration-based trajectories, the velocity field can also be generated by applying (5) to fitted sampling points, with offline algorithms available to optimize computational efficiency [21].

## 3.2. Variable admittance controller

The objective of the velocity field controller is to enable the robot to follow the desired velocity and minimize deviations. Considering the interaction force $F _ { h }$ between the patient and the robot, an admittance control model is employed to determine the reference velocity, thereby enabling more natural movement and enhancing patient engagement during the training process (see Fig. 3).

$$
M _ {d} (\dot {v} _ {r e f} - \dot {v} _ {d e s}) + B _ {d} (v _ {r e f} - v _ {d e s}) = F _ {h}\tag{6}
$$

where $v _ { r e f }$ is the reference velocity of the robot processed by the admittance model. The $M _ { d }$ and $B _ { d }$ represent the user-defined inertia and damping parameters, respectively, which are used to adjust the robot’s admittance characteristics. Both $M _ { d }$ and $B _ { d }$ are designed as positive-definite diagonal matrices to simplify the calculations. The damping parameter $B _ { d }$ is dynamically adjusted in real-time according to (7), enabling minimal intervention for small velocity errors and providing strong support for large errors, thereby ensuring the AAN property during rehabilitation training [22,23].

$$
B _ {d} = \frac {\bar {B} _ {d}}{1 + \exp (- \rho_ {B} \left\| e _ {v} \right\|)}\tag{7}
$$

where $e _ { v } = \dot { p } - v _ { r e f }$ denotes the velocity error, �<sup>̄</sup> is the upper bound of the damping parameter, and $\rho _ { B }$ represents a user-adjustable scaling factor. Larger values of �<sup>̄</sup> and $\rho _ { B }$ result in a greater and faster controller reaction to velocity tracking errors.

Due to the uncertainties in the system and the challenges in modeling, many studies typically use Radial Basis Function Neural Network (RBFNN) to approximate the unknown dynamic components of the robot [6,24]. The RBFNN function can be represented over a compact set $\varOmega _ { N }$ as follows:

$$
F _ {c o m p} (z) = W ^ {T} \sigma (z) + \delta (z)\tag{8}
$$

where $F _ { c o m p } ( z ) ~ \in ~ \mathbb { R } ^ { m }$ is defined as $F _ { c o m p } ~ = ~ - C _ { p } ( p , \dot { p } ) v _ { r e f } - D ( \dot { p } ) ~ -$ $M _ { p } ( p ) \dot { v } _ { r e f }$ with � denoting the dimension of the force vector. And $\sigma \in \mathbb { R } ^ { l }$ is the activation vector function with � denoting the number of neurons, $\boldsymbol { z } ~ = ~ [ p , \dot { p } , v _ { r e f } , \dot { v } _ { r e f } ]$ is the input vector of the network, $W ~ \in ~ \mathbb { R } ^ { l \times m }$ is the bounded ideal weight matrix of the network as $\| W \| _ { F } ^ { 2 } < \bar { W } _ { \mathrm { : } }$ , and $\delta ( z )$ is the functional fitting error. During the training task, the estimated weight matrix of the RBFNN is adaptively updated according to (9), while simultaneously computing $F _ { c o m p }$ in real-time to provide feedforward compensation for the robotic system.

![](images/e30715f75088ae6fc3448c1de2736c092e181dceed8f78a258309d7c2aaa899a.jpg)  
Fig. 3. Block diagram of the proposed controller.

$$
\dot {\hat {W}} = \left\{ \begin{array}{l l} \phi - \frac {t r (\hat {W} ^ {T} \phi)}{t r (\hat {W} ^ {T} \hat {W})} \hat {W}, i f t r (\hat {W} ^ {T} \hat {W}) > \bar {W} \\ \qquad \qquad \qquad \text {and} t r (\hat {W} ^ {T} \phi) > 0 \\ \phi , o t h e r w i s e \end{array} \right.\tag{9}
$$

where $\phi = \gamma \sigma ( z ) e _ { v } ^ { T }$ and $\gamma \in \mathbb { R } ^ { + }$ is adjustable positive constants. And (9) ensures the boundedness of the estimated weight matrix $\hat { W }$ through the projection operator.

To achieve stable velocity control and ensure the convergence of $e _ { v } ,$ h b l l d d f ll

$$
\boldsymbol {\tau} = - \boldsymbol {J} _ {r} ^ {T} (\boldsymbol {K} _ {r} t a n h (\rho \boldsymbol {e} _ {v}) + \hat {\boldsymbol {F}} _ {c o m p})\tag{10}
$$

where $\hat { F } _ { c o m p } = \hat { W } ^ { T } \sigma ( z ) \in \mathbb { R } ^ { m }$ is the estimate of $F _ { c o m p }$ through RBFNN, in which �<sup>̂</sup> is the estimate of the ideal weight matrix �. $K _ { r } \in \mathbb { R } ^ { l \times l }$ is a diagonal positive-definite matrix, and the ���ℎ(⋅) is used in the controller to have a saturated proportional-type feedback term. $\rho$ is a scaling factor that determines the controller’s sensitivity to velocity errors—larger values of $\dot { } \rho$ result in a stronger response to deviations. In this research, $\rho = 1 0$ was selected based on empirical tuning.

For clarity of presentation, the estimation error of the weight matrix is defined as $\tilde { W } = W - \hat { W }$ , while the robot dynamics estimation error is given by $F _ { c o m p } - \hat { F } _ { c o m p } = \hat { W } ^ { T } \sigma ( z ) + \varepsilon .$ . And substituting (10) into $( 4 ) ,$ the closed-loop dynamics of the velocity error $e _ { v }$ can be written as follows:

$$
M _ {p} (p) \dot {e} _ {v} = - C _ {p} (p, \dot {p}) e _ {v} - K _ {r} t a n h (\rho (e _ {v})) + \tilde {W} ^ {T} \sigma + \varepsilon + F _ {h}\tag{11}
$$

The stability of the closed-loop system (11) can be proved using Lya punov theory. First, consider the Lyapunov function $\begin{array} { r } { V = \frac { 1 } { 2 } e _ { v } ^ { T } M _ { p } ( p ) e _ { v } + } \end{array}$ ${ \textstyle \frac { 1 } { 7 } } t r ( \tilde { W } ^ { T } { \cal T } ^ { - 1 } \tilde { W } ) _ { : }$ , where ${ \boldsymbol { { r } } } = \gamma { \boldsymbol { { I } } } \in R ^ { l \times l }$ is a diagonal matrix. Then the time derivative of � can then be derived as:

$$
\dot {V} = e _ {v} ^ {T} M _ {p} (p) \dot {e} _ {v} + \frac {1}{2} e _ {v} ^ {T} \dot {M} _ {p} (p) e _ {v} + t r (\tilde {W} ^ {T} \Gamma^ {- 1} \tilde {W})\tag{12}
$$

Assuming that the Jacobian matrix remains nonsingular, then the robot dynamic model maintains the skew-symmetric property of $\dot { M } _ { p } ( p ) -$ $2 C _ { p } ( p , { \dot { p } } ) ,$ which implies:

$$
\frac {1}{2} e _ {v} ^ {T} \dot {M} _ {p} (p) e _ {v} = e _ {v} ^ {T} C _ {p} (p, \dot {p}) e _ {v}\tag{13}
$$

Substituting (11) and (13) into (12), we obtain the modified expression:

$$
\dot {V} = - e _ {v} ^ {T} K _ {r} \tanh (\rho e _ {v}) + e _ {v} ^ {T} (\varepsilon + F _ {h}) + e _ {v} ^ {T} \tilde {W} ^ {T} \sigma + t r (\tilde {W} ^ {T} \Gamma \tilde {W})\tag{14}
$$

Furthermore, since $\dot { \tilde { W } } = - \dot { \hat { W } } = - { \cal { \Gamma } } \sigma e _ { v } ^ { T } ,$ we have $t r ( \tilde { W } ^ { T } \bar { I } ^ { - 1 } \dot { \tilde { W } } ) =$ $- t r ( \tilde { W } ^ { T } \sigma e _ { { \scriptscriptstyle { \cal { v } } } } ^ { T } ) = - e _ { { \scriptscriptstyle { \cal { v } } } } ^ { T } \tilde { W } ^ { T } \sigma$ in (14). Under the boundedness assumptions ‖�‖ $\leq \bar { \varepsilon }$ and $\left\| F _ { h } \right\| \leq { \bar { F } } _ { h }$ , we can establish the following inequality:

$$
\dot {V} = - e _ {v} ^ {T} K _ {r} \tanh (\rho e _ {v}) + e _ {v} ^ {T} (\varepsilon + F _ {h})
$$

$$
\leq - K _ {r} \tanh (\rho e _ {v}) \left\| e _ {v} \right\| + (\bar {\varepsilon} + \bar {F} _ {h}) \left\| e _ {v} \right\|\tag{15}
$$

And by considering Lemma 1, $K _ { r } = \mathrm { d i a g } ( k _ { 1 } , k _ { 2 } )$ and define $k _ { \mathrm { m i n } } =$ min $\{ k _ { 1 } , k _ { 2 } \}$ , the (15) can be rewritten as:

$$
\dot {V} \leq - (k _ {\mathrm{min}} - \bar {\varepsilon} - \bar {F} _ {h}) \left\| e _ {v} \right\| + 0. 2 7 8 5 (\frac {k _ {1} + k _ {2}}{\rho})\tag{16}
$$

Note that if $\begin{array} { r } { \left\| \boldsymbol { e } _ { v } \right\| > \frac { 0 . 2 7 8 5 ( k _ { 1 } + k _ { 2 } ) } { \rho ( k _ { \mathrm { m i n } } - \bar { \varepsilon } - \bar { F } _ { h } ) } } \end{array}$ , then $\dot { V } \leq 0 .$ Consequently, there exists a finite time � that satisfies $\begin{array} { r } { \left\| e _ { v } \right\| \le \frac { 0 . 2 7 8 5 ( k _ { 1 } + k _ { 2 } ) } { \rho ( k _ { \operatorname* { m i n } } - \bar { \varepsilon } - \bar { F } _ { h } ) } } \end{array}$ for $\forall t > T$

Lemma 1. For $\forall e _ { v }$ and $\forall \rho > 0 ,$ it holds that $\begin{array} { r } { 0 < \| e _ { v } \| { - e _ { v } t a n h ( \rho e _ { v } ) } \leq \frac { 0 . 2 7 8 5 } { \rho } } \end{array}$ [25].

## 4. Spasticity detection and velocity switching

Currently, numerous sEMG-based spasticity detection methods pre dominantly employ threshold-based approaches [26,27](e.g., single threshold method, double threshold method) or statistical techniques [16] (e.g., Approximated Generalized Likelihood Ratio method and Cumulative Sum method). While these methods are relatively simple to implement, they are susceptible to noise interference and often prove inadequate for accurately detecting spasticity time instants. To enhance the accuracy of spasticity detection, various machine learning approaches have been proposed in recent years. Among these, the LSTM network demonstrates significant advantages in identifying temporal signal patterns, as it can maintain states activated by short-term events to form long-term memory [28]. Given the time-series characteristics of sEMG signals, LSTM network is particularly suitable for identifying muscle activity time instants without requiring heuristic feature selection and extraction from sEMG signals. In this research, we developed an LSTM-based classifier for automatic learning muscle activation features, thereby enabling real-time distinction between muscle spasticity and normal rehabilitation movements.

![](images/d758d4283aa5a03820c08c3bba0a9bb080fa07cd1f86354dbb513a1da0618871.jpg)  
(a)

![](images/c20c0d3cb0ad35a52f102e4348f24c107235734dbd8903a98169bcaaf257ac29.jpg)  
(b)  
Fig. 4. Functions used for spasticity detection and velocity switching. (a) Architecture of LSTM network. (b) Velocity smooth transition function.

Post-stroke upper limb spasticity predominantly manifests as elbow spasticity [29]. Therefore, in this research, we employed a dual-channel sEMG acquisition system to record elbow muscle activity from the BB and TB at a sampling rate of 2148 Hz. The raw sEMG signals were processed using a 4th-order Butterworth bandpass filter with cutoff frequencies of 20 Hz and 450 Hz to remove signal noise, then normalized to the peak sEMG recorded during the pre-experiment calibration. The processed signals were subsequently segmented using a 200 ms sliding window with 50 ms step size [30,31]. These processed sEMG data were utilized for both training and prediction in our LSTM model.

In our previous work [32], six post-stroke patients with spasticity were recruited to collect sEMG signals during both active and passive elbow flexion–extension exercises. For each patient, approximately 40–50 s of BB and TB signals were recorded. The acquired signals were then classified and labeled by experienced physicians into spasticity and non-spasticity categories. After filtering and segmentation, 17 consecutive overlapping windows were combined to form 1-second sample, resulting in a total of 6720 labeled samples. The current research utilizes this clinically validated sEMG dataset for training the LSTM-based spasticity detection model.

The architecture of the LSTM model is illustrated in Fig. 4(a), which comprises three fundamental components: the input gate, forget gate, and output gate:

$$
i _ {t} = s (W _ {i} [ h _ {t - 1}, x _ {t} ] + b _ {i})\tag{17}
$$

$$
f _ {t} = s (W _ {f} [ h _ {t - 1}, x _ {t} ] + b _ {f})\tag{18}
$$

$$
o _ {t} = s (W _ {o} [ h _ {t - 1}, x _ {t} ] + b _ {o})\tag{19}
$$

where $W _ { i } / b _ { i } , \ : W _ { f } / b _ { f } , \ : W _ { o } / b _ { o }$ denote the weight and bias for the input gate, forget gate, and output gate, respectively. The activation function � represents the sigmoid function, while $x _ { t }$ and $h _ { t - 1 }$ correspond to the current input and previous hidden state, respectively.

The input $x _ { t }$ consists of batched sEMG signal sequences segmented using the aforementioned method, with each sequence containing 200 time steps and 2 features representing the BT and TT channels. The hidden state $h _ { t }$ is updated through an intermediate variable $c _ { t } ,$ commonly referred to as the cell state:

$$
\left\{ \begin{array}{l l} \tilde {c} _ {t} = g (W _ {c} [ h _ {t - 1}, x _ {t} ] + b _ {c}) \\ c _ {t} = f _ {t} \cdot c _ {t - 1} + i _ {t} \cdot \tilde {c} _ {t} \end{array} \right.\tag{20}
$$

Table 1  
Clinical characteristics of post-stroke patients

<table><tr><td>ID</td><td>Age/Sex</td><td>IS</td><td>BRS</td><td>MAS</td></tr><tr><td>P1</td><td>75/M</td><td>Right</td><td>IV</td><td>0</td></tr><tr><td>P2</td><td>46/M</td><td>Right</td><td>III</td><td>1+</td></tr><tr><td>P3</td><td>49/M</td><td>Right</td><td>VI</td><td>0</td></tr><tr><td>P4</td><td>60/M</td><td>Right</td><td>VI</td><td>0</td></tr><tr><td>P5</td><td>63/M</td><td>Left</td><td>IV</td><td>1</td></tr><tr><td>P6</td><td>58/M</td><td>Left</td><td>V</td><td>0</td></tr><tr><td>P7</td><td>47/M</td><td>Left</td><td>III</td><td>1+</td></tr><tr><td>P8</td><td>45/F</td><td>Right</td><td>VI</td><td>1+</td></tr><tr><td>P9</td><td>51/F</td><td>Right</td><td>VI</td><td>0</td></tr><tr><td>P10</td><td>72/F</td><td>Left</td><td>VI</td><td>0</td></tr></table>

IS=Impaired Side, BRS=Brunnstrom Recovery Stage, MAS=Modified Ashworth Scale

where $W _ { c }$ and $b _ { c }$ represent the weight and bias, and $g$ denotes the hyperbolic tangent function that transforms the state values into the range of −1 to 1. Then the hidden state can be computed from $c _ { t }$ and $o _ { t }$ as follows:

$$
h _ {t} = o _ {t} \cdot g (c _ {t})\tag{21}
$$

The final output of the LSTM network is processed through a fully connected layer to generate predictions for spasticity states. Specifically, the LSTM network layer contains 200 LSTM cells, and the model optimization is performed using the Adam optimizer with a learning rate of 0.01. The classification probability distribution is determined by the output layer, which consists of a softmax layer followed by a classification layer. Based on this LSTM model, the patients’ sEMG dataset was divided into training and testing sets, with 80% used for training and 20% for testing. A 3-fold cross-validation strategy was applied for model training and optimization. The final spasticity detection accuracy reached 95.61%.

Relevant studies have demonstrated that stretching the upper limb during spasticity can help alleviate spasticity and promote functional recovery [29,33]. Based on clinical recommendations, we propose a switching function to transition the velocity field during spasticity state. When the system detects upper limb spasticity, the training velocity is adjusted to a slow speed to ensure safety. And this adjustment allows patients to continue performing slow, controlled stretching exercises with robotic assistance, thereby mitigating spasticity. The post-spasticity velocity field is calculated as follows:

$$
\left\{ \begin{array}{l l} & v _ {d e s} ^ {\prime} = k ^ {\prime} \hat {n} + c ^ {\prime} \hat {t} \\ & k ^ {\prime} = \omega (t) ^ {2} k \\ & c ^ {\prime} = \omega (t) ^ {2} c \\ & \omega (t) = (1 - m _ {0} \exp (- \frac {m _ {1}}{(t - t _ {s p a s})})) \\ & \text {if} m _ {0} \text {and} m _ {1} \text {for all } t \in [ 0, 1 ] \end{array} \right.\tag{22}
$$

As illustrated in Fig. 4(b), �(�) represents the switching function that smoothly transitions the velocity field from the original speed to a slower speed, $t _ { s p a s }$ denotes the time instant when spasticity is detected, $m _ { 0 } ~ \in ~ ( 0 , 1 )$ determines the magnitude of the slower speed, and $m _ { 1 }$ adjusts the rate of velocity change. In this research, $m _ { 0 } ~ = ~ 0 . 5$ and $m _ { 1 } = 0 . 1$ were selected.

h d i hi f i bl h i i i h robot’s velocity field, ensuring training safety while still providing patients with beneficial stretching movements.

## 5. Experiment

## 5.1. Experiment settings

The performance of the proposed controllers was validated through two experiments. Experiment 1 aimed to compare the proposed AAN controller with force field control (FFC) method [7] and impedancebased control (IMC) method [34]. Experiment 2 was designed to evaluate the effectiveness of the proposed spasticity detection and velocity switching methods. Ten healthy subjects (6 males and 4 females, all right-handed; age: 25.8 ± 3.1 years; labeled H1–H10) and ten post stroke patients (the clinical characteristics are summarized in Table 1; labeled P1–P10) were recruited to participate in the experiments. All subjects were fully informed of the experimental procedures and potential risks prior to participation and provided written informed consent. The study protocol was approved by the Ethics Committee of Zhongda Hospital (2022ZDSYLL397-P01). The entire experimental process was supervised by a physician to ensure safety and prevent any accidental injury.

Table 2  
Formulas and units of performance metrics used in this research.

<table><tr><td>Metric</td><td>Formula</td><td>Unit</td></tr><tr><td>TE</td><td> $\frac{1}{n}\sum_{i=1}^{n}\left\|\boldsymbol{p}_{i}-\boldsymbol{p}_{d,i}\right\|$ </td><td>cm</td></tr><tr><td>MV</td><td> $\frac{1}{n}\sum_{i=1}^{n}\left\|\dot{\boldsymbol{p}}_{i}\right\|$ </td><td>m/s</td></tr><tr><td>MS</td><td> $\frac{1}{n}\sum_{i=1}^{n}\left\|\ddot{\boldsymbol{p}}_{i}\right\|$ </td><td>m/s2</td></tr><tr><td>IF</td><td> $\frac{1}{n}\sum_{i=1}^{n}\left\|\boldsymbol{F}_{r,i}\right\|$ </td><td>N</td></tr><tr><td>SW</td><td> $\sum_{i=1}^{n}\left\|\boldsymbol{F}_{h,i}\dot{\boldsymbol{q}}_{i}\right\|\Delta t$ </td><td>J</td></tr><tr><td>MA</td><td> $\frac{1}{n}\sum_{i=1}^{n}\left(\left\|\boldsymbol{S}_{BB,i}\right\|+\left\|\boldsymbol{S}_{TB,i}\right\|\right)$ </td><td>-</td></tr></table>

As shown in Fig. 1, subjects were seated in front of a screen and on the right side of the robot to perform the task using their non dominant arm or impaired arm. Prior to the formal experiments, a 4-minute familiarization session was conducted to help subjects become accustomed to the procedures of experiments 1 and 2. In Experiment 1, ten healthy subjects and six patients without spasticity symptoms (MAS Score = 0) executed circular trajectory-tracking movements un der different training modes, with each complete circle counted as one episode. Ten episodes were conducted for each training mode, with two-minute rest intervals provided after each mode to mitigate fatigue effects. To further validate the effectiveness and clinical applicability of the proposed spasticity-detection-based velocity switching control strategy, Experiment 2 was conducted with four patients with spasticity symptoms (MAS Score = 1/1+). Experiment 2 followed a similar protocol to Experiment 1, in which the patients performed 10 episodes of circular trajectory-tracking under the VA-VFC mode. During the experiment, spasticity typically occurred when the affected limb reached certain postures, often manifesting as increased muscle tone and higher resistance to movement. In such cases, slow stretching of the affected limb is required to alleviate spasticity. In Experiment 2, this was achieved by automatically switching the controller to a lower-speed mode to assist limb stretching whenever spasticity was detected.

In order to quantitatively analyze the control performance with different control modes and experimenters, the following metrics are defined in Table 2 [3,11]: average trajectory error (TE), average movement velocity (MV), movement smoothness (MS), average interaction force (IF), subject mechanical work (SW), and muscle activation level (MA). Among these, TE, MV, and MS reflect the stability and accuracy of training task. Specifically, lower TE and MS values and higher MV indicate better control performance and task accomplishment. In contrast, IF, PW, and MA characterize the patients’ level of engagement during the training process, with higher values indicating a greater level of effort.

In Table 2, � denotes the sampling time, � represents the number of samples in a single episode, and $S _ { B B }$ and $S _ { T B }$ correspond to the normalized sEMG of the BB and TB, respectively.

## 5.2. Performance ofproposed VA-VFC method

In this experiment, we compared the performance of the proposed VA-VFC mode with two other control modes: FFC mode and IMC mode. Under all three modes, subjects were instructed to follow a circular trajectory as accurately and rapidly as possible. The motion and interaction force data during the training sessions were recorded for comparative analysis. To evaluate subjects’ active participation, only force components aligned with the movement direction $( F _ { h } \cdot \hat { t } > 0 , \cdots$ denotes the dot product) were considered as voluntary contributions.

![](images/b561d71eacfdcfa32cdf13b3d97c0f8bb72df6636b8fea060d2c1db2c0437398.jpg)  
(a)

![](images/6478fef25028f0bd9f4c0fc51e7fe600b23f19c6e80394b4dff52a9116cb14a6.jpg)  
(b)  
Fig. 5. Trajectory and velocity tracking performance of Subject H2 under VA-VFC mode during the circular trajectory task.

First, to evaluate the effectiveness of the proposed VA-VFC method in position and velocity control, Fig. 5 illustrates the trajectory and velocity tracking performance of Subject H2 under the VA-VFC mode. Fig. 5(a) presents the mean trajectory and corresponding standard deviation after the completion of Experiment 1. The results reveal that the motion trajectory under the VA-VFC mode was smooth and closely matched the desired trajectory, indicating that the VA-VFC provides excellent position control performance. As shown in Fig. 5(b), the reference and actual velocity profiles of Subject H2 in both the X and Y directions are compared. At the beginning of training, the end-effector started from rest, leading to a relatively large initial velocity error. According to (10), the robot provided substantial assistance to help the subject rapidly approach the reference velocity profile. Approximately 2 s after training began, the subject’s actual velocity closely matched the reference velocity in both directions. The mean velocity errors in the X and Y directions were $e _ { v , x } ~ { = } ~ 0 . 0 1 0 \pm 0 . 0 0 8 ~ \mathrm { m / s }$ and $\boldsymbol { e } _ { v , y } \ =$ 0.010±0.009 m∕s, respectively, demonstrating that the proposed VA-VFC method achieves effective velocity tracking.

Fig. 6 compares the performance metrics of healthy subjects under the three different training modes and presents the corresponding mean values and standard deviation for each metric. There was no significant difference in movement velocity was observed, as the controller parameters were pre-adjusted to maintain comparable movement velocities across all modes, thereby ensuring a fair evaluation of the other performance metrics. In contrast, the VA-VFC mode exhibited significantly lower average trajectory error and higher movement smoothness compared with the other two modes, indicating that this control strategy enables smoother and more accurate trajectory tracking. Conversely, the FFC mode resulted in larger trajectory error and lower smoothness, which can be attributed to the fact that it only provides force field assistance without strictly constraining the position or velocity of the movement, thereby allowing a higher motion freedom.

Furthermore, as shown in bottom row of Fig. 6, the interaction force, subject mechanical work, and muscle activation level under the VA-VFC and FFC modes were comparable and significantly higher than those in the IMC mode. This suggests that VA-VFC and FFC are more ef fective in eliciting active participation from the subjects and achieving more effective AAN. In contrast, the lower engagement level observed under the IMC mode may be due to its reliance on a time-based desired trajectory, which restricts the subjects’ freedom in movement timing.

![](images/a1479d4b01d53c5239dd478d602bd15e0a51b10c1b337c244f7540bcec57c1ff.jpg)

![](images/591a1e19434cdfcb345ec0b0c4eef9270dc7e08ed8bf25b4150730c9d74c02a4.jpg)

![](images/5448dd15ee4bbd0a1c9a617d05cceb951a7f81700bee108ac4c9ad5438b6baec.jpg)

![](images/b2b87317c8be7316d34eff830885ab5bd219fdccb9647d0d37447a3b8f449a9d.jpg)

![](images/0df986a8a68f3538a5fe4dc052618507276b044cf955127e53e6e9f70258be12.jpg)

![](images/972b4d8cbad6c37dc571801b20dc1cffa36082f56ff9af2d54626c23f501bb97.jpg)  
Fig. 6. Comparison of healthy subjects’ performance metrics under three training modes (‘ns’: No Significance; $\stackrel { \scriptscriptstyle < \ast , } { \cdot } p < 0 . 0 5 )$

![](images/4cf0d7cdfc5fb8d69f27ff83f5544ce81933dc6e6262911ccaa7cc5627368e26.jpg)

![](images/401f8f2d927d8991be1c7428cdab814e0a5a9b11d11f2222dfcea46aa3fe267b.jpg)

![](images/b11e8ee1d8091c553f2f348f832d3f65b89f344215a6f6b7f070a08bfc07a90d.jpg)

![](images/6a0e7e4d4fc0aa808e57bd47dddbca875cccb96f77cbcc3f0415e968cfb3ca7c.jpg)

![](images/aee422ded10e0343d42dff9cb69d7dbe8ba5bc5ecec6a5c2b73b025d499a1742.jpg)

![](images/de529def0b4c94befe0454f864cdc6ed28d104e2477144291f4f3a07beb2cec5.jpg)  
Fig. 7. Comparison of post-stroke patients’ performance metrics under three training modes (‘ns’: No Significance; $\stackrel { \scriptscriptstyle < \ast , } { \cdot } p < 0 . 0 5 )$

Similarly, Fig. 7 presents a comparison of the performance metrics of post-stroke patients under the three training modes. As shown in the top row of Fig. 7, there is no significant difference between the VA-VFC and IMC modes in terms of trajectory error, movement velocity, and movement smoothness. However, both modes yield substantially better performance than the FFC mode, indicating that VA-VFC and IMC enable smoother movements, higher trajectory tracking accuracy, and better velocity stability. This finding is consistent with the results obtained from healthy subjects in Fig. 6. In addition, the average movement velocity of patients under FFC mode is considerably lower than that of healthy subjects, primarily because patients exhibit reduced upper limb motor ability and often rely on robotic assistance to complete the motion. In such cases, the FFC mode cannot effectively ensure consistent velocity tracking.

In the bottom row of Fig. 7, it can be observed that the VA-VFC mode results in significantly higher interaction force, subject mechanical work, and muscle activation level compared to the IMC mode, suggesting that VA-VFC more effectively promotes active patient participation. This observation is also consistent with the results in Fig. 6 for healthy subjects. Moreover, the subject mechanical work and muscle activation in VA-VFC are not significantly different from those in the FFC mode, indicating that both modes are similarly effective in facilitating active engagement.

The experimental results from both healthy subjects and patients demonstrate that the proposed VA-VFC method effectively combines the advantages of both FFC (enhancing patient engagement) and IMC (smooth and accurate trajectory tracking) methods. Additionally, by providing temporal flexibility during motion, VA-VFC mode enables a more natural and effective AAN.

## 5.3. Performance of velocity switching based on spasticity detection

In this experiment, the proposed LSTM-based spasticity detection method and the subsequent velocity-switching strategy were validated. All four post-stroke patients with upper-limb spasticity exhibited spasticity during the training, and the system successfully detected the spasticity in every case. The detection and switching process for Subject P7 is illustrated in Fig. 8. As shown in Fig. 8(a), the BB and TB sEMG signals of Subject P7 during the third episode exhibited a sudden increase approximately 4.5 s after the onset of movement. This abrupt signal change was identified by the LSTM-based detection method (indicated by the vertical black dashed line). Immediately after detection, the controller triggered the velocity switching function described in (22), enabling a smooth transition to the low-velocity mode, as shown in Fig. 8(b). This adaptive adjustment effectively alleviated muscle spasticity and improved joint range of motion, ensuring the safety and stability of the rehabilitation training process.

Furthermore, Fig. 8(c) illustrates the variations in robot-assistive force $F _ { r }$ and human interaction force $F _ { h }$ before and after spasticity onset. It can be observed that, following the onset of spasticity, the patient was no longer able to generate active movement force, prompting the robot to increase its assistive force to help complete the training task. Consequently, the training mode transitioned smoothly from AAN training to robot-assisted passive training. Additionally, Fig. 8(c) indicates that relying solely on interaction force to detect spasticity is challenging, as such mechanical responses often exhibit a noticeable delay. Therefore, compared with biomechanical approaches [12,15], the proposed sEMG-based spasticity detection method provides superior accuracy and real-time performance in spasticity detection.

## 6. Conclusion

This paper presents a VA-VFC method for post-stroke upper limb AAN rehabilitation training, which simultaneously enables spasticity detection based on sEMG signals and adjusts the velocity field to ensure training safety. To provide freedom in movement timing and maximize subjects’ active participation during training, we proposed a novel VFC controller. The main idea was to define the motion control problem in the velocity domain with a variable velocity range, while incorporating a variable admittance strategy to enhance AAN performance. And the stability of the proposed controller was theoretically proven using the Lyapunov function. In addition, to address safety concerns arising from spasticity during rehabilitation, an sEMG-based spasticity detection method was developed. By collecting sEMG signals from the BB and TB, and leveraging a LSTM network, the system can detect the onset of spasticity and subsequently perform a smooth velocity transition in the controller to ensure safe and stable rehabilitation training.

![](images/0c79ce9f525176c63583a47f2450f6cbddfd89c263d640f542d73486a56eaa70.jpg)

(a)  
![](images/20b796d090ab384e524c207e9e09f2baff087ad099e6588e3f460c3b2c6e2807.jpg)  
(b)

![](images/57022406b02dbab9fd34f45cdb4675fc5440d1874d6009f2a038738155c6aae7.jpg)  
(c)  
Fig. 8. sEMG, velocity, and interaction force performance of Subject P7 before and after the onset of spasticity. The vertical black dashed line indicates the time point at which spasticity was detected by the LSTM model. (a) sEMG signals of the BB and TB. (b) Reference velocity $v _ { r e f }$ and actual velocity ‖ ̇�‖. (c) Robot-assistive force $F _ { r }$ and human interaction force $F _ { h } .$

The proposed VA-VFC method was implemented on a planar upperlimb rehabilitation robotic system and evaluated in experiments in volving ten healthy subjects and ten post-stroke patients. Experimental results demonstrate that the controller achieves stable velocity control, promotes active participation while maintaining high trajectory track ing accuracy, and enables smooth velocity transitions in response to detected spasticity without significant increases in tracking error.

Overall, this research provides preliminary evidence that the proposed controller effectively achieves velocity modulation and AAN assistance, while offering a viable solution to spasticity-related chal lenges in rehabilitation. Nonetheless, several limitations should be acknowledged. The current spasticity sEMG dataset is relatively small, and the experiments were conducted under simplified training condi tions, which may limit the generalizability of the results. Furthermore, the long-term clinical effects of the proposed approach have yet to be established. Future studies will aim to expand the dataset to a broader and more representative patient cohort, implement the proposed framework in activities of daily living (ADL) tasks, and perform longitudinal evaluations to assess its sustained clinical efficacy.

Table 3  
Values and units of velocity field controller parameters.

<table><tr><td>Parameter</td><td>Value</td><td>Unit</td></tr><tr><td> $k_0$ </td><td>0.01</td><td> $m^2/s^2$ </td></tr><tr><td> $c_0$ </td><td>0.02</td><td> $m^2/s^2$ </td></tr><tr><td> $k_1$ </td><td>10</td><td> $m^{-1}$ </td></tr><tr><td> $c_1$ </td><td>10</td><td> $m^{-1}$ </td></tr><tr><td> $ε_0$ </td><td>0.002</td><td>m/s</td></tr><tr><td> $M_d$ </td><td>diag(0.5, 0.5)</td><td>kg</td></tr><tr><td> $\bar{B}_d$ </td><td>diag(40, 40)</td><td>Ns/m</td></tr><tr><td> $ρ_B$ </td><td>25</td><td>-</td></tr></table>

Table 4  
Different sets of admittance parameters.

<table><tr><td>Classification</td><td>M(kg)</td><td> $\bar{B}$ (Ns/m)</td></tr><tr><td>Small</td><td>diag(0.5,0.5)</td><td>diag(10,10)</td></tr><tr><td>Medium</td><td>diag(1.0,1.0)</td><td>diag(40,40)</td></tr><tr><td>Large</td><td>diag(1.5,1.5)</td><td>diag(70,70)</td></tr></table>

## CRediT authorship contribution statement

Ye Li: Writing – original draft, Validation, Software, Methodology, Conceptualization. Aiguo Song: Supervision, Project administration. Jianwei Lai: Writing – review & editing, Investigation, Formal analysis. Ye Lu: Writing – review & editing, Data curation. Huijun Li: Supervision.

## Ethics statement

The studies involving human participants were reviewed and approved by the Ethics Committee of Zhongda Hospital (2022ZDSYLL397- P01). The author provided the ethical approval document for this research.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgments

This research was supported by the Jiangsu Province Key Research and Development Program under Grant BE2023023-1, and the National Natural Science Foundation of China under Grants 623B2025.

## Appendix A. Analysis of controller parameter performance in velocity field control

The parameters of the VA-VFC presented in this research were determined through simulation analysis and preliminary experimental tests, with their corresponding values summarized in Table 3.

The parameters $k _ { 0 } , \ c _ { 0 } , \ k _ { 1 }$ and $c _ { 1 }$ listed in Table 3 play a critical role in shaping the velocity field. Fig. 9 illustrates the effects of varying these weight parameters on the resulting velocity field. As illustrated, $k _ { 0 }$ and $c _ { 0 }$ primarily affect the magnitude of the velocity field, directly influencing the overall movement velocity. In contrast, $k _ { 1 }$ and $c _ { 1 }$ modulate the sensitivity of the velocity field to trajectory errors, thereby determining how strongly the controller responds to deviations from the desired path.

![](images/337053508c69a6d3a4392723a8e0e04bd9ed533024837565456fbd47d6ac1796.jpg)

![](images/f498f840e6c104e33c86a1e3c6257acbeaf459a1194a44a604590af978b902e2.jpg)

![](images/9d1df1d2ecb2cdfd7ad1ff3f97436aa87f155df571e1765f060f8d5a0d709207.jpg)  
X(m)

![](images/2e871fa4cc755d547b8e37660b62592937d03bc1e8f5290696866a279561238e.jpg)  
Fig. 9. Influence of weight parameters on velocity-field construction (with $\varepsilon _ { 0 } = 0 . 0 0 5 ~ \mathrm { m / s ) }$

![](images/0f0f8f8f6ccf941fc5c41a1c1d07bd5d0005d8c7f730f8a6dfd8321846f3533d.jpg)

![](images/f1d86d76fb2ee369c6ebb166bfda019bf6adf5eb10a621616ba7bc227c26bc6c.jpg)

![](images/b0283c76b64cc6333a438851dbdaf583c8263579c91e454cae4fea5248913842.jpg)

![](images/7251634d6849e2486386f68ee18991f1eae9c08f443cda38b18816fd21296147.jpg)  
Fig. 10. Trajectory, velocity, and interaction force under different admittance settings.

To analyze the effects of admittance characteristics on control performance, three different sets of admittance parameters (small, medium, and large admittance parameters) were applied in a pilot experiment with Subject H2 under the VA-VFC mode, the corresponding parameter values are listed in Table 4. For quantitative assessment, trajectory error, movement velocity, and interaction force were compared across the three settings, with results summarized in Fig. 10 and Table 5.

![](images/09371edfd0c9e4eba4f749b45e98ed9ec1f8b375ae89a937dae3a4502ab02349.jpg)  
Fig. 11. Rectangular trajectory reaching task with each side is 0.1 m, black arrows indicate the movement direction.

![](images/2d2fcc0ccd7e876b65b7db67c2db7597efd7ec1f33648d0a8e015d48e7fd3545.jpg)  
(a)

![](images/12c0340595f722b60939675130bfa97c222d8bd01b822841e1968d6c2599b9b6.jpg)  
(b)  
Fig. 12. Trajectory and velocity tracking performance of Subject P4 under VA-VFC mode during the rectangular trajectory task.

Table 5  
Mean and standard deviation of trajectory error, movement velocity, and interaction force under different admittance settings.

<table><tr><td>Classification</td><td>TE (cm)</td><td>MV (m/s)</td><td>IF (N)</td></tr><tr><td>Small</td><td>1.171 ± 0.651</td><td>0.096 ± 0.014</td><td>2.651 ± 0.362</td></tr><tr><td>Medium</td><td>1.036 ± 0.586</td><td>0.086 ± 0.011</td><td>2.548 ± 0.611</td></tr><tr><td>Large</td><td>0.862 ± 0.460</td><td>0.090 ± 0.010</td><td>2.072 ± 1.084</td></tr></table>

As shown in Fig. 10, the proposed controller effectively assisted the subject in completing the circular trajectory task under all three admittance settings. Moreover, both trajectory error and interaction force decreased as the admittance parameters increased. This is because higher admittance leads to a gradual shift of control dominance from the human to the robot. In contrast, the movement velocity remained relatively stable across different admittance levels, demonstrating the robustness of the proposed controller in maintaining consistent velocity control. Considering the trade-off between trajectory accuracy and interaction force, the medium admittance parameters were ultimately selected for the experiments.

## Appendix B. Rectangular trajectory reaching task

To further validate the feasibility of the proposed VA-VFC method in different scenarios, a rectangular trajectory reaching task was designed, as illustrated in Fig. 11. In the virtual environment, mushroom appeared sequentially at the corners of a rectangular path, and the subjects were required to grasp them as quickly and accurately as possible with the assistance of the controller.

Fig. 12(a) presents the trajectory and velocity tracking performance of Subject P4 under the VFC mode during the rectangular trajectory task. As shown in Fig. 12(a), the proposed VA-VFC method effectively assisted the patient in accurately following the desired rectangular path, even in the presence of sharp corner transitions. The resulting trajectory tracking error was $0 . 0 0 9 \pm 0 . 0 0 4 2$ m, indicating high spatial accuracy throughout the motion.

Furthermore, Fig. 12(b) compares the reference and actual velocity profiles of Subject P4 along both the X and Y directions. The actual velocity closely followed the reference velocity in both directions, with mean velocity errors of $e _ { v , x } = 0 . 0 1 3 \pm 0 . 0 1 0 ~ \mathrm { m / s }$ and $e _ { v , y } = 0 . 0 1 2 \pm 0 . 0 0 9$ $\mathrm { m } / \mathrm { s } ,$ respectively. These results demonstrate that the proposed VA VFC method achieves smooth and precise velocity control, confirming its robustness and adaptability in complex trajectory tracking scenarios.

## Data availability

Data will be made available on request.

## References

[1] Maeshima S, Osawa A, Mi azaki Y, Seki Y, Miura C, Tazawa Y, Tanahashi N. Influence of dysphagia on short-term outcome in patients with acute stroke. Am J Phys Med Rehabil 2011;90(4):316–20.

[2] Luo L, Peng L, Wang C, Hou Z-G. A greedy assist-as-needed controller for upper limb rehabilitation. IEEE Trans Neural Netw Learn Syst 2019;30(11):3433–43.

[3] Li X, Zeng H, Zhang J, Song A. Engagement enhancement based on Bayesian optimization for adaptive assist-as-needed controller. IEEE Robot Autom Lett 2022:7(1):49–56.

[4] Sommerhalder M, Zimmermann Y, Song J, Riener R, Wolf P. Polymorphic control f k f d d d d l d b d h b l Trans Robot. 2023:40:298–315

[5] Pezeshki L, Sadeghian H, Keshmiri M, Chen X, Haddadin S. Cooperative assistas-needed control for robotic rehabilitation: A two-player game approach. IEEE Robot, Autom Lett 2023:8(5):2852–9.

[6] Pehlivan AU, Losey DP, O’Malley MK. Minimal assist-as-needed controller for upper limb robotic rehabilitation. IEEE Trans Robot 2015;32(1):113–24.

[7] Pan J, Zhang L, Sun Q. Development of a force-field-based control strategy for an upper-limb rehabilitation robot. Mech Sci 2022;13(2):949–59.

[8] Rossa C, Najafi M, Tavakoli M, Adams K. Robotic rehabilitation and assistance for individuals with movement disorders based on a kinematic model of the upper limb. JEEE Trans Med Robot Bionics 2021:3(1):190–203.

[9] Najafi M, Rossa C, Adams K, Tavakoli M. Using potential field function with a velocity field controller to learn and reproduce the therapist’s assistance in robot-assisted rehabilitation. JEEE Trans Mechatronics 2020:25(3):1622–33.

[10] Guo Y, Wang H, Tian Y, Caldwell DG. Task performance-based adaptive velocity assist-as-needed control for an upper limb exoskeleton. Biomed Signal Process Control 2022;73:103474.

[11] Garcia-Hernandez N, Munguia-Angeles C, Parra-Vega V. Assist-as-needed roboti strategy based on velocity fields for enhancing motor training. IEEE Trans Mechatronics 2024

[12] Guo X, Wallace R, Tan Y, Oetomo D, Klaic M, Crocher V. Technologyassisted assessment of spasticity: a systematic review. J NeuroEng. Rehabi 2022:19(1):138.

[13] Chalard A. Amarantini D. Tissevre J. Marque P. Tallet J. Gasq D. Spastic co-contraction, rather that spasticity, is associated with impaired active function in adults with acquired brain injury: a pilot study. J Rehabil Med 2019:51(4):307–11.

[14] Hu B. Zhang X. Mu J. Wu M. Zhu Z. Liu Z. Wang Y. Spasticity measurement based on the HHT marginal spectrum entropy of sEMG using a portable system: A preliminary study. IEEE Trans Neural Syst Rehabil Eng 2018;26(7):1424–34.

[15] Trigili E, Crea S, Moisè M, Baldoni A, Cempini M, Ercolini G, Marconi D, Posteraro E. Carrozza MC. Vitiello N. Design and experimental characterization of a shoulder-elbow exoskeleton with compliant joints for post-stroke rehabilitation. JEEE Trans Mechatronics 2019:24(4):1485–96

[16] Tigrini A, Mengarelli A, Cardarelli S, Fioretti S, Verdini F. Improving EMG signal change point detection for low SNR by using extended Teager-Kaiser energy operator. IEEE Trans Med Robot Bionics 2020;PP(99)

[17] Asl HJ, Narikiyo T. An assistive control strategy for rehabilitation robots using velocity field and force field, In: Proc. IEEE int, conf, rehabil, robot., JEEE: 2019 p. 790–5.

[18] Shi D, Li L, Zhang W, Ding X. Field-based human-centred control on so (3) for assist-as-needed robotic rehabilitation. IEEE Trans Med Robot Bionics 2022:4(3):785–95.

[19] Fan Y, Zhu L, Wang H, Song A. Synthesize personalized training for robot-assisted upper limb rehabilitation with diversity enhancement. IEEE Trans Vis Comput Graphics 2023;30(8):5705–18.

[20] Cervantes I, Kelly R, Alvarez-Ramirez J, Moreno J. A robust velocity field control. IEEE Trans Control Syst Technol 2002;10(6):888–94.

[21] Cheng M-Y, Wang Y-H. Velocity field construction for contour following tasks represented in NURBS form. IEEE Trans Autom Control 2009;54(10):2405–10.

[22] Asl HJ, Yamashita M, Narikiyo T, Kawanishi M. Field-based assist-asneeded control schemes for rehabilitation robots. IEEE Trans Mechatronic 2020;25(4):2100–11.

[ ] h ifi k i h h bi h h k li Impedance variation and learning strategies in human–robot interaction. IEEE T. Cybern 2021:52(7):6462–75.

[24] He W, Xue C, Yu X, Li Z, Yang C. Admittance-based controller design for physical human–robot interaction in the constrained task space. IEEE Trans Autom Sci Eng 2020:17(4):1937-49

[25] Liu S, Liu Y, Wang N. Nonlinear disturbance observer-based backstepping f l d d k l f d h l h uncertainties and external disturbances. Nonlinear Dynam 2017:88(1):465–76.

[26] Merlo, Andrea, Farina, Dario, Merletti, Roberto. A fast and reliable technique for muscle activity detection from surface EMG signals.. IEEE Trans Biomed Eng 2003.

[27] Mccool P, Chatlani N, Petropoulakis L, Soraghan JJ, Menon R, Lakany H. Lower arm electromyography (EMG) activity detection using local binary patterns. IEEE Trans Neural Syst Rehabil Eng 2014;22(5):1003–12.

[28] Wang C, Peng L, Hou Z-G, Zhang P, Fang P. An easy-to-use assessment system for spasticity severity quantification in post-stroke rehabilitation. IEEE Trans Cogn Dey Syst 2023:16(3):828–39.

[29] Pilla A, Trigili E, McKinney Z, Fanciullacci C, Malasoma C, Posteraro F, Crea S, Vitiello N Robotic rehabilitation and multimodal instrumented assessment, of post-stroke elbow motor functions—A randomized controlled trial protocol. Front Neurol 2020:11:587293.

[30] Wang W, Zhao C, Li X, Zhang Z-Q, Yuan X, Li H. Research on multimodal fusion recognition method of upper limb motion patterns. IEEE Trans Instrum Meas.2023:72

[31] Wang W, Ren H, Su S, Zhang P, Zhang J. A brain-inspired decision-making h d f l b k l b d l b d multimodal information fusion, Measurement 2025:241:115728

[32] Li Y, Song A, Huang G, Shen Y, Lu Y, Li H. Feature selection and machine l i f li li b i i d i h bil assist. technol., i-cREATe world rehabil.. IEEE; 2024, p. 1–4.

[33] Gandolfi M, Valè N, Dimitrova EK, Mazzoleni S, Battini E, Filippetti M, Picelli A, Santamato A, Gravina M, Saltuari L, et al. Effectiveness of robot-assisted upper limb training on spasticity, function and muscle activity in chronic stroke patients treated with botulinum toxin: a randomized single-blinded controlled trial. Front Neurol 2019:10:41.

[34] Bai J, Song A, Wang T, Li H. A novel backstepping adaptive impedance control for an upper limb rehabilitation robot. Comput Electr Eng 2019;80:106465.

![](images/88612db809417a96b7e9b0e927589caaec1f841dfef94f04409803a1322423ec.jpg)  
Ye Li received his B.S. degree in measurement and control technology and instrumentation from the School of Instruments Science and Engineering, Southeast University, Nanjing, China, in 2018. He is currently working toward the Ph.D. degree in instrument science and technology with the School of Instrument Science and Engineering, Southeast University.

His research interests include rehabilitation robot and human–machine interaction.

![](images/4fb57ed7f410971e4f9a9cb6e946f70e342863eab7f50c76a61211799ee90e85.jpg)

![](images/a64349080a0dd591a99d6182f585e58f04cf222f962f18e2a6aa6d4510db7e36.jpg)

Aiguo Song (Senior Member, IEEE) received his Ph.D. degree in measurement and control from Southeast University, Nanjing, China, in 1996.

He has authored or coauthored more than 210 technical papers in the peer-refereed journals. His H-index (Googl scholar) is 39. He is currently a Professor with the School of Instrument Science and Engineering, Southeast University. His current interests concentrate on teleoperation, hapti display, mobile robotics, and rehabilitation robotics.

Jianwei Lai received his B.S. degree in automation from Northeast Forestry University, Harbin, China, in 2018. He is currently working toward the Ph.D. degree in instrument science and technology with the School of Instrument Sci ence and Engineering, Southeast University, Nanjing, China.

His current research interests include soft robot and hand rehabilitation robot

![](images/258d7ae1cb41e8c6ddb3d8f25b0f99467fe4fd2807f7b43c02a712f9ceb52bc1.jpg)

![](images/8363ce362c4b956804e025e00e7f161b07a4a91d18b80a28195a326dc2f99f45.jpg)

Ye Lu received his B.S. degree in the school of control engineering from Northeastern University at Qinhuangdao, China, in 2020. He is currently pursuing the Ph.D. degree in instrument science and technology with the School of Instrument Science and Engineering, Southeast University, Nanjing, China.

His research interests include exoskeleton robots, human–robotic interaction, and novel actuator design.

Huijun Li received her B.S. degree in instrument science and the M.S. degree in condensed matter physics from Zhengzhou University, Zhengzhou, China, in 1999 and 2002, respectively, and the Ph.D. degree in measurement and control from Southeast University, Nanjing, China, in 2005.

She is currently a Professor in the School of Instrument Science and Engineering, Southeast University. Her current research interests include teleoperation and rehabilitation robotics.