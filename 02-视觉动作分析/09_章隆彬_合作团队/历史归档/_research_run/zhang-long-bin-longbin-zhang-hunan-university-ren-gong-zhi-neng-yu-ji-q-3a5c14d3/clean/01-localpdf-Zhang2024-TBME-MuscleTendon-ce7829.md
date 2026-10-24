<!-- source: https://doi.org/10.1109/tbme.2024.3352556 -->
<!-- title: EMG-Constrained and Ultrasound-Informed Muscle-Tendon Parameter Estimation in Post-Stroke Hemiparesis -->
<!-- fetched: 2026-09-30T13:46:08Z -->
<!-- via: local_paper_fulltext(_translation_outputs/Zhang_2024_TBME_MuscleTendon/Zhang_2024_TBME_MuscleTendon/auto/Zhang_2024_TBME_MuscleTendon.md) -->
<!-- note: 原文 PDF 存于 09_章隆彬_合作团队/papers/，此为 PDF 解析后的全文 Markdown -->

Abstract—Secondary morphological and mechanical property changes in the muscle-tendon unit at the ankle joint are often observed in post-stroke individuals. These changes may alter the force generation capacity and affect daily activities such as locomotion. This work aimed to estimate subject-specific muscle-tendon parameters in individuals after stroke by solving the muscle redundancy problem using direct collocation optimal control methods based on experimental electromyography (EMG) signals and measured muscle fiber length. Subject-specific muscle-tendon parameters of the gastrocnemius, soleus, and tibialis anterior were estimated in seven post-stroke individuals and seven healthy controls. We found that the maximum isometric force, tendon stiffness and optimal fiber length in the post-stroke group were considerably lower than in the control group. We also computed the root mean square error between estimated and experimental values of muscle excitation and fiber length. The musculoskeletal model with estimated subject-specific muscle tendon parameters (from the muscle redundancy solver), yielded better muscle excitation and fiber length estimations than did scaled generic parameters. Our findings also showed that the muscle redundancy solver can estimate muscle-tendon parameters that produce force behavior in better accordance with the experimentally-measured value. These muscle-tendon parameters in the post-stroke individuals were physiologically meaningful and may shed light on treatment and/or rehabilitation planning.

Index Terms—Musculoskeletal modeling, muscle redundancy solver, direct collocation, optimization, muscle fiber length

## I. INTRODUCTION

Stroke is classically characterized as a neurological disease caused by a sudden rupture or blockage of a cerebral blood vessel of the central nervous system cells [1]. It is the secondleading cause of death and the third-leading cause of disability worldwide [2], [3]. The skeletal muscles may also experience secondary structural and functional alterations after stroke. The muscles around the ankle joint are often affected after stroke. For example, weakness in the ankle plantarflexros may not be able to counteract the external dorsiflexion moment during stance phase resulting in uncontrolled tibial advancements. Weakness in ankle dorsiflexors may lead to difficulties in achieving foot clearance during swing phase [4]–[6]. Therefore, a quantitative and qualitative understanding of muscle behavior in post-stroke could be helpful in the design of tailored-rehabilitation interventions and the development of assistive technologies.

In order to better understand the biomechanical behavior of a muscle, computational musculoskeletal modeling with Hill-type muscle models is commonly used [7]. Three-element Hill-type muscle models generally consist of a series elastic element (SE), a contractile element (CE), and a parallel elastic element (PE), which describe individual muscles’ force-length, force-velocity, and tendon force-length relationships. In the Hill-type muscle model, several muscle-tendon parameters including the maximum isometric force, the optimal fiber length, the tendon slack length, and tendon stiffness, play a determining role in transforming muscle excitation into muscle force and thus characterizing the force generation capacity of the muscle during a movement [8]. These muscle-tendon parameters also provide valuable information about how the muscle mechanism is altered due to various conditions. Nevertheless, these parameters are often difficult to measure directly in vivo. Generic muscle-tendon parameters can be estimated using scaled musculoskeletal models that typically depend only on the subject’s height and/or weight. The scaling process is called anthropometric scaling which does not take essential subject-specific information into consideration, such as muscle function [8], [9], age [10], and physical activity level [11]. To investigate muscle behavior in individuals with pathological conditions or elite athletes [12], a better approach with more accurate muscle-tendon parameters estimation is desired, especially for subject-specific applications.

In recent years, there has been an increasing interest in the use of optimization algorithms to estimate subject-specific muscle-tendon parameters. Modenese et al. [13] applied the least squares optimization technique with morphometric scaling to estimate optimal fiber length and tendon slack length while ensuring the preservation of physiologically consistent. However, only optimal fiber length and tendon slack length of hip muscles were estimated based on the images from an open-source Living Human Digital Library dataset [14]. Other information such as muscle excitation, kinematics and kinetics of subjects during different movements was not considered. Pizzolato et al. [15] presented a calibrated electromyography (EMG)-informed neuromusculoskeletal modelling (CEINMS)

toolbox that included a subject-specific muscle-tendon parameter calibration process, i.e., maximum isometric force, optimal fiber length, and tendon slack length. Their EMGmodeling methods (CEINMS) have been applied to various pathological groups to predict muscle force and joint torque such as in individuals with Parkinson’s disease [16] and post-stroke patients [17]. The calibration process proposed by Pizzolato et al., however, does not incorporate the experimental fiber length as a constraint. The muscle-tendon parameters and tendon stiffness therefore may not be wellcalibrated. Tendon stiffness plays a crucial role in the overall function of the muscle-tendon unit (MTU), affecting both the viscoelastic properties of the tendon and the operational range of muscle fibers [18]. Notably, altered tendon stiffness has been observed in individuals post-stroke [19]–[21]. Investigating and quantifying tendon stiffness is essential for a comprehensive understanding of the muscle-tendon interplay during locomotion in post-stroke individuals, providing valuable insights into tailored rehabilitation strategies. Delabastita et al. [22] proposed a new approach to estimate muscle-tendon parameters by solving a dynamic optimization problem using combined information from ultrasound images and EMG data of the calf muscles, as well as joint torque from able-bodied individuals during walking. Their proposed muscle redundancy solver algorithm estimated calf muscle–tendon parameters in a good accordance with values reported in the literature, and also improved computation of muscle–tendon dynamic interaction during walking and highlighted the importance of individualizing calf muscle-tendon parameters. However, only the fiber length of gastrocnemius medialis was estimated based on ultrasound images. In addition, their approach of normalizing EMG values to the maximum value during walking may not be applicable to pathological groups with impaired mobility, such as post-stroke patients in our study, who may not have independent walking or running ability. Furthermore, normalizing EMG signals with a free parameter can pose problems when estimating maximum isometric force. For instance, a weighting factor was applied to proportionally adjust the experimental EMG, enabling the generation of an equivalent amount of muscle force without altering the maximum isometric force. It could result in more uncertainty in the estimation of the maximum isometric force, which was a critical parameter in our study. Other EMG-driven modeling framework has also been applied in individuals post-stroke for muscle force and joint moment estimation [17], [23], wearable exoskeleton control strategy design [24]–[26], and joint kinematics prediction [27]. To the best of our knowledge, an EMG-driven modeling with Ultrasound informed optimization approach has not yet been applied to evaluate muscle-tendon parameters in post-stroke individuals.

The objective of this study was to evaluate the possible alterations in muscle-tendon parameters in post-stroke individuals within a musculoskeletal model that best explained experimentally collected EMG data and muscle fiber lengths. This was solved by combining the experimental data of several isometric ankle dorsi- and plantar-flexion trials of each individual in one optimal control problem.

## II. METHODS

## A. Experimental Setup and Data Processing

Muscle activation (surface EMG, 3000 Hz, Noraxon Inc., AZ, USA) of the gastrocnemius (GAS), soleus (SOL), and tibialis anterior (TA) of each subject’s one leg (randomized side for control and paretic side for post-stroke group) were collected, with electrodes placed according to European recommendations for surface EMGs [29]. Raw EMG signals were high-pass (20-500 Hz) filtered, rectified, low-pass filtered (6 Hz), and then normalized to the maximum EMG signal during all MVC trials [26], [30], [31]. A low-pass zero-lag Butterworth filter (6Hz) was applied to the measured ankle joint angle and torque [32]–[34].

![](images/bfb2b42c7b6959811269365af45a8a658e955a41519f656f1302712d3fc20642.jpg)  
Fig. 1. Experiment setup. All participants were seated in a comfortable semiupright position with their knee flexed at 30°and their foot firmly fixated to the foot plate connected to a dynamometer. Muscle activation (by EMG electrodes) and fiber length (by ultrasound) were recorded simultaneously during experiments.

Ultrasound images of GAS or SOL were recorded simultaneously by an ultrasonography system (Mindray M9, Shenzhen, China) with a 38 mm wide linear transducer (6–14 MHz) during plantarflexion MVCs, and Ultrasound images of TA were recorded during dorsiflexion MVCs. Muscle fiber lengths were automatically tracked using UltraTrack [35] and then carefully visually inspected to correct any potential tracking error. Based on the ultrasound-tracked muscle, we divided movements into three cases: $P F _ { G A S }$ (plantarflexion with measured GAS fiber length), $P F _ { S O L }$ (plantarflexion with measured SOL fiber length), and DF (dorsiflexion with measured TA fiber length).

## B. Muscle-Tendon Parameters Estimation

Estimation of the subject-specific muscle-tendon parameters was formulated as an optimal control problem. For each subject muscle-tendon parameters (maximum isometric force, optimal fiber length, tendon slack length and tendon stiffness) for a musculoskeletal model (gait10musc18, OpenSim [36], [37]) were estimated such that the difference between experimental observations and simulated behavior throughout the different isometric contraction tasks were minimized. The model (gait10musc18) was modified to only consist of foot and tibia segments, as well as the muscles actuating the ankle joint: GAS, SOL and TA. Each muscle was modelled as a Hill-type muscle with a compliant tendon as in De Groote et al. [38].

For each individual, the musculoskeletal geometry of the modified model was scaled using the OpenSim scaling tool. For healthy controls, we relied on the available anatomical marker positions captured by a Mocap system (Qualisys, Gothenburg, Sweden) to scale the leg and foot dimensions. For post-stroke subjects, the scaling was done based on manually measured anthropometric length. The scaled model served as a starting point to estimate individualized muscle-tendon parameters $\textbf { ( p ) } = \ [ l _ { M , o p t } , l _ { T S } , k _ { T } , F _ { m a x } ] )$ . by solving the following optimal control problem:

$$
\operatorname* { m i n } _ { \mathbf { p } , e _ { 1 } ( t ) , \dots e _ { n _ { t r i a l s } } ( t ) , \tau _ { r e s } }
$$

subject to

$$
\tau _ { e x p } = R \times F _ { T } + \tau _ { r e s }\tag{2}
$$

$$
\dot { e } = f _ { a c t } ( e , a )\tag{3}
$$

$$
\dot { F _ { T } } = f _ { m t } ( F _ { T } , a , q , \mathbf { p } )\tag{4}
$$

$$
c _ { p h y s i o l o g i c a l } ( F _ { T } , a , q , \mathbf { p } ) \geq 0\tag{5}
$$

The states of this optimal control problem are the muscle activations (a) and the muscle tendon force $( F _ { T } )$ . The controls are the muscle excitations (e) and the reserve actuator torques $( \tau _ { r e s } )$ . The cost function J is composed of a weighted sum of four terms, which is summed over all performed trials $( n _ { t r i a l s } )$ for all three muscles in one optimization procedure.

$$
J = \sum _ { i = 1 } ^ { n _ { t r i a l s } } \quad w _ { E M G } J _ { E M G , i } + w _ { F L } J _ { F L , i } +\tag{6}
$$

where $w _ { E M G } = 1 , w _ { F L } = 1 , w _ { e f f o r t } = 0 . 1$ , and $w _ { r e s } = 1 0$ are initial weight factors. The weights may vary slightly among subjects because the weight-tuning process was performed individually to achieve optimal performance. To determine the optimal weights for each component, a weight-tuning methodology was adopted based on a previous report [22]. Specifically, the effect of increasing or decreasing the weight of one term on the remaining tracking terms within the cost function was first visualized. We then selected the weight that effectively minimized the desired term while avoiding an increase in other terms.

The cost function is composed of an EMG term:

$$
J _ { E M G } = \intop _ { t _ { 0 } } ^ { t _ { f } } ( e ^ { 2 } ( t ) - \hat { e } ^ { 2 } ( t ) ) d t\tag{7}
$$

is a measure for the difference between simulated muscle excitations $e ( t )$ and measured normalized EMG signals $\hat { e } ( t )$ A fiber length term:

$$
J _ { F L } = \int _ { t _ { 0 } } ^ { t _ { f } } ( l _ { f } ^ { 2 } ( t ) - \hat { l } _ { f } ^ { 2 } ( t ) ) d t\tag{8}
$$

is a measure for the difference between simulated $l _ { f } ( t )$ and measured $\hat { l } _ { f } ( t )$ muscle fiber lengths.

An effort term:

$$
J _ { e f f o r t } = \intop _ { t _ { 0 } } ^ { t _ { f } } ( e ^ { 2 } ( t ) + a ^ { 2 } ( t ) ) d t\tag{9}
$$

regularizes the optimization problem to avoid redundant solutions that require more muscle effort, where $a ( t )$ is the muscle activation.

A reserve actuator term:

$$
J _ { r e s } = \intop _ { t _ { 0 } } ^ { t _ { f } } \tau _ { r e s } ^ { 2 } ( t ) d t\tag{10}
$$

![](images/0b7bc2ab4a6866ef97805eccbe437cc1c19bc5e093497a1587cce08fb517ebc6.jpg)

![](images/95b259a7a2d0c37c96369d4976f2f8b4c7cdfe6bc184fc395af5cd0711632a7a.jpg)  
Fig. 2. (a) Overview of the individualized muscle-tendon parameters estimation by using the muscle redundancy solver. We first scaled a generic musculoskeletal model (gait10dof18musc) by using recorded static marker trajectories or measured leg length to fit each subject’s anthropometry. The ankle joint angle measured from the Dynamometer was used as the inputs for muscle analysis to compute the muscle-tendon length and moment arm of each muscle-tendon unit. During optimization, we also added constraints based on experimental measurements, which were muscle excitations and fiber lengths of three muscles (GAS, SOL, TA). Then we solved for the muscle-tendon parameters by minimizing the sum of squared muscle activations as well as maximizing the consistency between measured and simulated joint moments, EMGs and muscle-fiber lengths. Finally, we obtained individualized muscle-tendon parameters: maximum isometric force $F _ { m a x } ,$ optimal fiber length $l _ { M , o p t }$ , tendon stiffness $k _ { T }$ and tendon slack length $l _ { T S }$ for individuals. (b) Overview of the optimization performance evaluation of the muscle redundancy solver through a cross-validation method. Specifically, 75% of the trials were used for calibration to obtain an individualized model with calibrated muscle-tendon parameters, and then tested on the remaining trials. Two cases were evaluated (Case 1: generic model; Case 2: individualized model). In Case 2, the muscle-tendon parameters of the individualized model were the output of the calibration procedure using the muscle redundancy solver optimization, while the parameters of the generic model in Case 1 were from a scaled musculoskeletal model. In both cases, joint angle, joint moment, EMG data, and generic/individualized musculoskeletal model were used as the inputs of the muscle redundancy solver. The outcome parameters were tracked muscle excitations, joint moment, and fiber length.

represents a reserve actuator that is added to the ankle joint and can provide an additional joint moment to the ankle muscles. It is included in the model to improve the numerical stability of the optimization problem and to ensure that a feasible solution is obtained even when the modeled muscles are not strong enough to generate the measured joint torque. In the calibration trials, this term is heavily weighted to ensure that its contribution is negligible. During the evaluation process, the reserve actuator term serves as a measure of the mismatch between the model and experimental data.

The optimization is constrained such that the sum of joint moments produced by the muscles and the reserve actuator moments $\left( \tau _ { r e s } \right)$ is equal to the experimental joint moment $( \tau _ { e x p } )$ measured by the dynamometer (Eq. 2). R is the moment-arm matrix, that describes the moment arm of each muscle with respect to the ankle joint and depends on the ankle and knee joint angle $( q )$ . The ankle joint angle is directly obtained from the dynamometer recording. The knee joint angle was constant across trials and subjects at $3 0 ^ { \circ }$

We impose muscle excitation-activation dynamics $( f _ { a c t } )$ and muscle-tendon dynamics $( f _ { m t } )$ as described in [38]:

$$
\begin{array} { r l r } { \dot { e } } & { { } } & { = f _ { a c t } ( e , a ) } \end{array}\tag{11}
$$

$$
\begin{array} { r l } { \dot { F _ { T } } } & { { } = f _ { m t } ( F _ { T } , a , q , \mathbf { p } ) } \end{array}\tag{12}
$$

Note that the estimated parameters appear as input to the muscle-tendon dynamics. Additional constraints $( c _ { p h y s i o l o g i c a l } ( F _ { T } , a , q , \mathbf { p } ) \ \geq \ 0 .$ , Eq. (5)) were imposed to ensure the optimal solution was physiologically feasible:

• Optimal fiber lengths of GAS, SOL, and TA were bounded between 40% and 120% of the maximal measured fiber length respectively [22].

• The ratio between the estimated and generic values for optimal fiber length, tendon slack length and tendon stiffness were constrained to be equal for the GAS and SOL as they form the triceps surae muscle complex and share the achilles tendon.

(a)

(b)

• Excitations and activations were bounded between zero and one.

We used direct collocation to transcribe each optimal control problem into a large sparse nonlinear program. We used a trapezoidal integration scheme with a mesh size of 10ms and solved the resulting NLP with the solver IPOPT. All gradients were computed using automatic differentiation, where we relied on CasADi [39].

## C. Evaluation of Estimated Parameters and Data Analysis

We analyzed group-level differences in the GAS, SOL, and TA in terms of maximum isometric force, tendon stiffness, optimal fiber length, and tendon slack length. The optimal fiber length and tendon slack length were normalized by height. Shapiro-Wilk tests were used to check the normality of data distribution (significance level at $p \ SOL: $p = 0 . 0 7 ;$ Fig. 3).

For muscle excitation prediction, in the post-stroke group, the individualized model had significantly lower RMSEs in all muscles for cases $P F _ { G A S }$ and $D F ~ ( P F _ { G A S } \colon \mathbf { G A S } { \mathrel { : } } p =$ 0.02, SOL: $p = 0 . 0 3$ , and TA: $p = 0 . 0 5$ ; DF: GAS: $p =$ $0 . 0 5 , \ \mathrm { S O L } { : p } = 0 . 0 5$ , and $\mathrm { T A } { : } p = 0 . 0 2$ ; Fig. 5 (b), (d) and (f)). Furthermore, in Case $P F _ { S O L }$ , significant differences were observed in GAS $( p = 0 . 0 3$ , Fig. 5 (b)) and SOL $( p = 0 . 0 3 ,$ Fig. 5 (d)). Although not statistically significant, lower RMSEs were also observed in TA $\left( p = 0 . 0 8 , \mathrm { F i g } . \textbf { 5 } ( \mathrm { f } ) \right)$ . However, in the control group, no significant differences were observed in all cases $( \mathrm { F i g . ~ 5 ~ ( a ) , ~ ( c ) } .$ , and (e)).

![](images/5dc0d22fb859dfdefd1adda5a08e8bfe4f18d597acf3ef03c01b9b8a4cf4ba65.jpg)

![](images/a877bbc5a1b6834a818af5ee6ee36dd742d07bfaf266f4617017539d2087e5f8.jpg)

![](images/e275f5a7d6ba13d8e50fafef4e216c5a717d102042bb3bfc622b0767984110a3.jpg)

![](images/1fd186e9673388d2ff5229cac29befb689e8f541aba47a2b3c273323395881d8.jpg)

![](images/6ab4ec8a9ac1dc99c7c9f9a659b12ece3ed9f7998df862e51818aa3c1b5a2958.jpg)

![](images/1fb4534ae2f03353aad6db5a76fd3c17a813eda49ae788c620d6e671b2718f98.jpg)  
Fig. 5. Violin plots depicting the distributions of the prediction RMSE in the muscle excitation of gastrocnemius (GAS), soleus (SOL), and tibialis anterior (TA) via individualized and generic models in healthy and post-stroke group during three cases: $P F _ { G A S }$ (Plantarflexion with measured GAS fiber length), $P \bar { F _ { S O L } }$ (Plantarflexion with measured SOL fiber length), and $D F$ (Dorsiflexion with measured TA fiber length). A significant difference between the two models was indicated with ∗.

![](images/afa6fbb102fae6e251c24d4e46b181a4a925b9eb8f0129028fd3a8ab3bcb741d.jpg)  
Fig. 6. One example of measured and predicted muscle excitation via generic and individualized models during maximum voluntary contraction (MVC) in both healthy and post-stroke groups. In this example, the gastrocnemius (GAS) and soleus (SOL) were compared during plantarflexion MVC while the tibialis anterior (TA) muscle was compared during dorsiflexion MVC.

![](images/1f1b40c3ea05e6d5bd7f42f011f397773a5cc166a581b1a4fe543e7bde3e8657.jpg)

![](images/0e1b8993d3a83e4f7ee80d05aaaee18c8b92fc1377596d71f7bbcc0065c9c802.jpg)

Fig. 7. Violin plots depicting the distributions of the prediction RMSE in the ankle dorsi/plantarflexion torque via individualized and generic models in healthy and post-stroke group during three cases: $P F _ { G A S }$ (Plantarflexion with measured GAS fiber length), $P \bar { F } _ { S O L }$ (Plantarflexion with measured SOL fiber length), and DF (Dorsiflexion with measured TA fiber length). A significant difference between the two models was indicated with ∗.  
![](images/2a48d868dceaced637781f60152518b05bac43bd9c4aa5861881e3fe9bcd04ec.jpg)  
Fig. 8. One example of measured and predicted ankle dorsi/plantarflexion torque via generic and individualized models during maximum voluntary contraction (MVC) in both healthy and post-stroke groups during three cases: $P F _ { G A S }$ (Plantarflexion with measured GAS fiber length), $P F _ { S O L }$ (Plantarflexion with measured SOL fiber length), and $D F$ (Dorsiflexion with measured TA fiber length).

For joint torque prediction, the individualized model showed significantly lower RMSEs compared to the generic model in all cases in the post-stroke group $( P F _ { G A S } \colon p \ Torque (SD) CaseHealthy ControlPost-stroke $P F _ { G A S }$  $P F _ { S O L }$  $_ { D F }$  $P F _ { G A S }$  $P F _ { S O L }$ DF5°dorsi-1.47 (0.37)-1.61 (0.51)0.48 (0.15)-0.27 (0.22)-0.29 (0.23)0.02 (0.01)0°plantar-1.48 (0.29)-1.58 (0.33)0.49 (0.19)-0.19 (0.16)-0.23 (0.17)0.02 (0.02)10°plantar-1.13 (0.36)-1.09 (0.31)0.52 (0.23)-0.13 (0.11)-0.11 (0.12)0.03 (0.04)20°plantar-0.87 (0.31)-0.89 (0.31)0.52 (0.18)-0.10 (0.10)-0.06 (0.10)0.04 (0.06)