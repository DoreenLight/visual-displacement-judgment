<!-- source: https://doi.org/10.1109/tbme.2024.3352556 -->
<!-- title: EMG-Constrained and Ultrasound-Informed Muscle-Tendon Parameter Estimation in Post-Stroke Hemiparesis -->
<!-- fetched: 2026-09-30T13:46:08Z -->
<!-- via: local_paper_fulltext(_translation_outputs/Zhang_2024_TBME_MuscleTendon/Zhang_2024_TBME_MuscleTendon/auto/Zhang_2024_TBME_MuscleTendon.md) -->
<!-- note: 原文 PDF 存于 09_章隆彬_合作团队/papers/，此为 PDF 解析后的全文 Markdown -->

# EMG-Constrained and Ultrasound-informed Muscle-tendon Parameter Estimation in Post-stroke Hemiparesis

Longbin Zhang, Tom Van Wouwe, Shiyang Yan, and Ruoli Wang

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

Seven post-stroke participants (age: $5 3 . 7 \pm 1 6 . 2$ years; sex: 1F/6M; weight: $7 4 . 1 \ \pm \ 1 4 . 4$ kg; height: $1 7 2 . 1 \pm \ : 1 0 . 1$ cm) were recruited from a local rehabilitation clinic. The inclusion criteria were: (1) stroke > 6 months prior to inclusion; (2) no anti-spastic treatment within three months; and (3) absence of other lower limb injuries or disorders. Seven able-bodied subjects (age: $5 4 . 4 \pm 6 . 4$ years; sex: 3F/4M; weight: 70.0 $\pm \nobreakspace 9 . 5 \nobreakspace$ kg; height: $1 6 8 . 4 ~ \pm ~ 9 . 5$ cm) were recruited among advertisement as a control group. The Swedish Ethical Review Authority approved this study (Dnr. 2016/286-32) and all participants gave informed written consent. The data set of post-stroke subjects was a subset of a previous study [9]. In total, Eleven post-stroke subjects were recruited. Three subjects dropped out and data from one subject was excluded due to the poor image quality. All post-stroke subjects required assistive devices for mobility. Among all, three relied on wheelchairs, three used crutches and ankle foot orthoses (AFO) and one with only AFO. During measurement, all participants were seated in a comfortable semi-upright position with their knee flexed at 30°and their foot firmly fixated to the foot plate connected to a dynamometer (IsoMed 2000, Hemau, Germany) (Fig. 1). In its initial position, the footplate was positioned perpendicularly to the tibia of the subject; this position was defined as 0°ankle rotation. Prior to testing, the available range of motion (ROM) of each subject was assessed, and no discomfort was discovered within the test ROM. Both control and post-stroke subjects performed several maximum voluntary contraction (MVC) tasks. For controls, maximum voluntary isometric dorsi and plantar flexion contractions against the dynamometer while maintaining specific joint angles. During these contractions, the dynamometer recorded both joint torque and angle measurements simultaneously at 3000 Hz. To ensure that the functional flexion-extension axis of the ankle was well aligned with the flexion-extension axis of the dynamometer, we followed a standardized procedure recommended by previous studies [28]. Specifically, the axis of ankle flexion/extension was aligned with the axis of the dynamometer using a laser-pointing device. The experiments were performed at four different ankle angles (5°dorsiflexion, 0°, 10°, and 20°plantarflexion). For the post-stroke group, due to limited ROM in dorsiflexion, three out of seven subjects were not able to perform MVC in 5°dorsiflexion. Therefore, measurements were done only at three angle positions. The investigator gave verbal encouragement throughout the measurements. At each angle, subjects were instructed to perform dorsi-/plantar-flexion MVC with a duration of 5s. This was repeated once with 30s rest in between repetitions.

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

$$
J\tag{1}
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

We analyzed group-level differences in the GAS, SOL, and TA in terms of maximum isometric force, tendon stiffness, optimal fiber length, and tendon slack length. The optimal fiber length and tendon slack length were normalized by height. Shapiro-Wilk tests were used to check the normality of data distribution (significance level at $p \ < \ 0 . 0 5 )$ . Twosample t-tests were used for normally distributed muscletendon parameters, while Mann-Whitney U tests were used for abnormally distributed data between two groups (significance level at $p < 0 . 0 5 )$ .

We evaluated whether the predictive power of the model could be improved through EMG- and US-constrained calibration by comparing the performance of a model with calibrated muscle-tendon parameters to that of a model that was only scaled. This was achieved through cross-validation by using the muscle redundancy solver. Two cases were evaluated (Case 1: generic model; Case 2: individualized model; Fig. 2 (b)). In Case 2, the muscle-tendon parameters of the individualized model were obtained based on the calibration procedure using the muscle redundancy solver optimization, while the parameters of the generic model in Case 1 were from a scaled musculoskeletal model. The predicted muscle fiber lengths, joint moment, and muscle excitations by the two cases were compared. Specifically, 75% of the trials were used to obtain an individualized model with calibrated muscletendon parameters and then tested on the remaining trials for each subject. In total, there are 12 trials for each subject, with the exception of three post-stroke subjects who have only 9 trials due to limited range of motion. For both cases, joint angle, joint moment, EMG data, and either a generic or individualized musculoskeletal model were used as inputs to the muscle redundancy solver. Notably, fiber length tracking $( J _ { F L } )$ terms were not included in the objective function. RMSE differences between predicted and experimental muscle excitations, joint moment, and fiber lengths by individualized and generic models were investigated. Significant difference was determined either by paired t-tests for normally distributed data or Wilcoxon signed-rank tests for abnormally distributed data, at a significance level of $( p < 0 . 0 5 )$ .

## III. RESULTS

A. Evaluation of the optimization performance of the muscle redundancy solver

Overall, the individualized model yielded a more realistic prediction of muscle-tendon behavior in the post-stroke group. The RMSEs in fiber length and joint torque estimation were lower in the individualized model than the generic one in both healthy-control and post-stroke groups (Fig. 3-8).

![](images/24d4d00781ba0555392464a988727ca80f0efb198e74d83ee328aafae6383de4.jpg)

![](images/6f908af13f9411712d1bb83d5c428736e9f00a9b17d786b84f44cf8eff74f04c.jpg)

Fig. 3. Violin plots depicting the distributions of the prediction RMSE differences between the fiber length of gastrocnemius (GAS), soleus (SOL), and tibialis anterior (TA) via individualized and generic models in healthy and post-stroke group during three cases: $P F _ { G A S }$ (Plantarflexion with measured GAS fiber length), $P F _ { S O L }$ (Plantarflexion with measured SOL fiber length), and DF (Dorsiflexion with measured TA fiber length). A significant difference between the two models was indicated with ∗.  
![](images/c5adf84b8208fb37f0ff50430d5f3bc9066f006b3e663f0a66eaae80d5912702.jpg)  
Fig. 4. One example of measured and predicted fiber length via generic and individualized models during maximum voluntary contraction (MVC) in both healthy and post-stroke groups. In this example, the gastrocnemius (GAS) and soleus (SOL) muscles were compared during plantarflexion MVC while the tibialis anterior (TA) muscle was compared during dorsiflexion MVC.

For fiber length prediction, the individualized model showed significantly lower RMSEs in Case DF, both in the poststroke group $( p \ < \ 0 . 0 1 )$ and the control group $( p \ : = \ : 0 . 0 5 ;$ Fig. 3). Although not statistically significant, the RMSEs were found somehow lower in $P F _ { G A S }$ and $P F _ { S O L }$ in both poststroke groups $( P F _ { G A S } \colon p = 0 . 1 6$ and $P F _ { S O L } \colon p = 0 . 1 1 $ and control groups $( P F _ { G A S } \colon p = 0 . 3 7 .$ , P F<sub>SOL</sub>: $p = 0 . 0 7 ;$ Fig. 3).

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

For joint torque prediction, the individualized model showed significantly lower RMSEs compared to the generic model in all cases in the post-stroke group $( P F _ { G A S } \colon p \ < \ 0 . 0 1$ $P F _ { S O L } { : } p = 0 . 0 2$ and $D F \colon p = 0 . 0 3 )$ . In the control group, although not statistically significant, the RMSEs were also lower in all cases $( P F _ { G A S } \colon p = 0 . 0 9 , P F _ { S O L } \colon p = 0 . 0 5$ and $D F \colon p = 0 . 1 0 ; \mathrm { { F i g } } . 7 )$ .

## B. Estimated muscle-tendon parameters

Overall, the estimated maximum isometric force, optimal fiber length, and tendon stiffness in the post-stroke group were considerably lower than the control group (Fig. 9).

The estimated maximum isometric force of GAS, SOL and TA in the post-stroke group were $0 . 5 \pm 0 . 2 \ \% , \ 1 . 0 \pm 0 . 3 \ \%$ and $1 . 0 \pm 0 . 5 \%$ body weight (BW) respectively. They were all significantly lower than those in the control group $\mathrm { ( G A S ; }$ $3 . 8 \pm 1 . 3$ %BW, $p < 0 . 0 1$ ; SOL: 7.9 ± 2.7 %BW, $p < 0 . 0 1$ $\mathrm { T A } \colon 4 . 9 \pm 0 . 8$ %BW, $p < 0 . 0 1 ;$ Fig. 9 (a)).

![](images/28c4d6f2b99baad84396a95d58d2097088fe6212e8d7342f6ad463cd069bf118.jpg)

![](images/e8e3def53a87ec9be516bfdad78e0b1f1068596b07624eef1a97fa570da5907f.jpg)

(a)  
![](images/daf0e646524bb877607b04b274d7c50661a9a61d24e98b33945bade451e26092.jpg)

![](images/2ef8233cacc88fc6d0929fcf94be578876388f650313dd24796457d8cd0ed4cd.jpg)

![](images/60d407bd7dff09e153b89724cec0d7862e8f6b519789fbf10e726e6af6b48275.jpg)

![](images/a45cf17ecc8cf5000c6a79adfee8708552c8e735599f59d83d64dfc2ab56b81a.jpg)  
Fig. 9. Violin plots depicting the distributions of estimated muscle-tendon parameters of gastrocnemius (GAS), soleus (SOL), and tibialis anterior (TA) in control and post-stroke group; (a) normalized maximum isometric force (by body weight (BW)), (b) tendon stiffness, (c) optimal fiber length, (d) normalized optimal fiber length (by height), (e) tendon slack length, and (f) normalized tendon slack length (by height). \* indicates a significant difference between two groups.

The estimated optimal fiber length of TA in the post-stroke group was found significantly lower than that of in control group $( 5 7 . 6 \pm 1 0 . 2 $ mm vs. $7 1 . 7 \pm 1 1 . 3$ mm, $p = 0 . 0 3 , \mathrm { F i g }$ . 9 (c)). Although not significant, shorter estimated optimal fiber length in GAS and SOL (GAS: $4 6 . 1 \pm 3 . 3$ mm vs. $4 8 . 4 \pm$ 7.9 mm, $p = 0 . 4 8 ;$ SOL: 40.4 ± 4.8 mm vs. $4 4 . 3 \pm 7 . 8$ mm, $p = 0 . 3 0 )$ was also observed in the post-stroke. Normalized optimal fiber length (by body height) follows the same trend as the optimal fiber length. (Fig. 9 (d)).

No significant difference was found in the tendon slack length between groups (post-stroke vs. healthy-control: GAS $4 1 . 0 \pm 5 . 2$ mm vs. $4 1 . 3 \pm 4 . 4 \ : \mathrm { m m }$ , SOL $2 6 . 4 \pm 3 . 3$ mm vs. $2 6 . 3 \pm 2 . 8$ mm, and TA $2 5 . 9 \pm 2 . 8$ mm vs. $2 3 . 8 \pm 3 . 4$ mm; GAS: $p = 0 . 9 8$ , SOL: $p = 0 . 9 8$ , TA: $p = 0 . 2 9$ , respectively; Fig. 9 (e)). Normalized tendon slack length (by body height) also follows the same trend as the tendon slack length. (Fig. 9 (f)). The stiffness of Achilles and TA tendon in the poststroke group were $8 9 \pm 7 9$ N/mm and $7 9 \pm 7 1$ N/mm; and these were also significantly smaller than in the control group (Achilles tendon: $5 6 4 \pm 2 2 0$ N/mm, $p < 0 . 0 1$ ; TA tendon: $2 3 0 \pm 6 7$ N/mm, $p < 0 . 0 1$ ; Fig. 9 (b)).

![](images/bf2c8350c9e3c24b7625b479ab21e3e2b9d9c0157edb528e76d08956d00eabf2.jpg)  
(a)

![](images/16ecae540c102daac37ffad2fbeca0a03843a6e96e2ba8441a698824321ec84e.jpg)

![](images/8a1c21229c4833709fbc327798635de2a69a44bea8f100ef30de381fbc1b3c3f.jpg)  
Fig. 10. Active (solid line) and passive (dashed line) muscle force-length relationship (derived based on Hill-type model) of gastrocnemius (GAS), soleus (SOL), and tibialis anterior (TA) in control and post-stroke group. The grey dashed line illustrated that the maximal isometric force occurred at the optimal fiber length.

Compared to the control group, the active muscle forcelength curve in the post-stroke was left-shifted with a considerably lower maximum isometric force and optimal fiber length in all three muscles (Fig. 10).

## IV. DISCUSSION

In this work, we estimated subject-specific muscle-tendon parameters of major ankle dorsi/plantarflexors in post-stroke individuals and healthy controls by solving the muscle redundancy problem using the direct collocation optimization method. The proposed optimal algorithm takes muscle activation and muscle dynamics into account based on experimental EMG signals and measured muscle fiber length through US images. We found that the maximum isometric force, tendon stiffness and optimal fiber length in the post-stroke group were considerably lower than in the control group. In addition, we showed that, based on EMG, US, and dynamometry data, we could improve the subject-specificity of musculoskeletal models by personalizing muscle-tendon parameters using a direct collocation optimization method, which yields a more realistic movement simulation that better matches experimental data than a scaled generic model. Compared to existing methods, our developed direct collocation optimization algorithm in combination with the unique experimental data set offers several advantages. It enables the incorporation of different types of trials into a single optimization, providing more realistic and physiologically consistent parameter estimation. Furthermore, the algorithm not only provides essential muscletendon parameters determining the muscle force production capacity, such as optimal fiber length, tendon slack length, and tendon stiffness, but also maximum isometric force, which has significant implications for the understanding and treatment of post-stroke motor deficits. To the best of our knowledge, this is the first study that attempted to evaluate the muscletendon parameters of GAS, SOL, and TA in post-stroke individuals with both experimental EMG signals and fiber length constraints during ankle plantarflexion and dorsiflexion MVC movements. In addition, the proposed muscle-tendon parameter optimization framework was further expanded to include a pathological group that does not require independent walking and running abilities. Our approach therefore has the potential to apply to other groups with motion impairment, which may offer valuable insights into muscle-tendon mechanics and inform future rehabilitation strategies.

## A. Evaluation of the optimization performance of the muscle redundancy solver

Instead of simply scaled muscle-tendon parameters, a personalized musculoskeletal model with calibrated muscletendon parameters yielded a significant improvement in the agreement of the simulated and measured fiber length, muscle excitation, and joint moment in most cases. The benefit of personalized muscle-tendon calibration was more predominately in the post-stroke group. For instance, a significantly different torque prediction of the post-stroke group (Fig. 7 (b)) was likely due to the fact that the maximum isometric force was significantly lower than in controls (Fig. 9 (a)). As a result, it is unlikely that the generic model with uncalibrated MTU parameters would yield accurate estimation, in other means of realistic muscle-tendon behavior description. Moreover, unlike controls, there was a significant improvement in EMG excitation prediction in the post-stroke group in most cases with calibrated muscle-tendon parameters (Fig. 5 (b), (d), and (f)). This may be attributed to the fact that co-contraction is often observed in post-stroke individuals to maintain stability [40]. However, the optimization algorithm minimized the cost function of muscular effort, the integral of the summed squared muscle excitations. Thus, co-contraction was not favorable for the optimization solution. In future studies, taking cocontraction into account in the cost function should be further investigated in order to better model the muscle-tendon behavior of individuals with motor disorders, where co-contraction scenarios could be expected.

In the current study, the direct collocation method was used to formulate the optimal control problem of finding the muscle-tendon parameters that best explain experimental observation, as a nonlinear programming problem. Compared to other optimization methods, collocation methods are generally more computationally efficient; thus have been previously investigated in several studies with respect to muscle redundancy problems. Groote et al. [38] first proposed the direct collocation method as a robust and computationally efficient approach for solving muscle redundancy problems during walking on a male test subject. The method has also been used to estimate subject-specific muscle-tendon parameters of knee actuators on healthy subjects [41]. Delabastita et al. [22] further incorporated experimental measurement of motion capture, ultrasound images, and EMG data of the calf muscles to estimate muscle-tendon parameters in young and elderly adults during gait. However, the individualized maximum isometric force was not incorporated in their optimization process. However, the maximum isometric force was usually suggested to be adjusted within the musculoskeletal model in simulating participants’ strength most likely deviating from the average population, such as pathological groups or elite athletes [12]. Here, we incorporated a unique experimental measurement protocol with direct collocation to provide a more comprehensive evaluation of the muscle-tendon parameters alterations in individuals with motor deficiency, i.e., poststroke individuals.

## B. Estimated muscle-tendon parameters

Muscle-tendon parameters are commonly investigated parameters in persons with motor impairment, in order to better understand the mechanical behavior of muscle-tendon units during movement. Based on our simulation, the post-stroke group had a shorter optimal fiber length, reduced maximum isometric force, and smaller tendon stiffness in both plantarflexors and dorsiflexor than the control group. However, no significant difference was observed in tendon slack length. These parameters have physiological meanings and determine the muscle excitation-force relationship in the Hill-type muscle model. They are also relevant and informative in clinical interpretation, e.g., these parameters can provide essential information for personalized rehabilitation planning.

Active muscle force generation capacity could be remarkably affected after stroke. We found that post-stroke individuals exhibited significantly reduced maximum isometric force compared to controls. Maximum isometric force is defined as the peak force a muscle can produce while maintaining a constant length. The common method for estimating an individual muscle’s maximum isometric force is to multiply the physiological cross-sectional area (PCSA) with the specific tension [42]. However, directly estimating the maximum isometric force experimentally could be very challenging and resource-demanding. Rather, maximal isometric torque is often reported [43], [44]. However, it can not reflect the potential differences among synergistic muscles. Therefore, an optimization technique incorporating feasible experimental measurements could be a good solution for estimating the maximum isometric force of the individual muscle. We found that the estimated maximum isometric force of GAS, SOL, and TA were all much weaker in post-stroke individuals and were only 13.9%, 13.9%, and 22.4% of the control group. Very few studies have investigated the maximum isometric force of ankle muscles in post-stroke individuals in vivo. Our findings are in line with previous studies reporting reduced MVC torque in plantarflexors [45] and dorsiflexors [46]. Optimal fiber length is another essential parameter influencing the active force generation during dynamic muscle contraction. To the best of our knowledge, only one study investigated the alteration of the optimal fiber length on ankle plantarflexors in post-stroke individuals. Gao et. al [47] reported a shorter optimal fiber length (33.2 ± 3.2 mm) in the gastrocnemius in post-stroke individuals than healthy controls $( 4 7 . 4 \pm 2 . 7$ mm). They established the muscle-active force relationship by curve-fitting scattered data points for each subject during submaximal MVC measurements. Based on our simulation, the optimal fiber lengths were found unexpectedly shorter not only in plantar-flexors but also in ankle dorsiflexion. Further investigation is needed for clarification. In addition, maximal isometric force and optimal fiber length essentially determine the force - length profile of an individual muscle (Fig. 10). Evaluating the individualized force - length relationship yields the possibility to assess the selective weakness at particular muscle length (joint angle) as well as the optimal movement range after stroke, which can provide valuable information for effective strength training intervention.

Tendon stiffness is an important mechanical feature that influences the whole MTU function in terms of functional properties of the tendon and the operational range of muscle fibers [18]. Tendon stiffness is typically characterized by a toe region with low stiffness and a linear region with high stiffness. In recent years, tendon stiffness in the linear region has been commonly investigated by ultrasonography-based approaches. Our previous study has investigated the stiffness of GAS and SOL aspects of Achilles tendon in able-bodied subjects during passive stretching and found the stiffness of the two sub-compartments were similar [28]. However, due to variant joint configurations and assumptions, incomparable findings were often reported in vivo studies. For example, the Achilles tendon stiffness was determined previously in young $( 1 7 0 ~ \pm ~ 3 7$ N/mm) and older (141 ± 48 N/mm) healthy adultsin vivo as the slope of muscle force - muscle length elongation relationship between 10% and 80% of the maximal voluntary isometric contractions [48]. Another study by Khair et al. [49] reported the stiffness of Achilles tendon in uninjured limb $( 4 8 6 . 5 ~ \pm ~ 2 1 0 . 2 $ N/mm) of persons one year after non-surgically treated Achilles tendon rupture as the tangent at 50% MVC with the tendon elongation - force curve. Alternatively, tendon stiffness can be estimated based on computational modeling and simulation methods, where the stiffness is less dependent on the in vivo measurement methods and conditions. A recent study by Delabastita et al. [22] used a direct collocation optimal simulation method to estimate Achilles tendon stiffness in young $( 2 1 5 . 2 ~ \pm ~ 6 1 . 4$ N/mm)

and old adults $( 2 0 2 . 4 \pm 6 2 . 9$ N/mm) during walking. In the current study, we implemented the same direct collocation optimal method and obtained both Achilles tendon and TA tendon stiffness in post-stroke and control group (post-stroke vs. healthy-control: Achilles tendon $8 9 \pm 7 9$ N/mm vs. 564 ± 220 N/mm; TA tendon $7 9 \pm 7 1$ N/mm vs. $2 3 0 ~ \pm ~ 6 7$ N/mm). Our finding of a more compliant tendon in the poststroke group aligns with previous clinical observations in Achilles tendon [19], [50], [51]. For example, Zhao et al. [19] observed a compliant Achilles tendon in post-stroke by tracking tendon elongation and measuring ankle joint torque during plantarflexors MVC.

There were several limitations in this study. In our study, we normalized EMG by the maximum value over MVC trials, instead of dynamic movements such as walking and running. We acknowledge that MVC isometric trials migh not necessarily reflect the true maximal muscle activation. During the dynamic movement, the EMG signals for some muscles may exceed the peak value recorded during isometric MVC trials [52], [53]. However, in a recent study, the impact of three different EMG normalization methods, namely “isometric MVC”, “isokinetic $\mathbf { M V } { \boldsymbol { \mathbf { C } } } ^ {  }$ , and “isokinetic MVC with consideration of joint range-of-motion and velocity”, on muscle activation and the antagonist-agonist co-contraction index in post-stroke individuals was investigated [54]. The study found that there was no significant difference observed between the three normalization methods and thus concluded that EMG normalization with isometric MVC is appropriate for post-stroke participants. Given our focus on participants with impaired mobility, who lack independent walking or running ability, we assumed that the EMG data obtained during isometric MVC is an appropriate approximation of the true maximum EMG signals. It is nevertheless worth noting that our normalization method may impact the estimated muscle tendon model parameters, necessitating further evaluation in future studies. Also, we only modeled the linear region of tendon force-elongation behavior without a toe region. This may increase the discrepancy between the computed tendon stiffness in vivo and model estimated value [22]. Further studies using a non-linear tendon model might provide a more accurate stiffness estimation. It is worth mentioning that a significant difference was observed only in the post-stroke group when evaluating the optimization performance of the muscle redundancy solver. While the RMSEs in fiber length and joint torque estimation were lower in the individualized model compared to the generic model in both control and post-stroke groups, this difference was not evident in muscle activation. The discrepancy may be attributed to the limited sample size and may also indicate that the conventional scaled model remains suitable for the healthy group. Further investigations with a larger sample size are warranted.

Another limitation is that only experimentally measured muscle fiber length and EMG signals were used to constrain the optimization process. We did not include pennation angle due to the large angle variability [9], especially in the poststroke group. For example, due to rheological muscle property changes after stroke, increased probe pressure has to be applied in order to achieve better muscle visibility in the ultrasound measurement which could lead to a high angle variability. Previous studies have also shown that compared to other parameters, pennation angle was the least influential parameter in musculoskeletal model-based muscle force estimation [55]– [58], it would be interesting to explore whether including reliable pennation angle could improve the robustness of MTU parameter estimation. Also, further analysis of the impact of integrating ultrasound measurement into the EMG-driven model calibration procedure on the muscle-tendon parameter estimation in post-stroke persons is valuable. Additionally, both neural and muscular changes can lead to weakness in post-stroke subjects. While our musculoskeletal modeling and optimization framework considered the neural drive by constraining muscle excitation with normalized EMG data, it is crucial to recognize that further investigation and additional experimental measurements are necessary to determine whether there is a decrease in the neural drive. From a muscle modeling perspective, the parameter ‘maximal isometric force can potentially incorporate information about the potential changes in the neural drive. However, it is essential to exercise caution and avoid over-interpreting this parameter.

It is important to note that our study solely focused on the ankle flexion/extension joint due to the specific area of interest. However, our muscle-tendon parameter estimation framework could be transferred to other joints with available experimental data, if desired. Specifically, we included three crucial ankle dorsi/plantar flexor muscles, namely GAS, SOL, and TA. Within our experimental setup, we treated gastrocnemius medialis (GM) and gastrocnemius lateralis (GL) as a lumped entity, considering their anatomical and functional similarities [59], [60], thus avoiding separate analysis for them. This decision aimed to mitigate the unnecessary complexity of the experimental setup, particularly on the post-stroke group. Including a separate measurement for GL, such as through ultrasound, could potentially induce fatigue in post-stroke subjects due to the additional MVC trials required for ultrasound image acquisition and prolonged acquisition sessions. In our opinion, using musculoskeletal models with more muscles has very limited benefits. We chose the Gait10muscle18 model, a simplified version derived from gait2392 [61]–[63]. This model features 10 degrees of freedom and 9 major muscles including GAS, SOL and TA, and with the identical Hilltype muscle model as gait2392. In addition, the lumped GM and GL matched with our study setup as stated earlier. In addition, tibialis posterior was not considered in the model as it was challenging to obtain EMG and ultrasound data of such a small and deep muscle. Though Distefano [64] have previously reported that the gastrocnemius and soleus are the primary ankle plantarflexors and that the other plantar flexors only contribute 7% of the remaining plantarflexor force, we acknowledge that our exclusion of smaller muscles may have led to an overestimation of the maximum isometric force in the muscle-tendon unit of GAS and SOL.

It is worth acknowledging that the gastrocnemius is a biarticular muscle that spans both the ankle and knee joints. While incorporating an estimate of the knee and even hip torque could potentially improve the accuracy of the gastrocnemius activation estimation. However, it was not feasible with the current experimental setup with the dynamometer. Furthermore, the position of the knee joint during calibration may also have an impact on the musculotendon parameters of the gastrocnemius. Although these parameters are primarily determined by the muscle’s architecture and anatomical structure, including additional knee angle setups would contribute to a more realistic optimization of our results. Future studies could explore the effects of different knee angles to provide a more comprehensive understanding of the relationship between knee joint position and musculotendon calibration. Also, expanding the analysis to include a larger cohort is desirable in future work.

## V. CONCLUSION

We estimated subject-specific muscle-tendon parameters of gastrocnemius, soleus, and tibialis anterior in post-stroke individuals by solving the muscle redundancy problem based on experimental EMG signals and the measured muscle fiber length. The proposed direct collocation method performed well in terms of matching experimental data, as evidenced by the agreement with experimental measurement. Based on our simulation, the post-stroke group had a shorter optimal fiber length, reduced maximum isometric force, and more compliant tendons in both plantarflexors and dorsiflexor. The estimated muscle-tendon parameters are physiologically meaningful and clinically relevant. Our study further demonstrated that incorporating feasible experimental measurements and the direct collocation optimization method is a more comprehensive approach to quantifying the muscle-tendon parameter alterations in individuals with motor impairment, which are otherwise challenging to evaluate solely through experimental in vivo methods.

## ACKNOWLEDGMENT

We would like to thank G. Valentina Pennati, O. Tarassova, and A. Arndt for their support in the experimental setup and data collection.

## REFERENCES

[1] R. L. Sacco, S. E. Kasner, J. P. Broderick, L. R. Caplan, J. Connors, A. Culebras, M. S. Elkind, M. G. George, A. D. Hamdan, R. T. Higashida et al., “An updated definition of stroke for the 21st century: a statement for healthcare professionals from the american heart association/american stroke association,” Stroke, vol. 44, no. 7, pp. 2064–2089, 2013.

[2] V. L. Feigin, M. Brainin, B. Norrving, S. Martins, R. L. Sacco, W. Hacke, M. Fisher, J. Pandian, and P. Lindsay, “World stroke organization (wso): Global stroke fact sheet 2022,” International Journal of Stroke, vol. 17, no. 1, pp. 18–29, 2022.

[3] J. Kim, T. Thayabaranathan, G. A. Donnan, G. Howard, V. J. Howard, P. M. Rothwell, V. Feigin, B. Norrving, M. Owolabi, J. Pandian et al., “Global stroke statistics 2019,” International Journal of Stroke, vol. 15, no. 8, pp. 819–838, 2020.

[4] A. E. Chisholm, S. D. Perry, and W. E. McIlroy, “Correlations between ankle–foot impairments and dropped foot gait deviations among stroke survivors,” Clinical Biomechanics, vol. 28, no. 9-10, pp. 1049–1054, 2013.

[5] B. Chen, B. Zi, Y. Zeng, L. Qin, and W.-H. Liao, “Ankle-foot orthoses for rehabilitation and reducing metabolic cost of walking: Possibilities and challenges,” Mechatronics, vol. 53, pp. 241–250, 2018.

[6] P. Levinger, D. T. Lai, H. B. Menz, A. D. Morrow, J. A. Feller, J. R. Bartlett, N. R. Bergman, and R. Begg, “Swing limb mechanics and minimum toe clearance in people with knee osteoarthritis,” Gait & Posture, vol. 35, no. 2, pp. 277–281, 2012.

[7] S. A. Ross, N. Nigam, and J. M. Wakeling, “A modelling approach for exploring muscle dynamics during cyclic contractions,” PLoS Computational Biology, vol. 14, no. 4, pp. 1–18, 2018.

[8] K. M. Kearney, J. B. Harley, and J. A. Nichols, “Classifying muscle parameters with artificial neural networks and simulated lateral pinch data,” Plos One, vol. 16, no. 9, pp. 1–16, 2021.

[9] C. Korting, M. Schlippe, S. Petersson, G. V. Pennati, O. Tarassova,¨ A. Arndt, T. Finni, K. Zhao, and R. Wang, “In vivo muscle morphology comparison in post-stroke survivors using ultrasonography and diffusion tensor imaging,” Scientific Reports, vol. 9, no. 1, pp. 1–11, 2019.

[10] D. G. Thelen, L. A. Wojcik, A. B. Schultz, J. A. Ashton-Miller, and N. B. Alexander, “Age differences in using a rapid step to regain balance during a forward fall,” The Journals of Gerontology Series A: Biological Sciences and Medical Sciences, vol. 52, no. 1, pp. M8–M13, 1997.

[11] W. F. Brechue and T. Abe, “The role of ffm accumulation and skeletal muscle architecture in powerlifting performance,” European Journal of Applied Physiology, vol. 86, no. 4, pp. 327–336, 2002.

[12] F. Heinen, S. N. Sørensen, M. King, M. Lewis, M. E. Lund, J. Rasmussen, and M. de Zee, “Muscle-tendon unit parameter estimation of a hill-type musculoskeletal model based on experimentally obtained subject-specific torque profiles,” Journal of Biomechanical Engineering, vol. 141, no. 6, p. 061005, 2019.

[13] L. Modenese, E. Ceseracciu, M. Reggiani, and D. G. Lloyd, “Estimation of musculotendon parameters for scaled and subject specific musculoskeletal models using an optimization technique,” Journal of Biomechanics, vol. 49, no. 2, pp. 141–148, 2016.

[14] M. Viceconti, G. Clapworthy, and S. V. S. Jan, “The virtual physiological human—a european initiative for in silico human modelling,” The Journal of Physiological Sciences, vol. 58, pp. 441–446, 2008.

[15] C. Pizzolato, D. G. Lloyd, M. Sartori, E. Ceseracciu, T. F. Besier, B. J. Fregly, and M. Reggiani, “Ceinms: A toolbox to investigate the influence of different neural control solutions on the prediction of muscle excitation and joint moments during dynamic motor tasks,” Journal of Biomechanics, vol. 48, no. 14, pp. 3929–3936, 2015.

[16] M. Romanato, F. Spolaor, C. Beretta, F. Fichera, A. Bertoldo, D. Volpe, and Z. Sawacha, “Quantitative assessment of training effects using eksogt® exoskeleton in parkinson’s disease patients: A randomized single blind clinical trial,” Contemporary Clinical Trials Communications, vol. 28, p. 100926, 2022.

[17] K. Choudhury, “Subject specific hd-emg driven musculoskeletal modeling for stroke subjects,” Master’s thesis, University of Twente, 2019.

[18] J. Peltonen, N. J. Cronin, L. Stenroth, T. Finni, and J. Avela, “Viscoelastic properties of the achilles tendon in vivo,” Springerplus, vol. 2, no. 1, pp. 1–8, 2013.

[19] H. Zhao, Y. Ren, E. J. Roth, R. L. Harvey, and L.-Q. Zhang, “Concurrent deficits of soleus and gastrocnemius muscle fascicles and achilles tendon post stroke,” Journal of Applied Physiology, vol. 118, no. 7, pp. 863– 871, 2015.

[20] C. M. Nelson, V. Marchese, K. Rock, R. M. Henshaw, and O. Addison, “Alterations in muscle architecture: a review of the relevance to individuals after limb salvage surgery for bone sarcoma,” Frontiers in Pediatrics, vol. 8, p. 292, 2020.

[21] J. N. Liang and K.-Y. Ho, “Altered achilles tendon morphology in individuals with chronic post-stroke hemiparesis: a case report,” BMC Medical Imaging, vol. 20, no. 1, pp. 1–5, 2020.

[22] T. Delabastita, M. Afschrift, B. Vanwanseele, and F. De Groote, “Ultrasound-based optimal parameter estimation improves assessment of calf muscle–tendon interaction during walking,” Annals of Biomedical Engineering, vol. 48, no. 2, pp. 722–733, 2020.

[23] Q. Shao, D. N. Bassett, K. Manal, and T. S. Buchanan, “An emg-driven model to estimate muscle forces and joint moments in stroke patients,” Computers in Biology and Medicine, vol. 39, no. 12, pp. 1083–1088, 2009.

[24] N. Lotti, M. Xiloyannis, G. Durandau, E. Galofaro, V. Sanguineti, L. Masia, and M. Sartori, “Adaptive model-based myoelectric control for a soft wearable arm exosuit: A new generation of wearable robot control,” IEEE Robotics & Automation Magazine, vol. 27, no. 1, pp. 43–53, 2020.

[25] G. Durandau, D. Farina, G. As´ın-Prieto, I. Dimbwadyo-Terrer, S. Lerma-Lara, J. L. Pons, J. C. Moreno, and M. Sartori, “Voluntary control of wearable robotic exoskeletons by patients with paresis via neuromechanical modeling,” Journal of Neuroengineering and Rehabilitation, vol. 16, pp. 1–18, 2019.

[26] M. Sartori, D. G. Llyod, and D. Farina, “Neural data-driven musculoskeletal modeling for personalized neurorehabilitation technologies,” IEEE Transactions on Biomedical Engineering, vol. 63, no. 5, pp. 879– 893, 2016.

[27] L. Li, K. Tong, X. Hu, L. Hung, and T. Koo, “Incorporating ultrasoundmeasured musculotendon parameters to subject-specific emg-driven model to simulate voluntary elbow flexion for persons after stroke,” Clinical Biomechanics, vol. 24, no. 1, pp. 101–109, 2009.

[28] R. Wang, S. Yan, M. Schlippe, O. Tarassova, G. V. Pennati, F. Lindberg, C. Korting, A. Destro, L. Yang, B. Shi¨ et al., “Passive mechanical properties of human medial gastrocnemius and soleus musculotendinous unit,” BioMed Research International, vol. 2021, 2021.

[29] H. J. Hermens, B. Freriks, R. Merletti, D. Stegeman, J. Blok, G. Rau, C. Disselhorst-Klug, and G. Hagg, “European recommendations for sur-¨ face electromyography,” Roessingh Research and Development, vol. 8, no. 2, pp. 13–54, 1999.

[30] C. Pizzolato, M. Reggiani, L. Modenese, and D. Lloyd, “Real-time inverse kinematics and inverse dynamics for lower limb applications using opensim,” Computer Methods in Biomechanics and Biomedical Engineering, vol. 20, no. 4, pp. 436–445, 2017.

[31] H. X. Hoang, C. Pizzolato, L. E. Diamond, and D. G. Lloyd, “Subjectspecific calibration of neuromuscular parameters enables neuromusculoskeletal models to estimate physiologically plausible hip joint contact forces in healthy adults,” Journal ofBiomechanics, vol. 80, pp. 111–120, 2018.

[32] D. A. Winter, H. G. Sidwall, and D. A. Hobson, “Measurement and reduction of noise in kinematics of locomotion,” Journal of Biomechanics, vol. 7, no. 2, pp. 157–159, 1974.

[33] E. Kristianslund, T. Krosshaug, and A. J. Van den Bogert, “Effect of low pass filtering on joint moments from inverse dynamics: implications for injury prevention,” Journal of Biomechanics, vol. 45, no. 4, pp. 666–671, 2012.

[34] A. Mantoan, C. Pizzolato, M. Sartori, Z. Sawacha, C. Cobelli, and M. Reggiani, “Motonms: A matlab toolbox to process motion data for neuromusculoskeletal modeling and simulation,” Source Code for Biology and Medicine, vol. 10, no. 1, pp. 1–14, 2015.

[35] D. J. Farris and G. A. Lichtwark, “Ultratrack: Software for semiautomated tracking of muscle fascicles in sequences of b-mode ultrasound images,” Computer Methods and Programs in Biomedicine, vol. 128, pp. 111–118, 2016.

[36] S. L. Delp, F. C. Anderson, A. S. Arnold, P. Loan, A. Habib, C. T. John, E. Guendelman, and D. G. Thelen, “Opensim: open-source software to create and analyze dynamic simulations of movement,” IEEE Transactions on Biomedical Engineering, vol. 54, no. 11, pp. 1940–1950, 2007.

[37] A. Seth, J. L. Hicks, T. K. Uchida, A. Habib, C. L. Dembia, J. J. Dunne, C. F. Ong, M. S. DeMers, A. Rajagopal, M. Millard et al., “Opensim: Simulating musculoskeletal dynamics and neuromuscular control to study human and animal movement,” PLoS Computational Biology, vol. 14, no. 7, pp. 1–20, 2018.

[38] F. De Groote, A. L. Kinney, A. V. Rao, and B. J. Fregly, “Evaluation of direct collocation optimal control problem formulations for solving the muscle redundancy problem,” Annals of Biomedical Engineering, vol. 44, no. 10, pp. 2922–2936, 2016.

[39] J. A. Andersson, J. Gillis, G. Horn, J. B. Rawlings, and M. Diehl, “Casadi: a software framework for nonlinear optimization and optimal control,” Mathematical Programming Computation, vol. 11, pp. 1–36, 2019.

[40] C. G. Canning, L. Ada, and N. J. O’Dwyer, “Abnormal muscle activation characteristics associated with loss of dexterity after stroke,” Journal of the Neurological Sciences, vol. 176, no. 1, pp. 45–56, 2000.

[41] A. Falisse, S. Van Rossom, I. Jonkers, and F. De Groote, “Emg-driven optimal estimation of subject-specific hill model muscle–tendon parameters of the knee joint actuators,” IEEE Transactions on Biomedical Engineering, vol. 64, no. 9, pp. 2253–2262, 2016.

[42] A. Bheemreddy, A. Friederich, L. Lombardo, R. J. Triolo, and M. L. Audu, “Estimating total maximum isometric force output of trunk and hip muscles after spinal cord injury,” Medical & Biological Engineering & Computing, vol. 58, no. 4, pp. 739–751, 2020.

[43] P. S. Lum, C. Patten, D. Kothari, and R. Yap, “Effects of velocity on maximal torque production in poststroke hemiparesis,” Muscle & Nerve: Official Journal ofthe American Association ofElectrodiagnostic Medicine, vol. 30, no. 6, pp. 732–742, 2004.

[44] P. H. McCrea, J. J. Eng, and A. J. Hodgson, “Time and magnitude of torque generation is impaired in both arms following stroke,” Muscle & Nerve: Official Journal ofthe American Association ofElectrodiagnostic Medicine, vol. 28, no. 1, pp. 46–53, 2003.

[45] C. S. Klein, D. Brooks, D. Richardson, W. E. McIlroy, and M. T. Bayley, “Voluntary activation failure contributes more to plantar flexor weakness than antagonist coactivation and muscle atrophy in chronic stroke survivors,” Journal of Applied Physiology, vol. 109, no. 5, pp. 1337–1346, 2010.

[46] A. Bani-Ahmed, “The evidence for prolonged muscle stretching in ankle joint management in upper motor neuron lesions: considerations for rehabilitation–a systematic review,” Topics in Stroke Rehabilitation, vol. 26, no. 2, pp. 153–161, 2019.

[47] F. Gao and L.-Q. Zhang, “Altered contractile properties of the gastrocnemius muscle poststroke,” Journal of Applied Physiology, vol. 105, no. 6, pp. 1802–1808, 2008.

[48] L. Stenroth, J. Peltonen, N. J. Cronin, S. Sipila, and T. Finni, “Age-¨ related differences in achilles tendon properties and triceps surae muscle architecture in vivo,” Journal of Applied Physiology, vol. 113, no. 10, pp. 1537–1544, 2012.

[49] M. K. Ra’ad, L. Stenroth, N. J. Cronin, A. Reito, J. Paloneva, and T. Finni, “Muscle-tendon morphomechanical properties of nonsurgically treated achilles tendon 1-year post-rupture,” Clinical Biomechanics, p. 105568, 2022.

[50] C. P. Dias, B. Freire, N. B. A. Goulart, C. Dias De Castro, F. D. A. Lemos, J. Becker, A. Arndt, and M. A. Vaz, “Impaired mechanical properties of achilles tendon in spastic stroke survivors: an observational study,” Topics in Stroke Rehabilitation, vol. 26, no. 4, pp. 261–266, 2019.

[51] C.-L. Kuo and G.-C. Hu, “Post-stroke spasticity: a review of epidemiology, pathophysiology, and treatments,” International Journal of Gerontology, vol. 12, no. 4, pp. 280–284, 2018.

[52] G. A. Mirka, “The quantification of emg normalization error,” Ergonomics, vol. 34, no. 3, pp. 343–352, 1991.

[53] S. M. Suydam, K. Manal, and T. S. Buchanan, “The advantages of normalizing electromyography to ballistic rather than isometric or isokinetic tasks,” Journal of Applied Biomechanics, vol. 33, no. 3, pp. 189–196, 2017.

[54] A. Chalard, M. Belle, E. Montane, P. Marque, D. Amarantini, and´ D. Gasq, “Impact of the emg normalization method on muscle activation and the antagonist-agonist co-contraction index during active elbow extension: Practical implications for post-stroke subjects,” Journal of Electromyography and Kinesiology, vol. 51, pp. 1–7, 2020.

[55] C. Y. Scovil and J. L. Ronsky, “Sensitivity of a hill-based muscle model to perturbations in model parameters,” Journal of Biomechanics, vol. 39, no. 11, pp. 2055–2063, 2006.

[56] Z. Chen and D. W. Franklin, “Musculotendon parameters in lower limb models: Simplifications, uncertainties, and muscle force estimation sensitivity,” Annals of Biomedical Engineering, vol. 51, no. 6, pp. 1147– 1164, 2023.

[57] C. N. Maganaris, “A predictive model of moment–angle characteristics in human skeletal muscle: application and validation in muscles across

## APPENDIX

A. MVC torque in healthy control and post-stroke group

the ankle joint,” Journal of Theoretical Biology, vol. 230, no. 1, pp. 89–98, 2004.

[58] F. E. Zajac, “Muscle and tendon: properties, models, scaling, and application to biomechanics and motor control.” Critical Reviews in Biomedical Engineering, vol. 17, no. 4, pp. 359–411, 1989.

[59] A. G. Cresswell, W. Loscher, and A. Thorstensson, “Influence of¨ gastrocnemius muscle length on triceps surae torque development and electromyographic activity in man,” Experimental Brain Research, vol. 105, pp. 283–290, 1995.

[60] M. Cibulka, A. Wenthe, Z. Boyle, D. Callier, A. Schwerdt, D. Jarman, and M. J. Strube, “Variation in medial and lateral gastrocnemius muscle activity with foot position,” International Journal of Sports Physical Therapy, vol. 12, no. 2, pp. 233–241, 2017.

[61] S. L. Delp, J. P. Loan, M. G. Hoy, F. E. Zajac, E. L. Topp, and J. M. Rosen, “An interactive graphics-based model of the lower extremity to study orthopaedic surgical procedures,” IEEE Transactions on Biomedical engineering, vol. 37, no. 8, pp. 757–767, 1990.

[62] F. C. Anderson and M. G. Pandy, “A dynamic optimization solution for vertical jumping in three dimensions,” Computer Methods in Biomechanics and Biomedical engineering, vol. 2, no. 3, pp. 201–231, 1999.

[63] G. T. Yamaguchi and F. E. Zajac, “A planar model of the knee joint to characterize the knee extensor mechanism,” Journal of Biomechanics, vol. 22, no. 1, pp. 1–10, 1989.

[64] V. Distefano, “Anatomy and biomechanics of the ankle and foot,” Athletic Training, vol. 16, no. 1, pp. 43–47, 1981.

## TABLE I

The mean and standard deviation (SD) of ankle dorsiflexion (+)/plantarflexion (-) torque (Nm/kg) across subjects during maximum   
voluntary contraction (MVC) at four angle positions (5°dorsiflexion, 0°, 10°,   
and 20°plantarflexion) in both healthy and post-stroke groups in three cases: $P F _ { G A S }$ (Plantarflexion with measured GAS fiber length), $P F _ { S O L }$ (Plantarflexion with measured SOL fiber length), and DF (Dorsiflexion with measured TA fiber length).

<table><tr><td rowspan="2">Torque (SD) Case</td><td colspan="3">Healthy Control</td><td colspan="3">Post-stroke</td></tr><tr><td> $P F _ { G A S }$ </td><td> $P F _ { S O L }$ </td><td> $_ { D F }$ </td><td> $P F _ { G A S }$ </td><td> $P F _ { S O L }$ </td><td>DF</td></tr><tr><td>5°dorsi</td><td>-1.47 (0.37)</td><td>-1.61 (0.51)</td><td>0.48 (0.15)</td><td>-0.27 (0.22)</td><td>-0.29 (0.23)</td><td>0.02 (0.01)</td></tr><tr><td>0°plantar</td><td>-1.48 (0.29)</td><td>-1.58 (0.33)</td><td>0.49 (0.19)</td><td>-0.19 (0.16)</td><td>-0.23 (0.17)</td><td>0.02 (0.02)</td></tr><tr><td>10°plantar</td><td>-1.13 (0.36)</td><td>-1.09 (0.31)</td><td>0.52 (0.23)</td><td>-0.13 (0.11)</td><td>-0.11 (0.12)</td><td>0.03 (0.04)</td></tr><tr><td>20°plantar</td><td>-0.87 (0.31)</td><td>-0.89 (0.31)</td><td>0.52 (0.18)</td><td>-0.10 (0.10)</td><td>-0.06 (0.10)</td><td>0.04 (0.06)</td></tr></table>