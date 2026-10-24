![](images/214b159fc36d280204aeb43ec0fb3bd3f0e9cc95c39705e0721d1eac5a6895f1.jpg)

## OPEN ACCESS

EDITED BY Tianzhe Bao, University of Health and Rehabilitation Sciences, China

REVIEWED BY Hui Zhou, Nanjing University of Science and Technology, China Hu Xiaoling, Hong Kong Polytechnic University, Hong Kong SAR, China

\*CORRESPONDENCE Guozhi Huang, drhuang66@163.com Rong Song, songrong@mail.sysu.edu.cn

<sup>†</sup>These author have contributed equally to this work

RECEIVED 24 June 2023 ACCEPTED 26 July 2023 PUBLISHED 07 August 2023

## CITATION

Dong Y, Wang K, He R, Zheng K, Wang X, Huang G and Song R (2023), Hybrid and adaptive control of functional electrical stimulation to correct hemiplegic gait fo patients after stroke. Front Bioeng Biotechnol 11:1246014. doi: 10.3389/fbioe.2023.1246014

## CO G

© 2023 Dong, Wang, He, Zheng, Wang, Huang and Song. This is an open-access article distributed under the terms of the Creative Commons Attribution License (CC BY). The use, distribution or reproduction in other forums is permitted, provided the original author(s) and the copyright owner(s) are credited and that the original publication in this journal is cited, in accordance with accepted academic practice. No use, distribution or reproduction is permitted which does not comply with these terms

# Hybrid and adaptive control of functional electrical stimulation to correct hemiplegic gait for patients after stroke

Yiqun Dong<sup>1,2†</sup>, Kangling Wang<sup>3,4†</sup>, Ruxin He<sup>1,2</sup>, Kai Zheng<sup>1,2</sup>, Xiaohong Wang<sup>3,4</sup>, Guozhi Huang<sup>3,4</sup>\* and Rong Song<sup>1,2,5</sup>\*

<sup>1</sup>School of Biomedical Engineering, Shenzhen Campus of Sun Yat-sen University, Shenzhen, China, <sup>2</sup>The Key Laboratory of Sensing Technology and Biomedical Instrument of Guangdong Province, School of Biomedical Engineering Sun Yat-sen University Guangzhou China ³Department of Rehabilitatior Medicine, Zhujiang Hospital, Southern Medical University, Guangzhou, Guangdong, China, <sup>4</sup>School of Rehabilitation Medicine, Southern Medical University, Guangzhou, China, <sup>5</sup>Shenzhen Research Institute o Sun Yat-sen University, Shenzhen, China

Introduction: Gait, as a fundamental human movement, necessitates the coordination of muscles across swing and stance phases. Functional electrical stimulation (FES) of the tibialis anterior (TA) has been widely applied to foot drop correction for patients with post-stroke during the swing phase. Although the gastrocnemius (GAS) during the stance phase is also affected, the Functional electrical stimulation of the gastrocnemius received less attention.

Methods: To address this limitation, a timing- and intensity-adaptive Functional electrical stimulation control strategy was developed for both the TA and GAS. Each channel incorporates a speed-adaptive (SA) module to control stimulation timing and an iterative learning control (ILC) module to regulate the stimulation intensity. These modules rely on real-time kinematic or kinetic data during the swing or stance phase, respectively. The orthotic effects of the system were evaluated on eight patients with post-stroke foot drop. Gait kinematics and kinetics were assessed under three conditions: no stimulation (NS), Functional electrical stimulation to the ankle dorsiflexor tibialis anterior (SA-ILC DS) and FES to the tibialis anterior and the ankle plantarflexor gastrocnemius (SA-ILC DPS).

Results: The ankle plantarflexion angle, the knee flexion angle, and the anterior ground reaction force (AGRF) in the SA-ILC DPS condition were significantly larger than those in the NS and SA-ILC DS conditions (p < 0.05). The maximum dorsiflexion angle during the swing phase in the SA-ILC DPS condition was similar to that in the SA-ILC DS condition, with both being significantly larger than the angle observed in the NS condition (p < 0.05). Furthermore, the angle error and force error relative to the set targets were minimized in the SA-ILC DPS condition.

Discussion: The observed improvements can be ascribed to the appropriate stimulation timing and intensity provided by the SA-ILC DPS strategy. This study demonstrates that the hybrid and adaptive control strategy of functional electrical stimulation system offers a significant orthotic effect, and has considerable potential for future clinical application.

functional electrical stimulation, hemiplegic gait, stroke, iterative learning control, hybrid

## KEYWORDS

## 1 Introduction

Stroke ranks as the second leading cause of death globally, and one of the primary causes of disability, affecting 15 million people annually. From 1990 to 2019, stroke incidence increased by 70%, and the burden of stroke grew substantially (Melo et al., 2015). Stroke is a neurological disorder that leads to movement disorders, such as hemiplegic gait (Lyons et al., 2002; Johnson et al., 2016). Hemiplegic gait typically involves two distinct forms (Lee et al., 2014). On one hand, paralysis or significant weakness of ankle dorsiflexion muscle results in inadequate dorsiflexion, preventing the ability to lift the toes off the ground during the swing phase of gait (Prenton et al., 2016). On the other hand, weakness or spasticity of the plantarflexors hinders patients’ inability to both support their own weight and provide adequate forward propulsive force during the stance phase (Neptune et al., 2001). Consequently, identifying appropriate intervention strategies to address foot drop symptom and enhance forward propulsion is vital.

Functional electrical stimulation (FES) is the most common intervention for correcting foot drop (Gil-Castillo et al., 2020). A typical FES system integrates a stimulation unit, a network of sensors, and a controller that adjusts the output current based on information from the sensing network to stimulate muscles, helping patients regain muscle control and reshape the nervous system (Melo et al., 2015). FES was first applied to foot drop correction by Liberson et al. (Liberson et al., 1961), who applied electrical stimulation to the tibialis anterior (TA) muscle during the heel-off and heel-strike moment. Since then, numerous studies have investigated the control and effects of FES. However, most studies of FES have focused on single-channel stimulation of the TA using open-loop controllers (Cameron, 2010; Stein et al., 2010; Gil-Castillo et al., 2020). The stimulation parameters of these studies are generally preset and fixed, lacking feedback from gait information (Lyons et al., 2002; Kesar et al., 2009). This traditional approach results in insufficient robustness to nonlinear, time-varying, and coupled responses of the system to the stimulated muscles. It also causes the rapid muscle fatigue of patients (Lynch and Popovic, 2008), which affects the ankle plantarflexion angle (Lee et al., 2014; Spaich et al., 2014). However, closed-loop control, another FES control method, can adapt stimulation parameters in real time to track predetermined targets (Nekoukar and Erfanian, 2012; Melo et al., 2015), adjusting the patients’ gait to emulate that of healthy individuals (van der Linden and Mercer, 2017) and perform natural physiological movements (Valtin et al., 2014). Closed-loop control strategies include finite state control (Hausdorff and Ring, 2008), artificial neural network (Riess and Abbas, 2001), proportional-integralderivative control (Rouhani et al., 2017), fuzzy network (Jonic et al., 1999), and iterative learning control (ILC) (Seel et al., 2014; Valtin et al., 2014; Seel et al., 2015; Seel et al., 2016b). In particular, ILC algorithms have been found to address the convergence limitation, providing stable tracking performance as well as updating parameters to compensate for both internal and external disturbances (Arimoto et al., 1984; Freeman et al., 2012; Müller et al., 2017).

Furthermore, while traditional single-channel FES delivered to the TA increased ankle dorsiflexion angles during the swing phase, other gait deficits remained unaddressed. For instance, these deficits include decreased ankle plantarflexion, reduced swing-phase knee flexion, and diminished stance-phase propulsive force (Kesar et al., 2010). These deficits are mainly caused by the impaired ankle plantarflexor muscle, the gastrocnemius (GAS) (Chen et al., 2018a; Jiang et al., 2020). Therefore, applying hybrid control strategy of FES to mimic normal muscle activity in both TA and GAS should be helpful to the patients’ muscle acting in concert to improve gait symmetry and stability (Yang et al., 2012). Indeed, recent advancements in hybrid control strategy of FES have demonstrated promising orthotic effects (van Swigchem et al., 2010; Khamis et al., 2015; van Bloemendaal et al., 2016) and positive contributions to combating muscle fatigue (Nekoukar and Erfanian, 2012; Melo et al., 2015). However, significant challenges remain due to its complex combination with closed loop control (Lynch and Popovic, 2008; Gil-Castillo et al., 2020). The dual-channel FES proposed by Kesar et al. (Kesar et al., 2009) provided patients with greater ankle plantarflexion angle, knee flexion angle and anterior ground reaction force (AGRF). Nevertheless, the timing of the stimulation of the two muscles obtained through a footswitch open-loop controller, may be unsuitable. In comparison to the single-channel case, it resulted decreased ankle dorsiflexion angles. In addition, the dual-channel FES designed by Lee et al. (Lee et al., 2014) improved both knee flexion angle and ankle dorsiflexion angle during the swing phase, but not ankle plantarflexion angle and AGRF, which may be caused by a simplified single switch system and the fixed stimulation intensity. In our previous studies, Chen and Jiang et al. (Chen et al., 2018a; Jiang et al., 2020) established a linear model based on walking speed to modulate stimulation timing, and used an ILC algorithm to adjust the stimulation intensity, improving the walking ability of patients with post-stroke foot drop. These previous studies have proven the benefit of integrating a speed-adaptive model and ILC algorithm in a single-channel FES control architecture to the TA (Chen et al., 2018a; Jiang et al., 2020). Therefore, a hybrid and adaptive control strategy that simultaneously regulates the stimulation timing and intensity is essential to enhance the orthotic effect of FES-assisted gait. Currently, such a strategy has not yet been realized.

The objective of this study is to establish a hybrid and adaptive control strategy of FES system, aiming at correcting hemiplegic gait physiologically and effectively, addressing foot drop and inadequate forward propulsion. To assess the orthotic effect of this system, the kinematic and kinetic data of patients after stroke were compared under conditions of no FES, single-channel FES, and the FES with hybrid and adaptive control strategy.

## 2 Methods

## 2.1 Subjects

Eight post-stroke subjects (seven males and one female, Table1) were asked to enroll in the study. Before the experiment, specialists conducted the Fugl-Meyer motor assessment of lower extremity (FMA-L). The inclusion criteria stipulated that the subjects must have experienced a singular stroke at least 6 months prior to the study, with the FMA-L score >20; possess the capability to walk independently for a minimum of 2 min continuously on a treadmill;

TABLE 1 Demographic and clinical information about subjects with stroke.

<table><tr><td>NO.</td><td>Sex</td><td>Age</td><td>Months after stroke</td><td>Lesion side</td><td>FMA-L</td><td>Comfortable speed (km/h)</td></tr><tr><td>1</td><td>M</td><td>62</td><td>23</td><td>R</td><td>24/34</td><td>2.0</td></tr><tr><td>2</td><td>M</td><td>53</td><td>15</td><td>R</td><td>27/34</td><td>1.0</td></tr><tr><td>3</td><td>M</td><td>66</td><td>7</td><td>L</td><td>24/34</td><td>0.8</td></tr><tr><td>4</td><td>M</td><td>67</td><td>32</td><td>L</td><td>25/34</td><td>1.3</td></tr><tr><td>5</td><td>M</td><td>58</td><td>15</td><td>R</td><td>27/34</td><td>2.4</td></tr><tr><td>6</td><td>M</td><td>24</td><td>17</td><td>L</td><td>24/34</td><td>0.8</td></tr><tr><td>7</td><td>M</td><td>59</td><td>19</td><td>L</td><td>24/34</td><td>0.9</td></tr><tr><td>8</td><td>F</td><td>22</td><td>6</td><td>R</td><td>26/34</td><td>0.8</td></tr></table>

Abbreviations: M, male, F, female, L, left, R:right, FMA -L, Fugl-meyer motor assessment for lower limb.

and have sufficient passive ankle dorsiflexion range to achieve a neutral ankle angle (0<sup>°</sup>) or at least 5<sup>°</sup> of plantarflexion with the knee bent at 90<sup>°</sup>(Kesar et al., 2009). Before taking part in the experiment, all participants signed written informed consent, which was approved by the ethics committee of Zhujiang Hospital, Southern Medical University.

## 2.2 Experimental protocol

Prior to the start of the experiment, participants were requested to walk on a treadmill while ensuring safety by holding the handrail (Kesar et al., 2010). The purpose was to determine their most comfortable walking speed. To eliminate the impact of treadmill acceleration and deceleration on walking, each subject took at least 20 steps at their preferred speed (Gabel and Brand, 1994). Ankle and knee angles were measured in the sagittal plane when the subject was standing, and defined as the neutral angles, respectively. The neutral ankle angle was set at 0<sup>°</sup> (Wu et al., 2005). Positive ankle angle values indicated dorsiflexion, while negative values indicated plantarflexion.

The maximum stimulation intensity of the TA was ascertained either by the subject’s maximum tolerance or when the paralyzed ankle achieved $4 . 9 ^ { \circ }$ dorsiflexion while sitting, whichever occurred early, a customary approach in many previous studies (Chen et al., 2009; Kesar et al., 2010; Seel et al., 2016a). During the experiment, the actual stimulation intensity did not surpass the maximum threshold. The minimum stimulation intensity of the TA was set when the subject felt a slight stimulus, with the actual stimulation intensity being stronger than the minimum. For the GAS, the maximum stimulation intensity was determined when the subject reached maximum tolerance or when the subject stood in a position resembling terminal double support during gait, aiding in raising the paralyzed heel off the groud, whichever happened first (Kesar et al., 2009). The minimum stimulation intensity of the GAS was set at a level where the subject’s gastrocnemius muscle felt a slight stimulus. The actual stimulation intensity of the GAS during the experiment ranged between the maximum and minimum values.

During the experiment, subjects walked under the three distinct conditions: walking with no stimulation (NS), walking with single-channel FES delivered to the TA (SA-ILC DS), and walking with FES delivered to both the TA and the GAS (SA-ILC DPS). Subjects with post-stroke performed three trials for each condition in a random order, totaling nine trials per subject. During each trial, participants were instructed to walk at their most comfortable speed for a minimum of 2 min, with a 2-min rest interval between successive trials to avert muscle fatigue. The stimulation profile used in all conditions was biphasic waveform. The stimulation pulse width was set at 390 µs, and the frequency was 40 Hz.

## 2.3 System structure

Figure 1 shows the structure of the FES system. The FES system included a treadmill (Y-P260EA, Yusheng, China) with four threedimensional force sensors (CL-TR2, Obatel Automation Equipment, China) mounted underneath, a motion capture system (OptiTrack, Natural Point, United States of America), an A-D converter with 32- channel (USB-6343, National Instruments, Texas, United States of America), a footswitch (B-201, Tekscan, United States of America) and a functional electrical stimulator (P2-9632, Fisco, China). The three-dimensional force sensor system detected the kinetic signal of the subject walking on the treadmill at a sampling rate of 1000 Hz. The calibration method was based on the study by Belli et al. (Belli et al., 2001): horizontal forces ranging from 0 to 20 kg were applied on the treadmill using standard test weights. A linear relationship between the standard force and the actual measured force was observed. Linear fitting was performed, achieving a fitting accuracy greater than 0.99. The footswitch, which was positioned on the subject’s hind foot, detected heel-strike and heel-offevents. The motion capture system acquired and stored kinematic data using four infrared cameras, capturing information at a frequency of 100 Hz. To facilitate this, the subject affixed five 12-mm spherical reflective markers to the paralyzed lower limb. The five markers used to measure knee and ankle angles were placed in a bottom-to-top sequence, following specific anatomical reference locations: the space between the second and third metatarsal bones, the lateral malleolus, the midshank, the lateral knee joint, and the mid-thigh (Kadaba et al., 1990). The aforementioned signals were transmitted to the PC via the A-D converter.

![](images/0b5a1feb47f62c1f2b187359565fb65ff4dfa9c451847eba74aa057a7856780f.jpg)  
FIGURE 1  
The structure of the FES system.

![](images/1d3bf563903fc400524dd058a271ee9eb483580e7f4ef28066289327f7baa8c8.jpg)  
FIGURE 2 The hybrid and adaptive control strategy of the FES system.

## 2.4 Control strategy

The hybrid timing- and intensity-adaptive control strategy of FES system was applied to facilitate proper ankle movement during walking. The control strategy comprised two independent components: the stimulation timing was regulated by linear models, and the stimulation intensity was controlled by iterative learning controllers. The hybrid and adaptive control strategy of FES system is shown in Figure 2.

Linear models correlating walking speed with muscle activation phase parameters were established (Murley et al., 2014; Ma et al., 2015) due to the different activation of muscles at different speeds (Kadaba et al., 1990). The walking speed was calculated based on averaging the horizontal displacement change ofthe toe marker over the last five consecutive steps during gait (Chen et al., 2013). It should be noted that the study did not take into account the relative motion between the treadmill and the subjects’ center of mass. The activation timing parameters were composed of delay time $y _ { 1 } \left( i \right) ,$ $y _ { 2 } \left( i \right)$ and duration time $y _ { 3 } \left( i \right)$ . The calculation formulas were as follows:

$$
\boldsymbol {y} _ {k} (\boldsymbol {i}) = \boldsymbol {\alpha} _ {k} \boldsymbol {v} _ {i} + \boldsymbol {\beta} _ {k}\tag{1}
$$

where k is the order of the outputs of the linear models, the coefficients $\alpha _ { 1 } = - 2 8 6 . 8$ , $\beta _ { 1 } = 5 4 1 . 6 ,$ $\alpha _ { 2 } = - 1 1 1 . 7 ,$ $\beta _ { 2 } = 4 1 6 . 9 ,$ $\alpha _ { 3 } = - 2 1 3 . 2 $ , and $\beta _ { 3 } = 8 7 7 . 7 $ , When the footswitch discovered the heel-strike event in the step i th, the stimulation for the GAS was initiated after the linear speed-adaptive time interval $y _ { 1 } \left( i \right)$ , and the stimulation ended when the motion capture system detected that the toe-off event (Chen et al., 2018a; Jiang et al., 2020). The stimulation for the TA was triggered after the speed-adaptive time interval $y _ { 2 } \left( i \right)$ when the footswitch found the heel-off event in the step i th, and the stimulation terminated after another speed-adaptive duration time $y _ { 3 } \left( i \right)$ (Chen et al., 2018a; Jiang et al., 2020). The unit for $\nu _ { i }$ was in meters per second, and the units for the parameters $y _ { 1 } \left( i \right) , y _ { 2 } \left( i \right)$ , and $y _ { 3 } \left( i \right)$ were seconds.

The rules for the iterative learning controllers to regulate the stimulation intensity for the GAS (when $k = 1 )$ and the TA (when $k = 2 )$ were as follows:

$$
\boldsymbol {U} _ {k} (\boldsymbol {i}) = \boldsymbol {T} _ {k} \times \left(\boldsymbol {U} _ {k} (\boldsymbol {i} - \boldsymbol {1}) + \boldsymbol {L} _ {k} \times \boldsymbol {e} _ {k}\right)\tag{2}
$$

$$
\boldsymbol {T} _ {k} = \left\{ \begin{array}{l l} \underline {{I}} _ {k}, I <   \underline {{I}} _ {k}; \\ \overline {{I}}, \overline {{I}} _ {k} <   I <   \overline {{I}} _ {k}; \\ \overline {{I}} _ {k}, \overline {{I}} > \overline {{I}} _ {k}; \end{array} \right.\tag{3}
$$

The target value of the peak anterior ground reaction force $F _ { r }$ during the stand phase was acquired by the following formula (Ray et al., 2018).

$$
F _ {r} = 0. 1 5 \times v _ {\text { treadmill }} + 0. 0 3 5\tag{4}
$$

where, $e _ { 1 }$ was the error between the actual peak anterior ground reaction force $P A G R F _ { i }$ and the target value $F _ { r }$ during the stance phase. The target swing-phase ankle dorsiflexion angle was defined as $4 . 9 ^ { \circ } ,$ which represents the maximum dorsiflexion angle observed in healthy individuals during walking (Arifin et al., 2003). The parameter $e _ { 2 }$ represented the error between the actual maximum ankle dorsiflexion angle $\theta _ { m a x }$ and the target angle θ during the swing phase of step i th. $U _ { k } ( i )$ represented the controller’s output stimulation intensity to the GAS or TA in the step i th, $U _ { k } \left( i - 1 \right)$ was the controller’s output stimulation intensity to the GAS or TA in the step i 1)th, $L _ { k }$ denoted the learning parameter, the initial intensity of the first step was set to the maximum stimulation intensity $\overline { { I _ { k } } } ,$ and $\underline { { I _ { k } } }$ was the minimum stimulation intensity.

## 2.5 Data analysis

A second-order, low-pass Butterworth filter with a cut-off frequency of 15 Hz was applied to filter the kinematic signals. Additionally, based on a three-segment rigid body model of the lower limbs (Kadaba et al., 1990) and the law of cosines, ankle and knee joint angles were obtained. A sixth-order, low-pass Butterworth filter with a cut-off frequency of 10 Hz was used to filter the kinetic signals. The sum of the forces in the anterior direction measured by the four three-dimensional force sensors represented the AGRF (Schmiedmayer and Kastner, 2000). Moreover, the three-dimensional force data obtained in the treadmill’s no-load operation at the corresponding subject’s comfortable speed was subtracted during the calculation process to eliminate interference of the treadmill’s weight and noise. The joint angles and AGRF were calculated and averaged over continuous steps in each experiment and then normalized to a 100% gait cycle.

The Kolmogorov-Smirnov test was utilized to all data to ensure normal distribution. Subsequently, one-way analysis of the variance with repeated measures was conducted, followed by the Bonferroni post hoc analysis to compare the effects of different experimental conditions (NS, SA-ILC DS, and SA-ILC DPS) on gait parameters. The significance level was set at 0.05, and when $\hbar < 0 . 0 5$ , it indicates a statistically significant difference. The statistical analyses were conducted using SPSS 19 (SPSS, Inc., Chicago, IL, United States of America).

![](images/6245ba493a96fd1480ce89a5b829098351333b0ad2e00b8a719efc4ace26b776.jpg)  
B

![](images/c79eacf79cc1575ed4be4542a24b48f9acb4e113f20eca2f11194f379debb03a.jpg)  
FIGURE 3  
(A) ankle angles (mean ± std) and (B) knee angles (mean ± std) during the gait cycle for eight post-stroke subjects at comfortable speed.

## 3 Result

## 3.1 Gait kinematics

In this experiment, the normalized ankle and knee angles are illustrated in Figure 3. The lines represents the average angles throughout the gait cycle, while the shaded region denotes the standard deviation corresponding to these angles.

Figure 4A shows the average maximum ankle dorsiflexion angle during the swing phase of the eight subjects across the three conditions. In the NS, SA-ILC DS, and SA-ILC DPS conditions, the average maximum dorsiflexion angles were 0.09<sup>°</sup>, 3.34<sup>°</sup>, and 4.24<sup>°</sup>, respectively. In contrast to the NS condition, both the SA-ILC DS and SA-ILC DPS conditions exhibited larger ankle dorsiflexion angles $( p < 0 . 0 5 )$ . Nevertheless, there was no significant difference between the SA-ILC DS and SA-ILC DPS conditions. In the SA-ILC DS and SA-ILC DPS conditions, the maximum ankle dorsiflexion angles were closer to 4.9<sup>°</sup> relative to the NS condition.

FIGURE 5  
![](images/3731e18bca050cfa3519af1947cc0682aed38dde2d54644488f9ce89da6e802d.jpg)

![](images/52e899d67de54317a862097e53ac954fa6ffa3f2feaac00f6aa665738b1d57bf.jpg)  
FIGURE 4

![](images/a07bb01e02ce43e749af57bba390716fba6e50e01753098cd2bf7fac1e502fe0.jpg)  
(A) Maximum ankle dorsiflexion angles during swing phase for 8 subjects after stroke, \*Significant difference (p < 0.05). (B) Peak ankle plantarflexion angles at toe-off event for 8 subjects after stroke, \*Significant difference (p < 0.05). (C) Maximum knee flexion angles during swing phase for 8 subjects after stroke, \*Significant difference $( p < 0 . 0 5 )$

![](images/75051534e7f0d6258ea1d36341b041efefcb482d92a52d2b1ad54bd56742ac0a.jpg)  
The error between the actual maximum ankle dorsiflexion during swing phase and the target angle of each successive step when the eight subjects performed FES-assisted treadmill walking in the SA-ILC DPS conditions.  
A

![](images/a4b1f0bc518086a1fa10e297a4c1afd6d10b29d16b150f7ce0b11f0c13ce0670.jpg)  
B

Figure 4B displays the peak ankle plantarflexion angle of the eight subjects across the three conditions. The peak ankle plantarflexion angles in the NS, SA-ILC DS, and SA-ILC DPS conditions were −0.98°, −0.07°, and −4.07°, respectively. The average peak ankle plantarflexion angle in the SA-ILC DPS condition was significantly larger than those in the NS and SA-ILC DS conditions (p < 0.05). No significant difference was observed between the NS and SA-ILC DS conditions.

Figure 4C illustrates the maximum knee flexion angle of the eight subjects across the three conditions. In the NS, SA-ILC DS, and SA-ILC DPS conditions, the maximum knee flexion angles were 28.33<sup>°</sup>, 29.74<sup>°</sup>, and 32.45<sup>°</sup>, respectively. The maximum knee flexion angle in the SA-ILC DS condition closely resembled that observed in the NS condition. However, the maximum knee flexion angle in the SA-ILC DPS condition was significantly larger than those in the NS and SA-ILC DS conditions $\left( p < 0 . 0 5 \right)$ . There was no significant difference in the maximum knee flexion angle between the SA-ILC DS and NS conditions.

![](images/51c2e54f430a29d0fa0a959b76a4b77538ef3c094343b98bd2b98d320e279e21.jpg)  
(A) The anterior ground reaction force (AGRF) during the gait cycle for eight post-stroke subjects at comfortable speed. (B) The peak AGRF during swing phase for 8 subjects after stroke, \*Significant ( < )

Figure 5 demonstrates the maximum ankle dorsiflexion angle error between the actual angle and the target angle during the swing phase for each consecutive step when the eight subjects walked in the SA-ILC DPS condition. Except for subjects 5 and 6, who exhibited larger maximum ankle dorsiflexion angles during the initial few steps, all other participants had ankle dorsiflexion angles smaller than the target angle. At the beginning, the error was relatively substantial, but it diminished to less than 2<sup>°</sup> after approximately five steps during the gait.

![](images/f2f56cb6602378bf5d928c81ddf20e45c61b7a8ab1f7e9691a72bbef6671034b.jpg)

B  
![](images/e3ed61488b9b53dfa07088ea135d62ff97a9cc753a47fd69c679df3e25ce451c.jpg)  
(A) The error between the ratio of the peak AGRF during the stance phase to the subject’s body weight and the target ratio of each successive step when 8 subjects performed FES-assisted treadmill walking in the SA-ILC DPS condition. (B) The peak AGRF during the stance phase and the target ratio when subject 2 performed FES-assisted treadmill walking in NS, SA-ILC DS, and SA-ILC DPS conditions.

## 3.2 Gait kinetics

In this experiment, the normalized anterior reaction force (AGRF) is displayed in Figure 6A. The lines and shaded area represent the average AGRF and the corresponding standard deviation throughout the gait cycle, respectively. Figure 6B shows the average peak AGRF during the stance phase, normalized to the subject’s body weight (BW), for the three conditions. The ratios were 4.56%, 4.77%, and 8.34% in the NS, SA-ILC DS, and SA-ILC DPS conditions, respectively. The AGRF percentage relative to body weight in the SA-ILC DPS condition was significantly greater than that in the NS and SA-ILC DS conditions (p < 0.05). However, the ratio in the NS condition was similar to that in the SA-ILC DS condition, and there was no significant difference between them.

Figure 7A illustrates the error between the ratio of the peak AGRF during the stance phase to the subject’s body weight and the target ratio for each consecutive step when eight subjects walked in the SA-ILC DPS condition. At the beginning of the trial, all subjects displayed relatively large errors, with subject 6 showing the largest error at the fifth step. However, after approximately six steps, the error between the ratio of the average peak AGRF to body weight and the target ratio decreased to a range of −1.5%–2% for all subjects.

Figure 7B displays the error between the ratio of the peak AGRF to the subject’s body weight during the stance phase and the target ratio when subject 2 walked across the three conditions. In the NS, SA-ILC DS, and SA-ILC DPS conditions, the average standard deviations of the errors were −2.90% ± 0.74%, 2.83% ± 0.87%, and −0.33% ± 1.38%, respectively. The mean errors in the SA-ILC DS and SA-ILC DPS conditions were smaller than those in the NS condition $( p < 0 . 0 5 )$

## 4 Discussion

In this study, a hybrid and adaptive control strategy of FES system was developed for the ankle plantarflexor muscle (GAS) and the dorsiflexor muscle (TA), incorporating a linear model and an ILC controller to respectively modulate stimulation timing and intensity. The orthotic effect of the system for foot drop correction was evaluated by measuring the maximum ankle dorsiflexion angle in the swing phase, peak ankle plantarflexion angle, maximum knee flexion angle in the swing phase and peak AGRF in the stance phase under NS, SA-ILC DS, and SA-ILC DPS conditions.

## 4.1 Timing- and intensity-adaptive control strategy

Triggering stimulation at fixed time intervals, rather than adjusting the stimulation timing according to the walking speed, led to adverse effects such as decreased ankle plantarflexion angle during the push-off phase (Spaich et al., 2014), reduced swing-phase knee flexion angle (Chen et al., 2018b), and diminished forward propulsion (Lee et al., 2014). Furthermore, adaptive adjustment of stimulation intensity was necessary due to the uncertainties in the external environment and disturbances arising from the characteristics of the time-varying muscle (Freeman et al., 2012).

The implemented timing-adaptive linear models of muscle activation/termination time and walking speed were physiologically suitable for FES-assisted gait, which were consistent with the previous research (Chen et al., 2018a; Chen et al., 2018b; Jiang et al., 2020), significantly increasing the swingphase maximum ankle dorsiflexion angle in patients when compared to the NS condition. Unlike Kesar’s study (Kesar et al., 2009), ankle plantarflexion improved without adversely worsening ankle dorsiflexion in the proposed hybrid and adaptive control strategy, which could be ascribed to optimal stimulation timing. The stimulation timing was adaptively adjusted based on velocity, rather than fixed-phase triggering, which proved to be a more suitable timing for stimulation (Jiang and Song, 2018).

Utilizing the ILC algorithms in the FES system enabled the provision of well-suited stimulation intensity, facilitated swift convergence towards the desired objectives (Seel et al., 2016a) and prevented muscle fatigue. On one hand, the SA-ILC algorithm achieved continuous adaptive modulation of the stimulation intensity for the GAS, allowing patients to reach the most stable AGRF after six steps in gait, with the force error converging to a range of −1.5%–2% BW in subsequent steps. In contrast, the errors in NS and SA-ILC DS conditions were sizable and non-converging. Additionally, this strategy accounted for the relationship between AGRF and walking speed (Ray et al., 2018). By setting different target AGRF values based on the most comfortable walking speed of individual subjects, it achieved personalized adaptive control. On the other hand, the SA-ILC algorithm adaptively modulated the stimulation intensity of the TA, causing the angle error to converge to a range of −2° \~ 2° after five steps of gait. Conversely, the error in the NS condition was significant and non-converging, consistent with previous results (Jiang et al., 2020). The iterative learning controllers were capable of rapidly converging the error to a minimal range and enhancing the system’s robustness to ensure responsiveness to disturbances arising from internal and external environments (Freeman et al., 2012). Furthermore, prior studies have shown that the ILC strategy could reduce stimulation intensity (Seel et al., 2015), and the SA-ILC strategy was proven to provide adaptive adjustments in stimulation intensity, effectively preventing muscle fatigue (Jiang et al., 2020). It helped avoid fixed or excessive intensity that led to rapid muscle fatigue (Lynch and Popovic, 2008), ensuring improvements in gait kinematics and kinetics.

## 4.2 Hybrid control strategy of FES system

In previous studies, FES was usually applied only to the ankle dorsiflexor (TA) to correct foot drop during the swing phase (Chen et al., 2018b; Jiang et al., 2020). However, merely stimulating the TA was not enough to improve other abnormalities of gait, such as insufficient forward propulsion during the stance phase, and decreased knee flexion angle during the swing phase (Woolley, 2001). The gastrocnemius muscle, serving as both ankle plantarflexor and knee flexor (Wang and Gutierrez-Farewik, 2011), is instrumental in producing push-off forces at terminal stance and raising knee flexion angle (Nadeau et al., 1999; Chen et al., 2005).

The designed hybrid control strategy of FES system not only improved the angles of ankle and knee joints of patients after stroke, but also increased anterior ground reaction force. Significantly, larger ankle dorsiflexion, ankle plantarflexion and knee flexion angles were observed in the SA-ILC DPS condition compared to the NS condition, signifying an enhancement over the results of the dual-channel FES reported in the Kesar study (Kesar et al., 2009). In comparison to NS and SA-ILC DS conditions, the AGRF significantly increased in SA-ILC DPS conditions by over 3% BW, approaching the peak AGRF necessary for ambulating in healthy individuals. The augmentation in AGRF under the SA-ILC DPS condition was not present in the SA-ILC DS. Furthermore, it outperformed the FES system designed by Lee et al. (Lee et al., 2014), which failed to enlarge AGRF.

This was attributed to the developed hybrid control strategy of FES that incorporated appropriate stimulation to the GAS. Moreover, the controller utilized peak AGRF as a control parameter, distinguishing it from previous control parameters that only focus on kinematic signals or electromyographic signals. The incorporation of this kinetic parameter effectively enhanced forward propulsion. The study outcomes demonstrated the SA-ILC hybrid system’s ability to rectify insufficient forward propulsion for the paralyzed leg in FESassisted walking, thereby providing greater kinetic energy for the paretic leg at toe-off and ankle plantarflexion, increasing paretic knee flexion and ankle dorsiflexion in the swing phase, and yielding superior orthotic effects.

## 4.3 Limitations and future work

Despite the numerous advantages of the hybrid timing- and intensity-adaptive control strategy of FES system, the study presented some limitations. Firstly, further research will examine an expanded pool of stroke subjects in longitudinal clinical trials to validate the long-term rehabilitation effects of rehabilitation training. Additionally, only three joint angles and AGRF were utilized to evaluate FES-assisted walking performance. More comprehensive evaluation metrics will be encompassed to optimize control and improve other gait abnormalities, such as center of pressure, gait symmetry, and so on.

## 5 Conclusion

In this study, a hybrid and adaptive control strategy of FES system targeting the tibialis anterior and the gastrocnemius was devised. The system integrated linear models of muscle activation time and walking speed with iterative learning controllers algorithm to respectively modulate stimulation timing and intensity. The proposed SA-ILC hybrid control strategy of FES demonstrated its capacity to enhance maximum ankle dorsiflexion angle, ankle plantarflexion angle and knee flexion angle during the swing phase, as well as anterior ground reaction force. These improvements yielded stable orthotic effects for patients afflicted by foot drop, suggesting promising clinical potential for this innovative approach.

## Data availability statement

The raw data supporting the conclusion of this article will be made available by the authors, without undue reservation.

## Ethics statement

The studies involving humans were approved by the ethics committee of Zhujiang Hospital, Southern Medical University. The studies were conducted in accordance with the local legislation and institutional requirements. The participants provided their written informed consent to participate in this study. Written informed consent was obtained from the individual(s) for the publication of any potentially identifiable images or data included in this article.

## Author contributions

YD and RS designed the study. YD conducted the experiments, analyzed the data and wrote the first draft of the manuscript. KW, RH, XW, and GH helped to perform the experiments. RS and KZ revised the manuscript. All authors contributed to the article and approved the submitted version.

## Funding

This research was supported by the National Key Research and Development Program of China (Grant No. 2022YFE0201900), the Shenzhen Science and Technology Research Program (Grant No. SGDX20210823103405040), the Shenzhen Science and Technology Plan Project (Grant No. GJHZ20200731095211034), the Guangdong Science and Technology Plan Project (Grant No.

## References

Arifin, A., Watanabe, T., and Hoshimiya, N. (2003). Fuzzy controller for cycle-to cycle control of SwingPhase of FES-induced hemiplegic gait: A computer simulation in two-joints control. Int. Conf. IEEE Eng. Med. Biol. - Proc. 2, 1519–1522. doi:10.1109/ iembs.2003.1279636

Arimoto, S., Kawamura, S., and Miyazaki, F. (1984). Bettering operation of Robots by learning. J. Robot. Syst. 1, 123–140. doi:10.1002/rob.4620010203

Belli, A., Bui, P., Berger, A., Geyssant, A., and Lacour, J.-R. (2001). A treadmill ergometer for three-dimensional ground reaction forces measurement during walking J. Biomech 34, 105-112 doi:10 1016/S0021-9290(00)00125-1

Cameron, M. H. (2010). The WalkAide functional electrical stimulation system - a novel therapeutic approach for foot drop in central nervous system disorders. Eur. Neurol, Rev, 5. 112, doi:10 17925/USN 2010.06 02.112

Chen, G., Ma, L., Song, R., Li, L., Wang, X., and Tong, K. (2018a). Speed-adaptive control of functional electrical stimulation for dropfoot correction. J. Neuroengineering Rehabil, 15, 98, doi:10.1186/s12984-018-0448-x

Chen, G., Patten, C., Kothari, D. H., and Zajac, F. E. (2005). Gait deviations associated with post-stroke hemiparesis: Improvement during treadmill walking using weight support, speed, support stiffness, and handrail hold. Gait Posture 22, 57–62. doi:10. 1016/j.gaitpost.2004.06.008

Chen, G., Shen, Z., Zhuang, Y., Wang, X., and Song, R. (2018b). Intensity- and duration-adaptive functional electrical stimulation using fuzzy logic control and a linear model for dropfoot correction, Eront., Neurol. 9., 165, doi:10 3389/fneur 2018 00165

Chen, M., Wu, B., Lou, X., Zhao, T., Li, J., Xu, Z., et al. (2013). A self-adaptive foot drop corrector using functional electrical stimulation (FES) modulated by tibialis anterior electromyography (EMG) dataset. Med. Eng. Phys. 35, 195–204. doi:10. 1016/j.medengphy.2012.04.016

Chen, W. L., Chen, S. C., Chen, C. C., Chou, C. H., Shih, Y. Y., Chen, Y. L., et al. (2009). Patient-driven loop control for ambulation function restoration in a non invasive functional electrical stimulation system. Disabil. Rehabil. 32, 65–71. doi:10 3109/09638280903026564

Freeman, C. T., Rogers, E., Hughes, A.-M., Burridge, J. H., and Meadmore, K. L. (2012). Iterative learning control in Health care: Electrical stimulation and roboticassisted upper-limb stroke rehabilitation. IEEE Control Syst. Mag. 32, 18–43. doi:10. 1109/MCS2011.2173261

Gabel, R. H., and Brand, R. A. (1994). The effects of signal conditioning on the statistical analyses of gait EMG. Electroencephalogr. Clin. Neurophysiol. Potentials Sect. 93, 188–201. doi:10.1016/0168-5597(94)90040-X

Gil-Castillo, J., Alnajjar, F., Koutsou, A., Torricelli, D., and Moreno, J. C. (2020). Advances in neuroprosthetic management of foot drop: A review. J. NeuroEngineering Rehabil, 17. 46, doi:10.1186/s12984-020-00668-4

2020B1212060077), the Natural Science Foundation of Guangdong Province (Grant No. 2020A1515010735), the Natural Science Foundation of Guangdong Province (Grant No. 2020A1515011292), and the Foundation of Guangdong Provincial Key Laboratory of Sensor Technology and Biomedical Instrument (Grant No. 2020B1212060077).

## Conflict of interest

The authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.

## Publisher’s note

All claims expressed in this article are solely those of the authors and do not necessarily represent those of their affiliated organizations, or those of the publisher, the editors and the reviewers. Any product that may be evaluated in this article, or claim that may be made by its manufacturer, is not guaranteed or endorsed by the publisher.

Hausdorff, J. M., and Ring, H. (2008). Effects of a new radio frequency - controlled neuroprosthesis on gait symmetry and Rhythmicity in patients with chronic hemiparesis. Am. J. Phys. Med. Rehabil. 87, 4–13. doi:10.1097/PHM.0b013e31815e6680

Jiang, C., and Song, R. (2018). "Dual-channel speed-adaptive control of functional electrical stimulation of tibialis anterior (TA) and gastrocnemius (GAS) for dropfoot correction, IEEE International Conference on Cyborg and Bionic Systems (CBS, November 24-26, 2022, (New York, 624–627. doi:10.1109/ CBS 2018.8612252

Jiang, C., Zheng, M., Li, Y., Wang, X., Li, L., and Song, R. (2020). Iterative adjustment of stimulation timing and intensity during FES-assisted treadmill walking for patients after stroke. IEEE Trans. Neural Syst. Rehabil. Eng. 28, 1292–1298. doi:10.1109/TNSRE. 2020.2986295

Johnson, W., Onuma, O., Owolabi, M., and Sachdev, S. (2016). Stroke: A global response is needed. Bull. World Health Organ. 94, 634. doi:10.2471/BLT.16.181636

Jonic, S., Jankovic, T., Gajic, V., and Popvic, D. (1999). Three machine learning techniques for automatic determination of rules to control locomotion. IEEE Trans. Biomed. Eng. 46, 300–310. doi:10.1109/10.748983

Kadaba, M. P., Ramakrishnan, H. K., and Wootten, M. E. (1990). Measurement of lower extremity kinematics during level walking. J. Orthop. Res. 8, 383–392. doi:10.1002/ jor.1100080310

Kesar, T. M., Perumal, R., Jancosko, A., Reisman, D. S., Rudolph, K. S., Higginson, J. S., et al. (2010). Novel patterns of functional electrical stimulation have an immediate effect on dorsiflexor muscle function during gait for people poststroke. Phys. Ther. 90, 55–66. doi:10.2522/ptj.20090140

Kesar, T. M., Perumal, R., Reisman, D. S., Jancosko, A., Rudolph, K. S., Higginson, J. S., et al. (2009). Functional electrical stimulation ofankle plantarflexor and dorsiflexor muscles effects on poststroke gait. Stroke 40, 3821–3827. doi:10.1161/strokeaha.109. 560375

Khamis, S., Martikaro, R., Wientroub, S., Hemo, Y., and Hayek, S. (2015). A functional electrical stimulation system improves knee control in crouch gait. J. Child. Orthop. 9, 137–143. doi:10.1007/s11832-015-0651-2

Lee, Y.-H., Yong, S. Y., Kim, S. H., Kim, J. H., Shinn, J. M., Kim, Y., et al. (2014). Functional electrical stimulation to ankle dorsiflexor and plantarflexor using single foot switch in patients with hemiplegia from hemorrhagic stroke. Ann. Rehabil. Med. 38, 310–316,doi:10.5535/arm 2014.38.3.310

Liberson, W. T., Holmquest, H. J., Scot, D., and Dow, M. (1961). Functional electrotherapy: Stimulation of the peroneal nerve synchronized with the swing phase of the gait of hemiplegic patients. Arch. Phys. Med. Rehabil. 42, 101–105.

Lynch, C. L., and Popovic, M. R. (2008). Functional electrical stimulation. IEEE Control Svst, Mag, 28. 40–50 doi:10.1109/MCS 2007 914689

Lyons, G. M., Sinkjaer, T., Burridge, J. H., and Wilcox, D. J. (2002). A review of portable FES-based neural orthoses for the correction of drop foot. IEEE Trans. Neural Syst. Rehabil. Eng. 10, 260–279. doi:10.1109/TNSRE.2002.806832

Ma, L., Yang, Y., Chen, N., Song, R., and Li, L. (2015). Effect of different terrains on onset timing, duration and amplitude of tibialis anterior activation. Biomed. Signal Process. Control 19, 115–121. doi:10.1016/j.bspc.2015.03.012

Melo, P. L., Silva, M. T., Martins, J. M., and Newman, D. J. (2015). Technical developments of functional electrical stimulation to correct drop foot: Sensing, actuation and control strategies. Clin. Biomech. 30, 101–113. doi:10.1016/j. clinbiomech.2014.11.007

Müller, P., Balligand, C., Seel, T., and Schauer, T. (2017). Iterative learning control and system identification of the antagonistic knee muscle complex during gait using functional electrical stimulation. IFAC-Pap. 50, 8786–8791. doi:10.1016/j.ifacol.2017 08.1738

Murley, G. S., Menz, H. B., and Landorf, K. B. (2014). Electromyographic patterns of tibialis posterior and related muscles when walking at different speeds. Gait Posture 39, 1080–1085. doi:10.1016/j.gaitpost.2014.01.018

Nadeau, S., Gravel, D., Arsenault, A. B., and Bourbonnais, D. (1999). Plantarflexor weakness as a limiting factor of gait speed in stroke subjects and the compensating rol of hip flexors. Clin. Biomech. 14, 125–135. doi:10.1016/S0268-0033(98)00062-X

Nekoukar, V., and Erfanian, A. (2012). A decentralized modular control framework for robust control of FES-activated walker-assisted paraplegic walking using terminal sliding mode and fuzzy logic control. IEEE Trans, Biomed. Eng, 59, 2818–2827, doi:10. 1109/TBME,2012,2208963

Neptune, R. R., Kautz, S. A., and Zajac, F. E. (2001). Contributions of the individual ankle plantar flexors to support, forward progression and swing initiation during walking. J. Biomech. 34, 1387–1398. doi:10.1016/S0021-9290(01)00105-1

Prenton, S., Hollands, K. L., and Kenney, L. P. J. (2016). Functional electrical stimulation versus ankle foot orthoses for foot-drop: A meta-analysis of orthotic effects. J. Rehabil. Med. 48, 646–656. doi:10.2340/16501977-2136

Ray, N. T., Knarr, B. A., and Higginson, J. S. (2018). Walking speed changes in response to novel user-driven treadmill control. J. Biomech. 78, 143–149. doi:10.1016/j. jbiomech.2018.07.035

Riess, J., and Abbas, J. J. (2001). Adaptive control of cyclic movements as muscles fatigue using functional neuromuscular stimulation. IEEE Trans. Neural Syst. Rehabil. Eng. 9, 326–330. doi:10.1109/7333.948462

Rouhani, H., Same, M., Masani, K., Li, Y. Q., and Popovic, M. R. (2017). PID controller design for FES applied to ankle muscles in neuroprosthesis for standing balance. Front. Neurosci. 11, 347. doi:10.3389/fnins.2017.00347

Schmiedmayer, H.-B., and Kastner, J. (2000). Enhancements in the accuracy of the center of pressure (COP) determined with piezoelectric force plates are dependent on the load distribution, J. Biomech, Eng, 122. 523–527, doi:10.1115/1.1289687

Seel, T., Laidig, D., Valtin, M., Werner, C., Raisch, J., and Schauer, T. (2014). "Feedback control of foot eversion in the adaptive peroneal stimulator," in 22nd Mediterranean Conference on Control and Automation. 16-19 June 2014, China, IEEE, 1482–1487. doi:10.1109/MED.2014.6961585

Seel, T., Valtin, M., Werner, C., and Schauer, T. (2015). Multivariable control of foot motion during gait by peroneal nerve stimulation via two skin ElectrodesppAs part of

the research project APeroStim, this work was supported by the German federal ministry of education and research(FKZ 01EZ1204B). IFAC-Pap 48, 315–320. doi:10.1016/j.ifacol.2015.10.158

Seel, T., Werner, C., Raisch, J., and Schauer, T. (2016a). Iterative learning control of a drop foot neuroprosthesis - generating physiological foot motion in paretic gait by automatic feedback control. Control Eng. Pract. 48, 87–97. doi:10.1016/j.conengprac. 2015.11.007

Seel, T., Werner, C., and Schauer, T. (2016b). The adaptive drop foot stimulator – multivariable learning control of foot pitch and roll motion in paretic gait. Med. Eng. Phys. 38, 1205–1213. doi:10.1016/j.medengphy.2016.06.009

Spaich, E. G., Bøg, M. F., Erkocevic, E., Smidstrup, A., Andersen, O. K., and Nielsen, J. F. (2014). “Gait orthosis lokomat combined with functional electrical stimulation for foot drop correction: A feasibility study,” in Replace, repair, restore, relieve – bridging clinical and engineering solutions in neurorehabilitation, 751. –757. doi:10.1007/978-3- 319-08072-7\_104

Stein, R. B., Everaert, D. G., Thompson, A. K., Chong, S. L., Whittaker, M., et al. (2010). Long-term therapeutic and orthotic effects of a foot drop stimulator on walking performance in progressive and nonprogressive neurological disorders. Neurorehabil. Neural Repair. 24, 152–167. doi:10.1177/1545968309347681

Valtin, M., Seel, T., Raisch, J., and Schauer, T. (2014). Iterative learning control of drop foot stimulation with array electrodes for selective muscle activation. IFAC Proc. 47, 6587–6592. doi:10.3182/20140824-6-ZA-1003.0199

van Bloemendaal, M., Bus, S. A., de Boer, C. E., Nollet, F., Geurts, A. C. H., and Beelen, A. (2016). Gait training assisted by multi-channel functional electrical stimulation early after stroke: Study protocol for a randomized controlled trial. Trials 17, 477. doi:10. 1186/s13063-016-1604-x

van der Linden, M. L., and Mercer, T. H. (2017). “Functional electrical stimulation to treat foot drop as a result of an upper motor neuron lesion,” in Electroceuticals. Editor A. Majid (Cham: Springer), 1–28. doi:10.1007/978-3-319-28612-9\_1

van Swigchem, R., Vloothuis, J., den Boer, J., Weerdesteyn, V., and Geurts, A. C. H. (2010). Is transcutaneous peroneal stimulation beneficial to patients with chronic stroke using an ankle-foot orthosis? A within-subjects study of patients’ satisfaction, walking speed and physical activity level. J. Rehabil. Med. 42, 117–121. doi:10.2340/16501977- 0489

Wang, R., and Gutierrez-Farewik, E. M. (2011). The effect of subtalar inversion/ eversion on the dynamic function of the tibialis anterior, soleus, and gastrocnemius during the stance phase of gait. Gait Posture 34, 29–35. doi:10.1016/j.gaitpost.2011. 03.003

Woolley, S. M. (2001). Characteristics of gait in hemiplegia. Top. Stroke Rehabil. 7, 1–18. doi:10.1310/JB16-V04F-JAL5-H1UV

Wu, G., van der Helm, F. C. T., Veeger, H. E. J., Makhsous, M., Van Roy, P., Anglin, C., et al. (2005). ISB recommendation on definitions of joint coordinate systems of various joints for the reporting of human joint motion - Part II: Shoulder, elbow, wrist and hand. J. Biomech. 38, 981–992. doi:10.1016/j.jbiomech.2004.05.042

Yang, N., Duan, F., Liu, S., Dai, L., Tan, J. T. C., and Xu, B. (2012). "Ankle-angle analysis of drop-foot patients in walking with FES," in 2012 IEEE international conference on robotics and biomimetics (ROBIO), 11-14 Dec. 2012, Germany, (Guangzhou, IEEE, 831–836. doi:10.1109/ROBIO.2012.6491071