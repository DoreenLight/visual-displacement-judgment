<!-- source: https://doi.org/10.5772/intechopen.115174 -->
<!-- title: Neural Networks to Control Voluntary Movements in a Lower Limb Exoskeleton Using Electromyographic Signals -->
<!-- fetched: 2026-09-30T14:14:37Z -->
<!-- via: crossref_search("章隆彬 Longbin Zhang Hunan University 人工智能与机器人学院 rehabilitation robotics exoskeleton neuromusculoskeletal model surface EMG markerless motion capture brain computer interface muscle tendon estimation joint torque prediction gait balance assessment 实验室团队 合作网络") -->

Open Access is an initiative that aims to make scientific research freely available to all. To date our community has made over 100 million downloads. It’s based on principles of collaboration, unobstructed discovery, and, most importantly, scientific progression. As PhD students, we found it difficult to access the research we needed, so we decided to create a new Open Access publisher that levels the playing field for scientists across the world. How? By making research easy to access, and puts the academic needs of the researchers before the business interests of publishers.
[Authors, Editors and Institutions](https://www.intechopen.com/institutions)
We are a community of more than 103,000 authors and editors from 3,291 institutions spanning 160 countries, including Nobel Prize winners and some of the world’s most-cited researchers. Publishing on IntechOpen allows authors to earn citations and find new collaborators, meaning more people see your work not only from your own field of study, but from other related fields too.
[Content Alerts](https://www.intechopen.com/page/content-alerts)
Brief introduction to this section that descibes Open Access especially from an IntechOpen perspective
[How it Works](https://www.intechopen.com/how-open-access-publishing-with-intechopen-works) [Manage preferences](https://www.intechopen.com/chapters/89687)
[Contact](https://www.intechopen.com/page/contact-us)
Want to get in touch? Contact our London head office or [media team here](https://www.intechopen.com/chapters/89687)
[Careers](https://www.intechopen.com/page/careers-at-intechopen)
Our team is growing all the time, so we’re always on the lookout for smart people who want to help us reshape the world of scientific publishing.
[Home](https://www.intechopen.com/) > [Books](https://www.intechopen.com/books) > [New Insights in Brain-Computer Interface Systems](https://www.intechopen.com/books/12824)
![](https://cdnintech.com/web/frontend/www/assets/06.166/journals/OpenAccessLock.svg)Open access peer-reviewed chapter
# Neural Networks to Control Voluntary Movements in a Lower Limb Exoskeleton Using Electromyographic Signals
Written By
Giuseppe Menga, Jie Geng and Massimo Mancin
Reviewed: 11 June 2024 Published: 11 September 2024
DOI: 10.5772/intechopen.115174
[ DOWNLOAD FOR FREE](https://www.intechopen.com/chapters/89687)
![](https://cdnintech.com/web/frontend/www/assets/06.166/series/icons/ic-share.svg) Share
![](https://cdnintech.com/web/frontend/www/assets/06.166/series/icons/ic-copy.svg) Cite
Cite this chapter
There are two ways to cite this chapter:
1. Choose citation style Select style Vancouver  APA  Harvard  IEEE  MLA  Chicago  Copy to clipboard  Get citation 
2. Choose citation style Select format Bibtex  RIS  [ Download citation ](https://www.intechopen.com/chapters/89687)
![](https://cdnintech.com/web/frontend/www/assets/06.166/series/icons/ic-checkmark-circle-checked.svg)
[ IntechOpen New Insights in Brain-Computer Interface Systems Edited by Nasser H. Kashou ](https://www.intechopen.com/books/12824)
From the Edited Volume
## [New Insights in Brain-Computer Interface Systems](https://www.intechopen.com/books/12824)
Edited by Nasser H. Kashou
[Book Details](https://www.intechopen.com/books/12824) [Order Print](https://www.intechopen.com/books/12824)
Chapter metrics overview
71 Chapter Downloads 
View Full Metrics
[ DOWNLOAD FOR FREE](https://www.intechopen.com/chapters/89687)
![](https://cdnintech.com/web/frontend/www/assets/06.166/series/icons/ic-share.svg) Share
![](https://cdnintech.com/web/frontend/www/assets/06.166/series/icons/ic-copy.svg) Cite
Cite this chapter
There are two ways to cite this chapter:
1. Choose citation style Select style Vancouver  APA  Harvard  IEEE  MLA  Chicago  Copy to clipboard  Get citation 
2. Choose citation style Select format Bibtex  RIS  [ Download citation ](https://www.intechopen.com/chapters/89687)
![](https://cdnintech.com/web/frontend/www/assets/06.166/series/icons/ic-checkmark-circle-checked.svg)
Impact of this chapter
![IntechOpen](https://cdnintech.com/web/frontend/www/assets/06.166/svg/logoSmall_red.svg)
IntechOpen Downloads
71
Total Chapter Downloads on intechopen.com
![Citations](https://cdnintech.com/web/frontend/www/assets/06.166/series/icons/quotation_red.svg)
Citations
![Dimensions](https://cdnintech.com/web/frontend/www/assets/06.166/institutions/dimension-trimmed.svg)
Citations
![Crossref](https://cdnintech.com/web/frontend/www/assets/06.166/institutions/crossref.svg)
1
Citations
## Abstract
A critical point in the human–exoskeleton interfaces is the multivariable voluntary control of several joints independently. The lower limb exoskeleton ESROB, which helpes a patient to perform the sit-to-stand postural exercise, has been used for testing a new control based on electromyographic (EMG) signals and artifical neural networks (ANN). The approach is of “ _admittance control_ ”, i.e. the joints of the exoskeleton are controlled in speed, instead of torque as usual, by mixing an automatic postural control loop (especially for the balance) with a voluntary action of the patient through EMG signals, measured on suitable muscles of the legs and of the trunk, processed by ANN. Mixing the automatic postural loop with the voluntary action by the patient helps during the training of ANN to exercise the different degrees of freedom of the exoskeleton and during the control to improve balance. This chapter describes the automatic postual control of ESROB as well as the experiments of training and of multivariable voluntary control by the patient. In particular, exploiting the separation offered by the algorithms, it is shown that the three degrees of freedom of the exoskeleton are controlled independently intermixing, the automatic control loop, through external sensors, and the voluntary control of the patient.
### Keywords
  * bipedal robotics
  * exoskeleton
  * sit-to-stand
  * postural balance
  * electromyographic signals
  * neural networks
  * human-computer interface

## Author Information
Show +
  * #### Giuseppe Menga *
    * DAUIN - Politecnico di Torino, Turin, Italy
  * #### Jie Geng 
    * CRISP, Zhejiang University of Finance and Economics, Hangzhou, China
  * #### Massimo Mancin 
    * Automation, Martoglio S.R.L.,Chiusa di San Michele, Turin, Italy

*Address all correspondence to: menga@polito.it
## 1. Introduction
The multivariable independent control of several degrees of freedom (DOFs) of an exoskeleton by the patient is a crtitical aspect of the human-machine interface, as it is easily deduced from the references contained in the next subsection.
In the present chapter, this problem is approached with a new control based on electromyographic (EMG) signals and artificial neural networks. The control is experimented on the lower limb exoskeleton ESROB for sit-to-stand exercises, realized in a previous research project.
To present the interface between the human and the exoskeleton, two aspects of biped robotics have to be considered: the DOFs of its kinematics (giving the dimensions of the two linked spaces of joints and Cartesian posture) and the balance. From the robotic point of view, the considered exoskeleton is relatively simple as it has only three DOFs, allowing the joint-legged movements of the two ankles, knees, and hips, corresponding to three postural variables in the sagittal plane of the Cartesian space, as show in [Figure 1](https://www.intechopen.com/chapters/89687#F1). That is to say the position of the projection on the ground of the center of gravity (COG) (and hence balance on the x-axis COGx), the height of the pelvis hz, and the angle of the trunk with respect to the vertical θtrunk.
![](https://cdnintech.com/media/chapter/89687/1733863932-357495903/media/F1.png)
#### Figure 1.
ESROB exoskeleton (a) in action, (b) its kinematics.
We presented a detailed analysis of the DOFs of a biped in these conditions in [[1](https://www.intechopen.com/chapters/89687#B1)]. We also described the relationship that exists between COG, given by the angular position of the joints, and the center of pressure underfoot (COP) (also called “ _zero moment point_ ” (ZMP)), given by the dynamics, and their implications on balance control in [[2](https://www.intechopen.com/chapters/89687#B2)].
In the present approach, the joints of the exoskeleton are controlled in speed and position by combining an automatic postural control loop, closed from external sensors, with the voluntary action of the patient through “ _admittance control_ ”, using the surface EMG signals measured on some of his muscles. The combination of the automatic loop and the voluntary action can be configured and from time to time changed, leaving some postural variables to the automatic control and the others to the patient’s voluntary action, which can be achieved through appropriate joints, or by blending the two controls on the same variables.
The “ _admittance control_ ”, i.e. from torque to speed [[3](https://www.intechopen.com/chapters/89687#B3)], for the man–machine interface is unusual, but it is necessary if it is to coexist with the automatic postural loop. It offers some advantages but, also, presents some drawbacks. Certaintly, it improves the automatic postural control on some DOFs, offering the advantage during the training to concentrate on single movements substaining automatically the patient on the others, but it limits the patient’s freedom. So the approach is useful for persons with strongh impaired balance and asks for their adaptation and physiological retraining.
The automatic postural control (COP position and consequent COGx), the height of the pelvis (substituted for simplicity by the angle of the knees), and angle of the trunk are obtained with a feedback from the measurements of the joint angles, from dynamometers under the feet, and by an accelerometer/gyroscope on the trunk [[1](https://www.intechopen.com/chapters/89687#B1)].
For the voluntary action of the patient in “ _admittance control_ ”, the applied torque is represented, approximately, by the EMG signals, and they are processed by an ANN. Two types of ANNs have been tested: NARX and LSTM. The output of the “ _admittance filter_ ” implemented by the ANN controls in velocity the servomotors of the joints. Speed controls, unlike torque controls, offer postural stiffness. The patient is kept in a postural position. By pushing forward or backward on the exoskeleton, through the EMG signals generated from his muscles, he starts the motion which, once he has reached the desired final position, he has to stop, through a physiological feedback (to stop pushing).
The pairs of agonist and antagonist muscles chosen for each joint are: for the ankles the _tibialis anterior_ (dorsiflexion) and the _soleus_ (plantarflexion), for the knee the _rectus femoris_ (knee extension) and _biceps femoris_ (knee flexion), and for the hips the _gluteus maximus_ (hip extension) and still the _rectus femoris_ (hip flexion).
The approach followed in this chapter, using the functions offered by the ESROB control, is to record EMG signals, joint angular positions and velocities, while the patient performs a series of elementary exercises with voluntary motion of only one postural variable through one joint at a time (or at most two), while the other postural variables are controlled automatically by the postural loop. The recordings of all the elementary experiments are merged into a unique file to train a single multi-variable ANN capable to explain not only elementary motions but also complex ones involving multiple joints.
This chapter describes the techniques used, the training of the ANNs, the exercises performed by a healty person on the prototype, and the discussions of the obtained results.
### 1.1 State of the art and comparisons
A very recent survey [[4](https://www.intechopen.com/chapters/89687#B4)] classifies the large body of literature for lower limb exoskeleton analysis and control. In this survey, two classifications are interesting for comparisons with the present approach: the “ _exoskeleton’s signal acquisition_ ” and the “ _exoskeleton’s system modeling and control strategy_ ”.
In the first classification, at the item of “ _prior signals volitional non invasive EMG_ ” papers dealing with torque controls for 2 DOFs or walking speed and slope estimation for 1 DOF [[5](https://www.intechopen.com/chapters/89687#B5), [6](https://www.intechopen.com/chapters/89687#B6), [7](https://www.intechopen.com/chapters/89687#B7)] are referenced. For “ _posterior signals_ ” typically any type of of mechanical sensors can be integrated into the exoskeletons [[8](https://www.intechopen.com/chapters/89687#B8), [9](https://www.intechopen.com/chapters/89687#B9)].
The ESROB control merges “ _prior signals volitional non invasive EMG_ ” with “ _posterior mechanical signals_ ” using joint angle positions and load cells at the insole of the exoskeleton.
In the second classification, most of the papers are listed under the item “ _gait control_ ” that does not apply here. Vice versa, similarities with the present approach exist with the papers at the item “ _mechanical impedance control_ ” [[10](https://www.intechopen.com/chapters/89687#B10), [11](https://www.intechopen.com/chapters/89687#B11), [12](https://www.intechopen.com/chapters/89687#B12), [13](https://www.intechopen.com/chapters/89687#B13)], but admittance control has not been considered. Impedance is the relationship moving from velocity to torque, and torque is estimated and controlled. In this chapter, admittance control is used, i.e. the inverse relationship moving from torque to velocity, and velocity is the objective. In fact, admittance control is critical to guaratee postural balance.
Many papers propose estimating torque from EMG signals, e.g. [[14](https://www.intechopen.com/chapters/89687#B14)]. Here, the EMG signals are used directly, as representative of torque (or the intention of the patient to move), in input of the admittance filter.
A certain number of papers are dealing with analysis and control of joint positions and velocities, using EMG signals and different kinds of ANN. Experimental examples of control hip [[7](https://www.intechopen.com/chapters/89687#B7)] or knee [[15](https://www.intechopen.com/chapters/89687#B15)], alone, were considered. All three joints, hips, knees, and ankles, as in this chapter, have been treated for analysis only in [[16](https://www.intechopen.com/chapters/89687#B16)], but no practical experiments are presented.
A general discussion of admittance control for human–robot interaction, comparing this technique with impedance control, is available in [[17](https://www.intechopen.com/chapters/89687#B17)]. The paper [[18](https://www.intechopen.com/chapters/89687#B18)] approaches the control of a 4 DOFs (hips and knees of the two legs) of the exoskeleton. However, only the EMG signals of the knees are used through a central pattern generator to transform the network into one-dimension joint space admittance control. In fact, the paper claims there is too much complexity in N-dimension admittance controllers as well as incongruous movement of each joint. In the survey [[9](https://www.intechopen.com/chapters/89687#B9)], two papers refer to admittance control [[19](https://www.intechopen.com/chapters/89687#B19), [20](https://www.intechopen.com/chapters/89687#B20)]. Both papers treat only the control of the ankle using EMG signals, the second, in particular, compares and shows the advantages of the EMG signals with respect to the direct measure of joint torque as input. Inside the EUROBENCH project, the authors [[21](https://www.intechopen.com/chapters/89687#B21)] compare in the Exo-H3 exoskeleton an admittance control based on the trajectory error with a control triggered from EMG signals during walking on a treadmill.
To the knowledge of the authors, there have been no previous controls, apart from gait (where EMG signals are mostly used for classification or synchronization) or experiments of the voluntary multi-variable movements of all three joints of a leg (in the sagittal plane hip, knee, and ankle) when wearing a lower limb exoskeleton in actual operative postural positions during training. In fact, controlling all three joints is critical for balance. Another paper of voluntary multi-variable movements controls the torque only of hip and knee and carries out training without exoskeleton and in no operative postural position [[22](https://www.intechopen.com/chapters/89687#B22)].
The experiments were performed here on a exoskeleton for sit-to-stand exercise because it was available, and it has only three DOFs, but, as is explained in the conclusions, it will be not difficult to free the two legs, separate EMG signals of both, and apply this technique to a complete exoskeleton.
The originality of the human–exosketeleton interface proposed in this chapter relies on its integration with an automatic postural control loop and in the linking of each joint with one corresponding DOF in the Cartesian space. Moreover in the integration, the DOFs between the automatic control loop and the voluntary joint actions can be configured in number, relationship, and intensity.
### 1.2 Organization of the chapter
The following Section 2 is technical: subsection 2.1 describes the control of ESROB, and subsection 2.2 presents the ANNs used for the experiments: NARX and LSTM.
The processing of recorded data and the types and sequences of exercises performed for training the ANNs are discussed in the sections 3 and 4.
Preliminarily, before the networks are implemented in the control, the performances of several experiments of training (optimization) only are evaluated. These experiments, in Section 5, are used for verifying the importance of input signals and for testing the hypothesis that networks trained by linking experiments with each single movement explain movements involving multiple joints.
Sections 6, 7, 8 contain the results of training the networks of NARX and LSTM and the results of using those networks for the voluntary control of independent 2 or 3 DOFs.
Finally, as an example of coordinate control between automatic posture and voluntary action, the sit-to-stand exercise is described in Section 9.
The conclusions, with the discussion of the results, are drawn in Section 10.
The MATLAB files of the last recorded datas and the films of the experiments are available to download at the end of the chapter, in the section titled “Additional materials”.
## 2. Technical aspects
### 2.1 The control of ESROB: postural control and patient compliance
The control of ESROB integrates two aspects: balance and hapticity [[1](https://www.intechopen.com/chapters/89687#B1), [3](https://www.intechopen.com/chapters/89687#B3), [23](https://www.intechopen.com/chapters/89687#B23)]. This is achieved with two feedback loops, one inside the other, as shown in [Figure 2](https://www.intechopen.com/chapters/89687#F2).
![](https://cdnintech.com/media/chapter/89687/1733863932-357495903/media/F2.png)
#### Figure 2.
Two control loops offering automatic balance and hapticity to the patient.
With x, ẋ indicate in the Cartesian space the vectors of coordinates or angles and velocities/angular speeds that uniquely define a posture and a motion of the joint patient–exoskeleton system (these have dimensions equal to the DOFs of the exoskeleton). The components of the vector are x=COGxθkneeθtrunkT, the projection of the COG onto the ground, the height of the pelvis (for simplicity θknee), and the angle of the trunk. With θ=θankleθkneeθhipT, θ̇, in joint space, indicate the angles and angular velocities of ankles, knees, and hips. A classical linear relationship exists between the two velocity vectors ẋ=Jθ⋅θ̇, where the square matrix J, a nonlinear function of θ, is called Jacobian. For any consistent postural position, with the exception of a few singular points where an extended Jacobian has to be used, as described in [[24](https://www.intechopen.com/chapters/89687#B24)], J is invertible, so that θ̇=Jθ−1⋅ẋ, through which, from a desired postural motion, the corresponding motion of the angles of the kinematic chain is obtained.
Therefore, looking at [Figure 2](https://www.intechopen.com/chapters/89687#F2), the outer loop, operating in the Cartesian postural space, automatically controls posture (and therefore also balance), and the inner loop, which offers “ _compliance_ ”, acts directly on the joints of the exoskeleton with patient-operated admittance control via surface EMG signals measured on a number of muscles in the legs and the trunk.
The block “ _Preview_ ” contains the desidered trajectory (positions and velocities) of the postural variables (eventually velocity is zero if a variable is maintained constant). The block “Ks” contains the gains and the compensation filters of the postural loop. “Gs” includes the compensation filters of the joint velocity loops.
On the other hand, the two functions, maintaining a posture and offering compliance, clearly compete during a rehabilitation cycle when the patient is impaired. In fact, the exoskeleton must be able to offer stiff postural support in the initial stages and gradually becomes more “ _compliant_ ”, i.e. enslaved to the patient, as he gradually regains his motorial functionalities. At the end of the rehabilitation cycle, the external postural loop simply has to monitor balance and intervene, through the block “ _Inhibitor_ ” (admittance is set to zero), only when the limits are reached.
To achieve this result, the interactions between automatic postural control and voluntary joint control by the patient can be configured at two levels:
  * separating in the postural space two subspaces in which automatic control on the one hand and voluntary control, on the other, act independently, and contemporaneously define the appropiate subspace in the joint space operated by the patient;
  * moreover, when the two controls coexist in the same subspace, merging, through a parameter indicated with β (0≦β≦1), the postural loop and the cooperative action of the patient.

To ensure a posture and balance so that the biped does not fall down, or to perform a postural task, a “ _whole body coordination_ ” (WBC) [[25](https://www.intechopen.com/chapters/89687#B25)] is postulated in biped robotics. In the realm of the WBC, Choi et al. [[26](https://www.intechopen.com/chapters/89687#B26)] introduced the kinematic resolution of the Jacobian of the COG with embedded movement. In this way, all DOFs are controlled.
Here, an original reinterpretation of the WBC is introduced which is called “ _tutoring-cooperation-coordination_ ”: where some postural variables through appropriate joints are under the patient’s control, and the complementary ones are controlled separately by the automatic postural loop.
_Tutoring_ means that the exercise is completely performed by the automatic postural loop, through a program contained in the block _Preview_ of [Figure 2](https://www.intechopen.com/chapters/89687#F2), with a passive patient.
_Coordination_ is when the patient has complete control, through an equal number of joints, of one or more components of the postural task of the exercise, without interfering with the complementary components controlled by the outer loop, and therefore, he must perform a coordinated movement with the postural loop to generate a meaningful exercise.
Between the two extremes, _Cooperation_ means that on some components of the postural task the two actions are contemporaneously present and overlap. The patient has partial control over these components, with intensity modulated by the coefficient β, and must cooperate with the block _Preview_.
To obtain the desired decoupling of postural and joint variables in the interaction, partially modify the block diagram of [Figure 2](https://www.intechopen.com/chapters/89687#F2), replacing the dotted part 1 with 3a of [Figure 3](https://www.intechopen.com/chapters/89687#F3) and detailing the admittance control 2 with the artificial neural network 3b, where, in [Figure 3a](https://www.intechopen.com/chapters/89687#F3), ẋdesired are the desired postural movements, θ̇p is the patient’s contribution, as an output of the admittance control, θ̇ref is the reference, input of the motor speed servo mechanisms, Ru,Nu are diagonal selection matrices with, one in the range and in the null-space, respectively, otherwise zero, of the postural tasks ẋ automatically controlled in the Cartesian space. Rp,Np are analogous matrices referring to the patient contributions θ̇p in the joint space. For an example, if the automatic postural loop maintains the position of the COP (hence controls the balance), and the patient controls the height of the pelvis and the angle of the trunk from the knees and the hips, with the order of variables in the Cartesian space and in the joint space, specified at the beginning of this section, the matrices are:
![](https://cdnintech.com/media/chapter/89687/1733863932-357495903/media/F3.png)
#### Figure 3.
Tutoring-cooperation-coordination - (a) is the modification to the control to achieve indendency of the automatic posture and the patient action, (b) the admittance filter is implemented by a ANN, and (c) the schematical representation of the interaction of the two controls.
Ru=100000000Nu=000010001Rp=000010001,E1
F=Ru⋅J is an algebraic matrix, and 0⩽β⩽1 a coefficient that provides the intensity of the patient’s contribution (0 = no contribution-tutoring, 1 = total contribution-coordination, intermediate values = cooperation). The value of β, obviously, has consequences on the admittance felt by the patient (β=0 completely stiff system, and β=1 system fully “ _compliant_ ”). The proof is in Appendix A.
With an appropriate choice of Ru,Rp and β, some of the components in the postural space are automatically controlled by the outer loop, and the others are controlled independently with coordination or in cooperation, by the patient through his EMG signals. As an extreme case with Nu=Rp=I and β=1, the entire exoskeleton is controlled by the patient. An interesting case is when Ru and Rp are chosen, as in [Eq. (1)](https://www.intechopen.com/chapters/89687#E1), so that the patient controls all the components to perform a postural exercise through all joints but the ankles. While balance, on the other hand, is automatically guaranteed by the postural loop. Actually, this technique will be used at the end of the chapter for the “ _sit-to-stand_ ” exercise, leaving the patient to control the height of the pelvis and the trunk angle, through the knees and hips and the postural loop to control balance (COGx−COP).
The interaction between the two players (the automatic system and the patient) can be summarized in [Figure 3c](https://www.intechopen.com/chapters/89687#F3).
Finally, the patient’s contribution θ̇p is obtained by finding muscle synergies [[27](https://www.intechopen.com/chapters/89687#B27), [28](https://www.intechopen.com/chapters/89687#B28)] from the EMG signals taken from appropriately chosen muscles from the legs and trunk (not applied in these experiments) and training neural networks, to be used for real-time control, as shown in [Figure 3b](https://www.intechopen.com/chapters/89687#F3).
### 2.2 Adopted neural networks
Neural networks implement the multivariale admittance control filter of the patient–exoskeleton interface. The inputs are the EMG signals of the patient’s muscles involved in the movement (indicators of the intention, speed, and intensity of performing joint movements) and possibly the joint angle position values and the outputs the angular velocities used as references to the speed servomotors of the joints of the exoskeleton. The training of these networks is achieved by joining together the data of a series of partial exercises (sub-experiments, at the limit voluntary movements of single joints) performed by the patient, recording EMG signals, angular positions, and velocities of the joints. The speed references, output of the filter, initiate the movement of the exoskeleton joints, and it is then the patient’s task (always through the EMG signals) with a physiological reaction to make them to stop once they have reached the desired position. Finally, as it happens in the training of neural networks, a part of these signals is used for the actual training, a part for the validation, and a third part for the final test (not used in these experiments). This is to avoid over-parameterizing the network or biasing it too much towards the specific data used for training.
The two types of networks tested in the experiments were the NARX and the LSTM. As MATLAB toolkits are used, all the information regarding these networks can be found there.
#### 2.2.1 The NARX networks
NARX are nonlinear autoregressive neural networks with external moving average inputs. As parameters, the number of delays of the samples of the autoregressive part and of the moving average part, and the number of hidden layers of the network are assigned. The number of delay samples in the network allows the dynamics of the system to be taken into account, and the number of hidden layers considers the complexity of the phenomenon. In these networks, it is possible to assign a different number of delays to different groups of inputs (as in [Figure 4](https://www.intechopen.com/chapters/89687#F4)). This is particularly useful in our case to treat EMG signals and joint angle positions differently, as described in the following sections.
![](https://cdnintech.com/media/chapter/89687/1733863932-357495903/media/F4.png)
#### Figure 4.
Layout of the NARX network used in the experiments.
The optimization was performed using the Leveberg–Marquardt algorithm with hyperparameters given in [Table 1](https://www.intechopen.com/chapters/89687#tab1).  
| Minimum gradient  | 1e-07  |  
| --- | --- |  
| Maximum validation checks  | 16  |  
| Mu  | 0.001  |  
| Mu decrease ratio  | 0.1  |  
| Mu increase ratio  | 10  |  
| Maximum mu  | 10,000,000,000  |  
### Table 1.
Hyperparameters for training NARX networks.
#### 2.2.2 The LSTM networks
Long short-term memory (LSTM) is a recurring artificial neural network used in the field of “deep learning”. Unlike “feed-forward” standard neural networks, LSTMs, as well as NARXs, have a feedback connection, and the outputs are brought back to the input, as it occurs in autoregressive filters. The design parameters are the number of inputs, the number of outputs, and the number of hidden layers. The dynamics of the phenomenon in these networks is captured by the hidden layers, in a way not completely transparent, and it is not possible to treat in input EMG signals and joint angle positions, as in NARX networks, differently ([Figure 5](https://www.intechopen.com/chapters/89687#F5)).
![](https://cdnintech.com/media/chapter/89687/1733863932-357495903/media/F5.png)
#### Figure 5.
Example of the layout of a LSTM network.
The optimization was performed using the Adam (adaptive moment estimation) optimizer, including learning rate information, L2 regularization factor, and mini-batch size algorithm with hyperparameters given in [Table 2](https://www.intechopen.com/chapters/89687#tab2).  
| MaxEpochs  | 300  |  
| --- | --- |  
| GradientThreshold  | 1  |  
| InitialLearnRate  | 0.005  |  
| LearnRateSchedule  | “piecewise”  |  
| LearnRateDropPeriod  | 125  |  
| LearnRateDropFactor  | 0.2  |  
| Shuffle  | “every-epoch”  |  
| ValidationFrequency  | 10  |  
### Table 2.
Hyperparameters for training LSTM networks.
## 3. Preliminaries on carrying out the experiments
Each experiment consists in recording the EMG signals, position, and speed of the joint angles in a series of elementary exercises performed by the patient wearing the exoskeleton as described in the following sections; in training a neural network from the data; in implementing it in the control code of the exoskeleton and in testing the control on the same patient who provided the training data. It should not be overlooked that the patient–exoskeleton interface is then personalized to the specific individual and his characteristics.
The degree of success is not only evaluated from the performance obtained in the trainining but especially is subjectively evaluated qualitatively by the patient on the ease of performing both the exercises for which the net has been trained and other more complex exercises. As an example, by introducing in input a number of delays of the joint angle samples helps, in NARX networks, to improve the performance in training, but the patient has more difficulty to voluntary control the movements. This is because the angles in input correlate with the velocities in output, and this reduces the predictive power of EMG signals. For this reason only current angle positions are adopted in these networks. The same phenomenon does not occur for LSMT networks, where the two groups of inputs, EMG signals, and angle positions cannot be treated differently, and this aspect is under further investigation.
Usually recordings have been made in training experiments on moving voluntarily single joints and, even if the patient has not complete control of the exoskeleton, on pairs of joints in order to improve training or simply for “ _validation_ ”.
### 3.1 Processing of EMG signals
The motion was on the sagittal plane, only, with joint legs. For this, two approaches have been tested with similar results: averaging the corresponding EMG signals on the two legs; measuring the signals on just one leg.
The measures, as all the real-time control, were done with a sampling frequency of 1 KHz. The EMG signals were pre-processed by a band pass filter (100–150 Hz), rectified, and further filtered with a low pass filter of 100 rad/s.
However, the recording of pre-processed EMG signals, positions, and angular velocities was performed in no strict real time with an average sampling period of 0.02 sec. The absence of real time needs a resampling at exactly 0.02 sec of the signals before applying them to the network training algorithm. The control of the exoskeleton is obviously in real time at 1 ms. The filters obtained, based on 0.02 sec samples, therefore require a moving average operation on 20 input and output samples.
### 3.2 Normalization of the inputs
All input signals are normalized. Two normalization techniques have been used with very similar results: from each signal its average value is subtracted, and then, the zero mean signal is divided by its standard deviation, or simply, without taking it to zero average value, it is divided by its maximum amplitude (maximum minus minimum values).
### 3.3 Elemental esercies
The functionalities of the exoskeleton control, described in Section 2.1, were used, which allow automatic postural controls and voluntary actions of the patient to be merged together (“ _Coordination_ ”, i.e. β=1). During training, elementary voluntary exercises were performed one joint at a time or, exceptionally, pairs of joints, while the remaining DOFs were maintained automatically to a constant value through the postural loop. Indeed, without an already trained network, it is not possible to move all three joints voluntarily, independently, in a coordinated way.
The following experiments were performed:
  * voluntary movements of the COGx through the ankles by automatically maintaining the height of the pelvis (angle of the knees) and the angle of the trunk;
  * voluntary movements of the knees (height of the pelvis) automatically holding COP (balance) and trunk angle;
  * voluntary movements of the angle of the trunk through the hips, automatically maintaining the COP and the angle of the knees;
  * voluntary movements of COGx and height of the pelvis through ankles and knees with automatic holding the angle of the trunk;
  * voluntary movements of COGx and angle of the trunk through ankles and hips maintaining with automatic control the height of the pelvis;
  * voluntary movements of the height of the pelvis and angle of the trunk through knees and hips with automatic control of the COP.

Remember that the postural position is guaranteed by the integration of the two controls, and some DOFs are rigidly controlled by the postual loop. The patient has to concentrate his attention on the only (or only two) joint/s (in which the admittance is high) and on the voluntary movements that he can perform. Unlike other works on training for voluntary controls using EMG signals, the exercise is performed here in functionally operative postural positions (i.e. under the effect of the gravity) and in the presence (i.e. with the constraint) of the exoskeleton and its control.
Regardless of voluntarily controlled muscles the entire six-elements vector of EMG signals, all angles, and all angular velocities are recorded.
#### 3.3.1 How EMG signals are processed in training
Since the joints are speed-controlled, the patient is constrained to the postural position of the exoskeleton. In training, the postural variables under test are released from the postural loop, and the joints under patient control are independently actuated through the two EMG signals of the corresponding pair of muscles, agonist, and antagonist. Since a trained filter does not exist yet, the control action is obtained by experimentally implementing for each joint a simple admittance filter of the type:
velreft=fAgot⋅kago−fAntat⋅kanta+h⋅K,E2
where the reference velref is applied to the velocity control of the joint, f is a function with unitary gain and a central dead band, Ago and Anta are the two pre-processed EMG signals, agonist and antagonist, related to the considered joint, kago,kanta,K are gains, and h is an offset.
The patient tries to perform the exercise, with maximum naturalness and minimum effort: the dead zone of the function f and h are regulated to allow to remain stationary at rest, the gains kago,kanta to equalize the effort in both directions, and K to minimize it.
In an attempt to move, the patient excites the corresponding muscles by forcing on the exoskeleton link, and with a reduced effort he believe to perform the movement of the joint and the postural coordinate under his control, and in reality the servomotor of the joint has intervened, instead. Simultaneously, the postural loop controls the remaining postural coordinates.
Slow and fast movements are performed, with variable amplitude, interspersed with periods of rest. The EMG signals related to all joints, both voluntarily and automatically moved, are recorded, together with joint velocities and angular positions.
#### 3.3.2 Multiple movements of joints
As already been noted in the ESROB project, the technique described in Section 3.3.1, where the joints are independently controlled by pairs of EMG signals from corresponding antagonistic muscles, the patient allows to move one joint at a time, and very approximately two joints, while when all three joints are controlled at the same time there are incongruous movements. So in the training, in addition to the experiments on single joints, some movements on two joints have been added.
## 4. Performed exercises and trained models
The experiments were made on an healthy person. In fact, one of the authors offered himself. Twenty-one training sessions were carried out between November 2021 and October 2022, with a total of 31 admittance filters being only identifed or tested in the exoskeleton control. The initial experiments were used to choose the acquisition operating modes, the suitable parameters for the filters and to compare the two types of neural networks. The last acquisitions of 27/06/22, 21/07/22, and 02/08/22 generated the filters for the final experiments, illustrated in this chapter.
A typical example of an experiment (21/07/22) consists of repeating three times similar groups of voluntary elemental movements (ankles only, knees only, and hips only) and one repetition of three double movements (ankles–knees, ankles–hips, and knees–hips). Angles, speeds, and pre-processed normalized EMG signals of the whole experiment are presented in [Figure 6](https://www.intechopen.com/chapters/89687#F6), with details of the joint velocities and EMG signals in [Figure 7](https://www.intechopen.com/chapters/89687#F7). These belong to the sub-experiment where the patient is voluntary moving the ankles and knees and the automatic postural control maintains the angle of the trunk constant.
![](https://cdnintech.com/media/chapter/89687/1733863932-357495903/media/F6.png)
#### Figure 6.
A typical example of data acquisition for training, where each sub-experiment is evident—the experiment of 21/07/22. (a) the joint angle positions, (b) the joint angle velocities, and (c) the normalized EMG signals.
![](https://cdnintech.com/media/chapter/89687/1733863932-357495903/media/F7.png)
#### Figure 7.
Details of the recording of the experiment of 21/07/22—voluntary motion of the ankle and knee—(a) joint angle velocities, (b) EMG signals.
The inputs of the network are the six EMG signals and the three joint angular positions, and the outputs are their angular velocities. The input EMG signals (true predictors of the process) and the output angular velocities require a number of delay samples, necessary to capture the dynamics of the phenomenon. The joint angles, on the other hand, only serve to qualify the instantaneous position where the signals were recorded and must not create a correlation with the output angular velocities, so only the current sample and not their delayed samples have to be used. In NARX networks, assigning a number of different delay samples to groups of different inputs is possible. In LSTM networks to indicate the number of delay samples for each input is not possible, as the past is stored, without distinction, for all the inputs in the hidden layers of the network.
The entire registration is divided into two sections: one part that is used for training and the other part for validation to allow the over-parameterization of the network[1](https://www.intechopen.com/chapters/89687#fn1) to be verified. Furthermore, to avoid the network being influenced too much by the training data, it is convenient in the training process to keep the two cost functionals (the mean squared error between the output data and their estimates) of training and validation separate and to stop the optimization when the minimum of the validation functional is reached. Both optimization frameworks of NARX and LSTM incorporate these features. In fact, in all the figures, where the process of optimization is presented, training and validation performances are shown separate, and the process is stopped when the validation performance reaches the minimum.
## 5. Initial results of consistency of hypotheses
Before using the neural networks in the control, some training and validation experiments were carried out with the aim of verifying the initial hypotheses. The two types of networks have different characteristics and have been analyzed separately.
### 5.1 NARX networks
When initial experiments were completed, 5 input and output delays and 10 hidden layers were chosen for the networks used ([Figure 4](https://www.intechopen.com/chapters/89687#F4)), whether or not input joint angles were present. Two aspects were examined: the importance of joint angle signals in input and the possibility of using elementary motion experiments to explain complex motions. The indices used are the quadratic error mean (MSE) of the validation performance index and the percentage to which the speed prediction explains the speed used for training (one minus the square root of the mean squared error over the square root of the mean square of the signal).
#### 5.1.1 The presence of joint angle signals at the input
The experiment from 27/06/2022 was used for this analysis. Data were obtained by merging three repetitions of similar groups of voluntary elemental movements (ankles only, knees only, hips only) and three repetitions of two groups of double movements (ankles–knees, knees–hips). Training was using 75% of all sub-experiments, while the 25% terminal part of each one was used for validation.
Three tests were performed with different settings of the angle positions in input. The results were in [Table 3](https://www.intechopen.com/chapters/89687#tab3).  
| MES of validation  | Explanation  | Samples of joint positions in input  |  
| --- | --- | --- |  
| 3.0  | 97%  | number of delayed samples the same as EMG signals  |  
| 11.8  | 86%  | only present time sample  |  
| 17.0  | 78%  | no joint positions in input  |  
### Table 3.
Performances with different settings of joint angles position in input using NARX networks.
The presence in input of the joint angles, obviously, improves the fitting. However, using past samples helps to correlate inputs angle positions and output angular velocities, and in these conditions, the voluntary control of the exoskeleton by the patient has been found to be more difficult. Therefore, the solution of using the present sample only has been adopted in the following.
#### 5.1.2 Simple and complex movements
The analysis provided here refers to the acquisition experiment of 21/07/22 in [Figure 6](https://www.intechopen.com/chapters/89687#F6). The typical trend of the training performance index in the various optimization phases (epochs) is given in [Figure 8](https://www.intechopen.com/chapters/89687#F8). Even if the performance index of training continues to decrease during the iterations, the process is stopped when the validation performance starts to increase. Afterwards the network explains the noise present in the part of the signal used for training, but not the essential behavior of the phenomenon.
![](https://cdnintech.com/media/chapter/89687/1733863932-357495903/media/F8.png)
#### Figure 8.
Typical behaviors of the MSE for training and validation data in a NARX network—in this example all three speeds are estimated using all sub-experiments, splitting 75%, 25% between training and validation.
The results obtained by performing training on parts or on the entirety of the samples are detailed below.
In the first five experiments training and validation periods for all sub-experiments were 75%, 25%. In the final experiment, training was done on the whole single movements and validation on the whole double movements. Performances of the three networks with data limited to the three sub-experiments with voluntary movement of only one joint, and estimation of the corresponding joint velocity is contained in the [Table 4](https://www.intechopen.com/chapters/89687#tab4).  
| MSE of validation  | Explanation  | Joint velocity  |  
| --- | --- | --- |  
| 0.3  | 98%  | ankle  |  
| 1.8  | 99%  | knee  |  
| 3.6  | 92%  | hip  |  
### Table 4.
Performances estimating each single joint velocity using NARX networks.
Performances of a single network explaining the 3 voluntary movements, with different conditions, are contained in the [Table 5](https://www.intechopen.com/chapters/89687#tab5).  
| MSE of validation  | Explanation  | Condition of the experiment  |  
| --- | --- | --- |  
| 7.3  | 83%  | excluding the sub-experiments with the double movements  |  
| 13.2  | 77%  | including the sub-experiments with the double movements in training and validation, therefore, the double movements are also included in the training  |  
| 31.7  | 70%  | training on the single voluntary movement sub-experiments and validation on the double movement sub-experiments, and therefore, the double movements are only used for validation and not for training  |  
### Table 5.
Performances estimating a unique network for all veloicities with different settings using NARX networks.
As can be seen, performance degrades when moving from single angle voluntary movement to multiple single angle and double angle movements. However, the last result is interesting, because it highlights how training on single voluntary movements is, to a certain extent, able to interpret even complex movements.
#### 5.1.3 Concluding remarks
Based on the experience in testing the networks, as the input presence of the joint angles with their delays reduced the predictive ability of the EMG signals, angles in input without delay samples were adopted. Simple and double movements were used with the periods of 75%, 25% of each sub-experiment, between training and validation.
### 5.2 LSTM networks
LSTM networks cannot handle differently EMG signals and joint angles in input. Therefore, experiments were performed with and without input angles and with different values of the number of hidden layers.
A number of hidden layers equal to or greater than 100 is needed to have an acceptable prediction of the outputs and reasonable bandwidth of the filters, but, for data size issues, with 200 hidden layers the exoskeleton real-time processor is not able to run the control program. Therefore, 100 hidden layers have been chosen.
Apparently, the presence or absence of input angles did not provide appreciable differences in the performance of the filters. In particular, different to NARX networks, the presence of input angles in real-time control applications did not reduce the predictive power of EMG signals. For this reason, the angles at the input were left in the final real-time experiments, with periods of 75% and 25% of each sub-interval between training and validation.
[Figure 9](https://www.intechopen.com/chapters/89687#F9) and the results of [Table 6](https://www.intechopen.com/chapters/89687#tab6) refer to the identification of the three velocities with a unique network, from the recordings of 27/06/22, using all single and double joints voluntary movement sub-experiments, with a validation on the final parts of each one.

<!-- [clean: truncated to max_content_length] -->